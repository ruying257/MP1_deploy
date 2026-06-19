import sys
sys.path.append('mp1')

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
from termcolor import cprint

from mp1.policy.pisb_policy import PISBPolicy
from mp1.common.pytorch_util import dict_apply
from mp1.policy.pisb_policy import stopgrad, adaptive_l2_loss


class PISBDinoPolicy(PISBPolicy):
    """
    PISB + DINOv3
    设计原则：
    1) 不改 physics bridge
    2) 不改 PISB 主 loss
    3) DINOv3 只作为冻结视觉 backbone
    4) 只增强 global_cond
    5) 训练 / 推理严格共用同一套 global_cond 构造逻辑
    """
    def __init__(self, shape_meta, **kwargs):
        super().__init__(shape_meta, **kwargs)

        # -----------------------------
        # DINOv3 配置
        # -----------------------------
        self.dino_repo_dir = kwargs["dino_repo_dir"]
        self.dino_weights = kwargs["dino_weights"]
        self.dino_model_name = kwargs.get("dino_model_name", "dinov3_vits16")
        self.dino_input_size = kwargs.get("dino_input_size", 224)
        self.use_last_image_only = kwargs.get("use_last_image_only", True)
        self.dino_feature_dim = kwargs.get("dino_feature_dim", 384)  # ViT-S/16 通常是 384
        self.dino_film_scale = kwargs.get("dino_film_scale", 0.1)

        # -----------------------------
        # 加载官方 DINOv3 backbone
        # 官方 README 推荐 torch.hub.load(REPO_DIR, name, source='local', weights=...)
        # -----------------------------
        self.dino = torch.hub.load(
            self.dino_repo_dir,
            self.dino_model_name,
            source="local",
            weights=self.dino_weights
        ).to(self.device)

        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad = False

        # DINO 标准归一化
        self.dino_normalize = Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

        # -----------------------------
        # global_cond 维度
        # -----------------------------
        cond_dim = self.obs_feature_dim * self.n_obs_steps \
            if "cross_attention" not in self.condition_type else self.obs_feature_dim
        self.cond_dim = cond_dim

        # -----------------------------
        # DINO -> FiLM(global_cond)
        # 输出 gamma/beta，零初始化，初始严格退化为原始 PISB
        # -----------------------------
        self.dino_adapter = nn.Sequential(
            nn.Linear(self.dino_feature_dim, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, 2 * cond_dim)
        ).to(self.device)

        with torch.no_grad():
            self.dino_adapter[-1].weight.zero_()
            self.dino_adapter[-1].bias.zero_()

        cprint("[PISB-DINOv3] Activated.", "cyan")
        cprint(f"[PISB-DINOv3] Frozen backbone: {self.dino_model_name}", "cyan")
        cprint("[PISB-DINOv3] Train/Test share the same global_cond construction.", "cyan")

    # ============================================================
    # DINO helper
    # ============================================================
    def _prepare_dino_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: (B, C, H, W), 可能是 uint8 [0,255] 或 float
        返回: (B, C, 224, 224), float
        """
        if image.dtype != torch.float32:
            image = image.float()

        if image.max() > 1.0:
            image = image / 255.0

        image = F.interpolate(
            image,
            size=(self.dino_input_size, self.dino_input_size),
            mode="bilinear",
            align_corners=False
        )

        image = self.dino_normalize(image)
        return image

    def _extract_dino_feat(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        从原始 obs_dict 中抽图像，不走 LinearNormalizer。
        因为 DINO 需要标准 RGB 预处理，不应该吃你行为策略的 normalizer 输出。
        """
        assert "image" in obs_dict, \
            "PISBDinoPolicy requires obs['image']. Please add image to shape_meta / dataset."

        images = obs_dict["image"]  # (B, To, C, H, W) or (B, C, H, W)

        if images.dim() == 5:
            if self.use_last_image_only:
                image = images[:, self.n_obs_steps - 1]   # 取最后一帧
            else:
                image = images[:, 0]
        elif images.dim() == 4:
            image = images
        else:
            raise ValueError(f"Unsupported image shape: {images.shape}")

        image = self._prepare_dino_image(image).to(self.device)

        with torch.inference_mode():
            feat = self.dino(image)

        # 官方 hub 模型通常直接给 backbone 输出；
        # 为稳妥起见兼容 tuple / token 序列
        if isinstance(feat, (list, tuple)):
            feat = feat[0]

        # 如果输出是 token 序列，优先取 CLS token
        if feat.ndim == 3:
            feat = feat[:, 0, :]

        return feat  # (B, dino_feature_dim)

    # ============================================================
    # 统一构造 global_cond：训练 / 推理都调用这个
    # ============================================================
    def _build_global_cond(self, nobs: Dict[str, torch.Tensor], raw_obs: Dict[str, torch.Tensor], batch_size: int):
        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)

        if "cross_attention" in self.condition_type:
            global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
        else:
            global_cond = nobs_features.reshape(batch_size, -1)

        # DINOv3 image embedding
        dino_feat = self._extract_dino_feat(raw_obs)   # (B, dino_dim)

        # FiLM 参数
        gamma_beta = self.dino_adapter(dino_feat)      # (B, 2*cond_dim)
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        # 限制调制幅度，避免一开始破坏 PISB
        gamma = self.dino_film_scale * torch.tanh(gamma)
        beta  = self.dino_film_scale * torch.tanh(beta)

        if global_cond.dim() == 2:
            global_cond = global_cond * (1.0 + gamma) + beta
        else:
            global_cond = global_cond * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        return global_cond, dino_feat

    # ============================================================
    # training
    # ============================================================
    def compute_loss(self, batch):
        # nobs 继续给 PISB 主干用
        raw_obs = batch['obs']

        # 只归一化非 image 的部分
        obs_wo_image = {k: v for k, v in raw_obs.items() if k != 'image'}
        nobs = self.normalizer.normalize(obs_wo_image)

        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color and 'point_cloud' in nobs:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        trajectory = nactions
        device = trajectory.device

        if not self.obs_as_global_cond:
            raise NotImplementedError("PISBDinoPolicy is designed for obs_as_global_cond=True")

        # 统一 global_cond
        global_cond, dino_feat = self._build_global_cond(
            nobs=nobs,
            raw_obs=batch['obs'],
            batch_size=batch_size
        )

        # ---------------- 原始 PISB target，不改 ----------------
        eps = 1e-5
        t = torch.rand((batch_size,), device=device).float() * (1 - 2 * eps) + eps
        x1 = trajectory
        x0 = torch.randn_like(x1)
        t_expand = t.view(batch_size, 1, 1)

        xt, mu_t, std_t = self.physics_bridge.sample_path(x0, x1, t_expand)
        u_tgt = self.physics_bridge.compute_target_velocity(x0, x1, t_expand, xt)

        r_zeros = torch.zeros_like(t)
        model_output = self.model(
            sample=xt,
            timestep=t,
            global_cond=global_cond,
            r=r_zeros,
            training=True
        )

        if isinstance(model_output, tuple):
            v_pred, features = model_output
        else:
            v_pred, features = model_output, []

        error = v_pred - stopgrad(u_tgt)
        meanflow_loss = adaptive_l2_loss(error)

        dis_loss = 0.0
        if features is not None and len(features) > 0:
            if isinstance(features, (list, tuple)):
                for feat in features:
                    dis_loss = dis_loss + self.dispersive_loss(feat)
            else:
                dis_loss = self.dispersive_loss(features)

        loss = meanflow_loss + 0.5 * dis_loss
        mse_val = (stopgrad(error) ** 2).mean()

        loss_dict = {
            'loss': loss.item(),
            'bc_loss': loss.item(),
            'mse_val': mse_val.item(),
            'meanflow_loss': meanflow_loss.item(),
            'dis_loss': dis_loss.item() if isinstance(dis_loss, torch.Tensor) else dis_loss,
            'dino_feat_norm': dino_feat.norm(dim=-1).mean().item(),
        }

        gamma_beta = self.dino_adapter(dino_feat)
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        loss_dict.update({
            'dino_gamma_abs': gamma.abs().mean().item(),
            'dino_beta_abs': beta.abs().mean().item(),
        })
        return loss, loss_dict

    # ============================================================
    # inference
    # ============================================================
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raw_obs = obs_dict
        obs_wo_image = {k: v for k, v in raw_obs.items() if k != 'image'}
        nobs = self.normalizer.normalize(obs_wo_image)

        if not self.use_pc_color and 'point_cloud' in nobs:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        if not self.obs_as_global_cond:
            raise NotImplementedError("PISBDinoPolicy is designed for obs_as_global_cond=True")

        global_cond, dino_feat = self._build_global_cond(
            nobs=nobs,
            raw_obs=obs_dict,
            batch_size=B
        )

        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)

        model = self.model
        model.eval()

        x_current = torch.randn(
            size=cond_data.shape,
            dtype=cond_data.dtype,
            device=cond_data.device
        )

        steps = self.num_inference_steps if self.num_inference_steps is not None else 10
        dt = 1.0 / steps
        r_zeros = torch.zeros((B,), device=device)

        with torch.no_grad():
            for i in range(steps):
                t_val = i / steps
                t_tensor = torch.full((B,), t_val, device=device, dtype=dtype)

                model_output = model(
                    sample=x_current,
                    timestep=t_tensor,
                    global_cond=global_cond,
                    r=r_zeros,
                    training=False
                )

                if isinstance(model_output, tuple):
                    v_pred = model_output[0]
                else:
                    v_pred = model_output

                x_current = x_current + v_pred * dt

        naction_pred = x_current[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred,
        }
        return result