import sys
sys.path.append('mp1')

from typing import Dict, Tuple
import torch
import torch.nn as nn
from termcolor import cprint

from mp1.policy.pisb_policy import PISBPolicy, stopgrad, adaptive_l2_loss
from mp1.common.pytorch_util import dict_apply


class PISBMSGPolicy(PISBPolicy):
    """
    PISB + Multi-Scale Geometric Conditioning (MSG)

    核心思想：
    1) 保持 PISB 的 physics bridge / target velocity / Euler sampling 不变
    2) 只增强 point cloud 条件特征
    3) 使用“全局 + 局部核心 + 外围边界”三种几何视角
    4) 通过零初始化 residual adapter 融入 global_cond，初始严格退化为原始 PISB
    """

    def __init__(self, shape_meta, **kwargs):
        super().__init__(shape_meta, **kwargs)

        if not self.obs_as_global_cond:
            raise NotImplementedError("PISBMSGPolicy is designed for obs_as_global_cond=True")

        self.local_k_small = kwargs.get("local_k_small", 128)
        self.local_k_large = kwargs.get("local_k_large", 128)
        self.local_feat_dim = kwargs.get("local_feat_dim", 64)
        self.local_hidden_dim = kwargs.get("local_hidden_dim", 128)
        self.local_residual_scale = kwargs.get("local_residual_scale", 0.10)

        cond_dim = self.obs_feature_dim * self.n_obs_steps \
            if "cross_attention" not in self.condition_type else self.obs_feature_dim

        # 轻量局部点特征编码器：输入相对坐标 (x-c)
        self.local_point_mlp = nn.Sequential(
            nn.Linear(3, self.local_hidden_dim),
            nn.LayerNorm(self.local_hidden_dim),
            nn.Mish(),
            nn.Linear(self.local_hidden_dim, self.local_feat_dim),
            nn.LayerNorm(self.local_feat_dim),
            nn.Mish()
        ).to(self.device)

        # 融合: pooled_global + local_core + local_boundary -> residual cond
        self.local_fusion = nn.Sequential(
            nn.Linear(cond_dim + 2 * self.local_feat_dim, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, cond_dim)
        ).to(self.device)

        # 零初始化，保证初始行为严格等于原始 PISB
        with torch.no_grad():
            self.local_fusion[-1].weight.zero_()
            self.local_fusion[-1].bias.zero_()

        cprint("[PISB-MSG] Activated.", "cyan")
        cprint("[PISB-MSG] Physics bridge unchanged. Only multi-scale point conditioning is enabled.", "cyan")
        cprint("[PISB-MSG] Zero-init residual fusion confirmed.", "cyan")

    # ============================================================
    # helper: gather points by index
    # ============================================================
    def _gather_points(self, points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        points: (B, N, C)
        idx:    (B, K)
        return: (B, K, C)
        """
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, points.shape[-1])
        return torch.gather(points, 1, idx_expanded)

    # ============================================================
    # helper: extract local multi-scale geometric descriptors
    # ============================================================
    def _extract_local_geom_features(self, nobs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入:
            nobs['point_cloud']: (B, To, N, C)
        输出:
            core_feat:     (B, local_feat_dim)
            boundary_feat: (B, local_feat_dim)
        """
        assert 'point_cloud' in nobs, "PISBMSGPolicy requires point_cloud in obs."

        pc = nobs['point_cloud'][:, :self.n_obs_steps, ...]   # (B, To, N, C)
        pc = pc[..., :3]                                      # 只用 xyz
        B, To, N, C = pc.shape

        pc = pc.reshape(B * To, N, C)                         # (B*To, N, 3)

        # 以点云质心为参考，构造“局部核心/外围边界”两种几何视角
        centroid = pc.mean(dim=1, keepdim=True)               # (B*To, 1, 3)
        rel = pc - centroid                                   # (B*To, N, 3)
        dist = torch.norm(rel, dim=-1)                        # (B*To, N)

        k_small = min(self.local_k_small, N)
        k_large = min(self.local_k_large, N)

        # 距离质心最近的点：核心区
        idx_small = torch.topk(dist, k=k_small, dim=1, largest=False).indices
        # 距离质心最远的点：边界区
        idx_large = torch.topk(dist, k=k_large, dim=1, largest=True).indices

        core_points = self._gather_points(rel, idx_small)     # (B*To, Ks, 3)
        boundary_points = self._gather_points(rel, idx_large) # (B*To, Kl, 3)

        core_feat = self.local_point_mlp(core_points).mean(dim=1)         # (B*To, D)
        boundary_feat = self.local_point_mlp(boundary_points).mean(dim=1) # (B*To, D)

        core_feat = core_feat.reshape(B, To, -1).mean(dim=1)              # (B, D)
        boundary_feat = boundary_feat.reshape(B, To, -1).mean(dim=1)      # (B, D)

        return core_feat, boundary_feat

    # ============================================================
    # helper: build global condition for both train / inference
    # ============================================================
    def _build_global_cond(self, nobs: Dict[str, torch.Tensor], batch_size: int):
        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)

        if "cross_attention" in self.condition_type:
            global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)  # (B, To, D)
            pooled_global = global_cond.mean(dim=1)                                 # (B, D)
        else:
            global_cond = nobs_features.reshape(batch_size, -1)                     # (B, D)
            pooled_global = global_cond

        core_feat, boundary_feat = self._extract_local_geom_features(nobs)

        fusion_in = torch.cat([pooled_global, core_feat, boundary_feat], dim=-1)
        cond_residual = self.local_fusion(fusion_in)
        cond_residual = self.local_residual_scale * torch.tanh(cond_residual)

        if global_cond.dim() == 3:
            global_cond = global_cond + cond_residual.unsqueeze(1)
        else:
            global_cond = global_cond + cond_residual

        aux_stats = {
            "msg_core_feat_norm": core_feat.norm(dim=-1).mean().item(),
            "msg_boundary_feat_norm": boundary_feat.norm(dim=-1).mean().item(),
            "msg_residual_norm": cond_residual.norm(dim=-1).mean().item(),
        }
        return global_cond, aux_stats

    # ============================================================
    # training
    # ============================================================
    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        batch_size = nactions.shape[0]
        trajectory = nactions
        device = trajectory.device

        global_cond, aux_stats = self._build_global_cond(nobs, batch_size)

        # 保持原始 PISB 的固定 physics target
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
            **aux_stats
        }
        return loss, loss_dict

    # ============================================================
    # inference
    # ============================================================
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        global_cond, _ = self._build_global_cond(nobs, B)

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