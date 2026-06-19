import sys
sys.path.append('mp1')

from typing import Dict, Tuple
import torch
import torch.nn as nn

from mp1.policy.pisb_policy import PISBPolicy, stopgrad, adaptive_l2_loss
from mp1.common.pytorch_util import dict_apply
from mp1.model.memory.causal_memory import CausalTemporalEncoder


class PISBCausalMemoryPolicy(PISBPolicy):
    """
    PISB + Causal Temporal Memory

    设计目标：
    1) 保持 PISB 的 physics bridge / target velocity / Euler sampling 不变
    2) 不维护跨 batch / 跨 episode 的隐状态，避免 BC 训练时的状态污染
    3) 每个样本内部使用连续 observation chunk（由 n_obs_steps 决定）
    4) 用 causal temporal encoder 在 chunk 内提取“历史 -> 当前”的因果记忆
    5) 训练 / 推理都复用同一个 _build_global_cond()，避免 train-test mismatch
    """

    def __init__(self, shape_meta, **kwargs):
        super().__init__(shape_meta, **kwargs)

        if not self.obs_as_global_cond:
            raise NotImplementedError("PISBCausalMemoryPolicy requires obs_as_global_cond=True")

        self.memory_depth = kwargs.get("memory_depth", 2)
        self.memory_heads = kwargs.get("memory_heads", 4)
        self.memory_mlp_ratio = kwargs.get("memory_mlp_ratio", 4.0)
        self.memory_dropout = kwargs.get("memory_dropout", 0.0)
        self.memory_residual_scale = kwargs.get("memory_residual_scale", 0.10)

        # 获取 obs_encoder 的真实输出维度
        dummy = {}
        for k,v in shape_meta["obs"].items():
            shape = tuple(v["shape"])
            dummy[k] = torch.zeros((1,*shape), device=self.device)

        with torch.no_grad():
            dummy_feat = self.obs_encoder(dummy)

        self.step_feature_dim = dummy_feat.shape[-1]
        self.memory_dim = self.step_feature_dim

        cond_dim = self.obs_feature_dim * self.n_obs_steps \
            if "cross_attention" not in self.condition_type else self.obs_feature_dim

        # 因果时序编码器：输入逐帧 obs feature (B, To, D)，输出同 shape 的 memory-enhanced feature
        # 当前 obs_encoder 的逐帧输出已经与单帧条件维度对齐，不再额外投影
        self.step_proj = nn.Identity()
        self.temporal_memory = CausalTemporalEncoder(
            dim=self.memory_dim,
            depth=self.memory_depth,
            heads=self.memory_heads,
            mlp_ratio=self.memory_mlp_ratio,
            dropout=self.memory_dropout,
            max_seq_len=max(self.n_obs_steps, 32),
        ).to(self.device)

        # 将 memory summary 投影到 global_cond 空间，作为 bounded residual
        self.memory_adapter = nn.Sequential(
            nn.Linear(self.memory_dim, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, self.step_feature_dim)
        ).to(self.device)

        # 零初始化：初始行为严格退化为原始 PISB
        with torch.no_grad():
            self.memory_adapter[-1].weight.zero_()
            self.memory_adapter[-1].bias.zero_()

        print(f"[PISB-CausalMemory] obs_feature_dim = {self.obs_feature_dim}")
        print(f"[PISB-CausalMemory] memory_dim = {self.memory_dim}")

    def _encode_step_features(self, nobs: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
        """
        将每个 observation step 单独编码成逐帧特征:
            raw_step_features shape = (B, To, D_raw)
            projected_step_features shape = (B, To, memory_dim)
        """
        this_nobs = dict_apply(
            nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )

        raw_step_features = self.obs_encoder(this_nobs)                  # (B*To, D_raw)
        raw_step_features = raw_step_features.reshape(batch_size, self.n_obs_steps, -1)

        # 调试时可保留，确认原始逐帧特征维度
        # print("raw_step_features:", raw_step_features.shape)

        return raw_step_features

    def _build_global_cond(self, nobs: Dict[str, torch.Tensor], batch_size: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        统一构造训练 / 推理都使用的 global_cond
        """
        step_features = self._encode_step_features(nobs, batch_size)          # (B, To, D)

        memory_seq = self.temporal_memory(step_features)                      # (B, To, D)
        assert memory_seq.shape[-1] == self.step_feature_dim, f"memory_seq dim {memory_seq.shape[-1]} != step_feature_dim {self.step_feature_dim}"
        memory_state = memory_seq[:, -1]                                      # (B, D), “当前时刻”的因果历史摘要

        # 原始 PISB 的 global_cond
        if "cross_attention" in self.condition_type:
            # cross-attention: global_cond shape = (B, To, D)
            base_global_cond = memory_seq   # 用 memory-enhanced sequence 作为基础条件

            # bounded residual: (B, D)
            memory_delta = self.memory_adapter(memory_state)
            assert memory_delta.shape[-1] == self.step_feature_dim, f"memory_delta dim {memory_delta.shape[-1]} != step_feature_dim {self.step_feature_dim}"
            memory_delta = self.memory_residual_scale * torch.tanh(memory_delta)

            # broadcast 到每个 observation step
            memory_delta = memory_delta.unsqueeze(1).repeat(1, self.n_obs_steps, 1)  # (B, To, D)

            global_cond = base_global_cond + memory_delta

        else:
            # non-cross-attention: 先用 memory_seq 展平
            base_global_cond = memory_seq.reshape(batch_size, -1)  # (B, To * D)

            # bounded residual 先在单帧维度上生成，再 repeat 到 To
            memory_delta = self.memory_adapter(memory_state)       # (B, D)
            assert memory_delta.shape[-1] == self.step_feature_dim, f"memory_delta dim {memory_delta.shape[-1]} != step_feature_dim {self.step_feature_dim}"
            memory_delta = self.memory_residual_scale * torch.tanh(memory_delta)

            memory_delta = memory_delta.unsqueeze(1).repeat(1, self.n_obs_steps, 1)  # (B, To, D)
            memory_delta = memory_delta.reshape(batch_size, -1)                       # (B, To * D)

            global_cond = base_global_cond + memory_delta

        assert global_cond.shape == base_global_cond.shape, \
            f"global_cond shape {global_cond.shape} != base_global_cond shape {base_global_cond.shape}"

        aux_stats = {
            "memory_seq_norm": memory_seq.norm(dim=-1).mean().item(),
            "memory_state_norm": memory_state.norm(dim=-1).mean().item(),
            "memory_delta_norm": memory_delta.norm(dim=-1).mean().item(),
        }
        return global_cond, aux_stats

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color and 'point_cloud' in nobs:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        batch_size = nactions.shape[0]
        trajectory = nactions
        device = trajectory.device

        global_cond, aux_stats = self._build_global_cond(nobs, batch_size)

        # ===== 保持原始 PISB 的固定 physics target =====
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

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)

        if not self.use_pc_color and 'point_cloud' in nobs:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim

        device = self.device
        dtype = self.dtype

        global_cond, _ = self._build_global_cond(nobs, B)

        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)

        model = self.model
        model.eval()

        x_current = torch.randn(size=cond_data.shape, dtype=cond_data.dtype, device=cond_data.device)

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

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred,
        }
        return result
