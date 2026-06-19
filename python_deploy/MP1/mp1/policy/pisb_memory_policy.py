import sys
sys.path.append('mp1')

from typing import Dict
import torch
import torch.nn as nn

from mp1.policy.pisb_policy import PISBPolicy, stopgrad, adaptive_l2_loss
from mp1.common.pytorch_util import dict_apply


class PISBMemoryPolicy(PISBPolicy):
    """
    PISB + GRU-based History Memory

    核心思想：
    1) 保持 PISB 的 physics bridge / target velocity / Euler sampling 完全不变
    2) 逐帧编码 obs，得到 per-step feature
    3) 用 GRU 将时间窗口内的历史 obs feature 压成 memory state
    4) 用 zero-init residual adapter 将 memory 融入 global_cond
    """

    def __init__(self, shape_meta, **kwargs):
        super().__init__(shape_meta, **kwargs)

        if not self.obs_as_global_cond:
            raise NotImplementedError("PISBMemoryPolicy requires obs_as_global_cond=True")

        self.memory_hidden_dim = kwargs.get("memory_hidden_dim", self.obs_feature_dim)
        self.memory_num_layers = kwargs.get("memory_num_layers", 1)
        self.memory_residual_scale = kwargs.get("memory_residual_scale", 0.10)

        cond_dim = self.obs_feature_dim * self.n_obs_steps \
            if "cross_attention" not in self.condition_type else self.obs_feature_dim

        # 对逐帧 obs feature 做时序压缩
        self.history_gru = nn.GRU(
            input_size=self.obs_feature_dim,
            hidden_size=self.memory_hidden_dim,
            num_layers=self.memory_num_layers,
            batch_first=True
        ).to(self.device)

        # 将 memory state 投到 global_cond 维度，做 bounded residual
        self.memory_adapter = nn.Sequential(
            nn.Linear(self.memory_hidden_dim, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, cond_dim)
        ).to(self.device)

        with torch.no_grad():
            self.memory_adapter[-1].weight.zero_()
            self.memory_adapter[-1].bias.zero_()

    def _encode_obs_sequence(self, nobs: Dict[str, torch.Tensor], batch_size: int):
        """
        对每一时刻分别编码，得到 per-step features: (B, To, D)
        """
        this_nobs = dict_apply(
            nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        step_features = self.obs_encoder(this_nobs)  # (B*To, D)
        step_features = step_features.reshape(batch_size, self.n_obs_steps, -1)
        return step_features

    def _build_global_cond(self, nobs: Dict[str, torch.Tensor], batch_size: int):
        step_features = self._encode_obs_sequence(nobs, batch_size)  # (B, To, D)

        if "cross_attention" in self.condition_type:
            global_cond = step_features
        else:
            global_cond = step_features.reshape(batch_size, -1)

        # GRU 历史压缩
        memory_seq, h_n = self.history_gru(step_features)         # h_n: (L, B, H)
        memory_state = h_n[-1]                                    # (B, H)

        # bounded residual adapter
        memory_residual = self.memory_adapter(memory_state)       # (B, cond_dim)
        memory_residual = self.memory_residual_scale * torch.tanh(memory_residual)

        if global_cond.dim() == 3:
            global_cond = global_cond + memory_residual.unsqueeze(1)
        else:
            global_cond = global_cond + memory_residual

        aux_stats = {
            "memory_state_norm": memory_state.norm(dim=-1).mean().item(),
            "memory_residual_norm": memory_residual.norm(dim=-1).mean().item(),
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