import sys
sys.path.append('mp1')

from typing import Dict
import torch
import torch.nn as nn

from mp1.policy.pisb_policy import PISBPolicy, stopgrad, adaptive_l2_loss
from mp1.common.pytorch_util import dict_apply


class PISBPointAdapterPolicy(PISBPolicy):
    """
    PISB + Point-Adapter-X

    核心思想：
    1) 保持 PISB 的 physics bridge / target velocity / Euler sampling 完全不变
    2) 在输入条件侧增加一个 bounded adapter
    3) adapter 同时学习:
       - cond_delta: 对 global_cond 的小幅修正
       - feature_gate: 对 global_cond 的逐维重加权
    4) zero-init，初始严格退化为原始 PISB
    """

    def __init__(self, shape_meta, **kwargs):
        super().__init__(shape_meta, **kwargs)

        if not self.obs_as_global_cond:
            raise NotImplementedError("PISBPointAdapterPolicy requires obs_as_global_cond=True")

        self.adapter_hidden_dim = kwargs.get("adapter_hidden_dim", 256)
        self.adapter_scale = kwargs.get("adapter_scale", 0.10)
        self.adapter_gate_scale = kwargs.get("adapter_gate_scale", 0.10)

        cond_dim = self.obs_feature_dim * self.n_obs_steps \
            if "cross_attention" not in self.condition_type else self.obs_feature_dim

        # 输入侧 bounded adapter
        self.point_adapter = nn.Sequential(
            nn.Linear(cond_dim, self.adapter_hidden_dim),
            nn.LayerNorm(self.adapter_hidden_dim),
            nn.Mish(),
            nn.Linear(self.adapter_hidden_dim, 2 * cond_dim)
        ).to(self.device)

        with torch.no_grad():
            self.point_adapter[-1].weight.zero_()
            self.point_adapter[-1].bias.zero_()

    def _build_global_cond(self, nobs: Dict[str, torch.Tensor], batch_size: int):
        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)

        if "cross_attention" in self.condition_type:
            global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            pooled = global_cond.mean(dim=1)
        else:
            global_cond = nobs_features.reshape(batch_size, -1)
            pooled = global_cond

        raw = self.point_adapter(pooled)
        cond_delta, cond_gate = raw.chunk(2, dim=-1)

        cond_delta = self.adapter_scale * torch.tanh(cond_delta)
        cond_gate = 1.0 + self.adapter_gate_scale * torch.tanh(cond_gate)

        if global_cond.dim() == 3:
            global_cond = global_cond * cond_gate.unsqueeze(1) + cond_delta.unsqueeze(1)
        else:
            global_cond = global_cond * cond_gate + cond_delta

        aux_stats = {
            "adapter_delta_norm": cond_delta.norm(dim=-1).mean().item(),
            "adapter_gate_mean": cond_gate.mean().item(),
            "adapter_gate_abs": (cond_gate - 1.0).abs().mean().item(),
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