import sys
sys.path.append('mp1')

from typing import Dict, Tuple
import torch
import torch.nn as nn

from mp1.policy.pisb_policy import PISBPolicy, stopgrad, adaptive_l2_loss
from mp1.common.pytorch_util import dict_apply


class PISBPointAdapterPolicy(PISBPolicy):
    """
    PISB + Point-Adapter-X (revised)

    Design rules:
    1) Keep the original PISB physics bridge / target velocity / Euler inference unchanged.
    2) Only apply a bounded residual adapter on the input-side global conditioning.
    3) The adapter is deliberately weak: zero-init, low-capacity, small gate/delta scales.
    4) Add a tiny regularization term so the adapter stays a small correction instead of becoming
       another strong backbone that disturbs an already-strong PISB baseline.

    Note:
    - This implementation is actually a global-cond adapter, not a token-level point adapter.
      It operates on obs_encoder outputs after point-cloud encoding.
    - cross_attention mode is intentionally disabled here. The current stable use case is film-style
      global_cond with shape (B, obs_feature_dim * n_obs_steps).
    """

    def __init__(self, shape_meta, **kwargs):
        super().__init__(shape_meta, **kwargs)

        if not self.obs_as_global_cond:
            raise NotImplementedError("PISBPointAdapterPolicy requires obs_as_global_cond=True")
        if "cross_attention" in self.condition_type:
            raise NotImplementedError(
                "This revised Point-Adapter only supports non-cross-attention conditioning. "
                "Use condition_type='film' or other non-cross-attention variants."
            )

        self.adapter_hidden_dim = kwargs.get("adapter_hidden_dim", 128)
        self.adapter_scale = kwargs.get("adapter_scale", 0.02)
        self.adapter_gate_scale = kwargs.get("adapter_gate_scale", 0.02)
        self.adapter_reg_weight = kwargs.get("adapter_reg_weight", 1.0e-4)
        self.adapter_use_stopgrad_gate = kwargs.get("adapter_use_stopgrad_gate", False)

        # For the current non-cross-attention PISB, global_cond is flattened over n_obs_steps.
        self.cond_dim = self.obs_feature_dim * self.n_obs_steps

        self.point_adapter = nn.Sequential(
            nn.Linear(self.cond_dim, self.adapter_hidden_dim),
            nn.LayerNorm(self.adapter_hidden_dim),
            nn.Mish(),
            nn.Linear(self.adapter_hidden_dim, 2 * self.cond_dim),
        ).to(self.device)

        with torch.no_grad():
            self.point_adapter[-1].weight.zero_()
            self.point_adapter[-1].bias.zero_()

        print(
            f"[PISB-PointAdapter] cond_dim={self.cond_dim}, hidden={self.adapter_hidden_dim}, "
            f"delta_scale={self.adapter_scale}, gate_scale={self.adapter_gate_scale}, "
            f"reg={self.adapter_reg_weight}"
        )

    def _encode_obs_features(self, nobs: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(batch_size, -1)
        return global_cond

    def _apply_adapter(self, base_global_cond: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:
        raw = self.point_adapter(base_global_cond)
        cond_delta_raw, cond_gate_raw = raw.chunk(2, dim=-1)

        cond_delta = self.adapter_scale * torch.tanh(cond_delta_raw)
        gate_residual = self.adapter_gate_scale * torch.tanh(cond_gate_raw)

        gate_source = stopgrad(base_global_cond) if self.adapter_use_stopgrad_gate else base_global_cond
        adapted_global_cond = base_global_cond + gate_residual * gate_source + cond_delta

        adapter_reg = (cond_delta.pow(2).mean() + gate_residual.pow(2).mean())

        aux_tensors = {
            "adapter_reg": adapter_reg,
            "adapter_delta": cond_delta,
            "adapter_gate_residual": gate_residual,
        }
        aux_stats = {
            "adapter_delta_norm": cond_delta.norm(dim=-1).mean().item(),
            "adapter_gate_abs": gate_residual.abs().mean().item(),
            "adapter_cond_shift": (adapted_global_cond - base_global_cond).norm(dim=-1).mean().item(),
            "adapter_reg": adapter_reg.item(),
        }
        return adapted_global_cond, aux_tensors, aux_stats

    def _build_global_cond(self, nobs: Dict[str, torch.Tensor], batch_size: int):
        base_global_cond = self._encode_obs_features(nobs, batch_size)
        adapted_global_cond, aux_tensors, aux_stats = self._apply_adapter(base_global_cond)
        return adapted_global_cond, aux_tensors, aux_stats

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color and 'point_cloud' in nobs:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        batch_size = nactions.shape[0]
        trajectory = nactions
        device = trajectory.device

        global_cond, aux_tensors, aux_stats = self._build_global_cond(nobs, batch_size)

        # Original PISB target construction stays unchanged.
        eps = 1e-5
        t = torch.rand((batch_size,), device=device).float() * (1 - 2 * eps) + eps
        x1 = trajectory
        x0 = torch.randn_like(x1)
        t_expand = t.view(batch_size, 1, 1)

        xt, _, _ = self.physics_bridge.sample_path(x0, x1, t_expand)
        u_tgt = self.physics_bridge.compute_target_velocity(x0, x1, t_expand, xt)

        r_zeros = torch.zeros_like(t)
        model_output = self.model(
            sample=xt,
            timestep=t,
            local_cond=None,
            global_cond=global_cond,
            r=r_zeros,
            training=True,
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

        adapter_reg_loss = self.adapter_reg_weight * aux_tensors["adapter_reg"]
        loss = meanflow_loss + 0.5 * dis_loss + adapter_reg_loss
        mse_val = (stopgrad(error) ** 2).mean()

        loss_dict = {
            'loss': loss.item(),
            'bc_loss': loss.item(),
            'mse_val': mse_val.item(),
            'meanflow_loss': meanflow_loss.item(),
            'dis_loss': dis_loss.item() if isinstance(dis_loss, torch.Tensor) else dis_loss,
            'adapter_reg_loss': adapter_reg_loss.item(),
            **aux_stats,
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

        global_cond, _, _ = self._build_global_cond(nobs, B)

        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        model = self.model
        model.eval()

        x_current = torch.randn(size=cond_data.shape, dtype=cond_data.dtype, device=cond_data.device)

        steps = self.num_inference_steps if self.num_inference_steps is not None else 10
        dt = 1.0 / steps
        r_zeros = torch.zeros((B,), device=device, dtype=dtype)

        with torch.no_grad():
            for i in range(steps):
                t_val = i / steps
                t_tensor = torch.full((B,), t_val, device=device, dtype=dtype)

                model_output = model(
                    sample=x_current,
                    timestep=t_tensor,
                    local_cond=None,
                    global_cond=global_cond,
                    r=r_zeros,
                    training=False,
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
