import sys
sys.path.append('mp1')

from typing import Dict

import torch
from termcolor import cprint

from mp1.policy.pa_pisb_nocollapse_policy import PAPISBNoCollapsePolicy


class PAPISBStructurePreservingPolicy(PAPISBNoCollapsePolicy):
    """
    Innovation 2 on top of PA-PISB:
    bridge-structure-preserving integration at inference time.

    PA-PISB already exposes a closed-form linear bridge coefficient

        a_t(c) = dot(rho)(t) / rho(t) - 0.5 * dot(lambda_t)(c) / lambda_t(c)

    from the target velocity

        u_tgt = d mu_t / dt + a_t(c) * (x_t - mu_t).

    Instead of applying plain Euler to the whole learned field, we split the
    learned velocity into

        v_theta(x, t, c) = a_t(c) * x + g_theta(x, t, c),

    A naive exponential integrator on the full a_t(c) is unstable here because
    early bridge time has a_t(c) > 0: it deliberately expands the tube before
    the late-time contraction phase. Exponentiating that expansive part amplifies
    rollout noise too aggressively.

    Therefore we only preserve the contractive part

        a_t^-(c) = min(a_t(c), 0)

    and integrate

        x_{k+1} = exp(a_t^- * dt) * x_k + dt * phi_1(a_t^- * dt) * g_theta,
        phi_1(z) = (exp(z) - 1) / z.

    The residual term g_theta is clipped in bridge-normalized coordinates using
    PA-PISB's anisotropic tube scale

        s_t(c) = rho(t) * lambda_t(c)^(-1/2),

    so each update stays within a small number of local bridge radii. This makes
    the sampler domain-specific to robot manipulation: the bridge's stiffness and
    anisotropy directly control rollout stability.
    """

    def __init__(
        self,
        *args,
        sp_time_eps: float = 1.0e-5,
        sp_drift_dt_clip: float = 3.0,
        sp_residual_scale_clip: float = 2.5,
        sp_midpoint_eval: bool = True,
        sp_std_floor_ratio: float = 0.25,
        **kwargs
    ):
        self.sp_time_eps = sp_time_eps
        self.sp_drift_dt_clip = sp_drift_dt_clip
        self.sp_residual_scale_clip = sp_residual_scale_clip
        self.sp_midpoint_eval = sp_midpoint_eval
        self.sp_std_floor_ratio = sp_std_floor_ratio
        super().__init__(*args, **kwargs)

        cprint(
            "[PA-PISB-SPI] "
            f"time_eps={sp_time_eps}, drift_dt_clip={sp_drift_dt_clip}, "
            f"residual_scale_clip={sp_residual_scale_clip}, midpoint_eval={sp_midpoint_eval}, "
            f"std_floor_ratio={sp_std_floor_ratio}",
            "cyan",
        )

    def _phi1(self, z: torch.Tensor) -> torch.Tensor:
        small = z.abs() < 1.0e-4
        series = 1.0 + 0.5 * z + (z * z) / 6.0
        stable = torch.expm1(z) / (z + 1.0e-12)
        return torch.where(small, series, stable)

    def _solver_time(self, step_idx: int, steps: int, batch_size: int, device, dtype) -> torch.Tensor:
        if self.sp_midpoint_eval:
            t_val = (step_idx + 0.5) / steps
        else:
            t_val = step_idx / steps
        t_val = min(max(t_val, self.sp_time_eps), 1.0 - self.sp_time_eps)
        return torch.full((batch_size,), t_val, device=device, dtype=dtype)

    def _compute_bridge_solver_terms(self, cond_feat: torch.Tensor, t_expand: torch.Tensor):
        safe_t = torch.clamp(t_expand, min=self.sp_time_eps, max=1.0 - self.sp_time_eps)

        _, _, rho_t = self.physics_bridge.compute_scalar_coefficients(safe_t)
        lambda_t, dot_lambda_t, _ = self.physics_bridge._compute_lambda_terms(cond_feat, safe_t)

        # d log rho / dt for rho(t) = sigma * sqrt(t(1-t)) * exp(-k t)
        dlog_rho_dt = 0.5 * (1.0 - 2.0 * safe_t) / (safe_t * (1.0 - safe_t) + 1.0e-6)
        dlog_rho_dt = dlog_rho_dt - self.physics_bridge.stiffness

        drift_coeff = dlog_rho_dt - 0.5 * (dot_lambda_t / (lambda_t + 1.0e-6))
        contractive_coeff = torch.minimum(drift_coeff, torch.zeros_like(drift_coeff))
        std_diag = rho_t * torch.rsqrt(lambda_t)
        std_floor = self.physics_bridge.sigma_max * self.sp_std_floor_ratio
        std_diag = torch.clamp(std_diag, min=std_floor)

        contractive_coeff = contractive_coeff.reshape(cond_feat.shape[0], self.horizon, self.action_dim)
        std_diag = std_diag.reshape(cond_feat.shape[0], self.horizon, self.action_dim)
        return contractive_coeff, std_diag

    def _structure_preserving_step(
        self,
        x_current: torch.Tensor,
        v_pred: torch.Tensor,
        contractive_coeff: torch.Tensor,
        std_diag: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        residual = v_pred - contractive_coeff * x_current

        z = torch.clamp(
            contractive_coeff * dt,
            min=-self.sp_drift_dt_clip,
            max=0.0,
        )
        exp_z = torch.exp(z)
        phi_1 = self._phi1(z)
        delta = dt * phi_1 * residual

        if self.sp_residual_scale_clip > 0:
            delta_scaled = delta / (std_diag + 1.0e-6)
            delta_scaled = torch.clamp(
                delta_scaled,
                min=-self.sp_residual_scale_clip,
                max=self.sp_residual_scale_clip,
            )
            delta = delta_scaled * std_diag

        return exp_z * x_current + delta

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        value = next(iter(nobs.values()))
        batch_size, n_obs = value.shape[:2]
        horizon = self.horizon
        action_dim = self.action_dim
        device = self.device
        dtype = self.dtype

        _, pooled_features, global_cond = self._encode_obs(nobs, batch_size)
        local_cond = None
        cond_data = torch.zeros(size=(batch_size, horizon, action_dim), device=device, dtype=dtype)

        model = self.model
        model.eval()
        self.physics_bridge.eval()
        x_current = torch.randn(size=cond_data.shape, dtype=cond_data.dtype, device=cond_data.device)
        steps = self.num_inference_steps if self.num_inference_steps is not None else 10
        dt = 1.0 / steps
        r_zeros = torch.zeros((batch_size,), device=device)

        with torch.no_grad():
            for step_idx in range(steps):
                t_tensor = self._solver_time(step_idx, steps, batch_size, device, dtype)
                model_output = model(
                    sample=x_current,
                    timestep=t_tensor,
                    local_cond=local_cond,
                    global_cond=global_cond,
                    r=r_zeros,
                    training=False,
                )
                v_pred = model_output[0] if isinstance(model_output, tuple) else model_output
                contractive_coeff, std_diag = self._compute_bridge_solver_terms(
                    pooled_features, t_tensor.view(batch_size, 1, 1)
                )
                x_current = self._structure_preserving_step(
                    x_current=x_current,
                    v_pred=v_pred,
                    contractive_coeff=contractive_coeff,
                    std_diag=std_diag,
                    dt=dt,
                )

        naction_pred = x_current[..., :action_dim]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        start = n_obs - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return {'action': action, 'action_pred': action_pred}
