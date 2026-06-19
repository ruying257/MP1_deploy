import sys

sys.path.append('mp1')

from typing import Dict

import math
import time
import torch
import torch.nn as nn
from termcolor import cprint

from mp1.policy.pa_pisb_inertial_bridge_policy import PAPISBInertialBridgePolicy
from mp1.policy.pa_pisb_nocollapse_policy import adaptive_l2_loss, stopgrad


class DisturbanceCoupledPAPISBBridge(nn.Module):
    """
    Disturbance-state-coupled PA-PISB bridge.

    Relative to PA-PISB-InertialBridge, this bridge does not only feed the
    inertial summary as a generic condition. It explicitly uses that summary to
    modulate:
        1) channel-wise precision lambda_t(c)
        2) bridge stiffness k(c)
        3) bridge damping gamma(c)
        4) bridge tube scale sigma(c)

    All disturbance-to-parameter heads are zero-initialized, so training starts
    exactly from the PA-PISB-InertialBridge baseline and only departs from it
    when the disturbance-coupled parameterization improves the objective.
    """

    def __init__(
        self,
        cond_dim: int,
        traj_dim: int,
        sigma_min: float = 1e-4,
        sigma_max: float = 0.1,
        stiffness: float = 0.1,
        damping: float = 0.1,
        lambda_amplitude: float = 0.35,
        stiffness_mod_amplitude: float = 0.30,
        damping_mod_amplitude: float = 0.30,
        sigma_mod_amplitude: float = 0.20,
        hidden_dim: int = 256,
        head_dropout: float = 0.0,
        device: str = 'cuda',
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.stiffness = stiffness
        self.damping = damping
        self.lambda_amplitude = lambda_amplitude
        self.stiffness_mod_amplitude = stiffness_mod_amplitude
        self.damping_mod_amplitude = damping_mod_amplitude
        self.sigma_mod_amplitude = sigma_mod_amplitude
        self.cond_dim = cond_dim
        self.traj_dim = traj_dim
        self.device = device

        self.feature_net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(head_dropout),
        )
        self.lambda_base_head = nn.Linear(hidden_dim, traj_dim)
        self.lambda_phase_head = nn.Linear(hidden_dim, traj_dim)
        self.stiffness_head = nn.Linear(hidden_dim, 1)
        self.damping_head = nn.Linear(hidden_dim, 1)
        self.sigma_head = nn.Linear(hidden_dim, 1)

        for head in (
            self.lambda_base_head,
            self.lambda_phase_head,
            self.stiffness_head,
            self.damping_head,
            self.sigma_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _compute_lambda_terms_from_hidden(self, h: torch.Tensor, t_expand: torch.Tensor):
        base_logits = self.lambda_base_head(h)
        phase_logits = self.lambda_phase_head(h)

        phase_argument = base_logits + t_expand.squeeze(-1) * phase_logits
        tanh_phase = torch.tanh(phase_argument)
        lambda_t = 1.0 + self.lambda_amplitude * tanh_phase
        dot_lambda_t = self.lambda_amplitude * (1.0 - tanh_phase.pow(2)) * phase_logits

        lambda_t = lambda_t.unsqueeze(1)
        dot_lambda_t = dot_lambda_t.unsqueeze(1)

        metrics = {
            'lambda_mean': lambda_t.mean(),
            'lambda_std': lambda_t.std(unbiased=False),
            'lambda_min': lambda_t.min(),
            'lambda_max': lambda_t.max(),
            'delta_lambda_mean': dot_lambda_t.mean(),
            'delta_lambda_abs_mean': dot_lambda_t.abs().mean(),
            'lambda_dim_std': lambda_t.std(dim=-1, unbiased=False).mean(),
            'std_ratio_mean': torch.rsqrt(lambda_t).mean(),
            'std_ratio_min': torch.rsqrt(lambda_t).min(),
            'std_ratio_max': torch.rsqrt(lambda_t).max(),
            'lambda_base_logit_abs_mean': base_logits.abs().mean(),
            'lambda_phase_logit_abs_mean': phase_logits.abs().mean(),
        }
        return lambda_t, dot_lambda_t, metrics

    def _compute_scalar_coefficients_from_hidden(self, h: torch.Tensor, t_expand: torch.Tensor):
        stiffness_residual = self.stiffness_mod_amplitude * torch.tanh(self.stiffness_head(h))
        damping_residual = self.damping_mod_amplitude * torch.tanh(self.damping_head(h))
        sigma_residual = self.sigma_mod_amplitude * torch.tanh(self.sigma_head(h))

        stiffness_eff = torch.clamp(self.stiffness * (1.0 + stiffness_residual), min=1e-5)
        damping_eff = torch.clamp(self.damping * (1.0 + damping_residual), min=1e-5)
        sigma_eff = torch.clamp(self.sigma_max * (1.0 + sigma_residual), min=max(self.sigma_min, 1e-5))

        stiffness_eff = stiffness_eff.unsqueeze(-1)
        damping_eff = damping_eff.unsqueeze(-1)
        sigma_eff = sigma_eff.unsqueeze(-1)

        safe_denom = torch.where(
            damping_eff.abs() > 1e-6,
            torch.sinh(damping_eff),
            damping_eff + 1e-6,
        )
        alpha_t = torch.sinh(damping_eff * (1.0 - t_expand)) / (safe_denom + 1e-6)
        beta_t = torch.sinh(damping_eff * t_expand) / (safe_denom + 1e-6)

        rho_t = sigma_eff * torch.sqrt(torch.clamp(t_expand * (1.0 - t_expand), min=1e-12))
        rho_t = rho_t * torch.exp(-stiffness_eff * t_expand)
        rho_t = torch.clamp(rho_t, min=self.sigma_min)

        aux = {
            'stiffness_eff': stiffness_eff,
            'damping_eff': damping_eff,
            'sigma_eff': sigma_eff,
            'stiffness_residual': stiffness_residual.unsqueeze(-1),
            'damping_residual': damping_residual.unsqueeze(-1),
            'sigma_residual': sigma_residual.unsqueeze(-1),
            'stiffness_eff_mean': stiffness_eff.mean(),
            'stiffness_eff_std': stiffness_eff.std(unbiased=False),
            'damping_eff_mean': damping_eff.mean(),
            'damping_eff_std': damping_eff.std(unbiased=False),
            'sigma_eff_mean': sigma_eff.mean(),
            'sigma_eff_std': sigma_eff.std(unbiased=False),
            'stiffness_residual_abs_mean': stiffness_residual.abs().mean(),
            'damping_residual_abs_mean': damping_residual.abs().mean(),
            'sigma_residual_abs_mean': sigma_residual.abs().mean(),
        }
        return alpha_t, beta_t, rho_t, aux

    def sample_path(self, x0, x1, t_expand, cond_feat):
        h = self.feature_net(cond_feat)
        alpha, beta, rho, scalar_aux = self._compute_scalar_coefficients_from_hidden(h, t_expand)
        lambda_t, dot_lambda_t, lambda_metrics = self._compute_lambda_terms_from_hidden(h, t_expand)

        mu_t = alpha * x0 + beta * x1
        eps = torch.randn_like(x0)
        std_diag = rho * torch.rsqrt(lambda_t)
        std_diag = std_diag.reshape_as(x0)
        lambda_t = lambda_t.reshape_as(x0)
        dot_lambda_t = dot_lambda_t.reshape_as(x0)
        xt = mu_t + std_diag * eps

        aux = {
            'mu_t': mu_t,
            'rho_t': rho,
            'lambda_t': lambda_t,
            'dot_lambda_t': dot_lambda_t,
            'std_diag': std_diag,
            **lambda_metrics,
            **scalar_aux,
        }
        return xt, aux

    def compute_target_velocity(self, x0, x1, t_expand, xt, cond_feat):
        t_in = t_expand.clone().detach().requires_grad_(True)
        h = self.feature_net(cond_feat)
        alpha, beta, rho, scalar_aux = self._compute_scalar_coefficients_from_hidden(h, t_in)
        lambda_t, dot_lambda_t, lambda_metrics = self._compute_lambda_terms_from_hidden(h, t_in)

        mu_t = alpha * x0 + beta * x1

        d_alpha = torch.autograd.grad(alpha.sum(), t_in, create_graph=True)[0]
        d_beta = torch.autograd.grad(beta.sum(), t_in, create_graph=True)[0]
        d_rho = torch.autograd.grad(rho.sum(), t_in, create_graph=True)[0]

        dt_mu = d_alpha * x0 + d_beta * x1
        drift_coeff = (d_rho / (rho + 1e-6)) - 0.5 * (dot_lambda_t / (lambda_t + 1e-6))
        lambda_t = lambda_t.reshape_as(x0)
        dot_lambda_t = dot_lambda_t.reshape_as(x0)
        drift_coeff = drift_coeff.reshape_as(x0)
        target_v = dt_mu + drift_coeff * (xt - mu_t)

        aux = {
            'mu_t': mu_t.detach(),
            'rho_t': rho.detach(),
            'lambda_t': lambda_t.detach(),
            'dot_lambda_t': dot_lambda_t.detach(),
            'drift_coeff': drift_coeff.detach(),
            **{k: v.detach() if torch.is_tensor(v) else v for k, v in lambda_metrics.items()},
            **{k: v.detach() if torch.is_tensor(v) else v for k, v in scalar_aux.items()},
        }
        return target_v.detach(), aux

    def regularization(
        self,
        aux: dict,
        center_weight: float = 1e-3,
        phase_weight: float = 1e-3,
        aniso_floor_weight: float = 0.0,
        aniso_floor: float = 0.02,
        scalar_weight: float = 1e-4,
    ):
        lambda_t = aux['lambda_t']
        dot_lambda_t = aux['dot_lambda_t']
        center_reg = (lambda_t - 1.0).pow(2).mean()
        phase_reg = dot_lambda_t.pow(2).mean()
        reg = center_weight * center_reg + phase_weight * phase_reg

        if aniso_floor_weight > 0:
            dim_std = lambda_t.std(dim=-1, unbiased=False)
            floor_penalty = torch.relu(torch.as_tensor(aniso_floor, device=lambda_t.device) - dim_std).pow(2).mean()
            reg = reg + aniso_floor_weight * floor_penalty

        scalar_reg = (
            aux['stiffness_residual'].pow(2).mean()
            + aux['damping_residual'].pow(2).mean()
            + aux['sigma_residual'].pow(2).mean()
        )
        reg = reg + scalar_weight * scalar_reg
        return reg


class PAPISBDisturbanceCoupledBridgePolicy(PAPISBInertialBridgePolicy):
    """
    Stronger disturbance-aware PA-PISB variant.

    Compared with PA-PISB-InertialBridge, this variant upgrades the inertial
    summary from a passive condition input to an explicit modulator of bridge
    dynamics. The inferred disturbance state can now directly reshape damping,
    stiffness, bridge scale, and anisotropic precision.
    """

    def __init__(
        self,
        shape_meta: dict,
        *args,
        dcb_stiffness_mod_amplitude: float = 0.30,
        dcb_damping_mod_amplitude: float = 0.30,
        dcb_sigma_mod_amplitude: float = 0.20,
        dcb_scalar_reg_weight: float = 1.0e-4,
        **kwargs,
    ):
        self.dcb_stiffness_mod_amplitude = dcb_stiffness_mod_amplitude
        self.dcb_damping_mod_amplitude = dcb_damping_mod_amplitude
        self.dcb_sigma_mod_amplitude = dcb_sigma_mod_amplitude
        self.dcb_scalar_reg_weight = dcb_scalar_reg_weight

        bridge_sigma = kwargs.get('bridge_sigma', 0.1)
        bridge_stiffness = kwargs.get('bridge_stiffness', 0.1)
        bridge_damping = kwargs.get('bridge_damping', 0.1)
        pa_lambda_amplitude = kwargs.get('pa_lambda_amplitude', 0.35)
        pa_head_hidden_dim = kwargs.get('pa_head_hidden_dim', 256)
        pa_head_dropout = kwargs.get('pa_head_dropout', 0.0)

        super().__init__(shape_meta, *args, **kwargs)

        summary_dim = self.obs_feature_dim * 3
        self.physics_bridge = DisturbanceCoupledPAPISBBridge(
            cond_dim=summary_dim,
            traj_dim=self.horizon * self.action_dim,
            sigma_min=1e-4,
            sigma_max=bridge_sigma,
            stiffness=bridge_stiffness,
            damping=bridge_damping,
            lambda_amplitude=pa_lambda_amplitude,
            stiffness_mod_amplitude=dcb_stiffness_mod_amplitude,
            damping_mod_amplitude=dcb_damping_mod_amplitude,
            sigma_mod_amplitude=dcb_sigma_mod_amplitude,
            hidden_dim=pa_head_hidden_dim,
            head_dropout=pa_head_dropout,
            device=self.device,
        )

        cprint(
            "[PA-PISB-DisturbanceCoupledBridge] "
            f"k_amp={dcb_stiffness_mod_amplitude}, gamma_amp={dcb_damping_mod_amplitude}, "
            f"sigma_amp={dcb_sigma_mod_amplitude}, scalar_reg={dcb_scalar_reg_weight}",
            "cyan",
        )

    def compute_loss(self, batch):
        profile_enabled = self._loss_profile_enabled
        loss_profile = {}

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        batch_size = nactions.shape[0]
        device = nactions.device
        self._maybe_sync(device, profile_enabled)
        total_start = time.perf_counter() if profile_enabled else None
        prep_start = time.perf_counter() if profile_enabled else None
        local_cond = None
        trajectory = nactions
        x1 = trajectory
        x0 = torch.randn_like(x1)
        if profile_enabled:
            self._maybe_sync(device, True)
            loss_profile['prep'] = time.perf_counter() - prep_start

        encode_start = time.perf_counter() if profile_enabled else None
        _, bridge_cond, global_cond = self._encode_obs(nobs, batch_size)
        if profile_enabled:
            self._maybe_sync(device, True)
            loss_profile['obs_encode'] = time.perf_counter() - encode_start

        eps = 1e-5
        t = torch.rand((batch_size,), device=device).float() * (1 - 2 * eps) + eps
        t_expand = t.view(batch_size, 1, 1)

        bridge_start = time.perf_counter() if profile_enabled else None
        xt, sample_aux = self.physics_bridge.sample_path(x0, x1, t_expand, bridge_cond)
        u_tgt, target_aux = self.physics_bridge.compute_target_velocity(x0, x1, t_expand, xt, bridge_cond)
        if profile_enabled:
            self._maybe_sync(device, True)
            loss_profile['bridge'] = time.perf_counter() - bridge_start

        r_zeros = torch.zeros_like(t)
        unet_start = time.perf_counter() if profile_enabled else None
        model_output = self.model(
            sample=xt,
            timestep=t,
            global_cond=global_cond,
            local_cond=local_cond,
            r=r_zeros,
        )
        if isinstance(model_output, tuple):
            v_pred, features = model_output
        else:
            v_pred, features = model_output, []
        if profile_enabled:
            self._maybe_sync(device, True)
            loss_profile['unet_forward'] = time.perf_counter() - unet_start

        loss_terms_start = time.perf_counter() if profile_enabled else None
        error = v_pred - stopgrad(u_tgt)
        meanflow_loss = adaptive_l2_loss(error)

        dis_loss = 0.0
        if features is not None and len(features) > 0:
            if isinstance(features, (list, tuple)):
                for feat in features:
                    dis_loss = dis_loss + self.dispersive_loss(feat)
            else:
                dis_loss = self.dispersive_loss(features)

        bridge_reg = self.physics_bridge.regularization(
            sample_aux,
            center_weight=self.pa_center_reg_weight,
            phase_weight=self.pa_phase_reg_weight,
            aniso_floor_weight=self.pa_aniso_floor_weight,
            aniso_floor=self.pa_aniso_floor,
            scalar_weight=self.dcb_scalar_reg_weight,
        )

        ib_reg = 0.0
        if self._ib_last_reg is not None:
            ib_reg = self.ib_reg_weight * self._ib_last_reg

        loss = meanflow_loss + 0.5 * dis_loss + bridge_reg + ib_reg
        mse_val = (stopgrad(error) ** 2).mean()
        if profile_enabled:
            self._maybe_sync(device, True)
            loss_profile['loss_terms'] = time.perf_counter() - loss_terms_start

        loss_dict = {
            'loss': loss.item(),
            'bc_loss': loss.item(),
            'mse_val': mse_val.item(),
            'meanflow_loss': meanflow_loss.item(),
            'dis_loss': dis_loss.item() if isinstance(dis_loss, torch.Tensor) else dis_loss,
            'bridge_reg': bridge_reg.item(),
            'ib_reg': ib_reg.item() if isinstance(ib_reg, torch.Tensor) else ib_reg,
        }
        if profile_enabled:
            loss_profile['total'] = time.perf_counter() - total_start
            self._last_loss_profile = loss_profile
        else:
            self._last_loss_profile = {}
        return loss, loss_dict
