import sys
sys.path.append('mp1')

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint

from mp1.policy.pisb_policy import adaptive_l2_loss, stopgrad
from mp1.policy.pa_pisb_nocollapse_policy import PAPISBNoCollapsePolicy


class PAPISBPhaseWarpBridge(nn.Module):
    """
    PA-PISB + Phase Warp bridge.

    Combined bridge:
        tau_c(t) = sigmoid(a(c) * logit(t) + b(c))
        mu_c(t) = alpha(tau_c(t)) x0 + beta(tau_c(t)) x1
        rho_c(t) = sigma_max * sqrt(tau_c(t)(1-tau_c(t))) * exp(-k tau_c(t))
        lambda_c(t) = 1 + a_lambda * tanh(g0(c) + tau_c(t) * g1(c))
        Sigma_c(t) = rho_c(t)^2 * Diag(lambda_c(t))^{-1}
        x_t = mu_c(t) + rho_c(t) * lambda_c(t)^(-1/2) * eps

    Target velocity:
        u_tgt = d mu_c / dt
                + (d rho_c / dt / rho_c - 0.5 * d lambda_c / dt / lambda_c) * (x_t - mu_c)

    All added heads are zero-initialized, so the bridge starts exactly from the
    same isotropic baseline as PA-PISB with tau_c(t)=t and lambda_c(t)=1.
    """

    def __init__(
        self,
        cond_dim: int,
        traj_dim: int,
        sigma_min: float = 1.0e-4,
        sigma_max: float = 0.1,
        stiffness: float = 0.1,
        damping: float = 0.1,
        lambda_amplitude: float = 0.35,
        hidden_dim: int = 256,
        head_dropout: float = 0.0,
        phase_scale_amplitude: float = 0.10,
        phase_shift_amplitude: float = 0.10,
        time_eps: float = 1.0e-5,
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.stiffness = stiffness
        self.damping = damping
        self.lambda_amplitude = lambda_amplitude
        self.phase_scale_amplitude = phase_scale_amplitude
        self.phase_shift_amplitude = phase_shift_amplitude
        self.time_eps = time_eps

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
        self.phase_scale_head = nn.Linear(hidden_dim, 1)
        self.phase_shift_head = nn.Linear(hidden_dim, 1)

        nn.init.zeros_(self.lambda_base_head.weight)
        nn.init.zeros_(self.lambda_base_head.bias)
        nn.init.zeros_(self.lambda_phase_head.weight)
        nn.init.zeros_(self.lambda_phase_head.bias)
        nn.init.zeros_(self.phase_scale_head.weight)
        nn.init.zeros_(self.phase_scale_head.bias)
        nn.init.zeros_(self.phase_shift_head.weight)
        nn.init.zeros_(self.phase_shift_head.bias)

    def compute_scalar_coefficients(self, tau_t: torch.Tensor):
        gamma = self.damping
        denom = math.sinh(gamma) if abs(gamma) > 1.0e-6 else 1.0

        alpha_t = torch.sinh(gamma * (1.0 - tau_t)) / (denom + 1.0e-6)
        beta_t = torch.sinh(gamma * tau_t) / (denom + 1.0e-6)
        rho_t = self.sigma_max * torch.sqrt(torch.clamp(tau_t * (1.0 - tau_t), min=1.0e-12))
        rho_t = rho_t * torch.exp(-self.stiffness * tau_t)
        rho_t = torch.clamp(rho_t, min=self.sigma_min)
        return alpha_t, beta_t, rho_t

    def _compute_phase_warp(self, cond_feat: torch.Tensor, t_expand: torch.Tensor):
        safe_t = torch.clamp(t_expand, min=self.time_eps, max=1.0 - self.time_eps)
        logit_t = torch.log(safe_t) - torch.log1p(-safe_t)

        h = self.feature_net(cond_feat)
        scale_logits = self.phase_scale_head(h)
        shift_logits = self.phase_shift_head(h)

        log_scale = self.phase_scale_amplitude * torch.tanh(scale_logits)
        phase_scale = torch.exp(log_scale).unsqueeze(1)
        phase_shift = (
            self.phase_shift_amplitude * torch.tanh(shift_logits)
        ).unsqueeze(1)

        tau_t = torch.sigmoid(phase_scale * logit_t + phase_shift)
        tau_t = torch.clamp(tau_t, min=self.time_eps, max=1.0 - self.time_eps)
        dtau_dt = phase_scale * tau_t * (1.0 - tau_t) / (safe_t * (1.0 - safe_t) + 1.0e-6)

        metrics = {
            'tau_mean': tau_t.mean(),
            'tau_std': tau_t.std(unbiased=False),
            'dtau_dt_mean': dtau_dt.mean(),
            'dtau_dt_min': dtau_dt.min(),
            'dtau_dt_max': dtau_dt.max(),
            'phase_scale_mean': phase_scale.mean(),
            'phase_scale_min': phase_scale.min(),
            'phase_scale_max': phase_scale.max(),
            'phase_shift_mean': phase_shift.mean(),
            'phase_shift_abs_mean': phase_shift.abs().mean(),
        }
        return h, tau_t, dtau_dt, phase_scale, phase_shift, metrics

    def _compute_lambda_terms(self, h: torch.Tensor, tau_t: torch.Tensor, dtau_dt: torch.Tensor):
        base_logits = self.lambda_base_head(h)
        phase_logits = self.lambda_phase_head(h)

        phase_argument = base_logits + tau_t.squeeze(-1) * phase_logits
        tanh_phase = torch.tanh(phase_argument)
        lambda_t = 1.0 + self.lambda_amplitude * tanh_phase
        dlambda_dt = self.lambda_amplitude * (1.0 - tanh_phase.pow(2)) * phase_logits * dtau_dt.squeeze(-1)

        lambda_t = lambda_t.unsqueeze(1)
        dlambda_dt = dlambda_dt.unsqueeze(1)

        std_ratio = torch.rsqrt(lambda_t)
        metrics = {
            'lambda_mean': lambda_t.mean(),
            'lambda_std': lambda_t.std(unbiased=False),
            'lambda_min': lambda_t.min(),
            'lambda_max': lambda_t.max(),
            'delta_lambda_mean': dlambda_dt.mean(),
            'delta_lambda_abs_mean': dlambda_dt.abs().mean(),
            'lambda_dim_std': lambda_t.std(dim=-1, unbiased=False).mean(),
            'std_ratio_mean': std_ratio.mean(),
            'std_ratio_min': std_ratio.min(),
            'std_ratio_max': std_ratio.max(),
            'lambda_base_logit_abs_mean': base_logits.abs().mean(),
            'lambda_phase_logit_abs_mean': phase_logits.abs().mean(),
        }
        return lambda_t, dlambda_dt, metrics

    def sample_path(self, x0, x1, t_expand, cond_feat):
        h, tau_t, dtau_dt, phase_scale, phase_shift, warp_metrics = self._compute_phase_warp(
            cond_feat, t_expand
        )
        lambda_t, dlambda_dt, lambda_metrics = self._compute_lambda_terms(h, tau_t, dtau_dt)
        alpha_t, beta_t, rho_t = self.compute_scalar_coefficients(tau_t)

        mu_t = alpha_t * x0 + beta_t * x1
        eps = torch.randn_like(x0)
        std_diag = rho_t * torch.rsqrt(lambda_t)
        std_diag = std_diag.reshape_as(x0)
        lambda_t = lambda_t.reshape_as(x0)
        dlambda_dt = dlambda_dt.reshape_as(x0)
        xt = mu_t + std_diag * eps

        aux = {
            'tau_t': tau_t,
            'dtau_dt': dtau_dt,
            'phase_scale': phase_scale,
            'phase_shift': phase_shift,
            'lambda_t': lambda_t,
            'dlambda_dt': dlambda_dt,
            'mu_t': mu_t,
            'rho_t': rho_t,
            'std_diag': std_diag,
            **warp_metrics,
            **lambda_metrics,
        }
        return xt, aux

    def compute_target_velocity(self, x0, x1, t_expand, xt, cond_feat):
        t_in = t_expand.clone().detach().requires_grad_(True)
        h, tau_t, dtau_dt, phase_scale, phase_shift, warp_metrics = self._compute_phase_warp(
            cond_feat, t_in
        )
        lambda_t, dlambda_dt, lambda_metrics = self._compute_lambda_terms(h, tau_t, dtau_dt)
        alpha_t, beta_t, rho_t = self.compute_scalar_coefficients(tau_t)
        mu_t = alpha_t * x0 + beta_t * x1

        d_alpha_dt = torch.autograd.grad(alpha_t.sum(), t_in, create_graph=True)[0]
        d_beta_dt = torch.autograd.grad(beta_t.sum(), t_in, create_graph=True)[0]
        d_rho_dt = torch.autograd.grad(rho_t.sum(), t_in, create_graph=True)[0]

        dt_mu = d_alpha_dt * x0 + d_beta_dt * x1
        drift_coeff = (d_rho_dt / (rho_t + 1.0e-6)) - 0.5 * (dlambda_dt / (lambda_t + 1.0e-6))
        lambda_t = lambda_t.reshape_as(x0)
        dlambda_dt = dlambda_dt.reshape_as(x0)
        drift_coeff = drift_coeff.reshape_as(x0)
        target_v = dt_mu + drift_coeff * (xt - mu_t)

        aux = {
            'tau_t': tau_t.detach(),
            'dtau_dt': dtau_dt.detach(),
            'phase_scale': phase_scale.detach(),
            'phase_shift': phase_shift.detach(),
            'lambda_t': lambda_t.detach(),
            'dlambda_dt': dlambda_dt.detach(),
            'mu_t': mu_t.detach(),
            'rho_t': rho_t.detach(),
            'drift_coeff': drift_coeff.detach(),
            **{k: v.detach() if torch.is_tensor(v) else v for k, v in warp_metrics.items()},
            **{k: v.detach() if torch.is_tensor(v) else v for k, v in lambda_metrics.items()},
        }
        return target_v.detach(), aux

    def regularization(
        self,
        aux: dict,
        center_weight: float = 1.0e-3,
        phase_weight: float = 1.0e-3,
        aniso_floor_weight: float = 0.0,
        aniso_floor: float = 0.02,
        identity_weight: float = 1.0e-3,
        slope_weight: float = 1.0e-4,
    ):
        lambda_t = aux['lambda_t']
        dlambda_dt = aux['dlambda_dt']
        phase_scale = aux['phase_scale']
        phase_shift = aux['phase_shift']
        dtau_dt = aux['dtau_dt']

        center_reg = (lambda_t - 1.0).pow(2).mean()
        phase_reg = dlambda_dt.pow(2).mean()
        identity_reg = (phase_scale - 1.0).pow(2).mean() + phase_shift.pow(2).mean()
        slope_reg = (dtau_dt - 1.0).pow(2).mean()

        reg = (
            center_weight * center_reg
            + phase_weight * phase_reg
            + identity_weight * identity_reg
            + slope_weight * slope_reg
        )

        if aniso_floor_weight > 0:
            dim_std = lambda_t.std(dim=-1, unbiased=False)
            floor_penalty = F.relu(aniso_floor - dim_std).pow(2).mean()
            reg = reg + aniso_floor_weight * floor_penalty
        return reg


class PAPISBPhaseWarpPolicy(PAPISBNoCollapsePolicy):
    def __init__(
        self,
        *args,
        pw_phase_scale_amplitude: float = 0.10,
        pw_phase_shift_amplitude: float = 0.10,
        pw_identity_reg_weight: float = 5.0e-3,
        pw_slope_reg_weight: float = 1.0e-3,
        pw_time_eps: float = 1.0e-5,
        **kwargs
    ):
        self.pw_phase_scale_amplitude = pw_phase_scale_amplitude
        self.pw_phase_shift_amplitude = pw_phase_shift_amplitude
        self.pw_identity_reg_weight = pw_identity_reg_weight
        self.pw_slope_reg_weight = pw_slope_reg_weight
        self.pw_time_eps = pw_time_eps
        super().__init__(*args, **kwargs)

        self.physics_bridge = PAPISBPhaseWarpBridge(
            cond_dim=self.obs_feature_dim,
            traj_dim=self.horizon * self.action_dim,
            sigma_min=1.0e-4,
            sigma_max=kwargs.get('bridge_sigma', 0.1),
            stiffness=kwargs.get('bridge_stiffness', 0.1),
            damping=kwargs.get('bridge_damping', 0.1),
            lambda_amplitude=kwargs.get('pa_lambda_amplitude', 0.35),
            hidden_dim=kwargs.get('pa_head_hidden_dim', 256),
            head_dropout=kwargs.get('pa_head_dropout', 0.0),
            phase_scale_amplitude=pw_phase_scale_amplitude,
            phase_shift_amplitude=pw_phase_shift_amplitude,
            time_eps=pw_time_eps,
        )

        cprint(
            "[PA-PISB-PhaseWarp] "
            f"lambda_amp={kwargs.get('pa_lambda_amplitude', 0.35)}, "
            f"pw_scale_amp={pw_phase_scale_amplitude}, pw_shift_amp={pw_phase_shift_amplitude}",
            "cyan",
        )

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        batch_size = nactions.shape[0]
        device = nactions.device
        x1 = nactions
        x0 = torch.randn_like(x1)

        _, pooled_features, global_cond = self._encode_obs(nobs, batch_size)

        eps = self.pw_time_eps
        t = torch.rand((batch_size,), device=device).float() * (1.0 - 2.0 * eps) + eps
        t_expand = t.view(batch_size, 1, 1)

        xt, _ = self.physics_bridge.sample_path(x0, x1, t_expand, pooled_features)
        u_tgt, target_aux = self.physics_bridge.compute_target_velocity(
            x0, x1, t_expand, xt, pooled_features
        )

        r_zeros = torch.zeros_like(t)
        model_output = self.model(
            sample=xt,
            timestep=t,
            global_cond=global_cond,
            local_cond=None,
            r=r_zeros,
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

        pa_pw_reg = self.physics_bridge.regularization(
            target_aux,
            center_weight=self.pa_center_reg_weight,
            phase_weight=self.pa_phase_reg_weight,
            aniso_floor_weight=self.pa_aniso_floor_weight,
            aniso_floor=self.pa_aniso_floor,
            identity_weight=self.pw_identity_reg_weight,
            slope_weight=self.pw_slope_reg_weight,
        )

        loss = meanflow_loss + 0.5 * dis_loss + pa_pw_reg
        mse_val = (stopgrad(error) ** 2).mean()

        loss_dict = {
            'loss': loss.item(),
            'bc_loss': loss.item(),
            'mse_val': mse_val.item(),
            'meanflow_loss': meanflow_loss.item(),
            'dis_loss': dis_loss.item() if isinstance(dis_loss, torch.Tensor) else dis_loss,
            'pa_pw_reg': pa_pw_reg.item(),
            'lambda_mean': target_aux['lambda_mean'].detach().item(),
            'lambda_std': target_aux['lambda_std'].detach().item(),
            'lambda_min': target_aux['lambda_min'].detach().item(),
            'lambda_max': target_aux['lambda_max'].detach().item(),
            'delta_lambda_mean': target_aux['delta_lambda_mean'].detach().item(),
            'delta_lambda_abs_mean': target_aux['delta_lambda_abs_mean'].detach().item(),
            'lambda_dim_std': target_aux['lambda_dim_std'].detach().item(),
            'std_ratio_mean': target_aux['std_ratio_mean'].detach().item(),
            'std_ratio_min': target_aux['std_ratio_min'].detach().item(),
            'std_ratio_max': target_aux['std_ratio_max'].detach().item(),
            'lambda_base_logit_abs_mean': target_aux['lambda_base_logit_abs_mean'].detach().item(),
            'lambda_phase_logit_abs_mean': target_aux['lambda_phase_logit_abs_mean'].detach().item(),
            'tau_mean': target_aux['tau_mean'].detach().item(),
            'tau_std': target_aux['tau_std'].detach().item(),
            'dtau_dt_mean': target_aux['dtau_dt_mean'].detach().item(),
            'dtau_dt_min': target_aux['dtau_dt_min'].detach().item(),
            'dtau_dt_max': target_aux['dtau_dt_max'].detach().item(),
            'phase_scale_mean': target_aux['phase_scale_mean'].detach().item(),
            'phase_scale_min': target_aux['phase_scale_min'].detach().item(),
            'phase_scale_max': target_aux['phase_scale_max'].detach().item(),
            'phase_shift_mean': target_aux['phase_shift_mean'].detach().item(),
            'phase_shift_abs_mean': target_aux['phase_shift_abs_mean'].detach().item(),
        }
        return loss, loss_dict
