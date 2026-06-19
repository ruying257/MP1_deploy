import sys
sys.path.append('mp1')

from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint

from mp1.common.pytorch_util import dict_apply
from mp1.policy.pisb_policy import PISBPolicy, adaptive_l2_loss, stopgrad


class LogitPhaseWarpBridge(nn.Module):
    """
    Context-conditioned phase-warped PISB bridge.

    Original PISB:
        mu(t) = alpha(t) x0 + beta(t) x1
        rho(t) = sigma_max * sqrt(t(1-t)) * exp(-k t)
        x_t = mu(t) + rho(t) * eps

    Proposed phase-warped bridge:
        tau_c(t) = sigmoid(a(c) * logit(t) + b(c))
        mu_c(t) = alpha(tau_c(t)) x0 + beta(tau_c(t)) x1
        rho_c(t) = sigma_max * sqrt(tau_c(t)(1-tau_c(t))) * exp(-k tau_c(t))
        x_t = mu_c(t) + rho_c(t) * eps

    where
        a(c) = exp(eta_a * tanh(g_a(c))) > 0
        b(c) = eta_b * tanh(g_b(c))

    This is a bounded, monotone, identity-initialized perturbation of PISB:
    - zero-init gives a(c)=1 and b(c)=0, so tau_c(t)=t exactly.
    - positivity of a(c) keeps tau_c(t) monotone in t.
    - bounded eta_a / eta_b keep the warp small and stable.
    """

    def __init__(
        self,
        cond_dim: int,
        sigma_min: float = 1.0e-4,
        sigma_max: float = 0.1,
        stiffness: float = 1.5,
        damping: float = 0.5,
        hidden_dim: int = 128,
        head_dropout: float = 0.0,
        phase_scale_amplitude: float = 0.25,
        phase_shift_amplitude: float = 0.25,
        time_eps: float = 1.0e-5,
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.stiffness = stiffness
        self.damping = damping
        self.phase_scale_amplitude = phase_scale_amplitude
        self.phase_shift_amplitude = phase_shift_amplitude
        self.time_eps = time_eps

        self.feature_net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(head_dropout),
        )
        self.scale_head = nn.Linear(hidden_dim, 1)
        self.shift_head = nn.Linear(hidden_dim, 1)

        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)
        nn.init.zeros_(self.shift_head.weight)
        nn.init.zeros_(self.shift_head.bias)

    def compute_scalar_coefficients(self, tau: torch.Tensor):
        gamma = self.damping
        denom = math.sinh(gamma) if abs(gamma) > 1.0e-6 else 1.0

        alpha_t = torch.sinh(gamma * (1.0 - tau)) / (denom + 1.0e-6)
        beta_t = torch.sinh(gamma * tau) / (denom + 1.0e-6)
        rho_t = self.sigma_max * torch.sqrt(torch.clamp(tau * (1.0 - tau), min=1.0e-12))
        rho_t = rho_t * torch.exp(-self.stiffness * tau)
        rho_t = torch.clamp(rho_t, min=self.sigma_min)
        return alpha_t, beta_t, rho_t

    def _compute_phase_warp(self, cond_feat: torch.Tensor, t_expand: torch.Tensor):
        safe_t = torch.clamp(t_expand, min=self.time_eps, max=1.0 - self.time_eps)
        logit_t = torch.log(safe_t) - torch.log1p(-safe_t)

        h = self.feature_net(cond_feat)
        scale_logits = self.scale_head(h)
        shift_logits = self.shift_head(h)

        log_scale = self.phase_scale_amplitude * torch.tanh(scale_logits)
        phase_scale = torch.exp(log_scale).unsqueeze(1)
        phase_shift = (
            self.phase_shift_amplitude * torch.tanh(shift_logits)
        ).unsqueeze(1)

        tau_t = torch.sigmoid(phase_scale * logit_t + phase_shift)
        tau_t = torch.clamp(tau_t, min=self.time_eps, max=1.0 - self.time_eps)
        dtau_dt = phase_scale * tau_t * (1.0 - tau_t) / (
            safe_t * (1.0 - safe_t) + 1.0e-6
        )

        metrics = {
            'phase_scale_mean': phase_scale.mean(),
            'phase_scale_min': phase_scale.min(),
            'phase_scale_max': phase_scale.max(),
            'phase_shift_mean': phase_shift.mean(),
            'phase_shift_abs_mean': phase_shift.abs().mean(),
            'tau_mean': tau_t.mean(),
            'tau_std': tau_t.std(unbiased=False),
            'dtau_dt_mean': dtau_dt.mean(),
            'dtau_dt_min': dtau_dt.min(),
            'dtau_dt_max': dtau_dt.max(),
        }
        return tau_t, dtau_dt, phase_scale, phase_shift, metrics

    def sample_path(self, x0, x1, t_expand, cond_feat):
        tau_t, dtau_dt, phase_scale, phase_shift, metrics = self._compute_phase_warp(
            cond_feat, t_expand
        )
        alpha_t, beta_t, rho_t = self.compute_scalar_coefficients(tau_t)

        mu_t = alpha_t * x0 + beta_t * x1
        eps = torch.randn_like(x0)
        xt = mu_t + rho_t * eps

        aux = {
            'tau_t': tau_t,
            'dtau_dt': dtau_dt,
            'phase_scale': phase_scale,
            'phase_shift': phase_shift,
            'mu_t': mu_t,
            'rho_t': rho_t,
            **metrics,
        }
        return xt, aux

    def compute_target_velocity(self, x0, x1, t_expand, xt, cond_feat):
        t_in = t_expand.clone().detach().requires_grad_(True)
        tau_t, dtau_dt, phase_scale, phase_shift, metrics = self._compute_phase_warp(
            cond_feat, t_in
        )
        alpha_t, beta_t, rho_t = self.compute_scalar_coefficients(tau_t)
        mu_t = alpha_t * x0 + beta_t * x1

        d_alpha_dt = torch.autograd.grad(alpha_t.sum(), t_in, create_graph=True)[0]
        d_beta_dt = torch.autograd.grad(beta_t.sum(), t_in, create_graph=True)[0]
        d_rho_dt = torch.autograd.grad(rho_t.sum(), t_in, create_graph=True)[0]

        dt_mu = d_alpha_dt * x0 + d_beta_dt * x1
        drift_coeff = d_rho_dt / (rho_t + 1.0e-6)
        target_v = dt_mu + drift_coeff * (xt - mu_t)

        aux = {
            'tau_t': tau_t,
            'dtau_dt': dtau_dt,
            'phase_scale': phase_scale,
            'phase_shift': phase_shift,
            'mu_t': mu_t,
            'rho_t': rho_t,
            'drift_coeff': drift_coeff,
            **metrics,
        }
        return target_v, aux

    def regularization(
        self,
        aux: dict,
        identity_weight: float = 1.0e-3,
        slope_weight: float = 1.0e-4,
    ):
        phase_scale = aux['phase_scale']
        phase_shift = aux['phase_shift']
        dtau_dt = aux['dtau_dt']

        identity_reg = (phase_scale - 1.0).pow(2).mean() + phase_shift.pow(2).mean()
        slope_reg = (dtau_dt - 1.0).pow(2).mean()
        return identity_weight * identity_reg + slope_weight * slope_reg


class PISBPhaseWarpPolicy(PISBPolicy):
    """
    Theoretical bridge-level innovation:
    replace fixed bridge time t by a context-conditioned monotone phase warp tau_c(t).
    """

    def __init__(
        self,
        *args,
        pw_hidden_dim: int = 128,
        pw_head_dropout: float = 0.0,
        pw_phase_scale_amplitude: float = 0.25,
        pw_phase_shift_amplitude: float = 0.25,
        pw_identity_reg_weight: float = 1.0e-3,
        pw_slope_reg_weight: float = 1.0e-4,
        pw_time_eps: float = 1.0e-5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if not self.obs_as_global_cond:
            raise NotImplementedError("PISBPhaseWarpPolicy requires obs_as_global_cond=True")

        self.pw_identity_reg_weight = pw_identity_reg_weight
        self.pw_slope_reg_weight = pw_slope_reg_weight
        self.pw_time_eps = pw_time_eps

        self.physics_bridge = LogitPhaseWarpBridge(
            cond_dim=self.obs_feature_dim,
            sigma_min=1.0e-4,
            sigma_max=kwargs.get('bridge_sigma', 0.1),
            stiffness=kwargs.get('bridge_stiffness', 1.5),
            damping=kwargs.get('bridge_damping', 0.5),
            hidden_dim=pw_hidden_dim,
            head_dropout=pw_head_dropout,
            phase_scale_amplitude=pw_phase_scale_amplitude,
            phase_shift_amplitude=pw_phase_shift_amplitude,
            time_eps=pw_time_eps,
        )

        cprint(
            "[PISB-PhaseWarp] "
            f"scale_amp={pw_phase_scale_amplitude}, shift_amp={pw_phase_shift_amplitude}, "
            f"id_reg={pw_identity_reg_weight}, slope_reg={pw_slope_reg_weight}",
            "cyan",
        )

    def _encode_obs(self, nobs: Dict[str, torch.Tensor], batch_size: int):
        this_nobs = dict_apply(
            nobs,
            lambda n: n[:, :self.n_obs_steps, ...].reshape(-1, *n.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)
        seq_features = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
        pooled_features = seq_features.mean(dim=1)

        if "cross_attention" in self.condition_type:
            global_cond = seq_features
        else:
            global_cond = seq_features.reshape(batch_size, -1)
        return pooled_features, global_cond

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color and 'point_cloud' in nobs:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        batch_size = nactions.shape[0]
        device = nactions.device

        x1 = nactions
        x0 = torch.randn_like(x1)

        pooled_features, global_cond = self._encode_obs(nobs, batch_size)

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

        pw_reg = self.physics_bridge.regularization(
            target_aux,
            identity_weight=self.pw_identity_reg_weight,
            slope_weight=self.pw_slope_reg_weight,
        )

        loss = meanflow_loss + 0.5 * dis_loss + pw_reg
        mse_val = (stopgrad(error) ** 2).mean()

        loss_dict = {
            'loss': loss.item(),
            'bc_loss': loss.item(),
            'mse_val': mse_val.item(),
            'meanflow_loss': meanflow_loss.item(),
            'dis_loss': dis_loss.item() if isinstance(dis_loss, torch.Tensor) else dis_loss,
            'pw_reg': pw_reg.item(),
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
