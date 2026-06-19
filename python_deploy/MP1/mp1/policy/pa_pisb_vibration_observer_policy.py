import math
import sys

sys.path.append('mp1')

from typing import Dict

import torch
import torch.nn as nn
from termcolor import cprint

from mp1.policy.pa_pisb_nocollapse_policy import PAPISBNoCollapsePolicy


class PAPISBVibrationObserverPolicy(PAPISBNoCollapsePolicy):
    """
    Physics-inspired innovation point for future vibration/shaking scenes.

    The hidden disturbance is modeled as a damped forced oscillator with a
    Kalman-style innovation update, but the policy remains fully trainable on
    clean public datasets because the observer acts only as a small residual
    conditioner over PA-PISB features.

    Continuous-time intuition:

        q_dot = v
        v_dot = -2 * zeta * omega * v - omega^2 * q + b * u
        y = C q + noise

    We discretize a latent disturbance observer over encoded observation steps:

        q^-_{k+1} = q_k + dt * v_k
        v^-_{k+1} = v_k + dt * (-2*zeta*omega*v_k - omega^2*q_k + b*u_k)

        e_k = y_k - q^-_k
        q_k = q^-_k + K_q \odot e_k
        v_k = v^-_k + K_v \odot e_k

    The estimated vibration state [q_k, v_k] is then projected back to a
    compensation residual that modifies both the observation conditioning and the
    PA bridge condition. With small initialization and explicit energy regularization,
    this stays close to PA-PISB on clean data but provides a principled path to
    future sinusoidal shaking scenarios.
    """

    def __init__(
        self,
        shape_meta: dict,
        *args,
        vo_state_dim: int = 64,
        vo_hidden_dim: int = 128,
        vo_omega_min: float = 1.5,
        vo_omega_max: float = 8.0,
        vo_damping_min: float = 0.05,
        vo_damping_max: float = 0.60,
        vo_gain_max: float = 0.30,
        vo_drive_max: float = 0.50,
        vo_seq_comp_scale: float = 0.25,
        vo_bridge_comp_scale: float = 0.25,
        vo_energy_reg_weight: float = 1.0e-4,
        vo_residual_reg_weight: float = 1.0e-4,
        vo_innovation_reg_weight: float = 5.0e-5,
        **kwargs
    ):
        self.vo_state_dim = vo_state_dim
        self.vo_hidden_dim = vo_hidden_dim
        self.vo_omega_min = vo_omega_min
        self.vo_omega_max = vo_omega_max
        self.vo_damping_min = vo_damping_min
        self.vo_damping_max = vo_damping_max
        self.vo_gain_max = vo_gain_max
        self.vo_drive_max = vo_drive_max
        self.vo_seq_comp_scale = vo_seq_comp_scale
        self.vo_bridge_comp_scale = vo_bridge_comp_scale
        self.vo_energy_reg_weight = vo_energy_reg_weight
        self.vo_residual_reg_weight = vo_residual_reg_weight
        self.vo_innovation_reg_weight = vo_innovation_reg_weight
        self._vo_last_regs = {}
        self._vo_last_metrics = {}

        super().__init__(shape_meta, *args, **kwargs)

        summary_dim = self.obs_feature_dim * 2
        self.vo_summary_net = nn.Sequential(
            nn.Linear(summary_dim, vo_hidden_dim),
            nn.SiLU(),
            nn.Linear(vo_hidden_dim, vo_hidden_dim),
            nn.SiLU(),
        )
        self.vo_obs_proj = nn.Linear(self.obs_feature_dim, vo_state_dim)
        self.vo_omega_head = nn.Linear(vo_hidden_dim, vo_state_dim)
        self.vo_damping_head = nn.Linear(vo_hidden_dim, vo_state_dim)
        self.vo_gain_q_head = nn.Linear(vo_hidden_dim, vo_state_dim)
        self.vo_gain_v_head = nn.Linear(vo_hidden_dim, vo_state_dim)
        self.vo_drive_head = nn.Linear(vo_hidden_dim, vo_state_dim)
        self.vo_residual_proj = nn.Sequential(
            nn.Linear(2 * vo_state_dim, self.obs_feature_dim),
            nn.SiLU(),
            nn.Linear(self.obs_feature_dim, self.obs_feature_dim),
        )
        self.vo_bridge_proj = nn.Linear(2 * vo_state_dim, self.obs_feature_dim)

        nn.init.zeros_(self.vo_omega_head.weight)
        nn.init.zeros_(self.vo_omega_head.bias)
        nn.init.zeros_(self.vo_damping_head.weight)
        nn.init.zeros_(self.vo_damping_head.bias)
        nn.init.zeros_(self.vo_gain_q_head.weight)
        nn.init.zeros_(self.vo_gain_q_head.bias)
        nn.init.zeros_(self.vo_gain_v_head.weight)
        nn.init.zeros_(self.vo_gain_v_head.bias)
        nn.init.zeros_(self.vo_drive_head.weight)
        nn.init.zeros_(self.vo_drive_head.bias)

        for module in [self.vo_obs_proj, self.vo_bridge_proj]:
            nn.init.normal_(module.weight, mean=0.0, std=1.0e-3)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        for module in self.vo_residual_proj:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=1.0e-3)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        cprint(
            "[PA-PISB-VibrationObserver] "
            f"state_dim={vo_state_dim}, hidden_dim={vo_hidden_dim}, "
            f"omega=[{vo_omega_min}, {vo_omega_max}], damping=[{vo_damping_min}, {vo_damping_max}], "
            f"gain_max={vo_gain_max}, drive_max={vo_drive_max}, "
            f"seq_scale={vo_seq_comp_scale}, bridge_scale={vo_bridge_comp_scale}",
            "cyan",
        )

    def _compute_vo_summary(self, seq_features: torch.Tensor) -> torch.Tensor:
        seq_mean = seq_features.mean(dim=1)
        if seq_features.shape[1] > 1:
            seq_diff = seq_features[:, 1:, :] - seq_features[:, :-1, :]
            diff_mean = seq_diff.mean(dim=1)
        else:
            diff_mean = torch.zeros_like(seq_mean)
        return torch.cat([seq_mean, diff_mean], dim=-1)

    def _bounded_param(self, logits: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
        return lower + (upper - lower) * torch.sigmoid(logits)

    def _run_vibration_observer(self, seq_features: torch.Tensor):
        batch_size, steps, _ = seq_features.shape
        device = seq_features.device
        dtype = seq_features.dtype
        dt = 1.0 / max(steps - 1, 1)

        summary = self._compute_vo_summary(seq_features)
        h = self.vo_summary_net(summary)

        omega = self._bounded_param(self.vo_omega_head(h), self.vo_omega_min, self.vo_omega_max)
        damping = self._bounded_param(self.vo_damping_head(h), self.vo_damping_min, self.vo_damping_max)
        gain_q = self.vo_gain_max * torch.sigmoid(self.vo_gain_q_head(h))
        gain_v = self.vo_gain_max * torch.sigmoid(self.vo_gain_v_head(h))
        drive = self.vo_drive_max * torch.tanh(self.vo_drive_head(h))

        obs_latent = self.vo_obs_proj(seq_features.reshape(-1, self.obs_feature_dim))
        obs_latent = obs_latent.reshape(batch_size, steps, self.vo_state_dim)

        q = torch.zeros((batch_size, self.vo_state_dim), device=device, dtype=dtype)
        v = torch.zeros_like(q)
        prev_obs = torch.zeros_like(q)

        q_list = []
        v_list = []
        innovation_list = []

        for step_idx in range(steps):
            if step_idx > 0:
                q_pred = q + dt * v
                restoring = -(omega ** 2) * q
                damping_force = -2.0 * damping * omega * v
                excitation = drive * prev_obs
                v_pred = v + dt * (restoring + damping_force + excitation)
            else:
                q_pred = q
                v_pred = v

            innovation = obs_latent[:, step_idx] - q_pred
            q = q_pred + gain_q * innovation
            v = v_pred + gain_v * innovation

            q_list.append(q)
            v_list.append(v)
            innovation_list.append(innovation)
            prev_obs = obs_latent[:, step_idx]

        q_seq = torch.stack(q_list, dim=1)
        v_seq = torch.stack(v_list, dim=1)
        innovation_seq = torch.stack(innovation_list, dim=1)
        state_seq = torch.cat([q_seq, v_seq], dim=-1)

        residual = self.vo_residual_proj(state_seq.reshape(-1, 2 * self.vo_state_dim))
        residual = residual.reshape(batch_size, steps, self.obs_feature_dim)

        pooled_state = torch.cat([q_seq.mean(dim=1), v_seq.mean(dim=1)], dim=-1)
        bridge_residual = self.vo_bridge_proj(pooled_state)

        seq_features_mod = seq_features - self.vo_seq_comp_scale * residual
        pooled_features_mod = seq_features_mod.mean(dim=1) - self.vo_bridge_comp_scale * bridge_residual

        self._vo_last_regs = {
            'vo_energy_reg': q_seq.pow(2).mean() + 0.25 * v_seq.pow(2).mean(),
            'vo_residual_reg': residual.pow(2).mean(),
            'vo_innovation_reg': innovation_seq.pow(2).mean(),
        }
        self._vo_last_metrics = {
            'vo_omega_mean': omega.mean().detach(),
            'vo_damping_mean': damping.mean().detach(),
            'vo_gain_q_mean': gain_q.mean().detach(),
            'vo_gain_v_mean': gain_v.mean().detach(),
            'vo_drive_abs_mean': drive.abs().mean().detach(),
            'vo_q_abs_mean': q_seq.abs().mean().detach(),
            'vo_v_abs_mean': v_seq.abs().mean().detach(),
            'vo_innovation_abs_mean': innovation_seq.abs().mean().detach(),
            'vo_residual_abs_mean': residual.abs().mean().detach(),
            'vo_bridge_residual_abs_mean': bridge_residual.abs().mean().detach(),
        }
        return seq_features_mod, pooled_features_mod

    def _encode_obs(self, nobs, batch_size):
        this_nobs = {k: v[:, :self.n_obs_steps, ...].reshape(-1, *v.shape[2:]) for k, v in nobs.items()}
        nobs_features = self.obs_encoder(this_nobs)
        seq_features = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
        seq_features, pooled_features = self._run_vibration_observer(seq_features)

        if self.obs_as_global_cond:
            if "cross_attention" in self.condition_type:
                global_cond = seq_features
            else:
                global_cond = seq_features.reshape(batch_size, -1)
        else:
            global_cond = None
        return seq_features, pooled_features, global_cond

    def compute_loss(self, batch):
        loss, loss_dict = super().compute_loss(batch)
        if self._vo_last_regs:
            vo_reg = (
                self.vo_energy_reg_weight * self._vo_last_regs['vo_energy_reg']
                + self.vo_residual_reg_weight * self._vo_last_regs['vo_residual_reg']
                + self.vo_innovation_reg_weight * self._vo_last_regs['vo_innovation_reg']
            )
            loss = loss + vo_reg
            loss_dict['loss'] = loss.item()
            loss_dict['bc_loss'] = loss.item()
            loss_dict['vo_reg'] = vo_reg.item()
            loss_dict['vo_energy_reg'] = self._vo_last_regs['vo_energy_reg'].item()
            loss_dict['vo_residual_reg'] = self._vo_last_regs['vo_residual_reg'].item()
            loss_dict['vo_innovation_reg'] = self._vo_last_regs['vo_innovation_reg'].item()
            for key, value in self._vo_last_metrics.items():
                loss_dict[key] = value.item()
        return loss, loss_dict
