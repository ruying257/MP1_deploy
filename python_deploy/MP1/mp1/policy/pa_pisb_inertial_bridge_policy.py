import sys

sys.path.append('mp1')

from typing import Dict

import torch
import torch.nn as nn
from termcolor import cprint

from mp1.policy.pa_pisb_nocollapse_policy import (
    PAPISBNoCollapsePolicy,
    NoCollapsePAPISBBridge,
)


class PAPISBInertialBridgePolicy(PAPISBNoCollapsePolicy):
    """
    Clean-first PA-PISB variant with an inertial bridge summary.

    Motivation:
    the original PA-PISB bridge only sees mean(seq_features), which discards
    temporal derivatives. This policy augments the bridge condition with
    feature-space velocity and acceleration summaries:

        c0 = mean(f_i)
        c1 = mean(f_{i+1} - f_i)
        c2 = mean(f_{i+2} - 2 f_{i+1} + f_i)

        c_inertial = [c0, c1, c2]

    The bridge then becomes:

        lambda_t(c_inertial) = 1 + a * tanh(g0(c_inertial) + t * g1(c_inertial)).

    A zero-init polynomial residual is also injected back into the observation
    sequence so the UNet can exploit the same inertial cues, while training still
    starts exactly from PA-PISB.

    Observation encoding is inherited from PAPISBNoCollapsePolicy. This means
    encoder-related kwargs such as `obs_encoder_kind`, image encoder settings,
    and point-cloud encoder settings are resolved by the parent policy and are
    therefore shared by both the point-cloud-only and real-multimodal variants.
    """

    def __init__(
        self,
        shape_meta: dict,
        *args,
        ib_hidden_dim: int = 128,
        ib_seq_residual_scale: float = 0.20,
        ib_bridge_residual_scale: float = 0.20,
        ib_reg_weight: float = 1.0e-4,
        **kwargs
    ):
        self.ib_hidden_dim = ib_hidden_dim
        self.ib_seq_residual_scale = ib_seq_residual_scale
        self.ib_bridge_residual_scale = ib_bridge_residual_scale
        self.ib_reg_weight = ib_reg_weight
        self._ib_last_reg = None
        self._ib_last_metrics = {}

        bridge_sigma = kwargs.get('bridge_sigma', 0.1)
        bridge_stiffness = kwargs.get('bridge_stiffness', 0.1)
        bridge_damping = kwargs.get('bridge_damping', 0.1)
        pa_lambda_amplitude = kwargs.get('pa_lambda_amplitude', 0.35)
        pa_head_hidden_dim = kwargs.get('pa_head_hidden_dim', 256)
        pa_head_dropout = kwargs.get('pa_head_dropout', 0.0)

        # Encoder construction is delegated to the parent class so this policy
        # stays compatible with both the legacy point-cloud pipeline and the new
        # real-multimodal pipeline.
        super().__init__(shape_meta, *args, **kwargs)

        summary_dim = self.obs_feature_dim * 3
        self.ib_feature_net = nn.Sequential(
            nn.Linear(summary_dim, ib_hidden_dim),
            nn.SiLU(),
            nn.Linear(ib_hidden_dim, ib_hidden_dim),
            nn.SiLU(),
        )
        self.ib_seq_head = nn.Linear(ib_hidden_dim, self.obs_feature_dim * 3)
        self.ib_bridge_head = nn.Linear(ib_hidden_dim, self.obs_feature_dim)

        nn.init.zeros_(self.ib_seq_head.weight)
        nn.init.zeros_(self.ib_seq_head.bias)
        nn.init.zeros_(self.ib_bridge_head.weight)
        nn.init.zeros_(self.ib_bridge_head.bias)

        # Replace the original bridge with one that sees static + velocity + acceleration summary.
        self.physics_bridge = NoCollapsePAPISBBridge(
            cond_dim=summary_dim,
            traj_dim=self.horizon * self.action_dim,
            sigma_min=1e-4,
            sigma_max=bridge_sigma,
            stiffness=bridge_stiffness,
            damping=bridge_damping,
            lambda_amplitude=pa_lambda_amplitude,
            hidden_dim=pa_head_hidden_dim,
            head_dropout=pa_head_dropout,
            device=self.device,
        )

        cprint(
            "[PA-PISB-InertialBridge] "
            f"hidden_dim={ib_hidden_dim}, seq_scale={ib_seq_residual_scale}, "
            f"bridge_scale={ib_bridge_residual_scale}, reg={ib_reg_weight}",
            "cyan",
        )

    def _compute_inertial_summary(self, seq_features: torch.Tensor):
        seq_mean = seq_features.mean(dim=1)

        if seq_features.shape[1] > 1:
            vel = seq_features[:, 1:, :] - seq_features[:, :-1, :]
            vel_mean = vel.mean(dim=1)
        else:
            vel_mean = torch.zeros_like(seq_mean)

        if seq_features.shape[1] > 2:
            acc = seq_features[:, 2:, :] - 2.0 * seq_features[:, 1:-1, :] + seq_features[:, :-2, :]
            acc_mean = acc.mean(dim=1)
        else:
            acc_mean = torch.zeros_like(seq_mean)

        summary = torch.cat([seq_mean, vel_mean, acc_mean], dim=-1)
        return seq_mean, vel_mean, acc_mean, summary

    def _apply_inertial_bridge(self, seq_features: torch.Tensor):
        batch_size, _, feat_dim = seq_features.shape
        device = seq_features.device
        dtype = seq_features.dtype

        seq_mean, vel_mean, acc_mean, summary = self._compute_inertial_summary(seq_features)
        h = self.ib_feature_net(summary)

        basis_coeff = self.ib_seq_head(h).reshape(batch_size, 3, feat_dim)
        if self.n_obs_steps > 1:
            tau = torch.linspace(0.0, 1.0, steps=self.n_obs_steps, device=device, dtype=dtype)
        else:
            tau = torch.zeros((1,), device=device, dtype=dtype)
        basis = torch.stack([torch.ones_like(tau), tau, tau * tau], dim=-1)  # (To, 3)
        residual = torch.einsum('bkd,tk->btd', basis_coeff, basis)

        seq_features_mod = seq_features + self.ib_seq_residual_scale * residual
        pooled_static = seq_features_mod.mean(dim=1) + self.ib_bridge_residual_scale * self.ib_bridge_head(h)
        bridge_cond = torch.cat([pooled_static, vel_mean, acc_mean], dim=-1)

        self._ib_last_reg = residual.pow(2).mean() + 0.5 * vel_mean.pow(2).mean() + 0.25 * acc_mean.pow(2).mean()
        self._ib_last_metrics = {
            'ib_vel_abs_mean': vel_mean.abs().mean().detach(),
            'ib_acc_abs_mean': acc_mean.abs().mean().detach(),
            'ib_residual_abs_mean': residual.abs().mean().detach(),
            'ib_bridge_residual_abs_mean': self.ib_bridge_head(h).abs().mean().detach(),
        }
        return seq_features_mod, bridge_cond

    def _encode_obs(self, nobs, batch_size):
        this_nobs = {
            k: v[:, :self.n_obs_steps, ...].reshape(-1, *v.shape[2:])
            for k, v in nobs.items()
        }
        nobs_features = self.obs_encoder(this_nobs)
        seq_features = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
        seq_features, bridge_cond = self._apply_inertial_bridge(seq_features)

        if self.obs_as_global_cond:
            if "cross_attention" in self.condition_type:
                global_cond = seq_features
            else:
                global_cond = seq_features.reshape(batch_size, -1)
        else:
            global_cond = None
        return seq_features, bridge_cond, global_cond

    def compute_loss(self, batch):
        loss, loss_dict = super().compute_loss(batch)
        if self._ib_last_reg is not None:
            ib_reg = self.ib_reg_weight * self._ib_last_reg
            loss = loss + ib_reg
            loss_dict['loss'] = loss.item()
            loss_dict['bc_loss'] = loss.item()
            loss_dict['ib_reg'] = ib_reg.item()
        return loss, loss_dict
