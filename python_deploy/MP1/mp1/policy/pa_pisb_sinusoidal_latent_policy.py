import math
import sys

sys.path.append('mp1')

from typing import Dict

import torch
import torch.nn as nn
from termcolor import cprint

from mp1.policy.pa_pisb_nocollapse_policy import PAPISBNoCollapsePolicy


class PAPISBSinusoidalLatentPolicy(PAPISBNoCollapsePolicy):
    """
    Clean-dataset-compatible innovation point inspired by future sinusoidal shaking.

    The model does not require any shake labels or synthetic shake augmentation.
    Instead, it injects a bounded sinusoidal disturbance prior into the observation
    conditioning pathway:

        h(c) = MLP([mean(seq_feat), mean(diff(seq_feat))])
        a(c) = a_max * tanh(W_a h(c))
        phi(c) = pi * tanh(W_phi h(c))
        r_i(c) = a(c) * sin(2 pi f tau_i + phi(c))

    where tau_i indexes the observation step. The residual is then added to the
    encoded observation sequence and pooled bridge condition with zero-init heads,
    so training starts exactly from PA-PISB.

    Intuition:
    - On clean public datasets, the sinusoidal latent can stay near zero, so the
      policy trains normally and remains comparable to the clean baseline.
    - On future observation-level sinusoidal disturbance data, the same module is
      already structured to represent periodic platform motion.
    """

    def __init__(
        self,
        shape_meta: dict,
        *args,
        sl_hidden_dim: int = 128,
        sl_amplitude_max: float = 0.15,
        sl_seq_residual_scale: float = 0.20,
        sl_bridge_residual_scale: float = 0.20,
        sl_frequency: float = 1.0,
        sl_reg_weight: float = 1.0e-4,
        **kwargs
    ):
        self.sl_hidden_dim = sl_hidden_dim
        self.sl_amplitude_max = sl_amplitude_max
        self.sl_seq_residual_scale = sl_seq_residual_scale
        self.sl_bridge_residual_scale = sl_bridge_residual_scale
        self.sl_frequency = sl_frequency
        self.sl_reg_weight = sl_reg_weight
        self._sl_last_reg = None
        self._sl_last_metrics = {}

        super().__init__(shape_meta, *args, **kwargs)

        summary_dim = self.obs_feature_dim * 2
        self.sl_feature_net = nn.Sequential(
            nn.Linear(summary_dim, sl_hidden_dim),
            nn.SiLU(),
            nn.Linear(sl_hidden_dim, sl_hidden_dim),
            nn.SiLU(),
        )
        self.sl_amp_head = nn.Linear(sl_hidden_dim, self.obs_feature_dim)
        self.sl_phase_head = nn.Linear(sl_hidden_dim, self.obs_feature_dim)

        nn.init.zeros_(self.sl_amp_head.weight)
        nn.init.zeros_(self.sl_amp_head.bias)
        nn.init.zeros_(self.sl_phase_head.weight)
        nn.init.zeros_(self.sl_phase_head.bias)

        cprint(
            "[PA-PISB-SinusoidalLatent] "
            f"hidden_dim={sl_hidden_dim}, amp_max={sl_amplitude_max}, "
            f"seq_scale={sl_seq_residual_scale}, bridge_scale={sl_bridge_residual_scale}, "
            f"freq={sl_frequency}, reg={sl_reg_weight}",
            "cyan",
        )

    def _compute_sl_summary(self, seq_features: torch.Tensor) -> torch.Tensor:
        seq_mean = seq_features.mean(dim=1)
        if seq_features.shape[1] > 1:
            seq_diff = seq_features[:, 1:, :] - seq_features[:, :-1, :]
            diff_mean = seq_diff.mean(dim=1)
        else:
            diff_mean = torch.zeros_like(seq_mean)
        return torch.cat([seq_mean, diff_mean], dim=-1)

    def _apply_sinusoidal_latent(self, seq_features: torch.Tensor):
        batch_size = seq_features.shape[0]
        device = seq_features.device
        dtype = seq_features.dtype

        summary = self._compute_sl_summary(seq_features)
        h = self.sl_feature_net(summary)

        amplitude = self.sl_amplitude_max * torch.tanh(self.sl_amp_head(h))
        phase = math.pi * torch.tanh(self.sl_phase_head(h))

        if self.n_obs_steps > 1:
            tau = torch.linspace(0.0, 1.0, steps=self.n_obs_steps, device=device, dtype=dtype)
        else:
            tau = torch.zeros((1,), device=device, dtype=dtype)
        tau = tau.view(1, self.n_obs_steps, 1)

        basis = torch.sin(2.0 * math.pi * self.sl_frequency * tau + phase.unsqueeze(1))
        sinusoidal_residual = amplitude.unsqueeze(1) * basis

        seq_features_mod = seq_features + self.sl_seq_residual_scale * sinusoidal_residual
        pooled_features_mod = seq_features_mod.mean(dim=1) + self.sl_bridge_residual_scale * sinusoidal_residual.mean(dim=1)

        self._sl_last_reg = amplitude.pow(2).mean()
        self._sl_last_metrics = {
            'sl_amp_abs_mean': amplitude.abs().mean().detach(),
            'sl_phase_abs_mean': phase.abs().mean().detach(),
            'sl_basis_abs_mean': basis.abs().mean().detach(),
            'sl_residual_abs_mean': sinusoidal_residual.abs().mean().detach(),
        }
        return seq_features_mod, pooled_features_mod

    def _encode_obs(self, nobs, batch_size):
        this_nobs = {k: v[:, :self.n_obs_steps, ...].reshape(-1, *v.shape[2:]) for k, v in nobs.items()}
        nobs_features = self.obs_encoder(this_nobs)
        seq_features = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
        seq_features, pooled_features = self._apply_sinusoidal_latent(seq_features)

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
        if self._sl_last_reg is not None:
            sl_reg = self.sl_reg_weight * self._sl_last_reg
            loss = loss + sl_reg
            loss_dict['loss'] = loss.item()
            loss_dict['bc_loss'] = loss.item()
            loss_dict['sl_reg'] = sl_reg.item()
            for key, value in self._sl_last_metrics.items():
                loss_dict[key] = value.item()
        return loss, loss_dict
