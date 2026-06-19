import sys
sys.path.append('mp1')
from typing import Dict
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint
import warnings
from mp1.model.common.normalizer import LinearNormalizer
from mp1.policy.base_policy import BasePolicy
from mp1.model.mean.conditional_unet1d_meanflow_dis import ConditionalUnet1D
from mp1.model.mean.mask_generator import LowdimMaskGenerator
from mp1.common.pytorch_util import dict_apply
from mp1.common.model_util import print_params
from mp1.model.vision.obs_encoder_factory import build_obs_encoder

warnings.filterwarnings("ignore")


class NoCollapsePAPISBBridge(nn.Module):
    """
    Phase-Anisotropic PISB with bounded, centered precision.

    Core formula:
        mu_t = alpha(t) x0 + beta(t) x1
        rho(t) = sigma_max * sqrt(t(1-t)) * exp(-k t)
        lambda_t(c) = 1 + a * tanh(g0(c) + t * g1(c))
        Sigma_t(c) = rho(t)^2 * Diag(lambda_t(c))^{-1}
        x_t = mu_t + rho(t) * lambda_t(c)^(-1/2) * eps

    Thus:
        u_tgt = d mu_t / dt
                + ( d rho / rho - 0.5 * dot_lambda_t / lambda_t ) * (x_t - mu_t)

    Why this avoids collapse:
        - lambda_t is centered at 1, so initialization can exactly recover isotropic PISB.
        - lambda_t is bounded in [1-a, 1+a], so the bridge cannot shrink or blow up globally.
        - zero-init on g0/g1 heads makes training start from the strong PISB baseline.
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

        # Exact isotropic-PISB initialization: lambda_t(c) == 1 at step 0.
        nn.init.zeros_(self.lambda_base_head.weight)
        nn.init.zeros_(self.lambda_base_head.bias)
        nn.init.zeros_(self.lambda_phase_head.weight)
        nn.init.zeros_(self.lambda_phase_head.bias)

    def compute_scalar_coefficients(self, t: torch.Tensor):
        target_device = t.device if isinstance(t, torch.Tensor) else self.device
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=target_device)
        else:
            t = t.to(target_device)

        gamma = self.damping
        denom = math.sinh(gamma) if abs(gamma) > 1e-6 else 1.0

        alpha_t = torch.sinh(gamma * (1.0 - t)) / (denom + 1e-6)
        beta_t = torch.sinh(gamma * t) / (denom + 1e-6)

        rho_t = self.sigma_max * torch.sqrt(torch.clamp(t * (1.0 - t), min=1e-12)) * torch.exp(-self.stiffness * t)
        rho_t = torch.clamp(rho_t, min=self.sigma_min)
        return alpha_t, beta_t, rho_t

    def _compute_lambda_terms(self, cond_feat: torch.Tensor, t_expand: torch.Tensor):
        """
        cond_feat: (B, C)
        t_expand:  (B, 1, 1)
        returns:
            lambda_t:     (B, 1, D)
            dot_lambda_t: (B, 1, D)
            metrics:      dict
        """
        h = self.feature_net(cond_feat)
        base_logits = self.lambda_base_head(h)
        phase_logits = self.lambda_phase_head(h)

        # bounded, centered precision around 1
        phase_argument = base_logits + t_expand.squeeze(-1) * phase_logits
        tanh_phase = torch.tanh(phase_argument)
        lambda_t = 1.0 + self.lambda_amplitude * tanh_phase
        dot_lambda_t = self.lambda_amplitude * (1.0 - tanh_phase.pow(2)) * phase_logits

        lambda_t = lambda_t.unsqueeze(1)      # (B,1,D)
        dot_lambda_t = dot_lambda_t.unsqueeze(1)

        # metrics for collapse diagnosis
        lambda_mean = lambda_t.mean()
        lambda_std = lambda_t.std(unbiased=False)
        lambda_min = lambda_t.min()
        lambda_max = lambda_t.max()
        delta_lambda_mean = dot_lambda_t.mean()
        delta_lambda_abs_mean = dot_lambda_t.abs().mean()
        # anisotropy across dims (if this is near zero, no anisotropy learned)
        lambda_dim_std = lambda_t.std(dim=-1, unbiased=False).mean()
        # effective std ratio relative to isotropic PISB: std_new / std_old = 1 / sqrt(lambda)
        std_ratio = torch.rsqrt(lambda_t)
        std_ratio_mean = std_ratio.mean()
        std_ratio_min = std_ratio.min()
        std_ratio_max = std_ratio.max()

        metrics = {
            'lambda_mean': lambda_mean,
            'lambda_std': lambda_std,
            'lambda_min': lambda_min,
            'lambda_max': lambda_max,
            'delta_lambda_mean': delta_lambda_mean,
            'delta_lambda_abs_mean': delta_lambda_abs_mean,
            'lambda_dim_std': lambda_dim_std,
            'std_ratio_mean': std_ratio_mean,
            'std_ratio_min': std_ratio_min,
            'std_ratio_max': std_ratio_max,
            # raw diagnostics
            'lambda_base_logit_abs_mean': base_logits.abs().mean(),
            'lambda_phase_logit_abs_mean': phase_logits.abs().mean(),
        }
        return lambda_t, dot_lambda_t, metrics

    def sample_path(self, x0, x1, t_expand, cond_feat):
        """
        x0, x1:    (B, T, D)
        t_expand:  (B, 1, 1)
        cond_feat: (B, C)
        """
        alpha, beta, rho = self.compute_scalar_coefficients(t_expand)
        lambda_t, dot_lambda_t, metrics = self._compute_lambda_terms(cond_feat, t_expand)

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
            **metrics,
        }
        return xt, aux

    def compute_target_velocity(self, x0, x1, t_expand, xt, cond_feat):
        t_in = t_expand.clone().detach().requires_grad_(True)
        alpha, beta, rho = self.compute_scalar_coefficients(t_in)
        lambda_t, dot_lambda_t, metrics = self._compute_lambda_terms(cond_feat, t_in)

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
            **{k: v.detach() if torch.is_tensor(v) else v for k, v in metrics.items()},
        }
        return target_v.detach(), aux

    def regularization(self, aux: dict, center_weight: float = 1e-3, phase_weight: float = 1e-3, aniso_floor_weight: float = 0.0, aniso_floor: float = 0.02):
        """
        Deterministic regularizer matched to the new formula.
        - center term keeps lambda_t near 1, so PA-PISB stays a small perturbation of PISB.
        - phase term keeps dot_lambda_t bounded.
        - optional anisotropy-floor term can discourage complete constant collapse, but default off.
        """
        lambda_t = aux['lambda_t']
        dot_lambda_t = aux['dot_lambda_t']
        center_reg = (lambda_t - 1.0).pow(2).mean()
        phase_reg = dot_lambda_t.pow(2).mean()
        reg = center_weight * center_reg + phase_weight * phase_reg

        if aniso_floor_weight > 0:
            dim_std = lambda_t.std(dim=-1, unbiased=False)
            floor_penalty = F.relu(aniso_floor - dim_std).pow(2).mean()
            reg = reg + aniso_floor_weight * floor_penalty
        return reg


class PAPISBNoCollapsePolicy(BasePolicy):
    def __init__(self,
            shape_meta: dict,
            horizon,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256, 512, 1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            obs_encoder_kind="pointcloud",
            image_encoder_output_dim=64,
            image_base_channels=32,
            share_image_encoder=False,
            # original PISB scalars
            bridge_stiffness=0.1,
            bridge_damping=0.1,
            bridge_sigma=0.1,
            # no-collapse PA-PISB params
            pa_lambda_amplitude=0.35,
            pa_head_hidden_dim=256,
            pa_head_dropout=0.0,
            pa_center_reg_weight=1e-3,
            pa_phase_reg_weight=1e-3,
            pa_aniso_floor_weight=0.0,
            pa_aniso_floor=0.02,
            **kwargs):
        super().__init__()
        self.condition_type = condition_type

        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        obs_encoder = build_obs_encoder(
            observation_space=obs_dict,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
            obs_encoder_kind=obs_encoder_kind,
            image_encoder_output_dim=image_encoder_output_dim,
            image_base_channels=image_base_channels,
            share_image_encoder=share_image_encoder,
        )

        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        self.num_inference_steps = num_inference_steps
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        self.pa_center_reg_weight = pa_center_reg_weight
        self.pa_phase_reg_weight = pa_phase_reg_weight
        self.pa_aniso_floor_weight = pa_aniso_floor_weight
        self.pa_aniso_floor = pa_aniso_floor

        self.physics_bridge = NoCollapsePAPISBBridge(
            cond_dim=obs_feature_dim,
            traj_dim=horizon * action_dim,
            sigma_min=1e-4,
            sigma_max=bridge_sigma,
            stiffness=bridge_stiffness,
            damping=bridge_damping,
            lambda_amplitude=pa_lambda_amplitude,
            hidden_dim=pa_head_hidden_dim,
            head_dropout=pa_head_dropout,
            device=self.device,
        )

        cprint(f"[PA-PISB-NoCollapse] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[PA-PISB-NoCollapse] pointnet_type: {self.pointnet_type}", "yellow")
        cprint(
            f"[PA-PISB-NoCollapse] k={bridge_stiffness}, gamma={bridge_damping}, sigma={bridge_sigma}, "
            f"lambda_amp={pa_lambda_amplitude}, head_dropout={pa_head_dropout}",
            "cyan"
        )
        self._loss_profile_enabled = (
            os.getenv("MP1_PROFILE_LOSS", "0") == "1"
            or os.getenv("MP1_PROFILE_TRAIN", "0") == "1"
        )
        self._last_loss_profile = {}
        print_params(self)

    def reset(self):
        return

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    @staticmethod
    def _maybe_sync(device: torch.device, enabled: bool):
        if enabled and device.type == 'cuda':
            torch.cuda.synchronize(device)

    def _encode_obs(self, nobs, batch_size):
        this_nobs = dict_apply(nobs, lambda n: n[:, :self.n_obs_steps, ...].reshape(-1, *n.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)  # (B*To, Do)
        seq_features = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
        pooled_features = seq_features.mean(dim=1)

        if self.obs_as_global_cond:
            if "cross_attention" in self.condition_type:
                global_cond = seq_features
            else:
                global_cond = seq_features.reshape(batch_size, -1)
        else:
            global_cond = None
        return seq_features, pooled_features, global_cond

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        device = self.device
        dtype = self.dtype

        _, _, global_cond = self._encode_obs(nobs, B)
        local_cond = None
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
                    local_cond=local_cond,
                    global_cond=global_cond,
                    r=r_zeros,
                    training=False,
                )
                v_pred = model_output[0] if isinstance(model_output, tuple) else model_output
                x_current = x_current + v_pred * dt

        naction_pred = x_current[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return {'action': action, 'action_pred': action_pred}

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
        seq_features, pooled_features, global_cond = self._encode_obs(nobs, batch_size)
        if profile_enabled:
            self._maybe_sync(device, True)
            loss_profile['obs_encode'] = time.perf_counter() - encode_start

        eps = 1e-5
        t = torch.rand((batch_size,), device=device).float() * (1 - 2 * eps) + eps
        t_expand = t.view(batch_size, 1, 1)

        bridge_start = time.perf_counter() if profile_enabled else None
        xt, sample_aux = self.physics_bridge.sample_path(x0, x1, t_expand, pooled_features)
        u_tgt, target_aux = self.physics_bridge.compute_target_velocity(x0, x1, t_expand, xt, pooled_features)
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

        pa_reg = self.physics_bridge.regularization(
            target_aux,
            center_weight=self.pa_center_reg_weight,
            phase_weight=self.pa_phase_reg_weight,
            aniso_floor_weight=self.pa_aniso_floor_weight,
            aniso_floor=self.pa_aniso_floor,
        )

        loss = meanflow_loss + 0.5 * dis_loss + pa_reg
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
            'pa_reg': pa_reg.item(),
        }
        if profile_enabled:
            loss_profile['total'] = time.perf_counter() - total_start
            self._last_loss_profile = loss_profile
        else:
            self._last_loss_profile = {}
        return loss, loss_dict

    def dispersive_loss(self, z, tau=1.0):
        dist_matrix = torch.cdist(z, z, p=2) ** 2
        dist_matrix = dist_matrix / (torch.max(dist_matrix) + 1e-6)
        exp_term = torch.exp(-dist_matrix / tau)
        mean_exp = torch.mean(exp_term)
        loss = torch.log(mean_exp)
        return loss


def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    delta_sq = torch.mean(error ** 2, dim=tuple(range(1, error.ndim)))
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq
    return (stopgrad(w) * loss).mean()
