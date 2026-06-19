import sys
sys.path.append('mp1')
from typing import Dict, Optional
import math
import torch
import torch.nn as nn
import numpy as np
from termcolor import cprint

from mp1.model.common.normalizer import LinearNormalizer
from mp1.policy.base_policy import BasePolicy
from mp1.model.mean.conditional_unet1d_meanflow_dis import ConditionalUnet1D
from mp1.model.mean.mask_generator import LowdimMaskGenerator
from mp1.common.pytorch_util import dict_apply
from mp1.common.model_util import print_params
from mp1.model.vision.pointnet_extractor import MP1Encoder


def stopgrad(x: torch.Tensor) -> torch.Tensor:
    return x.detach()


def adaptive_l2_loss(error: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # simple robust L2; preserves original training objective scale reasonably well
    return torch.sqrt(error.pow(2) + eps).mean()


class PAPISBBridge(nn.Module):
    """
    PA-PISB: Phase-Anisotropic Physics-Informed Schrödinger Bridge

    Core formulas implemented exactly as described in the method note:
        mu_t = alpha(t) x0 + beta(t) x1
        rho(t) = sigma_max * sqrt(t(1-t)) * exp(-k t)
        lambda_t(c) = lambda0(c) + t * delta_lambda(c)
        Sigma_t(c) = rho(t)^2 * Lambda_t(c)^(-1)
        x_t = mu_t + rho(t) * Lambda_t(c)^(-1/2) * eps
        u_tgt = mu_dot + [rho_dot/rho - 0.5 * delta_lambda/lambda_t] \odot (x_t - mu_t)
    """
    def __init__(
        self,
        cond_dim: int,
        trajectory_dim: int,
        sigma_min: float = 1e-4,
        sigma_max: float = 0.1,
        stiffness: float = 1.5,
        damping: float = 0.5,
        lambda_min: float = 0.1,
        lambda_max: float = 10.0,
        delta_lambda_min: float = 0.0,
        delta_lambda_max: float = 5.0,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.stiffness = float(stiffness)
        self.damping = float(damping)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.delta_lambda_min = float(delta_lambda_min)
        self.delta_lambda_max = float(delta_lambda_max)
        self.trajectory_dim = int(trajectory_dim)

        self.lambda0_head = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, trajectory_dim),
        )
        self.delta_lambda_head = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, trajectory_dim),
        )

        # near-isotropic initialization: lambda0 ~ 1, delta_lambda ~ 0
        nn.init.zeros_(self.lambda0_head[-1].weight)
        nn.init.zeros_(self.lambda0_head[-1].bias)
        nn.init.zeros_(self.delta_lambda_head[-1].weight)
        nn.init.zeros_(self.delta_lambda_head[-1].bias)

    def _broadcast_t(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.view(-1, 1)
        while t.ndim < x.ndim:
            t = t.unsqueeze(-1)
        return t

    def _trajectory_shape(self, x: torch.Tensor):
        return x.shape[0], x.shape[1], x.shape[2]

    def _flat_cond_to_precision(self, cond: torch.Tensor, x_like: torch.Tensor):
        B, T, D = x_like.shape
        lambda0_raw = self.lambda0_head(cond)
        delta_raw = self.delta_lambda_head(cond)

        lam0 = self.lambda_min + (self.lambda_max - self.lambda_min) * torch.sigmoid(lambda0_raw)
        dlam = self.delta_lambda_min + (self.delta_lambda_max - self.delta_lambda_min) * torch.sigmoid(delta_raw)
        lam0 = lam0.view(B, T, D)
        dlam = dlam.view(B, T, D)
        return lam0, dlam

    def compute_coefficients(self, t: torch.Tensor):
        gamma = self.damping
        denom = math.sinh(gamma) if gamma > 1e-3 else gamma
        alpha_t = torch.sinh(gamma * (1.0 - t)) / (denom + 1e-6)
        beta_t = torch.sinh(gamma * t) / (denom + 1e-6)

        rho_t = self.sigma_max * torch.sqrt(t * (1.0 - t)) * torch.exp(-self.stiffness * t)
        rho_t = torch.maximum(rho_t, torch.tensor(self.sigma_min, device=t.device, dtype=t.dtype))
        return alpha_t, beta_t, rho_t

    def compute_bridge_statistics(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        t = self._broadcast_t(t, x1)
        alpha_t, beta_t, rho_t = self.compute_coefficients(t)
        mu_t = alpha_t * x0 + beta_t * x1

        lambda0, delta_lambda = self._flat_cond_to_precision(cond, x1)
        lambda_t = lambda0 + t * delta_lambda
        lambda_t = torch.clamp(lambda_t, min=self.lambda_min)
        std_t = rho_t / torch.sqrt(lambda_t)
        return {
            'alpha_t': alpha_t,
            'beta_t': beta_t,
            'rho_t': rho_t,
            'mu_t': mu_t,
            'lambda0': lambda0,
            'delta_lambda': delta_lambda,
            'lambda_t': lambda_t,
            'std_t': std_t,
        }

    def sample_path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        stats = self.compute_bridge_statistics(x0, x1, t, cond)
        eps = torch.randn_like(x0)
        xt = stats['mu_t'] + stats['std_t'] * eps
        return xt, stats

    def compute_target_velocity(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, xt: torch.Tensor, cond: torch.Tensor):
        # analytical derivatives for alpha, beta, rho; exact derivative for lambda_t from construction
        t_b = self._broadcast_t(t, x1)
        alpha_t, beta_t, rho_t = self.compute_coefficients(t_b)
        mu_t = alpha_t * x0 + beta_t * x1

        gamma = self.damping
        denom = math.sinh(gamma) if gamma > 1e-3 else gamma
        d_alpha = -gamma * torch.cosh(gamma * (1.0 - t_b)) / (denom + 1e-6)
        d_beta = gamma * torch.cosh(gamma * t_b) / (denom + 1e-6)

        # rho(t) = sigma_max * sqrt(t(1-t)) * exp(-k t)
        safe_sqrt = torch.sqrt(torch.clamp(t_b * (1.0 - t_b), min=1e-10))
        d_rho = self.sigma_max * torch.exp(-self.stiffness * t_b) * (
            (1.0 - 2.0 * t_b) / (2.0 * safe_sqrt) - self.stiffness * safe_sqrt
        )
        # derivative is zero when rho is clamped at sigma_min
        d_rho = torch.where(rho_t <= (self.sigma_min + 1e-12), torch.zeros_like(d_rho), d_rho)

        lambda0, delta_lambda = self._flat_cond_to_precision(cond, x1)
        lambda_t = torch.clamp(lambda0 + t_b * delta_lambda, min=self.lambda_min)

        dt_mu = d_alpha * x0 + d_beta * x1
        coeff = (d_rho / (rho_t + 1e-6)) - 0.5 * (delta_lambda / (lambda_t + 1e-6))
        target_v = dt_mu + coeff * (xt - mu_t)
        return target_v.detach(), {
            'mu_t': mu_t,
            'rho_t': rho_t,
            'lambda_t': lambda_t,
            'delta_lambda': delta_lambda,
            'coeff': coeff,
        }


class PAPISBPolicy(BasePolicy):
    def __init__(
        self,
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
        bridge_stiffness=1.5,
        bridge_damping=0.5,
        bridge_sigma=0.1,
        pa_sigma_min=1e-4,
        pa_lambda_min=0.1,
        pa_lambda_max=10.0,
        pa_delta_lambda_min=0.0,
        pa_delta_lambda_max=5.0,
        pa_bridge_hidden_dim=256,
        pa_reg_weight=1e-4,
        **kwargs,
    ):
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

        obs_encoder = MP1Encoder(
            observation_space=obs_dict,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
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

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[PA-PISB] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[PA-PISB] pointnet_type: {self.pointnet_type}", "yellow")

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
            action_visible=False,
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
        self.pa_reg_weight = pa_reg_weight

        self.flow_ratio = 0.5
        self.time_dist = ['lognorm', -0.4, 1.0]
        self.cfg_ratio = 0.10
        self.cfg_uncond = 'u'
        self.w = 2.0

        bridge_cond_dim = global_cond_dim if global_cond_dim is not None else (obs_feature_dim * n_obs_steps)
        trajectory_dim = horizon * action_dim if obs_as_global_cond else horizon * (action_dim + obs_feature_dim)
        self.physics_bridge = PAPISBBridge(
            cond_dim=bridge_cond_dim,
            trajectory_dim=trajectory_dim,
            sigma_min=pa_sigma_min,
            sigma_max=bridge_sigma,
            stiffness=bridge_stiffness,
            damping=bridge_damping,
            lambda_min=pa_lambda_min,
            lambda_max=pa_lambda_max,
            delta_lambda_min=pa_delta_lambda_min,
            delta_lambda_max=pa_delta_lambda_max,
            hidden_dim=pa_bridge_hidden_dim,
        )
        print(
            f"[PA-PISB] Initialized with k={bridge_stiffness}, gamma={bridge_damping}, "
            f"sigma={bridge_sigma}, lambda=[{pa_lambda_min},{pa_lambda_max}], "
            f"delta_lambda=[{pa_delta_lambda_min},{pa_delta_lambda_max}]"
        )
        print_params(self)

    def _encode_obs(self, nobs: Dict[str, torch.Tensor], batch_size: int, horizon: int):
        global_cond = None
        cond_data = None
        trajectory = None
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda n: n[:, :self.n_obs_steps, ...].reshape(-1, *n.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
                bridge_cond = global_cond.reshape(batch_size, -1)
            else:
                global_cond = nobs_features.reshape(batch_size, -1)
                bridge_cond = global_cond
            cond_data = None
            trajectory = None
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            bridge_cond = nobs_features[:, :self.n_obs_steps, :].reshape(batch_size, -1)
            global_cond = None
            cond_data = nobs_features
            trajectory = None
        return global_cond, bridge_cond, cond_data

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps
        device = self.device
        dtype = self.dtype

        global_cond = None
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                global_cond = nobs_features.reshape(B, -1)
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_data[:, :To, Da:] = nobs_features

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
                model_output = model(sample=x_current, timestep=t_tensor, local_cond=None, global_cond=global_cond, r=r_zeros, training=False)
                v_pred = model_output[0] if isinstance(model_output, tuple) else model_output
                x_current = x_current + v_pred * dt

        naction_pred = x_current[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return {'action': action, 'action_pred': action_pred}

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        device = nactions.device

        local_cond = None
        trajectory = nactions
        global_cond, bridge_cond, cond_features = self._encode_obs(nobs, batch_size, horizon)
        if not self.obs_as_global_cond:
            trajectory = torch.cat([nactions, cond_features], dim=-1).detach()

        eps = 1e-5
        t = torch.rand((batch_size,), device=device).float() * (1 - 2 * eps) + eps
        x1 = trajectory
        x0 = torch.randn_like(x1)

        xt, bridge_stats = self.physics_bridge.sample_path(x0, x1, t, bridge_cond)
        u_tgt, vt_stats = self.physics_bridge.compute_target_velocity(x0, x1, t, xt, bridge_cond)

        r_zeros = torch.zeros_like(t)
        model_output = self.model(sample=xt, timestep=t, global_cond=global_cond, r=r_zeros)
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
                dis_loss = dis_loss + self.dispersive_loss(features)

        # exact regularizer corresponding to bounded phase-slope assumption
        delta_lambda = bridge_stats['delta_lambda']
        lambda_t = bridge_stats['lambda_t']
        reg_phase = (delta_lambda.pow(2).mean())
        reg_cond = ((delta_lambda / (lambda_t + 1e-6)).pow(2).mean())
        pa_reg = reg_phase + reg_cond

        loss = meanflow_loss + 0.5 * dis_loss + self.pa_reg_weight * pa_reg
        mse_val = (stopgrad(error) ** 2).mean()
        loss_dict = {
            'loss': loss.item(),
            'bc_loss': loss.item(),
            'mse_val': mse_val.item(),
            'meanflow_loss': meanflow_loss.item(),
            'dis_loss': dis_loss.item() if isinstance(dis_loss, torch.Tensor) else float(dis_loss),
            'pa_reg': pa_reg.item(),
            'lambda_mean': bridge_stats['lambda_t'].mean().item(),
            'delta_lambda_mean': bridge_stats['delta_lambda'].mean().item(),
        }
        return loss, loss_dict

    def dispersive_loss(self, z, tau=1.0):
        dist_matrix = torch.cdist(z, z, p=2) ** 2
        dist_matrix = dist_matrix / (torch.max(dist_matrix) + 1e-8)
        exp_term = torch.exp(-dist_matrix / tau)
        mean_exp = torch.mean(exp_term)
        loss = torch.log(mean_exp + 1e-8)
        return loss

    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)
        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])
        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]
        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r
