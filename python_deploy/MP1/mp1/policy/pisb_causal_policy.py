import sys
sys.path.append('mp1')
from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint
import copy
import time
import numpy as np
from mp1.sde_lib import ConsistencyFM
from mp1.model.common.normalizer import LinearNormalizer
from mp1.policy.base_policy import BasePolicy
from mp1.model.mean.conditional_unet1d_meanflow_dis import ConditionalUnet1D
from mp1.model.mean.mask_generator import LowdimMaskGenerator
from mp1.common.pytorch_util import dict_apply
from mp1.common.model_util import print_params
from mp1.model.vision.pointnet_extractor import MP1Encoder

# [引入我们新写的 Causal Memory]
from mp1.model.vision.causal_memory import CausalMemoryNetwork 

from functools import partial
import warnings
from einops import rearrange, reduce

warnings.filterwarnings("ignore")

# ==========================================
# Physics-Informed Generalized Schrödinger Bridge (PI-GSB)
# ==========================================
class PhysicsInformedBridge:
    def __init__(self, sigma_min=1e-4, sigma_max=1.0, stiffness=1.0, damping=0.5, device='cuda'):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.stiffness = stiffness
        self.damping = damping
        self.device = device 

    def compute_coefficients(self, t):
        target_device = t.device if isinstance(t, torch.Tensor) else self.device
        if isinstance(t, torch.Tensor):
            t = t.to(target_device)
        else:
            t = torch.tensor(t, device=target_device)

        gamma = self.damping
        denom = np.sinh(gamma) if gamma > 1e-3 else gamma 
        
        numer_alpha = torch.sinh(gamma * (1 - t))
        alpha_t = numer_alpha / (denom + 1e-6)
        numer_beta = torch.sinh(gamma * t)
        beta_t = numer_beta / (denom + 1e-6)

        std_t = self.sigma_max * torch.sqrt(t * (1 - t)) * torch.exp(-self.stiffness * t)
        min_std = torch.tensor(self.sigma_min, device=target_device)
        std_t = torch.maximum(std_t, min_std)
        return alpha_t, beta_t, std_t

    def sample_path(self, x0, x1, t):
        if t.ndim == 1: t = t.view(-1, 1)
        while t.ndim < x1.ndim: t = t.unsqueeze(-1)
        alpha, beta, std = self.compute_coefficients(t)
        mu_t = alpha * x0 + beta * x1
        eps = torch.randn_like(x0)
        xt = mu_t + std * eps
        return xt, mu_t, std

    def compute_target_velocity(self, x0, x1, t, xt):
        t_in = t.clone().detach().requires_grad_(True)
        if t_in.ndim == 1: t_in = t_in.view(-1, 1)
        while t_in.ndim < x1.ndim: t_in = t_in.unsqueeze(-1)

        alpha, beta, sigma = self.compute_coefficients(t_in)
        mu_t = alpha * x0 + beta * x1
        
        d_alpha = torch.autograd.grad(alpha.sum(), t_in, create_graph=True)[0]
        d_beta = torch.autograd.grad(beta.sum(), t_in, create_graph=True)[0]
        d_sigma = torch.autograd.grad(sigma.sum(), t_in, create_graph=True)[0]
        
        dt_mu = d_alpha * x0 + d_beta * x1
        score_component = (d_sigma / (sigma + 1e-6)) * (xt - mu_t)
        target_v = dt_mu + score_component
        return target_v.detach()
#=====================================================

class PISBCausalPolicy(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
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
            # [PISB Params]
            bridge_stiffness=1.0,
            bridge_damping=0.5,
            bridge_sigma=0.8,
            # [Causal Memory Params]
            use_causal_memory=True,
            memory_num_layers=2,
            memory_num_heads=8,
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

        obs_encoder = MP1Encoder(observation_space=obs_dict,
                                 img_crop_shape=crop_shape,
                                 out_channel=encoder_output_dim,
                                 pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                 use_pc_color=use_pc_color,
                                 pointnet_type=pointnet_type)

        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None

        self.use_causal_memory = use_causal_memory
        if self.use_causal_memory:
            cprint(f"[PISBCausalPolicy] Injecting Causal Memory Network (Layers: {memory_num_layers}, Heads: {memory_num_heads})", "cyan")
            self.causal_memory = CausalMemoryNetwork(
                feature_dim=obs_feature_dim,
                num_layers=memory_num_layers,
                num_heads=memory_num_heads,
                dropout=0.1
            )

        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                # 巧妙的 Trick：使用 Causal Memory 的 Attention Pooling 后，维度变回 obs_feature_dim
                global_cond_dim = obs_feature_dim if self.use_causal_memory else obs_feature_dim * n_obs_steps
        
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type

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

        # 初始化物理桥
        self.physics_bridge = PhysicsInformedBridge(
            stiffness=bridge_stiffness,
            damping=bridge_damping,
            sigma_max=bridge_sigma,
            device=self.device
        )
        print_params(self)
        
    def _extract_causal_conditions(self, nobs_features, batch_size):
        """将原始观测特征通过因果记忆模块转换为鲁棒的全局/局部条件"""
        B = batch_size
        if self.obs_as_global_cond:
            if self.use_causal_memory:
                # 重新整理形状以适应序列模型 (B, T, D)
                seq_features = nobs_features.reshape(B, self.n_obs_steps, -1)
                # 穿过因果记忆网络
                causal_features = self.causal_memory(seq_features)
                
                if "cross_attention" in self.condition_type:
                    global_cond = causal_features # (B, T, D) 作为交叉注意力序列
                else:
                    # 使用 Attention Pooling 获取融合后的全局历史特征
                    global_cond = self.causal_memory.get_pooled_memory(causal_features) # (B, D)
            else:
                if "cross_attention" in self.condition_type:
                    global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
                else:
                    global_cond = nobs_features.reshape(B, -1)
            return global_cond
        else:
            raise NotImplementedError("Currently Causal Memory relies on obs_as_global_cond=True")

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raw_obs = obs_dict
        obs_wo_image = {k: v for k, v in raw_obs.items() if k != 'image'}
        nobs = self.normalizer.normalize(obs_wo_image)
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        device = self.device
        dtype = self.dtype

        local_cond = None
        
        # 特征提取与因果推断
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = self._extract_causal_conditions(nobs_features, B)
        
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
                
                model_output = model(sample=x_current, timestep=t_tensor, 
                                   local_cond=local_cond, global_cond=global_cond, 
                                   r=r_zeros, training=False)
                
                v_pred = model_output[0] if isinstance(model_output, tuple) else model_output
                x_current = x_current + v_pred * dt

        naction_pred = x_current[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        return {'action': action, 'action_pred': action_pred}

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        batch_size = nactions.shape[0]
        trajectory = nactions
        device = trajectory.device
        
        this_nobs = dict_apply(nobs, lambda n: n[:,:self.n_obs_steps,...].reshape(-1,*n.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        
        global_cond = self._extract_causal_conditions(nobs_features, batch_size)

        eps = 1e-5
        t = torch.rand((batch_size,), device=device).float() * (1 - 2 * eps) + eps
        x1 = trajectory 
        x0 = torch.randn_like(x1)
        t_expand = t.view(batch_size, 1, 1)
        
        xt, mu_t, std_t = self.physics_bridge.sample_path(x0, x1, t_expand)
        u_tgt = self.physics_bridge.compute_target_velocity(x0, x1, t_expand, xt)
        
        r_zeros = torch.zeros_like(t)
        model_output = self.model(sample=xt, timestep=t, global_cond=global_cond, r=r_zeros)
        
        if isinstance(model_output, tuple):
            v_pred, features = model_output
        else:
            v_pred, features = model_output, [] 

        error = v_pred - stopgrad(u_tgt)
        meanflow_loss = adaptive_l2_loss(error)

        dis_loss = 0
        if features is not None and len(features) > 0:
            if isinstance(features, (list, tuple)):
                for feat in features:
                    dis_loss += self.dispersive_loss(feat)
            else:
                dis_loss += self.dispersive_loss(features)

        # 辅助损失：鼓励因果记忆不要坍缩 (Contrastive Regularization on memory)
        memory_reg_loss = 0
        if self.use_causal_memory and "cross_attention" not in self.condition_type:
            # 简单 L2 正则化避免提取的全局特征过大
            memory_reg_loss = 1e-4 * torch.mean(global_cond ** 2)

        loss = meanflow_loss + 0.5 * dis_loss + memory_reg_loss
        mse_val = (stopgrad(error) ** 2).mean()

        loss_dict = {
                'loss': loss.item(), 
                'mse_val': mse_val.item(),
                'meanflow_loss': meanflow_loss.item(),
                'dis_loss': dis_loss.item() if isinstance(dis_loss, torch.Tensor) else dis_loss
            }
        return loss, loss_dict

    def dispersive_loss(self, z, tau=1.0):
        dist_matrix = torch.cdist(z, z, p=2) ** 2
        dist_matrix = dist_matrix / (torch.max(dist_matrix) + 1e-8)
        exp_term = torch.exp(-dist_matrix / tau)
        mean_exp = torch.mean(exp_term)
        return torch.log(mean_exp + 1e-8)

def stopgrad(x):
    return x.detach()

def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    delta_sq = torch.mean(error ** 2, dim=tuple(range(1, error.ndim)))    
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    return (stopgrad(w) * delta_sq).mean()