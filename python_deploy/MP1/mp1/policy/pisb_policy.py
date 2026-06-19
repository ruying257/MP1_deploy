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
import os
import numpy as np
from mp1.sde_lib import ConsistencyFM
from mp1.model.common.normalizer import LinearNormalizer
from mp1.policy.base_policy import BasePolicy
from mp1.model.mean.conditional_unet1d_meanflow_dis import ConditionalUnet1D
from mp1.model.mean.mask_generator import LowdimMaskGenerator
from mp1.common.pytorch_util import dict_apply
from mp1.common.model_util import print_params
from mp1.model.vision.obs_encoder_factory import build_obs_encoder
from functools import partial
import warnings
from einops import rearrange, reduce

warnings.filterwarnings("ignore")

# ==========================================
# [创新点] Physics-Informed Generalized Schrödinger Bridge (PI-GSB)
# ==========================================
class PhysicsInformedBridge:
    def __init__(self, sigma_min=1e-4, sigma_max=1.0, stiffness=1.0, damping=0.5, device='cuda'):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.stiffness = stiffness
        self.damping = damping
        # [修改] 这里的 device 仅作为默认值，主要逻辑改为动态获取
        self.device = device 

    def compute_coefficients(self, t):
        """计算带阻尼 OU 过程的漂移和扩散系数"""
        # [核心修复] 始终跟随输入张量 t 的设备，而不是使用 self.device
        target_device = t.device if isinstance(t, torch.Tensor) else self.device
        
        if isinstance(t, torch.Tensor):
            # 确保 t 在计算图中
            t = t.to(target_device)
        else:
            t = torch.tensor(t, device=target_device)

        gamma = self.damping
        # 避免除零
        denom = np.sinh(gamma) if gamma > 1e-3 else gamma 
        
        # 1. 均值系数 (Mean Coefficients)
        numer_alpha = torch.sinh(gamma * (1 - t))
        alpha_t = numer_alpha / (denom + 1e-6)
        
        numer_beta = torch.sinh(gamma * t)
        beta_t = numer_beta / (denom + 1e-6)

        # 2. 扩散系数 (Diffusion Coefficients)
        # 钟形曲线 + 刚度指数衰减
        std_t = self.sigma_max * torch.sqrt(t * (1 - t)) * torch.exp(-self.stiffness * t)
        
        # [核心修复] 创建常量时也使用 target_device
        min_std = torch.tensor(self.sigma_min, device=target_device)
        std_t = torch.maximum(std_t, min_std)
        
        return alpha_t, beta_t, std_t

    def sample_path(self, x0, x1, t):
        """采样中间状态 x_t"""
        # 维度适配
        if t.ndim == 1: t = t.view(-1, 1)
        # 适配序列数据 (B, T, D)
        while t.ndim < x1.ndim: t = t.unsqueeze(-1)
            
        # 这里 t 的设备已经被传入的 t 决定了 (通常是 cuda)
        alpha, beta, std = self.compute_coefficients(t)
        
        # 此时 alpha, beta, std 都在 cuda 上，与 x0, x1 一致
        mu_t = alpha * x0 + beta * x1
        eps = torch.randn_like(x0)
        xt = mu_t + std * eps
        return xt, mu_t, std

    def compute_target_velocity(self, x0, x1, t, xt):
        """计算目标向量场 (Target Vector Field)"""
        # 确保 t 启用梯度用于自动微分
        t_in = t.clone().detach().requires_grad_(True)
        
        # 维度适配以便求导
        if t_in.ndim == 1: t_in = t_in.view(-1, 1)
        while t_in.ndim < x1.ndim: t_in = t_in.unsqueeze(-1)

        alpha, beta, sigma = self.compute_coefficients(t_in)
        mu_t = alpha * x0 + beta * x1
        
        # 自动微分求导 (Drift Term)
        # 注意：create_graph=True 是必须的
        d_alpha = torch.autograd.grad(alpha.sum(), t_in, create_graph=True)[0]
        d_beta = torch.autograd.grad(beta.sum(), t_in, create_graph=True)[0]
        d_sigma = torch.autograd.grad(sigma.sum(), t_in, create_graph=True)[0]
        
        dt_mu = d_alpha * x0 + d_beta * x1
        
        # 扩散修正项 (Score Term)
        score_component = (d_sigma / (sigma + 1e-6)) * (xt - mu_t)
        
        target_v = dt_mu + score_component
        return target_v.detach()
#=====================================================


class PISBPolicy(BasePolicy):
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
            obs_encoder_kind="pointcloud",
            image_encoder_output_dim=64,
            image_base_channels=32,
            share_image_encoder=False,
            # [新增] 在参数列表最后添加 PISB 参数
            bridge_stiffness=1.0,
            bridge_damping=0.5,
            bridge_sigma=0.8,
            **kwargs):
        super().__init__()

        self.condition_type = condition_type

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
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

        # create diffusion model
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
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")



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

        self.flow_ratio=0.5
        self.time_dist=['lognorm', -0.4, 1.0]
        self.cfg_ratio=0.10
        cfg_scale=2.0
        # experimental
        self.cfg_uncond='u'
        self.w = cfg_scale

        # [新增] 初始化物理桥
        self.physics_bridge = PhysicsInformedBridge(
            stiffness=bridge_stiffness,
            damping=bridge_damping,
            sigma_max=bridge_sigma,
            device=self.device
        )
        self._loss_profile_enabled = (
            os.getenv("MP1_PROFILE_LOSS", "0") == "1"
            or os.getenv("MP1_PROFILE_TRAIN", "0") == "1"
        )
        self._last_loss_profile = {}
        print(f"[PI-GSB] Initialized with k={bridge_stiffness}, gamma={bridge_damping}, sigma={bridge_sigma}")

        print_params(self)

    @staticmethod
    def _maybe_sync(device: torch.device, enabled: bool):
        if enabled and device.type == 'cuda':
            torch.cuda.synchronize(device)
        


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud']
        
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        model = self.model
        model.eval()
        
        # 1. 初始化噪声 x_current (Prior)
        # 形状与 cond_data 一致 (B, T, Da) 或 (B, T, Da+Do)
        x_current = torch.randn(size=cond_data.shape, dtype=cond_data.dtype, device=cond_data.device)

        # 2. 设置推理步数
        # 如果 yaml 里没写，默认给 10 步 (建议 10~20 步)
        steps = self.num_inference_steps if self.num_inference_steps is not None else 10
        dt = 1.0 / steps
        
        # 准备辅助变量
        r_zeros = torch.zeros((B,), device=device) # r parameter

        # 3. Euler 积分循环 (从 t=0 到 t=1)
        with torch.no_grad():
            for i in range(steps):
                # 当前时间点 t (从 0 开始，每次增加 dt)
                t_val = i / steps
                t_tensor = torch.full((B,), t_val, device=device, dtype=dtype)
                
                # 预测当前位置的速度场 v(x_t, t)
                model_output = model(sample=x_current, 
                                   timestep=t_tensor, 
                                   local_cond=local_cond, 
                                   global_cond=global_cond, 
                                   r=r_zeros, 
                                   training=False)
                
                # 兼容模型输出格式 (可能是 tuple)
                if isinstance(model_output, tuple):
                    v_pred = model_output[0]
                else:
                    v_pred = model_output
                
                # Euler 更新: x_{t+1} = x_t + v * dt
                # 这就像是在物理场中一步步“走”向目标
                x_current = x_current + v_pred * dt

        # 最终结果
        naction_pred = x_current
        
        # 截取 action 维度 (如果有 obs concat 在里面的话)
        naction_pred = naction_pred[..., :Da]
        # ==========================
        
        # unnormalize prediction
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred,
        }
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        profile_enabled = self._loss_profile_enabled
        loss_profile = {}

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        x = trajectory
        
        device = cond_data.device
        self._maybe_sync(device, profile_enabled)
        total_start = time.perf_counter() if profile_enabled else None
        prep_start = time.perf_counter() if profile_enabled else None
        if profile_enabled:
            self._maybe_sync(device, True)
            loss_profile['prep'] = time.perf_counter() - prep_start
        
        encode_start = time.perf_counter() if profile_enabled else None
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda n: n[:,:self.n_obs_steps,...].reshape(-1,*n.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
            # this_n_point_cloud = this_nobs['imagin_robot'].reshape(batch_size,-1, *this_nobs['imagin_robot'].shape[1:])
            this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
            this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()
        if profile_enabled:
            self._maybe_sync(device, True)
            loss_profile['obs_encode'] = time.perf_counter() - encode_start


        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)
        
        # === [PISB Loss 核心修改开始] ===
        # 1. 采样时间步 t ~ Uniform[0, 1]
        # t = torch.rand((batch_size,), device=device).float()
        eps = 1e-5  # 安全边界
        t = torch.rand((batch_size,), device=device).float() * (1 - 2 * eps) + eps
        
        # 2. 构造两端分布
        x1 = trajectory # Data (Target Action)
        x0 = torch.randn_like(x1) # Noise (Prior)
        
        # 3. 维度适配 (B,) -> (B, 1, 1)
        t_expand = t.view(batch_size, 1, 1)
        
        # 4. 物理桥采样中间状态 xt
        bridge_start = time.perf_counter() if profile_enabled else None
        xt, mu_t, std_t = self.physics_bridge.sample_path(x0, x1, t_expand)
        
        # 5. 计算物理场定义的目标速度 u_tgt
        u_tgt = self.physics_bridge.compute_target_velocity(x0, x1, t_expand, xt)
        if profile_enabled:
            self._maybe_sync(device, True)
            loss_profile['bridge'] = time.perf_counter() - bridge_start
        
        # 6. 模型前向传播
        # 注意：PISB 关注瞬时场，传入 r=0 即可
        r_zeros = torch.zeros_like(t)
        
        # 调用模型，获取预测值和特征
        # Unet 的输出通常是 (v_pred, features)
        unet_start = time.perf_counter() if profile_enabled else None
        model_output = self.model(
            sample=xt, 
            timestep=t, 
            global_cond=global_cond, 
            r=r_zeros
        )
        
        if isinstance(model_output, tuple):
            v_pred, features = model_output
        else:
            v_pred, features = model_output, [] # 兼容如果不返回特征的情况
        if profile_enabled:
            self._maybe_sync(device, True)
            loss_profile['unet_forward'] = time.perf_counter() - unet_start

        # 7. 计算损失
        # 主损失：Flow Matching Loss (使用原代码的 adaptive_l2_loss 以保持鲁棒性)
        loss_terms_start = time.perf_counter() if profile_enabled else None
        error = v_pred - stopgrad(u_tgt)
        meanflow_loss = adaptive_l2_loss(error)

        # 辅助损失：Dispersive Loss (完全保留原代码逻辑)
        dis_loss = 0
        # 检查 features 是否有效且非空
        if features is not None and len(features) > 0:
            # 假设 features 是一个 list，像原代码 loop 处理
            # 原代码逻辑: for i in range(pred[1].shape[0]) ... 但这里 pred[1] 应该是一个 list of tensors
            # 这种 UNet 通常返回多层特征用于计算 dispersive loss
            if isinstance(features, (list, tuple)):
                for feat in features:
                    dis_loss += self.dispersive_loss(feat)
            else:
                # 如果 features 只是一个 tensor
                dis_loss += self.dispersive_loss(features)

        loss = meanflow_loss + 0.5 * dis_loss
        # === [PISB Loss 核心修改结束] ===
        
        mse_val = (stopgrad(error) ** 2).mean()
        if profile_enabled:
            self._maybe_sync(device, True)
            loss_profile['loss_terms'] = time.perf_counter() - loss_terms_start

        loss_dict = {
                'loss': loss.item(), # Renamed to standard 'loss' for clarity
                'bc_loss': loss.item(), # Keep compatibility
                'mse_val': mse_val.item(),
                'meanflow_loss': meanflow_loss.item(),
                'dis_loss': dis_loss.item() if isinstance(dis_loss, torch.Tensor) else dis_loss
            }
        if profile_enabled:
            loss_profile['total'] = time.perf_counter() - total_start
            self._last_loss_profile = loss_profile
        else:
            self._last_loss_profile = {}
        
        return loss, loss_dict

    def dispersive_loss(self, z, tau=1.0):
        
        dist_matrix = torch.cdist(z, z, p=2) ** 2
        # 归一化到均值0、标准差1
        # mean = torch.mean(dist_matrix)
        # std = torch.std(dist_matrix) + 1e-8  # 避免除零
        # dist_matrix = (dist_matrix - mean) / std
        dist_matrix = dist_matrix / (torch.max(dist_matrix))
        exp_term = torch.exp(-dist_matrix / tau)
        mean_exp = torch.mean(exp_term)
        loss = torch.log(mean_exp) # 
        
        return loss
    
    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)

        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))  # Apply sigmoid

        # Assign t = max, r = min, for each pair
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

def normalize_to_neg1_1(x):
    return x * 2 - 1


def unnormalize_to_0_1(x):
    return (x + 1) * 0.5

def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, C, W, H)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.mean(error ** 2, dim=tuple(range(1, error.ndim)))    
    # delta_sq = torch.sum(error ** 2, dim=tuple(range(1, error.ndim)))
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq
    return (stopgrad(w) * loss).mean()
