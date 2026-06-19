# import torch
# import numpy as np
# from typing import Dict
# from termcolor import cprint
# from mp1.policy.pisb_policy import PISBPolicy
# from mp1.common.pytorch_util import dict_apply

# class PISBAbiPolicy(PISBPolicy):
#     """
#     [创新点 2] Analytical Bridge Inversion (ABI) 策略
    
#     继承自 PISBPolicy (保持训练过程和物理先验绝对一致)。
#     在推理阶段 (predict_action) 彻底抛弃存在巨大截断误差的欧拉积分，
#     利用物理桥在 t=0 时刻的解析逆公式，直接一步 (1-NFE) 求解出极其精准的目标动作 x_1。
#     """
#     def __init__(self, shape_meta, **kwargs):
#         super().__init__(shape_meta, **kwargs)
#         cprint("[PISB-ABI] Analytical Bridge Inversion (Exact 1-Step Inference) Activated.", "yellow")

#     def get_analytical_coeffs(self):
#         """
#         [核心数学创新] 
#         根据 PISB 的 Ornstein-Uhlenbeck 过程定义，
#         计算漂移系数在 t=0 时刻的精确解析导数。
#         v_0 = d(alpha)/dt * x_0 + d(beta)/dt * x_1
#         """
#         gamma = self.physics_bridge.damping
        
#         # 避免除零，给极小阻尼加上保护
#         if gamma < 1e-3:
#             # 泰勒展开在 gamma 趋于 0 时的极限退化为标准流匹配 (Straight-line)
#             # A_t0 = -1.0, B_t0 = 1.0  => v = x_1 - x_0
#             A_t0 = -1.0
#             B_t0 = 1.0
#         else:
#             # PISB 物理桥的精确导数公式
#             sinh_gamma = np.sinh(gamma)
#             cosh_gamma = np.cosh(gamma)
            
#             A_t0 = -gamma * cosh_gamma / sinh_gamma
#             B_t0 = gamma / sinh_gamma
            
#         return A_t0, B_t0

#     def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         """
#         重写推理流程，实现真正的零截断单步生成。
#         """
#         # 1. 提取观测特征 (完全复用父类的前处理逻辑)
#         nobs = self.normalizer.normalize(obs_dict)
#         if not self.use_pc_color:
#             nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
#         value = next(iter(nobs.values()))
#         B, To = value.shape[:2]
#         T = self.horizon
#         Da = self.action_dim
#         Do = self.obs_feature_dim
#         To = self.n_obs_steps

#         device = self.device
#         dtype = self.dtype

#         local_cond = None
#         global_cond = None
#         if self.obs_as_global_cond:
#             this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
#             nobs_features = self.obs_encoder(this_nobs)
#             if "cross_attention" in self.condition_type:
#                 global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
#             else:
#                 global_cond = nobs_features.reshape(B, -1)
#             cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
#             cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
#         else:
#             this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
#             nobs_features = self.obs_encoder(this_nobs)
#             nobs_features = nobs_features.reshape(B, To, -1)
#             cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
#             cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
#             cond_data[:,:To,Da:] = nobs_features
#             cond_mask[:,:To,Da:] = True

#         model = self.model
#         model.eval()
        
#         # 2. 初始化纯噪声 x_0 
#         x_0 = torch.randn(size=cond_data.shape, dtype=cond_data.dtype, device=cond_data.device)

#         r_zeros = torch.zeros((B,), device=device)
        
#         # [核心]: ABI 只需要在 t=0 评估一次模型 (1-NFE)！
#         t_tensor = torch.zeros((B,), device=device, dtype=dtype)

#         # 获取解析逆系数 A 和 B (这是一个极快的标量运算)
#         A_coeff, B_coeff = self.get_analytical_coeffs()

#         with torch.no_grad():
#             # 3. 让网络基于当前噪声和条件，预测出 t=0 时刻的速度场 v_pred
#             model_output = model(sample=x_0, 
#                                timestep=t_tensor, 
#                                local_cond=local_cond, 
#                                global_cond=global_cond, 
#                                r=r_zeros, 
#                                training=False)
            
#             if isinstance(model_output, tuple):
#                 v_pred = model_output[0]
#             else:
#                 v_pred = model_output
            
#             # =========================================================
#             # [终极创新] 解析逆运算 (Analytical Bridge Inversion)
#             # 根据物理桥的真实物理规律：v_pred = A * x_0 + B * x_1   
#             # 代数精确反解 =>   x_1 = (v_pred - A * x_0) / B
#             # 彻底干掉 Euler 积分！0 截断误差！
#             # =========================================================
#             x_1 = (v_pred - A_coeff * x_0) / B_coeff
            
#             # 我们直接把 10 步才能走完的曲线目标，1 步算了出来！
#             x_current = x_1

#         # 4. 截取并反归一化目标动作
#         naction_pred = x_current[..., :Da]
#         action_pred = self.normalizer['action'].unnormalize(naction_pred)

#         start = To - 1
#         end = start + self.n_action_steps
#         action = action_pred[:,start:end]
        
#         result = {
#             'action': action,
#             'action_pred': action_pred,
#         }
        
#         return result


import torch
import numpy as np
from typing import Dict
from termcolor import cprint
from mp1.policy.pisb_policy import PISBPolicy
from mp1.common.pytorch_util import dict_apply

class PISBAbiPolicy(PISBPolicy):
    """
    [创新点 2] Analytical Bridge Inversion (ABI) 策略 - 自动微分版
    
    无需手动推导数学公式！无论 PISB 的物理先验如何改变，
    本策略均能利用 PyTorch Autograd 自动、精确地求解出 t=0 时刻的解析逆矩阵。
    """
    def __init__(self, shape_meta, **kwargs):
        super().__init__(shape_meta, **kwargs)
        cprint("[PISB-ABI] Schedule-Agnostic Analytical Bridge Inversion (Autograd 1-Step) Activated.", "yellow")

    def get_autograd_coeffs(self):
        """
        [核心创新] 基于自动微分的通用漂移系数求解器。
        v_0 = d(alpha)/dt * x_0 + d(beta)/dt * x_1
        """
        # =======================================================
        # [致命 Bug 修复] 因为外部 Eval runner 默认全局禁用了梯度，
        # 我们必须用 enable_grad() 强制撕开一个计算图的口子！
        # =======================================================
        with torch.enable_grad():
            # 1. 创建带有梯度的叶子节点张量 t=0
            t_in = torch.tensor([0.0], device=self.device, requires_grad=True)
            
            # 2. 前向传播：调用任何你喜欢的物理桥公式
            alpha, beta, _ = self.physics_bridge.compute_coefficients(t_in)
            
            # 3. 自动求导：精确计算导数 A 和 B
            # 注意 retain_graph=True，因为 alpha 和 beta 共享同一个 t_in 的计算图
            A_t0 = torch.autograd.grad(alpha.sum(), t_in, retain_graph=True)[0].item()
            B_t0 = torch.autograd.grad(beta.sum(), t_in)[0].item()
            
        return A_t0, B_t0

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        重写推理流程，实现真正的零截断单步生成。
        """
        # 1. 提取观测特征 (完全复用父类的前处理逻辑)
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

        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                global_cond = nobs_features.reshape(B, -1)
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        model = self.model
        model.eval()
        
        # 2. 初始化纯噪声 x_0 
        x_0 = torch.randn(size=cond_data.shape, dtype=cond_data.dtype, device=cond_data.device)

        r_zeros = torch.zeros((B,), device=device)
        t_tensor = torch.zeros((B,), device=device, dtype=dtype)

        # [核心]：调用刚刚写好的 Autograd 求解器
        A_coeff, B_coeff = self.get_autograd_coeffs()

        with torch.no_grad():
            # 3. 让网络基于当前噪声和条件，预测出 t=0 时刻的速度场 v_pred
            model_output = model(sample=x_0, 
                               timestep=t_tensor, 
                               local_cond=local_cond, 
                               global_cond=global_cond, 
                               r=r_zeros, 
                               training=False)
            
            if isinstance(model_output, tuple):
                v_pred = model_output[0]
            else:
                v_pred = model_output
            
            # =========================================================
            # [终极创新] 解析逆运算 (Analytical Bridge Inversion)
            # 代数精确反解 =>   x_1 = (v_pred - A * x_0) / B
            # =========================================================
            # 防止除零保护
            if abs(B_coeff) < 1e-6:
                B_coeff = 1e-6 if B_coeff >= 0 else -1e-6
                
            x_1 = (v_pred - A_coeff * x_0) / B_coeff
            x_current = x_1

        # 4. 截取并反归一化目标动作
        naction_pred = x_current[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred,
        }
        
        return result