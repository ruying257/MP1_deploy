# import torch
# import torch.nn.functional as F
# from termcolor import cprint
# from mp1.policy.pisb_policy import PISBPolicy
# from mp1.model.mean.het_unet import HeteroscedasticUnet

# class PISBHetPolicy(PISBPolicy):
#     """
#     [创新点 2 - 策略层] Thermodynamic PISB Policy
    
#     引入 Heteroscedastic NLL Loss。
#     物理意义：在薛定谔桥中，我们不再假设恒定的扩散系数，而是允许策略根据当前状态
#     自动推断"局部温度" (Local Temperature / Uncertainty)。
#     """
#     def __init__(self, shape_meta, **kwargs):
#         # 1. 调用父类初始化，但拦截 model 的创建
#         # 我们需要先保存 kwargs，稍后手动创建 HetUnet
#         super().__init__(shape_meta, **kwargs)
        
#         # 提取新增的 YAML 参数，设置默认值以防万一
#         self.nll_weight = kwargs.get('nll_weight', 1.0)
#         het_init_bias = kwargs.get('het_init_bias', -5.0)
#         log_var_min = kwargs.get('log_var_min', -10.0)
#         log_var_max = kwargs.get('log_var_max', 2.0)

#         # ====================================================
#         # [修复报错核心] 重新计算 input_dim 和 global_cond_dim
#         # 根据父类解析好的属性，重新推导维度，而不是去读取 self.model
#         # ====================================================
#         input_dim = self.action_dim
#         global_cond_dim = None
        
#         if self.obs_as_global_cond:
#             if "cross_attention" in self.condition_type:
#                 global_cond_dim = self.obs_feature_dim
#             else:
#                 global_cond_dim = self.obs_feature_dim * self.n_obs_steps
#         else:
#             input_dim = self.action_dim + self.obs_feature_dim
        
#         # 创建模型时传入新增参数
#         self.model = HeteroscedasticUnet(
#             input_dim=input_dim,
#             global_cond_dim=global_cond_dim,
#             het_init_bias=het_init_bias,
#             log_var_min=log_var_min,
#             log_var_max=log_var_max,
#             diffusion_step_embed_dim=kwargs.get('diffusion_step_embed_dim', 128),
#             down_dims=kwargs.get('down_dims', (512, 1024, 2048)),
#             kernel_size=kwargs.get('kernel_size', 5),
#             n_groups=kwargs.get('n_groups', 8),
#             condition_type=kwargs.get('condition_type', "film"),
#             use_down_condition=kwargs.get('use_down_condition', True),
#             use_mid_condition=kwargs.get('use_mid_condition', True),
#             use_up_condition=kwargs.get('use_up_condition', True),
#         ).to(self.device)
        
#         cprint("[PISB-Het] Heteroscedastic Uncertainty Module Activated.", "cyan")

#     def compute_loss(self, batch):
#         """
#         重写 Loss 计算，实现 NLL Loss
#         """
#         # 1. 数据预处理 (复用父类逻辑，避免重复代码)
#         # 注意：这里我们手动执行父类的前半部分逻辑
#         nobs = self.normalizer.normalize(batch['obs'])
#         nactions = self.normalizer['action'].normalize(batch['action'])
#         if not self.use_pc_color:
#             nobs['point_cloud'] = nobs['point_cloud'][..., :3]
            
#         batch_size = nactions.shape[0]
        
#         # Condition Encoding
#         this_nobs = self.get_condition_input(nobs) # 假设提取为 helper 函数，或者直接写
#         # 为了完整性，这里展开写：
#         from mp1.common.pytorch_util import dict_apply
#         this_nobs_reshaped = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
#         nobs_features = self.obs_encoder(this_nobs_reshaped)
#         if "cross_attention" in self.condition_type:
#             global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
#         else:
#             global_cond = nobs_features.reshape(batch_size, -1)

#         # 2. PISB 采样 (核心物理过程)
#         # 使用 epsilon 截断避免边界梯度爆炸
#         eps = 1e-5
#         t = torch.rand((batch_size,), device=self.device).float() * (1 - 2*eps) + eps
        
#         x1 = nactions
#         x0 = torch.randn_like(x1)
#         t_expand = t.view(batch_size, 1, 1)
        
#         # 调用父类的 physics_bridge
#         xt, mu_t, std_t = self.physics_bridge.sample_path(x0, x1, t_expand)
#         u_tgt = self.physics_bridge.compute_target_velocity(x0, x1, t_expand, xt)
        
#         # 3. 模型预测 (双输出)
#         r_zeros = torch.zeros_like(t)
#         v_pred, log_var = self.model(sample=xt, timestep=t, global_cond=global_cond, r=r_zeros)
        
#         # 4. [核心创新] Heteroscedastic NLL Loss
#         # Gaussian NLL = 0.5 * ( log(var) + (target - pred)^2 / var )
#         # log_var = log(sigma^2)
#         # inv_var = exp(-log_var)
        
#         inv_var = torch.exp(-log_var)
#         mse = (v_pred - u_tgt.detach()) ** 2
        
#         # [修改] 引入 yaml 控制的权重
#         nll_loss = 0.5 * (inv_var * mse + log_var)
#         loss = self.nll_weight * nll_loss.mean()
        
#         # 记录详细指标
#         loss_dict = {
#             "loss": loss.item(),
#             "nll_loss": nll_loss.mean().item(),
#             "mse_loss": mse.mean().item(), # 监控纯精度
#             "mean_sigma": torch.exp(0.5 * log_var).mean().item() # 监控预测的不确定性大小
#         }
        
#         return loss, loss_dict

#     def predict_action(self, obs_dict):
#         """
#         推理时，我们依然主要关注均值 v_pred。
#         但 Heteroscedasticity 训练出的 v_pred 通常比 MSE 训练出的更准，
#         因为它学会了忽略异常值(Outliers)。
#         """
#         # 复用父类的多步推理逻辑，但需要适配 model 的双输出
#         # 这里必须重写，因为 model 输出变了
        
#         nobs = self.normalizer.normalize(obs_dict)
#         if not self.use_pc_color: nobs['point_cloud'] = nobs['point_cloud'][..., :3]
#         B = nobs['point_cloud'].shape[0]
        
#         # Prepare Condition
#         from mp1.common.pytorch_util import dict_apply
#         this_nobs_reshaped = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
#         nobs_features = self.obs_encoder(this_nobs_reshaped)
#         if "cross_attention" in self.condition_type:
#             global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
#         else:
#             global_cond = nobs_features.reshape(B, -1)

#         # Euler Integration
#         x_current = torch.randn((B, self.horizon, self.action_dim), device=self.device)
#         steps = self.num_inference_steps if self.num_inference_steps is not None else 10
#         dt = 1.0 / steps
#         r_zeros = torch.zeros((B,), device=self.device)

#         self.model.eval()
#         with torch.no_grad():
#             for i in range(steps):
#                 t_val = i / steps
#                 t_tensor = torch.full((B,), t_val, device=self.device)
                
#                 # [适配] 获取 v_pred，忽略 log_var
#                 # 除非你想利用 log_var 做 test-time adaptation (进阶玩法)
#                 v_pred, _ = self.model(sample=x_current, timestep=t_tensor, global_cond=global_cond, r=r_zeros)
                
#                 x_current = x_current + v_pred * dt

#         action_pred = self.normalizer['action'].unnormalize(x_current)
#         start = self.n_obs_steps - 1
#         end = start + self.n_action_steps
#         action = action_pred[:, start:end]
        
#         return {'action': action, 'action_pred': action_pred}

import torch
import torch.nn.functional as F
from termcolor import cprint
from mp1.policy.pisb_policy import PISBPolicy
from mp1.model.mean.het_unet import HeteroscedasticUnet
from mp1.common.pytorch_util import dict_apply

class PISBHetPolicy(PISBPolicy):
    def __init__(self, shape_meta, **kwargs):
        super().__init__(shape_meta, **kwargs)
        
        # 提取 yaml 参数
        self.nll_weight = kwargs.get('nll_weight', 1.0)
        het_init_bias = kwargs.get('het_init_bias', -5.0)
        log_var_min = kwargs.get('log_var_min', -10.0)
        log_var_max = kwargs.get('log_var_max', 2.0)
        
        # ====================================================
        # [核心修复] 根据父类解析的维度重新推导，不再依赖 self.model
        # ====================================================
        input_dim = self.action_dim
        global_cond_dim = None
        
        if self.obs_as_global_cond:
            if "cross_attention" in self.condition_type:
                global_cond_dim = self.obs_feature_dim
            else:
                global_cond_dim = self.obs_feature_dim * self.n_obs_steps
        else:
            input_dim = self.action_dim + self.obs_feature_dim
            
        # 实例化我们刚写好的 HeteroscedasticUnet
        self.model = HeteroscedasticUnet(
            input_dim=input_dim,
            global_cond_dim=global_cond_dim,
            het_init_bias=het_init_bias,
            log_var_min=log_var_min,
            log_var_max=log_var_max,
            diffusion_step_embed_dim=kwargs.get('diffusion_step_embed_dim', 128),
            down_dims=kwargs.get('down_dims', (512, 1024, 2048)),
            kernel_size=kwargs.get('kernel_size', 5),
            n_groups=kwargs.get('n_groups', 8),
            condition_type=kwargs.get('condition_type', "film"),
            use_down_condition=kwargs.get('use_down_condition', True),
            use_mid_condition=kwargs.get('use_mid_condition', True),
            use_up_condition=kwargs.get('use_up_condition', True),
        ).to(self.device)
        
        cprint("[PISB-Het] Heteroscedastic Uncertainty Module Activated.", "cyan")

    # 之前的版本，也就是加了NLL loss的创新点2
    # def compute_loss(self, batch):
    #     nobs = self.normalizer.normalize(batch['obs'])
    #     nactions = self.normalizer['action'].normalize(batch['action'])
    #     if not self.use_pc_color:
    #         nobs['point_cloud'] = nobs['point_cloud'][..., :3]
    #     batch_size = nactions.shape[0]
        
    #     # Condition Encoding
    #     this_nobs_reshaped = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
    #     nobs_features = self.obs_encoder(this_nobs_reshaped)
    #     if "cross_attention" in self.condition_type:
    #         global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
    #     else:
    #         global_cond = nobs_features.reshape(batch_size, -1)

    #     # PISB 采样
    #     eps = 1e-5
    #     t = torch.rand((batch_size,), device=self.device).float() * (1 - 2*eps) + eps
        
    #     x1 = nactions
    #     x0 = torch.randn_like(x1)
    #     t_expand = t.view(batch_size, 1, 1)
        
    #     xt, mu_t, std_t = self.physics_bridge.sample_path(x0, x1, t_expand)
    #     u_tgt = self.physics_bridge.compute_target_velocity(x0, x1, t_expand, xt)
        
    #     # 模型预测 (接收三个返回值，包含 features)
    #     r_zeros = torch.zeros_like(t)
    #     v_pred, log_var, features = self.model(sample=xt, timestep=t, global_cond=global_cond, r=r_zeros, training=True)
        
    #     # NLL 损失计算
    #     inv_var = torch.exp(-log_var)
    #     mse = (v_pred - u_tgt.detach()) ** 2
    #     # 这里的 detach() 是防止作弊的核心！
    #     # 让 inv_var 仅作为标量系数，不反传梯度给方差网络来逃避 MSE 惩罚
    #     nll_loss = 0.5 * (inv_var.detach() * mse + log_var)
        
    #     # 2. 方差正则化惩罚 (Variance Penalty)
    #     # 强迫网络保持"自信"！如果 sigma 超过某个阈值，给予额外惩罚
    #     # 这就逼着网络去好好拟合均值，而不是用方差摆烂
    #     var_penalty = 0.01 * torch.exp(log_var)
        
    #     # 3. 最终流匹配 Loss
    #     meanflow_loss = self.nll_weight * (nll_loss + var_penalty).mean()
        
    #     # [完美保留] 原版 Dispersive Loss
    #     dis_loss = 0
    #     if features is not None and len(features) > 0:
    #         if isinstance(features, (list, tuple)):
    #             for feat in features:
    #                 dis_loss += self.dispersive_loss(feat)
    #         else:
    #             dis_loss += self.dispersive_loss(features)
                
    #     loss = meanflow_loss + 0.5 * dis_loss + 5.0
        
    #     loss_dict = {
    #         "loss": loss.item(),
    #         "nll_loss": nll_loss.mean().item(),
    #         "mse_loss": mse.mean().item(), 
    #         "mean_sigma": torch.exp(0.5 * log_var).mean().item(),
    #         "dis_loss": dis_loss.item() if isinstance(dis_loss, torch.Tensor) else dis_loss
    #     }
    #     return loss, loss_dict

    # 创新点2修改，改为“基于不确定性感知的特征对比学习 (Uncertainty-Aware Dispersive Learning)。既然原版 MP1 最得意的操作是 dispersive_loss（让不同的状态特征在空间里互相推开），我们让网络预测的不确定性，去指导这个“推开”的过程。
    # 在特征空间（Latent Space）预测不确定性
    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        batch_size = nactions.shape[0]
        
        # 提取特征
        from mp1.common.pytorch_util import dict_apply
        this_nobs_reshaped = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs_reshaped)
        if "cross_attention" in self.condition_type:
            global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
        else:
            global_cond = nobs_features.reshape(batch_size, -1)

        # 物理采样
        eps = 1e-5
        t = torch.rand((batch_size,), device=self.device).float() * (1 - 2*eps) + eps
        x1 = nactions
        x0 = torch.randn_like(x1)
        t_expand = t.view(batch_size, 1, 1)
        
        xt, mu_t, std_t = self.physics_bridge.sample_path(x0, x1, t_expand)
        u_tgt = self.physics_bridge.compute_target_velocity(x0, x1, t_expand, xt)
        
        # 网络输出
        r_zeros = torch.zeros_like(t)
        v_pred, log_var, features = self.model(sample=xt, timestep=t, global_cond=global_cond, r=r_zeros, training=True)
        
        # ==========================================================
        # [核心 1] 原汁原味的动作匹配 Loss (保证不掉点！)
        # ==========================================================
        mse_raw = (v_pred - u_tgt.detach()) ** 2
        meanflow_loss = mse_raw.mean()
        
        # ==========================================================
        # [核心 2] 原汁原味的特征排斥 Loss (恢复原版 1.0 的比例，修复暴跌 Bug！)
        # ==========================================================
        dis_loss = 0
        if features is not None and len(features) > 0:
            if isinstance(features, (list, tuple)):
                for feat in features:
                    dis_loss += self.dispersive_loss(feat)
            else:
                dis_loss += self.dispersive_loss(features)
                
        # ==========================================================
        # [核心 3] 旁观者方差 Loss (只训练 Uncertainty Head，不传给特征)
        # ==========================================================
        # 提取当前 batch 的均方误差，并 detached 断开反向传播
        mse_per_batch = mse_raw.mean(dim=(1, 2), keepdim=True).detach() # (B, 1)
        
        inv_var = torch.exp(-log_var)
        # 让方差头自己去拟合这个 detached 误差
        var_loss = 0.5 * (inv_var * mse_per_batch + log_var).mean()
        
        # 最终组合损失：互不干扰
        # meanflow_loss + 0.5 * dis_loss 完美等价于纯 PISB
        # self.nll_weight * var_loss 负责独立训练你的雷达
        loss = meanflow_loss + 0.5 * dis_loss + self.nll_weight * var_loss
        
        loss_dict = {
            "loss": loss.item(),
            "meanflow_loss": meanflow_loss.item(), # 这个数值现在可以直接和 PISB 比较了！
            "var_loss": var_loss.item(),
            "mean_sigma": torch.exp(0.5 * log_var).mean().item(),
            "dis_loss": dis_loss.item() if isinstance(dis_loss, torch.Tensor) else dis_loss
        }
        return loss, loss_dict

    def predict_action(self, obs_dict):
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color: nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        B = nobs['point_cloud'].shape[0]
        
        this_nobs_reshaped = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs_reshaped)
        if "cross_attention" in self.condition_type:
            global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
        else:
            global_cond = nobs_features.reshape(B, -1)

        x_current = torch.randn((B, self.horizon, self.action_dim), device=self.device)
        steps = self.num_inference_steps if self.num_inference_steps is not None else 10
        dt = 1.0 / steps
        r_zeros = torch.zeros((B,), device=self.device)

        self.model.eval()
        with torch.no_grad():
            for i in range(steps):
                t_val = i / steps
                t_tensor = torch.full((B,), t_val, device=self.device)
                
                # training=False 时，只返回 v_pred 和 log_var
                v_pred, log_var = self.model(sample=x_current, timestep=t_tensor, global_cond=global_cond, r=r_zeros, training=False)
                
                x_current = x_current + v_pred * dt

        action_pred = self.normalizer['action'].unnormalize(x_current)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        
        return {'action': action, 'action_pred': action_pred}