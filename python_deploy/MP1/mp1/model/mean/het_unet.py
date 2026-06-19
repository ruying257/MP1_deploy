# import torch
# import torch.nn as nn
# from mp1.model.mean.conditional_unet1d_meanflow_dis import ConditionalUnet1D

# class HeteroscedasticUnet(ConditionalUnet1D):
#     def __init__(self, input_dim, het_init_bias=-5.0, log_var_min=-10.0, log_var_max=2.0, **kwargs):
#         # 1. 正常初始化父类，这会构建完整的原版 UNet 结构
#         super().__init__(input_dim=input_dim, **kwargs)
        
#         # 2. 保存我们的异方差专属参数
#         self.input_dim = input_dim
#         self.log_var_min = log_var_min
#         self.log_var_max = log_var_max
        
#         # 3. [核心修复] 原版 self.final_conv 是一个 nn.Sequential(Conv1dBlock, nn.Conv1d)
#         # 我们只替换它的最后一层，保留前面的 Conv1dBlock
#         last_conv = self.final_conv[-1]
#         in_channels = last_conv.in_channels
        
#         # 替换为输出双倍通道（均值 + 方差）的新卷积层
#         new_last_conv = nn.Conv1d(in_channels, input_dim * 2, 1)
        
#         # 鲁棒性初始化
#         with torch.no_grad():
#             new_last_conv.weight.data[:, :, :] *= 0.01
#             new_last_conv.bias.data[:] = 0
#             new_last_conv.bias.data[input_dim:] = het_init_bias
            
#         # 把替换好的层塞回 Sequential
#         self.final_conv[-1] = new_last_conv

#     def forward(self, sample, timestep, global_cond=None, r=None, training=True, **kwargs):
#         """
#         优雅拦截：先让父类跑完所有复杂逻辑（包括特征提取和维度变换），
#         然后我们在最后拦截输出，拆分出均值和方差。
#         """
#         # 调用父类 forward (输出形状已经是 rearrange 好的 b h t)
#         out = super().forward(sample, timestep, global_cond=global_cond, r=r, training=training, **kwargs)
        
#         # 父类在 training=True 时返回 (velocity, features), eval 时只返回 velocity
#         if training:
#             pred, features = out
#         else:
#             pred = out
#             features = None
            
#         # pred 的形状是 (Batch, Horizon, 2 * Action_Dim)
#         # 沿最后一个维度切分出均值和方差
#         mean, log_var = torch.chunk(pred, 2, dim=-1)
        
#         # 数值截断防止 NLL Loss 爆炸
#         log_var = torch.clamp(log_var, min=self.log_var_min, max=self.log_var_max)
        
#         if training:
#             # 训练时返回三元组，完美保留原版 Dispersive Loss 需要的 features
#             return mean, log_var, features
#         else:
#             return mean, log_var

import torch
import torch.nn as nn
from mp1.model.mean.conditional_unet1d_meanflow_dis import ConditionalUnet1D

class HeteroscedasticUnet(ConditionalUnet1D):
    def __init__(self, input_dim, **kwargs):
        # 1. 安全提取我们的自定义参数
        het_init_bias = kwargs.pop('het_init_bias', -5.0)
        self.log_var_min = kwargs.pop('log_var_min', -10.0)
        self.log_var_max = kwargs.pop('log_var_max', 2.0)
        
        # ==========================================
        # [修复报错核心] 直接从 kwargs 里读取 global_cond_dim
        # 我们用 .get() 获取，这样它还会保留在 kwargs 里传给父类
        # ==========================================
        cond_dim = kwargs.get('global_cond_dim', None)
        if cond_dim is None:
            cond_dim = 256  # 如果没传，给个默认后备值
        
        # 2. 调用原版父类初始化
        super().__init__(input_dim=input_dim, **kwargs)
        
        # 3. 增加"不确定性感知头" (Uncertainty Head)
        # 用刚刚截获的 cond_dim 作为输入特征维度
        self.uncertainty_head = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.Mish(),
            nn.Linear(128, 1)  # 输出一个标量，代表当前状态的不确定性 Sigma
        )
        
        # 鲁棒性初始化
        with torch.no_grad():
            self.uncertainty_head[-1].bias.data.fill_(het_init_bias)

    def forward(self, sample, timestep, global_cond=None, r=None, training=True, **kwargs):
        # 1. 调用原版前向传播，拿到 v_pred 和 features
        out = super().forward(sample, timestep, global_cond=global_cond, r=r, training=training, **kwargs)
        
        if training:
            pred, features = out
        else:
            pred = out
            features = None
            
        # 2. 从全局条件中预测当前状态的不确定性
        if global_cond is not None:
            if global_cond.dim() == 3:
                # 如果是序列条件，取时间维度的平均
                cond_feat = global_cond.mean(dim=1)
            else:
                cond_feat = global_cond
        else:
            # Fallback 保护：如果环境不需要 global_cond
            cond_feat = torch.zeros((sample.shape[0], self.uncertainty_head[0].in_features), device=sample.device)
            
        log_var = self.uncertainty_head(cond_feat) # 形状: (B, 1)
        
        # 限制防止数值爆炸
        log_var = torch.clamp(log_var, min=self.log_var_min, max=self.log_var_max)
        
        if training:
            # 训练时返回三元组
            return pred, log_var, features
        else:
            return pred, log_var