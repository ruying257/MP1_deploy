import torch
import torch.nn.functional as F
from termcolor import cprint
from mp1.policy.pisb_het_policy import PISBHetPolicy
from mp1.common.pytorch_util import dict_apply

class PISBHetKalPolicy(PISBHetPolicy):
    """
    [创新点 3] Adaptive Kalman-Guided Composition (A-KGC)
    
    继承自 PISBHetPolicy，训练过程（解耦异方差 Loss）完全一致。
    仅在推理阶段 (predict_action) 引入自适应卡尔曼增益和保守安全势场 (Conservative Safety Potential)。
    当网络输出高方差时，自主触发物理阻尼和平滑约束。
    """
    def __init__(self, shape_meta, **kwargs):
        # 1. 拦截并提取卡尔曼滤波与势场专属的超参数
        self.R_phys = kwargs.pop('R_phys', 0.1)             # 观测噪声协方差 (对物理公式的信任度)
        self.lambda_safe = kwargs.pop('lambda_safe', 0.5)   # 物理修正的整体缩放强度
        self.w_damping = kwargs.pop('w_damping', 0.05)      # 减速惩罚权重 (限制绝对速度)
        self.w_smooth = kwargs.pop('w_smooth', 1.0)         # 平滑惩罚权重 (防止动作高频抖动)
        
        # 2. 调用父类 (PISBHetPolicy) 完成网络初始化
        super().__init__(shape_meta, **kwargs)
        
        cprint(f"[PISB-Het-Kal] Adaptive Kalman Safety Guidance Activated.", "green")
        cprint(f"  -> R_phys: {self.R_phys}, lambda_safe: {self.lambda_safe}", "green")

    # 注意：我们不需要重写 compute_loss！直接完美继承父类的解耦训练逻辑！

    def predict_action(self, obs_dict):
        """
        [创新点 3 完美版] A-KGC: 1-step MAP Inference
        """
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color: nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        B = nobs['point_cloud'].shape[0]
        
        from mp1.common.pytorch_util import dict_apply
        this_nobs_reshaped = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs_reshaped)
        if "cross_attention" in self.condition_type:
            global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
        else:
            global_cond = nobs_features.reshape(B, -1)

        # 初始高斯噪声
        x_noise = torch.randn((B, self.horizon, self.action_dim), device=self.device)
        
        # [修复 Bug 1] 强制继承 baseline 的 1 步预测
        steps = self.num_inference_steps if self.num_inference_steps is not None else 1
        dt = 1.0 / steps
        r_zeros = torch.zeros((B,), device=self.device)

        self.model.eval()
        x_current = x_noise
        
        for i in range(steps):
            t_val = i / steps
            t_tensor = torch.full((B,), t_val, device=self.device)
            
            # 【步骤 A：神经网络输出先验预测】
            with torch.no_grad():
                v_pred, log_var = self.model(sample=x_current, timestep=t_tensor, global_cond=global_cond, r=r_zeros, training=False)
                
                # [修复 Bug 2 核心] 计算出网络建议的"干净物理动作" (Prior Mean)
                x_clean_prior = x_current + v_pred * dt

            # 【步骤 B：物理安全评估 (仅在干净动作上评估)】
            with torch.enable_grad():
                # 让系统对预测出的动作求导
                x_clean_eval = x_clean_prior.detach().requires_grad_(True)
                
                # 计算动作轨迹的速度和加速度
                action_diff1 = torch.diff(x_clean_eval, dim=1) 
                action_diff2 = torch.diff(action_diff1, dim=1) 
                
                E_damping = torch.sum(action_diff1 ** 2) 
                E_smooth = torch.sum(action_diff2 ** 2)
                # 任务无关的安全惩罚能量场
                E_safe = self.w_damping * E_damping + self.w_smooth * E_smooth
                
                # 计算修正梯度 (向着最平滑、最安全的方向)
                score_safe = -torch.autograd.grad(E_safe, x_clean_eval)[0]
                score_safe = torch.clamp(score_safe, min=-5.0, max=5.0)

            # 【步骤 C：自适应卡尔曼融合】
            with torch.no_grad():
                sigma_prior = torch.exp(log_var)
                
                # K = Sigma / (Sigma + R)
                kalman_gain = sigma_prior / (sigma_prior + self.R_phys)
                
                # 修复维度报错: (B, 1) -> (B, 1, 1)，完美对齐 (B, Horizon, Action_dim)
                kalman_gain = kalman_gain.unsqueeze(-1) 
                
                # MAP 更新: 最终动作 = 预测动作 + 卡尔曼权重 * 物理安全修正
                x_composed = x_clean_prior + self.lambda_safe * kalman_gain * score_safe
                
                # 准备进入下一步（如果是多步的话，但这里是1步，所以直接输出）
                x_current = x_composed

        action_pred = self.normalizer['action'].unnormalize(x_current)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        
        return {'action': action, 'action_pred': action_pred}