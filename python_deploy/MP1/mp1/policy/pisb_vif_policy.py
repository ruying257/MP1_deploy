import sys
sys.path.append('mp1')

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint

from mp1.policy.pisb_policy import PISBPolicy
from mp1.common.pytorch_util import dict_apply
from mp1.policy.pisb_policy import stopgrad, adaptive_l2_loss


class PISBVifPolicy(PISBPolicy):
    """
    稳定版 VIF:
    1) 保持 PISB 的 physics bridge 完全不变
    2) 只让视觉条件去调制 global_cond / v_pred / dis_loss 权重
    3) 不破坏 train-test consistency
    """
    def __init__(self, shape_meta, **kwargs):
        super().__init__(shape_meta, **kwargs)

        cond_dim = self.obs_feature_dim * self.n_obs_steps \
            if "cross_attention" not in self.condition_type else self.obs_feature_dim

        hidden_dim = kwargs.get("vif_hidden_dim", 64)
        self.vif_scale_limit = kwargs.get("vif_scale_limit", 0.15)   # 控制调制幅度
        self.vif_dis_base = kwargs.get("vif_dis_base", 1.0)

        # 输出两个量:
        # 1) cond_scale: 调制 global_cond
        # 2) dis_gate: 调制 dispersive loss 权重
        self.impedance_head = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 2)
        ).to(self.device)

        # 零初始化，确保初始行为严格退化为 PISB
        with torch.no_grad():
            self.impedance_head[-1].weight.zero_()
            self.impedance_head[-1].bias.zero_()

        cprint("[Stable-VIF] Activated.", "magenta")
        cprint("[Stable-VIF] Physics bridge is fixed; only feature modulation is enabled.", "magenta")

    def _get_vif_factors(self, global_cond):
        if global_cond.dim() == 3:
            pooled = global_cond.mean(dim=1)   # (B, D)
        else:
            pooled = global_cond               # (B, D)

        raw = self.impedance_head(pooled)      # (B, 2)

        # 小范围残差调制，初始等于 1
        cond_scale = 1.0 + self.vif_scale_limit * torch.tanh(raw[:, 0:1])
        # 正的 loss gate，范围大致在 [0.5, 1.5]
        dis_gate = self.vif_dis_base + 0.5 * torch.tanh(raw[:, 1:2])

        return cond_scale, dis_gate

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        trajectory = nactions
        device = trajectory.device

        local_cond = None
        global_cond = None

        if self.obs_as_global_cond:
            this_nobs = dict_apply(
                nobs,
                lambda n: n[:, :self.n_obs_steps, ...].reshape(-1, *n.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                global_cond = nobs_features.reshape(batch_size, -1)
        else:
            raise NotImplementedError("Stable-VIF is designed for obs_as_global_cond=True")

        # -------- Stable VIF: 只调制条件特征，不改 physics bridge --------
        cond_scale, dis_gate = self._get_vif_factors(global_cond)

        if global_cond.dim() == 3:
            global_cond_mod = global_cond * cond_scale.unsqueeze(1)
        else:
            global_cond_mod = global_cond * cond_scale

        # -------- 原版 PISB 的固定 physics target --------
        eps = 1e-5
        t = torch.rand((batch_size,), device=device).float() * (1 - 2 * eps) + eps
        x1 = trajectory
        x0 = torch.randn_like(x1)
        t_expand = t.view(batch_size, 1, 1)

        xt, mu_t, std_t = self.physics_bridge.sample_path(x0, x1, t_expand)
        u_tgt = self.physics_bridge.compute_target_velocity(x0, x1, t_expand, xt)

        r_zeros = torch.zeros_like(t)
        model_output = self.model(
            sample=xt,
            timestep=t,
            global_cond=global_cond_mod,
            r=r_zeros,
            training=True
        )

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
                dis_loss = self.dispersive_loss(features)

        # 用 VIF 控制正则强度，而不是控制 target 本身
        dis_gate_mean = dis_gate.mean()
        loss = meanflow_loss + 0.5 * dis_gate_mean * dis_loss

        mse_val = (stopgrad(error) ** 2).mean()

        loss_dict = {
            'loss': loss.item(),
            'bc_loss': loss.item(),
            'mse_val': mse_val.item(),
            'meanflow_loss': meanflow_loss.item(),
            'dis_loss': dis_loss.item() if isinstance(dis_loss, torch.Tensor) else dis_loss,
            'vif_cond_scale': cond_scale.mean().item(),
            'vif_dis_gate': dis_gate_mean.item()
        }
        return loss, loss_dict