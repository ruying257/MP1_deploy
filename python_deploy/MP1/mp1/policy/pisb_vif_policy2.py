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
        self.vif_scale_limit = kwargs.get("vif_scale_limit", 0.15)
        self.vif_residual_weight = kwargs.get("vif_residual_weight", 0.1)

        # 输出两个量:
        # 1) cond_scale: 调制 global_cond
        # 2) dis_gate: 调制 dispersive loss 权重
        self.impedance_head = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)

        # 零初始化，确保初始行为严格退化为 PISB
        with torch.no_grad():
            self.impedance_head[-1].weight.zero_()
            self.impedance_head[-1].bias.zero_()

        cprint("[Stable-VIF] Activated.", "magenta")
        self.residual_head = nn.Sequential(
            nn.Linear(self.action_dim + cond_dim + 1, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, self.action_dim)
        ).to(self.device)

        # with torch.no_grad():
        #     self.residual_head[-1].weight.zero_()
        #     self.residual_head[-1].bias.zero_()

        nn.init.normal_(self.residual_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.residual_head[-1].bias)

        cprint("[Stable-VIF] Physics bridge is fixed; only feature modulation is enabled.", "magenta")

    def _get_vif_factors(self, global_cond):
        if global_cond.dim() == 3:
            pooled = global_cond.mean(dim=1)   # (B, D)
        else:
            pooled = global_cond               # (B, D)

        raw = self.impedance_head(pooled)

        alpha = self.vif_scale_limit * torch.tanh(raw)

        return alpha, pooled

    def _predict_residual_velocity(self, xt, t, pooled_cond):

        B, T, Da = xt.shape

        t_feat = t.view(B,1).expand(B,T).unsqueeze(-1)
        cond_feat = pooled_cond.unsqueeze(1).expand(B,T,-1)

        residual_in = torch.cat([xt, cond_feat, t_feat], dim=-1)

        residual = self.residual_head(residual_in)

        return residual

    def compute_loss(self, batch):
        raw_obs = batch['obs']
        obs_wo_image = {k: v for k, v in raw_obs.items() if k != 'image'}
        nobs = self.normalizer.normalize(obs_wo_image)
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
        alpha, pooled_cond = self._get_vif_factors(global_cond)

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
            global_cond=global_cond,
            r=r_zeros,
            training=True
        )

        if isinstance(model_output, tuple):
            v_base, features = model_output
        else:
            v_base, features = model_output, []
        
        v_residual = torch.tanh(self._predict_residual_velocity(xt, t, pooled_cond))

        v_pred = v_base + self.vif_residual_weight * alpha.unsqueeze(1) * v_residual
        effective_residual = self.vif_residual_weight * alpha.unsqueeze(1) * v_residual

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
        loss = meanflow_loss + 0.5 * dis_loss

        mse_val = (stopgrad(error) ** 2).mean()

        loss_dict = {
            'loss': loss.item(),
            'bc_loss': loss.item(),
            'mse_val': mse_val.item(),
            'meanflow_loss': meanflow_loss.item(),
            'dis_loss': dis_loss.item() if isinstance(dis_loss, torch.Tensor) else dis_loss,
            'vif_alpha_mean': alpha.mean().item(),
            'vif_alpha_abs': alpha.abs().mean().item(),
            'vif_residual_norm': v_residual.norm(dim=-1).mean().item(),
            'effective_residual_norm': effective_residual.norm(dim=-1).mean().item(),
            'v_base_norm': v_base.norm(dim=-1).mean().item()
        }
        return loss, loss_dict