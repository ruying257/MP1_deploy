
import sys
sys.path.append('mp1')

from collections import deque
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from mp1.common.pytorch_util import dict_apply
from mp1.policy.pisb_policy import PISBPolicy, adaptive_l2_loss, stopgrad
from mp1.model.common.action_history_mamba_pro import ActionHistoryFusionMamba


class PISBActionHistoryMambaProPolicy(PISBPolicy):
    """
    Stronger practical variant:
      - keeps original PISB bridge / target velocity / Euler inference
      - adds action-history conditioner with:
          * Mamba raw-history branch
          * Mamba delta-history branch
          * observation-aware fusion gate
          * zero-init bounded FiLM-style adapter on global_cond
          * history dropout + adapter regularization
      - inference maintains rolling executed-action buffer internally
    """
    def __init__(
        self,
        *args,
        use_action_history: bool = True,
        action_history_len: int = 8,
        history_model_dim: int = 128,
        history_depth: int = 2,
        history_d_state: int = 16,
        history_d_conv: int = 4,
        history_expand: int = 2,
        history_dropout: float = 0.1,
        history_fusion_dim: int = 256,
        history_dropout_prob: float = 0.10,
        history_cond_scale: float = 0.05,
        history_gate_scale: float = 0.05,
        history_reg_weight: float = 1.0e-4,
        adapter_hidden_dim: int = 256,
        max_history_len: Optional[int] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.use_action_history = bool(use_action_history)
        self.action_history_len = int(action_history_len)
        self.history_dropout_prob = float(history_dropout_prob)
        self.history_cond_scale = float(history_cond_scale)
        self.history_gate_scale = float(history_gate_scale)
        self.history_reg_weight = float(history_reg_weight)
        self.max_history_len = int(max_history_len or action_history_len)

        step_cond_dim = self.obs_feature_dim
        self.history_encoder = ActionHistoryFusionMamba(
            action_dim=self.action_dim,
            model_dim=history_model_dim,
            depth=history_depth,
            d_state=history_d_state,
            d_conv=history_d_conv,
            expand=history_expand,
            dropout=history_dropout,
            max_seq_len=self.max_history_len,
            fusion_dim=history_fusion_dim,
        )

        self.obs_summary_proj = nn.Sequential(
            nn.Linear(self.obs_feature_dim, adapter_hidden_dim),
            nn.LayerNorm(adapter_hidden_dim),
            nn.SiLU(),
        )
        self.history_summary_proj = nn.Sequential(
            nn.Linear(history_fusion_dim, adapter_hidden_dim),
            nn.LayerNorm(adapter_hidden_dim),
            nn.SiLU(),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(adapter_hidden_dim * 2, adapter_hidden_dim),
            nn.LayerNorm(adapter_hidden_dim),
            nn.SiLU(),
        )
        self.delta_adapter = nn.Sequential(
            nn.Linear(adapter_hidden_dim, adapter_hidden_dim),
            nn.SiLU(),
            nn.Linear(adapter_hidden_dim, step_cond_dim)
        )
        self.gate_adapter = nn.Sequential(
            nn.Linear(adapter_hidden_dim, adapter_hidden_dim),
            nn.SiLU(),
            nn.Linear(adapter_hidden_dim, step_cond_dim)
        )

        # zero-init last layer => starts from exact PISB
        nn.init.zeros_(self.delta_adapter[-1].weight)
        nn.init.zeros_(self.delta_adapter[-1].bias)
        nn.init.zeros_(self.gate_adapter[-1].weight)
        nn.init.zeros_(self.gate_adapter[-1].bias)

        self._history_buffer = deque(maxlen=self.max_history_len)

    def reset(self):
        self._history_buffer.clear()

    def _split_obs_and_top_level_history(self, raw_input: Dict[str, torch.Tensor]):
        obs_dict = raw_input.get('obs', raw_input)
        top_history = raw_input.get('action_history', None)
        top_mask = raw_input.get('action_history_mask', None)
        top_delta = raw_input.get('action_history_delta', None)
        return obs_dict, top_history, top_mask, top_delta

    def _encode_obs(self, raw_obs: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        nobs = self.normalizer.normalize(raw_obs)
        if not self.use_pc_color and 'point_cloud' in nobs:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        B = next(iter(nobs.values())).shape[0]
        this_nobs = dict_apply(
            nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(B, self.n_obs_steps, -1)
        return nobs, nobs_features

    def _prepare_history_inputs(
        self,
        batch_size: int,
        device: torch.device,
        training: bool,
        action_history: Optional[torch.Tensor],
        action_history_mask: Optional[torch.Tensor],
        action_history_delta: Optional[torch.Tensor],
    ):
        if action_history is None:
            # inference fallback: use internal buffer
            if len(self._history_buffer) == 0:
                hist = torch.zeros(batch_size, self.action_history_len, self.action_dim, device=device)
                mask = torch.zeros(batch_size, self.action_history_len, device=device)
            else:
                hist_tensor = torch.stack(list(self._history_buffer), dim=0).to(device=device)
                hist_tensor = hist_tensor[-self.action_history_len:]
                valid = hist_tensor.shape[0]
                if valid < self.action_history_len:
                    pad = torch.zeros(self.action_history_len - valid, self.action_dim, device=device, dtype=hist_tensor.dtype)
                    hist_tensor = torch.cat([pad, hist_tensor], dim=0)
                hist = hist_tensor.unsqueeze(0).repeat(batch_size, 1, 1)
                mask = torch.zeros(batch_size, self.action_history_len, device=device, dtype=hist.dtype)
                mask[:, -valid:] = 1.0
        else:
            hist = action_history.to(device=device)
            mask = action_history_mask.to(device=device) if action_history_mask is not None \
                else torch.ones(hist.shape[:2], device=device, dtype=hist.dtype)

        if action_history_delta is None:
            delta = torch.zeros_like(hist)
            delta[:, 1:] = hist[:, 1:] - hist[:, :-1]
            delta = delta * mask.unsqueeze(-1)
        else:
            delta = action_history_delta.to(device=device)

        # normalize with action normalizer
        hist = self.normalizer['action'].normalize(hist)
        # delta scaled by action normalizer statistics indirectly via normalize difference
        delta = torch.zeros_like(hist)
        delta[:, 1:] = hist[:, 1:] - hist[:, :-1]
        delta = delta * mask.unsqueeze(-1)

        if training and self.history_dropout_prob > 0:
            keep = (torch.rand(batch_size, device=device) >= self.history_dropout_prob).float().unsqueeze(-1)
            hist = hist * keep.unsqueeze(-1)
            delta = delta * keep.unsqueeze(-1)
            mask = mask * keep

        return hist, mask, delta

    def _build_history_condition(
        self,
        raw_input: Dict[str, torch.Tensor],
        training: bool,
    ) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        raw_obs, top_history, top_mask, top_delta = self._split_obs_and_top_level_history(raw_input)
        nobs, nobs_features = self._encode_obs(raw_obs)
        B = nobs_features.shape[0]
        device = nobs_features.device

        if 'cross_attention' in self.condition_type:
            base_global_cond = nobs_features
        else:
            base_global_cond = nobs_features.reshape(B, -1)

        if not self.use_action_history:
            return base_global_cond, {}, torch.zeros((), device=device)

        hist, mask, delta = self._prepare_history_inputs(
            batch_size=B,
            device=device,
            training=training,
            action_history=top_history,
            action_history_mask=top_mask,
            action_history_delta=top_delta,
        )

        history_context, aux = self.history_encoder(hist, delta, mask=mask)
        obs_summary = nobs_features.mean(dim=1)

        fused = self.fusion_mlp(torch.cat([
            self.history_summary_proj(history_context),
            self.obs_summary_proj(obs_summary)
        ], dim=-1))

        delta_step = self.history_cond_scale * torch.tanh(self.delta_adapter(fused))
        gate_step = 1.0 + self.history_gate_scale * torch.tanh(self.gate_adapter(fused))

        if 'cross_attention' in self.condition_type:
            delta_full = delta_step.unsqueeze(1).expand(-1, self.n_obs_steps, -1)
            gate_full = gate_step.unsqueeze(1).expand(-1, self.n_obs_steps, -1)
            cond = nobs_features * gate_full + delta_full
        else:
            cond_step = nobs_features * gate_step.unsqueeze(1) + delta_step.unsqueeze(1)
            cond = cond_step.reshape(B, -1)

        reg = delta_step.pow(2).mean() + (gate_step - 1.0).pow(2).mean() + history_context.pow(2).mean() * 0.1
        stats = {
            'history_ctx_norm': history_context.norm(dim=-1).mean().item(),
            'history_delta_norm': delta_step.norm(dim=-1).mean().item(),
            'history_gate_abs': (gate_step - 1.0).abs().mean().item(),
            'history_valid_tokens': mask.sum(dim=-1).float().mean().item(),
        }
        return cond, stats, reg

    def compute_loss(self, batch):
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        device = nactions.device

        global_cond, aux_stats, history_reg = self._build_history_condition(batch, training=True)

        trajectory = nactions
        eps = 1e-5
        t = torch.rand((batch_size,), device=device).float() * (1 - 2 * eps) + eps
        x1 = trajectory
        x0 = torch.randn_like(x1)
        t_expand = t.view(batch_size, 1, 1)

        xt, _, _ = self.physics_bridge.sample_path(x0, x1, t_expand)
        u_tgt = self.physics_bridge.compute_target_velocity(x0, x1, t_expand, xt)

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
                dis_loss = self.dispersive_loss(features)

        loss = meanflow_loss + 0.5 * dis_loss + self.history_reg_weight * history_reg
        mse_val = (stopgrad(error) ** 2).mean()
        loss_dict = {
            'loss': loss.item(),
            'bc_loss': loss.item(),
            'mse_val': mse_val.item(),
            'meanflow_loss': meanflow_loss.item(),
            'dis_loss': dis_loss.item() if isinstance(dis_loss, torch.Tensor) else dis_loss,
            'history_reg': history_reg.item(),
            **aux_stats,
        }
        return loss, loss_dict

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs_only, _, _, _ = self._split_obs_and_top_level_history(obs_dict)
        B = next(iter(obs_only.values())).shape[0]
        T = self.horizon
        Da = self.action_dim
        device = self.device
        dtype = self.dtype

        global_cond, _, _ = self._build_history_condition(obs_dict, training=False)
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)

        model = self.model
        model.eval()
        x_current = torch.randn(size=cond_data.shape, dtype=cond_data.dtype, device=cond_data.device)
        steps = self.num_inference_steps if self.num_inference_steps is not None else 10
        dt = 1.0 / steps
        r_zeros = torch.zeros((B,), device=device)

        for i in range(steps):
            t_val = i / steps
            t_tensor = torch.full((B,), t_val, device=device, dtype=dtype)
            model_output = model(
                sample=x_current,
                timestep=t_tensor,
                global_cond=global_cond,
                r=r_zeros,
                training=False
            )
            v_pred = model_output[0] if isinstance(model_output, tuple) else model_output
            x_current = x_current + v_pred * dt

        naction_pred = x_current[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        # update internal history buffer using executed chunk prediction
        if B == 1:
            for a in action[0]:
                self._history_buffer.append(a.detach().to(device='cpu'))

        result = {
            'action': action,
            'action_pred': action_pred,
        }
        return result
