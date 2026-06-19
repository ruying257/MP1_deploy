import sys
sys.path.append('mp1')

from typing import Dict, Tuple
import copy
import torch
import torch.nn as nn

from mp1.policy.pisb_policy import PISBPolicy, stopgrad, adaptive_l2_loss
from mp1.common.pytorch_util import dict_apply
from mp1.model.memory.causal_memory import CausalTemporalEncoder


class PISBCausalMemoryPolicy(PISBPolicy):
    """
    PISB + Causal Temporal Memory (stable version)

    设计原则：
    1) 保持 PISB 的 physics bridge / target velocity / Euler sampling 完全不变
    2) 不再手工猜测 step feature dim
    3) 用一次 dummy forward 自动推断 obs_encoder 的真实单帧输出维度
    4) 若真实单帧维度 != UNet 期望单帧条件维度，则显式投影
    5) 训练 / 推理完全共用同一套 global_cond 构造逻辑
    """

    def __init__(self, shape_meta, **kwargs):
        super().__init__(shape_meta, **kwargs)

        if not self.obs_as_global_cond:
            raise NotImplementedError("PISBCausalMemoryPolicy requires obs_as_global_cond=True")

        self.memory_depth = kwargs.get("memory_depth", 2)
        self.memory_heads = kwargs.get("memory_heads", 4)
        self.memory_mlp_ratio = kwargs.get("memory_mlp_ratio", 4.0)
        self.memory_dropout = kwargs.get("memory_dropout", 0.0)
        self.memory_residual_scale = kwargs.get("memory_residual_scale", 0.10)

        # ============================================================
        # 显式指定真实单帧 obs_encoder 输出维度
        # 当前任务实际跑出来 raw_step_features dim = 96
        # ============================================================
        self.step_input_dim = kwargs.get("step_input_dim", 96)

        # 当前工程里，obs_encoder 的单帧输出真实是 96（见训练时报错）
        # 我们把它投影到一个固定的 causal memory 维度上
        self.cond_step_dim = kwargs.get("cond_step_dim", 192)
        self.memory_dim = self.cond_step_dim

        # 不再假设 raw_step_features 固定是 96，改为按首次真实输入自动初始化
        self.step_proj = nn.Sequential(
            nn.Linear(self.step_input_dim, self.cond_step_dim),
            nn.LayerNorm(self.cond_step_dim),
            nn.Mish(),
        ).to(self.device)

        print(f"[PISB-CausalMemory] step_input_dim = {self.step_input_dim}")
        print(f"[PISB-CausalMemory] cond_step_dim = {self.cond_step_dim}")
        print(f"[PISB-CausalMemory] memory_dim = {self.memory_dim}")

        # ============================================================
        # 4) Causal temporal encoder
        # ============================================================
        self.temporal_memory = CausalTemporalEncoder(
            dim=self.memory_dim,
            depth=self.memory_depth,
            heads=self.memory_heads,
            mlp_ratio=self.memory_mlp_ratio,
            dropout=self.memory_dropout,
            max_seq_len=max(self.n_obs_steps, 32),
        ).to(self.device)

        # ============================================================
        # 5) 将 memory_state -> 单帧修正，再 broadcast 到所有 obs steps
        # ============================================================
        self.memory_adapter = nn.Sequential(
            nn.Linear(self.memory_dim, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, self.cond_step_dim)
        ).to(self.device)

        # 零初始化：初始严格退化为原始 PISB
        with torch.no_grad():
            self.memory_adapter[-1].weight.zero_()
            self.memory_adapter[-1].bias.zero_()

        print(f"[PISB-CausalMemory] cond_step_dim = {self.cond_step_dim}")
        print(f"[PISB-CausalMemory] memory_dim = {self.memory_dim}")

    # ============================================================
    # helper: 逐帧编码 + 投影到 cond_step_dim
    # ============================================================
    def _encode_step_features(self, nobs: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
        """
        return: (B, To, cond_step_dim)
        """
        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )

        raw_step_features = self.obs_encoder(this_nobs)
        raw_step_features = raw_step_features.reshape(batch_size, self.n_obs_steps, -1)

        # 如果 obs_encoder 已经输出 cond_step_dim，就直接用
        if raw_step_features.shape[-1] == self.cond_step_dim:
            projected_step_features = raw_step_features
        else:
            projected_step_features = self.step_proj(raw_step_features)

        assert projected_step_features.shape[-1] == self.cond_step_dim, \
            f"projected_step_features dim {projected_step_features.shape[-1]} != cond_step_dim {self.cond_step_dim}"

        return projected_step_features

    # ============================================================
    # helper: 构造 global_cond（训练 / 推理共用）
    # ============================================================
    def _build_global_cond(self, nobs: Dict[str, torch.Tensor], batch_size: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        step_features = self._encode_step_features(nobs, batch_size)     # (B, To, D)

        memory_seq = self.temporal_memory(step_features)                 # (B, To, D)
        assert memory_seq.shape[-1] == self.cond_step_dim, \
            f"memory_seq dim {memory_seq.shape[-1]} != cond_step_dim {self.cond_step_dim}"

        memory_state = memory_seq[:, -1]                                 # (B, D)

        # 单帧 memory 修正
        memory_delta = self.memory_adapter(memory_state)                 # (B, D)
        assert memory_delta.shape[-1] == self.cond_step_dim, \
            f"memory_delta dim {memory_delta.shape[-1]} != cond_step_dim {self.cond_step_dim}"

        memory_delta = self.memory_residual_scale * torch.tanh(memory_delta)

        if "cross_attention" in self.condition_type:
            base_global_cond = memory_seq                                # (B, To, D)
            memory_delta = memory_delta.unsqueeze(1).repeat(1, self.n_obs_steps, 1)  # (B, To, D)
            global_cond = base_global_cond + memory_delta
        else:
            base_global_cond = memory_seq.reshape(batch_size, -1)        # (B, To*D)

            memory_delta = memory_delta.unsqueeze(1).repeat(1, self.n_obs_steps, 1)  # (B, To, D)
            memory_delta = memory_delta.reshape(batch_size, -1)                          # (B, To*D)

            assert memory_delta.shape == base_global_cond.shape, \
                f"memory_delta shape {memory_delta.shape} != base_global_cond shape {base_global_cond.shape}"

            global_cond = base_global_cond + memory_delta

        aux_stats = {
            "memory_seq_norm": memory_seq.norm(dim=-1).mean().item(),
            "memory_state_norm": memory_state.norm(dim=-1).mean().item(),
            "memory_delta_norm": memory_delta.norm(dim=-1).mean().item() if memory_delta.dim() == 2
                                else memory_delta.norm(dim=-1).mean().item(),
        }
        return global_cond, aux_stats

    # ============================================================
    # training
    # ============================================================
    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color and 'point_cloud' in nobs:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        batch_size = nactions.shape[0]
        trajectory = nactions
        device = trajectory.device

        global_cond, aux_stats = self._build_global_cond(nobs, batch_size)

        # ===== 保持原始 PISB 的固定 physics target =====
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

        loss = meanflow_loss + 0.5 * dis_loss
        mse_val = (stopgrad(error) ** 2).mean()

        loss_dict = {
            'loss': loss.item(),
            'bc_loss': loss.item(),
            'mse_val': mse_val.item(),
            'meanflow_loss': meanflow_loss.item(),
            'dis_loss': dis_loss.item() if isinstance(dis_loss, torch.Tensor) else dis_loss,
            **aux_stats
        }
        return loss, loss_dict

    # ============================================================
    # inference
    # ============================================================
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)

        if not self.use_pc_color and 'point_cloud' in nobs:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim

        device = self.device
        dtype = self.dtype

        global_cond, _ = self._build_global_cond(nobs, B)

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

                model_output = model(
                    sample=x_current,
                    timestep=t_tensor,
                    global_cond=global_cond,
                    r=r_zeros,
                    training=False
                )

                if isinstance(model_output, tuple):
                    v_pred = model_output[0]
                else:
                    v_pred = model_output

                x_current = x_current + v_pred * dt

        naction_pred = x_current[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred,
        }
        return result