
from typing import Optional, Tuple
import torch
import torch.nn as nn
from mamba_ssm import Mamba


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    mask = mask.float().unsqueeze(-1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (x * mask).sum(dim=1) / denom


def masked_last(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x[:, -1]
    lengths = mask.long().sum(dim=1).clamp(min=1) - 1
    b_idx = torch.arange(x.shape[0], device=x.device)
    return x[b_idx, lengths]


class ResidualMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mamba(self.norm(x)))


class HistoryBranchEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        depth: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        max_seq_len: int,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.SiLU(),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, model_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            ResidualMambaBlock(
                d_model=model_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout
            ) for _ in range(depth)
        ])
        self.out_norm = nn.LayerNorm(model_dim)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        if t > self.max_seq_len:
            raise ValueError(f"History length {t} exceeds max_seq_len={self.max_seq_len}")
        h = self.in_proj(x)
        h = h + self.pos_embed[:, :t, :]
        for blk in self.blocks:
            h = blk(h)
        return self.out_norm(h)


class ActionHistoryFusionMamba(nn.Module):
    """
    Two-branch history encoder:
      - raw action history
      - action delta history

    Returns:
      fused_context: (B, C)
      aux: dict with intermediate tensors for regularization/debug
    """
    def __init__(
        self,
        action_dim: int,
        model_dim: int = 128,
        depth: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        max_seq_len: int = 32,
        fusion_dim: int = 256,
    ):
        super().__init__()
        self.raw_encoder = HistoryBranchEncoder(
            in_dim=action_dim,
            model_dim=model_dim,
            depth=depth,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.delta_encoder = HistoryBranchEncoder(
            in_dim=action_dim,
            model_dim=model_dim,
            depth=depth,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.fuse = nn.Sequential(
            nn.Linear(model_dim * 4, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        action_history: torch.Tensor,
        action_delta: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        raw_seq = self.raw_encoder(action_history)
        delta_seq = self.delta_encoder(action_delta)

        raw_last = masked_last(raw_seq, mask)
        raw_mean = masked_mean(raw_seq, mask)
        delta_last = masked_last(delta_seq, mask)
        delta_mean = masked_mean(delta_seq, mask)

        fused_in = torch.cat([raw_last, raw_mean, delta_last, delta_mean], dim=-1)
        fused_context = self.fuse(fused_in)

        aux = {
            "raw_last": raw_last,
            "raw_mean": raw_mean,
            "delta_last": delta_last,
            "delta_mean": delta_mean,
        }
        return fused_context, aux
