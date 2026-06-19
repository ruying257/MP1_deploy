import math
import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0, f"dim={dim} must be divisible by heads={heads}"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        qkv = self.qkv(x)  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)  # (B, H, T, Dh)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # causal mask: 只看过去和当前
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        attn = attn.masked_fill(causal_mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                                # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, D)      # (B, T, D)
        out = self.out_proj(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads=heads, dropout=dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CausalTemporalEncoder(nn.Module):
    """
    对 observation feature sequence 做因果时序编码。
    输入 / 输出 shape 都是 (B, T, D)
    """
    def __init__(
        self,
        dim: int,
        depth: int = 2,
        heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_seq_len: int = 32,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                heads=heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        assert T <= self.max_seq_len, f"Sequence length T={T} exceeds max_seq_len={self.max_seq_len}"

        x = x + self.pos_embed[:, :T, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x
