import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryPositionEmbedding(nn.Module):
    """旋转位置编码 (RoPE)，增强相对时间感知"""
    def __init__(self, dim, max_seq_len=100):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        # 复制以匹配维度 (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.unsqueeze(0) # (1, seq_len, dim)

def apply_rotary_pos_emb(x, sin, cos):
    # x shape: (B, num_heads, seq_len, head_dim)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    # Rotate half
    x_rot = torch.cat([-x2, x1], dim=-1)
    return (x * cos) + (x_rot * sin)

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, rope_emb=None):
        B, T, C = x.shape
        # 计算 QKV 并拆分多头
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, num_heads, T, head_dim)

        if rope_emb is not None:
            # 应用 RoPE
            cos = rope_emb.cos().unsqueeze(1) # (1, 1, T, C)
            sin = rope_emb.sin().unsqueeze(1)
            q = apply_rotary_pos_emb(q, sin, cos)
            k = apply_rotary_pos_emb(k, sin, cos)

        # 因果掩码机制 (Causal Mask)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        attn = attn.masked_fill(causal_mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        y = attn @ v # (B, num_heads, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

class GatedResidualBlock(nn.Module):
    """门控残差网络 (GRN) - 涨点 Trick，控制记忆与当前观测的融合比例"""
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.gate = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        h = F.gelu(self.fc1(self.ln(x)))
        h = self.fc2(h)
        gate = torch.sigmoid(self.gate(x))
        return x + gate * h

class CausalMemoryNetwork(nn.Module):
    def __init__(self, feature_dim, num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.rope = RotaryPositionEmbedding(feature_dim // num_heads)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'ln_1': nn.LayerNorm(feature_dim),
                'attn': CausalSelfAttention(feature_dim, num_heads, dropout),
                'grn': GatedResidualBlock(feature_dim)
            }) for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(feature_dim)
        
        # Attention Pooling (用于压缩全局特征)
        self.pool_q = nn.Parameter(torch.randn(1, 1, feature_dim))

    def forward(self, obs_features):
        """
        obs_features: (B, T, D) 过去的观测序列
        返回: 具有因果感知的更新特征
        """
        B, T, D = obs_features.shape
        x = obs_features
        rope_emb = self.rope(T, x.device)

        for layer in self.layers:
            # Self Attention with Causal Mask
            x = x + layer['attn'](layer['ln_1'](x), rope_emb)
            # Gated FFN
            x = layer['grn'](x)
            
        x = self.ln_f(x)
        return x

    def get_pooled_memory(self, causal_features):
        """将序列因果记忆池化为单一全局向量"""
        B, T, D = causal_features.shape
        q = self.pool_q.expand(B, -1, -1) # (B, 1, D)
        k = v = causal_features # (B, T, D)
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D))
        attn = F.softmax(attn, dim=-1)
        pooled = (attn @ v).squeeze(1) # (B, D)
        return pooled