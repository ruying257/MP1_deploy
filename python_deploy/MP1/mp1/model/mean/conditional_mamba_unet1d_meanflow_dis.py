import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Official package preferred
    from mamba_ssm import Mamba2 as _OfficialMamba
except Exception:
    try:
        from mamba_ssm import Mamba as _OfficialMamba
    except Exception:
        _OfficialMamba = None


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class _FallbackMamba(nn.Module):
    """
    Fallback only if mamba_ssm is not installed. This keeps the file importable,
    but for best results please install official mamba-ssm.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        hidden = d_model * expand
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.dwconv = nn.Conv1d(d_model, d_model, kernel_size=d_conv, padding=d_conv - 1, groups=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dwconv(x.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        return self.net(x + y)


class MambaLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if _OfficialMamba is None:
            self.layer = _FallbackMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            # Support both Mamba and Mamba2 signatures.
            try:
                self.layer = _OfficialMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            except TypeError:
                self.layer = _OfficialMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class ConvNormAct1d(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 5, n_groups: int = 8):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(num_groups=min(n_groups, out_dim), num_channels=out_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MambaResBlock1D(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        cond_dim: int,
        kernel_size: int = 5,
        n_groups: int = 8,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv1 = ConvNormAct1d(in_dim, out_dim, kernel_size=kernel_size, n_groups=n_groups)
        self.conv2 = ConvNormAct1d(out_dim, out_dim, kernel_size=kernel_size, n_groups=n_groups)
        self.film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_dim * 4)
        )
        self.mamba_norm = nn.LayerNorm(out_dim)
        self.mamba = MambaLayer(d_model=out_dim, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Identity() if in_dim == out_dim else nn.Conv1d(in_dim, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T), cond: (B, D)
        scale1, shift1, scale2, shift2 = self.film(cond).chunk(4, dim=-1)

        h = self.conv1(x)
        h = h * (1 + scale1.unsqueeze(-1)) + shift1.unsqueeze(-1)

        h = self.conv2(h)
        h = h * (1 + scale2.unsqueeze(-1)) + shift2.unsqueeze(-1)
        h = self.dropout(h)

        # Mamba expects (B, T, C)
        hm = h.transpose(1, 2)
        hm = self.mamba_norm(hm)
        hm = self.mamba(hm)
        h = h + hm.transpose(1, 2)

        return h + self.residual(x)


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.op = nn.Conv1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] <= 1:
            return x
        return self.op(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.op = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, target_len: Optional[int] = None) -> torch.Tensor:
        if target_len is None or x.shape[-1] > 1:
            x = self.op(x)
        if target_len is not None and x.shape[-1] != target_len:
            x = F.interpolate(x, size=target_len, mode='nearest')
        return x


class ConditionalMambaUnet1D(nn.Module):
    """
    Lightweight Conditional 1D U-Net with official mamba-ssm blocks.
    Interface is aligned with the existing ConditionalUnet1D used by PISB.
    Returns (prediction, feature_list) for compatibility with dispersive loss.
    """
    def __init__(
        self,
        input_dim: int,
        local_cond_dim: Optional[int] = None,
        global_cond_dim: Optional[int] = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: Tuple[int, ...] = (256, 512, 512),
        kernel_size: int = 5,
        n_groups: int = 8,
        condition_type: str = 'film',
        use_down_condition: bool = True,
        use_mid_condition: bool = True,
        use_up_condition: bool = True,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        if local_cond_dim is not None:
            raise NotImplementedError('ConditionalMambaUnet1D does not use local_cond.')
        if global_cond_dim is None:
            raise ValueError('ConditionalMambaUnet1D requires global_cond_dim.')
        if 'cross_attention' in condition_type:
            raise NotImplementedError('ConditionalMambaUnet1D currently supports film/global conditioning only.')

        self.input_dim = input_dim
        self.global_cond_dim = global_cond_dim
        self.condition_type = condition_type
        self.use_down_condition = use_down_condition
        self.use_mid_condition = use_mid_condition
        self.use_up_condition = use_up_condition

        dims = list(down_dims)
        assert len(dims) >= 2, 'down_dims must have at least 2 stages.'

        self.time_emb = SinusoidalPosEmb(diffusion_step_embed_dim)
        self.r_emb = SinusoidalPosEmb(diffusion_step_embed_dim)
        cond_in_dim = diffusion_step_embed_dim * 2 + global_cond_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in_dim, dims[0] * 4),
            nn.SiLU(),
            nn.Linear(dims[0] * 4, dims[0] * 4),
            nn.SiLU(),
            nn.Linear(dims[0] * 4, dims[0]),
        )

        self.init_conv = nn.Conv1d(input_dim, dims[0], kernel_size=kernel_size, padding=kernel_size // 2)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        curr_dim = dims[0]
        for next_dim in dims:
            block1 = MambaResBlock1D(
                in_dim=curr_dim,
                out_dim=next_dim,
                cond_dim=dims[0],
                kernel_size=kernel_size,
                n_groups=n_groups,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                dropout=mamba_dropout,
            )
            block2 = MambaResBlock1D(
                in_dim=next_dim,
                out_dim=next_dim,
                cond_dim=dims[0],
                kernel_size=kernel_size,
                n_groups=n_groups,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                dropout=mamba_dropout,
            )
            self.down_blocks.append(nn.ModuleList([block1, block2]))
            self.downsamples.append(Downsample1d(next_dim))
            curr_dim = next_dim

        mid_dim = dims[-1]
        self.mid_block1 = MambaResBlock1D(
            in_dim=mid_dim,
            out_dim=mid_dim,
            cond_dim=dims[0],
            kernel_size=kernel_size,
            n_groups=n_groups,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            dropout=mamba_dropout,
        )
        self.mid_block2 = MambaResBlock1D(
            in_dim=mid_dim,
            out_dim=mid_dim,
            cond_dim=dims[0],
            kernel_size=kernel_size,
            n_groups=n_groups,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            dropout=mamba_dropout,
        )

        rev_dims = list(reversed(dims))
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        curr_dim = rev_dims[0]
        for skip_dim in rev_dims[1:]:
            self.upsamples.append(Upsample1d(curr_dim))
            block1 = MambaResBlock1D(
                in_dim=curr_dim + skip_dim,
                out_dim=skip_dim,
                cond_dim=dims[0],
                kernel_size=kernel_size,
                n_groups=n_groups,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                dropout=mamba_dropout,
            )
            block2 = MambaResBlock1D(
                in_dim=skip_dim,
                out_dim=skip_dim,
                cond_dim=dims[0],
                kernel_size=kernel_size,
                n_groups=n_groups,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                dropout=mamba_dropout,
            )
            self.up_blocks.append(nn.ModuleList([block1, block2]))
            curr_dim = skip_dim

        self.final_block = MambaResBlock1D(
            in_dim=dims[0] * 2,
            out_dim=dims[0],
            cond_dim=dims[0],
            kernel_size=kernel_size,
            n_groups=n_groups,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            dropout=mamba_dropout,
        )
        self.final_conv = nn.Conv1d(dims[0], input_dim, kernel_size=1)

    def _build_cond(self, timestep: torch.Tensor, global_cond: torch.Tensor, r: Optional[torch.Tensor]) -> torch.Tensor:
        if r is None:
            r = torch.zeros_like(timestep)
        t_emb = self.time_emb(timestep)
        r_emb = self.r_emb(r)
        cond = torch.cat([t_emb, r_emb, global_cond], dim=-1)
        return self.cond_mlp(cond)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        r: Optional[torch.Tensor] = None,
        training: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if local_cond is not None:
            raise NotImplementedError('ConditionalMambaUnet1D does not use local_cond.')
        if global_cond is None:
            raise ValueError('global_cond is required.')
        if global_cond.dim() == 3:
            raise NotImplementedError('cross-attention style global_cond is not supported.')

        x = sample.transpose(1, 2)  # (B, Da, T) -> (B, C, T)
        x0 = self.init_conv(x)
        cond = self._build_cond(timestep=timestep, global_cond=global_cond, r=r)

        features: List[torch.Tensor] = []
        skips: List[torch.Tensor] = []

        h = x0
        for (block1, block2), downsample in zip(self.down_blocks, self.downsamples):
            h = block1(h, cond)
            h = block2(h, cond)
            features.append(h.transpose(1, 2))
            skips.append(h)
            h = downsample(h)

        h = self.mid_block1(h, cond)
        h = self.mid_block2(h, cond)
        features.append(h.transpose(1, 2))

        for upsample, (block1, block2), skip in zip(self.upsamples, self.up_blocks, reversed(skips[:-1])):
            h = upsample(h, target_len=skip.shape[-1])
            h = torch.cat([h, skip], dim=1)
            h = block1(h, cond)
            h = block2(h, cond)
            features.append(h.transpose(1, 2))

        # fuse with the highest-resolution skip
        h = torch.cat([h, skips[0]], dim=1)
        h = self.final_block(h, cond)
        features.append(h.transpose(1, 2))

        out = self.final_conv(h).transpose(1, 2)
        return out, features
