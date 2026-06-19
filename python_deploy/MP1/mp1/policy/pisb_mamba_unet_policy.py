import sys
sys.path.append('mp1')
from termcolor import cprint

from mp1.policy.pisb_policy import PISBPolicy
from mp1.model.mean.conditional_mamba_unet1d_meanflow_dis import ConditionalMambaUnet1D


class PISBMambaUNetPolicy(PISBPolicy):
    """
    PISB + official mamba-ssm based ConditionalMambaUnet1D.
    Keeps:
      - physics-informed bridge
      - target velocity objective
      - Euler low-NFE inference
    Replaces only:
      - ConditionalUnet1D backbone -> ConditionalMambaUnet1D
    """
    def __init__(self, shape_meta, **kwargs):
        super().__init__(shape_meta=shape_meta, **kwargs)

        if not self.obs_as_global_cond:
            raise NotImplementedError('PISBMambaUNetPolicy currently requires obs_as_global_cond=True.')
        if 'cross_attention' in self.condition_type:
            raise NotImplementedError('PISBMambaUNetPolicy currently supports film/global conditioning only.')

        global_cond_dim = self.obs_feature_dim * self.n_obs_steps
        self.model = ConditionalMambaUnet1D(
            input_dim=self.action_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=kwargs.get('diffusion_step_embed_dim', 256),
            down_dims=tuple(kwargs.get('down_dims', (256, 512, 512))),
            kernel_size=kwargs.get('kernel_size', 5),
            n_groups=kwargs.get('n_groups', 8),
            condition_type=kwargs.get('condition_type', 'film'),
            use_down_condition=kwargs.get('use_down_condition', True),
            use_mid_condition=kwargs.get('use_mid_condition', True),
            use_up_condition=kwargs.get('use_up_condition', True),
            mamba_d_state=kwargs.get('mamba_d_state', 16),
            mamba_d_conv=kwargs.get('mamba_d_conv', 4),
            mamba_expand=kwargs.get('mamba_expand', 2),
            mamba_dropout=kwargs.get('mamba_dropout', 0.0),
        ).to(self.device)

        cprint('[PISB-MambaUNet] Activated official Mamba-UNet backbone.', 'cyan')
        cprint('[PISB-MambaUNet] Physics bridge / target velocity / Euler inference unchanged.', 'cyan')
        cprint(
            f"[PISB-MambaUNet] down_dims={kwargs.get('down_dims', (256, 512, 512))}, "
            f"mamba_d_state={kwargs.get('mamba_d_state', 16)}, "
            f"mamba_d_conv={kwargs.get('mamba_d_conv', 4)}, "
            f"mamba_expand={kwargs.get('mamba_expand', 2)}",
            'cyan'
        )
