from mp1.model.vision.pointnet_extractor import MP1Encoder
from mp1.model.vision.real_multimodal_encoder import RealMultimodalEncoder


def build_obs_encoder(
    observation_space,
    img_crop_shape=None,
    out_channel=256,
    pointcloud_encoder_cfg=None,
    use_pc_color=False,
    pointnet_type="pointnet",
    obs_encoder_kind="pointcloud",
    image_encoder_output_dim=64,
    image_base_channels=32,
    share_image_encoder=False,
):
    if obs_encoder_kind == "pointcloud":
        return MP1Encoder(
            observation_space=observation_space,
            img_crop_shape=img_crop_shape,
            out_channel=out_channel,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )
    if obs_encoder_kind in ("real_multimodal", "multimodal_real"):
        return RealMultimodalEncoder(
            observation_space=observation_space,
            img_crop_shape=img_crop_shape,
            out_channel=out_channel,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
            image_encoder_output_dim=image_encoder_output_dim,
            image_base_channels=image_base_channels,
            share_image_encoder=share_image_encoder,
        )
    raise ValueError(f"Unsupported obs_encoder_kind: {obs_encoder_kind}")
