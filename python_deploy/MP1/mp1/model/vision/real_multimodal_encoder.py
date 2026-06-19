import copy
from typing import Dict, List, Type

import torch
import torch.nn as nn
from termcolor import cprint

from mp1.model.vision.pointnet_extractor import (
    PointNetEncoderXYZ,
    PointNetEncoderXYZRGB,
    create_mlp,
)


class SmallImageEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        out_channels: int = 64,
    ):
        super().__init__()
        channels = [
            in_channels,
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 4,
        ]

        layers = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            num_groups = min(8, out_ch)
            if out_ch % num_groups != 0:
                num_groups = 1
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
                nn.SiLU(),
            ])
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels[-1], out_channels),
            nn.LayerNorm(out_channels),
            nn.SiLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise RuntimeError(f"Expected image tensor with 4 dims, got {image.shape}")
        if image.shape[1] not in (1, 3) and image.shape[-1] in (1, 3):
            image = image.permute(0, 3, 1, 2).contiguous()
        image = image.float()
        if image.max().item() > 1.5:
            image = image / 255.0
        feature = self.backbone(image)
        return self.head(feature)


class RealMultimodalEncoder(nn.Module):
    def __init__(
        self,
        observation_space: Dict,
        img_crop_shape=None,
        out_channel=256,
        state_mlp_size=(64, 64),
        state_mlp_activation_fn: Type[nn.Module] = nn.ReLU,
        pointcloud_encoder_cfg=None,
        use_pc_color=False,
        pointnet_type="pointnet",
        global_image_key="global_image",
        wrist_image_key="wrist_image",
        image_encoder_output_dim=64,
        image_base_channels=32,
        share_image_encoder=False,
    ):
        super().__init__()
        del img_crop_shape
        del out_channel

        self.state_key = "agent_pos"
        self.point_cloud_key = "point_cloud"
        self.global_image_key = global_image_key
        self.wrist_image_key = wrist_image_key
        self.n_output_channels = 0

        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        self.global_image_shape = observation_space[self.global_image_key]
        self.wrist_image_shape = observation_space[self.wrist_image_key]

        cprint(f"[RealMultimodalEncoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[RealMultimodalEncoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[RealMultimodalEncoder] global image shape: {self.global_image_shape}", "yellow")
        cprint(f"[RealMultimodalEncoder] wrist image shape: {self.wrist_image_shape}", "yellow")

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        if pointcloud_encoder_cfg is None:
            raise RuntimeError("pointcloud_encoder_cfg must be provided for RealMultimodalEncoder")
        if pointnet_type != "mlp":
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")

        pointcloud_encoder_cfg = copy.deepcopy(pointcloud_encoder_cfg)
        if use_pc_color:
            pointcloud_encoder_cfg.in_channels = 6
            self.pointcloud_encoder = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
        else:
            pointcloud_encoder_cfg.in_channels = 3
            self.pointcloud_encoder = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        self.n_output_channels += int(pointcloud_encoder_cfg.out_channels)

        if len(state_mlp_size) == 0:
            raise RuntimeError("State mlp size is empty")
        if len(state_mlp_size) == 1:
            net_arch: List[int] = []
        else:
            net_arch = list(state_mlp_size[:-1])
        state_output_dim = int(state_mlp_size[-1])
        self.state_mlp = nn.Sequential(
            *create_mlp(
                self.state_shape[0],
                state_output_dim,
                net_arch,
                state_mlp_activation_fn,
            )
        )
        self.n_output_channels += state_output_dim

        self.global_image_encoder = SmallImageEncoder(
            in_channels=int(self.global_image_shape[0]),
            base_channels=image_base_channels,
            out_channels=image_encoder_output_dim,
        )
        if share_image_encoder:
            self.wrist_image_encoder = self.global_image_encoder
        else:
            self.wrist_image_encoder = SmallImageEncoder(
                in_channels=int(self.wrist_image_shape[0]),
                base_channels=image_base_channels,
                out_channels=image_encoder_output_dim,
            )
        self.n_output_channels += int(image_encoder_output_dim) * 2

        cprint(f"[RealMultimodalEncoder] output dim: {self.n_output_channels}", "red")

    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(
            f"point cloud shape: {points.shape}, length should be 3",
            "red",
        )
        point_feat = self.pointcloud_encoder(points)

        state = observations[self.state_key]
        state_feat = self.state_mlp(state)

        global_image = observations[self.global_image_key]
        wrist_image = observations[self.wrist_image_key]
        global_feat = self.global_image_encoder(global_image)
        wrist_feat = self.wrist_image_encoder(wrist_image)

        return torch.cat([point_feat, global_feat, wrist_feat, state_feat], dim=-1)

    def output_shape(self):
        return self.n_output_channels
