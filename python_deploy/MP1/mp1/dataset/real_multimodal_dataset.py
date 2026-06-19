from typing import Dict

import copy
import numpy as np
import torch

from mp1.common.pytorch_util import dict_apply
from mp1.common.replay_buffer import ReplayBuffer
from mp1.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from mp1.dataset.base_dataset import BaseDataset
from mp1.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


class RealMultimodalDataset(BaseDataset):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        state_key="state",
        action_key="action",
        point_cloud_key="point_cloud",
        global_image_key="global_image",
        wrist_image_key="wrist_image",
    ):
        super().__init__()
        self.state_key = state_key
        self.action_key = action_key
        self.point_cloud_key = point_cloud_key
        self.global_image_key = global_image_key
        self.wrist_image_key = wrist_image_key

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=[
                self.state_key,
                self.action_key,
                self.point_cloud_key,
                self.global_image_key,
                self.wrist_image_key,
            ],
        )
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed,
        )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer[self.action_key],
            "agent_pos": self.replay_buffer[self.state_key][..., :],
            "point_cloud": self.replay_buffer[self.point_cloud_key],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["global_image"] = SingleFieldLinearNormalizer.create_identity()
        normalizer["wrist_image"] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        data = {
            "obs": {
                "point_cloud": sample[self.point_cloud_key].astype(np.float32),
                "agent_pos": sample[self.state_key].astype(np.float32),
                "global_image": sample[self.global_image_key].astype(np.uint8),
                "wrist_image": sample[self.wrist_image_key].astype(np.uint8),
            },
            "action": sample[self.action_key].astype(np.float32),
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)
