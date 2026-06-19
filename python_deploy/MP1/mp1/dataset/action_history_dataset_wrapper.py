
from typing import Dict, Optional
import copy
import numpy as np
import torch

from mp1.dataset.base_dataset import BaseDataset
from mp1.common.pytorch_util import dict_apply


class ActionHistoryDatasetWrapper(BaseDataset):
    """
    Wrap any dataset that exposes:
      - base_dataset.replay_buffer['action']
      - base_dataset.replay_buffer.episode_ends
      - base_dataset.sampler.indices  (N, 4), see SequenceSampler

    Returns the original sample plus:
      - action_history:      (H, Da)
      - action_history_mask: (H,)
      - action_history_delta:(H, Da)

    This wrapper is generic and works for both MetaWorld and Adroit style datasets
    used in MP1, as long as they rely on ReplayBuffer + SequenceSampler.
    """

    def __init__(
        self,
        base_dataset: BaseDataset,
        history_len: int = 8,
        pad_mode: str = "zero",
        return_delta: bool = True,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.history_len = int(history_len)
        self.pad_mode = str(pad_mode)
        self.return_delta = bool(return_delta)

        if not hasattr(base_dataset, "replay_buffer"):
            raise AttributeError("base_dataset must expose replay_buffer")
        if not hasattr(base_dataset, "sampler"):
            raise AttributeError("base_dataset must expose sampler")
        if not hasattr(base_dataset.sampler, "indices"):
            raise AttributeError("base_dataset.sampler must expose indices")
        if "action" not in base_dataset.replay_buffer:
            raise KeyError("base_dataset.replay_buffer must contain key 'action'")

        self.replay_buffer = base_dataset.replay_buffer
        self.sampler = base_dataset.sampler
        self.train_mask = getattr(base_dataset, "train_mask", None)
        self.horizon = getattr(base_dataset, "horizon", None)
        self.pad_before = getattr(base_dataset, "pad_before", 0)
        self.pad_after = getattr(base_dataset, "pad_after", 0)

        self._all_actions = np.asarray(self.replay_buffer["action"])
        self._episode_ends = np.asarray(self.replay_buffer.episode_ends)
        self._indices = np.asarray(self.sampler.indices)

    def __len__(self):
        return len(self.base_dataset)

    def get_validation_dataset(self):
        base_val = self.base_dataset.get_validation_dataset()
        return ActionHistoryDatasetWrapper(
            base_dataset=base_val,
            history_len=self.history_len,
            pad_mode=self.pad_mode,
            return_delta=self.return_delta,
        )

    def get_normalizer(self, *args, **kwargs):
        return self.base_dataset.get_normalizer(*args, **kwargs)

    def _episode_start_from_buffer_idx(self, buffer_idx: int) -> int:
        ep_id = np.searchsorted(self._episode_ends, buffer_idx, side="right")
        return 0 if ep_id == 0 else int(self._episode_ends[ep_id - 1])

    def _pad_history(self, hist: np.ndarray):
        valid_len = hist.shape[0]
        Da = self._all_actions.shape[-1]
        if valid_len > self.history_len:
            hist = hist[-self.history_len:]
            valid_len = self.history_len

        if valid_len == self.history_len:
            mask = np.ones((self.history_len,), dtype=np.float32)
            return hist, mask

        pad_len = self.history_len - valid_len
        if self.pad_mode == "zero":
            pad = np.zeros((pad_len, Da), dtype=hist.dtype)
        elif self.pad_mode == "repeat":
            if valid_len == 0:
                pad = np.zeros((pad_len, Da), dtype=self._all_actions.dtype)
            else:
                pad = np.repeat(hist[:1], pad_len, axis=0)
        else:
            raise ValueError(f"Unsupported pad_mode: {self.pad_mode}")

        hist = np.concatenate([pad, hist], axis=0)
        mask = np.concatenate([
            np.zeros((pad_len,), dtype=np.float32),
            np.ones((valid_len,), dtype=np.float32)
        ], axis=0)
        return hist, mask

    def _build_history(self, sample_idx: int):
        # sampler.indices columns:
        # [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
        buffer_start_idx = int(self._indices[sample_idx][0])

        ep_start = self._episode_start_from_buffer_idx(buffer_start_idx)
        hist_start = max(ep_start, buffer_start_idx - self.history_len)
        hist = self._all_actions[hist_start:buffer_start_idx].astype(np.float32, copy=False)

        hist, mask = self._pad_history(hist)

        if self.return_delta:
            delta = np.zeros_like(hist, dtype=np.float32)
            if hist.shape[0] > 1:
                delta[1:] = hist[1:] - hist[:-1]
            delta = delta * mask[:, None]
        else:
            delta = None

        return hist.astype(np.float32), mask.astype(np.float32), delta

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.base_dataset[idx]
        hist, mask, delta = self._build_history(idx)

        # base dataset already returns torch tensors
        out = {}
        for k, v in sample.items():
            out[k] = v

        out["action_history"] = torch.from_numpy(hist)
        out["action_history_mask"] = torch.from_numpy(mask)
        if delta is not None:
            out["action_history_delta"] = torch.from_numpy(delta)
        return out
