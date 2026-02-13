
from __future__ import annotations

import pickle
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .utils import TreeUtils, pad_sequence


class MazeTreeDataset(Dataset):
    """
    Loads pickled maze trees in the same format as your existing code.
    Produces a flattened BFS tensor [N,4] with columns:
      (depth, rank, type, parent_idx)

    IMPORTANT: This dataset does *not* pad; padding is done in the collate_fn.
    """

    def __init__(self, pkl_path: str, *, max_depth: int = 100, k: int = 3):
        super().__init__()
        self.max_depth = int(max_depth)
        self.k = int(k)

        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
        # raw may be dict or list
        if isinstance(raw, dict):
            self.roots = list(raw.values())
        elif isinstance(raw, list):
            self.roots = raw
        else:
            raise ValueError(f"Unexpected pickle type: {type(raw)}")

        self.data: List[torch.Tensor] = []
        for root in self.roots:
            flat_nodes = TreeUtils.flatten_tree(root, max_depth=self.max_depth, k=self.k)
            feats = torch.tensor(
                [[n["depth"], n["rank"], n["type"], n["parent_idx"]] for n in flat_nodes],
                dtype=torch.long,
            )
            self.data.append(feats)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def collate_tree_batch(
    batch: List[torch.Tensor],
    *,
    pad_to: Optional[int] = None,
    padding_value: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads batch of [Ni,4] into x [B,N,4] and pad_mask [B,N].
    If pad_to is provided, pad to max(max_len_in_batch, pad_to).
    """
    x, pad_mask = pad_sequence(batch, padding_value=padding_value, pad_to=pad_to)
    return x, pad_mask
