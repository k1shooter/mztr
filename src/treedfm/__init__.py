"""TreeDFM (rewrite)

This package contains a modular implementation of a *tree* variant of
discrete flow matching with edit operations (insert / delete / substitute).

Public API (minimal):
  - MazeTreeDataset, collate_tree_batch
  - TreeEditDFM (plain PyTorch)
"""

from .data import MazeTreeDataset, collate_tree_batch
from .dfm import TreeEditDFM, TreeInsertOnlyDFM

__all__ = [
    "MazeTreeDataset",
    "collate_tree_batch",
    "TreeEditDFM",
    "TreeInsertOnlyDFM",
]
