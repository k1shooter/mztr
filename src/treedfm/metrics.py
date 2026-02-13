"""treedfm.metrics

Lightweight evaluation utilities for generated trees.

We keep these metrics *representation-level* (on the flattened BFS list)
so they work for both the insert-only and edit-flow variants.

Tree format: list[[depth, rank, type, parent_idx], ...]
  - ``type == 0`` means inactive/deleted.

These checks are intentionally conservative ("does this look like a well-formed k-ary tree
in our flattened representation?") and do not assume any maze-specific geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


def count_active_nodes(tree: List[List[int]]) -> int:
    """Number of nodes with type != 0."""
    return sum(1 for n in tree if int(n[2]) != 0)


def _iter_active_edges(tree: List[List[int]]):
    for i, n in enumerate(tree):
        typ = int(n[2])
        parent = int(n[3])
        if typ == 0:
            continue
        if parent >= 0:
            yield parent, i


def has_unique_child_slots(tree: List[List[int]], k: int) -> bool:
    """Checks that each (parent_idx, rank) pair appears at most once among active nodes."""
    seen = set()
    for i, n in enumerate(tree):
        typ = int(n[2])
        if typ == 0:
            continue
        parent = int(n[3])
        if parent < 0:
            continue
        rank = int(n[1])
        if rank < 0 or rank >= k:
            return False
        key = (parent, rank)
        if key in seen:
            return False
        seen.add(key)
    return True


def is_connected_from_root(tree: List[List[int]], k: int) -> bool:
    """Checks that all active nodes are reachable from root by following parent links."""
    if not tree:
        return False
    # Find root(s): parent == -1 and type != 0
    roots = [i for i, n in enumerate(tree) if int(n[2]) != 0 and int(n[3]) < 0]
    if len(roots) != 1:
        return False
    root = roots[0]
    # BFS via parent pointers (reverse edges)
    children = {i: [] for i, n in enumerate(tree) if int(n[2]) != 0}
    for p, c in _iter_active_edges(tree):
        if p not in children:
            # active child points to inactive/missing parent
            return False
        children[p].append(c)

    visited = set([root])
    q = [root]
    while q:
        u = q.pop(0)
        for v in children.get(u, []):
            if v in visited:
                # cycle (shouldn't happen in parent-pointer rep)
                return False
            visited.add(v)
            q.append(v)

    n_active = count_active_nodes(tree)
    return len(visited) == n_active


def depths_consistent(tree: List[List[int]]) -> bool:
    """Checks depth(child) == depth(parent)+1 for active nodes."""
    for i, n in enumerate(tree):
        typ = int(n[2])
        if typ == 0:
            continue
        depth = int(n[0])
        parent = int(n[3])
        if parent < 0:
            if depth != 0:
                return False
            continue
        if parent >= len(tree):
            return False
        p_typ = int(tree[parent][2])
        if p_typ == 0:
            return False
        p_depth = int(tree[parent][0])
        if depth != p_depth + 1:
            return False
    return True


def is_valid_tree(tree: List[List[int]], k: int) -> bool:
    """Combined structural validity check."""
    return has_unique_child_slots(tree, k=k) and is_connected_from_root(tree, k=k) and depths_consistent(tree)


@dataclass
class SampleMetrics:
    avg_nodes: float
    exact_size_rate: float | None
    valid_rate: float


def compute_sample_metrics(
    trees: Iterable[List[List[int]]],
    *,
    k: int,
    expected_nodes: int | None = None,
    size_tol: int = 0,
) -> SampleMetrics:
    """Compute a few simple metrics over a list of sampled trees."""
    trees = list(trees)
    if len(trees) == 0:
        return SampleMetrics(avg_nodes=0.0, exact_size_rate=None, valid_rate=0.0)

    sizes = [count_active_nodes(t) for t in trees]
    avg_nodes = sum(sizes) / float(len(sizes))

    valid = [is_valid_tree(t, k=k) for t in trees]
    valid_rate = sum(1 for v in valid if v) / float(len(valid))

    exact = None
    if expected_nodes is not None:
        exact = sum(1 for s in sizes if abs(int(s) - int(expected_nodes)) <= int(size_tol)) / float(len(sizes))

    return SampleMetrics(avg_nodes=avg_nodes, exact_size_rate=exact, valid_rate=valid_rate)
