
from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .utils import pad_sequence


def _build_child_index(tree: List[List[int]], k: int) -> Dict[Tuple[int, int], int]:
    """
    Returns mapping (parent_idx, rank) -> child_idx for all nodes that exist (even type==0),
    but callers can check type.
    """
    mapping: Dict[Tuple[int, int], int] = {}
    for idx, node in enumerate(tree):
        depth, rank, typ, parent = node
        if parent is None or parent < 0:
            continue
        mapping[(int(parent), int(rank))] = idx
    return mapping

def _bfs_canonicalize_active(tree: List[List[int]], k: int) -> List[List[int]]:
    """
    Canonicalize a tree to BFS order (like training data) and reindex parent_idx.
    Keeps only active nodes (type!=0) reachable from root.
    """
    if not tree:
        return []
    child_map = _build_child_index(tree, k=k)
    clean: List[List[int]] = []
    queue: List[Tuple[int, int]] = [(0, -1)]  # (old_idx, new_parent_idx)
    while queue:
        old, new_parent = queue.pop(0)
        if old >= len(tree):
            continue
        depth, rank, typ, _parent = tree[old]
        if int(typ) == 0:
            continue
        new_idx = len(clean)
        clean.append([int(depth), int(rank), int(typ), int(new_parent)])
        for r in range(k):
            c = child_map.get((old, r))
            if c is not None:
                queue.append((c, new_idx))
    return clean

def _cascade_delete(tree: List[List[int]], node_idx: int, child_map: Dict[Tuple[int, int], int], k: int) -> None:
    """
    Sets node type=0 for node_idx and all descendants (recursive).
    """
    tree[node_idx][2] = 0
    for r in range(k):
        c = child_map.get((node_idx, r))
        if c is not None and tree[c][2] != 0:
            _cascade_delete(tree, c, child_map, k)


@torch.no_grad()
def sample_tree_ctmc(
    model,
    *,
    num_samples: int = 1,
    steps: int = 200,
    max_nodes: int = 256,
    k: int = 3,
    num_types: int = 3,
    root_type: int = 2,
    temperature: float = 1.0,
    device: torch.device | None = None,
) -> List[List[List[int]]]:
    """
    Sample trees with CTMC tau-leaping using model rates.

    model must accept (x, mask, t) and return (rates, ins_logits, sub_logits).

    Returns:
      list of trees, each tree is list of [depth, rank, type, parent_idx].
    """
    if device is None:
        device = next(model.parameters()).device

    # Initialize each tree with only a root node.
    trees: List[List[List[int]]] = [[[0, 0, int(root_type), -1]] for _ in range(num_samples)]
    dt = 1.0 / float(steps)

    for s in range(steps):
        t_val = float(s) * dt
        t_tensor = torch.full((num_samples,), t_val, device=device)
        # NEW: match training coordinate system (BFS order) to reduce pos_emb mismatch
        trees = [_bfs_canonicalize_active(t, k=k) for t in trees]

        # Convert variable-length trees into padded tensor
        feats = [torch.tensor(t, dtype=torch.long) for t in trees]
        x, pad_mask = pad_sequence(feats, padding_value=0, pad_to=None)
        x = x.to(device)
        pad_mask = pad_mask.to(device)

        # Mask inactive nodes (type==0) as well
        inactive = (x[:, :, 2] == 0)
        mask = pad_mask | inactive

        rates, ins_logits, sub_logits = model(x, mask, t_tensor)

        # Prepare probabilities
        ins_probs = F.softmax(ins_logits / max(temperature, 1e-6), dim=-1)  # [B,N,K,C]
        sub_probs = F.softmax(sub_logits / max(temperature, 1e-6), dim=-1)

        lam_ins = rates[..., 0].clamp_min(0.0)
        lam_del = rates[..., 1].clamp_min(0.0)
        lam_sub = rates[..., 2].clamp_min(0.0)

        # NOTE: The event simulation below is Python-loop heavy. If we call `.item()` on CUDA
        # tensors inside those loops, we force a GPU sync thousands of times per sampling step.
        # To keep sampling fast (and avoid stalling training when you sample every few epochs),
        # move the small tensors to CPU once per step.
        ins_probs = ins_probs.cpu()
        sub_probs = sub_probs.cpu()
        lam_ins = lam_ins.cpu()
        lam_del = lam_del.cpu()
        lam_sub = lam_sub.cpu()

        for b in range(num_samples):
            tree = trees[b]
            if len(tree) >= max_nodes:
                continue

            # Build child mapping for this tree (based on stored parent/rank)
            child_map = _build_child_index(tree, k=k)

            # Iterate over current nodes and slots
            n_curr = len(tree)
            for parent_idx in range(n_curr):
                if tree[parent_idx][2] == 0:
                    continue
                p_depth = int(tree[parent_idx][0])

                for r in range(k):
                    c_idx = child_map.get((parent_idx, r))
                    current_child_type = 0
                    if c_idx is not None:
                        current_child_type = int(tree[c_idx][2])

                    if current_child_type == 0:
                        # Insertion
                        rate = float(lam_ins[b, parent_idx, r].item())
                        if rate <= 0.0:
                            continue
                        p_event = 1.0 - math.exp(-rate * dt)
                        if random.random() < p_event:
                            new_type = int(torch.multinomial(ins_probs[b, parent_idx, r], 1).item()) + 1
                            if c_idx is None:
                                # create new node
                                if len(tree) >= max_nodes:
                                    continue
                                tree.append([p_depth + 1, r, new_type, parent_idx])
                                child_map[(parent_idx, r)] = len(tree) - 1
                            else:
                                tree[c_idx][2] = new_type
                    else:
                        # Deletion or substitution
                        r_del = float(lam_del[b, parent_idx, r].item())
                        r_sub = float(lam_sub[b, parent_idx, r].item())
                        total = r_del + r_sub
                        if total <= 0.0:
                            continue
                        p_event = 1.0 - math.exp(-total * dt)
                        if random.random() < p_event:
                            if random.random() < (r_del / total):
                                _cascade_delete(tree, c_idx, child_map, k=k)
                            else:
                                new_type = int(torch.multinomial(sub_probs[b, parent_idx, r], 1).item()) + 1
                                tree[c_idx][2] = new_type
    return [_bfs_canonicalize_active(t, k=k) for t in trees]
