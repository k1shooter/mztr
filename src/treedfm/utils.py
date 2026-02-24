
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import networkx as nx
import matplotlib.pyplot as plt


def pad_sequence(
    batch: List[torch.Tensor],
    padding_value: int = 0,
    pad_to: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads a list of [Ni, D] tensors into [B, N, D] with a boolean padding mask [B, N].
    padding_mask[b, j] = True if padded.
    """
    if len(batch) == 0:
        raise ValueError("pad_sequence: empty batch")

    max_len = max(x.size(0) for x in batch)
    if pad_to is not None:
        max_len = max(max_len, int(pad_to))

    padded = []
    masks = []
    for x in batch:
        n, d = x.size()
        if n > max_len:
            raise ValueError(f"pad_sequence: item length {n} > max_len {max_len}. "
                             f"Increase pad_to or check inputs.")
        pad_n = max_len - n
        if pad_n > 0:
            pad = torch.full((pad_n, d), padding_value, dtype=x.dtype, device=x.device)
            x_padded = torch.cat([x, pad], dim=0)
        else:
            x_padded = x

        mask = torch.zeros(max_len, dtype=torch.bool, device=x.device)
        if pad_n > 0:
            mask[n:] = True
        padded.append(x_padded)
        masks.append(mask)

    return torch.stack(padded, dim=0), torch.stack(masks, dim=0)


def build_child_slot_types(
    x: torch.Tensor,
    padding_mask: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    Given node features x [B, N, 4] = (depth, rank, type, parent_idx),
    construct slot tensor child_types [B, N, K] where child_types[b, p, r] = type(child)
    for active (type>0) nodes, else 0.

    Notes:
    - Ignores padding rows (padding_mask == True).
    - Ignores "inactive" nodes with type==0.
    """
    if x.dim() != 3 or x.size(-1) != 4:
        raise ValueError(f"build_child_slot_types: expected x [B,N,4], got {tuple(x.shape)}")
    B, N, _ = x.shape

    child_types = torch.zeros((B, N, k), dtype=torch.long, device=x.device)

    parent_idx = x[:, :, 3].clone()
    rank = x[:, :, 1].clone()
    typ = x[:, :, 2].clone()

    valid = (~padding_mask) & (typ > 0) & (parent_idx >= 0) & (rank >= 0) & (rank < k)
    if valid.any():
        b_idx = torch.arange(B, device=x.device).view(B, 1).expand(B, N)[valid]
        p_idx = parent_idx[valid].long()
        r_idx = rank[valid].long()
        t_val = typ[valid].long()
        child_types.index_put_((b_idx, p_idx, r_idx), t_val, accumulate=False)

    return child_types


class TreeUtils:
    @staticmethod
    def flatten_tree(root: dict, max_depth: int, k: int) -> List[Dict]:
        """
        BFS flattening: each node dict has keys:
        - depth: int
        - rank: int (slot index in parent: 0..k-1; root has 0)
        - type: int
        - parent_idx: int (-1 for root)
        """
        flat_nodes: List[Dict] = []
        queue: List[Tuple[dict, Tuple[int, ...], int]] = [(root, (), -1)]  # node, path, parent_idx
        while queue:
            node, path, parent_idx = queue.pop(0)
            if len(path) > max_depth:
                continue

            node_type = node.get("type", 1)
            depth = len(path)
            rank = path[-1] if depth > 0 else 0

            flat_nodes.append(
                {
                    "depth": depth,
                    "rank": rank,
                    "type": node_type,
                    "parent_idx": parent_idx,
                    "path": path,
                }
            )

            children = node.get("children", [])
            # Keep deterministic ordering if present
            children = sorted(children, key=lambda x: x.get("id", 0))
            for i, child in enumerate(children):
                if i >= k:
                    break
                queue.append((child, path + (i,), len(flat_nodes) - 1))
        return flat_nodes

    @staticmethod
    def save_tree_plot(nodes_list: List, filename: str, title: str = "") -> None:
        """
        nodes_list: list of [depth, rank, type, parent_idx] rows or torch tensors.
        Skips inactive nodes with type==0.
        """
        G = nx.Graph()
        labels = {}
        for i, n in enumerate(nodes_list):
            if isinstance(n, torch.Tensor):
                n = n.tolist()
            depth, rank, typ, parent = n
            if typ == 0:
                continue
            G.add_node(i, type=typ, depth=depth)
            labels[i] = f"{i}\n(T:{int(typ)})"
            if parent != -1 and parent < len(nodes_list):
                # only connect if parent exists and is active
                parent_typ = nodes_list[parent][2] if not isinstance(nodes_list[parent], torch.Tensor) else int(nodes_list[parent][2])
                if parent_typ != 0:
                    G.add_edge(parent, i)

        plt.figure(figsize=(10, 8))
        if G.number_of_nodes() > 0:
            try:
                pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
            except Exception:
                pos = nx.spring_layout(G)

            colors = [
                "#ffffff",
                "#ff9999",
                "#99ccff",
                "#99ff99",
                "#ffcc00",
                "#cc99ff",
                "#ffcc99",
            ]
            node_colors = []
            for node_id in G.nodes():
                typ = G.nodes[node_id].get("type", 0)
                node_colors.append(colors[min(max(int(typ), 0), len(colors) - 1)])
            nx.draw(G, pos, node_color=node_colors, with_labels=True, labels=labels, 
                    node_size=600, font_size=8, edge_color="gray")

        plt.title(title)
        plt.savefig(filename)
        plt.close()

# -----------------------------------------------------------------------------
# BFS canonicalization (GPU-friendly)
# -----------------------------------------------------------------------------
def canonicalize_bfs_with_orig_idx(
    x: torch.Tensor,
    pad_mask: torch.Tensor,
    *,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Canonicalize a batch of trees into *active-node-only* BFS order, and return the
    mapping from canonical indices back to the original row indices.

    This function is used to keep training/sampling consistent when nodes can be
    inserted/deleted (EditFlow-style). It matches the semantics of the original
    Python BFS canonicalizer (dict-based queue), but is implemented with batched
    tensor ops so it can run efficiently on GPU.

    Inputs
      x:        [B, N, 4] long tensor (depth, rank, type, parent_idx)
      pad_mask: [B, N] bool tensor (True for padding rows)
      k:        branching factor

    Returns
      x_can:    [B, M, 4] canonicalized tensor (M = max #active nodes in batch)
      pad_can:  [B, M] bool mask for x_can padding
      orig_idx: [B, M] long mapping s.t. x_can[b,i] came from x[b, orig_idx[b,i]]
               (=-1 for padding rows)

    Notes on correctness
      * We keep only nodes with (type > 0) reachable from the root via active parents.
      * Children are ordered by (parent BFS order, then rank 0..k-1).
      * If the input is invalid and the root is not active, we return a single dummy row.
    """
    assert x.dim() == 3 and x.size(-1) == 4, f"x must be [B,N,4], got {tuple(x.shape)}"
    assert pad_mask.dim() == 2 and pad_mask.shape[:2] == x.shape[:2], "pad_mask must be [B,N]"
    B, N, _ = x.shape
    device = x.device

    if N == 0:
        # Degenerate.
        return x, pad_mask, torch.empty((B, 0), dtype=torch.long, device=device)

    # Active nodes = non-padding rows with type > 0.
    active = (~pad_mask) & (x[:, :, 2] > 0)

    # Build child index table: child_idx[b, parent, r] = child_old_idx or -1.
    parent = x[:, :, 3]
    rank = x[:, :, 1]

    child_idx = torch.full((B, N, k), -1, dtype=torch.long, device=device)
    valid_child = active & (parent >= 0) & (rank >= 0) & (rank < k)
    if valid_child.any():
        # (b, p, r) -> child_old_idx
        b_ids = torch.arange(B, device=device).view(B, 1).expand(B, N)[valid_child]
        p_ids = parent[valid_child].long().clamp(0, N - 1)
        r_ids = rank[valid_child].long().clamp(0, k - 1)
        n_ids = torch.arange(N, device=device).view(1, N).expand(B, N)[valid_child].long()

        # Match the Python dict overwrite behavior deterministically:
        # later rows overwrite earlier, and because we iterate idx increasing,
        # the final value is the maximum index among duplicates.
        key = b_ids * (N * k) + p_ids * k + r_ids
        flat = torch.full((B * N * k,), -1, dtype=torch.long, device=device)
        if hasattr(flat, "scatter_reduce"):
            flat = flat.scatter_reduce(0, key, n_ids, reduce="amax", include_self=True)
        else:
            # Fallback (older torch): order with duplicates may be nondeterministic,
            # but duplicates should not occur in valid trees.
            flat.index_put_((key,), n_ids, accumulate=False)
        child_idx = flat.view(B, N, k)

    # BFS traversal by levels (frontier expansion). This avoids the common pitfall of
    # "heap-index" BFS unrolling, which can miss deep nodes in sparse trees.
    out = torch.full((B, N), -1, dtype=torch.long, device=device)
    out_count = torch.zeros((B,), dtype=torch.long, device=device)

    # Start frontier with root if it is active.
    root_ok = (~pad_mask[:, 0]) & (x[:, 0, 2] > 0)
    frontier = torch.full((B, N), -1, dtype=torch.long, device=device)
    frontier_count = torch.zeros((B,), dtype=torch.long, device=device)
    frontier[root_ok, 0] = 0
    frontier_count[root_ok] = 1

    b_ar = torch.arange(B, device=device)

    for _ in range(N):
        Fmax = int(frontier_count.max().item())
        if Fmax == 0:
            break

        front = frontier[:, :Fmax]  # [B,Fmax] (=-1 for invalid entries)
        arF = torch.arange(Fmax, device=device).view(1, Fmax)

        # Append current frontier into out at per-sample offsets.
        pos = out_count.view(B, 1) + arF                 # [B,Fmax]
        pos_clamped = pos.clamp(max=N - 1)               # guard out-of-range for invalid entries
        if hasattr(out, "scatter_reduce_"):
            out.scatter_reduce_(1, pos_clamped, front, reduce="amax", include_self=True)
        else:
            # Older torch fallback (assumes pos is in-range for valid entries).
            mask = arF < frontier_count.view(B, 1)
            b_ids, j_ids = mask.nonzero(as_tuple=True)
            out[b_ids, pos[b_ids, j_ids]] = front[b_ids, j_ids]

        out_count = out_count + frontier_count

        # Expand to next frontier.
        front_clamped = front.clamp(0, N - 1)
        kids = child_idx[b_ar.view(B, 1), front_clamped]          # [B,Fmax,k]
        valid_front = front >= 0
        kids = torch.where(valid_front.unsqueeze(-1), kids, torch.full_like(kids, -1))
        kids = kids.view(B, Fmax * k)                              # [B,Fmax*k]

        valid_kids = kids >= 0
        frontier_count = valid_kids.sum(dim=1).clamp(max=N)

        if int(frontier_count.max().item()) == 0:
            # No more children anywhere.
            break

        # Stable compaction of kids to the left per batch.
        r = valid_kids.cumsum(dim=1) - 1                           # [B,L]
        r_clamped = torch.where(valid_kids, r, torch.zeros_like(r)).clamp(max=N - 1)
        vals = torch.where(valid_kids, kids, torch.full_like(kids, -1))

        next_frontier = torch.full((B, N), -1, dtype=torch.long, device=device)
        if hasattr(next_frontier, "scatter_reduce_"):
            next_frontier.scatter_reduce_(1, r_clamped, vals, reduce="amax", include_self=True)
        else:
            # Fallback using nonzero.
            b_ids, j_ids = valid_kids.nonzero(as_tuple=True)
            pos2 = r[b_ids, j_ids]
            keep = pos2 < N
            b_ids = b_ids[keep]
            j_ids = j_ids[keep]
            pos2 = pos2[keep]
            next_frontier[b_ids, pos2] = kids[b_ids, j_ids]

        frontier = next_frontier

    max_n = int(out_count.max().item())
    if max_n == 0:
        # Match original behavior: return a single dummy row.
        x0 = torch.zeros((B, 1, 4), dtype=torch.long, device=device)
        x0[:, :, 3] = -1
        pm0 = torch.zeros((B, 1), dtype=torch.bool, device=device)
        oi0 = torch.full((B, 1), -1, dtype=torch.long, device=device)
        return x0, pm0, oi0

    orig_idx = out[:, :max_n]
    pad_can = orig_idx < 0

    gather_idx = orig_idx.clamp(min=0)
    x_can = x.gather(1, gather_idx.unsqueeze(-1).expand(B, max_n, 4)).clone()

    # Remap parent indices from old indexing to new canonical indexing.
    old_to_new = torch.full((B, N), -1, dtype=torch.long, device=device)
    valid_out = ~pad_can
    if valid_out.any():
        b_ids, new_pos = valid_out.nonzero(as_tuple=True)
        old = orig_idx[b_ids, new_pos]
        old_to_new[b_ids, old] = new_pos

    old_parent = x_can[:, :, 3]
    parent_clamped = old_parent.clamp(0, N - 1)
    new_parent = old_to_new.gather(1, parent_clamped)
    new_parent = torch.where(old_parent >= 0, new_parent, torch.full_like(new_parent, -1))
    new_parent = torch.where(pad_can, torch.full_like(new_parent, -1), new_parent)
    x_can[:, :, 3] = new_parent

    # Zero out padding rows (keep parent=-1 for aesthetics/consistency).
    x_can = x_can.masked_fill(pad_can.unsqueeze(-1), 0)
    x_can[:, :, 3] = torch.where(pad_can, torch.full_like(x_can[:, :, 3], -1), x_can[:, :, 3])

    return x_can, pad_can, orig_idx