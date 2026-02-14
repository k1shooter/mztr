
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
        for i, n in enumerate(nodes_list):
            if isinstance(n, torch.Tensor):
                n = n.tolist()
            depth, rank, typ, parent = n
            if typ == 0:
                continue
            G.add_node(i, type=typ, depth=depth)
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
            nx.draw(G, pos, node_color=node_colors, with_labels=True, node_size=300)

        plt.title(title)
        plt.savefig(filename)
        plt.close()
