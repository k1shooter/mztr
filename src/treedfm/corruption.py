
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .schedule import DepthStratifiedSchedule
from .utils import build_child_slot_types


@dataclass
class NoiseConfig:
    """
    Noise distribution for z0 symbols in the EditFlow-style mixture z_t = z1 w.p. kappa else z0.

    For a target slot that *exists* (z1 token > 0):
      z0 = blank (0) with prob p_blank_when_target_token
      z0 = random token in [1..num_types] with prob (1 - p_blank_when_target_token)
        (optionally resampled to avoid matching target type)

    For a target slot that is *empty* (z1 blank == 0):
      z0 = blank (0) with prob p_blank_when_target_blank
      z0 = random token with prob (1 - p_blank_when_target_blank)  -> creates spurious child nodes
    """
    p_blank_when_target_token: float = 0.9
    p_blank_when_target_blank: float = 0.98
    avoid_substitution_identity: bool = True

    # Controls how many spurious nodes we allow per sample when padding is available.
    max_spurious_per_tree: int = 64


def _rand_token_like(shape, num_types: int, device: torch.device) -> torch.Tensor:
    return torch.randint(low=1, high=num_types + 1, size=shape, device=device)


def corrupt_batch_tree(
    x1: torch.Tensor,
    pad_mask: torch.Tensor,
    t: torch.Tensor,
    *,
    k: int,
    num_types: int,
    scheduler: DepthStratifiedSchedule,
    noise: NoiseConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build x_t by:
      (A) Corrupting target nodes (existence + type) using a depth-aware kappa schedule.
      (B) Injecting spurious leaf nodes into padding slots for empty target child-slots.

    Inputs:
      x1: [B, N, 4] (depth, rank, type, parent_idx) padded
      pad_mask: [B, N] True for padding rows
      t: [B] in [0,1]

    Returns:
      x_t: [B, N, 4] (same canvas length N)
      pad_mask_t: [B, N] updated padding mask (some padding rows may become active spurious nodes)
    """
    if x1.dim() != 3 or x1.size(-1) != 4:
        raise ValueError(f"corrupt_batch_tree: expected x1 [B,N,4], got {tuple(x1.shape)}")
    B, N, _ = x1.shape
    device = x1.device

    # Target content for child slots (used for spurious injection)
    target_slots = build_child_slot_types(x1, pad_mask, k=k)  # [B, N, K]

    x_t = x1.clone()
    pad_mask_t = pad_mask.clone()

    depths = x1[:, :, 0].clone()
    types_1 = x1[:, :, 2].clone()
    parents = x1[:, :, 3].clone()

    # --- (A) Corrupt existing target nodes (except root) ---
    # Sample z0 types for target nodes: blank or random token
    # For padding rows, keep them as blank.
    kappa_nodes = scheduler.kappa(t, depths)  # [B,N]
    # Root always kept as its target type
    # We'll sample zt for all nodes then enforce closure.
    u_choose = torch.rand((B, N), device=device)
    choose_target = u_choose < kappa_nodes

    # noise selector for target-token positions
    u_noise = torch.rand((B, N), device=device)
    z0_blank = (u_noise < noise.p_blank_when_target_token)  # True => blank
    z0_type = _rand_token_like((B, N), num_types=num_types, device=device)
    # Optionally avoid "substitution identity" where z0 == z1
    if noise.avoid_substitution_identity:
        same = (z0_type == types_1) & (types_1 > 0)
        if same.any():
            # resample once; if still same it's ok (rare)
            z0_type2 = _rand_token_like((B, N), num_types=num_types, device=device)
            z0_type = torch.where(same, z0_type2, z0_type)

    z0 = torch.where(z0_blank, torch.zeros_like(z0_type), z0_type)
    # Padding and root handling
    is_pad = pad_mask
    is_root = (parents < 0) & (~is_pad)

    # For root: always take target type
    zt = torch.where(choose_target, types_1, z0)
    zt = torch.where(is_pad, torch.zeros_like(zt), zt)
    zt = torch.where(is_root, types_1, zt)

    # Set types in x_t
    x_t[:, :, 2] = zt

    # --- Enforce structural integrity: if parent absent then child absent ---
    # Use iterative projection similar to your vec code.
    safe_parents = parents.clone()
    safe_parents = torch.where(safe_parents < 0, torch.zeros_like(safe_parents), safe_parents)
    exist = (x_t[:, :, 2] > 0) & (~pad_mask_t)
    exist = exist.clone()
    # Root exists
    exist[:, 0] = True

    for _ in range(scheduler.max_depth + 2):
        parent_exist = exist.gather(1, safe_parents.clamp(0, N - 1))
        has_parent = (parents >= 0) & (~pad_mask_t)
        new_exist = exist & (~has_parent | parent_exist)
        if torch.equal(new_exist, exist):
            break
        exist = new_exist

    # Apply existence mask to types
    x_t[:, :, 2] = torch.where(exist, x_t[:, :, 2], torch.zeros_like(x_t[:, :, 2]))

    # --- (B) Inject spurious leaves into empty target slots, using padding capacity ---
    # We only inject under parents that currently exist in x_t.
    # Precompute current child-slot occupancy before injecting spurious nodes.
    current_slots = build_child_slot_types(x_t, pad_mask_t, k=k)  # [B,N,K]
    parent_exist_mask = (x_t[:, :, 2] > 0) & (~pad_mask_t)  # [B,N]

    # Compute depth for child slots (parent depth + 1)
    child_depths = (x_t[:, :, 0].clamp(min=0) + 1).clamp(max=scheduler.max_depth).long()  # [B,N]
    kappa_child = scheduler.kappa(t, child_depths)  # [B,N]
    # We will reuse this scalar for all K slots of that parent.

    # Identify empty slots in target under each parent: target_slots==0
    empty_target_slots = (target_slots == 0)  # [B,N,K]

    # We'll allocate spurious nodes sequentially from padding region per sample.
    for b in range(B):
        # Free indices are those still marked as padding in pad_mask_t
        free = torch.nonzero(pad_mask_t[b], as_tuple=False).view(-1).tolist()
        if not free:
            continue

        spurious_budget = noise.max_spurious_per_tree
        # Iterate over parents that exist
        parent_indices = torch.nonzero(parent_exist_mask[b], as_tuple=False).view(-1).tolist()
        if not parent_indices:
            continue

        for p in parent_indices:
            if spurious_budget <= 0 or not free:
                break
            # Skip if parent is padding (shouldn't happen)
            if pad_mask_t[b, p].item():
                continue
            # Time-dependent mix: z1 is blank for empty slots; so spurious happens when choose_target=False and z0 token.
            kappa_p = float(kappa_child[b, p].item())
            # Pre-sample choose_target for the K slots
            u = torch.rand((k,), device=device)
            choose_target_slot = (u < kappa_p)  # True => keep blank
            # Noise z0 for empty target slot: blank or token
            u0 = torch.rand((k,), device=device)
            z0_blank_slot = (u0 < noise.p_blank_when_target_blank)
            z0_type_slot = _rand_token_like((k,), num_types=num_types, device=device)
            zt_slot = torch.where(choose_target_slot, torch.zeros_like(z0_type_slot), torch.where(z0_blank_slot, torch.zeros_like(z0_type_slot), z0_type_slot))

            for r in range(k):
                if spurious_budget <= 0 or not free:
                    break
                if not empty_target_slots[b, p, r].item():
                    continue  # target already has a child at this slot
                if zt_slot[r].item() == 0:
                    continue  # no spurious here                # Also ensure the slot is currently empty in x_t (no real child survived)
                if current_slots[b, p, r].item() != 0:
                    continue

                idx = free.pop(0)
                # Activate this padding row
                pad_mask_t[b, idx] = False
                spurious_budget -= 1

                # Fill node features
                parent_depth = int(x_t[b, p, 0].item())
                x_t[b, idx, 0] = parent_depth + 1
                x_t[b, idx, 1] = r
                x_t[b, idx, 2] = int(zt_slot[r].item())
                current_slots[b, p, r] = int(zt_slot[r].item())
                x_t[b, idx, 3] = p

        # Any remaining free indices stay padding

    return x_t, pad_mask_t
