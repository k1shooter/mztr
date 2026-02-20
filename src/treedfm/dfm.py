
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .corruption import NoiseConfig, corrupt_batch_tree, corrupt_batch_tree_insert_only
from .losses import PermutationEditFlowLoss
from .model import TreeEditTransformer
from .schedule import DepthStratifiedSchedule, ProfiledDepthSchedule
from .utils import build_child_slot_types, TreeUtils, pad_sequence, canonicalize_bfs_with_orig_idx
from .sampler import sample_tree_ctmc



class TreeEditDFM(nn.Module):
    """
    A plain PyTorch module (no Lightning dependency) implementing:
      - depth-aware EditFlow-style corruption
      - Tree transformer that outputs per-slot edit rates
      - permutation-invariant EditFlow loss

    Call:
      loss, metrics = model.loss_on_batch(x1, pad_mask)
    """

    def __init__(
        self,
        *,
        num_types: int = 3,
        k: int = 3,
        max_depth: int = 100,
        max_nodes: int = 256,
        schedule_width: float = 0.5,
        schedule_max_psi: float = 200.0,
        # Noise config
        p_blank_when_target_token: float = 0.9,
        p_blank_when_target_blank: float = 0.98,
        max_spurious_per_tree: int = 64,
        avoid_substitution_identity: bool = True,
        permutation_invariant: bool = True,
        root_type: int = 2,
        d_model: int = 384,
        n_heads: int = 8,
        n_layers: int = 8,
        dropout: float = 0.1,
        scheduler: Optional[object] = None,
    ):
        super().__init__()
        self.num_types = int(num_types)
        self.k = int(k)
        self.max_depth = int(max_depth)
        self.max_nodes = int(max_nodes)
        self.root_type = int(root_type)

        if scheduler is None:
            self.scheduler = DepthStratifiedSchedule(
                max_depth=max_depth,
                width=schedule_width,
                max_psi=schedule_max_psi,
            )
        else:
            # We only require the schedule object to expose:
            #   - max_depth (int)
            #   - kappa(t, depths)
            #   - psi(t, depths)
            self.scheduler = scheduler

        self.noise = NoiseConfig(
            p_blank_when_target_token=p_blank_when_target_token,
            p_blank_when_target_blank=p_blank_when_target_blank,
            avoid_substitution_identity=avoid_substitution_identity,
            max_spurious_per_tree=max_spurious_per_tree,
        )
        self.net = TreeEditTransformer(
            num_types=num_types,
            k=k,
            max_depth=max_depth,
            max_nodes=max_nodes,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.loss_fn = PermutationEditFlowLoss(
            k=k,
            num_types=num_types,
            permutation_invariant=permutation_invariant,
        )

    def forward(self, x_t: torch.Tensor, mask: torch.Tensor, t: torch.Tensor):
        return self.net(x_t, mask, t)

    def loss_on_batch(self, x1: torch.Tensor, pad_mask: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        x1: [B,N,4]
        pad_mask: [B,N]
        Returns (loss, metrics_dict)
        """
        device = x1.device
        B, N, _ = x1.shape

        # Sample time
        t = torch.rand((B,), device=device)

        # Corrupt
        x_t, pad_mask_t = corrupt_batch_tree(
            x1, pad_mask, t,
            k=self.k,
            num_types=self.num_types,
            scheduler=self.scheduler,
            noise=self.noise,
        )
        x_t, pad_mask_t, orig_idx = canonicalize_bfs_with_orig_idx(x_t, pad_mask_t, k=self.k)

        # Model mask: padding or inactive
        mask = pad_mask_t | (x_t[:, :, 2] == 0)

        rates, ins_logits, sub_logits = self.net(x_t, mask, t)

        current_slots = build_child_slot_types(x_t, pad_mask_t, k=self.k)
        # target slots computed on original x1 index space
        target_slots_full = build_child_slot_types(x1, pad_mask, k=self.k)   # [B,N_orig,K]

        # gather target slots for canonical parents using orig_idx
        B, Nc, _ = x_t.shape
        orig_clamped = orig_idx.clamp(min=0, max=target_slots_full.size(1) - 1)  # [B,Nc]
        target_slots = target_slots_full.gather(
            1, orig_clamped.unsqueeze(-1).expand(B, Nc, self.k)
        )  # [B,Nc,K]
        target_slots = torch.where(
            (orig_idx < 0).unsqueeze(-1) | pad_mask_t.unsqueeze(-1),
            torch.zeros_like(target_slots),
            target_slots,
        )

        ## psi per parent (depth+1)
        # ---- psi scheduling (optionally op-specific) ----
        # Convention: depth is the *child* depth (parent depth + 1) because edits are on child slots.
        parent_depths = x_t[:, :, 0].clamp(min=0, max=self.max_depth).long()
        child_depths = (parent_depths + 1).clamp(max=self.max_depth).long()
        # psi_parent = self.scheduler.psi(t, child_depths)  # [B,N]
        # psi = psi_parent.unsqueeze(-1).expand(B, Nc, self.k)
        #####################################################################################################################
        # Allow the schedule to provide operation-specific psi.
        #   - insertion & deletion: existence schedule
        #   - substitution: type schedule
        if hasattr(self.scheduler, "psi_ops"):
            psi_ins_p, psi_del_p, psi_sub_p = self.scheduler.psi_ops(t, child_depths)
        else:
            # Backward compatible fallback
            psi_exist = self.scheduler.psi(t, child_depths)
            psi_ins_p, psi_del_p, psi_sub_p = psi_exist, psi_exist, psi_exist

        psi_ins = psi_ins_p.unsqueeze(-1).expand(B, Nc, self.k)
        psi_del = psi_del_p.unsqueeze(-1).expand(B, Nc, self.k)
        psi_sub = psi_sub_p.unsqueeze(-1).expand(B, Nc, self.k)
        # shape: [B, N, K, 3]
        psi = torch.stack([psi_ins, psi_del, psi_sub], dim=-1)
        #####################################################################################################################

        parent_active = (~pad_mask_t) & (x_t[:, :, 2] > 0)
        if parent_active.sum() == 0:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            nodes_x1 = ((~pad_mask) & (x1[:, :, 2] > 0)).sum(dim=1).float().mean()
            return loss, {"nodes_xt": torch.zeros((), device=device), "nodes_x1": nodes_x1}

        # Slice to active parents
        rates_s = rates[parent_active]
        ins_s = ins_logits[parent_active]
        sub_s = sub_logits[parent_active]
        cur_s = current_slots[parent_active]
        tgt_s = target_slots[parent_active]
        psi_s = psi[parent_active]

        loss = self.loss_fn(rates_s, ins_s, sub_s, cur_s, tgt_s, psi_s)
        with torch.no_grad():
            nodes_xt = ((~pad_mask_t) & (x_t[:, :, 2] > 0)).sum(dim=1).float().mean()
            nodes_x1 = ((~pad_mask) & (x1[:, :, 2] > 0)).sum(dim=1).float().mean()

        return loss, {"nodes_xt": nodes_xt, "nodes_x1": nodes_x1}

    @torch.no_grad()
    def sample(self, num_samples: int = 4, steps: int = 300, max_nodes: Optional[int] = None, temperature: float = 1.0):
        if max_nodes is None:
            max_nodes = self.max_nodes
        return sample_tree_ctmc(
            self.net,
            num_samples=num_samples,
            steps=steps,
            max_nodes=max_nodes,
            k=self.k,
            num_types=self.num_types,
            root_type=self.root_type,
            temperature=temperature,
            device=next(self.parameters()).device,
        )

    @torch.no_grad()
    def save_samples(self, save_dir: str, epoch: int, num_samples: int = 4, steps: int = 300):
        os.makedirs(save_dir, exist_ok=True)
        trees = self.sample(num_samples=num_samples, steps=steps)
        for i, tree in enumerate(trees):
            fn = os.path.join(save_dir, f"epoch_{epoch:04d}_sample_{i}.png")
            TreeUtils.save_tree_plot(tree, fn, title=f"epoch {epoch} | sample {i} | nodes={len(tree)}")

class TreeInsertOnlyDFM(nn.Module):
    """Insert-only ablation model (same backbone / size as TreeEditDFM).

    Differences vs TreeEditDFM:
      - corruption: only *blank out* target nodes (no random-token noise, no spurious nodes)
      - operations: only insertion is enabled (delete/sub rates are forced to 0)
      - loss: still uses the same EditFlow-style objective, but only insertion terms are active

    This isolates how much the EDIT operations (del/sub) are helping vs a pure growth process.
    """

    def __init__(
        self,
        *,
        num_types: int = 3,
        k: int = 3,
        max_depth: int = 100,
        max_nodes: int = 256,
        schedule_width: float = 0.5,
        schedule_max_psi: float = 200.0,
        # keep the same signature for easy ablation swapping
        p_blank_when_target_token: float = 0.9,
        p_blank_when_target_blank: float = 0.98,
        max_spurious_per_tree: int = 64,
        avoid_substitution_identity: bool = True,
        permutation_invariant: bool = True,
        root_type: int = 2,
        d_model: int = 384,
        n_heads: int = 8,
        n_layers: int = 8,
        dropout: float = 0.1,
        scheduler: Optional[object] = None,
    ):
        super().__init__()
        self.num_types = int(num_types)
        self.k = int(k)
        self.max_depth = int(max_depth)
        self.max_nodes = int(max_nodes)
        self.root_type = int(root_type)

        # Schedule object: same API as TreeEditDFM expects.
        if scheduler is None:
            self.scheduler = DepthStratifiedSchedule(
                max_depth=max_depth,
                width=schedule_width,
                max_psi=schedule_max_psi,
            )
        else:
            self.scheduler = scheduler

        # Backbone identical to TreeEditDFM (same size)
        self.net = TreeEditTransformer(
            num_types=num_types,
            k=k,
            max_depth=max_depth,
            max_nodes=max_nodes,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        # We can reuse the same loss implementation; del/sub will be disabled.
        self.loss_fn = PermutationEditFlowLoss(
            k=k,
            num_types=num_types,
            permutation_invariant=permutation_invariant,
        )

    def forward(self, x_t: torch.Tensor, mask: torch.Tensor, t: torch.Tensor):
        return self.net(x_t, mask, t)

    def loss_on_batch(self, x1: torch.Tensor, pad_mask: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        device = x1.device
        B, N, _ = x1.shape

        t = torch.rand((B,), device=device)

        # Insert-only corruption (blanking only; projection enforced).
        x_t, pad_mask_t = corrupt_batch_tree_insert_only(
            x1, pad_mask, t,
            k=self.k,
           num_types=self.num_types,
            scheduler=self.scheduler,
        )

        # Keep the exact same canonicalization logic as the edit model.
        x_t, pad_mask_t, orig_idx = canonicalize_bfs_with_orig_idx(x_t, pad_mask_t, k=self.k)

        # Mask for transformer
        mask = pad_mask_t | (x_t[:, :, 2] == 0)

        rates, ins_logits, sub_logits = self.net(x_t, mask, t)

        # Disable delete/substitute at the *loss / sampling* interface.
        rates = rates.clone()
        rates[..., 1:] = 0.0
        sub_logits = torch.zeros_like(sub_logits)

        current_slots = build_child_slot_types(x_t, pad_mask_t, k=self.k)
        target_slots_full = build_child_slot_types(x1, pad_mask, k=self.k)

        # gather target slots for canonical parents using orig_idx
        B, Nc, _ = x_t.shape
        orig_clamped = orig_idx.clamp(min=0, max=target_slots_full.size(1) - 1)
        target_slots = target_slots_full.gather(1, orig_clamped.unsqueeze(-1).expand(B, Nc, self.k))
        target_slots = torch.where(
            (orig_idx < 0).unsqueeze(-1) | pad_mask_t.unsqueeze(-1),
            torch.zeros_like(target_slots),
            target_slots,
        )

        # psi for insertion only: use existence schedule (child depth = parent depth + 1).
        parent_depths = x_t[:, :, 0].clamp(min=0, max=self.max_depth).long()
        child_depths = (parent_depths + 1).clamp(max=self.max_depth).long()
        if hasattr(self.scheduler, "psi_exist"):
            psi_parent = self.scheduler.psi_exist(t, child_depths)
        else:
            psi_parent = self.scheduler.psi(t, child_depths)
        psi = psi_parent.unsqueeze(-1).expand(B, Nc, self.k)  # [B,N,K]

        parent_active = (~pad_mask_t) & (x_t[:, :, 2] > 0)
        if parent_active.sum() == 0:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            nodes_x1 = float(((~pad_mask) & (x1[:, :, 2] > 0)).sum(dim=1).float().mean().item())
            return loss, {"nodes_xt": 0.0, "nodes_x1": nodes_x1}

        rates_s = rates[parent_active]
        ins_s = ins_logits[parent_active]
        sub_s = sub_logits[parent_active]  # all zeros
        cur_s = current_slots[parent_active]
        tgt_s = target_slots[parent_active]
        psi_s = psi[parent_active]         # [M,K]

        loss = self.loss_fn(rates_s, ins_s, sub_s, cur_s, tgt_s, psi_s)

        with torch.no_grad():
            nodes_xt = float(((~pad_mask_t) & (x_t[:, :, 2] > 0)).sum(dim=1).float().mean().item())
            nodes_x1 = float(((~pad_mask) & (x1[:, :, 2] > 0)).sum(dim=1).float().mean().item())
        return loss, {"nodes_xt": nodes_xt, "nodes_x1": nodes_x1}

    @torch.no_grad()
    def sample(self, num_samples: int = 4, steps: int = 300, max_nodes: Optional[int] = None, temperature: float = 1.0):
        if max_nodes is None:
            max_nodes = self.max_nodes

        # Wrap net so sampler never sees del/sub rates.
        class _InsertOnlyWrapper(nn.Module):
            def __init__(self, net: nn.Module):
                super().__init__()
                self.net = net
            def forward(self, x, mask, t):
                rates, ins_logits, sub_logits = self.net(x, mask, t)
                rates = rates.clone()
                rates[..., 1:] = 0.0
                sub_logits = torch.zeros_like(sub_logits)
                return rates, ins_logits, sub_logits

        return sample_tree_ctmc(
            _InsertOnlyWrapper(self.net),
            num_samples=num_samples,
            steps=steps,
            max_nodes=max_nodes,
            k=self.k,
            num_types=self.num_types,
            root_type=self.root_type,
            temperature=temperature,
            device=next(self.parameters()).device,
        )