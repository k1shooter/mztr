
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .corruption import NoiseConfig, corrupt_batch_tree
from .losses import PermutationEditFlowLoss
from .model import TreeEditTransformer
from .schedule import DepthStratifiedSchedule, ProfiledDepthSchedule
from .utils import build_child_slot_types, TreeUtils
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

        # Model mask: padding or inactive
        mask = pad_mask_t | (x_t[:, :, 2] == 0)

        rates, ins_logits, sub_logits = self.net(x_t, mask, t)

        current_slots = build_child_slot_types(x_t, pad_mask_t, k=self.k)
        target_slots = build_child_slot_types(x1, pad_mask, k=self.k)

        # psi per parent (depth+1)
        parent_depths = x_t[:, :, 0].clamp(min=0, max=self.max_depth).long()
        child_depths = (parent_depths + 1).clamp(max=self.max_depth).long()
        psi_parent = self.scheduler.psi(t, child_depths)  # [B,N]
        psi = psi_parent.unsqueeze(-1).expand(B, N, self.k)

        parent_active = (~pad_mask_t) & (x_t[:, :, 2] > 0)
        if parent_active.sum() == 0:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            return loss, {"nodes_xt": 0.0, "nodes_x1": float(((~pad_mask)&(x1[:,:,2]>0)).sum().item()/B)}

        # Slice to active parents
        rates_s = rates[parent_active]
        ins_s = ins_logits[parent_active]
        sub_s = sub_logits[parent_active]
        cur_s = current_slots[parent_active]
        tgt_s = target_slots[parent_active]
        psi_s = psi[parent_active]

        loss = self.loss_fn(rates_s, ins_s, sub_s, cur_s, tgt_s, psi_s)

        with torch.no_grad():
            nodes_xt = float(((~pad_mask_t) & (x_t[:, :, 2] > 0)).sum(dim=1).float().mean().item())
            nodes_x1 = float(((~pad_mask) & (x1[:, :, 2] > 0)).sum(dim=1).float().mean().item())

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
