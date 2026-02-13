
from __future__ import annotations

import itertools
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PermutationEditFlowLoss(nn.Module):
    """
    EditFlow-style Bregman (cross-entropy) loss for per-parent K slots, with optional
    permutation-invariant matching across the K child slots.

    Model outputs per slot:
      - rates: [M, K, 3]  (lambda_ins, lambda_del, lambda_sub), nonnegative
      - ins_logits: [M, K, C] where C=num_types
      - sub_logits: [M, K, C]

    States per slot:
      - current: [M, K] values in {0..num_types}
      - target:  [M, K] values in {0..num_types}

    Schedule:
      - psi: [M, K] (kappa_dot/(1-kappa)) or any positive weight

    Loss per slot:
      outgoing =  { lambda_ins                  if current==0
                 { lambda_del + lambda_sub      if current>0
      if current == target:
        L = outgoing
      else:
        correct_rate =
          - insertion: lambda_ins * softmax(ins_logits)[target-1]
          - deletion:  lambda_del
          - substitution: lambda_sub * softmax(sub_logits)[target-1]
        L = outgoing - psi * log(correct_rate)
    """

    def __init__(self, k: int, num_types: int, permutation_invariant: bool = True, eps: float = 1e-12):
        super().__init__()
        self.k = int(k)
        self.num_types = int(num_types)
        self.permutation_invariant = bool(permutation_invariant)
        self.eps = float(eps)

        perms = list(itertools.permutations(range(self.k)))
        self.register_buffer("perms", torch.tensor(perms, dtype=torch.long))  # [P, K]

    def _slot_loss(
        self,
        rates: torch.Tensor,       # [M, K, 3]
        ins_logits: torch.Tensor,  # [M, K, C]
        sub_logits: torch.Tensor,  # [M, K, C]
        current: torch.Tensor,     # [M, K]
        target: torch.Tensor,      # [M, K]
        psi: torch.Tensor,         # [M, K]
    ) -> torch.Tensor:
        M, K, _ = rates.shape
        C = ins_logits.size(-1)
        assert K == self.k and C == self.num_types

        lam_ins = rates[..., 0].clamp_min(0.0)  # [M,K]
        lam_del = rates[..., 1].clamp_min(0.0)
        lam_sub = rates[..., 2].clamp_min(0.0)

        ins_probs = F.softmax(ins_logits, dim=-1)  # [M,K,C]
        sub_probs = F.softmax(sub_logits, dim=-1)

        cur0 = (current == 0)
        cur1 = (current > 0)

        outgoing = torch.zeros((M, K), device=rates.device, dtype=rates.dtype)
        outgoing[cur0] = lam_ins[cur0]
        outgoing[cur1] = (lam_del + lam_sub)[cur1]

        # correct_rate init
        correct_rate = torch.zeros((M, K), device=rates.device, dtype=rates.dtype)

        # insertion: cur=0, tgt>0
        mask_ins = cur0 & (target > 0)
        if mask_ins.any():
            # target in 1..C
            tgt_idx = (target[mask_ins] - 1).long().clamp(0, C - 1)
            probs = ins_probs[mask_ins].gather(-1, tgt_idx.unsqueeze(-1)).squeeze(-1)
            correct_rate[mask_ins] = lam_ins[mask_ins] * probs

        # deletion: cur>0, tgt=0
        mask_del = cur1 & (target == 0)
        if mask_del.any():
            correct_rate[mask_del] = lam_del[mask_del]

        # substitution: cur>0, tgt>0 and different
        mask_sub = cur1 & (target > 0) & (target != current)
        if mask_sub.any():
            tgt_idx = (target[mask_sub] - 1).long().clamp(0, C - 1)
            probs = sub_probs[mask_sub].gather(-1, tgt_idx.unsqueeze(-1)).squeeze(-1)
            correct_rate[mask_sub] = lam_sub[mask_sub] * probs

        mismatch = (target != current)
        # If mismatch but operation would be illegal (e.g., cur=0 & tgt=0) this is False anyway.
        # Here mismatch includes (cur>0 & tgt>0 same) False.
        # For mismatch positions, apply reward term.
        loss = outgoing
        if mismatch.any():
            safe_rate = correct_rate.clamp_min(self.eps)
            loss = loss - psi * mismatch.float() * torch.log(safe_rate)

        return loss  # [M,K]

    def forward(
        self,
        rates: torch.Tensor,
        ins_logits: torch.Tensor,
        sub_logits: torch.Tensor,
        current: torch.Tensor,
        target: torch.Tensor,
        psi: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns scalar mean loss.
        """
        if rates.dim() != 3:
            raise ValueError(f"rates must be [M,K,3], got {tuple(rates.shape)}")
        M, K, _ = rates.shape
        if K != self.k:
            raise ValueError(f"K mismatch: expected {self.k}, got {K}")
        if current.shape != (M, K) or target.shape != (M, K) or psi.shape != (M, K):
            raise ValueError("current/target/psi must be [M,K] matching rates")

        if not self.permutation_invariant:
            loss = self._slot_loss(rates, ins_logits, sub_logits, current, target, psi)  # [M,K]
            return loss.sum(dim=-1).mean()

        # Permute target to best align with current ordering.
        P = self.perms.size(0)
        # [M, P, K]
        target_perm = target.unsqueeze(1).expand(M, P, K).gather(2, self.perms.unsqueeze(0).expand(M, P, K))
        # Expand model outputs to [M,P,K,...] using broadcast
        rates_e = rates.unsqueeze(1).expand(M, P, K, 3)
        ins_e = ins_logits.unsqueeze(1).expand(M, P, K, self.num_types)
        sub_e = sub_logits.unsqueeze(1).expand(M, P, K, self.num_types)
        cur_e = current.unsqueeze(1).expand(M, P, K)
        psi_e = psi.unsqueeze(1).expand(M, P, K)

        # Compute loss per perm
        # We'll vectorize by reshaping (M*P, K, ...)
        MP = M * P
        loss_slot = self._slot_loss(
            rates_e.reshape(MP, K, 3),
            ins_e.reshape(MP, K, self.num_types),
            sub_e.reshape(MP, K, self.num_types),
            cur_e.reshape(MP, K),
            target_perm.reshape(MP, K),
            psi_e.reshape(MP, K),
        )  # [MP,K]
        loss_perm = loss_slot.sum(dim=-1).reshape(M, P)  # [M,P]
        min_loss, _ = torch.min(loss_perm, dim=-1)  # [M]
        return min_loss.mean()
