
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from .corruption import NoiseConfig, corrupt_batch_tree
from .losses import PermutationEditFlowLoss
from .model import TreeEditTransformer
from .schedule import DepthStratifiedSchedule
from .utils import build_child_slot_types
from .sampler import sample_tree_ctmc
from .utils import TreeUtils


class TreeEditDFM(pl.LightningModule):
    def __init__(
        self,
        *,
        num_types: int = 3,
        k: int = 3,
        max_depth: int = 100,
        max_nodes: int = 256,
        lr: float = 2e-4,
        schedule_width: float = 0.5,
        schedule_max_psi: float = 200.0,
        # Noise config
        p_blank_when_target_token: float = 0.9,
        p_blank_when_target_blank: float = 0.98,
        max_spurious_per_tree: int = 64,
        permutation_invariant: bool = True,
        root_type: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.scheduler = DepthStratifiedSchedule(
            max_depth=max_depth,
            width=schedule_width,
            max_psi=schedule_max_psi,
        )
        self.noise = NoiseConfig(
            p_blank_when_target_token=p_blank_when_target_token,
            p_blank_when_target_blank=p_blank_when_target_blank,
            max_spurious_per_tree=max_spurious_per_tree,
        )

        self.model = TreeEditTransformer(
            num_types=num_types,
            k=k,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )

        self.loss_fn = PermutationEditFlowLoss(
            k=k,
            num_types=num_types,
            permutation_invariant=permutation_invariant,
        )

        self.root_type = int(root_type)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        x1, pad_mask = batch  # x1 [B,N,4]
        B, N, _ = x1.shape
        device = x1.device

        # Sample time uniformly
        t = torch.rand((B,), device=device)

        # Corrupt to x_t
        x_t, pad_mask_t = corrupt_batch_tree(
            x1, pad_mask, t,
            k=self.hparams.k,
            num_types=self.hparams.num_types,
            scheduler=self.scheduler,
            noise=self.noise,
        )

        # Mask inactive nodes (type==0) AND padding
        model_mask = pad_mask_t | (x_t[:, :, 2] == 0)

        # Forward model
        rates, ins_logits, sub_logits = self.model(x_t, model_mask, t)

        # Build slot representations
        current_slots = build_child_slot_types(x_t, pad_mask_t, k=self.hparams.k)   # [B,N,K]
        target_slots = build_child_slot_types(x1, pad_mask, k=self.hparams.k)       # [B,N,K]

        # psi per parent (applied to all K child slots)
        parent_depths = x_t[:, :, 0].clamp(min=0, max=self.hparams.max_depth).long()
        child_depths = (parent_depths + 1).clamp(max=self.hparams.max_depth).long()
        psi_parent = self.scheduler.psi(t, child_depths)  # [B,N]
        psi = psi_parent.unsqueeze(-1).expand(B, N, self.hparams.k)  # [B,N,K]

        # Active parents are those that exist in x_t (not padding, type>0)
        parent_active = (~pad_mask_t) & (x_t[:, :, 2] > 0)

        if parent_active.sum() == 0:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            self.log("train_loss", loss)
            return loss

        # Slice
        rates_s = rates[parent_active]          # [M,K,3]
        ins_s = ins_logits[parent_active]       # [M,K,C]
        sub_s = sub_logits[parent_active]       # [M,K,C]
        cur_s = current_slots[parent_active]    # [M,K]
        tgt_s = target_slots[parent_active]     # [M,K]
        psi_s = psi[parent_active]              # [M,K]

        loss = self.loss_fn(rates_s, ins_s, sub_s, cur_s, tgt_s, psi_s)
        self.log("train_loss", loss, prog_bar=True)

        # Some diagnostics
        with torch.no_grad():
            num_nodes_t = ((x_t[:, :, 2] > 0) & (~pad_mask_t)).sum(dim=1).float().mean()
            num_nodes_1 = ((x1[:, :, 2] > 0) & (~pad_mask)).sum(dim=1).float().mean()
            self.log("nodes_xt", num_nodes_t, prog_bar=False)
            self.log("nodes_x1", num_nodes_1, prog_bar=False)

        return loss

    @torch.no_grad()
    def sample(self, num_samples: int = 4, steps: int = 300, max_nodes: Optional[int] = None, temperature: float = 1.0):
        if max_nodes is None:
            max_nodes = self.hparams.max_nodes
        trees = sample_tree_ctmc(
            self.model,
            num_samples=num_samples,
            steps=steps,
            max_nodes=max_nodes,
            k=self.hparams.k,
            num_types=self.hparams.num_types,
            root_type=self.root_type,
            temperature=temperature,
            device=self.device,
        )
        return trees


class TrainingVisualizer(pl.Callback):
    """Lightning callback version of the original `treedfm_vec.py` visualizer.

    - samples trees every N epochs
    - saves per-epoch sample plots
    - maintains a loss / size history and writes plots to disk

    If `expected_nodes` is provided, we also log an "exact size" accuracy
    (useful for spanning-tree mazes: 4x4->16, 10x10->100).
    """

    def __init__(
        self,
        save_dir: str,
        every_n_epochs: int = 10,
        sample_steps: int = 300,
        num_samples: int = 4,
        expected_nodes: int | None = None,
        size_tol: int = 0,
    ):
        super().__init__()
        self.save_dir = save_dir
        self.every_n_epochs = int(every_n_epochs)
        self.sample_steps = int(sample_steps)
        self.num_samples = int(num_samples)
        self.expected_nodes = expected_nodes
        self.size_tol = int(size_tol)

        # Histories (kept in-memory, written as plots)
        self.loss_history: list[float] = []
        self.sample_epochs: list[int] = []
        self.sample_avg_nodes: list[float] = []
        self.sample_exact_rate: list[float] = []

        os.makedirs(save_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = int(trainer.current_epoch)

        # Track loss like the original vec callback
        if "train_loss" in trainer.callback_metrics:
            try:
                self.loss_history.append(float(trainer.callback_metrics["train_loss"].item()))
            except Exception:
                pass

        if epoch % self.every_n_epochs != 0:
            return

        epoch_dir = os.path.join(self.save_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        trees = pl_module.sample(num_samples=self.num_samples, steps=self.sample_steps, temperature=1.0)

        # Size metrics
        sizes = [sum(1 for n in t if int(n[2]) != 0) for t in trees]
        avg_size = float(sum(sizes) / max(1, len(sizes)))
        self.sample_epochs.append(epoch)
        self.sample_avg_nodes.append(avg_size)

        if self.expected_nodes is not None:
            exact = sum(1 for s in sizes if abs(int(s) - int(self.expected_nodes)) <= self.size_tol) / float(len(sizes))
            self.sample_exact_rate.append(float(exact))
        else:
            self.sample_exact_rate.append(float("nan"))

        for i, tree in enumerate(trees):
            fn = os.path.join(epoch_dir, f"sample_{i}.png")
            TreeUtils.save_tree_plot(tree, fn, title=f"epoch {epoch} | sample {i} | nodes={sizes[i]}")

        self._plot_metrics()

    def _plot_metrics(self):
        """Write `loss_size_plot.png` and `size_accuracy_plot.png` to `save_dir`."""
        import matplotlib.pyplot as plt

        if not self.loss_history:
            return

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(self.loss_history)
        ax1.set_ylabel("Loss")
        ax1.set_xlabel("Epoch")

        if self.sample_epochs:
            ax2 = ax1.twinx()
            ax2.plot(self.sample_epochs, self.sample_avg_nodes, marker="o")
            ax2.set_ylabel("Sample avg #nodes")

        plt.title("Training loss & sampled tree size")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "loss_size_plot.png"))
        plt.close(fig)

        if self.sample_epochs:
            fig2, ax = plt.subplots(figsize=(10, 4))
            ax.plot(self.sample_epochs, self.sample_exact_rate, marker="o", label="exact_size_rate")
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Rate")
            ax.legend()
            plt.title("Sample exact-size accuracy")
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, "size_accuracy_plot.png"))
            plt.close(fig2)
