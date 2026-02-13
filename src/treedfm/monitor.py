"""treedfm.monitor

Training-time sampling / plotting utilities.

This is a rewrite of the "TrainingVisualizer" callback from your original
`treedfm_vec.py`, but adapted to the modular edit-flow codebase.

It supports both:
  - the plain PyTorch training loop in `train_edit.py`
  - the Lightning callback in `treedfm.lit.TrainingVisualizer`
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt

from .metrics import compute_sample_metrics


@dataclass
class History:
    # Per-step loss trace
    loss_steps: List[int]
    loss_history: List[float]
    loss_epochs: List[float]

    # Sampling trace (recorded at sampling time)
    sample_steps: List[int]
    sample_epochs: List[int]
    sample_avg_nodes: List[float]
    sample_valid_rate: List[float]
    sample_exact_rate: List[float]


class TrainingMonitor:
    """Keeps a small history + periodically samples and saves plots.

    Notes
    -----
    - We intentionally keep this *simple* and file-system based.
    - For accuracy, we provide "exact node count" against `expected_nodes`.
      This matches how people often sanity-check spanning-tree maze generation
      (4x4 -> 16 nodes, 10x10 -> 100 nodes).
    """

    def __init__(
        self,
        *,
        save_dir: str,
        k: int,
        expected_nodes: Optional[int] = None,
        size_tol: int = 0,
        sample_interval_epochs: int = 10,
        num_sample_trees: int = 4,
        sample_steps: int = 300,
    ):
        self.save_dir = save_dir
        self.k = int(k)
        self.expected_nodes = expected_nodes
        self.size_tol = int(size_tol)
        self.sample_interval_epochs = int(sample_interval_epochs)
        self.num_sample_trees = int(num_sample_trees)
        self.sample_steps = int(sample_steps)

        os.makedirs(save_dir, exist_ok=True)

        self.history = History(
            loss_steps=[],
            loss_history=[],
            loss_epochs=[],
            sample_steps=[],
            sample_epochs=[],
            sample_avg_nodes=[],
            sample_valid_rate=[],
            sample_exact_rate=[],
        )

    def record_loss(self, loss_value: float, *, step: int, epoch: float) -> None:
        """Record a single loss value at a specific global training step."""
        self.history.loss_steps.append(int(step))
        self.history.loss_history.append(float(loss_value))
        self.history.loss_epochs.append(float(epoch))

    def maybe_sample_and_save(self, *, epoch: int, step: int, model, tree_plot_fn) -> None:
        """If `epoch` hits the interval, sample, save images, and update plots.

        Parameters
        ----------
        model:
            Must implement `sample(num_samples, steps)`.
        tree_plot_fn:
            Callable(tree, filename, title) -> None
        """
        if epoch % self.sample_interval_epochs != 0:
            return

        epoch_dir = os.path.join(self.save_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        trees = model.sample(num_samples=self.num_sample_trees, steps=self.sample_steps)

        # Metrics
        m = compute_sample_metrics(
            trees,
            k=self.k,
            expected_nodes=self.expected_nodes,
            size_tol=self.size_tol,
        )
        self.history.sample_steps.append(int(step))
        self.history.sample_epochs.append(int(epoch))
        self.history.sample_avg_nodes.append(float(m.avg_nodes))
        self.history.sample_valid_rate.append(float(m.valid_rate))
        self.history.sample_exact_rate.append(float(m.exact_size_rate) if m.exact_size_rate is not None else float("nan"))

        # Save images
        for i, tree in enumerate(trees):
            title = f"epoch {epoch} | sample {i} | nodes={len(tree)}"
            fn = os.path.join(epoch_dir, f"sample_{i}.png")
            tree_plot_fn(tree, fn, title=title)

        # Update plots
        self._plot_metrics()

    def _plot_metrics(self) -> None:
        if len(self.history.loss_history) == 0:
            return

        fig, ax1 = plt.subplots(figsize=(10, 5))
        # Plot loss against *actual* global steps.

        if len(self.history.loss_epochs) == len(self.history.loss_history):
            x_loss = self.history.loss_epochs
            xlabel = "Epoch"
        else:
            # If no epoch info, fall back to simple index (old behavior)
            x_loss = range(len(self.history.loss_history))
            xlabel = "Steps (or Epoch Index)"

        ax1.plot(x_loss, self.history.loss_history, label='Loss', color='blue', alpha=0.6)
        ax1.set_ylabel("Loss", color='blue')
        ax1.set_xlabel(xlabel)
        ax1.tick_params(axis='y', labelcolor='blue')

        if len(self.history.sample_epochs) > 0:
            ax2 = ax1.twinx()
            # Plot against sample_epochs (which are integers like 10, 20, 30)
            ax2.plot(self.history.sample_epochs, self.history.sample_avg_nodes, 
                     color='red', marker="o", label='Avg Nodes')
            ax2.set_ylabel("Sample avg #nodes", color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Align grid
            # ax2.grid(None) # Optional: remove grid from 2nd axis to avoid clutter

        plt.title("Training loss & sampled tree size")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "loss_size_plot.png"))
        plt.close(fig)

        # Optional extra plot: validity and exact-size accuracy
        if len(self.history.sample_epochs) > 0:
            fig2, ax = plt.subplots(figsize=(10, 4))
            ax.plot(self.history.sample_epochs, self.history.sample_valid_rate, marker="o", label="valid_rate")
            # exact rate may be NaN if expected_nodes is None
            ax.plot(self.history.sample_epochs, self.history.sample_exact_rate, marker="o", label="exact_size_rate")
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Rate")
            ax.legend()
            plt.title("Sample validity / size accuracy")
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, "sample_quality_plot.png"))
            plt.close(fig2)
