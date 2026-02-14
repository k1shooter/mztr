
import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from treedfm.data import MazeTreeDataset, collate_tree_batch
from treedfm.dfm import TreeEditDFM
from treedfm.monitor import TrainingMonitor
from treedfm.utils import TreeUtils
from treedfm.schedule import estimate_depth_profile, make_profiled_sequential_schedule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="treedfm_edit_runs")

    # Data
    parser.add_argument("--max_depth", type=int, default=128)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--num_types", type=int, default=3)

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=4)

    # Canvas padding (IMPORTANT for deletion learning if your dataset length is fixed)
    parser.add_argument("--pad_to", type=int, default=None,
                        help="Pad all trees to this length (e.g., 2*N). Enables spurious nodes in padding region.")

    # Schedule
    parser.add_argument("--schedule_max_psi", type=float, default=200.0)
    parser.add_argument(
        "--schedule_mode",
        type=str,
        default="simple",
        choices=["simple", "profiled"],
        help="simple: t_start=d/max_depth with fixed width. profiled: dataset-profiled sequential schedule.",
    )
    # simple schedule
    parser.add_argument("--schedule_width", type=float, default=0.5)
    # profiled schedule
    parser.add_argument("--profile_count_mode", type=str, default="nodes", choices=["nodes", "all_slots"])
    parser.add_argument("--profile_smoothing", type=float, default=1.0)
    parser.add_argument("--profile_power", type=float, default=1.0)
    parser.add_argument("--profile_min_width", type=float, default=1e-3)
    parser.add_argument("--profile_overlap", type=float, default=0.0)
    parser.add_argument("--profile_mode", type=str, default="linear", choices=["linear", "exp"])
    parser.add_argument("--profile_exp_eps", type=float, default=1e-3)

    # Noise
    parser.add_argument("--p_blank_token", type=float, default=0.9)
    parser.add_argument("--p_blank_blank", type=float, default=0.98)
    parser.add_argument("--max_spurious", type=int, default=64)

    # Matching
    parser.add_argument(
        "--no_perm_invariant",
        action="store_true",
        help="Disable permutation-invariant matching across the K child slots (useful if slot order is semantic).",
    )

    # Model
    parser.add_argument("--max_nodes", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Sampling / logging
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--sample_every_epochs", type=int, default=10)
    parser.add_argument("--sample_steps", type=int, default=300)

    # Simple "accuracy" for spanning-tree style datasets
    parser.add_argument(
        "--expected_nodes",
        type=int,
        default=None,
        help="If set, report exact-size accuracy for samples (e.g., 16 for 4x4, 100 for 10x10).",
    )
    parser.add_argument("--size_tol", type=int, default=0, help="Tolerance for expected_nodes exact match.")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    ds = MazeTreeDataset(args.pkl_path, max_depth=args.max_depth, k=args.k)

    # If user didn't provide expected_nodes, try to infer a stable target size.
    if args.expected_nodes is None and len(ds) > 0:
        # Many maze spanning-tree datasets have a fixed number of nodes.
        # We infer using the median length across all samples we loaded.
        lengths = sorted(int(x.size(0)) for x in ds.data)
        args.expected_nodes = lengths[len(lengths) // 2]
        print(f"[info] Inferred expected_nodes={args.expected_nodes} (median of dataset lengths).")
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_tree_batch(batch, pad_to=args.pad_to),
        pin_memory=(args.device.startswith("cuda")),
        drop_last=True,
    )

    # Build schedule
    scheduler = None
    if args.schedule_mode == "profiled":
        profile = estimate_depth_profile(
            ds.data,
            max_depth=args.max_depth,
            k=args.k,
            count_mode=args.profile_count_mode,
            include_root=False,
        )
        scheduler = make_profiled_sequential_schedule(
            profile,
            smoothing=args.profile_smoothing,
            power=args.profile_power,
            min_width=args.profile_min_width,
            max_psi=args.schedule_max_psi,
            include_root=False,
            overlap=args.profile_overlap,
            mode=args.profile_mode,
            exp_eps=args.profile_exp_eps,
        )
        print("[info] Using profiled sequential schedule.")
        print("       depth mass (top 10 depths):", profile.counts[:10].tolist())

    model = TreeEditDFM(
        num_types=args.num_types,
        k=args.k,
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
        schedule_width=args.schedule_width,
        schedule_max_psi=args.schedule_max_psi,
        p_blank_when_target_token=args.p_blank_token,
        p_blank_when_target_blank=args.p_blank_blank,
        max_spurious_per_tree=args.max_spurious,
        permutation_invariant=(not args.no_perm_invariant),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        scheduler=scheduler,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    monitor = TrainingMonitor(
        save_dir=args.save_dir,
        k=args.k,
        expected_nodes=args.expected_nodes,
        size_tol=args.size_tol,
        sample_interval_epochs=max(1, args.sample_every_epochs),
        num_sample_trees=4,
        sample_steps=args.sample_steps,
    )

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for it, batch in enumerate(dl):
            x1, pad_mask = batch
            x1 = x1.to(args.device)
            pad_mask = pad_mask.to(args.device)

            loss, metrics = model.loss_on_batch(x1, pad_mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += float(loss.item())
            global_step += 1

            if global_step % args.log_every == 0:
                avg_loss = running / max(1, args.log_every)
                running = 0.0
                print(
                    f"[epoch {epoch:03d} step {global_step:07d}] "
                    f"loss={avg_loss:.4f} nodes_xt={metrics['nodes_xt']:.1f} nodes_x1={metrics['nodes_x1']:.1f}"
                )

            # Track loss for plots (per-step, but record the *true* global step).
            monitor.record_loss(float(loss.item()), step=global_step, epoch=epoch)

        dt = time.time() - t0
        print(f"Epoch {epoch:03d} finished in {dt:.1f}s")

        if (epoch % args.sample_every_epochs) == 0:
            model.eval()

            # Save the same sample images as before, but also keep an "accuracy" curve.
            monitor.maybe_sample_and_save(
                epoch=epoch,
                step=global_step,
                model=model,
                tree_plot_fn=TreeUtils.save_tree_plot,
            )

        # Optional: save checkpoint
        ckpt_path = os.path.join(args.save_dir, "last.pt")
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state": model.state_dict(),
                "opt_state": optimizer.state_dict(),
                "args": vars(args),
            },
            ckpt_path,
        )


if __name__ == "__main__":
    main()
