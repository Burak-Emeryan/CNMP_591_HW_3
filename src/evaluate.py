"""Evaluate the trained CNMP model on held-out trajectories.

Reports MSE in original (de normalized) units for end-effector and object
positions separately. Produces a bar plot with exactly two bars.

Usage:
    cd src
    python evaluate.py --n-tests 100
"""

import argparse
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import utils
from model import CNMP
from train import forward_masked


def denormalize(
    pred: torch.Tensor,
    stats: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """De normalize predicted (e_y, e_z, o_y, o_z) back to original units.

    Parameters
    ----------
    pred : Tensor of shape (..., 4)
        Normalized predictions for [e_y, e_z, o_y, o_z].
    stats : dict
        Must contain 'traj_mean' (5,) and 'traj_std' (5,).

    Returns
    -------
    Tensor of shape (..., 4) in original units.
    """
    # Columns 1:5 of trajj stats correspond to [e_y, e_z, o_y, o_z]
    mean = stats["traj_mean"][1:]  # (4,)
    std = stats["traj_std"][1:]    # (4,)
    return pred * std.to(pred.device) + mean.to(pred.device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the CNMP model.")
    parser.add_argument("--ckpt", type=str, default="../checkpoints/cnmp.pt")
    parser.add_argument("--data", type=str, default="../data/trajectories.pt")
    parser.add_argument("--n-tests", type=int, default=100)
    parser.add_argument("--fig", type=str, default="../figures/mse_bar.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    utils.set_seed(args.seed)
    device = utils.get_device()

    # ---- Load checkpoint ---------------------------------------------
    ckpt = torch.load(args.ckpt, map_location="cpu")
    norm_stats = ckpt["norm_stats"]
    T = ckpt["T"]

    model = CNMP().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, T={T}")

    # ---- Load data & recover val split ----------------------------------
    trajectories, heights = utils.load_trajectories(args.data)
    N = len(trajectories)
    _, val_idx = utils.train_val_split(N, val_ratio=0.15, seed=42)
    print(f"Evaluating on {len(val_idx)} held-out trajectories, "
          f"{args.n_tests} random tests")

    # ---- Normalization helpers (inline) ---------------------------
    def normalize_traj(traj: torch.Tensor) -> torch.Tensor:
        return (traj - norm_stats["traj_mean"]) / norm_stats["traj_std"]

    def normalize_h(h: torch.Tensor) -> torch.Tensor:
        return (h - norm_stats["h_mean"]) / norm_stats["h_std"]

    # ---- Evaluation loop -------------------------------------------------
    ee_mse_list: List[float] = []
    ob_mse_list: List[float] = []

    with torch.no_grad():
        for _ in tqdm(range(args.n_tests), desc="Evaluating"):
            # a.  Pick a random val trajectory as instructed
            idx = val_idx[torch.randint(len(val_idx), (1,)).item()]
            traj_raw = trajectories[idx]       # (T, 5) raw [t, e_y, e_z, o_y, o_z]
            h_raw = heights[idx]               # scalar

            # b. Sample n_ctx and n_tgt independently
            n_ctx = torch.randint(1, T + 1, (1,)).item()
            n_tgt = torch.randint(1, T + 1, (1,)).item()

            # c. Pick random indices without replacement
            ctx_indices = torch.randperm(T)[:n_ctx]
            tgt_indices = torch.randperm(T)[:n_tgt]

            # d. Normalize
            traj_norm = normalize_traj(traj_raw)           # (T, 5)
            h_norm = normalize_h(h_raw)                     # scalar

            context = traj_norm[ctx_indices].unsqueeze(0).to(device)    # (1, n_ctx, 5)
            target_t = traj_norm[tgt_indices, 0:1].unsqueeze(0).to(device)  # (1, n_tgt, 1)
            h_tensor = torch.tensor([[h_norm.item()]], dtype=torch.float32, device=device)  # (1, 1)
            ctx_mask = torch.ones(1, n_ctx, device=device)              # (1, n_ctx)

            # e. Forward pass
            mean, log_var = forward_masked(model, context, target_t, h_tensor, ctx_mask)
            # mean shape: (1, n_tgt, 4)  normalized predictions

            # f. De-normalize predictions
            pred = denormalize(mean.squeeze(0), norm_stats)  # (n_tgt, 4) in original units

            # g. Ground truth in original (raw) units
            gt = traj_raw[tgt_indices, 1:]  # (n_tgt, 4) [e_y, e_z, o_y, o_z]
            gt = gt.to(device)

            # h. Per-test MSE
            ee_err = ((pred[:, 0:2] - gt[:, 0:2]) ** 2).mean().item()
            obj_err = ((pred[:, 2:4] - gt[:, 2:4]) ** 2).mean().item()
            ee_mse_list.append(ee_err)
            ob_mse_list.append(obj_err)

    # ---- Aggregate results --------------------------------------
    ee_mean = np.mean(ee_mse_list)
    ee_std = np.std(ee_mse_list)
    ob_mean = np.mean(ob_mse_list)
    ob_std = np.std(ob_mse_list)

    print(f"\nEnd-Effector MSE: {ee_mean:.6f} ± {ee_std:.6f}")
    print(f"Object MSE:       {ob_mean:.6f} ± {ob_std:.6f}")

    # ---- Bar plot ------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.fig)), exist_ok=True)

    labels = ["End-Effector", "Object"]
    means = [ee_mean, ob_mean]
    stds = [ee_std, ob_std]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, means, yerr=stds, capsize=10,
                  color=["#4C72B0", "#DD8452"], edgecolor="black", linewidth=0.8)
    ax.set_ylabel("MSE")
    ax.set_title(f"Prediction MSE over {args.n_tests} tests (mean ± std)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Add value labels on bars
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.0002,
                f"{m:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(args.fig, dpi=150)
    plt.close(fig)
    print(f"Bar plot saved to {args.fig}")


if __name__ == "__main__":
    main()
