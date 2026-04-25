"""Train the CNMP model on collected demonstration trajectories.

Usage:
    cd src
    python train.py --epochs 500 --batch-size 32 --lr 1e-4
    python train.py --overfit --epochs 300   # smoke-test on 2 trajectories
"""

import argparse
import math
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import utils
from model import CNMP


# ---- Trajectory length (fixed across all trajectories) --------------
T = 100


# -------------------------------------------------------------------------
# Normalization helpers
# ------------------------------------------------------------------------

def compute_norm_stats(
    trajectories: List[torch.Tensor],
    heights: torch.Tensor,
    indices: List[int],
) -> Dict[str, torch.Tensor]:
    """Compute per-column mean/std from the training split only.

    Parameters
    ----------
    trajectories : list of (T, 5) tensors  — columns [t, e_y, e_z, o_y, o_z]
    heights : (N,) tensor
    indices : training trajectory indices

    Returns
    -------
    dict with keys traj_mean (5,), traj_std (5,), h_mean (scalar), h_std (scalar)
    """
    train_data = torch.cat([trajectories[i] for i in indices], dim=0)  # (N_train*T, 5)
    train_h = heights[indices]

    traj_mean = train_data.mean(dim=0)   # (5,)
    traj_std = train_data.std(dim=0)     # (5,)
    traj_std = torch.clamp(traj_std, min=1e-8)

    h_mean = train_h.mean()
    h_std = train_h.std().clamp(min=1e-8)

    return {
        "traj_mean": traj_mean,
        "traj_std": traj_std,
        "h_mean": h_mean,
        "h_std": h_std,
    }


def normalize_trajectory(
    traj: torch.Tensor, stats: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """Normalize a (T, 5) trajectory using precomputed stats."""
    return (traj - stats["traj_mean"]) / stats["traj_std"]


def normalize_h(h: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Normalize height(s). Works for scalars or tensors."""
    return (h - stats["h_mean"]) / stats["h_std"]


# ----------------------------------------------------------
# Padded batch sampling
# --------------------------------------------------------------

def sample_batch(
    trajectories: List[torch.Tensor],
    heights: torch.Tensor,
    indices: List[int],
    norm_stats: Dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a padded batch for training.

    For each item in the batch:
      - Pick a random trajectory from `indices`
      - Sample n_ctx ~ U{1, T} and n_tgt ~ U{1, T} independently
      - Sample context/target step indices without replacement

    All items are padded to the batch max n_ctx and n_tgt.
    Padding uses zeros (which is fine since data is normalized around 0).

    Returns
    -------
    context  : (B, max_ctx, 5)
    target_t : (B, max_tgt, 1)
    target_y : (B, max_tgt, 4)
    h_batch  : (B, 1)
    """
    ctx_list = []
    tgt_t_list = []
    tgt_y_list = []
    h_list = []
    n_ctx_list = []
    n_tgt_list = []

    for _ in range(batch_size):
        idx = indices[torch.randint(len(indices), (1,)).item()]
        traj_norm = normalize_trajectory(trajectories[idx], norm_stats)
        h_norm = normalize_h(heights[idx], norm_stats)

        n_ctx = torch.randint(1, T + 1, (1,)).item()
        n_tgt = torch.randint(1, T + 1, (1,)).item()

        ctx_idx = torch.randperm(T)[:n_ctx]
        tgt_idx = torch.randperm(T)[:n_tgt]

        ctx_list.append(traj_norm[ctx_idx])           # (n_ctx, 5)
        tgt_t_list.append(traj_norm[tgt_idx, 0:1])    # (n_tgt, 1)
        tgt_y_list.append(traj_norm[tgt_idx, 1:])     # (n_tgt, 4)
        h_list.append(h_norm)
        n_ctx_list.append(n_ctx)
        n_tgt_list.append(n_tgt)

    max_ctx = max(n_ctx_list)
    max_tgt = max(n_tgt_list)

    # Pad and stack
    context = torch.zeros(batch_size, max_ctx, 5)
    target_t = torch.zeros(batch_size, max_tgt, 1)
    target_y = torch.zeros(batch_size, max_tgt, 4)
    ctx_mask = torch.zeros(batch_size, max_ctx)
    tgt_mask = torch.zeros(batch_size, max_tgt)

    for i in range(batch_size):
        nc = n_ctx_list[i]
        nt = n_tgt_list[i]
        context[i, :nc] = ctx_list[i]
        target_t[i, :nt] = tgt_t_list[i]
        target_y[i, :nt] = tgt_y_list[i]
        ctx_mask[i, :nc] = 1.0
        tgt_mask[i, :nt] = 1.0

    h_batch = torch.tensor([[h.item()] for h in h_list], dtype=torch.float32)  # (B, 1)

    return (context.to(device), target_t.to(device), target_y.to(device),
            h_batch.to(device), ctx_mask.to(device), tgt_mask.to(device))


# --------------------------------------------------------------
# Loss (masked)
# ------------------------------------------------------

def gaussian_nll_masked(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Gaussian NLL with mask for padded targets.

    Parameters
    ----------
    mean    : (B, N_tgt, 4)
    log_var : (B, N_tgt, 4)
    target  : (B, N_tgt, 4)
    mask    : (B, N_tgt) — 1 for real targets, 0 for padding

    Returns
    -------
    Scalar loss averaged over all real entries.
    """
    nll = 0.5 * (log_var + (target - mean) ** 2 / log_var.exp() + math.log(2 * math.pi))
    # Mask: expand mask to (B, N_tgt, 1) and broadcast over 4 dims
    mask_expanded = mask.unsqueeze(-1)  # (B, N_tgt, 1)
    nll_masked = nll * mask_expanded
    # Average over all real entries
    n_real = mask_expanded.sum() * 4  # total number of real scalar predictions
    return nll_masked.sum() / n_real.clamp(min=1.0)


# ----------------------------------------------------------
# Masked mean-pool aggregator
# ---------------------------------------------------------

def masked_mean_pool(encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool encoded context points respecting a padding mask.

    Parameters
    ----------
    encoded : (B, max_ctx, r_dim)
    mask    : (B, max_ctx) — 1 for real context, 0 for padding

    Returns
    -------
    r : (B, r_dim)
    """
    mask_expanded = mask.unsqueeze(-1)                  # (B, max_ctx, 1)
    summed = (encoded * mask_expanded).sum(dim=1)       # (B, r_dim)
    counts = mask_expanded.sum(dim=1).clamp(min=1.0)    # (B, 1)
    return summed / counts


# ---------------------------------------------------------------
# Forward with masking (wraps CNMP model)
# ---------------------------------------------------------------

def forward_masked(
    model: CNMP,
    context: torch.Tensor,
    target_t: torch.Tensor,
    h: torch.Tensor,
    ctx_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass using masked mean-pooling for padded context.

    Parameters
    ----------
    model    : CNMP
    context  : (B, max_ctx, 5)
    target_t : (B, max_tgt, 1)
    h        : (B, 1)
    ctx_mask : (B, max_ctx)

    Returns
    -------
    mean    : (B, max_tgt, 4)
    log_var : (B, max_tgt, 4)
    """
    n_tgt = target_t.shape[1]

    # Encode all context (including padding will be masked out)
    encoded = model.encoder(context)               # (B, max_ctx, 128)

    # Masked mean pool
    r = masked_mean_pool(encoded, ctx_mask)         # (B, 128)

    # Expand r and h
    r_expanded = r.unsqueeze(1).expand(-1, n_tgt, -1)   # (B, n_tgt, 128)
    h_expanded = h.unsqueeze(1).expand(-1, n_tgt, -1)   # (B, n_tgt, 1)

    # Decode
    decoder_input = torch.cat([target_t, r_expanded, h_expanded], dim=-1)
    out = model.decoder(decoder_input)              # (B, n_tgt, 8)

    mean = out[..., :4]
    log_var = out[..., 4:]
    log_var = torch.clamp(log_var, min=-10.0, max=2.0)

    return mean, log_var


# ----------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CNMP model.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (padded)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data", type=str, default="../data/trajectories.pt")
    parser.add_argument("--ckpt", type=str, default="../checkpoints/cnmp.pt")
    parser.add_argument("--loss-hist", type=str, default="../checkpoints/loss_history.pt")
    parser.add_argument("--fig", type=str, default="../figures/loss_curve.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit", action="store_true",
                        help="Train on first 2 trajectories only (smoke test)")
    args = parser.parse_args()

    utils.set_seed(args.seed)
    device = utils.get_device()

    # ---- Load data --------------------------------------------------
    trajectories, heights = utils.load_trajectories(args.data)
    N = len(trajectories)
    print(f"Loaded {N} trajectories, T={T}")

    # ---- Train / val split ---------------------------------------
    train_idx, val_idx = utils.train_val_split(N, val_ratio=0.15, seed=args.seed)
    if args.overfit:
        train_idx = train_idx[:2]
        print(f"[overfit mode] Training on {len(train_idx)} trajectories")
    else:
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # ---- Normalization stats (training set only) --------------------
    norm_stats = compute_norm_stats(trajectories, heights, train_idx)
    print(f"Norm stats — traj_mean: {norm_stats['traj_mean'].tolist()}")
    print(f"             traj_std:  {norm_stats['traj_std'].tolist()}")
    print(f"             h_mean: {norm_stats['h_mean'].item():.4f}, "
          f"h_std: {norm_stats['h_std'].item():.4f}")

    # ---- Model & optimizer ---------------------------------------
    model = CNMP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Training loop -------------------------------------------
    steps_per_epoch = 100
    loss_history: List[float] = []

    pbar = tqdm(range(args.epochs), desc="Training")
    for epoch in pbar:
        model.train()
        epoch_loss_sum = 0.0

        for _ in range(steps_per_epoch):
            context, target_t, target_y, h, ctx_mask, tgt_mask = sample_batch(
                trajectories, heights, train_idx, norm_stats,
                args.batch_size, device,
            )

            optimizer.zero_grad()
            mean, log_var = forward_masked(model, context, target_t, h, ctx_mask)
            loss = gaussian_nll_masked(mean, log_var, target_y, tgt_mask)
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item()

        avg_loss = epoch_loss_sum / steps_per_epoch
        loss_history.append(avg_loss)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    # ---- Save checkpoint -----------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.ckpt)), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "norm_stats": norm_stats,
            "T": T,
            "epoch": args.epochs,
        },
        args.ckpt,
    )
    print(f"\nCheckpoint saved to {args.ckpt}")

    # ---- Save loss history -----------------------------------------
    torch.save(loss_history, args.loss_hist)
    print(f"Loss history saved to {args.loss_hist}")

    # ---- Plot loss curve --------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.fig)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(loss_history) + 1), loss_history, linewidth=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL Loss")
    ax.set_title("CNMP Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.fig, dpi=150)
    plt.close(fig)
    print(f"Loss curve saved to {args.fig}")
    print(f"Final loss: {loss_history[-1]:.4f}")


if __name__ == "__main__":
    main()
