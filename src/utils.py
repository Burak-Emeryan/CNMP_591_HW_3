"""Shared utilities for the CNMP project.

Provides deterministic seeding, device selection, train/val splitting,
and data loading helpers. No training logic lives here.
"""

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Seed all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

_DEVICE_PRINTED = False


def get_device() -> torch.device:
    """Return the best available device; prints on first call."""
    global _DEVICE_PRINTED
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not _DEVICE_PRINTED:
        print(f"[utils] Using device: {device}")
        _DEVICE_PRINTED = True
    return device


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------

def train_val_split(
    n_traj: int,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Deterministic split of trajectory indices into train and val sets.

    Parameters
    ----------
    n_traj : int
        Total number of trajectories.
    val_ratio : float
        Fraction reserved for validation / testing.
    seed : int
        Random seed so the split is always identical.

    Returns
    -------
    train_indices, val_indices : tuple of lists
    """
    rng = random.Random(seed)
    indices = list(range(n_traj))
    rng.shuffle(indices)
    n_val = max(1, int(n_traj * val_ratio))
    val_indices = sorted(indices[:n_val])
    train_indices = sorted(indices[n_val:])
    return train_indices, val_indices


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trajectories(
    path: str = "data/trajectories.pt",
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Load collected trajectories from disk.

    The file is expected to contain a dict with keys:
      - "trajectories" : list of (T, 5) float tensors  [t, e_y, e_z, o_y, o_z]
      - "heights"      : (N,) float tensor             [one h per trajectory]

    Parameters
    ----------
    path : str
        Path to the .pt file.

    Returns
    -------
    trajectories : list of torch.Tensor
        Each element has shape (T, 5).
    heights : torch.Tensor
        Shape (N,), one scalar height per trajectory.
    """
    data = torch.load(Path(path), map_location="cpu")
    return data["trajectories"], data["heights"]
