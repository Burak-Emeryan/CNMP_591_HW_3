"""Collect demonstration trajectories from the HW4 MuJoCo environment.

Mirrors the collection logic in homework4.py's __main__ block exactly:
each trajectory follows a random cubic Bézier curve in Cartesian space
while the environment records end-effector and object positions.

Usage:
    cd src
    python collect_data.py --n 200 --out ../data/trajectories.pt
"""

import argparse
import os
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

import utils
from homework4 import Hw5Env, bezier


# Fixed trajectory length (Bézier curve with steps=100)
T = 100


def collect_one_trajectory(env: Hw5Env) -> Tuple[torch.Tensor, float]:
    """Collect a single trajectory by following a random Bézier curve.

    Replicates the exact policy from homework4.py __main__:
      - 4 control points, p2/p3 have random z ∈ [1.04, 1.4]
      - Bézier curve with T=100 steps
      - EE moves to curve[0] via _set_ee_in_cartesian, then tracks via _set_ee_pose

    Parameters
    ----------
    env : Hw5Env
        Already-reset environment instance.

    Returns
    -------
    traj : torch.Tensor
        Shape (T, 5) with columns [t, e_y, e_z, o_y, o_z].
    h : float
        Object height for this episode (fixed per trajectory).
    """
    # --- Bézier control points (copied verbatim from homework4.py) ---
    p_1 = np.array([0.5, 0.3, 1.04])
    p_2 = np.array([0.5, 0.15, np.random.uniform(1.04, 1.4)])
    p_3 = np.array([0.5, -0.15, np.random.uniform(1.04, 1.4)])
    p_4 = np.array([0.5, -0.3, 1.04])
    points = np.stack([p_1, p_2, p_3, p_4], axis=0)
    curve = bezier(points)  # shape (100, 3)

    # Move EE to the first curve point (same args as homework4.py)
    env._set_ee_in_cartesian(
        curve[0], rotation=[-90, 0, 180],
        n_splits=100, max_iters=100, threshold=0.05,
    )

    # Follow the curve and record state at each step
    states: List[np.ndarray] = []
    for p in curve:
        env._set_ee_pose(p, rotation=[-90, 0, 180], max_iters=10)
        states.append(env.high_level_state())  # [e_y, e_z, o_y, o_z, h]

    states_np = np.stack(states)  # (T, 5)

    # Extract h from the environment attribute (constant across the episode)
    h = float(env.obj_height)

    # Build (T, 5) tensor: [t, e_y, e_z, o_y, o_z]
    t_col = np.arange(T, dtype=np.float32).reshape(-1, 1)  # (T, 1)
    traj_np = np.concatenate([t_col, states_np[:, :4]], axis=1)  # drop h column
    traj = torch.tensor(traj_np, dtype=torch.float32)

    return traj, h


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect demonstration trajectories from the HW4 environment.",
    )
    parser.add_argument("--n", type=int, default=200,
                        help="Number of trajectories to collect (default: 200)")
    parser.add_argument("--out", type=str, default="../data/trajectories.pt",
                        help="Output path (default: ../data/trajectories.pt)")
    parser.add_argument("--render", action="store_true",
                        help="If set, open GUI; otherwise run headless")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    utils.set_seed(args.seed)

    render_mode = "gui" if args.render else "offscreen"
    env = Hw5Env(render_mode=render_mode)

    trajectories: List[torch.Tensor] = []
    heights: List[float] = []

    for i in tqdm(range(args.n), desc="Collecting trajectories"):
        env.reset()
        traj, h = collect_one_trajectory(env)
        trajectories.append(traj)
        heights.append(h)

    heights_tensor = torch.tensor(heights, dtype=torch.float32)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"trajectories": trajectories, "heights": heights_tensor}, args.out)

    # Summary
    all_data = torch.cat(trajectories, dim=0)  # (N*T, 5)
    col_names = ["t", "e_y", "e_z", "o_y", "o_z"]
    print(f"\nSaved {args.n} trajectories (T={T} each) to {args.out}")
    print(f"{'Column':<8} {'min':>10} {'max':>10}")
    print("-" * 30)
    for j, name in enumerate(col_names):
        print(f"{name:<8} {all_data[:, j].min().item():>10.4f} {all_data[:, j].max().item():>10.4f}")
    print(f"{'h':<8} {heights_tensor.min().item():>10.4f} {heights_tensor.max().item():>10.4f}")


if __name__ == "__main__":
    main()
