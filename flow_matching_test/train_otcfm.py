#!/usr/bin/env python3
"""
OT-CFM training script for the toy flow-matching obstacle-bypass experiment.

Identical to ``train.py`` except the flow-matching loss uses mini-batch
optimal-transport coupling (Tong et al., 2023) to pair source noise and
target data, producing straighter conditional flow paths.

Usage (from project root or from within flow_matching_test/):
    python flow_matching_test/train_otcfm.py [OPTIONS]
    # or
    cd flow_matching_test && python train_otcfm.py [OPTIONS]

Requires: torch, numpy, matplotlib, pot (Python Optimal Transport)
"""

import sys
import os
import json

# Make sibling modules importable regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import ot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from dataset_v3 import (
    OfflineDataset,
    generate_and_save,
    generate_eval_gt,
    EVAL_SCENARIOS,
)
from model_otcfm import FlowMatchingMLP

# --------------------------------------------------------------------------- #
# OT-CFM loss & sampler
# --------------------------------------------------------------------------- #

def ot_cfm_loss(model, batch, device):
    """
    OT Conditional Flow Matching loss (Tong et al., 2023).

    Instead of randomly pairing x0 ~ N(0,I) with x1 (data), we solve a
    mini-batch optimal-transport problem to find the coupling that minimises
    total squared displacement.  This yields straighter flows and faster
    convergence.

    x_t = (1 - t) x0_pi + t x1,   u_t = x1 - x0_pi
    Loss = || v_theta(x_t, t, c) - u_t ||^2

    where pi is the OT permutation within the mini-batch.
    """
    traj = batch["trajectory"].to(device)   # (B, T, 2)
    cond = batch["condition"].to(device)    # (B, cond_dim)
    B = traj.shape[0]

    # Convert absolute positions to increments: delta_i = p_i - p_{i-1}
    # (first increment is relative to origin (0,0), so delta_0 = p_0)
    deltas = torch.diff(traj, dim=1, prepend=torch.zeros(B, 1, 2, device=device))

    x0 = torch.randn_like(deltas)          # source noise
    x1 = deltas                             # target data (increments)

    # --- mini-batch OT coupling ---
    # Flatten to (B, T*2) for cost computation
    x0_flat = x0.reshape(B, -1)
    x1_flat = x1.reshape(B, -1)

    # Pairwise squared-Euclidean cost matrix  (B, B)
    with torch.no_grad():
        M = torch.cdist(x0_flat, x1_flat, p=2).pow(2)
        M_np = M.cpu().numpy()

        a = np.ones(B, dtype=np.float64) / B
        b = np.ones(B, dtype=np.float64) / B
        pi = ot.emd(a, b, M_np)            # (B, B) transport plan

        # For uniform marginals of equal size, pi is a permutation matrix
        # (scaled by 1/B).  Extract the permutation via argmax per row.
        perm = torch.tensor(pi.argmax(axis=1), device=device, dtype=torch.long)

    # Permute x0 according to OT plan
    x0 = x0[perm]

    # --- standard CFM loss with OT-paired samples ---
    t = torch.rand(B, device=device)
    t_ = t.view(B, 1, 1)

    xt = (1 - t_) * x0 + t_ * x1
    ut = x1 - x0

    v_pred = model(sample=xt, timestep=t, condition=cond)
    return F.mse_loss(v_pred, ut)


@torch.no_grad()
def euler_sample(model, condition, num_samples=300, T=8,
                 num_steps=100, device="cpu"):
    """Euler ODE integration from t=0 (noise) to t=1 (data)."""
    B = condition.shape[0]
    cond = condition.repeat_interleave(num_samples, dim=0)
    xt = torch.randn(B * num_samples, T, 2, device=device)

    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((B * num_samples,), i * dt, device=device)
        xt = xt + model(sample=xt, timestep=t, condition=cond) * dt

    # Model outputs are in delta space; cumsum to recover absolute positions
    xt = torch.cumsum(xt, dim=-2)

    return xt.view(B, num_samples, T, 2)

# --------------------------------------------------------------------------- #
# Visualisation helpers (identical to train.py)
# --------------------------------------------------------------------------- #

_OCC_CMAP = ListedColormap(["#ffffff", "#aaaaaa"])   # 0=free, 1=occupied
_ROAD_CMAP = ListedColormap(["#ffffff00", "#8cc476"])  # transparent, green

def _draw_env(ax, d):
    """Draw occupancy map + road overlay for a v2 eval scenario."""
    occ, xs, ys = d["occ"], d["xs"], d["ys"]
    road_map = d.get("road_map")
    extent = [xs[0], xs[-1], ys[0], ys[-1]]
    ax.imshow(occ, origin="lower", extent=extent, cmap=_OCC_CMAP,
              alpha=0.6, aspect="auto", zorder=1)
    if road_map is not None and road_map.any():
        ax.imshow(road_map, origin="lower", extent=extent, cmap=_ROAD_CMAP,
                  alpha=0.5, aspect="auto", zorder=2)


def _setup_panel(ax, title="", xlim=(-3.5, 3.5), ylim=(-0.5, 7.5)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.set_facecolor("#f7f7f7")
    if title:
        ax.set_title(title, fontsize=9)


def _draw_trajs(ax, trajs, color, line_alpha=0.25, lw=1.0,
                endpoint_ms=3.5, endpoint_alpha=0.5):
    """Draw trajectories with visible lines and prominent endpoint markers."""
    origin = np.array([[0.0, 0.0]])
    for tr in trajs:
        full = np.concatenate([origin, tr], axis=0)
        ax.plot(full[:, 0], full[:, 1], color=color, alpha=line_alpha,
                lw=lw, solid_capstyle="round", zorder=4)
        ax.plot(tr[-1, 0], tr[-1, 1], "o", color=color,
                ms=endpoint_ms, alpha=endpoint_alpha, zorder=6,
                markeredgewidth=0)


def visualize_gt(eval_data, save_path):
    """Show ground-truth trajectory distributions for all eval scenarios."""
    n = len(eval_data)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    for col, d in enumerate(eval_data):
        ax = axes[col]
        _draw_env(ax, d)
        for gap_trajs in d["trajs_by_gap"]:
            _draw_trajs(ax, gap_trajs, color="#2255a4")
        ax.plot(0, 0, "k^", ms=6, zorder=10)   # origin marker
        _setup_panel(ax, d["label"])

    fig.suptitle("Ground Truth", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_comparison(model, eval_data, device, save_path,
                         num_samples=300, num_steps=100, T=8, title=''):
    """Model samples with occupancy-map background."""
    model.eval()
    n = len(eval_data)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.3))
    if n == 1:
        axes = [axes]

    for col, d in enumerate(eval_data):
        ax = axes[col]
        _draw_env(ax, d)
        cond_t = torch.from_numpy(d["condition"]).unsqueeze(0).to(device)
        samples = euler_sample(model, cond_t, num_samples=num_samples,
                               T=T, num_steps=num_steps, device=device)
        samples = samples[0].cpu().numpy()  # (S, T, 2)
        _draw_trajs(ax, samples, color="#bb3344")
        ax.plot(0, 0, "k^", ms=6, zorder=10)
        _setup_panel(ax, f"OT-CFM: {d['label']}")

    fig.suptitle(title if title else "OT-CFM Samples", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_rays(eval_data, save_path, n_rays=11, fov_deg=90.0,
                   max_dist=10.0):
    """Show raycasting geometry for each eval scenario."""
    n = len(eval_data)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    angles = np.linspace(-np.radians(fov_deg / 2),
                         np.radians(fov_deg / 2), n_rays)

    for col, d in enumerate(eval_data):
        ax = axes[col]
        _draw_env(ax, d)

        # depth rays are the first n_rays entries of condition
        dists = d["condition"][:n_rays] * max_dist
        # road depth rays are the last n_rays entries of condition
        road_dists = d["condition"][2 * n_rays:3 * n_rays] * max_dist
        for i, angle in enumerate(angles):
            dx, dy = np.sin(angle), np.cos(angle)
            # obstacle depth ray
            dist = dists[i]
            hit_x, hit_y = dist * dx, dist * dy
            hit_obs = dist < max_dist * 0.99
            color = "#DB6057" if hit_obs else "#999999"
            ax.plot([0, hit_x], [0, hit_y], color=color, lw=0.8, alpha=0.7)
            if hit_obs:
                ax.plot(hit_x, hit_y, "o", color=color, ms=3)
            # road depth ray
            rd = road_dists[i]
            if rd < max_dist * 0.99:
                rx, ry = rd * dx, rd * dy
                ax.plot([0, rx], [0, ry], color="#3388cc", lw=0.6, alpha=0.5)
                ax.plot(rx, ry, "s", color="#3388cc", ms=2.5)

        ax.plot(0, 0, "ko", ms=4, zorder=10)
        _setup_panel(ax, f"{d['label']}  rays")

    fig.suptitle("Raycasting (red = obstacle, blue = road)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(train_losses, val_losses, save_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="train")
    ax.plot(epochs, val_losses,   label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("OT-CFM Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Toy OT conditional flow matching: obstacle bypass")
    # --- dataset ---
    parser.add_argument("--dataset",            type=str,   default=None,
                        help="Path to pre-generated .npz dataset. "
                             "If not set, generates one on the fly.")
    parser.add_argument("--val_dataset",        type=str,   default=None,
                        help="Path to pre-generated .npz val dataset.")
    parser.add_argument("--n_scenes",           type=int,   default=1000)
    parser.add_argument("--trajs_per_scene",    type=int,   default=1)
    parser.add_argument("--max_samples",        type=int,   default=None,
                        help="Load only this many samples from the dataset.")
    # --- training ---
    parser.add_argument("--epochs",             type=int,   default=600)
    parser.add_argument("--n_rays",             type=int,   default=120)
    parser.add_argument("--hidden_dim",         type=int,   default=256)
    parser.add_argument("--batch_size",         type=int,   default=256)
    parser.add_argument("--lr",                 type=float, default=1e-3)
    parser.add_argument("--n_blocks",           type=int,   default=8)
    parser.add_argument("--vis_every",          type=int,   default=50)
    parser.add_argument("--output_dir",         type=str,   default=None)
    parser.add_argument("--no_vis",             action="store_true",
                        help="Skip all visualisation (faster for sweeps)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results_otcfm")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[OT-CFM] Device: {device} | Results -> {args.output_dir}")

    # ---- data -----------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_train_path = os.path.join(script_dir, "data", "train.npz")
    default_val_path   = os.path.join(script_dir, "data", "val.npz")

    train_path = args.dataset or default_train_path
    val_path   = args.val_dataset or default_val_path

    # generate datasets if they don't exist yet
    if not os.path.exists(train_path):
        print(f"Dataset not found at {train_path}, generating ...")
        generate_and_save(train_path, n_scenes=args.n_scenes,
                          trajs_per_scene=args.trajs_per_scene,
                          n_rays=args.n_rays, seed=42)
    if not os.path.exists(val_path):
        print(f"Val dataset not found at {val_path}, generating ...")
        generate_and_save(val_path,
                          n_scenes=max(args.n_scenes // 10, 50),
                          trajs_per_scene=args.trajs_per_scene,
                          n_rays=args.n_rays, seed=123)

    print(f"Loading training data from {train_path} ...")
    train_ds = OfflineDataset(train_path, max_samples=args.max_samples, seed=42)
    val_ds   = OfflineDataset(val_path, seed=123)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    print(f"  train: {len(train_ds):,} samples  |  val: {len(val_ds):,} samples")

    # ---- model ----------------------------------------------------------
    T = train_ds.T
    n_rays = train_ds.n_rays
    cond_dim = train_ds.cond_dim   # depth (n_rays) + semantic (n_rays)
    model = FlowMatchingMLP(
        input_dim=T * 2,
        cond_dim=cond_dim,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}  |  cond_dim: {cond_dim}")

    cfg_str = f'otcfm|n_scenes={args.n_scenes}|epochs={args.epochs}|n_rays={n_rays}|hidden_dim={args.hidden_dim}'

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # ---- evaluation data (fixed scenarios) ------------------------------
    eval_data = generate_eval_gt(EVAL_SCENARIOS, T=T, n_rays=n_rays, L=7.0)
    if not args.no_vis:
        visualize_gt(eval_data, os.path.join(args.output_dir, cfg_str+"gt_trajectories.png"))
        visualize_rays(eval_data, os.path.join(args.output_dir, cfg_str+"_raycasting.png"),
                       n_rays=n_rays)
        print("  saved gt_trajectories.png & raycasting.png")

        # ---- initial (untrained) samples --------------------------------
        visualize_comparison(model, eval_data, device,
                             os.path.join(args.output_dir, "epoch_000.png"), T=T)

    # ---- training loop --------------------------------------------------
    best_val = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        # -- train --
        model.train()
        running = 0.0
        n_batch = 0
        for batch in train_loader:
            loss = ot_cfm_loss(model, batch, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batch += 1
        train_losses.append(running / n_batch)

        # -- validate --
        model.eval()
        running = 0.0
        n_batch = 0
        with torch.no_grad():
            for batch in val_loader:
                running += ot_cfm_loss(model, batch, device).item()
                n_batch += 1
        val_losses.append(running / n_batch)

        scheduler.step()

        print(f"  [{epoch:3d}/{args.epochs}]  "
              f"train {train_losses[-1]:.4f}  val {val_losses[-1]:.4f}")

        if val_losses[-1] < best_val:
            best_val = val_losses[-1]

        if not args.no_vis and epoch % args.vis_every == 0:
            visualize_comparison(
                model, eval_data, device,
                os.path.join(args.output_dir, f"epoch_{epoch:03d}.png"), T=T)

    # ---- final outputs --------------------------------------------------
    if not args.no_vis:
        plot_loss_curves(train_losses, val_losses, os.path.join(args.output_dir, cfg_str+"_loss_curves.png"))

    visualize_comparison(
            model, eval_data, device,
            os.path.join(args.output_dir, cfg_str+".png"),
            num_samples=300, num_steps=100, T=T, title=cfg_str
        )

    # ---- save loss history as JSON (always) -----------------------------
    history = {
        "config": {k: v for k, v in vars(args).items() if k != "no_vis"},
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f)

    print(f"\nDone.  Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
