#!/usr/bin/env python3
"""
Hyperparameter sweep for flow_matching_test/train_otcfm.py.

Varies n_scenes, epochs, and hidden_dim, distributes experiments
across multiple GPUs, then collects validation-loss histories and plots
them in a single figure.

Expects a pre-computed dataset pair at
    flow_matching_test/data/{train,val}_v3.npz
Different n_scenes values are realised by loading subsets of the same
dataset via --max_samples (= n_scenes × trajs_per_scene).

Usage:
    python flow_matching_test/sweep_otcfm.py
    python flow_matching_test/sweep_otcfm.py --jobs_per_gpu 2 --num_gpus 4
    python flow_matching_test/sweep_otcfm.py --plot_only   # skip training, just re-plot
"""

import argparse
import itertools
import subprocess
import os
import json
import concurrent.futures
import threading
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# --------------------------------------------------------------------------- #
# Default parameter grid & fixed values
# --------------------------------------------------------------------------- #
PARAM_GRID = {
    "n_scenes": [1000, 2000, 4000, 8000, 16000],
    "epochs": [200, 400, 600, 800, 1000],
    "hidden_dim": [128, 256, 512],
}

N_RAYS = 120                # fixed
TRAJS_PER_SCENE = 50        # used to compute max_samples from n_scenes

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train_otcfm.py")
DEFAULT_SWEEP_DIR = os.path.join(SCRIPT_DIR, "sweep_results_otcfm")

DATASET_TRAIN = os.path.join(SCRIPT_DIR, "data", "train_v3.npz")
DATASET_VAL   = os.path.join(SCRIPT_DIR, "data", "val_v3.npz")

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def run_name(p):
    return f"ns{p['n_scenes']}_ep{p['epochs']}_hd{p['hidden_dim']}"


def run_experiment(params, gpu_id, sweep_dir):
    """Launch a single training run as a subprocess on the given GPU."""
    name = run_name(params)
    out_dir = os.path.join(sweep_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    max_samples = params["n_scenes"] * TRAJS_PER_SCENE

    cmd = [
        "python",
        TRAIN_SCRIPT,
        "--dataset", DATASET_TRAIN,
        "--val_dataset", DATASET_VAL,
        "--n_scenes", str(params["n_scenes"]),
        "--n_rays", str(N_RAYS),
        "--max_samples", str(max_samples),
        "--epochs", str(params["epochs"]),
        "--hidden_dim", str(params["hidden_dim"]),
        "--output_dir", out_dir,
        "--no_vis",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_path = os.path.join(out_dir, "log.txt")
    with open(log_path, "w") as log_f:
        proc = subprocess.run(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)

    return name, proc.returncode


def collect_results(sweep_dir):
    """Read all history.json files produced by completed runs."""
    results = {}
    for p in sorted(Path(sweep_dir).iterdir()):
        if not p.is_dir():
            continue
        hist = p / "history.json"
        if hist.exists():
            with open(hist) as f:
                results[p.name] = json.load(f)
    return results


# --------------------------------------------------------------------------- #
# GPU scheduler
# --------------------------------------------------------------------------- #

class GPUScheduler:
    """Simple round-robin GPU scheduler with per-GPU concurrency limits."""

    def __init__(self, num_gpus, jobs_per_gpu):
        self.num_gpus = num_gpus
        self.jobs_per_gpu = jobs_per_gpu
        self._semas = [threading.Semaphore(jobs_per_gpu) for _ in range(num_gpus)]
        self._counts = [0] * num_gpus
        self._lock = threading.Lock()

    def acquire(self):
        """Block until a GPU slot is free; return the gpu_id."""
        while True:
            with self._lock:
                order = sorted(range(self.num_gpus), key=lambda i: self._counts[i])
                for gpu_id in order:
                    if self._semas[gpu_id].acquire(blocking=False):
                        self._counts[gpu_id] += 1
                        return gpu_id
            threading.Event().wait(0.05)

    def release(self, gpu_id):
        with self._lock:
            self._counts[gpu_id] -= 1
        self._semas[gpu_id].release()


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_val_loss_grid(results, sweep_dir):
    """
    One subplot per hidden_dim.  Val-loss curves coloured by n_scenes,
    opacity scaled by epochs.
    """
    hidden_dims = sorted(PARAM_GRID["hidden_dim"])
    n_scenes_list = sorted(PARAM_GRID["n_scenes"])
    epochs_list = sorted(PARAM_GRID["epochs"])

    cmap = plt.colormaps.get_cmap("viridis")
    ns_colors = {
        ns: cmap(i / max(len(n_scenes_list) - 1, 1))
        for i, ns in enumerate(n_scenes_list)
    }
    ep_alphas = {
        ep: 0.3 + 0.7 * i / max(len(epochs_list) - 1, 1)
        for i, ep in enumerate(epochs_list)
    }

    ncols = len(hidden_dims)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), squeeze=False)
    axes = axes[0]

    for name, data in results.items():
        cfg = data["config"]
        try:
            ci = hidden_dims.index(cfg["hidden_dim"])
        except ValueError:
            continue
        ax = axes[ci]
        color = ns_colors.get(cfg["n_scenes"], "gray")
        alpha = ep_alphas.get(cfg["epochs"], 0.5)
        ax.plot(data["val_losses"], color=color, alpha=alpha, lw=0.7)

    for ci, hd in enumerate(hidden_dims):
        ax = axes[ci]
        ax.set_title(f"hidden_dim={hd}", fontsize=9)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.2)
        if ci == 0:
            ax.set_ylabel("Val Loss")

    ns_handles = [
        mlines.Line2D([], [], color=ns_colors[ns], lw=2, label=f"n_scenes={ns}")
        for ns in n_scenes_list
    ]
    ep_handles = [
        mlines.Line2D([], [], color="gray", alpha=ep_alphas[ep], lw=2,
                       label=f"epochs={ep}")
        for ep in epochs_list
    ]
    fig.legend(
        handles=ns_handles + ep_handles,
        loc="lower center", ncol=5, fontsize=8,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.suptitle("OT-CFM Validation Loss — Hyperparameter Sweep", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(sweep_dir, "sweep_val_loss.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_best_val_summary(results, sweep_dir):
    """
    Best validation loss vs n_scenes, one subplot per hidden_dim,
    separate curves for each epoch count.
    """
    hidden_dims = sorted(PARAM_GRID["hidden_dim"])
    n_scenes_list = sorted(PARAM_GRID["n_scenes"])
    epochs_list = sorted(PARAM_GRID["epochs"])

    best = {}
    for name, data in results.items():
        cfg = data["config"]
        key = (cfg["hidden_dim"], cfg["n_scenes"], cfg["epochs"])
        best[key] = min(data["val_losses"])

    cmap = plt.colormaps.get_cmap("plasma")
    ep_colors = {
        ep: cmap(i / max(len(epochs_list) - 1, 1))
        for i, ep in enumerate(epochs_list)
    }

    ncols = len(hidden_dims)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), squeeze=False)
    axes = axes[0]

    for ci, hd in enumerate(hidden_dims):
        ax = axes[ci]
        for ep in epochs_list:
            vals = [best.get((hd, ns, ep), np.nan) for ns in n_scenes_list]
            ax.plot(n_scenes_list, vals, "o-", color=ep_colors[ep],
                    lw=1.5, ms=4, label=f"ep={ep}")
        ax.set_title(f"hidden_dim={hd}", fontsize=9)
        ax.set_xlabel("n_scenes")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.2)
        if ci == 0:
            ax.set_ylabel("Best Val Loss")

    handles = [
        mlines.Line2D([], [], color=ep_colors[ep], marker="o", lw=1.5, ms=4,
                       label=f"epochs={ep}")
        for ep in epochs_list
    ]
    fig.legend(
        handles=handles,
        loc="lower center", ncol=len(epochs_list), fontsize=8,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.suptitle("OT-CFM Best Validation Loss vs. n_scenes", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(sweep_dir, "sweep_best_val.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="OT-CFM Hyperparameter sweep")
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--jobs_per_gpu", type=int, default=4,
                        help="Concurrent experiments per GPU")
    parser.add_argument("--sweep_dir", type=str, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip training; just re-collect & plot")
    args = parser.parse_args()

    sweep_dir = args.sweep_dir
    os.makedirs(sweep_dir, exist_ok=True)

    # verify pre-computed dataset exists
    for path, label in [(DATASET_TRAIN, "train"), (DATASET_VAL, "val")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} dataset not found at {path}")
            print(f"Generate it first, e.g.:\n"
                  f"  python flow_matching_test/dataset_v3.py "
                  f"--output {path} --n_scenes 20000")
            return

    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    all_combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    total = len(all_combos)

    if not args.plot_only:
        remaining = []
        for p in all_combos:
            hist = os.path.join(sweep_dir, run_name(p), "history.json")
            if not os.path.exists(hist):
                remaining.append(p)

        skipped = total - len(remaining)
        if skipped:
            print(f"Skipping {skipped} already-completed experiments.")

        print(f"Total experiments: {total}  |  To run: {len(remaining)}")
        print(f"GPUs: {args.num_gpus}  |  Jobs/GPU: {args.jobs_per_gpu}")
        print(f"Dataset: {DATASET_TRAIN}")
        print(f"n_rays: {N_RAYS} (fixed)")

        if remaining:
            scheduler = GPUScheduler(args.num_gpus, args.jobs_per_gpu)
            max_workers = args.num_gpus * args.jobs_per_gpu
            completed = 0
            failed = 0

            def _run(params):
                gpu_id = scheduler.acquire()
                try:
                    return run_experiment(params, gpu_id, sweep_dir)
                finally:
                    scheduler.release(gpu_id)

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_run, p): p for p in remaining}
                for future in concurrent.futures.as_completed(futures):
                    name, rc = future.result()
                    completed += 1
                    status = "OK" if rc == 0 else "FAIL"
                    if rc != 0:
                        failed += 1
                    print(f"  [{completed}/{len(remaining)}] {status}: {name}")

            print(f"\nTraining done.  {completed - failed}/{len(remaining)} succeeded, "
                  f"{failed} failed.")

    # ---- collect & plot -------------------------------------------------
    print("\nCollecting results ...")
    results = collect_results(sweep_dir)
    print(f"  Found {len(results)}/{total} completed experiments.")

    if results:
        plot_val_loss_grid(results, sweep_dir)
        plot_best_val_summary(results, sweep_dir)
    else:
        print("  No results to plot.")


if __name__ == "__main__":
    main()
