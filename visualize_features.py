"""
Visualize precomputed DINOv2 features from CarlaFeatDataset via UMAP embedding.

Loads per-frame features from .pt files, projects them to 2D with UMAP,
and produces side-by-side scatter plots colored by episode index and speed.

Usage:
    python visualize_features.py \
        --config config/goal_agnostic_fm.yaml \
        --feature_dir /path/to/precomputed/features \
        [--max_episodes 50] \
        [--max_frames_per_episode 200] \
        [--output umap_features.png] \
        [--n_neighbors 15] \
        [--min_dist 0.1]
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


def load_features_and_poses(cfg, feature_dir, max_episodes, max_frames_per_episode):
    """
    Load per-frame features and corresponding poses.

    Returns
    -------
    features : (N, feature_dim) ndarray
    episode_ids : (N,) int ndarray   — which episode each frame belongs to
    speeds : (N,) float ndarray      — speed at each frame (m/frame)
    episode_names : list[str]
    """
    pose_dir = os.path.join(cfg.data.root_dir, cfg.data.pose_dir)
    pose_fps = cfg.data.pose_fps
    target_fps = cfg.data.target_fps
    pose_step = max(1, pose_fps // target_fps)
    frame_step = cfg.data.video_fps // target_fps

    pose_files = sorted(f for f in os.listdir(pose_dir) if f.endswith('.txt'))
    if max_episodes > 0:
        pose_files = pose_files[:max_episodes]

    all_features = []
    all_episode_ids = []
    all_speeds = []
    episode_names = []

    for ep_idx, pose_file in enumerate(tqdm(pose_files, desc="Loading episodes")):
        name = os.path.splitext(pose_file)[0]
        feat_path = os.path.join(feature_dir, f'{name}.pt')
        if not os.path.exists(feat_path):
            continue

        # Load poses
        pose_path = os.path.join(pose_dir, pose_file)
        raw_pose = np.loadtxt(pose_path, delimiter=" ")[:, 1:]  # drop timestamp col
        pose = raw_pose[::pose_step]

        # Truncate at first NaN
        nan_mask = np.isnan(pose).any(axis=1)
        if np.any(nan_mask):
            pose = pose[:np.argmin(nan_mask)]

        # Load features
        feat_data = torch.load(feat_path, weights_only=True)
        feats = feat_data['features'].numpy()  # (num_frames, feature_dim)

        # Align lengths: features are indexed at frame_step intervals from pose
        # The pose array has one entry per target_fps step; features have one per
        # raw video frame.  We sample features at the same stride the dataset uses.
        n_pose = pose.shape[0]
        n_feat_raw = feats.shape[0]
        frame_indices = frame_step * np.arange(n_pose)
        frame_indices = frame_indices[frame_indices < n_feat_raw]
        n_usable = min(len(frame_indices), n_pose)
        frame_indices = frame_indices[:n_usable]
        pose = pose[:n_usable]
        feats = feats[frame_indices]

        if n_usable < 2:
            continue

        # Subsample if needed
        if max_frames_per_episode > 0 and n_usable > max_frames_per_episode:
            idx = np.linspace(0, n_usable - 1, max_frames_per_episode, dtype=int)
            feats = feats[idx]
            pose = pose[idx]
            n_usable = len(idx)

        # Compute speed from pose (xz displacement between consecutive frames)
        xz = pose[:, [0, 2]]  # standard x, z
        disp = np.linalg.norm(np.diff(xz, axis=0), axis=1)
        speeds = np.concatenate([disp, disp[-1:]])  # repeat last for equal length

        all_features.append(feats)
        all_episode_ids.append(np.full(n_usable, ep_idx, dtype=int))
        all_speeds.append(speeds)
        episode_names.append(name)

    features = np.concatenate(all_features, axis=0)
    episode_ids = np.concatenate(all_episode_ids, axis=0)
    speeds = np.concatenate(all_speeds, axis=0)

    return features, episode_ids, speeds, episode_names


def run_umap(features, n_neighbors=15, min_dist=0.1):
    """Project features to 2D with UMAP."""
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError("umap-learn is required. Install with: pip install umap-learn")

    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2,
                   metric='cosine', random_state=42)
    print(f"Running UMAP on {features.shape[0]} points "
          f"(dim={features.shape[1]}, n_neighbors={n_neighbors}, "
          f"min_dist={min_dist}) ...")
    embedding = reducer.fit_transform(features)
    return embedding


def plot_embedding(embedding, episode_ids, speeds, episode_names, output_path):
    """Side-by-side scatter: colored by episode (left) and speed (right)."""
    fig, (ax_ep, ax_spd) = plt.subplots(1, 2, figsize=(18, 8))

    x, y = embedding[:, 0], embedding[:, 1]
    n_episodes = len(episode_names)

    # --- Left: episode index ---
    cmap_ep = cm.get_cmap('tab20', n_episodes) if n_episodes <= 20 else cm.get_cmap('hsv', n_episodes)
    sc_ep = ax_ep.scatter(x, y, c=episode_ids, cmap=cmap_ep, s=3, alpha=0.6,
                          rasterized=True)
    cbar_ep = fig.colorbar(sc_ep, ax=ax_ep, shrink=0.8)
    cbar_ep.set_label("Episode index")
    if n_episodes <= 20:
        cbar_ep.set_ticks(range(n_episodes))
    ax_ep.set_title("Colored by episode")
    ax_ep.set_xlabel("UMAP 1")
    ax_ep.set_ylabel("UMAP 2")
    ax_ep.set_aspect("equal", adjustable="datalim")

    # --- Right: speed ---
    # Clip speed to 99th percentile for better color contrast
    spd_clip = np.clip(speeds, 0, np.percentile(speeds, 99))
    sc_spd = ax_spd.scatter(x, y, c=spd_clip, cmap='viridis', s=3, alpha=0.6,
                            rasterized=True)
    cbar_spd = fig.colorbar(sc_spd, ax=ax_spd, shrink=0.8)
    cbar_spd.set_label("Speed (m / frame)")
    ax_spd.set_title("Colored by speed")
    ax_spd.set_xlabel("UMAP 1")
    ax_spd.set_ylabel("UMAP 2")
    ax_spd.set_aspect("equal", adjustable="datalim")

    fig.suptitle(f"UMAP of DINOv2 features ({embedding.shape[0]:,} frames, "
                 f"{n_episodes} episodes)", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="UMAP visualization of precomputed DINOv2 features")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--feature_dir", type=str, required=True,
                        help="Directory with precomputed .pt feature files")
    parser.add_argument("--max_episodes", type=int, default=50,
                        help="Max episodes to load (0 = all)")
    parser.add_argument("--max_frames_per_episode", type=int, default=200,
                        help="Max frames per episode (0 = all)")
    parser.add_argument("--output", type=str, default="umap_features.png")
    parser.add_argument("--n_neighbors", type=int, default=15,
                        help="UMAP n_neighbors (larger = more global structure)")
    parser.add_argument("--min_dist", type=float, default=0.1,
                        help="UMAP min_dist (smaller = tighter clusters)")
    args = parser.parse_args()

    from config.utils import load_config
    cfg = load_config(args.config)

    features, episode_ids, speeds, episode_names = load_features_and_poses(
        cfg, args.feature_dir, args.max_episodes, args.max_frames_per_episode,
    )
    print(f"Loaded {features.shape[0]:,} frames from {len(episode_names)} episodes "
          f"(feature dim = {features.shape[1]})")

    embedding = run_umap(features, n_neighbors=args.n_neighbors,
                         min_dist=args.min_dist)

    plot_embedding(embedding, episode_ids, speeds, episode_names, args.output)


if __name__ == "__main__":
    main()
