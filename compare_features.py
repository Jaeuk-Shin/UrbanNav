"""
Compare DINOv2 features across heterogeneous datasets via UMAP or t-SNE.

Each dataset is a directory containing .pt files (as produced by
precompute_features.py).  The script loads features from all datasets,
subsamples to a manageable size, projects to 2D, and produces a scatter
plot colored by dataset origin.

Usage:
    python compare_features.py \
        --dirs /path/to/dataset_A /path/to/dataset_B \
        --labels "CARLA" "YouTube" \
        [--max_files_per_dataset 50] \
        [--max_frames_per_file 200] \
        [--method umap] \
        [--output compare_features.png]

    # t-SNE example with custom perplexity
    python compare_features.py \
        --dirs /data/feat_sim /data/feat_real /data/feat_augmented \
        --labels Sim Real Augmented \
        --method tsne --perplexity 50 \
        --output tsne_3way.png
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_dataset_features(feat_dir, max_files, max_frames_per_file):
    """Load features from .pt files in a single directory.

    Each .pt file is expected to contain either:
      - a dict with a 'features' key  (precompute_features.py format)
      - a raw tensor of shape (N, D)

    Returns
    -------
    features : (M, D) ndarray — concatenated (and subsampled) features.
    """
    pt_files = sorted(f for f in os.listdir(feat_dir) if f.endswith('.pt'))
    if max_files > 0 and len(pt_files) > max_files:
        rng = np.random.default_rng(seed=42)
        pt_files = list(rng.choice(pt_files, size=max_files, replace=False))

    all_feats = []
    for fname in tqdm(pt_files, desc=f"  {os.path.basename(feat_dir)}", leave=False):
        data = torch.load(os.path.join(feat_dir, fname), weights_only=True)
        if isinstance(data, dict):
            feats = data['features']
        else:
            feats = data
        if isinstance(feats, torch.Tensor):
            feats = feats.numpy()

        # Flatten to 2D if needed (e.g. multi-view features stored as (N, V, D))
        if feats.ndim > 2:
            feats = feats.reshape(-1, feats.shape[-1])

        if max_frames_per_file > 0 and feats.shape[0] > max_frames_per_file:
            idx = np.linspace(0, feats.shape[0] - 1, max_frames_per_file, dtype=int)
            feats = feats[idx]

        all_feats.append(feats)

    if not all_feats:
        raise ValueError(f"No .pt files found in {feat_dir}")

    return np.concatenate(all_feats, axis=0)


def run_projection(features, method, **kwargs):
    """Project features to 2D using the specified method."""
    print(f"Running {method.upper()} on {features.shape[0]:,} points "
          f"(dim={features.shape[1]}) ...")

    if method == 'umap':
        try:
            from umap import UMAP
        except ImportError:
            raise ImportError("umap-learn is required: pip install umap-learn")
        reducer = UMAP(
            n_neighbors=kwargs.get('n_neighbors', 15),
            min_dist=kwargs.get('min_dist', 0.1),
            n_components=2,
            metric='cosine',
            random_state=42,
        )
        return reducer.fit_transform(features)

    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=2,
            perplexity=kwargs.get('perplexity', 30),
            metric='cosine',
            init='pca',
            random_state=42,
        )
        return reducer.fit_transform(features)

    else:
        raise ValueError(f"Unknown method: {method!r}. Choose 'umap' or 'tsne'.")


def plot_comparison(embedding, dataset_ids, labels, output_path, method):
    """Scatter plot colored by dataset origin."""
    n_datasets = len(labels)
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab10.colors[:n_datasets] if n_datasets <= 10 else \
        plt.cm.Set3.colors[:n_datasets]

    for i, label in enumerate(labels):
        mask = dataset_ids == i
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[colors[i]], label=f"{label} ({mask.sum():,})",
            s=3, alpha=0.5, rasterized=True,
        )

    ax.legend(markerscale=5, fontsize=10)
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"{method.upper()} — DINOv2 feature comparison "
                 f"({embedding.shape[0]:,} frames, {n_datasets} datasets)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare DINOv2 features across datasets via UMAP / t-SNE")
    parser.add_argument("--dirs", type=str, nargs='+', required=True,
                        help="Directories containing .pt feature files")
    parser.add_argument("--labels", type=str, nargs='+', default=None,
                        help="Labels for each dataset (defaults to directory names)")
    parser.add_argument("--max_files_per_dataset", type=int, default=50,
                        help="Max .pt files to load per dataset (0 = all)")
    parser.add_argument("--max_frames_per_file", type=int, default=200,
                        help="Max frames to keep per .pt file (0 = all)")
    parser.add_argument("--method", type=str, default="umap",
                        choices=["umap", "tsne"],
                        help="Dimensionality reduction method")
    parser.add_argument("--output", type=str, default="compare_features.png")
    # UMAP params
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)
    # t-SNE params
    parser.add_argument("--perplexity", type=float, default=30)
    args = parser.parse_args()

    labels = args.labels or [os.path.basename(d.rstrip('/')) for d in args.dirs]
    if len(labels) != len(args.dirs):
        parser.error("Number of --labels must match number of --dirs")

    all_features = []
    all_ids = []
    for i, feat_dir in enumerate(args.dirs):
        feats = load_dataset_features(
            feat_dir, args.max_files_per_dataset, args.max_frames_per_file)
        print(f"  [{labels[i]}] {feats.shape[0]:,} frames (dim={feats.shape[1]})")
        all_features.append(feats)
        all_ids.append(np.full(feats.shape[0], i, dtype=int))

    features = np.concatenate(all_features, axis=0)
    dataset_ids = np.concatenate(all_ids, axis=0)
    print(f"Total: {features.shape[0]:,} frames")

    embedding = run_projection(
        features, args.method,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        perplexity=args.perplexity,
    )

    plot_comparison(embedding, dataset_ids, labels, args.output, args.method)


if __name__ == "__main__":
    main()
