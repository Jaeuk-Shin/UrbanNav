"""
Linear probing to measure domain separability between datasets.

Loads precomputed DINOv2 features from datasets defined in a mixture config,
balances them to equal size, and trains logistic regression to predict dataset
membership.  Accuracy ≈ 0.5 → domains overlap; ≈ 1.0 → domains are separable.

Usage:
    python analysis/linear_probe.py \
        --config config/urban_nav_feat.yaml \
        [--max_episodes 50] \
        [--max_frames_per_episode 200] \
        [--n_folds 5] \
        [--seed 42]
"""

import argparse
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.utils import load_config
from omegaconf import OmegaConf
from visualize_features import load_features_and_poses


def load_mixture_features(cfg, max_episodes, max_frames_per_episode):
    """Load features from each dataset in the mixture config.

    Returns list of (features, dataset_name) tuples.
    """
    mixture = OmegaConf.to_container(cfg.data.mixture, resolve=True)
    assert len(mixture) >= 2, "Need at least 2 datasets in mixture"

    datasets = []
    for entry in mixture:
        root = entry["root"]
        pose_dir = entry.get("pose_subdir", "pose")
        feature_dir = entry.get(
            "feature_dir", f"{root}/{entry.get('feature_subdir', 'dino')}"
        )

        sub_cfg = OmegaConf.create(
            {
                "data": {
                    "root_dir": root,
                    "pose_dir": pose_dir,
                    "pose_fps": cfg.data.pose_fps,
                    "target_fps": cfg.data.target_fps,
                    "video_fps": cfg.data.video_fps,
                }
            }
        )

        features, _, _, episode_names = load_features_and_poses(
            sub_cfg, feature_dir, max_episodes, max_frames_per_episode
        )
        name = os.path.basename(root.rstrip("/"))
        print(
            f"  [{name}] {features.shape[0]:,} frames from "
            f"{len(episode_names)} episodes  (dim={features.shape[1]})"
        )
        datasets.append((features, name))

    return datasets


def run_linear_probe(datasets, n_folds, seed):
    """Train logistic regression to distinguish dataset membership."""
    rng = np.random.default_rng(seed)

    # Balance to equal size via random subsampling
    sizes = [feats.shape[0] for feats, _ in datasets]
    min_size = min(sizes)
    print(f"\nBalancing: {' vs '.join(str(s) for s in sizes)} → {min_size} each")

    balanced_X, balanced_y = [], []
    for label, (feats, _) in enumerate(datasets):
        idx = rng.choice(feats.shape[0], size=min_size, replace=False)
        balanced_X.append(feats[idx])
        balanced_y.append(np.full(min_size, label, dtype=int))

    X = np.concatenate(balanced_X)
    y = np.concatenate(balanced_y)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Stratified k-fold cross-validation
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_accs = []
    y_prob = np.zeros(len(y), dtype=float)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        clf.fit(X[train_idx], y[train_idx])
        acc = clf.score(X[test_idx], y[test_idx])
        fold_accs.append(acc)
        y_prob[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    auc = roc_auc_score(y, y_prob)

    return mean_acc, std_acc, fold_accs, auc, min_size, X.shape[1]


def main():
    parser = argparse.ArgumentParser(
        description="Linear probe for dataset domain separability"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_episodes", type=int, default=50)
    parser.add_argument("--max_frames_per_episode", type=int, default=200)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    mixture = OmegaConf.to_container(cfg.data.mixture, resolve=True)
    dataset_names = [
        os.path.basename(e["root"].rstrip("/")) for e in mixture
    ]

    print("Loading features ...")
    datasets = load_mixture_features(
        cfg, args.max_episodes, args.max_frames_per_episode
    )

    mean_acc, std_acc, fold_accs, auc, n_per_ds, feat_dim = run_linear_probe(
        datasets, args.n_folds, args.seed
    )

    print(f"\n{'=' * 55}")
    print("  Linear Probe  —  Domain Separability Test")
    print(f"{'=' * 55}")
    print(f"  Datasets      : {' vs '.join(dataset_names)}")
    print(f"  Samples/class : {n_per_ds:,}")
    print(f"  Feature dim   : {feat_dim}")
    print(f"  {args.n_folds}-fold CV acc  : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  ROC-AUC       : {auc:.4f}")
    print(f"  Per-fold accs : {[f'{a:.4f}' for a in fold_accs]}")
    print(f"{'=' * 55}")

    if mean_acc > 0.90:
        verdict = "CLEARLY SEPARABLE  — large domain gap"
    elif mean_acc > 0.75:
        verdict = "MODERATELY SEPARABLE  — noticeable domain gap"
    elif mean_acc > 0.60:
        verdict = "WEAKLY SEPARABLE  — mild domain gap"
    else:
        verdict = "INDISTINGUISHABLE  — domains overlap well"
    print(f"  → {verdict}")
    print()


if __name__ == "__main__":
    main()
