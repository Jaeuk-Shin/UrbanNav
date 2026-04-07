"""Validation-only script — runs validation and saves visualizations without training.

Usage:
    python validate.py --config config/urban_nav_feat.yaml --checkpoint path/to/best.ckpt
    python validate.py --config config/urban_nav_feat.yaml --checkpoint path/to/best.ckpt \
        model.type=flow_matching_feat data.type=carla_feat
    python validate.py --config config/urban_nav_feat.yaml --checkpoint path/to/best.ckpt \
        validation.num_visualize=200

    # Perturbation sensitivity analysis (empirical Wasserstein distance)
    python validate.py --config config/urban_nav_feat.yaml --checkpoint path/to/best.ckpt \
        perturbation.mode=zero                          # zero-out input coordinates
    python validate.py --config config/urban_nav_feat.yaml --checkpoint path/to/best.ckpt \
        perturbation.mode=noise perturbation.noise_std=0.1   # Gaussian perturbation
    python validate.py --config config/urban_nav_feat.yaml --checkpoint path/to/best.ckpt \
        perturbation.mode=sweep                         # sweep noise_std from 0.01 to 1.0

Outputs are saved to {result_dir}/val_vis/epoch_0/.
"""

import os
import sys

import pytorch_lightning as pl
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm

torch.set_float32_matmul_precision('medium')
pl.seed_everything(42, workers=True)

from pl_modules.citywalk_datamodule import CityWalkDataModule
from pl_modules.carla_datamodule import CarlaDataModule
from pl_modules.carla_feat_datamodule import CarlaFeatDataModule
from pl_modules.citywalk_feat_datamodule import CityWalkFeatDataModule
from pl_modules.urban_nav_feat_mixture_datamodule import UrbanNavFeatMixtureDataModule
from pl_modules.flow_matching_module import FlowMatchingModule
from pl_modules.flow_matching_feat_module import FlowMatchingFeatModule

import hydra
from omegaconf import OmegaConf


def build_datamodule(cfg):
    constructors = {
        'carla': CarlaDataModule,
        'citywalk': CityWalkDataModule,
        'carla_feat': CarlaFeatDataModule,
        'citywalk_feat': CityWalkFeatDataModule,
        'urban_nav_feat_mixture': UrbanNavFeatMixtureDataModule,
    }
    if cfg.data.type not in constructors:
        raise ValueError(f"Unsupported data type for validation: {cfg.data.type}")
    return constructors[cfg.data.type](cfg)


def load_model(cfg, checkpoint_path):
    if cfg.model.type in ('flow_matching', 'citywalker_fm'):
        model = FlowMatchingModule.load_from_checkpoint(checkpoint_path, cfg=cfg)
    elif cfg.model.type == 'flow_matching_feat':
        model = FlowMatchingFeatModule.load_from_checkpoint(checkpoint_path, cfg=cfg)
    else:
        raise ValueError(f"Unsupported model type for validation: {cfg.model.type}")
    return model


# ---------------------------------------------------------------------------
# Empirical Wasserstein distance via optimal bipartite matching
# ---------------------------------------------------------------------------

def empirical_wasserstein(samples_a, samples_b):
    """Compute the 1-Wasserstein distance between two empirical trajectory
    distributions using minimum-weight bipartite matching (Hungarian algorithm).

    Parameters
    ----------
    samples_a, samples_b : np.ndarray, shape (S, T, 2)
        Two sets of S trajectory samples, each with T waypoints in 2-D.

    Returns
    -------
    w1 : float
        W1 distance (average matched cost under L2 ground metric).
    """
    S = samples_a.shape[0]
    flat_a = samples_a.reshape(S, -1)  # (S, T*2)
    flat_b = samples_b.reshape(S, -1)
    cost = cdist(flat_a, flat_b, metric='euclidean')  # (S, S)
    row_ind, col_ind = linear_sum_assignment(cost)
    return cost[row_ind, col_ind].mean()


# ---------------------------------------------------------------------------
# Perturbation analysis
# ---------------------------------------------------------------------------

def perturb_coordinates(cord, mode, noise_std=0.0):
    """Return a perturbed copy of the input coordinates.

    Parameters
    ----------
    cord : torch.Tensor, shape (B, N, 2)
    mode : str
        'zero'  — replace all coordinates with zeros.
        'noise' — add i.i.d. Gaussian noise with std ``noise_std``.
    noise_std : float
        Standard deviation for the 'noise' mode.

    Returns
    -------
    cord_perturbed : torch.Tensor, same shape as ``cord``
    """
    if mode == 'zero':
        return torch.zeros_like(cord)
    elif mode == 'noise':
        return cord + torch.randn_like(cord) * noise_std
    else:
        raise ValueError(f"Unknown perturbation mode: {mode}")


@torch.no_grad()
def run_perturbation_analysis(model, dataloader, mode, noise_std=0.0,
                              num_samples=200, device='cuda'):
    """Run perturbation sensitivity analysis over the validation set.

    For each batch, draw ``num_samples`` trajectory samples with original
    coordinates and with perturbed coordinates, then measure the empirical
    W1 distance between the two distributions.

    Parameters
    ----------
    model : FlowMatchingModule or FlowMatchingFeatModule
    dataloader : DataLoader
    mode : str
        Perturbation mode ('zero' or 'noise').
    noise_std : float
        Noise std (only used when mode='noise').
    num_samples : int
        Number of trajectory samples to draw per distribution.
    device : str

    Returns
    -------
    w1_all : np.ndarray, shape (N_total,)
        Per-sample W1 distances across the entire validation set.
    """
    is_feat = isinstance(model, FlowMatchingFeatModule)
    inner = model.model  # the actual nn.Module
    model.to(device)

    w1_list = []

    for batch in tqdm(dataloader, desc=f"Perturbation ({mode},"
                      f" std={noise_std:.3g})"):
        # Move tensors to device
        if is_feat:
            obs = batch['obs_features'].to(device)
        else:
            obs = batch['video_frames'].to(device)
        cord = batch['input_positions'].to(device)
        step_scale = batch['step_scale'].to(device)
        B = cord.shape[0]

        # Scale factor for converting model output to metres
        scale = step_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Original samples: (B, S, T, 2)
        samples_orig = inner.sample(obs, cord, num_samples=num_samples) * scale

        # Perturbed samples
        cord_pert = perturb_coordinates(cord, mode, noise_std)
        samples_pert = inner.sample(obs, cord_pert, num_samples=num_samples) * scale

        # Compute W1 per batch element
        samples_orig_np = samples_orig.cpu().numpy()
        samples_pert_np = samples_pert.cpu().numpy()
        for i in range(B):
            w1 = empirical_wasserstein(samples_orig_np[i], samples_pert_np[i])
            w1_list.append(w1)

    return np.array(w1_list)


@hydra.main(config_path="config", config_name="urban_nav", version_base=None)
def main(cfg):
    checkpoint_path = cfg.get("checkpoint", None)
    if not checkpoint_path:
        print("Error: --checkpoint is required. Usage:")
        print("  python validate.py --config config/goal_agnostic_fm.yaml --checkpoint path/to/best.ckpt")
        sys.exit(1)
    if not os.path.isfile(checkpoint_path):
        print(f"Error: checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Set up output directory
    result_dir = os.path.join(cfg.project.result_dir, cfg.project.run_name)
    os.makedirs(result_dir, exist_ok=True)
    cfg.project.result_dir = result_dir

    print(f"Checkpoint : {checkpoint_path}")
    print(f"Output dir : {result_dir}/val_vis/epoch_0/")
    print(f"Model type : {cfg.model.type}")
    print(f"Data type  : {cfg.data.type}")
    print(f"Num vis    : {cfg.validation.num_visualize}")

    datamodule = build_datamodule(cfg)
    model = load_model(cfg, checkpoint_path)
    model.eval()

    print(pl.utilities.model_summary.ModelSummary(model, max_depth=2))

    num_gpu = torch.cuda.device_count()
    trainer = pl.Trainer(
        default_root_dir=result_dir,
        devices=max(num_gpu, 1),
        accelerator='gpu' if num_gpu > 0 else 'cpu',
        logger=False,
        enable_checkpointing=False,
        callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=10)],
    )

    results = trainer.validate(model, datamodule=datamodule)

    print("\n--- Validation Results ---")
    for k, v in results[0].items():
        print(f"  {k}: {v:.6f}")
    print(f"\nVisualizations saved to: {result_dir}/val_vis/epoch_0/")

    # ------------------------------------------------------------------
    # Perturbation sensitivity analysis
    # ------------------------------------------------------------------
    pert_cfg = cfg.get("perturbation", None)
    if pert_cfg is not None:
        pert_mode = pert_cfg.get("mode", "zero")
        noise_std = float(pert_cfg.get("noise_std", 0.1))
        num_samples = int(pert_cfg.get("num_samples", 200))
        device = 'cuda' if num_gpu > 0 else 'cpu'

        datamodule.setup('validate')
        val_loader = datamodule.val_dataloader()

        if pert_mode == 'sweep':
            # Sweep over multiple noise levels
            stds = [float(s) for s in pert_cfg.get(
                "sweep_stds", [0.01, 0.05, 0.1, 0.2, 0.5, 1.0])]
            print(f"\n--- Perturbation Sweep ({len(stds)} levels,"
                  f" {num_samples} samples) ---")
            sweep_results = {}
            for std in stds:
                w1_arr = run_perturbation_analysis(
                    model, val_loader, 'noise', noise_std=std,
                    num_samples=num_samples, device=device)
                sweep_results[std] = w1_arr
                print(f"  noise_std={std:.3g}:  W1 mean={w1_arr.mean():.4f}"
                      f"  median={np.median(w1_arr):.4f}"
                      f"  std={w1_arr.std():.4f}"
                      f"  max={w1_arr.max():.4f}")
            save_path = os.path.join(result_dir, 'perturbation_sweep.npz')
            np.savez(save_path,
                     stds=np.array(stds),
                     **{f'w1_std{s:.3g}': v for s, v in sweep_results.items()})
            print(f"Sweep results saved to: {save_path}")
        else:
            # Single perturbation mode (zero or noise)
            print(f"\n--- Perturbation Analysis (mode={pert_mode},"
                  f" noise_std={noise_std:.3g}, {num_samples} samples) ---")
            w1_arr = run_perturbation_analysis(
                model, val_loader, pert_mode, noise_std=noise_std,
                num_samples=num_samples, device=device)
            print(f"  W1 mean   = {w1_arr.mean():.4f}")
            print(f"  W1 median = {np.median(w1_arr):.4f}")
            print(f"  W1 std    = {w1_arr.std():.4f}")
            print(f"  W1 max    = {w1_arr.max():.4f}")
            save_path = os.path.join(result_dir,
                                     f'perturbation_{pert_mode}.npy')
            np.save(save_path, w1_arr)
            print(f"Per-sample W1 saved to: {save_path}")


if __name__ == '__main__':
    main()
