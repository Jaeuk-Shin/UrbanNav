"""Validation-only script — runs validation and saves visualizations without training.

Usage (Hydra — all arguments are key=value overrides, no ``--`` prefix):

    python validate.py --config-name urban_nav_feat checkpoint=path/to/best.ckpt
    python validate.py --config-name urban_nav_feat checkpoint=path/to/best.ckpt \
        model.type=flow_matching_feat data.type=carla_feat
    python validate.py --config-name urban_nav_feat checkpoint=path/to/best.ckpt \
        validation.num_visualize=200

    # Perturbation sensitivity analysis (empirical Wasserstein distance)
    python validate.py --config-name urban_nav_feat checkpoint=path/to/best.ckpt \
        perturbation.mode=zero                          # zero-out input coordinates
    python validate.py --config-name urban_nav_feat checkpoint=path/to/best.ckpt \
        perturbation.mode=noise perturbation.noise_std=0.1   # Gaussian perturbation
    python validate.py --config-name urban_nav_feat checkpoint=path/to/best.ckpt \
        perturbation.mode=sweep                         # sweep noise_std from 0.01 to 1.0

    # On-manifold perturbation modes
    python validate.py --config-name urban_nav_feat checkpoint=path/to/best.ckpt \
        perturbation.mode=shuffle                       # swap coords across batch
    python validate.py --config-name urban_nav_feat checkpoint=path/to/best.ckpt \
        perturbation.mode=scale perturbation.scale=0.5  # multiply coords by scalar
    python validate.py --config-name urban_nav_feat checkpoint=path/to/best.ckpt \
        perturbation.mode=scale_sweep                   # sweep scale from 0.0 to 2.0

Outputs are saved to {result_dir}/val_vis/epoch_0/.
"""

import os
import sys

import pytorch_lightning as pl
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision('medium')
pl.seed_everything(42, workers=True)

from pl_modules.citywalk_datamodule import CityWalkDataModule
from pl_modules.carla_datamodule import CarlaDataModule
from pl_modules.carla_feat_datamodule import CarlaFeatDataModule
from pl_modules.citywalk_feat_datamodule import CityWalkFeatDataModule
from pl_modules.urban_nav_feat_mixture_datamodule import UrbanNavFeatMixtureDataModule
from pl_modules.flow_matching_module import FlowMatchingModule
from pl_modules.flow_matching_feat_module import FlowMatchingFeatModule
from vis_utils import project_waypoints_onto_image_plane

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
# Empirical Wasserstein distance via minimum-weight bipartite matching
# ---------------------------------------------------------------------------

def empirical_wasserstein(samples_a, samples_b):
    """
    compute the 1-Wasserstein distance between two empirical trajectory
    distributions using Hungarian algorithm

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

# ---------------------------------------------------------------------------
# Perturbation visualization
# ---------------------------------------------------------------------------

COLOR_GT = '#92DB58'
COLOR_ORIG = '#DB6057'
COLOR_PERT = '#A157DB'
COLOR_INPUT_ORIG = '#5771DB'
COLOR_INPUT_NOISY = '#DBC257'
MAX_VIS_SAMPLES = 50  # cap drawn trajectories per distribution on image panel


def _load_image(batch, idx):
    """Try to load a raw RGB image for sample *idx* from the batch."""
    # Prefer pre-split image file
    if ('image_path' in batch and idx < len(batch['image_path'])
            and batch['image_path'][idx]
            and os.path.exists(batch['image_path'][idx])):
        return Image.open(batch['image_path'][idx]).convert('RGB')
    # Fall back to video extraction
    if ('video_path' in batch and idx < len(batch['video_path'])
            and batch['video_path'][idx]
            and batch['video_frame_idx'][idx].item() >= 0):
        from decord import VideoReader, cpu
        vr = VideoReader(batch['video_path'][idx], ctx=cpu(0))
        fi = min(batch['video_frame_idx'][idx].item(), len(vr) - 1)
        return Image.fromarray(vr[fi].asnumpy())
    return None


def _get_camera(batch, idx):
    """Return (fx, fy, cx, cy, dw, dh, fov) or None."""
    cam_all = batch.get('camera_intrinsics')
    if cam_all is None or cam_all[idx, 0].item() <= 0:
        return None
    fx, fy, cx, cy, dw, dh = cam_all[idx].tolist()
    fov = 2.0 * np.degrees(np.arctan(cx / fx)) if fx > 0 else None
    return fx, fy, cx, cy, dw, dh, fov


def _draw_samples_on_image(ax, img, samples, gt_waypoints_y, cam, color,
                           alpha=0.6):
    """Project trajectory samples onto an image panel."""
    fx, fy, cx, cy, dw, dh, _ = cam
    W_orig, H_orig = img.size
    dw, dh = int(dw), int(dh)
    left = max(0, (W_orig - dw) // 2)
    top = max(0, (H_orig - dh) // 2)
    K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])

    n = min(samples.shape[0], MAX_VIS_SAMPLES)
    for s in range(n):
        u, v, valid = project_waypoints_onto_image_plane(
            samples[s], gt_waypoints_y, K=K)
        u, v = u - left, v - top
        if np.all(valid):
            ax.plot(u, v, color=color, alpha=alpha, linewidth=1)
    ax.set_xlim(0.0, dw)
    ax.set_ylim(dh, 0.0)


def visualize_perturbation(batch, idx, samples_orig, samples_pert, w1,
                           mode_label, vis_dir, vis_idx):
    """Save a perturbation comparison figure for one sample.

    Layout: up to 3 panels —
      [RGB + original]  [RGB + perturbed]  [2-D coord overlay + W1]
    Falls back to a single coord panel when no image is available.
    """
    gt_wp = batch['gt_waypoints'][idx].cpu().numpy()
    gt_wp_y = batch['gt_waypoints_y'][idx].cpu().numpy()
    orig_input = batch['original_input_positions'][idx].cpu().numpy()
    noisy_input = batch['noisy_input_positions'][idx].cpu().numpy()

    img = _load_image(batch, idx)
    cam = _get_camera(batch, idx)
    has_image = img is not None and cam is not None

    if has_image:
        dw, dh = int(cam[4]), int(cam[5])
        W_orig, H_orig = img.size
        left = max(0, (W_orig - dw) // 2)
        top = max(0, (H_orig - dh) // 2)
        if W_orig != dw or H_orig != dh:
            img_crop = img.crop((left, top, left + dw, top + dh))
        else:
            img_crop = img
        K = np.array([[cam[0], 0., cam[2]],
                       [0., cam[1], cam[3]],
                       [0., 0., 1.]])

        fig, (ax_orig, ax_pert, ax_coord) = plt.subplots(
            1, 3, figsize=(18, 6))
        plt.subplots_adjust(wspace=0.3)

        # --- left: RGB + original samples ---
        ax_orig.imshow(np.array(img_crop))
        ax_orig.axis('off')
        ax_orig.set_title('Original coords', fontsize=12)
        u_gt, v_gt, valid = project_waypoints_onto_image_plane(
            gt_wp, gt_wp_y, K=K)
        u_gt, v_gt = u_gt - left, v_gt - top
        if np.all(valid):
            ax_orig.plot(u_gt, v_gt, color=COLOR_GT, linewidth=3)
        _draw_samples_on_image(ax_orig, img, samples_orig, gt_wp_y, cam,
                               COLOR_ORIG)

        # --- middle: RGB + perturbed samples ---
        ax_pert.imshow(np.array(img_crop))
        ax_pert.axis('off')
        ax_pert.set_title(f'Perturbed ({mode_label})', fontsize=12)
        if np.all(valid):
            ax_pert.plot(u_gt, v_gt, color=COLOR_GT, linewidth=3)
        _draw_samples_on_image(ax_pert, img, samples_pert, gt_wp_y, cam,
                               COLOR_PERT)
    else:
        fig, ax_coord = plt.subplots(figsize=(7, 6))

    # --- right (or only): 2-D coordinate overlay ---
    fov = cam[6] if cam is not None else None
    if fov is not None:
        th = np.pi / 2.0 - np.deg2rad(fov) / 2.0
        r = np.linspace(0.0, 7.0, num=100)
        c, s = np.cos(th), np.sin(th)
        ax_coord.plot(r * c, r * s, ls='dashed', color='tab:gray')
        ax_coord.plot(-r * c, r * s, ls='dashed', color='tab:gray')

    ax_coord.axis('equal')
    ax_coord.plot(orig_input[:, 0], orig_input[:, 1],
                  'o-', color=COLOR_INPUT_ORIG, label='Input (orig)')
    ax_coord.plot(noisy_input[:, 0], noisy_input[:, 1],
                  'o-', color=COLOR_INPUT_NOISY, label='Input (noisy)')
    ax_coord.plot(gt_wp[:, 0], gt_wp[:, 1],
                  'X-', color=COLOR_GT, label='GT', linewidth=2)

    # Draw trajectory samples
    for s in range(min(samples_orig.shape[0], MAX_VIS_SAMPLES)):
        ax_coord.plot(samples_orig[s, :, 0], samples_orig[s, :, 1],
                      color=COLOR_ORIG, alpha=0.5,
                      label='Original' if s == 0 else None)
    for s in range(min(samples_pert.shape[0], MAX_VIS_SAMPLES)):
        ax_coord.plot(samples_pert[s, :, 0], samples_pert[s, :, 1],
                      color=COLOR_PERT, alpha=0.5,
                      label='Perturbed' if s == 0 else None)

    ax_coord.set_title(f'$W_1$ = {w1:.4f} m', fontsize=14)
    ax_coord.set_xlabel('X (m)')
    ax_coord.set_ylabel('Z (m)')
    ax_coord.legend(fontsize=8, loc='upper left')
    ax_coord.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(vis_dir, exist_ok=True)
    fig.savefig(os.path.join(vis_dir, f'pert_{vis_idx:04d}.png'), dpi=150)
    plt.close(fig)


def perturb_coordinates(cord, mode, noise_std=0.0, scale=1.0):
    """Return a perturbed copy of the input coordinates.

    Parameters
    ----------
    cord : torch.Tensor, shape (B, N, 2)
    mode : str
        'zero'    — replace all coordinates with zeros.
        'noise'   — add i.i.d. Gaussian noise with std ``noise_std``.
        'shuffle' — randomly permute coordinates across the batch dimension
                     so each sample gets another sample's real coordinates.
        'scale'   — multiply all coordinates by ``scale``.
    noise_std : float
        Standard deviation for the 'noise' mode.
    scale : float
        Multiplicative factor for the 'scale' mode.

    Returns
    -------
    cord_perturbed : torch.Tensor, same shape as ``cord``
    """
    if mode == 'zero':
        return torch.zeros_like(cord)
    elif mode == 'noise':
        return cord + torch.randn_like(cord) * noise_std
    elif mode == 'shuffle':
        perm = torch.randperm(cord.shape[0], device=cord.device)
        # Avoid identity permutation when possible (re-roll once)
        if cord.shape[0] > 1 and (perm == torch.arange(cord.shape[0],
                                                        device=cord.device)).all():
            perm = torch.randperm(cord.shape[0], device=cord.device)
        return cord[perm]
    elif mode == 'scale':
        return cord * scale
    else:
        raise ValueError(f"Unknown perturbation mode: {mode}")


@torch.no_grad()
def run_perturbation_analysis(model, dataloader, mode, noise_std=0.0,
                              coord_scale=1.0, num_samples=200, device='cuda',
                              vis_dir=None, num_visualize=0):
    """Run perturbation sensitivity analysis over the validation set.

    For each batch, draw ``num_samples`` trajectory samples with original
    coordinates and with perturbed coordinates, then measure the empirical
    W1 distance between the two distributions.

    Parameters
    ----------
    model : FlowMatchingModule or FlowMatchingFeatModule
    dataloader : DataLoader
    mode : str
        Perturbation mode ('zero', 'noise', 'shuffle', or 'scale').
    noise_std : float
        Noise std (only used when mode='noise').
    coord_scale : float
        Scale factor (only used when mode='scale').
    num_samples : int
        Number of trajectory samples to draw per distribution.
    device : str
    vis_dir : str or None
        Directory for saving per-sample comparison figures.
    num_visualize : int
        Number of samples to visualize (0 = no visualization).

    Returns
    -------
    w1_all : np.ndarray, shape (N_total,)
        Per-sample W1 distances across the entire validation set.
    """
    is_feat = isinstance(model, FlowMatchingFeatModule)
    inner = model.model  # the actual nn.Module
    model.to(device)

    mode_labels = {
        'zero': 'zero',
        'noise': f'noise {noise_std:.3g}',
        'shuffle': 'shuffle',
        'scale': f'scale {coord_scale:.3g}',
    }
    mode_label = mode_labels.get(mode, mode)

    w1_list = []
    vis_count = 0

    for batch in tqdm(dataloader, desc=f"Perturbation ({mode_label})"):
        # Move tensors to device
        if is_feat:
            obs = batch['obs_features'].to(device)
        else:
            obs = batch['video_frames'].to(device)
        cord = batch['input_positions'].to(device)
        step_scale = batch['step_scale'].to(device)
        B = cord.shape[0]

        # Scale factor for converting model output to metres
        out_scale = step_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Original samples: (B, S, T, 2)
        samples_orig = inner.sample(
            obs, cord, num_samples=num_samples) * out_scale

        # Perturbed samples
        cord_pert = perturb_coordinates(cord, mode, noise_std, coord_scale)
        samples_pert = inner.sample(
            obs, cord_pert, num_samples=num_samples) * out_scale

        # Compute W1 per batch element
        samples_orig_np = samples_orig.cpu().numpy()
        samples_pert_np = samples_pert.cpu().numpy()
        for i in range(B):
            w1 = empirical_wasserstein(samples_orig_np[i], samples_pert_np[i])
            w1_list.append(w1)

            if vis_dir and vis_count < num_visualize:
                visualize_perturbation(
                    batch, i, samples_orig_np[i], samples_pert_np[i],
                    w1, mode_label, vis_dir, vis_count)
                vis_count += 1

    return np.array(w1_list)


@hydra.main(config_path="config", config_name="urban_nav", version_base=None)
def main(cfg):
    checkpoint_path = cfg.get("checkpoint", None)
    if not checkpoint_path:
        print("Error: checkpoint= is required. Usage:")
        print("  python validate.py --config-name urban_nav_feat checkpoint=path/to/best.ckpt")
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
        coord_scale = float(pert_cfg.get("scale", 1.0))
        num_samples = int(pert_cfg.get("num_samples", 200))
        num_vis = int(pert_cfg.get("num_visualize",
                                   cfg.validation.num_visualize))
        device = 'cuda' if num_gpu > 0 else 'cpu'

        datamodule.setup('validate')
        val_loader = datamodule.val_dataloader()

        def _run_and_report(mode, label, vis_subdir, **kwargs):
            """Run one perturbation analysis, print stats, return W1 array."""
            vd = os.path.join(result_dir, 'pert_vis', vis_subdir)
            w1 = run_perturbation_analysis(
                model, val_loader, mode, num_samples=num_samples,
                device=device, vis_dir=vd, num_visualize=num_vis, **kwargs)
            print(f"  {label}:  W1 mean={w1.mean():.4f}"
                  f"  median={np.median(w1):.4f}"
                  f"  std={w1.std():.4f}"
                  f"  max={w1.max():.4f}")
            return w1

        if pert_mode == 'sweep':
            # Sweep over Gaussian noise levels
            stds = [float(s) for s in pert_cfg.get(
                "sweep_stds", [0.01, 0.05, 0.1, 0.2, 0.5, 1.0])]
            print(f"\n--- Noise Sweep ({len(stds)} levels,"
                  f" {num_samples} samples, {num_vis} vis) ---")
            sweep_results = {}
            for std in stds:
                w1_arr = _run_and_report(
                    'noise', f'noise_std={std:.3g}', f'noise_{std:.3g}',
                    noise_std=std)
                sweep_results[std] = w1_arr
            save_path = os.path.join(result_dir, 'perturbation_sweep.npz')
            np.savez(save_path, stds=np.array(stds),
                     **{f'w1_std{s:.3g}': v
                        for s, v in sweep_results.items()})
            print(f"Sweep results saved to: {save_path}")

        elif pert_mode == 'scale_sweep':
            # Sweep over coordinate scale factors
            scales = [float(s) for s in pert_cfg.get(
                "sweep_scales", [0.0, 0.25, 0.5, 0.75, 1.5, 2.0])]
            print(f"\n--- Scale Sweep ({len(scales)} levels,"
                  f" {num_samples} samples, {num_vis} vis) ---")
            sweep_results = {}
            for sc in scales:
                w1_arr = _run_and_report(
                    'scale', f'scale={sc:.3g}', f'scale_{sc:.3g}',
                    coord_scale=sc)
                sweep_results[sc] = w1_arr
            save_path = os.path.join(result_dir,
                                     'perturbation_scale_sweep.npz')
            np.savez(save_path, scales=np.array(scales),
                     **{f'w1_scale{s:.3g}': v
                        for s, v in sweep_results.items()})
            print(f"Sweep results saved to: {save_path}")

        else:
            # Single perturbation mode
            print(f"\n--- Perturbation Analysis (mode={pert_mode},"
                  f" {num_samples} samples, {num_vis} vis) ---")
            w1_arr = _run_and_report(
                pert_mode, pert_mode, pert_mode,
                noise_std=noise_std, coord_scale=coord_scale)
            save_path = os.path.join(result_dir,
                                     f'perturbation_{pert_mode}.npy')
            np.save(save_path, w1_arr)
            vis_dir = os.path.join(result_dir, 'pert_vis', pert_mode)
            print(f"Per-sample W1 saved to: {save_path}")
            print(f"Visualizations saved to: {vis_dir}/")


if __name__ == '__main__':
    main()
