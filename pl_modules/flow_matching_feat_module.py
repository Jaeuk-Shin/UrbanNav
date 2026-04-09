import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from model.flow_matching_feat import FlowMatchingFeat
from vis_utils import project_waypoints_onto_image_plane
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
mpl.use('Agg')
import os
import cv2


class FlowMatchingFeatModule(pl.LightningModule):
    """
    Lightning module for training FlowMatchingFeat on precomputed features.

    To initialise from a pretrained FlowMatchingTrajectorySampler checkpoint,
    pass the path via cfg.checkpoint. The encoder weights will be skipped
    automatically (strict=False).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = FlowMatchingFeat(cfg)
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

        self.val_num_visualize = cfg.validation.num_visualize
        self.test_num_visualize = cfg.testing.num_visualize
        self.vis_count = 0
        self.result_dir = cfg.project.result_dir
        self.batch_size = cfg.training.batch_size

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward(self, obs_features, cord, gt_action=None):
        return self.model(obs_features, cord, gt_action)

    def training_step(self, batch, batch_idx):
        obs_features = batch['obs_features']
        cord = batch['input_positions']

        wp_pred, v_pred, ut = self(obs_features, cord, batch['waypoints'])
        velocity_loss = F.mse_loss(v_pred, ut)

        self.log('train/l_velocity', velocity_loss,
                 on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/loss', velocity_loss,
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return velocity_loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        obs_features = batch['obs_features']
        cord = batch['input_positions']

        wp_pred, v_pred, ut = self(obs_features, cord, batch['waypoints'])
        velocity_loss = F.mse_loss(v_pred, ut)
        self.log('val/velocity_loss', velocity_loss,
                 on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        wp_preds, info = self.model.sample(obs_features, cord, num_samples=200)
        wp_preds = wp_preds * batch['step_scale'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        self.process_visualization(mode='val', batch=batch, wp_pred=wp_preds, noise=info['noise'])
        return velocity_loss

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(self, batch, batch_idx):
        obs_features = batch['obs_features']
        cord = batch['input_positions']

        wp_pred, v_pred, ut = self(obs_features, cord, batch['waypoints'])
        velocity_loss = F.mse_loss(v_pred, ut)
        self.log('test/velocity_loss', velocity_loss,
                 on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        self.process_visualization(mode='test', batch=batch, wp_pred=wp_pred)
        return velocity_loss

    # ------------------------------------------------------------------
    # Epoch hooks
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self):
        self.vis_count = 0

    def on_test_epoch_start(self):
        self.vis_count = 0

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer_name = self.cfg.optimizer.name.lower()
        lr = float(self.cfg.optimizer.lr)

        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), lr=lr,
                weight_decay=self.cfg.optimizer.weight_decay,
            )
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), lr=lr,
                weight_decay=self.cfg.optimizer.weight_decay,
            )
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        scheduler_cfg = self.cfg.scheduler
        if scheduler_cfg.name.lower() == 'step_lr':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=scheduler_cfg.step_size,
                gamma=scheduler_cfg.gamma,
            )
            return [optimizer], [scheduler]
        elif scheduler_cfg.name.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.cfg.training.max_epochs,
            )
            return [optimizer], [scheduler]
        elif scheduler_cfg.name.lower() == 'none':
            return optimizer
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_cfg.name}")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def process_visualization(self, mode, batch, wp_pred, noise: np.ndarray | None = None):
        if mode == 'val':
            num_visualize = self.val_num_visualize
            vis_dir = os.path.join(
                self.result_dir, 'val_vis', f'epoch_{self.current_epoch}'
            )
        elif mode == 'test':
            num_visualize = self.test_num_visualize
            vis_dir = os.path.join(self.result_dir, 'test_vis')
        else:
            raise ValueError("Mode should be 'val' or 'test'.")

        os.makedirs(vis_dir, exist_ok=True)
        batch_size = batch['obs_features'].size(0)

        # Per-sample camera intrinsics: (B, 5) tensor or None
        cam_all = batch.get('camera_intrinsics')

        for idx in range(batch_size):
            if self.vis_count >= num_visualize:
                break

            original_input_positions = batch['original_input_positions'][idx].cpu().numpy()
            noisy_input_positions = batch['noisy_input_positions'][idx].cpu().numpy()
            gt_waypoints = batch['gt_waypoints'][idx].cpu().numpy()
            pred_waypoints = wp_pred[idx].detach().cpu().numpy()                

            # |z|_2
            d = noise.shape[-2] * noise.shape[-1]
            squared_norms = np.sum(noise[idx] ** 2, axis=(-2, -1))            
            # -|z|^2_2 / 2 - d / 2 log (2\pi)
            log_densities = -.5 * (squared_norms + d * np.log(2. * np.pi))

            
            neg_ent = -.5 * d * (1. + np.log(2. * np.pi))  # -H[X] = E[log p(X)]
            # concentration bounds; see Lemma 1 of [Laurent & Massart, 2000]
            DELTA = 1e-2        # confidence level
            t = np.log(2. / DELTA)
            radius = np.sqrt(t * d)
            # [-sqrt(td) - t, sqrt(td)]
            lb = neg_ent - radius
            ub = neg_ent + radius
            # normalization; fall into [0, 1]
            log_densities = (log_densities - lb) / (ub - lb)
            # clipping
            log_densities = np.clip(log_densities, 0., 1.)

            # Check if a raw image is available for this sample
            has_image = (
                'image_path' in batch
                and idx < len(batch['image_path'])
                and batch['image_path'][idx]
                and os.path.exists(batch['image_path'][idx])
            )

            # Check if a video file is available for on-the-fly extraction
            has_video = (
                not has_image
                and 'video_path' in batch
                and idx < len(batch['video_path'])
                and batch['video_path'][idx]
                and batch['video_frame_idx'][idx].item() >= 0
            )

            # Per-sample camera params (sentinel -1 means absent)
            has_camera = (
                cam_all is not None
                and cam_all[idx, 0].item() > 0
            )

            if has_camera:
                fx, fy, cx, cy, dw, dh = cam_all[idx].tolist()
                # Approximate horizontal FOV for coordinate-panel overlay
                fov = 2.0 * np.degrees(np.arctan(cx / fx)) if fx > 0 else None
            else:
                fx = fy = cx = cy = dw = dh = 0.0
                fov = None

            # Obtain raw image from either pre-split frames or video
            img = None
            if has_image:
                img = Image.open(batch['image_path'][idx]).convert('RGB')
            elif has_video:
                img = self._extract_video_frame(
                    batch['video_path'][idx],
                    batch['video_frame_idx'][idx].item(),
                )

            if img is not None and has_camera:
                fig, (ax_img, ax_coord) = plt.subplots(1, 2, figsize=(12, 6))
                plt.subplots_adjust(wspace=0.3)
                draw_image_panel(
                    ax_img, img,
                    gt_waypoints, pred_waypoints,
                    batch['gt_waypoints_y'][idx].cpu().numpy(),
                    fx=fx, fy=fy, cx=cx, cy=cy,
                        desired_width=dw, desired_height=dh, densities=log_densities
                )
            else:
                fig, ax_coord = plt.subplots(figsize=(6, 6))

            # -- Coordinate plot (always shown) --
            draw_coord_panel(
                fig, ax_coord, original_input_positions, noisy_input_positions,
                gt_waypoints, pred_waypoints, fov=fov, densities=log_densities
            )

            plt.savefig(os.path.join(vis_dir, f'sample_{self.vis_count}.png'))
            plt.close(fig)
            self.vis_count += 1

    # -- visualisation helpers --------------------------------------------------

    @staticmethod
    def _extract_video_frame(video_path, frame_idx):
        """Extract a single frame from a video file using decord."""
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        frame_idx = min(frame_idx, len(vr) - 1)
        frame = vr[frame_idx].asnumpy()  # (H, W, 3) uint8
        return Image.fromarray(frame)

    # ------------------------------------------------------------------
    # Utility: load weights from image-based checkpoint
    # ------------------------------------------------------------------

    @classmethod
    def from_image_checkpoint(cls, cfg, checkpoint_path):
        """
        Create a FlowMatchingFeatModule and load decoder weights from an
        existing FlowMatchingModule (image-based) checkpoint.

        The encoder, RGB normalisation buffers, and any other image-only
        parameters are silently skipped.
        """
        module = cls(cfg)
        state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']

        # Filter out image-encoder-only keys
        skip_prefixes = ('model.obs_encoder.', 'model.mean', 'model.std')
        filtered = {
            k: v for k, v in state_dict.items()
            if not any(k.startswith(p) for p in skip_prefixes)
        }

        missing, unexpected = module.load_state_dict(filtered, strict=False)
        if missing:
            print(f"[from_image_checkpoint] Missing keys (expected): {missing}")
        if unexpected:
            print(f"[from_image_checkpoint] Unexpected keys (skipped): {unexpected}")
        return module



def draw_coord_panel(fig, ax, original_input, noisy_input, gt_waypoints,
                        pred_waypoints, *, fov=None, densities=None):
    """Render the 2-D coordinate trajectory plot."""
    if fov is not None:
        th = (np.pi - np.deg2rad(fov)) / 2.0
        r = np.linspace(0.0, 7.0, num=100)
        c, s = np.cos(th), np.sin(th)
        ax.plot(r * c, r * s, linestyle='dashed', color='tab:gray',
                label='fov')
        ax.plot(-r * c, r * s, linestyle='dashed', color='tab:gray')
    # add the origin to the beginning of each trajectory
    gt_waypoints = np.insert(gt_waypoints, 0, values=0., axis=0)
    pred_waypoints = np.insert(pred_waypoints, 0, values=0., axis=-2)
    
    ax.plot(original_input[:, 0], original_input[:, 1],
            'o-', label='Original Input', color='#5771DB', zorder=2)
    ax.plot(noisy_input[:, 0], noisy_input[:, 1],
            'o-', label='Noisy Input', color='#DBC257', zorder=2)
    ax.plot(gt_waypoints[:, 0], gt_waypoints[:, 1],
            'X-', label='GT Waypoints', color='#92DB58', zorder=2)

    if densities is not None:
        cmap = plt.get_cmap('coolwarm')
        colors = cmap(densities)        # RGBA
        # colors[..., -1] = densities     # alpha proportional to density

    line_collection = LineCollection(pred_waypoints, colors=colors, linewidth=2., label='Predicted', zorder=1)
    ax.add_collection(line_collection)

    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap='coolwarm'),
            ax=ax, orientation='vertical', label='normalized log density')

    ax.legend()
    ax.set_title('Coordinates')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.grid(True)
    ax.axis('equal')
    return


def draw_image_panel(
    ax, img,
    gt_waypoints, pred_waypoints, gt_waypoints_y, *,
    fx, fy, cx, cy,
    desired_width, desired_height,
    densities: np.ndarray | None = None
    ):
    """
    Render the raw image with projected waypoint overlays.

    Parameters
    ----------
    img : PIL.Image.Image
        raw RGB frame (from file or video extraction)
    gt_waypoints, pred_waypoints, gt_waypoints_y: np.ndarray
        coordinates expressed in camera frame
    fx, fy, cx, cy: float
        intrinsic parameters
    desired_width, desired_height: float
        input rgb size (DINO)
    densities: np.ndarray | None
        shape: (sample size, 12, 2), where 12: padded length (1d conditional Unet)

    """
    W_orig, H_orig = img.size

    dw = int(desired_width)
    dh = int(desired_height)

    # margins
    # < 0: pad / > 0: crop
    left, right = (W_orig - dw) // 2, (W_orig + dw) // 2
    top, bottom = (H_orig - dh) // 2, (H_orig + dh) // 2
    
    ax.imshow(np.array(img))
    ax.axis('off')

    # camera matrix
    K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])

    if densities is not None:
        cmap = plt.get_cmap('coolwarm')
        colors = cmap(densities)
        # colors[..., -1] = densities     # transparency proportional to density
    else:
        colors = np.array(to_rgba('#DB6057'))

    # inference results
    curves_pred = np.insert(pred_waypoints, 1, gt_waypoints_y, axis=-1)
    curves_pred[..., 1] += 1.8
    line_collection = project_curves_onto_image_plane(
        curves_pred,
        colors,
        camera_matrix=K,
        z_near=1e-3
    )
    ax.add_collection(line_collection)

    # ground truth
    curve_gt = np.insert(gt_waypoints, 1, gt_waypoints_y, axis=-1)
    curve_gt[..., 1] += 1.8
    line_collection = project_curves_onto_image_plane(
        curve_gt,
        colors=np.array(to_rgba('#92DB58')),
        camera_matrix=K,
        z_near=1e-3
    )
    ax.add_collection(line_collection)
    
    # crop/pad
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    return


def project_curves_onto_image_plane(points, colors, camera_matrix, z_near=1e-2):
    """
    Returns a set of edges on the image plane; has shape (*, 2, 2).

    Parameters
    ----------
    points: np.ndarray of shape (*, # of vertices, 3), where the last two dimensions represent a PL curve
            (v[0], v[1], ...,  v[n]), where n: curve length
            Must be represented in the camera frame.
    colors: np.ndarray of shape (*, 4) containing RGBA values
            one color per curve
    """
    
    # vertices -> edges: e_in[i] := v[i] / e_out[i] := v[i+1]
    # (*, # of edges, 3)
    e_in = np.copy(points[..., :-1, :])
    e_out = np.copy(points[..., 1:, :])

    n_edges = e_in.shape[-2]

    points_z = points[..., -1]      # z-coordinates

    in_front = points_z >= z_near
    f_in, f_out = in_front[..., :-1], in_front[..., 1:]     # (*, # of edges)

    fb = f_in & ~f_out      # (front, back)
    bf = ~f_in & f_out      # (back, front)

    visible = f_in | f_out
    visible = visible.reshape((-1,))

    # interpolation (intersection with z = z_near)
    z0, z1 = e_in[bf, -1], e_out[bf, -1]
    t = (z_near - z0) / (z1 - z0)
    t = t[..., np.newaxis]
    e_in[bf] = (1. - t) * e_in[bf] + t * e_out[bf]

    z0, z1 = e_out[fb, -1], e_in[fb, -1]
    t = (z_near - z0) / (z1 - z0)
    t = t[..., np.newaxis]
    e_out[fb] = (1. - t) * e_out[fb] + t * e_in[fb]

    edges_clipped = np.stack((e_in, e_out), axis=-2)    # (*, # of edges, 2, 3)

    edges_clipped = edges_clipped.reshape((-1, 2, 3))
    edges_in_front = edges_clipped[visible]
    
    colors = np.repeat(colors[..., np.newaxis, :], n_edges, axis=-2)      # (*, # of edges, 3)
    colors = colors.reshape((-1, 4))
    colors = colors[visible]
    points_in_front = edges_in_front.reshape((-1, 3))    # flatten

    # arguments: object points, rvec, tvec, camera matrix, distortion coefficients
    # image_points, _ = cv2.projectPoints(points_in_front, np.zeros(3), np.zeros(3), camera_matrix, np.zeros(5))
    points_img = points_in_front @ camera_matrix.T
    x_img, y_img, z_img = np.split(points_img, 3, axis=-1)
    u, v = x_img / z_img, y_img / z_img

    image_points = np.stack((u, v), axis=-1)
    image_edges = image_points.reshape((-1, 2, 2))
    line_collection = LineCollection(image_edges, colors=colors, linewidths=2)


    return line_collection
