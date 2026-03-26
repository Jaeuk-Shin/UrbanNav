import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from model.flow_matching_feat import FlowMatchingFeat
from vis_utils import project_waypoints_onto_image_plane
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os


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

        wp_preds = self.model.sample(obs_features, cord, num_samples=200)
        wp_preds = wp_preds * batch['step_scale'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        self.process_visualization(mode='val', batch=batch, wp_pred=wp_preds)
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

    def process_visualization(self, mode, batch, wp_pred):
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

            # Check if a raw image is available for this sample
            has_image = (
                'image_path' in batch
                and idx < len(batch['image_path'])
                and batch['image_path'][idx]
                and os.path.exists(batch['image_path'][idx])
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

            if has_image and has_camera:
                fig, (ax_img, ax_coord) = plt.subplots(1, 2, figsize=(12, 6))
                plt.subplots_adjust(wspace=0.3)
                self._draw_image_panel(
                    ax_img, batch['image_path'][idx],
                    gt_waypoints, pred_waypoints,
                    batch['gt_waypoints_y'][idx].cpu().numpy(),
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    desired_width=dw, desired_height=dh,
                )
            else:
                fig, ax_coord = plt.subplots(figsize=(6, 6))

            # -- Coordinate plot (always shown) --
            self._draw_coord_panel(
                ax_coord, original_input_positions, noisy_input_positions,
                gt_waypoints, pred_waypoints, fov=fov,
            )

            plt.savefig(os.path.join(vis_dir, f'sample_{self.vis_count}.png'))
            plt.close(fig)
            self.vis_count += 1

    # -- visualisation helpers --------------------------------------------------

    def _draw_image_panel(self, ax, image_path, gt_waypoints, pred_waypoints,
                          gt_waypoints_y, *, fx, fy, cx, cy,
                          desired_width, desired_height):
        """Render the raw image with projected waypoint overlays."""
        img = Image.open(image_path).convert('RGB')
        W_orig, H_orig = img.size

        dw = int(desired_width)
        dh = int(desired_height)

        # Center-crop offset (used to shift projected coords into crop space)
        left = max(0, (W_orig - dw) // 2)
        top = max(0, (H_orig - dh) // 2)

        if W_orig != dw or H_orig != dh:
            img = img.crop((left, top, left + dw, top + dh))

        ax.imshow(np.array(img))
        ax.axis('off')

        # Camera intrinsics
        K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])

        def shift(u, v):
            return u - left, v - top

        # Ground-truth waypoints
        u_gt, v_gt, valid = project_waypoints_onto_image_plane(
            gt_waypoints, gt_waypoints_y, K=K)
        u_gt, v_gt = shift(u_gt, v_gt)
        if np.all(valid):
            ax.plot(u_gt, v_gt, color='#92DB58', linewidth=3)

        # Predicted waypoints
        if pred_waypoints.ndim == 3:
            for s in range(pred_waypoints.shape[0]):
                u_p, v_p, valid = project_waypoints_onto_image_plane(
                    pred_waypoints[s], gt_waypoints_y, K=K)
                u_p, v_p = shift(u_p, v_p)
                if np.all(valid):
                    ax.plot(u_p, v_p, color='#DB6057', alpha=0.8)
        elif pred_waypoints.ndim == 2:
            u_p, v_p, valid = project_waypoints_onto_image_plane(
                pred_waypoints, gt_waypoints_y, K=K)
            u_p, v_p = shift(u_p, v_p)
            if np.all(valid):
                ax.plot(u_p, v_p, color='#DB6057')

        ax.set_xlim(0.0, dw)
        ax.set_ylim(dh, 0.0)

    def _draw_coord_panel(self, ax, original_input, noisy_input, gt_waypoints,
                          pred_waypoints, *, fov=None):
        """Render the 2-D coordinate trajectory plot."""
        if fov is not None:
            th = np.pi / 2.0 - np.deg2rad(fov) / 2.0
            r = np.linspace(0.0, 7.0, num=100)
            c, s = np.cos(th), np.sin(th)
            ax.plot(r * c, r * s, linestyle='dashed', color='tab:gray',
                    label='fov')
            ax.plot(-r * c, r * s, linestyle='dashed', color='tab:gray')

        ax.axis('equal')
        ax.plot(original_input[:, 0], original_input[:, 1],
                'o-', label='Original Input', color='#5771DB')
        ax.plot(noisy_input[:, 0], noisy_input[:, 1],
                'o-', label='Noisy Input', color='#DBC257')
        ax.plot(gt_waypoints[:, 0], gt_waypoints[:, 1],
                'X-', label='GT Waypoints', color='#92DB58')

        if pred_waypoints.ndim == 3:
            labeled = False
            for s in range(pred_waypoints.shape[0]):
                label = 'Predicted' if not labeled else None
                ax.plot(pred_waypoints[s, :, 0], pred_waypoints[s, :, 1],
                        color='#DB6057', alpha=0.8, label=label)
                labeled = True
        elif pred_waypoints.ndim == 2:
            ax.plot(pred_waypoints[:, 0], pred_waypoints[:, 1],
                    's-', label='Predicted', color='#DB6057')

        ax.legend()
        ax.set_title('Coordinates')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.grid(True)

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
