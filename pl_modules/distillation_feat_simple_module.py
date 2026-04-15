import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from omegaconf import OmegaConf
from PIL import Image
from pl_modules.flow_matching_feat_simple_module import FlowMatchingFeatSimpleModule
from model.flow_matching_feat_simple_distilled import DistilledFeatSimpleMLP
from vis_utils import project_waypoints_onto_image_plane

matplotlib.use('Agg')


class DistillationFeatSimpleModule(pl.LightningModule):
    """
    Feature-based distillation module that distills a FlowMatchingFeat teacher
    into a ResidualMLP student using precomputed DINOv2 features.

    Mirrors DistillationModule but replaces image inputs with precomputed
    features and loads the teacher from a FlowMatchingFeatModule checkpoint.
    """

    def __init__(self, cfg, teacher_checkpoint_path=None):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

        if teacher_checkpoint_path is not None:
            # Load teacher from FlowMatchingFeatModule checkpoint
            fm_module = FlowMatchingFeatSimpleModule.load_from_checkpoint(
                teacher_checkpoint_path, cfg=cfg
            )
            teacher_model = fm_module.model
        else:
            # Create teacher architecture from cfg; weights will be loaded
            # from the distillation checkpoint's state_dict.
            from model.flow_matching_feat_simple import FlowMatchingFeatSimple
            teacher_model = FlowMatchingFeatSimple(cfg)
        self.model = DistilledFeatSimpleMLP(teacher_model)

        self.val_num_visualize = cfg.validation.num_visualize
        self.test_num_visualize = cfg.testing.num_visualize
        self.vis_count = 0
        self.result_dir = cfg.project.result_dir
        self.batch_size = cfg.training.batch_size
        self.num_inference_steps = 100

    def training_step(self, batch, batch_idx):
        obs_features = batch['obs_features']
        cord = batch['input_positions']

        out = self.model(obs_features, cord)
        student_deltas = out["deltas_pred"]
        noise = out["noise"]
        dec_out = out["dec_out"]
        B = student_deltas.size(0)

        num_steps = self.num_inference_steps

        with torch.no_grad():
            xt = noise.clone()
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t_curr = i / num_steps
                t_tensor = torch.full((B,), t_curr, device=xt.device)
                v_pred = self.model.teacher.wp_predictor(
                    sample=xt, timestep=t_tensor, global_cond=dec_out
                )
                xt = xt + v_pred * dt
            teacher_targets = xt

        distill_loss = F.mse_loss(student_deltas, teacher_targets)
        self.log('train/distill_loss', distill_loss, prog_bar=True, sync_dist=True)
        return distill_loss

    def validation_step(self, batch, batch_idx):
        obs_features = batch['obs_features']
        cord = batch['input_positions']
        num_steps = self.num_inference_steps
        num_samples = 200

        out = self.model.sample(obs_features, cord, num_samples=num_samples)
        student_deltas = out["deltas_pred"]
        noise = out["noise"]
        dec_out = out["dec_out"]
        B = student_deltas.size(0)

        with torch.no_grad():
            xt = noise.clone()
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t_curr = i / num_steps
                t_tensor = torch.full((B,), t_curr, device=xt.device)
                v_pred = self.model.teacher.wp_predictor(
                    sample=xt, timestep=t_tensor, global_cond=dec_out
                )
                xt = xt + v_pred * dt
            teacher_targets = xt

        distill_loss = F.mse_loss(student_deltas, teacher_targets)

        wp_pred_teacher = torch.cumsum(teacher_targets, dim=1)
        wp_pred_teacher = wp_pred_teacher.view(
            -1, num_samples, self.model.len_traj_pred, 2
        )

        self.log('val/distill_loss', distill_loss, prog_bar=True, sync_dist=True)
        self.process_visualization(
            mode='val', batch=batch,
            wp_pred_teacher=wp_pred_teacher,
            wp_pred_student=out["wp_pred"],
        )
        return distill_loss

    def on_validation_epoch_start(self):
        self.vis_count = 0

    def on_test_epoch_start(self):
        self.vis_count = 0

    def configure_optimizers(self):
        optimizer_name = self.cfg.optimizer.name.lower()
        lr = float(self.cfg.optimizer.lr)

        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                self.model.wp_predictor.parameters(), lr=lr,
                weight_decay=self.cfg.optimizer.weight_decay,
            )
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.wp_predictor.parameters(), lr=lr,
                weight_decay=self.cfg.optimizer.weight_decay,
            )
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.wp_predictor.parameters(), lr=lr,
            )
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
    # Visualization (shows both teacher and student predictions)
    # ------------------------------------------------------------------

    def process_visualization(self, mode, batch, wp_pred_teacher, wp_pred_student):
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
        cam_all = batch.get('camera_intrinsics')

        for idx in range(batch_size):
            if self.vis_count >= num_visualize:
                break

            original_input_positions = batch['original_input_positions'][idx].cpu().numpy()
            noisy_input_positions = batch['noisy_input_positions'][idx].cpu().numpy()
            pred_teacher = wp_pred_teacher[idx].detach().cpu().numpy()
            pred_student = wp_pred_student[idx].detach().cpu().numpy()
            gt_waypoints_y = batch.get('gt_waypoints_y')

            has_camera = cam_all is not None and cam_all[idx, 0].item() > 0

            has_image = (
                'image_path' in batch
                and idx < len(batch['image_path'])
                and batch['image_path'][idx]
                and os.path.exists(batch['image_path'][idx])
            )
            has_video = (
                not has_image
                and 'video_path' in batch
                and idx < len(batch['video_path'])
                and batch['video_path'][idx]
                and batch['video_frame_idx'][idx].item() >= 0
            )

            if has_camera:
                fx, fy, cx, cy, dw, dh = cam_all[idx].tolist()
                fov = 2.0 * np.degrees(np.arctan(cx / fx)) if fx > 0 else None
            else:
                fov = None

            img = None
            if has_image:
                img = Image.open(batch['image_path'][idx]).convert('RGB')
            elif has_video:
                img = self._extract_video_frame(
                    batch['video_path'][idx],
                    batch['video_frame_idx'][idx].item(),
                )

            if img is not None and has_camera and gt_waypoints_y is not None:
                fig, (ax_img, ax_coord) = plt.subplots(1, 2, figsize=(12, 6))
                plt.subplots_adjust(wspace=0.3)
                self._draw_image_panel(
                    ax_img, img,
                    pred_teacher, pred_student,
                    gt_waypoints_y[idx].cpu().numpy(),
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    desired_width=dw, desired_height=dh,
                )
            else:
                fig, ax_coord = plt.subplots(figsize=(6, 6))

            self._draw_coord_panel(
                ax_coord, original_input_positions, noisy_input_positions,
                pred_teacher, pred_student, fov=fov,
            )

            plt.savefig(os.path.join(vis_dir, f'sample_{self.vis_count}.png'))
            plt.close(fig)
            self.vis_count += 1

    # -- visualisation helpers -----------------------------------------

    @staticmethod
    def _extract_video_frame(video_path, frame_idx):
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        frame_idx = min(frame_idx, len(vr) - 1)
        frame = vr[frame_idx].asnumpy()
        return Image.fromarray(frame)

    def _draw_image_panel(self, ax, img, pred_teacher, pred_student,
                          gt_waypoints_y, *, fx, fy, cx, cy,
                          desired_width, desired_height):
        W_orig, H_orig = img.size
        dw = int(desired_width)
        dh = int(desired_height)
        left = max(0, (W_orig - dw) // 2)
        top = max(0, (H_orig - dh) // 2)

        if W_orig != dw or H_orig != dh:
            img = img.crop((left, top, left + dw, top + dh))

        ax.imshow(np.array(img))
        ax.axis('off')
        K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])

        def shift(u, v):
            return u - left, v - top

        # Teacher predictions (red)
        if pred_teacher.ndim == 3:
            for s in range(pred_teacher.shape[0]):
                u_p, v_p, valid = project_waypoints_onto_image_plane(
                    pred_teacher[s], gt_waypoints_y, K=K)
                u_p, v_p = shift(u_p, v_p)
                if np.all(valid):
                    ax.plot(u_p, v_p, color='#DB6057', alpha=0.5)
        elif pred_teacher.ndim == 2:
            u_p, v_p, valid = project_waypoints_onto_image_plane(
                pred_teacher, gt_waypoints_y, K=K)
            u_p, v_p = shift(u_p, v_p)
            if np.all(valid):
                ax.plot(u_p, v_p, color='#DB6057', alpha=0.5)

        # Student predictions (blue)
        if pred_student.ndim == 3:
            for s in range(pred_student.shape[0]):
                u_p, v_p, valid = project_waypoints_onto_image_plane(
                    pred_student[s], gt_waypoints_y, K=K)
                u_p, v_p = shift(u_p, v_p)
                if np.all(valid):
                    ax.plot(u_p, v_p, color='#5795DB', alpha=0.8)
        elif pred_student.ndim == 2:
            u_p, v_p, valid = project_waypoints_onto_image_plane(
                pred_student, gt_waypoints_y, K=K)
            u_p, v_p = shift(u_p, v_p)
            if np.all(valid):
                ax.plot(u_p, v_p, color='#5795DB')

        ax.set_xlim(0.0, dw)
        ax.set_ylim(dh, 0.0)

    def _draw_coord_panel(self, ax, original_input, noisy_input,
                          pred_teacher, pred_student, *, fov=None):
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

        if pred_teacher.ndim == 3:
            labeled = False
            for s in range(pred_teacher.shape[0]):
                label = 'Flow Matching' if not labeled else None
                ax.plot(pred_teacher[s, :, 0], pred_teacher[s, :, 1],
                        color='#DB6057', alpha=0.5, label=label)
                labeled = True
        elif pred_teacher.ndim == 2:
            ax.plot(pred_teacher[:, 0], pred_teacher[:, 1],
                    label='Flow Matching', color='#DB6057', alpha=0.5)

        if pred_student.ndim == 3:
            labeled = False
            for s in range(pred_student.shape[0]):
                label = 'Distilled' if not labeled else None
                ax.plot(pred_student[s, :, 0], pred_student[s, :, 1],
                        color='#5795DB', alpha=0.5, label=label)
                labeled = True
        elif pred_student.ndim == 2:
            ax.plot(pred_student[:, 0], pred_student[:, 1],
                    label='Distilled', color='#5795DB', alpha=0.5)

        ax.legend()
        ax.set_title('Coordinates')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.grid(True)
