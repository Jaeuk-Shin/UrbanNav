import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pl_modules.flow_matching_module import FlowMatchingModule
from model.flow_matching_distilled import DistilledMLP
from vis_utils import build_intrinsic_matrix, project_waypoints_onto_image_plane

matplotlib.use('Agg')



class DistillationModule(pl.LightningModule):
    def __init__(self, cfg, teacher_checkpoint_path):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.do_normalize = cfg.training.normalize_step_length
        # Load teacher from checkpoint
        # No longer needed: DictNamespace serialization workaround removed
        fm_module = FlowMatchingModule.load_from_checkpoint(teacher_checkpoint_path, cfg=cfg)
        self.model = DistilledMLP(fm_module.model)
        
        # Visualization settings
        self.val_num_visualize = cfg.validation.num_visualize
        self.test_num_visualize = cfg.testing.num_visualize
        self.vis_count = 0
        
        self.result_dir = cfg.project.result_dir
        self.batch_size = cfg.training.batch_size
        self.image_mean = np.array([0.485, 0.456, 0.406])
        self.image_std = np.array([0.229, 0.224, 0.225])
        

        self.num_inference_steps = 100

    def training_step(self, batch, batch_idx):
        obs = batch['video_frames']
        cord = batch['input_positions']
        
        # 1. Forward pass (Student MLP)
        # returns dict: {"wp_pred", "deltas_pred", "noise", "context"}
        out = self.model(obs, cord)
        
        student_deltas = out["deltas_pred"]          # (B, N, 2)
        
        noise = out["noise"]                         # (B, N, 2)
        dec_out = out["dec_out"]
        B = student_deltas.size(0)
        
        num_steps = self.num_inference_steps

        # 1. Teacher ODE Integration (The "Target" Generator)
        # We do not track gradients for the teacher to save memory
        with torch.no_grad():
            # Flow Matching integration starts at x0 (noise) and ends at x1 (data)
            # We pad the noise if the Teacher UNet expects a fixed sequence length (e.g., 12)
            xt = F.pad(noise, (0, 0, 0, 12 - noise.size(1))) 
            
            dt = 1.0 / num_steps
            for i in range(num_steps):
                # i ranges from 0 to num_steps-1
                t_curr = i / num_steps
                # Scale time to [0, 1000] for the Teacher's Sinusoidal Embedding
                t_tensor = torch.full((B,), t_curr * 1000, device=xt.device)
                # Teacher predicts velocity v = dx/dt
                v_pred = self.model.teacher.wp_predictor(
                    sample=xt, 
                    timestep=t_tensor, 
                    global_cond=dec_out
                )
                
                # Euler integration step
                xt = xt + v_pred * dt
            
            # Extract the relevant trajectory length and set as target
            teacher_targets = xt[:, :self.model.len_traj_pred]
        distill_loss = F.mse_loss(student_deltas, teacher_targets)
    
        self.log('train/distill_loss', distill_loss, prog_bar=True, sync_dist=True)
        return distill_loss

    def validation_step(self, batch, batch_idx):
        
        num_steps = self.num_inference_steps

        num_samples = 200
        # Sample 10 for visualization, use 1 for metric calculation
        out = self.model.sample(batch['video_frames'], batch['input_positions'], num_samples=num_samples)

        student_deltas = out["deltas_pred"]          # (B*S, N, 2)
        
        noise = out["noise"]                         # (B*S, N, 2)
        dec_out = out["dec_out"]
        B = student_deltas.size(0)
        
        # 1. Teacher ODE Integration (The "Target" Generator)
        # We do not track gradients for the teacher to save memory
        with torch.no_grad():
            # Flow Matching integration starts at x0 (noise) and ends at x1 (data)
            # We pad the noise if the Teacher UNet expects a fixed sequence length (e.g., 12)
            xt = F.pad(noise, (0, 0, 0, 12 - noise.size(1))) 
            
            dt = 1.0 / num_steps
            for i in range(num_steps):
                # i ranges from 0 to num_steps-1
                t_curr = i / num_steps
                # Scale time to [0, 1000] for the Teacher's Sinusoidal Embedding
                t_tensor = torch.full((B,), t_curr * 1000, device=xt.device)
                # Teacher predicts velocity v = dx/dt
                v_pred = self.model.teacher.wp_predictor(
                    sample=xt, 
                    timestep=t_tensor, 
                    global_cond=dec_out
                )
                
                # Euler integration step
                xt = xt + v_pred * dt
            
            # Extract the relevant trajectory length and set as target
            teacher_targets = xt[:, :self.model.len_traj_pred]
        distill_loss = F.mse_loss(student_deltas, teacher_targets)

        wp_pred_teacher = torch.cumsum(teacher_targets, dim=1)
        
        # Reshape to (batch size, # of samples, prediction length, 2)
        wp_pred_teacher = wp_pred_teacher.view(-1, num_samples, self.model.len_traj_pred, 2)
        self.log('val/distill_loss', distill_loss, prog_bar=True, sync_dist=True)
        # visualization
        self.process_visualization(mode='val', batch=batch, obs=batch['video_frames'], wp_pred_teacher=wp_pred_teacher, wp_pred_student=out["wp_pred"])

        return distill_loss

    def on_validation_epoch_start(self):
        self.vis_count = 0

    def configure_optimizers(self):
        optimizer_name = self.cfg.optimizer.name.lower()
        lr = float(self.cfg.optimizer.lr)

        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.model.wp_predictor.parameters(), lr=lr, weight_decay=self.cfg.optimizer.weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.model.wp_predictor.parameters(), lr=lr, weight_decay=self.cfg.optimizer.weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(self.model.wp_predictor.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Scheduler
        scheduler_cfg = self.cfg.scheduler
        if scheduler_cfg.name.lower() == 'step_lr':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_cfg.step_size, gamma=scheduler_cfg.gamma)
            return [optimizer], [scheduler]
        elif scheduler_cfg.name.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.training.max_epochs)
            return [optimizer], [scheduler]
        elif scheduler_cfg.name.lower() == 'none':
            return optimizer
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_cfg.name}")
        
    
    def process_visualization(self, mode, batch, obs, wp_pred_teacher, wp_pred_student):
        """
        Handles visualization for both validation and testing.

        Args:
            mode (str): 'val' or 'test'
            batch (dict): Batch data
            obs (torch.Tensor): Observation frames;  (batch size, history length, 3, h, w)
            wp_pred (torch.Tensor): Predicted waypoints
            arrive_pred (torch.Tensor): Predicted arrival logits
        """
        if mode == 'val':
            num_visualize = self.val_num_visualize
            vis_dir = os.path.join(self.result_dir, 'val_vis', f'epoch_{self.current_epoch}')
        elif mode == 'test':
            num_visualize = self.test_num_visualize
            vis_dir = os.path.join(self.result_dir, 'test_vis')
        else:
            raise ValueError("Mode should be either 'val' or 'test'.")

        os.makedirs(vis_dir, exist_ok=True)

        batch_size = obs.size(0)    
        for batch_idx in range(batch_size):
            if self.vis_count >= num_visualize:
                break

            original_input_positions = batch['original_input_positions'][batch_idx].cpu().numpy()
            noisy_input_positions = batch['noisy_input_positions'][batch_idx].cpu().numpy()
            
            pred_waypoints_teacher = wp_pred_teacher[batch_idx].detach().cpu().numpy()
            pred_waypoints_student = wp_pred_student[batch_idx].detach().cpu().numpy()
            # gt_waypoints = batch['gt_waypoints'][idx].cpu().numpy()
            gt_waypoints_y = batch['gt_waypoints_y'][batch_idx].cpu().numpy()     # for visualization
            # Get the last frame from the sequence
            frame = obs[batch_idx, -1].permute(1, 2, 0).cpu().numpy()     # (h, w, 3); most recent frame
            frame = (frame * 255).astype(np.uint8)  # Convert to uint8 for visualization
            H, W, _ = frame.shape

            # Plotting
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            plt.subplots_adjust(wspace=0.3)

            # Left axis: plot the current observation (frame) with arrived info in title
            ax1.imshow(frame)
            ax1.axis('off')
            # ax1.set_title(arrive_title, fontsize=20)

            # overlay on image
            # intrinsic matrix
            # TODO: will be defined as a separate method
            width = self.cfg.data.width
            height = self.cfg.data.height
            fov = self.cfg.data.fov

            

            desired_width = self.cfg.data.desired_width
            desired_height = self.cfg.data.desired_height

            assert desired_width == W and desired_height == H


            x_offset = .5 * (width - desired_width)
            y_offset = .5 * (height - desired_height)

            K = build_intrinsic_matrix(w=width, h=height, fov=fov)

            def shift(x_img, y_img):
                """
                shift image coordinates to account for center crop
                images are cropped/padded within the dataloader; must be considered during coordinate computation
                e.g., 640 x 480 -> 640 x 360
                """
                return x_img - x_offset, y_img - y_offset


            # visualize teacher's outputs
            if pred_waypoints_teacher.ndim == 3:
                # multiple samples drawn from the model
                for sample_idx in range(pred_waypoints_teacher.shape[0]):
                    # Since the model does not output y, just augment the outputs with ground truth y
                    u_pred, v_pred, valid = project_waypoints_onto_image_plane(pred_waypoints_teacher[sample_idx], gt_waypoints_y, K=K)
                    u_pred, v_pred = shift(u_pred, v_pred)
                    if np.all(valid):
                        ax1.plot(u_pred[valid], v_pred[valid], color='#DB6057', alpha=0.5)
            elif pred_waypoints_teacher.ndim == 2:
                u_pred, v_pred, valid = project_waypoints_onto_image_plane(pred_waypoints_teacher, gt_waypoints_y, K=K)
                u_pred, v_pred = shift(u_pred, v_pred)
                if np.all(valid):
                    ax1.plot(u_pred[valid], v_pred[valid], color='#DB6057', alpha=0.5)
            else:
                raise ValueError
            
            # student's outputs
            if pred_waypoints_student.ndim == 3:
                # multiple samples drawn from the model
                for sample_idx in range(pred_waypoints_student.shape[0]):
                    u_pred, v_pred, valid = project_waypoints_onto_image_plane(pred_waypoints_student[sample_idx], gt_waypoints_y, K=K)
                    u_pred, v_pred = shift(u_pred, v_pred)
                    if np.all(valid):
                        ax1.plot(u_pred[valid], v_pred[valid], color="#5795DB")
            elif pred_waypoints_student.ndim == 2:
                u_pred, v_pred, valid = project_waypoints_onto_image_plane(pred_waypoints_student, gt_waypoints_y, K=K)
                u_pred, v_pred = shift(u_pred, v_pred)
                if np.all(valid):
                    ax1.plot(u_pred[valid], v_pred[valid], color="#5795DB")
            else:
                raise ValueError
            ax1.set_xlim(0., W)
            ax1.set_ylim(H, 0.)

            # Right axis: plot the coordinates

            # fov visualization
            th = np.pi / 2. - np.deg2rad(fov) / 2.
            r = np.linspace(0., 7., num=100)
            c, s = np.cos(th), np.sin(th)
            ax2.plot(r * c, r * s, linestyle='dashed', color='tab:gray', label='fov')
            ax2.plot(-r * c, r * s, linestyle='dashed', color='tab:gray')
            
            ax2.axis('equal')   
            ax2.plot(original_input_positions[:, 0], original_input_positions[:, 1],
                     'o-', label='Original Input Positions', color='#5771DB')
            ax2.plot(noisy_input_positions[:, 0], noisy_input_positions[:, 1],
                     'o-', label='Noisy Input Positions', color='#DBC257')
            
            '''
            ax2.plot(gt_waypoints[:, 0], gt_waypoints[:, 1],
                     'X-', label='GT Waypoints', color='#92DB58')
            '''
            if pred_waypoints_teacher.ndim == 3:
                # multiple samples drawn from the model
                labeled = False
                for sample_idx in range(pred_waypoints_teacher.shape[0]):
                    if not labeled:
                        label = 'Flow Matching'
                        labeled = True
                    else:
                        label = None
                    ax2.plot(pred_waypoints_teacher[sample_idx, :, 0], pred_waypoints_teacher[sample_idx, :, 1],
                        label=label, color='#DB6057', alpha=0.5)
            elif pred_waypoints_teacher.ndim == 2:
                ax2.plot(pred_waypoints_teacher[:, 0], pred_waypoints_teacher[:, 1],
                        label='Flow Matching', color='#DB6057', alpha=0.5)
            else:
                raise ValueError
            
            if pred_waypoints_student.ndim == 3:
                # multiple samples drawn from the model
                labeled = False
                for sample_idx in range(pred_waypoints_student.shape[0]):
                    if not labeled:
                        label = 'Distilled'
                        labeled = True
                    else:
                        label = None
                    ax2.plot(pred_waypoints_student[sample_idx, :, 0], pred_waypoints_student[sample_idx, :, 1],
                        label=label, color="#5795DB", alpha=0.5)
            elif pred_waypoints_student.ndim == 2:
                ax2.plot(pred_waypoints_student[:, 0], pred_waypoints_student[:, 1],
                        label='Distilled', color="#5795DB", alpha=0.5)
            else:
                raise ValueError

            # ax2.plot(target_transformed[0], target_transformed[1],
            #          marker='*', markersize=15, label='Target Coordinate', color='#A157DB')
            ax2.legend()
            ax2.set_title('Coordinates', fontsize=20)
            ax2.set_xlabel('X (m)', fontsize=20)
            ax2.set_ylabel('Z (m)', fontsize=20)
            ax2.tick_params(axis='both', labelsize=18)
            ax2.grid(True)

            # Save the plot
            output_path = os.path.join(vis_dir, f'sample_{self.vis_count}.png')
            plt.savefig(output_path)
            plt.close(fig)

            self.vis_count += 1