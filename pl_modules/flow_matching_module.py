import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from model.flow_matching import FlowMatchingTrajectorySampler
from vis_utils import build_intrinsic_matrix, get_image_coordinates, project_waypoints_onto_image_plane
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os

class FlowMatchingModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = FlowMatchingTrajectorySampler(cfg)
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.do_normalize = cfg.training.normalize_step_length
        self.datatype = cfg.data.type
        
        # Coordinate representation
        self.output_coordinate_repr = cfg.model.output_coordinate_repr
        if self.output_coordinate_repr not in ["euclidean", "polar"]:
            raise ValueError(f"Unsupported coordinate representation: {self.output_coordinate_repr}")
        
        self.decoder = cfg.model.decoder.type
        if self.decoder not in ["flow_matching", "diff_policy", "attention"]:
            raise ValueError(f"Unsupported decoder: {self.decoder}")
        
        # Direction loss weight (you can adjust this value in your cfg)
        self.direction_loss_weight = cfg.training.direction_loss_weight
        
        # Visualization settings
        self.val_num_visualize = cfg.validation.num_visualize
        self.test_num_visualize = cfg.testing.num_visualize
        self.vis_count = 0
        
        self.result_dir = cfg.project.result_dir
        self.batch_size = cfg.training.batch_size
        self.image_mean = np.array([0.485, 0.456, 0.406])
        self.image_std = np.array([0.229, 0.224, 0.225])
        
        # If polar, define additional loss weights
        if self.output_coordinate_repr == "polar":
            self.distance_loss_weight = cfg.training.distance_loss_weight
            self.angle_loss_weight = cfg.training.angle_loss_weight

        if self.datatype == "urbannav":
            self.test_catetories = ['crowd', 'person_close_by', 'turn', 'action_target_mismatch', 'crossing', 'other']
            self.num_categories = len(self.test_catetories)

    def forward(self, obs, cord, gt_action=None):
        return self.model(obs, cord, gt_action)
    
    def training_step(self, batch, batch_idx):
        obs = batch['video_frames']
        cord = batch['input_positions']
        
        assert self.decoder == "flow_matching"
        wp_pred, v_pred, ut = self(obs, cord, batch['waypoints'])
        losses = self.compute_loss_flow_matching(wp_pred, v_pred, ut, batch)
        velocity_loss = losses['velocity_loss']
        # arrived_loss = losses['arrived_loss']
        # direction_loss = losses['direction_loss']
        # total_loss = noise_loss + self.direction_loss_weight * direction_loss
        total_loss = velocity_loss
        self.log('train/l_velocity', velocity_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
    
        # Common logs
        # self.log('train/l_arvd', arrived_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        # self.log('train/l_dir', direction_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return total_loss
    
    def compute_loss_flow_matching(self, wp_pred, v_pred, ut, batch):
        """
        Parameters
        ----------    
        v_pred: Predicted velocity from the model (B, T, 2)
        ut: Target velocity (x1 - x0) (B, T, 2)
        """
        velocity_loss = F.mse_loss(v_pred, ut)
        return {'velocity_loss': velocity_loss}


    def validation_step(self, batch, batch_idx):
        obs = batch['video_frames']
        cord = batch['input_positions']
        
        assert self.decoder == "flow_matching"
        wp_pred, v_pred, ut = self(obs, cord, batch['waypoints'])
        losses = self.compute_loss_flow_matching(wp_pred, v_pred, ut, batch)
        noise_loss = losses['velocity_loss']
        # direction_loss = losses['direction_loss']
        self.log('val/velocity_loss', noise_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        # Log the metrics
        # self.log('val/direction_loss', direction_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # Handle visualization
        # wp_pred_vis = wp_pred * batch['step_scale'].unsqueeze(-1).unsqueeze(-1)

        wp_preds = self.model.sample(obs, cord, num_samples=200) * batch['step_scale'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        self.process_visualization(
            mode='val',
            batch=batch,
            obs=obs,
            wp_pred=wp_preds
        )
        
        return noise_loss

    def test_step(self, batch, batch_idx):
        obs = batch['video_frames']
        cord = batch['input_positions']
        B, T, _ = batch['waypoints'].shape
        
        if self.datatype == "citywalk":
            if self.output_coordinate_repr == "euclidean":
                wp_pred = self(obs, cord)
                # Compute L1 loss for waypoints
                waypoints_target = batch['waypoints']
                l1_loss = F.l1_loss(wp_pred, waypoints_target, reduction='mean').item()
            '''
            # Compute accuracy for "arrived" prediction
            arrived_target = batch['arrived']
            arrived_logits = arrive_pred.flatten()
            arrived_probs = torch.sigmoid(arrived_logits)
            arrived_pred_binary = (arrived_probs >= 0.5).float()
            correct = (arrived_pred_binary == arrived_target).float()
            accuracy = correct.sum().item() / correct.numel()
            '''
            # wp_pred_last = wp_pred[:, -1, :]  # shape [batch_size, 2]
            # waypoints_target_last = waypoints_target[:, -1, :]  # shape [batch_size, 2]

            # Compute cosine similarity
            wp_pred_view = wp_pred.view(-1, 2)
            waypoints_target_view = waypoints_target.view(-1, 2)
            # dot_product = (wp_pred_view * waypoints_target_view).sum(dim=1)  # shape [batch_size]
            # norm_pred = wp_pred_view.norm(dim=1)  # shape [batch_size]
            # norm_target = waypoints_target_view.norm(dim=1)  # shape [batch_size]
            # cos_sim = dot_product / (norm_pred * norm_target + 1e-8)  # avoid division by zero
            cos_sim = F.cosine_similarity(wp_pred_view, waypoints_target_view, dim=1)
            
            # Compute angle in degrees
            angle = torch.acos(cos_sim) * 180 / torch.pi  # shape [batch_size]
            angle = angle.view(B, T)
            
            # Take mean angle
            mean_angle = angle.mean(dim=0).cpu().numpy()
            
            # Store the metrics
            if self.output_coordinate_repr == "euclidean":
                self.test_metrics['l1_loss'].append(l1_loss)
            # self.test_metrics['arrived_accuracy'].append(accuracy)
            self.test_metrics['mean_angle'].append(mean_angle)
        elif self.datatype == "urbannav":
            category = batch['categories']
            wp_pred = self(obs, cord)
            wp_pred *= batch['step_scale'].unsqueeze(-1).unsqueeze(-1)
            
            # Compute L1 loss for waypoints
            waypoints_target = batch['waypoints']
            waypoints_target *= batch['step_scale'].unsqueeze(-1).unsqueeze(-1)
            # l1_loss = F.l1_loss(wp_pred, waypoints_target, reduction='none')
            # l1_loss = F.mse_loss(wp_pred, waypoints_target, reduction='none') ** 0.5
            l1_loss = (wp_pred - waypoints_target).norm(dim=-1)
            
            # Compute accuracy for "arrived" prediction
            '''
            arrived_target = batch['arrived']
            arrived_probs = torch.sigmoid(arrive_pred)
            arrived_pred_binary = (arrived_probs >= 0.5).float().squeeze(-1)
            correct = (arrived_pred_binary == arrived_target).float()
            '''
            # Compute cosine similarity
            wp_pred_view = wp_pred.view(-1, 2)
            waypoints_target_view = waypoints_target.view(-1, 2)
            # dot_product = (wp_pred_view * waypoints_target_view).sum(dim=1)  # shape [batch_size]
            # norm_pred = wp_pred_view.norm(dim=1)  # shape [batch_size]
            # norm_target = waypoints_target_view.norm(dim=1)  # shape [batch_size]
            # cos_sim = dot_product / (norm_pred * norm_target + 1e-8)  # avoid division by zero
            cos_sim = F.cosine_similarity(wp_pred_view, waypoints_target_view, dim=1)
            # Compute angle in degrees
            angle = torch.acos(cos_sim) * 180 / torch.pi  # shape [batch_size]
            angle = angle.view(B, T)

            gt_wp_last_norm = waypoints_target[:, -1, :].norm(dim=1)

            for batch_idx in range(B):
                for category_idx in range(self.num_categories):
                    if category[batch_idx, category_idx] == 1:
                        category_name = self.test_catetories[category_idx]
                        self.test_metrics[category_name]['l1_loss'].append(l1_loss[batch_idx].max().item())
                        # self.test_metrics[category_name]['arrived_accuracy'].append(correct[batch_idx].item())
                        if gt_wp_last_norm[batch_idx] > 1:
                            self.test_metrics[category_name]['mean_angle'].append(angle[batch_idx].max().item())
                            self.test_metrics[category_name]['angle_step1'].append(angle[batch_idx, 0].item())
                            self.test_metrics[category_name]['angle_step2'].append(angle[batch_idx, 1].item())
                            self.test_metrics[category_name]['angle_step3'].append(angle[batch_idx, 2].item())
                            self.test_metrics[category_name]['angle_step4'].append(angle[batch_idx, 3].item())
                            self.test_metrics[category_name]['angle_step5'].append(angle[batch_idx, 4].item())
                    else:
                        continue
                self.test_metrics['overall']['l1_loss'].append(l1_loss[batch_idx].max().item())
                # self.test_metrics['overall']['arrived_accuracy'].append(correct[batch_idx].item())
                if gt_wp_last_norm[batch_idx] > 1:
                    self.test_metrics['overall']['mean_angle'].append(angle[batch_idx].max().item())
                    self.test_metrics['overall']['angle_step1'].append(angle[batch_idx, 0].item())
                    self.test_metrics['overall']['angle_step2'].append(angle[batch_idx, 1].item())
                    self.test_metrics['overall']['angle_step3'].append(angle[batch_idx, 2].item())
                    self.test_metrics['overall']['angle_step4'].append(angle[batch_idx, 3].item())
                    self.test_metrics['overall']['angle_step5'].append(angle[batch_idx, 4].item())

        
        # Handle visualization
        if self.datatype == "citywalk":
            wp_pred *= batch['step_scale'].unsqueeze(-1).unsqueeze(-1)
        if self.output_coordinate_repr == "euclidean":
            self.process_visualization(
                mode='test',
                batch=batch,
                obs=obs,
                wp_pred=wp_pred
            )
        elif self.output_coordinate_repr == "polar":
            self.process_visualization(
                mode='test',
                batch=batch,
                obs=obs,
                wp_pred=wp_pred
            )

    def on_test_epoch_end(self):
        if self.datatype == "citywalk":
            for metric in self.test_metrics:
                metric_array = np.array(self.test_metrics[metric])
                save_path = os.path.join(self.result_dir, f'test_{metric}.npy')
                np.save(save_path, metric_array)
                if not metric == "mean_angle":
                    print(f"Test mean {metric} {metric_array.mean():.4f} saved to {save_path}")
                else:
                    mean_angle = metric_array.mean(axis=0)
                    for i in range(len(mean_angle)):
                        print(f"Test mean angle at step {i} {mean_angle[i]:.4f}")
        elif self.datatype == "urbannav":
            import pandas as pd
            for category in self.test_catetories:
                # Add a new 'count' metric for each category by counting 'l1_loss' entries
                self.test_metrics[category]['count'] = len(self.test_metrics[category]['l1_loss'])
            self.test_metrics['overall']['count'] = sum(self.test_metrics[category]['count'] for category in self.test_catetories)
            self.test_metrics['mean']['count'] = 0

            for category in self.test_catetories:
                for metric in self.test_metrics[category]:
                    if metric != 'count':
                        # print(f"{category} {metric}: {self.test_metrics[category][metric]}")
                        self.test_metrics[category][metric] = np.nanmean(np.array(self.test_metrics[category][metric]))
            for metric in self.test_metrics['overall']:
                if metric != 'count':
                    self.test_metrics['overall'][metric] = np.nanmean(np.array(self.test_metrics['overall'][metric]))
            metrics = ['l1_loss', 'arrived_accuracy', 'angle_step1', 'angle_step2', 'angle_step3', 'angle_step4', 'angle_step5', 'mean_angle']
            for metric in metrics:
                category_val = []
                for category in self.test_catetories:
                    category_val.append(self.test_metrics[category][metric])
                self.test_metrics['mean'][metric] = np.array(category_val).mean()
                print(f"{metric}: Sample mean {self.test_metrics['overall'][metric]:.4f}, Category mean {self.test_metrics['mean'][metric]:.4f}")

            df = pd.DataFrame(self.test_metrics)
            df = df.reset_index().rename(columns={'index': 'Metrics'})
            save_path = os.path.join(self.result_dir, 'test_metrics.csv')
            df.to_csv(save_path, index=False)


    def on_validation_epoch_start(self):
        self.vis_count = 0

    def on_test_epoch_start(self):
        self.vis_count = 0
        if self.datatype == "citywalk":
            if self.output_coordinate_repr == "euclidean":
                self.test_metrics = {'l1_loss': [], 'arrived_accuracy': [], 'mean_angle': []}
            elif self.output_coordinate_repr == "polar":
                self.test_metrics = {'distance_loss': [], 'angle_loss': [], 'arrived_accuracy': [], 'mean_angle': []}
        elif self.datatype == "urbannav":
            self.test_metrics = {}
            categories = self.test_catetories[:]
            categories.extend(['mean', 'overall'])
            for category in categories:
                if self.output_coordinate_repr == "euclidean":
                    self.test_metrics[category] = {
                        'l1_loss': [], 
                        'arrived_accuracy': [], 
                        'angle_step1': [],
                        'angle_step2': [],
                        'angle_step3': [],
                        'angle_step4': [],
                        'angle_step5': [],
                        'mean_angle': []
                    }
                elif self.output_coordinate_repr == "polar":
                    raise ValueError("Polar representation is not supported for UrbanNav dataset.")

    def configure_optimizers(self):
        optimizer_name = self.cfg.optimizer.name.lower()
        lr = float(self.cfg.optimizer.lr)

        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=self.cfg.optimizer.weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=self.cfg.optimizer.weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
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

    def process_visualization(self, mode, batch, obs, wp_pred):
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
        for idx in range(batch_size):
            if self.vis_count >= num_visualize:
                break

            original_input_positions = batch['original_input_positions'][idx].cpu().numpy()
            noisy_input_positions = batch['noisy_input_positions'][idx].cpu().numpy()
            gt_waypoints = batch['gt_waypoints'][idx].cpu().numpy()
            pred_waypoints = wp_pred[idx].detach().cpu().numpy()

            gt_waypoints_y = batch['gt_waypoints_y'][idx].cpu().numpy()     # for visualization

            # Get the last frame from the sequence
            frame = obs[idx, -1].permute(1, 2, 0).cpu().numpy()     # (h, w, 3); most recent frame
            frame = (frame * 255).astype(np.uint8)  # Convert to uint8 for visualization
            H, W, _ = frame.shape
            # Visualization title

            arrive_title = ''
            # Plotting
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            plt.subplots_adjust(wspace=0.3)

            # Left axis: plot the current observation (frame) with arrived info in title

            ax1.imshow(frame)
            ax1.axis('off')
            ax1.set_title(arrive_title, fontsize=20)


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

            u_gt, v_gt, valid = project_waypoints_onto_image_plane(gt_waypoints, gt_waypoints_y, K=K)
            u_gt, v_gt = shift(u_gt, v_gt)
            
            if np.all(valid):
                ax1.plot(u_gt, v_gt, color='#92DB58', linewidth=3)
            ax1.set_xlim(0., W)
            ax1.set_ylim(H, 0.)

            # Since the model does not output y, just augment the outputs with ground truth y
            
            if pred_waypoints.ndim == 3:
                # multiple samples drawn from the model
                for sample_idx in range(pred_waypoints.shape[0]):
                    u_pred, v_pred, valid = project_waypoints_onto_image_plane(pred_waypoints[sample_idx], gt_waypoints_y, K=K)
                    u_pred, v_pred = shift(u_pred, v_pred)
                    if np.all(valid):
                        ax1.plot(u_pred[valid], v_pred[valid], color='#DB6057')
            elif pred_waypoints.ndim == 2:
                u_pred, v_pred, valid = project_waypoints_onto_image_plane(pred_waypoints, gt_waypoints_y, K=K)
                u_pred, v_pred = shift(u_pred, v_pred)
                if np.all(valid):
                    ax1.plot(u_pred[valid], v_pred[valid], color='#DB6057')
            else:
                raise ValueError


            # Right axis: plot the coordinates
            
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
            ax2.plot(gt_waypoints[:, 0], gt_waypoints[:, 1],
                     'X-', label='GT Waypoints', color='#92DB58')
            if pred_waypoints.ndim == 3:
                # multiple samples drawn from the model
                labeled = False
                for sample_idx in range(pred_waypoints.shape[0]):
                    if not labeled:
                        label = 'Predicted Waypoints'
                        labeled = True
                    else:
                        label = None
                    ax2.plot(pred_waypoints[sample_idx, :, 0], pred_waypoints[sample_idx, :, 1],
                         label=label, color='#DB6057', alpha=0.8)
            elif pred_waypoints.ndim == 2:
                ax2.plot(pred_waypoints[:, 0], pred_waypoints[:, 1],
                         label='Predicted Waypoints', color='#DB6057')
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

    def waypoints_to_polar(self, waypoints):
        # Compute relative differences
        deltas = torch.diff(waypoints, dim=1, prepend=torch.zeros_like(waypoints[:, :1, :]))
        distance = torch.norm(deltas, dim=2)
        angle = torch.atan2(deltas[:, :, 1], deltas[:, :, 0])
        return distance, angle


