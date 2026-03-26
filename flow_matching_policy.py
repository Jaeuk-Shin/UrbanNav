from model.citywalker import CityWalker
from pl_modules.citywalker_module import CityWalkerModule
from pl_modules.flow_matching_module import FlowMatchingModule
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import carla
from model.model_utils import PolarEmbedding


class TrajectorySampler:
    def __init__(self, cfg, ckpt, future_length, history_length, step_scale=1.):
        self.model = FlowMatchingModule.load_from_checkpoint(ckpt, cfg=cfg)
        self.model.to(torch.bfloat16)
        self.model.eval()

        assert future_length == cfg.model.decoder.len_traj_pred
        assert history_length == cfg.model.obs_encoder.context_size

        self.N = future_length
        self.H = history_length

        self.step_scale = step_scale

        # buffers; stack historical inputs here
        self.observations = deque(maxlen=self.H)  # past H images
        self.poses = deque(maxlen=self.H)         # past H poses

    def reset(self):
        """
        initialize the observation & pose buffer
        must be called at the beginning of every episode
        """
        self.observations = deque(maxlen=self.H)
        self.poses = deque(maxlen=self.H)


    def sample_actions(self, o, pose, n_samples=10):
        """
        Given an RGB image I_t & camera pose T_t (camera-to-world), sample N waypoints
        
        Parameters
        ----------
        o: RGB image, given as a numpy array of shape (height, width, 3)
        pose: camera pose (camera-to-world); numpy array of shape (7,)
              x, y, z, qx, qy, qz, qw
        """
        self.observations.append(o)
        self.poses.append(pose)

        self.prepend()  # just for case where insufficient amount of data have been collected

        frames = np.array(self.observations)                # (H, height, width, C)
        input_poses = np.array(self.poses)                  # (H, 7)

        input_positions = transform_poses(input_poses, pose)[:, [0, 2]]         # (H, 2); xz
        cord = torch.tensor(input_positions).to(torch.bfloat16) / self.step_scale        
        cord = cord.unsqueeze(0).to(self.model.device)                                                      # (1, H, 2)
        obs = process_frames(frames).unsqueeze(0).to(self.model.device).to(torch.bfloat16)                  # (1, H, C, height, width)

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = self.model.model.sample(obs, cord, num_samples=n_samples)     # (1, # of samples, N, 2)

        info = {'input_positions': np.copy(input_positions)}

        # (# of samples, N, 2)
        return out.squeeze(0).cpu().numpy(), info     


    def prepend(self):
        """
        padding
        """
        buffer_len = len(self.observations)
        if buffer_len < self.H:
            o_init = self.observations[0]
            pose_init = self.poses[0]
            
            for _ in range(self.H - buffer_len):
                self.observations.appendleft(o_init)
                self.poses.appendleft(pose_init)
        return



def transform_poses(poses, current_pose_array):
    current_pose_matrix = pose_to_matrix(current_pose_array)
    current_pose_inv = np.linalg.inv(current_pose_matrix)
    pose_matrices = poses_to_matrices(poses)
    transformed_matrices = np.matmul(current_pose_inv[np.newaxis, :, :], pose_matrices)
    positions = transformed_matrices[:, :3, 3]
    return positions

def pose_to_matrix(pose):
    position = pose[:3]
    rotation = R.from_quat(pose[3:])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = position
    return matrix

def poses_to_matrices(poses):
    positions = poses[:, :3]
    quats = poses[:, 3:]
    rotations = R.from_quat(quats)
    matrices = np.tile(np.eye(4), (poses.shape[0], 1, 1))
    matrices[:, :3, :3] = rotations.as_matrix()
    matrices[:, :3, 3] = positions
    return matrices

# Warning: This must match that of CityWalkDataset!
def process_frames(frames):
    frames = torch.tensor(frames).permute(0, 3, 1, 2).to(torch.bfloat16) / 255.0  # Corrected normalization
    # Desired resolution
    desired_height = 360
    desired_width = 640
    
    # Current resolution
    _, _, H, W = frames.shape
    
    # Calculate padding needed
    pad_height = desired_height - H
    pad_width = desired_width - W
    
    # Only pad if necessary
    if pad_height > 0 or pad_width > 0:
        # Calculate padding for each side (left, right, top, bottom)
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        
        # Apply padding
        frames = TF.pad(
            frames, 
            (pad_left, pad_top, pad_right, pad_bottom),
        )
        
        # Optional: Verify the new shape
        assert frames.shape[2] == desired_height and frames.shape[3] == desired_width, \
            f"Padded frames have incorrect shape: {frames.shape}. Expected ({desired_height}, {desired_width})"
        
    if pad_height < 0  or pad_width < 0:
        frames = TF.center_crop(frames, (desired_height, desired_width))
    
    return frames