import os
import numpy as np
import torch
from torch.utils.data import Dataset
from data.carla_dataset import CarlaSampler
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import random


class CarlaFeatDataset(Dataset):
    """
    CARLA dataset that loads precomputed DINOv2 features instead of raw images.

    Expects features precomputed by precompute_features.py, stored as:
        {feature_dir}/{episode_name}.pt
    Each file contains:
        'features':      (num_frames, feature_dim) tensor
        'features_flip': (num_frames, feature_dim) tensor  [optional]

    Returns 'obs_features' of shape (N, feature_dim) instead of 'video_frames'.
    """

    def __init__(self, cfg, mode):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.data_dir = cfg.data.root_dir
        self.pose_dir = os.path.join(self.data_dir, cfg.data.pose_dir)
        self.feature_dir = cfg.data.feature_dir
        self.context_size = cfg.model.obs_encoder.context_size
        self.wp_length = cfg.model.decoder.len_traj_pred
        self.video_fps = cfg.data.video_fps
        self.pose_fps = cfg.data.pose_fps
        self.target_fps = cfg.data.target_fps
        self.num_workers = cfg.data.num_workers
        self.input_noise = cfg.data.input_noise

        self.pose_step = max(1, self.pose_fps // self.target_fps)
        self.frame_step = self.video_fps // self.target_fps

        # Load pose paths
        self.pose_path = [
            os.path.join(self.pose_dir, f)
            for f in sorted(os.listdir(self.pose_dir))
            if f.endswith('.txt')
        ]

        if mode == 'train':
            self.pose_path = self.pose_path[:cfg.data.num_train]
        elif mode == 'val':
            self.pose_path = self.pose_path[-cfg.data.num_val:]
        elif mode == 'test':
            self.pose_path = self.pose_path[-cfg.data.num_test:]
        else:
            raise ValueError(f"Invalid mode {mode}")

        # Load corresponding feature file paths
        self.feature_paths = []
        for f in self.pose_path:
            name, _ = os.path.splitext(os.path.basename(f))
            feat_path = os.path.join(self.feature_dir, f'{name}.pt')
            if not os.path.exists(feat_path):
                raise FileNotFoundError(
                    f"Feature file {feat_path} not found. "
                    f"Run precompute_features.py first."
                )
            self.feature_paths.append(feat_path)

        # Load metadata
        metadata_path = os.path.join(self.feature_dir, 'metadata.pt')
        if os.path.exists(metadata_path):
            self.metadata = torch.load(metadata_path, weights_only=True)
            self.feature_dim = self.metadata['feature_dim']
            self.include_flip = self.metadata.get('include_flip', False)
        else:
            self.feature_dim = 768
            self.include_flip = False

        # Load poses and compute usable counts
        self.poses = []
        self.count = []
        for f in tqdm(self.pose_path, desc="Loading poses"):
            pose = np.loadtxt(f, delimiter=" ")[::self.pose_step, 1:]
            pose_nan = np.isnan(pose).any(axis=1)
            if np.any(pose_nan):
                first_nan_idx = np.argmin(pose_nan)
                pose = pose[:first_nan_idx]
            self.poses.append(pose)
            usable = pose.shape[0] - self.context_size - self.wp_length
            self.count.append(max(usable, 0))

        # Remove episodes with zero usable samples
        valid_indices = [i for i, c in enumerate(self.count) if c > 0]
        self.poses = [self.poses[i] for i in valid_indices]
        self.pose_path = [self.pose_path[i] for i in valid_indices]
        self.feature_paths = [self.feature_paths[i] for i in valid_indices]
        self.count = [self.count[i] for i in valid_indices]

        # Locate raw image directories (optional — used for visualization)
        self.image_dirs = []
        for pp in self.pose_path:
            name = os.path.splitext(os.path.basename(pp))[0]
            img_dir = os.path.join(self.data_dir, name, 'fcam')
            self.image_dirs.append(img_dir if os.path.isdir(img_dir) else None)

        self.step_scale = []
        for pose in self.poses:
            step_scale = np.linalg.norm(np.diff(pose[:, [0, 2]], axis=0), axis=1).mean()
            self.step_scale.append(step_scale)

        # Build look-up table and video_ranges
        self.lut = []
        self.video_ranges = []
        idx_counter = 0
        interval = self.context_size
        for video_idx, count in enumerate(self.count):
            start_idx = idx_counter
            for pose_start in range(0, count, interval):
                self.lut.append((video_idx, pose_start))
                idx_counter += 1
            end_idx = idx_counter
            self.video_ranges.append((start_idx, end_idx))
        assert len(self.lut) > 0, "No usable samples found."

        # Data augmentation
        self.augment = (mode == 'train')
        self.horizontal_flip_prob = (
            getattr(cfg.data, 'horizontal_flip_prob', 0.5)
            if (self.augment and self.include_flip) else 0.0
        )

        # Per-worker feature file cache
        self._feat_cache = {'idx': None, 'data': None}

    def __len__(self):
        return len(self.lut)

    def _load_features(self, video_idx):
        if self._feat_cache['idx'] != video_idx:
            self._feat_cache['data'] = torch.load(
                self.feature_paths[video_idx], weights_only=True
            )
            self._feat_cache['idx'] = video_idx
        return self._feat_cache['data']

    def __getitem__(self, index):
        video_idx, pose_start = self.lut[index]
        frame_indices = self.frame_step * np.arange(
            pose_start, pose_start + self.context_size
        )

        # Load cached features
        feat_data = self._load_features(video_idx)
        features = feat_data['features']  # (num_frames, feature_dim)
        num_frames = features.shape[0]
        frame_indices = [min(idx, num_frames - 1) for idx in frame_indices]

        # Decide horizontal flip augmentation
        use_flip = (
            self.augment
            and self.include_flip
            and random.random() < self.horizontal_flip_prob
        )

        if use_flip:
            obs_features = feat_data['features_flip'][frame_indices]
        else:
            obs_features = features[frame_indices]  # (N, feature_dim)

        # --- Pose processing (identical to CarlaDataset) ---
        pose = self.poses[video_idx]
        input_poses = pose[pose_start: pose_start + self.context_size]
        original_input_poses = np.copy(input_poses)

        waypoint_poses = pose[
            pose_start + self.context_size:
            pose_start + self.context_size + self.wp_length
        ]

        if self.input_noise > 0:
            input_poses = self._add_noise(input_poses)

        current_pose = input_poses[-1]

        if self.cfg.model.cord_embedding.type == 'input_target':
            transformed_input_positions = self._transform_poses(
                input_poses, current_pose
            )[:, [0, 2]]
        else:
            raise NotImplementedError(
                f"Coordinate embedding type {self.cfg.model.cord_embedding.type} "
                f"not implemented"
            )

        waypoints_transformed = self._transform_waypoints(waypoint_poses, current_pose)
        waypoint_transformed_y = np.copy(waypoints_transformed[:, 1])

        input_positions = torch.tensor(transformed_input_positions, dtype=torch.float32)
        waypoints_transformed = torch.tensor(
            waypoints_transformed[:, [0, 2]], dtype=torch.float32
        )
        step_scale = torch.tensor(self.step_scale[video_idx], dtype=torch.float32)
        step_scale = torch.clamp(step_scale, min=1e-2)
        input_positions_scaled = input_positions / step_scale
        waypoints_scaled = waypoints_transformed / step_scale
        input_positions_scaled[:self.context_size - 1] += (
            torch.randn(self.context_size - 1, 2) * self.input_noise
        )

        # Apply coordinate flip when using flipped features
        if use_flip:
            input_positions_scaled[:, 0] *= -1
            waypoints_scaled[:, 0] *= -1

        sample = {
            'obs_features': obs_features,       # (N, feature_dim)
            'input_positions': input_positions_scaled,
            'waypoints': waypoints_scaled,
            'step_scale': step_scale,
        }

        if self.mode in ['val', 'test']:
            transformed_original = self._transform_poses(
                original_input_poses, current_pose
            )
            original_input_positions = torch.tensor(
                transformed_original[:, [0, 2]], dtype=torch.float32
            )
            noisy_input_positions = input_positions_scaled[:-1] * step_scale

            sample['original_input_positions'] = original_input_positions
            sample['noisy_input_positions'] = noisy_input_positions
            sample['gt_waypoints'] = waypoints_transformed
            sample['gt_waypoints_y'] = waypoint_transformed_y

            # Include raw image path for visualization (empty string when
            # unavailable so that every val/test sample carries the key for
            # default collation)
            image_path = ''
            if self.image_dirs[video_idx] is not None:
                dirpath = self.image_dirs[video_idx]
                img_files = sorted(
                    f for f in os.listdir(dirpath) if f.endswith('.jpg')
                )
                if img_files:
                    last_idx = min(frame_indices[-1], len(img_files) - 1)
                    image_path = os.path.join(dirpath, img_files[last_idx])
            sample['image_path'] = image_path

        return sample

    # --- Helper methods (same as CarlaDataset) ---

    def _add_noise(self, input_poses):
        noise = np.random.normal(0, self.input_noise, input_poses[:, :3].shape)
        scale = np.linalg.norm(input_poses[-1, :3] - input_poses[-2, :3])
        input_poses = np.copy(input_poses)
        input_poses[:, :3] += noise * scale
        return input_poses

    def _transform_poses(self, poses, current_pose_array):
        current_pose_matrix = self._pose_to_matrix(current_pose_array)
        current_pose_inv = np.linalg.inv(current_pose_matrix)
        pose_matrices = self._poses_to_matrices(poses)
        transformed = np.matmul(current_pose_inv[np.newaxis, :, :], pose_matrices)
        return transformed[:, :3, 3]

    def _transform_waypoints(self, waypoint_poses, current_pose_array):
        current_pose_matrix = self._pose_to_matrix(current_pose_array)
        current_pose_inv = np.linalg.inv(current_pose_matrix)
        waypoint_matrices = self._poses_to_matrices(waypoint_poses)
        transformed = np.matmul(current_pose_inv[np.newaxis, :, :], waypoint_matrices)
        return transformed[:, :3, 3]

    def _pose_to_matrix(self, pose):
        position = pose[:3]
        rotation = R.from_quat(pose[3:])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation.as_matrix()
        matrix[:3, 3] = position
        return matrix

    def _poses_to_matrices(self, poses):
        positions = poses[:, :3]
        quats = poses[:, 3:]
        rotations = R.from_quat(quats)
        matrices = np.tile(np.eye(4), (poses.shape[0], 1, 1))
        matrices[:, :3, :3] = rotations.as_matrix()
        matrices[:, :3, 3] = positions
        return matrices
