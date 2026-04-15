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
        self.data_dir = cfg.data.root
        pose_subdir = getattr(cfg.data, 'pose_subdir', 'pose')
        self.pose_dir = os.path.join(self.data_dir, pose_subdir)
        feature_subdir = getattr(cfg.data, 'feature_subdir', 'dino')
        self.feature_dir = getattr(cfg.data, 'feature_dir',
                                   os.path.join(self.data_dir, feature_subdir))
        self.context_size = cfg.model.obs_encoder.context_size
        self.wp_length = cfg.model.decoder.len_traj_pred
        self.video_fps = cfg.data.video_fps
        self.pose_fps = cfg.data.pose_fps
        self.target_fps = cfg.data.target_fps
        self.num_workers = cfg.data.num_workers
        self.input_noise = cfg.data.input_noise

        self.pose_step = max(1, self.pose_fps // self.target_fps)
        self.frame_step = self.video_fps // self.target_fps

        # Load pose paths — sort first for determinism, then shuffle with a
        # fixed seed so that train/val/test splits are random but reproducible.
        # Without the shuffle the tail-slice val/test sets are biased toward
        # whichever episodes sort last alphabetically.
        self.pose_path = [
            os.path.join(self.pose_dir, f)
            for f in sorted(os.listdir(self.pose_dir))
            if f.endswith('.txt')
        ]
        rng = random.Random(42)
        rng.shuffle(self.pose_path)

        if mode == 'train':
            self.pose_path = self.pose_path[:cfg.data.num_train]
        elif mode == 'val':
            self.pose_path = self.pose_path[-cfg.data.num_val:]
        elif mode == 'test':
            self.pose_path = self.pose_path[-cfg.data.num_test:]
        else:
            raise ValueError(f"Invalid mode {mode}")

        # Optional keep-list filtering (produced by filter_episodes.py)
        keep_list_path = getattr(cfg.data, 'keep_list', None)
        if keep_list_path and os.path.exists(keep_list_path):
            with open(keep_list_path) as f:
                keep_names = set(line.strip() for line in f if line.strip())
            before = len(self.pose_path)
            self.pose_path = [
                p for p in self.pose_path
                if os.path.basename(p) in keep_names
            ]
            print(f"[{mode}] keep_list filter: {before} → {len(self.pose_path)} episodes")

        # Load corresponding feature file paths.
        # Build a lookup from normalized name (single quotes stripped) to
        # actual filename so we can tolerate quote mismatches between pose
        # files and precomputed feature files (common with yt-dlp downloads).
        available_pts = {}
        if os.path.isdir(self.feature_dir):
            for fn in os.listdir(self.feature_dir):
                if fn.endswith('.pt') and fn != 'metadata.pt':
                    key = fn.replace("'", "")
                    available_pts[key] = fn

        self.feature_paths = []
        for f in self.pose_path:
            name, _ = os.path.splitext(os.path.basename(f))
            feat_path = os.path.join(self.feature_dir, f'{name}.pt')
            if not os.path.exists(feat_path):
                # Try matching with quotes stripped from both sides
                norm_key = f'{name}.pt'.replace("'", "")
                if norm_key in available_pts:
                    feat_path = os.path.join(
                        self.feature_dir, available_pts[norm_key])
                else:
                    # Show available files for diagnosis
                    avail = sorted(available_pts.values())[:10]
                    raise FileNotFoundError(
                        f"Feature file {feat_path} not found. "
                        f"Run precompute_features.py first.\n"
                        f"Available .pt files in {self.feature_dir} "
                        f"(first 10): {avail}"
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

        # Locate raw image directories (optional — used for visualization).
        # When cfg.data.rgb_dir is set (e.g. by mixture datamodule), look for
        # frame directories at {rgb_dir}/{episode_name}/.  Otherwise fall back
        # to the legacy CARLA layout: {root_dir}/{episode_name}/fcam/.
        rgb_subdir = getattr(cfg.data, 'rgb_subdir', 'rgb')
        rgb_dir = getattr(cfg.data, 'rgb_dir',
                          os.path.join(self.data_dir, rgb_subdir))
        self.image_dirs = []
        for pp in self.pose_path:
            name = os.path.splitext(os.path.basename(pp))[0]
            if rgb_dir:
                img_dir = os.path.join(rgb_dir, name)
            else:
                img_dir = os.path.join(self.data_dir, name, 'fcam')
            self.image_dirs.append(img_dir if os.path.isdir(img_dir) else None)

        # Locate video files for episodes that lack a frame directory.
        # Used for on-the-fly frame extraction during visualization via decord.
        # Like the feature-file lookup above, we build a quote-stripped index
        # so that yt-dlp naming mismatches (e.g. apostrophes in video titles)
        # don't prevent matching.
        _VIDEO_EXTS = ('.mp4', '.webm', '.mkv', '.avi', '.mov')
        available_videos = {}
        if rgb_dir and os.path.isdir(rgb_dir):
            for fn in os.listdir(rgb_dir):
                if any(fn.lower().endswith(ext) for ext in _VIDEO_EXTS):
                    key = fn.replace("'", "")
                    available_videos[key] = fn

        self.video_paths = []
        for i, pp in enumerate(self.pose_path):
            video_path = None
            if self.image_dirs[i] is None and rgb_dir:
                name = os.path.splitext(os.path.basename(pp))[0]
                # Direct lookup
                for ext in _VIDEO_EXTS:
                    candidate = os.path.join(rgb_dir, f'{name}{ext}')
                    if os.path.exists(candidate):
                        video_path = candidate
                        break
                # Quote-tolerant fallback
                if video_path is None:
                    for ext in _VIDEO_EXTS:
                        norm_key = f'{name}{ext}'.replace("'", "")
                        if norm_key in available_videos:
                            video_path = os.path.join(
                                rgb_dir, available_videos[norm_key])
                            break
            self.video_paths.append(video_path)

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

        # Camera intrinsics (optional — for image-overlay visualization).
        # Internally normalized to [fx, fy, cx, cy, desired_width, desired_height].
        # Accepts two config styles:
        #   Explicit:  camera: {fx, fy, cx, cy, desired_width, desired_height}
        #   FOV-based: camera: {width, height, fov, desired_width, desired_height}
        # Legacy top-level data fields (width/height/fov/…) are also supported.
        self._camera = self._parse_camera_intrinsics(cfg)

        # Data augmentation
        self.augment = (mode == 'train')
        self.horizontal_flip_prob = (
            getattr(cfg.data, 'horizontal_flip_prob', 0.5)
            if (self.augment and self.include_flip) else 0.0
        )

        # Per-worker feature file cache
        self._feat_cache = {'idx': None, 'data': None}

    @staticmethod
    def _parse_camera_intrinsics(cfg):
        """Return [fx, fy, cx, cy, dw, dh] or None."""

        def _from_fov(width, height, fov, dw, dh):
            f = 0.5 * width / np.tan(float(fov) * np.pi / 360.0)
            return [f, f, 0.5 * width, 0.5 * height, dw, dh]

        cam = getattr(cfg.data, 'camera', None)
        if cam is not None:
            dw = float(cam['desired_width'])
            dh = float(cam['desired_height'])
            if 'fx' in cam:
                return [float(cam['fx']), float(cam['fy']),
                        float(cam['cx']), float(cam['cy']), dw, dh]
            else:
                return _from_fov(float(cam['width']), float(cam['height']),
                                 cam['fov'], dw, dh)

        # Legacy top-level data fields (always FOV-based)
        keys = ('width', 'height', 'fov', 'desired_width', 'desired_height')
        if all(hasattr(cfg.data, k) for k in keys):
            d = cfg.data
            return _from_fov(float(d.width), float(d.height), d.fov,
                             float(d.desired_width), float(d.desired_height))

        return None

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

            # Video path + frame index for on-the-fly decord extraction when
            # no pre-split frames are available.  Empty/sentinel values keep
            # every sample's keys uniform for default collation.
            video_path = ''
            video_frame_idx = -1
            if not image_path and self.video_paths[video_idx] is not None:
                video_path = self.video_paths[video_idx]
                video_frame_idx = int(frame_indices[-1])
            sample['video_path'] = video_path
            sample['video_frame_idx'] = video_frame_idx

            # Per-sample camera intrinsics [fx, fy, cx, cy, dw, dh].
            # Sentinel -1 means "no camera" — keeps keys uniform across
            # datasets in a ConcatDataset so default collation works.
            if self._camera is not None:
                sample['camera_intrinsics'] = torch.tensor(
                    self._camera, dtype=torch.float32)
            else:
                sample['camera_intrinsics'] = torch.full((6,), -1.0)

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
