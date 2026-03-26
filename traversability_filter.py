#!/usr/bin/env python3
"""
Traversability filter for wheeled robot training data.

Detects stairs, steep slopes, and rough terrain in YouTube/CityWalk data
that a wheeled robot cannot traverse, by:

  1. Estimating metric depth per frame (ZoeDepth)
  2. Backprojecting depth to 3D point clouds
  3. Registering point clouds into a common frame using ego poses (DPVO)
  4. Building local elevation maps on the XZ ground plane
  5. Checking slope, step height, and roughness along the trajectory

Usage:
  # Image directory format (CarlaDataset-style: episode_dir/fcam/*.jpg)
  python traversability_filter.py \\
      --data_dir /path/to/dataset \\
      --pose_dir /path/to/poses \\
      --output_dir ./traversability_output \\
      --fov 90

  # Video format (CityWalk-style: *.mp4 + matching *.txt poses)
  python traversability_filter.py \\
      --data_format video \\
      --video_dir /path/to/videos \\
      --pose_dir /path/to/poses \\
      --output_dir ./traversability_output \\
      --fov 90 --video_fps 30 --target_fps 2
"""

import argparse
import json
import os
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

try:
    from decord import VideoReader, cpu as decord_cpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False

from data.camera_utils import build_intrinsic_matrix


# ---------------------------------------------------------------------------
# Depth estimation
# ---------------------------------------------------------------------------

def load_depth_model(model_type="zoedepth", device="cuda"):
    """Load a pretrained monocular depth estimation model."""
    if model_type == "zoedepth":
        # Temporarily allow non-strict state_dict loading to handle
        # checkpoint version mismatches (e.g. relative_position_index
        # buffers present in the checkpoint but not registered by the
        # current model definition).
        _orig_load = torch.nn.Module.load_state_dict
        torch.nn.Module.load_state_dict = (
            lambda self, state_dict, strict=True, **kw:
                _orig_load(self, state_dict, strict=False, **kw)
        )
        try:
            model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
        finally:
            torch.nn.Module.load_state_dict = _orig_load
    else:
        raise ValueError(f"Unknown depth model: {model_type}")
    # Patch timm compatibility: newer timm renamed drop_path → drop_path1/drop_path2
    for m in model.modules():
        if hasattr(m, 'drop_path1') and not hasattr(m, 'drop_path'):
            m.drop_path = m.drop_path1
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def estimate_depth(model, frames, device="cuda", batch_size=1):
    """
    Estimate metric depth for a sequence of RGB frames.

    Replicates ZoeDepth's ``infer_pil`` preprocessing: ToTensor ([0,1],
    NO ImageNet normalisation) → resize to nearest multiple of 32 →
    forward → resize depth back.

    Args:
        model: ZoeDepth model
        frames: (N, H, W, 3) uint8 numpy array
        device: torch device string
        batch_size: frames per forward pass (higher = faster, more VRAM)

    Returns:
        depths: (N, H, W) float32 numpy array, depth in metres
    """
    from torchvision.transforms import functional as TF

    N = len(frames)
    if N == 0:
        return np.empty((0,), dtype=np.float32)
    H, W = frames[0].shape[:2]
    all_depths = []

    # Matching ZoeDepth infer(): resize to nearest multiple of 32
    new_h = ((H + 31) // 32) * 32
    new_w = ((W + 31) // 32) * 32
    needs_resize = (new_h != H) or (new_w != W)

    for b in range(0, N, batch_size):
        batch = frames[b:b + batch_size]
        # ZoeDepth expects [0, 1] input — NO ImageNet normalisation
        tensors = [TF.to_tensor(f) for f in batch]
        x = torch.stack(tensors).to(device)
        if needs_resize:
            x = torch.nn.functional.interpolate(
                x, (new_h, new_w), mode="bilinear", align_corners=False)
        out = model(x)
        d = out["metric_depth"] if isinstance(out, dict) else out
        if d.ndim == 4:
            d = d.squeeze(1)
        # Resize depth back to original resolution
        if d.shape[-2:] != (H, W):
            d = torch.nn.functional.interpolate(
                d.unsqueeze(1), size=(H, W),
                mode="bilinear", align_corners=False).squeeze(1)
        all_depths.append(d.cpu().numpy())
    return np.concatenate(all_depths).astype(np.float32)


# ---------------------------------------------------------------------------
# Semantic segmentation for dynamic object filtering
# ---------------------------------------------------------------------------

# Pascal VOC class IDs (used by DeepLabV3) for potentially dynamic objects.
DYNAMIC_CLASSES = frozenset({
    2,   # bicycle
    3,   # bird
    6,   # bus
    7,   # car
    8,   # cat
    10,  # cow
    12,  # dog
    13,  # horse
    14,  # motorbike
    15,  # person
    17,  # sheep
    19,  # train
})


def load_segmentation_model(device="cuda"):
    """Load DeepLabV3-ResNet101 for semantic segmentation of dynamic objects."""
    from torchvision.models.segmentation import (
        deeplabv3_resnet101,
        DeepLabV3_ResNet101_Weights,
    )
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def compute_dynamic_masks(seg_model, frames, device="cuda",
                          dynamic_classes=None, batch_size=1):
    """
    Run semantic segmentation and return binary masks of dynamic pixels.

    Args:
        seg_model:       DeepLabV3 model
        frames:          (N, H, W, 3) uint8 numpy array
        dynamic_classes: set of Pascal VOC class IDs to treat as dynamic
        batch_size:      frames per forward pass

    Returns:
        masks: (N, H, W) bool numpy array (True = dynamic pixel to exclude)
    """
    if dynamic_classes is None:
        dynamic_classes = DYNAMIC_CLASSES

    from torchvision.transforms import functional as TF

    _cls_arr = np.array(sorted(dynamic_classes))
    all_masks = []
    for b in range(0, len(frames), batch_size):
        batch = frames[b:b + batch_size]
        tensors = [TF.normalize(TF.to_tensor(f),
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                   for f in batch]
        x = torch.stack(tensors).to(device)
        preds = seg_model(x)["out"].argmax(1).cpu().numpy()
        all_masks.append(np.isin(preds, _cls_arr))
    return np.concatenate(all_masks)


# ---------------------------------------------------------------------------
# Point cloud writers (no external dependencies)
# ---------------------------------------------------------------------------

def _pack_binary_columns(points, scalar_fields):
    """Interleave xyz + scalar columns into a single contiguous buffer."""
    pts = np.ascontiguousarray(points, dtype=np.float32)
    N = len(pts)

    extra_arrays = []
    if scalar_fields:
        for arr in scalar_fields.values():
            arr = np.asarray(arr)
            if np.issubdtype(arr.dtype, np.floating):
                extra_arrays.append(arr.astype(np.float32))
            else:
                # reinterpret int32 as float32 bytes for column stacking
                extra_arrays.append(arr.astype(np.int32).view(np.float32))

    if extra_arrays:
        # Stack xyz + extras into (N, 3+K) float32 then write as flat bytes
        cols = [pts] + [a.reshape(N, 1) for a in extra_arrays]
        row_data = np.hstack(cols)
    else:
        row_data = pts
    return np.ascontiguousarray(row_data).tobytes()


def write_pcd(path, points, scalar_fields=None):
    """
    Write a binary PCD (Point Cloud Data) file.

    Args:
        path:           output file path
        points:         (N, 3) float array (x, y, z)
        scalar_fields:  dict of {name: (N,) array} for extra per-point scalars.
                        Supported dtypes: float32/64 → FLOAT, int32 → INT.
    """
    N = len(points)
    if scalar_fields is None:
        scalar_fields = {}

    fields = ["x", "y", "z"]
    sizes = [4, 4, 4]
    types = ["F", "F", "F"]
    counts = [1, 1, 1]

    for name, arr in scalar_fields.items():
        arr = np.asarray(arr)
        fields.append(name)
        counts.append(1)
        sizes.append(4)
        types.append("F" if np.issubdtype(arr.dtype, np.floating) else "I")

    header = (
        f"# .PCD v0.7 - Point Cloud Data file format\n"
        f"VERSION 0.7\n"
        f"FIELDS {' '.join(fields)}\n"
        f"SIZE {' '.join(str(s) for s in sizes)}\n"
        f"TYPE {' '.join(types)}\n"
        f"COUNT {' '.join(str(c) for c in counts)}\n"
        f"WIDTH {N}\n"
        f"HEIGHT 1\n"
        f"VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {N}\n"
        f"DATA binary\n"
    )

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(_pack_binary_columns(points, scalar_fields))


def write_ply(path, points, scalar_fields=None):
    """
    Write a binary little-endian PLY file.

    Args:
        path:           output file path
        points:         (N, 3) float array (x, y, z)
        scalar_fields:  dict of {name: (N,) array} for extra per-point scalars.
    """
    N = len(points)
    if scalar_fields is None:
        scalar_fields = {}

    props = "property float x\nproperty float y\nproperty float z\n"
    for name, arr in scalar_fields.items():
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.floating):
            props += f"property float {name}\n"
        else:
            props += f"property int {name}\n"

    header = (
        f"ply\n"
        f"format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        f"{props}"
        f"end_header\n"
    )

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(_pack_binary_columns(points, scalar_fields))


_PC_WRITERS = {
    "pcd": write_pcd,
    "ply": write_ply,
}


def _write_pc(path_no_ext, points, formats, scalar_fields=None):
    """Write a point cloud in each requested format."""
    for fmt in formats:
        if fmt == "npz":
            continue  # handled separately (has camera-frame data too)
        writer = _PC_WRITERS[fmt]
        writer(f"{path_no_ext}.{fmt}", points, scalar_fields=scalar_fields)


# ---------------------------------------------------------------------------
# Intermediate result saving
# ---------------------------------------------------------------------------

def save_depth_images(depths, save_dir):
    """Save raw depth arrays (.npy) and colorized depth PNGs."""
    raw_dir = os.path.join(save_dir, "depth", "raw")
    vis_dir = os.path.join(save_dir, "depth", "colorized")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    for i, d in enumerate(depths):
        np.save(os.path.join(raw_dir, f"{i:06d}.npy"), d)
        # Colorize with turbo colormap, clamp to [0, max_depth_vis]
        d_clip = np.clip(d, 0, np.nanpercentile(d, 98))
        d_norm = d_clip / (d_clip.max() + 1e-6)
        colored = (cm.turbo(d_norm)[:, :, :3] * 255).astype(np.uint8)
        Image.fromarray(colored).save(os.path.join(vis_dir, f"{i:06d}.png"))


def save_segmentation_results(frames, masks, save_dir):
    """Save per-frame segmentation overlays, binary masks, and statistics.

    Output layout under ``save_dir/segmentation/``:
        overlay/XXXXXX.png  — original frame with red tint on dynamic pixels
        mask/XXXXXX.png     — binary mask (white = dynamic)
        stats.json          — per-frame masked-pixel counts and percentages
    """
    seg_dir = os.path.join(save_dir, "segmentation")
    overlay_dir = os.path.join(seg_dir, "overlay")
    mask_dir = os.path.join(seg_dir, "mask")
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    stats = []
    for i in range(len(frames)):
        # Binary mask
        mask_img = (masks[i].astype(np.uint8) * 255)
        Image.fromarray(mask_img).save(
            os.path.join(mask_dir, f"{i:06d}.png"))

        # Overlay: original + semi-transparent red on dynamic pixels
        overlay = frames[i].copy()
        overlay[masks[i]] = (
            overlay[masks[i]].astype(np.float32) * 0.4
            + np.array([255, 0, 0], dtype=np.float32) * 0.6
        ).astype(np.uint8)
        Image.fromarray(overlay).save(
            os.path.join(overlay_dir, f"{i:06d}.png"))

        total = masks[i].size
        masked = int(masks[i].sum())
        stats.append({
            "frame": i,
            "masked_pixels": masked,
            "total_pixels": total,
            "masked_pct": round(100 * masked / total, 2),
        })

    with open(os.path.join(seg_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


def save_pointclouds(depths, K, T_all, save_dir, subsample=4,
                     max_depth=20.0, crop_top=0.3, pc_formats=("npz",),
                     render_views=True, max_view_frames=20):
    """Save per-frame point clouds in requested formats.

    Args:
        render_views:    If True, render 2D CCTV/BEV views for a subset
                         of frames into pointclouds/views/.
        max_view_frames: Maximum number of per-frame views to render.
    """
    pc_dir = os.path.join(save_dir, "pointclouds")
    os.makedirs(pc_dir, exist_ok=True)

    N = len(depths)
    # Indices for which to render 2D views (evenly spaced)
    if render_views and N > 0:
        view_step = max(1, N // max_view_frames)
        view_indices = set(range(0, N, view_step))
        # view_dir = os.path.join(pc_dir, "views")
        # os.makedirs(view_dir, exist_ok=True)
    else:
        view_indices = set()

    T_ref_inv = np.linalg.inv(T_all[0])
    for i, d in enumerate(depths):
        pc_cam = depth_to_pointcloud(d, K, subsample=subsample,
                                     max_depth=max_depth,
                                     crop_top_fraction=crop_top)
        if len(pc_cam) == 0:
            if "npz" in pc_formats:
                np.savez_compressed(
                    os.path.join(pc_dir, f"{i:06d}.npz"),
                    points_cam=np.zeros((0, 3), dtype=np.float64),
                    points_world=np.zeros((0, 3), dtype=np.float64))
            continue
        T_i2ref = T_ref_inv @ T_all[i]
        pc_world = transform_points(pc_cam, T_i2ref)
        if "npz" in pc_formats:
            np.savez_compressed(os.path.join(pc_dir, f"{i:06d}.npz"),
                                points_cam=pc_cam, points_world=pc_world)
        _write_pc(os.path.join(pc_dir, f"{i:06d}"), pc_world, pc_formats)
        '''
        if i in view_indices:
            save_pointcloud_frame_view(
                pc_cam, os.path.join(view_dir, f"{i:06d}.png"))
        '''
    return


def save_elevation_map_data(emap, window_idx, w0, w1, save_dir):
    """Save elevation map grid arrays as .npz."""
    em_dir = os.path.join(save_dir, "elevation_maps")
    os.makedirs(em_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(em_dir, f"window_{w0:04d}_{w1:04d}.npz"),
        elevation=emap.elevation(),
        slope=emap.slope(),
        step_height=emap.step_height(),
        roughness=emap.roughness(),
        point_count=emap._cnt,
        resolution=emap.resolution,
        extent=emap.extent,
    )


def save_merged_pointcloud(depths, K, T_all, save_dir, subsample=4,
                           max_depth=20.0, crop_top=0.3, pc_formats=("npz",),
                           render_views=True):
    """Save a single merged point cloud (all frames) in requested formats."""
    os.makedirs(save_dir, exist_ok=True)
    T_ref_inv = np.linalg.inv(T_all[0])
    all_pts = []
    all_frame_ids = []
    for i, d in enumerate(depths):
        pc_cam = depth_to_pointcloud(d, K, subsample=subsample,
                                     max_depth=max_depth,
                                     crop_top_fraction=crop_top)
        if len(pc_cam) == 0:
            continue
        pc_world = transform_points(pc_cam, T_ref_inv @ T_all[i])
        all_pts.append(pc_world)
        all_frame_ids.append(np.full(len(pc_world), i, dtype=np.int32))
    if all_pts:
        pts = np.concatenate(all_pts)
        ids = np.concatenate(all_frame_ids)
        if "npz" in pc_formats:
            np.savez_compressed(
                os.path.join(save_dir, "merged_pointcloud.npz"),
                points=pts, frame_ids=ids)
        _write_pc(os.path.join(save_dir, "merged_pointcloud"), pts,
                   pc_formats, scalar_fields={"frame_id": ids})

        if render_views:
            traj = np.array([(T_ref_inv @ T_all[i])[:3, 3]
                             for i in range(len(T_all))])
            save_pointcloud_views(
                pts, os.path.join(save_dir, "merged_pointcloud_views.png"),
                trajectory=traj)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def depth_to_pointcloud(depth, K, subsample=4, max_depth=20.0,
                        crop_top_fraction=0.3):
    """
    Backproject a depth map into a 3D point cloud in the camera frame.

    Camera convention (standard):
        X -> right,  Y -> down,  Z -> forward

    Args:
        depth:              (H, W) depth in metres
        K:                  3 x 3 intrinsic matrix
        subsample:          spatial stride (speed / memory trade-off)
        max_depth:          ignore pixels deeper than this (metres)
        crop_top_fraction:  fraction of image rows to discard from the top.
                            Keeps only the bottom part where ground is visible,
                            reducing noise from walls / sky.

    Returns:
        points: (M, 3) float64 array in camera coordinates
    """
    H, W = depth.shape
    row_start = int(H * crop_top_fraction)

    depth_sub = depth[row_start::subsample, ::subsample]
    h_sub, w_sub = depth_sub.shape

    u = np.arange(0, W, subsample).astype(np.float64)
    v = np.arange(row_start, H, subsample).astype(np.float64)
    u, v = np.meshgrid(u, v)

    valid = (depth_sub > 0.1) & (depth_sub < max_depth) & np.isfinite(depth_sub)
    d = depth_sub[valid]
    u = u[valid]
    v = v[valid]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    z = d

    return np.stack([x, y, z], axis=-1)


def pose_to_matrix(pose):
    """[x, y, z, qx, qy, qz, qw] → 4 x 4 SE(3)"""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(pose[3:], scalar_first=False).as_matrix()
    T[:3, 3] = pose[:3]
    return T


def poses_to_matrices(poses):
    """(N, 7) → (N, 4, 4)."""
    N = len(poses)
    matrices = np.tile(np.eye(4), (N, 1, 1))
    matrices[:, :3, :3] = Rotation.from_quat(poses[:, 3:]).as_matrix()
    matrices[:, :3, 3] = poses[:, :3]
    return matrices


def transform_points(points, T):
    """Apply 4 x 4 transform to (N, 3) points."""
    ones = np.ones((len(points), 1))
    return (T @ np.hstack([points, ones]).T).T[:, :3]


# ---------------------------------------------------------------------------
# Elevation map
# ---------------------------------------------------------------------------

class ElevationMap:
    """
    2-D grid on the XZ ground-plane that accumulates height (-Y) statistics.

    Coordinate convention (camera / DPVO world frame):
        X -> right,  Y -> down,  Z -> forward
        elevation = -Y  (higher elevation ↔ smaller Y)
    """

    def __init__(self, resolution=0.05, extent=15.0):
        """
        Args:
            resolution: cell side length (metres)
            extent:     half-width of the grid (metres).
                        Grid covers [-extent, +extent] in both X and Z.
        """
        self.resolution = resolution
        self.extent = extent
        self.size = int(2 * extent / resolution)

        self._sum = np.zeros((self.size, self.size), dtype=np.float64)
        self._sq  = np.zeros((self.size, self.size), dtype=np.float64)
        self._min = np.full((self.size, self.size), np.inf)
        self._max = np.full((self.size, self.size), -np.inf)
        self._cnt = np.zeros((self.size, self.size), dtype=np.int64)

        # Per-cell list of per-frame mean heights (for temporal consistency)
        self._frame_heights = defaultdict(list)  # linear_idx -> [mean_h, ...]

    # -- accumulate ---------------------------------------------------------

    def add_points(self, points, frame_id=None):
        """
        Insert (N, 3) points in world/reference frame into the map.

        Args:
            points:   (N, 3) array in reference frame
            frame_id: optional int — if provided, per-frame mean heights
                      are tracked per cell for temporal consistency filtering
        """
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        h = -y                       # elevation

        gi = ((x + self.extent) / self.resolution).astype(np.int32)
        gj = ((z + self.extent) / self.resolution).astype(np.int32)
        ok = (gi >= 0) & (gi < self.size) & (gj >= 0) & (gj < self.size)
        gi, gj, h = gi[ok], gj[ok], h[ok]

        if len(h) == 0:
            return

        np.add.at(self._sum, (gi, gj), h)
        np.add.at(self._sq,  (gi, gj), h ** 2)
        np.minimum.at(self._min, (gi, gj), h)
        np.maximum.at(self._max, (gi, gj), h)
        np.add.at(self._cnt, (gi, gj), 1)

        # Track per-frame mean height per cell (for temporal filtering)
        if frame_id is not None:
            linear = gi * self.size + gj
            n_cells = self.size * self.size
            frame_sum = np.bincount(linear, weights=h, minlength=n_cells)
            frame_cnt = np.bincount(linear, minlength=n_cells)
            active = frame_cnt > 0
            for idx in np.where(active)[0]:
                self._frame_heights[idx].append(
                    frame_sum[idx] / frame_cnt[idx]
                )

    # -- queries ------------------------------------------------------------

    def elevation(self, min_pts=3):
        """Mean elevation per cell; NaN where insufficient data."""
        out = np.full((self.size, self.size), np.nan)
        ok = self._cnt >= min_pts
        out[ok] = self._sum[ok] / self._cnt[ok]
        return out

    def roughness(self, min_pts=5):
        """Standard deviation of point heights per cell."""
        out = np.full((self.size, self.size), np.nan)
        ok = self._cnt >= min_pts
        mu = self._sum[ok] / self._cnt[ok]
        var = self._sq[ok] / self._cnt[ok] - mu ** 2
        out[ok] = np.sqrt(np.clip(var, 0, None))
        return out

    def slope(self, min_pts=3):
        """Terrain slope in degrees (via finite-difference gradient)."""
        elev = self.elevation(min_pts)
        gx = np.gradient(elev, self.resolution, axis=0)
        gz = np.gradient(elev, self.resolution, axis=1)
        return np.degrees(np.arctan(np.sqrt(gx ** 2 + gz ** 2)))

    def step_height(self, min_pts=3):
        """Max absolute height difference to any 4-connected neighbour."""
        elev = self.elevation(min_pts)
        dx = np.abs(np.diff(elev, axis=0))
        dz = np.abs(np.diff(elev, axis=1))
        out = np.zeros_like(elev)
        out[:-1, :] = np.fmax(out[:-1, :], dx)
        out[1:,  :] = np.fmax(out[1:,  :], dx)
        out[:, :-1] = np.fmax(out[:, :-1], dz)
        out[:,  1:] = np.fmax(out[:,  1:], dz)
        return out

    # -- temporal consistency -----------------------------------------------

    def temporal_std(self, min_frames=2):
        """
        Std of per-frame mean heights per cell.

        Cells where different frames contribute very different heights
        likely contain dynamic objects (pedestrians, vehicles, etc.).
        Returns (size, size) array; 0 where insufficient frame data.
        """
        out = np.zeros((self.size, self.size), dtype=np.float64)
        for lin, heights in self._frame_heights.items():
            if len(heights) >= min_frames:
                gi, gj = divmod(lin, self.size)
                out[gi, gj] = np.std(heights)
        return out

    def filter_unstable_cells(self, max_temporal_std=0.15, min_frames=2):
        """
        Invalidate cells where temporal height variance exceeds threshold.

        This removes ghost points from dynamic objects (pedestrians, cars)
        that pass through a cell at different heights across frames.

        Args:
            max_temporal_std: height std threshold (metres)
            min_frames: require at least this many frames to judge stability

        Returns:
            n_filtered: number of cells invalidated
        """
        n_filtered = 0
        for lin, heights in self._frame_heights.items():
            if len(heights) >= min_frames and np.std(heights) > max_temporal_std:
                gi, gj = divmod(lin, self.size)
                self._sum[gi, gj] = 0
                self._sq[gi, gj] = 0
                self._min[gi, gj] = np.inf
                self._max[gi, gj] = -np.inf
                self._cnt[gi, gj] = 0
                n_filtered += 1
        return n_filtered

    # -- trajectory analysis ------------------------------------------------

    def world_to_grid(self, x, z):
        gi = int((x + self.extent) / self.resolution)
        gj = int((z + self.extent) / self.resolution)
        return np.clip(gi, 0, self.size - 1), np.clip(gj, 0, self.size - 1)

    def query_trajectory(self, positions, radius_cells=3):
        """
        Compute worst-case slope / step / roughness in a neighbourhood
        around each trajectory position.

        Args:
            positions: (T, 3) world-frame positions
            radius_cells: neighbourhood half-width in grid cells

        Returns:
            dict  {max_slope, max_step, max_roughness, has_data}
                  each (T,) float array
        """

        elev   = self.elevation()
        sl     = self.slope()
        st     = self.step_height()
        ro     = self.roughness()
        r      = radius_cells

        out = {k: np.full(len(positions), np.nan)
               for k in ("max_slope", "max_step", "max_roughness")}
        out["has_data"] = np.zeros(len(positions), dtype=bool)

        for t, pos in enumerate(positions):
            gi, gj = self.world_to_grid(pos[0], pos[2])
            i0, i1 = max(gi - r, 0), min(gi + r + 1, self.size)
            j0, j1 = max(gj - r, 0), min(gj + r + 1, self.size)

            patch_elev = elev[i0:i1, j0:j1]
            if np.all(np.isnan(patch_elev)):
                continue
            out["has_data"][t] = True
            out["max_slope"][t]     = np.nanmax(sl[i0:i1, j0:j1])
            out["max_step"][t]      = np.nanmax(st[i0:i1, j0:j1])
            out["max_roughness"][t] = np.nanmax(ro[i0:i1, j0:j1])

        return out


# ---------------------------------------------------------------------------
# Traversability decision
# ---------------------------------------------------------------------------

def is_traversable(stats, max_slope_deg, max_step_m, max_rough_m):
    """
    Returns (feasible: bool, reason: str | None).
    Conservative: segments with no elevation data are assumed feasible.
    """
    if not np.any(stats["has_data"]):
        return True, None

    s = np.nanmax(stats["max_slope"])
    h = np.nanmax(stats["max_step"])
    r = np.nanmax(stats["max_roughness"])

    if h > max_step_m:
        return False, f"step={h:.3f}m>{max_step_m}m"
    if s > max_slope_deg:
        return False, f"slope={s:.1f}°>{max_slope_deg}°"
    if r > max_rough_m:
        return False, f"rough={r:.3f}m>{max_rough_m}m"
    return True, None


# ---------------------------------------------------------------------------
# Data I/O (mirrors CarlaDataset / CityWalkDataset)
# ---------------------------------------------------------------------------

def load_poses(path, pose_step=1):
    """Load [ts, x, y, z, qx, qy, qz, qw] → (N, 7).

    Returns:
        poses: (M, 7) array of valid (non-NaN) poses.
        valid_mask: (N,) bool array indicating which subsampled rows are valid.
    """
    raw = np.loadtxt(path, delimiter=" ")
    poses = raw[::pose_step, 1:]          # drop timestamp, subsample
    valid_mask = ~np.isnan(poses).any(axis=1)
    return poses[valid_mask], valid_mask


def load_frames_images(image_dir, indices=None):
    """Load frames from a directory of JPG/PNG files."""
    paths = sorted(p for p in os.listdir(image_dir)
                   if p.lower().endswith((".jpg", ".jpeg", ".png")))
    if indices is not None:
        paths = [paths[min(i, len(paths) - 1)] for i in indices]
    imgs = [np.array(Image.open(os.path.join(image_dir, p)).convert("RGB"))
            for p in paths]
    return np.stack(imgs)


def load_frames_video(video_path, indices=None):
    """Load frames from an MP4 via decord."""
    if not HAS_DECORD:
        raise ImportError("pip install decord")
    vr = VideoReader(str(video_path), ctx=decord_cpu(0))
    if indices is None:
        indices = list(range(len(vr)))
    else:
        indices = [i for i in indices if i < len(vr)]
    return vr.get_batch(indices).asnumpy()


# ---------------------------------------------------------------------------
# Episode processing
# ---------------------------------------------------------------------------

def process_episode(
    frames,           # (N, H, W, 3) uint8
    poses,            # (N, 7) float
    K,                # (3, 3) intrinsic
    depth_model,
    *,
    window_size=15,
    window_stride=5,
    grid_res=0.05,
    grid_extent=15.0,
    depth_subsample=4,
    crop_top=0.3,
    max_depth=20.0,
    max_slope_deg=15.0,
    max_step_m=0.12,
    max_rough_m=0.08,
    device="cuda",
    save_dir=None,
    pc_formats=("npz",),
    seg_model=None,
    temporal_filter=False,
    temporal_std_threshold=0.15,
    batch_size=8,
    seg_device=None,
):
    """
    Run the full traversability pipeline on one episode.

    Uses batched GPU inference (strategy 1), pipeline overlap between GPU
    and CPU work (strategy 3), and optional concurrent depth + segmentation
    on separate GPUs (strategy 4).

    Args:
        save_dir:                If not None, save intermediate results
                                 (depth images, point clouds, elevation maps).
        pc_formats:              Point cloud formats ("npz", "pcd", "ply").
        seg_model:               DeepLabV3 model for semantic masking of
                                 dynamic objects.  None = disabled.
        temporal_filter:         If True, filter elevation map cells whose
                                 per-frame height std exceeds the threshold.
        temporal_std_threshold:  Height std threshold (metres) for temporal
                                 consistency filter.
        batch_size:              Frames per GPU forward pass.
        seg_device:              Device for segmentation model.  If different
                                 from ``device``, depth and seg run concurrently
                                 (strategy 4).  None = same as ``device``.

    Returns:
        feasible  : (N,) bool — per-frame label (True = OK for wheeled robot)
        windows   : list[dict] — per-window diagnostics
    """
    N = min(len(frames), len(poses))
    frames, poses = frames[:N], poses[:N]
    H, W = frames[0].shape[:2]

    # --- Phase 1: Batched GPU inference (strategies 1, 3, 4) ---------------
    _seg_device = seg_device or device
    _concurrent_seg = seg_model is not None and _seg_device != device

    tag = ""
    if _concurrent_seg:
        tag = f", depth@{device} seg@{_seg_device}"
    print(f"  GPU inference ({N} frames, batch_size={batch_size}{tag}) ...")

    depths = np.empty((N, H, W), dtype=np.float32)
    n_masked = 0
    # Accumulate segmentation masks when saving intermediates
    _save_seg = save_dir is not None and seg_model is not None
    seg_masks = np.empty((N, H, W), dtype=bool) if _save_seg else None

    def _gpu_infer(batch_frames):
        """Strategies 1 & 4: batched, optionally concurrent depth + seg."""
        B = len(batch_frames)
        if _concurrent_seg:
            # Strategy 4: depth and seg on separate GPUs in parallel
            with ThreadPoolExecutor(max_workers=2) as pool:
                fd = pool.submit(estimate_depth, depth_model,
                                 batch_frames, device=device,
                                 batch_size=B)
                fs = pool.submit(compute_dynamic_masks, seg_model,
                                 batch_frames, device=_seg_device,
                                 batch_size=B)
                return fd.result(), fs.result()
        d = estimate_depth(depth_model, batch_frames,
                           device=device, batch_size=B)
        m = None
        if seg_model is not None:
            m = compute_dynamic_masks(seg_model, batch_frames,
                                      device=_seg_device, batch_size=B)
        return d, m

    def _cpu_store(b_start, d_batch, m_batch):
        """Strategy 3: store results on CPU thread while GPU runs next batch."""
        nonlocal n_masked
        for i in range(len(d_batch)):
            if m_batch is not None:
                n_masked += int(m_batch[i].sum())
                if seg_masks is not None:
                    seg_masks[b_start + i] = m_batch[i]
                d_batch[i][m_batch[i]] = 0.0
            depths[b_start + i] = d_batch[i]

    # Strategy 3: pipeline — GPU batch k overlaps with CPU store of batch k-1
    with ThreadPoolExecutor(max_workers=1) as cpu_pool:
        pending = None
        for b in range(0, N, batch_size):
            d_batch, m_batch = _gpu_infer(frames[b:min(b + batch_size, N)])
            if pending is not None:
                pending.result()
            pending = cpu_pool.submit(_cpu_store, b, d_batch, m_batch)
        if pending is not None:
            pending.result()

    if seg_model is not None:
        total_px = N * H * W
        print(f"    masked {n_masked:,}/{total_px:,} dynamic pixels "
              f"({100 * n_masked / total_px:.1f}%)")

    # --- Phase 2: Precompute pose matrices ---------------------------------
    T_all = poses_to_matrices(poses)                        # (N, 4, 4)

    # -- save intermediates: depth maps + segmentation + point clouds --
    if save_dir is not None:
        fmt_str = ", ".join(pc_formats)
        print(f"  saving depth images ...")
        save_depth_images(depths, save_dir)
        if seg_masks is not None:
            print(f"  saving segmentation results ...")
            save_segmentation_results(frames, seg_masks, save_dir)
        print(f"  saving per-frame point clouds ({fmt_str}) ...")
        save_pointclouds(depths, K, T_all, save_dir,
                         subsample=depth_subsample,
                         max_depth=max_depth, crop_top=crop_top,
                         pc_formats=pc_formats)
        print(f"  saving merged point cloud ({fmt_str}) ...")
        save_merged_pointcloud(depths, K, T_all, save_dir,
                               subsample=depth_subsample,
                               max_depth=max_depth, crop_top=crop_top,
                               pc_formats=pc_formats)

    # --- Phase 3: Sliding-window elevation map + traversability check ------
    feasible = np.ones(N, dtype=bool)
    windows  = []

    for w_idx, w0 in enumerate(
        range(0, max(1, N - window_size + 1), window_stride)
    ):
        w1 = min(w0 + window_size, N)
        mid = (w0 + w1) // 2

        T_ref_inv = np.linalg.inv(T_all[mid])

        emap = ElevationMap(resolution=grid_res, extent=grid_extent)

        for i in range(w0, w1):
            pc = depth_to_pointcloud(depths[i], K,
                                     subsample=depth_subsample,
                                     max_depth=max_depth,
                                     crop_top_fraction=crop_top)
            if len(pc) == 0:
                continue
            T_i2ref = T_ref_inv @ T_all[i]
            emap.add_points(transform_points(pc, T_i2ref),
                            frame_id=i if temporal_filter else None)

        # Temporal consistency: remove cells with unstable heights
        if temporal_filter:
            emap.filter_unstable_cells(
                max_temporal_std=temporal_std_threshold)

        # trajectory in reference frame
        traj = np.array([
            (T_ref_inv @ T_all[i])[:3, 3] for i in range(w0, w1)
        ])

        stats = emap.query_trajectory(traj, radius_cells=3)
        ok, reason = is_traversable(stats, max_slope_deg, max_step_m,
                                    max_rough_m)

        windows.append({
            "start": int(w0), "end": int(w1),
            "feasible": bool(ok), "reason": reason,
            "max_slope":  _safe_max(stats["max_slope"]),
            "max_step":   _safe_max(stats["max_step"]),
            "max_rough":  _safe_max(stats["max_roughness"]),
        })

        if not ok:
            feasible[w0:w1] = False

        # -- save intermediate: per-window elevation map --
        if save_dir is not None:
            save_elevation_map_data(emap, w_idx, w0, w1, save_dir)

    return feasible, windows


def _safe_max(arr):
    v = np.nanmax(arr) if np.any(np.isfinite(arr)) else None
    return float(v) if v is not None else None


# ---------------------------------------------------------------------------
# Optional visualisation
# ---------------------------------------------------------------------------

def _style_dark_ax(ax, title=""):
    """Apply dark theme to an axes."""
    ax.set_facecolor("black")
    ax.set_title(title, color="white", fontsize=11)
    ax.tick_params(colors="white", which="both")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")


def _dark_colorbar(fig, sc, ax, label=""):
    """Add a colorbar with white labels to a dark-themed axes."""
    cb = fig.colorbar(sc, ax=ax, label=label, shrink=0.7)
    cb.ax.yaxis.set_tick_params(color="white")
    cb.ax.yaxis.label.set_color("white")
    for lbl in cb.ax.get_yticklabels():
        lbl.set_color("white")


def _overlay_trajectory(ax, traj_2d, feasible=None, size=12):
    """Draw trajectory dots (green/red by feasibility) on an axes."""
    if traj_2d is None or len(traj_2d) == 0:
        return
    if feasible is not None and len(feasible) == len(traj_2d):
        c = ["lime" if f else "red" for f in feasible]
    else:
        c = "red"
    ax.scatter(traj_2d[:, 0], traj_2d[:, 1], c=c, s=size, zorder=5,
               edgecolors="white", linewidths=0.3)


def _perspective_project(points, eye, center, up):
    """
    Perspective-project (N,3) points from a virtual camera.

    Returns (x2d, y2d, depth, mask) where mask selects visible points.
    Convention: camera y-axis points up on screen (Y is flipped).
    """
    f = center - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    ns = np.linalg.norm(s)
    if ns < 1e-6:  # up ∥ forward — pick a different up
        s = np.cross(f, np.array([1.0, 0.0, 0.0]))
        ns = np.linalg.norm(s)
    s /= ns
    u = np.cross(s, f)

    # world → camera  (x=right, y=up, z=into screen i.e. forward)
    R = np.array([s, u, f])                          # (3, 3)
    pts_cam = (points - eye) @ R.T                   # (N, 3)

    mask = pts_cam[:, 2] > 0.1
    pc = pts_cam[mask]
    x2d = pc[:, 0] / pc[:, 2]
    y2d = pc[:, 1] / pc[:, 2]                        # +y = up
    depth = pc[:, 2]
    return x2d, y2d, depth, mask


def _render_perspective(ax, points, eye, center, up, color_values,
                        point_size=0.5, trajectory=None, feasible=None):
    """Perspective-projected scatter with painter's algorithm depth sort."""
    x2d, y2d, depth, mask = _perspective_project(points, eye, center, up)
    if len(x2d) == 0:
        return
    c = color_values[mask]

    order = np.argsort(-depth)                       # far → near
    ax.scatter(x2d[order], y2d[order], c=c[order], cmap="plasma",
               s=point_size, marker=".", edgecolors="none", rasterized=True)

    if trajectory is not None and len(trajectory) > 0:
        tx, ty, _, tmask = _perspective_project(trajectory, eye, center, up)
        feas = feasible[tmask] if feasible is not None else None
        _overlay_trajectory(ax, np.column_stack([tx, ty]), feas, size=14)

    ax.set_facecolor("black")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def save_pointcloud_views(points, path, trajectory=None, feasible=None,
                          max_points=200_000):
    """
    Render four 2D projected views of a 3D point cloud and save as PNG.

    Panels:
      1. Bird's-eye view (BEV) — top-down XZ, colored by elevation
      2. Side view — ZY profile, colored by lateral position
      3. CCTV front-left — elevated oblique perspective
      4. CCTV front-right — elevated oblique perspective

    Coordinate convention: X→right, Y→down, Z→forward.
    """
    if len(points) == 0:
        return

    # Subsample for rendering speed
    rng = np.random.default_rng(42)
    if len(points) > max_points:
        idx = rng.choice(len(points), max_points, replace=False)
        pts = points[idx]
    else:
        pts = points

    elev = -pts[:, 1]  # elevation = -Y

    pmin, pmax = pts.min(axis=0), pts.max(axis=0)
    center = (pmin + pmax) / 2.0
    span = (pmax - pmin).max()
    if span < 1e-3:
        return

    up = np.array([0.0, -1.0, 0.0])     # scene up = -Y

    fig, axes = plt.subplots(2, 2, figsize=(16, 14), facecolor="black")

    # ---- BEV (top-down: X vs Z, colour by elevation) ----
    ax = axes[0, 0]
    order = np.argsort(elev)
    sc = ax.scatter(pts[order, 0], pts[order, 2], c=elev[order],
                    cmap="plasma", s=0.5, marker=".",
                    edgecolors="none", rasterized=True)
    _overlay_trajectory(
        ax,
        np.column_stack([trajectory[:, 0], trajectory[:, 2]])
            if trajectory is not None else None,
        feasible,
    )
    _dark_colorbar(fig, sc, ax, label="Elevation (m)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    _style_dark_ax(ax, "Bird's-eye view (top-down)")
    ax.set_aspect("equal")

    # ---- Side view (Z vs elevation, colour by X) ----
    ax = axes[0, 1]
    order = np.argsort(pts[:, 0])
    sc = ax.scatter(pts[order, 2], elev[order], c=pts[order, 0],
                    cmap="coolwarm", s=0.5, marker=".",
                    edgecolors="none", rasterized=True)
    if trajectory is not None and len(trajectory) > 0:
        ax.plot(trajectory[:, 2], -trajectory[:, 1], "cyan", lw=1.5, zorder=5)
    _dark_colorbar(fig, sc, ax, label="X (m)")
    ax.set_xlabel("Z (m, forward)")
    ax.set_ylabel("Elevation (m)")
    _style_dark_ax(ax, "Side view (lateral profile)")
    ax.set_aspect("equal")

    # ---- CCTV front-left ----
    ax = axes[1, 0]
    eye_fl = center + np.array([-0.7, -0.5, -0.3]) * span
    _render_perspective(ax, pts, eye_fl, center, up, elev,
                        trajectory=trajectory, feasible=feasible)
    _style_dark_ax(ax, "CCTV (front-left)")

    # ---- CCTV front-right ----
    ax = axes[1, 1]
    eye_fr = center + np.array([0.7, -0.5, -0.3]) * span
    _render_perspective(ax, pts, eye_fr, center, up, elev,
                        trajectory=trajectory, feasible=feasible)
    _style_dark_ax(ax, "CCTV (front-right)")

    total = len(points)
    shown = len(pts)
    title = f"Point Cloud ({total:,} pts"
    if shown < total:
        title += f", showing {shown:,}"
    title += ")"
    fig.suptitle(title, fontsize=14, color="white")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def save_pointcloud_frame_view(points_cam, path, max_points=100_000):
    """
    Render a single-frame point cloud (camera frame) as a 2-panel image:
      Left  — BEV (X vs Z)
      Right — CCTV from behind-above the camera

    Coordinate convention (camera frame): X→right, Y→down, Z→forward.
    """
    if len(points_cam) == 0:
        return

    rng = np.random.default_rng(42)
    if len(points_cam) > max_points:
        idx = rng.choice(len(points_cam), max_points, replace=False)
        pts = points_cam[idx]
    else:
        pts = points_cam

    elev = -pts[:, 1]
    pmin, pmax = pts.min(axis=0), pts.max(axis=0)
    center = (pmin + pmax) / 2.0
    span = (pmax - pmin).max()
    if span < 1e-3:
        return

    up = np.array([0.0, -1.0, 0.0])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="black")

    # BEV
    ax = axes[0]
    order = np.argsort(elev)
    sc = ax.scatter(pts[order, 0], pts[order, 2], c=elev[order],
                    cmap="plasma", s=0.5, marker=".",
                    edgecolors="none", rasterized=True)
    _dark_colorbar(fig, sc, ax, label="Elevation (m)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    _style_dark_ax(ax, "Bird's-eye view")
    ax.set_aspect("equal")

    # CCTV from behind-above
    ax = axes[1]
    eye = np.array([0.0, -0.6 * span, -0.4 * span]) + center
    _render_perspective(ax, pts, eye, center, up, elev)
    _style_dark_ax(ax, "CCTV (behind-above)")

    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def save_elevation_vis(emap, traj, feasible_mask, path,
                       max_slope_deg=15.0, max_step_m=0.12):
    """Save a 2 x 2 diagnostic figure (elevation, slope, step, roughness)."""
    elev = emap.elevation()
    sl   = emap.slope()
    st   = emap.step_height()
    ro   = emap.roughness()

    extent_m = emap.extent
    ext = [-extent_m, extent_m, -extent_m, extent_m]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    titles = ["Elevation (m)", "Slope (deg)", "Step height (m)", "Roughness (m)"]
    data   = [elev, sl, st, ro]

    for ax, d, t in zip(axes.flat, data, titles):
        im = ax.imshow(d.T, origin="lower", extent=ext, cmap="terrain")
        fig.colorbar(im, ax=ax, shrink=0.7)
        ax.set_title(t)

        # overlay trajectory
        if traj is not None and len(traj) > 0:
            colors = ["green" if f else "red" for f in feasible_mask]
            ax.scatter(traj[:, 0], traj[:, 2], c=colors, s=8, zorder=5)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_episodes_images(data_dir, pose_dir, image_subdir):
    """Discover (name, image_dir, pose_file) tuples in CarlaDataset layout."""
    pose_files = sorted(
        p for p in os.listdir(pose_dir) if p.endswith(".txt")
    )
    episodes = []
    for pf in pose_files:
        name = os.path.splitext(pf)[0]
        img_dir = os.path.join(data_dir, name, image_subdir)
        if os.path.isdir(img_dir):
            episodes.append((name, img_dir, os.path.join(pose_dir, pf)))
    return episodes


def discover_episodes_video(video_dir, pose_dir):
    """Discover (name, video_path, pose_file) tuples in CityWalk layout."""
    pose_files = sorted(
        p for p in os.listdir(pose_dir) if p.endswith(".txt")
    )
    episodes = []
    for pf in pose_files:
        name = os.path.splitext(pf)[0]
        for ext in (".mp4", ".mkv", ".webm"):
            vp = os.path.join(video_dir, name + ext)
            if os.path.isfile(vp):
                episodes.append((name, vp, os.path.join(pose_dir, pf)))
                break
    return episodes


def _process_and_save_episode(name, source, pose_file, *, args, K, pose_step,
                               depth_model, seg_model, device,
                               seg_device=None):
    """Load data, run traversability pipeline, and save results for one episode.

    Returns:
        summary dict, or None if skipped.
    """
    poses, valid_mask = load_poses(pose_file, pose_step=pose_step)

    if args.data_format == "images":
        all_indices = list(range(0, len(os.listdir(source)), pose_step))
        frame_indices = [all_indices[i] for i, v in enumerate(valid_mask)
                         if v and i < len(all_indices)]
        frames = load_frames_images(source, indices=frame_indices)
    else:
        frame_step = args.video_fps // args.target_fps
        frame_indices = [i * frame_step for i, v in enumerate(valid_mask) if v]
        frames = load_frames_video(source, indices=frame_indices)

    n = min(len(frames), len(poses))
    frames, poses = frames[:n], poses[:n]

    if n < args.window_size:
        print(f"  skipping (only {n} frames, need {args.window_size})")
        return None

    episode_save_dir = None
    if args.save_intermediates:
        episode_save_dir = os.path.join(args.output_dir, name)

    feasible, windows = process_episode(
        frames, poses, K, depth_model,
        window_size=args.window_size,
        window_stride=args.window_stride,
        grid_res=args.grid_resolution,
        grid_extent=args.grid_extent,
        depth_subsample=args.depth_subsample,
        crop_top=args.crop_top,
        max_depth=args.max_depth,
        max_slope_deg=args.max_slope,
        max_step_m=args.max_step_height,
        max_rough_m=args.max_roughness,
        device=device,
        save_dir=episode_save_dir,
        pc_formats=tuple(args.pointcloud_format),
        seg_model=seg_model,
        temporal_filter=args.temporal_filter,
        temporal_std_threshold=args.temporal_std_threshold,
        batch_size=args.batch_size,
        seg_device=seg_device,
    )

    feas_pct = 100.0 * feasible.mean()
    print(f"    feasible: {feasible.sum()}/{n} frames ({feas_pct:.1f}%)")

    np.save(os.path.join(args.output_dir, f"{name}_feasibility.npy"), feasible)
    with open(os.path.join(args.output_dir, f"{name}_windows.json"), "w") as f:
        json.dump(windows, f, indent=2)

    summary = {
        "episode": name,
        "total_frames": int(n),
        "feasible_frames": int(feasible.sum()),
        "feasible_pct": round(feas_pct, 2),
        "infeasible_windows": [w for w in windows if not w["feasible"]],
    }

    # Optional diagnostic visualisation
    if args.visualize:
        T_all = poses_to_matrices(poses)
        T_ref_inv = np.linalg.inv(T_all[n // 2])
        depths_all = estimate_depth(depth_model, frames, device=device,
                                    batch_size=args.batch_size)
        if seg_model is not None:
            _sd = seg_device or device
            vis_masks = compute_dynamic_masks(seg_model, frames, device=_sd,
                                             batch_size=args.batch_size)
            for i in range(n):
                depths_all[i][vis_masks[i]] = 0.0
        emap = ElevationMap(resolution=args.grid_resolution,
                            extent=args.grid_extent)
        for i in range(n):
            pc = depth_to_pointcloud(
                depths_all[i], K,
                subsample=args.depth_subsample,
                max_depth=args.max_depth,
                crop_top_fraction=args.crop_top)
            if len(pc) > 0:
                emap.add_points(
                    transform_points(pc, T_ref_inv @ T_all[i]),
                    frame_id=i if args.temporal_filter else None)
        if args.temporal_filter:
            emap.filter_unstable_cells(
                max_temporal_std=args.temporal_std_threshold)

        traj = np.array([(T_ref_inv @ T_all[i])[:3, 3] for i in range(n)])
        vis_path = os.path.join(args.output_dir, f"{name}_elevation.png")
        save_elevation_vis(emap, traj, feasible, vis_path,
                           max_slope_deg=args.max_slope,
                           max_step_m=args.max_step_height)
        print(f"    saved visualisation -> {vis_path}")

    return summary


def _gpu_worker(gpu_id, episode_list, args_ns, K, pose_step, result_queue):
    """Strategy 2: process a subset of episodes on a single GPU.

    Each worker loads its own depth + segmentation models on the assigned GPU,
    processes its episodes sequentially, and puts per-episode summary dicts
    into ``result_queue``.  Sends ``None`` as a done sentinel.
    """
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading models ...")
    depth_model = load_depth_model(args_ns.depth_model, device=device)
    seg_model = None
    if args_ns.semantic_filter:
        seg_model = load_segmentation_model(device=device)

    for idx, (name, source, pose_file) in enumerate(episode_list):
        print(f"\n[GPU {gpu_id}] [{idx + 1}/{len(episode_list)}] {name}")
        result = _process_and_save_episode(
            name, source, pose_file,
            args=args_ns, K=K, pose_step=pose_step,
            depth_model=depth_model, seg_model=seg_model,
            device=device, seg_device=None)
        if result is not None:
            result_queue.put(result)
    result_queue.put(None)  # done sentinel


def main():
    p = argparse.ArgumentParser(
        description="Filter dynamically infeasible data for wheeled robots")

    # --- paths ---
    p.add_argument("--data_dir", type=str, default=None,
                   help="Root dir with episode subdirectories (images mode)")
    p.add_argument("--video_dir", type=str, default=None,
                   help="Directory with .mp4 files (video mode)")
    p.add_argument("--pose_dir", type=str, required=True,
                   help="Directory with per-episode .txt pose files")
    p.add_argument("--output_dir", type=str, default="./traversability_output")

    # --- format ---
    p.add_argument("--data_format", choices=["images", "video"],
                   default="images")
    p.add_argument("--image_subdir", type=str, default="fcam",
                   help="Image sub-folder name inside each episode dir")

    # --- camera ---
    p.add_argument("--fov", type=float, default=90.0,
                   help="Horizontal FOV in degrees (for unknown cameras, "
                        "70–100 is typical for phones / action cams)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)

    # --- pose sampling ---
    p.add_argument("--pose_fps", type=int, default=2)
    p.add_argument("--target_fps", type=int, default=1)
    p.add_argument("--video_fps", type=int, default=30,
                   help="Source video frame rate (only for video mode)")

    # --- depth ---
    p.add_argument("--depth_model", default="zoedepth")
    p.add_argument("--depth_subsample", type=int, default=4,
                   help="Spatial sub-sampling for point clouds")
    p.add_argument("--crop_top", type=float, default=0.3,
                   help="Fraction of image rows to discard from the top "
                        "(removes sky / buildings)")
    p.add_argument("--max_depth", type=float, default=20.0,
                   help="Ignore depth beyond this distance (metres)")

    # --- elevation map ---
    p.add_argument("--grid_resolution", type=float, default=0.05,
                   help="Grid cell size (metres)")
    p.add_argument("--grid_extent", type=float, default=15.0,
                   help="Half-width of elevation map (metres)")

    # --- sliding window ---
    p.add_argument("--window_size", type=int, default=15)
    p.add_argument("--window_stride", type=int, default=5)

    # --- traversability thresholds ---
    p.add_argument("--max_slope", type=float, default=15.0,
                   help="Max traversable slope (degrees)")
    p.add_argument("--max_step_height", type=float, default=0.12,
                   help="Max traversable step height (metres); "
                        "standard stair riser ≈ 0.18 m")
    p.add_argument("--max_roughness", type=float, default=0.08,
                   help="Max acceptable height std-dev (metres)")

    # --- dynamic object filtering ---
    p.add_argument("--semantic_filter", action="store_true",
                   help="Use DeepLabV3 semantic segmentation to mask out "
                        "dynamic objects (pedestrians, cars, etc.) in depth "
                        "maps before backprojection")
    p.add_argument("--temporal_filter", action="store_true",
                   help="Filter elevation map cells where per-frame height "
                        "std is high (indicates dynamic objects)")
    p.add_argument("--temporal_std_threshold", type=float, default=0.15,
                   help="Height std threshold (metres) for temporal "
                        "consistency filter (default: 0.15)")

    # --- parallelism ---
    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size for GPU inference — depth estimation and "
                        "segmentation (default: 8; tune to fit VRAM)")
    p.add_argument("--num_gpus", type=int, default=1,
                   help="Number of GPUs for episode-level parallelism "
                        "(0 = auto-detect all available, 1 = single-GPU)")
    p.add_argument("--seg_device", type=str, default=None,
                   help="Device for segmentation model to enable concurrent "
                        "depth + seg (strategy 4, e.g. 'cuda:1'). "
                        "Only used in single-GPU mode.")

    # --- misc ---
    p.add_argument("--device", default="cuda")
    p.add_argument("--visualize", action="store_true",
                   help="Save elevation-map diagnostic plots")
    p.add_argument("--save_intermediates", action="store_true",
                   help="Save intermediate results (depth images, "
                        "point clouds, elevation maps) per episode")
    p.add_argument("--pointcloud_format", nargs="+", default=["npz"],
                   choices=["npz", "pcd", "ply"],
                   help="Point cloud output format(s) (default: npz). "
                        "pcd/ply can be opened in CloudCompare, Open3D, etc.")

    args = p.parse_args()

    # ----- validation -----
    if args.data_format == "images" and args.data_dir is None:
        p.error("--data_dir required for images mode")
    if args.data_format == "video" and args.video_dir is None:
        p.error("--video_dir required for video mode")

    os.makedirs(args.output_dir, exist_ok=True)

    # ----- discover episodes -----
    if args.data_format == "images":
        episodes = discover_episodes_images(
            args.data_dir, args.pose_dir, args.image_subdir)
    else:
        episodes = discover_episodes_video(args.video_dir, args.pose_dir)

    print(f"Found {len(episodes)} episodes")
    if len(episodes) == 0:
        return

    # ----- camera intrinsics -----
    K = build_intrinsic_matrix(args.width, args.height, args.fov)
    pose_step = max(1, args.pose_fps // args.target_fps)

    # ----- resolve GPU count -----
    num_gpus = args.num_gpus
    if num_gpus == 0:
        num_gpus = torch.cuda.device_count()

    # ----- Strategy 2: Multi-GPU episode parallelism -----------------------
    if num_gpus > 1 and len(episodes) > 1:
        print(f"Using {num_gpus} GPUs for episode-level parallelism")
        import torch.multiprocessing as mp
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()

        # Round-robin partition of episodes across GPUs
        partitions = [[] for _ in range(num_gpus)]
        for i, ep in enumerate(episodes):
            partitions[i % num_gpus].append(ep)

        processes = []
        for gpu_id in range(num_gpus):
            if not partitions[gpu_id]:
                continue
            proc = ctx.Process(
                target=_gpu_worker,
                args=(gpu_id, partitions[gpu_id], args,
                      K, pose_step, result_queue))
            proc.start()
            processes.append(proc)

        # Collect results as workers finish
        summary = []
        n_done = 0
        n_workers = len(processes)
        while n_done < n_workers:
            result = result_queue.get()
            if result is None:
                n_done += 1
            else:
                summary.append(result)

        for proc in processes:
            proc.join()

    # ----- Single-GPU with strategies 1, 3, 4 -----------------------------
    else:
        print("Loading depth model ...")
        depth_model = load_depth_model(args.depth_model, device=args.device)

        seg_model = None
        _seg_dev = args.seg_device or args.device
        if args.semantic_filter:
            print(f"Loading segmentation model (DeepLabV3) on {_seg_dev} ...")
            seg_model = load_segmentation_model(device=_seg_dev)

        summary = []
        for idx, (name, source, pose_file) in enumerate(episodes):
            print(f"\n[{idx + 1}/{len(episodes)}] {name}")
            result = _process_and_save_episode(
                name, source, pose_file,
                args=args, K=K, pose_step=pose_step,
                depth_model=depth_model, seg_model=seg_model,
                device=args.device, seg_device=args.seg_device)
            if result is not None:
                summary.append(result)

    # global summary
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    total = sum(s["total_frames"] for s in summary)
    total_ok = sum(s["feasible_frames"] for s in summary)
    print(f"\n{'='*60}")
    print(f"Total: {total_ok}/{total} feasible frames "
          f"({100*total_ok/total:.1f}%)" if total > 0 else "No data processed")
    print(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
