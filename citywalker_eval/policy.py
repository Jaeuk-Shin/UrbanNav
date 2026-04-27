"""CityWalker policy wrapper for closed-loop inference in CARLA.

The CityWalker model consumes:
    obs  : (B, N, 3, H, W)     — RGB history (float or uint8, see ``do_rgb_normalize``)
    cord : (B, N + 1, 2)       — N ego-relative past positions + 1 subgoal, all
                                  scaled by a ``step_scale`` normalizer

It returns:
    wp_pred    : (B, T, 2)     — cumulative ego-frame waypoints (normalized)
    arrive_pred: (B, 1)        — arrival logits

This module adapts the raw CarlaMultiAgentEnv observations to that contract and
converts the output back to the camera-frame metric waypoints expected by the
env's action space.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


# ── CityWalker import helper ──────────────────────────────────────────


def _ensure_citywalker_on_path(citywalker_root: str | Path) -> None:
    """Put the CityWalker repo on ``sys.path`` so ``model.*`` / ``pl_modules.*`` import."""
    root = str(Path(citywalker_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


# ── Pose helpers (replicated from CityWalker's CityWalkDataset) ───────


def _pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = R.from_quat(pose[3:]).as_matrix()
    m[:3, 3] = pose[:3]
    return m


def _transform_positions_to_current(poses: np.ndarray, current: np.ndarray) -> np.ndarray:
    """(N, 7) poses → (N, 3) positions in current pose's frame."""
    cur_inv = np.linalg.inv(_pose_to_matrix(current))
    mats = np.tile(np.eye(4), (poses.shape[0], 1, 1))
    mats[:, :3, :3] = R.from_quat(poses[:, 3:]).as_matrix()
    mats[:, :3, 3] = poses[:, :3]
    transformed = np.matmul(cur_inv[np.newaxis], mats)
    return transformed[:, :3, 3]


def _transform_point_to_current(xyz: np.ndarray, current: np.ndarray) -> np.ndarray:
    cur_inv = np.linalg.inv(_pose_to_matrix(current))
    p = np.append(np.asarray(xyz, dtype=np.float64), 1.0)
    return (cur_inv @ p)[:3]


# ── Policy wrapper ────────────────────────────────────────────────────


class CityWalkerPolicy:
    """Thin inference-time wrapper around a trained CityWalker checkpoint.

    Parameters
    ----------
    checkpoint : path to a Lightning ``.ckpt`` file
    cfg        : the YAML config the checkpoint was trained with (as an
                 attribute-style object, e.g. OmegaConf DictConfig)
    citywalker_root : path to the CityWalker repo (so its ``model`` /
                      ``pl_modules`` packages are importable)
    step_scale : metric normalization factor for cord/wp coordinates
                 (typical pedestrian value at 1 Hz ≈ 1.0)
    device     : torch device
    module_type: override of ``cfg.model.type``; by default picks
                 ``CityWalkerFeatModule`` for ``citywalker_feat`` otherwise
                 ``CityWalkerModule``
    """

    def __init__(
        self,
        checkpoint: str | Path,
        cfg,
        citywalker_root: str | Path,
        step_scale: float = 1.0,
        device: str = "cuda",
        module_type: Optional[str] = None,
    ) -> None:
        _ensure_citywalker_on_path(citywalker_root)

        # Lazy import so merely importing this module doesn't require the
        # CityWalker repo to be present.
        pl_feat = importlib.import_module("pl_modules.citywalker_feat_module")
        pl_std = importlib.import_module("pl_modules.citywalker_module")

        model_type = module_type or getattr(cfg.model, "type", "")
        if "feat" in str(model_type):
            ModuleCls = pl_feat.CityWalkerFeatModule
        else:
            ModuleCls = pl_std.CityWalkerModule

        self.module_cls_name = ModuleCls.__name__
        self.device = torch.device(device)
        self.cfg = cfg
        self.step_scale = float(step_scale)
        self.context_size = int(cfg.model.obs_encoder.context_size)
        self.len_traj_pred = int(cfg.model.decoder.len_traj_pred)
        self.output_coord = getattr(cfg.model, "output_coordinate_repr", "euclidean")

        self._module = ModuleCls.load_from_checkpoint(
            str(checkpoint), cfg=cfg, map_location=self.device
        )
        self._module.eval().to(self.device)

    # ── input preparation ─────────────────────────────────────────────

    def _build_cord(
        self,
        pose_history: np.ndarray,   # (N, 7) std-frame poses
        subgoal_std: np.ndarray,    # (2,)
    ) -> torch.Tensor:
        """Build the ``(1, N + 1, 2)`` cord tensor expected by CityWalker.

        Matches ``CityWalkDataset.__getitem__`` (cord_embedding.type ==
        ``input_target``) with ``current_pose = pose_history[-1]``.
        """
        assert pose_history.shape[0] == self.context_size, (
            f"pose_history must have {self.context_size} rows, got {pose_history.shape[0]}"
        )
        current = pose_history[-1]
        input_positions = _transform_positions_to_current(pose_history, current)  # (N, 3)
        subgoal_xyz = np.array([subgoal_std[0], 0.0, subgoal_std[1]], dtype=np.float64)
        subgoal_in_cur = _transform_point_to_current(subgoal_xyz, current)         # (3,)

        # Keep only [x, z] components.
        coords = np.concatenate(
            [input_positions[:, [0, 2]], subgoal_in_cur[[0, 2]][None, :]], axis=0
        )
        coords = coords / max(self.step_scale, 1e-2)
        return torch.from_numpy(coords).float().unsqueeze(0).to(self.device)

    def _build_obs(self, rgb_history: np.ndarray) -> torch.Tensor:
        """(N, H, W, 3) uint8 → (1, N, 3, H, W) float32 in [0, 1]."""
        assert rgb_history.shape[0] == self.context_size
        x = torch.from_numpy(rgb_history).float() / 255.0  # (N, H, W, 3)
        x = x.permute(0, 3, 1, 2).contiguous()             # (N, 3, H, W)
        return x.unsqueeze(0).to(self.device)              # (1, N, 3, H, W)

    # ── inference ─────────────────────────────────────────────────────

    @torch.no_grad()
    def act(
        self,
        rgb_history: np.ndarray,    # (N, H, W, 3) uint8
        pose_history: np.ndarray,   # (N, 7) std frame camera-to-world poses
        subgoal_std: np.ndarray,    # (2,) subgoal in std (x, z) world coords
    ) -> dict:
        """Run one forward pass and return env-ready waypoints.

        Returns
        -------
        dict with keys
            waypoints_cam : (T, 2) camera-frame waypoints in metres (cumulative)
            action        : (T * 2,) flattened version — matches the env's
                            per-agent action slot
            arrive_prob   : float
            wp_normalized : (T, 2) raw model output prior to step_scale
        """
        obs = self._build_obs(rgb_history)
        cord = self._build_cord(pose_history, subgoal_std)

        out = self._module.model(obs, cord)
        # Both model variants return wp_pred first and arrive_pred second.
        wp_pred = out[0]           # (1, T, 2)
        arrive_pred = out[1]       # (1, 1)

        wp_norm = wp_pred[0].detach().cpu().numpy()
        wp_cam = wp_norm * self.step_scale       # metres, camera frame
        action = wp_cam.reshape(-1).astype(np.float32)
        arrive_prob = float(torch.sigmoid(arrive_pred[0, 0]).item())
        return {
            "waypoints_cam": wp_cam.astype(np.float32),
            "action": action,
            "arrive_prob": arrive_prob,
            "wp_normalized": wp_norm.astype(np.float32),
        }
