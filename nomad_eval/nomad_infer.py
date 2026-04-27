"""
ROS-free NoMaD policy wrapper.

Ports the core inference logic from
``visualnav-transformer/deployment/src/navigate.py`` so NoMaD can be driven
from a CARLA loop (or anything else) without rospy.

Usage
-----
    policy = NomadPolicy(ckpt_path, config_path, device)
    policy.reset(topomap)            # list[PIL.Image]
    policy.push_obs(rgb_pil)
    ...
    waypoints, info = policy.infer()   # (T, 2) displacements in robot frame
                                       # x = forward, y = left (meters, if denormalized)
"""

from __future__ import annotations

import os
import sys
import pathlib
import time
from collections import deque
from typing import List, Optional

import numpy as np
import torch
import yaml
from PIL import Image as PILImage
from torchvision import transforms
import torchvision.transforms.functional as TF

# visualnav-transformer model classes expect their package to be importable.
# The user installs it with `pip install -e visualnav-transformer/train/`.
# If that hasn't been done, add the train/ path so the imports still resolve.
_VNT_ROOT = pathlib.Path(
    os.environ.get('VISUALNAV_TRANSFORMER_ROOT',
                   '/home3/rvl/baselines/visualnav-transformer')
)
_VNT_TRAIN = _VNT_ROOT / 'train'
if _VNT_TRAIN.is_dir() and str(_VNT_TRAIN) not in sys.path:
    sys.path.insert(0, str(_VNT_TRAIN))

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # noqa: E402
from diffusion_policy.model.diffusion.conditional_unet1d import (  # noqa: E402
    ConditionalUnet1D,
)
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork  # noqa: E402
from vint_train.models.nomad.nomad_vint import (  # noqa: E402
    NoMaD_ViNT, replace_bn_with_gn,
)


# Action normalization stats used at training time (from
# ``train/vint_train/data/data_config.yaml``).  Replicated here to avoid
# importing ``vint_train.training.train_utils`` — that module imports wandb
# and the full training stack, which we don't need for inference.
_DATA_CONFIG_PATH = _VNT_TRAIN / 'vint_train' / 'data' / 'data_config.yaml'
if _DATA_CONFIG_PATH.is_file():
    with open(_DATA_CONFIG_PATH, 'r') as f:
        _data_cfg = yaml.safe_load(f)
    ACTION_STATS = {
        k: np.asarray(v, dtype=np.float64)
        for k, v in _data_cfg['action_stats'].items()
    }
else:
    ACTION_STATS = {
        'min': np.array([-2.5, -4.0]),
        'max': np.array([5.0, 4.0]),
    }


# ── Image preprocessing (mirrors deployment/src/utils.py) ────────────────
_IMAGE_ASPECT_RATIO = 4.0 / 3.0
_IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
_TO_TENSOR = transforms.Compose([transforms.ToTensor(), _IMAGENET_NORMALIZE])


def transform_images(
    pil_imgs, image_size, center_crop: bool = False,
) -> torch.Tensor:
    """Convert list of PIL images → (1, 3*N, H, W) torch tensor."""
    if not isinstance(pil_imgs, list):
        pil_imgs = [pil_imgs]
    out = []
    for img in pil_imgs:
        w, h = img.size
        if center_crop:
            if w > h:
                img = TF.center_crop(img, (h, int(h * _IMAGE_ASPECT_RATIO)))
            else:
                img = TF.center_crop(img, (int(w / _IMAGE_ASPECT_RATIO), w))
        img = img.resize(image_size)
        t = _TO_TENSOR(img).unsqueeze(0)   # (1, 3, H, W)
        out.append(t)
    return torch.cat(out, dim=1)           # (1, 3*N, H, W)


def _unnormalize_deltas(ndeltas: np.ndarray) -> np.ndarray:
    """[-1, 1] → action-space deltas using ACTION_STATS."""
    mn, mx = ACTION_STATS['min'], ACTION_STATS['max']
    d = (ndeltas + 1.0) / 2.0
    return d * (mx - mn) + mn


def _get_action_np(diffusion_output: torch.Tensor) -> np.ndarray:
    """Diffusion output (B, T, 2) → waypoints (B, T, 2) via cumsum."""
    ndeltas = diffusion_output.detach().cpu().numpy()
    deltas = _unnormalize_deltas(ndeltas)
    return np.cumsum(deltas, axis=1)


# ── Model loading (ROS-free port of deployment/src/utils.load_model) ─────
def load_nomad(ckpt_path: str, config: dict,
               device: torch.device) -> torch.nn.Module:
    assert config['model_type'] == 'nomad', (
        f"Only NoMaD is supported here; got {config['model_type']}")
    if config['vision_encoder'] == 'nomad_vint':
        vision_encoder = NoMaD_ViNT(
            obs_encoding_size=config['encoding_size'],
            context_size=config['context_size'],
            mha_num_attention_heads=config['mha_num_attention_heads'],
            mha_num_attention_layers=config['mha_num_attention_layers'],
            mha_ff_dim_factor=config['mha_ff_dim_factor'],
        )
        vision_encoder = replace_bn_with_gn(vision_encoder)
    else:
        raise ValueError(
            f"Vision encoder {config['vision_encoder']} not supported")

    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=config['encoding_size'],
        down_dims=config['down_dims'],
        cond_predict_scale=config['cond_predict_scale'],
    )
    dist_pred_network = DenseNetwork(embedding_dim=config['encoding_size'])

    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
    )

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


# ── Policy wrapper ───────────────────────────────────────────────────────
class NomadPolicy:
    """Stateful NoMaD navigation policy for topomap-following.

    Mirrors the control flow in ``navigate.py`` but without ROS:
      * ``reset(topomap)``        — install topomap, clear context queue
      * ``push_obs(pil_img)``     — append a new RGB observation
      * ``infer()``               — run model (requires full context);
                                    returns ``(waypoints, info)``

    The waypoints are in NoMaD's local frame (x=forward, y=left), either
    raw cumulative deltas (if ``normalize=False`` in the config) or
    denormalized by the action stats.  Callers apply a
    ``velocity_scale = MAX_V / RATE`` factor when the model was trained
    with ``normalize=True`` (see navigate.py).
    """

    def __init__(
        self,
        ckpt_path: str,
        config_path: str,
        device: Optional[torch.device] = None,
        radius: int = 4,
        close_threshold: int = 3,
        num_samples: int = 8,
        waypoint_idx: int = 2,
    ):
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"NoMaD checkpoint not found: {ckpt_path}\n"
                "Download nomad.pth from the visualnav-transformer release "
                "(see its README) and point `ckpt_path` at it.")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"NoMaD model config not found: {config_path}")

        with open(config_path, 'r') as f:
            self.model_params = yaml.safe_load(f)

        self.context_size = int(self.model_params['context_size'])
        self.image_size = tuple(self.model_params['image_size'])
        self.len_traj_pred = int(self.model_params['len_traj_pred'])
        self.num_diffusion_iters = int(
            self.model_params['num_diffusion_iters'])
        self.normalize_actions = bool(self.model_params.get('normalize', True))

        self.model = load_nomad(ckpt_path, self.model_params, self.device)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon',
        )

        # Navigation hyper-params
        self.radius = int(radius)
        self.close_threshold = int(close_threshold)
        self.num_samples = int(num_samples)
        self.waypoint_idx = int(waypoint_idx)

        # Per-episode state
        self._queue: deque = deque(maxlen=self.context_size + 1)
        self._topomap_tensors: Optional[List[torch.Tensor]] = None
        self._closest_node = 0
        self._goal_node = 0

    # ── Episode management ──────────────────────────────────────────────
    def reset(self, topomap: List[PILImage.Image], goal_node: int = -1):
        """Install a topomap (list of PIL images) for the new episode."""
        assert len(topomap) >= 1, "Topomap must have ≥1 sub-goal image"
        self._queue.clear()

        # Pre-transform topomap once; each entry is (1, 3, H, W)
        self._topomap_tensors = [
            transform_images(g, self.image_size, center_crop=False).to(self.device)
            for g in topomap
        ]
        self._closest_node = 0
        self._goal_node = (len(self._topomap_tensors) - 1
                           if goal_node == -1 else int(goal_node))
        assert 0 <= self._goal_node < len(self._topomap_tensors)

    def push_obs(self, pil_img: PILImage.Image):
        self._queue.append(pil_img)

    @property
    def is_ready(self) -> bool:
        return len(self._queue) == self._queue.maxlen

    @property
    def closest_node(self) -> int:
        return self._closest_node

    @property
    def goal_node(self) -> int:
        return self._goal_node

    @property
    def reached_goal(self) -> bool:
        return self._closest_node >= self._goal_node

    # ── Inference ───────────────────────────────────────────────────────
    @torch.no_grad()
    def infer(self) -> tuple[np.ndarray, dict]:
        """Run NoMaD on the current context + topomap.

        Returns
        -------
        waypoints : (T, 2) np.float32
            Predicted trajectory in NoMaD's local frame (x=forward, y=left),
            un-normalized action units from the first diffusion sample.
        info : dict
            Diagnostic info: closest_node, goal_distance, diffusion_time_s,
            all_samples ((S, T, 2)).
        """
        assert self.is_ready, "Context queue not full; call push_obs first"
        assert self._topomap_tensors is not None, "Call reset(topomap) first"

        mp = self.model_params

        obs_images = transform_images(
            list(self._queue), self.image_size, center_crop=False,
        )
        # Mirror navigate.py's split+cat (no-op but matches reference)
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1).to(self.device)

        # Temporal localization window around the current closest node
        start = max(self._closest_node - self.radius, 0)
        end = min(self._closest_node + self.radius + 1, self._goal_node)
        if end <= start:
            end = start + 1
        goal_batch = torch.cat(
            self._topomap_tensors[start:end + 1], dim=0)   # (K, 3, H, W)
        k = goal_batch.shape[0]

        obs_batch = obs_images.repeat(k, 1, 1, 1)
        mask = torch.zeros(k, dtype=torch.long, device=self.device)

        obsgoal_cond = self.model(
            'vision_encoder',
            obs_img=obs_batch,
            goal_img=goal_batch,
            input_goal_mask=mask,
        )

        dists = self.model(
            'dist_pred_net', obsgoal_cond=obsgoal_cond,
        ).flatten().detach().cpu().numpy()

        min_idx = int(np.argmin(dists))
        self._closest_node = min_idx + start

        # Advance one node ahead if we're close enough
        sg_idx = min(
            min_idx + int(dists[min_idx] < self.close_threshold),
            len(obsgoal_cond) - 1,
        )
        obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)

        if obs_cond.ndim == 2:
            obs_cond = obs_cond.repeat(self.num_samples, 1)
        else:
            obs_cond = obs_cond.repeat(self.num_samples, 1, 1)

        # Diffusion sampling
        noisy_action = torch.randn(
            (self.num_samples, self.len_traj_pred, 2), device=self.device)
        naction = noisy_action
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

        t0 = time.time()
        for k_t in self.noise_scheduler.timesteps[:]:
            noise_pred = self.model(
                'noise_pred_net',
                sample=naction,
                timestep=k_t,
                global_cond=obs_cond,
            )
            naction = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k_t, sample=naction,
            ).prev_sample
        diffusion_s = time.time() - t0

        all_waypoints = _get_action_np(naction)       # (S, T, 2)
        waypoints = all_waypoints[0].astype(np.float32)

        info = {
            'closest_node': self._closest_node,
            'goal_node': self._goal_node,
            'dist_to_subgoal': float(dists[min_idx]),
            'diffusion_time_s': float(diffusion_s),
            'all_samples': all_waypoints.astype(np.float32),
            'subgoal_idx_used': sg_idx + start,
        }
        return waypoints, info
