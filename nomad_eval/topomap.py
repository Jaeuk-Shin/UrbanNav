"""
Build a NoMaD topomap from a CARLA global route.

Given start and goal transforms (UE coords), this module:
  1. Queries CARLA's built-in GlobalRoutePlanner (via
     ``rl.navigation.global_planner.CarlaGlobalPlanner``) for a dense
     ``(N, 2)`` waypoint array in standard (x_std, z_std) coords.
  2. Re-samples the route at a minimum metric spacing.
  3. Teleports the ego walker to each sub-goal pose (with yaw tangent
     to the path) and captures a forward-camera RGB frame per node.

The resulting list of PIL images is what NoMaD expects as its topomap.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image as PILImage

from rl.navigation.global_planner import CarlaGlobalPlanner, Route

from .env import NomadCarlaEnv, ue_xy_to_std, std_to_ue_xy


# ── Route re-sampling ────────────────────────────────────────────────────
def resample_route(route_xz: np.ndarray,
                   min_spacing_m: float) -> np.ndarray:
    """Down-sample a dense ``(N, 2)`` route to ≥ ``min_spacing_m`` between
    consecutive kept waypoints.  Always keeps the first and last points.
    """
    if len(route_xz) <= 1:
        return route_xz.copy()
    kept = [route_xz[0]]
    last = route_xz[0]
    for pt in route_xz[1:-1]:
        if np.linalg.norm(pt - last) >= min_spacing_m:
            kept.append(pt)
            last = pt
    # Ensure the final goal node is always included
    if not np.allclose(kept[-1], route_xz[-1]):
        kept.append(route_xz[-1])
    return np.asarray(kept, dtype=np.float64)


def _yaw_deg_along_path(path_std: np.ndarray, i: int) -> float:
    """Yaw (UE degrees) along the path direction at index ``i``.

    Uses the segment ``i → i+1`` when possible; otherwise ``i-1 → i``.
    Converts from standard coords (x_std=UE_y, z_std=UE_x) to UE yaw, where
    yaw 0° points along +UE_x.
    """
    if i + 1 < len(path_std):
        a, b = path_std[i], path_std[i + 1]
    elif i > 0:
        a, b = path_std[i - 1], path_std[i]
    else:
        return 0.0
    # Path tangent in standard coords
    dx_std = b[0] - a[0]    # change in x_std == change in UE_y
    dz_std = b[1] - a[1]    # change in z_std == change in UE_x
    # UE yaw = atan2(ΔUE_y, ΔUE_x) = atan2(dx_std, dz_std)
    return math.degrees(math.atan2(dx_std, dz_std))


# ── Topomap generation ──────────────────────────────────────────────────
def plan_route_ue(env: NomadCarlaEnv,
                  start_ue: Tuple[float, float, float],
                  goal_ue: Tuple[float, float, float],
                  resolution_m: float = 2.0) -> Route:
    """Plan a route from start to goal using CARLA's GRP."""
    planner = CarlaGlobalPlanner(env.world, resolution=resolution_m)
    start_xz = ue_xy_to_std(float(start_ue[0]), float(start_ue[1]))
    goal_xz = ue_xy_to_std(float(goal_ue[0]), float(goal_ue[1]))
    return planner.plan(start_xz, goal_xz)


def build_topomap(
    env: NomadCarlaEnv,
    route: Route,
    min_spacing_m: float = 3.0,
    include_start: bool = True,
    drain_ticks: int = 1,
    sensor_id: str = 'fcam',
    ground_z: Optional[float] = None,
) -> Tuple[List[PILImage.Image], np.ndarray]:
    """Build a list of sub-goal PIL images by teleport-rendering along the
    planned route.

    Parameters
    ----------
    env : NomadCarlaEnv
        Already connected and reset (so the walker + camera exist).  The
        walker is temporarily teleported to each sub-goal pose; the caller
        is responsible for restoring it to the desired start pose afterward
        via ``env.teleport_robot_ue(...)`` (or another ``env.reset``).
    route : Route
        Output of ``plan_route_ue``; waypoints in standard (x_std, z_std).
    min_spacing_m : float
        Minimum metric distance between kept sub-goal nodes.
    include_start : bool
        If False, drop the first node (e.g. when the start image is
        already in the context queue at runtime).
    drain_ticks : int
        Extra world ticks per teleport before reading the camera — one is
        usually enough to let the camera re-render.
    ground_z : float, optional
        UE z to place the walker at each subgoal (to avoid floating).  If
        None, keeps the current z each time.

    Returns
    -------
    images : list of PIL.Image
    nodes_std : (M, 2) np.ndarray
        Standard (x_std, z_std) coordinates of each captured sub-goal.
    """
    nodes_std = resample_route(route.waypoints, min_spacing_m)
    if not include_start and len(nodes_std) > 1:
        nodes_std = nodes_std[1:]

    images: List[PILImage.Image] = []
    for i in range(len(nodes_std)):
        ue_xy = std_to_ue_xy(nodes_std[i])
        yaw_deg = _yaw_deg_along_path(nodes_std, i)
        env.teleport_robot_ue(ue_xy, yaw_deg=yaw_deg, ground_z=ground_z)
        # Drain stale pre-teleport frames, then tick to render at the new pose.
        env._drain_queues()
        for _ in range(max(1, drain_ticks)):
            env.world.tick()
        rgb = env._get_sensor_data(sensor_id)
        images.append(PILImage.fromarray(rgb))

    return images, nodes_std


def save_topomap(images: List[PILImage.Image],
                 out_dir: str) -> None:
    """Save images to ``<out_dir>/<i>.jpg`` for later inspection."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(images):
        img.save(f"{out_dir}/{i}.jpg")
