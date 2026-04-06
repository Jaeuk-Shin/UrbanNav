#!/usr/bin/env python3
"""Offline scenario precomputation for UrbanNav RL training.

Generates (start, goal, obstacles, distance_field) tuples for each town
and quadrant, eliminating runtime Dijkstra computation during training.

Usage:
    python generate_scenarios.py \
        --navmesh_dir navmeshes/ \
        --output_dir rl/scenarios/ \
        --towns Town02 Town03 Town05 Town10HD \
        --scenarios_per_quadrant 5 \
        --goal_range 40.0 \
        --obstacles \
        --p_crosswalk_challenge 0.5 \
        --resolution 1.0 \
        --num_agents 4
"""
import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from rl.envs.obstacle_manager import (
    BARRIER_PROP_BPS,
    BLOCKER_VEHICLE_BPS,
    BLOCKER_VEHICLE_FALLBACKS,
    CLUTTER_PROP_BPS,
    ObstacleConfig,
    _crosswalk_axes,
    _crosswalk_centre,
    _crosswalk_heading,
    _crosswalks_in_region,
)
from rl.utils.blueprint_dims import get_half_extents
from rl.utils.geodesic import GeodesicDistanceField
from rl.utils.navmesh_cache import NavmeshCache


# ── Data structures ──────────────────────────────────────────────────


@dataclass
class ObstacleSpec:
    """One obstacle's placement parameters (CARLA-independent)."""
    blueprint_id: str
    position_ue: np.ndarray       # (3,) float: (ue_x, ue_y, ue_z)
    yaw_deg: float
    scenario_type: str
    corners_std: np.ndarray       # (4, 2) float: OBB in standard coords


# ── OBB computation ──────────────────────────────────────────────────


def compute_obb_corners_std(
    pos_ue_x: float, pos_ue_y: float,
    yaw_deg: float,
    half_ex: float, half_ey: float,
) -> np.ndarray:
    """Compute (4, 2) OBB corners in standard coords.

    Replicates the OBB computation from obstacle_manager.py
    ``get_obstacle_layout()`` (lines 815-836).

    Parameters
    ----------
    pos_ue_x, pos_ue_y : actor world position in UE frame
    yaw_deg : actor yaw in degrees (CARLA convention)
    half_ex, half_ey : bounding-box half-extents in actor-local frame
    """
    # bb.location is (0, 0) for spawned actors
    local_corners = [
        ( half_ex,  half_ey),
        ( half_ex, -half_ey),
        (-half_ex, -half_ey),
        (-half_ex,  half_ey),
    ]
    yaw_rad = math.radians(yaw_deg)
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)

    corners_std = np.empty((4, 2), dtype=np.float64)
    for i, (lx, ly) in enumerate(local_corners):
        wx = pos_ue_x + lx * cos_y - ly * sin_y
        wy = pos_ue_y + lx * sin_y + ly * cos_y
        # UE -> standard: x_std = ue_y, z_std = ue_x
        corners_std[i, 0] = wy
        corners_std[i, 1] = wx
    return corners_std


def _make_obstacle_spec(
    bp_id: str, pos_ue: np.ndarray, yaw_deg: float, scenario_type: str,
) -> ObstacleSpec:
    """Build an ObstacleSpec with auto-computed OBB corners."""
    half_ex, half_ey = get_half_extents(bp_id)
    corners = compute_obb_corners_std(
        float(pos_ue[0]), float(pos_ue[1]), yaw_deg, half_ex, half_ey)
    return ObstacleSpec(
        blueprint_id=bp_id,
        position_ue=pos_ue.astype(np.float32),
        yaw_deg=yaw_deg,
        scenario_type=scenario_type,
        corners_std=corners,
    )


# ── Navmesh helpers ──────────────────────────────────────────────────


def snap_to_navmesh_ue(
    target_xy: np.ndarray,
    sidewalk_pts_ue: np.ndarray,
    snap_radius: float = 5.0,
) -> np.ndarray | None:
    """Find nearest sidewalk point to target_xy, return (3,) UE or None.

    Replaces ``ObstacleManager._snap_to_navmesh`` without CARLA.
    """
    if sidewalk_pts_ue is None or len(sidewalk_pts_ue) == 0:
        return None
    dists = np.linalg.norm(sidewalk_pts_ue[:, :2] - target_xy, axis=1)
    idx = int(np.argmin(dists))
    if dists[idx] > snap_radius:
        return None
    return sidewalk_pts_ue[idx].copy()


def sample_sidewalk_in_bounds(
    sidewalk_pts_ue: np.ndarray,
    bounds: Tuple[float, float, float, float],
) -> np.ndarray | None:
    """Sample a random sidewalk point within UE bounds, return (3,) or None."""
    xlo, xhi, ylo, yhi = bounds
    mask = (
        (sidewalk_pts_ue[:, 0] >= xlo) & (sidewalk_pts_ue[:, 0] <= xhi) &
        (sidewalk_pts_ue[:, 1] >= ylo) & (sidewalk_pts_ue[:, 1] <= yhi)
    )
    bounded = sidewalk_pts_ue[mask]
    if len(bounded) == 0:
        return None
    return bounded[np.random.randint(len(bounded))].copy()


# ── Quadrant splitting ───────────────────────────────────────────────


def compute_quadrant_bounds(
    walkable_points_ue: np.ndarray,
    num_agents: int = 4,
    quadrant_margin: float = 8.0,
) -> Tuple[List[Tuple], List[Tuple], Tuple[float, float]]:
    """Replicate ``CarlaMultiAgentEnv._compute_regions()`` quadrant logic.

    Returns (quadrant_bounds, quadrant_inner_bounds, (center_x, center_y)).
    """
    center_x = float(np.median(walkable_points_ue[:, 0]))
    center_y = float(np.median(walkable_points_ue[:, 1]))
    x_min = float(walkable_points_ue[:, 0].min()) - 1.0
    x_max = float(walkable_points_ue[:, 0].max()) + 1.0
    y_min = float(walkable_points_ue[:, 1].min()) - 1.0
    y_max = float(walkable_points_ue[:, 1].max()) + 1.0

    quadrant_bounds = [
        (x_min, center_x, y_min, center_y),
        (x_min, center_x, center_y, y_max),
        (center_x, x_max, y_min, center_y),
        (center_x, x_max, center_y, y_max),
    ][:num_agents]

    m = quadrant_margin
    quadrant_inner_bounds = [
        (xlo + m, xhi - m, ylo + m, yhi - m)
        for xlo, xhi, ylo, yhi in quadrant_bounds
    ]
    return quadrant_bounds, quadrant_inner_bounds, (center_x, center_y)


def build_per_quadrant_geo(
    cache: NavmeshCache,
    town: str,
    quadrant_bounds: List[Tuple],
    quadrant_margin: float,
    resolution: float = 1.0,
) -> List[Optional[GeodesicDistanceField]]:
    """Build one GeodesicDistanceField per quadrant.

    Replicates ``CarlaMultiAgentEnv._init_geodesic_grid()``.
    """
    all_tris = cache.get_sidewalk_crosswalk_tris_std(town)
    if all_tris is None or len(all_tris) == 0:
        all_tris = cache.get_walkable_tris_std(town)
    if all_tris is None or len(all_tris) == 0:
        return [None] * len(quadrant_bounds)

    geos: List[Optional[GeodesicDistanceField]] = []
    for xlo, xhi, ylo, yhi in quadrant_bounds:
        pad = quadrant_margin
        # UE bounds -> standard coords: x_std = ue_y, z_std = ue_x
        x_std_lo, x_std_hi = ylo - pad, yhi + pad
        z_std_lo, z_std_hi = xlo - pad, xhi + pad

        # Filter triangles: keep if any vertex is within padded bounds
        verts = all_tris.reshape(-1, 2)  # (3N, 2)
        xs, zs = verts[:, 0], verts[:, 1]
        in_x = (xs >= x_std_lo) & (xs <= x_std_hi)
        in_z = (zs >= z_std_lo) & (zs <= z_std_hi)
        in_bounds = (in_x & in_z).reshape(-1, 3).any(axis=1)
        quad_tris = all_tris[in_bounds]

        if len(quad_tris) == 0:
            geos.append(None)
            continue
        geos.append(GeodesicDistanceField(quad_tris, resolution=resolution))
    return geos


def build_per_quadrant_kdtrees(
    sidewalk_pts_ue: np.ndarray,
    quadrant_inner_bounds: List[Tuple],
) -> Tuple[List[cKDTree], List[np.ndarray]]:
    """Build per-quadrant KD-trees in standard coords for goal sampling."""
    trees = []
    pts_std_list = []
    for xlo, xhi, ylo, yhi in quadrant_inner_bounds:
        mask = (
            (sidewalk_pts_ue[:, 0] >= xlo) & (sidewalk_pts_ue[:, 0] <= xhi) &
            (sidewalk_pts_ue[:, 1] >= ylo) & (sidewalk_pts_ue[:, 1] <= yhi)
        )
        region_ue = sidewalk_pts_ue[mask, :2]
        # Standard: x_std = ue_y, z_std = ue_x
        region_std = np.stack([region_ue[:, 1], region_ue[:, 0]], axis=1)
        trees.append(cKDTree(region_std) if len(region_std) > 0 else None)
        pts_std_list.append(region_std)
    return trees, pts_std_list


# ── Obstacle generation (pure geometry) ──────────────────────────────


def _crosswalk_pca_obb_std(cw: np.ndarray) -> np.ndarray:
    """Compute a PCA-aligned OBB for a crosswalk polygon in standard coords.

    Returns ``(4, 2)`` corners (standard coords) covering the full
    crosswalk area.  Used to deactivate an entire crosswalk by passing
    it as an obstacle polygon to ``puncture_triangles``.
    """
    centre = _crosswalk_centre(cw)
    long_axis, cross_axis, half_length, half_width = _crosswalk_axes(cw)
    c = centre[:2]
    # 4 corners of the PCA-aligned bounding rectangle in UE (x, y)
    corners_ue = np.array([
        c + half_length * long_axis + half_width * cross_axis,
        c + half_length * long_axis - half_width * cross_axis,
        c - half_length * long_axis - half_width * cross_axis,
        c - half_length * long_axis + half_width * cross_axis,
    ])
    # UE -> standard: x_std = ue_y, z_std = ue_x
    return corners_ue[:, ::-1].copy()


def generate_blocked_crosswalk(
    crosswalks: List[np.ndarray],
    bounds: Tuple[float, float, float, float],
    cfg: ObstacleConfig,
) -> Tuple[List[ObstacleSpec], List[np.ndarray]]:
    """Pure-geometry version of ``ObstacleManager._spawn_blocked_crosswalk``.

    Returns ``(obstacle_specs, blocked_crosswalk_obbs_std)`` where each
    blocked OBB is ``(4, 2)`` in standard coords covering the full
    crosswalk — used to deactivate the crosswalk in the geodesic mesh.
    """
    region_cws = _crosswalks_in_region(crosswalks, bounds)
    if not region_cws:
        return [], []

    all_bp_ids = BLOCKER_VEHICLE_BPS + BLOCKER_VEHICLE_FALLBACKS
    cw = random.choice(region_cws)
    centre = _crosswalk_centre(cw)
    long_axis, cross_axis, half_length, half_width = _crosswalk_axes(cw)
    heading = math.degrees(math.atan2(long_axis[1], long_axis[0]))

    # Place blocker near one edge of the crosswalk but pulled slightly
    # inward (1.5m) so it sits on the crosswalk surface rather than
    # straddling the curb boundary.
    edge_side = random.choice([-1.0, 1.0])
    inset = min(2.0, half_length * 0.3)
    edge_xy = centre[:2] + edge_side * (half_length - inset) * long_axis
    # Small jitter along the road axis for visual variety
    along_jitter = random.uniform(-0.3 * half_width, 0.3 * half_width)
    pos_ue = np.array([
        edge_xy[0] + along_jitter * cross_axis[0],
        edge_xy[1] + along_jitter * cross_axis[1],
        centre[2] + 0.5,
    ], dtype=np.float32)
    # Vehicle perpendicular to crossing direction (road-aligned)
    yaw = heading + 90

    specs = [_make_obstacle_spec(
        random.choice(all_bp_ids), pos_ue, yaw, 'blocked_crosswalk')]

    # Flanking barriers along the road to widen the blockage
    barrier_heading_rad = math.radians(heading + 90)
    for sign in [-1, 1]:
        flank_pos = np.array([
            edge_xy[0] + sign * 3.0 * math.cos(barrier_heading_rad),
            edge_xy[1] + sign * 3.0 * math.sin(barrier_heading_rad),
            centre[2] + 0.3,
        ], dtype=np.float32)
        specs.append(_make_obstacle_spec(
            random.choice(BARRIER_PROP_BPS), flank_pos, heading + 90,
            'blocked_crosswalk_flank'))

    # Deactivate the entire crosswalk in the geodesic mesh
    cw_obb = _crosswalk_pca_obb_std(cw)
    return specs, [cw_obb]


def generate_sidewalk_obstruction(
    sidewalk_pts_ue: np.ndarray,
    bounds: Tuple[float, float, float, float],
    cfg: ObstacleConfig,
) -> List[ObstacleSpec]:
    """Pure-geometry version of ``ObstacleManager._spawn_sidewalk_obstruction``."""
    centre = sample_sidewalk_in_bounds(sidewalk_pts_ue, bounds)
    if centre is None:
        return []

    num_props = random.randint(*cfg.num_clutter_range)
    spread = cfg.clutter_spread
    heading = random.uniform(0, 360)
    heading_rad = math.radians(heading)

    specs = []
    for _ in range(num_props):
        along = random.uniform(-spread / 2, spread / 2)
        perp = random.uniform(-0.5, 0.5)
        dx = along * math.cos(heading_rad) - perp * math.sin(heading_rad)
        dy = along * math.sin(heading_rad) + perp * math.cos(heading_rad)

        pos_ue = np.array([
            centre[0] + dx,
            centre[1] + dy,
            centre[2] + 0.3,
        ], dtype=np.float32)
        yaw = random.uniform(0, 360)
        specs.append(_make_obstacle_spec(
            random.choice(CLUTTER_PROP_BPS), pos_ue, yaw,
            'sidewalk_obstruction'))
    return specs


def generate_narrow_passage(
    sidewalk_pts_ue: np.ndarray,
    bounds: Tuple[float, float, float, float],
    cfg: ObstacleConfig,
) -> List[ObstacleSpec]:
    """Pure-geometry version of ``ObstacleManager._spawn_narrow_passage``."""
    centre = sample_sidewalk_in_bounds(sidewalk_pts_ue, bounds)
    if centre is None:
        return []

    gap = random.uniform(*cfg.gap_width_range)
    heading = random.uniform(0, 360)
    heading_rad = math.radians(heading)

    specs = []
    for sign in [-1, 1]:
        num_barriers = random.randint(2, 3)
        for k in range(num_barriers):
            perp_offset = sign * (gap / 2 + 0.3 + k * 0.8)
            along_offset = random.uniform(-0.3, 0.3)
            dx = (along_offset * math.cos(heading_rad)
                  - perp_offset * math.sin(heading_rad))
            dy = (along_offset * math.sin(heading_rad)
                  + perp_offset * math.cos(heading_rad))

            pos_ue = np.array([
                centre[0] + dx,
                centre[1] + dy,
                centre[2] + 0.3,
            ], dtype=np.float32)
            yaw = heading + 90
            specs.append(_make_obstacle_spec(
                random.choice(BARRIER_PROP_BPS), pos_ue, yaw,
                'narrow_passage'))
    return specs


def generate_crosswalk_challenge(
    crosswalks: List[np.ndarray],
    sidewalk_pts_ue: np.ndarray,
    bounds: Tuple[float, float, float, float],
    cfg: ObstacleConfig,
) -> Optional[Tuple[np.ndarray, np.ndarray, List[ObstacleSpec], List[np.ndarray]]]:
    """Pure-geometry version of ``ObstacleManager.setup_crosswalk_challenge``.

    Returns ``(start_ue, goal_std, obstacle_specs, blocked_cw_obbs_std)``
    or ``None``.
    """
    region_cws = _crosswalks_in_region(crosswalks, bounds)
    if len(region_cws) < 2:
        return None

    centres = np.array([_crosswalk_centre(cw)[:2] for cw in region_cws])
    order = list(range(len(region_cws)))
    random.shuffle(order)

    all_bp_ids = BLOCKER_VEHICLE_BPS + BLOCKER_VEHICLE_FALLBACKS

    for idx in order:
        cw = region_cws[idx]
        centre = _crosswalk_centre(cw)
        long_axis, cross_axis, half_length, half_width = _crosswalk_axes(cw)

        # Filter 1: minimum crossing width
        if half_width * 2 < cfg.min_crosswalk_width:
            continue

        # Filter 2: alternative crosswalk within radius
        c_xy = centres[idx]
        dists_to_others = np.linalg.norm(centres - c_xy, axis=1)
        dists_to_others[idx] = np.inf
        if dists_to_others.min() > cfg.alt_crosswalk_radius:
            continue

        # Agent & goal ideal positions
        side = random.choice([-1.0, 1.0])
        agent_offset = half_length + cfg.agent_standoff
        goal_offset = half_length + cfg.goal_standoff
        agent_ideal_xy = centre[:2] + side * long_axis * agent_offset
        goal_ideal_xy = centre[:2] - side * long_axis * goal_offset

        # Snap to navmesh
        agent_loc = snap_to_navmesh_ue(
            agent_ideal_xy, sidewalk_pts_ue, cfg.navmesh_snap_radius)
        if agent_loc is None:
            continue
        goal_loc = snap_to_navmesh_ue(
            goal_ideal_xy, sidewalk_pts_ue, cfg.navmesh_snap_radius)
        if goal_loc is None:
            continue

        # Place blocker near one edge of the crosswalk but pulled slightly
        # inward (1.5m) so it sits on the crosswalk surface.
        heading = math.degrees(math.atan2(long_axis[1], long_axis[0]))
        edge_side = random.choice([-1.0, 1.0])
        inset = min(1.5, half_length * 0.3)
        edge_xy = centre[:2] + edge_side * (half_length - inset) * long_axis
        along_jitter = random.uniform(-0.3 * half_width, 0.3 * half_width)
        blocker_pos = np.array([
            edge_xy[0] + along_jitter * cross_axis[0],
            edge_xy[1] + along_jitter * cross_axis[1],
            centre[2] + 0.5,
        ], dtype=np.float32)
        yaw = heading + 90

        specs = [_make_obstacle_spec(
            random.choice(all_bp_ids), blocker_pos, yaw,
            'crosswalk_challenge')]

        # Flanking barriers along the road at the same edge
        barrier_heading_rad = math.radians(heading + 90)
        for sign in [-1, 1]:
            flank_pos = np.array([
                edge_xy[0] + sign * 3.0 * math.cos(barrier_heading_rad),
                edge_xy[1] + sign * 3.0 * math.sin(barrier_heading_rad),
                centre[2] + 0.3,
            ], dtype=np.float32)
            specs.append(_make_obstacle_spec(
                random.choice(BARRIER_PROP_BPS), flank_pos, heading + 90,
                'crosswalk_challenge_flank'))

        # Deactivate the entire crosswalk in the geodesic mesh
        cw_obb = _crosswalk_pca_obb_std(cw)

        # Build return values
        start_ue = agent_loc.copy()
        goal_std = np.array([goal_loc[1], goal_loc[0]], dtype=np.float64)
        return start_ue, goal_std, specs, [cw_obb]

    return None


def generate_region_obstacles(
    crosswalks: List[np.ndarray],
    sidewalk_pts_ue: np.ndarray,
    bounds: Tuple[float, float, float, float],
    cfg: ObstacleConfig,
) -> Tuple[List[ObstacleSpec], List[np.ndarray]]:
    """Generate region-level persistent obstacles (non-challenge).

    Returns ``(obstacle_specs, blocked_crosswalk_obbs_std)``.
    """
    specs: List[ObstacleSpec] = []
    blocked_cw_obbs: List[np.ndarray] = []

    if random.random() < cfg.p_blocked_crosswalk:
        s, cw_obbs = generate_blocked_crosswalk(crosswalks, bounds, cfg)
        specs.extend(s)
        blocked_cw_obbs.extend(cw_obbs)

    if random.random() < cfg.p_sidewalk_obstruction:
        specs.extend(generate_sidewalk_obstruction(
            sidewalk_pts_ue, bounds, cfg))

    if random.random() < cfg.p_narrow_passage:
        specs.extend(generate_narrow_passage(
            sidewalk_pts_ue, bounds, cfg))

    return specs[:cfg.max_obstacles_per_region], blocked_cw_obbs


def _build_obstacle_obbs(
    obstacle_specs: List[ObstacleSpec],
    blocked_cw_obbs: List[np.ndarray],
) -> Optional[List[dict]]:
    """Merge obstacle OBBs and blocked crosswalk OBBs into a single list.

    Returns the format expected by
    ``GeodesicDistanceField.compute_distance_field(obstacle_obbs_std=...)``.
    """
    obbs = []
    if obstacle_specs:
        obbs.extend({'corners_std': o.corners_std} for o in obstacle_specs)
    if blocked_cw_obbs:
        obbs.extend({'corners_std': cw} for cw in blocked_cw_obbs)
    return obbs or None


# ── Scenario generation ──────────────────────────────────────────────

def _ue_to_recast(ue_x, ue_y, ue_z=0.0):
    """Convert UE coordinates to Recast frame for Detour queries.

    UE: (X-forward, Y-right, Z-up)
    Recast: (X-right, Y-up, Z-forward)
    Mapping: recast = (ue_x, ue_z, ue_y)
    """
    return (float(ue_x), float(ue_z), float(ue_y))


def _detour_check_feasibility(
detour_pf, start_ue, goal_std, budget,
) -> Tuple[bool, float]:
    """Fast feasibility check using Detour (sub-ms).

    Returns (is_feasible, geodesic_distance).
    """
    # goal_std = (ue_y, ue_x), so goal_ue = (goal_std[1], goal_std[0])
    sx, sy, sz = _ue_to_recast(start_ue[0], start_ue[1], start_ue[2])
    ex, ey, ez = _ue_to_recast(goal_std[1], goal_std[0], 0.0)
    d = detour_pf.geodesic_distance(sx, sy, sz, ex, ey, ez)
    if (not np.isfinite(d)) or (d > budget):
        return False, d
    return True, d


def generate_scenario(
    quadrant_idx: int,
    geo: GeodesicDistanceField,
    kdtree: cKDTree,
    pts_std: np.ndarray,
    sidewalk_pts_ue: np.ndarray,
    crosswalks: List[np.ndarray],
    quadrant_bounds: Tuple[float, float, float, float],
    quadrant_inner_bounds: Tuple[float, float, float, float],
    cfg: ObstacleConfig,
    goal_range: float = 8.0,
    max_speed: float = 1.4,
    max_episode_steps: int = 32,
    max_retries: int = 5,
    use_obstacles: bool = True,
    detour_pf=None,
) -> Optional[dict]:
    """Generate one complete scenario for a quadrant.

    When *detour_pf* is provided (a ``detour_nav.DetourPathfinder``),
    feasibility checking uses sub-millisecond Detour queries instead of
    running a full grid Dijkstra per retry.  The expensive grid Dijkstra
    is only run **once** after a feasible (start, goal) pair is confirmed.

    Without *detour_pf*, falls back to running grid Dijkstra on every
    attempt (slower but no C++ dependency needed).

    Returns a dict ready for ``np.savez_compressed``, or ``None`` if
    all retries fail.
    """
    if geo is None or kdtree is None or len(pts_std) == 0:
        return None

    budget = max_speed * max_episode_steps

    # Decide scenario template
    is_challenge = (use_obstacles
                    and random.random() < cfg.p_crosswalk_challenge)

    challenge_result = None
    if is_challenge:
        challenge_result = generate_crosswalk_challenge(
            crosswalks, sidewalk_pts_ue, quadrant_inner_bounds, cfg)

    # Generate region-level obstacles
    region_obstacles: List[ObstacleSpec] = []
    region_blocked_cws: List[np.ndarray] = []
    if use_obstacles:
        region_obstacles, region_blocked_cws = generate_region_obstacles(
            crosswalks, sidewalk_pts_ue, quadrant_bounds, cfg)

    # ── Retry loop: find a feasible (start, goal) pair ──
    found_start_ue = None
    found_goal_std = None
    found_obstacles = None
    found_blocked_cws = None
    found_template = None
    found_method = None

    for attempt in range(max_retries + 1):
        if challenge_result is not None:
            start_ue, goal_std, challenge_obstacles, challenge_cw_obbs = \
                challenge_result
            all_obstacles = region_obstacles + challenge_obstacles
            all_blocked_cws = region_blocked_cws + challenge_cw_obbs
            scenario_template = 'crosswalk_challenge'
            goal_method = 'crosswalk_challenge'
        else:
            # Normal scenario: sample start and goal
            start_pt = sample_sidewalk_in_bounds(
                sidewalk_pts_ue, quadrant_inner_bounds)
            if start_pt is None:
                return None
            start_ue = start_pt

            # Sample goal via KD-tree annulus
            start_std = np.array([start_ue[1], start_ue[0]])
            r_min, r_max = 0.8 * goal_range, goal_range
            idxs = kdtree.query_ball_point(start_std, r_max)
            if not idxs:
                continue

            candidates = pts_std[idxs]
            dists = np.linalg.norm(candidates - start_std, axis=1)
            annulus_mask = dists >= r_min
            annulus = candidates[annulus_mask]

            if len(annulus) > 0:
                goal_std = annulus[np.random.randint(len(annulus))]
                goal_method = 'navmesh_annulus'
            elif len(candidates) > 0:
                goal_std = candidates[np.argmax(dists)]
                goal_method = 'navmesh_relaxed'
            else:
                continue

            all_obstacles = region_obstacles
            all_blocked_cws = region_blocked_cws
            scenario_template = 'normal'

        # ── Fast feasibility check ──
        if detour_pf is not None:
            # Sub-ms Detour query — no grid Dijkstra needed for rejection
            feasible, _ = _detour_check_feasibility(
                detour_pf, start_ue, goal_std, budget)
            if not feasible:
                if challenge_result is not None:
                    challenge_result = None
                continue
        else:
            # Slow path: run full grid Dijkstra to check feasibility
            obstacle_obbs = _build_obstacle_obbs(
                all_obstacles, all_blocked_cws)
            dist_field = geo.compute_distance_field(
                goal_std, obstacle_obbs_std=obstacle_obbs)
            start_std_check = np.array([start_ue[1], start_ue[0]])
            d = geo.query(dist_field, start_std_check)
            if not np.isfinite(d) or d > budget:
                if challenge_result is not None:
                    challenge_result = None
                continue

        # Pair is feasible
        found_start_ue = start_ue
        found_goal_std = goal_std
        found_obstacles = all_obstacles
        found_blocked_cws = all_blocked_cws
        found_template = scenario_template
        found_method = goal_method
        break

    if found_start_ue is None:
        return None

    # ── Compute distance field (once, for the confirmed pair) ──
    obstacle_obbs = _build_obstacle_obbs(found_obstacles, found_blocked_cws)

    # When Detour was used for feasibility, we haven't computed the
    # grid distance field yet — do it now (the only Dijkstra per scenario)
    if detour_pf is not None or 'dist_field' not in dir():
        dist_field = geo.compute_distance_field(
            found_goal_std, obstacle_obbs_std=obstacle_obbs)

    start_std_check = np.array([found_start_ue[1], found_start_ue[0]])
    d = geo.query(dist_field, start_std_check)
    if not np.isfinite(d):
        # Rare: Detour said feasible but grid disagrees (resolution diff)
        return None

    # Trace geodesic path
    geo_path = geo.trace_path(dist_field, start_std_check)

    # Euclidean distance
    euc_dist = float(np.linalg.norm(
        start_std_check - found_goal_std.astype(np.float64)))

    # Serialize obstacle specs
    all_obstacles = found_obstacles
    if all_obstacles:
        bp_ids = np.array([o.blueprint_id for o in all_obstacles],
                          dtype='U64')
        positions_ue = np.stack([o.position_ue for o in all_obstacles])
        yaws = np.array([o.yaw_deg for o in all_obstacles],
                        dtype=np.float32)
        corners = np.stack([o.corners_std for o in all_obstacles])
        types = np.array([o.scenario_type for o in all_obstacles],
                         dtype='U32')
    else:
        bp_ids = np.array([], dtype='U64')
        positions_ue = np.zeros((0, 3), dtype=np.float32)
        yaws = np.array([], dtype=np.float32)
        corners = np.zeros((0, 4, 2), dtype=np.float64)
        types = np.array([], dtype='U32')

    # Blocked crosswalk OBBs (for visualization — mark deactivated crosswalks)
    if found_blocked_cws:
        blocked_cw_arr = np.stack(found_blocked_cws).astype(np.float32)
    else:
        blocked_cw_arr = np.zeros((0, 4, 2), dtype=np.float32)

    return dict(
        start_ue=found_start_ue.astype(np.float32),
        start_std=start_std_check.astype(np.float32),
        goal_std=found_goal_std.astype(np.float32),
        obstacle_bp_ids=bp_ids,
        obstacle_positions_ue=positions_ue,
        obstacle_yaws_deg=yaws,
        obstacle_corners_std=corners.astype(np.float32),
        obstacle_scenario_types=types,
        blocked_crosswalk_obbs_std=blocked_cw_arr,
        dist_field=dist_field.astype(np.float32),
        grid_x_min=np.float64(geo._x_min),
        grid_z_min=np.float64(geo._z_min),
        grid_resolution=np.float64(geo._resolution),
        geodesic_distance=np.float32(d),
        euclidean_distance=np.float32(euc_dist),
        scenario_template=np.array(found_template, dtype='U32'),
        goal_method=np.array(found_method, dtype='U32'),
    )


# ── Town-level orchestrator ──────────────────────────────────────────


def _try_load_detour(navmesh_bin_dir: Optional[str], town: str):
    """Try to load a Detour pathfinder for fast feasibility checking.

    Returns a ``detour_nav.DetourPathfinder`` or ``None``.
    """
    if navmesh_bin_dir is None:
        return None
    try:
        import detour_nav
    except ImportError:
        print("  [Detour] detour_nav not built — using grid Dijkstra "
              "(slower). Build with: cd detour_nav && bash build.sh")
        return None

    bin_path = Path(navmesh_bin_dir) / f'{town}.bin'
    if not bin_path.exists():
        # Try alternative naming
        for pattern in [f'{town}_navmesh.bin', f'{town}.bin']:
            alt = Path(navmesh_bin_dir) / pattern
            if alt.exists():
                bin_path = alt
                break
        else:
            print(f"  [Detour] No .bin navmesh found for {town} "
                  f"in {navmesh_bin_dir}")
            return None

    pf = detour_nav.DetourPathfinder()
    if not pf.load(str(bin_path)):
        print(f"  [Detour] Failed to load {bin_path}")
        return None

    pf.set_sidewalk_only()
    print(f"  [Detour] Loaded {bin_path.name} — using fast feasibility")
    return pf


def generate_town_scenarios(
    town: str,
    cache: NavmeshCache,
    output_dir: str,
    scenarios_per_quadrant: int,
    cfg: ObstacleConfig,
    goal_range: float,
    quadrant_margin: float,
    resolution: float,
    max_speed: float = 1.4,
    max_episode_steps: int = 32,
    num_agents: int = 4,
    use_obstacles: bool = True,
    seed: int = 0,
    navmesh_bin_dir: Optional[str] = None,
):
    """Generate all scenarios for one town."""
    random.seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"Generating scenarios for {town}")
    print(f"{'='*60}")

    # Load navmesh data
    if not cache.load(town):
        print(f"  ERROR: No navmesh cache for {town}, skipping")
        return

    walkable_pts = cache.get_walkable_points_ue(town)
    if walkable_pts is None or len(walkable_pts) == 0:
        print(f"  ERROR: No walkable points for {town}, skipping")
        return

    sidewalk_pts = cache.get_sidewalk_points_ue(town)
    if sidewalk_pts is None or len(sidewalk_pts) == 0:
        print(f"  WARNING: No sidewalk-only points, using all walkable")
        sidewalk_pts = walkable_pts

    crosswalks = cache.get_crosswalks_ue(town)

    # Quadrant splitting (replicates _compute_regions)
    q_bounds, q_inner, center = compute_quadrant_bounds(
        walkable_pts[:, :2], num_agents, quadrant_margin)
    print(f"  Center: ({center[0]:.1f}, {center[1]:.1f}), "
          f"margin: {quadrant_margin:.1f}m")

    # Per-quadrant geodesic grids
    geos = build_per_quadrant_geo(
        cache, town, q_bounds, quadrant_margin, resolution)

    # Per-quadrant KD-trees for goal sampling
    trees, pts_std_list = build_per_quadrant_kdtrees(
        sidewalk_pts[:, :2], q_inner)

    # Optional Detour pathfinder for fast feasibility checking
    detour_pf = _try_load_detour(navmesh_bin_dir, town)

    # Output directories
    town_dir = Path(output_dir) / town
    for qi in range(num_agents):
        (town_dir / f'q{qi}').mkdir(parents=True, exist_ok=True)

    # Write metadata
    grid_metadata = {}
    for qi in range(num_agents):
        geo = geos[qi]
        if geo is not None:
            grid_metadata[f'q{qi}'] = {
                'x_min': float(geo._x_min),
                'z_min': float(geo._z_min),
                'H': int(geo._H),
                'W': int(geo._W),
                'resolution': float(geo._resolution),
            }

    meta = {
        'town': town,
        'quadrant_bounds': [list(b) for b in q_bounds],
        'quadrant_inner_bounds': [list(b) for b in q_inner],
        'quadrant_margin': quadrant_margin,
        'center_ue': list(center),
        'resolution': resolution,
        'goal_range': goal_range,
        'max_speed': max_speed,
        'max_episode_steps': max_episode_steps,
        'num_agents': num_agents,
        'grid_metadata': grid_metadata,
    }
    with open(town_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # Generate scenarios per quadrant
    total_generated = 0
    total_failed = 0
    t0 = time.perf_counter()

    for qi in range(num_agents):
        geo = geos[qi]
        tree = trees[qi]
        pts_std = pts_std_list[qi]

        if geo is None or tree is None or len(pts_std) == 0:
            print(f"  Quadrant {qi}: skipped (no geodesic grid or points)")
            continue

        generated = 0
        failed = 0
        qt0 = time.perf_counter()

        # Retry until we have exactly scenarios_per_quadrant successes,
        # or hit the attempt cap (3x target to avoid infinite loops).
        max_attempts = scenarios_per_quadrant * 3
        attempts = 0
        pbar = tqdm(total=scenarios_per_quadrant,
                    desc=f'  q{qi}', unit='scenario',
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                               '[{elapsed}<{remaining}, {rate_fmt}, '
                               '{postfix}]')
        while generated < scenarios_per_quadrant and attempts < max_attempts:
            result = generate_scenario(
                quadrant_idx=qi,
                geo=geo,
                kdtree=tree,
                pts_std=pts_std,
                sidewalk_pts_ue=sidewalk_pts,
                crosswalks=crosswalks,
                quadrant_bounds=q_bounds[qi],
                quadrant_inner_bounds=q_inner[qi],
                cfg=cfg,
                goal_range=goal_range,
                max_speed=max_speed,
                max_episode_steps=max_episode_steps,
                use_obstacles=use_obstacles,
                detour_pf=detour_pf,
            )
            attempts += 1

            if result is not None:
                path = town_dir / f'q{qi}' / f'scenario_{generated:06d}.npz'
                np.savez_compressed(str(path), **result)
                generated += 1
                pbar.update(1)
            else:
                failed += 1

            pbar.set_postfix(failed=failed, attempts=attempts)

        pbar.close()
        if generated < scenarios_per_quadrant:
            print(f"  WARNING: q{qi} only generated {generated}/"
                  f"{scenarios_per_quadrant} after {max_attempts} attempts")

        total_generated += generated
        total_failed += failed
        elapsed = time.perf_counter() - qt0
        print(f"  Quadrant {qi}: {generated}/{scenarios_per_quadrant} "
              f"generated ({failed} failed, {attempts} attempts) "
              f"in {elapsed:.1f}s")

    total_time = time.perf_counter() - t0
    print(f"\n  {town} total: {total_generated} scenarios in {total_time:.1f}s "
          f"({total_failed} failed)")


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description='Offline scenario precomputation for UrbanNav RL training')
    parser.add_argument('--navmesh_dir', type=str, required=True,
                        help='Directory containing *_navmesh_cache.npz files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for scenarios')
    parser.add_argument('--towns', nargs='+', required=True,
                        help='Town names (e.g., Town02 Town03)')
    parser.add_argument('--scenarios_per_quadrant', type=int, default=1000,
                        help='Number of scenarios per quadrant (default: 1000)')
    parser.add_argument('--goal_range', type=float, default=8.0,
                        help='Max Euclidean goal distance (default: 8.0)')
    parser.add_argument('--quadrant_margin', type=float, default=None,
                        help='Quadrant inner margin (default: goal_range)')
    parser.add_argument('--resolution', type=float, default=1.0,
                        help='Geodesic grid resolution in metres (default: 1.0)')
    parser.add_argument('--max_speed', type=float, default=1.4,
                        help='Agent max speed for budget calc (default: 1.4)')
    parser.add_argument('--max_episode_steps', type=int, default=32,
                        help='Max episode steps for budget calc (default: 32)')
    parser.add_argument('--num_agents', type=int, default=4,
                        help='Number of quadrants/agents (default: 4)')
    parser.add_argument('--obstacles', action='store_true',
                        help='Generate obstacle configurations')
    parser.add_argument('--p_crosswalk_challenge', type=float, default=0.5,
                        help='Probability of crosswalk challenge (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--navmesh_bin_dir', type=str, default=None,
                        help='Directory containing CARLA .bin navmesh files '
                             '(for Detour-accelerated feasibility checking). '
                             'Falls back to grid Dijkstra if not provided.')
    args = parser.parse_args()

    if args.quadrant_margin is None:
        args.quadrant_margin = args.goal_range

    cache = NavmeshCache(args.navmesh_dir)
    cfg = ObstacleConfig(p_crosswalk_challenge=args.p_crosswalk_challenge)

    for town in args.towns:
        generate_town_scenarios(
            town=town,
            cache=cache,
            output_dir=args.output_dir,
            scenarios_per_quadrant=args.scenarios_per_quadrant,
            cfg=cfg,
            goal_range=args.goal_range,
            quadrant_margin=args.quadrant_margin,
            resolution=args.resolution,
            max_speed=args.max_speed,
            max_episode_steps=args.max_episode_steps,
            num_agents=args.num_agents,
            use_obstacles=args.obstacles,
            seed=args.seed,
            navmesh_bin_dir=args.navmesh_bin_dir,
        )

    print(f"\nDone. Scenarios saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
