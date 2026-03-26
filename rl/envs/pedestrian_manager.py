"""
Social Force Model (SFM) pedestrian manager for CARLA.

Spawns and controls NPC pedestrians using the Helbing & Molnar (1995)
social force model.  Unlike CARLA's built-in AI walker controller, this
handles dynamically spawned obstacles that are absent from the navmesh.

Forces applied per pedestrian:
  - Desired    : drives toward a navmesh-sampled destination
  - Ped–ped    : repulsion from other pedestrians
  - Obstacle   : repulsion from spawned static obstacles
  - Boundary   : repulsion from walkable-region edges (punctured mesh)
  - NO ego     : the RL agent must learn to avoid pedestrians itself

Boundary constraint
~~~~~~~~~~~~~~~~~~~
When ``set_walkable_mesh()`` is called with navmesh triangles and obstacle
bounding boxes, the manager "punctures" the walkable mesh by removing
triangles that overlap with obstacle geometry.  Boundary edges of the
remaining mesh are extracted and used to generate repulsion forces that
keep pedestrians within walkable regions and away from obstacle interiors.
"""

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import carla

from rl.utils.mesh_utils import puncture_triangles


# ── Config ────────────────────────────────────────────────────────────


@dataclass
class PedestrianConfig:
    """Tunable parameters for SFM pedestrian simulation."""
    num_per_region: int = 4                          # pedestrians per quadrant
    desired_speed_range: Tuple[float, float] = (0.8, 1.4)   # m/s
    max_speed: float = 2.0                           # absolute speed cap (m/s)

    # SFM force parameters (Helbing & Molnar 1995 defaults)
    tau: float = 0.5             # relaxation time (s)
    A_ped: float = 2.1           # ped–ped repulsion strength
    B_ped: float = 0.3           # ped–ped repulsion range (m)
    A_obs: float = 10.0          # obstacle repulsion strength
    B_obs: float = 0.2           # obstacle repulsion range (m)
    ped_radius: float = 0.3      # body radius (m)

    # Boundary (walkable-region edge) repulsion
    A_boundary: float = 5.0          # boundary repulsion strength
    B_boundary: float = 0.5          # boundary repulsion range (m)

    # Navigation
    dest_reach_threshold: float = 2.0   # resample when this close (m)
    interaction_radius: float = 5.0     # ignore forces beyond this (m)


# ── Per-pedestrian state ──────────────────────────────────────────────


class _Pedestrian:
    __slots__ = ("actor", "desired_speed", "region_idx", "destination_ue",
                 "_traj_buf")

    _TRAJ_MAXLEN = 200   # keep last N positions for visualisation

    def __init__(self, actor, desired_speed: float, region_idx: int):
        self.actor = actor
        self.desired_speed = desired_speed
        self.region_idx = region_idx
        self.destination_ue: Optional[np.ndarray] = None   # (3,) UE xyz
        self._traj_buf: deque = deque(maxlen=_Pedestrian._TRAJ_MAXLEN)


# ── Manager ───────────────────────────────────────────────────────────


class PedestrianManager:
    """Spawn and control SFM pedestrians on a single CARLA world.

    Lifecycle::

        mgr = PedestrianManager(config)
        mgr.initialize(world, bp_lib, quadrant_bounds)   # spawns walkers
        mgr.set_walkable_mesh(tris_std, obstacle_corners) # boundary constraint
        # each simulation tick:
        mgr.update(dt, obstacle_positions_ue)             # SFM step
        world.tick()
        # cleanup:
        mgr.clear_all(client)
    """

    def __init__(self, config: Optional[PedestrianConfig] = None):
        self.config = config
        self.enabled = config is not None
        self.world = None
        self.bp_lib = None
        self.quadrant_bounds = None
        self.pedestrians: List[_Pedestrian] = []
        self._navmesh_cache = None
        self._current_town = None
        # Boundary constraint (populated by set_walkable_mesh)
        self._boundary_edges_ue: Optional[np.ndarray] = None   # (E, 2, 2)
        self._boundary_normals_ue: Optional[np.ndarray] = None  # (E, 2)

    # ── Lifecycle ─────────────────────────────────────────────────────

    def initialize(self, world, bp_lib, quadrant_bounds,
                   navmesh_cache=None, town=None):
        """Spawn pedestrians.  Call once after map load + region split."""
        if not self.enabled:
            return
        self._navmesh_cache = navmesh_cache
        self._current_town = town
        self.world = world
        self.bp_lib = bp_lib
        self.quadrant_bounds = quadrant_bounds
        self._spawn_all()

    def clear_all(self, client):
        """Destroy every pedestrian actor."""
        if not self.enabled or not self.pedestrians:
            return
        ids = [p.actor.id for p in self.pedestrians]
        try:
            client.apply_batch([carla.command.DestroyActor(x) for x in ids])
        except Exception as e:
            print(f"  [PedestrianManager] clear_all error: {e}")
        self.pedestrians.clear()

    # ── Walkable-mesh boundary constraint ─────────────────────────────

    def set_walkable_mesh(self, walkable_tris_std: np.ndarray,
                          obstacle_corners_std: Optional[List[np.ndarray]] = None):
        """Build boundary edges from a punctured walkable mesh.

        Call after obstacles are spawned to enable boundary constraints.
        Triangles overlapping with obstacle bounding boxes are removed
        ("punctured") and boundary edges of the remaining mesh produce
        inward-pointing repulsion forces during ``update()``.

        Parameters
        ----------
        walkable_tris_std : (N, 3, 2)
            Walkable navmesh triangles in standard 2D coords (x_std, z_std).
        obstacle_corners_std : list of (4, 2) ndarrays, optional
            Oriented bounding-box corners per obstacle in standard coords.
        """
        if not self.enabled:
            return

        n_before = walkable_tris_std.shape[0]

        # Convert standard → UE: (x_std, z_std) → (ue_x=z_std, ue_y=x_std)
        tris_ue = walkable_tris_std[:, :, ::-1].copy()

        # Puncture: remove triangles overlapping with obstacle OBBs
        if obstacle_corners_std:
            obbs_ue = [c[:, ::-1].copy() for c in obstacle_corners_std]
            tris_ue = puncture_triangles(tris_ue, obbs_ue,
                                         buffer=self.config.ped_radius)

        if tris_ue.shape[0] == 0:
            self._boundary_edges_ue = None
            self._boundary_normals_ue = None
            return

        # Extract boundary edges and inward normals
        edges, normals = self._extract_boundary(tris_ue)

        # Filter to edges within the active quadrant region (+ margin)
        if self.quadrant_bounds and edges.shape[0] > 0:
            margin = self.config.interaction_radius
            xlo = min(b[0] for b in self.quadrant_bounds) - margin
            xhi = max(b[1] for b in self.quadrant_bounds) + margin
            ylo = min(b[2] for b in self.quadrant_bounds) - margin
            yhi = max(b[3] for b in self.quadrant_bounds) + margin
            mids = (edges[:, 0] + edges[:, 1]) / 2
            keep = ((mids[:, 0] >= xlo) & (mids[:, 0] <= xhi)
                    & (mids[:, 1] >= ylo) & (mids[:, 1] <= yhi))
            edges = edges[keep]
            normals = normals[keep]

        self._boundary_edges_ue = edges
        self._boundary_normals_ue = normals

        n_punctured = n_before - tris_ue.shape[0]
        print(f"  [PedestrianManager] Walkable mesh: {n_before} tris, "
              f"{n_punctured} punctured, {edges.shape[0]} boundary edges")

    @staticmethod
    def _extract_boundary(tris_ue: np.ndarray):
        """Extract boundary edges and inward normals from a triangle mesh.

        Boundary edges appear in exactly one triangle.  The inward normal
        points toward the triangle interior (the opposite vertex).

        Parameters
        ----------
        tris_ue : (N, 3, 2) — triangles in UE (ue_x, ue_y)

        Returns
        -------
        edges   : (E, 2, 2) — boundary edge endpoint pairs
        normals : (E, 2)    — unit inward normals
        """
        PREC = 3   # quantise to 1 mm for edge key stability

        edge_count: dict = {}
        edge_data: dict = {}

        N = tris_ue.shape[0]
        for t in range(N):
            v = tris_ue[t]                             # (3, 2)
            for i in range(3):
                j = (i + 1) % 3
                k = 3 - i - j                          # opposite vertex

                key_a = tuple(np.round(v[i], PREC).tolist())
                key_b = tuple(np.round(v[j], PREC).tolist())
                edge_key = (min(key_a, key_b), max(key_a, key_b))

                edge_count[edge_key] = edge_count.get(edge_key, 0) + 1
                if edge_count[edge_key] == 1:
                    # Store geometry only on first encounter
                    edge_data[edge_key] = (v[i].copy(), v[j].copy(),
                                           v[k].copy())

        boundary_edges = []
        boundary_normals = []

        for edge_key, count in edge_count.items():
            if count != 1:
                continue
            a, b, opp = edge_data[edge_key]

            edge_dir = b - a
            # Perpendicular (90° CCW rotation)
            n_candidate = np.array([-edge_dir[1], edge_dir[0]])

            # Choose the direction pointing toward the opposite vertex
            mid = (a + b) * 0.5
            if np.dot(n_candidate, opp - mid) < 0:
                n_candidate = -n_candidate

            norm_len = np.linalg.norm(n_candidate)
            if norm_len < 1e-8:
                continue
            n_candidate /= norm_len

            boundary_edges.append((a, b))
            boundary_normals.append(n_candidate)

        if not boundary_edges:
            return np.empty((0, 2, 2)), np.empty((0, 2))

        return np.array(boundary_edges), np.array(boundary_normals)

    def _compute_boundary_force(self, positions: np.ndarray) -> np.ndarray:
        """Vectorised boundary repulsion for all pedestrians.

        For each pedestrian, finds the distance to every boundary edge
        (nearest point on the segment) and applies an exponential
        repulsion force along the edge's inward normal.

        Parameters
        ----------
        positions : (n, 2) — pedestrian positions in UE

        Returns
        -------
        (n, 2) — boundary repulsion forces
        """
        if (self._boundary_edges_ue is None
                or self._boundary_normals_ue is None
                or self._boundary_edges_ue.shape[0] == 0):
            return np.zeros_like(positions)

        cfg = self.config
        edges = self._boundary_edges_ue       # (E, 2, 2)
        normals = self._boundary_normals_ue   # (E, 2)

        a = edges[:, 0, :]                     # (E, 2) edge start
        b = edges[:, 1, :]                     # (E, 2) edge end
        ab = b - a                             # (E, 2)
        ab_len_sq = (ab ** 2).sum(axis=1)      # (E,)
        ab_len_sq = np.maximum(ab_len_sq, 1e-10)

        # Vector from a to each position: (n, E, 2)
        ap = positions[:, None, :] - a[None, :, :]

        # Project onto segment, clamp to [0, 1]
        t = (ap * ab[None, :, :]).sum(axis=2) / ab_len_sq[None, :]   # (n, E)
        t = np.clip(t, 0.0, 1.0)

        # Closest point on each edge
        closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]     # (n,E,2)

        # Distance from pedestrian to closest point
        diff = positions[:, None, :] - closest                        # (n,E,2)
        dist = np.linalg.norm(diff, axis=2)                           # (n, E)

        # Exponential repulsion, masked by interaction radius
        mask = dist < cfg.interaction_radius
        mag = cfg.A_boundary * np.exp(-dist / cfg.B_boundary) * mask  # (n, E)

        # Sum forces along inward normals over all edges
        f_boundary = (mag[:, :, None] * normals[None, :, :]).sum(axis=1)
        return f_boundary                                              # (n, 2)

    # ── Spawning ──────────────────────────────────────────────────────

    def _spawn_all(self):
        cfg = self.config
        walker_bps = list(self.bp_lib.filter('walker.pedestrian.*'))
        if not walker_bps:
            print("  [PedestrianManager] WARNING: no walker blueprints found")
            return

        num_regions = len(self.quadrant_bounds)
        for region_idx in range(num_regions):
            bounds = self.quadrant_bounds[region_idx]
            for _ in range(cfg.num_per_region):
                loc = self._sample_navmesh_in_bounds(bounds)
                if loc is None:
                    continue
                loc.z += 1.0       # prevent ground-clip on spawn

                bp = random.choice(walker_bps)
                if bp.has_attribute('is_invincible'):
                    bp.set_attribute('is_invincible', 'false')

                actor = self.world.try_spawn_actor(bp, carla.Transform(loc))
                if actor is None:
                    continue

                speed = random.uniform(*cfg.desired_speed_range)
                ped = _Pedestrian(actor, speed, region_idx)

                dest = self._sample_navmesh_in_bounds(bounds)
                if dest is not None:
                    ped.destination_ue = np.array([dest.x, dest.y, dest.z])

                self.pedestrians.append(ped)

        print(f"  [PedestrianManager] Spawned {len(self.pedestrians)} "
              f"SFM pedestrians ({cfg.num_per_region}/region)")

    # ── SFM update (vectorised) ───────────────────────────────────────

    def update(self, dt: float,
               obstacle_positions_ue: Optional[np.ndarray] = None):
        """Compute SFM forces and apply ``WalkerControl`` for every ped.

        Call this **before** each ``world.tick()``.

        Parameters
        ----------
        dt : float — simulation timestep (seconds)
        obstacle_positions_ue : (M, 2) or None — static obstacle positions
        """
        if not self.enabled or not self.pedestrians:
            return

        cfg = self.config
        n = len(self.pedestrians)

        # -- gather state --
        positions = np.zeros((n, 2))
        velocities = np.zeros((n, 2))
        destinations = np.zeros((n, 2))
        desired_speeds = np.zeros(n)

        for i, ped in enumerate(self.pedestrians):
            loc = ped.actor.get_location()
            positions[i] = (loc.x, loc.y)
            vel = ped.actor.get_velocity()
            velocities[i] = (vel.x, vel.y)
            if ped.destination_ue is not None:
                destinations[i] = ped.destination_ue[:2]
            desired_speeds[i] = ped.desired_speed

        # -- destination re-sampling --
        dist_to_dest = np.linalg.norm(destinations - positions, axis=1)
        for i, ped in enumerate(self.pedestrians):
            if dist_to_dest[i] < cfg.dest_reach_threshold:
                self._resample_destination(ped)
                if ped.destination_ue is not None:
                    destinations[i] = ped.destination_ue[:2]

        # -- 1. desired force: (v0 * e_d − v) / τ --
        diff_dest = destinations - positions                         # (n, 2)
        d2d = np.linalg.norm(diff_dest, axis=1, keepdims=True)      # (n, 1)
        d2d = np.maximum(d2d, 1e-6)
        e_d = diff_dest / d2d                                        # (n, 2)
        f_desired = (desired_speeds[:, None] * e_d - velocities) / cfg.tau

        # -- 2. ped–ped repulsion --
        # diff[i,j] = pos_i − pos_j
        diff_pp = positions[:, None, :] - positions[None, :, :]      # (n,n,2)
        dist_pp = np.linalg.norm(diff_pp, axis=2)                   # (n,n)
        np.fill_diagonal(dist_pp, np.inf)

        eff_dist = np.maximum(dist_pp - 2 * cfg.ped_radius, 0.0)
        n_pp = diff_pp / np.maximum(dist_pp[:, :, None], 1e-6)      # unit away

        mask_pp = dist_pp < cfg.interaction_radius
        mag_pp = cfg.A_ped * np.exp(-eff_dist / cfg.B_ped) * mask_pp
        f_ped = (mag_pp[:, :, None] * n_pp).sum(axis=1)             # (n, 2)

        # -- 3. obstacle repulsion --
        f_obs = np.zeros((n, 2))
        if obstacle_positions_ue is not None and obstacle_positions_ue.shape[0] > 0:
            m = obstacle_positions_ue.shape[0]
            diff_po = (positions[:, None, :]
                       - obstacle_positions_ue[None, :, :])          # (n,m,2)
            dist_po = np.linalg.norm(diff_po, axis=2)               # (n,m)
            n_po = diff_po / np.maximum(dist_po[:, :, None], 1e-6)
            mask_po = dist_po < cfg.interaction_radius
            mag_po = cfg.A_obs * np.exp(-dist_po / cfg.B_obs) * mask_po
            f_obs = (mag_po[:, :, None] * n_po).sum(axis=1)         # (n, 2)

        # -- 4. boundary repulsion (punctured walkable mesh) --
        f_boundary = self._compute_boundary_force(positions)

        # -- velocity integration + clamping --
        f_total = f_desired + f_ped + f_obs + f_boundary
        new_vel = velocities + f_total * dt

        speeds = np.linalg.norm(new_vel, axis=1)
        too_fast = speeds > cfg.max_speed
        new_vel[too_fast] *= cfg.max_speed / speeds[too_fast, None]
        speeds[too_fast] = cfg.max_speed

        # ── apply WalkerControl ──
        for i, ped in enumerate(self.pedestrians):
            s = float(speeds[i])
            if s > 0.01:
                d = new_vel[i] / s
                ped.actor.apply_control(carla.WalkerControl(
                    carla.Vector3D(float(d[0]), float(d[1]), 0.0), s))
            else:
                # Idle — zero speed, arbitrary direction
                ped.actor.apply_control(carla.WalkerControl(
                    carla.Vector3D(1.0, 0.0, 0.0), 0.0))

            # Record position for trajectory visualisation
            ped._traj_buf.append((float(positions[i, 0]),
                                  float(positions[i, 1])))

    # ── Queries ───────────────────────────────────────────────────────

    def get_pedestrian_layout(self) -> dict:
        """Return pedestrian positions, trajectories, and destinations.

        All coordinates in standard frame (x_std, z_std).

        Returns
        -------
        dict with key ``'pedestrians'``: list of dict, each with
            'position_std'    : (2,) current position
            'trajectory_std'  : (T, 2) recent trajectory
            'destination_std' : (2,) or None — current nav destination
        """
        if not self.enabled or not self.pedestrians:
            return {'pedestrians': []}

        result = []
        for ped in self.pedestrians:
            loc = ped.actor.get_location()
            pos_std = np.array([loc.y, loc.x])   # UE -> standard

            if ped._traj_buf:
                traj_ue = np.array(list(ped._traj_buf))        # (T, 2) UE
                traj_std = traj_ue[:, ::-1].copy()              # swap columns
            else:
                traj_std = pos_std.reshape(1, 2)

            dest_std = None
            if ped.destination_ue is not None:
                dest_std = np.array([ped.destination_ue[1],
                                     ped.destination_ue[0]])

            result.append({
                'position_std': pos_std,
                'trajectory_std': traj_std,
                'destination_std': dest_std,
            })

        return {'pedestrians': result}

    def get_pedestrian_positions_ue(self) -> np.ndarray:
        """Return (N, 2) pedestrian positions in UE (x, y)."""
        if not self.enabled or not self.pedestrians:
            return np.empty((0, 2))
        out = np.zeros((len(self.pedestrians), 2))
        for i, ped in enumerate(self.pedestrians):
            loc = ped.actor.get_location()
            out[i] = (loc.x, loc.y)
        return out

    def get_actor_ids(self) -> List[int]:
        """Actor IDs for cleanup tracking."""
        return [p.actor.id for p in self.pedestrians]

    # ── Helpers ───────────────────────────────────────────────────────

    def _resample_destination(self, ped: _Pedestrian):
        bounds = self.quadrant_bounds[ped.region_idx]
        loc = self._sample_navmesh_in_bounds(bounds)
        if loc is not None:
            ped.destination_ue = np.array([loc.x, loc.y, loc.z])

    def _sample_navmesh_in_bounds(self, bounds,
                                  max_attempts=50,
                                  sidewalk_only: bool = True,
                                  ) -> Optional[carla.Location]:
        # Try precomputed cache first (prefer sidewalk-only points)
        if (self._navmesh_cache is not None
                and self._current_town is not None
                and self._navmesh_cache.has_town(self._current_town)):
            if sidewalk_only:
                pt = self._navmesh_cache.sample_sidewalk_in_bounds_ue(
                    self._current_town, bounds)
            else:
                pt = self._navmesh_cache.sample_in_bounds_ue(
                    self._current_town, bounds)
            if pt is not None:
                return carla.Location(
                    x=float(pt[0]), y=float(pt[1]), z=float(pt[2]))
            # Cache exists but no suitable points — don't fall back to
            # unfiltered runtime sampling which includes roads.
            if sidewalk_only:
                return None

        # Fallback: runtime sampling (no area filtering available)
        xlo, xhi, ylo, yhi = bounds
        for _ in range(max_attempts):
            loc = self.world.get_random_location_from_navigation()
            if (loc is not None
                    and xlo <= loc.x <= xhi
                    and ylo <= loc.y <= yhi):
                return loc
        return None
