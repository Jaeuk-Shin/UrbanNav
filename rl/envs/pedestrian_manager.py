"""
Social Force Model (SFM) pedestrian manager for CARLA.

Spawns and controls NPC pedestrians using the Helbing & Molnar (1995)
social force model.  Unlike CARLA's built-in AI walker controller, this
handles dynamically spawned obstacles that are absent from the navmesh.

Forces applied per pedestrian:
  - Desired    : drives toward the next waypoint along a Detour navmesh path
  - Ped–ped    : repulsion from other pedestrians
  - Obstacle   : repulsion from spawned static obstacles
  - NO ego     : the RL agent must learn to avoid pedestrians itself

Detour navmesh pathfinding
~~~~~~~~~~~~~~~~~~~~~~~~~~
When a Detour ``.bin`` navmesh is available, the desired force follows a
sidewalk-only path computed by the C++ Detour extension.  Pedestrians
navigate through intermediate waypoints and only cross roads at
crosswalks.  When Detour is unavailable, falls back to straight-line
desired force (original behaviour).
"""

import math
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import carla


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

    # Navigation
    dest_reach_threshold: float = 2.0   # resample when this close (m)
    waypoint_advance_threshold: float = 1.5  # advance to next waypoint (m)
    interaction_radius: float = 5.0     # ignore forces beyond this (m)


# ── Per-pedestrian state ──────────────────────────────────────────────


class _Pedestrian:
    __slots__ = ("actor", "desired_speed", "region_idx", "destination_ue",
                 "_traj_buf", "_waypoints_ue", "_wp_idx")

    _TRAJ_MAXLEN = 200   # keep last N positions for visualisation

    def __init__(self, actor, desired_speed: float, region_idx: int):
        self.actor = actor
        self.desired_speed = desired_speed
        self.region_idx = region_idx
        self.destination_ue: Optional[np.ndarray] = None   # (3,) UE xyz
        self._traj_buf: deque = deque(maxlen=_Pedestrian._TRAJ_MAXLEN)
        self._waypoints_ue: Optional[np.ndarray] = None    # (W, 3) Detour path
        self._wp_idx: int = 0


# ── Manager ───────────────────────────────────────────────────────────


class PedestrianManager:
    """Spawn and control SFM pedestrians on a single CARLA world.

    Lifecycle::

        mgr = PedestrianManager(config)
        mgr.initialize(world, bp_lib, quadrant_bounds,
                        obstacle_layout=obstacle_layout)  # loads Detour, spawns
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
        # Detour navmesh pathfinder (populated by initialize())
        self._detour = None

    # ── Lifecycle ─────────────────────────────────────────────────────

    def initialize(self, world, bp_lib, quadrant_bounds,
                   navmesh_cache=None, town=None,
                   obstacle_layout=None):
        """Spawn pedestrians.  Call once after map load + region split.

        Parameters
        ----------
        obstacle_layout : dict or None
            From ``ObstacleManager.get_obstacle_layout()``.  When provided
            together with a Detour navmesh, obstacle polygons are blocked
            so pedestrian paths route around them.
        """
        if not self.enabled:
            return
        self._navmesh_cache = navmesh_cache
        self._current_town = town
        self.world = world
        self.bp_lib = bp_lib
        self.quadrant_bounds = quadrant_bounds

        # Load Detour navmesh pathfinder (optional)
        self._detour = self._load_detour()

        # Block navmesh polygons under obstacles before spawning
        if obstacle_layout is not None:
            self._block_obstacle_polygons(obstacle_layout)

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

    # ── Detour navmesh integration ────────────────────────────────────

    def _load_detour(self):
        """Try to load a Detour pathfinder for sidewalk-only pathfinding.

        Uses the navmesh cache directory to find ``.bin`` files.
        Returns a ``DetourPathfinder`` or ``None``.
        """
        if self._navmesh_cache is None or self._current_town is None:
            return None
        cache_dir = self._navmesh_cache.cache_dir
        if cache_dir is None:
            return None

        try:
            import detour_nav
        except ImportError:
            print("  [PedestrianManager] detour_nav not built — "
                  "using straight-line fallback. "
                  "Build with: cd detour_nav && bash build.sh")
            return None

        bin_path = os.path.join(cache_dir, f"{self._current_town}.bin")
        if not os.path.exists(bin_path):
            print(f"  [PedestrianManager] No .bin navmesh for "
                  f"{self._current_town} in {cache_dir}")
            return None

        pf = detour_nav.DetourPathfinder()
        if not pf.load(bin_path):
            print(f"  [PedestrianManager] Failed to load {bin_path}")
            return None

        pf.set_sidewalk_only()
        print(f"  [PedestrianManager] Detour loaded: "
              f"{self._current_town}.bin — sidewalk-only paths")
        return pf

    def _block_obstacle_polygons(self, obstacle_layout: dict):
        """Block navmesh polygons under spawned obstacles.

        Call after obstacles are spawned so Detour paths route around them.
        """
        if self._detour is None:
            return
        self._detour.unblock_all()

        total_blocked = 0
        margin = self.config.ped_radius
        for obs in obstacle_layout.get('obstacles', []):
            corners_std = obs['corners_std']         # (4, 2) standard frame
            corners_ue = corners_std[:, ::-1].copy()  # std → UE: swap columns

            ue_x_min, ue_x_max = corners_ue[:, 0].min(), corners_ue[:, 0].max()
            ue_y_min, ue_y_max = corners_ue[:, 1].min(), corners_ue[:, 1].max()

            # AABB centre + half-extents in Recast frame
            cx = (ue_x_min + ue_x_max) / 2
            cz = (ue_y_min + ue_y_max) / 2
            cy = 0.0                                  # ground level
            half_ex = (ue_x_max - ue_x_min) / 2 + margin
            half_ey = 4.0                             # generous vertical
            half_ez = (ue_y_max - ue_y_min) / 2 + margin

            # UE → Recast: (ue_x, ue_z, ue_y)
            total_blocked += self._detour.block_polygons_in_aabb(
                cx, cy, cz, half_ex, half_ey, half_ez)

        if total_blocked > 0:
            print(f"  [PedestrianManager] Blocked {total_blocked} navmesh "
                  f"polygons under obstacles")

    def _compute_detour_path(self, ped: _Pedestrian) -> bool:
        """Compute a Detour sidewalk path from current position to destination.

        Stores waypoints on ``ped._waypoints_ue`` and resets ``ped._wp_idx``.
        Returns True if a valid path was found.
        """
        if self._detour is None or ped.destination_ue is None:
            ped._waypoints_ue = None
            return False

        loc = ped.actor.get_location()
        dest = ped.destination_ue

        # UE (x, y, z) → Recast (x, z, y)
        sx, sy, sz = float(loc.x), float(loc.z), float(loc.y)
        ex, ey, ez = float(dest[0]), float(dest[2]), float(dest[1])

        dist, waypoints = self._detour.find_path(sx, sy, sz, ex, ey, ez)

        if math.isinf(dist) or len(waypoints) < 2:
            ped._waypoints_ue = None
            return False

        # Recast (rx, ry, rz) → UE (rx, rz, ry)
        wps_ue = np.array([[wp[0], wp[2], wp[1]] for wp in waypoints])
        ped._waypoints_ue = wps_ue
        ped._wp_idx = 1   # skip start point (current position)
        return True

    def _get_desired_target_ue_2d(self, ped: _Pedestrian,
                                  pos_ue_2d: np.ndarray) -> np.ndarray:
        """Return the 2D UE (ue_x, ue_y) point the pedestrian should steer toward.

        Follows Detour waypoints if available, otherwise straight-line to dest.
        """
        # Straight-line fallback
        if ped._waypoints_ue is None or ped._wp_idx >= len(ped._waypoints_ue):
            if ped.destination_ue is not None:
                return ped.destination_ue[:2]
            return pos_ue_2d  # no destination — stay in place

        cfg = self.config
        wp = ped._waypoints_ue[ped._wp_idx, :2]   # 2D UE
        dist = np.linalg.norm(pos_ue_2d - wp)

        if dist < cfg.waypoint_advance_threshold:
            ped._wp_idx += 1
            if ped._wp_idx >= len(ped._waypoints_ue):
                # Reached end of path — target final destination
                return ped.destination_ue[:2]
            wp = ped._waypoints_ue[ped._wp_idx, :2]

        return wp

    # ── Spawning ──────────────────────────────────────────────────────

    def _spawn_all(self):
        cfg = self.config
        walker_bps = list(self.bp_lib.filter('walker.pedestrian.*'))
        if not walker_bps:
            print("  [PedestrianManager] WARNING: no walker blueprints found")
            return

        num_regions = len(self.quadrant_bounds)
        n_detour_paths = 0
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
                    if self._compute_detour_path(ped):
                        n_detour_paths += 1

                self.pedestrians.append(ped)

        print(f"  [PedestrianManager] Spawned {len(self.pedestrians)} "
              f"SFM pedestrians ({cfg.num_per_region}/region)")
        if self._detour is not None:
            print(f"  [PedestrianManager] {n_detour_paths}/"
                  f"{len(self.pedestrians)} initial Detour paths computed")

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

        # -- 1. desired force toward waypoint target: (v0 * e_d − v) / τ --
        targets = np.zeros((n, 2))
        for i, ped in enumerate(self.pedestrians):
            targets[i] = self._get_desired_target_ue_2d(ped, positions[i])

        diff_target = targets - positions
        d2t = np.linalg.norm(diff_target, axis=1, keepdims=True)
        d2t = np.maximum(d2t, 1e-6)
        e_d = diff_target / d2t
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

        # -- velocity integration + clamping --
        f_total = f_desired + f_ped + f_obs
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

    def predict_trajectories_std(
        self,
        horizon_steps: int,
        dt: float,
        obstacle_positions_ue: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Forward-simulate SFM pedestrian trajectories in pure numpy.

        Runs the same force model as :meth:`update` but without touching
        CARLA actors — uses current actor state as initial conditions and
        integrates forward for *horizon_steps* steps.

        Uses the current Detour waypoint target as the desired direction
        for the entire prediction horizon (a good approximation for short
        horizons since waypoints represent straight segments).

        Parameters
        ----------
        horizon_steps : int
            Number of simulation steps to predict.
        dt : float
            Time step per simulation step (seconds).
        obstacle_positions_ue : (M, 2) or None
            Static obstacle positions in UE frame.

        Returns
        -------
        positions_std : (horizon_steps, N, 2)
            Predicted pedestrian positions in standard frame ``(x_std, z_std)``.
            Empty ``(horizon_steps, 0, 2)`` if no pedestrians.
        """
        if not self.enabled or not self.pedestrians:
            return np.empty((max(horizon_steps, 1), 0, 2))

        cfg = self.config
        n = len(self.pedestrians)

        # -- gather initial state from CARLA actors --
        pos = np.zeros((n, 2))
        vel = np.zeros((n, 2))
        targets = np.zeros((n, 2))
        speeds = np.zeros(n)
        for i, ped in enumerate(self.pedestrians):
            loc = ped.actor.get_location()
            pos[i] = (loc.x, loc.y)
            v = ped.actor.get_velocity()
            vel[i] = (v.x, v.y)
            # Use current waypoint target (peek without advancing)
            targets[i] = self._peek_desired_target_ue_2d(ped, pos[i])
            speeds[i] = ped.desired_speed

        # -- forward simulate --
        out = np.empty((horizon_steps, n, 2))
        for t in range(horizon_steps):
            out[t] = pos

            # destination reached → decelerate in place
            d2t_vec = targets - pos
            d2t = np.linalg.norm(d2t_vec, axis=1, keepdims=True)
            d2t = np.maximum(d2t, 1e-6)
            arrived = (d2t.squeeze(1) < cfg.dest_reach_threshold)

            # 1. desired force
            e_d = d2t_vec / d2t
            f_desired = (speeds[:, None] * e_d - vel) / cfg.tau
            f_desired[arrived] = -vel[arrived] / cfg.tau

            # 2. ped–ped repulsion
            diff_pp = pos[:, None, :] - pos[None, :, :]
            dist_pp = np.linalg.norm(diff_pp, axis=2)
            np.fill_diagonal(dist_pp, np.inf)
            eff_dist = np.maximum(dist_pp - 2 * cfg.ped_radius, 0.0)
            n_pp = diff_pp / np.maximum(dist_pp[:, :, None], 1e-6)
            mask_pp = dist_pp < cfg.interaction_radius
            mag_pp = cfg.A_ped * np.exp(-eff_dist / cfg.B_ped) * mask_pp
            f_ped = (mag_pp[:, :, None] * n_pp).sum(axis=1)

            # 3. obstacle repulsion
            f_obs = np.zeros((n, 2))
            if (obstacle_positions_ue is not None
                    and obstacle_positions_ue.shape[0] > 0):
                diff_po = (pos[:, None, :]
                           - obstacle_positions_ue[None, :, :])
                dist_po = np.linalg.norm(diff_po, axis=2)
                n_po = diff_po / np.maximum(dist_po[:, :, None], 1e-6)
                mask_po = dist_po < cfg.interaction_radius
                mag_po = cfg.A_obs * np.exp(-dist_po / cfg.B_obs) * mask_po
                f_obs = (mag_po[:, :, None] * n_po).sum(axis=1)

            # integrate
            f_total = f_desired + f_ped + f_obs
            vel = vel + f_total * dt
            spd = np.linalg.norm(vel, axis=1)
            too_fast = spd > cfg.max_speed
            vel[too_fast] *= cfg.max_speed / spd[too_fast, None]
            pos = pos + vel * dt

        # Convert UE (ue_x, ue_y) → standard (x_std, z_std) = (ue_y, ue_x)
        out_std = out[:, :, ::-1].copy()
        return out_std

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

    def _peek_desired_target_ue_2d(self, ped: _Pedestrian,
                                   pos_ue_2d: np.ndarray) -> np.ndarray:
        """Like ``_get_desired_target_ue_2d`` but never advances waypoint index.

        Used by ``predict_trajectories_std`` to snapshot the current target
        without side effects.
        """
        if ped._waypoints_ue is None or ped._wp_idx >= len(ped._waypoints_ue):
            if ped.destination_ue is not None:
                return ped.destination_ue[:2]
            return pos_ue_2d
        return ped._waypoints_ue[ped._wp_idx, :2]

    def _resample_destination(self, ped: _Pedestrian):
        bounds = self.quadrant_bounds[ped.region_idx]
        loc = self._sample_navmesh_in_bounds(bounds)
        if loc is not None:
            ped.destination_ue = np.array([loc.x, loc.y, loc.z])
            if not self._compute_detour_path(ped):
                ped._waypoints_ue = None

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
