"""
Multi-agent CARLA environment.

Runs N independent navigation episodes on a **single** CARLA server by
splitting the town's walkable area into N equal regions (quadrants for N=4)
and spawning one ego agent per region.

The interface mirrors VecCarlaEnv so it can be used as a drop-in replacement
in the PPO trainer (same step / reset / capture_bev / close API).
"""

import time
import math
import pathlib
import random
import json
from collections import deque
from typing import List
from queue import Empty, Queue

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
import gymnasium as gym

import carla
from carla_utils.tf import UE
from rl.envs.mpc.mpc import MPC
from rl.envs.obstacle_manager import ObstacleManager, ObstacleConfig
from rl.envs.pedestrian_manager import PedestrianManager, PedestrianConfig
from rl.utils.geodesic import GeodesicDistanceField
from rl.utils.navmesh_cache import NavmeshCache


# ── Helpers ────────────────────────────────────────────────────────────


def _load_sensor_configs(filename) -> List[dict]:
    sensor_cfg_path = pathlib.Path(__file__).parent / 'assets' / filename
    with open(sensor_cfg_path, 'r') as f:
        sensor_cfg = json.load(f)
    return sensor_cfg['sensors']


def _to_walker_control(vx, vz, c2w) -> carla.WalkerControl:
    direction_cam = np.array([vx, 0., vz])
    c2w_R = R.from_quat(c2w[3:])
    direction = c2w_R.apply(direction_cam)
    x, y, z = direction
    x, y, z = z, x, -y          # standard -> UE
    norm = (x**2 + y**2 + z**2)**0.5 + 1e-10
    x, y, z = x / norm, y / norm, z / norm
    return carla.WalkerControl(
        carla.Vector3D(float(x), float(y), float(z)),
        float((vx**2 + vz**2)**0.5),
    )

def sample_point_from_annulus(r_min, r_max):
    # uniform sampling from an annulus
    th = np.pi * (2. * np.random.rand() - 1.)
    p = np.random.rand()
    r = (r_max**2 * (1. - p) + r_min**2 * p) ** 0.5
    return r * np.array([np.cos(th), np.sin(th)])


def _repeat_and_shift(data, repeats, shifts):
    """Upsample waypoints along the temporal axis for MPC horizon alignment.

    (*, data_size, data_dim) -> (*, repeats * data_size, data_dim)
    then cyclically shift by *shifts* positions.
    """
    n_waypoints = data.shape[-2]
    data_rep = np.repeat(data, repeats=repeats, axis=-2)
    data_shifted = np.concatenate(
        (data_rep[..., shifts:, :],
         data_rep[..., repeats * n_waypoints - shifts:, :]),
        axis=-2)
    return data_shifted


def _to_walker_control_mpc(velocities, c2w, dt) -> carla.WalkerControl:
    """Convert MPC unicycle control (lin_v, ang_v) to CARLA WalkerControl."""
    lin_v, ang_v = velocities
    th_next = 0.5 * np.pi + dt * ang_v
    c, s = np.cos(th_next), np.sin(th_next)
    direction_cam = np.array([c, 0., s])
    c2w_R = R.from_quat(c2w[3:])
    direction = c2w_R.apply(direction_cam)
    x, y, z = direction
    x, y, z = z, x, -y        # standard -> UE
    norm = (x**2 + y**2 + z**2)**0.5 + 1e-10
    x, y, z = x / norm, y / norm, z / norm
    return carla.WalkerControl(
        carla.Vector3D(float(x), float(y), float(z)),
        float(lin_v),
    )   


# ── Multi-agent environment ───────────────────────────────────────────


class CarlaMultiAgentEnv:
    """
    Manages N independent navigation agents on a single CARLA server.

    The town's walkable area is split into N equal regions (quadrants for N=4).
    Each agent spawns and navigates within its assigned region.  A single
    ``world.tick()`` advances **all** agents simultaneously, making this much
    cheaper than running N separate CARLA servers.

    Public interface matches ``VecCarlaEnv``:
        reset()  -> (obs_dict, info_list)
        step(actions)  -> (obs_dict, rewards, terminateds, truncateds, infos)
        capture_bev(env_idx, ...)  -> (img, meta)
        set_collect_substep_frames(enabled)
        close()
        num_envs  (property)
    """

    def __init__(
            self, cfg, port=2000, num_agents=4, max_speed=1.4, fps=5,
            gamma=0.99, teleport=False, goal_range=8.0,
            max_episode_steps=32, towns=None, map_change_interval=0,
            obstacle_config=None, pedestrian_config=None,
            navmesh_cache: NavmeshCache | None = None,
            quadrant_margin=None,
            randomize_weather=False,
            use_mpc=False,
            ):
        assert 1 <= num_agents <= 4, "Currently supports 1-4 agents (quadrants)"

        self.num_agents = num_agents
        self.towns = towns              # e.g. ['Town02', 'Town03', 'Town05', 'Town10']
        self._current_town = None
        self._navmesh_cache = navmesh_cache   # optional NavmeshCache instance
        self.map_change_interval = map_change_interval   # 0 = no mid-training changes
        self._total_episodes = 0
        self.fps = fps                  # carla world tick fps
        self.dt = 1. / fps              # carla world tick dt
        self.n_skips = int(fps // cfg.data.target_fps)      # number of ticks per step
        self.port = port
        self.teleport = teleport
        self.use_mpc = use_mpc
        self.max_speed = max_speed
        self.gamma = gamma
        self.goal_range = goal_range
        self.quadrant_margin = quadrant_margin if quadrant_margin is not None else goal_range
        self.randomize_weather = randomize_weather
        self._current_weather = None       # set by _randomize_weather()
        self.max_episode_steps = max_episode_steps
        self.history_length = cfg.model.obs_encoder.context_size
        self.future_length = cfg.model.decoder.len_traj_pred

        width = cfg.data.width
        height = cfg.data.height

        # Per-agent gym spaces (for reference; the vectorised API stacks them)
        self.action_space = gym.spaces.Box(
            low=-100., high=100., shape=(self.future_length * 2,), dtype=np.float32)

        # MPC: created once and kept alive across resets (JIT compiles on
        # first solve, so re-creating would incur non-negligible overhead).
        if self.use_mpc:
            mpc_horizon = self.future_length * self.n_skips
            ulb = np.array([-max_speed, -0.8])
            uub = np.array([max_speed, 0.8])
            max_wall_time = 2.0 * self.dt
            self._mpc = MPC(mpc_horizon, self.dt, ulb, uub,
                            max_wall_time=max_wall_time)
        else:
            self._mpc = None
        self._last_mpc_solve_time = 0.0
        self._last_sim_tick_time = 0.0
        self.observation_space = gym.spaces.Dict({
            'obs': gym.spaces.Box(0., 255., (self.history_length, height, width, 3), np.uint8),
            'cord': gym.spaces.Box(-100., 100., (self.history_length * 2,), np.float32),
            'goal': gym.spaces.Box(-100., 100., (2,), np.float32),
        })

        self._collect_substep_frames = False
        self._substep_frames: List[list] = [[] for _ in range(num_agents)]

        # CARLA handles (populated on reset)
        self.client = None
        self.world = None
        self.bp_lib = None      # blueprint library
        self.tm = None          # traffic manager

        # Per-agent state (populated on reset)
        self.robots: List = [None] * num_agents
        self.rgb_buffers: List[deque] = []
        self.pose_buffers: List[deque] = []                 # 7-dim camera-to-world (in the standard coordinate system)
        self.sensor_queues: List[dict] = [{}] * num_agents
        self.c2r: List[dict] = [{}] * num_agents            # camera-to-robot (camera rigidly attached to the robot)
        self.sensors: List[list] = [[]] * num_agents
        self.goal_globals = [None] * num_agents             # 2d goal position (standard world coordinate)
        self._goal_methods = [''] * num_agents             # how each goal was sampled
        self.initial_distances = [0.] * num_agents
        self.initial_geodesic_distances = [0.] * num_agents
        self.path_lengths = [0.] * num_agents
        self.step_counts = [0] * num_agents
        self.all_actor_ids: List[int] = []
        self.quadrant_bounds = None

        # Procedural obstacle generation
        self.obstacle_mgr = ObstacleManager(obstacle_config)
        self._obstacle_ids: List[int] = []
        self._challenge_actor_ids: List[List[int]] = [
            [] for _ in range(num_agents)]  # per-agent crosswalk challenge actors

        # SFM pedestrians
        self.ped_mgr = PedestrianManager(pedestrian_config)

        # Per-quadrant geodesic distance grids (initialised per-map when
        # navmesh cache provides walkable triangles; distance fields per-episode)
        self._geo_grids: list = [None] * num_agents
        self._geo_dists: list = [None] * num_agents
        self._geo_paths: list = [None] * num_agents   # traced geodesic paths (std coords)

        # Episode solvability tracking (cumulative across auto-resets)
        self._max_goal_retries = 5
        self._goal_retries_total = 0      # total retries across all episodes
        self._unsolvable_episodes = 0     # episodes that remained unsolvable after retries
        self._solvable_episodes = 0       # episodes confirmed solvable
        self._last_retries = [0] * num_agents  # retries for the most recent episode per agent

    # ── VecCarlaEnv-compatible property ───────────────────────────────

    @property
    def num_envs(self):
        return self.num_agents

    # ── Region splitting ──────────────────────────────────────────────

    def _compute_regions(self, n_samples=1000, n_navmesh_samples=10000):
        """Sample navmesh points to determine walkable extent, then split into
        quadrants in UE coordinates (Location.x, Location.y).

        Also caches navmesh points per region as a KD-tree (in standard
        coordinates) for efficient navigable goal sampling.

        Results are cached per town — repeated calls for the same map are free.
        """
        if self.quadrant_bounds is not None:
            return

        # Try precomputed cache first
        used_cache = False
        cache = self._navmesh_cache
        if (cache is not None and self._current_town is not None
                and cache.has_town(self._current_town)):
            pts_ue_3d = cache.get_walkable_points_ue(self._current_town)
            if pts_ue_3d is not None and len(pts_ue_3d) > 0:
                points_ue = pts_ue_3d[:, :2]   # (N, 2) UE (x, y)
                print(f"  [NavmeshCache] Using {len(points_ue)} cached "
                      f"walkable points for region splitting")
                used_cache = True

        if not used_cache:
            # Fallback: runtime sampling
            total = max(n_samples, n_navmesh_samples)
            points_ue = []
            for _ in range(total):
                loc = self.world.get_random_location_from_navigation()
                if loc is not None:
                    points_ue.append((loc.x, loc.y))
            points_ue = np.array(points_ue)

        center_x = np.median(points_ue[:, 0])
        center_y = np.median(points_ue[:, 1])
        # Small padding so boundary points are always included
        x_min, x_max = points_ue[:, 0].min() - 1., points_ue[:, 0].max() + 1.
        y_min, y_max = points_ue[:, 1].min() - 1., points_ue[:, 1].max() + 1.

        # 4 quadrants: (ue_x_lo, ue_x_hi, ue_y_lo, ue_y_hi)
        self.quadrant_bounds = [
            (x_min, center_x, y_min, center_y),   # lower-left
            (x_min, center_x, center_y, y_max),   # lower-right
            (center_x, x_max, y_min, center_y),   # upper-left
            (center_x, x_max, center_y, y_max),   # upper-right
        ]

        # Inner bounds: quadrant shrunk by margin so that both spawn and goal
        # stay well inside the quadrant, avoiding edge-crossing geodesic paths.
        m = self.quadrant_margin
        self.quadrant_inner_bounds = [
            (xlo + m, xhi - m, ylo + m, yhi - m)
            for xlo, xhi, ylo, yhi in self.quadrant_bounds
        ]

        # Store full navmesh points for last-resort spawn fallback
        self._all_navmesh_points_ue = points_ue  # (N, 2) UE (x, y)

        # Use sidewalk-only points for goal sampling / KD-trees so that
        # goals and spawns land on sidewalks rather than roads or grass.
        sw_points_ue = None
        if (cache is not None and self._current_town is not None
                and cache.has_town(self._current_town)):
            sw_pts_3d = cache.get_sidewalk_points_ue(self._current_town)
            if sw_pts_3d is not None and len(sw_pts_3d) > 0:
                sw_points_ue = sw_pts_3d[:, :2]
                print(f"  [NavmeshCache] Using {len(sw_points_ue)} "
                      f"sidewalk-only points for goal sampling KD-trees")
        # Fall back to all walkable if no sidewalk data (old cache format)
        if sw_points_ue is not None:
            goal_points_ue = sw_points_ue
        else:
            print("  WARNING: No sidewalk-only points in cache — "
                  "KD-trees will include roads. Rebuild cache with "
                  "latest export_carla_navmesh.py")
            goal_points_ue = points_ue

        # Build per-region KD-trees in standard coordinates for goal sampling.
        # Points are filtered to the *inner* bounds so that sampled goals
        # are guaranteed to be at least `quadrant_margin` from the boundary.
        # UE (x, y): standard (x_std, z_std) = (ue_y, ue_x)
        self._navmesh_trees: List[cKDTree] = []
        self._navmesh_points_std: List[np.ndarray] = []
        self._navmesh_points_ue_full = []  # per-region, filtered by FULL bounds
        for i in range(self.num_agents):
            # Inner bounds (for goal sampling KD-tree) — sidewalk-only
            xlo, xhi, ylo, yhi = self.quadrant_inner_bounds[i]
            mask = (
                (goal_points_ue[:, 0] >= xlo) & (goal_points_ue[:, 0] <= xhi) &
                (goal_points_ue[:, 1] >= ylo) & (goal_points_ue[:, 1] <= yhi)
            )
            region_ue = goal_points_ue[mask]
            # Convert to standard: x_std = ue_y, z_std = ue_x
            region_std = np.stack([region_ue[:, 1], region_ue[:, 0]], axis=1)
            self._navmesh_points_std.append(region_std)
            self._navmesh_trees.append(cKDTree(region_std))

            # Full bounds (for spawn fallback — relaxed constraint)
            fxlo, fxhi, fylo, fyhi = self.quadrant_bounds[i]
            fmask = (
                (goal_points_ue[:, 0] >= fxlo) & (goal_points_ue[:, 0] <= fxhi) &
                (goal_points_ue[:, 1] >= fylo) & (goal_points_ue[:, 1] <= fyhi)
            )
            self._navmesh_points_ue_full.append(goal_points_ue[fmask])

        print(f"  Map bounds (UE): x=[{x_min:.1f}, {x_max:.1f}], "
              f"y=[{y_min:.1f}, {y_max:.1f}]")
        print(f"  Split center: ({center_x:.1f}, {center_y:.1f}), "
              f"margin: {m:.1f}m")
        for i in range(self.num_agents):
            xlo, xhi, ylo, yhi = self.quadrant_bounds[i]
            ilo, ihi, jlo, jhi = self.quadrant_inner_bounds[i]
            print(f"  Region {i}: ue_x=[{xlo:.1f}, {xhi:.1f}], "
                  f"ue_y=[{ylo:.1f}, {yhi:.1f}], "
                  f"inner ue_x=[{ilo:.1f}, {ihi:.1f}], "
                  f"inner ue_y=[{jlo:.1f}, {jhi:.1f}], "
                  f"navmesh_pts={len(self._navmesh_points_std[i])}")

    def _sample_location_in_region(self, region_idx, max_attempts=200):
        """Sample a navmesh location for an agent spawn in *region_idx*.

        Uses **progressive bound relaxation** so that a valid location is
        always returned (never raises):

        1. Inner bounds + navmesh cache
        2. Inner bounds + per-region cached points (2D, z=0)
        3. Full quadrant bounds + per-region cached points
        4. Any cached navmesh point on the map
        5. Live runtime random sampling (last resort)
        """
        inner = self.quadrant_inner_bounds[region_idx]
        full = self.quadrant_bounds[region_idx]

        # -- Tier 1: navmesh cache, inner bounds (best quality) --
        # Prefer sidewalk-only points; fall back to all walkable.
        cache = self._navmesh_cache
        if (cache is not None and self._current_town is not None
                and cache.has_town(self._current_town)):
            pt = cache.sample_sidewalk_in_bounds_ue(self._current_town, inner)
            if pt is not None:
                return carla.Location(
                    x=float(pt[0]), y=float(pt[1]), z=float(pt[2]))
            # Relax to full quadrant bounds
            pt = cache.sample_sidewalk_in_bounds_ue(self._current_town, full)
            if pt is not None:
                return carla.Location(
                    x=float(pt[0]), y=float(pt[1]), z=float(pt[2]))

        # -- Tier 2: per-region cached points, inner bounds ──
        if (hasattr(self, '_navmesh_points_std')
                and region_idx < len(self._navmesh_points_std)
                and len(self._navmesh_points_std[region_idx]) > 0):
            pts_std = self._navmesh_points_std[region_idx]
            pt = pts_std[np.random.randint(len(pts_std))]
            return carla.Location(x=float(pt[1]), y=float(pt[0]), z=0.0)

        # -- Tier 3: per-region cached points, full quadrant bounds --
        if (hasattr(self, '_navmesh_points_ue_full')
                and region_idx < len(self._navmesh_points_ue_full)
                and len(self._navmesh_points_ue_full[region_idx]) > 0):
            pts = self._navmesh_points_ue_full[region_idx]
            pt = pts[np.random.randint(len(pts))]
            return carla.Location(x=float(pt[0]), y=float(pt[1]), z=0.0)

        # -- Tier 4: any navmesh point on the whole map --
        if (hasattr(self, '_all_navmesh_points_ue')
                and len(self._all_navmesh_points_ue) > 0):
            pt = self._all_navmesh_points_ue[
                np.random.randint(len(self._all_navmesh_points_ue))]
            print(f"  WARNING: region {region_idx} has no navmesh points "
                  f"in bounds, using map-wide fallback")
            return carla.Location(x=float(pt[0]), y=float(pt[1]), z=0.0)

        # -- Tier 5: live runtime sampling (no cached data at all) --
        x_min, x_max, y_min, y_max = full  # use full bounds, not inner
        for _ in range(max_attempts):
            loc = self.world.get_random_location_from_navigation()
            if (loc is not None
                    and x_min <= loc.x <= x_max
                    and y_min <= loc.y <= y_max):
                return loc
        # Absolute last resort: any navmesh point, ignore bounds
        for _ in range(max_attempts):
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                print(f"  WARNING: region {region_idx} spawn fell back "
                      f"to unbounded navmesh sample")
                return loc
        raise RuntimeError(
            f"Could not sample any navmesh location after "
            f"{max_attempts * 2} attempts — is the navmesh loaded?")

    # ── Coordinate helpers ────────────────────────────────────────────

    def _camera2world(self, agent_idx, sensor_id='fcam'):
        """
        7-dim camera-to-world pose in standard coordinates
            standard - x: right / z: front / y: down
            UE       - x: front / y: right / z: up
        """
        r2w = np.array(self.robots[agent_idx].get_transform().get_matrix())     # robot-to-world (in UE coordinate system)
        c2w = r2w @ self.c2r[agent_idx][sensor_id]                              # camera-to-world
        c2w = UE @ c2w @ UE.T                                                   # to standard coorinate system (right-handed)
        # 3d position of the camera in the (standard) world frame

        xyz = c2w[:3, -1]               
        q = R.from_matrix(c2w[:3, :3]).as_quat()        # [qx, qy, qz, qw]
        return np.concatenate((xyz, q))

    def _get_pose(self, agent_idx):
        # standard camera-to-world
        return np.copy(self.pose_buffers[agent_idx][-1])

    def _get_xz(self, agent_idx):
        """camera position on the standard xz-plane: [x_std, z_std]"""
        p = self._get_pose(agent_idx)
        return np.array([p[0], p[2]])

    def _goal_cam(self, agent_idx):
        """goal in the camera frame: [x_cam, z_cam]"""
        goal_world = self.goal_globals[agent_idx] - self._get_xz(agent_idx)     # goal pos w.r.t. ego (but orientation w.r.t. world)
        c2w_R = R.from_quat(self._get_pose(agent_idx)[3:])
        g = c2w_R.inv().apply(np.array([goal_world[0], 0., goal_world[1]]))
        return np.array([g[0], g[2]])

    def _distance_to_goal(self, agent_idx):
        g = self._goal_cam(agent_idx)
        return float(np.sqrt(g @ g))

    def _potential(self, agent_idx):
        return -self._distance_to_goal(agent_idx)

    # ── Geodesic distance ─────────────────────────────────────────────

    def _setup_ped_walkable_mesh(self):
        """Pass walkable triangles + obstacle OBBs to the PedestrianManager.

        This enables the boundary constraint that keeps SFM pedestrians
        within the walkable region and away from obstacle interiors.
        Requires a navmesh cache with walkable triangles.
        """
        if not self.ped_mgr.enabled:
            return
        if self._navmesh_cache is None or self._current_town is None:
            return
        walkable_tris = self._navmesh_cache.get_sidewalk_crosswalk_tris_std(
            self._current_town)
        if walkable_tris is None:
            # Fall back to all walkable tris (old cache without per-area data)
            walkable_tris = self._navmesh_cache.get_walkable_tris_std(
                self._current_town)
        if walkable_tris is None:
            return

        # Collect obstacle OBB corners in standard coords
        layout = self.obstacle_mgr.get_obstacle_layout()
        obs_corners = [o['corners_std'] for o in layout['obstacles']]

        self.ped_mgr.set_walkable_mesh(walkable_tris,
                                       obs_corners if obs_corners else None)

    def _init_geodesic_grid(self):
        """Create per-quadrant rasterized geodesic grids from the navmesh cache.

        Each quadrant gets its own grid built from triangles within the
        quadrant bounds plus ``quadrant_margin`` padding — much smaller
        than the full map (~4x fewer cells for 4 quadrants).

        Called once per map load.  If the cache does not contain walkable
        triangles (old cache format), geodesic distance is unavailable and
        the environment falls back to Euclidean potential.
        """
        self._geo_grids = [None] * self.num_agents
        if self._navmesh_cache is None or self._current_town is None:
            return
        # Use sidewalk + crosswalk triangles only (exclude roads) so that
        # geodesic paths stay on pedestrian-walkable surfaces.
        all_tris = self._navmesh_cache.get_sidewalk_crosswalk_tris_std(
            self._current_town)
        if all_tris is None or len(all_tris) == 0:
            # Fall back to all walkable tris (old cache without per-area data)
            all_tris = self._navmesh_cache.get_walkable_tris_std(
                self._current_town)
            if all_tris is not None and len(all_tris) > 0:
                print("  [Geodesic] WARNING: No per-area triangle data — "
                      "using all walkable tris (may route through roads). "
                      "Rebuild cache with latest export_carla_navmesh.py")
        if all_tris is None or len(all_tris) == 0:
            print("  [Geodesic] No walkable triangle data in cache — "
                  "falling back to Euclidean potential")
            return

        geo_pad = self.quadrant_margin
        for i in range(self.num_agents):
            xlo, xhi, ylo, yhi = self.quadrant_bounds[i]
            # Quadrant bounds are UE (x, y); triangles are standard (x_std, z_std).
            # Standard ↔ UE:  x_std = ue_y,  z_std = ue_x
            x_std_lo, x_std_hi = ylo - geo_pad, yhi + geo_pad
            z_std_lo, z_std_hi = xlo - geo_pad, xhi + geo_pad

            # Include any triangle with at least one vertex in the padded box
            in_x = ((all_tris[:, :, 0] >= x_std_lo)
                    & (all_tris[:, :, 0] <= x_std_hi))
            in_z = ((all_tris[:, :, 1] >= z_std_lo)
                    & (all_tris[:, :, 1] <= z_std_hi))
            in_bounds = (in_x & in_z).any(axis=1)
            quad_tris = all_tris[in_bounds]

            if len(quad_tris) == 0:
                print(f"  [Geodesic] Region {i}: no triangles — skipping")
                continue
            print(f"  [Geodesic] Region {i}: "
                  f"{len(quad_tris)}/{len(all_tris)} tris")
            self._geo_grids[i] = GeodesicDistanceField(
                quad_tris, resolution=1.0)

    def _update_geodesic(self, agent_idx):
        """Recompute the geodesic distance field for *agent_idx*.

        Called after each goal change (reset / auto-reset).  Stamps the
        current obstacle layout onto the walkable grid and runs Dijkstra
        from the agent's new goal.
        """
        if self._geo_grids[agent_idx] is None:
            self._geo_paths[agent_idx] = None
            return
        geo = self._geo_grids[agent_idx]
        obstacles = self.obstacle_mgr.get_obstacle_layout().get('obstacles', [])
        self._geo_dists[agent_idx] = geo.compute_distance_field(
            self.goal_globals[agent_idx], obstacles)
        # Trace the shortest geodesic path from spawn to goal for BEV vis
        self._geo_paths[agent_idx] = geo.trace_path(
            self._geo_dists[agent_idx], self._get_xz(agent_idx))

    def _geodesic_potential(self, agent_idx):
        """Potential based on geodesic (obstacle-aware) distance.

        Falls back to Euclidean potential when geodesic data is unavailable.
        """
        geo = self._geo_grids[agent_idx]
        if geo is not None and self._geo_dists[agent_idx] is not None:
            d = geo.query(self._geo_dists[agent_idx], self._get_xz(agent_idx))
            if np.isfinite(d):
                return -d
        return self._potential(agent_idx)

    def _geodesic_distance_to_goal(self, agent_idx):
        """Geodesic distance to goal, or Euclidean if unavailable."""
        geo = self._geo_grids[agent_idx]
        if geo is not None and self._geo_dists[agent_idx] is not None:
            d = geo.query(self._geo_dists[agent_idx], self._get_xz(agent_idx))
            if np.isfinite(d):
                return d
        return self._distance_to_goal(agent_idx)

    def _is_episode_solvable(self, agent_idx):
        """Check whether the current episode is solvable.

        An episode is unsolvable if:
        1. The goal is geodesically unreachable (disconnected mesh), or
        2. The geodesic distance exceeds the agent's travel budget
           (max_speed * sqrt(2) * max_episode_steps).

        Returns ``(solvable, reason)`` where *reason* is ``''`` when
        solvable, or a short description of why it's not.
        """
        geo = self._geo_grids[agent_idx]
        if geo is None or self._geo_dists[agent_idx] is None:
            # No geodesic grid available — can't check, assume solvable
            return True, ''
        d = geo.query(self._geo_dists[agent_idx], self._get_xz(agent_idx))
        if not np.isfinite(d):
            return False, 'unreachable'
        budget = self.max_speed * math.sqrt(2) * self.max_episode_steps
        if d > budget:
            return False, f'too_far(geodesic={d:.1f}m > budget={budget:.1f}m)'
        return True, ''

    # ── Sensor data ───────────────────────────────────────────────────

    def _get_sensor_data(self, agent_idx, sensor_id='fcam') -> np.ndarray:
        try:
            data = self.sensor_queues[agent_idx][sensor_id].get(timeout=10.0)
        except Empty:
            # Sensor missed a frame — kick the sim and retry once
            self.world.tick()
            data = self.sensor_queues[agent_idx][sensor_id].get(timeout=10.0)
        img = np.frombuffer(data.raw_data, dtype=np.uint8)
        img = img.reshape((data.height, data.width, 4))[..., :3]
        return img[..., [2, 1, 0]]   # BGR -> RGB

    def _update_buffer(self, agent_idx):
        rgb = self._get_sensor_data(agent_idx, 'fcam')
        pose = self._camera2world(agent_idx, 'fcam')
        self.pose_buffers[agent_idx].append(pose)
        self.rgb_buffers[agent_idx].append(rgb)

    def _drain_sensor_queues(self, agent_idx):
        """Consume pending sensor data without updating buffers."""
        for sq in self.sensor_queues[agent_idx].values():
            try:
                while not sq.empty():
                    sq.get_nowait()
            except Exception:
                pass

    def _initialize_buffer(self, agent_idx):
        """Read one frame and replicate to fill the history buffer."""
        rgb = self._get_sensor_data(agent_idx, 'fcam')
        pose = self._camera2world(agent_idx, 'fcam')
        self.rgb_buffers[agent_idx].clear()
        self.pose_buffers[agent_idx].clear()
        for _ in range(self.history_length):
            self.pose_buffers[agent_idx].append(pose)
            self.rgb_buffers[agent_idx].append(rgb)

    # ── Goal sampling ─────────────────────────────────────────────────

    def _sample_goal(self, agent_idx):
        """
        Sample a navigable goal from the cached navmesh KD-tree.

        Queries all navmesh points within r_max of the agent, filters to
        those at least r_min away (annulus), and picks one uniformly at
        random.

        Note: crosswalk-challenge goals are handled separately in the
        auto-reset flow (before agent teleportation) and bypass this method.
        """
        # --- Normal goal sampling ---
        r_min, r_max = 0.8 * self.goal_range, self.goal_range   # inner & outer radius of the annulus
        xz = self._get_xz(agent_idx)        # standard world coordinate

        tree = self._navmesh_trees[agent_idx]
        pts = self._navmesh_points_std[agent_idx]

        # All cached navmesh points within r_max
        idxs = tree.query_ball_point(xz, r_max)
        if idxs:
            candidates = pts[idxs]
            dists = np.linalg.norm(candidates - xz, axis=1)
            annulus_mask = dists >= r_min
            candidates = candidates[annulus_mask]

        if idxs and len(candidates) > 0:
            choice = candidates[np.random.randint(len(candidates))]
            self.goal_globals[agent_idx] = choice
            self._goal_methods[agent_idx] = 'navmesh_annulus'
        else:
            # Fallback: relax distance constraints but stay on cached
            # sidewalk/navmesh points (never use unvalidated random sampling).
            print(f'port {self.port} ({self._current_town}) agent {agent_idx}: '
                  f'No navmesh points in annulus [{r_min:.1f}, {r_max:.1f}]m, '
                  f'relaxing distance constraints')
            self._goal_methods[agent_idx] = 'navmesh_relaxed'

            if idxs:
                # Points within r_max exist but none outside r_min — use
                # the farthest available point within r_max.
                nearby = pts[idxs]
                dists = np.linalg.norm(nearby - xz, axis=1)
                self.goal_globals[agent_idx] = nearby[np.argmax(dists)]
            elif len(pts) > 0:
                # No points within r_max — pick a sidewalk point closest to
                # the target distance (midpoint of the original annulus).
                target_dist = (r_min + r_max) / 2
                dists = np.linalg.norm(pts - xz, axis=1)
                self.goal_globals[agent_idx] = pts[
                    np.argmin(np.abs(dists - target_dist))]
            else:
                # No cached sidewalk points at all (shouldn't happen with a
                # valid cache).  Fall back to random geometric sampling as
                # an absolute last resort.
                print(f'  WARNING: agent {agent_idx} has no cached navmesh '
                      f'points — using unvalidated random goal')
                self._goal_methods[agent_idx] = 'random_fallback'
                delta_xz = sample_point_from_annulus(r_min, r_max)
                self.goal_globals[agent_idx] = xz + delta_xz

    def _sample_solvable_goal(self, agent_idx):
        """Sample a goal and verify solvability, retrying up to
        ``_max_goal_retries`` times.

        On each attempt a new goal is sampled via ``_sample_goal``, the
        geodesic field is recomputed, and ``_is_episode_solvable`` is
        checked.  If all attempts fail the last sampled goal is kept
        (best-effort) and the episode is counted as unsolvable.

        Updates ``_last_retries``, ``_goal_retries_total``,
        ``_solvable_episodes``, and ``_unsolvable_episodes``.
        """
        retries = 0
        for attempt in range(self._max_goal_retries + 1):
            self._sample_goal(agent_idx)
            self._update_geodesic(agent_idx)
            solvable, reason = self._is_episode_solvable(agent_idx)
            if solvable:
                break
            retries += 1
            if attempt < self._max_goal_retries:
                print(f'  Agent {agent_idx}: goal unsolvable ({reason}), '
                      f'resampling (attempt {attempt + 2}/{self._max_goal_retries + 1})')

        self._last_retries[agent_idx] = retries
        self._goal_retries_total += retries
        if solvable:
            self._solvable_episodes += 1
        else:
            self._unsolvable_episodes += 1
            print(f'  Agent {agent_idx}: WARNING — episode unsolvable after '
                  f'{self._max_goal_retries + 1} attempts ({reason}), '
                  f'proceeding anyway')

    def _try_crosswalk_or_fallback(self, agent_idx, crosswalk_spawns,
                                   crosswalk_goal_std):
        """Set a crosswalk-challenge goal and check solvability.

        If the crosswalk goal is unsolvable, destroy the challenge actors
        and fall back to ``_sample_solvable_goal`` with normal goal
        sampling.

        Parameters
        ----------
        agent_idx : int
        crosswalk_spawns : dict
            Mutable mapping — entry for *agent_idx* is removed on fallback.
        crosswalk_goal_std : (2,) array
            Goal in standard coordinates from the crosswalk challenge.
        """
        self.goal_globals[agent_idx] = crosswalk_goal_std
        self._goal_methods[agent_idx] = 'crosswalk_challenge'
        self.obstacle_mgr._region_scenarios.setdefault(
            agent_idx, []).append('crosswalk_challenge')
        self._update_geodesic(agent_idx)
        solvable, reason = self._is_episode_solvable(agent_idx)
        if solvable:
            self._last_retries[agent_idx] = 0
            self._solvable_episodes += 1
            return

        # Crosswalk challenge produced an unsolvable episode — tear it
        # down and fall back to normal goal sampling.
        print(f'  Agent {agent_idx}: crosswalk challenge unsolvable '
              f'({reason}), falling back to normal goal')
        if self._challenge_actor_ids[agent_idx]:
            self.obstacle_mgr.destroy_actors(
                self._challenge_actor_ids[agent_idx])
            self._challenge_actor_ids[agent_idx] = []
        crosswalk_spawns.pop(agent_idx, None)
        self._sample_solvable_goal(agent_idx)

    # ── Observation / info builders ───────────────────────────────────

    def _get_observation(self, agent_idx):
        goal_cam = self._goal_cam(agent_idx).astype(np.float32)
        # TODO: where is the following used?
        goal_world = (self.goal_globals[agent_idx] - self._get_xz(agent_idx)).astype(np.float32)
        return {
            'obs': np.array(self.rgb_buffers[agent_idx], dtype=np.uint8),
            'cord': np.array(self.pose_buffers[agent_idx])[:, [0, 2]].flatten().astype(np.float32),
            'goal': goal_cam,
            'goal_world': goal_world
        }

    def _get_info(self, agent_idx):
        speed = self.robots[agent_idx].get_velocity().length()
        info = {
            'xz': self._get_xz(agent_idx),
            'speed': speed,
            'distance_to_goal': self._distance_to_goal(agent_idx),
            'geodesic_distance_to_goal': self._geodesic_distance_to_goal(agent_idx),
            'initial_distance': self.initial_distances[agent_idx],
            'initial_geodesic_distance': self.initial_geodesic_distances[agent_idx],
            'path_length': self.path_lengths[agent_idx],
            'is_success': self._distance_to_goal(agent_idx) <= 0.2,
            'goal_retries': self._last_retries[agent_idx],
            'agent_idx': agent_idx,
        }
        if self._current_weather is not None:
            info['weather'] = self._current_weather
        return info

    def _stack_obs(self, obs_list):
        keys = obs_list[0].keys()
        return {k: np.stack([o[k] for o in obs_list]) for k in keys}

    # ── Weather randomisation ─────────────────────────────────────────

    def _randomize_weather(self):
        """Sample random weather and sun position for the current episode."""
        if not self.randomize_weather:
            return
        params = dict(
            cloudiness=float(np.random.uniform(0, 90)),
            precipitation=float(np.random.uniform(0, 80)),
            precipitation_deposits=float(np.random.uniform(0, 80)),
            wind_intensity=float(np.random.uniform(0, 100)),
            sun_azimuth_angle=float(np.random.uniform(0, 360)),
            sun_altitude_angle=float(np.random.uniform(-15, 90)),
            fog_density=float(np.random.uniform(0, 40)),
            fog_distance=float(np.random.uniform(0, 100)),
            wetness=float(np.random.uniform(0, 80)),
        )
        self._current_weather = params
        self.world.set_weather(carla.WeatherParameters(**params))

    # ── Map loading ────────────────────────────────────────────────────

    def _load_town(self):
        """Load the assigned town (if provided).

        Each environment receives a single-element town list from the
        vectorized wrapper, so no random selection is needed.
        """
        if not self.towns:
            return
        town = self.towns[0]
        if town != self._current_town:
            print(f"  Loading map: {town} (port {self.port})")
            prev_timeout = 30.0
            self.client.set_timeout(300.0)
            self.client.load_world(town)
            self.client.set_timeout(prev_timeout)
            # Invalidate cached quadrant bounds — new map has different geometry
            self.quadrant_bounds = None
            self._current_town = town
            # Pre-load navmesh cache for the new town
            if self._navmesh_cache is not None:
                self._navmesh_cache.load(town)

    def _full_reload(self):
        """Tear down all agents, load a new map, and respawn everything.

        Called during step() when map_change_interval is reached.
        After this call all per-agent state (buffers, goals, counters)
        is fresh — callers should return the new observations directly.
        """
        # ── Destroy actors ──
        for agent_sensors in self.sensors:
            for s in agent_sensors:
                try:
                    s.stop()
                    s.destroy()
                except Exception:
                    pass
        if self.all_actor_ids:
            try:
                self.client.apply_batch(
                    [carla.command.DestroyActor(x) for x in self.all_actor_ids])
            except Exception:
                pass

        # ── Load new map ──
        self._load_town()
        self.world = self.client.get_world()

        # Re-apply world settings (load_world resets them)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.dt
        settings.max_substeps = min(16, math.ceil(self.dt / 0.01))
        settings.max_substep_delta_time = self.dt / settings.max_substeps + 1e-5
        self.world.apply_settings(settings)
        self._randomize_weather()

        self.bp_lib = self.world.get_blueprint_library()
        self.tm = self.client.get_trafficmanager(self.port + 6000)
        self.tm.set_synchronous_mode(True)

        # ── Recompute regions for the new map ──
        self._compute_regions()

        # ── Geodesic distance grid for the new map ──
        self._init_geodesic_grid()

        # ── Re-spawn obstacles for new map ──
        self.obstacle_mgr.initialize(self.world, self.bp_lib,
                                     self.quadrant_bounds,
                                     navmesh_cache=self._navmesh_cache,
                                     town=self._current_town)
        self._obstacle_ids = self.obstacle_mgr.spawn_all_regions(
            self.num_agents)

        # ── Re-spawn SFM pedestrians for new map ──
        self.ped_mgr.initialize(self.world, self.bp_lib,
                                self.quadrant_bounds,
                                navmesh_cache=self._navmesh_cache,
                                town=self._current_town)

        # ── Build punctured walkable mesh for pedestrian boundary constraint ──
        self._setup_ped_walkable_mesh()

        # ── Re-initialise per-agent state ──
        self.sensor_queues = [{} for _ in range(self.num_agents)]
        self.c2r = [{} for _ in range(self.num_agents)]
        self.sensors = [[] for _ in range(self.num_agents)]
        self.all_actor_ids = (list(self._obstacle_ids)
                              + self.ped_mgr.get_actor_ids())

        # Pre-decide crosswalk challenges for new map
        # (old challenge actors were destroyed with the map change above)
        crosswalk_spawns = {}
        self._challenge_actor_ids = [[] for _ in range(self.num_agents)]
        cfg = self.obstacle_mgr.config
        if cfg is not None:
            for i in range(self.num_agents):
                if random.random() < cfg.p_crosswalk_challenge:
                    bounds = self.quadrant_inner_bounds[i]
                    result = self.obstacle_mgr.setup_crosswalk_challenge(
                        bounds)
                    if result is not None:
                        agent_pos, goal_std, actor_ids = result
                        crosswalk_spawns[i] = (agent_pos, goal_std)
                        self._challenge_actor_ids[i] = actor_ids

        walker_bps = self.bp_lib.filter('walker.pedestrian.*')
        for i in range(self.num_agents):
            if i in crosswalk_spawns:
                pos = crosswalk_spawns[i][0]
                loc = carla.Location(
                    x=float(pos[0]), y=float(pos[1]),
                    z=float(pos[2]) + 2.0)
            else:
                loc = self._sample_location_in_region(i)
                loc.z += 2.0
            spawn_tf = carla.Transform()
            spawn_tf.location = loc
            robot = self.world.try_spawn_actor(random.choice(walker_bps), spawn_tf)
            assert robot is not None, f"Failed to spawn agent {i} after map reload"
            self.robots[i] = robot
            self.all_actor_ids.append(robot.id)

        for i in range(self.num_agents):
            self._spawn_sensors_for_agent(i)

        self.world.tick()

        for i in range(self.num_agents):
            self._initialize_buffer(i)
            if i in crosswalk_spawns:
                _, goal_std = crosswalk_spawns[i]
                self.goal_globals[i] = goal_std
                self._goal_methods[i] = 'crosswalk_challenge'
                self.obstacle_mgr._region_scenarios.setdefault(
                    i, []).append('crosswalk_challenge')
            else:
                self._sample_goal(i)
            self._update_geodesic(i)
            self.initial_distances[i] = self._distance_to_goal(i)
            self.path_lengths[i] = 0.0
            self.step_counts[i] = 0

    # ── Spawn / destroy ───────────────────────────────────────────────

    def _destroy_stale_actors(self):
        stale_prefixes = ('walker.', 'sensor.', 'controller.',
                          'static.prop.', 'vehicle.')
        stale = [a for a in self.world.get_actors()
                 if any(a.type_id.startswith(p) for p in stale_prefixes)]
        if stale:
            print(f"  Destroying {len(stale)} stale actor(s) on port {self.port}")
            self.client.apply_batch(
                [carla.command.DestroyActor(a.id) for a in stale])

    def _spawn_sensors_for_agent(self, agent_idx):
        """Attach sensor(s) to agent's walker."""
        sensor_cfgs = _load_sensor_configs(filename='stack_omr4.json')
        self.sensors[agent_idx] = []
        self.sensor_queues[agent_idx] = {}
        self.c2r[agent_idx] = {}

        for sensor_cfg in sensor_cfgs:
            sensor_bp = self.bp_lib.find(sensor_cfg['type'])
            sensor_id = sensor_cfg['id']
            for attr, val in sensor_cfg['attributes'].items():
                sensor_bp.set_attribute(attr, str(val))

            tf_dict = sensor_cfg['spawn_point']
            tf = carla.Transform(
                carla.Location(x=tf_dict['x'], y=tf_dict['y'], z=tf_dict['z']),
                carla.Rotation(pitch=tf_dict['pitch'], roll=tf_dict['roll'],
                               yaw=tf_dict['yaw']),
            )

            self.c2r[agent_idx][sensor_id] = np.array(tf.get_matrix())
            q = Queue()
            self.sensor_queues[agent_idx][sensor_id] = q

            sensor = self.world.spawn_actor(
                sensor_bp, tf, attach_to=self.robots[agent_idx])
            sensor.listen(lambda data, q=q: q.put(data))
            self.sensors[agent_idx].append(sensor)
            self.all_actor_ids.append(sensor.id)

    def _reset_agent_pose(self, agent_idx):
        """Teleport an existing agent to a new spawn point in its region."""
        loc = self._sample_location_in_region(agent_idx)
        loc.z += 2.0      # prevent ground collision
        tf = carla.Transform()
        tf.location = loc
        self.robots[agent_idx].set_transform(tf)

    # ── Public API ────────────────────────────────────────────────────

    def set_collect_substep_frames(self, enabled: bool):
        self._collect_substep_frames = enabled

    def reset(self):
        """Full reset: connect to CARLA, split map, spawn all agents."""
        # TODO: check if this can be improved in terms of efficiency?
        self.close()

        self.client = carla.Client('127.0.0.1', self.port)
        self.client.set_timeout(30.0)

        # Load the assigned town
        self._load_town()

        self.tm = self.client.get_trafficmanager(self.port + 6000)
        self.tm.set_synchronous_mode(True)
        self.world = self.client.get_world()

        self._destroy_stale_actors()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.dt
        settings.max_substeps = min(16, math.ceil(self.dt / 0.01))
        settings.max_substep_delta_time = self.dt / settings.max_substeps + 1e-5
        self.world.apply_settings(settings)
        self._randomize_weather()

        self.bp_lib = self.world.get_blueprint_library()

        # ── Region splitting (cached per town) ──
        self._compute_regions()

        # ── Geodesic distance grid (from navmesh cache) ──
        self._init_geodesic_grid()

        # ── Procedural obstacles ──
        self.obstacle_mgr.initialize(self.world, self.bp_lib,
                                     self.quadrant_bounds,
                                     navmesh_cache=self._navmesh_cache,
                                     town=self._current_town)
        self._obstacle_ids = self.obstacle_mgr.spawn_all_regions(
            self.num_agents)
        self.all_actor_ids.extend(self._obstacle_ids)

        # ── SFM pedestrians ──
        self.ped_mgr.initialize(self.world, self.bp_lib,
                                self.quadrant_bounds,
                                navmesh_cache=self._navmesh_cache,
                                town=self._current_town)
        self.all_actor_ids.extend(self.ped_mgr.get_actor_ids())

        # ── Per-agent state ──
        self.rgb_buffers = [deque(maxlen=self.history_length)
                            for _ in range(self.num_agents)]
        self.pose_buffers = [deque(maxlen=self.history_length)
                             for _ in range(self.num_agents)]
        self.sensor_queues = [{} for _ in range(self.num_agents)]
        self.c2r = [{} for _ in range(self.num_agents)]
        self.sensors = [[] for _ in range(self.num_agents)]
        self.all_actor_ids = []

        # ── Pre-decide crosswalk challenges (need spawn pos before creating agents) ──
        crosswalk_spawns = {}  # agent_idx -> (agent_pos_ue, goal_std)
        self._challenge_actor_ids = [[] for _ in range(self.num_agents)]
        cfg = self.obstacle_mgr.config
        if cfg is not None:
            for i in range(self.num_agents):
                if random.random() < cfg.p_crosswalk_challenge:
                    bounds = self.quadrant_inner_bounds[i]
                    result = self.obstacle_mgr.setup_crosswalk_challenge(
                        bounds)
                    if result is not None:
                        agent_pos, goal_std, actor_ids = result
                        crosswalk_spawns[i] = (agent_pos, goal_std)
                        self._challenge_actor_ids[i] = actor_ids

        # ── Spawn walkers (one per region) ──
        walker_bps = self.bp_lib.filter('walker.pedestrian.*')
        for i in range(self.num_agents):
            if i in crosswalk_spawns:
                pos = crosswalk_spawns[i][0]
                loc = carla.Location(
                    x=float(pos[0]), y=float(pos[1]),
                    z=float(pos[2]) + 2.0)
            else:
                loc = self._sample_location_in_region(i)
                loc.z += 2.0
            spawn_tf = carla.Transform()
            spawn_tf.location = loc

            robot = self.world.try_spawn_actor(random.choice(walker_bps), spawn_tf)
            assert robot is not None, f"Failed to spawn agent {i}"
            self.robots[i] = robot
            self.all_actor_ids.append(robot.id)
            print(f"  Agent {i}: spawned at UE ({loc.x:.1f}, {loc.y:.1f})")

        # ── Attach sensors ──
        for i in range(self.num_agents):
            self._spawn_sensors_for_agent(i)

        # ── Initial tick to populate sensor queues ──
        self.world.tick()

        # ── Initialise buffers & goals ──
        for i in range(self.num_agents):
            self._initialize_buffer(i)
            if i in crosswalk_spawns:
                _, goal_std = crosswalk_spawns[i]
                self._try_crosswalk_or_fallback(
                    i, crosswalk_spawns, goal_std)
            else:
                self._sample_solvable_goal(i)
            self.initial_distances[i] = self._distance_to_goal(i)
            self.initial_geodesic_distances[i] = \
                self._geodesic_distance_to_goal(i)
            self.path_lengths[i] = 0.0
            self.step_counts[i] = 0

        obs = self._stack_obs(
            [self._get_observation(i) for i in range(self.num_agents)])
        infos = [self._get_info(i) for i in range(self.num_agents)]
        return obs, infos

    def step(self, actions: np.ndarray):
        """
        Step all agents simultaneously.

        Parameters
        ----------
        actions : (num_agents, future_length * 2)  float32
            Per-agent waypoints in the camera frame, flattened from
            (future_length, 2) where each row is [x_cam, z_cam].

        Returns  (matches VecCarlaEnv interface)
        -------
        obs_dict   : dict of stacked arrays, each (num_agents, ...)
        rewards    : (num_agents,) float32
        terminateds: (num_agents,) bool
        truncateds : (num_agents,) bool
        infos      : list[dict]
        """
        N = self.num_agents
        assert actions.shape == (N, self.future_length * 2)

        self._substep_frames = [[] for _ in range(N)]

        # ── Pre-step bookkeeping ──
        # phi0 = [self._potential(i) for i in range(N)]          # Euclidean
        phi0 = [self._geodesic_potential(i) for i in range(N)]   # geodesic
        pre_xz = [self._get_xz(i).copy() for i in range(N)]

        # Parse per-agent waypoints (camera frame)
        per_agent_waypoints = []
        for i in range(N):
            per_agent_waypoints.append(actions[i].reshape(-1, 2))

        # Command diagnostics (based on first waypoint → approximate velocity)
        cmd_infos: List[dict] = []
        for i in range(N):
            dx, dz = per_agent_waypoints[i][0]
            vx = dx / (self.dt * self.n_skips)
            vz = dz / (self.dt * self.n_skips)
            pose0 = self._get_pose(i)
            cmd_cam = np.array([vx, 0., vz])
            cmd_world = R.from_quat(pose0[3:]).apply(cmd_cam)
            cmd_infos.append({
                'cmd_vel_xz': np.array([cmd_world[0], cmd_world[2]], dtype=np.float32),
                'cmd_speed': float(np.sqrt(vx**2 + vz**2)),
            })

        # ── Cache obstacle positions (used by SFM + collision check) ──
        self._cached_obs_pos_ue = self.obstacle_mgr.get_obstacle_positions_ue()

        # ── Execute sub-steps ──
        if self.use_mpc:
            self._step_mpc_all(per_agent_waypoints)
        elif self.teleport:
            # Derive velocities from first waypoint for non-MPC modes
            vels = []
            for i in range(N):
                dx, dz = per_agent_waypoints[i][0]
                vx = dx / (self.dt * self.n_skips)
                vz = dz / (self.dt * self.n_skips)
                vels.append((vx, vz))
            self._step_teleport_all(vels)
        else:
            vels = []
            for i in range(N):
                dx, dz = per_agent_waypoints[i][0]
                vx = dx / (self.dt * self.n_skips)
                vz = dz / (self.dt * self.n_skips)
                vels.append((vx, vz))
            self._step_physics_all(vels)

        obs_pos_ue = self._cached_obs_pos_ue
        ped_pos_ue = self.ped_mgr.get_pedestrian_positions_ue()

        # ── Compute per-agent results ──
        rewards = np.zeros(N, dtype=np.float32)
        terminateds = np.zeros(N, dtype=bool)
        truncateds = np.zeros(N, dtype=bool)
        infos: List[dict] = []
        reset_set: List[int] = []

        for i in range(N):
            post_xz = self._get_xz(i)
            real_disp = post_xz - pre_xz[i]
            step_dur = self.dt * self.n_skips
            real_vel_xz = (real_disp / step_dur).astype(np.float32)

            self.path_lengths[i] += float(np.linalg.norm(real_disp))
            self.step_counts[i] += 1

            # phi1 = self._potential(i)                        # Euclidean
            phi1 = self._geodesic_potential(i)                  # geodesic
            dist = self._distance_to_goal(i)                    # Euclidean (for success check)

            # reward (DD-PPO style)
            reward = 2.5 if dist <= 0.2 else 0.0

            # decrease in geodesic distance to the goal
            reward += phi1 - phi0[i] - 0.01

            # collision penalties (proximity-based)
            robot_loc = self.robots[i].get_transform().location
            robot_ue = np.array([robot_loc.x, robot_loc.y])

            obs_collided = False
            if obs_pos_ue.shape[0] > 0:
                dists_sq = ((obs_pos_ue - robot_ue) ** 2).sum(axis=1)
                if dists_sq.min() < 0.5 ** 2:   # 0.5 m collision radius
                    reward -= 0.5
                    obs_collided = True

            ped_collided = False
            if ped_pos_ue.shape[0] > 0:
                dists_sq = ((ped_pos_ue - robot_ue) ** 2).sum(axis=1)
                if dists_sq.min() < 0.6 ** 2:   # 0.6 m (two body radii)
                    reward -= 0.5
                    ped_collided = True

            rewards[i] = reward

            # success condition: same as DD-PPO
            terminated = dist <= .2

            truncated = self.step_counts[i] >= self.max_episode_steps
            terminateds[i] = terminated
            truncateds[i] = truncated

            info = self._get_info(i)
            info['obstacle_collision'] = obs_collided
            info['pedestrian_collision'] = ped_collided
            info.update(cmd_infos[i])
            info.update({
                'real_vel_xz': real_vel_xz,
                'real_speed': float(np.linalg.norm(real_vel_xz)),
            })
            # Timing data (same for all agents on this server)
            info['sim_tick_time'] = self._last_sim_tick_time
            if self.use_mpc:
                info['mpc_solve_time'] = self._last_mpc_solve_time
            if self._collect_substep_frames and self._substep_frames[i]:
                info['substep_frames'] = np.stack(self._substep_frames[i])
            # MPC & policy vis data (only on vis iterations to avoid overhead)
            if self._collect_substep_frames:
                info['policy_waypoints_cam'] = per_agent_waypoints[i].copy()
                if (self.use_mpc
                        and hasattr(self, '_last_mpc_solutions')
                        and self._last_mpc_solutions[i] is not None):
                    x_sol, u_sol = self._last_mpc_solutions[i]
                    info['mpc_x_sol'] = x_sol    # (horizon+1, 3)
                    info['mpc_u_sol'] = u_sol    # (horizon, 2)
            infos.append(info)

            if terminated or truncated:
                # determine agents at the termination condition
                reset_set.append(i)

        # ── Capture terminal observations for truncated (not terminated)
        #    agents BEFORE auto-reset, so the trainer can bootstrap V(s_T).
        terminal_obs = {}
        for i in reset_set:
            if truncateds[i] and not terminateds[i]:
                terminal_obs[i] = self._get_observation(i)

        # ── Auto-reset finished agents ──
        if reset_set:
            self._total_episodes += len(reset_set)

            # Check if a map change is due
            if (self.towns and self.map_change_interval > 0
                    and self._total_episodes >= self.map_change_interval):
                self._total_episodes = 0
                # Capture terminal obs for ALL non-terminated agents before reload
                for i in range(N):
                    if i not in reset_set and not terminateds[i]:
                        terminal_obs[i] = self._get_observation(i)
                self._full_reload()
                # All agents were force-reset; mark all as truncated
                truncateds[:] = True
                obs = self._stack_obs(
                    [self._get_observation(i) for i in range(N)])
                # Keep infos from the terminal step (contains is_success,
                # distance_to_goal, etc.) — don't overwrite with post-reset info
                for i, tobs in terminal_obs.items():
                    infos[i]['terminal_observation'] = tobs
                return obs, rewards, terminateds, truncateds, infos

            # Normal auto-reset (same map)

            # Phase 0: clean up previous crosswalk-challenge actors for
            #          resetting agents (prevents unbounded actor accumulation)
            for i in reset_set:
                if self._challenge_actor_ids[i]:
                    self.obstacle_mgr.destroy_actors(
                        self._challenge_actor_ids[i])
                    self._challenge_actor_ids[i] = []

            # Phase 1: pre-decide crosswalk challenges (need spawn pos
            #          before teleporting agents)
            crosswalk_spawns = {}  # agent_idx -> (agent_pos_ue, goal_std)
            cfg = self.obstacle_mgr.config
            if cfg is not None:
                for i in reset_set:
                    if random.random() < cfg.p_crosswalk_challenge:
                        bounds = self.quadrant_inner_bounds[i]
                        result = self.obstacle_mgr.setup_crosswalk_challenge(
                            bounds)
                        if result is not None:
                            agent_pos, goal_std, actor_ids = result
                            crosswalk_spawns[i] = (agent_pos, goal_std)
                            self._challenge_actor_ids[i] = actor_ids

            # Phase 2: teleport finished agents
            for i in reset_set:
                if i in crosswalk_spawns:
                    pos = crosswalk_spawns[i][0]  # agent_pos_ue (3,)
                    tf = carla.Transform(carla.Location(
                        x=float(pos[0]), y=float(pos[1]),
                        z=float(pos[2]) + 2.0))
                    self.robots[i].set_transform(tf)
                else:
                    self._reset_agent_pose(i)

            # One tick to settle teleported agents & produce fresh sensor data
            self.world.tick()

            for i in range(N):
                if i in reset_set:
                    # Fresh observation for the new episode
                    self._initialize_buffer(i)
                    if i in crosswalk_spawns:
                        _, goal_std = crosswalk_spawns[i]
                        self._try_crosswalk_or_fallback(
                            i, crosswalk_spawns, goal_std)
                    else:
                        self._sample_solvable_goal(i)
                    self.initial_distances[i] = self._distance_to_goal(i)
                    self.initial_geodesic_distances[i] = \
                        self._geodesic_distance_to_goal(i)
                    self.path_lengths[i] = 0.0
                    self.step_counts[i] = 0
                else:
                    # Update buffers with the extra tick's data so non-reset
                    # agents stay in sync (fixes stale-pose bug)
                    self._update_buffer(i)

        obs = self._stack_obs(
            [self._get_observation(i) for i in range(N)])
        # Attach terminal observations for truncated agents
        for i, tobs in terminal_obs.items():
            infos[i]['terminal_observation'] = tobs
        return obs, rewards, terminateds, truncateds, infos

    # ── Sub-step implementations ──────────────────────────────────────

    def _step_teleport_all(self, vels):
        """Teleport-based stepping for all agents."""
        # Precompute world-frame displacements (frozen from step start)
        disps = []
        for i, (vx, vz) in enumerate(vels):
            disp_cam = np.array([vx * self.dt, 0., vz * self.dt])
            c2w_R = R.from_quat(self._get_pose(i)[3:])
            disp_std = c2w_R.apply(disp_cam)
            # Standard -> UE: UE_x = z_std, UE_y = x_std
            dx_ue = float(disp_std[2])
            dy_ue = float(disp_std[0])
            disps.append((dx_ue, dy_ue))

        sim_tick_time = 0.0
        for _t in range(self.n_skips):
            for i, (dx_ue, dy_ue) in enumerate(disps):
                tf = self.robots[i].get_transform()
                tf.location.x += dx_ue
                tf.location.y += dy_ue
                speed_2d = math.sqrt(dx_ue**2 + dy_ue**2)
                if speed_2d > 1e-6:
                    tf.rotation.yaw = math.degrees(math.atan2(dy_ue, dx_ue))
                self.robots[i].set_transform(tf)

            self.ped_mgr.update(self.dt, self._cached_obs_pos_ue)
            t_tick0 = time.perf_counter()
            self.world.tick()
            sim_tick_time += time.perf_counter() - t_tick0

            for i in range(self.num_agents):
                self._update_buffer(i)
                if self._collect_substep_frames:
                    self._substep_frames[i].append(
                        self.rgb_buffers[i][-1].copy())

        self._last_sim_tick_time = sim_tick_time

    def _step_physics_all(self, vels):
        """WalkerControl-based stepping for all agents."""
        controls = []
        for i, (vx, vz) in enumerate(vels):
            controls.append(_to_walker_control(vx, vz, self._get_pose(i)))

        sim_tick_time = 0.0
        for _t in range(self.n_skips):
            for i, ctrl in enumerate(controls):
                self.robots[i].apply_control(ctrl)

            self.ped_mgr.update(self.dt, self._cached_obs_pos_ue)
            t_tick0 = time.perf_counter()
            self.world.tick()
            sim_tick_time += time.perf_counter() - t_tick0

            for i in range(self.num_agents):
                self._update_buffer(i)
                if self._collect_substep_frames:
                    self._substep_frames[i].append(
                        self.rgb_buffers[i][-1].copy())

        self._last_sim_tick_time = sim_tick_time

    def _step_mpc_all(self, per_agent_waypoints):
        """MPC-based stepping for all agents.

        Each agent's camera-frame waypoints are converted to absolute world
        coordinates once (frozen at step start).  At every sub-step the world
        waypoints are re-projected into the agent's *current* camera frame,
        up-sampled to the MPC horizon via ``_repeat_and_shift``, and fed to
        the shared MPC solver.  The first unicycle control ``(v, ω)`` is
        applied via ``WalkerControl``.

        Parameters
        ----------
        per_agent_waypoints : list of (future_length, 2) ndarray
            Waypoints in each agent's camera frame at step start.
        """
        N = self.num_agents

        # Phase 1: convert camera-frame waypoints → absolute world (x_std, z_std)
        waypoints_world = []
        for i in range(N):
            wp_cam = per_agent_waypoints[i]              # (future_length, 2)
            c2w_R0 = R.from_quat(self._get_pose(i)[3:])
            xz0 = self._get_xz(i)
            wp_3d_cam = np.column_stack([
                wp_cam[:, 0],
                np.zeros(len(wp_cam)),
                wp_cam[:, 1],
            ])
            wp_3d_world = c2w_R0.apply(wp_3d_cam)
            waypoints_world.append(wp_3d_world[:, [0, 2]] + xz0)

        # Phase 2: sub-steps with MPC
        mpc_solve_time = 0.0
        sim_tick_time = 0.0
        # Capture the first sub-step's MPC solution per agent for vis
        self._last_mpc_solutions = [None] * N
        for t in range(self.n_skips):
            for i in range(N):
                # Re-project world waypoints into the current camera frame
                c2w_R = R.from_quat(self._get_pose(i)[3:])
                wp_rel = waypoints_world[i] - self._get_xz(i)
                wp_3d_rel = np.column_stack([
                    wp_rel[:, 0],
                    np.zeros(len(wp_rel)),
                    wp_rel[:, 1],
                ])
                wp_cam_3d = c2w_R.inv().apply(wp_3d_rel)
                waypoints = wp_cam_3d[:, [0, 2]]          # (future_length, 2)

                # Up-sample to MPC horizon and shift for current sub-step
                cost_weights = np.ones(waypoints.shape[0])
                cost_weights[0] = 10.0
                processed_wp = _repeat_and_shift(
                    data=waypoints, repeats=self.n_skips, shifts=t)
                processed_cw = np.squeeze(_repeat_and_shift(
                    data=cost_weights[:, None],
                    repeats=self.n_skips, shifts=t), axis=-1)

                t_mpc0 = time.perf_counter()
                x_sol, u_sol, _ = self._mpc.solve(
                    initial_pose=np.array([0.0, 0.0, 0.5 * np.pi]),
                    waypoints=processed_wp,
                    cost_weights=processed_cw,
                )
                mpc_solve_time += time.perf_counter() - t_mpc0

                # Store first sub-step solution (full open-loop plan from
                # the moment the policy output was received)
                if t == 0:
                    self._last_mpc_solutions[i] = (
                        x_sol.copy(), u_sol.copy())

                ctrl = _to_walker_control_mpc(
                    u_sol[0], self._get_pose(i), self.dt)
                self.robots[i].apply_control(ctrl)

            self.ped_mgr.update(self.dt, self._cached_obs_pos_ue)

            t_tick0 = time.perf_counter()
            self.world.tick()
            sim_tick_time += time.perf_counter() - t_tick0

            for i in range(N):
                self._update_buffer(i)
                if self._collect_substep_frames:
                    self._substep_frames[i].append(
                        self.rgb_buffers[i][-1].copy())

        self._last_mpc_solve_time = mpc_solve_time
        self._last_sim_tick_time = sim_tick_time

    # ── BEV capture ───────────────────────────────────────────────────

    def capture_bev(self, env_idx: int = 0, altitude: float = 50.0,
                    fov: float = 90.0, img_size: int = 512):
        """Capture an overhead BEV image centred on agent env_idx."""
        ego_loc = self.robots[env_idx].get_transform().location

        cam_bp = self.bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(img_size))
        cam_bp.set_attribute('image_size_y', str(img_size))
        cam_bp.set_attribute('fov', str(fov))

        cam_tf = carla.Transform(
            carla.Location(x=ego_loc.x, y=ego_loc.y,
                           z=ego_loc.z + altitude),
            carla.Rotation(pitch=-90.0, roll=0.0, yaw=0.0),
        )

        bev_q = Queue()
        cam = self.world.spawn_actor(cam_bp, cam_tf)
        cam.listen(lambda data: bev_q.put(data))
        self.world.tick()

        data = bev_q.get(timeout=30.0)
        img = np.frombuffer(data.raw_data, dtype=np.uint8)
        img = img.reshape((img_size, img_size, 4))[..., :3][..., [2, 1, 0]]

        cam.stop()
        cam.destroy()

        # Drain the extra tick's sensor data from all agents
        for i in range(self.num_agents):
            self._drain_sensor_queues(i)

        cam_pose = self._camera2world(env_idx, 'fcam')
        center_xz = np.array([cam_pose[0], cam_pose[2]], dtype=np.float32)
        meta = {
            'center_xz': center_xz,
            'altitude': float(altitude),
            'fov_deg': float(fov),
            'img_size': int(img_size),
        }
        return img, meta

    def capture_bev_batch(self, specs, fov=90.0, img_size=512):
        """Capture BEV images at multiple positions in a single world tick.

        Parameters
        ----------
        specs : list of dict, each with keys
            ``center_xz`` – (x_std, z_std) standard world coords
            ``altitude``  – metres above estimated ground
        fov      : horizontal FOV (degrees)
        img_size : pixel side length

        Returns
        -------
        list of (img, meta) tuples, one per spec.
        """
        cams = []
        queues = []

        for spec in specs:
            cx, cz = float(spec['center_xz'][0]), float(spec['center_xz'][1])
            alt = float(spec['altitude'])
            # Standard -> UE: ue_x = z_std, ue_y = x_std
            ue_x, ue_y = cz, cx

            cam_bp = self.bp_lib.find('sensor.camera.rgb')
            cam_bp.set_attribute('image_size_x', str(img_size))
            cam_bp.set_attribute('image_size_y', str(img_size))
            cam_bp.set_attribute('fov', str(fov))

            cam_tf = carla.Transform(
                carla.Location(x=ue_x, y=ue_y, z=alt + 2.0),
                carla.Rotation(pitch=-90.0, roll=0.0, yaw=0.0),
            )

            q = Queue()
            cam = self.world.spawn_actor(cam_bp, cam_tf)
            cam.listen(lambda data, q=q: q.put(data))
            cams.append(cam)
            queues.append(q)

        # Single tick renders all cameras
        self.world.tick()

        results = []
        for spec, cam, q in zip(specs, cams, queues):
            data = q.get(timeout=30.0)
            img = np.frombuffer(data.raw_data, dtype=np.uint8)
            img = img.reshape((img_size, img_size, 4))[..., :3][..., [2, 1, 0]]

            meta = {
                'center_xz': np.array(spec['center_xz'], dtype=np.float32),
                'altitude': float(spec['altitude']),
                'fov_deg': float(fov),
                'img_size': int(img_size),
            }
            results.append((img, meta))

            cam.stop()
            cam.destroy()

        # Drain sensor queues for all agents
        for i in range(self.num_agents):
            self._drain_sensor_queues(i)

        return results

    def get_obstacle_layout(self):
        """Return obstacle, pedestrian, and per-agent metadata for BEV visualisation."""
        layout = self.obstacle_mgr.get_obstacle_layout()
        layout.update(self.ped_mgr.get_pedestrian_layout())

        # Navmesh area triangles for segmentation-style BEV rendering
        if (self._navmesh_cache is not None
                and self._current_town is not None
                and self._navmesh_cache.has_town(self._current_town)):
            layout.update(
                self._navmesh_cache.get_area_triangles_std(self._current_town))

        # Per-agent metadata (ego position, goal, sampling method, scenarios)
        _QUADRANT_NAMES = ['lower-left', 'lower-right',
                           'upper-left', 'upper-right']
        agents = []
        for i in range(self.num_agents):
            ego_std = self._get_xz(i).copy()
            goal_std = (self.goal_globals[i].copy()
                        if self.goal_globals[i] is not None
                        else ego_std)
            geo_path = self._geo_paths[i]
            agents.append({
                'ego_std': ego_std,
                'goal_std': goal_std,
                'geodesic_path_std': geo_path if geo_path is not None and len(geo_path) > 0 else None,
                'goal_method': self._goal_methods[i],
                'initial_distance': float(self.initial_distances[i]),
                'geodesic_distance': float(self._geodesic_distance_to_goal(i)),
                'step_count': int(self.step_counts[i]),
                'region_scenarios': self.obstacle_mgr.get_region_scenarios(i),
                'town': self._current_town or '?',
                'quadrant': _QUADRANT_NAMES[i] if i < len(_QUADRANT_NAMES) else f'region {i}',
            })
        layout['agents'] = agents
        return layout

    @staticmethod
    def bev_world_to_pixel(xz_world, meta) -> np.ndarray:
        """Project standard xz positions to BEV image pixel coords."""
        xz = np.asarray(xz_world, dtype=np.float64)
        h = meta['altitude']
        s = meta['img_size']
        ctr = meta['center_xz']
        fov_rad = np.deg2rad(meta['fov_deg'])
        scale = s / (2.0 * h * np.tan(fov_rad / 2.0))
        dx = xz[..., 0] - ctr[0]
        dz = xz[..., 1] - ctr[1]
        u = s / 2.0 + scale * dx
        v = s / 2.0 - scale * dz
        return np.stack([u, v], axis=-1)

    # ── Solvability statistics ─────────────────────────────────────────

    def get_solvability_stats(self, reset=False):
        """Return cumulative solvability statistics and optionally reset.

        Returns a dict with:
        - ``solvable_episodes``: count of episodes confirmed solvable
        - ``unsolvable_episodes``: count that remained unsolvable after retries
        - ``unsolvable_rate``: fraction unsolvable (0 if no episodes)
        - ``goal_retries_total``: total goal-resampling retries
        - ``mean_goal_retries``: mean retries per episode (0 if no episodes)
        """
        total = self._solvable_episodes + self._unsolvable_episodes
        stats = {
            'solvable_episodes': self._solvable_episodes,
            'unsolvable_episodes': self._unsolvable_episodes,
            'unsolvable_rate': (
                self._unsolvable_episodes / total if total > 0 else 0.0),
            'goal_retries_total': self._goal_retries_total,
            'mean_goal_retries': (
                self._goal_retries_total / total if total > 0 else 0.0),
        }
        if reset:
            self._solvable_episodes = 0
            self._unsolvable_episodes = 0
            self._goal_retries_total = 0
        return stats

    # ── Cleanup ───────────────────────────────────────────────────────

    def close(self):
        if self.client is None:
            return

        print("\n  Cleaning up multi-agent episode...")
        # Destroy procedural obstacles and SFM pedestrians
        self.obstacle_mgr.clear_all(self.client)
        self.ped_mgr.clear_all(self.client)

        # Destroy sensors
        for agent_sensors in self.sensors:
            for s in agent_sensors:
                try:
                    s.stop()
                    s.destroy()
                except Exception:
                    pass

        # Destroy all actors (walkers + any remaining sensors)
        if self.all_actor_ids:
            try:
                self.client.apply_batch(
                    [carla.command.DestroyActor(x) for x in self.all_actor_ids])
            except Exception:
                pass

        try:
            settings = self.world.get_settings()
            settings.fixed_delta_seconds = None
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        except Exception:
            pass

        self.client = None
