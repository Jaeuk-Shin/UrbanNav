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
from rl.envs.obstacle_manager import (
    ObstacleManager, ObstacleConfig, SpawnedObstacle,
    BLOCKER_VEHICLE_BPS, BLOCKER_VEHICLE_FALLBACKS,
)
from rl.envs.pedestrian_manager import PedestrianManager, PedestrianConfig
from rl.utils.geodesic import GeodesicDistanceField
from rl.utils.geodesic_dynamic import DynamicGeodesicField
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
            dynamic_geo_mode='off',
            dynamic_geo_horizon=5.0,
            scenario_dir: str | None = None,
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
        # action space
        self.action_space = gym.spaces.Box(
            low=-100., high=100., shape=(self.future_length * 2,), dtype=np.float32)
        # observation space
        self.observation_space = gym.spaces.Dict({
            'obs': gym.spaces.Box(0., 255., (self.history_length, height, width, 3), np.uint8),
            'cord': gym.spaces.Box(-100., 100., (self.history_length * 2,), np.float32),
            'goal': gym.spaces.Box(-100., 100., (2,), np.float32),
        })

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
        
        # for smooth rendering result
        self._collect_substep_frames = False
        self._substep_frames: List[list] = [[] for _ in range(num_agents)]

        # CCTV cameras (persistent sensors for vis iterations)
        self._cctv_cameras: List = []
        self._cctv_queues: List = []
        self._cctv_specs: List[dict] = []
        self._cctv_frames: dict = {}           # agent_idx -> list of (H,W,3) uint8

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
        # Blocked crosswalk OBBs from precomputed scenarios (per-agent)
        self._blocked_cw_obbs: List[np.ndarray | None] = [
            None] * num_agents  # each (M, 4, 2) std coords or None
        # Precomputed obstacle corners for vis (per-agent); used instead of
        # querying CARLA actors which may not be available before a tick.
        self._precomp_obstacles: List[list] = [
            [] for _ in range(num_agents)]  # each: list of {corners_std, center_std, scenario_type}

        # SFM pedestrians
        self.ped_mgr = PedestrianManager(pedestrian_config)

        # Per-quadrant geodesic distance grids (initialised per-map when
        # navmesh cache provides walkable triangles; distance fields per-episode)
        self._geo_grids: list = [None] * num_agents
        self._geo_dists: list = [None] * num_agents
        self._geo_dist_max: list = [None] * num_agents  # max finite geodesic dist (off-navmesh clamp)
        self._geo_paths: list = [None] * num_agents   # traced geodesic paths (std coords)

        # Dynamic geodesic reward mode:
        #   'off'       — static geodesic (default)
        #   'soft'      — soft-cost swept-volume heuristic (per-step 2D Dijkstra)
        #   'timespace' — exact time-space backward DP
        _valid = ('off', 'soft', 'timespace')
        assert dynamic_geo_mode == 'off'            # currently turned off
        assert dynamic_geo_mode in _valid, f"dynamic_geo_mode must be one of {_valid}"
        self._dynamic_geo_mode = dynamic_geo_mode if pedestrian_config is not None else 'off'
        self._dynamic_geo_horizon = dynamic_geo_horizon  # seconds
        # Per-quadrant DynamicGeodesicField wrappers (created after geo_grids)
        self._dgeo_fields: list = [None] * num_agents
        # Cached 3-D value fields for timespace vis (updated each step)
        self._dgeo_V: list = [None] * num_agents

        # Episode solvability tracking (cumulative across auto-resets)
        self._max_goal_retries = 5
        self._goal_retries_total = 0      # total retries across all episodes
        self._unsolvable_episodes = 0     # episodes that remained unsolvable after retries
        self._solvable_episodes = 0       # episodes confirmed solvable
        self._last_retries = [0] * num_agents  # retries for the most recent episode per agent

        # Obstacle spawn tracking (cumulative, reset with solvability stats)
        self._obstacle_spawn_requested = 0
        self._obstacle_spawn_failed = 0

        # Precomputed scenario pool (DD-PPO style)
        self._scenario_dir = scenario_dir
        self._scenario_pools: list = [[] for _ in range(num_agents)]
        self._scenario_cursors: list = [0] * num_agents
        self._scenario_meta: dict | None = None

    # ── VecCarlaEnv-compatible property ───────────────────────────────
    @property
    def num_envs(self):
        return self.num_agents

    # --- Commented out: unnecessary when precomputed scenarios are always used ---
    # (quadrant bounds are loaded from scenario meta.json instead)
    #
    # def _compute_regions(self):
    #     """Sample navmesh points to determine walkable extent, then split into
    #     quadrants in UE coordinates (Location.x, Location.y).
    #
    #     Also caches navmesh points per region as a KD-tree (in standard
    #     coordinates) for efficient navigable goal sampling.
    #     """
    #     if self.quadrant_bounds is not None:
    #         return
    #     cache = self._navmesh_cache
    #     is_town_cached = (cache is not None) and (self._current_town is not None) and cache.has_town(self._current_town)
    #     assert is_town_cached
    #     pts_ue_3d = cache.get_walkable_points_ue(self._current_town)
    #     assert (pts_ue_3d is not None) and (len(pts_ue_3d) > 0)
    #     points_ue = pts_ue_3d[:, :2]
    #     self._all_navmesh_points_ue = points_ue
    #     center_x = np.median(points_ue[:, 0])
    #     center_y = np.median(points_ue[:, 1])
    #     x_min, x_max = points_ue[:, 0].min() - 1., points_ue[:, 0].max() + 1.
    #     y_min, y_max = points_ue[:, 1].min() - 1., points_ue[:, 1].max() + 1.
    #     self.quadrant_bounds = [
    #         (x_min, center_x, y_min, center_y),
    #         (x_min, center_x, center_y, y_max),
    #         (center_x, x_max, y_min, center_y),
    #         (center_x, x_max, center_y, y_max),
    #     ]
    #     m = self.quadrant_margin
    #     self.quadrant_inner_bounds = [
    #         (xlo + m, xhi - m, ylo + m, yhi - m)
    #         for xlo, xhi, ylo, yhi in self.quadrant_bounds
    #     ]
    #     sw_points_ue = None
    #     if cache is not None and self._current_town is not None and cache.has_town(self._current_town):
    #         sw_pts_3d = cache.get_sidewalk_points_ue(self._current_town)
    #         if sw_pts_3d is not None and len(sw_pts_3d) > 0:
    #             sw_points_ue = sw_pts_3d[:, :2]
    #     goal_points_ue = sw_points_ue if sw_points_ue is not None else points_ue
    #     self._navmesh_trees = []
    #     self._navmesh_points_std = []
    #     self._navmesh_points_ue_full = []
    #     for i in range(self.num_agents):
    #         xlo, xhi, ylo, yhi = self.quadrant_inner_bounds[i]
    #         mask = ((goal_points_ue[:, 0] >= xlo) & (goal_points_ue[:, 0] <= xhi) &
    #                 (goal_points_ue[:, 1] >= ylo) & (goal_points_ue[:, 1] <= yhi))
    #         region_ue = goal_points_ue[mask]
    #         region_std = np.stack([region_ue[:, 1], region_ue[:, 0]], axis=1)
    #         self._navmesh_points_std.append(region_std)
    #         self._navmesh_trees.append(cKDTree(region_std))
    #         fxlo, fxhi, fylo, fyhi = self.quadrant_bounds[i]
    #         fmask = ((goal_points_ue[:, 0] >= fxlo) & (goal_points_ue[:, 0] <= fxhi) &
    #                  (goal_points_ue[:, 1] >= fylo) & (goal_points_ue[:, 1] <= fyhi))
    #         self._navmesh_points_ue_full.append(goal_points_ue[fmask])
    # --- End commented out ---

    # ── Geodesic distance ─────────────────────────────────────────────


    def _init_geodesic_grid(self):
        """Create per-quadrant geodesic grid helpers from scenario metadata.

        With precomputed scenarios, only the coordinate system metadata
        (origin, resolution, dimensions) is needed — no triangle loading
        or rasterization.  The grid metadata comes from the scenario
        meta.json written by ``generate_scenarios.py``.
        """
        self._geo_grids = [None] * self.num_agents
        if self._scenario_meta is None:
            return
        grid_meta = self._scenario_meta.get('grid_metadata', {})
        for i in range(self.num_agents):
            key = f'q{i}'
            if key not in grid_meta:
                print(f"  [Geodesic] Region {i}: no grid metadata — skipping")
                continue
            gm = grid_meta[key]
            self._geo_grids[i] = GeodesicDistanceField.from_metadata(
                x_min=gm['x_min'], z_min=gm['z_min'],
                H=gm['H'], W=gm['W'], resolution=gm['resolution'])

    # --- Commented out: triangle-based geodesic grid construction ---
    # (unnecessary when precomputed scenarios provide grid metadata)
    # def _init_geodesic_grid_from_navmesh(self):
    #     self._geo_grids = [None] * self.num_agents
    #     if self._navmesh_cache is None or self._current_town is None:
    #         return
    #     all_tris = self._navmesh_cache.get_sidewalk_crosswalk_tris_std(
    #         self._current_town)
    #     if all_tris is None or len(all_tris) == 0:
    #         all_tris = self._navmesh_cache.get_walkable_tris_std(
    #             self._current_town)
    #     if all_tris is None or len(all_tris) == 0:
    #         return
    #     geo_pad = self.quadrant_margin
    #     for i in range(self.num_agents):
    #         xlo, xhi, ylo, yhi = self.quadrant_bounds[i]
    #         x_std_lo, x_std_hi = ylo - geo_pad, yhi + geo_pad
    #         z_std_lo, z_std_hi = xlo - geo_pad, xhi + geo_pad
    #         in_x = ((all_tris[:, :, 0] >= x_std_lo)
    #                 & (all_tris[:, :, 0] <= x_std_hi))
    #         in_z = ((all_tris[:, :, 1] >= z_std_lo)
    #                 & (all_tris[:, :, 1] <= z_std_hi))
    #         in_bounds = (in_x & in_z).any(axis=1)
    #         quad_tris = all_tris[in_bounds]
    #         if len(quad_tris) == 0:
    #             continue
    #         self._geo_grids[i] = GeodesicDistanceField(
    #             quad_tris, resolution=1.0)
    # --- End commented out ---

    # ── Precomputed scenario loading ─────────────────────────────────

    def _load_scenario_pool(self):
        """Load precomputed scenario file paths for the current town.

        Populates ``_scenario_pools`` (per-quadrant lists of .npz paths)
        and validates that precomputed quadrant bounds match computed ones.
        """
        if self._scenario_dir is None:
            return
        town_dir = pathlib.Path(self._scenario_dir) / self._current_town
        if not town_dir.exists():
            print(f"  [Scenarios] No precomputed scenarios for "
                  f"{self._current_town}")
            self._scenario_pools = [[] for _ in range(self.num_agents)]
            return

        meta_path = town_dir / 'meta.json'
        if not meta_path.exists():
            print(f"  [Scenarios] No meta.json in {town_dir}")
            self._scenario_pools = [[] for _ in range(self.num_agents)]
            return

        with open(meta_path) as f:
            self._scenario_meta = json.load(f)

        # Load quadrant bounds from scenario metadata (replaces _compute_regions)
        stored = self._scenario_meta['quadrant_bounds']
        self.quadrant_bounds = [tuple(b) for b in stored[:self.num_agents]]
        print(f"  [Scenarios] Loaded quadrant bounds from meta.json")
        for i, (xlo, xhi, ylo, yhi) in enumerate(self.quadrant_bounds):
            print(f"  Region {i}: ue_x=[{xlo:.1f}, {xhi:.1f}], "
                  f"ue_y=[{ylo:.1f}, {yhi:.1f}]")

        for i in range(self.num_agents):
            q_dir = town_dir / f'q{i}'
            if q_dir.exists():
                paths = sorted(q_dir.glob('scenario_*.npz'))
                random.shuffle(paths)
                self._scenario_pools[i] = [str(p) for p in paths]
            else:
                self._scenario_pools[i] = []
            print(f"  [Scenarios] Region {i}: "
                  f"{len(self._scenario_pools[i])} precomputed scenarios")
        self._scenario_cursors = [0] * self.num_agents

    def _load_next_scenario(self, agent_idx: int) -> dict | None:
        """
        Load the next precomputed scenario for *agent_idx*.

        Circular buffer with reshuffle on wrap.
        """
        pool = self._scenario_pools[agent_idx]
        if not pool:
            return None

        cursor = self._scenario_cursors[agent_idx]
        if cursor >= len(pool):
            random.shuffle(pool)
            cursor = 0

        path = pool[cursor]
        self._scenario_cursors[agent_idx] = cursor + 1

        data = np.load(path, allow_pickle=False)
        return {k: data[k] for k in data.files}

    def _apply_precomputed_scenario(self, agent_idx: int,
                                     scenario: dict | None = None) -> bool:
        """
        Apply a precomputed scenario to *agent_idx*.

        Sets goal, installs the precomputed distance field, spawns CARLA
        obstacle actors, and traces the geodesic path for vis.
        Returns True on success, False to fall back to rejection sampling.

        If *scenario* is provided it is used directly; otherwise the next
        scenario is loaded from the pool (advancing the cursor).
        """
        if scenario is None:
            scenario = self._load_next_scenario(agent_idx)
        if scenario is None:
            return False

        # 1. Set goal
        self.goal_globals[agent_idx] = scenario['goal_std'].astype(np.float64)
        self._goal_methods[agent_idx] = str(scenario.get(
            'goal_method', 'precomputed'))

        # 2. Install precomputed distance field
        dist_field = scenario['dist_field'].astype(np.float64)
        geo = self._geo_grids[agent_idx]

        if geo is not None:
            # Validate grid compatibility
            if (geo._H, geo._W) != dist_field.shape:
                print(f"  [Scenarios] Agent {agent_idx}: grid shape mismatch "
                      f"({geo._H},{geo._W}) vs {dist_field.shape}, "
                      f"falling back")
                return False

        self._geo_dists[agent_idx] = dist_field
        self._cache_geo_dist_max(agent_idx, dist_field)

        # 3. Spawn obstacles in CARLA
        bp_ids = scenario.get('obstacle_bp_ids', np.array([], dtype='U64'))
        positions = scenario.get('obstacle_positions_ue',
                                 np.zeros((0, 3), dtype=np.float32))
        yaws = scenario.get('obstacle_yaws_deg',
                            np.array([], dtype=np.float32))
        types = scenario.get('obstacle_scenario_types',
                             np.array([], dtype='U32'))

        spawned_ids = []
        n_requested = len(bp_ids)
        self._obstacle_spawn_requested += n_requested
        _SPAWN_Z_OFFSETS = [5.0, 20.0, 50.0]
        for j in range(n_requested):
            bp_id = str(bp_ids[j])
            scenario_type = str(types[j])
            is_blocker = scenario_type in (
                'blocked_crosswalk', 'crosswalk_challenge')

            # Resolve blueprint (with fallback list for blockers)
            bp_candidates = []
            try:
                bp_candidates.append(self.bp_lib.find(bp_id))
            except (IndexError, RuntimeError):
                pass
            if is_blocker and not bp_candidates:
                # Original blueprint missing — try all blocker blueprints
                for alt_id in (BLOCKER_VEHICLE_BPS
                               + BLOCKER_VEHICLE_FALLBACKS):
                    if alt_id == bp_id:
                        continue
                    try:
                        bp_candidates.append(self.bp_lib.find(alt_id))
                    except (IndexError, RuntimeError):
                        continue
            if not bp_candidates:
                print(f"  [Scenarios] Agent {agent_idx}: no blueprint "
                      f"available for '{bp_id}', skipping obstacle {j}")
                self._obstacle_spawn_failed += 1
                continue

            loc = carla.Location(
                x=float(positions[j, 0]),
                y=float(positions[j, 1]),
                z=float(positions[j, 2]))
            rot = carla.Rotation(yaw=float(yaws[j]))

            # Try spawning at increasing z-offsets to clear static
            # geometry (curbs, building overhangs), then teleport to
            # ground level.  For blockers, also try alternative
            # blueprints if the original fails at all heights.
            actor = None
            for bp in bp_candidates:
                for z_off in _SPAWN_Z_OFFSETS:
                    loc_up = carla.Location(loc.x, loc.y, loc.z + z_off)
                    tf_up = carla.Transform(loc_up, rot)
                    actor = self.world.try_spawn_actor(bp, tf_up)
                    if actor is not None:
                        break
                if actor is not None:
                    break

            if actor is not None:
                actor.set_simulate_physics(False)
                actor.set_transform(carla.Transform(loc, rot))
                self.obstacle_mgr.spawned.append(
                    SpawnedObstacle(actor.id, scenario_type))
                spawned_ids.append(actor.id)
            else:
                self._obstacle_spawn_failed += 1
                print(f"  [Scenarios] Agent {agent_idx}: FAILED to spawn "
                      f"'{bp_id}' at UE ({loc.x:.1f}, {loc.y:.1f}, "
                      f"{loc.z:.1f}) yaw={float(yaws[j]):.1f}° "
                      f"type={scenario_type} "
                      f"(tried {len(bp_candidates)} bp × "
                      f"{len(_SPAWN_Z_OFFSETS)} heights)")

        self._challenge_actor_ids[agent_idx] = spawned_ids

        # 3b. Store blocked crosswalk OBBs for vis (mark crosswalks red)
        blocked_cws = scenario.get('blocked_crosswalk_obbs_std')
        if blocked_cws is not None and len(blocked_cws) > 0:
            self._blocked_cw_obbs[agent_idx] = np.asarray(
                blocked_cws, dtype=np.float64)
        else:
            self._blocked_cw_obbs[agent_idx] = None

        # 4. Trace geodesic path for BEV vis
        if geo is not None:
            start_std = scenario['start_std'].astype(np.float64)
            self._geo_paths[agent_idx] = geo.trace_path(
                dist_field, start_std)
        else:
            self._geo_paths[agent_idx] = None

        # 5. Update stats (precomputed = always solvable)
        self._last_retries[agent_idx] = 0
        self._solvable_episodes += 1

        return True

    # ── Geodesic distance field ──────────────────────────────────────

    _OFF_NAVMESH_MARGIN = 2.0   # metres beyond max reachable distance

    def _cache_geo_dist_max(self, agent_idx, dist_field: np.ndarray):
        """Cache the maximum finite geodesic distance for off-navmesh clamping."""
        finite = dist_field[np.isfinite(dist_field)]
        if finite.size > 0:
            self._geo_dist_max[agent_idx] = (
                float(finite.max()) + self._OFF_NAVMESH_MARGIN)
        else:
            self._geo_dist_max[agent_idx] = None

    def _geodesic_potential(self, agent_idx):
        """Potential based on geodesic (obstacle-aware) distance.

        When the agent is off the walkable navmesh (geodesic query returns
        inf), clamps to ``-(max_finite_distance + margin)`` so that leaving
        the navmesh always incurs a sharp penalty rather than silently
        switching to a Euclidean metric.

        Falls back to Euclidean potential only when no geodesic grid exists.
        """
        geo = self._geo_grids[agent_idx]
        if geo is not None and self._geo_dists[agent_idx] is not None:
            d = geo.query(self._geo_dists[agent_idx], self._get_xz(agent_idx))
            if np.isfinite(d):
                return -d
            # Off-navmesh: clamp to worst reachable distance + margin
            if self._geo_dist_max[agent_idx] is not None:
                return -self._geo_dist_max[agent_idx]
        return self._potential(agent_idx)

    def _geodesic_distance_to_goal(self, agent_idx):
        """Geodesic distance to goal, or clamped max if off-navmesh."""
        geo = self._geo_grids[agent_idx]
        if geo is not None and self._geo_dists[agent_idx] is not None:
            d = geo.query(self._geo_dists[agent_idx], self._get_xz(agent_idx))
            if np.isfinite(d):
                return d
            if self._geo_dist_max[agent_idx] is not None:
                return self._geo_dist_max[agent_idx]
        return self._distance_to_goal(agent_idx)

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

    # ── Weather randomization ─────────────────────────────────────────

    def _randomize_weather(self):
        """Sample random weather and sun position for the current episode."""
        
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
        return

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

        if self.randomize_weather:
            self._randomize_weather()

        self.bp_lib = self.world.get_blueprint_library()

        # ── Precomputed scenario pool (loads quadrant bounds from meta.json) ──
        self._load_scenario_pool()
        assert self._scenario_dir is not None and any(self._scenario_pools), \
            "Precomputed scenarios are required (pass --scenario_dir)"

        # ── Geodesic grid coordinate system (from scenario metadata) ──
        self._init_geodesic_grid()

        # ── Obstacle manager (world/blueprint refs for precomputed spawning) ──
        self.obstacle_mgr.initialize(self.world, self.bp_lib,
                                     self.quadrant_bounds,
                                     navmesh_cache=self._navmesh_cache,
                                     town=self._current_town)
        # NOTE: do NOT call spawn_all_regions() here — precomputed scenarios
        # specify their own obstacle placements which are spawned by
        # _apply_precomputed_scenario().  The old online generator uses
        # different placement logic (blocker at crosswalk centre) that
        # conflicts with the precomputed distance fields.

        # ── SFM pedestrians ──
        self.ped_mgr.initialize(self.world, self.bp_lib,
                                self.quadrant_bounds,
                                navmesh_cache=self._navmesh_cache,
                                town=self._current_town,
                                obstacle_layout=self.obstacle_mgr.get_obstacle_layout())
        self.all_actor_ids.extend(self.ped_mgr.get_actor_ids())

        # ── Per-agent state ──
        self.rgb_buffers = [deque(maxlen=self.history_length) for _ in range(self.num_agents)]
        self.pose_buffers = [deque(maxlen=self.history_length) for _ in range(self.num_agents)]
        self.sensor_queues = [{} for _ in range(self.num_agents)]
        self.c2r = [{} for _ in range(self.num_agents)]
        self.sensors = [[] for _ in range(self.num_agents)]
        self.all_actor_ids = []

        # ── Load precomputed scenarios ──
        precomp_scenarios = {}  # agent_idx -> scenario dict
        self._challenge_actor_ids = [[] for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            scenario = self._load_next_scenario(i)
            if scenario is not None:
                precomp_scenarios[i] = scenario

        # ── Spawn walkers (one per region) ──
        walker_bps = self.bp_lib.filter('walker.pedestrian.*')
        for i in range(self.num_agents):
            assert i in precomp_scenarios, f"No precomputed scenario for agent {i}"
            pos = precomp_scenarios[i]['start_ue']
            loc = carla.Location(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]) + 2.0)

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
            applied = self._apply_precomputed_scenario(
                i, precomp_scenarios[i])
            assert applied, f"Failed to apply precomputed scenario for agent {i}"

            self.initial_distances[i] = self._distance_to_goal(i)
            self.initial_geodesic_distances[i] = self._geodesic_distance_to_goal(i)
            self.path_lengths[i], self.step_counts[i] = 0.0, 0.0

        # Tick once so spawned obstacle actors are registered and queryable
        self.world.tick()

        obs = self._stack_obs([self._get_observation(i) for i in range(self.num_agents)])
        infos = [self._get_info(i) for i in range(self.num_agents)]
        return obs, infos

    def step(self, actions: np.ndarray):
        """
        Step all agents simultaneously.

        Parameters
        ----------
        actions : (num_agents, future_length * 2)
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
        _t_pre0 = time.perf_counter()

        # Cache obstacle positions early (needed by dynamic geodesic + SFM)
        self._cached_obs_pos_ue = self.obstacle_mgr.get_obstacle_positions_ue()

        # Dynamic geodesic: forward-predict pedestrian trajectories and
        # recompute distance fields accounting for predicted pedestrians.
        _dyn_V = None            # 3-D value field for timespace mode

        assert self._dynamic_geo_mode == 'off'
        '''
        if self._dynamic_geo_mode != 'off' and self.ped_mgr.enabled:
            horizon_steps = max(1, int(self._dynamic_geo_horizon / self.dt))
            ped_traj_std = self.ped_mgr.predict_trajectories_std(
                horizon_steps, self.dt, self._cached_obs_pos_ue)

            if self._dynamic_geo_mode == 'soft':
                for i in range(N):
                    geo = self._geo_grids[i]
                    if geo is not None and geo._static_walkable is not None:
                        self._geo_dists[i] = geo.compute_distance_field_dynamic(
                            self.goal_globals[i], ped_traj_std)
                        self._cache_geo_dist_max(i, self._geo_dists[i])
            elif self._dynamic_geo_mode == 'timespace':
                # Compute 3-D value field V[t, r, c] per agent
                _dyn_V = [None] * N
                for i in range(N):
                    dgeo = self._dgeo_fields[i]
                    if dgeo is not None and self._geo_dists[i] is not None:
                        _dyn_V[i] = dgeo.compute(
                            self._geo_dists[i], ped_traj_std)
                self._dgeo_V = _dyn_V
        '''
        # phi0: potential before action
        if _dyn_V is not None:
            phi0 = []
            for i in range(N):
                if _dyn_V[i] is not None:
                    d = self._dgeo_fields[i].query(_dyn_V[i], self._get_xz(i), t=0)
                    phi0.append(-d if np.isfinite(d)
                                else self._geodesic_potential(i))
                else:
                    phi0.append(self._geodesic_potential(i))
        else:
            # phi0 = [self._potential(i) for i in range(N)]      # Euclidean
            phi0 = [self._geodesic_potential(i) for i in range(N)]
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

        _t_pre = time.perf_counter() - _t_pre0

        # ── Execute sub-steps ──
        _t_step0 = time.perf_counter()
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
        _t_step = time.perf_counter() - _t_step0

        _t_reward0 = time.perf_counter()
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

            # phi1: potential after action
            if _dyn_V is not None and _dyn_V[i] is not None:
                d1 = self._dgeo_fields[i].query(_dyn_V[i], post_xz, t=self.n_skips)
                phi1 = -d1 if np.isfinite(d1) else self._geodesic_potential(i)
            else:
                phi1 = self._geodesic_potential(i)
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
                # Pedestrian positions for CCTV overlay
                ped_layout = self.ped_mgr.get_pedestrian_layout()
                ped_pos = [p['position_std'] for p in ped_layout.get('pedestrians', [])]
                info['pedestrian_positions_std'] = (
                    np.stack(ped_pos) if ped_pos else np.empty((0, 2)))
            infos.append(info)

            if terminated or truncated:
                # determine agents at the termination condition
                reset_set.append(i)
        _t_reward = time.perf_counter() - _t_reward0

        # ── Capture terminal observations for truncated (not terminated)
        #    agents BEFORE auto-reset, so the trainer can bootstrap V(s_T).
        _t_reset0 = time.perf_counter()
        terminal_obs = {}
        for i in reset_set:
            if truncateds[i] and not terminateds[i]:
                terminal_obs[i] = self._get_observation(i)

        # ── Auto-reset finished agents ──
        if reset_set:
            self._total_episodes += len(reset_set)

            # Normal auto-reset (same map)

            # Phase 0: clean up previous crosswalk-challenge actors for
            #          resetting agents (prevents unbounded actor accumulation)
            for i in reset_set:
                if self._challenge_actor_ids[i]:
                    self.obstacle_mgr.destroy_actors(
                        self._challenge_actor_ids[i])
                    self._challenge_actor_ids[i] = []

            # Phase 1: load precomputed scenarios (always required)
            precomp_scenarios = {}  # agent_idx -> scenario dict
            for i in reset_set:
                scenario = self._load_next_scenario(i)
                if scenario is not None:
                    precomp_scenarios[i] = scenario
            
            # Phase 2: teleport finished agents
            for i in reset_set:
                assert i in precomp_scenarios, f"No precomputed scenario for agent {i}"
                pos = precomp_scenarios[i]['start_ue']
                tf = carla.Transform(carla.Location(
                    x=float(pos[0]), y=float(pos[1]),
                    z=float(pos[2]) + 2.0))
                self.robots[i].set_transform(tf)
                
            # One tick to settle teleported agents & produce fresh sensor data
            self.world.tick()

            for i in range(N):
                if i in reset_set:
                    # Fresh observation for the new episode
                    self._initialize_buffer(i)
                    applied = self._apply_precomputed_scenario(
                        i, precomp_scenarios[i])
                    assert applied, f"Failed to apply precomputed scenario for agent {i}"
                    
                    self.initial_distances[i] = self._distance_to_goal(i)
                    self.initial_geodesic_distances[i] = \
                        self._geodesic_distance_to_goal(i)
                    self.path_lengths[i] = 0.0
                    self.step_counts[i] = 0
                else:
                    # Update buffers with the extra tick's data so non-reset
                    # agents stay in sync (fixes stale-pose bug)
                    self._update_buffer(i)

        _t_reset = time.perf_counter() - _t_reset0

        _t_obs0 = time.perf_counter()
        obs = self._stack_obs(
            [self._get_observation(i) for i in range(N)])
        _t_obs = time.perf_counter() - _t_obs0

        # Attach detailed env-side timing to all info dicts
        _step_timing = {
            'env_pre_step_time': _t_pre,
            'env_stepping_time': _t_step,
            'env_reward_time': _t_reward,
            'env_reset_time': _t_reset,
            'env_obs_time': _t_obs,
        }
        for info in infos:
            info.update(_step_timing)
        # Attach terminal observations for truncated agents
        for i, tobs in terminal_obs.items():
            infos[i]['terminal_observation'] = tobs
        return obs, rewards, terminateds, truncateds, infos


    # ── Cleanup ───────────────────────────────────────────────────────
    def close(self):
        if self.client is None:
            return

        print("\n  Cleaning up multi-agent episode...")
        # Destroy CCTV cameras
        self.destroy_cctv_cameras()
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
        return

    # ── Public API ────────────────────────────────────────────────────

    def set_collect_substep_frames(self, enabled: bool):
        self._collect_substep_frames = enabled

    # ── CCTV cameras (persistent sensors for vis iterations) ─────────

    def spawn_cctv_cameras(self, specs, fov=90.0, img_size=512):
        """Spawn persistent CCTV cameras for video capture during rollouts.

        Parameters
        ----------
        specs : list of dict, one per camera, each with keys
            ``agent_idx``  – which agent this camera is associated with
            ``cam_ue``     – (3,) camera position in UE world coords (x, y, z)
            ``pitch_deg``  – CARLA pitch (negative = look down)
            ``yaw_deg``    – CARLA yaw in degrees
            ``ground_z_ue`` – ground elevation in UE z
        fov      : horizontal FOV in degrees
        img_size : pixel side length (square)
        """
        self.destroy_cctv_cameras()
        self._cctv_specs = []
        self._cctv_frames = {}

        for spec in specs:
            aidx = spec['agent_idx']
            cam_bp = self.bp_lib.find('sensor.camera.rgb')
            cam_bp.set_attribute('image_size_x', str(img_size))
            cam_bp.set_attribute('image_size_y', str(img_size))
            cam_bp.set_attribute('fov', str(fov))

            cam_tf = carla.Transform(
                carla.Location(x=float(spec['cam_ue'][0]),
                               y=float(spec['cam_ue'][1]),
                               z=float(spec['cam_ue'][2])),
                carla.Rotation(pitch=float(spec['pitch_deg']),
                               yaw=float(spec['yaw_deg']),
                               roll=0.0),
            )

            q = Queue()
            cam = self.world.spawn_actor(cam_bp, cam_tf)
            cam.listen(lambda data, _q=q: _q.put(data))
            self._cctv_cameras.append(cam)
            self._cctv_queues.append(q)
            self._cctv_specs.append({
                **spec,
                'fov_deg': fov,
                'img_size': img_size,
            })
            self._cctv_frames.setdefault(aidx, [])

        # Tick once so each camera produces an initial frame, then discard.
        self.world.tick()
        for q in self._cctv_queues:
            try:
                q.get(timeout=5.0)
            except Exception:
                pass
        for i in range(self.num_agents):
            self._drain_sensor_queues(i)

    def _collect_cctv_step_frame(self):
        """Drain CCTV queues and keep the last frame per camera (one per step)."""
        if not self._cctv_cameras:
            return
        for spec, q in zip(self._cctv_specs, self._cctv_queues):
            latest = None
            while True:
                try:
                    latest = q.get_nowait()
                except Empty:
                    break
            if latest is not None:
                sz = spec['img_size']
                img = np.frombuffer(latest.raw_data, dtype=np.uint8)
                img = img.reshape((sz, sz, 4))[..., :3][..., [2, 1, 0]]
                self._cctv_frames[spec['agent_idx']].append(img)

    def collect_cctv_frames(self):
        """Return accumulated CCTV frames and per-camera metadata.

        Returns dict with ``'frames'`` (agent_idx -> (T,H,W,3) uint8 or None)
        and ``'specs'`` (list of spec dicts).
        """
        frames = {}
        for aidx, flist in self._cctv_frames.items():
            frames[aidx] = np.stack(flist) if flist else None
        return {'frames': frames, 'specs': self._cctv_specs}

    def destroy_cctv_cameras(self):
        """Stop and destroy all CCTV camera actors."""
        for cam in self._cctv_cameras:
            try:
                cam.stop()
                cam.destroy()
            except Exception:
                pass
        self._cctv_cameras = []
        self._cctv_queues = []
        self._cctv_specs = []
        self._cctv_frames = {}



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
            ``center_xz`` - (x_std, z_std) standard world coords
            ``altitude``  - metres above estimated ground
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

        # Mark crosswalks that are blocked by precomputed scenario obstacles
        all_blocked = [obs for obs in self._blocked_cw_obbs if obs is not None]
        if all_blocked and layout.get('crosswalks'):
            from rl.utils.mesh_utils import points_in_convex_polygon
            for cw_entry in layout['crosswalks']:
                if not isinstance(cw_entry, dict):
                    continue
                cw_center = cw_entry.get('center_std')
                if cw_center is None:
                    continue
                pt = cw_center.reshape(1, 2)
                for obbs in all_blocked:
                    for obb in obbs:
                        if points_in_convex_polygon(pt, obb)[0]:
                            cw_entry['blocked'] = True
                            break
                    if cw_entry.get('blocked'):
                        break

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
            # Re-trace geodesic from the *current* position so the path
            # starts at ego_std rather than the (possibly stale) spawn
            # position recorded at episode reset.
            geo_path = self._geo_paths[i]
            if (self._dynamic_geo_mode == 'timespace'
                    and self._dgeo_fields[i] is not None
                    and self._dgeo_V[i] is not None
                    and self._geo_dists[i] is not None):
                geo_path = self._dgeo_fields[i].trace_path(
                    self._dgeo_V[i], ego_std, self._geo_dists[i])
            elif (self._geo_grids[i] is not None
                    and self._geo_dists[i] is not None):
                geo_path = self._geo_grids[i].trace_path(
                    self._geo_dists[i], ego_std)
            # Geodesic field data for heatmap visualisation
            geo = self._geo_grids[i]
            geo_grid_meta = None
            geo_field_2d = None
            geo_field_3d = None
            if geo is not None:
                geo_grid_meta = {
                    'x_min': geo._x_min, 'z_min': geo._z_min,
                    'resolution': geo._resolution,
                    'H': geo._H, 'W': geo._W,
                }
                if self._geo_dists[i] is not None:
                    geo_field_2d = self._geo_dists[i]
                if self._dgeo_V[i] is not None:
                    geo_field_3d = self._dgeo_V[i]

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
                'geo_grid_meta': geo_grid_meta,
                'geo_field_2d': geo_field_2d,
                'geo_field_3d': geo_field_3d,
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
        obs_req = self._obstacle_spawn_requested
        obs_fail = self._obstacle_spawn_failed
        stats = {
            'solvable_episodes': self._solvable_episodes,
            'unsolvable_episodes': self._unsolvable_episodes,
            'unsolvable_rate': (
                self._unsolvable_episodes / total if total > 0 else 0.0),
            'goal_retries_total': self._goal_retries_total,
            'mean_goal_retries': (
                self._goal_retries_total / total if total > 0 else 0.0),
            'obstacle_spawn_requested': obs_req,
            'obstacle_spawn_failed': obs_fail,
            'obstacle_spawn_fail_rate': (
                obs_fail / obs_req if obs_req > 0 else 0.0),
        }
        if reset:
            self._solvable_episodes = 0
            self._unsolvable_episodes = 0
            self._goal_retries_total = 0
            self._obstacle_spawn_requested = 0
            self._obstacle_spawn_failed = 0
        return stats


    # ── Sub-step implementations ──────────────────────────────────────

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
        self._collect_cctv_step_frame()
        return

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
        self._collect_cctv_step_frame()

    
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
        self._collect_cctv_step_frame()
        return

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
    