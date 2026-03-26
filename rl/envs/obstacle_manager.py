"""
Procedural obstacle generation for CARLA multi-agent environments.

Spawns static obstacles (parked vehicles, barriers, cones) at strategically
chosen locations (crosswalks, sidewalks) to create navigation challenges
that require the ego agent to find detours.

Two-layer design:
  1. Scenario templates — parameterized obstacle placements
     (BlockedCrosswalk, SidewalkObstruction, NarrowPassage)
  2. Domain randomization — each episode randomly samples which templates
     to activate and with what parameters.

Future extension: ACCEL-style regret-based curation (mutation + replay
buffer) to adaptively evolve obstacle difficulty.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

import carla

from rl.utils.navmesh_cache import NavmeshCache


# ── Blueprint catalogs ────────────────────────────────────────────────

# Large vehicles for blocking crosswalks / sidewalks
BLOCKER_VEHICLE_BPS = [
    'vehicle.carlamotors.firetruck',
    'vehicle.tesla.cybertruck',
    'vehicle.carlamotors.european_hgv',
]

# Fallback: any large vehicle
BLOCKER_VEHICLE_FALLBACKS = [
    'vehicle.ford.ambulance',
    'vehicle.volkswagen.t2_2021',
    'vehicle.mercedes.sprinter',
]

# Static props for barriers / cones
BARRIER_PROP_BPS = [
    'static.prop.streetbarrier',
    'static.prop.constructioncone',
    'static.prop.trafficcone01',
    'static.prop.trafficcone02',
]

# Medium-sized props for sidewalk clutter
CLUTTER_PROP_BPS = [
    'static.prop.trashcan01',
    'static.prop.trashcan03',
    'static.prop.trashcan04',
    'static.prop.trashcan05',
    'static.prop.bench01',
    'static.prop.bench02',
    'static.prop.bench03',
    'static.prop.table',
    'static.prop.shoppingcart',
]


# ── Scenario dataclasses ─────────────────────────────────────────────


@dataclass
class SpawnedObstacle:
    """Tracks a spawned CARLA actor for cleanup."""
    actor_id: int
    scenario_type: str  # which template spawned this


@dataclass
class ObstacleConfig:
    """Per-episode obstacle generation parameters."""
    # Probability of activating each scenario type per region
    p_blocked_crosswalk: float = 0.7
    p_sidewalk_obstruction: float = 0.5
    p_narrow_passage: float = 0.3

    # Crosswalk challenge: pick a crosswalk, block it, place agent & goal
    # on opposite sides.  Overrides normal goal AND agent spawn.
    p_crosswalk_challenge: float = 0.5
    agent_standoff: float = 4.0    # metres past crosswalk edge for agent spawn
    goal_standoff: float = 4.0     # metres past crosswalk edge for goal
    min_crosswalk_width: float = 2.0   # skip crosswalks narrower than this (m)
    alt_crosswalk_radius: float = 40.0 # must have ≥1 alternative crosswalk within this (m)
    navmesh_snap_radius: float = 5.0   # max distance to snap agent/goal to navmesh (m)
    navmesh_snap_attempts: int = 80    # random samples when snapping without cache

    # Blocked crosswalk params
    blocker_offset_range: Tuple[float, float] = (0.5, 2.0)  # metres from crosswalk centre

    # Sidewalk obstruction params
    num_clutter_range: Tuple[int, int] = (3, 8)  # number of props per obstruction
    clutter_spread: float = 3.0  # metres along sidewalk

    # Narrow passage params
    gap_width_range: Tuple[float, float] = (0.6, 1.2)  # metres

    # Global
    max_obstacles_per_region: int = 10


# ── Crosswalk helpers ─────────────────────────────────────────────────

'''
def _parse_crosswalks(world) -> List[np.ndarray]:
    """
    Detect crosswalk polygons.  Tries the native API first, then falls
    back to navmesh-based heuristic detection.

    Returns list of (N, 3) arrays in UE coordinates.
    """
    # --- Method 1: native API (CARLA ≥ 0.9.12) ---
    try:
        raw = world.get_crosswalks()
        if raw:
            return _parse_crosswalk_vertices(raw)
    except AttributeError:
        pass

    # --- Method 2: navmesh + waypoint heuristic ---
    print("[ObstacleManager] get_crosswalks() unavailable, "
          "detecting from navmesh...")
    return _detect_crosswalks_from_navmesh(world)


def _parse_crosswalk_vertices(raw) -> List[np.ndarray]:
    """Parse the flat vertex list returned by ``world.get_crosswalks()``
    into individual closed polygons."""
    crosswalks = []
    current = []
    for loc in raw:
        pt = np.array([loc.x, loc.y, loc.z])
        if len(current) >= 3:
            if np.linalg.norm(pt - current[0]) < 0.1:
                current.append(pt)
                crosswalks.append(np.array(current))
                current = []
                continue
        current.append(pt)
    if len(current) >= 3:
        crosswalks.append(np.array(current))
    return crosswalks


def _detect_crosswalks_from_navmesh(
    world,
    n_samples: int = 3000,
    road_dist_threshold: float = 3.5,
    cluster_dist: float = 5.0,
) -> List[np.ndarray]:
    """
    Detect crosswalks by finding navmesh points that lie on top of
    driving lanes.

    The pedestrian navmesh covers sidewalks but *not* roads — except at
    crosswalks, where it extends across the road surface.  Navmesh points
    whose nearest driving-lane waypoint is closer than
    ``road_dist_threshold`` are therefore crosswalk candidates.  These
    points are clustered spatially and converted to bounding-box polygons.
    """
    from scipy.cluster.hierarchy import fclusterdata

    carla_map = world.get_map()

    # Sample navmesh points and test proximity to driving lanes
    candidates = []
    for _ in range(n_samples):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        wp = carla_map.get_waypoint(
            loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            continue
        if loc.distance(wp.transform.location) < road_dist_threshold:
            candidates.append([loc.x, loc.y, loc.z])

    if len(candidates) < 4:
        print(f"  [ObstacleManager] navmesh crosswalk detection: "
              f"only {len(candidates)} candidate point(s), giving up")
        return []

    candidates = np.array(candidates)

    # Cluster nearby candidates into individual crosswalks
    labels = fclusterdata(candidates[:, :2], t=cluster_dist,
                          criterion='distance')

    crosswalks = []
    for label in np.unique(labels):
        cluster = candidates[labels == label]
        if len(cluster) < 3:
            continue
        # Build a bounding-box polygon (5 vertices, closed)
        xmin, ymin = cluster[:, 0].min(), cluster[:, 1].min()
        xmax, ymax = cluster[:, 0].max(), cluster[:, 1].max()
        z = float(cluster[:, 2].mean())
        polygon = np.array([
            [xmin, ymin, z],
            [xmax, ymin, z],
            [xmax, ymax, z],
            [xmin, ymax, z],
            [xmin, ymin, z],
        ])
        crosswalks.append(polygon)

    print(f"  [ObstacleManager] navmesh crosswalk detection: "
          f"{len(candidates)} candidate pts → {len(crosswalks)} crosswalks")
    return crosswalks
'''

# crosswalk geometry extraction
def _crosswalk_centre(polygon: np.ndarray) -> np.ndarray:
    """Centre of a crosswalk polygon (UE coords)."""
    return polygon.mean(axis=0)


def _crosswalk_axes(polygon: np.ndarray):
    """Compute crosswalk orientation and dimensions via PCA.

    PCA robustly identifies the long axis (road-parallel) and short axis
    (crossing direction) regardless of polygon vertex order or shape.

    Returns
    -------
    long_axis   : (2,) unit vector along the crosswalk's longest dimension.
                  For typical crosswalks spanning a road, this points in the
                  crossing direction (sidewalk-to-sidewalk).
    cross_axis  : (2,) unit vector along the crosswalk's shortest dimension
                  (perpendicular to long_axis).
    half_length : half-extent along long_axis (metres)
    half_width  : half-extent along cross_axis (metres)
    """
    pts = polygon[:, :2].copy()
    # Remove closing vertex if present
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]

    centroid = pts.mean(axis=0)
    centered = pts - centroid

    # PCA via covariance eigen-decomposition
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # eigh returns eigenvalues in ascending order
    # long axis = eigenvector with *larger* eigenvalue (more variance)
    long_axis = eigenvectors[:, 1].copy()
    cross_axis = eigenvectors[:, 0].copy()

    # Project vertices to get half-extents
    proj_long = centered @ long_axis
    proj_cross = centered @ cross_axis
    half_length = (proj_long.max() - proj_long.min()) / 2.0
    half_width = (proj_cross.max() - proj_cross.min()) / 2.0

    return long_axis, cross_axis, half_length, half_width


def _crosswalk_heading(polygon: np.ndarray) -> float:
    """Heading (yaw in degrees) along the crosswalk's long axis (PCA-based)."""
    long_axis, _, _, _ = _crosswalk_axes(polygon)
    return math.degrees(math.atan2(long_axis[1], long_axis[0]))

'''
def _crosswalk_crossing_direction(polygon: np.ndarray) -> np.ndarray:
    """Unit vector along the crosswalk's short axis (the crossing direction).

    This is perpendicular to the long axis (which runs parallel to the road).
    Returns a 2D unit vector in UE (x, y).
    """
    _, cross_axis, _, _ = _crosswalk_axes(polygon)
    return cross_axis
'''

def _crosswalks_in_region(crosswalks: List[np.ndarray],
                          bounds: Tuple[float, float, float, float]
                          ) -> List[np.ndarray]:
    """Filter crosswalks whose centre falls within a quadrant."""
    xlo, xhi, ylo, yhi = bounds
    result = []
    for cw in crosswalks:
        c = _crosswalk_centre(cw)
        if xlo <= c[0] <= xhi and ylo <= c[1] <= yhi:
            result.append(cw)
    return result


# ── Main obstacle manager ────────────────────────────────────────────


class ObstacleManager:
    """Manages procedural obstacle spawning and cleanup for one CARLA world.

    Usage:
        mgr = ObstacleManager(config)
        mgr.initialize(world, bp_lib, quadrant_bounds)
        actor_ids = mgr.spawn_obstacles_for_region(region_idx)
        ...
        mgr.clear_all(client)
    """

    def __init__(self, config: Optional[ObstacleConfig] = None):
        self.config = config          # None = disabled (no obstacles)
        self.enabled = config is not None
        self.world = None
        self.bp_lib = None
        self.quadrant_bounds = None
        self.crosswalks: List[np.ndarray] = []
        self.spawned: List[SpawnedObstacle] = []
        self._navmesh_cache = None     # optional NavmeshCache
        self._current_town = None
        self._region_scenarios: dict = {}  # region_idx -> list of activated scenario names
        self._sampled_navmesh_pts: Optional[np.ndarray] = None  # (K, 3) fallback for _snap_to_navmesh

        # Resolved blueprints (populated on initialize)
        self._blocker_bps = []
        self._barrier_bps = []
        self._clutter_bps = []
        return

    def _has_cache(self, navmesh_cache: NavmeshCache, town: str) -> bool:
        has_cache = (navmesh_cache is not None and town is not None
                     and navmesh_cache.has_town(town))
        return has_cache

    def initialize(self, world, bp_lib, quadrant_bounds,
                   navmesh_cache=None, town=None):
        """Call once after CARLA world is ready (after map load).

        Parameters
        ----------
        navmesh_cache : NavmeshCache or None
            If provided, crosswalks are loaded from the cache instead of
            being detected at runtime.
        town : str or None
            Current town name (needed for cache lookup).
        """
        if not self.enabled:
            return
        self.world = world
        self.bp_lib = bp_lib
        self.quadrant_bounds = quadrant_bounds
        self._navmesh_cache = navmesh_cache
        self._current_town = town

        # Parse crosswalks (prefer cache → CARLA API → navmesh heuristic)
        has_cache = self._has_cache(navmesh_cache, town)
        if has_cache:
            self.crosswalks = navmesh_cache.get_crosswalks_ue(town)
            print(f"[ObstacleManager] Loaded {len(self.crosswalks)} crosswalks from cache")
        else:
            raise RuntimeError(f"No cached navmesh detected (town {town})")
            # self.crosswalks = _parse_crosswalks(world)
            # print(f"  [ObstacleManager] Found {len(self.crosswalks)} crosswalks")

        # Resolve available blueprints (not all exist in every CARLA build)
        self._blocker_bps = self._resolve_bps(
            BLOCKER_VEHICLE_BPS + BLOCKER_VEHICLE_FALLBACKS)
        self._barrier_bps = self._resolve_bps(BARRIER_PROP_BPS)
        self._clutter_bps = self._resolve_bps(CLUTTER_PROP_BPS)

        if not self._blocker_bps:
            # Ultimate fallback: any vehicle
            all_vehicles = list(bp_lib.filter('vehicle.*'))
            if all_vehicles:
                self._blocker_bps = all_vehicles[:5]
            print("[ObstacleManager] WARNING: no preferred blocker vehicles found, using generic vehicles")

        print(f"[ObstacleManager] Blueprints: "
              f"{len(self._blocker_bps)} blockers, "
              f"{len(self._barrier_bps)} barriers, "
              f"{len(self._clutter_bps)} clutter")

        # Pre-sample navmesh points for _snap_to_navmesh when no cache

        if not has_cache:
            raise RuntimeError(f"No cached navmesh detected (town {town})")
            # self._presample_navmesh()

        return

    def _resolve_bps(self, bp_ids):
        """Try to find each blueprint ID; return list of found ones."""
        found = []
        for bp_id in bp_ids:
            try:
                bp = self.bp_lib.find(bp_id)
                found.append(bp)
            except (IndexError, RuntimeError):
                pass
        return found

    # ── Spawning ──────────────────────────────────────────────────────

    def spawn_obstacles_for_region(self, region_idx: int) -> List[int]:
        """Procedurally generate obstacles for one agent's quadrant.

        Returns list of spawned actor IDs (caller should track for cleanup).
        """
        cfg = self.config
        bounds = self.quadrant_bounds[region_idx]
        new_ids = []
        activated = []

        # 1. Blocked crosswalk
        if random.random() < cfg.p_blocked_crosswalk:
            ids = self._spawn_blocked_crosswalk(bounds)
            if ids:
                activated.append('blocked_crosswalk')
            new_ids.extend(ids)

        # 2. Sidewalk obstruction
        if random.random() < cfg.p_sidewalk_obstruction:
            ids = self._spawn_sidewalk_obstruction(bounds)
            if ids:
                activated.append('sidewalk_obstruction')
            new_ids.extend(ids)

        # 3. Narrow passage
        if random.random() < cfg.p_narrow_passage:
            ids = self._spawn_narrow_passage(bounds)
            if ids:
                activated.append('narrow_passage')
            new_ids.extend(ids)

        self._region_scenarios[region_idx] = activated

        # Enforce cap
        return new_ids[:cfg.max_obstacles_per_region]

    def spawn_all_regions(self, num_regions: int) -> List[int]:
        """Spawn obstacles for all regions. Returns all actor IDs."""
        if not self.enabled:
            return []
        all_ids = []
        for i in range(num_regions):
            ids = self.spawn_obstacles_for_region(i)
            all_ids.extend(ids)
        return all_ids

    def get_region_scenarios(self, region_idx: int) -> List[str]:
        """Return list of scenario names activated for *region_idx*."""
        return list(self._region_scenarios.get(region_idx, []))

    def clear_all(self, client):
        """Destroy all spawned obstacle actors."""
        if not self.enabled or not self.spawned:
            return
        ids = [s.actor_id for s in self.spawned]
        try:
            client.apply_batch([carla.command.DestroyActor(x) for x in ids])
        except Exception as e:
            print(f"  [ObstacleManager] clear_all error: {e}")
        self.spawned.clear()

    def destroy_actors(self, actor_ids: List[int]):
        """Destroy specific actors and remove them from the spawned list."""
        if not actor_ids:
            return
        id_set = set(actor_ids)
        try:
            self.world.get_actor(actor_ids[0])  # validate world is alive
            cmds = [carla.command.DestroyActor(x) for x in actor_ids]
            # Need client for batch — use world's parent client
            # Fall back to individual destroy if batch unavailable
            for aid in actor_ids:
                actor = self.world.get_actor(aid)
                if actor is not None:
                    actor.destroy()
        except Exception as e:
            print(f"  [ObstacleManager] destroy_actors error: {e}")
        self.spawned = [s for s in self.spawned if s.actor_id not in id_set]

    # ── Scenario: Crosswalk Challenge (goal-coupled) ────────────────

    def setup_crosswalk_challenge(
        self, bounds,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, List[int]]]:
        """Pick a crosswalk, block it, and return agent + goal positions.

        **Crosswalk-first** approach: We first select a suitable
        crosswalk, then position the agent on one side and the goal on the
        other. This guarantees that the crosswalk geometry determines the
        challenge layout, not the (random) agent spawn.

        A crosswalk is *suitable* when:
        1. Its crossing width >= ``min_crosswalk_width`` (not a sliver).
        2. At least one *alternative* crosswalk exists within
           ``alt_crosswalk_radius`` so a detour is feasible.
        3. Both agent and goal snap-positions land on walkable navmesh.

        Parameters
        ----------
        bounds : (xlo, xhi, ylo, yhi) quadrant bounds

        Returns
        -------
        (agent_pos_ue, goal_std, actor_ids) or None
            agent_pos_ue : (3,) UE coordinates (x, y, z) for agent spawn
            goal_std     : (2,) goal in standard (x_std, z_std) coords
            actor_ids    : list of CARLA actor IDs spawned for this challenge
                           (caller must pass these to destroy_actors on reset)
        """
        if not self.enabled or not self._blocker_bps:
            return None

        cfg = self.config
        region_cws = _crosswalks_in_region(self.crosswalks, bounds)
        if len(region_cws) < 2:
            # Need at least 2 crosswalks (one to block, one as alternative)
            return None

        # Precompute centres for alternative-crosswalk check
        centres = np.array([_crosswalk_centre(cw)[:2] for cw in region_cws])

        # Shuffle so we try different crosswalks each episode
        order = list(range(len(region_cws)))
        random.shuffle(order)

        for idx in order:
            cw = region_cws[idx]
            centre = _crosswalk_centre(cw)
            long_axis, cross_axis, half_length, half_width = _crosswalk_axes(cw)

            # ── Filter 1: minimum crossing width ──
            if half_width * 2 < cfg.min_crosswalk_width:
                continue

            # ── Filter 2: alternative crosswalk within radius ──
            c_xy = centres[idx]
            dists_to_others = np.linalg.norm(centres - c_xy, axis=1)
            dists_to_others[idx] = np.inf  # exclude self
            if dists_to_others.min() > cfg.alt_crosswalk_radius:
                continue

            # ── Compute agent & goal ideal positions ──
            # Randomly assign which side gets the agent vs. the goal
            side = random.choice([-1.0, 1.0])

            agent_offset = half_length + cfg.agent_standoff
            goal_offset = half_length + cfg.goal_standoff

            agent_ideal_xy = centre[:2] + side * long_axis * agent_offset
            goal_ideal_xy = centre[:2] - side * long_axis * goal_offset

            # ── Filter 3: snap to navmesh ──
            agent_loc = self._snap_to_navmesh(agent_ideal_xy, bounds)
            if agent_loc is None:
                continue
            goal_loc = self._snap_to_navmesh(goal_ideal_xy, bounds)
            if goal_loc is None:
                continue

            # ── Block the crosswalk ──
            heading = math.degrees(math.atan2(long_axis[1], long_axis[0]))
            # Small random offset along the road axis for visual variety
            along_jitter = random.uniform(-0.3 * half_width,
                                          0.3 * half_width)
            spawn_loc = carla.Location(
                x=float(centre[0] + along_jitter * cross_axis[0]),
                y=float(centre[1] + along_jitter * cross_axis[1]),
                z=float(centre[2] + 0.5),
            )
            # Rotate 90° so the vehicle is perpendicular to the long axis,
            # i.e. oriented along the road like an illegally parked vehicle.
            spawn_rot = carla.Rotation(yaw=heading + 90)
            spawn_tf = carla.Transform(spawn_loc, spawn_rot)

            bp = random.choice(self._blocker_bps)
            actor = self.world.try_spawn_actor(bp, spawn_tf)
            if actor is None:
                continue

            actor.set_simulate_physics(False)
            self.spawned.append(
                SpawnedObstacle(actor.id, 'crosswalk_challenge'))
            challenge_ids = [actor.id]

            # Flanking barriers along the road to widen the blockage
            flank_ids = self._add_flanking_barriers(
                centre, heading + 90, width=3.0)
            challenge_ids.extend(flank_ids)

            # ── Build return values ──
            agent_pos_ue = np.array([agent_loc.x, agent_loc.y, agent_loc.z])
            goal_ue_xy = np.array([goal_loc.x, goal_loc.y])
            # Standard coords: x_std = ue_y, z_std = ue_x
            goal_std = np.array([goal_ue_xy[1], goal_ue_xy[0]])

            print(f"  [ObstacleManager] Crosswalk challenge: blocked at "
                  f"UE ({centre[0]:.1f}, {centre[1]:.1f}), "
                  f"cw_size {half_length*2:.1f}×{half_width*2:.1f}m, "
                  f"agent UE ({agent_loc.x:.1f}, {agent_loc.y:.1f}), "
                  f"goal_std ({goal_std[0]:.1f}, {goal_std[1]:.1f})")

            return agent_pos_ue, goal_std, challenge_ids

        # No suitable crosswalk found after trying all candidates
        return None

    # ── Scenario: Blocked Crosswalk (decoupled from goal) ─────────

    def _spawn_blocked_crosswalk(self, bounds) -> List[int]:
        """Place a large vehicle in front of a crosswalk to block passage."""
        region_cws = _crosswalks_in_region(self.crosswalks, bounds)
        if not region_cws or not self._blocker_bps:
            return []

        cw = random.choice(region_cws)
        centre = _crosswalk_centre(cw)
        heading = _crosswalk_heading(cw)

        # Offset the blocker perpendicular to the crosswalk
        offset = random.uniform(*self.config.blocker_offset_range)
        perp_rad = math.radians(heading + 90)
        spawn_loc = carla.Location(
            x=float(centre[0] + offset * math.cos(perp_rad)),
            y=float(centre[1] + offset * math.sin(perp_rad)),
            z=float(centre[2] + 0.5),  # slight lift to avoid ground clip
        )
        # Rotate 90° so the vehicle is perpendicular to the long axis,
        # i.e. oriented along the road like an illegally parked vehicle.
        spawn_rot = carla.Rotation(yaw=heading + 90)
        spawn_tf = carla.Transform(spawn_loc, spawn_rot)

        bp = random.choice(self._blocker_bps)
        # Disable physics so it stays put
        actor = self.world.try_spawn_actor(bp, spawn_tf)
        if actor is None:
            return []

        actor.set_simulate_physics(False)
        self.spawned.append(SpawnedObstacle(actor.id, 'blocked_crosswalk'))

        # Flanking barriers along the road to widen the blockage
        flank_ids = self._add_flanking_barriers(centre, heading + 90, width=3.0)
        return [actor.id] + flank_ids

    def _add_flanking_barriers(self, centre_ue, heading_deg,
                               width=3.0) -> List[int]:
        """Add barrier props on both sides of a blocked crosswalk."""
        if not self._barrier_bps:
            return []

        ids = []
        heading_rad = math.radians(heading_deg)
        for sign in [-1, 1]:
            offset = sign * width
            loc = carla.Location(
                x=float(centre_ue[0] + offset * math.cos(heading_rad)),
                y=float(centre_ue[1] + offset * math.sin(heading_rad)),
                z=float(centre_ue[2] + 0.3),
            )
            tf = carla.Transform(loc, carla.Rotation(yaw=heading_deg))
            bp = random.choice(self._barrier_bps)
            actor = self.world.try_spawn_actor(bp, tf)
            if actor is not None:
                actor.set_simulate_physics(False)
                self.spawned.append(
                    SpawnedObstacle(actor.id, 'blocked_crosswalk_flank'))
                ids.append(actor.id)
        return ids

    # ── Scenario: Sidewalk Obstruction ────────────────────────────────

    def _spawn_sidewalk_obstruction(self, bounds) -> List[int]:
        """Scatter clutter props along a stretch of sidewalk to block it."""
        if not self._clutter_bps:
            return []

        # Sample a random walkable point in the region as the obstruction centre
        centre_loc = self._sample_navmesh_in_bounds(bounds)
        if centre_loc is None:
            return []

        cfg = self.config
        num_props = random.randint(*cfg.num_clutter_range)
        spread = cfg.clutter_spread

        ids = []
        # Scatter props in a line/cluster around the centre
        heading = random.uniform(0, 360)
        heading_rad = math.radians(heading)

        for j in range(num_props):
            # Position along the heading direction + small perpendicular jitter
            along = random.uniform(-spread / 2, spread / 2)
            perp = random.uniform(-0.5, 0.5)
            dx = along * math.cos(heading_rad) - perp * math.sin(heading_rad)
            dy = along * math.sin(heading_rad) + perp * math.cos(heading_rad)

            loc = carla.Location(
                x=float(centre_loc.x + dx),
                y=float(centre_loc.y + dy),
                z=float(centre_loc.z + 0.3),
            )
            rot = carla.Rotation(yaw=random.uniform(0, 360))
            tf = carla.Transform(loc, rot)

            bp = random.choice(self._clutter_bps)
            actor = self.world.try_spawn_actor(bp, tf)
            if actor is not None:
                actor.set_simulate_physics(False)
                self.spawned.append(
                    SpawnedObstacle(actor.id, 'sidewalk_obstruction'))
                ids.append(actor.id)

        return ids

    # ── Scenario: Narrow Passage ──────────────────────────────────────

    def _spawn_narrow_passage(self, bounds) -> List[int]:
        """Create a chokepoint with barriers on both sides, leaving a narrow gap."""
        if not self._barrier_bps:
            return []

        centre_loc = self._sample_navmesh_in_bounds(bounds)
        if centre_loc is None:
            return []

        gap = random.uniform(*self.config.gap_width_range)
        heading = random.uniform(0, 360)
        heading_rad = math.radians(heading)

        ids = []
        # Place 2-3 barriers on each side of the gap
        for sign in [-1, 1]:
            num_barriers = random.randint(2, 3)
            for k in range(num_barriers):
                # Perpendicular offset: gap/2 + barrier spacing
                perp_offset = sign * (gap / 2 + 0.3 + k * 0.8)
                # Small along-axis jitter for visual variety
                along_offset = random.uniform(-0.3, 0.3)

                dx = (along_offset * math.cos(heading_rad)
                      - perp_offset * math.sin(heading_rad))
                dy = (along_offset * math.sin(heading_rad)
                      + perp_offset * math.cos(heading_rad))

                loc = carla.Location(
                    x=float(centre_loc.x + dx),
                    y=float(centre_loc.y + dy),
                    z=float(centre_loc.z + 0.3),
                )
                rot = carla.Rotation(yaw=heading + 90)
                tf = carla.Transform(loc, rot)

                bp = random.choice(self._barrier_bps)
                actor = self.world.try_spawn_actor(bp, tf)
                if actor is not None:
                    actor.set_simulate_physics(False)
                    self.spawned.append(
                        SpawnedObstacle(actor.id, 'narrow_passage'))
                    ids.append(actor.id)

        return ids

    # ── Helpers ───────────────────────────────────────────────────────

    def check_collision(self, location_ue, radius=0.5) -> bool:
        """Check if a UE location is within `radius` of any spawned obstacle.

        Used for teleport-mode collision detection where physics sensors
        don't fire.
        """
        if not self.spawned:
            return False
        x, y = location_ue.x, location_ue.y
        r2 = radius * radius
        for obs in self.spawned:
            actor = self.world.get_actor(obs.actor_id)
            if actor is None:
                continue
            aloc = actor.get_location()
            dx = x - aloc.x
            dy = y - aloc.y
            if dx * dx + dy * dy < r2:
                return True
        return False

    def get_obstacle_layout(self) -> dict:
        """Return obstacle bounding boxes and crosswalk polygons for BEV visualisation.

        All coordinates are in standard frame (x_std, z_std).

        Returns
        -------
        dict with keys:
            'obstacles' : list of dict, each with
                'corners_std'   : (4, 2) ndarray — oriented bounding-box corners
                'center_std'    : (2,)   ndarray — centre position
                'scenario_type' : str
            'crosswalks' : list of (N, 2) ndarray — closed polygon vertices
        """
        if not self.enabled:
            return {'obstacles': [], 'crosswalks': []}

        obstacles = []
        for obs in self.spawned:
            actor = self.world.get_actor(obs.actor_id)
            if actor is None:
                continue
            tf = actor.get_transform()
            bb = actor.bounding_box

            # 4 corners of the OBB in local actor frame (top-down)
            ex, ey = bb.extent.x, bb.extent.y
            cx, cy = bb.location.x, bb.location.y
            local_corners = [
                (cx + ex, cy + ey),
                (cx + ex, cy - ey),
                (cx - ex, cy - ey),
                (cx - ex, cy + ey),
            ]

            # Rotate + translate to UE world coordinates
            yaw_rad = math.radians(tf.rotation.yaw)
            cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)

            world_corners_ue = []
            for lx, ly in local_corners:
                wx = tf.location.x + lx * cos_y - ly * sin_y
                wy = tf.location.y + lx * sin_y + ly * cos_y
                world_corners_ue.append((wx, wy))

            # UE -> standard: x_std = ue_y, z_std = ue_x
            corners_std = np.array([[wy, wx] for wx, wy in world_corners_ue])
            center_std = np.array([tf.location.y, tf.location.x])

            obstacles.append({
                'corners_std': corners_std,
                'center_std': center_std,
                'scenario_type': obs.scenario_type,
            })

        # Crosswalk polygons in standard coords, enriched with axes/center
        crosswalks_std = []
        for cw in self.crosswalks:
            # cw: (N, 3) in UE (x, y, z)  →  (N, 2) standard (x_std, z_std)
            cw_std = np.stack([cw[:, 1], cw[:, 0]], axis=1)

            # Compute axes and center in UE, then convert to standard
            centre_ue = _crosswalk_centre(cw)
            long_axis_ue, cross_axis_ue, half_length, half_width = _crosswalk_axes(cw)

            # UE -> standard: x_std = ue_y, z_std = ue_x
            centre_std = np.array([centre_ue[1], centre_ue[0]])
            # Axis vectors: (ue_x, ue_y) -> (std_x, std_z) = (ue_y, ue_x)
            long_axis_std = np.array([long_axis_ue[1], long_axis_ue[0]])
            cross_axis_std = np.array([cross_axis_ue[1], cross_axis_ue[0]])

            crosswalks_std.append({
                'polygon_std': cw_std,
                'center_std': centre_std,
                'long_axis_std': long_axis_std,
                'cross_axis_std': cross_axis_std,
                'half_length': float(half_length),
                'half_width': float(half_width),
            })

        return {'obstacles': obstacles, 'crosswalks': crosswalks_std}

    def get_obstacle_positions_ue(self) -> np.ndarray:
        """Return (N, 2) array of obstacle positions in UE (x, y) coords.

        Cached per call — call once per step, not per agent.
        """
        if not self.enabled or not self.spawned:
            return np.empty((0, 2))
        positions = []
        for obs in self.spawned:
            actor = self.world.get_actor(obs.actor_id)
            if actor is not None:
                loc = actor.get_location()
                positions.append([loc.x, loc.y])
        if not positions:
            return np.empty((0, 2))
        return np.array(positions)

    def _presample_navmesh(self, n: int = 3000):
        """Pre-sample walkable navmesh points for fast _snap_to_navmesh.

        Called during initialize() when no NavmeshCache is available.
        Stores an (K, 3) array of UE (x, y, z) points.
        """
        pts = []
        for _ in range(n * 3):  # oversample to tolerate None returns
            if len(pts) >= n:
                break
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                pts.append([loc.x, loc.y, loc.z])
        if pts:
            self._sampled_navmesh_pts = np.array(pts)
            print(f"  [ObstacleManager] Pre-sampled {len(pts)} navmesh points")
        else:
            self._sampled_navmesh_pts = None
            print("  [ObstacleManager] WARNING: navmesh pre-sampling failed")

    def _snap_to_navmesh(
        self, target_xy: np.ndarray, bounds,
        sidewalk_only: bool = True,
    ) -> Optional[carla.Location]:
        """Find the nearest walkable navmesh point to *target_xy* (UE x, y).

        Parameters
        ----------
        sidewalk_only : bool
            If True, prefer sidewalk-only points from the cache.
            Falls back to all walkable points if no sidewalk data exists.

        Resolution order:
        1. Full navmesh cache (100k+ points, KD-tree quality)
        2. Pre-sampled points (3k points, sampled during initialize())
        3. Last resort: live random sampling (slow, low success rate)
        """
        cfg = self.config
        tx, ty = float(target_xy[0]), float(target_xy[1])
        xlo, xhi, ylo, yhi = bounds

        # ── Tier 1: full navmesh cache ──
        if (self._navmesh_cache is not None
                and self._current_town is not None
                and self._navmesh_cache.has_town(self._current_town)):
            if sidewalk_only:
                pts = self._navmesh_cache.get_sidewalk_points_ue(
                    self._current_town)
                if pts is not None and len(pts) > 0:
                    result = self._nearest_in_bounds(pts, tx, ty, bounds)
                    if result is not None:
                        return result
            # Fall back to all walkable points
            pts = self._navmesh_cache.get_walkable_points_ue(
                self._current_town)
            result = self._nearest_in_bounds(pts, tx, ty, bounds)
            if result is not None:
                return result

        # ── Tier 2: pre-sampled points ──
        if self._sampled_navmesh_pts is not None:
            result = self._nearest_in_bounds(
                self._sampled_navmesh_pts, tx, ty, bounds)
            if result is not None:
                return result

        # ── Tier 3: live random sampling (last resort) ──
        best, best_dist = None, float('inf')
        for _ in range(cfg.navmesh_snap_attempts):
            loc = self.world.get_random_location_from_navigation()
            if loc is None:
                continue
            if not (xlo <= loc.x <= xhi and ylo <= loc.y <= yhi):
                continue
            d = math.hypot(loc.x - tx, loc.y - ty)
            if d < best_dist and d <= cfg.navmesh_snap_radius:
                best, best_dist = loc, d
        return best

    def _nearest_in_bounds(
        self, pts: Optional[np.ndarray], tx: float, ty: float, bounds,
    ) -> Optional[carla.Location]:
        """Find closest point in *pts* to (tx, ty) within snap radius & bounds."""
        if pts is None or len(pts) == 0:
            return None
        cfg = self.config
        xlo, xhi, ylo, yhi = bounds
        # Filter to bounds first (vectorised)
        mask = ((pts[:, 0] >= xlo) & (pts[:, 0] <= xhi) &
                (pts[:, 1] >= ylo) & (pts[:, 1] <= yhi))
        bounded = pts[mask]
        if len(bounded) == 0:
            return None
        dists = np.linalg.norm(bounded[:, :2] - np.array([[tx, ty]]), axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] > cfg.navmesh_snap_radius:
            return None
        p = bounded[idx]
        return carla.Location(x=float(p[0]), y=float(p[1]), z=float(p[2]))

    def _sample_navmesh_in_bounds(self, bounds,
                                  max_attempts=100,
                                  sidewalk_only: bool = True,
                                  ) -> Optional[carla.Location]:
        """Sample a walkable navmesh location within the given quadrant.

        Parameters
        ----------
        sidewalk_only : bool
            If True, sample from sidewalk-only points (area_type == 1).
            Falls back to all walkable points if no sidewalk data exists.
        """
        xlo, xhi, ylo, yhi = bounds

        # Try precomputed cache first
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

        # Try pre-sampled points (much faster than live random sampling)
        if self._sampled_navmesh_pts is not None:
            pts = self._sampled_navmesh_pts
            mask = ((pts[:, 0] >= xlo) & (pts[:, 0] <= xhi) &
                    (pts[:, 1] >= ylo) & (pts[:, 1] <= yhi))
            bounded = pts[mask]
            if len(bounded) > 0:
                p = bounded[np.random.randint(len(bounded))]
                return carla.Location(
                    x=float(p[0]), y=float(p[1]), z=float(p[2]))

        # Last resort: live runtime sampling
        for _ in range(max_attempts):
            loc = self.world.get_random_location_from_navigation()
            if (loc is not None
                    and xlo <= loc.x <= xhi
                    and ylo <= loc.y <= yhi):
                return loc
        return None
