"""Global goal sampling, reference path planning, and subgoal scheduling.

All 2-D positions use the UrbanNav *standard* frame:
    x_std = UE_y  (rightward)
    z_std = UE_x  (forward)
which is what ``CarlaMultiAgentEnv._get_xz`` and ``goal_globals`` speak.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from rl.navigation.global_planner import CarlaGlobalPlanner, Route


# ── Goal sampling ─────────────────────────────────────────────────────


def _std_to_ue(xz: np.ndarray) -> Tuple[float, float]:
    """Standard (x_std, z_std) → UE (x_ue, y_ue)."""
    return float(xz[1]), float(xz[0])


def _ue_to_std(x_ue: float, y_ue: float) -> np.ndarray:
    return np.array([y_ue, x_ue], dtype=np.float64)


def sample_goal_location(
    world,
    start_xz: np.ndarray,
    distance: float,
    tolerance: float = 10.0,
    max_attempts: int = 200,
    snap_to_road: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample a reachable goal location at roughly ``distance`` metres from start.

    Strategy: rejection-sample random navigable locations from CARLA's navmesh
    (``world.get_random_location_from_navigation``), keep ones whose distance
    to ``start_xz`` is within ``[distance - tolerance, distance + tolerance]``,
    then optionally snap to the nearest driving waypoint so the road-network
    planner can find a route.
    """
    import carla

    rng = rng or np.random.default_rng()
    carla_map = world.get_map()

    best: Optional[np.ndarray] = None
    best_err = float("inf")
    for _ in range(max_attempts):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        if snap_to_road:
            wp = carla_map.get_waypoint(loc, project_to_road=True)
            loc = wp.transform.location
        cand = _ue_to_std(loc.x, loc.y)
        err = abs(float(np.linalg.norm(cand - start_xz)) - distance)
        if err < tolerance:
            return cand
        if err < best_err:
            best, best_err = cand, err

    if best is None:
        raise RuntimeError(
            f"Failed to sample any navigable goal after {max_attempts} attempts"
        )
    # Fallback: best-effort closest match
    return best


# ── Reference path ────────────────────────────────────────────────────


def plan_reference_path(
    world,
    start_xz: np.ndarray,
    goal_xz: np.ndarray,
    resolution: float = 2.0,
) -> Route:
    """Plan a reference path on the CARLA road graph.

    Returns a ``Route`` whose ``waypoints`` are (N, 2) in std coords.
    """
    planner = CarlaGlobalPlanner(world, resolution=resolution)
    return planner.plan(np.asarray(start_xz), np.asarray(goal_xz))


# ── Sparse subgoals ───────────────────────────────────────────────────


def sparsify_path(
    waypoints: np.ndarray,
    spacing: float,
    include_goal: bool = True,
) -> np.ndarray:
    """Resample a polyline at roughly ``spacing`` metres of arc length.

    The first point is always included; intermediate points are picked so the
    cumulative arc length between kept points is ≥ ``spacing``. If
    ``include_goal``, the final waypoint is appended (possibly closer than
    ``spacing`` to its predecessor).
    """
    wps = np.asarray(waypoints, dtype=np.float64)
    if len(wps) == 0:
        return wps
    if len(wps) == 1:
        return wps.copy()

    seg = np.linalg.norm(np.diff(wps, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])

    if total <= spacing:
        out = [wps[0]]
        if include_goal:
            out.append(wps[-1])
        return np.asarray(out)

    targets = np.arange(0.0, total, spacing)
    # Linear interpolation along arc length
    xs = np.interp(targets, cum, wps[:, 0])
    ys = np.interp(targets, cum, wps[:, 1])
    subgoals = np.stack([xs, ys], axis=1)
    if include_goal and float(np.linalg.norm(subgoals[-1] - wps[-1])) > 1e-3:
        subgoals = np.concatenate([subgoals, wps[-1:]], axis=0)
    return subgoals


# ── Subgoal schedule ──────────────────────────────────────────────────


@dataclass
class SubgoalSchedule:
    """Advance through a list of subgoals as the agent makes progress.

    Uses a distance-based advancement rule: once the agent is within
    ``reach_threshold`` metres of the *current* subgoal, the cursor jumps to
    the next one. Completion is declared when the agent is within
    ``final_threshold`` of the *final* waypoint.
    """

    subgoals: np.ndarray            # (K, 2) std coords
    reach_threshold: float = 2.0    # m — advance intermediate subgoals
    final_threshold: float = 1.0    # m — declare arrival
    _cursor: int = 0
    _arrived: bool = False

    def __post_init__(self):
        self.subgoals = np.asarray(self.subgoals, dtype=np.float64)
        assert self.subgoals.ndim == 2 and self.subgoals.shape[1] == 2
        assert len(self.subgoals) >= 1

    @property
    def current(self) -> np.ndarray:
        return self.subgoals[self._cursor]

    @property
    def final(self) -> np.ndarray:
        return self.subgoals[-1]

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def num_subgoals(self) -> int:
        return len(self.subgoals)

    @property
    def arrived(self) -> bool:
        return self._arrived

    def update(self, agent_xz: np.ndarray) -> None:
        """Advance past reached intermediate subgoals and check arrival."""
        agent_xz = np.asarray(agent_xz, dtype=np.float64)
        # Greedy skip of intermediate subgoals we're already past.
        while self._cursor < len(self.subgoals) - 1:
            if np.linalg.norm(self.subgoals[self._cursor] - agent_xz) <= self.reach_threshold:
                self._cursor += 1
            else:
                break
        # Arrival check (always against the true final goal).
        if np.linalg.norm(self.final - agent_xz) <= self.final_threshold:
            self._arrived = True
            self._cursor = len(self.subgoals) - 1

    def remaining_distance(self, agent_xz: np.ndarray) -> float:
        """Approximate remaining arc length from agent along the subgoal chain."""
        agent_xz = np.asarray(agent_xz, dtype=np.float64)
        wps = self.subgoals[self._cursor:]
        if len(wps) == 0:
            return 0.0
        d = float(np.linalg.norm(wps[0] - agent_xz))
        if len(wps) > 1:
            d += float(np.sum(np.linalg.norm(np.diff(wps, axis=0), axis=1)))
        return d
