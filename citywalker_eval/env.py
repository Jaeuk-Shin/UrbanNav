"""Thin wrapper around ``CarlaMultiAgentEnv`` for subgoal-driven rollouts.

We reuse the existing multi-agent env with ``num_agents=1`` and override the
per-step "goal" so that the policy sees *our* current subgoal rather than the
scenario-file goal. Arrival is tracked against the *final* goal externally —
the env's own ``terminated`` flag (which fires at any in-scenario goal within
0.2 m) is ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from rl.envs.carla_multi import CarlaMultiAgentEnv

from .planning import (
    SubgoalSchedule,
    plan_reference_path,
    sample_goal_location,
    sparsify_path,
)


@dataclass
class RouteInfo:
    start_xz: np.ndarray
    goal_xz: np.ndarray
    reference_path: np.ndarray    # (N, 2) std coords
    subgoals: np.ndarray          # (K, 2) std coords
    total_length: float


class NavTestEnv:
    """Single-agent CARLA environment with an external subgoal schedule.

    Construction is delegated to ``CarlaMultiAgentEnv`` (``num_agents=1``).
    The schedule is built lazily by ``plan_route``; callers typically do::

        env = NavTestEnv(cfg, scenario_dir=..., port=2000)
        obs, info = env.reset()
        route = env.plan_route(goal_distance=100.0, subgoal_spacing=8.0)
        while not env.arrived:
            action = policy.act(...)
            obs, info = env.step(action)
    """

    def __init__(
        self,
        cfg,
        scenario_dir: str,
        port: int = 2000,
        fps: int = 5,
        max_speed: float = 1.4,
        teleport: bool = False,
        use_mpc: bool = False,
        towns=None,
        max_episode_steps: int = 500,
        reach_threshold: float = 2.0,
        final_threshold: float = 1.0,
        rng_seed: Optional[int] = None,
    ) -> None:
        self._inner = CarlaMultiAgentEnv(
            cfg=cfg,
            port=port,
            num_agents=1,
            max_speed=max_speed,
            fps=fps,
            teleport=teleport,
            use_mpc=use_mpc,
            towns=towns,
            max_episode_steps=max_episode_steps,
            scenario_dir=scenario_dir,
        )
        self.cfg = cfg
        self.reach_threshold = reach_threshold
        self.final_threshold = final_threshold
        self._rng = np.random.default_rng(rng_seed)

        self._schedule: Optional[SubgoalSchedule] = None
        self._route_info: Optional[RouteInfo] = None
        self._steps = 0

    # ── passthroughs ──────────────────────────────────────────────────

    @property
    def inner(self) -> CarlaMultiAgentEnv:
        return self._inner

    @property
    def world(self):
        return self._inner.world

    @property
    def future_length(self) -> int:
        return self._inner.future_length

    @property
    def history_length(self) -> int:
        return self._inner.history_length

    def agent_xz(self) -> np.ndarray:
        return np.asarray(self._inner._get_xz(0), dtype=np.float64)

    def agent_pose(self) -> np.ndarray:
        return np.asarray(self._inner._get_pose(0), dtype=np.float64)

    def rgb_history(self) -> np.ndarray:
        return np.asarray(list(self._inner.rgb_buffers[0]), dtype=np.uint8)

    def pose_history(self) -> np.ndarray:
        return np.asarray(list(self._inner.pose_buffers[0]), dtype=np.float64)

    # ── lifecycle ─────────────────────────────────────────────────────

    def reset(self):
        obs_dict, infos = self._inner.reset()
        self._schedule = None
        self._route_info = None
        self._steps = 0
        return obs_dict, infos[0]

    def close(self):
        self._inner.close()

    # ── route setup ───────────────────────────────────────────────────

    def plan_route(
        self,
        goal_distance: float = 100.0,
        goal_tolerance: float = 15.0,
        goal_xz: Optional[np.ndarray] = None,
        subgoal_spacing: float = 8.0,
        path_resolution: float = 2.0,
    ) -> RouteInfo:
        """Sample (or accept) a goal, plan the path, build the subgoal schedule."""
        start_xz = self.agent_xz()

        if goal_xz is None:
            goal_xz = sample_goal_location(
                world=self.world,
                start_xz=start_xz,
                distance=goal_distance,
                tolerance=goal_tolerance,
                rng=self._rng,
            )
        goal_xz = np.asarray(goal_xz, dtype=np.float64)

        route = plan_reference_path(
            world=self.world,
            start_xz=start_xz,
            goal_xz=goal_xz,
            resolution=path_resolution,
        )
        subgoals = sparsify_path(
            route.waypoints, spacing=subgoal_spacing, include_goal=True
        )
        self._schedule = SubgoalSchedule(
            subgoals=subgoals,
            reach_threshold=self.reach_threshold,
            final_threshold=self.final_threshold,
        )
        self._route_info = RouteInfo(
            start_xz=start_xz,
            goal_xz=goal_xz,
            reference_path=np.asarray(route.waypoints, dtype=np.float64),
            subgoals=subgoals,
            total_length=float(route.total_length),
        )
        # Install the first subgoal as the env's "goal" so the policy-facing
        # obs['goal'] points at it.
        self._inner.goal_globals[0] = np.asarray(
            self._schedule.current, dtype=np.float64
        )
        return self._route_info

    # ── state queries ─────────────────────────────────────────────────

    @property
    def route_info(self) -> Optional[RouteInfo]:
        return self._route_info

    @property
    def schedule(self) -> Optional[SubgoalSchedule]:
        return self._schedule

    @property
    def current_subgoal(self) -> np.ndarray:
        assert self._schedule is not None, "Call plan_route() first"
        return self._schedule.current

    @property
    def final_goal(self) -> np.ndarray:
        assert self._schedule is not None, "Call plan_route() first"
        return self._schedule.final

    @property
    def arrived(self) -> bool:
        return self._schedule is not None and self._schedule.arrived

    @property
    def steps(self) -> int:
        return self._steps

    # ── step ──────────────────────────────────────────────────────────

    def step(self, action_cam: np.ndarray):
        """Advance the sim by one policy step.

        Parameters
        ----------
        action_cam : (future_length * 2,) flattened camera-frame waypoints,
                     metres — exactly what ``CityWalkerPolicy.act`` returns.

        Returns
        -------
        obs  : per-agent observation dict (first-slot extracted)
        info : augmented info dict (adds subgoal/route fields)
        """
        assert self._schedule is not None, "Call plan_route() first"

        action = np.asarray(action_cam, dtype=np.float32).reshape(1, -1)
        assert action.shape == (1, self._inner.future_length * 2), (
            f"expected action shape (1, {self._inner.future_length * 2}), "
            f"got {action.shape}"
        )

        # Keep env goal aligned with current subgoal (the policy reads
        # obs['goal'] derived from goal_globals).
        self._inner.goal_globals[0] = np.asarray(
            self._schedule.current, dtype=np.float64
        )

        obs_dict, rewards, term, trunc, infos = self._inner.step(action)
        self._steps += 1

        # Advance the subgoal based on the (post-step) real agent position.
        agent_xz = self.agent_xz()
        self._schedule.update(agent_xz)
        # Re-install the new subgoal so the NEXT observation.goal is correct.
        self._inner.goal_globals[0] = np.asarray(
            self._schedule.current, dtype=np.float64
        )

        obs = {k: v[0] for k, v in obs_dict.items()}
        info = dict(infos[0])
        info.update(
            {
                "subgoal_idx": int(self._schedule.cursor),
                "num_subgoals": int(self._schedule.num_subgoals),
                "subgoal_xz": self._schedule.current.copy(),
                "final_goal_xz": self._schedule.final.copy(),
                "dist_to_subgoal": float(
                    np.linalg.norm(self._schedule.current - agent_xz)
                ),
                "dist_to_final": float(
                    np.linalg.norm(self._schedule.final - agent_xz)
                ),
                "remaining_route_distance": self._schedule.remaining_distance(
                    agent_xz
                ),
                "arrived": bool(self._schedule.arrived),
                "env_terminated": bool(term[0]),
                "env_truncated": bool(trunc[0]),
            }
        )
        return obs, info
