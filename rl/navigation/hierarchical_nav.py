"""
Hierarchical navigator: global planner → local goal-conditioned policy.

Uses pure-pursuit-style lookahead to select intermediate goals from a
global route, transforms them to the ego camera frame, and feeds them
to the learned PPO policy.

Works identically in CARLA and real-world deployments — only the
GlobalPlanner implementation and pose source differ.

Example (CARLA):
    planner   = CarlaGlobalPlanner(env.world)
    navigator = HierarchicalNavigator(agent, planner)
    navigator.set_destination(env.xz, goal_xz)

    obs, _ = env.reset()
    while not navigator.arrived:
        action = navigator.step(obs, env.pose)
        obs, *_ = env.step(action)

Example (real-world):
    planner   = OSMGlobalPlanner(center_gps=(37.77, -122.42))
    navigator = HierarchicalNavigator(agent, planner)
    navigator.set_destination(start_gps, goal_gps)

    while not navigator.arrived:
        obs  = robot.get_observation()
        pose = robot.get_pose()          # 7-D: xyz + quaternion
        action = navigator.step(obs, pose)
        robot.execute(action)
"""

from typing import Dict, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from rl.navigation.global_planner import GlobalPlanner, Route


class HierarchicalNavigator:
    """
    Two-level navigation controller.

    Global level : A* route planning (CARLA road network or OSM)
    Local level  : Learned goal-conditioned policy (PPOAgent)
    """

    def __init__(
        self,
        agent,
        global_planner: GlobalPlanner,
        lookahead_dist: float = 6.0,
        goal_reached_dist: float = 2.0,
        final_goal_dist: float = 1.0,
        device: str = "cuda",
    ):
        """
        Args:
            agent: PPOAgent with ``get_action_and_value()`` and
                   ``get_initial_lstm_state()``.
            global_planner: GlobalPlanner instance (Carla or OSM).
            lookahead_dist: How far ahead to place the local goal (metres).
                            Keep within the policy's training range (~6-8 m).
            goal_reached_dist: Distance to advance past route waypoints.
            final_goal_dist: Distance threshold for declaring arrival.
            device: Torch device for inference.
        """
        self.agent = agent
        self.planner = global_planner
        self.lookahead_dist = lookahead_dist
        self.goal_reached_dist = goal_reached_dist
        self.final_goal_dist = final_goal_dist
        self.device = device

        self.route: Optional[Route] = None
        self._progress_idx = 0
        self._lstm_state = None
        self._arrived = False
        self._local_goal = None

    # ── public API ────────────────────────────────────────────────────

    def set_destination(self, start, goal):
        """
        Plan a global route and prepare for navigation.

        Coordinate format depends on the planner:
          - CarlaGlobalPlanner: standard (x_std, z_std)
          - OSMGlobalPlanner:   GPS (lat, lon)
        """
        self.route = self.planner.plan(start, goal)
        self._progress_idx = 0
        self._arrived = False
        self._local_goal = None
        self._lstm_state = self.agent.get_initial_lstm_state(
            batch_size=1, device=self.device,
        )
        print(
            f"Route planned: {len(self.route)} waypoints, "
            f"{self.route.total_length:.1f} m"
        )

    @torch.no_grad()
    def step(
        self,
        observation: Dict[str, np.ndarray],
        pose: np.ndarray,
    ) -> np.ndarray:
        """
        One navigation step: select local goal → run policy → action.

        Args:
            observation: Dict with at least ``obs`` and ``cord`` keys.
            pose: (7,) [x, y, z, qx, qy, qz, qw] in the *same* metric
                  frame as the route waypoints.
                  In CARLA this is ``env.pose`` (standard coords).
                  In real-world this comes from your localisation system.

        Returns:
            action: numpy array matching the policy's action space.
        """
        assert self.route is not None, "Call set_destination() first."

        ego_xz = np.array([pose[0], pose[2]])

        # 1. Pick a local goal along the global route
        goal_world = self._select_local_goal(ego_xz)

        # 2. Transform to ego camera frame
        goal_cam = self._world_to_cam(goal_world, ego_xz, pose[3:])

        # 3. Prepare tensors
        obs_t = torch.from_numpy(observation["obs"]).unsqueeze(0).to(self.device)
        cord_t = torch.from_numpy(observation["cord"]).unsqueeze(0).float().to(self.device)
        goal_t = torch.from_numpy(goal_cam).unsqueeze(0).to(self.device)

        # 4. Run the policy
        action, _, _, _, self._lstm_state = self.agent.get_action_and_value(
            obs_t, cord_t, goal_t, self._lstm_state,
        )
        return action.squeeze(0).cpu().numpy()

    @property
    def arrived(self) -> bool:
        return self._arrived

    @property
    def progress(self) -> float:
        """Fraction of route completed (0 → 1)."""
        if self.route is None:
            return 0.0
        return self._progress_idx / max(len(self.route) - 1, 1)

    @property
    def local_goal(self) -> Optional[np.ndarray]:
        """Current local goal in world frame (for visualisation)."""
        return self._local_goal

    @property
    def remaining_distance(self) -> float:
        """Approximate remaining distance along the route (metres)."""
        if self.route is None:
            return float("inf")
        wps = self.route.waypoints[self._progress_idx:]
        if len(wps) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(wps, axis=0), axis=1)))

    # ── internals ─────────────────────────────────────────────────────

    def _select_local_goal(self, ego_xz: np.ndarray) -> np.ndarray:
        """
        Pure-pursuit lookahead along the route.

        1. Advance past waypoints closer than ``goal_reached_dist``.
        2. Walk along the route from the current waypoint for
           ``lookahead_dist`` metres and interpolate a goal point.
        """
        wps = self.route.waypoints
        n = len(wps)

        # Advance past visited waypoints
        while self._progress_idx < n - 1:
            if np.linalg.norm(wps[self._progress_idx] - ego_xz) > self.goal_reached_dist:
                break
            self._progress_idx += 1

        # Check arrival
        if np.linalg.norm(wps[-1] - ego_xz) < self.final_goal_dist:
            self._arrived = True
            self._local_goal = wps[-1]
            return wps[-1]

        # Walk along segments to find the lookahead point
        accumulated = 0.0
        goal = wps[min(self._progress_idx, n - 1)].copy()

        for i in range(self._progress_idx, n - 1):
            seg = wps[i + 1] - wps[i]
            seg_len = np.linalg.norm(seg)

            if accumulated + seg_len >= self.lookahead_dist:
                frac = (self.lookahead_dist - accumulated) / (seg_len + 1e-8)
                goal = wps[i] + np.clip(frac, 0.0, 1.0) * seg
                break

            accumulated += seg_len
            goal = wps[i + 1]

        self._local_goal = goal
        return goal

    @staticmethod
    def _world_to_cam(
        goal_xz: np.ndarray,
        ego_xz: np.ndarray,
        ego_quat: np.ndarray,
    ) -> np.ndarray:
        """
        World-frame 2D goal → ego camera frame 2D goal.

        Replicates the transform in ``CarlaEnv.goal``:
            delta   = goal_global - ego_xz
            delta3d = [delta_x, 0, delta_z]
            goal_cam = R_c2w⁻¹ · delta3d   →  [cam_x, cam_z]
        """
        delta = goal_xz - ego_xz
        delta_3d = np.array([delta[0], 0.0, delta[1]])
        c2w_R = R.from_quat(ego_quat)
        goal_cam = c2w_R.inv().apply(delta_3d)
        return np.array([goal_cam[0], goal_cam[2]], dtype=np.float32)
