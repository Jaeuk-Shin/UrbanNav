"""
Minimal 2D point-navigation environment with single-integrator dynamics.

    x_{t+1} = x_t + action * dt

No rendering, no CARLA — pure numpy.  Supports batched (vectorized)
operation so the trainer can step N environments in a single call.

Observation: relative goal vector (2D).
Action:      velocity (vx, vz), clipped to max_speed.
Reward:      potential-based shaping identical to CarlaBasicEnv.
"""

import numpy as np
import gymnasium as gym


class PointNavEnv(gym.Env):
    """Single-environment version (gym-compatible)."""

    def __init__(
        self,
        dt: float = 0.2,
        max_speed: float = 1.4,
        goal_radius: float = 0.2,
        max_steps: int = 64,
        arena_size: float = 8.0,
        gamma: float = 0.99,
    ):
        super().__init__()
        self.dt = dt
        self.max_speed = max_speed
        self.goal_radius = goal_radius
        self.max_steps = max_steps
        self.arena_size = arena_size
        self.gamma = gamma

        self.action_space = gym.spaces.Box(
            low=-max_speed, high=max_speed, shape=(2,), dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict({
            "goal": gym.spaces.Box(low=-100.0, high=100.0, shape=(2,), dtype=np.float32),
        })

        self.pos = np.zeros(2, dtype=np.float32)
        self.goal_global = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self.initial_distance = 0.0
        self.path_length = 0.0

    # ── reset / step ─────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = np.zeros(2, dtype=np.float32)
        self._sample_goal()
        self.step_count = 0
        self.initial_distance = float(self.distance_to_goal)
        self.path_length = 0.0
        return self._obs(), self._info()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        speed = np.linalg.norm(action)
        if speed > self.max_speed:
            action = action / speed * self.max_speed

        phi0 = self._potential()

        disp = action * self.dt
        self.pos = self.pos + disp
        self.path_length += float(np.linalg.norm(disp))
        self.step_count += 1

        phi1 = self._potential()
        dist = self.distance_to_goal

        # terminated = dist <= self.goal_radius
        terminated = False
        truncated = self.step_count >= self.max_steps

        reward = 2.5 if terminated else 0.0
        reward += (self.gamma * phi1 - phi0) / (self.gamma * self.dt * self.max_speed) - 0.01

        return self._obs(), float(reward), terminated, truncated, self._info()

    # ── helpers ───────────────────────────────────────────────────────

    def _sample_goal(self):
        r_min = 0.8 * self.arena_size
        r_max = self.arena_size
        th = np.pi * (2.0 * self.np_random.random() - 1.0)
        p = self.np_random.random()
        r = np.sqrt(r_max**2 * (1.0 - p) + r_min**2 * p)
        self.goal_global = self.pos + r * np.array([np.cos(th), np.sin(th)], dtype=np.float32)

    @property
    def goal(self):
        return self.goal_global - self.pos

    @property
    def distance_to_goal(self):
        return float(np.linalg.norm(self.goal))

    def _potential(self):
        return -self.distance_to_goal

    def _obs(self):
        return {"goal": self.goal.astype(np.float32)}

    def _info(self):
        return {
            "xz": self.pos.copy(),
            "distance_to_goal": self.distance_to_goal,
            "initial_distance": self.initial_distance,
            "path_length": self.path_length,
            "is_success": self.distance_to_goal <= self.goal_radius,
        }


# ═════════════════════════════════════════════════════════════════════
# Vectorized version — all N envs updated in a single numpy call
# ═════════════════════════════════════════════════════════════════════

class VecPointNavEnv:
    """
    Batched point-nav: all N envs as (N, 2) arrays.
    Auto-resets individual envs on termination / truncation.
    """

    def __init__(
        self,
        num_envs: int = 8,
        dt: float = 0.2,
        max_speed: float = 1.4,
        goal_radius: float = 0.2,
        max_steps: int = 64,
        arena_size: float = 8.0,
        gamma: float = 0.99,
        seed: int = 0,
    ):
        self.num_envs = num_envs
        self.dt = dt
        self.max_speed = max_speed
        self.goal_radius = goal_radius
        self.max_steps = max_steps
        self.arena_size = arena_size
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

        self.pos = np.zeros((num_envs, 2), dtype=np.float32)
        self.goal_global = np.zeros((num_envs, 2), dtype=np.float32)
        self.step_counts = np.zeros(num_envs, dtype=np.int64)
        self.initial_distances = np.zeros(num_envs, dtype=np.float32)
        self.path_lengths = np.zeros(num_envs, dtype=np.float32)

    # ── reset / step ─────────────────────────────────────────────────

    def reset(self):
        """Reset all envs. Returns obs dict and list of info dicts."""
        self.pos[:] = 0.0
        self._sample_goals(np.arange(self.num_envs))
        self.step_counts[:] = 0
        self.initial_distances = self._distances().copy()
        self.path_lengths[:] = 0.0
        return self._obs(), self._infos()

    def step(self, actions: np.ndarray):
        """
        actions: (num_envs, 2)
        Returns: obs_dict, rewards (N,), terminateds (N,), truncateds (N,), infos (list[dict])
        """
        actions = np.asarray(actions, dtype=np.float32)
        speeds = np.linalg.norm(actions, axis=-1, keepdims=True)
        mask = (speeds > self.max_speed).squeeze(-1)
        actions[mask] = actions[mask] / speeds[mask] * self.max_speed

        phi0 = self._potentials()

        disp = actions * self.dt
        self.pos += disp
        self.path_lengths += np.linalg.norm(disp, axis=-1)
        self.step_counts += 1

        phi1 = self._potentials()
        dists = self._distances()

        terminateds = dists <= self.goal_radius
        truncateds = self.step_counts >= self.max_steps

        rewards = np.where(terminateds, 2.5, 0.0)
        rewards += (self.gamma * phi1 - phi0) / (self.gamma * self.dt * self.max_speed) - 0.01

        infos = self._infos()
        dones = terminateds | truncateds

        # auto-reset finished envs
        done_idx = np.where(dones)[0]
        if len(done_idx) > 0:
            self.pos[done_idx] = 0.0
            self._sample_goals(done_idx)
            self.step_counts[done_idx] = 0
            self.initial_distances[done_idx] = self._distances_for(done_idx)
            self.path_lengths[done_idx] = 0.0

        return (
            self._obs(),
            rewards.astype(np.float32),
            terminateds.astype(np.float32),
            truncateds.astype(np.float32),
            infos,
        )

    # ── internal ─────────────────────────────────────────────────────

    def _sample_goals(self, idx):
        n = len(idx)
        r_min = 0.8 * self.arena_size
        r_max = self.arena_size
        th = np.pi * (2.0 * self.rng.random(n) - 1.0)
        p = self.rng.random(n)
        r = np.sqrt(r_max**2 * (1.0 - p) + r_min**2 * p)
        offsets = np.stack([r * np.cos(th), r * np.sin(th)], axis=-1).astype(np.float32)
        self.goal_global[idx] = self.pos[idx] + offsets

    def _goals(self):
        return (self.goal_global - self.pos).astype(np.float32)

    def _distances(self):
        return np.linalg.norm(self._goals(), axis=-1)

    def _distances_for(self, idx):
        return np.linalg.norm(self.goal_global[idx] - self.pos[idx], axis=-1)

    def _potentials(self):
        return -self._distances()

    def _obs(self):
        return {"goal": self._goals()}

    def _infos(self):
        dists = self._distances()
        return [
            {
                "xz": self.pos[i].copy(),
                "distance_to_goal": float(dists[i]),
                "initial_distance": float(self.initial_distances[i]),
                "path_length": float(self.path_lengths[i]),
                "is_success": bool(dists[i] <= self.goal_radius),
            }
            for i in range(self.num_envs)
        ]
