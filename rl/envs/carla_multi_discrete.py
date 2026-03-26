"""
DD-PPO-style discrete-action multi-agent CARLA environment.

4 actions matching Habitat's PointNav (Wijmans et al., ICLR 2020):
  0 = STOP         — end the episode (success iff within success_distance)
  1 = MOVE_FORWARD — advance forward_step_size (default 0.25 m)
  2 = TURN_LEFT    — rotate left by turn_angle  (default 10 deg)
  3 = TURN_RIGHT   — rotate right by turn_angle (default 10 deg)

All other behaviour (region splitting, sensor management, goal sampling,
BEV capture, etc.) is inherited unchanged from CarlaMultiAgentEnv.
"""

import math

import numpy as np
import gymnasium as gym

from rl.envs.carla_multi import CarlaMultiAgentEnv

# Action indices (matching Habitat convention)
STOP = 0
MOVE_FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3

NUM_ACTIONS = 4


class CarlaMultiAgentDiscreteEnv(CarlaMultiAgentEnv):
    """CarlaMultiAgentEnv with DD-PPO's 4-action Discrete space.

    Instead of continuous (dx, dz) camera-frame displacements, the agent
    controls heading via TURN_LEFT / TURN_RIGHT and advances a fixed step
    via MOVE_FORWARD.  STOP terminates the episode.
    """

    def __init__(self, *args, forward_step_size=0.25, turn_angle=10.0,
                 success_distance=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_step_size = forward_step_size
        self.turn_angle_deg = turn_angle
        self.success_distance = success_distance
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

    # ── Step ──────────────────────────────────────────────────────────

    def step(self, actions):
        """
        Parameters
        ----------
        actions : (num_agents,) int array — DD-PPO action indices.

        Returns the same tuple as ``CarlaMultiAgentEnv.step()``.
        """
        N = self.num_agents
        actions = np.asarray(actions, dtype=int)

        self._substep_frames = [[] for _ in range(N)]

        # ── pre-step bookkeeping ──
        phi0 = [self._potential(i) for i in range(N)]
        pre_xz = [self._get_xz(i).copy() for i in range(N)]

        # ── apply actions via teleport ──
        for i in range(N):
            tf = self.robots[i].get_transform()
            a = actions[i]
            if a == MOVE_FORWARD:
                yaw_rad = math.radians(tf.rotation.yaw)
                # UE forward = +x at yaw 0; +y is right, positive yaw = right
                tf.location.x += self.forward_step_size * math.cos(yaw_rad)
                tf.location.y += self.forward_step_size * math.sin(yaw_rad)
                self.robots[i].set_transform(tf)
            elif a == TURN_LEFT:
                tf.rotation.yaw -= self.turn_angle_deg
                self.robots[i].set_transform(tf)
            elif a == TURN_RIGHT:
                tf.rotation.yaw += self.turn_angle_deg
                self.robots[i].set_transform(tf)
            # STOP: no transform change

        # ── single world tick + sensor update ──
        obs_pos_ue = self.obstacle_mgr.get_obstacle_positions_ue()
        self.ped_mgr.update(self.dt, obs_pos_ue)
        self.world.tick()
        for i in range(N):
            self._update_buffer(i)
            if self._collect_substep_frames:
                self._substep_frames[i].append(
                    self.rgb_buffers[i][-1].copy())

        # ── cache positions for collision check ──
        ped_pos_ue = self.ped_mgr.get_pedestrian_positions_ue()

        # ── compute per-agent results ──
        rewards = np.zeros(N, dtype=np.float32)
        terminateds = np.zeros(N, dtype=bool)
        truncateds = np.zeros(N, dtype=bool)
        infos = []
        reset_set = []

        for i in range(N):
            post_xz = self._get_xz(i)
            real_disp = post_xz - pre_xz[i]
            self.path_lengths[i] += float(np.linalg.norm(real_disp))
            self.step_counts[i] += 1

            phi1 = self._potential(i)
            dist = self._distance_to_goal(i)

            # DD-PPO reward: success bonus only when agent calls STOP
            if actions[i] == STOP:
                terminated = True
                reward = 2.5 if dist <= self.success_distance else 0.0
            else:
                terminated = False
                reward = 0.0

            # potential shaping + slack penalty (same as continuous)
            reward += phi1 - phi0[i] - 0.01

            # collision penalties (proximity-based)
            robot_loc = self.robots[i].get_transform().location
            robot_ue = np.array([robot_loc.x, robot_loc.y])

            obs_collided = False
            if obs_pos_ue.shape[0] > 0:
                dists_sq = ((obs_pos_ue - robot_ue) ** 2).sum(axis=1)
                if dists_sq.min() < 0.5 ** 2:
                    reward -= 0.5
                    obs_collided = True

            ped_collided = False
            if ped_pos_ue.shape[0] > 0:
                dists_sq = ((ped_pos_ue - robot_ue) ** 2).sum(axis=1)
                if dists_sq.min() < 0.6 ** 2:
                    reward -= 0.5
                    ped_collided = True

            rewards[i] = reward

            truncated = (not terminated
                         and self.step_counts[i] >= self.max_episode_steps)
            terminateds[i] = terminated
            truncateds[i] = truncated

            info = self._get_info(i)
            info['obstacle_collision'] = obs_collided
            info['pedestrian_collision'] = ped_collided
            # DD-PPO: success requires explicit STOP within radius
            info['is_success'] = bool(
                terminated and dist <= self.success_distance)
            info['action'] = int(actions[i])
            info['displacement_xz'] = real_disp.astype(np.float32)
            # control diagnostics (for buffer.store_control_info compat)
            info['real_vel_xz'] = (real_disp / self.dt).astype(np.float32)
            info['real_speed'] = float(
                np.linalg.norm(real_disp) / self.dt)
            info['cmd_vel_xz'] = np.zeros(2, dtype=np.float32)
            info['cmd_speed'] = 0.0

            if self._collect_substep_frames and self._substep_frames[i]:
                info['substep_frames'] = np.stack(self._substep_frames[i])
            infos.append(info)

            if terminated or truncated:
                reset_set.append(i)

        # ── terminal observations for truncation bootstrap ──
        terminal_obs = {}
        for i in reset_set:
            if truncateds[i] and not terminateds[i]:
                terminal_obs[i] = self._get_observation(i)

        # ── auto-reset finished agents ──
        if reset_set:
            self._total_episodes += len(reset_set)

            # map change check
            if (self.towns and self.map_change_interval > 0
                    and self._total_episodes >= self.map_change_interval):
                self._total_episodes = 0
                for i in range(N):
                    if i not in reset_set and not terminateds[i]:
                        terminal_obs[i] = self._get_observation(i)
                self._full_reload()
                truncateds[:] = True
                obs = self._stack_obs(
                    [self._get_observation(i) for i in range(N)])
                for i, tobs in terminal_obs.items():
                    infos[i]['terminal_observation'] = tobs
                return obs, rewards, terminateds, truncateds, infos

            # normal auto-reset (same map)
            for i in reset_set:
                self._reset_agent_pose(i)

            self.world.tick()

            for i in range(N):
                if i in reset_set:
                    self._initialize_buffer(i)
                    self._sample_goal(i)
                    self.initial_distances[i] = self._distance_to_goal(i)
                    self.path_lengths[i] = 0.0
                    self.step_counts[i] = 0
                else:
                    self._update_buffer(i)

        obs = self._stack_obs(
            [self._get_observation(i) for i in range(N)])
        for i, tobs in terminal_obs.items():
            infos[i]['terminal_observation'] = tobs
        return obs, rewards, terminateds, truncateds, infos
