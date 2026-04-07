import numpy as np
import torch


# ─── GAE ──────────────────────────────────────────────────────────────
def compute_gae(rewards, values, terminateds, dones, next_values,
                gamma=0.99, gae_lambda=0.95):
    """
    Correct GAE that distinguishes terminated vs truncated episodes.

    Parameters
    ----------
    rewards      : (T, E) per-step rewards
    values       : (T, E) value estimates V(s_t)
    terminateds  : (T, E) 1.0 at true episode end (goal reached)
    dones        : (T, E) 1.0 at any episode boundary (terminated | truncated)
    next_values  : (T, E) pre-computed bootstrap values.
                   For normal steps:    next_values[t] = values[t+1]
                   For last step:       next_values[T-1] = V(s_{T+1}) from obs
                   For truncated steps: next_values[t] = V(s_terminal)
                   (computed from the terminal obs before auto-reset)

    Returns advantages, returns — both (T, E).
    """
    T = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros(rewards.shape[1])

    for t in reversed(range(T)):
        # non_terminal: 0 only for true terminations (goal reached).
        # Truncated episodes keep the bootstrap (non_terminal = 1)
        # because next_values[t] already has V(s_terminal).
        non_terminal = 1.0 - terminateds[t]
        # Cut the GAE trace at any episode boundary so advantages
        # don't leak across episodes.
        not_new_episode = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * not_new_episode * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# ─── Rollout Buffers ─────────────────────────────────────────────────
class RolloutBuffers:
    """Pre-allocated numpy + torch buffers for one rollout."""

    def __init__(self, num_steps, num_envs, obs_shape, cord_shape, action_dim,
                 num_tokens, obs_feat_dim, device,
                 action_history_dim=0,
                 aux_heads=None, aux_grid_size=16, aux_max_objects=8):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.num_tokens = num_tokens
        self.obs_feat_dim = obs_feat_dim
        self.action_history_dim = action_history_dim
        self.aux_heads = aux_heads or []

        # numpy buffers
        self.obs = np.zeros((num_steps, num_envs, *obs_shape), dtype=np.uint8)
        self.cord = np.zeros((num_steps, num_envs, *cord_shape), dtype=np.float32)
        self.goal = np.zeros((num_steps, num_envs, 2), dtype=np.float32)
        self.goal_world = np.zeros((num_steps, num_envs, 2), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs, action_dim), dtype=np.float32)
        self.logprobs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.raw_rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.terminateds = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)

        # Truncation bootstrap: V(s_terminal) for truncated (not terminated)
        # episodes, used to build the next_values array for GAE.
        self.trunc_values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.has_trunc_value = np.zeros((num_steps, num_envs), dtype=np.float32)

        # action history buffer (flat vector per env per step)
        if action_history_dim > 0:
            self.action_hist = np.zeros((num_steps, num_envs, action_history_dim), dtype=np.float32)

        # control diagnostics (commanded vs actual)
        self.cmd_vel = np.zeros((num_steps, num_envs, 2), dtype=np.float32)
        self.real_vel = np.zeros((num_steps, num_envs, 2), dtype=np.float32)
        self.cmd_speed = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.real_speed = np.zeros((num_steps, num_envs), dtype=np.float32)

        # GPU feature buffers — filled during rollout, reused for PPO update
        self.features_gpu = torch.zeros(
            num_steps, num_envs, num_tokens, obs_feat_dim,
            device=device,
        )
        self.dec_out_gpu = torch.zeros(
            num_steps, num_envs, obs_feat_dim,
            device=device,
        )

        # Auxiliary target buffers (for LSTM probing heads)
        if "occupancy" in self.aux_heads:
            self.aux_occupancy = np.zeros(
                (num_steps, num_envs, aux_grid_size, aux_grid_size),
                dtype=np.float32)
        if "obstacle_pos" in self.aux_heads:
            self.aux_obstacle_pos = np.zeros(
                (num_steps, num_envs, aux_max_objects, 2), dtype=np.float32)
            self.aux_obstacle_mask = np.zeros(
                (num_steps, num_envs, aux_max_objects), dtype=np.float32)
        if "geodesic_dist" in self.aux_heads:
            self.aux_geodesic_dist = np.zeros(
                (num_steps, num_envs), dtype=np.float32)

    def store_control_info(self, step, infos):
        """Extract cmd_vel_xz, real_vel_xz, etc. from env info dicts."""
        for i, info in enumerate(infos):
            self.cmd_vel[step, i] = info.get('cmd_vel_xz', np.zeros(2))
            self.real_vel[step, i] = info.get('real_vel_xz', np.zeros(2))
            self.cmd_speed[step, i] = info.get('cmd_speed', 0.0)
            self.real_speed[step, i] = info.get('real_speed', 0.0)

    def store_aux_targets(self, step, infos):
        """Extract auxiliary prediction targets from env info dicts."""
        for i, info in enumerate(infos):
            if "occupancy" in self.aux_heads:
                occ = info.get('aux_occupancy')
                if occ is not None:
                    self.aux_occupancy[step, i] = occ
            if "obstacle_pos" in self.aux_heads:
                pos = info.get('aux_obstacle_pos')
                msk = info.get('aux_obstacle_mask')
                if pos is not None:
                    self.aux_obstacle_pos[step, i] = pos
                    self.aux_obstacle_mask[step, i] = msk
            if "geodesic_dist" in self.aux_heads:
                gd = info.get('geodesic_distance_to_goal', 0.0)
                self.aux_geodesic_dist[step, i] = gd

    def flatten(self, advantages, returns, action_dim,
                context_size=None, obs_feat_dim=None,
                norm_adv=True):
        """Flatten (steps, envs, ...) -> (batch, ...) for PPO. Returns dict."""
        batch_size = self.num_steps * self.num_envs

        b_advantages = advantages.reshape(batch_size)
        if norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        result = {
            "goal": self.goal.reshape(batch_size, 2),
            "actions": self.actions.reshape(batch_size, action_dim),
            "logprobs": self.logprobs.reshape(batch_size),
            "values": self.values.reshape(batch_size),
            "advantages": b_advantages,
            "returns": returns.reshape(batch_size),
            "dones": self.dones.reshape(batch_size),
            # GPU tensors — already on device
            "features": self.features_gpu.reshape(batch_size, self.num_tokens, self.obs_feat_dim),
            "dec_out": self.dec_out_gpu.reshape(batch_size, self.obs_feat_dim),
        }
        if self.action_history_dim > 0:
            result["action_hist"] = self.action_hist.reshape(batch_size, self.action_history_dim)

        # Auxiliary targets
        if "occupancy" in self.aux_heads:
            G = self.aux_occupancy.shape[-1]
            result["aux_occupancy"] = self.aux_occupancy.reshape(batch_size, G, G)
        if "obstacle_pos" in self.aux_heads:
            K = self.aux_obstacle_pos.shape[-2]
            result["aux_obstacle_pos"] = self.aux_obstacle_pos.reshape(batch_size, K, 2)
            result["aux_obstacle_mask"] = self.aux_obstacle_mask.reshape(batch_size, K)
        if "geodesic_dist" in self.aux_heads:
            result["aux_geodesic_dist"] = self.aux_geodesic_dist.reshape(batch_size)

        return result


# ─── Discrete Rollout Buffers ────────────────────────────────────────
class DiscreteRolloutBuffers(RolloutBuffers):
    """RolloutBuffers variant for discrete (Categorical) action spaces.

    Actions are stored as int64 indices (T, E) instead of float32 (T, E, D).
    Also stores the 2D displacement each action maps to (for diagnostics/vis).
    """

    def __init__(self, num_steps, num_envs, obs_shape, cord_shape,
                 num_actions, num_tokens, obs_feat_dim, device,
                 action_history_dim=0):
        # Initialise parent with action_dim=2 (allocates float actions buffer
        # that we'll override — the 2 is just a placeholder).
        super().__init__(
            num_steps, num_envs, obs_shape, cord_shape, 2,
            num_tokens, obs_feat_dim, device,
            action_history_dim=action_history_dim,
        )
        self.num_actions = num_actions
        # Override: discrete action indices
        self.actions = np.zeros((num_steps, num_envs), dtype=np.int64)
        # Mirror: 2D displacement for each discrete action (vis / diagnostics)
        self.actions_2d = np.zeros((num_steps, num_envs, 2), dtype=np.float32)

    def flatten(self, advantages, returns, action_dim,
                context_size=None, obs_feat_dim=None, norm_adv=True):
        """Flatten for PPO.  ``action_dim`` is ignored (kept for API compat)."""
        batch_size = self.num_steps * self.num_envs

        b_advantages = advantages.reshape(batch_size)
        if norm_adv:
            b_advantages = ((b_advantages - b_advantages.mean())
                            / (b_advantages.std() + 1e-8))

        result = {
            "goal": self.goal.reshape(batch_size, 2),
            "actions": self.actions.reshape(batch_size),   # (T*E,) int64
            "logprobs": self.logprobs.reshape(batch_size),
            "values": self.values.reshape(batch_size),
            "advantages": b_advantages,
            "returns": returns.reshape(batch_size),
            "dones": self.dones.reshape(batch_size),
            "features": self.features_gpu.reshape(
                batch_size, self.num_tokens, self.obs_feat_dim),
            "dec_out": self.dec_out_gpu.reshape(batch_size, self.obs_feat_dim),
        }
        if self.action_history_dim > 0:
            result["action_hist"] = self.action_hist.reshape(
                batch_size, self.action_history_dim)
        return result
