"""
Minimal goal-only MLP baseline for PPO sanity checking.

Takes only the 2D goal coordinate as input and directly outputs actions.
No images, no pose history, no LSTM — just a small MLP with actor/critic heads.
"""

import math

import torch
import torch.nn as nn
from torch.distributions import Normal


class GoalOnlyMLPAgent(nn.Module):
    """
    goal (2D) → MLP → actor (action_dim) + critic (1).

    Interface-compatible with PPOAgent so the trainer can use either.
    """

    def __init__(self, hidden_dim=64, action_dim=2, num_layers=2,
                 n_action_history=0):
        super().__init__()
        self.action_dim = action_dim
        self.n_action_history = n_action_history
        self.action_history_dim = n_action_history * action_dim

        # Dummy attributes so RolloutBuffers allocation doesn't break.
        # The actual LSTM state is never used; get_initial_lstm_state returns
        # a tiny placeholder that the trainer carries around harmlessly.
        self.lstm_num_layers = 1
        self.lstm_hidden_dim = 1

        # ── shared backbone ──
        layers = []
        in_dim = 2 + self.action_history_dim  # goal (x, z) + flattened action history
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # ── actor ──
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # ── critic ──
        self.value_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.backbone:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def trainable_parameters(self):
        return list(self.parameters())

    def get_initial_lstm_state(self, batch_size=1, device="cpu"):
        """Return a tiny dummy state so the trainer's LSTM-state bookkeeping is happy."""
        h = torch.zeros(1, batch_size, 1, device=device)
        c = torch.zeros(1, batch_size, 1, device=device)
        return h, c

    # ── forward ──────────────────────────────────────────────────────

    def get_action_and_value(self, obs, cord, goal, lstm_state, action=None,
                             features=None, dec_out=None, action_history=None):
        """Same signature as PPOAgent. obs, cord, features, dec_out are ignored."""
        if action_history is not None and self.n_action_history > 0:
            goal = torch.cat([goal, action_history], dim=-1)
        h = self.backbone(goal)
        mu = self.mu_head(h)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value_head(h).squeeze(-1)
        return action, log_prob, entropy, value, lstm_state

    def get_value(self, obs, cord, goal, lstm_state, action_history=None):
        if action_history is not None and self.n_action_history > 0:
            goal = torch.cat([goal, action_history], dim=-1)
        h = self.backbone(goal)
        return self.value_head(h).squeeze(-1)
