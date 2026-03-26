"""
PPO agents with discrete (Categorical) action spaces.

* ``PPODiscreteAgent``          — full DINOv2+LSTM agent
* ``GoalOnlyDiscreteMLPAgent``  — minimal MLP baseline (goal-only, no vision)

Both use ``torch.distributions.Categorical`` and share the same
get_action_and_value / get_value interface as the continuous PPOAgent,
so the discrete trainer can swap them in transparently.
"""

import math

import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl.models.encoder import ObservationEncoder, SimpleObservationEncoder
from model.model_utils import PolarEmbedding


# ─── Full PPO Agent (DINOv2 + LSTM + discrete actor) ─────────────────


class PPODiscreteAgent(nn.Module):
    """Frozen DINOv2 encoder + learnable LSTM + discrete actor/critic heads."""

    def __init__(self, cfg, num_actions=4, lstm_hidden_dim=256,
                 lstm_num_layers=1, n_action_history=0,
                 goal_mode=None, norm_obs=False, encoder_type="full"):
        super().__init__()

        self.num_actions = num_actions
        self.n_action_history = n_action_history
        self.norm_obs = norm_obs
        self.encoder_type = encoder_type

        # goal_mode ∈ {"lstm", "heads", "both"}
        #   "lstm"  — goal fused into LSTM input only (default)
        #   "heads" — goal concatenated with head input only
        #   "both"  — goal fed to LSTM *and* skip-connected to heads
        if goal_mode is not None:
            assert goal_mode in ("lstm", "heads", "both"), \
                f"goal_mode must be 'lstm', 'heads', or 'both', got '{goal_mode}'"
            self.goal_mode = goal_mode
        else:
            self.goal_mode = "lstm"

        # ── frozen encoder ──
        if encoder_type == "simple":
            self.obs_encoder = SimpleObservationEncoder(cfg)
        else:
            self.obs_encoder = ObservationEncoder(cfg)
        for p in self.obs_encoder.parameters():
            p.requires_grad = False

        # ── dimensions ──
        obs_feat_dim = cfg.model.encoder_feat_dim   # 768
        self.context_size = cfg.model.obs_encoder.context_size
        # action history: one-hot per step
        action_history_dim = n_action_history * num_actions

        # ── goal projection ──
        if encoder_type == "simple":
            self.goal_embedding = PolarEmbedding(cfg)       # standalone
        else:
            self.goal_embedding = self.obs_encoder.cord_embedding
        goal_feat_dim = self.goal_embedding.out_dim
        self.goal_feat_dim = goal_feat_dim
        self.goal_mlp = nn.Linear(goal_feat_dim, goal_feat_dim)
        self.goal_dir_mlp = nn.Linear(goal_feat_dim, goal_feat_dim)

        # ── LSTM backbone ──
        # "heads" mode: LSTM sees only visual features (+ action history if enabled)
        # "lstm" / "both": LSTM sees visual features + goal
        goal_to_lstm = self.goal_mode in ("lstm", "both")
        lstm_input_dim = obs_feat_dim + goal_feat_dim if goal_to_lstm else obs_feat_dim
        lstm_input_dim += action_history_dim  # action history feeds into LSTM
        self.lstm_layer_norm = nn.LayerNorm(lstm_input_dim) if norm_obs else None

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
        )
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers

        # ── actor / critic heads ──
        # "lstm" mode: heads see only LSTM output
        # "heads" / "both": heads see LSTM output + goal + goal direction (skip)
        goal_to_head = self.goal_mode in ("heads", "both")
        head_input_dim = lstm_hidden_dim
        if goal_to_head:
            head_input_dim += 2 * goal_feat_dim
        self.actor_head = nn.Linear(head_input_dim, num_actions)
        self.value_head = nn.Linear(head_input_dim, 1)

        self._init_weights()

    # -- weight init ---------------------------------------------------

    def _init_weights(self):
        nn.init.orthogonal_(self.goal_mlp.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.goal_mlp.bias)
        nn.init.orthogonal_(self.goal_dir_mlp.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.goal_dir_mlp.bias)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    # -- helpers -------------------------------------------------------

    @property
    def num_tokens(self):
        """Number of feature tokens produced by the encoder per observation."""
        if self.encoder_type == "simple":
            return 1
        return self.context_size + 1

    def get_initial_lstm_state(self, batch_size=1, device="cpu"):
        h = torch.zeros(self.lstm_num_layers, batch_size,
                         self.lstm_hidden_dim, device=device)
        c = torch.zeros(self.lstm_num_layers, batch_size,
                         self.lstm_hidden_dim, device=device)
        return h, c

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # -- forward -------------------------------------------------------

    def _encode(self, obs, cord, goal, lstm_state, features=None, action_history=None):
        """Shared forward: encoder → (optional goal/goal_dir/action) → LSTM."""
        if features is None:
            with torch.no_grad():
                if self.encoder_type == "simple":
                    obs_in = obs[:, -1:] if obs.dim() == 5 else obs
                    features = self.obs_encoder(obs_in)
                else:
                    features = self.obs_encoder(obs, cord)

        goal_feat = self.goal_embedding(goal.unsqueeze(1))
        goal_feat = self.goal_mlp(goal_feat)
        goal_vec = goal_feat[:, 0, :]

        # Goal direction: heading-aware unit vector from ego to goal
        goal_dir = goal / goal.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        goal_dir_feat = self.goal_embedding(goal_dir.unsqueeze(1))
        goal_dir_feat = self.goal_dir_mlp(goal_dir_feat)
        goal_dir_vec = goal_dir_feat[:, 0, :]

        if self.goal_mode in ("lstm", "both"):
            lstm_input = torch.cat(
                [features, goal_feat.expand(-1, features.size(1), -1)],
                dim=-1,
            )
        else:  # "heads"
            lstm_input = features

        # Action history feeds into LSTM
        if action_history is not None and self.n_action_history > 0:
            ah_expanded = action_history.unsqueeze(1).expand(-1, lstm_input.size(1), -1)
            lstm_input = torch.cat([lstm_input, ah_expanded], dim=-1)

        if self.lstm_layer_norm is not None:
            lstm_input = self.lstm_layer_norm(lstm_input)

        lstm_out, new_state = self.lstm(lstm_input, lstm_state)
        last_h = lstm_out[:, -1, :]
        return last_h, features, new_state, goal_vec, goal_dir_vec

    def get_action_and_value(self, obs, cord, goal, lstm_state, action=None,
                             features=None, dec_out=None, action_history=None):
        """
        Rollout (action=None): sample a discrete action.
        Training (action given): re-evaluate log_prob & value.

        Returns: action (long), log_prob, entropy, value, new_lstm_state
        """
        last_h, features, new_state, goal_vec, goal_dir_vec = self._encode(
            obs, cord, goal, lstm_state, features=features,
            action_history=action_history,
        )

        head_input = last_h
        if self.goal_mode in ("heads", "both"):
            head_input = torch.cat([head_input, goal_vec, goal_dir_vec], dim=-1)

        logits = self.actor_head(head_input)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value_head(head_input).squeeze(-1)
        return action, log_prob, entropy, value, new_state

    def get_action_and_value_sequential(self, goal, initial_lstm_state, dones,
                                        num_steps, actions,
                                        features=None, dec_out=None,
                                        action_history=None):
        """Sequential LSTM replay for PPO update (discrete actions)."""
        envsperbatch = initial_lstm_state[0].shape[1]

        goal_feat = self.goal_embedding(goal.unsqueeze(1))
        goal_feat = self.goal_mlp(goal_feat)
        goal_vec = goal_feat[:, 0, :]

        # Goal direction embedding
        goal_dir = goal / goal.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        goal_dir_feat = self.goal_embedding(goal_dir.unsqueeze(1))
        goal_dir_feat = self.goal_dir_mlp(goal_dir_feat)
        goal_dir_vec = goal_dir_feat[:, 0, :]

        if self.goal_mode in ("lstm", "both"):
            lstm_input = torch.cat(
                [features, goal_feat.expand(-1, features.size(1), -1)],
                dim=-1,
            )
        else:  # "heads"
            lstm_input = features

        # Action history feeds into LSTM
        if action_history is not None and self.n_action_history > 0:
            ah_expanded = action_history.unsqueeze(1).expand(-1, lstm_input.size(1), -1)
            lstm_input = torch.cat([lstm_input, ah_expanded], dim=-1)

        if self.lstm_layer_norm is not None:
            lstm_input = self.lstm_layer_norm(lstm_input)

        lstm_input = lstm_input.reshape(
            num_steps, envsperbatch, *lstm_input.shape[1:])
        dones_2d = dones.reshape(num_steps, envsperbatch)

        lstm_state = initial_lstm_state
        all_last_h = []
        for t in range(num_steps):
            lstm_out, lstm_state = self.lstm(lstm_input[t], lstm_state)
            last_h = lstm_out[:, -1, :]
            all_last_h.append(last_h)
            lstm_state = (
                ((1.0 - dones_2d[t]).view(1, -1, 1) * lstm_state[0]).contiguous(),
                ((1.0 - dones_2d[t]).view(1, -1, 1) * lstm_state[1]).contiguous(),
            )

        last_h = torch.stack(all_last_h, dim=0).reshape(
            -1, self.lstm_hidden_dim)

        head_input = last_h
        if self.goal_mode in ("heads", "both"):
            head_input = torch.cat([head_input, goal_vec, goal_dir_vec], dim=-1)

        logits = self.actor_head(head_input)
        dist = Categorical(logits=logits)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.value_head(head_input).squeeze(-1)
        return actions, log_prob, entropy, value, lstm_state

    def get_value(self, obs, cord, goal, lstm_state, action_history=None):
        last_h, _, _, goal_vec, goal_dir_vec = self._encode(
            obs, cord, goal, lstm_state, action_history=action_history,
        )
        head_input = last_h
        if self.goal_mode in ("heads", "both"):
            head_input = torch.cat([head_input, goal_vec, goal_dir_vec], dim=-1)
        return self.value_head(head_input).squeeze(-1)


# ─── Goal-Only MLP Baseline (discrete) ───────────────────────────────


class GoalOnlyDiscreteMLPAgent(nn.Module):
    """
    goal (2D) → MLP → Categorical actor (num_actions) + critic (1).

    Interface-compatible with PPODiscreteAgent.
    """

    def __init__(self, num_actions=4, hidden_dim=64, num_layers=2,
                 n_action_history=0):
        super().__init__()
        self.num_actions = num_actions
        self.n_action_history = n_action_history
        self.action_history_dim = n_action_history * num_actions

        # Dummy LSTM attributes (RolloutBuffers compat)
        self.lstm_num_layers = 1
        self.lstm_hidden_dim = 1

        # ── shared backbone ──
        layers = []
        in_dim = 2 + self.action_history_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # ── actor / critic ──
        self.actor_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.backbone:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def trainable_parameters(self):
        return list(self.parameters())

    def get_initial_lstm_state(self, batch_size=1, device="cpu"):
        h = torch.zeros(1, batch_size, 1, device=device)
        c = torch.zeros(1, batch_size, 1, device=device)
        return h, c

    def get_action_and_value(self, obs, cord, goal, lstm_state, action=None,
                             features=None, dec_out=None, action_history=None):
        inp = goal
        if action_history is not None and self.n_action_history > 0:
            inp = torch.cat([inp, action_history], dim=-1)
        h = self.backbone(inp)
        logits = self.actor_head(h)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value_head(h).squeeze(-1)
        return action, log_prob, entropy, value, lstm_state

    def get_value(self, obs, cord, goal, lstm_state, action_history=None):
        inp = goal
        if action_history is not None and self.n_action_history > 0:
            inp = torch.cat([inp, action_history], dim=-1)
        h = self.backbone(inp)
        return self.value_head(h).squeeze(-1)
