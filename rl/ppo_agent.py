import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


from rl.models.encoder import ObservationEncoder, SimpleObservationEncoder
from rl.models.decoder import DistilledActionDecoder
from model.model_utils import PolarEmbedding


class PPOAgent(nn.Module):
    """
    Frozen DINOv2 encoder  +  frozen distilled decoder
    + learnable LSTM  +  actor/critic heads
    """

    def __init__(self, cfg, lstm_hidden_dim=256, lstm_num_layers=1, use_decoder=False,
                 n_action_history=0, goal_mode=None,
                 norm_obs=False, encoder_type="full"):
        super().__init__()

        self.use_decoder = use_decoder
        self.n_action_history = n_action_history
        self.norm_obs = norm_obs
        self.encoder_type = encoder_type

        # goal_mode in {"lstm", "heads", "both"}
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

        if self.use_decoder:
            self.action_decoder = DistilledActionDecoder(cfg)
            for p in self.action_decoder.parameters():
                p.requires_grad = False

        # ── dimensions ──
        obs_feat_dim = cfg.model.encoder_feat_dim
        self.len_traj_pred = cfg.model.decoder.len_traj_pred
        self.context_size = cfg.model.obs_encoder.context_size
        action_dim = self.len_traj_pred * 2 if self.use_decoder else 2
        action_history_dim = n_action_history * action_dim

        # ── goal projection ──
        if encoder_type == "simple":
            self.goal_embedding = PolarEmbedding(cfg)       # standalone
        else:
            self.goal_embedding = self.obs_encoder.cord_embedding   # frozen
        goal_feat_dim = self.goal_embedding.out_dim             # 26
        self.goal_feat_dim = goal_feat_dim
        self.goal_mlp = nn.Linear(goal_feat_dim, goal_feat_dim)
        self.goal_dir_mlp = nn.Linear(goal_feat_dim, goal_feat_dim)

        # ── recurrent backbone ──
        # "heads" mode: LSTM sees only visual features (+ action history if enabled)
        # "lstm" / "both": LSTM sees visual features + goal
        goal_to_lstm = self.goal_mode in ("lstm", "both")
        lstm_input_dim = obs_feat_dim + goal_feat_dim if goal_to_lstm else obs_feat_dim
        lstm_input_dim += action_history_dim  # action history feeds into LSTM

        # ── optional LayerNorm on LSTM input ──
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
        self.mu_head = nn.Linear(head_input_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.value_head = nn.Linear(head_input_dim, 1)

        self._init_weights()

    # -- weight init ---------------------------------------------------

    def _init_weights(self):
        """Orthogonal init (CleanRL convention) for learnable layers only."""
        # goal_mlp
        nn.init.orthogonal_(self.goal_mlp.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.goal_mlp.bias)
        # goal_dir_mlp
        nn.init.orthogonal_(self.goal_dir_mlp.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.goal_dir_mlp.bias)
        # LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.zeros_(param)
        # actor — small init for stable exploration
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.zeros_(self.mu_head.bias)
        # critic
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
        h = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_dim, device=device)
        c = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_dim, device=device)
        return h, c

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _decode_from_dec_out(self, dec_out, noise):
        """Run only the MLP of the distilled decoder with pre-computed transformer output."""
        decoder = self.action_decoder
        batch_size = dec_out.size(0)
        noise = noise.view(batch_size, -1, 2)
        padding_size = 12 - noise.size(1)
        if padding_size > 0:
            noise = F.pad(input=noise, pad=(0, 0, 0, padding_size))
        noise = noise.view(batch_size, -1)
        inputs = torch.cat([dec_out, noise], dim=-1)
        actions = decoder.wp_predictor(inputs)
        actions = actions.view(batch_size, -1, 2)
        actions = actions[:, :self.len_traj_pred]
        actions = torch.cumsum(actions, dim=1)
        return actions.view(batch_size, -1)

    # -- forward -------------------------------------------------------

    def _encode(self, obs, cord, goal, lstm_state, features=None, action_history=None):
        """Shared forward: encoder → (optional goal/goal_dir/action) → LSTM.

        Returns (last_h, features, new_lstm_state, goal_vec, goal_dir_vec).
        ``goal_vec`` and ``goal_dir_vec`` are (B, goal_feat_dim); used by
        the heads when ``goal_mode`` is ``"heads"`` or ``"both"``.
        """
        if features is None:
            with torch.no_grad():
                if self.encoder_type == "simple":
                    # Take only the last frame from the history
                    obs_in = obs[:, -1:] if obs.dim() == 5 else obs
                    features = self.obs_encoder(obs_in)     # (B, 1, 768)
                else:
                    features = self.obs_encoder(obs, cord)  # (B, N+1, 768)

        goal_feat = self.goal_embedding(goal.unsqueeze(1))  # (B, 1, 26)
        goal_feat = self.goal_mlp(goal_feat)                # (B, 1, 26)
        goal_vec = goal_feat[:, 0, :]                       # (B, 26)

        # Goal direction: heading-aware unit vector from ego to goal
        goal_dir = goal / goal.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        goal_dir_feat = self.goal_embedding(goal_dir.unsqueeze(1))  # (B, 1, 26)
        goal_dir_feat = self.goal_dir_mlp(goal_dir_feat)            # (B, 1, 26)
        goal_dir_vec = goal_dir_feat[:, 0, :]                       # (B, 26)

        if self.goal_mode in ("lstm", "both"):
            lstm_input = torch.cat(
                [features, goal_feat.expand(-1, features.size(1), -1)], dim=-1
            )  # (B, N+1, 768+26)
        else:  # "heads"
            lstm_input = features

        # Action history feeds into LSTM
        if action_history is not None and self.n_action_history > 0:
            ah_expanded = action_history.unsqueeze(1).expand(-1, lstm_input.size(1), -1)
            lstm_input = torch.cat([lstm_input, ah_expanded], dim=-1)

        if self.lstm_layer_norm is not None:
            lstm_input = self.lstm_layer_norm(lstm_input)

        lstm_out, new_state = self.lstm(lstm_input, lstm_state)
        last_h = lstm_out[:, -1, :]                         # (B, hidden)
        return last_h, features, new_state, goal_vec, goal_dir_vec

    def get_action_and_value(self, obs, cord, goal, lstm_state, action=None,
                             features=None, dec_out=None, action_history=None):
        """
        Rollout (action=None): sample an action.
        Training (action given): re-evaluate log_prob & value.

        Returns: action, log_prob, entropy, value, new_lstm_state
        """
        last_h, features, new_state, goal_vec, goal_dir_vec = self._encode(
            obs, cord, goal, lstm_state, features=features,
            action_history=action_history,
        )

        head_input = last_h
        if self.goal_mode in ("heads", "both"):
            head_input = torch.cat([head_input, goal_vec, goal_dir_vec], dim=-1)

        # actor
        mu = self.mu_head(head_input)

        if self.use_decoder:
            # Decoder weights are frozen (requires_grad=False) so they won't
            # accumulate gradients, but we must NOT use torch.no_grad() here:
            # gradients need to flow through the decoder back to mu_head.
            if dec_out is not None:
                mu = self._decode_from_dec_out(dec_out, mu)
            else:
                mu = self.action_decoder(features, mu)   # (B, action_dim)

        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value_head(head_input).squeeze(-1)
        return action, log_prob, entropy, value, new_state

    def get_action_and_value_sequential(self, goal, initial_lstm_state, dones,
                                         num_steps, actions,
                                         features=None, dec_out=None,
                                         action_history=None):
        """
        Re-evaluate actions with sequential LSTM replay for PPO update.
        Preserves temporal ordering so gradients propagate through time (BPTT),
        and resets LSTM hidden state at episode boundaries.

        All tensor inputs are (T*E', ...) in step-major order, where
        T = num_steps, E' = envs_per_batch.

        Parameters
        ----------
        goal             : (T*E', 2)
        initial_lstm_state : tuple of (layers, E', hidden) — state at rollout start
        dones            : (T*E',) — done flags AFTER each step
        num_steps        : int (T)
        actions          : (T*E', action_dim) — actions to evaluate
        features         : (T*E', N+1, feat_dim) — precomputed encoder features
        dec_out          : (T*E', feat_dim) — precomputed decoder transformer output
        action_history   : (T*E', action_history_dim) or None
        """
        envsperbatch = initial_lstm_state[0].shape[1]

        # Goal embedding (batched over all T*E' steps)
        goal_feat = self.goal_embedding(goal.unsqueeze(1))  # (T*E', 1, goal_dim)
        goal_feat = self.goal_mlp(goal_feat)
        goal_vec = goal_feat[:, 0, :]                       # (T*E', goal_dim)

        # Goal direction embedding
        goal_dir = goal / goal.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        goal_dir_feat = self.goal_embedding(goal_dir.unsqueeze(1))
        goal_dir_feat = self.goal_dir_mlp(goal_dir_feat)
        goal_dir_vec = goal_dir_feat[:, 0, :]

        if self.goal_mode in ("lstm", "both"):
            lstm_input = torch.cat(
                [features, goal_feat.expand(-1, features.size(1), -1)], dim=-1
            )  # (T*E', N+1, feat_dim+goal_dim)
        else:  # "heads"
            lstm_input = features

        # Action history feeds into LSTM
        if action_history is not None and self.n_action_history > 0:
            ah_expanded = action_history.unsqueeze(1).expand(-1, lstm_input.size(1), -1)
            lstm_input = torch.cat([lstm_input, ah_expanded], dim=-1)

        if self.lstm_layer_norm is not None:
            lstm_input = self.lstm_layer_norm(lstm_input)

        # Reshape to (T, E', ...)
        lstm_input = lstm_input.reshape(num_steps, envsperbatch, *lstm_input.shape[1:])
        dones_2d = dones.reshape(num_steps, envsperbatch)

        # Sequential LSTM replay with done-based reset
        lstm_state = initial_lstm_state
        all_last_h = []
        for t in range(num_steps):
            lstm_out, lstm_state = self.lstm(lstm_input[t], lstm_state)
            last_h = lstm_out[:, -1, :]  # (E', hidden)
            all_last_h.append(last_h)
            # Reset LSTM state for the next step where this step ended an episode
            lstm_state = (
                ((1.0 - dones_2d[t]).view(1, -1, 1) * lstm_state[0]).contiguous(),
                ((1.0 - dones_2d[t]).view(1, -1, 1) * lstm_state[1]).contiguous(),
            )

        # (T, E', hidden) → (T*E', hidden) in step-major order
        last_h = torch.stack(all_last_h, dim=0).reshape(-1, self.lstm_hidden_dim)

        head_input = last_h
        if self.goal_mode in ("heads", "both"):
            head_input = torch.cat([head_input, goal_vec, goal_dir_vec], dim=-1)

        # Actor
        mu = self.mu_head(head_input)
        if self.use_decoder:
            if dec_out is not None:
                mu = self._decode_from_dec_out(dec_out, mu)
            else:
                mu = self.action_decoder(features, mu)

        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
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
