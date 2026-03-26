import torch
import torch.nn as nn


class LSTMDoubleCritic(nn.Module):
    def __init__(self, obs_feature_dim, hidden_dim, num_layers, action_dim):
        super().__init__()

        # unshared LSTM with the actor
        self.lstm = nn.LSTM(obs_feature_dim+2, hidden_dim, num_layers, batch_first=True)
        
        # two Q-networks (for double Q-learning)
        # action_dim: 2 * trajectory length
        self.q1 = nn.Sequential(        # Q1
            nn.Linear(hidden_dim+action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(        # Q2
            nn.Linear(hidden_dim+action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs_tokens, target_tokens, x_init):
        """
        Parameters
        ----------
        obs_tokens: (B, H, input dim)
        target_tokens: (B, H, target dim)
        x_init: (B, N, 2) - action
        """
        B, N, _ = obs_tokens.shape
        
        # 1. Process Temporal Context

        combined_input = torch.cat([obs_tokens, target_tokens], dim=-1)
        
        # We only care about the last output for the current state representation
        lstm_out, _ = self.lstm(combined_input)
        state_repr = lstm_out[:, -1, :] # (B, hidden_dim)
        
        # 2. Flatten Action (Seed)
        action_flat = x_init.reshape(B, -1) # (B, 24)
        
        # 3. Concatenate State and Action
        sa_input = torch.cat([state_repr, action_flat], dim=-1)
        
        # 4. Output twin Q-values
        return self.q1(sa_input), self.q2(sa_input)