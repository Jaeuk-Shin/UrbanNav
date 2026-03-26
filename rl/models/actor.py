import torch
import torch.nn as nn
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.utils.annotations import override

from rl.models.encoder import ObservationEncoder


class DinoLSTMRModule(TorchRLModule):
    def __init__(self, rl_cfg: RLModuleConfig, cfg, hidden_dim, num_layers, output_dim):
        super().__init__(rl_cfg)
        self.cfg = cfg
        self.output_dim = output_dim
        
        # encoder (frozen)
        self.obs_encoder = ObservationEncoder(self.cfg)
        obs_feature_dim = self.cfg.model.encoder_feat_dim
        
        # 2. recurrent Layer
        self.lstm = nn.LSTM(
            obs_feature_dim+2,      # feature dim + goal dim 
            hidden_dim,
            num_layers, 
            batch_first=True
        )
        
        # 3. heads
        self.mu_head = nn.Linear(hidden_dim, output_dim)
        self.log_std_head = nn.Linear(hidden_dim, output_dim)



    @override
    def get_initial_state(self):
        # RLlib calls this to initialize the LSTM hidden/cell states
        return {
            "h": torch.zeros(self.lstm.num_layers, self.lstm.hidden_size),
            "c": torch.zeros(self.lstm.num_layers, self.lstm.hidden_size),
        }

    def _common_forward(self, batch):
        """helper to process observations through Encoder + LSTM"""
        obs = batch["obs"]
        cord = batch.get("cord") # Assumes 'cord' is passed in info or custom obs
        
        # Get features from your DINO encoder
        features = self.obs_encoder(obs, cord) # Shape: (B, T, D)
        
        # Handle states
        state_in = batch["state_in"]
        # Reshape states for torch: (L, B, H)
        h_in = state_in["h"].transpose(0, 1).contiguous()
        c_in = state_in["c"].transpose(0, 1).contiguous()
        
        lstm_out, (h_out, c_out) = self.lstm(features, (h_in, c_in))
        
        # Use last temporal output
        last_h = lstm_out[:, -1, :]
        
        mu = self.mu_head(last_h)
        log_std = torch.clamp(self.log_std_head(last_h), -20, 2)
        
        # Transpose states back for RLlib storage: (B, L, H)
        state_out = {
            "h": h_out.transpose(0, 1),
            "c": c_out.transpose(0, 1),
        }
        return mu, log_std, state_out

    @override
    def _forward_inference(self, batch):
        mu, _, state_out = self._common_forward(batch)
        return {"action": mu, "state_out": state_out}

    @override
    def _forward_exploration(self, batch):
        mu, log_std, state_out = self._common_forward(batch)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()
        return {
            "action": action, 
            "logp": dist.log_prob(action).sum(-1),
            "state_out": state_out
        }