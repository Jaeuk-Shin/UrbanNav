import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_utils import PositionalEncoding
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from model.residual_mlp import ResidualMLP

class FlowMatchingActionDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.context_size = cfg.model.obs_encoder.context_size
        self.cord_embedding_type = cfg.model.cord_embedding.type
        self.decoder_type = cfg.model.decoder.type
        self.encoder_feat_dim = cfg.model.encoder_feat_dim
        self.len_traj_pred = cfg.model.decoder.len_traj_pred
        self.output_coordinate_repr = cfg.model.output_coordinate_repr  # 'polar' or 'euclidean'

        self.n_inference_steps = 10

        # decoder      
        assert cfg.model.decoder.type == "flow_matching"
        
        self.positional_encoding = PositionalEncoding(self.encoder_feat_dim, max_seq_len=self.context_size+1)
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_feat_dim, 
            nhead=cfg.model.decoder.num_heads,
            dim_feedforward=cfg.model.decoder.ff_dim_factor*self.encoder_feat_dim, 
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=cfg.model.decoder.num_layers)
        self.wp_predictor = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=self.encoder_feat_dim,
            down_dims=[64, 128, 256],
            cond_predict_scale=True,                    # enable FiLM
        )
        if hasattr(cfg, 'checkpoint'):
            checkpoint_path = cfg.checkpoint
            self.load_from_checkpoint(checkpoint_path=checkpoint_path)

    @torch.no_grad()
    def forward(self, tokens, actions):
        n_steps = self.n_inference_steps
        tokens = self.positional_encoding(tokens)
        dec_out = self.sa_decoder(tokens).mean(dim=1)      # (B, latent_dim)
        dt = 1.0 / n_steps
        
        B = dec_out.size(0)
        for i in range(self.n_inference_steps):
            # current t in [0, 1]
            t = i / n_steps
            # scale t 
            t_tensor = torch.full((B,), t * 1000, device=self.wp_predictor.device)
            
            vt = self.wp_predictor(
                sample=actions, 
                timestep=t_tensor, 
                global_cond=dec_out
            )
            # forward Euler
            xt = xt + vt * dt

        wp_pred = xt[:, :self.len_traj_pred]
        # convert deltas to absolute positions
        wp_pred = torch.cumsum(wp_pred, dim=1)
        
        # -> (batch size, future length, 2)
        return wp_pred.view(B, self.len_traj_pred, 2)
        
    def load_from_checkpoint(self, checkpoint_path):
        assert os.path.exists(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint["state_dict"]

        state_dict_without_prefix = {}
        for key, val in state_dict.items():
            if key.startswith('model.'):
                key_without_prefix = key.replace('model.', '')
                state_dict_without_prefix[key_without_prefix] = val
        
        self.load_state_dict(state_dict_without_prefix, strict=False)
        print(f'loaded a checkpoint from {checkpoint_path}')
        return
    


class DistilledActionDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.context_size = cfg.model.obs_encoder.context_size
        self.cord_embedding_type = cfg.model.cord_embedding.type
        self.decoder_type = cfg.model.decoder.type
        self.encoder_feat_dim = cfg.model.encoder_feat_dim
        self.len_traj_pred = cfg.model.decoder.len_traj_pred
        self.output_coordinate_repr = cfg.model.output_coordinate_repr  # 'polar' or 'euclidean'

        self.n_inference_steps = 10

        # decoder      
        assert cfg.model.decoder.type == "flow_matching"
        
        self.positional_encoding = PositionalEncoding(self.encoder_feat_dim, max_seq_len=self.context_size+1)
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_feat_dim, 
            nhead=cfg.model.decoder.num_heads,
            dim_feedforward=cfg.model.decoder.ff_dim_factor*self.encoder_feat_dim, 
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=cfg.model.decoder.num_layers)
        
        out_dim = 12 * 2

        self.wp_predictor = ResidualMLP(
            input_dim=self.encoder_feat_dim+out_dim,
            output_dim=2*self.len_traj_pred,
            hidden_dim=1024, 
            num_blocks=3
        )
        # TODO: check if the trained weights of the distilled network are successfully loaded

        if hasattr(cfg, 'checkpoint'):
            checkpoint_path = cfg.checkpoint
            self.load_from_checkpoint(checkpoint_path=checkpoint_path)

    @torch.no_grad()
    def forward(self, tokens, noise):

        n_steps = self.n_inference_steps
        tokens = self.positional_encoding(tokens)
        dec_out = self.sa_decoder(tokens).mean(dim=1)      # (B, latent_dim)
        batch_size = dec_out.size(0)
        noise = noise.view(batch_size, -1, 2)     # unflatten
        padding_size = 12 - noise.size(1)
        if padding_size > 0:
            noise = F.pad(input=noise, pad=(0, 0, 0, padding_size))

        noise = noise.view(batch_size, -1)        # flatten
        inputs = torch.cat([dec_out, noise], dim=-1)
        actions = self.wp_predictor(inputs)
        actions = actions.view(batch_size, -1, 2)
        actions = actions[:, :self.len_traj_pred]
        # deltas -> absolute positions
        actions = torch.cumsum(actions, dim=1)
        return actions.view(batch_size, -1)
        
    def load_from_checkpoint(self, checkpoint_path):
        assert os.path.exists(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint["state_dict"]

        state_dict_without_prefix = {}
        for key, val in state_dict.items():
            if key.startswith('model.'):
                key_without_prefix = key.replace('model.', '')
                state_dict_without_prefix[key_without_prefix] = val
        
        self.load_state_dict(state_dict_without_prefix, strict=False)
        print(f'loaded a checkpoint from {checkpoint_path}')
        return





if __name__ == '__main__':
    import argparse
    from config.utils import load_config

    parser = argparse.ArgumentParser(description='Train UrbanNav model')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint. If not provided, the latest checkpoint will be used.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    encoder = FlowMatchingActionDecoder(cfg)