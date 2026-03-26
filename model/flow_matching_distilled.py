import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from model.residual_mlp import ResidualMLP
from model.model_utils import PolarEmbedding, PositionalEncoding
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


class DistilledMLP(nn.Module):
    def __init__(self, teacher_model):
        super().__init__()
        # freeze the teacher model
        self.teacher = teacher_model
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        self.len_traj_pred = self.teacher.len_traj_pred

        # out_dim = self.teacher.len_traj_pred * 2
        out_dim = 12 * 2       # zero-padded inputs for UNet
        '''
        self.wp_predictor = nn.Sequential(
            nn.Linear(self.encoder_feat_dim+out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )
        '''
        # TODO: add FiLM here?
        self.wp_predictor = ResidualMLP(
            input_dim=self.teacher.encoder_feat_dim+out_dim,
            output_dim=2*self.len_traj_pred,
            hidden_dim=1024,        # Increased width for better expressivity
            num_blocks=5            # Sufficient depth for trajectory mapping
        )

        return

    def forward_student(self, dec_out, noise):
        # Flatten noise if it isn't already (B, traj_len * 2)
        z = noise.reshape(dec_out.size(0), -1)
        combined = torch.cat([dec_out, z], dim=-1)
        
        # Predict deltas directly
        deltas_pred = self.wp_predictor(combined)
        return deltas_pred.view(-1, self.teacher.len_traj_pred, 2)


    def forward(self, obs, cord):
        
        teacher = self.teacher
        B, N, _, H, W = obs.shape
        obs_flat = obs.view(B * N, 3, H, W)
        
        if teacher.do_rgb_normalize:
            obs_flat = (obs_flat - teacher.mean) / teacher.std
        if teacher.do_resize:
            obs_flat = TF.center_crop(obs_flat, teacher.crop)
            obs_flat = TF.resize(obs_flat, teacher.resize)
        obs_enc = teacher.obs_encoder(obs_flat)
        obs_enc = teacher.compress_obs_enc(obs_enc).view(B, N, -1)
        cord_enc = teacher.cord_embedding(cord).view(B, -1)
        cord_enc = teacher.compress_goal_enc(cord_enc).view(B, 1, -1)
        
        tokens = torch.cat([obs_enc, cord_enc], dim=1)
        tokens = teacher.positional_encoding(tokens)
        dec_out = teacher.sa_decoder(tokens).mean(dim=1)

        noise = torch.randn((B, self.len_traj_pred, 2), device=obs.device)
        padding_size = 12 - noise.size(1)
        if padding_size > 0:
            noise = F.pad(input=noise, pad=(0, 0, 0, padding_size))

        deltas_pred = self.forward_student(dec_out, noise)
        wp_pred = torch.cumsum(deltas_pred, dim=1)
        
        return {
        "wp_pred": wp_pred,
        "deltas_pred": deltas_pred,
        "noise": noise,
        "dec_out": dec_out
    }

    @torch.no_grad()
    def sample(self, obs, cord, num_samples=5):
        """
        Parameters
        ----------
        obs: (batch size, history length, 3, H, W) tensor
        num_samples: # of trajectory samples to generate per observation
        """
        teacher = self.teacher

        B, N, _, H, W = obs.shape
        obs_flat = obs.view(B * N, 3, H, W)
        
        device = obs.device

        # Preprocessing (Teacher's logic)
        if teacher.do_rgb_normalize:
            obs_flat = (obs_flat - teacher.mean) / teacher.std
        if teacher.do_resize:
            obs_flat = TF.center_crop(obs_flat, teacher.crop)
            obs_flat = TF.resize(obs_flat, teacher.resize)
            
        obs_enc = teacher.obs_encoder(obs_flat)
        obs_enc = teacher.compress_obs_enc(obs_enc).view(B, N, -1)
        cord_enc = teacher.cord_embedding(cord).view(B, -1)
        cord_enc = teacher.compress_goal_enc(cord_enc).view(B, 1, -1)
        
        tokens = torch.cat([obs_enc, cord_enc], dim=1)
        tokens = teacher.positional_encoding(tokens)
        dec_out = teacher.sa_decoder(tokens).mean(dim=1)
        
        # Expand context: (B * num_samples, feat_dim)
        dec_out = dec_out.repeat_interleave(num_samples, dim=0)

        # 2. Initialize with pure Gaussian noise (x0)
        # We start at t=0 (pure noise) and integrate toward t=1 (data)
        noise = torch.randn((B * num_samples, self.len_traj_pred, 2), device=device)
        padding_size = 12 - noise.size(1)
        if padding_size > 0:
            noise = F.pad(input=noise, pad=(0, 0, 0, padding_size))


        deltas_pred = self.forward_student(dec_out, noise)
        wp_pred = torch.cumsum(deltas_pred, dim=1)
        
        # -> (batch size, # of samples, prediction length, 2)
        return {
            "wp_pred": wp_pred.view(B, num_samples, self.len_traj_pred, 2),
            "deltas_pred": deltas_pred,     # (B * # of samples, prediction length, 2)
            "noise": noise,
            "dec_out": dec_out
        }
