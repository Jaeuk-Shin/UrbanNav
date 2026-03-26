import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

from model.model_utils import PolarEmbedding, MultiLayerDecoder, PositionalEncoding
from torchvision import models
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class CityWalker(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.context_size = cfg.model.obs_encoder.context_size
        self.obs_encoder_type = cfg.model.obs_encoder.type
        self.cord_embedding_type = cfg.model.cord_embedding.type
        self.decoder_type = cfg.model.decoder.type
        self.encoder_feat_dim = cfg.model.encoder_feat_dim
        self.len_traj_pred = cfg.model.decoder.len_traj_pred
        self.do_rgb_normalize = cfg.model.do_rgb_normalize
        self.do_resize = cfg.model.do_resize
        self.output_coordinate_repr = cfg.model.output_coordinate_repr  # 'polar' or 'euclidean'

        self.crop = cfg.model.obs_encoder.crop
        self.resize = cfg.model.obs_encoder.resize

        if self.do_rgb_normalize:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.obs_encoder = torch.hub.load('facebookresearch/dinov2', self.obs_encoder_type)
        feature_dim = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536,
        }
        if cfg.model.obs_encoder.freeze:
            for param in self.obs_encoder.parameters():
                param.requires_grad = False
        self.num_obs_features = feature_dim[self.obs_encoder_type]


        if self.num_obs_features != self.encoder_feat_dim:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.encoder_feat_dim)
        else:
            self.compress_obs_enc = nn.Identity()


        if self.cord_embedding_type == 'polar':
            self.cord_embedding = PolarEmbedding(cfg)
            self.dim_cord_embedding = self.cord_embedding.out_dim * self.context_size
        elif self.cord_embedding_type == 'target':
            self.cord_embedding = PolarEmbedding(cfg)
            self.dim_cord_embedding = self.cord_embedding.out_dim
        elif self.cord_embedding_type == 'input_target':
            self.cord_embedding = PolarEmbedding(cfg)
            self.dim_cord_embedding = self.cord_embedding.out_dim * self.context_size
        else:
            raise NotImplementedError(f"Coordinate embedding type {self.cord_embedding_type} not implemented")

        if self.dim_cord_embedding != self.encoder_feat_dim:
            self.compress_goal_enc = nn.Linear(self.dim_cord_embedding, self.encoder_feat_dim)
        else:
            self.compress_goal_enc = nn.Identity()

        # Decoder
        
        assert cfg.model.decoder.type == "diff_policy"
        
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
            cond_predict_scale=False,
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.model.decoder.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        
    def forward(self, obs, cord, gt_action=None):
        """
        Args:
            obs: (B, N, 3, H, W) tensor
            cord: (B, N, 2) tensor
        """
        B, N, _, H, W = obs.shape
        obs = obs.view(B * N, 3, H, W)
        if self.do_rgb_normalize:
            obs = (obs - self.mean) / self.std
        if self.do_resize:
            obs = TF.center_crop(obs, self.crop)
            obs = TF.resize(obs, self.resize)
        
        obs_enc = self.obs_encoder(obs)
        obs_enc = self.compress_obs_enc(obs_enc).view(B, N, -1)

        # Coordinate Encoding
        cord_enc = self.cord_embedding(cord).view(B, -1)
        cord_enc = self.compress_goal_enc(cord_enc).view(B, 1, -1)
        tokens = torch.cat([obs_enc, cord_enc], dim=1)
        


        # Decoder
        
        assert self.decoder_type == "diff_policy"
        assert gt_action is not None

        tokens = self.positional_encoding(tokens)
        dec_out = self.sa_decoder(tokens).mean(dim=1)
        
        deltas = torch.diff(gt_action, dim=1, prepend=torch.zeros_like(gt_action[:, :1, :]))    # (batch size, # of waypoints, 2)
        '''
        if self.output_coordinate_repr == 'polar':
            distances = torch.norm(deltas, dim=-1)
            angles = torch.atan2(deltas[:, :, 1], deltas[:, :, 0])
            deltas = torch.stack([distances, angles], dim=-1)
        '''
        if self.output_coordinate_repr == 'euclidean':
            pass
        else:
            raise NotImplementedError(f"Output coordinate representation {self.output_coordinate_repr} not implemented")
        noise = torch.randn_like(deltas)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=noise.device).long()
        noisy_action = self.noise_scheduler.add_noise(deltas, noise, timesteps)     # forward diffusion
        
        # Pad noisy_action with zeros to make the second dimension 12
        # Refer to https://github.com/real-stanford/diffusion_policy/issues/32#issuecomment-1834622174
        padding_size = 12 - noisy_action.size(1)
        if padding_size > 0:
            noisy_action_pad = F.pad(noisy_action, (0, 0, 0, padding_size))
        
        # score function
        noise_pred = self.wp_predictor(sample=noisy_action_pad, timestep=timesteps, global_cond=dec_out)
        noise_pred = noise_pred[:, :self.len_traj_pred]
        alpha_cumprod = self.noise_scheduler.alphas_cumprod[timesteps].view(B, 1, 1)
        # reverse diffusion
        wp_pred = (noisy_action - noise_pred * (1 - alpha_cumprod).sqrt()) / alpha_cumprod.sqrt()
        '''
        if self.output_coordinate_repr == 'polar':
            distances = wp_pred[:, :, 0]
            angles = wp_pred[:, :, 1]
            dx = distances * torch.cos(angles)
            dy = distances * torch.sin(angles)
            wp_pred = torch.stack([dx, dy], dim=-1)
        '''
        wp_pred = torch.cumsum(wp_pred, dim=1)  # to waypoints

        return wp_pred, noise_pred, noise

    @torch.no_grad
    def sample(self, obs, cord, num_samples=5):
        """
        Args:
            obs: (batch size, history length, 3, H, W) tensor
            num_samples: # of trajectory samples to generate per observation
        """
        B, N, _, H, W = obs.shape   # B: batch size, N: history length
        device = obs.device
        
        # 1. Encode context (Observation features)
        # We repeat these features B * num_samples times to generate in parallel
        obs_flat = obs.view(B * N, -1, H, W)
        if self.do_rgb_normalize:
            obs_flat = (obs_flat - self.mean) / self.std
        if self.do_resize:
            obs_flat = TF.center_crop(obs_flat, self.crop)
            obs_flat = TF.resize(obs_flat, self.resize)
            
        obs_enc = self.obs_encoder(obs_flat)
        obs_enc = self.compress_obs_enc(obs_enc).view(B, N, -1)

        cord_enc = self.cord_embedding(cord).view(B, -1)
        cord_enc = self.compress_goal_enc(cord_enc).view(B, 1, -1)
        tokens = torch.cat([obs_enc, cord_enc], dim=1)

        tokens = self.positional_encoding(tokens)
        dec_out = self.sa_decoder(tokens).mean(dim=1) # (B, feat_dim)
        
        # Expand context for multiple sampling: (B * num_samples, feat_dim)
        dec_out = dec_out.repeat_interleave(num_samples, dim=0)

        # 2. Initialize noisy trajectories (B * num_samples, len_traj_pred, 2)
        # Note: Use 12 for the second dim if your Unet expects fixed 1d size
        noisy_action = torch.randn((B * num_samples, 12, 2), device=device)
        
        # 3. Reverse Diffusion Loop
        self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
        
        for t in self.noise_scheduler.timesteps:
            # Predict noise residual
            noise_pred = self.wp_predictor(
                sample=noisy_action, 
                timestep=t, 
                global_cond=dec_out
            )
            
            # Compute the previous noisy sample (denoising step)
            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=noisy_action
            ).prev_sample

        # 4. Post-process trajectories
        # Trim padding if necessary
        wp_pred = noisy_action[:, :self.len_traj_pred]
        '''
        if self.output_coordinate_repr == 'polar':
            distances, angles = wp_pred[:, :, 0], wp_pred[:, :, 1]
            wp_pred = torch.stack([distances * torch.cos(angles), distances * torch.sin(angles)], dim=-1)
        '''
        # Convert deltas to absolute positions
        wp_pred = torch.cumsum(wp_pred, dim=1)
        
        # to (batch size, # of samples, prediction length, 2)
        return wp_pred.view(B, num_samples, self.len_traj_pred, 2)
