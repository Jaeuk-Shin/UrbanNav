import os
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

from model.model_utils import PolarEmbedding, PositionalEncoding
from torchvision import models
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


class SimpleObservationEncoder(nn.Module):
    """DINOv2 encoder that takes only the current RGB image (no history, no cord).

    A lightweight baseline for RL: encodes a single frame into one feature token.
    """

    def __init__(self, cfg):
        super().__init__()
        self.obs_encoder_type = cfg.model.obs_encoder.type
        self.encoder_feat_dim = cfg.model.encoder_feat_dim

        # preprocessing
        self.do_rgb_normalize = cfg.model.do_rgb_normalize
        self.do_resize = cfg.model.do_resize
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
        self.num_obs_features = feature_dim[self.obs_encoder_type]

        if self.num_obs_features != self.encoder_feat_dim:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.encoder_feat_dim)
        else:
            self.compress_obs_enc = nn.Identity()

    @torch.no_grad()
    def forward(self, obs):
        """
        Args:
            obs: (B, H, W, C) uint8 single frame, or (B, 1, H, W, C) with N=1.
        Returns:
            (B, 1, encoder_feat_dim) feature token.
        """
        obs = process_frames(obs)
        # process_frames returns (B, C, H, W) for 4D input
        #                    or (B, 1, C, H, W) for 5D input
        if obs.dim() == 5:
            B = obs.shape[0]
            obs = obs[:, 0]  # (B, C, H, W)
        else:
            B = obs.shape[0]

        if self.do_rgb_normalize:
            obs = (obs - self.mean) / self.std
        if self.do_resize:
            obs = TF.center_crop(obs, self.crop)
            obs = TF.resize(obs, self.resize)

        obs_enc = self.obs_encoder(obs)                  # (B, feat_dim)
        obs_enc = self.compress_obs_enc(obs_enc)         # (B, encoder_feat_dim)
        return obs_enc.unsqueeze(1)                      # (B, 1, encoder_feat_dim)


class ObservationEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.context_size = cfg.model.obs_encoder.context_size
        self.obs_encoder_type = cfg.model.obs_encoder.type
        self.cord_embedding_type = cfg.model.cord_embedding.type
        self.decoder_type = cfg.model.decoder.type
        self.encoder_feat_dim = cfg.model.encoder_feat_dim

        # preprocessing
        self.do_rgb_normalize = cfg.model.do_rgb_normalize
        self.do_resize = cfg.model.do_resize
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
        self.num_obs_features = feature_dim[self.obs_encoder_type]

        if self.num_obs_features != self.encoder_feat_dim:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.encoder_feat_dim)
        else:
            self.compress_obs_enc = nn.Identity()

        assert self.cord_embedding_type == 'input_target'
        self.cord_embedding = PolarEmbedding(cfg)
        self.dim_cord_embedding = self.cord_embedding.out_dim * self.context_size
    
        if self.dim_cord_embedding != self.encoder_feat_dim:
            self.compress_goal_enc = nn.Linear(self.dim_cord_embedding, self.encoder_feat_dim)
        else:
            self.compress_goal_enc = nn.Identity()
        '''
        # freeze all parameters
        for net in [self.obs_encoder, self.compress_obs_enc, self.cord_embedding, self.compress_goal_enc]:
            for param in net.parameters():
                param.requires_grad = False
        '''
        if hasattr(cfg, 'checkpoint'):
            checkpoint_path = cfg.checkpoint
            self.load_from_checkpoint(checkpoint_path=checkpoint_path)

        return

    @torch.no_grad()
    def forward(self, obs, cord):

        obs = process_frames(obs)
        B, N, _, H, W = obs.shape
        obs = obs.view(B * N, 3, H, W)

        if self.do_rgb_normalize:
            obs = (obs - self.mean) / self.std
        if self.do_resize:
            obs = TF.center_crop(obs, self.crop)
            obs = TF.resize(obs, self.resize)

        obs_enc = self.obs_encoder(obs)
        obs_enc = self.compress_obs_enc(obs_enc).view(B, N, -1)
        # Handle flat cord input (B, N*2) from RL observation space
        if cord.dim() == 2:
            cord = cord.view(B, -1, 2)
        cord_enc = self.cord_embedding(cord).view(B, -1)
        cord_enc = self.compress_goal_enc(cord_enc).view(B, 1, -1)
        tokens = torch.cat([obs_enc, cord_enc], dim=1)
        return tokens

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


# Warning: This must match that of CityWalkDataset!
def process_frames(frames):
    """
    Normalize and reshape frames for the vision encoder.
    Input:  (B, N, H, W, C) uint8  — batched sequence (RL / inference)
            (B, H, W, C) uint8     — batched single frame (training)
    Output: same leading dims with C before H,W, bfloat16 in [0,1],
            spatially padded/cropped to 360x640.
    """
    if not isinstance(frames, torch.Tensor):
        frames = torch.as_tensor(np.asarray(frames))
    frames = frames.to(torch.bfloat16) / 255.0

    if frames.dim() == 5:
        # (B, N, H, W, C) -> (B, N, C, H, W)
        frames = frames.permute(0, 1, 4, 2, 3)
    elif frames.dim() == 4:
        # (B, H, W, C) -> (B, C, H, W)
        frames = frames.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Expected 4D or 5D input, got {frames.dim()}D")

    desired_height = 360
    desired_width = 640
    H, W = frames.shape[-2], frames.shape[-1]

    pad_height = desired_height - H
    pad_width = desired_width - W

    # Pad if smaller than target
    if pad_height > 0 or pad_width > 0:
        pad_top = max(pad_height // 2, 0)
        pad_bottom = max(pad_height - pad_top, 0)
        pad_left = max(pad_width // 2, 0)
        pad_right = max(pad_width - pad_left, 0)
        # F.pad works on any number of leading dims; pads last 2 dims
        frames = F.pad(frames, (pad_left, pad_right, pad_top, pad_bottom))

    # Center-crop if larger than target
    if pad_height < 0 or pad_width < 0:
        H, W = frames.shape[-2], frames.shape[-1]
        top = (H - desired_height) // 2
        left = (W - desired_width) // 2
        frames = frames[..., top:top + desired_height, left:left + desired_width]

    return frames
