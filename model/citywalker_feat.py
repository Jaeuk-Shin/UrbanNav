import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

from model.model_utils import PolarEmbedding, FeatPredictor, PositionalEncoding
from torchvision import models

class CityWalkerFeat(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.context_size = cfg.model.obs_encoder.context_size
        self.obs_encoder_type = cfg.model.obs_encoder.type
        self.cord_embedding_type = cfg.model.cord_embedding.type
        self.decoder_type = cfg.model.decoder.type
        self.len_traj_pred = cfg.model.decoder.len_traj_pred
        self.do_rgb_normalize = cfg.model.do_rgb_normalize
        self.do_resize = cfg.model.do_resize
        self.output_coordinate_repr = cfg.model.output_coordinate_repr  # 'polar' or 'euclidean'

        # if self.obs_encoder_type.startswith("dinov2"):
        self.crop = cfg.model.obs_encoder.crop
        self.resize = cfg.model.obs_encoder.resize

        if self.do_rgb_normalize:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Observation Encoder
        if self.obs_encoder_type.startswith("dinov2"):
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
        else:
            raise NotImplementedError(f"Observation encoder type {self.obs_encoder_type} not implemented")

        '''
        # Coordinate Embedding
        if self.cord_embedding_type == 'input_target':
            self.cord_embedding = PolarEmbedding(cfg)
            self.dim_cord_embedding = self.cord_embedding.out_dim * (self.context_size + 1)
            self.compress_goal_enc = nn.Linear(self.dim_cord_embedding, self.num_obs_features)
        else:
            raise NotImplementedError(f"Coordinate embedding type {self.cord_embedding_type} not implemented")
        '''
        # Decoder
        self.predictor = FeatPredictor(
            embed_dim=self.num_obs_features,
            seq_len=self.context_size,
            nhead=cfg.model.decoder.num_heads,
            num_layers=cfg.model.decoder.num_layers,
            ff_dim_factor=cfg.model.decoder.ff_dim_factor,
        )
        self.predictor_mlp = nn.Sequential(
            # nn.Linear((self.context_size+1) * self.num_obs_features, 256),
            nn.Linear(self.context_size * self.num_obs_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.wp_predictor = nn.Linear(32, self.len_traj_pred * 2)
        
        '''
        self.arrive_predictor = nn.Linear(32, 1)
        '''


    def forward(self, obs, cord, future_obs=None):
        """
        Args:
            obs: (B, N, 3, H, W) tensor
            cord: (B, N, 2) tensor
        """
        B, N, _, H, W = obs.shape
        obs = obs.view(B * N, 3, H, W)
        if future_obs is not None:
            future_obs = future_obs.view(B * N, 3, H, W)
        if self.do_rgb_normalize:
            obs = (obs - self.mean) / self.std
            if future_obs is not None:
                future_obs = (future_obs - self.mean) / self.std
        if self.do_resize:
            obs = TF.center_crop(obs, self.crop)
            obs = TF.resize(obs, self.resize)
            if future_obs is not None:
                future_obs = TF.center_crop(future_obs, self.crop)
                future_obs = TF.resize(future_obs, self.resize)

        obs_enc = self.obs_encoder(obs).view(B, N, -1)
        if future_obs is not None:
            future_obs_enc = self.obs_encoder(future_obs).view(B, N, -1)
        else:
            future_obs_enc = None

        '''
        # Coordinate Encoding
        cord_enc = self.cord_embedding(cord).view(B, -1)
        cord_enc = self.compress_goal_enc(cord_enc).view(B, 1, -1)
        tokens = torch.cat([obs_enc, cord_enc], dim=1)
        '''
        tokens = obs_enc

        # Decoder
        feature_pred = self.predictor(tokens) # (B, N+1, D)
        
        dec_out = self.predictor_mlp(feature_pred.view(B, -1))
        wp_pred = self.wp_predictor(dec_out).view(B, self.len_traj_pred, 2)
        print(wp_pred.shape)

        '''
        arrive_pred = self.arrive_predictor(dec_out).view(B, 1)
        '''
        
        # Waypoint Prediction Processing
        if self.output_coordinate_repr == 'euclidean':
            # Predict deltas and compute cumulative sum
            wp_pred = torch.cumsum(wp_pred, dim=1)
            return wp_pred, feature_pred[:, :-1], future_obs_enc[:, :-1]
        else:
            raise NotImplementedError(f"Output coordinate representation {self.output_coordinate_repr} not implemented")
