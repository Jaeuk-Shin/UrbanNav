import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

from model.model_utils import PolarEmbedding, MultiLayerDecoder, PositionalEncoding
from torchvision import models
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


class FlowMatchingTrajectorySampler(nn.Module):
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

        assert self.cord_embedding_type == 'input_target'
        self.cord_embedding = PolarEmbedding(cfg)
        self.dim_cord_embedding = self.cord_embedding.out_dim * self.context_size
    
        if self.dim_cord_embedding != self.encoder_feat_dim:
            self.compress_goal_enc = nn.Linear(self.dim_cord_embedding, self.encoder_feat_dim)
        else:
            self.compress_goal_enc = nn.Identity()

        # Decoder      
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
            cond_predict_scale=True,        # enable FiLM
        )

    def forward(self, obs, cord, gt_action):
        """
        Parameters
        ----------
        obs: (B, N, 3, H, W) tensor
        cord: (B, N, 2) tensor
        gt_action: (B, T, 2) tensor (The ground truth trajectory)
        """
        B, N, _, H, W = obs.shape
        obs = obs.view(B * N, 3, H, W)
        
        # ... [Preprocessing & Encoding logic remains the same] ...
        if self.do_rgb_normalize:
            obs = (obs - self.mean) / self.std
        if self.do_resize:
            obs = TF.center_crop(obs, self.crop)
            obs = TF.resize(obs, self.resize)
        
        obs_enc = self.obs_encoder(obs)
        obs_enc = self.compress_obs_enc(obs_enc).view(B, N, -1)
        cord_enc = self.cord_embedding(cord).view(B, -1)
        cord_enc = self.compress_goal_enc(cord_enc).view(B, 1, -1)
        tokens = torch.cat([obs_enc, cord_enc], dim=1)
        
        tokens = self.positional_encoding(tokens)
        dec_out = self.sa_decoder(tokens).mean(dim=1) # (B, latent_dim)
        deltas = torch.diff(gt_action, dim=1, prepend=torch.zeros_like(gt_action[:, :1, :]))

        # t ~ U[0, 1]
        t = torch.rand((B,), device=deltas.device)
        t_reshape = t.view(B, 1, 1)
        
        # sample X_0 ~ N(0, I)
        x0 = torch.randn_like(deltas)
        x1 = deltas  # This is our target data
        
        # linear interpolation
        xt = (1 - t_reshape) * x0 + t_reshape * x1
        
        # target velocity
        ut = x1 - x0

        # 3. Padding for the predictor (if architecture requires fixed size, e.g., 12)
        padding_size = 12 - xt.size(1)
        if padding_size > 0:
            xt_pad = F.pad(input=xt, pad=(0, 0, 0, padding_size))
        else:
            xt_pad = xt

        t_scaled = (t * 1000).long()
        # 4. velocity prediction
        # real numbers in [0, 1] -> integers in [0, 1000]
        v_pred = self.wp_predictor(sample=xt_pad, timestep=t_scaled, global_cond=dec_out)
        v_pred = v_pred[:, :self.len_traj_pred]

        # 5. recovery
        # During training, the "predicted waypoint" is the integration of the velocity
        deltas_pred = xt + (1 - t_reshape) * v_pred
        wp_pred = torch.cumsum(deltas_pred, dim=1)

        return wp_pred, v_pred, ut
    
    @torch.no_grad()
    def sample(self, obs, cord, num_samples=5, num_inference_steps=10):
        """
        Parameters
        ----------
        obs: (batch size, history length, 3, H, W) tensor
        num_samples: # of trajectory samples to generate per observation
        num_inference_steps: Number of Euler integration steps
        """
        B, N, _, H, W = obs.shape
        device = obs.device
        
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
        dec_out = self.sa_decoder(tokens).mean(dim=1)
        
        # Expand context: (B * num_samples, feat_dim)
        dec_out = dec_out.repeat_interleave(num_samples, dim=0)

        # 2. Initialize with pure Gaussian noise (x0)
        # We start at t=0 (pure noise) and integrate toward t=1 (data)
        xt = torch.randn((B * num_samples, 12, 2), device=device)

        padding_size = 12 - xt.size(1)
        if padding_size > 0:
            xt = F.pad(input=xt, pad=(0, 0, 0, padding_size))
        
        # 3. ODE Integration (Euler Method)
        dt = 1.0 / num_inference_steps
        
        for i in range(num_inference_steps):
            # Calculate current time t in [0, 1]
            t_curr = i / num_inference_steps
            # Scale t for the Unet's Sinusoidal Positional Embedding (0 to 1000)
            t_tensor = torch.full((B * num_samples,), t_curr * 1000, device=device)
            
            # Predict Velocity (v_pred)
            # Flow matching models the derivative dx/dt
            v_pred = self.wp_predictor(
                sample=xt, 
                timestep=t_tensor, 
                global_cond=dec_out
            )
            
            # Euler Step: x_{t+dt} = x_t + v(x_t, t) * dt
            xt = xt + v_pred * dt

        wp_pred = xt[:, :self.len_traj_pred]
        
        # Convert deltas to absolute positions
        wp_pred = torch.cumsum(wp_pred, dim=1)
        
        # Reshape to (batch size, # of samples, prediction length, 2)
        return wp_pred.view(B, num_samples, self.len_traj_pred, 2)
    
    '''
    @torch.no_grad()
    def extract_features(self, obs, cord):
        B, N, _, H, W = obs.shape
        obs = obs.view(B * N, 3, H, W)
        
        if self.do_rgb_normalize:
            obs = (obs - self.mean) / self.std
        if self.do_resize:
            obs = TF.center_crop(obs, self.crop)
            obs = TF.resize(obs, self.resize)
        
        obs_enc = self.obs_encoder(obs)
        obs_enc = self.compress_obs_enc(obs_enc).view(B, N, -1)
        cord_enc = self.cord_embedding(cord).view(B, -1)
        cord_enc = self.compress_goal_enc(cord_enc).view(B, 1, -1)
        tokens = torch.cat([obs_enc, cord_enc], dim=1)
        return tokens

    @torch.no_grad()
    def decode(self, tokens):
        tokens = self.positional_encoding(tokens)
        return self.sa_decoder(tokens).mean(dim=1) # (B, latent_dim)

    @torch.no_grad()
    def sim_ode(self, xt, dec_out, num_inference_steps=10):
        
        B = dec_out.shape[0]
        device = dec_out.device
        dt = 1.0 / num_inference_steps
        
        for i in range(num_inference_steps):
            # current t (in [0, 1])
            t_curr = i / num_inference_steps
            # scale t 
            t_tensor = torch.full((B,), t_curr * 1000, device=device)
            
            v_pred = self.wp_predictor(
                sample=xt, 
                timestep=t_tensor, 
                global_cond=dec_out
            )
            # forward Euler
            xt = xt + v_pred * dt

        wp_pred = xt[:, :self.len_traj_pred]
        # Convert deltas to absolute positions
        wp_pred = torch.cumsum(wp_pred, dim=1)
        
        # Reshape to (batch size, future length, 2)
        return wp_pred.view(B, self.len_traj_pred, 2)
    '''