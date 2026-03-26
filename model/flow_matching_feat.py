import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_utils import PolarEmbedding, PositionalEncoding
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


class FlowMatchingFeat(nn.Module):
    """
    Flow matching trajectory sampler that operates on precomputed DINOv2
    features instead of raw images.

    All decoder weights (compress_obs_enc, cord_embedding, compress_goal_enc,
    positional_encoding, sa_decoder, wp_predictor) are architecturally
    identical to FlowMatchingTrajectorySampler, so checkpoints can be loaded
    with strict=False (missing keys are the encoder and RGB norm buffers).
    """

    def __init__(self, cfg):
        super().__init__()
        self.context_size = cfg.model.obs_encoder.context_size
        self.encoder_feat_dim = cfg.model.encoder_feat_dim
        self.len_traj_pred = cfg.model.decoder.len_traj_pred

        obs_feature_dim = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536,
        }
        self.num_obs_features = obs_feature_dim[cfg.model.obs_encoder.type]

        if self.num_obs_features != self.encoder_feat_dim:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.encoder_feat_dim)
        else:
            self.compress_obs_enc = nn.Identity()

        # Coordinate embedding
        assert cfg.model.cord_embedding.type == 'input_target'
        self.cord_embedding = PolarEmbedding(cfg)
        self.dim_cord_embedding = self.cord_embedding.out_dim * self.context_size

        if self.dim_cord_embedding != self.encoder_feat_dim:
            self.compress_goal_enc = nn.Linear(self.dim_cord_embedding, self.encoder_feat_dim)
        else:
            self.compress_goal_enc = nn.Identity()

        # Decoder (identical architecture to FlowMatchingTrajectorySampler)
        self.positional_encoding = PositionalEncoding(
            self.encoder_feat_dim, max_seq_len=self.context_size + 1
        )
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_feat_dim,
            nhead=cfg.model.decoder.num_heads,
            dim_feedforward=cfg.model.decoder.ff_dim_factor * self.encoder_feat_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sa_decoder = nn.TransformerEncoder(
            self.sa_layer, num_layers=cfg.model.decoder.num_layers
        )
        self.wp_predictor = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=self.encoder_feat_dim,
            down_dims=[64, 128, 256],
            cond_predict_scale=True,
        )

    def forward(self, obs_features, cord, gt_action):
        """
        Parameters
        ----------
        obs_features : (B, N, feature_dim) precomputed DINOv2 CLS-token features
        cord         : (B, N, 2) scaled input positions
        gt_action    : (B, T, 2) ground-truth waypoints
        """
        B = obs_features.shape[0]

        obs_enc = self.compress_obs_enc(obs_features)  # (B, N, encoder_feat_dim)
        cord_enc = self.cord_embedding(cord).view(B, -1)
        cord_enc = self.compress_goal_enc(cord_enc).view(B, 1, -1)
        tokens = torch.cat([obs_enc, cord_enc], dim=1)

        tokens = self.positional_encoding(tokens)
        dec_out = self.sa_decoder(tokens).mean(dim=1)  # (B, encoder_feat_dim)

        deltas = torch.diff(gt_action, dim=1,
                            prepend=torch.zeros_like(gt_action[:, :1, :]))

        t = torch.rand((B,), device=obs_features.device)
        t_reshape = t.view(B, 1, 1)

        x0 = torch.randn_like(deltas)
        x1 = deltas
        xt = (1 - t_reshape) * x0 + t_reshape * x1
        ut = x1 - x0

        padding_size = 12 - xt.size(1)
        xt_pad = F.pad(xt, (0, 0, 0, padding_size)) if padding_size > 0 else xt

        t_scaled = (t * 1000).long()
        v_pred = self.wp_predictor(sample=xt_pad, timestep=t_scaled, global_cond=dec_out)
        v_pred = v_pred[:, :self.len_traj_pred]

        deltas_pred = xt + (1 - t_reshape) * v_pred
        wp_pred = torch.cumsum(deltas_pred, dim=1)

        return wp_pred, v_pred, ut

    @torch.no_grad()
    def sample(self, obs_features, cord, num_samples=5, num_inference_steps=10):
        """
        ODE integration to generate trajectory samples.

        Parameters
        ----------
        obs_features : (B, N, feature_dim)
        cord         : (B, N, 2)
        """
        B = obs_features.shape[0]
        device = obs_features.device

        obs_enc = self.compress_obs_enc(obs_features)
        cord_enc = self.cord_embedding(cord).view(B, -1)
        cord_enc = self.compress_goal_enc(cord_enc).view(B, 1, -1)
        tokens = torch.cat([obs_enc, cord_enc], dim=1)

        tokens = self.positional_encoding(tokens)
        dec_out = self.sa_decoder(tokens).mean(dim=1)

        dec_out = dec_out.repeat_interleave(num_samples, dim=0)
        xt = torch.randn((B * num_samples, 12, 2), device=device)

        dt = 1.0 / num_inference_steps
        for i in range(num_inference_steps):
            t_curr = i / num_inference_steps
            t_tensor = torch.full((B * num_samples,), t_curr * 1000, device=device)
            v_pred = self.wp_predictor(sample=xt, timestep=t_tensor, global_cond=dec_out)
            xt = xt + v_pred * dt

        wp_pred = xt[:, :self.len_traj_pred]
        wp_pred = torch.cumsum(wp_pred, dim=1)
        return wp_pred.view(B, num_samples, self.len_traj_pred, 2)
