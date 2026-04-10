import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_utils import PolarEmbedding, PositionalEncoding
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb


class FiLMResBlock(nn.Module):
    """
    Pre-norm residual MLP block with FiLM conditioning.

    FiLM(h, cond) = h * (1 + scale(cond)) + shift(cond),
    applied right after LayerNorm — equivalent to the adaLN style used in
    DiT / diffusion-policy's ConditionalResidualBlock1D.
    """

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.cond_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, 2 * hidden_dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.cond_mlp(cond).chunk(2, dim=-1)
        h = self.norm(x) * (1 + scale) + shift
        h = self.lin1(h)
        h = F.mish(h)
        h = self.lin2(h)
        return x + h


class SimpleFiLMMLP(nn.Module):
    """
    Compact MLP velocity field for flow matching on low-dimensional data.

    Pipeline:
      sinusoidal(t * 1000) --> time MLP --+
                                          +--> cond  -->  FiLM blocks  -->  v
      global_cond           --> cond MLP -+              ^
                                                         |
      sample (B, T*D) ------> in_proj ---------- hidden --+

    The final output projection is zero-initialised so the network outputs
    zero velocity at init (a clean identity flow prior).
    """

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        hidden_dim: int = 256,
        n_blocks: int = 4,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.cond_proj = nn.Sequential(
            nn.Linear(global_cond_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.in_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            [FiLMResBlock(hidden_dim, cond_dim=hidden_dim) for _ in range(n_blocks)]
        )

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        sample      : (B, T, D) noised trajectory
        timestep    : (B,) float in [0, 1]  (or scalar, auto-broadcast)
        global_cond : (B, global_cond_dim)

        Returns
        -------
        v : (B, T, D) predicted velocity field
        """
        B, T, D = sample.shape
        x = sample.reshape(B, T * D)

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device)
        if timestep.ndim == 0:
            timestep = timestep[None]
        timestep = timestep.expand(B).to(dtype=sample.dtype)

        # Scale t ∈ [0,1] → [0,1000] so the sinusoidal frequencies (down to 1e-4)
        # are all meaningfully activated, matching the DDPM-style convention.
        t_emb = self.time_mlp(timestep * 1000.0)
        c_emb = self.cond_proj(global_cond)
        cond = t_emb + c_emb

        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, cond)
        h = self.out_norm(h)
        v = self.out_proj(h)
        return v.reshape(B, T, D)


class FlowMatchingFeatSimple(nn.Module):
    """
    Flow matching trajectory sampler operating on precomputed DINOv2 features,
    using a compact FiLM-conditioned MLP velocity field instead of the
    ConditionalUnet1D used by FlowMatchingFeat.

    Appropriate when the action dimension is small (here 2 * len_traj_pred),
    where a 1D conv backbone is overkill. The context encoder (compress_obs_enc,
    cord_embedding, compress_goal_enc, positional_encoding, sa_decoder) is
    architecturally identical to FlowMatchingFeat so pretrained context weights
    can still be loaded with strict=False.
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

        assert cfg.model.cord_embedding.type == 'input_target'
        self.cord_embedding = PolarEmbedding(cfg)
        self.dim_cord_embedding = self.cord_embedding.out_dim * self.context_size

        if self.dim_cord_embedding != self.encoder_feat_dim:
            self.compress_goal_enc = nn.Linear(self.dim_cord_embedding, self.encoder_feat_dim)
        else:
            self.compress_goal_enc = nn.Identity()

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

        simple_cfg = getattr(cfg.model, 'simple_mlp', None)
        hidden_dim = getattr(simple_cfg, 'hidden_dim', 256) if simple_cfg is not None else 256
        n_blocks = getattr(simple_cfg, 'n_blocks', 4) if simple_cfg is not None else 4
        time_embed_dim = getattr(simple_cfg, 'time_embed_dim', 64) if simple_cfg is not None else 64

        self.wp_predictor = SimpleFiLMMLP(
            input_dim=2 * self.len_traj_pred,
            global_cond_dim=self.encoder_feat_dim,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            time_embed_dim=time_embed_dim,
        )

    def _encode_context(self, obs_features: torch.Tensor, cord: torch.Tensor) -> torch.Tensor:
        B = obs_features.shape[0]
        obs_enc = self.compress_obs_enc(obs_features)
        cord_enc = self.cord_embedding(cord).view(B, -1)
        cord_enc = self.compress_goal_enc(cord_enc).view(B, 1, -1)
        tokens = torch.cat([obs_enc, cord_enc], dim=1)
        tokens = self.positional_encoding(tokens)
        return self.sa_decoder(tokens).mean(dim=1)

    def forward(self, obs_features, cord, gt_action):
        """
        Parameters
        ----------
        obs_features : (B, N, feature_dim) precomputed DINOv2 CLS-token features
        cord         : (B, N, 2) scaled input positions
        gt_action    : (B, T, 2) ground-truth waypoints
        """
        B = obs_features.shape[0]
        dec_out = self._encode_context(obs_features, cord)

        deltas = torch.diff(
            gt_action, dim=1, prepend=torch.zeros_like(gt_action[:, :1, :])
        )

        t = torch.rand((B,), device=obs_features.device)
        t_reshape = t.view(B, 1, 1)

        x0 = torch.randn_like(deltas)
        x1 = deltas
        xt = (1 - t_reshape) * x0 + t_reshape * x1
        ut = x1 - x0

        v_pred = self.wp_predictor(sample=xt, timestep=t, global_cond=dec_out)

        deltas_pred = xt + (1 - t_reshape) * v_pred
        wp_pred = torch.cumsum(deltas_pred, dim=1)

        return wp_pred, v_pred, ut

    @torch.no_grad()
    def sample(self, obs_features, cord, num_samples=5, num_inference_steps=10):
        """
        Euler ODE integration of the learned velocity field.

        Parameters
        ----------
        obs_features : (B, N, feature_dim)
        cord         : (B, N, 2)
        """
        B = obs_features.shape[0]
        device = obs_features.device

        dec_out = self._encode_context(obs_features, cord)
        dec_out = dec_out.repeat_interleave(num_samples, dim=0)

        xt = torch.randn((B * num_samples, self.len_traj_pred, 2), device=device)

        noise = xt.detach().cpu().numpy().reshape(
            B, num_samples, self.len_traj_pred, 2
        )
        info = {'noise': noise}

        dt = 1.0 / num_inference_steps
        for i in range(num_inference_steps):
            t_curr = i / num_inference_steps
            t_tensor = torch.full((B * num_samples,), t_curr, device=device)
            v_pred = self.wp_predictor(
                sample=xt, timestep=t_tensor, global_cond=dec_out
            )
            xt = xt + v_pred * dt

        wp_pred = torch.cumsum(xt, dim=1)
        return wp_pred.view(B, num_samples, self.len_traj_pred, 2), info
