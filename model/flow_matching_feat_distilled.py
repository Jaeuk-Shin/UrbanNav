import torch
import torch.nn as nn
import torch.nn.functional as F

from model.residual_mlp import ResidualMLP


class DistilledFeatMLP(nn.Module):
    """
    Feature-based distilled MLP that operates on precomputed DINOv2 features
    instead of raw images.

    Mirrors DistilledMLP but uses FlowMatchingFeat as the frozen teacher,
    removing all image-encoder dependencies (DINOv2, RGB normalization,
    resize/crop).
    """

    def __init__(self, teacher_model):
        super().__init__()
        # Freeze the teacher model
        self.teacher = teacher_model
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        self.len_traj_pred = self.teacher.len_traj_pred
        self.encoder_feat_dim = self.teacher.encoder_feat_dim

        out_dim = 12 * 2  # zero-padded inputs for UNet
        self.wp_predictor = ResidualMLP(
            input_dim=self.encoder_feat_dim + out_dim,
            output_dim=2 * self.len_traj_pred,
            hidden_dim=1024,
            num_blocks=5,
        )

    def forward_student(self, dec_out, noise):
        z = noise.reshape(dec_out.size(0), -1)
        combined = torch.cat([dec_out, z], dim=-1)
        deltas_pred = self.wp_predictor(combined)
        return deltas_pred.view(-1, self.len_traj_pred, 2)

    def forward(self, obs_features, cord):
        """
        Parameters
        ----------
        obs_features : (B, N, feature_dim) precomputed DINOv2 CLS-token features
        cord         : (B, N, 2) scaled input positions
        """
        teacher = self.teacher
        B = obs_features.shape[0]

        obs_enc = teacher.compress_obs_enc(obs_features)  # (B, N, encoder_feat_dim)
        cord_enc = teacher.cord_embedding(cord).view(B, -1)
        cord_enc = teacher.compress_goal_enc(cord_enc).view(B, 1, -1)

        tokens = torch.cat([obs_enc, cord_enc], dim=1)
        tokens = teacher.positional_encoding(tokens)
        dec_out = teacher.sa_decoder(tokens).mean(dim=1)  # (B, encoder_feat_dim)

        noise = torch.randn((B, self.len_traj_pred, 2), device=obs_features.device)
        padding_size = 12 - noise.size(1)
        if padding_size > 0:
            noise = F.pad(input=noise, pad=(0, 0, 0, padding_size))

        deltas_pred = self.forward_student(dec_out, noise)
        wp_pred = torch.cumsum(deltas_pred, dim=1)

        return {
            "wp_pred": wp_pred,
            "deltas_pred": deltas_pred,
            "noise": noise,
            "dec_out": dec_out,
        }

    @torch.no_grad()
    def sample(self, obs_features, cord, num_samples=5):
        """
        Parameters
        ----------
        obs_features : (B, N, feature_dim) precomputed DINOv2 CLS-token features
        cord         : (B, N, 2)
        num_samples  : number of trajectory samples per observation
        """
        teacher = self.teacher
        B = obs_features.shape[0]
        device = obs_features.device

        obs_enc = teacher.compress_obs_enc(obs_features)
        cord_enc = teacher.cord_embedding(cord).view(B, -1)
        cord_enc = teacher.compress_goal_enc(cord_enc).view(B, 1, -1)

        tokens = torch.cat([obs_enc, cord_enc], dim=1)
        tokens = teacher.positional_encoding(tokens)
        dec_out = teacher.sa_decoder(tokens).mean(dim=1)

        dec_out = dec_out.repeat_interleave(num_samples, dim=0)

        noise = torch.randn((B * num_samples, self.len_traj_pred, 2), device=device)
        padding_size = 12 - noise.size(1)
        if padding_size > 0:
            noise = F.pad(input=noise, pad=(0, 0, 0, padding_size))

        deltas_pred = self.forward_student(dec_out, noise)
        wp_pred = torch.cumsum(deltas_pred, dim=1)

        return {
            "wp_pred": wp_pred.view(B, num_samples, self.len_traj_pred, 2),
            "deltas_pred": deltas_pred,
            "noise": noise,
            "dec_out": dec_out,
        }
