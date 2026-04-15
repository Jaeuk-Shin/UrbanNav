import torch
import torch.nn as nn
import torch.nn.functional as F

from model.residual_mlp import ResidualMLP


class DistilledFeatSimpleMLP(nn.Module):
    """
    Feature-based distilled MLP for the SimpleFiLM teacher.

    Unlike DistilledFeatMLP (which pads to 12 timesteps for the UNet teacher),
    this student works directly with len_traj_pred timesteps since the
    SimpleFiLMMLP teacher has no padding requirement.
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

        out_dim = self.len_traj_pred * 2  # no padding needed for SimpleFiLMMLP
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
        dec_out = self.teacher._encode_context(obs_features, cord)

        noise = torch.randn(
            (dec_out.size(0), self.len_traj_pred, 2), device=obs_features.device
        )

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
        B = obs_features.size(0)
        device = obs_features.device

        dec_out = self.teacher._encode_context(obs_features, cord)
        dec_out = dec_out.repeat_interleave(num_samples, dim=0)

        noise = torch.randn(
            (B * num_samples, self.len_traj_pred, 2), device=device
        )

        deltas_pred = self.forward_student(dec_out, noise)
        wp_pred = torch.cumsum(deltas_pred, dim=1)

        return {
            "wp_pred": wp_pred.view(B, num_samples, self.len_traj_pred, 2),
            "deltas_pred": deltas_pred,
            "noise": noise,
            "dec_out": dec_out,
        }
