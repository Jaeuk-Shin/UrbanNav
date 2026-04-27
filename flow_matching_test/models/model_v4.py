"""
FiLM-conditioned MLP velocity field with a CNN condition encoder for v4.

Extends ``FlowMatchingMLP`` from ``model.py``.  The only change is that the
flat condition vector (a concatenated depth image + road mask) is first
reshaped into a 2-channel image and compressed by a small CNN before
entering the existing ``cond_proj`` pathway:

    camera image (B, 2, H, W) ──► CNN encoder ──► feat (B, enc_dim)
                                                        │
                                                    cond_proj ──┐
                                                                ├──► cond ──► [FiLMResBlock × N] ──► v
                          sinusoidal(t * 1000) ──► time_mlp ──┘              ▲
                                                                              │
                          sample (B, T·D)      ──► in_proj ──────► hidden ───┘
"""

import torch
import torch.nn as nn

from model import FlowMatchingMLP


class FlowMatchingMLPv4(FlowMatchingMLP):
    """
    FlowMatchingMLP with a CNN front-end for image conditions.

    Parameters
    ----------
    input_dim      : int   - flattened trajectory dimension (T * D)
    image_h        : int   - camera image height  (default 24)
    image_w        : int   - camera image width   (default 32)
    n_channels     : int   - number of image channels (default 2: depth + road mask)
    hidden_dim     : int   - width of the residual blocks
    n_blocks       : int   - number of FiLM residual blocks
    time_embed_dim : int   - sinusoidal embedding size for the timestep
    encoder_dims   : tuple - output channels per CNN stage
    """

    def __init__(self, input_dim, image_h=24, image_w=32, n_channels=2,
                 hidden_dim=256, n_blocks=4, time_embed_dim=64,
                 encoder_dims=(32, 64, 128)):
        enc_out_dim = encoder_dims[-1]

        # Parent's cond_proj will map enc_out_dim → hidden_dim.
        super().__init__(
            input_dim=input_dim,
            cond_dim=enc_out_dim,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            time_embed_dim=time_embed_dim,
        )

        self.image_h = image_h
        self.image_w = image_w
        self.n_channels = n_channels

        # Build lightweight CNN encoder.
        layers = []
        in_ch = n_channels
        for out_ch in encoder_dims:
            layers.append(nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1))
            layers.append(nn.Mish())
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        self.cond_encoder = nn.Sequential(*layers)

    def forward(self, sample, timestep, condition):
        """
        Parameters
        ----------
        sample    : (B, T, D)  or  (B, T*D)
        timestep  : (B,)  float in [0, 1]
        condition : (B, n_channels * image_h * image_w)  — flat camera output

        Returns
        -------
        v : same shape as *sample*
        """
        B = sample.shape[0]
        img = condition.view(B, self.n_channels, self.image_h, self.image_w)
        cond_feat = self.cond_encoder(img)
        return super().forward(sample, timestep, cond_feat)
