"""
Noise-space variational proposal for exponential tilting.

Learns q_phi(z | ctx) = N(mu_phi(ctx), sigma_phi(ctx)^2 I) such that
the pushforward f(ctx, z), z ~ q_phi, approximates the tilted distribution
p_tilted(x | ctx) proportional to exp(-beta R(x)) p_base(x | ctx).

See rl/docs/exponential_tilted_model.md for full derivation.
"""

import torch
import torch.nn as nn


class NoiseProposal(nn.Module):
    """
    Context-dependent Gaussian proposal in the distilled model's noise space.

    Given encoder output ``dec_out`` (from the frozen DINOv2 + Transformer
    pipeline), predicts mean and log-std of a diagonal Gaussian over the
    24-dim noise vector (12 x 2, zero-padded from 5 x 2).

    Initialized near identity (mu=0, sigma=1) so the proposal starts
    exactly at the base distribution N(0, I).
    """

    def __init__(self, context_dim, noise_dim=24, hidden_dim=256):
        super().__init__()
        self.noise_dim = noise_dim

        self.mu_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, noise_dim),
        )
        self.log_std_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, noise_dim),
        )

        # Zero-init output layers so q_phi starts at N(0, I)
        nn.init.zeros_(self.mu_net[-1].weight)
        nn.init.zeros_(self.mu_net[-1].bias)
        nn.init.zeros_(self.log_std_net[-1].weight)
        nn.init.zeros_(self.log_std_net[-1].bias)

    def forward(self, dec_out, num_samples=1):
        """
        Parameters
        ----------
        dec_out : (B, context_dim)
            Encoder output from the frozen teacher pipeline.
        num_samples : int
            Number of noise samples per context.

        Returns
        -------
        z : (B, num_samples, noise_dim)
            Reparameterised noise samples.
        log_q : (B, num_samples)
            Log-probability of each sample under q_phi.
        kl : (B,)
            Analytic KL(q_phi || N(0, I)).
        """
        B = dec_out.shape[0]
        device = dec_out.device

        mu = self.mu_net(dec_out)           # (B, noise_dim)
        log_std = self.log_std_net(dec_out)  # (B, noise_dim)
        std = log_std.exp()

        # Reparameterised sampling
        eps = torch.randn(B, num_samples, self.noise_dim, device=device)
        z = mu.unsqueeze(1) + std.unsqueeze(1) * eps  # (B, K, noise_dim)

        # log q_phi(z | ctx) for REINFORCE
        dist = torch.distributions.Normal(mu.unsqueeze(1), std.unsqueeze(1))
        log_q = dist.log_prob(z).sum(dim=-1)  # (B, K)

        # Analytic KL(N(mu, sigma^2 I) || N(0, I))
        kl = 0.5 * (mu.pow(2) + std.pow(2) - 2 * log_std - 1).sum(dim=-1)  # (B,)

        return z, log_q, kl

    def sample_for_inference(self, dec_out, num_samples=1):
        """Sample noise without computing log_q / kl (faster at test time)."""
        mu = self.mu_net(dec_out)
        std = self.log_std_net(dec_out).exp()
        eps = torch.randn(dec_out.shape[0], num_samples, self.noise_dim,
                          device=dec_out.device)
        return mu.unsqueeze(1) + std.unsqueeze(1) * eps
