"""
FiLM-conditioned MLP velocity field for conditional flow matching.

Self-contained version of ``model/flow_matching_feat_simple.py::SimpleFiLMMLP``
with no external dependencies beyond PyTorch.  The architecture is identical:

    sinusoidal(t * 1000) ──► time_mlp ──┐
                                         ├──► cond ──► [FiLMResBlock × N] ──► v
    condition            ──► cond_proj ──┘              ▲
                                                        │
    sample (B, T·D)      ──► in_proj ─────────► hidden ─┘

The output projection is zero-initialised so the network starts as an
identity flow (zero velocity everywhere).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for scalar timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class FiLMResBlock(nn.Module):
    """Pre-norm residual block with FiLM (adaLN) conditioning."""

    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.cond_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, 2 * hidden_dim),
        )

    def forward(self, x, cond):
        scale, shift = self.cond_mlp(cond).chunk(2, dim=-1)
        h = self.norm(x) * (1 + scale) + shift
        h = F.mish(self.lin1(h))
        h = self.lin2(h)
        return x + h


class FlowMatchingMLP(nn.Module):
    """
    Compact FiLM-conditioned MLP velocity field for flow matching.

    Parameters
    ----------
    input_dim      : int   - flattened trajectory dimension (T * D)
    cond_dim       : int   - dimension of the external condition vector
    hidden_dim     : int   - width of the residual blocks
    n_blocks       : int   - number of FiLM residual blocks
    time_embed_dim : int   - sinusoidal embedding size for the timestep
    """

    def __init__(self, input_dim, cond_dim, hidden_dim=256,
                 n_blocks=4, time_embed_dim=64):
        super().__init__()
        self.input_dim = input_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.in_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            [FiLMResBlock(hidden_dim, hidden_dim) for _ in range(n_blocks)]
        )

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, sample, timestep, condition):
        """
        Parameters
        ----------
        sample    : (B, T, D)  or  (B, T*D)   – noised trajectory
        timestep  : (B,)  float in [0, 1]
        condition : (B, cond_dim)

        Returns
        -------
        v : same shape as *sample* – predicted velocity
        """
        orig_shape = sample.shape
        B = sample.shape[0]
        x = sample.reshape(B, -1) if sample.ndim == 3 else sample

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device)
        if timestep.ndim == 0:
            timestep = timestep[None]
        timestep = timestep.expand(B).to(dtype=x.dtype)

        t_emb = self.time_mlp(timestep * 1000.0)
        c_emb = self.cond_proj(condition)
        cond = t_emb + c_emb

        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, cond)
        h = self.out_norm(h)
        v = self.out_proj(h)
        return v.reshape(orig_shape)
