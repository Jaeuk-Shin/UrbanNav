"""Auxiliary prediction heads for probing LSTM hidden-state representations.

These heads attach to the LSTM backbone of PPOAgent and predict environmental
geometry from the hidden state.  They serve two purposes:

1. **Diagnostic probes** — measure whether LSTM representations capture spatial
   structure.  Set ``detach=True`` to stop gradients flowing to the backbone.
2. **Auxiliary training signals** — improve representations by adding supervised
   geometry-prediction objectives (``detach=False``).

Three approaches are implemented:

OccupancyHead
    Predicts an ego-centric binary occupancy grid from the LSTM hidden state.

    * Jaderberg et al. (2016) "Reinforcement Learning with Unsupervised
      Auxiliary Tasks" (UNREAL)
    * Chaplot et al. (2020) "Learning to Explore using Active Neural SLAM"

ObstaclePositionHead
    Predicts relative positions of K nearest obstacles/pedestrians in the
    agent's ego-centric frame (forward, right).

    * Ye et al. (2020) "Auxiliary Tasks Speed Up Learning Reach-Avoid
      Policies for Autonomous Aerial Vehicles"

GeodesicDistanceHead
    Predicts obstacle-aware geodesic distance to goal (scalar).

    * Wijmans et al. (2020) "DD-PPO: Learning Near-Perfect PointGoal
      Navigators from 2.5 Billion Frames"
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Individual Heads ─────────────────────────────────────────────────


class OccupancyHead(nn.Module):
    """Predict ego-centric binary occupancy grid from LSTM hidden state.

    Architecture: hidden → MLP → (G, G) with sigmoid.

    Target: binary grid where 1 = obstacle/non-walkable, 0 = free.
    Loss: binary cross-entropy.
    """

    def __init__(self, hidden_dim: int, grid_size: int = 16):
        super().__init__()
        self.grid_size = grid_size
        out_dim = grid_size * grid_size
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        # Small init on last layer for stable early training
        nn.init.orthogonal_(self.net[-1].weight, gain=0.1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden : (B, hidden_dim)

        Returns
        -------
        logits : (B, G, G) — raw logits (apply sigmoid for probabilities)
        """
        return self.net(hidden).view(-1, self.grid_size, self.grid_size)

    @staticmethod
    def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy loss.

        Parameters
        ----------
        logits : (B, G, G) raw logits
        targets : (B, G, G) binary occupancy (0=free, 1=occupied)
        """
        return F.binary_cross_entropy_with_logits(logits, targets)


class ObstaclePositionHead(nn.Module):
    """Predict relative positions of K nearest objects from LSTM hidden state.

    Each slot predicts (forward, right) displacement in ego-centric frame.
    Invalid slots (fewer than K objects nearby) are masked out of the loss.

    Architecture: hidden → MLP → (K, 2).
    Loss: MSE on valid slots.
    """

    def __init__(self, hidden_dim: int, max_objects: int = 8):
        super().__init__()
        self.max_objects = max_objects
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, max_objects * 2),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden : (B, hidden_dim)

        Returns
        -------
        positions : (B, K, 2) predicted (forward, right) in ego frame
        """
        return self.net(hidden).view(-1, self.max_objects, 2)

    @staticmethod
    def compute_loss(pred: torch.Tensor, targets: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
        """Masked MSE loss over valid object slots.

        Parameters
        ----------
        pred    : (B, K, 2) predicted positions
        targets : (B, K, 2) ground-truth positions
        mask    : (B, K)    1.0 for valid slots, 0.0 for padding
        """
        # (B, K, 2) → per-slot squared error
        sq_err = (pred - targets).pow(2).sum(dim=-1)  # (B, K)
        # Mask and average over valid slots
        n_valid = mask.sum().clamp(min=1.0)
        return (sq_err * mask).sum() / n_valid


class GeodesicDistanceHead(nn.Module):
    """Predict geodesic distance to goal from LSTM hidden state.

    Architecture: hidden → MLP → scalar (non-negative via softplus).
    Loss: smooth L1 (Huber).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden : (B, hidden_dim)

        Returns
        -------
        dist : (B,) predicted geodesic distance (non-negative)
        """
        return F.softplus(self.net(hidden).squeeze(-1))

    @staticmethod
    def compute_loss(pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Smooth L1 loss (robust to large-distance outliers).

        Parameters
        ----------
        pred    : (B,) predicted distances
        targets : (B,) ground-truth geodesic distances
        """
        return F.smooth_l1_loss(pred, targets)


# ─── Head Group ───────────────────────────────────────────────────────


class AuxiliaryHeadGroup(nn.Module):
    """Manages one or more auxiliary prediction heads.

    Parameters
    ----------
    hidden_dim : int
        LSTM hidden dimension.
    heads : list of str
        Which heads to enable.  Options: ``"occupancy"``, ``"obstacle_pos"``,
        ``"geodesic_dist"``.
    detach : bool
        If True, detach LSTM hidden states before feeding to heads (probe
        mode — no gradient to backbone).
    grid_size : int
        Occupancy grid resolution (default 16 → 16x16).
    max_objects : int
        Max object slots for obstacle position head (default 8).
    """

    def __init__(self, hidden_dim: int, heads: list[str],
                 detach: bool = False,
                 grid_size: int = 16,
                 max_objects: int = 8):
        super().__init__()
        self.detach = detach
        self.head_names = sorted(heads)

        self.heads = nn.ModuleDict()
        if "occupancy" in heads:
            self.heads["occupancy"] = OccupancyHead(hidden_dim, grid_size)
        if "obstacle_pos" in heads:
            self.heads["obstacle_pos"] = ObstaclePositionHead(hidden_dim, max_objects)
        if "geodesic_dist" in heads:
            self.heads["geodesic_dist"] = GeodesicDistanceHead(hidden_dim)

        self.grid_size = grid_size
        self.max_objects = max_objects

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run all enabled heads on the LSTM hidden state.

        Parameters
        ----------
        hidden : (B, hidden_dim) LSTM output

        Returns
        -------
        preds : dict mapping head name → prediction tensor
        """
        if self.detach:
            hidden = hidden.detach()

        preds = {}
        for name, head in self.heads.items():
            preds[name] = head(hidden)
        return preds

    def compute_losses(self, preds: dict, targets: dict) -> dict[str, torch.Tensor]:
        """Compute per-head losses.

        Parameters
        ----------
        preds : dict from ``forward()``
        targets : dict with matching keys, containing ground-truth tensors:
            - ``"occupancy"``: (B, G, G) binary
            - ``"obstacle_pos"``: (B, K, 2) positions
            - ``"obstacle_mask"``: (B, K) validity mask
            - ``"geodesic_dist"``: (B,) distances

        Returns
        -------
        losses : dict mapping head name → scalar loss
        """
        losses = {}
        if "occupancy" in preds and "occupancy" in targets:
            losses["occupancy"] = OccupancyHead.compute_loss(
                preds["occupancy"], targets["occupancy"])
        if "obstacle_pos" in preds and "obstacle_pos" in targets:
            losses["obstacle_pos"] = ObstaclePositionHead.compute_loss(
                preds["obstacle_pos"], targets["obstacle_pos"],
                targets["obstacle_mask"])
        if "geodesic_dist" in preds and "geodesic_dist" in targets:
            losses["geodesic_dist"] = GeodesicDistanceHead.compute_loss(
                preds["geodesic_dist"], targets["geodesic_dist"])
        return losses

    def total_loss(self, preds: dict, targets: dict) -> torch.Tensor:
        """Sum of all head losses (for adding to PPO objective)."""
        losses = self.compute_losses(preds, targets)
        if not losses:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return sum(losses.values())
