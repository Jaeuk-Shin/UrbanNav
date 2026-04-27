"""
Toy dataset: multi-obstacle bypass with raycasting conditions.

A row of N obstacles partitions the corridor into N+1 navigable gaps.
Each gap is a distinct trajectory mode whose endpoint naturally varies
with the gap position, yielding a rich multi-modal distribution with
diverse endpoints.  N is sampled randomly per scene (default 2-4),
giving 3-5+ modes.

Condition (input):
    Raycasting distances — rays cast from the origin in a forward-facing
    cone, reporting the nearest surface (obstacle or wall), normalised
    by MAX_DIST.

Trajectory (output):
    T waypoints in 2-D (x, y).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Evaluation scenarios (hand-crafted obstacle configurations)
# --------------------------------------------------------------------------- #
EVAL_SCENARIOS = [
    {"obs_xs": [-1.5, 0.0, 1.5],         "obs_y": 3.0,
     "label": "3-even (4 gaps)"},
    {"obs_xs": [0.5, 1.8],               "obs_y": 3.0,
     "label": "2-right (3 gaps)"},
    {"obs_xs": [-1.5, -0.3, 0.9, 2.1],   "obs_y": 3.0,
     "label": "4-obs (5 gaps)"},
    {"obs_xs": [0.0],                     "obs_y": 3.0,
     "label": "1-center (2 gaps)"},
    {"obs_xs": [-1.8, -0.3, 1.3],        "obs_y": 3.5,
     "label": "3-uneven"},
]

# --------------------------------------------------------------------------- #
# Raycasting
# --------------------------------------------------------------------------- #

def raycast_to_circle(origin, direction, center, radius, max_dist):
    """Distance along a ray to the first intersection with a circle."""
    oc = origin - center
    a = direction @ direction
    b = 2.0 * (oc @ direction)
    c = oc @ oc - radius ** 2
    disc = b * b - 4 * a * c
    if disc < 0:
        return max_dist
    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    if t1 > 0:
        return min(t1, max_dist)
    if t2 > 0:
        return min(t2, max_dist)
    return max_dist


def compute_ray_distances(obstacles_xy, obs_r, n_rays=11, fov_deg=90.0,
                          max_dist=10.0, wall_x=3.0):
    """
    Cast rays from the origin against *all* obstacles and the corridor walls.

    Parameters
    ----------
    obstacles_xy : (K, 2) array of obstacle centres
    obs_r        : float, shared obstacle radius
    """
    angles = np.linspace(-np.radians(fov_deg / 2),
                         np.radians(fov_deg / 2), n_rays)
    origin = np.zeros(2)
    distances = np.full(n_rays, max_dist, dtype=np.float64)

    for i, angle in enumerate(angles):
        dx, dy = np.sin(angle), np.cos(angle)
        direction = np.array([dx, dy])

        # All obstacles
        for obs_xy in obstacles_xy:
            d = raycast_to_circle(origin, direction, obs_xy, obs_r, max_dist)
            distances[i] = min(distances[i], d)

        # Corridor walls
        if dx > 1e-8:
            distances[i] = min(distances[i], wall_x / dx)
        elif dx < -1e-8:
            distances[i] = min(distances[i], -wall_x / dx)

    return distances

# --------------------------------------------------------------------------- #
# Obstacle placement & gap computation
# --------------------------------------------------------------------------- #

def place_obstacles(n_obstacles, obs_r, wall_x=3.0, min_gap=0.5, rng=None):
    """
    Place *n_obstacles* circles of radius *obs_r* across the corridor,
    guaranteeing at least *min_gap* clearance between every pair of
    adjacent surfaces (including walls).

    Returns sorted array of obstacle x-centres.
    """
    total_width = 2 * wall_x
    slack = total_width - n_obstacles * 2 * obs_r - (n_obstacles + 1) * min_gap
    assert slack >= -1e-6, "corridor too narrow for requested obstacles"
    slack = max(slack, 0.0)

    extras = rng.dirichlet(np.ones(n_obstacles + 1)) * slack
    obs_xs = np.empty(n_obstacles)
    cursor = -wall_x
    for i in range(n_obstacles):
        cursor += min_gap + extras[i]
        obs_xs[i] = cursor + obs_r
        cursor += 2 * obs_r
    return obs_xs                              # already sorted left→right


def compute_gaps(obs_xs, obs_r, wall_x=3.0):
    """
    Return arrays of gap *centres* and *widths* (including wall boundaries).

    For K sorted obstacles there are K+1 gaps.
    """
    sorted_xs = np.sort(obs_xs)
    centres, widths = [], []
    left = -wall_x
    for ox in sorted_xs:
        right = ox - obs_r
        centres.append((left + right) / 2)
        widths.append(right - left)
        left = ox + obs_r
    centres.append((left + wall_x) / 2)
    widths.append(wall_x - left)
    return np.asarray(centres), np.asarray(widths)


def compute_gap_probs(gap_centres, gap_widths, obs_y, beta=2.0):
    """Mode probability ∝ gap_width × exp(-β |angle_to_gap|)."""
    angles = np.arctan2(np.abs(gap_centres), obs_y)
    weights = gap_widths * np.exp(-beta * angles)
    return weights / weights.sum()

# --------------------------------------------------------------------------- #
# Trajectory generation
# --------------------------------------------------------------------------- #

def _smooth(x, sigma=1.0, radius=2):
    """Simple 1-D Gaussian smoothing (no scipy)."""
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, radius, mode="edge"), k, mode="valid")


def generate_gap_trajectory(gap_centre, gap_width, obs_y, T=8, L=6.0,
                            end_spread=0.7, rng=None):
    """
    Smooth trajectory from the origin through a specific gap to a diverse
    endpoint whose x-position depends on the gap.

    Parameters
    ----------
    gap_centre : float - x-centre of the gap at y = obs_y
    gap_width  : float - width of the gap
    obs_y      : float - y-coordinate of the obstacle row
    end_spread : float - how strongly the endpoint inherits the gap position
    """
    if rng is None:
        rng = np.random.RandomState()

    # Pass-through x: gap centre ± bounded noise
    lateral_noise = rng.uniform(-1, 1) * gap_width * 0.25
    pass_x = gap_centre + lateral_noise

    # Endpoint: partially inherits gap position → diverse endpoints
    end_x = gap_centre * end_spread + rng.randn() * 0.35

    # Five key-points → piecewise-linear → smooth
    key_ys = np.array([0.0,
                       obs_y * 0.3,
                       obs_y,
                       obs_y + (L - obs_y) * 0.4,
                       L])
    key_xs = np.array([0.0,
                       pass_x * 0.3,
                       pass_x,
                       pass_x * 0.35 + end_x * 0.65,
                       end_x])

    y = np.linspace(L / T, L, T)
    x = np.interp(y, key_ys, key_xs)
    x = _smooth(x, sigma=1.0, radius=2)
    x += rng.randn(T) * 0.05
    y += rng.randn(T) * 0.025

    return np.stack([x, y], axis=-1)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class ObstacleBypassDataset(Dataset):
    """
    Synthetic dataset of multi-modal obstacle-bypass trajectories.

    Each *scene* consists of a random number of obstacles (a single row)
    creating multiple navigable gaps.  For every scene,
    *trajs_per_scene* trajectories are drawn — each passing through a
    stochastically chosen gap (mode), with endpoint diversity arising
    naturally from the gap position.
    """

    def __init__(self, n_scenes=1000, trajs_per_scene=50,
                 T=5, n_rays=120, obs_r=0.35, L=6.0,
                 n_obs_range=(2, 4), min_gap=0.5,
                 fov_deg=90.0, max_dist=10.0, wall_x=3.0,
                 end_spread=0.7, seed=42):
        super().__init__()
        self.T, self.n_rays, self.L = T, n_rays, L
        self.obs_r, self.max_dist = obs_r, max_dist

        rng = np.random.RandomState(seed)
        conditions, trajectories, modes = [], [], []

        for _ in range(n_scenes):
            n_obs = rng.randint(n_obs_range[0], n_obs_range[1] + 1)
            obs_y = rng.uniform(2.5, 4.5)
            obs_xs = place_obstacles(n_obs, obs_r, wall_x, min_gap, rng)
            obs_xy = np.column_stack([obs_xs, np.full(n_obs, obs_y)])

            cond = compute_ray_distances(
                obs_xy, obs_r, n_rays=n_rays, fov_deg=fov_deg,
                max_dist=max_dist, wall_x=wall_x,
            ) / max_dist

            gap_c, gap_w = compute_gaps(obs_xs, obs_r, wall_x)
            gap_p = compute_gap_probs(gap_c, gap_w, obs_y)

            for _ in range(trajs_per_scene):
                gi = rng.choice(len(gap_c), p=gap_p)
                traj = generate_gap_trajectory(
                    gap_c[gi], gap_w[gi], obs_y,
                    T=T, L=L, end_spread=end_spread, rng=rng)

                conditions.append(cond)
                trajectories.append(traj)
                modes.append(gi)

        self.conditions   = np.asarray(conditions,  dtype=np.float32)
        self.trajectories = np.asarray(trajectories, dtype=np.float32)
        self.modes        = np.asarray(modes,         dtype=np.int64)

    def __len__(self):
        return len(self.conditions)

    def __getitem__(self, idx):
        return {
            "condition":  torch.from_numpy(self.conditions[idx]),
            "trajectory": torch.from_numpy(self.trajectories[idx]),
        }

# --------------------------------------------------------------------------- #
# Evaluation ground-truth generator
# --------------------------------------------------------------------------- #

def generate_eval_gt(scenarios=None, obs_r=0.35, T=8, n_rays=11, L=6.0,
                     fov_deg=90.0, max_dist=10.0, wall_x=3.0,
                     end_spread=0.7, n_per_gap=200, seed=999):
    """
    Ground-truth data for fixed evaluation scenarios.

    Returns a list of dicts containing per-gap trajectory bundles and
    metadata needed for visualisation.
    """
    if scenarios is None:
        scenarios = EVAL_SCENARIOS
    rng = np.random.RandomState(seed)
    results = []

    for sc in scenarios:
        obs_xs = np.asarray(sc["obs_xs"], dtype=np.float64)
        obs_y  = sc["obs_y"]
        obs_xy = np.column_stack([obs_xs, np.full(len(obs_xs), obs_y)])

        cond = compute_ray_distances(
            obs_xy, obs_r, n_rays=n_rays, fov_deg=fov_deg,
            max_dist=max_dist, wall_x=wall_x,
        ) / max_dist

        gap_c, gap_w = compute_gaps(obs_xs, obs_r, wall_x)
        gap_p = compute_gap_probs(gap_c, gap_w, obs_y)

        trajs_by_gap = []
        for gi in range(len(gap_c)):
            n = max(int(n_per_gap * gap_p[gi] / gap_p.max()), 40)
            ts = []
            for _ in range(n):
                t = generate_gap_trajectory(
                    gap_c[gi], gap_w[gi], obs_y,
                    T=T, L=L, end_spread=end_spread, rng=rng)
                ts.append(t)
            trajs_by_gap.append(np.asarray(ts))

        results.append({
            "label":        sc["label"],
            "obs_xs":       obs_xs,
            "obs_y":        obs_y,
            "obs_r":        obs_r,
            "condition":    cond.astype(np.float32),
            "gap_centres":  gap_c,
            "gap_widths":   gap_w,
            "gap_probs":    gap_p,
            "trajs_by_gap": trajs_by_gap,
        })

    return results
