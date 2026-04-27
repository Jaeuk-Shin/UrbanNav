"""
Rich multi-modal obstacle-bypass dataset (v2).

Environment geometry is generated entirely by a cellular-automaton cave
algorithm with tunable fill probability and iteration count, producing
realistic, organic occupancy maps.  No explicit obstacle primitives are
placed — structure emerges naturally from the automaton.

Features:
  1. **Occupancy maps** via cellular automaton with varied density.
  2. **Semantic sensor** — 1-D binary road channel (1 = road, 0 = non-road).
     Roads are not sensed by the depth lidar.
  3. **Road generation** — procedural polyline roads on a separate layer.
  4. **A*-based trajectories** — collision-free paths planned on the
     inflated occupancy + road cost map, then smoothed and subsampled.
  5. **Offline save / load** — ``.npz`` archives with subset loading.

Condition vector layout:  [depth_rays (n_rays) | semantic_rays (n_rays)]
"""

from __future__ import annotations
import heapq
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# =========================================================================== #
#  Constants
# =========================================================================== #

CELL_SIZE     = 0.1     # metres per grid cell
CLEAR_RADIUS  = 0.8     # guaranteed free zone around origin (metres)
ROAD_WIDTH    = 0.35    # half-width of a generated road (metres)
SAFETY_MARGIN = 0.10    # trajectory must be this far from any occupied cell

# =========================================================================== #
#  Occupancy-map generation  (cellular-automaton caves)
# =========================================================================== #

def _cave_automaton(grid: np.ndarray, steps: int = 4,
                    birth: int = 5, survive: int = 4) -> np.ndarray:
    """Run a Moore-neighbourhood cellular automaton on a binary grid."""
    h, w = grid.shape
    for _ in range(steps):
        nbrs = np.zeros_like(grid, dtype=int)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                shifted = np.roll(np.roll(grid, di, axis=0), dj, axis=1)
                nbrs += shifted
        new = np.zeros_like(grid)
        new[(grid == 1) & (nbrs >= survive)] = 1
        new[(grid == 0) & (nbrs >= birth)]   = 1
        new[0, :] = new[-1, :] = new[:, 0] = new[:, -1] = 1
        grid = new
    return grid


def generate_occupancy_map(
    x_range: Tuple[float, float] = (-3.5, 3.5),
    y_range: Tuple[float, float] = (-1.0, 7.5),
    cell_size: float = CELL_SIZE,
    cave_prob: float = 0.48,
    cave_steps: int = 4,
    rng: np.random.RandomState = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a binary occupancy map via cellular automaton.

    Returns  occ (H,W) uint8,  xs (W,),  ys (H,).
    """
    if rng is None:
        rng = np.random.RandomState()

    nx = int(round((x_range[1] - x_range[0]) / cell_size))
    ny = int(round((y_range[1] - y_range[0]) / cell_size))
    xs = np.linspace(x_range[0] + cell_size / 2, x_range[1] - cell_size / 2, nx)
    ys = np.linspace(y_range[0] + cell_size / 2, y_range[1] - cell_size / 2, ny)

    grid = (rng.rand(ny, nx) < cave_prob).astype(np.uint8)
    grid = _cave_automaton(grid, steps=cave_steps, birth=5, survive=4)

    # clear the area around origin
    iy, ix = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    dist = np.sqrt(xs[ix] ** 2 + ys[iy] ** 2)
    grid[dist < CLEAR_RADIUS] = 0
    # ensure a cone ahead of origin is free
    ahead_mask = (ys[iy] > 0) & (np.abs(xs[ix]) < 0.5) & (ys[iy] < 1.5)
    grid[ahead_mask] = 0

    return grid, xs, ys


# =========================================================================== #
#  Road generation
# =========================================================================== #

def _stamp_road(road_map, xs, ys, path_xy, half_width=ROAD_WIDTH):
    """Paint road cells along a polyline path."""
    for px, py in path_xy:
        mask = (np.abs(xs[None, :] - px) <= half_width) & \
               (np.abs(ys[:, None] - py) <= half_width)
        road_map[mask] = 1


def generate_road(
    occ: np.ndarray, xs: np.ndarray, ys: np.ndarray,
    rng: np.random.RandomState,
    spawn_prob: float = 0.5,
) -> np.ndarray:
    """
    Occasionally generate a meandering road (separate layer, not an obstacle).

    Returns road_map (H, W) uint8, 1 = road cell.
    """
    road_map = np.zeros_like(occ)
    if rng.rand() > spawn_prob:
        return road_map

    n_roads = rng.choice([1, 2], p=[0.6, 0.4])
    for _ in range(n_roads):
        start_x = rng.uniform(xs[0] + 0.5, xs[-1] - 0.5)
        end_x   = rng.uniform(xs[0] + 0.5, xs[-1] - 0.5)
        n_kp = rng.randint(4, 8)
        kp_ys = np.sort(rng.uniform(ys[0] + 0.3, ys[-1] - 0.3, size=n_kp))
        kp_xs = np.linspace(start_x, end_x, n_kp) + rng.randn(n_kp) * 0.3
        kp_xs = np.clip(kp_xs, xs[0] + 0.3, xs[-1] - 0.3)

        fine_ys = np.linspace(kp_ys[0], kp_ys[-1], 200)
        fine_xs = np.interp(fine_ys, kp_ys, kp_xs)
        path_xy = np.column_stack([fine_xs, fine_ys])
        half_w = rng.uniform(0.2, 0.5)
        _stamp_road(road_map, xs, ys, path_xy, half_width=half_w)

    return road_map


# =========================================================================== #
#  Raycasting on occupancy maps
# =========================================================================== #

def raycast_occ(
    occ: np.ndarray, xs: np.ndarray, ys: np.ndarray,
    n_rays: int = 120, fov_deg: float = 90.0,
    max_dist: float = 10.0,
) -> np.ndarray:
    """Cast rays from origin against the occupancy grid.  Returns normalised distances."""
    angles = np.linspace(-np.radians(fov_deg / 2),
                         np.radians(fov_deg / 2), n_rays)
    step = CELL_SIZE * 0.5
    n_steps = int(max_dist / step)

    distances = np.full(n_rays, max_dist, dtype=np.float64)
    nx, ny = len(xs), len(ys)
    x0, y0 = xs[0], ys[0]
    dx_grid = xs[1] - xs[0] if nx > 1 else CELL_SIZE
    dy_grid = ys[1] - ys[0] if ny > 1 else CELL_SIZE

    for i, angle in enumerate(angles):
        dx, dy = np.sin(angle), np.cos(angle)
        for s in range(1, n_steps + 1):
            px, py = dx * s * step, dy * s * step
            ci = int((px - x0) / dx_grid)
            ri = int((py - y0) / dy_grid)
            if ci < 0 or ci >= nx or ri < 0 or ri >= ny:
                distances[i] = s * step
                break
            if occ[ri, ci]:
                distances[i] = s * step
                break

    return distances / max_dist


def semantic_raycast(
    road_map: np.ndarray, xs: np.ndarray, ys: np.ndarray,
    n_rays: int = 120, fov_deg: float = 90.0,
    max_dist: float = 10.0,
) -> np.ndarray:
    """1-D semantic sensor: 1 if a ray hits a road cell before max_dist, else 0."""
    angles = np.linspace(-np.radians(fov_deg / 2),
                         np.radians(fov_deg / 2), n_rays)
    step = CELL_SIZE * 0.5
    n_steps = int(max_dist / step)

    sem = np.zeros(n_rays, dtype=np.float32)
    nx, ny = len(xs), len(ys)
    x0, y0 = xs[0], ys[0]
    dx_grid = xs[1] - xs[0] if nx > 1 else CELL_SIZE
    dy_grid = ys[1] - ys[0] if ny > 1 else CELL_SIZE

    for i, angle in enumerate(angles):
        dx, dy = np.sin(angle), np.cos(angle)
        for s in range(1, n_steps + 1):
            px, py = dx * s * step, dy * s * step
            ci = int((px - x0) / dx_grid)
            ri = int((py - y0) / dy_grid)
            if ci < 0 or ci >= nx or ri < 0 or ri >= ny:
                break
            if road_map[ri, ci]:
                sem[i] = 1.0
                break
    return sem


# =========================================================================== #
#  A* path planner on the occupancy grid
# =========================================================================== #

def _build_cost_map(
    occ: np.ndarray,
    road_map: Optional[np.ndarray] = None,
    margin_cells: int = 2,
) -> np.ndarray:
    """Build a boolean *blocked* map (inflated obstacles + road cells)."""
    H, W = occ.shape
    blocked = np.zeros((H, W), dtype=bool)
    for dr in range(-margin_cells, margin_cells + 1):
        for dc in range(-margin_cells, margin_cells + 1):
            shifted = np.roll(np.roll(occ, dr, axis=0), dc, axis=1)
            blocked |= shifted.astype(bool)
    if road_map is not None:
        blocked |= road_map.astype(bool)
    return blocked


def _astar(
    blocked: np.ndarray,
    start_rc: Tuple[int, int],
    goal_rc: Tuple[int, int],
    max_expansions: int = 40000,
) -> Optional[List[Tuple[int, int]]]:
    """A* on an 8-connected grid.  Returns (row, col) path or None."""
    H, W = blocked.shape
    sr, sc = start_rc
    gr, gc = goal_rc
    if blocked[sr, sc] or blocked[gr, gc]:
        return None

    def h(r, c):
        dr, dc = abs(r - gr), abs(c - gc)
        return max(dr, dc) + 0.414 * min(dr, dc)

    SQRT2 = 1.4142
    open_set: list = [(h(sr, sc), 0.0, sr, sc)]
    g_score = {(sr, sc): 0.0}
    came_from: dict = {}
    expansions = 0

    while open_set and expansions < max_expansions:
        _, g, r, c = heapq.heappop(open_set)
        if (r, c) == (gr, gc):
            path = [(gr, gc)]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            return path[::-1]
        if g > g_score.get((r, c), float("inf")):
            continue
        expansions += 1
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= H or nc < 0 or nc >= W or blocked[nr, nc]:
                    continue
                cost = SQRT2 if (dr != 0 and dc != 0) else 1.0
                ng = g + cost
                if ng < g_score.get((nr, nc), float("inf")):
                    g_score[(nr, nc)] = ng
                    came_from[(nr, nc)] = (r, c)
                    heapq.heappush(open_set, (ng + h(nr, nc), ng, nr, nc))
    return None


# =========================================================================== #
#  Trajectory generation via A* + smoothing
# =========================================================================== #

def _smooth(x, sigma=1.0, radius=2):
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, radius, mode="edge"), k, mode="valid")


def _rc_to_xy(path_rc, xs, ys):
    return np.array([(xs[c], ys[r]) for r, c in path_rc])


def _subsample_smooth(path_xy, T, rng, noise_xy=0.04):
    """Uniformly subsample a dense path to T waypoints with smoothing + noise."""
    diffs = np.diff(path_xy, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum_len[-1]
    if total < 1e-6:
        return None
    t_vals = np.linspace(total / T, total, T)
    pts = np.empty((T, 2))
    for i, t in enumerate(t_vals):
        idx = np.searchsorted(cum_len, t, side="right") - 1
        idx = np.clip(idx, 0, len(cum_len) - 2)
        frac = (t - cum_len[idx]) / max(seg_lens[idx], 1e-8)
        pts[i] = path_xy[idx] + frac * diffs[idx]
    pts[:, 0] = _smooth(pts[:, 0], sigma=1.2, radius=2)
    pts[:, 1] = _smooth(pts[:, 1], sigma=1.2, radius=2)
    pts[:, 0] += rng.randn(T) * noise_xy
    pts[:, 1] += rng.randn(T) * noise_xy * 0.5
    return pts


def sample_goal(
    xs: np.ndarray, ys: np.ndarray,
    blocked: np.ndarray,
    min_goal_y: float = 2.0,
    L: float = 7.0,
    wall_x: float = 3.0,
    rng: np.random.RandomState = None,
    max_attempts: int = 50,
) -> Optional[Tuple[float, float, int, int]]:
    """Sample a reachable goal in free space.  Returns (x, y, row, col) or None."""
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    for _ in range(max_attempts):
        if rng.rand() < 0.3:
            gy = rng.uniform(min_goal_y, min(min_goal_y + 1.5, L))
        else:
            gy = rng.uniform(min_goal_y, L)
        gx = rng.uniform(-wall_x + 0.4, wall_x - 0.4)
        gc = int((gx - xs[0]) / dx)
        gr = int((gy - ys[0]) / dy)
        if 0 <= gr < len(ys) and 0 <= gc < len(xs) and not blocked[gr, gc]:
            return gx, gy, gr, gc
    return None


def generate_trajectory(
    xs: np.ndarray, ys: np.ndarray,
    blocked: np.ndarray,
    T: int = 8, L: float = 7.0, wall_x: float = 3.0,
    min_goal_y: float = 2.0,
    rng: np.random.RandomState = None,
) -> Optional[np.ndarray]:
    """
    Plan a collision-free trajectory via A*: sample goal, plan path, smooth.

    Returns (T, 2) array or None.
    """
    if rng is None:
        rng = np.random.RandomState()
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    sc = np.clip(int((0.0 - xs[0]) / dx), 0, len(xs) - 1)
    sr = np.clip(int((0.0 - ys[0]) / dy), 0, len(ys) - 1)

    goal = sample_goal(xs, ys, blocked, min_goal_y=min_goal_y,
                       L=L, wall_x=wall_x, rng=rng)
    if goal is None:
        return None
    _, _, gr, gc = goal

    path_rc = _astar(blocked, (sr, sc), (gr, gc))
    if path_rc is None or len(path_rc) < 3:
        return None

    path_xy = _rc_to_xy(path_rc, xs, ys)
    return _subsample_smooth(path_xy, T, rng)


# =========================================================================== #
#  Scene generation
# =========================================================================== #

def generate_scene(
    n_rays: int = 120,
    T: int = 8,
    L: float = 7.0,
    trajs_per_scene: int = 50,
    fov_deg: float = 90.0,
    max_dist: float = 10.0,
    wall_x: float = 3.0,
    rng: np.random.RandomState = None,
) -> Optional[Dict]:
    """
    Generate a single scene: cave occupancy + road + A* trajectories.

    Returns None if too few trajectories could be planned.
    """
    if rng is None:
        rng = np.random.RandomState()

    # 1. occupancy map — always cave, with varied density
    cave_prob = rng.uniform(0.38, 0.48)
    cave_steps = rng.randint(3, 5)
    occ, xs, ys = generate_occupancy_map(
        cave_prob=cave_prob, cave_steps=cave_steps, rng=rng)

    # 2. road layer (does not affect occupancy)
    road_map = generate_road(occ, xs, ys, rng=rng)

    # 3. raycasting conditions
    depth_cond = raycast_occ(occ, xs, ys, n_rays=n_rays, fov_deg=fov_deg,
                             max_dist=max_dist)
    sem_cond = semantic_raycast(road_map, xs, ys, n_rays=n_rays, fov_deg=fov_deg,
                                max_dist=max_dist)
    condition = np.concatenate([depth_cond, sem_cond]).astype(np.float32)

    # 4. build cost map (inflated occ + roads)
    cell_dx = xs[1] - xs[0] if len(xs) > 1 else CELL_SIZE
    margin_cells = max(1, int(np.ceil(SAFETY_MARGIN / cell_dx)))
    blocked = _build_cost_map(occ, road_map=road_map, margin_cells=margin_cells)

    # 5. generate trajectories via A*
    trajectories = []
    for _ in range(trajs_per_scene):
        traj = generate_trajectory(xs, ys, blocked, T=T, L=L,
                                   wall_x=wall_x, rng=rng)
        if traj is not None:
            trajectories.append(traj)

    if len(trajectories) < trajs_per_scene // 4:
        return None

    return dict(
        condition=condition,
        trajectories=np.array(trajectories, dtype=np.float32),
        occ=occ, xs=xs, ys=ys,
        road_map=road_map,
    )


# =========================================================================== #
#  Offline dataset generation & loading
# =========================================================================== #

def generate_and_save(
    path: str,
    n_scenes: int = 2000,
    trajs_per_scene: int = 50,
    n_rays: int = 120,
    T: int = 8,
    L: float = 7.0,
    seed: int = 42,
    **kwargs,
):
    """Generate scenes, flatten, and save to ``.npz``."""
    rng = np.random.RandomState(seed)
    all_conds, all_trajs = [], []
    generated = 0

    print(f"Generating {n_scenes} scenes (seed={seed}) ...")
    while generated < n_scenes:
        scene = generate_scene(
            n_rays=n_rays, T=T, L=L,
            trajs_per_scene=trajs_per_scene,
            rng=rng, **kwargs)
        if scene is None:
            continue
        n_traj = len(scene["trajectories"])
        conds = np.tile(scene["condition"], (n_traj, 1))
        all_conds.append(conds)
        all_trajs.append(scene["trajectories"])
        generated += 1
        if generated % 200 == 0:
            print(f"  {generated}/{n_scenes} scenes done")

    conditions = np.concatenate(all_conds, axis=0)
    trajectories = np.concatenate(all_trajs, axis=0)

    import json
    meta = json.dumps(dict(
        n_scenes=n_scenes, trajs_per_scene=trajs_per_scene,
        n_rays=n_rays, T=T, L=L, seed=seed,
        cond_dim=int(conditions.shape[1]),
    ))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), conditions=conditions,
                        trajectories=trajectories, meta=meta)
    print(f"Saved {len(conditions):,} samples to {path}")
    return conditions, trajectories


class OfflineDataset(Dataset):
    """Load a pre-generated ``.npz`` dataset with optional subset sampling."""

    def __init__(self, path: str, max_samples: Optional[int] = None,
                 seed: int = 0):
        data = np.load(path, allow_pickle=True)
        self.conditions   = data["conditions"]
        self.trajectories = data["trajectories"]

        import json
        self.meta = json.loads(str(data["meta"]))
        self.T = self.meta["T"]
        self.n_rays = self.meta["n_rays"]
        self.cond_dim = self.meta["cond_dim"]

        if max_samples is not None and max_samples < len(self.conditions):
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(self.conditions), max_samples, replace=False)
            self.conditions = self.conditions[idx]
            self.trajectories = self.trajectories[idx]

    def __len__(self):
        return len(self.conditions)

    def __getitem__(self, idx):
        return {
            "condition":  torch.from_numpy(self.conditions[idx]),
            "trajectory": torch.from_numpy(self.trajectories[idx]),
        }


# =========================================================================== #
#  Evaluation helpers
# =========================================================================== #

EVAL_SCENARIOS = [
    {"label": "dense cave",     "cave_prob": 0.46, "cave_steps": 4, "road": True},
    {"label": "medium cave",    "cave_prob": 0.43, "cave_steps": 4, "road": False},
    {"label": "sparse cave",    "cave_prob": 0.40, "cave_steps": 3, "road": True},
    {"label": "open + road",    "cave_prob": 0.40, "cave_steps": 3, "road": True},
]


def generate_eval_gt(
    scenarios=None, T: int = 8, n_rays: int = 120, L: float = 7.0,
    fov_deg: float = 90.0, max_dist: float = 10.0,
    n_trajs: int = 200, seed: int = 999,
) -> List[Dict]:
    """Generate ground-truth data for fixed evaluation scenarios using A*."""
    if scenarios is None:
        scenarios = EVAL_SCENARIOS
    results = []

    for si, sc in enumerate(scenarios):
        rng = np.random.RandomState(seed + si)
        occ, xs, ys = generate_occupancy_map(
            cave_prob=sc["cave_prob"], cave_steps=sc["cave_steps"], rng=rng)

        road_map = np.zeros_like(occ)
        if sc.get("road", False):
            road_map = generate_road(occ, xs, ys, rng=rng, spawn_prob=1.0)

        depth = raycast_occ(occ, xs, ys, n_rays=n_rays, fov_deg=fov_deg,
                            max_dist=max_dist)
        sem = semantic_raycast(road_map, xs, ys, n_rays=n_rays, fov_deg=fov_deg,
                               max_dist=max_dist)
        condition = np.concatenate([depth, sem]).astype(np.float32)

        cell_dx = xs[1] - xs[0] if len(xs) > 1 else CELL_SIZE
        m_cells = max(1, int(np.ceil(SAFETY_MARGIN / cell_dx)))
        blocked = _build_cost_map(occ, road_map=road_map, margin_cells=m_cells)

        trajs = []
        for _ in range(n_trajs):
            t = generate_trajectory(xs, ys, blocked, T=T, L=L, rng=rng)
            if t is not None:
                trajs.append(t)

        trajs_by_gap = [np.array(trajs)] if trajs else []

        results.append(dict(
            label=sc["label"],
            condition=condition,
            trajs_by_gap=trajs_by_gap,
            occ=occ, xs=xs, ys=ys,
            road_map=road_map,
        ))

    return results


# =========================================================================== #
#  CLI
# =========================================================================== #

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate offline dataset v2")
    parser.add_argument("--output", type=str, default="flow_matching_test/data/dataset.npz")
    parser.add_argument("--n_scenes", type=int, default=2000)
    parser.add_argument("--trajs_per_scene", type=int, default=50)
    parser.add_argument("--n_rays", type=int, default=120)
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_and_save(
        path=args.output,
        n_scenes=args.n_scenes,
        trajs_per_scene=args.trajs_per_scene,
        n_rays=args.n_rays,
        T=args.T,
        seed=args.seed,
    )
