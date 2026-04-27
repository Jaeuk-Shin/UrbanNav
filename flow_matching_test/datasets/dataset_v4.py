"""
Obstacle-bypass dataset v4.

Changes over v3:
  1. **3D obstacles** — each connected obstacle component is assigned a
     consistent random height, producing a 2.5-D environment.
  2. **Virtual camera** — replaces the 1-D depth / semantic / road-depth
     raycasting with a pinhole camera that renders a grayscale depth image
     and a binary road-visibility mask from the ego's perspective.

Trajectory generation (A* + speed-limited walk) is identical to v3 —
planning remains 2-D on the ground plane.

Condition vector layout:  [depth_image (H×W) | road_mask (H×W)]
"""

from __future__ import annotations
import heapq
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from scipy.ndimage import label as ndimage_label

# =========================================================================== #
#  Constants
# =========================================================================== #

CELL_SIZE       = 0.2       # metres per grid cell
CLEAR_RADIUS    = 1.0       # guaranteed free zone around origin (metres)
ROAD_WIDTH      = 0.35      # half-width of a generated road (metres)
SAFETY_MARGIN   = 0.0

PREFERRED_SPEED = 1.0       # m/s
DT              = 1.0       # seconds per timestep

# Camera defaults
IMAGE_W      = 32           # pixels  (horizontal resolution)
IMAGE_H      = 24           # pixels  (vertical resolution)
CAM_H_FOV    = 90.0         # degrees (horizontal field of view)
CAM_V_FOV    = 60.0         # degrees (vertical field of view)
CAM_Z        = 1.5          # metres  (camera height above ground)
HEIGHT_RANGE = (1.0, 5.0)   # min / max obstacle height (metres)

# =========================================================================== #
#  Occupancy-map generation  (cellular-automaton caves — same as v3)
# =========================================================================== #

def _cave_automaton(grid: np.ndarray, steps: int = 4,
                    birth: int = 5, survive: int = 4) -> np.ndarray:
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
    cave_prob: float = 0.45,
    cave_steps: int = 4,
    birth: int = 5,
    survive: int = 4,
    rng: np.random.RandomState = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.RandomState()

    nx = int(round((x_range[1] - x_range[0]) / cell_size))
    ny = int(round((y_range[1] - y_range[0]) / cell_size))
    xs = np.linspace(x_range[0] + cell_size / 2, x_range[1] - cell_size / 2, nx)
    ys = np.linspace(y_range[0] + cell_size / 2, y_range[1] - cell_size / 2, ny)

    grid = (rng.rand(ny, nx) < cave_prob).astype(np.uint8)
    grid = _cave_automaton(grid, steps=cave_steps, birth=birth, survive=survive)

    iy, ix = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    dist = np.sqrt(xs[ix] ** 2 + ys[iy] ** 2)
    grid[dist < CLEAR_RADIUS] = 0
    ahead = (ys[iy] > 0) & (np.abs(xs[ix]) < 0.6) & (ys[iy] < 1.8)
    grid[ahead] = 0

    return grid, xs, ys


# =========================================================================== #
#  Road generation (same as v3)
# =========================================================================== #

def _stamp_road(road_map, xs, ys, path_xy, half_width=ROAD_WIDTH):
    for px, py in path_xy:
        mask = (np.abs(xs[None, :] - px) <= half_width) & \
               (np.abs(ys[:, None] - py) <= half_width)
        road_map[mask] = 1


def generate_road(
    occ: np.ndarray, xs: np.ndarray, ys: np.ndarray,
    rng: np.random.RandomState,
    spawn_prob: float = 0.5,
) -> np.ndarray:
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
    iy, ix = np.meshgrid(np.arange(len(ys)), np.arange(len(xs)), indexing="ij")
    road_map[np.sqrt(xs[ix] ** 2 + ys[iy] ** 2) < CLEAR_RADIUS] = 0
    return road_map


# =========================================================================== #
#  Height-map generation
# =========================================================================== #

def generate_height_map(
    occ: np.ndarray,
    rng: np.random.RandomState,
    height_range: Tuple[float, float] = HEIGHT_RANGE,
) -> np.ndarray:
    """Assign a consistent random height to each connected obstacle component.

    Returns a float32 array the same shape as *occ* where free cells are 0
    and obstacle cells hold their component's height.
    """
    struct = np.ones((3, 3), dtype=int)  # 8-connectivity
    labeled, n_comp = ndimage_label(occ, structure=struct)
    height_map = np.zeros(occ.shape, dtype=np.float32)
    for cid in range(1, n_comp + 1):
        height_map[labeled == cid] = rng.uniform(*height_range)
    return height_map


# =========================================================================== #
#  Virtual camera
# =========================================================================== #

def render_camera(
    height_map: np.ndarray,
    road_map: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    image_w: int = IMAGE_W,
    image_h: int = IMAGE_H,
    h_fov_deg: float = CAM_H_FOV,
    v_fov_deg: float = CAM_V_FOV,
    cam_z: float = CAM_Z,
    max_dist: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render a depth image and road-visibility mask from the ego camera.

    The camera sits at ``(0, 0, cam_z)`` looking forward (+y).

    For each pixel the renderer marches a 3-D ray through the 2-D grid:
      * If the ray enters an obstacle cell whose height covers the ray's
        current z-coordinate → obstacle hit (depth recorded).
      * If the ray reaches the ground plane (z ≤ 0) → ground hit; the
        road mask is set if the ground cell is a road.
      * If neither surface is reached within *max_dist* → sky (depth = 1).

    Returns
    -------
    depth_image : (image_h, image_w)  float32 in [0, 1]
    road_mask   : (image_h, image_w)  float32 in {0, 1}
    """
    depth_image = np.ones((image_h, image_w), dtype=np.float32)
    road_mask   = np.zeros((image_h, image_w), dtype=np.float32)

    h_angles = np.linspace(-np.radians(h_fov_deg / 2),
                            np.radians(h_fov_deg / 2), image_w)
    v_angles = np.linspace( np.radians(v_fov_deg / 2),
                           -np.radians(v_fov_deg / 2), image_h)

    sin_h = np.sin(h_angles)   # (W,)
    cos_h = np.cos(h_angles)   # (W,)
    tan_v = np.tan(v_angles)   # (H,)

    step    = CELL_SIZE * 0.5
    n_steps = int(max_dist / step)
    nx, ny  = len(xs), len(ys)
    x0, y0  = xs[0], ys[0]
    dx_grid = xs[1] - xs[0] if nx > 1 else CELL_SIZE
    dy_grid = ys[1] - ys[0] if ny > 1 else CELL_SIZE

    for col in range(image_w):
        sh, ch = sin_h[col], cos_h[col]

        # Collect obstacle cells along this column's horizontal ray.
        # Each entry: (horizontal distance, cell height).
        hits: List[Tuple[float, float]] = []
        for s in range(1, n_steps + 1):
            t = s * step
            px, py = sh * t, ch * t
            ci = int((px - x0) / dx_grid)
            ri = int((py - y0) / dy_grid)
            if ci < 0 or ci >= nx or ri < 0 or ri >= ny:
                break
            h = height_map[ri, ci]
            if h > 0:
                hits.append((t, h))

        # Resolve each row against the obstacle list + ground plane.
        for row in range(image_h):
            tv = tan_v[row]
            # Horizontal distance at which the ray hits the ground (z=0).
            t_ground = cam_z / (-tv) if tv < -1e-9 else np.inf

            hit_found = False
            for t, h in hits:
                if t > t_ground:
                    break   # ground is closer — stop searching obstacles
                z = cam_z + t * tv
                if 0.0 <= z <= h:
                    depth_image[row, col] = t / max_dist
                    hit_found = True
                    break

            if not hit_found and t_ground <= max_dist:
                depth_image[row, col] = t_ground / max_dist
                # Check whether ground hit is on a road cell.
                gpx, gpy = sh * t_ground, ch * t_ground
                gci = int((gpx - x0) / dx_grid)
                gri = int((gpy - y0) / dy_grid)
                if 0 <= gci < nx and 0 <= gri < ny and road_map[gri, gci]:
                    road_mask[row, col] = 1.0

    return depth_image, road_mask


# =========================================================================== #
#  A* path planner (same as v3)
# =========================================================================== #

def _build_cost_map(
    occ: np.ndarray,
    road_map: Optional[np.ndarray] = None,
    margin_cells: int = 1,
) -> np.ndarray:
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
                # Prevent diagonal corner-cutting: both intermediate
                # cells must be free for a diagonal move.
                if dr != 0 and dc != 0:
                    if blocked[r + dr, c] or blocked[r, c + dc]:
                        continue
                cost = SQRT2 if (dr != 0 and dc != 0) else 1.0
                ng = g + cost
                if ng < g_score.get((nr, nc), float("inf")):
                    g_score[(nr, nc)] = ng
                    came_from[(nr, nc)] = (r, c)
                    heapq.heappush(open_set, (ng + h(nr, nc), ng, nr, nc))
    return None


# =========================================================================== #
#  Trajectory generation: A* + speed-limited walk + smoothing  (same as v3)
# =========================================================================== #

def _smooth(x, sigma=1.0, radius=2):
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, radius, mode="edge"), k, mode="valid")


def _rc_to_xy(path_rc, xs, ys):
    return np.array([(xs[c], ys[r]) for r, c in path_rc])


def _walk_path(path_xy, T, speed, dt, rng, noise_xy=0.03):
    diffs = np.diff(path_xy, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum_len[-1]
    if total < 1e-6:
        return None

    step_dist = speed * dt
    pts = np.empty((T, 2))
    for i in range(T):
        d = step_dist * (i + 1)
        if d >= total:
            pts[i] = path_xy[-1]
        else:
            idx = np.searchsorted(cum_len, d, side="right") - 1
            idx = np.clip(idx, 0, len(cum_len) - 2)
            frac = (d - cum_len[idx]) / max(seg_lens[idx], 1e-8)
            pts[i] = path_xy[idx] + frac * diffs[idx]

    pts[:, 0] = _smooth(pts[:, 0], sigma=1.0, radius=2)
    pts[:, 1] = _smooth(pts[:, 1], sigma=1.0, radius=2)
    pts[:, 0] += rng.randn(T) * noise_xy
    pts[:, 1] += rng.randn(T) * noise_xy * 0.5
    return pts


def _trajectory_collides(pts, occ, xs, ys):
    """Check whether any segment of the trajectory passes through an obstacle.

    Samples points along each segment at half-cell resolution (including
    the segment from the origin to the first waypoint) and tests each
    against the raw occupancy map.
    """
    nx, ny = len(xs), len(ys)
    x0, y0 = xs[0], ys[0]
    dx_grid = xs[1] - xs[0] if nx > 1 else CELL_SIZE
    dy_grid = ys[1] - ys[0] if ny > 1 else CELL_SIZE
    half_cell = CELL_SIZE * 0.5

    # Prepend origin so the first segment (origin → pts[0]) is checked.
    all_pts = np.concatenate([[[0.0, 0.0]], pts], axis=0)

    for i in range(len(all_pts) - 1):
        p0 = all_pts[i]
        p1 = all_pts[i + 1]
        seg_len = np.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
        n_sub = max(2, int(np.ceil(seg_len / half_cell)))
        for s in range(n_sub + 1):
            frac = s / n_sub
            px = p0[0] + frac * (p1[0] - p0[0])
            py = p0[1] + frac * (p1[1] - p0[1])
            ci = int((px - x0) / dx_grid)
            ri = int((py - y0) / dy_grid)
            if 0 <= ci < nx and 0 <= ri < ny and occ[ri, ci]:
                return True
    return False


def sample_goal(
    xs, ys, blocked,
    min_goal_y=2.0, L=7.0, wall_x=3.0,
    rng=None, max_attempts=50,
):
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
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
    xs, ys, blocked,
    T=5, L=7.0, wall_x=3.0,
    speed=PREFERRED_SPEED, dt=DT,
    min_goal_y=2.0, rng=None,
):
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
    actual_speed = speed * rng.uniform(0.8, 1.0)
    return _walk_path(path_xy, T, actual_speed, dt, rng)


# =========================================================================== #
#  Scene generation
# =========================================================================== #

def generate_scene(
    T: int = 5,
    L: float = 7.0,
    trajs_per_scene: int = 50,
    max_dist: float = 10.0,
    wall_x: float = 3.0,
    speed: float = PREFERRED_SPEED,
    dt: float = DT,
    image_w: int = IMAGE_W,
    image_h: int = IMAGE_H,
    h_fov_deg: float = CAM_H_FOV,
    v_fov_deg: float = CAM_V_FOV,
    cam_z: float = CAM_Z,
    height_range: Tuple[float, float] = HEIGHT_RANGE,
    rng: np.random.RandomState = None,
) -> Optional[Dict]:
    if rng is None:
        rng = np.random.RandomState()

    # 1. coarse occupancy map
    cave_prob = rng.uniform(0.38, 0.48)
    cave_steps = rng.randint(3, 5)
    occ, xs, ys = generate_occupancy_map(
        cave_prob=cave_prob, cave_steps=cave_steps, rng=rng)

    # 2. road layer
    road_map = generate_road(occ, xs, ys, rng=rng)

    # 3. obstacle heights
    height_map = generate_height_map(occ, rng=rng, height_range=height_range)

    # 4. virtual camera
    depth_image, road_mask = render_camera(
        height_map, road_map, xs, ys,
        image_w=image_w, image_h=image_h,
        h_fov_deg=h_fov_deg, v_fov_deg=v_fov_deg,
        cam_z=cam_z, max_dist=max_dist,
    )
    condition = np.concatenate([depth_image.ravel(),
                                road_mask.ravel()]).astype(np.float32)

    # 5. cost map  (2-D planning on the ground plane, same as v3)
    cell_dx = xs[1] - xs[0] if len(xs) > 1 else CELL_SIZE
    margin_cells = max(0, int(np.ceil(SAFETY_MARGIN / cell_dx)))
    blocked = _build_cost_map(occ, road_map=road_map, margin_cells=margin_cells)
    iy, ix = np.meshgrid(np.arange(len(ys)), np.arange(len(xs)), indexing="ij")
    origin_mask = np.sqrt(xs[ix] ** 2 + ys[iy] ** 2) < CLEAR_RADIUS * 0.8
    blocked[origin_mask] = False

    # 6. A*-planned, speed-limited trajectories
    trajectories = []
    for _ in range(trajs_per_scene):
        traj = generate_trajectory(xs, ys, blocked, T=T, L=L,
                                   wall_x=wall_x, speed=speed, dt=dt,
                                   rng=rng)
        if traj is not None and traj.shape == (T, 2):
            if not _trajectory_collides(traj, occ, xs, ys):
                trajectories.append(traj)

    if len(trajectories) < max(trajs_per_scene // 4, 1):
        return None

    return dict(
        condition=condition,
        trajectories=np.stack(trajectories).astype(np.float32),
        occ=occ, xs=xs, ys=ys,
        road_map=road_map,
        height_map=height_map,
    )


# =========================================================================== #
#  Offline dataset generation & loading
# =========================================================================== #

def generate_and_save(
    path: str,
    n_scenes: int = 2000,
    trajs_per_scene: int = 50,
    T: int = 5,
    L: float = 7.0,
    seed: int = 42,
    speed: float = PREFERRED_SPEED,
    dt: float = DT,
    image_w: int = IMAGE_W,
    image_h: int = IMAGE_H,
    h_fov_deg: float = CAM_H_FOV,
    v_fov_deg: float = CAM_V_FOV,
    cam_z: float = CAM_Z,
    height_range: Tuple[float, float] = HEIGHT_RANGE,
    **kwargs,
):
    rng = np.random.RandomState(seed)
    all_conds, all_trajs = [], []
    generated = 0

    print(f"Generating {n_scenes} scenes (seed={seed}, "
          f"speed={speed} m/s, dt={dt} s, "
          f"camera {image_w}x{image_h}) ...")
    while generated < n_scenes:
        scene = generate_scene(
            T=T, L=L,
            trajs_per_scene=trajs_per_scene,
            speed=speed, dt=dt,
            image_w=image_w, image_h=image_h,
            h_fov_deg=h_fov_deg, v_fov_deg=v_fov_deg,
            cam_z=cam_z, height_range=height_range,
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
        T=T, L=L, seed=seed,
        speed=speed, dt=dt,
        image_w=image_w, image_h=image_h,
        h_fov_deg=h_fov_deg, v_fov_deg=v_fov_deg,
        cam_z=cam_z,
        height_range=list(height_range),
        cond_dim=int(conditions.shape[1]),
    ))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), conditions=conditions,
                        trajectories=trajectories, meta=meta)
    print(f"Saved {len(conditions):,} samples to {path}")
    return conditions, trajectories


class OfflineDataset(Dataset):
    def __init__(self, path: str, max_samples: Optional[int] = None,
                 seed: int = 0):
        data = np.load(path, allow_pickle=True)
        self.conditions   = data["conditions"]
        self.trajectories = data["trajectories"]

        import json
        self.meta = json.loads(str(data["meta"]))
        self.T        = self.meta["T"]
        self.image_w  = self.meta["image_w"]
        self.image_h  = self.meta["image_h"]
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
    {"label": "dense cave",  "cave_prob": 0.46, "cave_steps": 4, "road": True},
    {"label": "medium cave", "cave_prob": 0.43, "cave_steps": 4, "road": False},
    {"label": "sparse cave", "cave_prob": 0.40, "cave_steps": 3, "road": True},
    {"label": "open + road", "cave_prob": 0.40, "cave_steps": 3, "road": True},
]


def generate_eval_gt(
    scenarios=None, T: int = 5, L: float = 7.0,
    max_dist: float = 10.0,
    image_w: int = IMAGE_W, image_h: int = IMAGE_H,
    h_fov_deg: float = CAM_H_FOV, v_fov_deg: float = CAM_V_FOV,
    cam_z: float = CAM_Z,
    height_range: Tuple[float, float] = HEIGHT_RANGE,
    speed: float = PREFERRED_SPEED, dt: float = DT,
    n_trajs: int = 200, seed: int = 999,
) -> List[Dict]:
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

        height_map = generate_height_map(occ, rng=rng, height_range=height_range)

        depth_image, road_mask = render_camera(
            height_map, road_map, xs, ys,
            image_w=image_w, image_h=image_h,
            h_fov_deg=h_fov_deg, v_fov_deg=v_fov_deg,
            cam_z=cam_z, max_dist=max_dist,
        )
        condition = np.concatenate([depth_image.ravel(),
                                    road_mask.ravel()]).astype(np.float32)

        cell_dx = xs[1] - xs[0] if len(xs) > 1 else CELL_SIZE
        m_cells = max(0, int(np.ceil(SAFETY_MARGIN / cell_dx)))
        blocked = _build_cost_map(occ, road_map=road_map, margin_cells=m_cells)
        iy, ix = np.meshgrid(np.arange(len(ys)), np.arange(len(xs)), indexing="ij")
        blocked[np.sqrt(xs[ix] ** 2 + ys[iy] ** 2) < CLEAR_RADIUS * 0.8] = False

        trajs = []
        for _ in range(n_trajs):
            t = generate_trajectory(xs, ys, blocked, T=T, L=L,
                                    speed=speed, dt=dt, rng=rng)
            if t is not None and not _trajectory_collides(t, occ, xs, ys):
                trajs.append(t)

        trajs_by_gap = [np.array(trajs)] if trajs else []

        results.append(dict(
            label=sc["label"],
            condition=condition,
            depth_image=depth_image,
            road_mask=road_mask,
            trajs_by_gap=trajs_by_gap,
            occ=occ, xs=xs, ys=ys,
            road_map=road_map,
            height_map=height_map,
        ))

    return results


# =========================================================================== #
#  CLI
# =========================================================================== #

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate offline dataset v4")
    parser.add_argument("--output",          type=str,   default="flow_matching_test/data/dataset_v4.npz")
    parser.add_argument("--n_scenes",        type=int,   default=10000)
    parser.add_argument("--trajs_per_scene", type=int,   default=1)
    parser.add_argument("--T",               type=int,   default=5)
    parser.add_argument("--speed",           type=float, default=PREFERRED_SPEED)
    parser.add_argument("--dt",              type=float, default=DT)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--image_w",         type=int,   default=IMAGE_W)
    parser.add_argument("--image_h",         type=int,   default=IMAGE_H)
    parser.add_argument("--h_fov",           type=float, default=CAM_H_FOV)
    parser.add_argument("--v_fov",           type=float, default=CAM_V_FOV)
    parser.add_argument("--cam_z",           type=float, default=CAM_Z)
    parser.add_argument("--height_min",      type=float, default=HEIGHT_RANGE[0])
    parser.add_argument("--height_max",      type=float, default=HEIGHT_RANGE[1])
    args = parser.parse_args()

    generate_and_save(
        path=args.output,
        n_scenes=args.n_scenes,
        trajs_per_scene=args.trajs_per_scene,
        T=args.T,
        speed=args.speed,
        dt=args.dt,
        seed=args.seed,
        image_w=args.image_w,
        image_h=args.image_h,
        h_fov_deg=args.h_fov,
        v_fov_deg=args.v_fov,
        cam_z=args.cam_z,
        height_range=(args.height_min, args.height_max),
    )
