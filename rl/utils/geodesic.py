"""Obstacle-aware geodesic distance field on a rasterized navmesh grid.

Rasterizes the navmesh walkable triangles into a 2D occupancy grid (once per
map), then for each episode computes single-source Dijkstra from the goal
with obstacle-blocked cells.

This implements the DD-PPO-style geodesic reward:

    r = d_geo(s_t, goal) - d_geo(s_{t+1}, goal)

where ``d_geo`` accounts for static navmesh geometry AND dynamically placed
obstacles (e.g., firetrucks blocking crosswalks).

Obstacle blocking
~~~~~~~~~~~~~~~~~
Two strategies are supported (selected automatically):

**Mesh puncturing** (preferred) — removes navmesh triangles that overlap
with obstacle OBBs, then re-rasterizes.  Operates in world coordinates,
giving sub-cell accuracy and consistency with the SFM pedestrian boundary
constraint.

**Pixel stamping** (legacy fallback) — fills obstacle OBB polygons directly
on the rasterized grid.  Used when raw triangles are unavailable (old cache
format).

Background
----------
Indoor navigation benchmarks like Habitat (Savva et al., 2019) compute
geodesic distance via Recast/Detour's built-in A* on the navigation polygon
graph (``habitat_sim.PathFinder.geodesic_distance``).  This works because
Habitat environments are static -- the navmesh accurately represents the
walkable area.

In our setting, CARLA's navmesh is also static but we spawn dynamic obstacles
(blocked crosswalks, barriers, narrow passages) that are NOT reflected in the
navmesh.  To handle this we:

1. Rasterize the navmesh walkable triangles into a 2D grid (once per map).
2. Puncture obstacle-overlapping triangles and re-rasterize, or stamp
   obstacle OBBs as blocked cells (once per episode).
3. Build a sparse 8-connected graph and run ``scipy`` Dijkstra from the goal
   (once per episode).
4. Query the precomputed distance field at O(1) per step.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

from rl.utils.mesh_utils import puncture_triangles

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

_SQRT2 = 1.4142135623730951


class GeodesicDistanceField:
    """Rasterized navmesh grid for obstacle-aware geodesic distance queries.

    Usage::

        geo = GeodesicDistanceField(walkable_tris_std, resolution=1.0)
        dist_field = geo.compute_distance_field(goal_std, obstacle_obbs)
        d = geo.query(dist_field, agent_pos_std)
    """

    def __init__(self, walkable_tris_std: np.ndarray, resolution: float = 1.0):
        """Rasterize navmesh walkable triangles into a binary 2D grid.

        Parameters
        ----------
        walkable_tris_std : (N, 3, 2) float32
            Navmesh triangles (all walkable area types) in standard 2D
            coordinates ``(x_std, z_std)``.
        resolution : float
            Grid cell size in metres.  1.0 m gives fast computation (~1 s
            per Dijkstra on a typical CARLA town); 0.5 m gives better accuracy
            for narrow passages at ~4x cost.
        """
        self._resolution = resolution
        # Keep raw triangles for mesh-puncturing in compute_distance_field
        self._walkable_tris_std = walkable_tris_std

        # Grid bounds with padding
        all_pts = walkable_tris_std.reshape(-1, 2)
        pad = resolution * 2
        self._x_min = float(all_pts[:, 0].min()) - pad
        self._z_min = float(all_pts[:, 1].min()) - pad
        x_max = float(all_pts[:, 0].max()) + pad
        z_max = float(all_pts[:, 1].max()) + pad

        self._W = int(np.ceil((x_max - self._x_min) / resolution)) + 1
        self._H = int(np.ceil((z_max - self._z_min) / resolution)) + 1

        # Rasterize walkable triangles (one-time per map)
        self._base_grid = self._rasterize(walkable_tris_std)
        # Cached static-obstacle grid (set by compute_distance_field)
        self._static_walkable: np.ndarray | None = None

        n_walk = int(self._base_grid.sum())
        print(f"  [Geodesic] Grid {self._H}x{self._W} @ {resolution}m, "
              f"{n_walk} walkable cells "
              f"({100. * n_walk / (self._H * self._W):.1f}%)")

    @classmethod
    def from_metadata(cls, x_min: float, z_min: float,
                      H: int, W: int, resolution: float):
        """Create a lightweight instance from grid metadata only.

        Skips triangle loading and rasterization — only provides
        ``query()`` and ``trace_path()`` on precomputed distance fields.
        """
        obj = object.__new__(cls)
        obj._x_min = float(x_min)
        obj._z_min = float(z_min)
        obj._H = int(H)
        obj._W = int(W)
        obj._resolution = float(resolution)
        obj._walkable_tris_std = None
        obj._base_grid = None
        obj._static_walkable = None
        print(f"  [Geodesic] Grid {H}x{W} @ {resolution}m (metadata-only)")
        return obj

    # ── Rasterisation ─────────────────────────────────────────────────

    def _rasterize(self, tris_std: np.ndarray) -> np.ndarray:
        """Fill walkable triangles onto a uint8 grid, return as bool."""
        grid = np.zeros((self._H, self._W), dtype=np.uint8)
        origin = np.array([self._x_min, self._z_min])

        # Triangle vertices → pixel coords (col=x, row=z)
        px = np.round(
            (tris_std - origin) / self._resolution
        ).astype(np.int32)  # (N, 3, 2)

        if _HAS_CV2:
            # cv2.fillPoly expects list of (K, 1, 2) int32 contours
            cv2.fillPoly(grid, list(px.reshape(-1, 3, 1, 2)), 1)
        else:
            # Fallback: skimage (slower, per-triangle loop)
            from skimage.draw import polygon as draw_polygon
            for i in range(len(px)):
                rr, cc = draw_polygon(
                    px[i, :, 1], px[i, :, 0], shape=(self._H, self._W))
                grid[rr, cc] = 1

        return grid.astype(bool)

    # ── Grid helpers ──────────────────────────────────────────────────

    def _to_cell(self, pos_std: np.ndarray):
        """Convert standard ``(x_std, z_std)`` to grid ``(row, col)``."""
        col = int(round((pos_std[0] - self._x_min) / self._resolution))
        row = int(round((pos_std[1] - self._z_min) / self._resolution))
        return row, col

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self._H and 0 <= c < self._W

    # ── Distance field computation ────────────────────────────────────

    def compute_distance_field(
        self,
        goal_std: np.ndarray,
        obstacle_obbs_std=None,
        obstacle_buffer: float = 0.0,
    ) -> np.ndarray:
        """Block obstacles and compute Dijkstra distance field from goal.

        Uses mesh puncturing (remove overlapping triangles, re-rasterize)
        when raw triangles are available, otherwise falls back to pixel-
        level OBB stamping.

        Parameters
        ----------
        goal_std : (2,) array
            Goal position in standard coords ``(x_std, z_std)``.
        obstacle_obbs_std : list of dicts, optional
            Each dict must have ``'corners_std'`` key with a ``(4, 2)``
            oriented bounding-box in standard coords.
        obstacle_buffer : float
            Inflate obstacle OBBs outward by this amount (metres) before
            blocking.  Default 0.0 (no inflation).

        Returns
        -------
        dist_field : (H, W) float64
            Geodesic distance from each cell to the goal, in metres.
            ``np.inf`` for unreachable cells.
        """
        if obstacle_obbs_std and self._walkable_tris_std is not None:
            # ── Mesh puncturing (preferred) ──
            corners_list = [np.asarray(o['corners_std'], dtype=np.float64)
                            for o in obstacle_obbs_std]
            remaining = puncture_triangles(
                self._walkable_tris_std, corners_list, buffer=obstacle_buffer)
            walkable = self._rasterize(remaining)
        else:
            # ── Legacy pixel stamping (fallback) ──
            grid = self._base_grid.astype(np.uint8)
            if obstacle_obbs_std:
                self._stamp_obstacles(grid, obstacle_obbs_std, obstacle_buffer)
            walkable = grid.astype(bool)

        # Cache the static-obstacle-blocked grid for dynamic updates
        self._static_walkable = walkable.copy()

        # Find goal cell; snap to nearest walkable if needed
        goal_r, goal_c = self._to_cell(goal_std)
        goal_r = int(np.clip(goal_r, 0, self._H - 1))
        goal_c = int(np.clip(goal_c, 0, self._W - 1))

        if not walkable[goal_r, goal_c]:
            goal_r, goal_c = self._snap_to_walkable(walkable, goal_r, goal_c)
            if goal_r is None:
                return np.full((self._H, self._W), np.inf)

        # Build sparse graph and run Dijkstra
        graph = self._build_graph(walkable)
        goal_idx = goal_r * self._W + goal_c
        dists = sp_dijkstra(graph, indices=goal_idx, directed=False)

        return dists.reshape(self._H, self._W)

    def compute_distance_field_dynamic(
        self,
        goal_std: np.ndarray,
        ped_positions_std: np.ndarray,
        cost_scale: float = 5.0,
        cost_decay: float = 0.5,
        temporal_discount: float = 0.85,
    ) -> np.ndarray:
        """Recompute distance field with soft pedestrian-proximity costs.

        Uses the static-obstacle grid cached by the last call to
        :meth:`compute_distance_field`.  Instead of hard-blocking cells
        (which would be overly conservative over a multi-second swept
        volume), each cell near a predicted pedestrian position receives
        an additive edge-weight penalty:

        .. math::

            c(r,c) = \\sum_{t,i} \\alpha \\, e^{-d / \\beta} \\, \\lambda^{t}

        where :math:`d` is the Euclidean distance from cell ``(r, c)``
        to pedestrian :math:`i` at time :math:`t`, :math:`\\alpha` is
        ``cost_scale``, :math:`\\beta` is ``cost_decay``, and
        :math:`\\lambda` is ``temporal_discount``.  This lets Dijkstra
        route *around* pedestrians when cheap detours exist but still
        *through* them when no alternative is available.

        Parameters
        ----------
        goal_std : (2,) array
            Goal position in standard coords.
        ped_positions_std : (T, N, 2) or (N, 2) float
            Predicted pedestrian positions in standard coords.
        cost_scale : float
            Peak cost (metres-equivalent) added at a pedestrian's centre.
        cost_decay : float
            Spatial decay length (metres) for the exponential penalty.
        temporal_discount : float
            Per-step discount factor; positions further in the future
            contribute less cost.  Set to 1.0 for uniform weighting.

        Returns
        -------
        dist_field : (H, W) float64
        """
        if self._static_walkable is None:
            raise RuntimeError(
                "Call compute_distance_field() first to build the static grid")

        walkable = self._static_walkable   # read-only; not modified

        # Compute soft cost map from pedestrian predictions
        pts = np.asarray(ped_positions_std, dtype=np.float64)
        if pts.ndim == 2:
            pts = pts[np.newaxis]            # (1, N, 2)
        cost_map = None
        if pts.shape[1] > 0:
            cost_map = self._compute_ped_cost_map(
                pts, cost_scale, cost_decay, temporal_discount)

        # Goal cell (same snapping logic as compute_distance_field)
        goal_r, goal_c = self._to_cell(goal_std)
        goal_r = int(np.clip(goal_r, 0, self._H - 1))
        goal_c = int(np.clip(goal_c, 0, self._W - 1))

        if not walkable[goal_r, goal_c]:
            goal_r, goal_c = self._snap_to_walkable(walkable, goal_r, goal_c)
            if goal_r is None:
                return np.full((self._H, self._W), np.inf)

        graph = self._build_graph(walkable, cost_map=cost_map)
        goal_idx = goal_r * self._W + goal_c
        dists = sp_dijkstra(graph, indices=goal_idx, directed=False)
        return dists.reshape(self._H, self._W)

    def _compute_ped_cost_map(
        self,
        ped_positions_std: np.ndarray,
        cost_scale: float,
        cost_decay: float,
        temporal_discount: float,
    ) -> np.ndarray:
        """Build a soft per-cell cost map from predicted pedestrian positions.

        Parameters
        ----------
        ped_positions_std : (T, N, 2) float64
            Predicted positions in standard coords across T time steps.

        Returns
        -------
        cost_map : (H, W) float64 — additive cost per cell (metres).
        """
        cost = np.zeros((self._H, self._W), dtype=np.float64)
        origin = np.array([self._x_min, self._z_min])
        res = self._resolution
        # Spatial cutoff at ~3 decay lengths (contribution < 5 %)
        cutoff = int(np.ceil(3.0 * cost_decay / res))

        T = ped_positions_std.shape[0]
        for t in range(T):
            discount = temporal_discount ** t
            if discount < 0.01:
                break

            centres = np.round(
                (ped_positions_std[t] - origin) / res
            ).astype(np.int32)              # (N, 2)  col=x, row=z

            scaled = discount * cost_scale
            for cx, cz in centres:
                r_lo = max(0, cz - cutoff)
                r_hi = min(self._H, cz + cutoff + 1)
                c_lo = max(0, cx - cutoff)
                c_hi = min(self._W, cx + cutoff + 1)
                if r_lo >= r_hi or c_lo >= c_hi:
                    continue
                rr = np.arange(r_lo, r_hi)
                cc = np.arange(c_lo, c_hi)
                dr = (rr - cz)[:, None]
                dc = (cc - cx)[None, :]
                d = np.sqrt(dr * dr + dc * dc) * res
                cost[r_lo:r_hi, c_lo:c_hi] += scaled * np.exp(-d / cost_decay)

        return cost

    def _stamp_obstacles(self, grid_u8, obstacle_obbs_std, buffer):
        """Mark cells inside obstacle OBBs (optionally inflated) as blocked."""
        origin = np.array([self._x_min, self._z_min])

        for obs in obstacle_obbs_std:
            corners = np.asarray(obs['corners_std'], dtype=np.float64)  # (4,2)

            if buffer > 0:
                centre = corners.mean(axis=0)
                dirs = corners - centre
                norms = np.linalg.norm(dirs, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-6)
                corners = corners + dirs / norms * buffer

            px = np.round(
                (corners - origin) / self._resolution
            ).astype(np.int32)

            if _HAS_CV2:
                cv2.fillConvexPoly(grid_u8, px.reshape(-1, 1, 2), 0)
            else:
                # Conservative bounding-box fallback
                r_lo = max(0, int(px[:, 1].min()))
                r_hi = min(self._H - 1, int(px[:, 1].max()))
                c_lo = max(0, int(px[:, 0].min()))
                c_hi = min(self._W - 1, int(px[:, 0].max()))
                grid_u8[r_lo:r_hi + 1, c_lo:c_hi + 1] = 0

    def _snap_to_walkable(self, walkable, r, c, max_radius=100):
        """Find nearest walkable cell via expanding Chebyshev-ring search."""
        for rad in range(1, max_radius):
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    if abs(dr) != rad and abs(dc) != rad:
                        continue  # only check the ring boundary
                    nr, nc = r + dr, c + dc
                    if self._in_bounds(nr, nc) and walkable[nr, nc]:
                        return nr, nc
        return None, None

    # ── Graph construction ────────────────────────────────────────────

    def _build_graph(self, walkable: np.ndarray,
                     cost_map: np.ndarray | None = None) -> csr_matrix:
        """Build 8-connected sparse adjacency with Euclidean edge weights.

        Parameters
        ----------
        walkable : (H, W) bool
        cost_map : (H, W) float64, optional
            Additive per-cell cost.  When provided, each edge's weight
            becomes ``base_weight + cost_map[dst_row, dst_col]``
            (node-entry cost model).
        """
        H, W = walkable.shape
        N = H * W
        idx = np.arange(N, dtype=np.int32).reshape(H, W)
        res = self._resolution

        src_parts, dst_parts, w_parts = [], [], []

        for dr, dc, w in [
            (-1, 0, res), (1, 0, res), (0, -1, res), (0, 1, res),
            (-1, -1, res * _SQRT2), (-1, 1, res * _SQRT2),
            (1, -1, res * _SQRT2), (1, 1, res * _SQRT2),
        ]:
            a_r, b_r = max(0, -dr), H - max(0, dr)
            a_c, b_c = max(0, -dc), W - max(0, dc)

            ss = (slice(a_r, b_r), slice(a_c, b_c))
            ds = (slice(a_r + dr, b_r + dr), slice(a_c + dc, b_c + dc))

            valid = walkable[ss] & walkable[ds]
            n_valid = int(valid.sum())
            if n_valid == 0:
                continue

            src_parts.append(idx[ss][valid].ravel())
            dst_parts.append(idx[ds][valid].ravel())

            weights = np.full(n_valid, w, dtype=np.float32)
            if cost_map is not None:
                weights = weights + cost_map[ds][valid].astype(np.float32)
            w_parts.append(weights)

        if not src_parts:
            return csr_matrix((N, N), dtype=np.float32)

        return csr_matrix(
            (np.concatenate(w_parts),
             (np.concatenate(src_parts), np.concatenate(dst_parts))),
            shape=(N, N),
        )

    # ── Query ─────────────────────────────────────────────────────────

    def query(self, dist_field: np.ndarray, pos_std: np.ndarray) -> float:
        """Look up geodesic distance at a standard-coords position.  O(1).

        Returns distance in metres, or ``float('inf')`` if unreachable or
        out of bounds.
        """
        r, c = self._to_cell(pos_std)
        if self._in_bounds(r, c):
            d = dist_field[r, c]
            if np.isfinite(d):
                return float(d)
        return float('inf')

    # ── Path tracing ───────────────────────────────────────────────────

    def trace_path(
        self,
        dist_field: np.ndarray,
        start_std: np.ndarray,
        max_steps: int = 5000,
    ) -> np.ndarray:
        """Trace the shortest path from *start* to the goal by gradient descent.

        Follows the steepest descent on the precomputed distance field,
        walking one grid cell at each step.

        Parameters
        ----------
        dist_field : (H, W) float64
            Precomputed distance field from :meth:`compute_distance_field`.
        start_std : (2,) array
            Start position in standard coords ``(x_std, z_std)``.
        max_steps : int
            Safety limit to avoid infinite loops on degenerate fields.

        Returns
        -------
        path_std : (L, 2) float64
            Sequence of waypoints in standard coords from *start* toward the
            goal.  Empty ``(0, 2)`` array if the start is unreachable.
        """
        r, c = self._to_cell(start_std)
        r = int(np.clip(r, 0, self._H - 1))
        c = int(np.clip(c, 0, self._W - 1))

        if not np.isfinite(dist_field[r, c]):
            return np.empty((0, 2), dtype=np.float64)

        path_rc = [(r, c)]
        visited = {(r, c)}

        for _ in range(max_steps):
            cur_d = dist_field[r, c]
            if cur_d <= 0:
                break  # reached the goal cell

            # 8-connected neighbours: pick the one with smallest distance
            best_r, best_c, best_d = r, c, cur_d
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if not self._in_bounds(nr, nc):
                    continue
                nd = dist_field[nr, nc]
                if nd < best_d:
                    best_r, best_c, best_d = nr, nc, nd

            if (best_r, best_c) == (r, c):
                break  # stuck in a local minimum (shouldn't happen)
            if (best_r, best_c) in visited:
                break  # cycle detection
            r, c = best_r, best_c
            path_rc.append((r, c))
            visited.add((r, c))

        # Convert grid (row, col) → standard coords (x_std, z_std)
        path_rc = np.array(path_rc, dtype=np.float64)
        path_std = np.empty_like(path_rc)
        path_std[:, 0] = path_rc[:, 1] * self._resolution + self._x_min  # col → x_std
        path_std[:, 1] = path_rc[:, 0] * self._resolution + self._z_min  # row → z_std
        return path_std
