"""Time-space geodesic distance field with predicted pedestrian obstacles.

Computes the optimal cost-to-go ``V(x, t)`` from any position *x* at time
*t* to the goal, accounting for static obstacles (hard-blocked) and
time-varying pedestrian positions (hard-blocked per time step).

Algorithm
---------
**Backward dynamic programming** on a 3-D grid ``(row, col, t)``:

1. Terminal layer ``t = T``:  ``V(x, T) = d_static(x, goal)`` — the
   pre-computed static geodesic distance acts as the terminal cost,
   smoothly connecting the time-space field to the static field beyond
   the prediction horizon.

2. Backward sweep ``t = T-1, ..., 0``:  for every walkable cell *not*
   occupied by a pedestrian at time *t*:

   .. math::

       V(x, t) = \\min_{u \\in \\mathcal{N}(x) \\cup \\{x\\}}
                  \\bigl[ w(x \\to u) + V(u, t{+}1) \\bigr]

   where :math:`\\mathcal{N}(x)` are the 8-connected spatial neighbours
   and :math:`w` is the Euclidean edge weight (``res`` for cardinal,
   ``res * sqrt(2)`` for diagonal, ``wait_cost`` for staying in place).

3. Query:  ``V(agent_cell, t)`` is an O(1) array look-up.

Because time provides a topological ordering (edges only go from *t* to
*t + 1*), no priority queue is needed — a plain backward sweep gives the
exact optimal value.

Complexity
----------
``O(H * W * T * 9)`` — with a 100 × 100 quadrant grid and T = 25 time
steps this is ~2.25 M operations, completing in ~10–50 ms in NumPy.

Usage
-----
::

    from rl.utils.geodesic_dynamic import DynamicGeodesicField

    dgeo = DynamicGeodesicField(static_geo, wait_cost=0.05)
    value_field = dgeo.compute(
        static_dist_field, ped_positions_std, ped_radius=0.6)
    d = dgeo.query(value_field, agent_pos_std, t=0)
"""

import numpy as np

_SQRT2 = 1.4142135623730951


class DynamicGeodesicField:
    """Time-space backward-DP geodesic field.

    Wraps an existing :class:`GeodesicDistanceField` and adds a time
    dimension with per-step pedestrian blocking.

    Parameters
    ----------
    static_geo : GeodesicDistanceField
        Pre-built static geodesic grid (provides grid geometry, walkable
        mask, and coordinate helpers).
    wait_cost : float
        Cost (metres-equivalent) charged for staying in place for one
        time step.  A small positive value (e.g. 0.05) breaks ties in
        favour of moving and prevents the agent from idling indefinitely.
    """

    def __init__(self, static_geo, wait_cost: float = 0.05):
        self._geo = static_geo
        self._wait_cost = wait_cost

    # ── Public API ────────────────────────────────────────────────────

    def compute(
        self,
        static_dist_field: np.ndarray,
        ped_positions_std: np.ndarray,
        ped_radius: float = 0.6,
    ) -> np.ndarray:
        """Compute the 3-D value field ``V[t, row, col]``.

        Parameters
        ----------
        static_dist_field : (H, W) float64
            Pre-computed static geodesic distance field (from
            ``GeodesicDistanceField.compute_distance_field``).  Used as
            the terminal cost at ``t = T``.
        ped_positions_std : (T, N, 2) float64
            Predicted pedestrian positions in standard coords for T time
            steps.
        ped_radius : float
            Radius (metres) around each pedestrian that is blocked.

        Returns
        -------
        V : (T + 1, H, W) float64
            Value field.  ``V[t, r, c]`` is the optimal cost-to-go from
            cell ``(r, c)`` at time ``t`` to the goal.  ``V[T]`` equals
            the static distance field.  ``np.inf`` for unreachable cells.
        """
        geo = self._geo
        H, W = geo._H, geo._W
        res = geo._resolution

        pts = np.asarray(ped_positions_std, dtype=np.float64)
        if pts.ndim == 2:
            pts = pts[np.newaxis]
        T, N_ped, _ = pts.shape

        # Base walkable mask (static obstacles already blocked)
        if geo._static_walkable is not None:
            base_walkable = geo._static_walkable
        else:
            base_walkable = geo._base_grid

        # ── Build per-step pedestrian occupancy masks ─────────────
        # ped_blocked[t] is True where a pedestrian blocks at time t
        ped_blocked = np.zeros((T, H, W), dtype=bool)
        if N_ped > 0:
            r_cells = int(np.ceil(ped_radius / res))
            r_sq = (ped_radius / res) ** 2
            origin = np.array([geo._x_min, geo._z_min])

            for t in range(T):
                centres = np.round(
                    (pts[t] - origin) / res
                ).astype(np.int32)           # (N_ped, 2): col, row

                for cx, cz in centres:
                    r_lo = max(0, cz - r_cells)
                    r_hi = min(H, cz + r_cells + 1)
                    c_lo = max(0, cx - r_cells)
                    c_hi = min(W, cx + r_cells + 1)
                    if r_lo >= r_hi or c_lo >= c_hi:
                        continue
                    rr = np.arange(r_lo, r_hi)
                    cc = np.arange(c_lo, c_hi)
                    dr = (rr - cz)[:, None]
                    dc = (cc - cx)[None, :]
                    mask = (dr * dr + dc * dc) <= r_sq
                    ped_blocked[t, r_lo:r_hi, c_lo:c_hi] |= mask

        # ── Allocate value field ──────────────────────────────────
        V = np.full((T + 1, H, W), np.inf, dtype=np.float64)

        # Terminal layer: static geodesic distance
        V[T] = static_dist_field.copy()

        # ── Neighbour offsets (8-connected + wait) ────────────────
        #   (dr, dc, cost)
        neighbours = [
            (-1,  0, res), ( 1,  0, res),
            ( 0, -1, res), ( 0,  1, res),
            (-1, -1, res * _SQRT2), (-1,  1, res * _SQRT2),
            ( 1, -1, res * _SQRT2), ( 1,  1, res * _SQRT2),
            ( 0,  0, self._wait_cost),          # wait in place
        ]

        # ── Backward sweep ────────────────────────────────────────
        for t in range(T - 1, -1, -1):
            # Cells walkable at time t: base walkable AND not ped-blocked
            free = base_walkable & ~ped_blocked[t]

            V_next = V[t + 1]           # (H, W) — already computed
            V_t = np.full((H, W), np.inf, dtype=np.float64)

            for dr, dc, w in neighbours:
                # Source slice (cells at time t that are free)
                a_r = max(0, -dr)
                b_r = H - max(0, dr)
                a_c = max(0, -dc)
                b_c = W - max(0, dc)
                ss = (slice(a_r, b_r), slice(a_c, b_c))

                # Destination slice (cells at time t+1)
                ds = (slice(a_r + dr, b_r + dr), slice(a_c + dc, b_c + dc))

                # Both source (at t) and destination (at t+1) must be
                # walkable.  For wait (dr=dc=0), ss == ds so the
                # destination walkability check at t+1 is implicit in
                # V_next being inf for blocked cells.
                valid = free[ss]

                candidate = w + V_next[ds]
                # np.minimum is element-wise; only update valid cells
                cur = V_t[ss]
                improved = valid & (candidate < cur)
                cur[improved] = candidate[improved]
                V_t[ss] = cur

            V[t] = V_t

        return V

    def query(self, V: np.ndarray, pos_std: np.ndarray,
              t: int = 0) -> float:
        """Look up ``V[t, row, col]`` for a standard-coords position.

        Returns ``float('inf')`` if unreachable or out of bounds.
        """
        geo = self._geo
        r, c = geo._to_cell(pos_std)
        T_max = V.shape[0] - 1
        t = min(t, T_max)
        if geo._in_bounds(r, c):
            d = V[t, r, c]
            if np.isfinite(d):
                return float(d)
        return float('inf')

    def trace_path(
        self,
        V: np.ndarray,
        start_std: np.ndarray,
        static_dist_field: np.ndarray,
        t_start: int = 0,
        max_steps: int = 5000,
    ) -> np.ndarray:
        """Trace the optimal path through the 3-D value field.

        Follows steepest descent on ``V[t, r, c]``, advancing time by
        one step for every spatial or wait move.  Once ``t`` exceeds the
        prediction horizon, falls back to gradient descent on the static
        2-D distance field.

        Parameters
        ----------
        V : (T+1, H, W) float64
            Value field from :meth:`compute`.
        start_std : (2,) array
            Start position in standard coords.
        static_dist_field : (H, W) float64
            Static geodesic distance field (fallback beyond horizon).
        t_start : int
            Starting time index.

        Returns
        -------
        path_std : (L, 2) float64
            Waypoints in standard coords.  Empty ``(0, 2)`` if
            unreachable.
        """
        geo = self._geo
        H, W = geo._H, geo._W
        res = geo._resolution
        T_max = V.shape[0] - 1   # = T

        r, c = geo._to_cell(start_std)
        r = int(np.clip(r, 0, H - 1))
        c = int(np.clip(c, 0, W - 1))
        t = min(t_start, T_max)

        if not np.isfinite(V[t, r, c]):
            return np.empty((0, 2), dtype=np.float64)

        path_rc = [(r, c)]

        _MOVES = [
            (-1,  0), ( 1,  0), ( 0, -1), ( 0,  1),
            (-1, -1), (-1,  1), ( 1, -1), ( 1,  1),
            ( 0,  0),           # wait
        ]

        for _ in range(max_steps):
            # Check goal reached
            if t <= T_max:
                if V[t, r, c] <= 0:
                    break
            else:
                if static_dist_field[r, c] <= 0:
                    break

            best_r, best_c, best_v = r, c, float('inf')

            if t < T_max:
                # Within prediction horizon: search (r', c', t+1)
                for dr, dc in _MOVES:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        v = V[t + 1, nr, nc]
                        if v < best_v:
                            best_r, best_c, best_v = nr, nc, v
            else:
                # Beyond horizon: fall back to static field
                for dr, dc in _MOVES[:-1]:  # no wait in static
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        v = static_dist_field[nr, nc]
                        if v < best_v:
                            best_r, best_c, best_v = nr, nc, v

            if not np.isfinite(best_v):
                break
            if (best_r, best_c) == (r, c) and t >= T_max:
                break  # stuck in static fallback

            r, c = best_r, best_c
            t = min(t + 1, T_max)
            path_rc.append((r, c))

        # Convert grid (row, col) -> standard coords (x_std, z_std)
        path_rc = np.array(path_rc, dtype=np.float64)
        path_std = np.empty_like(path_rc)
        path_std[:, 0] = path_rc[:, 1] * res + geo._x_min  # col -> x_std
        path_std[:, 1] = path_rc[:, 0] * res + geo._z_min  # row -> z_std
        return path_std
