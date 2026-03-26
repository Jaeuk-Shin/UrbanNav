"""Shared mesh geometry utilities for navmesh puncturing.

Used by both ``PedestrianManager`` (SFM boundary constraint) and
``GeodesicDistanceField`` (obstacle-aware geodesic distance).

The core operation is "puncturing": given a set of walkable navmesh
triangles and obstacle oriented bounding boxes (OBBs), remove triangles
that overlap with any OBB.  The result is a mesh with holes where
obstacles sit.
"""

from typing import List

import numpy as np


# ── Point-in-polygon ──────────────────────────────────────────────────


def points_in_convex_polygon(points: np.ndarray,
                             polygon: np.ndarray) -> np.ndarray:
    """Test which *points* lie inside a convex *polygon*.

    Works for either clockwise or counter-clockwise vertex ordering.

    Parameters
    ----------
    points  : (N, 2) — test points
    polygon : (K, 2) — ordered convex polygon vertices

    Returns
    -------
    (N,) boolean array — True where the point is inside
    """
    K = polygon.shape[0]
    all_pos = np.ones(points.shape[0], dtype=bool)
    all_neg = np.ones(points.shape[0], dtype=bool)

    for i in range(K):
        j = (i + 1) % K
        edge = polygon[j] - polygon[i]
        d = points - polygon[i]
        cross = edge[0] * d[:, 1] - edge[1] * d[:, 0]
        all_pos &= cross >= 0
        all_neg &= cross <= 0

    return all_pos | all_neg


# ── OBB inflation ────────────────────────────────────────────────────


def inflate_obb(corners: np.ndarray, amount: float) -> np.ndarray:
    """Push each corner of a convex polygon outward by *amount* metres.

    Parameters
    ----------
    corners : (K, 2) — ordered polygon vertices
    amount  : float  — inflation distance (metres)

    Returns
    -------
    (K, 2) — inflated corners
    """
    centre = corners.mean(axis=0)
    dirs = corners - centre
    lengths = np.linalg.norm(dirs, axis=1, keepdims=True)
    return corners + dirs / np.maximum(lengths, 1e-6) * amount


# ── Triangle puncturing ──────────────────────────────────────────────


def _points_in_triangles(points: np.ndarray,
                         triangles: np.ndarray) -> np.ndarray:
    """Test which triangles contain at least one of the given *points*.

    Uses barycentric coordinate test (vectorised over both points and
    triangles).

    Parameters
    ----------
    points    : (P, 2)
    triangles : (N, 3, 2)

    Returns
    -------
    (N,) boolean — True where the triangle contains at least one point
    """
    # For each triangle, compute vectors from v0
    v0 = triangles[:, 0]                         # (N, 2)
    e1 = triangles[:, 1] - v0                    # (N, 2)
    e2 = triangles[:, 2] - v0                    # (N, 2)

    # Precompute dot products for the triangle side
    d00 = (e1 * e1).sum(axis=1)                  # (N,)
    d01 = (e1 * e2).sum(axis=1)                  # (N,)
    d11 = (e2 * e2).sum(axis=1)                  # (N,)
    inv_denom = d00 * d11 - d01 * d01            # (N,)

    hit = np.zeros(len(triangles), dtype=bool)

    # Test each point against all triangles (P is small — typically 4 OBB corners)
    for p in points:
        dp = p - v0                              # (N, 2)
        d20 = (dp * e1).sum(axis=1)              # (N,)
        d21 = (dp * e2).sum(axis=1)              # (N,)

        # Barycentric coords (guard against degenerate triangles)
        safe_denom = np.where(np.abs(inv_denom) < 1e-12, 1.0, inv_denom)
        u = (d11 * d20 - d01 * d21) / safe_denom
        v = (d00 * d21 - d01 * d20) / safe_denom

        inside = (u >= 0) & (v >= 0) & (u + v <= 1)
        # Degenerate triangles never match
        inside &= np.abs(inv_denom) >= 1e-12
        hit |= inside

    return hit


def puncture_triangles(triangles: np.ndarray,
                       obstacle_polygons: List[np.ndarray],
                       buffer: float = 0.0) -> np.ndarray:
    """Remove triangles that overlap with any obstacle polygon.

    A triangle is removed when **any** of the following is true:

    1. Its centroid or any vertex falls inside an obstacle polygon
       (triangle-points-in-obstacle test).
    2. Any obstacle polygon corner falls inside the triangle
       (obstacle-points-in-triangle — reverse containment test).

    Together these catch both the common case (small obstacle crossing a
    triangle edge) and the enclosed case (obstacle OBB fits entirely
    inside a large triangle, e.g. a truck parked on a wide crosswalk).

    Parameters
    ----------
    triangles : (N, 3, 2)
        Triangle mesh vertices in any consistent 2D coordinate system.
    obstacle_polygons : list of (K, 2)
        Convex obstacle polygons in the **same** coordinate system as
        *triangles*.
    buffer : float
        Inflate each obstacle polygon outward by this amount (metres)
        before the overlap test.  Use e.g. pedestrian body radius or
        agent collision radius.

    Returns
    -------
    (M, 3, 2) — remaining triangles (M ≤ N)
    """
    if not obstacle_polygons or triangles.shape[0] == 0:
        return triangles

    # Optionally inflate
    polys = obstacle_polygons
    if buffer > 0:
        polys = [inflate_obb(p, buffer) for p in polys]

    # Check centroids AND all three vertices (triangle points in obstacle)
    centroids = triangles.mean(axis=1)                    # (N, 2)
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]

    remove = np.zeros(triangles.shape[0], dtype=bool)
    for poly in polys:
        remove |= points_in_convex_polygon(centroids, poly)
        remove |= points_in_convex_polygon(v0, poly)
        remove |= points_in_convex_polygon(v1, poly)
        remove |= points_in_convex_polygon(v2, poly)
        # Reverse containment: obstacle corners inside triangle
        remove |= _points_in_triangles(poly, triangles)

    return triangles[~remove]
