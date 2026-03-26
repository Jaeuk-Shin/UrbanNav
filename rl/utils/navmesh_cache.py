"""
Precomputed navmesh cache for fast crosswalk and walkable-point queries.

Loads ``{Town}_navmesh_cache.npz`` files produced by
``export_carla_navmesh.py --cache`` and provides:

- Crosswalk polygons in UE (x, y, z) coordinates — replaces the runtime
  ``world.get_crosswalks()`` / navmesh-heuristic detection.
- Walkable-area point sampling within quadrant bounds — replaces the
  runtime ``world.get_random_location_from_navigation()`` loop.
"""

import os
from re import L
from typing import List, Optional, Tuple

import numpy as np


class NavmeshCache:
    """Loads and queries precomputed navmesh data for one or more towns.

    Usage::

        cache = NavmeshCache("navmeshes")
        if cache.load("Town02"):
            crosswalks = cache.get_crosswalks_ue("Town02")
            pt = cache.sample_in_bounds_ue("Town02", bounds)

    Note that roads are also walkable, so post-processing is needed to exclude roads from candidate spawn locations.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self._data: dict = {}   # town -> loaded arrays

    # ── Loading ────────────────────────────────────────────────────────

    def load(self, town: str) -> bool:
        """Load cache for *town*.  Returns True if the file was found."""
        if town in self._data:
            return True
        path = os.path.join(self.cache_dir, f"{town}_navmesh_cache.npz")
        if not os.path.exists(path):
            return False
        data = np.load(path)
        entry = {
            "crosswalk_vertices_ue": data["crosswalk_vertices_ue"],
            "crosswalk_offsets": data["crosswalk_offsets"],
            "walkable_points_ue": data["walkable_points_ue"],
        }
        # Sidewalk-only points (optional, may not exist in older caches)
        if "sidewalk_points_ue" in data:
            entry["sidewalk_points_ue"] = data["sidewalk_points_ue"]
        # Triangle meshes for BEV segmentation and geodesic distance
        # (optional, may not exist in caches built before these features).
        for key in ("sidewalk_tris_std", "crosswalk_tris_std",
                     "road_tris_std", "walkable_tris_std"):
            if key in data:
                entry[key] = data[key]
        self._data[town] = entry

        n_cw = len(data["crosswalk_offsets"]) - 1
        n_pts = len(data["walkable_points_ue"])
        n_sw_pts = len(entry.get("sidewalk_points_ue", []))
        n_sw_tris = len(entry.get("sidewalk_tris_std", []))
        print(f"  [NavmeshCache] Loaded {town}: "
              f"{n_cw} crosswalk polygons, {n_pts} walkable samples, "
              f"{n_sw_pts} sidewalk-only samples, "
              f"{n_sw_tris} sidewalk tris")
        return True


    def has_town(self, town: str) -> bool:
        return town in self._data


    # ── Crosswalks ─────────────────────────────────────────────────────

    def get_crosswalks_ue(self, town: str) -> List[np.ndarray]:
        """
        Return crosswalk polygons as ``List[(N, 3)]`` in UE (x, y, z).

        The format matches what ``ObstacleManager`` expects from
        ``_parse_crosswalks()``.
        """
        d = self._data.get(town)
        if d is None:
            return []
        verts = d["crosswalk_vertices_ue"]
        offsets = d["crosswalk_offsets"]
        polygons = []
        for i in range(len(offsets) - 1):
            poly = verts[offsets[i]:offsets[i + 1]]
            polygons.append(poly)
        return polygons

    # ── Walkable point queries ─────────────────────────────────────────

    def get_walkable_points_ue(self, town: str) -> np.ndarray | None:
        """Return ``(K, 3)`` walkable points in UE (x, y, z), or None."""
        d = self._data.get(town)
        if d is None:
            return None
        return d["walkable_points_ue"]

    def sample_in_bounds_ue(
        self,
        town: str,
        bounds: Tuple[float, float, float, float],
    ) -> np.ndarray | None:
        """Sample one random walkable point within *bounds*.

        Parameters
        ----------
        bounds : (ue_x_lo, ue_x_hi, ue_y_lo, ue_y_hi)

        Returns
        -------
        (3,) float32 array ``[ue_x, ue_y, ue_z]``, or None if no points
        fall within the bounds.
        """
        pts = self.get_walkable_points_ue(town)
        if pts is None or len(pts) == 0:
            return None
        xlo, xhi, ylo, yhi = bounds
        mask = ((pts[:, 0] >= xlo) & (pts[:, 0] <= xhi) &
                (pts[:, 1] >= ylo) & (pts[:, 1] <= yhi))
        region_pts = pts[mask]
        if len(region_pts) == 0:
            return None
        return region_pts[np.random.randint(len(region_pts))]

    def get_points_in_bounds_ue(
        self,
        town: str,
        bounds: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """Return all walkable points within *bounds*.

        Returns ``(N, 3)`` array (may be empty).
        """
        pts = self.get_walkable_points_ue(town)
        if pts is None or len(pts) == 0:
            return np.empty((0, 3), dtype=np.float32)
        xlo, xhi, ylo, yhi = bounds
        mask = ((pts[:, 0] >= xlo) & (pts[:, 0] <= xhi) &
                (pts[:, 1] >= ylo) & (pts[:, 1] <= yhi))
        return pts[mask]

    # ── Sidewalk-only point queries ──────────────────────────────────────

    def get_sidewalk_points_ue(self, town: str) -> np.ndarray | None:
        """Return ``(K, 3)`` sidewalk-only points in UE (x, y, z), or None."""
        d = self._data.get(town)
        if d is None:
            return None
        return d.get("sidewalk_points_ue")

    def sample_sidewalk_in_bounds_ue(
        self,
        town: str,
        bounds: Tuple[float, float, float, float],
    ) -> np.ndarray | None:
        """Sample one random sidewalk point within *bounds*.

        Returns None if no sidewalk-only data exists (callers should
        handle the None case explicitly rather than silently falling
        back to all-walkable points which include roads).

        Parameters
        ----------
        bounds : (ue_x_lo, ue_x_hi, ue_y_lo, ue_y_hi)

        Returns
        -------
        (3,) float32 array ``[ue_x, ue_y, ue_z]``, or None.
        """
        pts = self.get_sidewalk_points_ue(town)
        if pts is None or len(pts) == 0:
            return None
        xlo, xhi, ylo, yhi = bounds
        mask = ((pts[:, 0] >= xlo) & (pts[:, 0] <= xhi) &
                (pts[:, 1] >= ylo) & (pts[:, 1] <= yhi))
        region_pts = pts[mask]
        if len(region_pts) == 0:
            return None
        return region_pts[np.random.randint(len(region_pts))]

    def get_sidewalk_points_in_bounds_ue(
        self,
        town: str,
        bounds: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """Return all sidewalk-only points within *bounds*.

        Returns ``(N, 3)`` array (may be empty).
        """
        pts = self.get_sidewalk_points_ue(town)
        if pts is None or len(pts) == 0:
            return np.empty((0, 3), dtype=np.float32)
        xlo, xhi, ylo, yhi = bounds
        mask = ((pts[:, 0] >= xlo) & (pts[:, 0] <= xhi) &
                (pts[:, 1] >= ylo) & (pts[:, 1] <= yhi))
        return pts[mask]

    # ── Walkable triangle mesh (for geodesic distance) ─────────────────

    def get_walkable_tris_std(self, town: str) -> Optional[np.ndarray]:
        """Return all walkable navmesh triangles in standard 2D coords.

        Returns (N, 3, 2) float32 array, or None if not available.
        Used by class rl.utils.geodesic.GeodesicDistanceField to build
        the rasterized walkable grid.
        """
        d = self._data.get(town)
        if d is None:
            return None
        return d.get("walkable_tris_std")

    def get_sidewalk_crosswalk_tris_std(self, town: str) -> Optional[np.ndarray]:
        """Return sidewalk + crosswalk triangles (excluding roads) in standard 2D coords.

        Returns (N, 3, 2) float32 array, or None if neither is available.
        Used for geodesic distance computation so that paths stay on
        sidewalks and crosswalks rather than routing through roads.
        """
        d = self._data.get(town)
        if d is None:
            return None
        parts = []
        for key in ("sidewalk_tris_std", "crosswalk_tris_std"):
            arr = d.get(key)
            if arr is not None and len(arr) > 0:
                parts.append(arr)
        if not parts:
            return None
        return np.concatenate(parts, axis=0)

    # ── Area triangle meshes (for BEV segmentation rendering) ──────────

    def get_area_triangles_std(self, town: str) -> dict:
        """
        Return navmesh triangles per area type in standard 2D coords.

        Returns dict with keys 
        - sidewalk_tris_std
        - crosswalk_tris_std
        - road_tris_std
        each an
        (N, 3, 2) float32 array (may be empty).
        Suitable for rendering as matplotlib.collections.PolyCollection.
        """
        d = self._data.get(town)
        empty = np.empty((0, 3, 2), dtype=np.float32)
        if d is None:
            return {"sidewalk_tris_std": empty,
                    "crosswalk_tris_std": empty,
                    "road_tris_std": empty}
        return {
            "sidewalk_tris_std": d.get("sidewalk_tris_std", empty),
            "crosswalk_tris_std": d.get("crosswalk_tris_std", empty),
            "road_tris_std": d.get("road_tris_std", empty),
        }

    # methods for debugging
    def get_all_mesh_types(self, town: str):
        # just for debugging
        return self._data[town].keys()

    def get_crosswalks_offsets(self, town):
        # just for debugging
        d = self._data.get(town)
        if d is None:
            return []
        else:
            return d["crosswalk_offsets"]

    def num_crosswalks(self, town):
        offsets = self.get_crosswalks_offsets(town)
        return max(len(offsets) - 1, 0)


def test_NavmeshCache():
    import pathlib
    cache_dir = pathlib.Path(__file__).parent.parent.parent / 'navmeshes'
    mesh_cache = NavmeshCache(cache_dir=cache_dir)
    for town in ['Town02', 'Town03', 'Town05', 'Town10HD']:
        mesh_cache.load(town=town)
    
        print(f'[{town}] # crosswalk polygons', mesh_cache.num_crosswalks(town))

    return


if __name__ == '__main__':
    test_NavmeshCache()