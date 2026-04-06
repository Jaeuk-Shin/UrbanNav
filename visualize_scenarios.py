#!/usr/bin/env python3
"""Visualize precomputed scenarios using the existing BEV obstacle figure.

Loads ``.npz`` scenario files from ``--scenario_dir`` and renders each one
using ``log_obstacle_bev_figure`` from ``rl/utils/vis.py``.  Outputs are
saved to ``--output_dir`` as PNG files.

Modes:
  - **2x2 quadrant grid** (default): One scenario per quadrant in a 2x2
    grid.  Generates ``--num_pages`` figures, each showing a different
    random scenario per quadrant.
  - **Single quadrant**: ``--quadrant 0 --num 16`` shows 16 scenarios
    from quadrant 0 in a 4x4 grid.

Usage:
    # 2x2 grid (one per quadrant)
    python visualize_scenarios.py \
        --scenario_dir rl/scenarios/ \
        --navmesh_dir navmeshes/ \
        --town Town02

    # Single quadrant, 16 scenarios
    python visualize_scenarios.py \
        --scenario_dir rl/scenarios/ \
        --navmesh_dir navmeshes/ \
        --town Town02 \
        --quadrant 0 --num 16
"""
import argparse
import math
from pathlib import Path
from typing import List, Optional

import numpy as np

from rl.utils.mesh_utils import points_in_convex_polygon
from rl.utils.navmesh_cache import NavmeshCache
from rl.utils.vis import log_obstacle_bev_figure


# ── Helpers ──────────────────────────────────────────────────────────


def _compute_bev_meta(
    start_std: np.ndarray,
    goal_std: np.ndarray,
    geo_path_std: Optional[np.ndarray] = None,
    obstacle_corners_std: Optional[np.ndarray] = None,
    img_size: int = 512,
    fov_deg: float = 90.0,
    padding: float = 5.0,
) -> dict:
    """Synthesize a ``bev_meta`` dict centred on the scenario geometry.

    Computes the minimal altitude that frames all key points (start,
    goal, geodesic path, obstacle corners) within the image.
    """
    # Collect all points that should be visible
    points = [start_std.reshape(1, 2), goal_std.reshape(1, 2)]
    if geo_path_std is not None and len(geo_path_std) > 0:
        points.append(np.asarray(geo_path_std).reshape(-1, 2))
    if obstacle_corners_std is not None and len(obstacle_corners_std) > 0:
        flat = np.asarray(obstacle_corners_std).reshape(-1, 2)
        points.append(flat)

    all_pts = np.concatenate(points, axis=0)
    all_pts = all_pts[np.isfinite(all_pts).all(axis=1)]

    if len(all_pts) == 0:
        center = (start_std + goal_std) / 2.0
        altitude = 30.0
    else:
        center = all_pts.mean(axis=0)
        # Extent from centre in each axis
        dx = np.abs(all_pts[:, 0] - center[0]).max() + padding
        dz = np.abs(all_pts[:, 1] - center[1]).max() + padding
        half_extent = max(dx, dz)
        # altitude = half_extent / tan(fov/2) so everything fits
        fov_rad = math.radians(fov_deg)
        altitude = max(half_extent / math.tan(fov_rad / 2.0), 10.0)

    return {
        'altitude': altitude,
        'img_size': img_size,
        'center_xz': center.astype(np.float64),
        'fov_deg': fov_deg,
    }


def scenario_to_layout(
    scenario: dict,
    cache: Optional[NavmeshCache] = None,
    town: Optional[str] = None,
    quadrant: int = 0,
    env_idx: int = 0,
) -> dict:
    """Convert a loaded ``.npz`` scenario dict into an ``obstacle_layout``.

    The returned dict matches the format expected by
    ``log_obstacle_bev_figure``.
    """
    # Obstacles
    bp_ids = scenario.get('obstacle_bp_ids', np.array([], dtype='U64'))
    corners_all = scenario.get('obstacle_corners_std',
                               np.zeros((0, 4, 2), dtype=np.float32))
    types_all = scenario.get('obstacle_scenario_types',
                             np.array([], dtype='U32'))
    obstacles = []
    for j in range(len(bp_ids)):
        obstacles.append({
            'corners_std': corners_all[j].astype(np.float64),
            'center_std': corners_all[j].mean(axis=0).astype(np.float64),
            'scenario_type': str(types_all[j]),
        })

    # Crosswalks from navmesh cache, tagged as blocked when applicable
    blocked_obbs = scenario.get('blocked_crosswalk_obbs_std',
                                np.zeros((0, 4, 2), dtype=np.float32))
    crosswalks_std = []
    if cache is not None and town is not None:
        cws_ue = cache.get_crosswalks_ue(town)
        for cw in cws_ue:
            # UE (N, 3) -> standard (N, 2): x_std = ue_y, z_std = ue_x
            cw_std = np.stack([cw[:, 1], cw[:, 0]], axis=1)
            # Check if this crosswalk's centroid falls inside any
            # blocked crosswalk OBB
            is_blocked = False
            centroid = cw_std.mean(axis=0).reshape(1, 2)
            for obb in blocked_obbs:
                if len(obb) >= 3:
                    inside = points_in_convex_polygon(centroid, obb)
                    if inside[0]:
                        is_blocked = True
                        break
            crosswalks_std.append({
                'polygon_std': cw_std,
                'blocked': is_blocked,
            })

    # Navmesh segmentation triangles (optional, for background rendering)
    seg_data = {}
    if cache is not None and town is not None:
        area_tris = cache.get_area_triangles_std(town)
        for key in ('sidewalk_tris_std', 'crosswalk_tris_std', 'road_tris_std'):
            tris = area_tris.get(key)
            if tris is not None and len(tris) > 0:
                seg_data[key] = tris

    # Geodesic path: trace from the distance field if available
    geo_path = None
    start_std = scenario['start_std'].astype(np.float64)
    goal_std = scenario['goal_std'].astype(np.float64)
    dist_field = scenario.get('dist_field')
    if dist_field is not None:
        from rl.utils.geodesic import GeodesicDistanceField
        x_min = float(scenario['grid_x_min'])
        z_min = float(scenario['grid_z_min'])
        res = float(scenario['grid_resolution'])
        H, W = dist_field.shape
        # Reconstruct a minimal GeodesicDistanceField for trace_path
        # We only need _x_min, _z_min, _resolution, _W, _H
        geo = GeodesicDistanceField.__new__(GeodesicDistanceField)
        geo._x_min = x_min
        geo._z_min = z_min
        geo._resolution = res
        geo._W = W
        geo._H = H
        geo_path = geo.trace_path(dist_field.astype(np.float64), start_std)

    # Scenario metadata
    template = str(scenario.get('scenario_template', 'unknown'))
    goal_method = str(scenario.get('goal_method', 'unknown'))
    geo_dist = float(scenario.get('geodesic_distance', 0.0))
    euc_dist = float(scenario.get('euclidean_distance', 0.0))

    # Collect unique scenario types for the info box
    region_scenarios = list(set(str(t) for t in types_all)) if len(types_all) > 0 else []

    # Agent entry
    agent = {
        'ego_std': start_std,
        'goal_std': goal_std,
        'goal_method': goal_method,
        'initial_distance': euc_dist,
        'geodesic_distance': geo_dist,
        'geodesic_path_std': geo_path,
        'step_count': 0,
        'town': town or '?',
        'quadrant': f'q{quadrant}',
        'region_scenarios': region_scenarios,
    }

    layout = {
        'obstacles': obstacles,
        'crosswalks': crosswalks_std,
        'agents': [agent],
        **seg_data,
    }
    return layout


# ── Main ─────────────────────────────────────────────────────────────


def _load_scenario(path, cache, town, quadrant, env_idx, args):
    """Load one .npz and return (layout, bev_meta)."""
    data = np.load(str(path), allow_pickle=False)
    scenario = {k: data[k] for k in data.files}

    layout = scenario_to_layout(
        scenario, cache=cache, town=town,
        quadrant=quadrant, env_idx=env_idx)

    corners = scenario.get('obstacle_corners_std',
                           np.zeros((0, 4, 2), dtype=np.float32))
    meta = _compute_bev_meta(
        scenario['start_std'].astype(np.float64),
        scenario['goal_std'].astype(np.float64),
        geo_path_std=layout['agents'][0].get('geodesic_path_std'),
        obstacle_corners_std=corners,
        img_size=args.img_size,
        fov_deg=args.fov,
        padding=args.padding,
    )
    return layout, meta


def main():
    parser = argparse.ArgumentParser(
        description='Visualize precomputed scenarios')
    parser.add_argument('--scenario_dir', type=str, required=True,
                        help='Root scenario directory (contains Town*/q*/*.npz)')
    parser.add_argument('--navmesh_dir', type=str, default=None,
                        help='Navmesh cache directory (for crosswalk/mesh overlays)')
    parser.add_argument('--town', type=str, required=True,
                        help='Town name (e.g., Town02)')
    parser.add_argument('--quadrant', type=int, default=None,
                        help='Single quadrant to visualize. If omitted, '
                             'renders a 2x2 grid (one per quadrant).')
    parser.add_argument('--num', type=int, default=16,
                        help='Number of scenarios per quadrant in single-'
                             'quadrant mode (default: 16)')
    parser.add_argument('--num_pages', type=int, default=4,
                        help='Number of 2x2 grid pages to generate in '
                             'quadrant-grid mode (default: 4)')
    parser.add_argument('--offset', type=int, default=0,
                        help='Skip first N scenarios (default: 0)')
    parser.add_argument('--output_dir', type=str, default='vis_scenarios',
                        help='Output directory for PNGs (default: vis_scenarios)')
    parser.add_argument('--img_size', type=int, default=512,
                        help='BEV image size in pixels (default: 512)')
    parser.add_argument('--fov', type=float, default=90.0,
                        help='BEV camera FOV in degrees (default: 90)')
    parser.add_argument('--padding', type=float, default=5.0,
                        help='Padding around geometry in metres (default: 5.0)')
    parser.add_argument('--grid_cols', type=int, default=None,
                        help='Grid columns (default: 2 for grid mode, '
                             'auto for single-quadrant mode)')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load navmesh cache
    cache = None
    if args.navmesh_dir:
        cache = NavmeshCache(args.navmesh_dir)
        cache.load(args.town)

    if args.quadrant is not None:
        # ── Single-quadrant mode ─────────────────────────────────────
        q_dir = Path(args.scenario_dir) / args.town / f'q{args.quadrant}'
        if not q_dir.exists():
            print(f"ERROR: {q_dir} does not exist")
            return
        files = sorted(q_dir.glob('scenario_*.npz'))
        files = files[args.offset:args.offset + args.num]
        if not files:
            print(f"ERROR: No scenario files in {q_dir}")
            return

        print(f"Visualizing {len(files)} scenarios from q{args.quadrant}")
        layouts, metas = [], []
        for f in files:
            layout, meta = _load_scenario(
                f, cache, args.town, args.quadrant, len(layouts), args)
            layouts.append(layout)
            metas.append(meta)

        log_obstacle_bev_figure(
            obstacle_layouts=layouts,
            bev_metas=metas,
            grid_cols=args.grid_cols,
            iteration=0,
            save_dir=str(out_dir),
            wandb=None,
        )

    else:
        # ── 2x2 quadrant grid mode ──────────────────────────────────
        # Discover available quadrants
        town_dir = Path(args.scenario_dir) / args.town
        quadrant_files = {}   # qi -> sorted list of .npz paths
        for qi in range(4):
            q_dir = town_dir / f'q{qi}'
            if q_dir.exists():
                files = sorted(q_dir.glob('scenario_*.npz'))
                if files:
                    quadrant_files[qi] = files

        if not quadrant_files:
            print(f"ERROR: No scenario files found under {town_dir}")
            return

        available_qs = sorted(quadrant_files.keys())
        print(f"Found quadrants: {available_qs} "
              f"({', '.join(f'q{q}={len(quadrant_files[q])}' for q in available_qs)})")

        for page in range(args.num_pages):
            layouts, metas = [], []
            for qi in range(4):
                if qi not in quadrant_files:
                    # Empty placeholder — will be hidden by vis code
                    layouts.append(None)
                    metas.append(None)
                    continue

                files = quadrant_files[qi]
                idx = args.offset + page
                if idx >= len(files):
                    idx = idx % len(files)  # wrap around

                layout, meta = _load_scenario(
                    files[idx], cache, args.town, qi, len(layouts), args)
                layouts.append(layout)
                metas.append(meta)

            log_obstacle_bev_figure(
                obstacle_layouts=[l for l in layouts if l is not None],
                bev_metas=[m for m in metas if m is not None],
                grid_cols=args.grid_cols or 2,
                iteration=page,
                save_dir=str(out_dir),
                wandb=None,
            )

    print(f"Saved to {out_dir}/")


if __name__ == '__main__':
    main()
