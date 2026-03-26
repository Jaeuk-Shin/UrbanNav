"""Visualize precomputed navmesh caches.

Usage:
    # Single town
    python visualize_navmesh_cache.py --town Town02

    # All cached towns
    python visualize_navmesh_cache.py --all

    # Custom cache directory and output
    python visualize_navmesh_cache.py --town Town03 --cache_dir navmeshes/ --output town03_vis.png
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from rl.utils.navmesh_cache import NavmeshCache


def visualize_town(cache: NavmeshCache, town: str, output: str | None = None):
    """Render a single BEV figure for one town's navmesh cache."""
    if not cache.load(town):
        print(f"No cache found for {town}, skipping.")
        return

    area_tris = cache.get_area_triangles_std(town)
    crosswalks_ue = cache.get_crosswalks_ue(town)
    walkable_pts = cache.get_walkable_points_ue(town)

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle(f"{town} — Navmesh Cache", fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")

    # Layer 1: Area triangles (road / sidewalk)
    layer_cfg = [
        ("road_tris_std",      "#555555", "#444444", "road"),
        ("sidewalk_tris_std",  "#8a8a5c", "#6e6e48", "sidewalk"),
        # ("crosswalk_tris_std", "#7ecfcf", "#3fc1c1", "crosswalk"),
    ]
    for key, fc, ec, label in layer_cfg:
        tris = area_tris.get(key)
        if tris is not None and len(tris) > 0:
            pc = PolyCollection(tris, facecolor=fc, edgecolor=ec,
                                alpha=0.7, linewidth=0.1)
            ax.add_collection(pc)
            ax.scatter([], [], c=fc, s=30, label=f"{label} ({len(tris)})")

    # Layer 2: Walkable sample points (UE → std 2D: x_std=ue_y, z_std=ue_x)
    if walkable_pts is not None and len(walkable_pts) > 0:
        step = max(1, len(walkable_pts) // 20000)
        pts = walkable_pts[::step]
        ax.scatter(pts[:, 1], pts[:, 0], s=0.4, c="#4a6fa5", alpha=0.2,
                   label=f"walkable ({len(walkable_pts):,})")

    # Layer 2b: Sidewalk-only sample points (highlighted)
    sidewalk_pts = cache.get_sidewalk_points_ue(town)
    if sidewalk_pts is not None and len(sidewalk_pts) > 0:
        step = max(1, len(sidewalk_pts) // 20000)
        sw_pts = sidewalk_pts[::step]
        ax.scatter(sw_pts[:, 1], sw_pts[:, 0], s=0.4, c="#f5a623", alpha=0.3,
                   label=f"sidewalk pts ({len(sidewalk_pts):,})")

    # Layer 3: Crosswalk polygons (UE → std 2D)
    if crosswalks_ue:
        polys_2d = [np.stack([p[:, 1], p[:, 0]], axis=1) for p in crosswalks_ue]
        pc = PolyCollection(polys_2d, facecolor="#7ecfcf", edgecolor="#3fc1c1",
                            alpha=0.6, linewidth=0.5)
        ax.add_collection(pc)
        ax.scatter([], [], c="#7ecfcf", s=30,
                   label=f"crosswalk polys ({len(crosswalks_ue)})")

    ax.legend(loc="upper right", fontsize=8)
    ax.autoscale_view()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.tight_layout()
    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved → {output}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize navmesh caches")
    parser.add_argument("--town", type=str, help="Town name (e.g. Town02)")
    parser.add_argument("--all", action="store_true", help="Visualize all cached towns")
    parser.add_argument("--cache_dir", type=str, default="navmeshes",
                        help="Path to navmesh cache directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (shows interactive plot if omitted). "
                             "With --all, this is treated as a directory.")
    args = parser.parse_args()

    if not args.town and not args.all:
        parser.error("Specify --town <name> or --all")

    cache = NavmeshCache(args.cache_dir)

    if args.all:
        npz_files = sorted(Path(args.cache_dir).glob("*_navmesh_cache.npz"))
        towns = [f.stem.replace("_navmesh_cache", "") for f in npz_files]
        if not towns:
            print(f"No *_navmesh_cache.npz files found in {args.cache_dir}")
            return
        out_dir = args.output or "navmesh_vis"
        os.makedirs(out_dir, exist_ok=True)
        for town in towns:
            visualize_town(cache, town, output=os.path.join(out_dir, f"{town}.png"))
    else:
        visualize_town(cache, args.town, output=args.output)


if __name__ == "__main__":
    main()
