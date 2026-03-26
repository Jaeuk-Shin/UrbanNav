#!/usr/bin/env python3
"""
Export CARLA navigation mesh (.bin) to OBJ or NumPy format.

Parses the pre-built Recast/Detour binary navmesh files shipped with CARLA
(CarlaUE4/Content/Carla/Maps/Nav/*.bin) and exports them as triangulated
meshes. No running CARLA server is required.

Overview
--------
CARLA stores pedestrian navigation meshes as Recast/Detour .bin files.
This script reads those binary files and converts them to standard formats
for visualization, analysis, or downstream use in planning/RL pipelines.

Output formats
--------------
- OBJ (default): Wavefront OBJ with triangles grouped by surface type.
  Opens in Blender, MeshLab, Open3D, trimesh, etc. Text-based and easy
  to parse in any language.
- NPZ: Compressed NumPy archive with arrays `vertices` (N,3 float32),
  `triangles` (M,3 int32), and `area_types` (M, int32). Convenient for
  direct use in Python without file parsing.

Surface types (OBJ groups / area_types values)
-----------------------------------------------
  0 = block        Blocked / non-walkable area (CARLA_AREA_BLOCK)
  1 = sidewalk     Sidewalk surfaces          (CARLA_AREA_SIDEWALK)
  2 = crosswalk    Designated crosswalks       (CARLA_AREA_CROSSWALK)
  3 = road         Road surface                (CARLA_AREA_ROAD)
  4 = grass        Grass areas                 (CARLA_AREA_GRASS)
  63 = obstacle    Default Recast walkable area / obstacle polygons

Coordinate system
-----------------
Vertices are in Recast convention: X-right, Y-up, Z-forward (meters).

Usage
-----
    # List available maps
    python export_carla_navmesh.py --list

    # Export Town01 navmesh to OBJ
    python export_carla_navmesh.py --map Town01

    # Custom CARLA path and output filename
    python export_carla_navmesh.py --map Town03 --carla /path/to/Carla -o town03_nav.obj

    # Export as NumPy arrays (.npz)
    python export_carla_navmesh.py --map Town01 --format npz

    # Load the NPZ in Python
    data = np.load("Town01_navmesh.npz")
    verts, tris, areas = data["vertices"], data["triangles"], data["area_types"]
    sidewalk_mask = areas == 4
    sidewalk_tris = tris[sidewalk_mask]
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import ListedColormap, BoundaryNorm

# ── Recast/Detour constants ──────────────────────────────────────────────────

NAVMESHSET_MAGIC = 0x4D534554   # 'MSET' little-endian
NAVMESHSET_VERSION = 1
DT_NAVMESH_MAGIC = 0x444E4156   # 'DNAV' little-endian
DT_NAVMESH_VERSION = 7
DT_VERTS_PER_POLYGON = 6

# CARLA area types (from NavAreas enum in Navigation.h)
#   0 = CARLA_AREA_BLOCK      (flags 0x01 CARLA_TYPE_NONE)
#   1 = CARLA_AREA_SIDEWALK   (flags 0x02 CARLA_TYPE_SIDEWALK)
#   2 = CARLA_AREA_CROSSWALK  (flags 0x04 CARLA_TYPE_CROSSWALK)
#   3 = CARLA_AREA_ROAD       (flags 0x08 CARLA_TYPE_ROAD)
#   4 = CARLA_AREA_GRASS      (flags 0x10 CARLA_TYPE_GRASS)
AREA_NAMES = {
    0: "block",
    1: "sidewalk",
    2: "crosswalk",
    3: "road",
    4: "grass",
    63: "obstacle",  # RC_WALKABLE_AREA / default non-walkable
}

# ── Struct sizes ─────────────────────────────────────────────────────────────

SIZEOF_NAVMESH_SET_HEADER = 40   # 3 ints + dtNavMeshParams(28)
SIZEOF_TILE_HEADER = 8           # uint32 tileRef + int32 dataSize
SIZEOF_MESH_HEADER = 100         # dtMeshHeader
SIZEOF_POLY = 32                 # dtPoly
SIZEOF_LINK = 12                 # dtLink
SIZEOF_DETAIL_MESH = 12          # dtPolyDetail (with padding)
SIZEOF_DETAIL_TRI = 4            # 4 unsigned chars
SIZEOF_BV_NODE = 16              # dtBVNode
SIZEOF_OFF_MESH_CON = 48         # dtOffMeshConnection


def parse_navmesh(filepath: str) -> dict:
    """Parse a CARLA Recast/Detour .bin navmesh file.

    Returns a dict with keys: vertices, triangles, area_types, metadata.
    """
    filepath = Path(filepath)
    data = filepath.read_bytes()
    off = 0

    def read(fmt):
        nonlocal off
        size = struct.calcsize(fmt)
        vals = struct.unpack_from(fmt, data, off)
        off += size
        return vals

    # ── NavMeshSetHeader ─────────────────────────────────────────────────
    magic, version, num_tiles = read("<iii")
    assert magic == NAVMESHSET_MAGIC, f"Bad magic: 0x{magic:08X}"
    assert version == NAVMESHSET_VERSION, f"Bad version: {version}"

    orig = read("<3f")
    tile_width, tile_height = read("<ff")
    max_tiles, max_polys = read("<ii")

    all_vertices = []
    all_triangles = []
    all_area_types = []
    vertex_offset = 0

    for tile_idx in range(num_tiles):
        # ── NavMeshTileHeader ────────────────────────────────────────────
        tile_ref, data_size = read("<Ii")
        if data_size == 0:
            continue
        tile_data_start = off

        # ── dtMeshHeader ─────────────────────────────────────────────────
        tile_magic, tile_ver = read("<ii")
        assert tile_magic == DT_NAVMESH_MAGIC, f"Bad tile magic: 0x{tile_magic:08X}"

        tx, ty, tlayer, tuser_id = read("<iiiI")
        poly_count, vert_count, max_link_count = read("<iii")
        detail_mesh_count, detail_vert_count, detail_tri_count = read("<iii")
        bv_node_count, off_mesh_con_count, off_mesh_base = read("<iii")
        walkable_h, walkable_r, walkable_c = read("<fff")
        bmin = read("<3f")
        bmax = read("<3f")
        bv_quant_factor = read("<f")[0]

        # ── Vertices: float[3] * vert_count ──────────────────────────────
        verts_start = off
        base_verts = np.frombuffer(
            data, dtype="<f4", count=vert_count * 3, offset=verts_start
        ).reshape(-1, 3).copy()
        off = verts_start + vert_count * 12

        # ── Polys: dtPoly * poly_count ───────────────────────────────────
        polys = []
        polys_start = off
        for _ in range(poly_count):
            first_link = struct.unpack_from("<I", data, off)[0]
            off += 4
            pverts = struct.unpack_from("<6H", data, off)
            off += 12
            pneis = struct.unpack_from("<6H", data, off)
            off += 12
            pflags, pvert_count, parea_and_type = struct.unpack_from("<HBB", data, off)
            off += 4
            polys.append({
                "verts": pverts,
                "vert_count": pvert_count,
                "area": parea_and_type & 0x3F,
                "type": (parea_and_type >> 6) & 0x03,
            })

        # ── Links: skip ──────────────────────────────────────────────────
        off += max_link_count * SIZEOF_LINK

        # ── Detail meshes: dtPolyDetail * detail_mesh_count ──────────────
        detail_meshes = []
        for _ in range(detail_mesh_count):
            dm_vert_base, dm_tri_base = struct.unpack_from("<II", data, off)
            off += 8
            dm_vert_count, dm_tri_count = struct.unpack_from("<BB", data, off)
            off += 2
            off += 2  # padding to 12 bytes
            detail_meshes.append({
                "vert_base": dm_vert_base,
                "tri_base": dm_tri_base,
                "vert_count": dm_vert_count,
                "tri_count": dm_tri_count,
            })

        # ── Detail vertices: float[3] * detail_vert_count ────────────────
        detail_verts_start = off
        if detail_vert_count > 0:
            detail_verts = np.frombuffer(
                data, dtype="<f4", count=detail_vert_count * 3, offset=detail_verts_start
            ).reshape(-1, 3).copy()
        else:
            detail_verts = np.empty((0, 3), dtype=np.float32)
        off = detail_verts_start + detail_vert_count * 12

        # ── Detail triangles: uchar[4] * detail_tri_count ────────────────
        detail_tris_start = off
        if detail_tri_count > 0:
            detail_tris = np.frombuffer(
                data, dtype=np.uint8, count=detail_tri_count * 4, offset=detail_tris_start
            ).reshape(-1, 4).copy()
        else:
            detail_tris = np.empty((0, 4), dtype=np.uint8)
        off = detail_tris_start + detail_tri_count * 4

        # ── BV tree + off-mesh: skip ─────────────────────────────────────
        off = tile_data_start + data_size

        # ── Build triangulated mesh from detail sub-meshes ───────────────
        # Combine base_verts and detail_verts into a single array.
        # Detail triangle vertex indices work as follows:
        #   index < poly.vert_count → base_verts[poly.verts[index]]
        #   index >= poly.vert_count → detail_verts[dm.vert_base + (index - poly.vert_count)]
        combined_verts = []
        combined_tris = []
        combined_areas = []
        vert_map = {}  # (source, index) -> combined index

        def get_vert_idx(source, idx):
            key = (source, idx)
            if key not in vert_map:
                vert_map[key] = len(combined_verts)
                if source == "base":
                    combined_verts.append(base_verts[idx])
                else:
                    combined_verts.append(detail_verts[idx])
            return vert_map[key]

        for pi in range(min(poly_count, detail_mesh_count)):
            poly = polys[pi]
            dm = detail_meshes[pi]

            # Skip off-mesh connection polygons
            if poly["type"] != 0:
                continue

            for ti in range(dm["tri_count"]):
                tri = detail_tris[dm["tri_base"] + ti]
                tri_indices = []
                for k in range(3):
                    vi = int(tri[k])
                    if vi < poly["vert_count"]:
                        ci = get_vert_idx("base", poly["verts"][vi])
                    else:
                        ci = get_vert_idx("detail", dm["vert_base"] + (vi - poly["vert_count"]))
                    tri_indices.append(ci)
                combined_tris.append(tri_indices)
                combined_areas.append(poly["area"])

        if combined_verts:
            tile_verts = np.array(combined_verts, dtype=np.float32)
            tile_tris = np.array(combined_tris, dtype=np.int32) + vertex_offset
            tile_areas = np.array(combined_areas, dtype=np.int32)

            all_vertices.append(tile_verts)
            all_triangles.append(tile_tris)
            all_area_types.append(tile_areas)
            vertex_offset += len(tile_verts)

    vertices = np.concatenate(all_vertices, axis=0) if all_vertices else np.empty((0, 3))
    triangles = np.concatenate(all_triangles, axis=0) if all_triangles else np.empty((0, 3), dtype=np.int32)
    area_types = np.concatenate(all_area_types, axis=0) if all_area_types else np.empty(0, dtype=np.int32)

    return {
        "vertices": vertices,
        "triangles": triangles,
        "area_types": area_types,
        "metadata": {
            "origin": orig,
            "tile_width": tile_width,
            "tile_height": tile_height,
            "num_tiles": num_tiles,
            "walkable_height": walkable_h,
            "walkable_radius": walkable_r,
            "walkable_climb": walkable_c,
        },
    }


def write_obj(navmesh: dict, filepath: str):
    """Write navmesh to Wavefront OBJ with area-type groups."""
    verts = navmesh["vertices"]
    tris = navmesh["triangles"]
    areas = navmesh["area_types"]

    with open(filepath, "w") as f:
        f.write(f"# CARLA navmesh export\n")
        f.write(f"# Vertices: {len(verts)}, Triangles: {len(tris)}\n\n")

        # Write vertices (Recast uses Y-up; CARLA/UE4 uses Z-up internally
        # but the navmesh stores coords in Recast convention: X-right, Y-up, Z-forward)
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write("\n")

        # Group triangles by area type
        unique_areas = np.unique(areas) if len(areas) > 0 else []
        for area_id in unique_areas:
            area_name = AREA_NAMES.get(area_id, f"area_{area_id}")
            mask = areas == area_id
            area_tris = tris[mask]
            f.write(f"\ng {area_name}\n")
            for tri in area_tris:
                # OBJ faces are 1-indexed
                f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

    print(f"Wrote {filepath} ({len(verts)} vertices, {len(tris)} triangles)")


def write_npz(navmesh: dict, filepath: str):
    """Write navmesh to compressed NumPy archive."""
    np.savez_compressed(
        filepath,
        vertices=navmesh["vertices"],
        triangles=navmesh["triangles"],
        area_types=navmesh["area_types"],
        **{f"meta_{k}": v for k, v in navmesh["metadata"].items()},
    )
    print(f"Wrote {filepath}")


def write_bev_png(navmesh: dict, filepath: str, dpi: int = 150):
    """Render a bird's-eye-view image of the navmesh, color-coded by area type.

    Uses the "standard" 2D convention: x_std = UE_y (= Recast Z),
    z_std = UE_x (= Recast X).  This matches the orientation produced by
    ``visualize_navmesh_cache.py`` so both tools yield comparable images.
    """
    verts = navmesh["vertices"]
    tris = navmesh["triangles"]
    areas = navmesh["area_types"]

    if len(tris) == 0:
        print("No triangles to render.")
        return

    # Standard 2D ground plane: x_std = Recast Z, y_std = Recast X
    # (matches UE_y, UE_x used by visualize_navmesh_cache.py)
    x_std = verts[:, 2]   # Recast Z = UE y
    y_std = verts[:, 0]   # Recast X = UE x

    # Color palette per area type (matches NavAreas enum)
    area_colors = {
        0: (0.4, 0.4, 0.4),    # block     — dark gray
        1: (0.55, 0.7, 0.85),  # sidewalk  — light blue
        2: (1.0, 1.0, 0.3),    # crosswalk — yellow
        3: (0.3, 0.3, 0.3),    # road      — charcoal
        4: (0.4, 0.75, 0.3),   # grass     — green
        63: (0.85, 0.55, 0.5), # obstacle  — salmon
    }
    default_color = (0.5, 0.5, 0.5)

    # Build per-triangle RGBA array
    face_colors = np.array(
        [area_colors.get(a, default_color) for a in areas], dtype=np.float64
    )

    # Figure size proportional to map extent
    x_range = x_std.max() - x_std.min()
    y_range = y_std.max() - y_std.min()
    aspect = x_range / y_range if y_range > 0 else 1.0
    fig_h = 10
    fig_w = fig_h * aspect
    fig_w = max(4.0, min(fig_w, 30.0))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_facecolor("black")
    ax.axis('off')

    from matplotlib.collections import PolyCollection

    # (N_tri, 3, 2) in std coords: (Recast_Z, Recast_X)
    tri_verts = verts[tris][:, :, [2, 0]]
    pc = PolyCollection(
        tri_verts, facecolors=face_colors, edgecolors="none", linewidths=0
    )
    ax.add_collection(pc)
    ax.set_xlim(x_std.min(), x_std.max())
    ax.set_ylim(y_std.min(), y_std.max())
    ax.set_aspect("equal")
    ax.set_xlabel("X_std (UE_y)")
    ax.set_ylabel("Y_std (UE_x)")
    ax.set_title(filepath.rsplit("/", 1)[-1].rsplit(".", 1)[0])

    # Legend
    from matplotlib.patches import Patch
    legend_handles = []
    for area_id in sorted(np.unique(areas)):
        name = AREA_NAMES.get(area_id, f"area_{area_id}")
        color = area_colors.get(area_id, default_color)
        legend_handles.append(Patch(facecolor=color, label=name))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {filepath}")


# ── OBJ parser (for building cache from existing exports) ───────────────────


NAME_TO_AREA = {v: k for k, v in AREA_NAMES.items()}


def parse_obj(filepath: str) -> dict:
    """Parse a previously exported navmesh OBJ file back into the dict format.

    Returns dict with keys: vertices, triangles, area_types (no metadata).
    """
    verts = []
    tris = []
    areas = []
    current_area = 63   # default

    with open(filepath) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("g "):
                name = line.strip().split(" ", 1)[1]
                current_area = NAME_TO_AREA.get(name, 63)
            elif line.startswith("f "):
                parts = line.split()
                # OBJ faces are 1-indexed; handle v, v/vt, v/vt/vn formats
                idx = [int(p.split("/")[0]) - 1 for p in parts[1:4]]
                tris.append(idx)
                areas.append(current_area)

    return {
        "vertices": np.array(verts, dtype=np.float32) if verts else np.empty((0, 3), dtype=np.float32),
        "triangles": np.array(tris, dtype=np.int32) if tris else np.empty((0, 3), dtype=np.int32),
        "area_types": np.array(areas, dtype=np.int32) if areas else np.empty(0, dtype=np.int32),
        "metadata": {},
    }


# ── Navmesh cache builder ──────────────────────────────────────────────────


def _recast_to_ue(vertices: np.ndarray) -> np.ndarray:
    """Convert Recast (X-right, Y-up, Z-forward) to UE (x, y, z) metres.

    Mapping (from CARLA Navigation.cpp RecastToUnreal):
        UE x = Recast X
        UE y = Recast Z
        UE z = Recast Y   (height)
    """
    return np.stack([vertices[:, 0], vertices[:, 2], vertices[:, 1]], axis=1)


def build_navmesh_cache(navmesh: dict, output_path: str,
                        n_walkable_samples: int = 100_000):
    """Build a precomputed cache of crosswalk polygons and walkable samples.

    Extracts crosswalk polygons via connected-component analysis on the
    crosswalk triangles, then computes convex hulls.  Samples walkable
    points via area-weighted barycentric sampling from all non-block
    triangles.

    Parameters
    ----------
    navmesh : dict from ``parse_navmesh()`` or ``parse_obj()``
    output_path : path for the output ``.npz`` file
    n_walkable_samples : number of walkable surface samples

    Saved arrays
    -------------
    crosswalk_vertices_ue : (N, 3) float32 — concatenated polygon vertices
    crosswalk_offsets     : (M+1,) int32   — CSR-style polygon boundaries
    walkable_points_ue    : (K, 3) float32 — uniformly sampled walkable points

    All coordinates in UE metres (x, y, z).
    """
    from scipy.sparse import lil_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import ConvexHull
    from sklearn.cluster import DBSCAN

    vertices = navmesh["vertices"]
    triangles = navmesh["triangles"]
    area_types = navmesh["area_types"]

    # Convert all vertices to UE coordinates
    verts_ue = _recast_to_ue(vertices)

    # ── Global artifact filter ──────────────────────────────────────────
    # Recast can produce triangles that bridge between different walkable
    # surfaces (e.g. ground-level ↔ overpass in Town03).  These corrupt
    # polygon extraction, BEV rendering, and area-weighted sampling.
    #
    # Two kinds of artifact are detected:
    #   (a) Multi-level + long edge: vertices span a large height range
    #       (≥ Z_THRESH) AND have a long edge (> EDGE_MIN).  This catches
    #       triangles connecting ground and overpass geometry while keeping
    #       legitimate short-edge slopes, ramps, and elevated walkways
    #       (e.g. Town04 hillside, Town10HD elevated surfaces).
    #   (b) Map-spanning: very long edges (> EDGE_HARD_MAX) regardless of
    #       height.  Both endpoints may be at the same elevation on
    #       opposite sides of the map.
    Z_THRESH = 2.0          # metres — height span within one triangle
    EDGE_MIN = 50.0         # metres — min edge to be considered artifact
    EDGE_HARD_MAX = 200.0   # metres — unconditional edge-length cap

    v0 = verts_ue[triangles[:, 0]]
    v1 = verts_ue[triangles[:, 1]]
    v2 = verts_ue[triangles[:, 2]]
    max_edges = np.maximum(
        np.linalg.norm(v1 - v0, axis=1),
        np.maximum(
            np.linalg.norm(v2 - v1, axis=1),
            np.linalg.norm(v0 - v2, axis=1),
        ),
    )
    z_all = np.stack([v0[:, 2], v1[:, 2], v2[:, 2]], axis=1)
    z_range = z_all.max(axis=1) - z_all.min(axis=1)

    artifact_mask = (
        ((z_range >= Z_THRESH) & (max_edges > EDGE_MIN))   # multi-level
        | (max_edges > EDGE_HARD_MAX)                       # map-spanning
    )
    n_artifact = int(artifact_mask.sum())
    if n_artifact:
        n_multi = int(((z_range >= Z_THRESH) & (max_edges > EDGE_MIN)).sum())
        n_span = int((artifact_mask & ~((z_range >= Z_THRESH) & (max_edges > EDGE_MIN))).sum())
        print(f"  [navmesh] filtered {n_artifact} artifact triangles "
              f"({n_multi} multi-level, {n_span} map-spanning)")
    triangles = triangles[~artifact_mask]
    area_types = area_types[~artifact_mask]

    # ── Crosswalk polygon extraction ────────────────────────────────────
    #
    # Two issues are addressed here (giant triangles already removed above):
    #
    # 1. Merged crosswalks at intersections — At 4-way intersections the
    #    navmesh triangulation may create small "bridging" crosswalk
    #    triangles at corners that share edges with perpendicular
    #    crosswalks, merging all of them into a single connected
    #    component.  After edge-connectivity grouping, components whose
    #    oriented bounding box has a low aspect ratio (roughly square)
    #    and a large area are split into individual crosswalks via
    #    DBSCAN on triangle centroids.
    #
    # 2. Convex hull inflation — The old convex-hull approach included
    #    vertices from bridging triangles at intersection corners,
    #    bloating polygons beyond the actual crosswalk area.  Now we
    #    extract the outer boundary of each triangle group (edges
    #    belonging to exactly one triangle, chained into a closed loop)
    #    which tightly follows the actual crosswalk shape.

    cw_tris = triangles[area_types == 2]

    # Tunables
    DBSCAN_EPS = 2.0             # metres — centroid proximity for same crosswalk
    DBSCAN_MIN_SAMPLES = 2       # minimum triangles per crosswalk
    SPLIT_ASPECT_THRESH = 2.5    # split components with aspect ratio below this
    SPLIT_AREA_THRESH = 100.0    # … and bbox area above this (m²)
    MIN_POLY_AREA = 0.5          # m² — drop degenerate slivers

    cw_verts_list = []
    cw_offsets = [0]

    if len(cw_tris) > 0:
        # ── Step 1: Edge-based connected components ──────────────────────
        edge_to_tri = {}
        for i, tri in enumerate(cw_tris):
            for j in range(3):
                edge = tuple(sorted((int(tri[j]), int(tri[(j + 1) % 3]))))
                edge_to_tri.setdefault(edge, []).append(i)

        n_cw = len(cw_tris)
        adj = lil_matrix((n_cw, n_cw), dtype=bool)
        for tris_sharing in edge_to_tri.values():
            for a in tris_sharing:
                for b in tris_sharing:
                    if a != b:
                        adj[a, b] = True

        n_comp, labels = connected_components(adj.tocsr(), directed=False)

        def _boundary_polygon(tri_subset):
            """Extract the outer boundary of a triangle mesh as a closed polygon.

            Boundary edges (shared by exactly one triangle) are chained into
            closed loops.  Returns the longest *closed* loop as an (N+1, 3)
            polygon in UE coordinates.  Falls back to convex hull if no
            closed loop can be found (e.g. at T-junctions where the greedy
            traversal gets stranded).
            """
            if len(tri_subset) < 1:
                return None

            # Count undirected edge occurrences
            edge_count: dict = {}
            for tri in tri_subset:
                for j in range(3):
                    e = tuple(sorted((int(tri[j]), int(tri[(j + 1) % 3]))))
                    edge_count[e] = edge_count.get(e, 0) + 1

            # Boundary edges: shared by exactly 1 triangle
            boundary_adj: dict = {}
            for (a, b), cnt in edge_count.items():
                if cnt == 1:
                    boundary_adj.setdefault(a, set()).add(b)
                    boundary_adj.setdefault(b, set()).add(a)

            if not boundary_adj:
                return _hull_polygon(tri_subset)

            # Chain boundary edges into closed loops.
            # Only keep loops that actually close (last == first).
            visited: set = set()
            best_closed_loop: list = []

            for start in sorted(boundary_adj.keys()):
                if start in visited:
                    continue
                loop = [start]
                visited.add(start)
                current = start
                while True:
                    neighbors = boundary_adj[current] - visited
                    if not neighbors:
                        # Close the loop if possible
                        if start in boundary_adj.get(current, set()):
                            loop.append(start)
                        break
                    # At T-junctions (>1 unvisited neighbor), pick the
                    # neighbor that continues most smoothly (smallest turn).
                    if len(neighbors) == 1:
                        nxt = next(iter(neighbors))
                    else:
                        p_curr = verts_ue[current, :2]
                        p_prev = verts_ue[loop[-2], :2] if len(loop) >= 2 \
                            else p_curr + np.array([1.0, 0.0])
                        dir_in = p_curr - p_prev
                        dir_in_len = np.linalg.norm(dir_in)
                        if dir_in_len > 0:
                            dir_in /= dir_in_len
                        best_ang = -np.inf
                        nxt = next(iter(neighbors))
                        for n in neighbors:
                            d = verts_ue[n, :2] - p_curr
                            d_len = np.linalg.norm(d)
                            if d_len > 0:
                                d /= d_len
                            # Signed angle (prefer left turns → outer boundary)
                            cross = dir_in[0] * d[1] - dir_in[1] * d[0]
                            dot = dir_in[0] * d[0] + dir_in[1] * d[1]
                            ang = np.arctan2(cross, dot)
                            if ang > best_ang:
                                best_ang = ang
                                nxt = n
                    visited.add(nxt)
                    loop.append(nxt)
                    current = nxt

                # Only accept closed loops (last vertex == start)
                is_closed = len(loop) >= 4 and loop[-1] == loop[0]
                if is_closed and len(loop) > len(best_closed_loop):
                    best_closed_loop = loop

            if len(best_closed_loop) < 4:
                # No closed loop found — fall back to convex hull
                return _hull_polygon(tri_subset)

            vert_ids_loop = np.array(best_closed_loop)
            pts = verts_ue[vert_ids_loop].copy()
            all_vert_ids = np.unique(tri_subset.ravel())
            mean_z = float(verts_ue[all_vert_ids, 2].mean())
            pts[:, 2] = mean_z
            return pts.astype(np.float32)

        def _hull_polygon(tri_subset):
            """Convex-hull fallback for boundary extraction failures.

            Returns (N+1, 3) float32 closed polygon in UE, or None.
            """
            vert_ids = np.unique(tri_subset.ravel())
            pts_ue = verts_ue[vert_ids]
            pts_2d = pts_ue[:, :2]
            if len(pts_2d) < 3:
                return None
            try:
                hull = ConvexHull(pts_2d)
            except Exception:
                return None
            hull_pts = pts_ue[hull.vertices]
            mean_z = float(pts_ue[:, 2].mean())
            hull_pts = np.column_stack([
                hull_pts[:, 0], hull_pts[:, 1],
                np.full(len(hull_pts), mean_z),
            ])
            return np.vstack([hull_pts, hull_pts[:1]]).astype(np.float32)

        def _accept_polygon(poly):
            """Append *poly* to the output lists if it has sufficient area."""
            if poly is None:
                return
            xy = poly[:-1, :2]  # exclude closing duplicate
            # Shoelace area
            x, y = xy[:, 0], xy[:, 1]
            area = 0.5 * abs(
                float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
                + float(x[-1] * y[0] - x[0] * y[-1])
            )
            if area < MIN_POLY_AREA:
                return
            cw_verts_list.append(poly)
            cw_offsets.append(cw_offsets[-1] + len(poly))

        # ── Step 3: Process each component, splitting merged ones ────────
        n_split = 0
        for comp_id in range(n_comp):
            comp_mask = labels == comp_id
            comp_tris = cw_tris[comp_mask]

            # Check whether this component needs splitting
            vert_ids = np.unique(comp_tris.ravel())
            pts_2d = verts_ue[vert_ids, :2]
            ext = pts_2d.max(axis=0) - pts_2d.min(axis=0)
            bbox_area = ext[0] * ext[1]
            aspect = max(ext) / (min(ext) + 1e-6)

            needs_split = (aspect < SPLIT_ASPECT_THRESH
                           and bbox_area > SPLIT_AREA_THRESH)

            if needs_split:
                # DBSCAN on triangle centroids to find sub-crosswalks
                centroids = np.array([
                    verts_ue[tri].mean(axis=0)[:2] for tri in comp_tris
                ])
                db = DBSCAN(eps=DBSCAN_EPS,
                            min_samples=DBSCAN_MIN_SAMPLES).fit(centroids)
                cluster_ids = set(db.labels_) - {-1}

                if len(cluster_ids) >= 2:
                    n_split += 1
                    for cl_id in sorted(cluster_ids):
                        cl_tris = comp_tris[db.labels_ == cl_id]
                        _accept_polygon(_boundary_polygon(cl_tris))
                    continue  # skip default single-polygon path

            # Default: single polygon for this component
            _accept_polygon(_boundary_polygon(comp_tris))

        if n_split:
            print(f"  [crosswalk] split {n_split} merged intersection "
                  f"components via DBSCAN (eps={DBSCAN_EPS}m)")

    crosswalk_vertices = (np.concatenate(cw_verts_list, axis=0)
                          if cw_verts_list
                          else np.empty((0, 3), dtype=np.float32))
    crosswalk_offsets = np.array(cw_offsets, dtype=np.int32)

    # ── Walkable area sampling ──────────────────────────────────────────
    walkable_mask = area_types != 0       # everything except block
    walk_tris = triangles[walkable_mask]

    def _sample_from_tris(tris, n_samples):
        """Area-weighted barycentric sampling from triangles."""
        if len(tris) == 0:
            return np.empty((0, 3), dtype=np.float32)
        v0 = verts_ue[tris[:, 0]]
        v1 = verts_ue[tris[:, 1]]
        v2 = verts_ue[tris[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        tri_areas = 0.5 * np.linalg.norm(cross, axis=1)
        probs = tri_areas / tri_areas.sum()
        chosen = np.random.choice(len(tris), size=n_samples,
                                  p=probs, replace=True)
        r1 = np.sqrt(np.random.rand(n_samples, 1))
        r2 = np.random.rand(n_samples, 1)
        a, b, c = 1 - r1, r1 * (1 - r2), r1 * r2
        pts = (a * verts_ue[tris[chosen, 0]] +
               b * verts_ue[tris[chosen, 1]] +
               c * verts_ue[tris[chosen, 2]])
        return pts.astype(np.float32)

    walkable_points = _sample_from_tris(walk_tris, n_walkable_samples)

    # ── Sidewalk-only sampling ─────────────────────────────────────────
    sidewalk_mask = area_types == 1
    sw_tris = triangles[sidewalk_mask]
    sidewalk_points = _sample_from_tris(sw_tris, n_walkable_samples)

    # ── Area triangle meshes for BEV segmentation rendering ────────────
    # Store triangle vertices in standard 2D coords: (x_std, z_std) =
    # (ue_y, ue_x).  Each array is (N_tris, 3, 2) float32.
    # Giant artifact triangles were already removed from `triangles` and
    # `area_types` by the global filter above.
    def _tris_to_std_2d(mask):
        sel = triangles[mask]
        if len(sel) == 0:
            return np.empty((0, 3, 2), dtype=np.float32)
        v = verts_ue[sel]           # (N, 3_verts, 3_xyz_ue)
        # standard: x_std = ue_y, z_std = ue_x
        return np.stack([v[:, :, 1], v[:, :, 0]], axis=-1).astype(np.float32)

    sidewalk_tris_std = _tris_to_std_2d(area_types == 1)
    crosswalk_tris_std = _tris_to_std_2d(area_types == 2)
    road_tris_std = _tris_to_std_2d(area_types == 3)
    walkable_tris_std = _tris_to_std_2d(area_types != 0)

    print(f"  Mesh tris: {len(sidewalk_tris_std)} sidewalk, "
          f"{len(crosswalk_tris_std)} crosswalk, "
          f"{len(road_tris_std)} road, "
          f"{len(walkable_tris_std)} walkable (total)")

    # ── Save ────────────────────────────────────────────────────────────
    np.savez_compressed(
        output_path,
        crosswalk_vertices_ue=crosswalk_vertices,
        crosswalk_offsets=crosswalk_offsets,
        walkable_points_ue=walkable_points,
        sidewalk_points_ue=sidewalk_points,
        sidewalk_tris_std=sidewalk_tris_std,
        crosswalk_tris_std=crosswalk_tris_std,
        road_tris_std=road_tris_std,
        walkable_tris_std=walkable_tris_std,
    )
    n_polys = len(crosswalk_offsets) - 1
    print(f"  Cache: {n_polys} crosswalk polygons, "
          f"{len(walkable_points)} walkable samples, "
          f"{len(sidewalk_points)} sidewalk-only samples")
    print(f"  Wrote {output_path}")


def list_maps(carla_root: str):
    """List available navmesh .bin files."""
    nav_dir = Path(carla_root) / "CarlaUE4" / "Content" / "Carla" / "Maps" / "Nav"
    if not nav_dir.exists():
        print(f"Nav directory not found: {nav_dir}")
        return
    bins = sorted(nav_dir.glob("*.bin"))
    print(f"Available navmesh files in {nav_dir}:\n")
    for b in bins:
        size_kb = b.stat().st_size / 1024
        print(f"  {b.stem:20s}  ({size_kb:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Export CARLA Recast/Detour navmesh (.bin) to OBJ or NumPy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--map", type=str, help="Map name (e.g. Town01, Town03_Opt)")
    parser.add_argument(
        "--carla", type=str, default="/raid/robot/Carla",
        help="Path to CARLA root directory (default: /raid/robot/Carla)",
    )
    parser.add_argument("--output", "-o", type=str, help="Output file path")
    parser.add_argument(
        "--format", choices=["obj", "npz"], default="obj",
        help="Output format (default: obj)",
    )
    parser.add_argument(
        "--bev", action="store_true",
        help="Save a bird's-eye-view PNG image of the navmesh",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="DPI for BEV image (default: 150)",
    )
    parser.add_argument("--list", action="store_true", help="List available maps")
    parser.add_argument(
        "--cache", action="store_true",
        help="Build a navmesh cache (.npz) with crosswalk polygons and "
             "walkable-area samples for use by the RL training pipeline.",
    )
    parser.add_argument(
        "--cache-samples", type=int, default=100_000,
        help="Number of walkable-area point samples in the cache (default: 100k).",
    )
    parser.add_argument(
        "--from-obj", type=str, default=None, metavar="OBJ",
        help="Build cache from an existing OBJ file instead of a .bin file. "
             "Implies --cache.",
    )
    parser.add_argument(
        "--cache-all", type=str, default=None, metavar="DIR",
        help="Batch mode: build caches for every *_navmesh.obj in DIR.",
    )
    args = parser.parse_args()

    if args.list:
        list_maps(args.carla)
        return

    # ── Batch cache mode: process all OBJ files in a directory ──────────
    if args.cache_all:
        obj_dir = Path(args.cache_all)
        obj_files = sorted(obj_dir.glob("*_navmesh.obj"))
        if not obj_files:
            print(f"No *_navmesh.obj files found in {obj_dir}")
            sys.exit(1)
        for obj_path in obj_files:
            town = obj_path.stem.replace("_navmesh", "")
            cache_path = str(obj_path.with_name(f"{town}_navmesh_cache.npz"))
            print(f"\n{town}: parsing {obj_path.name} ...")
            nm = parse_obj(str(obj_path))
            print(f"  {len(nm['vertices'])} vertices, "
                  f"{len(nm['triangles'])} triangles")
            build_navmesh_cache(nm, cache_path,
                                n_walkable_samples=args.cache_samples)
        return

    # ── Single-file cache from OBJ ──────────────────────────────────────
    if args.from_obj:
        obj_path = Path(args.from_obj)
        if not obj_path.exists():
            print(f"Error: {obj_path} not found.")
            sys.exit(1)
        town = obj_path.stem.replace("_navmesh", "")
        cache_path = str(obj_path.with_name(f"{town}_navmesh_cache.npz"))
        print(f"Parsing {obj_path} ...")
        nm = parse_obj(str(obj_path))
        print(f"  {len(nm['vertices'])} vertices, "
              f"{len(nm['triangles'])} triangles")
        build_navmesh_cache(nm, cache_path,
                            n_walkable_samples=args.cache_samples)
        return

    # ── Standard .bin export flow ───────────────────────────────────────
    if not args.map:
        parser.error("--map is required (or use --list / --from-obj / --cache-all)")

    nav_dir = Path(args.carla) / "CarlaUE4" / "Content" / "Carla" / "Maps" / "Nav"
    bin_path = nav_dir / f"{args.map}.bin"
    if not bin_path.exists():
        print(f"Error: {bin_path} not found.")
        print("Available maps:")
        list_maps(args.carla)
        sys.exit(1)

    print(f"Parsing {bin_path} ...")
    navmesh = parse_navmesh(str(bin_path))

    meta = navmesh["metadata"]
    print(f"  Tiles: {meta['num_tiles']}")
    print(f"  Vertices: {len(navmesh['vertices'])}")
    print(f"  Triangles: {len(navmesh['triangles'])}")

    # Area breakdown
    areas = navmesh["area_types"]
    for area_id in np.unique(areas):
        name = AREA_NAMES.get(area_id, f"area_{area_id}")
        count = np.sum(areas == area_id)
        print(f"  [{name}]: {count} triangles")

    # Output
    if args.output:
        out_path = args.output
    else:
        out_path = f"{args.map}_navmesh.{args.format}"

    if args.format == "obj":
        write_obj(navmesh, out_path)
    else:
        write_npz(navmesh, out_path)

    if args.bev:
        bev_path = args.output.rsplit(".", 1)[0] + ".png" if args.output else f"{args.map}_navmesh_bev.png"
        write_bev_png(navmesh, bev_path, dpi=args.dpi)

    if args.cache:
        cache_path = f"{args.map}_navmesh_cache.npz"
        build_navmesh_cache(navmesh, cache_path,
                            n_walkable_samples=args.cache_samples)


if __name__ == "__main__":
    main()
