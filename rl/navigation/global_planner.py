"""
Global path planners for long-range navigation.

Provides a unified interface for computing routes in both
CARLA simulation and real-world environments using OpenStreetMap.

    # CARLA — uses the built-in road network A* planner
    planner = CarlaGlobalPlanner(carla_world)
    route = planner.plan(start_xz, goal_xz)

    # Real-world — uses OSM walkable network + Dijkstra
    planner = OSMGlobalPlanner(center_gps=(37.77, -122.42))
    route = planner.plan(start_gps, goal_gps)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ─── Route ────────────────────────────────────────────────────────────


@dataclass
class Route:
    """A planned route as a sequence of 2D waypoints in local metric coords."""

    waypoints: np.ndarray  # (N, 2)
    total_length: float = 0.0

    def __post_init__(self):
        if len(self.waypoints) > 1:
            diffs = np.diff(self.waypoints, axis=0)
            self.total_length = float(np.sum(np.linalg.norm(diffs, axis=1)))

    def __len__(self):
        return len(self.waypoints)

    def __getitem__(self, idx):
        return self.waypoints[idx]


# ─── Abstract Planner ─────────────────────────────────────────────────


class GlobalPlanner(ABC):
    """Abstract base class for global route planners."""

    @abstractmethod
    def plan(self, start, goal) -> Route:
        """Compute a route from start to goal."""
        ...


# ─── CARLA Planner ────────────────────────────────────────────────────


class CarlaGlobalPlanner(GlobalPlanner):
    """
    Global planner for CARLA using the built-in road network A* planner.

    Input/output coordinates are in the UrbanNav *standard* frame:
        x_std = UE_Y  (rightward)
        z_std = UE_X  (forward)
    This is the same frame used by ``CarlaEnv.xz``.

    Requirements:
        The CARLA PythonAPI ``agents`` package must be importable.
        It ships with the CARLA pip wheel (>= 0.9.13).
    """

    def __init__(self, world, resolution: float = 2.0):
        """
        Args:
            world: ``carla.World`` instance.
            resolution: Spacing (meters) between route waypoints.
        """
        import carla
        from agents.navigation.global_route_planner import GlobalRoutePlanner

        self._carla = carla
        self._map = world.get_map()
        self._grp = GlobalRoutePlanner(self._map, resolution)

    # ── coordinate helpers ────────────────────────────────────────────

    def _std_to_carla_loc(self, xz: np.ndarray):
        """Standard (x_std, z_std) → carla.Location."""
        return self._carla.Location(
            x=float(xz[1]), y=float(xz[0]), z=0.0
        )

    @staticmethod
    def _carla_loc_to_std(loc) -> np.ndarray:
        """carla.Location → (x_std, z_std)."""
        return np.array([loc.y, loc.x])

    # ── planning ──────────────────────────────────────────────────────

    def plan(self, start_xz: np.ndarray, goal_xz: np.ndarray) -> Route:
        """
        Plan a route between two points in standard coordinates.

        Args:
            start_xz: (2,) array [x_std, z_std]
            goal_xz:  (2,) array [x_std, z_std]
        """
        start_loc = self._std_to_carla_loc(start_xz)
        goal_loc = self._std_to_carla_loc(goal_xz)

        # Snap to nearest road waypoints
        start_wp = self._map.get_waypoint(
            start_loc, project_to_road=True
        )
        goal_wp = self._map.get_waypoint(
            goal_loc, project_to_road=True
        )

        # A* on road topology
        route_wps = self._grp.trace_route(
            start_wp.transform.location,
            goal_wp.transform.location,
        )

        waypoints = np.array(
            [self._carla_loc_to_std(wp.transform.location)
             for wp, _ in route_wps],
            dtype=np.float64,
        )
        return Route(waypoints)


# ─── OSM Planner ──────────────────────────────────────────────────────


class OSMGlobalPlanner(GlobalPlanner):
    """
    Global planner using OpenStreetMap for real-world pedestrian routing.

    Downloads the walkable road network around a center point and
    computes shortest paths via Dijkstra (weighted by edge length).

    Input:  GPS coordinates ``(lat, lon)``
    Output: Waypoints in local East-North metres relative to *center_gps*.

    Dependencies: ``pip install osmnx networkx``
    """

    def __init__(
        self,
        center_gps: Tuple[float, float],
        network_type: str = "walk",
        radius: float = 2000,
    ):
        """
        Args:
            center_gps: (lat, lon) center of the area to download.
            network_type: OSM network type ('walk', 'bike', 'drive', 'all').
            radius: Download radius in metres.
        """
        import osmnx as ox
        import networkx as nx

        self._ox = ox
        self._nx = nx
        self._lat0, self._lon0 = center_gps
        self._cos_lat0 = np.cos(np.radians(self._lat0))

        print(f"Downloading OSM '{network_type}' network "
              f"around ({self._lat0:.5f}, {self._lon0:.5f}) …")
        self.G = ox.graph_from_point(
            center_gps, dist=radius, network_type=network_type,
        )
        # Keep an unprojected copy for lat/lon lookups
        self._G_latlon = self.G.copy()
        # Project to local UTM for metric edge weights
        self.G = ox.project_graph(self.G)
        print(f"  {self.G.number_of_nodes()} nodes, "
              f"{self.G.number_of_edges()} edges")

    # ── coordinate helpers ────────────────────────────────────────────

    def gps_to_enu(self, lat: float, lon: float) -> np.ndarray:
        """
        GPS (lat, lon) → local East-North metres.

        Uses equirectangular approximation (accurate within ~50 km
        of the center).
        """
        R_EARTH = 6_371_000.0
        east = R_EARTH * np.radians(lon - self._lon0) * self._cos_lat0
        north = R_EARTH * np.radians(lat - self._lat0)
        return np.array([east, north])

    def enu_to_gps(self, east: float, north: float) -> Tuple[float, float]:
        """Local East-North metres → GPS (lat, lon)."""
        R_EARTH = 6_371_000.0
        lat = self._lat0 + np.degrees(north / R_EARTH)
        lon = self._lon0 + np.degrees(east / (R_EARTH * self._cos_lat0))
        return (lat, lon)

    # ── planning ──────────────────────────────────────────────────────

    def plan(
        self,
        start_gps: Tuple[float, float],
        goal_gps: Tuple[float, float],
    ) -> Route:
        """
        Plan a walking route between two GPS coordinates.

        Args:
            start_gps: (lat, lon)
            goal_gps:  (lat, lon)

        Returns:
            Route with waypoints in local ENU metres (east, north).
        """
        start_node = self._ox.nearest_nodes(
            self._G_latlon, X=start_gps[1], Y=start_gps[0],
        )
        goal_node = self._ox.nearest_nodes(
            self._G_latlon, X=goal_gps[1], Y=goal_gps[0],
        )

        path_nodes = self._nx.shortest_path(
            self.G, start_node, goal_node, weight="length",
        )

        waypoints = []
        for node in path_nodes:
            data = self._G_latlon.nodes[node]
            enu = self.gps_to_enu(data["y"], data["x"])  # y=lat, x=lon
            waypoints.append(enu)

        return Route(np.array(waypoints, dtype=np.float64))
