/**
 * pybind11 bindings for the Detour navmesh pathfinder.
 *
 * Build:
 *   cd detour_nav && mkdir build && cd build
 *   cmake .. -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
 *   make -j$(nproc)
 *   cp detour_nav*.so ..
 *
 * Usage:
 *   import detour_nav
 *   pf = detour_nav.DetourPathfinder()
 *   pf.load("navmeshes/Town02.bin")
 *   pf.set_sidewalk_only()
 *   d = pf.geodesic_distance(100.0, 0.0, 200.0, 150.0, 0.0, 250.0)
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "detour_pathfinder.h"

namespace py = pybind11;

PYBIND11_MODULE(detour_nav, m) {
    m.doc() = "Minimal Detour navmesh pathfinding wrapper for CARLA .bin files";

    // Expose CARLA flag constants
    m.attr("CARLA_TYPE_NONE")      = CARLA_TYPE_NONE;
    m.attr("CARLA_TYPE_SIDEWALK")  = CARLA_TYPE_SIDEWALK;
    m.attr("CARLA_TYPE_CROSSWALK") = CARLA_TYPE_CROSSWALK;
    m.attr("CARLA_TYPE_ROAD")      = CARLA_TYPE_ROAD;
    m.attr("CARLA_TYPE_GRASS")     = CARLA_TYPE_GRASS;
    m.attr("POLY_FLAG_BLOCKED")    = POLY_FLAG_BLOCKED;

    py::class_<DetourPathfinder>(m, "DetourPathfinder",
        "Load a CARLA .bin navmesh and query geodesic distances.\n\n"
        "Coordinates are in Recast frame (X-right, Y-up, Z-forward).\n"
        "Conversion from UE: recast = (ue_x, ue_z, ue_y).")
        .def(py::init<>())

        // Loading
        .def("load", &DetourPathfinder::load,
             py::arg("path"),
             "Load a CARLA .bin navmesh file. Returns True on success.")
        .def("is_loaded", &DetourPathfinder::is_loaded)

        // Filter configuration
        .def("set_include_flags", &DetourPathfinder::set_include_flags,
             py::arg("flags"),
             "Set which polygon flag bits to include in queries.")
        .def("set_exclude_flags", &DetourPathfinder::set_exclude_flags,
             py::arg("flags"),
             "Set which polygon flag bits to exclude from queries.")
        .def("set_area_cost", &DetourPathfinder::set_area_cost,
             py::arg("area"), py::arg("cost"),
             "Set traversal cost multiplier for an area type.")
        .def("set_sidewalk_only", &DetourPathfinder::set_sidewalk_only,
             "Restrict paths to sidewalk + crosswalk polygons only.")
        .def("set_all_areas", &DetourPathfinder::set_all_areas,
             "Allow all walkable area types (default).")

        // Obstacle blocking
        .def("block_polygons_in_aabb",
             &DetourPathfinder::block_polygons_in_aabb,
             py::arg("cx"), py::arg("cy"), py::arg("cz"),
             py::arg("half_ex"), py::arg("half_ey"), py::arg("half_ez"),
             "Block all polygons whose centres fall within an AABB.\n"
             "Returns the number of polygons blocked.")
        .def("unblock_all", &DetourPathfinder::unblock_all,
             "Clear POLY_FLAG_BLOCKED from all polygons.")

        // Pathfinding
        .def("find_path", &DetourPathfinder::find_path,
             py::arg("sx"), py::arg("sy"), py::arg("sz"),
             py::arg("ex"), py::arg("ey"), py::arg("ez"),
             "Find shortest path. Returns (geodesic_distance, [(x,y,z)...]).\n"
             "Returns (inf, []) if no path exists.")
        .def("geodesic_distance", &DetourPathfinder::geodesic_distance,
             py::arg("sx"), py::arg("sy"), py::arg("sz"),
             py::arg("ex"), py::arg("ey"), py::arg("ez"),
             "Geodesic distance between two points. Returns inf if unreachable.")
        .def("get_random_point", &DetourPathfinder::get_random_point,
             "Sample a random navigable point on the navmesh.");
}
