"""Hardcoded CARLA blueprint bounding-box half-extents.

Provides the same information as ``actor.bounding_box.extent`` without
requiring a running CARLA server.  Values are reasonable approximations;
run ``scripts/measure_blueprints.py`` (spawn each actor, read
``actor.bounding_box.extent``, print results) to calibrate for your
CARLA build.

Each entry maps blueprint ID -> (half_extent_x, half_extent_y) in metres,
matching the actor-local frame used by ``actor.bounding_box.extent``.
The local bounding-box offset (``bb.location``) is assumed to be (0, 0)
for all blueprints used in obstacle generation.
"""

from typing import Dict, Tuple

# (half_extent_x, half_extent_y) in local actor frame.
# x = forward/backward, y = left/right for vehicles.
BLUEPRINT_HALF_EXTENTS: Dict[str, Tuple[float, float]] = {
    # -- Blocker vehicles --
    'vehicle.carlamotors.firetruck':    (4.5, 1.2),
    'vehicle.tesla.cybertruck':         (3.0, 1.1),
    'vehicle.carlamotors.european_hgv': (5.0, 1.3),
    'vehicle.ford.ambulance':           (3.2, 1.0),
    'vehicle.volkswagen.t2_2021':       (2.4, 0.9),
    'vehicle.mercedes.sprinter':        (3.5, 1.0),
    # -- Barrier props --
    'static.prop.streetbarrier':        (0.6, 0.15),
    'static.prop.constructioncone':     (0.2, 0.2),
    'static.prop.trafficcone01':        (0.15, 0.15),
    'static.prop.trafficcone02':        (0.15, 0.15),
    # -- Clutter props --
    'static.prop.trashcan01':           (0.3, 0.3),
    'static.prop.trashcan03':           (0.3, 0.3),
    'static.prop.trashcan04':           (0.3, 0.3),
    'static.prop.trashcan05':           (0.3, 0.3),
    'static.prop.bench01':              (0.8, 0.3),
    'static.prop.bench02':              (0.8, 0.3),
    'static.prop.bench03':              (0.8, 0.3),
    'static.prop.table':                (0.5, 0.5),
    'static.prop.shoppingcart':         (0.5, 0.3),
}

DEFAULT_HALF_EXTENTS: Tuple[float, float] = (1.0, 0.5)


def get_half_extents(blueprint_id: str) -> Tuple[float, float]:
    """Return (half_extent_x, half_extent_y) for a blueprint.

    Falls back to DEFAULT_HALF_EXTENTS if the blueprint is not in the
    lookup table.
    """
    return BLUEPRINT_HALF_EXTENTS.get(blueprint_id, DEFAULT_HALF_EXTENTS)
