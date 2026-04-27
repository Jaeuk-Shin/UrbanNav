"""Test harness for the CityWalker visual navigation policy in CARLA.

Pipeline:
    1. Sample a global goal at a user-specified distance.
    2. Query the CARLA road-network planner for a reference path.
    3. Sparsify the path into subgoals.
    4. Feed the current subgoal to the policy; advance the subgoal when the
       agent is close enough; terminate when the final goal is reached.
"""

from .planning import (
    Route,
    sample_goal_location,
    plan_reference_path,
    sparsify_path,
    SubgoalSchedule,
)
from .policy import CityWalkerPolicy
from .env import NavTestEnv

__all__ = [
    "Route",
    "sample_goal_location",
    "plan_reference_path",
    "sparsify_path",
    "SubgoalSchedule",
    "CityWalkerPolicy",
    "NavTestEnv",
]
