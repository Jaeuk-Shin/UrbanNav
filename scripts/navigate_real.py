#!/usr/bin/env python3
"""
Long-range real-world navigation using OSM + learned local policy.

Global planner : OpenStreetMap walkable network (via osmnx)
Local policy   : PPO-trained goal-conditioned visuomotor policy

This is a reference implementation — you will need to adapt the
RobotInterface class to match your actual robot's sensors and actuators.

Usage:
    pip install osmnx networkx

    python scripts/navigate_real.py \
        --config config/rl.yaml \
        --checkpoint checkpoints/ppo_best.pt \
        --start_gps 37.7749 -122.4194 \
        --goal_gps  37.7760 -122.4180
"""

import argparse
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, Tuple

import numpy as np
import torch

from config.utils import load_config
from rl.ppo_agent import PPOAgent
from rl.navigation import OSMGlobalPlanner, HierarchicalNavigator


# ─── Robot Interface (adapt to your hardware) ────────────────────────


class RobotInterface(ABC):
    """
    Abstract interface for your robot. Implement the three methods below
    to connect UrbanNav's policy to your specific hardware.
    """

    @abstractmethod
    def get_observation(self) -> Dict[str, np.ndarray]:
        """
        Return the observation dict expected by the policy:
            obs  : (H, height, width, 3) uint8 — recent RGB frames
            cord : (H*2,) float32 — recent [x, z] positions flattened
        """
        ...

    @abstractmethod
    def get_pose(self) -> np.ndarray:
        """
        Return 7-D pose [x, y, z, qx, qy, qz, qw] in a local metric
        frame consistent with the route waypoints.

        For OSMGlobalPlanner, route waypoints are in local ENU
        (east, north) metres relative to center_gps. Your localisation
        must output poses in the same frame:
            x = east, y = up, z = north

        In practice, use the first GPS reading as origin and compute
        east/north offsets via the same equirectangular projection that
        OSMGlobalPlanner.gps_to_enu() uses.
        """
        ...

    @abstractmethod
    def execute(self, action: np.ndarray):
        """
        Execute the policy's action output.

        If use_decoder=True, action is (10,) — five 2D waypoints in
        camera frame that should be tracked by your low-level controller.
        If use_decoder=False, action is (2,) — velocity [vx, vz] in
        camera frame.
        """
        ...


# ─── Example: USB Camera + GPS/IMU Robot ─────────────────────────────


class SimpleGPSRobot(RobotInterface):
    """
    Minimal example using a USB camera and a GPS+IMU module.

    Replace the sensor-reading stubs with your actual driver calls.
    """

    def __init__(self, cfg, center_gps: Tuple[float, float]):
        self.context_size = cfg.model.obs_encoder.context_size
        self.width = cfg.data.width
        self.height = cfg.data.height

        self._rgb_buf = deque(maxlen=self.context_size)
        self._pose_buf = deque(maxlen=self.context_size)

        # GPS → ENU conversion (same projection as OSMGlobalPlanner)
        self._lat0, self._lon0 = center_gps
        self._cos_lat0 = np.cos(np.radians(self._lat0))

    def _gps_to_enu(self, lat, lon):
        R_EARTH = 6_371_000.0
        east = R_EARTH * np.radians(lon - self._lon0) * self._cos_lat0
        north = R_EARTH * np.radians(lat - self._lat0)
        return east, north

    def _read_camera(self) -> np.ndarray:
        """Return (height, width, 3) uint8 RGB frame."""
        # TODO: replace with your camera driver
        # e.g., ret, frame = cv2.VideoCapture(0).read()
        #       frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #       frame = cv2.resize(frame, (self.width, self.height))
        raise NotImplementedError("Implement camera capture")

    def _read_gps_imu(self) -> Tuple[float, float, float]:
        """Return (latitude, longitude, heading_radians)."""
        # TODO: replace with your GPS/IMU driver
        raise NotImplementedError("Implement GPS/IMU reading")

    def get_observation(self) -> Dict[str, np.ndarray]:
        rgb = self._read_camera()
        self._rgb_buf.append(rgb)

        lat, lon, heading = self._read_gps_imu()
        east, north = self._gps_to_enu(lat, lon)

        # Build 7-D pose: [x=east, y=0, z=north, qx, qy, qz, qw]
        # Heading → quaternion (rotation about y-axis in standard coords)
        from scipy.spatial.transform import Rotation as R
        quat = R.from_euler("y", -heading).as_quat()  # [qx,qy,qz,qw]
        pose = np.array([east, 0.0, north, *quat])
        self._pose_buf.append(pose)

        # Pad if not enough history yet
        while len(self._rgb_buf) < self.context_size:
            self._rgb_buf.appendleft(self._rgb_buf[0])
            self._pose_buf.appendleft(self._pose_buf[0])

        return {
            "obs": np.array(list(self._rgb_buf), dtype=np.uint8),
            "cord": np.array(list(self._pose_buf))[:, [0, 2]]
                      .flatten().astype(np.float32),
        }

    def get_pose(self) -> np.ndarray:
        return np.copy(self._pose_buf[-1])

    def execute(self, action: np.ndarray):
        # TODO: convert waypoints/velocity to your robot's control API
        # For a differential-drive robot, the first waypoint gives
        # a heading direction in camera frame; use a PID or pure-pursuit
        # controller on your motor driver.
        raise NotImplementedError("Implement motor control")


# ─── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--start_gps", nargs=2, type=float, required=True,
                        help="Start GPS: lat lon")
    parser.add_argument("--goal_gps", nargs=2, type=float, required=True,
                        help="Goal GPS: lat lon")
    parser.add_argument("--radius", type=float, default=2000,
                        help="OSM download radius (metres)")
    parser.add_argument("--lookahead", type=float, default=6.0)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    cfg = load_config(args.config)

    start_gps = tuple(args.start_gps)
    goal_gps = tuple(args.goal_gps)
    center_gps = start_gps  # use start as ENU origin

    # ── Agent ─────────────────────────────────────────────────────────
    agent = PPOAgent(cfg, use_decoder=True).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()

    # ── Global Planner ────────────────────────────────────────────────
    planner = OSMGlobalPlanner(
        center_gps=center_gps,
        network_type="walk",
        radius=args.radius,
    )

    # ── Robot ─────────────────────────────────────────────────────────
    robot = SimpleGPSRobot(cfg, center_gps=center_gps)

    # ── Navigator ─────────────────────────────────────────────────────
    navigator = HierarchicalNavigator(
        agent=agent,
        global_planner=planner,
        lookahead_dist=args.lookahead,
        goal_reached_dist=2.0,
        final_goal_dist=1.5,
        device=device,
    )
    navigator.set_destination(start_gps, goal_gps)

    # ── Navigation Loop ──────────────────────────────────────────────
    for step_i in range(args.max_steps):
        obs = robot.get_observation()
        pose = robot.get_pose()

        action = navigator.step(obs, pose)
        robot.execute(action)

        if step_i % 20 == 0:
            print(
                f"  step {step_i:4d} | "
                f"progress {navigator.progress:.0%} | "
                f"remaining ~{navigator.remaining_distance:.1f} m"
            )

        if navigator.arrived:
            print(f"\n  Arrived in {step_i + 1} steps!")
            break


if __name__ == "__main__":
    main()
