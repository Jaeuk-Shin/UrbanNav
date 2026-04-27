"""
Minimal single-agent CARLA environment for NoMaD evaluation.

Shares sensor rig + coordinate conventions with ``rl/envs/carla_multi.py``
(the PPO training env) but drops the precomputed-scenario machinery so we
can inject an arbitrary user-supplied ``(start, goal)`` transform.

Public API (intentionally lean):
    env = NomadCarlaEnv(port, sensor_cfg_path, fps=5, n_skips=1,
                        teleport=False, max_speed=1.4)
    env.connect()
    env.reset(start_ue=(x, y, z), goal_ue=(x, y, z))
    img_pil, info = env.render_observation()
    env.step_walker(vx_cam, vz_cam)   # or env.step_teleport(...)
    env.close()
"""

from __future__ import annotations

import json
import math
import pathlib
import time
from queue import Empty, Queue
from typing import Optional, Tuple, List

import numpy as np
import carla
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation as R

from carla_utils.tf import UE


# ── Helpers (adapted from rl/envs/carla_multi.py) ────────────────────────
def _to_walker_control(vx: float, vz: float,
                       c2w: np.ndarray) -> carla.WalkerControl:
    """Camera-frame velocity → CARLA WalkerControl (3D direction + speed)."""
    direction_cam = np.array([vx, 0.0, vz])
    c2w_R = R.from_quat(c2w[3:])
    direction = c2w_R.apply(direction_cam)
    x, y, z = direction
    # standard -> UE
    x, y, z = z, x, -y
    norm = math.sqrt(x * x + y * y + z * z) + 1e-10
    return carla.WalkerControl(
        carla.Vector3D(float(x / norm), float(y / norm), float(z / norm)),
        float(math.sqrt(vx * vx + vz * vz)),
    )


def ue_xy_to_std(ue_x: float, ue_y: float) -> np.ndarray:
    """UE (x, y) → standard (x_std, z_std) = (UE_y, UE_x)."""
    return np.array([ue_y, ue_x], dtype=np.float64)


def std_to_ue_xy(std_xz: np.ndarray) -> Tuple[float, float]:
    """Standard (x_std, z_std) → UE (x, y)."""
    return float(std_xz[1]), float(std_xz[0])


# ── Env ──────────────────────────────────────────────────────────────────
class NomadCarlaEnv:
    """Single-walker CARLA env wired for NoMaD topomap evaluation."""

    def __init__(
        self,
        port: int = 2000,
        fps: int = 5,
        n_skips: int = 1,
        sensor_cfg_path: Optional[str] = None,
        teleport: bool = False,
        max_speed: float = 1.4,
        success_radius: float = 2.0,
    ):
        self.port = port
        self.fps = fps
        self.dt = 1.0 / fps
        self.n_skips = int(n_skips)
        self.teleport = teleport
        self.max_speed = max_speed
        self.success_radius = success_radius

        # Default: reuse the same sensor rig as CarlaMultiAgentEnv
        if sensor_cfg_path is None:
            sensor_cfg_path = str(
                pathlib.Path(__file__).resolve().parent.parent
                / 'rl' / 'envs' / 'assets' / 'stack_omr4.json')
        with open(sensor_cfg_path, 'r') as f:
            cfg = json.load(f)
        # The file wraps sensors in a top-level dict; use the first fwd cam.
        self._sensor_specs: List[dict] = cfg['sensors']

        # CARLA state (populated in connect/reset)
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.bp_lib = None
        self.robot = None
        self.sensors: List = []
        self.sensor_queues: dict = {}
        self.c2r: dict = {}

        # Goal in UE (user-supplied) and standard coords
        self.goal_ue: Optional[np.ndarray] = None     # (3,)
        self.goal_std: Optional[np.ndarray] = None    # (2,)

    # ── Connection / shutdown ─────────────────────────────────────────
    def connect(self, timeout: float = 30.0):
        self.client = carla.Client('127.0.0.1', self.port)
        self.client.set_timeout(timeout)
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.dt
        settings.max_substeps = min(16, math.ceil(self.dt / 0.01))
        settings.max_substep_delta_time = (
            self.dt / settings.max_substeps + 1e-5)
        self.world.apply_settings(settings)
        self.bp_lib = self.world.get_blueprint_library()

    def close(self):
        for s in self.sensors:
            try:
                s.stop()
                s.destroy()
            except Exception:
                pass
        self.sensors = []
        self.sensor_queues = {}
        if self.robot is not None:
            try:
                self.robot.destroy()
            except Exception:
                pass
            self.robot = None
        if self.world is not None:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
            except Exception:
                pass
        self.client = None
        self.world = None

    # ── Actor setup ───────────────────────────────────────────────────
    def _spawn_walker(self, start_ue: Tuple[float, float, float],
                      start_yaw_deg: float = 0.0):
        walker_bps = self.bp_lib.filter('walker.pedestrian.*')
        spawn_tf = carla.Transform(
            carla.Location(x=float(start_ue[0]),
                           y=float(start_ue[1]),
                           z=float(start_ue[2]) + 2.0),
            carla.Rotation(yaw=float(start_yaw_deg)),
        )

        robot = None
        # Try a few random blueprints — not every point is walkable.
        for _ in range(10):
            bp = walker_bps[np.random.randint(len(walker_bps))]
            robot = self.world.try_spawn_actor(bp, spawn_tf)
            if robot is not None:
                break
        if robot is None:
            raise RuntimeError(
                f"Failed to spawn walker at UE {start_ue}")
        self.robot = robot

    def _spawn_sensors(self):
        self.sensors = []
        self.sensor_queues = {}
        self.c2r = {}
        for spec in self._sensor_specs:
            bp = self.bp_lib.find(spec['type'])
            for k, v in spec['attributes'].items():
                bp.set_attribute(k, str(v))
            tf_dict = spec['spawn_point']
            tf = carla.Transform(
                carla.Location(
                    x=tf_dict['x'], y=tf_dict['y'], z=tf_dict['z']),
                carla.Rotation(
                    pitch=tf_dict['pitch'],
                    roll=tf_dict['roll'],
                    yaw=tf_dict['yaw']),
            )
            self.c2r[spec['id']] = np.array(tf.get_matrix())
            q = Queue()
            self.sensor_queues[spec['id']] = q
            sensor = self.world.spawn_actor(bp, tf, attach_to=self.robot)
            sensor.listen(lambda data, q=q: q.put(data))
            self.sensors.append(sensor)

    # ── Public reset ───────────────────────────────────────────────────
    def reset(
        self,
        start_ue: Tuple[float, float, float],
        goal_ue: Tuple[float, float, float],
        start_yaw_deg: float = 0.0,
    ):
        """Spawn walker at ``start_ue`` and register ``goal_ue`` as target.

        Both coordinates are UE (CARLA native: Location.x, Location.y, z).
        """
        assert self.world is not None, "call connect() first"
        # Clean up any prior episode
        for s in self.sensors:
            try:
                s.stop(); s.destroy()
            except Exception:
                pass
        self.sensors = []
        self.sensor_queues = {}
        if self.robot is not None:
            try:
                self.robot.destroy()
            except Exception:
                pass
            self.robot = None

        self.goal_ue = np.asarray(goal_ue, dtype=np.float64)
        self.goal_std = ue_xy_to_std(float(goal_ue[0]), float(goal_ue[1]))

        self._spawn_walker(start_ue, start_yaw_deg=start_yaw_deg)
        self._spawn_sensors()

        # Two ticks: first to register sensors, second to get a settled frame.
        self.world.tick()
        self._drain_queues()
        self.world.tick()

    def _drain_queues(self):
        """Discard any pending sensor data (used before reading fresh frames)."""
        for q in self.sensor_queues.values():
            while True:
                try:
                    q.get_nowait()
                except Empty:
                    break

    # ── Observation ───────────────────────────────────────────────────
    def _get_sensor_data(self, sensor_id='fcam') -> np.ndarray:
        """Return the latest camera frame, discarding any stale frames
        that piled up in the queue (e.g. from multi-tick physics steps).
        """
        q = self.sensor_queues[sensor_id]
        try:
            data = q.get(timeout=10.0)
        except Empty:
            self.world.tick()
            data = q.get(timeout=10.0)
        # Drain any additional frames and keep only the newest
        while True:
            try:
                data = q.get_nowait()
            except Empty:
                break
        img = np.frombuffer(data.raw_data, dtype=np.uint8)
        img = img.reshape((data.height, data.width, 4))[..., :3]
        return img[..., [2, 1, 0]]     # BGR→RGB

    def render_observation(self, sensor_id='fcam'
                           ) -> Tuple[PILImage.Image, dict]:
        """Read the latest camera frame as a PIL image + pose/goal info."""
        rgb = self._get_sensor_data(sensor_id)
        pil = PILImage.fromarray(rgb)
        info = {
            'xz_std': self.get_xz_std(),
            'pose_std': self.camera2world(sensor_id),
            'goal_std': self.goal_std.copy() if self.goal_std is not None else None,
            'distance_to_goal': self.distance_to_goal(),
        }
        return pil, info

    # ── Stepping ──────────────────────────────────────────────────────
    def step_walker(self, vx_cam: float, vz_cam: float, sensor_id='fcam'):
        """Apply one step of WalkerControl at ``(vx, vz)`` camera-frame m/s.

        Executes ``n_skips`` ticks with the same control, matching the
        convention of ``CarlaMultiAgentEnv._step_physics_all``.
        """
        ctrl = _to_walker_control(vx_cam, vz_cam,
                                  self.camera2world(sensor_id))
        for _ in range(self.n_skips):
            self.robot.apply_control(ctrl)
            self.world.tick()

    def step_teleport(self, vx_cam: float, vz_cam: float, sensor_id='fcam'):
        """Teleport-mode step — mirrors ``_step_teleport_all``."""
        pose = self.camera2world(sensor_id)
        disp_cam = np.array([vx_cam * self.dt, 0.0, vz_cam * self.dt])
        disp_std = R.from_quat(pose[3:]).apply(disp_cam)
        dx_ue, dy_ue = float(disp_std[2]), float(disp_std[0])
        for _ in range(self.n_skips):
            tf = self.robot.get_transform()
            tf.location.x += dx_ue
            tf.location.y += dy_ue
            speed_2d = math.sqrt(dx_ue * dx_ue + dy_ue * dy_ue)
            if speed_2d > 1e-6:
                tf.rotation.yaw = math.degrees(math.atan2(dy_ue, dx_ue))
            self.robot.set_transform(tf)
            self.world.tick()

    def step_camera_waypoint(self, waypoint_cam: np.ndarray,
                             sensor_id='fcam'):
        """Convenience: apply a single ``(dx, dz)`` camera-frame waypoint
        as constant velocity for one step.  Used by the runner.
        """
        dx, dz = float(waypoint_cam[0]), float(waypoint_cam[1])
        step_dur = self.dt * self.n_skips
        vx = dx / step_dur
        vz = dz / step_dur
        speed = math.sqrt(vx * vx + vz * vz)
        if speed > self.max_speed:
            k = self.max_speed / speed
            vx *= k
            vz *= k
        if self.teleport:
            self.step_teleport(vx, vz, sensor_id)
        else:
            self.step_walker(vx, vz, sensor_id)

    # ── Coordinate helpers (copied from carla_multi.py) ──────────────
    def camera2world(self, sensor_id='fcam') -> np.ndarray:
        r2w = np.array(self.robot.get_transform().get_matrix())
        c2w = r2w @ self.c2r[sensor_id]
        c2w = UE @ c2w @ UE.T
        xyz = c2w[:3, -1]
        q = R.from_matrix(c2w[:3, :3]).as_quat()
        return np.concatenate((xyz, q))

    def get_xz_std(self, sensor_id='fcam') -> np.ndarray:
        p = self.camera2world(sensor_id)
        return np.array([p[0], p[2]])

    def distance_to_goal(self, sensor_id='fcam') -> float:
        if self.goal_std is None:
            return float('inf')
        xz = self.get_xz_std(sensor_id)
        return float(np.linalg.norm(xz - self.goal_std))

    def is_success(self) -> bool:
        return self.distance_to_goal() <= self.success_radius

    # ── Teleport utility (used by topomap builder) ────────────────────
    def teleport_robot_ue(self, ue_xy: Tuple[float, float],
                          yaw_deg: float,
                          ground_z: Optional[float] = None):
        """Move the robot to an absolute UE ``(x, y)`` with a given yaw.

        Keeps the current z unless ``ground_z`` is provided.
        """
        tf = self.robot.get_transform()
        tf.location.x = float(ue_xy[0])
        tf.location.y = float(ue_xy[1])
        if ground_z is not None:
            tf.location.z = float(ground_z)
        tf.rotation.yaw = float(yaw_deg)
        self.robot.set_transform(tf)
