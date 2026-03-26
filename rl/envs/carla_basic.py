
import time
import math
import pathlib
import random
import argparse
from collections import deque
from typing import List
from queue import Queue
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import torchvision.transforms.functional as TF
import gymnasium as gym

import carla
from rl.envs.mpc.mpc import MPC


from carla_utils.tf import UE

SpawnActor = carla.command.SpawnActor

'''
def to_polar(xy):
    """
    (x, y) -> (r, cos(theta), sin(theta))
    """
    assert xy.shape[-1] == 2
    x, y = xy[..., 0], xy[..., 1]
    r = (x ** 2 + y ** 2) ** .5
    c, s = x / r, y / r
    return np.stack((r, c, s), axis=-1)
'''

class CarlaBasicEnv(gym.Env):
    """
    gym wrapper of CARLA
    """

    def __init__(self, cfg, port=2000, max_speed=1.4, fps=5, gamma=0.99,
                 teleport=False, towns=None, randomize_weather=False):

        self.fps = fps          # carla simulation speed
        self.dt = 1. / fps
        self.n_skips = int(fps // cfg.data.target_fps)      # ratio between high-level & low level
        self.port = port

        self.towns = towns          # e.g. ['Town02', 'Town03', 'Town05', 'Town10']
        self._current_town = None
        self.randomize_weather = randomize_weather
        self._current_weather = None       # set by _randomize_weather()

        self.teleport = teleport
        self.max_speed = max_speed
        self.gamma = gamma
        self.history_length = cfg.model.obs_encoder.context_size
        self.future_length = cfg.model.decoder.len_traj_pred

        # continuous action space; set of reachable waypoints
        self.action_space = gym.spaces.Box(low=-1.4, high=1.4, shape=(2,), dtype=np.float32)

        width = cfg.data.width
        height = cfg.data.height

        self.observation_space = gym.spaces.Dict(
            {
                'obs': gym.spaces.Box(low=0., high=255., shape=(self.history_length, height, width, 3), dtype=np.uint8),
                'cord': gym.spaces.Box(low=-100., high=100., shape=(self.history_length*2,), dtype=np.float32),
                'goal': gym.spaces.Box(low=-100., high=100., shape=(2,), dtype=np.float32)
                }
        )
        '''
        # MPC's planning horizon
        mpc_horizon = self.future_length * self.n_skips
        ulb = np.array([-max_speed, -max_speed])
        uub = np.array([max_speed, max_speed])
        # maximum budget for an MPC
        max_wall_time = 2. * self.dt
        self.mpc = MPC(mpc_horizon, self.dt, ulb, uub, max_wall_time=max_wall_time)
        '''
        # buffers; stack historical inputs here
        self.rgb_buffer = deque(maxlen=self.history_length)  # past H images
        self.pose_buffer = deque(maxlen=self.history_length)         # past H poses

        self._collect_substep_frames = False
        self._substep_frames = []

        self.client = None
        return

    def set_collect_substep_frames(self, enabled: bool):
        self._collect_substep_frames = enabled

    def _destroy_stale_actors(self):
        """Remove any leftover actors from a previous failed session."""
        stale_prefixes = ('walker.', 'sensor.', 'controller.')
        stale = [
            a for a in self.world.get_actors()
            if any(a.type_id.startswith(p) for p in stale_prefixes)
        ]
        if stale:
            print(f"  Destroying {len(stale)} stale actor(s) on port {self.port}")
            self.client.apply_batch(
                [carla.command.DestroyActor(a.id) for a in stale]
            )

    def _randomize_weather(self):
        """Sample random weather and sun position for the current episode."""
        if not self.randomize_weather:
            return
        params = dict(
            cloudiness=float(np.random.uniform(0, 90)),
            precipitation=float(np.random.uniform(0, 80)),
            precipitation_deposits=float(np.random.uniform(0, 80)),
            wind_intensity=float(np.random.uniform(0, 100)),
            sun_azimuth_angle=float(np.random.uniform(0, 360)),
            sun_altitude_angle=float(np.random.uniform(-15, 90)),
            fog_density=float(np.random.uniform(0, 40)),
            fog_distance=float(np.random.uniform(0, 100)),
            wetness=float(np.random.uniform(0, 80)),
        )
        self._current_weather = params
        self.world.set_weather(carla.WeatherParameters(**params))

    def reset(self, *, seed=None, options=None):

        num_walkers = (options or {}).get('num_walkers', 0)
        self.close()    # clean up the previous eposide if exists

        self.client = carla.Client('127.0.0.1', self.port)
        self.client.set_timeout(30.0)

        # Load the assigned town (single-element list from vectorized wrapper)
        if self.towns:
            town = self.towns[0]
            if town != self._current_town:
                print(f"  Loading map: {town} (port {self.port})")
                self.client.set_timeout(300.0)
                self.client.load_world(town)
                self.client.set_timeout(30.0)
                self._current_town = town

        # unique TM port per server to avoid conflicts in multi-server setups
        self.tm = self.client.get_trafficmanager(self.port + 6000)
        self.tm.set_synchronous_mode(True)
        self.world = self.client.get_world()

        # Safety net: destroy any actors orphaned by a previous failed reset
        self._destroy_stale_actors()

        settings = self.world.get_settings()
        settings.synchronous_mode = True

        # fixed_delta_seconds <= max_substep_delta_time * max_substeps
        # https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/
        settings.fixed_delta_seconds = self.dt
        # max_substeps must be in [1, 16]; adjust substep delta to compensate
        settings.max_substeps = min(16, math.ceil(self.dt / 0.01))
        # add small offset to suppress warning
        settings.max_substep_delta_time = self.dt / settings.max_substeps + 1e-5
        self.world.apply_settings(settings)
        self._randomize_weather()

        self.bp_lib = self.world.get_blueprint_library()
        self.n_walkers = 0

        self.start_scenario(num_walkers=num_walkers)
        self.initialize_buffer()
        self.sample_goal(range=8.)
        self.initial_distance = float(self.distance_to_goal)
        self.path_length = 0.0
        return self._get_observation(), self._get_info()

    def sample_goal(self, range):
        # x, z = self.xz    # camera-to-world (standard coordinate system)
        r_min, r_max = .8 * range, range
        th = np.pi * (2. * np.random.rand() - 1.)
        p = np.random.rand()
        r = (r_max ** 2 * (1. - p) + r_min ** 2 * p) ** .5

        # global coordinates of the goal
        self.goal_global = self.xz + r * np.array([np.cos(th), np.sin(th)])
        return
    
    @property
    def goal(self):
        # project the goal onto the xz-plane of the camera frame
        goal_world = self.goal_global - self.xz
        c2w_R = R.from_quat(self.pose[3:])
        goal_cam = c2w_R.inv().apply(np.array([goal_world[0], 0., goal_world[1]]))
        return np.array([goal_cam[0], goal_cam[2]])

    @property
    def distance_to_goal(self):
        return (self.goal ** 2).sum() ** .5

    def potential_function(self):
        return -self.distance_to_goal


    def step(self, action: np.ndarray):
        """
        action: tuple of direction & speed
        """
        assert isinstance(action, np.ndarray) and action.shape == (2,)
        self._substep_frames = []
        phi0 = self.potential_function()
        pre_step_xz = self.xz.copy()

        dx, dz = action                     # desired delta
        vx, vz = dx / (self.dt * self.n_skips), dz / (self.dt * self.n_skips)   # desired velocity (in the xz-plane)

        # Snapshot pose at step start for commanded velocity computation
        pose0 = self.pose
        cmd_vel_cam = np.array([vx, 0., vz])
        c2w_R = R.from_quat(pose0[3:])
        cmd_vel_world = c2w_R.apply(cmd_vel_cam)
        cmd_vel_xz = np.array([cmd_vel_world[0], cmd_vel_world[2]], dtype=np.float32)
        cmd_speed = float(np.sqrt(vx**2 + vz**2))

        if self.teleport:
            self._step_teleport(vx, vz)
        else:
            self._step_physics(vx, vz)

        # Compute real velocity from actual position change (reliable for both modes)
        post_step_xz = self.xz
        real_disp = post_step_xz - pre_step_xz
        step_duration = self.dt * self.n_skips
        real_vel_xz = (real_disp / step_duration).astype(np.float32)
        real_speed = float(np.linalg.norm(real_vel_xz))

        self.path_length += float(np.linalg.norm(real_disp))

        terminated = False
        truncated = False

        # reward function
        # see [Wijmans et al., 2020, ICLR] & [Chattopadhyay et al., 2021, CVPR]
        # potential-based reward shaping: mere binary reward followed by shaping terms
        phi1 = self.potential_function()
        reward = 2.5 if self.distance_to_goal <= .2 else 0.
        # normalization
        reward += (self.gamma * phi1 - phi0) / (self.gamma * self.n_skips * self.dt * self.max_speed) - .01

        info = self._get_info()
        info.update({
            'cmd_vel_xz': cmd_vel_xz,
            'cmd_speed': cmd_speed,
            'real_vel_xz': real_vel_xz,
            'real_speed': real_speed,
        })
        if self._collect_substep_frames and self._substep_frames:
            info['substep_frames'] = np.stack(self._substep_frames)

        return self._get_observation(), reward, terminated, truncated, info

    def _step_physics(self, vx, vz):
        """Original WalkerControl-based stepping."""
        # Compute the world-frame direction ONCE from the pose at step start.
        # Re-reading self.pose each substep would cause compounding rotation
        # because WalkerControl turns the walker (and attached camera) to face
        # the movement direction, shifting the camera frame each tick.
        walker_control = to_walker_control(vx, vz, c2w=self.pose)
        for t in range(self.n_skips):
            self.robot.apply_control(walker_control)
            self.world.tick()
            self.update_buffer()
            if self._collect_substep_frames:
                self._substep_frames.append(self.rgb_buffer[-1].copy())

    def _step_teleport(self, vx, vz):
        """Direct set_transform stepping — no physics lag."""
        # Compute world-frame displacement ONCE from the pose at step start
        # (same rationale as _step_physics — avoid compounding rotation).
        disp_cam = np.array([vx * self.dt, 0., vz * self.dt])
        c2w_R = R.from_quat(self.pose[3:])
        disp_std = c2w_R.apply(disp_cam)

        # Standard → UE: UE_x = z_std, UE_y = x_std, UE_z = -y_std
        dx_ue = float(disp_std[2])
        dy_ue = float(disp_std[0])

        for t in range(self.n_skips):
            tf = self.robot.get_transform()
            tf.location.x += dx_ue
            tf.location.y += dy_ue
            # Keep z unchanged to stay on the ground plane

            # Face the movement direction so the attached camera rotates correctly
            speed_2d = math.sqrt(dx_ue ** 2 + dy_ue ** 2)
            if speed_2d > 1e-6:
                tf.rotation.yaw = math.degrees(math.atan2(dy_ue, dx_ue))

            self.robot.set_transform(tf)
            self.world.tick()
            self.update_buffer()
            if self._collect_substep_frames:
                self._substep_frames.append(self.rgb_buffer[-1].copy())
    
    def close(self):
        if self.client is None:
            return

        print("\n Cleaning the episode...")
        # destroy all sensors (may not exist if reset() failed mid-init)
        for s in getattr(self, 'sensors', []):
            try:
                s.stop()
                s.destroy()
            except Exception:
                pass
        # stop controllers -> destroy all actors & controllers
        for walker_dict in getattr(self, 'walkers_list', []):
            try:
                controller = self.world.get_actor(walker_dict["con"])
                controller.stop()
            except Exception:
                pass

        # destroy all known actors
        all_id = getattr(self, 'all_id', [])
        if all_id:
            try:
                self.client.apply_batch(
                    [carla.command.DestroyActor(x) for x in all_id]
                )
            except Exception:
                pass

        try:
            settings = self.world.get_settings()
            settings.fixed_delta_seconds = None
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        except Exception:
            pass

        self.client = None
        return

    def start_scenario(self, num_walkers):
        self.spawn_agents(num_walkers=num_walkers)
        self.spawn_sensors()
        self.world.tick()
        return

    def spawn_agents(self, num_walkers=0):
        """
        spawn ego-robot & walkers
        """

        walker_bps = self.bp_lib.filter('walker.pedestrian.*')
        controller_bp = self.bp_lib.find('controller.ai.walker')

        self.walkers_list = []
        self.all_id = []            # manage all walker & controller ids
        all_actors = []             # manage all walker & controller instances

        spawn_points = self.sample_spawn_points(num_walkers+1)

        # spawn the Walker Actors
        # for spawn_point in spawn_points:
        for i in range(num_walkers): 
            walker = self.world.try_spawn_actor(random.choice(walker_bps), spawn_points[i])
            
            if walker:
                walker_dict = {"walker": walker}
                self.walkers_list.append(walker_dict)
            else:
                print('spawn failed')

        # spawn an ego robot; modeled as a walker here
        # TODO: use the omr4 model
        robot_bp = random.choice(walker_bps)
        self.robot = self.world.try_spawn_actor(robot_bp, spawn_points[-1])
        assert self.robot is not None
        self.robot_id = self.robot.id
        print(f'id of ego: {self.robot_id}')

        self.all_id.append(self.robot_id)

        # 3. Spawn the AI Controllers & attach them to walkers
        batch = [SpawnActor(controller_bp, carla.Transform(), walker_dict["walker"]) for walker_dict in self.walkers_list]

        responses = self.client.apply_batch_sync(batch, True)
        
        for i, response in enumerate(responses):
            if response.error:
                print(response.error)
            else:
                self.walkers_list[i]["con"] = response.actor_id

        # 4. Initialize the AI and set goals
        for walker_dict in self.walkers_list:
            # Find the controller actor by ID
            controller = self.world.get_actor(walker_dict["con"])
            
            # Start the AI
            controller.start()
            
            # Set a random destination from the navigation mesh
            goal = self.world.get_random_location_from_navigation()
            controller.go_to_location(goal)
            
            # Set walking speed (default is usually ~1.4 m/s)
            controller.set_max_speed(1. + random.random()) 
            
            self.all_id.append(walker_dict["con"])
            self.all_id.append(walker_dict["walker"].id)
            all_actors.append(walker_dict["walker"])
            all_actors.append(controller)

        print(f"Spawned {len(self.walkers_list)} pedestrians.")

    def initialize_buffer(self):
        """
        pad with the same data
        """
        rgb = self._get_data(sensor_id='fcam')
        pose = self.camera2world(sensor_id='fcam')
        for _ in range(self.history_length):
            self.pose_buffer.append(pose)
            self.rgb_buffer.append(rgb)
        return


    def sample_spawn_points(self, num, distance_threshold=1.):
        locations = []
        spawn_points = []
        n_spawn_points = 0
        # generate spawn points
        while n_spawn_points < num + 1:
            loc = self.world.get_random_location_from_navigation()
            loc.z += 2.     # to prevent collision with ground
            success = True
            for loc2 in locations:
                # ensure that the distance from existing spawn points is above the threshold
                if loc.distance(loc2) <= distance_threshold:
                    success = False
                    break
            if success:
                spawn_point = carla.Transform()
                spawn_point.location = loc 
                spawn_points.append(spawn_point)
                locations.append(loc)
                n_spawn_points += 1
        return spawn_points
    
    def spawn_sensors(self):
        '''
        attach sensors to the robot
        '''

        self.c2r = {}       # camera-to-robot (4 x 4 matrices)
        self.sensor_queues = {}
        
        sensor_cfgs = load_sensor_configs(filename='stack_omr4.json')
        self.sensors = []

        for sensor_cfg in sensor_cfgs:
            # spawn & attach each sensor to the robot
            sensor_type = sensor_cfg['type']
            sensor_bp = self.bp_lib.find(sensor_type)
            '''
            if sensor_type.endswith('rgb'):
                sensor_bp.set_attribute('post_process_profile','Town10HD_Opt')
            '''
            sensor_id = sensor_cfg['id']
            attr_dict = sensor_cfg['attributes']
            for attr, val in attr_dict.items():
                sensor_bp.set_attribute(attr, str(val))

            tf_dict = sensor_cfg['spawn_point']
            tf = carla.Transform(
                carla.Location(x=tf_dict['x'], y=tf_dict['y'], z=tf_dict['z']),
                carla.Rotation(pitch=tf_dict['pitch'], roll=tf_dict['roll'], yaw=tf_dict['yaw'])
            )

            self.c2r[sensor_id] = np.array(tf.get_matrix())
            q = Queue()
            self.sensor_queues[sensor_id] = q

            sensor = self.world.spawn_actor(sensor_bp, tf, attach_to=self.robot)
            sensor.listen(lambda data, q=q: q.put(data))

            self.sensors.append(sensor)
            print(f'Spawned {sensor_id}')
        return
    
    def update_buffer(self):
        rgb = self._get_data(sensor_id='fcam')
        pose = self.camera2world(sensor_id='fcam')
        self.pose_buffer.append(pose)
        self.rgb_buffer.append(rgb)
        return

    @property
    def pose(self):
        return np.copy(self.pose_buffer[-1])

    @property
    def xz(self):
        return np.array([self.pose[0], self.pose[2]])

    def _get_info(self):
        speed = self.robot.get_velocity().length()
        info = {
            'xz': self.xz,
            'speed': speed,
            'distance_to_goal': float(self.distance_to_goal),
            'initial_distance': getattr(self, 'initial_distance', 0.0),
            'path_length': getattr(self, 'path_length', 0.0),
            'is_success': self.distance_to_goal <= 0.2,
        }
        if self._current_weather is not None:
            info['weather'] = self._current_weather
        return info

    def _get_observation(self):
        goal = self.goal.astype(np.float32)  # camera frame
        goal_world = (self.goal_global - self.xz).astype(np.float32)  # world frame
        o = {
            'obs': np.array(self.rgb_buffer, dtype=np.uint8),
            'cord': np.array(self.pose_buffer)[:, [0, 2]].flatten().astype(np.float32),
            'goal': goal,
            'goal_world': goal_world,
        }
        return o

    def _get_data(self, sensor_id) -> np.ndarray:
        # RGB data
        data = self.sensor_queues[sensor_id].get(timeout=30.0)
        img = np.frombuffer(data.raw_data, dtype=np.uint8)
        img = img.reshape((data.height, data.width, 4))[..., :3]
        # BGR -> RGB
        return img[..., [2, 1, 0]]

    def camera2world(self, sensor_id) -> np.ndarray:
        '''
        return the 7d-repr. of the camera-to-world
        '''
        r2w = np.array(self.robot.get_transform().get_matrix())
        c2w = r2w @ self.c2r[sensor_id]
        # camera-to-world (in standard)
        c2w = UE @ c2w @ UE.T       # 4 x 4 matrix in SE(3)
        xyz = c2w[:3, -1]
        q = R.from_matrix(c2w[:3, :3]).as_quat()   # orientation as quat
        return np.concatenate((xyz, q))
        

    def capture_bev(self, altitude: float = 50.0, fov: float = 90.0,
                    img_size: int = 512) -> tuple:
        """
        Spawn a temporary downward-looking RGB camera directly above the ego,
        capture one frame, then destroy the camera.

        After the required world.tick() the operational sensor queues (flcam_rgb,
        etc.) will have accumulated one extra frame.  We drain them before
        returning so the rollout's update_buffer() calls are not affected.

        Coordinate convention
        ---------------------
        CARLA / UE: X = forward, Y = right, Z = up.
        Standard  : x = UE_Y (right), z = UE_X (forward)   [same as self.xz]

        Overhead camera at yaw=0°, pitch=-90°:
          image right (+u) ↔ standard +x  (= UE right)
          image up   (−v) ↔ standard +z  (= UE forward)

        Returns
        -------
        img  : (img_size, img_size, 3)  uint8 RGB
        meta : dict with keys
               center_xz  – camera footprint in standard (x, z)
               altitude   – metres
               fov_deg    – horizontal field-of-view in degrees
               img_size   – pixel side length
        """
        from queue import Queue as TQueue

        ego_loc = self.robot.get_transform().location

        cam_bp = self.bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(img_size))
        cam_bp.set_attribute('image_size_y', str(img_size))
        cam_bp.set_attribute('fov', str(fov))

        cam_tf = carla.Transform(
            carla.Location(x=ego_loc.x, y=ego_loc.y, z=ego_loc.z + altitude),
            carla.Rotation(pitch=-90.0, roll=0.0, yaw=0.0),
        )

        bev_q = TQueue()
        cam = self.world.spawn_actor(cam_bp, cam_tf)
        cam.listen(lambda data: bev_q.put(data))
        self.world.tick()

        data = bev_q.get(timeout=30.0)
        img = np.frombuffer(data.raw_data, dtype=np.uint8)
        img = img.reshape((img_size, img_size, 4))[..., :3][..., [2, 1, 0]]  # BGRA→RGB

        cam.stop()
        cam.destroy()

        # Drain the extra frame that the tick pushed into each operational sensor
        # queue so that subsequent update_buffer() calls stay in sync.
        for sq in self.sensor_queues.values():
            try:
                while not sq.empty():
                    sq.get_nowait()
            except Exception:
                pass

        # Use the camera (fcam) position for center_xz so it matches the
        # coordinate source of buf_cord (which stores camera2world positions).
        # camera2world returns [x_std, y_std, z_std, ...] in standard coords.
        cam_pose = self.camera2world(sensor_id='fcam')
        center_xz = np.array([cam_pose[0], cam_pose[2]], dtype=np.float32)
        meta = {
            'center_xz': center_xz,
            'altitude': float(altitude),
            'fov_deg': float(fov),
            'img_size': int(img_size),
        }
        return img, meta

    @staticmethod
    def bev_world_to_pixel(xz_world, meta) -> np.ndarray:
        """
        Project standard-coordinate (x, z) positions to BEV image pixels.

        Parameters
        ----------
        xz_world : array-like (..., 2)  – (x_std, z_std)
        meta     : dict returned by capture_bev()

        Returns
        -------
        pixels : ndarray (..., 2) float  – (u, v) in image coordinates
                 u increases rightward, v increases downward.
        """
        xz = np.asarray(xz_world, dtype=np.float64)
        h   = meta['altitude']
        s   = meta['img_size']
        ctr = meta['center_xz']
        fov_rad = np.deg2rad(meta['fov_deg'])

        # pixels per metre at ground level
        scale = s / (2.0 * h * np.tan(fov_rad / 2.0))

        dx = xz[..., 0] - ctr[0]   # standard-x offset  (right  → +u)
        dz = xz[..., 1] - ctr[1]   # standard-z offset  (forward → −v)

        u = s / 2.0 + scale * dx
        v = s / 2.0 - scale * dz
        return np.stack([u, v], axis=-1)


def load_sensor_configs(filename) -> List[dict]:
    sensor_cfg_path = pathlib.Path(__file__).parent / 'assets' / filename
    with open(sensor_cfg_path, 'r') as f:
        sensor_cfg = json.load(f)
    return sensor_cfg['sensors']


def to_walker_control(vx, vz, c2w) -> carla.WalkerControl:
    """
    CARLA walker-specific adapter
    convert actions to carla.WalkerControl
    """
    # TODO: must check if this is correct!
    direction_cam = np.array([vx, 0., vz])   # direction of e_3 after the angular velocity is applied (in the camera frame)            
    c2w_R = R.from_quat(c2w[3:])
    direction = c2w_R.apply(direction_cam)  # direction vector (in the world frame)
    x, y, z = direction
    x, y, z = z, x, -y        # to UE coordinate system
    norm = (x ** 2 + y ** 2 + z ** 2) ** .5 + 1e-10
    x, y, z = x / norm, y / norm, z / norm      # normalize the direction vector

    direction = carla.Vector3D(float(x), float(y), float(z))
    speed = float((vx ** 2 + vz ** 2) ** .5)
    ctrl = carla.WalkerControl(direction, speed)

    return ctrl


def repeat_and_shift(data, repeats, shifts):
    # waypoints: array of shape (*, data size, data dim)
    # repeat each waypoints (along the temporal axis)
    # (*, data size, data dim) -> (*, repeats x data size, data dim)
    n_waypoints = data.shape[-2]
    data_rep = np.repeat(data, repeats=repeats, axis=-2)
    data_shifted = np.concatenate((data_rep[..., shifts:, :], data_rep[..., repeats*n_waypoints-shifts:, :]), axis=-2)

    return data_shifted


def transform_poses(poses, current_pose_array):
    current_pose_matrix = pose_to_matrix(current_pose_array)
    current_pose_inv = np.linalg.inv(current_pose_matrix)
    pose_matrices = poses_to_matrices(poses)
    transformed_matrices = np.matmul(current_pose_inv[np.newaxis, :, :], pose_matrices)
    positions = transformed_matrices[:, :3, 3]
    return positions

def pose_to_matrix(pose):
    position = pose[:3]
    rotation = R.from_quat(pose[3:])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = position
    return matrix

def poses_to_matrices(poses):
    positions = poses[:, :3]
    quats = poses[:, 3:]
    rotations = R.from_quat(quats)
    matrices = np.tile(np.eye(4), (poses.shape[0], 1, 1))
    matrices[:, :3, :3] = rotations.as_matrix()
    matrices[:, :3, 3] = positions
    return matrices
