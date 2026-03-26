import pathlib
import random
from typing import Any, Dict, List
import carla
import json
from queue import Queue
import numpy as np
import time
import math
from scipy.spatial.transform import Rotation as R


SpawnActor = carla.command.SpawnActor


# for UE -> standard (right-handed) coordinate system conversion
# See https://carla.readthedocs.io/en/latest/coordinates/
UE = np.array([
    [0., 1., 0.,  0.],
    [0., 0., -1., 0.],
    [1., 0., 0.,  0.],
    [0., 0., 0.,  1.]
])


class CarlaSimEnv:
    """
    Gym-like environment wrapping Carla
    """

    def __init__(self, port=2000, fps=10, n_skips=5, n_walkers=100):
        
        self.fps = fps
        self.dt = int(1. / fps)
        self.n_skips = n_skips
        self.client = carla.Client('127.0.0.1', port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.dt
        settings.substepping = True
        settings.max_substeps_delta_time = 0.01
        settings.max_substeps = math.ceil(self.dt/0.01)     
        self.world.apply_settings(settings)

        self.n_walkers = 0

        

    def reset(self, n_walkers=100):
        
        
        self.n_walkers = n_walkers

        bp_lib = self.world.get_blueprint_library()
        walker_bps = bp_lib.filter('walker.pedestrian.*')
        controller_bp = bp_lib.find('controller.ai.walker')


        self.walkers_list: List[Dict[str, Any]] = []
        self.all_id = []         # manage all walker & controller ids
        all_actors = []     # manage all walker & controller instances

        distance_threshold = 1.
        locations = []
        spawn_points = []
        n_spawn_points = 0
        # generate spawn points
        while n_spawn_points < n_walkers + 1:
            loc = self.world.get_random_location_from_navigation()
            loc.z += 2.
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
        
        # 2. Spawn the Walker Actors
        # for spawn_point in spawn_points:
        for i in range(n_walkers):
            walker_bp = random.choice(walker_bps)
            walker = self.world.try_spawn_actor(walker_bp, spawn_points[i])
            
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

        # spawn sensors
        # TODO: queue
        self.latest_observations = {}
        '''
        converter_dict = {
        'sensor.camera.rgb': carla.ColorConverter.Raw,
        'sensor.camera.depth': carla.ColorConverter.Depth,
        'sensor.camera.semantic_segmentation': carla.ColorConverter.CityScapesPalette
            }
        '''
        
        sensor_cfg_path = pathlib.Path(__file__).parent / 'stack_omr4.json'
        with open(sensor_cfg_path, 'r') as f:
            sensor_cfg = json.load(f)

        self.c2r = {}
        self.sensor_queues = {}
        

        self.sensors = []
        sensor_dict_list = sensor_cfg['sensors']
        for sensor_dict in sensor_dict_list:
            # spawn & attach each sensor to the robot
            sensor_type = sensor_dict['type']
            sensor_bp = bp_lib.find(sensor_type)

            if sensor_type.endswith('rgb'):
                sensor_bp.set_attribute('post_process_profile','Town10HD_Opt')
            sensor_id = sensor_dict['id']
            attr_dict = sensor_dict['attributes']
            for attr, val in attr_dict.items():
                sensor_bp.set_attribute(attr, str(val))

            tf_dict = sensor_dict['spawn_point']
            tf = carla.Transform(
                carla.Location(x=tf_dict['x'], y=tf_dict['y'], z=tf_dict['z']),
                carla.Rotation(pitch=tf_dict['pitch'], roll=tf_dict['roll'], yaw=tf_dict['yaw'])
            )

            self.c2r[sensor_id] = np.array(tf.get_matrix())
            q = Queue()
            self.sensor_queues[sensor_id] = q

            sensor = self.world.spawn_actor(sensor_bp, tf, attach_to=self.robot)
            # TODO: save & return an RGB
            
            sensor.listen(lambda data, q=q: q.put(data))
            self.sensors.append(sensor)
            print(f'Spawned {sensor_id}')

        c2w_7d = self.camera2world(sensor_id)
        self.xz = np.array([c2w_7d[0], c2w_7d[2]])

        self.world.tick()
        
        return self._get_observation('flcam_rgb')


    def step(self, action):
        """
        action: tuple of direction & speed
        """
        assert action.size == 4
        x, y, z, speed = action
        speed *= 100.
        norm = (x ** 2 + y ** 2 + z ** 2) ** .5 + 1e-10
        x, y, z = x / norm, y / norm, z / norm      # normalize the direction vector
        direction = carla.Vector3D(x, y, z)
        ctrl = carla.WalkerControl(direction, speed)
        self.robot.apply_control(ctrl)
        self.world.tick()
        
        return self._get_observation('flcam_rgb')


    def _get_observation(self, sensor_id):
        """
        return a pose (camera-to-world) & an RGB image of a sensor (specified by sensor_id) 
        """
        yaw = self.robot.get_transform().rotation.yaw     # 2d orientation of the robot (in the world frame)
        c2w_7d = self.camera2world(sensor_id)

        # TODO: ray casting to measure ground height

        speed = self.robot.get_velocity().length()
        xz = np.array([c2w_7d[0], c2w_7d[2]])

        delta_xz = xz - self.xz     # change in xz coordinates (in the global frame)
        self.xz = np.copy(xz)


        return {'pose': c2w_7d, 'rgb': self._get_data(sensor_id), 'yaw': yaw, 'speed': speed, 'delta_xz': delta_xz}

    def close(self):
        print("\n Cleaning the episode...")
        # destroy all sensors
        for s in self.sensors:
            s.stop()
            s.destroy()
        # stop controllers -> destroy all actors & controllers 
        for walker_dict in self.walkers_list:
            controller = self.world.get_actor(walker_dict["con"])
            controller.stop()
        
        # destroy
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])        
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = None
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        return
    
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
        


    def _get_data(self, sensor_id):
        # RGB
        data = self.sensor_queues[sensor_id].get()
        img = np.frombuffer(data.raw_data, dtype=np.uint8)
        img = img.reshape((data.height, data.width, 4))[..., :3]
        img = img[..., [2, 1, 0]]
        return img
