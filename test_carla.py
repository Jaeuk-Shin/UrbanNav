import argparse
import numpy as np
import os
from carla_env import CarlaSimEnv
from flow_matching_policy import TrajectorySampler
from mpc import MPC
from PIL import Image
from scipy.spatial.transform import Rotation as R
import time

from omegaconf import OmegaConf
from vis_utils import visualize


def to_walker_control(velocities, c2w, dt):
    """
    CARLA walker-specific adapter
    velocities: linear & angular velocity
    convert the velocity commands to the inputs of carla.WalkerControl
    """
    # TODO: must check if this is correct!
    lin_v, ang_v = velocities
    th_next = .5 * np.pi + dt * ang_v
    c, s = np.cos(th_next), np.sin(th_next)
    # direction_cam = np.array([c, 0., s])   # direction of e_3 after the angular velocity is applied (in the camera frame)            
    direction = np.array([c, 0., s])
    # c2w_R = R.from_quat(c2w[3:])
    # direction = c2w_R.apply(direction_cam)  # direction vector (in the world frame)
    x, y, z = direction
    # TODO: check if the direction must be given in the walker coordinate (it seems so)
    direction = np.array([z, x, -y])        # to UE coordinate system
    # direction = np.array([1., 0., 0.])      # for sanity check
    return np.append(direction, np.abs(lin_v))


def score_waypoints(waypoints, goal):
    """
    score each sequence of waypoints
    """
    distances = np.sum((waypoints[..., -1, :] - goal) ** 2, axis=-1) ** .5      # final waypoints
    return np.argmin(distances)


def repeat_and_shift(data, repeats, shifts):
    # waypoints: array of shape (*, data size, data dim)
    # repeat each waypoints (along the temporal axis)
    # (*, data size, data dim) -> (*, repeats x data size, data dim)
    n_waypoints = data.shape[-2]
    data_rep = np.repeat(data, repeats=repeats, axis=-2)
    data_shifted = np.concatenate((data_rep[..., shifts:, :], data_rep[..., repeats*n_waypoints-shifts:, :]), axis=-2)

    return data_shifted


def eval_policy_in_simulation(cfg, ckpt):
    cfg = OmegaConf.load(args.config)

    # from cfg
    future_length = cfg.model.decoder.len_traj_pred
    history_length = cfg.model.obs_encoder.context_size

    local_policy = TrajectorySampler(cfg, ckpt, future_length=future_length, history_length=history_length, step_scale=1.)

    # TODO: make these as variables (or manage these as a config file)
    dt_policy = .5
    
    dt = .1
    
    repeats = int(dt_policy / dt)      # num. of control steps between local planning

    # MPC's planning horizon
    ts = future_length * repeats
    
    max_speed = 1.4
    ulb = np.array([-max_speed, -0.8])
    uub = np.array([max_speed, 0.8])

    # maximum budget for an MPC
    max_wall_time = 0.02
    mpc = MPC(ts, dt, ulb, uub, max_wall_time=max_wall_time)

    xmax = ts * dt * max_speed
    xmin = -xmax

    zmax = xmax
    zmin = -zmax

    x_bounds = (xmin, xmax)
    z_bounds = (zmin, zmax)


    goal = np.array([5., 5.*np.tan(np.pi/3.)])      # 10m in front of the ego; moving goal

    env = CarlaSimEnv(port=2000, fps=int(1./dt), n_walkers=150)

    local_policy.reset()
    o = env.reset()
    
    # os.makedirs('test_carla', exist_ok=True)
    os.makedirs('test_carla_vis', exist_ok=True)

    speed_commands = []
    speeds = []
    velocity_commands = []
    t_walls_mpc = []
    t_walls_policy = []
    t_walls_eval = []

    try:
        for t in range(300):
            print(f'[step {t}]', end=' ')
            t_wall_total = time.perf_counter()

            rgb_arr = o['rgb']
            img = Image.fromarray(rgb_arr)

            # img.save(f'test_carla/{t}.png')
            # TODO: visualize the following:
            # goal
            t_inner = t % repeats

            if t_inner == 0:
                t_wall_policy = time.perf_counter()
                actions, policy_info = local_policy.sample_actions(o['rgb'], o['pose'], n_samples=100)
                t_wall_policy = time.perf_counter() - t_wall_policy

                t_wall_eval = time.perf_counter()
                idx = score_waypoints(actions, goal)
                t_wall_eval = time.perf_counter() - t_wall_eval
            
            # waypoints given in the camera frame
            waypoints = actions[idx]                            # shape: (# of waypoints, 2)
            # cost weights of the MPC cost function
            cost_weights = np.ones(waypoints.shape[0])
            cost_weights[0] = 10.        # prioritize reaching the first waypoint
            processed_waypoints = repeat_and_shift(data=waypoints, repeats=repeats, shifts=t_inner)
            cost_weights = np.squeeze(repeat_and_shift(data=cost_weights[:, None], repeats=repeats, shifts=t_inner), axis=-1)
            x, u, mpc_stats = mpc.solve(
                initial_pose=np.array([0., 0., .5*np.pi]),      # (x, z, th)
                waypoints=processed_waypoints,
                cost_weights=cost_weights
                )       
            
            print(f'lin. vel.={u[0, 0]:.4f}m/s / ang. vel.={u[0, 1]:.4f}rad/s', end=' ')
            # differential drive
            action = to_walker_control(u[0], c2w=o['pose'], dt=dt)

            '''
            if t < history_length * repeats:
                # move forward during the first 2.5 seconds, and then start inference
                # will be removed once we fix the problem
                action = np.array([1., 0., 0., max_speed])
            '''
            speed_commands.append(action[-1])
            speeds.append(o['speed'])
            velocity_commands.append(u[0])
            t_walls_mpc.append(mpc_stats['t_wall'])
            t_walls_policy.append(t_wall_policy)
            t_walls_eval.append(t_wall_eval)

            visualize(
                timestamp=t,
                input_positions=policy_info['input_positions'],
                img=img,
                waypoints=actions,
                best_sample_idx=idx,
                waypoints_y=o['pose'][1],
                waypoints_z=processed_waypoints[:, 1],
                speeds=speeds,
                speed_commands=speed_commands,
                velocity_commands=velocity_commands,
                t_walls_mpc=t_walls_mpc,
                t_walls_policy=t_walls_policy,
                t_walls_eval=t_walls_eval,
                max_wall_time=max_wall_time,
                max_speed=max_speed,
                x_bounds=x_bounds,
                z_bounds=z_bounds,
                n_past_steps=history_length*repeats,
                n_future_steps=ts,
                mpc_res=(x, u),
                dir_to_save='test_carla_vis'
            )

            o = env.step(action)
            actions -= o['delta_xz']        # adjust the waypoints according to the translation of the camera frame
            print(f'/ wall time={time.perf_counter()-t_wall_total}s')


    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
        raise
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test UrbanNav model')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint. If not provided, the latest checkpoint will be used.')

    args = parser.parse_args()

    eval_policy_in_simulation(cfg=args.config, ckpt=args.checkpoint)