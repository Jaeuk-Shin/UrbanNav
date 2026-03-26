import numpy as np    
from typing import Tuple, List
import PIL
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from data.camera_utils import build_intrinsic_matrix, get_image_coordinates


def visualize(
        timestamp: int,
        input_positions: np.ndarray,
        img: PIL.Image,
        waypoints: np.ndarray,
        best_sample_idx: int,
        waypoints_y: float,
        waypoints_z: np.ndarray,
        speeds: List[float],
        speed_commands: List[float],
        velocity_commands: List[np.ndarray],
        t_walls_mpc: List[float],
        t_walls_policy: List[float],
        t_walls_eval: List[float],
        max_wall_time: float,
        max_speed: float,
        x_bounds: Tuple[float, float],
        z_bounds: Tuple[float, float],
        n_past_steps: int,
        n_future_steps: int,
        mpc_res: Tuple[np.ndarray, np.ndarray],
        dir_to_save: str | bytes | os.PathLike
        ):

    width, height, = img.size
    # Visualization title

    fig = plt.figure(layout="constrained", figsize=(24, 6))
    gs = GridSpec(3, 12, figure=fig)

    ax1 = fig.add_subplot(gs[:, :3])    # rgb image
    ax2 = fig.add_subplot(gs[:, 3:6])   # bird's-eye view
    ax3 = fig.add_subplot(gs[0, 6:9])    # speed
    ax4 = fig.add_subplot(gs[1, 6:9])    # x, z, yaw & waypoints
    ax5 = fig.add_subplot(gs[2, 6:9])    # velocity commands
    ax6 = fig.add_subplot(gs[0, 9:])        # wall time of mpc
    ax7 = fig.add_subplot(gs[1, 9:])        # wall time of local planner
    ax8 = fig.add_subplot(gs[2, 9:])        # wall time of waypoint evaluation

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    # plt.subplots_adjust(wspace=0.3)

    # Left axis: plot the current observation (frame) with arrived info in title

    ax1.imshow(img)
    ax1.axis('off')

    # overlay on image
    # The following are sensor & data-specific; willbe refactored
    # front left camera
    # images: cropped & padded within the dataloader; must be considered during coordinate computation
    # 640 x 480 -> 640 x 360; see data/citywalk_dataset.py

    ax1.set_title('flcam_rgb', fontsize=20)
    fov = 117.89925552494842
    K = build_intrinsic_matrix(w=640, h=480, fov=fov)

    ax1.set_xlim(0., width)
    ax1.set_ylim(height, 0.)
    '''
    def shift_points(u, v):
        return u, v - .5 * (480 - 360)
    '''
    def shift_points(x_img, y_img):
        return x_img, y_img

    n_waypoints = waypoints.shape[-2]
    waypoints_ys = 0. * np.ones(n_waypoints)
    assert waypoints.ndim == 3
    
    # multiple samples drawn from the model
    for sample_idx in range(waypoints.shape[0]):
        u_pred, v_pred, valid = project_waypoints_onto_image_plane(waypoints[sample_idx], waypoints_ys, K=K)
        u_pred, v_pred = shift_points(u_pred, v_pred)
        if np.all(valid):
            if sample_idx == best_sample_idx:
                ax1.plot(u_pred, v_pred, color='tab:green')
            else:
            
                ax1.plot(u_pred, v_pred, color='#DB6057')

    x_mpc, u_mpc = mpc_res

    xzs = x_mpc[:, [0, 1]]
    ys = 0. * np.ones(x_mpc.shape[0])
    x_img_mpc, y_img_mpc, valid = project_waypoints_onto_image_plane(xzs, ys, K=K)
    x_img_mpc, y_img_mpc = shift_points(x_img_mpc, y_img_mpc)
    if np.all(valid):
        ax1.plot(x_img_mpc, y_img_mpc, color='tab:cyan')

    # Right axis: plot the coordinates

    xmin, xmax = x_bounds
    zmin, zmax = z_bounds
    ax2.axis('scaled')   
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(zmin, zmax)

    # illustrate camera fov
    th = np.pi / 2. - np.deg2rad(fov) / 2.
    r = np.linspace(0., 7., num=100)
    c, s = np.cos(th), np.sin(th)

    ax2.plot(r * c, r * s, linestyle='dashed', color='tab:gray', label='fov')
    ax2.plot(-r * c, r * s, linestyle='dashed', color='tab:gray')


    ax2.plot(input_positions[:, 0], input_positions[:, 1],
                'o-', label='past positions', color='#5771DB', zorder=2)
    
    assert waypoints.ndim == 3
    # multiple samples drawn from the model
    labeled = False
    for sample_idx in range(waypoints.shape[0]):
        if sample_idx == best_sample_idx:
            label = 'waypoints (selected)'
            color = 'tab:green'
            zorder = 2
            alpha = 1.
            linewidth = 3
        else:
            label = 'waypoints'
            color = '#DB6057'
            zorder = 1
            alpha = 0.5
            linewidth = 1

        if not labeled:                
            labeled = True
        else:
            label = None
        ax2.plot(waypoints[sample_idx, :, 0], waypoints[sample_idx, :, 1], label=label, color=color, alpha=alpha, zorder=zorder, linewidth=linewidth)

    # mpc results
    ax2.plot(xzs[:, 0], xzs[:, 1], color='teal', label='open-loop (MPC)', zorder=3, linewidth=3)
    ax2.axis('equal')
    ax2.legend()
    ax2.set_title('coordinates (fl_cam)', fontsize=20)
    ax2.set_xlabel('X (m)', fontsize=20)
    ax2.set_ylabel('Z (m)', fontsize=20)
    ax2.tick_params(axis='both', labelsize=18)
    ax2.grid(True)

    
    # applied speed so far 
    ax3.grid(True)
    ax3.set_xlim(-n_past_steps+1, n_future_steps)
    ax3.set_ylim(0., 1.05*max_speed)                        # margin
    ax3.axvline(0., color='k', linestyle='dashed')
    n_speed_points = min(n_past_steps, len(speeds))         # history length to visualize
    ax3.plot(np.arange(-n_speed_points+1, 1), speed_commands[-n_speed_points:], label='applied', linestyle='dashed', color='tab:blue')
    ax3.plot(np.arange(-n_speed_points+1, 1), speeds[-n_speed_points:], label='real', color='tab:blue')
    ax3.set_title('speed (m/s)', fontsize=20)
    ax3.set_xlabel('step', fontsize=20)
    ax3.legend(ncols=2)

    # z coordinates
    ax4.grid(True)
    ax4.set_xlim(-n_past_steps+1, n_future_steps)
    ax4.axvline(0., color='k', linestyle='dashed')
    ax4.set_xlabel('step', fontsize=20)
    ax4.set_title('z', fontsize=20)
    ax4.plot(np.arange(n_future_steps+1), xzs[:, 1], alpha=0.5, color='teal', label='open loop')
    ax4.plot(np.arange(1, n_future_steps+1), waypoints_z, alpha=0.5, linestyle='dashed', color='teal', label='waypoints')
    ax4.legend()

    ax5.grid(True)
    ax5.set_xlim(-n_past_steps+1, n_future_steps)
    ax5.axvline(0., color='k', linestyle='dashed')
    ax5.set_xlabel('step', fontsize=20)
    ax5.set_title('velocity command', fontsize=20)

    vs = np.array(velocity_commands)
    n_vel_points = min(n_past_steps, len(velocity_commands))
    ax5.plot(np.arange(-n_vel_points+1, 1), vs[-n_vel_points:, 0], label='linear (m/s)', color='tab:purple')
    ax5.plot(np.arange(-n_vel_points+1, 1), vs[-n_vel_points:, 1], label='angular (rad/s)', color='tab:brown')

    ax5.plot(np.arange(n_future_steps), u_mpc[:, 0], alpha=0.5, color='tab:purple')
    ax5.plot(np.arange(n_future_steps), u_mpc[:, 1], alpha=0.5, color='tab:brown')
    ax5.legend(ncols=2, loc='upper right')

    # TODO: MPC computation time + local policy inference time + optimization time
    ax6.grid(True)
    ax6.set_xlim(-n_past_steps+1, n_future_steps)
    ax6.set_ylim(0., 2.*max_wall_time)
    ax6.axvline(0., color='k', linestyle='dashed')
    ax6.set_xlabel('step', fontsize=20)
    ax6.set_title('MPC wall time (s)', fontsize=20)
    n_points = min(n_past_steps, len(t_walls_mpc))
    ax6.plot(np.arange(-n_points+1, 1), t_walls_mpc[-n_points:], color='crimson')

    ax7.grid(True)
    ax7.set_xlim(-n_past_steps+1, n_future_steps)
    ax7.set_ylim(0., .6)
    ax7.axvline(0., color='k', linestyle='dashed')
    ax7.set_xlabel('step', fontsize=20)
    ax7.set_title('policy wall time (s)', fontsize=20)
    n_points = min(n_past_steps, len(t_walls_policy))
    ax7.plot(np.arange(-n_points+1, 1), t_walls_policy[-n_points:], color='crimson')

    ax8.grid(True)
    ax8.set_xlim(-n_past_steps+1, n_future_steps)
    ax8.set_ylim(0.)
    ax8.axvline(0., color='k', linestyle='dashed')
    ax8.set_xlabel('step', fontsize=20)
    ax8.set_title('evaluation wall time (s)', fontsize=20)
    n_points = min(n_past_steps, len(t_walls_eval))
    ax8.plot(np.arange(-n_points+1, 1), t_walls_eval[-n_points:], color='crimson')

    # Save the plot
    # TODO: must adjust the formatting: # of digits 
    output_path = os.path.join(dir_to_save, f'{timestamp:04d}.png')
    plt.savefig(output_path)
    plt.close(fig)
    return


def project_waypoints_onto_image_plane(waypoints, y_coordinates, K):
    """
    waypoint: array of shape (N, 2); xz plane of the camera frame
    y_coordinates: array of shape (N,) containing the y-coordinates of the waypoints

    return: two arrays of shape (N,) representing x & y coordinates of the projected points, respectively
    """
    assert waypoints.ndim == 2 and waypoints.shape[-1] == 2
    N = waypoints.shape[0]
    # xz -> xyz
    waypoints_3d = np.zeros((N, 3))        # (*, # of waypoints, 3); origin prepended
    waypoints_3d[:, [0, 2]] = waypoints
    waypoints_3d[:, 1] = y_coordinates + 1.8        # project onto the ground
    # waypoints_3d[0, 2] = FRONT_THRESHOLD + 1e-8
    u, v, valid = get_image_coordinates(waypoints_3d, K)
    # TODO: multiple segments
    return u, v, valid