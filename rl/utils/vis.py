import numpy as np
import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as MplPolygon
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from PIL import Image, ImageDraw
import imageio  # noqa: F401
import tempfile


# ─── Visualisation ───────────────────────────────────────────────────
def visualize(iteration, args, bufs, bev_images, bev_metas, wandb_module,
              substep_frames=None, obs_encoder=None, obstacle_layouts=None,
              mpc_vis_data=None):
    """Log BEV trajectory, control, and ego-video figures."""
    import time as _time
    vis_dir = os.path.join(args.save_dir, "vis")
    wrun = wandb_module

    _t = _time.time()
    log_bev_figure(
        bufs.cord, bufs.goal_world, bufs.raw_rewards, bufs.dones,
        buf_actions=bufs.actions,
        bev_images=bev_images,
        bev_metas=bev_metas,
        obstacle_layouts=obstacle_layouts,
        grid_cols=args.num_agents_per_server,
        iteration=iteration,
        save_dir=vis_dir,
        wandb=wrun,
    )
    _bev_t = _time.time() - _t

    '''
    _t = _time.time()
    if getattr(args, 'vis_ego_per_env', False):
        log_ego_video(
            bufs.obs,
            buf_rewards=bufs.raw_rewards,
            buf_dones=bufs.dones,
            buf_goal=bufs.goal,
            fps=args.vis_video_fps,
            iteration=iteration,
            save_dir=vis_dir,
            wandb=wrun,
            substep_frames=substep_frames,
        )
    _ego_t = _time.time() - _t
    '''
    
    _t = _time.time()
    log_ego_video_grid(
        bufs.obs,
        buf_rewards=bufs.raw_rewards,
        buf_dones=bufs.dones,
        buf_goal=bufs.goal,
        buf_terminateds=bufs.terminateds,
        fps=args.vis_video_fps,
        iteration=iteration,
        save_dir=vis_dir,
        wandb=wrun,
        substep_frames=substep_frames,
        grid_cols=args.num_agents_per_server,
        mpc_vis_data=mpc_vis_data,
    )
    _grid_t = _time.time() - _t
    
    '''
    _t = _time.time()
    if obs_encoder is not None:
        log_dino_pca_video_grid(
            obs_encoder, bufs.obs,
            fps=args.vis_video_fps,
            iteration=iteration,
            save_dir=vis_dir,
            wandb=wrun,
            grid_cols=args.num_agents_per_server,
            substep_frames=substep_frames,
        )
    _dino_t = _time.time() - _t

    # MPC/policy dashboard (speed, velocity commands, wall times)
    _t = _time.time()
    if mpc_vis_data is not None and any(len(d) > 0 for d in mpc_vis_data):
        log_mpc_dashboard(
            bufs.cmd_speed, bufs.real_speed,
            bufs.dones,
            mpc_vis_data=mpc_vis_data,
            grid_cols=args.num_agents_per_server,
            iteration=iteration,
            save_dir=vis_dir,
            wandb=wrun,
        )

        log_mpc_detail_video_grid(
            mpc_vis_data,
            buf_cmd_speed=bufs.cmd_speed,
            buf_real_speed=bufs.real_speed,
            buf_dones=bufs.dones,
            grid_cols=args.num_agents_per_server,
            fps=args.vis_video_fps,
            iteration=iteration,
            save_dir=vis_dir,
            wandb=wrun,
        )
    _mpc_t = _time.time() - _t
    
    print(
        f"  [VIS TIMING] bev={_bev_t:.1f}s  ego_video={_ego_t:.1f}s  "
        f"ego_grid={_grid_t:.1f}s  dino_pca={_dino_t:.1f}s  mpc={_mpc_t:.1f}s"
    )
    '''

# ─── BEV Trajectory Visualization ────────────────────────────────────


def _bev_world_to_pixel(xz_world, meta):
    """
    Project standard-coord (x, z) positions into BEV image pixel coordinates.

    Parameters
    ----------
    xz_world : (..., 2)   - (x_std, z_std)
    meta     : dict       - from CarlaEnv.capture_bev()

    Returns  : (..., 2) float  - (u, v), u rightward, v downward
    """
    xz = np.asarray(xz_world, dtype=np.float64)
    h, s = meta['altitude'], meta['img_size']
    ctr  = meta['center_xz']
    fov_rad = np.deg2rad(meta['fov_deg'])
    scale = s / (2.0 * h * np.tan(fov_rad / 2.0))   # px / m
    dx = xz[..., 0] - ctr[0]   # standard-x (right  → +u)
    dz = xz[..., 1] - ctr[1]   # standard-z (forward → −v)
    u = s / 2.0 + scale * dx
    v = s / 2.0 - scale * dz
    return np.stack([u, v], axis=-1)


# ─── CCTV camera helpers ─────────────────────────────────────────────

def compute_cctv_specs(obs_cord, obs_goal_world, obstacle_layouts=None,
                       pitch_deg=-45.0, fov=90.0, min_altitude=12.0,
                       margin=5.0):
    """Compute per-env CCTV camera placement specs.

    The camera is placed to the side of the start→goal line at an
    elevation determined by the scene extent, looking at the scene
    centre with *pitch_deg* tilt (default −45°, i.e. looking downward).

    Returns list of dicts with keys needed by
    ``CarlaMultiAgentEnv.spawn_cctv_cameras``.
    """
    num_envs = len(obs_cord)
    current_xz = obs_cord[:, -2:]                          # (E, 2)
    goal_global = current_xz + obs_goal_world              # (E, 2)
    fov_rad = math.radians(fov)

    specs = []
    for env_idx in range(num_envs):
        pts = [current_xz[env_idx].reshape(1, 2),
               goal_global[env_idx].reshape(1, 2)]

        if obstacle_layouts is not None and env_idx < len(obstacle_layouts):
            layout = obstacle_layouts[env_idx]
            agents = layout.get('agents', []) if layout else []
            aidx = env_idx % len(agents) if agents else -1
            ag = agents[aidx] if 0 <= aidx < len(agents) else None
            if ag is not None:
                geo = ag.get('geodesic_path_std')
                if geo is not None and len(geo) >= 2:
                    pts.append(np.asarray(geo))

        all_pts = np.vstack(pts)
        valid = np.isfinite(all_pts).all(axis=1)
        if not valid.any():
            specs.append(None)
            continue
        all_pts = all_pts[valid]

        # Bounding box of start + goal + geodesic
        lo = all_pts.min(axis=0)
        hi = all_pts.max(axis=0)
        center_xz = (lo + hi) / 2.0
        extent = np.max(hi - lo) / 2.0 + margin

        # Altitude: ensure the FOV covers the full extent on the ground
        # At pitch_deg tilt, the effective ground footprint is larger
        # than for a top-down camera, but we size for the horizontal FOV
        # projected onto the near-ground plane.
        pitch_rad = math.radians(abs(pitch_deg))
        # Ground width visible = 2 * altitude * tan(fov/2) / sin(pitch)
        # Solve for altitude: altitude = extent * sin(pitch) / tan(fov/2)
        altitude = max(min_altitude,
                       extent * math.sin(pitch_rad) / math.tan(fov_rad / 2.0))

        # Camera offset direction: perpendicular to start→goal for a side view
        sg = goal_global[env_idx] - current_xz[env_idx]
        sg_len = np.linalg.norm(sg)
        if sg_len > 1e-3:
            sg_unit = sg / sg_len
            # Perpendicular (rotate 90° CCW in the xz plane)
            perp = np.array([-sg_unit[1], sg_unit[0]])
        else:
            perp = np.array([1.0, 0.0])

        # Horizontal offset from center: at the given pitch, the camera
        # needs to be offset so it looks at the center from the side.
        horiz_offset = altitude / math.tan(pitch_rad)
        cam_xz = center_xz + perp * horiz_offset   # standard (x_std, z_std)

        # Yaw: camera should look from cam_xz toward center_xz
        # In UE: ue_x = z_std, ue_y = x_std
        # Camera forward in UE is +X. Yaw rotates around UE Z-up.
        look_dir_std = center_xz - cam_xz
        # Convert to UE ground direction: (ue_x, ue_y) = (z_std, x_std)
        look_ue_x = look_dir_std[1]   # z_std
        look_ue_y = look_dir_std[0]   # x_std
        yaw_deg = math.degrees(math.atan2(look_ue_y, look_ue_x))

        # Camera UE position
        cam_ue_x = cam_xz[1]          # z_std -> ue_x
        cam_ue_y = cam_xz[0]          # x_std -> ue_y
        # Ground height approximation: use the first agent's ground level.
        # A better estimate would come from the env, but 2.0 is reasonable
        # for CARLA pedestrian maps.
        ground_z_ue = 2.0
        cam_ue_z = ground_z_ue + altitude

        specs.append({
            'agent_idx': env_idx,
            'cam_ue': np.array([cam_ue_x, cam_ue_y, cam_ue_z]),
            'pitch_deg': float(pitch_deg),
            'yaw_deg': float(yaw_deg),
            'ground_z_ue': float(ground_z_ue),
            'center_xz': center_xz.astype(np.float32),
        })

    return [s for s in specs if s is not None]


def _cctv_world_to_pixel(xz_std, spec):
    """Project standard-coord ground points to CCTV image pixels.

    Parameters
    ----------
    xz_std  : (..., 2) standard coords (x_right, z_forward)
    spec    : dict from ``compute_cctv_specs`` enriched with
              ``fov_deg`` and ``img_size``

    Returns  : (..., 2) float (u, v), (...) bool visibility mask
    """
    xz = np.asarray(xz_std, dtype=np.float64)
    orig_shape = xz.shape[:-1]
    xz = xz.reshape(-1, 2)
    N = xz.shape[0]

    img_size = spec['img_size']
    fov_rad = math.radians(spec['fov_deg'])
    fx = fy = img_size / (2.0 * math.tan(fov_rad / 2.0))
    cx = cy = img_size / 2.0

    # World points in UE: ue_x = z_std, ue_y = x_std, ue_z = ground
    ground_z = spec['ground_z_ue']
    pts_ue = np.column_stack([xz[:, 1], xz[:, 0],
                              np.full(N, ground_z)])

    # Camera position in UE
    cam_ue = np.asarray(spec['cam_ue'], dtype=np.float64)

    # Camera rotation: build world-to-local rotation matrix
    # CARLA convention: yaw around UE-Z, then pitch around UE-Y
    yaw_r = math.radians(spec['yaw_deg'])
    pitch_r = math.radians(spec['pitch_deg'])

    cy_, sy = math.cos(yaw_r), math.sin(yaw_r)
    cp, sp = math.cos(pitch_r), math.sin(pitch_r)

    # R_yaw (around UE-Z up)
    R_yaw = np.array([[cy_, -sy, 0.],
                       [sy,  cy_, 0.],
                       [0.,  0.,  1.]])
    # R_pitch (around UE-Y right, UE/CARLA left-hand convention)
    R_pitch = np.array([[cp,  0., -sp],
                         [0.,  1., 0.],
                         [sp,  0., cp]])

    # Actor local-to-world: R_l2w = R_yaw @ R_pitch
    R_l2w = R_yaw @ R_pitch
    # World-to-local: R_w2l = R_l2w^T
    R_w2l = R_l2w.T

    # Transform to actor-local UE frame
    diff = pts_ue - cam_ue[None, :]           # (N, 3)
    local_ue = (R_w2l @ diff.T).T             # (N, 3)

    # UE-local to camera convention: cam_x=ue_y, cam_y=-ue_z, cam_z=ue_x
    cam_x = local_ue[:, 1]
    cam_y = -local_ue[:, 2]
    cam_z = local_ue[:, 0]

    # Pinhole projection
    visible = cam_z > 0.1
    u = np.where(visible, fx * cam_x / cam_z + cx, -1.0)
    v = np.where(visible, fy * cam_y / cam_z + cy, -1.0)

    uv = np.stack([u, v], axis=-1).reshape(*orig_shape, 2)
    visible = visible.reshape(orig_shape)
    return uv, visible


def compute_bev_specs(obs_cord, obs_goal_world,
                      obstacle_layouts=None,
                      fov=90.0, min_altitude=15.0, margin=5.0):
    """Compute per-env BEV capture specs encompassing start, goal, and geodesic.

    The returned altitude is chosen so that the camera's field of view
    covers the bounding box of the agent position, goal position, and
    (when available) the geodesic path connecting them, plus *margin*
    metres of padding on each side.

    Parameters
    ----------
    obs_cord : (E, context_size*2)
        Flattened position history; ``[..., -2:]`` gives current (x, z).
    obs_goal_world : (E, 2)
        Goal position relative to the agent in world frame.
    obstacle_layouts : list[dict] or None
        Per-env layouts from ``get_obstacle_layouts()``.  When provided,
        each agent's ``geodesic_path_std`` is included in the bounding box.
    fov : float
        Camera horizontal FOV in degrees.
    min_altitude : float
        Minimum BEV altitude in metres.
    margin : float
        Padding around the bounding box in metres.

    Returns
    -------
    list of dict with keys ``env_idx``, ``center_xz``, ``altitude``
    """
    num_envs = len(obs_cord)
    current_xz = obs_cord[:, -2:]                 # (E, 2)
    goal_global = current_xz + obs_goal_world     # (E, 2)
    fov_rad = np.deg2rad(fov)

    specs = []
    for env_idx in range(num_envs):
        pts = [current_xz[env_idx].reshape(1, 2),
               goal_global[env_idx].reshape(1, 2)]

        # Include geodesic path when available
        if obstacle_layouts is not None and env_idx < len(obstacle_layouts):
            layout = obstacle_layouts[env_idx]
            agents = layout.get('agents', []) if layout else []
            aidx = env_idx % len(agents) if agents else -1
            ag = agents[aidx] if 0 <= aidx < len(agents) else None
            if ag is not None:
                geo = ag.get('geodesic_path_std')
                if geo is not None and len(geo) >= 2:
                    pts.append(np.asarray(geo))

        all_pts = np.vstack(pts)
        valid = np.isfinite(all_pts).all(axis=1)
        if not valid.any():
            specs.append({
                'env_idx': env_idx,
                'center_xz': current_xz[env_idx].astype(np.float32),
                'altitude': float(min_altitude),
            })
            continue

        all_pts = all_pts[valid]
        bb_min = np.min(all_pts, axis=0)
        bb_max = np.max(all_pts, axis=0)
        center = (bb_min + bb_max) / 2.0
        half_ext = np.max((bb_max - bb_min) / 2.0) + margin

        altitude = half_ext / np.tan(fov_rad / 2.0)
        altitude = max(altitude, min_altitude)

        specs.append({
            'env_idx': env_idx,
            'center_xz': center.astype(np.float32),
            'altitude': float(altitude),
        })

    return specs


def log_bev_figure(
    buf_cord, buf_goal, buf_rewards, buf_dones,
    buf_actions=None,
    bev_images=None,
    bev_metas=None,
    obstacle_layouts=None,
    grid_cols=None,
    iteration=0,
    save_dir=None,
    wandb=None,
):
    """
    Render Bird's-Eye-View trajectory plots in a grid layout.

    Data layout
    -----------
    buf_cord    : (num_steps, num_envs, context_size*2)
                  Flattened x,z position history.  buf_cord[t, e, -2:] gives
                  the current robot (x_std, z_std) at step t in env e.
    buf_goal    : (num_steps, num_envs, 2)
                  Goal in world frame relative to the robot:
                  goal_global = buf_cord[..., -2:] + buf_goal
    buf_rewards : (num_steps, num_envs)
    buf_dones   : (num_steps, num_envs)   1 at episode end
    buf_actions : (num_steps, num_envs, 10)  optional - 5 cumulative
                  waypoints x 2 in camera frame
    bev_images  : list[ndarray | None]  - per-env BEV images from
                  CarlaEnv.capture_bev(); None → plain axes background
    bev_metas   : list[dict | None]     - corresponding projection metadata
    obstacle_layouts : list[dict] or None
                  Per-env obstacle layout data from get_obstacle_layouts().
                  When provided, the per-agent geodesic path
                  (``agents[i]['geodesic_path_std']``) is drawn as a cyan line.
    grid_cols   : int or None - number of columns in the grid layout.
                  When None, uses ceil(sqrt(num_envs)) for a roughly square grid.
    """


    num_steps, num_envs = buf_rewards.shape

    # current (x, z) position at every step — buf_cord[t, e, -2:]
    current_xz = buf_cord[:, :, -2:]          # (steps, envs, 2)
    goal_global = current_xz + buf_goal       # (steps, envs, 2)

    # ── sanity diagnostics ────────────────────────────────────────────
    n_nan = int(np.isnan(current_xz).any(axis=-1).sum())
    n_inf = int(np.isinf(current_xz).any(axis=-1).sum())
    if n_nan or n_inf:
        print(f"  [BEV] WARNING: trajectory has {n_nan} NaN and {n_inf} Inf entries")

    # ── visual style (cohesive palette, all white-edged) ────────────
    _TRAJ_COLOR = '#FFA726'          # warm amber — visible on BEV & plain bg
    _MARKER = dict(
        start=dict(marker='^', s=90, c='#43E97B', edgecolors='white',
                   linewidths=1.2, zorder=5),
        end=dict(marker='D', s=70, c='#F5576C', edgecolors='white',
                 linewidths=1.2, zorder=5),
        goal=dict(marker='*', s=260, c='#667EEA', edgecolors='white',
                  linewidths=0.9, zorder=6),
    )

    # Layout: grid of BEV plots — no spacing between cells
    if grid_cols is None:
        grid_cols = math.ceil(math.sqrt(num_envs))
    grid_rows = math.ceil(num_envs / grid_cols)
    cell_size = 4.5
    legend_frac = 0.045                     # fraction of height for legend
    fig_h = cell_size * grid_rows / (1 - legend_frac)
    fig, axes = plt.subplots(
        grid_rows, grid_cols,
        figsize=(cell_size * grid_cols, fig_h),
        squeeze=False,
    )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=legend_frac,
                        wspace=0, hspace=0)

    for env_idx in range(num_envs):
        row_i, col_i = divmod(env_idx, grid_cols)
        ax = axes[row_i][col_i]

        # ── choose coordinate system ──────────────────────────────────
        bev_img  = (bev_images or [None] * num_envs)[env_idx]
        bev_meta = (bev_metas  or [None] * num_envs)[env_idx]
        use_bev  = bev_img is not None and bev_meta is not None

        def _to_plot(xz_arr):
            """Convert (N, 2) standard-coord array to plot coords."""
            if use_bev:
                return _bev_world_to_pixel(xz_arr, bev_meta)
            return xz_arr   # world coords directly

        if use_bev:
            s = bev_meta['img_size']
            ax.imshow(bev_img, origin="upper",
                      extent=[0, s, s, 0],   # (left, right, bottom, top)
                      aspect="auto")
            ax.set_xlim(0, s)
            ax.set_ylim(s, 0)   # y-axis: 0 at top, s at bottom
        else:
            s = None
            ax.set_aspect("auto")
            ax.grid(True, alpha=0.3)
        # strip all tick labels, ticks, and spines for compact grid
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # ── obstacle layout overlays (from obstacle_layouts) ─────────
        _layout = None
        if obstacle_layouts is not None and env_idx < len(obstacle_layouts):
            _layout = obstacle_layouts[env_idx]

        # Geodesic path
        _geo_path_std = None
        if _layout is not None:
            _agents = _layout.get('agents', [])
            _aidx = env_idx % len(_agents) if _agents else -1
            _ag = _agents[_aidx] if 0 <= _aidx < len(_agents) else None
            if _ag is not None:
                _geo_path_std = _ag.get('geodesic_path_std')
        if _geo_path_std is not None and len(_geo_path_std) >= 2:
            _geo_plot = _to_plot(np.asarray(_geo_path_std))
            _gv = np.isfinite(_geo_plot).all(axis=1)
            _geo_plot_v = _geo_plot[_gv]
            if len(_geo_plot_v) >= 2:
                ax.plot(_geo_plot_v[:, 0], _geo_plot_v[:, 1],
                        '-', color='#00E5FF', linewidth=1.2,
                        alpha=0.55, zorder=2,
                        label='geodesic' if env_idx == 0 else None)

        # Obstacle OBBs
        _OBS_COLORS = {
            'blocked_crosswalk':       '#E53935',
            'crosswalk_challenge':     '#E53935',
            'blocked_crosswalk_flank': '#FB8C00',
            'crosswalk_challenge_flank': '#FB8C00',
            'narrow_passage':          '#FFB300',
            'sidewalk_obstruction':    '#FDD835',
        }
        if _layout is not None:
            for obs in _layout.get('obstacles', []):
                obs_plot = _to_plot(obs['corners_std'])
                if np.isfinite(obs_plot).all():
                    _c = _OBS_COLORS.get(obs['scenario_type'], '#FFFFFF')
                    poly = MplPolygon(
                        obs_plot, closed=True,
                        facecolor=_c, edgecolor='white',
                        alpha=0.55, linewidth=0.8, zorder=2)
                    ax.add_patch(poly)

        # ── per-env trajectory data ───────────────────────────────────
        xs = current_xz[:, env_idx, 0]
        zs = current_xz[:, env_idx, 1]
        dones = buf_dones[:, env_idx]

        gx = goal_global[:, env_idx, 0]
        gz = goal_global[:, env_idx, 1]

        ep_starts = [0] + (np.where(dones[:-1] > 0)[0] + 1).tolist()
        ep_ends   = np.where(dones > 0)[0].tolist() + [num_steps - 1]

        for ep_i, (t0, t1) in enumerate(zip(ep_starts, ep_ends)):
            ep_xz = np.stack([xs[t0:t1+1], zs[t0:t1+1]], axis=-1)  # (L, 2)

            # skip episodes with NaN/Inf positions
            valid = np.isfinite(ep_xz).all(axis=-1)
            if not valid.any():
                continue

            # Diagnostic: verify goal-start distance for each episode.
            # ep 0 is a continuation from the previous rollout, so its
            # "start" is mid-episode — skip it to avoid misleading distances.
            _goal_xz = np.array([gx[t0], gz[t0]])
            _start_xz = ep_xz[0]
            if np.isfinite(_goal_xz).all() and np.isfinite(_start_xz).all():
                _gs_dist = float(np.linalg.norm(_goal_xz - _start_xz))
                tag = "" if ep_i > 0 else " (continuation)"
                print(f"  [BEV] env {env_idx} ep {ep_i}: "
                      f"steps={t1-t0+1}  "
                      f"goal-start dist={_gs_dist:.2f}m  "
                      f"start=({_start_xz[0]:.1f}, {_start_xz[1]:.1f})  "
                      f"goal=({_goal_xz[0]:.1f}, {_goal_xz[1]:.1f})"
                      f"{tag}")

            ep_plot = _to_plot(ep_xz)                                # (L, 2)

            # ── trajectory line (single colour) ─────────────────────
            L = len(ep_plot)
            if L >= 2:
                points = ep_plot.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                seg_valid = np.isfinite(segments).all(axis=(1, 2))
                lc = LineCollection(
                    segments[seg_valid],
                    colors=_TRAJ_COLOR,
                    linewidths=2.0,
                    zorder=3,
                )
                ax.add_collection(lc)

            # start / end (only plot if coordinates are finite)
            if np.isfinite(ep_plot[0]).all():
                ax.scatter(*ep_plot[0], **_MARKER['start'],
                           label="start" if ep_i == 0 else None)
            if np.isfinite(ep_plot[-1]).all():
                ax.scatter(*ep_plot[-1], **_MARKER['end'],
                           label="end" if ep_i == 0 else None)

            # ── goal (★) ─────────────────────────────────────────────
            goal_xz = np.array([gx[t0], gz[t0]])
            if np.isfinite(goal_xz).all():
                gp = _to_plot(goal_xz.reshape(1, 2))[0]
                if np.isfinite(gp).all():
                    ax.scatter(*gp, **_MARKER['goal'],
                               label="goal" if ep_i == 0 else None)
                    ax.plot([ep_plot[0, 0], gp[0]], [ep_plot[0, 1], gp[1]],
                            "--", color="#667EEA", alpha=0.45, linewidth=1.0,
                            zorder=2)
            '''
            # ── predicted waypoints (every ~8 samples) ───────────────
            if buf_actions is not None and t1 > t0:
                stride = max(1, (t1 - t0) // 8)
                for t in range(t0, t1, stride):
                    if not np.isfinite(xs[t]) or not np.isfinite(zs[t]):
                        continue
                    if t > 0 and np.isfinite(xs[t - 1]) and np.isfinite(zs[t - 1]):
                        ddx = xs[t] - xs[t - 1]
                        ddz = zs[t] - zs[t - 1]
                    else:
                        ddx, ddz = 0.0, 1e-6
                    yaw = math.atan2(ddx, ddz)          # angle from +z axis
                    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
                    # camera frame → world frame
                    Rot = np.array([[cos_y, sin_y], [-sin_y, cos_y]])
                    wp_cam = buf_actions[t, env_idx].reshape(5, 2)
                    if not np.isfinite(wp_cam).all():
                        continue
                    wp_world = (Rot @ wp_cam.T).T + np.array([xs[t], zs[t]])
                    wp_plot  = _to_plot(wp_world)                    # (5, 2)
                    cur_plot = _to_plot(np.array([[xs[t], zs[t]]]))[0]
                    ax.plot(
                        [cur_plot[0]] + wp_plot[:, 0].tolist(),
                        [cur_plot[1]] + wp_plot[:, 1].tolist(),
                        "-", color="mediumpurple", linewidth=0.8, alpha=0.6,
                        zorder=4,
                        label="waypoints" if (ep_i == 0 and t == t0) else None,
                    )
                    ax.scatter(wp_plot[:, 0], wp_plot[:, 1],
                               s=12, c="mediumpurple", alpha=0.6, zorder=4)
            '''
    # hide unused axes when num_envs doesn't fill the grid
    for idx in range(num_envs, grid_rows * grid_cols):
        r, c = divmod(idx, grid_cols)
        axes[r][c].set_visible(False)

    # ── shared legend (bottom-centre) ──────────────────────────────
    
    legend_handles = [
        Line2D([], [], marker=_MARKER['start']['marker'], color='none',
               markerfacecolor=_MARKER['start']['c'],
               markeredgecolor='white', markersize=10, label='start'),
        Line2D([], [], marker=_MARKER['end']['marker'], color='none',
               markerfacecolor=_MARKER['end']['c'],
               markeredgecolor='white', markersize=10, label='end'),
        Line2D([], [], marker=_MARKER['goal']['marker'], color='none',
               markerfacecolor=_MARKER['goal']['c'],
               markeredgecolor='white', markersize=12, label='goal'),
    ]
    if obstacle_layouts is not None:
        has_obs = any(len(l.get('obstacles', [])) > 0
                      for l in obstacle_layouts if l is not None)
        if has_obs:
            legend_handles.append(
                Patch(facecolor='#E53935', edgecolor='white',
                      alpha=0.55, label='obstacle'))
        if any(
            any(a.get('geodesic_path_std') is not None
                for a in l.get('agents', []))
            for l in obstacle_layouts if l is not None
        ):
            legend_handles.append(
                Line2D([], [], color='#00E5FF', linewidth=1.5,
                       alpha=0.55, label='geodesic'))
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=len(legend_handles), fontsize=11, framealpha=0.8,
               borderpad=0.4, handletextpad=0.4, columnspacing=1.2)

    # ── diagnostics ───────────────────────────────────────────────────
    print(f"  [BEV] reward range: [{float(np.nanmin(buf_rewards)):.3f}, {float(np.nanmax(buf_rewards)):.3f}]")
    for env_idx in range(num_envs):
        xs_e = current_xz[:, env_idx, 0]
        zs_e = current_xz[:, env_idx, 1]
        print(f"  [BEV] env {env_idx}: x=[{np.nanmin(xs_e):.1f}, {np.nanmax(xs_e):.1f}] "
              f"z=[{np.nanmin(zs_e):.1f}, {np.nanmax(zs_e):.1f}]")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"bev_{iteration:04d}.png")
        fig.savefig(path, dpi=120)
        print(f"  → BEV figure saved to {path}")

    if wandb is not None:
        wandb.log({"rollout/bev_trajectories": wandb.Image(fig)}, step=iteration)

    plt.close(fig)


# ─── Obstacle Layout BEV Visualization ───────────────────────────────


def log_obstacle_bev_figure(
    obstacle_layouts,
    bev_metas,
    ped_positions=None,
    grid_cols=None,
    iteration=0,
    save_dir=None,
    wandb=None,
):
    """
    Render Bird's-Eye-View obstacle layout on a dark background.

    Draws oriented bounding boxes for each spawned obstacle, filled
    polygons for detected crosswalks, pedestrian trajectories, and
    per-agent ego/goal markers with text annotations describing the
    generation procedure and results.

    Parameters
    ----------
    obstacle_layouts : list[dict]
        Per-env data from ``VecCarlaMultiAgentEnv.get_obstacle_layouts()``.
        Keys: ``'obstacles'``, ``'crosswalks'``, ``'pedestrians'``,
        ``'agents'`` (per-agent metadata with ego/goal/method/scenarios).
    bev_metas : list[dict | None]
        Per-env projection metadata from ``capture_bev()``.
    grid_cols : int or None
    iteration : int
    save_dir  : str or None
    wandb     : wandb module or None
    """


    num_envs = len(obstacle_layouts)
    if grid_cols is None:
        grid_cols = math.ceil(math.sqrt(num_envs))
    grid_rows = math.ceil(num_envs / grid_cols)
    cell_size = 4.5
    legend_frac = 0.10
    fig_h = cell_size * grid_rows / (1 - legend_frac)

    BG_COLOR = '#1A1A2E'

    fig, axes = plt.subplots(
        grid_rows, grid_cols,
        figsize=(cell_size * grid_cols, fig_h),
        squeeze=False,
    )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=legend_frac,
                        wspace=0, hspace=0)
    fig.patch.set_facecolor(BG_COLOR)

    # ── colour palette ────────────────────────────────────────────────
    SCENARIO_COLORS = {
        'blocked_crosswalk':       '#E53935',   # red
        'crosswalk_challenge':     '#E53935',
        'blocked_crosswalk_flank': '#FB8C00',   # orange
        'narrow_passage':          '#FFB300',    # amber
        'sidewalk_obstruction':    '#FDD835',    # yellow
    }
    # Segmentation layer colours (muted, semi-transparent)
    SEG_SIDEWALK_COLOR = (0.42, 0.58, 0.72, 0.45)    # steel blue
    SEG_CROSSWALK_COLOR = (0.85, 0.85, 0.25, 0.45)   # muted yellow
    SEG_ROAD_COLOR = (0.30, 0.30, 0.30, 0.35)         # dark gray

    CROSSWALK_COLOR  = '#29B6F6'         # light blue (polygon outlines)
    CROSSWALK_BLOCKED_COLOR = '#EF5350'  # red (blocked crosswalks)
    CROSSWALK_ALPHA  = 0.35
    OBSTACLE_ALPHA   = 0.75
    PED_TRAJ_COLOR   = '#E040FB'         # magenta / pink
    PED_MARKER_COLOR = '#F8F8F8'         # near-white
    PED_DEST_COLOR   = '#CE93D8'         # light purple
    EGO_COLOR        = '#43E97B'         # green (matches main BEV start)
    GOAL_COLOR       = '#667EEA'         # blue  (matches main BEV goal)

    # Human-readable labels for goal sampling methods
    _METHOD_LABEL = {
        'crosswalk_challenge': 'crosswalk challenge',
        'navmesh_annulus':     'navmesh annulus',
        'random_fallback':     'random fallback',
    }
    # Short labels for scenario names
    _SCENARIO_LABEL = {
        'blocked_crosswalk':    'blocked CW',
        'crosswalk_challenge':  'CW challenge',
        'sidewalk_obstruction': 'sidewalk clutter',
        'narrow_passage':       'narrow passage',
    }

    for env_idx in range(num_envs):
        row_i, col_i = divmod(env_idx, grid_cols)
        ax = axes[row_i][col_i]

        meta = (bev_metas or [None] * num_envs)[env_idx]
        layout = obstacle_layouts[env_idx]
        if meta is None or layout is None:
            ax.set_visible(False)
            continue

        s = meta['img_size']
        ax.set_facecolor(BG_COLOR)
        ax.set_xlim(0, s)
        ax.set_ylim(s, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        margin = s * 0.3

        # ── navmesh segmentation layer (road → sidewalk → crosswalk) ─
        for key, color in [('road_tris_std',      SEG_ROAD_COLOR),
                           ('sidewalk_tris_std',  SEG_SIDEWALK_COLOR),
                           ('crosswalk_tris_std', SEG_CROSSWALK_COLOR)]:
            tris_std = layout.get(key)
            if tris_std is None or len(tris_std) == 0:
                continue
            # tris_std: (N, 3, 2) standard coords → project to pixel
            n_tris = len(tris_std)
            flat = tris_std.reshape(-1, 2)
            flat_px = _bev_world_to_pixel(flat, meta)
            tri_px = flat_px.reshape(n_tris, 3, 2)
            # Cull triangles entirely outside the image
            in_x = (tri_px[:, :, 0] > -margin) & (tri_px[:, :, 0] < s + margin)
            in_y = (tri_px[:, :, 1] > -margin) & (tri_px[:, :, 1] < s + margin)
            visible = (in_x & in_y).any(axis=1)
            if visible.any():
                pc = PolyCollection(
                    tri_px[visible], facecolors=[color],
                    edgecolors='none', zorder=1)
                ax.add_collection(pc)

        # ── crosswalks (filled polygons + axes + centers) ─────────────
        CW_CENTER_COLOR = '#FFFFFF'
        CW_LONG_AXIS_COLOR = '#FF6F00'    # orange — road-parallel axis
        CW_CROSS_AXIS_COLOR = '#00E676'   # green — crossing direction
        CW_AXIS_LW = 1.5
        CW_AXIS_ALPHA = 0.8

        for cw_entry in layout.get('crosswalks', []):
            # Support both dict (enriched) and plain ndarray (legacy)
            if isinstance(cw_entry, dict):
                cw_std = cw_entry['polygon_std']
                cw_center = cw_entry.get('center_std')
                cw_long = cw_entry.get('long_axis_std')
                cw_cross = cw_entry.get('cross_axis_std')
                cw_hl = cw_entry.get('half_length', 0)
                cw_hw = cw_entry.get('half_width', 0)
                cw_blocked = cw_entry.get('blocked', False)
            else:
                cw_std = cw_entry
                cw_center = cw_long = cw_cross = None
                cw_hl = cw_hw = 0
                cw_blocked = False

            cw_px = _bev_world_to_pixel(cw_std, meta)
            visible = np.any(
                (cw_px[:, 0] > -margin) & (cw_px[:, 0] < s + margin) &
                (cw_px[:, 1] > -margin) & (cw_px[:, 1] < s + margin))
            if not visible:
                continue

            cw_color = CROSSWALK_BLOCKED_COLOR if cw_blocked else CROSSWALK_COLOR
            poly = MplPolygon(
                cw_px, closed=True,
                facecolor=cw_color, edgecolor='white',
                alpha=CROSSWALK_ALPHA, linewidth=0.8, zorder=2,
            )
            ax.add_patch(poly)

            '''
            # Draw center and axes if available
            if cw_center is not None:
                center_px = _bev_world_to_pixel(
                    np.asarray(cw_center).reshape(1, 2), meta)[0]
                if np.isfinite(center_px).all():
                    ax.plot(center_px[0], center_px[1], 'o',
                            color=CW_CENTER_COLOR, markersize=4,
                            markeredgecolor='black', markeredgewidth=0.6,
                            zorder=7)

                    if cw_long is not None and cw_hl > 0:
                        # Long axis (road-parallel)
                        long_end1 = cw_center + np.asarray(cw_long) * cw_hl
                        long_end2 = cw_center - np.asarray(cw_long) * cw_hl
                        ends_std = np.stack([long_end1, long_end2])
                        ends_px = _bev_world_to_pixel(ends_std, meta)
                        ax.plot(ends_px[:, 0], ends_px[:, 1],
                                '-', color=CW_LONG_AXIS_COLOR,
                                linewidth=CW_AXIS_LW, alpha=CW_AXIS_ALPHA,
                                zorder=6)

                    if cw_cross is not None and cw_hw > 0:
                        # Cross axis (crossing direction)
                        cross_end1 = cw_center + np.asarray(cw_cross) * cw_hw
                        cross_end2 = cw_center - np.asarray(cw_cross) * cw_hw
                        ends_std = np.stack([cross_end1, cross_end2])
                        ends_px = _bev_world_to_pixel(ends_std, meta)
                        ax.plot(ends_px[:, 0], ends_px[:, 1],
                                '-', color=CW_CROSS_AXIS_COLOR,
                                linewidth=CW_AXIS_LW, alpha=CW_AXIS_ALPHA,
                                zorder=6)
            '''
        # ── obstacles (oriented bounding boxes) ──────────────────────
        for obs in layout.get('obstacles', []):
            corners_px = _bev_world_to_pixel(obs['corners_std'], meta)
            if np.any((corners_px[:, 0] > -margin) & (corners_px[:, 0] < s + margin) &
                       (corners_px[:, 1] > -margin) & (corners_px[:, 1] < s + margin)):
                color = SCENARIO_COLORS.get(obs['scenario_type'], '#FFFFFF')
                poly = MplPolygon(
                    corners_px, closed=True,
                    facecolor=color, edgecolor='white',
                    alpha=OBSTACLE_ALPHA, linewidth=1.0, zorder=3,
                )
                ax.add_patch(poly)

        
        # ── pedestrians (accumulated trajectories from ped_positions) ──
        if (ped_positions is not None
                and env_idx < len(ped_positions)
                and len(ped_positions[env_idx]) > 0):
            # ped_positions[env_idx] is a list of (N_ped, 2) arrays,
            # one per rollout step.  Build per-pedestrian trails.
            env_peds = ped_positions[env_idx]
            T = len(env_peds)
            n_peds = max((p.shape[0] for p in env_peds if p is not None
                          and len(p) > 0), default=0)
            for pi in range(n_peds):
                trail = []
                for t in range(T):
                    if (env_peds[t] is not None
                            and pi < env_peds[t].shape[0]):
                        trail.append(env_peds[t][pi])
                if len(trail) >= 2:
                    trail_std = np.array(trail)
                    trail_px = _bev_world_to_pixel(trail_std, meta)
                    valid = np.isfinite(trail_px).all(axis=1)
                    trail_px_v = trail_px[valid]
                    if len(trail_px_v) >= 2:
                        ax.plot(trail_px_v[:, 0], trail_px_v[:, 1],
                                '-', color=PED_TRAJ_COLOR, linewidth=1.0,
                                alpha=0.4, zorder=4)
                # Current position (last known)
                if trail:
                    pos_px = _bev_world_to_pixel(
                        trail[-1].reshape(1, 2), meta)[0]
                    if (0 <= pos_px[0] <= s and 0 <= pos_px[1] <= s):
                        ax.plot(pos_px[0], pos_px[1], 'o',
                                color=PED_MARKER_COLOR, markersize=3,
                                markeredgecolor=PED_TRAJ_COLOR,
                                markeredgewidth=0.6, zorder=5, alpha=0.4)
        else:
            # Fallback: use snapshot from obstacle_layouts
            for ped in layout.get('pedestrians', []):
                traj = ped['trajectory_std']
                traj_px = _bev_world_to_pixel(traj, meta)
                if len(traj_px) >= 2:
                    seg_valid = np.isfinite(traj_px).all(axis=1)
                    traj_px_v = traj_px[seg_valid]
                    if len(traj_px_v) >= 2:
                        ax.plot(traj_px_v[:, 0], traj_px_v[:, 1],
                                '-', color=PED_TRAJ_COLOR, linewidth=1.0,
                                alpha=0.4, zorder=4)
                pos_px = _bev_world_to_pixel(
                    ped['position_std'].reshape(1, 2), meta)[0]
                if (0 <= pos_px[0] <= s and 0 <= pos_px[1] <= s):
                    ax.plot(pos_px[0], pos_px[1], 'o',
                            color=PED_MARKER_COLOR, markersize=3,
                            markeredgecolor=PED_TRAJ_COLOR,
                            markeredgewidth=0.6, zorder=5, alpha=0.4)
        
        # ── per-agent annotations (ego, goal, metadata) ──────────────
        agents = layout.get('agents', [])
        # Map flat env_idx to the per-server agent index.  All agents on
        # one server share the same layout dict, so the agent list length
        # equals the number of agents per server.
        agent_idx_local = env_idx % len(agents) if agents else -1
        agent = agents[agent_idx_local] if 0 <= agent_idx_local < len(agents) else None

        if agent is not None:
            ego_std = np.asarray(agent['ego_std'])
            goal_std = np.asarray(agent['goal_std'])

            ego_px = _bev_world_to_pixel(ego_std.reshape(1, 2), meta)[0]
            goal_px = _bev_world_to_pixel(goal_std.reshape(1, 2), meta)[0]

            # Geodesic path (replaces dashed ego→goal line when available)
            geo_path_std = agent.get('geodesic_path_std')
            if geo_path_std is not None and len(geo_path_std) >= 2:
                geo_px = _bev_world_to_pixel(np.asarray(geo_path_std), meta)
                geo_valid = np.isfinite(geo_px).all(axis=1)
                geo_px_v = geo_px[geo_valid]
                if len(geo_px_v) >= 2:
                    ax.plot(geo_px_v[:, 0], geo_px_v[:, 1],
                            '-', color='#00E5FF', linewidth=1.5,
                            alpha=0.7, zorder=6)
            elif np.isfinite(ego_px).all() and np.isfinite(goal_px).all():
                # Fallback: dashed straight line from ego to goal
                ax.plot([ego_px[0], goal_px[0]], [ego_px[1], goal_px[1]],
                        '--', color=GOAL_COLOR, alpha=0.45,
                        linewidth=1.0, zorder=6)

            # Ego marker (green triangle)
            if np.isfinite(ego_px).all():
                ax.scatter(ego_px[0], ego_px[1], marker='^', s=90,
                           c=EGO_COLOR, edgecolors='white',
                           linewidths=1.0, zorder=8)

            # Goal marker (blue star)
            if np.isfinite(goal_px).all():
                ax.scatter(goal_px[0], goal_px[1], marker='*', s=200,
                           c=GOAL_COLOR, edgecolors='white',
                           linewidths=0.8, zorder=8)

            # ── text info box (top-left of each cell) ────────────────
            town = agent.get('town', '?')
            quadrant = agent.get('quadrant', '?')
            goal_method = agent.get('goal_method', '')
            init_dist = agent.get('initial_distance', 0.0)
            geo_dist = agent.get('geodesic_distance', None)
            step_count = agent.get('step_count', 0)
            region_scenarios = agent.get('region_scenarios', [])

            method_label = _METHOD_LABEL.get(goal_method, goal_method or '?')
            scenario_strs = [_SCENARIO_LABEL.get(sc, sc)
                             for sc in region_scenarios]

            lines = [f'env {env_idx}  {town} ({quadrant})']
            dist_str = f'd={init_dist:.1f}m'
            if geo_dist is not None and np.isfinite(geo_dist):
                dist_str += f'  geo={geo_dist:.1f}m'
            lines.append(f'goal: {method_label}  {dist_str}')
            if scenario_strs:
                lines.append('obs: ' + ', '.join(scenario_strs))
            else:
                lines.append('obs: none')
            lines.append(f'step: {step_count}')

            info_text = '\n'.join(lines)
            ax.text(4, 4, info_text, fontsize=5.5, color='white',
                    alpha=0.85, family='monospace',
                    verticalalignment='top', zorder=10,
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='#0D0D1A', edgecolor='none',
                              alpha=0.7))
        else:
            # No agent metadata — minimal label
            ax.text(5, 15, f'env {env_idx}', fontsize=7, color='white',
                    alpha=0.7, zorder=10)

    # hide unused axes
    for idx in range(num_envs, grid_rows * grid_cols):
        r, c = divmod(idx, grid_cols)
        axes[r][c].set_visible(False)

    # ── legend ────────────────────────────────────────────────────────
    # Check if segmentation data is present
    has_seg = any(layout.get('sidewalk_tris_std') is not None
                  and len(layout.get('sidewalk_tris_std', [])) > 0
                  for layout in obstacle_layouts)

    legend_handles = []
    if has_seg:
        legend_handles.extend([
            Patch(facecolor=SEG_ROAD_COLOR, edgecolor='none',
                  label='road'),
            Patch(facecolor=SEG_SIDEWALK_COLOR, edgecolor='none',
                  label='sidewalk'),
            Patch(facecolor=SEG_CROSSWALK_COLOR, edgecolor='none',
                  label='crosswalk'),
        ])
    else:
        legend_handles.append(
            Patch(facecolor=CROSSWALK_COLOR, edgecolor='white',
                  alpha=CROSSWALK_ALPHA, label='crosswalk'))

    # Check if any crosswalk is marked as blocked
    has_blocked_cw = any(
        any(isinstance(cw, dict) and cw.get('blocked', False)
            for cw in l.get('crosswalks', []))
        for l in obstacle_layouts)
    if has_blocked_cw:
        legend_handles.append(
            Patch(facecolor=CROSSWALK_BLOCKED_COLOR, edgecolor='white',
                  alpha=CROSSWALK_ALPHA, label='blocked crosswalk'))

    legend_handles.extend([
        Patch(facecolor='#E53935', edgecolor='white',
              alpha=OBSTACLE_ALPHA, label='blocker / CW challenge'),
        Patch(facecolor='#FB8C00', edgecolor='white',
              alpha=OBSTACLE_ALPHA, label='barrier'),
        Patch(facecolor='#FFB300', edgecolor='white',
              alpha=OBSTACLE_ALPHA, label='narrow passage'),
        Patch(facecolor='#FDD835', edgecolor='white',
              alpha=OBSTACLE_ALPHA, label='sidewalk clutter'),
        Line2D([], [], marker='^', color='none', markerfacecolor=EGO_COLOR,
               markeredgecolor='white', markersize=8, label='ego'),
        Line2D([], [], marker='*', color='none', markerfacecolor=GOAL_COLOR,
               markeredgecolor='white', markersize=10, label='goal'),
    ])
    has_geo = any(
        any(a.get('geodesic_path_std') is not None
            for a in l.get('agents', []))
        for l in obstacle_layouts)
    if has_geo:
        legend_handles.append(
            Line2D([], [], color='#00E5FF', linewidth=1.5,
                   alpha=0.7, label='geodesic path'))
    has_peds = any(len(l.get('pedestrians', [])) > 0
                   for l in obstacle_layouts)
    if has_peds:
        legend_handles.append(
            Line2D([], [], color=PED_TRAJ_COLOR, linewidth=1.5,
                   marker='o', markersize=4, markerfacecolor=PED_MARKER_COLOR,
                   markeredgecolor=PED_TRAJ_COLOR, markeredgewidth=0.6,
                   label='pedestrian'))
    # Crosswalk axes (only when enriched crosswalk data is present)
    has_cw_axes = any(
        any(isinstance(cw, dict) and cw.get('long_axis_std') is not None
            for cw in l.get('crosswalks', []))
        for l in obstacle_layouts)
    if has_cw_axes:
        legend_handles.extend([
            Line2D([], [], color='#FF6F00', linewidth=1.5,
                   alpha=0.8, label='CW long axis'),
            Line2D([], [], color='#00E676', linewidth=1.5,
                   alpha=0.8, label='CW cross axis'),
        ])

    ncol = min(len(legend_handles), 8)
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=ncol, fontsize=8, framealpha=0.8,
               borderpad=0.4, handletextpad=0.3, columnspacing=0.8,
               facecolor='#2A2A3E', edgecolor='gray', labelcolor='white')

    n_obs = sum(len(l.get('obstacles', [])) for l in obstacle_layouts)
    n_cw = sum(len(l.get('crosswalks', [])) for l in obstacle_layouts)
    n_ped = sum(len(l.get('pedestrians', [])) for l in obstacle_layouts)
    print(f"  [Obstacle BEV] {n_obs} obstacles, {n_cw} crosswalk polygons, "
          f"{n_ped} pedestrians across {num_envs} envs")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"obstacle_bev_{iteration:04d}.png")
        fig.savefig(path, dpi=120, facecolor=BG_COLOR)
        print(f"  -> Obstacle BEV figure saved to {path}")

    if wandb is not None:
        wandb.log({"rollout/obstacle_bev": wandb.Image(fig)}, step=iteration)

    plt.close(fig)


# ─── Ego-View Video ───────────────────────────────────────────────────


def _annotate_ego_frames(frames, rewards=None, dones=None, goals=None,
                         terminateds=None, mpc_vis_data=None, n_skips=None):
    """
    Add per-step overlays to a sequence of RGB frames.

    Overlays
    --------
    * Top-left text  : step index, per-step reward, cumulative reward,
      goal distance.
    * Green tint + "SUCCESS" on terminated steps (goal reached).
    * Red tint + "TRUNCATED" on truncated steps (time limit).
    * Top-right minimap: ego-centric bird's-eye square showing the agent
      (centre, facing up) and goal position with distance rings.
    * Goal projection onto the ego view (pinhole model).
    * Policy waypoints projected onto image (green dots+line).
    * MPC trajectory projected onto image (cyan line), when available.

    Parameters
    ----------
    frames      : (T, H, W, 3) uint8
    rewards     : (T,) float  or None
    dones       : (T,) float  or None  - 1.0 at any episode boundary
    goals       : (T, 2) float or None - (goal_x_std, goal_z_std), relative
    terminateds : (T,) float  or None  - 1.0 only at true terminations (success)
    mpc_vis_data : list[dict] or None  - per *policy-step* MPC/policy data;
        each dict may contain 'policy_waypoints_cam' (F,2),
        'mpc_x_sol' (H+1,3), 'mpc_u_sol' (H,2).
    n_skips     : int or None  - substep expansion factor.  When substep
        frames are used, T = num_policy_steps * n_skips; MPC data is drawn
        only on the first substep of each policy step.

    Returns : (T, H, W, 3) uint8
    """

    T, H, W = frames.shape[:3]
    out = []
    cum_r = 0.0

    for t in range(T):
        # ── base frame ────────────────────────────────────────────────
        arr = frames[t].copy()
        is_done = dones is not None and dones[t] > 0
        is_terminated = (terminateds is not None and terminateds[t] > 0)
        is_truncated = is_done and not is_terminated

        # colour tint: green for success, red for truncation
        if is_done:
            arr = arr.astype(np.float32)
            if is_terminated:
                arr[..., 0] = np.clip(arr[..., 0] * 0.4,        0, 255)
                arr[..., 1] = np.clip(arr[..., 1] * 0.4 + 153,  0, 255)
                arr[..., 2] = np.clip(arr[..., 2] * 0.4,        0, 255)
            else:
                arr[..., 0] = np.clip(arr[..., 0] * 0.4 + 153,  0, 255)
                arr[..., 1] = np.clip(arr[..., 1] * 0.4,        0, 255)
                arr[..., 2] = np.clip(arr[..., 2] * 0.4,        0, 255)
            arr = arr.astype(np.uint8)

        img  = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)

        # ── text overlay ──────────────────────────────────────────────
        lines = [f"step {t + 1:3d}/{T}"]
        if rewards is not None:
            r = float(rewards[t])
            cum_r += r
            lines.append(f"r   = {r:+.3f}")
            lines.append(f"cum = {cum_r:+.2f}")

        # always show distance to goal
        if goals is not None:
            gx = float(goals[t, 0])
            gz = float(goals[t, 1])
            dist = math.sqrt(gx ** 2 + gz ** 2) + 1e-8
            lines.append(f"dist= {dist:.1f}m")
        else:
            dist = None

        if is_done:
            if is_terminated:
                lines.append("SUCCESS")
            else:
                lines.append("TRUNCATED")
            cum_r = 0.0   # reset AFTER including terminal reward in display

        for i, line in enumerate(lines):
            y = 4 + i * 15
            draw.text((5, y), line, fill=(0, 0, 0))       # shadow
            draw.text((4, y - 1), line, fill=(255, 255, 0))

        # ── goal projection onto ego view ────────────────────────────────
        # Project the goal onto the image plane using a pinhole model.
        # Standard coords: x = right, y = up, z = forward.
        # Camera intrinsics: 640×480, 110° horizontal FOV (stack_omr4.json).
        # Camera height ~0.73 m above ground → ground is at y = -0.73 in cam frame.
        if goals is not None:
            # pinhole focal length (horizontal FOV)
            fov_h_rad = math.radians(110.0)
            fx = W / (2.0 * math.tan(fov_h_rad / 2.0))
            cam_height = 0.73  # metres above ground

            # project goal onto image if in front of camera
            if gz > 0.5:
                u = fx * gx / gz + W / 2.0
                v = fx * cam_height / gz + H / 2.0  # ground plane → v > H/2

                # clamp u to frame bounds for the marker
                u_clamped = max(0, min(W - 1, u))
                in_frame = 0 <= u < W

                color = (0, 255, 100) if in_frame else (255, 80, 80)

                # vertical guide line (dashed effect via short segments)
                for yy in range(0, H, 8):
                    draw.line([(u_clamped, yy), (u_clamped, min(yy + 4, H))],
                              fill=(*color, 120), width=1)

                # diamond marker at projected ground point (or frame edge)
                mv = min(max(v, 0), H - 1)
                r_diamond = 6
                draw.polygon([
                    (u_clamped, mv - r_diamond),
                    (u_clamped + r_diamond, mv),
                    (u_clamped, mv + r_diamond),
                    (u_clamped - r_diamond, mv),
                ], fill=color, outline=(255, 255, 255))

                # distance label near the marker
                dist_txt = f"{dist:.0f}m"
                draw.text((u_clamped + r_diamond + 2, mv - 7), dist_txt,
                          fill=(0, 0, 0))
                draw.text((u_clamped + r_diamond + 1, mv - 8), dist_txt,
                          fill=color)

            # ── ego-centric minimap (top-right) ──────────────────────────
            # Square minimap with the agent at centre facing up (+z).
            # Goal shown as a dot at correct relative position.
            # Distance rings provide scale reference.
            ms = 80              # minimap size (px)
            margin = 6
            mx = W - ms - margin # top-left x of minimap
            my = margin          # top-left y of minimap

            # semi-transparent dark background
            overlay = Image.new('RGBA', (ms, ms), (20, 20, 30, 200))
            overlay_draw = ImageDraw.Draw(overlay)

            # auto-scale: fit goal with 20% margin, minimum 5m radius
            map_radius = max(dist * 1.2, 5.0)
            scale = (ms / 2.0 - 4) / map_radius   # px per metre

            centre = ms / 2.0

            # distance rings (concentric circles for scale)
            ring_distances = _minimap_ring_distances(map_radius)
            for rd in ring_distances:
                rr = rd * scale
                if rr < 3 or rr > centre - 2:
                    continue
                overlay_draw.ellipse(
                    [centre - rr, centre - rr, centre + rr, centre + rr],
                    outline=(80, 80, 80, 160), width=1)
                # distance label on the ring (right side)
                label = f"{rd:.0f}" if rd >= 1 else f"{rd:.1f}"
                overlay_draw.text((centre + rr - 8, centre - 10),
                                  label, fill=(120, 120, 120, 200))

            # agent marker: small upward triangle at centre
            a_size = 5
            overlay_draw.polygon([
                (centre, centre - a_size - 1),
                (centre - a_size, centre + a_size - 1),
                (centre + a_size, centre + a_size - 1),
            ], fill=(255, 255, 255, 230), outline=(200, 200, 200, 255))

            # agent forward indicator: thin line upward from agent
            overlay_draw.line(
                [(centre, centre - a_size - 1), (centre, centre - a_size - 6)],
                fill=(100, 255, 100, 200), width=1)

            # goal marker: bright dot at relative position
            # In ego frame: x = right, z = forward → minimap: u = right, v = up
            goal_u = centre + gx * scale
            goal_v = centre - gz * scale   # forward → up in image
            goal_r = 5
            # pulsing glow effect: outer ring
            overlay_draw.ellipse(
                [goal_u - goal_r - 2, goal_v - goal_r - 2,
                 goal_u + goal_r + 2, goal_v + goal_r + 2],
                fill=(255, 200, 50, 80))
            # solid goal dot
            overlay_draw.ellipse(
                [goal_u - goal_r, goal_v - goal_r,
                 goal_u + goal_r, goal_v + goal_r],
                fill=(255, 200, 50, 255), outline=(255, 255, 255, 255))

            # border
            overlay_draw.rectangle(
                [0, 0, ms - 1, ms - 1],
                outline=(180, 180, 180, 200), width=1)

            # composite minimap onto frame
            img.paste(
                Image.fromarray(
                    np.array(overlay.convert('RGB'), dtype=np.uint8)),
                (mx, my),
                mask=overlay.split()[3])

            # distance text below minimap
            draw = ImageDraw.Draw(img)  # refresh draw after paste
            dist_label = f"{dist:.1f}m"
            tw = draw.textlength(dist_label) if hasattr(draw, 'textlength') else len(dist_label) * 6
            tx = mx + ms / 2 - tw / 2
            ty = my + ms + 2
            draw.text((tx + 1, ty + 1), dist_label, fill=(0, 0, 0))
            draw.text((tx, ty), dist_label, fill=(255, 200, 50))

        # ── MPC/policy waypoint overlay ──────────────────────────────
        # Draw on the first substep of each policy step (or every frame
        # when n_skips is None / 1).
        if mpc_vis_data is not None:
            _nskip = n_skips if n_skips is not None else 1
            policy_step = t // _nskip
            is_first_substep = (t % _nskip == 0)
            if is_first_substep and policy_step < len(mpc_vis_data):
                _mpc_entry = mpc_vis_data[policy_step]
                draw = ImageDraw.Draw(img)

                # Pinhole projection: camera-frame (x_cam, y_cam, z_cam) → image
                # Camera: 640×480, 110° horizontal FOV (stack_omr4.json)
                _fov_h = math.radians(110.0)
                _fx = W / (2.0 * math.tan(_fov_h / 2.0))
                _cam_h = 0.73  # camera height above ground

                def _project_cam_xz(xz_arr):
                    """Project (N, 2) camera-frame xz waypoints to image pixels.
                    Returns (N, 2) array of (u, v) and (N,) bool validity mask."""
                    pts = np.asarray(xz_arr)
                    x_c, z_c = pts[:, 0], pts[:, 1]
                    valid = z_c > 0.1
                    u = np.where(valid, _fx * x_c / z_c + W / 2.0, 0.0)
                    v = np.where(valid, _fx * _cam_h / z_c + H / 2.0, 0.0)
                    return np.stack([u, v], axis=-1), valid

                # Policy waypoints (green)
                wp = _mpc_entry.get('policy_waypoints_cam')
                if wp is not None and len(wp) >= 2:
                    uv, valid = _project_cam_xz(wp)
                    pts_uv = [(float(uv[j, 0]), float(uv[j, 1]))
                              for j in range(len(uv)) if valid[j]]
                    if len(pts_uv) >= 2:
                        draw.line(pts_uv, fill=(50, 220, 50, 200), width=2)
                    for pu, pv in pts_uv:
                        r = 3
                        draw.ellipse([pu - r, pv - r, pu + r, pv + r],
                                     fill=(50, 220, 50), outline=(255, 255, 255))

                # MPC open-loop trajectory (cyan)
                x_sol = _mpc_entry.get('mpc_x_sol')
                if x_sol is not None and len(x_sol) >= 2:
                    # MPC state: (x_mpc, y_mpc, theta) where x=right, y=forward
                    mpc_xz = x_sol[:, [0, 1]]  # (x_cam, z_cam)
                    uv, valid = _project_cam_xz(mpc_xz)
                    pts_uv = [(float(uv[j, 0]), float(uv[j, 1]))
                              for j in range(len(uv)) if valid[j]]
                    if len(pts_uv) >= 2:
                        draw.line(pts_uv, fill=(0, 230, 255, 200), width=2)

        out.append(np.array(img, dtype=np.uint8))

    return np.array(out, dtype=np.uint8)


def log_mpc_dashboard(
    buf_cmd_speed, buf_real_speed, buf_dones,
    mpc_vis_data=None,
    grid_cols=None,
    iteration=0,
    save_dir=None,
    wandb=None,
):
    """
    Render a per-env MPC/policy dashboard with time-series panels.

    Panels per env (one column per env):
      Row 0: Speed — commanded vs real
      Row 1: MPC velocity commands — linear (v) and angular (omega)

    Parameters
    ----------
    buf_cmd_speed  : (num_steps, num_envs) float
    buf_real_speed : (num_steps, num_envs) float
    buf_dones      : (num_steps, num_envs) float
    mpc_vis_data   : list[list[dict]] or None — per-env, per-step MPC data
    """

    num_steps, num_envs = buf_cmd_speed.shape
    if grid_cols is None:
        grid_cols = math.ceil(math.sqrt(num_envs))
    grid_rows = math.ceil(num_envs / grid_cols)

    n_panel_rows = 2  # speed, velocity commands
    fig, axes = plt.subplots(
        n_panel_rows * grid_rows, grid_cols,
        figsize=(4.5 * grid_cols, 2.5 * n_panel_rows * grid_rows),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.45, wspace=0.3)

    for env_idx in range(num_envs):
        grid_r, grid_c = divmod(env_idx, grid_cols)
        base_row = grid_r * n_panel_rows
        steps = np.arange(num_steps)

        # ── Row 0: Speed ──
        ax_speed = axes[base_row][grid_c]
        ax_speed.plot(steps, buf_cmd_speed[:, env_idx],
                      label='cmd', color='tab:blue', linewidth=0.8, alpha=0.7)
        ax_speed.plot(steps, buf_real_speed[:, env_idx],
                      label='real', color='tab:blue', linewidth=1.2)
        # Mark episode boundaries
        ep_bounds = np.where(buf_dones[:-1, env_idx] > 0)[0]
        for eb in ep_bounds:
            ax_speed.axvline(eb, color='gray', linewidth=0.5, alpha=0.4)
        ax_speed.set_ylabel('m/s', fontsize=8)
        ax_speed.set_title(f'env {env_idx} — speed', fontsize=9)
        ax_speed.legend(fontsize=7, ncol=2, loc='upper right')
        ax_speed.tick_params(labelsize=7)
        ax_speed.grid(True, alpha=0.3)

        # ── Row 1: MPC velocity commands (v, omega) ──
        ax_vel = axes[base_row + 1][grid_c]
        if (mpc_vis_data is not None
                and env_idx < len(mpc_vis_data)
                and len(mpc_vis_data[env_idx]) > 0):
            lin_vs, ang_vs = [], []
            for entry in mpc_vis_data[env_idx]:
                u_sol = entry.get('mpc_u_sol')
                if u_sol is not None and len(u_sol) > 0:
                    lin_vs.append(float(u_sol[0, 0]))
                    ang_vs.append(float(u_sol[0, 1]))
                else:
                    lin_vs.append(0.0)
                    ang_vs.append(0.0)
            t_mpc = np.arange(len(lin_vs))
            ax_vel.plot(t_mpc, lin_vs, label='v (m/s)',
                        color='tab:purple', linewidth=1.0)
            ax_vel.plot(t_mpc, ang_vs, label='\u03c9 (rad/s)',
                        color='tab:brown', linewidth=1.0)
            for eb in ep_bounds:
                ax_vel.axvline(eb, color='gray', linewidth=0.5, alpha=0.4)
        ax_vel.set_ylabel('cmd', fontsize=8)
        ax_vel.set_title(f'env {env_idx} — MPC velocity', fontsize=9)
        ax_vel.legend(fontsize=7, ncol=2, loc='upper right')
        ax_vel.tick_params(labelsize=7)
        ax_vel.grid(True, alpha=0.3)
        ax_vel.set_xlabel('step', fontsize=8)

    # Hide unused axes
    for idx in range(num_envs, grid_rows * grid_cols):
        r, c = divmod(idx, grid_cols)
        for pr in range(n_panel_rows):
            axes[r * n_panel_rows + pr][c].set_visible(False)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"mpc_dashboard_{iteration:04d}.png")
        fig.savefig(path, dpi=120, bbox_inches='tight')
        print(f"  → MPC dashboard saved to {path}")

    if wandb is not None:
        wandb.log({"rollout/mpc_dashboard": wandb.Image(fig)})

    plt.close(fig)


def _render_mpc_detail_frame(entry, step_idx, cmd_speed, real_speed,
                             speed_history_len=32, fov_deg=110.0):
    """Render a single MPC detail frame as an RGBA numpy array.

    Panels (1 row, 3 columns):
      0 — Camera-frame BEV: policy waypoints, MPC trajectory, FOV cone
      1 — Z-coordinate over MPC horizon: open-loop vs waypoint reference
      2 — Velocity commands over MPC horizon: v and omega

    Returns (H, W, 3) uint8.
    """
    wp_cam = entry.get('policy_waypoints_cam')   # (F, 2) camera frame
    x_sol = entry.get('mpc_x_sol')               # (horizon+1, 3)
    u_sol = entry.get('mpc_u_sol')               # (horizon, 2)

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    fig.subplots_adjust(wspace=0.38, left=0.06, right=0.97,
                        top=0.86, bottom=0.16)

    # ── Panel 0: Camera-frame BEV ────────────────────────────────────
    ax_bev = axes[0]
    fov_rad = math.radians(fov_deg)
    half_fov = fov_rad / 2.0
    r_line = np.linspace(0., 7., 80)
    ax_bev.plot(r_line * math.cos(math.pi / 2 - half_fov),
                r_line * math.sin(math.pi / 2 - half_fov),
                '--', color='gray', linewidth=0.7, alpha=0.5)
    ax_bev.plot(-r_line * math.cos(math.pi / 2 - half_fov),
                r_line * math.sin(math.pi / 2 - half_fov),
                '--', color='gray', linewidth=0.7, alpha=0.5)

    # Policy waypoints
    if wp_cam is not None and len(wp_cam) >= 1:
        ax_bev.plot(wp_cam[:, 0], wp_cam[:, 1], 'o-',
                    color='tab:green', markersize=4, linewidth=2,
                    label='waypoints', zorder=3)

    # MPC open-loop trajectory
    if x_sol is not None and len(x_sol) >= 2:
        mpc_xz = x_sol[:, [0, 1]]  # (x_cam, z_cam)
        ax_bev.plot(mpc_xz[:, 0], mpc_xz[:, 1],
                    color='teal', linewidth=2.5,
                    label='MPC open-loop', zorder=4)

    # Origin marker (ego)
    ax_bev.scatter([0], [0], marker='^', s=60, c='white',
                   edgecolors='black', zorder=5)

    ax_bev.set_aspect('equal')
    ax_bev.grid(True, alpha=0.3)
    ax_bev.set_xlabel('x (m)', fontsize=8)
    ax_bev.set_ylabel('z (m)', fontsize=8)
    ax_bev.set_title(f'cam-frame BEV  [step {step_idx}]', fontsize=9)
    ax_bev.legend(fontsize=6, loc='upper left')
    ax_bev.tick_params(labelsize=7)

    # ── Panel 1: Z-coordinate over horizon ───────────────────────────
    ax_z = axes[1]
    if x_sol is not None:
        horizon = len(x_sol) - 1
        ax_z.plot(np.arange(horizon + 1), x_sol[:, 1],
                  color='teal', linewidth=1.5, label='MPC open-loop')
    if wp_cam is not None:
        n_wp = len(wp_cam)
        if x_sol is not None:
            # Waypoints are at indices 1, 1+n_skips, 1+2*n_skips, ...
            n_skips = max(1, horizon // n_wp) if x_sol is not None else 1
            wp_t = np.arange(1, n_wp + 1) * n_skips
            wp_t = np.clip(wp_t, 0, horizon)
        else:
            wp_t = np.arange(1, n_wp + 1)
        ax_z.plot(wp_t, wp_cam[:, 1], '--', color='teal',
                  linewidth=1.2, alpha=0.7, label='waypoint ref')
    ax_z.grid(True, alpha=0.3)
    ax_z.set_xlabel('MPC step', fontsize=8)
    ax_z.set_ylabel('z (m)', fontsize=8)
    ax_z.set_title('z over horizon', fontsize=9)
    ax_z.legend(fontsize=6)
    ax_z.tick_params(labelsize=7)

    # ── Panel 2: Velocity commands over horizon ──────────────────────
    ax_v = axes[2]
    if u_sol is not None and len(u_sol) >= 1:
        t_u = np.arange(len(u_sol))
        ax_v.plot(t_u, u_sol[:, 0], color='tab:purple',
                  linewidth=1.2, label='v (m/s)')
        ax_v.plot(t_u, u_sol[:, 1], color='tab:brown',
                  linewidth=1.2, label='\u03c9 (rad/s)')
    ax_v.grid(True, alpha=0.3)
    ax_v.set_xlabel('MPC step', fontsize=8)
    ax_v.set_title('velocity commands', fontsize=9)
    ax_v.legend(fontsize=6, ncol=2, loc='upper right')
    ax_v.tick_params(labelsize=7)

    # ── Rasterise to numpy ───────────────────────────────────────────
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    frame = buf[:, :, :3].copy()
    plt.close(fig)
    return frame


def log_mpc_detail_video_grid(
    mpc_vis_data,
    buf_cmd_speed=None,
    buf_real_speed=None,
    buf_dones=None,
    grid_cols=None,
    fps=5,
    iteration=0,
    save_dir=None,
    wandb=None,
):
    """
    Render a grid video of per-step MPC planning-horizon details.

    Each cell shows one env's multi-panel figure (camera-frame BEV with
    FOV cone + waypoints + MPC trajectory; z open-loop vs reference;
    velocity profile over the planning horizon).  One frame per policy
    step — gives a step-by-step view of what the MPC sees and plans.

    Parameters
    ----------
    mpc_vis_data : list[list[dict]]
        ``mpc_vis_data[env_idx][step]`` is a dict with optional keys
        ``policy_waypoints_cam``, ``mpc_x_sol``, ``mpc_u_sol``.
    buf_cmd_speed, buf_real_speed : (num_steps, num_envs) or None
    buf_dones : (num_steps, num_envs) or None
    """

    num_envs = len(mpc_vis_data)
    num_steps = max((len(d) for d in mpc_vis_data), default=0)
    if num_steps == 0:
        return

    if grid_cols is None:
        grid_cols = math.ceil(math.sqrt(num_envs))
    grid_rows = math.ceil(num_envs / grid_cols)

    # ── Render per-env frame sequences ───────────────────────────────
    # Render one reference frame to get the cell size
    _ref = mpc_vis_data[0][0] if mpc_vis_data[0] else {}
    ref_frame = _render_mpc_detail_frame(_ref, 0, 0.0, 0.0)
    cell_h, cell_w = ref_frame.shape[:2]

    # Pre-allocate grid frames
    grid_H = cell_h * grid_rows
    grid_W = cell_w * grid_cols
    grid_frames = []

    for step in range(num_steps):
        grid_frame = np.zeros((grid_H, grid_W, 3), dtype=np.uint8)
        for env_idx in range(num_envs):
            r, c = divmod(env_idx, grid_cols)
            if step < len(mpc_vis_data[env_idx]):
                entry = mpc_vis_data[env_idx][step]
            else:
                entry = {}

            cs = 0.0
            rs = 0.0
            if buf_cmd_speed is not None and step < buf_cmd_speed.shape[0]:
                cs = float(buf_cmd_speed[step, env_idx])
            if buf_real_speed is not None and step < buf_real_speed.shape[0]:
                rs = float(buf_real_speed[step, env_idx])

            cell = _render_mpc_detail_frame(entry, step, cs, rs)
            # Resize if needed (should match ref_frame)
            y0, x0 = r * cell_h, c * cell_w
            grid_frame[y0:y0 + cell_h, x0:x0 + cell_w] = cell[:cell_h, :cell_w]

        grid_frames.append(grid_frame)

    grid_video = np.array(grid_frames, dtype=np.uint8)

    # ── Write to disk ────────────────────────────────────────────────
    stem = f"mpc_detail_{iteration:04d}"
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.join(save_dir, stem)
        path = _write_video(grid_video, base, fps)
    else:
        
        tmp_dir = tempfile.mkdtemp()
        path = _write_video(grid_video, os.path.join(tmp_dir, stem), fps)

    if path is None:
        return

    print(f"  → MPC detail video ({grid_rows}×{grid_cols}, "
          f"{num_steps} steps) saved to {path}")

    if wandb is not None:
        wandb.log({
            "rollout/mpc_detail_video": wandb.Video(path, fps=fps,
                                                     format="mp4"),
        })


def _minimap_ring_distances(map_radius):
    """Choose clean distance-ring values for the minimap."""
    # Pick 2-3 rings at round intervals
    candidates = [1, 2, 5, 10, 15, 20, 25, 30, 50, 75, 100]
    rings = [c for c in candidates if 0.15 * map_radius < c < 0.95 * map_radius]
    if not rings:
        rings = [map_radius * 0.5]
    return rings[:3]


# ── Geodesic distance field heatmap / video ──────────────────────────

def log_geodesic_field(
    obstacle_layouts,
    bev_metas,
    grid_cols=None,
    iteration=0,
    save_dir=None,
    wandb=None,
    video_fps=5,
):
    """Render geodesic distance field as a heatmap image or time-space video.

    For static / soft-cost fields (2-D), produces a single PNG figure.
    For time-space fields (3-D), produces an MP4 video where each frame
    shows the value field at one time step.

    Parameters
    ----------
    obstacle_layouts : list[dict]
        Per-server layouts from ``get_obstacle_layouts()``.
    bev_metas : list[dict | None]
        Per-env BEV projection metadata.
    """
    if not obstacle_layouts:
        return

    # ── Collect per-env agent data ────────────────────────────────────
    agents_flat = []       # list of per-env agent dicts
    metas_flat = []
    for server_layout in obstacle_layouts:
        if server_layout is None:
            continue
        for ag in server_layout.get('agents', []):
            agents_flat.append(ag)
    num_envs = len(agents_flat)
    if num_envs == 0:
        return
    metas_flat = (bev_metas or [None] * num_envs)[:num_envs]

    has_3d = any(ag.get('geo_field_3d') is not None for ag in agents_flat)

    if has_3d:
        _log_geodesic_video(agents_flat, metas_flat, grid_cols,
                            iteration, save_dir, wandb, video_fps)
    else:
        fig = _render_geodesic_frame(agents_flat, metas_flat, grid_cols,
                                     t=None)
        if fig is None:
            return
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir,
                        f"geodesic_field_{iteration:04d}.png"), dpi=120)
        if wandb is not None:
            wandb.log({"rollout/geodesic_field": wandb.Image(fig)},
                      step=iteration)
        plt.close(fig)


def _log_geodesic_video(agents, metas, grid_cols, iteration,
                         save_dir, wandb, fps):
    """Assemble a time-space geodesic field video."""
    # Determine max T across agents
    T_max = 0
    for ag in agents:
        f3 = ag.get('geo_field_3d')
        if f3 is not None:
            T_max = max(T_max, f3.shape[0] - 1)
    if T_max == 0:
        return

    frames = []
    for t in range(T_max + 1):
        fig = _render_geodesic_frame(agents, metas, grid_cols, t=t)
        if fig is None:
            continue
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(buf[:, :, :3].copy())
        plt.close(fig)

    if not frames:
        return

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.join(save_dir, f"geodesic_field_{iteration:04d}")
        vid_path = _write_video(frames, base, fps)
    else:
        vid_path = None

    if wandb is not None:
        if vid_path is not None:
            wandb.log({"rollout/geodesic_field_video":
                       wandb.Video(vid_path, fps=fps, format="mp4")},
                      step=iteration)
        else:
            # Fallback: log first frame as image
            fig = _render_geodesic_frame(agents, metas, grid_cols, t=0)
            if fig is not None:
                wandb.log({"rollout/geodesic_field": wandb.Image(fig)},
                          step=iteration)
                plt.close(fig)


def _render_geodesic_frame(agents, metas, grid_cols, t=None):
    """Render one frame of the geodesic field heatmap grid.

    Parameters
    ----------
    agents : list[dict]   — per-env agent dicts with geo data.
    metas  : list[dict]   — per-env BEV metadata.
    grid_cols : int | None
    t : int | None
        Time index into geo_field_3d.  ``None`` means use geo_field_2d.

    Returns
    -------
    fig : matplotlib.Figure or None
    """

    num_envs = len(agents)
    if grid_cols is None:
        grid_cols = math.ceil(math.sqrt(num_envs))
    grid_rows = math.ceil(num_envs / grid_cols)
    cell = 4.5

    BG = '#1A1A2E'
    CMAP = 'plasma_r'

    fig, axes = plt.subplots(grid_rows, grid_cols,
                             figsize=(cell * grid_cols,
                                      cell * grid_rows + 0.4),
                             squeeze=False)
    fig.subplots_adjust(left=0.02, right=0.90, top=0.95, bottom=0.02,
                        wspace=0.05, hspace=0.05)
    fig.patch.set_facecolor(BG)

    if t is not None:
        fig.suptitle(f"t = {t}", color='white', fontsize=12, y=0.98)

    # Shared colour limits across envs for comparability
    vmin, vmax = np.inf, 0.0
    for ag in agents:
        field = _pick_field(ag, t)
        if field is None:
            continue
        finite = field[np.isfinite(field)]
        if finite.size == 0:
            continue
        vmin = min(vmin, float(finite.min()))
        vmax = max(vmax, float(np.percentile(finite, 97)))
    if vmin >= vmax:
        vmin, vmax = 0.0, 1.0

    for idx in range(num_envs):
        row_i, col_i = divmod(idx, grid_cols)
        ax = axes[row_i][col_i]
        ax.set_facecolor(BG)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        ag = agents[idx]
        meta = metas[idx] if idx < len(metas) else None
        gm = ag.get('geo_grid_meta')
        field = _pick_field(ag, t)

        if field is None or gm is None:
            ax.set_visible(False)
            continue

        H, W = gm['H'], gm['W']
        res = gm['resolution']
        x_min, z_min = gm['x_min'], gm['z_min']
        x_max = x_min + W * res
        z_max = z_min + H * res

        # Prepare display array: NaN for unreachable (renders transparent)
        disp = field.astype(np.float64).copy()
        disp[~np.isfinite(disp)] = np.nan

        # imshow with origin='lower' so row 0 = z_min = bottom
        ax.imshow(disp, origin='lower', cmap=CMAP,
                  vmin=vmin, vmax=vmax,
                  extent=[x_min, x_max, z_min, z_max],
                  aspect='equal', interpolation='nearest')

        # Viewport from BEV meta (zoom to match other BEV figures)
        if meta is not None:
            ctr = meta['center_xz']
            alt = meta['altitude']
            fov_rad = np.deg2rad(meta.get('fov_deg', 90.0))
            half = alt * np.tan(fov_rad / 2.0)
            ax.set_xlim(ctr[0] - half, ctr[0] + half)
            ax.set_ylim(ctr[1] - half, ctr[1] + half)
        else:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(z_min, z_max)

        # Overlay: geodesic path (cyan)
        gp = ag.get('geodesic_path_std')
        if gp is not None and len(gp) > 1:
            gp = np.asarray(gp)
            ax.plot(gp[:, 0], gp[:, 1], '-', color='#00E5FF',
                    linewidth=1.5, alpha=0.8, zorder=3)

        # Overlay: ego (green) and goal (blue)
        ego = ag.get('ego_std')
        goal = ag.get('goal_std')
        if ego is not None:
            ax.plot(ego[0], ego[1], '^', color='#43E97B',
                    markersize=8, markeredgecolor='white',
                    markeredgewidth=0.5, zorder=5)
        if goal is not None:
            ax.plot(goal[0], goal[1], '*', color='#667EEA',
                    markersize=10, markeredgecolor='white',
                    markeredgewidth=0.5, zorder=5)

    # Hide unused axes
    for idx in range(num_envs, grid_rows * grid_cols):
        row_i, col_i = divmod(idx, grid_cols)
        axes[row_i][col_i].set_visible(False)

    # Shared colorbar
    sm = ScalarMappable(cmap=CMAP, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('distance to goal (m)', color='white', fontsize=9)
    cb.ax.yaxis.set_tick_params(color='white', labelcolor='white',
                                labelsize=7)

    return fig


def _pick_field(ag, t):
    """Select the right 2-D slice for rendering."""
    if t is not None:
        f3 = ag.get('geo_field_3d')
        if f3 is not None and t < f3.shape[0]:
            return f3[t]
        # Beyond horizon: fall back to 2D static
    return ag.get('geo_field_2d')


def _write_video(frames, base_path, fps):
    """
    Write *frames* (T, H, W, 3) uint8 to disk.

    Tries MP4 first (requires imageio-ffmpeg), falls back to GIF.
    Returns the path of the file actually written, or None on failure.
    """
    

    for ext, kwargs in [
        (".mp4", {"fps": fps, "quality": 7, "macro_block_size": 1}),
        (".gif", {"fps": fps, "loop": 0}),
    ]:
        path = base_path + ext
        try:
            imageio.mimwrite(path, frames, **kwargs)
            return path
        except Exception:
            pass

    print("  [video] could not write MP4 or GIF — "
          "install imageio-ffmpeg for MP4 support.")
    return None


# ─── CCTV video ──────────────────────────────────────────────────────

def log_cctv_video_grid(
    cctv_data,
    buf_cord,
    buf_actions,
    buf_rewards=None,
    buf_dones=None,
    buf_terminateds=None,
    ped_positions=None,
    obstacle_layouts=None,
    fps=5,
    iteration=0,
    save_dir=None,
    wandb=None,
    grid_cols=None,
):
    """Assemble CCTV video grid with progressive trajectory overlays.

    Parameters
    ----------
    cctv_data   : dict from ``VecCarlaMultiAgentEnv.collect_cctv_frames()``
                  with ``'frames'`` (env_idx -> (T,H,W,3)) and ``'specs'``.
    buf_cord    : (num_steps, num_envs, context_size*2) positions
    buf_actions : (num_steps, num_envs, action_dim)  camera-frame waypoints
    buf_rewards : (num_steps, num_envs) float or None
    buf_dones   : (num_steps, num_envs) float or None
    buf_terminateds : (num_steps, num_envs) float or None
    ped_positions : list[list[ndarray]]  per-env, per-step (N_ped, 2) std or None
    obstacle_layouts : list[dict] or None  per-env layout from get_obstacle_layouts
    fps         : output video fps
    iteration   : PPO iteration number
    save_dir    : directory for saved files
    wandb       : wandb run object or None
    grid_cols   : columns in the video grid
    """

    frames_dict = cctv_data.get('frames', {})
    specs_list = cctv_data.get('specs', [])
    if not frames_dict or not specs_list:
        return

    # Build spec lookup by env_idx
    spec_by_env = {s['agent_idx']: s for s in specs_list}

    # Determine envs that have CCTV frames
    env_ids = sorted(frames_dict.keys())
    if not env_ids:
        return

    num_steps = buf_cord.shape[0]

    if grid_cols is None:
        grid_cols = math.ceil(math.sqrt(len(env_ids)))
    grid_rows = math.ceil(len(env_ids) / grid_cols)

    # ── per-env annotated frame sequences ──
    env_frame_seqs = []

    for env_idx in env_ids:
        raw = frames_dict[env_idx]
        spec = spec_by_env.get(env_idx)
        if raw is None or spec is None:
            continue

        T_cctv = raw.shape[0]
        T = min(T_cctv, num_steps)
        img_size = spec['img_size']

        annotated = []
        # Ego trajectory accumulator (reset on episode boundary)
        ego_trail = []
        cum_r = 0.0

        for t in range(T):
            arr = raw[t].copy()
            is_done = buf_dones is not None and buf_dones[t, env_idx] > 0
            is_terminated = (buf_terminateds is not None
                             and buf_terminateds[t, env_idx] > 0)

            # ── episode boundary tint ──
            if is_done:
                arr = arr.astype(np.float32)
                if is_terminated:
                    arr[..., 0] = np.clip(arr[..., 0] * 0.5, 0, 255)
                    arr[..., 1] = np.clip(arr[..., 1] * 0.5 + 128, 0, 255)
                    arr[..., 2] = np.clip(arr[..., 2] * 0.5, 0, 255)
                else:
                    arr[..., 0] = np.clip(arr[..., 0] * 0.5 + 128, 0, 255)
                    arr[..., 1] = np.clip(arr[..., 1] * 0.5, 0, 255)
                    arr[..., 2] = np.clip(arr[..., 2] * 0.5, 0, 255)
                arr = arr.astype(np.uint8)

            img = Image.fromarray(arr)
            draw = ImageDraw.Draw(img)

            # Current ego position (standard xz)
            ego_xz = buf_cord[t, env_idx, -2:]
            ego_trail.append(ego_xz.copy())

            # ── goal marker ──
            if obstacle_layouts is not None and env_idx < len(obstacle_layouts):
                layout = obstacle_layouts[env_idx]
                agents = layout.get('agents', []) if layout else []
                aidx = env_idx % len(agents) if agents else -1
                ag = agents[aidx] if 0 <= aidx < len(agents) else None
                if ag is not None:
                    goal_std = ag.get('goal_std')
                    if goal_std is not None:
                        guv, gvis = _cctv_world_to_pixel(
                            np.array(goal_std).reshape(1, 2), spec)
                        if gvis[0]:
                            gu, gv = float(guv[0, 0]), float(guv[0, 1])
                            r = 7
                            draw.polygon([
                                (gu, gv - r), (gu + r * 0.7, gv + r * 0.5),
                                (gu - r * 0.7, gv + r * 0.5),
                            ], fill=(50, 120, 255), outline=(255, 255, 255))

                    # Geodesic path (thin cyan line)
                    geo = ag.get('geodesic_path_std')
                    if geo is not None and len(geo) >= 2:
                        geo_arr = np.asarray(geo)
                        guv, gvis = _cctv_world_to_pixel(geo_arr, spec)
                        pts = [(float(guv[j, 0]), float(guv[j, 1]))
                               for j in range(len(guv)) if gvis[j]]
                        if len(pts) >= 2:
                            draw.line(pts, fill=(0, 200, 200, 100), width=1)

            # ── ego trajectory up to t (green) ──
            if len(ego_trail) >= 2:
                trail_arr = np.array(ego_trail)
                tuv, tvis = _cctv_world_to_pixel(trail_arr, spec)
                pts = [(float(tuv[j, 0]), float(tuv[j, 1]))
                       for j in range(len(tuv)) if tvis[j]]
                if len(pts) >= 2:
                    draw.line(pts, fill=(50, 220, 50), width=2)

            # Ego current position (green dot)
            euv, evis = _cctv_world_to_pixel(
                ego_xz.reshape(1, 2), spec)
            if evis[0]:
                eu, ev = float(euv[0, 0]), float(euv[0, 1])
                r = 4
                draw.ellipse([eu - r, ev - r, eu + r, ev + r],
                             fill=(50, 220, 50), outline=(255, 255, 255))

            # ── ego action waypoints at step t (orange) ──
            if buf_actions is not None:
                # Actions are camera-frame waypoints (future_length, 2)
                # Convert to world frame using ego pose
                act = buf_actions[t, env_idx]
                wp_cam = act.reshape(-1, 2)           # (F, 2): x_cam, z_cam

                # Simple conversion: rotate camera-frame to world frame
                # The camera faces along the agent's forward direction.
                # We approximate the agent heading from consecutive positions.
                if t > 0:
                    prev_xz = buf_cord[t - 1, env_idx, -2:]
                    delta = ego_xz - prev_xz
                    heading = math.atan2(delta[0], delta[1])  # atan2(dx, dz)
                else:
                    heading = 0.0

                cos_h, sin_h = math.cos(heading), math.sin(heading)
                # Rotate each waypoint from camera frame to world frame
                # cam: x=right, z=forward -> world: x=right, z=forward
                # rotated by heading
                wp_world = np.zeros_like(wp_cam)
                wp_world[:, 0] = (cos_h * wp_cam[:, 0]
                                  + sin_h * wp_cam[:, 1]) + ego_xz[0]
                wp_world[:, 1] = (-sin_h * wp_cam[:, 0]
                                  + cos_h * wp_cam[:, 1]) + ego_xz[1]

                wuv, wvis = _cctv_world_to_pixel(wp_world, spec)
                # Connect ego to first waypoint, then waypoints
                wp_pts = [(eu, ev)] if evis[0] else []
                wp_pts += [(float(wuv[j, 0]), float(wuv[j, 1]))
                           for j in range(len(wuv)) if wvis[j]]
                if len(wp_pts) >= 2:
                    draw.line(wp_pts, fill=(255, 160, 30), width=2)
                for j in range(len(wuv)):
                    if wvis[j]:
                        wu, wv = float(wuv[j, 0]), float(wuv[j, 1])
                        r = 3
                        draw.ellipse([wu - r, wv - r, wu + r, wv + r],
                                     fill=(255, 160, 30),
                                     outline=(255, 255, 255))

            # ── pedestrian trajectories up to t (magenta) ──
            if ped_positions is not None and env_idx < len(ped_positions):
                env_peds = ped_positions[env_idx]
                if t < len(env_peds) and env_peds[t] is not None:
                    curr_ped = env_peds[t]   # (N_ped, 2)
                    if len(curr_ped) > 0:
                        # Build per-pedestrian trail from steps 0..t
                        n_peds = curr_ped.shape[0]
                        for pi in range(n_peds):
                            trail = []
                            for s in range(t + 1):
                                if (s < len(env_peds)
                                        and env_peds[s] is not None
                                        and pi < env_peds[s].shape[0]):
                                    trail.append(env_peds[s][pi])
                            if len(trail) >= 2:
                                trail_arr = np.array(trail)
                                puv, pvis = _cctv_world_to_pixel(
                                    trail_arr, spec)
                                pts = [(float(puv[j, 0]),
                                        float(puv[j, 1]))
                                       for j in range(len(puv))
                                       if pvis[j]]
                                if len(pts) >= 2:
                                    draw.line(pts, fill=(220, 50, 220),
                                              width=1)
                            # Current position (white dot)
                            puv, pvis = _cctv_world_to_pixel(
                                curr_ped[pi:pi + 1], spec)
                            if pvis[0]:
                                pu = float(puv[0, 0])
                                pv = float(puv[0, 1])
                                r = 3
                                draw.ellipse(
                                    [pu - r, pv - r, pu + r, pv + r],
                                    fill=(255, 255, 255),
                                    outline=(220, 50, 220))

            # ── text overlay ──
            lines = [f"step {t + 1}/{T}"]
            if buf_rewards is not None:
                r_val = float(buf_rewards[t, env_idx])
                cum_r += r_val
                lines.append(f"r={r_val:+.2f}  cum={cum_r:+.1f}")
            if is_done:
                lines.append("SUCCESS" if is_terminated else "TRUNCATED")
                cum_r = 0.0
                ego_trail = []
            for li, line in enumerate(lines):
                y = 4 + li * 14
                draw.text((5, y), line, fill=(0, 0, 0))
                draw.text((4, y - 1), line, fill=(255, 255, 0))

            # Legend (bottom-left)
            legend_y = img_size - 56
            for label, color in [("ego trail", (50, 220, 50)),
                                 ("waypoints", (255, 160, 30)),
                                 ("pedestrian", (220, 50, 220)),
                                 ("goal", (50, 120, 255))]:
                draw.rectangle([4, legend_y, 14, legend_y + 8],
                               fill=color)
                draw.text((18, legend_y - 2), label, fill=(0, 0, 0))
                draw.text((17, legend_y - 3), label, fill=(255, 255, 255))
                legend_y += 12

            annotated.append(np.array(img, dtype=np.uint8))

        if annotated:
            env_frame_seqs.append(np.stack(annotated))

    if not env_frame_seqs:
        return

    # ── tile into grid ──
    max_T = max(seq.shape[0] for seq in env_frame_seqs)
    H, W = env_frame_seqs[0].shape[1:3]
    grid_H = grid_rows * H
    grid_W = grid_cols * W

    grid_frames = np.zeros((max_T, grid_H, grid_W, 3), dtype=np.uint8)
    for idx, seq in enumerate(env_frame_seqs):
        r, c = divmod(idx, grid_cols)
        T_seq = seq.shape[0]
        grid_frames[:T_seq, r * H:(r + 1) * H, c * W:(c + 1) * W] = seq
        if T_seq < max_T:
            grid_frames[T_seq:, r * H:(r + 1) * H,
                        c * W:(c + 1) * W] = seq[-1]

    # ── write video ──
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.join(save_dir, f"cctv_{iteration:04d}")
        path = _write_video(grid_frames, base, fps)
        if path:
            print(f"  [cctv] saved {path}")
            if wandb is not None:
                try:
                    wandb.log({
                        "rollout/cctv_video": wandb.Video(
                            path, fps=fps, format="mp4"),
                    })
                except Exception as e:
                    print(f"  [cctv] wandb log failed: {e}")


def log_ego_video_grid(
    buf_obs,
    buf_rewards=None,
    buf_dones=None,
    buf_goal=None,
    buf_terminateds=None,
    fps=5,
    iteration=0,
    save_dir=None,
    wandb=None,
    substep_frames=None,
    grid_cols=None,
    mpc_vis_data=None,
):
    """
    Log a single grid video tiling all parallel environments.

    Arranges each environment's ego-camera view into a rows x cols grid,
    producing one combined video per iteration.  For example, 4 servers
    x 4 agents → 16 envs → 4 x 4 grid.

    Parameters are identical to ``log_ego_video`` with the addition of:

    grid_cols : int or None
        Number of columns in the grid.  When *None*, defaults to
        ``ceil(sqrt(num_envs))`` for a roughly square layout.
    """

    num_steps, num_envs = buf_obs.shape[:2]

    if grid_cols is None:
        grid_cols = math.ceil(math.sqrt(num_envs))
    grid_rows = math.ceil(num_envs / grid_cols)

    # ── build per-env annotated frame sequences ───────────────────────
    env_frames = []          # list of (T, H, W, 3) uint8 arrays
    video_fps = fps          # will be updated if substep frames present

    for env_idx in range(num_envs):
        has_substeps = (
            substep_frames is not None
            and len(substep_frames[env_idx]) == num_steps
        )

        if has_substeps:
            all_frames = np.concatenate(substep_frames[env_idx], axis=0)
            n_skips = substep_frames[env_idx][0].shape[0]
            video_fps = fps * n_skips

            T_total = all_frames.shape[0]
            expanded_rewards = None
            expanded_dones = None
            expanded_goals = None
            expanded_terminateds = None

            if buf_rewards is not None:
                expanded_rewards = np.zeros(T_total, dtype=np.float32)
                for s in range(num_steps):
                    expanded_rewards[s * n_skips + n_skips - 1] = buf_rewards[s, env_idx]

            if buf_dones is not None:
                expanded_dones = np.zeros(T_total, dtype=np.float32)
                for s in range(num_steps):
                    expanded_dones[s * n_skips + n_skips - 1] = buf_dones[s, env_idx]

            if buf_terminateds is not None:
                expanded_terminateds = np.zeros(T_total, dtype=np.float32)
                for s in range(num_steps):
                    expanded_terminateds[s * n_skips + n_skips - 1] = buf_terminateds[s, env_idx]

            if buf_goal is not None:
                expanded_goals = np.repeat(
                    buf_goal[:, env_idx], n_skips, axis=0
                )

            # Expand MPC vis data to substep frame count
            _mpc_env = None
            if (mpc_vis_data is not None
                    and env_idx < len(mpc_vis_data)
                    and len(mpc_vis_data[env_idx]) == num_steps):
                _mpc_env = mpc_vis_data[env_idx]

            frames = _annotate_ego_frames(
                all_frames,
                rewards=expanded_rewards,
                dones=expanded_dones,
                goals=expanded_goals,
                terminateds=expanded_terminateds,
                mpc_vis_data=_mpc_env,
                n_skips=n_skips,
            )
        else:
            _mpc_env = None
            if (mpc_vis_data is not None
                    and env_idx < len(mpc_vis_data)
                    and len(mpc_vis_data[env_idx]) == num_steps):
                _mpc_env = mpc_vis_data[env_idx]

            raw = buf_obs[:, env_idx, -1]
            frames = _annotate_ego_frames(
                raw,
                rewards=buf_rewards[:, env_idx] if buf_rewards is not None else None,
                dones=buf_dones[:, env_idx]   if buf_dones   is not None else None,
                goals=buf_goal[:, env_idx]    if buf_goal    is not None else None,
                terminateds=buf_terminateds[:, env_idx] if buf_terminateds is not None else None,
                mpc_vis_data=_mpc_env,
            )

        env_frames.append(frames)

    # ── tile into grid ────────────────────────────────────────────────
    # Different envs may have different frame counts (substep vs policy-step
    # fallback).  Truncate to the minimum T so the grid dimensions match.
    T = min(f.shape[0] for f in env_frames)
    env_frames = [f[:T] for f in env_frames]
    H, W = env_frames[0].shape[1], env_frames[0].shape[2]

    # pad the list with black frames if num_envs doesn't fill the grid
    black = np.zeros((T, H, W, 3), dtype=np.uint8)
    while len(env_frames) < grid_rows * grid_cols:
        env_frames.append(black)

    # assemble: (grid_rows, grid_cols, T, H, W, 3) → (T, grid_rows*H, grid_cols*W, 3)
    rows = []
    for r in range(grid_rows):
        row = np.concatenate(
            env_frames[r * grid_cols : (r + 1) * grid_cols], axis=2   # concat along W
        )
        rows.append(row)
    grid_video = np.concatenate(rows, axis=1)  # concat along H

    # ── write to disk ─────────────────────────────────────────────────
    stem = f"ego_grid_{iteration:04d}"
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.join(save_dir, stem)
        path = _write_video(grid_video, base, video_fps)
    else:
        tmp_dir = tempfile.mkdtemp()
        path = _write_video(grid_video, os.path.join(tmp_dir, stem), video_fps)

    if path is None:
        return

    print(f"  → ego grid video ({grid_rows}×{grid_cols}) saved to {path}")

    if wandb is not None:
        wandb.log(
            {"rollout/ego_video_grid": wandb.Video(path, fps=video_fps)},
            step=iteration,
        )


def log_ego_video(
    buf_obs,
    buf_rewards=None,
    buf_dones=None,
    buf_goal=None,
    fps=5,
    iteration=0,
    save_dir=None,
    wandb=None,
    substep_frames=None,
):
    """
    Log a video of the ego camera view for every parallel environment.

    Parameters
    ----------
    buf_obs     : (num_steps, num_envs, context_size, H, W, 3)  uint8
                  The most-recent frame used per step is buf_obs[:, e, -1].
    buf_rewards : (num_steps, num_envs)  optional - shown as text overlay.
    buf_dones   : (num_steps, num_envs)  optional - triggers red-tint flash.
    buf_goal    : (num_steps, num_envs, 2) optional - goal (dx, dz) in
                  standard coords, used to draw the goal compass.
    fps         : playback frame rate (should match policy step rate).
    save_dir    : directory to write video files; None → temp dir for W&B.
    wandb       : wandb module (or None).
    substep_frames : list[list[ndarray]] or None
                  Per-env list of per-step substep frame arrays.
                  substep_frames[env_idx][step] has shape (n_skips, H, W, 3).
                  When provided, all substep frames are included in the video
                  for smoother playback. Annotations (reward, done, goal) are
                  shown only on the last substep frame of each policy step.
    """


    num_steps, num_envs = buf_obs.shape[:2]

    for env_idx in range(num_envs):
        # ── build frame sequence ──────────────────────────────────────
        has_substeps = (
            substep_frames is not None
            and len(substep_frames[env_idx]) == num_steps
        )

        if has_substeps:
            # Concatenate all substep frames: (num_steps * n_skips, H, W, 3)
            all_frames = np.concatenate(substep_frames[env_idx], axis=0)
            n_skips = substep_frames[env_idx][0].shape[0]
            video_fps = fps * n_skips

            # Expand annotations: repeat per-step values across substep frames,
            # but only mark reward/done on the last substep of each policy step.
            T_total = all_frames.shape[0]
            print(f'total # of frames: {T_total} (fps: {video_fps} / n_skips: {n_skips})')
            expanded_rewards = None
            expanded_dones = None
            expanded_goals = None

            if buf_rewards is not None:
                expanded_rewards = np.zeros(T_total, dtype=np.float32)
                for s in range(num_steps):
                    # Show reward only on the last substep frame
                    expanded_rewards[s * n_skips + n_skips - 1] = buf_rewards[s, env_idx]

            if buf_dones is not None:
                expanded_dones = np.zeros(T_total, dtype=np.float32)
                for s in range(num_steps):
                    # Show done only on the last substep frame
                    expanded_dones[s * n_skips + n_skips - 1] = buf_dones[s, env_idx]

            if buf_goal is not None:
                # Repeat goal for all substep frames within each step
                expanded_goals = np.repeat(
                    buf_goal[:, env_idx], n_skips, axis=0
                )

            frames = _annotate_ego_frames(
                all_frames,
                rewards=expanded_rewards,
                dones=expanded_dones,
                goals=expanded_goals,
            )
        else:
            raw = buf_obs[:, env_idx, -1]  # (T, H, W, 3) – latest frame
            video_fps = fps
            frames = _annotate_ego_frames(
                raw,
                rewards=buf_rewards[:, env_idx] if buf_rewards is not None else None,
                dones=buf_dones[:, env_idx]   if buf_dones   is not None else None,
                goals=buf_goal[:, env_idx]    if buf_goal    is not None else None,
            )

        # ── determine output path ──────────────────────────────────────
        stem = f"ego_env{env_idx}_{iteration:04d}"
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            base = os.path.join(save_dir, stem)
            path = _write_video(frames, base, video_fps)
        else:
            # write to a temp file so wandb.Video can read it
            tmp_dir = tempfile.mkdtemp()
            path = _write_video(frames, os.path.join(tmp_dir, stem), video_fps)

        if path is None:
            continue

        print(f"  -> ego video saved to {path}")

        if wandb is not None:
            wandb.log(
                {f"rollout/ego_video_env{env_idx}": wandb.Video(path, fps=video_fps)},
                step=iteration,
            )


# ─── DINOv2 PCA Feature Visualization ────────────────────────────────


def _extract_patch_tokens(obs_encoder, frames_uint8, no_pos_embed=False):
    """
    Run the DINOv2 backbone on uint8 frames and return spatial patch tokens.

    Parameters
    ----------
    obs_encoder : ObservationEncoder
    frames_uint8 : (N, H, W, 3) uint8 ndarray
    no_pos_embed : bool
        If True, temporarily zero out positional embeddings so the resulting
        features reflect only semantic content without spatial position bias.

    Returns
    -------
    patch_tokens : (N, C, Hp, Wp) float32 tensor  (CPU)
    """
    import torch
    import torchvision.transforms.functional as TF
    from rl.models.encoder import process_frames

    device = next(obs_encoder.parameters()).device

    # process_frames expects (B, N_ctx, H, W, C) or (B, H, W, C)
    imgs = torch.from_numpy(frames_uint8).to(device)       # (N, H, W, 3)
    imgs = process_frames(imgs)                             # (N, 3, 360, 640)

    if obs_encoder.do_rgb_normalize:
        imgs = (imgs - obs_encoder.mean) / obs_encoder.std
    if obs_encoder.do_resize:
        imgs = TF.center_crop(imgs, obs_encoder.crop)
        imgs = TF.resize(imgs, obs_encoder.resize)

    backbone = obs_encoder.obs_encoder

    with torch.no_grad():
        if no_pos_embed and hasattr(backbone, "pos_embed"):
            saved_pos_embed = backbone.pos_embed.data.clone()
            backbone.pos_embed.data.zero_()

        try:
            feats = backbone.get_intermediate_layers(
                imgs, n=1, reshape=True,
            )
            patch_tokens = feats[0]                         # (N, C, Hp, Wp)
        finally:
            if no_pos_embed and hasattr(backbone, "pos_embed"):
                backbone.pos_embed.data.copy_(saved_pos_embed)

    return patch_tokens.float().cpu()


def _pca_visualize_patches(patch_tokens, pca_model=None, skip_components=0):
    """
    Project patch tokens to RGB via PCA.

    Parameters
    ----------
    patch_tokens : (N, C, Hp, Wp) float tensor
    pca_model    : fitted sklearn PCA, or None to fit on the given tokens
    skip_components : int
        Number of leading PCA components to skip before taking 3 for RGB.
        E.g. skip_components=1 fits (1+3)=4 components and uses the last 3,
        discarding the first (position-dominated) component.

    Returns
    -------
    vis_images : list of (Hp, Wp, 3) uint8 ndarrays
    pca_model  : the (possibly newly fitted) PCA object
    """
    from sklearn.decomposition import PCA

    n_total = 3 + skip_components

    N, C, Hp, Wp = patch_tokens.shape
    # Flatten all patches across all images for shared PCA
    all_feats = patch_tokens.permute(0, 2, 3, 1).reshape(-1, C).numpy()  # (N*Hp*Wp, C)

    if pca_model is None:
        pca_model = PCA(n_components=n_total)
        pca_model.fit(all_feats)

    projected = pca_model.transform(all_feats)  # (N*Hp*Wp, n_total)
    projected = projected[:, skip_components:]   # (N*Hp*Wp, 3)

    # Normalize to [0, 255] using robust percentile-based scaling
    for i in range(3):
        ch = projected[:, i]
        lo, hi = np.percentile(ch, [1, 99])
        if hi - lo > 1e-6:
            projected[:, i] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0) * 255.0
        else:
            # Degenerate channel: map to mid-gray instead of black
            projected[:, i] = 128.0

    projected = projected.reshape(N, Hp, Wp, 3).astype(np.uint8)
    return [projected[i] for i in range(N)], pca_model


def log_dino_pca_video_grid(
    obs_encoder, buf_obs,
    fps=5,
    iteration=0,
    save_dir=None,
    wandb=None,
    grid_cols=None,
    pca_batch_size=32,
    substep_frames=None,
    upsample_scale=1.0,
    interpolation="LANCZOS",
    no_pos_embed=False,
    skip_components=3,
):
    """
    Render a grid video of DINOv2 PCA feature maps, one cell per environment,
    mirroring the layout of ``log_ego_video_grid``.

    A shared PCA is fitted once on a subsample of patches from all envs/steps,
    then every frame is projected with the same components so colours are
    consistent across the whole video.

    Parameters
    ----------
    obs_encoder    : ObservationEncoder
    buf_obs        : (num_steps, num_envs, context_size, H, W, 3) uint8
    fps            : playback frame rate
    grid_cols      : columns in the tiled grid (None → ceil(sqrt(num_envs)))
    pca_batch_size : max frames fed through DINOv2 in one forward pass
    substep_frames : list[list[ndarray]] or None
                     Per-env list of per-step substep frame arrays.
                     substep_frames[env_idx][step] has shape (n_skips, H, W, 3).
                     When provided, all substep frames are visualised for
                     smoother playback (matching the RGB ego video).
    upsample_scale : float
                     Scale factor relative to the original image resolution.
                     1.0 = original size, 2.0 = double resolution, etc.
    interpolation  : str
                     PIL resampling filter name: "NEAREST", "BILINEAR",
                     "BICUBIC", or "LANCZOS".  Default "LANCZOS" gives
                     the smoothest output.
    """

    from sklearn.decomposition import PCA
    from PIL import Image as PILImage
    resample_filter = getattr(PILImage, interpolation.upper(), PILImage.LANCZOS)

    num_steps, num_envs = buf_obs.shape[:2]
    H_img, W_img = buf_obs.shape[3], buf_obs.shape[4]  # original RGB size
    H_out = int(H_img * upsample_scale)
    W_out = int(W_img * upsample_scale)

    if grid_cols is None:
        grid_cols = math.ceil(math.sqrt(num_envs))
    grid_rows = math.ceil(num_envs / grid_cols)

    # ── 0. Build per-env frame sequences (handle substep frames) ──────
    # For each env, produce (T_env, H, W, 3) uint8 frames to feed DINOv2.
    video_fps = fps
    env_all_frames = []  # list of (T_env, H, W, 3) uint8

    for e in range(num_envs):
        has_substeps = (
            substep_frames is not None
            and len(substep_frames[e]) == num_steps
        )
        if has_substeps:
            all_f = np.concatenate(substep_frames[e], axis=0)  # (T*n_skips, H, W, 3)
            n_skips = substep_frames[e][0].shape[0]
            video_fps = fps * n_skips
            env_all_frames.append(all_f)
        else:
            env_all_frames.append(buf_obs[:, e, -1])  # (T, H, W, 3)

    # ── 1. Fit shared PCA on a subsample of frames ────────────────────
    T_total = env_all_frames[0].shape[0]
    n_fit = min(T_total, 16)
    fit_step_indices = np.linspace(0, T_total - 1, n_fit, dtype=int)
    fit_frames = []
    for t in fit_step_indices:
        for e in range(num_envs):
            fit_frames.append(env_all_frames[e][t])
    fit_frames = np.stack(fit_frames)  # (n_fit * num_envs, H, W, 3)

    # Extract patch tokens in batches
    fit_tokens_list = []
    for i in range(0, len(fit_frames), pca_batch_size):
        batch = fit_frames[i : i + pca_batch_size]
        fit_tokens_list.append(_extract_patch_tokens(obs_encoder, batch, no_pos_embed=no_pos_embed))
    import torch
    fit_tokens = torch.cat(fit_tokens_list, dim=0)  # (n_fit*num_envs, C, Hp, Wp)

    N_fit, C, Hp, Wp = fit_tokens.shape
    fit_feats = fit_tokens.permute(0, 2, 3, 1).reshape(-1, C).numpy()

    pca = PCA(n_components=3 + skip_components)
    pca.fit(fit_feats)

    # ── 2. Project every frame for every env through the shared PCA ───
    env_pca_frames = []  # list of (T_env, H_out, W_out, 3) uint8 per env
    for e in range(num_envs):
        env_rgb = env_all_frames[e]  # (T_env, H, W, 3) uint8

        # Extract patch tokens in batches
        tokens_list = []
        for i in range(0, len(env_rgb), pca_batch_size):
            batch = env_rgb[i : i + pca_batch_size]
            tokens_list.append(_extract_patch_tokens(obs_encoder, batch, no_pos_embed=no_pos_embed))
        tokens = torch.cat(tokens_list, dim=0)  # (T_env, C, Hp, Wp)

        vis_list, _ = _pca_visualize_patches(tokens, pca_model=pca, skip_components=skip_components)

        # Upsample each PCA frame to target resolution
        upsampled = np.stack([
            np.array(PILImage.fromarray(v).resize(
                (W_out, H_out), resample_filter,
            ))
            for v in vis_list
        ])  # (T_env, H_out, W_out, 3)

        env_pca_frames.append(upsampled)

    # ── 3. Tile into grid (same as log_ego_video_grid) ────────────────
    T = env_pca_frames[0].shape[0]
    black = np.zeros((T, H_out, W_out, 3), dtype=np.uint8)
    while len(env_pca_frames) < grid_rows * grid_cols:
        env_pca_frames.append(black)

    rows = []
    for r in range(grid_rows):
        row = np.concatenate(
            env_pca_frames[r * grid_cols : (r + 1) * grid_cols], axis=2,
        )
        rows.append(row)
    grid_video = np.concatenate(rows, axis=1)

    # ── 4. Write to disk ──────────────────────────────────────────────
    stem = f"dino_pca_grid_{iteration:04d}"
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.join(save_dir, stem)
        path = _write_video(grid_video, base, video_fps)
    else:
        tmp_dir = tempfile.mkdtemp()
        path = _write_video(grid_video, os.path.join(tmp_dir, stem), video_fps)

    if path is None:
        return

    print(f"  → DINOv2 PCA grid video ({grid_rows}×{grid_cols}, "
          f"{H_out}×{W_out}) saved to {path}")

    if wandb is not None:
        wandb.log(
            {"rollout/dino_pca_video": wandb.Video(path, fps=video_fps)},
            step=iteration,
        )


# ─── Weather Distribution Tracking ───────────────────────────────────

# Parameter display metadata: (label, sampling range min, sampling range max)
_WEATHER_PARAMS = [
    ("cloudiness",              0,  90),
    ("precipitation",           0,  80),
    ("precipitation_deposits",  0,  80),
    ("wind_intensity",          0, 100),
    ("sun_azimuth_angle",       0, 360),
    ("sun_altitude_angle",    -15,  90),
    ("fog_density",             0,  40),
    ("fog_distance",            0, 100),
    ("wetness",                 0,  80),
]


class WeatherTracker:
    """Accumulates per-reset weather parameter samples for distribution logging.

    Call ``update(infos)`` after each env step.  The tracker records one sample
    per environment whenever the weather dict in the info changes (i.e. on full
    environment resets).  Duplicate consecutive weather settings from the same
    environment are ignored.
    """

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        # {param_name: list[float]} — accumulated samples across all envs/resets
        self.samples = {name: [] for name, _, _ in _WEATHER_PARAMS}
        # per-env last-seen weather to detect changes
        self._last = [None] * num_envs

    def update(self, infos):
        """Extract weather from info dicts; record new samples on change."""
        for i, info in enumerate(infos):
            w = info.get("weather")
            if w is None:
                continue
            if w != self._last[i]:
                self._last[i] = w
                for name, _, _ in _WEATHER_PARAMS:
                    self.samples[name].append(w[name])

    @property
    def num_samples(self) -> int:
        first_key = _WEATHER_PARAMS[0][0]
        return len(self.samples[first_key])


def log_weather_distributions(
    tracker,
    iteration=0,
    save_dir=None,
    wandb=None,
):
    """Plot 3x3 histogram grid of accumulated weather parameter distributions.

    Saves to ``<save_dir>/weather_dist_<iteration>.png`` (removing older
    snapshots) and logs to W&B as ``rollout/weather_distributions``.
    Also logs per-parameter scalar statistics to W&B.
    """
    n = tracker.num_samples
    if n == 0:
        return

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    fig.suptitle(f"Weather distributions  (n={n}, iter {iteration})", fontsize=13)
    axes = axes.ravel()

    for ax, (name, lo, hi) in zip(axes, _WEATHER_PARAMS):
        vals = np.array(tracker.samples[name])
        nbins = min(30, max(5, n // 3))
        ax.hist(vals, bins=nbins, range=(lo, hi), color="#5b9bd5",
                edgecolor="white", linewidth=0.4)
        ax.set_xlim(lo, hi)
        ax.set_title(name.replace("_", " "), fontsize=10)
        ax.tick_params(labelsize=8)
        mu, sigma = vals.mean(), vals.std()
        ax.text(0.97, 0.93, f"$\\mu$={mu:.1f}  $\\sigma$={sigma:.1f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        '''
        # remove previous snapshots
        import glob as _glob
        for old in _glob.glob(os.path.join(save_dir, "weather_dist_*.png")):
            try:
                os.remove(old)
            except OSError:
                pass
        '''
        path = os.path.join(save_dir, f"weather_dist_{iteration:04d}.png")
        fig.savefig(path, dpi=120)
        print(f"  → Weather distribution figure saved to {path}")

    if wandb is not None:
        wandb.log({"rollout/weather_distributions": wandb.Image(fig)},
                  step=iteration)
        # scalar statistics per parameter
        for name, _, _ in _WEATHER_PARAMS:
            vals = np.array(tracker.samples[name])
            wandb.log({
                f"weather/{name}_mean": vals.mean(),
                f"weather/{name}_std": vals.std(),
            }, step=iteration)

    plt.close(fig)


# ── geodesic distance distribution tracking ──────────────────────────────

class GeodesicTracker:
    """Accumulates initial geodesic distances on episode resets.

    Call ``update(infos)`` after each env step.  The tracker records one
    sample per environment whenever the ``initial_geodesic_distance`` in the
    info dict changes (i.e. on a new episode).
    """

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.samples: list[float] = []
        self._last = [None] * num_envs

    def update(self, infos):
        """Extract initial geodesic distance from info dicts on change."""
        for i, info in enumerate(infos):
            d = info.get("initial_geodesic_distance")
            if d is None or d <= 0:
                continue
            if d != self._last[i]:
                self._last[i] = d
                self.samples.append(d)

    @property
    def num_samples(self) -> int:
        return len(self.samples)


def log_geodesic_distributions(
    tracker,
    iteration=0,
    save_dir=None,
    wandb=None,
):
    """Plot histogram of initial geodesic distances across training episodes.

    Saves to ``<save_dir>/geodesic_dist_<iteration>.png`` and logs to W&B
    as ``rollout/geodesic_distributions``.
    """
    n = tracker.num_samples
    if n == 0:
        return

    vals = np.array(tracker.samples)
    mu, sigma = vals.mean(), vals.std()
    lo, hi = 0.0, max(vals.max() * 1.05, 1.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    nbins = min(40, max(5, n // 3))
    ax.hist(vals, bins=nbins, range=(lo, hi), color="#5b9bd5",
            edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Initial geodesic distance (m)")
    ax.set_ylabel("Count")
    ax.set_title(f"Geodesic distance distribution  (n={n}, iter {iteration})")
    ax.text(0.97, 0.93, f"$\\mu$={mu:.1f}  $\\sigma$={sigma:.1f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    fig.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"geodesic_dist_{iteration:04d}.png")
        fig.savefig(path, dpi=120)
        print(f"  → Geodesic distribution figure saved to {path}")

    if wandb is not None:
        wandb.log({"rollout/geodesic_distributions": wandb.Image(fig)},
                  step=iteration)
        wandb.log({
            "geodesic/initial_distance_mean": mu,
            "geodesic/initial_distance_std": sigma,
            "geodesic/initial_distance_min": float(vals.min()),
            "geodesic/initial_distance_max": float(vals.max()),
        }, step=iteration)

    plt.close(fig)

