"""
Entry point: evaluate NoMaD in CARLA with a topomap built from a
global-planner route.

Example
-------
    python -m nomad_eval.run_eval \
        --config nomad_eval/config/default.yaml \
        --goal-ue 120 -30 1

Run from /home3/rvl/UrbanNav so that package imports resolve.

Pipeline
--------
    1. Connect to CARLA, spawn a walker at ``start_ue``.
    2. Plan route start → goal via CARLA's GlobalRoutePlanner.
    3. Teleport along the route to capture sub-goal RGB images.
    4. Restore the walker to start, then loop:
         obs → NomadPolicy.infer() → WalkerControl → env.step
       until goal reached or ``max_steps`` exceeded.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import List, Optional

import numpy as np
import torch
import yaml
from PIL import Image as PILImage

# Make the repo root importable when this file is run directly.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nomad_eval.env import NomadCarlaEnv
from nomad_eval.nomad_infer import NomadPolicy
from nomad_eval.topomap import (
    plan_route_ue, build_topomap, save_topomap,
)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=str,
                   default=str(pathlib.Path(__file__).parent
                               / 'config' / 'default.yaml'))
    p.add_argument('--port', type=int, default=None,
                   help='override carla.port from config')
    p.add_argument('--start-ue', type=float, nargs=3, default=None,
                   metavar=('X', 'Y', 'Z'),
                   help='override episode.start_ue from config')
    p.add_argument('--goal-ue', type=float, nargs=3, default=None,
                   metavar=('X', 'Y', 'Z'),
                   help='override episode.goal_ue from config')
    p.add_argument('--save-topomap', type=str, default=None,
                   help='override topomap.save_dir (directory to dump jpgs)')
    p.add_argument('--max-steps', type=int, default=None,
                   help='override episode.max_steps')
    p.add_argument('--teleport', action='store_true',
                   help='use teleport stepping instead of WalkerControl')
    return p.parse_args()


def _load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _apply_cli_overrides(cfg: dict, args) -> dict:
    if args.port is not None:
        cfg['carla']['port'] = args.port
    if args.start_ue is not None:
        cfg['episode']['start_ue'] = list(args.start_ue)
    if args.goal_ue is not None:
        cfg['episode']['goal_ue'] = list(args.goal_ue)
    if args.save_topomap is not None:
        cfg['topomap']['save_dir'] = args.save_topomap
    if args.max_steps is not None:
        cfg['episode']['max_steps'] = args.max_steps
    if args.teleport:
        cfg['carla']['teleport'] = True
    return cfg


def _nomad_wp_to_cam_xz(wp_nomad: np.ndarray,
                        velocity_scale: float) -> np.ndarray:
    """NoMaD local frame (x=forward, y=left) → CARLA camera frame
    (x_cam=right, z_cam=forward).

        x_cam =  -y_nomad
        z_cam =   x_nomad

    ``velocity_scale = MAX_V / RATE`` rescales normalized NoMaD outputs
    into metric displacements per policy step (see navigate.py).
    """
    x_fwd = wp_nomad[0] * velocity_scale
    y_left = wp_nomad[1] * velocity_scale
    return np.array([-y_left, x_fwd], dtype=np.float32)


def run(cfg: dict) -> dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[nomad_eval] device={device}")

    # ── Policy ──────────────────────────────────────────────────────────
    nomad_cfg = cfg['nomad']
    policy = NomadPolicy(
        ckpt_path=nomad_cfg['ckpt_path'],
        config_path=nomad_cfg['config_path'],
        device=device,
        radius=int(nomad_cfg['radius']),
        close_threshold=int(nomad_cfg['close_threshold']),
        num_samples=int(nomad_cfg['num_samples']),
        waypoint_idx=int(nomad_cfg['waypoint_idx']),
    )

    # ── Env ─────────────────────────────────────────────────────────────
    carla_cfg = cfg['carla']
    env = NomadCarlaEnv(
        port=int(carla_cfg['port']),
        fps=int(carla_cfg['fps']),
        n_skips=int(carla_cfg['n_skips']),
        teleport=bool(carla_cfg['teleport']),
        max_speed=float(carla_cfg['max_speed']),
        success_radius=float(carla_cfg['success_radius']),
    )
    env.connect()

    start_ue = tuple(cfg['episode']['start_ue'])
    goal_ue = tuple(cfg['episode']['goal_ue'])
    start_yaw = float(cfg['episode'].get('start_yaw_deg', 0.0))
    max_steps = int(cfg['episode']['max_steps'])

    env.reset(start_ue=start_ue, goal_ue=goal_ue, start_yaw_deg=start_yaw)

    # ── Plan route + build topomap ──────────────────────────────────────
    topo_cfg = cfg['topomap']
    print(f"[nomad_eval] planning route "
          f"from UE {start_ue} to UE {goal_ue} …")
    route = plan_route_ue(
        env, start_ue, goal_ue,
        resolution_m=float(topo_cfg['planner_resolution_m']),
    )
    print(f"[nomad_eval] raw route: {len(route)} waypoints, "
          f"length {route.total_length:.1f} m")

    ground_z = topo_cfg.get('ground_z_ue')
    ground_z = None if ground_z is None else float(ground_z)

    print("[nomad_eval] building topomap …")
    topomap_imgs, topomap_nodes = build_topomap(
        env, route,
        min_spacing_m=float(topo_cfg['min_spacing_m']),
        include_start=bool(topo_cfg['include_start']),
        ground_z=ground_z,
    )
    print(f"[nomad_eval] topomap: {len(topomap_imgs)} sub-goal images")

    if topo_cfg.get('save_dir'):
        save_topomap(topomap_imgs, topo_cfg['save_dir'])
        print(f"[nomad_eval] topomap images saved → {topo_cfg['save_dir']}")

    # Restore walker to start pose before running the policy.
    env.teleport_robot_ue(
        ue_xy=(float(start_ue[0]), float(start_ue[1])),
        yaw_deg=start_yaw,
        ground_z=float(start_ue[2]),
    )
    env._drain_queues()
    env.world.tick()

    policy.reset(topomap_imgs)

    # ── Main loop ──────────────────────────────────────────────────────
    rate = env.fps / max(1, env.n_skips)
    velocity_scale = float(cfg['control']['max_v']) / rate
    print(f"[nomad_eval] velocity_scale = MAX_V / RATE = "
          f"{cfg['control']['max_v']}/{rate} = {velocity_scale:.4f}")

    log: List[dict] = []
    for t in range(max_steps):
        pil, obs_info = env.render_observation()
        policy.push_obs(pil)

        if not policy.is_ready:
            # Still filling the context queue — take no action, just tick
            env.world.tick()
            continue

        waypoints_nomad, policy_info = policy.infer()

        wp_sel = waypoints_nomad[policy.waypoint_idx]
        cam_xz = _nomad_wp_to_cam_xz(wp_sel, velocity_scale)

        env.step_camera_waypoint(cam_xz)

        d = env.distance_to_goal()
        entry = {
            't': t,
            'xz_std': obs_info['xz_std'].tolist(),
            'distance_to_goal': float(d),
            'closest_node': int(policy_info['closest_node']),
            'goal_node': int(policy_info['goal_node']),
            'dist_to_subgoal': float(policy_info['dist_to_subgoal']),
            'diffusion_time_s': float(policy_info['diffusion_time_s']),
        }
        log.append(entry)

        if t % 10 == 0:
            print(f"  t={t:3d}  pos={obs_info['xz_std']}  "
                  f"d2goal={d:.2f}m  node={policy_info['closest_node']}/"
                  f"{policy_info['goal_node']}  "
                  f"subgoal_dist={policy_info['dist_to_subgoal']:.2f}")

        if env.is_success():
            print(f"[nomad_eval] reached goal at step {t} "
                  f"(d={d:.2f} m ≤ {env.success_radius} m)")
            break
        if policy.reached_goal:
            print(f"[nomad_eval] topomap localization reached final node "
                  f"at step {t} (d2goal={d:.2f} m)")
            break
    else:
        print(f"[nomad_eval] max_steps ({max_steps}) exceeded; "
              f"final d2goal={env.distance_to_goal():.2f} m")

    result = {
        'success': bool(env.is_success()),
        'steps': len(log),
        'final_distance_to_goal': float(env.distance_to_goal()),
        'num_topomap_nodes': int(len(topomap_imgs)),
        'route_length_m': float(route.total_length),
        'trajectory': log,
    }
    env.close()
    return result


def main():
    args = _parse_args()
    cfg = _apply_cli_overrides(_load_config(args.config), args)
    result = run(cfg)
    print(
        f"\n[nomad_eval] done: success={result['success']}, "
        f"steps={result['steps']}, "
        f"final d2goal={result['final_distance_to_goal']:.2f} m, "
        f"topomap nodes={result['num_topomap_nodes']}, "
        f"route length={result['route_length_m']:.1f} m"
    )


if __name__ == '__main__':
    main()
