"""
Test rollout script for rapid scenario/visualization debugging.

Runs a single rollout (no training) with random or zero actions, then
produces the same BEV / ego-video / obstacle-layout / CCTV / geodesic-field
visualizations as the PPO trainer.

Usage:
    # Random actions (default):
    python -m rl.test_rollout --config config/rl.yaml \
        --carla_bin /path/to/CarlaUnreal.sh --num_envs 1

    # Stay in place (zero actions):
    python -m rl.test_rollout --config config/rl.yaml \
        --carla_bin /path/to/CarlaUnreal.sh --num_envs 1 --action_mode zero

    # With obstacles and pedestrians:
    python -m rl.test_rollout --config config/rl.yaml \
        --carla_bin /path/to/CarlaUnreal.sh --num_envs 1 \
        --obstacles --pedestrians --num_pedestrians_per_region 6
"""

import argparse
import os
import signal
import time

import numpy as np

import multiprocessing as mp
from config.utils import load_config

from rl.utils.carla_manager import VecCarlaMultiAgentEnv, launch_carla_servers, stop_carla_servers
from rl.utils.vis import (visualize, compute_bev_specs,
                          log_obstacle_bev_figure,
                          log_geodesic_field,
                          compute_cctv_specs, log_cctv_video_grid,
                          WeatherTracker, log_weather_distributions,
                          GeodesicTracker, log_geodesic_distributions)
from rl.utils.buffer import RolloutBuffers


# ─── Rollout ─────────────────────────────────────────────────────────
def run_test(args):
    cfg = load_config(args.config)

    # ── CARLA server ports
    ports = [args.base_port + 4 * i for i in range(args.num_envs)]

    # ── optionally auto-launch CARLA servers ──
    server_procs = []
    gpu_ids = None
    if args.carla_bin:
        gpu_ids = _parse_gpu_ids(args.gpu_ids, args.num_envs)
        print(f"Launching {args.num_envs} CARLA server(s):")
        server_procs = launch_carla_servers(
            args.carla_bin, ports, gpu_ids, startup_wait=args.carla_startup_wait,
            stagger_delay=args.carla_stagger_delay,
        )

    # Procedural obstacles
    obstacle_config = None
    if args.obstacles:
        from rl.envs.obstacle_manager import ObstacleConfig
        obstacle_config = ObstacleConfig(
            p_crosswalk_challenge=args.p_crosswalk_challenge,
        )
        print(f"  Procedural obstacles ENABLED "
              f"(p_crosswalk_challenge={args.p_crosswalk_challenge})")

    # SFM pedestrians
    pedestrian_config = None
    if args.pedestrians:
        from rl.envs.pedestrian_manager import PedestrianConfig
        pedestrian_config = PedestrianConfig(
            num_per_region=args.num_pedestrians_per_region,
        )
        print(f"  SFM pedestrians ENABLED "
              f"({args.num_pedestrians_per_region}/region)")

    if args.weather:
        print("  Weather randomisation ENABLED")

    vec_env = VecCarlaMultiAgentEnv(
        cfg, ports=ports,
        max_speed=args.max_speed,
        fps=args.fps,
        max_episode_steps=args.max_episode_steps,
        gamma=0.99,
        num_agents_per_server=args.num_agents_per_server,
        towns=args.towns,
        map_change_interval=0,
        teleport=args.teleport,
        goal_range=args.goal_range,
        carla_bin=args.carla_bin,
        gpu_ids=gpu_ids,
        server_procs=server_procs if server_procs else None,
        carla_startup_wait=args.carla_startup_wait,
        obstacle_config=obstacle_config,
        pedestrian_config=pedestrian_config,
        navmesh_cache_dir=args.navmesh_cache_dir,
        quadrant_margin=args.quadrant_margin,
        randomize_weather=args.weather,
        use_mpc=args.use_mpc,
        dynamic_geo_mode=args.dynamic_geo_mode,
        dynamic_geo_horizon=args.dynamic_geo_horizon,
        scenario_dir=args.scenario_dir,
    )

    num_envs = vec_env.num_envs
    num_steps = args.num_steps
    action_dim = 2

    # ── obs shapes from config ──
    context_size = cfg.model.obs_encoder.context_size
    height, width = cfg.data.height, cfg.data.width
    obs_shape = (context_size, height, width, 3)
    cord_shape = (context_size * 2,)
    obs_feat_dim = cfg.model.encoder_feat_dim  # 768
    num_tokens = context_size + 1

    # ── pre-allocate rollout buffers (on CPU, no GPU features needed) ──
    bufs = RolloutBuffers(
        num_steps, num_envs, obs_shape, cord_shape, action_dim,
        num_tokens, obs_feat_dim, device="cpu",
    )

    weather_tracker = WeatherTracker(num_envs)
    geodesic_tracker = GeodesicTracker(num_envs)

    # ── reset ──
    print(f"Resetting {num_envs} envs (action_mode={args.action_mode}) ...")
    obs_dict, _ = vec_env.reset()

    # ── obstacle layouts (needed before BEV capture for auto-altitude) ──
    obstacle_layouts = None
    try:
        obstacle_layouts = vec_env.get_obstacle_layouts()
    except Exception as e:
        print(f"  [BEV-obstacle] layout query failed: {e}")

    # ── BEV images + metadata (auto-altitude from start/goal/geodesic) ──
    print("Capturing BEV images ...")
    vec_env.set_collect_substep_frames(True)
    bev_specs = compute_bev_specs(
        obs_dict['cord'], obs_dict.get('goal_world', obs_dict['goal']),
        obstacle_layouts=obstacle_layouts,
        fov=args.bev_fov, min_altitude=args.bev_altitude,
    )
    bev_results = vec_env.capture_bev_at_positions(
        bev_specs, fov=args.bev_fov, img_size=args.bev_img_size,
    )
    bev_images = [None] * num_envs
    bev_metas = [None] * num_envs
    for i, result in enumerate(bev_results):
        if result is not None:
            bev_images[i], bev_metas[i] = result

    substep_frames = [[] for _ in range(num_envs)]
    mpc_vis_data = [[] for _ in range(num_envs)]
    ped_positions = [[] for _ in range(num_envs)]

    # ── CCTV cameras (spawned before rollout, captured during) ──
    try:
        _cctv_specs = compute_cctv_specs(
            obs_dict['cord'],
            obs_dict.get('goal_world', obs_dict['goal']),
            obstacle_layouts=obstacle_layouts,
        )
        if _cctv_specs:
            vec_env.spawn_cctv_cameras(_cctv_specs, fov=90.0, img_size=512)
            print(f"  Spawned {len(_cctv_specs)} CCTV camera(s)")
    except Exception as e:
        print(f"  [CCTV] camera spawn failed: {e}")

    # ── rollout ──
    print(f"Running {num_steps}-step rollout ...")
    t0 = time.time()
    for step in range(num_steps):
        bufs.obs[step] = obs_dict["obs"]
        bufs.cord[step] = obs_dict["cord"]
        bufs.goal[step] = obs_dict["goal"]
        bufs.goal_world[step] = obs_dict.get("goal_world", obs_dict["goal"])

        # generate actions
        if args.action_mode == "zero":
            actions_np = np.zeros((num_envs, action_dim), dtype=np.float32)
        else:  # "random"
            actions_np = np.random.uniform(
                -args.max_speed, args.max_speed,
                size=(num_envs, action_dim),
            ).astype(np.float32)

        bufs.actions[step] = actions_np

        env_actions = np.clip(actions_np, -args.max_speed, args.max_speed)
        obs_dict, rewards, terminateds, truncateds, infos = vec_env.step(env_actions)
        dones = np.logical_or(terminateds, truncateds).astype(np.float32)

        weather_tracker.update(infos)
        geodesic_tracker.update(infos)

        # collect vis data
        for i in range(num_envs):
            sf = infos[i].get('substep_frames')
            if sf is not None:
                substep_frames[i].append(sf)
            entry = {}
            wp = infos[i].get('policy_waypoints_cam')
            if wp is not None:
                entry['policy_waypoints_cam'] = wp
            x_sol = infos[i].get('mpc_x_sol')
            u_sol = infos[i].get('mpc_u_sol')
            if x_sol is not None:
                entry['mpc_x_sol'] = x_sol
                entry['mpc_u_sol'] = u_sol
            if entry:
                mpc_vis_data[i].append(entry)
            pp = infos[i].get('pedestrian_positions_std')
            ped_positions[i].append(
                pp if pp is not None else np.empty((0, 2)))

        bufs.store_control_info(step, infos)
        bufs.raw_rewards[step] = np.copy(rewards)
        bufs.rewards[step] = rewards
        bufs.dones[step] = dones
        bufs.terminateds[step] = terminateds.astype(np.float32)

        if (step + 1) % 20 == 0:
            print(f"  step {step + 1}/{num_steps}")

    elapsed = time.time() - t0
    print(f"Rollout done in {elapsed:.1f}s ({num_steps / elapsed:.1f} steps/s)")

    vec_env.set_collect_substep_frames(False)

    # ── visualization ──
    os.makedirs(args.save_dir, exist_ok=True)
    vis_dir = os.path.join(args.save_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    iteration = 0

    print("Generating visualizations ...")
    visualize(iteration, args, bufs, bev_images, bev_metas, wandb_module=None,
              substep_frames=substep_frames, obs_encoder=None,
              obstacle_layouts=obstacle_layouts,
              mpc_vis_data=mpc_vis_data)

    '''
    # ── CCTV video (fixed cameras with trajectory overlay) ──
    try:
        cctv_data = vec_env.collect_cctv_frames()
        if cctv_data.get('frames'):
            log_cctv_video_grid(
                cctv_data,
                bufs.cord, bufs.actions,
                buf_rewards=bufs.raw_rewards,
                buf_dones=bufs.dones,
                buf_terminateds=bufs.terminateds,
                ped_positions=ped_positions,
                obstacle_layouts=obstacle_layouts,
                fps=args.vis_video_fps,
                iteration=iteration,
                save_dir=vis_dir,
                wandb=None,
                grid_cols=args.num_agents_per_server,
            )
    except Exception as e:
        print(f"  [CCTV] video generation failed: {e}")
    try:
        vec_env.destroy_cctv_cameras()
    except Exception:
        pass
    '''

    # ── obstacle layout BEV ──
    if obstacle_layouts is not None:
        try:
            log_obstacle_bev_figure(
                obstacle_layouts,
                bev_metas,
                ped_positions=ped_positions,
                grid_cols=args.num_agents_per_server,
                iteration=iteration,
                save_dir=vis_dir,
                wandb=None,
            )
        except Exception as e:
            print(f"  [BEV-obstacle] obstacle layout vis failed: {e}")

    # ── geodesic distance field heatmap / video ──
    if obstacle_layouts is not None and args.dynamic_geo_mode != 'off':
        try:
            log_geodesic_field(
                obstacle_layouts,
                bev_metas,
                grid_cols=args.num_agents_per_server,
                iteration=iteration,
                save_dir=vis_dir,
                wandb=None,
                video_fps=args.vis_video_fps,
            )
        except Exception as e:
            print(f"  [BEV-geodesic] geodesic field vis failed: {e}")

    # ── weather parameter distributions ──
    if weather_tracker.num_samples > 0:
        try:
            log_weather_distributions(
                weather_tracker,
                iteration=iteration,
                save_dir=vis_dir,
                wandb=None,
            )
        except Exception as e:
            print(f"  [weather] distribution vis failed: {e}")

    # ── geodesic distance distributions ──
    if geodesic_tracker.num_samples > 0:
        try:
            log_geodesic_distributions(
                geodesic_tracker,
                iteration=iteration,
                save_dir=vis_dir,
                wandb=None,
            )
        except Exception as e:
            print(f"  [geodesic] distribution vis failed: {e}")

    print(f"Visualizations saved to {vis_dir}/")

    # ── cleanup ──
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print("Cleaning up ...")

    try:
        vec_env.close()
    except Exception as e:
        print(f"  vec_env.close() error: {e}")

    final_procs = getattr(vec_env, 'server_procs', server_procs)
    if final_procs and any(p is not None for p in final_procs):
        print("Stopping CARLA servers ...")
        stop_carla_servers([p for p in final_procs if p is not None])

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    print("Done.")


def _parse_gpu_ids(gpu_ids_str, num_envs):
    if gpu_ids_str:
        ids = [int(x.strip()) for x in gpu_ids_str.split(",")]
        return ids[:num_envs]
    return list(range(num_envs))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Test rollout — no training, just env + visualization"
    )
    # environment
    parser.add_argument("--config", type=str, default="config/rl.yaml")
    parser.add_argument("--base_port", type=int, default=2000)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--max_speed", type=float, default=1.4)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--max_episode_steps", type=int, default=60)        # 1 minute
    parser.add_argument("--teleport", action="store_true", default=False)
    parser.add_argument("--goal_range", type=float, default=40.0)
    parser.add_argument("--quadrant_margin", type=float, default=None)
    parser.add_argument("--use_mpc", action="store_true", default=False,
                        help="Use MPC to convert waypoints to WalkerControl")
    parser.add_argument("--dynamic_geo_mode", type=str, default="off",
                        choices=["off", "soft", "timespace"],
                        help="Dynamic geodesic reward mode")
    parser.add_argument("--dynamic_geo_horizon", type=float, default=5.0,
                        help="Prediction horizon for dynamic geodesic (seconds)")
    # procedural obstacles
    parser.add_argument("--obstacles", action="store_true", default=False)
    parser.add_argument("--p_crosswalk_challenge", type=float, default=0.3)
    parser.add_argument("--pedestrians", action="store_true", default=False)
    parser.add_argument("--num_pedestrians_per_region", type=int, default=30)
    parser.add_argument("--weather", action="store_true", default=False,
                        help="Randomize weather and sun position each episode")
    # navmesh cache
    parser.add_argument("--navmesh_cache_dir", type=str, default=None)
    parser.add_argument("--scenario_dir", type=str, default=None,
                        help="Directory containing precomputed scenario files "
                             "(from generate_scenarios.py). Eliminates runtime "
                             "Dijkstra by loading baked distance fields.")
    # multi-agent
    parser.add_argument("--num_agents_per_server", type=int, default=4)
    parser.add_argument("--towns", type=str, nargs="+",
                        default=["Town02", "Town03", "Town05", "Town10HD"])
    # CARLA server
    parser.add_argument("--carla_bin", type=str, default=None)
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--carla_startup_wait", type=int, default=30)
    parser.add_argument("--carla_stagger_delay", type=int, default=5)
    # rollout
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of env steps in the test rollout")
    parser.add_argument("--action_mode", type=str, default="random",
                        choices=["random", "zero"],
                        help="'random' = uniform random actions, "
                             "'zero' = stay in place")
    # output
    parser.add_argument("--save_dir", type=str, default="checkpoints/test_rollout")
    # visualisation
    parser.add_argument("--bev_altitude", type=float, default=15.0,
                        help="Minimum BEV camera altitude (m). Actual altitude "
                             "is computed automatically to encompass start, "
                             "goal, and geodesic path.")
    parser.add_argument("--bev_fov", type=float, default=90.0)
    parser.add_argument("--bev_img_size", type=int, default=512)
    parser.add_argument("--vis_video_fps", type=int, default=1)

    run_test(parser.parse_args())
