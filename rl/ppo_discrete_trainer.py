"""
CleanRL-style PPO trainer for CARLA with DD-PPO discrete action space.

Uses the same 4-action space as Habitat's DD-PPO (Wijmans et al., ICLR 2020):
  STOP, MOVE_FORWARD (0.25 m), TURN_LEFT (10 deg), TURN_RIGHT (10 deg).

Usage:
    python -m rl.ppo_discrete_trainer --config config/rl.yaml \
        --carla_bin /path/to/CarlaUnreal.sh \
        --num_envs 4 --gpu_ids 0,1,2,3
"""

import argparse
import os
import signal
import time
from itertools import cycle

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import multiprocessing as mp
from config.utils import load_config

from rl.utils.carla_manager import (
    VecCarlaMultiAgentEnv, launch_carla_servers, stop_carla_servers,
)
from rl.ppo_discrete_agent import PPODiscreteAgent, GoalOnlyDiscreteMLPAgent
from rl.utils.vis import (visualize, compute_bev_specs,
                          compute_episode_bev_specs, log_complete_episode_bev,
                          WeatherTracker, log_weather_distributions)
from rl.utils.logger import log_metrics, EpisodeTracker, EpisodeTrajectoryAccumulator
from rl.utils.buffer import DiscreteRolloutBuffers, compute_gae

# Shared utilities from continuous trainer
from rl.ppo_trainer import RunningMeanStd, save_checkpoint, _parse_gpu_ids


# ─── PPO Update (discrete) ──────────────────────────────────────────
def ppo_update(agent, optimizer, batch, args, num_steps, num_envs,
               initial_lstm_state, device, use_goal_only=False):
    """
    PPO with env-based minibatching and sequential LSTM replay
    (discrete / Categorical action variant).
    """
    agent.train()
    if hasattr(agent, 'obs_encoder'):
        agent.obs_encoder.eval()

    batch_size = num_steps * num_envs
    envsperbatch = num_envs // args.num_minibatches
    flatinds = np.arange(batch_size).reshape(num_steps, num_envs)
    envinds = np.arange(num_envs)

    pg_losses, v_losses, ent_losses, clip_fracs, approx_kls = [], [], [], [], []
    stopped_epoch = args.num_epochs

    for epoch in range(args.num_epochs):
        np.random.shuffle(envinds)
        kl_exceeded = False
        for start in range(0, num_envs, envsperbatch):
            end = start + envsperbatch
            mbenvinds = envinds[start:end]
            mb_inds = flatinds[:, mbenvinds].ravel()

            mb_features = batch["features"][mb_inds]
            mb_dec_out = batch["dec_out"][mb_inds]
            mb_goal = torch.as_tensor(
                batch["goal"][mb_inds], dtype=torch.float32, device=device)
            mb_actions = torch.as_tensor(
                batch["actions"][mb_inds], dtype=torch.long, device=device)
            mb_old_lp = torch.as_tensor(
                batch["logprobs"][mb_inds], dtype=torch.float32, device=device)
            mb_old_val = torch.as_tensor(
                batch["values"][mb_inds], dtype=torch.float32, device=device)
            mb_adv = torch.as_tensor(
                batch["advantages"][mb_inds], dtype=torch.float32, device=device)
            mb_ret = torch.as_tensor(
                batch["returns"][mb_inds], dtype=torch.float32, device=device)
            mb_dones = torch.as_tensor(
                batch["dones"][mb_inds], dtype=torch.float32, device=device)

            mb_action_hist = None
            if "action_hist" in batch:
                mb_action_hist = torch.as_tensor(
                    batch["action_hist"][mb_inds],
                    dtype=torch.float32, device=device,
                )

            if use_goal_only:
                _, new_lp, entropy, new_val, _ = agent.get_action_and_value(
                    None, None, mb_goal, None, action=mb_actions,
                    action_history=mb_action_hist,
                )
            else:
                mb_initial_h = initial_lstm_state[0][:, mbenvinds].contiguous()
                mb_initial_c = initial_lstm_state[1][:, mbenvinds].contiguous()

                _, new_lp, entropy, new_val, _ = (
                    agent.get_action_and_value_sequential(
                        mb_goal, (mb_initial_h, mb_initial_c), mb_dones,
                        num_steps, actions=mb_actions,
                        features=mb_features, dec_out=mb_dec_out,
                        action_history=mb_action_hist,
                    )
                )

            # policy loss (clipped)
            ratio = (new_lp - mb_old_lp).exp()
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(
                ratio, 1 - args.clip_coef, 1 + args.clip_coef) * mb_adv
            pg_loss = -torch.min(surr1, surr2).mean()

            # value loss (clipped)
            if args.clip_vloss:
                v_clipped = mb_old_val + torch.clamp(
                    new_val - mb_old_val, -args.clip_coef, args.clip_coef)
                v_loss = torch.max(
                    (new_val - mb_ret) ** 2,
                    (v_clipped - mb_ret) ** 2,
                ).mean()
            else:
                v_loss = F.mse_loss(new_val, mb_ret)

            ent_loss = -entropy.mean()

            loss = pg_loss + args.vf_coef * v_loss + args.ent_coef * ent_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                agent.trainable_parameters(), args.max_grad_norm)
            optimizer.step()

            pg_losses.append(pg_loss.item())
            v_losses.append(v_loss.item())
            ent_losses.append(ent_loss.item())
            with torch.no_grad():
                clip_fracs.append(
                    ((ratio - 1.0).abs() > args.clip_coef)
                    .float().mean().item()
                )
                approx_kl = ((ratio - 1) - ratio.log()).mean().item()
                approx_kls.append(approx_kl)

            if args.target_kl is not None and approx_kl > args.target_kl:
                kl_exceeded = True
                break

        if kl_exceeded:
            stopped_epoch = epoch + 1
            break

    return {
        "pg_loss": np.mean(pg_losses),
        "vf_loss": np.mean(v_losses),
        "entropy": -np.mean(ent_losses),
        "clip_frac": np.mean(clip_fracs),
        "approx_kl": np.mean(approx_kls),
        "stopped_epoch": stopped_epoch,
    }


# ─── Training Loop ────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)

    num_actions = args.num_actions   # 4 (DD-PPO default)

    # ── CARLA server ports ──
    ports = [args.base_port + 4 * i for i in range(args.num_envs)]

    # ── optionally auto-launch CARLA servers ──
    server_procs = []
    if args.carla_bin:
        gpu_ids = _parse_gpu_ids(args.gpu_ids, args.num_envs)
        print(f"Launching {args.num_envs} CARLA server(s):")
        server_procs = launch_carla_servers(
            args.carla_bin, ports, gpu_ids,
            startup_wait=args.carla_startup_wait,
            stagger_delay=args.carla_stagger_delay,
        )

    # Procedural obstacles (disabled by default)
    obstacle_config = None
    if args.obstacles:
        from rl.envs.obstacle_manager import ObstacleConfig
        obstacle_config = ObstacleConfig(
            p_crosswalk_challenge=args.p_crosswalk_challenge,
        )
        print(f"  Procedural obstacles ENABLED "
              f"(p_crosswalk_challenge={args.p_crosswalk_challenge})")

    # SFM pedestrians (disabled by default)
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
        gamma=args.gamma,
        num_agents_per_server=args.num_agents_per_server,
        towns=args.towns,
        map_change_interval=args.map_change_interval,
        teleport=args.teleport,
        carla_bin=args.carla_bin,
        gpu_ids=gpu_ids if args.carla_bin else None,
        server_procs=server_procs if server_procs else None,
        carla_startup_wait=args.carla_startup_wait,
        discrete=True,
        obstacle_config=obstacle_config,
        pedestrian_config=pedestrian_config,
        navmesh_cache_dir=args.navmesh_cache_dir,
        quadrant_margin=args.quadrant_margin,
        randomize_weather=args.weather,
    )

    num_envs = vec_env.num_envs
    num_steps = args.num_steps
    batch_size = num_steps * num_envs
    assert num_envs % args.num_minibatches == 0, (
        f"num_envs ({num_envs}) must be divisible by num_minibatches "
        f"({args.num_minibatches})"
    )
    minibatch_size = batch_size // args.num_minibatches

    # ── action history (one-hot per timestep) ──
    n_action_history = args.n_action_history
    action_history_dim = n_action_history * num_actions

    # ── agent ──
    use_goal_only = args.agent == "goal_only"
    if use_goal_only:
        agent = GoalOnlyDiscreteMLPAgent(
            num_actions=num_actions,
            hidden_dim=args.goal_only_hidden,
            num_layers=args.goal_only_layers,
            n_action_history=n_action_history,
        ).to(device)
    else:
        agent = PPODiscreteAgent(
            cfg,
            num_actions=num_actions,
            lstm_hidden_dim=args.lstm_hidden,
            lstm_num_layers=args.lstm_layers,
            n_action_history=n_action_history,
            goal_to_heads=args.goal_to_heads,
            goal_mode=args.goal_mode,
            norm_obs=args.norm_obs,
            encoder_type=args.encoder_type,
        ).to(device)
    optimizer = torch.optim.Adam(
        agent.trainable_parameters(), lr=args.lr, eps=1e-8)

    # ── wandb ──
    use_wandb = (getattr(cfg, "logging", None)
                 and getattr(cfg.logging, "enable_wandb", False))
    if args.no_wandb:
        use_wandb = False
    if use_wandb:
        import wandb
        wandb.init(
            project=cfg.project.name,
            name=cfg.project.run_name,
            config={
                "agent": args.agent,
                "action_space": "discrete_ddppo",
                "num_actions": num_actions,
                "num_envs": num_envs,
                "num_steps": num_steps,
                "batch_size": batch_size,
                "minibatch_size": minibatch_size,
                "num_epochs": args.num_epochs,
                "lr": args.lr,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "clip_coef": args.clip_coef,
                "vf_coef": args.vf_coef,
                "ent_coef": args.ent_coef,
                "max_grad_norm": args.max_grad_norm,
                "lstm_hidden": args.lstm_hidden,
                "lstm_layers": args.lstm_layers,
                "n_action_history": n_action_history,
            },
        )
        print("W&B logging enabled.")
    else:
        wandb = None

    # ── reward normalization ──
    reward_rms = RunningMeanStd() if args.norm_reward else None

    # ── obs shapes ──
    context_size = cfg.model.obs_encoder.context_size
    height, width = cfg.data.height, cfg.data.width
    obs_shape = (context_size, height, width, 3)
    cord_shape = (context_size * 2,)
    obs_feat_dim = cfg.model.encoder_feat_dim  # 768
    use_simple_encoder = args.encoder_type == "simple"
    num_tokens = 1 if use_simple_encoder else context_size + 1

    # ── rollout buffers (discrete) ──
    bufs = DiscreteRolloutBuffers(
        num_steps, num_envs, obs_shape, cord_shape, num_actions,
        num_tokens, obs_feat_dim, device,
        action_history_dim=action_history_dim,
    )

    # ── episode tracker ──
    ep_tracker = EpisodeTracker(num_envs)
    ep_accum = EpisodeTrajectoryAccumulator(num_envs)

    # ── kick off ──
    obs_dict, initial_infos = vec_env.reset()
    ep_accum.set_initial_spawns(obs_dict['cord'])
    lstm_state = agent.get_initial_lstm_state(num_envs, device)
    global_step = 0
    best_reward = -float("inf")

    # ── weather tracking ──
    weather_tracker = WeatherTracker(num_envs)
    weather_tracker.update(initial_infos)

    # per-env rolling action history (flat one-hot vector)
    action_hist = np.zeros((num_envs, action_history_dim), dtype=np.float32)

    # per-env BEV images + metadata
    bev_images = [None] * num_envs
    bev_metas = [None] * num_envs

    n_trainable = sum(p.numel() for p in agent.trainable_parameters())
    print(
        f"PPO (DD-PPO discrete) | agent={args.agent} | "
        f"{n_trainable:,} trainable params | "
        f"{num_envs} envs | {num_steps} steps/rollout | "
        f"batch {batch_size} | minibatch {minibatch_size} | "
        f"{num_actions} actions (STOP/FWD/LEFT/RIGHT)"
    )

    t_start = time.time()

    try:
        for iteration in range(1, args.num_iterations + 1):
            t0 = time.time()

            if args.anneal_lr:
                frac = 1.0 - (iteration - 1) / args.num_iterations
                for pg in optimizer.param_groups:
                    pg["lr"] = frac * args.lr

            # ── BEV capture (auto-altitude from start/goal/geodesic) ──
            is_vis_iter = (args.vis_every > 0
                           and iteration % args.vis_every == 0)
            obstacle_layouts = None
            if is_vis_iter:
                vec_env.set_collect_substep_frames(True)
                try:
                    obstacle_layouts = vec_env.get_obstacle_layouts()
                except Exception as e:
                    print(f"  [BEV-obstacle] layout query failed: {e}")
                try:
                    _bev_specs = compute_bev_specs(
                        obs_dict['cord'],
                        obs_dict.get('goal_world', obs_dict['goal']),
                        obstacle_layouts=obstacle_layouts,
                        fov=args.bev_fov,
                        min_altitude=args.bev_altitude,
                    )
                    _bev_results = vec_env.capture_bev_at_positions(
                        _bev_specs,
                        fov=args.bev_fov,
                        img_size=args.bev_img_size,
                    )
                    for i, result in enumerate(_bev_results):
                        if result is not None:
                            bev_images[i], bev_metas[i] = result
                except Exception as e:
                    print(f"  [BEV] capture failed: {e}")
            substep_frames = ([[] for _ in range(num_envs)]
                              if is_vis_iter else None)

            # ── rollout collection ──
            agent.eval()
            if not use_goal_only:
                agent.obs_encoder.eval()

            initial_lstm_state = (lstm_state[0].detach().clone(),
                                  lstm_state[1].detach().clone())

            for step in range(num_steps):
                global_step += num_envs

                bufs.obs[step] = obs_dict["obs"]
                bufs.cord[step] = obs_dict["cord"]
                bufs.goal[step] = obs_dict["goal"]
                bufs.goal_world[step] = obs_dict.get(
                    "goal_world", obs_dict["goal"])

                if action_history_dim > 0:
                    bufs.action_hist[step] = action_hist

                with torch.no_grad():
                    goal_t = torch.as_tensor(
                        obs_dict["goal"],
                        dtype=torch.float32, device=device)

                    ah_t = None
                    if action_history_dim > 0:
                        ah_t = torch.as_tensor(
                            action_hist,
                            dtype=torch.float32, device=device)

                    if use_goal_only:
                        features = None
                        obs_t = None
                        cord_t = None
                    else:
                        obs_t = torch.as_tensor(
                            obs_dict["obs"], device=device)
                        cord_t = torch.as_tensor(
                            obs_dict["cord"],
                            dtype=torch.float32, device=device)
                        if use_simple_encoder:
                            features = agent.obs_encoder(obs_t[:, -1:])
                        else:
                            features = agent.obs_encoder(obs_t, cord_t)
                        bufs.features_gpu[step] = features

                    action, logprob, _, value, lstm_state = (
                        agent.get_action_and_value(
                            obs_t, cord_t, goal_t, lstm_state,
                            features=features,
                            action_history=ah_t,
                        )
                    )

                actions_np = action.cpu().numpy()       # (E,) int64
                bufs.actions[step] = actions_np
                bufs.logprobs[step] = logprob.cpu().numpy()
                bufs.values[step] = value.cpu().numpy()

                # update rolling one-hot action history
                if action_history_dim > 0:
                    if n_action_history > 1:
                        action_hist[:, :-num_actions] = \
                            action_hist[:, num_actions:]
                    action_hist[:, -num_actions:] = 0.0
                    action_hist[
                        np.arange(num_envs),
                        (n_action_history - 1) * num_actions + actions_np,
                    ] = 1.0

                # Step env with discrete actions
                obs_dict, rewards, terminateds, truncateds, infos = \
                    vec_env.step(actions_np)
                dones = np.logical_or(
                    terminateds, truncateds).astype(np.float32)

                weather_tracker.update(infos)

                # Store actual displacement for vis
                for i in range(num_envs):
                    disp = infos[i].get('displacement_xz')
                    if disp is not None:
                        bufs.actions_2d[step, i] = disp

                if substep_frames is not None:
                    for i in range(num_envs):
                        sf = infos[i].get('substep_frames')
                        if sf is not None:
                            substep_frames[i].append(sf)

                ep_tracker.step(rewards, dones, infos)
                bufs.store_control_info(step, infos)
                bufs.raw_rewards[step] = np.copy(rewards)

                # truncation bootstrap
                for i in range(num_envs):
                    if truncateds[i] and not terminateds[i]:
                        term_obs = infos[i].get('terminal_observation')
                        if term_obs is not None:
                            with torch.no_grad():
                                t_goal = torch.as_tensor(
                                    term_obs["goal"][None],
                                    dtype=torch.float32, device=device)
                                t_ah = None
                                if action_history_dim > 0:
                                    t_ah = torch.as_tensor(
                                        action_hist[i:i+1],
                                        dtype=torch.float32, device=device)
                                if use_goal_only:
                                    tv = agent.get_value(
                                        None, None, t_goal,
                                        (lstm_state[0][:, i:i+1].contiguous(),
                                         lstm_state[1][:, i:i+1].contiguous()),
                                        action_history=t_ah,
                                    ).cpu().item()
                                else:
                                    t_obs = torch.as_tensor(
                                        term_obs["obs"][None], device=device)
                                    t_cord = torch.as_tensor(
                                        term_obs["cord"][None],
                                        dtype=torch.float32, device=device)
                                    tv = agent.get_value(
                                        t_obs, t_cord, t_goal,
                                        (lstm_state[0][:, i:i+1].contiguous(),
                                         lstm_state[1][:, i:i+1].contiguous()),
                                        action_history=t_ah,
                                    ).cpu().item()
                                bufs.trunc_values[step, i] = tv
                                bufs.has_trunc_value[step, i] = 1.0

                # reward clipping
                if args.reward_clip > 0:
                    rewards = np.clip(
                        rewards, -args.reward_clip, args.reward_clip)

                # reward normalization
                if reward_rms is not None:
                    reward_rms.update(rewards)
                    rewards = reward_rms.normalize(rewards)

                bufs.rewards[step] = rewards
                bufs.dones[step] = dones
                bufs.terminateds[step] = terminateds.astype(np.float32)

                # reset LSTM and action history for finished episodes
                for i in range(num_envs):
                    if dones[i]:
                        lstm_state[0][:, i, :] = 0
                        lstm_state[1][:, i, :] = 0
                        if action_history_dim > 0:
                            action_hist[i] = 0
                        ep_accum.on_episode_end(i, obs_dict['cord'][i])

            # ── bootstrap value for end of rollout ──
            with torch.no_grad():
                goal_t = torch.as_tensor(
                    obs_dict["goal"], dtype=torch.float32, device=device)
                ah_t = None
                if action_history_dim > 0:
                    ah_t = torch.as_tensor(
                        action_hist, dtype=torch.float32, device=device)
                if use_goal_only:
                    last_value = agent.get_value(
                        None, None, goal_t, lstm_state,
                        action_history=ah_t,
                    ).cpu().numpy()
                else:
                    obs_t = torch.as_tensor(
                        obs_dict["obs"], device=device)
                    cord_t = torch.as_tensor(
                        obs_dict["cord"],
                        dtype=torch.float32, device=device)
                    last_value = agent.get_value(
                        obs_t, cord_t, goal_t, lstm_state,
                        action_history=ah_t,
                    ).cpu().numpy()

            # ── next_values for GAE ──
            next_values = np.zeros_like(bufs.values)
            next_values[:-1] = bufs.values[1:]
            next_values[-1] = last_value
            mask = bufs.has_trunc_value > 0
            next_values[mask] = bufs.trunc_values[mask]

            # ── GAE ──
            advantages, returns = compute_gae(
                bufs.rewards, bufs.values, bufs.terminateds, bufs.dones,
                next_values,
                gamma=args.gamma, gae_lambda=args.gae_lambda,
            )

            bufs.trunc_values[:] = 0
            bufs.has_trunc_value[:] = 0

            # ── flatten + PPO update ──
            batch = bufs.flatten(
                advantages, returns, num_actions,
                context_size, obs_feat_dim,
                norm_adv=args.norm_adv,
            )
            stats = ppo_update(
                agent, optimizer, batch, args,
                num_steps, num_envs,
                initial_lstm_state, device,
                use_goal_only=use_goal_only,
            )

            # ── logging / vis / checkpoint ──
            ep_stats = ep_tracker.flush()
            try:
                solv_stats = vec_env.get_solvability_stats(reset=True)
                if ep_stats is not None:
                    ep_stats.update({
                        'unsolvable_rate': solv_stats['unsolvable_rate'],
                        'unsolvable_episodes': solv_stats['unsolvable_episodes'],
                    })
                elif solv_stats.get('unsolvable_episodes', 0) > 0:
                    ep_stats = solv_stats
            except Exception:
                pass
            should_log_wandb = (args.log_every > 0
                                and iteration % args.log_every == 0)
            ep_reward = log_metrics(
                iteration, args.num_iterations, global_step,
                bufs, stats, optimizer, t_start, t0, wandb,
                ep_stats=ep_stats,
                log_wandb=should_log_wandb,
            )

            if is_vis_iter:
                orig_actions = bufs.actions
                bufs.actions = bufs.actions_2d
                try:
                    _enc = agent.obs_encoder if not use_goal_only else None
                    visualize(
                        iteration, args, bufs,
                        bev_images, bev_metas, wandb,
                        substep_frames=substep_frames, obs_encoder=_enc,
                        obstacle_layouts=obstacle_layouts,
                    )
                except Exception as e:
                    print(f"  [VIS] visualize failed: {e}")
                bufs.actions = orig_actions
                vec_env.set_collect_substep_frames(False)

                try:
                    ep_specs, ep_key_map = compute_episode_bev_specs(
                        bufs.cord, bufs.goal_world, bufs.dones,
                        ep_accum.continuation_spawns,
                        default_altitude=args.bev_altitude,
                        fov=args.bev_fov,
                    )
                    ep_bev_results = vec_env.capture_bev_at_positions(
                        ep_specs,
                        fov=args.bev_fov,
                        img_size=args.bev_img_size,
                    )
                    vis_dir = os.path.join(args.save_dir, "vis")
                    log_complete_episode_bev(
                        bufs.cord, bufs.goal_world, bufs.raw_rewards,
                        bufs.dones, ep_accum.continuation_spawns,
                        ep_bev_results, ep_key_map,
                        grid_cols=args.num_agents_per_server,
                        iteration=iteration,
                        save_dir=vis_dir,
                        wandb=wandb,
                    )
                except Exception as e:
                    print(f"  [BEV-ep] complete-episode vis failed: {e}")

                # ── weather parameter distributions ──
                if weather_tracker.num_samples > 0:
                    try:
                        vis_dir = os.path.join(args.save_dir, "vis")
                        log_weather_distributions(
                            weather_tracker,
                            iteration=iteration,
                            save_dir=vis_dir,
                            wandb=wandb,
                        )
                    except Exception as e:
                        print(f"  [VIS] weather distribution vis failed: {e}")

            if iteration % args.save_every == 0:
                best_reward = save_checkpoint(
                    iteration, args, agent, optimizer,
                    global_step, ep_reward, best_reward,
                )

    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("\nCleaning up (Ctrl+C disabled during shutdown)...")

        try:
            vec_env.close()
        except Exception as e:
            print(f"  vec_env.close() error: {e}")

        final_procs = getattr(vec_env, 'server_procs', server_procs)
        if final_procs and any(p is not None for p in final_procs):
            print("Stopping CARLA servers ...")
            stop_carla_servers([p for p in final_procs if p is not None])

        if wandb is not None:
            wandb.finish()

        signal.signal(signal.SIGINT, signal.SIG_DFL)

    print("Training complete.")


# ─── Entry Point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="DD-PPO discrete-action PPO trainer for CARLA")
    # environment
    parser.add_argument("--config", type=str, default="config/rl.yaml")
    parser.add_argument("--base_port", type=int, default=2000)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--max_speed", type=float, default=1.4,
                        help="Only used for the continuous-env fallback; "
                             "discrete env uses forward_step_size instead")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--max_episode_steps", type=int, default=500,
                        help="DD-PPO default is 500 steps per episode")
    parser.add_argument("--teleport", action="store_true", default=True,
                        help="DD-PPO always teleports (default: True)")
    # procedural obstacles
    parser.add_argument("--obstacles", action="store_true", default=False,
                        help="Enable procedural obstacle generation")
    parser.add_argument("--p_crosswalk_challenge", type=float, default=0.3,
                        help="Per-episode probability of blocked-crosswalk scenario")
    parser.add_argument("--pedestrians", action="store_true", default=False,
                        help="Enable SFM-controlled pedestrians")
    parser.add_argument("--num_pedestrians_per_region", type=int, default=4,
                        help="Number of SFM pedestrians per quadrant")
    parser.add_argument("--weather", action="store_true", default=False,
                        help="Randomize weather and sun position each episode")
    parser.add_argument("--navmesh_cache_dir", type=str, default=None,
                        help="Directory with precomputed navmesh cache NPZ files")
    parser.add_argument("--quadrant_margin", type=float, default=None,
                        help="Inset margin (metres) for spawn/goal sampling within each "
                             "quadrant.  Defaults to goal_range.")
    # multi-agent
    parser.add_argument("--num_agents_per_server", type=int, default=4)
    parser.add_argument("--towns", type=str, nargs="+",
                        default=["Town02", "Town03", "Town05", "Town10HD"])
    parser.add_argument("--map_change_interval", type=int, default=0)
    # CARLA server
    parser.add_argument("--carla_bin", type=str, default=None)
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--carla_startup_wait", type=int, default=30)
    parser.add_argument("--carla_stagger_delay", type=int, default=5)
    # discrete action space
    parser.add_argument("--num_actions", type=int, default=4,
                        help="DD-PPO: 4 (STOP, FORWARD, LEFT, RIGHT)")
    # PPO hyperparameters (DD-PPO defaults)
    parser.add_argument("--num_steps", type=int, default=128,
                        help="Rollout length (DD-PPO uses 128)")
    parser.add_argument("--num_iterations", type=int, default=10000)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=2,
                        help="DD-PPO uses 2 minibatches")
    parser.add_argument("--lr", type=float, default=2.5e-4,
                        help="DD-PPO default learning rate")
    parser.add_argument("--anneal_lr", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01,
                        help="Entropy coefficient (DD-PPO default 0.01)")
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--clip_vloss", action="store_true", default=True)
    parser.add_argument("--norm_reward", action="store_true", default=False,
                        help="DD-PPO does not normalize rewards by default")
    parser.add_argument("--norm-adv", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--target_kl", type=float, default=None)
    parser.add_argument("--reward_clip", type=float, default=0)
    # model
    parser.add_argument("--agent", type=str, default="ppo",
                        choices=["ppo", "goal_only"])
    parser.add_argument("--encoder_type", type=str, default="full",
                        choices=["full", "simple"],
                        help="Observation encoder: 'full' (DINOv2 + history + cord) "
                             "or 'simple' (DINOv2 on current frame only)")
    parser.add_argument("--lstm_hidden", type=int, default=512)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--goal_only_hidden", type=int, default=64)
    parser.add_argument("--goal_only_layers", type=int, default=2)
    parser.add_argument("--n_action_history", type=int, default=0)
    parser.add_argument("--goal_to_heads", action="store_true",
                        help="(deprecated) use --goal_mode=heads instead")
    parser.add_argument("--goal_mode", type=str, default=None,
                        choices=["lstm", "heads", "both"],
                        help="Where to inject goal: lstm, heads, or both "
                             "(overrides --goal_to_heads)")
    parser.add_argument("--norm_obs", action="store_true")
    # checkpointing
    parser.add_argument("--save_dir", type=str,
                        default="checkpoints/ppo_discrete")
    parser.add_argument("--save_every", type=int, default=100)
    # logging
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--log_every", type=int, default=1)
    # visualisation
    parser.add_argument("--vis_every", type=int, default=100)
    parser.add_argument("--bev_altitude", type=float, default=15.0,
                        help="Minimum BEV camera altitude (m). Actual altitude "
                             "is computed automatically to encompass start, "
                             "goal, and geodesic path.")
    parser.add_argument("--bev_fov", type=float, default=90.0)
    parser.add_argument("--bev_img_size", type=int, default=512)
    parser.add_argument("--vis_video_fps", type=int, default=1)

    train(parser.parse_args())
