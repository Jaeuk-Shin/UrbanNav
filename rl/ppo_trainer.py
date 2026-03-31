"""
CleanRL-style PPO trainer for CARLA with multi-env support.

Replaces RLlib with a single-file implementation.
Each CARLA instance runs in its own subprocess; the main process
handles all GPU inference and PPO updates.

Usage:
    # Auto-launch CARLA servers (one per GPU):
    python -m rl.ppo_trainer --config config/rl.yaml \
        --carla_bin /path/to/CarlaUnreal.sh \
        --num_envs 4 --gpu_ids 0,1,2,3

    # Or manually launch servers first, then connect:
    #   GPU=0: ./CarlaUnreal.sh -RenderOffScreen -carla-rpc-port=2000 -graphicsadapter=0 -nosound
    #   GPU=1: ./CarlaUnreal.sh -RenderOffScreen -carla-rpc-port=2002 -graphicsadapter=1
    python -m rl.ppo_trainer --config config/rl.yaml --num_envs 2 --base_port 2000
"""

import argparse
import math
import os
import signal
import subprocess
import time
from itertools import cycle

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import multiprocessing as mp
from config.utils import load_config

from rl.utils.carla_manager import VecCarlaEnv, VecCarlaMultiAgentEnv, launch_carla_servers, stop_carla_servers
from rl.ppo_agent import PPOAgent
from rl.goal_only_agent import GoalOnlyMLPAgent
from rl.utils.vis import (visualize, compute_bev_specs,
                          compute_episode_bev_specs, log_complete_episode_bev,
                          log_obstacle_bev_figure,
                          log_geodesic_field,
                          WeatherTracker, log_weather_distributions)
from rl.utils.logger import log_metrics, EpisodeTracker, EpisodeTrajectoryAccumulator
from rl.utils.buffer import RolloutBuffers, compute_gae
from rl.utils.timer import StepTimer




# ─── Reward Normalization ─────────────────────────────────────────────
class RunningMeanStd:
    """Welford online estimator for reward normalization."""

    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, batch):
        batch = np.asarray(batch, dtype=np.float64).ravel()
        batch_mean = batch.mean()
        batch_var = batch.var()
        batch_count = len(batch)

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


# ─── PPO Update ──────────────────────────────────────────────────────
def ppo_update(agent, optimizer, batch, args, num_steps, num_envs,
               initial_lstm_state, device, use_goal_only=False):
    """
    Run PPO with env-based minibatching and sequential LSTM replay.

    Instead of randomly shuffling all (step, env) pairs, we shuffle
    *environments* and replay complete time-sequences for each env subset.
    This enables back-propagation through time (BPTT) in the LSTM and
    correctly resets LSTM hidden state at episode boundaries within
    each rollout.
    """
    agent.train()
    if hasattr(agent, 'obs_encoder'):
        agent.obs_encoder.eval()
    if args.use_decoder and hasattr(agent, 'action_decoder'):
        agent.action_decoder.eval()

    batch_size = num_steps * num_envs
    envsperbatch = num_envs // args.num_minibatches
    # flatinds[t, e] = t * num_envs + e  (step-major layout matching flatten())
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
            # All timesteps for the selected envs, in step-major order
            mb_inds = flatinds[:, mbenvinds].ravel()

            mb_features = batch["features"][mb_inds]
            mb_dec_out = batch["dec_out"][mb_inds]
            mb_goal = torch.as_tensor(batch["goal"][mb_inds], dtype=torch.float32, device=device)
            mb_actions = torch.as_tensor(batch["actions"][mb_inds], dtype=torch.float32, device=device)
            mb_old_lp = torch.as_tensor(batch["logprobs"][mb_inds], dtype=torch.float32, device=device)
            mb_old_val = torch.as_tensor(batch["values"][mb_inds], dtype=torch.float32, device=device)
            mb_adv = torch.as_tensor(batch["advantages"][mb_inds], dtype=torch.float32, device=device)
            mb_ret = torch.as_tensor(batch["returns"][mb_inds], dtype=torch.float32, device=device)
            mb_dones = torch.as_tensor(batch["dones"][mb_inds], dtype=torch.float32, device=device)

            mb_action_hist = None
            if "action_hist" in batch:
                mb_action_hist = torch.as_tensor(
                    batch["action_hist"][mb_inds], dtype=torch.float32, device=device
                )

            if use_goal_only:
                # Feedforward agent — no LSTM replay needed
                _, new_lp, entropy, new_val, _ = agent.get_action_and_value(
                    None, None, mb_goal, None, action=mb_actions,
                    action_history=mb_action_hist,
                )
            else:
                # Sequential LSTM replay from the rollout's initial state
                mb_initial_h = initial_lstm_state[0][:, mbenvinds].contiguous()
                mb_initial_c = initial_lstm_state[1][:, mbenvinds].contiguous()

                _, new_lp, entropy, new_val, _ = agent.get_action_and_value_sequential(
                    mb_goal, (mb_initial_h, mb_initial_c), mb_dones,
                    num_steps, actions=mb_actions,
                    features=mb_features, dec_out=mb_dec_out,
                    action_history=mb_action_hist,
                )

            # policy loss (clipped)
            ratio = (new_lp - mb_old_lp).exp()
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * mb_adv
            pg_loss = -torch.min(surr1, surr2).mean()

            # value loss (clipped)
            if args.clip_vloss:
                v_clipped = mb_old_val + torch.clamp(
                    new_val - mb_old_val, -args.clip_coef, args.clip_coef
                )
                v_loss = torch.max(
                    (new_val - mb_ret) ** 2,
                    (v_clipped - mb_ret) ** 2,
                ).mean()
            else:
                v_loss = F.mse_loss(new_val, mb_ret)

            # entropy bonus
            ent_loss = -entropy.mean()

            loss = pg_loss + args.vf_coef * v_loss + args.ent_coef * ent_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.trainable_parameters(), args.max_grad_norm)
            optimizer.step()

            pg_losses.append(pg_loss.item())
            v_losses.append(v_loss.item())
            ent_losses.append(ent_loss.item())
            with torch.no_grad():
                clip_fracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
                approx_kl = ((ratio - 1) - ratio.log()).mean().item()
                approx_kls.append(approx_kl)

            # KL-based early stopping
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




# ─── Checkpointing ──────────────────────────────────────────────────
def save_checkpoint(iteration, args, agent, optimizer, global_step, ep_reward, best_reward):
    """Save last.pt and optionally best.pt. Returns updated best_reward."""
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = {
        "iteration": iteration,
        "global_step": global_step,
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "reward": float(ep_reward),
    }

    last_path = os.path.join(args.save_dir, "last.pt")
    torch.save(ckpt, last_path)

    if ep_reward > best_reward:
        best_reward = ep_reward
        best_path = os.path.join(args.save_dir, "best.pt")
        torch.save(ckpt, best_path)
        print(f"  -> new best reward={ep_reward:+.2f}, saved {best_path}")

    return best_reward


# ─── Training Loop ────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)

    # ── CARLA server ports
    # Don't know why, but spacing by 2 results in errors...use 4 instead
    ports = [args.base_port + 4 * i for i in range(args.num_envs)]

    # ── optionally auto-launch CARLA servers ──
    server_procs = []
    if args.carla_bin:
        gpu_ids = _parse_gpu_ids(args.gpu_ids, args.num_envs)
        print(f"Launching {args.num_envs} CARLA server(s):")
        server_procs = launch_carla_servers(
            args.carla_bin, ports, gpu_ids, startup_wait=args.carla_startup_wait,
            stagger_delay=args.carla_stagger_delay,
        )


    '''
    simple_action = not args.use_decoder
    vec_env = VecCarlaEnv(
        cfg, ports,
        max_speed=args.max_speed,
        fps=args.fps,
        max_episode_steps=args.max_episode_steps,
        gamma=args.gamma,
        simple_action=simple_action,
        teleport=args.teleport,
    )
    '''

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

    if args.dynamic_geo_mode != 'off':
        if not args.pedestrians:
            print(f"  WARNING: --dynamic_geo_mode={args.dynamic_geo_mode} "
                  f"has no effect without --pedestrians")
        else:
            print(f"  Dynamic geodesic reward: {args.dynamic_geo_mode} "
                  f"(horizon={args.dynamic_geo_horizon}s)")

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
        goal_range=args.goal_range,
        carla_bin=args.carla_bin,
        gpu_ids=gpu_ids if args.carla_bin else None,
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
    )

    # num_envs = total rollout slots (servers × agents_per_server),
    # NOT the number of CARLA servers (which is args.num_envs / len(ports)).
    num_envs = vec_env.num_envs
    num_steps = args.num_steps
    batch_size = num_steps * num_envs
    assert num_envs % args.num_minibatches == 0, (
        f"num_envs ({num_envs}) must be divisible by num_minibatches "
        f"({args.num_minibatches}) for env-based minibatching with "
        f"sequential LSTM replay"
    )
    minibatch_size = batch_size // args.num_minibatches
    action_dim = cfg.model.decoder.len_traj_pred * 2

    # ── action history ──
    n_action_history = args.n_action_history
    action_history_dim = n_action_history * action_dim

    # ── agent ──
    use_goal_only = args.agent == "goal_only"
    if use_goal_only:
        agent = GoalOnlyMLPAgent(
            hidden_dim=args.goal_only_hidden,
            action_dim=action_dim,
            num_layers=args.goal_only_layers,
            n_action_history=n_action_history,
        ).to(device)
    else:
        agent = PPOAgent(
            cfg,
            lstm_hidden_dim=args.lstm_hidden,
            lstm_num_layers=args.lstm_layers,
            use_decoder=args.use_decoder,
            n_action_history=n_action_history,
            goal_mode=args.goal_mode,
            norm_obs=args.norm_obs,
            encoder_type=args.encoder_type,
        ).to(device)
    optimizer = torch.optim.Adam(agent.trainable_parameters(), lr=args.lr, eps=1e-8)

    # ── wandb ──
    use_wandb = getattr(cfg, "logging", None) and getattr(cfg.logging, "enable_wandb", False)
    if args.no_wandb:
        use_wandb = False
    if use_wandb:
        import wandb
        wandb.init(
            project=cfg.project.name,
            name=cfg.project.run_name,
            config={
                "agent": args.agent,
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
                "goal_only_hidden": args.goal_only_hidden if use_goal_only else None,
                "goal_only_layers": args.goal_only_layers if use_goal_only else None,
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

    # ── pre-allocate rollout buffers ──
    bufs = RolloutBuffers(
        num_steps, num_envs, obs_shape, cord_shape, action_dim,
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

    # per-env rolling action history (flat vector)
    action_hist = np.zeros((num_envs, action_history_dim), dtype=np.float32)

    # per-env BEV images + metadata captured just before each vis rollout
    bev_images = [None] * num_envs
    bev_metas  = [None] * num_envs

    n_trainable = sum(p.numel() for p in agent.trainable_parameters())
    print(
        f"PPO | agent={args.agent} | {n_trainable:,} trainable params | "
        f"{num_envs} envs | {num_steps} steps/rollout | "
        f"batch {batch_size} | minibatch {minibatch_size}"
    )

    t_start = time.time()
    timer = StepTimer(cuda_sync=True)

    try:
        for iteration in range(1, args.num_iterations + 1):
            t0 = time.time()
            timer.reset()

            # optional LR annealing
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1) / args.num_iterations
                for pg in optimizer.param_groups:
                    pg["lr"] = frac * args.lr

            # ── capture overhead BEV image before this rollout ──────────
            # Obstacle/agent layouts are fetched first so that the
            # auto-altitude computation can include geodesic paths.
            is_vis_iter = args.vis_every > 0 and iteration % args.vis_every == 0
            obstacle_layouts = None
            if is_vis_iter:
                vec_env.set_collect_substep_frames(True)
                # Snapshot obstacle/pedestrian/agent layout NOW (before the
                # rollout moves agents) so ego/goal positions match the BEV.
                try:
                    obstacle_layouts = vec_env.get_obstacle_layouts()
                except Exception as e:
                    print(f"  [BEV-obstacle] layout query failed: {e}")
                    obstacle_layouts = None
                # Auto-altitude BEV capture (start + goal + geodesic)
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
            # per-env list of substep frame lists (only populated on vis iterations)
            substep_frames = [[] for _ in range(num_envs)] if is_vis_iter else None
            # per-env MPC/policy vis data (only on vis iterations)
            mpc_vis_data = [[] for _ in range(num_envs)] if is_vis_iter else None

            # ── rollout collection ──
            agent.eval()
            if not use_goal_only:
                agent.obs_encoder.eval()
                if args.use_decoder:
                    agent.action_decoder.eval()

            # Save LSTM state at rollout start for sequential replay in PPO update
            initial_lstm_state = (lstm_state[0].detach().clone(),
                                  lstm_state[1].detach().clone())

            for step in range(num_steps):
                global_step += num_envs

                # store current observation
                bufs.obs[step] = obs_dict["obs"]
                bufs.cord[step] = obs_dict["cord"]
                bufs.goal[step] = obs_dict["goal"]
                bufs.goal_world[step] = obs_dict.get("goal_world", obs_dict["goal"])

                # store current action history snapshot
                if action_history_dim > 0:
                    bufs.action_hist[step] = action_hist

                with torch.no_grad():
                    goal_t = torch.as_tensor(obs_dict["goal"], dtype=torch.float32, device=device)

                    # action history tensor (None when disabled)
                    ah_t = None
                    if action_history_dim > 0:
                        ah_t = torch.as_tensor(action_hist, dtype=torch.float32, device=device)

                    if use_goal_only:
                        features = None
                        dec_out = None
                        obs_t = None
                        cord_t = None
                    else:
                        obs_t = torch.as_tensor(obs_dict["obs"], device=device)
                        cord_t = torch.as_tensor(obs_dict["cord"], dtype=torch.float32, device=device)

                        # compute frozen encoder + decoder transformer once; cache on GPU
                        with timer("encoder"):
                            if use_simple_encoder:
                                features = agent.obs_encoder(obs_t[:, -1:])  # (num_envs, 1, 768)
                            else:
                                features = agent.obs_encoder(obs_t, cord_t)  # (num_envs, context_size+1, 768)
                        bufs.features_gpu[step] = features

                        if args.use_decoder:
                            with timer("decoder"):
                                tokens = agent.action_decoder.positional_encoding(features)
                                dec_out = agent.action_decoder.sa_decoder(tokens).mean(dim=1)  # (num_envs, 768)
                            bufs.dec_out_gpu[step] = dec_out
                        else:
                            dec_out = None

                    with timer("policy"):
                        action, logprob, _, value, lstm_state = (
                            agent.get_action_and_value(
                                obs_t, cord_t, goal_t, lstm_state,
                                features=features, dec_out=dec_out,
                                action_history=ah_t,
                            )
                        )

                actions_np = action.cpu().numpy()
                bufs.actions[step] = actions_np
                bufs.logprobs[step] = logprob.cpu().numpy()
                bufs.values[step] = value.cpu().numpy()

                # update rolling action history: shift left by action_dim, append new action
                if action_history_dim > 0:
                    if n_action_history > 1:
                        action_hist[:, :-action_dim] = action_hist[:, action_dim:]
                    action_hist[:, -action_dim:] = actions_np

                # The env now expects (num_envs, future_length * 2) waypoints.
                env_actions = actions_np
                with timer("env_step"):
                    obs_dict, rewards, terminateds, truncateds, infos = vec_env.step(env_actions)
                dones = np.logical_or(terminateds, truncateds).astype(np.float32)

                # Aggregate env-side timing (MPC, sim tick) from info dicts
                sim_ticks = [info.get('sim_tick_time', 0.0) for info in infos]
                if sim_ticks:
                    timer.record("sim_tick", np.mean(sim_ticks))
                if args.use_mpc:
                    mpc_times = [info.get('mpc_solve_time', 0.0) for info in infos]
                    if mpc_times:
                        timer.record("mpc_solve", np.mean(mpc_times))

                weather_tracker.update(infos)

                # collect substep frames on vis iterations
                if substep_frames is not None:
                    for i in range(num_envs):
                        sf = infos[i].get('substep_frames')
                        if sf is not None:
                            substep_frames[i].append(sf)

                # collect MPC/policy vis data on vis iterations
                if mpc_vis_data is not None:
                    for i in range(num_envs):
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

                # episode tracking (before reward clipping/normalization)
                ep_tracker.step(rewards, dones, infos)

                bufs.store_control_info(step, infos)

                bufs.raw_rewards[step] = np.copy(rewards)

                # For truncated (not terminated) episodes, compute the
                # bootstrap value from the terminal observation.
                # Truncation is artificial — the future value should not be
                # zero; instead, we bootstrap from V(s_terminal).
                for i in range(num_envs):
                    if truncateds[i] and not terminateds[i]:
                        term_obs = infos[i].get('terminal_observation')
                        if term_obs is not None:
                            with torch.no_grad():
                                t_goal = torch.as_tensor(
                                    term_obs["goal"][None], dtype=torch.float32, device=device)
                                t_ah = None
                                if action_history_dim > 0:
                                    t_ah = torch.as_tensor(
                                        action_hist[i:i+1], dtype=torch.float32, device=device)
                                if use_goal_only:
                                    tv = agent.get_value(
                                        None, None, t_goal,
                                        (lstm_state[0][:, i:i+1].contiguous(), lstm_state[1][:, i:i+1].contiguous()),
                                        action_history=t_ah,
                                    ).cpu().item()
                                else:
                                    t_obs = torch.as_tensor(
                                        term_obs["obs"][None], device=device)
                                    t_cord = torch.as_tensor(
                                        term_obs["cord"][None], dtype=torch.float32, device=device)
                                    tv = agent.get_value(
                                        t_obs, t_cord, t_goal,
                                        (lstm_state[0][:, i:i+1].contiguous(), lstm_state[1][:, i:i+1].contiguous()),
                                        action_history=t_ah,
                                    ).cpu().item()
                                bufs.trunc_values[step, i] = tv
                                bufs.has_trunc_value[step, i] = 1.0

                # reward clipping
                if args.reward_clip > 0:
                    rewards = np.clip(rewards, -args.reward_clip, args.reward_clip)

                # reward normalization (update stats with raw, store normalized)
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
                        # record post-reset spawn for the new episode
                        ep_accum.on_episode_end(i, obs_dict['cord'][i])

            # ── bootstrap value for end of rollout ──
            with timer("bootstrap"):
                with torch.no_grad():
                    goal_t = torch.as_tensor(obs_dict["goal"], dtype=torch.float32, device=device)
                    ah_t = None
                    if action_history_dim > 0:
                        ah_t = torch.as_tensor(action_hist, dtype=torch.float32, device=device)
                    if use_goal_only:
                        last_value = agent.get_value(
                            None, None, goal_t, lstm_state, action_history=ah_t
                        ).cpu().numpy()
                    else:
                        obs_t = torch.as_tensor(obs_dict["obs"], device=device)
                        cord_t = torch.as_tensor(obs_dict["cord"], dtype=torch.float32, device=device)
                        last_value = agent.get_value(
                            obs_t, cord_t, goal_t, lstm_state, action_history=ah_t
                        ).cpu().numpy()

            # ── Build next_values array for GAE ──
            # Default: next_values[t] = values[t+1], last step = last_value.
            # Override: at truncated (not terminated) steps, use the stored
            # terminal-state value so the bootstrap is correct.
            next_values = np.zeros_like(bufs.values)
            next_values[:-1] = bufs.values[1:]
            next_values[-1] = last_value
            # Overwrite with terminal bootstrap values where available
            mask = bufs.has_trunc_value > 0
            next_values[mask] = bufs.trunc_values[mask]

            # ── GAE ──
            with timer("gae"):
                advantages, returns = compute_gae(
                    bufs.rewards, bufs.values, bufs.terminateds, bufs.dones,
                    next_values,
                    gamma=args.gamma, gae_lambda=args.gae_lambda,
                )

            # Reset truncation buffers for next rollout
            bufs.trunc_values[:] = 0
            bufs.has_trunc_value[:] = 0

            # ── flatten + PPO update ──
            batch = bufs.flatten(
                advantages, returns, action_dim,
                context_size, obs_feat_dim,
                norm_adv=args.norm_adv,
            )
            with timer("ppo_update"):
                stats = ppo_update(agent, optimizer, batch, args,
                                   num_steps, num_envs,
                                   initial_lstm_state, device,
                                   use_goal_only=use_goal_only)

            # ── logging / vis / checkpoint ──
            ep_stats = ep_tracker.flush()
            # Merge env-level solvability stats into episode stats for
            # unified logging (reset counters each flush cycle).
            try:
                solv_stats = vec_env.get_solvability_stats(reset=True)
                if ep_stats is not None:
                    ep_stats.update({
                        'unsolvable_rate': solv_stats['unsolvable_rate'],
                        'unsolvable_episodes': solv_stats['unsolvable_episodes'],
                    })
                elif solv_stats.get('unsolvable_episodes', 0) > 0:
                    # No completed episodes this iteration, but we still
                    # want to surface solvability warnings.
                    ep_stats = solv_stats
            except Exception:
                pass  # env may not support solvability stats (e.g. PointNavEnv)
            should_log_wandb = (args.log_every > 0
                                and iteration % args.log_every == 0)
            ep_reward = log_metrics(iteration, args.num_iterations, global_step,
                                    bufs, stats, optimizer, t_start, t0, wandb,
                                    ep_stats=ep_stats,
                                    log_wandb=should_log_wandb,
                                    timer=timer)

            if is_vis_iter:
                _enc = agent.obs_encoder if not use_goal_only else None
                visualize(iteration, args, bufs, bev_images, bev_metas, wandb,
                          substep_frames=substep_frames, obs_encoder=_enc,
                          obstacle_layouts=obstacle_layouts,
                          mpc_vis_data=mpc_vis_data)
                vec_env.set_collect_substep_frames(False)

                # ── complete-episode BEV (retroactive capture at spawn positions)
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

                # ── obstacle layout BEV (dark background, obstacles + crosswalks + pedestrians)
                if obstacle_layouts is not None:
                    try:
                        vis_dir = os.path.join(args.save_dir, "vis")
                        log_obstacle_bev_figure(
                            obstacle_layouts,
                            bev_metas,
                            grid_cols=args.num_agents_per_server,
                            iteration=iteration,
                            save_dir=vis_dir,
                            wandb=wandb,
                        )
                    except Exception as e:
                        print(f"  [BEV-obstacle] obstacle layout vis failed: {e}")

                # ── geodesic distance field heatmap / video ──
                if obstacle_layouts is not None and args.dynamic_geo_mode != 'off':
                    try:
                        vis_dir = os.path.join(args.save_dir, "vis")
                        log_geodesic_field(
                            obstacle_layouts,
                            bev_metas,
                            grid_cols=args.num_agents_per_server,
                            iteration=iteration,
                            save_dir=vis_dir,
                            wandb=wandb,
                            video_fps=args.vis_video_fps,
                        )
                    except Exception as e:
                        print(f"  [BEV-geodesic] geodesic field vis failed: {e}")

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
                best_reward = save_checkpoint(iteration, args, agent, optimizer,
                                              global_step, ep_reward, best_reward)

    finally:
        # Mask SIGINT so a second Ctrl+C cannot interrupt cleanup
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("\nCleaning up (Ctrl+C disabled during shutdown)...")

        try:
            vec_env.close()
        except Exception as e:
            print(f"  vec_env.close() error: {e}")

        # Use vec_env.server_procs (may have been updated during restarts)
        final_procs = getattr(vec_env, 'server_procs', server_procs)
        if final_procs and any(p is not None for p in final_procs):
            print("Stopping CARLA servers ...")
            stop_carla_servers([p for p in final_procs if p is not None])

        if wandb is not None:
            wandb.finish()

        signal.signal(signal.SIGINT, signal.SIG_DFL)

    print("Training complete.")


# ─── Entry Point ──────────────────────────────────────────────────────


def _parse_gpu_ids(gpu_ids_str, num_envs):
    """
    Parse --gpu_ids into a list of ints.
    If not provided, defaults to [0, 1, 2, ..., num_envs-1].
    """
    if gpu_ids_str:
        ids = [int(x.strip()) for x in gpu_ids_str.split(",")]
        '''
        if len(ids) < num_envs:
            raise ValueError(
                f"--gpu_ids provides {len(ids)} IDs but --num_envs={num_envs}. "
                f"Supply at least {num_envs} GPU IDs (may repeat for sharing)."
            )
        '''
        return ids[:num_envs]
    return list(range(num_envs))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="PPO trainer for CARLA")
    # environment
    parser.add_argument("--config", type=str, default="config/rl.yaml")
    parser.add_argument("--base_port", type=int, default=2000)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--max_speed", type=float, default=1.4)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--max_episode_steps", type=int, default=128)
    parser.add_argument("--teleport", action="store_true", default=False,
                        help="Use direct set_transform control instead of WalkerControl "
                             "(eliminates physics lag between cmd/real velocity)")
    parser.add_argument("--use_mpc", action="store_true", default=False,
                        help="Enable MPC waypoint tracking (unicycle model). The MPC is "
                             "JIT-compiled once at env creation and persists across resets.")
    
    parser.add_argument("--goal_range", type=float, default=40.0)
    parser.add_argument("--quadrant_margin", type=float, default=None,
                        help="Inset margin (metres) for spawn/goal sampling within each "
                             "quadrant.  Defaults to goal_range.  Prevents agents from "
                             "needing to leave their quadrant to reach the goal.")
    # procedural obstacles
    parser.add_argument("--obstacles", action="store_true", default=False,
                        help="Enable procedural obstacle generation (blocked crosswalks, "
                             "sidewalk obstructions, narrow passages)")
    parser.add_argument("--p_crosswalk_challenge", type=float, default=0.3,
                        help="Per-episode probability of blocked-crosswalk scenario")
    parser.add_argument("--pedestrians", action="store_true", default=False,
                        help="Enable SFM-controlled pedestrians")
    parser.add_argument("--num_pedestrians_per_region", type=int, default=30,
                        help="Number of SFM pedestrians per quadrant")
    parser.add_argument("--dynamic_geo_mode", type=str, default="off",
                        choices=["off", "soft", "timespace"],
                        help="Dynamic geodesic reward mode. "
                             "'soft': heuristic soft-cost swept volume; "
                             "'timespace': exact time-space backward DP. "
                             "Requires --pedestrians.")
    parser.add_argument("--dynamic_geo_horizon", type=float, default=5.0,
                        help="Prediction horizon (seconds) for dynamic geodesic reward")
    # weather randomisation
    parser.add_argument("--weather", action="store_true", default=False,
                        help="Randomize weather and sun position each episode")
    # navmesh cache
    parser.add_argument("--navmesh_cache_dir", type=str, default=None,
                        help="Directory containing precomputed navmesh cache NPZ files "
                             "(from export_carla_navmesh.py --cache-all). "
                             "Speeds up crosswalk detection and walkable-area sampling.")
    # multi-agent (quadrant split)
    parser.add_argument("--num_agents_per_server", type=int, default=4,
                        help="Number of agents per CARLA server (quadrant split). "
                             "Total rollout envs = num_envs x num_agents_per_server.")
    parser.add_argument("--towns", type=str, nargs="+",
                        default=["Town02", "Town03", "Town05", "Town10HD"],
                        help="CARLA town names to randomly cycle through.")
    parser.add_argument("--map_change_interval", type=int, default=0,
                        help="Change map every N completed episodes per server "
                             "(0 = only on full reset).")
    # CARLA server auto-launch (optional — omit --carla_bin to connect manually)
    parser.add_argument("--carla_bin", type=str, default=None,
                        help="Path to CarlaUnreal.sh. If set, servers are auto-launched.")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Comma-separated GPU IDs for CARLA servers (e.g. 0,1,2,3). "
                             "Defaults to 0..num_envs-1.")
    parser.add_argument("--carla_startup_wait", type=int, default=30,
                        help="Seconds to wait after launching CARLA servers.")
    parser.add_argument("--carla_stagger_delay", type=int, default=5,
                        help="Seconds to wait between launching each CARLA server "
                             "to avoid Vulkan init contention (0 = no stagger).")
    # PPO
    parser.add_argument("--num_steps", type=int, default=128, help="rollout length per env")
    parser.add_argument("--num_iterations", type=int, default=10000)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--lr", type=float, default=25e-5)
    parser.add_argument("--anneal_lr", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=0.2)
    parser.add_argument("--clip_vloss", action="store_true", default=True,
                        help="Clip value function loss (PPO-style)")
    parser.add_argument("--norm_reward", action="store_true", default=True,
                        help="Normalize rewards with running statistics")
    parser.add_argument("--norm-adv", action=argparse.BooleanOptionalAction, default=True,
                        help="Normalize advantages (--norm-adv / --no-norm-adv)")
    parser.add_argument("--use_decoder", action="store_true", default=False,
                        help="Use pretrained decoder to produce actions"
                        )
    parser.add_argument("--target_kl", type=float, default=None,
                        help="KL-based early stopping threshold for PPO epochs "
                             "(None = disabled, typical values: 0.01-0.02)")
    parser.add_argument("--reward_clip", type=float, default=0,
                        help="Clip rewards to [-val, val] (0 = disabled)")
    # model
    parser.add_argument("--agent", type=str, default="ppo", choices=["ppo", "goal_only"],
                        help="Agent type: 'ppo' (full DINOv2+LSTM) or 'goal_only' (MLP baseline)")
    parser.add_argument("--encoder_type", type=str, default="full",
                        choices=["full", "simple"],
                        help="Observation encoder: 'full' (DINOv2 + history + cord) "
                             "or 'simple' (DINOv2 on current frame only)")
    parser.add_argument("--lstm_hidden", type=int, default=512)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--goal_only_hidden", type=int, default=64,
                        help="Hidden dim for goal_only MLP agent")
    parser.add_argument("--goal_only_layers", type=int, default=2,
                        help="Number of hidden layers for goal_only MLP agent")
    parser.add_argument("--n_action_history", type=int, default=0,
                        help="Number of past actions to feed as input (0 = disabled)")
    parser.add_argument("--goal_mode", type=str, default='both',
                        choices=["lstm", "heads", "both"],
                        help="Where to inject goal: lstm, heads, or both ")
    parser.add_argument("--norm_obs", action="store_true",
                        help="Apply LayerNorm to LSTM input features "
                             "(stabilizes training with frozen DINOv2 features)")
    # checkpointing
    parser.add_argument("--save_dir", type=str, default="checkpoints/ppo")
    parser.add_argument("--save_every", type=int, default=100)
    # logging
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging even if enabled in config")
    parser.add_argument("--log_every", type=int, default=1,
                        help="Log numerical metrics to W&B every N iterations "
                             "(0 = disable W&B metrics logging). "
                             "Console output is printed every iteration regardless.")
    # visualisation
    parser.add_argument("--vis_every", type=int, default=100,
                        help="Render BEV trajectory plot every N iterations "
                             "(0 = disabled). Saved to <save_dir>/vis/ and "
                             "uploaded to W&B when enabled.")
    parser.add_argument("--bev_altitude", type=float, default=15.0,
                        help="Minimum BEV camera altitude (m). Actual altitude "
                             "is computed automatically to encompass start, "
                             "goal, and geodesic path.")
    parser.add_argument("--bev_fov", type=float, default=90.0,
                        help="Horizontal field-of-view (degrees) of the BEV camera.")
    parser.add_argument("--bev_img_size", type=int, default=512,
                        help="Side length (pixels) of the square BEV image.")
    parser.add_argument("--vis_video_fps", type=int, default=1,
                        help="Playback FPS for the ego-view video. "
                             "Matches the policy step rate (default 5 Hz).")

    train(parser.parse_args())
