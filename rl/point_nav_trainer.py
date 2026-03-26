"""
Minimal PPO trainer for the PointNav single-integrator environment.

No CARLA, no images, no encoder/decoder, no subprocesses.
Uses GoalOnlyMLPAgent and VecPointNavEnv — everything runs in-process.

Usage:
    python -m rl.point_nav_trainer
    python -m rl.point_nav_trainer --num_envs 16 --num_steps 128 --lr 3e-4
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection

from rl.envs.point_nav import VecPointNavEnv
from rl.goal_only_agent import GoalOnlyMLPAgent


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


# ─── Episode Tracker ─────────────────────────────────────────────────
class EpisodeTracker:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self._ep_return = np.zeros(num_envs, dtype=np.float64)
        self._ep_length = np.zeros(num_envs, dtype=np.int64)
        self._completed = []

    def step(self, rewards, dones, infos):
        self._ep_return += rewards
        self._ep_length += 1
        for i in range(self.num_envs):
            if dones[i]:
                self._completed.append({
                    "return": float(self._ep_return[i]),
                    "length": int(self._ep_length[i]),
                    "is_success": bool(infos[i].get("is_success", False)),
                    "initial_distance": float(infos[i].get("initial_distance", 0.0)),
                    "path_length": float(infos[i].get("path_length", 0.0)),
                    "final_distance": float(infos[i].get("distance_to_goal", 0.0)),
                })
                self._ep_return[i] = 0.0
                self._ep_length[i] = 0

    def flush(self):
        if not self._completed:
            return None
        n = len(self._completed)
        success_rate = np.mean([e["is_success"] for e in self._completed])

        spl_terms = []
        for e in self._completed:
            li = e["initial_distance"]
            pi = e["path_length"]
            si = float(e["is_success"])
            denom = max(pi, li)
            spl_terms.append(si * li / denom if denom > 0 else 0.0)
        spl = float(np.mean(spl_terms))

        stats = {
            "success_rate": float(success_rate),
            "spl": spl,
            "mean_episode_return": float(np.mean([e["return"] for e in self._completed])),
            "mean_episode_length": float(np.mean([e["length"] for e in self._completed])),
            "mean_final_distance": float(np.mean([e["final_distance"] for e in self._completed])),
            "num_episodes": n,
        }
        self._completed.clear()
        return stats


# ─── GAE ──────────────────────────────────────────────────────────────
def compute_gae(rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95):
    T = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros(rewards.shape[1])

    for t in reversed(range(T)):
        next_val = next_value if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# ─── PPO Update ───────────────────────────────────────────────────────
def ppo_update(agent, optimizer, batch, args, batch_size, minibatch_size, device):
    agent.train()
    pg_losses, v_losses, ent_losses, clip_fracs = [], [], [], []

    for _ in range(args.num_epochs):
        indices = np.random.permutation(batch_size)
        for start in range(0, batch_size, minibatch_size):
            mb = indices[start : start + minibatch_size]

            mb_goal = torch.as_tensor(batch["goal"][mb], dtype=torch.float32, device=device)
            mb_actions = torch.as_tensor(batch["actions"][mb], dtype=torch.float32, device=device)
            mb_old_lp = torch.as_tensor(batch["logprobs"][mb], dtype=torch.float32, device=device)
            mb_old_val = torch.as_tensor(batch["values"][mb], dtype=torch.float32, device=device)
            mb_adv = torch.as_tensor(batch["advantages"][mb], dtype=torch.float32, device=device)
            mb_ret = torch.as_tensor(batch["returns"][mb], dtype=torch.float32, device=device)

            # dummy LSTM state
            mb_h = torch.zeros(1, len(mb), 1, device=device)
            mb_c = torch.zeros(1, len(mb), 1, device=device)

            _, new_lp, entropy, new_val, _ = agent.get_action_and_value(
                None, None, mb_goal, (mb_h, mb_c), action=mb_actions,
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

    return {
        "pg_loss": np.mean(pg_losses),
        "vf_loss": np.mean(v_losses),
        "entropy": -np.mean(ent_losses),
        "clip_frac": np.mean(clip_fracs),
    }


# ─── Visualization ────────────────────────────────────────────────────
def visualize_trajectories(
    buf_xz, buf_goal, buf_actions, buf_raw_rewards, buf_dones,
    iteration, save_dir, wandb=None, max_envs=4,
):
    """
    2D trajectory plot + action/distance time-series for the latest rollout.

    Layout per env (up to max_envs):
      Col 0: 2D trajectory — reward-coloured path, start/end/goal markers
      Col 1: action components (vx, vz) over time
      Col 2: distance-to-goal over time
    """
    num_steps, num_envs = buf_raw_rewards.shape
    n_show = min(num_envs, max_envs)

    r_min, r_max = float(buf_raw_rewards.min()), float(buf_raw_rewards.max())
    if r_max - r_min < 1e-8:
        r_max = r_min + 1e-8
    norm = Normalize(vmin=r_min, vmax=r_max)
    cmap = cm.RdYlGn

    fig, axes = plt.subplots(n_show, 3, figsize=(15, 4.5 * n_show), squeeze=False)
    fig.suptitle(f"PointNav Trajectories — Iteration {iteration}", fontsize=14)

    for env_idx in range(n_show):
        xs = buf_xz[:, env_idx, 0]
        zs = buf_xz[:, env_idx, 1]
        rs = buf_raw_rewards[:, env_idx]
        dones = buf_dones[:, env_idx]
        goals_global = buf_xz[:, env_idx] + buf_goal[:, env_idx]

        ep_starts = [0] + (np.where(dones[:-1] > 0)[0] + 1).tolist()
        ep_ends = np.where(dones > 0)[0].tolist() + [num_steps - 1]

        # ── Col 0: 2D trajectory ──────────────────────────────────────
        ax = axes[env_idx, 0]
        for ep_i, (t0, t1) in enumerate(zip(ep_starts, ep_ends)):
            ep_xz = np.stack([xs[t0:t1+1], zs[t0:t1+1]], axis=-1)
            ep_rs = rs[t0:t1+1]
            L = len(ep_xz)

            if L >= 2:
                points = ep_xz.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                seg_colors = cmap(norm(ep_rs[:-1]))
                lc = LineCollection(segments, colors=seg_colors, linewidths=2.0, zorder=3)
                ax.add_collection(lc)

            ax.scatter(*ep_xz[0], marker="o", s=60, c="steelblue", zorder=5,
                       label="start" if ep_i == 0 else None)
            ax.scatter(*ep_xz[-1], marker="s", s=60, c="black", zorder=5,
                       label="end" if ep_i == 0 else None)

            goal_xz = goals_global[t0]
            ax.scatter(*goal_xz, marker="*", s=200, c="gold",
                       edgecolors="darkorange", linewidths=0.8, zorder=6,
                       label="goal" if ep_i == 0 else None)
            ax.plot([ep_xz[0, 0], goal_xz[0]], [ep_xz[0, 1], goal_xz[1]],
                    "--", color="orange", alpha=0.5, linewidth=1.0, zorder=2)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Env {env_idx} | {len(ep_starts)} ep(s)")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.7)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="reward", shrink=0.8)

        # ── Col 1: actions over time ──────────────────────────────────
        ax = axes[env_idx, 1]
        t_axis = np.arange(num_steps)
        ax.plot(t_axis, buf_actions[:, env_idx, 0], label="vx", color="tab:blue")
        ax.plot(t_axis, buf_actions[:, env_idx, 1], label="vz", color="tab:red")
        for td in np.where(dones > 0)[0]:
            ax.axvline(td, color="gray", linestyle="--", alpha=0.4)
        ax.set_xlabel("step")
        ax.set_ylabel("action (m/s)")
        ax.set_title(f"Env {env_idx} — Actions")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # ── Col 2: distance to goal over time ─────────────────────────
        ax = axes[env_idx, 2]
        dist = np.linalg.norm(buf_goal[:, env_idx], axis=-1)
        ax.plot(t_axis, dist, color="tab:purple", linewidth=1.5)
        ax.axhline(0.2, color="green", linestyle="--", alpha=0.6, label="goal radius")
        for td in np.where(dones > 0)[0]:
            ax.axvline(td, color="gray", linestyle="--", alpha=0.4)
        ax.set_xlabel("step")
        ax.set_ylabel("distance (m)")
        ax.set_title(f"Env {env_idx} — Distance to Goal")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir is not None:
        vis_dir = os.path.join(save_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)
        path = os.path.join(vis_dir, f"traj_{iteration:04d}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"  → trajectory figure saved to {path}")

    if wandb is not None:
        wandb.log({"rollout/trajectories": wandb.Image(fig)}, step=iteration)

    plt.close(fig)


# ─── Training Loop ────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vec_env = VecPointNavEnv(
        num_envs=args.num_envs,
        dt=args.dt,
        max_speed=args.max_speed,
        goal_radius=args.goal_radius,
        max_steps=args.max_episode_steps,
        arena_size=args.arena_size,
        gamma=args.gamma,
        seed=args.seed,
    )

    num_envs = args.num_envs
    num_steps = args.num_steps
    batch_size = num_steps * num_envs
    minibatch_size = batch_size // args.num_minibatches
    action_dim = 2

    agent = GoalOnlyMLPAgent(
        hidden_dim=args.hidden_dim,
        action_dim=action_dim,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(agent.trainable_parameters(), lr=args.lr, eps=1e-5)

    # ── wandb ──
    wandb = None
    if args.wandb:
        import wandb as wandb_module
        wandb = wandb_module
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
        )

    # ── reward normalization ──
    reward_rms = RunningMeanStd() if args.norm_reward else None

    # ── pre-allocate rollout buffers (simple numpy arrays) ──
    buf_goal = np.zeros((num_steps, num_envs, 2), dtype=np.float32)
    buf_xz = np.zeros((num_steps, num_envs, 2), dtype=np.float32)
    buf_actions = np.zeros((num_steps, num_envs, action_dim), dtype=np.float32)
    buf_logprobs = np.zeros((num_steps, num_envs), dtype=np.float32)
    buf_rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
    buf_raw_rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
    buf_dones = np.zeros((num_steps, num_envs), dtype=np.float32)
    buf_values = np.zeros((num_steps, num_envs), dtype=np.float32)

    ep_tracker = EpisodeTracker(num_envs)

    # ── kick off ──
    obs_dict, _ = vec_env.reset()
    lstm_state = agent.get_initial_lstm_state(num_envs, device)
    global_step = 0
    best_reward = -float("inf")

    n_trainable = sum(p.numel() for p in agent.trainable_parameters())
    print(
        f"PointNav PPO | {n_trainable:,} params | "
        f"{num_envs} envs | {num_steps} steps/rollout | "
        f"batch {batch_size} | minibatch {minibatch_size} | "
        f"dt={args.dt} | arena={args.arena_size}m"
    )

    t_start = time.time()

    for iteration in range(1, args.num_iterations + 1):
        t0 = time.time()

        # optional LR annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1) / args.num_iterations
            for pg in optimizer.param_groups:
                pg["lr"] = frac * args.lr

        # ── rollout collection ──
        agent.eval()
        for step in range(num_steps):
            global_step += num_envs

            buf_goal[step] = obs_dict["goal"]
            buf_xz[step] = vec_env.pos.copy()

            with torch.no_grad():
                goal_t = torch.as_tensor(obs_dict["goal"], dtype=torch.float32, device=device)
                action, logprob, _, value, lstm_state = agent.get_action_and_value(
                    None, None, goal_t, lstm_state,
                )

            actions_np = action.cpu().numpy()
            buf_actions[step] = actions_np
            buf_logprobs[step] = logprob.cpu().numpy()
            buf_values[step] = value.cpu().numpy()

            obs_dict, rewards, terminateds, truncateds, infos = vec_env.step(actions_np)
            dones = np.maximum(terminateds, truncateds)

            ep_tracker.step(rewards, dones, infos)

            buf_raw_rewards[step] = rewards.copy()

            if args.reward_clip > 0:
                rewards = np.clip(rewards, -args.reward_clip, args.reward_clip)

            if reward_rms is not None:
                reward_rms.update(rewards)
                rewards = reward_rms.normalize(rewards)

            buf_rewards[step] = rewards
            buf_dones[step] = dones

        # ── bootstrap value ──
        with torch.no_grad():
            goal_t = torch.as_tensor(obs_dict["goal"], dtype=torch.float32, device=device)
            next_value = agent.get_value(None, None, goal_t, lstm_state).cpu().numpy()

        # ── GAE ──
        advantages, returns = compute_gae(
            buf_rewards, buf_values, buf_dones, next_value,
            gamma=args.gamma, gae_lambda=args.gae_lambda,
        )

        # ── flatten for PPO ──
        b_advantages = advantages.reshape(batch_size)
        if args.norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        batch = {
            "goal": buf_goal.reshape(batch_size, 2),
            "actions": buf_actions.reshape(batch_size, action_dim),
            "logprobs": buf_logprobs.reshape(batch_size),
            "values": buf_values.reshape(batch_size),
            "advantages": b_advantages,
            "returns": returns.reshape(batch_size),
        }

        stats = ppo_update(agent, optimizer, batch, args,
                           batch_size, minibatch_size, device)

        # ── logging ──
        ep_stats = ep_tracker.flush()
        iter_time = time.time() - t0
        total_time = time.time() - t_start
        sps = int(global_step / total_time)
        ep_reward = buf_rewards.sum(axis=0).mean()
        raw_ep_reward = buf_raw_rewards.sum(axis=0).mean()
        cur_lr = optimizer.param_groups[0]["lr"]

        h, rem = divmod(int(total_time), 3600)
        m, s = divmod(rem, 60)

        print(
            f"[{iteration:4d}/{args.num_iterations}]  "
            f"steps={global_step:>8d}  "
            f"reward={ep_reward:+.2f}  "
            f"raw={raw_ep_reward:+.2f}  "
            f"pg={stats['pg_loss']:.4f}  "
            f"vf={stats['vf_loss']:.4f}  "
            f"ent={stats['entropy']:.4f}  "
            f"clip={stats['clip_frac']:.3f}  "
            f"lr={cur_lr:.2e}  "
            f"t={iter_time:.1f}s  "
            f"total={h}:{m:02d}:{s:02d}  "
            f"SPS={sps}"
        )

        if ep_stats is not None:
            print(
                f"  episodes: n={ep_stats['num_episodes']}  "
                f"success={ep_stats['success_rate']:.2f}  "
                f"spl={ep_stats['spl']:.3f}  "
                f"ep_return={ep_stats['mean_episode_return']:+.2f}  "
                f"ep_len={ep_stats['mean_episode_length']:.1f}  "
                f"final_dist={ep_stats['mean_final_distance']:.2f}"
            )

        if wandb is not None:
            log_dict = {
                "rollout/reward": ep_reward,
                "rollout/raw_reward": raw_ep_reward,
                "losses/policy": stats["pg_loss"],
                "losses/value": stats["vf_loss"],
                "losses/entropy": stats["entropy"],
                "losses/clip_frac": stats["clip_frac"],
                "train/lr": cur_lr,
                "train/iteration": iteration,
                "train/global_step": global_step,
                "time/SPS": sps,
            }
            if ep_stats is not None:
                for k, v in ep_stats.items():
                    log_dict[f"episode/{k}"] = v
            wandb.log(log_dict, step=global_step)

        # ── visualization ──
        if args.vis_every > 0 and iteration % args.vis_every == 0:
            visualize_trajectories(
                buf_xz, buf_goal, buf_actions, buf_raw_rewards, buf_dones,
                iteration=iteration, save_dir=args.save_dir, wandb=wandb,
            )

        # ── checkpoint ──
        if args.save_every > 0 and iteration % args.save_every == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            ckpt = {
                "iteration": iteration,
                "global_step": global_step,
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "reward": float(ep_reward),
            }
            torch.save(ckpt, os.path.join(args.save_dir, "last.pt"))
            if ep_reward > best_reward:
                best_reward = ep_reward
                torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))
                print(f"  → new best reward={ep_reward:+.2f}")

    if wandb is not None:
        wandb.finish()

    print("Training complete.")


# ─── Entry Point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO trainer for PointNav")

    # environment
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--max_speed", type=float, default=1.4)
    parser.add_argument("--goal_radius", type=float, default=0.2)
    parser.add_argument("--max_episode_steps", type=int, default=64)
    parser.add_argument("--arena_size", type=float, default=8.0,
                        help="Goal distance range: [0.8*arena, arena]")
    parser.add_argument("--seed", type=int, default=0)

    # model
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)

    # PPO
    parser.add_argument("--num_steps", type=int, default=128, help="rollout length per env")
    parser.add_argument("--num_iterations", type=int, default=500)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--anneal_lr", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--clip_vloss", action="store_true", default=True)
    parser.add_argument("--norm_reward", action="store_true", default=False)
    parser.add_argument("--norm_adv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reward_clip", type=float, default=10.0)

    # checkpointing
    parser.add_argument("--save_dir", type=str, default="checkpoints/point_nav")
    parser.add_argument("--save_every", type=int, default=100)

    # logging / visualization
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="point-nav")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--vis_every", type=int, default=50,
                        help="Save trajectory figure every N iterations (0=disabled)")

    train(parser.parse_args())
