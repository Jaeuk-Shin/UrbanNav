import time
import numpy as np



# ─── Logging ─────────────────────────────────────────────────────────
def log_metrics(iteration, num_iterations, global_step, bufs, stats,
                optimizer, t_start, t0, wandb_module, ep_stats=None,
                log_wandb=True, timer=None):
    """Print console summary and log to W&B. Returns mean episode reward."""
    iter_time = time.time() - t0
    total_time = time.time() - t_start
    sps = int(global_step / total_time)
    ep_reward = bufs.rewards.sum(axis=0).mean()
    raw_ep_reward = bufs.raw_rewards.sum(axis=0).mean()
    cur_lr = optimizer.param_groups[0]["lr"]

    # format total elapsed as H:MM:SS
    h, rem = divmod(int(total_time), 3600)
    m, s = divmod(rem, 60)

    kl_str = ""
    if "approx_kl" in stats:
        kl_str = f"  kl={stats['approx_kl']:.4f}"
        if "stopped_epoch" in stats and stats["stopped_epoch"] < stats.get("num_epochs", 999):
            kl_str += f"(stopped@{stats['stopped_epoch']})"

    print(
        f"[{iteration:4d}/{num_iterations}]  "
        f"steps={global_step:>8d}  "
        f"reward={ep_reward:+.2f}  "
        f"raw reward={raw_ep_reward:+.2f}  "
        f"pg={stats['pg_loss']:.4f}  "
        f"vf={stats['vf_loss']:.4f}  "
        f"ent={stats['entropy']:.4f}  "
        f"clip={stats['clip_frac']:.3f}"
        f"{kl_str}  "
        f"lr={cur_lr:.2e}  "
        f"t={iter_time:.1f}s  "
        f"total={h}:{m:02d}:{s:02d}  "
        f"SPS={sps}"
    )

    if ep_stats is not None:
        col_str = ""
        if ep_stats.get('obstacle_collision_rate', 0) > 0 or \
           ep_stats.get('pedestrian_collision_rate', 0) > 0:
            col_str = (
                f"\n  collisions: "
                f"obs={ep_stats['mean_obstacle_collisions']:.1f}/ep "
                f"({ep_stats['obstacle_collision_rate']:.0%})  "
                f"ped={ep_stats['mean_pedestrian_collisions']:.1f}/ep "
                f"({ep_stats['pedestrian_collision_rate']:.0%})"
            )
        solv_str = ""
        if ep_stats.get('unsolvable_rate', 0) > 0:
            solv_str = (
                f"\n  solvability: "
                f"unsolvable={ep_stats.get('unsolvable_episodes', 0)}  "
                f"rate={ep_stats['unsolvable_rate']:.1%}  "
                f"retries={ep_stats.get('mean_goal_retries', 0):.2f}/ep"
            )
        spawn_str = ""
        obs_fail = ep_stats.get('obstacle_spawn_failed', 0)
        if obs_fail > 0:
            obs_req = ep_stats.get('obstacle_spawn_requested', 0)
            spawn_str = (
                f"\n  obstacle spawns: "
                f"failed={obs_fail}/{obs_req}  "
                f"rate={ep_stats.get('obstacle_spawn_fail_rate', 0):.1%}"
            )
        print(
            f"  episodes: n={ep_stats['num_episodes']}  "
            f"success={ep_stats['success_rate']:.2f}  "
            f"spl={ep_stats['spl']:.3f}  "
            f"ep_return={ep_stats['mean_episode_return']:+.2f}  "
            f"ep_len={ep_stats['mean_episode_length']:.1f}  "
            f"final_dist={ep_stats['mean_final_distance']:.2f}"
            f"{col_str}"
            f"{solv_str}"
            f"{spawn_str}"
        )

    # Print timing summary to console
    if timer is not None:
        timing_line = timer.summary_str()
        if timing_line:
            print(timing_line)

    if wandb_module is not None and log_wandb:
        log_dict = {
            "rollout/reward": ep_reward,
            "rollout/reward_per_step": bufs.rewards.mean(),
            "rollout/raw_reward": raw_ep_reward,
            "rollout/raw_reward_per_step": bufs.raw_rewards.mean(),
            "losses/policy": stats["pg_loss"],
            "losses/value": stats["vf_loss"],
            "losses/entropy": stats["entropy"],
            "losses/clip_frac": stats["clip_frac"],
            "losses/approx_kl": stats.get("approx_kl", 0.0),
            "train/stopped_epoch": stats.get("stopped_epoch", 0),
            "control/cmd_speed_mean": float(bufs.cmd_speed.mean()),
            "control/real_speed_mean": float(bufs.real_speed.mean()),
            "control/speed_error_mean": float(np.abs(bufs.cmd_speed - bufs.real_speed).mean()),
            "control/vel_error_mean": float(np.linalg.norm(bufs.cmd_vel - bufs.real_vel, axis=-1).mean()),
            "train/lr": cur_lr,
            "train/iteration": iteration,
            "train/global_step": global_step,
            "time/iter_time": iter_time,
            "time/total_time": total_time,
            "time/SPS": sps,
        }
        if ep_stats is not None:
            for k, v in ep_stats.items():
                log_dict[f"episode/{k}"] = v
        # Per-component timing stats
        if timer is not None:
            log_dict.update(timer.wandb_dict())
        wandb_module.log(log_dict, step=global_step)

    return ep_reward


def log_eval_metrics(eval_stats, global_step, wandb_module, log_wandb=True):
    """Print eval summary and log to W&B with eval/ prefix."""
    if eval_stats is None:
        return
    col_str = ""
    if eval_stats.get('obstacle_collision_rate', 0) > 0 or \
       eval_stats.get('pedestrian_collision_rate', 0) > 0:
        col_str = (
            f"  obs={eval_stats['mean_obstacle_collisions']:.1f}/ep "
            f"({eval_stats['obstacle_collision_rate']:.0%})  "
            f"ped={eval_stats['mean_pedestrian_collisions']:.1f}/ep "
            f"({eval_stats['pedestrian_collision_rate']:.0%})"
        )
    print(
        f"  [EVAL] n={eval_stats['num_episodes']}  "
        f"success={eval_stats['success_rate']:.2f}  "
        f"spl={eval_stats['spl']:.3f}  "
        f"return={eval_stats['mean_episode_return']:+.2f}  "
        f"len={eval_stats['mean_episode_length']:.1f}  "
        f"dist={eval_stats['mean_final_distance']:.2f}"
        f"{col_str}"
    )
    if wandb_module is not None and log_wandb:
        log_dict = {}
        for k, v in eval_stats.items():
            log_dict[f"eval/{k}"] = v
        wandb_module.log(log_dict, step=global_step)


# ─── Episode Tracker ─────────────────────────────────────────────────
class EpisodeTracker:
    """Track per-env episode returns/lengths and compute SPL on flush."""

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self._ep_return = np.zeros(num_envs, dtype=np.float64)
        self._ep_length = np.zeros(num_envs, dtype=np.int64)
        self._ep_obs_collisions = np.zeros(num_envs, dtype=np.int64)
        self._ep_ped_collisions = np.zeros(num_envs, dtype=np.int64)
        self._completed = []  # list of dicts

    def step(self, rewards, dones, infos):
        """Call after vec_env.step(), before reward clipping."""
        self._ep_return += rewards
        self._ep_length += 1
        for i in range(self.num_envs):
            if infos[i].get('obstacle_collision', False):
                self._ep_obs_collisions[i] += 1
            if infos[i].get('pedestrian_collision', False):
                self._ep_ped_collisions[i] += 1
            if dones[i]:
                self._completed.append({
                    'return': float(self._ep_return[i]),
                    'length': int(self._ep_length[i]),
                    'is_success': bool(infos[i].get('is_success', False)),
                    'initial_distance': float(infos[i].get('initial_distance', 0.0)),
                    'initial_geodesic_distance': float(
                        infos[i].get('initial_geodesic_distance', 0.0)),
                    'path_length': float(infos[i].get('path_length', 0.0)),
                    'final_distance': float(infos[i].get('distance_to_goal', 0.0)),
                    'goal_retries': int(infos[i].get('goal_retries', 0)),
                    'obstacle_collisions': int(self._ep_obs_collisions[i]),
                    'pedestrian_collisions': int(self._ep_ped_collisions[i]),
                })
                self._ep_return[i] = 0.0
                self._ep_length[i] = 0
                self._ep_obs_collisions[i] = 0
                self._ep_ped_collisions[i] = 0

    def reset_in_progress(self):
        """Zero per-env accumulators for in-progress episodes.

        Call after resetting environments to discard stale partial-episode
        state (e.g. when inserting a dedicated eval phase that resets envs).
        """
        self._ep_return[:] = 0
        self._ep_length[:] = 0
        self._ep_obs_collisions[:] = 0
        self._ep_ped_collisions[:] = 0

    def flush(self):
        """Return aggregated stats and clear buffer. Returns None if no episodes completed."""
        if not self._completed:
            return None
        n = len(self._completed)
        success_rate = np.mean([e['is_success'] for e in self._completed])

        # SPL: (1/N) * Σ S_i * L_i / max(P_i, L_i)
        # Uses geodesic initial distance (L_i) when available, falling
        # back to Euclidean.
        spl_terms = []
        for e in self._completed:
            geo_li = e.get('initial_geodesic_distance', 0.0)
            li = geo_li if geo_li > 0 else e['initial_distance']
            pi = e['path_length']
            si = float(e['is_success'])
            denom = max(pi, li)
            spl_terms.append(si * li / denom if denom > 0 else 0.0)
        spl = float(np.mean(spl_terms))

        obs_cols = [e['obstacle_collisions'] for e in self._completed]
        ped_cols = [e['pedestrian_collisions'] for e in self._completed]
        retries = [e.get('goal_retries', 0) for e in self._completed]

        geo_dists = [e.get('initial_geodesic_distance', 0.0)
                     for e in self._completed]

        stats = {
            'success_rate': float(success_rate),
            'spl': spl,
            'mean_episode_return': float(np.mean([e['return'] for e in self._completed])),
            'mean_episode_length': float(np.mean([e['length'] for e in self._completed])),
            'mean_final_distance': float(np.mean([e['final_distance'] for e in self._completed])),
            'mean_initial_geodesic_distance': float(np.mean(geo_dists)),
            'mean_goal_retries': float(np.mean(retries)),
            'num_episodes': n,
            'obstacle_collision_rate': float(np.mean([c > 0 for c in obs_cols])),
            'pedestrian_collision_rate': float(np.mean([c > 0 for c in ped_cols])),
            'mean_obstacle_collisions': float(np.mean(obs_cols)),
            'mean_pedestrian_collisions': float(np.mean(ped_cols)),
        }
        self._completed.clear()
        return stats
