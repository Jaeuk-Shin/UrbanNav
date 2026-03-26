#!/usr/bin/env python3
"""
Long-range navigation in CARLA using hierarchical planning.

Global planner : CARLA road network A* (via GlobalRoutePlanner)
Local policy   : PPO-trained goal-conditioned visuomotor policy

Usage:
    # Start CARLA first:  ./CarlaUnreal.sh -RenderOffScreen
    python scripts/navigate_carla.py \
        --config config/rl.yaml \
        --checkpoint checkpoints/ppo_best.pt \
        --goal 50.0 80.0 \
        --port 2000
"""

import argparse
import time

import numpy as np
import torch

from config.utils import load_config
from rl.envs.carla import CarlaEnv
from rl.ppo_agent import PPOAgent
from rl.navigation import CarlaGlobalPlanner, HierarchicalNavigator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True,
                        help="Path to PPOAgent checkpoint (.pt)")
    parser.add_argument("--goal", nargs=2, type=float, required=True,
                        help="Goal in standard coords: x_std z_std")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--lookahead", type=float, default=6.0,
                        help="Lookahead distance for local goals (metres)")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    cfg = load_config(args.config)

    # ── Environment ───────────────────────────────────────────────────
    env = CarlaEnv(cfg, port=args.port)
    obs, info = env.reset()

    # ── Agent ─────────────────────────────────────────────────────────
    agent = PPOAgent(cfg, use_decoder=True).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()

    # ── Hierarchical Navigator ────────────────────────────────────────
    planner = CarlaGlobalPlanner(env.world, resolution=2.0)
    navigator = HierarchicalNavigator(
        agent=agent,
        global_planner=planner,
        lookahead_dist=args.lookahead,
        goal_reached_dist=2.0,
        final_goal_dist=1.5,
        device=device,
    )

    start_xz = env.xz
    goal_xz = np.array(args.goal)
    navigator.set_destination(start_xz, goal_xz)

    print(f"Start : ({start_xz[0]:.1f}, {start_xz[1]:.1f})")
    print(f"Goal  : ({goal_xz[0]:.1f}, {goal_xz[1]:.1f})")
    print(f"Straight-line distance: "
          f"{np.linalg.norm(goal_xz - start_xz):.1f} m")
    print()

    # ── Navigation Loop ──────────────────────────────────────────────
    for step_i in range(args.max_steps):
        action = navigator.step(obs, env.pose)
        obs, reward, terminated, truncated, info = env.step(action)

        if step_i % 10 == 0:
            lg = navigator.local_goal
            print(
                f"  step {step_i:4d} | "
                f"progress {navigator.progress:.0%} | "
                f"remaining ~{navigator.remaining_distance:.1f} m | "
                f"local goal ({lg[0]:.1f}, {lg[1]:.1f})"
            )

        if navigator.arrived:
            print(f"\n  Arrived at destination in {step_i + 1} steps!")
            break
    else:
        print(f"\n  Max steps ({args.max_steps}) reached. "
              f"Progress: {navigator.progress:.0%}")

    env.close()


if __name__ == "__main__":
    main()
