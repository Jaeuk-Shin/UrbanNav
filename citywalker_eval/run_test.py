#!/usr/bin/env python3
"""Run a single closed-loop CityWalker test episode in CARLA.

Usage:
    # Start CARLA first:  ./CarlaUE4.sh -RenderOffScreen
    python -m citywalker_eval.run_test \
        --config citywalker_eval/config/default.yaml \
        --checkpoint /path/to/citywalker.ckpt \
        --scenario_dir /path/to/scenarios \
        --citywalker_root /home3/rvl/baselines/CityWalker \
        --goal_distance 100
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from omegaconf import OmegaConf

# Ensure UrbanNav repo root is importable when running as a script.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from citywalker_eval.env import NavTestEnv           # noqa: E402
from citywalker_eval.policy import CityWalkerPolicy  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="citywalker_eval/config/default.yaml")
    p.add_argument("--checkpoint", required=True,
                   help="Path to CityWalker .ckpt file")
    p.add_argument("--scenario_dir", default=None,
                   help="Override carla.scenario_dir from the config")
    p.add_argument("--citywalker_root",
                   default="/home3/rvl/baselines/CityWalker",
                   help="Path to the CityWalker repository")
    p.add_argument("--port", type=int, default=None,
                   help="Override carla.port")
    p.add_argument("--town", default=None,
                   help="Override carla.towns[0]")
    p.add_argument("--goal_distance", type=float, default=None,
                   help="Override test.goal_distance (metres)")
    p.add_argument("--goal", nargs=2, type=float, default=None,
                   metavar=("X_STD", "Z_STD"),
                   help="Explicit goal in std coords (skips random sampling)")
    p.add_argument("--subgoal_spacing", type=float, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _resolve(value, override):
    return value if override is None else override


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = OmegaConf.create(yaml.safe_load(f))

    # ── Apply CLI overrides ───────────────────────────────────────────
    cfg.carla.scenario_dir = _resolve(
        cfg.carla.scenario_dir, args.scenario_dir
    )
    assert cfg.carla.scenario_dir, (
        "carla.scenario_dir is required (set in the config or via "
        "--scenario_dir). CarlaMultiAgentEnv needs a precomputed scenario "
        "pool to spawn the walker."
    )
    if args.port is not None:
        cfg.carla.port = args.port
    if args.town is not None:
        cfg.carla.towns = [args.town]
    if args.goal_distance is not None:
        cfg.test.goal_distance = args.goal_distance
    if args.subgoal_spacing is not None:
        cfg.test.subgoal_spacing = args.subgoal_spacing
    if args.max_steps is not None:
        cfg.test.max_steps = args.max_steps
    if args.seed is not None:
        cfg.test.seed = args.seed

    # ── Environment ───────────────────────────────────────────────────
    env = NavTestEnv(
        cfg=cfg,
        scenario_dir=cfg.carla.scenario_dir,
        port=cfg.carla.port,
        fps=cfg.carla.fps,
        max_speed=cfg.carla.max_speed,
        teleport=cfg.carla.teleport,
        use_mpc=cfg.carla.use_mpc,
        towns=list(cfg.carla.towns),
        max_episode_steps=cfg.carla.max_episode_steps,
        reach_threshold=cfg.test.reach_threshold,
        final_threshold=cfg.test.final_threshold,
        rng_seed=cfg.test.seed,
    )

    obs, info = env.reset()
    start_xz = env.agent_xz()
    print(f"Agent spawned at (x_std, z_std) = "
          f"({start_xz[0]:.1f}, {start_xz[1]:.1f})")

    # ── Plan route & subgoals ─────────────────────────────────────────
    goal_xz = np.asarray(args.goal) if args.goal is not None else None
    route = env.plan_route(
        goal_distance=cfg.test.goal_distance,
        goal_tolerance=cfg.test.goal_tolerance,
        goal_xz=goal_xz,
        subgoal_spacing=cfg.test.subgoal_spacing,
        path_resolution=cfg.test.path_resolution,
    )
    print(f"Goal  : ({route.goal_xz[0]:.1f}, {route.goal_xz[1]:.1f})  "
          f"| straight-line {np.linalg.norm(route.goal_xz - start_xz):.1f} m  "
          f"| route length {route.total_length:.1f} m")
    print(f"Reference path: {len(route.reference_path)} waypoints; "
          f"sparsified to {len(route.subgoals)} subgoals "
          f"(~{cfg.test.subgoal_spacing:.1f} m spacing)")

    # ── Policy ────────────────────────────────────────────────────────
    policy = CityWalkerPolicy(
        checkpoint=args.checkpoint,
        cfg=cfg,
        citywalker_root=args.citywalker_root,
        step_scale=cfg.test.step_scale,
        device=args.device,
    )
    print(f"Policy loaded ({policy.module_cls_name}) "
          f"from {args.checkpoint}")

    # ── Rollout ───────────────────────────────────────────────────────
    try:
        for step_i in range(int(cfg.test.max_steps)):
            result = policy.act(
                rgb_history=env.rgb_history(),
                pose_history=env.pose_history(),
                subgoal_std=env.current_subgoal,
            )
            obs, info = env.step(result["action"])

            if step_i % 10 == 0 or env.arrived:
                print(
                    f"  step {step_i:4d} | "
                    f"subgoal {info['subgoal_idx']:3d}/{info['num_subgoals']:3d} | "
                    f"d(sg) {info['dist_to_subgoal']:5.1f} m | "
                    f"d(final) {info['dist_to_final']:5.1f} m | "
                    f"arrive_prob {result['arrive_prob']:.2f}"
                )

            if env.arrived:
                print(f"\n  Arrived in {step_i + 1} steps "
                      f"(final distance "
                      f"{info['dist_to_final']:.2f} m)")
                break
        else:
            print(f"\n  Max steps ({cfg.test.max_steps}) reached. "
                  f"Final dist to goal: {info['dist_to_final']:.1f} m")
    finally:
        env.close()


if __name__ == "__main__":
    main()
