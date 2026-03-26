import os
import argparse
import numpy as np
from omegaconf import OmegaConf
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from rl.ppo import DinoLSTMPPO
from rl.envs.carla import CarlaEnv

from rl.envs.mpc.mpc import MPC

from ray import tune
from ray.tune.registry import register_env

from config.utils import load_and_merge_configs


def env_creator(env_config):
    return CarlaEnv(
        cfg=env_config.get("cfg"),
        port=env_config.get("port", 3000),
        max_speed=env_config.get("max_speed", 1.4),
        fps=env_config.get("fps", 5),
        mpc=env_config.get("mpc")
    )

register_env("carla_rllib", env_creator)


parser = argparse.ArgumentParser(description='Test UrbanNav model')
parser.add_argument('--rl_config', type=str, default='config/rl_config/ppo.yaml', help='Path to config file')
parser.add_argument('--base_config', type=str, default='config/rl.yaml', help='Path to config file')
parser.add_argument('--port', type=int, default=2000, help='Port of the Carla server')
args = parser.parse_args()


# --- Execution ---
merged_dict = load_and_merge_configs(args.base_config, args.rl_config)
model_config = merged_dict.pop('model_config', {})


cfg = OmegaConf.create(model_config['base_config'])

fps = 5
dt = 1. / fps
future_length = cfg.model.decoder.len_traj_pred
n_skips = int(fps // cfg.data.target_fps)
mpc_horizon = future_length * n_skips
max_speed = 1.4

max_wall_time = .3 * dt
ulb = np.array([-max_speed, -0.8])
uub = np.array([max_speed, 0.8])

# This must be outside of the environment class
# JIT takes too long to rebuild the problem every time
mpc = MPC(mpc_horizon, dt, ulb, uub, max_wall_time=max_wall_time)


env_config = {
        "cfg": cfg,
        "max_speed": max_speed,
        "fps": fps,
        "port": args.port,
        "mpc": mpc
    }

env: CarlaEnv = env_creator(env_config)

config = (
    PPOConfig()
    .environment(
        env="carla_rllib",
        env_config=env_config
    )
    .update_from_dict(merged_dict)
    .rl_module(
        rl_module_spec=RLModuleSpec(
            module_class=DinoLSTMPPO,
            observation_space=env.observation_space,
            action_space=env.action_space,
            inference_only=False,
            learner_only=False,
            model_config=model_config,
            catalog_class=None
        )
    )
)

algo = config.build_algo()

tuner = tune.Tuner(
    "PPO",
    run_config=tune.RunConfig(
        stop={"training_iteration": 100}, # Stop after 10 iterations
    ),
    param_space=config.to_dict(), # Pass the config as a dictionary
)

results = tuner.fit()
