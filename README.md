# End-to-End Urban Navigator

built upon the official implementation of [CityWalker](https://github.com/ai4ce/CityWalker)

# Getting Started
## Installation
Successfully tested with Python 3.12, PyTorch 2.9.1, and CUDA 12.8. To install the dependencies, run:
```
conda env create -f environment.yml
conda activate urban-nav
```
Also need to install [diffusion_policy](https://github.com/real-stanford/diffusion_policy).



### Blackwell Compatibility Issue with xformers
[xformers](https://github.com/facebookresearch/xformers) seems to have a compatible issue with Blackwell GPUs; see [link](https://github.com/facebookresearch/xformers/issues/1342) for a temporary solution.
In light of this, our codebase also uses `bf16-mixed` instead of `float32` for both training and closed-loop evaluation.

## Data Preparation
Please see [dataset/README.md](./dataset/README.md) for details on how to prepare the dataset.

## Training
To train the model, run:
```
python train.py --config config/goal_agnostic.yaml
```

## Testing
This repository offers a closed-loop evaluation interface in Carla. This can be used to assess a combination of a learned trajectory sampler and a trajectory selector. To run a simulation loop, first run a Carla simulator, e.g.:
```
./CarlaUnreal.sh
```
 and run:
```
python test_carla.py --config config/goal_agnostic.yaml --checkpoint <path_to_checkpoint>
```

### Closed-Loop Evaluation: Example
![](./assets/closed_loop.mp4)

## Training a PPO Agent
Run
```
python -m rl.ppo_trainer --num_envs 4 --gpu_ids 2,3,4,5 --carla_bin /raid/robot/Carla/CarlaUE4.sh --norm_reward --anneal_lr --clip_vloss --target_kl 0.015 --fps 5 --vis_every 1 --norm_obs --goal_mode both --obstacles --pedestrians --num_pedestrians_per_region 30 --use_decoder --navmesh_cache_dir navmeshes/
```

<details>
<summary><strong>Full argument reference (click to expand)</strong></summary>

#### Configuration & Checkpointing

| Argument | Type | Default | Description |
|---|---|---|---|
| `--config` | str | `config/rl.yaml` | Path to YAML configuration file |
| `--save_dir` | str | `checkpoints/ppo` | Directory to save checkpoints |
| `--save_every` | int | `100` | Save checkpoint every N iterations |

#### CARLA Server

| Argument | Type | Default | Description |
|---|---|---|---|
| `--carla_bin` | str | `None` | Path to `CarlaUnreal.sh`. If set, servers are auto-launched |
| `--gpu_ids` | str | `None` | Comma-separated GPU IDs for CARLA servers (e.g. `0,1,2,3`). Defaults to `0..num_envs-1` |
| `--base_port` | int | `2000` | Base port for CARLA server connections |
| `--carla_startup_wait` | int | `30` | Seconds to wait after launching CARLA servers |
| `--carla_stagger_delay` | int | `5` | Delay (seconds) between launching each CARLA server to avoid Vulkan init contention |
| `--towns` | str (list) | `Town02 Town03 Town05 Town10HD` | CARLA town names to randomly cycle through |
| `--map_change_interval` | int | `0` | Change map every N completed episodes per server (0 = only on full reset) |

#### Environment

| Argument | Type | Default | Description |
|---|---|---|---|
| `--num_envs` | int | `1` | Number of parallel CARLA server instances |
| `--num_agents_per_server` | int | `4` | Agents per server (quadrant split). Total rollout envs = `num_envs` &times; `num_agents_per_server` |
| `--max_speed` | float | `1.4` | Maximum agent speed (m/s) |
| `--fps` | int | `5` | Simulation frames per second |
| `--max_episode_steps` | int | `128` | Maximum steps per episode |
| `--teleport` | flag | `False` | Use `set_transform` control instead of `WalkerControl` (eliminates physics lag) |
| `--goal_range` | float | `40.0` | Goal sampling range (metres) |
| `--quadrant_margin` | float | `None` | Inset margin (metres) for spawn/goal sampling within each quadrant. Defaults to `goal_range` |

#### Obstacles, Pedestrians & Weather

| Argument | Type | Default | Description |
|---|---|---|---|
| `--obstacles` | flag | `False` | Enable procedural obstacle generation (blocked crosswalks, sidewalk obstructions, narrow passages) |
| `--p_crosswalk_challenge` | float | `0.3` | Per-episode probability of the blocked-crosswalk (anti-greedy) scenario |
| `--pedestrians` | flag | `False` | Enable SFM-controlled pedestrians |
| `--num_pedestrians_per_region` | int | `30` | Number of SFM pedestrians per quadrant |
| `--dynamic_geo_mode` | str | `off` | Dynamic geodesic reward mode: `off` (static), `soft` (heuristic swept-volume cost), `timespace` (exact time-space backward DP). Requires `--pedestrians`. See [rl/docs/dynamic_geodesic.md](rl/docs/dynamic_geodesic.md) |
| `--dynamic_geo_horizon` | float | `5.0` | Prediction horizon (seconds) for dynamic geodesic reward |
| `--weather` | flag | `False` | Randomize weather and sun position each episode |
| `--navmesh_cache_dir` | str | `None` | Directory containing precomputed navmesh cache NPZ files. Speeds up crosswalk detection, walkable-area sampling, and enables geodesic reward |

#### PPO Hyperparameters

| Argument | Type | Default | Description |
|---|---|---|---|
| `--num_steps` | int | `128` | Rollout length per environment |
| `--num_iterations` | int | `10000` | Total training iterations |
| `--num_epochs` | int | `4` | PPO epochs per iteration |
| `--num_minibatches` | int | `8` | Minibatches per PPO update |
| `--lr` | float | `2.5e-4` | Learning rate |
| `--anneal_lr` | flag | `False` | Linearly anneal learning rate to 0 over training |
| `--gamma` | float | `0.99` | Discount factor |
| `--gae_lambda` | float | `0.95` | GAE lambda |
| `--clip_coef` | float | `0.2` | PPO clipping coefficient |
| `--vf_coef` | float | `0.5` | Value function loss coefficient |
| `--ent_coef` | float | `1e-4` | Entropy bonus coefficient |
| `--max_grad_norm` | float | `0.2` | Maximum gradient norm for clipping |
| `--clip_vloss` | flag | `True` | Clip value function loss (PPO-style) |
| `--norm_reward` | flag | `True` | Normalize rewards with running statistics |
| `--norm-adv` | bool | `True` | Normalize advantages (`--norm-adv` / `--no-norm-adv`) |
| `--target_kl` | float | `None` | KL-based early stopping threshold for PPO epochs (typical: 0.01–0.02, `None` = disabled) |
| `--reward_clip` | float | `0` | Clip rewards to `[-val, val]` (0 = disabled) |

#### Model Architecture

| Argument | Type | Default | Description |
|---|---|---|---|
| `--agent` | str | `ppo` | Agent type: `ppo` (full DINOv2+LSTM) or `goal_only` (MLP baseline for debugging) |
| `--encoder_type` | str | `full` | Observation encoder: `full` (DINOv2 + history + coordinates) or `simple` (DINOv2 on current frame only) |
| `--lstm_hidden` | int | `512` | LSTM hidden dimension |
| `--lstm_layers` | int | `2` | Number of LSTM layers |
| `--goal_only_hidden` | int | `64` | Hidden dim for `goal_only` MLP agent |
| `--goal_only_layers` | int | `2` | Hidden layers for `goal_only` MLP agent |
| `--n_action_history` | int | `0` | Number of past actions to feed as input (0 = disabled) |
| `--goal_mode` | str | `both` | Where to inject goal: `lstm` (LSTM input only), `heads` (actor/critic heads only), or `both` |
| `--use_decoder` | flag | `False` | Use pretrained flow-matching decoder to produce actions |
| `--norm_obs` | flag | `False` | Apply LayerNorm to LSTM input features (stabilizes training with frozen DINOv2 features) |

#### Logging & Visualization

| Argument | Type | Default | Description |
|---|---|---|---|
| `--no_wandb` | flag | `False` | Disable W&B logging even if enabled in config |
| `--log_every` | int | `1` | Log metrics to W&B every N iterations (0 = disable) |
| `--vis_every` | int | `100` | Render BEV trajectory plot every N iterations (0 = disabled). Saved to `<save_dir>/vis/` and uploaded to W&B |
| `--bev_altitude` | float | `15.0` | Minimum BEV camera altitude (m); auto-adjusted to encompass start, goal, and geodesic path |
| `--bev_fov` | float | `90.0` | BEV camera horizontal field-of-view (degrees) |
| `--bev_img_size` | int | `512` | Side length (pixels) of BEV images |
| `--vis_video_fps` | int | `1` | Playback FPS for ego-view video |

</details>

## Test Rollout (No Training)

Run a diagnostic rollout to verify environment setup and visualize agent behaviour without any training:

```bash
# Random actions (default)
python -m rl.test_rollout --config config/rl.yaml --carla_bin /path/to/CarlaUnreal.sh --num_envs 1

# Zero actions (agent stays still — useful for verifying resets and observation capture)
python -m rl.test_rollout --config config/rl.yaml --carla_bin /path/to/CarlaUnreal.sh --num_envs 1 --action_mode zero

# With obstacles and pedestrians
python -m rl.test_rollout --config config/rl.yaml --carla_bin /path/to/CarlaUnreal.sh --num_envs 1 --obstacles --pedestrians
```

<details>
<summary><strong>Full argument reference (click to expand)</strong></summary>

#### Configuration & Output

| Argument | Type | Default | Description |
|---|---|---|---|
| `--config` | str | `config/rl.yaml` | Path to YAML configuration file |
| `--save_dir` | str | `checkpoints/test_rollout` | Directory for saved visualizations |

#### CARLA Server

| Argument | Type | Default | Description |
|---|---|---|---|
| `--carla_bin` | str | `None` | Path to `CarlaUnreal.sh`. If set, servers are auto-launched |
| `--gpu_ids` | str | `None` | Comma-separated GPU IDs for CARLA servers |
| `--base_port` | int | `2000` | Base port for CARLA server connections |
| `--carla_startup_wait` | int | `30` | Seconds to wait after launching CARLA servers |
| `--carla_stagger_delay` | int | `5` | Delay between launching each CARLA server |
| `--towns` | str (list) | `Town02 Town03 Town05 Town10HD` | CARLA town names to sample from |

#### Environment

| Argument | Type | Default | Description |
|---|---|---|---|
| `--num_envs` | int | `1` | Number of parallel CARLA server instances |
| `--num_agents_per_server` | int | `4` | Agents per server (quadrant split) |
| `--max_speed` | float | `1.4` | Maximum agent speed (m/s) |
| `--fps` | int | `5` | Simulation frames per second |
| `--max_episode_steps` | int | `60` | Maximum steps per episode (~12 s at 5 FPS) |
| `--teleport` | flag | `False` | Use `set_transform` control instead of `WalkerControl` |
| `--goal_range` | float | `40.0` | Goal sampling range (metres) |
| `--quadrant_margin` | float | `None` | Inset margin for spawn/goal sampling within each quadrant |

#### Obstacles, Pedestrians & Weather

| Argument | Type | Default | Description |
|---|---|---|---|
| `--obstacles` | flag | `False` | Enable procedural obstacle generation |
| `--p_crosswalk_challenge` | float | `0.3` | Per-episode probability of blocked-crosswalk scenario |
| `--pedestrians` | flag | `False` | Enable SFM-controlled pedestrians |
| `--num_pedestrians_per_region` | int | `30` | Number of SFM pedestrians per quadrant |
| `--weather` | flag | `False` | Randomize weather and sun position each episode |
| `--navmesh_cache_dir` | str | `None` | Directory containing precomputed navmesh cache NPZ files |

#### Rollout Control

| Argument | Type | Default | Description |
|---|---|---|---|
| `--num_steps` | int | `50` | Number of environment steps to run |
| `--action_mode` | str | `random` | Action generation: `random` (uniform in `[-max_speed, max_speed]`) or `zero` (stay in place) |

#### Visualization

| Argument | Type | Default | Description |
|---|---|---|---|
| `--bev_altitude` | float | `15.0` | Minimum BEV camera altitude (m) |
| `--bev_fov` | float | `90.0` | BEV camera horizontal field-of-view (degrees) |
| `--bev_img_size` | int | `512` | Side length (pixels) of BEV images |
| `--vis_video_fps` | int | `1` | Playback FPS for visualization video |

</details>

### Reward Function

The PPO reward follows the **DD-PPO** (Wijmans et al., 2019) design, using the change in geodesic distance to the goal as the primary shaping signal:

$$r_t = r_{\text{success}} + \bigl(d_{\text{geo}}(s_t, g) - d_{\text{geo}}(s_{t+1}, g)\bigr) - c_{\text{slack}} + r_{\text{collision}}$$

| Component | Value | Description |
|---|---|---|
| $r_{\text{success}}$ | $+2.5$ | Awarded when the Euclidean distance to goal $\leq 0.2$ m |
| $d_{\text{geo}}(s, g)$ | — | Obstacle-aware geodesic distance from state $s$ to goal $g$ (metres) |
| $c_{\text{slack}}$ | $0.01$ | Per-step penalty encouraging efficiency |
| $r_{\text{collision}}$ | $-0.5$ | Penalty per obstacle or pedestrian collision (0.5 m and 0.6 m radii, respectively) |

The agent receives positive reward when it decreases its geodesic distance to the goal, negative reward when it moves farther away, and an additional penalty for collisions.

#### Geodesic Distance Computation

Indoor navigation benchmarks such as [Habitat](https://aihabitat.org/) (Savva et al., 2019) compute geodesic distance via Recast/Detour's built-in A\* on the navigation polygon graph (`habitat_sim.PathFinder.geodesic_distance`). This works because Habitat environments are fully static — the navmesh perfectly represents the walkable area.

In our CARLA-based setting, the navmesh is static but we spawn **dynamic obstacles** at runtime (e.g., a firetruck blocking a crosswalk, barriers forming a narrow passage) that are not reflected in the navmesh. A naïve A\* on the unmodified navmesh would compute shortest paths *through* blocked areas, producing incorrect distance estimates and misleading reward signals.

To handle this, we use a **rasterized grid + Dijkstra** approach:

1. **One-time per map** — Rasterize all walkable navmesh triangles (sidewalk, crosswalk, road, grass) into a 2D occupancy grid at 1.0 m resolution. For a typical CARLA town this produces a ~1000×1000 grid. The rasterized grid is built from `walkable_tris_std` stored in the precomputed navmesh cache (`navmeshes/{Town}_navmesh_cache.npz`).

2. **Once per episode** — When a new goal is sampled (at reset or auto-reset):
   - Copy the base walkable grid.
   - **Stamp** each spawned obstacle's oriented bounding box (OBB) as blocked cells.
   - Run **single-source Dijkstra** from the goal cell on an 8-connected sparse graph (edge weights: $\Delta x$ for cardinal, $\sqrt{2}\,\Delta x$ for diagonal neighbours) using `scipy.sparse.csgraph.dijkstra`.
   - The result is a full distance field: geodesic distance from every cell to the goal, in metres. Cells behind obstacles have longer distances reflecting the required detour.

3. **Per step** — Look up the agent's current grid cell in the precomputed distance field. This is an O(1) array access.

When no navmesh cache is available, or the cache lacks walkable triangle data, the system falls back to Euclidean distance.

The implementation lives in `rl/utils/geodesic.py` (`GeodesicDistanceField` class), with integration points in `rl/envs/carla_multi.py` (`_geodesic_potential`, `_update_geodesic`).

#### Dynamic Geodesic Distance (`--dynamic_geo_mode`)

The static geodesic field treats pedestrians as point-collision penalties ($-0.5$ on contact) but otherwise ignores them when computing the shortest path. With `--dynamic_geo_mode`, the distance field is **recomputed every step** to account for predicted pedestrian motion, giving the agent a denser shaping signal to route *around* pedestrians rather than only reacting on collision. Because pedestrians in our SFM do **not** react to the ego agent, their forward trajectories are deterministic given the current state and can be forward-simulated in pure NumPy.

Two modes are available. See [rl/docs/dynamic_geodesic.md](rl/docs/dynamic_geodesic.md) for a full treatment including problem formulation, alternative methods (HJ reachability, MPC), feasibility analysis, and references.

##### `timespace` — Exact Time-Space Backward DP (recommended)

Expands the 2-D grid into a 3-D graph $(r, c, t)$ and computes the optimal value function $V(x, t)$ via backward dynamic programming:

$$V(x, T) = d_{\text{static}}(x, g)$$

$$V(x, t) = \min_{u \in \mathcal{N}(x) \cup \{x\}} \bigl[ w(x \to u) + V(u, t{+}1) \bigr], \quad x \notin \mathcal{O}(t)$$

where $\mathcal{N}(x)$ are the 8-connected spatial neighbours, $\mathcal{O}(t)$ is the set of cells occupied by predicted pedestrians at time $t$, and $d_{\text{static}}$ is the pre-computed static geodesic (terminal cost connecting to the static field beyond the horizon). Since time provides a topological ordering, no priority queue is needed — a plain backward sweep gives the exact optimum.

The reward queries the 3-D field at two time indices:

$$r_t = r_{\text{success}} + \bigl( V(s_t, 0) - V(s_{t+1}, \Delta t) \bigr) - c_{\text{slack}} + r_{\text{collision}}$$

where $\Delta t = $ `n_skips` (the number of simulation sub-steps per environment step).

**Key advantage:** can represent the **wait** strategy — if a pedestrian is crossing a narrow passage, the optimal plan is to wait and then proceed through the cleared passage, rather than taking a long detour.

**Complexity:** $O(H \times W \times T \times 9)$.  With a 100 $\times$ 100 quadrant grid and $T = 25$, this is ~2.25 M operations (~10–50 ms per agent).

```bash
python -m rl.ppo_trainer ... --pedestrians --dynamic_geo_mode timespace --dynamic_geo_horizon 5.0
```

##### `soft` — Heuristic Soft-Cost Approximation

Collapses the time dimension into a single 2-D cost map by summing temporally-discounted exponential penalties over all predicted pedestrian positions:

$$c(r,c) = \sum_{t=0}^{H-1}\sum_{i=1}^{N_{\text{ped}}} \alpha \, e^{-\lVert(r,c) - \hat{p}_i^{(t)}\rVert / \beta} \, \lambda^t$$

($\alpha = 5.0$, $\beta = 0.5$ m, $\lambda = 0.85$/step). Dijkstra runs on the 2-D grid with edge weights $w_{\text{base}} + c(\text{dst})$. Faster (~5–10 ms) but **cannot represent waiting** — it over-penalises areas that are only briefly occupied.

```bash
python -m rl.ppo_trainer ... --pedestrians --dynamic_geo_mode soft --dynamic_geo_horizon 5.0
```

#### Building the Navmesh Cache

The geodesic distance field requires `walkable_tris_std` in the navmesh cache. To build (or rebuild) caches from existing navmesh OBJ exports:

```bash
python export_carla_navmesh.py --cache-all navmeshes/
```
