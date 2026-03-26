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

#### Building the Navmesh Cache

The geodesic distance field requires `walkable_tris_std` in the navmesh cache. To build (or rebuild) caches from existing navmesh OBJ exports:

```bash
python export_carla_navmesh.py --cache-all navmeshes/
```
