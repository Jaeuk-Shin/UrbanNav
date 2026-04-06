from itertools import cycle
import multiprocessing as mp
import subprocess
import time
import os
import signal
import numpy as np
import random

from omegaconf import OmegaConf
from rl.envs.carla import CarlaEnv
from rl.envs.carla_basic import CarlaBasicEnv


# ─── CARLA Server Launcher ───────────────────────────────────────────


def launch_carla_servers(carla_bin, ports, gpu_ids, startup_wait=15,
                         stagger_delay=5):
    """
    Launch one CARLA server per (port, gpu) pair.

    Each server is started with -RenderOffScreen and -graphicsadapter=<gpu>
    to pin its Vulkan renderer to a specific GPU.

    CARLA uses two consecutive ports per server:
        RPC port  = port       (client ↔ server commands)
        Stream port = port + 1   (sensor data streaming)
    So ports must be spaced by at least 2 (the default base_port + 2*i).

    Don't know why, but spacing by 2 results in errors...use 4 instead

    Servers are launched with a stagger_delay (seconds) between each to avoid
    Vulkan init contention when multiple instances share a GPU.

    Returns a list of Popen handles for cleanup.
    """
    procs = []
    for port, gpu_id in zip(ports, gpu_ids):
        cmd = [
            carla_bin,
            "-RenderOffScreen",
            f"-carla-rpc-port={port}",
            f"-graphicsadapter={gpu_id}",
            "-nosound"
        ]
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group for clean teardown
        )
        procs.append(p)
        print(f"  CARLA  pid={p.pid}  port={port}  gpu={gpu_id}")
        '''
        if stagger_delay > 0 and i < len(ports) - 1:
            print(f"  Staggering next launch by {stagger_delay}s ...")
            time.sleep(stagger_delay)
        '''
    print(f"  Waiting {startup_wait}s for CARLA servers to initialise ...")
    time.sleep(startup_wait)
    return procs


def stop_carla_servers(server_procs):
    """Terminate CARLA servers launched by launch_carla_servers."""
    # SIGTERM first (give 3s)
    for p in server_procs:
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
    for p in server_procs:
        try:
            p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
    # SIGKILL anything still alive (CARLA/UE often ignores SIGTERM)
    for p in server_procs:
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            p.wait(timeout=5)


# ─── Vectorized CARLA Environment ────────────────────────────────────

def _env_worker(pipe, cfg_dict, port, max_speed, fps, max_episode_steps, gamma, simple_action,
                teleport=False, towns=None, randomize_weather=False):
    """Subprocess that owns one CarlaEnv and communicates via pipe."""
    # Ignore SIGINT — only the main process handles Ctrl+C.
    # Workers stay alive to receive the "close" command and clean up properly.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Each spawned subprocess inherits the same random state from the parent.
    # Re-seed per worker so environments diverge.
    worker_seed = port + os.getpid()
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    MAX_RETRIES = 5
    RETRY_DELAY = 5.0

    cfg = OmegaConf.create(cfg_dict)
    env = None
    step_count = 0

    def _create_env():
        nonlocal env
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if simple_action:
            env = CarlaBasicEnv(cfg, port=port, max_speed=max_speed, fps=fps, gamma=gamma,
                                teleport=teleport, towns=towns,
                                randomize_weather=randomize_weather)
        else:
            env = CarlaEnv(cfg, port=port, max_speed=max_speed, fps=fps, gamma=gamma,
                           towns=towns, randomize_weather=randomize_weather)

    def _safe_reset():
        nonlocal env, step_count
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                if env is None:
                    _create_env()
                obs, info = env.reset()
                step_count = 0
                return obs, info
            except Exception as e:
                last_err = e
                print(f"[worker port={port}] reset attempt {attempt+1}/{MAX_RETRIES} "
                      f"failed: {e}")
                # Close env before discarding to clean up CARLA resources
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
                env = None          # force re-creation on next attempt
                time.sleep(RETRY_DELAY)
        raise RuntimeError(
            f"[worker port={port}] reset failed after {MAX_RETRIES} retries: {last_err}"
        )

    try:
        _create_env()
    except Exception as e:
        print(f"[worker port={port}] initial env creation failed: {e}")
        pipe.close()
        return

    while True:
        try:
            cmd, data = pipe.recv()
        except (EOFError, OSError):
            break

        if cmd == "reset":
            try:
                obs, info = _safe_reset()
                pipe.send((obs, info))
            except Exception as e:
                print(f"[worker port={port}] unrecoverable reset error: {e}")
                break
        elif cmd == "step":
            try:
                obs, reward, terminated, truncated, info = env.step(data)
                step_count += 1
                if step_count >= max_episode_steps:
                    truncated = True
                done = terminated or truncated
                if done:
                    step_info = info  # preserve control diagnostics from step
                    # For truncated (not terminated) episodes, capture the
                    # terminal observation before reset so the trainer can
                    # bootstrap V(s_terminal) correctly.
                    if truncated and not terminated:
                        step_info['terminal_observation'] = obs
                    obs, info = _safe_reset()
                    # carry over control diagnostics so the trainer can log them
                    for key in ('cmd_vel_xz', 'cmd_speed', 'real_vel_xz',
                                'real_speed', 'real_ctrl_direction_ue', 'real_ctrl_speed',
                                'distance_to_goal', 'initial_distance',
                                'path_length', 'is_success',
                                'substep_frames', 'terminal_observation'):
                        if key in step_info:
                            info[key] = step_info[key]
                pipe.send((obs, reward, terminated, truncated, info))
            except Exception as e:
                print(f"[worker port={port}] step failed: {e}, attempting reset...")
                try:
                    # Close env before discarding to clean up CARLA resources
                    if env is not None:
                        try:
                            env.close()
                        except Exception:
                            pass
                    env = None      # force re-creation
                    obs, info = _safe_reset()
                    # treat the failed step as a truncation with zero reward
                    pipe.send((obs, 0.0, False, True, info))
                except Exception as e2:
                    print(f"[worker port={port}] unrecoverable step error: {e2}")
                    break
        elif cmd == "set_collect_substep_frames":
            env.set_collect_substep_frames(data)
            pipe.send(True)
        elif cmd == "capture_bev":
            altitude, fov, img_size = data
            try:
                img, meta = env.capture_bev(
                    altitude=altitude, fov=fov, img_size=img_size
                )
                pipe.send((img, meta))
            except Exception as e:
                print(f"[worker port={port}] capture_bev failed: {e}")
                pipe.send((None, None))
        elif cmd == "close":
            break

    # cleanup
    if env is not None:
        try:
            env.close()
        except Exception:
            pass
    try:
        pipe.close()
    except Exception:
        pass


def _stack_obs(obs_list):
    """Stack a list of dict-observations into a batched dict of arrays."""
    keys = obs_list[0].keys()
    return {k: np.stack([o[k] for o in obs_list]) for k in keys}


class VecCarlaEnv:
    """Vectorised CARLA environments backed by subprocesses."""

    def __init__(self, cfg, ports, max_speed=1.4, fps=5, max_episode_steps=200, gamma=0.99,
                 simple_action=True, teleport=False, towns=None, randomize_weather=False):
        from omegaconf import OmegaConf

        self.cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        self.ports = ports
        self.max_speed = max_speed
        self.fps = fps
        self.max_episode_steps = max_episode_steps
        self.num_envs = len(ports)
        self.simple_action = simple_action
        self.teleport = teleport
        self.gamma = gamma
        self.towns = towns
        self.randomize_weather = randomize_weather

        # Deterministic town-to-worker mapping (same pattern as VecCarlaMultiAgentEnv)
        if towns:
            self.worker_towns = [t for _, t in zip(range(self.num_envs), cycle(towns))]
            print(f"[VecCarlaEnv] Town mapping: {dict(enumerate(self.worker_towns))}")
        else:
            self.worker_towns = [None] * self.num_envs

        self.pipes = [None] * self.num_envs
        self.procs = [None] * self.num_envs
        for i in range(self.num_envs):
            self._start_worker(i)

    def _start_worker(self, idx):
        """Start a fresh worker subprocess for environment *idx*."""
        parent, child = mp.Pipe()
        worker_towns = [self.worker_towns[idx]] if self.worker_towns[idx] else None
        p = mp.Process(
            target=_env_worker,
            args=(child, self.cfg_dict, self.ports[idx], self.max_speed,
                  self.fps, self.max_episode_steps, self.gamma, self.simple_action,
                  self.teleport, worker_towns, self.randomize_weather),
            daemon=True,
        )
        p.start()
        child.close()
        self.pipes[idx] = parent
        self.procs[idx] = p

    def _restart_worker(self, idx):
        """Kill a crashed worker and start a replacement."""
        print(f"[VecCarlaEnv] Restarting worker {idx} (port={self.ports[idx]})...")
        old = self.procs[idx]
        if old is not None:
            if old.is_alive():
                old.kill()
            old.join(timeout=10)
        try:
            self.pipes[idx].close()
        except Exception:
            pass
        self._start_worker(idx)

    def _restart_and_reset(self, idx):
        """Restart a crashed worker, send reset, and return the result."""
        for attempt in range(3):
            try:
                self._restart_worker(idx)
                self.pipes[idx].send(("reset", None))
                return self.pipes[idx].recv()
            except (EOFError, BrokenPipeError, OSError) as e:
                print(f"[VecCarlaEnv] Worker {idx} restart attempt "
                      f"{attempt+1}/3 failed: {e}")
                time.sleep(5)
        raise RuntimeError(
            f"Worker {idx} (port={self.ports[idx]}) unrecoverable after 3 restarts"
        )

    # -- public API ----------------------------------------------------

    def reset(self):
        restart_needed = set()
        for i in range(self.num_envs):
            try:
                self.pipes[i].send(("reset", None))
            except (BrokenPipeError, OSError):
                restart_needed.add(i)

        results = []
        for i in range(self.num_envs):
            if i in restart_needed:
                results.append(self._restart_and_reset(i))
            else:
                try:
                    results.append(self.pipes[i].recv())
                except (EOFError, BrokenPipeError, OSError):
                    results.append(self._restart_and_reset(i))

        obs_list, info_list = zip(*results)
        return _stack_obs(obs_list), list(info_list)

    def step(self, actions):
        """actions: np array (num_envs, action_dim)"""
        restart_needed = set()
        for i, action in enumerate(actions):
            try:
                self.pipes[i].send(("step", action))
            except (BrokenPipeError, OSError):
                restart_needed.add(i)

        results = []
        for i in range(self.num_envs):
            if i in restart_needed:
                obs, info = self._restart_and_reset(i)
                results.append((obs, 0.0, False, True, info))
            else:
                try:
                    results.append(self.pipes[i].recv())
                except (EOFError, BrokenPipeError, OSError):
                    obs, info = self._restart_and_reset(i)
                    results.append((obs, 0.0, False, True, info))

        obs_list, rewards, terminateds, truncateds, infos = zip(*results)
        return (
            _stack_obs(obs_list),
            np.array(rewards, dtype=np.float32),
            np.array(terminateds),
            np.array(truncateds),
            list(infos),
        )

    def set_collect_substep_frames(self, enabled: bool):
        """Toggle substep RGB frame collection on all env workers."""
        for i in range(self.num_envs):
            try:
                self.pipes[i].send(("set_collect_substep_frames", enabled))
            except (BrokenPipeError, OSError):
                pass
        for i in range(self.num_envs):
            try:
                self.pipes[i].recv()
            except (EOFError, BrokenPipeError, OSError):
                pass

    def capture_bev(self, env_idx: int = 0,
                    altitude: float = 50.0,
                    fov: float = 90.0,
                    img_size: int = 512):
        """
        Ask env worker *env_idx* to capture a BEV overhead image.

        The worker spawns a temporary downward camera, ticks the CARLA world
        once, drains the extra sensor data, then destroys the camera.

        Returns (img, meta) — see CarlaEnv.capture_bev() for meta format.
        The single extra world-tick advances the simulation by one step;
        the obs_dict held by the trainer stays valid because the worker's
        pose/rgb deques are unchanged (only the raw sensor queues are
        flushed).  The first rollout step after a BEV capture will therefore
        use a one-tick-stale obs_dict, which is negligible in practice.
        """
        self.pipes[env_idx].send(("capture_bev", (altitude, fov, img_size)))
        return self.pipes[env_idx].recv()

    def close(self):
        # send close command to all workers
        for i in range(self.num_envs):
            try:
                self.pipes[i].send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        # wait for workers to finish env.close() (sensor/actor cleanup)
        for p in self.procs:
            if p is not None:
                p.join(timeout=15)
        # force-kill any that are still hanging (e.g. blocked on CARLA RPC)
        for p in self.procs:
            if p is not None and p.is_alive():
                p.kill()
                p.join(timeout=5)


# ─── Multi-Agent Vectorized Environment ─────────────────────────────


def _multi_env_worker(pipe, cfg_dict, port, num_agents, max_speed, fps,
                      max_episode_steps, gamma, teleport, goal_range,
                      towns=None, map_change_interval=0,
                      discrete=False, obstacle_config=None,
                      pedestrian_config=None,
                      navmesh_cache_dir=None,
                      quadrant_margin=None,
                      randomize_weather=False,
                      use_mpc=False,
                      dynamic_geo_mode='off',
                      dynamic_geo_horizon=5.0,
                      scenario_dir=None):
    """Subprocess that owns one CarlaMultiAgentEnv (N agents on 1 server)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    worker_seed = port + os.getpid()
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    if discrete:
        from rl.envs.carla_multi_discrete import CarlaMultiAgentDiscreteEnv as EnvClass
    else:
        from rl.envs.carla_multi import CarlaMultiAgentEnv as EnvClass

    MAX_RETRIES = 5
    RETRY_DELAY = 5.0

    cfg = OmegaConf.create(cfg_dict)
    env = None

    # Load navmesh cache (per-worker, from disk)
    navmesh_cache = None
    if navmesh_cache_dir is not None:
        from rl.utils.navmesh_cache import NavmeshCache
        navmesh_cache = NavmeshCache(navmesh_cache_dir)
        # Pre-load cache for assigned town(s)
        if towns:
            for t in towns:
                navmesh_cache.load(t)

    def _create_env():
        nonlocal env
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        kwargs = dict(
            cfg=cfg, port=port, num_agents=num_agents, max_speed=max_speed,
            fps=fps, gamma=gamma, teleport=teleport, goal_range=goal_range,
            max_episode_steps=max_episode_steps,
            towns=towns, map_change_interval=map_change_interval,
            obstacle_config=obstacle_config,
            pedestrian_config=pedestrian_config,
            navmesh_cache=navmesh_cache,
            quadrant_margin=quadrant_margin,
            randomize_weather=randomize_weather,
            scenario_dir=scenario_dir,
        )
        if not discrete:
            kwargs['use_mpc'] = use_mpc
            kwargs['dynamic_geo_mode'] = dynamic_geo_mode
            kwargs['dynamic_geo_horizon'] = dynamic_geo_horizon
        env = EnvClass(**kwargs)

    def _safe_reset():
        nonlocal env
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                if env is None:
                    _create_env()
                obs, infos = env.reset()
                return obs, infos
            except Exception as e:
                last_err = e
                print(f"[multi-worker port={port}] reset attempt "
                      f"{attempt+1}/{MAX_RETRIES} failed: {e}")
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
                env = None
                time.sleep(RETRY_DELAY)
        raise RuntimeError(
            f"[multi-worker port={port}] reset failed after "
            f"{MAX_RETRIES} retries: {last_err}")

    try:
        _create_env()
    except Exception as e:
        print(f"[multi-worker port={port}] initial env creation failed: {e}")
        pipe.close()
        return

    while True:
        try:
            cmd, data = pipe.recv()
        except (EOFError, OSError):
            break

        if cmd == "reset":
            try:
                obs, infos = _safe_reset()
                pipe.send((obs, infos))
            except Exception as e:
                print(f"[multi-worker port={port}] unrecoverable reset: {e}")
                break

        elif cmd == "step":
            try:
                # data = (num_agents, 2) actions array
                obs, rewards, terminateds, truncateds, infos = env.step(data)
                pipe.send((obs, rewards, terminateds, truncateds, infos))
            except Exception as e:
                import traceback
                print(f"[multi-worker port={port}] step failed: "
                      f"{type(e).__name__}: {e}, attempting reset...")
                traceback.print_exc()
                try:
                    if env is not None:
                        try:
                            env.close()
                        except Exception:
                            pass
                    env = None
                    obs, infos = _safe_reset()
                    n = num_agents
                    pipe.send((
                        obs,
                        np.zeros(n, dtype=np.float32),
                        np.zeros(n, dtype=bool),
                        np.ones(n, dtype=bool),   # treat as truncation
                        infos,
                    ))
                except Exception as e2:
                    print(f"[multi-worker port={port}] unrecoverable step: {e2}")
                    break

        elif cmd == "set_collect_substep_frames":
            env.set_collect_substep_frames(data)
            pipe.send(True)

        elif cmd == "capture_bev":
            agent_idx, altitude, fov, img_size = data
            try:
                img, meta = env.capture_bev(
                    env_idx=agent_idx, altitude=altitude,
                    fov=fov, img_size=img_size,
                )
                pipe.send((img, meta))
            except Exception as e:
                print(f"[multi-worker port={port}] capture_bev failed: {e}")
                pipe.send((None, None))

        elif cmd == "capture_bev_batch":
            specs, fov, img_size = data
            try:
                results = env.capture_bev_batch(
                    specs, fov=fov, img_size=img_size,
                )
                pipe.send(results)
            except Exception as e:
                print(f"[multi-worker port={port}] capture_bev_batch failed: {e}")
                pipe.send([(None, None)] * len(specs))

        elif cmd == "get_obstacle_layout":
            try:
                layout = env.get_obstacle_layout()
                pipe.send(layout)
            except Exception as e:
                print(f"[multi-worker port={port}] get_obstacle_layout failed: {e}")
                pipe.send({'obstacles': [], 'crosswalks': []})

        elif cmd == "get_solvability_stats":
            try:
                stats = env.get_solvability_stats(reset=bool(data))
                pipe.send(stats)
            except Exception as e:
                print(f"[multi-worker port={port}] get_solvability_stats failed: {e}")
                pipe.send(None)

        elif cmd == "spawn_cctv_cameras":
            specs, fov, img_size = data
            try:
                env.spawn_cctv_cameras(specs, fov=fov, img_size=img_size)
                pipe.send(True)
            except Exception as e:
                print(f"[multi-worker port={port}] spawn_cctv_cameras failed: {e}")
                pipe.send(False)

        elif cmd == "collect_cctv_frames":
            try:
                result = env.collect_cctv_frames()
                pipe.send(result)
            except Exception as e:
                print(f"[multi-worker port={port}] collect_cctv_frames failed: {e}")
                pipe.send({'frames': {}, 'specs': []})

        elif cmd == "destroy_cctv_cameras":
            try:
                env.destroy_cctv_cameras()
                pipe.send(True)
            except Exception as e:
                print(f"[multi-worker port={port}] destroy_cctv_cameras failed: {e}")
                pipe.send(False)

        elif cmd == "close":
            break

    if env is not None:
        try:
            env.close()
        except Exception:
            pass
    try:
        pipe.close()
    except Exception:
        pass


class VecCarlaMultiAgentEnv:
    """
    Vectorised environment: N CARLA servers x M agents each = N x M rollouts.

    Each server runs a CarlaMultiAgentEnv with M agents (quadrant split).
    The public interface is identical to VecCarlaEnv: step / reset / close /
    capture_bev / set_collect_substep_frames, with num_envs = N x M.
    """

    def __init__(self, cfg, ports, num_agents_per_server=4,
                 max_speed=1.4, fps=5, max_episode_steps=32, gamma=0.99,
                 teleport=False, goal_range=8.0,
                 towns=None, map_change_interval=0,
                 carla_bin=None, gpu_ids=None, server_procs=None,
                 carla_startup_wait=30,
                 discrete=False, obstacle_config=None,
                 pedestrian_config=None,
                 navmesh_cache_dir=None,
                 quadrant_margin=None,
                 randomize_weather=False,
                 use_mpc=False,
                 dynamic_geo_mode='off',
                 dynamic_geo_horizon=5.0,
                 scenario_dir=None):

        self.cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        self.ports = ports
        self.num_servers = len(ports)
        self.agents_per_server = num_agents_per_server
        self.num_envs = self.num_servers * num_agents_per_server
        self.max_speed = max_speed
        self.fps = fps
        self.max_episode_steps = max_episode_steps
        self.gamma = gamma
        self.teleport = teleport
        self.goal_range = goal_range
        self.towns = towns
        self.map_change_interval = map_change_interval
        self.discrete = discrete
        self.obstacle_config = obstacle_config
        self.pedestrian_config = pedestrian_config
        self.navmesh_cache_dir = navmesh_cache_dir
        self.quadrant_margin = quadrant_margin
        self.randomize_weather = randomize_weather
        self.use_mpc = use_mpc
        self.dynamic_geo_mode = dynamic_geo_mode
        self.dynamic_geo_horizon = dynamic_geo_horizon
        self.scenario_dir = scenario_dir

        # Server lifecycle management (for autonomous restart)
        self.carla_bin = carla_bin
        self.gpu_ids = gpu_ids or []
        self.server_procs = list(server_procs) if server_procs else [None] * self.num_servers
        self.carla_startup_wait = carla_startup_wait

        # Deterministic town-to-server mapping: each server loads exactly one
        # town and never reloads.  Uses itertools.cycle to handle
        # len(towns) != num_servers.
        if towns:
            self.server_towns = [t for _, t in zip(range(self.num_servers), cycle(towns))]
            print(f"[VecMultiAgent] Town mapping: {dict(enumerate(self.server_towns))}")
        else:
            self.server_towns = [None] * self.num_servers

        self.pipes = [None] * self.num_servers
        self.procs = [None] * self.num_servers
        for i in range(self.num_servers):
            self._start_worker(i)

    def _start_worker(self, idx):
        parent, child = mp.Pipe()
        # Each worker gets its deterministically assigned town (single-element
        # list) and map_change_interval=0 so no map reloads ever happen.
        worker_towns = [self.server_towns[idx]] if self.server_towns[idx] else None
        p = mp.Process(
            target=_multi_env_worker,
            args=(child, self.cfg_dict, self.ports[idx],
                  self.agents_per_server, self.max_speed, self.fps,
                  self.max_episode_steps, self.gamma, self.teleport,
                  self.goal_range, worker_towns, 0,
                  self.discrete, self.obstacle_config,
                  self.pedestrian_config,
                  self.navmesh_cache_dir,
                  self.quadrant_margin,
                  self.randomize_weather,
                  self.use_mpc,
                  self.dynamic_geo_mode,
                  self.dynamic_geo_horizon,
                  self.scenario_dir),
            daemon=True,
        )
        p.start()
        child.close()
        self.pipes[idx] = parent
        self.procs[idx] = p

    def _restart_worker(self, idx):
        print(f"[VecMultiAgent] Restarting worker {idx} "
              f"(port={self.ports[idx]})...")
        old = self.procs[idx]
        if old is not None:
            if old.is_alive():
                old.kill()
            old.join(timeout=10)
        try:
            self.pipes[idx].close()
        except Exception:
            pass
        self._start_worker(idx)

    def _restart_server(self, idx):
        """Kill and relaunch the CARLA server process for server *idx*."""
        port = self.ports[idx]
        print(f"[VecMultiAgent] Restarting CARLA server {idx} "
              f"(port={port})...")

        # Kill old server process
        old_proc = self.server_procs[idx]
        if old_proc is not None and old_proc.poll() is None:
            try:
                os.killpg(os.getpgid(old_proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                old_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            if old_proc.poll() is None:
                try:
                    os.killpg(os.getpgid(old_proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    old_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass

        # Relaunch with the same port and GPU
        gpu_id = self.gpu_ids[idx % len(self.gpu_ids)] if self.gpu_ids else 0
        cmd = [
            self.carla_bin,
            "-RenderOffScreen",
            f"-carla-rpc-port={port}",
            f"-graphicsadapter={gpu_id}",
            "-nosound",
        ]
        new_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.server_procs[idx] = new_proc
        print(f"  CARLA  pid={new_proc.pid}  port={port}  gpu={gpu_id}")
        print(f"  Waiting {self.carla_startup_wait}s for server to initialise ...")
        time.sleep(self.carla_startup_wait)

    def _restart_and_reset(self, idx):
        for attempt in range(3):
            try:
                self._restart_worker(idx)
                self.pipes[idx].send(("reset", None))
                return self.pipes[idx].recv()
            except (EOFError, BrokenPipeError, OSError) as e:
                print(f"[VecMultiAgent] Worker {idx} restart attempt "
                      f"{attempt+1}/3 failed: {e}")
                time.sleep(5)

        # All 3 worker restarts failed — try restarting the CARLA server
        if self.carla_bin:
            print(f"[VecMultiAgent] Worker {idx} unrecoverable after 3 "
                  f"restarts, restarting CARLA server...")
            self._restart_server(idx)
            try:
                self._restart_worker(idx)
                self.pipes[idx].send(("reset", None))
                return self.pipes[idx].recv()
            except (EOFError, BrokenPipeError, OSError) as e:
                raise RuntimeError(
                    f"Worker {idx} (port={self.ports[idx]}) unrecoverable "
                    f"even after CARLA server restart: {e}")

        raise RuntimeError(
            f"Worker {idx} (port={self.ports[idx]}) unrecoverable "
            f"after 3 restarts")

    # -- public API ----------------------------------------------------

    def reset(self):
        restart_needed = set()
        for i in range(self.num_servers):
            try:
                self.pipes[i].send(("reset", None))
            except (BrokenPipeError, OSError):
                restart_needed.add(i)

        results = []
        for i in range(self.num_servers):
            if i in restart_needed:
                results.append(self._restart_and_reset(i))
            else:
                try:
                    results.append(self.pipes[i].recv())
                except (EOFError, BrokenPipeError, OSError):
                    results.append(self._restart_and_reset(i))

        # Each result is (obs_dict{(M,...)}, info_list[M])
        # Concatenate across servers → (N*M, ...)
        all_obs = [r[0] for r in results]
        all_infos = []
        for r in results:
            all_infos.extend(r[1])

        keys = all_obs[0].keys()
        flat_obs = {k: np.concatenate([o[k] for o in all_obs], axis=0)
                    for k in keys}
        return flat_obs, all_infos

    def step(self, actions):
        """actions: (num_envs, action_dim) = (N*M, action_dim)"""
        M = self.agents_per_server

        # Split actions into per-server chunks
        restart_needed = set()
        for i in range(self.num_servers):
            chunk = actions[i * M : (i + 1) * M]
            try:
                self.pipes[i].send(("step", chunk))
            except (BrokenPipeError, OSError):
                restart_needed.add(i)

        results = []
        for i in range(self.num_servers):
            if i in restart_needed:
                obs, infos = self._restart_and_reset(i)
                results.append((
                    obs,
                    np.zeros(M, dtype=np.float32),
                    np.zeros(M, dtype=bool),
                    np.ones(M, dtype=bool),
                    infos,
                ))
            else:
                try:
                    results.append(self.pipes[i].recv())
                except (EOFError, BrokenPipeError, OSError):
                    obs, infos = self._restart_and_reset(i)
                    results.append((
                        obs,
                        np.zeros(M, dtype=np.float32),
                        np.zeros(M, dtype=bool),
                        np.ones(M, dtype=bool),
                        infos,
                    ))

        # Flatten across servers
        all_obs = [r[0] for r in results]
        keys = all_obs[0].keys()
        flat_obs = {k: np.concatenate([o[k] for o in all_obs], axis=0)
                    for k in keys}
        flat_rewards = np.concatenate([r[1] for r in results])
        flat_terminateds = np.concatenate([r[2] for r in results])
        flat_truncateds = np.concatenate([r[3] for r in results])
        flat_infos = []
        for r in results:
            flat_infos.extend(r[4])

        return flat_obs, flat_rewards, flat_terminateds, flat_truncateds, flat_infos

    def set_collect_substep_frames(self, enabled: bool):
        for i in range(self.num_servers):
            try:
                self.pipes[i].send(("set_collect_substep_frames", enabled))
            except (BrokenPipeError, OSError):
                pass
        for i in range(self.num_servers):
            try:
                self.pipes[i].recv()
            except (EOFError, BrokenPipeError, OSError):
                pass

    def capture_bev(self, env_idx: int = 0,
                    altitude: float = 50.0,
                    fov: float = 90.0,
                    img_size: int = 512):
        """Capture BEV for flat env_idx (mapped to server + agent)."""
        M = self.agents_per_server
        server_idx = env_idx // M
        agent_idx = env_idx % M
        self.pipes[server_idx].send((
            "capture_bev", (agent_idx, altitude, fov, img_size)))
        return self.pipes[server_idx].recv()

    def capture_bev_at_positions(self, specs, fov=90.0, img_size=512):
        """Capture BEV images at arbitrary standard-coord positions.

        Groups specs by server (via ``env_idx``) and issues one
        ``capture_bev_batch`` per server (single ``world.tick()`` each).

        Parameters
        ----------
        specs : list of dict, each with keys
            ``env_idx``   - flat env index (for routing to the correct server)
            ``center_xz`` - (2,) standard world coords
            ``altitude``  - float, metres
        fov, img_size : camera parameters

        Returns
        -------
        list of ``(img, meta)`` in the same order as *specs*.
        ``(None, None)`` for entries that failed.
        """
        M = self.agents_per_server

        # Group by server, preserving original index
        server_batches = {}  # server_idx -> [(orig_idx, spec_for_worker), ...]
        for orig_idx, spec in enumerate(specs):
            server_idx = spec['env_idx'] // M
            worker_spec = {
                'center_xz': spec['center_xz'],
                'altitude': spec['altitude'],
            }
            server_batches.setdefault(server_idx, []).append(
                (orig_idx, worker_spec))

        # Send one batch per server
        for server_idx, batch in server_batches.items():
            worker_specs = [item[1] for item in batch]
            try:
                self.pipes[server_idx].send((
                    "capture_bev_batch", (worker_specs, fov, img_size)))
            except (BrokenPipeError, OSError) as e:
                print(f"[VecMultiAgent] capture_bev_batch send failed "
                      f"for server {server_idx}: {e}")

        # Collect results and reassemble in original order
        results = [None] * len(specs)
        for server_idx, batch in server_batches.items():
            try:
                server_results = self.pipes[server_idx].recv()
                for (orig_idx, _), result in zip(batch, server_results):
                    results[orig_idx] = result
            except (EOFError, BrokenPipeError, OSError) as e:
                print(f"[VecMultiAgent] capture_bev_batch recv failed "
                      f"for server {server_idx}: {e}")

        return results

    def get_obstacle_layouts(self):
        """Get obstacle layout data from all servers for BEV visualisation.

        Returns a flat list of dicts (one per env_idx), each with keys
        ``'obstacles'`` and ``'crosswalks'``.  All agents on the same server
        share the same layout dict.
        """
        M = self.agents_per_server

        for i in range(self.num_servers):
            try:
                self.pipes[i].send(("get_obstacle_layout", None))
            except (BrokenPipeError, OSError):
                pass

        server_layouts = []
        for i in range(self.num_servers):
            try:
                server_layouts.append(self.pipes[i].recv())
            except (EOFError, BrokenPipeError, OSError):
                server_layouts.append({'obstacles': [], 'crosswalks': []})

        # Duplicate each server's layout for every agent on that server
        flat_layouts = []
        for server_idx in range(self.num_servers):
            for _ in range(M):
                flat_layouts.append(server_layouts[server_idx])
        return flat_layouts

    def get_solvability_stats(self, reset=False):
        """Aggregate solvability stats across all servers.

        Returns a dict with summed counts and recomputed rates.
        """
        for i in range(self.num_servers):
            try:
                self.pipes[i].send(("get_solvability_stats", reset))
            except (BrokenPipeError, OSError):
                pass

        agg = {
            'solvable_episodes': 0,
            'unsolvable_episodes': 0,
            'goal_retries_total': 0,
            'obstacle_spawn_requested': 0,
            'obstacle_spawn_failed': 0,
        }
        for i in range(self.num_servers):
            try:
                stats = self.pipes[i].recv()
                if stats is not None:
                    agg['solvable_episodes'] += stats['solvable_episodes']
                    agg['unsolvable_episodes'] += stats['unsolvable_episodes']
                    agg['goal_retries_total'] += stats['goal_retries_total']
                    agg['obstacle_spawn_requested'] += stats.get(
                        'obstacle_spawn_requested', 0)
                    agg['obstacle_spawn_failed'] += stats.get(
                        'obstacle_spawn_failed', 0)
            except (EOFError, BrokenPipeError, OSError):
                pass

        total = agg['solvable_episodes'] + agg['unsolvable_episodes']
        agg['unsolvable_rate'] = (
            agg['unsolvable_episodes'] / total if total > 0 else 0.0)
        agg['mean_goal_retries'] = (
            agg['goal_retries_total'] / total if total > 0 else 0.0)
        obs_req = agg['obstacle_spawn_requested']
        agg['obstacle_spawn_fail_rate'] = (
            agg['obstacle_spawn_failed'] / obs_req if obs_req > 0 else 0.0)
        return agg

    # ── CCTV cameras ────────────────────────────────────────────────────

    def spawn_cctv_cameras(self, all_specs, fov=90.0, img_size=512):
        """Spawn persistent CCTV cameras on each server.

        Parameters
        ----------
        all_specs : list of dict
            Each dict has ``agent_idx`` (flat env index) and camera placement
            fields.  Specs are grouped by server automatically.
        fov, img_size : camera parameters
        """
        M = self.agents_per_server
        server_specs = {}
        for spec in all_specs:
            server_idx = spec['agent_idx'] // M
            worker_spec = {**spec, 'agent_idx': spec['agent_idx'] % M}
            server_specs.setdefault(server_idx, []).append(worker_spec)

        for server_idx in range(self.num_servers):
            specs = server_specs.get(server_idx, [])
            try:
                self.pipes[server_idx].send((
                    "spawn_cctv_cameras", (specs, fov, img_size)))
            except (BrokenPipeError, OSError):
                pass

        for server_idx in range(self.num_servers):
            if server_idx in server_specs or True:
                try:
                    self.pipes[server_idx].recv()
                except (EOFError, BrokenPipeError, OSError):
                    pass

    def collect_cctv_frames(self):
        """Collect accumulated CCTV frames from all servers.

        Returns dict mapping flat env_idx -> (T, H, W, 3) uint8 or None,
        plus list of spec dicts with flat env indices.
        """
        M = self.agents_per_server

        for i in range(self.num_servers):
            try:
                self.pipes[i].send(("collect_cctv_frames", None))
            except (BrokenPipeError, OSError):
                pass

        flat_frames = {}
        all_specs = []
        for server_idx in range(self.num_servers):
            try:
                result = self.pipes[server_idx].recv()
                for local_aidx, arr in result['frames'].items():
                    flat_idx = server_idx * M + local_aidx
                    flat_frames[flat_idx] = arr
                for spec in result['specs']:
                    flat_spec = {**spec, 'agent_idx': server_idx * M + spec['agent_idx']}
                    all_specs.append(flat_spec)
            except (EOFError, BrokenPipeError, OSError):
                pass

        return {'frames': flat_frames, 'specs': all_specs}

    def destroy_cctv_cameras(self):
        """Destroy CCTV cameras on all servers."""
        for i in range(self.num_servers):
            try:
                self.pipes[i].send(("destroy_cctv_cameras", None))
            except (BrokenPipeError, OSError):
                pass
        for i in range(self.num_servers):
            try:
                self.pipes[i].recv()
            except (EOFError, BrokenPipeError, OSError):
                pass

    def close(self):
        for i in range(self.num_servers):
            try:
                self.pipes[i].send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            if p is not None:
                p.join(timeout=15)
        for p in self.procs:
            if p is not None and p.is_alive():
                p.kill()
                p.join(timeout=5)
