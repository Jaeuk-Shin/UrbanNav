"""
Multi-GPU DPVO batch processor.

Launches one subprocess per GPU, each with CUDA_VISIBLE_DEVICES set in the
environment *before* Python starts, avoiding all CUDA-multiprocessing pitfalls.

Two modes:
  Coordinator (default) — discovers videos, partitions work, launches workers.
  Worker (--_worker)    — processes videos listed in a file on a single GPU.

Usage:
    python dpvo_slurm.py --videodir dataset/videos --calib calib/citywalk.txt \\
        --output_dir dataset/poses --stride 6 --save_trajectory --num_gpus 4
"""

import os
import sys
import json
import time
import argparse
import subprocess
import tempfile
from pathlib import Path


# ── Coordinator ───────────────────────────────────────────────────────

def coordinator(args):
    """Discover videos, partition across GPUs, launch worker subprocesses."""
    import torch
    num_gpus = args.num_gpus or torch.cuda.device_count()
    assert num_gpus > 0, "No GPUs detected"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Gather and sort video files
    exts = ('.mp4', '.avi', '.mov', '.mkv')
    all_videos = sorted(
        str(p) for p in Path(args.videodir).iterdir()
        if p.suffix.lower() in exts
    )

    # Filter out already-processed videos
    pending = [
        v for v in all_videos
        if not (Path(args.output_dir) / f"{Path(v).stem}.txt").exists()
    ]

    print(f"Videos: {len(all_videos)} total, {len(all_videos) - len(pending)} done, "
          f"{len(pending)} pending")

    if not pending:
        print("Nothing to do.")
        return

    # Round-robin partition across GPUs
    shards = [[] for _ in range(num_gpus)]
    for i, v in enumerate(pending):
        shards[i % num_gpus].append(v)

    print(f"Launching {num_gpus} GPU workers "
          f"({', '.join(str(len(s)) for s in shards)} videos each)...")

    # Launch one subprocess per GPU
    processes = []
    tmp_files = []
    for gpu_id in range(num_gpus):
        if not shards[gpu_id]:
            continue

        # Write this GPU's video list to a temp file
        tmp = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', prefix=f'dpvo_gpu{gpu_id}_',
            delete=False,
        )
        json.dump(shards[gpu_id], tmp)
        tmp.close()
        tmp_files.append(tmp.name)

        # Build worker command — same script, same args, plus worker flags
        cmd = [
            sys.executable, __file__, '--_worker',
            '--_gpu_id', str(gpu_id),
            '--_video_list', tmp.name,
            '--calib', args.calib,
            '--output_dir', args.output_dir,
            '--network', args.network,
            '--config', args.config,
            '--stride', str(args.stride),
            '--skip', str(args.skip),
        ]
        if args.timeit:
            cmd.append('--timeit')
        if args.save_trajectory:
            cmd.append('--save_trajectory')
        if args.save_ply:
            cmd.append('--save_ply')
        if args.save_colmap:
            cmd.append('--save_colmap')
        if args.plot:
            cmd.append('--plot')
        if args.opts:
            cmd += ['--opts'] + args.opts

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        p = subprocess.Popen(cmd, env=env)
        processes.append((gpu_id, p))
        print(f"  [GPU {gpu_id}] PID {p.pid}, {len(shards[gpu_id])} videos")

    # Wait for all workers
    failed = []
    for gpu_id, p in processes:
        p.wait()
        if p.returncode != 0:
            failed.append(gpu_id)

    # Clean up temp files
    for f in tmp_files:
        try:
            os.unlink(f)
        except OSError:
            pass

    if failed:
        print(f"WARNING: workers on GPU(s) {failed} exited with errors")
    else:
        print("--- All processing complete ---")


# ── Worker (runs in a subprocess with CUDA_VISIBLE_DEVICES already set) ──

def worker(args):
    """Process a list of videos on a single GPU."""
    import json
    import threading
    import queue as thread_queue

    import torch
    import numpy as np
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface

    from dpvo.config import cfg
    from dpvo.dpvo import DPVO
    from dpvo.plot_utils import plot_trajectory, save_output_for_COLMAP, save_ply
    from dpvo.stream import video_stream
    from dpvo.utils import Timer

    gpu_id = args._gpu_id

    # Load config
    cfg.merge_from_file(args.config)
    if args.opts:
        cfg.merge_from_list(args.opts)

    # Load video list
    with open(args._video_list) as f:
        videos = json.load(f)

    @torch.no_grad()
    def run_one(video_path):
        slam = None
        q = thread_queue.Queue(maxsize=16)
        reader = threading.Thread(
            target=video_stream,
            args=(q, video_path, args.calib, args.stride, args.skip),
        )
        reader.start()

        while True:
            item = q.get()
            if item is None:
                break
            (t, image, intrinsics) = item
            if t < 0:
                break

            image = torch.from_numpy(image).permute(2, 0, 1).cuda()
            intrinsics = torch.from_numpy(intrinsics).cuda()

            if slam is None:
                _, H, W = image.shape
                slam = DPVO(cfg, args.network, ht=H, wd=W, viz=False)

            with Timer("SLAM", enabled=args.timeit):
                slam(t, image, intrinsics)

        reader.join()

        points = slam.pg.points_.cpu().numpy()[:slam.m]
        colors = slam.pg.colors_.view(-1, 3).cpu().numpy()[:slam.m]
        return slam.terminate(), (points, colors, (*intrinsics, H, W))

    processed = 0
    t_start = time.time()

    for video_path in videos:
        video_name = Path(video_path).stem
        output_path = Path(args.output_dir) / f"{video_name}.txt"

        # Skip if already done (e.g. another run completed it)
        if output_path.exists():
            continue

        print(f"[GPU {gpu_id}] Processing: {video_name}")

        try:
            (poses, tstamps), (points, colors, calib_info) = run_one(video_path)

            trajectory = PoseTrajectory3D(
                positions_xyz=poses[:, :3],
                orientations_quat_wxyz=poses[:, [6, 3, 4, 5]],
                timestamps=tstamps,
            )

            if args.save_trajectory:
                file_interface.write_tum_trajectory_file(str(output_path), trajectory)

            if args.save_ply:
                save_ply(
                    os.path.join(args.output_dir, f"{video_name}.ply"),
                    points, colors,
                )

            if args.save_colmap:
                save_output_for_COLMAP(
                    os.path.join(args.output_dir, video_name),
                    trajectory, points, colors, *calib_info,
                )

            if args.plot:
                plot_dir = os.path.join(args.output_dir, "trajectory_plots")
                Path(plot_dir).mkdir(exist_ok=True)
                plot_trajectory(
                    trajectory,
                    title=f"DPVO Trajectory Prediction for {video_name}",
                    filename=os.path.join(plot_dir, f"{video_name}.pdf"),
                )

            processed += 1

        except Exception as e:
            print(f"[GPU {gpu_id}] FAILED on {video_name}: {e}")

    elapsed = time.time() - t_start
    print(f"[GPU {gpu_id}] Done — {processed} videos in {elapsed:.1f}s")


# ── Entry point ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU DPVO batch processor")
    parser.add_argument('--videodir', type=str, default='')
    parser.add_argument('--calib', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--stride', type=int, default=6)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--timeit', action='store_true')
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--save_ply', action='store_true')
    parser.add_argument('--save_colmap', action='store_true')
    parser.add_argument('--save_trajectory', action='store_true', default=True)
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='GPUs to use (default: all visible)')

    # Internal flags for worker subprocess — not for direct use
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--_gpu_id', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--_video_list', type=str, default='', help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args._worker:
        worker(args)
    else:
        coordinator(args)


if __name__ == '__main__':
    main()
