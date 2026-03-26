import os
from multiprocessing import Process, Queue
from pathlib import Path
import cv2
import numpy as np
import torch
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.plot_utils import plot_trajectory, save_output_for_COLMAP, save_ply
from dpvo.stream import image_stream, video_stream
from dpvo.utils import Timer

SKIP = 0

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

@torch.no_grad()
def run(cfg, network, video_path, calib, stride=1, skip=0, viz=False, timeit=False):
    slam = None
    queue = Queue(maxsize=16)
    
    reader = Process(target=video_stream, args=(queue, video_path, calib, stride, skip))
    reader.start()

    while True:
        item = queue.get()
        if item is None:
            break
        (t, image, intrinsics) = item

        if t < 0: 
            break

        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if slam is None:
            _, H, W = image.shape
            slam = DPVO(cfg, network, ht=H, wd=W, viz=viz)

        with Timer("SLAM", enabled=timeit):
            slam(t, image, intrinsics)

    reader.join()

    points = slam.pg.points_.cpu().numpy()[:slam.m]
    colors = slam.pg.colors_.view(-1, 3).cpu().numpy()[:slam.m]

    return slam.terminate(), (points, colors, (*intrinsics, H, W))

def process_videos(videos, cfg, network, calib, stride, skip, viz, timeit, output_dir, args):
    for video_file in videos:
        video_path = video_file
        video_name = Path(video_file).stem
        video_output_dir = os.path.join(output_dir)
        Path(video_output_dir).mkdir(parents=True, exist_ok=True)

        print(f"Processing {video_file}...")

        (poses, tstamps), (points, colors, calib_info) = run(cfg, network, video_path, calib, stride, skip, viz, timeit)
        trajectory = PoseTrajectory3D(positions_xyz=poses[:, :3], orientations_quat_wxyz=poses[:, [6, 3, 4, 5]], timestamps=tstamps)

        if args.save_ply:
            save_ply(os.path.join(video_output_dir, f"{video_name}.ply"), points, colors)

        if args.save_colmap:
            save_output_for_COLMAP(os.path.join(video_output_dir, video_name), trajectory, points, colors, *calib_info)

        if args.save_trajectory:
            # traj_dir = os.path.join(video_output_dir, "saved_trajectories")
            traj_dir = video_output_dir
            Path(traj_dir).mkdir(exist_ok=True)
            file_interface.write_tum_trajectory_file(os.path.join(traj_dir, f"{video_name}.txt"), trajectory)

        if args.plot:
            plot_dir = os.path.join(video_output_dir, "trajectory_plots")
            Path(plot_dir).mkdir(exist_ok=True)
            plot_trajectory(trajectory, title=f"DPVO Trajectory Prediction for {video_name}", filename=os.path.join(plot_dir, f"{video_name}.pdf"))

def partition_videos(video_files, total_jobs, job_index):
    """
    Partition the list of video files into subsets based on total_jobs and job_index.
    """
    num_videos = len(video_files)
    videos_per_job = num_videos // total_jobs
    remainder = num_videos % total_jobs

    start_idx = job_index * videos_per_job + min(job_index, remainder)
    end_idx = start_idx + videos_per_job + (1 if job_index < remainder else 0)

    return video_files[start_idx:end_idx]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="DPVO Parallel Video Processing Script")
    parser.add_argument('--network', type=str, default='dpvo.pth', help='Path to the network weights file')
    parser.add_argument('--videodir', type=str, required=True, help='Directory containing video files')
    parser.add_argument('--calib', type=str, required=True, help='Path to the calibration file')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output files')
    parser.add_argument('--stride', type=int, default=6, help='Stride for frame processing')
    parser.add_argument('--skip', type=int, default=0, help='Number of frames to skip')
    parser.add_argument('--config', type=str, default="config/default.yaml", help='Path to the config file')
    parser.add_argument('--timeit', action='store_true', help='Enable timing information')
    parser.add_argument('--viz', action="store_true", help='Enable visualization')
    parser.add_argument('--plot', action="store_true", help='Enable plotting of trajectories')
    parser.add_argument('--opts', nargs='+', default=[], help='Additional configuration options')
    parser.add_argument('--save_ply', action="store_true", help='Save point cloud as PLY')
    parser.add_argument('--save_colmap', action="store_true", help='Save output for COLMAP')
    parser.add_argument('--save_trajectory', action="store_true", help='Save trajectory file')
    
    # New arguments for array job partitioning
    parser.add_argument('--total_jobs', type=int, required=True, help='Total number of array jobs')
    parser.add_argument('--job_index', type=int, required=True, help='Index of this job (0-based)')
    
    args = parser.parse_args()

    # Load configuration
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    print("Running with config...")
    print(cfg)

    # List all video files
    supported_extensions = ('.mp4', '.avi', '.mov', '.mkv')  # Add more if needed
    all_videos = [str(Path(args.videodir) / f) for f in os.listdir(args.videodir) if f.lower().endswith(supported_extensions)]
    all_videos.sort()  # Optional: sort the list for consistency

    total_videos = len(all_videos)
    print(f"Total videos found: {total_videos}")

    # Partition videos based on total_jobs and job_index
    assigned_videos = partition_videos(all_videos, args.total_jobs, args.job_index)
    print(f"Job {args.job_index} processing {len(assigned_videos)} videos.")

    if not assigned_videos:
        print("No videos assigned to this job. Exiting.")
        exit(0)

    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Process assigned videos
    process_videos(assigned_videos, cfg, args.network, args.calib, args.stride, args.skip, args.viz, args.timeit, args.output_dir, args)
