import os
import sys
import subprocess
import argparse
from tqdm import tqdm

# Configuration
INPUT_DIR = "dataset/citywalk/videos"               # Directory containing original videos
OUTPUT_DIR = "dataset/citywalk_2min"     # Directory to save split segments
SEGMENT_DURATION = 120                                      # Duration of each segment in seconds (10 minutes)

def ensure_directories():
    """Ensure that input and output directories exist."""
    if not os.path.isdir(INPUT_DIR):
        raise FileNotFoundError(f"Input directory '{INPUT_DIR}' does not exist.")
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory '{OUTPUT_DIR}'.")

def get_video_files():
    """Retrieve a list of video files from the input directory."""
    supported_extensions = ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv')
    return [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(supported_extensions)]

def split_segment(input_path, output_path, start_time, pbar):
    """
    Split a single segment using FFmpeg.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the output segment.
        start_time (float): Start time of the segment.
        pbar (tqdm): Progress bar to update.
    """
    # Check if the output file already exists
    if os.path.exists(output_path):
        print(f"Skipping {output_path} as it already exists.")
        pbar.update(1)
        return  # Skip to the next segment

    cmd_split = [
        'ffmpeg',
        '-y',                 # Overwrite without asking
        '-i', input_path,
        '-ss', str(start_time),
        '-t', str(SEGMENT_DURATION),
        '-c:v', 'libx264',    # Use libx264 codec
        '-an',                # Discard audio
        output_path
    ]

    try:
        subprocess.run(cmd_split, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error splitting {input_path} at {start_time} seconds: {e}")
    finally:
        pbar.update(1)


def main():
    ensure_directories()
    video_files = get_video_files()
    if not video_files:
        print(f"No video files found in '{INPUT_DIR}'.")
        return

    # Parse command-line arguments or environment variables
    parser = argparse.ArgumentParser(description="Split videos into segments for SLURM array job.")
    parser.add_argument('--task-id', type=int, default=None, help='SLURM_ARRAY_TASK_ID')
    parser.add_argument('--num-tasks', type=int, default=None, help='Total number of tasks')
    args = parser.parse_args()

    # Get task_id and num_tasks
    if args.task_id is not None and args.num_tasks is not None:
        task_id = args.task_id
        num_tasks = args.num_tasks
    else:
        # Try to get from environment variables
        task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
        num_tasks = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', '1'))

    # Adjust task_id to be zero-based
    task_min = int(os.environ.get('SLURM_ARRAY_TASK_MIN', '0'))
    task_id = task_id - task_min

    # Build a list of all segments
    all_segments = []
    for filename in video_files:
        input_path = os.path.join(INPUT_DIR, filename)

        # Get total duration of the video in seconds
        cmd_duration = [
            'ffprobe', '-v', 'error', '-show_entries',
            'format=duration', '-of',
            'default=noprint_wrappers=1:nokey=1', input_path
        ]
        try:
            result = subprocess.run(cmd_duration, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            total_duration = float(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            print(f"Error getting duration of {filename}: {e.stderr}")
            continue  # skip to next video

        # Calculate the number of segments
        num_segments = int(total_duration // SEGMENT_DURATION) + (1 if total_duration % SEGMENT_DURATION > 0 else 0)

        # Prepare segment information
        for segment_idx in range(num_segments):
            start_time = segment_idx * SEGMENT_DURATION
            base_name = os.path.splitext(filename)[0]
            segment_name = f"{base_name}_{segment_idx:04d}.mp4"
            output_path = os.path.join(OUTPUT_DIR, segment_name)
            all_segments.append((input_path, output_path, start_time))

    # Now, divide all_segments among the subjobs
    total_segments = len(all_segments)
    segments_per_task = total_segments // num_tasks
    remainder = total_segments % num_tasks

    # Compute the start and end indices for this task
    if task_id < remainder:
        start_idx = task_id * (segments_per_task + 1)
        end_idx = start_idx + segments_per_task + 1
    else:
        start_idx = task_id * segments_per_task + remainder
        end_idx = start_idx + segments_per_task

    # Get the segments for this task
    task_segments = all_segments[start_idx:end_idx]

    # Process the segments
    if not task_segments:
        print(f"No segments assigned to task {task_id}.")
        return

    print(f"Task {task_id}: Processing {len(task_segments)} segments out of {total_segments} total segments.")

    # Use a progress bar for the task
    with tqdm(total=len(task_segments), desc=f"Task {task_id}", ncols=100) as pbar:
        for segment_info in task_segments:
            input_path, output_path, start_time = segment_info
            split_segment(input_path, output_path, start_time, pbar)

    print(f"Task {task_id} completed.")

if __name__ == "__main__":
    main()
