import os
import subprocess
import argparse
from tqdm import tqdm

# Configuration Defaults
DEFAULT_INPUT_DIR = "dataset/citywalk/videos"      # Directory containing original videos
DEFAULT_OUTPUT_DIR = "dataset/citywalk_2min"      # Directory to save split segments
DEFAULT_SEGMENT_DURATION = 120                     # Duration of each segment in seconds (2 minutes)

def ensure_directories(input_dir, output_dir):
    """Ensure that input and output directories exist."""
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory '{output_dir}'.")

def get_video_files(input_dir):
    """Retrieve a list of video files from the input directory."""
    supported_extensions = ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv')
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_extensions)]
    if not video_files:
        print(f"No video files found in '{input_dir}'.")
    return video_files

def get_video_duration(input_path):
    """Get the total duration of the video in seconds using ffprobe."""
    cmd_duration = [
        'ffprobe', '-v', 'error', '-show_entries',
        'format=duration', '-of',
        'default=noprint_wrappers=1:nokey=1', input_path
    ]
    try:
        result = subprocess.run(cmd_duration, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        total_duration = float(result.stdout.strip())
        return total_duration
    except subprocess.CalledProcessError as e:
        print(f"Error getting duration of '{input_path}': {e.stderr}")
        return None

def split_segment(input_path, output_path, start_time, segment_duration):
    """
    Split a single segment using FFmpeg.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the output segment.
        start_time (float): Start time of the segment in seconds.
        segment_duration (int): Duration of the segment in seconds.
    """
    # Check if the output file already exists
    if os.path.exists(output_path):
        print(f"Skipping '{output_path}' as it already exists.")
        return

    cmd_split = [
        'ffmpeg',
        '-y',                 # Overwrite without asking
        '-i', input_path,
        '-ss', str(start_time),
        '-t', str(segment_duration),
        '-c:v', 'libx264',    # Use libx264 codec
        '-an',                # Discard audio
        output_path
    ]

    try:
        subprocess.run(cmd_split, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error splitting '{input_path}' at {start_time} seconds: {e}")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Split videos into segments on a local machine.")
    parser.add_argument('--input-dir', type=str, default=DEFAULT_INPUT_DIR,
                        help=f"Directory containing original videos (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save split segments (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument('--segment-duration', type=int, default=DEFAULT_SEGMENT_DURATION,
                        help=f"Duration of each segment in seconds (default: {DEFAULT_SEGMENT_DURATION})")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    segment_duration = args.segment_duration

    print("Starting video segmentation with the following parameters:")
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Segment Duration: {segment_duration} seconds")
    print("-" * 50)

    ensure_directories(input_dir, output_dir)
    video_files = get_video_files(input_dir)

    if not video_files:
        return

    # Build a list of all segments
    all_segments = []
    for filename in video_files:
        input_path = os.path.join(input_dir, filename)
        total_duration = get_video_duration(input_path)
        if total_duration is None:
            continue  # Skip this video due to error

        num_segments = int(total_duration // segment_duration) + (1 if total_duration % segment_duration > 0 else 0)

        base_name = os.path.splitext(filename)[0]
        for segment_idx in range(num_segments):
            start_time = segment_idx * segment_duration
            segment_name = f"{base_name}_{segment_idx:04d}.mp4"
            output_path = os.path.join(output_dir, segment_name)
            all_segments.append((input_path, output_path, start_time, segment_duration))

    total_segments = len(all_segments)
    print(f"Total segments to process: {total_segments}")
    if total_segments == 0:
        print("No segments to process. Exiting.")
        return

    # Process the segments with a progress bar
    with tqdm(total=total_segments, desc="Processing Segments", ncols=100) as pbar:
        for segment_info in all_segments:
            input_path, output_path, start_time, seg_duration = segment_info
            split_segment(input_path, output_path, start_time, seg_duration)
            pbar.update(1)

    print("All segments have been processed successfully.")

if __name__ == "__main__":
    main()
