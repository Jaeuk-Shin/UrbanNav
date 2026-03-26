"""
Precompute DINOv2 features for video/image datasets with multi-GPU support.

Supports two input modes:
  - Video files (.mp4) read directly via decord — no frame extraction needed
  - Image directories (legacy mode, reads .jpg files from fcam/ subdirs)

Supports multi-GPU processing via torch.multiprocessing.spawn, distributing
videos across GPUs for near-linear speedup.

Usage:
    # Single GPU, video mode (extracts ALL frames)
    python precompute_features.py \
        --config config/goal_agnostic_fm.yaml \
        --video_dir /path/to/videos \
        --output_dir /path/to/features

    # Multi-GPU (4 GPUs), subsample to 2fps (auto-detects each video's native fps)
    python precompute_features.py \
        --config config/goal_agnostic_fm.yaml \
        --video_dir /path/to/videos \
        --output_dir /path/to/features \
        --num_gpus 4 --target_fps 2

    # Manual fixed frame step (every 15th frame, ignoring native fps)
    python precompute_features.py \
        --config config/goal_agnostic_fm.yaml \
        --video_dir /path/to/videos \
        --output_dir /path/to/features \
        --num_gpus 4 --frame_step 15

    # Legacy image-directory mode (same as before)
    python precompute_features.py \
        --config config/urban_nav.yaml \
        --output_dir /path/to/features

The script saves one .pt file per video/episode containing:
    'features':      (num_frames, feature_dim) tensor
    'features_flip': (num_frames, feature_dim) tensor  [optional, skip with --no_flip]

Already-completed files are automatically skipped, so the script is
safe to resume after interruption.
"""

import os
import tempfile
import argparse
import threading
import torch
import torch.multiprocessing as mp
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
from omegaconf import OmegaConf


# Default timeout (seconds) for reading a single video with decord.
# Corrupted videos can cause decord to hang indefinitely in a retry loop;
# this timeout lets the worker skip them and continue.
VIDEO_READ_TIMEOUT = 120


FEATURE_DIM_MAP = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


def process_frames_to_tensor(frames_np, desired_height, desired_width):
    """Convert (N, H, W, C) uint8 numpy array to (N, C, H, W) float tensor
    with padding/cropping to desired resolution."""
    frames = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0

    _, _, H, W = frames.shape
    pad_height = desired_height - H
    pad_width = desired_width - W

    if pad_height > 0 or pad_width > 0:
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        frames = TF.pad(frames, (pad_left, pad_top, pad_right, pad_bottom))

    if pad_height < 0 or pad_width < 0:
        frames = TF.center_crop(frames, (desired_height, desired_width))

    return frames


def encode_batch(encoder, frames, mean, std, crop, resize,
                 do_rgb_normalize, do_resize, device):
    """Run the frozen DINOv2 encoder on a batch of preprocessed frames."""
    frames = frames.to(device)
    if do_rgb_normalize:
        frames = (frames - mean) / std
    if do_resize:
        frames = TF.center_crop(frames, crop)
        frames = TF.resize(frames, resize)
    with torch.no_grad():
        return encoder(frames).cpu()


# ── Input discovery ──────────────────────────────────────────────────────────


def discover_video_tasks(video_dir):
    """Build task list from a directory of .mp4 files."""
    tasks = []
    for f in sorted(os.listdir(video_dir)):
        if not f.endswith('.mp4'):
            continue
        name = os.path.splitext(f)[0]
        tasks.append({'name': name, 'type': 'video',
                      'path': os.path.join(video_dir, f)})
    return tasks


def discover_image_tasks(data_dir, pose_dir):
    """Build task list from pose files + fcam image directories (legacy)."""
    tasks = []
    pose_files = sorted([f for f in os.listdir(pose_dir) if f.endswith('.txt')])
    for pf in pose_files:
        name = os.path.splitext(pf)[0]
        img_dir = os.path.join(data_dir, name, 'fcam')
        if os.path.isdir(img_dir):
            tasks.append({'name': name, 'type': 'images', 'path': img_dir})
    return tasks


# ── Per-video feature extraction ─────────────────────────────────────────────


def _run_with_timeout(fn, args=(), kwargs=None, timeout=VIDEO_READ_TIMEOUT):
    """Run *fn* in a daemon thread and return its result, or raise
    ``TimeoutError`` if it doesn't finish within *timeout* seconds.

    This is used to guard against decord hanging indefinitely on corrupted
    videos.  A daemon thread is used (instead of a subprocess) so it works
    inside already-spawned worker processes without nested-spawn issues.
    """
    result = [None]
    error = [None]

    def _target():
        try:
            result[0] = fn(*args, **(kwargs or {}))
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        # Thread is stuck in decord C code — we can't kill it, but as a
        # daemon thread it will be cleaned up when the process exits.
        raise TimeoutError(
            f"Video read timed out after {timeout}s (likely corrupted)")
    if error[0] is not None:
        raise error[0]
    return result[0]


def _read_video_frames(video_path, frame_step, target_fps):
    """Read and subsample frames from a video file using decord."""
    from decord import VideoReader, cpu as decord_cpu

    vr = VideoReader(video_path, ctx=decord_cpu(0))
    total_frames = len(vr)

    if frame_step > 1:
        effective_step = frame_step
    elif target_fps is not None:
        native_fps = vr.get_avg_fps()
        effective_step = max(1, round(native_fps / target_fps))
    else:
        effective_step = 1

    frame_indices = list(range(0, total_frames, effective_step))
    frames_np = vr.get_batch(frame_indices).asnumpy()
    return frame_indices, frames_np


def extract_from_video(video_path, encoder, batch_size, desired_height,
                       desired_width, mean, std, crop, resize,
                       do_rgb_normalize, do_resize, device, include_flip,
                       frame_step, target_fps):
    """Extract DINOv2 features from all (or subsampled) frames of a video.

    Frame selection priority:
      1. If frame_step is given (> 1), use it directly.
      2. If target_fps is given, compute frame_step from the video's native fps.
      3. Otherwise, extract all frames (frame_step=1).

    The decord read is wrapped in a timeout to skip videos that hang.
    """
    frame_indices, frames_np = _run_with_timeout(
        _read_video_frames, args=(video_path, frame_step, target_fps))
    num_frames = len(frame_indices)

    all_features = []
    all_features_flip = [] if include_flip else None

    for start in range(0, num_frames, batch_size):
        end = min(start + batch_size, num_frames)
        batch_np = frames_np[start:end]

        frames = process_frames_to_tensor(batch_np, desired_height,
                                          desired_width)
        features = encode_batch(encoder, frames, mean, std, crop, resize,
                                do_rgb_normalize, do_resize, device)
        all_features.append(features)

        if include_flip:
            frames_flip = torch.flip(frames, dims=[-1])
            features_flip = encode_batch(encoder, frames_flip, mean, std, crop,
                                         resize, do_rgb_normalize, do_resize,
                                         device)
            all_features_flip.append(features_flip)

    result = {'features': torch.cat(all_features, dim=0)}
    if include_flip:
        result['features_flip'] = torch.cat(all_features_flip, dim=0)
    return result


def extract_from_images(img_dir, encoder, batch_size, desired_height,
                        desired_width, mean, std, crop, resize,
                        do_rgb_normalize, do_resize, device, include_flip,
                        frame_step):
    """Extract DINOv2 features from .jpg files in a directory."""
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    img_files = img_files[::frame_step]
    num_frames = len(img_files)
    if num_frames == 0:
        return None

    all_features = []
    all_features_flip = [] if include_flip else None

    for start in range(0, num_frames, batch_size):
        end = min(start + batch_size, num_frames)
        imgs = [Image.open(os.path.join(img_dir, img_files[i])).convert('RGB')
                for i in range(start, end)]
        frames_np = np.stack([np.array(img) for img in imgs])

        frames = process_frames_to_tensor(frames_np, desired_height,
                                          desired_width)
        features = encode_batch(encoder, frames, mean, std, crop, resize,
                                do_rgb_normalize, do_resize, device)
        all_features.append(features)

        if include_flip:
            frames_flip = torch.flip(frames, dims=[-1])
            features_flip = encode_batch(encoder, frames_flip, mean, std, crop,
                                         resize, do_rgb_normalize, do_resize,
                                         device)
            all_features_flip.append(features_flip)

    result = {'features': torch.cat(all_features, dim=0)}
    if include_flip:
        result['features_flip'] = torch.cat(all_features_flip, dim=0)
    return result


# ── Task filtering ───────────────────────────────────────────────────────────


def _is_valid_pt(path):
    """Check whether a .pt file is loadable without reading full tensors.

    torch.load with mmap loads only the pickle metadata — tensor data stays
    on disk.  This catches truncated/corrupt files at negligible I/O cost.
    """
    try:
        torch.load(path, weights_only=True, map_location='cpu',
                   mmap=True)
        return True
    except Exception:
        return False


def _filter_remaining_tasks(tasks, output_dir):
    """Remove already-completed tasks before distributing to workers.

    Each existing .pt file is validated via torch.load (memory-mapped, so
    only the pickle header is read — fast even for large tensors).  Corrupt
    files are deleted so they get re-processed.
    Returns (remaining, num_skipped, num_removed).
    """
    remaining = []
    skipped = 0
    removed = 0
    for task in tasks:
        output_path = os.path.join(output_dir, f"{task['name']}.pt")
        if os.path.exists(output_path):
            if _is_valid_pt(output_path):
                skipped += 1
                continue
            # Corrupt / truncated — remove and re-process
            print(f"  Removing corrupt .pt: {task['name']}")
            try:
                os.remove(output_path)
            except OSError:
                pass
            removed += 1
        remaining.append(task)
    return remaining, skipped, removed


# ── Worker (one per GPU) ─────────────────────────────────────────────────────


def worker(rank, world_size, shard, cfg, output_dir, batch_size, include_flip,
           frame_step, target_fps):
    """Process a pre-assigned shard of tasks on a single GPU."""
    import warnings
    warnings.filterwarnings('ignore', message='xFormers is available')

    device = f'cuda:{rank}'

    encoder_type = cfg.model.obs_encoder.type
    print(f"[GPU {rank}] Loading {encoder_type} — {len(shard)} tasks assigned",
          flush=True)
    encoder = torch.hub.load('facebookresearch/dinov2', encoder_type)
    encoder.eval()
    encoder.to(device)

    do_rgb_normalize = cfg.model.do_rgb_normalize
    do_resize = cfg.model.do_resize
    crop = list(cfg.model.obs_encoder.crop)
    resize = list(cfg.model.obs_encoder.resize)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    desired_height = int(cfg.data.desired_height)
    desired_width = int(cfg.data.desired_width)

    processed = 0
    errors = 0
    pbar = tqdm(shard, desc=f"GPU {rank}", position=rank, leave=True)

    for task in pbar:
        name = task['name']
        output_path = os.path.join(output_dir, f'{name}.pt')

        pbar.set_postfix_str(name[:30])

        try:
            if task['type'] == 'video':
                result = extract_from_video(
                    task['path'], encoder, batch_size, desired_height,
                    desired_width, mean, std, crop, resize,
                    do_rgb_normalize, do_resize, device, include_flip,
                    frame_step, target_fps)
            else:
                result = extract_from_images(
                    task['path'], encoder, batch_size, desired_height,
                    desired_width, mean, std, crop, resize,
                    do_rgb_normalize, do_resize, device, include_flip,
                    frame_step)

            if result is not None:
                # Atomic write: save to temp file then rename, so a crash
                # never leaves a half-written .pt that would be skipped
                fd, tmp_path = tempfile.mkstemp(
                    dir=output_dir, suffix='.pt.tmp')
                os.close(fd)
                try:
                    torch.save(result, tmp_path)
                    os.replace(tmp_path, output_path)
                except BaseException:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    raise
        except Exception as e:
            tqdm.write(f"[GPU {rank}] Error processing {name}: {e}")
            errors += 1

        processed += 1

    pbar.close()
    print(f"[GPU {rank}] Done. Processed {processed} tasks"
          f"{f' ({errors} errors)' if errors else ''}.")


# ── Entry point ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Precompute DINOv2 features (video/image, multi-GPU)")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML (for model/encoder settings)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for .pt feature files')
    parser.add_argument('--video_dir', type=str, default=None,
                        help='Directory of .mp4 video files. If omitted, falls '
                             'back to image directories from config.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs for parallel processing')
    parser.add_argument('--target_fps', type=float, default=None,
                        help='Target FPS for frame subsampling. Each video\'s '
                             'native FPS is queried via decord to compute the '
                             'frame step automatically. Ignored if --frame_step '
                             'is also set.')
    parser.add_argument('--frame_step', type=int, default=1,
                        help='Fixed frame step (process every N-th frame). '
                             'Takes priority over --target_fps. Default 1 = '
                             'all frames.')
    parser.add_argument('--no_flip', action='store_true',
                        help='Skip precomputing features for flipped frames')
    parser.add_argument('--desired_height', type=int, default=None,
                        help='Override desired frame height (default: from config)')
    parser.add_argument('--desired_width', type=int, default=None,
                        help='Override desired frame width (default: from config)')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # Allow CLI overrides for desired resolution
    if args.desired_height is not None:
        cfg.data.desired_height = args.desired_height
    elif not hasattr(cfg.data, 'desired_height'):
        cfg.data.desired_height = 360  # sensible default
    if args.desired_width is not None:
        cfg.data.desired_width = args.desired_width
    elif not hasattr(cfg.data, 'desired_width'):
        cfg.data.desired_width = 640

    include_flip = not args.no_flip

    # Discover tasks
    if args.video_dir is not None:
        tasks = discover_video_tasks(args.video_dir)
        print(f"Found {len(tasks)} videos in {args.video_dir}")
    else:
        data_dir = cfg.data.root_dir
        pose_dir_raw = cfg.data.pose_dir
        # Handle both absolute and relative pose_dir
        if os.path.isabs(pose_dir_raw):
            pose_dir = pose_dir_raw
        else:
            pose_dir = os.path.join(data_dir, pose_dir_raw)
        tasks = discover_image_tasks(data_dir, pose_dir)
        print(f"Found {len(tasks)} image episodes in {data_dir}")

    if len(tasks) == 0:
        print("No tasks found. Nothing to do.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Save metadata
    encoder_type = cfg.model.obs_encoder.type
    target_fps = args.target_fps
    metadata = {
        'encoder_type': encoder_type,
        'feature_dim': FEATURE_DIM_MAP[encoder_type],
        'crop': list(cfg.model.obs_encoder.crop),
        'resize': list(cfg.model.obs_encoder.resize),
        'do_rgb_normalize': cfg.model.do_rgb_normalize,
        'do_resize': cfg.model.do_resize,
        'desired_height': int(cfg.data.desired_height),
        'desired_width': int(cfg.data.desired_width),
        'include_flip': include_flip,
        'frame_step': args.frame_step,
        'target_fps': target_fps,
    }
    torch.save(metadata, os.path.join(args.output_dir, 'metadata.pt'))

    # Filter out already-completed tasks so work is balanced across GPUs
    remaining, skipped, removed = _filter_remaining_tasks(
        tasks, args.output_dir)
    total_remaining = len(remaining)
    print(f"{total_remaining} tasks remaining out of {len(tasks)} total "
          f"({skipped} already done"
          f"{f', {removed} corrupt removed' if removed else ''})")

    if total_remaining == 0:
        print("All tasks already completed. Nothing to do.")
        return

    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    subsample_desc = (f"target_fps={target_fps}" if target_fps
                      else f"frame_step={args.frame_step}")
    print(f"Using {num_gpus} GPU(s), batch_size={args.batch_size}, "
          f"{subsample_desc}, include_flip={include_flip}")

    # Round-robin shard the *remaining* tasks so each GPU gets an equal
    # share.  Since completed tasks were already filtered out above, the
    # shards are balanced even on resume.
    shards = [remaining[i::num_gpus] for i in range(num_gpus)]
    for i, s in enumerate(shards):
        print(f"  GPU {i}: {len(s)} tasks")

    if num_gpus <= 1:
        worker(0, 1, shards[0], cfg, args.output_dir, args.batch_size,
               include_flip, args.frame_step, target_fps)
    else:
        processes = []
        for rank in range(num_gpus):
            p = mp.Process(
                target=worker,
                args=(rank, num_gpus, shards[rank], cfg, args.output_dir,
                      args.batch_size, include_flip, args.frame_step,
                      target_fps))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        for rank, p in enumerate(processes):
            if p.exitcode != 0:
                print(f"Warning: GPU {rank} worker exited with code "
                      f"{p.exitcode}")

    print(f"All done. Features saved to {args.output_dir}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
