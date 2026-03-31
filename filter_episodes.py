"""filter_episodes.py — Compute per-episode pose quality metrics and produce a
keep-list for training data curation.

Designed for the YouTube walking-video dataset where some episodes contain
camera poses that are unsuitable for wheeled-robot trajectory learning (e.g.
looking at the sky, sideways gaze, escalators).

Usage
-----
# 1. Compute metrics and write keep-list (default thresholds)
python filter_episodes.py --pose_dir /data/youtube_videos/pose

# 2. Same, but also save histogram plots for threshold tuning
python filter_episodes.py --pose_dir /data/youtube_videos/pose --plot

# 3. Inspect borderline rejected episodes (closest to each threshold)
python filter_episodes.py --pose_dir /data/youtube_videos/pose --sample_rejected 20

# 4. Override a threshold
python filter_episodes.py --pose_dir /data/youtube_videos/pose --max_pitch 30

Outputs
-------
- <output>.json          per-episode metrics + valid flag
- <output>_keep.txt      newline-delimited filenames of valid episodes
- <output>_plots/        (with --plot) histogram PNGs with threshold lines
- stdout                 (with --sample_rejected) borderline episodes to inspect
"""

import os
import argparse
import json

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


# ── Metric computation ────────────────────────────────────────────────────────

def compute_metrics(pose):
    """Compute pose-quality metrics for a single episode.

    Parameters
    ----------
    pose : ndarray, shape (N, 7)
        Each row is [x, y, z, qx, qy, qz, qw] in a Y-up world frame.

    Returns
    -------
    dict with keys:
        median_pitch, std_pitch      – camera optical-axis elevation (degrees)
        median_roll, std_roll        – camera roll deviation from upright (degrees)
        gaze_motion_angle            – median angle between gaze and velocity (degrees)
        height_std                   – std of Y coordinate (metres, up to DPVO scale)
        step_scale                   – mean inter-frame XZ displacement
        num_frames                   – number of valid frames
    """
    quats = pose[:, 3:]
    rot = R.from_quat(quats)

    # Camera optical axis (+Z in camera frame) in world coordinates
    cam_fwd = rot.apply([0, 0, 1])  # (N, 3)

    # ── Pitch: elevation of optical axis ──
    pitch_deg = np.degrees(np.arcsin(np.clip(cam_fwd[:, 1], -1, 1)))

    # ── Roll: deviation of camera-up from world-up ──
    cam_up = rot.apply([0, -1, 0])  # (N, 3)
    roll_cos = np.clip(np.abs(cam_up[:, 1]), 0, 1)
    roll_deg = np.degrees(np.arccos(roll_cos))

    # ── Motion–gaze alignment ──
    pos_xz = pose[:, [0, 2]]
    vel = np.diff(pos_xz, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    gaze_xz = cam_fwd[:-1, [0, 2]]
    gaze_norm = np.linalg.norm(gaze_xz, axis=1)
    moving = speed > 0.05
    if moving.sum() >= 5:
        cos_a = np.sum(vel[moving] * gaze_xz[moving], axis=1) / (
            speed[moving] * gaze_norm[moving] + 1e-8
        )
        gaze_angle = float(np.median(np.degrees(np.arccos(np.clip(cos_a, -1, 1)))))
    else:
        gaze_angle = 180.0

    # ── Height variation ──
    height_std = float(np.std(pose[:, 1]))

    # ── Step scale (mean XZ displacement per frame) ──
    step_scale = float(np.linalg.norm(np.diff(pos_xz, axis=0), axis=1).mean())

    return {
        'median_pitch': float(np.median(pitch_deg)),
        'std_pitch': float(np.std(pitch_deg)),
        'median_roll': float(np.median(roll_deg)),
        'std_roll': float(np.std(roll_deg)),
        'gaze_motion_angle': gaze_angle,
        'height_std': height_std,
        'step_scale': step_scale,
        'num_frames': int(pose.shape[0]),
    }


# ── Threshold logic ──────────────────────────────────────────────────────────

DEFAULT_THRESHOLDS = {
    'max_pitch': 25.0,         # |median pitch| must be below this (degrees)
    'max_pitch_std': 30.0,     # pitch standard deviation cap
    'max_roll': 20.0,          # median roll cap
    'max_gaze_angle': 45.0,    # gaze–motion misalignment cap
    'max_height_std': 1.5,     # height variation cap (metres, DPVO scale)
    'min_step_scale': 0.1,     # minimum walking speed proxy
    'max_step_scale': 4.0,     # maximum walking speed proxy
    'min_frames': 20,          # minimum usable frames after NaN truncation
}


def is_valid(m, thresh):
    return (
        abs(m['median_pitch']) < thresh['max_pitch']
        and m['std_pitch'] < thresh['max_pitch_std']
        and m['median_roll'] < thresh['max_roll']
        and m['gaze_motion_angle'] < thresh['max_gaze_angle']
        and m['height_std'] < thresh['max_height_std']
        and thresh['min_step_scale'] < m['step_scale'] < thresh['max_step_scale']
        and m['num_frames'] >= thresh['min_frames']
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

METRIC_INFO = [
    # (metric_key,          display_name,               threshold_key,     use_abs, side)
    ('median_pitch',       'Median Pitch (°)',          'max_pitch',        True,   'right'),
    ('std_pitch',          'Pitch Std Dev (°)',         'max_pitch_std',    False,  'right'),
    ('median_roll',        'Median Roll (°)',           'max_roll',         False,  'right'),
    ('gaze_motion_angle',  'Gaze–Motion Angle (°)',     'max_gaze_angle',  False,  'right'),
    ('height_std',         'Height Std Dev',            'max_height_std',   False,  'right'),
    ('step_scale',         'Step Scale (m/frame)',       None,              False,  'both'),
]


def save_plots(results, thresh, plot_dir):
    """Save per-metric histograms with threshold lines."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)

    for metric_key, display_name, thresh_key, use_abs, side in METRIC_INFO:
        values = [m[metric_key] for m in results.values() if metric_key in m]
        if use_abs:
            values = [abs(v) for v in values]
        values = np.array(values)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(values, bins=100, color='steelblue', edgecolor='none', alpha=0.8)
        ax.set_xlabel(display_name)
        ax.set_ylabel('Episode count')
        ax.set_title(f'{display_name}  (n={len(values)})')

        if thresh_key is not None and side in ('right', 'both'):
            ax.axvline(thresh[thresh_key], color='red', ls='--', lw=1.5,
                       label=f'threshold = {thresh[thresh_key]}')
        if thresh_key is None and side == 'both':
            # step_scale has both min and max
            ax.axvline(thresh['min_step_scale'], color='red', ls='--', lw=1.5,
                       label=f'min = {thresh["min_step_scale"]}')
            ax.axvline(thresh['max_step_scale'], color='red', ls='--', lw=1.5,
                       label=f'max = {thresh["max_step_scale"]}')

        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f'{metric_key}.png'), dpi=150)
        plt.close(fig)

    # Summary: pass-rate pie chart
    valid_count = sum(1 for m in results.values() if m.get('valid', False))
    rejected_count = len(results) - valid_count
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie([valid_count, rejected_count],
           labels=[f'Keep ({valid_count})', f'Reject ({rejected_count})'],
           colors=['#4CAF50', '#F44336'], autopct='%1.1f%%', startangle=90)
    ax.set_title('Episode Filter Summary')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'summary.png'), dpi=150)
    plt.close(fig)

    print(f"Plots saved to {plot_dir}/")


# ── Borderline sampling ──────────────────────────────────────────────────────

def print_borderline(results, thresh, n=20):
    """Print episodes closest to each rejection threshold (both sides)."""

    print(f"\n{'='*80}")
    print(f"Borderline episodes (closest {n} to each threshold)")
    print(f"{'='*80}")

    checks = [
        ('median_pitch', 'max_pitch', True),
        ('std_pitch', 'max_pitch_std', False),
        ('median_roll', 'max_roll', False),
        ('gaze_motion_angle', 'max_gaze_angle', False),
        ('height_std', 'max_height_std', False),
        ('step_scale', 'min_step_scale', False),
        ('step_scale', 'max_step_scale', False),
    ]

    for metric_key, thresh_key, use_abs in checks:
        t = thresh[thresh_key]
        items = []
        for fname, m in results.items():
            val = abs(m[metric_key]) if use_abs else m[metric_key]
            distance = val - t  # positive = rejected side
            items.append((fname, val, distance, m.get('valid', False)))

        # Sort by absolute distance to threshold
        items.sort(key=lambda x: abs(x[2]))

        print(f"\n── {metric_key} (threshold: {thresh_key} = {t}) ──")
        print(f"{'File':<60} {'Value':>8} {'Dist':>8} {'Valid':>6}")
        for fname, val, dist, valid in items[:n]:
            tag = 'KEEP' if valid else 'REJECT'
            print(f"{fname:<60} {val:>8.2f} {dist:>+8.2f} {tag:>6}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compute pose-quality metrics and filter invalid episodes.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--pose_dir', required=True,
                        help='Directory containing per-episode .txt pose files')
    parser.add_argument('--output', default='valid_episodes.json',
                        help='Output JSON path for per-episode metrics')
    parser.add_argument('--pose_step', type=int, default=2,
                        help='Downsample factor for pose files (pose_fps / target_fps)')

    # Threshold overrides
    for key, default in DEFAULT_THRESHOLDS.items():
        typ = type(default)
        parser.add_argument(f'--{key}', type=typ, default=default,
                            help=f'Threshold: {key}')

    # Inspection modes
    parser.add_argument('--plot', action='store_true',
                        help='Save per-metric histogram plots')
    parser.add_argument('--sample_rejected', type=int, default=0, metavar='N',
                        help='Print N borderline episodes per metric')

    args = parser.parse_args()

    thresh = {k: getattr(args, k) for k in DEFAULT_THRESHOLDS}

    # Scan pose files
    pose_files = sorted(f for f in os.listdir(args.pose_dir) if f.endswith('.txt'))
    if not pose_files:
        print(f"No .txt files found in {args.pose_dir}")
        return

    print(f"Processing {len(pose_files)} pose files from {args.pose_dir} ...")

    results = {}
    valid, total = 0, 0
    for fname in tqdm(pose_files, desc='Computing metrics'):
        raw = np.loadtxt(os.path.join(args.pose_dir, fname), delimiter=' ')
        pose = raw[::args.pose_step, 1:]  # skip timestamp column, downsample

        # Truncate at first NaN
        nan_mask = np.isnan(pose).any(axis=1)
        if nan_mask.any():
            pose = pose[:np.argmax(nan_mask)]
        if pose.shape[0] < 3:
            results[fname] = {'num_frames': int(pose.shape[0]), 'valid': False,
                              'reject_reason': 'too_short_after_nan_truncation'}
            continue

        total += 1
        m = compute_metrics(pose)
        m['valid'] = is_valid(m, thresh)
        if not m['valid']:
            # Record which criterion failed first (for diagnostics)
            reasons = []
            if abs(m['median_pitch']) >= thresh['max_pitch']:
                reasons.append('pitch')
            if m['std_pitch'] >= thresh['max_pitch_std']:
                reasons.append('pitch_std')
            if m['median_roll'] >= thresh['max_roll']:
                reasons.append('roll')
            if m['gaze_motion_angle'] >= thresh['max_gaze_angle']:
                reasons.append('gaze_align')
            if m['height_std'] >= thresh['max_height_std']:
                reasons.append('height')
            if m['step_scale'] <= thresh['min_step_scale']:
                reasons.append('too_slow')
            if m['step_scale'] >= thresh['max_step_scale']:
                reasons.append('too_fast')
            if m['num_frames'] < thresh['min_frames']:
                reasons.append('too_short')
            m['reject_reasons'] = reasons
        results[fname] = m
        if m['valid']:
            valid += 1

    print(f"\nValid: {valid}/{total} ({100*valid/max(total,1):.1f}%)")

    # Per-reason rejection breakdown
    reason_counts = {}
    for m in results.values():
        for r in m.get('reject_reasons', []):
            reason_counts[r] = reason_counts.get(r, 0) + 1
    if reason_counts:
        print("\nRejection breakdown (episodes may fail multiple criteria):")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason:<16} {count:>6}")

    # Write JSON
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics written to {args.output}")

    # Write keep-list
    keep_path = args.output.replace('.json', '_keep.txt')
    keep = [k for k, v in results.items() if v.get('valid', False)]
    with open(keep_path, 'w') as f:
        f.write('\n'.join(keep) + '\n')
    print(f"Keep-list written to {keep_path} ({len(keep)} episodes)")

    # Optional: plots
    if args.plot:
        plot_dir = args.output.replace('.json', '_plots')
        save_plots(results, thresh, plot_dir)

    # Optional: borderline sampling
    if args.sample_rejected > 0:
        print_borderline(results, thresh, n=args.sample_rejected)


if __name__ == '__main__':
    main()
