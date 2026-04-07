"""Plot perturbation sensitivity sweep results.

Reads the .npz file produced by:
    python validate.py ... perturbation.mode=sweep

and produces a single figure showing how the W1 distribution shifts as the
perturbation scale (noise_std) grows.

Usage:
    python plot_perturbation_sweep.py path/to/perturbation_sweep.npz
    python plot_perturbation_sweep.py path/to/perturbation_sweep.npz -o fig.pdf
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def main():
    parser = argparse.ArgumentParser(
        description="Plot W1 sensitivity vs. perturbation scale")
    parser.add_argument("npz", help="Path to perturbation_sweep.npz")
    parser.add_argument("-o", "--output", default=None,
                        help="Output figure path (default: next to the .npz)")
    args = parser.parse_args()

    data = np.load(args.npz)
    stds = np.sort(data['stds'])

    # Collect per-level W1 arrays
    w1_by_std = []
    for s in stds:
        key = f'w1_std{s:.3g}'
        w1_by_std.append(data[key])

    # --- single figure: violin + median trend line ---
    fig, ax = plt.subplots(figsize=(8, 5))

    positions = np.arange(len(stds))
    parts = ax.violinplot(w1_by_std, positions=positions, showextrema=False)

    # Style the violin bodies
    for pc in parts['bodies']:
        pc.set_facecolor('#6C8EBF')
        pc.set_edgecolor('#34495E')
        pc.set_alpha(0.6)

    # Overlay box plots (narrow) for quartiles
    bp = ax.boxplot(w1_by_std, positions=positions, widths=0.18,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='#E74C3C', linewidth=2),
                    boxprops=dict(facecolor='white', edgecolor='#34495E'),
                    whiskerprops=dict(color='#34495E'),
                    capprops=dict(color='#34495E'))

    # Connect medians with a line
    medians = [np.median(w) for w in w1_by_std]
    ax.plot(positions, medians, 'o-', color='#E74C3C', linewidth=1.5,
            markersize=5, zorder=5, label='median')

    # Connect means with a dashed line
    means = [np.mean(w) for w in w1_by_std]
    ax.plot(positions, means, 's--', color='#2ECC71', linewidth=1.5,
            markersize=5, zorder=5, label='mean')

    ax.set_xticks(positions)
    ax.set_xticklabels([f'{s:.3g}' for s in stds])
    ax.set_xlabel('Perturbation scale (noise std)', fontsize=12)
    ax.set_ylabel('Empirical $W_1$ distance (m)', fontsize=12)
    ax.set_title('Trajectory distribution sensitivity to input coordinate perturbation',
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Print summary table
    print(f"{'std':>8s}  {'n':>5s}  {'mean':>8s}  {'median':>8s}"
          f"  {'std_w1':>8s}  {'max':>8s}")
    print('-' * 54)
    for s, w in zip(stds, w1_by_std):
        print(f"{s:8.3g}  {len(w):5d}  {w.mean():8.4f}  {np.median(w):8.4f}"
              f"  {w.std():8.4f}  {w.max():8.4f}")

    fig.tight_layout()

    out_path = args.output
    if out_path is None:
        out_path = args.npz.replace('.npz', '.pdf')
    fig.savefig(out_path, dpi=200)
    print(f"\nFigure saved to: {out_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
