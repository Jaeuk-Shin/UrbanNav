"""Plot perturbation sensitivity sweep results.

Reads the .npz file produced by:
    python validate.py ... perturbation.mode=sweep         # noise sweep
    python validate.py ... perturbation.mode=scale_sweep   # scale sweep

and produces a single figure showing how the W1 distribution shifts as the
perturbation parameter grows.

Usage:
    python plot_perturbation_sweep.py path/to/perturbation_sweep.npz
    python plot_perturbation_sweep.py path/to/perturbation_scale_sweep.npz
    python plot_perturbation_sweep.py path/to/perturbation_sweep.npz -o fig.pdf
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def _load_sweep(path):
    """Load an .npz sweep file and return (param_values, w1_arrays, xlabel).

    Supports both noise sweeps ('stds' key, 'w1_std*' arrays) and scale
    sweeps ('scales' key, 'w1_scale*' arrays).
    """
    data = np.load(path)
    if 'stds' in data:
        params = np.sort(data['stds'])
        w1s = [data[f'w1_std{s:.3g}'] for s in params]
        xlabel = 'Perturbation scale (noise std)'
    elif 'scales' in data:
        params = np.sort(data['scales'])
        w1s = [data[f'w1_scale{s:.3g}'] for s in params]
        xlabel = 'Coordinate scale factor'
    else:
        raise ValueError(f"Unrecognised sweep file — expected 'stds' or "
                         f"'scales' key, got: {list(data.keys())}")
    return params, w1s, xlabel


def main():
    parser = argparse.ArgumentParser(
        description="Plot W1 sensitivity vs. perturbation scale")
    parser.add_argument("npz", help="Path to perturbation_sweep.npz or "
                        "perturbation_scale_sweep.npz")
    parser.add_argument("-o", "--output", default=None,
                        help="Output figure path (default: next to the .npz)")
    args = parser.parse_args()

    params, w1_by_level, xlabel = _load_sweep(args.npz)

    # --- single figure: violin + median trend line ---
    fig, ax = plt.subplots(figsize=(8, 5))

    positions = np.arange(len(params))
    parts = ax.violinplot(w1_by_level, positions=positions, showextrema=False)

    # Style the violin bodies
    for pc in parts['bodies']:
        pc.set_facecolor('#6C8EBF')
        pc.set_edgecolor('#34495E')
        pc.set_alpha(0.6)

    # Overlay box plots (narrow) for quartiles
    ax.boxplot(w1_by_level, positions=positions, widths=0.18,
               patch_artist=True, showfliers=False,
               medianprops=dict(color='#E74C3C', linewidth=2),
               boxprops=dict(facecolor='white', edgecolor='#34495E'),
               whiskerprops=dict(color='#34495E'),
               capprops=dict(color='#34495E'))

    # Connect medians with a line
    medians = [np.median(w) for w in w1_by_level]
    ax.plot(positions, medians, 'o-', color='#E74C3C', linewidth=1.5,
            markersize=5, zorder=5, label='median')

    # Connect means with a dashed line
    means = [np.mean(w) for w in w1_by_level]
    ax.plot(positions, means, 's--', color='#2ECC71', linewidth=1.5,
            markersize=5, zorder=5, label='mean')

    ax.set_xticks(positions)
    ax.set_xticklabels([f'{p:.3g}' for p in params])
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Empirical $W_1$ distance (m)', fontsize=12)
    ax.set_title('Trajectory distribution sensitivity to input coordinate '
                 'perturbation', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Print summary table
    print(f"{'param':>8s}  {'n':>5s}  {'mean':>8s}  {'median':>8s}"
          f"  {'std_w1':>8s}  {'max':>8s}")
    print('-' * 54)
    for p, w in zip(params, w1_by_level):
        print(f"{p:8.3g}  {len(w):5d}  {w.mean():8.4f}  {np.median(w):8.4f}"
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
