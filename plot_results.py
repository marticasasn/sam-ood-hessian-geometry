"""
Figures for the experiments section.

  Fig A - Train / test accuracy vs epoch (mean ±1 std across seeds)
  Fig B - Mean CIFAR-10-C accuracy per config, bar chart
  Fig C - Per-corruption accuracy heatmap (config × 15 corruptions, averaged over severities)

Usage:
  pip install matplotlib pandas numpy
  python plot_results.py [--results-dir ./results] [--output-dir ./results/figures]
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Global style ──────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family':        'sans-serif',
    'font.size':          10,
    'axes.titlesize':     11,
    'axes.labelsize':     10,
    'legend.fontsize':    9,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.dpi':         150,
})

CONFIGS = ['sgd_baseline', 'sam_rho005', 'sam_rho010', 'sam_rho020']

LABELS = {
    'sgd_baseline': 'SGD',
    'sam_rho005':   r'SAM $\rho$=0.05',
    'sam_rho010':   r'SAM $\rho$=0.10',
    'sam_rho020':   r'SAM $\rho$=0.20',
}

# Seaborn "deep" palette - consistent across all three figures
COLORS = {
    'sgd_baseline': '#4C72B0',
    'sam_rho005':   '#DD8452',
    'sam_rho010':   '#55A868',
    'sam_rho020':   '#C44E52',
}

CORRUPTION_ORDER = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]
CORRUPTION_TICK_LABELS = [
    'Gauss.', 'Shot', 'Impulse', 'Defocus', 'Glass',
    'Motion', 'Zoom', 'Snow', 'Frost', 'Fog',
    'Bright.', 'Contrast', 'Elastic', 'Pixelate', 'JPEG',
]


def save(fig, stem, output_dir):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(output_dir, f'{stem}.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)


# ── Fig A: training curves ────────────────────────────────────────────────────
def fig_a(tr, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, split, title in [
        (axes[0], 'train', 'Train Accuracy'),
        (axes[1], 'test',  'Test Accuracy'),
    ]:
        mean_col = f'mean_{split}_acc'
        std_col  = f'std_{split}_acc'

        for cfg in CONFIGS:
            sub = tr[tr['config'] == cfg].sort_values('epoch')
            if sub.empty:
                continue
            c   = COLORS[cfg]
            mu  = sub[mean_col]
            sig = sub[std_col].fillna(0)
            ax.plot(sub['epoch'], mu, color=c, label=LABELS[cfg], linewidth=1.8)
            ax.fill_between(sub['epoch'], mu - sig, mu + sig, alpha=0.15, color=c)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(title)
        ax.set_xlim(tr['epoch'].min(), tr['epoch'].max())

    axes[1].legend(loc='lower right', framealpha=0.9)
    fig.tight_layout()
    save(fig, 'fig_a_training_curves', output_dir)


# ── Fig B: headline robustness bar chart ─────────────────────────────────────
def fig_b(cifar10c_summary, output_dir):
    summary = cifar10c_summary.set_index('config').reindex(CONFIGS)

    fig, ax = plt.subplots(figsize=(5, 4))
    xs = range(len(CONFIGS))
    bars = ax.bar(
        xs,
        summary['mean_acc'].values,
        yerr=summary['std_acc'].values,
        color=[COLORS[c] for c in CONFIGS],
        width=0.55,
        edgecolor='white',
        linewidth=0.8,
        capsize=4,
        error_kw={'elinewidth': 1.2, 'ecolor': 'black'},
    )
    ax.bar_label(bars, fmt='%.1f%%', padding=6, fontsize=9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([LABELS[c] for c in CONFIGS], rotation=20, ha='right')
    ax.set_ylabel('Mean CIFAR-10-C Accuracy (%)')
    ax.set_title('CIFAR-10-C Robustness')
    ymin = max(0, summary['mean_acc'].min() - 5)
    ax.set_ylim(ymin, summary['mean_acc'].max() + 9)

    fig.tight_layout()
    save(fig, 'fig_b_corruption_bar', output_dir)


# ── Fig C: per-corruption delta heatmap (vs SGD) ─────────────────────────────
def fig_c(cifar10c_by_corruption, output_dir):
    pivot = (
        cifar10c_by_corruption
        .set_index(['config', 'corruption'])['mean_acc']
        .unstack('corruption')
        .reindex(index=CONFIGS, columns=CORRUPTION_ORDER)
    )
    data = pivot.values.astype(float)   # (4, 15)

    # delta relative to sgd_baseline row
    sgd_idx = CONFIGS.index('sgd_baseline')
    delta = data - data[sgd_idx, :]

    abs_max = np.nanmax(np.abs(delta))
    vmin, vmax = -abs_max, abs_max

    fig, ax = plt.subplots(figsize=(13, 2.8))
    im = ax.imshow(delta, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)

    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            val = delta[i, j]
            if np.isnan(val):
                continue
            sign = '+' if val >= 0 else ''
            rel = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            text_color = 'white' if rel < 0.15 or rel > 0.85 else 'black'
            ax.text(j, i, f'{sign}{val:.1f}', ha='center', va='center',
                    fontsize=7.5, color=text_color)

    ax.set_xticks(range(len(CORRUPTION_ORDER)))
    ax.set_xticklabels(CORRUPTION_TICK_LABELS, rotation=40, ha='right', fontsize=8.5)
    ax.set_yticks(range(len(CONFIGS)))
    ax.set_yticklabels([LABELS[c] for c in CONFIGS], fontsize=9)
    ax.set_title('Per-Corruption Accuracy Difference vs SGD')

    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
    cbar.set_label('Accuracy difference (percentage points)', fontsize=9)

    fig.tight_layout()
    save(fig, 'fig_c_corruption_delta_heatmap', output_dir)


# ── Entry point ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--results-dir', type=str, default='./results')
parser.add_argument('--output-dir',  type=str, default=None)
args = parser.parse_args()

OUTPUT_DIR = args.output_dir or os.path.join(args.results_dir, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

tr                   = pd.read_csv(os.path.join(args.results_dir, 'training_agg.csv'))
cifar10c_summary     = pd.read_csv(os.path.join(args.results_dir, 'cifar10c_summary.csv'))
cifar10c_by_corr     = pd.read_csv(os.path.join(args.results_dir, 'cifar10c_by_corruption.csv'))

fig_a(tr,               OUTPUT_DIR)
fig_b(cifar10c_summary, OUTPUT_DIR)
fig_c(cifar10c_by_corr, OUTPUT_DIR)

print(f'6 files written to {OUTPUT_DIR}')
