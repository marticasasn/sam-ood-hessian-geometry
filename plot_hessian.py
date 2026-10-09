"""
Figures for the Hessian analysis section.

  Fig D - λ_max vs training epoch, one line per config (±1 std across seeds)
  Fig E - λ_max at epoch 150 vs SAM ρ  (core flatness claim; SGD as baseline)
  Fig F - Hessian trace vs training epoch

Usage:
  python plot_hessian.py [--results-dir ./results] [--output-dir ./results/figures]
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Shared style (identical to plot_results.py) ───────────────────────────────
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
SAM_CONFIGS = ['sam_rho005', 'sam_rho010', 'sam_rho020']
SAM_RHO = {'sam_rho005': 0.05, 'sam_rho010': 0.10, 'sam_rho020': 0.20}

LABELS = {
    'sgd_baseline': 'SGD',
    'sam_rho005':   r'SAM $\rho$=0.05',
    'sam_rho010':   r'SAM $\rho$=0.10',
    'sam_rho020':   r'SAM $\rho$=0.20',
}
COLORS = {
    'sgd_baseline': '#4C72B0',
    'sam_rho005':   '#DD8452',
    'sam_rho010':   '#55A868',
    'sam_rho020':   '#C44E52',
}


def save(fig, stem, output_dir):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(output_dir, f'{stem}.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)


def _epoch_line(ax, h, metric_mean, metric_std, ylabel, title):
    """Shared helper for Figs D and F - metric vs epoch."""
    for cfg in CONFIGS:
        sub = h[h['config'] == cfg].sort_values('epoch')
        if sub.empty:
            continue
        c   = COLORS[cfg]
        mu  = sub[metric_mean]
        sig = sub[metric_std].fillna(0)
        ax.plot(sub['epoch'], mu, color=c, label=LABELS[cfg], linewidth=1.8)
        ax.fill_between(sub['epoch'], mu - sig, mu + sig, alpha=0.15, color=c)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(h['epoch'].min(), h['epoch'].max())
    ax.legend(loc='upper right', framealpha=0.9)


# ── Fig D: λ_max vs epoch ────────────────────────────────────────────────────
def fig_d(h, output_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    _epoch_line(ax, h,
                metric_mean='mean_lambda_max',
                metric_std='std_lambda_max',
                ylabel=r'$\lambda_{\max}$',
                title=r'Leading Hessian Eigenvalue $\lambda_{\max}$ vs Epoch')
    fig.tight_layout()
    save(fig, 'fig_d_lambda_max_epoch', output_dir)


# ── Fig E: λ_max at epoch 150 vs ρ ──────────────────────────────────────────
def fig_e(h, output_dir):
    last = h[h['epoch'] == h['epoch'].max()].set_index('config')

    sgd_mu  = last.loc['sgd_baseline', 'mean_lambda_max'] if 'sgd_baseline' in last.index else None
    sgd_sig = last.loc['sgd_baseline', 'std_lambda_max']  if 'sgd_baseline' in last.index else 0.0

    rhos, mus, sigs = [], [], []
    for cfg in SAM_CONFIGS:
        if cfg not in last.index:
            continue
        rhos.append(SAM_RHO[cfg])
        mus.append(last.loc[cfg, 'mean_lambda_max'])
        sigs.append(last.loc[cfg, 'std_lambda_max'] if not np.isnan(last.loc[cfg, 'std_lambda_max']) else 0.0)

    fig, ax = plt.subplots(figsize=(5, 4))

    # SGD as horizontal reference band
    if sgd_mu is not None:
        ax.axhline(sgd_mu, color=COLORS['sgd_baseline'], linewidth=1.5,
                   linestyle='--', label=LABELS['sgd_baseline'])
        ax.axhspan(sgd_mu - sgd_sig, sgd_mu + sgd_sig,
                   color=COLORS['sgd_baseline'], alpha=0.10)

    # SAM configs: scatter + connecting line + error bars
    if rhos:
        ax.plot(rhos, mus, color='#333333', linewidth=1.2, zorder=1)
        for rho, mu, sig, cfg in zip(rhos, mus, sigs, SAM_CONFIGS):
            ax.errorbar(rho, mu, yerr=sig, fmt='o', color=COLORS[cfg],
                        markersize=7, capsize=4, linewidth=1.5,
                        label=LABELS[cfg], zorder=2)

    ax.set_xlabel(r'SAM perturbation radius $\rho$')
    ax.set_ylabel(r'$\lambda_{\max}$ at final epoch')
    ax.set_title(r'Leading Eigenvalue at Convergence vs $\rho$')
    ax.set_xticks(rhos)
    ax.legend(loc='upper right', framealpha=0.9)

    fig.tight_layout()
    save(fig, 'fig_e_lambda_max_vs_rho', output_dir)


# ── Fig F: trace vs epoch ────────────────────────────────────────────────────
def fig_f(h, output_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    _epoch_line(ax, h,
                metric_mean='mean_trace',
                metric_std='std_trace',
                ylabel=r'Hessian Trace',
                title='Hessian Trace vs Epoch')
    fig.tight_layout()
    save(fig, 'fig_f_trace_epoch', output_dir)


# ── Entry point ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--results-dir', type=str, default='./results')
parser.add_argument('--output-dir',  type=str, default=None)
args = parser.parse_args()

OUTPUT_DIR = args.output_dir or os.path.join(args.results_dir, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

h = pd.read_csv(os.path.join(args.results_dir, 'hessian_agg.csv'))

fig_d(h, OUTPUT_DIR)
fig_e(h, OUTPUT_DIR)
fig_f(h, OUTPUT_DIR)

print(f'6 files written to {OUTPUT_DIR}')
