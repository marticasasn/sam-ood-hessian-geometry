"""
Figures for the Trace-to-Peak Ratio (TPR) analysis.

Fig G - TPR = Tr(H) / lambda_max vs epoch, one line per config
Fig H - TPR at epoch 150 vs SAM rho
Fig I - Final TPR vs mean CIFAR-10-C accuracy

Usage:
  python plot_tpr.py --results-dir ./results --output-dir ./results/figures
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

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
        fig.savefig(
            os.path.join(output_dir, f'{stem}.{ext}'),
            bbox_inches='tight',
            dpi=150,
        )
    plt.close(fig)


def fig_g_tpr_epoch(h, output_dir):
    """TPR vs epoch."""
    fig, ax = plt.subplots(figsize=(6, 4))

    for cfg in CONFIGS:
        sub = h[h['config'] == cfg].sort_values('epoch')
        if sub.empty:
            continue

        c = COLORS[cfg]
        mu = sub['mean_tpr']
        sig = sub['std_tpr'].fillna(0)

        ax.plot(sub['epoch'], mu, color=c, label=LABELS[cfg], linewidth=1.8)
        ax.fill_between(sub['epoch'], mu - sig, mu + sig, alpha=0.15, color=c)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'TPR, $\mathrm{Tr}(H)/\lambda_{\max}$')
    ax.set_title('Hessian Trace-to-Peak Ratio vs Epoch')
    ax.set_xlim(h['epoch'].min(), h['epoch'].max())
    ax.legend(loc='upper right', framealpha=0.9)

    fig.tight_layout()
    save(fig, 'fig_g_tpr_epoch', output_dir)


def fig_h_tpr_vs_rho(h, output_dir):
    """TPR at final epoch vs SAM rho."""
    last = h[h['epoch'] == h['epoch'].max()].set_index('config')

    sgd_mu = last.loc['sgd_baseline', 'mean_tpr'] if 'sgd_baseline' in last.index else None
    sgd_sig = last.loc['sgd_baseline', 'std_tpr'] if 'sgd_baseline' in last.index else 0.0

    rhos, mus, sigs, cfgs = [], [], [], []
    for cfg in SAM_CONFIGS:
        if cfg not in last.index:
            continue
        rhos.append(SAM_RHO[cfg])
        mus.append(last.loc[cfg, 'mean_tpr'])
        sigs.append(last.loc[cfg, 'std_tpr'] if not np.isnan(last.loc[cfg, 'std_tpr']) else 0.0)
        cfgs.append(cfg)

    fig, ax = plt.subplots(figsize=(5, 4))

    if sgd_mu is not None:
        ax.axhline(
            sgd_mu,
            color=COLORS['sgd_baseline'],
            linewidth=1.5,
            linestyle='--',
            label=LABELS['sgd_baseline'],
        )
        ax.axhspan(
            sgd_mu - sgd_sig,
            sgd_mu + sgd_sig,
            color=COLORS['sgd_baseline'],
            alpha=0.10,
        )

    if rhos:
        ax.plot(rhos, mus, color='#333333', linewidth=1.2, zorder=1)
        for rho, mu, sig, cfg in zip(rhos, mus, sigs, cfgs):
            ax.errorbar(
                rho,
                mu,
                yerr=sig,
                fmt='o',
                color=COLORS[cfg],
                markersize=7,
                capsize=4,
                linewidth=1.5,
                label=LABELS[cfg],
                zorder=2,
            )

    ax.set_xlabel(r'SAM perturbation radius $\rho$')
    ax.set_ylabel(r'TPR at final epoch')
    ax.set_title(r'Trace-to-Peak Ratio at Convergence vs $\rho$')
    ax.set_xticks(rhos)
    ax.legend(loc='upper left', framealpha=0.9)

    fig.tight_layout()
    save(fig, 'fig_h_tpr_vs_rho', output_dir)


def fig_i_tpr_vs_accuracy(h_final, c10c_summary, output_dir):
    """Final TPR vs CIFAR-10-C accuracy."""
    h = h_final.set_index('config')
    c = c10c_summary.set_index('config')

    rows = []
    for cfg in CONFIGS:
        if cfg not in h.index or cfg not in c.index:
            continue
        rows.append({
            'config': cfg,
            'tpr': h.loc[cfg, 'mean_tpr'],
            'tpr_std': h.loc[cfg, 'std_tpr'] if not np.isnan(h.loc[cfg, 'std_tpr']) else 0.0,
            'acc': c.loc[cfg, 'mean_acc'],
            'acc_std': c.loc[cfg, 'std_acc'] if not np.isnan(c.loc[cfg, 'std_acc']) else 0.0,
        })

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(5, 4))

    for _, row in df.iterrows():
        cfg = row['config']
        ax.errorbar(
            row['tpr'],
            row['acc'],
            xerr=row['tpr_std'],
            yerr=row['acc_std'],
            fmt='o',
            color=COLORS[cfg],
            markersize=7,
            capsize=3,
            linewidth=1.2,
            label=LABELS[cfg],
        )

    ax.set_xlabel(r'Final TPR, $\mathrm{Tr}(H)/\lambda_{\max}$')
    ax.set_ylabel('Mean CIFAR-10-C accuracy (%)')
    ax.set_title('TPR vs Corruption Robustness')
    ax.legend(loc='best', framealpha=0.9)

    fig.tight_layout()
    save(fig, 'fig_i_tpr_vs_cifar10c', output_dir)


parser = argparse.ArgumentParser()
parser.add_argument('--results-dir', type=str, default='./results')
parser.add_argument('--output-dir',  type=str, default=None)
args = parser.parse_args()

OUTPUT_DIR = args.output_dir or os.path.join(args.results_dir, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

h = pd.read_csv(os.path.join(args.results_dir, 'hessian_agg.csv'))
h_final = pd.read_csv(os.path.join(args.results_dir, 'hessian_final_summary.csv'))
c10c_summary = pd.read_csv(os.path.join(args.results_dir, 'cifar10c_summary.csv'))

fig_g_tpr_epoch(h, OUTPUT_DIR)
fig_h_tpr_vs_rho(h, OUTPUT_DIR)
fig_i_tpr_vs_accuracy(h_final, c10c_summary, OUTPUT_DIR)

print(f'6 files written to {OUTPUT_DIR}')