"""
Aggregate results across seeds into three CSVs for plotting.

Expected directory layout (--results-dir):
  <results-dir>/
    seed42/
      sgd_baseline_history.csv
      sam_rho005_history.csv
      ...
      cifar10c_all.csv
      hessian_results.csv
    seed123/
      ...

Usage:
  pip install pandas
  python aggregate_results.py --results-dir ./results
"""
import os
import sys
import glob
import argparse

import pandas as pd

CONFIGS = ['sgd_baseline', 'sam_rho005', 'sam_rho010', 'sam_rho020']

parser = argparse.ArgumentParser()
parser.add_argument('--results-dir', type=str, default='./results',
                    help='directory containing per-seed subdirectories')
parser.add_argument('--output-dir',  type=str, default=None,
                    help='where to write aggregated CSVs (default: same as --results-dir)')
args = parser.parse_args()

OUTPUT_DIR = args.output_dir or args.results_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

seed_dirs = sorted(
    d for d in glob.glob(os.path.join(args.results_dir, 'seed*'))
    if os.path.isdir(d)
)
if not seed_dirs:
    sys.exit(f'ERROR: no subdirectories found in {args.results_dir}')

print(f'Seeds found: {[os.path.basename(d) for d in seed_dirs]}')


# ── Training history ──────────────────────────────────────────────────────────
# Columns: epoch, train_loss, train_acc, test_loss, test_acc, time_s
frames = []
for seed_dir in seed_dirs:
    for cfg in CONFIGS:
        path = os.path.join(seed_dir, f'{cfg}_history.csv')
        if not os.path.isfile(path):
            print(f'WARNING: {path} not found, skipping')
            continue
        df = pd.read_csv(path)
        df['config'] = cfg
        frames.append(df)

if frames:
    tr = pd.concat(frames, ignore_index=True)
    agg = (
        tr.groupby(['config', 'epoch'])
          .agg(
              mean_train_acc=('train_acc', 'mean'),
              std_train_acc =('train_acc', 'std'),
              mean_test_acc =('test_acc',  'mean'),
              std_test_acc  =('test_acc',  'std'),
          )
          .reset_index()
    )
    out = os.path.join(OUTPUT_DIR, 'training_agg.csv')
    agg.to_csv(out, index=False)
    print(f'training_agg.csv  - {len(agg)} rows')


# ── CIFAR-10-C ────────────────────────────────────────────────────────────────
# Columns: config, corruption, severity, accuracy
frames = []
for seed_dir in seed_dirs:
    path = os.path.join(seed_dir, 'cifar10c_all.csv')
    if not os.path.isfile(path):
        print(f'WARNING: {path} not found, skipping')
        continue
    df = pd.read_csv(path)
    df['seed'] = os.path.basename(seed_dir)
    frames.append(df)

if frames:
    c = pd.concat(frames, ignore_index=True)

    # cifar10c_agg.csv - unchanged behaviour
    agg = (
        c.groupby(['config', 'corruption', 'severity'])
         .agg(
             mean_acc=('accuracy', 'mean'),
             std_acc =('accuracy', 'std'),
         )
         .reset_index()
    )
    out = os.path.join(OUTPUT_DIR, 'cifar10c_agg.csv')
    agg.to_csv(out, index=False)
    print(f'cifar10c_agg.csv       - {len(agg)} rows')

    # cifar10c_summary.csv - per seed+config mean over all corruptions/severities
    per_seed = (
        c.groupby(['seed', 'config'])['accuracy']
         .mean()
         .reset_index()
         .rename(columns={'accuracy': 'acc'})
    )
    summary = (
        per_seed.groupby('config')
                .agg(
                    mean_acc=('acc', 'mean'),
                    std_acc =('acc', 'std'),
                )
                .reset_index()
    )
    out = os.path.join(OUTPUT_DIR, 'cifar10c_summary.csv')
    summary.to_csv(out, index=False)
    print(f'cifar10c_summary.csv   - {len(summary)} rows')

    # cifar10c_by_corruption.csv - per seed+config+corruption mean over severities
    per_seed_corr = (
        c.groupby(['seed', 'config', 'corruption'])['accuracy']
         .mean()
         .reset_index()
         .rename(columns={'accuracy': 'acc'})
    )
    by_corr = (
        per_seed_corr.groupby(['config', 'corruption'])
                     .agg(
                         mean_acc=('acc', 'mean'),
                         std_acc =('acc', 'std'),
                     )
                     .reset_index()
    )
    out = os.path.join(OUTPUT_DIR, 'cifar10c_by_corruption.csv')
    by_corr.to_csv(out, index=False)
    print(f'cifar10c_by_corruption.csv - {len(by_corr)} rows')


# ── Hessian ───────────────────────────────────────────────────────────────────
# Columns: seed, config, epoch, lambda_max, trace
frames = []
for seed_dir in seed_dirs:
    path = os.path.join(seed_dir, 'hessian_results.csv')
    if not os.path.isfile(path):
        print(f'WARNING: {path} not found, skipping')
        continue
    df = pd.read_csv(path)
    if 'seed' not in df.columns:
        df['seed'] = os.path.basename(seed_dir)
    frames.append(df)

if frames:
    h = pd.concat(frames, ignore_index=True)

    # TPR computed per row before aggregation
    h['tpr'] = h['trace'] / h['lambda_max']

    agg = (
        h.groupby(['config', 'epoch'])
         .agg(
             mean_lambda_max=('lambda_max', 'mean'),
             std_lambda_max =('lambda_max', 'std'),
             mean_trace     =('trace',      'mean'),
             std_trace      =('trace',      'std'),
             mean_tpr       =('tpr',        'mean'),
             std_tpr        =('tpr',        'std'),
         )
         .reset_index()
    )
    out = os.path.join(OUTPUT_DIR, 'hessian_agg.csv')
    agg.to_csv(out, index=False)
    print(f'hessian_agg.csv        - {len(agg)} rows')

    # hessian_final_summary.csv - last epoch only, grouped by config
    max_epoch = h['epoch'].max()
    h_final = h[h['epoch'] == max_epoch]
    final_summary = (
        h_final.groupby('config')
               .agg(
                   mean_lambda_max=('lambda_max', 'mean'),
                   std_lambda_max =('lambda_max', 'std'),
                   mean_trace     =('trace',      'mean'),
                   std_trace      =('trace',      'std'),
                   mean_tpr       =('tpr',        'mean'),
                   std_tpr        =('tpr',        'std'),
               )
               .reset_index()
    )
    out = os.path.join(OUTPUT_DIR, 'hessian_final_summary.csv')
    final_summary.to_csv(out, index=False)
    print(f'hessian_final_summary.csv - {len(final_summary)} rows')


print(f'\nDone. Aggregated CSVs in {OUTPUT_DIR}')
