import os
import sys
import csv
import argparse

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import ResNet18CIFAR10
from data import get_test_loader
from hessian import compute_all

parser = argparse.ArgumentParser()
parser.add_argument('--seed',             type=int, default=42)
parser.add_argument('--run-name',         type=str, default=None)
parser.add_argument('--checkpoint-dir',   type=str, default=None)
parser.add_argument('--output-dir',       type=str, default=None)
parser.add_argument('--hessian-iters',    type=int, default=100)
parser.add_argument('--hessian-samples',  type=int, default=30)
args = parser.parse_args()

RUN_NAME       = args.run_name or f'seed{args.seed}'
CHECKPOINT_DIR = args.checkpoint_dir or f'/kaggle/working/checkpoints/{RUN_NAME}'
OUTPUT_DIR     = args.output_dir     or f'/kaggle/working/results/{RUN_NAME}'

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}  |  Run: {RUN_NAME}  |  Seed: {args.seed}')

# Fixed held-out batch - test set, shuffle=False → same 128 samples every call
held_out_loader = get_test_loader(batch_size=128)
criterion = nn.CrossEntropyLoss()

CONFIGS           = ['sgd_baseline', 'sam_rho005', 'sam_rho010', 'sam_rho020']
CHECKPOINT_EPOCHS = [5, 30, 60, 90, 120, 150]

rows = []

for cfg_name in CONFIGS:
    cfg_dir = os.path.join(CHECKPOINT_DIR, cfg_name)
    if not os.path.isdir(cfg_dir):
        print(f'WARNING: {cfg_dir} not found, skipping')
        continue

    for epoch in CHECKPOINT_EPOCHS:
        ckpt_path = os.path.join(cfg_dir, f'epoch_{epoch:03d}.pt')
        if not os.path.isfile(ckpt_path):
            print(f'WARNING: {ckpt_path} not found, skipping')
            continue

        print(f'[{cfg_name}  epoch {epoch:3d}] loading checkpoint...')
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model = ResNet18CIFAR10().to(device)
        model.load_state_dict(ckpt['model_state_dict'])

        lmax, tr, _ = compute_all(
            model, criterion, held_out_loader, device,
            num_iters=args.hessian_iters,
            num_samples=args.hessian_samples,
        )
        print(f'  lambda_max={lmax:.4f}  trace={tr:.4f}')

        rows.append({
            'seed':       args.seed,
            'config':     cfg_name,
            'epoch':      epoch,
            'lambda_max': lmax,
            'trace':      tr,
        })

output_csv = os.path.join(OUTPUT_DIR, 'hessian_results.csv')
with open(output_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['seed', 'config', 'epoch', 'lambda_max', 'trace'])
    writer.writeheader()
    writer.writerows(rows)

print(f'\nDone. {len(rows)} rows written to {output_csv}')
