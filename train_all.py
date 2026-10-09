import os
import sys
import csv
import json
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import ResNet18CIFAR10
from sam import SAM
from data import get_train_loader, get_test_loader, get_cifar10c_loader, CIFAR10C_CORRUPTIONS
from train import train_single_epoch, evaluate

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--seed',     type=int, default=42,       help='random seed')
parser.add_argument('--run-name', type=str, default=None,     help='subdirectory name for outputs (defaults to seed<seed>)')
parser.add_argument('--cifar10c-root', type=str, default='/kaggle/input/cifar10c')
args = parser.parse_args()

RUN_NAME = args.run_name or f'seed{args.seed}'

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = f'/kaggle/working/checkpoints/{RUN_NAME}'
RESULTS_DIR    = f'/kaggle/working/results/{RUN_NAME}'
CIFAR10C_ROOT  = args.cifar10c_root

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,    exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS            = 150
BATCH_SIZE        = 128
LR                = 0.1
MOMENTUM          = 0.9
WEIGHT_DECAY      = 5e-4
CHECKPOINT_EPOCHS = {5, 30, 60, 90, 120, 150}

# ── Experiment configs ────────────────────────────────────────────────────────
CONFIGS = [
    {'name': 'sgd_baseline', 'use_sam': False, 'rho': None},
    {'name': 'sam_rho005',   'use_sam': True,  'rho': 0.05},
    {'name': 'sam_rho010',   'use_sam': True,  'rho': 0.10},
    {'name': 'sam_rho020',   'use_sam': True,  'rho': 0.20},
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}  |  Run: {RUN_NAME}  |  Seed: {args.seed}')


def build_optimizer(model, cfg):
    sgd_kwargs = dict(lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    if cfg['use_sam']:
        return SAM(model.parameters(), torch.optim.SGD, rho=cfg['rho'], **sgd_kwargs)
    return torch.optim.SGD(model.parameters(), **sgd_kwargs)


def save_checkpoint(path, epoch, model, optimizer, scheduler, cfg, metrics):
    ckpt = {
        'epoch':                epoch,
        'config':               cfg,
        'model_state_dict':     model.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics':              metrics,
        'optimizer_state_dict': optimizer.state_dict(),
    }
    # SAM wraps a base SGD; its momentum buffers live in base_optimizer.state
    if cfg['use_sam']:
        ckpt['base_optimizer_state_dict'] = optimizer.base_optimizer.state_dict()
    torch.save(ckpt, path)


def train_config(cfg):
    print(f"\n{'='*60}\nConfig: {cfg['name']}\n{'='*60}")

    train_loader = get_train_loader(batch_size=BATCH_SIZE)
    test_loader  = get_test_loader()

    model     = ResNet18CIFAR10().to(device)
    optimizer = build_optimizer(model, cfg)
    criterion = nn.CrossEntropyLoss()
    # CosineAnnealingLR reads param_groups from optimizer; SAM delegates these
    # to base_optimizer, so the scheduler correctly updates base SGD's lr.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0)

    cfg_ckpt_dir = os.path.join(CHECKPOINT_DIR, cfg['name'])
    os.makedirs(cfg_ckpt_dir, exist_ok=True)

    history = []

    for epoch in tqdm(range(1, EPOCHS + 1), desc=cfg['name']):
        t0 = time.time()
        train_loss, train_acc = train_single_epoch(model, train_loader, optimizer, criterion, device)
        test_loss,  test_acc  = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        row = {
            'epoch':      epoch,
            'train_loss': round(train_loss, 4),
            'train_acc':  round(train_acc,  2),
            'test_loss':  round(test_loss,  4),
            'test_acc':   round(test_acc,   2),
            'time_s':     round(elapsed,    1),
        }
        history.append(row)

        if epoch in CHECKPOINT_EPOCHS:
            ckpt_path = os.path.join(cfg_ckpt_dir, f'epoch_{epoch:03d}.pt')
            save_checkpoint(ckpt_path, epoch, model, optimizer, scheduler, cfg, row)
            tqdm.write(
                f'  [epoch {epoch:3d}] train={train_acc:.1f}%  test={test_acc:.1f}%'
                f'  lr={scheduler.get_last_lr()[0]:.5f}  ({elapsed:.0f}s)  → ckpt saved'
            )

    history_csv = os.path.join(RESULTS_DIR, f"{cfg['name']}_history.csv")
    with open(history_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    return model, history


def eval_cifar10c(model, cfg_name):
    print(f'\nCIFAR-10-C evaluation: {cfg_name}')
    criterion = nn.CrossEntropyLoss()
    results   = []

    for corruption in tqdm(CIFAR10C_CORRUPTIONS, desc='corruptions'):
        for severity in range(1, 6):
            try:
                loader = get_cifar10c_loader(corruption, severity, root_dir=CIFAR10C_ROOT)
                _, acc = evaluate(model, loader, criterion, device)
            except FileNotFoundError as e:
                tqdm.write(f'  WARNING: {e}')
                acc = float('nan')

            results.append({
                'config':     cfg_name,
                'corruption': corruption,
                'severity':   severity,
                'accuracy':   round(acc, 2),
            })

    return results


# ── Run all configs ───────────────────────────────────────────────────────────
all_history = {}
all_c10c    = []

for cfg in CONFIGS:
    model, history = train_config(cfg)
    all_history[cfg['name']] = history

    c10c = eval_cifar10c(model, cfg['name'])
    all_c10c.extend(c10c)

    per_cfg_csv = os.path.join(RESULTS_DIR, f"{cfg['name']}_cifar10c.csv")
    with open(per_cfg_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=c10c[0].keys())
        writer.writeheader()
        writer.writerows(c10c)

# Combined CIFAR-10-C table (all configs in one file for easy plotting)
combined_csv = os.path.join(RESULTS_DIR, 'cifar10c_all.csv')
with open(combined_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=all_c10c[0].keys())
    writer.writeheader()
    writer.writerows(all_c10c)

# Full results as JSON (training curves + C10-C)
with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
    json.dump({'seed': args.seed, 'run_name': RUN_NAME, 'training': all_history, 'cifar10c': all_c10c}, f, indent=2)

print(f'\nAll done. Artifacts in {RESULTS_DIR} and checkpoints in {CHECKPOINT_DIR}')
