#!/usr/bin/env python3
"""Quick test: 64+112 only (no 224), verify pipeline works."""

import gc, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.symbolic.tensor_operators import TENSOR_OPERATORS, SymbolicKernelBank

OUTPUT_DIR = Path('outputs/imagenet_v3/phase3_v3')
NUM_CLASSES = 1000
N_ENCODINGS = 12
RESOLUTIONS = [64, 112]  # no 224

torch.manual_seed(42); np.random.seed(42)
device = 'cuda'

# Load labels
y_train = np.load(str(OUTPUT_DIR / 'y_train.npy'))
y_val = np.load(str(OUTPUT_DIR / 'y_val.npy'))
n_train, n_val = len(y_train), len(y_val)

bodies = json.load(open(OUTPUT_DIR / 'bodies_sorted.json'))
n_bodies = len(bodies)
n_feats_per_res = n_bodies * N_ENCODINGS
print(f"Bodies: {n_bodies}, feats/res: {n_feats_per_res}")
print(f"Train: {n_train}, Val: {n_val}")
print(f"Resolutions: {RESOLUTIONS}")


def online_mean_std(X_mmap, n_total, n_feats, chunk_size=10000):
    s = np.zeros(n_feats, dtype=np.float64)
    sq = np.zeros(n_feats, dtype=np.float64)
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        c = np.nan_to_num(np.array(X_mmap[start:end], dtype=np.float64))
        s += c.sum(axis=0); sq += (c**2).sum(axis=0)
    mean = s / n_total
    std = np.sqrt(np.maximum(sq / n_total - mean**2, 0.0))
    return mean.astype(np.float32), std.astype(np.float32)


# Concatenate 64+112 per-resolution into selected features
# Use all features (no L1 selection for quick test)
n_feats = n_feats_per_res * len(RESOLUTIONS)
print(f"Total features: {n_feats}")

# Compute mean/std per resolution, then concatenate
all_means, all_stds = [], []
for res in RESOLUTIONS:
    X = np.memmap(str(OUTPUT_DIR / f'X_train_{res}.mmap'), dtype='float32', mode='r',
                  shape=(n_train, n_feats_per_res))
    m, s = online_mean_std(X, n_train, n_feats_per_res)
    all_means.append(m); all_stds.append(s)
    del X

feat_mean = np.concatenate(all_means)
feat_std = np.maximum(np.concatenate(all_stds), 1e-8)
mean_t = torch.tensor(feat_mean, device=device)
std_t = torch.tensor(feat_std, device=device)

# Load mmap refs
train_mmaps = [np.memmap(str(OUTPUT_DIR / f'X_train_{r}.mmap'), dtype='float32', mode='r',
               shape=(n_train, n_feats_per_res)) for r in RESOLUTIONS]
val_mmaps = [np.memmap(str(OUTPUT_DIR / f'X_val_{r}.mmap'), dtype='float32', mode='r',
             shape=(n_val, n_feats_per_res)) for r in RESOLUTIONS]


def load_batch(mmaps, idx):
    """Load batch from multiple mmap files, concatenate columns."""
    parts = [np.array(m[idx]) for m in mmaps]
    return np.concatenate(parts, axis=1)


def train_and_eval(weight_decay, epochs=20):
    model = nn.Linear(n_feats, NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n_train)
        for start in range(0, n_train, 1024):
            end = min(start + 1024, n_train)
            idx = perm[start:end]
            X_b = torch.tensor(load_batch(train_mmaps, idx), dtype=torch.float32, device=device)
            X_b = torch.nan_to_num(X_b)
            X_b = (X_b - mean_t) / std_t
            y_b = torch.tensor(y_train[idx], dtype=torch.long, device=device)
            opt.zero_grad(); crit(model(X_b), y_b).backward(); opt.step()
        sched.step()
        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1}/{epochs}")

    elapsed = time.time() - t0
    model.eval()
    # Eval val
    c1 = c5 = 0
    with torch.no_grad():
        for s in range(0, n_val, 2048):
            e = min(s + 2048, n_val)
            X_b = torch.tensor(load_batch(val_mmaps, slice(s, e)), dtype=torch.float32, device=device)
            X_b = torch.nan_to_num(X_b)
            X_b = (X_b - mean_t) / std_t
            logits = model(X_b)
            y_b = torch.tensor(y_val[s:e], dtype=torch.long, device=device)
            c1 += (logits.argmax(1) == y_b).sum().item()
            _, tk = logits.topk(5, dim=1)
            c5 += (tk == y_b.unsqueeze(1)).any(1).sum().item()
    # Eval train (sample)
    tc = 0
    with torch.no_grad():
        for s in range(0, min(50000, n_train), 2048):
            e = min(s + 2048, min(50000, n_train))
            X_b = torch.tensor(load_batch(train_mmaps, slice(s, e)), dtype=torch.float32, device=device)
            X_b = torch.nan_to_num(X_b)
            X_b = (X_b - mean_t) / std_t
            tc += (model(X_b).argmax(1) == torch.tensor(y_train[s:e], device=device)).sum().item()

    return tc / min(50000, n_train), c1 / n_val, c5 / n_val, elapsed


print(f"\n{'='*60}")
print(f"  64+112 Quick Test ({n_feats} features)")
print(f"{'='*60}")

for wd in [5.0, 10.0, 20.0, 50.0]:
    tr, v1, v5, t = train_and_eval(wd, epochs=20)
    print(f"  wd={wd:<6}  Train={tr*100:6.2f}%  Val-T1={v1*100:6.2f}%  Val-T5={v5*100:6.2f}%  ({t:.0f}s)")
