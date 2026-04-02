#!/usr/bin/env python3
"""Step 3 only: sweep weight_decay and dropout on existing feature matrices."""

import json, time, sys, os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = 'cuda'
OUTPUT_DIR = Path('outputs/imagenet_v2/phase3_v2')
NUM_CLASSES = 1000

torch.manual_seed(42)
np.random.seed(42)

# ── Load existing features ──────────────────────────────────
formula_strs = json.load(open(OUTPUT_DIR / 'formula_list_sorted.json'))
n_feats = len(formula_strs)
y_train = np.load(str(OUTPUT_DIR / 'y_train.npy'))
y_val = np.load(str(OUTPUT_DIR / 'y_val.npy'))
n_train, n_val = len(y_train), len(y_val)

X_train = np.memmap(str(OUTPUT_DIR / 'X_train.mmap'), dtype='float32', mode='r', shape=(n_train, n_feats))
X_val = np.memmap(str(OUTPUT_DIR / 'X_val.mmap'), dtype='float32', mode='r', shape=(n_val, n_feats))
print(f"Loaded features: train ({n_train}, {n_feats}), val ({n_val}, {n_feats})")

# ── Standardization ─────────────────────────────────────────
feat_mean = np.load(str(OUTPUT_DIR / 'feat_mean.npy'))
feat_std = np.load(str(OUTPUT_DIR / 'feat_std.npy'))
feat_std = np.maximum(feat_std, 1e-8)
mean_t = torch.tensor(feat_mean, dtype=torch.float32, device=DEVICE)
std_t = torch.tensor(feat_std, dtype=torch.float32, device=DEVICE)


def train_and_eval(model, weight_decay, epochs=20, lr=1e-3, batch_size=1024):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]
            X_b = torch.tensor(np.array(X_train[idx]), dtype=torch.float32, device=DEVICE)
            X_b = torch.nan_to_num(X_b)
            X_b = (X_b - mean_t) / std_t
            y_b = torch.tensor(y_train[idx], dtype=torch.long, device=DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
    elapsed = time.time() - t0

    # Evaluate
    model.eval()
    def eval_split(X_mmap, y):
        correct1 = correct5 = 0
        n = len(y)
        with torch.no_grad():
            for start in range(0, n, 2048):
                end = min(start + 2048, n)
                X_b = torch.tensor(np.array(X_mmap[start:end]), dtype=torch.float32, device=DEVICE)
                X_b = torch.nan_to_num(X_b)
                X_b = (X_b - mean_t) / std_t
                logits = model(X_b)
                y_b = torch.tensor(y[start:end], dtype=torch.long, device=DEVICE)
                correct1 += (logits.argmax(1) == y_b).sum().item()
                _, tk = logits.topk(min(5, NUM_CLASSES), dim=1)
                correct5 += (tk == y_b.unsqueeze(1)).any(1).sum().item()
        return correct1 / n, correct5 / n

    train_top1, _ = eval_split(X_train, y_train)
    val_top1, val_top5 = eval_split(X_val, y_val)
    return train_top1, val_top1, val_top5, elapsed


# ══════════════════════════════════════════════════════════════
# Sweep A: Weight Decay
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  SWEEP A: Weight Decay (nn.Linear, no dropout)")
print(f"{'='*65}")

wd_values = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
results_a = {}

for wd in wd_values:
    model = nn.Linear(n_feats, NUM_CLASSES).to(DEVICE)
    train1, val1, val5, t = train_and_eval(model, weight_decay=wd)
    results_a[wd] = (train1, val1, val5, t)
    print(f"  wd={wd:<6}  Train={train1*100:6.2f}%  Val-T1={val1*100:6.2f}%  Val-T5={val5*100:6.2f}%  ({t:.1f}s)")

best_wd_a = max(results_a, key=lambda k: results_a[k][1])
print(f"\n  BEST: wd={best_wd_a}  Val-T1={results_a[best_wd_a][1]*100:.2f}%  Val-T5={results_a[best_wd_a][2]*100:.2f}%")


# ══════════════════════════════════════════════════════════════
# Sweep B: Dropout
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  SWEEP B: Dropout + best wd from A (wd={best_wd_a})")
print(f"{'='*65}")

dropout_values = [0.3, 0.5, 0.7, 0.9]
results_b = {}

for dp in dropout_values:
    model = nn.Sequential(
        nn.Dropout(p=dp),
        nn.Linear(n_feats, NUM_CLASSES),
    ).to(DEVICE)
    train1, val1, val5, t = train_and_eval(model, weight_decay=best_wd_a)
    results_b[dp] = (train1, val1, val5, t)
    print(f"  dropout={dp:<4}  Train={train1*100:6.2f}%  Val-T1={val1*100:6.2f}%  Val-T5={val5*100:6.2f}%  ({t:.1f}s)")

best_dp = max(results_b, key=lambda k: results_b[k][1])
print(f"\n  BEST: dropout={best_dp}  Val-T1={results_b[best_dp][1]*100:.2f}%  Val-T5={results_b[best_dp][2]*100:.2f}%")


# ══════════════════════════════════════════════════════════════
# Sweep C: Dropout × Weight Decay grid (top combos)
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  SWEEP C: Dropout × Weight Decay grid")
print(f"{'='*65}")

grid_wd = [0.5, 1.0, 2.0, 5.0]
grid_dp = [0.5, 0.7, 0.9]
results_c = {}

for dp in grid_dp:
    for wd in grid_wd:
        model = nn.Sequential(
            nn.Dropout(p=dp),
            nn.Linear(n_feats, NUM_CLASSES),
        ).to(DEVICE)
        train1, val1, val5, t = train_and_eval(model, weight_decay=wd)
        results_c[(dp, wd)] = (train1, val1, val5, t)
        print(f"  dp={dp} wd={wd:<4}  Train={train1*100:6.2f}%  Val-T1={val1*100:6.2f}%  Val-T5={val5*100:6.2f}%  ({t:.1f}s)")

best_c = max(results_c, key=lambda k: results_c[k][1])
print(f"\n  BEST: dropout={best_c[0]} wd={best_c[1]}  Val-T1={results_c[best_c][1]*100:.2f}%  Val-T5={results_c[best_c][2]*100:.2f}%")


# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  SUMMARY")
print(f"{'='*65}")
print(f"  Previous best (wd=0.1, no dropout):  Val-T1=13.89%  Val-T5=29.27%")
print(f"  Sweep A best (wd={best_wd_a}):            Val-T1={results_a[best_wd_a][1]*100:.2f}%  Val-T5={results_a[best_wd_a][2]*100:.2f}%")
print(f"  Sweep B best (dp={best_dp}, wd={best_wd_a}):  Val-T1={results_b[best_dp][1]*100:.2f}%  Val-T5={results_b[best_dp][2]*100:.2f}%")
print(f"  Sweep C best (dp={best_c[0]}, wd={best_c[1]}): Val-T1={results_c[best_c][1]*100:.2f}%  Val-T5={results_c[best_c][2]*100:.2f}%")

# Save
all_results = {
    'sweep_a': {str(k): {'train': v[0], 'val_top1': v[1], 'val_top5': v[2]} for k, v in results_a.items()},
    'sweep_b': {str(k): {'train': v[0], 'val_top1': v[1], 'val_top5': v[2]} for k, v in results_b.items()},
    'sweep_c': {f"dp={k[0]}_wd={k[1]}": {'train': v[0], 'val_top1': v[1], 'val_top5': v[2]} for k, v in results_c.items()},
}
with open(OUTPUT_DIR / 'sweep_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {OUTPUT_DIR / 'sweep_results.json'}")
