#!/usr/bin/env python3
"""
Batch 1 补全: Feature Interactions + L1 Selection + Train.

1. Load trained model from Batch 1 to find top-300 important features
2. Compute pairwise products → ~45K interaction features
3. Concatenate base 30K + interactions
4. L1 selection → ~50K effective features
5. Train nn.Linear with wd sweep
"""

import json, os, sys, time, itertools
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = Path('outputs/imagenet_v3/phase3_v3')
NUM_CLASSES = 1000

torch.manual_seed(42); np.random.seed(42)
device = 'cuda'

# ── Load existing selected features ──────────────────────
n_train, n_val, n_selected = 200000, 50000, 30000

X_train = np.memmap(str(OUTPUT_DIR / 'X_train_selected.mmap'), dtype='float32', mode='r',
                    shape=(n_train, n_selected))
X_val = np.memmap(str(OUTPUT_DIR / 'X_val_selected.mmap'), dtype='float32', mode='r',
                  shape=(n_val, n_selected))
y_train = np.load(str(OUTPUT_DIR / 'y_train.npy'))
y_val = np.load(str(OUTPUT_DIR / 'y_val.npy'))

sel_mean = np.load(str(OUTPUT_DIR / 'feat_mean_selected.npy'))
sel_std = np.load(str(OUTPUT_DIR / 'feat_std_selected.npy'))

print(f"Loaded: train ({n_train}, {n_selected}), val ({n_val}, {n_selected})")

# ── Step 1: Train model to find top-K important features ─
print(f"\n{'='*60}")
print(f"  Step 1: Find top-300 features by importance")
print(f"{'='*60}")

mean_t = torch.tensor(sel_mean, dtype=torch.float32, device=device)
std_t = torch.tensor(sel_std, dtype=torch.float32, device=device)

model = nn.Linear(n_selected, NUM_CLASSES).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=20.0)
crit = nn.CrossEntropyLoss()
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)

for epoch in range(15):
    model.train()
    perm = np.random.permutation(n_train)
    for start in range(0, n_train, 1024):
        end = min(start + 1024, n_train)
        idx = perm[start:end]
        X_b = torch.tensor(np.array(X_train[idx]), dtype=torch.float32, device=device)
        X_b = torch.nan_to_num(X_b); X_b = (X_b - mean_t) / std_t
        y_b = torch.tensor(y_train[idx], dtype=torch.long, device=device)
        opt.zero_grad(); crit(model(X_b), y_b).backward(); opt.step()
    sched.step()

importance = model.weight.abs().sum(dim=0).detach().cpu().numpy()
top_k = 300
top_indices = np.argsort(importance)[::-1][:top_k]
print(f"  Top {top_k} features selected (importance range: "
      f"[{importance[top_indices[-1]]:.2f}, {importance[top_indices[0]]:.2f}])")

# ── Step 2: Compute pairwise interactions ─────────────────
print(f"\n{'='*60}")
print(f"  Step 2: Compute pairwise interactions")
print(f"{'='*60}")

pairs = list(itertools.combinations(range(top_k), 2))
n_interactions = len(pairs)
print(f"  {top_k} choose 2 = {n_interactions} interaction features")

# Compute interactions in chunks to save memory
interact_train_path = str(OUTPUT_DIR / 'X_train_interact.mmap')
interact_val_path = str(OUTPUT_DIR / 'X_val_interact.mmap')

for split, X_mm, n_total, path in [
    ('train', X_train, n_train, interact_train_path),
    ('val', X_val, n_val, interact_val_path),
]:
    mmap_out = np.memmap(path, dtype='float32', mode='w+', shape=(n_total, n_interactions))
    chunk_size = 10000
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        # Load top-K columns
        chunk = np.array(X_mm[start:end])[:, top_indices]  # [chunk, 300]
        # Standardize
        chunk_std = (chunk - sel_mean[top_indices]) / sel_std[top_indices]
        # Compute pairwise products
        for p_idx, (i, j) in enumerate(pairs):
            mmap_out[start:end, p_idx] = chunk_std[:, i] * chunk_std[:, j]
    mmap_out.flush()
    print(f"  {split}: ({n_total}, {n_interactions})")

# ── Step 3: Concatenate base + interactions ───────────────
print(f"\n{'='*60}")
print(f"  Step 3: Build combined feature matrix")
print(f"{'='*60}")

n_combined = n_selected + n_interactions
print(f"  Base: {n_selected} + Interactions: {n_interactions} = {n_combined}")

# Compute mean/std for interactions
X_interact_train = np.memmap(interact_train_path, dtype='float32', mode='r',
                             shape=(n_train, n_interactions))
int_sum = np.zeros(n_interactions, dtype=np.float64)
int_sq = np.zeros(n_interactions, dtype=np.float64)
for start in range(0, n_train, 10000):
    end = min(start + 10000, n_train)
    c = np.nan_to_num(np.array(X_interact_train[start:end], dtype=np.float64))
    int_sum += c.sum(axis=0); int_sq += (c**2).sum(axis=0)
int_mean = (int_sum / n_train).astype(np.float32)
int_var = int_sq / n_train - (int_sum / n_train)**2
int_std = np.maximum(np.sqrt(np.maximum(int_var, 0.0)).astype(np.float32), 1e-8)

# Combined mean/std
combined_mean = np.concatenate([sel_mean, int_mean])
combined_std = np.concatenate([sel_std, int_std])

# ── Step 4: L1 feature selection on combined ──────────────
print(f"\n{'='*60}")
print(f"  Step 4: L1 selection on {n_combined} features")
print(f"{'='*60}")

mean_tc = torch.tensor(combined_mean, dtype=torch.float32, device=device)
std_tc = torch.tensor(combined_std, dtype=torch.float32, device=device)

X_interact_val = np.memmap(interact_val_path, dtype='float32', mode='r',
                           shape=(n_val, n_interactions))

model_l1 = nn.Linear(n_combined, NUM_CLASSES).to(device)
opt_l1 = torch.optim.AdamW(model_l1.parameters(), lr=1e-3, weight_decay=30.0)
sched_l1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_l1, T_max=15)

for epoch in range(15):
    model_l1.train()
    perm = np.random.permutation(n_train)
    for start in range(0, n_train, 1024):
        end = min(start + 1024, n_train)
        idx = perm[start:end]
        base = np.array(X_train[idx])
        interact = np.array(X_interact_train[idx])
        combined = np.concatenate([base, interact], axis=1)
        X_b = torch.tensor(combined, dtype=torch.float32, device=device)
        X_b = torch.nan_to_num(X_b); X_b = (X_b - mean_tc) / std_tc
        y_b = torch.tensor(y_train[idx], dtype=torch.long, device=device)
        opt_l1.zero_grad(); crit(model_l1(X_b), y_b).backward(); opt_l1.step()
    sched_l1.step()

imp_combined = model_l1.weight.abs().sum(dim=0).detach().cpu().numpy()
target = 50000
thresh = np.sort(imp_combined)[::-1][min(target, len(imp_combined) - 1)]
selected_mask = imp_combined >= thresh
n_final = selected_mask.sum()
selected_final = np.where(selected_mask)[0]
print(f"  Selected {n_final} features from {n_combined}")

# Count how many from base vs interactions
n_base_kept = (selected_final < n_selected).sum()
n_int_kept = (selected_final >= n_selected).sum()
print(f"  Base: {n_base_kept}, Interactions: {n_int_kept}")

# ── Step 5: Train final classifier ───────────────────────
print(f"\n{'='*60}")
print(f"  Step 5: Train Classifier ({n_final} features)")
print(f"{'='*60}")

final_mean = combined_mean[selected_final]
final_std = combined_std[selected_final]
mean_tf = torch.tensor(final_mean, dtype=torch.float32, device=device)
std_tf = torch.tensor(final_std, dtype=torch.float32, device=device)

# Helper to load combined+selected batch
base_sel = selected_final[selected_final < n_selected]
int_sel = selected_final[selected_final >= n_selected] - n_selected

def load_selected_batch(X_base, X_int, idx):
    base = np.array(X_base[idx])[:, base_sel] if len(base_sel) > 0 else np.empty((len(idx) if isinstance(idx, np.ndarray) else idx.stop - idx.start, 0), dtype=np.float32)
    interact = np.array(X_int[idx])[:, int_sel] if len(int_sel) > 0 else np.empty((base.shape[0], 0), dtype=np.float32)
    return np.concatenate([base, interact], axis=1)


def train_and_eval(weight_decay, epochs=30):
    model = nn.Linear(n_final, NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n_train)
        for start in range(0, n_train, 1024):
            end = min(start + 1024, n_train)
            idx = perm[start:end]
            batch = load_selected_batch(X_train, X_interact_train, idx)
            X_b = torch.tensor(batch, dtype=torch.float32, device=device)
            X_b = torch.nan_to_num(X_b); X_b = (X_b - mean_tf) / std_tf
            y_b = torch.tensor(y_train[idx], dtype=torch.long, device=device)
            opt.zero_grad(); crit(model(X_b), y_b).backward(); opt.step()
        sched.step()
        if (epoch + 1) % 10 == 0: print(f"      epoch {epoch+1}/{epochs}")

    elapsed = time.time() - t0
    model.eval()
    c1 = c5 = 0
    with torch.no_grad():
        for s in range(0, n_val, 2048):
            e = min(s + 2048, n_val)
            batch = load_selected_batch(X_val, X_interact_val, slice(s, e))
            X_b = torch.tensor(batch, dtype=torch.float32, device=device)
            X_b = torch.nan_to_num(X_b); X_b = (X_b - mean_tf) / std_tf
            y_b = torch.tensor(y_val[s:e], dtype=torch.long, device=device)
            logits = model(X_b)
            c1 += (logits.argmax(1) == y_b).sum().item()
            _, tk = logits.topk(5, dim=1)
            c5 += (tk == y_b.unsqueeze(1)).any(1).sum().item()
    return c1 / n_val, c5 / n_val, elapsed


results = {}
best_val = 0
for wd in [10.0, 20.0, 50.0]:
    v1, v5, t = train_and_eval(wd, epochs=30)
    results[f'wd={wd}'] = (v1, v5, t)
    print(f"  wd={wd:<6}  Val-T1={v1*100:6.2f}%  Val-T5={v5*100:6.2f}%  ({t:.0f}s)")
    if v1 > best_val: best_val = v1; best_wd = wd

print(f"\n  BEST: wd={best_wd}  Val-T1={best_val*100:.2f}%")
print(f"  Previous (no interactions): 19.45%")

# Save
with open(OUTPUT_DIR / 'results_with_interactions.json', 'w') as f:
    json.dump({
        'n_base': n_selected, 'n_interactions': n_interactions, 'n_final': int(n_final),
        'n_base_kept': int(n_base_kept), 'n_int_kept': int(n_int_kept),
        'best_wd': best_wd, 'best_val_top1': best_val,
        'results': {k: {'val_top1': v[0], 'val_top5': v[1]} for k, v in results.items()},
    }, f, indent=2)
