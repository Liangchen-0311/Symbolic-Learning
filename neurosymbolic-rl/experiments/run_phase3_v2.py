#!/usr/bin/env python3
"""
Phase 3: Feature Extraction + Linear Classifier for Symbolic Feature Discovery.

Loads 10,039 formulas from 4 Phase-1 banks, then:
  Step 1: Fast dedup at 64×64 (Pearson r=0.88, greedy by accuracy desc)
  Step 2: Feature extraction at 112×112 with LRU sub-expression cache
  Step 3: Train nn.Linear classifier with weight_decay sweep

Usage:
    python experiments/run_phase3_v2.py --device cuda
"""

import argparse
import gc
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.imagenet_loader import ImageNetDataModule, IMAGENET_SUPERCLASS_NAMES
from src.symbolic.tensor_operators import TENSOR_OPERATORS


# ======================================================================
# Config
# ======================================================================

DATA_DIR = '/workspace/neurosymbolic-rl/data/imagenet'
BANK_DIRS = [
    Path('outputs/imagenet_v2/phase1/bank_0/feature_bank'),
    Path('outputs/imagenet_v2/phase1/bank_1/feature_bank'),
    Path('outputs/imagenet_v2/phase1/bank_2/feature_bank'),
    Path('outputs/imagenet_v2/phase1/bank_3/feature_bank'),
]
OUTPUT_DIR = Path('outputs/imagenet_v2/phase3_v2')
NUM_CLASSES = 1000


# ======================================================================
# Formula execution helpers
# ======================================================================

def build_data_batch(images, device):
    """Build terminal dict from a batch of RGB images [B, 3, H, W]."""
    images = images.to(device)
    I_R = images[:, 0]
    I_G = images[:, 1]
    I_B = images[:, 2]

    I_GRAY = 0.2989 * I_R + 0.5870 * I_G + 0.1140 * I_B

    Cmax, _ = images.max(dim=1)
    Cmin, _ = images.min(dim=1)
    delta = Cmax - Cmin + 1e-8

    H = torch.zeros_like(I_R)
    mask_r = (Cmax == I_R)
    mask_g = (Cmax == I_G) & ~mask_r
    mask_b = ~mask_r & ~mask_g
    H[mask_r] = (((I_G[mask_r] - I_B[mask_r]) / delta[mask_r]) % 6)
    H[mask_g] = ((I_B[mask_g] - I_R[mask_g]) / delta[mask_g]) + 2
    H[mask_b] = ((I_R[mask_b] - I_G[mask_b]) / delta[mask_b]) + 4
    H = H / 6.0

    S = torch.where(Cmax > 1e-8, delta / (Cmax + 1e-8), torch.zeros_like(Cmax))

    return {
        'I_R': I_R, 'I_G': I_G, 'I_B': I_B,
        'I_GRAY': I_GRAY, 'I_H': H, 'I_S': S,
    }


def execute_formula(formula_str, data_batch):
    """Execute a single RPN formula. Returns [batch] scalar or None on failure."""
    tokens = formula_str.strip().split()
    stack = []
    for token in tokens:
        if token in data_batch:
            stack.append(data_batch[token])
        elif token in TENSOR_OPERATORS:
            op_func, arity, _ = TENSOR_OPERATORS[token]
            if len(stack) < arity:
                return None
            operands = [stack.pop() for _ in range(arity)]
            operands.reverse()
            result = op_func(*operands)
            result = torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)
            stack.append(result)
        else:
            return None
    if len(stack) != 1:
        return None
    out = stack[0]
    out = torch.clamp(out, -1e4, 1e4)
    # Reduce spatial dims to scalar if needed
    if out.dim() > 1:
        out = out.mean(dim=tuple(range(1, out.dim())))
    return out


# ======================================================================
# LRU Sub-expression Cache
# ======================================================================

class LRUCache:
    """LRU cache for sub-expression tensors on GPU."""

    def __init__(self, max_size=200):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = value
            return
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[key] = value

    def clear(self):
        self.cache.clear()

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


def execute_formula_with_lru(formula_str, data_batch, lru_cache):
    """Execute RPN formula with LRU caching of sub-expression prefixes.

    For formula "I_R edge_x blur global_avg_pool", caches:
      - "I_R" → tensor
      - "I_R edge_x" → tensor
      - "I_R edge_x blur" → tensor
    Adjacent sorted formulas sharing prefixes get cache hits.
    """
    tokens = formula_str.strip().split()
    stack = []
    # Track the prefix string for caching
    prefix_parts = []

    for token in tokens:
        if token in data_batch:
            stack.append(data_batch[token])
            prefix_parts.append(token)
        elif token in TENSOR_OPERATORS:
            op_func, arity, _ = TENSOR_OPERATORS[token]
            if len(stack) < arity:
                return None
            prefix_parts.append(token)
            prefix_key = ' '.join(prefix_parts)

            # Check cache for this prefix
            cached = lru_cache.get(prefix_key)
            if cached is not None:
                # Pop consumed operands from stack
                for _ in range(arity):
                    stack.pop()
                stack.append(cached)
                continue

            operands = [stack.pop() for _ in range(arity)]
            operands.reverse()
            result = op_func(*operands)
            # No intermediate NaN/Inf check (avoids GPU-CPU sync).
            # Sanitize in-place on GPU instead.
            result = torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)

            # Cache intermediate results (not the final pooling step)
            # Only cache spatial tensors (dim > 1), not scalars
            if result.dim() > 1:
                lru_cache.put(prefix_key, result)

            stack.append(result)
        else:
            return None

    if len(stack) != 1:
        return None

    out = stack[0]
    out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
    out = torch.clamp(out, -1e4, 1e4)
    if out.dim() > 1:
        out = out.mean(dim=tuple(range(1, out.dim())))
    return out


# ======================================================================
# Step 1: Fast Deduplication
# ======================================================================

def step1_dedup(device, dedup_threshold=0.88):
    """Load all formulas, execute on 1000 images at 64×64, Pearson dedup."""
    print("\n" + "=" * 70)
    print("  STEP 1: FAST DEDUPLICATION")
    print("=" * 70)

    # Load all formulas from 4 banks
    all_formulas = []
    seen = set()
    for bank_dir in BANK_DIRS:
        fb_path = bank_dir / 'feature_bank.json'
        if not fb_path.exists():
            print(f"  WARNING: {fb_path} not found, skipping")
            continue
        with open(fb_path) as f:
            bank = json.load(f)
        for entry in bank['formulas']:
            fstr = entry['str']
            if fstr not in seen:
                seen.add(fstr)
                all_formulas.append({
                    'str': fstr,
                    'accuracy': entry.get('accuracy', 0.0),
                })
    print(f"  Loaded {len(all_formulas)} unique formulas from {len(BANK_DIRS)} banks")

    # Sort by accuracy descending (greedy: keep best first)
    all_formulas.sort(key=lambda x: x['accuracy'], reverse=True)

    # Load 1 image per class at 64×64
    print("  Loading 1000 images (1/class) at 64×64...")
    dm = ImageNetDataModule(
        data_dir=DATA_DIR, resolution=64, batch_size=200,
        num_workers=8, samples_per_class=1,
    )
    dm.setup()
    loader = DataLoader(
        dm.train_dataset, batch_size=200,
        shuffle=False, num_workers=8, pin_memory=True,
    )

    # Collect all images into one batch
    all_imgs = []
    for imgs, _ in loader:
        all_imgs.append(imgs)
    all_imgs = torch.cat(all_imgs, dim=0)  # [1000, 3, 64, 64]
    n_imgs = all_imgs.shape[0]
    print(f"  Got {n_imgs} images")

    # Execute all formulas → output matrix [n_imgs, n_formulas]
    print(f"  Executing {len(all_formulas)} formulas...")
    data_batch = build_data_batch(all_imgs, device)

    n_formulas = len(all_formulas)
    out_matrix = np.zeros((n_imgs, n_formulas), dtype=np.float32)
    valid_mask = np.ones(n_formulas, dtype=bool)

    t0 = time.time()
    for f_idx, entry in enumerate(all_formulas):
        try:
            out = execute_formula(entry['str'], data_batch)
            if out is None:
                valid_mask[f_idx] = False
                continue
            out_matrix[:, f_idx] = out.cpu().numpy()
        except Exception:
            valid_mask[f_idx] = False

        if (f_idx + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f"    {f_idx+1}/{n_formulas} formulas ({elapsed:.1f}s)")

    del data_batch, all_imgs
    torch.cuda.empty_cache()

    n_valid = valid_mask.sum()
    print(f"  Executed in {time.time()-t0:.1f}s — {n_valid}/{n_formulas} valid")

    # Filter out invalid formulas
    valid_indices = np.where(valid_mask)[0]
    out_valid = out_matrix[:, valid_indices]
    formulas_valid = [all_formulas[i] for i in valid_indices]

    # Filter constant-output formulas (std < 1e-10)
    stds = out_valid.std(axis=0)
    nonconstant = stds > 1e-10
    out_valid = out_valid[:, nonconstant]
    formulas_valid = [f for f, nc in zip(formulas_valid, nonconstant) if nc]
    print(f"  After removing constants: {len(formulas_valid)} formulas")

    # Pearson correlation dedup (greedy, already sorted by accuracy desc)
    print(f"  Pearson dedup (threshold={dedup_threshold})...")
    t0 = time.time()

    # Standardize columns
    means = out_valid.mean(axis=0)
    stds = out_valid.std(axis=0)
    stds = np.maximum(stds, 1e-10)
    X = (out_valid - means) / stds  # [n_imgs, n_formulas]

    kept_indices = []
    kept_vecs = []  # list of standardized column vectors

    for i in range(len(formulas_valid)):
        vec = X[:, i]  # [n_imgs]

        if len(kept_vecs) == 0:
            kept_indices.append(i)
            kept_vecs.append(vec)
            continue

        # Vectorized correlation: |vec @ kept_mat| / n_imgs
        kept_mat = np.stack(kept_vecs, axis=1)  # [n_imgs, n_kept]
        corrs = np.abs(vec @ kept_mat / n_imgs)

        if corrs.max() < dedup_threshold:
            kept_indices.append(i)
            kept_vecs.append(vec)

        if (i + 1) % 1000 == 0:
            print(f"    Processed {i+1}/{len(formulas_valid)}, kept {len(kept_indices)}")

    dedup_formulas = [formulas_valid[i] for i in kept_indices]
    print(f"  Dedup done in {time.time()-t0:.1f}s: "
          f"{len(formulas_valid)} → {len(dedup_formulas)} formulas")

    # Save deduped formula list
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dedup_path = OUTPUT_DIR / 'dedup_formulas.json'
    with open(dedup_path, 'w') as f:
        json.dump(dedup_formulas, f, indent=2)
    print(f"  Saved to {dedup_path}")

    return dedup_formulas


# ======================================================================
# Step 2: Feature Matrix Extraction
# ======================================================================

def step2_extract_features(dedup_formulas, device):
    """Extract features at 112×112 with LRU sub-expression cache."""
    print("\n" + "=" * 70)
    print("  STEP 2: FEATURE MATRIX EXTRACTION (112×112)")
    print("=" * 70)

    formula_strs = [f['str'] for f in dedup_formulas]
    n_feats = len(formula_strs)

    # Sort lexicographically for LRU cache efficiency
    formula_strs_sorted = sorted(formula_strs)
    print(f"  {n_feats} formulas sorted lexicographically for cache optimization")

    # Save sorted formula list
    with open(OUTPUT_DIR / 'formula_list_sorted.json', 'w') as f:
        json.dump(formula_strs_sorted, f)

    # Load ImageNet: train 200/class, val full
    print("  Loading ImageNet at 112×112 (train: 200/class, val: full)...")
    dm_train = ImageNetDataModule(
        data_dir=DATA_DIR, resolution=112, batch_size=512,
        num_workers=8, samples_per_class=200,
    )
    dm_train.setup()
    train_loader = DataLoader(
        dm_train.train_dataset, batch_size=512,
        shuffle=False, num_workers=8, pin_memory=True, drop_last=False,
    )
    val_loader = dm_train.get_val_loader()

    n_train = len(dm_train.train_dataset)
    n_val = len(dm_train.val_dataset)
    print(f"  Train: {n_train} images")
    print(f"  Val:   {n_val} images")
    print(f"  Feature matrix: train ({n_train}, {n_feats}) = "
          f"{n_train * n_feats * 4 / 1e9:.2f} GB")
    print(f"  Feature matrix: val   ({n_val}, {n_feats}) = "
          f"{n_val * n_feats * 4 / 1e9:.2f} GB")

    # Extract train features
    train_mmap_path = str(OUTPUT_DIR / 'X_train.mmap')
    print(f"\n  Extracting training features...")
    t0 = time.time()
    X_train, y_train = _extract_features_lru(
        formula_strs_sorted, train_loader, device,
        mmap_path=train_mmap_path, n_total=n_train, n_feats=n_feats, tag="train",
    )
    np.save(str(OUTPUT_DIR / 'y_train.npy'), y_train)
    print(f"  Train done: ({n_train}, {n_feats}) in {(time.time()-t0)/60:.1f} min")

    gc.collect()
    torch.cuda.empty_cache()

    # Extract val features
    val_mmap_path = str(OUTPUT_DIR / 'X_val.mmap')
    print(f"\n  Extracting validation features...")
    t0 = time.time()
    X_val, y_val = _extract_features_lru(
        formula_strs_sorted, val_loader, device,
        mmap_path=val_mmap_path, n_total=n_val, n_feats=n_feats, tag="val",
    )
    np.save(str(OUTPUT_DIR / 'y_val.npy'), y_val)
    print(f"  Val done: ({n_val}, {n_feats}) in {(time.time()-t0)/60:.1f} min")

    return X_train, y_train, X_val, y_val, n_feats, formula_strs_sorted


def _extract_features_lru(formula_strs, loader, device, mmap_path, n_total, n_feats, tag=""):
    """Extract features to memory-mapped file with LRU sub-expression cache.

    Formulas must be sorted lexicographically for optimal cache performance.
    Supports checkpoint/resume for crash safety.
    """
    progress_path = mmap_path + '.progress.json'
    label_path = mmap_path.replace('.mmap', '_labels_partial.npy')

    # Check for existing progress (resume after crash)
    start_batch = 0
    row_offset = 0
    all_labels = []
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
        start_batch = progress['next_batch']
        row_offset = progress['row_offset']
        if os.path.exists(label_path):
            all_labels = list(np.load(label_path, allow_pickle=True))
        print(f"    [{tag}] Resuming from batch {start_batch} (row {row_offset}/{n_total})")
        mmap = np.memmap(mmap_path, dtype='float32', mode='r+', shape=(n_total, n_feats))
    else:
        mmap = np.memmap(mmap_path, dtype='float32', mode='w+', shape=(n_total, n_feats))

    total_batches = len(loader)
    # LRU cache: max_size=500 ≈ 500 × B × 112 × 112 × 4 bytes
    # For B=512: ~12.5 GB, fits on A100 (80GB)
    lru = LRUCache(max_size=500)

    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx < start_batch:
            continue

        B = images.shape[0]
        data_batch = build_data_batch(images, device)

        # Clear LRU between batches (batch size may change, tensor shapes differ)
        lru.clear()

        # Accumulate all formula outputs on GPU (avoids per-formula GPU→CPU sync)
        gpu_buf = torch.zeros(B, n_feats, device=device)

        for f_idx, formula_str in enumerate(formula_strs):
            try:
                out = execute_formula_with_lru(formula_str, data_batch, lru)
                if out is not None:
                    gpu_buf[:, f_idx] = out
            except Exception:
                pass  # stays zero

        # Single GPU→CPU transfer per batch
        batch_feats = gpu_buf.cpu().numpy()
        del gpu_buf, data_batch
        torch.cuda.empty_cache()

        end = min(row_offset + B, n_total)
        actual_B = end - row_offset
        mmap[row_offset:end] = batch_feats[:actual_B]
        all_labels.append(labels.numpy()[:actual_B])
        row_offset = end

        # Checkpoint every 10 batches
        if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
            mmap.flush()
            with open(progress_path, 'w') as f:
                json.dump({'next_batch': batch_idx + 1, 'row_offset': row_offset}, f)
            np.save(label_path, np.concatenate(all_labels, axis=0))

        if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
            pct = row_offset / n_total * 100
            hr = lru.hit_rate
            print(f"    [{tag}] batch {batch_idx+1}/{total_batches}  "
                  f"({row_offset}/{n_total} imgs, {pct:.1f}%)  "
                  f"LRU hit={hr:.1%}")

    mmap.flush()
    y = np.concatenate(all_labels, axis=0)

    # Clean up progress files on success
    for p in [progress_path, label_path]:
        if os.path.exists(p):
            os.remove(p)

    print(f"    [{tag}] Final LRU stats: hits={lru.hits}, misses={lru.misses}, "
          f"rate={lru.hit_rate:.1%}")
    return mmap, y


# ======================================================================
# Step 3: Train Linear Classifier
# ======================================================================

def step3_train_classifier(X_train, y_train, X_val, y_val, n_feats, device):
    """Train nn.Linear with weight_decay sweep, report results."""
    print("\n" + "=" * 70)
    print("  STEP 3: TRAIN LINEAR CLASSIFIER")
    print("=" * 70)

    n_train = len(y_train)

    # Online StandardScaler
    print("  Computing feature statistics...")
    feat_mean, feat_std = _online_mean_std(X_train, n_train, n_feats)
    feat_std = np.maximum(feat_std, 1e-8)
    np.save(str(OUTPUT_DIR / 'feat_mean.npy'), feat_mean)
    np.save(str(OUTPUT_DIR / 'feat_std.npy'), feat_std)

    # Weight decay sweep
    weight_decay_values = [1e-4, 1e-3, 1e-2, 1e-1]
    all_results = {}
    best_val_acc = 0.0
    best_wd = None

    print(f"\n  {'='*60}")
    print(f"  nn.Linear({n_feats}, {NUM_CLASSES}) + AdamW + CosineAnnealingLR")
    print(f"  {'='*60}")

    for wd in weight_decay_values:
        print(f"\n  weight_decay={wd} ...")
        result = _train_linear(
            X_train, y_train, X_val, y_val,
            feat_mean, feat_std, n_feats,
            device=device, weight_decay=wd, epochs=20, lr=1e-3, batch_size=1024,
        )
        label = f"wd={wd}"
        all_results[label] = result
        print(f"    Train={result['train_top1']*100:.2f}%  "
              f"Val-Top1={result['val_top1']*100:.2f}%  "
              f"Val-Top5={result['val_top5']*100:.2f}%  "
              f"({result['time_s']:.1f}s)")

        if result['val_top1'] > best_val_acc:
            best_val_acc = result['val_top1']
            best_wd = wd

    # Per-Superclass Accuracy
    print(f"\n  {'='*60}")
    print(f"  Per-Superclass Accuracy (best wd={best_wd})")
    print(f"  {'='*60}")

    best_result = all_results[f"wd={best_wd}"]
    superclass_accs = _per_superclass_accuracy(
        X_val, y_val, feat_mean, feat_std, n_feats, device,
        best_wd, best_result.get('model_state'),
    )
    for sc_name, sc_acc in sorted(superclass_accs.items(), key=lambda x: x[1]):
        print(f"    {sc_name:<25s} {sc_acc*100:6.2f}%")

    # Final Report
    print(f"\n  {'='*60}")
    print(f"  FINAL RESULTS ({n_feats} symbolic features)")
    print(f"  {'='*60}")
    print(f"  {'Config':<15} {'Train':>8} {'Val-T1':>8} {'Val-T5':>8}")
    print(f"  {'-'*43}")
    for name, r in all_results.items():
        print(f"  {name:<15} {r['train_top1']*100:>7.2f}% "
              f"{r['val_top1']*100:>7.2f}% "
              f"{r['val_top5']*100:>7.2f}%")
    print(f"  {'-'*43}")
    best_r = all_results[f'wd={best_wd}']
    print(f"  BEST: wd={best_wd}  "
          f"Top-1={best_r['val_top1']*100:.2f}%  "
          f"Top-5={best_r['val_top5']*100:.2f}%")
    print(f"  {'='*60}\n")

    # Save results
    save_results = {
        k: {kk: vv for kk, vv in v.items() if kk != 'model_state'}
        for k, v in all_results.items()
    }
    with open(OUTPUT_DIR / 'final_results.json', 'w') as f:
        json.dump({
            'num_formulas': n_feats,
            'train_samples': n_train,
            'val_samples': len(y_val),
            'best_weight_decay': best_wd,
            'best_val_top1': float(best_val_acc),
            'best_val_top5': float(best_r['val_top5']),
            'all_results': save_results,
            'superclass_accuracies': superclass_accs,
        }, f, indent=2)
    print(f"  Results saved to {OUTPUT_DIR / 'final_results.json'}")

    return all_results


def _online_mean_std(X_mmap, n_total, n_feats, chunk_size=10000):
    """Compute mean and std from memory-mapped features without loading all into RAM."""
    running_sum = np.zeros(n_feats, dtype=np.float64)
    running_sq_sum = np.zeros(n_feats, dtype=np.float64)

    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        chunk = np.array(X_mmap[start:end], dtype=np.float64)
        chunk = np.nan_to_num(chunk)
        running_sum += chunk.sum(axis=0)
        running_sq_sum += (chunk ** 2).sum(axis=0)

    mean = running_sum / n_total
    var = running_sq_sum / n_total - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))
    return mean.astype(np.float32), std.astype(np.float32)


def _train_linear(
    X_train_mmap, y_train, X_val_mmap, y_val,
    feat_mean, feat_std, n_feats,
    device='cuda', weight_decay=1e-2, epochs=20, lr=1e-3, batch_size=1024,
):
    """Train nn.Linear + AdamW + CosineAnnealingLR."""
    model = torch.nn.Linear(n_feats, NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_train = len(y_train)
    mean_t = torch.tensor(feat_mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(feat_std, dtype=torch.float32, device=device)

    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]

            X_batch = torch.tensor(
                np.array(X_train_mmap[idx]), dtype=torch.float32, device=device
            )
            X_batch = torch.nan_to_num(X_batch)
            X_batch = (X_batch - mean_t) / std_t
            y_batch = torch.tensor(y_train[idx], dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"      epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")

    elapsed = time.time() - t0

    # Evaluate
    model.eval()
    train_top1 = _eval_accuracy(model, X_train_mmap, y_train, mean_t, std_t, device, batch_size)
    val_top1, val_top5 = _eval_accuracy(
        model, X_val_mmap, y_val, mean_t, std_t, device, batch_size, top5=True
    )

    return {
        'train_top1': train_top1,
        'val_top1': val_top1,
        'val_top5': val_top5,
        'weight_decay': weight_decay,
        'time_s': elapsed,
        'model_state': model.state_dict(),
    }


def _eval_accuracy(model, X_mmap, y, mean_t, std_t, device, batch_size=2048, top5=False):
    """Evaluate accuracy on memory-mapped features."""
    n = len(y)
    correct_top1 = 0
    correct_top5 = 0

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_batch = torch.tensor(
                np.array(X_mmap[start:end]), dtype=torch.float32, device=device
            )
            X_batch = torch.nan_to_num(X_batch)
            X_batch = (X_batch - mean_t) / std_t
            y_batch = torch.tensor(y[start:end], dtype=torch.long, device=device)

            logits = model(X_batch)
            preds = logits.argmax(dim=1)
            correct_top1 += (preds == y_batch).sum().item()

            if top5:
                k = min(5, logits.shape[1])
                _, top_k = logits.topk(k, dim=1)
                correct_top5 += (top_k == y_batch.unsqueeze(1)).any(dim=1).sum().item()

    if top5:
        return correct_top1 / n, correct_top5 / n
    return correct_top1 / n


def _per_superclass_accuracy(
    X_val_mmap, y_val, feat_mean, feat_std, n_feats, device, weight_decay, model_state
):
    """Compute accuracy for each of the 20 ImageNet superclasses."""
    model = torch.nn.Linear(n_feats, NUM_CLASSES).to(device)
    if model_state is not None:
        model.load_state_dict(model_state)
    model.eval()

    mean_t = torch.tensor(feat_mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(feat_std, dtype=torch.float32, device=device)

    all_preds = []
    n = len(y_val)
    with torch.no_grad():
        for start in range(0, n, 2048):
            end = min(start + 2048, n)
            X_batch = torch.tensor(
                np.array(X_val_mmap[start:end]), dtype=torch.float32, device=device
            )
            X_batch = torch.nan_to_num(X_batch)
            X_batch = (X_batch - mean_t) / std_t
            preds = model(X_batch).argmax(dim=1).cpu().numpy()
            all_preds.append(preds)

    all_preds = np.concatenate(all_preds)

    num_superclasses = 20
    superclass_correct = {i: 0 for i in range(num_superclasses)}
    superclass_total = {i: 0 for i in range(num_superclasses)}

    for pred, true_label in zip(all_preds, y_val):
        sc = true_label % num_superclasses
        superclass_total[sc] += 1
        if pred == true_label:
            superclass_correct[sc] += 1

    results = {}
    for sc_id in range(num_superclasses):
        name = IMAGENET_SUPERCLASS_NAMES.get(sc_id, f'superclass_{sc_id}')
        total = superclass_total[sc_id]
        acc = superclass_correct[sc_id] / max(total, 1)
        results[name] = float(acc)

    return results


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 3: Symbolic Feature → Linear Classifier')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--dedup_threshold', type=float, default=0.88,
                        help='Pearson correlation threshold for dedup (default: 0.88)')
    parser.add_argument('--start_step', type=int, default=1, choices=[1, 2, 3],
                        help='Start from this step (1=dedup, 2=extract, 3=train)')
    parser.add_argument('--max_formulas', type=int, default=6000,
                        help='Max formulas to keep after dedup (by accuracy, default: 6000)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output directory')
    args = parser.parse_args()

    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU")
        device = 'cpu'

    torch.manual_seed(42)
    np.random.seed(42)
    if device == 'cuda':
        torch.cuda.manual_seed(42)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_start = time.time()

    print(f"Device: {device}")
    print(f"Output: {OUTPUT_DIR}")

    # Step 1: Dedup
    if args.start_step <= 1:
        dedup_formulas = step1_dedup(device, dedup_threshold=args.dedup_threshold)
    else:
        dedup_path = OUTPUT_DIR / 'dedup_formulas.json'
        print(f"\nLoading deduped formulas from {dedup_path}")
        with open(dedup_path) as f:
            dedup_formulas = json.load(f)
        print(f"  Loaded {len(dedup_formulas)} formulas")

    # Cap formula count by accuracy
    if len(dedup_formulas) > args.max_formulas:
        # dedup_formulas are already sorted by accuracy desc from step1
        dedup_formulas = sorted(dedup_formulas, key=lambda x: x['accuracy'], reverse=True)
        dedup_formulas = dedup_formulas[:args.max_formulas]
        print(f"  Capped to top {args.max_formulas} formulas by accuracy "
              f"(min_acc={dedup_formulas[-1]['accuracy']:.4f})")

    # Step 2: Feature extraction
    if args.start_step <= 2:
        X_train, y_train, X_val, y_val, n_feats, formula_strs = \
            step2_extract_features(dedup_formulas, device)
    else:
        # Load existing mmap files
        formula_path = OUTPUT_DIR / 'formula_list_sorted.json'
        with open(formula_path) as f:
            formula_strs = json.load(f)
        n_feats = len(formula_strs)
        y_train = np.load(str(OUTPUT_DIR / 'y_train.npy'))
        y_val = np.load(str(OUTPUT_DIR / 'y_val.npy'))
        n_train = len(y_train)
        n_val = len(y_val)
        X_train = np.memmap(
            str(OUTPUT_DIR / 'X_train.mmap'), dtype='float32', mode='r',
            shape=(n_train, n_feats)
        )
        X_val = np.memmap(
            str(OUTPUT_DIR / 'X_val.mmap'), dtype='float32', mode='r',
            shape=(n_val, n_feats)
        )
        print(f"\nLoaded existing features: train ({n_train}, {n_feats}), val ({n_val}, {n_feats})")

    # Step 3: Train classifier
    step3_train_classifier(X_train, y_train, X_val, y_val, n_feats, device)

    total_time = time.time() - pipeline_start
    print(f"\n{'='*70}")
    print(f"  PHASE 3 COMPLETE — Total time: {total_time/3600:.2f} hours")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
