#!/usr/bin/env python3
"""
Phase 3 v3: Multi-resolution SPP + Histogram + Interactions + L1 Selection.

Pipeline:
  1. Load 2676 Phase 2 formulas, strip final pooling → bodies
  2. Extract at 3 resolutions (64, 112, 224) × 12 encodings per body
     - 8 SPP pools (global_avg/max/std, quad_tl/tr/bl/br, center)
     - 4 histogram bins (patch_histogram_4x4)
     = 2676 × 12 × 3 = 96,336 base features
  3. L1 feature selection → reduce to ~30K
  4. Feature interactions (top-300 pairwise products) → ~45K additional
  5. Final L1 selection → ~50K effective features
  6. Train nn.Linear(n_features, 1000)
"""

import gc, json, os, sys, time, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.imagenet_loader import ImageNetDataModule
from src.symbolic.tensor_operators import TENSOR_OPERATORS, ROOT_OPERATORS, SymbolicKernelBank

# ======================================================================
# Config
# ======================================================================

DATA_DIR = '/workspace/neurosymbolic-rl/data/imagenet'
PHASE2_DIR = Path('outputs/imagenet_v3/phase2/feature_bank')
OUTPUT_DIR = Path('outputs/imagenet_v3/phase3_v3')
NUM_CLASSES = 1000
RESOLUTIONS = [112, 224]  # skip 64 (worst resolution, save 32GB disk)
BATCH_SIZES = {112: 512, 224: 256}

# 8 SPP pooling operators
SPP_POOLS = [
    'global_avg_pool', 'global_max_pool', 'global_std_pool',
    'pool_quad_tl', 'pool_quad_tr', 'pool_quad_bl', 'pool_quad_br',
    'pool_center',
]
SPP_POOL_FUNCS = [(name, TENSOR_OPERATORS[name][0]) for name in SPP_POOLS]
N_SPP = len(SPP_POOLS)
N_HIST = 4  # patch_histogram_4x4 outputs 4 bins
N_ENCODINGS = N_SPP + N_HIST  # 12

# Register kernel bank with FIXED weights (saved during Phase 1)
kernel_bank = SymbolicKernelBank(device='cuda' if torch.cuda.is_available() else 'cpu')
_kb_weights_path = OUTPUT_DIR / 'kernel_bank_weights.pt'
if _kb_weights_path.exists():
    kernel_bank.load_state_dict(torch.load(str(_kb_weights_path), weights_only=True))
    print(f"Loaded kernel bank weights from {_kb_weights_path}")
else:
    # First run: save weights for future consistency
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(kernel_bank.state_dict(), str(_kb_weights_path))
    print(f"Saved kernel bank weights to {_kb_weights_path}")
kernel_bank.register_operators(TENSOR_OPERATORS)


# ======================================================================
# Helpers
# ======================================================================

def build_data_batch(images, device):
    images = images.to(device)
    I_R, I_G, I_B = images[:, 0], images[:, 1], images[:, 2]
    I_GRAY = 0.2989 * I_R + 0.5870 * I_G + 0.1140 * I_B
    Cmax, _ = images.max(dim=1); Cmin, _ = images.min(dim=1)
    delta = Cmax - Cmin + 1e-8
    H = torch.zeros_like(I_R)
    mask_r = (Cmax == I_R); mask_g = (Cmax == I_G) & ~mask_r; mask_b = ~mask_r & ~mask_g
    H[mask_r] = (((I_G[mask_r] - I_B[mask_r]) / delta[mask_r]) % 6)
    H[mask_g] = ((I_B[mask_g] - I_R[mask_g]) / delta[mask_g]) + 2
    H[mask_b] = ((I_R[mask_b] - I_G[mask_b]) / delta[mask_b]) + 4
    H = H / 6.0
    S = torch.where(Cmax > 1e-8, delta / (Cmax + 1e-8), torch.zeros_like(Cmax))
    total = I_R + I_G + I_B + 1e-8
    return {
        'I_R': I_R, 'I_G': I_G, 'I_B': I_B, 'I_GRAY': I_GRAY,
        'I_H': H, 'I_S': S,
        'I_r': I_R / total, 'I_g': I_G / total,
        'I_RG': I_R - I_G, 'I_BY': I_B - (I_R + I_G) / 2,
    }


def execute_body(body_str, data_batch):
    """Execute formula body → spatial [B, H, W] or None."""
    tokens = body_str.strip().split()
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
    if out.dim() < 2:
        return None
    return out


def encode_feature_map(feat_map):
    """Apply 8 SPP pools + 4 histogram bins to [B, H, W] → [B, 12]."""
    B = feat_map.shape[0]
    encodings = []

    # 8 SPP pools
    for name, pool_func in SPP_POOL_FUNCS:
        try:
            pooled = pool_func(feat_map)
            pooled = torch.nan_to_num(pooled, nan=0.0, posinf=1e4, neginf=-1e4)
            pooled = torch.clamp(pooled, -1e4, 1e4)
            encodings.append(pooled)
        except Exception:
            encodings.append(torch.zeros(B, device=feat_map.device))

    # 4 histogram bins via patch_histogram_4x4
    try:
        hist = TENSOR_OPERATORS['patch_histogram_4x4'][0](feat_map)  # [B, 4]
        hist = torch.nan_to_num(hist, nan=0.0, posinf=1e4, neginf=-1e4)
        hist = torch.clamp(hist, -1e4, 1e4)
        for i in range(N_HIST):
            encodings.append(hist[:, i])
    except Exception:
        for _ in range(N_HIST):
            encodings.append(torch.zeros(B, device=feat_map.device))

    return torch.stack(encodings, dim=1)  # [B, 12]


# ======================================================================
# Feature Extraction
# ======================================================================

def extract_multires_features(body_strs, device, samples_per_class=200):
    """Extract features at multiple resolutions → mmap files."""
    n_bodies = len(body_strs)
    n_feats_per_res = n_bodies * N_ENCODINGS
    n_feats_total = n_feats_per_res * len(RESOLUTIONS)

    for res in RESOLUTIONS:
        bs = BATCH_SIZES[res]
        print(f"\n  === Resolution {res}×{res} ===")

        dm = ImageNetDataModule(
            data_dir=DATA_DIR, resolution=res, batch_size=bs,
            num_workers=8, samples_per_class=samples_per_class,
        )
        dm.setup()
        train_loader = DataLoader(
            dm.train_dataset, batch_size=bs,
            shuffle=False, num_workers=8, pin_memory=True, drop_last=False,
        )
        val_loader = dm.get_val_loader()
        n_train = len(dm.train_dataset)
        n_val = len(dm.val_dataset)

        for split, loader, n_total in [('train', train_loader, n_train), ('val', val_loader, n_val)]:
            mmap_path = str(OUTPUT_DIR / f'X_{split}_{res}.mmap')
            progress_path = mmap_path + '.progress.json'

            # Skip if already completed (mmap exists, no progress file = completed)
            if os.path.exists(mmap_path) and not os.path.exists(progress_path):
                print(f"    [{split}@{res}] Already completed, skipping")
                continue

            # Check resume
            start_batch = 0
            row_offset = 0
            all_labels = []
            label_path = mmap_path.replace('.mmap', '_labels_partial.npy')

            if os.path.exists(progress_path):
                with open(progress_path) as f:
                    progress = json.load(f)
                start_batch = progress['next_batch']
                row_offset = progress['row_offset']
                if os.path.exists(label_path):
                    saved_labels = np.load(label_path)
                    all_labels = [saved_labels]  # wrap as single array in list
                print(f"    [{split}] Resuming from batch {start_batch} (row {row_offset}/{n_total})")
                mmap = np.memmap(mmap_path, dtype='float32', mode='r+', shape=(n_total, n_feats_per_res))
            else:
                mmap = np.memmap(mmap_path, dtype='float32', mode='w+', shape=(n_total, n_feats_per_res))

            total_batches = len(loader)
            t0 = time.time()

            for batch_idx, (images, labels) in enumerate(loader):
                if batch_idx < start_batch:
                    continue

                B = images.shape[0]
                data_batch = build_data_batch(images, device)
                gpu_buf = torch.zeros(B, n_feats_per_res, device=device)

                for b_idx, body_str in enumerate(body_strs):
                    try:
                        feat_map = execute_body(body_str, data_batch)
                        if feat_map is not None:
                            encoded = encode_feature_map(feat_map)  # [B, 12]
                            gpu_buf[:, b_idx * N_ENCODINGS:(b_idx + 1) * N_ENCODINGS] = encoded
                    except Exception:
                        pass

                batch_feats = gpu_buf.cpu().numpy()
                del gpu_buf, data_batch
                torch.cuda.empty_cache()

                end = min(row_offset + B, n_total)
                actual_B = end - row_offset
                mmap[row_offset:end] = batch_feats[:actual_B]
                all_labels.append(labels.numpy()[:actual_B])
                row_offset = end

                if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
                    mmap.flush()
                    with open(progress_path, 'w') as f:
                        json.dump({'next_batch': batch_idx + 1, 'row_offset': row_offset}, f)
                    np.save(label_path, np.concatenate(all_labels, axis=0))

                if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
                    elapsed = time.time() - t0
                    pct = row_offset / n_total * 100
                    print(f"    [{split}@{res}] batch {batch_idx+1}/{total_batches}  "
                          f"({row_offset}/{n_total}, {pct:.1f}%)  {elapsed:.0f}s")

            mmap.flush()
            y = np.concatenate(all_labels, axis=0)
            # Save labels only from the first resolution that completes this split
            label_file = OUTPUT_DIR / f'y_{split}.npy'
            if not label_file.exists():
                np.save(str(label_file), y)
                print(f"    Saved {split} labels: {y.shape}")

            for p in [progress_path, label_path]:
                if os.path.exists(p):
                    os.remove(p)

            print(f"    [{split}@{res}] Done: ({n_total}, {n_feats_per_res}) in {(time.time()-t0)/60:.1f} min")

        gc.collect(); torch.cuda.empty_cache()


def concatenate_resolutions(n_train, n_val, n_feats_per_res):
    """Concatenate feature matrices from 3 resolutions into one."""
    n_feats_total = n_feats_per_res * len(RESOLUTIONS)

    for split, n_total in [('train', n_train), ('val', n_val)]:
        out_path = str(OUTPUT_DIR / f'X_{split}_all.mmap')
        if os.path.exists(out_path):
            print(f"  {split} concatenated mmap already exists, skipping")
            continue

        mmap_out = np.memmap(out_path, dtype='float32', mode='w+', shape=(n_total, n_feats_total))
        offset = 0
        for res in RESOLUTIONS:
            mmap_in = np.memmap(
                str(OUTPUT_DIR / f'X_{split}_{res}.mmap'), dtype='float32', mode='r',
                shape=(n_total, n_feats_per_res)
            )
            # Copy in chunks
            chunk = 10000
            for start in range(0, n_total, chunk):
                end = min(start + chunk, n_total)
                mmap_out[start:end, offset:offset + n_feats_per_res] = np.array(mmap_in[start:end])
            offset += n_feats_per_res
            del mmap_in

        mmap_out.flush()
        print(f"  {split}: concatenated ({n_total}, {n_feats_total})")

    return n_feats_total


# ======================================================================
# Online StandardScaler
# ======================================================================

def online_mean_std(X_mmap, n_total, n_feats, chunk_size=10000):
    running_sum = np.zeros(n_feats, dtype=np.float64)
    running_sq = np.zeros(n_feats, dtype=np.float64)
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        chunk = np.array(X_mmap[start:end], dtype=np.float64)
        chunk = np.nan_to_num(chunk)
        running_sum += chunk.sum(axis=0)
        running_sq += (chunk ** 2).sum(axis=0)
    mean = running_sum / n_total
    var = running_sq / n_total - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))
    return mean.astype(np.float32), std.astype(np.float32)


# ======================================================================
# L1 Feature Selection
# ======================================================================

def l1_feature_selection(X_train, y_train, X_val, y_val, feat_mean, feat_std,
                         n_feats, device, target_features=30000, wd=20.0, epochs=15):
    """Train with high L1-like regularization, keep features with non-zero weights."""
    print(f"\n  L1 Selection: {n_feats} → ~{target_features} features (wd={wd}, {epochs} epochs)")

    model = nn.Linear(n_feats, NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    mean_t = torch.tensor(feat_mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(feat_std, dtype=torch.float32, device=device)
    n_train = len(y_train)

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n_train)
        for start in range(0, n_train, 1024):
            end = min(start + 1024, n_train)
            idx = perm[start:end]
            X_b = torch.tensor(np.array(X_train[idx]), dtype=torch.float32, device=device)
            X_b = torch.nan_to_num(X_b)
            X_b = (X_b - mean_t) / std_t
            y_b = torch.tensor(y_train[idx], dtype=torch.long, device=device)
            optimizer.zero_grad()
            criterion(model(X_b), y_b).backward()
            optimizer.step()
        scheduler.step()

    # Select features by weight importance
    importance = model.weight.abs().sum(dim=0).detach().cpu().numpy()  # [n_feats]
    threshold = np.sort(importance)[::-1][min(target_features, len(importance) - 1)]
    selected = importance >= threshold
    n_selected = selected.sum()
    print(f"  Selected {n_selected} features (threshold={threshold:.6f})")

    return selected, importance


# ======================================================================
# Training
# ======================================================================

def train_and_eval(model, X_train, y_train, X_val, y_val, mean_t, std_t,
                   device, weight_decay, epochs=30, lr=1e-3, batch_size=1024):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    n_train = len(y_train)

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n_train)
        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]
            X_b = torch.tensor(np.array(X_train[idx]), dtype=torch.float32, device=device)
            X_b = torch.nan_to_num(X_b)
            X_b = (X_b - mean_t) / std_t
            y_b = torch.tensor(y_train[idx], dtype=torch.long, device=device)
            optimizer.zero_grad()
            criterion(model(X_b), y_b).backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"      epoch {epoch+1}/{epochs}")
    elapsed = time.time() - t0

    model.eval()
    def eval_split(X_mm, y):
        c1 = c5 = 0
        n = len(y)
        with torch.no_grad():
            for s in range(0, n, 2048):
                e = min(s + 2048, n)
                X_b = torch.tensor(np.array(X_mm[s:e]), dtype=torch.float32, device=device)
                X_b = torch.nan_to_num(X_b)
                X_b = (X_b - mean_t) / std_t
                logits = model(X_b)
                y_b = torch.tensor(y[s:e], dtype=torch.long, device=device)
                c1 += (logits.argmax(1) == y_b).sum().item()
                _, tk = logits.topk(min(5, NUM_CLASSES), dim=1)
                c5 += (tk == y_b.unsqueeze(1)).any(1).sum().item()
        return c1 / n, c5 / n

    tr1, _ = eval_split(X_train, y_train)
    v1, v5 = eval_split(X_val, y_val)
    return tr1, v1, v5, elapsed


# ======================================================================
# Main
# ======================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    np.random.seed(42)
    if device == 'cuda':
        torch.cuda.manual_seed(42)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_start = time.time()
    print(f"Device: {device}")
    print(f"Output: {OUTPUT_DIR}")

    # ── Load formulas, strip pooling → bodies ─────────────────
    d = json.load(open(PHASE2_DIR / 'feature_bank.json'))
    formulas = d['formulas']
    bodies = []
    for f in formulas:
        tokens = f['str'].strip().split()
        if tokens[-1] in ROOT_OPERATORS:
            bodies.append(' '.join(tokens[:-1]))
        else:
            bodies.append(f['str'])
    bodies_sorted = sorted(set(bodies))
    n_bodies = len(bodies_sorted)
    n_feats_per_res = n_bodies * N_ENCODINGS

    print(f"\n  {n_bodies} bodies × {N_ENCODINGS} encodings × {len(RESOLUTIONS)} resolutions "
          f"= {n_bodies * N_ENCODINGS * len(RESOLUTIONS)} base features")

    with open(OUTPUT_DIR / 'bodies_sorted.json', 'w') as f:
        json.dump(bodies_sorted, f)

    # ── Step 1: Multi-resolution feature extraction ───────────
    print(f"\n{'='*65}")
    print(f"  STEP 1: Multi-Resolution Feature Extraction")
    print(f"{'='*65}")

    extract_multires_features(bodies_sorted, device, samples_per_class=200)

    # ── Step 2: Per-resolution feature selection (no concatenation) ──
    # Avoids creating a ~77GB concatenated mmap. Instead, select features
    # from each resolution independently, then merge selected into one small mmap.
    print(f"\n{'='*65}")
    print(f"  STEP 2: Per-Resolution Feature Selection")
    print(f"{'='*65}")

    y_train = np.load(str(OUTPUT_DIR / 'y_train.npy'))
    y_val = np.load(str(OUTPUT_DIR / 'y_val.npy'))
    n_train, n_val = len(y_train), len(y_val)
    n_feats_total = n_feats_per_res * len(RESOLUTIONS)

    # Resolution quotas: 224 gets 50%, 112 gets 33%, 64 gets 17%
    res_quotas = {112: 12000, 224: 18000}  # total 30K, skip 64
    print(f"  Resolution quotas: {res_quotas}")
    print(f"  Total base features: {n_feats_total}")

    # For each resolution: compute mean/std, train L1 model, get importance
    all_selected = {}  # res -> (local_indices, mean, std)

    for res in RESOLUTIONS:
        X_tr = np.memmap(str(OUTPUT_DIR / f'X_train_{res}.mmap'), dtype='float32', mode='r',
                         shape=(n_train, n_feats_per_res))
        X_va = np.memmap(str(OUTPUT_DIR / f'X_val_{res}.mmap'), dtype='float32', mode='r',
                         shape=(n_val, n_feats_per_res))

        print(f"\n  --- {res}×{res} (quota={res_quotas[res]}) ---")
        res_mean, res_std = online_mean_std(X_tr, n_train, n_feats_per_res)
        res_std = np.maximum(res_std, 1e-8)

        # Train to get feature importance for this resolution
        _, importance = l1_feature_selection(
            X_tr, y_train, X_va, y_val,
            res_mean, res_std, n_feats_per_res, device,
            target_features=res_quotas[res], wd=20.0, epochs=15,
        )

        # Select top features by importance
        quota = res_quotas[res]
        top_local = np.argsort(importance)[::-1][:quota]
        top_local = top_local[importance[top_local] > 1e-8]
        top_local = np.sort(top_local)
        all_selected[res] = (top_local, res_mean, res_std)
        print(f"    Selected {len(top_local)} features")

        del X_tr, X_va

    # ── Step 3: Build selected feature mmap (small) ───────────
    print(f"\n{'='*65}")
    print(f"  STEP 3: Build Selected Feature Matrix")
    print(f"{'='*65}")

    n_selected = sum(len(v[0]) for v in all_selected.values())
    print(f"  Total selected: {n_selected} features")
    print(f"  Selected mmap: {n_train} × {n_selected} × 4 = {n_train * n_selected * 4 / 1e9:.1f} GB")

    # Build combined mean/std for selected features
    sel_mean_parts = []
    sel_std_parts = []
    for res in RESOLUTIONS:
        local_idx, res_mean, res_std = all_selected[res]
        sel_mean_parts.append(res_mean[local_idx])
        sel_std_parts.append(res_std[local_idx])
    sel_mean = np.concatenate(sel_mean_parts)
    sel_std = np.concatenate(sel_std_parts)
    np.save(str(OUTPUT_DIR / 'feat_mean_selected.npy'), sel_mean)
    np.save(str(OUTPUT_DIR / 'feat_std_selected.npy'), sel_std)

    # Extract selected features to compact mmap
    for split, n_total in [('train', n_train), ('val', n_val)]:
        out_path = str(OUTPUT_DIR / f'X_{split}_selected.mmap')
        mmap_out = np.memmap(out_path, dtype='float32', mode='w+', shape=(n_total, n_selected))
        col_offset = 0
        for res in RESOLUTIONS:
            local_idx = all_selected[res][0]
            n_res_sel = len(local_idx)
            X_res = np.memmap(str(OUTPUT_DIR / f'X_{split}_{res}.mmap'), dtype='float32', mode='r',
                              shape=(n_total, n_feats_per_res))
            chunk = 10000
            for start in range(0, n_total, chunk):
                end = min(start + chunk, n_total)
                mmap_out[start:end, col_offset:col_offset + n_res_sel] = np.array(X_res[start:end])[:, local_idx]
            col_offset += n_res_sel
            del X_res
        mmap_out.flush()
        print(f"  {split}: ({n_total}, {n_selected})")

    X_train_sel = np.memmap(str(OUTPUT_DIR / 'X_train_selected.mmap'), dtype='float32', mode='r',
                            shape=(n_train, n_selected))
    X_val_sel = np.memmap(str(OUTPUT_DIR / 'X_val_selected.mmap'), dtype='float32', mode='r',
                          shape=(n_val, n_selected))

    mean_t = torch.tensor(sel_mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(sel_std, dtype=torch.float32, device=device)

    # ── Step 4: Train classifier (sweep) ──────────────────────
    print(f"\n{'='*65}")
    print(f"  STEP 4: Train Classifier ({n_selected} features)")
    print(f"{'='*65}")

    wd_values = [5.0, 10.0, 20.0, 50.0]
    results = {}
    best_val = 0.0
    best_cfg = None

    for wd in wd_values:
        model = nn.Linear(n_selected, NUM_CLASSES).to(device)
        tr, v1, v5, t = train_and_eval(
            model, X_train_sel, y_train, X_val_sel, y_val, mean_t, std_t,
            device, weight_decay=wd, epochs=30,
        )
        label = f'wd={wd}'
        results[label] = (tr, v1, v5, t)
        print(f"  {label:<10}  Train={tr*100:6.2f}%  Val-T1={v1*100:6.2f}%  Val-T5={v5*100:6.2f}%  ({t:.0f}s)")
        if v1 > best_val:
            best_val = v1
            best_cfg = label

    print(f"\n  BEST: {best_cfg}  Val-T1={best_val*100:.2f}%")

    # ── Summary ───────────────────────────────────────────────
    total_time = time.time() - pipeline_start
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  Bodies: {n_bodies}")
    print(f"  Base features: {n_feats_total} ({n_bodies} × {N_ENCODINGS} × {len(RESOLUTIONS)})")
    print(f"  After L1 selection: {n_selected}")
    per_res = {res: len(all_selected[res][0]) for res in RESOLUTIONS}
    print(f"  Per resolution: {per_res}")
    print(f"  Best: {best_cfg}  Val-T1={best_val*100:.2f}%")
    print(f"  Total time: {total_time/3600:.2f} hours")

    # Save
    save = {k: {'train': v[0], 'val_top1': v[1], 'val_top5': v[2]} for k, v in results.items()}
    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump({
            'n_bodies': n_bodies, 'n_encodings': N_ENCODINGS,
            'n_resolutions': len(RESOLUTIONS), 'n_base_features': n_feats_total,
            'n_selected': n_selected, 'per_resolution': per_res,
            'best_cfg': best_cfg, 'best_val_top1': best_val, 'all_results': save,
        }, f, indent=2)
    print(f"  Saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == '__main__':
    main()
