#!/usr/bin/env python3
"""
ImageNet v3 Full Pipeline — From scratch to 50%+

Implements the complete v3 plan:
  Batch 1: L1 formulas + new ops + SPP+BoW encoding + multi-res + interactions
  Batch 2: Hierarchical L2 formulas (L1 feature maps as terminals)
  Batch 3: End-to-end kernel fine-tuning + scale up

Usage:
    # Full pipeline
    python experiments/run_v3_full_pipeline.py --start_batch 1

    # From Batch 2 (L1 formulas already discovered)
    python experiments/run_v3_full_pipeline.py --start_batch 2

    # From Batch 3 (L1+L2 features extracted)
    python experiments/run_v3_full_pipeline.py --start_batch 3
"""

import argparse, gc, json, os, sys, time, itertools
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
BASE_DIR = Path('outputs/imagenet_v3')
NUM_CLASSES = 1000
RESOLUTIONS = [112, 224]
BATCH_SIZES = {112: 512, 224: 256}

SPP_POOLS = ['global_avg_pool', 'global_max_pool', 'global_std_pool',
             'pool_quad_tl', 'pool_quad_tr', 'pool_quad_bl', 'pool_quad_br', 'pool_center']
SPP_POOL_FUNCS = [(name, TENSOR_OPERATORS[name][0]) for name in SPP_POOLS]
N_SPP = len(SPP_POOLS)
N_HIST = 4
N_ENCODINGS = N_SPP + N_HIST  # 12


# ======================================================================
# Shared helpers
# ======================================================================

def build_data_batch(images, device):
    images = images.to(device)
    I_R, I_G, I_B = images[:, 0], images[:, 1], images[:, 2]
    I_GRAY = 0.2989 * I_R + 0.5870 * I_G + 0.1140 * I_B
    Cmax, _ = images.max(dim=1); Cmin, _ = images.min(dim=1)
    delta = Cmax - Cmin + 1e-8
    H = torch.zeros_like(I_R)
    mr = (Cmax == I_R); mg = (Cmax == I_G) & ~mr; mb = ~mr & ~mg
    H[mr] = (((I_G[mr] - I_B[mr]) / delta[mr]) % 6)
    H[mg] = ((I_B[mg] - I_R[mg]) / delta[mg]) + 2
    H[mb] = ((I_R[mb] - I_G[mb]) / delta[mb]) + 4
    H = H / 6.0
    S = torch.where(Cmax > 1e-8, delta / (Cmax + 1e-8), torch.zeros_like(Cmax))
    total = I_R + I_G + I_B + 1e-8
    return {
        'I_R': I_R, 'I_G': I_G, 'I_B': I_B, 'I_GRAY': I_GRAY,
        'I_H': H, 'I_S': S, 'I_r': I_R / total, 'I_g': I_G / total,
        'I_RG': I_R - I_G, 'I_BY': I_B - (I_R + I_G) / 2,
    }


def execute_body(body_str, data_batch):
    """Execute formula body → spatial [B,H,W] or None."""
    tokens = body_str.strip().split()
    stack = []
    for token in tokens:
        if token in data_batch:
            stack.append(data_batch[token])
        elif token in TENSOR_OPERATORS:
            op_func, arity, _ = TENSOR_OPERATORS[token]
            if len(stack) < arity: return None
            operands = [stack.pop() for _ in range(arity)]
            operands.reverse()
            result = op_func(*operands)
            result = torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)
            stack.append(result)
        else: return None
    if len(stack) != 1: return None
    out = torch.clamp(stack[0], -1e4, 1e4)
    return out if out.dim() >= 2 else None


def encode_feature_map(feat_map):
    """SPP (8) + histogram (4) = 12 encodings."""
    B = feat_map.shape[0]
    encodings = []
    for name, pool_func in SPP_POOL_FUNCS:
        try:
            p = pool_func(feat_map)
            p = torch.nan_to_num(p, nan=0.0, posinf=1e4, neginf=-1e4)
            encodings.append(torch.clamp(p, -1e4, 1e4))
        except:
            encodings.append(torch.zeros(B, device=feat_map.device))
    try:
        hist = TENSOR_OPERATORS['patch_histogram_4x4'][0](feat_map)
        hist = torch.nan_to_num(hist, nan=0.0, posinf=1e4, neginf=-1e4)
        hist = torch.clamp(hist, -1e4, 1e4)
        for i in range(N_HIST): encodings.append(hist[:, i])
    except:
        for _ in range(N_HIST): encodings.append(torch.zeros(B, device=feat_map.device))
    return torch.stack(encodings, dim=1)


def online_mean_std(X_mmap, n_total, n_feats, chunk_size=10000):
    s = np.zeros(n_feats, dtype=np.float64)
    sq = np.zeros(n_feats, dtype=np.float64)
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        c = np.nan_to_num(np.array(X_mmap[start:end], dtype=np.float64))
        s += c.sum(axis=0); sq += (c ** 2).sum(axis=0)
    mean = s / n_total
    std = np.sqrt(np.maximum(sq / n_total - mean ** 2, 0.0))
    return mean.astype(np.float32), std.astype(np.float32)


def extract_features_mmap(body_strs, loader, device, mmap_path, n_total, n_feats,
                          data_batch_fn, tag=""):
    """Generic feature extraction to mmap with checkpoint/resume."""
    progress_path = mmap_path + '.progress.json'
    label_path = mmap_path.replace('.mmap', '_labels_partial.npy')

    start_batch = 0; row_offset = 0; all_labels = []
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            p = json.load(f)
        start_batch = p['next_batch']; row_offset = p['row_offset']
        if os.path.exists(label_path):
            all_labels = [np.load(label_path)]
        print(f"    [{tag}] Resuming from batch {start_batch} (row {row_offset}/{n_total})")
        mmap = np.memmap(mmap_path, dtype='float32', mode='r+', shape=(n_total, n_feats))
    else:
        mmap = np.memmap(mmap_path, dtype='float32', mode='w+', shape=(n_total, n_feats))

    total_batches = len(loader)
    t0 = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx < start_batch: continue
        B = images.shape[0]
        data_batch = data_batch_fn(images, device)
        gpu_buf = torch.zeros(B, n_feats, device=device)

        for b_idx, body_str in enumerate(body_strs):
            try:
                feat_map = execute_body(body_str, data_batch)
                if feat_map is not None:
                    gpu_buf[:, b_idx * N_ENCODINGS:(b_idx + 1) * N_ENCODINGS] = encode_feature_map(feat_map)
            except: pass

        batch_feats = gpu_buf.cpu().numpy()
        del gpu_buf, data_batch; torch.cuda.empty_cache()
        end = min(row_offset + B, n_total)
        mmap[row_offset:end] = batch_feats[:end - row_offset]
        all_labels.append(labels.numpy()[:end - row_offset])
        row_offset = end

        if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
            mmap.flush()
            with open(progress_path, 'w') as f:
                json.dump({'next_batch': batch_idx + 1, 'row_offset': row_offset}, f)
            np.save(label_path, np.concatenate(all_labels))

        if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
            print(f"    [{tag}] batch {batch_idx+1}/{total_batches} "
                  f"({row_offset}/{n_total}, {row_offset/n_total*100:.1f}%) {time.time()-t0:.0f}s")

    mmap.flush()
    y = np.concatenate(all_labels)
    for p in [progress_path, label_path]:
        if os.path.exists(p): os.remove(p)
    print(f"    [{tag}] Done: ({n_total}, {n_feats}) in {(time.time()-t0)/60:.1f} min")
    return mmap, y


def train_classifier(X_train, y_train, X_val, y_val, n_feats, feat_mean, feat_std,
                     device, weight_decay=20.0, epochs=30, lr=1e-3, batch_size=1024):
    """Train nn.Linear classifier."""
    model = nn.Linear(n_feats, NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mean_t = torch.tensor(feat_mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(feat_std, dtype=torch.float32, device=device)
    n_train = len(y_train)

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n_train)
        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]
            X_b = torch.tensor(np.array(X_train[idx]), dtype=torch.float32, device=device)
            X_b = torch.nan_to_num(X_b); X_b = (X_b - mean_t) / std_t
            y_b = torch.tensor(y_train[idx], dtype=torch.long, device=device)
            opt.zero_grad(); crit(model(X_b), y_b).backward(); opt.step()
        sched.step()
        if (epoch + 1) % 10 == 0: print(f"      epoch {epoch+1}/{epochs}")

    model.eval()
    c1 = c5 = 0
    n_val = len(y_val)
    with torch.no_grad():
        for s in range(0, n_val, 2048):
            e = min(s + 2048, n_val)
            X_b = torch.tensor(np.array(X_val[s:e]), dtype=torch.float32, device=device)
            X_b = torch.nan_to_num(X_b); X_b = (X_b - mean_t) / std_t
            logits = model(X_b)
            y_b = torch.tensor(y_val[s:e], dtype=torch.long, device=device)
            c1 += (logits.argmax(1) == y_b).sum().item()
            _, tk = logits.topk(5, dim=1)
            c5 += (tk == y_b.unsqueeze(1)).any(1).sum().item()
    return c1 / n_val, c5 / n_val, time.time() - t0, model


# ======================================================================
# Batch 1: L1 features + interactions
# ======================================================================

def run_batch1(device):
    """Extract L1 features at multiple resolutions, add interactions, train."""
    print(f"\n{'='*70}")
    print(f"  BATCH 1: Layer 1 Features + Interactions")
    print(f"{'='*70}")

    out_dir = BASE_DIR / 'batch1'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load L1 bodies — try Phase 2 first, fall back to Phase 1 banks directly
    phase2_path = BASE_DIR / 'phase2' / 'feature_bank' / 'feature_bank.json'
    if phase2_path.exists():
        d = json.load(open(phase2_path))
        all_formulas = d['formulas']
        print(f"  Loaded {len(all_formulas)} formulas from Phase 2")
    else:
        # v3.1: skip Phase 2, load directly from Phase 1 banks
        all_formulas = []
        seen = set()
        for bank_dir in sorted(BASE_DIR.glob('phase1/bank_*/feature_bank')):
            fb_path = bank_dir / 'feature_bank.json'
            if not fb_path.exists(): continue
            bank = json.load(open(fb_path))
            for f in bank['formulas']:
                if f['str'] not in seen:
                    seen.add(f['str'])
                    all_formulas.append(f)
        print(f"  Loaded {len(all_formulas)} formulas directly from Phase 1 (skipping Phase 2)")

    bodies = []
    for f in all_formulas:
        tokens = f['str'].strip().split()
        if tokens[-1] in ROOT_OPERATORS:
            bodies.append(' '.join(tokens[:-1]))
        else:
            bodies.append(f['str'])
    bodies = sorted(set(bodies))
    n_bodies = len(bodies)
    n_feats_per_res = n_bodies * N_ENCODINGS
    print(f"  {n_bodies} bodies × {N_ENCODINGS} encodings × {len(RESOLUTIONS)} res")

    with open(out_dir / 'bodies.json', 'w') as f:
        json.dump(bodies, f)

    # Load kernel bank — prefer pretrained, fall back to saved
    kb = SymbolicKernelBank(device=device)
    for kb_path in [BASE_DIR / 'kernel_bank_pretrained.pt',
                    BASE_DIR / 'phase3_v3' / 'kernel_bank_weights.pt']:
        if kb_path.exists():
            kb.load_state_dict(torch.load(str(kb_path), map_location=device, weights_only=True))
            print(f"  Loaded kernel weights from {kb_path}")
            break
    kb.register_operators(TENSOR_OPERATORS)

    # Extract at each resolution
    for res in RESOLUTIONS:
        bs = BATCH_SIZES[res]
        dm = ImageNetDataModule(data_dir=DATA_DIR, resolution=res, batch_size=bs,
                                num_workers=8, samples_per_class=500)
        dm.setup()
        train_loader = DataLoader(dm.train_dataset, batch_size=bs, shuffle=False,
                                  num_workers=8, pin_memory=True)
        val_loader = dm.get_val_loader()
        n_train, n_val = len(dm.train_dataset), len(dm.val_dataset)

        for split, loader, n_total in [('train', train_loader, n_train), ('val', val_loader, n_val)]:
            mmap_path = str(out_dir / f'X_{split}_{res}.mmap')
            if os.path.exists(mmap_path) and not os.path.exists(mmap_path + '.progress.json'):
                print(f"    [{split}@{res}] Already done, skipping")
                continue
            extract_features_mmap(
                bodies, loader, device, mmap_path, n_total, n_feats_per_res,
                build_data_batch, tag=f"{split}@{res}",
            )
            # Save labels from first completed split
            label_file = out_dir / f'y_{split}.npy'
            if not label_file.exists():
                y = []
                for _, labels in loader: y.append(labels.numpy())
                np.save(str(label_file), np.concatenate(y))

        gc.collect(); torch.cuda.empty_cache()

    # Load labels
    y_train = np.load(str(out_dir / 'y_train.npy'))
    y_val = np.load(str(out_dir / 'y_val.npy'))
    n_train, n_val = len(y_train), len(y_val)

    # Per-resolution L1 feature selection
    print(f"\n  --- L1 Feature Selection ---")
    res_quotas = {112: 12000, 224: 18000}
    all_selected = {}

    for res in RESOLUTIONS:
        X_tr = np.memmap(str(out_dir / f'X_train_{res}.mmap'), dtype='float32', mode='r',
                         shape=(n_train, n_feats_per_res))
        mean, std = online_mean_std(X_tr, n_train, n_feats_per_res)
        std = np.maximum(std, 1e-8)

        # Quick L1 model to rank features
        mean_t = torch.tensor(mean, device=device)
        std_t = torch.tensor(std, device=device)
        model = nn.Linear(n_feats_per_res, NUM_CLASSES).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=20.0)
        crit = nn.CrossEntropyLoss()
        for epoch in range(15):
            model.train()
            perm = np.random.permutation(n_train)
            for start in range(0, n_train, 1024):
                end = min(start + 1024, n_train)
                idx = perm[start:end]
                X_b = torch.tensor(np.array(X_tr[idx]), dtype=torch.float32, device=device)
                X_b = torch.nan_to_num(X_b); X_b = (X_b - mean_t) / std_t
                y_b = torch.tensor(y_train[idx], dtype=torch.long, device=device)
                opt.zero_grad(); crit(model(X_b), y_b).backward(); opt.step()
        importance = model.weight.abs().sum(dim=0).detach().cpu().numpy()
        quota = res_quotas[res]
        top_idx = np.argsort(importance)[::-1][:quota]
        top_idx = np.sort(top_idx[importance[top_idx] > 1e-8])
        all_selected[res] = (top_idx, mean, std)
        print(f"    {res}×{res}: selected {len(top_idx)} features")
        del X_tr, model; gc.collect(); torch.cuda.empty_cache()

    # Build selected feature mmap
    n_selected = sum(len(v[0]) for v in all_selected.values())
    print(f"  Total selected: {n_selected}")

    sel_mean_parts, sel_std_parts = [], []
    for res in RESOLUTIONS:
        idx, m, s = all_selected[res]
        sel_mean_parts.append(m[idx]); sel_std_parts.append(s[idx])
    sel_mean = np.concatenate(sel_mean_parts)
    sel_std = np.concatenate(sel_std_parts)

    for split, n_total in [('train', n_train), ('val', n_val)]:
        sel_path = str(out_dir / f'X_{split}_selected.mmap')
        mmap_out = np.memmap(sel_path, dtype='float32', mode='w+', shape=(n_total, n_selected))
        col = 0
        for res in RESOLUTIONS:
            idx = all_selected[res][0]
            X_res = np.memmap(str(out_dir / f'X_{split}_{res}.mmap'), dtype='float32', mode='r',
                              shape=(n_total, n_feats_per_res))
            for start in range(0, n_total, 10000):
                end = min(start + 10000, n_total)
                mmap_out[start:end, col:col + len(idx)] = np.array(X_res[start:end])[:, idx]
            col += len(idx); del X_res
        mmap_out.flush()

    # Feature interactions: top-300 pairwise products
    print(f"\n  --- Feature Interactions ---")
    X_sel_tr = np.memmap(str(out_dir / 'X_train_selected.mmap'), dtype='float32', mode='r',
                         shape=(n_train, n_selected))

    # Train quick model for importance ranking
    mean_t = torch.tensor(sel_mean, device=device)
    std_t = torch.tensor(np.maximum(sel_std, 1e-8), device=device)
    model = nn.Linear(n_selected, NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=20.0)
    crit = nn.CrossEntropyLoss()
    for epoch in range(10):
        model.train()
        perm = np.random.permutation(n_train)
        for start in range(0, min(100000, n_train), 1024):
            idx = perm[start:start + 1024]
            X_b = torch.tensor(np.array(X_sel_tr[idx]), dtype=torch.float32, device=device)
            X_b = torch.nan_to_num(X_b); X_b = (X_b - mean_t) / std_t
            y_b = torch.tensor(y_train[idx], dtype=torch.long, device=device)
            opt.zero_grad(); crit(model(X_b), y_b).backward(); opt.step()
    importance = model.weight.abs().sum(dim=0).detach().cpu().numpy()
    top300 = np.argsort(importance)[::-1][:300]
    pairs = list(itertools.combinations(range(300), 2))
    n_interact = len(pairs)
    print(f"  Top-300 → {n_interact} interaction features")

    for split, n_total in [('train', n_train), ('val', n_val)]:
        X_sel = np.memmap(str(out_dir / f'X_{split}_selected.mmap'), dtype='float32', mode='r',
                          shape=(n_total, n_selected))
        int_path = str(out_dir / f'X_{split}_interact.mmap')
        mmap_int = np.memmap(int_path, dtype='float32', mode='w+', shape=(n_total, n_interact))
        for start in range(0, n_total, 10000):
            end = min(start + 10000, n_total)
            chunk = np.array(X_sel[start:end])[:, top300]
            chunk = (chunk - sel_mean[top300]) / np.maximum(sel_std[top300], 1e-8)
            for p_idx, (i, j) in enumerate(pairs):
                mmap_int[start:end, p_idx] = chunk[:, i] * chunk[:, j]
        mmap_int.flush()
        print(f"    {split}: ({n_total}, {n_interact})")

    # Save Batch 1 metadata
    np.save(str(out_dir / 'sel_mean.npy'), sel_mean)
    np.save(str(out_dir / 'sel_std.npy'), sel_std)
    np.savez(str(out_dir / 'interact_meta.npz'), top300=top300, n_interact=n_interact)
    json.dump({'n_bodies': n_bodies, 'n_selected': n_selected, 'n_interact': n_interact,
               'resolutions': RESOLUTIONS}, open(out_dir / 'batch1_meta.json', 'w'), indent=2)

    # Train classifier on base + interactions
    print(f"\n  --- Train Batch 1 Classifier ---")
    # Compute interaction mean/std
    X_int_tr = np.memmap(str(out_dir / 'X_train_interact.mmap'), dtype='float32', mode='r',
                         shape=(n_train, n_interact))
    int_mean, int_std = online_mean_std(X_int_tr, n_train, n_interact)
    int_std = np.maximum(int_std, 1e-8)

    # For now, train on base features only (interactions added via concat during training)
    # Final feature selection on combined will be done after L2 in Batch 2
    for wd in [10.0, 20.0, 50.0]:
        v1, v5, t, _ = train_classifier(
            X_sel_tr, y_train,
            np.memmap(str(out_dir / 'X_val_selected.mmap'), dtype='float32', mode='r',
                      shape=(n_val, n_selected)),
            y_val, n_selected, sel_mean, np.maximum(sel_std, 1e-8), device,
            weight_decay=wd, epochs=30,
        )
        print(f"    wd={wd:<6} Val-T1={v1*100:6.2f}% Val-T5={v5*100:6.2f}% ({t:.0f}s)")

    print(f"\n  Batch 1 complete. Features saved to {out_dir}")
    return out_dir


# ======================================================================
# Batch 2: Layer 2 Hierarchical Formulas
# ======================================================================

def run_batch2(device, batch1_dir):
    """Forward selection + Phase 1B + L2 extraction + combined training."""
    print(f"\n{'='*70}")
    print(f"  BATCH 2: Layer 2 Hierarchical Formulas")
    print(f"{'='*70}")

    out_dir = BASE_DIR / 'batch2'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Forward selection (already done if l1_selected_bodies.json exists)
    l2_dir = BASE_DIR / 'layer2'
    selected_path = l2_dir / 'l1_selected_bodies.json'
    if selected_path.exists():
        l1_selected = json.load(open(selected_path))
        print(f"  Loaded {len(l1_selected)} pre-selected L1 bodies")
    else:
        print(f"  ERROR: Run forward selection first (run_batch2_layer2.py)")
        return None

    # Step 2: Phase 1B — Run RL with L1 terminals
    # Check if Phase 1B already ran
    phase1b_meta = l2_dir / 'phase1' / 'phase1_meta.json'
    if not phase1b_meta.exists():
        print(f"\n  Running Phase 1B (Layer 2 RL)...")
        os.system(f"PYTHONUNBUFFERED=1 python experiments/train_imagenet_pipeline.py "
                  f"--config configs/tensor_vsr_imagenet_v3_layer2.yaml "
                  f"--device {device} "
                  f"--output_dir {l2_dir} "
                  f"--start_phase 1")
    else:
        print(f"  Phase 1B already complete")

    # Step 3: Load L2 formulas (skip Phase 2 for L2 — use all Phase 1B formulas)
    l2_bodies = []
    for bank_dir in sorted(l2_dir.glob('phase1/bank_*')):
        fb_path = bank_dir / 'feature_bank' / 'feature_bank.json'
        if not fb_path.exists(): continue
        bank = json.load(open(fb_path))
        for f in bank['formulas']:
            tokens = f['str'].strip().split()
            if tokens[-1] in ROOT_OPERATORS:
                body = ' '.join(tokens[:-1])
            else:
                body = f['str']
            l2_bodies.append(body)
    l2_bodies = sorted(set(l2_bodies))
    print(f"  L2 bodies: {len(l2_bodies)}")
    with open(out_dir / 'l2_bodies.json', 'w') as f:
        json.dump(l2_bodies, f)

    # Step 4: Extract L2 features
    # L2 formulas use L1 terminals — need to compute L1 feature maps on the fly
    n_l2 = len(l2_bodies)
    n_feats_l2_per_res = n_l2 * N_ENCODINGS

    def build_data_batch_l2(images, device):
        """Build data batch with L1 feature maps as extra terminals."""
        base = build_data_batch(images, device)
        for i, body in enumerate(l1_selected):
            try:
                fm = execute_body(body, base)
                if fm is not None:
                    base[f'L1_{i}'] = fm
                else:
                    base[f'L1_{i}'] = torch.zeros_like(base['I_R'])
            except:
                base[f'L1_{i}'] = torch.zeros_like(base['I_R'])
        return base

    for res in RESOLUTIONS:
        bs = BATCH_SIZES[res]
        dm = ImageNetDataModule(data_dir=DATA_DIR, resolution=res, batch_size=bs,
                                num_workers=8, samples_per_class=500)
        dm.setup()
        train_loader = DataLoader(dm.train_dataset, batch_size=bs, shuffle=False,
                                  num_workers=8, pin_memory=True)
        val_loader = dm.get_val_loader()
        n_train, n_val = len(dm.train_dataset), len(dm.val_dataset)

        for split, loader, n_total in [('train', train_loader, n_train), ('val', val_loader, n_val)]:
            mmap_path = str(out_dir / f'X_l2_{split}_{res}.mmap')
            if os.path.exists(mmap_path) and not os.path.exists(mmap_path + '.progress.json'):
                print(f"    [L2 {split}@{res}] Already done, skipping")
                continue
            extract_features_mmap(
                l2_bodies, loader, device, mmap_path, n_total, n_feats_l2_per_res,
                build_data_batch_l2, tag=f"L2 {split}@{res}",
            )
        gc.collect(); torch.cuda.empty_cache()

    # Step 5: Combine L1 + L2 features
    print(f"\n  --- Combining L1 + L2 features ---")
    y_train = np.load(str(batch1_dir / 'y_train.npy'))
    y_val = np.load(str(batch1_dir / 'y_val.npy'))
    n_train, n_val = len(y_train), len(y_val)

    # Load Batch 1 selected features
    b1_meta = json.load(open(batch1_dir / 'batch1_meta.json'))
    n_l1_selected = b1_meta['n_selected']

    # L2: per-resolution selection (same approach as Batch 1)
    l2_quotas = {112: 6000, 224: 9000}
    l2_selected = {}
    for res in RESOLUTIONS:
        X_tr = np.memmap(str(out_dir / f'X_l2_train_{res}.mmap'), dtype='float32', mode='r',
                         shape=(n_train, n_feats_l2_per_res))
        mean, std = online_mean_std(X_tr, n_train, n_feats_l2_per_res)
        std = np.maximum(std, 1e-8)
        mean_t = torch.tensor(mean, device=device)
        std_t = torch.tensor(std, device=device)
        model = nn.Linear(n_feats_l2_per_res, NUM_CLASSES).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=20.0)
        crit = nn.CrossEntropyLoss()
        for epoch in range(15):
            model.train()
            perm = np.random.permutation(n_train)
            for start in range(0, n_train, 1024):
                end = min(start + 1024, n_train)
                idx = perm[start:end]
                X_b = torch.tensor(np.array(X_tr[idx]), dtype=torch.float32, device=device)
                X_b = torch.nan_to_num(X_b); X_b = (X_b - mean_t) / std_t
                y_b = torch.tensor(y_train[idx], dtype=torch.long, device=device)
                opt.zero_grad(); crit(model(X_b), y_b).backward(); opt.step()
        importance = model.weight.abs().sum(dim=0).detach().cpu().numpy()
        quota = l2_quotas[res]
        top_idx = np.argsort(importance)[::-1][:quota]
        top_idx = np.sort(top_idx[importance[top_idx] > 1e-8])
        l2_selected[res] = (top_idx, mean, std)
        print(f"    L2 {res}×{res}: selected {len(top_idx)}")
        del X_tr, model; gc.collect(); torch.cuda.empty_cache()

    n_l2_selected = sum(len(v[0]) for v in l2_selected.values())
    n_combined = n_l1_selected + n_l2_selected
    print(f"  Combined: L1={n_l1_selected} + L2={n_l2_selected} = {n_combined}")

    # Build combined mmap
    l2_mean_parts, l2_std_parts = [], []
    for res in RESOLUTIONS:
        idx, m, s = l2_selected[res]
        l2_mean_parts.append(m[idx]); l2_std_parts.append(s[idx])

    l1_mean = np.load(str(batch1_dir / 'sel_mean.npy'))
    l1_std = np.load(str(batch1_dir / 'sel_std.npy'))
    combined_mean = np.concatenate([l1_mean, *l2_mean_parts])
    combined_std = np.concatenate([np.maximum(l1_std, 1e-8), *[np.maximum(s, 1e-8) for s in l2_std_parts]])

    for split, n_total in [('train', n_train), ('val', n_val)]:
        comb_path = str(out_dir / f'X_combined_{split}.mmap')
        mmap_out = np.memmap(comb_path, dtype='float32', mode='w+', shape=(n_total, n_combined))
        # Copy L1 selected
        X_l1 = np.memmap(str(batch1_dir / f'X_{split}_selected.mmap'), dtype='float32', mode='r',
                         shape=(n_total, n_l1_selected))
        for start in range(0, n_total, 10000):
            end = min(start + 10000, n_total)
            mmap_out[start:end, :n_l1_selected] = np.array(X_l1[start:end])
        del X_l1
        # Copy L2 selected
        col = n_l1_selected
        for res in RESOLUTIONS:
            idx = l2_selected[res][0]
            X_l2 = np.memmap(str(out_dir / f'X_l2_train_{res}.mmap' if split == 'train'
                              else out_dir / f'X_l2_val_{res}.mmap'),
                             dtype='float32', mode='r', shape=(n_total, n_feats_l2_per_res))
            for start in range(0, n_total, 10000):
                end = min(start + 10000, n_total)
                mmap_out[start:end, col:col + len(idx)] = np.array(X_l2[start:end])[:, idx]
            col += len(idx); del X_l2
        mmap_out.flush()
        print(f"    {split} combined: ({n_total}, {n_combined})")

    np.save(str(out_dir / 'combined_mean.npy'), combined_mean)
    np.save(str(out_dir / 'combined_std.npy'), combined_std)

    # Step 6: Train combined classifier
    print(f"\n  --- Train Combined Classifier ---")
    X_comb_tr = np.memmap(str(out_dir / 'X_combined_train.mmap'), dtype='float32', mode='r',
                          shape=(n_train, n_combined))
    X_comb_va = np.memmap(str(out_dir / 'X_combined_val.mmap'), dtype='float32', mode='r',
                          shape=(n_val, n_combined))

    for wd in [10.0, 20.0, 50.0]:
        v1, v5, t, model = train_classifier(
            X_comb_tr, y_train, X_comb_va, y_val,
            n_combined, combined_mean, combined_std, device,
            weight_decay=wd, epochs=30,
        )
        print(f"    wd={wd:<6} Val-T1={v1*100:6.2f}% Val-T5={v5*100:6.2f}% ({t:.0f}s)")

    # Save best model
    torch.save(model.state_dict(), str(out_dir / 'classifier.pt'))
    json.dump({
        'n_l1_selected': n_l1_selected, 'n_l2_selected': n_l2_selected,
        'n_combined': n_combined,
    }, open(out_dir / 'batch2_meta.json', 'w'), indent=2)

    print(f"\n  Batch 2 complete. Features saved to {out_dir}")
    return out_dir


# ======================================================================
# Batch 3: End-to-end kernel fine-tuning + scale up
# ======================================================================

def run_batch3(device, batch2_dir):
    """Fine-tune learnable kernels + classifier jointly, then re-extract."""
    print(f"\n{'='*70}")
    print(f"  BATCH 3: End-to-End Kernel Fine-Tuning + Scale Up")
    print(f"{'='*70}")

    out_dir = BASE_DIR / 'batch3'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all formula bodies
    b1_bodies = json.load(open(BASE_DIR / 'batch1' / 'bodies.json'))
    l2_bodies_path = BASE_DIR / 'batch2' / 'l2_bodies.json'
    l2_bodies = json.load(open(l2_bodies_path)) if l2_bodies_path.exists() else []
    l1_selected_path = BASE_DIR / 'layer2' / 'l1_selected_bodies.json'
    l1_selected = json.load(open(l1_selected_path)) if l1_selected_path.exists() else []
    all_bodies = b1_bodies + l2_bodies

    # Load Batch 2 metadata
    b2_meta = json.load(open(batch2_dir / 'batch2_meta.json'))
    n_combined = b2_meta['n_combined']
    combined_mean = np.load(str(batch2_dir / 'combined_mean.npy'))
    combined_std = np.load(str(batch2_dir / 'combined_std.npy'))

    # Load kernel bank and switch to fine-tune mode
    kb = SymbolicKernelBank(device=device)
    kb_path = BASE_DIR / 'phase3_v3' / 'kernel_bank_weights.pt'
    if kb_path.exists():
        kb.load_state_dict(torch.load(str(kb_path), weights_only=True))
    kb.finetune_mode = True
    kb.register_operators(TENSOR_OPERATORS)

    # Load pre-trained classifier (warm start)
    classifier = nn.Linear(n_combined, NUM_CLASSES).to(device)
    cls_path = batch2_dir / 'classifier.pt'
    if cls_path.exists():
        classifier.load_state_dict(torch.load(str(cls_path), weights_only=True))

    mean_t = torch.tensor(combined_mean, device=device)
    std_t = torch.tensor(np.maximum(combined_std, 1e-8), device=device)

    # ── Step 16: Online fine-tuning ───────────────────────────
    # For each batch: images → execute bodies → encode → classify → backprop to kernels
    # Use a SUBSET of bodies for speed (top-N by importance from Batch 2)
    # The key bodies that use learnable kernels benefit from fine-tuning
    print(f"\n  Step 16: Online fine-tuning (kernels + classifier)")
    print(f"  Bodies: {len(b1_bodies)} L1 + {len(l2_bodies)} L2 = {len(all_bodies)}")

    optimizer = torch.optim.AdamW([
        {'params': classifier.parameters(), 'lr': 1e-3, 'weight_decay': 10.0},
        {'params': kb.classic_3x3, 'lr': 1e-5},  # classic: small LR
        {'params': kb.classic_7x7, 'lr': 1e-5},
        {'params': kb.conv3x3, 'lr': 1e-4},       # learned: larger LR
        {'params': kb.conv5x5, 'lr': 1e-4},
    ])
    crit = nn.CrossEntropyLoss()

    # Use smaller batch + fewer bodies for online training speed
    dm = ImageNetDataModule(data_dir=DATA_DIR, resolution=112, batch_size=64,
                            num_workers=8, samples_per_class=500)
    dm.setup()
    train_loader = DataLoader(dm.train_dataset, batch_size=64, shuffle=True,
                              num_workers=8, pin_memory=True)

    def online_extract_features(images, device):
        """Execute all bodies on a batch, return feature tensor [B, n_combined]."""
        data_batch = build_data_batch(images, device)
        # Add L1 terminals for L2 bodies
        for i, body in enumerate(l1_selected):
            try:
                fm = execute_body(body, data_batch)
                data_batch[f'L1_{i}'] = fm if fm is not None else torch.zeros_like(data_batch['I_R'])
            except:
                data_batch[f'L1_{i}'] = torch.zeros_like(data_batch['I_R'])

        B = images.shape[0]
        feats = torch.zeros(B, n_combined, device=device)
        col = 0
        for body in all_bodies:
            try:
                fm = execute_body(body, data_batch)
                if fm is not None:
                    encoded = encode_feature_map(fm)  # [B, 12]
                    feats[:, col:col + N_ENCODINGS] = encoded
            except:
                pass
            col += N_ENCODINGS
        return feats

    for epoch in range(10):
        classifier.train(); kb.train()
        total_loss = 0; n_batches = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            feats = online_extract_features(images, device)
            feats = torch.nan_to_num(feats)
            feats = (feats - mean_t) / std_t

            logits = classifier(feats)
            loss = crit(logits, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item(); n_batches += 1

            if (batch_idx + 1) % 100 == 0:
                print(f"    Epoch {epoch+1}, batch {batch_idx+1}: loss={total_loss/n_batches:.4f}")

        avg_loss = total_loss / max(n_batches, 1)
        print(f"    Epoch {epoch+1}/10: avg_loss={avg_loss:.4f}")

    # Save fine-tuned kernels
    torch.save(kb.state_dict(), str(out_dir / 'kernel_bank_finetuned.pt'))
    torch.save(classifier.state_dict(), str(out_dir / 'classifier_finetuned.pt'))
    print(f"  Saved fine-tuned kernel bank and classifier")

    # ── Step 17: Re-extract features with updated kernels ─────
    print(f"\n  Step 17: Re-extract features with fine-tuned kernels")
    kb.finetune_mode = False  # Switch back to detached mode for extraction
    kb.register_operators(TENSOR_OPERATORS)  # Re-register with updated weights

    # Re-run Batch 1 + Batch 2 extraction with new kernels
    # Save updated kernel weights for the extraction scripts to load
    torch.save(kb.state_dict(), str(out_dir / 'kernel_bank_finetuned.pt'))

    # Re-extract L1 features
    for res in RESOLUTIONS:
        bs = BATCH_SIZES[res]
        dm = ImageNetDataModule(data_dir=DATA_DIR, resolution=res, batch_size=bs,
                                num_workers=8, samples_per_class=500)
        dm.setup()
        train_loader = DataLoader(dm.train_dataset, batch_size=bs, shuffle=False,
                                  num_workers=8, pin_memory=True)
        val_loader = dm.get_val_loader()
        n_train, n_val = len(dm.train_dataset), len(dm.val_dataset)
        n_feats = len(b1_bodies) * N_ENCODINGS

        for split, loader, n_total in [('train', train_loader, n_train), ('val', val_loader, n_val)]:
            mmap_path = str(out_dir / f'X_l1_{split}_{res}.mmap')
            if os.path.exists(mmap_path) and not os.path.exists(mmap_path + '.progress.json'):
                print(f"    [L1 {split}@{res}] Already done, skipping")
                continue
            extract_features_mmap(b1_bodies, loader, device, mmap_path, n_total, n_feats,
                                  build_data_batch, tag=f"L1 {split}@{res}")
        gc.collect(); torch.cuda.empty_cache()

    # Re-extract L2 features
    if l2_bodies:
        def build_data_batch_l2(images, device):
            base = build_data_batch(images, device)
            for i, body in enumerate(l1_selected):
                try:
                    fm = execute_body(body, base)
                    base[f'L1_{i}'] = fm if fm is not None else torch.zeros_like(base['I_R'])
                except:
                    base[f'L1_{i}'] = torch.zeros_like(base['I_R'])
            return base

        n_feats_l2 = len(l2_bodies) * N_ENCODINGS
        for res in RESOLUTIONS:
            bs = BATCH_SIZES[res]
            dm = ImageNetDataModule(data_dir=DATA_DIR, resolution=res, batch_size=bs,
                                    num_workers=8, samples_per_class=500)
            dm.setup()
            train_loader = DataLoader(dm.train_dataset, batch_size=bs, shuffle=False,
                                      num_workers=8, pin_memory=True)
            val_loader = dm.get_val_loader()
            n_train, n_val = len(dm.train_dataset), len(dm.val_dataset)

            for split, loader, n_total in [('train', train_loader, n_train), ('val', val_loader, n_val)]:
                mmap_path = str(out_dir / f'X_l2_{split}_{res}.mmap')
                if os.path.exists(mmap_path) and not os.path.exists(mmap_path + '.progress.json'):
                    print(f"    [L2 {split}@{res}] Already done, skipping")
                    continue
                extract_features_mmap(l2_bodies, loader, device, mmap_path, n_total, n_feats_l2,
                                      build_data_batch_l2, tag=f"L2 {split}@{res}")
            gc.collect(); torch.cuda.empty_cache()

    # ── Step 18: Retrain classifier on re-extracted features ──
    print(f"\n  Step 18: Retrain classifier on fine-tuned features")
    y_train = np.load(str(BASE_DIR / 'batch1' / 'y_train.npy'))
    y_val = np.load(str(BASE_DIR / 'batch1' / 'y_val.npy'))
    n_train, n_val = len(y_train), len(y_val)

    # TODO: L1 selection + combine + train (same as Batch 1+2 but with new features)
    # For now, train on L1 features only as a quick check
    n_feats_l1 = len(b1_bodies) * N_ENCODINGS
    for res in RESOLUTIONS:
        X_tr = np.memmap(str(out_dir / f'X_l1_train_{res}.mmap'), dtype='float32', mode='r',
                         shape=(n_train, n_feats_l1))
        mean, std = online_mean_std(X_tr, n_train, n_feats_l1)
        std = np.maximum(std, 1e-8)
        v1, v5, t, _ = train_classifier(X_tr, y_train,
            np.memmap(str(out_dir / f'X_l1_val_{res}.mmap'), dtype='float32', mode='r',
                      shape=(n_val, n_feats_l1)),
            y_val, n_feats_l1, mean, std, device, weight_decay=20.0, epochs=30)
        print(f"    L1@{res} (finetuned): Val-T1={v1*100:.2f}% Val-T5={v5*100:.2f}%")
        del X_tr

    print(f"\n  Batch 3 complete. Outputs saved to {out_dir}")
    return out_dir


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='ImageNet v3 Full Pipeline')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--start_batch', type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()

    device = args.device
    torch.manual_seed(42); np.random.seed(42)

    if args.start_batch <= 1:
        batch1_dir = run_batch1(device)
    else:
        batch1_dir = BASE_DIR / 'batch1'

    if args.start_batch <= 2:
        batch2_dir = run_batch2(device, batch1_dir)
    else:
        batch2_dir = BASE_DIR / 'batch2'

    if args.start_batch <= 3:
        run_batch3(device, batch2_dir)

    print(f"\n{'='*70}")
    print(f"  V3 PIPELINE COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
