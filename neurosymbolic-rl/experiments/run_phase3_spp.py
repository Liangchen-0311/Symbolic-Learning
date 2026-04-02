#!/usr/bin/env python3
"""
Phase 3 with SPP: Strip final pooling from each formula, apply 8 pooling operators
to the intermediate feature map → 6000 bodies × 8 pools = 48,000 features.
"""

import gc, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.imagenet_loader import ImageNetDataModule, IMAGENET_SUPERCLASS_NAMES
from src.symbolic.tensor_operators import TENSOR_OPERATORS, ROOT_OPERATORS

# ======================================================================
# Config
# ======================================================================

DATA_DIR = '/workspace/neurosymbolic-rl/data/imagenet'
OUTPUT_DIR = Path('outputs/imagenet_v2/phase3_spp')
NUM_CLASSES = 1000

# 8 spatial pooling operators for SPP
SPP_POOLS = [
    'global_avg_pool',
    'global_max_pool',
    'global_std_pool',
    'pool_quad_tl',
    'pool_quad_tr',
    'pool_quad_bl',
    'pool_quad_br',
    'pool_center',
]
N_POOLS = len(SPP_POOLS)

# Pre-resolve pool functions
SPP_POOL_FUNCS = [(name, TENSOR_OPERATORS[name][0]) for name in SPP_POOLS]


# ======================================================================
# Helpers
# ======================================================================

def build_data_batch(images, device):
    images = images.to(device)
    I_R, I_G, I_B = images[:, 0], images[:, 1], images[:, 2]
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
    return {'I_R': I_R, 'I_G': I_G, 'I_B': I_B, 'I_GRAY': I_GRAY, 'I_H': H, 'I_S': S}


def execute_body(body_str, data_batch):
    """Execute formula body (without final pooling). Returns spatial [B, H, W] or None."""
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
    # Must be spatial (dim > 1) for SPP to make sense
    if out.dim() < 2:
        return None
    return out


def apply_spp(feature_map):
    """Apply 8 pooling operators to spatial feature map [B, H, W] → [B, 8]."""
    results = []
    for name, pool_func in SPP_POOL_FUNCS:
        try:
            pooled = pool_func(feature_map)  # [B]
            pooled = torch.nan_to_num(pooled, nan=0.0, posinf=1e4, neginf=-1e4)
            pooled = torch.clamp(pooled, -1e4, 1e4)
            results.append(pooled)
        except Exception:
            results.append(torch.zeros(feature_map.shape[0], device=feature_map.device))
    return torch.stack(results, dim=1)  # [B, 8]


class LRUCache:
    def __init__(self, max_size=500):
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


def execute_body_with_lru(body_str, data_batch, lru_cache):
    """Execute formula body with LRU caching. Returns spatial [B, H, W] or None."""
    tokens = body_str.strip().split()
    stack = []
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

            cached = lru_cache.get(prefix_key)
            if cached is not None:
                for _ in range(arity):
                    stack.pop()
                stack.append(cached)
                continue

            operands = [stack.pop() for _ in range(arity)]
            operands.reverse()
            result = op_func(*operands)
            result = torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)

            if result.dim() > 1:
                lru_cache.put(prefix_key, result)

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


# ======================================================================
# Feature Extraction with SPP
# ======================================================================

def extract_features_spp(body_strs, loader, device, mmap_path, n_total, n_bodies, tag=""):
    """Extract SPP features: each body → 8 pooled scalars."""
    n_feats = n_bodies * N_POOLS
    progress_path = mmap_path + '.progress.json'
    label_path = mmap_path.replace('.mmap', '_labels_partial.npy')

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
    lru = LRUCache(max_size=500)

    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx < start_batch:
            continue

        B = images.shape[0]
        data_batch = build_data_batch(images, device)
        lru.clear()

        # Accumulate on GPU: [B, n_bodies * N_POOLS]
        gpu_buf = torch.zeros(B, n_feats, device=device)

        for b_idx, body_str in enumerate(body_strs):
            try:
                feat_map = execute_body_with_lru(body_str, data_batch, lru)
                if feat_map is not None:
                    spp_out = apply_spp(feat_map)  # [B, 8]
                    gpu_buf[:, b_idx * N_POOLS:(b_idx + 1) * N_POOLS] = spp_out
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
            pct = row_offset / n_total * 100
            print(f"    [{tag}] batch {batch_idx+1}/{total_batches}  "
                  f"({row_offset}/{n_total} imgs, {pct:.1f}%)  "
                  f"LRU hit={lru.hit_rate:.1%}")

    mmap.flush()
    y = np.concatenate(all_labels, axis=0)
    for p in [progress_path, label_path]:
        if os.path.exists(p):
            os.remove(p)

    print(f"    [{tag}] Final LRU: hits={lru.hits}, misses={lru.misses}, rate={lru.hit_rate:.1%}")
    return mmap, y


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
# Training
# ======================================================================

def train_and_eval(model, X_train, y_train, X_val, y_val, mean_t, std_t,
                   device, weight_decay, epochs=20, lr=1e-3, batch_size=1024):
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
        if (epoch + 1) % 5 == 0:
            print(f"      epoch {epoch+1}/{epochs}")
    elapsed = time.time() - t0

    model.eval()
    def eval_split(X_mmap, y):
        c1 = c5 = 0
        n = len(y)
        with torch.no_grad():
            for start in range(0, n, 2048):
                end = min(start + 2048, n)
                X_b = torch.tensor(np.array(X_mmap[start:end]), dtype=torch.float32, device=device)
                X_b = torch.nan_to_num(X_b)
                X_b = (X_b - mean_t) / std_t
                logits = model(X_b)
                y_b = torch.tensor(y[start:end], dtype=torch.long, device=device)
                c1 += (logits.argmax(1) == y_b).sum().item()
                _, tk = logits.topk(min(5, NUM_CLASSES), dim=1)
                c5 += (tk == y_b.unsqueeze(1)).any(1).sum().item()
        return c1 / n, c5 / n

    train1, _ = eval_split(X_train, y_train)
    val1, val5 = eval_split(X_val, y_val)
    return train1, val1, val5, elapsed


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
    print(f"SPP pools: {SPP_POOLS}")

    # ── Load formulas, strip final pooling to get bodies ──────
    dedup_path = Path('outputs/imagenet_v2/phase3_v2/dedup_formulas.json')
    with open(dedup_path) as f:
        dedup_formulas = json.load(f)
    dedup_formulas.sort(key=lambda x: x['accuracy'], reverse=True)
    dedup_formulas = dedup_formulas[:6000]

    bodies = []
    for entry in dedup_formulas:
        tokens = entry['str'].strip().split()
        if tokens[-1] in ROOT_OPERATORS:
            bodies.append(' '.join(tokens[:-1]))
        else:
            bodies.append(entry['str'])  # keep as-is if no root pooling

    # Sort lexicographically for LRU cache
    bodies_sorted = sorted(set(bodies))  # deduplicate just in case
    n_bodies = len(bodies_sorted)
    n_feats = n_bodies * N_POOLS

    print(f"\n  {n_bodies} unique bodies × {N_POOLS} pools = {n_feats} features")

    with open(OUTPUT_DIR / 'bodies_sorted.json', 'w') as f:
        json.dump(bodies_sorted, f)

    # ── Load ImageNet ─────────────────────────────────────────
    print(f"  Loading ImageNet at 224×224 (train: 200/class, val: full)...")
    dm = ImageNetDataModule(
        data_dir=DATA_DIR, resolution=224, batch_size=256,
        num_workers=8, samples_per_class=200,
    )
    dm.setup()
    train_loader = DataLoader(
        dm.train_dataset, batch_size=256,
        shuffle=False, num_workers=8, pin_memory=True, drop_last=False,
    )
    val_loader = dm.get_val_loader()
    n_train = len(dm.train_dataset)
    n_val = len(dm.val_dataset)
    print(f"  Train: {n_train}, Val: {n_val}, Features: {n_feats}")

    # ── Extract SPP features ──────────────────────────────────
    train_mmap_path = str(OUTPUT_DIR / 'X_train_spp.mmap')
    val_mmap_path = str(OUTPUT_DIR / 'X_val_spp.mmap')

    print(f"\n  Extracting train SPP features...")
    t0 = time.time()
    X_train, y_train = extract_features_spp(
        bodies_sorted, train_loader, device,
        mmap_path=train_mmap_path, n_total=n_train, n_bodies=n_bodies, tag="train",
    )
    np.save(str(OUTPUT_DIR / 'y_train.npy'), y_train)
    print(f"  Train done: ({n_train}, {n_feats}) in {(time.time()-t0)/60:.1f} min")

    gc.collect(); torch.cuda.empty_cache()

    print(f"\n  Extracting val SPP features...")
    t0 = time.time()
    X_val, y_val = extract_features_spp(
        bodies_sorted, val_loader, device,
        mmap_path=val_mmap_path, n_total=n_val, n_bodies=n_bodies, tag="val",
    )
    np.save(str(OUTPUT_DIR / 'y_val.npy'), y_val)
    print(f"  Val done: ({n_val}, {n_feats}) in {(time.time()-t0)/60:.1f} min")

    # ── Standardize ───────────────────────────────────────────
    print(f"\n  Computing feature statistics...")
    feat_mean, feat_std = online_mean_std(X_train, n_train, n_feats)
    feat_std = np.maximum(feat_std, 1e-8)
    np.save(str(OUTPUT_DIR / 'feat_mean.npy'), feat_mean)
    np.save(str(OUTPUT_DIR / 'feat_std.npy'), feat_std)
    mean_t = torch.tensor(feat_mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(feat_std, dtype=torch.float32, device=device)

    # ── Train: weight_decay sweep ─────────────────────────────
    print(f"\n{'='*65}")
    print(f"  nn.Linear({n_feats}, {NUM_CLASSES}) — weight_decay sweep")
    print(f"{'='*65}")

    wd_values = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    results = {}
    best_val = 0.0
    best_cfg = None

    for wd in wd_values:
        model = nn.Linear(n_feats, NUM_CLASSES).to(device)
        tr, v1, v5, t = train_and_eval(
            model, X_train, y_train, X_val, y_val, mean_t, std_t,
            device, weight_decay=wd, epochs=20,
        )
        results[f'wd={wd}'] = (tr, v1, v5, t)
        print(f"  wd={wd:<6}  Train={tr*100:6.2f}%  Val-T1={v1*100:6.2f}%  Val-T5={v5*100:6.2f}%  ({t:.0f}s)")
        if v1 > best_val:
            best_val = v1
            best_cfg = f'wd={wd}'

    print(f"\n  BEST: {best_cfg}  Val-T1={best_val*100:.2f}%")

    # ── Train: dropout + wd sweep ─────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Dropout + Weight Decay sweep")
    print(f"{'='*65}")

    for dp in [0.5, 0.7, 0.9]:
        for wd in [2.0, 5.0, 10.0]:
            model = nn.Sequential(
                nn.Dropout(p=dp),
                nn.Linear(n_feats, NUM_CLASSES),
            ).to(device)
            tr, v1, v5, t = train_and_eval(
                model, X_train, y_train, X_val, y_val, mean_t, std_t,
                device, weight_decay=wd, epochs=20,
            )
            label = f'dp={dp}_wd={wd}'
            results[label] = (tr, v1, v5, t)
            print(f"  {label:<16}  Train={tr*100:6.2f}%  Val-T1={v1*100:6.2f}%  Val-T5={v5*100:6.2f}%  ({t:.0f}s)")
            if v1 > best_val:
                best_val = v1
                best_cfg = label

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  Previous (no SPP, 6000 feats):  Val-T1=17.27%  Val-T5=33.64%")
    print(f"  SPP best ({best_cfg}):  Val-T1={best_val*100:.2f}%")
    print(f"  Features: {n_feats} ({n_bodies} bodies × {N_POOLS} pools)")

    total_time = time.time() - pipeline_start
    print(f"\n  Total time: {total_time/3600:.2f} hours")

    # Save
    save = {k: {'train': v[0], 'val_top1': v[1], 'val_top5': v[2]} for k, v in results.items()}
    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump({'n_bodies': n_bodies, 'n_pools': N_POOLS, 'n_feats': n_feats,
                   'best_cfg': best_cfg, 'best_val_top1': best_val,
                   'all_results': save}, f, indent=2)
    print(f"  Saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == '__main__':
    main()
