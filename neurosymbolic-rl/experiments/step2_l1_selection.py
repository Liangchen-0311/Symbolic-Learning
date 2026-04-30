#!/usr/bin/env python3
"""
Step 2 replacement: L1 Selection for top-100 complementary L1 bodies.

Instead of slow greedy forward selection (500 candidates × 5 SGD steps × 100 rounds),
use a single linear model with L1 penalty to rank all bodies by importance.

Pipeline:
  1. Extract 8 SPP features per body for all 14K+ bodies on 20K train images → mmap
  2. Train nn.Linear(n_bodies*8, 1000) with AdamW + L1 penalty, 20 epochs
  3. Rank bodies by importance: sum(abs(W[:, 8*i : 8*i+8]))
  4. Take top-100 → save to l1_selected_bodies.json
  5. Print standalone val accuracy distribution as sanity check
"""

import gc, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.imagenet_loader import ImageNetDataModule
from src.symbolic.tensor_operators import (
    TensorOperators, TENSOR_OPERATORS, ROOT_OPERATORS, SymbolicKernelBank,
)

# ── Config ──
DATA_DIR = '/workspace/neurosymbolic-rl/data/imagenet'
BASE_DIR = Path('outputs/imagenet_v3_2')
NUM_CLASSES = 1000
N_SELECT = 100
L1_COEF = 0.001

SPP_POOLS = [
    'global_avg_pool', 'global_max_pool', 'global_std_pool',
    'pool_quad_tl', 'pool_quad_tr', 'pool_quad_bl', 'pool_quad_br',
    'pool_center',
]
SPP_POOL_FUNCS = [(name, TENSOR_OPERATORS[name][0]) for name in SPP_POOLS]
N_SPP = len(SPP_POOLS)  # 8


def build_data_batch(images, device):
    images = images.to(device, dtype=torch.float32)
    I_R, I_G, I_B = images[:, 0], images[:, 1], images[:, 2]
    I_GRAY = 0.2989 * I_R + 0.5870 * I_G + 0.1140 * I_B
    Cmax, _ = images.max(dim=1)
    Cmin, _ = images.min(dim=1)
    delta = Cmax - Cmin + 1e-8
    H = torch.zeros_like(I_R)
    mr = (Cmax == I_R)
    mg = (Cmax == I_G) & ~mr
    mb = ~mr & ~mg
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
    out = torch.clamp(stack[0], -1e4, 1e4)
    return out if out.dim() >= 2 else None


def load_formulas_from_phase1(phase1_dir):
    all_formulas, seen = [], set()
    for bank_dir in sorted(Path(phase1_dir).glob('bank_*/feature_bank')):
        fb_path = bank_dir / 'feature_bank.json'
        if not fb_path.exists():
            continue
        bank = json.load(open(fb_path))
        for f in bank['formulas']:
            if f['str'] not in seen:
                seen.add(f['str'])
                all_formulas.append(f)
    return all_formulas


def formulas_to_bodies(formulas):
    bodies = set()
    for f in formulas:
        tokens = f['str'].strip().split()
        if tokens[-1] in ROOT_OPERATORS:
            bodies.add(' '.join(tokens[:-1]))
        else:
            bodies.add(f['str'])
    return sorted(bodies)


def main():
    device = 'cuda'
    torch.manual_seed(42)
    np.random.seed(42)

    out_dir = BASE_DIR / 'layer2'
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_path = out_dir / 'l1_selected_bodies.json'

    # ── Load kernel bank ──
    kb = SymbolicKernelBank(device=device)
    kb_path = BASE_DIR / 'kernel_bank_pretrained.pt'
    if kb_path.exists():
        kb.load_state_dict(torch.load(str(kb_path), map_location=device, weights_only=True))
    kb.register_operators(TENSOR_OPERATORS)

    # ── Load bodies ──
    formulas = load_formulas_from_phase1(BASE_DIR / 'phase1')
    bodies = formulas_to_bodies(formulas)
    n_bodies = len(bodies)
    n_feats = n_bodies * N_SPP
    print(f"Bodies: {n_bodies}, Features: {n_bodies} x {N_SPP} = {n_feats}")

    # ── Step 1: Extract 8 SPP features on 20K training images ──
    print(f"\n{'='*60}")
    print(f"  Step 1: Extract {N_SPP} SPP features per body")
    print(f"{'='*60}")

    dm = ImageNetDataModule(data_dir=DATA_DIR, resolution=112, batch_size=256,
                            num_workers=8, samples_per_class=20)
    dm.setup()
    train_loader = DataLoader(dm.train_dataset, batch_size=256, shuffle=False,
                              num_workers=8, pin_memory=True)
    n_train = len(dm.train_dataset)
    print(f"  Train images: {n_train}")

    mmap_path = str(out_dir / 'X_l1select_train.mmap')
    label_path = str(out_dir / 'y_l1select_train.npy')

    if os.path.exists(mmap_path) and os.path.exists(label_path):
        print(f"  Mmap already exists, loading...")
        X_mmap = np.memmap(mmap_path, dtype='float32', mode='r', shape=(n_train, n_feats))
        y_train = np.load(label_path)
    else:
        X_mmap = np.memmap(mmap_path, dtype='float32', mode='w+', shape=(n_train, n_feats))
        all_labels = []
        row = 0
        t0 = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            B = images.shape[0]
            data_batch = build_data_batch(images, device)
            gpu_buf = torch.zeros(B, n_feats, device=device)

            for b_idx, body_str in enumerate(bodies):
                try:
                    feat_map = execute_body(body_str, data_batch)
                    if feat_map is not None:
                        for p_idx, (_, pool_fn) in enumerate(SPP_POOL_FUNCS):
                            col = b_idx * N_SPP + p_idx
                            gpu_buf[:, col] = pool_fn(feat_map)
                except Exception:
                    pass

            end = min(row + B, n_train)
            X_mmap[row:end] = gpu_buf[:end - row].cpu().numpy()
            all_labels.append(labels.numpy()[:end - row])
            row = end

            del gpu_buf, data_batch
            torch.cuda.empty_cache()

            if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                elapsed = time.time() - t0
                print(f"    batch {batch_idx+1}/{len(train_loader)} "
                      f"({row}/{n_train}) {elapsed:.0f}s")

        X_mmap.flush()
        y_train = np.concatenate(all_labels)
        np.save(label_path, y_train)
        print(f"  Extraction done: ({n_train}, {n_feats}) in {(time.time()-t0)/60:.1f} min")

    # ── Step 2: Train nn.Linear with L1 penalty ──
    print(f"\n{'='*60}")
    print(f"  Step 2: Train nn.Linear({n_feats}, {NUM_CLASSES}) with L1")
    print(f"{'='*60}")

    # Compute mean/std
    print("  Computing mean/std...")
    s = np.zeros(n_feats, dtype=np.float64)
    sq = np.zeros(n_feats, dtype=np.float64)
    for start in range(0, n_train, 10000):
        end = min(start + 10000, n_train)
        c = np.nan_to_num(np.array(X_mmap[start:end], dtype=np.float64))
        s += c.sum(axis=0)
        sq += (c ** 2).sum(axis=0)
    mean = (s / n_train).astype(np.float32)
    std = np.sqrt(np.maximum(sq / n_train - (s / n_train) ** 2, 0.0)).astype(np.float32)
    std = np.maximum(std, 1e-8)
    mean_t = torch.tensor(mean, device=device)
    std_t = torch.tensor(std, device=device)

    model = nn.Linear(n_feats, NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=10.0)
    crit = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    for epoch in range(20):
        model.train()
        perm = np.random.permutation(n_train)
        epoch_loss = 0
        n_b = 0

        for start in range(0, n_train, 1024):
            end = min(start + 1024, n_train)
            idx = perm[start:end]

            X_b = torch.tensor(np.array(X_mmap[idx]), dtype=torch.float32, device=device)
            X_b = torch.nan_to_num(X_b)
            X_b = (X_b - mean_t) / std_t
            y_b = y_train_tensor[idx].to(device)

            logits = model(X_b)
            loss = crit(logits, y_b)

            # L1 penalty on weights
            l1_loss = L1_COEF * model.weight.abs().sum()
            total_loss = loss + l1_loss

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_b += 1

        sched.step()
        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1}/20, CE={epoch_loss/n_b:.4f}")

    # ── Step 3: Rank bodies by importance ──
    print(f"\n{'='*60}")
    print(f"  Step 3: Rank bodies by importance")
    print(f"{'='*60}")

    W = model.weight.detach().cpu().numpy()  # [1000, n_feats]
    body_importance = np.zeros(n_bodies)
    for i in range(n_bodies):
        body_importance[i] = np.abs(W[:, i * N_SPP:(i + 1) * N_SPP]).sum()

    top_idx = np.argsort(body_importance)[::-1][:N_SELECT]
    selected_bodies = [bodies[i] for i in top_idx]
    selected_importance = body_importance[top_idx]

    print(f"  Top-{N_SELECT} importance: max={selected_importance[0]:.4f}, "
          f"min={selected_importance[-1]:.4f}, "
          f"median={np.median(selected_importance):.4f}")
    print(f"  Cutoff importance: {selected_importance[-1]:.4f} "
          f"(vs mean all: {body_importance.mean():.4f})")

    # ── Step 4: Save ──
    with open(selected_path, 'w') as f:
        json.dump(selected_bodies, f, indent=2)
    print(f"\n  Saved {len(selected_bodies)} bodies → {selected_path}")

    # Also save importance for analysis
    np.savez(str(out_dir / 'l1_selection_meta.npz'),
             top_idx=top_idx, importance=body_importance,
             selected_importance=selected_importance)

    # ── Step 5: Sanity check — standalone val accuracy ──
    print(f"\n{'='*60}")
    print(f"  Step 5: Standalone val accuracy of selected bodies")
    print(f"{'='*60}")

    del model, X_mmap
    gc.collect()
    torch.cuda.empty_cache()

    # Load val data
    dm_val = ImageNetDataModule(data_dir=DATA_DIR, resolution=112, batch_size=256,
                                num_workers=8, samples_per_class=5)
    dm_val.setup()
    val_images, val_labels = [], []
    for images, labels in dm_val.get_val_loader():
        val_images.append(images)
        val_labels.append(labels)
    val_images = torch.cat(val_images, dim=0)
    val_labels = torch.cat(val_labels, dim=0).to(device)
    N_val = val_images.shape[0]
    print(f"  Val set: {N_val} images")

    data_batch = build_data_batch(val_images, device)
    del val_images

    standalone_accs = []
    for i, body in enumerate(selected_bodies):
        try:
            fm = execute_body(body, data_batch)
            if fm is None:
                standalone_accs.append(0.0)
                continue

            # 8 SPP features
            feats = []
            for _, pool_fn in SPP_POOL_FUNCS:
                feats.append(pool_fn(fm))
            feats = torch.stack(feats, dim=1)  # [N, 8]

            # Quick linear eval: 20 SGD steps
            m = nn.Linear(N_SPP, NUM_CLASSES).to(device)
            o = torch.optim.SGD(m.parameters(), lr=0.1, weight_decay=1.0)
            cr = nn.CrossEntropyLoss()
            f_mean = feats.mean(0, keepdim=True)
            f_std = feats.std(0, keepdim=True).clamp(min=1e-8)
            X = (feats - f_mean) / f_std
            for _ in range(20):
                cr(m(X), val_labels).backward()
                o.step()
                o.zero_grad()

            with torch.no_grad():
                acc = (m(X).argmax(1) == val_labels).float().mean().item()
            standalone_accs.append(acc)
            del m, o
        except Exception:
            standalone_accs.append(0.0)

    torch.cuda.empty_cache()

    accs = np.array(standalone_accs)
    print(f"\n  Standalone accuracy of {N_SELECT} selected bodies:")
    print(f"    max:    {accs.max()*100:.3f}%")
    print(f"    median: {np.median(accs)*100:.3f}%")
    print(f"    min:    {accs.min()*100:.3f}%")
    print(f"    mean:   {accs.mean()*100:.3f}%")
    print(f"    random: {100/NUM_CLASSES:.3f}%")

    # Print top-5 bodies
    top5 = np.argsort(accs)[::-1][:5]
    print(f"\n  Top-5 by standalone accuracy:")
    for j in top5:
        print(f"    acc={accs[j]*100:.3f}%  importance={selected_importance[j]:.2f}  {selected_bodies[j]}")

    print(f"\n  Done.")


if __name__ == '__main__':
    main()
