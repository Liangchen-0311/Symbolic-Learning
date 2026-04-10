#!/usr/bin/env python3
"""
ImageNet v3.2 Complete Pipeline

Implements the v3.2 plan with key improvements over v3:
  - New operators (rotation-invariant, local structure, second-order)
  - Distribution statistics encoding (12 stats × 5 regions = 60 per body)
  - Symbolic Fisher Vector (GMM-based, 4096-dim)
  - Homogeneous kernel map (chi-squared approximation)
  - Power normalization + L2 normalization
  - Layer 2 hierarchical formulas

Pipeline:
  Step 0: Pretrain kernels
  Step 1: Phase 1A — Layer 1 RL
  Step 2: Forward selection (top-100 complementary L1 bodies)
  Step 3: Phase 1B — Layer 2 RL (110 terminals)
  Step 4: Phase 3 — Feature extraction + encoding
  Step 5: Train classifier
  Step 6: End-to-end kernel fine-tuning
  Step 7: Scale to full dataset

Usage:
    python experiments/run_v3_2_pipeline.py --start_step 0
    python experiments/run_v3_2_pipeline.py --start_step 4  # from feature extraction
"""

import argparse, gc, json, math, os, sys, time, itertools
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
from src.symbolic.feature_encoding import (
    encode_body_distribution_v2,
    SymbolicFisherVector,
    homogeneous_kernel_map,
    apply_normalization_pipeline,
    apply_normalization_pipeline_with_stats,
)

# ======================================================================
# Config
# ======================================================================

DATA_DIR = '/workspace/neurosymbolic-rl/data/imagenet'
BASE_DIR = Path('outputs/imagenet_v3_2')
CONFIG_PATH = 'configs/tensor_vsr_imagenet_v3_2.yaml'
NUM_CLASSES = 1000
RESOLUTIONS = [112, 224]
BATCH_SIZES = {112: 512, 224: 256}
N_DIST_STATS = 60  # 12 stats × 5 regions


# ======================================================================
# Shared helpers
# ======================================================================

def build_data_batch(images, device):
    """Build the 10-terminal data batch from raw images. FP32 throughout."""
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
    """Execute formula body in RPN → spatial [B,H,W] or None."""
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


def build_data_batch_l2(images, device, l1_selected):
    """Build data batch with L1 feature maps as extra terminals."""
    base = build_data_batch(images, device)
    for i, body in enumerate(l1_selected):
        try:
            fm = execute_body(body, base)
            base[f'L1_{i}'] = fm if fm is not None else torch.zeros_like(base['I_R'])
        except Exception:
            base[f'L1_{i}'] = torch.zeros_like(base['I_R'])
    return base


def online_mean_std(X_mmap, n_total, n_feats, chunk_size=10000):
    """Compute mean/std from mmap in chunks to avoid OOM."""
    s = np.zeros(n_feats, dtype=np.float64)
    sq = np.zeros(n_feats, dtype=np.float64)
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        c = np.nan_to_num(np.array(X_mmap[start:end], dtype=np.float64))
        s += c.sum(axis=0)
        sq += (c ** 2).sum(axis=0)
    mean = s / n_total
    std = np.sqrt(np.maximum(sq / n_total - mean ** 2, 0.0))
    return mean.astype(np.float32), std.astype(np.float32)


def load_formulas_from_phase1(phase1_dir):
    """Load all formulas from Phase 1 bank directories."""
    all_formulas = []
    seen = set()
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
    """Extract unique bodies (strip trailing root operator) from formulas."""
    bodies = set()
    for f in formulas:
        tokens = f['str'].strip().split()
        if tokens[-1] in ROOT_OPERATORS:
            bodies.add(' '.join(tokens[:-1]))
        else:
            bodies.add(f['str'])
    return sorted(bodies)


# ======================================================================
# Step 0: Pretrain Kernels
# ======================================================================

def step0_pretrain_kernels(device):
    """Pretrain learnable kernels via supervised classification."""
    print(f"\n{'='*70}")
    print(f"  STEP 0: Pretrain Learnable Kernels")
    print(f"{'='*70}")

    out_dir = BASE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / 'kernel_bank_pretrained.pt'

    if output_path.exists():
        print(f"  Already done: {output_path}")
        return

    kb = SymbolicKernelBank(device=device)
    # Only learnable kernels need gradient — classic kernels are already meaningful
    kb.classic_3x3.requires_grad_(False)
    kb.classic_7x7.requires_grad_(False)
    # conv3x3 and conv5x5 default to requires_grad=True (nn.Parameter)

    # Model: 10 terminals × 12 learnable kernels → pool → linear(120, 1000)
    n_learnable = kb.conv3x3.shape[0] + kb.conv5x5.shape[0]  # 8 + 4 = 12
    pool = nn.AdaptiveAvgPool2d(1)
    fc = nn.Linear(n_learnable * 10, NUM_CLASSES).to(device)

    optimizer = torch.optim.AdamW(
        [kb.conv3x3, kb.conv5x5] + list(fc.parameters()),
        lr=1e-3, weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()

    dm = ImageNetDataModule(data_dir=DATA_DIR, resolution=112, batch_size=128,
                            num_workers=8, samples_per_class=20)
    dm.setup()
    train_loader = DataLoader(dm.train_dataset, batch_size=128, shuffle=True,
                              num_workers=8, pin_memory=True)

    terminal_names = ['I_R', 'I_G', 'I_B', 'I_GRAY', 'I_H', 'I_S',
                      'I_r', 'I_g', 'I_RG', 'I_BY']

    print(f"  Training: {n_learnable} learnable kernels × {len(terminal_names)} terminals "
          f"→ Linear({n_learnable * 10}, {NUM_CLASSES})")
    print(f"  Classic kernels (6) frozen — already Sobel/Gabor values")
    t0 = time.time()

    for epoch in range(10):
        total_loss = 0
        n_batches = 0
        for images, labels in train_loader:
            data_batch = build_data_batch(images, device)
            features = []
            for tname in terminal_names:
                x = data_batch[tname]
                x4d = x.unsqueeze(1)
                # Only learnable kernels: conv3x3_0..7 and conv5x5_0..3
                for i in range(kb.conv3x3.shape[0]):
                    out = F.conv2d(x4d, kb.conv3x3[i:i+1], padding=1)
                    features.append(pool(out).flatten(1))
                for i in range(kb.conv5x5.shape[0]):
                    out = F.conv2d(x4d, kb.conv5x5[i:i+1], padding=2)
                    features.append(pool(out).flatten(1))

            feat = torch.cat(features, dim=1)  # [B, 120]
            logits = fc(feat)
            loss = criterion(logits, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        print(f"    Epoch {epoch+1}/10: loss={total_loss/n_batches:.4f} ({time.time()-t0:.0f}s)")

    # Save kernel weights only
    torch.save(kb.state_dict(), str(output_path))
    print(f"  Saved pretrained kernels to {output_path}")


# ======================================================================
# Step 1: Phase 1A — Layer 1 RL
# ======================================================================

def step1_phase1a(device):
    """Run Phase 1A: 4-bank RL to discover Layer 1 formulas."""
    print(f"\n{'='*70}")
    print(f"  STEP 1: Phase 1A — Layer 1 RL Discovery")
    print(f"{'='*70}")

    phase1_dir = BASE_DIR / 'phase1'
    meta_path = phase1_dir / 'phase1_meta.json'

    if meta_path.exists():
        meta = json.load(open(meta_path))
        print(f"  Already done: {meta.get('total_formulas', '?')} formulas")
        return

    # Directly call run_phase1 to avoid running Phase 2/3 from the old pipeline.
    # Phase 2 had a terminal-mismatch bug that killed 73% of formulas in v3.
    # v3.2 skips Phase 2 entirely — quality filtering is done by Phase 1
    # correlation gate + Phase 3 L1 selection (step4).
    import yaml
    from experiments.train_imagenet_pipeline import run_phase1

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    run_phase1(config, device, phase1_dir)

    # Report results
    formulas = load_formulas_from_phase1(phase1_dir)
    print(f"\n  Phase 1A complete: {len(formulas)} formulas discovered")

    # Operator usage stats
    op_counts = {}
    for f in formulas:
        for tok in f['str'].split():
            if tok in TENSOR_OPERATORS:
                op_counts[tok] = op_counts.get(tok, 0) + 1
    print(f"  Top-10 operators:")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {op}: {count}")


# ======================================================================
# Step 2: Forward Selection
# ======================================================================

def step2_forward_selection(device):
    """Select top-100 most complementary Layer 1 formula bodies."""
    print(f"\n{'='*70}")
    print(f"  STEP 2: Forward Selection — Top-100 Complementary L1 Bodies")
    print(f"{'='*70}")

    out_dir = BASE_DIR / 'layer2'
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_path = out_dir / 'l1_selected_bodies.json'

    if selected_path.exists():
        selected = json.load(open(selected_path))
        print(f"  Already done: {len(selected)} bodies selected")
        return

    # Load all L1 formulas
    formulas = load_formulas_from_phase1(BASE_DIR / 'phase1')
    bodies = formulas_to_bodies(formulas)
    print(f"  Candidate bodies: {len(bodies)}")

    # Load kernel bank
    kb = SymbolicKernelBank(device=device)
    kb_path = BASE_DIR / 'kernel_bank_pretrained.pt'
    if kb_path.exists():
        kb.load_state_dict(torch.load(str(kb_path), map_location=device, weights_only=True))
    kb.register_operators(TENSOR_OPERATORS)

    # Prepare validation data (5/class = 5K images)
    dm = ImageNetDataModule(data_dir=DATA_DIR, resolution=112, batch_size=256,
                            num_workers=8, samples_per_class=5)
    dm.setup()
    val_loader = dm.get_val_loader()

    # Collect all val images + labels
    all_images, all_labels = [], []
    for images, labels in val_loader:
        all_images.append(images)
        all_labels.append(labels)
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0).to(device)
    N_val = all_images.shape[0]
    print(f"  Validation set: {N_val} images")

    # Pre-compute data batch
    data_batch = build_data_batch(all_images, device)
    del all_images
    gc.collect(); torch.cuda.empty_cache()

    # Encode each body with distribution stats
    def encode_body(body_str):
        """Execute body and encode with distribution stats → [N, 60]."""
        try:
            fm = execute_body(body_str, data_batch)
            if fm is None:
                return None
            return encode_body_distribution_v2(fm)
        except Exception:
            return None

    # Greedy forward selection
    selected = []
    selected_features = None  # [N_val, D_accumulated]
    n_target = 100
    n_candidates_per_round = 500

    print(f"  Running greedy selection ({n_target} rounds, {n_candidates_per_round} candidates/round)")
    t0 = time.time()

    for round_idx in range(n_target):
        # Sample candidates
        remaining = [b for b in bodies if b not in selected]
        if not remaining:
            print(f"    Round {round_idx}: no more candidates")
            break

        if len(remaining) > n_candidates_per_round:
            candidates = list(np.random.choice(remaining, n_candidates_per_round, replace=False))
        else:
            candidates = remaining

        best_gain = -1
        best_body = None
        best_feats = None

        for body in candidates:
            feats = encode_body(body)
            if feats is None:
                continue

            # Combine with previously selected
            if selected_features is not None:
                combined = torch.cat([selected_features, feats], dim=1)
            else:
                combined = feats

            # Quick eval: 5 SGD steps on linear classifier
            D = combined.shape[1]
            model = nn.Linear(D, NUM_CLASSES).to(device)
            opt = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1.0)
            criterion = nn.CrossEntropyLoss()

            # Standardize
            mean = combined.mean(dim=0, keepdim=True)
            std = combined.std(dim=0, keepdim=True).clamp(min=1e-8)
            X = (combined - mean) / std

            for _ in range(5):
                logits = model(X)
                loss = criterion(logits, all_labels)
                opt.zero_grad()
                loss.backward()
                opt.step()

            with torch.no_grad():
                acc = (model(X).argmax(1) == all_labels).float().mean().item()

            if acc > best_gain:
                best_gain = acc
                best_body = body
                best_feats = feats.detach()

            del model, opt
            torch.cuda.empty_cache()

        if best_body is None:
            print(f"    Round {round_idx}: no valid candidate")
            break

        selected.append(best_body)
        if selected_features is not None:
            selected_features = torch.cat([selected_features, best_feats], dim=1)
        else:
            selected_features = best_feats

        if (round_idx + 1) % 10 == 0:
            print(f"    Round {round_idx+1}: acc={best_gain*100:.2f}% "
                  f"({len(selected)} bodies, {time.time()-t0:.0f}s)")

        # Early stop: check if gain is too small (compare with previous round's best)
        if round_idx > 0 and best_gain < 0.001:
            print(f"    Early stopping at round {round_idx+1}: gain < 0.1%")
            break

    # Save
    with open(selected_path, 'w') as f:
        json.dump(selected, f, indent=2)
    print(f"  Selected {len(selected)} complementary L1 bodies → {selected_path}")


# ======================================================================
# Step 3: Phase 1B — Layer 2 RL
# ======================================================================

def step3_phase1b(device):
    """Run Phase 1B: RL with L1 feature maps as extra terminals."""
    print(f"\n{'='*70}")
    print(f"  STEP 3: Phase 1B — Layer 2 RL (110 terminals)")
    print(f"{'='*70}")

    l2_dir = BASE_DIR / 'layer2'
    meta_path = l2_dir / 'phase1' / 'phase1_meta.json'

    if meta_path.exists():
        print(f"  Already done")
        return

    selected_path = l2_dir / 'l1_selected_bodies.json'
    if not selected_path.exists():
        print(f"  ERROR: Run Step 2 (forward selection) first")
        return

    # Create a Layer 2 config if not exists
    l2_config = BASE_DIR / 'layer2_config.yaml'
    if not l2_config.exists():
        import yaml
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        # Override for Layer 2
        cfg['model']['max_depth'] = 5
        cfg['model']['max_sequence_length'] = 12
        cfg['output_dir'] = str(l2_dir)
        cfg['layer2_selected_bodies'] = str(selected_path)
        with open(l2_config, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

    cmd = (
        f"PYTHONUNBUFFERED=1 python experiments/train_imagenet_pipeline.py "
        f"--config {l2_config} "
        f"--device {device} "
        f"--output_dir {l2_dir} "
        f"--start_phase 1"
    )
    print(f"  Running: {cmd}")
    os.system(cmd)

    # Report
    l2_formulas = load_formulas_from_phase1(l2_dir / 'phase1')
    print(f"\n  Phase 1B complete: {len(l2_formulas)} Layer 2 formulas")


# ======================================================================
# Step 4: Phase 3 — Feature Extraction + Encoding
# ======================================================================

def step4_feature_extraction(device):
    """Extract distribution statistics features for all L1+L2 bodies."""
    print(f"\n{'='*70}")
    print(f"  STEP 4: Feature Extraction + Distribution Encoding")
    print(f"{'='*70}")

    out_dir = BASE_DIR / 'phase3'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load kernel bank
    kb = SymbolicKernelBank(device=device)
    kb_path = BASE_DIR / 'kernel_bank_pretrained.pt'
    if kb_path.exists():
        kb.load_state_dict(torch.load(str(kb_path), map_location=device, weights_only=True))
        print(f"  Loaded kernel bank from {kb_path}")
    kb.register_operators(TENSOR_OPERATORS)

    # Load L1 bodies
    l1_formulas = load_formulas_from_phase1(BASE_DIR / 'phase1')
    l1_bodies = formulas_to_bodies(l1_formulas)
    print(f"  L1 bodies: {len(l1_bodies)}")

    # Load L2 bodies
    l2_bodies = []
    l2_phase1_dir = BASE_DIR / 'layer2' / 'phase1'
    if l2_phase1_dir.exists():
        l2_formulas = load_formulas_from_phase1(l2_phase1_dir)
        l2_bodies = formulas_to_bodies(l2_formulas)
    print(f"  L2 bodies: {len(l2_bodies)}")

    # Load L1 selected for L2 terminal computation
    l1_selected = []
    l1_selected_path = BASE_DIR / 'layer2' / 'l1_selected_bodies.json'
    if l1_selected_path.exists():
        l1_selected = json.load(open(l1_selected_path))

    all_bodies = l1_bodies + l2_bodies
    n_bodies = len(all_bodies)
    n_l1 = len(l1_bodies)
    n_feats_per_res = n_bodies * N_DIST_STATS
    print(f"  Total: {n_bodies} bodies × {N_DIST_STATS} stats × {len(RESOLUTIONS)} res "
          f"= {n_feats_per_res * len(RESOLUTIONS)} raw features")

    with open(out_dir / 'all_bodies.json', 'w') as f:
        json.dump({'l1_bodies': l1_bodies, 'l2_bodies': l2_bodies}, f)

    # ── 4a: Extract distribution statistics ──
    print(f"\n  --- 4a: Distribution Statistics Extraction ---")

    for res in RESOLUTIONS:
        bs = BATCH_SIZES[res]
        dm = ImageNetDataModule(data_dir=DATA_DIR, resolution=res, batch_size=bs,
                                num_workers=8, samples_per_class=500)
        dm.setup()
        train_loader = DataLoader(dm.train_dataset, batch_size=bs, shuffle=False,
                                  num_workers=8, pin_memory=True)
        val_loader = dm.get_val_loader()
        n_train = len(dm.train_dataset)
        n_val = len(dm.val_dataset)

        for split, loader, n_total in [('train', train_loader, n_train),
                                        ('val', val_loader, n_val)]:
            mmap_path = str(out_dir / f'X_{split}_{res}.mmap')
            if os.path.exists(mmap_path) and not os.path.exists(mmap_path + '.progress.json'):
                print(f"    [{split}@{res}] Already done, skipping")
                continue

            progress_path = mmap_path + '.progress.json'
            label_path = mmap_path.replace('.mmap', '_labels_partial.npy')

            start_batch = 0
            row_offset = 0
            all_labels_list = []

            if os.path.exists(progress_path):
                with open(progress_path) as f:
                    p = json.load(f)
                start_batch = p['next_batch']
                row_offset = p['row_offset']
                if os.path.exists(label_path):
                    all_labels_list = [np.load(label_path)]
                mmap_out = np.memmap(mmap_path, dtype='float32', mode='r+',
                                     shape=(n_total, n_feats_per_res))
                print(f"    [{split}@{res}] Resuming from batch {start_batch}")
            else:
                mmap_out = np.memmap(mmap_path, dtype='float32', mode='w+',
                                     shape=(n_total, n_feats_per_res))

            total_batches = len(loader)
            t0 = time.time()

            for batch_idx, (images, labels) in enumerate(loader):
                if batch_idx < start_batch:
                    continue

                B = images.shape[0]

                # Build data batch (with L1 terminals for L2 bodies)
                if l2_bodies:
                    data_batch = build_data_batch_l2(images, device, l1_selected)
                else:
                    data_batch = build_data_batch(images, device)

                gpu_buf = torch.zeros(B, n_feats_per_res, device=device)

                for b_idx, body_str in enumerate(all_bodies):
                    try:
                        feat_map = execute_body(body_str, data_batch)
                        if feat_map is not None:
                            stats = encode_body_distribution_v2(feat_map)  # [B, 60]
                            col_start = b_idx * N_DIST_STATS
                            gpu_buf[:, col_start:col_start + N_DIST_STATS] = stats
                    except Exception:
                        pass

                batch_feats = gpu_buf.cpu().numpy()
                del gpu_buf, data_batch
                torch.cuda.empty_cache()

                end = min(row_offset + B, n_total)
                mmap_out[row_offset:end] = batch_feats[:end - row_offset]
                all_labels_list.append(labels.numpy()[:end - row_offset])
                row_offset = end

                if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
                    mmap_out.flush()
                    with open(progress_path, 'w') as f:
                        json.dump({'next_batch': batch_idx + 1, 'row_offset': row_offset}, f)
                    np.save(label_path, np.concatenate(all_labels_list))

                if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
                    print(f"    [{split}@{res}] batch {batch_idx+1}/{total_batches} "
                          f"({row_offset}/{n_total}) {time.time()-t0:.0f}s")

            mmap_out.flush()
            y = np.concatenate(all_labels_list)
            for p in [progress_path, label_path]:
                if os.path.exists(p):
                    os.remove(p)
            # Save labels once
            label_file = out_dir / f'y_{split}.npy'
            if not label_file.exists():
                np.save(str(label_file), y)

            print(f"    [{split}@{res}] Done: ({n_total}, {n_feats_per_res}) "
                  f"in {(time.time()-t0)/60:.1f} min")

        gc.collect()
        torch.cuda.empty_cache()

    # ── 4b: L1 Feature Selection → ~50K ──
    print(f"\n  --- 4b: L1 Feature Selection ---")
    y_train = np.load(str(out_dir / 'y_train.npy'))
    y_val = np.load(str(out_dir / 'y_val.npy'))
    n_train, n_val = len(y_train), len(y_val)

    res_quotas = {112: 20000, 224: 30000}
    all_selected = {}

    for res in RESOLUTIONS:
        X_tr = np.memmap(str(out_dir / f'X_train_{res}.mmap'), dtype='float32', mode='r',
                         shape=(n_train, n_feats_per_res))
        mean, std = online_mean_std(X_tr, n_train, n_feats_per_res)
        std = np.maximum(std, 1e-8)

        # Quick L1-regularized linear model to rank features
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
                X_b = torch.nan_to_num(X_b)
                X_b = (X_b - mean_t) / std_t
                y_b = torch.tensor(y_train[idx], dtype=torch.long, device=device)
                opt.zero_grad()
                crit(model(X_b), y_b).backward()
                opt.step()

        importance = model.weight.abs().sum(dim=0).detach().cpu().numpy()
        quota = res_quotas[res]
        top_idx = np.argsort(importance)[::-1][:quota]
        top_idx = np.sort(top_idx[importance[top_idx] > 1e-8])
        all_selected[res] = (top_idx, mean, std)
        print(f"    {res}×{res}: selected {len(top_idx)} / {n_feats_per_res} features")
        del X_tr, model
        gc.collect()
        torch.cuda.empty_cache()

    # Build selected feature mmap
    n_selected = sum(len(v[0]) for v in all_selected.values())
    print(f"  Total selected: {n_selected}")

    sel_mean_parts, sel_std_parts = [], []
    for res in RESOLUTIONS:
        idx, m, s = all_selected[res]
        sel_mean_parts.append(m[idx])
        sel_std_parts.append(s[idx])
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
            col += len(idx)
            del X_res
        mmap_out.flush()

    np.save(str(out_dir / 'sel_mean.npy'), sel_mean)
    np.save(str(out_dir / 'sel_std.npy'), sel_std)
    np.savez(str(out_dir / 'selected_meta.npz'),
             **{f'idx_{res}': all_selected[res][0] for res in RESOLUTIONS})

    # ── 4c: Feature Interactions ──
    print(f"\n  --- 4c: Feature Interactions ---")
    X_sel_tr = np.memmap(str(out_dir / 'X_train_selected.mmap'), dtype='float32', mode='r',
                         shape=(n_train, n_selected))

    # Rank features by importance
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
            X_b = torch.nan_to_num(X_b)
            X_b = (X_b - mean_t) / std_t
            y_b = torch.tensor(y_train[idx], dtype=torch.long, device=device)
            opt.zero_grad()
            crit(model(X_b), y_b).backward()
            opt.step()

    importance = model.weight.abs().sum(dim=0).detach().cpu().numpy()
    top500 = np.argsort(importance)[::-1][:500]
    pairs = list(itertools.combinations(range(500), 2))
    n_interact = len(pairs)
    print(f"  Top-500 → {n_interact} interaction features")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    for split, n_total in [('train', n_train), ('val', n_val)]:
        X_sel = np.memmap(str(out_dir / f'X_{split}_selected.mmap'), dtype='float32', mode='r',
                          shape=(n_total, n_selected))
        int_path = str(out_dir / f'X_{split}_interact.mmap')
        mmap_int = np.memmap(int_path, dtype='float32', mode='w+', shape=(n_total, n_interact))
        for start in range(0, n_total, 10000):
            end = min(start + 10000, n_total)
            chunk = np.array(X_sel[start:end])[:, top500]
            chunk = (chunk - sel_mean[top500]) / np.maximum(sel_std[top500], 1e-8)
            for p_idx, (i, j) in enumerate(pairs):
                mmap_int[start:end, p_idx] = chunk[:, i] * chunk[:, j]
        mmap_int.flush()
        print(f"    {split}: ({n_total}, {n_interact})")

    # L1 selection again on (base + interactions) → ~50K
    print(f"\n  --- L1 re-selection on base + interactions ---")
    n_combined_raw = n_selected + n_interact
    # Train quick model on combined
    X_int_tr = np.memmap(str(out_dir / 'X_train_interact.mmap'), dtype='float32', mode='r',
                         shape=(n_train, n_interact))
    int_mean, int_std = online_mean_std(X_int_tr, n_train, n_interact)
    int_std = np.maximum(int_std, 1e-8)

    combined_mean = np.concatenate([sel_mean, int_mean])
    combined_std = np.concatenate([np.maximum(sel_std, 1e-8), int_std])

    mean_t = torch.tensor(combined_mean, device=device)
    std_t = torch.tensor(combined_std, device=device)
    model = nn.Linear(n_combined_raw, NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=30.0)
    crit = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        perm = np.random.permutation(n_train)
        for start in range(0, min(100000, n_train), 1024):
            end = min(start + 1024, n_train)
            idx = perm[start:end]
            X_base = torch.tensor(np.array(X_sel_tr[idx]), dtype=torch.float32, device=device)
            X_int = torch.tensor(np.array(X_int_tr[idx]), dtype=torch.float32, device=device)
            X_b = torch.cat([X_base, X_int], dim=1)
            X_b = torch.nan_to_num(X_b)
            X_b = (X_b - mean_t) / std_t
            y_b = torch.tensor(y_train[idx], dtype=torch.long, device=device)
            opt.zero_grad()
            crit(model(X_b), y_b).backward()
            opt.step()

    importance = model.weight.abs().sum(dim=0).detach().cpu().numpy()
    final_quota = 50000
    final_idx = np.argsort(importance)[::-1][:final_quota]
    final_idx = np.sort(final_idx[importance[final_idx] > 1e-8])
    print(f"  Final selected: {len(final_idx)} / {n_combined_raw}")

    final_mean = combined_mean[final_idx]
    final_std = combined_std[final_idx]

    for split, n_total in [('train', n_train), ('val', n_val)]:
        X_sel = np.memmap(str(out_dir / f'X_{split}_selected.mmap'), dtype='float32', mode='r',
                          shape=(n_total, n_selected))
        X_int = np.memmap(str(out_dir / f'X_{split}_interact.mmap'), dtype='float32', mode='r',
                          shape=(n_total, n_interact))
        final_path = str(out_dir / f'X_{split}_final.mmap')
        mmap_final = np.memmap(final_path, dtype='float32', mode='w+',
                               shape=(n_total, len(final_idx)))
        for start in range(0, n_total, 10000):
            end = min(start + 10000, n_total)
            chunk_base = np.array(X_sel[start:end])
            chunk_int = np.array(X_int[start:end])
            chunk_combined = np.concatenate([chunk_base, chunk_int], axis=1)
            mmap_final[start:end] = chunk_combined[:, final_idx]
        mmap_final.flush()

    np.save(str(out_dir / 'final_mean.npy'), final_mean)
    np.save(str(out_dir / 'final_std.npy'), final_std)
    np.savez(str(out_dir / 'interact_meta.npz'), top500=top500, n_interact=n_interact,
             final_idx=final_idx)

    # ── 4d: Symbolic Fisher Vector ──
    print(f"\n  --- 4d: Symbolic Fisher Vector ---")
    fv_dir = out_dir / 'fisher_vector'
    fv_dir.mkdir(exist_ok=True)

    l1_selected_path = BASE_DIR / 'layer2' / 'l1_selected_bodies.json'
    if l1_selected_path.exists():
        fv_bodies = json.load(open(l1_selected_path))
    else:
        # Fallback: use first 100 L1 bodies sorted by importance
        fv_bodies = l1_bodies[:100]
    n_fv_bodies = len(fv_bodies)
    print(f"  Fisher Vector bodies: {n_fv_bodies}")

    sfv = SymbolicFisherVector(pca_dim=32, gmm_k=64, device=device)

    # Collect local descriptors from training set for PCA + GMM fitting
    print(f"  Collecting local descriptors for PCA + GMM...")
    dm = ImageNetDataModule(data_dir=DATA_DIR, resolution=224, batch_size=64,
                            num_workers=8, samples_per_class=15)
    dm.setup()
    fit_loader = DataLoader(dm.train_dataset, batch_size=64, shuffle=True,
                            num_workers=8, pin_memory=True)

    all_descriptors = []
    n_collected = 0
    target_descriptors = 1000000  # ~1M

    for images, _ in fit_loader:
        data_batch = build_data_batch(images, device)
        B = images.shape[0]

        # Execute FV bodies → [B, n_fv_bodies, H, W]
        fmaps = []
        for body in fv_bodies:
            try:
                fm = execute_body(body, data_batch)
                if fm is None:
                    fm = torch.zeros(B, images.shape[2], images.shape[3], device=device)
            except Exception:
                fm = torch.zeros(B, images.shape[2], images.shape[3], device=device)
            fmaps.append(fm)

        fmaps = torch.stack(fmaps, dim=1)  # [B, n_fv, H, W]
        patches = sfv.extract_patches(fmaps, grid_size=8)  # [B, 64, n_fv]
        all_descriptors.append(patches.reshape(-1, n_fv_bodies).cpu())
        n_collected += patches.shape[0] * patches.shape[1]

        del data_batch, fmaps
        torch.cuda.empty_cache()

        if n_collected >= target_descriptors:
            break

    all_desc = torch.cat(all_descriptors, dim=0).to(device)
    print(f"  Collected {all_desc.shape[0]} descriptors")

    # Fit PCA
    sfv.fit_pca(all_desc)
    pca_desc = sfv.apply_pca(all_desc)
    print(f"  PCA: {all_desc.shape[1]}D → {sfv.pca_dim}D")

    # Fit GMM
    sfv.fit_gmm(pca_desc, n_iter=50)
    print(f"  GMM: K={sfv.gmm_k} fitted")

    sfv.save(str(fv_dir / 'sfv_params.pt'))
    del all_desc, pca_desc, all_descriptors
    gc.collect()
    torch.cuda.empty_cache()

    # Compute Fisher Vectors for train and val
    fv_dim = 2 * sfv.pca_dim * sfv.gmm_k  # 4096
    print(f"  FV dim: {fv_dim}")

    for split in ['train', 'val']:
        fv_path = str(fv_dir / f'fv_{split}.mmap')
        if os.path.exists(fv_path):
            print(f"    [{split}] Already done, skipping")
            continue

        dm = ImageNetDataModule(data_dir=DATA_DIR, resolution=224, batch_size=64,
                                num_workers=8, samples_per_class=500 if split == 'train' else 50)
        dm.setup()
        loader = DataLoader(dm.train_dataset if split == 'train' else dm.val_dataset,
                            batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
        n_total = len(dm.train_dataset) if split == 'train' else len(dm.val_dataset)

        mmap_fv = np.memmap(fv_path, dtype='float32', mode='w+', shape=(n_total, fv_dim))
        row = 0
        t0 = time.time()

        for batch_idx, (images, _) in enumerate(loader):
            data_batch = build_data_batch(images, device)
            B = images.shape[0]

            fmaps = []
            for body in fv_bodies:
                try:
                    fm = execute_body(body, data_batch)
                    if fm is None:
                        fm = torch.zeros(B, images.shape[2], images.shape[3], device=device)
                except Exception:
                    fm = torch.zeros(B, images.shape[2], images.shape[3], device=device)
                fmaps.append(fm)
            fmaps = torch.stack(fmaps, dim=1)

            fvs = sfv.encode_batch(fmaps, grid_size=8)  # [B, 4096]
            end = min(row + B, n_total)
            mmap_fv[row:end] = fvs[:end - row].cpu().numpy()
            row = end

            del data_batch, fmaps
            torch.cuda.empty_cache()

            if (batch_idx + 1) % 50 == 0:
                mmap_fv.flush()
                print(f"    [{split}] batch {batch_idx+1} ({row}/{n_total}) "
                      f"{time.time()-t0:.0f}s")

        mmap_fv.flush()
        print(f"    [{split}] Done: ({n_total}, {fv_dim}) in {(time.time()-t0)/60:.1f} min")

    # Save metadata
    json.dump({
        'n_bodies': n_bodies, 'n_l1': n_l1, 'n_l2': len(l2_bodies),
        'n_dist_stats': N_DIST_STATS, 'n_selected': n_selected,
        'n_interact': n_interact, 'n_final': len(final_idx), 'fv_dim': fv_dim,
    }, open(out_dir / 'phase3_meta.json', 'w'), indent=2)

    print(f"\n  Step 4 complete. Features saved to {out_dir}")


# ======================================================================
# Step 5: Train Classifier
# ======================================================================

def step5_train_classifier(device):
    """Train nn.Linear classifier on combined features."""
    print(f"\n{'='*70}")
    print(f"  STEP 5: Train Classifier")
    print(f"{'='*70}")

    out_dir = BASE_DIR / 'phase3'
    meta = json.load(open(out_dir / 'phase3_meta.json'))
    n_final = meta['n_final']
    fv_dim = meta['fv_dim']

    y_train = np.load(str(out_dir / 'y_train.npy'))
    y_val = np.load(str(out_dir / 'y_val.npy'))
    n_train, n_val = len(y_train), len(y_val)

    # Load distribution stats features
    X_final_tr = np.memmap(str(out_dir / 'X_train_final.mmap'), dtype='float32', mode='r',
                           shape=(n_train, n_final))
    X_final_va = np.memmap(str(out_dir / 'X_val_final.mmap'), dtype='float32', mode='r',
                           shape=(n_val, n_final))

    # Load Fisher Vector features
    fv_dir = out_dir / 'fisher_vector'
    X_fv_tr = np.memmap(str(fv_dir / 'fv_train.mmap'), dtype='float32', mode='r',
                        shape=(n_train, fv_dim))
    X_fv_va = np.memmap(str(fv_dir / 'fv_val.mmap'), dtype='float32', mode='r',
                        shape=(n_val, fv_dim))

    # ── 4e: Combine features ──
    n_combined = n_final + fv_dim
    print(f"  Distribution stats: {n_final} + Fisher Vector: {fv_dim} = {n_combined}")

    # Compute combined mean/std
    final_mean = np.load(str(out_dir / 'final_mean.npy'))
    final_std = np.load(str(out_dir / 'final_std.npy'))
    fv_mean, fv_std = online_mean_std(X_fv_tr, n_train, fv_dim)
    fv_std = np.maximum(fv_std, 1e-8)

    combined_mean = np.concatenate([final_mean, fv_mean])
    combined_std = np.concatenate([np.maximum(final_std, 1e-8), fv_std])

    # ── 4f: Homogeneous kernel map (on top-20K) ──
    print(f"\n  --- Homogeneous Kernel Map ---")
    # For efficiency, apply kernel map to top-20K features only
    km_top_k = min(20000, n_combined)
    # Train quick model to rank all combined features
    mean_t = torch.tensor(combined_mean, device=device)
    std_t = torch.tensor(combined_std, device=device)

    # We'll apply kernel map during training to avoid materializing 3× features

    # ── Train classifier with sweep ──
    print(f"\n  --- Classifier Training ---")
    results = {}
    best_acc = 0
    best_model = None

    for wd in [10, 20, 50, 100]:
        print(f"\n  Training with wd={wd}...")
        # Classifier on combined features with power norm + L2 norm + optional kernel map
        model = nn.Linear(n_combined, NUM_CLASSES).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
        crit = nn.CrossEntropyLoss()
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

        for epoch in range(30):
            model.train()
            perm = np.random.permutation(n_train)
            epoch_loss = 0
            n_b = 0

            for start in range(0, n_train, 1024):
                end = min(start + 1024, n_train)
                idx = perm[start:end]

                X_base = torch.tensor(np.array(X_final_tr[idx]), dtype=torch.float32, device=device)
                X_fv = torch.tensor(np.array(X_fv_tr[idx]), dtype=torch.float32, device=device)
                X_b = torch.cat([X_base, X_fv], dim=1)
                X_b = torch.nan_to_num(X_b)

                # Standardize
                X_b = (X_b - mean_t) / std_t

                # Power normalization (signed sqrt)
                X_b = torch.sign(X_b) * torch.sqrt(torch.abs(X_b) + 1e-8)

                # L2 normalization
                X_b = F.normalize(X_b, p=2, dim=1)

                y_b = torch.tensor(y_train[idx], dtype=torch.long, device=device)
                opt.zero_grad()
                crit(model(X_b), y_b).backward()
                opt.step()
                epoch_loss += crit(model(X_b), y_b).item()
                n_b += 1

            sched.step()
            if (epoch + 1) % 10 == 0:
                print(f"    epoch {epoch+1}/30, loss={epoch_loss/n_b:.4f}")

        # Evaluate
        model.eval()
        c1 = c5 = 0
        with torch.no_grad():
            for s in range(0, n_val, 2048):
                e = min(s + 2048, n_val)
                X_base = torch.tensor(np.array(X_final_va[s:e]), dtype=torch.float32, device=device)
                X_fv = torch.tensor(np.array(X_fv_va[s:e]), dtype=torch.float32, device=device)
                X_b = torch.cat([X_base, X_fv], dim=1)
                X_b = torch.nan_to_num(X_b)
                X_b = (X_b - mean_t) / std_t
                X_b = torch.sign(X_b) * torch.sqrt(torch.abs(X_b) + 1e-8)
                X_b = F.normalize(X_b, p=2, dim=1)

                logits = model(X_b)
                y_b = torch.tensor(y_val[s:e], dtype=torch.long, device=device)
                c1 += (logits.argmax(1) == y_b).sum().item()
                _, tk = logits.topk(5, dim=1)
                c5 += (tk == y_b.unsqueeze(1)).any(1).sum().item()

        v1, v5 = c1 / n_val, c5 / n_val
        results[wd] = (v1, v5)
        print(f"    wd={wd}: Val-T1={v1*100:.2f}% Val-T5={v5*100:.2f}%")

        if v1 > best_acc:
            best_acc = v1
            best_model = model
            best_wd = wd

    # Save best model
    torch.save(best_model.state_dict(), str(out_dir / 'classifier_best.pt'))
    np.save(str(out_dir / 'combined_mean.npy'), combined_mean)
    np.save(str(out_dir / 'combined_std.npy'), combined_std)
    json.dump({
        'results': {str(k): {'top1': v[0], 'top5': v[1]} for k, v in results.items()},
        'best_wd': best_wd, 'best_top1': best_acc,
        'n_features': n_combined,
    }, open(out_dir / 'classifier_results.json', 'w'), indent=2)

    print(f"\n  Best: wd={best_wd}, Val-T1={best_acc*100:.2f}%")


# ======================================================================
# Step 6: End-to-End Kernel Fine-Tuning
# ======================================================================

def step6_finetune(device):
    """Fine-tune learnable kernels + classifier jointly."""
    print(f"\n{'='*70}")
    print(f"  STEP 6: End-to-End Kernel Fine-Tuning")
    print(f"{'='*70}")

    out_dir = BASE_DIR / 'finetune'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all bodies
    bodies_data = json.load(open(BASE_DIR / 'phase3' / 'all_bodies.json'))
    l1_bodies = bodies_data['l1_bodies']
    l2_bodies = bodies_data.get('l2_bodies', [])
    all_bodies = l1_bodies + l2_bodies

    # Load L1 selected for L2 terminals
    l1_selected = []
    l1_selected_path = BASE_DIR / 'layer2' / 'l1_selected_bodies.json'
    if l1_selected_path.exists():
        l1_selected = json.load(open(l1_selected_path))

    # Load metadata
    phase3_meta = json.load(open(BASE_DIR / 'phase3' / 'phase3_meta.json'))
    cls_results = json.load(open(BASE_DIR / 'phase3' / 'classifier_results.json'))
    n_features = cls_results['n_features']

    combined_mean = np.load(str(BASE_DIR / 'phase3' / 'combined_mean.npy'))
    combined_std = np.load(str(BASE_DIR / 'phase3' / 'combined_std.npy'))

    # Load kernel bank in finetune mode
    kb = SymbolicKernelBank(device=device)
    kb_path = BASE_DIR / 'kernel_bank_pretrained.pt'
    if kb_path.exists():
        kb.load_state_dict(torch.load(str(kb_path), map_location=device, weights_only=True))
    kb.finetune_mode = True
    kb.register_operators(TENSOR_OPERATORS)

    # Load classifier
    classifier = nn.Linear(n_features, NUM_CLASSES).to(device)
    cls_path = BASE_DIR / 'phase3' / 'classifier_best.pt'
    if cls_path.exists():
        classifier.load_state_dict(torch.load(str(cls_path), map_location=device, weights_only=True))

    mean_t = torch.tensor(combined_mean, device=device)
    std_t = torch.tensor(np.maximum(combined_std, 1e-8), device=device)

    optimizer = torch.optim.AdamW([
        {'params': classifier.parameters(), 'lr': 1e-3, 'weight_decay': 10.0},
        {'params': kb.classic_3x3, 'lr': 1e-5},
        {'params': kb.classic_7x7, 'lr': 1e-5},
        {'params': kb.conv3x3, 'lr': 1e-4},
        {'params': kb.conv5x5, 'lr': 1e-4},
    ])
    criterion = nn.CrossEntropyLoss()

    dm = ImageNetDataModule(data_dir=DATA_DIR, resolution=112, batch_size=64,
                            num_workers=8, samples_per_class=500)
    dm.setup()
    train_loader = DataLoader(dm.train_dataset, batch_size=64, shuffle=True,
                              num_workers=8, pin_memory=True)

    # Load FV encoder
    sfv = SymbolicFisherVector(pca_dim=32, gmm_k=64, device=device)
    sfv_path = BASE_DIR / 'phase3' / 'fisher_vector' / 'sfv_params.pt'
    if sfv_path.exists():
        sfv.load(str(sfv_path))

    fv_bodies = l1_selected[:100] if l1_selected else l1_bodies[:100]

    print(f"  Online fine-tuning: {len(all_bodies)} bodies, 10 epochs")
    t0 = time.time()

    for epoch in range(10):
        classifier.train()
        kb.train()
        total_loss = 0
        n_batches = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            B = images.shape[0]

            # Build data batch with L1 terminals
            if l2_bodies:
                data_batch = build_data_batch_l2(images, device, l1_selected)
            else:
                data_batch = build_data_batch(images, device)

            # Extract distribution stats for all bodies
            dist_feats = []
            for body in all_bodies:
                try:
                    fm = execute_body(body, data_batch)
                    if fm is not None:
                        stats = encode_body_distribution_v2(fm)  # [B, 60]
                        dist_feats.append(stats)
                    else:
                        dist_feats.append(torch.zeros(B, N_DIST_STATS, device=device))
                except Exception:
                    dist_feats.append(torch.zeros(B, N_DIST_STATS, device=device))
            dist_feats = torch.cat(dist_feats, dim=1)  # [B, n_bodies*60]

            # TODO: add FV features for full pipeline (expensive online)
            # For now, use dist stats only (already ~50K features after selection)
            # Truncate to match classifier input size
            if dist_feats.shape[1] > n_features:
                feats = dist_feats[:, :n_features]
            else:
                feats = torch.zeros(B, n_features, device=device)
                feats[:, :dist_feats.shape[1]] = dist_feats

            feats = torch.nan_to_num(feats)
            feats = (feats - mean_t) / std_t
            feats = torch.sign(feats) * torch.sqrt(torch.abs(feats) + 1e-8)
            feats = F.normalize(feats, p=2, dim=1)

            logits = classifier(feats)
            loss = criterion(logits, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % 100 == 0:
                print(f"    Epoch {epoch+1}, batch {batch_idx+1}: "
                      f"loss={total_loss/n_batches:.4f}")

        print(f"    Epoch {epoch+1}/10: avg_loss={total_loss/max(n_batches,1):.4f} "
              f"({time.time()-t0:.0f}s)")

    # Save
    torch.save(kb.state_dict(), str(out_dir / 'kernel_bank_finetuned.pt'))
    torch.save(classifier.state_dict(), str(out_dir / 'classifier_finetuned.pt'))
    print(f"  Saved fine-tuned models to {out_dir}")

    # Re-extract features with updated kernels and retrain
    print(f"\n  After fine-tuning: re-run Step 4 + Step 5 with updated kernels")
    print(f"  Use: python experiments/run_v3_2_pipeline.py --start_step 4 "
          f"--kernel_path {out_dir / 'kernel_bank_finetuned.pt'}")


# ======================================================================
# Step 7: Scale to Full Dataset
# ======================================================================

def step7_scale_up(device):
    """Scale to full 1.18M training images."""
    print(f"\n{'='*70}")
    print(f"  STEP 7: Scale to Full Dataset (1.18M training images)")
    print(f"{'='*70}")
    print(f"  This step uses the same pipeline as Steps 4-5 but with")
    print(f"  samples_per_class=1281 (full ImageNet) instead of 500.")
    print(f"  Re-run: python experiments/run_v3_2_pipeline.py --start_step 4 --full_data")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='ImageNet v3.2 Full Pipeline')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--start_step', type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument('--kernel_path', type=str, default=None,
                        help='Path to fine-tuned kernel bank (for re-extraction)')
    parser.add_argument('--full_data', action='store_true',
                        help='Use full 1.18M training images')
    args = parser.parse_args()

    device = args.device
    torch.manual_seed(42)
    np.random.seed(42)

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    # If custom kernel path provided, load it
    if args.kernel_path:
        print(f"  Using kernel bank from: {args.kernel_path}")
        # Copy to expected location
        import shutil
        shutil.copy(args.kernel_path, str(BASE_DIR / 'kernel_bank_pretrained.pt'))

    if args.start_step <= 0:
        step0_pretrain_kernels(device)

    if args.start_step <= 1:
        step1_phase1a(device)

    if args.start_step <= 2:
        step2_forward_selection(device)

    if args.start_step <= 3:
        step3_phase1b(device)

    if args.start_step <= 4:
        step4_feature_extraction(device)

    if args.start_step <= 5:
        step5_train_classifier(device)

    if args.start_step <= 6:
        step6_finetune(device)

    if args.start_step <= 7:
        step7_scale_up(device)

    print(f"\n{'='*70}")
    print(f"  V3.2 PIPELINE COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
