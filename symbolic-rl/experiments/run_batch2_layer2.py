#!/usr/bin/env python3
"""
Batch 2: Hierarchical Layer 2 Formula Discovery.

1. Forward selection: pick top-100 Layer 1 bodies (most complementary)
2. Run Phase 1B: RL discovers Layer 2 formulas using 110 terminals
   (10 original + 100 L1 feature maps)
3. Extract Layer 2 features at 112+224
4. Combine L1 + L2 features + interactions
5. L1 selection → ~50K features
6. Train nn.Linear
"""

import gc, json, os, sys, time, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.imagenet_loader import ImageNetDataModule
from src.symbolic.tensor_operators import TENSOR_OPERATORS, ROOT_OPERATORS, SymbolicKernelBank

OUTPUT_DIR = Path('outputs/imagenet_v3/phase3_v3')
L2_OUTPUT_DIR = Path('outputs/imagenet_v3/layer2')
DATA_DIR = '/workspace/neurosymbolic-rl/data/imagenet'
NUM_CLASSES = 1000
N_ENCODINGS = 12
RESOLUTIONS = [112, 224]
BATCH_SIZES = {112: 512, 224: 256}

# Load kernel bank
kernel_bank = SymbolicKernelBank(device='cuda')
kb_path = OUTPUT_DIR / 'kernel_bank_weights.pt'
if kb_path.exists():
    kernel_bank.load_state_dict(torch.load(str(kb_path), weights_only=True))
kernel_bank.register_operators(TENSOR_OPERATORS)

SPP_POOLS = ['global_avg_pool', 'global_max_pool', 'global_std_pool',
             'pool_quad_tl', 'pool_quad_tr', 'pool_quad_bl', 'pool_quad_br', 'pool_center']
SPP_POOL_FUNCS = [(name, TENSOR_OPERATORS[name][0]) for name in SPP_POOLS]


# ======================================================================
# Helpers (same as v3)
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
    B = feat_map.shape[0]
    encodings = []
    for name, pool_func in SPP_POOL_FUNCS:
        try:
            pooled = pool_func(feat_map)
            pooled = torch.nan_to_num(pooled, nan=0.0, posinf=1e4, neginf=-1e4)
            encodings.append(torch.clamp(pooled, -1e4, 1e4))
        except: encodings.append(torch.zeros(B, device=feat_map.device))
    try:
        hist = TENSOR_OPERATORS['patch_histogram_4x4'][0](feat_map)
        hist = torch.nan_to_num(hist, nan=0.0, posinf=1e4, neginf=-1e4)
        hist = torch.clamp(hist, -1e4, 1e4)
        for i in range(4): encodings.append(hist[:, i])
    except:
        for _ in range(4): encodings.append(torch.zeros(B, device=feat_map.device))
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


# ======================================================================
# Step 1: Forward Selection of Top-100 Layer 1 Bodies
# ======================================================================

def forward_select_bodies(bodies, data_batch, labels, n_select=100):
    """Greedily select the most complementary bodies."""
    print(f"\n{'='*60}")
    print(f"  Step 1: Forward Selection ({n_select} from {len(bodies)})")
    print(f"{'='*60}")

    num_classes = labels.max().item() + 1
    device = labels.device

    selected = []
    selected_features = None  # [N, n_selected * 12]
    current_acc = 0.0

    remaining = list(range(len(bodies)))
    np.random.seed(42)

    for round_idx in range(n_select):
        best_gain, best_idx, best_feat = 0, -1, None

        # Sample candidates (max 200 per round for speed)
        candidates = np.random.choice(remaining, size=min(200, len(remaining)), replace=False)

        for c_idx in candidates:
            body = bodies[c_idx]
            try:
                feat_map = execute_body(body, data_batch)
                if feat_map is None: continue
                encoded = encode_feature_map(feat_map)  # [N, 12]
            except: continue

            # Build trial features
            if selected_features is not None:
                trial = torch.cat([selected_features, encoded], dim=1)
            else:
                trial = encoded

            # Quick linear probe (5 steps)
            n_feat = trial.shape[1]
            probe = nn.Linear(n_feat, num_classes).to(device)
            opt = torch.optim.Adam(probe.parameters(), lr=0.01)
            crit = nn.CrossEntropyLoss()

            # Standardize
            t_mean = trial.mean(dim=0, keepdim=True)
            t_std = trial.std(dim=0, keepdim=True).clamp(min=1e-8)
            trial_std = (trial - t_mean) / t_std

            probe.train()
            for _ in range(5):
                opt.zero_grad(); crit(probe(trial_std), labels).backward(); opt.step()

            probe.eval()
            with torch.no_grad():
                acc = (probe(trial_std).argmax(1) == labels).float().mean().item()

            gain = acc - current_acc
            if gain > best_gain:
                best_gain = gain
                best_idx = c_idx
                best_feat = encoded.detach()

        if best_idx < 0:
            print(f"  Round {round_idx+1}: no improvement, stopping")
            break

        selected.append(best_idx)
        remaining.remove(best_idx)
        if selected_features is not None:
            selected_features = torch.cat([selected_features, best_feat], dim=1)
        else:
            selected_features = best_feat
        current_acc += best_gain

        if (round_idx + 1) % 10 == 0:
            print(f"  Round {round_idx+1}/{n_select}: acc={current_acc*100:.2f}%, "
                  f"gain={best_gain*100:.3f}%")

    print(f"  Final: {len(selected)} bodies, acc={current_acc*100:.2f}%")
    return [bodies[i] for i in selected]


# ======================================================================
# Step 2: Run Phase 1B (Layer 2 RL Discovery)
# ======================================================================

def run_phase1b(l1_bodies, config_path, output_dir):
    """Run RL formula discovery with Layer 1 feature maps as terminals."""
    print(f"\n{'='*60}")
    print(f"  Step 2: Phase 1B — Layer 2 Formula Discovery")
    print(f"{'='*60}")

    # This requires modifying the RL environment to accept L1 bodies as terminals.
    # For now, we'll create a config and call the pipeline.

    # Save L1 bodies for the environment to load
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'l1_bodies.json', 'w') as f:
        json.dump(l1_bodies, f)

    print(f"  Saved {len(l1_bodies)} L1 bodies to {output_dir / 'l1_bodies.json'}")
    print(f"  Layer 2 terminals: 10 original + {len(l1_bodies)} L1 = {10 + len(l1_bodies)}")
    print(f"  TODO: Need to extend TensorVSREnvironmentLargeBank to support L1 terminals")
    print(f"  This requires modifying get_data_batch() to compute L1 feature maps on-the-fly")

    return None  # Placeholder — needs RL environment modification


# ======================================================================
# Main
# ======================================================================

def main():
    device = 'cuda'
    torch.manual_seed(42); np.random.seed(42)
    L2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load bodies
    bodies = json.load(open(OUTPUT_DIR / 'bodies_sorted.json'))
    print(f"Loaded {len(bodies)} Layer 1 bodies")

    # Load a small batch of images for forward selection
    print("Loading images for forward selection...")
    dm = ImageNetDataModule(data_dir=DATA_DIR, resolution=112, batch_size=512,
                            num_workers=8, samples_per_class=5)  # 5K images
    dm.setup()
    loader = DataLoader(dm.train_dataset, batch_size=5000, shuffle=False, num_workers=8)
    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)
    data_batch = build_data_batch(images, device)
    print(f"  {images.shape[0]} images loaded")

    # Step 1: Forward selection
    l1_selected = forward_select_bodies(bodies, data_batch, labels, n_select=100)

    # Save selected bodies
    with open(L2_OUTPUT_DIR / 'l1_selected_bodies.json', 'w') as f:
        json.dump(l1_selected, f)
    print(f"\nSaved {len(l1_selected)} selected bodies to {L2_OUTPUT_DIR / 'l1_selected_bodies.json'}")

    # Step 2: Phase 1B
    run_phase1b(l1_selected, None, L2_OUTPUT_DIR)


if __name__ == '__main__':
    main()
