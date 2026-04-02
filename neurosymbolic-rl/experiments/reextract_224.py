#!/usr/bin/env python3
"""Re-extract 224×224 features with consistent kernel bank weights."""

import gc, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.imagenet_loader import ImageNetDataModule
from src.symbolic.tensor_operators import TENSOR_OPERATORS, ROOT_OPERATORS, SymbolicKernelBank

OUTPUT_DIR = Path('outputs/imagenet_v3/phase3_v3')
DATA_DIR = '/workspace/neurosymbolic-rl/data/imagenet'
N_ENCODINGS = 12

# Load saved kernel bank weights for consistency
kb = SymbolicKernelBank(device='cuda')
weights_path = OUTPUT_DIR / 'kernel_bank_weights.pt'
if weights_path.exists():
    kb.load_state_dict(torch.load(str(weights_path), weights_only=True))
    print(f"Loaded kernel bank weights from {weights_path}")
kb.register_operators(TENSOR_OPERATORS)

bodies = json.load(open(OUTPUT_DIR / 'bodies_sorted.json'))
n_bodies = len(bodies)
n_feats = n_bodies * N_ENCODINGS

SPP_POOLS = ['global_avg_pool', 'global_max_pool', 'global_std_pool',
             'pool_quad_tl', 'pool_quad_tr', 'pool_quad_bl', 'pool_quad_br', 'pool_center']
SPP_POOL_FUNCS = [(name, TENSOR_OPERATORS[name][0]) for name in SPP_POOLS]


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
    return {'I_R': I_R, 'I_G': I_G, 'I_B': I_B, 'I_GRAY': I_GRAY,
            'I_H': H, 'I_S': S, 'I_r': I_R/total, 'I_g': I_G/total,
            'I_RG': I_R - I_G, 'I_BY': I_B - (I_R + I_G) / 2}


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
    out = stack[0]
    out = torch.clamp(out, -1e4, 1e4)
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


print(f"Re-extracting 224×224: {n_bodies} bodies × {N_ENCODINGS} = {n_feats} features")

dm = ImageNetDataModule(data_dir=DATA_DIR, resolution=224, batch_size=256, num_workers=8, samples_per_class=200)
dm.setup()
train_loader = DataLoader(dm.train_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
val_loader = dm.get_val_loader()
n_train, n_val = len(dm.train_dataset), len(dm.val_dataset)

for split, loader, n_total in [('train', train_loader, n_train), ('val', val_loader, n_val)]:
    mmap_path = str(OUTPUT_DIR / f'X_{split}_224.mmap')
    progress_path = mmap_path + '.progress.json'

    start_batch = 0
    row_offset = 0
    all_labels = []

    if os.path.exists(progress_path):
        with open(progress_path) as f:
            p = json.load(f)
        start_batch = p['next_batch']
        row_offset = p['row_offset']
        if os.path.exists(mmap_path.replace('.mmap', '_labels_partial.npy')):
            all_labels = [np.load(mmap_path.replace('.mmap', '_labels_partial.npy'))]
        print(f"  [{split}] Resuming from batch {start_batch} (row {row_offset}/{n_total})")
        mmap = np.memmap(mmap_path, dtype='float32', mode='r+', shape=(n_total, n_feats))
    else:
        mmap = np.memmap(mmap_path, dtype='float32', mode='w+', shape=(n_total, n_feats))

    total_batches = len(loader)
    t0 = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx < start_batch: continue
        B = images.shape[0]
        data_batch = build_data_batch(images, 'cuda')
        gpu_buf = torch.zeros(B, n_feats, device='cuda')
        for b_idx, body_str in enumerate(bodies):
            try:
                fm = execute_body(body_str, data_batch)
                if fm is not None:
                    gpu_buf[:, b_idx*N_ENCODINGS:(b_idx+1)*N_ENCODINGS] = encode_feature_map(fm)
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
            np.save(mmap_path.replace('.mmap', '_labels_partial.npy'), np.concatenate(all_labels))
        if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
            print(f"  [{split}@224] batch {batch_idx+1}/{total_batches} ({row_offset}/{n_total}, {row_offset/n_total*100:.1f}%) {time.time()-t0:.0f}s")

    mmap.flush()
    for p in [progress_path, mmap_path.replace('.mmap', '_labels_partial.npy')]:
        if os.path.exists(p): os.remove(p)
    print(f"  [{split}@224] Done: ({n_total}, {n_feats}) in {(time.time()-t0)/60:.1f} min")

print("224 re-extraction complete!")
