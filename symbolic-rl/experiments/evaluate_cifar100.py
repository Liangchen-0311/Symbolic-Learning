#!/usr/bin/env python3
"""
Evaluation script for CIFAR-100 Neurosymbolic RL.

Pipeline:
  1. Load the trained Feature Bank (up to 1000 symbolic formulas).
  2. Forward-pass every formula on the FULL train set (50 000) and test set (10 000)
     to produce feature matrices of shape (N, num_formulas).
  3. Apply sklearn.preprocessing.StandardScaler (zero-mean, unit-variance).
  4. Fit sklearn.linear_model.LogisticRegression(max_iter=5000, n_jobs=-1,
     multi_class='multinomial') on the training features.
  5. Report and log Train Accuracy / Test Accuracy.

Usage:
    python experiments/evaluate_cifar100.py \
        --config configs/tensor_vsr_a100_cifar100.yaml
"""

import os
import sys
import json
import time
import argparse

import yaml
import numpy as np
import torch

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.mnist_loader import MNISTDataModule
from src.symbolic.tensor_operators import TENSOR_OPERATORS
from src.symbolic.large_feature_bank import LargeFeatureBank

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ======================================================================
# Formula execution (same RPN interpreter used elsewhere)
# ======================================================================

def execute_formula(formula_str: str, data_batch: dict) -> torch.Tensor:
    """Execute a single RPN formula on a batch of RGB+GRAY channels.

    Args:
        formula_str: e.g. "I_R edge_x blur global_avg_pool"
        data_batch:  {'I_R': [B,H,W], 'I_G': ..., 'I_B': ..., 'I_GRAY': ...}

    Returns:
        [B] scalar features or [B, D] multi-dim features.
    """
    tokens = formula_str.strip().split()
    stack = []
    for token in tokens:
        if token in data_batch:
            stack.append(data_batch[token])
        elif token in TENSOR_OPERATORS:
            op_func, arity, _ = TENSOR_OPERATORS[token]
            if len(stack) < arity:
                raise ValueError(f"Stack underflow at {token}")
            operands = [stack.pop() for _ in range(arity)]
            operands.reverse()
            result = op_func(*operands)
            if torch.isnan(result).any() or torch.isinf(result).any():
                raise ValueError(f"NaN/Inf at {token}")
            stack.append(result)
    if len(stack) != 1:
        raise ValueError(f"Bad stack depth {len(stack)}")
    return stack[0]


# ======================================================================
# Feature extraction
# ======================================================================

def _build_data_batch(images, device):
    """Build terminal dict from RGB images, including GRAY and HSV."""
    images = images.to(device)
    I_R = images[:, 0, :, :]
    I_G = images[:, 1, :, :]
    I_B = images[:, 2, :, :]
    I_GRAY = 0.2989 * I_R + 0.5870 * I_G + 0.1140 * I_B

    data_batch = {'I_R': I_R, 'I_G': I_G, 'I_B': I_B, 'I_GRAY': I_GRAY}

    # HSV conversion (needed for formulas using I_H / I_S)
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

    data_batch['I_H'] = H
    data_batch['I_S'] = S
    return data_batch


def extract_features(formula_strs, loader, device, tag=""):
    """Apply every formula to every image in *loader*.

    Returns:
        features: (num_images, num_formulas) float32 numpy
        labels:   (num_images,) int64 numpy
    """
    all_features, all_labels = [], []

    for batch_idx, (images, labels) in enumerate(loader):
        data_batch = _build_data_batch(images, device)

        batch_feats = []
        for formula_str in formula_strs:
            try:
                out = execute_formula(formula_str, data_batch)
                out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
                out = torch.clamp(out, -1e4, 1e4)
                # Ensure 2D: [B] -> [B, 1], [B, D] stays
                if out.dim() == 1:
                    out = out.unsqueeze(1)
                batch_feats.append(out.cpu().numpy())
            except Exception:
                # Failed formula -> zero column (1 dim)
                batch_feats.append(np.zeros((images.shape[0], 1), dtype=np.float32))

        all_features.append(np.concatenate(batch_feats, axis=1))   # (B, total_dims)
        all_labels.append(labels.numpy())

        if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
            done = sum(f.shape[0] for f in all_features)
            print(f"  [{tag}] batch {batch_idx+1}/{len(loader)}  "
                  f"({done} images)")

    return (np.concatenate(all_features, axis=0),
            np.concatenate(all_labels, axis=0))


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate neurosymbolic feature bank on CIFAR-100")
    parser.add_argument('--config', default='configs/tensor_vsr_a100_cifar100.yaml')
    parser.add_argument('--bank_path', default=None,
                        help='Override feature bank directory')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device or config.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA unavailable — falling back to CPU")
        device = 'cpu'

    dataset_name = config.get('dataset', 'cifar100')
    output_dir = config.get('output_dir', 'outputs/cifar100')
    bank_path = args.bank_path or os.path.join(output_dir, 'feature_bank')

    # ---- Load feature bank -------------------------------------------
    print(f"\nLoading feature bank from {bank_path} ...")
    bank = LargeFeatureBank.load(bank_path, device=device)
    formula_strs = bank.formula_strs
    print(f"  {len(formula_strs)} formulas loaded")
    print(bank.get_summary())

    # ---- Load dataset (CIFAR or ImageNet) ----------------------------
    batch_size = config.get('training', {}).get('eval_batch_size',
                  config.get('training', {}).get('batch_size', 2048))

    if dataset_name == 'imagenet':
        from src.data.imagenet_loader import ImageNetDataModule
        dataset_opts = config.get('dataset_options', {}) or {}
        resolution = dataset_opts.get('resolution_full', 224)
        data_module = ImageNetDataModule(
            data_dir=dataset_opts.get('data_dir', '/data/imagenet'),
            resolution=resolution,
            batch_size=batch_size,
            num_workers=8,
            samples_per_class=None,  # full dataset for final eval
        )
    else:
        data_module = MNISTDataModule(
            dataset=dataset_name,
            batch_size=batch_size,
            num_workers=4,
            val_split=0.0,
        )
    data_module.setup()

    train_loader = data_module.get_train_loader()
    test_loader = data_module.get_test_loader()

    print(f"\nDataset: {dataset_name.upper()}")
    print(f"  Train images : {len(data_module.train_dataset)}")
    print(f"  Test images  : {len(data_module.test_dataset)}")

    # ---- Extract features --------------------------------------------
    print(f"\nExtracting training features ({len(formula_strs)} formulas) ...")
    t0 = time.time()
    X_train, y_train = extract_features(
        formula_strs, train_loader, device, tag="train")
    print(f"  shape {X_train.shape}  ({time.time()-t0:.1f}s)")

    print(f"\nExtracting test features ...")
    t0 = time.time()
    X_test, y_test = extract_features(
        formula_strs, test_loader, device, tag="test")
    print(f"  shape {X_test.shape}  ({time.time()-t0:.1f}s)")

    # ---- StandardScaler ----------------------------------------------
    print("\nStandardScaler (zero-mean, unit-variance) ...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Safety: kill any residual NaN / Inf
    X_train_s = np.nan_to_num(X_train_s)
    X_test_s = np.nan_to_num(X_test_s)

    # ---- Logistic Regression: sweep C values ---------------------------
    all_results = {}
    C_values = [1.0, 0.1, 0.01, 0.001]

    print(f"\n{'='*60}")
    print(f"  LogisticRegression — C-value sweep")
    print(f"{'='*60}")

    for C_val in C_values:
        print(f"\n  C={C_val} ...", end=" ", flush=True)
        t0 = time.time()
        clf = LogisticRegression(
            C=C_val,
            max_iter=5000,
            solver='lbfgs',
        )
        clf.fit(X_train_s, y_train)
        fit_time = time.time() - t0

        tr_acc = clf.score(X_train_s, y_train)
        te_acc = clf.score(X_test_s, y_test)
        gap = tr_acc - te_acc
        print(f"Train={tr_acc*100:.2f}%  Test={te_acc*100:.2f}%  "
              f"Gap={gap*100:.2f}%  ({fit_time:.1f}s)")

        all_results[f'LR_C={C_val}'] = {
            'train_accuracy': float(tr_acc),
            'test_accuracy': float(te_acc),
            'gap': float(gap),
            'fit_time_s': fit_time,
        }

    # ---- XGBoost ---------------------------------------------------------
    from xgboost import XGBClassifier

    xgb_configs = [
        {'name': 'XGB_default',
         'params': dict(n_estimators=300, max_depth=6, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8,
                        tree_method='hist', device='cuda',
                        verbosity=0, random_state=42)},
        {'name': 'XGB_deep',
         'params': dict(n_estimators=500, max_depth=8, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.6,
                        reg_alpha=0.1, reg_lambda=1.0,
                        tree_method='hist', device='cuda',
                        verbosity=0, random_state=42)},
        {'name': 'XGB_regularized',
         'params': dict(n_estimators=500, max_depth=5, learning_rate=0.05,
                        subsample=0.7, colsample_bytree=0.5,
                        reg_alpha=0.5, reg_lambda=2.0,
                        min_child_weight=5,
                        tree_method='hist', device='cuda',
                        verbosity=0, random_state=42)},
    ]

    print(f"\n{'='*60}")
    print(f"  XGBoost Classifiers")
    print(f"{'='*60}")

    for xgb_cfg in xgb_configs:
        name = xgb_cfg['name']
        params = xgb_cfg['params']
        print(f"\n  {name} ...", flush=True)
        t0 = time.time()
        xgb_clf = XGBClassifier(**params)
        xgb_clf.fit(X_train_s, y_train)
        fit_time = time.time() - t0

        tr_acc = xgb_clf.score(X_train_s, y_train)
        te_acc = xgb_clf.score(X_test_s, y_test)
        gap = tr_acc - te_acc
        print(f"    Train={tr_acc*100:.2f}%  Test={te_acc*100:.2f}%  "
              f"Gap={gap*100:.2f}%  ({fit_time:.1f}s)")

        all_results[name] = {
            'train_accuracy': float(tr_acc),
            'test_accuracy': float(te_acc),
            'gap': float(gap),
            'fit_time_s': fit_time,
        }

    # ---- Final Report ----------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  ALL RESULTS  ({len(formula_strs)} symbolic features, {dataset_name.upper()})")
    print(f"{'='*60}")
    print(f"  {'Method':<22} {'Train':>8} {'Test':>8} {'Gap':>8}")
    print(f"  {'-'*48}")
    best_test = 0.0
    best_name = ''
    for name, r in all_results.items():
        print(f"  {name:<22} {r['train_accuracy']*100:>7.2f}% {r['test_accuracy']*100:>7.2f}% {r['gap']*100:>7.2f}%")
        if r['test_accuracy'] > best_test:
            best_test = r['test_accuracy']
            best_name = name
    print(f"  {'-'*48}")
    print(f"  BEST: {best_name}  Test={best_test*100:.2f}%")
    print(f"{'='*60}\n")

    # ---- Persist results -------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")

    return all_results


if __name__ == '__main__':
    main()
