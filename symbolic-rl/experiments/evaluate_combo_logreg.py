#!/usr/bin/env python3
"""
Combine feature banks from multiple runs and evaluate with logistic regression.

Usage:
    python experiments/evaluate_combo_logreg.py \
        --banks outputs/cifar10_spp/feature_bank outputs/cifar10_v3/feature_bank \
        --dataset cifar10
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.mnist_loader import MNISTDataModule
from src.symbolic.tensor_operators import TENSOR_OPERATORS
from src.symbolic.large_feature_bank import LargeFeatureBank


def execute_formula(formula_str, data_batch):
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


def extract_features(formula_strs, loader, device, tag=""):
    all_features, all_labels = [], []
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        I_R = images[:, 0, :, :]
        I_G = images[:, 1, :, :]
        I_B = images[:, 2, :, :]
        data_batch = {
            'I_R': I_R, 'I_G': I_G, 'I_B': I_B,
            'I_GRAY': 0.2989 * I_R + 0.5870 * I_G + 0.1140 * I_B,
        }
        batch_feats = []
        for formula_str in formula_strs:
            try:
                out = execute_formula(formula_str, data_batch)
                out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
                out = torch.clamp(out, -1e4, 1e4)
                if out.dim() == 1:
                    out = out.unsqueeze(1)
                batch_feats.append(out.cpu().numpy())
            except Exception:
                batch_feats.append(np.zeros((images.shape[0], 1), dtype=np.float32))

        all_features.append(np.concatenate(batch_feats, axis=1))
        all_labels.append(labels.numpy())
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            done = sum(f.shape[0] for f in all_features)
            print(f"  [{tag}] batch {batch_idx+1}/{len(loader)}  "
                  f"({done} images, {all_features[-1].shape[1]} dims)")

    return (np.concatenate(all_features, axis=0),
            np.concatenate(all_labels, axis=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--banks', nargs='+', required=True,
                        help='Paths to feature bank directories')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_dir', default='outputs/cifar10_combo_v2')
    parser.add_argument('--min_acc', type=float, default=0.14,
                        help='Filter formulas below this accuracy (removes injected ones)')
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    # --- Load and combine banks ---
    all_formulas = []  # (formula_str, accuracy, source)
    seen = set()

    for bank_path in args.banks:
        print(f"\nLoading {bank_path} ...")
        with open(os.path.join(bank_path, 'feature_bank.json')) as f:
            meta = json.load(f)

        count = 0
        for entry in meta['formulas']:
            fstr = entry['str']
            acc = entry['accuracy']
            # Filter: skip low-accuracy (injected) and duplicates
            if acc < args.min_acc:
                continue
            if fstr in seen:
                continue
            seen.add(fstr)
            all_formulas.append((fstr, acc))
            count += 1
        print(f"  Loaded {count} formulas (after filter acc>={args.min_acc}, dedup)")

    formula_strs = [f[0] for f in all_formulas]
    print(f"\nTotal combined formulas: {len(formula_strs)}")

    # --- Load dataset ---
    data_module = MNISTDataModule(
        dataset=args.dataset, batch_size=2048,
        num_workers=4, val_split=0.0,
    )
    data_module.setup()
    print(f"Dataset: {args.dataset.upper()}")
    print(f"  Train: {len(data_module.train_dataset)}")
    print(f"  Test:  {len(data_module.test_dataset)}")

    # --- Extract features ---
    print(f"\nExtracting training features ({len(formula_strs)} formulas) ...")
    t0 = time.time()
    X_train, y_train = extract_features(
        formula_strs, data_module.get_train_loader(), device, tag="train")
    print(f"  shape {X_train.shape}  ({time.time()-t0:.1f}s)")

    print(f"\nExtracting test features ...")
    t0 = time.time()
    X_test, y_test = extract_features(
        formula_strs, data_module.get_test_loader(), device, tag="test")
    print(f"  shape {X_test.shape}  ({time.time()-t0:.1f}s)")

    # --- Standardize ---
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = np.nan_to_num(scaler.fit_transform(X_train))
    X_test_s = np.nan_to_num(scaler.transform(X_test))
    input_dim = X_train_s.shape[1]
    print(f"\n  Feature dimension: {input_dim}")

    # --- Logistic Regression ---
    from sklearn.linear_model import LogisticRegression

    C_values = [0.01, 0.1, 0.5, 1.0, 5.0]

    print(f"\n{'='*60}")
    print(f"  Logistic Regression — Combo ({input_dim} dims)")
    print(f"{'='*60}")
    print(f"  {'C':<10} {'Train':>8} {'Test':>8} {'Time':>8}")
    print(f"  {'-'*38}")

    all_results = {}
    best_test = 0.0
    best_C = None

    for C_val in C_values:
        t0 = time.time()
        clf = LogisticRegression(C=C_val, max_iter=5000, solver='lbfgs')
        clf.fit(X_train_s, y_train)
        tr_acc = clf.score(X_train_s, y_train)
        te_acc = clf.score(X_test_s, y_test)
        elapsed = time.time() - t0

        marker = " *" if te_acc > best_test else ""
        print(f"  {C_val:<10} {tr_acc*100:>7.2f}% {te_acc*100:>7.2f}% {elapsed:>7.1f}s{marker}")

        all_results[f'C={C_val}'] = {
            'train': float(tr_acc), 'test': float(te_acc),
            'dims': input_dim, 'time_s': float(elapsed),
        }
        if te_acc > best_test:
            best_test = te_acc
            best_C = C_val

    print(f"  {'-'*38}")
    print(f"  BEST: C={best_C}  Test={best_test*100:.2f}%")
    print(f"{'='*60}\n")

    # --- Save ---
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, 'eval_logreg_results.json')
    all_results['_meta'] = {
        'banks': args.banks,
        'total_formulas': len(formula_strs),
        'total_dims': input_dim,
        'min_acc_filter': args.min_acc,
    }
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
