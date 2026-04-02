#!/usr/bin/env python3
"""
Inject hand-crafted formulas into a feature bank.

Bypasses min_accuracy gate but still checks for duplicates and correlation.
Computes output vectors and accuracy on the eval batch so injected formulas
participate in the survival-of-the-fittest replacement like any other.

Usage:
    python experiments/inject_formulas.py \
        --bank_path outputs/cifar10_v3/feature_bank \
        --config configs/tensor_vsr_cifar10_v3.yaml
"""

import os
import sys
import json
import argparse

import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.mnist_loader import MNISTDataModule
from src.symbolic.tensor_operators import TENSOR_OPERATORS
from src.symbolic.large_feature_bank import LargeFeatureBank


# ======================================================================
# Hand-crafted formula candidates for CIFAR-10
# ======================================================================

HANDCRAFTED_FORMULAS = [
    # --- Basic channel statistics (4 channels x 4 poolings = 16) ---
    "I_R global_avg_pool",
    "I_G global_avg_pool",
    "I_B global_avg_pool",
    "I_GRAY global_avg_pool",
    "I_R global_std_pool",
    "I_G global_std_pool",
    "I_B global_std_pool",
    "I_GRAY global_std_pool",
    "I_R global_max_pool",
    "I_G global_max_pool",
    "I_B global_max_pool",
    "I_GRAY global_max_pool",
    "I_R global_l2_pool",
    "I_G global_l2_pool",
    "I_B global_l2_pool",
    "I_GRAY global_l2_pool",

    # --- Cross-channel color differences ---
    "I_R I_G subtract global_avg_pool",
    "I_R I_B subtract global_avg_pool",
    "I_G I_B subtract global_avg_pool",
    "I_R I_G subtract abs global_avg_pool",
    "I_R I_B subtract abs global_avg_pool",
    "I_G I_B subtract abs global_avg_pool",
    "I_R I_G subtract global_std_pool",
    "I_R I_B subtract global_std_pool",
    "I_G I_B subtract global_std_pool",

    # --- Color products (interaction terms) ---
    "I_R I_G multiply global_avg_pool",
    "I_R I_B multiply global_avg_pool",
    "I_G I_B multiply global_avg_pool",
    "I_R I_G multiply global_std_pool",

    # --- Edge magnitude per channel ---
    "I_GRAY edge_x abs global_avg_pool",
    "I_GRAY edge_y abs global_avg_pool",
    "I_R edge_x abs global_avg_pool",
    "I_G edge_x abs global_avg_pool",
    "I_B edge_x abs global_avg_pool",
    "I_R edge_y abs global_avg_pool",
    "I_G edge_y abs global_avg_pool",
    "I_B edge_y abs global_avg_pool",
    "I_GRAY edge_x global_std_pool",
    "I_GRAY edge_y global_std_pool",

    # --- Texture energy (Laplacian, high-freq) ---
    "I_GRAY laplacian abs global_avg_pool",
    "I_GRAY laplacian abs global_std_pool",
    "I_R laplacian abs global_avg_pool",
    "I_G laplacian abs global_avg_pool",
    "I_B laplacian abs global_avg_pool",
    "I_GRAY laplacian global_std_pool",

    # --- Morphological gradient (edge thickness) ---
    "I_GRAY dilate I_GRAY erode subtract global_avg_pool",
    "I_GRAY dilate I_GRAY erode subtract global_std_pool",
    "I_R dilate I_R erode subtract global_avg_pool",

    # --- Smoothness (blur residual) ---
    "I_GRAY blur I_GRAY subtract abs global_avg_pool",
    "I_GRAY blur I_GRAY subtract abs global_std_pool",
    "I_R blur I_R subtract abs global_avg_pool",

    # --- Sharpness ---
    "I_GRAY sharpen global_std_pool",
    "I_GRAY sharpen global_avg_pool",
    "I_GRAY sharpen abs global_avg_pool",

    # --- Spatial features (positional) ---
    "I_R pool_top_half",
    "I_G pool_top_half",
    "I_B pool_top_half",
    "I_GRAY pool_top_half",
    "I_R pool_bottom_half",
    "I_G pool_bottom_half",
    "I_B pool_bottom_half",
    "I_GRAY pool_bottom_half",
    "I_R pool_center",
    "I_G pool_center",
    "I_B pool_center",
    "I_GRAY pool_center",
    "I_R pool_corners",
    "I_G pool_corners",
    "I_B pool_corners",
    "I_GRAY pool_corners",
    "I_R pool_left_half",
    "I_R pool_right_half",
    "I_B pool_left_half",
    "I_B pool_right_half",

    # --- Spatial color differences (center vs corners, top vs bottom) ---
    "I_B pool_top_half",       # blue sky → airplane/bird
    "I_G pool_bottom_half",    # green ground → deer/horse
    "I_GRAY edge_x abs pool_center",
    "I_GRAY edge_y abs pool_center",
    "I_GRAY laplacian abs pool_center",

    # --- Normalized features ---
    "I_GRAY normalize global_std_pool",
    "I_R normalize global_avg_pool",
    "I_GRAY normalize edge_x abs global_avg_pool",
    "I_GRAY normalize edge_y abs global_avg_pool",

    # --- Cross-channel edges ---
    "I_R edge_x I_G edge_x subtract global_avg_pool",
    "I_R edge_y I_G edge_y subtract global_avg_pool",
    "I_R edge_x I_B edge_x subtract abs global_avg_pool",

    # --- Deeper compositions ---
    "I_R I_G subtract edge_x abs global_avg_pool",
    "I_R I_B subtract edge_y abs global_avg_pool",
    "I_R I_G subtract laplacian abs global_avg_pool",
    "I_GRAY edge_x abs I_GRAY edge_y abs add global_avg_pool",  # total edge magnitude
    "I_GRAY edge_x abs I_GRAY edge_y abs multiply global_avg_pool",  # corner-ness
    "I_GRAY blur laplacian abs global_avg_pool",  # LoG (blob detection)
    "I_GRAY blur edge_x abs global_avg_pool",     # smoothed edges
    "I_GRAY blur edge_y abs global_avg_pool",
]


def execute_formula(formula_str: str, data_batch: dict) -> torch.Tensor:
    """Execute a single RPN formula. Returns [B] or [B, D]."""
    tokens = formula_str.strip().split()
    stack = []
    for token in tokens:
        if token in data_batch:
            stack.append(data_batch[token])
        elif token in TENSOR_OPERATORS:
            op_func, arity, _ = TENSOR_OPERATORS[token]
            if len(stack) < arity:
                raise ValueError(f"Stack underflow at '{token}' (need {arity}, have {len(stack)})")
            operands = [stack.pop() for _ in range(arity)]
            operands.reverse()
            result = op_func(*operands)
            if torch.isnan(result).any() or torch.isinf(result).any():
                raise ValueError(f"NaN/Inf at '{token}'")
            stack.append(result)
        else:
            raise ValueError(f"Unknown token '{token}'")
    if len(stack) != 1:
        raise ValueError(f"Bad stack depth {len(stack)}")
    return stack[0]


def compute_accuracy(features: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """Quick 1-feature accuracy via per-class mean nearest centroid."""
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    centroids = np.zeros((num_classes, features.shape[1]))
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            centroids[c] = features[mask].mean(axis=0)
    # Nearest centroid
    dists = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)
    preds = dists.argmin(axis=1)
    return (preds == labels).mean()


def main():
    parser = argparse.ArgumentParser(description="Inject hand-crafted formulas into feature bank")
    parser.add_argument('--bank_path', required=True, help='Path to feature bank directory')
    parser.add_argument('--config', default='configs/tensor_vsr_cifar10_v3.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--skip_correlation', action='store_true',
                        help='Skip correlation check (force inject all)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    num_classes = config.get('training', {}).get('num_classes', 10)

    # --- Load feature bank ---
    print(f"\nLoading feature bank from {args.bank_path} ...")
    bank = LargeFeatureBank.load(args.bank_path, device=device)
    print(bank.get_summary())
    original_size = bank.size()

    # --- Load dataset (use same eval batch size as training) ---
    # Bank output_vectors were computed on training eval_batch_size (e.g. 2048),
    # so we must use the same size for correlation checks to work.
    eval_batch_size = config.get('training', {}).get('eval_batch_size',
                      config.get('training', {}).get('batch_size', 2048))

    data_module = MNISTDataModule(
        dataset=config.get('dataset', 'cifar10'),
        batch_size=eval_batch_size,
        num_workers=4,
        val_split=0.0,
    )
    data_module.setup()

    # Use first eval_batch_size images from train set (same as training env)
    train_loader = data_module.get_train_loader()
    images, labels = next(iter(train_loader))
    images = images[:eval_batch_size].to(device)
    labels = labels[:eval_batch_size].numpy()
    print(f"  Using eval batch of {len(labels)} images (matching bank vector size)")

    I_R = images[:, 0, :, :]
    I_G = images[:, 1, :, :]
    I_B = images[:, 2, :, :]
    data_batch = {
        'I_R': I_R,
        'I_G': I_G,
        'I_B': I_B,
        'I_GRAY': 0.2989 * I_R + 0.5870 * I_G + 0.1140 * I_B,
    }

    print(f"\nEval images: {images.shape[0]}")
    print(f"Candidate formulas: {len(HANDCRAFTED_FORMULAS)}")

    # --- Deduplicate candidates ---
    unique_formulas = list(dict.fromkeys(HANDCRAFTED_FORMULAS))
    print(f"Unique candidates: {len(unique_formulas)}")

    # --- Execute and inject ---
    injected = 0
    skipped_dup = 0
    skipped_corr = 0
    skipped_err = 0

    for i, formula_str in enumerate(unique_formulas):
        # Skip if already in bank
        if formula_str in bank.formula_strs:
            skipped_dup += 1
            continue

        # Execute formula
        try:
            with torch.no_grad():
                out = execute_formula(formula_str, data_batch)
                out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
                out = torch.clamp(out, -1e4, 1e4)

                # Flatten multi-dim to mean for correlation vector
                if out.dim() > 1:
                    corr_vector = out.mean(dim=1)
                else:
                    corr_vector = out

                corr_np = corr_vector.cpu().numpy()
        except Exception as e:
            print(f"  [{i+1}] FAIL: {formula_str}  ({e})")
            skipped_err += 1
            continue

        # Compute accuracy
        feat_np = corr_np.reshape(-1, 1)
        acc = compute_accuracy(feat_np, labels, num_classes)

        # Correlation check (optional)
        if not args.skip_correlation:
            if not bank._passes_correlation_check(corr_np):
                print(f"  [{i+1}] CORR: {formula_str}  acc={acc:.4f}")
                skipped_corr += 1
                continue

        # Inject (bypass min_accuracy)
        length = len(formula_str.strip().split())
        bank._insert(None, formula_str, length, acc, corr_np)
        bank.total_added += 1
        injected += 1
        print(f"  [{i+1}] ADD:  {formula_str}  acc={acc:.4f}  bank={bank.size()}")

    # --- Save ---
    print(f"\n{'='*60}")
    print(f"  Injection complete")
    print(f"  Original bank size: {original_size}")
    print(f"  Injected: {injected}")
    print(f"  Skipped (duplicate): {skipped_dup}")
    print(f"  Skipped (correlated): {skipped_corr}")
    print(f"  Skipped (error): {skipped_err}")
    print(f"  New bank size: {bank.size()}")
    print(f"{'='*60}")

    bank.save(args.bank_path)
    print(f"\nSaved to {args.bank_path}")
    print(bank.get_summary())


if __name__ == '__main__':
    main()
