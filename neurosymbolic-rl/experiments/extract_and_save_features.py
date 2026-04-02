#!/usr/bin/env python3
"""One-shot: extract features from symbolic formulas and save as .npy for fast reuse.

Supports CIFAR-10/100 and ImageNet. Includes all terminals (RGB, GRAY, HSV).
"""

import os, sys, time, argparse
import yaml, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.symbolic.tensor_operators import TENSOR_OPERATORS
from src.symbolic.large_feature_bank import LargeFeatureBank


def _build_data_batch(images, device):
    """Build terminal dict from RGB images, including GRAY and HSV."""
    images = images.to(device)
    I_R = images[:, 0]
    I_G = images[:, 1]
    I_B = images[:, 2]
    I_GRAY = 0.2989 * I_R + 0.5870 * I_G + 0.1140 * I_B

    data_batch = {'I_R': I_R, 'I_G': I_G, 'I_B': I_B, 'I_GRAY': I_GRAY}

    # HSV conversion
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
        data_batch = _build_data_batch(images, device)
        batch_feats = []
        for fs in formula_strs:
            try:
                out = execute_formula(fs, data_batch)
                out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
                out = torch.clamp(out, -1e4, 1e4)
                if out.dim() == 1:
                    out = out.unsqueeze(1)
                batch_feats.append(out.cpu().numpy())
            except Exception:
                batch_feats.append(np.zeros((images.shape[0], 1), dtype=np.float32))
        all_features.append(np.concatenate(batch_feats, axis=1))
        all_labels.append(labels.numpy())
        if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
            done = sum(f.shape[0] for f in all_features)
            print(f"  [{tag}] batch {batch_idx+1}/{len(loader)}  ({done} images)")
    return np.concatenate(all_features), np.concatenate(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/tensor_vsr_a100_cifar100.yaml')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device or config.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    output_dir = config.get('output_dir', 'outputs/cifar100')
    bank_path = os.path.join(output_dir, 'feature_bank')
    dataset_name = config.get('dataset', 'cifar100')

    bank = LargeFeatureBank.load(bank_path, device=device)
    formula_strs = bank.formula_strs
    print(f"Loaded {len(formula_strs)} formulas")

    batch_size = config.get('training', {}).get('eval_batch_size', 2048)

    if dataset_name == 'imagenet':
        from src.data.imagenet_loader import ImageNetDataModule
        dataset_opts = config.get('dataset_options', {}) or {}
        resolution = dataset_opts.get('resolution_full', 224)
        dm = ImageNetDataModule(
            data_dir=dataset_opts.get('data_dir', '/data/imagenet'),
            resolution=resolution,
            batch_size=batch_size,
            num_workers=8,
            samples_per_class=None,
        )
    else:
        from src.data.mnist_loader import MNISTDataModule
        dm = MNISTDataModule(dataset=dataset_name, batch_size=batch_size,
                             num_workers=4, val_split=0.0)
    dm.setup()

    print(f"\nExtracting train features ...")
    t0 = time.time()
    X_train, y_train = extract_features(formula_strs, dm.get_train_loader(), device, "train")
    print(f"  {X_train.shape}  ({time.time()-t0:.1f}s)")

    print(f"Extracting test features ...")
    t0 = time.time()
    X_test, y_test = extract_features(formula_strs, dm.get_test_loader(), device, "test")
    print(f"  {X_test.shape}  ({time.time()-t0:.1f}s)")

    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    print(f"\nSaved to {output_dir}/X_train.npy, y_train.npy, X_test.npy, y_test.npy")


if __name__ == '__main__':
    main()
