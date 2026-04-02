#!/usr/bin/env python3
"""Run Phase 3 only from existing Phase 2 output.

Uses a train subset (200/class = 200K images) to fit within disk quota,
and full val set (50K) for evaluation.
"""
import os, sys, time, json, gc
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.imagenet_loader import ImageNetDataModule
from src.symbolic.large_feature_bank import LargeFeatureBank
from experiments.train_imagenet_pipeline import (
    build_data_batch, extract_features_mmap,
    _online_mean_std, _train_linear_classifier,
    _per_superclass_accuracy, _compute_baselines,
)

DEVICE = 'cuda'
CONFIG_PATH = 'configs/tensor_vsr_imagenet_single_bank.yaml'
OUTPUT_DIR = Path('outputs/imagenet_single_bank')

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

torch.manual_seed(42)
np.random.seed(42)

dataset_opts = config.get('dataset_options', {})
num_classes = dataset_opts.get('num_classes', 1000)

# Load Phase 2 formulas
phase2_dir = OUTPUT_DIR / 'phase2'
bank = LargeFeatureBank.load(str(phase2_dir / 'feature_bank'), device='cpu')
formula_strs = bank.formula_strs
n_feats = len(formula_strs)
print(f"Loaded {n_feats} validated formulas from Phase 2")

phase3_dir = OUTPUT_DIR / 'phase3'
phase3_dir.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────
# Train subset: 200 per class = 200K images (~2 GB mmap) to fit disk quota
# Val: full 50K
print("\nLoading ImageNet...")
train_module = ImageNetDataModule(
    data_dir=dataset_opts['data_dir'],
    resolution=224, batch_size=512, num_workers=8,
    samples_per_class=200,  # subset for training classifier
)
train_module.setup()
train_loader = DataLoader(
    train_module.train_dataset, batch_size=512,
    shuffle=False, num_workers=8, pin_memory=True,
)
val_loader = train_module.get_val_loader()

n_train = len(train_module.train_dataset)
n_val = len(train_module.val_dataset)
print(f"  Train subset: {n_train} images (200/class)")
print(f"  Val: {n_val} images")

# ── Extract features ───────────────────────────────────────
train_mmap_path = str(phase3_dir / 'X_train.mmap')
val_mmap_path = str(phase3_dir / 'X_val.mmap')

print(f"\nExtracting train features ({n_feats} formulas)...")
t0 = time.time()
X_train, y_train = extract_features_mmap(
    formula_strs, train_loader, DEVICE,
    mmap_path=train_mmap_path, n_total=n_train, n_feats=n_feats, tag="train"
)
np.save(str(phase3_dir / 'y_train.npy'), y_train)
print(f"  Done: ({n_train}, {n_feats}) in {(time.time()-t0)/60:.1f} min")

gc.collect(); torch.cuda.empty_cache()

print(f"\nExtracting val features...")
t0 = time.time()
X_val, y_val = extract_features_mmap(
    formula_strs, val_loader, DEVICE,
    mmap_path=val_mmap_path, n_total=n_val, n_feats=n_feats, tag="val"
)
np.save(str(phase3_dir / 'y_val.npy'), y_val)
print(f"  Done: ({n_val}, {n_feats}) in {(time.time()-t0)/60:.1f} min")

# ── StandardScaler ─────────────────────────────────────────
print("\nComputing feature statistics...")
feat_mean, feat_std = _online_mean_std(X_train, n_train, n_feats)
feat_std = np.maximum(feat_std, 1e-8)

# ── Train linear classifier (weight_decay sweep) ──────────
weight_decay_values = [1e-4, 1e-3, 1e-2, 1e-1]
all_results = {}
best_val_acc = 0.0
best_wd = None

print(f"\n{'='*60}")
print(f"  PyTorch nn.Linear — weight_decay sweep")
print(f"{'='*60}")

for wd in weight_decay_values:
    print(f"\n  weight_decay={wd} ...")
    result = _train_linear_classifier(
        X_train, y_train, X_val, y_val,
        feat_mean, feat_std, n_feats, num_classes,
        device=DEVICE, weight_decay=wd, epochs=20, lr=1e-3, batch_size=1024,
    )
    label = f"wd={wd}"
    all_results[label] = result
    print(f"    Train={result['train_top1']*100:.2f}%  "
          f"Val-Top1={result['val_top1']*100:.2f}%  "
          f"Val-Top5={result['val_top5']*100:.2f}%  "
          f"({result['time_s']:.1f}s)")

    if result['val_top1'] > best_val_acc:
        best_val_acc = result['val_top1']
        best_wd = wd

# ── Per-Superclass Accuracy ───────────────────────────────
print(f"\n{'='*60}")
print(f"  Per-Superclass Accuracy (best wd={best_wd})")
print(f"{'='*60}")

best_result = all_results[f"wd={best_wd}"]
superclass_accs = _per_superclass_accuracy(
    X_val, y_val, feat_mean, feat_std,
    n_feats, num_classes, DEVICE, best_wd, best_result.get('model_state')
)
for sc_name, sc_acc in sorted(superclass_accs.items(), key=lambda x: x[1]):
    print(f"    {sc_name:<25s} {sc_acc*100:6.2f}%")

# ── Baselines ─────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Baselines")
print(f"{'='*60}")

baselines = _compute_baselines(
    X_train, y_train, X_val, y_val,
    feat_mean, feat_std, n_feats, num_classes, DEVICE
)
for bname, bval in baselines.items():
    print(f"    {bname:<35s} Top1={bval['top1']*100:6.2f}%  Top5={bval['top5']*100:6.2f}%")

# ── Final Report ──────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  FINAL RESULTS ({n_feats} symbolic features)")
print(f"{'='*60}")
print(f"  {'Config':<15} {'Train':>8} {'Val-T1':>8} {'Val-T5':>8}")
print(f"  {'-'*43}")
for name, r in all_results.items():
    print(f"  {name:<15} {r['train_top1']*100:>7.2f}% "
          f"{r['val_top1']*100:>7.2f}% "
          f"{r['val_top5']*100:>7.2f}%")
print(f"  {'-'*43}")
best_r = all_results[f'wd={best_wd}']
print(f"  BEST: wd={best_wd}  "
      f"Top-1={best_r['val_top1']*100:.2f}%  "
      f"Top-5={best_r['val_top5']*100:.2f}%")
print(f"{'='*60}\n")

# Save results
save_results = {k: {kk: vv for kk, vv in v.items() if kk != 'model_state'} for k, v in all_results.items()}
with open(phase3_dir / 'final_results.json', 'w') as f:
    json.dump({
        'num_formulas': n_feats,
        'train_samples': n_train,
        'val_samples': n_val,
        'best_weight_decay': best_wd,
        'best_val_top1': float(best_val_acc),
        'all_results': save_results,
        'superclass_accuracies': superclass_accs,
        'baselines': baselines,
    }, f, indent=2)
print(f"Results saved to {phase3_dir / 'final_results.json'}")
