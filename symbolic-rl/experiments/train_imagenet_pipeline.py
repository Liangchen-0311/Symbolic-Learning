#!/usr/bin/env python3
"""
Three-Phase ImageNet Training Pipeline for Symbolic Feature Discovery.

Phase 1: Fast Discovery (Low Resolution)
    - Train RL agent at 64×64 resolution to rapidly discover candidate formulas
    - Uses stratified mini-batch (20 images/class = 20,000 total)
    - Supports multi-bank training (4 independent banks)
    - Output: 30,000-40,000 candidate formulas across all banks

Phase 2: Full-Resolution Validation
    - Re-evaluate all Phase 1 formulas at 224×224 resolution
    - Drop formulas with >30% relative accuracy drop
    - Deduplicate across banks using Pearson correlation (threshold < 0.78)
    - Output: 15,000-25,000 validated, diverse formulas

Phase 3: Final Classification
    - Extract full feature matrix on entire train/val set
    - StandardScaler + multinomial LogisticRegression (L-BFGS, L2)
    - Cross-validate C ∈ {0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0}
    - Report top-1 and top-5 accuracy on ImageNet validation set

Usage:
    # Full pipeline (all three phases)
    python experiments/train_imagenet_pipeline.py \
        --config configs/tensor_vsr_imagenet.yaml \
        --device cuda \
        --output_dir outputs/imagenet_pipeline

    # Run only Phase 2 + 3 (from existing Phase 1 output)
    python experiments/train_imagenet_pipeline.py \
        --config configs/tensor_vsr_imagenet.yaml \
        --start_phase 2 \
        --phase1_dir outputs/imagenet_pipeline/phase1

    # Run only Phase 3 (from existing Phase 2 output)
    python experiments/train_imagenet_pipeline.py \
        --config configs/tensor_vsr_imagenet.yaml \
        --start_phase 3 \
        --phase2_dir outputs/imagenet_pipeline/phase2
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.imagenet_loader import ImageNetDataModule, build_imagenet_superclass_mapping
from src.models.policy_agent import PolicyAgent
from src.rl.tensor_environment_large_bank import TensorVSREnvironmentLargeBank
from src.rl.ppo_trainer import PPOTrainer
from src.symbolic.large_feature_bank import LargeFeatureBank
from src.symbolic.tensor_operators import TENSOR_OPERATORS


# ======================================================================
# Formula execution helpers
# ======================================================================

def build_data_batch(images, device):
    """Build terminal dict from a batch of RGB images [B, 3, H, W]."""
    images = images.to(device)
    I_R = images[:, 0]
    I_G = images[:, 1]
    I_B = images[:, 2]

    # Grayscale
    I_GRAY = 0.2989 * I_R + 0.5870 * I_G + 0.1140 * I_B

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

    # Color ratios (illumination invariant) + opponent channels
    total = I_R + I_G + I_B + 1e-8

    return {
        'I_R': I_R, 'I_G': I_G, 'I_B': I_B,
        'I_GRAY': I_GRAY, 'I_H': H, 'I_S': S,
        'I_r': I_R / total, 'I_g': I_G / total,
        'I_RG': I_R - I_G, 'I_BY': I_B - (I_R + I_G) / 2,
    }


def execute_formula(formula_str, data_batch):
    """Execute a single RPN formula. Returns output tensor or None on failure."""
    tokens = formula_str.strip().split()
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
            if torch.isnan(result).any() or torch.isinf(result).any():
                return None
            stack.append(result)
        else:
            return None
    if len(stack) != 1:
        return None
    out = stack[0]
    out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
    return torch.clamp(out, -1e4, 1e4)


def evaluate_formula_accuracy(formula_str, data_batch, labels, num_classes, device):
    """Evaluate a formula's top-1 accuracy via quick linear probe."""
    out = execute_formula(formula_str, data_batch)
    if out is None:
        return 0.0

    # Build feature tensor
    if out.dim() == 1:
        feat = out.unsqueeze(1)
    else:
        feat = out

    feat_mean = feat.mean(dim=0, keepdim=True)
    feat_std = feat.std(dim=0, keepdim=True) + 1e-8
    feat = (feat - feat_mean) / feat_std

    # Quick linear probe (20 steps)
    model = torch.nn.Linear(feat.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for _ in range(20):
        optimizer.zero_grad()
        loss = criterion(model(feat), labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(feat).argmax(dim=1)
        acc = (preds == labels).float().mean().item()
    return acc


# ======================================================================
# Phase 1: Fast Discovery
# ======================================================================

def run_phase1(config, device, output_dir, num_banks=1):
    """Phase 1: Train RL agents at low resolution to discover formulas."""
    print("\n" + "=" * 70)
    print("  PHASE 1: FAST DISCOVERY (Low Resolution)")
    print("=" * 70)

    dataset_opts = config.get('dataset_options', {}) or {}
    resolution = dataset_opts.get('resolution_quick', 64)
    samples_per_class = dataset_opts.get('samples_per_class_quick', 20)

    multi_bank_cfg = config.get('multi_bank', {})
    if multi_bank_cfg.get('enabled', False):
        num_banks = multi_bank_cfg.get('num_banks', num_banks)
        bank_configs = multi_bank_cfg.get('bank_configs', [])
    else:
        bank_configs = []

    train_cfg = config['training']
    strategy_cfg = config.get('strategy', {})
    n_iters = train_cfg.get('iterations', 10000)

    all_bank_dirs = []

    for bank_id in range(num_banks):
        bank_dir = output_dir / f'bank_{bank_id}'
        bank_dir.mkdir(parents=True, exist_ok=True)
        all_bank_dirs.append(bank_dir)

        print(f"\n--- Bank {bank_id}/{num_banks} ---")

        # Override config for this bank if multi-bank configs are available
        bank_config = dict(config)
        if bank_id < len(bank_configs):
            bc = bank_configs[bank_id]
            bank_config = {**config}
            bank_config['model'] = {**config['model']}
            bank_config['training'] = {**config['training']}
            for key in ('max_depth', 'max_sequence_length'):
                if key in bc:
                    bank_config['model'][key] = bc[key]
            # Per-bank binary_op_bias override
            if 'binary_op_bias' in bc:
                bank_config['training']['binary_op_bias'] = bc['binary_op_bias']
            print(f"  Bank config: max_depth={bank_config['model']['max_depth']}, "
                  f"max_seq_len={bank_config['model']['max_sequence_length']}, "
                  f"binary_op_bias={bank_config['training'].get('binary_op_bias', 0.0)}, "
                  f"focus={bc.get('focus', 'default')}")

        # Load data at quick resolution
        data_module = ImageNetDataModule(
            data_dir=dataset_opts.get('data_dir', '/data/imagenet'),
            resolution=resolution,
            batch_size=train_cfg['batch_size'],
            num_workers=8,
            samples_per_class=samples_per_class,
        )
        data_module.setup()
        train_loader = data_module.get_train_loader()

        # Create environment
        env = TensorVSREnvironmentLargeBank(
            data_loader=train_loader,
            config=bank_config,
            device=device,
        )

        # Hierarchical evaluation
        if strategy_cfg.get('use_hierarchical_eval', False):
            base_ds = data_module.train_dataset
            while hasattr(base_ds, 'dataset'):
                base_ds = base_ds.dataset
            superclass_map = build_imagenet_superclass_mapping(base_ds)
            env.set_superclass_mapping(superclass_map, num_superclasses=20)

        # Create policy
        policy = PolicyAgent(
            vocab_size=len(env.vocabulary),
            embedding_dim=bank_config['model']['embedding_dim'],
            hidden_size=bank_config['model']['hidden_size'],
            num_layers=bank_config['model']['num_layers'],
            dropout=bank_config['model']['dropout'],
        ).to(device)

        # Create PPO trainer
        trainer = PPOTrainer(
            policy=policy, env=env,
            learning_rate=train_cfg['learning_rate'],
            gamma=train_cfg['gamma'],
            gae_lambda=train_cfg.get('gae_lambda', 0.95),
            clip_epsilon=train_cfg['clip_epsilon'],
            value_coef=train_cfg['value_coef'],
            entropy_coef=train_cfg['entropy_coef'],
            max_grad_norm=train_cfg.get('max_grad_norm', 0.5),
            n_epochs=train_cfg['n_epochs_ppo'],
            batch_size=train_cfg['batch_size_ppo'],
            device=device,
            entropy_coef_start=train_cfg.get('entropy_coef_start'),
            entropy_coef_end=train_cfg.get('entropy_coef_end'),
            entropy_decay_fraction=train_cfg.get('entropy_decay_fraction', 0.5),
            lr_warmup_iterations=train_cfg.get('lr_warmup_iterations', 0),
            total_iterations=n_iters,
        )

        binary_op_bias = train_cfg.get('binary_op_bias', 0.0)
        if binary_op_bias > 0:
            trainer.set_binary_op_bias(binary_op_bias, env.vocabulary)

        checkpoint_interval = config.get('checkpoint_interval', 500)

        # Training loop
        print(f"  Training {n_iters} iterations...")
        start_time = time.time()

        # Early stopping: if bank stagnates for 50 consecutive iterations, skip to next bank
        stagnation_window = 50
        stagnation_count = 0
        prev_bank_size = 0
        prev_total_replaced = 0

        for iteration in range(n_iters):
            metrics = trainer.update(n_episodes=train_cfg['episodes_per_iteration'])

            if hasattr(env, 'update_hierarchical_state'):
                env.update_hierarchical_state()

            # Check for stagnation (every iteration)
            cur_bank_size = env.feature_bank.size()
            cur_total_replaced = env.feature_bank.total_replaced
            if cur_bank_size == prev_bank_size and (cur_total_replaced - prev_total_replaced) < 10:
                stagnation_count += 1
            else:
                stagnation_count = 0
                prev_bank_size = cur_bank_size
                prev_total_replaced = cur_total_replaced

            if stagnation_count >= stagnation_window:
                elapsed = time.time() - start_time
                print(f"  [Bank {bank_id}] Early stop at iter {iteration+1}: "
                      f"no growth & <10 replacements for {stagnation_window} iters "
                      f"({elapsed/60:.1f}min)")
                break

            if (iteration + 1) % 100 == 0:
                elapsed = time.time() - start_time
                bank_size = env.feature_bank.size()
                reward = metrics.get('avg_reward', 0)
                print(f"  [Bank {bank_id}] Iter {iteration+1}/{n_iters}  "
                      f"R={reward:.3f}  Bank={bank_size}/{env.feature_bank.max_size}  "
                      f"stag={stagnation_count}/{stagnation_window}  "
                      f"({elapsed/60:.1f}min)")

            if checkpoint_interval > 0 and (iteration + 1) % checkpoint_interval == 0:
                ckpt_data = {
                    'iteration': iteration + 1,
                    'policy_state_dict': policy.state_dict(),
                    'bank_size': env.feature_bank.size(),
                }
                torch.save(ckpt_data, bank_dir / 'checkpoint_latest.pt')
                env.feature_bank.save(str(bank_dir / 'feature_bank'))

        # Save final bank
        env.feature_bank.save(str(bank_dir / 'feature_bank'))
        elapsed = time.time() - start_time
        print(f"  Bank {bank_id} done: {env.feature_bank.size()} formulas in {elapsed/3600:.2f}h")
        print(env.feature_bank.get_summary())

    # Save phase 1 metadata
    phase1_meta = {
        'num_banks': num_banks,
        'bank_dirs': [str(d) for d in all_bank_dirs],
        'resolution': resolution,
        'total_time_s': time.time() - start_time if num_banks == 1 else None,
    }
    with open(output_dir / 'phase1_meta.json', 'w') as f:
        json.dump(phase1_meta, f, indent=2)

    return all_bank_dirs


# ======================================================================
# Phase 2: Full-Resolution Validation
# ======================================================================

def run_phase2(config, device, output_dir, phase1_dirs):
    """Phase 2: Re-evaluate formulas at full resolution and filter.

    Two-stage approach:
      Stage A: Extract output vectors (batch-wise forward only, no probes)
               → matrix [n_images, n_formulas] in CPU RAM (~800MB)
      Stage B: Vectorized Pearson dedup + single linear probe on survivors
    """
    print("\n" + "=" * 70)
    print("  PHASE 2: FULL-RESOLUTION VALIDATION")
    print("=" * 70)

    dataset_opts = config.get('dataset_options', {}) or {}
    strategy_cfg = config.get('strategy', {})
    resolution_full = dataset_opts.get('resolution_full', 224)
    samples_per_class_full = dataset_opts.get('samples_per_class_full', 20)
    num_classes = dataset_opts.get('num_classes', 1000)
    corr_threshold_full = strategy_cfg.get('correlation_threshold_full', 0.95)

    # Load data at full resolution
    print(f"Loading ImageNet at {resolution_full}×{resolution_full} "
          f"({samples_per_class_full} per class)...")
    data_module = ImageNetDataModule(
        data_dir=dataset_opts.get('data_dir', '/data/imagenet'),
        resolution=resolution_full,
        batch_size=512,
        num_workers=8,
        samples_per_class=samples_per_class_full,
    )
    data_module.setup()
    eval_loader = data_module.get_train_loader()
    n_eval = len(data_module.train_dataset)
    print(f"  Eval set: {n_eval} images at {resolution_full}×{resolution_full}")

    # Collect all formulas from all Phase 1 banks
    all_formulas = []  # list of (formula_str, low_res_acc, bank_id)
    for bank_id, bank_dir in enumerate(phase1_dirs):
        bank_path = Path(bank_dir) / 'feature_bank'
        if not bank_path.exists():
            print(f"  WARNING: Bank not found at {bank_path}, skipping")
            continue
        bank = LargeFeatureBank.load(str(bank_path), device='cpu')
        for fstr, acc in zip(bank.formula_strs, bank.accuracies):
            all_formulas.append((fstr, acc, bank_id))
        print(f"  Bank {bank_id}: {bank.size()} formulas loaded")

    n_formulas = len(all_formulas)
    print(f"\nTotal candidate formulas: {n_formulas}")

    # ================================================================
    # Stage A: Extract output vectors (forward only, no probes)
    # ================================================================
    print(f"\n--- Stage A: Extracting output vectors ({n_formulas} formulas) ---")
    t0 = time.time()

    # Pre-allocate output matrix on CPU: [n_eval, n_formulas]
    out_matrix = np.zeros((n_eval, n_formulas), dtype=np.float32)
    all_labels = []
    formula_failed = np.zeros(n_formulas, dtype=bool)
    row_offset = 0

    for batch_idx, (images, labels) in enumerate(eval_loader):
        B = images.shape[0]
        data_batch = build_data_batch(images, device)
        all_labels.append(labels.numpy())

        for f_idx, (fstr, _, _) in enumerate(all_formulas):
            if formula_failed[f_idx]:
                continue
            out = execute_formula(fstr, data_batch)
            if out is None:
                formula_failed[f_idx] = True
                continue
            # Reduce to scalar per image
            if out.dim() > 1:
                out = out.mean(dim=tuple(range(1, out.dim())))
            out_matrix[row_offset:row_offset + B, f_idx] = out.cpu().numpy()

        del data_batch
        torch.cuda.empty_cache()
        row_offset += B

        if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
            elapsed = time.time() - t0
            print(f"  batch {batch_idx+1}/{len(eval_loader)}  "
                  f"({row_offset}/{n_eval} images, {elapsed:.1f}s)")

    all_labels = np.concatenate(all_labels)
    n_failed = formula_failed.sum()
    print(f"Stage A done in {(time.time()-t0)/60:.1f} min  "
          f"({n_failed} formulas failed)")

    # ================================================================
    # Stage B: Vectorized dedup + linear probe on survivors
    # ================================================================
    print(f"\n--- Stage B: Dedup (threshold={corr_threshold_full}) + Accuracy ---")
    t0 = time.time()

    # Filter out failed formulas
    valid_mask = ~formula_failed
    valid_indices = np.where(valid_mask)[0]
    # Also filter constant-output formulas
    stds = out_matrix[:, valid_indices].std(axis=0)
    non_const_mask = stds > 1e-10
    valid_indices = valid_indices[non_const_mask]

    print(f"  Valid formulas (non-failed, non-constant): {len(valid_indices)}/{n_formulas}")

    # Vectorized Pearson correlation dedup
    # Standardize columns (zero-mean, unit-std)
    X = out_matrix[:, valid_indices].copy()  # [n_eval, n_valid]
    X -= X.mean(axis=0, keepdims=True)
    X /= (X.std(axis=0, keepdims=True) + 1e-8)

    # Greedy dedup: sort by Phase-1 accuracy (descending), keep if not too correlated
    # with any already-kept formula
    p1_accs = np.array([all_formulas[i][1] for i in valid_indices])
    sorted_order = np.argsort(-p1_accs)

    kept_local = []  # indices into valid_indices
    kept_vecs = []   # standardized column vectors

    for rank, local_idx in enumerate(sorted_order):
        vec = X[:, local_idx]  # already standardized

        is_diverse = True
        if kept_vecs:
            # Vectorized correlation check against all kept
            kept_mat = np.column_stack(kept_vecs)  # [n_eval, n_kept]
            corrs = np.abs(vec @ kept_mat / n_eval)
            if corrs.max() >= corr_threshold_full:
                is_diverse = False

        if is_diverse:
            kept_local.append(local_idx)
            kept_vecs.append(vec)

        if (rank + 1) % 2000 == 0:
            print(f"    dedup progress: {rank+1}/{len(sorted_order)}, kept={len(kept_local)}")

    kept_global_indices = valid_indices[np.array(kept_local)]
    print(f"  After dedup: {len(kept_global_indices)} unique formulas  "
          f"({(time.time()-t0):.1f}s)")

    # Linear probe accuracy for each surviving formula
    print(f"\n  Evaluating accuracy for {len(kept_global_indices)} formulas...")
    t0 = time.time()
    labels_t = torch.tensor(all_labels, dtype=torch.long, device=device)

    kept_results = []
    for rank, g_idx in enumerate(kept_global_indices):
        fstr, low_acc, bank_id = all_formulas[g_idx]

        # Get feature vector and standardize
        feat_np = out_matrix[:, g_idx]
        feat_t = torch.tensor(feat_np, dtype=torch.float32, device=device).unsqueeze(1)
        feat_mean = feat_t.mean()
        feat_std = feat_t.std() + 1e-8
        feat_t = (feat_t - feat_mean) / feat_std

        # Quick linear probe (20 steps)
        probe = torch.nn.Linear(1, num_classes).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        probe.train()
        for _ in range(20):
            opt.zero_grad()
            criterion(probe(feat_t), labels_t).backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            acc = (probe(feat_t).argmax(1) == labels_t).float().mean().item()

        kept_results.append({
            'formula': fstr,
            'low_res_acc': low_acc,
            'full_res_acc': acc,
            'bank_id': bank_id,
        })

        if (rank + 1) % 500 == 0:
            print(f"    probe {rank+1}/{len(kept_global_indices)}  ({time.time()-t0:.1f}s)")

        del probe, opt, feat_t

    print(f"  Probes done in {(time.time()-t0):.1f}s")

    # Drop zero-accuracy formulas
    kept_results = [r for r in kept_results if r['full_res_acc'] > 0.0]
    print(f"\n  Final: {len(kept_results)} formulas with acc > 0")

    # Free large matrix
    del out_matrix, X

    # ================================================================
    # Save Phase 2 results
    # ================================================================
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_bank = LargeFeatureBank(
        max_size=len(kept_results) + 100,
        min_accuracy=0.0,
        correlation_threshold=corr_threshold_full,
        num_classes=num_classes,
        device='cpu',
    )

    for entry in kept_results:
        merged_bank.formulas.append(None)
        merged_bank.formula_strs.append(entry['formula'])
        merged_bank.formula_lengths.append(len(entry['formula'].split()))
        merged_bank.accuracies.append(entry['full_res_acc'])
        merged_bank.output_vectors.append(None)

    merged_bank.save(str(output_dir / 'feature_bank'))

    phase2_meta = {
        'total_candidates': n_formulas,
        'valid_non_constant': int(len(valid_indices)),
        'after_deduplication': len(kept_results),
        'failed_formulas': int(n_failed),
        'correlation_threshold': corr_threshold_full,
        'resolution': resolution_full,
        'samples_per_class': samples_per_class_full,
    }
    with open(output_dir / 'phase2_meta.json', 'w') as f:
        json.dump(phase2_meta, f, indent=2)

    print(f"\nPhase 2 complete. {len(kept_results)} formulas saved.")
    return output_dir


# ======================================================================
# Phase 3: Final Classification
# ======================================================================

def run_phase3(config, device, output_dir, phase2_dir, phase1_dirs=None):
    """Phase 3: Extract features, train PyTorch linear classifier, report metrics.

    Can load formulas from Phase 2 bank OR directly from Phase 1 banks
    (when Phase 2 is skipped). Sorts formulas lexicographically before
    extraction to maximize sub-expression cache hits.
    """
    # Free GPU memory from prior phases
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("  PHASE 3: FINAL CLASSIFICATION")
    print("=" * 70)

    dataset_opts = config.get('dataset_options', {}) or {}
    resolution = dataset_opts.get('resolution_quick', 112)
    num_classes = dataset_opts.get('num_classes', 1000)

    # Load formulas — try Phase 2 bank first, fall back to Phase 1 banks
    bank_path = Path(phase2_dir) / 'feature_bank'
    if bank_path.exists():
        bank = LargeFeatureBank.load(str(bank_path), device='cpu')
        formula_strs = bank.formula_strs
        print(f"Loaded {len(formula_strs)} formulas from Phase 2 bank")
    elif phase1_dirs:
        # Skip Phase 2: merge all Phase 1 banks directly
        print("Phase 2 bank not found — loading directly from Phase 1 banks")
        formula_set = set()
        formula_strs = []
        for bank_dir in phase1_dirs:
            bp = Path(bank_dir) / 'feature_bank'
            if not bp.exists():
                continue
            b = LargeFeatureBank.load(str(bp), device='cpu')
            for fstr in b.formula_strs:
                if fstr not in formula_set:
                    formula_set.add(fstr)
                    formula_strs.append(fstr)
            print(f"  {bp}: {b.size()} formulas (unique so far: {len(formula_strs)})")
        print(f"Merged {len(formula_strs)} unique formulas from Phase 1")
    else:
        raise FileNotFoundError(f"No formula bank found at {bank_path} and no Phase 1 dirs")

    # Sort formulas lexicographically to maximize sub-expression cache hits
    # e.g. "I_R edge_x blur ..." and "I_R edge_x gabor ..." share prefix
    formula_strs_sorted = sorted(formula_strs)
    print(f"Formulas sorted lexicographically for cache optimization")

    # Load ImageNet — subsample train for tractable feature extraction
    train_samples_per_class = dataset_opts.get('samples_per_class_train', 400)
    print(f"\nLoading ImageNet at {resolution}x{resolution} "
          f"(train: {train_samples_per_class}/class, val: full)...")
    train_module = ImageNetDataModule(
        data_dir=dataset_opts.get('data_dir', '/data/imagenet'),
        resolution=resolution,
        batch_size=512,
        num_workers=8,
        samples_per_class=train_samples_per_class,
    )
    train_module.setup()
    # Use shuffle=False for deterministic ordering (required for checkpoint/resume)
    train_loader = DataLoader(
        train_module.train_dataset, batch_size=512,
        shuffle=False, num_workers=8, pin_memory=True, drop_last=False,
    )
    val_loader = train_module.get_val_loader()

    n_train = len(train_module.train_dataset)
    n_val = len(train_module.val_dataset)
    n_feats = len(formula_strs_sorted)
    print(f"  Train: {n_train} images")
    print(f"  Val:   {n_val} images")
    print(f"  Features: {n_feats} formulas (sorted)")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save sorted formula list for reproducibility
    with open(output_dir / 'formula_list.json', 'w') as f:
        json.dump(formula_strs_sorted, f)

    # ── Feature Extraction (memory-mapped, sorted for cache) ───
    train_feat_path = str(output_dir / 'X_train.mmap')
    val_feat_path = str(output_dir / 'X_val.mmap')
    train_label_path = str(output_dir / 'y_train.npy')
    val_label_path = str(output_dir / 'y_val.npy')

    print(f"\nExtracting training features ({n_feats} formulas, sorted)...")
    t0 = time.time()
    X_train_mmap, y_train = extract_features_mmap(
        formula_strs_sorted, train_loader, device,
        mmap_path=train_feat_path, n_total=n_train, n_feats=n_feats, tag="train"
    )
    np.save(train_label_path, y_train)
    print(f"  Shape: ({n_train}, {n_feats})  ({(time.time()-t0)/60:.1f} min)")

    print(f"\nExtracting validation features...")
    t0 = time.time()
    X_val_mmap, y_val = extract_features_mmap(
        formula_strs_sorted, val_loader, device,
        mmap_path=val_feat_path, n_total=n_val, n_feats=n_feats, tag="val"
    )
    np.save(val_label_path, y_val)
    print(f"  Shape: ({n_val}, {n_feats})  ({(time.time()-t0)/60:.1f} min)")
    print(f"Features saved to {output_dir}")

    # ── Standardise features (online, memory-friendly) ─────────
    print("\nComputing feature statistics (online StandardScaler)...")
    feat_mean, feat_std = _online_mean_std(X_train_mmap, n_train, n_feats)
    feat_std = np.maximum(feat_std, 1e-8)
    np.save(str(output_dir / 'feat_mean.npy'), feat_mean)
    np.save(str(output_dir / 'feat_std.npy'), feat_std)

    # ── Train PyTorch Linear Classifier ────────────────────────
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
            X_train_mmap, y_train, X_val_mmap, y_val,
            feat_mean, feat_std, n_feats, num_classes,
            device=device, weight_decay=wd, epochs=20, lr=1e-3, batch_size=1024,
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

    # ── Per-Superclass Accuracy (P3) ──────────────────────────
    print(f"\n{'='*60}")
    print(f"  Per-Superclass Accuracy")
    print(f"{'='*60}")

    best_result = all_results[f"wd={best_wd}"]
    superclass_accs = _per_superclass_accuracy(
        X_val_mmap, y_val, feat_mean, feat_std,
        n_feats, num_classes, device, best_wd, best_result.get('model_state', None)
    )
    for sc_name, sc_acc in sorted(superclass_accs.items(), key=lambda x: x[1]):
        print(f"    {sc_name:<25s} {sc_acc*100:6.2f}%")

    # ── Baselines (P3) ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Baselines")
    print(f"{'='*60}")

    baselines = _compute_baselines(
        X_train_mmap, y_train, X_val_mmap, y_val,
        feat_mean, feat_std, n_feats, num_classes, device
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
    results_path = output_dir / 'final_results.json'
    # Remove non-serializable items
    save_results = {}
    for k, v in all_results.items():
        save_results[k] = {kk: vv for kk, vv in v.items() if kk != 'model_state'}
    with open(results_path, 'w') as f:
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
    print(f"Results saved to {results_path}")

    return all_results


# ======================================================================
# Feature Extraction — Memory-Mapped (P2)
# ======================================================================

def extract_features_mmap(formula_strs, loader, device, mmap_path, n_total, n_feats, tag=""):
    """Extract features to a memory-mapped file on disk with checkpoint/resume.

    Crash-safe: writes a progress file after each batch. On restart, resumes
    from the last completed batch instead of starting over.

    Requires loader to have shuffle=False for deterministic ordering.
    """
    progress_path = mmap_path + '.progress.json'
    label_path = mmap_path.replace('.mmap', '_labels_partial.npy')

    # Check for existing progress (resume after crash)
    start_batch = 0
    row_offset = 0
    all_labels = []
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
        start_batch = progress['next_batch']
        row_offset = progress['row_offset']
        # Reload labels saved so far
        if os.path.exists(label_path):
            all_labels = list(np.load(label_path, allow_pickle=True))
        print(f"  [{tag}] Resuming from batch {start_batch} (row {row_offset}/{n_total})")
        mmap = np.memmap(mmap_path, dtype='float32', mode='r+', shape=(n_total, n_feats))
    else:
        mmap = np.memmap(mmap_path, dtype='float32', mode='w+', shape=(n_total, n_feats))

    total_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        # Skip already-completed batches
        if batch_idx < start_batch:
            continue

        B = images.shape[0]
        data_batch = build_data_batch(images, device)

        batch_feats = np.empty((B, n_feats), dtype=np.float32)

        for f_idx, formula_str in enumerate(formula_strs):
            try:
                out = _execute_with_cache(formula_str, data_batch, None)
                if out is None:
                    batch_feats[:, f_idx] = 0.0
                    continue
                out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
                out = torch.clamp(out, -1e4, 1e4)
                if out.dim() > 1:
                    out = out.mean(dim=tuple(range(1, out.dim())))
                batch_feats[:, f_idx] = out.cpu().numpy()
            except Exception:
                batch_feats[:, f_idx] = 0.0

        # Free GPU memory
        del data_batch
        torch.cuda.empty_cache()

        end = min(row_offset + B, n_total)
        actual_B = end - row_offset
        mmap[row_offset:end] = batch_feats[:actual_B]
        all_labels.append(labels.numpy()[:actual_B])
        row_offset = end

        # Save checkpoint every batch (cheap: just a small JSON + label array)
        if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
            mmap.flush()
            with open(progress_path, 'w') as f:
                json.dump({'next_batch': batch_idx + 1, 'row_offset': row_offset}, f)
            np.save(label_path, np.concatenate(all_labels, axis=0))

        if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
            pct = row_offset / n_total * 100
            print(f"  [{tag}] batch {batch_idx+1}/{total_batches}  "
                  f"({row_offset}/{n_total} images, {pct:.1f}%)")

    mmap.flush()
    y = np.concatenate(all_labels, axis=0)

    # Clean up progress files on successful completion
    for p in [progress_path, label_path]:
        if os.path.exists(p):
            os.remove(p)

    return mmap, y


def extract_features_batched(formula_strs, loader, device, tag=""):
    """Extract features in-memory (fallback for small datasets)."""
    all_features = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(loader):
        data_batch = build_data_batch(images, device)
        batch_feats = []
        subexpr_cache = {}

        for formula_str in formula_strs:
            try:
                out = _execute_with_cache(formula_str, data_batch, subexpr_cache)
                if out is None:
                    raise ValueError("execution failed")
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
            print(f"  [{tag}] batch {batch_idx+1}/{len(loader)}  ({done} images)")

    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


# ======================================================================
# Online StandardScaler (P2)
# ======================================================================

def _online_mean_std(X_mmap, n_total, n_feats, chunk_size=10000):
    """Compute mean and std from memory-mapped features without loading all into RAM."""
    running_sum = np.zeros(n_feats, dtype=np.float64)
    running_sq_sum = np.zeros(n_feats, dtype=np.float64)

    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        chunk = np.array(X_mmap[start:end], dtype=np.float64)
        chunk = np.nan_to_num(chunk)
        running_sum += chunk.sum(axis=0)
        running_sq_sum += (chunk ** 2).sum(axis=0)

    mean = running_sum / n_total
    var = running_sq_sum / n_total - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))
    return mean.astype(np.float32), std.astype(np.float32)


# ======================================================================
# PyTorch Linear Classifier (P3)
# ======================================================================

def _train_linear_classifier(
    X_train_mmap, y_train, X_val_mmap, y_val,
    feat_mean, feat_std, n_feats, num_classes,
    device='cuda', weight_decay=1e-2, epochs=20, lr=1e-3, batch_size=1024,
):
    """Train a GPU-accelerated linear classifier (nn.Linear + CrossEntropyLoss).

    Uses mini-batch SGD over memory-mapped features — constant GPU memory
    regardless of dataset size.
    """
    model = torch.nn.Linear(n_feats, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    # LR schedule: cosine decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_train = len(y_train)
    mean_t = torch.tensor(feat_mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(feat_std, dtype=torch.float32, device=device)

    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]

            # Load chunk from mmap → GPU, standardize on GPU
            X_batch = torch.tensor(
                np.array(X_train_mmap[idx]), dtype=torch.float32, device=device
            )
            X_batch = torch.nan_to_num(X_batch)
            X_batch = (X_batch - mean_t) / std_t
            y_batch = torch.tensor(y_train[idx], dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"      epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")

    elapsed = time.time() - t0

    # Evaluate
    model.eval()
    train_top1 = _eval_accuracy(model, X_train_mmap, y_train, mean_t, std_t, device, batch_size)
    val_top1, val_top5 = _eval_accuracy(
        model, X_val_mmap, y_val, mean_t, std_t, device, batch_size, top5=True
    )

    return {
        'train_top1': train_top1,
        'val_top1': val_top1,
        'val_top5': val_top5,
        'weight_decay': weight_decay,
        'time_s': elapsed,
        'model_state': model.state_dict(),
    }


def _eval_accuracy(model, X_mmap, y, mean_t, std_t, device, batch_size=2048, top5=False):
    """Evaluate accuracy on memory-mapped features."""
    n = len(y)
    correct_top1 = 0
    correct_top5 = 0

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_batch = torch.tensor(
                np.array(X_mmap[start:end]), dtype=torch.float32, device=device
            )
            X_batch = torch.nan_to_num(X_batch)
            X_batch = (X_batch - mean_t) / std_t
            y_batch = torch.tensor(y[start:end], dtype=torch.long, device=device)

            logits = model(X_batch)
            preds = logits.argmax(dim=1)
            correct_top1 += (preds == y_batch).sum().item()

            if top5:
                k = min(5, logits.shape[1])
                _, top_k = logits.topk(k, dim=1)
                correct_top5 += (top_k == y_batch.unsqueeze(1)).any(dim=1).sum().item()

    if top5:
        return correct_top1 / n, correct_top5 / n
    return correct_top1 / n


# ======================================================================
# Per-Superclass Accuracy (P3)
# ======================================================================

def _per_superclass_accuracy(
    X_val_mmap, y_val, feat_mean, feat_std, n_feats, num_classes, device, weight_decay,
    model_state=None
):
    """Compute accuracy for each of the 20 ImageNet superclasses."""
    from src.data.imagenet_loader import IMAGENET_SUPERCLASS_NAMES

    # Re-create model with best weight_decay and load state
    model = torch.nn.Linear(n_feats, num_classes).to(device)
    if model_state is not None:
        model.load_state_dict(model_state)
    model.eval()

    mean_t = torch.tensor(feat_mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(feat_std, dtype=torch.float32, device=device)

    # Collect all predictions
    all_preds = []
    n = len(y_val)
    with torch.no_grad():
        for start in range(0, n, 2048):
            end = min(start + 2048, n)
            X_batch = torch.tensor(
                np.array(X_val_mmap[start:end]), dtype=torch.float32, device=device
            )
            X_batch = torch.nan_to_num(X_batch)
            X_batch = (X_batch - mean_t) / std_t
            preds = model(X_batch).argmax(dim=1).cpu().numpy()
            all_preds.append(preds)

    all_preds = np.concatenate(all_preds)

    # Map classes to superclasses (modulo mapping as fallback)
    num_superclasses = 20
    superclass_correct = {i: 0 for i in range(num_superclasses)}
    superclass_total = {i: 0 for i in range(num_superclasses)}

    for pred, true_label in zip(all_preds, y_val):
        sc = true_label % num_superclasses
        superclass_total[sc] += 1
        if pred == true_label:
            superclass_correct[sc] += 1

    results = {}
    for sc_id in range(num_superclasses):
        name = IMAGENET_SUPERCLASS_NAMES.get(sc_id, f'superclass_{sc_id}')
        total = superclass_total[sc_id]
        acc = superclass_correct[sc_id] / max(total, 1)
        results[name] = float(acc)

    return results


# ======================================================================
# Baselines (P3)
# ======================================================================

def _compute_baselines(
    X_train_mmap, y_train, X_val_mmap, y_val,
    feat_mean, feat_std, n_feats, num_classes, device
):
    """Compute baseline comparisons."""
    results = {}
    n_train = len(y_train)
    n_val = len(y_val)

    # ── Baseline 1: Random features ───────────────────────────
    print("    Computing random feature baseline...")
    n_random = min(n_feats, 1000)  # cap for speed
    rng = np.random.RandomState(42)
    X_rand_train = rng.randn(n_train, n_random).astype(np.float32)
    X_rand_val = rng.randn(n_val, n_random).astype(np.float32)

    model_rand = torch.nn.Linear(n_random, num_classes).to(device)
    optimizer = torch.optim.AdamW(model_rand.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = torch.nn.CrossEntropyLoss()

    model_rand.train()
    for epoch in range(10):
        perm = rng.permutation(n_train)
        for start in range(0, n_train, 1024):
            end = min(start + 1024, n_train)
            idx = perm[start:end]
            X_b = torch.tensor(X_rand_train[idx], device=device)
            y_b = torch.tensor(y_train[idx], dtype=torch.long, device=device)
            optimizer.zero_grad()
            criterion(model_rand(X_b), y_b).backward()
            optimizer.step()

    model_rand.eval()
    correct1 = correct5 = 0
    with torch.no_grad():
        for start in range(0, n_val, 2048):
            end = min(start + 2048, n_val)
            X_b = torch.tensor(X_rand_val[start:end], device=device)
            y_b = torch.tensor(y_val[start:end], dtype=torch.long, device=device)
            logits = model_rand(X_b)
            correct1 += (logits.argmax(1) == y_b).sum().item()
            _, tk = logits.topk(min(5, num_classes), dim=1)
            correct5 += (tk == y_b.unsqueeze(1)).any(1).sum().item()

    results['random_features'] = {'top1': correct1 / n_val, 'top5': correct5 / n_val}

    # ── Baseline 2: PCA (top-k principal components of symbolic features) ──
    print("    Computing PCA baseline...")
    n_pca = min(100, n_feats)

    # Sample a subset for PCA fit (max 50K)
    n_sample = min(50000, n_train)
    sample_idx = rng.choice(n_train, size=n_sample, replace=False)
    X_sample = np.array(X_train_mmap[sample_idx], dtype=np.float32)
    X_sample = np.nan_to_num(X_sample)
    X_sample = (X_sample - feat_mean) / feat_std

    # SVD for PCA
    X_centered = X_sample - X_sample.mean(axis=0)
    cov = X_centered.T @ X_centered / n_sample
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    pca_components = eigenvectors[:, -n_pca:][:, ::-1].copy()  # [n_feats, n_pca]

    # Project features
    pca_comp_t = torch.tensor(pca_components, dtype=torch.float32, device=device)
    mean_t = torch.tensor(feat_mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(feat_std, dtype=torch.float32, device=device)

    # Train linear on PCA features
    model_pca = torch.nn.Linear(n_pca, num_classes).to(device)
    optimizer = torch.optim.AdamW(model_pca.parameters(), lr=1e-3, weight_decay=1e-2)

    model_pca.train()
    for epoch in range(10):
        perm = rng.permutation(n_train)
        for start in range(0, n_train, 1024):
            end = min(start + 1024, n_train)
            idx = perm[start:end]
            X_b = torch.tensor(
                np.array(X_train_mmap[idx]), dtype=torch.float32, device=device
            )
            X_b = torch.nan_to_num(X_b)
            X_b = (X_b - mean_t) / std_t
            X_b = X_b @ pca_comp_t  # project to PCA space
            y_b = torch.tensor(y_train[idx], dtype=torch.long, device=device)
            optimizer.zero_grad()
            criterion(model_pca(X_b), y_b).backward()
            optimizer.step()

    model_pca.eval()
    correct1 = correct5 = 0
    with torch.no_grad():
        for start in range(0, n_val, 2048):
            end = min(start + 2048, n_val)
            X_b = torch.tensor(
                np.array(X_val_mmap[start:end]), dtype=torch.float32, device=device
            )
            X_b = torch.nan_to_num(X_b)
            X_b = (X_b - mean_t) / std_t
            X_b = X_b @ pca_comp_t
            y_b = torch.tensor(y_val[start:end], dtype=torch.long, device=device)
            logits = model_pca(X_b)
            correct1 += (logits.argmax(1) == y_b).sum().item()
            _, tk = logits.topk(min(5, num_classes), dim=1)
            correct5 += (tk == y_b.unsqueeze(1)).any(1).sum().item()

    results[f'pca_{n_pca}_components'] = {'top1': correct1 / n_val, 'top5': correct5 / n_val}

    # ── Baseline 3: Pixel mean (simplest symbolic formula: just channel means) ──
    print("    Computing pixel-mean baseline...")
    n_pixel = 6  # R, G, B, GRAY, H, S
    model_pixel = torch.nn.Linear(n_pixel, num_classes).to(device)
    optimizer = torch.optim.AdamW(model_pixel.parameters(), lr=1e-3, weight_decay=1e-2)

    model_pixel.train()
    for epoch in range(15):
        perm = rng.permutation(n_train)
        for start in range(0, n_train, 1024):
            end = min(start + 1024, n_train)
            idx = perm[start:end]
            X_b = torch.tensor(
                np.array(X_train_mmap[idx, :n_pixel]), dtype=torch.float32, device=device
            )
            X_b = torch.nan_to_num(X_b)
            y_b = torch.tensor(y_train[idx], dtype=torch.long, device=device)
            optimizer.zero_grad()
            criterion(model_pixel(X_b), y_b).backward()
            optimizer.step()

    model_pixel.eval()
    correct1 = correct5 = 0
    with torch.no_grad():
        for start in range(0, n_val, 2048):
            end = min(start + 2048, n_val)
            X_b = torch.tensor(
                np.array(X_val_mmap[start:end, :n_pixel]), dtype=torch.float32, device=device
            )
            X_b = torch.nan_to_num(X_b)
            y_b = torch.tensor(y_val[start:end], dtype=torch.long, device=device)
            logits = model_pixel(X_b)
            correct1 += (logits.argmax(1) == y_b).sum().item()
            _, tk = logits.topk(min(5, num_classes), dim=1)
            correct5 += (tk == y_b.unsqueeze(1)).any(1).sum().item()

    results['pixel_channel_means_only'] = {'top1': correct1 / n_val, 'top5': correct5 / n_val}

    return results



def _execute_with_cache(formula_str, data_batch, cache):
    """Execute formula. Only terminal values (I_R, I_G, etc.) are cached via data_batch.

    No intermediate sub-expression caching to avoid GPU OOM with large formula sets.
    """
    tokens = formula_str.strip().split()
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
            if torch.isnan(result).any() or torch.isinf(result).any():
                return None
            stack.append(result)
        else:
            return None

    if len(stack) != 1:
        return None

    out = stack[0]
    return torch.clamp(torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4), -1e4, 1e4)


# ======================================================================
# Main entry point
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Three-Phase ImageNet Training Pipeline')
    parser.add_argument('--config', type=str,
                        default='configs/tensor_vsr_imagenet.yaml')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--output_dir', type=str,
                        default='outputs/imagenet_pipeline')
    parser.add_argument('--start_phase', type=int, default=1,
                        choices=[1, 2, 3],
                        help='Start from this phase (default: 1)')
    parser.add_argument('--skip_phase2', action='store_true',
                        help='Skip Phase 2, load formulas directly from Phase 1 banks')
    parser.add_argument('--phase1_dir', type=str, default=None,
                        help='Phase 1 output dir (for starting at Phase 2)')
    parser.add_argument('--phase2_dir', type=str, default=None,
                        help='Phase 2 output dir (for starting at Phase 3)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Device
    device = args.device or config.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU")
        device = 'cpu'

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Starting from Phase {args.start_phase}")

    pipeline_start = time.time()

    # Phase 1
    if args.start_phase <= 1:
        phase1_dir = output_dir / 'phase1'
        phase1_dirs = run_phase1(config, device, phase1_dir)
    else:
        phase1_dir = Path(args.phase1_dir) if args.phase1_dir else output_dir / 'phase1'
        # Load bank dirs from metadata
        meta_path = phase1_dir / 'phase1_meta.json'
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            phase1_dirs = [Path(d) for d in meta['bank_dirs']]
        else:
            # Fallback: look for bank_* directories
            phase1_dirs = sorted(phase1_dir.glob('bank_*'))
            if not phase1_dirs:
                phase1_dirs = [phase1_dir]

    # Phase 2
    if args.skip_phase2:
        print("Skipping Phase 2 — formulas will be loaded directly from Phase 1 banks")
        phase2_dir = output_dir / 'phase2'  # may not exist, that's OK
    elif args.start_phase <= 2:
        phase2_dir = output_dir / 'phase2'
        run_phase2(config, device, phase2_dir, phase1_dirs)
    else:
        phase2_dir = Path(args.phase2_dir) if args.phase2_dir else output_dir / 'phase2'

    # Phase 3
    phase3_dir = output_dir / 'phase3'
    run_phase3(config, device, phase3_dir, phase2_dir, phase1_dirs=phase1_dirs)

    total_time = time.time() - pipeline_start
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE — Total time: {total_time/3600:.2f} hours")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
