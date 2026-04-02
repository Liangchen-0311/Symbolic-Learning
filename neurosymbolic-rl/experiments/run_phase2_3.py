#!/usr/bin/env python3
"""
Run Phase 2 + Phase 3 using an existing Phase 1 feature bank.

If no saved bank exists, re-discovers formulas with a quick Phase 1 run,
then proceeds to Phase 2 (validation + dedup) and Phase 3 (classification).

Usage:
    python experiments/run_phase2_3.py
"""

import os
import sys
import time
import json
import numpy as np
import torch
import yaml
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.imagenet_loader import ImageNetDataModule, build_imagenet_superclass_mapping
from src.models.policy_agent import PolicyAgent
from src.rl.tensor_environment_large_bank import TensorVSREnvironmentLargeBank
from src.rl.ppo_trainer import PPOTrainer
from src.symbolic.large_feature_bank import LargeFeatureBank
from experiments.train_imagenet_pipeline import (
    run_phase2, run_phase3, build_data_batch
)

DEVICE = 'cuda'
CONFIG_PATH = 'configs/tensor_vsr_imagenet_single_bank.yaml'
OUTPUT_DIR = Path('outputs/imagenet_single_bank')


def quick_phase1(config, device, output_dir, target_formulas=4000, max_iters=2000):
    """Quick Phase 1: discover formulas until bank reaches target or max_iters."""
    print("\n" + "=" * 70)
    print(f"  QUICK PHASE 1: Discover ~{target_formulas} formulas (max {max_iters} iters)")
    print("=" * 70)

    dataset_opts = config.get('dataset_options', {}) or {}
    train_cfg = config['training']
    strategy_cfg = config.get('strategy', {})

    # Load data
    dm = ImageNetDataModule(
        data_dir=dataset_opts['data_dir'],
        resolution=dataset_opts.get('resolution_quick', 64),
        batch_size=train_cfg['batch_size'],
        num_workers=4,
        samples_per_class=dataset_opts.get('samples_per_class_quick', 20),
    )
    dm.setup()
    loader = dm.get_train_loader()

    # Environment
    env = TensorVSREnvironmentLargeBank(
        data_loader=loader, config=config, device=device
    )

    # Hierarchical eval
    if strategy_cfg.get('use_hierarchical_eval', False):
        base_ds = dm.train_dataset
        while hasattr(base_ds, 'dataset'):
            base_ds = base_ds.dataset
        sc_map = build_imagenet_superclass_mapping(base_ds)
        env.set_superclass_mapping(sc_map, num_superclasses=20)

    # Policy
    model_cfg = config['model']
    policy = PolicyAgent(
        vocab_size=len(env.vocabulary),
        embedding_dim=model_cfg['embedding_dim'],
        hidden_size=model_cfg['hidden_size'],
        num_layers=model_cfg['num_layers'],
        dropout=model_cfg['dropout'],
    ).to(device)

    # Trainer
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
        total_iterations=max_iters,
    )
    bias = train_cfg.get('binary_op_bias', 0.0)
    if bias > 0:
        trainer.set_binary_op_bias(bias, env.vocabulary)

    t0 = time.time()
    for iteration in range(max_iters):
        metrics = trainer.update(n_episodes=train_cfg['episodes_per_iteration'])
        if hasattr(env, 'update_hierarchical_state'):
            env.update_hierarchical_state()

        bank_sz = env.feature_bank.size()
        if (iteration + 1) % 10 == 0:
            elapsed = time.time() - t0
            reward = metrics.get('avg_reward', 0)
            print(f"  Iter {iteration+1}/{max_iters}  "
                  f"R={reward:.3f}  Bank={bank_sz}/{env.feature_bank.max_size}  "
                  f"({elapsed/60:.1f}min)")

        if bank_sz >= target_formulas:
            print(f"\n  Reached target {target_formulas} formulas at iter {iteration+1}")
            break

    # Save bank
    bank_dir = output_dir / 'phase1' / 'bank_0'
    bank_dir.mkdir(parents=True, exist_ok=True)
    env.feature_bank.save(str(bank_dir / 'feature_bank'))

    elapsed = time.time() - t0
    print(f"  Phase 1 done: {env.feature_bank.size()} formulas in {elapsed/60:.1f} min")
    print(env.feature_bank.get_summary())

    # Save metadata
    meta = {
        'num_banks': 1,
        'bank_dirs': [str(bank_dir)],
        'resolution': dataset_opts.get('resolution_quick', 64),
        'total_formulas': env.feature_bank.size(),
    }
    phase1_dir = output_dir / 'phase1'
    with open(phase1_dir / 'phase1_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    return [bank_dir]


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    # Check if Phase 2 output already exists (skip Phase 1 + 2)
    phase2_dir = OUTPUT_DIR / 'phase2'
    phase2_bank = phase2_dir / 'feature_bank'
    if phase2_bank.exists():
        print(f"Found existing Phase 2 bank at {phase2_bank}, skipping to Phase 3")
    else:
        # Check if Phase 1 bank already exists
        bank_path = OUTPUT_DIR / 'phase1' / 'bank_0' / 'feature_bank'
        if bank_path.exists():
            print(f"Found existing Phase 1 bank at {bank_path}")
            bank = LargeFeatureBank.load(str(bank_path), device='cpu')
            print(f"  {bank.size()} formulas loaded")
            phase1_dirs = [bank_path.parent]
        else:
            print("No saved Phase 1 bank found. Running quick Phase 1...")
            phase1_dirs = quick_phase1(config, DEVICE, OUTPUT_DIR, target_formulas=3500)

        # Phase 2
        run_phase2(config, DEVICE, phase2_dir, phase1_dirs)

    # Phase 3
    phase3_dir = OUTPUT_DIR / 'phase3'
    run_phase3(config, DEVICE, phase3_dir, phase2_dir)

    total_time = time.time()
    print(f"\nDone! Results in {phase3_dir / 'final_results.json'}")


if __name__ == '__main__':
    main()
