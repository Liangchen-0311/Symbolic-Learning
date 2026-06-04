"""
Train Tensor VSR with Large Feature Bank Strategy.

This implements the "Over-generate & Prune" strategy:
1. Generate 200 formulas with low threshold (10%)
2. Use strong LASSO (L1=0.5) to select 60-100 best features
3. Target accuracy: 50%+ (from current 20%)

Key improvements:
- Large feature bank (200 formulas)
- Strong LASSO with automatic feature selection
- Spatial pooling operators (position-aware)
- Diversity-aware rewards

Usage:
    # M1 Mac (MPS)
    python experiments/train_tensor_vsr_large_bank.py \
        --config configs/tensor_vsr_m1_cifar10_large_bank.yaml \
        --dataset cifar10 \
        --device mps

    # CUDA
    python experiments/train_tensor_vsr_large_bank.py \
        --config configs/tensor_vsr_m1_cifar10_large_bank.yaml \
        --dataset cifar10 \
        --device cuda
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import time
import json

from src.data.mnist_loader import MNISTDataModule
from src.models.policy_agent import PolicyAgent
from src.rl.tensor_environment_large_bank import TensorVSREnvironmentLargeBank
from src.rl.ppo_trainer import PPOTrainer
from src.symbolic.large_feature_bank import LargeFeatureBank


def main():
    parser = argparse.ArgumentParser(description='Train Tensor VSR with Large Feature Bank')
    parser.add_argument('--config', type=str,
                       default='configs/tensor_vsr_m1_cifar10_large_bank.yaml',
                       help='Path to config file')
    parser.add_argument('--dataset', type=str, default=None,
                       choices=['cifar10', 'cifar100', 'imagenet'],
                       help='Dataset to use (default: from config)')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu', 'mps'],
                       help='Device to use (default: from config)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: from config or outputs/tensor_vsr_large_bank)')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Resume training from a checkpoint directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Fill defaults from config if not specified on CLI
    if args.dataset is None:
        args.dataset = config.get('dataset', 'cifar10')
    if args.device is None:
        args.device = config.get('device', 'cuda')

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    # Set device
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU")
            device = 'cpu'
        else:
            device = 'cuda'
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif args.device == 'mps':
        if not torch.backends.mps.is_available():
            print("WARNING: MPS not available, falling back to CPU")
            device = 'cpu'
        else:
            device = 'mps'
            print("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = 'cpu'
        print("Using CPU")

    # Override config device
    config['training']['device'] = device

    # Read strategy params (from 'strategy' section with 'training' fallback)
    strategy_cfg = config.get('strategy', {})
    train_cfg = config['training']

    fb_size = strategy_cfg.get('feature_bank_size',
               train_cfg.get('feature_bank_size', 1000))
    min_acc = strategy_cfg.get('min_accuracy_threshold',
               train_cfg.get('min_accuracy', 0.015))
    corr_thr = strategy_cfg.get('correlation_threshold', 0.90)
    div_pen = strategy_cfg.get('diversity_penalty',
               train_cfg.get('diversity_penalty', 0.15))
    l1_lam = strategy_cfg.get('l1_lambda', train_cfg.get('l1_lambda', 0.0))
    n_iters = train_cfg.get('iterations', train_cfg.get('n_iterations', 5000))

    # Print config
    print("\n" + "=" * 60)
    print("SCHEME A: SURVIVAL OF THE FITTEST — TRAINING CONFIG")
    print("=" * 60)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Device: {device}")
    print(f"\nStrategy Parameters (Scheme A):")
    print(f"  Feature Bank Size: {fb_size}")
    print(f"  Min Accuracy Threshold: {min_acc}")
    print(f"  Correlation Threshold: {corr_thr}")
    print(f"  Diversity Penalty: {div_pen}")
    print(f"  L1 Lambda: {l1_lam}")
    print(f"\nModel Parameters:")
    print(f"  Max Depth: {config['model']['max_depth']}")
    print(f"  Max Sequence Length: {config['model']['max_sequence_length']}")
    print(f"\nTraining Parameters:")
    print(f"  Iterations: {n_iters}")
    print(f"  Episodes/Iteration: {train_cfg['episodes_per_iteration']}")
    print(f"  Batch Size: {train_cfg['batch_size']}")
    print(f"  Learning Rate: {train_cfg['learning_rate']}")
    print("=" * 60 + "\n")

    # Load data
    print("Loading data...")
    dataset_opts = config.get('dataset_options', {}) or {}

    if args.dataset == 'imagenet':
        from src.data.imagenet_loader import ImageNetDataModule, build_imagenet_superclass_mapping
        resolution = dataset_opts.get('resolution_quick', 64)
        samples_per_class = dataset_opts.get('samples_per_class_quick', 20)
        data_module = ImageNetDataModule(
            data_dir=dataset_opts.get('data_dir', '/data/imagenet'),
            resolution=resolution,
            batch_size=config['training']['batch_size'],
            num_workers=8,
            samples_per_class=samples_per_class,
        )
    else:
        data_module = MNISTDataModule(
            dataset=args.dataset,
            batch_size=config['training']['batch_size'],
            num_workers=0,
            train_subset=dataset_opts.get('train_subset', None),
            test_subset=dataset_opts.get('test_subset', None),
        )
    data_module.setup()

    # Get train loader (used for validation in RL)
    train_loader = data_module.get_train_loader()
    print(f"Loaded {len(train_loader.dataset)} training samples")

    # Create environment with large feature bank
    print("\nCreating environment with Large Feature Bank...")
    env = TensorVSREnvironmentLargeBank(
        data_loader=train_loader,
        config=config,
        device=device
    )
    print(f"Environment created with {len(env.vocabulary)} tokens")
    print(f"Spatial pooling operators enabled: {6} new operators")

    # Create policy agent
    print("\nCreating policy agent...")
    policy = PolicyAgent(
        vocab_size=len(env.vocabulary),
        embedding_dim=config['model']['embedding_dim'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)

    num_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy network created with {num_params:,} parameters")

    # Create trainer — branch on search_algorithm (v3.3 Section 2C).
    # Both PPO and GRPO share the same outward interface (update/train/save_checkpoint),
    # so the rest of this entrypoint is identical for either.
    search_algorithm = str(config.get('search_algorithm', 'ppo')).lower()
    common_kwargs = dict(
        policy=policy,
        env=env,
        learning_rate=config['training']['learning_rate'],
        clip_epsilon=config['training']['clip_epsilon'],
        entropy_coef=config['training']['entropy_coef'],
        max_grad_norm=config['training'].get('max_grad_norm', 0.5),
        n_epochs=config['training']['n_epochs_ppo'],
        batch_size=config['training']['batch_size_ppo'],
        device=device,
        entropy_coef_start=train_cfg.get('entropy_coef_start', None),
        entropy_coef_end=train_cfg.get('entropy_coef_end', None),
        entropy_decay_fraction=train_cfg.get('entropy_decay_fraction', 0.5),
        lr_warmup_iterations=train_cfg.get('lr_warmup_iterations', 0),
        total_iterations=n_iters,
    )
    if search_algorithm == 'grpo':
        from src.rl.grpo_trainer import GRPOTrainer
        grpo_cfg = config.get('grpo', {})
        # Section 2E: classifier-dependent parsimony strength λ_len. Use an explicit
        # grpo.lambda_len if given; otherwise derive from classifier.type — linear → strict
        # (1e-3, anti-bloat is free since a linear model just sums features), histgb →
        # relaxed (2e-4, long formulas may carry non-linear structure HistGB exploits).
        clf_type = str(config.get('classifier', {}).get('type', 'linear')).lower()
        default_lambda = 1.0e-3 if clf_type in ('linear', 'ebm') else 2.0e-4
        lambda_len = grpo_cfg.get('lambda_len', default_lambda)
        print("\nCreating GRPO trainer (critic-free, Pareto group-relative advantage)...")
        print(f"  Section 2E λ_len = {lambda_len} (classifier.type={clf_type})")
        trainer = GRPOTrainer(
            group_size=grpo_cfg.get('group_size', 16),
            acc_tol=grpo_cfg.get('acc_tol', 0.003),
            crowding_weight=grpo_cfg.get('crowding_weight', 0.1),
            lambda_len=lambda_len,
            **common_kwargs,
        )
    else:
        print("\nCreating PPO trainer...")
        trainer = PPOTrainer(
            gamma=config['training']['gamma'],
            gae_lambda=config['training'].get('gae_lambda', 0.95),
            value_coef=config['training']['value_coef'],
            **common_kwargs,
        )

    # Set binary operator bias for cross-channel fusion
    binary_op_bias = train_cfg.get('binary_op_bias', 0.0)
    if binary_op_bias > 0:
        trainer.set_binary_op_bias(binary_op_bias, env.vocabulary)
        print(f"  Binary operator bias: +{binary_op_bias} on subtract & multiply logits")

    # Set up hierarchical evaluation for ImageNet
    if args.dataset == 'imagenet' and strategy_cfg.get('use_hierarchical_eval', False):
        superclass_map = build_imagenet_superclass_mapping(data_module.train_dataset)
        env.set_superclass_mapping(superclass_map, num_superclasses=20)
        print("  Hierarchical evaluation enabled (20 superclasses)")

    # Create output directory
    output_dir = Path(args.output_dir or config.get('output_dir', 'outputs/tensor_vsr_large_bank'))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save config
    config_path = output_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Saved config to: {config_path}")

    # Resume from checkpoint if specified
    start_iteration = 0
    if args.resume_from:
        resume_dir = Path(args.resume_from)
        # Load feature bank
        bank_path = resume_dir / 'feature_bank'
        if bank_path.exists():
            env.feature_bank = LargeFeatureBank.load(str(bank_path), device=device)
            # Restore bank config
            env.feature_bank.max_size = env.feature_bank_size
            print(f"Resumed feature bank: {env.feature_bank.size()} formulas")
        # Load policy
        ckpt_path = resume_dir / 'checkpoint_latest.pt'
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            policy.load_state_dict(ckpt['policy_state_dict'])
            start_iteration = ckpt.get('iteration', 0)
            trainer.iteration_count = start_iteration
            print(f"Resumed policy from iteration {start_iteration}")

    checkpoint_interval = config.get('checkpoint_interval', 500)

    # Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60 + "\n")

    start_time = time.time()
    best_accuracy = 0.0
    results = []

    try:
        for iteration in range(start_iteration, n_iters):
            iter_start = time.time()

            # Train for one iteration (collect episodes and update policy)
            metrics = trainer.update(n_episodes=config['training']['episodes_per_iteration'])

            # v3.3 Section 7B: periodic group-level reshuffle (no-op unless enabled in config).
            reshuffle_cfg = config.get('bank_reshuffle', None)
            if reshuffle_cfg:
                from src.symbolic.bank_reshuffle import maybe_reshuffle_bank
                maybe_reshuffle_bank(env, iteration + 1, reshuffle_cfg)

            iter_time = time.time() - iter_start
            elapsed_time = time.time() - start_time

            # Log results — compact single line per iteration
            reward_str = f"R={metrics.get('avg_reward', 0):.3f}"
            loss_str = f"L={metrics.get('loss', metrics.get('policy_loss', 0)):.3f}"
            bank_str = f"Bank={env.feature_bank.size()}/{env.feature_bank.max_size}"
            print(f"[Iter {iteration+1:>5}/{n_iters}] "
                  f"{iter_time:.1f}s  {reward_str}  {loss_str}  {bank_str}  "
                  f"(total: {elapsed_time/60:.1f}min)")

            # Save metrics
            result = {
                'iteration': iteration,
                'elapsed_time': elapsed_time,
                'bank_size': env.feature_bank.size(),
                **metrics
            }
            results.append(result)

            # Update hierarchical eval state (switches off once bank fills)
            if hasattr(env, 'update_hierarchical_state'):
                env.update_hierarchical_state()

            # Periodic checkpointing
            if checkpoint_interval > 0 and (iteration + 1) % checkpoint_interval == 0:
                ckpt_data = {
                    'iteration': iteration + 1,
                    'policy_state_dict': policy.state_dict(),
                    'bank_size': env.feature_bank.size(),
                }
                torch.save(ckpt_data, output_dir / 'checkpoint_latest.pt')
                env.feature_bank.save(str(output_dir / 'feature_bank'))
                print(f"  [Checkpoint] Saved at iteration {iteration + 1}")

            # Print bank summary every 100 iterations
            if (iteration + 1) % 100 == 0:
                bank = env.feature_bank
                print(bank.get_summary())

                if bank.size() > 0:
                    bank_accs = bank.accuracies
                    current_best = max(bank_accs)
                    current_mean = sum(bank_accs) / len(bank_accs)
                    if current_best > best_accuracy:
                        best_accuracy = current_best
                    print(f"  >>> Mean Acc: {current_mean*100:.2f}%  "
                          f"Max Acc: {current_best*100:.2f}%  "
                          f"Best Ever: {best_accuracy*100:.2f}%")
                    print(f"  >>> Fill Rate: {bank.size()}/{bank.max_size} "
                          f"({bank.size()/bank.max_size*100:.1f}%)")
                    print(f"  >>> Turnover: added={bank.total_added} "
                          f"replaced={bank.total_replaced} "
                          f"rejected={bank.total_rejected}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    if env.feature_bank.size() > 0:
        # Save final feature bank
        bank_dir = output_dir / 'feature_bank'
        env.feature_bank.save(str(bank_dir))

        # Save final model
        final_model_path = output_dir / 'final_model.pt'
        torch.save({
            'iteration': n_iters,
            'policy_state_dict': policy.state_dict(),
            'bank_size': env.feature_bank.size(),
            'best_individual_accuracy': best_accuracy,
        }, final_model_path)
        print(f"Saved final model to: {final_model_path}")

        print(env.feature_bank.get_summary())

        # Save final results
        bank_accs = env.feature_bank.accuracies
        final_results = {
            'total_formulas': env.feature_bank.size(),
            'bank_acc_min': float(min(bank_accs)),
            'bank_acc_mean': float(sum(bank_accs) / len(bank_accs)),
            'bank_acc_max': float(max(bank_accs)),
            'total_added': env.feature_bank.total_added,
            'total_replaced': env.feature_bank.total_replaced,
            'total_rejected': env.feature_bank.total_rejected,
            'formula_list': env.feature_bank.get_selected_formulas(),
            'training_time': time.time() - start_time
        }

        results_path = output_dir / 'final_results.json'
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)

        print(f"\n{'='*60}")
        print("TRAINING COMPLETE (Scheme A)")
        print(f"{'='*60}")
        print(f"Total Time: {(time.time() - start_time)/3600:.2f} hours")
        print(f"Feature Bank: {env.feature_bank.size()}/{env.feature_bank.max_size}")
        print(f"  Added: {env.feature_bank.total_added}  "
              f"Replaced: {env.feature_bank.total_replaced}  "
              f"Rejected: {env.feature_bank.total_rejected}")
        print(f"Acc range: min={min(bank_accs):.4f} mean={sum(bank_accs)/len(bank_accs):.4f} max={max(bank_accs):.4f}")
        print(f"\nRun evaluate_cifar100.py next for StandardScaler + LogisticRegression.")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*60}\n")

    else:
        print("No formulas generated. Training may have failed.")

    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Training history saved to: {history_path}")


if __name__ == '__main__':
    main()
