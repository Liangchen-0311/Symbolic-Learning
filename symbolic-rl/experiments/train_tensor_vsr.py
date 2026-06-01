"""
Train tensor-based VSR+RL on CIFAR-10/100.

This script trains a neuro-symbolic system that:
1. Works directly on raw CIFAR images (no CNN encoder)
2. Uses tensor operators (blur, edge detection, pooling, etc.)
3. Learns formulas like: global_avg_pool(blur(add(I_R, I_G)))
4. Uses LASSO for feature selection
5. Employs action masking (root must be pooling operator)

Usage:
    # CIFAR-10
    CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/train_tensor_vsr.py \
        --dataset cifar10 \
        --gpu_ids 0,1,2,3

    # CIFAR-100
    CUDA_VISIBLE_DEVICES=4,5,6,7 python experiments/train_tensor_vsr.py \
        --dataset cifar100 \
        --gpu_ids 0,1,2,3
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
from src.rl.tensor_environment import TensorVSREnvironment
from src.rl.ppo_trainer import PPOTrainer


def main():
    parser = argparse.ArgumentParser(description='Train Tensor VSR on CIFAR')
    parser.add_argument('--config', type=str, default='configs/tensor_vsr_config.yaml',
                       help='Path to config file')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cifar10', 'cifar100'],
                       help='Dataset to use')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3',
                       help='GPU IDs to use, e.g., 0,1,2,3')
    parser.add_argument('--output_dir', type=str, default='outputs/tensor_vsr',
                       help='Output directory for checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override with dataset-specific config
    if args.dataset in config:
        print(f"Applying {args.dataset} specific config overrides...")
        config['training'].update(config[args.dataset])

    # Set GPUs
    if args.device == 'cuda':
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
        torch.cuda.set_device(gpu_ids[0])
        print(f"Using GPUs: {gpu_ids}")
        device = f"cuda:{gpu_ids[0]}"
    else:
        device = "cpu"
        print("Using CPU")

    # Print config
    print("\n" + "=" * 60)
    print("TENSOR VSR TRAINING CONFIG")
    print("=" * 60)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Device: {device}")
    print(f"Feature Bank Size: {config['training']['feature_bank_size']}")
    print(f"Max Depth: {config['model']['max_depth']}")
    print(f"Iterations: {config['training']['n_iterations']}")
    print(f"L1 Lambda: {config['training']['l1_lambda']}")
    print("=" * 60 + "\n")

    # Load data
    print("Loading data...")
    data_module = MNISTDataModule(
        dataset=args.dataset,
        batch_size=config['training']['batch_size'],
        num_workers=4,
        train_subset=config.get('dataset', {}).get('train_subset', None),
        test_subset=config.get('dataset', {}).get('test_subset', None)
    )
    data_module.setup()

    # Get train loader (used for validation in RL)
    train_loader = data_module.get_train_loader()
    print(f"Loaded {args.dataset.upper()} with {len(train_loader.dataset)} training samples")

    # Create environment
    print("\nInitializing Tensor VSR Environment...")
    env = TensorVSREnvironment(
        data_loader=train_loader,
        config=config,
        device=device
    )
    print(f"Vocabulary size: {len(env.vocabulary)}")
    print(f"Action space: {env.action_space}")

    # Create policy agent
    print("\nInitializing Policy Agent...")
    policy = PolicyAgent(
        vocab_size=len(env.vocabulary),
        embedding_dim=config['model']['embedding_dim'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy parameters: {total_params:,}")

    # Create PPO trainer
    print("\nInitializing PPO Trainer...")
    trainer = PPOTrainer(
        policy=policy,
        env=env,
        learning_rate=config['training']['learning_rate'],
        gamma=config['training']['gamma'],
        gae_lambda=config['training']['gae_lambda'],
        clip_epsilon=config['training']['clip_epsilon'],
        value_coef=config['training']['value_coef'],
        entropy_coef=config['training']['entropy_coef'],
        max_grad_norm=config['training']['max_grad_norm'],
        n_epochs=config['training']['n_epochs_ppo'],
        batch_size=config['training']['batch_size_ppo'],
        device=device
    )

    # Output directory
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving outputs to: {output_dir}")

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING TENSOR VSR")
    print("=" * 60 + "\n")

    n_iterations = config['training']['n_iterations']
    episodes_per_iteration = config['training']['episodes_per_iteration']
    best_accuracy = 0.0

    # Metrics tracking
    all_metrics = {
        'iteration': [],
        'avg_reward': [],
        'avg_accuracy': [],
        'bank_size': [],
        'active_features': []
    }

    start_time = time.time()

    for iteration in range(n_iterations):
        iter_start = time.time()

        # Collect episodes and update policy
        metrics = trainer.update(n_episodes=episodes_per_iteration)

        # Track metrics
        all_metrics['iteration'].append(iteration)
        all_metrics['avg_reward'].append(metrics['avg_reward'])
        all_metrics['bank_size'].append(len(env.feature_bank))

        # Get accuracy from recent episodes (if available)
        avg_accuracy = 0.0
        if hasattr(env, 'recent_accuracy'):
            avg_accuracy = env.recent_accuracy
        all_metrics['avg_accuracy'].append(avg_accuracy)
        all_metrics['active_features'].append(0)  # Will be updated below

        iter_time = time.time() - iter_start

        # Log progress
        if (iteration + 1) % 10 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (iteration + 1)) * (n_iterations - iteration - 1)

            print(f"\n[Iter {iteration+1}/{n_iterations}] "
                  f"Time: {iter_time:.1f}s | ETA: {eta/60:.1f}min")
            print(f"  Reward: {metrics['avg_reward']:.4f}")
            print(f"  Policy Loss: {metrics.get('policy_loss', 0):.4f}")
            print(f"  Value Loss: {metrics.get('value_loss', 0):.4f}")
            print(f"  Entropy: {metrics.get('entropy', 0):.4f}")
            print(f"  Bank Size: {len(env.feature_bank)}/{config['training']['feature_bank_size']}")

        # Periodic checkpoint
        if (iteration + 1) % 100 == 0:
            checkpoint_path = output_dir / f"checkpoint_{iteration+1}.pth"
            torch.save({
                'iteration': iteration,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'config': config,
                'feature_bank': env.feature_bank,
                'metrics': all_metrics
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")

        # Save best model
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_path = output_dir / "best_model.pth"
            torch.save({
                'iteration': iteration,
                'policy_state_dict': policy.state_dict(),
                'config': config,
                'feature_bank': env.feature_bank,
                'best_accuracy': best_accuracy
            }, best_path)

    # Final evaluation
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Final bank size: {len(env.feature_bank)}/{config['training']['feature_bank_size']}")

    # Print final formulas
    if len(env.feature_bank) > 0:
        print("\n" + "=" * 60)
        print("FINAL FEATURE BANK FORMULAS")
        print("=" * 60)
        for i, formula_dict in enumerate(env.feature_bank):
            print(f"  [{i+1}] {formula_dict['str']} (len={formula_dict['length']})")

    # Save final metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # Save final model
    final_path = output_dir / "final_model.pth"
    torch.save({
        'iteration': n_iterations,
        'policy_state_dict': policy.state_dict(),
        'config': config,
        'feature_bank': env.feature_bank,
        'metrics': all_metrics
    }, final_path)
    print(f"Saved final model to {final_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
