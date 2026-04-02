"""
Main RL training script for symbolic expression search.

This script:
1. Loads pre-trained frozen encoder
2. Initializes policy agent (LSTM/Transformer)
3. Creates symbolic expression environment
4. Runs PPO training loop
5. Logs metrics and saves checkpoints

Usage:
    python experiments/train_symbolic.py \\
        --encoder_path outputs/checkpoints/encoder.pth \\
        --config configs/rl_config.yaml
"""

import argparse
import yaml
import torch
from pathlib import Path

from src.data.mnist_loader import MNISTDataModule
from src.models.encoder import MNISTEncoder
from src.models.policy_agent import PolicyAgent
from src.symbolic.operators import TokenVocabulary
from src.rl.environment import SymbolicExpressionEnv
from src.rl.ppo_trainer import PPOTrainer


def train_symbolic_solver(
    encoder_path: str,
    config: dict,
    device_override: str = None,
    dataset: str = "mnist",
    feature_bank_size: int = None,
    n_iterations_override: int = None
) -> None:
    """
    Train RL-based symbolic solver.

    Args:
        encoder_path: Path to pre-trained encoder
        config: RL training configuration
        device_override: Force a specific device (e.g. "cpu")
        dataset: 'mnist' or 'fashion_mnist'
        feature_bank_size: Feature bank size (None = use config or default 8)
    """
    if device_override:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data Loader (load first to get dataset metadata)
    print(f"Loading data: {dataset.upper()}...")
    data_module = MNISTDataModule(batch_size=config['training']['batch_size'], dataset=dataset)
    data_module.setup()
    train_loader = data_module.get_train_loader()

    num_classes = data_module.num_classes
    input_channels = data_module.input_channels
    image_size = data_module.image_size

    # 2. Load frozen encoder
    print(f"Loading encoder from {encoder_path}...")
    encoder = MNISTEncoder(
        latent_dim=config['model']['latent_dim'],
        input_channels=input_channels,
        image_size=image_size,
        hidden_dim=int(config['model'].get('hidden_dim', 128)),
        deep=bool(config['model'].get('deep', False))
    )

    try:
        state_dict = torch.load(encoder_path, map_location=device)
        encoder.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading encoder: {e}")
        print("Proceeding with randomly initialized encoder (NOT RECOMMENDED)")

    encoder.freeze_weights()
    encoder.eval()
    encoder.to(device)

    # 3. Vocabulary
    vocab = TokenVocabulary(latent_dim=config['model']['latent_dim'])

    # 4. Policy Agent
    print("Initializing policy...")
    policy = PolicyAgent(
        vocab_size=len(vocab),
        embedding_dim=config['model']['embedding_dim'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)

    # 5. Environment
    print("Creating environment...")
    if feature_bank_size is None:
        feature_bank_size = int(config['training'].get('feature_bank_size', 8))
    print(f"Feature bank size: {feature_bank_size}")

    env = SymbolicExpressionEnv(
        encoder=encoder,
        data_loader=train_loader,
        vocabulary=vocab,
        max_sequence_length=config['training']['max_sequence_length'],
        device=str(device),
        length_penalty=float(config['training'].get('length_penalty', 0.01)),
        classifier_train_steps=int(config['training'].get('classifier_train_steps', 20)),
        feature_bank_size=feature_bank_size,
        num_classes=num_classes
    )

    # 6. PPO Trainer
    print("Initializing trainer...")
    trainer = PPOTrainer(
        policy=policy,
        env=env,
        learning_rate=float(config['training']['learning_rate']),
        gamma=float(config['training']['gamma']),
        gae_lambda=float(config['training']['gae_lambda']),
        clip_epsilon=float(config['training']['clip_epsilon']),
        value_coef=float(config['training']['value_coef']),
        entropy_coef=float(config['training']['entropy_coef']),
        max_grad_norm=float(config['training']['max_grad_norm']),
        n_epochs=int(config['training']['n_epochs_ppo']),
        batch_size=int(config['training']['batch_size_ppo']),
        device=str(device)
    )

    # 7. Train
    print("Starting training...")
    save_dir = config['training']['save_dir']
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    import time
    start_time = time.time()

    n_iterations = n_iterations_override or int(config['training']['n_iterations'])
    trainer.train(
        n_iterations=n_iterations,
        episodes_per_iteration=int(config['training']['episodes_per_iteration']),
        save_dir=save_dir
    )

    training_time = time.time() - start_time

    # 8. Save results
    import json
    bank = env.feature_bank
    ensemble_acc = 0.0
    if bank.size() > 0:
        z, labels = env._get_validation_batch(batch_size=1024)
        ensemble_acc, _ = bank.evaluate_ensemble(z, labels, n_train_steps=50)

    results_dir = Path(f"outputs/results/pure_rl_mnist/{dataset}")
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'dataset': dataset,
        'method': 'VSR+RL (Pure RL)',
        'test_accuracy': float(ensemble_acc),
        'best_reward': float(trainer.best_reward),
        'best_program': trainer.best_program,
        'training_time': float(training_time),
        'feature_bank_size': feature_bank_size,
        'bank_formulas': bank.formula_strs,
        'bank_accuracies': [float(a) for a in bank.accuracies],
        'n_iterations': n_iterations
    }

    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open(results_dir / 'feature_bank_final.txt', 'w') as f:
        f.write(bank.get_summary())

    print(f"\nResults saved to {results_dir}/")
    print(f"Ensemble accuracy: {ensemble_acc:.4f}")
    print(f"Training time: {training_time/60:.1f} min")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder_path",
        type=str,
        required=True,
        help="Path to pre-trained encoder"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rl_config.yaml",
        help="Path to RL config file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device (e.g. 'cpu', 'cuda', 'mps')"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist", "cifar10", "cifar100"],
        help="Dataset: 'mnist', 'fashion_mnist', 'cifar10', or 'cifar100'"
    )
    parser.add_argument(
        "--feature_bank_size",
        type=int,
        default=None,
        help="Feature bank size (overrides config value)"
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=None,
        help="Override n_iterations from config"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_symbolic_solver(
        args.encoder_path, config,
        device_override=args.device,
        dataset=args.dataset,
        feature_bank_size=args.feature_bank_size,
        n_iterations_override=args.n_iterations
    )


if __name__ == "__main__":
    main()

"""
## Key Implementation Notes

### 1. **Critical Architectural Differences**
- **Source Paper**: Uses Genetic Programming (GP) with crossover/mutation
- **This Project**: Uses Reinforcement Learning with LSTM/Transformer controller
- **Both**: Share the concept of Vectorized Symbolic Regression (VSR)

### 2. **Vectorized Execution (VSR)**
From the source paper, VSR means:
- All variables undergo the **same mathematical operations**
- Formulas are expressed as vector operations: `(X ⊙ X ⊙ X + X ⊙ X + X) · [1,1,1]ᵀ`
- This enables GPU acceleration and avoids combinatorial explosion

### 3. **RL Training Flow**

1. Pre-train Encoder (MNIST → z)
2. Freeze Encoder
3. Initialize Policy Agent (LSTM)
4. For each episode:
   a. Generate expression tokens sequentially
   b. Execute expression on z (vectorized)
   c. Compute accuracy as reward
   d. Update policy using PPO
"""
