"""
Pure RL training for tabular symbolic regression.

This trains the RL policy to discover symbolic formulas for
tabular regression datasets (Concrete, Energy, Power, Airfoil).

Usage:
    python experiments/train_tabular.py --dataset concrete --device cpu
    python experiments/train_tabular.py --dataset energy --n_iterations 50
"""

import argparse
import yaml
import torch
import time
import json
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

from src.data.tabular_loader import TabularDataModule
from src.models.policy_agent import PolicyAgent
from src.symbolic.operators import TokenVocabulary
from src.rl.tabular_environment import TabularRegressionEnv
from src.rl.ppo_trainer import PPOTrainer
from src.utils.evaluation_metrics import compute_all_metrics


def main():
    parser = argparse.ArgumentParser(description="Pure RL Symbolic Regression")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['concrete', 'airfoil', 'energy', 'power',
                                 'particle', 'asteroid'])
    parser.add_argument('--config', type=str,
                        default='configs/rl_config_tabular.yaml')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--n_iterations', type=int, default=500,
                        help='Override n_iterations from config')
    parser.add_argument('--method', choices=['ppo', 'dqn'], default='ppo',
                        help='Training method: ppo or dqn')
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("PURE RL SYMBOLIC REGRESSION")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    n_iterations = args.n_iterations or config['training']['n_iterations']
    print(f"Iterations: {n_iterations}")

    # Load data
    data_module = TabularDataModule(
        dataset_name=args.dataset,
        batch_size=config['training']['batch_size']
    )

    feature_names = data_module.get_feature_names()
    n_features = data_module.n_features

    # Override latent_dim to match actual features
    config['model']['latent_dim'] = n_features

    # Setup components
    vocab = TokenVocabulary(latent_dim=n_features)
    vocab.feature_names_mapping = {
        f'z{i}': name for i, name in enumerate(feature_names)
    }

    policy = PolicyAgent(
        vocab_size=len(vocab),
        embedding_dim=config['model']['embedding_dim'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)

    train_loader = data_module.get_train_loader()

    env = TabularRegressionEnv(
        data_loader=train_loader,
        vocabulary=vocab,
        max_sequence_length=config['training']['max_sequence_length'],
        device=str(device),
        length_penalty=float(config['training']['length_penalty']),
        parsimony_weight=float(config['training'].get('parsimony_weight', 0.3)),
        interpretability_weight=float(config['training'].get('interpretability_weight', 0.1)),
        feature_bank_size=int(config['training'].get('feature_bank_size', 15))
    )

    # Train
    start_time = time.time()
    episodes_per_iteration = int(config['training']['episodes_per_iteration'])
    feature_bank_size = int(config['training'].get('feature_bank_size', 15))

    if args.method == 'ppo':
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

        for iteration in range(n_iterations):
            print(f"\n=== Iteration {iteration+1}/{n_iterations} ===")

            # Curriculum learning schedule
            if iteration < 100:
                env.max_sequence_length = 8
                env.feature_bank.min_accuracy = 0.05
            elif iteration < 300:
                env.max_sequence_length = 12
                env.feature_bank.min_accuracy = 0.15
            else:
                env.max_sequence_length = 15
                env.feature_bank.min_accuracy = 0.25

            metrics = trainer.update(n_episodes=episodes_per_iteration)

            print(f"Avg Reward: {metrics['avg_reward']:.4f}")
            print(f"Best Program: {trainer.best_program}")
            print(f"Bank Size: {env.feature_bank.size()}/{feature_bank_size}")

    elif args.method == 'dqn':
        from src.rl.dqn_trainer import DQNTrainer
        trainer = DQNTrainer(
            env=env,
            vocab_size=len(vocab),
            learning_rate=float(config['training']['learning_rate']),
            device=str(device)
        )
        trainer.train(n_iterations, episodes_per_iteration)

    training_time = time.time() - start_time

    # Final test evaluation using environment's held-out test set
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    bank = env.feature_bank
    best_program_obj = bank.formulas[0] if bank.size() > 0 else None
    if best_program_obj and bank.size() > 0:
        best_bank_idx = max(range(bank.size()), key=lambda i: bank.accuracies[i])
        best_program_obj = bank.formulas[best_bank_idx]
        test_results = env.evaluate_on_test(best_program_obj)
        print(f"Test MSE:  {test_results['test_mse']:.6f}")
        print(f"Test R2:   {test_results['test_r2']:.4f}")
        print(f"Formula:   {bank.formula_strs[best_bank_idx]}")
    else:
        test_results = {'test_mse': float('inf'), 'test_r2': 0.0}
        print("No valid formulas found")
    print("=" * 60)

    # Evaluate on data_module test set
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    if bank.size() > 0:
        # Find best formula by R^2
        best_idx = max(range(bank.size()), key=lambda i: bank.accuracies[i])
        best_program = bank.formulas[best_idx]
        best_formula = bank.formula_strs[best_idx]

        with torch.no_grad():
            test_output = best_program.execute(data_module.test_X.to(device))
            test_output = torch.nan_to_num(test_output, nan=0.0,
                                           posinf=1e4, neginf=-1e4)

        out_np = test_output.view(-1, 1).cpu().numpy()
        y_np = data_module.test_y.numpy()

        if np.std(out_np) > 1e-8:
            reg = LinearRegression()
            reg.fit(out_np, y_np)
            pred = reg.predict(out_np)
            mse = mean_squared_error(y_np, pred)
            r2 = r2_score(y_np, pred)
        else:
            mse = float('inf')
            r2 = 0.0
    else:
        best_formula = "None"
        mse = float('inf')
        r2 = 0.0

    # Also compute ensemble R^2 (all formulas combined via LinearRegression)
    ensemble_r2 = 0.0
    if bank.size() > 1:
        all_outputs = []
        for f in bank.formulas:
            with torch.no_grad():
                out = f.execute(data_module.test_X.to(device))
                out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
                out = out.view(-1, 1).cpu().numpy()
                std = np.std(out)
                if std > 1e-8:
                    out = (out - np.mean(out)) / std
                all_outputs.append(out)
        ens_X = np.column_stack(all_outputs)
        ens_reg = LinearRegression()
        ens_reg.fit(ens_X, y_np)
        ens_pred = ens_reg.predict(ens_X)
        ensemble_r2 = r2_score(y_np, ens_pred)
        ensemble_r2 = max(0.0, ensemble_r2)

    print(f"\nTest Results:")
    print(f"  Best Formula: {best_formula}")
    print(f"  Solo R^2: {r2:.4f}")
    print(f"  Ensemble R^2: {ensemble_r2:.4f}  ({bank.size()} formulas)")
    print(f"  MSE: {mse:.6f}")
    print(f"  Training Time: {training_time/60:.1f} min")

    # Save results
    method_key = f"pure_rl_{args.method}"
    save_dir = Path(f"outputs/results/{method_key}/{args.dataset}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save feature bank summary
    with open(save_dir / "feature_bank_final.txt", 'w') as f:
        f.write(bank.get_summary())
        f.write(f"\n\nTest R^2: {r2:.4f}")
        f.write(f"\nTest MSE: {mse:.6f}")
        f.write(f"\nTraining Time: {training_time/60:.1f} min")

    # Compute comprehensive metrics
    n_terms = len(best_formula.split()) if best_formula != "None" else 0
    full_metrics = compute_all_metrics(
        y_true=y_np,
        y_pred=pred if bank.size() > 0 and np.std(out_np) > 1e-8 else np.zeros_like(y_np),
        model_name="VSR+RL",
        n_params=bank.size(),
        n_terms=n_terms
    )

    results = {
        'dataset': args.dataset,
        'method': f'Pure RL ({args.method.upper()})',
        'mse': float(mse),
        'r2': float(r2),
        'ensemble_r2': float(ensemble_r2),
        'test_mse': float(test_results['test_mse']),
        'test_r2': float(test_results['test_r2']),
        'training_time': float(training_time),
        'best_formula': best_formula,
        'bank_size': bank.size(),
        'all_formulas': bank.formula_strs,
        'all_r2': [float(a) for a in bank.accuracies],
        'n_params': bank.size(),
        'n_terms': n_terms,
        'param_efficiency': full_metrics['param_efficiency'],
        'interpretability': full_metrics['interpretability'],
        'composite_score': full_metrics['composite_score'],
    }

    with open(save_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {save_dir}/")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Method: Pure RL ({args.method.upper()})")
    print(f"Test R^2: {r2:.4f}")
    print(f"Test MSE: {mse:.6f}")
    print(f"Training Time: {training_time/60:.1f} min")
    print(f"Bank: {bank.size()} formulas")
    print(bank.get_summary())
    print("=" * 60)


if __name__ == '__main__':
    main()
