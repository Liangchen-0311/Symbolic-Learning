"""
RL + LASSO Ensemble Method.

This method combines the best of both worlds:
1. RL discovers diverse, interpretable symbolic formulas
2. LASSO selects the best linear combination of these formulas

Workflow:
    Step 1: Train RL to generate diverse formulas (Feature Bank)
    Step 2: Extract top K formulas from Feature Bank
    Step 3: Evaluate each formula on training data
    Step 4: Use LASSO to select best linear combination
    Step 5: Final prediction = weighted ensemble

Expected advantage: Higher R² than Pure RL (ensemble power)

Usage:
    python rl_lasso_ensemble.py --dataset concrete
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import time
import json
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import r2_score, mean_squared_error

# Import from existing code
from src.data.tabular_loader import TabularDataModule
from src.models.identity_encoder import IdentityEncoder
from src.models.policy_agent import PolicyAgent
from src.symbolic.operators import TokenVocabulary
from src.rl.tabular_environment import TabularRegressionEnv
from src.rl.ppo_trainer import PPOTrainer


class RLLassoEnsemble:
    """
    Ensemble method: RL generates candidates, LASSO selects best combination.
    
    This addresses a key limitation of Pure RL: a single formula may not
    capture all patterns in the data. By combining multiple formulas, we
    can achieve higher R² while maintaining some interpretability.
    """
    
    def __init__(
        self,
        n_candidates: int = 20,
        lasso_alpha: float = None,
        min_formula_quality: float = 0.15,
        device: str = 'cpu'
    ):
        """
        Args:
            n_candidates: Number of RL formulas to generate
            lasso_alpha: LASSO regularization strength (None = use CV)
            min_formula_quality: Minimum R² for candidate formulas
            device: Device for computation
        """
        self.n_candidates = n_candidates
        self.lasso_alpha = lasso_alpha
        self.min_formula_quality = min_formula_quality
        self.device = device
        
        # Will be populated during training
        self.candidate_formulas = []
        self.candidate_programs = []
        self.lasso_model = None
        self.selected_indices = []
    
    def generate_candidates(
        self,
        data_module: TabularDataModule,
        rl_config: dict,
        n_iterations: int = 100,
        method: str = 'ppo'
    ):
        """
        Step 1: Use RL to generate diverse candidate formulas.

        This runs standard RL training and collects formulas from the
        Feature Bank. We aim for diversity rather than just high R².

        Args:
            data_module: Tabular data module
            rl_config: RL training configuration
            n_iterations: Number of RL training iterations
            method: Training method ('ppo' or 'dqn')
        """
        print("\n" + "="*60)
        print("STEP 1: RL CANDIDATE GENERATION")
        print("="*60)
        
        # Setup (same as Pure RL)
        feature_names = data_module.get_feature_names()
        n_features = data_module.n_features
        
        encoder = IdentityEncoder(feature_dim=n_features)
        encoder.freeze_weights()
        encoder.to(self.device)
        
        vocab = TokenVocabulary(latent_dim=n_features)
        vocab.feature_names_mapping = {f'z{i}': name for i, name in enumerate(feature_names)}
        
        policy = PolicyAgent(
            vocab_size=len(vocab),
            embedding_dim=rl_config['model']['embedding_dim'],
            hidden_size=rl_config['model']['hidden_size'],
            num_layers=rl_config['model']['num_layers'],
            dropout=rl_config['model']['dropout']
        ).to(self.device)
        
        train_loader = data_module.get_train_loader()
        
        env = TabularRegressionEnv(
            data_loader=train_loader,
            vocabulary=vocab,
            max_sequence_length=rl_config['training']['max_sequence_length'],
            device=self.device,
            length_penalty=float(rl_config['training']['length_penalty']),
            parsimony_weight=float(rl_config['training'].get('parsimony_weight', 0.3)),
            interpretability_weight=float(rl_config['training'].get('interpretability_weight', 0.1)),
            feature_bank_size=20  # Larger bank for ensemble
        )
        
        # Train RL
        print(f"Training RL ({method.upper()}) for {n_iterations} iterations...")
        episodes_per_iteration = int(rl_config['training']['episodes_per_iteration'])

        if method == 'ppo':
            trainer = PPOTrainer(
                policy=policy,
                env=env,
                learning_rate=float(rl_config['training']['learning_rate']),
                gamma=float(rl_config['training']['gamma']),
                gae_lambda=float(rl_config['training']['gae_lambda']),
                clip_epsilon=float(rl_config['training']['clip_epsilon']),
                value_coef=float(rl_config['training']['value_coef']),
                entropy_coef=float(rl_config['training']['entropy_coef']),
                max_grad_norm=float(rl_config['training']['max_grad_norm']),
                n_epochs=int(rl_config['training']['n_epochs_ppo']),
                batch_size=int(rl_config['training']['batch_size_ppo']),
                device=self.device
            )

            for iteration in range(n_iterations):
                print(f"\nIteration {iteration+1}/{n_iterations}")

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
                print(f"Bank Size: {env.feature_bank.size()}/20")

        elif method == 'dqn':
            from src.rl.dqn_trainer import DQNTrainer
            trainer = DQNTrainer(
                env=env,
                vocab_size=len(vocab),
                learning_rate=float(rl_config['training']['learning_rate']),
                device=self.device
            )
            trainer.train(n_iterations, episodes_per_iteration)
        
        # Extract candidates from Feature Bank
        print("\n" + "="*60)
        print("EXTRACTING CANDIDATES FROM FEATURE BANK")
        print("="*60)
        
        bank = env.feature_bank
        self.candidate_programs = bank.formulas[:self.n_candidates]
        self.candidate_formulas = bank.formula_strs[:self.n_candidates]
        self.candidate_accuracies = bank.accuracies[:self.n_candidates]
        
        print(f"Extracted {len(self.candidate_formulas)} candidate formulas:")
        for i, (formula, acc) in enumerate(zip(self.candidate_formulas, self.candidate_accuracies)):
            print(f"  [{i+1}] {formula} (R²={acc:.3f})")
        
        return env
    
    def fit_ensemble(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor = None,
        y_val: torch.Tensor = None
    ):
        """
        Step 2: Fit LASSO ensemble on candidate formula outputs.
        
        For each formula, we evaluate it on the training data to get
        predictions. Then LASSO selects the best linear combination.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional, for alpha selection)
            y_val: Validation targets (optional)
        """
        print("\n" + "="*60)
        print("STEP 2: LASSO ENSEMBLE FITTING")
        print("="*60)
        
        # Evaluate each candidate formula on training data
        print("Evaluating candidate formulas on training data...")
        formula_outputs = []
        
        for i, program in enumerate(self.candidate_programs):
            with torch.no_grad():
                output = program.execute(X_train)
                output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)
                output = output.view(-1).cpu().numpy()
            
            # Normalize output
            mean = output.mean()
            std = output.std()
            if std > 1e-5:
                output = (output - mean) / std
            
            formula_outputs.append(output)
            print(f"  Formula {i+1}: mean={mean:.3f}, std={std:.3f}")
        
        # Create feature matrix: [n_samples, n_candidates]
        X_ensemble = np.column_stack(formula_outputs)
        y_ensemble = y_train.cpu().numpy()
        
        print(f"\nEnsemble feature matrix: {X_ensemble.shape}")
        
        # Fit LASSO
        if self.lasso_alpha is None:
            # Use cross-validation to select alpha
            print("Using LassoCV to select alpha...")
            alphas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
            self.lasso_model = LassoCV(alphas=alphas, cv=5, max_iter=10000)
        else:
            print(f"Using fixed alpha={self.lasso_alpha}")
            self.lasso_model = Lasso(alpha=self.lasso_alpha, max_iter=10000)
        
        self.lasso_model.fit(X_ensemble, y_ensemble)
        
        # Extract selected formulas (non-zero coefficients)
        coef = self.lasso_model.coef_
        self.selected_indices = [i for i, c in enumerate(coef) if abs(c) > 1e-6]
        
        print(f"\nLASSO Results:")
        if hasattr(self.lasso_model, 'alpha_'):
            print(f"  Selected alpha: {self.lasso_model.alpha_:.4f}")
        print(f"  Selected formulas: {len(self.selected_indices)}/{len(coef)}")
        print("\nSelected Formulas:")
        for i in self.selected_indices:
            print(f"  [{i+1}] {self.candidate_formulas[i]}")
            print(f"      Weight: {coef[i]:.3f}")
            print(f"      Solo R²: {self.candidate_accuracies[i]:.3f}")
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        Make predictions using the ensemble.
        
        Args:
            X: Input features
        
        Returns:
            predictions: Ensemble predictions
        """
        if self.lasso_model is None:
            raise RuntimeError("Ensemble not fitted. Call fit_ensemble() first.")
        
        # Evaluate each candidate formula
        formula_outputs = []
        for program in self.candidate_programs:
            with torch.no_grad():
                output = program.execute(X)
                output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)
                output = output.view(-1).cpu().numpy()
            
            # Normalize (using same statistics as training)
            mean = output.mean()
            std = output.std()
            if std > 1e-5:
                output = (output - mean) / std
            
            formula_outputs.append(output)
        
        X_ensemble = np.column_stack(formula_outputs)
        return self.lasso_model.predict(X_ensemble)
    
    def get_summary(self) -> dict:
        """
        Get summary of the ensemble.
        
        Returns:
            summary: Dict with ensemble information
        """
        return {
            'n_candidates': len(self.candidate_formulas),
            'n_selected': len(self.selected_indices),
            'selected_formulas': [self.candidate_formulas[i] for i in self.selected_indices],
            'selected_weights': [float(self.lasso_model.coef_[i]) for i in self.selected_indices],
            'selected_r2': [float(self.candidate_accuracies[i]) for i in self.selected_indices]
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['concrete', 'airfoil', 'energy', 'power',
                                'particle', 'asteroid'])
    parser.add_argument('--config', type=str, default='configs/rl_config_tabular.yaml')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--n_iterations', type=int, default=500,
                       help='RL training iterations')
    parser.add_argument('--n_candidates', type=int, default=20,
                       help='Number of candidate formulas to generate')
    parser.add_argument('--method', choices=['ppo', 'dqn'], default='ppo',
                       help='RL training method: ppo or dqn')
    args = parser.parse_args()
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 
                            'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print("="*60)
    print("RL + LASSO ENSEMBLE METHOD")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"RL Iterations: {args.n_iterations}")
    print(f"Candidates: {args.n_candidates}")
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load data
    data_module = TabularDataModule(
        dataset_name=args.dataset,
        batch_size=config['training']['batch_size']
    )
    
    # Start timing
    start_time = time.time()
    
    # Create ensemble
    ensemble = RLLassoEnsemble(
        n_candidates=args.n_candidates,
        lasso_alpha=None,  # Use CV
        device=str(device)
    )
    
    # Generate candidates with RL
    ensemble.generate_candidates(
        data_module=data_module,
        rl_config=config,
        n_iterations=args.n_iterations,
        method=args.method
    )
    
    # Fit LASSO ensemble
    ensemble.fit_ensemble(
        X_train=data_module.train_X,
        y_train=data_module.train_y,
        X_val=data_module.val_X,
        y_val=data_module.val_y
    )
    
    training_time = time.time() - start_time
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    y_pred = ensemble.predict(data_module.test_X)
    y_true = data_module.test_y.numpy()
    
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nTest Results:")
    print(f"  MSE: {mse:.6f}")
    print(f"  R²:  {r2:.4f}")
    print(f"  Total Time: {training_time:.1f}s ({training_time/60:.1f}min)")
    
    # Save results
    method_key = f"rl_lasso_{args.method}"
    save_dir = Path(f"outputs/results/{method_key}/{args.dataset}")
    save_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'dataset': args.dataset,
        'method': f'RL+LASSO Ensemble ({args.method.upper()})',
        'mse': float(mse),
        'r2': float(r2),
        'training_time': float(training_time),
        'ensemble_summary': ensemble.get_summary()
    }
    
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {save_dir}/")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    summary = ensemble.get_summary()
    print(f"Dataset: {args.dataset}")
    print(f"Method: RL + LASSO Ensemble")
    print(f"Test R²: {r2:.4f}")
    print(f"Test MSE: {mse:.6f}")
    print(f"Training Time: {training_time/60:.1f} minutes")
    print(f"Ensemble: {summary['n_selected']} formulas selected from {summary['n_candidates']} candidates")
    print("\nSelected Formulas:")
    for i, (formula, weight, solo_r2) in enumerate(zip(
        summary['selected_formulas'],
        summary['selected_weights'],
        summary['selected_r2']
    )):
        print(f"  [{i+1}] {formula}")
        print(f"      Weight: {weight:.3f}, Solo R²: {solo_r2:.3f}")
    print("="*60)


if __name__ == '__main__':
    main()
