"""
Environment with Feature Bank for ensemble symbolic features.

Key Changes:
1. Added FeatureBank to collect diverse formulas
2. Two-phase training:
   - Phase 1: Collect formulas (solo reward)
   - Phase 2: Ensemble reward (weighted combination)
3. Diversity-based selection
"""

import gymnasium as gym
import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional
from src.symbolic.program import SymbolicProgram
from src.symbolic.evaluator import ProgramEvaluator
from src.symbolic.operators import TokenVocabulary
from src.symbolic.action_mask import ActionMasker
from src.symbolic.feature_bank import FeatureBank


class SymbolicExpressionEnv(gym.Env):
    """
    Environment with Feature Bank for ensemble learning.
    
    Training Flow:
    1. RL generates formula
    2. Evaluate solo accuracy (Linear(1, 10))
    3. Check diversity vs bank
    4. If diverse enough, add to bank
    5. If bank full (5 formulas), evaluate ensemble
    6. Reward = 0.3 * solo + 0.7 * ensemble_improvement
    """
    
    def __init__(
        self,
        encoder: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        vocabulary: TokenVocabulary,
        max_sequence_length: int = 20,
        device: str = "cuda",
        length_penalty: float = 0.001,
        classifier_train_steps: int = 20,
        feature_bank_size: int = 8,
        num_classes: int = 10
    ):
        super().__init__()
        
        self.encoder = encoder
        self.data_loader = data_loader
        self.vocabulary = vocabulary
        self.max_sequence_length = max_sequence_length
        self.device = device
        self.length_penalty = length_penalty
        self.classifier_train_steps = classifier_train_steps
        
        # Define spaces
        self.action_space = gym.spaces.Discrete(len(vocabulary))
        self.observation_space = gym.spaces.Box(
            low=0,
            high=len(vocabulary),
            shape=(max_sequence_length,),
            dtype=int
        )
        
        self.num_classes = num_classes

        # Evaluator (for solo formula evaluation)
        self.evaluator = ProgramEvaluator(
            num_classes=num_classes,
            latent_dim=vocabulary.latent_dim,
            device=device
        )

        # Dynamic tier sizing: 40% simple, 60% complex
        max_simple = max(3, int(feature_bank_size * 0.4))
        max_complex = feature_bank_size - max_simple

        # Adaptive thresholds based on number of classes
        if num_classes >= 100:
            min_accuracy = 0.02   # 2% (2x random baseline of 1%)
            min_diversity = 0.2
        elif num_classes >= 50:
            min_accuracy = 0.10
            min_diversity = 0.25
        else:
            min_accuracy = 0.15   # MNIST, Fashion-MNIST, CIFAR-10
            min_diversity = 0.3

        print(f"FeatureBank config: {num_classes} classes, min_accuracy={min_accuracy}, min_diversity={min_diversity}")

        # Feature Bank
        self.feature_bank = FeatureBank(
            max_size=feature_bank_size,
            max_simple=max_simple,
            max_complex=max_complex,
            min_accuracy=min_accuracy,
            min_diversity=min_diversity,
            num_classes=num_classes,
            device=device
        )
        
        # Action masker
        self.action_masker = ActionMasker(vocabulary)
        
        # Cache validation data
        self.cached_z = None
        self.cached_labels = None
        self.cached_batch_size = 1024
        self._cache_validation_set()
        
        # Tracking
        self.seen_programs = {}
        self.episode_count = 0
        self.last_ensemble_acc = 0.0
        
        # Episode state
        self.current_sequence = []
        self.step_count = 0
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Reset environment."""
        super().reset(seed=seed)
        
        self.current_sequence = [self.vocabulary.encode('START')]
        self.step_count = 0
        
        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, bool, Dict]:
        """Execute action."""
        self.current_sequence.append(action)
        self.step_count += 1
        
        token_str = self.vocabulary.decode(action)
        
        terminated = False
        truncated = False
        reward = 0.0
        info = {}
        
        if token_str == 'END':
            terminated = True
        elif self.step_count >= self.max_sequence_length:
            truncated = True
        
        if terminated or truncated:
            reward, info = self._calculate_reward()
            self.episode_count += 1
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Return padded sequence."""
        obs = np.array(self.current_sequence, dtype=int)
        if len(obs) < self.max_sequence_length:
            obs = np.pad(
                obs,
                (0, self.max_sequence_length - len(obs)),
                mode='constant',
                constant_values=self.vocabulary.encode('PAD')
            )
        else:
            obs = obs[:self.max_sequence_length]
        return obs

    def get_action_mask(self) -> torch.Tensor:
        """Get valid actions."""
        return self.action_masker.get_valid_actions(self.current_sequence)

    def _get_validation_batch(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached validation batch."""
        if self.cached_z is None or self.cached_labels is None:
            raise RuntimeError("cached_validation_set_missing")
        if batch_size is None:
            batch_size = self.cached_batch_size
        batch_size = min(batch_size, self.cached_z.size(0))
        indices = torch.randint(0, self.cached_z.size(0), (batch_size,), device=self.cached_z.device)
        return self.cached_z[indices], self.cached_labels[indices]

    def _cache_validation_set(self) -> None:
        """Cache all validation data."""
        cached_z = []
        cached_labels = []
        self.encoder.eval()
        with torch.no_grad():
            for images, labels in self.data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                z = self.encoder(images)
                cached_z.append(z.detach())
                cached_labels.append(labels)
        if not cached_z:
            raise RuntimeError("empty_validation_cache")
        self.cached_z = torch.cat(cached_z, dim=0)
        self.cached_labels = torch.cat(cached_labels, dim=0)
    
    def _calculate_reward(self) -> Tuple[float, Dict]:
        """
        Calculate reward with feature bank ensemble.
        
        Steps:
        1. Validate and execute program
        2. Evaluate solo accuracy (Linear(1, 10))
        3. Compute diversity vs bank
        4. Try to add to bank
        5. If bank full, evaluate ensemble
        6. Reward = solo_weight * solo_acc + ensemble_weight * improvement
        """
        expression_tokens = [
            t for t in self.current_sequence
            if self.vocabulary.decode(t) not in ["START", "END", "PAD"]
        ]
        
        try:
            # 1. Validate program
            if not expression_tokens:
                return -1.0, {"valid": False, "reason": "empty"}
            
            program = SymbolicProgram(expression_tokens, self.vocabulary, self.device)
            if not program.validate():
                return -1.0, {"valid": False, "reason": "invalid_syntax"}
            
            prog_str = program.to_string()
            
            # 2. Get validation data
            z, labels = self._get_validation_batch()
            
            # 3. Execute program and get output
            with torch.no_grad():
                prog_output = program.execute(z)
                prog_output = torch.nan_to_num(prog_output, nan=0.0, posinf=1e4, neginf=-1e4)
            
            # 4. Evaluate solo accuracy
            solo_accuracy, solo_metrics = self.evaluator.evaluate(
                program, z, labels,
                update_classifier=True,
                n_train_steps=self.classifier_train_steps
            )
            
            # 5. Compute diversity
            length = len(expression_tokens)
            decoded_tokens = [self.vocabulary.decode(t) for t in expression_tokens]
            var_tokens = [tok for tok in decoded_tokens if tok.startswith("z")]
            variables_set = set(var_tokens)
            
            diversity = self.feature_bank.compute_diversity(
                prog_output,
                prog_str,
                variables_set
            )
            
            # 6. Try to add to feature bank
            bank_msg = "Not added"
            ensemble_accuracy = 0.0
            self.feature_bank.tick()

            if self.feature_bank.should_accept(solo_accuracy, diversity, prog_str, length):
                added, bank_msg = self.feature_bank.add_formula(
                    program,
                    solo_accuracy,
                    prog_output,
                    prog_str,
                    length
                )

                # Print bank status every time bank changes
                if added:
                    print("\n" + self.feature_bank.get_summary() + "\n")
            
            # 7. Evaluate ensemble if bank has formulas
            if self.feature_bank.size() > 0:
                ensemble_accuracy, ensemble_metrics = self.feature_bank.evaluate_ensemble(
                    z, labels, n_train_steps=self.classifier_train_steps
                )
                ensemble_improvement = ensemble_accuracy - self.last_ensemble_acc
                self.last_ensemble_acc = ensemble_accuracy
            else:
                ensemble_improvement = 0.0
            
            # 8. Compute final reward
            # Phase 1 (bank filling): 100% solo
            # Phase 2 (bank full): 30% solo + 70% ensemble improvement
            if self.feature_bank.is_full():
                # Ensemble phase
                solo_weight = 0.3
                ensemble_weight = 0.7
                base_reward = (
                    solo_weight * solo_accuracy +
                    ensemble_weight * max(0, ensemble_improvement)
                )
            else:
                # Collection phase
                base_reward = solo_accuracy
            
            # 9. Apply penalties and bonuses
            unique_vars = len(variables_set)
            duplicate_penalty = 0.03 * max(0, len(var_tokens) - unique_vars)

            # Novelty bonus
            novelty_bonus = 0.0
            if prog_str not in self.seen_programs:
                novelty_bonus = 0.02
                self.seen_programs[prog_str] = solo_accuracy

            # Complexity bonus: reward formulas with multiple operators
            operators = {'sin', 'cos', 'exp', 'log', 'relu', 'square', 'sqrt'}
            complexity_bonus = 0.0
            if length > 5:
                num_operators = len([t for t in decoded_tokens if t in operators])
                if num_operators >= 2:
                    complexity_bonus = 0.05

            # Final reward
            reward = base_reward + novelty_bonus + complexity_bonus - duplicate_penalty - self.length_penalty * length
            reward = max(-1.0, min(1.0, reward))
            
            # 10. Logging
            if self.feature_bank.is_full():
                print(
                    f"[Reward] Solo={solo_accuracy:.3f}, Ensemble={ensemble_accuracy:.3f}, "
                    f"Div={diversity:.2f}, Len={length}, Total={reward:.3f} | {prog_str}"
                )
                print(f"  Bank: {bank_msg}")
            else:
                print(
                    f"[Reward] Acc={solo_accuracy:.3f}, Div={diversity:.2f}, Len={length}, "
                    f"Total={reward:.3f} | {prog_str}"
                )
                print(f"  Bank: {bank_msg} ({self.feature_bank.size()}/{self.feature_bank.max_size})")
            
            # 11. Return
            metrics = {
                "valid": True,
                "program_str": prog_str,
                "solo_accuracy": solo_accuracy,
                "ensemble_accuracy": ensemble_accuracy,
                "diversity": diversity,
                "bank_size": self.feature_bank.size(),
                "total_reward": reward
            }
            
            return reward, metrics
            
        except Exception as e:
            import traceback
            print(f"[Reward Error] {e}")
            traceback.print_exc()
            return -1.0, {"valid": False, "error": str(e)}
