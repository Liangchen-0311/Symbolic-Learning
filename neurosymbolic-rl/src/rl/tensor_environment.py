"""
Tensor-based VSR Environment for CIFAR-10/100.

Key Differences from standard environment:
1. NO CNN encoder - works directly on raw images
2. Terminals are RGB channels (I_R, I_G, I_B)
3. Uses tensor operators (spatial filters, pooling)
4. LASSO classifier for feature selection
5. Action masking for root pooling operators
"""

import gymnasium as gym
import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional
from src.symbolic.program import SymbolicProgram
from src.symbolic.tensor_operators import TENSOR_OPERATORS, ROOT_OPERATORS
from src.models.lasso_classifier import train_lasso_classifier
from src.rl.tensor_action_masking import TensorActionMasker


class TensorTokenVocabulary:
    """
    Vocabulary for tensor-based VSR.

    Tokens:
    - Special: START, END, PAD
    - Operators: add, multiply, relu, blur, edge_x, global_avg_pool, etc.
    - Terminals: I_R, I_G, I_B (RGB channels)
    """

    def __init__(self):
        # Special tokens
        special_tokens = ['START', 'END', 'PAD']

        # Operators from tensor_operators
        operator_tokens = list(TENSOR_OPERATORS.keys())

        # Terminals (RGB channels + grayscale)
        terminal_tokens = ['I_R', 'I_G', 'I_B', 'I_GRAY']

        # Build vocabulary
        self.tokens = special_tokens + operator_tokens + terminal_tokens
        self.token_to_idx = {t: i for i, t in enumerate(self.tokens)}
        self.idx_to_token = {i: t for i, t in enumerate(self.tokens)}

    def encode(self, token: str) -> int:
        """Encode token to index."""
        return self.token_to_idx[token]

    def decode(self, idx: int) -> str:
        """Decode index to token."""
        return self.idx_to_token[idx]

    def __len__(self):
        return len(self.tokens)


class TensorVSREnvironment(gym.Env):
    """
    Environment for tensor-based VSR on images.

    Training Flow:
    1. RL generates formula (e.g., global_avg_pool(blur(add(I_R, I_G))))
    2. Evaluate on batch of images
    3. Extract features from all formulas in bank
    4. Train LASSO classifier
    5. Reward based on accuracy, active features, length
    """

    def __init__(
        self,
        data_loader: torch.utils.data.DataLoader,
        config: Dict[str, Any],
        device: str = "cuda"
    ):
        super().__init__()

        self.data_loader = data_loader
        self.config = config
        self.device = device

        # Extract config
        model_cfg = config['model']
        train_cfg = config['training']

        self.max_depth = model_cfg.get('max_depth', 5)
        self.max_sequence_length = model_cfg.get('max_sequence_length', 20)
        self.num_classes = train_cfg.get('num_classes', 10)
        self.feature_bank_size = train_cfg.get('feature_bank_size', 20)
        self.l1_lambda = train_cfg.get('l1_lambda', 0.01)
        self.lasso_epochs = train_cfg.get('lasso_epochs', 50)
        self.length_penalty = train_cfg.get('length_penalty', 0.01)
        self.inactive_penalty = train_cfg.get('inactive_feature_penalty', 0.05)

        # Vocabulary
        self.vocabulary = TensorTokenVocabulary()

        # Define spaces
        self.action_space = gym.spaces.Discrete(len(self.vocabulary))
        self.observation_space = gym.spaces.Box(
            low=0,
            high=len(self.vocabulary),
            shape=(self.max_sequence_length,),
            dtype=int
        )

        # Action masker
        self.action_masker = TensorActionMasker(
            self.vocabulary,
            max_depth=self.max_depth
        )

        # Feature bank (list of formulas)
        self.feature_bank = []

        # Seed with valid formulas to bootstrap learning
        self._seed_feature_bank()

        # Cache validation data (raw images)
        self.cached_images = None
        self.cached_labels = None
        self._cache_validation_set()

        # Data iterator for getting batches
        self.data_iter = iter(data_loader)
        self.batch_size = train_cfg.get('batch_size', 64)

        # Episode state
        self.current_sequence = []
        self.step_count = 0
        self.episode_count = 0

    def _seed_feature_bank(self):
        """Seed feature bank with diverse valid formulas including complex ones."""
        seed_formulas = [
            # Depth 2: Color combinations
            'I_R I_G add global_avg_pool',
            'I_R I_B multiply global_std_pool',
            'I_G I_B subtract global_max_pool',

            # Depth 3: Edge + Color
            'I_R edge_x I_G edge_y add global_avg_pool',
            'I_R edge_x blur global_max_pool',
            'I_B edge_y sharpen global_avg_pool',

            # Depth 3: Complex spatial
            'I_R blur edge_x global_avg_pool',
            'I_G sharpen I_B blur multiply global_std_pool',

            # Depth 4: Very complex (target)
            'I_R edge_x blur sharpen global_avg_pool',
            'I_G edge_y blur I_B edge_x add global_max_pool',
            'I_R blur I_G sharpen multiply edge_x global_std_pool',

            # Different pooling types
            'I_R I_G add global_std_pool',
            'I_B edge_x global_l2_pool',
            'I_R blur I_G sharpen subtract global_std_pool',

            # Maximum complexity depth 4
            'I_R edge_x blur I_G edge_y multiply global_avg_pool',
            'I_G I_B multiply blur edge_y sharpen global_max_pool',
            'I_R sharpen I_B edge_x multiply global_l2_pool',

            # Additional diverse formulas
            'I_G edge_x I_R blur add global_avg_pool',
            'I_B blur I_G edge_y subtract global_max_pool',
        ]

        # Only add seeds up to feature bank size
        max_seeds = min(len(seed_formulas), self.feature_bank_size)

        print(f"Seeding feature bank with {max_seeds} complex formulas:")
        for formula_str in seed_formulas[:max_seeds]:
            tokens = [self.vocabulary.encode(t) for t in formula_str.split()]

            # Calculate formula depth
            depth = self._calculate_formula_depth(tokens)

            self.feature_bank.append({
                'str': formula_str,
                'tokens': tokens,
                'length': len(tokens),
                'depth': depth
            })
            print(f"  [Seed] {formula_str} (depth={depth})")

    def _calculate_formula_depth(self, tokens):
        """Calculate the depth of a formula (max operator chain length)."""
        decoded = [self.vocabulary.decode(t) for t in tokens]
        stack = []
        max_depth = 0

        for token in decoded:
            if token in ['I_R', 'I_G', 'I_B', 'I_GRAY']:
                # Terminal: depth 0
                stack.append(0)
            elif token in TENSOR_OPERATORS:
                _, arity, _ = TENSOR_OPERATORS[token]
                if len(stack) >= arity:
                    operand_depths = [stack.pop() for _ in range(arity)]
                    new_depth = max(operand_depths) + 1
                    stack.append(new_depth)
                    max_depth = max(max_depth, new_depth)
                else:
                    # Invalid formula, return 0
                    return 0

        return max_depth

    def _cache_validation_set(self) -> None:
        """Cache all validation images (no encoding!)."""
        cached_images = []
        cached_labels = []

        for images, labels in self.data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            cached_images.append(images)
            cached_labels.append(labels)

        if not cached_images:
            raise RuntimeError("Empty validation cache")

        self.cached_images = torch.cat(cached_images, dim=0)
        self.cached_labels = torch.cat(cached_labels, dim=0)

        print(f"Cached {self.cached_images.shape[0]} images: {self.cached_images.shape}")

    def get_data_batch(self, batch_size: Optional[int] = None) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get a batch of CIFAR images as RGB channels.

        Returns:
            terminal_values: Dict with I_R, I_G, I_B [batch, H, W]
            labels: [batch]
        """
        if batch_size is None:
            batch_size = self.batch_size

        # Sample from cached data
        batch_size = min(batch_size, self.cached_images.size(0))
        indices = torch.randint(0, self.cached_images.size(0), (batch_size,), device=self.device)
        images = self.cached_images[indices]
        labels = self.cached_labels[indices]

        # Extract RGB channels + grayscale
        # images shape: [batch, 3, 32, 32]
        I_R = images[:, 0, :, :]  # Red channel [batch, 32, 32]
        I_G = images[:, 1, :, :]  # Green channel
        I_B = images[:, 2, :, :]  # Blue channel
        terminal_values = {
            'I_R': I_R,
            'I_G': I_G,
            'I_B': I_B,
            'I_GRAY': 0.2989 * I_R + 0.5870 * I_G + 0.1140 * I_B,
        }

        return terminal_values, labels

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
        """Get valid actions based on depth and root constraints."""
        # TODO: Implement depth tracking for proper masking
        # For now, return all actions as valid
        return torch.ones(len(self.vocabulary))

    def _execute_formula(self, formula_tokens, data_batch):
        """
        Execute a tensor formula.

        Args:
            formula_tokens: List of token indices
            data_batch: Dict with I_R, I_G, I_B

        Returns:
            output: [batch] scalar features
        """
        # Decode tokens
        decoded = [self.vocabulary.decode(t) for t in formula_tokens]

        # Execute in reverse Polish notation
        stack = []

        for token in decoded:
            if token in data_batch:
                # Terminal: push image channel
                stack.append(data_batch[token])
            elif token in TENSOR_OPERATORS:
                # Operator: pop operands and apply
                op_func, arity, output_type = TENSOR_OPERATORS[token]

                if len(stack) < arity:
                    raise ValueError(f"Not enough operands for {token}")

                # Pop operands (in reverse order for correct application)
                operands = [stack.pop() for _ in range(arity)]
                operands.reverse()

                # Apply operator
                result = op_func(*operands)
                stack.append(result)

        if len(stack) != 1:
            raise ValueError(f"Invalid formula: stack has {len(stack)} elements")

        output = stack[0]

        # Valid output: [batch] for scalar ops, [batch, D] for multi-dim ops
        if output.dim() not in (1, 2):
            raise ValueError(f"Formula output must be [batch] or [batch, D], got {output.shape}")

        return output

    def _calculate_reward(self) -> Tuple[float, Dict]:
        """
        Calculate reward using LASSO classifier.

        Steps:
        1. Validate formula
        2. Add to feature bank if diverse
        3. Extract features from all formulas
        4. Train LASSO classifier
        5. Compute reward based on accuracy, active features, length
        """
        expression_tokens = [
            t for t in self.current_sequence
            if self.vocabulary.decode(t) not in ["START", "END", "PAD"]
        ]

        try:
            # 1. Validate formula
            if not expression_tokens:
                print(f"  [Reward] Empty formula")
                return -1.0, {"valid": False, "reason": "empty"}

            formula_str = ' '.join([self.vocabulary.decode(t) for t in expression_tokens])

            # Get validation batch
            data_batch, labels = self.get_data_batch()

            # 2. Execute formula
            try:
                formula_output = self._execute_formula(expression_tokens, data_batch)
                formula_output = torch.nan_to_num(formula_output, nan=0.0, posinf=1e4, neginf=-1e4)
            except Exception as e:
                if self.episode_count % 10 == 0:  # Print every 10 episodes
                    print(f"  [Reward] Formula '{formula_str}' failed: {e}")

                # Reward shaping: give significant credit for formula structure
                partial_reward = -1.0

                # Strong rewards for good structure
                has_pooling = any(self.vocabulary.decode(t) in ['global_avg_pool', 'global_max_pool', 'global_std_pool', 'global_l2_pool'] for t in expression_tokens)
                has_terminal = any(self.vocabulary.decode(t).startswith('I_') for t in expression_tokens)

                # Both required for valid formula
                if has_pooling and has_terminal:
                    partial_reward += 0.4  # Big bonus for having both
                elif has_pooling:
                    partial_reward += 0.2
                elif has_terminal:
                    partial_reward += 0.2

                # Reward short formulas more (easier to get right)
                formula_length = len(expression_tokens)
                if formula_length == 2:  # Perfect: terminal + pooling
                    partial_reward += 0.3
                elif formula_length <= 4:
                    partial_reward += 0.2
                elif formula_length <= 6:
                    partial_reward += 0.1
                elif formula_length > 10:
                    partial_reward -= 0.1  # Penalty for overly complex

                # Extra reward if ends with pooling operator (good RPN sign)
                if expression_tokens and self.vocabulary.decode(expression_tokens[-1]) in ['global_avg_pool', 'global_max_pool', 'global_std_pool', 'global_l2_pool']:
                    partial_reward += 0.2

                return partial_reward, {"valid": False, "reason": f"execution_error: {e}"}

            # 3. Add to feature bank (if not duplicate)
            is_new = formula_str not in [f['str'] for f in self.feature_bank]
            if is_new and len(self.feature_bank) < self.feature_bank_size:
                formula_depth = self._calculate_formula_depth(expression_tokens)
                self.feature_bank.append({
                    'str': formula_str,
                    'tokens': expression_tokens,
                    'length': len(expression_tokens),
                    'depth': formula_depth
                })
                print(f"  [Bank] Added formula {len(self.feature_bank)}/{self.feature_bank_size}: {formula_str} (depth={formula_depth})")

            # 4. Extract features from all formulas in bank
            if len(self.feature_bank) == 0:
                # No features yet, return small reward
                reward = 0.1 if is_new else -0.5
                return reward, {
                    "valid": True,
                    "formula": formula_str,
                    "accuracy": 0.0,
                    "bank_size": 0
                }

            # Extract features from all formulas
            features_list = []
            for formula_dict in self.feature_bank:
                try:
                    feature = self._execute_formula(formula_dict['tokens'], data_batch)
                    feature = torch.nan_to_num(feature, nan=0.0, posinf=1e4, neginf=-1e4)
                    features_list.append(feature)
                except Exception as e:
                    print(f"  [Warning] Failed to execute formula: {e}")
                    continue

            if len(features_list) == 0:
                return -1.0, {"valid": False, "reason": "no_valid_features"}

            # Ensure all features are 2D [batch, D] then concatenate
            features_2d = []
            for f in features_list:
                if f.dim() == 1:
                    features_2d.append(f.unsqueeze(1))
                else:
                    features_2d.append(f)
            features_tensor = torch.cat(features_2d, dim=1)

            # 5. Train LASSO classifier
            accuracy, active_features, model = train_lasso_classifier(
                features_tensor,
                labels,
                num_classes=self.num_classes,
                l1_lambda=self.l1_lambda,
                epochs=self.lasso_epochs,
                device=self.device
            )

            # 6. Compute reward with complexity bonus
            total_features = len(self.feature_bank)
            inactive_features = total_features - active_features

            # Calculate formula depth
            formula_depth = self._calculate_formula_depth(expression_tokens)

            # Complexity bonus (reward deeper formulas)
            complexity_bonus_val = 0.0
            if formula_depth >= 3:
                complexity_bonus_coef = self.train_cfg.get('complexity_bonus', 0.0)
                complexity_bonus_val = complexity_bonus_coef * (formula_depth - 2)
                # depth=3 → bonus=0.1 (if coef=0.1)
                # depth=4 → bonus=0.2

            reward = (
                accuracy
                - self.length_penalty * len(expression_tokens)
                - self.inactive_penalty * inactive_features
                + complexity_bonus_val
            )

            # Bonus for new formulas
            if is_new:
                reward += 0.05

            reward = max(-1.0, min(1.0, reward))

            # 7. Logging
            print(
                f"[Reward] Acc={accuracy:.3f}, Active={active_features}/{total_features}, "
                f"Len={len(expression_tokens)}, Depth={formula_depth}, Reward={reward:.3f}"
            )
            print(f"  Formula: {formula_str}")

            # 8. Return
            metrics = {
                "valid": True,
                "formula": formula_str,
                "accuracy": accuracy,
                "active_features": active_features,
                "total_features": total_features,
                "bank_size": len(self.feature_bank),
                "reward": reward
            }

            return reward, metrics

        except Exception as e:
            import traceback
            print(f"[Reward Error] {e}")
            traceback.print_exc()
            return -1.0, {"valid": False, "error": str(e)}
