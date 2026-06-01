"""
Enhanced Tensor VSR Environment with Large Feature Bank and Diversity Penalty.

Improvements over base environment:
1. Large feature bank (200-300 formulas)
2. Diversity-aware rewards (penalize correlated features)
3. Periodic LASSO pruning
4. Spatial pooling operators
"""

import math
import os
import gymnasium as gym
import torch
import torch.nn.functional as F_func
import numpy as np
from typing import Tuple, Dict, Any, Optional
from src.symbolic.program import SymbolicProgram
from src.symbolic.tensor_operators import TENSOR_OPERATORS, ROOT_OPERATORS, MULTI_DIM_OPERATORS
from src.symbolic.large_feature_bank import LargeFeatureBank
from src.rl.tensor_action_masking import TensorActionMasker
from src.rl.rpn_grammar_mask import RPNGrammarMask


class TensorTokenVocabulary:
    """Vocabulary for tensor-based VSR."""

    def __init__(self, exclude_operators=None, extra_terminals=None):
        # Special tokens
        special_tokens = ['START', 'END', 'PAD']

        # Operators from tensor_operators (optionally filtered)
        exclude = set(exclude_operators or [])
        operator_tokens = [op for op in TENSOR_OPERATORS.keys() if op not in exclude]
        if exclude:
            print(f"  [Vocab] Excluded operators: {sorted(exclude)}")
            print(f"  [Vocab] Active operators: {len(operator_tokens)}")

        # Terminals (RGB + grayscale + HSV + color ratios + opponent channels)
        terminal_tokens = ['I_R', 'I_G', 'I_B', 'I_GRAY', 'I_H', 'I_S',
                           'I_r', 'I_g', 'I_RG', 'I_BY']

        # Layer 2: additional terminals from Layer 1 feature maps
        if extra_terminals:
            terminal_tokens.extend(extra_terminals)
            print(f"  [Vocab] Extra terminals: {len(extra_terminals)} (L1 feature maps)")

        # Build vocabulary
        self.tokens = special_tokens + operator_tokens + terminal_tokens
        self.token_to_idx = {t: i for i, t in enumerate(self.tokens)}
        self.idx_to_token = {i: t for i, t in enumerate(self.tokens)}

    def encode(self, token: str) -> int:
        return self.token_to_idx[token]

    def decode(self, idx: int) -> str:
        return self.idx_to_token[idx]

    def __len__(self):
        return len(self.tokens)


class TensorVSREnvironmentLargeBank(gym.Env):
    """
    Enhanced environment with large feature bank and diversity rewards.

    Key features:
    - Accepts formulas with low threshold (10%)
    - Computes diversity penalty to avoid redundant features
    - Periodically runs LASSO to prune redundant formulas
    - Supports 200-300 formula capacity
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
        dataset_cfg = config.get('dataset_options', {}) or {}
        self.num_classes = train_cfg.get('num_classes',
                           dataset_cfg.get('num_classes', 10))

        # Read feature-bank params from 'strategy' section (fallback: training)
        strategy_cfg = config.get('strategy', {})
        self.feature_bank_size = strategy_cfg.get('feature_bank_size',
                                  train_cfg.get('feature_bank_size', 1000))
        self.lasso_target = strategy_cfg.get('lasso_target_features',
                             train_cfg.get('lasso_target_features', 1000))
        self.l1_lambda = strategy_cfg.get('l1_lambda',
                          train_cfg.get('l1_lambda', 0.0))
        self.lasso_epochs = strategy_cfg.get('lasso_epochs',
                             train_cfg.get('lasso_epochs', 100))
        self.length_penalty = train_cfg.get('length_penalty', 0.01)
        self.diversity_penalty_coef = strategy_cfg.get('diversity_penalty',
                                      train_cfg.get('diversity_penalty', 0.15))
        self.min_accuracy = strategy_cfg.get('min_accuracy_threshold',
                             train_cfg.get('min_accuracy', 0.015))
        self.correlation_threshold = strategy_cfg.get('correlation_threshold', 0.90)

        # v3.3 Section 7A: bank admission gate (degeneracy rejection only — WIDE).
        # Off by default (legacy min_accuracy gating preserved); enable via config.
        self.admission_tracker = None
        admission_cfg = config.get('bank_admission', None)
        if admission_cfg:
            from src.symbolic.bank_admission import AdmissionTracker
            self.admission_tracker = AdmissionTracker(admission_cfg)
            print(f"  [Admission] Wide degeneracy gate active: {admission_cfg}")

        # Reward signal config
        self.reward_type = strategy_cfg.get('reward_type', 'accuracy')  # 'accuracy' or 'loss_based'
        self.use_hierarchical_eval = strategy_cfg.get('use_hierarchical_eval', False)
        self.hierarchical_switch_fraction = strategy_cfg.get('hierarchical_switch_fraction', 0.5)
        self._hierarchical_active = self.use_hierarchical_eval  # starts active if enabled

        # Superclass mapping for hierarchical evaluation (set externally for ImageNet)
        self.superclass_mapping = None  # Dict[int, int] mapping class_id -> superclass_id
        self.num_superclasses = None

        # Resolution-adaptive evaluation
        dataset_options = config.get('dataset_options', {}) or {}
        self.eval_resolution_quick = dataset_options.get('resolution_quick', None)
        self.eval_resolution_full = dataset_options.get('resolution_full', None)

        # Learnable kernel bank — must register BEFORE building vocabulary
        from src.symbolic.tensor_operators import SymbolicKernelBank
        self.kernel_bank = SymbolicKernelBank(device=self.device)
        # Load pretrained kernel weights if specified in config
        kb_path = config.get('kernel_bank_path', None)
        if kb_path and os.path.exists(kb_path):
            self.kernel_bank.load_state_dict(
                torch.load(kb_path, map_location=self.device, weights_only=True))
            print(f"  [Kernel] Loaded pretrained weights from {kb_path}")
        self.kernel_bank.register_operators(TENSOR_OPERATORS)

        # Layer 2 support: load Layer 1 bodies as extra terminals
        self.l1_bodies = None
        self.l1_terminal_names = []
        l1_bodies_path = config.get('l1_bodies_path', None)
        if l1_bodies_path and os.path.exists(l1_bodies_path):
            import json as _json
            self.l1_bodies = _json.load(open(l1_bodies_path))
            self.l1_terminal_names = [f'L1_{i}' for i in range(len(self.l1_bodies))]
            print(f"  [L2] Loaded {len(self.l1_bodies)} Layer 1 bodies as terminals")

        # Vocabulary (optionally exclude operators like spp_pool)
        self.exclude_operators = config.get('exclude_operators', []) or []
        self.vocabulary = TensorTokenVocabulary(
            exclude_operators=self.exclude_operators,
            extra_terminals=self.l1_terminal_names if self.l1_bodies else None,
        )

        # GPU-side data augmentation
        self.augment = config.get('augment', False)

        # Define spaces
        self.action_space = gym.spaces.Discrete(len(self.vocabulary))
        self.observation_space = gym.spaces.Box(
            low=0,
            high=len(self.vocabulary),
            shape=(self.max_sequence_length,),
            dtype=int
        )

        # Action masker (旧的基于深度的掩码)
        self.action_masker = TensorActionMasker(
            self.vocabulary,
            max_depth=self.max_depth
        )

        # RPN Grammar Masker (新的严格RPN语法掩码)
        self.rpn_masker = RPNGrammarMask(
            self.vocabulary,
            max_sequence_length=self.max_sequence_length
        )

        # Large feature bank (Scheme A: Survival of the Fittest)
        self.feature_bank = LargeFeatureBank(
            max_size=self.feature_bank_size,
            min_accuracy=self.min_accuracy,
            correlation_threshold=self.correlation_threshold,
            correlation_threshold_full=strategy_cfg.get('correlation_threshold_full', None),
            num_classes=self.num_classes,
            device=self.device,
            adaptive_threshold=strategy_cfg.get('adaptive_threshold', False),
            threshold_warmup_fraction=strategy_cfg.get('threshold_warmup_fraction', 0.5),
            lasso_target=self.lasso_target,
            l1_lambda=self.l1_lambda,
            lasso_epochs=self.lasso_epochs,
        )

        # Cache validation data
        self.cached_images = None
        self.cached_labels = None
        self._cache_validation_set()

        # Data iterator
        self.data_iter = iter(data_loader)
        self.batch_size = train_cfg.get('batch_size', 64)
        self.eval_batch_size = train_cfg.get('eval_batch_size', 512)  # Large batch for accurate rewards

        # Sub-expression cache (keyed by RPN prefix string → tensor)
        # Invalidated when the data batch changes
        self._subexpr_cache: Dict[str, torch.Tensor] = {}
        self._subexpr_cache_batch_id: int = -1

        # Episode state
        self.current_sequence = []
        self.step_count = 0
        self.episode_count = 0

        # LASSO pruning frequency (effectively disabled when l1_lambda=0)
        default_prune = 9999 if self.l1_lambda == 0.0 else 50
        self.lasso_prune_interval = train_cfg.get('lasso_prune_interval',
                                                   default_prune)

    def _cache_validation_set(self) -> None:
        """Cache all validation images."""
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

    def _gpu_augment(self, images: torch.Tensor) -> torch.Tensor:
        """On-the-fly GPU augmentation: random crop (pad=4) + horizontal flip."""
        B, C, H, W = images.shape
        # Pad with reflection: [B, C, H+8, W+8]
        padded = torch.nn.functional.pad(images, [4, 4, 4, 4], mode='reflect')
        # Random crop offsets per image
        top = torch.randint(0, 9, (B,), device=images.device)
        left = torch.randint(0, 9, (B,), device=images.device)
        # Vectorized crop: gather each image's crop
        cropped = torch.empty_like(images)
        for i in range(B):
            cropped[i] = padded[i, :, top[i]:top[i]+H, left[i]:left[i]+W]
        # Random horizontal flip (50% chance per image)
        flip_mask = torch.rand(B, device=images.device) < 0.5
        cropped[flip_mask] = cropped[flip_mask].flip(-1)
        return cropped

    def get_data_batch(
        self, batch_size: Optional[int] = None, resolution: Optional[int] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get a batch of images as RGB channels.

        Args:
            batch_size: Number of images to sample.
            resolution: If set, downscale images to this resolution (e.g. 64).
                        None means use native resolution.
        """
        if batch_size is None:
            batch_size = self.batch_size

        # Sample from cached data
        batch_size = min(batch_size, self.cached_images.size(0))
        indices = torch.randint(0, self.cached_images.size(0), (batch_size,), device=self.device)
        images = self.cached_images[indices]
        labels = self.cached_labels[indices]

        # Apply on-the-fly augmentation if enabled
        if self.augment:
            images = self._gpu_augment(images)

        # Resolution-adaptive: downscale for fast evaluation
        if resolution is not None and resolution < images.shape[-1]:
            images = F_func.interpolate(
                images, size=(resolution, resolution),
                mode='bilinear', align_corners=False
            )

        # Extract RGB channels + grayscale
        I_R = images[:, 0, :, :]
        I_G = images[:, 1, :, :]
        I_B = images[:, 2, :, :]

        # HSV conversion (deterministic mathematical formula)
        Cmax, _ = images.max(dim=1)   # [B, H, W]
        Cmin, _ = images.min(dim=1)
        delta = Cmax - Cmin + 1e-8

        # Hue: [0, 1] normalized
        H = torch.zeros_like(I_R)
        mask_r = (Cmax == I_R)
        mask_g = (Cmax == I_G) & ~mask_r
        mask_b = ~mask_r & ~mask_g
        H[mask_r] = (((I_G[mask_r] - I_B[mask_r]) / delta[mask_r]) % 6)
        H[mask_g] = ((I_B[mask_g] - I_R[mask_g]) / delta[mask_g]) + 2
        H[mask_b] = ((I_R[mask_b] - I_G[mask_b]) / delta[mask_b]) + 4
        H = H / 6.0  # normalize to [0, 1]

        # Saturation
        S = torch.where(Cmax > 1e-8, delta / (Cmax + 1e-8), torch.zeros_like(Cmax))

        I_GRAY = 0.2989 * I_R + 0.5870 * I_G + 0.1140 * I_B

        # Color ratios (illumination invariant)
        total = I_R + I_G + I_B + 1e-8
        I_r_ratio = I_R / total
        I_g_ratio = I_G / total
        # Opponent channels
        I_RG = I_R - I_G
        I_BY = I_B - (I_R + I_G) / 2

        terminal_values = {
            'I_R': I_R,
            'I_G': I_G,
            'I_B': I_B,
            'I_GRAY': I_GRAY,
            'I_H': H,
            'I_S': S,
            'I_r': I_r_ratio,
            'I_g': I_g_ratio,
            'I_RG': I_RG,
            'I_BY': I_BY,
        }

        # Layer 2: compute L1 feature maps as extra terminals
        if self.l1_bodies:
            for i, body_str in enumerate(self.l1_bodies):
                name = f'L1_{i}'
                try:
                    tokens = body_str.strip().split()
                    stack = []
                    for token in tokens:
                        if token in terminal_values:
                            stack.append(terminal_values[token])
                        elif token in TENSOR_OPERATORS:
                            op_func, arity, _ = TENSOR_OPERATORS[token]
                            if len(stack) < arity:
                                stack = [torch.zeros_like(I_R)]
                                break
                            operands = [stack.pop() for _ in range(arity)]
                            operands.reverse()
                            result = op_func(*operands)
                            result = torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)
                            stack.append(result)
                        else:
                            stack = [torch.zeros_like(I_R)]
                            break
                    if len(stack) == 1 and stack[0].dim() >= 2:
                        terminal_values[name] = torch.clamp(stack[0], -1e4, 1e4)
                    else:
                        terminal_values[name] = torch.zeros_like(I_R)
                except Exception:
                    terminal_values[name] = torch.zeros_like(I_R)

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

            # Periodic LASSO pruning
            if (self.episode_count % self.lasso_prune_interval == 0 and
                self.feature_bank.is_full()):
                self._run_lasso_pruning()

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
        """
        获取当前状态下的合法动作掩码。

        使用严格的RPN语法掩码确保100%合法的公式生成。
        """
        # 获取当前序列（去除START）
        current_tokens = [
            t for t in self.current_sequence
            if self.vocabulary.decode(t) not in ['START', 'PAD']
        ]

        # 使用RPN语法掩码
        mask = self.rpn_masker.get_valid_actions(
            current_tokens,
            device=self.device
        )

        return mask

    def _execute_formula(self, formula_tokens, data_batch, use_cache=False, batch_id=0):
        """
        Execute a tensor formula with NaN/Inf detection.

        Args:
            formula_tokens: List of token indices.
            data_batch: Dict of terminal tensors.
            use_cache: If True, use sub-expression caching for shared prefixes.
            batch_id: Identifier for the current data batch (cache is invalidated on change).

        Returns:
            output: [batch] or [batch, D] tensor (D>1 for multi-dim ops like spp_pool)
            is_valid: bool (False if NaN/Inf detected)
        """
        # Invalidate cache if batch changed
        if use_cache and batch_id != self._subexpr_cache_batch_id:
            self._subexpr_cache.clear()
            self._subexpr_cache_batch_id = batch_id

        decoded = [self.vocabulary.decode(t) for t in formula_tokens]
        stack = []
        prefix_parts = []

        for token in decoded:
            if use_cache:
                prefix_parts.append(token)
                prefix_key = ' '.join(prefix_parts)
                if prefix_key in self._subexpr_cache:
                    stack = [self._subexpr_cache[prefix_key]]
                    continue

            if token in data_batch:
                stack.append(data_batch[token])
            elif token in TENSOR_OPERATORS:
                op_func, arity, output_type = TENSOR_OPERATORS[token]

                if len(stack) < arity:
                    raise ValueError(f"Not enough operands for {token}")

                operands = [stack.pop() for _ in range(arity)]
                operands.reverse()

                result = op_func(*operands)

                # CHECK FOR NaN/Inf after each operation
                if torch.isnan(result).any() or torch.isinf(result).any():
                    formula_str = ' '.join(decoded)
                    print(f"WARNING: NaN/Inf detected at operator {token} in formula: {formula_str}")
                    return None, False

                stack.append(result)

            # Cache prefix result if stack has exactly 1 item
            if use_cache and len(stack) == 1:
                self._subexpr_cache[prefix_key] = stack[0]

        if len(stack) != 1:
            raise ValueError(f"Invalid formula: stack has {len(stack)} elements")

        output = stack[0]

        # Valid output: [batch] for scalar ops, [batch, D] for multi-dim ops (e.g. spp_pool)
        if output.dim() not in (1, 2):
            raise ValueError(f"Formula output must be [batch] or [batch, D], got {output.shape}")

        # Final NaN/Inf check
        if torch.isnan(output).any() or torch.isinf(output).any():
            formula_str = ' '.join(decoded)
            print(f"WARNING: NaN/Inf detected in final output: {formula_str}")
            return None, False

        return output, True

    def _compute_diversity_penalty(self, formula_output: torch.Tensor) -> float:
        """Max |Pearson r| between *formula_output* and all bank entries."""
        # Flatten multi-dim outputs for correlation computation
        flat_output = formula_output.reshape(formula_output.shape[0], -1).mean(dim=1) if formula_output.dim() > 1 else formula_output
        return self.feature_bank.get_max_correlation(flat_output)

    def set_superclass_mapping(self, mapping: Dict[int, int], num_superclasses: int):
        """Set the class→superclass mapping for hierarchical evaluation."""
        self.superclass_mapping = mapping
        self.num_superclasses = num_superclasses

    def update_hierarchical_state(self):
        """Disable hierarchical eval once bank passes switch fraction."""
        if not self.use_hierarchical_eval:
            return
        fill_ratio = self.feature_bank.size() / max(1, self.feature_bank.max_size)
        if fill_ratio >= self.hierarchical_switch_fraction:
            if self._hierarchical_active:
                print("[Env] Switching from hierarchical to full-class evaluation")
            self._hierarchical_active = False

    def _compute_loss_based_reward(
        self, feature_tensor: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[float, float, float]:
        """
        Compute continuous loss-based reward (more informative than accuracy for many classes).

        Returns:
            (normalized_reward, top1_accuracy, top5_accuracy)
        """
        from src.models.lasso_classifier import LASSOLinearClassifier

        eval_labels = labels
        n_classes = self.num_classes

        # Hierarchical: use superclass labels in early training
        if (self._hierarchical_active and self.superclass_mapping is not None
                and self.num_superclasses is not None):
            mapping_t = torch.zeros(max(self.superclass_mapping.keys()) + 1,
                                    dtype=torch.long, device=labels.device)
            for k, v in self.superclass_mapping.items():
                mapping_t[k] = v
            eval_labels = mapping_t[labels]
            n_classes = self.num_superclasses

        # Train a quick linear classifier
        model = LASSOLinearClassifier(feature_tensor.shape[1], n_classes).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        for _ in range(20):
            optimizer.zero_grad()
            logits = model(feature_tensor)
            loss = criterion(logits, eval_labels)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            logits = model(feature_tensor)
            ce_loss = criterion(logits, eval_labels).item()

            # Top-1 accuracy
            preds = logits.argmax(dim=1)
            top1_acc = (preds == eval_labels).float().mean().item()

            # Top-5 accuracy (capped at num_classes)
            k = min(5, n_classes)
            _, top_k_preds = logits.topk(k, dim=1)
            top5_correct = (top_k_preds == eval_labels.unsqueeze(1)).any(dim=1)
            top5_acc = top5_correct.float().mean().item()

        # Normalized loss reward: 0 at random chance, 1 at perfect
        max_loss = math.log(n_classes)
        normalized_loss_reward = max(0.0, 1.0 - ce_loss / max_loss)

        # Composite reward
        reward_score = (
            0.6 * normalized_loss_reward
            + 0.3 * top5_acc
            + 0.1 * top1_acc
        )

        return reward_score, top1_acc, top5_acc

    def _calculate_reward(self) -> Tuple[float, Dict]:
        """
        Calculate reward with diversity penalty.

        Supports two modes:
        - 'accuracy': original accuracy-based reward (for CIFAR)
        - 'loss_based': continuous loss + top-k reward (for ImageNet)
        """
        expression_tokens = [
            t for t in self.current_sequence
            if self.vocabulary.decode(t) not in ["START", "END", "PAD"]
        ]

        try:
            # 1. Validate formula
            if not expression_tokens:
                return -1.0, {"valid": False, "reason": "empty"}

            formula_str = ' '.join([self.vocabulary.decode(t) for t in expression_tokens])

            # Get validation batch (use LARGE batch for accurate reward)
            # Use quick resolution for RL episode evaluation (faster)
            data_batch, labels = self.get_data_batch(
                batch_size=self.eval_batch_size,
                resolution=self.eval_resolution_quick
            )

            # 2. Execute formula
            try:
                formula_output, is_valid = self._execute_formula(expression_tokens, data_batch)

                if not is_valid or formula_output is None:
                    return -1.0, {"valid": False, "reason": "nan_inf_detected"}

                formula_output = torch.nan_to_num(formula_output, nan=0.0, posinf=1e4, neginf=-1e4)
            except Exception as e:
                partial_reward = -1.0
                has_pooling = any(self.vocabulary.decode(t) in ROOT_OPERATORS for t in expression_tokens)
                has_terminal = any(self.vocabulary.decode(t).startswith('I_') for t in expression_tokens)

                if has_pooling and has_terminal:
                    partial_reward += 0.4
                elif has_pooling:
                    partial_reward += 0.2
                elif has_terminal:
                    partial_reward += 0.2

                formula_length = len(expression_tokens)
                if formula_length == 2:
                    partial_reward += 0.3
                elif formula_length <= 4:
                    partial_reward += 0.2

                return partial_reward, {"valid": False, "reason": f"execution_error: {e}"}

            # 3. Build feature tensor
            if formula_output.dim() == 1:
                feature_tensor = formula_output.unsqueeze(1)
            else:
                feature_tensor = formula_output

            feat_mean = feature_tensor.mean(dim=0, keepdim=True)
            feat_std = feature_tensor.std(dim=0, keepdim=True) + 1e-8
            feature_tensor = (feature_tensor - feat_mean) / feat_std

            # 4. Compute accuracy / reward score based on reward_type
            if self.reward_type == 'loss_based':
                reward_score, top1_acc, top5_acc = self._compute_loss_based_reward(
                    feature_tensor, labels
                )
                accuracy = top1_acc  # for bank admission
            else:
                # Original accuracy-based reward
                from src.models.lasso_classifier import train_lasso_classifier
                accuracy, _, _ = train_lasso_classifier(
                    feature_tensor, labels,
                    num_classes=self.num_classes,
                    l1_lambda=0.0, epochs=20,
                    device=self.device
                )
                reward_score = accuracy
                top5_acc = 0.0

            # 5. Compute diversity penalty
            diversity_penalty = self._compute_diversity_penalty(formula_output)

            # 6. Add to feature bank
            is_new = formula_str not in self.feature_bank.formula_strs
            if is_new:
                class SimpleFormula:
                    def __init__(self, tokens, vocabulary, operators):
                        self.tokens = tokens
                        self.vocabulary = vocabulary
                        self.operators = operators

                    def execute(self, data_batch):
                        decoded = [self.vocabulary.decode(t) for t in self.tokens]
                        stack = []
                        for token in decoded:
                            if token in data_batch:
                                stack.append(data_batch[token])
                            elif token in self.operators:
                                op_func, arity, _ = self.operators[token]
                                if len(stack) < arity:
                                    raise ValueError(f"Not enough operands for {token}")
                                operands = [stack.pop() for _ in range(arity)]
                                operands.reverse()
                                result = op_func(*operands)
                                if torch.isnan(result).any() or torch.isinf(result).any():
                                    raise ValueError(f"NaN/Inf at {token}")
                                stack.append(result)
                        output = stack[0]
                        if torch.isnan(output).any() or torch.isinf(output).any():
                            raise ValueError("NaN/Inf in final output")
                        return output

                formula_obj = SimpleFormula(expression_tokens, self.vocabulary, TENSOR_OPERATORS)
                corr_vector = formula_output.reshape(formula_output.shape[0], -1).mean(dim=1) if formula_output.dim() > 1 else formula_output

                # v3.3 Section 7A: wide degeneracy gate (only when configured). Rejects
                # statistically broken formulas (constant/non-finite/saturated) but NEVER
                # merely-weak ones; discriminative pruning happens in the periodic reshuffle.
                admit = True
                if self.admission_tracker is not None:
                    admit, reason = self.admission_tracker.check(corr_vector, accuracy=accuracy)
                    if not admit:
                        print(f"  [Admission] rejected ({reason}): {formula_str}")

                if admit:
                    success, msg = self.feature_bank.add_formula(
                        formula=formula_obj,
                        formula_str=formula_str,
                        length=len(expression_tokens),
                        accuracy=accuracy,
                        output_vector=corr_vector,
                    )
                    if success:
                        print(f"  [Bank] {msg}")

            # 7. Compute final reward
            decoded_tokens = [self.vocabulary.decode(t) for t in expression_tokens]

            # Consecutive duplicate unary operator penalty
            repeat_penalty = 0.0
            for i in range(1, len(decoded_tokens)):
                if decoded_tokens[i] == decoded_tokens[i - 1] and decoded_tokens[i] in TENSOR_OPERATORS:
                    _, arity, _ = TENSOR_OPERATORS[decoded_tokens[i]]
                    if arity == 1:
                        repeat_penalty += 0.05

            reward = (
                reward_score
                - self.length_penalty * len(expression_tokens)
                - self.diversity_penalty_coef * diversity_penalty
                - repeat_penalty
            )

            # Complexity bonus
            spatial_ops = {'blur', 'blur_7x7', 'edge_x', 'edge_y', 'dilate', 'laplacian', 'normalize',
                          'opening', 'closing', 'tophat', 'high_freq', 'low_freq'}
            num_spatial_ops = len([t for t in decoded_tokens if t in spatial_ops])
            complexity_bonus = 0.05 if (len(expression_tokens) > 3 and num_spatial_ops >= 1) else 0.0

            if not is_new:
                reward -= 0.3
            else:
                reward += 0.05 + complexity_bonus
                has_binary = any(t in ('add', 'subtract', 'multiply') for t in decoded_tokens)
                all_terminals = {'I_R', 'I_G', 'I_B', 'I_GRAY', 'I_H', 'I_S', 'I_r', 'I_g', 'I_RG', 'I_BY'}
                all_terminals.update(self.l1_terminal_names)
                has_multi_channel = len(set(decoded_tokens) & all_terminals) >= 2
                if has_binary and has_multi_channel:
                    reward += 0.05

            reward = max(-1.0, min(1.0, reward))

            metrics = {
                "valid": True,
                "formula": formula_str,
                "accuracy": accuracy,
                "top5_accuracy": top5_acc if self.reward_type == 'loss_based' else 0.0,
                "diversity_penalty": diversity_penalty,
                "bank_size": self.feature_bank.size(),
                "reward": reward
            }

            return reward, metrics

        except Exception as e:
            import traceback
            print(f"[Reward Error] {e}")
            traceback.print_exc()
            return -1.0, {"valid": False, "error": str(e)}

    def _run_lasso_pruning(self):
        """Run LASSO to identify the best features."""
        print(f"\n{'='*60}")
        print(f"Running LASSO Pruning (Episode {self.episode_count})")
        print(f"{'='*60}")

        # Get full dataset for LASSO
        data_batch, labels = self.get_data_batch(batch_size=min(1000, self.cached_images.size(0)))

        # Run LASSO
        accuracy, selected_count = self.feature_bank.train_lasso_and_prune(
            data_batch=data_batch,
            labels=labels
        )

        # Print summary
        print(self.feature_bank.get_summary())
