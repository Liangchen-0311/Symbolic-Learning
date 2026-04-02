"""
Evaluator for tensor-based VSR programs with LASSO.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from src.symbolic.tensor_operators import TENSOR_OPERATORS
from src.models.lasso_classifier import train_lasso_classifier


class TensorProgramEvaluator:
    """
    Evaluates tensor-based symbolic programs using LASSO classifier.

    Key differences from standard evaluator:
    1. Works with image tensors [batch, H, W] not scalar features
    2. Uses LASSO classifier for feature selection
    3. Evaluates feature banks (multiple formulas)
    4. Counts active features
    """

    def __init__(self, num_classes: int = 10, device: str = "cuda"):
        self.num_classes = num_classes
        self.device = device

    def execute_formula(self, formula_tokens, vocabulary, data_batch):
        """
        Execute a tensor formula with NaN/Inf detection.

        Args:
            formula_tokens: List of token indices
            vocabulary: TensorTokenVocabulary
            data_batch: Dict with I_R, I_G, I_B

        Returns:
            output: [batch] scalar features
            is_valid: bool (False if NaN/Inf detected)
        """
        # Decode tokens
        decoded = [vocabulary.decode(t) for t in formula_tokens]

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

                # Pop operands (in reverse order)
                operands = [stack.pop() for _ in range(arity)]
                operands.reverse()

                # Apply operator
                result = op_func(*operands)

                # CHECK FOR NaN/Inf after each operation
                if torch.isnan(result).any() or torch.isinf(result).any():
                    formula_str = ' '.join(decoded)
                    print(f"WARNING: NaN/Inf detected during formula execution: {formula_str} at operator {token}")
                    return None, False

                stack.append(result)

        if len(stack) != 1:
            raise ValueError(f"Invalid formula: stack has {len(stack)} elements")

        output = stack[0]

        # Valid output: [batch] for scalar ops, [batch, D] for multi-dim ops
        if output.dim() not in (1, 2):
            raise ValueError(f"Formula output must be [batch] or [batch, D], got {output.shape}")

        # Final NaN/Inf check
        if torch.isnan(output).any() or torch.isinf(output).any():
            formula_str = ' '.join(decoded)
            print(f"WARNING: NaN/Inf detected in final output: {formula_str}")
            return None, False

        return output, True

    def evaluate_feature_bank(
        self,
        feature_bank: List[Dict],
        vocabulary,
        data_batch: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        l1_lambda: float = 0.01,
        lasso_epochs: int = 50
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate feature bank with LASSO classifier.

        Args:
            feature_bank: List of formula dictionaries
            vocabulary: TensorTokenVocabulary
            data_batch: Dict with I_R, I_G, I_B
            labels: Class labels [batch]
            l1_lambda: L1 regularization strength
            lasso_epochs: LASSO training epochs

        Returns:
            accuracy: Classification accuracy
            active_features: Number of active features
            metrics: Additional metrics
        """
        if len(feature_bank) == 0:
            return 0.0, 0, {}

        # Extract features from all formulas
        features_list = []
        for formula_dict in feature_bank:
            try:
                feature, is_valid = self.execute_formula(
                    formula_dict['tokens'],
                    vocabulary,
                    data_batch
                )
                # If NaN/Inf detected, skip this formula
                if not is_valid or feature is None:
                    print(f"  [Warning] Skipping formula due to NaN/Inf")
                    continue

                feature = torch.nan_to_num(feature, nan=0.0, posinf=1e4, neginf=-1e4)
                features_list.append(feature)
            except Exception as e:
                print(f"  [Warning] Failed to execute formula: {e}")
                continue

        if len(features_list) == 0:
            return 0.0, 0, {"error": "no_valid_features"}

        # Concatenate features: [batch, total_dims]
        # Each feature may be [batch] (scalar) or [batch, D] (multi-dim)
        features_2d = []
        for f in features_list:
            if f.dim() == 1:
                features_2d.append(f.unsqueeze(1))
            else:
                features_2d.append(f)
        features_tensor = torch.cat(features_2d, dim=1)

        # Train LASSO classifier
        accuracy, active_features, model = train_lasso_classifier(
            features_tensor,
            labels,
            num_classes=self.num_classes,
            l1_lambda=l1_lambda,
            epochs=lasso_epochs,
            device=self.device
        )

        metrics = {
            "accuracy": accuracy,
            "active_features": active_features,
            "total_features": len(features_list),
            "sparsity": active_features / len(features_list) if len(features_list) > 0 else 0.0
        }

        return accuracy, active_features, metrics

    def evaluate_single_formula(
        self,
        formula_tokens: List[int],
        vocabulary,
        data_batch: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> Tuple[float, Dict]:
        """
        Evaluate a single formula (for initial testing).

        Args:
            formula_tokens: Token indices
            vocabulary: TensorTokenVocabulary
            data_batch: Dict with I_R, I_G, I_B
            labels: Class labels [batch]

        Returns:
            accuracy: Classification accuracy
            metrics: Additional metrics
        """
        try:
            # Execute formula
            output, is_valid = self.execute_formula(formula_tokens, vocabulary, data_batch)

            # If NaN/Inf detected, return 0.0 accuracy with large penalty
            if not is_valid or output is None:
                return 0.0, {"accuracy": 0.0, "error": "nan_inf_detected"}

            output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)

            # Normalize output
            mean = output.mean()
            std = output.std()
            if std > 1e-5:
                output_norm = (output - mean) / (std + 1e-8)
            else:
                output_norm = output - mean

            # Train simple linear classifier
            features = output_norm.view(-1, 1)

            # Split train/val
            n_samples = features.size(0)
            n_train = int(0.7 * n_samples)

            indices = torch.randperm(n_samples, device=self.device)
            train_idx = indices[:n_train]
            val_idx = indices[n_train:]

            train_features = features[train_idx]
            train_labels = labels[train_idx]
            val_features = features[val_idx]
            val_labels = labels[val_idx]

            # Train linear classifier
            classifier = nn.Linear(1, self.num_classes).to(self.device)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=0.05)
            criterion = nn.CrossEntropyLoss()

            classifier.train()
            for _ in range(20):
                optimizer.zero_grad()
                logits = classifier(train_features)
                loss = criterion(logits, train_labels)
                loss.backward()
                optimizer.step()

            # Evaluate
            classifier.eval()
            with torch.no_grad():
                val_logits = classifier(val_features)
                val_preds = torch.argmax(val_logits, dim=1)
                accuracy = (val_preds == val_labels).float().mean().item()

            metrics = {
                "accuracy": accuracy,
                "output_mean": mean.item(),
                "output_std": std.item()
            }

            return accuracy, metrics

        except Exception as e:
            return 0.0, {"error": str(e)}
