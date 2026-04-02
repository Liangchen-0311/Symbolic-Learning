import re
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from src.symbolic.program import SymbolicProgram


class FeatureBank:
    """
    Tiered feature bank: simple formulas (len<=3) + complex formulas (len>3).

    Improvements over v1:
    1. No duplicate formulas allowed
    2. Dynamic accuracy threshold (decays after stagnation)
    3. Tiered slots: simple (0-2) and complex (3-7) don't compete
    4. Bank size 8 for richer ensembles
    """

    SIMPLE_THRESHOLD = 3  # formulas with token length <= 3 are "simple"

    def __init__(
        self,
        max_size: int = 8,
        max_simple: int = 3,
        max_complex: int = 5,
        min_accuracy: float = 0.15,
        min_diversity: float = 0.3,
        num_classes: int = 10,
        device: str = "cpu"
    ):
        self.max_size = max_size
        self.max_simple = max_simple
        self.max_complex = max_complex
        self.min_accuracy = min_accuracy
        self.min_diversity = min_diversity
        self.num_classes = num_classes
        self.device = device

        # Storage
        self.formulas: List[SymbolicProgram] = []
        self.accuracies: List[float] = []
        self.outputs_cache: List[torch.Tensor] = []
        self.formula_strs: List[str] = []
        self.formula_lengths: List[int] = []

        # Stagnation tracking
        self.episodes_since_last_update = 0

    def size(self) -> int:
        return len(self.formulas)

    def is_full(self) -> bool:
        return len(self.formulas) >= self.max_size

    def _is_simple(self, length: int) -> bool:
        return length <= self.SIMPLE_THRESHOLD

    def _tier_indices(self, simple: bool) -> List[int]:
        return [
            i for i, l in enumerate(self.formula_lengths)
            if self._is_simple(l) == simple
        ]

    def _tier_count(self, simple: bool) -> int:
        return len(self._tier_indices(simple))

    def _tier_max(self, simple: bool) -> int:
        return self.max_simple if simple else self.max_complex

    def compute_diversity(
        self,
        new_output: torch.Tensor,
        new_formula_str: str,
        new_vars: set
    ) -> float:
        """Diversity: 0.0 (identical) to 1.0 (completely different)."""
        if len(self.formulas) == 0:
            return 1.0

        diversity_scores = []
        for i, cached_output in enumerate(self.outputs_cache):
            corr = self._correlation(new_output, cached_output)
            output_div = 1.0 - abs(corr)

            cached_vars = self._extract_variables(self.formula_strs[i])
            jaccard = len(new_vars & cached_vars) / max(len(new_vars | cached_vars), 1)
            var_div = 1.0 - jaccard

            edit_dist = self._edit_distance(new_formula_str, self.formula_strs[i])
            max_len = max(len(new_formula_str), len(self.formula_strs[i]))
            struct_div = min(edit_dist / max(max_len, 1), 1.0)

            formula_diversity = 0.5 * output_div + 0.3 * var_div + 0.2 * struct_div
            diversity_scores.append(formula_diversity)

        return min(diversity_scores)

    def should_accept(
        self,
        accuracy: float,
        diversity: float,
        formula_str: str,
        length: int
    ) -> bool:
        """
        Decide whether to accept new formula.

        Checks:
        1. No exact duplicates
        2. Dynamic accuracy threshold (decays after stagnation)
        3. Diversity threshold
        4. Tier-aware: only competes within its own tier
        """
        # Reject exact duplicates
        if formula_str in self.formula_strs:
            return False

        # Dynamic accuracy threshold: lower after stagnation
        if self.episodes_since_last_update > 20:
            decay = 0.01 * (self.episodes_since_last_update - 20) / 10
            effective_min_acc = max(0.10, self.min_accuracy - decay)
        else:
            effective_min_acc = self.min_accuracy

        if accuracy < effective_min_acc:
            return False

        if diversity < self.min_diversity:
            return False

        # Check tier capacity
        simple = self._is_simple(length)
        tier_idx = self._tier_indices(simple)
        tier_max = self._tier_max(simple)

        if len(tier_idx) < tier_max and len(self.formulas) < self.max_size:
            return True

        if len(tier_idx) > 0:
            worst_tier_acc = min(self.accuracies[i] for i in tier_idx)
            return accuracy > worst_tier_acc

        return False

    def add_formula(
        self,
        formula: SymbolicProgram,
        accuracy: float,
        output: torch.Tensor,
        formula_str: str,
        length: int
    ) -> Tuple[bool, str]:
        """Add formula to the appropriate tier."""
        simple = self._is_simple(length)
        tier_name = "simple" if simple else "complex"
        tier_idx = self._tier_indices(simple)
        tier_max = self._tier_max(simple)

        self.episodes_since_last_update = 0

        if len(tier_idx) < tier_max and len(self.formulas) < self.max_size:
            self.formulas.append(formula)
            self.accuracies.append(accuracy)
            self.outputs_cache.append(output.detach().cpu())
            self.formula_strs.append(formula_str)
            self.formula_lengths.append(length)
            return True, f"Added [{tier_name}] (slot {len(self.formulas)}/{self.max_size})"

        if len(tier_idx) > 0:
            worst_i = min(tier_idx, key=lambda i: self.accuracies[i])
            old_acc = self.accuracies[worst_i]

            self.formulas[worst_i] = formula
            self.accuracies[worst_i] = accuracy
            self.outputs_cache[worst_i] = output.detach().cpu()
            self.formula_strs[worst_i] = formula_str
            self.formula_lengths[worst_i] = length
            return True, f"Replaced [{tier_name}] slot {worst_i} (old: {old_acc:.3f}, new: {accuracy:.3f})"

        return False, "No room in tier"

    def tick(self):
        """Call once per episode to track stagnation."""
        self.episodes_since_last_update += 1

    def evaluate_ensemble(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        n_train_steps: int = 20
    ) -> Tuple[float, Dict]:
        """Evaluate ensemble: Linear(num_formulas, 10) on all bank outputs."""
        if len(self.formulas) == 0:
            return 0.0, {"error": "empty_bank"}

        try:
            all_outputs = []
            for formula in self.formulas:
                with torch.no_grad():
                    output = formula.execute(z)
                    output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)
                    output = output.view(-1, 1)
                    mean = output.mean()
                    std = output.std()
                    if std > 1e-5:
                        output_norm = (output - mean) / (std + 1e-8)
                    else:
                        output_norm = output - mean
                    all_outputs.append(output_norm)

            features = torch.cat(all_outputs, dim=1).to(self.device)
            labels = labels.to(self.device).view(-1).long()

            n_samples = features.size(0)
            n_train = int(0.7 * n_samples)
            indices = torch.randperm(n_samples, device=self.device)
            train_features = features[indices[:n_train]]
            train_labels = labels[indices[:n_train]]
            val_features = features[indices[n_train:]]
            val_labels = labels[indices[n_train:]]

            num_features = len(self.formulas)
            classifier = nn.Linear(num_features, self.num_classes).to(self.device)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=0.05)
            criterion = nn.CrossEntropyLoss()

            classifier.train()
            for step in range(n_train_steps):
                optimizer.zero_grad()
                logits = classifier(train_features)
                loss = criterion(logits, train_labels)
                loss.backward()
                optimizer.step()
                if step > 5 and loss.item() < 0.1:
                    break

            classifier.eval()
            with torch.no_grad():
                if len(val_labels) > 0:
                    val_logits = classifier(val_features)
                    predictions = val_logits.argmax(dim=1)
                    accuracy = (predictions == val_labels).float().mean().item()
                else:
                    logits = classifier(features)
                    predictions = logits.argmax(dim=1)
                    accuracy = (predictions == labels).float().mean().item()

            return accuracy, {
                "ensemble_accuracy": accuracy,
                "num_formulas": num_features,
                "formulas": self.formula_strs
            }

        except Exception as e:
            print(f"[Ensemble Error] {e}")
            return 0.0, {"error": str(e)}

    def get_summary(self) -> str:
        if len(self.formulas) == 0:
            return "FeatureBank: Empty"

        simple_count = self._tier_count(True)
        complex_count = self._tier_count(False)
        lines = [f"FeatureBank ({len(self.formulas)}/{self.max_size} | {simple_count}S+{complex_count}C):"]
        for i, (fs, acc, length) in enumerate(zip(self.formula_strs, self.accuracies, self.formula_lengths)):
            tier = "S" if self._is_simple(length) else "C"
            lines.append(f"  [{i+1}][{tier}] {fs} (acc: {acc:.3f}, len: {length})")
        return "\n".join(lines)

    # ==================== Helper Methods ====================

    def _correlation(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x = x.view(-1).float().cpu()
        y = y.view(-1).float().cpu()
        x_mean = x.mean()
        y_mean = y.mean()
        numerator = ((x - x_mean) * (y - y_mean)).sum()
        denominator = torch.sqrt(((x - x_mean) ** 2).sum() * ((y - y_mean) ** 2).sum())
        if denominator < 1e-8:
            return 0.0
        return (numerator / denominator).item()

    def _extract_variables(self, formula_str: str) -> set:
        return set(re.findall(r'z\d+', formula_str))

    def _edit_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]
