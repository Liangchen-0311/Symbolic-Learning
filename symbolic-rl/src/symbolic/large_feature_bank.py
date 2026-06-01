"""
Large Feature Bank — Scheme A: Survival of the Fittest.

Core logic:
  - Bank NOT full → admit if acc > threshold AND Pearson |r| < corr_threshold.
  - Bank FULL     → find worst formula; replace it if new acc > worst acc
                     AND new formula passes the diversity (correlation) gate.

All output vectors are stored on CPU as numpy arrays for fast vectorised
Pearson correlation (matrix multiply).
"""

import os
import json
import numpy as np
import torch
from typing import List, Tuple, Optional


class LargeFeatureBank:
    """Feature bank with dynamic survival-of-the-fittest replacement."""

    def __init__(
        self,
        max_size: int = 1000,
        min_accuracy: float = 0.015,
        correlation_threshold: float = 0.90,
        correlation_threshold_full: float = None,
        num_classes: int = 100,
        device: str = "cuda",
        # Adaptive threshold
        adaptive_threshold: bool = False,
        threshold_warmup_fraction: float = 0.5,
        # Legacy params (kept for constructor compatibility)
        lasso_target: int = 1000,
        l1_lambda: float = 0.0,
        lasso_epochs: int = 100,
    ):
        self.max_size = max_size
        self.min_accuracy = min_accuracy
        self.base_min_accuracy = min_accuracy  # original floor
        self.correlation_threshold = correlation_threshold
        self.correlation_threshold_initial = correlation_threshold
        self.correlation_threshold_full = correlation_threshold_full or correlation_threshold
        self.num_classes = num_classes
        self.device = device
        self.adaptive_threshold = adaptive_threshold
        self.threshold_warmup_fraction = threshold_warmup_fraction
        self.lasso_target = lasso_target
        self.l1_lambda = l1_lambda
        self.lasso_epochs = lasso_epochs

        # Storage — parallel lists, indexed identically
        self.formulas: list = []                      # formula objects
        self.formula_strs: List[str] = []
        self.formula_lengths: List[int] = []
        self.accuracies: List[float] = []
        self.output_vectors: List[Optional[np.ndarray]] = []   # (N,) float32

        # Counters
        self.total_added = 0
        self.total_replaced = 0
        self.total_rejected = 0

        # Legacy (unused in Scheme A but kept so callers don't crash)
        self.lasso_selected_indices = None
        self.final_accuracy = 0.0

    # ------------------------------------------------------------------
    # Size helpers
    # ------------------------------------------------------------------
    def size(self) -> int:
        return len(self.formulas)

    def prune_to_indices(self, keep_indices) -> int:
        """Keep only the formulas at ``keep_indices`` (v3.3 Section 7B reshuffle).

        Filters all parallel lists in lock-step. Returns the number of formulas dropped.
        """
        keep = sorted(set(int(i) for i in keep_indices))
        before = len(self.formulas)
        self.formulas = [self.formulas[i] for i in keep]
        self.formula_strs = [self.formula_strs[i] for i in keep]
        self.formula_lengths = [self.formula_lengths[i] for i in keep]
        self.accuracies = [self.accuracies[i] for i in keep]
        self.output_vectors = [self.output_vectors[i] for i in keep]
        return before - len(self.formulas)

    def is_full(self) -> bool:
        return len(self.formulas) >= self.max_size

    # ------------------------------------------------------------------
    # Adaptive threshold management
    # ------------------------------------------------------------------
    def _update_adaptive_thresholds(self):
        """Raise accuracy threshold and tighten correlation as bank fills."""
        if not self.adaptive_threshold:
            return
        fill_ratio = self.size() / max(1, self.max_size)

        # After warmup fraction, raise min_accuracy to 80% of bank mean
        if fill_ratio >= self.threshold_warmup_fraction and self.accuracies:
            mean_acc = float(np.mean(self.accuracies))
            new_threshold = max(self.base_min_accuracy, mean_acc * 0.8)
            if new_threshold != self.min_accuracy:
                self.min_accuracy = new_threshold

        # Tighten correlation threshold as bank fills past 80%
        if fill_ratio > 0.8:
            # Linearly interpolate from initial to full threshold
            t = (fill_ratio - 0.8) / 0.2  # 0→1 over 80%→100%
            self.correlation_threshold = (
                self.correlation_threshold_initial * (1 - t)
                + self.correlation_threshold_full * t
            )

    # ------------------------------------------------------------------
    # Core — Scheme A admission
    # ------------------------------------------------------------------
    def add_formula(
        self,
        formula,
        formula_str: str,
        length: int,
        accuracy: float,
        output_vector=None,
    ) -> Tuple[bool, str]:
        """
        Attempt to insert *formula* using Scheme A.

        Args:
            formula:       Executable formula object (stored as-is).
            formula_str:   Human-readable RPN string.
            length:        Token count.
            accuracy:      Individual accuracy on the eval batch.
            output_vector: [batch] Tensor or numpy array (CPU).

        Returns:
            (accepted, reason_string)
        """
        # Update adaptive thresholds based on bank fill level
        self._update_adaptive_thresholds()

        # Gate 1: accuracy floor
        if accuracy < self.min_accuracy:
            self.total_rejected += 1
            return False, f"acc {accuracy:.4f} < threshold {self.min_accuracy}"

        # Gate 2: no exact duplicates
        if formula_str in self.formula_strs:
            self.total_rejected += 1
            return False, "duplicate formula"

        # Prepare numpy output for correlation
        out_np = self._to_numpy(output_vector)

        # Gate 3: diversity (Pearson correlation gate)
        if out_np is not None and not self._passes_correlation_check(out_np):
            self.total_rejected += 1
            return False, "too correlated with existing formula"

        # ---- Scheme A decision -------------------------------------------
        if not self.is_full():
            self._insert(formula, formula_str, length, accuracy, out_np)
            self.total_added += 1
            return True, f"added [{self.size()}/{self.max_size}]"

        # Bank FULL → survival of the fittest
        worst_idx = self._find_worst_index()
        worst_acc = self.accuracies[worst_idx]

        if accuracy > worst_acc:
            self._replace(worst_idx, formula, formula_str, length, accuracy, out_np)
            self.total_replaced += 1
            return True, f"replaced worst (acc {worst_acc:.4f} -> {accuracy:.4f})"

        self.total_rejected += 1
        return False, f"acc {accuracy:.4f} <= worst {worst_acc:.4f}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_numpy(v) -> Optional[np.ndarray]:
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().float().numpy()
        return np.asarray(v, dtype=np.float32)

    def _insert(self, formula, formula_str, length, accuracy, out_np):
        self.formulas.append(formula)
        self.formula_strs.append(formula_str)
        self.formula_lengths.append(length)
        self.accuracies.append(accuracy)
        self.output_vectors.append(out_np)

    def _replace(self, idx, formula, formula_str, length, accuracy, out_np):
        self.formulas[idx] = formula
        self.formula_strs[idx] = formula_str
        self.formula_lengths[idx] = length
        self.accuracies[idx] = accuracy
        self.output_vectors[idx] = out_np

    def _find_worst_index(self) -> int:
        return int(np.argmin(self.accuracies))

    # ------------------------------------------------------------------
    # Pearson correlation — vectorised
    # ------------------------------------------------------------------
    def _passes_correlation_check(self, new_out: np.ndarray) -> bool:
        """Return True iff |r(new, existing)| < threshold for ALL existing."""
        valid = [v for v in self.output_vectors if v is not None]
        if len(valid) == 0:
            return True

        new_std = new_out.std()
        if new_std < 1e-10:
            return True  # constant vector — let accuracy gate decide

        existing = np.stack(valid)                                # (K, N)
        N = len(new_out)
        new_norm = (new_out - new_out.mean()) / (new_std + 1e-8) # (N,)

        ex_means = existing.mean(axis=1, keepdims=True)
        ex_stds  = np.maximum(existing.std(axis=1, keepdims=True), 1e-8)
        ex_norm  = (existing - ex_means) / ex_stds               # (K, N)

        corr = np.abs(ex_norm @ new_norm / N)                    # (K,)
        return bool(np.all(corr < self.correlation_threshold))

    def get_max_correlation(self, output_vector) -> float:
        """Max |Pearson r| between *output_vector* and every bank entry."""
        out_np = self._to_numpy(output_vector)
        if out_np is None:
            return 0.0
        valid = [v for v in self.output_vectors if v is not None]
        if len(valid) == 0:
            return 0.0

        new_std = out_np.std()
        if new_std < 1e-10:
            return 0.0

        existing = np.stack(valid)
        N = len(out_np)
        new_norm = (out_np - out_np.mean()) / (new_std + 1e-8)

        ex_means = existing.mean(axis=1, keepdims=True)
        ex_stds  = np.maximum(existing.std(axis=1, keepdims=True), 1e-8)
        ex_norm  = (existing - ex_means) / ex_stds

        corr = np.abs(ex_norm @ new_norm / N)
        return float(corr.max())

    # ------------------------------------------------------------------
    # Legacy compatibility helpers
    # ------------------------------------------------------------------
    def compute_diversity(self, new_output, new_formula_str) -> float:
        """Legacy: 1.0 = maximally diverse, 0.0 = duplicate."""
        if len(self.formulas) == 0:
            return 1.0
        if new_formula_str in self.formula_strs:
            return 0.0
        if new_output is not None:
            return 1.0 - self.get_max_correlation(new_output)
        return 1.0

    def train_lasso_and_prune(self, data_batch=None, labels=None):
        """No-op in Scheme A (kept so callers don't crash)."""
        print(f"[Scheme A] Bank at {self.size()}/{self.max_size} — "
              f"no LASSO pruning needed (l1_lambda={self.l1_lambda})")
        return 0.0, self.size()

    def get_selected_formulas(self) -> List[str]:
        """In Scheme A every formula in the bank is 'selected'."""
        return list(self.formula_strs)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def get_summary(self) -> str:
        accs = np.array(self.accuracies) if self.accuracies else np.array([0.0])
        return (
            f"\n{'='*60}\n"
            f"Feature Bank (Scheme A — Survival of the Fittest)\n"
            f"{'='*60}\n"
            f"  Capacity   : {self.size()}/{self.max_size}\n"
            f"  Acc range  : min={accs.min():.4f}  mean={accs.mean():.4f}  max={accs.max():.4f}\n"
            f"  Threshold  : {self.min_accuracy}   Corr gate: {self.correlation_threshold}\n"
            f"  Added      : {self.total_added}\n"
            f"  Replaced   : {self.total_replaced}\n"
            f"  Rejected   : {self.total_rejected}\n"
            f"{'='*60}"
        )

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, path):
        os.makedirs(path, exist_ok=True)

        meta = {
            'formulas': [
                {'str': s, 'tokens': s.split(), 'length': l, 'accuracy': a}
                for s, l, a in zip(self.formula_strs, self.formula_lengths,
                                   self.accuracies)
            ],
            'max_size': self.max_size,
            'min_accuracy': self.min_accuracy,
            'correlation_threshold': self.correlation_threshold,
            'total_added': self.total_added,
            'total_replaced': self.total_replaced,
            'total_rejected': self.total_rejected,
        }
        with open(os.path.join(path, 'feature_bank.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        valid = [v for v in self.output_vectors if v is not None]
        if valid:
            np.save(os.path.join(path, 'output_vectors.npy'),
                     np.stack(valid))

        print(f"[FeatureBank] Saved {self.size()} formulas to {path}")

    @classmethod
    def load(cls, path, device='cpu'):
        with open(os.path.join(path, 'feature_bank.json')) as f:
            meta = json.load(f)

        bank = cls(
            max_size=meta['max_size'],
            min_accuracy=meta['min_accuracy'],
            correlation_threshold=meta.get('correlation_threshold', 0.90),
            device=device,
        )

        vec_path = os.path.join(path, 'output_vectors.npy')
        has_vec = os.path.exists(vec_path)
        if has_vec:
            vectors = np.load(vec_path)

        for i, entry in enumerate(meta['formulas']):
            bank.formulas.append(None)
            bank.formula_strs.append(entry['str'])
            bank.formula_lengths.append(entry['length'])
            bank.accuracies.append(entry['accuracy'])
            bank.output_vectors.append(
                vectors[i] if (has_vec and i < len(vectors)) else None
            )

        bank.total_added = meta.get('total_added', 0)
        bank.total_replaced = meta.get('total_replaced', 0)
        bank.total_rejected = meta.get('total_rejected', 0)

        print(f"[FeatureBank] Loaded {bank.size()} formulas from {path}")
        return bank
