"""
Bank admission gate (v3.3 Section 7A) — DEGENERACY REJECTION ONLY (wide).

Core principle: WIDE admission, STRICT reshuffle. The admission gate must reject only
formulas that are statistically *broken* (near-constant, non-finite, saturated/clamped),
NEVER formulas that are merely *weak* — a healthy-but-weak formula may only be useful in
combination, so it must get in. Discriminative-power judgment is deliberately deferred to
the periodic group-level reshuffle (Section 7B / bank_reshuffle.py).

This separates the three roles statistics play (keep them conceptually distinct):
  - as FEATURES   -> the Section 1A.2 pooling ops
  - as ADMISSION  -> here: variance / finite-ratio / dynamic-range degeneracy only
  - as RESHUFFLE  -> bank_reshuffle.py: MI + importance + Wasserstein redundancy
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class AdmissionConfig:
    finite_min: float = 0.95     # reject if < this fraction of values are finite
    var_min: float = 1e-5        # reject near-constant outputs (generous: only true constants)
    iqr_min: float = 1e-4        # reject saturated/clamped outputs (collapsed dynamic range)
    keep_min_acc: float = 0.002  # optional legacy accuracy floor; set None for degeneracy-only

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        return cls(
            finite_min=float(d.get('finite_min', 0.95)),
            var_min=float(d.get('var_min', 1e-5)),
            iqr_min=float(d.get('iqr_min', 1e-4)),
            keep_min_acc=(None if d.get('keep_min_acc', 0.002) is None
                          else float(d.get('keep_min_acc', 0.002))),
        )


def admission_gate(v, cfg=None, accuracy=None):
    """Decide whether a formula's output vector ``v`` [N] is healthy enough to admit.

    Args:
        v: feature values over the current evaluation batch (torch or numpy, shape [N]).
        cfg: AdmissionConfig (or dict, or None for defaults).
        accuracy: optional univariate accuracy; only used if cfg.keep_min_acc is not None.

    Returns:
        (admit: bool, reason: str). reason in
        {'admit','nonfinite','degenerate_constant','saturated','below_min_acc'}.
    """
    if cfg is None:
        cfg = AdmissionConfig()
    elif isinstance(cfg, dict):
        cfg = AdmissionConfig.from_dict(cfg)

    t = torch.as_tensor(v, dtype=torch.float32).reshape(-1)
    if t.numel() == 0:
        return False, "nonfinite"

    # 1. finite-ratio: reject formulas producing many NaN/inf
    finite_mask = torch.isfinite(t)
    if finite_mask.float().mean().item() < cfg.finite_min:
        return False, "nonfinite"
    t = t[finite_mask]
    if t.numel() < 2:
        return False, "nonfinite"

    # 2. variance floor: reject near-constant (no information) outputs
    if t.var(unbiased=False).item() < cfg.var_min:
        return False, "degenerate_constant"

    # 3. dynamic-range floor: reject saturated/clamped outputs
    iqr = (torch.quantile(t, 0.75) - torch.quantile(t, 0.25)).item()
    if iqr < cfg.iqr_min:
        return False, "saturated"

    # 4. optional legacy accuracy floor (very low; kept for A/B with degeneracy-only gating)
    if cfg.keep_min_acc is not None and accuracy is not None:
        if float(accuracy) < cfg.keep_min_acc:
            return False, "below_min_acc"

    return True, "admit"


class AdmissionTracker:
    """Accumulates admission/rejection reason counts for the final report (Section 7C)."""

    REASONS = ('admit', 'nonfinite', 'degenerate_constant', 'saturated', 'below_min_acc')

    def __init__(self, cfg=None):
        self.cfg = cfg if isinstance(cfg, AdmissionConfig) else AdmissionConfig.from_dict(cfg or {})
        self.counts = {r: 0 for r in self.REASONS}

    def check(self, v, accuracy=None):
        admit, reason = admission_gate(v, self.cfg, accuracy)
        self.counts[reason] = self.counts.get(reason, 0) + 1
        return admit, reason

    def summary(self):
        total = sum(self.counts.values())
        admitted = self.counts.get('admit', 0)
        return {
            "total_seen": total,
            "admitted": admitted,
            "rejected": total - admitted,
            "admit_rate": (admitted / total) if total else 0.0,
            "by_reason": dict(self.counts),
        }
