"""Unit tests for v3.3 Section 7 — statistical gating (admission + reshuffle)."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.symbolic.bank_admission import (
    admission_gate, AdmissionConfig, AdmissionTracker,
)
from src.symbolic.bank_reshuffle import (
    ReshuffleConfig, reshuffle, wasserstein_dedup, mutual_info_per_feature,
    lasso_importance,
)


# --------------------------------------------------------------------------
# 7A — admission gate (degeneracy rejection only; wide)
# --------------------------------------------------------------------------

def test_admission_admits_healthy_but_weak():
    # a healthy, well-spread but weak feature must be ADMITTED (weakness != rejection)
    v = torch.randn(500)
    admit, reason = admission_gate(v, AdmissionConfig(keep_min_acc=None))
    assert admit and reason == 'admit'


def test_admission_rejects_constant():
    v = torch.full((500,), 3.14)
    admit, reason = admission_gate(v)
    assert not admit and reason == 'degenerate_constant'


def test_admission_rejects_nonfinite():
    v = torch.full((500,), float('nan'))
    admit, reason = admission_gate(v)
    assert not admit and reason == 'nonfinite'


def test_admission_rejects_saturated():
    # almost all identical (collapsed dynamic range) -> tiny IQR but nonzero variance
    v = torch.cat([torch.full((498,), 1.0), torch.tensor([60000.0, -60000.0])])
    admit, reason = admission_gate(v, AdmissionConfig(var_min=0.0))  # bypass var floor
    assert not admit and reason == 'saturated'


def test_admission_legacy_acc_floor():
    v = torch.randn(500)
    admit, reason = admission_gate(v, AdmissionConfig(keep_min_acc=0.5), accuracy=0.1)
    assert not admit and reason == 'below_min_acc'
    admit2, _ = admission_gate(v, AdmissionConfig(keep_min_acc=0.5), accuracy=0.9)
    assert admit2


def test_admission_tracker_counts():
    tr = AdmissionTracker(AdmissionConfig(keep_min_acc=None))
    tr.check(torch.randn(100))                 # admit
    tr.check(torch.full((100,), 2.0))          # constant
    tr.check(torch.full((100,), float('inf')))  # nonfinite
    s = tr.summary()
    assert s['total_seen'] == 3 and s['admitted'] == 1
    assert s['by_reason']['degenerate_constant'] == 1
    assert s['by_reason']['nonfinite'] == 1


# --------------------------------------------------------------------------
# 7B — reshuffle building blocks
# --------------------------------------------------------------------------

def _bank(n=400, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 4, size=n)
    cols = []
    names = []
    # 3 informative columns (class-dependent mean) -> high MI
    for k in range(3):
        cols.append(y * (1.0 + 0.3 * k) + rng.randn(n) * 0.4)
        names.append(f"info{k}")
    # 3 pure-noise columns -> ~0 MI
    for k in range(3):
        cols.append(rng.randn(n))
        names.append(f"noise{k}")
    # a near-duplicate of info0 in the 0.70-0.92 corr band
    cols.append(cols[0] * 0.85 + rng.randn(n) * 0.45)
    names.append("dup_info0")
    X = np.stack(cols, axis=1)
    return X, y, names


def test_mutual_info_and_importance():
    X, y, _ = _bank()
    mi = mutual_info_per_feature(X, y, subsample=1000)
    # informative cols have higher MI than noise cols
    assert mi[:3].mean() > mi[3:6].mean()
    imp = lasso_importance(X, y)
    assert imp.shape[0] == X.shape[1]


def test_wasserstein_dedup_drops_redundant():
    rng = np.random.RandomState(1)
    base = rng.randn(500)
    # c1 highly-but-not-perfectly correlated w/ c0 (corr ~0.8), similar standardized dist
    c0 = base
    c1 = 0.8 * base + 0.6 * rng.randn(500)
    c2 = rng.rand(500)                          # independent (corr~0) -> skipped by band
    X = np.stack([c0, c1, c2], axis=1)
    kept = wasserstein_dedup(X, corr_band=(0.70, 0.92), w_min=0.2)
    assert len(kept) < 3                        # one of the correlated pair dropped
    assert 2 in kept                            # independent column survives


def test_reshuffle_warmup_skips():
    X, y, _ = _bank()
    cfg = ReshuffleConfig(warmup_iters=500)
    kept, log = reshuffle(X, y, cfg, iteration=100)   # before warmup
    assert kept == list(range(X.shape[1])) and log['warmup_skipped']


def test_reshuffle_prunes_noise_keeps_info():
    X, y, names = _bank()
    cfg = ReshuffleConfig(warmup_iters=0, mi_floor=0.01, mi_subsample=1000,
                          imp_floor=1e-6, min_bank_size=2, wasserstein_min=0.15)
    kept, log = reshuffle(X, y, cfg, iteration=100)
    kept_names = {names[i] for i in kept}
    # noise columns pruned by the MI floor
    assert log['dropped_low_mi'] >= 2
    # at least one informative column retained
    assert any(n.startswith('info') for n in kept_names)
    assert log['n_out'] <= log['n_in']


def test_reshuffle_min_bank_size_floor():
    X, y, _ = _bank()
    # force aggressive pruning, but min_bank_size floor restores up to 5
    cfg = ReshuffleConfig(warmup_iters=0, mi_floor=10.0, mi_subsample=1000,
                          min_bank_size=5)
    kept, log = reshuffle(X, y, cfg, iteration=100)
    assert len(kept) >= 5                        # floor honored (never prune below)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f"ok: {name}")
    print('All v3.3 Section 7 gating tests passed.')
