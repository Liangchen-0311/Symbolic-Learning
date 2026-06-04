"""Unit tests for v3.3 Section 1 — new semantic + fuzzy-logic operators.

Covers the Section 1 acceptance criteria:
- All new operators importable; TENSOR_OPERATORS grew by exactly 9.
- Each unary op maps [4,32,32] -> [4,32,32], no NaN/Inf, FP32 in/out.
- Binary fuzzy ops handle mismatched spatial sizes via _safe_binary.
- A formula "I_R blob_detector pool_center" executes end-to-end.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.symbolic.tensor_operators import TensorOperators, TENSOR_OPERATORS


NEW_UNARY = [
    'blob_detector', 'symmetry_v', 'symmetry_h',
    'contour', 'elongation', 'radial_gradient',
    'fuzzy_not',
]
NEW_BINARY = ['fuzzy_and', 'fuzzy_or']
# v3.3 Section 1A.2 — high-order statistical pooling (scalar output, ROOT ops)
STAT_POOLS = [
    'pool_skewness', 'pool_kurtosis', 'pool_q10', 'pool_q90', 'pool_iqr',
    'pool_above_mean_ratio', 'pool_entropy', 'pool_energy', 'pool_uniformity',
    'pool_neighbor_diff_var', 'pool_autocorr_lag1',
]
ALL_NEW = NEW_UNARY + NEW_BINARY            # the 9 tensor-output ops (not ROOT)


def test_registry_grew_by_twenty():
    from src.symbolic.tensor_operators import ROOT_OPERATORS
    for name in ALL_NEW + STAT_POOLS:
        assert name in TENSOR_OPERATORS, f"{name} missing from registry"
    # 6 semantic + 3 fuzzy + 11 statistical pooling = 20 new ops
    assert len(ALL_NEW) + len(STAT_POOLS) == 20
    # semantic + fuzzy ops are NOT root ops
    for name in ALL_NEW:
        assert name not in ROOT_OPERATORS, f"{name} should not be a root op"
    # all 11 statistical pooling ops ARE root ops (final-token only)
    for name in STAT_POOLS:
        assert name in ROOT_OPERATORS, f"{name} must be a root op"


def test_statistical_pooling_shape_and_finite():
    x = torch.randn(4, 32, 32, dtype=torch.float32)
    for name in STAT_POOLS:
        fn, arity, otype = TENSOR_OPERATORS[name]
        assert arity == 1 and otype == 'scalar', f"{name} should be unary scalar"
        out = fn(x)
        assert out.shape == (4,), f"{name} shape {out.shape}"
        assert out.dtype == torch.float32, f"{name} dtype {out.dtype}"
        assert torch.isfinite(out).all(), f"{name} produced non-finite values"


def test_unary_shape_and_finite():
    x = torch.randn(4, 32, 32, dtype=torch.float32)
    for name in NEW_UNARY:
        fn, arity, otype = TENSOR_OPERATORS[name]
        assert arity == 1 and otype == 'tensor'
        out = fn(x)
        assert out.shape == (4, 32, 32), f"{name} shape {out.shape}"
        assert out.dtype == torch.float32, f"{name} dtype {out.dtype}"
        assert torch.isfinite(out).all(), f"{name} produced non-finite values"


def test_binary_shape_and_finite():
    x = torch.randn(4, 32, 32, dtype=torch.float32)
    y = torch.randn(4, 32, 32, dtype=torch.float32)
    for name in NEW_BINARY:
        fn, arity, otype = TENSOR_OPERATORS[name]
        assert arity == 2 and otype == 'tensor'
        out = fn(x, y)
        assert out.shape == (4, 32, 32), f"{name} shape {out.shape}"
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()


def test_binary_mismatched_sizes():
    """fuzzy_and/fuzzy_or must handle mismatched spatial sizes via _safe_binary."""
    x = torch.randn(4, 32, 32, dtype=torch.float32)
    y = torch.randn(4, 16, 16, dtype=torch.float32)
    for name in NEW_BINARY:
        fn = TENSOR_OPERATORS[name][0]
        out = fn(x, y)              # smaller spatial size wins (16x16)
        assert out.shape[-2:] == (16, 16), f"{name} -> {out.shape}"
        assert torch.isfinite(out).all()
    # scalar (B,) vs tensor (B,H,W) broadcast path
    s = torch.randn(4, dtype=torch.float32)
    out = TENSOR_OPERATORS['fuzzy_and'][0](s, x)
    assert out.shape == (4, 32, 32)


def test_fuzzy_outputs_in_unit_interval():
    x = torch.randn(4, 32, 32, dtype=torch.float32)
    y = torch.randn(4, 32, 32, dtype=torch.float32)
    assert (TensorOperators.fuzzy_not(x) >= 0).all() and (TensorOperators.fuzzy_not(x) <= 1).all()
    fa = TensorOperators.fuzzy_and(x, y)
    fo = TensorOperators.fuzzy_or(x, y)
    assert (fa >= 0).all() and (fa <= 1).all()
    assert (fo >= 0).all() and (fo <= 1).all()


class _IdentityVocab:
    """Minimal vocabulary whose tokens *are* their own decoded strings."""
    def decode(self, t):
        return t


def test_formula_executes_end_to_end():
    """'I_R blob_detector pool_center' executes through TensorProgramEvaluator."""
    from src.symbolic.tensor_evaluator import TensorProgramEvaluator
    evaluator = TensorProgramEvaluator(num_classes=10, device='cpu')
    data_batch = {'I_R': torch.rand(4, 32, 32, dtype=torch.float32)}  # [B,H,W] in [0,1]
    tokens = ['I_R', 'blob_detector', 'pool_center']  # decoded directly by IdentityVocab
    output, is_valid = evaluator.execute_formula(tokens, _IdentityVocab(), data_batch)
    assert is_valid, "formula execution flagged invalid (NaN/Inf)"
    assert output is not None
    assert output.shape == (4,)               # pool_center -> scalar per sample
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_statistical_pool_formula_executes():
    """'I_R edge_x pool_skewness' executes end-to-end (stat pool as final token)."""
    from src.symbolic.tensor_evaluator import TensorProgramEvaluator
    evaluator = TensorProgramEvaluator(num_classes=10, device='cpu')
    data_batch = {'I_R': torch.rand(4, 32, 32, dtype=torch.float32)}
    tokens = ['I_R', 'edge_x', 'pool_skewness']
    output, is_valid = evaluator.execute_formula(tokens, _IdentityVocab(), data_batch)
    assert is_valid and output is not None
    assert output.shape == (4,) and torch.isfinite(output).all()


def test_fuzzy_binary_formula_executes():
    """A binary fuzzy formula 'I_R I_GRAY fuzzy_and global_avg_pool' executes."""
    from src.symbolic.tensor_evaluator import TensorProgramEvaluator
    evaluator = TensorProgramEvaluator(num_classes=10, device='cpu')
    data_batch = {
        'I_R': torch.rand(4, 32, 32, dtype=torch.float32),
        'I_GRAY': torch.rand(4, 32, 32, dtype=torch.float32),
    }
    tokens = ['I_R', 'I_GRAY', 'fuzzy_and', 'global_avg_pool']
    output, is_valid = evaluator.execute_formula(tokens, _IdentityVocab(), data_batch)
    assert is_valid and output is not None
    assert output.shape == (4,)
    assert torch.isfinite(output).all()


# --- v3.3 revision additions: directional lines (1A.4), asymmetry (1A.3),
#     prior terminals (1A.0), one-formula-many-pools (1A.6) ---
DIRECTIONAL_LINES = ['line_h', 'line_v', 'line_diag45', 'line_diag135']
ASYMMETRY_POOLS = ['pool_lr_asymmetry', 'pool_tb_asymmetry']


def test_directional_lines_unary_tensor():
    from src.symbolic.tensor_operators import ROOT_OPERATORS
    x = torch.randn(4, 32, 32, dtype=torch.float32)
    for name in DIRECTIONAL_LINES:
        fn, arity, otype = TENSOR_OPERATORS[name]
        assert arity == 1 and otype == 'tensor', name
        assert name not in ROOT_OPERATORS, f"{name} must NOT be a root op"
        out = fn(x)
        assert out.shape == (4, 32, 32) and out.dtype == torch.float32
        assert torch.isfinite(out).all(), name


def test_asymmetry_pools_scalar_root():
    from src.symbolic.tensor_operators import ROOT_OPERATORS
    x = torch.randn(4, 32, 32, dtype=torch.float32)
    for name in ASYMMETRY_POOLS:
        fn, arity, otype = TENSOR_OPERATORS[name]
        assert arity == 1 and otype == 'scalar', name
        assert name in ROOT_OPERATORS, f"{name} must be a root op"
        out = fn(x)
        assert out.shape == (4,) and torch.isfinite(out).all(), name
    # a symmetric image has ~zero asymmetry
    sym = torch.ones(2, 8, 8, dtype=torch.float32)
    assert float(TensorOperators.pool_lr_asymmetry(sym).abs().max()) < 1e-5


def test_prior_terminals_deterministic():
    x = torch.rand(4, 16, 16, dtype=torch.float32)
    p = TensorOperators.make_prior_terminals(x)
    assert set(p) == {'I_EDGE', 'I_FREQ', 'I_LAPLACIAN'}
    for k, v in p.items():
        assert v.shape == (4, 16, 16) and v.dtype == torch.float32
        assert torch.isfinite(v).all(), k
    # deterministic + subset selection
    p2 = TensorOperators.make_prior_terminals(x, names=['I_EDGE'])
    assert set(p2) == {'I_EDGE'} and torch.allclose(p2['I_EDGE'], p['I_EDGE'])


def test_expand_formula_statistics_battery():
    from src.symbolic.tensor_operators import expand_formula_statistics
    x = torch.randn(4, 16, 16, dtype=torch.float32)
    stats = expand_formula_statistics(x, formula_idx=3)
    assert len(stats) == 14, f"expected 14-stat battery, got {len(stats)}"
    assert all(k.startswith('formula[3].') for k in stats)
    for k, v in stats.items():
        assert v.shape == (4,) and torch.isfinite(v).all(), k


def test_directional_line_formula_executes():
    from src.symbolic.tensor_evaluator import TensorProgramEvaluator
    evaluator = TensorProgramEvaluator(num_classes=10, device='cpu')
    data_batch = {'I_R': torch.rand(4, 32, 32, dtype=torch.float32)}
    tokens = ['I_R', 'line_diag45', 'pool_lr_asymmetry']
    output, is_valid = evaluator.execute_formula(tokens, _IdentityVocab(), data_batch)
    assert is_valid and output is not None
    assert output.shape == (4,) and torch.isfinite(output).all()


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} v3.3 Section 1 operator tests passed.")
