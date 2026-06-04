"""Unit tests for v3.3 Section 8 — interpretability & trust-verification tools.

Covers the Section 8 acceptance criteria:
- formula_to_text is purely rule-based (no LLM / external data); every channel/operator
  token has a phrase mapping; nested phrasing for compound formulas.
- probe images render top-k/bottom-k activators for an arbitrary formula, including a
  long (12+-token) unreadable one.
- occlusion_test / cross_source_test run and return interpretable verdicts.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.symbolic.tensor_operators import TENSOR_OPERATORS, ROOT_OPERATORS, TensorOperators
from src.interpretability.formula_to_text import (
    formula_to_text, TOKEN_PHRASES, UNARY_PHRASES, BINARY_PHRASES,
    POOL_PHRASES, TERMINAL_PHRASES)
from src.interpretability.probe_images import (
    formula_scalar_values, top_bottom_activators)
from src.interpretability.occlusion_test import region_mask, occlusion_curve
from src.interpretability.cross_source_test import cross_source_eval


def test_formula_to_text_total_coverage():
    """Acceptance: every operator token has an explicit phrase mapping, and translation
    yields a non-empty deterministic string for each."""
    op_phrase_keys = set(UNARY_PHRASES) | set(BINARY_PHRASES) | set(POOL_PHRASES)
    missing = [t for t in TENSOR_OPERATORS if t not in op_phrase_keys]
    assert not missing, f"operators without an explicit phrase mapping: {missing}"
    # Every operator translates to a non-empty string.
    for tok in TENSOR_OPERATORS:
        _, arity, _ = TENSOR_OPERATORS[tok]
        rpn = ' '.join(['I_R'] * max(arity, 1) + [tok])
        txt = formula_to_text(rpn)
        assert isinstance(txt, str) and len(txt) > 0
    # Every standard terminal (incl. prior terminals) has an explicit mapping.
    for t in ['I_R', 'I_G', 'I_B', 'I_GRAY', 'I_EDGE', 'I_FREQ', 'I_LAPLACIAN']:
        assert t in TERMINAL_PHRASES, f"terminal {t} unmapped"


def test_formula_to_text_nested():
    txt = formula_to_text("I_R edge_x blur pool_center")
    assert 'red channel' in txt and 'central region' in txt
    # a long compound formula still yields nested phrasing
    long_f = ("I_R edge_x I_G blur subtract lbp_like flip_h fuzzy_and "
              "normalize pool_skewness")
    txt2 = formula_to_text(long_f)
    assert 'skewness' in txt2 and 'AND' in txt2


def test_formula_to_text_no_external_imports():
    """Hard Constraint #5: the translator imports no LLM / external-knowledge package."""
    import importlib
    m = importlib.import_module('src.interpretability.formula_to_text')
    src = open(m.__file__).read()
    for forbidden in ['openai', 'transformers', 'anthropic', 'requests', 'urllib', 'clip', 'torch.hub']:
        assert forbidden not in src, f"forbidden dependency {forbidden} in formula_to_text"


def _synth_images(n=40, size=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, size, size, generator=g)


def test_probe_top_bottom_long_formula():
    imgs = _synth_images()
    # a 12+-token "unreadable" formula
    long_f = ("I_R edge_x blur I_G lbp_like multiply normalize abs "
              "I_B subtract sigmoid pool_q90")
    assert len(long_f.split()) >= 12
    act = top_bottom_activators(long_f, imgs, k=6)
    assert act['top_idx'].shape[0] == 6 and act['bottom_idx'].shape[0] == 6
    # top activations >= bottom activations
    assert float(act['top_val'].min()) >= float(act['bottom_val'].max()) - 1e-4
    vals = formula_scalar_values(long_f, imgs)
    assert vals.shape == (40,) and torch.isfinite(vals).all()


def test_region_mask_shapes():
    imgs = _synth_images(n=8, size=16)
    m_center = region_mask(imgs, 0.5, mode='center')
    m_bg = region_mask(imgs, 0.5, mode='background')
    assert m_center.shape == imgs.shape and m_bg.shape == imgs.shape
    # center mask changes the middle; background mask changes the periphery
    assert not torch.allclose(m_center[:, :, 4:12, 4:12], imgs[:, :, 4:12, 4:12])
    assert torch.allclose(m_bg[:, :, 4:12, 4:12], imgs[:, :, 4:12, 4:12], atol=1e-5)


class _DummyClf:
    """Tiny stand-in classifier: predicts by sign of feature 0; .score for occlusion test."""
    def fit(self, X, y):
        self._thr = np.median(X[:, 0]); self._y = y
        return self
    def score(self, X, y):
        pred = (X[:, 0] > self._thr).astype(int)
        return float((pred == (np.asarray(y) > 0)).mean())


def test_occlusion_curve_runs():
    imgs = _synth_images(n=60, size=16)
    # label = is the center bright? (object-in-center signal)
    center = imgs[:, :, 4:12, 4:12].mean(dim=(1, 2, 3))
    y = (center > center.median()).long()

    def feature_fn(images):
        return images[:, :, 4:12, 4:12].mean(dim=(1, 2, 3), keepdim=True).numpy()

    clf = _DummyClf().fit(feature_fn(imgs), y.numpy())
    res = occlusion_curve(imgs, y, feature_fn, clf,
                          fractions=(0.0, 0.5, 0.75), mode='center')
    assert res['baseline'] > 0 and len(res['accuracy']) == 3
    assert 'verdict' in res


def test_cross_source_eval():
    rng = np.random.RandomState(0)
    X = rng.randn(120, 5)
    y = (X[:, 0] > 0).astype(int)
    source = np.array([0] * 60 + [1] * 60)

    def factory():
        return _DummyClf()

    res = cross_source_eval(X, y, source, factory)
    assert 'cross_source_acc' in res and 'verdict' in res
    # single-source -> skipped
    res2 = cross_source_eval(X, y, np.zeros(120), factory)
    assert res2.get('skipped')


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} Section 8 interpretability tests passed.")
