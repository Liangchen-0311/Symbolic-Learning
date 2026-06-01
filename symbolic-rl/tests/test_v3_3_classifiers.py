"""Unit tests for v3.3 Section 6 — pluggable interpretable classifiers."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.classifiers import (
    BaseSymbolicClassifier, HistGBClassifier, LinearClassifier,
    ReferenceGBDTClassifier, make_classifier, normalize_features,
    select_features_mi, select_features_l1, _align_proba,
)


def _toy(n=600, d=30, n_classes=5, seed=0):
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n, n_features=d, n_informative=12,
                               n_redundant=8, n_classes=n_classes, random_state=seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]
    return X[tr], y[tr], X[te], y[te], n_classes


def test_normalize_features():
    Xtr, ytr, Xte, yte, _ = _toy()
    Xtr_n, Xte_n = normalize_features(Xtr, Xte)
    assert Xtr_n.shape == Xtr.shape and Xte_n.shape == Xte.shape
    assert np.isfinite(Xtr_n).all() and np.isfinite(Xte_n).all()
    # per-sample L2 norm ~ 1 (rows are L2-normalized)
    norms = np.linalg.norm(Xtr_n, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_select_features_mi_and_l1():
    Xtr, ytr, _, _, _ = _toy()
    idx, mi = select_features_mi(Xtr, ytr, k=10, subsample=10000)
    assert len(idx) == 10 and len(np.unique(idx)) == 10
    assert mi.shape[0] == Xtr.shape[1]
    idx2, imp = select_features_l1(Xtr, ytr, k=10)
    assert len(idx2) == 10


def test_linear_classifier_regression_and_explain():
    Xtr, ytr, Xte, yte, n_classes = _toy()
    Xtr_n, Xte_n = normalize_features(Xtr, Xte)
    clf = LinearClassifier(n_classes)
    clf.fit(Xtr_n, ytr)
    proba = clf.predict_proba(Xte_n)
    assert proba.shape == (len(yte), n_classes)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    acc = clf.score(Xte_n, yte)
    assert acc > 0.4, f"linear acc too low: {acc}"
    exp = clf.explain(Xte_n[0])
    assert exp['exact_additive'] is True and exp['type'] == 'additive'
    assert len(exp['top_features']) > 0
    rep = clf.interpretability_report()
    assert rep['exact_additive'] is True and rep['n_parameters'] > 0


def test_histgb_classifier_and_report():
    Xtr, ytr, Xte, yte, n_classes = _toy()
    Xtr_n, Xte_n = normalize_features(Xtr, Xte)
    clf = HistGBClassifier(n_classes, K=20, max_iter=60, max_depth=4)
    clf.fit(Xtr_n, ytr)
    proba = clf.predict_proba(Xte_n)
    assert proba.shape == (len(yte), n_classes)
    acc = clf.score(Xte_n, yte)
    assert acc > 0.4, f"histgb acc too low: {acc}"
    exp = clf.explain(Xte_n[0])
    assert exp['exact_additive'] is False and exp['type'] == 'path_level'
    rep = clf.interpretability_report()
    # total_trees = effective_iters x n_classes
    assert rep['total_trees'] == rep['effective_iters'] * n_classes
    assert rep['n_selected_features'] == 20


def test_factory_refuses_flat_gbdt():
    # delivered GBDT must require hierarchical.enabled
    raised = False
    try:
        make_classifier('histgb', n_classes=1000, hierarchical_enabled=False)
    except AssertionError:
        raised = True
    assert raised, "histgb must assert hierarchical.enabled"
    # with hierarchical enabled it builds
    clf = make_classifier('histgb', n_classes=20, hierarchical_enabled=True,
                          config={'histgb': {'fine': {'K': 50, 'max_iter': 50}}}, stage='fine')
    assert isinstance(clf, HistGBClassifier) and clf.K == 50
    # linear has no such restriction
    assert isinstance(make_classifier('linear', n_classes=10), LinearClassifier)
    # reference_gbdt exempt from the hierarchical requirement
    assert isinstance(make_classifier('reference_gbdt', n_classes=1000,
                                      hierarchical_enabled=False), ReferenceGBDTClassifier)


def test_align_proba_subset_classes():
    # backend trained on classes {0,2,4} of a 5-class space
    proba = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
    out = _align_proba(proba, classes=np.array([0, 2, 4]), n_classes=5)
    assert out.shape == (2, 5)
    assert out[0, 1] == 0.0 and out[0, 3] == 0.0
    assert out[0, 0] == 0.2 and out[0, 2] == 0.3 and out[0, 4] == 0.5


def test_reference_gbdt_not_delivered():
    clf = make_classifier('reference_gbdt', n_classes=5, hierarchical_enabled=False)
    Xtr, ytr, Xte, yte, _ = _toy()
    clf.fit(Xtr, ytr)
    assert clf.DELIVERED is False
    rep = clf.interpretability_report()
    assert rep['delivered'] is False and rep['interpretable'] is False


def test_ebm_optional_graceful():
    # EBM should either build (if interpret installed) or raise ImportError cleanly
    try:
        clf = make_classifier('ebm', n_classes=5)
        assert clf is not None
    except ImportError:
        pass  # acceptable: optional dependency missing


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f"ok: {name}")
    print('All v3.3 Section 6 classifier tests passed.')
