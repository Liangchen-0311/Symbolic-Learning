"""Unit tests for v3.3 Section 3 — WordNet hierarchy + hierarchical classifier."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.symbolic.wordnet_hierarchy import (
    group_by_ancestor, balance_groups, build_superclasses,
    HierarchyInfo, save_hierarchy, load_hierarchy, _balanced_fallback,
)
from src.models.hierarchical_classifier import HierarchicalClassifier


# --------------------------------------------------------------------------
# Grouping core (synthetic ancestor fn — no nltk needed)
# --------------------------------------------------------------------------

def _synthetic_ancestor_fn(n_per_group=20):
    """leaf i -> path [root, g{i//n_per_group}, leaf]; depth 1 yields the g-groups."""
    def fn(w):
        i = int(w[1:])
        return ['root', f"g{i // n_per_group}", w]
    return fn


def test_group_by_ancestor():
    wnids = [f"n{i:08d}" for i in range(60)]
    fn = _synthetic_ancestor_fn(20)
    groups = group_by_ancestor(wnids, fn, depth=1)
    assert len(groups) == 3
    assert all(len(v) == 20 for v in groups.values())


def test_balance_groups_merge_and_split():
    # one tiny group (merged to misc), one oversized (split)
    groups = {
        'big': [f"n{i:08d}" for i in range(300)],     # > max_size 130 -> split
        'ok': [f"n{i:08d}" for i in range(300, 360)], # 60 -> kept
        'tiny': [f"n{i:08d}" for i in range(360, 365)],  # 5 -> misc
    }
    final = balance_groups(groups, min_size=15, max_size=130)
    sizes = {k: len(v) for k, v in final.items()}
    assert all(s <= 130 for s in sizes.values())
    # big was split into >=3 chunks
    assert sum(1 for k in final if k.startswith('big__')) >= 3
    # all members preserved
    total = sum(len(v) for v in final.values())
    assert total == 300 + 60 + 5


def test_build_superclasses_structural():
    wnids = [f"n{i:08d}" for i in range(60)]
    fn = _synthetic_ancestor_fn(20)
    h = build_superclasses(wnids, target_groups=3, min_size=15, max_size=130,
                           ancestor_path_fn=fn)
    info = HierarchyInfo(h)
    assert info.n_superclasses == 3
    ok, msgs = info.validate(min_size=15, max_size=130)
    assert ok, msgs
    # every class mapped exactly once
    assert len(info.class_to_superclass) == 60


def test_fallback_partition_1000():
    wnids = [f"n{i:08d}" for i in range(1000)]
    h = build_superclasses(wnids, target_groups=20, use_wordnet=False)
    info = HierarchyInfo(h)
    assert info.n_superclasses == 20
    ok, msgs = info.validate(min_size=15, max_size=130)
    assert ok, msgs
    assert len(info.class_to_superclass) == 1000
    # 1000/20 = 50 per group -> within [15,130]
    assert all(15 <= len(c) <= 130 for c in info.classes_of.values())


def test_save_load_roundtrip():
    import tempfile
    wnids = [f"n{i:08d}" for i in range(60)]
    h = build_superclasses(wnids, target_groups=3, ancestor_path_fn=_synthetic_ancestor_fn(20))
    p = os.path.join(tempfile.mkdtemp(), 'superclasses.json')
    save_hierarchy(h, p)
    h2 = load_hierarchy(p)
    assert h2['meta']['n_superclasses'] == 3


# --------------------------------------------------------------------------
# Hierarchical classifier (soft-cascade) end-to-end on synthetic features
# --------------------------------------------------------------------------

def _toy_hierarchy(n_classes=12, n_super=3):
    per = n_classes // n_super
    superclasses = {}
    class_to_superclass = {}
    for s in range(n_super):
        classes = list(range(s * per, (s + 1) * per))
        superclasses[f"g{s}"] = {"id": s, "classes": classes}
        for c in classes:
            class_to_superclass[str(c)] = s
    h = {"superclasses": superclasses, "class_to_superclass": class_to_superclass,
         "meta": {"method": "synthetic", "n_classes": n_classes, "n_superclasses": n_super}}
    return HierarchyInfo(h)


def _toy_features(n_classes=12, n=900, d=40, seed=0):
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n, n_features=d, n_informative=20,
                               n_redundant=10, n_classes=n_classes,
                               n_clusters_per_class=1, random_state=seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n); cut = int(0.7 * n)
    return X[idx[:cut]], y[idx[:cut]], X[idx[cut:]], y[idx[cut:]]


def _fit_predict(soft_cascade):
    info = _toy_hierarchy(12, 3)
    Xtr, ytr, Xte, yte = _toy_features(12)
    cfg = {'histgb': {'coarse': {'K': 20, 'max_iter': 40, 'max_depth': 3},
                      'fine': {'K': 20, 'max_iter': 40, 'max_depth': 3},
                      'early_stopping': False}}
    clf = HierarchicalClassifier(info, config=cfg, coarse_type='linear',
                                 fine_type='histgb', soft_cascade=soft_cascade)
    clf.fit(Xtr, ytr, Xte, yte)
    pred = clf.predict(Xte)
    return clf, pred, Xte, yte


def test_hierarchical_soft_cascade_valid():
    clf, pred, Xte, yte = _fit_predict(soft_cascade=True)
    # valid 12-way predictions for every sample
    assert pred.shape == yte.shape
    assert pred.min() >= 0 and pred.max() < 12
    # beats chance (1/12 ~ 0.083)
    acc = float((pred == yte).mean())
    assert acc > 0.25, f"hierarchical acc too low: {acc}"
    # coarse router reports a valid superclass top-1 in [0,1]
    cs = clf.coarse_score(Xte, yte)
    assert 0.0 <= cs <= 1.0


def test_hierarchical_hard_cascade_valid():
    clf, pred, Xte, yte = _fit_predict(soft_cascade=False)
    assert pred.min() >= 0 and pred.max() < 12


def test_no_neurons_in_report():
    clf, _, _, _ = _fit_predict(soft_cascade=True)
    rep = clf.interpretability_report()
    assert rep['coarse']['backend'] == 'linear'      # exact-additive coarse router
    assert rep['fine_type'] == 'histgb'              # neuron-free trees
    assert rep['n_fine_classifiers'] == 3


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f"ok: {name}")
    print('All v3.3 Section 3 hierarchy tests passed.')
