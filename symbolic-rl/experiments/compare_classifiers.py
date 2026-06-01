"""
Classifier comparison harness (v3.3 Section 6F).

On a fixed feature matrix (e.g. CIFAR-10 Layer-1+Layer-2 features) with identical
train/test splits and identical normalization, compare the interpretable backends
against a black-box reference, and run a HistGB budget sweep.

Answers the two report questions:
  1. How much does HistGB+MI+balanced beat linear on our features?
  2. How much accuracy does the delivered (interpretable) model leave vs the black-box?

Usage:
    python experiments/compare_classifiers.py --features feats.npz          # X, y arrays
    python experiments/compare_classifiers.py --synthetic                   # smoke test

``feats.npz`` must contain arrays ``X`` [N, D] and ``y`` [N]; optionally ``feature_names``.
The harness instantiates the GBDT backends *directly* (bypassing the make_classifier
hierarchical guard) because this is an explicit small-class benchmark, not the delivered
1000-way pipeline.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.classifiers import (
    HistGBClassifier, LinearClassifier, EBMClassifier, ReferenceGBDTClassifier,
    normalize_features,
)


def _split(X, y, test_frac=0.3, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    cut = int((1 - test_frac) * len(y))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], y[tr], X[te], y[te]


def _eval(clf, Xtr, ytr, Xte, yte):
    t0 = time.time()
    clf.fit(Xtr, ytr)
    acc = clf.score(Xte, yte)
    rep = clf.interpretability_report()
    return acc, time.time() - t0, rep


def run_comparison(X, y, feature_names=None, seed=42):
    n_classes = int(np.max(y)) + 1
    Xtr, ytr, Xte, yte = _split(X, y, seed=seed)
    Xtr_n, Xte_n = normalize_features(Xtr, Xte)
    D = X.shape[1]
    K = min(400, D)

    rows = []

    # Linear (L1 selection) — reference / regression guard
    lin = LinearClassifier(n_classes, feature_names=feature_names,
                           feature_selection='l1', K=K if D > K else None)
    acc, dt, rep = _eval(lin, Xtr_n, ytr, Xte_n, yte)
    rows.append(("Linear", "L1", acc, "yes", "yes", dt, "reference / regression guard"))
    linear_acc = acc

    # EBM (MI) — optional middle ground
    try:
        ebm = EBMClassifier(n_classes, feature_names=feature_names, K=K if D > K else None)
        acc, dt, rep = _eval(ebm, Xtr_n, ytr, Xte_n, yte)
        rows.append(("EBM", "MI", acc, "yes", "yes(+inter)", dt, "optional middle ground"))
    except ImportError:
        rows.append(("EBM", "MI", None, "yes", "yes", 0.0, "skipped (interpret not installed)"))

    # HistGB + MI + balanced — PRIMARY
    hg = HistGBClassifier(n_classes, feature_names=feature_names, K=K,
                          max_iter=200, learning_rate=0.05, max_depth=5)
    acc, dt, rep = _eval(hg, Xtr_n, ytr, Xte_n, yte)
    rows.append(("HistGB+MI+balanced (PRIMARY)", "MI", acc, "yes(path)", "no", dt,
                 f"{rep['total_trees']} trees"))
    histgb_acc = acc

    # Reference GBDT (500x8) — black-box ceiling, never delivered
    ref = ReferenceGBDTClassifier(n_classes, feature_names=feature_names, K=K)
    acc, dt, rep = _eval(ref, Xtr_n, ytr, Xte_n, yte)
    rows.append(("Reference GBDT (500x8)", "MI", acc, "NO (ref only)", "no", dt,
                 "accuracy ceiling, never delivered"))
    reference_acc = acc

    # --- budget sweep for HistGB: collaborator settings vs tighter readable budget ---
    sweep = []
    for (mi, lr, md, tag) in [(100, 0.1, 3, "collab-small"), (200, 0.1, 5, "collab-mid"),
                              (300, 0.1, 5, "collab-large"), (100, 0.1, 3, "tight-readable")]:
        c = HistGBClassifier(n_classes, K=K, max_iter=mi, learning_rate=lr, max_depth=md,
                             early_stopping=(tag == "tight-readable"))
        a, dt, rp = _eval(c, Xtr_n, ytr, Xte_n, yte)
        sweep.append((tag, mi, lr, md, a, rp['total_trees']))

    return {
        "n_classes": n_classes, "n_features": D, "K": K,
        "rows": rows, "sweep": sweep,
        "delta_histgb_vs_linear": (histgb_acc - linear_acc) if histgb_acc and linear_acc else None,
        "delta_reference_vs_histgb": (reference_acc - histgb_acc) if reference_acc and histgb_acc else None,
    }


def render(result):
    lines = []
    lines.append(f"\n# Classifier comparison  (n_classes={result['n_classes']}, "
                 f"D={result['n_features']}, K={result['K']})\n")
    lines.append(f"{'Classifier':<32}{'FeatSel':<8}{'TestAcc':<10}{'Interp?':<14}"
                 f"{'ExactAdd?':<12}{'Time(s)':<9}Notes")
    lines.append("-" * 110)
    for name, fs, acc, interp, exact, dt, notes in result['rows']:
        acc_s = f"{acc*100:.2f}%" if acc is not None else "—"
        lines.append(f"{name:<32}{fs:<8}{acc_s:<10}{interp:<14}{exact:<12}{dt:<9.1f}{notes}")
    lines.append("\n## HistGB budget sweep (accuracy vs tree count)")
    lines.append(f"{'tag':<16}{'iter':<6}{'lr':<6}{'depth':<7}{'acc':<10}trees")
    for tag, mi, lr, md, a, trees in result['sweep']:
        acc_s = f"{a*100:.2f}%"
        lines.append(f"{tag:<16}{mi:<6}{lr:<6}{md:<7}{acc_s:<10}{trees}")
    d1 = result['delta_histgb_vs_linear']
    d2 = result['delta_reference_vs_histgb']
    lines.append("\n## Summary deltas")
    if d1 is not None:
        lines.append(f"  HistGB+MI+balanced vs Linear : {d1*100:+.2f}%  (upside of the collaborator's method)")
    if d2 is not None:
        lines.append(f"  Reference black-box vs HistGB : {d2*100:+.2f}%  (cost of staying interpretable)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', type=str, default=None, help='npz with X, y (and optional feature_names)')
    ap.add_argument('--synthetic', action='store_true', help='run on synthetic data (smoke test)')
    ap.add_argument('--out', type=str, default=None, help='optional path to write the rendered table')
    args = ap.parse_args()

    if args.features:
        data = np.load(args.features, allow_pickle=True)
        X, y = data['X'], data['y']
        feature_names = list(data['feature_names']) if 'feature_names' in data else None
    else:
        if not args.synthetic:
            print("No --features given; running --synthetic smoke test.")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=2000, n_features=60, n_informative=25,
                                   n_redundant=20, n_classes=10, random_state=0)
        feature_names = [f"formula_{i}" for i in range(X.shape[1])]

    result = run_comparison(X, y, feature_names=feature_names)
    text = render(result)
    print(text)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(text + "\n")
        print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
