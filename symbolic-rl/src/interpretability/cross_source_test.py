"""v3.3 Section 8.2b — cross-source test.

If the data has source/site/device metadata, train on one source and test on another. A
large drop reveals source-shortcut dependence (the model learned the acquisition device,
not the object). For the fracture study this was implicitly passed via same-source data;
this makes it an explicit, optional harness for any dataset with source labels.

Operates on an already-extracted feature matrix (Section 4D features), so it is
classifier-agnostic and deterministic. NO external model.
"""

import numpy as np


def cross_source_eval(X, y, source, classifier_factory, train_source=None, test_source=None):
    """Train on one source, test on another; compare to a within-source baseline.

    Args:
        X: [N, D] feature matrix (already normalized as the classifier expects).
        y: [N] int labels.
        source: [N] source/site/device id per sample (any hashable).
        classifier_factory: zero-arg callable -> a fresh classifier with .fit/.score
                             (e.g. lambda: LinearClassifier(n_classes)).
        train_source / test_source: which sources to use. If None, the two most common
                             distinct sources are chosen automatically.
    Returns:
        dict with cross-source accuracy, a within-train-source baseline (random split),
        the gap, and a shortcut verdict. Returns {'skipped': reason} if <2 sources.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    source = np.asarray(source)
    uniq, counts = np.unique(source, return_counts=True)
    if uniq.shape[0] < 2:
        return {'skipped': 'need >=2 distinct sources for a cross-source test'}

    if train_source is None or test_source is None:
        order = uniq[np.argsort(-counts)]
        train_source, test_source = order[0], order[1]

    tr = source == train_source
    te = source == test_source
    if tr.sum() < 10 or te.sum() < 10:
        return {'skipped': f'too few samples in chosen sources ({tr.sum()},{te.sum()})'}

    # Cross-source: fit on train_source, test on test_source.
    clf = classifier_factory()
    clf.fit(X[tr], y[tr])
    cross_acc = float(clf.score(X[te], y[te]))

    # Within-source baseline: random 80/20 split of train_source only.
    idx = np.where(tr)[0]
    rng = np.random.RandomState(0)
    rng.shuffle(idx)
    cut = max(1, int(0.8 * len(idx)))
    a, b = idx[:cut], idx[cut:]
    clf2 = classifier_factory()
    clf2.fit(X[a], y[a])
    within_acc = float(clf2.score(X[b], y[b])) if len(b) else float('nan')

    gap = within_acc - cross_acc if within_acc == within_acc else float('nan')
    verdict = ('robust across sources (small drop)' if (gap == gap and gap < 0.10) else
               'WARNING: large cross-source drop — likely source/device shortcut'
               if gap == gap else 'inconclusive')
    return {
        'train_source': str(train_source), 'test_source': str(test_source),
        'cross_source_acc': round(cross_acc, 4),
        'within_source_acc': round(within_acc, 4) if within_acc == within_acc else None,
        'gap': round(gap, 4) if gap == gap else None,
        'verdict': verdict,
    }
