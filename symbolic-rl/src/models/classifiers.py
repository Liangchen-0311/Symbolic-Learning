"""
Pluggable interpretable, neuron-free classifiers for symbolic features (v3.3 Section 6).

The hard constraint is "no neurons + interpretable", NOT "linear only". This module
exposes a common ``BaseSymbolicClassifier`` interface over four backends:

  - ``HistGBClassifier``        PRIMARY / delivered. MI feature selection +
                                HistGradientBoostingClassifier + class_weight='balanced'.
                                Neuron-free (trees + boosting). Path-level interpretable.
  - ``LinearClassifier``        Interpretability reference + regression guard.
                                Multinomial logistic regression (exact additive decomposition).
  - ``EBMClassifier``           Optional middle ground (InterpretML EBM). Skipped if
                                the ``interpret`` package is unavailable.
  - ``ReferenceGBDTClassifier`` Black-box accuracy ceiling. NEVER delivered.

Section 6.0 — multiclass tree explosion: any GBDT backend trains ``max_iter x n_classes``
trees, so a flat 1000-way GBDT explodes (and is unreadable). The factory therefore
*refuses* a non-reference GBDT unless ``hierarchical.enabled == true`` (Section 3 decomposition).

Feature normalization (standardize -> power-norm -> L2-norm) is provided here as
``normalize_features`` and applied identically before every backend (Section 6G). Tree
models are scale-invariant so it neither helps nor hurts them — it keeps the comparison
controlled.
"""

from __future__ import annotations

import numpy as np

try:  # sklearn is a hard project dependency
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import mutual_info_classif, f_classif
    from sklearn.utils.class_weight import compute_sample_weight
    _SKLEARN_OK = True
except Exception as _e:  # pragma: no cover
    _SKLEARN_OK = False
    _SKLEARN_ERR = _e


# ---------------------------------------------------------------------------
# Shared feature normalization (Section 6G): standardize -> power-norm -> L2
# ---------------------------------------------------------------------------

def normalize_features(X_train, X_test=None, eps=1e-8):
    """Standardize -> signed power-norm (sign(x)*sqrt(|x|)) -> per-sample L2-norm.

    Fit statistics on X_train only. Returns (X_train_n,) or (X_train_n, X_test_n).
    Deterministic; no learned parameters beyond mean/std.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + eps

    def _apply(X):
        X = np.asarray(X, dtype=np.float64)
        Z = (X - mu) / sd                              # standardize
        Z = np.sign(Z) * np.sqrt(np.abs(Z))            # signed power-norm
        n = np.linalg.norm(Z, axis=1, keepdims=True) + eps
        Z = Z / n                                      # per-sample L2-norm
        return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if X_test is None:
        return _apply(X_train)
    return _apply(X_train), _apply(X_test)


# ---------------------------------------------------------------------------
# Mutual-information feature selection on a subsample (Section 6B)
# ---------------------------------------------------------------------------

def select_features_mi(X, y, k, subsample=50000, rng=None):
    """Rank features by mutual information (estimated on a subsample) and return
    the indices of the top-``k`` features. MI captures nonlinear dependence, so it
    keeps features a tree can exploit even when their linear weight is ~0.

    The subsample only affects the *ranking* (top-K selection), so it is robust.
    """
    X = np.asarray(X)
    n, d = X.shape
    k = int(min(k, d))
    rng = rng if rng is not None else np.random.RandomState(42)
    if n > subsample:
        idx = rng.choice(n, size=subsample, replace=False)
        Xs, ys = X[idx], np.asarray(y)[idx]
    else:
        Xs, ys = X, np.asarray(y)
    mi = mutual_info_classif(Xs, ys, random_state=42)
    order = np.argsort(mi)[::-1][:k]
    order = np.sort(order)                 # keep original column order for readability
    return order, mi


def select_features_l1(X, y, k, C=0.1, rng=None):
    """Rank features for a *linear* downstream model. Uses an L1-penalized logistic
    regression's per-feature weight magnitude (falls back to ANOVA F if L1 degenerate).
    Returns indices of the top-``k`` features. Matches the classifier family (Section 6B)."""
    X = np.asarray(X)
    d = X.shape[1]
    k = int(min(k, d))
    try:
        clf = LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=2000)
        # liblinear is binary/ovr; for multiclass use saga (defaults to multinomial)
        if len(np.unique(y)) > 2:
            clf = LogisticRegression(penalty='l1', solver='saga', C=C, max_iter=2000)
        clf.fit(X, y)
        importance = np.abs(clf.coef_).sum(axis=0)
        if not np.any(importance > 0):
            raise ValueError('degenerate L1 importances')
    except Exception:
        importance, _ = f_classif(X, y)
        importance = np.nan_to_num(importance)
    order = np.argsort(importance)[::-1][:k]
    order = np.sort(order)
    return order, importance


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseSymbolicClassifier:
    """Consumes a feature matrix X [N, D] (D symbolic features) and integer labels y [N].

    All backends expose a per-prediction explanation, global feature importance,
    an interpretability report, and accept sample weights. ``feature_names`` (optional)
    are the RPN / plain-English readings used to render explanations.
    """

    def __init__(self, n_classes, feature_names=None, **kwargs):
        self.n_classes = int(n_classes)
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.selected_idx_ = None          # MI/L1 selected column indices (or None = all)
        self.importance_ = None            # selection scores aligned to original D
        self._fitted = False

    # -- helpers -----------------------------------------------------------
    def _feat_name(self, j):
        if self.feature_names is not None and j < len(self.feature_names):
            return self.feature_names[j]
        return f"f{j}"

    def _apply_selection(self, X):
        if self.selected_idx_ is None:
            return np.asarray(X)
        return np.asarray(X)[:, self.selected_idx_]

    # -- interface (override) ---------------------------------------------
    def fit(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None):
        raise NotImplementedError

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        raise NotImplementedError

    def explain(self, x_single):
        raise NotImplementedError

    def global_importance(self):
        """[D] importance aligned to the *original* feature columns."""
        return self.importance_

    def interpretability_report(self):
        raise NotImplementedError

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())


# ---------------------------------------------------------------------------
# HistGB (PRIMARY / delivered)
# ---------------------------------------------------------------------------

class HistGBClassifier(BaseSymbolicClassifier):
    """MI feature selection -> HistGradientBoostingClassifier + class_weight='balanced'.

    Neuron-free (decision trees + additive boosting). Path-level interpretable.
    NOTE total_trees = effective_iters x n_classes — keep n_classes small via the
    WordNet superclass decomposition (Section 3 / 6.0).
    """

    def __init__(self, n_classes, feature_names=None, K=400, max_iter=200,
                 learning_rate=0.05, max_depth=5, l2_regularization=1.0,
                 early_stopping=True, n_iter_no_change=10, validation_fraction=0.1,
                 class_weight='balanced', mi_subsample=50000, random_state=42,
                 feature_selection='mi', **kwargs):
        super().__init__(n_classes, feature_names, **kwargs)
        if not _SKLEARN_OK:
            raise ImportError(f"scikit-learn unavailable: {_SKLEARN_ERR}")
        self.K = K
        self.mi_subsample = mi_subsample
        self.feature_selection = feature_selection
        self.class_weight = class_weight
        self.random_state = random_state
        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter, learning_rate=learning_rate, max_depth=max_depth,
            l2_regularization=l2_regularization, early_stopping=early_stopping,
            n_iter_no_change=n_iter_no_change, validation_fraction=validation_fraction,
            class_weight=class_weight, random_state=random_state,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None):
        X_train = np.asarray(X_train); y_train = np.asarray(y_train)
        rng = np.random.RandomState(self.random_state)
        if self.feature_selection == 'mi':
            self.selected_idx_, self.importance_ = select_features_mi(
                X_train, y_train, self.K, subsample=self.mi_subsample, rng=rng)
        elif self.feature_selection == 'l1':
            self.selected_idx_, self.importance_ = select_features_l1(
                X_train, y_train, self.K, rng=rng)
        else:
            self.selected_idx_ = None
        Xs = self._apply_selection(X_train)
        # class_weight on the estimator already handles imbalance; honour an explicit
        # sample_weight if the caller threads one (Section 6G).
        self.model.fit(Xs, y_train, sample_weight=sample_weight)
        self._fitted = True
        return self

    def predict_proba(self, X):
        proba = self.model.predict_proba(self._apply_selection(X))
        return _align_proba(proba, self.model.classes_, self.n_classes)

    def explain(self, x_single):
        x = np.asarray(x_single).reshape(1, -1)
        proba = self.predict_proba(x)[0]
        pred = int(np.argmax(proba))
        # path-level + global importance: report the top selected features driving the
        # node, with their values. (HistGB is not exact-additive; we report path-level.)
        sel = self.selected_idx_ if self.selected_idx_ is not None else np.arange(x.shape[1])
        scores = self.importance_[sel] if self.importance_ is not None else np.ones(len(sel))
        top = np.argsort(scores)[::-1][:5]
        contributors = [
            {"feature": self._feat_name(int(sel[t])), "value": float(x[0, sel[t]]),
             "mi": float(scores[t])}
            for t in top
        ]
        return {
            "prediction": pred, "confidence": float(proba[pred]),
            "type": "path_level", "exact_additive": False,
            "top_features": contributors,
            "reading": " AND ".join(
                f"({c['feature']} ~ {c['value']:.3f})" for c in contributors
            ) + f"  ->  class {pred}",
        }

    def interpretability_report(self):
        n_cls = self.n_classes
        eff_iters = getattr(self.model, 'n_iter_', self.model.max_iter)
        total_trees = int(eff_iters) * int(n_cls)
        sample = None
        return {
            "backend": "histgb", "exact_additive": False, "interpretable": "path_level",
            "n_selected_features": int(len(self.selected_idx_)) if self.selected_idx_ is not None else None,
            "effective_iters": int(eff_iters), "n_classes": int(n_cls),
            "total_trees": total_trees, "max_depth": self.model.max_depth,
            "note": f"path-level interpretable; total_trees = {eff_iters} iters x {n_cls} classes",
        }


# ---------------------------------------------------------------------------
# Linear (interpretability reference + regression guard)
# ---------------------------------------------------------------------------

class LinearClassifier(BaseSymbolicClassifier):
    """Multinomial logistic regression. Exact additive decomposition: logit(class) =
    sum_j w[class,j] * x_j + b[class]. Interpretability gold standard + regression guard."""

    def __init__(self, n_classes, feature_names=None, C=1.0, max_iter=5000,
                 feature_selection='l1', K=None, random_state=42, **kwargs):
        super().__init__(n_classes, feature_names, **kwargs)
        if not _SKLEARN_OK:
            raise ImportError(f"scikit-learn unavailable: {_SKLEARN_ERR}")
        self.C = C
        self.max_iter = max_iter
        self.feature_selection = feature_selection
        self.K = K
        self.random_state = random_state
        self.model = LogisticRegression(C=C, max_iter=max_iter, solver='lbfgs')

    def fit(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None):
        X_train = np.asarray(X_train); y_train = np.asarray(y_train)
        if self.K is not None and self.feature_selection == 'l1':
            rng = np.random.RandomState(self.random_state)
            self.selected_idx_, self.importance_ = select_features_l1(
                X_train, y_train, self.K, rng=rng)
        else:
            self.selected_idx_ = None
        Xs = self._apply_selection(X_train)
        self.model.fit(Xs, y_train, sample_weight=sample_weight)
        self._fitted = True
        return self

    def predict_proba(self, X):
        proba = self.model.predict_proba(self._apply_selection(X))
        return _align_proba(proba, self.model.classes_, self.n_classes)

    def explain(self, x_single):
        x = np.asarray(x_single).reshape(1, -1)
        xs = self._apply_selection(x)[0]
        sel = self.selected_idx_ if self.selected_idx_ is not None else np.arange(x.shape[1])
        proba = self.predict_proba(x)[0]
        pred = int(np.argmax(proba))
        # exact additive contribution for the predicted class
        cls_row = list(self.model.classes_).index(pred) if pred in self.model.classes_ else 0
        w = self.model.coef_[cls_row]
        contrib = w * xs
        top = np.argsort(np.abs(contrib))[::-1][:5]
        top_features = [
            {"feature": self._feat_name(int(sel[t])), "value": float(xs[t]),
             "weight": float(w[t]), "contribution": float(contrib[t])}
            for t in top
        ]
        return {
            "prediction": pred, "confidence": float(proba[pred]),
            "type": "additive", "exact_additive": True,
            "top_features": top_features,
            "logit_sum": float(contrib.sum() + self.model.intercept_[cls_row]),
            "reading": " + ".join(
                f"{c['contribution']:+.3f}*[{c['feature']}]" for c in top_features
            ) + f"  ->  class {pred}",
        }

    def interpretability_report(self):
        n_params = int(self.model.coef_.size + self.model.intercept_.size)
        return {
            "backend": "linear", "exact_additive": True, "interpretable": "exact_additive",
            "n_parameters": n_params,
            "n_selected_features": int(len(self.selected_idx_)) if self.selected_idx_ is not None else None,
            "note": "prediction = sum of per-feature signed contributions (exact)",
        }


# ---------------------------------------------------------------------------
# EBM (optional middle ground)
# ---------------------------------------------------------------------------

class EBMClassifier(BaseSymbolicClassifier):
    """InterpretML ExplainableBoostingClassifier. Additive (per-feature shape functions
    + a few pairwise interactions) -> exact additive decomposition, models nonlinearity.
    Skipped gracefully if the optional ``interpret`` package is missing."""

    def __init__(self, n_classes, feature_names=None, interactions=10, max_bins=256,
                 feature_selection='mi', K=None, mi_subsample=50000, random_state=42, **kwargs):
        super().__init__(n_classes, feature_names, **kwargs)
        try:
            from interpret.glassbox import ExplainableBoostingClassifier
        except Exception as e:  # pragma: no cover - optional dep
            raise ImportError(f"interpret (EBM) unavailable: {e}")
        self.feature_selection = feature_selection
        self.K = K
        self.mi_subsample = mi_subsample
        self.random_state = random_state
        self.model = ExplainableBoostingClassifier(
            interactions=interactions, max_bins=max_bins, random_state=random_state)

    def fit(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None):
        X_train = np.asarray(X_train); y_train = np.asarray(y_train)
        if self.K is not None:
            rng = np.random.RandomState(self.random_state)
            self.selected_idx_, self.importance_ = select_features_mi(
                X_train, y_train, self.K, subsample=self.mi_subsample, rng=rng)
        else:
            self.selected_idx_ = None
        Xs = self._apply_selection(X_train)
        self.model.fit(Xs, y_train, sample_weight=sample_weight)
        self._fitted = True
        return self

    def predict_proba(self, X):
        proba = self.model.predict_proba(self._apply_selection(X))
        return _align_proba(proba, np.asarray(self.model.classes_), self.n_classes)

    def explain(self, x_single):
        x = np.asarray(x_single).reshape(1, -1)
        proba = self.predict_proba(x)[0]
        pred = int(np.argmax(proba))
        return {"prediction": pred, "confidence": float(proba[pred]),
                "type": "additive", "exact_additive": True,
                "reading": f"EBM additive shape-function sum -> class {pred}"}

    def interpretability_report(self):
        return {"backend": "ebm", "exact_additive": True, "interpretable": "exact_additive",
                "note": "additive shape functions + few pairwise interactions"}


# ---------------------------------------------------------------------------
# Reference GBDT (black-box, reference-only, NEVER delivered)
# ---------------------------------------------------------------------------

class ReferenceGBDTClassifier(BaseSymbolicClassifier):
    """Full-strength HistGB (deep, many iters). Measures the accuracy ceiling only.
    Reported in a separate 'upper-bound reference' row, never as an interpretable result."""

    DELIVERED = False

    def __init__(self, n_classes, feature_names=None, max_iter=500, learning_rate=0.1,
                 max_depth=8, K=None, mi_subsample=50000, random_state=42, **kwargs):
        super().__init__(n_classes, feature_names, **kwargs)
        if not _SKLEARN_OK:
            raise ImportError(f"scikit-learn unavailable: {_SKLEARN_ERR}")
        self.K = K
        self.mi_subsample = mi_subsample
        self.random_state = random_state
        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter, learning_rate=learning_rate, max_depth=max_depth,
            early_stopping=True, n_iter_no_change=10, random_state=random_state)

    def fit(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None):
        X_train = np.asarray(X_train); y_train = np.asarray(y_train)
        if self.K is not None:
            rng = np.random.RandomState(self.random_state)
            self.selected_idx_, self.importance_ = select_features_mi(
                X_train, y_train, self.K, subsample=self.mi_subsample, rng=rng)
        else:
            self.selected_idx_ = None
        self.model.fit(self._apply_selection(X_train), y_train, sample_weight=sample_weight)
        self._fitted = True
        return self

    def predict_proba(self, X):
        proba = self.model.predict_proba(self._apply_selection(X))
        return _align_proba(proba, self.model.classes_, self.n_classes)

    def explain(self, x_single):
        x = np.asarray(x_single).reshape(1, -1)
        pred = int(np.argmax(self.predict_proba(x)[0]))
        return {"prediction": pred, "type": "black_box", "exact_additive": False,
                "reading": "reference-only black-box GBDT (not interpretable, not delivered)"}

    def interpretability_report(self):
        eff_iters = getattr(self.model, 'n_iter_', self.model.max_iter)
        return {"backend": "reference_gbdt", "delivered": False, "interpretable": False,
                "exact_additive": False,
                "total_trees": int(eff_iters) * int(self.n_classes),
                "note": "accuracy ceiling reference; never delivered"}


# ---------------------------------------------------------------------------
# Helpers + factory
# ---------------------------------------------------------------------------

def _align_proba(proba, classes, n_classes):
    """Re-index a backend's [N, n_present] proba onto the full [N, n_classes] space.
    Handles the case where a (subset) training split is missing some class labels."""
    classes = np.asarray(classes)
    if proba.shape[1] == n_classes and np.array_equal(classes, np.arange(n_classes)):
        return proba
    out = np.zeros((proba.shape[0], n_classes), dtype=proba.dtype)
    for col, c in enumerate(classes):
        if 0 <= int(c) < n_classes:
            out[:, int(c)] = proba[:, col]
    return out


_REGISTRY = {
    'histgb': HistGBClassifier,
    'linear': LinearClassifier,
    'ebm': EBMClassifier,
    'reference_gbdt': ReferenceGBDTClassifier,
}

_GBDT_TYPES = {'histgb'}   # reference_gbdt is exempt (never delivered)


def make_classifier(clf_type, n_classes, config=None, feature_names=None,
                    hierarchical_enabled=False, **overrides):
    """Factory: build a ``BaseSymbolicClassifier`` from a config dict.

    Enforces Section 6.0: a delivered GBDT (``histgb``) is refused on a flat multiclass
    problem unless ``hierarchical_enabled`` is True. ``reference_gbdt`` is exempt.
    Raises a clear error for unknown types; missing optional deps (EBM) propagate as
    ImportError so the caller can skip gracefully.
    """
    clf_type = (clf_type or 'histgb').lower()
    if clf_type not in _REGISTRY:
        raise ValueError(f"Unknown classifier type '{clf_type}'. "
                         f"Options: {sorted(_REGISTRY)}")
    if clf_type in _GBDT_TYPES and not hierarchical_enabled:
        raise AssertionError(
            f"classifier.type='{clf_type}' is a delivered GBDT and requires "
            f"hierarchical.enabled=true (Section 6.0: flat multiclass GBDT trains "
            f"max_iter x n_classes trees and explodes). Use the WordNet decomposition, "
            f"or 'reference_gbdt' for a never-delivered flat upper-bound.")

    cfg = dict(config or {})
    # pull the backend-specific sub-block if present
    if clf_type == 'histgb':
        params = {}
        hg = cfg.get('histgb', {})
        # default to the "fine" scaled config; callers override stage explicitly
        stage = overrides.pop('stage', None)
        if stage and stage in hg:
            params.update(hg[stage])
        for key in ('early_stopping', 'n_iter_no_change', 'l2_regularization', 'class_weight'):
            if key in hg:
                params[key] = hg[key]
        params['mi_subsample'] = cfg.get('mi_subsample', 50000)
        params['feature_selection'] = cfg.get('feature_selection', 'mi')
        params.update(overrides)
        return HistGBClassifier(n_classes, feature_names, **params)

    if clf_type == 'linear':
        params = dict(feature_selection=cfg.get('feature_selection', 'l1'))
        params.update(overrides)
        return LinearClassifier(n_classes, feature_names, **params)

    if clf_type == 'ebm':
        params = dict(interactions=cfg.get('ebm_interactions', 10),
                      feature_selection=cfg.get('feature_selection', 'mi'),
                      mi_subsample=cfg.get('mi_subsample', 50000))
        params.update(overrides)
        return EBMClassifier(n_classes, feature_names, **params)

    if clf_type == 'reference_gbdt':
        params = dict(mi_subsample=cfg.get('mi_subsample', 50000))
        params.update(overrides)
        return ReferenceGBDTClassifier(n_classes, feature_names, **params)
