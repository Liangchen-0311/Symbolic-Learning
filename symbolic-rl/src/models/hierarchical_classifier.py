"""
Two-stage hierarchical classifier with soft-cascade inference (v3.3 Section 3B).

Coarse: one interpretable classifier over ~20 superclasses (default: linear — 20-way
        routing is easy and benefits from the strongest exact-additive interpretability).
Fine:   one interpretable classifier per superclass over its ~50 classes, trained only on
        that superclass's image subset (default: HistGB + MI + class_weight='balanced' —
        intra-superclass separation is the harder problem and benefits from nonlinearity).

Soft-cascade inference (robust to coarse errors): combine coarse routing probability with
the fine log-probability for every class, so a wrong-but-uncertain coarse decision can be
recovered. All operations are linear + softmax/log — fully interpretable, NO neurons.
"""

from __future__ import annotations

import numpy as np

from src.models.classifiers import make_classifier, normalize_features


def _log(p, eps=1e-9):
    return np.log(np.clip(p, eps, None))


class _ConstantFine:
    """Degenerate fine 'classifier' for a singleton superclass (one class)."""
    n_classes = 1
    def fit(self, *a, **k): return self
    def predict_proba(self, X):
        return np.ones((len(X), 1), dtype=np.float32)
    def interpretability_report(self):
        return {"backend": "constant", "exact_additive": True, "interpretable": "trivial"}


class HierarchicalClassifier:
    """Coarse superclass router + per-superclass fine classifiers + soft-cascade."""

    def __init__(self, hierarchy_info, config=None, coarse_type='linear',
                 fine_type='histgb', soft_cascade=True,
                 global_feature_names=None):
        self.h = hierarchy_info
        self.config = config or {}
        self.coarse_type = coarse_type
        self.fine_type = fine_type
        self.soft_cascade = soft_cascade
        self.global_feature_names = global_feature_names
        self.n_classes = hierarchy_info.n_classes
        self.S = hierarchy_info.n_superclasses
        self.coarse = None
        self.fine = {}                 # sid -> classifier
        self._norm_stats = None

    # -- fit ---------------------------------------------------------------
    def fit(self, X_train, y_train, X_val=None, y_val=None,
            features_by_superclass=None, sample_weight=None):
        """Train coarse + fine stages.

        Args:
            X_train: [N, D] global features.
            y_train: [N] global class labels.
            features_by_superclass: optional {sid: [N_s, D_s]} per-superclass feature
                matrices (Section 3C). If None, the global features are reused for fine.
        """
        X_train = np.asarray(X_train); y_train = np.asarray(y_train)

        # --- coarse: superclass labels over global features ---
        y_super = np.array([self.h.superclass_of(c) for c in y_train])
        # coarse is a delivered classifier but routes only S superclasses (small) -> ok flat;
        # GBDT coarse would still need hierarchical, so default/recommended is linear.
        self.coarse = make_classifier(
            self.coarse_type, n_classes=self.S, config=self.config,
            feature_names=self.global_feature_names,
            hierarchical_enabled=True,           # coarse is itself the top of the hierarchy
            stage='coarse')
        self.coarse.fit(X_train, y_super, X_val, y_val, sample_weight=sample_weight)

        # --- fine: per-superclass, on that superclass's image subset ---
        for sid, classes in self.h.classes_of.items():
            mask = np.isin(y_train, classes)
            n_s = len(classes)
            if n_s <= 1 or mask.sum() == 0:
                self.fine[sid] = _ConstantFine()
                continue
            # local labels 0..n_s-1
            local_map = self.h.local_index[sid]
            y_local = np.array([local_map[c] for c in y_train[mask]])
            Xs = (np.asarray(features_by_superclass[sid])[mask]
                  if features_by_superclass and sid in features_by_superclass
                  else X_train[mask])
            clf = make_classifier(
                self.fine_type, n_classes=n_s, config=self.config,
                feature_names=self.global_feature_names,
                hierarchical_enabled=True, stage='fine')
            clf.fit(Xs, y_local, sample_weight=None)
            self.fine[sid] = clf
        return self

    # -- inference ---------------------------------------------------------
    def predict_proba(self, X_global, features_by_superclass=None):
        X_global = np.asarray(X_global)
        N = X_global.shape[0]
        coarse_prob = self.coarse.predict_proba(X_global)          # [N, S]
        final_log = np.full((N, self.n_classes), -np.inf, dtype=np.float64)

        for sid, classes in self.h.classes_of.items():
            Xs = (np.asarray(features_by_superclass[sid])
                  if features_by_superclass and sid in features_by_superclass
                  else X_global)
            fine_clf = self.fine[sid]
            fine_prob = fine_clf.predict_proba(Xs)                 # [N, n_s]
            log_fine = _log(fine_prob)
            log_coarse_s = _log(coarse_prob[:, sid])[:, None]      # [N, 1]
            for local_i, global_c in enumerate(classes):
                if self.soft_cascade:
                    final_log[:, global_c] = log_coarse_s[:, 0] + log_fine[:, local_i]
                else:
                    # hard cascade: only the argmax-coarse superclass contributes
                    final_log[:, global_c] = log_fine[:, local_i]

        if not self.soft_cascade:
            # zero out classes whose superclass isn't the coarse argmax
            top_s = np.argmax(coarse_prob, axis=1)
            for sid, classes in self.h.classes_of.items():
                rows = np.where(top_s != sid)[0]
                if len(rows):
                    final_log[np.ix_(rows, classes)] = -np.inf

        # softmax over the 1000-way logits for a proper probability
        m = np.max(final_log, axis=1, keepdims=True)
        ex = np.exp(final_log - m)
        return ex / np.clip(ex.sum(axis=1, keepdims=True), 1e-12, None)

    def predict(self, X_global, features_by_superclass=None):
        return np.argmax(self.predict_proba(X_global, features_by_superclass), axis=1)

    def score(self, X_global, y, features_by_superclass=None):
        return float((self.predict(X_global, features_by_superclass) == np.asarray(y)).mean())

    def coarse_score(self, X_global, y):
        """Superclass-level top-1 of the coarse router (expect 70-80%)."""
        y_super = np.array([self.h.superclass_of(c) for c in np.asarray(y)])
        return float((self.coarse.predict(X_global) == y_super).mean())

    def interpretability_report(self):
        return {
            "coarse": {"type": self.coarse_type, **self.coarse.interpretability_report()},
            "n_superclasses": self.S,
            "n_fine_classifiers": len(self.fine),
            "fine_type": self.fine_type,
            "soft_cascade": self.soft_cascade,
            "note": "per prediction: inspect ONE ~20-way coarse model + ONE ~50-way fine model",
        }
