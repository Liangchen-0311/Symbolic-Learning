"""
Periodic bank reshuffle (v3.3 Section 7B) — STRICT group-level pruning.

This is where strictness lives (admission stays wide, see bank_admission.py). Every
``reshuffle_interval`` GRPO iterations, evaluate the WHOLE bank jointly and prune what is
genuinely useless or redundant. Group evaluation is far more accurate than per-formula
admission, because usefulness is contextual (a weak-alone formula can be complementary).

Pipeline (in order):
  1. MI floor       — drop formulas with near-zero non-linear MI to the labels (HistGB-matched,
                      Section 6B: MI, not linear accuracy). Estimated on a subsample.
  2. Importance floor — fit an L1 linear model (or read HistGB importances) on the survivors;
                        drop zero/near-zero importance formulas (the "Lasso reshuffle").
  3. Wasserstein dedup — non-linear redundancy: among medium-correlation pairs (0.70-0.92)
                        only, drop near-identical *distributions* the linear corr>0.92 gate misses.

Guardrails (Section 7C): never prune below ``min_bank_size``; keep >=1 representative per
correlation cluster (diversity quota); no pruning before ``warmup_iters``; log every reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.classifiers import select_features_l1

try:
    from sklearn.feature_selection import mutual_info_classif
    from scipy.stats import wasserstein_distance
    _DEPS_OK = True
except Exception:  # pragma: no cover
    _DEPS_OK = False


@dataclass
class ReshuffleConfig:
    enabled: bool = True
    reshuffle_interval: int = 100
    mi_floor: float = 0.005
    mi_subsample: int = 50000
    imp_floor: float = 1e-4
    wasserstein_min: float = 0.05
    corr_band: tuple = (0.70, 0.92)
    min_bank_size: int = 2000
    warmup_iters: int = 500

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        return cls(
            enabled=bool(d.get('enabled', True)),
            reshuffle_interval=int(d.get('reshuffle_interval', 100)),
            mi_floor=float(d.get('mi_floor', 0.005)),
            mi_subsample=int(d.get('mi_subsample', 50000)),
            imp_floor=float(d.get('imp_floor', 1e-4)),
            wasserstein_min=float(d.get('wasserstein_min', 0.05)),
            corr_band=tuple(d.get('corr_band', (0.70, 0.92))),
            min_bank_size=int(d.get('min_bank_size', 2000)),
            warmup_iters=int(d.get('warmup_iters', 500)),
        )


def mutual_info_per_feature(X, y, subsample=50000, seed=42):
    """Per-feature non-linear MI to the labels, estimated on a subsample (Section 6B)."""
    X = np.asarray(X); y = np.asarray(y)
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    if n > subsample:
        idx = rng.choice(n, size=subsample, replace=False)
        X, y = X[idx], y[idx]
    return mutual_info_classif(X, y, random_state=seed)


def lasso_importance(X, y, seed=42):
    """Per-feature L1-linear importance magnitude (the 'Lasso reshuffle')."""
    if X.shape[1] == 0:
        return np.zeros(0)
    _idx, importance = select_features_l1(X, y, k=X.shape[1], rng=np.random.RandomState(seed))
    return np.asarray(importance, dtype=float)


def wasserstein_dedup(X, mi=None, corr_band=(0.70, 0.92), w_min=0.05):
    """Drop distributionally-redundant columns among medium-correlation pairs only.

    For each pair whose |Pearson corr| in ``corr_band``, compute the 1-D Wasserstein distance
    between their standardized value distributions; if < ``w_min`` (near-identical shape), drop
    the lower-MI (or higher-index) column. Below the band: clearly different -> skip (save
    compute). Above the band: the existing linear corr>0.92 gate already removed them.

    Returns the sorted list of surviving column indices.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    if d <= 1:
        return list(range(d))
    # standardize columns for a scale-free distribution comparison
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    Z = (X - mu) / sd
    corr = np.corrcoef(Z, rowvar=False)
    corr = np.nan_to_num(corr)
    lo, hi = corr_band
    if mi is None:
        mi = np.ones(d)
    dropped = set()
    for i in range(d):
        if i in dropped:
            continue
        for j in range(i + 1, d):
            if j in dropped:
                continue
            c = abs(corr[i, j])
            if lo <= c <= hi:
                wd = wasserstein_distance(Z[:, i], Z[:, j])
                if wd < w_min:
                    # redundant distributions: keep the more-informative one
                    drop = j if mi[i] >= mi[j] else i
                    dropped.add(drop)
                    if drop == i:
                        break
    return sorted(set(range(d)) - dropped)


def _correlation_clusters(X, threshold=0.92):
    """Group columns into clusters by |corr| >= threshold (union-find)."""
    X = np.asarray(X, dtype=np.float64)
    d = X.shape[1]
    parent = list(range(d))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if d > 1:
        corr = np.nan_to_num(np.corrcoef(X, rowvar=False))
        for i in range(d):
            for j in range(i + 1, d):
                if abs(corr[i, j]) >= threshold:
                    union(i, j)
    clusters = {}
    for i in range(d):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def reshuffle(X_bank, y, cfg=None, iteration=10 ** 9):
    """Run the strict group-level prune over the whole bank.

    Args:
        X_bank: [N_images, n_formulas] cached feature values for all bank formulas.
        y: [N_images] labels.
        cfg: ReshuffleConfig | dict | None.
        iteration: current GRPO iteration (for warmup gating).

    Returns:
        (kept_indices: list[int], log: dict). ``kept_indices`` index into X_bank columns.
        ``log`` reports per-stage drop counts (Section 7C reporting).
    """
    if cfg is None:
        cfg = ReshuffleConfig()
    elif isinstance(cfg, dict):
        cfg = ReshuffleConfig.from_dict(cfg)

    X_bank = np.asarray(X_bank); y = np.asarray(y)
    d = X_bank.shape[1]
    log = {"n_in": d, "warmup_skipped": False,
           "dropped_low_mi": 0, "dropped_low_importance": 0,
           "dropped_wasserstein": 0, "restored_min_bank": 0,
           "restored_diversity": 0, "n_out": d}

    # Guardrail: no pruning during warmup
    if not cfg.enabled or iteration < cfg.warmup_iters:
        log["warmup_skipped"] = (iteration < cfg.warmup_iters)
        return list(range(d)), log
    if d == 0:
        return [], log

    all_idx = np.arange(d)

    # Step 1 — MI floor
    mi = mutual_info_per_feature(X_bank, y, subsample=cfg.mi_subsample)
    keep1 = all_idx[mi > cfg.mi_floor]
    log["dropped_low_mi"] = int(d - len(keep1))
    if len(keep1) == 0:
        keep1 = all_idx  # never drop everything on MI alone

    # Step 2 — importance floor (Lasso on the survivors)
    imp = lasso_importance(X_bank[:, keep1], y)
    keep2 = keep1[imp > cfg.imp_floor]
    log["dropped_low_importance"] = int(len(keep1) - len(keep2))
    if len(keep2) == 0:
        keep2 = keep1

    # Step 3 — Wasserstein distribution dedup (0.70-0.92 corr band only)
    mi_sub = mi[keep2]
    survivors_local = wasserstein_dedup(X_bank[:, keep2], mi=mi_sub,
                                        corr_band=cfg.corr_band, w_min=cfg.wasserstein_min)
    keep3 = keep2[survivors_local]
    log["dropped_wasserstein"] = int(len(keep2) - len(keep3))

    kept = set(int(i) for i in keep3)

    # Guardrail: diversity quota — ensure >=1 representative per correlation cluster
    clusters = _correlation_clusters(X_bank, threshold=cfg.corr_band[1])
    for cluster in clusters:
        if not (set(cluster) & kept):
            # restore the highest-MI member of an entirely-pruned cluster
            best = max(cluster, key=lambda c: mi[c])
            kept.add(int(best))
            log["restored_diversity"] += 1

    # Guardrail: floor on bank size — keep top-min_bank_size by MI if we pruned too hard
    if len(kept) < min(cfg.min_bank_size, d):
        order = np.argsort(mi)[::-1]
        for c in order:
            if len(kept) >= min(cfg.min_bank_size, d):
                break
            if int(c) not in kept:
                kept.add(int(c))
                log["restored_min_bank"] += 1

    kept_sorted = sorted(kept)
    log["n_out"] = len(kept_sorted)
    return kept_sorted, log


def maybe_reshuffle_bank(env, iteration, cfg=None, verbose=True):
    """Integration hook (Section 7B): every ``reshuffle_interval`` GRPO iters, re-evaluate
    the whole feature bank on one common batch and prune in place.

    No-op (returns None) when reshuffle is disabled, during warmup, or off-interval — so
    the legacy loop is unaffected unless explicitly enabled in config.

    Args:
        env: a TensorVSREnvironmentLargeBank (provides feature_bank + get_data_batch).
        iteration: current training iteration.
        cfg: ReshuffleConfig | dict | None.
    Returns:
        log dict (with 'pruned' count) if a reshuffle ran, else None.
    """
    if cfg is None:
        cfg = ReshuffleConfig()
    elif isinstance(cfg, dict):
        cfg = ReshuffleConfig.from_dict(cfg)
    if not cfg.enabled or iteration < cfg.warmup_iters:
        return None
    if cfg.reshuffle_interval <= 0 or (iteration % cfg.reshuffle_interval) != 0:
        return None

    bank = getattr(env, 'feature_bank', None)
    if bank is None or bank.size() == 0:
        return None

    import torch as _torch
    # Re-evaluate every bank formula on ONE common batch so columns share a label axis.
    data_batch, labels = env.get_data_batch(batch_size=min(2000, env.cached_images.size(0)))
    cols, col_idx = [], []
    for i, formula in enumerate(bank.formulas):
        try:
            out = formula.execute(data_batch)
            v = out.reshape(out.shape[0], -1).mean(dim=1) if out.dim() > 1 else out
            cols.append(v.detach().float().cpu().numpy())
            col_idx.append(i)
        except Exception:
            continue
    if len(cols) < 2:
        return None
    X_bank = np.stack(cols, axis=1)
    y = labels.detach().cpu().numpy() if hasattr(labels, 'detach') else np.asarray(labels)

    kept_local, log = reshuffle(X_bank, y, cfg, iteration=iteration)
    # map local survivor columns back to bank indices
    keep_bank_idx = [col_idx[k] for k in kept_local]
    pruned = bank.prune_to_indices(keep_bank_idx)
    log['pruned'] = int(pruned)
    if verbose:
        print(f"  [Reshuffle @ iter {iteration}] {log['n_in']}->{log['n_out']} "
              f"(MI:{log['dropped_low_mi']} imp:{log['dropped_low_importance']} "
              f"wass:{log['dropped_wasserstein']} | restored "
              f"div:{log['restored_diversity']} floor:{log['restored_min_bank']}) pruned={pruned}")
    return log
