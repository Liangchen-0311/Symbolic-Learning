"""
Layer-2 enumeration over Layer-1 feature maps (v3.3 Section 4C).

Constrained form (keeps the search ~10^4 and bloat-free — NO RL here):

    [ f_i  f_j  BinOp  {0-2 UnaryOps}  Pool ]
      f_i, f_j in top-30 Layer-1 bodies, i < j
      BinOp in {subtract, multiply, fuzzy_and, fuzzy_or}
      UnaryOps in size-0..2 subsets of {abs, relu, sigmoid, normalize, blob_detector, contour}
      Pool  in the scalar ROOT_OPERATORS

Two-stage multi-fidelity evaluation:
  Stage A (coarse): univariate accuracy on a subsample -> keep top ``stage_a_keep``.
  Stage B (precise): re-evaluate survivors on the full set -> keep top ``K``.

Dedup: multiply/fuzzy_and/fuzzy_or are symmetric -> only i<j. subtract is antisymmetric;
we keep i<j and let the downstream classifier's sign handle the other order (halves count).
Each surviving formula stores the L1 body indices + ops, so it re-executes from pixels and
stays fully interpretable.
"""

from __future__ import annotations

import itertools
import json

import torch

from src.symbolic.tensor_operators import (
    TensorOperators, TENSOR_OPERATORS, ROOT_OPERATORS, MULTI_DIM_OPERATORS,
)
from src.symbolic.layer1_cache import build_data_batch, execute_body_map

BINOPS = ['subtract', 'multiply', 'fuzzy_and', 'fuzzy_or']
UNARY_CHOICES = ['abs', 'relu', 'sigmoid', 'normalize', 'blob_detector', 'contour']
# scalar pools only (exclude multi-dim roots so each candidate -> one scalar feature)
SCALAR_POOLS = [p for p in sorted(ROOT_OPERATORS) if p not in MULTI_DIM_OPERATORS]


def _unary_subsets(max_unary=2):
    """All size-0..max_unary ordered subsets of UNARY_CHOICES (order matters for chaining,
    but we use combinations to bound the count; a fixed canonical order is applied)."""
    subs = [()]
    for r in range(1, max_unary + 1):
        subs.extend(itertools.combinations(UNARY_CHOICES, r))
    return subs


def univariate_accuracy(feature, labels, n_classes=None):
    """Fast 1-D nearest-centroid classification accuracy of a single feature.

    Standardize the feature, compute per-class means (1-D prototypes), assign each sample
    to the nearest class mean. Deterministic, vectorized, and a good *ranking* proxy for
    "how discriminative is this feature on its own". Returns a float in [0,1].
    """
    f = torch.as_tensor(feature, dtype=torch.float32).reshape(-1)
    y = torch.as_tensor(labels).reshape(-1).long()
    f = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    std = f.std()
    if std > 1e-8:
        f = (f - f.mean()) / std
    classes = torch.unique(y)
    means = torch.stack([f[y == c].mean() for c in classes])      # [C]
    # nearest centroid: argmin |f - mean_c|
    dists = (f.unsqueeze(1) - means.unsqueeze(0)).abs()           # [N, C]
    pred = classes[dists.argmin(dim=1)]
    return float((pred == y).float().mean())


def _apply_unaries(m, unaries):
    for u in unaries:
        fn = TENSOR_OPERATORS[u][0]
        m = fn(m)
        m = torch.nan_to_num(m, nan=0.0, posinf=1e4, neginf=-1e4)
    return m


def _combine(mi, mj, binop):
    fn = TENSOR_OPERATORS[binop][0]
    out = fn(mi, mj)
    return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)


def _pool_scalar(m, pool):
    fn = TENSOR_OPERATORS[pool][0]
    out = fn(m)
    if out.dim() > 1:            # safety: collapse any accidental multi-dim
        out = out.mean(dim=1)
    return out


def _candidate_map(cache, i, j, binop, unaries, image_indices, device):
    mi, mj = cache.get_pair(i, j, image_indices, device=device)
    m = _combine(mi, mj, binop)
    m = _apply_unaries(m, unaries)
    return m


def enumerate_layer2(cache, labels, top_k=2000, stage_a_keep=2000,
                     stage_a_subsample=5000, max_unary=2, device='cpu',
                     pools=None, verbose=False, seed=42):
    """Run the two-stage Layer-2 enumeration over a built ``Layer1Cache``.

    Args:
        cache: a built Layer1Cache (n_bodies maps over N images).
        labels: [N] integer labels aligned to the cache's images.
        top_k: final number of Layer-2 formulas to keep (Stage B).
        stage_a_keep: survivors after the coarse stage.
        stage_a_subsample: image subsample size for Stage A.
        pools: list of pooling ops to consider (default: all scalar ROOT ops).
    Returns:
        list of formula dicts sorted by descending accuracy:
        {i, j, binop, unaries, pool, rpn_l1, accuracy}
    """
    labels_t = torch.as_tensor(labels).long()
    N = cache.n_images
    pools = pools or SCALAR_POOLS
    g = torch.Generator().manual_seed(seed)

    valid_bodies = [b for b in range(cache.n_bodies) if cache.valid_mask[b]]
    unary_subs = _unary_subsets(max_unary)

    # subsample indices for Stage A
    if N > stage_a_subsample:
        sub_idx = torch.randperm(N, generator=g)[:stage_a_subsample]
    else:
        sub_idx = torch.arange(N)
    sub_labels = labels_t[sub_idx]

    # ---- Stage A: coarse univariate accuracy on subsample ----
    stage_a = []
    n_maps = 0
    for i, j in itertools.combinations(valid_bodies, 2):
        for binop in BINOPS:
            for unaries in unary_subs:
                m = _candidate_map(cache, i, j, binop, unaries, sub_idx, device)
                n_maps += 1
                for pool in pools:
                    feat = _pool_scalar(m, pool)
                    acc = univariate_accuracy(feat, sub_labels)
                    stage_a.append((acc, i, j, binop, unaries, pool))
    stage_a.sort(key=lambda t: t[0], reverse=True)
    survivors = stage_a[:stage_a_keep]
    if verbose:
        print(f"  [L2] Stage A: {len(stage_a)} candidates ({n_maps} maps), "
              f"kept {len(survivors)}")

    # ---- Stage B: precise univariate accuracy on full set ----
    stage_b = []
    full_idx = torch.arange(N)
    for _acc, i, j, binop, unaries, pool in survivors:
        m = _candidate_map(cache, i, j, binop, unaries, full_idx, device)
        feat = _pool_scalar(m, pool)
        acc = univariate_accuracy(feat, labels_t)
        stage_b.append((acc, i, j, binop, unaries, pool))
    stage_b.sort(key=lambda t: t[0], reverse=True)
    kept = stage_b[:top_k]
    if verbose:
        print(f"  [L2] Stage B: re-scored {len(survivors)}, kept {len(kept)}")

    out = []
    for acc, i, j, binop, unaries, pool in kept:
        out.append({
            'i': int(i), 'j': int(j), 'binop': binop,
            'unaries': list(unaries), 'pool': pool,
            'rpn_l1': layer2_symbolic_string(i, j, binop, unaries, pool),
            'accuracy': float(acc),
        })
    return out


def layer2_symbolic_string(i, j, binop, unaries, pool):
    """Compact Layer-2 RPN referencing body indices, e.g. 'L1_5 L1_12 subtract abs pool_center'."""
    toks = [f"L1_{i}", f"L1_{j}", binop] + list(unaries) + [pool]
    return ' '.join(toks)


def expand_layer2_to_pixels(formula, bodies):
    """Expand a Layer-2 formula dict into a full pixel-level RPN string by substituting
    each L1 body index with its body RPN. Fully traceable to pixels."""
    bi = bodies[formula['i']]
    bj = bodies[formula['j']]
    toks = [bi, bj, formula['binop']] + list(formula['unaries']) + [formula['pool']]
    return ' '.join(toks)


def execute_layer2_from_pixels(formula, bodies, images, resolution, device='cpu'):
    """Compute a Layer-2 formula's scalar feature directly from pixels (no cache).

    Used by the traceability check: with the same ``resolution`` the cache used, this must
    reproduce the cache-based value within 1e-4. Returns [N] or None if a body is invalid.
    """
    from src.symbolic.layer1_cache import _resize_map
    db = build_data_batch(images, device)
    mi = execute_body_map(bodies[formula['i']], db)
    mj = execute_body_map(bodies[formula['j']], db)
    if mi is None or mj is None:
        return None
    # match Layer1Cache: enumeration ran on resized maps, so resize here too.
    mi = _resize_map(mi.to(device), resolution).to(torch.float32)
    mj = _resize_map(mj.to(device), resolution).to(torch.float32)
    return execute_layer2_from_maps(formula, mi, mj)


def execute_layer2_from_maps(formula, map_i, map_j):
    """Compute the Layer-2 scalar feature from two pre-pool maps [N,R,R] (cache-equivalent)."""
    m = _combine(map_i, map_j, formula['binop'])
    m = _apply_unaries(m, formula['unaries'])
    return _pool_scalar(m, formula['pool'])


def compute_layer2_features(formulas, cache, device='cpu', image_indices=None):
    """Build the Layer-2 feature matrix [N, len(formulas)] from a built cache (Section 4D)."""
    feats = []
    for f in formulas:
        mi, mj = cache.get_pair(f['i'], f['j'], image_indices, device=device)
        feats.append(execute_layer2_from_maps(f, mi, mj))
    return torch.stack(feats, dim=1) if feats else torch.empty(0)


def save_layer2(formulas, path):
    with open(path, 'w') as fh:
        json.dump(formulas, fh, indent=2)


def load_layer2(path):
    with open(path) as fh:
        return json.load(fh)
