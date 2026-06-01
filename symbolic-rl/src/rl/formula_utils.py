"""
Formula / multi-objective utilities for GRPO (v3.3 Section 2).

Pure functions (no torch needed) so they are cheaply unit-testable:
  - ``rpn_depth``               expression-tree depth of an RPN token list
  - ``fast_non_dominated_sort`` NSGA-II non-dominated sort over objectives
  - ``crowding_distance``       NSGA-II crowding distance within a front
"""

from __future__ import annotations

import math

from src.symbolic.tensor_operators import TENSOR_OPERATORS


def rpn_depth(tokens, operators=None):
    """Expression-tree depth of an RPN formula given as a list of decoded token strings.

    terminals = depth 1; unary op = child + 1; binary op = max(children) + 1;
    pooling op (arity 1) = child + 1. Returns 0 for an empty token list.

    Arity is read from ``operators`` (defaults to TENSOR_OPERATORS); a token not in the
    registry is treated as a terminal (depth 1).

    Examples:
        "I_R edge_x pool_center"   -> 3
        "I_R I_G add pool_center"  -> 3
    """
    operators = operators if operators is not None else TENSOR_OPERATORS
    if not tokens:
        return 0
    stack = []
    for tok in tokens:
        if tok in operators:
            arity = operators[tok][1]
            if len(stack) < arity:
                # malformed RPN — be defensive, treat missing operands as depth 0
                children = [stack.pop() for _ in range(len(stack))]
            else:
                children = [stack.pop() for _ in range(arity)]
            stack.append((max(children) if children else 0) + 1)
        else:
            stack.append(1)  # terminal
    return max(stack) if stack else 0


def _dominates(a, b, directions):
    """True if objective-vector ``a`` Pareto-dominates ``b`` under per-axis directions
    ('max' = larger is better, 'min' = smaller is better). a dominates b iff a is no
    worse on every axis and strictly better on at least one."""
    strictly_better = False
    for av, bv, d in zip(a, b, directions):
        if d == 'max':
            if av < bv:
                return False
            if av > bv:
                strictly_better = True
        else:  # 'min'
            if av > bv:
                return False
            if av < bv:
                strictly_better = True
    return strictly_better


def fast_non_dominated_sort(objectives, directions):
    """Standard NSGA-II fast non-dominated sort.

    Args:
        objectives: list of objective tuples, one per individual.
        directions: list of 'max'/'min', one per objective axis.

    Returns:
        ranks: list[int] — Pareto front index per individual (0 = best front).
        fronts: list[list[int]] — indices grouped by front.
    """
    n = len(objectives)
    S = [[] for _ in range(n)]          # individuals each i dominates
    n_dom = [0] * n                     # how many dominate i
    ranks = [0] * n
    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(objectives[p], objectives[q], directions):
                S[p].append(q)
            elif _dominates(objectives[q], objectives[p], directions):
                n_dom[p] += 1
        if n_dom[p] == 0:
            ranks[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    ranks[q] = i + 1
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()  # drop trailing empty front
    return ranks, fronts


def crowding_distance(objectives, front, directions):
    """NSGA-II crowding distance for the individuals in ``front`` (a list of indices).

    Returns a dict {index: distance}. Boundary points get +inf (preserved for diversity).
    """
    dist = {i: 0.0 for i in front}
    if len(front) == 0:
        return dist
    if len(front) <= 2:
        for i in front:
            dist[i] = math.inf
        return dist
    m = len(objectives[front[0]])
    for k in range(m):
        ordered = sorted(front, key=lambda i: objectives[i][k])
        vmin = objectives[ordered[0]][k]
        vmax = objectives[ordered[-1]][k]
        span = (vmax - vmin) or 1.0
        dist[ordered[0]] = math.inf
        dist[ordered[-1]] = math.inf
        for j in range(1, len(ordered) - 1):
            prev_v = objectives[ordered[j - 1]][k]
            next_v = objectives[ordered[j + 1]][k]
            dist[ordered[j]] += (next_v - prev_v) / span
    return dist
