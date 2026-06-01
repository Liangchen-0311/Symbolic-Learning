"""
Select the top-30 Layer-1 bodies (v3.3 Section 4A).

Layer-2 combination count is C(k,2): k=30 -> 435 pairs (vs 100 -> 4950). Redundant
near-duplicates among the 100 contribute little, so we keep the 30 most important.

Importance source (in priority order):
  1. --importance <npy>: an array aligned to the bodies (e.g. per-body L1-weight magnitude
     as produced by step2_l1_selection.py) -> sort descending, take 30.
  2. otherwise: assume the input bodies file is ALREADY importance-ordered (l1_selected_bodies.json
     is saved in descending L1-weight order by step2) -> take the first 30.

Usage:
    python experiments/select_layer1_top30.py \
        --bodies outputs/imagenet_v3/layer2/l1_selected_bodies.json \
        --out    outputs/imagenet_v3/layer2/layer1_top30.json
"""

import argparse
import json
import os

import numpy as np


def select_top30(bodies, importance=None, n=30):
    """Return the top-``n`` bodies by importance (or the first n if already ordered)."""
    if importance is not None:
        importance = np.asarray(importance)
        assert len(importance) == len(bodies), "importance must align to bodies"
        order = np.argsort(importance)[::-1][:n]
        return [bodies[i] for i in order], [float(importance[i]) for i in order]
    return bodies[:n], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bodies', required=True, help='l1_selected_bodies.json (list of RPN bodies)')
    ap.add_argument('--out', required=True, help='output layer1_top30.json')
    ap.add_argument('--importance', default=None, help='optional .npy of per-body importance')
    ap.add_argument('--n', type=int, default=30)
    args = ap.parse_args()

    with open(args.bodies) as f:
        bodies = json.load(f)
    if not isinstance(bodies, list):
        raise ValueError(f"{args.bodies} must contain a JSON list of body strings")

    importance = np.load(args.importance) if args.importance else None
    top, imp = select_top30(bodies, importance, n=args.n)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(top, f, indent=2)
    print(f"Selected {len(top)} bodies -> {args.out}")
    if imp is not None:
        print(f"  importance range: [{min(imp):.4f}, {max(imp):.4f}]")


if __name__ == '__main__':
    main()
