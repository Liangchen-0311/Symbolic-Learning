"""
Build imagenet_superclasses.json from the ImageNet train/ directory (v3.3 Section 3A).

Reads the sorted wnids (class index = sorted folder position), builds ~20 WordNet
superclasses (via nltk if available, else a balanced fallback), validates, and saves.

Usage:
    python experiments/build_imagenet_hierarchy.py --data_dir /data/imagenet \
        --out imagenet_superclasses.json --target_groups 20

If you don't have the ImageNet tree handy, pass --wnids_file (a JSON list of 1000 wnids).
Install nltk + wordnet for semantic grouping:
    pip install nltk && python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
"""

import argparse
import json

from src.symbolic.wordnet_hierarchy import (
    build_superclasses, get_imagenet_wnids, save_hierarchy, HierarchyInfo,
    _nltk_available,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=None, help='ImageNet root containing train/')
    ap.add_argument('--wnids_file', default=None, help='JSON list of 1000 wnids (alt to --data_dir)')
    ap.add_argument('--out', default='imagenet_superclasses.json')
    ap.add_argument('--target_groups', type=int, default=20)
    ap.add_argument('--min_size', type=int, default=15)
    ap.add_argument('--max_size', type=int, default=130)
    args = ap.parse_args()

    if args.wnids_file:
        with open(args.wnids_file) as f:
            wnids = json.load(f)
    elif args.data_dir:
        wnids = get_imagenet_wnids(args.data_dir)
    else:
        raise SystemExit("Provide --data_dir or --wnids_file")

    print(f"Loaded {len(wnids)} wnids. nltk WordNet available: {_nltk_available()}")
    h = build_superclasses(wnids, target_groups=args.target_groups,
                           min_size=args.min_size, max_size=args.max_size)
    info = HierarchyInfo(h)
    ok, msgs = info.validate(args.min_size, args.max_size)
    print(f"Method: {h['meta']['method']} | superclasses: {info.n_superclasses}")
    print(f"Sizes: {sorted(h['meta']['sizes'].values())}")
    if not ok:
        print("VALIDATION WARNINGS:")
        for m in msgs:
            print("  -", m)
    save_hierarchy(h, args.out)
    print(f"Saved -> {args.out}")


if __name__ == '__main__':
    main()
