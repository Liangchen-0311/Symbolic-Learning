"""
WordNet hierarchical class decomposition (v3.3 Section 3A).

Turns the flat 1000-way ImageNet problem into ~20 superclasses (coarse) + per-superclass
fine groups, so each sub-problem is near CIFAR-100 scale. This is MANDATORY for HistGB
(Section 6.0: multiclass GBDT trains max_iter x n_classes trees; flat 1000-way explodes).

Constraint 5: WordNet is used ONLY as a class-index lookup (which classes share a parent),
never as a text/semantic feature source. No external embeddings, no pretrained models.

Two build paths:
  1. nltk WordNet (offline lexical DB) -> cut the hypernym tree at a depth giving ~20
     balanced groups (15-130 leaves each).
  2. Fallback (nltk/data unavailable) -> a deterministic balanced partition into
     ``target_groups`` contiguous buckets. Structurally valid (every class mapped once,
     sizes in range) but NOT semantically grouped; a warning is logged.

The grouping core (`group_by_ancestor`, `balance_groups`) takes a pluggable
``ancestor_path_fn(wnid) -> [root..leaf]`` so it is unit-testable without nltk.
"""

from __future__ import annotations

import json
import os
import warnings


# ---------------------------------------------------------------------------
# nltk ancestor function (real WordNet path)
# ---------------------------------------------------------------------------

def _nltk_available():
    try:
        import nltk  # noqa: F401
        from nltk.corpus import wordnet as wn
        wn.synsets  # trigger lazy load
        # confirm data is actually present
        from nltk.corpus import wordnet
        _ = wordnet.synset_from_pos_and_offset('n', 2084071)  # dog.n.01
        return True
    except Exception:
        return False


def make_nltk_ancestor_fn():
    """Return ancestor_path_fn(wnid) using nltk WordNet. wnid like 'n01440764'."""
    from nltk.corpus import wordnet as wn

    def synset_of(wnid):
        offset = int(wnid[1:])
        return wn.synset_from_pos_and_offset('n', offset)

    def ancestor_path_fn(wnid):
        syn = synset_of(wnid)
        # use the longest hypernym path to root (entity); list root..leaf
        paths = syn.hypernym_paths()
        path = max(paths, key=len) if paths else [syn]
        return [f"n{int(s.offset()):08d}" for s in path]

    return ancestor_path_fn


# ---------------------------------------------------------------------------
# Grouping core (pluggable ancestor fn — testable without nltk)
# ---------------------------------------------------------------------------

def group_by_ancestor(wnids, ancestor_path_fn, depth):
    """Group leaf wnids by their ancestor at ``depth`` (distance from root).

    Returns {ancestor_id: [leaf wnids]}. Leaves shorter than ``depth`` are grouped by
    their own deepest available ancestor (the leaf itself if necessary).
    """
    groups = {}
    for w in wnids:
        path = ancestor_path_fn(w)            # [root, ..., leaf]
        anc = path[min(depth, len(path) - 1)]
        groups.setdefault(anc, []).append(w)
    return groups


def _choose_depth(wnids, ancestor_path_fn, target_groups, depth_range=range(1, 14)):
    """Pick the cut depth whose group count is closest to ``target_groups``."""
    best_depth, best_diff = None, 10 ** 9
    for d in depth_range:
        g = group_by_ancestor(wnids, ancestor_path_fn, d)
        diff = abs(len(g) - target_groups)
        if diff < best_diff:
            best_diff, best_depth = diff, d
    return best_depth


def balance_groups(groups, min_size=15, max_size=130):
    """Merge undersized groups into a 'misc' bucket and split oversized groups so every
    final superclass has size in [min_size, max_size] (best effort).

    Returns an ordered dict {superclass_name: [wnids]}.
    """
    final = {}
    misc = []
    # sort by size desc for stable, readable ids
    for name, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(members) < min_size:
            misc.extend(members)
            continue
        if len(members) > max_size:
            # split into ceil(n/max_size) near-equal chunks
            n_chunks = (len(members) + max_size - 1) // max_size
            chunk = (len(members) + n_chunks - 1) // n_chunks
            for c in range(n_chunks):
                part = members[c * chunk:(c + 1) * chunk]
                if part:
                    final[f"{name}__{c}"] = part
        else:
            final[name] = members

    # distribute misc: if it fits as one group, keep it; else chunk it
    if misc:
        if len(misc) <= max_size and len(misc) >= min_size:
            final['misc'] = misc
        elif len(misc) < min_size and final:
            # attach to the smallest existing group
            smallest = min(final, key=lambda k: len(final[k]))
            final[smallest].extend(misc)
        else:
            n_chunks = max(1, (len(misc) + max_size - 1) // max_size)
            chunk = (len(misc) + n_chunks - 1) // n_chunks
            for c in range(n_chunks):
                part = misc[c * chunk:(c + 1) * chunk]
                if part:
                    final[f"misc__{c}"] = part
    return final


def _balanced_fallback(wnids, target_groups):
    """Deterministic balanced partition into ``target_groups`` contiguous buckets.
    Structurally valid but NOT semantic. Used when nltk WordNet is unavailable."""
    n = len(wnids)
    per = (n + target_groups - 1) // target_groups
    groups = {}
    for k in range(target_groups):
        part = wnids[k * per:(k + 1) * per]
        if part:
            groups[f"group_{k}"] = part
    return groups


# ---------------------------------------------------------------------------
# Top-level build
# ---------------------------------------------------------------------------

def build_superclasses(wnids, target_groups=20, min_size=15, max_size=130,
                       use_wordnet=True, ancestor_path_fn=None):
    """Build the superclass partition for an ordered list of wnids (class index = position).

    Returns the hierarchy dict:
        {"superclasses": {name: {"id": k, "classes": [global indices]}},
         "class_to_superclass": {str(idx): k},
         "meta": {...}}
    """
    wnid_to_idx = {w: i for i, w in enumerate(wnids)}

    if ancestor_path_fn is None and use_wordnet and _nltk_available():
        ancestor_path_fn = make_nltk_ancestor_fn()

    if ancestor_path_fn is not None:
        depth = _choose_depth(wnids, ancestor_path_fn, target_groups)
        raw = group_by_ancestor(wnids, ancestor_path_fn, depth)
        groups = balance_groups(raw, min_size, max_size)
        method = f"wordnet(depth={depth})"
    else:
        warnings.warn("WordNet (nltk) unavailable — using a balanced non-semantic "
                      "fallback partition. Install nltk + wordnet for semantic superclasses.")
        groups = _balanced_fallback(wnids, target_groups)
        method = "balanced_fallback"

    superclasses = {}
    class_to_superclass = {}
    for k, (name, members) in enumerate(groups.items()):
        idx_list = sorted(wnid_to_idx[w] for w in members)
        superclasses[name] = {"id": k, "classes": idx_list}
        for c in idx_list:
            class_to_superclass[str(c)] = k

    return {
        "superclasses": superclasses,
        "class_to_superclass": class_to_superclass,
        "meta": {"method": method, "n_classes": len(wnids),
                 "n_superclasses": len(superclasses),
                 "sizes": {name: len(v["classes"]) for name, v in superclasses.items()}},
    }


def get_imagenet_wnids(data_dir):
    """Sorted ImageNet-1K wnids from the train/ directory (class index = sorted position)."""
    train_dir = os.path.join(data_dir, 'train')
    wnids = sorted(d for d in os.listdir(train_dir)
                   if os.path.isdir(os.path.join(train_dir, d)))
    return wnids


def save_hierarchy(hierarchy, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(hierarchy, f, indent=2)


def load_hierarchy(path):
    with open(path) as f:
        return json.load(f)


class HierarchyInfo:
    """Convenience accessor over a hierarchy dict for the cascade classifier."""

    def __init__(self, hierarchy):
        self.h = hierarchy
        self.superclasses = hierarchy["superclasses"]
        self.class_to_superclass = {int(k): v for k, v in hierarchy["class_to_superclass"].items()}
        self.n_superclasses = len(self.superclasses)
        self.n_classes = hierarchy["meta"]["n_classes"]
        # id -> sorted global class indices
        self.classes_of = {info["id"]: list(info["classes"])
                           for info in self.superclasses.values()}
        # id -> {global_class: local_index}
        self.local_index = {}
        for sid, classes in self.classes_of.items():
            self.local_index[sid] = {c: li for li, c in enumerate(classes)}

    def superclass_of(self, global_class):
        return self.class_to_superclass[int(global_class)]

    def validate(self, min_size=15, max_size=130):
        """Assert every class maps to exactly one superclass and sizes are in range
        (allowing a 'misc' overflow bucket). Returns (ok, messages)."""
        msgs = []
        seen = {}
        for sid, classes in self.classes_of.items():
            for c in classes:
                if c in seen:
                    msgs.append(f"class {c} in multiple superclasses ({seen[c]}, {sid})")
                seen[c] = sid
        if len(seen) != self.n_classes:
            msgs.append(f"covered {len(seen)} classes, expected {self.n_classes}")
        for name, info in self.superclasses.items():
            sz = len(info["classes"])
            if not (min_size <= sz <= max_size) and not name.startswith('misc'):
                msgs.append(f"superclass '{name}' size {sz} out of [{min_size},{max_size}]")
        return (len(msgs) == 0), msgs
