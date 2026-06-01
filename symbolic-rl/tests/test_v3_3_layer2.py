"""Unit tests for v3.3 Section 4 — Layer-1 cache + Layer-2 enumeration."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.symbolic.layer1_cache import Layer1Cache, execute_body_map, build_data_batch
from src.symbolic.layer2_enumerate import (
    enumerate_layer2, univariate_accuracy, compute_layer2_features,
    execute_layer2_from_pixels, execute_layer2_from_maps,
    layer2_symbolic_string, expand_layer2_to_pixels, save_layer2, load_layer2,
    BINOPS, SCALAR_POOLS,
)
from experiments.select_layer1_top30 import select_top30


BODIES = ['I_R', 'I_G edge_x', 'I_GRAY blob_detector', 'I_B local_std_5x5',
          'I_RG abs', 'I_GRAY edge_mag']


def _toy_images(n=40, res=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, res, res, generator=g)


def _toy_labels(n=40, n_classes=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, n_classes, (n,), generator=g)


def test_execute_body_map_strips_pool():
    imgs = _toy_images()
    db = build_data_batch(imgs, 'cpu')
    # body without pool -> 2D map
    m = execute_body_map('I_R edge_x', db)
    assert m is not None and m.dim() == 3 and m.shape[0] == imgs.shape[0]
    # body that accidentally ends in a pool -> pool dropped, still a 2D map
    m2 = execute_body_map('I_R edge_x global_avg_pool', db)
    assert m2 is not None and m2.dim() == 3


def test_univariate_accuracy_separable():
    # perfectly separable feature -> high accuracy
    y = torch.tensor([0, 0, 0, 1, 1, 1])
    f = torch.tensor([-2.0, -1.8, -1.9, 2.0, 1.8, 2.1])
    assert univariate_accuracy(f, y) > 0.9
    # pure noise vs labels -> near chance (loose bound)
    yn = torch.randint(0, 4, (200,))
    fn = torch.randn(200)
    assert univariate_accuracy(fn, yn) < 0.6


def test_layer1_cache_build_get():
    imgs = _toy_images()
    cache = Layer1Cache(BODIES, device='cpu', resolution=8, storage='cpu')
    cache.build(imgs)
    assert cache.n_images == imgs.shape[0]
    m = cache.get(0)
    assert m.shape == (imgs.shape[0], 8, 8) and m.dtype == torch.float32
    sub = cache.get(1, image_indices=[0, 1, 2])
    assert sub.shape == (3, 8, 8)
    assert cache.valid_mask.all()         # all toy bodies execute cleanly


def test_enumerate_layer2_runs_and_sorts():
    imgs = _toy_images()
    labels = _toy_labels()
    cache = Layer1Cache(BODIES, device='cpu', resolution=8, storage='cpu').build(imgs)
    formulas = enumerate_layer2(cache, labels, top_k=20, stage_a_keep=40,
                                stage_a_subsample=30, max_unary=1,
                                pools=['global_avg_pool', 'pool_center'])
    assert 0 < len(formulas) <= 20
    accs = [f['accuracy'] for f in formulas]
    assert accs == sorted(accs, reverse=True)         # descending
    f0 = formulas[0]
    assert f0['i'] < f0['j']                           # dedup i<j
    assert f0['binop'] in BINOPS
    assert f0['pool'] in SCALAR_POOLS
    assert f0['rpn_l1'].startswith(f"L1_{f0['i']} L1_{f0['j']}")


def test_layer2_traceability():
    """Re-executing a saved Layer-2 formula from pixels reproduces the cached value (<1e-4)."""
    imgs = _toy_images()
    labels = _toy_labels()
    R = 8
    cache = Layer1Cache(BODIES, device='cpu', resolution=R, storage='cpu').build(imgs)
    formulas = enumerate_layer2(cache, labels, top_k=5, stage_a_keep=20,
                                stage_a_subsample=40, max_unary=2,
                                pools=['global_avg_pool'])
    f = formulas[0]
    cached_feat = compute_layer2_features([f], cache, device='cpu')[:, 0]   # [N]
    pixel_feat = execute_layer2_from_pixels(f, BODIES, imgs, resolution=R, device='cpu')
    assert pixel_feat is not None
    max_err = (cached_feat - pixel_feat).abs().max().item()
    assert max_err < 1e-4, f"traceability mismatch: {max_err}"


def test_symbolic_and_io_roundtrip(tmp_path=None):
    import tempfile
    f = {'i': 2, 'j': 5, 'binop': 'subtract', 'unaries': ['abs'], 'pool': 'pool_center',
         'rpn_l1': layer2_symbolic_string(2, 5, 'subtract', ['abs'], 'pool_center'),
         'accuracy': 0.42}
    assert f['rpn_l1'] == 'L1_2 L1_5 subtract abs pool_center'
    expanded = expand_layer2_to_pixels(f, BODIES)
    assert expanded.startswith(BODIES[2]) and BODIES[5] in expanded
    d = tempfile.mkdtemp()
    p = os.path.join(d, 'l2.json')
    save_layer2([f], p)
    assert load_layer2(p)[0]['rpn_l1'] == f['rpn_l1']


def test_select_top30():
    bodies = [f"body_{i}" for i in range(100)]
    top, imp = select_top30(bodies, n=30)
    assert len(top) == 30 and top[0] == 'body_0'      # first-30 (importance-ordered input)
    import numpy as np
    importance = np.arange(100)[::-1]                  # body_0 most important
    top2, imp2 = select_top30(bodies, importance=importance, n=30)
    assert top2[0] == 'body_0' and len(top2) == 30


def test_memmap_storage():
    import tempfile
    imgs = _toy_images(n=12, res=8)
    d = tempfile.mkdtemp()
    path = os.path.join(d, 'cache.mmap')
    cache = Layer1Cache(BODIES, device='cpu', resolution=8, storage='memmap',
                        memmap_path=path).build(imgs)
    m = cache.get(0, image_indices=[0, 1])
    assert m.shape == (2, 8, 8) and m.dtype == torch.float32


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f"ok: {name}")
    print('All v3.3 Section 4 Layer-2 tests passed.')
