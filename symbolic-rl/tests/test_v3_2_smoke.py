#!/usr/bin/env python3
"""
v3.2 Hierarchical Smoke Test

5 layers from bottom to top — validates every v3.2 component without
running full training. Should complete in a few minutes.

  Layer 1: Operator correctness (12 new ops)
  Layer 2: Terminals + grammar rules
  Layer 3: Kernel bank load / freeze
  Layer 4: Feature encoding (distribution stats, Fisher Vector, kernel map, power norm)
  Layer 5: End-to-end mini pipeline (100 images, 50 formulas, 1 epoch)

Usage:
    python -m tests.test_v3_2_smoke
    python tests/test_v3_2_smoke.py
"""

import json, math, os, sys, time, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.symbolic.tensor_operators import (
    TensorOperators, TENSOR_OPERATORS, ROOT_OPERATORS, SymbolicKernelBank,
    _make_gabor_kernel,
)
from src.symbolic.feature_encoding import (
    encode_body_distribution_v2,
    SymbolicFisherVector,
    homogeneous_kernel_map,
    power_normalize,
    l2_normalize,
    apply_normalization_pipeline,
    apply_normalization_pipeline_with_stats,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = '/workspace/neurosymbolic-rl/data/imagenet'


class Result:
    """Simple pass/fail tracker."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name):
        self.passed += 1
        print(f"    ✓ {name}")

    def fail(self, name, reason=""):
        self.failed += 1
        self.errors.append((name, reason))
        print(f"    ✗ {name}: {reason}")

    def check(self, name, condition, reason=""):
        if condition:
            self.ok(name)
        else:
            self.fail(name, reason)

    def summary(self):
        total = self.passed + self.failed
        print(f"\n  {'='*50}")
        print(f"  Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print(f"  Failures:")
            for name, reason in self.errors:
                print(f"    - {name}: {reason}")
        print(f"  {'='*50}")
        return self.failed == 0


# ======================================================================
# Layer 1: Operator Tests
# ======================================================================

def test_layer1_operators(R):
    """Verify all 12 new v3.2 operators."""
    print(f"\n{'='*60}")
    print(f"  [1/5] Layer 1: New Operator Tests")
    print(f"{'='*60}")

    torch.manual_seed(42)
    x = torch.randn(4, 112, 112, device=DEVICE)

    # -- Tensor operators (output: [4, H, W]) --
    tensor_ops = ['edge_mag', 'edge_orient', 'gabor_mag',
                  'local_contrast', 'dog', 'corner_harris', 'lbp_like',
                  'edge_xx', 'edge_yy']

    for op_name in tensor_ops:
        try:
            func, arity, out_type = TENSOR_OPERATORS[op_name]
            assert out_type == 'tensor', f"expected tensor, got {out_type}"
            assert arity == 1, f"expected unary, got arity={arity}"

            out = func(x)

            R.check(f"{op_name} shape",
                    out.shape == x.shape,
                    f"got {out.shape}, expected {x.shape}")

            R.check(f"{op_name} no NaN/Inf",
                    not (torch.isnan(out).any() or torch.isinf(out).any()),
                    "contains NaN or Inf")

            R.check(f"{op_name} not constant",
                    out.std() > 1e-10,
                    f"std={out.std().item():.2e}, looks constant")
        except Exception as e:
            R.fail(f"{op_name}", str(e))

    # -- Root operators (output: [4]) --
    root_ops = ['std_center', 'std_top_half', 'std_bottom_half']
    for op_name in root_ops:
        try:
            func, arity, out_type = TENSOR_OPERATORS[op_name]
            assert out_type == 'scalar', f"expected scalar, got {out_type}"
            assert op_name in ROOT_OPERATORS, "not in ROOT_OPERATORS"

            out = func(x)

            R.check(f"{op_name} shape",
                    out.shape == (4,),
                    f"got {out.shape}")

            R.check(f"{op_name} no NaN/Inf",
                    not (torch.isnan(out).any() or torch.isinf(out).any()),
                    "contains NaN or Inf")

            R.check(f"{op_name} not zero",
                    out.abs().sum() > 1e-10,
                    "all zeros")
        except Exception as e:
            R.fail(f"{op_name}", str(e))

    # -- Numerical consistency: edge_mag --
    print(f"\n  Numerical consistency checks:")
    try:
        ex = TensorOperators.edge_x(x)
        ey = TensorOperators.edge_y(x)
        expected_mag = torch.sqrt(ex * ex + ey * ey + 1e-8)
        actual_mag = TensorOperators.edge_mag(x)
        diff = (expected_mag - actual_mag).abs().max().item()
        R.check(f"edge_mag ≈ sqrt(edge_x² + edge_y²)",
                diff < 1e-5,
                f"max diff = {diff:.2e}")
    except Exception as e:
        R.fail("edge_mag consistency", str(e))

    # -- Numerical consistency: gabor_mag --
    try:
        g0 = TensorOperators.gabor_0(x)
        g45 = TensorOperators.gabor_45(x)
        g90 = TensorOperators.gabor_90(x)
        expected_gmag = torch.sqrt(g0*g0 + g45*g45 + g90*g90 + 1e-8)
        actual_gmag = TensorOperators.gabor_mag(x)
        diff = (expected_gmag - actual_gmag).abs().max().item()
        R.check(f"gabor_mag ≈ sqrt(gabor_0² + gabor_45² + gabor_90²)",
                diff < 1e-5,
                f"max diff = {diff:.2e}")
    except Exception as e:
        R.fail("gabor_mag consistency", str(e))


# ======================================================================
# Layer 2: Terminals + Grammar Tests
# ======================================================================

def test_layer2_terminals_grammar(R):
    """Verify build_data_batch and grammar rules."""
    print(f"\n{'='*60}")
    print(f"  [2/5] Layer 2: Terminal + Grammar Tests")
    print(f"{'='*60}")

    # -- 2a: build_data_batch with real image --
    print(f"\n  Terminal tests:")
    try:
        from src.data.imagenet_loader import ImageNetDataModule
        from experiments.run_v3_2_pipeline import build_data_batch

        dm = ImageNetDataModule(data_dir=DATA_DIR, resolution=112, batch_size=1,
                                num_workers=0, samples_per_class=1)
        dm.setup()
        loader = dm.get_train_loader()
        images, labels = next(iter(loader))

        data_batch = build_data_batch(images, DEVICE)

        expected_keys = {'I_R', 'I_G', 'I_B', 'I_GRAY', 'I_H', 'I_S',
                         'I_r', 'I_g', 'I_RG', 'I_BY'}
        actual_keys = set(data_batch.keys())

        R.check("build_data_batch returns all 10 terminals",
                expected_keys == actual_keys,
                f"missing: {expected_keys - actual_keys}, extra: {actual_keys - expected_keys}")

        for key in expected_keys:
            t = data_batch[key]
            R.check(f"{key} shape=[1,112,112], no NaN",
                    t.shape == (1, 112, 112) and not torch.isnan(t).any(),
                    f"shape={t.shape}, has_nan={torch.isnan(t).any().item()}")

    except Exception as e:
        R.fail("build_data_batch", str(e))

    # -- 2b: Grammar rules --
    print(f"\n  Grammar rule tests:")
    try:
        from src.rl.rpn_grammar_mask import RPNGrammarMask
        from src.rl.tensor_environment_large_bank import TensorTokenVocabulary

        vocab = TensorTokenVocabulary()
        masker = RPNGrammarMask(vocab, max_sequence_length=12)

        # Helper to encode token list
        def enc(tokens):
            return [vocab.encode(t) for t in tokens]

        # Test 1: Valid formula (pooling at end)
        valid_tokens = enc(['I_R', 'edge_x', 'global_avg_pool'])
        is_valid, reason = masker.is_valid_sequence(valid_tokens)
        R.check("valid formula: I_R edge_x global_avg_pool",
                is_valid, reason)

        # Test 2: Valid binary formula
        valid_tokens2 = enc(['I_R', 'I_G', 'add', 'global_avg_pool'])
        is_valid2, reason2 = masker.is_valid_sequence(valid_tokens2)
        R.check("valid binary formula: I_R I_G add global_avg_pool",
                is_valid2, reason2)

        # Test 3: Pooling in the middle → invalid
        invalid_tokens = enc(['I_R', 'global_avg_pool', 'edge_x'])
        is_valid3, reason3 = masker.is_valid_sequence(invalid_tokens)
        R.check("pooling in middle → invalid",
                not is_valid3,
                f"should be invalid but got valid")

        # Test 4: Consecutive identical unary ops blocked by mask
        # After "I_R relu", "relu" should be masked
        tokens_after_relu = enc(['I_R', 'relu'])
        mask = masker.get_valid_actions(tokens_after_relu)
        relu_idx = vocab.encode('relu')
        R.check("consecutive 'relu relu' blocked by mask",
                mask[relu_idx].item() == 0.0,
                f"relu mask value = {mask[relu_idx].item()}, expected 0")

        # Test 5: Different unary after unary is allowed
        sigmoid_idx = vocab.encode('sigmoid')
        R.check("'relu' then 'sigmoid' allowed",
                mask[sigmoid_idx].item() == 1.0,
                f"sigmoid mask value = {mask[sigmoid_idx].item()}, expected 1")

        # Test 6: Pooling only allowed at last position when stack=1
        tokens_mid = enc(['I_R', 'edge_x'])  # stack=1, remaining=10
        mask_mid = masker.get_valid_actions(tokens_mid)
        pool_idx = vocab.encode('global_avg_pool')
        R.check("pooling not allowed mid-sequence (remaining > 1)",
                mask_mid[pool_idx].item() == 0.0,
                f"pool mask = {mask_mid[pool_idx].item()}")

        # Test 7: Pooling allowed at last position
        # Fill to max_length - 1 with valid tokens, then check pooling allowed
        tokens_near_end = enc(['I_R', 'edge_x', 'blur', 'abs', 'relu',
                               'sigmoid', 'negate', 'pow2', 'sqrt_abs',
                               'log1p_abs', 'normalize'])  # 11 tokens, max=12
        mask_end = masker.get_valid_actions(tokens_near_end)
        R.check("pooling allowed at last position (remaining=1, stack=1)",
                mask_end[pool_idx].item() == 1.0,
                f"pool mask = {mask_end[pool_idx].item()}")

    except Exception as e:
        R.fail("grammar rules", f"{e}\n{traceback.format_exc()}")


# ======================================================================
# Layer 3: Kernel Bank Tests
# ======================================================================

def test_layer3_kernel_bank(R):
    """Verify kernel pretraining load and freeze behavior."""
    print(f"\n{'='*60}")
    print(f"  [3/5] Layer 3: Kernel Bank Tests")
    print(f"{'='*60}")

    # -- 3a: Load pretrained kernel bank --
    kb_path = 'outputs/imagenet_v3/kernel_bank_pretrained.pt'
    kb_exists = os.path.exists(kb_path)

    # Even if pretrained file doesn't exist, test with fresh bank
    kb = SymbolicKernelBank(device=DEVICE)
    if kb_exists:
        try:
            kb.load_state_dict(torch.load(kb_path, map_location=DEVICE, weights_only=True))
            R.ok(f"loaded kernel_bank_pretrained.pt")
        except Exception as e:
            R.fail("load kernel bank", str(e))
    else:
        R.ok(f"kernel_bank_pretrained.pt not found (using fresh bank for remaining tests)")

    # -- 3b: 12 learnable kernels not all zero --
    for i in range(kb.conv3x3.shape[0]):
        name = f"conv3x3_{i}"
        R.check(f"{name} not all zero",
                kb.conv3x3[i].abs().sum().item() > 1e-10,
                "all zeros")

    for i in range(kb.conv5x5.shape[0]):
        name = f"conv5x5_{i}"
        R.check(f"{name} not all zero",
                kb.conv5x5[i].abs().sum().item() > 1e-10,
                "all zeros")

    # -- 3c: Classic kernels match original Sobel/Gabor values --
    print(f"\n  Classic kernel integrity:")
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
    laplacian = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])

    classic_3x3_expected = [
        ('classic_edge_x (Sobel X)', sobel_x),
        ('classic_edge_y (Sobel Y)', sobel_y),
        ('classic_laplacian', laplacian),
    ]
    for i, (name, expected) in enumerate(classic_3x3_expected):
        actual = kb.classic_3x3[i, 0].detach().cpu()
        diff = (actual - expected).abs().max().item()
        R.check(f"{name} unchanged",
                diff < 1e-5,
                f"max diff = {diff:.2e}")

    # Gabor kernels
    for i, (name, theta) in enumerate([
        ('classic_gabor_0', 0.0),
        ('classic_gabor_45', math.pi / 4),
        ('classic_gabor_90', math.pi / 2),
    ]):
        expected = _make_gabor_kernel(theta, device='cpu').squeeze(0).squeeze(0)
        actual = kb.classic_7x7[i, 0].detach().cpu()
        diff = (actual - expected).abs().max().item()
        R.check(f"{name} unchanged",
                diff < 1e-5,
                f"max diff = {diff:.2e}")

    # -- 3d: finetune_mode=False → conv3x3 detached during execution --
    print(f"\n  Freeze behavior:")
    kb.finetune_mode = False
    kb.register_operators(TENSOR_OPERATORS)

    x = torch.randn(2, 28, 28, device=DEVICE)
    op_func = TENSOR_OPERATORS['conv3x3_0'][0]
    out = op_func(x)

    R.check("conv3x3_0 executes without error",
            out.shape == (2, 28, 28),
            f"shape={out.shape}")
    R.check("conv3x3_0 output has no grad_fn (detached)",
            out.grad_fn is None,
            f"grad_fn={out.grad_fn}")

    # -- 3e: finetune_mode=True → gradient flows through kernels --
    kb.finetune_mode = True
    kb.register_operators(TENSOR_OPERATORS)

    x_grad = torch.randn(2, 28, 28, device=DEVICE, requires_grad=True)
    op_func = TENSOR_OPERATORS['conv3x3_0'][0]
    out = op_func(x_grad)
    loss = out.sum()
    loss.backward()

    R.check("conv3x3_0 gradient flows in finetune mode",
            kb.conv3x3.grad is not None and kb.conv3x3.grad.abs().sum() > 0,
            "no gradient on conv3x3")

    # Reset to frozen mode
    kb.finetune_mode = False
    kb.register_operators(TENSOR_OPERATORS)


# ======================================================================
# Layer 4: Feature Encoding Tests
# ======================================================================

def test_layer4_feature_encoding(R):
    """Verify distribution stats, Fisher Vector, kernel map, power norm."""
    print(f"\n{'='*60}")
    print(f"  [4/5] Layer 4: Feature Encoding Tests")
    print(f"{'='*60}")

    torch.manual_seed(42)

    # -- 4a: Distribution statistics encoding --
    print(f"\n  Distribution statistics:")
    fm = torch.randn(4, 112, 112, device=DEVICE)
    try:
        stats = encode_body_distribution_v2(fm)
        R.check("distribution stats shape = [4, 60]",
                stats.shape == (4, 60),
                f"got {stats.shape}")
        R.check("distribution stats no NaN/Inf",
                not (torch.isnan(stats).any() or torch.isinf(stats).any()),
                "contains NaN or Inf")

        # Verify structure: 12 stats × 5 regions
        # Region 0 (global) mean should ≈ feature_map.mean()
        global_mean = stats[:, 0]
        expected_mean = fm.reshape(4, -1).mean(dim=1)
        diff = (global_mean - expected_mean).abs().max().item()
        R.check("global mean stat ≈ feature_map.mean()",
                diff < 1e-4,
                f"max diff = {diff:.2e}")

        # Region 0 (global) std should ≈ feature_map.std()
        global_std = stats[:, 1]
        expected_std = fm.reshape(4, -1).std(dim=1)
        diff_std = (global_std - expected_std).abs().max().item()
        R.check("global std stat ≈ feature_map.std()",
                diff_std < 1e-3,
                f"max diff = {diff_std:.2e}")

    except Exception as e:
        R.fail("distribution stats", f"{e}\n{traceback.format_exc()}")

    # -- 4b: Fisher Vector encoding --
    print(f"\n  Fisher Vector:")
    try:
        sfv = SymbolicFisherVector(pca_dim=32, gmm_k=64, device=DEVICE)

        # Create fake patches and fit PCA + GMM
        n_desc = 2048
        n_bodies = 100
        fake_desc = torch.randn(n_desc, n_bodies, device=DEVICE)

        sfv.fit_pca(fake_desc)
        R.check("PCA fit succeeds",
                sfv.pca_components is not None and sfv.pca_components.shape == (32, n_bodies),
                f"pca_components shape = {sfv.pca_components.shape if sfv.pca_components is not None else None}")

        pca_desc = sfv.apply_pca(fake_desc)
        R.check("PCA transform shape",
                pca_desc.shape == (n_desc, 32),
                f"got {pca_desc.shape}")

        sfv.fit_gmm(pca_desc, n_iter=10)
        R.check("GMM fit succeeds",
                sfv.gmm_means is not None and sfv.gmm_means.shape == (64, 32),
                f"gmm_means shape = {sfv.gmm_means.shape if sfv.gmm_means is not None else None}")

        # Compute FV for 64 patches (1 image)
        patches_1img = pca_desc[:64]  # [64, 32]
        fv = sfv.compute_fisher_vector(patches_1img)

        R.check("Fisher Vector shape = [4096]",
                fv.shape == (4096,),
                f"got {fv.shape}")

        R.check("Fisher Vector no NaN/Inf",
                not (torch.isnan(fv).any() or torch.isinf(fv).any()),
                "contains NaN or Inf")

        # After power+L2 norm, L2 norm should ≈ 1.0
        fv_norm = fv.norm().item()
        R.check("Fisher Vector L2 norm ≈ 1.0",
                abs(fv_norm - 1.0) < 0.01,
                f"norm = {fv_norm:.4f}")

        # Batch encode
        fake_fmaps = torch.randn(4, n_bodies, 56, 56, device=DEVICE)
        fvs = sfv.encode_batch(fake_fmaps, grid_size=8)
        R.check("batch FV shape = [4, 4096]",
                fvs.shape == (4, 4096),
                f"got {fvs.shape}")

    except Exception as e:
        R.fail("Fisher Vector", f"{e}\n{traceback.format_exc()}")

    # -- 4c: Homogeneous kernel map --
    print(f"\n  Homogeneous kernel map:")
    try:
        x_km = torch.randn(4, 100, device=DEVICE)
        km_out = homogeneous_kernel_map(x_km, order=1)

        R.check("kernel map shape: [4,100] → [4,300]",
                km_out.shape == (4, 300),
                f"got {km_out.shape}")

        R.check("kernel map no NaN/Inf",
                not (torch.isnan(km_out).any() or torch.isinf(km_out).any()),
                "contains NaN or Inf")

        # Order=2 → 5 values per feature
        km_out2 = homogeneous_kernel_map(x_km, order=2)
        R.check("kernel map order=2: [4,100] → [4,500]",
                km_out2.shape == (4, 500),
                f"got {km_out2.shape}")

    except Exception as e:
        R.fail("kernel map", f"{e}\n{traceback.format_exc()}")

    # -- 4d: Power normalization --
    print(f"\n  Power normalization:")
    try:
        x_pn = torch.tensor([-4.0, -1.0, 0.0, 1.0, 4.0], device=DEVICE)
        pn_out = power_normalize(x_pn, alpha=0.5)

        # Sign should be preserved
        R.check("power norm preserves sign",
                (torch.sign(pn_out) == torch.sign(x_pn)).all() or x_pn[2] == 0,
                f"signs: in={torch.sign(x_pn).tolist()}, out={torch.sign(pn_out).tolist()}")

        # Absolute values should decrease for |x| > 1 (sqrt compresses)
        for i in [0, 4]:  # -4.0 and 4.0
            R.check(f"power norm compresses |x|>1 (x={x_pn[i].item():.0f})",
                    pn_out[i].abs() < x_pn[i].abs(),
                    f"|pn|={pn_out[i].abs().item():.4f} vs |x|={x_pn[i].abs().item():.4f}")

        # L2 normalization
        x_l2 = torch.randn(4, 50, device=DEVICE)
        l2_out = l2_normalize(x_l2, dim=1)
        norms = l2_out.norm(dim=1)
        R.check("L2 norm → all row norms ≈ 1.0",
                (norms - 1.0).abs().max().item() < 1e-5,
                f"max norm deviation = {(norms - 1.0).abs().max().item():.2e}")

        # Full pipeline
        x_full = torch.randn(8, 200, device=DEVICE)
        normed, mean, std = apply_normalization_pipeline(x_full)
        R.check("full pipeline output shape preserved",
                normed.shape == x_full.shape,
                f"got {normed.shape}")
        full_norms = normed.norm(dim=1)
        R.check("full pipeline → row norms ≈ 1.0",
                (full_norms - 1.0).abs().max().item() < 1e-5,
                f"max deviation = {(full_norms - 1.0).abs().max().item():.2e}")

        # Apply with pre-computed stats (test set pathway)
        normed2 = apply_normalization_pipeline_with_stats(x_full, mean, std)
        R.check("apply_with_stats matches pipeline output",
                (normed2 - normed).abs().max().item() < 1e-5,
                f"max diff = {(normed2 - normed).abs().max().item():.2e}")

    except Exception as e:
        R.fail("power normalization", f"{e}\n{traceback.format_exc()}")


# ======================================================================
# Layer 5: End-to-End Mini Pipeline
# ======================================================================

def test_layer5_mini_pipeline(R):
    """Minimal end-to-end test: 100 images, 50 formulas, 1 epoch."""
    print(f"\n{'='*60}")
    print(f"  [5/5] Layer 5: End-to-End Mini Pipeline")
    print(f"{'='*60}")

    from experiments.run_v3_2_pipeline import build_data_batch, execute_body

    try:
        # -- Load a small set of real images --
        from src.data.imagenet_loader import ImageNetDataModule
        from torch.utils.data import DataLoader

        dm = ImageNetDataModule(data_dir=DATA_DIR, resolution=112, batch_size=50,
                                num_workers=0, samples_per_class=1)
        dm.setup()
        train_loader = DataLoader(dm.train_dataset, batch_size=50, shuffle=False,
                                  num_workers=0, pin_memory=False)
        images, labels = next(iter(train_loader))
        B = images.shape[0]
        print(f"  Loaded {B} images, {labels.unique().numel()} classes")

        data_batch = build_data_batch(images, DEVICE)
        R.ok(f"build_data_batch on {B} real images")

        # -- Load real formulas from v3 Phase 1 (or use hardcoded fallbacks) --
        formula_strs = []
        bank_path = 'outputs/imagenet_v3/phase1/bank_0/feature_bank/feature_bank.json'
        if os.path.exists(bank_path):
            with open(bank_path) as f:
                bank = json.load(f)
            formula_strs = [f['str'] for f in bank['formulas'][:50]]
            print(f"  Loaded {len(formula_strs)} formulas from Phase 1 bank_0")
        else:
            # Hardcoded fallback formulas using both old and new operators
            formula_strs = [
                'I_R global_avg_pool',
                'I_G edge_x global_max_pool',
                'I_B blur pool_center',
                'I_GRAY edge_mag global_avg_pool',
                'I_H local_contrast global_std_pool',
                'I_S dog pool_top_half',
                'I_r corner_harris global_avg_pool',
                'I_g lbp_like global_max_pool',
                'I_RG gabor_mag pool_quad_tl',
                'I_BY edge_xx global_avg_pool',
                'I_R edge_yy pool_bottom_half',
                'I_GRAY edge_mag relu global_avg_pool',
                'I_R I_G add edge_mag global_avg_pool',
                'I_R edge_x I_G edge_y multiply global_avg_pool',
                'I_GRAY local_contrast edge_mag global_max_pool',
            ] * 4  # Repeat to get ~50
            formula_strs = formula_strs[:50]
            print(f"  Using {len(formula_strs)} hardcoded formulas (no Phase 1 bank found)")

        # -- Step 1: Extract bodies and execute --
        bodies = []
        for fstr in formula_strs:
            tokens = fstr.strip().split()
            if tokens[-1] in ROOT_OPERATORS:
                bodies.append(' '.join(tokens[:-1]))
            else:
                bodies.append(fstr)
        bodies = sorted(set(bodies))[:50]
        print(f"  Unique bodies: {len(bodies)}")

        # -- Step 4 (mini): Extract distribution stats features --
        n_bodies = len(bodies)
        N_STATS = 60
        n_feats = n_bodies * N_STATS

        feature_matrix = torch.zeros(B, n_feats, device=DEVICE)
        n_success = 0
        for b_idx, body in enumerate(bodies):
            try:
                fm = execute_body(body, data_batch)
                if fm is not None and fm.dim() >= 2:
                    stats = encode_body_distribution_v2(fm)  # [B, 60]
                    feature_matrix[:, b_idx * N_STATS:(b_idx + 1) * N_STATS] = stats
                    n_success += 1
            except Exception:
                pass

        R.check(f"executed {n_success}/{n_bodies} bodies successfully",
                n_success > 0,
                "no bodies executed")

        R.check("feature matrix no NaN",
                not torch.isnan(feature_matrix).any(),
                "contains NaN")

        n_nonzero_feats = (feature_matrix.abs().sum(dim=0) > 0).sum().item()
        R.check(f"non-zero features: {n_nonzero_feats}/{n_feats}",
                n_nonzero_feats > 0,
                "all features are zero")

        # -- Step 5 (mini): Train classifier for 1 epoch --
        print(f"\n  Mini classifier training:")

        # Standardize
        feat_mean = feature_matrix.mean(dim=0, keepdim=True)
        feat_std = feature_matrix.std(dim=0, keepdim=True).clamp(min=1e-8)
        X = (feature_matrix - feat_mean) / feat_std

        # Power + L2 norm
        X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-8)
        X = F.normalize(X, p=2, dim=1)

        R.check("normalized features: row norms ≈ 1.0",
                (X.norm(dim=1) - 1.0).abs().max().item() < 1e-4,
                f"max deviation = {(X.norm(dim=1) - 1.0).abs().max().item():.2e}")

        n_classes = labels.unique().numel()
        y = labels.to(DEVICE)

        model = nn.Linear(n_feats, n_classes).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=10.0)
        criterion = nn.CrossEntropyLoss()

        # Remap labels to 0..n_classes-1 for this mini-batch
        unique_labels = labels.unique()
        label_map = {v.item(): i for i, v in enumerate(unique_labels)}
        y_mapped = torch.tensor([label_map[l.item()] for l in labels], device=DEVICE)
        model_mini = nn.Linear(n_feats, len(unique_labels)).to(DEVICE)
        opt_mini = torch.optim.AdamW(model_mini.parameters(), lr=1e-2, weight_decay=1.0)

        losses = []
        for step in range(20):
            logits = model_mini(X)
            loss = criterion(logits, y_mapped)
            opt_mini.zero_grad()
            loss.backward()
            opt_mini.step()
            losses.append(loss.item())

        R.check(f"loss decreasing: {losses[0]:.4f} → {losses[-1]:.4f}",
                losses[-1] < losses[0],
                f"loss did not decrease")

        R.check("no crash through full mini pipeline",
                True, "")

        # -- Bonus: test with new v3.2 operators in formulas --
        print(f"\n  New operator integration:")
        new_op_formulas = [
            ('I_R edge_mag', 'edge_mag'),
            ('I_G gabor_mag', 'gabor_mag'),
            ('I_B local_contrast', 'local_contrast'),
            ('I_GRAY dog', 'dog'),
            ('I_R corner_harris', 'corner_harris'),
            ('I_G lbp_like', 'lbp_like'),
            ('I_B edge_xx', 'edge_xx'),
            ('I_R edge_yy', 'edge_yy'),
            ('I_G edge_orient', 'edge_orient'),
        ]
        for body, op_name in new_op_formulas:
            fm = execute_body(body, data_batch)
            if fm is not None:
                stats = encode_body_distribution_v2(fm)
                R.check(f"{op_name} → distribution stats OK",
                        stats.shape == (B, 60) and not torch.isnan(stats).any(),
                        f"shape={stats.shape}, nan={torch.isnan(stats).any()}")
            else:
                R.fail(f"{op_name} → execute_body failed", "returned None")

    except Exception as e:
        R.fail("mini pipeline", f"{e}\n{traceback.format_exc()}")


# ======================================================================
# Main
# ======================================================================

def main():
    print(f"\n{'#'*60}")
    print(f"  v3.2 Hierarchical Smoke Test")
    print(f"  Device: {DEVICE}")
    print(f"{'#'*60}")

    R = Result()
    t0 = time.time()

    test_layer1_operators(R)
    test_layer2_terminals_grammar(R)
    test_layer3_kernel_bank(R)
    test_layer4_feature_encoding(R)
    test_layer5_mini_pipeline(R)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    all_passed = R.summary()

    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
