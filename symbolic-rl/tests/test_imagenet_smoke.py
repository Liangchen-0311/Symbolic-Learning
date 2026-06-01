"""
Smoke test for the full ImageNet symbolic RL pipeline.

Validates end-to-end:
  1. Data loading (10 images/class, 64x64 resolution)
  2. All 46 operators (27 non-root + 19 root) execute without NaN/Inf
  3. HSV channels (I_H, I_S) are correctly computed
  4. Loss-based reward signal returns valid composite reward
  5. Hierarchical (superclass) evaluation works
  6. Cross-scale binary ops (downsample + add/sub/mul/div) are stable
  7. Feature bank admission works
  8. 50 RL iterations complete without crash

Usage:
    python -m tests.test_imagenet_smoke
"""

import os
import sys
import math
import time
import traceback

import torch
import numpy as np
import yaml

# ── Setup ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.symbolic.tensor_operators import TENSOR_OPERATORS, ROOT_OPERATORS
from src.data.imagenet_loader import ImageNetDataModule, build_imagenet_superclass_mapping
from src.rl.tensor_environment_large_bank import TensorVSREnvironmentLargeBank, TensorTokenVocabulary
from src.models.policy_agent import PolicyAgent
from src.rl.ppo_trainer import PPOTrainer


# ── Helpers ────────────────────────────────────────────────────
class Result:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name):
        self.passed += 1
        print(f"  [PASS] {name}")

    def fail(self, name, msg):
        self.failed += 1
        self.errors.append((name, msg))
        print(f"  [FAIL] {name}: {msg}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print("\nFailures:")
            for name, msg in self.errors:
                print(f"  - {name}: {msg}")
        print(f"{'='*60}")
        return self.failed == 0


R = Result()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "/workspace/neurosymbolic-rl/data/imagenet"
SAMPLES_PER_CLASS = 10
RESOLUTION = 64
N_ITERATIONS = 50
EPISODES_PER_ITER = 5


# ── Test 1: Operator Registry ─────────────────────────────────
def test_operator_registry():
    print("\n[1/8] Operator Registry")

    non_root = {k for k in TENSOR_OPERATORS if k not in ROOT_OPERATORS}
    n_non_root = len(non_root)
    n_root = len(ROOT_OPERATORS)
    n_total = len(TENSOR_OPERATORS)

    if n_non_root == 27:
        R.ok(f"Non-root operators = {n_non_root}")
    else:
        R.fail(f"Non-root operators", f"expected 27, got {n_non_root}")

    if n_root == 19:
        R.ok(f"Root operators = {n_root}")
    else:
        R.fail(f"Root operators", f"expected 19, got {n_root}")

    # Check key new operators exist
    expected_new = [
        'div', 'negate', 'pow2', 'sqrt_abs', 'log1p_abs',
        'flip_h', 'flip_v', 'downsample_2x', 'downsample_4x', 'stride_pool_4',
        'pool_thirds_top', 'pool_thirds_mid', 'pool_thirds_bot',
        'pool_quad_tl', 'pool_quad_tr', 'pool_quad_bl', 'pool_quad_br',
        'pool_surround',
    ]
    missing = [op for op in expected_new if op not in TENSOR_OPERATORS]
    if not missing:
        R.ok(f"All 18 new operators registered")
    else:
        R.fail("New operators", f"missing: {missing}")


# ── Test 2: All Operators NaN/Inf Free ─────────────────────────
def test_operators_no_nan():
    print("\n[2/8] Operator NaN/Inf Stability")

    test_inputs = {
        'random': torch.randn(8, 64, 64, device=DEVICE),
        'zeros': torch.zeros(8, 64, 64, device=DEVICE),
        'large': torch.randn(8, 64, 64, device=DEVICE) * 1000,
        'tiny': torch.randn(8, 64, 64, device=DEVICE) * 1e-7,
    }

    failures = []
    for input_name, x in test_inputs.items():
        y = torch.randn_like(x)
        for op_name, (func, arity, otype) in TENSOR_OPERATORS.items():
            try:
                if arity == 2:
                    out = func(x, y)
                else:
                    out = func(x)
                if torch.isnan(out).any():
                    failures.append(f"{op_name}({input_name}): NaN")
                if torch.isinf(out).any():
                    failures.append(f"{op_name}({input_name}): Inf")
            except Exception as e:
                failures.append(f"{op_name}({input_name}): {e}")

    if not failures:
        R.ok(f"All {len(TENSOR_OPERATORS)} operators × 4 input types = no NaN/Inf")
    else:
        for f in failures[:5]:
            R.fail("Operator stability", f)
        if len(failures) > 5:
            R.fail("Operator stability", f"...and {len(failures)-5} more")


# ── Test 3: Cross-Scale Binary Ops ────────────────────────────
def test_cross_scale_binary():
    print("\n[3/8] Cross-Scale Binary Operations")

    x_full = torch.randn(8, 64, 64, device=DEVICE)
    x_half = TENSOR_OPERATORS['downsample_2x'][0](x_full)   # [8, 32, 32]
    x_quarter = TENSOR_OPERATORS['downsample_4x'][0](x_full) # [8, 16, 16]

    combos = [
        ('add', x_full, x_half, "64+32"),
        ('subtract', x_full, x_quarter, "64-16"),
        ('multiply', x_half, x_quarter, "32*16"),
        ('div', x_full, x_half, "64/32"),
    ]
    for op_name, a, b, label in combos:
        func = TENSOR_OPERATORS[op_name][0]
        try:
            out = func(a, b)
            if torch.isnan(out).any() or torch.isinf(out).any():
                R.fail(f"cross-scale {op_name} {label}", "NaN/Inf")
            else:
                R.ok(f"cross-scale {op_name} {label} -> {list(out.shape)}")
        except Exception as e:
            R.fail(f"cross-scale {op_name} {label}", str(e))


# ── Test 4: Vocabulary ────────────────────────────────────────
def test_vocabulary():
    print("\n[4/8] Vocabulary")

    vocab = TensorTokenVocabulary()
    if len(vocab) == 55:
        R.ok(f"Vocabulary size = {len(vocab)}")
    else:
        R.fail("Vocabulary size", f"expected 55, got {len(vocab)}")

    # HSV terminals present
    for t in ['I_H', 'I_S']:
        if t in vocab.token_to_idx:
            R.ok(f"Terminal '{t}' in vocabulary")
        else:
            R.fail(f"Terminal '{t}'", "not found")

    # Roundtrip
    for i in range(len(vocab)):
        token = vocab.decode(i)
        assert vocab.encode(token) == i
    R.ok("Encode/decode roundtrip all tokens")


# ── Test 5: Data Loading + HSV ────────────────────────────────
def test_data_loading():
    print("\n[5/8] Data Loading & HSV Channels")

    # Check train dir has enough data
    train_dir = os.path.join(DATA_DIR, 'train')
    if not os.path.isdir(train_dir):
        R.fail("ImageNet train dir", f"not found: {train_dir}")
        return None

    n_classes = len([d for d in os.listdir(train_dir)
                     if os.path.isdir(os.path.join(train_dir, d))])
    if n_classes < 10:
        R.fail("ImageNet classes", f"only {n_classes} found, need >= 10")
        return None
    R.ok(f"Found {n_classes} classes in {train_dir}")

    # Load data module — but we need val dir too; if missing, symlink train as val
    val_dir = os.path.join(DATA_DIR, 'val')
    created_val_symlink = False
    if not os.path.isdir(val_dir):
        os.symlink(train_dir, val_dir)
        created_val_symlink = True
        print(f"  (Created temp val -> train symlink for smoke test)")

    try:
        dm = ImageNetDataModule(
            data_dir=DATA_DIR,
            resolution=RESOLUTION,
            batch_size=32,
            num_workers=2,
            samples_per_class=SAMPLES_PER_CLASS,
        )
        dm.setup()
        R.ok(f"ImageNetDataModule loaded: {len(dm.train_dataset)} images")

        loader = dm.get_train_loader()
        images, labels = next(iter(loader))
        images = images.to(DEVICE)

        # Check shape
        B, C, H, W = images.shape
        assert C == 3, f"Expected 3 channels, got {C}"
        assert H == RESOLUTION and W == RESOLUTION
        R.ok(f"Batch shape: {list(images.shape)}, range [{images.min():.3f}, {images.max():.3f}]")

        # HSV check
        I_R = images[:, 0]
        I_G = images[:, 1]
        I_B = images[:, 2]
        Cmax, _ = images.max(dim=1)
        Cmin, _ = images.min(dim=1)
        delta = Cmax - Cmin + 1e-8
        S = torch.where(Cmax > 1e-8, delta / (Cmax + 1e-8), torch.zeros_like(Cmax))

        if S.min() >= 0 and S.max() <= 1.01:
            R.ok(f"Saturation range [{S.min():.3f}, {S.max():.3f}]")
        else:
            R.fail("Saturation range", f"[{S.min()}, {S.max()}]")

        return dm, loader, created_val_symlink
    except Exception as e:
        R.fail("Data loading", f"{e}")
        traceback.print_exc()
        return None


# ── Test 6: Environment + Loss-Based Reward + Hierarchical ────
def test_environment(dm, loader):
    print("\n[6/8] Environment (loss-based reward + hierarchical eval)")

    config = {
        'model': {
            'max_depth': 5,
            'max_sequence_length': 12,
            'embedding_dim': 64,
            'hidden_size': 128,
            'num_layers': 1,
            'dropout': 0.1,
        },
        'training': {
            'batch_size': 32,
            'eval_batch_size': 64,
            'episodes_per_iteration': EPISODES_PER_ITER,
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.05,
            'max_grad_norm': 0.5,
            'n_epochs_ppo': 2,
            'batch_size_ppo': 32,
            'length_penalty': 0.01,
            'binary_op_bias': 0.3,
        },
        'dataset_options': {
            'num_classes': 1000,
            'resolution_quick': RESOLUTION,
            'resolution_full': 224,
        },
        'strategy': {
            'feature_bank_size': 100,
            'min_accuracy_threshold': 0.001,  # Very low for smoke test
            'correlation_threshold': 0.95,
            'correlation_threshold_full': 0.90,
            'adaptive_threshold': False,
            'reward_type': 'loss_based',
            'use_hierarchical_eval': True,
            'hierarchical_switch_fraction': 0.5,
            'diversity_penalty': 0.15,
            'l1_lambda': 0.0,
            'lasso_epochs': 10,
        },
        'exclude_operators': [],
    }

    try:
        env = TensorVSREnvironmentLargeBank(
            data_loader=loader,
            config=config,
            device=DEVICE,
        )
        R.ok(f"Environment created (vocab={len(env.vocabulary)}, bank_cap={env.feature_bank.max_size})")

        # Set hierarchical eval
        # train_dataset may be a Subset after stratified sampling — get the underlying dataset
        base_dataset = dm.train_dataset
        if hasattr(base_dataset, 'dataset'):
            base_dataset = base_dataset.dataset
        superclass_map = build_imagenet_superclass_mapping(base_dataset)
        env.set_superclass_mapping(superclass_map, num_superclasses=20)
        R.ok("Superclass mapping set (20 superclasses)")

        # Test get_data_batch with HSV
        data_batch, labels = env.get_data_batch(batch_size=16, resolution=RESOLUTION)
        for key in ['I_R', 'I_G', 'I_B', 'I_GRAY', 'I_H', 'I_S']:
            if key not in data_batch:
                R.fail(f"data_batch['{key}']", "missing")
            else:
                t = data_batch[key]
                if torch.isnan(t).any() or torch.isinf(t).any():
                    R.fail(f"data_batch['{key}']", "NaN/Inf")
                else:
                    pass  # OK silently
        R.ok("All 6 terminal channels (RGB+GRAY+HSV) present, no NaN/Inf")

        # Test one reset/step cycle
        obs, info = env.reset()
        assert obs.shape == (config['model']['max_sequence_length'],)
        mask = env.get_action_mask()
        assert mask.shape[-1] == len(env.vocabulary)
        R.ok("reset() + get_action_mask() OK")

        # Manually execute a simple formula: I_R global_avg_pool END
        obs, info = env.reset()
        ir_idx = env.vocabulary.encode('I_R')
        gap_idx = env.vocabulary.encode('global_avg_pool')
        end_idx = env.vocabulary.encode('END')

        obs, r1, term, trunc, info1 = env.step(ir_idx)
        assert not term and not trunc
        obs, r2, term, trunc, info2 = env.step(gap_idx)
        assert not term and not trunc
        obs, r3, term, trunc, info3 = env.step(end_idx)
        assert term  # should terminate

        if info3.get('valid', False):
            reward = info3.get('reward', r3)
            R.ok(f"Formula 'I_R global_avg_pool': reward={reward:.4f}, "
                 f"acc={info3.get('accuracy', 0):.4f}, "
                 f"top5={info3.get('top5_accuracy', 0):.4f}")
        else:
            R.fail("Simple formula", f"invalid: {info3.get('reason', 'unknown')}")

        return env, config
    except Exception as e:
        R.fail("Environment", f"{e}")
        traceback.print_exc()
        return None, None


# ── Test 7: All New Operators Execute in Formula Context ──────
def test_new_operators_in_formulas(env):
    print("\n[7/8] New Operators in Formula Context")

    data_batch, labels = env.get_data_batch(batch_size=16, resolution=RESOLUTION)

    # Test formulas using each new operator
    test_formulas = [
        # Arithmetic
        ("I_R I_G div global_avg_pool", "div (color ratio)"),
        # Pointwise
        ("I_R negate global_avg_pool", "negate"),
        ("I_R pow2 global_avg_pool", "pow2"),
        ("I_R edge_x sqrt_abs global_avg_pool", "sqrt_abs"),
        ("I_R gabor_0 log1p_abs global_avg_pool", "log1p_abs"),
        # Geometric
        ("I_R I_R flip_h subtract global_avg_pool", "flip_h symmetry"),
        ("I_G I_G flip_v subtract global_avg_pool", "flip_v symmetry"),
        # Multi-scale
        ("I_R downsample_2x edge_x global_avg_pool", "downsample_2x + edge"),
        ("I_B downsample_4x laplacian global_avg_pool", "downsample_4x + laplacian"),
        ("I_GRAY stride_pool_4 global_max_pool", "stride_pool_4"),
        # Cross-scale binary
        ("I_R downsample_2x I_G downsample_2x subtract global_avg_pool",
         "cross-channel downsample subtract"),
        # HSV
        ("I_H global_avg_pool", "Hue channel"),
        ("I_S edge_x global_std_pool", "Saturation edges"),
        # New spatial pooling
        ("I_R edge_x pool_thirds_top", "pool_thirds_top"),
        ("I_G blur pool_thirds_mid", "pool_thirds_mid"),
        ("I_B pool_thirds_bot", "pool_thirds_bot"),
        ("I_R gabor_0 pool_quad_tl", "pool_quad_tl"),
        ("I_G pool_quad_tr", "pool_quad_tr"),
        ("I_B pool_quad_bl", "pool_quad_bl"),
        ("I_GRAY pool_quad_br", "pool_quad_br"),
        ("I_R edge_x pool_surround", "pool_surround"),
    ]

    n_pass = 0
    for formula_str, desc in test_formulas:
        tokens = [env.vocabulary.encode(t) for t in formula_str.split()]
        try:
            output, is_valid = env._execute_formula(tokens, data_batch)
            if not is_valid or output is None:
                R.fail(desc, "returned invalid")
            elif torch.isnan(output).any() or torch.isinf(output).any():
                R.fail(desc, "NaN/Inf in output")
            else:
                n_pass += 1
        except Exception as e:
            R.fail(desc, str(e))

    if n_pass == len(test_formulas):
        R.ok(f"All {n_pass} test formulas executed cleanly")
    else:
        print(f"  ({n_pass}/{len(test_formulas)} passed)")


# ── Test 8: Full RL Training (50 iterations) ──────────────────
def test_rl_training(env, config):
    print(f"\n[8/8] Full RL Training ({N_ITERATIONS} iterations × {EPISODES_PER_ITER} episodes)")

    try:
        policy = PolicyAgent(
            vocab_size=len(env.vocabulary),
            embedding_dim=config['model']['embedding_dim'],
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
        ).to(DEVICE)
        R.ok(f"PolicyAgent created ({sum(p.numel() for p in policy.parameters()):,} params)")

        trainer = PPOTrainer(
            policy=policy,
            env=env,
            learning_rate=config['training']['learning_rate'],
            gamma=config['training']['gamma'],
            gae_lambda=config['training']['gae_lambda'],
            clip_epsilon=config['training']['clip_epsilon'],
            value_coef=config['training']['value_coef'],
            entropy_coef=config['training']['entropy_coef'],
            max_grad_norm=config['training']['max_grad_norm'],
            n_epochs=config['training']['n_epochs_ppo'],
            batch_size=config['training']['batch_size_ppo'],
            device=DEVICE,
            entropy_coef_start=0.05,
            entropy_coef_end=0.005,
            entropy_decay_fraction=0.5,
            lr_warmup_iterations=10,
            total_iterations=N_ITERATIONS,
        )

        # Set binary op bias
        trainer.set_binary_op_bias(
            config['training'].get('binary_op_bias', 0.0),
            env.vocabulary
        )

        R.ok("PPOTrainer created")

        # Run iterations
        t0 = time.time()
        all_rewards = []
        nan_count = 0

        for i in range(N_ITERATIONS):
            metrics = trainer.update(n_episodes=EPISODES_PER_ITER)

            avg_reward = metrics.get('avg_reward', 0.0)
            all_rewards.append(avg_reward)

            # Check for NaN in metrics
            for k, v in metrics.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    nan_count += 1
                    if nan_count <= 3:
                        print(f"    WARNING: iter {i}, {k} = {v}")

            # Update hierarchical state
            env.update_hierarchical_state()

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                bank_sz = env.feature_bank.size()
                entropy = metrics.get('entropy_coef', metrics.get('entropy', 0))
                print(f"    iter {i+1:3d}/{N_ITERATIONS} | "
                      f"reward={avg_reward:+.4f} | "
                      f"bank={bank_sz} | "
                      f"entropy_coef={entropy:.4f} | "
                      f"{elapsed:.1f}s")

        elapsed = time.time() - t0
        bank_size = env.feature_bank.size()

        R.ok(f"Completed {N_ITERATIONS} iterations in {elapsed:.1f}s")

        if nan_count == 0:
            R.ok("No NaN/Inf in any training metrics")
        else:
            R.fail("Training metrics", f"{nan_count} NaN/Inf values detected")

        if bank_size > 0:
            R.ok(f"Feature bank: {bank_size} formulas discovered")
        else:
            R.fail("Feature bank", "empty after training (0 formulas)")

        # Check reward trend (should not be all -1)
        mean_first_10 = np.mean(all_rewards[:10])
        mean_last_10 = np.mean(all_rewards[-10:])
        print(f"    Reward trend: first 10 avg={mean_first_10:.4f}, last 10 avg={mean_last_10:.4f}")
        if mean_last_10 > -0.9:
            R.ok(f"Reward not collapsed (last 10 avg = {mean_last_10:.4f})")
        else:
            R.fail("Reward", f"collapsed to {mean_last_10:.4f}")

    except Exception as e:
        R.fail("RL Training", f"{e}")
        traceback.print_exc()


# ── Main ──────────────────────────────────────────────────────
def main():
    print(f"{'='*60}")
    print(f"ImageNet Pipeline Smoke Test")
    print(f"{'='*60}")
    print(f"Device: {DEVICE}")
    print(f"Data:   {DATA_DIR}")
    print(f"Config: {SAMPLES_PER_CLASS} img/class, {RESOLUTION}x{RESOLUTION}, "
          f"{N_ITERATIONS} iters × {EPISODES_PER_ITER} eps")

    test_operator_registry()
    test_operators_no_nan()
    test_cross_scale_binary()
    test_vocabulary()

    result = test_data_loading()
    if result is None:
        print("\n[ABORT] Cannot proceed without data.")
        R.summary()
        sys.exit(1)

    dm, loader, created_val_symlink = result

    env_result = test_environment(dm, loader)
    if env_result[0] is None:
        print("\n[ABORT] Cannot proceed without environment.")
        R.summary()
        sys.exit(1)

    env, config = env_result
    test_new_operators_in_formulas(env)
    test_rl_training(env, config)

    # Cleanup temp symlink
    if created_val_symlink:
        val_dir = os.path.join(DATA_DIR, 'val')
        if os.path.islink(val_dir):
            os.unlink(val_dir)
            print("\n  (Removed temp val symlink)")

    success = R.summary()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
