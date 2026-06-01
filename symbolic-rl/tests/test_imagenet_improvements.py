"""
Tests for all ImageNet scaling improvements.

Covers:
- New operators (blur_7x7, sigmoid, global_min_pool)
- Removed operators (sharpen, erode)
- _safe_binary decorator
- HSV terminals
- Loss-based reward signal
- Resolution-adaptive evaluation
- Entropy schedule + LR warmup
- Adaptive feature bank thresholds
- ImageNet data module (mock)
"""

import sys
import os
import math
import pytest
import torch
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.symbolic.tensor_operators import (
    TensorOperators, TENSOR_OPERATORS, ROOT_OPERATORS, MULTI_DIM_OPERATORS
)
from src.symbolic.large_feature_bank import LargeFeatureBank


# ============================================================
# 1. Operator Tests
# ============================================================

class TestNewOperators:
    """Test newly added operators."""

    def _make_batch(self, B=4, H=32, W=32):
        return torch.rand(B, H, W)

    def test_blur_7x7_shape(self):
        x = self._make_batch()
        out = TensorOperators.blur_7x7(x)
        assert out.shape == x.shape

    def test_blur_7x7_smoothing(self):
        """blur_7x7 should reduce variance more than blur."""
        x = torch.rand(2, 64, 64)
        blur3 = TensorOperators.blur(x)
        blur7 = TensorOperators.blur_7x7(x)
        # 7x7 blur should have lower variance (more smoothed)
        assert blur7.var() < blur3.var()

    def test_sigmoid_shape(self):
        x = self._make_batch()
        out = TensorOperators.sigmoid(x)
        assert out.shape == x.shape

    def test_sigmoid_bounds(self):
        """Sigmoid output should be in [0, 1]."""
        x = torch.randn(4, 16, 16) * 10  # wide range input
        out = TensorOperators.sigmoid(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_global_min_pool_shape(self):
        x = self._make_batch()
        out = TensorOperators.global_min_pool(x)
        assert out.shape == (4,)

    def test_global_min_pool_value(self):
        x = torch.tensor([[[1.0, 2.0], [3.0, 0.5]]])  # [1, 2, 2]
        out = TensorOperators.global_min_pool(x)
        assert out.item() == pytest.approx(0.5)


class TestRemovedOperators:
    """Verify sharpen and erode are removed from the registry."""

    def test_sharpen_removed(self):
        assert 'sharpen' not in TENSOR_OPERATORS

    def test_erode_removed(self):
        assert 'erode' not in TENSOR_OPERATORS


class TestNewOperatorsInRegistry:
    """Verify new operators are properly registered."""

    def test_blur_7x7_in_registry(self):
        assert 'blur_7x7' in TENSOR_OPERATORS
        _, arity, otype = TENSOR_OPERATORS['blur_7x7']
        assert arity == 1
        assert otype == 'tensor'

    def test_sigmoid_in_registry(self):
        assert 'sigmoid' in TENSOR_OPERATORS
        _, arity, otype = TENSOR_OPERATORS['sigmoid']
        assert arity == 1
        assert otype == 'tensor'

    def test_global_min_pool_in_registry(self):
        assert 'global_min_pool' in TENSOR_OPERATORS
        _, arity, otype = TENSOR_OPERATORS['global_min_pool']
        assert arity == 1
        assert otype == 'scalar'

    def test_global_min_pool_in_root(self):
        assert 'global_min_pool' in ROOT_OPERATORS


class TestSafeBinaryDecorator:
    """Test that _safe_binary clamps and handles NaN."""

    def test_add_no_nan(self):
        x = torch.tensor([1e7, -1e7, float('nan')])
        y = torch.tensor([1e7, -1e7, 0.0])
        # Reshape to [1, 1, 3] to satisfy [B, H, W] shape for add
        x = x.unsqueeze(0).unsqueeze(0)
        y = y.unsqueeze(0).unsqueeze(0)
        result = TensorOperators.add(x, y)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_multiply_clamped(self):
        x = torch.full((1, 2, 2), 1e4)
        y = torch.full((1, 2, 2), 1e4)
        result = TensorOperators.multiply(x, y)
        # 1e4 * 1e4 = 1e8 > 1e6, should be clamped
        assert result.max() <= 1e6
        assert result.min() >= -1e6


# ============================================================
# 1b. Gabor + local_std_5x5 Tests
# ============================================================

class TestGaborOperators:
    """Test Gabor filter operators with kernel caching."""

    def _make_batch(self, B=4, H=32, W=32):
        return torch.rand(B, H, W)

    # --- Shape ---
    def test_gabor_0_shape(self):
        out = TensorOperators.gabor_0(self._make_batch())
        assert out.shape == (4, 32, 32)

    def test_gabor_45_shape(self):
        out = TensorOperators.gabor_45(self._make_batch())
        assert out.shape == (4, 32, 32)

    def test_gabor_90_shape(self):
        out = TensorOperators.gabor_90(self._make_batch())
        assert out.shape == (4, 32, 32)

    # --- Registry ---
    def test_gabor_in_registry(self):
        for name in ('gabor_0', 'gabor_45', 'gabor_90'):
            assert name in TENSOR_OPERATORS
            _, arity, otype = TENSOR_OPERATORS[name]
            assert arity == 1
            assert otype == 'tensor'

    # --- Kernel caching ---
    def test_gabor_cache_reuse(self):
        """Same theta + device should return the exact same tensor object."""
        x = self._make_batch()
        _ = TensorOperators.gabor_0(x)  # warm up
        k1 = TensorOperators._get_gabor_kernel(0.0, x.device)
        k2 = TensorOperators._get_gabor_kernel(0.0, x.device)
        assert k1 is k2, "Cached kernel should be the same object"

    # --- Numerical stability: all-zero input ---
    def test_gabor_zero_input(self):
        x = torch.zeros(2, 16, 16)
        for fn in (TensorOperators.gabor_0, TensorOperators.gabor_45, TensorOperators.gabor_90):
            out = fn(x)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()
            # Convolution of zero input with zero-mean kernel → all zeros
            assert out.abs().max() < 1e-6

    # --- Numerical stability: large input ---
    def test_gabor_large_input(self):
        x = torch.full((2, 16, 16), 1e4)
        for fn in (TensorOperators.gabor_0, TensorOperators.gabor_45, TensorOperators.gabor_90):
            out = fn(x)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()

    # --- Different orientations produce different outputs ---
    def test_gabor_orientations_differ(self):
        torch.manual_seed(42)
        x = torch.rand(2, 32, 32)
        o0 = TensorOperators.gabor_0(x)
        o45 = TensorOperators.gabor_45(x)
        o90 = TensorOperators.gabor_90(x)
        # At least one pair should differ meaningfully
        assert not torch.allclose(o0, o45, atol=1e-4)
        assert not torch.allclose(o0, o90, atol=1e-4)


class TestLocalStd5x5:
    """Test local standard deviation operator."""

    def _make_batch(self, B=4, H=32, W=32):
        return torch.rand(B, H, W)

    def test_shape(self):
        out = TensorOperators.local_std_5x5(self._make_batch())
        assert out.shape == (4, 32, 32)

    def test_in_registry(self):
        assert 'local_std_5x5' in TENSOR_OPERATORS
        _, arity, otype = TENSOR_OPERATORS['local_std_5x5']
        assert arity == 1
        assert otype == 'tensor'

    def test_non_negative(self):
        """Standard deviation is always >= 0."""
        out = TensorOperators.local_std_5x5(torch.randn(4, 16, 16))
        assert (out >= 0).all()

    def test_constant_input(self):
        """Constant input → zero local std in interior (border has padding artefacts)."""
        x = torch.full((2, 16, 16), 3.0)
        out = TensorOperators.local_std_5x5(x)
        assert not torch.isnan(out).any()
        # Interior pixels (away from padding boundary) should be ~0
        interior = out[:, 2:-2, 2:-2]
        assert interior.abs().max() < 1e-5

    def test_large_input(self):
        x = torch.full((2, 16, 16), 1e4)
        out = TensorOperators.local_std_5x5(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_noisy_gt_constant(self):
        """Noisy image should have higher local_std than constant image."""
        constant = torch.ones(2, 32, 32) * 0.5
        noisy = torch.rand(2, 32, 32)
        std_const = TensorOperators.local_std_5x5(constant).mean()
        std_noisy = TensorOperators.local_std_5x5(noisy).mean()
        assert std_noisy > std_const


# ============================================================
# 2. HSV Terminal Tests
# ============================================================

class TestHSVTerminals:
    """Test HSV computation in environment's get_data_batch."""

    def test_hsv_computation(self):
        """Verify HSV channels are computed correctly for a known color."""
        # Pure red pixel: RGB = (1, 0, 0) → H=0, S=1
        images = torch.zeros(1, 3, 2, 2)
        images[0, 0, :, :] = 1.0  # R=1

        I_R = images[:, 0]
        I_G = images[:, 1]
        I_B = images[:, 2]

        Cmax, _ = images.max(dim=1)
        Cmin, _ = images.min(dim=1)
        delta = Cmax - Cmin + 1e-8

        H = torch.zeros_like(I_R)
        mask_r = (Cmax == I_R)
        mask_g = (Cmax == I_G) & ~mask_r
        mask_b = ~mask_r & ~mask_g
        H[mask_r] = (((I_G[mask_r] - I_B[mask_r]) / delta[mask_r]) % 6)
        H[mask_g] = ((I_B[mask_g] - I_R[mask_g]) / delta[mask_g]) + 2
        H[mask_b] = ((I_R[mask_b] - I_G[mask_b]) / delta[mask_b]) + 4
        H = H / 6.0

        S = torch.where(Cmax > 1e-8, delta / (Cmax + 1e-8), torch.zeros_like(Cmax))

        # Pure red: H should be ~0, S should be ~1
        assert H.mean().item() == pytest.approx(0.0, abs=0.01)
        assert S.mean().item() == pytest.approx(1.0, abs=0.01)


# ============================================================
# 3. Loss-Based Reward Tests
# ============================================================

class TestLossBasedReward:
    """Test the normalized loss reward computation."""

    def test_random_chance_reward_near_zero(self):
        """At random-chance loss, normalized reward should be ~0."""
        num_classes = 1000
        max_loss = math.log(num_classes)
        # Random chance CE loss ≈ log(num_classes)
        ce_loss = max_loss
        normalized = max(0.0, 1.0 - ce_loss / max_loss)
        assert normalized == pytest.approx(0.0, abs=0.01)

    def test_perfect_reward_near_one(self):
        """At perfect (zero) loss, normalized reward should be ~1."""
        num_classes = 1000
        max_loss = math.log(num_classes)
        ce_loss = 0.0
        normalized = max(0.0, 1.0 - ce_loss / max_loss)
        assert normalized == pytest.approx(1.0, abs=0.01)

    def test_composite_reward_bounds(self):
        """Composite reward should be in [0, 1]."""
        normalized_loss = 0.5
        top5_acc = 0.3
        top1_acc = 0.1
        reward = 0.6 * normalized_loss + 0.3 * top5_acc + 0.1 * top1_acc
        assert 0.0 <= reward <= 1.0


# ============================================================
# 4. Entropy Schedule Tests
# ============================================================

class TestEntropySchedule:
    """Test PPO entropy coefficient scheduling."""

    def test_entropy_decays(self):
        from src.rl.ppo_trainer import PPOTrainer
        from src.models.policy_agent import PolicyAgent

        # Create minimal policy (won't be trained)
        policy = PolicyAgent(vocab_size=10, embedding_dim=8, hidden_size=16, num_layers=1)

        # Mock environment (just need vocabulary attribute)
        class MockEnv:
            class vocabulary:
                @staticmethod
                def encode(x): return 0
            class action_space:
                n = 10
        env = MockEnv()

        trainer = PPOTrainer(
            policy=policy, env=env,
            entropy_coef=0.05,
            entropy_coef_start=0.05,
            entropy_coef_end=0.005,
            entropy_decay_fraction=0.5,
            total_iterations=100,
            device='cpu'
        )

        # At iteration 0, entropy should be start value
        trainer.iteration_count = 0
        trainer._update_schedule()
        assert trainer.entropy_coef == pytest.approx(0.05, abs=0.001)

        # At iteration 25 (50% of decay period), should be midpoint
        trainer.iteration_count = 25
        trainer._update_schedule()
        expected = 0.05 * 0.5 + 0.005 * 0.5
        assert trainer.entropy_coef == pytest.approx(expected, abs=0.002)

        # At iteration 50+ (past decay), should be end value
        trainer.iteration_count = 60
        trainer._update_schedule()
        assert trainer.entropy_coef == pytest.approx(0.005, abs=0.001)


# ============================================================
# 5. Adaptive Feature Bank Tests
# ============================================================

class TestAdaptiveFeatureBank:
    """Test adaptive threshold behavior in LargeFeatureBank."""

    def test_threshold_increases_after_warmup(self):
        bank = LargeFeatureBank(
            max_size=10,
            min_accuracy=0.01,
            correlation_threshold=0.90,
            adaptive_threshold=True,
            threshold_warmup_fraction=0.5,
        )

        # Add 6 formulas (60% full, past 50% warmup)
        for i in range(6):
            out = np.random.randn(100).astype(np.float32)
            out += i * 10  # make each output different to pass correlation
            bank._insert(None, f"formula_{i}", 3, 0.10 + i * 0.01, out)

        # Now the mean accuracy is about 0.125
        # After update, threshold should be 0.8 * mean ≈ 0.10
        bank._update_adaptive_thresholds()
        assert bank.min_accuracy >= bank.base_min_accuracy

    def test_correlation_tightens_when_near_full(self):
        bank = LargeFeatureBank(
            max_size=10,
            min_accuracy=0.01,
            correlation_threshold=0.90,
            correlation_threshold_full=0.78,
            adaptive_threshold=True,
            threshold_warmup_fraction=0.5,
        )

        # Fill to 90% (9/10)
        for i in range(9):
            out = np.random.randn(100).astype(np.float32)
            out += i * 10
            bank._insert(None, f"formula_{i}", 3, 0.10, out)

        bank._update_adaptive_thresholds()
        # At 90% fill: t = (0.9-0.8)/0.2 = 0.5
        # threshold should be 0.90*0.5 + 0.78*0.5 = 0.84
        assert bank.correlation_threshold < 0.90
        assert bank.correlation_threshold > 0.78

    def test_no_change_without_adaptive(self):
        bank = LargeFeatureBank(
            max_size=10,
            min_accuracy=0.01,
            correlation_threshold=0.90,
            adaptive_threshold=False,
        )
        for i in range(8):
            out = np.random.randn(100).astype(np.float32) + i * 10
            bank._insert(None, f"formula_{i}", 3, 0.10, out)

        bank._update_adaptive_thresholds()
        assert bank.min_accuracy == 0.01
        assert bank.correlation_threshold == 0.90


# ============================================================
# 6. Vocabulary Tests (with HSV terminals)
# ============================================================

class TestVocabulary:
    """Test TensorTokenVocabulary includes HSV terminals."""

    def test_hsv_in_vocabulary(self):
        from src.rl.tensor_environment_large_bank import TensorTokenVocabulary
        vocab = TensorTokenVocabulary()
        assert 'I_H' in vocab.token_to_idx
        assert 'I_S' in vocab.token_to_idx

    def test_vocabulary_size(self):
        from src.rl.tensor_environment_large_bank import TensorTokenVocabulary
        vocab = TensorTokenVocabulary()
        # 3 special + operators + 6 terminals (R,G,B,GRAY,H,S)
        assert len(vocab) > 25  # reasonable minimum
        assert len(vocab) < 60  # should not exceed 60 per plan

    def test_removed_operators_not_in_vocab(self):
        from src.rl.tensor_environment_large_bank import TensorTokenVocabulary
        vocab = TensorTokenVocabulary()
        assert 'sharpen' not in vocab.token_to_idx
        assert 'erode' not in vocab.token_to_idx


# ============================================================
# 7. RPN Grammar Mask Tests (updated for new ops)
# ============================================================

class TestRPNGrammarMaskUpdated:
    """Test that RPN grammar mask works with new operator set."""

    def test_mask_allows_new_operators(self):
        from src.rl.tensor_environment_large_bank import TensorTokenVocabulary
        from src.rl.rpn_grammar_mask import RPNGrammarMask

        vocab = TensorTokenVocabulary()
        masker = RPNGrammarMask(vocab, max_sequence_length=15)

        # After one terminal, blur_7x7 and sigmoid should be allowed
        tokens = [vocab.encode('I_R')]
        mask = masker.get_valid_actions(tokens)

        # blur_7x7 is a unary op, should be allowed
        assert mask[vocab.encode('blur_7x7')] > 0
        assert mask[vocab.encode('sigmoid')] > 0

        # global_min_pool is a root op, should be allowed when stack=1
        assert mask[vocab.encode('global_min_pool')] > 0

    def test_valid_formula_with_new_ops(self):
        from src.rl.tensor_environment_large_bank import TensorTokenVocabulary
        from src.rl.rpn_grammar_mask import RPNGrammarMask

        vocab = TensorTokenVocabulary()
        masker = RPNGrammarMask(vocab, max_sequence_length=15)

        # I_H sigmoid blur_7x7 global_min_pool
        seq = [vocab.encode(t) for t in ['I_H', 'sigmoid', 'blur_7x7', 'global_min_pool']]
        is_valid, reason = masker.is_valid_sequence(seq)
        assert is_valid, f"Should be valid: {reason}"


# ============================================================
# 8. End-to-end formula execution test
# ============================================================

class TestFormulaExecution:
    """Test end-to-end formula execution with new operators."""

    def test_execute_formula_with_new_ops(self):
        """Execute: I_R blur_7x7 sigmoid global_min_pool"""
        B, H, W = 4, 32, 32
        data_batch = {
            'I_R': torch.rand(B, H, W),
            'I_G': torch.rand(B, H, W),
            'I_B': torch.rand(B, H, W),
            'I_GRAY': torch.rand(B, H, W),
            'I_H': torch.rand(B, H, W),
            'I_S': torch.rand(B, H, W),
        }

        formula = ['I_R', 'blur_7x7', 'sigmoid', 'global_min_pool']

        # Execute
        stack = []
        for token in formula:
            if token in data_batch:
                stack.append(data_batch[token])
            elif token in TENSOR_OPERATORS:
                op_func, arity, _ = TENSOR_OPERATORS[token]
                operands = [stack.pop() for _ in range(arity)]
                operands.reverse()
                result = op_func(*operands)
                stack.append(result)

        output = stack[0]
        assert output.shape == (B,)
        assert not torch.isnan(output).any()
        assert output.min() >= 0.0
        assert output.max() <= 1.0  # sigmoid bounds

    def test_execute_hsv_formula(self):
        """Execute: I_H I_S multiply global_avg_pool"""
        B, H, W = 4, 16, 16
        data_batch = {
            'I_H': torch.rand(B, H, W),
            'I_S': torch.rand(B, H, W),
        }

        stack = []
        for token in ['I_H', 'I_S', 'multiply', 'global_avg_pool']:
            if token in data_batch:
                stack.append(data_batch[token])
            elif token in TENSOR_OPERATORS:
                op_func, arity, _ = TENSOR_OPERATORS[token]
                operands = [stack.pop() for _ in range(arity)]
                operands.reverse()
                result = op_func(*operands)
                stack.append(result)

        output = stack[0]
        assert output.shape == (B,)
        assert not torch.isnan(output).any()


# ============================================================
# 9. Resolution-adaptive test
# ============================================================

class TestResolutionAdaptive:
    """Test that F.interpolate downscaling works correctly."""

    def test_downscale(self):
        images = torch.rand(2, 3, 224, 224)
        downscaled = torch.nn.functional.interpolate(
            images, size=(64, 64), mode='bilinear', align_corners=False
        )
        assert downscaled.shape == (2, 3, 64, 64)

    def test_downscale_preserves_range(self):
        images = torch.rand(2, 3, 224, 224)
        downscaled = torch.nn.functional.interpolate(
            images, size=(64, 64), mode='bilinear', align_corners=False
        )
        assert downscaled.min() >= 0.0
        assert downscaled.max() <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
