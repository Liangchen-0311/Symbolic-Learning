"""
Test tensor operators on CIFAR images.

This script validates:
1. Tensor operators work on [batch, H, W] inputs
2. Output shapes are correct
3. Root operators reduce to [batch]
4. Gradients flow properly
5. LASSO classifier works on extracted features
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.symbolic.tensor_operators import TensorOperators, TENSOR_OPERATORS, ROOT_OPERATORS
from src.models.lasso_classifier import train_lasso_classifier
from src.data.mnist_loader import MNISTDataModule


def test_element_wise_ops():
    """Test element-wise operations."""
    print("\n" + "=" * 60)
    print("TEST 1: Element-wise Operations")
    print("=" * 60)

    batch_size = 4
    H, W = 32, 32

    x = torch.randn(batch_size, H, W)
    y = torch.randn(batch_size, H, W)

    # Test binary ops (divide was never in the registry)
    ops = ['add', 'subtract', 'multiply']
    for op_name in ops:
        op_func, arity, output_type = TENSOR_OPERATORS[op_name]
        result = op_func(x, y)
        assert result.shape == (batch_size, H, W), f"{op_name} shape mismatch"
        assert not torch.isnan(result).any(), f"{op_name} produced NaN"
        print(f"✓ {op_name}: {x.shape} -> {result.shape}")

    print("\nElement-wise ops PASSED!")


def test_activations():
    """Test non-linear activations."""
    print("\n" + "=" * 60)
    print("TEST 2: Non-linear Activations")
    print("=" * 60)

    batch_size = 4
    H, W = 32, 32

    x = torch.randn(batch_size, H, W)

    ops = ['relu', 'sigmoid', 'abs']
    for op_name in ops:
        op_func, arity, output_type = TENSOR_OPERATORS[op_name]
        result = op_func(x)
        assert result.shape == (batch_size, H, W), f"{op_name} shape mismatch"
        assert not torch.isnan(result).any(), f"{op_name} produced NaN"
        print(f"✓ {op_name}: {x.shape} -> {result.shape}")

    print("\nActivation ops PASSED!")


def test_spatial_ops():
    """Test spatial/structural operators."""
    print("\n" + "=" * 60)
    print("TEST 3: Spatial Operators")
    print("=" * 60)

    batch_size = 4
    H, W = 32, 32

    x = torch.randn(batch_size, H, W)

    ops = ['blur', 'blur_7x7', 'edge_x', 'edge_y', 'dilate', 'laplacian', 'normalize']
    for op_name in ops:
        op_func, arity, output_type = TENSOR_OPERATORS[op_name]
        result = op_func(x)
        assert result.shape == (batch_size, H, W), f"{op_name} shape mismatch: {result.shape}"
        assert not torch.isnan(result).any(), f"{op_name} produced NaN"
        print(f"✓ {op_name}: {x.shape} -> {result.shape}")

    print("\nSpatial ops PASSED!")


def test_pooling_ops():
    """Test pooling operators (root-only)."""
    print("\n" + "=" * 60)
    print("TEST 4: Pooling Operators (Root-only)")
    print("=" * 60)

    batch_size = 4
    H, W = 32, 32

    x = torch.randn(batch_size, H, W)

    ops = ['global_avg_pool', 'global_max_pool', 'global_std_pool', 'global_l2_pool']
    for op_name in ops:
        op_func, arity, output_type = TENSOR_OPERATORS[op_name]
        result = op_func(x)
        assert result.shape == (batch_size,), f"{op_name} shape mismatch: {result.shape}"
        assert not torch.isnan(result).any(), f"{op_name} produced NaN"
        assert output_type == 'scalar', f"{op_name} should output scalar"
        print(f"✓ {op_name}: {x.shape} -> {result.shape}")

    print("\nPooling ops PASSED!")


def test_math_ops():
    """Test mathematical operations."""
    print("\n" + "=" * 60)
    print("TEST 5: Mathematical Operations")
    print("=" * 60)

    batch_size = 4
    H, W = 32, 32

    x = torch.randn(batch_size, H, W)

    # Test all unary non-root operators that are in the current registry
    ops = [name for name, (_, arity, otype) in TENSOR_OPERATORS.items()
           if arity == 1 and otype == 'tensor']
    for op_name in ops:
        op_func, arity, output_type = TENSOR_OPERATORS[op_name]
        result = op_func(x)
        assert result.shape == (batch_size, H, W), f"{op_name} shape mismatch: {result.shape}"
        assert not torch.isnan(result).any(), f"{op_name} produced NaN"
        print(f"✓ {op_name}: {x.shape} -> {result.shape}")

    print("\nUnary ops PASSED!")


def test_gradients():
    """Test gradient flow through operators."""
    print("\n" + "=" * 60)
    print("TEST 6: Gradient Flow")
    print("=" * 60)

    batch_size = 4
    H, W = 32, 32

    x = torch.randn(batch_size, H, W, requires_grad=True)
    y = torch.randn(batch_size, H, W, requires_grad=True)

    # Test gradient through spatial ops
    blurred = TensorOperators.blur(x)
    edge = TensorOperators.edge_x(x)
    pooled = TensorOperators.global_avg_pool(edge)

    loss = pooled.sum()
    loss.backward()

    assert x.grad is not None, "No gradient for x"
    assert not torch.isnan(x.grad).any(), "Gradient contains NaN"
    print(f"✓ Gradient shape: {x.grad.shape}")
    print(f"✓ Gradient norm: {x.grad.norm():.4f}")

    print("\nGradient flow PASSED!")


def test_on_real_images():
    """Test on real CIFAR-10 images."""
    print("\n" + "=" * 60)
    print("TEST 7: Real CIFAR-10 Images")
    print("=" * 60)

    # Load CIFAR-10
    data_module = MNISTDataModule(
        dataset='cifar10',
        batch_size=16,
        num_workers=0
    )
    data_module.setup()

    train_loader = data_module.get_train_loader()
    images, labels = next(iter(train_loader))

    print(f"Loaded batch: {images.shape}, labels: {labels.shape}")

    # Extract RGB channels
    I_R = images[:, 0, :, :]  # [batch, 32, 32]
    I_G = images[:, 1, :, :]
    I_B = images[:, 2, :, :]

    print(f"R channel: {I_R.shape}, range: [{I_R.min():.2f}, {I_R.max():.2f}]")
    print(f"G channel: {I_G.shape}, range: [{I_G.min():.2f}, {I_G.max():.2f}]")
    print(f"B channel: {I_B.shape}, range: [{I_B.min():.2f}, {I_B.max():.2f}]")

    # Test formula: global_avg_pool(blur(add(I_R, I_G)))
    print("\nTesting formula: global_avg_pool(blur(add(I_R, I_G)))")

    # Step 1: add(I_R, I_G)
    added = TensorOperators.add(I_R, I_G)
    print(f"  add(I_R, I_G): {added.shape}")

    # Step 2: blur(...)
    blurred = TensorOperators.blur(added)
    print(f"  blur(...): {blurred.shape}")

    # Step 3: global_avg_pool(...)
    pooled = TensorOperators.global_avg_pool(blurred)
    print(f"  global_avg_pool(...): {pooled.shape}")

    assert pooled.shape == (images.shape[0],), "Final output should be [batch]"
    assert not torch.isnan(pooled).any(), "Output contains NaN"

    print(f"  Output range: [{pooled.min():.2f}, {pooled.max():.2f}]")

    print("\nReal images PASSED!")


def test_lasso_classifier():
    """Test LASSO classifier on extracted features."""
    print("\n" + "=" * 60)
    print("TEST 8: LASSO Classifier")
    print("=" * 60)

    # Load CIFAR-10
    data_module = MNISTDataModule(
        dataset='cifar10',
        batch_size=128,
        num_workers=0
    )
    data_module.setup()

    train_loader = data_module.get_train_loader()
    images, labels = next(iter(train_loader))

    print(f"Loaded batch: {images.shape}")

    # Extract RGB channels
    I_R = images[:, 0, :, :]
    I_G = images[:, 1, :, :]
    I_B = images[:, 2, :, :]

    # Create 5 simple features
    print("\nExtracting features from 5 formulas...")

    features = []

    # Feature 1: global_avg_pool(I_R)
    f1 = TensorOperators.global_avg_pool(I_R)
    features.append(f1)
    print(f"  F1: global_avg_pool(I_R)")

    # Feature 2: global_avg_pool(I_G)
    f2 = TensorOperators.global_avg_pool(I_G)
    features.append(f2)
    print(f"  F2: global_avg_pool(I_G)")

    # Feature 3: global_avg_pool(I_B)
    f3 = TensorOperators.global_avg_pool(I_B)
    features.append(f3)
    print(f"  F3: global_avg_pool(I_B)")

    # Feature 4: global_avg_pool(blur(I_R))
    f4 = TensorOperators.global_avg_pool(TensorOperators.blur(I_R))
    features.append(f4)
    print(f"  F4: global_avg_pool(blur(I_R))")

    # Feature 5: global_avg_pool(edge_x(I_R))
    f5 = TensorOperators.global_avg_pool(TensorOperators.edge_x(I_R))
    features.append(f5)
    print(f"  F5: global_avg_pool(edge_x(I_R))")

    # Stack features: [batch, 5]
    features_tensor = torch.stack(features, dim=1)
    print(f"\nFeature matrix: {features_tensor.shape}")

    # Train LASSO classifier
    print("\nTraining LASSO classifier...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    features_tensor = features_tensor.to(device)
    labels = labels.to(device)

    accuracy, active_features, model = train_lasso_classifier(
        features_tensor,
        labels,
        num_classes=10,
        l1_lambda=0.01,
        epochs=50,
        lr=0.01,
        device=device
    )

    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Active features: {active_features}/5")
    print(f"  Sparsity: {active_features/5:.2%}")

    assert accuracy > 0.1, "Accuracy should be above random baseline"

    print("\nLASSO classifier PASSED!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("TENSOR OPERATORS TEST SUITE")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    try:
        test_element_wise_ops()
        test_activations()
        test_spatial_ops()
        test_pooling_ops()
        test_math_ops()
        test_gradients()
        test_on_real_images()
        test_lasso_classifier()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
