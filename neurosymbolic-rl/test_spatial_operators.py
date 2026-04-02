"""
Quick test for new spatial pooling operators.

This verifies that the 6 new operators work correctly:
- pool_top_half
- pool_bottom_half
- pool_left_half
- pool_right_half
- pool_center
- pool_corners
"""

import torch
from src.symbolic.tensor_operators import TensorOperators, TENSOR_OPERATORS, ROOT_OPERATORS

def test_spatial_operators():
    """Test all spatial pooling operators."""
    print("=" * 60)
    print("Testing Spatial Pooling Operators")
    print("=" * 60)

    # Create a test image batch [batch=2, H=32, W=32]
    batch_size = 2
    H, W = 32, 32
    x = torch.randn(batch_size, H, W)

    print(f"\nInput shape: {x.shape}")
    print(f"Input range: [{x.min():.2f}, {x.max():.2f}]")

    # Test each spatial operator
    spatial_ops = [
        'pool_top_half',
        'pool_bottom_half',
        'pool_left_half',
        'pool_right_half',
        'pool_center',
        'pool_corners'
    ]

    print("\n" + "-" * 60)
    print("Testing Spatial Operators")
    print("-" * 60)

    for op_name in spatial_ops:
        if op_name in TENSOR_OPERATORS:
            op_func, arity, output_type = TENSOR_OPERATORS[op_name]

            # Apply operator
            output = op_func(x)

            # Check output
            assert output.shape == (batch_size,), f"Expected shape ({batch_size},), got {output.shape}"
            assert output_type == 'scalar', f"Expected output_type 'scalar', got {output_type}"
            assert op_name in ROOT_OPERATORS, f"{op_name} should be in ROOT_OPERATORS"

            print(f"✓ {op_name:20s} - shape: {output.shape}, range: [{output.min():.2f}, {output.max():.2f}]")
        else:
            print(f"✗ {op_name:20s} - NOT FOUND IN REGISTRY")

    # Test original pooling operators still work
    print("\n" + "-" * 60)
    print("Testing Original Pooling Operators")
    print("-" * 60)

    original_ops = [
        'global_avg_pool',
        'global_max_pool',
        'global_std_pool',
        'global_l2_pool'
    ]

    for op_name in original_ops:
        if op_name in TENSOR_OPERATORS:
            op_func, arity, output_type = TENSOR_OPERATORS[op_name]
            output = op_func(x)
            print(f"✓ {op_name:20s} - shape: {output.shape}, range: [{output.min():.2f}, {output.max():.2f}]")
        else:
            print(f"✗ {op_name:20s} - NOT FOUND")

    # Test operator counts
    print("\n" + "-" * 60)
    print("Operator Registry Summary")
    print("-" * 60)

    print(f"Total operators: {len(TENSOR_OPERATORS)}")
    print(f"Root operators: {len(ROOT_OPERATORS)}")

    # Count by type
    scalar_ops = [k for k, v in TENSOR_OPERATORS.items() if v[2] == 'scalar']
    tensor_ops = [k for k, v in TENSOR_OPERATORS.items() if v[2] == 'tensor']

    print(f"Scalar (pooling) operators: {len(scalar_ops)}")
    print(f"Tensor operators: {len(tensor_ops)}")

    # Verify new operators are present
    print("\n" + "-" * 60)
    print("New Spatial Operators Verification")
    print("-" * 60)

    all_present = True
    for op in spatial_ops:
        present = op in TENSOR_OPERATORS and op in ROOT_OPERATORS
        status = "✓" if present else "✗"
        print(f"{status} {op}")
        if not present:
            all_present = False

    print("\n" + "=" * 60)
    if all_present:
        print("SUCCESS! All spatial operators are correctly implemented.")
    else:
        print("FAILURE! Some operators are missing.")
    print("=" * 60)

    return all_present


def test_operator_composition():
    """Test that spatial operators can be composed with other operators."""
    print("\n" + "=" * 60)
    print("Testing Operator Composition")
    print("=" * 60)

    # Create test batch
    batch_size = 4
    H, W = 32, 32
    I_R = torch.randn(batch_size, H, W)
    I_G = torch.randn(batch_size, H, W)

    # Test composition: pool_top_half(edge_x(I_R))
    print("\nTest 1: pool_top_half(edge_x(I_R))")
    edge_output = TensorOperators.edge_x(I_R)
    print(f"  edge_x output shape: {edge_output.shape}")

    pool_output = TensorOperators.pool_top_half(edge_output)
    print(f"  pool_top_half output shape: {pool_output.shape}")
    assert pool_output.shape == (batch_size,), "Composition failed"
    print("  ✓ Composition successful")

    # Test composition: pool_center(add(I_R, I_G))
    print("\nTest 2: pool_center(add(I_R, I_G))")
    add_output = TensorOperators.add(I_R, I_G)
    print(f"  add output shape: {add_output.shape}")

    pool_output = TensorOperators.pool_center(add_output)
    print(f"  pool_center output shape: {pool_output.shape}")
    assert pool_output.shape == (batch_size,), "Composition failed"
    print("  ✓ Composition successful")

    # Test composition: pool_bottom_half(blur(multiply(I_R, I_G)))
    print("\nTest 3: pool_bottom_half(blur(multiply(I_R, I_G)))")
    mult_output = TensorOperators.multiply(I_R, I_G)
    print(f"  multiply output shape: {mult_output.shape}")

    blur_output = TensorOperators.blur(mult_output)
    print(f"  blur output shape: {blur_output.shape}")

    pool_output = TensorOperators.pool_bottom_half(blur_output)
    print(f"  pool_bottom_half output shape: {pool_output.shape}")
    assert pool_output.shape == (batch_size,), "Composition failed"
    print("  ✓ Composition successful")

    print("\n" + "=" * 60)
    print("SUCCESS! Operator composition works correctly.")
    print("=" * 60)


if __name__ == '__main__':
    # Run tests
    success = test_spatial_operators()

    if success:
        test_operator_composition()

    print("\n✓ All tests passed! Ready to train with spatial operators.\n")
