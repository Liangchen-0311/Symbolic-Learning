#!/bin/bash

# Run Large Feature Bank Training Script
#
# This implements the strategy to boost accuracy from 20% to 50%+
#
# Strategy:
# - Generate 200 formulas with low threshold
# - Use strong LASSO (L1=0.5) to select 60-100 best features
# - Spatial pooling operators for position awareness
# - Diversity-aware rewards
#
# Expected runtime: 3-4 hours on M1 64GB

echo "=========================================="
echo "Large Feature Bank Training"
echo "Target: 50%+ Accuracy (from 20%)"
echo "=========================================="
echo ""

# Check if running on M1 Mac
if [[ $(uname -m) == "arm64" ]]; then
    echo "Detected Apple Silicon (M1/M2)"
    DEVICE="mps"
else
    echo "Detected x86_64 architecture"
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA available"
        DEVICE="cuda"
    else
        echo "No CUDA, using CPU"
        DEVICE="cpu"
    fi
fi

echo "Device: $DEVICE"
echo ""

# Run training
python3 experiments/train_tensor_vsr_large_bank.py \
    --config configs/tensor_vsr_m1_cifar10_large_bank.yaml \
    --dataset cifar10 \
    --device $DEVICE \
    --output_dir outputs/large_bank_$(date +%Y%m%d_%H%M%S) \
    --seed 42

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
