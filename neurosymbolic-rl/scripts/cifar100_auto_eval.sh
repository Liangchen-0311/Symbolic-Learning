#!/bin/bash
# Auto evaluation pipeline for CIFAR-100: wait for training → extract features → run MLP
set -e

export PYTHONPATH=/workspace/neurosymbolic-rl
CONFIG=configs/tensor_vsr_cifar100_spp.yaml
OUTPUT_DIR=outputs/cifar100_spp
LOG=$OUTPUT_DIR/training.log

echo "[$(date)] Waiting for training to complete..."

# Wait for training to finish (check for final model or "TRAINING COMPLETE" in log)
while true; do
    if [ -f "$OUTPUT_DIR/feature_bank/feature_bank.json" ]; then
        # Check if training process is still running
        if ! pgrep -f "train_tensor_vsr_large_bank.*cifar100" > /dev/null 2>&1; then
            echo "[$(date)] Training process finished."
            break
        fi
    fi
    sleep 60
done

echo "[$(date)] Training complete. Starting feature extraction + MLP evaluation..."

# Extract features and save cache
echo "[$(date)] Step 1: Extract features (with caching)..."
python -u experiments/evaluate_mlp_only.py \
    --config $CONFIG \
    --device cuda \
    --hidden_dim 1024 \
    --epochs 200 \
    --patience 30 \
    --save_features \
    > $OUTPUT_DIR/eval_mlp_extract.log 2>&1

echo "[$(date)] Step 2: Run MLP v3 (Mixup + Label Smoothing)..."
python -u experiments/run_mlp_v3.py \
    --features_dir $OUTPUT_DIR/features_cache \
    --output_dir $OUTPUT_DIR \
    --num_classes 100 \
    --device cuda \
    --epochs 200 \
    --patience 30 \
    > $OUTPUT_DIR/eval_mlp_v3.log 2>&1

echo "[$(date)] ALL DONE."
echo "Results:"
cat $OUTPUT_DIR/eval_mlp_v3_results.json
