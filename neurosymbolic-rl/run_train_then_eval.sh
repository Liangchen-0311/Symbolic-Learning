#!/bin/bash
set -e
export PYTHONPATH=/workspace/neurosymbolic-rl

# Wait for training process to finish
TRAIN_PID=208204
echo "[$(date)] Waiting for training PID $TRAIN_PID to finish..."
while kill -0 $TRAIN_PID 2>/dev/null; do
  sleep 30
done
echo "[$(date)] Training finished."

# Print final training stats
echo ""
echo "=========================================="
echo "TRAINING FINAL STATS"
echo "=========================================="
grep "^\[Iter " outputs/cifar10_spp/training.log | tail -3
grep "Mean Acc" outputs/cifar10_spp/training.log | tail -1
grep "Fill Rate" outputs/cifar10_spp/training.log | tail -1
grep "Turnover" outputs/cifar10_spp/training.log | tail -1
echo ""

# Run MLP evaluation
echo "=========================================="
echo "[$(date)] Starting MLP evaluation..."
echo "=========================================="
python experiments/evaluate_mlp.py \
  --config configs/tensor_vsr_cifar10_spp.yaml \
  --bank_path outputs/cifar10_spp/feature_bank \
  --device cuda \
  --hidden_dim 512 \
  --dropout 0.3 \
  --lr 1e-3 \
  --epochs 100 \
  --batch_size 256 \
  --patience 15 \
  2>&1 | tee outputs/cifar10_spp/eval_mlp.log

echo ""
echo "[$(date)] ALL DONE."
