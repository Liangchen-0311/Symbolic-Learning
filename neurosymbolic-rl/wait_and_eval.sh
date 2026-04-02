#!/bin/bash
set -e
export PYTHONPATH=/workspace/neurosymbolic-rl

echo "[$(date)] Waiting for training to reach iter 5000..."
while true; do
  if grep -q "TRAINING COMPLETE" outputs/cifar10_spp/training.log 2>/dev/null; then
    break
  fi
  if grep -q "Training interrupted" outputs/cifar10_spp/training.log 2>/dev/null; then
    break  
  fi
  # Also check if the final_model.pt exists (saved at end of training)
  if [ -f outputs/cifar10_spp/final_model.pt ]; then
    break
  fi
  sleep 30
done

echo "[$(date)] Training finished!"
echo ""
echo "=========================================="
echo "TRAINING FINAL STATS"
echo "=========================================="
tail -30 outputs/cifar10_spp/training.log
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
