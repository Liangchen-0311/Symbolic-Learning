#!/bin/bash
# Wait for training to finish, then run LogReg eval
PID=$1
echo "Waiting for training PID $PID to finish..."
while kill -0 $PID 2>/dev/null; do sleep 30; done
echo "Training done! Starting LogReg evaluation..."
cd /workspace/neurosymbolic-rl
PYTHONPATH=/workspace/neurosymbolic-rl python -u experiments/evaluate_logreg.py \
    --config configs/tensor_vsr_cifar100_v3.yaml \
    --bank_path outputs/cifar100_v3/feature_bank
