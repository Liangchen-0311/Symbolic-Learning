#!/bin/bash
# Wait for v3 training to finish, then run injection + logistic regression eval
set -e

TRAIN_PID=1586625
BANK_PATH="outputs/cifar10_v3/feature_bank"
CONFIG="configs/tensor_vsr_cifar10_v3.yaml"
LOG="outputs/cifar10_v3/eval_pipeline.log"

echo "=== Waiting for training (PID $TRAIN_PID) to finish ===" | tee "$LOG"
while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 60
done
echo "=== Training finished ===" | tee -a "$LOG"

# Check feature bank exists
if [ ! -f "$BANK_PATH/feature_bank.json" ]; then
    echo "ERROR: Feature bank not found at $BANK_PATH" | tee -a "$LOG"
    exit 1
fi

# Step 1: Inject hand-crafted formulas
echo "" | tee -a "$LOG"
echo "=== Step 1: Injecting hand-crafted formulas ===" | tee -a "$LOG"
PYTHONPATH=/workspace/neurosymbolic-rl python -u experiments/inject_formulas.py \
    --bank_path "$BANK_PATH" \
    --config "$CONFIG" \
    2>&1 | tee -a "$LOG"

# Step 2: Logistic regression evaluation (sweep C values)
echo "" | tee -a "$LOG"
echo "=== Step 2: Logistic Regression Evaluation ===" | tee -a "$LOG"
PYTHONPATH=/workspace/neurosymbolic-rl python -u experiments/evaluate_logreg.py \
    --config "$CONFIG" \
    --bank_path "$BANK_PATH" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Pipeline complete ===" | tee -a "$LOG"
