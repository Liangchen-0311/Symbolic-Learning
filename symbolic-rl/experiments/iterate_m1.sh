#!/bin/bash

# Rapid iteration script for M1 Studio
# Test both CIFAR-10 and CIFAR-100 with different configs

echo "=========================================="
echo "Tensor VSR - M1 Rapid Iteration"
echo "CIFAR-10 and CIFAR-100 Testing"
echo "=========================================="

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/m1_iterations/$TIMESTAMP"
mkdir -p $OUTPUT_DIR

# Save iteration log
LOG_FILE="$OUTPUT_DIR/iteration_log.txt"
echo "M1 Iteration Started: $(date)" > $LOG_FILE

# Function to run one iteration
run_iteration() {
    local config=$1
    local dataset=$2
    local name=$3

    echo ""
    echo "=========================================="
    echo "Iteration: $name ($dataset)"
    echo "Config: $config"
    echo "=========================================="

    # Log to file
    echo "" >> $LOG_FILE
    echo "Iteration: $name ($dataset)" >> $LOG_FILE
    echo "Config: $config" >> $LOG_FILE
    echo "Started: $(date)" >> $LOG_FILE

    # Train
    PYTHONPATH=$(pwd) python3 experiments/train_tensor_vsr.py \
        --config $config \
        --dataset $dataset \
        --device mps \
        --output_dir $OUTPUT_DIR/${dataset}_${name} \
        2>&1 | tee $OUTPUT_DIR/${dataset}_${name}/train_output.txt

    # Check if training succeeded
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[ERROR] Training failed for $dataset - $name"
        echo "ERROR: Training failed" >> $LOG_FILE
        return 1
    fi

    # Evaluate
    PYTHONPATH=$(pwd) python3 experiments/quick_eval.py \
        --feature_bank $OUTPUT_DIR/${dataset}_${name}/best_model.pth \
        --config $config \
        --dataset $dataset \
        2>&1 | tee $OUTPUT_DIR/${dataset}_${name}/eval_results.txt

    # Check if evaluation succeeded
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[ERROR] Evaluation failed for $dataset - $name"
        echo "ERROR: Evaluation failed" >> $LOG_FILE
        return 1
    fi

    # Print summary
    echo ""
    echo "Results for $dataset - $name:"
    grep "Test Accuracy" $OUTPUT_DIR/${dataset}_${name}/eval_results.txt || echo "No accuracy found"
    grep "Active Features" $OUTPUT_DIR/${dataset}_${name}/eval_results.txt || echo "No features found"
    echo ""

    # Log completion
    echo "Completed: $(date)" >> $LOG_FILE
    grep "Test Accuracy" $OUTPUT_DIR/${dataset}_${name}/eval_results.txt >> $LOG_FILE 2>/dev/null || echo "No accuracy" >> $LOG_FILE
    grep "Active Features" $OUTPUT_DIR/${dataset}_${name}/eval_results.txt >> $LOG_FILE 2>/dev/null || echo "No features" >> $LOG_FILE
}

# ========================================
# CIFAR-10 Iterations
# ========================================

echo ""
echo "######################################"
echo "Starting CIFAR-10 Experiments"
echo "######################################"

# CIFAR-10: Baseline
run_iteration "configs/tensor_vsr_m1_cifar10.yaml" "cifar10" "baseline"

# CIFAR-10: Lower L1 (less regularization)
cat configs/tensor_vsr_m1_cifar10.yaml | \
    sed 's/l1_lambda: 0.01/l1_lambda: 0.001/' \
    > configs/tensor_vsr_m1_cifar10_l1_low.yaml
run_iteration "configs/tensor_vsr_m1_cifar10_l1_low.yaml" "cifar10" "l1_low"

# CIFAR-10: Higher L1 (more regularization)
cat configs/tensor_vsr_m1_cifar10.yaml | \
    sed 's/l1_lambda: 0.01/l1_lambda: 0.1/' \
    > configs/tensor_vsr_m1_cifar10_l1_high.yaml
run_iteration "configs/tensor_vsr_m1_cifar10_l1_high.yaml" "cifar10" "l1_high"

# CIFAR-10: Larger feature bank
cat configs/tensor_vsr_m1_cifar10.yaml | \
    sed 's/feature_bank_size: 10/feature_bank_size: 15/' \
    > configs/tensor_vsr_m1_cifar10_bank15.yaml
run_iteration "configs/tensor_vsr_m1_cifar10_bank15.yaml" "cifar10" "bank15"

# ========================================
# CIFAR-100 Iterations
# ========================================

echo ""
echo "######################################"
echo "Starting CIFAR-100 Experiments"
echo "######################################"

# CIFAR-100: Baseline
run_iteration "configs/tensor_vsr_m1_cifar100.yaml" "cifar100" "baseline"

# CIFAR-100: Lower min_accuracy threshold
cat configs/tensor_vsr_m1_cifar100.yaml | \
    sed 's/min_accuracy: 0.03/min_accuracy: 0.02/' \
    > configs/tensor_vsr_m1_cifar100_acc_low.yaml
run_iteration "configs/tensor_vsr_m1_cifar100_acc_low.yaml" "cifar100" "acc_low"

# CIFAR-100: Larger feature bank
cat configs/tensor_vsr_m1_cifar100.yaml | \
    sed 's/feature_bank_size: 15/feature_bank_size: 20/' \
    > configs/tensor_vsr_m1_cifar100_bank20.yaml
run_iteration "configs/tensor_vsr_m1_cifar100_bank20.yaml" "cifar100" "bank20"

# ========================================
# Generate Comparison Report
# ========================================

echo ""
echo "=========================================="
echo "COMPARISON REPORT"
echo "=========================================="

# Create comparison report file
REPORT_FILE="$OUTPUT_DIR/comparison_report.txt"
echo "M1 Iteration Comparison Report" > $REPORT_FILE
echo "Generated: $(date)" >> $REPORT_FILE
echo "" >> $REPORT_FILE

echo ""
echo "CIFAR-10 Results:"
echo "----------------------------------------"
echo "" >> $REPORT_FILE
echo "CIFAR-10 Results:" >> $REPORT_FILE
echo "----------------------------------------" >> $REPORT_FILE

for dir in $OUTPUT_DIR/cifar10_*/; do
    if [ -d "$dir" ]; then
        name=$(basename $dir | sed 's/cifar10_//')
        if [ -f "$dir/eval_results.txt" ]; then
            echo ""
            echo "$name:"
            grep "Test Accuracy" $dir/eval_results.txt || echo "  No accuracy found"
            grep "Active Features" $dir/eval_results.txt || echo "  No features found"

            # Write to report
            echo "" >> $REPORT_FILE
            echo "$name:" >> $REPORT_FILE
            grep "Test Accuracy" $dir/eval_results.txt >> $REPORT_FILE 2>/dev/null || echo "  No accuracy" >> $REPORT_FILE
            grep "Active Features" $dir/eval_results.txt >> $REPORT_FILE 2>/dev/null || echo "  No features" >> $REPORT_FILE
        else
            echo "$name: FAILED (no results)"
            echo "$name: FAILED" >> $REPORT_FILE
        fi
    fi
done

echo ""
echo "CIFAR-100 Results:"
echo "----------------------------------------"
echo "" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "CIFAR-100 Results:" >> $REPORT_FILE
echo "----------------------------------------" >> $REPORT_FILE

for dir in $OUTPUT_DIR/cifar100_*/; do
    if [ -d "$dir" ]; then
        name=$(basename $dir | sed 's/cifar100_//')
        if [ -f "$dir/eval_results.txt" ]; then
            echo ""
            echo "$name:"
            grep "Test Accuracy" $dir/eval_results.txt || echo "  No accuracy found"
            grep "Active Features" $dir/eval_results.txt || echo "  No features found"

            # Write to report
            echo "" >> $REPORT_FILE
            echo "$name:" >> $REPORT_FILE
            grep "Test Accuracy" $dir/eval_results.txt >> $REPORT_FILE 2>/dev/null || echo "  No accuracy" >> $REPORT_FILE
            grep "Active Features" $dir/eval_results.txt >> $REPORT_FILE 2>/dev/null || echo "  No features" >> $REPORT_FILE
        else
            echo "$name: FAILED (no results)"
            echo "$name: FAILED" >> $REPORT_FILE
        fi
    fi
done

echo ""
echo "All results saved to: $OUTPUT_DIR"
echo "Comparison report: $REPORT_FILE"
echo "=========================================="

# Find best configs
echo ""
echo "Recommendations:"
echo "----------------------------------------"
echo "" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "Recommendations:" >> $REPORT_FILE
echo "----------------------------------------" >> $REPORT_FILE

# Best CIFAR-10
best_c10=$(grep -r "Test Accuracy" $OUTPUT_DIR/cifar10_*/eval_results.txt 2>/dev/null | \
           sed 's/.*Test Accuracy: //' | sed 's/%//' | \
           awk -v max=0 -v file="" '{if($1>max){max=$1; file=FILENAME}} END{print file}' | \
           sed 's|.*/cifar10_||' | sed 's|/eval_results.txt||')

if [ ! -z "$best_c10" ]; then
    echo "Best CIFAR-10 config: $best_c10"
    echo "Best CIFAR-10 config: $best_c10" >> $REPORT_FILE
else
    echo "Could not determine best CIFAR-10 config"
    echo "Could not determine best CIFAR-10 config" >> $REPORT_FILE
fi

# Best CIFAR-100
best_c100=$(grep -r "Test Accuracy" $OUTPUT_DIR/cifar100_*/eval_results.txt 2>/dev/null | \
            sed 's/.*Test Accuracy: //' | sed 's/%//' | \
            awk -v max=0 -v file="" '{if($1>max){max=$1; file=FILENAME}} END{print file}' | \
            sed 's|.*/cifar100_||' | sed 's|/eval_results.txt||')

if [ ! -z "$best_c100" ]; then
    echo "Best CIFAR-100 config: $best_c100"
    echo "Best CIFAR-100 config: $best_c100" >> $REPORT_FILE
else
    echo "Could not determine best CIFAR-100 config"
    echo "Could not determine best CIFAR-100 config" >> $REPORT_FILE
fi

echo "=========================================="

echo ""
echo "Iteration completed: $(date)" >> $LOG_FILE
echo "See $REPORT_FILE for full comparison"
