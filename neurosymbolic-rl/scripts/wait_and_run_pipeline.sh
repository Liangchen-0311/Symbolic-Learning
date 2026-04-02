#!/bin/bash
set -e

LOG="/workspace/neurosymbolic-rl/logs/imagenet_download.log"
DATA="/workspace/neurosymbolic-rl/data/imagenet"

echo "Waiting for ImageNet download to finish (PID 17352)..."
while kill -0 17352 2>/dev/null; do
    TRAIN_COUNT=$(find "$DATA/train/" -name "*.JPEG" | wc -l)
    echo "  $(date +%H:%M:%S) train images: $TRAIN_COUNT / 1281167"
    sleep 60
done

echo "Download process finished at $(date)"

# Check results
TRAIN_COUNT=$(find "$DATA/train/" -name "*.JPEG" | wc -l)
echo "Train images: $TRAIN_COUNT"

# If val doesn't exist, create symlink to train for now (val download may have failed)
if [ ! -d "$DATA/val" ]; then
    echo "Val dir missing — creating symlink to train for pipeline run"
    ln -s "$DATA/train" "$DATA/val"
fi

VAL_COUNT=$(find "$DATA/val/" -name "*.JPEG" | wc -l)
echo "Val images: $VAL_COUNT"

echo ""
echo "Starting pipeline..."
cd /workspace/neurosymbolic-rl

python experiments/train_imagenet_pipeline.py \
    --config configs/tensor_vsr_imagenet_single_bank.yaml \
    --device cuda \
    --output_dir outputs/imagenet_single_bank \
    2>&1 | tee logs/pipeline_single_bank.log
