#!/bin/bash
# Usage: bash run_training.sh [skip_chunks]
# Example: bash run_training.sh 2440  (skip first 2440 chunks)
cd ~/sustainability/aideen
SKIP="${1:-0}"
SKIP_FLAG=""
if [ "$SKIP" -gt 0 ] 2>/dev/null; then
    SKIP_FLAG="--skip-chunks $SKIP"
fi
nohup cargo run --release -p aideen-training --features aideen-training/wgpu --bin train -- --file corpus_combined.txt --resume model_large --epochs 1 --log-every 1 --save-every 1 $SKIP_FLAG > training_full_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"
echo "Training launched (skip=$SKIP)! Monitor with: tail -f ~/sustainability/aideen/training_full_*.log"
