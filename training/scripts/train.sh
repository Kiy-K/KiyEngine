#!/bin/bash
# ============================================================
# KiyEngine NNUE Training Script
# ============================================================
#
# Runs the Bullet-based NNUE trainer on prepared bulletformat data.
# Designed for Lightning.AI L4 GPU (24GB VRAM).
#
# Usage:
#   ./train.sh [--epochs 400] [--lr 0.001] [--batch-size 16384]
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"
BINARY="$TRAINING_DIR/target/release/train"
DATA_DIR="$TRAINING_DIR/data"

# Defaults
EPOCHS=400
LR=0.001
BATCH_SIZE=16384
NET_ID="kiyengine-nnue-v1"

while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs) EPOCHS="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --id) NET_ID="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# Check data exists
if [ ! -f "$DATA_DIR/train.bullet" ]; then
    echo "ERROR: Training data not found at $DATA_DIR/train.bullet"
    echo "Run ./scripts/download_and_convert.sh first."
    exit 1
fi

# Check binary exists
if [ ! -f "$BINARY" ]; then
    echo "Building trainer..."
    cd "$TRAINING_DIR" && cargo build --release
fi

echo "=== Starting NNUE Training ==="
echo "  Data:       $DATA_DIR/train.bullet"
echo "  Net ID:     $NET_ID"
echo "  Epochs:     $EPOCHS"
echo "  LR:         $LR"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Set CUDA visible devices (Lightning.AI L4)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Run trainer
TEST_ARG=""
if [ -f "$DATA_DIR/test.bullet" ]; then
    TEST_ARG="--test $DATA_DIR/test.bullet"
fi

"$BINARY" \
    --data "$DATA_DIR/train.bullet" \
    $TEST_ARG \
    --id "$NET_ID" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --batch-size "$BATCH_SIZE"

echo ""
echo "=== Training finished ==="
echo "Checkpoints in: $TRAINING_DIR/checkpoints/"
echo ""
echo "To use the best network in KiyEngine:"
echo "  cp checkpoints/${NET_ID}_epoch*.bin ../kiyengine.nnue"
