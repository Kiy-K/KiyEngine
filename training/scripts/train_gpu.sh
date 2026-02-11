#!/bin/bash
# ============================================================
# KiyEngine NNUE Training — GPU (Lightning.AI L40S)
# ============================================================
#
# Trains a (768→512)×2 → SCReLU → 8 output bucket NNUE network
# using the Bullet framework on GPU.
#
# Prerequisites:
#   1. Run ./scripts/setup_lightning.sh first
#   2. Have training data in data/train.bullet (use download_and_convert.sh)
#
# Usage:
#   ./scripts/train_gpu.sh                          # defaults
#   ./scripts/train_gpu.sh --epochs 200 --lr 0.001  # custom
#   ./scripts/train_gpu.sh --resume checkpoints/kiyengine-v6.1/   # resume
#
# After training:
#   python3 scripts/convert_to_kynn.py \
#     checkpoints/kiyengine-v6.1/quantised.bin \
#     kiyengine.nnue --buckets 8
# ============================================================

set -euo pipefail

# ── Ensure cargo is in PATH ──
source "$HOME/.cargo/env" 2>/dev/null || true

# ── Defaults ──
EPOCHS=150
START_LR=0.001
END_LR=0.0000025
BATCH_SIZE=16384
SUPERBATCH_SIZE=6104
NET_ID="kiyengine-v6.1"
DATA_PATH=""
TEST_PATH=""
RESUME_DIR=""
HIDDEN_SIZE=512

# ── Parse arguments ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)      EPOCHS="$2";         shift 2 ;;
        --lr)          START_LR="$2";       shift 2 ;;
        --end-lr)      END_LR="$2";         shift 2 ;;
        --batch-size)  BATCH_SIZE="$2";     shift 2 ;;
        --superbatch)  SUPERBATCH_SIZE="$2"; shift 2 ;;
        --id)          NET_ID="$2";         shift 2 ;;
        --data)        DATA_PATH="$2";      shift 2 ;;
        --test)        TEST_PATH="$2";      shift 2 ;;
        --resume)      RESUME_DIR="$2";     shift 2 ;;
        --hidden)      HIDDEN_SIZE="$2";    shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --epochs N          Training epochs (default: $EPOCHS)"
            echo "  --lr FLOAT          Starting learning rate (default: $START_LR)"
            echo "  --end-lr FLOAT      End learning rate for cosine decay (default: $END_LR)"
            echo "  --batch-size N      Batch size (default: $BATCH_SIZE)"
            echo "  --superbatch N      Batches per superbatch (default: $SUPERBATCH_SIZE)"
            echo "  --id NAME           Network ID for saving (default: $NET_ID)"
            echo "  --data PATH         Training data path (default: auto-detect)"
            echo "  --test PATH         Test data path (default: auto-detect)"
            echo "  --resume DIR        Resume from checkpoint directory"
            echo "  --hidden N          Hidden layer size (default: $HIDDEN_SIZE)"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$TRAINING_DIR")"

cd "$TRAINING_DIR"

# ── Auto-detect training data ──
if [ -z "$DATA_PATH" ]; then
    for candidate in data/train.bullet data/*.bullet; do
        if [ -f "$candidate" ]; then
            DATA_PATH="$candidate"
            break
        fi
    done
fi

if [ -z "$DATA_PATH" ] || [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: No training data found!"
    echo "  Expected: $TRAINING_DIR/data/train.bullet"
    echo ""
    echo "  To prepare data, run:"
    echo "    ./scripts/download_and_convert.sh"
    exit 1
fi

# Auto-detect test data
if [ -z "$TEST_PATH" ]; then
    if [ -f "data/test.bullet" ]; then
        TEST_PATH="data/test.bullet"
    fi
fi

# ── Check GPU ──
GPU_NAME="CPU"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
fi

# ── Check binary ──
TRAIN_BIN="target/release/train"
if [ ! -f "$TRAIN_BIN" ]; then
    echo "Building trainer..."
    if grep -q 'cuda' Cargo.toml && command -v nvidia-smi &> /dev/null; then
        cargo build --release --features cuda 2>&1 | tail -3
    else
        cargo build --release 2>&1 | tail -3
    fi
fi

# ── Calculate stats ──
DATA_SIZE=$(du -sh "$DATA_PATH" | cut -f1)
DATA_POSITIONS=$(( $(stat -c%s "$DATA_PATH") / 32 ))  # Each position is 32 bytes in bulletformat

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  KiyEngine NNUE Training — GPU                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Architecture:  (768→${HIDDEN_SIZE})×2 → SCReLU → 8 buckets"
echo "  GPU:           $GPU_NAME"
echo "  Data:          $DATA_PATH ($DATA_SIZE, ~${DATA_POSITIONS} positions)"
echo "  Test:          ${TEST_PATH:-none}"
echo "  Epochs:        $EPOCHS"
echo "  LR:            $START_LR → $END_LR (cosine decay)"
echo "  Batch size:    $BATCH_SIZE"
echo "  Superbatch:    $SUPERBATCH_SIZE batches"
echo "  Net ID:        $NET_ID"
echo "  Output:        checkpoints/$NET_ID/"
echo ""

# ── Build trainer args ──
TRAIN_ARGS=(
    --data "$DATA_PATH"
    --epochs "$EPOCHS"
    --lr "$START_LR"
    --batch-size "$BATCH_SIZE"
    --id "$NET_ID"
)

if [ -n "$TEST_PATH" ] && [ -f "$TEST_PATH" ]; then
    TRAIN_ARGS+=(--test "$TEST_PATH")
fi

if [ -n "$SUPERBATCH_SIZE" ]; then
    TRAIN_ARGS+=(--superbatch-size "$SUPERBATCH_SIZE")
fi

# ── Start training ──
echo "Starting training..."
echo "  Command: $TRAIN_BIN ${TRAIN_ARGS[*]}"
echo ""
echo "═══════════════════════════════════════════════════════════"

START_TIME=$(date +%s)

"$TRAIN_BIN" "${TRAIN_ARGS[@]}"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINUTES=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Training complete! (${HOURS}h ${MINUTES}m)"
echo ""
echo "  Checkpoint: checkpoints/$NET_ID/"
echo ""

# ── Auto-convert if quantised.bin exists ──
QUANTISED="checkpoints/$NET_ID/quantised.bin"
OUTPUT_NNUE="$PROJECT_DIR/kiyengine.nnue"

if [ -f "$QUANTISED" ]; then
    echo "  Converting to KYNN format..."
    python3 scripts/convert_to_kynn.py "$QUANTISED" "$OUTPUT_NNUE" --buckets 8
    echo ""
    echo "  NNUE saved: $OUTPUT_NNUE ($(du -sh "$OUTPUT_NNUE" | cut -f1))"
    echo ""
    echo "  To test in engine:"
    echo "    cp $OUTPUT_NNUE $PROJECT_DIR/kiyengine.nnue"
    echo "    cd $PROJECT_DIR && cargo build --release"
    echo "    ./target/release/kiy_engine"
else
    echo "  NOTE: quantised.bin not found. Convert manually after training:"
    echo "    python3 scripts/convert_to_kynn.py checkpoints/$NET_ID/quantised.bin kiyengine.nnue --buckets 8"
fi
