#!/bin/bash
# ============================================================
# NNUE Training Data Pipeline: Broadcast + Standard
# ============================================================
#
# Two data sources, two strategies:
#   1. Broadcast PGNs: extract [%eval] annotations (fast, no SF)
#   2. Standard PGNs:  label with SF depth 12 (~3M positions)
#
# Usage:
#   ./download_and_convert.sh                          # Broadcast only (default)
#   ./download_and_convert.sh --with-standard 2025-01  # + standard month
#   ./download_and_convert.sh --standard-positions 3000000 --threads 8
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"
BULLET_UTILS="$HOME/bullet/target/release/bullet-utils"
DATA_DIR="$TRAINING_DIR/data"

mkdir -p "$DATA_DIR"

# Defaults
BROADCAST_MONTHS=""
DOWNLOAD_ALL=false
STANDARD_MONTH=""
STANDARD_POSITIONS=3000000
THREADS=8
LABEL_DEPTH=12
MIN_ELO_STANDARD=2200

while [[ $# -gt 0 ]]; do
    case $1 in
        --months) BROADCAST_MONTHS="$2"; shift 2 ;;
        --all) DOWNLOAD_ALL=true; shift ;;
        --with-standard) STANDARD_MONTH="$2"; shift 2 ;;
        --standard-positions) STANDARD_POSITIONS="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --label-depth) LABEL_DEPTH="$2"; shift 2 ;;
        --min-elo-standard) MIN_ELO_STANDARD="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "=== NNUE Training Data Pipeline ==="
echo "  Broadcast: extract [%eval] annotations (no SF)"
if [ -n "$STANDARD_MONTH" ]; then
    echo "  Standard:  SF depth $LABEL_DEPTH, ${STANDARD_POSITIONS} pos, ELO>=$MIN_ELO_STANDARD"
fi
echo ""

BROADCAST_FILE="$DATA_DIR/broadcast.txt"
STANDARD_FILE="$DATA_DIR/standard.txt"

# ── Step 1a: Broadcast data (extract [%eval]) ──
echo "[Step 1a] Broadcast data (extract [%eval])..."
cd "$TRAINING_DIR"
> "$BROADCAST_FILE"  # Clear

if [ "$DOWNLOAD_ALL" = true ]; then
    python3 scripts/prepare_data.py --download-all --output "$BROADCAST_FILE"
elif [ -n "$BROADCAST_MONTHS" ]; then
    for month in $BROADCAST_MONTHS; do
        echo "  Month: $month"
        python3 scripts/prepare_data.py --download "$month" --output "$BROADCAST_FILE" || true
    done
else
    echo "  Default: recent broadcast months..."
    for month in 2024-07 2024-08 2024-09 2024-10 2024-11 2024-12 2025-01; do
        echo "  Month: $month"
        python3 scripts/prepare_data.py --download "$month" --output "$BROADCAST_FILE" || true
    done
fi

BROADCAST_COUNT=$(wc -l < "$BROADCAST_FILE" || echo 0)
echo "  Broadcast positions: $BROADCAST_COUNT"

# ── Step 1b: Standard data (SF depth 12 labeling) ──
> "$STANDARD_FILE"  # Clear
STANDARD_COUNT=0

if [ -n "$STANDARD_MONTH" ]; then
    echo ""
    echo "[Step 1b] Standard data (SF depth $LABEL_DEPTH)..."
    python3 scripts/prepare_data.py --download-standard "$STANDARD_MONTH" \
        --output "$STANDARD_FILE" \
        --label-depth "$LABEL_DEPTH" --threads "$THREADS" \
        --min-elo "$MIN_ELO_STANDARD" --positions "$STANDARD_POSITIONS"
    STANDARD_COUNT=$(wc -l < "$STANDARD_FILE" || echo 0)
    echo "  Standard positions: $STANDARD_COUNT"
fi

# ── Step 2: Merge + Shuffle ──
echo ""
echo "[Step 2] Merging and shuffling..."
cat "$BROADCAST_FILE" "$STANDARD_FILE" > "$DATA_DIR/all.txt"
TOTAL_LINES=$(wc -l < "$DATA_DIR/all.txt")
echo "  Total: $TOTAL_LINES (broadcast: $BROADCAST_COUNT + standard: $STANDARD_COUNT)"
shuf "$DATA_DIR/all.txt" -o "$DATA_DIR/all_shuffled.txt"
mv "$DATA_DIR/all_shuffled.txt" "$DATA_DIR/all.txt"
echo "  Shuffled."

# ── Step 3: Split train/test (95/5) ──
echo ""
echo "[Step 3] Splitting train/test (95/5)..."
TOTAL=$(wc -l < "$DATA_DIR/all.txt")
TEST_SIZE=$((TOTAL / 20))
TRAIN_SIZE=$((TOTAL - TEST_SIZE))

head -n "$TRAIN_SIZE" "$DATA_DIR/all.txt" > "$DATA_DIR/train_split.txt"
tail -n "$TEST_SIZE" "$DATA_DIR/all.txt" > "$DATA_DIR/test_split.txt"
echo "  Train: $TRAIN_SIZE positions"
echo "  Test:  $TEST_SIZE positions"

# ── Step 4: Convert to bulletformat ──
echo ""
echo "[Step 4] Converting to bulletformat..."

if [ -f "$BULLET_UTILS" ]; then
    "$BULLET_UTILS" convert --from text --input "$DATA_DIR/train_split.txt" --output "$DATA_DIR/train.bullet" --threads 4
    "$BULLET_UTILS" convert --from text --input "$DATA_DIR/test_split.txt" --output "$DATA_DIR/test.bullet" --threads 4
    echo "  Created: $DATA_DIR/train.bullet"
    echo "  Created: $DATA_DIR/test.bullet"
else
    echo "  WARNING: bullet-utils not found at $BULLET_UTILS"
    echo "  Build it: cd ~/bullet && cargo build --release --package bullet-utils"
    echo "  Then re-run this script."
    exit 1
fi

# ── Cleanup ──
rm -f "$DATA_DIR/train_split.txt" "$DATA_DIR/test_split.txt"

echo ""
echo "=== Data pipeline complete! ==="
echo "  Broadcast: $BROADCAST_COUNT positions (extracted [%eval])"
echo "  Standard:  $STANDARD_COUNT positions (SF depth $LABEL_DEPTH)"
echo "  Train:     $DATA_DIR/train.bullet ($TRAIN_SIZE positions)"
echo "  Test:      $DATA_DIR/test.bullet ($TEST_SIZE positions)"
echo ""
echo "Next: cd $TRAINING_DIR && cargo run --release -- --data data/train.bullet --test data/test.bullet --epochs 400"
