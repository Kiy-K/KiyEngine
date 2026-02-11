#!/bin/bash
# ============================================================
# Download Lichess Broadcast Data & Convert to Bulletformat
# ============================================================
#
# Downloads recent Lichess broadcast PGN files (GM/IM games),
# extracts training positions, and converts to bulletformat.
#
# Usage:
#   ./download_and_convert.sh [--months "2025-01 2025-02 ..."] [--all]
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"
BULLET_UTILS="$HOME/bullet/target/release/bullet-utils"
DATA_DIR="$TRAINING_DIR/data"

mkdir -p "$DATA_DIR"

# Default: download last 6 months of broadcasts
MONTHS=""
DOWNLOAD_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --months) MONTHS="$2"; shift 2 ;;
        --all) DOWNLOAD_ALL=true; shift ;;
        *) shift ;;
    esac
done

echo "=== Lichess Broadcast → Bulletformat Pipeline ==="
echo ""

# ── Step 1: Download & extract positions ──
echo "[Step 1] Downloading and extracting positions..."
cd "$TRAINING_DIR"

if [ "$DOWNLOAD_ALL" = true ]; then
    python3 scripts/prepare_data.py --download-all --output "$DATA_DIR/train.txt" --min-elo 2400
elif [ -n "$MONTHS" ]; then
    for month in $MONTHS; do
        echo "  Processing month: $month"
        python3 scripts/prepare_data.py --download "$month" --output "$DATA_DIR/train.txt" --min-elo 2400
    done
else
    # Default: last few available months
    echo "  Downloading recent broadcast months..."
    for month in 2024-07 2024-08 2024-09 2024-10 2024-11 2024-12 2025-01; do
        echo "  Processing month: $month"
        python3 scripts/prepare_data.py --download "$month" --output "$DATA_DIR/train.txt" --min-elo 2400 || true
    done
fi

TOTAL_LINES=$(wc -l < "$DATA_DIR/train.txt" || echo 0)
echo ""
echo "  Total positions extracted: $TOTAL_LINES"

# ── Step 2: Shuffle ──
echo ""
echo "[Step 2] Shuffling training data..."
shuf "$DATA_DIR/train.txt" -o "$DATA_DIR/train_shuffled.txt"
mv "$DATA_DIR/train_shuffled.txt" "$DATA_DIR/train.txt"
echo "  Shuffled."

# ── Step 3: Split train/test ──
echo ""
echo "[Step 3] Splitting train/test (95/5)..."
TOTAL=$(wc -l < "$DATA_DIR/train.txt")
TEST_SIZE=$((TOTAL / 20))
TRAIN_SIZE=$((TOTAL - TEST_SIZE))

head -n "$TRAIN_SIZE" "$DATA_DIR/train.txt" > "$DATA_DIR/train_split.txt"
tail -n "$TEST_SIZE" "$DATA_DIR/train.txt" > "$DATA_DIR/test_split.txt"
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
echo "  Train: $DATA_DIR/train.bullet ($TRAIN_SIZE positions)"
echo "  Test:  $DATA_DIR/test.bullet ($TEST_SIZE positions)"
echo ""
echo "Next: ./scripts/train.sh"
