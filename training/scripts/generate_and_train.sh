#!/bin/bash
# ============================================================
# Full NNUE Training Pipeline for Lightning.AI L4
# ============================================================
#
# End-to-end: SF data generation → bullet conversion → GPU training
#
# Usage on Lightning.AI Studio:
#   git clone https://github.com/Kiy-K/KiyEngine.git
#   cd KiyEngine/training/scripts
#   chmod +x generate_and_train.sh
#   ./generate_and_train.sh [--positions 1000000]
#
# Prerequisites (auto-installed):
#   - Stockfish (apt or download)
#   - python-chess, tqdm
#   - Rust toolchain + bullet-utils
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$TRAINING_DIR/data"

POSITIONS=1000000
THREADS=$(nproc)
LABEL_DEPTH=10
EPOCHS=400

while [[ $# -gt 0 ]]; do
    case $1 in
        --positions) POSITIONS="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --label-depth) LABEL_DEPTH="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "============================================================"
echo "KiyEngine NNUE Training Pipeline"
echo "============================================================"
echo "  Positions:    $POSITIONS"
echo "  Label depth:  $LABEL_DEPTH"
echo "  CPU threads:  $THREADS"
echo "  GPU epochs:   $EPOCHS"
echo ""

# ── Step 0: Install dependencies ──
echo "[Step 0] Checking dependencies..."

# Stockfish
if ! command -v stockfish &>/dev/null; then
    echo "  Installing Stockfish..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y -qq stockfish
    else
        echo "  ERROR: Cannot install Stockfish automatically. Please install manually."
        exit 1
    fi
fi
echo "  ✓ Stockfish: $(stockfish --help 2>&1 | head -1 || echo 'installed')"

# Python deps
pip install -q python-chess tqdm 2>/dev/null || pip install --user -q python-chess tqdm
echo "  ✓ Python deps installed"

# Rust + Bullet
if ! command -v cargo &>/dev/null; then
    echo "  Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi
echo "  ✓ Rust: $(rustc --version)"

BULLET_UTILS="$HOME/bullet/target/release/bullet-utils"
if [ ! -f "$BULLET_UTILS" ]; then
    echo "  Building bullet-utils (one-time)..."
    cd "$HOME"
    if [ ! -d bullet ]; then
        git clone https://github.com/jw1912/bullet.git
    fi
    cd bullet
    cargo build --release --package bullet-utils 2>&1 | tail -3
    echo "  ✓ bullet-utils built"
fi

# Build trainer
cd "$TRAINING_DIR"
echo "  Building KiyEngine trainer..."
cargo build --release 2>&1 | tail -3
echo "  ✓ Trainer built"

mkdir -p "$DATA_DIR"

# ── Step 1: Generate SF-labeled data ──
echo ""
echo "[Step 1] Generating $POSITIONS SF-labeled positions..."
echo "  (This takes ~2-4 hours for 1M positions on 8 cores)"
echo ""

python3 "$SCRIPT_DIR/generate_sf_data.py" \
    --positions "$POSITIONS" \
    --output "$DATA_DIR/train_raw.txt" \
    --threads "$THREADS" \
    --label-depth "$LABEL_DEPTH"

# ── Step 2: Shuffle ──
echo ""
echo "[Step 2] Shuffling data..."
shuf "$DATA_DIR/train_raw.txt" -o "$DATA_DIR/train_shuffled.txt"

# ── Step 3: Split train/test (95/5) ──
echo "[Step 3] Splitting train/test..."
TOTAL=$(wc -l < "$DATA_DIR/train_shuffled.txt")
TEST_SIZE=$((TOTAL / 20))
TRAIN_SIZE=$((TOTAL - TEST_SIZE))

head -n "$TRAIN_SIZE" "$DATA_DIR/train_shuffled.txt" > "$DATA_DIR/train.txt"
tail -n "$TEST_SIZE" "$DATA_DIR/train_shuffled.txt" > "$DATA_DIR/test.txt"
echo "  Train: $TRAIN_SIZE | Test: $TEST_SIZE"

# ── Step 4: Convert to bulletformat ──
echo ""
echo "[Step 4] Converting to bulletformat..."
"$BULLET_UTILS" convert --from text --input "$DATA_DIR/train.txt" --output "$DATA_DIR/train.bullet" --threads "$THREADS"
"$BULLET_UTILS" convert --from text --input "$DATA_DIR/test.txt" --output "$DATA_DIR/test.bullet" --threads "$THREADS"
echo "  ✓ Created train.bullet and test.bullet"

# Cleanup intermediate files
rm -f "$DATA_DIR/train_raw.txt" "$DATA_DIR/train_shuffled.txt"

# ── Step 5: Train on GPU ──
echo ""
echo "[Step 5] Training NNUE on GPU ($EPOCHS epochs)..."
echo ""

"$TRAINING_DIR/target/release/train" \
    --data "$DATA_DIR/train.bullet" \
    --test "$DATA_DIR/test.bullet" \
    --id "kiyengine-v7-sf${LABEL_DEPTH}" \
    --epochs "$EPOCHS" \
    --lr 0.001 \
    --batch-size 16384

echo ""
echo "============================================================"
echo "Training complete!"
echo "============================================================"
echo ""
echo "Checkpoints in: $TRAINING_DIR/checkpoints/"
echo ""
echo "To use the best network:"
echo "  python3 scripts/convert_to_kynn.py checkpoints/kiyengine-v7-sf${LABEL_DEPTH}-<best>/quantised.bin --buckets 8"
echo "  cp kiyengine.nnue ../../kiyengine.nnue"
