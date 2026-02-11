#!/bin/bash
# ============================================================
# KiyEngine NNUE Training — Lightning.AI L40S GPU Setup
# ============================================================
#
# Run this on a Lightning.AI Studio with L40S (48GB VRAM).
# Installs Rust, CUDA deps, Bullet framework, and builds the trainer.
#
# Usage:
#   chmod +x scripts/setup_lightning.sh
#   ./scripts/setup_lightning.sh
#
# After setup:
#   ./scripts/train_gpu.sh
# ============================================================

set -euo pipefail

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  KiyEngine NNUE Training — Lightning.AI L40S Setup      ║"
echo "║  Architecture: (768→512)×2 → SCReLU → 8 output buckets ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$TRAINING_DIR")"

# ── 1. System dependencies ──
echo "[1/7] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential pkg-config libssl-dev zstd cmake

# ── 2. Rust toolchain ──
if ! command -v cargo &> /dev/null; then
    echo "[2/7] Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "[2/7] Rust already installed: $(rustc --version)"
    source "$HOME/.cargo/env" 2>/dev/null || true
fi

# ── 3. CUDA check ──
echo "[3/7] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo ""
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
    
    # Find CUDA path
    CUDA_PATH=""
    for p in /usr/local/cuda /usr/local/cuda-12* /usr/local/cuda-11*; do
        if [ -d "$p" ]; then
            CUDA_PATH="$p"
            break
        fi
    done
    
    if [ -n "$CUDA_PATH" ]; then
        echo "  CUDA: $CUDA_PATH"
        export PATH="$CUDA_PATH/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"
        nvcc --version 2>/dev/null | grep release || echo "  nvcc not found (will use runtime CUDA)"
    else
        echo "  WARNING: CUDA toolkit not found in /usr/local/cuda*"
        echo "  The Bullet GPU backend requires CUDA. Attempting to continue..."
    fi
else
    echo "  ⚠ No GPU detected! Training will be EXTREMELY slow on CPU."
    echo "  Make sure you're on a Lightning.AI Studio with L40S."
fi

# ── 4. Python dependencies (for data pipeline) ──
echo "[4/7] Installing Python dependencies..."
pip install --quiet python-chess zstandard requests tqdm

# ── 5. Clone Bullet trainer framework ──
BULLET_DIR="$HOME/bullet"
if [ ! -d "$BULLET_DIR" ]; then
    echo "[5/7] Cloning Bullet framework..."
    git clone --depth 1 https://github.com/jw1912/bullet.git "$BULLET_DIR"
else
    echo "[5/7] Bullet already cloned at $BULLET_DIR"
    cd "$BULLET_DIR" && git pull --ff-only 2>/dev/null || true && cd -
fi

# Build bullet-utils for data conversion
echo "  Building bullet-utils..."
cd "$BULLET_DIR"
cargo build --release --package bullet-utils 2>&1 | tail -5
BULLET_UTILS="$BULLET_DIR/target/release/bullet-utils"
echo "  bullet-utils: $BULLET_UTILS"
cd "$TRAINING_DIR"

# ── 6. Build KiyEngine NNUE trainer (GPU) ──
echo "[6/7] Building KiyEngine NNUE trainer with CUDA backend..."
cd "$TRAINING_DIR"

# Check if Cargo.toml has cuda feature, if not, we'll use CPU with threads
if grep -q 'cuda' Cargo.toml 2>/dev/null; then
    echo "  Building with --features cuda..."
    cargo build --release --features cuda 2>&1 | tail -5
else
    echo "  No cuda feature in Cargo.toml. Building CPU backend..."
    cargo build --release 2>&1 | tail -5
fi

# ── 7. Create directories ──
echo "[7/7] Creating directory structure..."
mkdir -p "$TRAINING_DIR/data"
mkdir -p "$TRAINING_DIR/checkpoints"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Setup complete!                                         ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Project:      $PROJECT_DIR"
echo "  Trainer:      $TRAINING_DIR"
echo "  Data dir:     $TRAINING_DIR/data/"
echo "  Checkpoints:  $TRAINING_DIR/checkpoints/"
echo "  Bullet:       $BULLET_DIR"
echo "  bullet-utils: $BULLET_UTILS"
echo ""
echo "Next steps:"
echo "  1. Prepare data:    ./scripts/download_and_convert.sh"
echo "  2. Train (GPU):     ./scripts/train_gpu.sh"
echo "  3. Convert to KYNN: python3 scripts/convert_to_kynn.py checkpoints/<net>/quantised.bin kiyengine.nnue --buckets 8"
