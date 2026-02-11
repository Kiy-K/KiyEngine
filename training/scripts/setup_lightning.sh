#!/bin/bash
# ============================================================
# KiyEngine NNUE Training — Lightning.AI L4 GPU Setup
# ============================================================
#
# Run this script on a Lightning.AI Studio with L4 GPU.
# It installs all dependencies and prepares the training environment.
#
# Usage:
#   chmod +x setup_lightning.sh
#   ./setup_lightning.sh
#
# After setup, run:
#   ./train.sh
# ============================================================

set -euo pipefail

echo "=== KiyEngine NNUE Training Setup (Lightning.AI L4) ==="
echo ""

# ── 1. System dependencies ──
echo "[1/6] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential pkg-config libssl-dev zstd

# ── 2. Rust toolchain ──
if ! command -v cargo &> /dev/null; then
    echo "[2/6] Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "[2/6] Rust already installed: $(rustc --version)"
fi

# ── 3. CUDA check ──
echo "[3/6] Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo "CUDA toolkit: $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not found — will use system CUDA')"
else
    echo "WARNING: No GPU detected. Training will fall back to CPU (very slow!)."
    echo "Make sure you're running on a Lightning.AI Studio with L4 GPU."
fi

# ── 4. Python dependencies (for data pipeline) ──
echo "[4/6] Installing Python dependencies..."
pip install --quiet python-chess zstandard requests tqdm

# ── 5. Clone Bullet (for bullet-utils) ──
BULLET_DIR="$HOME/bullet"
if [ ! -d "$BULLET_DIR" ]; then
    echo "[5/6] Cloning Bullet framework..."
    git clone --depth 1 https://github.com/jw1912/bullet.git "$BULLET_DIR"
else
    echo "[5/6] Bullet already cloned at $BULLET_DIR"
    cd "$BULLET_DIR" && git pull --ff-only && cd -
fi

# Build bullet-utils for data conversion
echo "  Building bullet-utils..."
cd "$BULLET_DIR"
cargo build --release --package bullet-utils 2>&1 | tail -3
cd -

# ── 6. Build KiyEngine NNUE trainer ──
echo "[6/6] Building KiyEngine NNUE trainer..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"
cd "$TRAINING_DIR"
cargo build --release 2>&1 | tail -3

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Directory structure:"
echo "  $TRAINING_DIR/          — Trainer source"
echo "  $TRAINING_DIR/data/     — Training data (will be created)"
echo "  $TRAINING_DIR/checkpoints/ — Saved networks"
echo "  $BULLET_DIR/            — Bullet framework + bullet-utils"
echo ""
echo "Next steps:"
echo "  1. Prepare data:  ./scripts/download_and_convert.sh"
echo "  2. Train:         ./scripts/train.sh"
