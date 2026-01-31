#!/bin/bash
set -e

# Configuration
ENGINE_BIN="./target/release/kiy_engine_v3"
CUTECHESS="/home/khoi/cutechess/build/cutechess-cli"
STOCKFISH="/usr/games/stockfish"
MODEL_FILE="model.safetensors"
TC="60+0.6" # Rapid 1m+0.6s
GAMES=10

echo "=== KiyEngine V3 Match Script ==="

# 1. Check if model exists
if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ Error: $MODEL_FILE not found!"
    exit 1
fi

# 2. Patch weights if needed
echo "🛠️  Checking and patching weights..."
./.venv/bin/python3 scripts/patch_weights.py --model "$MODEL_FILE"

# 3. Build Engine
echo "🦀 Building KiyEngine V3 (Release)..."
cargo build --release

# 4. Run Match
echo "⚔️  Starting Match: KiyEngine V3 vs Stockfish 17"
echo "   Time Control: $TC"
echo "   Games: $GAMES"

$CUTECHESS \
    -engine name=KiyEngine cmd=$ENGINE_BIN dir=. proto=uci \
    -engine name=Stockfish cmd=$STOCKFISH proto=uci \
    -each tc=$TC \
    -rounds $GAMES \
    -games 2 \
    -repeat \
    -pgnout match_results.pgn \
    -recover \
    -concurrency 1 \
    -ratinginterval 10 \
    -debug

echo "✅ Match complete! Results saved to match_results.pgn"
