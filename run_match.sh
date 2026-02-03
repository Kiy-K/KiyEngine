#!/bin/bash
set -e

# Configuration
ENGINE_BIN="/home/khoi/Workspace/KiyEngine/target/release/kiy_engine_v4_omega"
CUTECHESS="/home/khoi/cutechess/build/cutechess-cli"
STOCKFISH="/usr/games/stockfish"
MODEL_FILE="/home/khoi/Workspace/KiyEngine/kiyengine_v4.safetensors"
TC="40/60" # Blitz: 60 seconds for 40 moves
GAMES=2

echo "=== KiyEngine V4 Omega Match Script ==="

# 1. Check if model exists
if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ Error: $MODEL_FILE not found!"
    exit 1
fi

# 2. Build Engine
echo "🦀 Building KiyEngine V4 Omega (Release)..."
cargo build --release

# 3. Run Match
echo "⚔️  Starting Match: KiyEngine V4 Omega vs Stockfish"
echo "   Time Control: $TC"
echo "   Games: $GAMES"

$CUTECHESS \
    -engine name="KiyEngine_V4_Omega" cmd=$ENGINE_BIN dir=. proto=uci \
    -engine name="Stockfish" cmd=$STOCKFISH proto=uci \
    -each tc=$TC \
    -rounds $GAMES \
    -games 2 \
    -repeat \
    -pgnout omega_match.pgn \
    -recover \
    -concurrency 1 \
    -ratinginterval 1 \
    -debug

echo "✅ Match complete! Results saved to omega_match.pgn"
