#!/bin/bash
set -e

# Configuration
ENGINE_BIN="/home/khoi/Workspace/KiyEngine/target/release/kiy_engine_v5_alpha"
CUTECHESS="/home/khoi/cutechess/build/cutechess-cli"
STOCKFISH="/usr/games/stockfish"
MODEL_FILE="/home/khoi/Workspace/KiyEngine/kiyengine.gguf"
TC="40/60" 
GAMES=4 # Tăng lên 4 ván cho nó máu (2 Trắng, 2 Đen)
THREADS=4 # Cân bằng cho con Vivobook của ông
HASH=512

echo "=== KiyEngine V5 Match Script (Fair Play Edition) ==="

# 1. Check if model exists
if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ Error: $MODEL_FILE not found!"
    exit 1
fi

# 2. Build Engine
echo "🦀 Building KiyEngine V5 (Release)..."
cargo build --release

# 3. Clean old PGN
[ -f omega_match.pgn ] && rm omega_match.pgn

# 4. Run Match
echo "⚔️  Starting Match: KiyEngine V5 vs Stockfish"
echo "   Threads: $THREADS | Hash: ${HASH}MB | TC: $TC"

$CUTECHESS \
    -engine name="KiyEngine_V5" cmd=$ENGINE_BIN dir=. proto=uci \
    option.Threads=$THREADS option.Hash=$HASH \
    -engine name="Stockfish" cmd=$STOCKFISH proto=uci \
    option.Threads=$THREADS option.Hash=$HASH \
    -each tc=$TC \
    -rounds $GAMES \
    -repeat \
    -pgnout omega_match.pgn \
    -recover \
    -concurrency 1 \
    -ratinginterval 1 \
    -debug

echo "✅ Match complete! Results saved to omega_match.pgn"