#!/bin/bash
set -e

# Configuration for quick test
ENGINE_BIN="/home/khoi/Workspace/KiyEngine/target/release/kiy_engine_v4_omega"
CUTECHESS="/home/khoi/cutechess/build/cutechess-cli"
STOCKFISH="/usr/games/stockfish"
MODEL_FILE="/home/khoi/Workspace/KiyEngine/kiyengine.gguf"
TC="5/10" # Very fast: 10 seconds for 5 moves
GAMES=1

echo "=== Quick KiyEngine vs Stockfish Test ==="

# Check if model exists
if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ Error: $MODEL_FILE not found!"
    exit 1
fi

# Build Engine (already done, but just in case)
echo "🦀 Ensuring KiyEngine is built..."
cargo build --release --quiet

# Run Quick Match
echo "⚔️  Starting Quick Match: KiyEngine V5 vs Stockfish"
echo "   Time Control: $TC"
echo "   Games: $GAMES"

$CUTECHESS \
    -engine name="KiyEngine_V5" cmd=$ENGINE_BIN dir=. proto=uci \
    -engine name="Stockfish" cmd=$STOCKFISH proto=uci \
    -each tc=$TC \
    -rounds $GAMES \
    -games 2 \
    -repeat \
    -pgnout quick_test.pgn \
    -recover \
    -concurrency 1 \
    -ratinginterval 1

echo "✅ Quick test complete! Results saved to quick_test.pgn"

# Show results
if [ -f "quick_test.pgn" ]; then
    echo ""
    echo "📊 Game Results:"
    cat quick_test.pgn
fi
