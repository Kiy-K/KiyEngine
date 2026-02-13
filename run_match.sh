#!/bin/bash
set -e

# Configuration
ENGINE_BIN="/home/khoi/Workspace/KiyEngine/target/release/kiy_engine"
CUTECHESS="/home/khoi/cutechess/build/cutechess-cli"
STOCKFISH="/home/khoi/Documents/Chess_engine/stockfish-ubuntu-x86-64-avx2/stockfish/stockfish-ubuntu-x86-64-avx2"
TC="40/60"
GAMES=4
THREADS=4
HASH=512

echo "=== KiyEngine V6.1.0 vs Stockfish 15 ==="

# 1. Build Engine
echo "Building KiyEngine V6.1.0 (Release)..."
cargo build --release

# 2. Clean old PGN
[ -f omega_match.pgn ] && rm omega_match.pgn

# 3. Run Match
echo "Starting Match: KiyEngine V6.1.0 vs Stockfish"
echo "   Threads: $THREADS | Hash: ${HASH}MB | TC: $TC | Games: $GAMES"

$CUTECHESS \
    -engine name="KiyEngine_V6.1" cmd=$ENGINE_BIN dir=. proto=uci \
    option.Threads=$THREADS option.Hash=$HASH \
    -engine name="Stockfish" cmd=$STOCKFISH proto=uci \
    option.Threads=$THREADS option.Hash=$HASH \
    -each tc=$TC \
    -rounds $GAMES \
    -repeat \
    -pgnout omega_match.pgn \
    -recover \
    -concurrency 1 \
    -ratinginterval 1

echo "Match complete! Results saved to omega_match.pgn"