#!/bin/bash
set -e

# ============================================================================
#  KiyEngine Grand Tournament — 8-Engine Round-Robin
# ============================================================================
#
#  Engine Lineup (estimated Elo):
#    1. Stockfish 1500   (~1500)  — handicapped
#    2. KiyEngine V6.1   (~2000)  — our hero
#    3. Stockfish 2000   (~2000)  — KiyEngine's rival
#    4. GNU Chess         (~2200)  — classic engine
#    5. Stockfish 2500   (~2500)  — mid-tier
#    6. Ethereal          (~3000)  — strong open-source
#    7. Stockfish 3000   (~3000)  — strong handicapped
#    8. Komodo 14         (~3400)  — tournament boss
#
#  Format: Round-Robin, 2 games per pairing (1 white, 1 black)
#  Total: 8×7/2 × 2 = 56 games
# ============================================================================

# --- Paths ---
CUTECHESS="/home/khoi/cutechess/build/cutechess-cli"
KIYENGINE="/home/khoi/Workspace/KiyEngine/target/release/kiy_engine"
STOCKFISH="/home/khoi/Documents/Chess_engine/stockfish-ubuntu-x86-64-avx2/stockfish/stockfish-ubuntu-x86-64-avx2"
ETHEREAL="/home/khoi/Documents/Chess_engine/Ethereal/src/Ethereal-avx2"
KOMODO="/home/khoi/Documents/Chess_engine/komodo-14/komodo-14_224afb/Linux/komodo-14.1-linux"
GNUCHESS="/usr/games/gnuchess"

# --- Settings ---
TC="40/30"          # 30 seconds for 40 moves (fast but playable)
HASH=128            # MB per engine
PGN_OUT="tournament.pgn"
CONCURRENCY=2       # Run 2 games in parallel

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          KiyEngine Grand Tournament — Round Robin           ║"
echo "║   8 Engines • 56 Games • TC ${TC}                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# 1. Build KiyEngine
echo "Building KiyEngine V6.1 (Release)..."
cargo build --release 2>&1 | tail -1

# 2. Clean old PGN
[ -f "$PGN_OUT" ] && rm "$PGN_OUT"

echo ""
echo "Engines:"
echo "  1. Stockfish 1500  (SF handicapped to ~1500 Elo)"
echo "  2. KiyEngine V6.1  (our engine)"
echo "  3. Stockfish 2000  (SF handicapped to ~2000 Elo)"
echo "  4. GNU Chess       (classic ~2200 Elo)"
echo "  5. Stockfish 2500  (SF handicapped to ~2500 Elo)"
echo "  6. Ethereal 14.40  (strong open-source ~3000)"
echo "  7. Stockfish 3000  (SF handicapped to ~3000 Elo)"
echo "  8. Komodo 14       (tournament boss ~3400)"
echo ""
echo "Starting tournament..."
echo ""

# 3. Run Round-Robin Tournament
$CUTECHESS \
    -engine name="SF_1500"        cmd="$STOCKFISH" proto=uci \
        option.Threads=1 option.Hash=$HASH \
        option.UCI_LimitStrength=true option.UCI_Elo=1500 \
    -engine name="KiyEngine_V6.1" cmd="$KIYENGINE" dir=/home/khoi/Workspace/KiyEngine proto=uci \
        option.Threads=1 option.Hash=$HASH \
    -engine name="SF_2000"        cmd="$STOCKFISH" proto=uci \
        option.Threads=1 option.Hash=$HASH \
        option.UCI_LimitStrength=true option.UCI_Elo=2000 \
    -engine name="GNU_Chess"      cmd="$GNUCHESS" arg="--uci" proto=uci \
        option.Hash=$HASH \
    -engine name="SF_2500"        cmd="$STOCKFISH" proto=uci \
        option.Threads=1 option.Hash=$HASH \
        option.UCI_LimitStrength=true option.UCI_Elo=2500 \
    -engine name="Ethereal"       cmd="$ETHEREAL" proto=uci \
        option.Threads=1 option.Hash=$HASH \
    -engine name="SF_3000"        cmd="$STOCKFISH" proto=uci \
        option.Threads=1 option.Hash=$HASH \
        option.UCI_LimitStrength=true option.UCI_Elo=3000 \
    -engine name="Komodo_14"      cmd="$KOMODO" proto=uci \
        option.Threads=1 option.Hash=$HASH \
    -each tc=$TC \
    -games 2 \
    -rounds 1 \
    -tournament round-robin \
    -pgnout "$PGN_OUT" \
    -recover \
    -concurrency $CONCURRENCY \
    -ratinginterval 10

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Tournament complete! Results saved to $PGN_OUT"
echo "════════════════════════════════════════════════════════"
