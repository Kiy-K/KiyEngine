#!/usr/bin/env python3
"""
Generate NNUE Training Data using Stockfish Labeling

Generates 1M+ training positions by:
  1. Playing Stockfish vs itself at low depth (fast, diverse positions)
  2. Labeling each position with Stockfish at depth 10 (accurate eval)
  3. Output in Bullet text format: `FEN | score_cp | result`

This is the #1 highest-impact improvement for KiyEngine's NNUE.
Previous training used ~400K self-play positions. This generates 10-50x more
with much higher quality labels from Stockfish depth 10.

Usage:
  # Generate 1M positions (takes ~2-4 hours on 8-core CPU)
  python generate_sf_data.py --positions 1000000 --output data/train.txt --threads 8

  # Quick test run
  python generate_sf_data.py --positions 10000 --output data/test_gen.txt --threads 4

  # Then convert + train:
  bullet-utils convert --from text --input data/train.txt --output data/train.bullet --threads 4
  ../scripts/train.sh --epochs 400

Requirements:
  pip install python-chess tqdm

Environment:
  STOCKFISH_PATH: path to Stockfish binary (default: stockfish)
"""

import argparse
import os
import sys
import random
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

try:
    import chess
    import chess.engine
    import chess.pgn
except ImportError:
    print("ERROR: python-chess required. Install: pip install python-chess")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ============================================================================
# CONFIG
# ============================================================================

# Self-play generation settings
PLAY_DEPTH = 4           # Depth for self-play moves (fast, creates diverse positions)
LABEL_DEPTH = 10         # Depth for position labeling (accurate eval)
LABEL_NODES = None       # Alternative: limit by nodes instead of depth

# Position filtering
MIN_PLY = 8              # Skip first 8 half-moves (opening book territory)
MAX_PLY = 300            # Maximum game length
MIN_PIECES = 4           # Skip positions with fewer than 4 pieces (TB territory)
SKIP_CHECK = True        # Skip positions where side to move is in check
SAMPLE_RATE = 0.5        # Sample 50% of eligible positions (reduce correlation)
MAX_ABS_EVAL = 3000      # Skip positions with |eval| > 3000 (decided games)

# Game generation settings
RANDOM_PLY_RANGE = (4, 8)  # Random moves at start for diversity
HASH_MB = 64              # SF hash per instance

# Stockfish path
DEFAULT_SF_PATH = os.environ.get("STOCKFISH_PATH", "stockfish")


def get_sf_path():
    """Find Stockfish binary."""
    # Check common locations
    candidates = [
        DEFAULT_SF_PATH,
        "/usr/local/bin/stockfish",
        "/usr/bin/stockfish",
        os.path.expanduser("~/stockfish/stockfish"),
        os.path.expanduser("~/Documents/Chess_engine/stockfish-ubuntu-x86-64-avx2/stockfish/stockfish-ubuntu-x86-64-avx2"),
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return DEFAULT_SF_PATH  # Let it fail with a clear error


def result_to_float(outcome):
    """Convert chess.Outcome to white-relative float."""
    if outcome is None:
        return 0.5  # Unknown → draw
    if outcome.winner == chess.WHITE:
        return 1.0
    elif outcome.winner == chess.BLACK:
        return 0.0
    return 0.5


def generate_game_positions(sf_path, game_id, seed):
    """Generate training positions from a single self-play game.
    
    Returns list of (fen, score_cp, result) tuples.
    """
    random.seed(seed)
    positions = []
    
    try:
        engine = chess.engine.SimpleEngine.popen_uci(sf_path)
        engine.configure({"Hash": HASH_MB, "Threads": 1})
    except Exception as e:
        print(f"  [Game {game_id}] Failed to start SF: {e}", file=sys.stderr)
        return positions

    try:
        board = chess.Board()
        
        # Phase 1: Random opening moves for diversity
        num_random = random.randint(*RANDOM_PLY_RANGE)
        for _ in range(num_random):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(random.choice(legal))
        
        # Phase 2: Self-play with SF at low depth
        ply = len(board.move_stack)
        while not board.is_game_over(claim_draw=True) and ply < MAX_PLY:
            try:
                result = engine.play(board, chess.engine.Limit(depth=PLAY_DEPTH))
                if result.move is None:
                    break
                board.push(result.move)
                ply += 1
            except Exception:
                break
        
        # Get game result
        outcome = board.outcome(claim_draw=True)
        game_result = result_to_float(outcome)
        
        # Phase 3: Walk through game and label positions with SF at high depth
        replay = chess.Board()
        for i, move in enumerate(board.move_stack):
            replay.push(move)
            
            # Filter
            if i < MIN_PLY:
                continue
            if len(replay.piece_map()) < MIN_PIECES:
                continue
            if SKIP_CHECK and replay.is_check():
                continue
            if random.random() > SAMPLE_RATE:
                continue
            
            # Label with SF at high depth
            try:
                info = engine.analyse(replay, chess.engine.Limit(depth=LABEL_DEPTH))
                score = info["score"].white()
                
                if score.is_mate():
                    # Convert mate scores to high cp values
                    mate_in = score.mate()
                    if mate_in > 0:
                        score_cp = 29000 - mate_in * 100
                    else:
                        score_cp = -29000 - mate_in * 100
                else:
                    score_cp = score.score()
                
                if score_cp is None or abs(score_cp) > MAX_ABS_EVAL:
                    continue
                
                fen = replay.fen()
                positions.append((fen, score_cp, game_result))
                
            except Exception:
                continue
    
    except Exception as e:
        print(f"  [Game {game_id}] Error: {e}", file=sys.stderr)
    finally:
        try:
            engine.quit()
        except:
            pass
    
    return positions


def worker_batch(args):
    """Worker function for parallel game generation."""
    sf_path, game_ids, base_seed = args
    all_positions = []
    for gid in game_ids:
        positions = generate_game_positions(sf_path, gid, base_seed + gid)
        all_positions.extend(positions)
    return all_positions


def main():
    parser = argparse.ArgumentParser(
        description="Generate NNUE training data using Stockfish labeling"
    )
    parser.add_argument("--positions", type=int, default=1_000_000,
                        help="Target number of positions (default: 1M)")
    parser.add_argument("--output", type=str, default="data/train.txt",
                        help="Output file (Bullet text format)")
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of parallel SF instances")
    parser.add_argument("--sf-path", type=str, default=None,
                        help="Path to Stockfish binary")
    parser.add_argument("--play-depth", type=int, default=PLAY_DEPTH,
                        help="Depth for self-play moves (default: 4)")
    parser.add_argument("--label-depth", type=int, default=LABEL_DEPTH,
                        help="Depth for position labeling (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--append", action="store_true",
                        help="Append to output file instead of overwriting")
    args = parser.parse_args()

    global PLAY_DEPTH, LABEL_DEPTH
    PLAY_DEPTH = args.play_depth
    LABEL_DEPTH = args.label_depth

    sf_path = args.sf_path or get_sf_path()
    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Estimate games needed (~30-50 positions per game after filtering)
    avg_pos_per_game = 30
    estimated_games = int(args.positions / avg_pos_per_game * 1.2)  # 20% overestimate
    
    print("=" * 60)
    print("KiyEngine NNUE Data Generator (Stockfish Labeling)")
    print("=" * 60)
    print(f"  Target positions:  {args.positions:,}")
    print(f"  Estimated games:   {estimated_games:,}")
    print(f"  Play depth:        {PLAY_DEPTH}")
    print(f"  Label depth:       {LABEL_DEPTH}")
    print(f"  Threads:           {args.threads}")
    print(f"  Stockfish:         {sf_path}")
    print(f"  Output:            {args.output}")
    print()

    # Verify SF exists
    try:
        engine = chess.engine.SimpleEngine.popen_uci(sf_path)
        sf_name = engine.id.get("name", "Stockfish")
        print(f"  ✓ {sf_name} ready")
        engine.quit()
    except Exception as e:
        print(f"  ✗ Cannot start Stockfish: {e}")
        print(f"    Set STOCKFISH_PATH or use --sf-path")
        sys.exit(1)

    # Distribute games across workers in batches
    batch_size = max(1, 10 // args.threads + 1)  # Games per worker per batch
    total_positions = 0
    game_counter = 0
    start_time = time.time()

    mode = "a" if args.append else "w"
    with open(args.output, mode, encoding="utf-8") as out_f:
        # Use process pool for true parallelism
        with ProcessPoolExecutor(max_workers=args.threads) as pool:
            futures = []
            
            # Submit initial batches
            for _ in range(args.threads * 2):
                game_ids = list(range(game_counter, game_counter + batch_size))
                game_counter += batch_size
                futures.append(pool.submit(worker_batch, (sf_path, game_ids, args.seed)))
            
            if tqdm:
                pbar = tqdm(total=args.positions, unit="pos", desc="Generating")
            else:
                pbar = None
            
            while total_positions < args.positions:
                # Wait for any future to complete
                done = []
                for f in futures:
                    if f.done():
                        done.append(f)
                
                if not done:
                    # Wait for the first one
                    import concurrent.futures
                    completed = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    done = list(completed.done)
                
                for f in done:
                    futures.remove(f)
                    try:
                        positions = f.result()
                        for fen, score_cp, result in positions:
                            out_f.write(f"{fen} | {score_cp} | {result:.1f}\n")
                            total_positions += 1
                        
                        if pbar:
                            pbar.update(len(positions))
                        else:
                            elapsed = time.time() - start_time
                            rate = total_positions / max(elapsed, 0.1)
                            eta = (args.positions - total_positions) / max(rate, 0.1)
                            print(f"  {total_positions:,}/{args.positions:,} positions "
                                  f"({rate:.0f}/s, ETA: {eta/60:.0f}min)")
                    except Exception as e:
                        print(f"  Worker error: {e}", file=sys.stderr)
                    
                    # Submit replacement batch
                    if total_positions < args.positions:
                        game_ids = list(range(game_counter, game_counter + batch_size))
                        game_counter += batch_size
                        futures.append(pool.submit(worker_batch, (sf_path, game_ids, args.seed)))
            
            if pbar:
                pbar.close()
            
            # Cancel remaining futures
            for f in futures:
                f.cancel()

    elapsed = time.time() - start_time
    print()
    print(f"{'=' * 60}")
    print(f"  Generated:    {total_positions:,} positions")
    print(f"  Games played: {game_counter}")
    print(f"  Time:         {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
    print(f"  Rate:         {total_positions/max(elapsed,1):.0f} positions/sec")
    print(f"  Output:       {args.output}")
    print(f"{'=' * 60}")
    print()
    print("Next steps:")
    print(f"  1. Shuffle:    shuf {args.output} -o {args.output}")
    print(f"  2. Convert:    bullet-utils convert --from text --input {args.output} --output data/train.bullet")
    print(f"  3. Train:      ./scripts/train.sh --epochs 400")


if __name__ == "__main__":
    main()
