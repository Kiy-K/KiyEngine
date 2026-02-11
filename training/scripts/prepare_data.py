#!/usr/bin/env python3
"""
Lichess PGN → NNUE Training Data Pipeline

Two data sources, two strategies:
  1. Broadcast PGNs (~95% computer-analyzed): extract existing [%eval] annotations.
     No Stockfish needed — fast, high-quality labels from Lichess analysis.
  2. Standard-rated PGNs (~5% analyzed): label positions with Stockfish depth 12.
     Use --positions to cap output (e.g. 3M positions).

Output: Bullet text format `<FEN> | <score_cp> | <result>`
Convert: bullet-utils convert --from text --input train.txt --output train.bullet

Usage:
  # Broadcast games — extract [%eval] annotations (fast, no SF):
  python prepare_data.py --download 2025-01 --output data/broadcast.txt
  python prepare_data.py --download-all --output data/broadcast.txt

  # Standard-rated games — label with SF depth 12 (~3M positions):
  python prepare_data.py --download-standard 2025-01 --output data/standard.txt \
      --label-depth 12 --threads 8 --min-elo 2200 --positions 3000000

  # Local PGN file (auto-detect [%eval] or use SF):
  python prepare_data.py --input games.pgn.zst --output data/train.txt --mode broadcast
  python prepare_data.py --input games.pgn.zst --output data/train.txt --mode standard

Requirements:
  pip install python-chess zstandard requests tqdm
  Stockfish binary needed only for standard-rated games
"""

import argparse
import io
import os
import sys
import random

try:
    import chess
    import chess.pgn
except ImportError:
    print("ERROR: python-chess required. Install: pip install python-chess")
    sys.exit(1)

try:
    import zstandard as zstd
except ImportError:
    zstd = None

try:
    import requests
except ImportError:
    requests = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import chess.engine
except ImportError:
    pass  # Handled when --label-depth is used


BROADCAST_BASE = "https://database.lichess.org/broadcast"
BROADCAST_LIST = f"{BROADCAST_BASE}/list.txt"
STANDARD_BASE = "https://database.lichess.org/standard"

# Positions to skip (opening, endgame with few pieces)
MIN_PLY = 16          # Skip first 8 full moves (opening book territory)
MIN_PIECES = 6        # Skip positions with fewer than 6 pieces
MAX_POSITIONS_PER_GAME = 80  # Cap positions per game
SAMPLE_RATE = 0.6     # Sample 60% of eligible positions (reduce correlation)
MAX_ABS_EVAL = 3000   # Skip positions with |eval| > 3000cp (decided games)
SF_HASH_MB = 128      # Stockfish hash per instance


def get_sf_path(user_path=None):
    """Find Stockfish binary."""
    candidates = [
        user_path,
        os.environ.get("STOCKFISH_PATH"),
        "stockfish",
        "/usr/local/bin/stockfish",
        "/usr/bin/stockfish",
        os.path.expanduser("~/stockfish/stockfish"),
        os.path.expanduser("~/Documents/Chess_engine/stockfish-ubuntu-x86-64-avx2/stockfish/stockfish-ubuntu-x86-64-avx2"),
    ]
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return "stockfish"  # Let it fail with a clear error


def download_file(url: str, output_dir: str = "downloads") -> str:
    """Download a file from URL if not already cached locally."""
    if requests is None:
        print("ERROR: requests required for download. Install: pip install requests")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    filename = url.split("/")[-1]
    local_path = os.path.join(output_dir, filename)

    if os.path.exists(local_path):
        print(f"  Already downloaded: {local_path}")
        return local_path

    print(f"  Downloading {url} ...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    if tqdm:
        progress = tqdm(total=total, unit="B", unit_scale=True, desc=filename)
    else:
        progress = None

    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            if progress:
                progress.update(len(chunk))

    if progress:
        progress.close()

    print(f"  Saved: {local_path}")
    return local_path


def download_broadcast(month: str, output_dir: str = "downloads") -> str:
    """Download a Lichess broadcast PGN for a given month (e.g., '2025-01')."""
    filename = f"lichess_db_broadcast_{month}.pgn.zst"
    url = f"{BROADCAST_BASE}/{filename}"
    return download_file(url, output_dir)


def download_standard(month: str, output_dir: str = "downloads") -> str:
    """Download a Lichess standard-rated PGN for a given month.
    WARNING: These files are VERY large (multi-GB). Use with --min-elo to filter."""
    filename = f"lichess_db_standard_rated_{month}.pgn.zst"
    url = f"{STANDARD_BASE}/{filename}"
    return download_file(url, output_dir)


def list_available_months() -> list:
    """Fetch list of available broadcast months from Lichess."""
    if requests is None:
        print("ERROR: requests required. Install: pip install requests")
        sys.exit(1)

    resp = requests.get(BROADCAST_LIST)
    resp.raise_for_status()
    months = []
    for line in resp.text.strip().split("\n"):
        line = line.strip()
        if line and "broadcast" in line:
            # Extract month from URL like .../lichess_db_broadcast_2025-01.pgn.zst
            parts = line.split("_")
            for p in parts:
                if p.startswith("20") and ".pgn" in p:
                    month = p.replace(".pgn.zst", "")
                    months.append(month)
    return sorted(months)


def open_pgn(path: str):
    """Open a PGN file, handling .zst compression."""
    if path.endswith(".zst"):
        if zstd is None:
            print("ERROR: zstandard required for .zst files. Install: pip install zstandard")
            sys.exit(1)
        fh = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(fh)
        text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        return text_stream, fh
    else:
        fh = open(path, "r", encoding="utf-8", errors="replace")
        return fh, fh


def result_to_float(result_str: str) -> float:
    """Convert PGN result to white-relative float."""
    if result_str == "1-0":
        return 1.0
    elif result_str == "0-1":
        return 0.0
    elif result_str == "1/2-1/2":
        return 0.5
    else:
        return -1.0  # Unknown


def _check_elo(game, min_elo):
    """Return False if game should be skipped due to ELO filter."""
    if min_elo <= 0:
        return True
    try:
        white_elo = int(game.headers.get("WhiteElo", "0"))
        black_elo = int(game.headers.get("BlackElo", "0"))
        return white_elo >= min_elo and black_elo >= min_elo
    except ValueError:
        return True  # No ELO info — include anyway


def _parse_eval(comment):
    """Parse [%eval X.XX] or [%eval #N] from a PGN comment.
    Returns score in centipawns (int), or None if not found."""
    import re
    m = re.search(r'\[%eval\s+([^\]]+)\]', comment)
    if not m:
        return None
    val = m.group(1).strip()
    if val.startswith('#'):
        # Mate score: #N or #-N
        try:
            mate_in = int(val[1:])
            if mate_in > 0:
                return 29000 - mate_in * 100
            else:
                return -29000 - mate_in * 100
        except ValueError:
            return None
    try:
        return int(round(float(val) * 100))  # Convert pawns to centipawns
    except ValueError:
        return None


def extract_broadcast_positions(game, min_elo: int = 0) -> list:
    """Extract positions with existing [%eval] annotations from broadcast PGN.

    Returns list of (fen, score_cp, result) tuples — fully labeled, no SF needed.
    Positions without [%eval] are skipped.
    """
    result_str = game.headers.get("Result", "*")
    result = result_to_float(result_str)
    if result < 0:
        return []
    if not _check_elo(game, min_elo):
        return []

    positions = []
    node = game
    ply = 0

    while node.variations:
        node = node.variation(0)  # Follow mainline
        ply += 1

        if ply < MIN_PLY:
            continue

        board = node.board()
        if len(board.piece_map()) < MIN_PIECES:
            continue
        if board.is_check():
            continue
        if random.random() > SAMPLE_RATE:
            continue

        # Parse [%eval] from comment
        score_cp = _parse_eval(node.comment)
        if score_cp is None:
            continue
        if abs(score_cp) > MAX_ABS_EVAL:
            continue

        positions.append((board.fen(), score_cp, result))
        if len(positions) >= MAX_POSITIONS_PER_GAME:
            break

    return positions


def extract_fens_for_labeling(game, min_elo: int = 0) -> list:
    """Extract candidate FENs from a game that needs SF labeling.

    Returns list of (fen, result) tuples — score will be added by SF.
    """
    result_str = game.headers.get("Result", "*")
    result = result_to_float(result_str)
    if result < 0:
        return []
    if not _check_elo(game, min_elo):
        return []

    fens = []
    board = game.board()
    ply = 0

    for move in game.mainline_moves():
        board.push(move)
        ply += 1

        if ply < MIN_PLY:
            continue
        if len(board.piece_map()) < MIN_PIECES:
            continue
        if board.is_check():
            continue
        if random.random() > SAMPLE_RATE:
            continue

        fens.append((board.fen(), result))
        if len(fens) >= MAX_POSITIONS_PER_GAME:
            break

    return fens


def label_positions_sf(fens_with_results, sf_path, label_depth, threads=1):
    """Label a batch of (fen, result) tuples with Stockfish eval at given depth.

    Returns list of (fen, score_cp, result) tuples.
    Positions with |eval| > MAX_ABS_EVAL or mate scores are filtered.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import math

    if not fens_with_results:
        return []

    # Split into chunks for parallel labeling
    chunk_size = max(1, math.ceil(len(fens_with_results) / threads))
    chunks = [
        fens_with_results[i:i + chunk_size]
        for i in range(0, len(fens_with_results), chunk_size)
    ]

    results = []
    if threads <= 1 or len(chunks) <= 1:
        results = _label_chunk(sf_path, label_depth, fens_with_results)
    else:
        with ProcessPoolExecutor(max_workers=min(threads, len(chunks))) as pool:
            futures = {
                pool.submit(_label_chunk, sf_path, label_depth, chunk): i
                for i, chunk in enumerate(chunks)
            }
            for f in as_completed(futures):
                try:
                    results.extend(f.result())
                except Exception as e:
                    print(f"  Label worker error: {e}", file=sys.stderr)

    return results


def _label_chunk(sf_path, label_depth, fens_with_results):
    """Label a chunk of positions with a single Stockfish instance."""
    labeled = []
    try:
        engine = chess.engine.SimpleEngine.popen_uci(sf_path)
        engine.configure({"Hash": SF_HASH_MB, "Threads": 1})
    except Exception as e:
        print(f"  Failed to start Stockfish: {e}", file=sys.stderr)
        return labeled

    try:
        for fen, result in fens_with_results:
            try:
                board = chess.Board(fen)
                info = engine.analyse(board, chess.engine.Limit(depth=label_depth))
                score = info["score"].white()

                if score.is_mate():
                    mate_in = score.mate()
                    if mate_in > 0:
                        score_cp = 29000 - mate_in * 100
                    else:
                        score_cp = -29000 - mate_in * 100
                else:
                    score_cp = score.score()

                if score_cp is None or abs(score_cp) > MAX_ABS_EVAL:
                    continue

                labeled.append((fen, score_cp, result))
            except Exception:
                continue
    finally:
        try:
            engine.quit()
        except Exception:
            pass

    return labeled


def process_broadcast_pgn(pgn_path: str, output_path: str, min_elo: int = 0,
                          max_positions: int = 0) -> int:
    """Process a broadcast PGN: extract existing [%eval] annotations.
    No Stockfish needed — ~95% of broadcast games are computer-analyzed.
    """
    print(f"Processing (broadcast, extract [%%eval]): {pgn_path}")

    text_stream, raw_fh = open_pgn(pgn_path)
    total_positions = 0
    total_games = 0
    skipped_games = 0

    with open(output_path, "a", encoding="utf-8") as out:
        while True:
            if max_positions > 0 and total_positions >= max_positions:
                break
            try:
                game = chess.pgn.read_game(text_stream)
            except Exception:
                continue
            if game is None:
                break

            total_games += 1
            positions = extract_broadcast_positions(game, min_elo)

            if not positions:
                skipped_games += 1
                continue

            for fen, score, result in positions:
                out.write(f"{fen} | {score} | {result:.1f}\n")
                total_positions += 1
                if max_positions > 0 and total_positions >= max_positions:
                    break

            if total_games % 1000 == 0:
                print(f"  Games: {total_games}, Positions: {total_positions}, "
                      f"Skipped: {skipped_games}")

    raw_fh.close()
    print(f"  Done: {total_games} games → {total_positions} positions")
    return total_positions


def process_standard_pgn(pgn_path: str, output_path: str, min_elo: int = 0,
                          sf_path: str = None, label_depth: int = 12,
                          threads: int = 4, max_positions: int = 0) -> int:
    """Process a standard-rated PGN: label positions with Stockfish.
    Only ~5% of standard games have analysis — SF labeling is required.
    """
    print(f"Processing (standard, SF depth {label_depth}): {pgn_path}")
    if sf_path is None:
        print("  ERROR: --sf-path required for standard-rated games", file=sys.stderr)
        return 0

    text_stream, raw_fh = open_pgn(pgn_path)
    total_positions = 0
    total_games = 0
    skipped_games = 0

    GAME_BATCH = 500
    fen_buffer = []  # list of (fen, result)

    def flush_buffer():
        nonlocal total_positions, fen_buffer
        if not fen_buffer:
            return
        labeled = label_positions_sf(fen_buffer, sf_path, label_depth, threads)
        with open(output_path, "a", encoding="utf-8") as out:
            for fen, score, result in labeled:
                out.write(f"{fen} | {score} | {result:.1f}\n")
                total_positions += 1
        fen_buffer = []

    while True:
        if max_positions > 0 and total_positions >= max_positions:
            break
        try:
            game = chess.pgn.read_game(text_stream)
        except Exception:
            continue
        if game is None:
            break

        total_games += 1
        fens = extract_fens_for_labeling(game, min_elo)

        if not fens:
            skipped_games += 1
            continue

        fen_buffer.extend(fens)

        if total_games % GAME_BATCH == 0:
            flush_buffer()
            print(f"  Games: {total_games}, Positions: {total_positions}, "
                  f"Skipped: {skipped_games}")

    flush_buffer()
    raw_fh.close()
    print(f"  Done: {total_games} games → {total_positions} positions")
    return total_positions


def main():
    parser = argparse.ArgumentParser(
        description="Lichess PGN → NNUE Training Data"
    )
    # Input sources
    parser.add_argument("--input", type=str, help="Path to PGN file (.pgn or .pgn.zst)")
    parser.add_argument("--download", type=str, help="Download broadcast month (e.g., 2025-01)")
    parser.add_argument("--download-all", action="store_true", help="Download all broadcast months")
    parser.add_argument("--download-standard", type=str,
                        help="Download standard-rated month (WARNING: large files, use --min-elo)")
    parser.add_argument("--list-months", action="store_true", help="List available broadcast months")

    # Mode (auto-detected for --download/--download-standard)
    parser.add_argument("--mode", type=str, choices=["broadcast", "standard"], default=None,
                        help="Processing mode for --input (auto for download commands)")

    # Output
    parser.add_argument("--output", type=str, default="data/train.txt", help="Output text file path")
    parser.add_argument("--positions", type=int, default=0,
                        help="Max positions to extract (0 = unlimited, e.g. 3000000 for standard)")

    # SF labeling (standard mode only)
    parser.add_argument("--label-depth", type=int, default=12,
                        help="Stockfish labeling depth for standard mode (default: 12)")
    parser.add_argument("--sf-path", type=str, default=None, help="Path to Stockfish binary")
    parser.add_argument("--threads", type=int, default=4,
                        help="Parallel SF instances for labeling (default: 4)")

    # Filtering
    parser.add_argument("--min-elo", type=int, default=0, help="Minimum ELO filter (0 = no filter)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.list_months:
        months = list_available_months()
        print(f"Available broadcast months ({len(months)}):")
        for m in months:
            print(f"  {m}")
        return

    # Clear output file
    open(args.output, "w").close()
    total = 0

    # ── Broadcast: extract [%eval] annotations (no SF needed) ──
    if args.download_all:
        months = list_available_months()
        print(f"Downloading all {len(months)} broadcast months (extract [%eval])...")
        for month in months:
            path = download_broadcast(month)
            total += process_broadcast_pgn(path, args.output, args.min_elo, args.positions)
            if args.positions > 0 and total >= args.positions:
                break

    elif args.download:
        path = download_broadcast(args.download)
        total += process_broadcast_pgn(path, args.output, args.min_elo, args.positions)

    # ── Standard: label with SF depth 12 ──
    elif args.download_standard:
        sf_path = _resolve_sf(args)
        path = download_standard(args.download_standard)
        total += process_standard_pgn(path, args.output, args.min_elo,
                                       sf_path, args.label_depth, args.threads,
                                       args.positions)

    # ── Local file: user picks mode ──
    elif args.input:
        mode = args.mode
        if mode is None:
            # Auto-detect: if filename contains 'broadcast', use broadcast mode
            mode = "broadcast" if "broadcast" in args.input.lower() else "standard"
            print(f"  Auto-detected mode: {mode}")

        if mode == "broadcast":
            total += process_broadcast_pgn(args.input, args.output, args.min_elo,
                                            args.positions)
        else:
            sf_path = _resolve_sf(args)
            total += process_standard_pgn(args.input, args.output, args.min_elo,
                                           sf_path, args.label_depth, args.threads,
                                           args.positions)
    else:
        parser.print_help()
        return

    print(f"\n{'=' * 60}")
    print(f"  Total: {total:,} training positions → {args.output}")
    print(f"{'=' * 60}")
    print(f"\nNext steps:")
    print(f"  1. Shuffle:  shuf {args.output} -o {args.output}")
    print(f"  2. Convert:  bullet-utils convert --from text --input {args.output} --output data/train.bullet")
    print(f"  3. Train:    cargo run --release -- --data data/train.bullet --epochs 400")


def _resolve_sf(args):
    """Resolve and verify Stockfish binary for standard mode."""
    sf_path = get_sf_path(args.sf_path)
    try:
        engine = chess.engine.SimpleEngine.popen_uci(sf_path)
        sf_name = engine.id.get("name", "Stockfish")
        print(f"  ✓ {sf_name} ready (labeling at depth {args.label_depth})")
        engine.quit()
    except Exception as e:
        print(f"  ✗ Cannot start Stockfish: {e}")
        print(f"    Set STOCKFISH_PATH or use --sf-path")
        sys.exit(1)
    return sf_path


if __name__ == "__main__":
    main()
