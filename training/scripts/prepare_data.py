#!/usr/bin/env python3
"""
Lichess PGN → NNUE Training Data Pipeline (Stockfish Depth-12 Labels)

Extracts positions from Lichess broadcast PGN files (GM/IM games) and/or
standard-rated game PGNs, then labels each position with Stockfish at
configurable depth (default: 12) for high-quality NNUE training data.

Output: Bullet text format `<FEN> | <score_cp> | <result>`
Convert: bullet-utils convert --from text --input train.txt --output train.bullet

Usage:
  # Broadcast games (auto-download from Lichess):
  python prepare_data.py --download 2025-01 --output data/train.txt --label-depth 12

  # All recent broadcast months:
  python prepare_data.py --download-all --output data/train.txt --label-depth 12 --threads 8

  # Local PGN file (broadcast or standard-rated):
  python prepare_data.py --input games.pgn.zst --output data/train.txt --label-depth 12

  # Download standard-rated games from Lichess (by month):
  python prepare_data.py --download-standard 2025-01 --output data/train.txt --label-depth 12

  # WDL-only (no SF labeling, fast but lower quality):
  python prepare_data.py --download-all --output data/train.txt --no-label

Requirements:
  pip install python-chess zstandard requests tqdm
  Stockfish binary (set STOCKFISH_PATH or use --sf-path)
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


def extract_fens(game, min_elo: int = 0) -> list:
    """Extract candidate FENs from a single PGN game.

    Returns list of (fen, result) tuples — score will be added by SF labeling.
    """
    # Check result
    result_str = game.headers.get("Result", "*")
    result = result_to_float(result_str)
    if result < 0:
        return []  # Skip unfinished games

    # Check ELO filter
    if min_elo > 0:
        try:
            white_elo = int(game.headers.get("WhiteElo", "0"))
            black_elo = int(game.headers.get("BlackElo", "0"))
            if white_elo < min_elo or black_elo < min_elo:
                return []
        except ValueError:
            pass  # No ELO info, include anyway (broadcast games are generally high quality)

    # Walk through moves
    fens = []
    board = game.board()
    ply = 0

    for move in game.mainline_moves():
        board.push(move)
        ply += 1

        # Skip early opening and late endgame
        if ply < MIN_PLY:
            continue
        if len(board.piece_map()) < MIN_PIECES:
            continue

        # Skip positions in check (noisy for training)
        if board.is_check():
            continue

        # Subsample to reduce position correlation
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


def process_pgn_file(pgn_path: str, output_path: str, min_elo: int = 0,
                     sf_path: str = None, label_depth: int = 0, threads: int = 1) -> int:
    """Process a PGN file and write training positions to text format.

    If label_depth > 0 and sf_path is set, labels each position with Stockfish.
    Otherwise, writes score_cp = 0 (WDL-only).
    """
    print(f"Processing: {pgn_path}")
    use_sf = label_depth > 0 and sf_path is not None
    if use_sf:
        print(f"  SF labeling: depth {label_depth}, {threads} thread(s)")

    text_stream, raw_fh = open_pgn(pgn_path)
    total_positions = 0
    total_games = 0
    skipped_games = 0

    # Buffer FENs for batch SF labeling (process in chunks of 500 games)
    GAME_BATCH = 500
    fen_buffer = []  # list of (fen, result)

    def flush_buffer():
        nonlocal total_positions, fen_buffer
        if not fen_buffer:
            return
        if use_sf:
            labeled = label_positions_sf(fen_buffer, sf_path, label_depth, threads)
        else:
            labeled = [(fen, 0, result) for fen, result in fen_buffer]
        with open(output_path, "a", encoding="utf-8") as out:
            for fen, score, result in labeled:
                out.write(f"{fen} | {score} | {result:.1f}\n")
                total_positions += 1
        fen_buffer = []

    while True:
        try:
            game = chess.pgn.read_game(text_stream)
        except Exception:
            continue

        if game is None:
            break

        total_games += 1
        fens = extract_fens(game, min_elo)

        if not fens:
            skipped_games += 1
            continue

        fen_buffer.extend(fens)

        if total_games % GAME_BATCH == 0:
            flush_buffer()
            print(f"  Games: {total_games}, Positions: {total_positions}, Skipped: {skipped_games}")

    # Final flush
    flush_buffer()

    raw_fh.close()
    print(f"  Done: {total_games} games → {total_positions} positions")
    return total_positions


def main():
    parser = argparse.ArgumentParser(
        description="Lichess PGN → NNUE Training Data (SF Depth-12 Labels)"
    )
    # Input sources
    parser.add_argument("--input", type=str, help="Path to PGN file (.pgn or .pgn.zst)")
    parser.add_argument("--download", type=str, help="Download broadcast month (e.g., 2025-01)")
    parser.add_argument("--download-all", action="store_true", help="Download all broadcast months")
    parser.add_argument("--download-standard", type=str,
                        help="Download standard-rated month (WARNING: large files, use --min-elo)")
    parser.add_argument("--list-months", action="store_true", help="List available broadcast months")

    # Output
    parser.add_argument("--output", type=str, default="data/train.txt", help="Output text file path")

    # SF labeling
    parser.add_argument("--label-depth", type=int, default=12,
                        help="Stockfish labeling depth (default: 12, 0 = WDL-only)")
    parser.add_argument("--no-label", action="store_true",
                        help="Disable SF labeling (WDL-only, score=0)")
    parser.add_argument("--sf-path", type=str, default=None, help="Path to Stockfish binary")
    parser.add_argument("--threads", type=int, default=4,
                        help="Parallel SF instances for labeling (default: 4)")

    # Filtering
    parser.add_argument("--min-elo", type=int, default=0, help="Minimum ELO filter (0 = no filter)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Resolve SF labeling
    label_depth = 0 if args.no_label else args.label_depth
    sf_path = None
    if label_depth > 0:
        sf_path = get_sf_path(args.sf_path)
        # Verify SF works
        try:
            engine = chess.engine.SimpleEngine.popen_uci(sf_path)
            sf_name = engine.id.get("name", "Stockfish")
            print(f"  ✓ {sf_name} ready (labeling at depth {label_depth})")
            engine.quit()
        except Exception as e:
            print(f"  ✗ Cannot start Stockfish: {e}")
            print(f"    Set STOCKFISH_PATH, use --sf-path, or --no-label")
            sys.exit(1)

    if args.list_months:
        months = list_available_months()
        print(f"Available broadcast months ({len(months)}):")
        for m in months:
            print(f"  {m}")
        return

    # Clear output file
    open(args.output, "w").close()

    common_kwargs = dict(
        min_elo=args.min_elo, sf_path=sf_path,
        label_depth=label_depth, threads=args.threads,
    )
    total = 0

    if args.download_all:
        months = list_available_months()
        print(f"Downloading all {len(months)} broadcast months...")
        for month in months:
            path = download_broadcast(month)
            total += process_pgn_file(path, args.output, **common_kwargs)

    elif args.download:
        path = download_broadcast(args.download)
        total += process_pgn_file(path, args.output, **common_kwargs)

    elif args.download_standard:
        path = download_standard(args.download_standard)
        total += process_pgn_file(path, args.output, **common_kwargs)

    elif args.input:
        total += process_pgn_file(args.input, args.output, **common_kwargs)

    else:
        parser.print_help()
        return

    print(f"\n{'=' * 60}")
    print(f"  Total: {total:,} training positions → {args.output}")
    if label_depth > 0:
        print(f"  Labels: Stockfish depth {label_depth}")
    else:
        print(f"  Labels: WDL-only (score=0)")
    print(f"{'=' * 60}")
    print(f"\nNext steps:")
    print(f"  1. Shuffle:  shuf {args.output} -o {args.output}")
    print(f"  2. Convert:  bullet-utils convert --from text --input {args.output} --output data/train.bullet")
    print(f"  3. Train:    cargo run --release -- --data data/train.bullet --epochs 400")


if __name__ == "__main__":
    main()
