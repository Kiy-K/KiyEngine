#!/usr/bin/env python3
"""
Lichess Broadcast PGN → Bullet Training Data Pipeline

Downloads Lichess broadcast PGN files (high-quality GM/IM games) and converts
them to Bullet's text format: `<FEN> | <score_cp> | <result>`

The text output can then be converted to bulletformat using bullet-utils:
  cargo run --release --package bullet-utils -- text-to-bullet input.txt output.bullet

Usage:
  python prepare_data.py --input broadcast.pgn.zst --output data/train.txt [--min-elo 2400]
  python prepare_data.py --download 2025-01 --output data/train.txt
  python prepare_data.py --download-all --output data/train.txt

Requirements:
  pip install python-chess zstandard requests tqdm
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


BROADCAST_BASE = "https://database.lichess.org/broadcast"
BROADCAST_LIST = f"{BROADCAST_BASE}/list.txt"

# Positions to skip (opening, endgame with few pieces)
MIN_PLY = 16          # Skip first 8 full moves (opening book territory)
MIN_PIECES = 6        # Skip positions with fewer than 6 pieces
MAX_POSITIONS_PER_GAME = 80  # Cap positions per game
SAMPLE_RATE = 0.6     # Sample 60% of eligible positions (reduce correlation)


def download_broadcast(month: str, output_dir: str = "downloads") -> str:
    """Download a Lichess broadcast PGN for a given month (e.g., '2025-01')."""
    if requests is None:
        print("ERROR: requests required for download. Install: pip install requests")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    filename = f"lichess_db_broadcast_{month}.pgn.zst"
    url = f"{BROADCAST_BASE}/{filename}"
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


def extract_positions(game, min_elo: int = 0) -> list:
    """Extract training positions from a single PGN game.

    Returns list of (fen, score_cp, result) tuples.
    Score is set to 0 for pretraining (WDL-only), or can be filled by engine later.
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
    positions = []
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

        # FEN (we only need piece placement, side to move, castling, en passant)
        fen = board.fen()

        # Score: 0 for WDL-only pretraining
        # In a full pipeline, you'd re-score with Stockfish here
        score_cp = 0

        # Result is white-relative
        positions.append((fen, score_cp, result))

        if len(positions) >= MAX_POSITIONS_PER_GAME:
            break

    return positions


def process_pgn_file(pgn_path: str, output_path: str, min_elo: int = 0) -> int:
    """Process a PGN file and write training positions to text format."""
    print(f"Processing: {pgn_path}")

    text_stream, raw_fh = open_pgn(pgn_path)
    total_positions = 0
    total_games = 0
    skipped_games = 0

    with open(output_path, "a", encoding="utf-8") as out:
        while True:
            try:
                game = chess.pgn.read_game(text_stream)
            except Exception:
                continue

            if game is None:
                break

            total_games += 1
            positions = extract_positions(game, min_elo)

            if not positions:
                skipped_games += 1
                continue

            for fen, score, result in positions:
                # Bullet text format: FEN | score_cp | result
                out.write(f"{fen} | {score} | {result:.1f}\n")
                total_positions += 1

            if total_games % 1000 == 0:
                print(f"  Games: {total_games}, Positions: {total_positions}, Skipped: {skipped_games}")

    raw_fh.close()
    print(f"  Done: {total_games} games → {total_positions} positions")
    return total_positions


def main():
    parser = argparse.ArgumentParser(description="Lichess Broadcast → Bullet Training Data")
    parser.add_argument("--input", type=str, help="Path to PGN file (.pgn or .pgn.zst)")
    parser.add_argument("--download", type=str, help="Download specific month (e.g., 2025-01)")
    parser.add_argument("--download-all", action="store_true", help="Download all available months")
    parser.add_argument("--list-months", action="store_true", help="List available broadcast months")
    parser.add_argument("--output", type=str, default="data/train.txt", help="Output text file path")
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

    if args.download_all:
        months = list_available_months()
        print(f"Downloading all {len(months)} broadcast months...")
        for month in months:
            path = download_broadcast(month)
            total += process_pgn_file(path, args.output, args.min_elo)

    elif args.download:
        path = download_broadcast(args.download)
        total += process_pgn_file(path, args.output, args.min_elo)

    elif args.input:
        total += process_pgn_file(args.input, args.output, args.min_elo)

    else:
        parser.print_help()
        return

    print(f"\n=== Total: {total} training positions written to {args.output} ===")
    print(f"\nNext step: convert to bulletformat:")
    print(f"  cd ../bullet && cargo run --release --package bullet-utils -- text-to-bullet {args.output} data/train.bullet")


if __name__ == "__main__":
    main()
