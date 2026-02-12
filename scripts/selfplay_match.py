#!/usr/bin/env python3
"""Self-play match with proper clock tracking, draw detection, and diagnostics."""

import subprocess
import time
import re
import hashlib

ENGINE = "./target/release/kiy_engine"
NUM_GAMES = 6
TIME_MS = 10000  # 10 seconds per side
INC_MS = 100     # 0.1s increment
MAX_PLY = 400    # absolute cutoff

OPENINGS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",            # startpos
    "r1bqkbnr/pppppppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",    # Italian setup
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",        # French
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",       # Sicilian
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",          # 1.d4
    "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",        # 1.d4 Nf6
]


def start_engine():
    return subprocess.Popen(
        [ENGINE],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )


def send(proc, cmd):
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()


def read_until(proc, target, timeout=30):
    lines = []
    start = time.time()
    while time.time() - start < timeout:
        line = proc.stdout.readline().strip()
        if not line:
            continue
        lines.append(line)
        if target in line:
            return lines
    return lines


def position_hash(opening_fen, moves):
    """Simple hash of the position based on opening + move sequence.
    Two identical move sequences from the same opening = same position."""
    key = opening_fen + "|" + " ".join(moves)
    return hashlib.md5(key.encode()).hexdigest()


def play_game(opening_fen, game_num):
    w = start_engine()
    b = start_engine()

    for e in [w, b]:
        send(e, "uci")
        read_until(e, "uciok")
        send(e, "setoption name Threads value 1")
        send(e, "setoption name Hash value 64")
        send(e, "isready")
        read_until(e, "readyok")

    moves = []
    stm = "w" if " w " in opening_fen else "b"
    engines = {"w": w, "b": b}
    result = "*"
    reason = ""

    # ── Clock tracking ──
    clock = {"w": float(TIME_MS), "b": float(TIME_MS)}

    # ── Diagnostics ──
    max_depth = 0
    total_nodes = {"w": 0, "b": 0}
    total_time = {"w": 0.0, "b": 0.0}

    # ── Draw detection ──
    position_counts = {}  # hash → count for 3-fold repetition
    no_progress_ply = 0   # plies since last pawn move or capture (50-move rule approx)

    for ply in range(MAX_PLY):
        eng = engines[stm]

        # Send position
        if moves:
            pos_cmd = f"position fen {opening_fen} moves {' '.join(moves)}"
        else:
            pos_cmd = f"position fen {opening_fen}"
        send(eng, pos_cmd)

        # Send go with tracked clock
        wt = max(int(clock["w"]), 1)
        bt = max(int(clock["b"]), 1)
        send(eng, f"go wtime {wt} btime {bt} winc {INC_MS} binc {INC_MS}")

        t0 = time.time()
        lines = read_until(eng, "bestmove", timeout=max(wt / 1000 + 5, 10))
        elapsed_s = time.time() - t0
        elapsed_ms = elapsed_s * 1000.0

        # Update clock: subtract elapsed, add increment
        clock[stm] = clock[stm] - elapsed_ms + INC_MS

        # Time forfeit
        if clock[stm] < -500:  # allow small grace
            result = "0-1" if stm == "w" else "1-0"
            reason = f"{'White' if stm == 'w' else 'Black'} flagged"
            break

        # Parse engine output
        bestmove = None
        nps = 0
        score = None
        depth = 0
        for line in lines:
            if line.startswith("bestmove"):
                parts = line.split()
                bestmove = parts[1] if len(parts) > 1 else None
            m = re.search(r"info depth (\d+).*score (cp [-\d]+|mate [-\d]+)", line)
            if m:
                d = int(m.group(1))
                if d > depth:
                    depth = d
                    score = m.group(2)
            m2 = re.search(r"nps (\d+)", line)
            if m2:
                nps = int(m2.group(1))

        max_depth = max(max_depth, depth)
        if nps > 0:
            total_nodes[stm] += int(nps * elapsed_s)
            total_time[stm] += elapsed_s

        # No move = checkmate or stalemate
        if not bestmove or bestmove == "(none)":
            if score and "mate 0" in str(score):
                result = "0-1" if stm == "w" else "1-0"
                reason = "checkmate"
            else:
                # Could be stalemate or checkmate (bestmove none = side to move has no legal moves)
                # If the opponent delivered mate, it's a loss for stm
                result = "0-1" if stm == "w" else "1-0"
                reason = "no legal moves"
            break

        moves.append(bestmove)

        # ── Draw detection (approximate) ──
        # 50-move rule: heuristic — non-pawn non-capture moves
        # Pawn moves: source rank changes (files a-h, ranks 1-8), piece goes to rank 1/8 for promotion
        src = bestmove[:2]
        dst = bestmove[2:4]
        is_pawn_move = (src[1] == "2" and dst[1] == "4") or (src[1] == "7" and dst[1] == "5")  # double push
        is_pawn_move = is_pawn_move or (len(bestmove) > 4)  # promotion
        # Very rough: if source rank differs by 1 and file same, might be pawn
        if not is_pawn_move and src[0] == dst[0] and abs(int(dst[1]) - int(src[1])) == 1:
            is_pawn_move = True
        # Captures are harder to detect without board state — skip for now

        if is_pawn_move:
            no_progress_ply = 0
        else:
            no_progress_ply += 1

        if no_progress_ply >= 100:  # 50-move rule (100 half-moves)
            result = "1/2-1/2"
            reason = "50-move rule"
            break

        # 3-fold repetition (approximate: same move sequence hash)
        ph = position_hash(opening_fen, moves)
        position_counts[ph] = position_counts.get(ph, 0) + 1
        if position_counts[ph] >= 3:
            result = "1/2-1/2"
            reason = "3-fold repetition"
            break

        # Mate found — the move will be played, check if mate score is 1
        if score and "mate" in score:
            mate_val = int(score.split()[-1])
            if mate_val == 1:
                # Side to move just played a move that gives mate in 1
                result = "1-0" if stm == "w" else "0-1"
                reason = "checkmate"
                break
            elif mate_val == -1:
                # Opponent has mate in 1 — this side is about to be mated
                pass  # Let the game continue, opponent will deliver mate

        # Switch side
        stm = "b" if stm == "w" else "w"

    if result == "*":
        result = "1/2-1/2"
        reason = f"max ply ({MAX_PLY})"

    # Cleanup
    for e in [w, b]:
        try:
            send(e, "quit")
            e.wait(timeout=3)
        except Exception:
            e.kill()

    avg_nps_w = int(total_nodes["w"] / total_time["w"]) if total_time["w"] > 0 else 0
    avg_nps_b = int(total_nodes["b"] / total_time["b"]) if total_time["b"] > 0 else 0

    plies = len(moves)
    print(f"  Game {game_num}: {result} ({reason}) in {plies} plies, "
          f"max depth {max_depth}, "
          f"NPS W:{avg_nps_w:,} B:{avg_nps_b:,}, "
          f"clock W:{clock['w']:.0f}ms B:{clock['b']:.0f}ms")
    return result


def main():
    print(f"Self-play: {NUM_GAMES} games, {TIME_MS/1000:.0f}s+{INC_MS/1000:.1f}s, single-thread")
    print(f"Engine: {ENGINE}\n")

    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    for i in range(NUM_GAMES):
        fen = OPENINGS[i % len(OPENINGS)]
        res = play_game(fen, i + 1)
        results[res] = results.get(res, 0) + 1

    w, l, d = results["1-0"], results["0-1"], results["1/2-1/2"]
    print(f"\nFinal: +{w} -{l} ={d} (W:{w} B:{l} D:{d})")


if __name__ == "__main__":
    main()
