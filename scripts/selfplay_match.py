#!/usr/bin/env python3
"""Quick self-play match: KiyEngine vs itself with time control."""

import subprocess
import sys
import time
import re

ENGINE = "./target/release/kiy_engine"
NUM_GAMES = 6
TIME_MS = 5000   # 5 seconds per side
INC_MS = 100     # 0.1s increment

OPENINGS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # startpos
    "r1bqkbnr/pppppppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # 1.e4 e5 2.Nf3 Nc6
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",  # 1.e4 e6 (French)
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",  # 1.e4 c5 (Sicilian)
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",  # 1.d4
    "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",  # 1.d4 Nf6
]


def start_engine():
    p = subprocess.Popen(
        [ENGINE],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    return p


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
    fen = opening_fen
    stm = "w" if " w " in fen else "b"
    engines = {"w": w, "b": b}
    result = "*"
    ply = 0
    max_ply = 300
    total_nodes_w = 0
    total_nodes_b = 0
    total_time_w = 0.0
    total_time_b = 0.0

    while ply < max_ply:
        eng = engines[stm]
        if moves:
            pos_cmd = f"position fen {opening_fen} moves {' '.join(moves)}"
        else:
            pos_cmd = f"position fen {opening_fen}"
        send(eng, pos_cmd)
        send(eng, f"go wtime {TIME_MS} btime {TIME_MS} winc {INC_MS} binc {INC_MS}")

        t0 = time.time()
        lines = read_until(eng, "bestmove", timeout=15)
        elapsed = time.time() - t0

        bestmove = None
        nps = 0
        score = None
        depth = 0
        for line in lines:
            if line.startswith("bestmove"):
                bestmove = line.split()[1]
            m = re.search(r"info depth (\d+) .*score (cp [-\d]+|mate [-\d]+).*nps (\d+)", line)
            if m:
                depth = int(m.group(1))
                score = m.group(2)
                nps = int(m.group(3))

        if not bestmove or bestmove == "(none)":
            # Checkmate or stalemate
            if stm == "w":
                result = "0-1"
            else:
                result = "1-0"
            break

        if stm == "w":
            total_nodes_w += nps * elapsed if nps else 0
            total_time_w += elapsed
        else:
            total_nodes_b += nps * elapsed if nps else 0
            total_time_b += elapsed

        moves.append(bestmove)
        ply += 1

        # Check for mate score
        if score and "mate" in score:
            mate_in = int(score.split()[-1])
            if mate_in == 0:
                if stm == "w":
                    result = "0-1"
                else:
                    result = "1-0"
                break

        # Switch side
        stm = "b" if stm == "w" else "w"

    if result == "*":
        result = "1/2-1/2"  # Draw by move limit

    # Cleanup
    for e in [w, b]:
        send(e, "quit")
        e.wait(timeout=5)

    avg_nps_w = int(total_nodes_w / total_time_w) if total_time_w > 0 else 0
    avg_nps_b = int(total_nodes_b / total_time_b) if total_time_b > 0 else 0

    print(f"  Game {game_num}: {result} in {ply} plies "
          f"(W avg NPS: {avg_nps_w:,}, B avg NPS: {avg_nps_b:,}, "
          f"depth reached: {depth})")
    return result


def main():
    print(f"Self-play match: {NUM_GAMES} games, {TIME_MS/1000:.0f}s+{INC_MS/1000:.1f}s per side")
    print(f"Engine: {ENGINE}\n")

    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    for i in range(NUM_GAMES):
        fen = OPENINGS[i % len(OPENINGS)]
        res = play_game(fen, i + 1)
        results[res] = results.get(res, 0) + 1

    print(f"\nResults: +{results['1-0']} -{results['0-1']} ={results['1/2-1/2']}")
    print(f"White wins: {results['1-0']}, Black wins: {results['0-1']}, Draws: {results['1/2-1/2']}")


if __name__ == "__main__":
    main()
