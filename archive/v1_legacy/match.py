import subprocess
import time
import sys
import re

class Engine:
    def __init__(self, command, name):
        self.name = name
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
            shell=True,
            env={"RUST_BACKTRACE": "1", "RUST_MIN_STACK": "33554432"} # 32MB Stack
        )
    
    def send(self, cmd):
        # print(f"[{self.name}] <- {cmd}")
        self.process.stdin.write(cmd + "\n")
        self.process.stdin.flush()

    def read_until(self, token):
        while True:
            line = self.process.stdout.readline()
            if not line:
                break
            line = line.strip()
            # print(f"[{self.name}] -> {line}")
            if line.startswith(token):
                return line

def play_match():
    # stockfish path
    stockfish_cmd = "/usr/games/stockfish"
    # kiy_engine GM version
    kiy_cmd = "/home/khoi/Workspace/KiyEngine/KiyEngine_GM"

    sf = Engine(stockfish_cmd, "Stockfish")
    kiy = Engine(kiy_cmd, "KiyEngine")

    # Initialize
    for e in [sf, kiy]:
        e.send("uci")
        e.read_until("uciok")
        e.send("ucinewgame") # Reset TT and state
        e.send("isready")
        e.read_until("readyok")
    
    sf.send("setoption name Skill Level value 15")
    kiy.send("setoption name Skill Level value 10")

    print("\n=== Match Start (Rapid): KiyEngine GM vs Stockfish L15 ===")
    
    moves = []
    
    # Game Loop (Goal: 100 rounds)
    for i in range(100):
        move_num = i + 1
        
        # --- White: KiyEngine ---
        pos = "position startpos"
        if moves: pos += " moves " + " ".join(moves)
        kiy.send(pos)
        kiy.send("go depth 15") # Explicitly request depth 15
        
        line = kiy.read_until("bestmove")
        if not line: 
            print("KiyEngine stopped responding.")
            err = kiy.process.stderr.read()
            if err: print(f"ERROR LOG:\n{err}")
            break
        
        best_w = line.split()[1]
        if best_w in ["(none)", "0000"]:
             print(f"\nGame Over. White has no moves.")
             break
        moves.append(best_w)

        # --- Black: Stockfish ---
        pos = "position startpos moves " + " ".join(moves)
        sf.send(pos)
        sf.send("go movetime 1000") 
        
        line = sf.read_until("bestmove")
        if not line:
            print("Stockfish stopped responding.")
            err = sf.process.stderr.read()
            if err: print(f"ERROR LOG:\n{err}")
            break
        
        best_b = line.split()[1]
        if best_b in ["(none)", "0000"]:
             print(f"\nGame Over. Black has no moves.")
             break
        moves.append(best_b)

        print(f"{move_num}. {best_w} {best_b} ", end="", flush=True)
        if move_num % 10 == 0: print("")

    print("\n\nMATCH CONCLUDED")
    print("PGN:", " ".join([f"{i//2+1}. {moves[i]} {moves[i+1] if i+1 < len(moves) else ''}" for i in range(0, len(moves), 2)]))

    # Safe Shutdown
    for e in [sf, kiy]:
        if e.process.poll() is None: # Process still alive
            try:
                e.send("quit")
                e.process.wait(timeout=2)
            except:
                e.process.kill()

    sf.send("quit")
    kiy.send("quit")

if __name__ == "__main__":
    play_match()
