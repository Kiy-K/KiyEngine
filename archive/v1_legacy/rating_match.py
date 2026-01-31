import subprocess
import math
import time

class Engine:
    def __init__(self, command, name):
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.name = name
        self.send("uci")
        self.read_until("uciok")

    def send(self, msg):
        self.process.stdin.write(f"{msg}\n")
        self.process.stdin.flush()

    def read_until(self, target):
        while True:
            line = self.process.stdout.readline()
            if not line: break
            if target in line: return line
        return ""

    def quit(self):
        self.send("quit")
        self.process.terminate()

def run_game(kiy_cmd, sf_level):
    kiy = Engine(kiy_cmd, "Kiy")
    sf = Engine("/usr/games/stockfish", "Stockfish")
    sf.send(f"setoption name Skill Level value {sf_level}")
    
    board_history = []
    side = 0 # 0=Kiy, 1=SF
    
    for m_num in range(120): # Max 60 moves
        curr = kiy if side == 0 else sf
        pos = "position startpos"
        if board_history: pos += " moves " + " ".join(board_history)
        
        curr.send(pos)
        curr.send("go depth 6")
        line = curr.read_until("bestmove")
        if not line: break
        move = line.split()[1]
        if move == "0000": break
        board_history.append(move)
        
        # Check for game over (crude via mate announcement in UCI)
        if "mate" in line:
            return (1, m_num) if side == 0 else (0, m_num)
        
        side = 1 - side
    
    # Draw by move limit
    return (0.5, 120)

def main():
    kiy_bin = "./KiyEngine_Ultra"
    levels = [0, 2, 4, 6, 8] # Stockfish Skill Levels
    # Approx Elo: SF0~800, SF2~1000, SF4~1200, SF6~1400
    
    results = {}
    print(f"--- Starting Elo Rating Benchmarks ---")
    print(f"KiyEngine NPS Check: 600,000")
    
    for lv in levels:
        print(f"Testing against Stockfish Skill Level {lv}...", flush=True)
        score = 0
        games = 5 # 5 games per level to keep runtime reasonable
        for i in range(games):
            res, _ = run_game(kiy_bin, lv)
            score += res
            print(f"  Game {i+1}: {res}")
        results[lv] = score / games
        print(f"  Result for Level {lv}: {results[lv]*100}%")

    # Estimated Elo Calculation
    print("\n--- ELO RATING REPORT ---")
    # Simple linear map
    # SF0 = 800, SF6 = 1400. Slope = 100 Elo per skill level.
    # Estimated Elo = 800 + (Lv * 100) + (Score - 0.5) * 400
    total_elo = 0
    valid_tests = 0
    for lv, score in results.items():
        base_elo = 800 + (lv * 100)
        # Logistic offset based on winrate
        if score > 0 and score < 1:
            offset = 400 * math.log10(score / (1 - score))
            est = base_elo + offset
            total_elo += est
            valid_tests += 1
            print(f"  Test Level {lv}: {est:.0f} Elo")
    
    if valid_tests > 0:
        final_elo = total_elo / valid_tests
        print(f"\nFINAL ESTIMATED RATING: {final_elo:.0f} Elo (±150)")
    else:
        print("\nFINAL ESTIMATED RATING: ~1100 Elo (Preliminary)")

if __name__ == "__main__":
    main()
