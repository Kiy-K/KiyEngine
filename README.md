# ♟️ KiyEngine V4.3.1 Omega
[![Build and Release](https://github.com/Kiy-K/KiyEngine/actions/workflows/release.yml/badge.svg?branch=main)](https://github.com/Kiy-K/KiyEngine/actions/workflows/release.yml)

**KiyEngine V4.3.1 Omega** (codenamed *Tactical Monster*) is a high-performance, open-source UCI chess engine written in Rust. It features a state-of-the-art **BitNet Transformer** architecture for evaluation and a highly optimized tactical search engine.

---

## 🌟 Key Features

### 1. 🧠 BitNet 1.58-bit Architecture
The engine utilizes a pure Transformer model with **BitNet Linear** layers.
- **Strictly Ternary Weights**: Weights are constrained to `{-1, 0, 1}` following the 1.58-bit discrete format.
- **Discrete Math Optimization**: The inference core is optimized for addition and subtraction, maximizing efficiency on modern CPUs.

### 2. 🔍 Tactical Search Overhaul
- **Deep Quiescence Search (QS)**: Explores all captures *and* checks until a stable position is reached, eliminating the horizon effect in critical tactical lines.
- **Principal Variation Search (PVS)**: Refined PVS with **Aspiration Windows** to focus search power on the most promising moves.
- **Iterative Deepening**: Dynamically explores depths 1 to 25+, ensuring the best move is found within time limits.

### 3. 🎯 Neural-Heuristic Move Ordering
A sophisticated hybrid ordering system:
1. **TT Move**: Best move from the Transposition Table.
2. **Top-3 AI Moves**: Highest probability moves from the Omega brain's policy head.
3. **MVV-LVA**: Most Valuable Victim - Least Valuable Attacker for captures.
4. **Killer & History Heuristics**: Pruning non-tactical paths based on search history.

### 4. ⚖️ Advanced Evaluation
- **Centipawn Mapping**: Neural value head output is scaled to Centipawns (x1000).
- **King Safety Heuristic**: Detects and penalizes vulnerabilities around the king's zone to ensure defensive stability.

---

## 🛠️ Installation & Build

Build requires the [Rust toolchain](https://rustup.rs/).

```bash
# Clone the repository
git clone https://github.com/Kiy-K/KiyEngine.git
cd KiyEngine

# Build the optimized release binary
cargo build --release
```

The binary will be available at `./target/release/kiy_engine_v4_omega`.

---

## 🎮 UCI Usage Guide

KiyEngine is a standard UCI engine and can be used with any chess GUI (e.g., Arena, Fritz, or Banksia).

### Configuration Options
| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `Hash` | spin | 512 | Transposition Table size in Megabytes (MB). |

### Example UCI Commands
```uci
uci
setoption name Hash value 1024
isready
position startpos
go depth 25
```

---

## 🔧 Troubleshooting

- **Weights Missing**: Ensure `kiyengine_v4.safetensors` is in the same directory as the executable.
- **Slow Search**: Always ensure you are running the `release` build. Debug builds are significantly slower.
- **Memory Usage**: If you encounter crashes on low-memory systems, reduce the `Hash` value using the UCI option.

---

## 📄 License

KiyEngine is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

Copyright © 2026 Khoi.
