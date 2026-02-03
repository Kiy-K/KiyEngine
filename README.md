# ♟️ KiyEngine V4.4.2 Titan
[![Build and Release](https://github.com/Kiy-K/KiyEngine/actions/workflows/release.yml/badge.svg?branch=main)](https://github.com/Kiy-K/KiyEngine/actions/workflows/release.yml)

**KiyEngine V4.4.2 Titan** is a high-performance, open-source UCI chess engine written in Rust. It features a **BitNet 1.58-bit Transformer** architecture for evaluation and a highly optimized tactical search engine with AVX-512 acceleration and Lazy SMP.

---

## 🌟 Key Features (V4.4.2)

### 1. 🧠 BitNet 1.58-bit Architecture
The engine utilizes a Transformer model with **BitNet Linear** layers.
- **Strictly Ternary Weights**: Weights are quantized to strict `{-1.0, 0.0, 1.0}` following the 1.58-bit discrete format.
- **SIMD-Optimized Inference**: AVX-512/AVX2 kernels perform ternary GEMM using pure addition/subtraction masks, maximizing CPU efficiency.

### 2. 🔍 Tactical Search with Lazy SMP
- **Deep Quiescence Search (QS)**: Explores all captures *and* checks until a stable position is reached, reducing the horizon effect in critical tactical lines.
- **Principal Variation Search (PVS)**: PVS with **aspiration windows** to focus search power on the most promising moves.
- **Lazy SMP**: Multiple threads share a cache-friendly bucketed Transposition Table, with the main thread searching full depth and helpers feeding TT.

### 3. 🎯 Neural-Heuristic Move Ordering (Anti-a2a3)
A sophisticated hybrid ordering system:
1. **TT Move**: Best move from the Transposition Table.
2. **Policy-Guided Moves**: Highest probability moves from the BitNet policy head (scaled down in the opening to avoid overconfidence in side moves).
3. **MVV-LVA**: Most Valuable Victim - Least Valuable Attacker for captures.
4. **Killer & History Heuristics**: Pruning non-tactical paths based on search history.
5. **Opening Centralization Bonus**: Early moves towards `d4/e4/d5/e5` are slightly favored in ordering to encourage central play.

### 4. ⚖️ Advanced Evaluation
- **Classical + Neural Hybrid**: A classical bitboard evaluation (material, PST, pawn structure) is combined with BitNet-guided search.
- **Enhanced King Safety**: Detects attackers in the king zone, rewards pawn shields, and penalizes open files near the king.
- **Pawn Structure**: Penalizes doubled/isolated pawns, rewards passed pawns more as they advance.

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
| `Threads` | spin | 1 | Number of search threads (Lazy SMP). |
| `Move Overhead` | spin | 30 | Time buffer in milliseconds to avoid flagging in network/GUI latency. |

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
