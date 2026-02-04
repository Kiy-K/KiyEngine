# KiyEngine v4.4.3
[![Build and Release](https://github.com/Kiy-K/KiyEngine/actions/workflows/release.yml/badge.svg?branch=main)](https://github.com/Kiy-K/KiyEngine/actions/workflows/release.yml)

KiyEngine is a high-performance, open-source UCI chess engine written in Rust. It features a BitNet 1.58-bit Transformer architecture for evaluation and a highly optimized tactical search engine with Lazy SMP.

---

## Release v4.4.3
This patch release addresses a correctness issue and improves robustness in the search subsystem, as well as cleaning up and testing the transposition table implementation.

Notable changes in v4.4.3:
- Fix duplicated logic in Quiescence Search (QS) that could cause incorrect recursion and unstable behavior in tactical lines.
- Guard Transposition Table sizing to ensure at least one bucket and avoid mask underflow for small Hash settings.
- Add unit tests for `TranspositionTable::new` and `TranspositionTable::clear`.
- Remove unused imports and constants to reduce warnings and improve maintainability.

---

## Key Features
- BitNet-based evaluation with a transformer architecture for improved positional assessments.
- Efficient tactical search with Quiescence Search, Principal Variation Search, and aspiration windows.
- Lazy SMP support where helper threads populate the Transposition Table while a main thread searches full depth.
- Hybrid move ordering that combines TT moves, policy suggestions, MVV-LVA captures, killer/history heuristics, and simple opening bonuses.

---

## Installation & Build
Build requires the Rust toolchain (rustup).

```bash
git clone https://github.com/Kiy-K/KiyEngine.git
cd KiyEngine
cargo build --release
```

The release binary is available at `./target/release/kiy_engine_v4_omega`.

---

## UCI Usage
KiyEngine implements the UCI protocol and can be used with any UCI-compatible GUI.

Configuration options:
- `Hash` (spin) — default: 512 MB — Transposition Table size in megabytes.
- `Threads` (spin) — default: 1 — Number of search threads.
- `Move Overhead` (spin) — default: 30 ms — Time buffer to account for latency.

Example:
```
uci
setoption name Hash value 1024
isready
position startpos
go depth 25
```

---

## Troubleshooting
- If weights are missing, place `kiyengine_v4.safetensors` in the executable directory.
- Use the `release` build for best performance.
- Reduce `Hash` if you encounter out-of-memory issues.

---

## License
KiyEngine is licensed under the Apache License 2.0. See `LICENSE` for details.

© 2026 Khoi
