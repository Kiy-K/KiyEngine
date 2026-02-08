# KiyEngine v5.0.0

A high-performance UCI chess engine written in Rust, featuring a **BitNet 1.58-bit Transformer** neural network for evaluation and a heavily optimized alpha-beta search with Lazy SMP.

## Highlights

| Metric | Value |
|--------|-------|
| **NPS** | ~2.2 million nodes/sec (4 threads) |
| **Search Depth** | 15–19 ply in 40/60 time control |
| **Model Format** | GGUF (28 MB, memory-mapped) |
| **Quantization** | BitNet 1.58-bit (ternary weights) |

## Key Features

### Neural Network
- **BitNet 1.58-bit Transformer** with 72-plane policy head (4608 output classes)
- **GGUF model format** — 4× smaller than safetensors, instant mmap loading
- Single NN forward pass shared across all search threads via `Arc`
- Automatic fallback to legacy `.safetensors` format

### Search
- **Alpha-Beta with PVS** (Principal Variation Search)
- **Lazy SMP** — 4 threads (configurable), lock-free shared transposition table
- **Lock-Free TT** — Hyatt/Mann XOR trick, interleaved cache-line layout, depth-preferred replacement
- **Aspiration Windows** — ±50 cp starting at depth 4, progressive widening
- **Late Move Reductions (LMR)** — pre-computed lookup table (no transcendental math in hot path)
- **Razoring**, **Reverse Futility Pruning**, **Null Move Pruning** (static approximation)
- **Late Move Pruning (LMP)**, **Futility Pruning**, **SEE Pruning**
- **Internal Iterative Reduction (IIR)** when no TT move is found
- **Quiescence Search** with delta pruning and MVV-LVA ordering
- **Search Stability** — periodic time checks every 2048 nodes, `MAX_PLY` guard in qsearch

### Move Ordering
TT Move → NN Policy (root) → Captures (MVV-LVA) → Promotions → Killer Moves → History Heuristic

### Opening Book
- **Polyglot format** — dual-book support (`book1.bin`, `book2.bin`)
- Official Polyglot Zobrist hashing (781-key Random64 array)
- Weighted random selection, limited to first 30 plies
- Fully loaded into memory at startup — zero I/O during search

## Installation & Build

Requires Rust 1.70+ ([rustup](https://rustup.rs/)).

```bash
git clone https://github.com/Kiy-K/KiyEngine.git
cd KiyEngine
cargo build --release
```

The release binary is at `./target/release/kiy_engine_v5_alpha`.

Place `kiyengine.gguf` in the same directory as the binary (or alongside `Cargo.toml` for development).

## UCI Options

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `Hash` | spin | 512 | 64–65536 | Transposition table size (MB) |
| `Threads` | spin | 4 | 1–256 | Number of search threads |
| `Move Overhead` | spin | 60 | 0–5000 | Time buffer for latency (ms) |
| `Analysis Mode` | check | false | — | Disable book in analysis |
| `MultiPV` | spin | 1 | 1–500 | Number of principal variations |

### Example

```
uci
setoption name Hash value 1024
setoption name Threads value 4
isready
position startpos
go wtime 60000 btime 60000 movestogo 40
```

## Project Structure

```
src/
├── main.rs              # Binary entry point
├── lib.rs               # Library exports
├── constants.rs         # Global constants
├── book/                # Polyglot opening book
│   ├── mod.rs           # Book probing & Zobrist hashing
│   └── polyglot_keys.rs # Official 781-key Random64 array
├── engine/              # Neural network inference (GGUF/safetensors)
├── eval/                # Piece-square tables
├── search/
│   ├── mod.rs           # Alpha-beta, TT, move ordering, quiescence
│   ├── eval.rs          # Material + PST static evaluation
│   ├── time.rs          # Time management
│   ├── lazy_smp.rs      # Lazy SMP threading
│   ├── mcts*.rs         # MCTS implementations (experimental)
│   └── tt.rs            # Atomic TT (alternative implementation)
├── simd/                # AVX2/SIMD optimizations
└── uci/                 # UCI protocol handler
```

## Running Matches

Use [cutechess-cli](https://github.com/cutechess/cutechess) for automated matches:

```bash
bash run_match.sh
```

Results are saved to `omega_match.pgn`.

## Troubleshooting

- **Missing model**: Place `kiyengine.gguf` next to the binary.
- **Out of memory**: Reduce `Hash` (e.g., `setoption name Hash value 128`).
- **Slow NPS**: Ensure you are using `--release` build. Check `Threads` is set appropriately for your CPU.
- **Book not loading**: Ensure `src/book/book1.bin` and `src/book/book2.bin` exist relative to the executable.

## License

KiyEngine is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

© 2026 Khoi
