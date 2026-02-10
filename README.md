# KiyEngine v6.0.0

A high-performance UCI chess engine written in Rust, combining a **BitNet 1.58-bit Transformer** neural network with a heavily optimized alpha-beta search, Lazy SMP parallelism, and Stockfish-inspired time management.

## Highlights

| Metric | Value |
|--------|-------|
| **NPS** | ~1.93 million nodes/sec (4 threads) |
| **Search Depth** | 15-19 ply in 60s+1s time control |
| **SEE Pruning** | Full static exchange evaluation with x-ray |
| **Model** | 22.76M params, GGUF format (28 MB, mmap) |
| **Quantization** | BitNet 1.58-bit (ternary, dual-bitmask packed) |
| **KV Cache** | 4-5x speedup for incremental inference |

### v6.0.0 vs v5.2.0 (depth 20, 4 threads)

```text
Version     Nodes         NPS          Wall Time
-------     ----------    ----------   ---------
v5.2.0      33,900,000    2,180,000    ~15.6s
v6.0.0      23,358,906    1,931,426    ~12.1s
```

## Key Features

### Neural Network
- **BitNet 1.58-bit Transformer** -- 12 layers, 512 d_model, 8 heads (GQA, 2 KV heads), 1536 hidden dim
- **Bit-Packed Ternary Weights** -- dual bitmask format (pos_bits + neg_bits), 4x denser than i8
- **SIMD Packed GEMV** -- AVX-512, AVX2, NEON, and scalar fallback (auto-detected at runtime)
- **KV Cache** -- per-layer Key/Value caching for incremental inference
- **Pre-transposed weights** -- attention out_proj, policy head, value head transposed at load time
- **Fused RMSNorm+MatMul** -- single-pass normalization and projection in BitLinear
- **GGUF model format** -- 4x smaller than safetensors, instant mmap loading
- Single NN forward pass shared across all search threads via `Arc`

### Search
- **Alpha-Beta with PVS** (Principal Variation Search)
- **Lazy SMP** -- configurable thread count, varied depth offsets for exploration diversity
- **Lock-Free TT** -- Hyatt/Mann XOR trick, interleaved cache-line layout, depth-preferred replacement
- **Aspiration Windows** -- exponential widening on fail high/low
- **Late Move Reductions** -- pre-computed table, applies to PV and non-PV nodes, history + SEE adjustments
- **Razoring**, **Reverse Futility Pruning**, **Null Move Pruning**
- **Late Move Pruning**, **Futility Pruning**
- **Static Exchange Evaluation (SEE)** -- full exchange analysis with x-ray sliding attacks
  - Depth-scaled pruning for captures and quiet moves
  - SEE-based LMR adjustment (+1 reduction for losing captures)
  - Quiescence SEE pruning (skip all captures with SEE < 0)
- **Triangular PV Table** -- full principal variation tracking with multi-move PV in UCI output
- **Extensions** -- singular, recapture (same-square), passed pawn push (6th/7th rank)
- **Internal Iterative Reduction** when no TT move is found
- **Repetition Detection** -- search path (2-fold) and game history checking
- **Quiescence Search** with delta pruning, MVV-LVA ordering, SEE pruning, and check extensions

### Evaluation
- **Material Counting** -- standard piece values
- **Piece-Square Tables** -- PeSTO midgame tables
- **Bishop Pair Bonus** -- +30cp for having both bishops
- **Rook on Open/Semi-Open File** -- +20cp / +10cp
- **Passed Pawn Bonus** -- rank-scaled (5-100cp), computed with zero-loop bitboard masks
- **Doubled Pawn Penalty** -- -10cp per doubled pawn
- **Isolated Pawn Penalty** -- -15cp per isolated pawn
- **NNUE** -- accumulator infrastructure in place (activation planned)

### Move Ordering
- **Incremental move picking** -- selects best remaining move on demand, avoids sorting pruned moves
- TT Move > NN Policy (root) > Captures (MVV-LVA + Capture History) > Promotions > Killers > Countermove > History + Continuation History

### History Tables
- **History Heuristic** -- Stockfish-style gravity: `bonus - |h| * bonus / 16384`
- **Continuation History** -- `[prev_piece][prev_to][cur_piece][cur_to]` table
- **Capture History** -- `[piece][to_sq][captured]` table
- **History Aging** -- halves all tables between iterative deepening iterations

### Time Management
- **Stockfish-inspired soft/hard bounds** -- dynamic allocation with ply-aware scaling
- **Best Move Stability** -- stable best move reduces allocated time (sigmoid function)
- **Falling Eval** -- dropping score increases allocated time (EMA-based)
- **Best Move Instability** -- frequent best-move changes extend time
- **movetime** support for fixed-time searches

### Opening Book
- **Polyglot format** -- dual-book support (`book1.bin`, `book2.bin`)
- Official Polyglot Zobrist hashing (781-key Random64 array)
- Weighted random selection, limited to first 30 plies
- Fully loaded into memory at startup -- zero I/O during search

## Installation and Build

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
| `Hash` | spin | 512 | 64-65536 | Transposition table size (MB) |
| `Threads` | spin | 4 | 1-256 | Number of search threads |
| `Move Overhead` | spin | 60 | 0-5000 | Time buffer for latency (ms) |
| `Analysis Mode` | check | false | -- | Disable book in analysis |
| `MultiPV` | spin | 1 | 1-500 | Number of principal variations |

### Example Session

```text
uci
setoption name Hash value 1024
setoption name Threads value 4
isready
position startpos moves e2e4 e7e5
go wtime 60000 btime 60000 winc 1000 binc 1000
```

### Fixed-Time Search

```text
position fen 3r2k1/pp3pp1/2p1bn1p/8/4P3/2N2N2/PPP2PPP/3R2K1 w - - 0 1
go movetime 5000
```

## Project Structure

```text
src/
+-- main.rs              Entry point (UCI loop)
+-- lib.rs               Library exports
+-- constants.rs         Centralized constants (engine, NN, search)
+-- book/                Polyglot opening book
|   +-- mod.rs           Book probing and Zobrist hashing
|   +-- polyglot_keys.rs 781-key Random64 array
+-- engine/              Neural network inference
|   +-- mod.rs           Engine struct, forward / forward_with_cache
|   +-- bitlinear.rs     BitLinear: packed ternary weights, SIMD GEMV
|   +-- transformer.rs   MultiheadAttention, BitSwiGLU, KVCache
|   +-- heads.rs         PolicyHead, ValueHead (pre-transposed)
|   +-- gguf.rs          GGUF format parser and mmap loader
+-- eval/                Evaluation tables
|   +-- pst.rs           PeSTO piece-square tables
+-- search/              Search algorithms
|   +-- mod.rs           Alpha-beta, TT, move ordering, quiescence
|   +-- eval.rs          Static evaluation (material + PST + positional)
|   +-- time.rs          TimeManager (soft/hard bounds, dynamic factors)
|   +-- lazy_smp.rs      Lazy SMP thread management
|   +-- mcts*.rs         MCTS implementations (experimental)
+-- nnue/                NNUE evaluation (planned)
+-- simd/                SIMD utilities
+-- uci/                 UCI protocol handler
```

For a detailed architectural walkthrough with diagrams, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Running Matches

Use [cutechess-cli](https://github.com/cutechess/cutechess) for automated matches:

```bash
bash run_match.sh
```

Results are saved to `omega_match.pgn`.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) -- Detailed architecture with visual diagrams
- [DOCUMENTATION.md](DOCUMENTATION.md) -- Module reference and development guidelines
- [CHANGELOG.md](CHANGELOG.md) -- Version history and release notes
- [V5_MIGRATION.md](V5_MIGRATION.md) -- GGUF migration guide from v4/v5

## Troubleshooting

- **Missing model**: Place `kiyengine.gguf` next to the binary.
- **Out of memory**: Reduce `Hash` (e.g., `setoption name Hash value 128`).
- **Slow NPS**: Ensure `--release` build. Check `Threads` is appropriate for your CPU.
- **Book not loading**: Ensure `book1.bin` and `book2.bin` exist relative to the executable.

## License

KiyEngine is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Khoi
