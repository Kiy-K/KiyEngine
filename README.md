# KiyEngine v6.1.0

A high-performance UCI chess engine written in Rust, featuring a **hybrid evaluation system** combining a BitNet 1.58-bit Transformer for root move ordering with a **NNUE deep network** (SCReLU + CReLU) for fast per-node evaluation, all powered by hand-tuned AVX2 SIMD. Ships as a **single self-contained binary** with NNUE weights and opening books embedded at compile time.

## Highlights

| Metric | Value |
|--------|-------|
| **NPS** | ~900K single-thread, ~1.6M (4 threads) |
| **Search Depth** | 26-64 in 10s+0.1s self-play |
| **NNUE Eval** | (768→512)×2 → SCReLU → L1(16) → CReLU → L2(8) per bucket |
| **BitNet Model** | 22.76M params, GGUF format (79 MB, mmap) |
| **Quantization** | NNUE i16/i8, BitNet 1.58-bit ternary |
| **Distribution** | Single binary (NNUE + books embedded) |

## Key Features

### NNUE Evaluation (Primary)
- **Deep Network** -- (768→512)×2 → SCReLU → L1(1024→16, i8) → CReLU → L2(16→8, i16) → bucket select
- **AVX2 SIMD** -- all-i16 SCReLU div-255 trick, 8-wide L1 matmul with `madd_epi16`, SIMD CReLU and L2 dot product
- **Fused accumulator updates** -- single-pass copy+sub+add for quiet moves, copy+sub2+add for captures
- **Input buckets** (10) and **output buckets** (8) for piece-count-aware evaluation
- **Incrementally updated** -- per-perspective HalfKP accumulators with lazy refresh fallback
- **Embedded weights** -- compiled into binary via `include_bytes!`, no external files needed

### BitNet Transformer (Root Policy)
- **BitNet 1.58-bit** -- 12 layers, 512 d_model, 8 heads (GQA, 2 KV heads), 1536 hidden dim
- **Bit-Packed Ternary Weights** -- dual bitmask format, 4x denser than i8
- **SIMD Packed GEMV** -- AVX-512, AVX2, NEON, scalar (auto-detected)
- **KV Cache** -- 4-5x speedup for incremental inference
- **GGUF format** -- memory-mapped loading (optional, loaded from disk if present)

### Search
- **Alpha-Beta with PVS**, aspiration windows (exponential widening)
- **Lazy SMP** -- configurable threads, varied depth offsets for exploration diversity
- **Lock-Free TT** -- Hyatt/Mann XOR trick, depth-preferred replacement
- **Late Move Reductions** -- pre-computed table, PV and non-PV, history + SEE adjustments
- **Razoring**, **Reverse Futility Pruning**, **Null Move Pruning**
- **Late Move Pruning**, **Futility Pruning**
- **Static Exchange Evaluation (SEE)** -- full exchange with x-ray sliding attacks
- **Triangular PV Table** -- full multi-move PV in UCI output
- **Extensions** -- singular, recapture, passed pawn push (6th/7th rank)
- **Internal Iterative Reduction**, **Probcut**
- **Repetition Detection** -- search path (2-fold) + game history
- **Quiescence Search** with delta pruning, MVV-LVA, SEE pruning, check extensions
- **Syzygy Tablebases** -- WDL + DTZ probing (auto-extracted from `.tar.zst`)

### Move Ordering
- **Incremental move picking** -- selects best remaining move on demand
- TT Move > NN Policy (root) > Captures (MVV-LVA + Capture History) > Promotions > Killers > Countermove > History + Continuation History

### History Tables
- **History Heuristic** -- Stockfish-style gravity: `bonus - |h| * bonus / 16384`
- **Continuation History** -- `[prev_piece][prev_to][cur_piece][cur_to]`
- **Capture History** -- `[piece][to_sq][captured]`
- **History Aging** -- halves all tables between iterations

### Time Management
- **Stockfish-inspired soft/hard bounds** with ply-aware scaling
- **Best Move Stability**, **Falling Eval**, **Best Move Instability** factors
- **movetime** support for fixed-time searches

### Opening Book
- **Polyglot format** -- dual-book (embedded in binary, zero external files)
- Weighted random selection, limited to first 30 plies

## Installation and Build

Requires Rust 1.70+ ([rustup](https://rustup.rs/)).

```bash
git clone https://github.com/Kiy-K/KiyEngine.git
cd KiyEngine
cargo build --release
```

The release binary is at `./target/release/kiy_engine`. **No external files needed** -- NNUE weights and opening books are embedded in the binary.

Optionally place `kiyengine.gguf` alongside the binary for BitNet Transformer root policy (enhanced move ordering).

## UCI Options

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `Hash` | spin | 512 | 64-65536 | Transposition table size (MB) |
| `Threads` | spin | 4 | 1-256 | Number of search threads |
| `Move Overhead` | spin | 60 | 0-5000 | Time buffer for latency (ms) |
| `Analysis Mode` | check | false | -- | Disable book in analysis |
| `MultiPV` | spin | 1 | 1-500 | Number of principal variations |
| `SyzygyPath` | string | syzygy | -- | Path to Syzygy tablebases |

### Example Session

```text
uci
setoption name Hash value 1024
setoption name Threads value 4
isready
position startpos moves e2e4 e7e5
go wtime 60000 btime 60000 winc 1000 binc 1000
```

## Project Structure

```text
src/
+-- main.rs              Entry point (UCI loop)
+-- lib.rs               Library exports
+-- constants.rs         Centralized constants
+-- book/                Polyglot opening book (embedded)
|   +-- mod.rs           Book probing, Zobrist hashing, embedded fallback
|   +-- polyglot_keys.rs 781-key Random64 array
+-- engine/              BitNet Transformer inference
|   +-- mod.rs           Engine struct, forward / forward_with_cache
|   +-- bitlinear.rs     BitLinear: packed ternary weights, SIMD GEMV
|   +-- transformer.rs   MultiheadAttention, BitSwiGLU, KVCache
|   +-- heads.rs         PolicyHead, ValueHead (pre-transposed)
|   +-- gguf.rs          GGUF format parser and mmap loader
+-- eval/                Evaluation tables
|   +-- pst.rs           PeSTO piece-square tables
+-- search/              Search algorithms
|   +-- mod.rs           Alpha-beta, TT, move ordering, quiescence
|   +-- eval.rs          Static evaluation (NNUE or Material+PST fallback)
|   +-- time.rs          TimeManager (soft/hard bounds, dynamic factors)
|   +-- lazy_smp.rs      Lazy SMP thread management
|   +-- syzygy.rs        Syzygy tablebase probing
+-- nnue/                NNUE evaluation (active, embedded)
|   +-- mod.rs           Network struct, quantized inference, AVX2 SIMD
|   +-- accumulator.rs   Per-perspective accumulators, fused SIMD updates
|   +-- features.rs      HalfKP feature extraction, delta computation
+-- simd/                SIMD utilities
+-- uci/                 UCI protocol handler
```

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) -- Detailed architecture with visual diagrams
- [DOCUMENTATION.md](DOCUMENTATION.md) -- Module reference and development guidelines
- [CHANGELOG.md](CHANGELOG.md) -- Version history and release notes

## Troubleshooting

- **Missing GGUF model**: Optional. Engine works standalone with embedded NNUE. Place `kiyengine.gguf` alongside the binary for enhanced root policy.
- **Out of memory**: Reduce `Hash` (e.g., `setoption name Hash value 128`).
- **Slow NPS**: Ensure `--release` build. Check `Threads` is appropriate for your CPU.

## License

KiyEngine is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Khoi
