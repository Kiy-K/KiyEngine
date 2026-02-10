# KiyEngine v6.0.0 Documentation

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Module Reference](#module-reference)
- [Search Details](#search-details)
- [Evaluation Details](#evaluation-details)
- [Time Management Details](#time-management-details)
- [Building and Running](#building-and-running)
- [Configuration](#configuration)
- [Development Guidelines](#development-guidelines)

---

## Architecture Overview

KiyEngine is a high-performance UCI chess engine built in Rust with the following key subsystems:

### Core Systems

1. **Neural Network Evaluation** (`engine/`)
   - BitNet 1.58-bit Transformer (12 layers, 512 d_model, 8 heads, GQA with 2 KV heads, 1536 hidden dim)
   - Bit-packed ternary weights with dual bitmask format (4x denser than i8)
   - SIMD-optimized packed GEMV (AVX-512, AVX2, NEON, scalar fallback)
   - KV Cache for incremental inference (4-5x speedup)
   - Pre-transposed weights and fused RMSNorm+MatMul

2. **Search Algorithm** (`search/`)
   - Lazy SMP with configurable thread count, varied depth offsets, lock-free shared TT
   - Alpha-beta with PVS, aspiration windows (exponential widening)
   - Static Exchange Evaluation (SEE) with x-ray sliding attacks
   - LMR on both PV and non-PV nodes with history + SEE adjustments
   - Triangular PV table with full PV lines in UCI output
   - Repetition detection (search path + game history)
   - Incremental move picking (avoids sorting pruned moves)
   - Stockfish-style history gravity across all history tables
   - Extensions: singular, recapture, passed pawn push
   - Null move pruning, razoring, futility pruning, SEE pruning

3. **Static Evaluation** (`search/eval.rs`)
   - Material + PeSTO piece-square tables
   - Bishop pair bonus, rook on open/semi-open file
   - Passed pawn bonus (rank-scaled), doubled/isolated pawn penalties
   - All positional terms use efficient bitboard operations

4. **Time Management** (`search/time.rs`)
   - Stockfish-inspired soft/hard time bounds with ply-aware scaling
   - Dynamic adjustment: best move stability, falling eval, instability factor

### Inference Pipeline

```text
Input tokens -> Embedding -> [TransformerBlock x 12] -> ln_f -> PolicyHead / ValueHead
                                       |
                        RMSNorm -> MultiheadAttention (GQA) -> LayerScale -> Residual
                                 -> RMSNorm -> BitSwiGLU -> LayerScale -> Residual

BitSwiGLU: w_gate(BitLinear) -> SiLU * w_val(BitLinear) -> w_out(BitLinear)

BitLinear weights: dual bitmask packed {pos_bits, neg_bits}
  - AVX-512: 2 bytes -> __mmask16 -> mask_add/sub (16 weights/cycle)
  - AVX2:    1 byte -> set1+and+cmpeq -> 8 float masks (8 weights/cycle)
```

---

## Project Structure

```text
KiyEngine/
+-- src/
|   +-- main.rs              Application entry point (UCI loop)
|   +-- lib.rs               Library exports
|   +-- constants.rs         Centralized constants
|   +-- book/                Opening book (Polyglot format)
|   |   +-- mod.rs           Book probing and Zobrist hashing
|   |   +-- polyglot_keys.rs Official 781-key Random64 array
|   +-- engine/              Neural network inference
|   |   +-- mod.rs           Engine struct, forward, forward_with_cache
|   |   +-- bitlinear.rs     BitLinear: bit-packed weights, SIMD packed GEMV
|   |   +-- transformer.rs   MultiheadAttention, BitSwiGLU, KVCache
|   |   +-- heads.rs         PolicyHead, ValueHead (pre-transposed weights)
|   |   +-- gguf.rs          GGUF format parser and mmap loader
|   +-- eval/                Position evaluation
|   |   +-- pst.rs           PeSTO piece-square tables
|   +-- search/              Search algorithms
|   |   +-- mod.rs           Alpha-beta, TT, move ordering, quiescence, SearchWorker
|   |   +-- eval.rs          Static evaluation (material + PST + positional)
|   |   +-- lazy_smp.rs      Lazy SMP thread management
|   |   +-- time.rs          TimeManager (soft/hard bounds, dynamic factors)
|   |   +-- mcts*.rs         MCTS implementations (experimental)
|   +-- nnue/                NNUE evaluation (planned)
|   |   +-- accumulator.rs   Incrementally updated accumulator
|   |   +-- features.rs      HalfKP feature extraction
|   |   +-- mod.rs            Network loading and inference
|   +-- simd/                SIMD utilities
|   +-- uci/                 UCI protocol handler
+-- Cargo.toml               Package manifest (v6.0.0)
+-- .cargo/config.toml       SIMD target features (AVX2/AVX-512/BMI2)
+-- ARCHITECTURE.md           Detailed architecture with visual diagrams
+-- README.md                Quick start guide
```

For detailed architectural diagrams, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Module Reference

### `constants`

Centralized constants to avoid magic numbers:

| Constant | Value | Description |
|----------|-------|-------------|
| `ENGINE_NAME` | "KiyEngine" | Engine identifier |
| `ENGINE_VERSION` | "6.0.0" | Current version |
| `VOCAB_SIZE` | 4608 | NN move vocabulary (72 planes x 64 squares) |
| `CONTEXT_LENGTH` | 256 | NN attention context length |
| `D_MODEL` | 512 | NN embedding dimension |
| `HIDDEN_DIM` | 1536 | SwiGLU hidden dimension |
| `NUM_LAYERS` | 12 | Transformer layers |
| `NUM_HEADS` | 8 | Attention heads |
| `NUM_KV_HEADS` | 2 | Grouped-query attention KV heads |
| `HEAD_DIM` | 64 | Per-head dimension |
| `MATE_SCORE` | 30000 | Score for checkmate |
| `MAX_PLY` | 128 | Maximum search ply |
| `NODE_CHECK_INTERVAL` | 8191 | Stop-flag polling interval (was 2047 in v5) |

### `engine`

**Engine** -- Main neural network container

- `forward(tokens)` -> `(policy_logits, value_score)` -- full sequence inference
- `forward_with_cache(tokens, cache)` -> incremental inference using KV cache

**BitLinear** -- 1.58-bit quantized linear layer with bit-packed weights

- `pos_bits` / `neg_bits` -- dual bitmask ternary weight storage (4x denser than i8)
- `forward(x)` -- F32 matmul for sequences, fused RMSNorm+MatMul
- `forward_single(x)` -- SIMD packed GEMV for single tokens
- `dispatch_packed_gemv()` -- runtime ISA selection (AVX-512 > AVX2 > NEON > scalar)
- `pack_ternary_bits()` -- packs i8 weights into dual bitmasks at load time

**MultiheadAttention** -- 8-head GQA attention with 2 KV heads and RoPE

- `forward(x)` -- standard attention forward pass
- `forward_cached(x, cache)` -- incremental KV cache attention

**BitSwiGLU** -- gated MLP: `SiLU(gate) * val -> out`

- Shared RMS normalization for gate and val branches
- Three BitLinear layers: w_gate, w_val, w_out

**TransformerBlock** -- full block with LayerScale and residual connections

**KVCache** -- per-layer Key/Value caching for incremental inference

- Stores (K, V) tensors per layer, tracks cached sequence length
- Validates token prefix match, clears on mismatch

**PolicyHead** / **ValueHead** -- pre-transposed weight heads

### `search`

**Searcher** -- top-level search coordinator

- Manages Lazy SMP thread pool and shared resources
- `search_async(depth, time_params, position_history)` -- main entry point
- Single NN forward pass at root, shared via `Arc` to all workers

**SearchWorker** -- per-thread search state

- Alpha-beta with PVS, exponential aspiration windows
- Static Exchange Evaluation (SEE) with x-ray attacks for pruning and LMR
- Triangular PV table (`pv_table[ply][i]`, `pv_len[ply]`) for full PV tracking
- Killer moves (2 per ply), history heuristic, continuation history, capture history
- Countermove heuristic
- Extensions: singular, recapture (same-square), passed pawn push (6th/7th rank)
- Repetition detection: fixed-size search path array + game hash history
- NNUE accumulator (infrastructure ready, evaluation not yet active)
- Incremental move picking via `pick_next_move`

**Transposition Table** -- lock-free atomic TT (Hyatt/Mann XOR trick)

- 16 bytes per entry: `[key^data, data]` in same cache line
- Depth-preferred replacement, always-replace for same hash
- Shared across all Lazy SMP threads with zero synchronization

**TimeManager** -- Stockfish-inspired time control

- Soft/hard time bounds with ply-aware scaling
- Dynamic adjustment: best move stability (sigmoid), falling eval (EMA), instability
- Periodic hard-stop checks every 8192 nodes inside search
- Soft-stop checks between iterative deepening iterations

### `uci`

**UciHandler** -- UCI protocol implementation

- KV cache integration (`Arc<Mutex<KVCache>>`)
- Position history tracking for repetition detection
- `movetime` command support
- Supports: Hash, Threads, Move Overhead, Analysis Mode, MultiPV

---

## Search Details

### Pruning Techniques

| Technique | Condition | Description |
|-----------|-----------|-------------|
| Razoring | depth <= 3, eval far below alpha | Skip to qsearch |
| Reverse Futility | depth <= 9, eval above beta + margin | Return eval |
| Null Move | depth >= 3, non-PV, not in check | Reduce R = 4 + depth/6 |
| Probcut | depth >= 5, shallow search confirms | Early beta cutoff |
| Late Move Pruning | depth <= 8, quiet moves beyond threshold | Skip late quiet moves |
| Futility Pruning | depth <= 8, eval + margin < alpha | Skip futile quiet moves |
| SEE Pruning (captures) | SEE < -20 × depth | Skip losing captures |
| SEE Pruning (quiets) | depth <= 8, SEE < -50 × depth | Skip losing quiet moves |
| SEE Pruning (qsearch) | SEE < 0 | Skip losing captures in qsearch |

### Reduction Techniques

| Technique | Condition | Description |
|-----------|-----------|-------------|
| LMR (PV) | depth >= 3, moves_searched > 2 | Reduced search, 1 less than non-PV |
| LMR (non-PV) | depth >= 3, moves_searched > 2 | Pre-computed table reduction |
| History adj. | quiet moves, good/bad history | Reduce less/more based on history |
| Killer adj. | killer moves | 1 less reduction |
| Improving adj. | position not improving | 1 more reduction |
| SEE adj. | captures with SEE < 0 | 1 more reduction |
| IIR | no TT move found, depth >= 3 | Reduce depth by 1 |

### Extension Techniques

| Technique | Condition | Description |
|-----------|-----------|-------------|
| Check extension | king in check | +1 depth |
| Singular extension | TT move much better, depth >= 6 | +1 depth for TT move |
| Recapture extension | capture on same square as previous move | +1 depth |
| Passed pawn push | pawn pushed to 6th or 7th rank | +1 depth |

### History Tables

All tables use Stockfish-style gravity updates: `entry += bonus - |entry| * bonus / 16384`

| Table | Dimensions | Purpose |
|-------|-----------|---------|
| History | `[2][64][64]` (side, from, to) | Quiet move ordering |
| Continuation | `[12][64][12][64]` (prev_piece, prev_to, cur_piece, cur_to) | Context-aware ordering |
| Capture | `[6][64][6]` (attacker, to, victim) | Capture move ordering |
| Killers | `[MAX_PLY][2]` | Beta-cutoff quiet moves |
| Countermove | `[64][64]` (from, to) | Refutation moves |

---

## Evaluation Details

### Material and PST

Standard material values (centipawns): Pawn=100, Knight=320, Bishop=330, Rook=500, Queen=900.
PeSTO midgame piece-square tables provide positional bonuses for each piece on each square.

### Positional Terms

| Term | Bonus/Penalty | Detection Method |
|------|--------------|------------------|
| Bishop pair | +30cp | Count >= 2 bishops per side |
| Rook on open file | +20cp | No friendly pawns on file |
| Rook on semi-open file | +10cp | No friendly pawns, enemy pawns present |
| Passed pawn | +5 to +100cp | No enemy pawns on same/adjacent files ahead |
| Doubled pawn | -10cp | Another friendly pawn on same file |
| Isolated pawn | -15cp | No friendly pawns on adjacent files |

Passed pawn detection uses zero-loop bitboard computation:
`file_and_adj & !((1 << ((rank+1) * 8)) - 1)` -- a single bit-shift per pawn.

---

## Time Management Details

### Allocation Formula

```text
time_pool = remaining_time + increment * (mtg - 1) - overhead * mtg

Sudden death (no movestogo):
    mtg_estimate = (50 - 0.4 * ply).clamp(20, 50)
    optimum = ply_factor * time_pool
    maximum = min(5.5 * optimum, 0.80 * remaining - overhead)

Classical (movestogo given):
    optimum = (0.90 + ply / 120) * time_pool / mtg
    maximum = (1.5 + 0.12 * mtg) * optimum
```

### Dynamic Factors (applied at each ID iteration)

```text
adjusted_time = optimum * falling_eval * time_reduction * instability

falling_eval   = clamp((12 + 2 * (avg_score - cur_score)) / 100, 0.60, 1.70)
time_reduction = 1.40 / (2.20 * (0.70 + 0.80 / (1 + exp(-0.50 * (depth - center)))))
instability    = 1.0 + 1.5 * best_move_changes / 10.0
```

---

## Building and Running

### Prerequisites

- Rust 1.70+ (stable toolchain)
- Model file: `kiyengine.gguf` (GGUF format, ~28 MB)

### Build Commands

```bash
# Standard release build (optimized, LTO enabled)
cargo build --release

# Run engine (UCI mode)
cargo run --release --bin kiy_engine_v5_alpha

# Run model benchmarks
cargo run --release --bin model_benchmark

# CUDA support (if available)
cargo build --release --features cuda
```

### UCI Usage Example

```text
> uci
id name KiyEngine V6.0.0
id author Khoi
option name Hash type spin default 512 min 64 max 65536
option name Threads type spin default 4 min 1 max 256
option name Move Overhead type spin default 60 min 0 max 5000
option name Analysis Mode type check default false
option name MultiPV type spin default 1 min 1 max 500
uciok

> setoption name Hash value 1024
> setoption name Threads value 4
> isready
readyok

> position startpos moves e2e4 e7e5
> go wtime 60000 btime 60000 winc 1000 binc 1000
info depth 15 score cp 25 nodes 1223840 nps 1011026 pv b1c3 ...
bestmove b1c3

> position fen 3r2k1/pp3pp1/2p1bn1p/8/4P3/2N2N2/PPP2PPP/3R2K1 w - - 0 1
> go movetime 5000
info depth 15 score cp 925 nodes 1223840 nps 1011026 pv d1d8
bestmove d1d8
```

---

## Configuration

### UCI Options

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| Hash | spin | 512 | 64-65536 | TT size in MB |
| Threads | spin | 4 | 1-256 | Search threads (Lazy SMP) |
| Move Overhead | spin | 60 | 0-5000 | Time buffer (ms) |
| Analysis Mode | check | false | -- | Disable opening book |
| MultiPV | spin | 1 | 1-500 | Multiple principal variations |

### SIMD Configuration

Set in `.cargo/config.toml`:

```toml
[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "target-feature=+avx2,+bmi2,+avx512f"]
```

The engine auto-detects CPU features at runtime and selects the optimal SIMD path:
AVX-512 > AVX2 > NEON > Scalar

---

## Development Guidelines

### Code Style

1. **Safety Comments**: All `unsafe` blocks require `// SAFETY:` comments
2. **Constants**: Use `constants.rs` for magic numbers
3. **Documentation**: Public items should have doc comments (`///`)
4. **Error Handling**: Use `anyhow::Result` for application code

### Testing

```bash
# Run unit tests
cargo test

# Run with optimizations (for benchmark accuracy)
cargo test --release

# Run model benchmark suite
cargo run --release --bin model_benchmark

# Quick fixed-depth benchmark
echo "uci\nisready\nposition fen <FEN>\ngo depth 14" | ./target/release/kiy_engine_v5_alpha
```

### Architecture Principles

1. **Lockless Design**: TT uses atomic operations (XOR trick) for SMP safety
2. **Bit-Packed Weights**: Dual bitmask format for 4x memory reduction and SIMD efficiency
3. **SIMD Optimization**: AVX-512 > AVX2 > NEON > Scalar, auto-detected at runtime
4. **KV Cache**: Incremental inference -- only process new tokens
5. **Single NN Call**: Forward pass computed once at root, shared across all threads
6. **Memory Safety**: All unsafe code is isolated in SIMD modules with documented preconditions
7. **Zero-Allocation Hot Path**: Fixed-size arrays for search path, pre-computed LMR table
8. **History Gravity**: Stockfish-style update formula prevents table saturation

---

## Troubleshooting

### "Failed to load engine"

- Ensure `kiyengine.gguf` is in the executable directory or alongside `Cargo.toml`
- Falls back to `kiyengine.safetensors` if GGUF not found

### Low NPS

- Use `--release` build (debug builds are ~100x slower)
- Check `Threads` setting matches your CPU core count
- Verify AVX2/AVX-512 support: `lscpu | grep avx`

### Out of memory

- Reduce `Hash` (e.g., `setoption name Hash value 128`)
- Lower thread count

### Book not loading

- Ensure `book1.bin` and `book2.bin` exist relative to the executable
- Use `Analysis Mode` to disable book lookup

---

## License

Apache License 2.0 -- See LICENSE file for details.

Copyright 2026 Khoi
