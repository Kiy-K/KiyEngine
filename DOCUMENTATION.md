# KiyEngine v5.2.0 Documentation

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Module Reference](#module-reference)
- [Building and Running](#building-and-running)
- [Configuration](#configuration)
- [Development Guidelines](#development-guidelines)

---

## Architecture Overview

KiyEngine is a high-performance UCI chess engine built in Rust with the following key components:

### Core Systems

1. **Neural Network Evaluation** (`engine/`)
   - BitNet 1.58-bit Transformer (8 layers, 512 d_model, 8 heads, 1024 hidden dim)
   - Bit-packed ternary weights with dual bitmask format (4× denser than i8)
   - SIMD-optimized packed GEMV (AVX-512, AVX2, NEON, scalar fallback)
   - KV Cache for incremental inference (4–5× speedup)
   - Pre-transposed weights and fused RMSNorm+MatMul

2. **Search Algorithm** (`search/`)
   - Lazy SMP (4 threads default)
   - Alpha-beta with PVS, aspiration windows
   - Lock-free transposition table (Hyatt/Mann XOR trick)
   - Quiescence search with check extensions
   - Capture history and history aging

3. **Static Evaluation** (`eval/`)
   - Fast Material + PST evaluation for quiescence and pruning
   - NN evaluation at root only (shared via `Arc`)

### Inference Pipeline

```
Input tokens → Embedding → [TransformerBlock × 8] → ln_f → PolicyHead / ValueHead
                                    ↓
                     RMSNorm → MultiheadAttention → LayerScale → Residual
                              → RMSNorm → BitSwiGLU → LayerScale → Residual

BitSwiGLU: w_gate(BitLinear) → SiLU ⊙ w_val(BitLinear) → w_out(BitLinear)

BitLinear weights: dual bitmask packed {pos_bits, neg_bits}
  - AVX-512: 2 bytes → __mmask16 → mask_add/sub (16 weights/cycle)
  - AVX2:    1 byte → set1+and+cmpeq → 8 float masks (8 weights/cycle)
```

---

## Project Structure

```
KiyEngine/
├── src/
│   ├── main.rs              # Application entry point (UCI loop)
│   ├── lib.rs               # Library exports
│   ├── constants.rs         # Centralized constants
│   ├── book/                # Opening book (Polyglot format)
│   │   ├── mod.rs           # Book probing & Zobrist hashing
│   │   └── polyglot_keys.rs # Official 781-key Random64 array
│   ├── engine/              # Neural network inference
│   │   ├── mod.rs           # Engine struct, forward, forward_with_cache
│   │   ├── bitlinear.rs     # BitLinear: bit-packed weights, SIMD packed GEMV
│   │   ├── transformer.rs   # MultiheadAttention, BitSwiGLU, TransformerBlock, KVCache
│   │   ├── heads.rs         # PolicyHead, ValueHead (pre-transposed weights)
│   │   └── gguf.rs          # GGUF format parser & mmap loader
│   ├── eval/                # Position evaluation
│   │   └── pst.rs           # Piece-square tables
│   ├── search/              # Search algorithms
│   │   ├── mod.rs           # Alpha-beta, TT, move ordering, quiescence
│   │   ├── eval.rs          # Material + PST static evaluation
│   │   ├── lazy_smp.rs      # Lazy SMP thread management
│   │   ├── time.rs          # Time management
│   │   └── mcts*.rs         # MCTS implementations (experimental)
│   ├── simd/                # SIMD utilities
│   └── uci/                 # UCI protocol handler
├── Cargo.toml               # Package manifest
├── .cargo/config.toml       # SIMD target features (AVX2/AVX-512/BMI2)
└── README.md                # Quick start guide
```

---

## Module Reference

### `constants`

Centralized constants to avoid magic numbers:

| Constant | Value | Description |
|----------|-------|-------------|
| `ENGINE_NAME` | "KiyEngine" | Engine identifier |
| `ENGINE_VERSION` | "5.2.0" | Current version |
| `VOCAB_SIZE` | 4608 | NN move vocabulary (72 planes × 64 squares) |
| `CONTEXT_LENGTH` | 64 | NN attention context length |
| `D_MODEL` | 512 | NN embedding dimension |
| `HIDDEN_DIM` | 1024 | SwiGLU hidden dimension |
| `NUM_LAYERS` | 8 | Transformer layers |
| `NUM_HEADS` | 8 | Attention heads |
| `MATE_SCORE` | 30000 | Score for checkmate |
| `MAX_PLY` | 128 | Maximum search ply |
| `NODE_CHECK_INTERVAL` | 2047 | Stop-flag polling interval |

### `engine`

**Engine** — Main neural network container

- `forward(tokens)` → `(policy_logits, value_score)` — full sequence inference
- `forward_with_cache(tokens, cache)` → incremental inference using KV cache

**BitLinear** — 1.58-bit quantized linear layer with bit-packed weights

- `pos_bits` / `neg_bits` — dual bitmask ternary weight storage (4× denser than i8)
- `forward(x)` — F32 matmul for sequences, fused RMSNorm+MatMul
- `forward_single(x)` — SIMD packed GEMV for single tokens
- `dispatch_packed_gemv()` — runtime ISA selection (AVX-512 > AVX2 > NEON > scalar)
- `pack_ternary_bits()` — packs i8 weights into dual bitmasks at load time

**MultiheadAttention** — 8-head attention with pre-transposed `out_proj` weight

- `forward(x)` — standard attention forward pass
- `forward_cached(x, cache)` — incremental KV cache attention

**BitSwiGLU** — gated MLP: `SiLU(gate) ⊙ val → out`

- Shared RMS normalization for gate and val branches
- Three BitLinear layers: w_gate, w_val, w_out

**TransformerBlock** — full block with LayerScale and residual connections

**KVCache** — per-layer Key/Value caching for incremental inference

- Stores (K, V) tensors per layer, tracks cached sequence length
- Validates token prefix match, clears on mismatch

**PolicyHead** / **ValueHead** — pre-transposed weight heads

### `search`

**SearchWorker** — per-thread search state

- Alpha-beta with PVS, aspiration windows
- Killer moves (2 per ply), history heuristic, capture history
- History aging between iterative deepening iterations
- LMR with pre-computed lookup table
- Quiescence search with check extensions and `MAX_PLY` guard

**Transposition Table** — lock-free atomic TT (Hyatt/Mann XOR trick)

- 16 bytes per entry: `[key^data, data]` in same cache line
- Depth-preferred replacement, always-replace for same hash

**TimeManager** — time control with periodic in-search checks every 2048 nodes

### `uci`

**UciHandler** — UCI protocol implementation

- KV cache integration (`Arc<Mutex<KVCache>>`)
- Supports: Hash, Threads, Move Overhead, Analysis Mode, MultiPV

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
id name KiyEngine V5.2.0
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
> go wtime 60000 btime 60000 movestogo 40
info depth 15 score cp 30 nodes 2500000 nps 2100000 pv g1f3 ...
bestmove g1f3
```

---

## Configuration

### UCI Options

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| Hash | spin | 512 | 64–65536 | TT size in MB |
| Threads | spin | 4 | 1–256 | Search threads (Lazy SMP) |
| Move Overhead | spin | 60 | 0–5000 | Time buffer (ms) |
| Analysis Mode | check | false | — | Disable opening book |
| MultiPV | spin | 1 | 1–500 | Multiple principal variations |

### SIMD Configuration

Set in `.cargo/config.toml`:

```toml
[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "target-feature=+avx2,+bmi2,+avx512f"]
```

The engine auto-detects CPU features at runtime and selects the optimal SIMD path:
AVX-512 → AVX2 → NEON → Scalar

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
```

### Architecture Principles

1. **Lockless Design**: TT uses atomic operations (XOR trick) for SMP safety
2. **Bit-Packed Weights**: Dual bitmask format for 4× memory reduction and SIMD efficiency
3. **SIMD Optimization**: AVX-512 > AVX2 > NEON > Scalar, auto-detected at runtime
4. **KV Cache**: Incremental inference — only process new tokens
5. **Single NN Call**: Forward pass computed once at root, shared across all threads
6. **Memory Safety**: All unsafe code is isolated in SIMD modules with documented preconditions

---

## Troubleshooting

### "Failed to load engine"

- Ensure `kiyengine.gguf` is in the executable directory or alongside `Cargo.toml`
- Falls back to `kiyengine.safetensors` if GGUF not found

### Low NPS

- Use `--release` build (debug builds are ~100× slower)
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

Apache License 2.0 — See LICENSE file for details.

© 2026 Khoi
