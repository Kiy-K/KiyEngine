# KiyEngine Documentation

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
   - BitNet 1.58-bit Transformer architecture
   - Ternary weights {-1, 0, +1} for efficient inference
   - SIMD optimizations (AVX-512, AVX2, NEON)

2. **Search Algorithm** (`search/`)
   - Lazy SMP (Symmetric Multi-Processing)
   - Alpha-beta pruning with PVS (Principal Variation Search)
   - Lockless transposition table
   - Quiescence search

3. **Hybrid Evaluation** (`eval/`)
   - Fast PST + material evaluation for quiescence
   - Deep BitNet evaluation for full search
   - Configurable blending between methods

---

## Project Structure

```
KiyEngine/
├── src/
│   ├── main.rs              # Application entry point
│   ├── lib.rs               # Library exports and tests
│   ├── constants.rs         # Centralized constants
│   ├── book/                # Opening book (Polyglot format)
│   ├── engine/              # Neural network evaluation
│   │   ├── mod.rs           # Main Engine struct
│   │   ├── bitlinear.rs     # BitNet 1.58-bit linear layers
│   │   └── heads.rs         # Policy and value heads
│   ├── eval/                # Position evaluation
│   │   ├── hybrid.rs        # Hybrid fast/deep evaluation
│   │   ├── pst.rs           # Piece-square tables
│   │   └── batching.rs      # Batched inference (experimental)
│   ├── search/              # Search algorithms
│   │   ├── lazy_smp.rs      # Lazy SMP search
│   │   ├── tt.rs            # Lockless transposition table
│   │   └── time.rs          # Time management
│   ├── simd/                # SIMD optimizations
│   ├── uci/                 # UCI protocol implementation
│   └── bin/                 # Additional binaries
│       ├── bench.rs         # Performance benchmark
│       └── chess_bench.rs   # Position test suite
├── Cargo.toml               # Package manifest
└── README.md                # Quick start guide
```

---

## Module Reference

### `constants`

Centralized constants to avoid magic numbers:

| Constant | Value | Description |
|----------|-------|-------------|
| `ENGINE_NAME` | "KiyEngine" | Engine identifier |
| `ENGINE_VERSION` | "4.5.0" | Current version |
| `DEFAULT_TT_SIZE_MB` | 64 | Default hash table size |
| `VOCAB_SIZE` | 4608 | NN move vocabulary size |
| `CONTEXT_LENGTH` | 32 | NN attention context |
| `D_MODEL` | 1024 | NN embedding dimension |
| `MATE_SCORE` | 30000 | Score for checkmate |

### `engine`

**Engine** - Main neural network container
- `forward(tokens)` - Run inference on move history
- Returns: `(policy_logits, value_score)`

**BitLinear** - 1.58-bit quantized linear layer
- Ternary weights for CPU-efficient inference
- Auto-vectorization via SIMD

**ValueHead** - Position evaluation head
- Outputs scalar in range [-1, 1]
- Uses tanh activation

**PolicyHead** - Move probability head  
- Outputs logits for 4608 possible moves
- Used for move ordering hints

### `search::tt`

**AtomicTT** - Lockless transposition table
- 64-bit packed entries (atomic operations)
- Stores: hash, score, best_move, depth, flag, age
- Always-replace strategy optimized for Lazy SMP

Methods:
- `new(size_mb)` - Create TT with given size
- `store(hash, score, best_move, depth, flag, age)` - Store entry
- `probe(hash)` - Retrieve entry if available
- `clear()` - Clear all entries

### `search::lazy_smp`

**Searcher** - Main search controller
- Manages worker threads for Lazy SMP
- Handles time management and stop conditions

**Worker** - Per-thread search state
- Alpha-beta search implementation
- Killer moves and history heuristics
- PVS and LMR (Late Move Reduction)

### `uci`

**UciHandler** - UCI protocol implementation
- Supports standard UCI commands
- Options: Hash, Threads, Move Overhead, MultiPV

---

## Building and Running

### Prerequisites
- Rust 1.70+ (stable toolchain)
- Model file: `kiyengine_v4.safetensors` in executable directory

### Build Commands

```bash
# Standard release build (optimized)
cargo build --release

# Run with default settings
cargo run --release

# Run benchmarks
cargo run --release --bin bench
cargo run --release --bin chess_bench

# CUDA support (if available)
cargo build --release --features cuda
```

### UCI Usage Example

```
> uci
id name KiyEngine V4.5.0
id author Khoi
option name Hash type spin default 64 min 1 max 65536
option name Threads type spin default 1 min 1 max 256
...
uciok

> setoption name Hash value 256
> setoption name Threads value 4
> isready
readyok

> position startpos moves e2e4 e7e5
> go wtime 300000 btime 300000
info depth 1 score cp 30 nodes 25 nps 2500 pv e2e4
...
bestmove g1f3
```

---

## Configuration

### UCI Options

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| Hash | spin | 64 | 1-65536 | TT size in MB |
| Threads | spin | 1 | 1-256 | Search threads |
| Move Overhead | spin | 30 | 0-5000 | Time buffer (ms) |
| MultiPV | spin | 1 | 1-500 | Multiple principal variations |
| UCI_ShowWDL | check | false | - | Show win/draw/loss |

### Environment Variables

- `KIYENGINE_MODEL_PATH` - Override default model file location

---

## Development Guidelines

### Code Style

1. **Safety Comments**: All `unsafe` blocks require `// SAFETY:` comments explaining preconditions
2. **Constants**: Use `constants.rs` for magic numbers
3. **Documentation**: Public items should have doc comments (`///`)
4. **Error Handling**: Use `anyhow::Result` for application code, specific errors for libraries

### Testing

```bash
# Run unit tests
cargo test

# Run with optimizations (for benchmark accuracy)
cargo test --release
```

### Architecture Principles

1. **Lockless Design**: Transposition table uses atomic operations for SMP safety
2. **Hybrid Evaluation**: Fast PST evaluation + Deep NN evaluation
3. **SIMD Optimization**: Target AVX-512 > AVX2 > NEON > Scalar fallback
4. **Memory Safety**: All unsafe code is isolated and documented

### Adding New Features

1. Add constants to `constants.rs` if widely used
2. Implement in appropriate module
3. Export in parent `mod.rs` if public API
4. Update this documentation
5. Add tests in `lib.rs` or module tests

---

## Troubleshooting

### "Failed to load engine"
- Ensure `kiyengine_v4.safetensors` is in the executable directory
- Check file permissions

### Low NPS (nodes per second)
- Use `--release` flag for optimized builds
- Verify CPU supports AVX2 or AVX-512

### Out of memory
- Reduce `Hash` option size
- Lower thread count

---

## License

Apache License 2.0 - See LICENSE file for details.

© 2026 Khoi
