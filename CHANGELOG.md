# Changelog

All notable changes to KiyEngine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [5.0.0] - 2026-02-08

### Added
- **GGUF Format Support**: New `.gguf` model format replacing `.safetensors`
  - 4x smaller file size (~28MB vs ~114MB uncompressed)
  - Memory-mapped loading for instant startup
  - Native BitNet I8 quantization support
- Rust GGUF parser with full metadata and tensor support
- `inspect_gguf` binary for inspecting GGUF models
- Python conversion script `convert_to_gguf.py` for migrating v4 models
- Automatic format detection with GGUF priority and safetensors fallback
- **Polyglot Opening Book Integration**
  - Dual-book support (`book1.bin`, `book2.bin`) loaded fully into memory at startup
  - Official Polyglot Zobrist hashing with hardcoded 781-key Random64 array
  - Weighted random move selection with minimum weight threshold
  - Book usage limited to first 30 plies (`MAX_BOOK_PLY`)
  - Executable-relative path resolution for portability
  - Root-only probe — zero overhead inside recursive search
- **Lock-Free Transposition Table** (Hyatt/Mann XOR trick)
  - Replaced `Mutex<TTBucket>` with atomic `AtomicU64` interleaved layout
  - 16 bytes per entry: `[key ^ data, data]` in same cache line
  - Depth-preferred replacement policy for cross-position collisions
  - Always-replace for same-hash updates
- **Pre-computed LMR Lookup Table** (`[[u8; 64]; 64]`)
  - Eliminates `f32::ln()` transcendental math from the move loop hot path
- **Material + PST Static Evaluation** (`src/search/eval.rs`)
  - Fast fallback evaluation used in alpha-beta and quiescence search
  - Piece-square tables for positional scoring
- **Search Stability Safeguards**
  - `MAX_PLY` guard in quiescence search (prevents unbounded recursion)
  - `stop_flag` check in quiescence search (was entirely missing)
  - Periodic time check every 2048 nodes inside both alpha-beta and quiescence
  - `TimeManager` wired into `SearchWorker` for in-search abort

### Changed
- Default model path changed from `kiyengine.safetensors` to `kiyengine.gguf`
- Engine version bumped to 5.0.0
- Updated all shell scripts and binaries to use GGUF format
- Git LFS now tracks `.gguf` files
- **NN forward pass computed once** before spawning threads, shared via `Arc`
  - Previously: 4 redundant forward passes (1 per thread)
  - Now: 1 forward pass, result cloned into each worker
- Root policy accessed by reference (`as_deref()`) instead of `clone()` per iteration
- Removed unused `engine` and `move_history` fields from `SearchWorker`
- `Vec::with_capacity()` hints for move ordering (40) and quiescence captures (16)
- Default UCI options: Hash=512MB (min 64MB), Threads=4, Move Overhead=60ms
- Aspiration windows start at depth > 3 with ±50 centipawn window

### Deprecated
- Legacy `.safetensors` format (still supported for backward compatibility)

### Performance
- **2.19M NPS** at depth 18 (up from ~475K before optimizations)
- Depth 15–19 reached in timed games (40 moves/60 seconds)
- Zero heap allocations for NN policy access in search loop
- Lock-free TT eliminates all mutex contention in Lazy SMP

## [4.5.0] - 2026-02-07

### Added
- AlphaZero-style MCTS (Monte Carlo Tree Search) implementation
- Arena-based tree structure for cache-efficient MCTS nodes
- SIMD-optimized BitNet XNOR/popcount using AVX2 intrinsics
- Rayon-based parallel MCTS with Virtual Loss
- AtomicF32 using AtomicU64 bit-casting for lock-free updates
- PUCT selection with dynamic Cpuct scaling
- Dirichlet noise for root exploration
- First Play Urgency (FPU) for unvisited nodes
- Thread-safe node expansion using atomic CAS

### Changed
- Refactored search module structure
- Improved code documentation
- Standardized module exports
- Fixed all compiler warnings

### Fixed
- Resolved borrow checker issues in MCTS backpropagation
- Fixed unused variable warnings across codebase
- Cleaned up dead code in mcts_memory.rs

## [4.4.5] - Previous Release

### Features
- BitNet 1.58-bit Transformer architecture for evaluation
- Lazy SMP parallel search
- Alpha-Beta search with PVS, aspiration windows
- Transposition table with lock-free updates
- UCI protocol support
- Quiescence search
- MVV-LVA and killer/history move ordering
