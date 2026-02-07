# Changelog

All notable changes to KiyEngine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [5.0.0] - 2026-02-07

### Added
- **GGUF Format Support**: New `.gguf` model format replacing `.safetensors`
  - 4x smaller file size (~28MB vs ~114MB uncompressed)
  - Memory-mapped loading for instant startup
  - Native BitNet I8 quantization support
- Rust GGUF parser with full metadata and tensor support
- `inspect_gguf` binary for inspecting GGUF models
- Python conversion script `convert_to_gguf.py` for migrating v4 models
- Automatic format detection with GGUF priority and safetensors fallback

### Changed
- Default model path changed from `kiyengine.safetensors` to `kiyengine.gguf`
- Engine version bumped to 5.0.0
- Updated all shell scripts and binaries to use GGUF format
- Git LFS now tracks `.gguf` files

### Deprecated
- Legacy `.safetensors` format (still supported for backward compatibility)

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
