# Changelog

All notable changes to KiyEngine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [6.0.0] - 2026-02-12

### Added

- **Stockfish-Inspired Time Management** -- complete rewrite of `TimeManager`
  - Soft/hard time bounds with ply-aware scaling
  - Best move stability factor (sigmoid-based time reduction when best move is stable)
  - Falling eval factor (EMA-based score tracking, spends more time when eval drops)
  - Best move instability factor (extends time when best move changes frequently)
  - Adaptive moves-to-go estimation for sudden death: `(50 - 0.4 * ply).clamp(20, 50)`
- **Repetition Detection** in alpha-beta search
  - Fixed-size search path array (zero heap allocations)
  - 2-fold detection in search path, checking every-other position (same side to move)
  - Game history integration (last 100 positions from UCI position command)
  - Position history passed from UCI handler through `search_async` to all workers
- **Enhanced Static Evaluation** -- new positional terms beyond material + PST
  - Bishop pair bonus (+30cp)
  - Rook on open file (+20cp) and semi-open file (+10cp)
  - Passed pawn bonus (rank-scaled 5-100cp, zero-loop bitboard mask computation)
  - Doubled pawn penalty (-10cp)
  - Isolated pawn penalty (-15cp)
- **LMR on PV Nodes** -- Late Move Reductions now apply to PV nodes with 1 less reduction
- **Stockfish-Style History Gravity** -- `bonus - |h| * bonus / 16384` across all tables
  - Applied to: history heuristic, continuation history, capture history
  - Prevents table saturation, maintains ordering signal quality over long searches
- **Incremental Move Picking** -- `pick_next_move` selects best remaining move on demand
  - Avoids full sorting of moves pruned early by beta cutoffs
  - Applied in both alpha-beta and quiescence search
- **movetime UCI Command** -- fixed-time search support
  - Converts `movetime` to equivalent `wtime/btime` with `movestogo=1`
- **Singular Extensions** -- extend TT move when significantly better than alternatives
- **Continuation History** -- `[prev_piece][prev_to][cur_piece][cur_to]` table for move ordering
- **Countermove Heuristic** -- records refutation moves for improved quiet move ordering
- **NNUE Infrastructure** -- accumulator, feature extraction, and network loading in place
  - `use_nnue_eval` flag allows conditional activation (currently disabled)
  - Accumulator updates skipped when NNUE is inactive (+19% NPS savings)

### Changed

- **Iterative Deepening Loop** rewritten
  - Exponential aspiration window widening on fail high/low
  - Single legal move early exit (no need to search further)
  - Best move change tracking per iteration for TimeManager feedback
  - Search path reset at start of each iteration
- **Node Check Interval** increased from 4096 to 8192 (reduced polling overhead)
- **Aspiration Windows** start at depth >= 4 with exponential widening (was fixed progressive)
- **History Table Updates** use gravity formula instead of flat add/clamp
- **Move Ordering** changed from full sort to score-only with incremental picking
- Engine version bumped to 6.0.0

### Performance

- **3.6x fewer nodes** to reach depth 20 (9.3M vs 33.9M in v5.2.0)
- **2.2x faster wall time** to reach depth 20 (7.1s vs 15.6s in v5.2.0)
- **+19% NPS** from skipping unused NNUE accumulator updates
- Depth 15+ reached in 60s+1s time control (4 threads)
- Zero-allocation repetition detection (fixed-size array, no Vec)
- Incremental move picking avoids sorting moves never searched

### Benchmark (depth 20, 4 threads)

```
Position: 3r2k1/pp3pp1/2p1bn1p/8/4P3/2N2N2/PPP2PPP/3R2K1 w

Version     Nodes         NPS          Wall Time
-------     ----------    ----------   ---------
v5.2.0      33,900,000    2,180,000    ~15.6s
v6.0.0       9,285,144    1,312,392     ~7.1s
```

---

## [5.2.0] - 2026-02-10

### Added
- **Bit-Packed Ternary Weights** — dual bitmask format (`pos_bits` + `neg_bits`)
  - Each ternary weight {-1, 0, +1} stored in 2 bits across two masks (4× denser than i8)
  - `pack_ternary_bits()` packing computed at load time from i8 weights
  - Weight memory: ~3 MB packed vs ~12 MB i8 (4× reduction)
- **AVX-512 Packed GEMV** — load 2 bytes directly as native `__mmask16`, zero expansion needed
  - `_mm512_mask_add_ps` / `_mm512_mask_sub_ps` process 16 weights per cycle
- **AVX2 Packed GEMV** — byte→8-lane mask expansion using `set1_epi32` + `and` + `cmpeq` trick
- **NEON Packed GEMV** — 4-lane bit expansion with `vbslq_f32`
- **Scalar Packed GEMV** — portable fallback with bit extraction loop
- `dispatch_packed_gemv()` method for automatic ISA selection at runtime

### Changed
- `BitLinear` struct now stores `pos_bits`, `neg_bits`, `row_bytes` alongside `weight_i8`
- `forward_single()` uses packed GEMV instead of i8→f32 conversion SIMD
- `forward()` stays F32 matmul for sequences (candle's tiled GEMM beats per-token GEMV for batch > 1)

### Performance
- KV Cache incremental inference: 4.5–5× speedup (packed GEMV on single-token path)
- Weight data 4× smaller → improved CPU cache utilization
- Correctness verified: `diff=0.000000` between cached and full inference

## [5.1.0] - 2026-02-09

### Added
- **KV Cache** for incremental transformer inference
  - `KVCache` struct stores per-layer (K, V) tensors, tracks cached sequence length
  - `MultiheadAttention::forward_cached()` processes only new tokens, concatenates with cached K/V
  - `TransformerBlock::forward_cached()` and `Engine::forward_with_cache()`
  - UCI integration via `Arc<parking_lot::Mutex<KVCache>>`, cleared on `ucinewgame`
- **Capture History** — `[piece_moved][to_sq][captured_piece]` table (6×64×6)
  - Updated on capture beta cutoffs, used to improve capture ordering
- **Check Extensions in Quiescence** — searches all legal moves when in check, detects checkmate
- **History Aging** — halves all history + capture_history scores between iterative deepening iterations

### Changed
- **Pre-transposed `out_proj` weight** in `MultiheadAttention`
  - Replaced `candle_nn::Linear` with manual `broadcast_matmul` on pre-transposed contiguous weight
- **Pre-transposed head weights** — `PolicyHead.w_t` and `ValueHead.w1_t/w2_t` transposed at load time
- **Fused RMSNorm+MatMul** in `BitLinear` — `fused_rms_norm_matmul()` eliminates intermediate tensor allocations

### Performance
- KV Cache: 3.6–4.6× speedup for incremental inference
- seq_len=4: ~10.4ms (was 12.7ms), seq_len=8: ~9.6ms (was 16.7ms)
- KV cached +1 token: ~5.2ms
- Search NPS: ~625K at depth 14 (single-thread equivalent)

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
