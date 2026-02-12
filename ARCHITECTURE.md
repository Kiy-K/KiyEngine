# KiyEngine Architecture

This document describes the internal architecture of KiyEngine v6.1.0 -- a high-performance UCI chess engine written in Rust, featuring a hybrid evaluation system combining a BitNet 1.58-bit Transformer for root move ordering with a NNUE deep network for per-node evaluation, all powered by hand-tuned AVX2 SIMD.

---

## Table of Contents

- [System Overview](#system-overview)
- [High-Level Data Flow](#high-level-data-flow)
- [Search Architecture](#search-architecture)
- [Neural Network Pipeline](#neural-network-pipeline)
- [Evaluation System](#evaluation-system)
- [Time Management](#time-management)
- [Move Ordering](#move-ordering)
- [Transposition Table](#transposition-table)
- [Threading Model](#threading-model)
- [Opening Book](#opening-book)
- [UCI Protocol Layer](#uci-protocol-layer)
- [Module Map](#module-map)

---

## System Overview

```text
+---------------------------------------------------------------------+
|                         KiyEngine v6.1.0                            |
|                                                                     |
|  +------------+    +----------------+    +------------------------+ |
|  |            |    |                |    |                        | |
|  |  UCI Layer |--->|  Search Core   |--->|  NNUE Evaluation       | |
|  |            |    |  (Lazy SMP)    |    |  (SCReLU Deep Network) | |
|  +-----+------+    +-------+--------+    +------------------------+ |
|        |                   |                                        |
|        v                   v                                        |
|  +------------+    +----------------+    +------------------------+ |
|  |            |    |                |    |                        | |
|  | Opening    |    | Transposition  |    |  BitNet Transformer    | |
|  | Book       |    | Table (atomic) |    |  (Root Policy, opt.)  | |
|  | (embedded) |    |                |    |                        | |
|  +------------+    +----------------+    +------------------------+ |
+---------------------------------------------------------------------+
```

KiyEngine is composed of five major subsystems:

- **UCI Layer** -- Protocol handler, position management, game history tracking
- **Search Core** -- Lazy SMP alpha-beta with PVS, pruning, and reductions
- **NNUE Evaluation** -- Deep network with AVX2 SIMD, incrementally updated accumulators
- **BitNet Transformer** -- 1.58-bit Transformer for root policy (optional, loaded from disk)
- **Transposition Table** -- Lock-free shared hash table across all threads
- **Embedded Assets** -- NNUE weights and opening books compiled into the binary

---

## High-Level Data Flow

The following diagram traces a single `go` command from UCI input to `bestmove` output:

```text
UCI "go wtime 60000 ..."
        |
        v
+------------------+
|  Parse go params |
|  Build TimeManager (soft/hard bounds, ply-aware scaling)
+--------+---------+
         |
         v
+------------------+       +-------------------+
|  Opening Book    |--hit->| Return book move  |---> "bestmove ..."
|  Probe (<=30 ply)|       +-------------------+
+--------+---------+
         | miss
         v
+------------------+
|  NN Forward Pass |  Single forward pass, shared via Arc
|  (root only)     |  Produces: policy logits [4608], value score
+--------+---------+
         |
         v
+---------------------------+
|  Spawn Lazy SMP Workers   |  Thread 0: full depth
|  (N threads, shared TT)   |  Thread 1..N-1: depth - (1 + i%3)
+--------+------------------+
         |
         v
+---------------------------+
|  Iterative Deepening      |
|  depth 1 -> max_depth     |
|                           |
|  Each iteration:          |
|    - Aspiration window    |
|    - Alpha-beta + PVS     |
|    - TimeManager check    |
|    - Update best move     |
+--------+------------------+
         |
         v
+---------------------------+
|  Collect best result      |
|  from all threads         |
+--------+------------------+
         |
         v
"bestmove d1d8"
```

---

## Search Architecture

### Iterative Deepening Loop

```text
iterative_deepening(max_depth):
    for depth in 1..=max_depth:
        reset search_path
        alpha = -INFINITY
        beta  = +INFINITY

        if depth >= 4:
            alpha = prev_score - 50      // aspiration window
            beta  = prev_score + 50

        loop:
            score = alpha_beta(alpha, beta, depth, ply=0)

            if score <= alpha:            // fail low
                alpha = (alpha - delta)   // widen exponentially
                delta *= 2
            else if score >= beta:        // fail high
                beta = (beta + delta)
                delta *= 2
            else:
                break                     // score within window

        TimeManager.update_iteration(depth, score, best_move_changed)
        if TimeManager.should_stop_soft():
            break
```

### Alpha-Beta with PVS

The core search function processes each node through a structured pipeline:

```text
alpha_beta(alpha, beta, depth, ply):
    |
    +-- [1] Abort check (every 8192 nodes, poll TimeManager)
    +-- [2] Repetition detection (search path + game history)
    +-- [3] Check extension (+1 depth if in check)
    +-- [4] Mate distance pruning
    +-- [5] Transposition table probe
    +-- [6] Leaf node: delegate to quiescence search
    +-- [7] Static evaluation for pruning decisions
    |
    +-- Pruning (non-PV, non-check positions):
    |   +-- [8]  Razoring (depth <= 3, eval far below alpha)
    |   +-- [9]  Reverse futility pruning (depth <= 9, eval above beta)
    |   +-- [10] Probcut (depth >= 5, shallow search confirms cutoff)
    |   +-- [11] Null move pruning (R = 4 + depth/6, verify no zugzwang)
    |
    +-- [12] Internal iterative reduction (no TT move found)
    +-- [13] Singular extensions (TT move significantly better)
    |
    +-- Move loop (incremental pick_next_move):
        |
        +-- [14] Late move pruning (quiet moves beyond threshold)
        +-- [15] Futility pruning (depth <= 8, eval + margin < alpha)
        +-- [16] SEE pruning (depth-scaled thresholds):
        |       Captures: prune if SEE < -20 * depth
        |       Quiets:   prune if SEE < -50 * depth (depth <= 8)
        |
        +-- [17] Make move (push search path, update NNUE if active)
        |
        +-- Extensions (applied to effective depth):
        |   +-- Singular extension (+1 for TT move much better)
        |   +-- Recapture extension (+1 for same-square recapture)
        |   +-- Passed pawn push (+1 for pawns on 6th/7th rank)
        |
        +-- [18] PVS framework:
        |   +-- First move: full window (-beta, -alpha)
        |   +-- Later moves:
        |       +-- LMR (Late Move Reductions)
        |       |   - Pre-computed table (no transcendentals)
        |       |   - Applies to PV and non-PV nodes
        |       |   - Reduced for: PV nodes, killer moves, good history
        |       |   - Increased for: non-improving, SEE < 0 captures
        |       +-- Zero-window scout (-alpha-1, -alpha)
        |       +-- If scout fails high: re-search full window
        |
        +-- [19] Unmake move (pop search path)
        |
        +-- [20] Alpha/beta update + PV tracking:
        |       Copy child PV into parent (triangular PV table)
        +-- [21] Beta cutoff -> update history tables:
                +-- Killer moves (2 per ply)
                +-- History heuristic (gravity formula)
                +-- Countermove heuristic
                +-- Continuation history
                +-- Capture history
```

### Quiescence Search

```text
quiescence(alpha, beta, ply):
    |
    +-- Stand-pat evaluation (static eval as lower bound)
    +-- Delta pruning (prune if best capture cannot raise alpha)
    +-- TT probe
    |
    +-- If in check: search ALL moves (detect checkmate)
    +-- If not in check: search only captures + queen promotions
    |
    +-- For each capture (MVV-LVA + capture history order):
        +-- Delta pruning per-move (victim value + 200 < alpha)
        +-- SEE pruning (skip captures with SEE < 0)
        +-- Recursive quiescence call
        +-- Alpha/beta update
        +-- TT store on beta cutoff
```

---

## Neural Network Pipeline

### Model Architecture

```text
Input: Token sequence [seq_len] (encoded board positions)
                    |
                    v
            +---------------+
            |   Embedding   |  [4608, 512] lookup
            +-------+-------+
                    |
                    v
            +---------------+
            | + Pos Embed   |  [1, ctx_len, 512] learned positional
            +-------+-------+
                    |
                    v
        +----------------------------+
        |   TransformerBlock x 12    |  (repeated)
        |                            |
        |   +--------------------+   |
        |   | RMSNorm            |   |
        |   | MultiheadAttention |   |  8 heads, GQA (2 KV heads)
        |   |   (with RoPE)      |   |  head_dim = 64
        |   | LayerScale         |   |
        |   | + Residual         |   |
        |   +--------------------+   |
        |            |               |
        |   +--------------------+   |
        |   | RMSNorm            |   |
        |   | BitSwiGLU          |   |  gate/val/out (1.58-bit)
        |   |   hidden_dim=1536  |   |
        |   | LayerScale         |   |
        |   | + Residual         |   |
        |   +--------------------+   |
        +----------------------------+
                    |
                    v
            +---------------+
            |   ln_f        |  Final RMSNorm
            +---+-------+---+
                |       |
                v       v
        +--------+   +--------+
        | Policy |   | Value  |
        | Head   |   | Head   |
        | [4608] |   | [1]    |
        +--------+   +--------+
```

### BitNet 1.58-bit Weight Format

Each weight is ternary: {-1, 0, +1}, stored as dual bitmasks:

```text
Original i8 weights:  [+1, -1,  0, +1, -1, -1, +1,  0]

pos_bits (mask):       1    0   0   1    0    0   1   0   = 0x92
neg_bits (mask):       0    1   0   0    1    1   0   0   = 0x34

Reconstruction:  weight[i] = pos_bits[i] - neg_bits[i]

Memory: 2 bits per weight (4x denser than i8)
```

### SIMD Dispatch for Packed GEMV

```text
dispatch_packed_gemv(input, pos_bits, neg_bits):
    |
    +-- CPU supports AVX-512?
    |   YES --> avx512_packed_gemv()
    |           Load 2 bytes -> native __mmask16
    |           _mm512_mask_add_ps / _mm512_mask_sub_ps
    |           16 weights per cycle
    |
    +-- CPU supports AVX2?
    |   YES --> avx2_packed_gemv()
    |           1 byte -> set1_epi32 + and + cmpeq
    |           8 float lanes per cycle
    |
    +-- CPU supports NEON?
    |   YES --> neon_packed_gemv()
    |           4-lane bit expansion with vbslq_f32
    |
    +-- Fallback: scalar_packed_gemv()
            Bit extraction loop, 1 weight per cycle
```

### KV Cache (Incremental Inference)

```text
Full inference (first call):
    tokens = [t0, t1, t2, t3]
    Process all 4 tokens through all layers
    Cache K, V for each layer
    Return policy, value

Incremental inference (subsequent calls):
    tokens = [t0, t1, t2, t3, t4]  (t4 is new)
    Validate prefix [t0..t3] matches cache
    Process ONLY t4 through all layers
    Concatenate: K_cached | K_new, V_cached | V_new
    Attend over full sequence
    Return policy, value

    Speedup: 4-5x for single-token increments
```

---

## Evaluation System

### Evaluation Hierarchy

```text
+-----------------------------------------------+
|              Evaluation Stack                  |
|                                                |
|  [Root Only]  BitNet Transformer (optional)    |
|               - Policy logits (move ordering)  |
|               - Value score (unused in search) |
|               - Shared via Arc across threads  |
|               - Loaded from disk if present    |
|                                                |
|  [Every Node] NNUE Deep Network (primary)     |
|               - (768->512)x2 per perspective   |
|               - SCReLU activation              |
|               - L1: 1024->16 (i8 weights)     |
|               - CReLU activation               |
|               - L2: 16->8 (i16 weights)       |
|               - Output bucket by piece count   |
|               - AVX2 SIMD throughout           |
|               - Incrementally updated accum.   |
|               - Embedded in binary             |
|                                                |
|  [Fallback]   Material + PST + Positional      |
|               - Used when NNUE unavailable     |
+-----------------------------------------------+
```

### Material Values (centipawns)

```text
Pawn   = 100    Knight = 320    Bishop = 330
Rook   = 500    Queen  = 900    King   = 20000
```

### Pawn Structure Detection (Bitboard Operations)

```text
Passed Pawn (White pawn on e4, rank index 3):

    File mask (D+E+F):    Ranks ahead mask:       Combined:
    . . . 1 1 1 . .       1 1 1 1 1 1 1 1         . . . 1 1 1 . .  rank 8
    . . . 1 1 1 . .       1 1 1 1 1 1 1 1         . . . 1 1 1 . .  rank 7
    . . . 1 1 1 . .       1 1 1 1 1 1 1 1         . . . 1 1 1 . .  rank 6
    . . . 1 1 1 . .       1 1 1 1 1 1 1 1         . . . 1 1 1 . .  rank 5
    . . . 1 1 1 . .       . . . . . . . .         . . . . . . . .  rank 4 (pawn here)
    . . . 1 1 1 . .       . . . . . . . .         . . . . . . . .
    . . . 1 1 1 . .       . . . . . . . .         . . . . . . . .
    . . . 1 1 1 . .       . . . . . . . .         . . . . . . . .

    If (combined & enemy_pawns) == 0  -->  PASSED PAWN

    Implementation: file_and_adj & !((1 << ((rank+1) * 8)) - 1)
    Zero loops per pawn -- single bit-shift operation.
```

---

## Time Management

### Soft/Hard Bound Model

```text
                   Time axis (ms)
    0         optimum_ms              maximum_ms        remaining
    |============|=========================|===============|
    |  always    |  dynamic zone           |  hard abort   |
    |  search    |  (stability-dependent)  |  (never pass) |
    |            |                         |               |

    soft_stop = optimum * fallingEval * reduction * instability
    hard_stop = maximum (absolute cap)
```

### Dynamic Adjustment Factors

```text
+---------------------------+-------------------------------------------+
| Factor                    | Behavior                                  |
+---------------------------+-------------------------------------------+
| Best Move Stability       | Stable best move -> use less time         |
|                           | Sigmoid function of (depth - center)      |
+---------------------------+-------------------------------------------+
| Falling Eval              | Score dropping vs average -> use more time|
|                           | Clamped to [0.60, 1.70]                  |
+---------------------------+-------------------------------------------+
| Best Move Instability     | Frequent best-move changes -> more time  |
|                           | 1.0 + 1.5 * changes / 10.0              |
+---------------------------+-------------------------------------------+
| Ply-Aware Scaling         | Early game: more time per move           |
|                           | Endgame: less time needed                |
|                           | MTG estimate: 50 - 0.4 * ply            |
+---------------------------+-------------------------------------------+
```

### Time Allocation Flow

```text
go wtime=60000 btime=60000 winc=1000 binc=1000
    |
    v
TimeManager::new()
    |
    +-- Estimate moves to go: (50 - 0.4 * ply).clamp(20, 50)
    +-- Compute time pool:    time + inc * (mtg - 1) - overhead * mtg
    +-- Compute optimum:      ply-aware fraction of pool
    +-- Compute maximum:      5.5 * optimum, capped at 80% of remaining
    |
    v
ID Loop (each completed depth):
    |
    +-- TimeManager::update_iteration(depth, score, best_move_changed)
    |   +-- Update EMA of scores (70/30 blend)
    |   +-- Track best-move stability depth
    |   +-- Accumulate best-move change count
    |
    +-- TimeManager::should_stop_soft()
        +-- Compute: optimum * fallingEval * reduction * instability
        +-- Return: elapsed >= computed_limit
```

---

## Move Ordering

Move ordering is critical for alpha-beta pruning efficiency. Moves are scored but not fully sorted -- an incremental `pick_next_move` selects the best remaining move on demand, avoiding wasted sorting of moves pruned by beta cutoffs.

### Priority Scheme

```text
Priority    Category                    Score Range
--------    --------                    -----------
   1        TT Move (hash move)         +1,000,000
   2        NN Policy (root only)       +100,000 to +200,000
   3        Winning Captures            +50,000 to +60,000
            (MVV-LVA + Capture History)
   4        Queen Promotions            +45,000
   5        Killer Moves (2 per ply)    +40,000 / +39,000
   6        Countermove Heuristic       +38,000
   7        Quiet Moves                 History + Continuation History
            (history-ordered)           [-30,000 to +30,000]
   8        Losing Captures             -50,000
```

### History Gravity (Stockfish-style)

History updates use a gravity formula that prevents table saturation:

```text
update(entry, bonus):
    entry += bonus - |entry| * bonus / 16384

Effect:
    - Small values: nearly full bonus applied
    - Large values: bonus decays toward zero
    - Prevents runaway growth, maintains ordering signal
    - Applied uniformly to: history, continuation history, capture history
```

---

## Transposition Table

### Lock-Free Design (Hyatt/Mann XOR Trick)

```text
Cache Line (16 bytes per entry):
+------------------+------------------+
|  key XOR data    |      data        |   2 x AtomicU64
|   (8 bytes)      |   (8 bytes)      |
+------------------+------------------+

Store(key, data):
    slot[0] = key ^ data      // XOR trick: corrupted reads are detectable
    slot[1] = data

Load(key):
    xor_val = slot[0]
    data    = slot[1]
    if (xor_val ^ data) == key:
        return Some(unpack(data))   // Valid entry
    else:
        return None                 // Collision or race -- safe to ignore
```

### Entry Packing

```text
data (64 bits):
+--------+--------+--------+--------+--------+--------+--------+--------+
| score  | depth  | flag   |      best_move (16 bits)     | unused       |
| (i16)  | (u8)   | (u2)   |                              |              |
+--------+--------+--------+--------+--------+--------+--------+--------+

flag: 0 = EXACT, 1 = LOWER_BOUND (beta cutoff), 2 = UPPER_BOUND (fail low)
```

### Replacement Policy

```text
if incoming.hash == existing.hash:
    ALWAYS replace (fresher data for same position)
else:
    replace only if incoming.depth >= existing.depth  (depth-preferred)
```

---

## Threading Model

### Lazy SMP

```text
+-----------------------------------------------------------+
|                    Shared Resources                        |
|                                                           |
|   +-------------------+    +---------------------------+  |
|   | Transposition     |    | Stop Flag (AtomicBool)    |  |
|   | Table (AtomicU64) |    +---------------------------+  |
|   +-------------------+    +---------------------------+  |
|                            | TimeManager (atomics)     |  |
|                            +---------------------------+  |
+-----------------------------------------------------------+
         |              |              |              |
         v              v              v              v
    +---------+    +---------+    +---------+    +---------+
    | Worker 0|    | Worker 1|    | Worker 2|    | Worker 3|
    | depth=D |    |depth=D-2|    |depth=D-3|    |depth=D-1|
    |  (main) |    |         |    |         |    |         |
    +---------+    +---------+    +---------+    +---------+
    | Own:     |   | Own:     |   | Own:     |   | Own:     |
    | - Board  |   | - Board  |   | - Board  |   | - Board  |
    | - Killers|   | - Killers|   | - Killers|   | - Killers|
    | - History|   | - History|   | - History|   | - History|
    | - Eval   |   | - Eval   |   | - Eval   |   | - Eval   |
    | - Path   |   | - Path   |   | - Path   |   | - Path   |
    +---------+    +---------+    +---------+    +---------+

    Workers share TT entries implicitly (no synchronization needed).
    Varied depth offsets (1 + i%3) cause natural divergence in subtrees.
    Main thread (Worker 0) controls the iterative deepening loop.
    Result: deepest completed search across all workers is reported.
```

### Thread Communication

```text
main thread                        worker threads
     |                                  |
     +-- spawn workers ----------------->
     |                                  |
     +-- [blocking] wait on rx <--------+-- send (depth, score, bestmove) via mpsc
     |                                  |
     +-- set stop_flag = true --------->+-- check stop_flag periodically
     |                                  |
     +-- join all threads               |
     |                                  v
     +-- select best result         (threads exit)
```

---

## Opening Book

### Polyglot Format

```text
+-------------------------------------------+
|           Polyglot Book Probe             |
|                                           |
|  1. Compute Polyglot Zobrist hash         |
|     (781-key Random64 array, NOT          |
|      the engine's internal hash)          |
|                                           |
|  2. Binary search in sorted book entries  |
|     Each entry: [hash:u64, move:u16,      |
|                  weight:u16, learn:u32]    |
|                                           |
|  3. Collect all matching entries           |
|  4. Weighted random selection             |
|  5. Convert Polyglot move encoding to     |
|     engine's internal ChessMove           |
+-------------------------------------------+

Constraints:
    - Dual-book support (book1.bin, book2.bin)
    - Limited to first 30 plies (MAX_BOOK_PLY)
    - Fully loaded into memory at startup
    - Zero I/O during search
    - Disabled in Analysis Mode
```

---

## UCI Protocol Layer

### Command Processing Flow

```text
stdin
  |
  v
+------------------+
| UciHandler       |
| parse command     |
+--------+---------+
         |
         +-- "uci"        --> send id, options, "uciok"
         +-- "isready"    --> "readyok"
         +-- "ucinewgame" --> clear TT, reset KV cache
         +-- "position"   --> set board, record position_history
         +-- "go"         --> parse time controls:
         |                    wtime, btime, winc, binc,
         |                    movestogo, depth, movetime, infinite
         |                    |
         |                    v
         |               +------------------+
         |               | Book probe       |
         |               +--------+---------+
         |                        |
         |                 miss   v
         |               +------------------+
         |               | search_async()   |
         |               | (non-blocking)   |
         |               +------------------+
         |                        |
         |                        v
         |               "bestmove e2e4"
         |
         +-- "stop"       --> set stop_flag, collect result
         +-- "setoption"  --> Hash, Threads, Move Overhead, etc.
         +-- "quit"       --> exit
```

### Position History for Repetition Detection

```text
"position startpos moves e2e4 e7e5 g1f3 b8c6"

UciHandler maintains:
    position_history: Vec<u64> = [
        hash_after_e2e4,
        hash_after_e7e5,
        hash_after_g1f3,
        hash_after_b8c6,
    ]

Passed to search_async() -> each SearchWorker receives a clone.
Used in is_repetition() to detect 2-fold repetition:
    - Search path: checks every-other position (same side to move)
    - Game history: checks last 100 positions
```

---

## Module Map

```text
KiyEngine/
|
+-- src/
|   +-- main.rs                 Entry point (UCI loop)
|   +-- lib.rs                  Library exports
|   +-- constants.rs            Centralized constants (engine, NN, search)
|   |
|   +-- book/
|   |   +-- mod.rs              Polyglot book probing, Zobrist hashing, embedded fallback
|   |   +-- polyglot_keys.rs    781-key Random64 array
|   |   +-- book1.bin           Polyglot book 1 (embedded via include_bytes!)
|   |   +-- book2.bin           Polyglot book 2 (embedded via include_bytes!)
|   |
|   +-- engine/                 BitNet Transformer inference (optional)
|   |   +-- mod.rs              Engine struct, forward / forward_with_cache
|   |   +-- bitlinear.rs        BitLinear: packed ternary weights, SIMD GEMV
|   |   +-- transformer.rs      MultiheadAttention, BitSwiGLU, KVCache
|   |   +-- heads.rs            PolicyHead, ValueHead (pre-transposed)
|   |   +-- gguf.rs             GGUF format parser and mmap loader
|   |
|   +-- eval/                   Evaluation tables
|   |   +-- pst.rs              PeSTO piece-square tables
|   |
|   +-- search/                 Search algorithms
|   |   +-- mod.rs              Alpha-beta, TT, move ordering, quiescence
|   |   +-- eval.rs             Static evaluation (NNUE primary, Material+PST fallback)
|   |   +-- time.rs             TimeManager (soft/hard bounds, dynamic factors)
|   |   +-- lazy_smp.rs         Lazy SMP thread management
|   |   +-- syzygy.rs           Syzygy tablebase probing (WDL + DTZ)
|   |
|   +-- nnue/                   NNUE evaluation (active, embedded)
|   |   +-- mod.rs              Network struct, quantized inference, AVX2 SIMD
|   |   +-- accumulator.rs      Per-perspective accumulators, fused SIMD updates
|   |   +-- features.rs         HalfKP feature extraction, delta computation
|   |
|   +-- simd/                   SIMD utilities and intrinsics
|   |
|   +-- uci/                    UCI protocol handler
|       +-- mod.rs              Command parsing, option management,
|                               position history, game state
|
+-- Cargo.toml                  Package manifest (v6.1.0)
+-- .cargo/config.toml          SIMD target features (AVX2/AVX-512/BMI2)
+-- kiyengine.gguf              BitNet Transformer model (~79 MB, optional)
+-- kiyengine.nnue              NNUE weights (~7.6 MB, embedded in binary)
```

---

## Performance Characteristics

### Benchmark Summary (v6.1.0)

```text
Self-play (10s+0.1s, single-thread):
    - Max depth: 26-64 per game
    - NPS: ~900K single-thread, ~1.6M (4 threads)
    - NNUE eval active at every node
    - Decisive games with proper time control

NNUE SIMD Performance:
    - Scalar deep eval: ~337K NPS (4T)
    - AVX2 deep eval:   ~1.6M NPS (4T)
    - Speedup: ~4.7x from hand-tuned AVX2
```

### Key Optimization Techniques

```text
+-------------------------------+----------------------------------+
| Technique                     | Impact                           |
+-------------------------------+----------------------------------+
| Static Exchange Evaluation    | Prunes losing captures/quiets    |
| Triangular PV table           | Full PV lines in UCI output      |
| Recapture extensions          | +1 depth on same-square recaps   |
| Passed pawn push extensions   | +1 depth for 6th/7th rank pushes |
| Varied Lazy SMP offsets       | Better exploration diversity     |
| LMR on PV nodes              | Node reduction at deep depths    |
| History gravity (Stockfish)   | Better move ordering convergence |
| Incremental move picking      | Avoids sorting pruned moves      |
| Fixed-size search path        | Zero-alloc repetition detection  |
| NNUE AVX2 deep eval           | 4.7x speedup vs scalar           |
| Fused accumulator updates     | 50% fewer memory passes          |
| Embedded NNUE + books         | Single binary, zero I/O at start |
| Pre-computed LMR table        | No transcendentals in hot path   |
| Lock-free TT (XOR trick)     | Zero mutex contention in SMP     |
+-------------------------------+----------------------------------+
```

---

## References

- [Chess Programming Wiki](https://www.chessprogramming.org/)
- [Stockfish Source](https://github.com/official-stockfish/Stockfish)
- [BitNet 1.58-bit Paper](https://arxiv.org/abs/2402.12354)
- [GGUF Specification](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/README.md)
- [Hyatt/Mann XOR Trick](https://www.chessprogramming.org/Shared_Hash_Table)

---

Copyright 2026 Khoi. Licensed under the Apache License 2.0.

(Updated for v6.1.0)
