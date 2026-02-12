use crate::engine::{Engine, KVCache};
use crate::nnue::accumulator::{Accumulator, FinnyTable};
use crate::nnue::features;
use crate::nnue::NnueNetwork;
use crate::uci::MoveCodec;
use chess::{BitBoard, Board, ChessMove, Color, MoveGen, Piece, Square};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};

// =============================================================================
// CONSTANTS & TYPES
// =============================================================================

const MATE_SCORE: i32 = 30_000;
const INFINITY: i32 = 32_000;
const MAX_PLY: usize = 128;
const MAX_DEPTH: u8 = 64;

/// Maximum number of legal moves in any chess position (theoretical max is 218).
const MAX_MOVES: usize = 256;

/// Stack-allocated scored move list — eliminates per-node heap allocation.
/// This is THE critical optimization for NPS: no malloc/free per search node.
struct MoveList {
    moves: [(ChessMove, i32); MAX_MOVES],
    len: usize,
}

impl MoveList {
    #[inline]
    fn new() -> Self {
        Self {
            // Safety: ChessMove is Copy and zero-init is fine for scored pairs
            moves: [(ChessMove::default(), 0i32); MAX_MOVES],
            len: 0,
        }
    }

    #[inline]
    fn push(&mut self, mv: ChessMove, score: i32) {
        debug_assert!(self.len < MAX_MOVES);
        unsafe {
            *self.moves.get_unchecked_mut(self.len) = (mv, score);
        }
        self.len += 1;
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Incremental move picker: swap highest-scored move to position `idx`.
    #[inline]
    fn pick_move(&mut self, idx: usize) {
        let mut best_idx = idx;
        let mut best_score = self.moves[idx].1;
        for i in (idx + 1)..self.len {
            if self.moves[i].1 > best_score {
                best_score = self.moves[i].1;
                best_idx = i;
            }
        }
        if best_idx != idx {
            self.moves.swap(idx, best_idx);
        }
    }

    #[inline]
    fn get(&self, idx: usize) -> (ChessMove, i32) {
        self.moves[idx]
    }
}

/// Sentinel value meaning "no static eval stored in TT"
const NONE_EVAL: i32 = i32::MIN;

/// Correction history table size (power of 2 for fast modulo)
const CORRECTION_HIST_SIZE: usize = 16384;
/// Correction history weight limit (Stockfish-style gravity)
const CORRECTION_HIST_LIMIT: i32 = 1024;
/// Correction history grain (scaling factor for stored corrections)
const CORRECTION_GRAIN: i32 = 256;

// Pre-computed LMR reduction table: lmr_table[depth][moves_searched]
// Avoids calling f32::ln() in the hot path.
const LMR_MAX_DEPTH: usize = 64;
const LMR_MAX_MOVES: usize = 64;
static LMR_TABLE: std::sync::LazyLock<[[u8; LMR_MAX_MOVES]; LMR_MAX_DEPTH]> =
    std::sync::LazyLock::new(|| {
        let mut table = [[0u8; LMR_MAX_MOVES]; LMR_MAX_DEPTH];
        for d in 1..LMR_MAX_DEPTH {
            for m in 1..LMR_MAX_MOVES {
                let df = d as f64;
                let mf = m as f64;
                table[d][m] = (0.77 + (df.ln() * mf.ln() / 2.25)) as u8;
            }
        }
        table
    });
// File mask constants for pawn attack computation in SEE
const FILE_A_BB: u64 = 0x0101_0101_0101_0101;
const FILE_H_BB: u64 = 0x8080_8080_8080_8080;

// Điểm số từ Value Head thường là float [-1.0, 1.0].
// Ta nhân với scale này để ra Centipawn.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TTFlag {
    Exact,
    LowerBound,
    UpperBound,
}

#[derive(Clone, Debug, Copy)]
pub struct TTEntry {
    pub hash: u64,
    pub depth: u8,
    pub score: i32,
    pub flag: TTFlag,
    pub best_move: Option<ChessMove>,
    /// Cached static eval from NNUE (avoids recomputation on TT hit)
    pub static_eval: i32,
}

// =============================================================================
// LOCK-FREE 3-WAY BUCKETED TRANSPOSITION TABLE (Hyatt/Mann XOR trick)
// =============================================================================
//
// Each bucket has 3 entries (48 bytes total, fits in a cache line).
// Each entry is 16 bytes: two AtomicU64 values (key ^ data, data).
//
// On read:  scan all 3 entries, return first hash match.
// On write: find best replacement target using age+depth scoring.
//
// Data packing (64 bits):
//   bits  0-5:  from square (6 bits)
//   bits  6-11: to square (6 bits)
//   bits 12-15: promotion (4 bits: 0=none,1=N,2=B,3=R,4=Q)
//   bits 16-31: score (i16)
//   bits 32-39: depth (u8)
//   bits 40-41: flag (2 bits: 0=Exact,1=Lower,2=Upper)
//   bits 42-57: static eval (i16)
//   bits 58-62: generation (5 bits, 0-31 wrapping)
//   bit  63:    always set (sentinel — distinguishes from empty)

const TT_WAYS: usize = 3;

pub struct TranspositionTable {
    // Layout: [key0, data0, key1, data1, key2, data2, ...] per bucket
    // 3 entries × 2 u64 = 6 AtomicU64 per bucket
    slots: Vec<AtomicU64>,
    #[allow(dead_code)]
    num_buckets: usize,
    mask: usize,
    generation: AtomicU64,
}

impl TranspositionTable {
    pub fn new(size_mb: usize) -> Self {
        let size_mb = size_mb.max(64);
        let bytes = size_mb * 1024 * 1024;
        // Each bucket = 3 entries × 16 bytes = 48 bytes
        let num_buckets = (bytes / (TT_WAYS * 16)).next_power_of_two();
        let slots = (0..num_buckets * TT_WAYS * 2)
            .map(|_| AtomicU64::new(0))
            .collect();
        Self {
            slots,
            num_buckets,
            mask: num_buckets - 1,
            generation: AtomicU64::new(0),
        }
    }

    /// Increment generation counter — call at the start of each new search.
    pub fn new_search(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    pub fn resize(&self, _mb: usize) {
        self.clear();
    }

    #[inline(always)]
    pub fn get(&self, hash: u64) -> Option<TTEntry> {
        let bucket = (hash as usize) & self.mask;
        let base = bucket * TT_WAYS * 2;
        // Scan all 3 entries in the bucket
        for i in 0..TT_WAYS {
            let slot = base + i * 2;
            let d = self.slots[slot + 1].load(Ordering::Relaxed);
            if d == 0 {
                continue;
            }
            let k = self.slots[slot].load(Ordering::Relaxed);
            if k ^ d == hash {
                return Self::unpack(hash, d);
            }
        }
        None
    }

    #[inline(always)]
    pub fn store(&self, entry: TTEntry) {
        let bucket = (entry.hash as usize) & self.mask;
        let base = bucket * TT_WAYS * 2;
        let tt_gen = (self.generation.load(Ordering::Relaxed) & 0x1F) as u8;
        let new_data = Self::pack(&entry, tt_gen);

        // Find the best replacement target among the 3 entries:
        // Priority: 1) same hash (always replace), 2) empty slot, 3) worst score
        // Replacement score: lower = more replaceable
        //   score = 4 * depth - 4 * gen_age (stale entries are cheap to replace)
        let mut worst_idx = 0usize;
        let mut worst_score = i32::MAX;

        for i in 0..TT_WAYS {
            let slot = base + i * 2;
            let d = self.slots[slot + 1].load(Ordering::Relaxed);

            // Empty slot — use immediately
            if d == 0 {
                self.slots[slot].store(entry.hash ^ new_data, Ordering::Relaxed);
                self.slots[slot + 1].store(new_data, Ordering::Relaxed);
                return;
            }

            let k = self.slots[slot].load(Ordering::Relaxed);
            let old_hash = k ^ d;

            // Same position — always replace
            if old_hash == entry.hash {
                self.slots[slot].store(entry.hash ^ new_data, Ordering::Relaxed);
                self.slots[slot + 1].store(new_data, Ordering::Relaxed);
                return;
            }

            // Compute replacement score: depth favored, but stale entries penalized
            let old_depth = ((d >> 32) & 0xFF) as i32;
            let old_gen = ((d >> 58) & 0x1F) as i32;
            let gen_age = ((tt_gen as i32) - old_gen + 32) & 0x1F; // wrapping distance
            let score = 4 * old_depth - 4 * gen_age;
            if score < worst_score {
                worst_score = score;
                worst_idx = i;
            }
        }

        // Replace the worst entry
        let slot = base + worst_idx * 2;
        self.slots[slot].store(entry.hash ^ new_data, Ordering::Relaxed);
        self.slots[slot + 1].store(new_data, Ordering::Relaxed);
    }

    pub fn clear(&self) {
        for slot in self.slots.iter() {
            slot.store(0, Ordering::Relaxed);
        }
        self.generation.store(0, Ordering::Relaxed);
    }

    fn pack(entry: &TTEntry, tt_gen: u8) -> u64 {
        let mut d: u64 = 0;
        if let Some(mv) = entry.best_move {
            let from = mv.get_source().to_int() as u64;
            let to = mv.get_dest().to_int() as u64;
            let promo: u64 = match mv.get_promotion() {
                Some(chess::Piece::Knight) => 1,
                Some(chess::Piece::Bishop) => 2,
                Some(chess::Piece::Rook) => 3,
                Some(chess::Piece::Queen) => 4,
                _ => 0,
            };
            d |= from | (to << 6) | (promo << 12);
        }
        let score_u16 = entry.score.clamp(-32768, 32767) as i16 as u16;
        d |= (score_u16 as u64) << 16;
        d |= (entry.depth as u64) << 32;
        let flag_bits: u64 = match entry.flag {
            TTFlag::Exact => 0,
            TTFlag::LowerBound => 1,
            TTFlag::UpperBound => 2,
        };
        d |= flag_bits << 40;
        // Pack static eval into bits 42-57 (16 bits, i16)
        let eval_u16 = entry.static_eval.clamp(-32768, 32767) as i16 as u16;
        d |= (eval_u16 as u64) << 42;
        // Pack generation into bits 58-62 (5 bits)
        d |= ((tt_gen & 0x1F) as u64) << 58;
        d |= 1u64 << 63; // sentinel: never zero
        d
    }

    fn unpack(hash: u64, d: u64) -> Option<TTEntry> {
        if d == 0 {
            return None;
        }
        let from = (d & 0x3F) as u8;
        let to = ((d >> 6) & 0x3F) as u8;
        let promo = ((d >> 12) & 0xF) as u8;
        let best_move = if from == to && promo == 0 {
            None
        } else {
            let from_sq = unsafe { chess::Square::new(from) };
            let to_sq = unsafe { chess::Square::new(to) };
            let promotion = match promo {
                1 => Some(chess::Piece::Knight),
                2 => Some(chess::Piece::Bishop),
                3 => Some(chess::Piece::Rook),
                4 => Some(chess::Piece::Queen),
                _ => None,
            };
            Some(ChessMove::new(from_sq, to_sq, promotion))
        };
        let score = ((d >> 16) & 0xFFFF) as u16 as i16 as i32;
        let depth = ((d >> 32) & 0xFF) as u8;
        let flag = match (d >> 40) & 0x3 {
            0 => TTFlag::Exact,
            1 => TTFlag::LowerBound,
            _ => TTFlag::UpperBound,
        };
        // Extract static eval from bits 42-57 (i16)
        let static_eval = ((d >> 42) & 0xFFFF) as u16 as i16 as i32;

        Some(TTEntry {
            hash,
            depth,
            score,
            flag,
            best_move,
            static_eval,
        })
    }
}

// Module eval local (fallback)
pub mod eval;
pub mod lazy_smp;
pub mod syzygy;
pub mod time;
pub mod tt;
use self::syzygy::{SyzygyTB, TbWdl};
use self::time::TimeManager;

// =============================================================================
// SEARCHER (MANAGER)
// =============================================================================

pub struct Searcher {
    engine: Option<Arc<Engine>>,
    tt: Arc<TranspositionTable>,
    pub threads: usize,
    pub move_overhead: u32,
    pub nnue: Option<Arc<NnueNetwork>>,
    pub syzygy: Option<Arc<SyzygyTB>>,
}

impl Searcher {
    pub fn new(engine: Option<Arc<Engine>>, tt: Arc<TranspositionTable>) -> Self {
        Self {
            engine,
            tt,
            threads: 4,
            move_overhead: 60, // Default 60ms overhead
            nnue: None,
            syzygy: None,
        }
    }

    pub fn search_async(
        &self,
        board: Board,
        move_history: Vec<usize>,
        max_depth: u8,
        wtime: u32,
        btime: u32,
        winc: u32,
        binc: u32,
        movestogo: u32,
        stop_flag: Arc<AtomicBool>,
        tx: mpsc::Sender<String>,
        kv_cache: Option<Arc<parking_lot::Mutex<KVCache>>>,
        position_history: Vec<u64>,
    ) {
        let engine = self.engine.clone();
        let tt = Arc::clone(&self.tt);
        let num_threads = self.threads;
        let move_overhead = self.move_overhead;
        let is_white = board.side_to_move() == Color::White;
        let nnue = self.nnue.clone();
        let syzygy = self.syzygy.clone();

        let ply = move_history.len() as u32;
        let pos_hist = Arc::new(position_history);

        // Increment TT generation so stale entries get replaced
        tt.new_search();

        std::thread::spawn(move || {
            let time_manager = Arc::new(TimeManager::new(
                wtime,
                btime,
                winc,
                binc,
                movestogo,
                move_overhead,
                is_white,
                ply,
            ));

            // Pre-compute NN policy ONCE, then share across all threads
            // Use KV cache if available for incremental inference
            // Apply temperature-scaled softmax (LLM technique) to convert raw logits
            // to proper probabilities. T<1.0 sharpens, T>1.0 flattens the distribution.
            const POLICY_TEMPERATURE: f32 = 0.8;
            let shared_policy: Arc<Option<(Vec<f32>, f32)>> = {
                // Skip BitNet when playing Black — model has White-side bias.
                // NNUE + classical eval handles Black positions reliably.
                let result = if is_white {
                    if let Some(ref eng) = engine {
                        if let Some(ref cache) = kv_cache {
                            let mut cache_guard = cache.lock();
                            eng.forward_with_cache(&move_history, &mut cache_guard)
                        } else {
                            eng.forward(&move_history)
                        }
                    } else {
                        Err(candle_core::Error::Msg("No BitNet engine loaded".to_string()))
                    }
                } else {
                    Err(candle_core::Error::Msg("BitNet skipped for Black side".to_string()))
                };
                if let Ok(res) = result {
                    if let Ok(mut pol_vec) = res.0.to_vec1::<f32>() {
                        // Temperature-scaled softmax: softmax(logit / T)
                        // Scale logits by inverse temperature
                        let inv_t = 1.0 / POLICY_TEMPERATURE;
                        for v in pol_vec.iter_mut() {
                            *v *= inv_t;
                        }
                        // Numerically stable softmax: subtract max, exponentiate, normalize
                        let max_val = pol_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut sum = 0.0f32;
                        for v in pol_vec.iter_mut() {
                            *v = (*v - max_val).exp();
                            sum += *v;
                        }
                        if sum > 0.0 {
                            let inv_sum = 1.0 / sum;
                            for v in pol_vec.iter_mut() {
                                *v *= inv_sum;
                            }
                        }
                        Arc::new(Some((pol_vec, res.1)))
                    } else {
                        Arc::new(None)
                    }
                } else {
                    Arc::new(None)
                }
            };

            let mut handles = vec![];
            let global_nodes = Arc::new(AtomicU64::new(0));

            for i in 0..num_threads {
                let tt_clone = Arc::clone(&tt);
                let board_clone = board.clone();
                let stop_clone = Arc::clone(&stop_flag);
                let tm_clone = Arc::clone(&time_manager);
                let tx_clone = if i == 0 { Some(tx.clone()) } else { None };
                let is_main = i == 0;
                let policy_clone = Arc::clone(&shared_policy);
                let nnue_clone = nnue.clone();
                let syzygy_clone = syzygy.clone();
                let pos_hist_clone = Arc::clone(&pos_hist);
                let nodes_clone = Arc::clone(&global_nodes);

                handles.push(
                    std::thread::Builder::new()
                        .stack_size(64 * 1024 * 1024) // 64MB stack for deep recursion
                        .spawn(move || {
                            let mut worker = SearchWorker::with_shared_nodes(
                                tt_clone,
                                board_clone,
                                stop_clone,
                                nnue_clone,
                                syzygy_clone,
                                nodes_clone,
                            );
                            worker.set_game_history((*pos_hist_clone).clone());
                            let mut best_move = None;
                            let mut last_score = 0;
                            let mut prev_best: Option<ChessMove> = None;
                            let start_time = std::time::Instant::now();

                            // Use shared pre-computed NN policy (no per-thread forward pass)
                            worker.set_cached_policy(&*policy_clone);

                            // Initialize NNUE accumulator at root (ply 0)
                            worker.nnue_refresh(0);

                            // --- Root Syzygy TB probe ---
                            // If position is in TB range, play the best DTZ move immediately
                            if is_main {
                                if let Some((tb_move, tb_score)) = worker.probe_root_tb() {
                                    let _ = tx_clone.as_ref().unwrap().send(format!(
                                        "info depth 1 score cp {} nodes 0 nps 0 tbhits {} pv {}",
                                        tb_score, worker.tb_hits, tb_move
                                    ));
                                    let _ = tx_clone.unwrap().send(format!("bestmove {}", tb_move));
                                    worker.stop_flag.store(true, Ordering::Relaxed);
                                    return;
                                }
                            }

                            // Wire time manager for in-search abort checks
                            let depth_only = max_depth > 0 && wtime == 0 && btime == 0;
                            if !depth_only {
                                worker.time_manager = Some(Arc::clone(&tm_clone));
                            }

                            // --- Single legal move optimization ---
                            // If only one legal move, play it immediately (don't waste time)
                            let legal_count = MoveGen::new_legal(&board_clone).len();
                            let single_move_limit: u8 =
                                if legal_count == 1 { 1 } else { MAX_DEPTH };

                            // Lazy SMP: main thread full depth, helpers depth-1
                            let effective_max = if max_depth == 0 { MAX_DEPTH } else { max_depth };
                            let effective_max = effective_max.min(single_move_limit);
                            // Lazy SMP: main thread full depth, helpers use varied offsets
                            // for exploration diversity (Stockfish-inspired)
                            let local_max_depth = if is_main {
                                effective_max
                            } else {
                                let offset = 1 + (i % 3) as u8; // 1, 2, or 3
                                effective_max.saturating_sub(offset).max(1)
                            };

                            const MIN_DEPTH: u8 = 3;
                            // Mate verification: when mate score appears, force N more iterations
                            let mut mate_verify_until: u8 = 0;

                            for depth in 1..=local_max_depth {
                                // Reset search path for clean repetition detection
                                worker.search_path_len = 0;

                                // Age history tables between iterations
                                if depth > 1 {
                                    worker.age_history();
                                }

                                // Time abort checks (only after minimum depth)
                                if worker.stop_flag.load(Ordering::Relaxed) {
                                    break;
                                }
                                if !depth_only && depth > MIN_DEPTH && depth > mate_verify_until {
                                    if is_main {
                                        if tm_clone.should_stop_soft() {
                                            break;
                                        }
                                    } else if tm_clone.should_stop() {
                                        break;
                                    }
                                }

                                // --- Aspiration Windows with exponential widening ---
                                let mut delta: i32 = 12;
                                let mut alpha = if depth >= 5 {
                                    last_score - delta
                                } else {
                                    -INFINITY
                                };
                                let mut beta = if depth >= 5 {
                                    last_score + delta
                                } else {
                                    INFINITY
                                };

                                loop {
                                    let (score, mv) = worker.alpha_beta(alpha, beta, depth, 0);

                                    if worker.stop_flag.load(Ordering::Relaxed) {
                                        break;
                                    }
                                    if !depth_only && depth > MIN_DEPTH && tm_clone.should_stop() {
                                        break;
                                    }

                                    if score <= alpha {
                                        // Fail low: widen downward, keep beta
                                        beta = (alpha + beta) / 2;
                                        alpha = (score - delta).max(-INFINITY);
                                        delta += delta / 3 + 5;
                                    } else if score >= beta {
                                        // Fail high: widen upward
                                        beta = (score + delta).min(INFINITY);
                                        delta += delta / 3 + 5;
                                    } else {
                                        // Score within window — iteration complete
                                        let best_move_changed = mv != prev_best;
                                        if is_main {
                                            prev_best = mv;
                                            best_move = mv;
                                            last_score = score;

                                            // Mate verification: if mate score first appears,
                                            // verify with 2 more iterations (set once, don't escalate)
                                            if score.abs() >= MATE_SCORE - 100 && mate_verify_until == 0 {
                                                mate_verify_until = depth + 2;
                                            }

                                            // Update time manager with iteration results
                                            tm_clone.update_iteration(
                                                depth as u32,
                                                score,
                                                best_move_changed,
                                            );

                                            let elapsed =
                                                start_time.elapsed().as_secs_f64().max(0.001);
                                            // Use shared counter for accurate multi-thread NPS
                                            let total_nodes =
                                                worker.shared_nodes.load(Ordering::Relaxed)
                                                    + (worker.nodes & 4095); // add unflushed remainder
                                            let nps = (total_nodes as f64 / elapsed) as u64;

                                            // Extract full PV from triangular PV table
                                            let pv_str = {
                                                let pv_end = worker.pv_len[0].min(MAX_PLY);
                                                let mut parts = Vec::with_capacity(pv_end);
                                                for i in 0..pv_end {
                                                    if let Some(m) = worker.pv_table[0][i] {
                                                        parts.push(format!("{}", m));
                                                    } else {
                                                        break;
                                                    }
                                                }
                                                if parts.is_empty() {
                                                    if let Some(m) = mv {
                                                        format!("{}", m)
                                                    } else {
                                                        String::new()
                                                    }
                                                } else {
                                                    parts.join(" ")
                                                }
                                            };
                                            let tb_str = if worker.tb_hits > 0 {
                                                format!(" tbhits {}", worker.tb_hits)
                                            } else {
                                                String::new()
                                            };
                                            if let Some(ref sender) = tx_clone {
                                                let _ = sender.send(format!(
                                                    "info depth {} score cp {} nodes {} nps {}{} pv {}",
                                                    depth, score, worker.nodes, nps, tb_str, pv_str
                                                ));
                                            }
                                        }
                                        break;
                                    }
                                    // Safety: if window is already full, break
                                    if alpha <= -INFINITY && beta >= INFINITY {
                                        break;
                                    }
                                }
                            }

                            // Gửi kết quả cuối cùng (chỉ main thread)
                            if is_main {
                                if let Some(ref sender) = tx_clone {
                                    if let Some(mv) = best_move {
                                        let _ = sender.send(format!("bestmove {}", mv));
                                    } else {
                                        let mut moves = MoveGen::new_legal(&board_clone);
                                        if let Some(mv) = moves.next() {
                                            let _ = sender.send(format!("bestmove {}", mv));
                                        } else {
                                            let _ = sender.send("bestmove (none)".to_string());
                                        }
                                    }
                                }
                            }
                        })
                        .expect("failed to spawn search thread"),
                );
            }

            for handle in handles {
                let _ = handle.join();
            }
        });
    }
}

// =============================================================================
// SEARCH WORKER (CORE LOGIC)
// =============================================================================

pub struct SearchWorker {
    tt: Arc<TranspositionTable>,
    board: Board,
    stop_flag: Arc<AtomicBool>,
    time_manager: Option<Arc<TimeManager>>,
    pub nodes: u64,
    /// Shared atomic node counter across all SMP workers for accurate NPS
    shared_nodes: Arc<AtomicU64>,
    killers: [[Option<ChessMove>; 2]; MAX_PLY],
    history: [[[i32; 64]; 64]; 2],
    // Countermove heuristic: indexed by [prev_from][prev_to]
    countermove: [[Option<ChessMove>; 64]; 64],
    // Continuation history: indexed by [prev_piece][prev_to][piece][to]
    // Piece index: 0-5 for White P/N/B/R/Q/K, 6-11 for Black P/N/B/R/Q/K
    cont_history: Box<[[[[i32; 64]; 12]; 64]; 12]>,
    // Static eval stack for "improving" detection (eval at each ply)
    eval_stack: [i32; MAX_PLY],
    // Previous move at each ply (for countermove + continuation history)
    move_stack: [Option<ChessMove>; MAX_PLY],
    // Piece that moved at each ply
    piece_stack: [u8; MAX_PLY],
    // Capture history: indexed by [moved_piece_idx][to_sq][captured_piece_idx]
    // Piece index: P=0 N=1 B=2 R=3 Q=4 K=5
    capture_history: Box<[[[i32; 6]; 64]; 6]>,
    // Cached NN results — computed once, reused across all ID iterations
    cached_root_policy: Option<Vec<f32>>,
    cached_root_value: f32,
    // NNUE evaluation (optional — falls back to Material+PST if None)
    nnue: Option<Arc<NnueNetwork>>,
    nnue_acc: Vec<Accumulator>,
    // Finny Table: per-(king_sq, perspective) accumulator cache for fast refresh
    finny_table: Option<FinnyTable>,
    // Whether to actually use NNUE for eval (false = skip accumulator updates)
    use_nnue_eval: bool,
    // Correction histories (Stockfish-inspired): learn per-position eval corrections
    // Indexed by hash % CORRECTION_HIST_SIZE, stores [White correction, Black correction]
    pawn_correction: Box<[[i32; 2]; CORRECTION_HIST_SIZE]>,
    material_correction: Box<[[i32; 2]; CORRECTION_HIST_SIZE]>,
    // Repetition detection: game position hashes from UCI position command
    game_hash_history: Vec<u64>,
    // Search path hashes for 2-fold repetition during search (fixed-size, zero-alloc)
    search_path: [u64; MAX_PLY],
    search_path_len: usize,
    // Triangular PV table: pv_table[ply][i] = i-th move in the PV starting at ply
    pv_table: Box<[[Option<ChessMove>; MAX_PLY]; MAX_PLY]>,
    pv_len: [usize; MAX_PLY],
    // Syzygy tablebase (optional)
    syzygy: Option<Arc<SyzygyTB>>,
    // TB hit counter
    pub tb_hits: u64,
}

impl SearchWorker {
    pub fn new(
        tt: Arc<TranspositionTable>,
        board: Board,
        stop_flag: Arc<AtomicBool>,
        nnue: Option<Arc<NnueNetwork>>,
        syzygy: Option<Arc<SyzygyTB>>,
    ) -> Self {
        Self::with_shared_nodes(
            tt,
            board,
            stop_flag,
            nnue,
            syzygy,
            Arc::new(AtomicU64::new(0)),
        )
    }

    pub fn with_shared_nodes(
        tt: Arc<TranspositionTable>,
        board: Board,
        stop_flag: Arc<AtomicBool>,
        nnue: Option<Arc<NnueNetwork>>,
        syzygy: Option<Arc<SyzygyTB>>,
        shared_nodes: Arc<AtomicU64>,
    ) -> Self {
        let nnue_acc = if let Some(ref net) = nnue {
            (0..MAX_PLY)
                .map(|_| Accumulator::new(net.hidden_size))
                .collect()
        } else {
            Vec::new()
        };
        let finny_table = nnue.as_ref().map(|net| FinnyTable::new(net));
        Self {
            tt,
            board,
            stop_flag,
            time_manager: None,
            nodes: 0,
            shared_nodes,
            killers: [[None; 2]; MAX_PLY],
            history: [[[0; 64]; 64]; 2],
            countermove: [[None; 64]; 64],
            cont_history: Box::new([[[[0i32; 64]; 12]; 64]; 12]),
            capture_history: Box::new([[[0i32; 6]; 64]; 6]),
            eval_stack: [0; MAX_PLY],
            move_stack: [None; MAX_PLY],
            piece_stack: [0; MAX_PLY],
            cached_root_policy: None,
            cached_root_value: 0.0,
            use_nnue_eval: nnue.is_some(),
            nnue,
            nnue_acc,
            finny_table,
            pawn_correction: Box::new([[0i32; 2]; CORRECTION_HIST_SIZE]),
            material_correction: Box::new([[0i32; 2]; CORRECTION_HIST_SIZE]),
            game_hash_history: Vec::new(),
            search_path: [0u64; MAX_PLY],
            search_path_len: 0,
            pv_table: Box::new([[None; MAX_PLY]; MAX_PLY]),
            pv_len: [0; MAX_PLY],
            syzygy,
            tb_hits: 0,
        }
    }

    /// Set game position history for repetition detection.
    fn set_game_history(&mut self, history: Vec<u64>) {
        self.game_hash_history = history;
    }

    /// Check for repetition: 2-fold in search path OR position seen in game history.
    /// Only checks every-other position (same side to move) for efficiency.
    #[inline]
    fn is_repetition(&self) -> bool {
        let hash = self.board.get_hash();
        let len = self.search_path_len;
        // Check search path in reverse, skip 1 (same side to move repeats at ply-2, ply-4, ...)
        let mut i = len.wrapping_sub(2);
        while i < len {
            if self.search_path[i] == hash {
                return true;
            }
            i = i.wrapping_sub(2);
        }
        // Check game history (last 100 positions max for performance)
        let start = self.game_hash_history.len().saturating_sub(100);
        for &h in &self.game_hash_history[start..] {
            if h == hash {
                return true;
            }
        }
        false
    }

    #[inline]
    fn search_path_push(&mut self, hash: u64) {
        if self.search_path_len < MAX_PLY {
            self.search_path[self.search_path_len] = hash;
            self.search_path_len += 1;
        }
    }

    #[inline]
    fn search_path_pop(&mut self) {
        self.search_path_len = self.search_path_len.saturating_sub(1);
    }

    /// Age history tables: halve all scores to prevent stale values from dominating.
    /// Called between iterative deepening iterations.
    fn age_history(&mut self) {
        for side in 0..2 {
            for from in 0..64 {
                for to in 0..64 {
                    self.history[side][from][to] = self.history[side][from][to] * 3 / 4;
                }
            }
        }
        for p in 0..6 {
            for sq in 0..64 {
                for v in 0..6 {
                    self.capture_history[p][sq][v] = self.capture_history[p][sq][v] * 3 / 4;
                }
            }
        }
        // Age continuation history to prevent stale values from dominating
        for pp in 0..12 {
            for pt in 0..64 {
                for cp in 0..12 {
                    for ct in 0..64 {
                        self.cont_history[pp][pt][cp][ct] =
                            self.cont_history[pp][pt][cp][ct] * 3 / 4;
                    }
                }
            }
        }
    }

    // =========================================================================
    // Set pre-computed NN policy from shared Arc (computed once in search_async).
    // =========================================================================
    fn set_cached_policy(&mut self, shared: &Option<(Vec<f32>, f32)>) {
        if let Some((ref pol_vec, value)) = *shared {
            self.cached_root_policy = Some(pol_vec.clone());
            self.cached_root_value = value;
        }
    }

    // =========================================================================
    // ROOT SYZYGY TB PROBE — find best move by WDL then DTZ at root.
    // =========================================================================
    fn probe_root_tb(&mut self) -> Option<(ChessMove, i32)> {
        let tb = self.syzygy.as_ref()?;
        if !tb.can_probe(&self.board) {
            return None;
        }

        let mut best_move: Option<ChessMove> = None;
        let mut best_rank: i32 = -1;
        let mut best_dtz_key: i32 = i32::MIN;

        let moves = MoveGen::new_legal(&self.board);
        for mv in moves {
            let new_board = self.board.make_move_new(mv);
            // Probe WDL from opponent's perspective, negate to get ours
            let child_wdl = match tb.probe_wdl(&new_board) {
                Some(w) => w,
                None => continue,
            };
            let our_wdl = child_wdl.negate();
            let rank = our_wdl.rank();

            // DTZ for tiebreaking within same WDL
            let child_dtz = tb.probe_dtz(&new_board).map(|r| r.dtz).unwrap_or(0);
            let dtz_key = match our_wdl {
                TbWdl::Win | TbWdl::CursedWin => -child_dtz.abs(), // prefer faster win
                TbWdl::Loss | TbWdl::BlessedLoss => child_dtz.abs(), // prefer slower loss
                TbWdl::Draw => 0,
            };

            if rank > best_rank || (rank == best_rank && dtz_key > best_dtz_key) {
                best_move = Some(mv);
                best_rank = rank;
                best_dtz_key = dtz_key;
            }
        }

        if let Some(mv) = best_move {
            self.tb_hits += 1;
            let wdl = match best_rank {
                4 => TbWdl::Win,
                3 => TbWdl::CursedWin,
                2 => TbWdl::Draw,
                1 => TbWdl::BlessedLoss,
                _ => TbWdl::Loss,
            };
            let score = syzygy::wdl_to_score(wdl, 0);
            Some((mv, score))
        } else {
            None
        }
    }

    // =========================================================================
    // MATERIAL BALANCE — pure piece counting, no positional terms.
    // Used to correct NNUE material blindness.
    // =========================================================================
    #[inline]
    fn material_balance(&self) -> i32 {
        let mut score: i32 = 0;
        let white = *self.board.color_combined(Color::White);
        let black = *self.board.color_combined(Color::Black);
        let pieces = [
            (Piece::Pawn, 100),
            (Piece::Knight, 320),
            (Piece::Bishop, 330),
            (Piece::Rook, 500),
            (Piece::Queen, 900),
        ];
        for &(piece, val) in &pieces {
            let w = (self.board.pieces(piece) & white).0.count_ones() as i32;
            let b = (self.board.pieces(piece) & black).0.count_ones() as i32;
            score += (w - b) * val;
        }
        if self.board.side_to_move() == Color::White {
            score
        } else {
            -score
        }
    }

    // =========================================================================
    // RAW NNUE EVAL — network output + material correction.
    // =========================================================================
    #[inline(always)]
    fn raw_nnue_eval(&mut self, ply: usize) -> i32 {
        if self.use_nnue_eval {
            if let Some(ref net) = self.nnue {
                let idx = ply.min(MAX_PLY - 1);
                if !self.nnue_acc[idx].computed {
                    // Use Finny Table for fast refresh (diff-based, ~8× faster)
                    if let Some(ref mut ft) = self.finny_table {
                        ft.refresh(&mut self.nnue_acc[idx], &self.board, net);
                    } else {
                        self.nnue_acc[idx].refresh(&self.board, net);
                    }
                }
                let nnue_score =
                    net.evaluate(&self.nnue_acc[idx], self.board.side_to_move(), &self.board);
                // Material correction: NNUE is material-blind, so blend with
                // fast material balance.  Speed > positional accuracy here
                // because deeper search catches more tactics.
                let material = self.material_balance();
                // Sanity check: if NNUE and material diverge wildly,
                // trust material more to prevent eval hallucinations
                let divergence = (nnue_score - material).abs();
                if divergence > 300 {
                    return (nnue_score + 9 * material) / 10;
                }
                // Normal blend: 75% material + 25% NNUE positional
                return (nnue_score + 3 * material) / 4;
            }
        }
        eval::evaluate(&self.board, 0)
    }

    // =========================================================================
    // ADJUSTED STATIC EVAL — Stockfish-inspired eval corrections.
    //
    // Applies:
    // 1. Tempo bonus (small bonus for having the move)
    // 2. Correction histories (pawn structure + material config learning)
    // 3. Material scaling (reduce eval in drawish low-material positions)
    // =========================================================================
    #[inline(always)]
    fn fast_eval(&mut self, ply: usize) -> i32 {
        let raw = self.raw_nnue_eval(ply);
        self.adjust_eval(raw)
    }

    /// Apply Stockfish-style eval adjustments to a raw NNUE score.
    #[inline(always)]
    fn adjust_eval(&self, raw: i32) -> i32 {
        let mut eval = raw;

        // 1. Tempo bonus: small advantage for having the move (~15cp in Stockfish)
        eval += 16;

        // 2. Pawn correction history
        let pawn_hash = self.board.get_pawn_hash();
        let pawn_idx = (pawn_hash as usize) % CORRECTION_HIST_SIZE;
        let stm = if self.board.side_to_move() == Color::White {
            0
        } else {
            1
        };
        let pawn_corr = self.pawn_correction[pawn_idx][stm];

        // 3. Material correction history
        let mat_hash = self.material_hash();
        let mat_idx = (mat_hash as usize) % CORRECTION_HIST_SIZE;
        let mat_corr = self.material_correction[mat_idx][stm];

        // Apply corrections (scaled by CORRECTION_GRAIN)
        let correction = (pawn_corr + mat_corr) / CORRECTION_GRAIN;
        eval += correction;

        // 4. Material scaling: reduce eval magnitude when total material is low
        // This helps avoid overconfident evals in drawish endgames
        let total_pieces = self.board.combined().0.count_ones() as i32; // 2-32
                                                                        // Scale: full eval at 32 pieces, 60% at minimum (2 kings)
                                                                        // scale_factor = 600 + 400 * (pieces - 2) / 30, in millipawns
        let scale = (600 + 13 * (total_pieces - 2)).min(1000);
        eval = eval * scale / 1000;

        eval
    }

    /// King safety correction: penalize uncastled king that has moved to rank 2-3
    /// in the opening/middlegame. Scaled by opponent's attacking material.
    #[inline]
    fn king_safety_correction(&self) -> i32 {
        let stm = self.board.side_to_move();
        let opp = !stm;
        let our_king = self.board.king_square(stm);
        let opp_king = self.board.king_square(opp);

        // Game phase: higher = more pieces = more dangerous for exposed king
        let opp_material = {
            let opp_bb = *self.board.color_combined(opp);
            let n = (self.board.pieces(Piece::Knight) & opp_bb).0.count_ones() as i32;
            let b = (self.board.pieces(Piece::Bishop) & opp_bb).0.count_ones() as i32;
            let r = (self.board.pieces(Piece::Rook) & opp_bb).0.count_ones() as i32;
            let q = (self.board.pieces(Piece::Queen) & opp_bb).0.count_ones() as i32;
            n * 1 + b * 1 + r * 2 + q * 4 // phase-like weight
        };

        // Only apply when opponent has significant material (midgame)
        if opp_material < 4 {
            return 0;
        }

        let mut penalty = 0i32;

        // Our king exposure
        penalty -= Self::king_exposure_penalty(our_king, stm, opp_material);
        // Opponent king exposure (bonus for us)
        penalty += Self::king_exposure_penalty(opp_king, opp, opp_material) / 2;

        penalty
    }

    /// Calculate penalty for a single king based on rank and castling status.
    /// Returns a positive value (penalty).
    #[inline]
    fn king_exposure_penalty(king_sq: Square, color: Color, opp_material: i32) -> i32 {
        let rank = king_sq.to_index() / 8; // 0=rank1, 7=rank8
        let file = king_sq.to_index() % 8;

        // Effective rank from this color's perspective (0 = back rank)
        let eff_rank = if color == Color::White { rank } else { 7 - rank };

        // King on back rank in corner area = likely castled = safe
        if eff_rank == 0 && (file <= 2 || file >= 5) {
            return 0;
        }

        // Penalty for king advanced beyond rank 1 (not castled)
        // Scaled by opponent's attacking material
        let rank_penalty = match eff_rank {
            0 => {
                // Back rank but in center (d1/e1) = hasn't castled
                if file >= 3 && file <= 4 { 15 } else { 0 }
            }
            1 => 30, // Rank 2: moderate danger (Kd2, Ke2)
            2 => 60, // Rank 3: serious danger (Kd3 in midgame = suicidal)
            3 => 80, // Rank 4+: extreme
            _ => 100,
        };

        // Scale penalty by opponent material (more pieces = more dangerous)
        (rank_penalty * opp_material) / 8
    }

    /// Pawn chain correction: bonus for pawns defending each other diagonally.
    /// From side-to-move perspective.
    #[inline]
    fn pawn_chain_correction(&self) -> i32 {
        let white_bb = *self.board.color_combined(Color::White);
        let black_bb = *self.board.color_combined(Color::Black);
        let white_pawns = (self.board.pieces(Piece::Pawn) & white_bb).0;
        let black_pawns = (self.board.pieces(Piece::Pawn) & black_bb).0;

        // White pawn chains: pawn on (file, rank) defended by pawn on (file±1, rank-1)
        // Shift white pawns NE and NW to find defenders
        let w_defend_ne = (white_pawns >> 9) & 0xFEFEFEFEFEFEFEFE; // shift down-left
        let w_defend_nw = (white_pawns >> 7) & 0x7F7F7F7F7F7F7F7F; // shift down-right
        let w_defended = white_pawns & (w_defend_ne | w_defend_nw);
        let w_chain = w_defended.count_ones() as i32;

        // Black pawn chains: pawn defended by pawn on (file±1, rank+1)
        let b_defend_se = (black_pawns << 7) & 0xFEFEFEFEFEFEFEFE; // shift up-left
        let b_defend_sw = (black_pawns << 9) & 0x7F7F7F7F7F7F7F7F; // shift up-right
        let b_defended = black_pawns & (b_defend_se | b_defend_sw);
        let b_chain = b_defended.count_ones() as i32;

        let chain_score = (w_chain - b_chain) * 8; // 8cp per connected pawn

        if self.board.side_to_move() == Color::White {
            chain_score
        } else {
            -chain_score
        }
    }

    /// Compute a simple material configuration hash for correction history.
    /// Combines piece counts into a hash value.
    #[inline(always)]
    fn material_hash(&self) -> u64 {
        let white = *self.board.color_combined(Color::White);
        let black = *self.board.color_combined(Color::Black);
        let mut h: u64 = 0;
        // Unrolled for zero-overhead piece counting
        let wp = (*self.board.pieces(Piece::Pawn) & white).0.count_ones() as u64;
        let bp = (*self.board.pieces(Piece::Pawn) & black).0.count_ones() as u64;
        h = h.wrapping_mul(13).wrapping_add(wp);
        h = h.wrapping_mul(13).wrapping_add(bp);
        let wn = (*self.board.pieces(Piece::Knight) & white).0.count_ones() as u64;
        let bn = (*self.board.pieces(Piece::Knight) & black).0.count_ones() as u64;
        h = h.wrapping_mul(13).wrapping_add(wn);
        h = h.wrapping_mul(13).wrapping_add(bn);
        let wb = (*self.board.pieces(Piece::Bishop) & white).0.count_ones() as u64;
        let bb = (*self.board.pieces(Piece::Bishop) & black).0.count_ones() as u64;
        h = h.wrapping_mul(13).wrapping_add(wb);
        h = h.wrapping_mul(13).wrapping_add(bb);
        let wr = (*self.board.pieces(Piece::Rook) & white).0.count_ones() as u64;
        let br = (*self.board.pieces(Piece::Rook) & black).0.count_ones() as u64;
        h = h.wrapping_mul(13).wrapping_add(wr);
        h = h.wrapping_mul(13).wrapping_add(br);
        let wq = (*self.board.pieces(Piece::Queen) & white).0.count_ones() as u64;
        let bq = (*self.board.pieces(Piece::Queen) & black).0.count_ones() as u64;
        h = h.wrapping_mul(13).wrapping_add(wq);
        h = h.wrapping_mul(13).wrapping_add(bq);
        h
    }

    /// Update correction histories after a search completes at a node.
    /// Called when we have both a static eval and a search result (best_score).
    /// The difference between search result and static eval teaches the correction.
    #[inline]
    fn update_correction_histories(&mut self, static_eval: i32, search_score: i32, depth: u8) {
        // Don't update for mate scores
        if search_score.abs() >= MATE_SCORE - 100 {
            return;
        }
        if static_eval.abs() >= MATE_SCORE - 100 || static_eval == NONE_EVAL {
            return;
        }

        let diff = search_score - static_eval;
        let scaled_diff = diff * CORRECTION_GRAIN;
        let weight = (depth as i32).min(16);
        let stm = if self.board.side_to_move() == Color::White {
            0
        } else {
            1
        };

        // Pawn correction
        let pawn_hash = self.board.get_pawn_hash();
        let pawn_idx = (pawn_hash as usize) % CORRECTION_HIST_SIZE;
        let pc = &mut self.pawn_correction[pawn_idx][stm];
        // Stockfish-style gravity: new_val = (old * (limit - weight) + diff * weight) / limit
        *pc = ((*pc as i64 * (CORRECTION_HIST_LIMIT - weight) as i64
            + scaled_diff as i64 * weight as i64)
            / CORRECTION_HIST_LIMIT as i64) as i32;
        *pc = (*pc).clamp(-CORRECTION_GRAIN * 32, CORRECTION_GRAIN * 32);

        // Material correction
        let mat_hash = self.material_hash();
        let mat_idx = (mat_hash as usize) % CORRECTION_HIST_SIZE;
        let mc = &mut self.material_correction[mat_idx][stm];
        *mc = ((*mc as i64 * (CORRECTION_HIST_LIMIT - weight) as i64
            + scaled_diff as i64 * weight as i64)
            / CORRECTION_HIST_LIMIT as i64) as i32;
        *mc = (*mc).clamp(-CORRECTION_GRAIN * 32, CORRECTION_GRAIN * 32);
    }

    // =========================================================================
    // NNUE ACCUMULATOR MANAGEMENT
    // =========================================================================

    /// Refresh NNUE accumulator at given ply from the current board.
    pub fn nnue_refresh(&mut self, ply: usize) {
        if let Some(ref net) = self.nnue {
            let idx = ply.min(MAX_PLY - 1);
            if let Some(ref mut ft) = self.finny_table {
                ft.refresh(&mut self.nnue_acc[idx], &self.board, net);
            } else {
                self.nnue_acc[idx].refresh(&self.board, net);
            }
        }
    }

    /// Incrementally update NNUE accumulator at ply+1 from ply after a move.
    /// Falls back to marking child as uncomputed (lazy refresh will handle it).
    ///
    /// For king-bucketed networks: king moves trigger a full refresh (bucket changes).
    fn nnue_make_move(&mut self, mv: ChessMove, ply: usize) {
        if !self.use_nnue_eval {
            return;
        }
        if let Some(ref net) = self.nnue {
            let parent_idx = ply.min(MAX_PLY - 1);
            let child_idx = (ply + 1).min(MAX_PLY - 1);

            if parent_idx >= child_idx {
                self.nnue_acc[child_idx].computed = false;
                return;
            }

            if self.nnue_acc[parent_idx].computed {
                // Use bucketed or legacy delta computation based on network type
                let delta = if net.has_king_buckets() {
                    let wk = self.board.king_square(chess::Color::White).to_index();
                    let bk = self.board.king_square(chess::Color::Black).to_index();
                    features::compute_delta_bucketed(&self.board, mv, wk, bk)
                } else {
                    features::compute_delta(&self.board, mv)
                };

                if let Some(ref d) = delta {
                    // For king-bucketed networks, king moves need full refresh
                    if net.has_king_buckets() && (d.white_king_moved || d.black_king_moved) {
                        // We can't do incremental update because the bucket changed.
                        // Mark as uncomputed — lazy refresh will handle it after board.make_move.
                        self.nnue_acc[child_idx].computed = false;
                        return;
                    }

                    let (left, right) = self.nnue_acc.split_at_mut(child_idx);
                    let parent = &left[parent_idx];
                    let child = &mut right[0];
                    child.update_from(parent, d, net);
                    return;
                }
            }
            self.nnue_acc[child_idx].computed = false;
        }
    }

    /// Copy NNUE accumulator for null move (no pieces change, just side-to-move).
    fn nnue_null_move(&mut self, ply: usize) {
        if !self.use_nnue_eval {
            return;
        }
        if self.nnue.is_some() {
            let parent_idx = ply.min(MAX_PLY - 1);
            let child_idx = (ply + 1).min(MAX_PLY - 1);

            if parent_idx < child_idx && self.nnue_acc[parent_idx].computed {
                let (left, right) = self.nnue_acc.split_at_mut(child_idx);
                right[0].white = left[parent_idx].white;
                right[0].black = left[parent_idx].black;
                right[0].psqt_white = left[parent_idx].psqt_white;
                right[0].psqt_black = left[parent_idx].psqt_black;
                right[0].computed = true;
            } else {
                self.nnue_acc[child_idx].computed = false;
            }
        }
    }

    // =========================================================================
    // HELPER: check if a move is a capture
    // =========================================================================
    #[inline]
    fn is_capture(&self, mv: ChessMove) -> bool {
        let targets = *self.board.color_combined(!self.board.side_to_move());
        (targets & BitBoard::from_square(mv.get_dest())).0 != 0
    }

    // =========================================================================
    // HELPER: piece index for continuation history (0-11)
    // White P=0,N=1,B=2,R=3,Q=4,K=5  Black P=6,N=7,B=8,R=9,Q=10,K=11
    // =========================================================================
    #[inline]
    fn piece_index(piece: Piece, color: Color) -> u8 {
        let base = match piece {
            Piece::Pawn => 0,
            Piece::Knight => 1,
            Piece::Bishop => 2,
            Piece::Rook => 3,
            Piece::Queen => 4,
            Piece::King => 5,
        };
        if color == Color::Black {
            base + 6
        } else {
            base
        }
    }

    /// Piece index 0-5 (color-independent) for capture history
    #[inline]
    fn cap_piece_index(piece: Piece) -> usize {
        match piece {
            Piece::Pawn => 0,
            Piece::Knight => 1,
            Piece::Bishop => 2,
            Piece::Rook => 3,
            Piece::Queen => 4,
            Piece::King => 5,
        }
    }

    // =========================================================================
    // HELPER: has non-pawn material (for null move safety)
    // =========================================================================
    #[inline]
    fn has_non_pawn_material(&self) -> bool {
        let us = self.board.side_to_move();
        let our_pieces = self.board.color_combined(us);
        let pawns = self.board.pieces(Piece::Pawn);
        let king_bb = BitBoard::from_square(self.board.king_square(us));
        let non_pawn = our_pieces & !pawns & !king_bb;
        non_pawn.0 != 0
    }

    // =========================================================================
    // STATIC EXCHANGE EVALUATION (SEE)
    // Determines if a capture sequence is winning, losing, or equal.
    // Foundation for SEE pruning, qsearch pruning, and LMR adjustments.
    // =========================================================================

    /// Compute all attackers (both colors) to `sq` given current `occupied` bitboard.
    /// Recomputes sliding attacks to handle x-ray through removed pieces.
    fn all_attackers_to(&self, sq: Square, occupied: BitBoard) -> BitBoard {
        let knights = chess::get_knight_moves(sq) & *self.board.pieces(Piece::Knight);
        let king = chess::get_king_moves(sq) & *self.board.pieces(Piece::King);
        let bishops_queens = chess::get_bishop_moves(sq, occupied)
            & (*self.board.pieces(Piece::Bishop) | *self.board.pieces(Piece::Queen));
        let rooks_queens = chess::get_rook_moves(sq, occupied)
            & (*self.board.pieces(Piece::Rook) | *self.board.pieces(Piece::Queen));

        // Pawn attackers: reverse the attack direction to find pawns that attack sq
        let sq_val = BitBoard::from_square(sq).0;
        let white_pawns =
            *self.board.pieces(Piece::Pawn) & *self.board.color_combined(Color::White);
        let black_pawns =
            *self.board.pieces(Piece::Pawn) & *self.board.color_combined(Color::Black);
        let wp =
            BitBoard(((sq_val >> 7) & !FILE_A_BB) | ((sq_val >> 9) & !FILE_H_BB)) & white_pawns;
        let bp =
            BitBoard(((sq_val << 7) & !FILE_H_BB) | ((sq_val << 9) & !FILE_A_BB)) & black_pawns;

        (knights | king | bishops_queens | rooks_queens | wp | bp) & occupied
    }

    /// Static Exchange Evaluation — returns material gain/loss from the exchange.
    /// Positive = good for the moving side.
    fn see_score(&self, mv: ChessMove) -> i32 {
        let from = mv.get_source();
        let to = mv.get_dest();

        let mut gain = [0i32; 33];
        let mut d = 0usize;

        let target = self.board.piece_on(to);
        let attacker = self.board.piece_on(from).unwrap_or(Piece::Pawn);

        // Initial gain: value of captured piece (+ promotion bonus if applicable)
        gain[0] = if let Some(promo) = mv.get_promotion() {
            target.map_or(0, see_val) + see_val(promo) - see_val(Piece::Pawn)
        } else {
            target.map_or(0, see_val)
        };

        let mut occupied = *self.board.combined() ^ BitBoard::from_square(from);
        let mut current_piece = if let Some(promo) = mv.get_promotion() {
            promo
        } else {
            attacker
        };
        let mut side = !self.board.side_to_move();

        loop {
            d += 1;
            if d >= 32 {
                break;
            }
            gain[d] = see_val(current_piece) - gain[d - 1];

            // Stand pat pruning: if even capturing current piece can't help
            if (-gain[d - 1]).max(gain[d]) < 0 {
                break;
            }

            // Find all attackers of `to` with current occupancy (handles x-ray)
            let all_att = self.all_attackers_to(to, occupied);
            let side_att = all_att & *self.board.color_combined(side) & occupied;

            if side_att.0 == 0 {
                break;
            }

            // Find cheapest attacker for this side
            let mut found = false;
            for &piece in &[
                Piece::Pawn,
                Piece::Knight,
                Piece::Bishop,
                Piece::Rook,
                Piece::Queen,
                Piece::King,
            ] {
                let pieces = *self.board.pieces(piece) & side_att;
                if pieces.0 != 0 {
                    // King can't capture if opponent still has attackers
                    if piece == Piece::King {
                        let opp_att = all_att & *self.board.color_combined(!side) & occupied;
                        if opp_att.0 != 0 {
                            break;
                        }
                    }

                    let sq_idx = pieces.0.trailing_zeros() as u8;
                    let sq = unsafe { Square::new(sq_idx) };
                    current_piece = piece;
                    occupied ^= BitBoard::from_square(sq);
                    found = true;
                    break;
                }
            }

            if !found {
                break;
            }
            side = !side;
        }

        // Negamax walk-back
        while d > 0 {
            gain[d - 1] = -((-gain[d - 1]).max(gain[d]));
            d -= 1;
        }

        gain[0]
    }

    /// Public wrapper for alpha-beta search from root (used by self-play binary).
    pub fn alpha_beta_root(&mut self, depth: u8) -> (i32, Option<ChessMove>) {
        self.nodes = 0;
        self.alpha_beta(-INFINITY, INFINITY, depth, 0)
    }

    // =========================================================================
    // ALPHA-BETA with Stockfish-inspired pruning techniques
    // =========================================================================
    fn alpha_beta(
        &mut self,
        mut alpha: i32,
        mut beta: i32,
        mut depth: u8,
        ply: usize,
    ) -> (i32, Option<ChessMove>) {
        // --- Step 0: MAX_PLY guard (prevent stack overflow) ---
        if ply >= MAX_PLY {
            return (self.fast_eval(ply), None);
        }

        // Initialize PV length for this ply (no PV moves yet)
        self.pv_len[ply] = ply;

        // --- Step 1: Abort check (+ periodic time check every 2048 nodes) ---
        if self.nodes & 8191 == 0 {
            if let Some(ref tm) = self.time_manager {
                if tm.should_stop() {
                    self.stop_flag.store(true, Ordering::Relaxed);
                }
            }
        }
        if self.stop_flag.load(Ordering::Relaxed) {
            return (0, None);
        }

        let is_pv = beta - alpha > 1;
        let is_root = ply == 0;

        // --- Step 1b: Repetition detection ---
        if !is_root && self.is_repetition() {
            return (0, None); // Draw by repetition
        }

        // --- Step 1c: Syzygy tablebase probe ---
        // At non-root nodes, if piece count is within TB range, get exact WDL score.
        // At root, we still search to find the best move but use TB to bound scores.
        if !is_root {
            if let Some(ref tb) = self.syzygy {
                if tb.can_probe(&self.board) {
                    if let Some(wdl) = tb.probe_wdl(&self.board) {
                        self.tb_hits += 1;
                        let tb_score = syzygy::wdl_to_score(wdl, ply);
                        return (tb_score, None);
                    }
                }
            }
        }

        // --- Step 2: Detect check (extension handled at horizon and per-move) ---
        let in_check = self.board.checkers().0 != 0;
        // NOTE: unconditional depth+=1 here was removed — it caused exponential
        // search explosion when combined with per-move gives_check extension.
        // Check extensions are now handled by:
        //   (a) Horizon: depth==0 && in_check → depth=1  (line ~1503)
        //   (b) Per-move: gives_check && (depth>6 || is_pv) → ext=1  (line ~1833)

        let ply_idx = ply.min(MAX_PLY - 1);

        // --- Step 3: Mate Distance Pruning ---
        if !is_root {
            let mating_score = MATE_SCORE - ply as i32;
            let mated_score = -MATE_SCORE + ply as i32;
            if alpha >= mating_score {
                return (alpha, None);
            }
            if beta <= mated_score {
                return (beta, None);
            }
            alpha = alpha.max(mated_score);
            beta = beta.min(mating_score);
            if alpha >= beta {
                return (alpha, None);
            }
        }

        // --- Step 4: TT Lookup ---
        let hash = self.board.get_hash();
        let tt_entry = self.tt.get(hash);
        let tt_move = tt_entry.as_ref().and_then(|e| e.best_move).filter(|m| {
            // Validate TT move is legal (guards against Zobrist hash collisions)
            self.board.legal(*m)
        });
        let tt_score = tt_entry.as_ref().map(|e| e.score);
        let tt_depth = tt_entry.as_ref().map(|e| e.depth).unwrap_or(0);
        let tt_static_eval = tt_entry.as_ref().map(|e| e.static_eval);

        if !is_pv {
            if let Some(ref entry) = tt_entry {
                if entry.depth >= depth {
                    match entry.flag {
                        TTFlag::Exact => return (entry.score, entry.best_move),
                        TTFlag::LowerBound => {
                            if entry.score >= beta {
                                return (entry.score, entry.best_move);
                            }
                        }
                        TTFlag::UpperBound => {
                            if entry.score <= alpha {
                                return (entry.score, entry.best_move);
                            }
                        }
                    }
                }
            }
        }

        // --- Step 5: Leaf node → Quiescence ---
        // When in check at the horizon, extend by 1 instead of dropping into
        // qsearch.  This gives the engine a full alpha-beta ply to find quiet
        // escape moves (blocks, interpositions) that qsearch would miss.
        if depth == 0 {
            if in_check {
                depth = 1;
            } else {
                return (self.quiescence(alpha, beta, ply, 0), None);
            }
        }
        self.nodes += 1;
        // Periodically flush to shared counter (every 4096 nodes, low contention)
        if self.nodes & 4095 == 0 {
            self.shared_nodes.fetch_add(4096, Ordering::Relaxed);
        }

        // --- Step 6: Static Evaluation for pruning ---
        // Use TT cached static eval if available (avoids redundant NNUE computation)
        let static_eval = if in_check {
            -INFINITY
        } else if let Some(tt_eval) = tt_static_eval {
            if tt_eval != NONE_EVAL as i32 {
                tt_eval
            } else {
                self.fast_eval(ply)
            }
        } else {
            self.fast_eval(ply)
        };

        // Use TT score to refine static eval if applicable (Stockfish technique)
        // If TT score is more accurate than static eval, use it for pruning decisions
        let eval_for_pruning = if !in_check {
            if let Some(tt_sc) = tt_score {
                if let Some(ref entry) = tt_entry {
                    match entry.flag {
                        TTFlag::Exact => tt_sc,
                        TTFlag::LowerBound if tt_sc > static_eval => tt_sc,
                        TTFlag::UpperBound if tt_sc < static_eval => tt_sc,
                        _ => static_eval,
                    }
                } else {
                    static_eval
                }
            } else {
                static_eval
            }
        } else {
            static_eval
        };

        // Store eval in stack for improving detection
        self.eval_stack[ply_idx] = eval_for_pruning;

        // --- Step 6b: Improving flag ---
        // Position is "improving" if static eval is better than 2 plies ago
        let improving =
            !in_check && ply >= 2 && eval_for_pruning > self.eval_stack[ply_idx.saturating_sub(2)];

        // --- Step 7: Razoring (Stockfish-style) ---
        if !is_pv && !in_check && depth <= 3 {
            let razor_margin = 300 + 250 * (depth as i32);
            if static_eval + razor_margin < alpha {
                let q_score = self.quiescence(alpha, beta, ply, 0);
                if q_score < alpha {
                    return (q_score, None);
                }
            }
        }

        // --- Step 8: Reverse Futility Pruning (Static Null Move Pruning) ---
        // Improving flag: tighter margin when position is improving
        if !is_pv
            && !in_check
            && depth <= 9
            && eval_for_pruning - (if improving { 70 } else { 100 }) * (depth as i32) >= beta
            && eval_for_pruning < MATE_SCORE - 100
        {
            return (eval_for_pruning, None);
        }

        // --- Step 8b: Probcut ---
        // If a shallow search with a raised beta finds a cutoff, prune
        let probcut_beta = beta + 200;
        if !is_pv
            && !in_check
            && depth >= 5
            && eval_for_pruning >= probcut_beta - 100
            && eval_for_pruning < MATE_SCORE - 100
        {
            let pc_depth = depth.saturating_sub(4);
            let (val, _) = self.alpha_beta(probcut_beta - 1, probcut_beta, pc_depth, ply);
            if val >= probcut_beta {
                return (val, None);
            }
        }

        // --- Step 9: Real Null Move Pruning ---
        // Actually pass the move and search with reduced depth
        if !is_pv
            && !in_check
            && depth >= 3
            && eval_for_pruning >= beta
            && self.has_non_pawn_material()
        {
            if let Some(null_board) = self.board.null_move() {
                let old_board = self.board;
                self.search_path_push(self.board.get_hash());
                self.board = null_board;
                self.nnue_null_move(ply);

                // Adaptive reduction: R = 4 + depth/6, capped (Stockfish-like)
                let r = 4 + (depth / 6).min(3);
                let null_depth = depth.saturating_sub(r);
                let (val, _) = self.alpha_beta(-beta, -beta + 1, null_depth, ply + 1);
                let null_score = -val;

                self.board = old_board;
                self.search_path_pop();

                if self.stop_flag.load(Ordering::Relaxed) {
                    return (0, None);
                }

                if null_score >= beta {
                    // Don't return unproven mate scores
                    if null_score >= MATE_SCORE - 100 {
                        return (beta, None);
                    }
                    return (null_score, None);
                }
            }
        }

        // --- Step 10: Internal Iterative Reduction (IIR) ---
        // Non-PV expected cut nodes: reduce by 2 (Stockfish-inspired)
        // PV nodes: reduce by 1
        if tt_move.is_none() && !in_check {
            if !is_pv && depth >= 4 {
                depth -= 2;
            } else if depth >= 4 {
                depth -= 1;
            }
        }

        // --- Step 11: Neural Policy Lookup (DISABLED) ---
        // BitNet policy is currently harmful — suggests flashy but tactically
        // hollow moves (Bd5?, Nce2?, Ne5?). Disabled until the policy network
        // is retrained on quality data. Search heuristics (TT, MVV-LVA, killers,
        // history, countermove, cont_history) provide reliable move ordering.
        let policy: Option<&[f32]> = None;

        // --- Step 12: Generate and order moves ---
        let mut moves = self.order_moves(MoveGen::new_legal(&self.board), policy, ply, tt_move);

        if moves.is_empty() {
            return if in_check {
                (-MATE_SCORE + ply as i32, None)
            } else {
                (0, None)
            };
        }
        let num_moves = moves.len();

        // --- Step 12b: Singular Extensions (Stockfish-style excluded-move search) ---
        // If TT move exists at sufficient depth with a lower-bound, test if it's singular
        // by searching ALL non-TT moves at reduced depth with a lowered beta.
        // If no alternative beats se_beta, the TT move is "singular" → extend it.
        let mut singular_extension: i8 = 0;
        if !is_root
            && depth >= 6
            && tt_move.is_some()
            && tt_depth >= depth.saturating_sub(3)
            && tt_entry
                .as_ref()
                .map_or(false, |e| e.flag != TTFlag::UpperBound)
        {
            let tt_sc = tt_score.unwrap_or(0);
            let se_beta = tt_sc - 2 * (depth as i32);
            let se_depth = (depth - 1) / 2;

            // Excluded-move search: search ALL non-TT moves at reduced depth
            let mut se_best = -INFINITY;
            for mi in 0..moves.len() {
                moves.pick_move(mi);
                let (mv, _) = moves.get(mi);
                if Some(mv) == tt_move {
                    continue;
                }
                let old_board = self.board;
                self.search_path_push(self.board.get_hash());
                self.nnue_make_move(mv, ply);
                self.board = old_board.make_move_new(mv);
                let (val, _) = self.alpha_beta(-se_beta, -se_beta + 1, se_depth, ply + 1);
                let score = -val;
                self.board = old_board;
                self.search_path_pop();

                if self.stop_flag.load(Ordering::Relaxed) {
                    return (0, None);
                }

                if score > se_best {
                    se_best = score;
                    if se_best >= se_beta {
                        break; // No need to continue, singularity disproved
                    }
                }
            }

            if se_best < se_beta {
                singular_extension = 1;
            } else if se_best >= beta {
                // Multi-cut: if a non-TT move also beats beta, prune
                return (se_best, None);
            }
        }

        let original_alpha = alpha;
        let mut best_move = None;
        let mut best_score = -INFINITY;
        let mut moves_searched = 0;
        // Track quiet moves that failed to produce a cutoff (for history gravity)
        let mut quiets_tried: [Option<ChessMove>; 64] = [None; 64];
        let mut quiets_count = 0usize;

        for move_idx in 0..num_moves {
            moves.pick_move(move_idx);
            let mv = moves.get(move_idx).0;
            let is_cap = self.is_capture(mv);
            let is_promo = mv.get_promotion().is_some();
            let is_tactical = is_cap || is_promo;

            // Pre-compute gives_check for quiet moves BEFORE pruning.
            // Checking moves must NEVER be pruned (Stockfish principle).
            // Only compute at shallow depths where pruning (LMP/futility/history/SEE)
            // actually fires — avoids expensive board copy at deep nodes.
            let gives_check = if !is_tactical && moves_searched > 0 && depth <= 8 {
                self.board.make_move_new(mv).checkers().0 != 0
            } else {
                false
            };

            // --- Step 13: Late Move Pruning (LMP) ---
            // Improving flag: allow more moves when improving
            if !is_root && !in_check && !is_tactical && !gives_check && depth <= 8 {
                let base = 3 + (depth as usize) * (depth as usize) / 3;
                let lmp_threshold = if improving { base + 3 } else { base };
                if moves_searched >= lmp_threshold {
                    continue;
                }
            }

            // --- Step 14: Futility Pruning ---
            if !is_root && !in_check && !is_tactical && !gives_check && depth <= 8 && moves_searched > 0 {
                let futility_margin = static_eval + 150 + 100 * (depth as i32);
                if futility_margin <= alpha {
                    continue;
                }
            }

            // --- Step 14b: History Pruning (Stockfish-inspired) ---
            // Prune quiet moves with very negative history at shallow depths
            if !is_root && !in_check && !is_tactical && !gives_check && depth <= 4 && moves_searched > 0 {
                let moved_piece_local = self.board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
                let moved_color_local = self.board.side_to_move();
                let side_local = if moved_color_local == Color::White {
                    0
                } else {
                    1
                };
                let h =
                    self.history[side_local][mv.get_source().to_index()][mv.get_dest().to_index()];
                let threshold = -2000 * (depth as i32);
                if h < threshold {
                    continue;
                }
                // Also check continuation history
                if ply >= 1 {
                    let prev_ply_idx = (ply - 1).min(MAX_PLY - 1);
                    let pp = self.piece_stack[prev_ply_idx] as usize;
                    if let Some(prev_mv) = self.move_stack[prev_ply_idx] {
                        let pt = prev_mv.get_dest().to_index();
                        let cp = Self::piece_index(moved_piece_local, moved_color_local) as usize;
                        let ct = mv.get_dest().to_index();
                        if pp < 12 && cp < 12 {
                            let ch = self.cont_history[pp][pt][cp][ct];
                            if ch < threshold {
                                continue;
                            }
                        }
                    }
                }
            }

            // --- Step 15: SEE Pruning for bad captures/quiet moves ---
            if !is_root && !is_promo && !gives_check && moves_searched > 0 {
                let see_threshold = if is_cap {
                    // Captures: prune if SEE is sufficiently negative (depth-scaled)
                    -20 * (depth as i32)
                } else if !is_tactical && depth <= 8 {
                    // Quiet moves: prune if SEE is negative (losing material)
                    -50 * (depth as i32)
                } else {
                    i32::MIN // don't prune
                };
                if see_threshold > i32::MIN && self.see_score(mv) < see_threshold {
                    continue;
                }
            }

            // --- Step 16: Make move ---
            let moved_piece = self.board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
            let moved_color = self.board.side_to_move();
            let old_board = self.board;
            // Push current position hash for repetition detection
            self.search_path_push(self.board.get_hash());
            self.nnue_make_move(mv, ply);
            self.board = self.board.make_move_new(mv);

            // Record move/piece in stacks for continuation history
            self.move_stack[ply_idx] = Some(mv);
            self.piece_stack[ply_idx] = Self::piece_index(moved_piece, moved_color);

            // Recompute gives_check precisely after make_move (the pre-pruning
            // value was only computed for quiet non-first moves; update for all).
            let gives_check = self.board.checkers().0 != 0;

            let mut score;

            // --- Extensions ---
            let mut ext: u8 = 0;

            // Check extension: only at high depth or PV nodes (Stockfish-style).
            // Unconditional check extension causes search explosion.
            if gives_check && (depth > 6 || is_pv) {
                ext = 1;
            }

            // Singular extension (highest priority)
            if Some(mv) == tt_move && singular_extension > 0 {
                ext = 1;
            }

            // Recapture extension: if we're recapturing on the same square
            if ext == 0 && is_cap && ply >= 1 {
                let prev_ply = (ply - 1).min(MAX_PLY - 1);
                if let Some(prev_mv) = self.move_stack[prev_ply] {
                    if mv.get_dest() == prev_mv.get_dest() {
                        ext = 1;
                    }
                }
            }

            // Passed pawn push extension: pawn pushed to 6th or 7th rank
            // In endgame (≤12 pieces), extend more aggressively to see promotions
            if ext == 0 && moved_piece == Piece::Pawn {
                let rank = mv.get_dest().to_index() / 8;
                let piece_count = self.board.combined().0.count_ones();
                let is_endgame = piece_count <= 12;

                let is_7th = (moved_color == Color::White && rank == 6)
                    || (moved_color == Color::Black && rank == 1);
                let is_6th = (moved_color == Color::White && rank == 5)
                    || (moved_color == Color::Black && rank == 2);
                let is_5th = (moved_color == Color::White && rank == 4)
                    || (moved_color == Color::Black && rank == 3);

                if is_7th {
                    // 7th rank push: extend by 2 in endgame, 1 otherwise
                    ext = if is_endgame { 2 } else { 1 };
                } else if is_6th {
                    ext = 1;
                } else if is_5th && is_endgame {
                    // 5th rank push in endgame: extend by 1 to help see promotion plans
                    ext = 1;
                }
            }

            let effective_depth = depth - 1 + ext;

            // --- Step 17: PVS (Principal Variation Search) ---
            if moves_searched == 0 {
                let (val, _) = self.alpha_beta(-beta, -alpha, effective_depth, ply + 1);
                score = -val;
            } else {
                // --- Step 18: Late Move Reductions (LMR) ---
                // Now applies to PV nodes too (with less reduction)
                let mut r: u8 = 0;
                if depth >= 3 && !in_check {
                    if is_tactical || gives_check {
                        if moves_searched >= 6 {
                            r = 1;
                        }
                    } else {
                        if moves_searched > 4 {
                            let di = (depth as usize).min(LMR_MAX_DEPTH - 1);
                            let mi = moves_searched.min(LMR_MAX_MOVES - 1);
                            r = LMR_TABLE[di][mi];
                        } else if moves_searched >= 2 {
                            r = 1;
                        }
                    }
                    // Reduce less for killer moves
                    if self.killers[ply_idx][0] == Some(mv) || self.killers[ply_idx][1] == Some(mv)
                    {
                        r = r.saturating_sub(1);
                    }

                    // PV nodes get less reduction (proven ~30 Elo gain)
                    if is_pv {
                        r = r.saturating_sub(1);
                    }

                    // SEE-based LMR: reduce more for moves with negative SEE
                    if is_cap && self.see_score(mv) < 0 {
                        r = r.saturating_add(1);
                    }

                    // --- History-based LMR adjustment ---
                    if !is_tactical {
                        let side = if moved_color == Color::White { 0 } else { 1 };
                        let h = self.history[side][mv.get_source().to_index()]
                            [mv.get_dest().to_index()];
                        // Good history → reduce less; bad history → reduce more
                        let h_adj = (h / 5000).clamp(-2, 2) as i8;
                        r = (r as i8 - h_adj).max(0) as u8;

                        // Also use continuation history for LMR adjustment
                        if ply >= 1 {
                            let prev_ply = (ply - 1).min(MAX_PLY - 1);
                            let pp = self.piece_stack[prev_ply] as usize;
                            if let Some(prev_mv) = self.move_stack[prev_ply] {
                                let pt = prev_mv.get_dest().to_index();
                                let cp = Self::piece_index(moved_piece, moved_color) as usize;
                                let ct = mv.get_dest().to_index();
                                if pp < 12 && cp < 12 {
                                    let ch = self.cont_history[pp][pt][cp][ct];
                                    let ch_adj = (ch / 5000).clamp(-1, 1) as i8;
                                    r = (r as i8 - ch_adj).max(0) as u8;
                                }
                            }
                        }
                    }

                    // Reduce more when NOT improving
                    if !improving {
                        r = r.saturating_add(1);
                    }

                    // Never reduce into qsearch (keep at least depth 1)
                    r = r.min(effective_depth.saturating_sub(1));
                }

                // Null-window search with reduction
                let (val, _) = self.alpha_beta(
                    -alpha - 1,
                    -alpha,
                    effective_depth.saturating_sub(r),
                    ply + 1,
                );
                score = -val;

                // Re-search at full depth if LMR failed high
                if r > 0 && score > alpha {
                    let (val, _) = self.alpha_beta(-alpha - 1, -alpha, effective_depth, ply + 1);
                    score = -val;
                }

                // Full PV re-search if null-window failed high
                if score > alpha && score < beta {
                    let (val, _) = self.alpha_beta(-beta, -alpha, effective_depth, ply + 1);
                    score = -val;
                }
            }

            // --- Step 19: Undo move ---
            self.board = old_board;
            self.search_path_pop();

            if self.stop_flag.load(Ordering::Relaxed) {
                return (0, None);
            }

            // Track quiet moves for history gravity
            if !is_cap && !is_promo && quiets_count < 64 {
                quiets_tried[quiets_count] = Some(mv);
                quiets_count += 1;
            }

            moves_searched += 1;

            // --- Step 20: Update best ---
            if score > best_score {
                best_score = score;
                best_move = Some(mv);
            }

            if score > alpha {
                alpha = score;

                // PV tracking: record this move and copy child's PV
                if ply < MAX_PLY {
                    self.pv_table[ply][ply] = Some(mv);
                    let child = (ply + 1).min(MAX_PLY - 1);
                    let child_len = self.pv_len[child].min(MAX_PLY);
                    for i in child..child_len {
                        self.pv_table[ply][i] = self.pv_table[child][i];
                    }
                    self.pv_len[ply] = child_len;
                }

                if score >= beta {
                    // --- Step 21: Beta cutoff → update killers, history, countermove, cont_history ---
                    let side = if moved_color == Color::White { 0 } else { 1 };
                    let from = mv.get_source().to_index();
                    let to = mv.get_dest().to_index();

                    if is_cap {
                        // Capture history with Stockfish gravity formula
                        let bonus = (depth as i32) * (depth as i32);
                        let ai = Self::cap_piece_index(moved_piece);
                        let vi = Self::cap_piece_index(
                            old_board.piece_on(mv.get_dest()).unwrap_or(Piece::Pawn),
                        );
                        let ct = mv.get_dest().to_index();
                        let ch = &mut self.capture_history[ai][ct][vi];
                        *ch += bonus - ch.abs() * bonus / 16384;
                        *ch = (*ch).clamp(-30000, 30000);
                    }

                    if !is_cap {
                        // Killer moves
                        self.killers[ply_idx][1] = self.killers[ply_idx][0];
                        self.killers[ply_idx][0] = Some(mv);

                        // Stockfish-style history gravity:
                        // bonus = depth^2, update = bonus - |h| * bonus / 16384
                        let bonus = (depth as i32) * (depth as i32);
                        let h = &mut self.history[side][from][to];
                        *h += bonus - h.abs() * bonus / 16384;
                        *h = (*h).clamp(-30000, 30000);

                        // History gravity: penalize all quiet moves that didn't cause cutoff
                        for i in 0..quiets_count {
                            if let Some(q) = quiets_tried[i] {
                                if q != mv {
                                    let qf = q.get_source().to_index();
                                    let qt = q.get_dest().to_index();
                                    let qh = &mut self.history[side][qf][qt];
                                    *qh += -bonus - qh.abs() * bonus / 16384;
                                    *qh = (*qh).clamp(-30000, 30000);
                                }
                            }
                        }

                        // Countermove heuristic: record this move as refutation of previous move
                        if ply >= 1 {
                            let prev_ply = (ply - 1).min(MAX_PLY - 1);
                            if let Some(prev_mv) = self.move_stack[prev_ply] {
                                let pf = prev_mv.get_source().to_index();
                                let pt = prev_mv.get_dest().to_index();
                                self.countermove[pf][pt] = Some(mv);
                            }
                        }

                        // Continuation history: update [prev_piece][prev_to][cur_piece][cur_to]
                        if ply >= 1 {
                            let prev_ply = (ply - 1).min(MAX_PLY - 1);
                            let pp = self.piece_stack[prev_ply] as usize;
                            if let Some(prev_mv) = self.move_stack[prev_ply] {
                                let pt = prev_mv.get_dest().to_index();
                                let cp = Self::piece_index(moved_piece, moved_color) as usize;
                                let ct = mv.get_dest().to_index();
                                if pp < 12 && cp < 12 {
                                    let ch = &mut self.cont_history[pp][pt][cp][ct];
                                    *ch += bonus - ch.abs() * bonus / 16384;
                                    *ch = (*ch).clamp(-30000, 30000);

                                    // Penalize quiet non-cutoff moves in cont_history too
                                    for i in 0..quiets_count {
                                        if let Some(q) = quiets_tried[i] {
                                            if q != mv {
                                                let qp = self
                                                    .board
                                                    .piece_on(q.get_source())
                                                    .unwrap_or(Piece::Pawn);
                                                let qi =
                                                    Self::piece_index(qp, moved_color) as usize;
                                                let qt = q.get_dest().to_index();
                                                if qi < 12 {
                                                    let qch =
                                                        &mut self.cont_history[pp][pt][qi][qt];
                                                    *qch += -bonus - qch.abs() * bonus / 16384;
                                                    *qch = (*qch).clamp(-30000, 30000);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    self.tt.store(TTEntry {
                        hash,
                        depth,
                        score,
                        flag: TTFlag::LowerBound,
                        best_move: Some(mv),
                        static_eval,
                    });
                    // Update correction histories: search found score != static eval
                    self.update_correction_histories(static_eval, score, depth);
                    return (score, Some(mv));
                }
            }
        }

        // --- Step 22: Store in TT ---
        let flag = if best_move.is_some() && alpha > original_alpha {
            TTFlag::Exact
        } else {
            TTFlag::UpperBound
        };

        self.tt.store(TTEntry {
            hash,
            depth,
            score: best_score,
            flag,
            best_move,
            static_eval,
        });

        // Update correction histories at non-root PV/all nodes
        if !is_root {
            self.update_correction_histories(static_eval, best_score, depth);
        }

        (best_score, best_move)
    }

    // =========================================================================
    // QUIESCENCE SEARCH with Delta Pruning + Check Extensions
    // =========================================================================
    fn quiescence(&mut self, mut alpha: i32, beta: i32, ply: usize, qs_depth: i32) -> i32 {
        self.nodes += 1;
        if self.nodes & 4095 == 0 {
            self.shared_nodes.fetch_add(4096, Ordering::Relaxed);
        }

        // Abort check: prevent infinite qsearch and respect time limits
        if ply >= MAX_PLY {
            return self.fast_eval(ply);
        }
        if self.nodes & 8191 == 0 {
            if let Some(ref tm) = self.time_manager {
                if tm.should_stop() {
                    self.stop_flag.store(true, Ordering::Relaxed);
                }
            }
        }
        if self.stop_flag.load(Ordering::Relaxed) {
            return 0;
        }

        let in_check = self.board.checkers().0 != 0;

        // --- TT probe in qsearch ---
        let hash = self.board.get_hash();
        if let Some(entry) = self.tt.get(hash) {
            match entry.flag {
                TTFlag::Exact => return entry.score,
                TTFlag::LowerBound => {
                    if entry.score >= beta {
                        return entry.score;
                    }
                }
                TTFlag::UpperBound => {
                    if entry.score <= alpha {
                        return entry.score;
                    }
                }
            }
        }

        // Use fast static eval in quiescence (NN is too slow per-node)
        let stand_pat = if in_check {
            -INFINITY
        } else {
            self.fast_eval(ply)
        };

        if !in_check {
            if stand_pat >= beta {
                return stand_pat;
            }
            if stand_pat > alpha {
                alpha = stand_pat;
            }

            // Delta pruning threshold: if even capturing the best piece can't raise alpha
            let big_delta = 1000; // Queen value ~= 900, with margin
            if stand_pat + big_delta < alpha {
                return alpha;
            }
        }

        // When in check: search ALL legal moves (must escape check)
        // Otherwise: only search captures
        let mut scored_moves = MoveList::new();

        if in_check {
            let moves = MoveGen::new_legal(&self.board);
            let targets = *self.board.color_combined(!self.board.side_to_move());
            for mv in moves {
                let is_cap = (targets & BitBoard::from_square(mv.get_dest())).0 != 0;
                let mut score: i32 = 0;
                if is_cap {
                    let victim = self.board.piece_on(mv.get_dest()).unwrap_or(Piece::Pawn);
                    let attacker = self.board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
                    score = piece_value(victim) * 10 - piece_value(attacker) + 50_000;
                }
                scored_moves.push(mv, score);
            }
            // If no legal moves in check → checkmate
            if scored_moves.is_empty() {
                return -MATE_SCORE + ply as i32;
            }
        } else {
            let targets = *self.board.color_combined(!self.board.side_to_move());
            let mut moves = MoveGen::new_legal(&self.board);
            // First pass: captures
            moves.set_iterator_mask(targets);
            for mv in &mut moves {
                let victim = self.board.piece_on(mv.get_dest()).unwrap_or(Piece::Pawn);
                let attacker = self.board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
                let see_score = piece_value(victim) * 10 - piece_value(attacker);
                scored_moves.push(mv, see_score);
            }
            // Second pass: non-capture queen promotions (critical for endgames)
            moves.set_iterator_mask(!targets & !chess::EMPTY);
            for mv in &mut moves {
                if mv.get_promotion() == Some(Piece::Queen) {
                    scored_moves.push(mv, 44_000);
                }
                // Third pass: quiet check-giving moves in first 2 QS plies
                // Catches knight forks, discovered checks, and other tactical shots
                else if qs_depth == 0 && mv.get_promotion().is_none() {
                    let new_board = self.board.make_move_new(mv);
                    if new_board.checkers().0 != 0 {
                        scored_moves.push(mv, 30_000); // Below captures, above nothing
                    }
                }
            }
        }
        let original_alpha = alpha;
        let mut best_move = None;
        let mut best_score = stand_pat;

        let num_qmoves = scored_moves.len();
        for move_idx in 0..num_qmoves {
            scored_moves.pick_move(move_idx);
            let mv = scored_moves.get(move_idx).0;

            // Delta pruning per-move (skip when in check)
            if !in_check {
                let victim = self.board.piece_on(mv.get_dest()).unwrap_or(Piece::Pawn);
                if stand_pat + piece_value(victim) + 200 < alpha && mv.get_promotion().is_none() {
                    continue;
                }
                // SEE pruning in qsearch: skip captures that lose material
                if mv.get_promotion().is_none() && self.see_score(mv) < 0 {
                    continue;
                }
            }

            let old_board = self.board;
            self.search_path_push(self.board.get_hash());
            self.nnue_make_move(mv, ply);
            self.board = self.board.make_move_new(mv);

            let score = -self.quiescence(-beta, -alpha, ply + 1, qs_depth + 1);

            self.board = old_board;
            self.search_path_pop();

            if score > best_score {
                best_score = score;
                best_move = Some(mv);
            }

            if score >= beta {
                // Store TT entry for beta cutoff in qsearch
                self.tt.store(TTEntry {
                    hash,
                    depth: 0,
                    score,
                    flag: TTFlag::LowerBound,
                    best_move: Some(mv),
                    static_eval: stand_pat,
                });
                return score;
            }
            if score > alpha {
                alpha = score;
            }
        }

        // Store TT entry for qsearch result
        let flag = if best_move.is_some() && alpha > original_alpha {
            TTFlag::Exact
        } else {
            TTFlag::UpperBound
        };
        self.tt.store(TTEntry {
            hash,
            depth: 0,
            score: best_score,
            flag,
            best_move,
            static_eval: stand_pat,
        });

        best_score
    }

    // =========================================================================
    // MOVE ORDERING: TT > Policy > Captures > Promos > Killers > Countermove > ContHist > History
    // =========================================================================
    fn order_moves(
        &self,
        moves: MoveGen,
        policy: Option<&[f32]>,
        ply: usize,
        tt_move: Option<ChessMove>,
    ) -> MoveList {
        let ply_idx = ply.min(MAX_PLY - 1);
        let targets = *self.board.color_combined(!self.board.side_to_move());
        let side = if self.board.side_to_move() == Color::White {
            0
        } else {
            1
        };
        let mut scored = MoveList::new();

        // Pre-fetch previous move info for countermove + continuation history
        let prev_info: Option<(usize, usize, usize)> = if ply >= 1 {
            let prev_ply = (ply - 1).min(MAX_PLY - 1);
            if let Some(prev_mv) = self.move_stack[prev_ply] {
                let pf = prev_mv.get_source().to_index();
                let pt = prev_mv.get_dest().to_index();
                let pp = self.piece_stack[prev_ply] as usize;
                Some((pf, pt, pp))
            } else {
                None
            }
        } else {
            None
        };

        for mv in moves {
            let mut score: i32 = 0;

            // 1. TT move gets highest priority
            if Some(mv) == tt_move {
                score += 200_000;
            }

            // 2. Neural policy score (capped to prevent overriding tactical signals)
            if let Some(pol) = policy {
                let token = MoveCodec::get_policy_index(&mv);
                if token < pol.len() {
                    let mut policy_bonus = (pol[token] * 8_000.0) as i32;
                    // SEE validation: halve policy bonus for moves that lose material
                    if policy_bonus > 500 && self.see_score(mv) < 0 {
                        policy_bonus /= 4;
                    }
                    score += policy_bonus;
                }
            }

            // 3. Captures scored by MVV-LVA + capture history
            let dest_bb = BitBoard::from_square(mv.get_dest());
            if (targets & dest_bb).0 != 0 {
                let victim = self.board.piece_on(mv.get_dest()).unwrap_or(Piece::Pawn);
                let attacker = self.board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
                score += piece_value(victim) * 10 - piece_value(attacker) + 50_000;
                let ai = Self::cap_piece_index(attacker);
                let vi = Self::cap_piece_index(victim);
                let to = mv.get_dest().to_index();
                score += self.capture_history[ai][to][vi] / 2;
            }

            // 4. Promotions
            if mv.get_promotion().is_some() {
                score += 45_000;
            }

            // 5. Killer moves
            if self.killers[ply_idx][0] == Some(mv) {
                score += 40_000;
            } else if self.killers[ply_idx][1] == Some(mv) {
                score += 35_000;
            }

            // 6. Countermove heuristic
            if let Some((pf, pt, _)) = prev_info {
                if self.countermove[pf][pt] == Some(mv) {
                    score += 30_000;
                }
            }

            // 7. Continuation history
            if let Some((_, pt, pp)) = prev_info {
                if pp < 12 {
                    let piece = self.board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
                    let color = self.board.side_to_move();
                    let cp = Self::piece_index(piece, color) as usize;
                    let ct = mv.get_dest().to_index();
                    if cp < 12 {
                        score += self.cont_history[pp][pt][cp][ct] / 2;
                    }
                }
            }

            // 8. History heuristic
            let from = mv.get_source().to_index();
            let to = mv.get_dest().to_index();
            score += self.history[side][from][to];

            scored.push(mv, score);
        }

        scored
    }
}

fn piece_value(piece: Piece) -> i32 {
    match piece {
        Piece::Pawn => 100,
        Piece::Knight => 320,
        Piece::Bishop => 330,
        Piece::Rook => 500,
        Piece::Queen => 900,
        Piece::King => 20000,
    }
}

/// SEE piece values (used exclusively by Static Exchange Evaluation)
#[inline]
fn see_val(piece: Piece) -> i32 {
    match piece {
        Piece::Pawn => 100,
        Piece::Knight => 320,
        Piece::Bishop => 330,
        Piece::Rook => 500,
        Piece::Queen => 900,
        Piece::King => 20000,
    }
}
