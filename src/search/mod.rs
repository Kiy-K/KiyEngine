use crate::engine::{Engine, KVCache};
use crate::nnue::accumulator::Accumulator;
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
}

// =============================================================================
// LOCK-FREE TRANSPOSITION TABLE (Hyatt/Mann XOR trick)
// =============================================================================
//
// Each slot is 16 bytes: two AtomicU64 values.
//   slot[0] = key ^ data   (verification word)
//   slot[1] = data          (packed entry)
//
// On read:  load both, compute key = slot[0] ^ slot[1], verify against hash.
// On write: store key ^ data, then data.
// Race conditions produce corrupt keys that fail verification → harmless.
//
// Data packing (64 bits):
//   bits  0-5:  from square (6 bits)
//   bits  6-11: to square (6 bits)
//   bits 12-15: promotion (4 bits: 0=none,1=N,2=B,3=R,4=Q)
//   bits 16-31: score (i16)
//   bits 32-39: depth (u8)
//   bits 40-41: flag (2 bits: 0=Exact,1=Lower,2=Upper)
//   bit  63:    always set (sentinel — distinguishes from empty)

pub struct TranspositionTable {
    // Interleaved layout: [key0, data0, key1, data1, ...]
    // Key and data for same slot are adjacent → same cache line.
    slots: Vec<AtomicU64>,
    mask: usize,
}

impl TranspositionTable {
    pub fn new(size_mb: usize) -> Self {
        let size_mb = size_mb.max(64);
        let bytes = size_mb * 1024 * 1024;
        // Each entry = 2 × 8 bytes = 16 bytes, interleaved in one Vec
        let num_entries = (bytes / 16).next_power_of_two();
        let slots = (0..num_entries * 2).map(|_| AtomicU64::new(0)).collect();
        Self {
            slots,
            mask: num_entries - 1,
        }
    }

    pub fn resize(&self, mb: usize) {
        // Cannot truly reallocate through &self (would need &mut self or interior mutability).
        // For now, clear existing entries. The TT is pre-sized at construction;
        // to change size, recreate the TT via UciHandler.
        let _ = mb;
        self.clear();
    }

    #[inline(always)]
    pub fn get(&self, hash: u64) -> Option<TTEntry> {
        let idx = (hash as usize) & self.mask;
        let base = idx * 2;
        let d = self.slots[base + 1].load(Ordering::Relaxed);
        if d == 0 {
            return None;
        }
        let k = self.slots[base].load(Ordering::Relaxed);
        if k ^ d != hash {
            return None;
        }
        Self::unpack(hash, d)
    }

    #[inline(always)]
    pub fn store(&self, entry: TTEntry) {
        let idx = (entry.hash as usize) & self.mask;
        let base = idx * 2;
        let new_data = Self::pack(&entry);

        // Depth-preferred replacement: keep deeper entries from other positions
        let old_data = self.slots[base + 1].load(Ordering::Relaxed);
        if old_data != 0 {
            let old_key = self.slots[base].load(Ordering::Relaxed);
            let old_hash = old_key ^ old_data;
            if old_hash != entry.hash {
                let old_depth = ((old_data >> 32) & 0xFF) as u8;
                if entry.depth < old_depth {
                    return;
                }
            }
        }

        self.slots[base].store(entry.hash ^ new_data, Ordering::Relaxed);
        self.slots[base + 1].store(new_data, Ordering::Relaxed);
    }

    pub fn clear(&self) {
        for slot in self.slots.iter() {
            slot.store(0, Ordering::Relaxed);
        }
    }

    fn pack(entry: &TTEntry) -> u64 {
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
        Some(TTEntry {
            hash,
            depth,
            score,
            flag,
            best_move,
        })
    }
}

// Module eval local (fallback)
pub mod eval;
pub mod lazy_smp;
pub mod mcts;
pub mod mcts_arena;
pub mod mcts_batch;
pub mod mcts_memory;
pub mod time;
pub mod tt;
use self::time::TimeManager;

// =============================================================================
// SEARCHER (MANAGER)
// =============================================================================

pub struct Searcher {
    engine: Arc<Engine>,
    tt: Arc<TranspositionTable>,
    pub threads: usize,
    pub move_overhead: u32,
    pub nnue: Option<Arc<NnueNetwork>>,
}

impl Searcher {
    pub fn new(engine: Arc<Engine>, tt: Arc<TranspositionTable>) -> Self {
        Self {
            engine,
            tt,
            threads: 4,
            move_overhead: 60, // Default 60ms overhead
            nnue: None,
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
        let engine = Arc::clone(&self.engine);
        let tt = Arc::clone(&self.tt);
        let num_threads = self.threads;
        let move_overhead = self.move_overhead;
        let is_white = board.side_to_move() == Color::White;
        let nnue = self.nnue.clone();

        let ply = move_history.len() as u32;
        let pos_hist = Arc::new(position_history);

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
            let shared_policy: Arc<Option<(Vec<f32>, f32)>> = {
                let result = if let Some(ref cache) = kv_cache {
                    let mut cache_guard = cache.lock();
                    engine.forward_with_cache(&move_history, &mut cache_guard)
                } else {
                    engine.forward(&move_history)
                };
                if let Ok(res) = result {
                    if let Ok(pol_vec) = res.0.to_vec1::<f32>() {
                        Arc::new(Some((pol_vec, res.1)))
                    } else {
                        Arc::new(None)
                    }
                } else {
                    Arc::new(None)
                }
            };

            let mut handles = vec![];

            for i in 0..num_threads {
                let tt_clone = Arc::clone(&tt);
                let board_clone = board.clone();
                let stop_clone = Arc::clone(&stop_flag);
                let tm_clone = Arc::clone(&time_manager);
                let tx_clone = if i == 0 { Some(tx.clone()) } else { None };
                let is_main = i == 0;
                let policy_clone = Arc::clone(&shared_policy);
                let pos_hist_clone = Arc::clone(&pos_hist);

                handles.push(
                    std::thread::Builder::new()
                        .stack_size(64 * 1024 * 1024) // 64MB stack for deep recursion
                        .spawn(move || {
                            let mut worker = SearchWorker::new(tt_clone, board_clone, stop_clone, pos_hist_clone);
                            let mut best_move = None;
                            let mut last_score = 0;
                            let mut prev_best: Option<ChessMove> = None;
                            let start_time = std::time::Instant::now();

                            // Use shared pre-computed NN policy (no per-thread forward pass)
                            worker.set_cached_policy(&*policy_clone);

                            // Initialize NNUE accumulator at root (ply 0)
                            worker.nnue_refresh(0);

                            // Wire time manager for in-search abort checks
                            let depth_only = max_depth > 0 && wtime == 0 && btime == 0;
                            if !depth_only {
                                worker.time_manager = Some(Arc::clone(&tm_clone));
                            }

                            // Lazy SMP: Varied depth offsets per helper thread (Stockfish-style)
                            // Thread 0 (main): full depth
                            // Thread 1: depth-1, Thread 2: depth+0, Thread 3: depth-2, etc.
                            let effective_max = if max_depth == 0 { MAX_DEPTH } else { max_depth };
                            // Lazy SMP: main thread full depth, helpers use varied offsets
                            // for exploration diversity (Stockfish-inspired)
                            let local_max_depth = if is_main {
                                effective_max
                            } else {
                                // Alternate: even helpers search same depth, odd helpers search depth-1 or depth-2
                                let offset = match i % 4 {
                                    1 => 1,
                                    2 => 0, // Same depth as main (different move ordering due to TT races)
                                    3 => 2,
                                    _ => 1,
                                };
                                effective_max.saturating_sub(offset).max(1)
                            };

                            const MIN_DEPTH: u8 = 3;

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
                                if !depth_only && depth > MIN_DEPTH {
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

                                            // Update time manager with iteration results
                                            tm_clone.update_iteration(
                                                depth as u32,
                                                score,
                                                best_move_changed,
                                            );

                                            let elapsed =
                                                start_time.elapsed().as_secs_f64().max(0.001);
                                            let nps = (worker.nodes as f64 / elapsed) as u64;

                                            let pv_str = worker.extract_pv(&worker.board.clone(), depth as usize);
                                            let elapsed_ms = (elapsed * 1000.0) as u64;
                                            println!(
                                                "info depth {} score cp {} nodes {} nps {} time {} pv {}",
                                                depth, score, worker.nodes, nps, elapsed_ms, pv_str
                                            );
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
                                if let Some(mv) = best_move {
                                    let _ = tx_clone.unwrap().send(format!("bestmove {}", mv));
                                } else {
                                    // Fallback: Lấy đại một nước hợp lệ nếu chưa tìm thấy gì
                                    let mut moves = MoveGen::new_legal(&board_clone);
                                    if let Some(mv) = moves.next() {
                                        let _ = tx_clone.unwrap().send(format!("bestmove {}", mv));
                                    } else {
                                        // Checkmate/Stalemate detected at root
                                        let _ =
                                            tx_clone.unwrap().send("bestmove (none)".to_string());
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
    // Repetition detection: game history (from UCI position command) + search path
    game_history: Arc<Vec<u64>>,
    search_path: [u64; MAX_PLY],
    search_path_len: usize,
    // NNUE evaluation
    pub nnue: Option<Arc<NnueNetwork>>,
    pub nnue_acc: Vec<Accumulator>,
    pub use_nnue_eval: bool,
    // Triangular PV table
    pv_table: [[Option<ChessMove>; MAX_PLY]; MAX_PLY],
    pv_len: [usize; MAX_PLY],
}

impl SearchWorker {
    pub fn new(tt: Arc<TranspositionTable>, board: Board, stop_flag: Arc<AtomicBool>, game_history: Arc<Vec<u64>>) -> Self {
        Self {
            tt,
            board,
            stop_flag,
            time_manager: None,
            nodes: 0,
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
            game_history,
            search_path: [0u64; MAX_PLY],
            search_path_len: 0,
            nnue: None,
            nnue_acc: (0..MAX_PLY).map(|_| Accumulator::new(512)).collect(),
            use_nnue_eval: false,
            pv_table: [[None; MAX_PLY]; MAX_PLY],
            pv_len: [0; MAX_PLY],
        }
    }


    /// Extract principal variation from TT (up to `max_len` moves).
    fn extract_pv(&self, root_board: &Board, max_len: usize) -> String {
        let mut pv = String::new();
        let mut board = *root_board;
        let mut seen = Vec::with_capacity(max_len);
        for _ in 0..max_len {
            let hash = board.get_hash();
            if seen.contains(&hash) {
                break; // Prevent TT PV cycles
            }
            seen.push(hash);
            if let Some(entry) = self.tt.get(hash) {
                if let Some(mv) = entry.best_move {
                    if board.legal(mv) {
                        if !pv.is_empty() {
                            pv.push(' ');
                        }
                        pv.push_str(&format!("{}", mv));
                        board = board.make_move_new(mv);
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        pv
    }

    /// Set game position history for repetition detection.
    fn set_game_history(&mut self, history: Vec<u64>) {
        self.game_history = Arc::new(history);
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
        let start = self.game_history.len().saturating_sub(100);
        for &h in &self.game_history[start..] {
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
    // FAST STATIC EVAL — NNUE if available, else Material + PST.
    // =========================================================================
    #[inline]
    fn fast_eval(&mut self, _ply: usize) -> i32 {
        // NNUE not yet ready — using Material + PST fallback
        eval::evaluate(&self.board, 0)
    }

    // =========================================================================
    // NNUE ACCUMULATOR MANAGEMENT
    // =========================================================================

    /// Refresh NNUE accumulator at given ply from the current board.
    pub fn nnue_refresh(&mut self, ply: usize) {
        if let Some(ref net) = self.nnue {
            let idx = ply.min(MAX_PLY - 1);
            self.nnue_acc[idx].refresh(&self.board, net);
        }
    }

    /// Incrementally update NNUE accumulator at ply+1 from ply after a move.
    /// Falls back to marking child as uncomputed (lazy refresh will handle it).
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
                if let Some(delta) = features::compute_delta(&self.board, mv) {
                    let (left, right) = self.nnue_acc.split_at_mut(child_idx);
                    let parent = &left[parent_idx];
                    let child = &mut right[0];
                    child.update_from(parent, &delta, net);
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
                right[0].white.copy_from_slice(&left[parent_idx].white);
                right[0].black.copy_from_slice(&left[parent_idx].black);
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
        if !is_root {
            let hash = self.board.get_hash();
            if self.is_repetition() {
                return (0, None); // Draw by repetition
            }
        }

        // --- Step 2: Check extension ---
        let in_check = self.board.checkers().0 != 0;
        if in_check && depth < MAX_PLY as u8 {
            depth += 1;
        }

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
        let tt_move = tt_entry.as_ref().and_then(|e| e.best_move);
        let tt_score = tt_entry.as_ref().map(|e| e.score);
        let tt_depth = tt_entry.as_ref().map(|e| e.depth).unwrap_or(0);

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
        if depth == 0 {
            return (self.quiescence(alpha, beta, ply), None);
        }
        self.nodes += 1;

        // --- Step 6: Static Evaluation for pruning (fast, no NN) ---
        let static_eval = if in_check {
            -INFINITY
        } else {
            self.fast_eval(ply)
        };

        // Store eval in stack for improving detection
        self.eval_stack[ply_idx] = static_eval;

        // --- Step 6b: Improving flag ---
        // Position is "improving" if static eval is better than 2 plies ago
        let improving =
            !in_check && ply >= 2 && static_eval > self.eval_stack[ply_idx.saturating_sub(2)];

        // --- Step 7: Razoring (Stockfish-style) ---
        if !is_pv && !in_check && depth <= 3 {
            let razor_margin = 300 + 250 * (depth as i32);
            if static_eval + razor_margin < alpha {
                let q_score = self.quiescence(alpha, beta, ply);
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
            && static_eval - (if improving { 70 } else { 100 }) * (depth as i32) >= beta
            && static_eval < MATE_SCORE - 100
        {
            return (static_eval, None);
        }

        // --- Step 8b: Probcut ---
        // If a shallow search with a raised beta finds a cutoff, prune
        let probcut_beta = beta + 200;
        if !is_pv
            && !in_check
            && depth >= 5
            && static_eval >= probcut_beta - 100
            && static_eval < MATE_SCORE - 100
        {
            let pc_depth = depth.saturating_sub(4);
            let (val, _) = self.alpha_beta(probcut_beta - 1, probcut_beta, pc_depth, ply);
            if val >= probcut_beta {
                return (val, None);
            }
        }

        // --- Step 9: Real Null Move Pruning ---
        // Actually pass the move and search with reduced depth
        if !is_pv && !in_check && depth >= 3 && static_eval >= beta && self.has_non_pawn_material()
        {
            if let Some(null_board) = self.board.null_move() {
                let old_board = self.board;
                self.search_path_push(self.board.get_hash());
                self.board = null_board;
                self.search_path_push(self.board.get_hash());

                // Adaptive reduction: R = 3 + depth/6, capped
                let r = 3 + (depth / 6).min(3);
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
        // Non-PV: reduce at depth>=3; PV: reduce at depth>=4
        if tt_move.is_none() && !in_check && (depth >= 4 || (!is_pv && depth >= 3)) {
            depth -= 1;
        }

        // --- Step 11: Neural Policy Lookup (root only, from cache — zero cost) ---
        let policy: Option<&[f32]> = if is_root {
            self.cached_root_policy.as_deref()
        } else {
            None
        };

        // --- Step 12: Generate and order moves ---
        let mut moves = self.order_moves(MoveGen::new_legal(&self.board), policy, ply, tt_move);

        if moves.is_empty() {
            return if in_check {
                (-MATE_SCORE + ply as i32, None)
            } else {
                (0, None)
            };
        }

        // --- Step 12b: Singular Extensions (Stockfish restricted SE) ---
        // If TT move exists at sufficient depth with a lower-bound, test if it's singular
        // by searching *only the second-best move* at reduced depth.
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

            // Search only the best non-TT move at reduced depth
            let mut se_score = -INFINITY;
            for (mv, _) in &moves {
                if Some(*mv) == tt_move {
                    continue;
                }
                let old_board = self.board;
                self.nnue_make_move(*mv, ply);
                self.board = old_board.make_move_new(*mv);
                self.search_path_push(self.board.get_hash());
                let (val, _) = self.alpha_beta(-se_beta, -se_beta + 1, se_depth, ply + 1);
                let score = -val;
                self.board = old_board;
                self.search_path_pop();

                if self.stop_flag.load(Ordering::Relaxed) {
                    return (0, None);
                }

                se_score = score;
                // Only test the top-ranked non-TT move (restricted SE)
                break;
            }

            if se_score < se_beta {
                singular_extension = 1;
            } else if se_score >= beta {
                // Multi-cut: if the non-TT move also beats beta, prune
                return (se_score, None);
            }
        }

        let original_alpha = alpha;
        let mut best_move = None;
        let mut best_score = -INFINITY;
        let mut moves_searched = 0;
        // Track quiet moves that failed to produce a cutoff (for history gravity)
        let mut quiets_tried: [Option<ChessMove>; 64] = [None; 64];
        let mut quiets_count = 0usize;

        for move_idx in 0..moves.len() {
            Self::pick_next_move(&mut moves, move_idx);
            let mv = moves[move_idx].0;
            let is_cap = self.is_capture(mv);
            let is_promo = mv.get_promotion().is_some();
            let is_tactical = is_cap || is_promo;

            // --- Step 13: Late Move Pruning (LMP) ---
            // Improving flag: allow more moves when improving
            if !is_root && !in_check && !is_tactical && depth <= 8 {
                let base = 3 + (depth as usize) * (depth as usize) / 3;
                let lmp_threshold = if improving { base + 3 } else { base };
                if moves_searched >= lmp_threshold {
                    continue;
                }
            }

            // --- Step 14: Futility Pruning ---
            if !is_root && !in_check && !is_tactical && depth <= 8 && moves_searched > 0 {
                let futility_margin = static_eval + 150 + 100 * (depth as i32);
                if futility_margin <= alpha {
                    continue;
                }
            }

            // --- Step 15: SEE Pruning for bad captures/quiet moves ---
            if !is_root && !is_promo && moves_searched > 0 {
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
            self.search_path_push(self.board.get_hash());

            // Record move/piece in stacks for continuation history
            self.move_stack[ply_idx] = Some(mv);
            self.piece_stack[ply_idx] = Self::piece_index(moved_piece, moved_color);

            let gives_check = self.board.checkers().0 != 0;

            let mut score;

            // --- Extensions ---
            let mut ext: u8 = 0;

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
            if ext == 0 && moved_piece == Piece::Pawn {
                let rank = mv.get_dest().to_index() / 8;
                if (moved_color == Color::White && rank >= 5)
                    || (moved_color == Color::Black && rank <= 2)
                {
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
                    });
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
        });

        (best_score, best_move)
    }

    // =========================================================================
    // QUIESCENCE SEARCH with Delta Pruning + Check Extensions
    // =========================================================================
    fn quiescence(&mut self, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        self.nodes += 1;

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
        let mut scored_moves: Vec<(ChessMove, i32)> = Vec::with_capacity(16);

        if in_check {
            let moves = MoveGen::new_legal(&self.board);
            for mv in moves {
                let targets = *self.board.color_combined(!self.board.side_to_move());
                let is_cap = (targets & BitBoard::from_square(mv.get_dest())).0 != 0;
                let mut score: i32 = 0;
                if is_cap {
                    let victim = self.board.piece_on(mv.get_dest()).unwrap_or(Piece::Pawn);
                    let attacker = self.board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
                    score = piece_value(victim) * 10 - piece_value(attacker) + 50_000;
                }
                scored_moves.push((mv, score));
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
                scored_moves.push((mv, see_score));
            }
            // Second pass: non-capture queen promotions (critical for endgames)
            moves.set_iterator_mask(!targets & !chess::EMPTY);
            // Actually, iterate over all remaining non-capture moves and pick queen promos
            let all_moves = MoveGen::new_legal(&self.board);
            for mv in all_moves {
                if mv.get_promotion() == Some(Piece::Queen)
                    && (targets & BitBoard::from_square(mv.get_dest())).0 == 0
                {
                    scored_moves.push((mv, 44_000)); // Just below main search promo score
                }
            }
        }
        let original_alpha = alpha;
        let mut best_move = None;
        let mut best_score = stand_pat;

        for move_idx in 0..scored_moves.len() {
            Self::pick_next_move(&mut scored_moves, move_idx);
            let mv = scored_moves[move_idx].0;

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
            self.nnue_make_move(mv, ply);
            self.board = self.board.make_move_new(mv);

            let score = -self.quiescence(-beta, -alpha, ply + 1);

            self.board = old_board;

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
        });

        alpha
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
    ) -> Vec<(ChessMove, i32)> {
        let ply_idx = ply.min(MAX_PLY - 1);
        let targets = *self.board.color_combined(!self.board.side_to_move());
        let side = if self.board.side_to_move() == Color::White {
            0
        } else {
            1
        };
        let mut scored: Vec<(ChessMove, i32)> = Vec::with_capacity(40);

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

            // 2. Neural policy score (very high priority when available)
            if let Some(pol) = policy {
                let token = MoveCodec::get_policy_index(&mv);
                if token < pol.len() {
                    score += (pol[token] * 15_000.0) as i32;
                }
            }

            // 3. Captures scored by MVV-LVA + capture history
            if (targets & BitBoard::from_square(mv.get_dest())).0 != 0 {
                let victim = self.board.piece_on(mv.get_dest()).unwrap_or(Piece::Pawn);
                let attacker = self.board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
                score += piece_value(victim) * 10 - piece_value(attacker) + 50_000;
                // Capture history bonus
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

            scored.push((mv, score));
        }

        scored
    }

    /// Incremental move picker: swap highest-scored move to position `idx`.
    /// O(N-idx) per call, but total work is O(k*N) where k << N due to beta cutoffs.
    #[inline]
    fn pick_next_move(moves: &mut [(ChessMove, i32)], idx: usize) {
        let mut best_idx = idx;
        let mut best_score = moves[idx].1;
        for i in (idx + 1)..moves.len() {
            if moves[i].1 > best_score {
                best_score = moves[i].1;
                best_idx = i;
            }
        }
        if best_idx != idx {
            moves.swap(idx, best_idx);
        }
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
