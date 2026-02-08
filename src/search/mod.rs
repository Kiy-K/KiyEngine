use crate::engine::Engine;
use crate::uci::MoveCodec;
use chess::{BitBoard, Board, ChessMove, Color, MoveGen, Piece};
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
                table[d][m] = (0.75 + (df.ln() * mf.ln() / 2.0)) as u8;
            }
        }
        table
    });
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

    pub fn resize(&self, _mb: usize) {
        self.clear();
    }

    #[inline(always)]
    pub fn get(&self, hash: u64) -> Option<TTEntry> {
        let idx = (hash as usize) & self.mask;
        let base = idx * 2;
        let d = self.slots[base + 1].load(Ordering::Relaxed);
        if d == 0 { return None; }
        let k = self.slots[base].load(Ordering::Relaxed);
        if k ^ d != hash { return None; }
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
                Some(chess::Piece::Rook)   => 3,
                Some(chess::Piece::Queen)  => 4,
                _ => 0,
            };
            d |= from | (to << 6) | (promo << 12);
        }
        let score_u16 = entry.score.clamp(-32768, 32767) as i16 as u16;
        d |= (score_u16 as u64) << 16;
        d |= (entry.depth as u64) << 32;
        let flag_bits: u64 = match entry.flag {
            TTFlag::Exact      => 0,
            TTFlag::LowerBound => 1,
            TTFlag::UpperBound => 2,
        };
        d |= flag_bits << 40;
        d |= 1u64 << 63; // sentinel: never zero
        d
    }

    fn unpack(hash: u64, d: u64) -> Option<TTEntry> {
        if d == 0 { return None; }
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
        Some(TTEntry { hash, depth, score, flag, best_move })
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
}

impl Searcher {
    pub fn new(engine: Arc<Engine>, tt: Arc<TranspositionTable>) -> Self {
        Self {
            engine,
            tt,
            threads: 4,
            move_overhead: 60, // Default 60ms overhead
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
    ) {
        let engine = Arc::clone(&self.engine);
        let tt = Arc::clone(&self.tt);
        let num_threads = self.threads;
        let move_overhead = self.move_overhead;
        let is_white = board.side_to_move() == Color::White;

        std::thread::spawn(move || {
            let time_manager = Arc::new(TimeManager::new(
                wtime,
                btime,
                winc,
                binc,
                movestogo,
                move_overhead,
                is_white,
            ));

            // Pre-compute NN policy ONCE, then share across all threads
            let shared_policy: Arc<Option<(Vec<f32>, f32)>> = {
                if let Ok(res) = engine.forward(&move_history) {
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

                handles.push(std::thread::spawn(move || {
                    let mut worker = SearchWorker::new(
                        tt_clone,
                        board_clone,
                        stop_clone,
                    );
                    let mut best_move = None;
                    let mut last_score = 0;
                    let start_time = std::time::Instant::now();

                    // Use shared pre-computed NN policy (no per-thread forward pass)
                    worker.set_cached_policy(&*policy_clone);

                    // Wire time manager for in-search abort checks
                    let depth_only_check = max_depth > 0 && wtime == 0 && btime == 0;
                    if !depth_only_check {
                        worker.time_manager = Some(Arc::clone(&tm_clone));
                    }

                    // Lazy SMP: Thread chính chạy full depth, Thread phụ chạy depth-1 để feed TT
                    let effective_max = if max_depth == 0 { MAX_DEPTH } else { max_depth };
                    let local_max_depth = if is_main {
                        effective_max
                    } else {
                        effective_max.saturating_sub(1).max(1)
                    };

                    let depth_only = max_depth > 0 && wtime == 0 && btime == 0;
                    const MIN_DEPTH: u8 = 3; // Always complete at least depth 3

                    for depth in 1..=local_max_depth {
                        // Only allow time abort AFTER minimum depth is reached
                        if worker.stop_flag.load(Ordering::Relaxed) {
                            break;
                        }
                        if !depth_only && depth > MIN_DEPTH && tm_clone.should_stop() {
                            break;
                        }

                        // Aspiration Windows logic
                        let mut alpha = -INFINITY;
                        let mut beta = INFINITY;
                        if depth > 3 {
                            alpha = last_score - 50;
                            beta = last_score + 50;
                        }

                        loop {
                            let (score, mv) = worker.alpha_beta(alpha, beta, depth, 0);

                            if worker.stop_flag.load(Ordering::Relaxed) {
                                break;
                            }
                            if !depth_only && depth > MIN_DEPTH && tm_clone.should_stop() {
                                break;
                            }

                            if score <= alpha {
                                alpha = alpha.saturating_sub(150); // Widen window
                            } else if score >= beta {
                                beta = beta.saturating_add(150); // Widen window
                            } else {
                                if is_main {
                                    best_move = mv;
                                    last_score = score;
                                    let elapsed = start_time.elapsed().as_secs_f64().max(0.001);
                                    let nps = (worker.nodes as f64 / elapsed) as u64;

                                    // In thông tin UCI chuẩn
                                    let pv_str = if let Some(m) = mv {
                                        format!("{}", m)
                                    } else {
                                        "".to_string()
                                    };
                                    println!(
                                        "info depth {} score cp {} nodes {} nps {} pv {}",
                                        depth, score, worker.nodes, nps, pv_str
                                    );
                                }
                                break;
                            }
                            // Nếu window đã mở hết cỡ mà vẫn fail, break
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
                                let _ = tx_clone.unwrap().send("bestmove (none)".to_string());
                            }
                        }
                    }
                }));
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
    // Cached NN results — computed once, reused across all ID iterations
    cached_root_policy: Option<Vec<f32>>,
    cached_root_value: f32,
}

impl SearchWorker {
    pub fn new(
        tt: Arc<TranspositionTable>,
        board: Board,
        stop_flag: Arc<AtomicBool>,
    ) -> Self {
        Self {
            tt,
            board,
            stop_flag,
            time_manager: None,
            nodes: 0,
            killers: [[None; 2]; MAX_PLY],
            history: [[[0; 64]; 64]; 2],
            cached_root_policy: None,
            cached_root_value: 0.0,
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
    // FAST STATIC EVAL (Material + PST) — used everywhere except root policy.
    // =========================================================================
    #[inline]
    fn fast_eval(&self) -> i32 {
        eval::evaluate(&self.board, 0)
    }

    // =========================================================================
    // HELPER: check if a move is a capture
    // =========================================================================
    #[inline]
    fn is_capture(&self, mv: ChessMove) -> bool {
        let targets = *self.board.color_combined(!self.board.side_to_move());
        (targets & BitBoard::from_square(mv.get_dest())).popcnt() > 0
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
        non_pawn.popcnt() > 0
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
        // --- Step 1: Abort check (+ periodic time check every 2048 nodes) ---
        if self.nodes & 2047 == 0 {
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

        // --- Step 2: Check extension ---
        let in_check = self.board.checkers().popcnt() > 0;
        if in_check && depth < MAX_PLY as u8 {
            depth += 1;
        }

        let ply_idx = ply.min(MAX_PLY - 1);

        // --- Step 3: Mate Distance Pruning ---
        if !is_root {
            let mating_score = MATE_SCORE - ply as i32;
            let mated_score = -MATE_SCORE + ply as i32;
            if alpha >= mating_score { return (alpha, None); }
            if beta <= mated_score { return (beta, None); }
            alpha = alpha.max(mated_score);
            beta = beta.min(mating_score);
            if alpha >= beta { return (alpha, None); }
        }

        // --- Step 4: TT Lookup ---
        let hash = self.board.get_hash();
        let tt_entry = self.tt.get(hash);
        let tt_move = tt_entry.as_ref().and_then(|e| e.best_move);

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
        let static_eval = if in_check { -INFINITY } else { self.fast_eval() };

        // --- Step 7: Razoring (Stockfish-style) ---
        // If eval is far below alpha, drop into qsearch
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
        // If our position is so good that even subtracting a margin still >= beta
        if !is_pv && !in_check && depth <= 7
            && static_eval - 80 * (depth as i32) >= beta
            && static_eval < MATE_SCORE - 100
        {
            return (static_eval, None);
        }

        // --- Step 9: Null Move Pruning (approximation) ---
        // chess crate doesn't support null moves directly, so we use
        // an enhanced static pruning: if eval >= beta + depth-scaled margin, prune.
        if !is_pv && !in_check && depth >= 3
            && static_eval >= beta
            && self.has_non_pawn_material()
        {
            let null_margin = 120 * (depth as i32) / 3;
            if static_eval - null_margin >= beta {
                return (beta, None);
            }
        }

        // --- Step 10: Internal Iterative Reduction (IIR) ---
        // If no TT move found at this depth, reduce depth
        if depth >= 4 && tt_move.is_none() && !in_check {
            depth -= 1;
        }

        // --- Step 11: Neural Policy Lookup (root only, from cache — zero cost) ---
        let policy: Option<&[f32]> = if is_root {
            self.cached_root_policy.as_deref()
        } else {
            None
        };

        // --- Step 12: Generate and order moves ---
        let moves = self.order_moves(
            MoveGen::new_legal(&self.board),
            policy,
            ply,
            tt_move,
        );

        if moves.is_empty() {
            return if in_check {
                (-MATE_SCORE + ply as i32, None)
            } else {
                (0, None)
            };
        }

        let original_alpha = alpha;
        let mut best_move = None;
        let mut best_score = -INFINITY;
        let mut moves_searched = 0;

        for (mv, _move_score) in &moves {
            let mv = *mv;
            let is_cap = self.is_capture(mv);
            let is_promo = mv.get_promotion().is_some();
            let is_tactical = is_cap || is_promo;

            // --- Step 13: Late Move Pruning (LMP) ---
            // Skip quiet late moves at low depth
            if !is_root && !in_check && !is_tactical && depth <= 5 {
                let lmp_threshold = (3 + (depth as usize) * (depth as usize)) / 2;
                if moves_searched >= lmp_threshold {
                    continue;
                }
            }

            // --- Step 14: Futility Pruning ---
            // At frontier/pre-frontier nodes, skip quiet moves if eval + margin < alpha
            if !is_root && !in_check && !is_tactical && depth <= 6 && moves_searched > 0 {
                let futility_margin = static_eval + 100 + 120 * (depth as i32);
                if futility_margin <= alpha {
                    continue;
                }
            }

            // --- Step 15: SEE Pruning for bad captures ---
            // Skip captures where we lose material (e.g., QxP defended by pawn)
            if !is_root && is_cap && !is_promo && depth <= 4 && moves_searched > 0 {
                let victim = self.board.piece_on(mv.get_dest()).unwrap_or(Piece::Pawn);
                let attacker = self.board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
                // Simple SEE approximation: skip if attacker > victim + margin
                if piece_value(attacker) > piece_value(victim) + 200 {
                    continue;
                }
            }

            // --- Step 16: Make move ---
            let old_board = self.board;
            self.board = self.board.make_move_new(mv);

            let gives_check = self.board.checkers().popcnt() > 0;

            let mut score;

            // --- Step 17: PVS (Principal Variation Search) ---
            if moves_searched == 0 {
                let (val, _) = self.alpha_beta(-beta, -alpha, depth - 1, ply + 1);
                score = -val;
            } else {
                // --- Step 18: Late Move Reductions (LMR) ---
                // Conditions: depth >= 3, not in check, not PV node
                // Quiet moves (non-capture, non-promo, non-check) get
                // heavier reduction after move 4.
                let mut r: u8 = 0;
                if depth >= 3 && !in_check && !is_pv {
                    if is_tactical || gives_check {
                        // Tactical moves: only light reduction after many moves
                        if moves_searched >= 6 {
                            r = 1;
                        }
                    } else {
                        // Quiet moves: use pre-computed LMR table
                        if moves_searched > 4 {
                            let di = (depth as usize).min(LMR_MAX_DEPTH - 1);
                            let mi = moves_searched.min(LMR_MAX_MOVES - 1);
                            r = LMR_TABLE[di][mi];
                        } else if moves_searched >= 2 {
                            r = 1;
                        }
                    }
                    // Reduce less for killer moves
                    if self.killers[ply_idx][0] == Some(mv)
                        || self.killers[ply_idx][1] == Some(mv)
                    {
                        r = r.saturating_sub(1);
                    }
                    // Never reduce into qsearch (keep at least depth 1)
                    r = r.min(depth.saturating_sub(2));
                }

                // Null-window search with reduction
                let (val, _) = self.alpha_beta(
                    -alpha - 1,
                    -alpha,
                    depth.saturating_sub(1 + r),
                    ply + 1,
                );
                score = -val;

                // Re-search at full depth if LMR failed high
                if r > 0 && score > alpha {
                    let (val, _) = self.alpha_beta(-alpha - 1, -alpha, depth - 1, ply + 1);
                    score = -val;
                }

                // Full PV re-search if null-window failed high
                if score > alpha && score < beta {
                    let (val, _) = self.alpha_beta(-beta, -alpha, depth - 1, ply + 1);
                    score = -val;
                }
            }

            // --- Step 19: Undo move ---
            self.board = old_board;

            if self.stop_flag.load(Ordering::Relaxed) {
                return (0, None);
            }

            moves_searched += 1;

            // --- Step 20: Update best ---
            if score > best_score {
                best_score = score;
                best_move = Some(mv);
            }

            if score > alpha {
                alpha = score;

                if score >= beta {
                    // --- Step 21: Beta cutoff → update killers & history ---
                    if !is_cap {
                        self.killers[ply_idx][1] = self.killers[ply_idx][0];
                        self.killers[ply_idx][0] = Some(mv);
                    }
                    let side = if self.board.side_to_move() == Color::White { 0 } else { 1 };
                    let from = mv.get_source().to_index();
                    let to = mv.get_dest().to_index();
                    self.history[side][from][to] += (depth as i32) * (depth as i32);
                    // Prevent history overflow
                    if self.history[side][from][to] > 30000 {
                        for s in 0..2 {
                            for f in 0..64 {
                                for t in 0..64 {
                                    self.history[s][f][t] /= 2;
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
    // QUIESCENCE SEARCH with Delta Pruning
    // =========================================================================
    fn quiescence(&mut self, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        self.nodes += 1;

        // Abort check: prevent infinite qsearch and respect time limits
        if ply >= MAX_PLY {
            return self.fast_eval();
        }
        if self.nodes & 2047 == 0 {
            if let Some(ref tm) = self.time_manager {
                if tm.should_stop() {
                    self.stop_flag.store(true, Ordering::Relaxed);
                }
            }
        }
        if self.stop_flag.load(Ordering::Relaxed) {
            return 0;
        }

        // Use fast static eval in quiescence (NN is too slow per-node)
        let stand_pat = self.fast_eval();

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

        let targets = *self.board.color_combined(!self.board.side_to_move());
        let mut moves = MoveGen::new_legal(&self.board);
        moves.set_iterator_mask(targets);

        // Collect and sort captures by MVV-LVA
        let mut captures: Vec<(ChessMove, i32)> = Vec::with_capacity(16);
        for mv in moves {
            let victim = self.board.piece_on(mv.get_dest()).unwrap_or(Piece::Pawn);
            let attacker = self.board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
            let see_score = piece_value(victim) * 10 - piece_value(attacker);
            captures.push((mv, see_score));
        }
        captures.sort_by(|a, b| b.1.cmp(&a.1));

        for (mv, _) in captures {
            // Delta pruning per-move: skip if captured piece can't raise alpha
            let victim = self.board.piece_on(mv.get_dest()).unwrap_or(Piece::Pawn);
            if stand_pat + piece_value(victim) + 200 < alpha && mv.get_promotion().is_none() {
                continue;
            }

            let old_board = self.board;
            self.board = self.board.make_move_new(mv);

            let score = -self.quiescence(-beta, -alpha, ply + 1);

            self.board = old_board;

            if score >= beta {
                return score;
            }
            if score > alpha {
                alpha = score;
            }
        }

        alpha
    }

    // =========================================================================
    // MOVE ORDERING: TT move > Policy > Captures (MVV-LVA) > Killers > History
    // =========================================================================
    fn order_moves(
        &self,
        moves: MoveGen,
        policy: Option<&[f32]>,
        ply: usize,
        tt_move: Option<ChessMove>,
    ) -> Vec<(ChessMove, f32)> {
        let ply_idx = ply.min(MAX_PLY - 1);
        let targets = *self.board.color_combined(!self.board.side_to_move());
        let mut scored: Vec<(ChessMove, f32)> = Vec::with_capacity(40);

        for mv in moves {
            let mut score: f32 = 0.0;

            // 1. TT move gets highest priority
            if Some(mv) == tt_move {
                score += 200_000.0;
            }

            // 2. Neural policy score (very high priority when available)
            if let Some(pol) = policy {
                let token = MoveCodec::get_policy_index(&mv);
                if token < pol.len() {
                    score += pol[token] * 15_000.0;
                }
            }

            // 3. Captures scored by MVV-LVA
            if (targets & BitBoard::from_square(mv.get_dest())).popcnt() > 0 {
                let victim = self.board.piece_on(mv.get_dest()).unwrap_or(Piece::Pawn);
                let attacker = self.board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
                let victim_val = piece_value(victim);
                let attacker_val = piece_value(attacker);
                score += (victim_val * 10 - attacker_val) as f32 + 50_000.0;
            }

            // 4. Promotions
            if mv.get_promotion().is_some() {
                score += 45_000.0;
            }

            // 5. Killer moves
            if self.killers[ply_idx][0] == Some(mv) {
                score += 40_000.0;
            } else if self.killers[ply_idx][1] == Some(mv) {
                score += 35_000.0;
            }

            // 6. History heuristic
            let side = if self.board.side_to_move() == Color::White { 0 } else { 1 };
            let from = mv.get_source().to_index();
            let to = mv.get_dest().to_index();
            score += self.history[side][from][to] as f32;

            scored.push((mv, score));
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
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
