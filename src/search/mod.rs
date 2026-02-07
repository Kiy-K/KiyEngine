use crate::engine::Engine;
use crate::uci::MoveCodec;
use chess::{Board, ChessMove, Color, MoveGen, Piece, Square};
use parking_lot::Mutex;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

// =============================================================================
// CONSTANTS & TYPES
// =============================================================================

const MATE_SCORE: i32 = 30_000;
const INFINITY: i32 = 32_000;
const MAX_PLY: usize = 128;
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

const TT_BUCKET_SIZE: usize = 4;

#[derive(Clone, Debug)]
struct TTBucket {
    entries: [Option<TTEntry>; TT_BUCKET_SIZE],
}

impl TTBucket {
    fn new() -> Self {
        Self {
            entries: [None; TT_BUCKET_SIZE],
        }
    }
}

// =============================================================================
// TRANSPOSITION TABLE
// =============================================================================

pub struct TranspositionTable {
    buckets: Vec<Mutex<TTBucket>>,
    mask: usize,
}

impl TranspositionTable {
    pub fn new(size_mb: usize) -> Self {
        // Default 512MB: ~16M entries * ~32 bytes, bucketed
        // Tính toán số lượng bucket dựa trên size_mb
        // size_of(TTEntry) ~ 24 bytes, Option padding -> 32 bytes
        // Bucket 4 entries -> 128 bytes
        let bytes = size_mb * 1024 * 1024;
        let buckets_estimate = bytes / std::mem::size_of::<TTBucket>();
        let num_buckets = buckets_estimate.max(1).next_power_of_two();
        let buckets = (0..num_buckets)
            .map(|_| Mutex::new(TTBucket::new()))
            .collect();
        Self {
            buckets,
            mask: num_buckets - 1,
        }
    }

    pub fn resize(&self, mb: usize) {
        // Giả lập resize bằng cách clear (để giữ thread safety đơn giản)
        let _ = mb;
        self.clear();
    }

    pub fn get(&self, hash: u64) -> Option<TTEntry> {
        let idx = (hash as usize) & self.mask;
        // Vì buckets là Vec<Mutex>, ta không cần bounds check nếu mask đúng
        if let Some(bucket_mutex) = self.buckets.get(idx) {
            let bucket = bucket_mutex.lock();
            for entry in bucket.entries.iter() {
                if let Some(e) = entry {
                    if e.hash == hash {
                        return Some(*e);
                    }
                }
            }
        }
        None
    }

    pub fn store(&self, entry: TTEntry) {
        let idx = (entry.hash as usize) & self.mask;
        if let Some(bucket_mutex) = self.buckets.get(idx) {
            let mut bucket = bucket_mutex.lock();

            // 1. Thay thế slot trống
            for slot in bucket.entries.iter_mut() {
                if slot.is_none() {
                    *slot = Some(entry);
                    return;
                }
            }

            // 2. Thay thế slot cùng hash (update)
            for slot in bucket.entries.iter_mut() {
                if let Some(e) = slot {
                    if e.hash == entry.hash {
                        // Chỉ overwrite nếu depth mới >= depth cũ
                        if entry.depth >= e.depth {
                            *slot = Some(entry);
                        }
                        return;
                    }
                }
            }

            // 3. Thay thế slot có depth thấp nhất (Always replace shallowest)
            let mut replace_idx = 0;
            let mut lowest_depth = u8::MAX;
            for (i, slot) in bucket.entries.iter().enumerate() {
                if let Some(e) = slot {
                    if e.depth < lowest_depth {
                        lowest_depth = e.depth;
                        replace_idx = i;
                    }
                }
            }
            bucket.entries[replace_idx] = Some(entry);
        }
    }

    pub fn clear(&self) {
        for bucket in &self.buckets {
            let mut b = bucket.lock();
            for slot in b.entries.iter_mut() {
                *slot = None;
            }
        }
    }
}

// Module eval local (fallback)
pub mod eval;
pub mod time;
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
            threads: 1,
            move_overhead: 30, // Default 30ms overhead
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

            let mut handles = vec![];

            for i in 0..num_threads {
                let engine_clone = Arc::clone(&engine);
                let tt_clone = Arc::clone(&tt);
                let board_clone = board.clone();
                let history_clone = move_history.clone();
                let stop_clone = Arc::clone(&stop_flag);
                let tm_clone = Arc::clone(&time_manager);
                let tx_clone = if i == 0 { Some(tx.clone()) } else { None };
                let is_main = i == 0;

                handles.push(std::thread::spawn(move || {
                    let mut worker = SearchWorker::new(
                        engine_clone,
                        tt_clone,
                        board_clone,
                        history_clone,
                        stop_clone,
                    );
                    let mut best_move = None;
                    let mut last_score = 0;
                    let start_time = std::time::Instant::now();

                    // Lazy SMP: Thread chính chạy full depth, Thread phụ chạy depth-1 để feed TT
                    let local_max_depth = if is_main {
                        max_depth
                    } else {
                        max_depth.saturating_sub(1)
                    };

                    for depth in 1..=local_max_depth {
                        if worker.stop_flag.load(Ordering::Relaxed) || tm_clone.should_stop() {
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

                            if worker.stop_flag.load(Ordering::Relaxed) || tm_clone.should_stop() {
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
    engine: Arc<Engine>,
    tt: Arc<TranspositionTable>,
    board: Board,
    move_history: Vec<usize>,
    stop_flag: Arc<AtomicBool>,
    pub nodes: u64,
    killers: [[Option<ChessMove>; 2]; 128],
    history: [[[i32; 64]; 64]; 2],
}

impl SearchWorker {
    pub fn new(
        engine: Arc<Engine>,
        tt: Arc<TranspositionTable>,
        board: Board,
        move_history: Vec<usize>,
        stop_flag: Arc<AtomicBool>,
    ) -> Self {
        Self {
            engine,
            tt,
            board,
            move_history,
            stop_flag,
            nodes: 0,
            killers: [[None; 2]; 128],
            history: [[[0; 64]; 64]; 2],
        }
    }

    /// Đánh giá vị trí sử dụng Value Head (Smart Value)
    /// Policy head bị bỏ qua ở đây.
    fn evaluate_with_network(&mut self, ply: usize) -> i32 {
        // Forward pass qua BitNet
        if let Ok((_policy, value_score)) = self.engine.forward(&self.move_history) {
            // value_score [-1.0, 1.0] -> Centipawns
            let cp_score = (value_score * 10000.0) as i32;

            // Chuyển đổi góc nhìn (Engine trả về điểm của Trắng)
            if self.board.side_to_move() == Color::Black {
                -cp_score
            } else {
                cp_score
            }
        } else {
            // Fallback về heuristic đếm quân nếu mạng lỗi
            eval::evaluate(&self.board, ply)
        }
    }

    fn alpha_beta(
        &mut self,
        mut alpha: i32,
        mut beta: i32,
        mut depth: u8,
        ply: usize,
    ) -> (i32, Option<ChessMove>) {
        if self.stop_flag.load(Ordering::Relaxed) {
            return (0, None);
        }

        // Check Extension: Nếu bị chiếu, tăng depth
        let in_check = self.board.checkers().popcnt() > 0;
        if in_check && depth < MAX_PLY as u8 {
            depth += 1;
        }
        // Ensure ply index stays within killers/history bounds
        let ply_idx = usize::min(ply, self.killers.len() - 1);

        // 1. TT Lookup
        let hash = self.board.get_hash();
        let tt_entry = self.tt.get(hash);
        if let Some(ref entry) = tt_entry {
            if entry.depth >= depth {
                match entry.flag {
                    TTFlag::Exact => return (entry.score, entry.best_move),
                    TTFlag::LowerBound => {
                        if entry.score >= beta {
                            return (entry.score, entry.best_move);
                        }
                        alpha = alpha.max(entry.score);
                    }
                    TTFlag::UpperBound => {
                        if entry.score <= alpha {
                            return (entry.score, entry.best_move);
                        }
                        beta = beta.min(entry.score);
                    }
                }
            }
        }

        // 2. Leaf Node: Gọi Quiescence Search
        if depth == 0 {
            return (self.quiescence(alpha, beta, ply), None);
        }
        self.nodes += 1;

        // 3. Neural Policy Lookup (Chỉ dùng để Sort)
        // Chỉ gọi Policy ở ply thấp hoặc PV node để tiết kiệm chi phí
        let is_pv = beta - alpha > 1;
        let mut policy = None;
        if ply <= 6 && (is_pv || ply == 0) {
            if let Ok(res) = self.engine.forward(&self.move_history) {
                // Giả sử res.0 là tensor policy, convert sang vector
                // Cần implement trait hoặc method to_vec1 cho tensor output
                policy = res.0.to_vec1::<f32>().ok();
            }
        }

        let tt_move = tt_entry.as_ref().and_then(|e| e.best_move);
        let moves = self.order_moves(
            MoveGen::new_legal(&self.board),
            policy.as_deref(),
            ply,
            tt_move,
        );

        if moves.is_empty() {
            return if in_check {
                (-MATE_SCORE + ply as i32, None)
            } else {
                (0, None) // Stalemate
            };
        }

        let mut best_move = None;
        let mut best_score = -INFINITY;
        let mut moves_searched = 0;

        for (mv, _) in moves {
            let old_board = self.board.clone();
            self.board = self.board.make_move_new(mv);
            self.move_history
                .push(MoveCodec::move_to_token(&mv) as usize);
