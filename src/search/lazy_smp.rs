use crate::engine::Engine;
use crate::eval::pst::PST;
use crate::search::tt::{AtomicTT, TTFlag};
use chess::{BitBoard, Board, ChessMove, Color, MoveGen, Piece};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const INF: i32 = 32000;
const MATE: i32 = 30000;
const MAX_DEPTH: usize = 64;

// Optimized FastEval with PST
struct FastEval;

impl FastEval {
    #[inline(always)]
    fn evaluate(board: &Board) -> i32 {
        let mut score = 0;
        let material = [
            (Piece::Pawn, 100),
            (Piece::Knight, 320),
            (Piece::Bishop, 330),
            (Piece::Rook, 500),
            (Piece::Queen, 900),
            (Piece::King, 20000),
        ];

        for &color in &[Color::White, Color::Black] {
            let mult = if color == Color::White { 1 } else { -1 };
            let color_bb = board.color_combined(color);

            for &(piece, val) in &material {
                if piece == Piece::King {
                    continue;
                }
                let bb = board.pieces(piece) & color_bb;
                for sq in bb {
                    score += val * mult;
                    score += PST::get_score(piece, sq, color) * mult;
                }
            }
            let ksq = board.king_square(color);
            score += PST::get_score(Piece::King, ksq, color) * mult;
        }

        let side = board.side_to_move();
        if side == Color::White {
            score += 10;
        } else {
            score -= 10;
        }

        score
    }
}

pub struct Searcher {
    engine: Arc<Engine>,
    tt: Arc<AtomicTT>,
    stop: Arc<AtomicBool>,
    nodes: Arc<AtomicU64>,
    pub threads: usize,
    pub move_overhead: u32,
}

impl Searcher {
    pub fn new(engine: Arc<Engine>, tt: Arc<AtomicTT>) -> Self {
        Self {
            engine,
            tt,
            stop: Arc::new(AtomicBool::new(false)),
            nodes: Arc::new(AtomicU64::new(0)),
            threads: 1,
            move_overhead: 30,
        }
    }

    pub fn search_async(
        &self,
        board: Board,
        history: Vec<usize>,
        position_history: Vec<u64>, // NEW: Zobrist History
        max_depth: u8,
        wtime: u64,
        btime: u64,
        winc: u64,
        binc: u64,
        _movestogo: u64,
        stop_flag: Arc<AtomicBool>,
        tx: Sender<String>,
    ) {
        let engine = self.engine.clone();
        let tt = self.tt.clone();
        let nodes = self.nodes.clone();
        let threads = self.threads;

        let side = board.side_to_move();
        let (time_left, inc) = if side == Color::White {
            (wtime, winc)
        } else {
            (btime, binc)
        };

        // Time management
        let mut search_time = time_left / 20 + inc;
        let overhead = self.move_overhead as u64;
        if search_time > time_left {
            search_time = time_left.saturating_sub(overhead + 50);
        }
        if search_time < 10 {
            search_time = 50;
        }

        let is_infinite = wtime == 0 && btime == 0 && max_depth == 0;

        let self_stop = self.stop.clone();
        let pos_hist_arc = Arc::new(position_history); // Share history

        thread::spawn(move || {
            self_stop.store(false, Ordering::Relaxed);

            let strategic_bias = match engine.forward(&history) {
                Ok((_, val_tanh)) => {
                    let nn_eval = (val_tanh * 2000.0) as i32;
                    let static_eval = FastEval::evaluate(&board);
                    let bias = nn_eval - static_eval;
                    bias.clamp(-1000, 1000)
                }
                Err(_) => 0,
            };

            if !is_infinite {
                let timer_stop = self_stop.clone();
                let user_stop = stop_flag.clone();
                thread::spawn(move || {
                    let start = Instant::now();
                    while (start.elapsed().as_millis() as u64) < search_time {
                        if user_stop.load(Ordering::Relaxed) {
                            timer_stop.store(true, Ordering::Relaxed);
                            return;
                        }
                        thread::sleep(Duration::from_millis(5));
                    }
                    timer_stop.store(true, Ordering::Relaxed);
                });
            }

            let start_time = Instant::now();
            let mut handles = vec![];

            for id in 0..threads {
                let tt = tt.clone();
                let stop = self_stop.clone();
                let user_stop = stop_flag.clone();
                let nodes_counter = nodes.clone();
                let board = board.clone();
                let ph = pos_hist_arc.clone();
                let tx_clone = if id == 0 { Some(tx.clone()) } else { None };

                handles.push(thread::spawn(move || {
                    let mut worker = Worker::new(
                        id,
                        tt,
                        stop,
                        user_stop,
                        nodes_counter,
                        board,
                        ph,
                        strategic_bias,
                    );
                    worker.iterative_deepening(start_time, id == 0, max_depth, tx_clone);
                }));
            }

            for h in handles {
                h.join().unwrap();
            }
        });
    }
}

struct MoveList {
    moves: [ChessMove; 256],
    scores: [i32; 256],
    len: usize,
}

impl MoveList {
    fn new() -> Self {
        Self {
            moves: [ChessMove::default(); 256],
            scores: [0; 256],
            len: 0,
        }
    }

    #[inline(always)]
    fn push(&mut self, m: ChessMove, s: i32) {
        if self.len < 256 {
            self.moves[self.len] = m;
            self.scores[self.len] = s;
            self.len += 1;
        }
    }

    #[inline(always)]
    fn pick(&mut self, index: usize) -> Option<(ChessMove, i32)> {
        if index >= self.len {
            return None;
        }

        let mut best_idx = index;
        let mut best_score = self.scores[index];

        for i in (index + 1)..self.len {
            if self.scores[i] > best_score {
                best_score = self.scores[i];
                best_idx = i;
            }
        }

        if best_idx != index {
            self.scores.swap(index, best_idx);
            self.moves.swap(index, best_idx);
        }

        Some((self.moves[index], self.scores[index]))
    }
}

pub struct Worker {
    _id: usize,
    tt: Arc<AtomicTT>,
    stop: Arc<AtomicBool>,
    user_stop: Arc<AtomicBool>,
    nodes_counter: Arc<AtomicU64>,
    board: Board,
    game_history: Arc<Vec<u64>>,
    path_history: Vec<u64>,
    strategic_bias: i32,
    ply: usize,
    nodes: u64,
    killers: [[Option<ChessMove>; 2]; MAX_DEPTH],
    history: [i32; 4096],
}

impl Worker {
    fn new(
        id: usize,
        tt: Arc<AtomicTT>,
        stop: Arc<AtomicBool>,
        user_stop: Arc<AtomicBool>,
        nodes_counter: Arc<AtomicU64>,
        board: Board,
        game_history: Arc<Vec<u64>>,
        strategic_bias: i32,
    ) -> Self {
        Self {
            _id: id,
            tt,
            stop,
            user_stop,
            nodes_counter,
            board,
            game_history,
            path_history: Vec::with_capacity(64),
            strategic_bias,
            ply: 0,
            nodes: 0,
            killers: [[None; 2]; MAX_DEPTH],
            history: [0; 4096],
        }
    }

    fn check_stop(&self) -> bool {
        self.nodes & 2047 == 0
            && (self.stop.load(Ordering::Relaxed) || self.user_stop.load(Ordering::Relaxed))
    }

    fn is_repetition(&self, hash: u64) -> bool {
        // Check current search path
        for &h in &self.path_history {
            if h == hash {
                return true;
            }
        }
        // Check game history
        let mut count = 0;
        for &h in self.game_history.iter() {
            if h == hash {
                count += 1;
            }
        }
        // If count >= 2, we are at 3rd repetition now.
        if count >= 2 {
            return true;
        }

        false
    }

    fn iterative_deepening(
        &mut self,
        start_time: Instant,
        is_main: bool,
        max_depth: u8,
        tx: Option<Sender<String>>,
    ) {
        let mut best_move = None;
        let mut depth = 1;
        let limit = if max_depth > 0 { max_depth } else { 64 };
        let mut _last_node_count = 0u64;

        loop {
            if self.check_stop() {
                break;
            }

            // Aspiration Window (Basic)
            self.path_history.clear(); // Root is start of path
            let nodes_before = self.nodes;
            let (score, mv) = self.alpha_beta(-INF, INF, depth, true);
            let nodes_searched = self.nodes - nodes_before;

            if self.check_stop() {
                break;
            }

            // Validate that we actually searched nodes at this depth
            // If nodes_searched is 0, it means we got a TT cutoff without searching
            // This prevents "fake depth" reporting
            let actual_depth = if nodes_searched > 0 {
                depth
            } else {
                // Find the highest depth that actually searched nodes
                depth - 1
            };

            if is_main {
                let elapsed = start_time.elapsed().as_secs_f64();
                let global_nodes = self.nodes_counter.load(Ordering::Relaxed);
                let nps = if elapsed > 0.001 {
                    (global_nodes as f64 / elapsed) as u64
                } else {
                    0
                };

                let mv_str = mv.map(|m| format!("{}", m)).unwrap_or("-".to_string());
                // Clamp score to reasonable range for UCI output
                let display_score = score.clamp(-10000, 10000);
                if let Some(ref sender) = tx {
                    let _ = sender.send(format!(
                        "info depth {} score cp {} nodes {} nps {} pv {}",
                        actual_depth, display_score, global_nodes, nps, mv_str
                    ));
                }
                best_move = mv;

                // Track node count to detect stagnation
                _last_node_count = global_nodes;
            }

            depth += 1;
            if depth > limit {
                break;
            }
        }

        if is_main {
            let mv_str = best_move
                .map(|m| format!("{}", m))
                .unwrap_or("0000".to_string());
            if let Some(ref sender) = tx {
                let _ = sender.send(format!("bestmove {}", mv_str));
            }
        }
    }

    fn alpha_beta(
        &mut self,
        mut alpha: i32,
        mut beta: i32,
        mut depth: u8,
        allow_null: bool,
    ) -> (i32, Option<ChessMove>) {
        if self.nodes & 2047 == 0 {
            self.nodes_counter.fetch_add(2048, Ordering::Relaxed);
        }
        self.nodes += 1;
        if self.check_stop() {
            return (0, None);
        }

        // Repetition Check (except at root ply=0)
        let hash = self.board.get_hash();
        if self.ply > 0 && self.is_repetition(hash) {
            return (0, None);
        }

        if self.ply >= MAX_DEPTH {
            return (FastEval::evaluate(&self.board) + self.strategic_bias, None);
        }

        // TT Probe with strict depth validation
        let tt_entry = self.tt.probe(hash);
        if let Some((tt_score, tt_move, tt_depth, tt_flag)) = tt_entry {
            // Only use TT entry if it was searched to at least current depth
            // This prevents "fake depth" where engine reports high depth without searching
            if tt_depth >= depth {
                match tt_flag {
                    TTFlag::Exact => {
                        // Even with TT hit, count this as a node for statistics
                        self.nodes += 1;
                        return (tt_score, tt_move);
                    }
                    TTFlag::LowerBound => alpha = alpha.max(tt_score),
                    TTFlag::UpperBound => beta = beta.min(tt_score),
                    TTFlag::None => {}
                }
                if alpha >= beta {
                    // Even with TT hit, count this as a node for statistics
                    self.nodes += 1;
                    return (tt_score, tt_move);
                }
            }
            // If tt_depth < depth, use tt_move for move ordering but don't cutoff
        }

        let in_check = self.board.checkers().popcnt() > 0;

        if in_check {
            depth += 1;
        }

        if depth == 0 {
            let eval = self.quiescence(alpha, beta);
            return (eval, None);
        }

        // Null Move Pruning
        if allow_null && !in_check && depth >= 3 {
            let eval = FastEval::evaluate(&self.board) + self.strategic_bias;
            if eval - 50 * (depth as i32) >= beta {
                return (beta, None);
            }
        }

        let mut move_list = MoveList::new();
        let moves = MoveGen::new_legal(&self.board);
        let targets = *self.board.color_combined(!self.board.side_to_move());

        let tt_move_val = if let Some((_, tm, _, _)) = tt_entry {
            tm
        } else {
            None
        };

        for m in moves {
            let score;
            if Some(m) == tt_move_val {
                score = 100000;
            } else if (targets & BitBoard::from_square(m.get_dest())).popcnt() > 0 {
                score = 10000 + Self::mvv_lva(&self.board, m);
            } else {
                if self.killers[self.ply][0] == Some(m) {
                    score = 9000;
                } else if self.killers[self.ply][1] == Some(m) {
                    score = 8000;
                } else {
                    let from = m.get_source().to_index() as usize;
                    let to = m.get_dest().to_index() as usize;
                    score = self.history[from * 64 + to];
                }
            }
            move_list.push(m, score);
        }

        if move_list.len == 0 {
            if in_check {
                return (-MATE + self.ply as i32, None);
            }
            return (0, None);
        }

        let mut best_move = None;
        let mut best_score = -INF;
        let mut flag = TTFlag::UpperBound;
        let mut moves_searched = 0;

        // Push hash to path history
        self.path_history.push(hash);

        for i in 0..move_list.len {
            let (m, _) = move_list.pick(i).unwrap();

            let old_board = self.board;
            self.board = self.board.make_move_new(m);
            self.ply += 1;

            let mut score;

            // PVS
            if moves_searched == 0 {
                let (val, _) = self.alpha_beta(-beta, -alpha, depth - 1, true);
                score = -val;
            } else {
                let is_capture = (targets & BitBoard::from_square(m.get_dest())).popcnt() > 0;
                let is_promotion = m.get_promotion().is_some();
                let give_check = self.board.checkers().popcnt() > 0;

                let mut r = 0;
                if depth >= 3 && !is_capture && !is_promotion && !in_check && !give_check {
                    r = 1 + depth / 3;
                    if moves_searched > 6 {
                        r += 1;
                    }
                }

                let (val, _) = self.alpha_beta(-alpha - 1, -alpha, depth - 1 - r, true);
                score = -val;

                if r > 0 && score > alpha {
                    let (val, _) = self.alpha_beta(-alpha - 1, -alpha, depth - 1, true);
                    score = -val;
                }

                if score > alpha && score < beta {
                    let (val, _) = self.alpha_beta(-beta, -alpha, depth - 1, true);
                    score = -val;
                }
            }

            self.ply -= 1;
            self.board = old_board;

            if self.check_stop() {
                self.path_history.pop();
                return (0, None);
            }
            moves_searched += 1;

            if score > best_score {
                best_score = score;
                best_move = Some(m);
            }

            if score > alpha {
                alpha = score;
                flag = TTFlag::Exact;

                if (targets & BitBoard::from_square(m.get_dest())).popcnt() == 0 {
                    let d = self.ply;
                    if self.killers[d][0] != Some(m) {
                        self.killers[d][1] = self.killers[d][0];
                        self.killers[d][0] = Some(m);
                    }
                    let from = m.get_source().to_index() as usize;
                    let to = m.get_dest().to_index() as usize;
                    self.history[from * 64 + to] += (depth as i32) * (depth as i32);
                    if self.history[from * 64 + to] > 20000 {
                        for h in self.history.iter_mut() {
                            *h /= 2;
                        }
                    }
                }

                if score >= beta {
                    self.path_history.pop(); // Restore stack
                    self.tt
                        .store(hash, score, Some(m), depth, TTFlag::LowerBound, 0);
                    return (score, Some(m));
                }
            }
        }

        self.path_history.pop(); // Restore stack

        self.tt.store(hash, best_score, best_move, depth, flag, 0);
        (best_score, best_move)
    }

    fn quiescence(&mut self, mut alpha: i32, beta: i32) -> i32 {
        self.nodes += 1;
        if self.nodes & 2047 == 0 {
            self.nodes_counter.fetch_add(2048, Ordering::Relaxed);
        }

        let eval = FastEval::evaluate(&self.board) + self.strategic_bias;

        if eval >= beta {
            return eval;
        }
        if eval > alpha {
            alpha = eval;
        }

        let targets = *self.board.color_combined(!self.board.side_to_move());
        let moves = MoveGen::new_legal(&self.board);

        let mut move_list = MoveList::new();
        for m in moves {
            if (targets & BitBoard::from_square(m.get_dest())).popcnt() > 0 {
                let score = Self::mvv_lva(&self.board, m);
                move_list.push(m, score);
            }
        }

        for i in 0..move_list.len {
            let (m, _) = move_list.pick(i).unwrap();

            let old_board = self.board;
            self.board = self.board.make_move_new(m);
            let score = -self.quiescence(-beta, -alpha);
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

    fn mvv_lva(board: &Board, m: ChessMove) -> i32 {
        let victim = board.piece_on(m.get_dest()).unwrap_or(Piece::Pawn);
        let attacker = board.piece_on(m.get_source()).unwrap_or(Piece::Pawn);
        let victim_val = match victim {
            Piece::Pawn => 100,
            Piece::Knight => 300,
            Piece::Bishop => 300,
            Piece::Rook => 500,
            Piece::Queen => 900,
            Piece::King => 0,
        };
        let attacker_val = match attacker {
            Piece::Pawn => 100,
            Piece::Knight => 300,
            Piece::Bishop => 300,
            Piece::Rook => 500,
            Piece::Queen => 900,
            Piece::King => 20000,
        };
        victim_val * 10 - attacker_val
    }
}
