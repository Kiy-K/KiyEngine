// src/search/mod.rs

pub mod eval;

use crate::engine::Engine;
use crate::uci::MoveCodec;
use chess::{Board, ChessMove, MoveGen, Piece, Color};
use dashmap::DashMap;
use std::sync::{Arc, mpsc};
use std::sync::atomic::{AtomicBool, Ordering};

const MATE_SCORE: i32 = 30000;
const INFINITY: i32 = 32000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TTFlag { Exact, LowerBound, UpperBound }

#[derive(Clone, Debug)]
pub struct TTEntry {
    pub hash: u64,
    pub depth: u8,
    pub score: i32,
    pub flag: TTFlag,
    pub best_move: Option<ChessMove>,
}

pub struct TranspositionTable {
    table: DashMap<u64, TTEntry>,
}

impl TranspositionTable {
    pub fn new() -> Self {
        Self { table: DashMap::with_capacity(16_000_000) }
    }
    pub fn get(&self, hash: u64) -> Option<TTEntry> {
        self.table.get(&hash).map(|r| r.clone())
    }
    pub fn store(&self, entry: TTEntry) {
        self.table.insert(entry.hash, entry);
    }
    pub fn clear(&self) {
        self.table.clear();
    }
}

pub struct Searcher {
    engine: Arc<Engine>,
    tt: Arc<TranspositionTable>,
}

impl Searcher {
    pub fn new(engine: Arc<Engine>, tt: Arc<TranspositionTable>) -> Self {
        Self { engine, tt }
    }

    pub fn search_async(
        &self,
        board: Board,
        move_history: Vec<usize>,
        max_depth: u8,
        stop_flag: Arc<AtomicBool>,
        tx: mpsc::Sender<String>,
    ) {
        let engine = Arc::clone(&self.engine);
        let tt = Arc::clone(&self.tt);
        
        std::thread::spawn(move || {
            let mut worker = SearchWorker::new(engine, tt, board, move_history, stop_flag);
            let mut best_move = None;
            let start_time = std::time::Instant::now();
            
            // Neural at Root
            let (root_policy, _root_eval) = match worker.engine.forward(&worker.move_history) {
                Ok(res) => (res.0.to_vec1::<f32>().ok(), Some((res.1 * 1000.0) as i32)),
                Err(_) => (None, None),
            };
            worker.root_policy = root_policy;

            for depth in 1..=max_depth {
                if worker.stop_flag.load(Ordering::Relaxed) { break; }
                
                let (score, mv) = worker.alpha_beta(-INFINITY, INFINITY, depth, 0);
                
                if !worker.stop_flag.load(Ordering::Relaxed) {
                    if let Some(m) = mv {
                        best_move = Some(m);
                        let elapsed = start_time.elapsed().as_secs_f64().max(0.001);
                        let nps = (worker.nodes as f64 / elapsed) as u64;
                        println!("info depth {} score cp {} nodes {} nps {} pv {}", depth, score, worker.nodes, nps, m);
                    }
                }
            }
            
            if let Some(mv) = best_move {
                let _ = tx.send(format!("bestmove {}", mv));
            } else {
                let mut moves = MoveGen::new_legal(&board);
                if let Some(mv) = moves.next() { let _ = tx.send(format!("bestmove {}", mv)); }
            }
        });
    }
}

pub struct SearchWorker {
    engine: Arc<Engine>,
    tt: Arc<TranspositionTable>,
    board: Board,
    move_history: Vec<usize>,
    stop_flag: Arc<AtomicBool>,
    pub nodes: u64,
    killers: [[Option<ChessMove>; 2]; 128],
    history: [[[i32; 64]; 64]; 2],
    root_policy: Option<Vec<f32>>,
}

impl SearchWorker {
    pub fn new(engine: Arc<Engine>, tt: Arc<TranspositionTable>, board: Board, move_history: Vec<usize>, stop_flag: Arc<AtomicBool>) -> Self {
        Self { engine, tt, board, move_history, stop_flag, nodes: 0, killers: [[None; 2]; 128], history: [[[0; 64]; 64]; 2], root_policy: None }
    }

    fn alpha_beta(&mut self, mut alpha: i32, mut beta: i32, depth: u8, ply: usize) -> (i32, Option<ChessMove>) {
        if self.stop_flag.load(Ordering::Relaxed) { return (0, None); }

        let hash = self.board.get_hash();
        let tt_entry = self.tt.get(hash);
        if let Some(ref entry) = tt_entry {
            if entry.depth >= depth {
                match entry.flag {
                    TTFlag::Exact => return (entry.score, entry.best_move),
                    TTFlag::LowerBound => {
                        if entry.score >= beta { return (entry.score, entry.best_move); }
                        alpha = alpha.max(entry.score);
                    }
                    TTFlag::UpperBound => {
                        if entry.score <= alpha { return (entry.score, entry.best_move); }
                        beta = beta.min(entry.score);
                    }
                }
            }
        }

        if depth == 0 { return (self.quiescence(alpha, beta), None); }
        self.nodes += 1;

        let moves = self.order_moves(MoveGen::new_legal(&self.board), if ply == 0 { self.root_policy.as_deref() } else { None }, ply, tt_entry.and_then(|e| e.best_move));
        if moves.is_empty() {
            return if self.board.checkers().popcnt() > 0 { (-MATE_SCORE + ply as i32, None) } else { (0, None) };
        }

        let mut best_move = None;
        let mut best_score = -INFINITY;
        let mut found_pv = false;
        let side = self.board.side_to_move();

        for (mv, _) in moves {
            let old_board = self.board;
            self.board = self.board.make_move_new(mv);
            self.move_history.push(MoveCodec::move_to_token(&mv) as usize);

            let score = if !found_pv {
                -self.alpha_beta(-beta, -alpha, depth - 1, ply + 1).0
            } else {
                let s = -self.alpha_beta(-alpha - 1, -alpha, depth - 1, ply + 1).0;
                if s > alpha && s < beta { -self.alpha_beta(-beta, -alpha, depth - 1, ply + 1).0 } else { s }
            };

            self.move_history.pop();
            self.board = old_board;

            if score > best_score {
                best_score = score;
                best_move = Some(mv);
            }

            if score > alpha {
                alpha = score;
                found_pv = true;
                if score >= beta {
                    if self.board.piece_on(mv.get_dest()).is_none() {
                        self.killers[ply][1] = self.killers[ply][0];
                        self.killers[ply][0] = Some(mv);
                        self.history[side as usize][mv.get_source().to_index()][mv.get_dest().to_index()] += (depth as i32) * (depth as i32);
                    }
                    self.tt.store(TTEntry { hash, depth, score: best_score, flag: TTFlag::LowerBound, best_move: Some(mv) });
                    return (best_score, Some(mv));
                }
            }
        }

        let flag = if best_score <= alpha { TTFlag::UpperBound } else { TTFlag::Exact };
        self.tt.store(TTEntry { hash, depth, score: best_score, flag, best_move });
        (best_score, best_move)
    }

    fn quiescence(&mut self, mut alpha: i32, beta: i32) -> i32 {
        self.nodes += 1;
        let stand_pat = eval::evaluate(&self.board);
        if stand_pat >= beta { return stand_pat; }
        if stand_pat > alpha { alpha = stand_pat; }

        let in_check = self.board.checkers().popcnt() > 0;
        let moves = MoveGen::new_legal(&self.board);
        
        let mut moves_vec: Vec<(ChessMove, i32)> = if in_check {
            moves.map(|m| (m, self.mvv_lva(m))).collect()
        } else {
            moves.filter(|m| self.board.piece_on(m.get_dest()).is_some()).map(|m| (m, self.mvv_lva(m))).collect()
        };
        
        moves_vec.sort_by(|a, b| b.1.cmp(&a.1));

        for (mv, _) in moves_vec {
            let old_board = self.board;
            self.board = self.board.make_move_new(mv);
            let score = -self.quiescence(-beta, -alpha);
            self.board = old_board;
            if score >= beta { return score; }
            if score > alpha { alpha = score; }
        }
        alpha
    }

    fn order_moves(&self, moves: MoveGen, policy: Option<&[f32]>, ply: usize, tt_move: Option<ChessMove>) -> Vec<(ChessMove, i32)> {
        let side = self.board.side_to_move() as usize;
        let mut ordered: Vec<_> = moves.map(|mv| {
            let mut score = 0;
            if Some(mv) == tt_move { score += 100000; }
            if let Some(p) = policy {
                let tok = MoveCodec::move_to_token(&mv) as usize;
                if tok < p.len() { score += (p[tok] * 10000.0) as i32; }
            }
            if self.board.piece_on(mv.get_dest()).is_some() { score += 20000 + self.mvv_lva(mv); }
            if Some(mv) == self.killers[ply][0] { score += 15000; } else if Some(mv) == self.killers[ply][1] { score += 10000; }
            score += self.history[side][mv.get_source().to_index()][mv.get_dest().to_index()] / 100;
            (mv, score)
        }).collect();
        ordered.sort_by(|a, b| b.1.cmp(&a.1));
        ordered
    }

    fn mvv_lva(&self, mv: ChessMove) -> i32 {
        let attacker = self.board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
        let victim = self.board.piece_on(mv.get_dest()).unwrap_or(Piece::Pawn);
        let v_val = match victim { Piece::Pawn => 10, Piece::Knight => 30, Piece::Bishop => 30, Piece::Rook => 50, Piece::Queen => 90, Piece::King => 100 };
        let a_val = match attacker { Piece::Pawn => 1, Piece::Knight => 3, Piece::Bishop => 3, Piece::Rook => 5, Piece::Queen => 9, Piece::King => 10 };
        v_val - a_val
    }
}
