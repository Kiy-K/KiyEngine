// src/search/mod.rs

use crate::engine::Engine;
use crate::uci::MoveCodec;
use chess::{Board, ChessMove, MoveGen, Piece, Color};
use dashmap::DashMap;
use std::sync::{Arc, mpsc};
use std::sync::atomic::{AtomicBool, Ordering};

const MATE_SCORE: i32 = 30000;
const INFINITY: i32 = 32000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TTFlag {
    Exact,
    LowerBound,
    UpperBound,
}

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
        Self {
            table: DashMap::with_capacity(100_000),
        }
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
            
            for depth in 1..=max_depth {
                if worker.stop_flag.load(Ordering::Relaxed) {
                    break;
                }
                
                let (score, mv) = worker.alpha_beta(-INFINITY, INFINITY, depth, 0);
                
                if !worker.stop_flag.load(Ordering::Relaxed) {
                    if let Some(m) = mv {
                        best_move = Some(m);
                        let elapsed = start_time.elapsed().as_secs_f64().max(0.001);
                        let nps = (worker.nodes as f64 / elapsed) as u64;
                        println!("info depth {} score cp {} nodes {} nps {}", depth, score, worker.nodes, nps);
                    }
                }
            }
            
            if let Some(mv) = best_move {
                let _ = tx.send(format!("bestmove {}", mv));
            } else {
                let mut moves = MoveGen::new_legal(&board);
                if let Some(mv) = moves.next() {
                    let _ = tx.send(format!("bestmove {}", mv));
                }
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
        }
    }

    fn alpha_beta(&mut self, mut alpha: i32, mut beta: i32, depth: u8, ply: usize) -> (i32, Option<ChessMove>) {
        if self.stop_flag.load(Ordering::Relaxed) {
            return (0, None);
        }

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

        if depth == 0 {
            return (self.quiescence(alpha, beta), None);
        }

        self.nodes += 1;

        // Neural ordering & evaluation
        let (policy_scores, score_val) = if depth >= 2 || ply <= 1 {
            match self.engine.forward(&self.move_history) {
                Ok(res) => {
                    let logits = res.0.to_vec1::<f32>().ok();
                    let val = (res.1 * 1000.0) as i32;
                    let val = if self.board.side_to_move() == Color::White { val } else { -val };
                    (logits, Some(val))
                },
                Err(_) => (None, None),
            }
        } else {
            (None, None)
        };

        let moves = self.order_moves(MoveGen::new_legal(&self.board), policy_scores.as_deref(), ply);
        
        if moves.is_empty() {
            if self.board.checkers().popcnt() > 0 {
                return (-MATE_SCORE + ply as i32, None);
            } else {
                return (0, None);
            }
        }

        let mut best_move = None;
        let mut best_score = -INFINITY;
        let mut found_pv = false;

        for (mv, _) in moves {
            let old_board = self.board;
            self.board = self.board.make_move_new(mv);
            let token = MoveCodec::move_to_token(&mv);
            self.move_history.push(token as usize);

            let score = if !found_pv {
                -self.alpha_beta(-beta, -alpha, depth - 1, ply + 1).0
            } else {
                let s = -self.alpha_beta(-alpha - 1, -alpha, depth - 1, ply + 1).0;
                if s > alpha && s < beta {
                    -self.alpha_beta(-beta, -alpha, depth - 1, ply + 1).0
                } else {
                    s
                }
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
                    }
                    
                    self.tt.store(TTEntry {
                        hash,
                        depth,
                        score: best_score,
                        flag: TTFlag::LowerBound,
                        best_move: Some(mv),
                    });
                    return (best_score, Some(mv));
                }
            }
        }

        let flag = if best_score <= alpha {
            TTFlag::UpperBound
        } else {
            TTFlag::Exact
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

    fn quiescence(&mut self, mut alpha: i32, beta: i32) -> i32 {
        self.nodes += 1;
        let stand_pat = self.evaluate();
        if stand_pat >= beta { return stand_pat; }
        if stand_pat > alpha { alpha = stand_pat; }

        let moves = MoveGen::new_legal(&self.board);
        let mut captures: Vec<(ChessMove, i32)> = moves
            .filter(|m| self.board.piece_on(m.get_dest()).is_some())
            .map(|m| (m, self.mvv_lva(m)))
            .collect();
        
        captures.sort_by(|a, b| b.1.cmp(&a.1));

        for (mv, _) in captures {
            let old_board = self.board;
            self.board = self.board.make_move_new(mv);
            let score = -self.quiescence(-beta, -alpha);
            self.board = old_board;

            if score >= beta { return score; }
            if score > alpha { alpha = score; }
        }

        alpha
    }

    fn evaluate(&self) -> i32 {
        // Use Brain for static evaluation
        match self.engine.forward(&self.move_history) {
            Ok((_, val)) => {
                let score = (val * 1000.0) as i32;
                if self.board.side_to_move() == Color::White {
                    score
                } else {
                    -score
                }
            },
            Err(_) => 0,
        }
    }

    fn order_moves(&self, moves: MoveGen, policy_scores: Option<&[f32]>, ply: usize) -> Vec<(ChessMove, i32)> {
        let mut ordered_moves: Vec<(ChessMove, i32)> = moves.map(|mv| {
            let mut score = 0;

            if let Some(policy) = policy_scores {
                let token = MoveCodec::move_to_token(&mv);
                if (token as usize) < policy.len() {
                    score += (policy[token as usize] * 1000.0) as i32;
                }
            }

            if self.board.piece_on(mv.get_dest()).is_some() {
                score += 2000 + self.mvv_lva(mv);
            }

            if Some(mv) == self.killers[ply][0] {
                score += 1500;
            } else if Some(mv) == self.killers[ply][1] {
                score += 1000;
            }

            (mv, score)
        }).collect();

        ordered_moves.sort_by(|a, b| b.1.cmp(&a.1));
        ordered_moves
    }

    fn mvv_lva(&self, mv: ChessMove) -> i32 {
        let attacker = self.board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
        let victim = self.board.piece_on(mv.get_dest()).unwrap_or(Piece::Pawn);
        
        let victim_val = match victim {
            Piece::Pawn => 10, Piece::Knight => 30, Piece::Bishop => 30,
            Piece::Rook => 50, Piece::Queen => 90, Piece::King => 100,
        };
        
        let attacker_val = match attacker {
            Piece::Pawn => 1, Piece::Knight => 3, Piece::Bishop => 3,
            Piece::Rook => 5, Piece::Queen => 9, Piece::King => 10,
        };

        victim_val - attacker_val
    }
}
