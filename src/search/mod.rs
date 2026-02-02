// src/search/mod.rs

pub mod eval;

use crate::engine::Engine;
use crate::uci::MoveCodec;
use chess::{Board, ChessMove, MoveGen, Piece};
use dashmap::DashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

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

use parking_lot::RwLock;

pub struct TranspositionTable {
    table: RwLock<DashMap<u64, TTEntry>>,
}

impl TranspositionTable {
    pub fn new() -> Self {
        // Default 512MB: ~16M entries * ~32 bytes
        Self {
            table: RwLock::new(DashMap::with_capacity(16_000_000)),
        }
    }

    pub fn resize(&self, mb: usize) {
        // Each entry is roughly 32 bytes.
        let entries = (mb * 1024 * 1024) / 32;
        let mut table = self.table.write();
        *table = DashMap::with_capacity(entries);
    }

    pub fn get(&self, hash: u64) -> Option<TTEntry> {
        self.table.read().get(&hash).map(|r| r.clone())
    }
    pub fn store(&self, entry: TTEntry) {
        self.table.read().insert(entry.hash, entry);
    }
    pub fn clear(&self) {
        self.table.read().clear();
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
        _move_history: Vec<usize>,
        max_depth: u8,
        stop_flag: Arc<AtomicBool>,
        tx: mpsc::Sender<String>,
    ) {
        let engine = Arc::clone(&self.engine);
        let tt = Arc::clone(&self.tt);

        std::thread::spawn(move || {
            let mut worker = SearchWorker::new(engine, tt, board, _move_history, stop_flag);
            let mut best_move = None;
            let mut last_score = 0;
            let start_time = std::time::Instant::now();

            // Iterative Deepening
            for depth in 1..=max_depth {
                if worker.stop_flag.load(Ordering::Relaxed) {
                    break;
                }

                // Aspiration Windows
                let mut alpha = -INFINITY;
                let mut beta = INFINITY;
                if depth > 3 {
                    alpha = last_score - 30;
                    beta = last_score + 30;
                }

                let mut score_res;
                let mut mv_res;

                loop {
                    let (s, m) = worker.alpha_beta(alpha, beta, depth, 0);
                    score_res = s;
                    mv_res = m;

                    if worker.stop_flag.load(Ordering::Relaxed) {
                        break;
                    }

                    if score_res <= alpha {
                        alpha = alpha.saturating_sub(100);
                        if alpha < -INFINITY { alpha = -INFINITY; }
                    } else if score_res >= beta {
                        beta = beta.saturating_add(100);
                        if beta > INFINITY { beta = INFINITY; }
                    } else {
                        break;
                    }
                    
                    // Safety break if window explodes
                    if alpha == -INFINITY && beta == INFINITY { break; }
                }

                if !worker.stop_flag.load(Ordering::Relaxed) {
                    if let Some(m) = mv_res {
                        best_move = Some(m);
                        last_score = score_res;
                        let elapsed = start_time.elapsed().as_secs_f64().max(0.001);
                        let nps = (worker.nodes as f64 / elapsed) as u64;
                        println!(
                            "info depth {} score cp {} nodes {} nps {} pv {}",
                            depth, score_res, worker.nodes, nps, m
                        );
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

        let in_check = self.board.checkers().popcnt() > 0;
        if in_check {
            depth += 1; // Check Extension
        }

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

        if depth == 0 {
            return (self.quiescence(alpha, beta), None);
        }
        self.nodes += 1;

        // Neural Inference at Pv nodes or high depth
        let is_pv = beta - alpha > 1;
        let mut policy = None;
        if ply <= 6 && (is_pv || ply == 0) {
            if let Ok(res) = self.engine.forward(&self.move_history) {
                policy = res.0.to_vec1::<f32>().ok();
            }
        }

        let tt_move = tt_entry.as_ref().and_then(|e| e.best_move);
        let moves = self.order_moves(MoveGen::new_legal(&self.board), policy.as_deref(), ply, tt_move);

        if moves.is_empty() {
            return if in_check {
                (-MATE_SCORE + ply as i32, None)
            } else {
                (0, None)
            };
        }

        let mut best_move = None;
        let mut best_score = -INFINITY;
        let mut moves_searched = 0;

        for (mv, policy_prob) in moves {
            let old_board = self.board;
            self.board = self.board.make_move_new(mv);
            self.move_history.push(MoveCodec::move_to_token(&mv) as usize);

            // Policy-Guided Extensions
            let mut extension = 0;
            if policy_prob > 5000 {
                extension = 1;
            }

            let mut score;
            if moves_searched == 0 {
                // Principal Variation Search
                score = -self.alpha_beta(-beta, -alpha, depth - 1 + extension, ply + 1).0;
            } else {
                // LMR
                let mut reduction = if depth >= 3
                    && moves_searched > 4
                    && self.board.piece_on(mv.get_dest()).is_none()
                    && !in_check
                {
                    1
                } else {
                    0
                };

                if policy_prob < 100 && reduction > 0 {
                    reduction += 1;
                }

                score = -self
                    .alpha_beta(-alpha - 1, -alpha, depth.saturating_sub(1 + reduction), ply + 1)
                    .0;
                
                if score > alpha && reduction > 0 {
                    score = -self.alpha_beta(-alpha - 1, -alpha, depth - 1, ply + 1).0;
                }
                
                if score > alpha && score < beta {
                    score = -self.alpha_beta(-beta, -alpha, depth - 1 + extension, ply + 1).0;
                }
            }

            self.move_history.pop();
            self.board = old_board;
            moves_searched += 1;

            if score > best_score {
                best_score = score;
                best_move = Some(mv);
            }

            if score > alpha {
                alpha = score;
                if score >= beta {
                    if self.board.piece_on(mv.get_dest()).is_none() {
                        self.killers[ply][1] = self.killers[ply][0];
                        self.killers[ply][0] = Some(mv);
                        let side = self.board.side_to_move() as usize;
                        self.history[side][mv.get_source().to_index()][mv.get_dest().to_index()] +=
                            (depth as i32) * (depth as i32);
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
        let stand_pat = eval::evaluate(&self.board);
        if stand_pat >= beta {
            return stand_pat;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let in_check = self.board.checkers().popcnt() > 0;
        let moves = MoveGen::new_legal(&self.board);

        let mut moves_vec: Vec<(ChessMove, i32)> = if in_check {
            // Full search in check for stability
            moves.map(|m| (m, self.mvv_lva(m))).collect()
        } else {
            // Only captures
            moves
                .filter(|m| self.board.piece_on(m.get_dest()).is_some())
                .map(|m| (m, self.mvv_lva(m)))
                .collect()
        };

        moves_vec.sort_by(|a, b| b.1.cmp(&a.1));

        for (mv, _) in moves_vec {
            let old_board = self.board;
            self.board = self.board.make_move_new(mv);
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

    fn order_moves(
        &self,
        moves: MoveGen,
        policy: Option<&[f32]>,
        ply: usize,
        tt_move: Option<ChessMove>,
    ) -> Vec<(ChessMove, i32)> {
        let side = self.board.side_to_move() as usize;
        let mut full_ordered: Vec<(ChessMove, i32, i32)> = moves
            .map(|mv| {
                let mut total_score = 0;
                let mut p_score = 0;
                
                // 1. TT Move Priority
                if Some(mv) == tt_move {
                    total_score += 10_000_000;
                }
                
                // 2. Neural Policy (Top 3 Priority)
                if let Some(p) = policy {
                    let tok = MoveCodec::move_to_token(&mv) as usize;
                    if tok < p.len() {
                        p_score = (p[tok] * 100_000.0) as i32;
                        total_score += p_score;
                    }
                }
                
                // 3. Captures (MVV-LVA)
                if self.board.piece_on(mv.get_dest()).is_some() {
                    total_score += 200_000 + self.mvv_lva(mv);
                }
                
                // 4. Killers
                if Some(mv) == self.killers[ply][0] {
                    total_score += 100_000;
                } else if Some(mv) == self.killers[ply][1] {
                    total_score += 90_000;
                }
                
                // 5. History
                total_score += self.history[side][mv.get_source().to_index()][mv.get_dest().to_index()];
                
                (mv, p_score, total_score)
            })
            .collect();

        full_ordered.sort_by(|a, b| b.2.cmp(&a.2));
        full_ordered.into_iter().map(|(mv, p, _)| (mv, p)).collect()
    }

    fn mvv_lva(&self, mv: ChessMove) -> i32 {
        let attacker = self.board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
        let victim = self.board.piece_on(mv.get_dest()).unwrap_or(Piece::Pawn);
        let v_val = match victim {
            Piece::Pawn => 10,
            Piece::Knight => 30,
            Piece::Bishop => 30,
            Piece::Rook => 50,
            Piece::Queen => 90,
            Piece::King => 100,
        };
        let a_val = match attacker {
            Piece::Pawn => 1,
            Piece::Knight => 3,
            Piece::Bishop => 3,
            Piece::Rook => 5,
            Piece::Queen => 9,
            Piece::King => 10,
        };
        v_val - a_val
    }
}
