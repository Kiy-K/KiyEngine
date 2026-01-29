// src/search.rs

use std::time::Instant;

use crate::engine::{Engine, GameState};
use crate::tutor::Tutor;
use crate::tt::{TTEntry, TTFlag, TranspositionTable};
use chess::{Board, ChessMove, MoveGen, Piece, EMPTY};

// Constants for search evaluation in centipawns
const MATE_SCORE: i32 = 30_000;
const INFINITY: i32 = 32_000;

/// Maps a `ChessMove` to a token index for the model's embedding layer.
fn move_to_token(mv: &ChessMove, board: &Board) -> usize {
    let piece = board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
    let piece_idx = mv.get_promotion().map_or_else(
        || piece.to_index(),
        |promo_piece| promo_piece.to_index(),
    );
    let to_square_idx = mv.get_dest().to_int() as usize;
    piece_idx * 64 + to_square_idx
}

pub struct SearchHandler<'a> {
    engine: &'a Engine,
    tutor: &'a Tutor,
    tt: &'a mut TranspositionTable,
    pub game_state: GameState,
    pub nodes_searched: u64,
    pub tt_hits: u64,
    killers: [[Option<ChessMove>; 2]; 64],
    ply: usize,
}

impl<'a> SearchHandler<'a> {
    pub fn new(engine: &'a Engine, tutor: &'a Tutor, tt: &'a mut TranspositionTable, board: Board) -> Self {
        let game_state = GameState::new(board, &engine.device).expect("Failed to initialize GameState");
        Self {
            engine,
            tutor,
            tt,
            game_state,
            nodes_searched: 0,
            tt_hits: 0,
            killers: [[None; 2]; 64],
            ply: 0,
        }
    }

    pub fn search(&mut self, max_depth: u8) -> (Option<ChessMove>, i32) {
        let start_time = Instant::now();
        self.nodes_searched = 0;
        self.tt_hits = 0;
        self.killers = [[None; 2]; 64];
        self.ply = 0;

        let mut best_score = 0;

        for depth in 1..=max_depth {
            let score = self.negamax(depth, -INFINITY, INFINITY);

            let elapsed = start_time.elapsed();
            let nps = (self.nodes_searched * 1000) / (elapsed.as_millis() as u64 + 1);
            best_score = score;

            if let Some(entry) = self.tt.probe(self.game_state.board.get_hash()) {
                if let Some(mv) = entry.best_move {
                    println!(
                        "info depth {} score cp {} nodes {} nps {} pv {}",
                        depth, score, self.nodes_searched, nps, mv,
                    );
                }
            }
        }

        let best_move = self
            .tt
            .probe(self.game_state.board.get_hash())
            .and_then(|e| e.best_move);
        (best_move, best_score)
    }

    fn value_to_cp(value: f32) -> i32 {
        (value * 600.0) as i32
    }

    fn score_move(&self, mv: &ChessMove, tt_move: Option<ChessMove>) -> i32 {
        if Some(*mv) == tt_move {
            return 1_000_000;
        }

        let board = &self.game_state.board;
        if let Some(victim) = board.piece_on(mv.get_dest()) {
            let attacker = board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
            // MVV-LVA using Tutor's MG_VALUE
            return 100_000 + crate::tutor::MG_VALUE[victim.to_index()] * 10 - crate::tutor::MG_VALUE[attacker.to_index()];
        }

        if self.ply < 64 {
            if self.killers[self.ply][0] == Some(*mv) {
                return 90_000;
            }
            if self.killers[self.ply][1] == Some(*mv) {
                return 80_000;
            }
        }

        0
    }

    fn negamax(&mut self, depth: u8, mut alpha: i32, mut beta: i32) -> i32 {
        self.nodes_searched += 1;
        let is_root = self.ply == 0;
        let hash = self.game_state.board.get_hash();

        // Step 0: TT Probe
        let mut tt_move = None;
        if let Some(entry) = self.tt.probe(hash) {
            tt_move = entry.best_move;
            if entry.depth >= depth && !is_root {
                self.tt_hits += 1;
                match entry.flag {
                    TTFlag::EXACT => return entry.score,
                    TTFlag::LOWERBOUND => alpha = alpha.max(entry.score),
                    TTFlag::UPPERBOUND => beta = beta.min(entry.score),
                }
                if alpha >= beta {
                    return entry.score;
                }
            }
        }

        if self.game_state.board.status() == chess::BoardStatus::Checkmate {
            return -MATE_SCORE + self.ply as i32;
        }
        if self.game_state.board.status() == chess::BoardStatus::Stalemate {
            return 0;
        }

        if depth == 0 {
            return self.quiescence_search(alpha, beta);
        }

        // Step 1: Static Eval
        let static_score = self.tutor.evaluate(&self.game_state.board);

        // Step 2: Pruning
        let in_check = *self.game_state.board.checkers() != EMPTY;

        // Futility Pruning
        if depth >= 1 && !in_check && !is_root {
            let margin = 150 + 50 * (depth as i32);
            if static_score + margin < alpha {
                return static_score;
            }
        }

        // Null Move Pruning
        if depth >= 3 && !in_check && !is_root && static_score >= beta {
            if let Some(null_board) = self.game_state.board.null_move() {
                let board_backup = self.game_state.board;
                let mamba_backup = self.game_state.mamba_states.clone();

                self.game_state.board = null_board;
                self.ply += 1;
                // Shallow search with reduced depth
                let score = -self.negamax(depth - 3, -beta, -beta + 1);
                self.ply -= 1;

                self.game_state.board = board_backup;
                self.game_state.mamba_states = mamba_backup;

                if score >= beta {
                    return beta;
                }
            }
        }

        // Step 3 & 4: Neural Eval & Hybridization
        if depth >= 2 {
            // Neural Eval only if depth >= 2 and NOT in Q-search (already handled by depth > 0)
            let mamba_backup = self.game_state.mamba_states.clone();
            let (_logits, ai_value) = self.engine.forward(0, &mut self.game_state).expect("Forward pass failed");
            self.game_state.mamba_states = mamba_backup;

            let ai_score = Self::value_to_cp(ai_value);
            // Combined Arms Hybrid Formula: 0.7 AI + 0.3 Tutor
            let hybrid_score = (0.7 * ai_score as f32 + 0.3 * static_score as f32) as i32;

            // Prune if hybrid score fails high
            if hybrid_score >= beta && !is_root {
                return beta;
            }
        }

        // Move Loop
        let moves = MoveGen::new_legal(&self.game_state.board);
        let mut scored_moves: Vec<(ChessMove, i32)> = moves.map(|mv| {
            let score = self.score_move(&mv, tt_move);
            (mv, score)
        }).collect();

        // Sort by priority: TT Move > MVV-LVA > Killers > Others
        scored_moves.sort_by(|a, b| b.1.cmp(&a.1));

        let mut best_score = -INFINITY;
        let mut best_move = None;
        let mut tt_flag = TTFlag::UPPERBOUND;

        for (i, (mv, _)) in scored_moves.into_iter().enumerate() {
            let board_backup = self.game_state.board;
            let mamba_backup = self.game_state.mamba_states.clone();

            self.game_state.board = self.game_state.board.make_move_new(mv);
            let token = move_to_token(&mv, &board_backup);
            // Update Mamba state for recursion
            self.engine.forward(token, &mut self.game_state).expect("Forward pass failed");

            self.ply += 1;
            let score = if i == 0 {
                -self.negamax(depth - 1, -beta, -alpha)
            } else {
                // Simple Late Move Reduction
                let is_tactical = board_backup.piece_on(mv.get_dest()).is_some() || mv.get_promotion().is_some() || (*self.game_state.board.checkers() != EMPTY);
                let reduction = if depth >= 3 && i >= 4 && !is_tactical { 2 } else { 1 };
                let mut s = -self.negamax(depth - reduction, -alpha - 1, -alpha);
                if s > alpha && (reduction > 1 || s < beta) {
                    s = -self.negamax(depth - 1, -beta, -alpha);
                }
                s
            };
            self.ply -= 1;

            self.game_state.board = board_backup;
            self.game_state.mamba_states = mamba_backup;

            if score > best_score {
                best_score = score;
                best_move = Some(mv);
            }
            if best_score > alpha {
                alpha = best_score;
                tt_flag = TTFlag::EXACT;
            }

            if alpha >= beta {
                // Update Killers for quiet moves
                if board_backup.piece_on(mv.get_dest()).is_none() && self.ply < 64 {
                    if self.killers[self.ply][0] != Some(mv) {
                        self.killers[self.ply][1] = self.killers[self.ply][0];
                        self.killers[self.ply][0] = Some(mv);
                    }
                }
                self.tt.store(TTEntry { key: hash, best_move: Some(mv), score: best_score, depth, flag: TTFlag::LOWERBOUND });
                return best_score;
            }
        }

        self.tt.store(TTEntry { key: hash, best_move, score: best_score, depth, flag: tt_flag });
        best_score
    }

    fn quiescence_search(&mut self, mut alpha: i32, beta: i32) -> i32 {
        self.nodes_searched += 1;

        // Q-search is 100% Tutor-powered as requested
        let static_score = self.tutor.evaluate(&self.game_state.board);

        if static_score >= beta {
            return beta;
        }
        if alpha < static_score {
            alpha = static_score;
        }

        // Only search captures and promotions
        let moves = MoveGen::new_legal(&self.game_state.board)
            .filter(|m| self.game_state.board.piece_on(m.get_dest()).is_some() || m.get_promotion().is_some());

        let mut scored_moves: Vec<(ChessMove, i32)> = moves.map(|mv| {
            let victim = self.game_state.board.piece_on(mv.get_dest());
            let score = if let Some(v) = victim {
                let attacker = self.game_state.board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
                crate::tutor::MG_VALUE[v.to_index()] * 10 - crate::tutor::MG_VALUE[attacker.to_index()]
            } else {
                0 // promotion only
            };
            (mv, score)
        }).collect();
        scored_moves.sort_by(|a, b| b.1.cmp(&a.1));

        for (mv, _) in scored_moves {
            let board_backup = self.game_state.board;
            self.game_state.board = self.game_state.board.make_move_new(mv);

            let score = -self.quiescence_search(-beta, -alpha);

            self.game_state.board = board_backup;

            if score >= beta {
                return beta;
            }
            if score > alpha {
                alpha = score;
            }
        }

        alpha
    }
}
