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
        }
    }

    pub fn search(&mut self, max_depth: u8) -> (Option<ChessMove>, i32) {
        let start_time = Instant::now();
        self.nodes_searched = 0;
        self.tt_hits = 0;

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
        (value * 800.0) as i32
    }

    fn negamax(&mut self, depth: u8, mut alpha: i32, mut beta: i32) -> i32 {
        self.nodes_searched += 1;

        let hash = self.game_state.board.get_hash();
        if let Some(entry) = self.tt.probe(hash) {
            if entry.depth >= depth {
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
            return -MATE_SCORE;
        }
        if self.game_state.board.status() == chess::BoardStatus::Stalemate {
            return 0;
        }

        // --- Static Evaluation (Gatekeeper) ---
        let static_score = self.tutor.evaluate(&self.game_state.board);

        if depth == 0 {
            return self.quiescence_search(alpha, beta);
        }

        // --- Null Move Pruning ---
        if depth >= 3 && !(*self.game_state.board.checkers() != EMPTY) {
            // If static eval is high enough, try null move
            if static_score >= beta {
                let board_backup = self.game_state.board;
                if let Some(null_board) = board_backup.null_move() {
                    let mamba_backup = self.game_state.mamba_states.clone();
                    self.game_state.board = null_board;
                    // We don't update Mamba state for null move to keep it simple and fast
                    let score = -self.negamax(depth - 3, -beta, -beta + 1);
                    self.game_state.board = board_backup;
                    self.game_state.mamba_states = mamba_backup;

                    if score >= beta {
                        return beta;
                    }
                }
            }
        }

        // --- Futility Pruning ---
        // Handled in the move loop.

        // --- Model Evaluation (Policy) ---
        let mamba_backup = self.game_state.mamba_states.clone();
        let (policy_logits, _value) = self.engine.forward(0, &mut self.game_state).expect("Forward pass failed");
        self.game_state.mamba_states = mamba_backup;

        let logits_vec = policy_logits.to_vec1::<f32>().expect("Failed to get logits");

        let mut move_scores: Vec<(ChessMove, f32)> =
            MoveGen::new_legal(&self.game_state.board)
                .map(|mv| {
                    let token = move_to_token(&mv, &self.game_state.board);
                    let score = *logits_vec.get(token).unwrap_or(&-100.0);
                    (mv, score)
                })
                .collect();
        move_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut best_score = -INFINITY;
        let mut best_move = None;
        let mut tt_flag = TTFlag::UPPERBOUND;

        for (i, (mv, _)) in move_scores.iter().enumerate() {
            let board_backup = self.game_state.board;

            let is_capture = board_backup.piece_on(mv.get_dest()).is_some();
            let is_promotion = mv.get_promotion().is_some();

            // Futility Pruning check
            if depth == 1 && i > 0 && !is_capture && !is_promotion && !(*board_backup.checkers() != EMPTY) {
                if static_score + 150 < alpha {
                    continue;
                }
            }

            let mamba_backup = self.game_state.mamba_states.clone();

            self.game_state.board = self.game_state.board.make_move_new(*mv);
            let token = move_to_token(mv, &board_backup);
            self.engine.forward(token, &mut self.game_state).expect("Forward pass failed");

            let is_check = *self.game_state.board.checkers() != EMPTY;
            let is_tactical = is_capture || is_promotion || is_check;

            let score = if i == 0 {
                -self.negamax(depth - 1, -beta, -alpha)
            } else {
                let reduction = if depth > 2 && i > 3 && !is_tactical { 2 } else { 1 };
                let mut search_score = -self.negamax(depth - reduction, -alpha - 1, -alpha);

                if search_score > alpha && search_score < beta {
                    search_score = -self.negamax(depth - 1, -beta, -alpha);
                }
                search_score
            };

            self.game_state.board = board_backup;
            self.game_state.mamba_states = mamba_backup;

            if score > best_score {
                best_score = score;
                best_move = Some(*mv);
            }
            if best_score > alpha {
                alpha = best_score;
                tt_flag = TTFlag::EXACT;
            }

            if alpha >= beta {
                self.tt.store(TTEntry { key: hash, best_move, score: best_score, depth, flag: TTFlag::LOWERBOUND });
                return best_score;
            }
        }

        self.tt.store(TTEntry { key: hash, best_move, score: best_score, depth, flag: tt_flag });
        best_score
    }

    fn quiescence_search(&mut self, mut alpha: i32, beta: i32) -> i32 {
        self.nodes_searched += 1;

        // --- Combined Evaluation (Leaf) ---
        let mamba_backup = self.game_state.mamba_states.clone();
        let (_, ai_value) = self.engine.forward(0, &mut self.game_state).expect("Forward pass failed");
        self.game_state.mamba_states = mamba_backup;

        let ai_score = Self::value_to_cp(ai_value);
        let static_score = self.tutor.evaluate(&self.game_state.board);

        // Linear combination: 50% AI, 50% Static
        let stand_pat = (ai_score + static_score) / 2;

        if stand_pat >= beta {
            return beta;
        }
        if alpha < stand_pat {
            alpha = stand_pat;
        }

        let moves: Vec<ChessMove> = MoveGen::new_legal(&self.game_state.board)
            .filter(|m| self.game_state.board.piece_on(m.get_dest()).is_some() || m.get_promotion().is_some())
            .collect();

        for mv in moves {
            let board_backup = self.game_state.board;
            let mamba_backup = self.game_state.mamba_states.clone();

            self.game_state.board = self.game_state.board.make_move_new(mv);
            let token = move_to_token(&mv, &board_backup);
            self.engine.forward(token, &mut self.game_state).expect("Forward pass failed");

            let score = -self.quiescence_search(-beta, -alpha);

            self.game_state.board = board_backup;
            self.game_state.mamba_states = mamba_backup;

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
