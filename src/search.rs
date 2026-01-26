// src/search.rs

use std::time::Instant;

use crate::engine::{Engine, GameState};
use crate::tt::{TTEntry, TTFlag, TranspositionTable};
use chess::{Board, ChessMove, MoveGen, Piece, EMPTY};

// Constants for search evaluation in centipawns
const MATE_SCORE: i32 = 30_000;
const INFINITY: i32 = 32_000;

/// Maps a `ChessMove` to a token index for the model's embedding layer.
/// This is required to look up the move's probability in the policy head's output.
fn move_to_token(mv: &ChessMove, board: &Board) -> usize {
    // This function's implementation must be perfectly synchronized with the model's training.
    // A mismatch here would lead to the engine misinterpreting the policy output.
    let piece = board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn); // Should not happen in a legal position

    let piece_idx = mv.get_promotion().map_or_else(
        || piece.to_index(),
        |promo_piece| promo_piece.to_index(),
    );
    let to_square_idx = mv.get_dest().to_int() as usize;

    // The vocabulary is structured as [Piece; Square]
    piece_idx * 64 + to_square_idx
}

/// Manages the state and execution of the policy-guided Negamax search.
///
/// This struct holds references to the core engine components (model, TT) and tracks
/// search statistics.
pub struct SearchHandler<'a> {
    /// A reference to the neural network model used for evaluation (Policy + Value).
    engine: &'a Engine,
    /// A mutable reference to the transposition table to store and retrieve search results.
    tt: &'a mut TranspositionTable,
    /// The current state of the game being searched. This is updated as the search explores moves.
    pub game_state: GameState,
    /// A counter for the number of nodes (positions) evaluated during the search.
    pub nodes_searched: u64,
    /// A counter for the number of times a TT entry was used to prune the search.
    pub tt_hits: u64,
}

impl<'a> SearchHandler<'a> {
    /// Creates a new `SearchHandler` to begin a search.
    pub fn new(engine: &'a Engine, tt: &'a mut TranspositionTable, board: Board) -> Self {
        Self {
            engine,
            tt,
            game_state: GameState::new(board),
            nodes_searched: 0,
            tt_hits: 0,
        }
    }

    /// The main entry point for starting a search.
    ///
    /// This function implements iterative deepening. It calls the core `negamax` search
    /// for increasing depths and prints UCI-compliant information after each iteration.
    /// The best move is retrieved from the transposition table after the search.
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

            // After each depth, we probe the TT for the best move found so far.
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

    /// Converts the model's f32 value ([-1.0, 1.0]) to an i32 centipawn score.
    fn value_to_cp(value: f32) -> i32 {
        (value * 800.0) as i32
    }

    /// The core policy-guided Negamax search function.
    /// This function operates on `self.game_state` and uses a save/restore mechanism
    /// for the Mamba state to avoid expensive cloning.
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
        if depth == 0 {
            return self.quiescence_search(alpha, beta);
        }

        // --- Model Evaluation (Policy & Value) ---
        // To get the policy for the current node without permanently altering its Mamba state,
        // we back up the state, perform the forward pass, and then restore it. This avoids
        // a deep clone of the entire GameState.
        let mamba_backup = self.game_state.mamba_states.clone();
        let (policy_logits, _value) = self.engine.forward(0, &mut self.game_state);
        self.game_state.mamba_states = mamba_backup;

        let mut move_scores: Vec<(ChessMove, f32)> =
            MoveGen::new_legal(&self.game_state.board)
                .map(|mv| {
                    let token = move_to_token(&mv, &self.game_state.board);
                    let score = *policy_logits.get(token).unwrap_or(&-100.0);
                    (mv, score)
                })
                .collect();
        move_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut best_score = -INFINITY;
        let mut best_move = None;
        let mut tt_flag = TTFlag::UPPERBOUND;

        for (i, (mv, _)) in move_scores.iter().enumerate() {
            let board_backup = self.game_state.board;
            let mamba_backup = self.game_state.mamba_states.clone();

            // To apply LMR correctly, we must not reduce tactical moves.
            let is_capture = board_backup.piece_on(mv.get_dest()).is_some();
            let is_promotion = mv.get_promotion().is_some();

            self.game_state.board = self.game_state.board.make_move_new(*mv);
            let token = move_to_token(mv, &board_backup);
            self.engine.forward(token, &mut self.game_state);

            let is_check = *self.game_state.board.checkers() != EMPTY;
            let is_tactical = is_capture || is_promotion || is_check;

            let score = if i == 0 {
                -self.negamax(depth - 1, -beta, -alpha)
            } else {
                // Apply LMR to late, non-tactical moves.
                let reduction = if depth > 2 && i > 3 && !is_tactical { 2 } else { 1 };
                let mut search_score = -self.negamax(depth - reduction, -alpha - 1, -alpha);

                // Re-search if the null-window search beats alpha.
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

    /// A specialized search function to handle tactical "explosions" (captures and promotions)
    /// at the leaves of the main search tree. This prevents the horizon effect.
    fn quiescence_search(&mut self, mut alpha: i32, beta: i32) -> i32 {
        self.nodes_searched += 1;

        // --- Stand-Pat Evaluation ---
        // To get the stand-pat score, we must evaluate the position without permanently
        // altering the Mamba state. We use the same backup/restore pattern as in negamax.
        let mamba_backup = self.game_state.mamba_states.clone();
        let (_, value) = self.engine.forward(0, &mut self.game_state);
        self.game_state.mamba_states = mamba_backup; // Restore the state
        let stand_pat = Self::value_to_cp(value);

        if stand_pat >= beta {
            return beta;
        }
        if alpha < stand_pat {
            alpha = stand_pat;
        }

        // --- Generate and Score Tactical Moves ---
        let move_scores: Vec<(ChessMove, f32)> =
            MoveGen::new_legal(&self.game_state.board)
                .filter(|m| {
                    // Only consider captures and promotions in q-search
                    self.game_state.board.piece_on(m.get_dest()).is_some() || m.get_promotion().is_some()
                })
                .map(|mv| (mv, 0.0)) // We don't need policy for q-search move ordering
                .collect();

        // A simple MVV-LVA (Most Valuable Victim - Least Valuable Attacker) heuristic could be added here
        // for better move ordering, but for now, we search them as they come.

        for (mv, _) in move_scores.iter() {
            // --- PUSH/POP State Management ---
            let board_backup = self.game_state.board;
            let mamba_backup = self.game_state.mamba_states.clone();

            self.game_state.board = self.game_state.board.make_move_new(*mv);
            let token = move_to_token(mv, &board_backup);
            self.engine.forward(token, &mut self.game_state);

            let score = -self.quiescence_search(-beta, -alpha);

            self.game_state.board = board_backup;
            self.game_state.mamba_states = mamba_backup;

            if score >= beta {
                return beta; // Beta-cutoff
            }
            if score > alpha {
                alpha = score;
            }
        }

        alpha
    }
}
