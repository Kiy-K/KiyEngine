// src/search.rs

use crate::engine::{Engine, GameState};
use chess::{Board, ChessMove, MoveGen, Piece, EMPTY};

/// Manages the state and execution of the alpha-beta search algorithm.
///
/// This struct holds a reference to the `Engine` (for evaluation), the current `GameState`,
/// and statistics about the search, such as the number of nodes visited.
pub struct SearchHandler<'a> {
    /// A reference to the neural network model used for evaluation.
    engine: &'a Engine,
    /// The current state of the game being searched. This is updated as the search explores moves.
    pub game_state: GameState,
    /// A counter for the number of nodes (positions) evaluated during the search.
    pub nodes_searched: u64,
}

/// Maps a `ChessMove` to a token index for the model's embedding layer.
/// The vocabulary is structured as [Piece; Square], so we can calculate the index.
fn move_to_token(mv: &ChessMove, board: &Board) -> usize {
    let piece = board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn); // Default to pawn if source is empty (should not happen)

    let piece_idx = mv.get_promotion().map_or_else(
        || piece.to_index(),
        |promo_piece| promo_piece.to_index(),
    );
    let to_square_idx = mv.get_dest().to_int() as usize;

    // A simple mapping, this might need to be more sophisticated
    // depending on the final training setup.
    piece_idx * 64 + to_square_idx
}

impl<'a> SearchHandler<'a> {
    pub fn new(engine: &'a Engine, board: Board) -> Self {
        Self {
            engine,
            game_state: GameState::new(board),
            nodes_searched: 0,
        }
    }

    /// The main entry point for starting a search.
    /// Implements iterative deepening.
    pub fn search(&mut self, depth: u8) -> (Option<ChessMove>, f32) {
        let mut best_move = None;
        let mut best_eval = -f32::INFINITY;

        for i in 1..=depth {
            let (mv, eval) = self.alphabeta(-f32::INFINITY, f32::INFINITY, i);
            if let Some(m) = mv {
                best_move = Some(m);
                best_eval = eval;
                println!("info depth {} score cp {} pv {}", i, (eval * 100.0) as i32, m);
            }
        }

        (best_move, best_eval)
    }

    /// The core alpha-beta search function with state management.
    fn alphabeta(&mut self, mut alpha: f32, beta: f32, depth: u8) -> (Option<ChessMove>, f32) {
        self.nodes_searched += 1;

        let move_gen = MoveGen::new_legal(&self.game_state.board);
        let moves_empty = move_gen.len() == 0;

        // Handle terminal nodes (checkmate/stalemate)
        if moves_empty {
            if self.game_state.board.checkers() == &EMPTY {
                return (None, 0.0); // Stalemate
            } else {
                return (None, -f32::INFINITY); // Checkmate
            }
        }

        let mut best_move = None;
        let mut best_score = -f32::INFINITY;

        // --- Policy Head Move Ordering ---
        let (policy_logits, _) = self.engine.forward(0, &mut self.game_state.clone()); // Use a dummy token to get policy
        let mut move_scores: Vec<(ChessMove, f32)> = move_gen.map(|mv| {
            let token = move_to_token(&mv, &self.game_state.board);
            let score = policy_logits[token];
            (mv, score)
        }).collect();

        move_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (mv, _) in move_scores {
            // --- State Push ---
            // Clone the current state before making a move.
            let original_state = self.game_state.clone();

            // Make the move on the board and update the Mamba state.
            let token = move_to_token(&mv, &self.game_state.board);
            self.game_state.board = self.game_state.board.make_move_new(mv);
            let (_, mut score) = self.engine.forward(token, &mut self.game_state);

            // --- Recurse or Evaluate ---
            if depth > 1 {
                let (_, recursive_score) = self.alphabeta(-beta, -alpha, depth - 1);
                score = -recursive_score; // Negamax framework
            }

            // --- State Pop ---
            // Restore the original state for the next move at this ply.
            self.game_state = original_state;

            // --- Update Alpha and Best Move ---
            if score > best_score {
                best_score = score;
                best_move = Some(mv);
            }

            alpha = alpha.max(best_score);

            // --- Pruning ---
            if alpha >= beta {
                break;
            }
        }

        (best_move, best_score)
    }
}
