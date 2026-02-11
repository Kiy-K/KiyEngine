//! # KiyEngine - GGUF Format with BitNet Evaluation
//!
//! A high-performance AI chess engine with hybrid evaluation system
//! combining fast material/PST evaluation with deep BitNet inference.

pub mod book;
pub mod constants;
pub mod engine;
pub mod eval;
pub mod nnue;
pub mod search;
pub mod simd;
pub mod uci;

#[cfg(test)]
mod tests {
    use super::*;
    use chess::{Board, ChessMove};
    use std::str::FromStr;

    #[test]
    fn test_move_codec() {
        let board = Board::default();
        let mv = ChessMove::from_str("e2e4").unwrap();
        let token = uci::MoveCodec::move_to_token(&mv);
        let decoded = uci::MoveCodec::token_to_move(token, &board);

        assert_eq!(Some(mv), decoded);
    }

    #[test]
    fn test_transposition_table() {
        use crate::search::tt::{AtomicTT, TTFlag};

        let tt = AtomicTT::new(1); // 1MB table

        // Store an entry
        tt.store(
            12345,
            100,
            Some(ChessMove::from_str("e2e4").unwrap()),
            10,
            TTFlag::Exact,
            0,
        );

        // Retrieve it
        let retrieved = tt.probe(12345);

        assert!(retrieved.is_some());
        let (score, best_move, depth, flag) = retrieved.unwrap();
        assert_eq!(score, 100);
        assert_eq!(depth, 10);
        assert_eq!(flag, TTFlag::Exact);
        assert!(best_move.is_some());
    }

    #[test]
    fn test_engine_initialization() {
        let engine = engine::Engine::new();
        assert!(engine.is_ok());
    }
}
