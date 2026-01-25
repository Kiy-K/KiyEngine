pub mod defs;
pub mod board;
pub mod evaluate;
pub mod mv;
pub mod movegen;
pub mod search;
pub mod tt;
pub mod magic;

use std::sync::atomic::AtomicBool;
pub static STOP_SEARCH: AtomicBool = AtomicBool::new(false);
pub static SEARCHING: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::defs::{Color, PieceType};

    #[test]
    fn test_initial_position() {
        let board = board::Board::new();
        assert_ne!(board.occupancy[Color::White as usize], 0);
        let score = evaluate::evaluate(&board);
        assert_eq!(score, 0, "Initial position evaluation should be 0");
    }
}
