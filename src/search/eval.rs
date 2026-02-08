use chess::{Board, Color, Piece};
use crate::eval::pst::PST;

/// Simple material + PST evaluation (fallback when neural network fails)
pub fn evaluate(board: &Board, _ply: usize) -> i32 {
    let mut score = 0;
    let material = [
        (Piece::Pawn, 100),
        (Piece::Knight, 320),
        (Piece::Bishop, 330),
        (Piece::Rook, 500),
        (Piece::Queen, 900),
    ];

    for &color in &[Color::White, Color::Black] {
        let mult = if color == Color::White { 1 } else { -1 };
        let color_bb = board.color_combined(color);

        for &(piece, val) in &material {
            let bb = board.pieces(piece) & color_bb;
            for sq in bb {
                score += val * mult;
                score += PST::get_score(piece, sq, color) * mult;
            }
        }
        let ksq = board.king_square(color);
        score += PST::get_score(Piece::King, ksq, color) * mult;
    }

    if board.side_to_move() == Color::White { score } else { -score }
}
