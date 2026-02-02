// src/search/eval.rs

use chess::{Board, Color, Piece, Square};

pub const EVAL_PAWN: i32 = 100;
pub const EVAL_KNIGHT: i32 = 320;
pub const EVAL_BISHOP: i32 = 330;
pub const EVAL_ROOK: i32 = 500;
pub const EVAL_QUEEN: i32 = 900;
pub const EVAL_KING: i32 = 20000;

// Simple PSQT for optimization (oriented for White)
const PAWN_PST: [i32; 64] = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
];

const KNIGHT_PST: [i32; 64] = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
];

const BISHOP_PST: [i32; 64] = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
];

const ROOK_PST: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0
];

const QUEEN_PST: [i32; 64] = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
];

const KING_PST: [i32; 64] = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
];

pub fn evaluate(board: &Board) -> i32 {
    let mut score = 0;
    
    for sq in *board.combined() {
        let piece = board.piece_on(sq).unwrap();
        let color = board.color_on(sq).unwrap();
        
        let val = match piece {
            Piece::Pawn => EVAL_PAWN + PAWN_PST[get_pst_idx(sq, color)],
            Piece::Knight => EVAL_KNIGHT + KNIGHT_PST[get_pst_idx(sq, color)],
            Piece::Bishop => EVAL_BISHOP + BISHOP_PST[get_pst_idx(sq, color)],
            Piece::Rook => EVAL_ROOK + ROOK_PST[get_pst_idx(sq, color)],
            Piece::Queen => EVAL_QUEEN + QUEEN_PST[get_pst_idx(sq, color)],
            Piece::King => EVAL_KING + KING_PST[get_pst_idx(sq, color)],
        };
        
        if color == Color::White {
            score += val;
        } else {
            score -= val;
        }
    }
    
    if board.side_to_move() == Color::White { score } else { -score }
}

fn get_pst_idx(sq: Square, color: Color) -> usize {
    let rank = sq.get_rank().to_index();
    let file = sq.get_file().to_index();
    if color == Color::White {
        (7 - rank) * 8 + file
    } else {
        rank * 8 + file
    }
}
