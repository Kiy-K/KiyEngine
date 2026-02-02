// src/search/eval.rs

use chess::{Board, Color, Piece, Square, BitBoard};

pub const EVAL_PAWN: i32 = 100;
pub const EVAL_KNIGHT: i32 = 320;
pub const EVAL_BISHOP: i32 = 330;
pub const EVAL_ROOK: i32 = 500;
pub const EVAL_QUEEN: i32 = 900;
pub const EVAL_KING: i32 = 20000;

// opening values to discourage early sacrifices
pub const OPENING_KNIGHT: i32 = 450; 
pub const OPENING_BISHOP: i32 = 460;

const PAWN_PST: [i32; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 50, 50, 50, 50, 50, 50, 10, 10, 20, 30, 30, 20, 10, 10, 5, 5,
    10, 25, 25, 10, 5, 5, 0, 0, 0, 20, 20, 0, 0, 0, 5, -5, -10, 0, 0, -10, -5, 5, 5, 10, 10, -20,
    -20, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0,
];

const KNIGHT_PST: [i32; 64] = [
    -50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 0, 0, 0, -20, -40, -30, 0, 10, 15, 15, 10,
    0, -30, -30, 5, 15, 20, 20, 15, 5, -30, -30, 0, 15, 20, 20, 15, 0, -30, -30, 5, 10, 15, 15, 10,
    5, -30, -40, -20, 0, 5, 5, 0, -20, -40, -50, -40, -30, -30, -30, -30, -40, -50,
];

const BISHOP_PST: [i32; 64] = [
    -20, -10, -10, -10, -10, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 10, 10, 5, 0,
    -10, -10, 5, 5, 10, 10, 5, 5, -10, -10, 0, 10, 10, 10, 10, 0, -10, -10, 10, 10, 10, 10, 10, 10,
    -10, -10, 5, 0, 0, 0, 0, 5, -10, -20, -10, -10, -10, -10, -10, -10, -20,
];

const ROOK_PST: [i32; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, 10, 10, 10, 10, 5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0,
    0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, 0, 0,
    0, 5, 5, 0, 0, 0,
];

const QUEEN_PST: [i32; 64] = [
    -20, -10, -10, -5, -5, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5, 0, 0, 5, 5, 5, 5, 0, -5, -10, 5, 5, 5, 5, 5, 0, -10, -10, 0, 5, 0, 0,
    0, 0, -10, -20, -10, -10, -5, -5, -10, -10, -20,
];

const KING_PST: [i32; 64] = [
    -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40,
    -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -20, -30, -30, -40, -40, -30,
    -30, -20, -10, -20, -20, -20, -20, -20, -20, -10, 20, 20, 0, 0, 0, 0, 20, 20, 20, 30, 10, 0, 0,
    10, 30, 20,
];

pub fn evaluate(board: &Board, ply: usize) -> i32 {
    let mut score = 0;

    // Opening phase: first 10 moves (roughly)
    let is_opening = ply <= 20;

    for sq in *board.combined() {
        let piece = board.piece_on(sq).unwrap();
        let color = board.color_on(sq).unwrap();

        let val = match piece {
            Piece::Pawn => EVAL_PAWN + PAWN_PST[get_pst_idx(sq, color)],
            Piece::Knight => {
                let base = if is_opening { OPENING_KNIGHT } else { EVAL_KNIGHT };
                base + KNIGHT_PST[get_pst_idx(sq, color)]
            },
            Piece::Bishop => {
                let base = if is_opening { OPENING_BISHOP } else { EVAL_BISHOP };
                base + BISHOP_PST[get_pst_idx(sq, color)]
            },
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

    score += king_safety(board, Color::White);
    score -= king_safety(board, Color::Black);

    if board.side_to_move() == Color::White {
        score
    } else {
        -score
    }
}

fn king_safety(board: &Board, color: Color) -> i32 {
    let king_sq = board.king_square(color);
    let mut safety_score = 0;
    
    let enemy_color = !color;
    let attackers = board.color_combined(enemy_color);
    let king_zone = get_king_zone(king_sq);
    
    let attackers_bb = king_zone & attackers;
    let attackers_count = attackers_bb.popcnt() as i32;
    // Penalty from 40 to 60 per piece for V4.3.2
    safety_score -= attackers_count * 60;
    
    // Check penalty from 150 to 250
    if board.checkers().popcnt() > 0 && board.side_to_move() == color {
        safety_score -= 250;
    }
    
    safety_score
}

fn get_king_zone(sq: Square) -> BitBoard {
    let mut mask = 0u64;
    let r = sq.get_rank().to_index() as i32;
    let f = sq.get_file().to_index() as i32;
    
    for dr in -1..=1 {
        for df in -1..=1 {
            let nr = r + dr;
            let nf = f + df;
            if nr >= 0 && nr < 8 && nf >= 0 && nf < 8 {
                mask |= 1 << (nr * 8 + nf);
            }
        }
    }
    BitBoard::new(mask)
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
