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

        let mut val = match piece {
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

        // Opening centralization bonus: encourage fighting for e4/d4/e5/d5.
        if is_opening {
            let idx = get_pst_idx(sq, color);
            // Indices for the 4 central squares in white's perspective.
            const CENTER_INDICES: [usize; 4] = [27, 28, 35, 36];
            if CENTER_INDICES.contains(&idx) {
                match piece {
                    Piece::Pawn => val += 15,
                    Piece::Knight => val += 25,
                    _ => {}
                }
            }

            // Edge pawn penalty in the opening: discourage early a/h pawn pushes.
            if piece == Piece::Pawn {
                let file = sq.get_file().to_index();
                let rank = sq.get_rank().to_index();
                if (file == 0 || file == 7) && rank >= 2 {
                    // File a/h and advanced beyond starting rank.
                    val -= 15;
                }
            }
        }

        if color == Color::White {
            score += val;
        } else {
            score -= val;
        }
    }

    score += king_safety(board, Color::White);
    score -= king_safety(board, Color::Black);

    // Pawn structure bonuses/penalties (isolated, doubled, passed)
    score += pawn_structure(board, Color::White);
    score -= pawn_structure(board, Color::Black);

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

    // Pawn shield in front of the king (files around king, two ranks forward)
    let king_file = king_sq.get_file().to_index() as i32;
    let king_rank = king_sq.get_rank().to_index() as i32;
    let friendly_pawns = board.pieces(Piece::Pawn) & board.color_combined(color);

    let mut shield_pawns = 0;
    for df in -1..=1 {
        let file = king_file + df;
        if file < 0 || file >= 8 {
            continue;
        }
        for dr in 1..=2 {
            let rank = if color == Color::White {
                king_rank + dr
            } else {
                king_rank - dr
            };
            if rank < 0 || rank >= 8 {
                continue;
            }
            let idx = rank * 8 + file;
            let mask = BitBoard::new(1u64 << idx);
            if (friendly_pawns & mask).popcnt() > 0 {
                shield_pawns += 1;
            }
        }
    }
    // Reward good pawn shield, up to 3 pawns.
    safety_score += shield_pawns.min(3) * 15;

    // Penalty for open files near the king (no friendly pawn on that file)
    for df in -1..=1 {
        let file = king_file + df;
        if file < 0 || file >= 8 {
            continue;
        }
        let mut has_pawn_on_file = false;
        for rank in 0..8 {
            let idx = rank * 8 + file;
            let mask = BitBoard::new(1u64 << idx);
            if (friendly_pawns & mask).popcnt() > 0 {
                has_pawn_on_file = true;
                break;
            }
        }
        if !has_pawn_on_file {
            safety_score -= 10;
        }
    }
    
    safety_score
}

fn pawn_structure(board: &Board, color: Color) -> i32 {
    let mut score = 0;
    let pawns = board.pieces(Piece::Pawn) & board.color_combined(color);
    let enemy_pawns = board.pieces(Piece::Pawn) & board.color_combined(!color);

    // File masks for quick access.
    let mut file_masks = [0u64; 8];
    for file in 0..8 {
        let mut mask = 0u64;
        for rank in 0..8 {
            mask |= 1u64 << (rank * 8 + file);
        }
        file_masks[file as usize] = mask;
    }

    // Doubled and isolated pawns.
    for file in 0..8 {
        let file_bb = BitBoard::new(file_masks[file]);
        let pawns_on_file = (pawns & file_bb).popcnt() as i32;
        if pawns_on_file == 0 {
            continue;
        }

        // Doubled pawns penalty: each extra pawn beyond the first.
        if pawns_on_file > 1 {
            score -= (pawns_on_file - 1) * 12;
        }

        // Isolated pawn: no friendly pawns on adjacent files.
        let mut has_adjacent = false;
        for df in [-1, 1] {
            let adj_file = file as i32 + df;
            if adj_file < 0 || adj_file >= 8 {
                continue;
            }
            let adj_bb = BitBoard::new(file_masks[adj_file as usize]);
            if (pawns & adj_bb).popcnt() > 0 {
                has_adjacent = true;
                break;
            }
        }
        if !has_adjacent {
            score -= 10;
        }
    }

    // Passed pawns: no enemy pawn on same/adjacent files ahead.
    for sq in pawns {
        let file = sq.get_file().to_index() as i32;
        let rank = sq.get_rank().to_index() as i32;

        let mut is_passed = true;
        for df in -1..=1 {
            let f = file + df;
            if f < 0 || f >= 8 {
                continue;
            }
            for r in 0..8 {
                let ahead = if color == Color::White {
                    r > rank
                } else {
                    r < rank
                };
                if !ahead {
                    continue;
                }
                let idx = r * 8 + f;
                let mask = BitBoard::new(1u64 << idx);
                if (enemy_pawns & mask).popcnt() > 0 {
                    is_passed = false;
                    break;
                }
            }
            if !is_passed {
                break;
            }
        }

        if is_passed {
            // Reward passed pawns more the further they advance.
            let rank_from_start = if color == Color::White {
                rank as i32
            } else {
                7 - rank as i32
            };
            score += 20 + rank_from_start * 5;
        }
    }

    score
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
