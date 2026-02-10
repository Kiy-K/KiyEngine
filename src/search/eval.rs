use crate::eval::pst::PST;
use chess::{BitBoard, Board, Color, Piece};

// File masks for pawn structure evaluation
const FILE_MASKS: [u64; 8] = [
    0x0101010101010101, // A-file
    0x0202020202020202, // B-file
    0x0404040404040404, // C-file
    0x0808080808080808, // D-file
    0x1010101010101010, // E-file
    0x2020202020202020, // F-file
    0x4040404040404040, // G-file
    0x8080808080808080, // H-file
];

// Adjacent file masks for isolated pawn detection
const ADJACENT_FILES: [u64; 8] = [
    FILE_MASKS[1],                       // A: only B
    FILE_MASKS[0] | FILE_MASKS[2],       // B: A+C
    FILE_MASKS[1] | FILE_MASKS[3],       // C: B+D
    FILE_MASKS[2] | FILE_MASKS[4],       // D: C+E
    FILE_MASKS[3] | FILE_MASKS[5],       // E: D+F
    FILE_MASKS[4] | FILE_MASKS[6],       // F: E+G
    FILE_MASKS[5] | FILE_MASKS[7],       // G: F+H
    FILE_MASKS[6],                       // H: only G
];

// Passed pawn bonus by rank (from White's perspective, rank 1-8 index 0-7)
const PASSED_PAWN_BONUS: [i32; 8] = [0, 5, 10, 20, 35, 60, 100, 0];

/// Enhanced material + PST + positional evaluation
pub fn evaluate(board: &Board, _ply: usize) -> i32 {
    let mut score = 0;
    let material = [
        (Piece::Pawn, 100),
        (Piece::Knight, 320),
        (Piece::Bishop, 330),
        (Piece::Rook, 500),
        (Piece::Queen, 900),
    ];

    let white_bb = *board.color_combined(Color::White);
    let black_bb = *board.color_combined(Color::Black);
    let white_pawns = board.pieces(Piece::Pawn) & white_bb;
    let black_pawns = board.pieces(Piece::Pawn) & black_bb;

    // --- Material + PST ---
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

    // --- Bishop pair bonus (30cp) ---
    let white_bishops = (board.pieces(Piece::Bishop) & white_bb).0.count_ones();
    let black_bishops = (board.pieces(Piece::Bishop) & black_bb).0.count_ones();
    if white_bishops >= 2 {
        score += 30;
    }
    if black_bishops >= 2 {
        score -= 30;
    }

    // --- Rook on open/semi-open file ---
    let white_rooks = board.pieces(Piece::Rook) & white_bb;
    let black_rooks = board.pieces(Piece::Rook) & black_bb;

    for sq in white_rooks {
        let file = sq.to_index() % 8;
        let file_mask = BitBoard(FILE_MASKS[file]);
        if (white_pawns & file_mask).0 == 0 {
            if (black_pawns & file_mask).0 == 0 {
                score += 20; // Open file
            } else {
                score += 10; // Semi-open file
            }
        }
    }
    for sq in black_rooks {
        let file = sq.to_index() % 8;
        let file_mask = BitBoard(FILE_MASKS[file]);
        if (black_pawns & file_mask).0 == 0 {
            if (white_pawns & file_mask).0 == 0 {
                score -= 20;
            } else {
                score -= 10;
            }
        }
    }

    // --- Pawn structure: passed pawns, doubled, isolated ---
    for sq in white_pawns {
        let idx = sq.to_index();
        let file = idx % 8;
        let rank = idx / 8; // 0=rank1, 7=rank8

        // Passed pawn: no enemy pawns on same or adjacent files ahead
        let file_and_adj = FILE_MASKS[file] | ADJACENT_FILES[file];
        let passed_mask = if rank < 7 {
            file_and_adj & !((1u64 << ((rank + 1) * 8)) - 1)
        } else {
            0
        };
        if (black_pawns & BitBoard(passed_mask)).0 == 0 {
            score += PASSED_PAWN_BONUS[rank];
        }

        // Doubled pawn: another friendly pawn on same file ahead
        let file_ahead: u64 = FILE_MASKS[file] & !((1u64 << (idx + 1)) - 1) & !(1u64 << idx);
        if (white_pawns & BitBoard(file_ahead)).0 != 0 {
            score -= 10;
        }

        // Isolated pawn: no friendly pawns on adjacent files
        if (white_pawns & BitBoard(ADJACENT_FILES[file])).0 == 0 {
            score -= 15;
        }
    }

    for sq in black_pawns {
        let idx = sq.to_index();
        let file = idx % 8;
        let rank = idx / 8;

        // Passed pawn (from Black's perspective — lower ranks are "ahead")
        let file_and_adj = FILE_MASKS[file] | ADJACENT_FILES[file];
        let passed_mask = if rank > 0 {
            file_and_adj & ((1u64 << (rank * 8)) - 1)
        } else {
            0
        };
        if (white_pawns & BitBoard(passed_mask)).0 == 0 {
            score -= PASSED_PAWN_BONUS[7 - rank];
        }

        // Doubled pawn
        let file_behind: u64 = FILE_MASKS[file] & ((1u64 << idx) - 1);
        if (black_pawns & BitBoard(file_behind)).0 != 0 {
            score += 10; // Penalty for black = bonus for white
        }

        // Isolated pawn
        if (black_pawns & BitBoard(ADJACENT_FILES[file])).0 == 0 {
            score += 15;
        }
    }

    if board.side_to_move() == Color::White {
        score
    } else {
        -score
    }
}
