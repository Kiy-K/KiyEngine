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
    FILE_MASKS[1],                 // A: only B
    FILE_MASKS[0] | FILE_MASKS[2], // B: A+C
    FILE_MASKS[1] | FILE_MASKS[3], // C: B+D
    FILE_MASKS[2] | FILE_MASKS[4], // D: C+E
    FILE_MASKS[3] | FILE_MASKS[5], // E: D+F
    FILE_MASKS[4] | FILE_MASKS[6], // F: E+G
    FILE_MASKS[5] | FILE_MASKS[7], // G: F+H
    FILE_MASKS[6],                 // H: only G
];

// Passed pawn bonus by rank (from White's perspective, rank 1-8 index 0-7)
// Exponential scaling: each rank roughly doubles the bonus
const PASSED_PAWN_BONUS: [i32; 8] = [0, 10, 20, 45, 80, 140, 250, 0];
// Extra bonus when passed pawn is protected by another pawn
const PROTECTED_PASSER_BONUS: [i32; 8] = [0, 5, 10, 20, 40, 70, 125, 0];

// Total game phase for endgame scaling (same as pst.rs)
const TOTAL_PHASE: i32 = 24;

// Pawn shield masks: squares directly in front of king (2 ranks deep)
// These are relative masks applied based on king position
const PAWN_SHIELD_BONUS: i32 = 10; // per pawn in the shield zone

/// Enhanced material + tapered PST + king safety + mobility evaluation
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
    let occupied = *board.combined();

    // Compute game phase for tapered eval
    let phase = PST::game_phase(board);

    // --- Material + Tapered PST ---
    for &color in &[Color::White, Color::Black] {
        let mult = if color == Color::White { 1 } else { -1 };
        let color_bb = board.color_combined(color);

        for &(piece, val) in &material {
            let bb = board.pieces(piece) & color_bb;
            for sq in bb {
                score += val * mult;
                score += PST::get_score_tapered(piece, sq, color, phase) * mult;
            }
        }
        let ksq = board.king_square(color);
        score += PST::get_score_tapered(Piece::King, ksq, color, phase) * mult;
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

    // --- King Safety: Pawn Shield ---
    // Only relevant in midgame (phase > 8)
    if phase > 8 {
        score += king_pawn_shield(board, Color::White, white_pawns);
        score -= king_pawn_shield(board, Color::Black, black_pawns);
    }

    // --- Piece Mobility (lightweight) ---
    // Knight and Bishop mobility based on attack squares (excluding own pieces)
    score += mobility_score(board, Color::White, occupied, white_bb);
    score -= mobility_score(board, Color::Black, occupied, black_bb);

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
            // Endgame scaling: passed pawns are more valuable with fewer pieces
            let eg_scale = (TOTAL_PHASE - phase).max(0) as i32;
            let base = PASSED_PAWN_BONUS[rank];
            score += base + (base * eg_scale) / (TOTAL_PHASE * 2);

            // Protected passer: defended by another pawn (diagonal behind)
            let prot_mask = if rank > 0 {
                let behind_rank = (rank - 1) * 8;
                let mut m = 0u64;
                if file > 0 { m |= 1u64 << (behind_rank + file - 1); }
                if file < 7 { m |= 1u64 << (behind_rank + file + 1); }
                m
            } else { 0 };
            if (white_pawns & BitBoard(prot_mask)).0 != 0 {
                score += PROTECTED_PASSER_BONUS[rank];
            }
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
            let brank = 7 - rank;
            let eg_scale = (TOTAL_PHASE - phase).max(0) as i32;
            let base = PASSED_PAWN_BONUS[brank];
            score -= base + (base * eg_scale) / (TOTAL_PHASE * 2);

            // Protected passer for Black
            let prot_mask = if rank < 7 {
                let behind_rank = (rank + 1) * 8;
                let mut m = 0u64;
                if file > 0 { m |= 1u64 << (behind_rank + file - 1); }
                if file < 7 { m |= 1u64 << (behind_rank + file + 1); }
                m
            } else { 0 };
            if (black_pawns & BitBoard(prot_mask)).0 != 0 {
                score -= PROTECTED_PASSER_BONUS[brank];
            }
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

    // --- Minor piece development penalty (opening only, phase > 16) ---
    // Penalize knights/bishops still on starting squares
    if phase > 16 {
        // White undeveloped pieces
        let w_knights = board.pieces(Piece::Knight) & white_bb;
        let w_bishops = board.pieces(Piece::Bishop) & white_bb;
        // Starting squares: Nb1(1), Ng1(6), Bc1(2), Bf1(5)
        if (w_knights & BitBoard(1u64 << 1)).0 != 0 { score -= 15; }  // Nb1
        if (w_knights & BitBoard(1u64 << 6)).0 != 0 { score -= 15; }  // Ng1
        if (w_bishops & BitBoard(1u64 << 2)).0 != 0 { score -= 10; }  // Bc1
        if (w_bishops & BitBoard(1u64 << 5)).0 != 0 { score -= 10; }  // Bf1

        // Black undeveloped pieces
        let b_knights = board.pieces(Piece::Knight) & black_bb;
        let b_bishops = board.pieces(Piece::Bishop) & black_bb;
        // Starting squares: Nb8(57), Ng8(62), Bc8(58), Bf8(61)
        if (b_knights & BitBoard(1u64 << 57)).0 != 0 { score += 15; }  // Nb8
        if (b_knights & BitBoard(1u64 << 62)).0 != 0 { score += 15; }  // Ng8
        if (b_bishops & BitBoard(1u64 << 58)).0 != 0 { score += 10; }  // Bc8
        if (b_bishops & BitBoard(1u64 << 61)).0 != 0 { score += 10; }  // Bf8
    }

    if board.side_to_move() == Color::White {
        score
    } else {
        -score
    }
}

/// King pawn shield evaluation: bonus for pawns in front of the castled king
#[inline]
fn king_pawn_shield(board: &Board, color: Color, our_pawns: BitBoard) -> i32 {
    let ksq = board.king_square(color);
    let k_file = ksq.to_index() % 8;
    let k_rank = ksq.to_index() / 8;

    // Only evaluate pawn shield if king is on flanks (typical castled position)
    if k_file >= 2 && k_file <= 5 {
        return 0;
    }

    let mut shield = 0i32;
    // Check 3 files around king (king file, and both adjacent if they exist)
    let start_file = if k_file > 0 { k_file - 1 } else { 0 };
    let end_file = if k_file < 7 { k_file + 1 } else { 7 };

    for f in start_file..=end_file {
        // Check 1-2 ranks ahead of king for friendly pawns
        let ahead1 = if color == Color::White {
            if k_rank < 7 {
                Some((k_rank + 1) * 8 + f)
            } else {
                None
            }
        } else {
            if k_rank > 0 {
                Some((k_rank - 1) * 8 + f)
            } else {
                None
            }
        };
        let ahead2 = if color == Color::White {
            if k_rank < 6 {
                Some((k_rank + 2) * 8 + f)
            } else {
                None
            }
        } else {
            if k_rank > 1 {
                Some((k_rank - 2) * 8 + f)
            } else {
                None
            }
        };

        if let Some(sq_idx) = ahead1 {
            if sq_idx < 64 && (our_pawns & BitBoard(1u64 << sq_idx)).0 != 0 {
                shield += PAWN_SHIELD_BONUS;
            }
        }
        if let Some(sq_idx) = ahead2 {
            if sq_idx < 64 && (our_pawns & BitBoard(1u64 << sq_idx)).0 != 0 {
                shield += PAWN_SHIELD_BONUS / 2; // Half bonus for rank+2
            }
        }
    }
    shield
}

/// Lightweight piece mobility: count attack squares for knights and bishops
#[inline]
fn mobility_score(board: &Board, color: Color, occupied: BitBoard, our_bb: BitBoard) -> i32 {
    let mut mob = 0i32;

    // Knight mobility: 2cp per accessible square
    let knights = *board.pieces(Piece::Knight) & *board.color_combined(color);
    for sq in knights {
        let attacks = chess::get_knight_moves(sq);
        let accessible = (attacks & !our_bb).0.count_ones() as i32;
        mob += accessible * 2;
    }

    // Bishop mobility: 3cp per accessible square (bishops benefit more from open diagonals)
    let bishops = *board.pieces(Piece::Bishop) & *board.color_combined(color);
    for sq in bishops {
        let attacks = chess::get_bishop_moves(sq, occupied);
        let accessible = (attacks & !our_bb).0.count_ones() as i32;
        mob += accessible * 3;
    }

    // Rook mobility: 2cp per accessible square
    let rooks = *board.pieces(Piece::Rook) & *board.color_combined(color);
    for sq in rooks {
        let attacks = chess::get_rook_moves(sq, occupied);
        let accessible = (attacks & !our_bb).0.count_ones() as i32;
        mob += accessible * 2;
    }

    mob
}
