use crate::board::Board;
use crate::defs::{Color, PieceType, PIECE_TYPE_COUNT};

// --- Material Values (MG and EG) ---
const MG_VALUES: [i32; 6] = [82, 337, 365, 477, 1025, 0];
const EG_VALUES: [i32; 6] = [94, 281, 297, 512,  936, 0];

// --- Piece-Square Tables (PeSTo-inspired) ---
// Note: White perspective. Mirror for Black.

#[rustfmt::skip]
const MG_PAWN_PST: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0,
];

#[rustfmt::skip]
const EG_PAWN_PST: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
];

#[rustfmt::skip]
const MG_KNIGHT_PST: [i32; 64] = [
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -74, -58, -33, -17, -28, -73,  -65,
];

#[rustfmt::skip]
const EG_KNIGHT_PST: [i32; 64] = [
    -58, -38, -13, -28, -31, -27, -44, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,   1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
];

#[rustfmt::skip]
const MG_BISHOP_PST: [i32; 64] = [
    -33,   4, -39, -21, -20, -39,   4, -33,
    -15,  12,  15,   2,  15,   8,  12, -15,
    -14,  13,  19,  24,  39,  31,  13, -14,
    -12,  13,  28,  34,  21,  28,  13, -12,
    -16,  14,  32,  24,  35,  17,  14, -16,
    -21,  15,  11,  15,   7,  13,  15, -21,
    -22,  14,   6,   7,  11,  10,  14, -22,
    -36, -11, -23, -26, -26, -23, -11, -36,
];

#[rustfmt::skip]
const EG_BISHOP_PST: [i32; 64] = [
    -14, -21, -11,  -8, -7,  -9, -17, -24,
     -8,  -4,   7,  -1, -3, -13,  -4, -14,
      2,  -8,   0,  -1, -2,   6,   0,   4,
     -3,   9,  12,   9, 14,  10,   3,   2,
     -6,   3,  13,  19,  7,  10,  -3,  -9,
    -12,  -3,   8,  10, 13,   3,  -7, -15,
    -14, -18,  -7,  -1,  4,  -9, -15, -27,
    -23,  -9, -23,  -5, -9, -16,  -5, -17,
];

#[rustfmt::skip]
const MG_ROOK_PST: [i32; 64] = [
     -3,   5,  10,  12,  12,  10,   5,  -3,
     -5,   5,  10,  12,  12,  10,   5,  -5,
     -5,   5,  10,  12,  12,  10,   5,  -5,
     -5,   5,  10,  12,  12,  10,   5,  -5,
     -5,   5,  10,  12,  12,  10,   5,  -5,
     -5,   5,  10,  12,  12,  10,   5,  -5,
     -5,   5,  10,  12,  12,  10,   5,  -5,
     -3,   5,  10,  12,  12,  10,   5,  -3,
];

#[rustfmt::skip]
const EG_ROOK_PST: [i32; 64] = [
     13,  10,  18,  15,  12,  12,   8,   5,
     11,  13,  13,  11,  -3,   3,   8,   3,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
      1,   1,   1,   1,   0,  -3,  -5,  -8,
     -2,  -2,  -2,  -2,  -2,  -5,  -8,  -9,
     -5,  -5,  -5,  -5,  -5,  -7,  -8,  -9,
     -3,  -5,  -8, -11, -12, -13, -15, -19,
    -47, -27, -25, -22, -23, -23, -31, -40,
];

#[rustfmt::skip]
const MG_QUEEN_PST: [i32; 64] = [
     -2,   4,  15,  12,   7,  21,   2,  -6,
    -17, -19,  -1,   9,  31,  14,  20, -31,
     -9,  22,  22,  27,  35,  51,  57,  21,
     -6,  17,  49,  35,  58,  84,  42,  23,
    -15,  33,  40,  71,  73,  71,  38,   2,
    -23,  -6,  31,  21,  19,  42,  24, -23,
    -36, -28,  13,  40,  35,  14, -22, -32,
    -52, -54, -18,  -5, -17, -18, -45, -54,
];

#[rustfmt::skip]
const EG_QUEEN_PST: [i32; 64] = [
     -9,  22,  22,  27,  27,  19,  10,  17,
     -4,  17,  18,  28,  52,  54,  33,  28,
     -8,  22,  35,  71,  77,  94,  93,  27,
     -9,  17,  54,  84,  89,  71,  44,  18,
    -12,  15,  33,  73,  77,  58,  32,   2,
    -18,   3,  19,  40,  44,  25,   8,  -9,
    -54, -40, -12,  10,  12,   8, -22, -28,
    -60, -20, -10,  11,  15,  -8, -30, -60,
];

#[rustfmt::skip]
const MG_KING_PST: [i32; 64] = [
    -15,  36,  12, -54,   8, -28,  24,  14,
      1,   7,  -8, -64, -11, -16,   1,  16,
    -13, -19, -47, -72, -99, -47, -22, -14,
    -55, -43, -52, -97, -91, -43, -55, -53,
    -55, -40, -39, -56, -56, -39, -40, -55,
    -23, -23, -23, -23, -23, -23, -23, -23,
     31,  46,  13,  -7,  15,  33,  34,  45,
     18,  36,  33,  -2,   5,  20,  37,  26,
];

#[rustfmt::skip]
const EG_KING_PST: [i32; 64] = [
    -50, -20,   0,  10,  10,   0, -20, -50,
    -30,  10,  30,  35,  35,  30,  10, -30,
    -20,  30,  40,  45,  45,  40,  30, -20,
    -10,  35,  45,  50,  50,  45,  35, -10,
    -10,  35,  45,  50,  50,  45,  35, -10,
    -20,  30,  40,  45,  45,  40,  30, -20,
    -30,  10,  30,  35,  35,  30,  10, -30,
    -50, -20,   0,  10,  10,   0, -20, -50,
];

const MG_PSTS: [&[i32; 64]; 6] = [
    &MG_PAWN_PST, &MG_KNIGHT_PST, &MG_BISHOP_PST, &MG_ROOK_PST, &MG_QUEEN_PST, &MG_KING_PST
];

const EG_PSTS: [&[i32; 64]; 6] = [
    &EG_PAWN_PST, &EG_KNIGHT_PST, &EG_BISHOP_PST, &EG_ROOK_PST, &EG_QUEEN_PST, &EG_KING_PST
];

// Phase values for tapered evaluation
const KNIGHT_PHASE: i32 = 1;
const BISHOP_PHASE: i32 = 1;
const ROOK_PHASE: i32 = 2;
const QUEEN_PHASE: i32 = 4;
const TOTAL_PHASE: i32 = 24; // 4*1 + 4*1 + 4*2 + 2*4

pub fn evaluate(board: &Board) -> i32 {
    let mut mg_score = 0;
    let mut eg_score = 0;
    let mut phase = 0;

    // Evaluate White
    for pt in 0..PIECE_TYPE_COUNT {
        let mut bb = board.pieces[Color::White as usize][pt];
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            bb &= bb - 1;

            mg_score += MG_VALUES[pt] + MG_PSTS[pt][sq];
            eg_score += EG_VALUES[pt] + EG_PSTS[pt][sq];

            // Update phase (exclude pawns and kings)
            phase += match pt {
                1 => KNIGHT_PHASE,
                2 => BISHOP_PHASE,
                3 => ROOK_PHASE,
                4 => QUEEN_PHASE,
                _ => 0
            };
        }
    }

    // Evaluate Black
    for pt in 0..PIECE_TYPE_COUNT {
        let mut bb = board.pieces[Color::Black as usize][pt];
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            bb &= bb - 1;

            let mirrored_sq = sq ^ 56;
            mg_score -= MG_VALUES[pt] + MG_PSTS[pt][mirrored_sq];
            eg_score -= EG_VALUES[pt] + EG_PSTS[pt][mirrored_sq];

            phase += match pt {
                1 => KNIGHT_PHASE,
                2 => BISHOP_PHASE,
                3 => ROOK_PHASE,
                4 => QUEEN_PHASE,
                _ => 0
            };
        }
    }

    // Tapering
    let mg_phase = if phase > TOTAL_PHASE { TOTAL_PHASE } else { phase };
    let eg_phase = TOTAL_PHASE - mg_phase;
    
    let final_score = (mg_score * mg_phase + eg_score * eg_phase) / TOTAL_PHASE;

    // Still add some of our previous bonuses? 
    // PeSTo tables already cover a lot, but passed pawns are still good.
    // Simplifying: we'll stick to PeSTo + Bishop Pair bonus
    
    let white_bps = board.pieces[0][PieceType::Bishop as usize].count_ones();
    let black_bps = board.pieces[1][PieceType::Bishop as usize].count_ones();
    let bp_bonus = if white_bps >= 2 { 40 } else { 0 } - if black_bps >= 2 { 40 } else { 0 };

    final_score + bp_bonus
}
