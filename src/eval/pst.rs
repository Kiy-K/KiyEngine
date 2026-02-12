use chess::{Color, Piece, Square};

// PeSTO Piece-Square Tables (Midgame)
// Values are generally in centipawns (cp).
// Flipped for Black automatically by the logic or we define symmetric tables.
// We will define from White's perspective. "Rank 1" is index 0-7.

#[rustfmt::skip]
const PAWN_TABLE: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
];

#[rustfmt::skip]
const KNIGHT_TABLE: [i32; 64] = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
];

#[rustfmt::skip]
const BISHOP_TABLE: [i32; 64] = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
];

#[rustfmt::skip]
const ROOK_TABLE: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10,  10,  10,  10,  10,   5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   0,   5,   5,   0,   0,   0,
];

#[rustfmt::skip]
const QUEEN_TABLE: [i32; 64] = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
];

#[rustfmt::skip]
const KING_MID_TABLE: [i32; 64] = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
];

// Endgame King PST: king should centralize in endgames
#[rustfmt::skip]
const KING_END_TABLE: [i32; 64] = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
];

// Endgame Pawn PST: advanced pawns are much more valuable
#[rustfmt::skip]
const PAWN_END_TABLE: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
     80,  80,  80,  80,  80,  80,  80,  80,
     50,  50,  50,  50,  50,  50,  50,  50,
     30,  30,  30,  30,  30,  30,  30,  30,
     20,  20,  20,  20,  20,  20,  20,  20,
     10,  10,  10,  10,  10,  10,  10,  10,
     10,  10,  10,  10,  10,  10,  10,  10,
      0,   0,   0,   0,   0,   0,   0,   0,
];

/// Game phase values for each piece type (used for midgame/endgame interpolation)
/// Total phase = 24 (4*1 + 4*1 + 4*2 + 2*4 = 24 for all minors+rooks+queens)
const PHASE_VALS: [(Piece, i32); 5] = [
    (Piece::Pawn, 0),
    (Piece::Knight, 1),
    (Piece::Bishop, 1),
    (Piece::Rook, 2),
    (Piece::Queen, 4),
];
const TOTAL_PHASE: i32 = 24;

pub struct PST;

impl PST {
    #[inline(always)]
    pub fn get_score(piece: Piece, square: Square, color: Color) -> i32 {
        let table = match piece {
            Piece::Pawn => &PAWN_TABLE,
            Piece::Knight => &KNIGHT_TABLE,
            Piece::Bishop => &BISHOP_TABLE,
            Piece::Rook => &ROOK_TABLE,
            Piece::Queen => &QUEEN_TABLE,
            Piece::King => &KING_MID_TABLE,
        };

        let idx = square.to_index();
        if color == Color::White {
            table[idx]
        } else {
            table[idx ^ 56]
        }
    }

    /// Get endgame PST score (only King and Pawn differ; others return midgame)
    #[inline(always)]
    pub fn get_score_eg(piece: Piece, square: Square, color: Color) -> i32 {
        let table = match piece {
            Piece::King => &KING_END_TABLE,
            Piece::Pawn => &PAWN_END_TABLE,
            _ => return Self::get_score(piece, square, color),
        };
        let idx = square.to_index();
        if color == Color::White {
            table[idx]
        } else {
            table[idx ^ 56]
        }
    }

    /// Compute game phase (0=endgame, TOTAL_PHASE=opening) based on remaining material
    #[inline]
    pub fn game_phase(board: &chess::Board) -> i32 {
        let mut phase = 0i32;
        for &(piece, val) in &PHASE_VALS {
            if val == 0 {
                continue;
            }
            let count = board.pieces(piece).popcnt() as i32;
            phase += count * val;
        }
        phase.min(TOTAL_PHASE)
    }

    /// Interpolate between midgame and endgame PST score
    /// phase = 0 → pure endgame, phase = TOTAL_PHASE → pure midgame
    #[inline]
    pub fn get_score_tapered(piece: Piece, square: Square, color: Color, phase: i32) -> i32 {
        let mg = Self::get_score(piece, square, color);
        let eg = Self::get_score_eg(piece, square, color);
        (mg * phase + eg * (TOTAL_PHASE - phase)) / TOTAL_PHASE
    }
}
