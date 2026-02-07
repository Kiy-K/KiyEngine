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

        // If White, use Rank index as is (0 at bottom).
        // If Black, mirror the Rank (0 -> 7, 7 -> 0).
        // The table is defined from White's perspective (Rank 0 is bottom).
        // Wait, standard chess crate Square index 0 is A1.
        // My tables are visually layout out from A8 to H1? No, typical array layout 0..63.
        // Let's assume standard 0=A1.. 63=H8.
        // The array visual above:
        // Top row (index 0..7) -> A1..H1?
        // No, typically tables are written Rank 8 down to Rank 1 for readability, or Rank 1 up.
        // Let's assume Rank 1 (A1..H1) is at the BOTTOM of the visual block.
        // So the last row in code is the first row on board (Rank 1).

        // Let's correct specific indices.
        // Rank 0 (A1-H1) is the last row in the array definition above.
        // So idx = 0 is A1.
        // If I access `table[square.to_index()]`, I need to match the visual layout.
        // Visual layout above: Top lines are low indices?
        // Usually arrays are `[0, 1, 2...]`.
        // If I write `const T = [ a, b, ... ]`, `T[0]` is `a`.
        // If `a` corresponds to A1, then the top row in code is Rank 1.
        // BUT, traditionally `[0..63]` maps A1..H8.
        // Looking at PAWN_TABLE:
        // The first row `0,0...` is likely Rank 1 (where white pawns can't be).
        // The second row `50,50...` is Rank 2 (starting).
        // So the Top Line in Code = Rank 1.

        let idx = square.to_index();

        if color == Color::White {
            // White: A1 is 0. Access directly.
            // But we need to ensure the table matches.
            // If Top Code Row = Rank 1 (idx 0..7).
            // Then `table[idx]` works.
            table[idx]
        } else {
            // Black: Mirror the rank.
            // Square (rank, file). New Rank = 7 - rank.
            // Mirrored Index = (7 - rank) * 8 + file.
            // Or simpler: idx ^ 56 (flips vertical).
            table[idx ^ 56]
        }
    }
}
