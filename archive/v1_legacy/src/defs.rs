//! Definitions for the chess engine.
//! Contains basic types and constants.

/// A Bitboard is a 64-bit integer representing the board state for a specific piece or property.
/// Rank 1 is the least significant byte, file A is the least significant bit of each byte.
/// A1 = 0, H8 = 63.
pub type Bitboard = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    White,
    Black,
}

impl Color {
    pub fn opposite(&self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    pub fn from_usize(i: usize) -> Color {
        if i == 0 { Color::White } else { Color::Black }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PieceType {
    Pawn = 0,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

impl PieceType {
    pub fn from_usize(i: usize) -> PieceType {
        match i {
            0 => PieceType::Pawn,
            1 => PieceType::Knight,
            2 => PieceType::Bishop,
            3 => PieceType::Rook,
            4 => PieceType::Queen,
            _ => PieceType::King,
        }
    }
}

/// Helper constants for board squares to avoid magic numbers.
pub const A1: u8 = 0;
pub const H8: u8 = 63;

/// Number of pieces per color (P, N, B, R, Q, K)
pub const PIECE_TYPE_COUNT: usize = 6;
/// Number of colors
pub const COLOR_COUNT: usize = 2;

// Useful Bitboard constants
pub const EMPTY: Bitboard = 0;
pub const UNIVERSE: Bitboard = !0;

// Rank implementation details:
// Rank 1: 0xFF
// Rank 2: 0xFF << 8, etc.
pub const RANK_1: Bitboard = 0xFF;
pub const RANK_2: Bitboard = 0xFF << 8;
pub const RANK_7: Bitboard = 0xFF << 48;
pub const RANK_8: Bitboard = 0xFF << 56;

// File Masks
pub const FILE_A: Bitboard = 0x0101010101010101;
pub const FILE_B: Bitboard = FILE_A << 1;
pub const FILE_C: Bitboard = FILE_A << 2;
pub const FILE_D: Bitboard = FILE_A << 3;
pub const FILE_E: Bitboard = FILE_A << 4;
pub const FILE_F: Bitboard = FILE_A << 5;
pub const FILE_G: Bitboard = FILE_A << 6;
pub const FILE_H: Bitboard = FILE_A << 7;

pub const FILES: [Bitboard; 8] = [FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H];

// Zobrist Hashing Constants
pub struct Zobrist {
    pub pieces: [[[u64; 64]; PIECE_TYPE_COUNT]; COLOR_COUNT],
    pub side: u64,
    pub en_passant: [u64; 8],
}

pub struct AttackTables {
    pub knight: [Bitboard; 64],
    pub king: [Bitboard; 64],
    pub pawn_attacks: [[Bitboard; 64]; 2], // [0] White, [1] Black
    pub bishop_magics: [crate::magic::Magic; 64],
    pub rook_magics: [crate::magic::Magic; 64],
    pub bishop_table: Vec<Bitboard>,
    pub rook_table: Vec<Bitboard>,
}

pub fn get_zobrist_keys() -> Zobrist {
    let mut keys = Zobrist {
        pieces: [[[0; 64]; PIECE_TYPE_COUNT]; COLOR_COUNT],
        side: 0x123456789ABCDEF0,
        en_passant: [0; 8],
    };

    let mut seed = 0x9876543210FEDCBAu64;
    let mut next_rand = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };

    for c in 0..COLOR_COUNT {
        for p in 0..PIECE_TYPE_COUNT {
            for s in 0..64 {
                keys.pieces[c][p][s] = next_rand();
            }
        }
    }

    for f in 0..8 {
        keys.en_passant[f] = next_rand();
    }

    keys
}

pub fn get_attack_tables() -> AttackTables {
    let mut tables = AttackTables {
        knight: [0; 64],
        king: [0; 64],
        pawn_attacks: [[0; 64]; 2],
        bishop_magics: unsafe { std::mem::zeroed() },
        rook_magics: unsafe { std::mem::zeroed() },
        bishop_table: vec![0; 5248], // Bishop sum of (1 << bits)
        rook_table: vec![0; 102400], // Rook sum of (1 << bits)
    };

    for sq in 0..64 {
        let f = (sq % 8) as i8;
        let r = (sq / 8) as i8;

        // --- Knight/King/Pawn logic remains same ---
        for (df, dr) in [(-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1)].iter() {
            let nf = f + df; let nr = r + dr;
            if nf >= 0 && nf < 8 && nr >= 0 && nr < 8 { tables.knight[sq as usize] |= 1u64 << (nr * 8 + nf); }
        }
        for df in -1..=1 {
            for dr in -1..=1 {
                if df == 0 && dr == 0 { continue; }
                let nf = f + df; let nr = r + dr;
                if nf >= 0 && nf < 8 && nr >= 0 && nr < 8 { tables.king[sq as usize] |= 1u64 << (nr * 8 + nf); }
            }
        }
        if r < 7 {
            if f > 0 { tables.pawn_attacks[0][sq as usize] |= 1u64 << (sq + 7); }
            if f < 7 { tables.pawn_attacks[0][sq as usize] |= 1u64 << (sq + 9); }
        }
        if r > 0 {
            if f > 0 { tables.pawn_attacks[1][sq as usize] |= 1u64 << (sq - 9); }
            if f < 7 { tables.pawn_attacks[1][sq as usize] |= 1u64 << (sq - 7); }
        }
    }

    // --- Magic Bishop Init ---
    let mut bishop_offset = 0;
    for sq in 0..64 {
        let mask = crate::magic::get_bishop_mask(sq as u8);
        let bits = mask.count_ones();
        tables.bishop_magics[sq as usize] = crate::magic::Magic {
            mask,
            magic: crate::magic::BISHOP_MAGICS[sq as usize],
            shift: crate::magic::BISHOP_SHIFTS[sq as usize],
            offset: bishop_offset,
        };
        for i in 0..(1 << bits) {
            let occ = crate::magic::get_occupancy_from_index(i, mask);
            let index = (occ.wrapping_mul(tables.bishop_magics[sq as usize].magic)) >> tables.bishop_magics[sq as usize].shift;
            tables.bishop_table[bishop_offset + index as usize] = crate::magic::generate_bishop_attacks_on_the_fly(sq as u8, occ);
        }
        bishop_offset += 1 << bits;
    }

    // --- Magic Rook Init ---
    let mut rook_offset = 0;
    for sq in 0..64 {
        let mask = crate::magic::get_rook_mask(sq as u8);
        let bits = mask.count_ones();
        tables.rook_magics[sq as usize] = crate::magic::Magic {
            mask,
            magic: crate::magic::ROOK_MAGICS[sq as usize],
            shift: crate::magic::ROOK_SHIFTS[sq as usize],
            offset: rook_offset,
        };
        for i in 0..(1 << bits) {
            let occ = crate::magic::get_occupancy_from_index(i, mask);
            let index = (occ.wrapping_mul(tables.rook_magics[sq as usize].magic)) >> tables.rook_magics[sq as usize].shift;
            tables.rook_table[rook_offset + index as usize] = crate::magic::generate_rook_attacks_on_the_fly(sq as u8, occ);
        }
        rook_offset += 1 << bits;
    }

    tables
}
