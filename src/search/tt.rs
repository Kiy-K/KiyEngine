use chess::{ChessMove, Piece, Square};
use std::sync::atomic::{AtomicU64, Ordering};

// Pack layout (64 bits):
// - Hash (16 bits)
// - Score (16 bits, signed)
// - BestMove (16 bits)
// - Depth (8 bits)
// - Flag (2 bits)
// - Age (6 bits)

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum TTFlag {
    None = 0,
    Exact = 1,
    LowerBound = 2,
    UpperBound = 3,
}

pub struct AtomicTT {
    table: Vec<AtomicU64>,
    mask: usize,
}

impl AtomicTT {
    pub fn new(size_mb: usize) -> Self {
        let size_bytes = size_mb * 1024 * 1024;
        let num_entries = size_bytes / 8;
        let num_entries = num_entries.next_power_of_two();
        let mut table = Vec::with_capacity(num_entries);
        for _ in 0..num_entries {
            table.push(AtomicU64::new(0));
        }
        Self {
            table,
            mask: num_entries - 1,
        }
    }

    pub fn resize(&mut self, size_mb: usize) {
        let size_bytes = size_mb * 1024 * 1024;
        let num_entries = size_bytes / 8;
        let num_entries = num_entries.next_power_of_two();
        self.table = Vec::with_capacity(num_entries);
        for _ in 0..num_entries {
            self.table.push(AtomicU64::new(0));
        }
        self.mask = num_entries - 1;
    }

    pub fn clear(&self) {
        for entry in &self.table {
            entry.store(0, Ordering::Relaxed);
        }
    }

    fn pack(
        hash: u64,
        score: i32,
        best_move: Option<ChessMove>,
        depth: u8,
        flag: TTFlag,
        age: u8,
    ) -> u64 {
        let hash_part = (hash as u64) & 0xFFFF;
        let score_part = (score as u16) as u64; // Cast i16 bit pattern to u16
        let move_part = if let Some(m) = best_move {
            Self::encode_move(m) as u64
        } else {
            0
        };
        let depth_part = (depth as u64) & 0xFF;
        let flag_part = (flag as u64) & 0x3;
        let age_part = (age as u64) & 0x3F;

        // Layout:
        // [Age:6][Flag:2][Depth:8][Move:16][Score:16][Hash:16]
        (age_part << 58)
            | (flag_part << 56)
            | (depth_part << 48)
            | (move_part << 32)
            | (score_part << 16)
            | hash_part
    }

    fn unpack(data: u64) -> (u16, i32, Option<ChessMove>, u8, TTFlag, u8) {
        let hash_part = (data & 0xFFFF) as u16;
        let score_part = ((data >> 16) & 0xFFFF) as u16 as i16 as i32;
        let move_part = ((data >> 32) & 0xFFFF) as u16;
        let depth_part = ((data >> 48) & 0xFF) as u8;
        let flag_raw = ((data >> 56) & 0x3) as u8;
        let age_part = ((data >> 58) & 0x3F) as u8;

        let flag = match flag_raw {
            1 => TTFlag::Exact,
            2 => TTFlag::LowerBound,
            3 => TTFlag::UpperBound,
            _ => TTFlag::None,
        };

        let best_move = if move_part != 0 {
            Some(Self::decode_move(move_part))
        } else {
            None
        };

        (hash_part, score_part, best_move, depth_part, flag, age_part)
    }

    pub fn store(
        &self,
        hash: u64,
        score: i32,
        best_move: Option<ChessMove>,
        depth: u8,
        flag: TTFlag,
        age: u8,
    ) {
        let index = (hash as usize) & self.mask;
        let new_data = Self::pack(hash, score, best_move, depth, flag, age);

        // Lock-free "always replace if deeper or different generation" could be better,
        // but simple relaxed store is "Shared Memory Parallelism" friendly enough.
        // We generally don't want to overwrite a deep entry with a shallow one unless age differs?
        // But for Lazy SMP, simpler is often faster.
        self.table[index].store(new_data, Ordering::Relaxed);
    }

    pub fn probe(&self, hash: u64) -> Option<(i32, Option<ChessMove>, u8, TTFlag)> {
        let index = (hash as usize) & self.mask;
        let data = self.table[index].load(Ordering::Relaxed);
        if data == 0 {
            return None;
        }

        let (stored_hash, score, best_move, depth, flag, _) = Self::unpack(data);

        if stored_hash == ((hash & 0xFFFF) as u16) {
            Some((score, best_move, depth, flag))
        } else {
            None // Collision
        }
    }

    fn encode_move(m: ChessMove) -> u16 {
        let src = m.get_source().to_index() as u16;
        let dst = m.get_dest().to_index() as u16;
        // Simple encoding: src (6 bits) | dst (6 bits)
        // Missing promotion for brevity, but crucial for correctness.
        // Let's add promotion: 2 bits (Kn, Bi, Ro, Qu).
        // Total 14 bits.
        let promo = match m.get_promotion() {
            Some(Piece::Knight) => 1,
            Some(Piece::Bishop) => 2,
            Some(Piece::Rook) => 3,
            Some(Piece::Queen) => 4,
            _ => 0,
        };

        (promo << 12) | (dst << 6) | src
    }

    fn decode_move(val: u16) -> ChessMove {
        let src = unsafe {
            // SAFETY: val & 0x3F is always 0-63, valid for Square
            Square::new((val & 0x3F) as u8)
        };
        let dst = unsafe {
            // SAFETY: (val >> 6) & 0x3F is always 0-63, valid for Square
            Square::new(((val >> 6) & 0x3F) as u8)
        };
        let promo_raw = (val >> 12) & 0x7;
        let promo = match promo_raw {
            1 => Some(Piece::Knight),
            2 => Some(Piece::Bishop),
            3 => Some(Piece::Rook),
            4 => Some(Piece::Queen),
            _ => None,
        };
        ChessMove::new(src, dst, promo)
    }
}
