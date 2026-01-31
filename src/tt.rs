// src/tt.rs

use chess::{ChessMove, Piece, Square};
use std::sync::atomic::{AtomicU64, Ordering};

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TTFlag {
    EXACT = 0,
    LOWERBOUND = 1,
    UPPERBOUND = 2,
}

impl From<u8> for TTFlag {
    fn from(value: u8) -> Self {
        match value {
            0 => TTFlag::EXACT,
            1 => TTFlag::LOWERBOUND,
            _ => TTFlag::UPPERBOUND,
        }
    }
}

impl From<TTFlag> for u8 {
    fn from(value: TTFlag) -> Self {
        value as u8
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TTEntry {
    pub key: u64,
    pub best_move: Option<ChessMove>,
    pub score: i32,
    pub depth: u8,
    pub flag: TTFlag,
}

struct PackedEntry {
    key: AtomicU64, // stores key ^ data for lock-less atomicity
    data: AtomicU64,
}

pub struct TranspositionTable {
    entries: Vec<PackedEntry>,
    size: usize,
    age: AtomicU64, // Using Atomic for multi-threaded access
}

impl TranspositionTable {
    /// Creates a new `TranspositionTable` with a given size in Megabytes.
    #[must_use]
    pub fn new(size_mb: usize) -> Self {
        let entry_size = std::mem::size_of::<PackedEntry>();
        let size = (size_mb * 1024 * 1024) / entry_size;
        let mut entries = Vec::with_capacity(size);
        for _ in 0..size {
            entries.push(PackedEntry {
                key: AtomicU64::new(0),
                data: AtomicU64::new(0),
            });
        }
        Self {
            entries,
            size,
            age: AtomicU64::new(0),
        }
    }

    pub fn increment_age(&self) {
        self.age.fetch_add(1, Ordering::SeqCst);
    }

    /// Probes the TT for a given Zobrist key.
    #[must_use]
    pub fn probe(&self, key: u64) -> Option<TTEntry> {
        #[allow(clippy::cast_possible_truncation)]
        let index = (key % self.size as u64) as usize;
        let slot = &self.entries[index];

        let data = slot.data.load(Ordering::Acquire);
        let xored_key = slot.key.load(Ordering::Acquire);

        if (xored_key ^ data) == key && data != 0 {
            return Some(Self::unpack(key, data));
        }
        None
    }

    /// Stores a new entry in the TT using a lock-free replacement strategy.
    pub fn store(&self, entry: TTEntry) {
        #[allow(clippy::cast_possible_truncation)]
        let index = (entry.key % self.size as u64) as usize;
        let slot = &self.entries[index];

        let current_age = self.age.load(Ordering::Relaxed);

        let existing_data = slot.data.load(Ordering::Relaxed);
        let existing_xored_key = slot.key.load(Ordering::Relaxed);
        let existing_key = existing_xored_key ^ existing_data;

        if existing_data != 0 {
            let existing_depth = ((existing_data >> 32) & 0xFF) as u8;
            let existing_age = existing_data >> 48;

            if entry.key == existing_key {
                // Same position: only replace if new entry has higher or equal depth
                if entry.depth < existing_depth {
                    return;
                }
            } else {
                // Different position: replace if old or higher depth
                if current_age == existing_age && entry.depth < existing_depth {
                    return; // Keep existing deeper entry from current search
                }
            }
        }

        #[allow(clippy::cast_possible_truncation)]
        let data = Self::pack(&entry, current_age as u16);
        slot.key.store(entry.key ^ data, Ordering::Release);
        slot.data.store(data, Ordering::Release);
    }

    #[allow(clippy::many_single_char_names)]
    fn pack(entry: &TTEntry, age: u16) -> u64 {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let s = u64::from((entry.score as i16) as u16); // bits 0..16
        let m = match entry.best_move {
            Some(mv) => u64::from(Self::move_to_u16(mv)),
            None => 0,
        }; // bits 16..32
        let d = u64::from(entry.depth); // bits 32..40
        let f = u64::from(u8::from(entry.flag)); // bits 40..48
        let a = u64::from(age); // bits 48..64

        s | (m << 16) | (d << 32) | (f << 40) | (a << 48)
    }

    fn unpack(key: u64, data: u64) -> TTEntry {
        let score = i32::from((data & 0xFFFF) as i16);
        let mv_bits = ((data >> 16) & 0xFFFF) as u16;
        let best_move = if mv_bits == 0 {
            None
        } else {
            Some(Self::u16_to_move(mv_bits))
        };
        let depth = ((data >> 32) & 0xFF) as u8;
        let flag = TTFlag::from(((data >> 40) & 0xFF) as u8);

        TTEntry {
            key,
            best_move,
            score,
            depth,
            flag,
        }
    }

    fn move_to_u16(mv: ChessMove) -> u16 {
        let src = u16::from(mv.get_source().to_int());
        let dst = u16::from(mv.get_dest().to_int());
        let promo = match mv.get_promotion() {
            Some(Piece::Knight) => 1,
            Some(Piece::Bishop) => 2,
            Some(Piece::Rook) => 3,
            Some(Piece::Queen) => 4,
            _ => 0,
        };
        (promo << 12) | (src << 6) | dst
    }

    fn u16_to_move(val: u16) -> ChessMove {
        let promo = (val >> 12) & 0x7;
        let src = (val >> 6) & 0x3F;
        let dst = val & 0x3F;
        let promo_piece = match promo {
            1 => Some(Piece::Knight),
            2 => Some(Piece::Bishop),
            3 => Some(Piece::Rook),
            4 => Some(Piece::Queen),
            _ => None,
        };
        ChessMove::new(
            unsafe { Square::new(src as u8) },
            unsafe { Square::new(dst as u8) },
            promo_piece,
        )
    }

    /// Clears the transposition table.
    pub fn clear(&self) {
        for entry in &self.entries {
            entry.key.store(0, Ordering::Relaxed);
            entry.data.store(0, Ordering::Relaxed);
        }
    }
}

unsafe impl Send for TranspositionTable {}
unsafe impl Sync for TranspositionTable {}
