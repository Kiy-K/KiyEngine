//! Huge Transposition Table for caching search results.
//!
//! A large transposition table is critical for high-performance chess engines.
//! It stores the results of previous searches, allowing the engine to avoid
//! re-calculating positions it has already seen. This is especially important
//! when using a computationally expensive neural network for evaluation, as it
//! dramatically reduces the number of calls to the network.

use chess::ChessMove;
use std::mem;

/// The flag for a transposition table entry, indicating the type of score.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum TTFlag {
    Exact, // The score is an exact value for the position.
    Alpha, // The score is a lower bound (alpha).
    Beta,  // The score is an upper bound (beta).
}

/// A single entry in the transposition table.
///
/// We align this struct to 16 bytes for efficient memory access.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct TTEntry {
    pub key: u64,  // 8 bytes: The Zobrist key of the position.
    pub best_move: Option<ChessMove>, // 2 bytes + padding: The best move found.
    pub score: i16,       // 2 bytes: The score of the position.
    pub depth: i8,        // 1 byte: The depth of the search that found this score.
    pub flag: TTFlag,     // 1 byte: The type of score (Exact, Alpha, or Beta).
}

impl Default for TTEntry {
    fn default() -> Self {
        TTEntry {
            key: 0,
            best_move: None,
            score: 0,
            depth: 0,
            flag: TTFlag::Exact,
        }
    }
}

/// The transposition table itself.
pub struct TranspositionTable {
    entries: Vec<TTEntry>,
    size_mb: usize,
}

impl TranspositionTable {
    /// The default size of the transposition table in megabytes.
    pub const DEFAULT_SIZE_MB: usize = 512;

    /// Creates a new transposition table with the specified size in megabytes.
    pub fn new(size_mb: usize) -> Self {
        let entry_size = mem::size_of::<TTEntry>();
        let num_entries = (size_mb * 1024 * 1024) / entry_size;
        Self {
            entries: vec![TTEntry::default(); num_entries],
            size_mb,
        }
    }

    /// Resizes the transposition table to the new size in megabytes.
    pub fn resize(&mut self, new_size_mb: usize) {
        let entry_size = mem::size_of::<TTEntry>();
        let num_entries = (new_size_mb * 1024 * 1024) / entry_size;
        self.entries.resize(num_entries, TTEntry::default());
        self.size_mb = new_size_mb;
    }

    /// Probes the transposition table for a given key.
    pub fn probe(&self, key: u64) -> Option<&TTEntry> {
        let index = (key as usize) % self.entries.len();
        let entry = &self.entries[index];
        if entry.key == key {
            Some(entry)
        } else {
            None
        }
    }

    /// Stores an entry in the transposition table.
    pub fn store(&mut self, entry: TTEntry) {
        let index = (entry.key as usize) % self.entries.len();
        self.entries[index] = entry;
    }

    /// Returns the size of the table in megabytes.
    pub fn size_mb(&self) -> usize {
        self.size_mb
    }
}

impl Default for TranspositionTable {
    /// Creates a new transposition table with the default size.
    fn default() -> Self {
        Self::new(Self::DEFAULT_SIZE_MB)
    }
}
