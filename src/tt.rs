// src/tt.rs

use chess::ChessMove;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TTFlag {
    EXACT,
    LOWERBOUND,
    UPPERBOUND,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TTEntry {
    pub key: u64, // The full Zobrist key to prevent hash collisions
    pub best_move: Option<ChessMove>,
    pub score: i32,
    pub depth: u8,
    pub flag: TTFlag,
}

/// A fixed-size transposition table using a vector and a simple replacement strategy.
/// This is a significant improvement over a HashMap for performance-critical lookups.
pub struct TranspositionTable {
    entries: Vec<Option<TTEntry>>,
    size: usize,
}

impl TranspositionTable {
    /// Creates a new TranspositionTable with a given size in Megabytes.
    pub fn new(size_mb: usize) -> Self {
        let entry_size = std::mem::size_of::<Option<TTEntry>>();
        let size = (size_mb * 1024 * 1024) / entry_size;
        Self {
            entries: vec![None; size],
            size,
        }
    }

    /// Probes the TT for a given Zobrist key.
    /// It returns the entry only if the full key matches, to avoid collisions.
    pub fn probe(&self, key: u64) -> Option<&TTEntry> {
        let index = (key % self.size as u64) as usize;
        if let Some(entry) = &self.entries[index] {
            if entry.key == key {
                return Some(entry);
            }
        }
        None
    }

    /// Stores a new entry in the TT.
    /// It uses a depth-preferred replacement strategy:
    /// - If the new entry has a higher depth, replace.
    /// - If the depths are equal, replace if the existing entry is not EXACT.
    pub fn store(&mut self, entry: TTEntry) {
        let index = (entry.key % self.size as u64) as usize;
        if let Some(existing) = &self.entries[index] {
            if entry.depth > existing.depth {
                self.entries[index] = Some(entry);
            } else if entry.depth == existing.depth {
                // Same depth: replace if existing is not EXACT or if new is EXACT
                if existing.flag != TTFlag::EXACT || entry.flag == TTFlag::EXACT {
                    self.entries[index] = Some(entry);
                }
            }
        } else {
            self.entries[index] = Some(entry);
        }
    }

    /// Clears the transposition table by resetting all entries to None.
    pub fn clear(&mut self) {
        self.entries = vec![None; self.size];
    }
}
