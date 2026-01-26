use crate::mv::Move;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HashFlag {
    Exact,
    Alpha,
    Beta,
}

#[derive(Clone, Copy, Debug)]
pub struct TTEntry {
    pub hash: u64,
    pub depth: u8,
    pub score: i32,
    pub flag: HashFlag,
    pub best_move: Option<Move>,
}

pub struct TranspositionTable {
    pub entries: std::sync::Mutex<Vec<Option<TTEntry>>>,
    pub size: std::sync::atomic::AtomicUsize,
}

impl TranspositionTable {
    pub fn new(mb: usize) -> Self {
        let count = (mb * 1024 * 1024) >> 5; // ~32 bytes per entry
        TranspositionTable {
            entries: std::sync::Mutex::new(vec![None; count]),
            size: std::sync::atomic::AtomicUsize::new(count),
        }
    }

    pub fn resize(&self, mb: usize) {
        let count = (mb * 1024 * 1024) >> 5;
        let mut entries = self.entries.lock().unwrap();
        *entries = vec![None; count];
        self.size.store(count, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn store(&self, hash: u64, depth: u8, score: i32, flag: HashFlag, best_move: Option<Move>) {
        let size = self.size.load(std::sync::atomic::Ordering::Relaxed);
        if size == 0 { return; }
        let index = (hash as usize) % size;
        let mut entries = self.entries.lock().unwrap();
        // Guard against index out of bounds during concurrent resize
        if index >= entries.len() { return; }

        if let Some(existing) = entries[index] {
            if existing.depth > depth && existing.hash == hash {
                return;
            }
        }

        entries[index] = Some(TTEntry {
            hash, depth, score, flag, best_move,
        });
    }

    pub fn probe(&self, hash: u64, depth: u8, alpha: i32, beta: i32) -> Option<(i32, Option<Move>)> {
        let size = self.size.load(std::sync::atomic::Ordering::Relaxed);
        if size == 0 { return None; }
        let index = (hash as usize) % size;
        let entries = self.entries.lock().unwrap();
        if index >= entries.len() { return None; }

        if let Some(entry) = entries[index] {
            if entry.hash == hash {
                let stored_move = entry.best_move;
                if entry.depth >= depth {
                    let score = entry.score;
                    match entry.flag {
                        HashFlag::Exact => return Some((score, stored_move)),
                        HashFlag::Alpha if score <= alpha => return Some((alpha, stored_move)),
                        HashFlag::Beta if score >= beta => return Some((beta, stored_move)),
                        _ => {}
                    }
                }
                return Some((0, stored_move));
            }
        }
        None
    }
}

// Global TT (initialized in main)
// In a true engine, this would be inside a Search object.
