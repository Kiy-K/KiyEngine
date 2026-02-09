// src/book/mod.rs
//
// Polyglot opening book support.
// Loads one or more .bin files (Titan, Perfect2023, etc.) and probes them
// using the standard Polyglot Zobrist key algorithm.

use chess::{BitBoard, Board, CastleRights, ChessMove, Color, Piece, Square};

mod polyglot_keys;
use polyglot_keys::POLYGLOT_KEYS;

/// Compute the standard Polyglot Zobrist hash for a chess position.
///
/// Piece encoding: BlackPawn=0 WhitePawn=1 BlackKnight=2 WhiteKnight=3
///   BlackBishop=4 WhiteBishop=5 BlackRook=6 WhiteRook=7
///   BlackQueen=8 WhiteQueen=9 BlackKing=10 WhiteKing=11
/// Index into keys: kind * 64 + square
/// Castling: keys[768..772]  EnPassant: keys[772..780]  Turn: keys[780]
pub fn polyglot_hash(board: &Board) -> u64 {
    let keys = &POLYGLOT_KEYS;
    let mut hash: u64 = 0;

    // --- Pieces on squares ---
    for sq_idx in 0..64u8 {
        let sq = unsafe { Square::new(sq_idx) };
        if let Some(piece) = board.piece_on(sq) {
            if let Some(color) = board.color_on(sq) {
                let kind = match (piece, color) {
                    (Piece::Pawn, Color::Black) => 0,
                    (Piece::Pawn, Color::White) => 1,
                    (Piece::Knight, Color::Black) => 2,
                    (Piece::Knight, Color::White) => 3,
                    (Piece::Bishop, Color::Black) => 4,
                    (Piece::Bishop, Color::White) => 5,
                    (Piece::Rook, Color::Black) => 6,
                    (Piece::Rook, Color::White) => 7,
                    (Piece::Queen, Color::Black) => 8,
                    (Piece::Queen, Color::White) => 9,
                    (Piece::King, Color::Black) => 10,
                    (Piece::King, Color::White) => 11,
                };
                hash ^= keys[kind * 64 + sq_idx as usize];
            }
        }
    }

    // --- Castling rights ---
    let wr = board.castle_rights(Color::White);
    if wr == CastleRights::KingSide || wr == CastleRights::Both {
        hash ^= keys[768];
    }
    if wr == CastleRights::QueenSide || wr == CastleRights::Both {
        hash ^= keys[769];
    }
    let br = board.castle_rights(Color::Black);
    if br == CastleRights::KingSide || br == CastleRights::Both {
        hash ^= keys[770];
    }
    if br == CastleRights::QueenSide || br == CastleRights::Both {
        hash ^= keys[771];
    }

    // --- En passant (only if a pawn can actually capture) ---
    if let Some(ep_sq) = board.en_passant() {
        let file = ep_sq.get_file().to_index();
        let us = board.side_to_move();
        let our_pawns = *board.pieces(Piece::Pawn) & *board.color_combined(us);
        let ep_bb = BitBoard::from_square(ep_sq);
        // Check if any of our pawns can reach the EP square
        let can_capture = if us == Color::White {
            // White pawns attack from rank below: check sq-7 (left) and sq-9 (right)
            let left = BitBoard::new(ep_bb.0 >> 9) & BitBoard::new(!0x8080808080808080u64);
            let right = BitBoard::new(ep_bb.0 >> 7) & BitBoard::new(!0x0101010101010101u64);
            (left | right) & our_pawns
        } else {
            let left = BitBoard::new(ep_bb.0 << 7) & BitBoard::new(!0x8080808080808080u64);
            let right = BitBoard::new(ep_bb.0 << 9) & BitBoard::new(!0x0101010101010101u64);
            (left | right) & our_pawns
        };
        if can_capture.popcnt() > 0 {
            hash ^= keys[772 + file];
        }
    }

    // --- Turn ---
    if board.side_to_move() == Color::White {
        hash ^= keys[780];
    }

    hash
}

// =============================================================================
// OpeningBook — supports multiple .bin files
// =============================================================================

/// Maximum number of half-moves (ply) from startpos to use the book.
const MAX_BOOK_PLY: usize = 30;

/// Minimum weight threshold — skip dubious book moves below this.
const MIN_WEIGHT: u32 = 2;

/// Simple thread-local RNG for weighted book move selection.
/// Uses xorshift64 seeded from system time — no external crate needed.
fn simple_rand(max: u64) -> u64 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = Cell::new({
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            if t == 0 { 0xDEADBEEFCAFE } else { t }
        });
    }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        x % max
    })
}

pub struct OpeningBook {
    books: Vec<Vec<u8>>,
}

impl OpeningBook {
    /// Load multiple Polyglot .bin book files.
    /// Paths are resolved relative to the executable's directory so the engine
    /// works correctly regardless of the current working directory.
    pub fn load(paths: &[&str]) -> Self {
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()));

        let mut books = Vec::new();
        for path in paths {
            let candidates = {
                let mut c = vec![std::path::PathBuf::from(path)];
                if let Some(ref dir) = exe_dir {
                    c.push(dir.join(path));
                }
                c
            };
            for candidate in &candidates {
                if let Ok(data) = std::fs::read(candidate) {
                    if data.len() >= 16 {
                        eprintln!(
                            "Book loaded: {} ({} entries)",
                            candidate.display(),
                            data.len() / 16
                        );
                        books.push(data);
                        break;
                    }
                }
            }
        }
        if books.is_empty() {
            eprintln!("Warning: No opening books loaded");
        }
        Self { books }
    }

    /// Probe all loaded books for the given position.
    /// `ply` is the number of half-moves from the start position — used to
    /// stop book usage after MAX_BOOK_PLY.
    /// Uses weighted random selection for opening variety.
    pub fn probe(&self, board: &Board, ply: usize) -> Option<ChessMove> {
        if ply >= MAX_BOOK_PLY {
            return None;
        }

        let key = polyglot_hash(board);

        // Collect all matching entries from all books
        let mut candidates: Vec<(u16, u32)> = Vec::new();

        for book_data in &self.books {
            self.find_entries(book_data, key, &mut candidates);
        }

        // Filter out low-weight moves
        candidates.retain(|&(_, w)| w >= MIN_WEIGHT);

        if candidates.is_empty() {
            return None;
        }

        // Weighted random selection for variety (best practice from PolyGlot)
        let total_weight: u64 = candidates.iter().map(|&(_, w)| w as u64).sum();
        let mut pick = simple_rand(total_weight);
        for (move_raw, weight) in &candidates {
            if pick < *weight as u64 {
                if let Some(mv) = Self::parse_polyglot_move(*move_raw, board) {
                    return Some(mv);
                }
                // If this particular encoding didn't parse, fall through
            }
            pick = pick.saturating_sub(*weight as u64);
        }

        // Fallback: try all in weight-descending order
        candidates.sort_by(|a, b| b.1.cmp(&a.1));
        for (move_raw, _weight) in &candidates {
            if let Some(mv) = Self::parse_polyglot_move(*move_raw, board) {
                return Some(mv);
            }
        }

        None
    }

    /// Binary search + linear scan to find all entries matching `key` in a book.
    fn find_entries(&self, data: &[u8], key: u64, candidates: &mut Vec<(u16, u32)>) {
        let num_entries = data.len() / 16;
        if num_entries == 0 {
            return;
        }

        // Binary search for any entry with this key
        let mut low: i64 = 0;
        let mut high: i64 = num_entries as i64 - 1;
        let mut found: i64 = -1;

        while low <= high {
            let mid = (low + high) / 2;
            let offset = mid as usize * 16;
            let entry_key = u64::from_be_bytes(data[offset..offset + 8].try_into().unwrap());
            if entry_key < key {
                low = mid + 1;
            } else if entry_key > key {
                high = mid - 1;
            } else {
                found = mid;
                break;
            }
        }

        if found < 0 {
            return;
        }

        // Scan backwards to find the first entry with this key
        let mut start = found as usize;
        while start > 0 {
            let offset = (start - 1) * 16;
            let k = u64::from_be_bytes(data[offset..offset + 8].try_into().unwrap());
            if k != key {
                break;
            }
            start -= 1;
        }

        // Scan forward to collect all entries with this key
        let mut pos = start;
        while pos < num_entries {
            let offset = pos * 16;
            let k = u64::from_be_bytes(data[offset..offset + 8].try_into().unwrap());
            if k != key {
                break;
            }
            let move_raw = u16::from_be_bytes(data[offset + 8..offset + 10].try_into().unwrap());
            let weight = u16::from_be_bytes(data[offset + 10..offset + 12].try_into().unwrap());

            // Accumulate weight for duplicate moves across books
            if let Some(existing) = candidates.iter_mut().find(|(m, _)| *m == move_raw) {
                existing.1 += weight as u32;
            } else {
                candidates.push((move_raw, weight as u32));
            }

            pos += 1;
        }
    }

    /// Parse a Polyglot move encoding into a ChessMove.
    /// Polyglot encodes castling as king→rook (e.g., e1h1 for O-O),
    /// which must be converted to king→destination (e1g1).
    fn parse_polyglot_move(m: u16, board: &Board) -> Option<ChessMove> {
        let to_file = (m & 0x7) as u8;
        let to_rank = ((m >> 3) & 0x7) as u8;
        let from_file = ((m >> 6) & 0x7) as u8;
        let from_rank = ((m >> 9) & 0x7) as u8;
        let promo_raw = ((m >> 12) & 0x7) as u8;

        let from_idx = from_rank * 8 + from_file;
        let mut to_idx = to_rank * 8 + to_file;

        let from = unsafe { Square::new(from_idx) };

        // Handle Polyglot castling encoding: king moves to rook square
        // Convert to standard king destination
        if board.piece_on(from) == Some(Piece::King) {
            // White king on e1
            if from_idx == 4 {
                if to_idx == 7 {
                    to_idx = 6;
                } // e1h1 → e1g1 (O-O)
                if to_idx == 0 {
                    to_idx = 2;
                } // e1a1 → e1c1 (O-O-O)
            }
            // Black king on e8
            if from_idx == 60 {
                if to_idx == 63 {
                    to_idx = 62;
                } // e8h8 → e8g8 (O-O)
                if to_idx == 56 {
                    to_idx = 58;
                } // e8a8 → e8c8 (O-O-O)
            }
        }

        let to = unsafe { Square::new(to_idx) };

        let promo = match promo_raw {
            1 => Some(Piece::Knight),
            2 => Some(Piece::Bishop),
            3 => Some(Piece::Rook),
            4 => Some(Piece::Queen),
            _ => None,
        };

        let mv = ChessMove::new(from, to, promo);
        if board.legal(mv) {
            Some(mv)
        } else {
            None
        }
    }
}
