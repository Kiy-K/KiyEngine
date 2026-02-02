// src/book/mod.rs

use chess::{Board, ChessMove, Piece, Square};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub struct OpeningBook {
    file: Option<File>,
}

impl OpeningBook {
    pub fn open<P: AsRef<Path>>(path: P) -> Self {
        let file = File::open(path).ok();
        Self { file }
    }

    pub fn get_move(&mut self, board: &Board) -> Option<ChessMove> {
        let key = self.polyglot_key(board);
        let file = self.file.as_mut()?;

        let mut low = 0;
        let mut high = (file.metadata().ok()?.len() / 16) as i64 - 1;

        let mut best_move_raw = None;

        while low <= high {
            let mid = (low + high) / 2;
            file.seek(SeekFrom::Start(mid as u64 * 16)).ok()?;

            let mut buf = [0u8; 16];
            file.read_exact(&mut buf).ok()?;

            let entry_key = u64::from_be_bytes(buf[0..8].try_into().unwrap());
            if entry_key < key {
                low = mid + 1;
            } else if entry_key > key {
                high = mid - 1;
            } else {
                let move_raw = u16::from_be_bytes(buf[8..10].try_into().unwrap());

                best_move_raw = Some(move_raw);
                break;
            }
        }

        best_move_raw.and_then(|m| self.parse_polyglot_move(m, board))
    }

    fn parse_polyglot_move(&self, m: u16, board: &Board) -> Option<ChessMove> {
        let to_idx = (m & 0x3F) as u8;
        let from_idx = ((m >> 6) & 0x3F) as u8;
        let promo_idx = ((m >> 12) & 0x7) as u8;

        let from = unsafe { Square::new(from_idx) };
        let to = unsafe { Square::new(to_idx) };

        let promo = match promo_idx {
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

    fn polyglot_key(&self, board: &Board) -> u64 {
        board.get_hash()
    }
}
