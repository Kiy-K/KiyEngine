use crate::defs::{PieceType, A1, H8};

/// Represents a chess move.
/// Encoded as a 16-bit integer (u16).
///
/// Bit layout:
/// 0-5: Source Square (0-63)
/// 6-11: Target Square (0-63)
/// 12-13: Promotion Piece Type (0=None, 1=Knight, 2=Bishop, 3=Rook) - mapped slightly differently than PieceType
/// 14: Capture Flag
/// 15: Special Flag (Castling / En Passant / Promotion)
///
/// Note: This is a simplified encoding for this stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Move {
    raw: u16,
    pub score: i32,
}

impl Move {
    const SOURCE_MASK: u16 = 0x3F;
    const TARGET_MASK: u16 = 0x3F << 6;
    const PROM_SHIFT: u32 = 12;

    pub fn new(source: u8, target: u8, prom: Option<PieceType>) -> Self {
        let s = (source as u16) & 0x3F;
        let t = (target as u16) & 0x3F;
        let p = match prom {
            Some(PieceType::Knight) => 1,
            Some(PieceType::Bishop) => 2,
            Some(PieceType::Rook) => 3,
            Some(PieceType::Queen) => 4,
            _ => 0,
        };
        Move { 
            raw: s | (t << 6) | (p << Self::PROM_SHIFT),
            score: 0 
        }
    }

    pub fn source(&self) -> u8 { (self.raw & Self::SOURCE_MASK) as u8 }
    pub fn target(&self) -> u8 { ((self.raw & Self::TARGET_MASK) >> 6) as u8 }
    pub fn prom(&self) -> Option<PieceType> {
        match (self.raw >> Self::PROM_SHIFT) & 0x7 {
            1 => Some(PieceType::Knight),
            2 => Some(PieceType::Bishop),
            3 => Some(PieceType::Rook),
            4 => Some(PieceType::Queen),
            _ => None,
        }
    }

    pub fn to_string(&self) -> String {
        let s_file = (self.source() % 8) + b'a';
        let s_rank = (self.source() / 8) + b'1';
        let t_file = (self.target() % 8) + b'a';
        let t_rank = (self.target() / 8) + b'1';
        let mut s = format!("{}{}{}{}", s_file as char, s_rank as char, t_file as char, t_rank as char);
        if let Some(p) = self.prom() {
            let char = match p {
                PieceType::Knight => 'n',
                PieceType::Bishop => 'b',
                PieceType::Rook => 'r',
                _ => 'q',
            };
            s.push(char);
        }
        s
    }
}

/// A container for generated moves.
pub struct MoveList {
    pub moves: [Move; 256],
    pub count: usize,
}

impl MoveList {
    pub fn new() -> Self {
        MoveList {
            moves: [Move::new(0, 0, None); 256],
            count: 0,
        }
    }

    pub fn add(&mut self, m: Move) {
        if self.count < 256 {
            self.moves[self.count] = m;
            self.count += 1;
        }
    }

    /// Sorts moves by their score in descending order.
    pub fn sort(&mut self) {
        for i in 0..self.count {
            for j in i + 1..self.count {
                if self.moves[j].score > self.moves[i].score {
                    self.moves.swap(i, j);
                }
            }
        }
    }
}
