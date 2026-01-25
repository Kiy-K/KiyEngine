use crate::defs::{Bitboard, Color, PieceType, PIECE_TYPE_COUNT, COLOR_COUNT, RANK_1, RANK_2, RANK_7, RANK_8, EMPTY};
use crate::mv::Move;

/// Represents the state of the chess board.
#[derive(Debug, Clone)]
pub struct Board {
    /// Bitboards for each piece type (Pawn, Knight, Bishop, Rook, Queen, King).
    /// Indexed by [color][piece_type].
    pub pieces: [[Bitboard; PIECE_TYPE_COUNT]; COLOR_COUNT],

    /// Bitboards for occupancy by color.
    /// Indexed by [color].
    pub occupancy: [Bitboard; COLOR_COUNT],

    /// Combined occupancy of both colors.
    pub all_pieces: Bitboard,

    /// Side to move.
    pub side: Color,

    /// Stores the square *behind* the pawn that moved two squares.
    pub en_passant: Option<u8>,

    /// Zobrist hash of the current position.
    pub hash: u64,

    // Castling rights, halfmove clock would go here.
}

/// Global Zobrist keys (initialized once). 
/// In a production engine, this would be a static or passed around.
/// For simplicity, we'll recreate or pass it.
lazy_static::lazy_static! {
    pub static ref ZOBRIST: crate::defs::Zobrist = crate::defs::get_zobrist_keys();
    pub static ref ATTACKS: crate::defs::AttackTables = crate::defs::get_attack_tables();
}

impl Board {
    /// Creates a new `Board` setup with the standard chess starting position.
    pub fn new() -> Self {
        let mut board = Board {
            pieces: [[EMPTY; PIECE_TYPE_COUNT]; COLOR_COUNT],
            occupancy: [EMPTY; COLOR_COUNT],
            all_pieces: EMPTY,
            side: Color::White,
            en_passant: None,
            hash: 0,
        };

        board.init_start_position();
        board.hash = board.calculate_hash();
        board
    }

    fn init_start_position(&mut self) {
        // --- White Pieces ---
        // White Pawns start on Rank 2
        self.pieces[Color::White as usize][PieceType::Pawn as usize] = RANK_2;

        // White Rooks: A1 (bit 0), H1 (bit 7)
        self.pieces[Color::White as usize][PieceType::Rook as usize] = (1 << 0) | (1 << 7);
        // White Knights: B1 (bit 1), G1 (bit 6)
        self.pieces[Color::White as usize][PieceType::Knight as usize] = (1 << 1) | (1 << 6);
        // White Bishops: C1 (bit 2), F1 (bit 5)
        self.pieces[Color::White as usize][PieceType::Bishop as usize] = (1 << 2) | (1 << 5);
        // White Queen: D1 (bit 3)
        self.pieces[Color::White as usize][PieceType::Queen as usize] = 1 << 3;
        // White King: E1 (bit 4)
        self.pieces[Color::White as usize][PieceType::King as usize] = 1 << 4;

        // --- Black Pieces ---
        // Black Pawns start on Rank 7
        self.pieces[Color::Black as usize][PieceType::Pawn as usize] = RANK_7;

        // Black Rooks: A8 (bit 56), H8 (bit 63)
        self.pieces[Color::Black as usize][PieceType::Rook as usize] = (1 << 56) | (1 << 63);
        // Black Knights: B8 (bit 57), G8 (bit 62)
        self.pieces[Color::Black as usize][PieceType::Knight as usize] = (1 << 57) | (1 << 62);
        // Black Bishops: C8 (bit 58), F8 (bit 61)
        self.pieces[Color::Black as usize][PieceType::Bishop as usize] = (1 << 58) | (1 << 61);
        // Black Queen: D8 (bit 59)
        self.pieces[Color::Black as usize][PieceType::Queen as usize] = 1 << 59;
        // Black King: E8 (bit 60)
        self.pieces[Color::Black as usize][PieceType::King as usize] = 1 << 60;

        self.update_occupancies();
    }

    /// Updates the occupancy bitboards based on the piece bitboards.
    pub fn update_occupancies(&mut self) {
        // Reset occupancies
        self.occupancy = [EMPTY; COLOR_COUNT];
        self.all_pieces = EMPTY;

        // Iterate over all piece types for White
        for piece in 0..PIECE_TYPE_COUNT {
            self.occupancy[Color::White as usize] |= self.pieces[Color::White as usize][piece];
        }

        // Iterate over all piece types for Black
        for piece in 0..PIECE_TYPE_COUNT {
            self.occupancy[Color::Black as usize] |= self.pieces[Color::Black as usize][piece];
        }

        // Combine for all pieces
        self.all_pieces = self.occupancy[Color::White as usize] | self.occupancy[Color::Black as usize];
    }

    /// Calculates the full Zobrist hash of the current position.
    pub fn calculate_hash(&self) -> u64 {
        let mut h = 0u64;
        
        for c in 0..COLOR_COUNT {
            for p in 0..PIECE_TYPE_COUNT {
                let mut bb = self.pieces[c][p];
                while bb != 0 {
                    let sq = bb.trailing_zeros() as usize;
                    bb &= bb - 1;
                    h ^= ZOBRIST.pieces[c][p][sq];
                }
            }
        }

        if self.side == Color::Black {
            h ^= ZOBRIST.side;
        }

        if let Some(ep_sq) = self.en_passant {
            let file = ep_sq % 8;
            h ^= ZOBRIST.en_passant[file as usize];
        }

        h
    }

    /// Makes a move on the board.
    /// Updates bitboards, occupancy, side to move, and hash.
    pub fn make_move(&mut self, m: Move) -> bool {
        let source = m.source();
        let target = m.target();
        let side = self.side;
        let opp = side.opposite();

        // 1. Identify the moving piece type
        let mut piece_type = None;
        let mut actual_side = side;

        for pt in 0..PIECE_TYPE_COUNT {
            if (self.pieces[side as usize][pt] >> source) & 1 == 1 {
                piece_type = Some(pt);
                break;
            }
        }
        
        // Robustness: if not found on current turn, check other side
        if piece_type.is_none() {
            for pt in 0..PIECE_TYPE_COUNT {
                if (self.pieces[opp as usize][pt] >> source) & 1 == 1 {
                    piece_type = Some(pt);
                    actual_side = opp;
                    break;
                }
            }
        }

        if piece_type.is_none() {
            return false;
        }
        let pt = piece_type.unwrap();
        let side = actual_side;
        let opp = side.opposite();

        // XOR out side to move
        self.hash ^= ZOBRIST.side;
        
        // XOR out old EP
        if let Some(ep_sq) = self.en_passant {
            self.hash ^= ZOBRIST.en_passant[(ep_sq % 8) as usize];
        }

        // 2. Remove piece from source
        let from_bb = 1u64 << source;
        self.pieces[side as usize][pt] &= !from_bb;
        self.hash ^= ZOBRIST.pieces[side as usize][pt][source as usize];

        // En Passant Capture Handling
        if pt == PieceType::Pawn as usize {
             if let Some(ep_sq) = self.en_passant {
                 if target == ep_sq {
                     let captured_sq = if side == Color::White { target - 8 } else { target + 8 };
                     self.pieces[opp as usize][PieceType::Pawn as usize] &= !(1u64 << captured_sq);
                     self.hash ^= ZOBRIST.pieces[opp as usize][PieceType::Pawn as usize][captured_sq as usize];
                 }
             }
        }

        // 3. Handle Regular Capture
        if (self.occupancy[opp as usize] >> target) & 1 == 1 {
            for opp_pt in 0..PIECE_TYPE_COUNT {
                if (self.pieces[opp as usize][opp_pt] >> target) & 1 == 1 {
                     self.pieces[opp as usize][opp_pt] &= !(1u64 << target);
                     self.hash ^= ZOBRIST.pieces[opp as usize][opp_pt][target as usize];
                     break; 
                }
            }
        }

        // 4. Place piece at target
        let to_bb = 1u64 << target;
        let final_pt = match m.prom() {
            Some(p) => p as usize,
            None => pt,
        };
        self.pieces[side as usize][final_pt] |= to_bb;
        self.hash ^= ZOBRIST.pieces[side as usize][final_pt][target as usize];

        // 5. Update En Passant State
        self.en_passant = None;
        if pt == PieceType::Pawn as usize {
            let diff = (target as i16 - source as i16).abs();
            if diff == 16 {
                let ep_sq = if target > source { source + 8 } else { source - 8 };
                self.en_passant = Some(ep_sq);
                // XOR in new EP
                self.hash ^= ZOBRIST.en_passant[(ep_sq % 8) as usize];
            }
        }

        // 6. Update Occupancies
        self.update_occupancies();

        // 7. Switch Side
        self.side = opp;

        true
    }
    pub fn is_square_attacked(&self, sq: u8, attacker_color: Color) -> bool {
        let side = attacker_color as usize;
        let opp_side = attacker_color.opposite() as usize;
        let attacks_ref = &crate::board::ATTACKS;

        // 1. Pawns
        let attacker_pawn_bb = self.pieces[side][PieceType::Pawn as usize];
        let p_attacks = attacks_ref.pawn_attacks[opp_side][sq as usize];
        if (p_attacks & attacker_pawn_bb) != 0 { return true; }

        // 2. Knights
        let attacker_knight_bb = self.pieces[side][PieceType::Knight as usize];
        let k_attacks = attacks_ref.knight[sq as usize];
        if (k_attacks & attacker_knight_bb) != 0 { return true; }

        // 3. Kings
        let attacker_king_bb = self.pieces[side][PieceType::King as usize];
        let kg_attacks = attacks_ref.king[sq as usize];
        if (kg_attacks & attacker_king_bb) != 0 { return true; }

        // 4. Sliders (Magic Bitboards)
        let occ = self.all_pieces;
        
        // Bishop / Queen
        let bm = &attacks_ref.bishop_magics[sq as usize];
        let bi = ((occ & bm.mask).wrapping_mul(bm.magic)) >> bm.shift;
        let b_atk = attacks_ref.bishop_table[bm.offset + bi as usize];
        let attackers_bq = self.pieces[side][PieceType::Bishop as usize] | self.pieces[side][PieceType::Queen as usize];
        if (b_atk & attackers_bq) != 0 { return true; }

        // Rook / Queen
        let rm = &attacks_ref.rook_magics[sq as usize];
        let ri = ((occ & rm.mask).wrapping_mul(rm.magic)) >> rm.shift;
        let r_atk = attacks_ref.rook_table[rm.offset + ri as usize];
        let attackers_rq = self.pieces[side][PieceType::Rook as usize] | self.pieces[side][PieceType::Queen as usize];
        if (r_atk & attackers_rq) != 0 { return true; }

        false
    }

    /// Prints the board to the console for debugging/verification.
    pub fn print(&self) {
        println!("  +---+---+---+---+---+---+---+---+");
        for rank in (0..8).rev() {
            print!("{} |", rank + 1);
            for file in 0..8 {
                let square = rank * 8 + file;
                let mut piece_char = ' ';
                
                // Check all White pieces
                for (i, bb) in self.pieces[Color::White as usize].iter().enumerate() {
                    if (bb >> square) & 1 == 1 {
                        piece_char = match i {
                            0 => 'P', 1 => 'N', 2 => 'B', 3 => 'R', 4 => 'Q', 5 => 'K', _ => '?'
                        };
                        break;
                    }
                }
                
                // Check all Black pieces
                if piece_char == ' ' {
                    for (i, bb) in self.pieces[Color::Black as usize].iter().enumerate() {
                        if (bb >> square) & 1 == 1 {
                            piece_char = match i {
                                0 => 'p', 1 => 'n', 2 => 'b', 3 => 'r', 4 => 'q', 5 => 'k', _ => '?'
                            };
                            break;
                        }
                    }
                }

                print!(" {} |", piece_char);
            }
            println!("\n  +---+---+---+---+---+---+---+---+");
        }
        println!("    a   b   c   d   e   f   g   h");
        println!();
        println!("Side to move: {:?}", self.side);
        println!("Hash: {:016X}", self.hash);
    }
}
