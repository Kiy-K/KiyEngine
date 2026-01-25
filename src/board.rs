use crate::defs::{Bitboard, Color, PieceType, PIECE_TYPE_COUNT, COLOR_COUNT, RANK_2, RANK_7, EMPTY};
use crate::mv::Move;
use crate::nnue_v2;

/// Represents the state of the chess board.
#[derive(Clone)]
pub struct Board {
    pub pieces: [[Bitboard; PIECE_TYPE_COUNT]; COLOR_COUNT],
    pub occupancy: [Bitboard; COLOR_COUNT],
    pub all_pieces: Bitboard,
    pub side: Color,
    pub en_passant: Option<u8>,
    pub hash: u64,
    pub nnue: Option<Box<nnue_v2::KiyNNUE>>,
}

lazy_static::lazy_static! {
    pub static ref ZOBRIST: crate::defs::Zobrist = crate::defs::get_zobrist_keys();
    pub static ref ATTACKS: crate::defs::AttackTables = crate::defs::get_attack_tables();
}

impl Board {
    pub fn new() -> Self {
        let mut board = Board {
            pieces: [[EMPTY; PIECE_TYPE_COUNT]; COLOR_COUNT],
            occupancy: [EMPTY; COLOR_COUNT],
            all_pieces: EMPTY,
            side: Color::White,
            en_passant: None,
            hash: 0,
            nnue: None,
        };

        board.init_start_position();

        match nnue_v2::KiyNNUE::load("weights.bin") {
            Ok(mut nnue_instance) => {
                nnue_instance.refresh_accumulators(&board);
                board.nnue = Some(Box::new(nnue_instance));
            }
            Err(e) => {
                println!("[WARN] Failed to load NNUE weights: {}. Using fallback eval.", e);
            }
        }

        board.hash = board.calculate_hash();
        board
    }

    pub fn piece_at_sq(&self, sq: usize) -> Option<(PieceType, Color)> {
        let bb = 1u64 << sq;
        for c in 0..COLOR_COUNT {
            for p in 0..PIECE_TYPE_COUNT {
                if (self.pieces[c][p] & bb) != 0 {
                    return Some((PieceType::from_usize(p), Color::from_usize(c)));
                }
            }
        }
        None
    }

    fn init_start_position(&mut self) {
        self.pieces[Color::White as usize][PieceType::Pawn as usize] = RANK_2;
        self.pieces[Color::White as usize][PieceType::Rook as usize] = (1 << 0) | (1 << 7);
        self.pieces[Color::White as usize][PieceType::Knight as usize] = (1 << 1) | (1 << 6);
        self.pieces[Color::White as usize][PieceType::Bishop as usize] = (1 << 2) | (1 << 5);
        self.pieces[Color::White as usize][PieceType::Queen as usize] = 1 << 3;
        self.pieces[Color::White as usize][PieceType::King as usize] = 1 << 4;
        self.pieces[Color::Black as usize][PieceType::Pawn as usize] = RANK_7;
        self.pieces[Color::Black as usize][PieceType::Rook as usize] = (1 << 56) | (1 << 63);
        self.pieces[Color::Black as usize][PieceType::Knight as usize] = (1 << 57) | (1 << 62);
        self.pieces[Color::Black as usize][PieceType::Bishop as usize] = (1 << 58) | (1 << 61);
        self.pieces[Color::Black as usize][PieceType::Queen as usize] = 1 << 59;
        self.pieces[Color::Black as usize][PieceType::King as usize] = 1 << 60;
        self.update_occupancies();
    }

    pub fn update_occupancies(&mut self) {
        self.occupancy = [EMPTY; COLOR_COUNT];
        self.all_pieces = EMPTY;
        for piece in 0..PIECE_TYPE_COUNT {
            self.occupancy[Color::White as usize] |= self.pieces[Color::White as usize][piece];
            self.occupancy[Color::Black as usize] |= self.pieces[Color::Black as usize][piece];
        }
        self.all_pieces = self.occupancy[Color::White as usize] | self.occupancy[Color::Black as usize];
    }

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
        if self.side == Color::Black { h ^= ZOBRIST.side; }
        if let Some(ep_sq) = self.en_passant {
            h ^= ZOBRIST.en_passant[(ep_sq % 8) as usize];
        }
        h
    }

    pub fn make_move(&mut self, m: Move) -> bool {
        let source = m.source() as usize;
        let target = m.target() as usize;
        let side = self.side;
        let opp = side.opposite();

        let moving_piece_opt = self.piece_at_sq(source);
        if moving_piece_opt.is_none() { return false; }
        let (moving_pt, _) = moving_piece_opt.unwrap();

        let captured_piece_opt = self.piece_at_sq(target);

        if let Some(nnue) = self.nnue.as_mut() {
            let (w_idx, b_idx) = nnue_v2::get_feature_indices(moving_pt, side, source);
            unsafe {
                nnue_v2::update_accumulator_sub(&nnue.weights, &mut nnue.white_acc, w_idx);
                nnue_v2::update_accumulator_sub(&nnue.weights, &mut nnue.black_acc, b_idx);
            }
            if let Some((captured_pt, _)) = captured_piece_opt {
                let (w_idx_cap, b_idx_cap) = nnue_v2::get_feature_indices(captured_pt, opp, target);
                unsafe {
                    nnue_v2::update_accumulator_sub(&nnue.weights, &mut nnue.white_acc, w_idx_cap);
                    nnue_v2::update_accumulator_sub(&nnue.weights, &mut nnue.black_acc, b_idx_cap);
                }
            }
        }
        
        self.hash ^= ZOBRIST.side;
        if let Some(ep_sq) = self.en_passant { self.hash ^= ZOBRIST.en_passant[(ep_sq % 8) as usize]; }

        let from_bb = 1u64 << source;
        self.pieces[side as usize][moving_pt as usize] &= !from_bb;
        self.hash ^= ZOBRIST.pieces[side as usize][moving_pt as usize][source];

        if let Some((captured_pt, _)) = captured_piece_opt {
            self.pieces[opp as usize][captured_pt as usize] &= !(1u64 << target);
            self.hash ^= ZOBRIST.pieces[opp as usize][captured_pt as usize][target];
        }

        let final_pt = m.prom().unwrap_or(moving_pt);
        let to_bb = 1u64 << target;
        self.pieces[side as usize][final_pt as usize] |= to_bb;
        self.hash ^= ZOBRIST.pieces[side as usize][final_pt as usize][target];

        if let Some(nnue) = self.nnue.as_mut() {
            let (w_idx, b_idx) = nnue_v2::get_feature_indices(final_pt, side, target);
            unsafe {
                nnue_v2::update_accumulator_add(&nnue.weights, &mut nnue.white_acc, w_idx);
                nnue_v2::update_accumulator_add(&nnue.weights, &mut nnue.black_acc, b_idx);
            }
        }

        self.en_passant = None;
        if moving_pt == PieceType::Pawn {
            if (target as i16 - source as i16).abs() == 16 {
                let ep_sq = if target > source { source + 8 } else { source - 8 };
                self.en_passant = Some(ep_sq as u8);
                self.hash ^= ZOBRIST.en_passant[(ep_sq % 8) as usize];
            }
        }

        self.update_occupancies();
        self.side = opp;
        true
    }

    pub fn is_square_attacked(&self, sq: u8, attacker_color: Color) -> bool {
        let side = attacker_color as usize;
        let opp_side = attacker_color.opposite() as usize;
        let attacks_ref = &ATTACKS;

        let attacker_pawn_bb = self.pieces[side][PieceType::Pawn as usize];
        if (attacks_ref.pawn_attacks[opp_side][sq as usize] & attacker_pawn_bb) != 0 { return true; }

        let attacker_knight_bb = self.pieces[side][PieceType::Knight as usize];
        if (attacks_ref.knight[sq as usize] & attacker_knight_bb) != 0 { return true; }

        let attacker_king_bb = self.pieces[side][PieceType::King as usize];
        if (attacks_ref.king[sq as usize] & attacker_king_bb) != 0 { return true; }

        let occ = self.all_pieces;
        let bm = &attacks_ref.bishop_magics[sq as usize];
        let bi = ((occ & bm.mask).wrapping_mul(bm.magic)) >> bm.shift;
        let b_atk = attacks_ref.bishop_table[bm.offset + bi as usize];
        let attackers_bq = self.pieces[side][PieceType::Bishop as usize] | self.pieces[side][PieceType::Queen as usize];
        if (b_atk & attackers_bq) != 0 { return true; }

        let rm = &attacks_ref.rook_magics[sq as usize];
        let ri = ((occ & rm.mask).wrapping_mul(rm.magic)) >> rm.shift;
        let r_atk = attacks_ref.rook_table[rm.offset + ri as usize];
        let attackers_rq = self.pieces[side][PieceType::Rook as usize] | self.pieces[side][PieceType::Queen as usize];
        if (r_atk & attackers_rq) != 0 { return true; }

        false
    }

    pub fn print(&self) {
        println!("  +---+---+---+---+---+---+---+---+");
        for rank in (0..8).rev() {
            print!("{} |", rank + 1);
            for file in 0..8 {
                let square = rank * 8 + file;
                if let Some((pt, c)) = self.piece_at_sq(square as usize) {
                    let piece_char = match pt {
                        PieceType::Pawn => 'p', PieceType::Knight => 'n', PieceType::Bishop => 'b',
                        PieceType::Rook => 'r', PieceType::Queen => 'q', PieceType::King => 'k',
                    };
                    if c == Color::White {
                        print!(" {} |", piece_char.to_ascii_uppercase());
                    } else {
                        print!(" {} |", piece_char);
                    }
                } else {
                    print!("   |");
                }
            }
            println!("\n  +---+---+---+---+---+---+---+---+");
        }
        println!("    a   b   c   d   e   f   g   h");
        println!();
        println!("Side to move: {:?}", self.side);
        println!("Hash: {:016X}", self.hash);
    }
}
