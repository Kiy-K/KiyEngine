use crate::board::Board;
use crate::defs::{Bitboard, Color, PieceType, EMPTY};
use crate::mv::{Move, MoveList};

pub struct MoveGen;

impl MoveGen {
    pub fn generate_legal_moves(board: &Board) -> MoveList {
        let pseudo_moves = Self::generate_moves(board);
        let mut legal_moves = MoveList::new();
        let side = board.side;

        for i in 0..pseudo_moves.count {
            let m = pseudo_moves.moves[i];
            let mut temp_board = board.clone();
            
            // Note: temp_board.make_move will flip the side.
            if temp_board.make_move(m) {
                // We check if OUR king (side) is attacked by opponent (temp_board.side).
                let king_sq_bb = temp_board.pieces[side as usize][PieceType::King as usize];
                if king_sq_bb != 0 {
                    let king_sq = king_sq_bb.trailing_zeros() as u8;
                    if !temp_board.is_square_attacked(king_sq, temp_board.side) {
                        legal_moves.add(m);
                    }
                }
            }
        }
        legal_moves
    }

    /// Generates all pseudo-legal moves for the current side.
    pub fn generate_moves(board: &Board) -> MoveList {
        let mut list = MoveList::new();
        let side = board.side;
        
        // --- Knights ---
        let knights = board.pieces[side as usize][PieceType::Knight as usize];
        if knights != EMPTY {
            Self::generate_knight_moves(board, knights, side, &mut list);
        }

        // --- Kings ---
        let king = board.pieces[side as usize][PieceType::King as usize];
        if king != EMPTY {
            Self::generate_king_moves(board, king, side, &mut list);
        }

        // --- Pawns ---
        let pawns = board.pieces[side as usize][PieceType::Pawn as usize];
        if pawns != EMPTY {
            Self::generate_pawn_moves(board, pawns, side, &mut list);
        }

        // --- Sliding Pieces (Rooks, Bishops, Queens) ---
        let rooks = board.pieces[side as usize][PieceType::Rook as usize];
        let bishops = board.pieces[side as usize][PieceType::Bishop as usize];
        let queens = board.pieces[side as usize][PieceType::Queen as usize];

        if rooks != EMPTY { Self::generate_sliding_moves(board, rooks, side, PieceType::Rook, &mut list); }
        if bishops != EMPTY { Self::generate_sliding_moves(board, bishops, side, PieceType::Bishop, &mut list); }
        if queens != EMPTY { Self::generate_sliding_moves(board, queens, side, PieceType::Queen, &mut list); }

        list
    }

    fn generate_knight_moves(board: &Board, mut knights: Bitboard, side: Color, list: &mut MoveList) {
        let own_pieces = board.occupancy[side as usize];
        while knights != 0 {
            let from_sq = knights.trailing_zeros() as u8;
            knights &= knights - 1;

            let targets = crate::board::ATTACKS.knight[from_sq as usize];
            let mut moves_bb = targets & !own_pieces;

            while moves_bb != 0 {
                let to_sq = moves_bb.trailing_zeros() as u8;
                moves_bb &= moves_bb - 1;
                list.add(Move::new(from_sq, to_sq, None));
            }
        }
    }

    fn generate_king_moves(board: &Board, mut king: Bitboard, side: Color, list: &mut MoveList) {
        let own_pieces = board.occupancy[side as usize];
        while king != 0 {
            let from_sq = king.trailing_zeros() as u8;
            king &= king - 1;

            let targets = crate::board::ATTACKS.king[from_sq as usize];
            let mut moves_bb = targets & !own_pieces;

            while moves_bb != 0 {
                let to_sq = moves_bb.trailing_zeros() as u8;
                moves_bb &= moves_bb - 1;
                list.add(Move::new(from_sq, to_sq, None));
            }
        }
    }

    fn generate_pawn_moves(board: &Board, mut pawns: Bitboard, side: Color, list: &mut MoveList) {
        let white = side == Color::White;
        while pawns != 0 {
            let from_sq = pawns.trailing_zeros() as u8;
            pawns &= pawns - 1;
            let rank = from_sq / 8;

            if white {
                // Single push
                let target = from_sq + 8;
                if target < 64 && ((board.all_pieces >> target) & 1 == 0) {
                    if rank == 6 {
                        // All 4 promotions
                        list.add(Move::new(from_sq, target, Some(PieceType::Queen)));
                        list.add(Move::new(from_sq, target, Some(PieceType::Rook)));
                        list.add(Move::new(from_sq, target, Some(PieceType::Bishop)));
                        list.add(Move::new(from_sq, target, Some(PieceType::Knight)));
                    } else {
                        list.add(Move::new(from_sq, target, None));
                        if rank == 1 {
                            let dt = from_sq + 16;
                            if (board.all_pieces >> dt) & 1 == 0 {
                                list.add(Move::new(from_sq, dt, None));
                            }
                        }
                    }
                }
            } else {
                // Black
                let target = from_sq - 8;
                if (board.all_pieces >> target) & 1 == 0 {
                    if rank == 1 {
                        list.add(Move::new(from_sq, target, Some(PieceType::Queen)));
                        list.add(Move::new(from_sq, target, Some(PieceType::Rook)));
                        list.add(Move::new(from_sq, target, Some(PieceType::Bishop)));
                        list.add(Move::new(from_sq, target, Some(PieceType::Knight)));
                    } else {
                        list.add(Move::new(from_sq, target, None));
                        if rank == 6 {
                            let dt = from_sq - 16;
                            if (board.all_pieces >> dt) & 1 == 0 {
                                list.add(Move::new(from_sq, dt, None));
                            }
                        }
                    }
                }
            }

            // Captures
            let capture_bb = crate::board::ATTACKS.pawn_attacks[side as usize][from_sq as usize];
            let mut catches = capture_bb & board.occupancy[side.opposite() as usize];
            while catches != 0 {
                let to_sq = catches.trailing_zeros() as u8;
                catches &= catches - 1;
                let tr = to_sq / 8;
                if (white && tr == 7) || (!white && tr == 0) {
                    list.add(Move::new(from_sq, to_sq, Some(PieceType::Queen)));
                    list.add(Move::new(from_sq, to_sq, Some(PieceType::Rook)));
                    list.add(Move::new(from_sq, to_sq, Some(PieceType::Bishop)));
                    list.add(Move::new(from_sq, to_sq, Some(PieceType::Knight)));
                } else {
                    list.add(Move::new(from_sq, to_sq, None));
                }
            }
            
            // En Passant
            if let Some(ep_sq) = board.en_passant {
                if (capture_bb & (1u64 << ep_sq)) != 0 {
                    list.add(Move::new(from_sq, ep_sq, None));
                }
            }
        }
    }

    fn generate_sliding_moves(board: &Board, mut pieces: Bitboard, side: Color, piece_type: PieceType, list: &mut MoveList) {
        let occ = board.all_pieces;
        let own_pieces = board.occupancy[side as usize];
        let attacks_ref = &crate::board::ATTACKS; // Cache reference

        while pieces != 0 {
            let from_sq = pieces.trailing_zeros() as u8;
            pieces &= pieces - 1;

            let attacks = match piece_type {
                PieceType::Bishop => {
                    let m = &attacks_ref.bishop_magics[from_sq as usize];
                    let hash = ((occ & m.mask).wrapping_mul(m.magic)) >> m.shift;
                    attacks_ref.bishop_table[m.offset + hash as usize]
                },
                PieceType::Rook => {
                    let m = &attacks_ref.rook_magics[from_sq as usize];
                    let hash = ((occ & m.mask).wrapping_mul(m.magic)) >> m.shift;
                    attacks_ref.rook_table[m.offset + hash as usize]
                },
                PieceType::Queen => {
                    let bm = &attacks_ref.bishop_magics[from_sq as usize];
                    let bi = ((occ & bm.mask).wrapping_mul(bm.magic)) >> bm.shift;
                    let rm = &attacks_ref.rook_magics[from_sq as usize];
                    let ri = ((occ & rm.mask).wrapping_mul(rm.magic)) >> rm.shift;
                    attacks_ref.bishop_table[bm.offset + bi as usize] | 
                    attacks_ref.rook_table[rm.offset + ri as usize]
                },
                _ => 0,
            };

            let mut moves_bb = attacks & !own_pieces;
            while moves_bb != 0 {
                let to_sq = moves_bb.trailing_zeros() as u8;
                moves_bb &= moves_bb - 1;
                list.add(Move::new(from_sq, to_sq, None));
            }
        }
    }
}
