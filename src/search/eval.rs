use crate::eval::pst::PST;
use chess::{BitBoard, Board, Color, Piece, Square};

// =============================================================================
// MASKS
// =============================================================================

const FILE_MASKS: [u64; 8] = [
    0x0101010101010101,
    0x0202020202020202,
    0x0404040404040404,
    0x0808080808080808,
    0x1010101010101010,
    0x2020202020202020,
    0x4040404040404040,
    0x8080808080808080,
];

const ADJACENT_FILES: [u64; 8] = [
    FILE_MASKS[1],
    FILE_MASKS[0] | FILE_MASKS[2],
    FILE_MASKS[1] | FILE_MASKS[3],
    FILE_MASKS[2] | FILE_MASKS[4],
    FILE_MASKS[3] | FILE_MASKS[5],
    FILE_MASKS[4] | FILE_MASKS[6],
    FILE_MASKS[5] | FILE_MASKS[7],
    FILE_MASKS[6],
];

const RANK_MASKS: [u64; 8] = [
    0x00000000000000FF,
    0x000000000000FF00,
    0x0000000000FF0000,
    0x00000000FF000000,
    0x000000FF00000000,
    0x0000FF0000000000,
    0x00FF000000000000,
    0xFF00000000000000,
];

// =============================================================================
// EVALUATION CONSTANTS (tuned values from Stockfish classical / Fruit / CPW)
// =============================================================================

const TOTAL_PHASE: i32 = 24;

// Material values: [Pawn, Knight, Bishop, Rook, Queen] (midgame / endgame)
const PIECE_VALUE_MG: [i32; 5] = [100, 320, 330, 500, 975];
const PIECE_VALUE_EG: [i32; 5] = [110, 320, 350, 550, 1000];

// Tempo: side-to-move advantage
const TEMPO_MG: i32 = 20;
const TEMPO_EG: i32 = 10;

// Bishop pair
const BISHOP_PAIR_MG: i32 = 40;
const BISHOP_PAIR_EG: i32 = 55;

// Passed pawn bonus by rank (index 0=back rank, 7=promotion rank — never occupied)
const PASSED_PAWN_MG: [i32; 8] = [0, 5, 10, 25, 50, 90, 160, 0];
const PASSED_PAWN_EG: [i32; 8] = [0, 15, 25, 50, 85, 150, 250, 0];
const PROTECTED_PASSER: [i32; 8] = [0, 5, 10, 20, 40, 70, 125, 0];

// Pawn structure penalties
const ISOLATED_MG: i32 = -15;
const ISOLATED_EG: i32 = -20;
const DOUBLED_MG: i32 = -10;
const DOUBLED_EG: i32 = -20;
const BACKWARD_MG: i32 = -12;
const BACKWARD_EG: i32 = -15;
const CONNECTED_MG: i32 = 8;
const CONNECTED_EG: i32 = 12;

// Rook bonuses
const ROOK_OPEN_FILE_MG: i32 = 25;
const ROOK_OPEN_FILE_EG: i32 = 15;
const ROOK_SEMI_OPEN_MG: i32 = 12;
const ROOK_SEMI_OPEN_EG: i32 = 8;
const ROOK_ON_SEVENTH_MG: i32 = 30;
const ROOK_ON_SEVENTH_EG: i32 = 50;
const ROOK_BEHIND_PASSER_EG: i32 = 25;

// Outpost bonuses (knight/bishop on supported square, no enemy pawn can attack)
const KNIGHT_OUTPOST_MG: i32 = 30;
const KNIGHT_OUTPOST_EG: i32 = 20;
const BISHOP_OUTPOST_MG: i32 = 20;
const BISHOP_OUTPOST_EG: i32 = 15;

// Mobility per accessible square (midgame / endgame)
const KNIGHT_MOB_MG: i32 = 4;
const KNIGHT_MOB_EG: i32 = 4;
const BISHOP_MOB_MG: i32 = 5;
const BISHOP_MOB_EG: i32 = 5;
const ROOK_MOB_MG: i32 = 3;
const ROOK_MOB_EG: i32 = 4;
const QUEEN_MOB_MG: i32 = 2;
const QUEEN_MOB_EG: i32 = 3;

// King safety
const PAWN_SHIELD_MG: i32 = 12;
const PAWN_STORM_MG: i32 = -8;
// Quadratic king danger: indexed by number of attack units (capped at 7)
const KING_DANGER: [i32; 8] = [0, 0, 50, 75, 88, 94, 97, 100];

// Space: bonus per advanced center pawn
const SPACE_MG: i32 = 3;

// King proximity to passed pawns (endgame)
const KING_PASSER_PROX: i32 = 10;

// =============================================================================
// MAIN EVALUATION FUNCTION
// =============================================================================

/// Comprehensive classical evaluation with tapered midgame/endgame scoring.
/// Returns score from the perspective of the side to move.
pub fn evaluate(board: &Board, _ply: usize) -> i32 {
    let mut mg: i32 = 0;
    let mut eg: i32 = 0;

    let white_bb = *board.color_combined(Color::White);
    let black_bb = *board.color_combined(Color::Black);
    let white_pawns = *board.pieces(Piece::Pawn) & white_bb;
    let black_pawns = *board.pieces(Piece::Pawn) & black_bb;
    let occupied = *board.combined();
    let phase = PST::game_phase(board);
    let w_king = board.king_square(Color::White);
    let b_king = board.king_square(Color::Black);

    // --- 1. Material (tapered) + PST ---
    let pieces = [
        Piece::Pawn,
        Piece::Knight,
        Piece::Bishop,
        Piece::Rook,
        Piece::Queen,
    ];
    for (i, &piece) in pieces.iter().enumerate() {
        let w_bb = *board.pieces(piece) & white_bb;
        let b_bb = *board.pieces(piece) & black_bb;
        let diff = w_bb.0.count_ones() as i32 - b_bb.0.count_ones() as i32;
        mg += diff * PIECE_VALUE_MG[i];
        eg += diff * PIECE_VALUE_EG[i];
        for sq in w_bb {
            mg += PST::get_score(piece, sq, Color::White);
            eg += PST::get_score_eg(piece, sq, Color::White);
        }
        for sq in b_bb {
            mg -= PST::get_score(piece, sq, Color::Black);
            eg -= PST::get_score_eg(piece, sq, Color::Black);
        }
    }
    // King PST
    mg += PST::get_score(Piece::King, w_king, Color::White);
    eg += PST::get_score_eg(Piece::King, w_king, Color::White);
    mg -= PST::get_score(Piece::King, b_king, Color::Black);
    eg -= PST::get_score_eg(Piece::King, b_king, Color::Black);

    // --- 2. Tempo ---
    if board.side_to_move() == Color::White {
        mg += TEMPO_MG;
        eg += TEMPO_EG;
    } else {
        mg -= TEMPO_MG;
        eg -= TEMPO_EG;
    }

    // --- 3. Bishop pair ---
    if (*board.pieces(Piece::Bishop) & white_bb).0.count_ones() >= 2 {
        mg += BISHOP_PAIR_MG;
        eg += BISHOP_PAIR_EG;
    }
    if (*board.pieces(Piece::Bishop) & black_bb).0.count_ones() >= 2 {
        mg -= BISHOP_PAIR_MG;
        eg -= BISHOP_PAIR_EG;
    }

    // --- 4. Pawn structure ---
    eval_pawns_white(white_pawns, black_pawns, w_king, b_king, &mut mg, &mut eg);
    eval_pawns_black(white_pawns, black_pawns, w_king, b_king, &mut mg, &mut eg);

    // --- 5. Piece evaluation (mobility, outposts, rook bonuses, queen mob) ---
    eval_pieces(
        board,
        white_bb,
        black_bb,
        white_pawns,
        black_pawns,
        occupied,
        &mut mg,
        &mut eg,
    );

    // --- 6. King safety (midgame only) ---
    if phase > 6 {
        eval_king_safety(
            board,
            white_bb,
            black_bb,
            white_pawns,
            black_pawns,
            occupied,
            w_king,
            b_king,
            &mut mg,
        );
    }

    // --- 7. Space ---
    let center: u64 = FILE_MASKS[2] | FILE_MASKS[3] | FILE_MASKS[4] | FILE_MASKS[5];
    let w_space = (white_pawns & BitBoard(center & (RANK_MASKS[3] | RANK_MASKS[4])))
        .0
        .count_ones() as i32;
    let b_space = (black_pawns & BitBoard(center & (RANK_MASKS[3] | RANK_MASKS[4])))
        .0
        .count_ones() as i32;
    mg += (w_space - b_space) * SPACE_MG;

    // --- 8. Development penalty (opening only) ---
    if phase > 16 {
        mg += eval_development(board, white_bb, black_bb);
    }

    // --- Taper midgame/endgame ---
    let score = (mg * phase + eg * (TOTAL_PHASE - phase)) / TOTAL_PHASE;
    if board.side_to_move() == Color::White {
        score
    } else {
        -score
    }
}

// =============================================================================
// PAWN EVALUATION
// =============================================================================

#[inline]
fn eval_pawns_white(
    white_pawns: BitBoard,
    black_pawns: BitBoard,
    w_king: Square,
    b_king: Square,
    mg: &mut i32,
    eg: &mut i32,
) {
    for sq in white_pawns {
        let idx = sq.to_index();
        let file = idx % 8;
        let rank = idx / 8;
        let fmask = FILE_MASKS[file];
        let adj = ADJACENT_FILES[file];

        // Passed pawn
        let ahead = if rank < 7 {
            (fmask | adj) & !((1u64 << ((rank + 1) * 8)) - 1)
        } else {
            0
        };
        let is_passed = rank > 0 && (black_pawns & BitBoard(ahead)).0 == 0;
        if is_passed {
            *mg += PASSED_PAWN_MG[rank];
            *eg += PASSED_PAWN_EG[rank];
            // Protected passer
            if rank > 0 {
                let br = (rank - 1) * 8;
                let mut pm = 0u64;
                if file > 0 {
                    pm |= 1u64 << (br + file - 1);
                }
                if file < 7 {
                    pm |= 1u64 << (br + file + 1);
                }
                if (white_pawns & BitBoard(pm)).0 != 0 {
                    *eg += PROTECTED_PASSER[rank];
                }
            }
            // King proximity bonus (endgame)
            let own_d = chebyshev(w_king.to_index(), idx);
            let opp_d = chebyshev(b_king.to_index(), idx);
            *eg += (opp_d as i32 - own_d as i32) * KING_PASSER_PROX;
        }

        // Doubled pawn
        let file_ahead = fmask & !((1u64 << (idx + 1)) - 1) & !(1u64 << idx);
        if (white_pawns & BitBoard(file_ahead)).0 != 0 {
            *mg += DOUBLED_MG;
            *eg += DOUBLED_EG;
        }

        // Isolated pawn
        let is_isolated = (white_pawns & BitBoard(adj)).0 == 0;
        if is_isolated {
            *mg += ISOLATED_MG;
            *eg += ISOLATED_EG;
        }

        // Backward pawn: no friendly pawn on adjacent files at same or lower rank,
        // and stop square is attacked by enemy pawn
        if !is_isolated && !is_passed && rank < 7 {
            let behind = adj & ((1u64 << ((rank + 1) * 8)) - 1);
            if (white_pawns & BitBoard(behind)).0 == 0 {
                let stop_rank = rank + 1;
                let mut atk = 0u64;
                if stop_rank < 7 {
                    if file > 0 {
                        atk |= 1u64 << ((stop_rank + 1) * 8 + file - 1);
                    }
                    if file < 7 {
                        atk |= 1u64 << ((stop_rank + 1) * 8 + file + 1);
                    }
                }
                if (black_pawns & BitBoard(atk)).0 != 0 {
                    *mg += BACKWARD_MG;
                    *eg += BACKWARD_EG;
                }
            }
        }

        // Connected pawn (supported by friendly pawn diagonally or phalanx)
        let mut sup = 0u64;
        if rank > 0 {
            if file > 0 {
                sup |= 1u64 << ((rank - 1) * 8 + file - 1);
            }
            if file < 7 {
                sup |= 1u64 << ((rank - 1) * 8 + file + 1);
            }
        }
        if file > 0 {
            sup |= 1u64 << (rank * 8 + file - 1);
        }
        if file < 7 {
            sup |= 1u64 << (rank * 8 + file + 1);
        }
        if (white_pawns & BitBoard(sup)).0 != 0 {
            *mg += CONNECTED_MG;
            *eg += CONNECTED_EG;
        }
    }
}

#[inline]
fn eval_pawns_black(
    white_pawns: BitBoard,
    black_pawns: BitBoard,
    w_king: Square,
    b_king: Square,
    mg: &mut i32,
    eg: &mut i32,
) {
    for sq in black_pawns {
        let idx = sq.to_index();
        let file = idx % 8;
        let rank = idx / 8;
        let brank = 7 - rank;
        let fmask = FILE_MASKS[file];
        let adj = ADJACENT_FILES[file];

        // Passed pawn
        let ahead = if rank > 0 {
            (fmask | adj) & ((1u64 << (rank * 8)) - 1)
        } else {
            0
        };
        let is_passed = brank > 0 && (white_pawns & BitBoard(ahead)).0 == 0;
        if is_passed {
            *mg -= PASSED_PAWN_MG[brank];
            *eg -= PASSED_PAWN_EG[brank];
            if rank < 7 {
                let br = (rank + 1) * 8;
                let mut pm = 0u64;
                if file > 0 {
                    pm |= 1u64 << (br + file - 1);
                }
                if file < 7 {
                    pm |= 1u64 << (br + file + 1);
                }
                if (black_pawns & BitBoard(pm)).0 != 0 {
                    *eg -= PROTECTED_PASSER[brank];
                }
            }
            let own_d = chebyshev(b_king.to_index(), idx);
            let opp_d = chebyshev(w_king.to_index(), idx);
            *eg -= (opp_d as i32 - own_d as i32) * KING_PASSER_PROX;
        }

        // Doubled
        let file_behind = fmask & ((1u64 << idx) - 1);
        if (black_pawns & BitBoard(file_behind)).0 != 0 {
            *mg -= DOUBLED_MG;
            *eg -= DOUBLED_EG;
        }

        // Isolated
        let is_isolated = (black_pawns & BitBoard(adj)).0 == 0;
        if is_isolated {
            *mg -= ISOLATED_MG;
            *eg -= ISOLATED_EG;
        }

        // Backward
        if !is_isolated && !is_passed && rank > 0 {
            let behind = adj & !((1u64 << (rank * 8)) - 1) & !RANK_MASKS[rank];
            if (black_pawns & BitBoard(behind)).0 == 0 {
                let stop_rank = rank - 1;
                let mut atk = 0u64;
                if stop_rank > 0 {
                    if file > 0 {
                        atk |= 1u64 << ((stop_rank - 1) * 8 + file - 1);
                    }
                    if file < 7 {
                        atk |= 1u64 << ((stop_rank - 1) * 8 + file + 1);
                    }
                }
                if (white_pawns & BitBoard(atk)).0 != 0 {
                    *mg -= BACKWARD_MG;
                    *eg -= BACKWARD_EG;
                }
            }
        }

        // Connected
        let mut sup = 0u64;
        if rank < 7 {
            if file > 0 {
                sup |= 1u64 << ((rank + 1) * 8 + file - 1);
            }
            if file < 7 {
                sup |= 1u64 << ((rank + 1) * 8 + file + 1);
            }
        }
        if file > 0 {
            sup |= 1u64 << (rank * 8 + file - 1);
        }
        if file < 7 {
            sup |= 1u64 << (rank * 8 + file + 1);
        }
        if (black_pawns & BitBoard(sup)).0 != 0 {
            *mg -= CONNECTED_MG;
            *eg -= CONNECTED_EG;
        }
    }
}

// =============================================================================
// PIECE EVALUATION: Mobility + Outposts + Rook bonuses + Queen
// =============================================================================

#[inline]
fn eval_pieces(
    board: &Board,
    white_bb: BitBoard,
    black_bb: BitBoard,
    white_pawns: BitBoard,
    black_pawns: BitBoard,
    occupied: BitBoard,
    mg: &mut i32,
    eg: &mut i32,
) {
    // --- Knights ---
    for sq in *board.pieces(Piece::Knight) & white_bb {
        let acc = (chess::get_knight_moves(sq) & !white_bb).0.count_ones() as i32;
        *mg += acc * KNIGHT_MOB_MG;
        *eg += acc * KNIGHT_MOB_EG;
        eval_outpost_white(
            sq,
            white_pawns,
            black_pawns,
            KNIGHT_OUTPOST_MG,
            KNIGHT_OUTPOST_EG,
            mg,
            eg,
        );
    }
    for sq in *board.pieces(Piece::Knight) & black_bb {
        let acc = (chess::get_knight_moves(sq) & !black_bb).0.count_ones() as i32;
        *mg -= acc * KNIGHT_MOB_MG;
        *eg -= acc * KNIGHT_MOB_EG;
        eval_outpost_black(
            sq,
            white_pawns,
            black_pawns,
            KNIGHT_OUTPOST_MG,
            KNIGHT_OUTPOST_EG,
            mg,
            eg,
        );
    }

    // --- Bishops ---
    for sq in *board.pieces(Piece::Bishop) & white_bb {
        let acc = (chess::get_bishop_moves(sq, occupied) & !white_bb)
            .0
            .count_ones() as i32;
        *mg += acc * BISHOP_MOB_MG;
        *eg += acc * BISHOP_MOB_EG;
        eval_outpost_white(
            sq,
            white_pawns,
            black_pawns,
            BISHOP_OUTPOST_MG,
            BISHOP_OUTPOST_EG,
            mg,
            eg,
        );
    }
    for sq in *board.pieces(Piece::Bishop) & black_bb {
        let acc = (chess::get_bishop_moves(sq, occupied) & !black_bb)
            .0
            .count_ones() as i32;
        *mg -= acc * BISHOP_MOB_MG;
        *eg -= acc * BISHOP_MOB_EG;
        eval_outpost_black(
            sq,
            white_pawns,
            black_pawns,
            BISHOP_OUTPOST_MG,
            BISHOP_OUTPOST_EG,
            mg,
            eg,
        );
    }

    // --- Rooks ---
    for sq in *board.pieces(Piece::Rook) & white_bb {
        let acc = (chess::get_rook_moves(sq, occupied) & !white_bb)
            .0
            .count_ones() as i32;
        *mg += acc * ROOK_MOB_MG;
        *eg += acc * ROOK_MOB_EG;
        let idx = sq.to_index();
        let file = idx % 8;
        let rank = idx / 8;
        let fmask = BitBoard(FILE_MASKS[file]);
        if (white_pawns & fmask).0 == 0 {
            if (black_pawns & fmask).0 == 0 {
                *mg += ROOK_OPEN_FILE_MG;
                *eg += ROOK_OPEN_FILE_EG;
            } else {
                *mg += ROOK_SEMI_OPEN_MG;
                *eg += ROOK_SEMI_OPEN_EG;
            }
        }
        if rank == 6 {
            *mg += ROOK_ON_SEVENTH_MG;
            *eg += ROOK_ON_SEVENTH_EG;
        }
        // Rook behind own passed pawn
        for psq in white_pawns {
            let pi = psq.to_index();
            if pi % 8 == file && pi / 8 > rank {
                let ahead =
                    (FILE_MASKS[file] | ADJACENT_FILES[file]) & !((1u64 << ((pi / 8 + 1) * 8)) - 1);
                if (black_pawns & BitBoard(ahead)).0 == 0 {
                    *eg += ROOK_BEHIND_PASSER_EG;
                    break;
                }
            }
        }
    }
    for sq in *board.pieces(Piece::Rook) & black_bb {
        let acc = (chess::get_rook_moves(sq, occupied) & !black_bb)
            .0
            .count_ones() as i32;
        *mg -= acc * ROOK_MOB_MG;
        *eg -= acc * ROOK_MOB_EG;
        let idx = sq.to_index();
        let file = idx % 8;
        let rank = idx / 8;
        let fmask = BitBoard(FILE_MASKS[file]);
        if (black_pawns & fmask).0 == 0 {
            if (white_pawns & fmask).0 == 0 {
                *mg -= ROOK_OPEN_FILE_MG;
                *eg -= ROOK_OPEN_FILE_EG;
            } else {
                *mg -= ROOK_SEMI_OPEN_MG;
                *eg -= ROOK_SEMI_OPEN_EG;
            }
        }
        if rank == 1 {
            *mg -= ROOK_ON_SEVENTH_MG;
            *eg -= ROOK_ON_SEVENTH_EG;
        }
        for psq in black_pawns {
            let pi = psq.to_index();
            if pi % 8 == file && pi / 8 < rank {
                let ahead =
                    (FILE_MASKS[file] | ADJACENT_FILES[file]) & ((1u64 << ((pi / 8) * 8)) - 1);
                if (white_pawns & BitBoard(ahead)).0 == 0 {
                    *eg -= ROOK_BEHIND_PASSER_EG;
                    break;
                }
            }
        }
    }

    // --- Queens ---
    for sq in *board.pieces(Piece::Queen) & white_bb {
        let r = chess::get_rook_moves(sq, occupied);
        let b = chess::get_bishop_moves(sq, occupied);
        let acc = ((r | b) & !white_bb).0.count_ones() as i32;
        *mg += acc * QUEEN_MOB_MG;
        *eg += acc * QUEEN_MOB_EG;
    }
    for sq in *board.pieces(Piece::Queen) & black_bb {
        let r = chess::get_rook_moves(sq, occupied);
        let b = chess::get_bishop_moves(sq, occupied);
        let acc = ((r | b) & !black_bb).0.count_ones() as i32;
        *mg -= acc * QUEEN_MOB_MG;
        *eg -= acc * QUEEN_MOB_EG;
    }
}

// =============================================================================
// OUTPOST HELPERS
// =============================================================================

#[inline]
fn eval_outpost_white(
    sq: Square,
    white_pawns: BitBoard,
    black_pawns: BitBoard,
    bonus_mg: i32,
    bonus_eg: i32,
    mg: &mut i32,
    eg: &mut i32,
) {
    let idx = sq.to_index();
    let file = idx % 8;
    let rank = idx / 8;
    if rank >= 3 && rank <= 5 {
        // Supported by own pawn?
        let mut sup = 0u64;
        if rank > 0 {
            if file > 0 {
                sup |= 1u64 << ((rank - 1) * 8 + file - 1);
            }
            if file < 7 {
                sup |= 1u64 << ((rank - 1) * 8 + file + 1);
            }
        }
        if (white_pawns & BitBoard(sup)).0 != 0 {
            // No enemy pawn can attack this square?
            let enemy_ahead = ADJACENT_FILES[file] & !((1u64 << ((rank + 1) * 8)) - 1);
            if (black_pawns & BitBoard(enemy_ahead)).0 == 0 {
                *mg += bonus_mg;
                *eg += bonus_eg;
            }
        }
    }
}

#[inline]
fn eval_outpost_black(
    sq: Square,
    white_pawns: BitBoard,
    black_pawns: BitBoard,
    bonus_mg: i32,
    bonus_eg: i32,
    mg: &mut i32,
    eg: &mut i32,
) {
    let idx = sq.to_index();
    let file = idx % 8;
    let rank = idx / 8;
    if rank >= 2 && rank <= 4 {
        let mut sup = 0u64;
        if rank < 7 {
            if file > 0 {
                sup |= 1u64 << ((rank + 1) * 8 + file - 1);
            }
            if file < 7 {
                sup |= 1u64 << ((rank + 1) * 8 + file + 1);
            }
        }
        if (black_pawns & BitBoard(sup)).0 != 0 {
            let enemy_ahead = ADJACENT_FILES[file] & ((1u64 << (rank * 8)) - 1);
            if (white_pawns & BitBoard(enemy_ahead)).0 == 0 {
                *mg -= bonus_mg;
                *eg -= bonus_eg;
            }
        }
    }
}

// =============================================================================
// KING SAFETY (midgame only)
// =============================================================================

#[inline]
fn eval_king_safety(
    board: &Board,
    white_bb: BitBoard,
    black_bb: BitBoard,
    white_pawns: BitBoard,
    black_pawns: BitBoard,
    occupied: BitBoard,
    w_king: Square,
    b_king: Square,
    mg: &mut i32,
) {
    // --- White king ---
    let wk_file = w_king.to_index() % 8;
    let wk_rank = w_king.to_index() / 8;
    if wk_file < 3 || wk_file > 4 {
        let sf = wk_file.saturating_sub(1);
        let ef = (wk_file + 1).min(7);
        for f in sf..=ef {
            for dr in 1u32..=2 {
                let r = wk_rank + dr as usize;
                if r < 8 {
                    if (white_pawns & BitBoard(1u64 << (r * 8 + f))).0 != 0 {
                        *mg += PAWN_SHIELD_MG / dr as i32;
                    }
                }
            }
            for dr in 1u32..=3 {
                let r = wk_rank + dr as usize;
                if r < 8 {
                    if (black_pawns & BitBoard(1u64 << (r * 8 + f))).0 != 0 {
                        *mg += PAWN_STORM_MG * (4 - dr as i32) / 3;
                    }
                }
            }
        }
    }
    // Attack zone
    let wk_zone = chess::get_king_moves(w_king);
    let mut w_att = 0u32;
    for sq in *board.pieces(Piece::Knight) & black_bb {
        if (chess::get_knight_moves(sq) & wk_zone).0 != 0 {
            w_att += 1;
        }
    }
    for sq in *board.pieces(Piece::Bishop) & black_bb {
        if (chess::get_bishop_moves(sq, occupied) & wk_zone).0 != 0 {
            w_att += 1;
        }
    }
    for sq in *board.pieces(Piece::Rook) & black_bb {
        if (chess::get_rook_moves(sq, occupied) & wk_zone).0 != 0 {
            w_att += 1;
        }
    }
    for sq in *board.pieces(Piece::Queen) & black_bb {
        let a = chess::get_rook_moves(sq, occupied) | chess::get_bishop_moves(sq, occupied);
        if (a & wk_zone).0 != 0 {
            w_att += 2;
        }
    }
    *mg -= KING_DANGER[(w_att as usize).min(7)];

    // --- Black king ---
    let bk_file = b_king.to_index() % 8;
    let bk_rank = b_king.to_index() / 8;
    if bk_file < 3 || bk_file > 4 {
        let sf = bk_file.saturating_sub(1);
        let ef = (bk_file + 1).min(7);
        for f in sf..=ef {
            for dr in 1u32..=2 {
                if bk_rank >= dr as usize {
                    let r = bk_rank - dr as usize;
                    if (black_pawns & BitBoard(1u64 << (r * 8 + f))).0 != 0 {
                        *mg -= PAWN_SHIELD_MG / dr as i32;
                    }
                }
            }
            for dr in 1u32..=3 {
                if bk_rank >= dr as usize {
                    let r = bk_rank - dr as usize;
                    if (white_pawns & BitBoard(1u64 << (r * 8 + f))).0 != 0 {
                        *mg -= PAWN_STORM_MG * (4 - dr as i32) / 3;
                    }
                }
            }
        }
    }
    let bk_zone = chess::get_king_moves(b_king);
    let mut b_att = 0u32;
    for sq in *board.pieces(Piece::Knight) & white_bb {
        if (chess::get_knight_moves(sq) & bk_zone).0 != 0 {
            b_att += 1;
        }
    }
    for sq in *board.pieces(Piece::Bishop) & white_bb {
        if (chess::get_bishop_moves(sq, occupied) & bk_zone).0 != 0 {
            b_att += 1;
        }
    }
    for sq in *board.pieces(Piece::Rook) & white_bb {
        if (chess::get_rook_moves(sq, occupied) & bk_zone).0 != 0 {
            b_att += 1;
        }
    }
    for sq in *board.pieces(Piece::Queen) & white_bb {
        let a = chess::get_rook_moves(sq, occupied) | chess::get_bishop_moves(sq, occupied);
        if (a & bk_zone).0 != 0 {
            b_att += 2;
        }
    }
    *mg += KING_DANGER[(b_att as usize).min(7)];
}

// =============================================================================
// DEVELOPMENT (opening only)
// =============================================================================

#[inline]
fn eval_development(board: &Board, white_bb: BitBoard, black_bb: BitBoard) -> i32 {
    let mut s = 0i32;
    let wn = *board.pieces(Piece::Knight) & white_bb;
    let wb = *board.pieces(Piece::Bishop) & white_bb;
    if (wn & BitBoard(1u64 << 1)).0 != 0 {
        s -= 15;
    }
    if (wn & BitBoard(1u64 << 6)).0 != 0 {
        s -= 15;
    }
    if (wb & BitBoard(1u64 << 2)).0 != 0 {
        s -= 10;
    }
    if (wb & BitBoard(1u64 << 5)).0 != 0 {
        s -= 10;
    }
    let bn = *board.pieces(Piece::Knight) & black_bb;
    let bb = *board.pieces(Piece::Bishop) & black_bb;
    if (bn & BitBoard(1u64 << 57)).0 != 0 {
        s += 15;
    }
    if (bn & BitBoard(1u64 << 62)).0 != 0 {
        s += 15;
    }
    if (bb & BitBoard(1u64 << 58)).0 != 0 {
        s += 10;
    }
    if (bb & BitBoard(1u64 << 61)).0 != 0 {
        s += 10;
    }
    s
}

// =============================================================================
// UTILITY
// =============================================================================

#[inline]
fn chebyshev(sq1: usize, sq2: usize) -> u32 {
    let f1 = (sq1 % 8) as i32;
    let r1 = (sq1 / 8) as i32;
    let f2 = (sq2 % 8) as i32;
    let r2 = (sq2 / 8) as i32;
    (f1 - f2).abs().max((r1 - r2).abs()) as u32
}
