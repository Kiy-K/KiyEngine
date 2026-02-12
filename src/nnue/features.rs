// src/nnue/features.rs
//! NNUE feature extraction from chess::Board
//!
//! Stockfish-style HalfKAv2 with horizontal mirroring:
//! - 10 king buckets (mapping 32 half-board squares)
//! - Feature index = king_bucket * 768 + base_feature
//! - Horizontal mirroring: if king is on files e-h, flip entire board horizontally
//! - Two perspectives (white/black) with color/rank flipping for symmetry
//!
//! Also supports legacy Chess768 mode (num_input_buckets=1) for backward compat.

use chess::{Board, Color, Piece, Square};

/// Base feature count per king bucket: 2 colors × 6 piece types × 64 squares
pub const NUM_FEATURES: usize = 768;

/// Maximum active features in any legal position (at most 32 pieces)
pub const MAX_ACTIVE: usize = 32;

/// Number of king buckets (must match training config)
pub const NUM_INPUT_BUCKETS: usize = 10;

/// King bucket layout: maps 32 half-board squares (files a-d only) to 10 buckets.
/// Index = rank * 4 + file (for files 0-3 only, after horizontal mirroring).
///
/// This MUST match the BUCKET_LAYOUT in training/src/main.rs exactly.
#[rustfmt::skip]
pub const KING_BUCKET_LAYOUT: [u8; 32] = [
    0, 1, 2, 3,   // rank 1 (files a-d)
    4, 4, 5, 5,   // rank 2
    6, 6, 6, 6,   // rank 3
    7, 7, 7, 7,   // rank 4
    8, 8, 8, 8,   // rank 5
    8, 8, 8, 8,   // rank 6
    9, 9, 9, 9,   // rank 7
    9, 9, 9, 9,   // rank 8
];

/// Map chess::Piece to index 0-5
#[inline]
pub fn piece_index(piece: Piece) -> usize {
    match piece {
        Piece::Pawn => 0,
        Piece::Knight => 1,
        Piece::Bishop => 2,
        Piece::Rook => 3,
        Piece::Queen => 4,
        Piece::King => 5,
    }
}

/// Determine king bucket for a given king square (0-63).
/// Applies horizontal mirroring: if file >= 4 (e-h), mirror to files a-d.
/// Returns (bucket_index, needs_mirror).
#[inline]
pub fn king_bucket(king_sq: usize) -> (u8, bool) {
    let file = king_sq % 8;
    let rank = king_sq / 8;
    let mirror = file >= 4;
    let mirrored_file = if mirror { 7 - file } else { file };
    let half_idx = rank * 4 + mirrored_file;
    (KING_BUCKET_LAYOUT[half_idx], mirror)
}

/// Compute base feature index (within a single 768-feature bucket).
/// color: piece color, piece: piece type, sq: piece square index 0-63.
/// If `mirror` is true, horizontally flip the square (file = 7 - file).
#[inline]
fn base_feature(color: usize, piece: usize, sq: usize, mirror: bool) -> usize {
    let sq = if mirror { sq ^ 7 } else { sq }; // flip file: XOR with 7
    color * 384 + piece * 64 + sq
}

/// Compute full bucketed feature index for the WHITE perspective.
/// white_king_sq: white king's square (0-63).
#[inline]
pub fn white_bucketed_feature(
    white_king_sq: usize,
    piece_color: Color,
    piece: Piece,
    sq: Square,
) -> usize {
    let (bucket, mirror) = king_bucket(white_king_sq);
    let c = if piece_color == Color::White { 0 } else { 1 };
    let p = piece_index(piece);
    let base = base_feature(c, p, sq.to_index(), mirror);
    bucket as usize * NUM_FEATURES + base
}

/// Compute full bucketed feature index for the BLACK perspective.
/// Black perspective: flip colors (white↔black), flip ranks (sq ^ 56), then apply king bucket.
/// black_king_sq: black king's square (0-63), already in board coordinates.
#[inline]
pub fn black_bucketed_feature(
    black_king_sq: usize,
    piece_color: Color,
    piece: Piece,
    sq: Square,
) -> usize {
    // For black perspective: flip the king square rank-wise
    let flipped_king = black_king_sq ^ 56;
    let (bucket, mirror) = king_bucket(flipped_king);
    let c = if piece_color == Color::White { 1 } else { 0 }; // flip colors
    let p = piece_index(piece);
    let flipped_sq = sq.to_index() ^ 56; // flip rank
    let base = base_feature(c, p, flipped_sq, mirror);
    bucket as usize * NUM_FEATURES + base
}

/// Compute feature index for the WHITE perspective (legacy, no king buckets).
#[inline]
pub fn white_feature(color: Color, piece: Piece, sq: Square) -> usize {
    let c = if color == Color::White { 0 } else { 1 };
    let p = piece_index(piece);
    c * 384 + p * 64 + sq.to_index()
}

/// Compute feature index for the BLACK perspective (legacy, no king buckets).
#[inline]
pub fn black_feature(color: Color, piece: Piece, sq: Square) -> usize {
    let c = if color == Color::White { 1 } else { 0 };
    let p = piece_index(piece);
    c * 384 + p * 64 + (sq.to_index() ^ 56)
}

/// Total feature count including king buckets.
pub const fn total_input_features(num_input_buckets: usize) -> usize {
    num_input_buckets * NUM_FEATURES
}

/// Extract all active feature indices for both perspectives from a Board.
/// Uses king-bucketed features (HalfKAv2 with horizontal mirroring).
///
/// Returns (white_features, black_features, count).
pub fn extract_features_bucketed(
    board: &Board,
) -> ([usize; MAX_ACTIVE], [usize; MAX_ACTIVE], usize) {
    let mut white_feats = [0usize; MAX_ACTIVE];
    let mut black_feats = [0usize; MAX_ACTIVE];
    let mut count = 0;

    let wk_sq = board.king_square(Color::White).to_index();
    let bk_sq = board.king_square(Color::Black).to_index();

    for &color in &[Color::White, Color::Black] {
        let color_bb = board.color_combined(color);
        for &piece in &ALL_PIECES {
            let bb = board.pieces(piece) & color_bb;
            for sq in bb {
                debug_assert!(count < MAX_ACTIVE);
                white_feats[count] = white_bucketed_feature(wk_sq, color, piece, sq);
                black_feats[count] = black_bucketed_feature(bk_sq, color, piece, sq);
                count += 1;
            }
        }
    }

    (white_feats, black_feats, count)
}

/// Extract legacy (non-bucketed) features.
pub fn extract_features(board: &Board) -> ([usize; MAX_ACTIVE], [usize; MAX_ACTIVE], usize) {
    let mut white_feats = [0usize; MAX_ACTIVE];
    let mut black_feats = [0usize; MAX_ACTIVE];
    let mut count = 0;

    for &color in &[Color::White, Color::Black] {
        let color_bb = board.color_combined(color);
        for &piece in &ALL_PIECES {
            let bb = board.pieces(piece) & color_bb;
            for sq in bb {
                debug_assert!(count < MAX_ACTIVE);
                white_feats[count] = white_feature(color, piece, sq);
                black_feats[count] = black_feature(color, piece, sq);
                count += 1;
            }
        }
    }

    (white_feats, black_feats, count)
}

pub const ALL_PIECES: [Piece; 6] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
];

/// Feature delta for incremental accumulator updates.
/// Each entry: (white_feature_idx, black_feature_idx)
pub struct FeatureDelta {
    pub added: [(usize, usize); 2],
    pub removed: [(usize, usize); 2],
    pub num_added: usize,
    pub num_removed: usize,
    /// Whether the white king moved (requires full refresh of white accumulator)
    pub white_king_moved: bool,
    /// Whether the black king moved (requires full refresh of black accumulator)
    pub black_king_moved: bool,
}

/// Compute feature deltas for a move.
/// With king buckets: if the king moves, the bucket may change and we need a full refresh
/// for that perspective. We still compute deltas for the OTHER perspective.
/// Returns None for castling (always full refresh both perspectives).
pub fn compute_delta_bucketed(
    board: &Board,
    mv: chess::ChessMove,
    wk_sq: usize,
    bk_sq: usize,
) -> Option<FeatureDelta> {
    let src = mv.get_source();
    let dst = mv.get_dest();
    let stm = board.side_to_move();
    let moving_piece = board.piece_on(src)?;

    let mut delta = FeatureDelta {
        added: [(0, 0); 2],
        removed: [(0, 0); 2],
        num_added: 0,
        num_removed: 0,
        white_king_moved: false,
        black_king_moved: false,
    };

    // Detect castling: king moves more than 1 square
    let src_file = src.to_index() % 8;
    let dst_file = dst.to_index() % 8;
    if moving_piece == Piece::King && ((src_file as i32 - dst_file as i32).abs() > 1) {
        return None; // Full refresh both perspectives for castling
    }

    // Track king moves
    if moving_piece == Piece::King {
        if stm == Color::White {
            delta.white_king_moved = true;
        } else {
            delta.black_king_moved = true;
        }
    }

    // Remove moving piece from source
    delta.removed[delta.num_removed] = (
        white_bucketed_feature(wk_sq, stm, moving_piece, src),
        black_bucketed_feature(bk_sq, stm, moving_piece, src),
    );
    delta.num_removed += 1;

    // Handle capture
    let captured = board.piece_on(dst);
    if let Some(cap_piece) = captured {
        let cap_color = !stm;
        delta.removed[delta.num_removed] = (
            white_bucketed_feature(wk_sq, cap_color, cap_piece, dst),
            black_bucketed_feature(bk_sq, cap_color, cap_piece, dst),
        );
        delta.num_removed += 1;
    }

    // Handle en passant
    if moving_piece == Piece::Pawn && captured.is_none() {
        let src_file_idx = src.to_index() % 8;
        let dst_file_idx = dst.to_index() % 8;
        if src_file_idx != dst_file_idx {
            let ep_rank = src.to_index() / 8;
            let ep_sq = unsafe { Square::new((ep_rank * 8 + dst_file_idx) as u8) };
            delta.removed[delta.num_removed] = (
                white_bucketed_feature(wk_sq, !stm, Piece::Pawn, ep_sq),
                black_bucketed_feature(bk_sq, !stm, Piece::Pawn, ep_sq),
            );
            delta.num_removed += 1;
            if delta.num_removed > 2 {
                return None;
            }
        }
    }

    // Add piece to destination (possibly promoted)
    let placed_piece = mv.get_promotion().unwrap_or(moving_piece);
    delta.added[delta.num_added] = (
        white_bucketed_feature(wk_sq, stm, placed_piece, dst),
        black_bucketed_feature(bk_sq, stm, placed_piece, dst),
    );
    delta.num_added += 1;

    Some(delta)
}

/// Legacy non-bucketed delta computation (for v2/v3 networks).
pub fn compute_delta(board: &Board, mv: chess::ChessMove) -> Option<FeatureDelta> {
    let src = mv.get_source();
    let dst = mv.get_dest();
    let stm = board.side_to_move();
    let moving_piece = board.piece_on(src)?;

    let mut delta = FeatureDelta {
        added: [(0, 0); 2],
        removed: [(0, 0); 2],
        num_added: 0,
        num_removed: 0,
        white_king_moved: false,
        black_king_moved: false,
    };

    let src_file = src.to_index() % 8;
    let dst_file = dst.to_index() % 8;
    if moving_piece == Piece::King && ((src_file as i32 - dst_file as i32).abs() > 1) {
        return None;
    }

    if moving_piece == Piece::King {
        if stm == Color::White {
            delta.white_king_moved = true;
        } else {
            delta.black_king_moved = true;
        }
    }

    delta.removed[delta.num_removed] = (
        white_feature(stm, moving_piece, src),
        black_feature(stm, moving_piece, src),
    );
    delta.num_removed += 1;

    let captured = board.piece_on(dst);
    if let Some(cap_piece) = captured {
        delta.removed[delta.num_removed] = (
            white_feature(!stm, cap_piece, dst),
            black_feature(!stm, cap_piece, dst),
        );
        delta.num_removed += 1;
    }

    if moving_piece == Piece::Pawn && captured.is_none() {
        let src_file_idx = src.to_index() % 8;
        let dst_file_idx = dst.to_index() % 8;
        if src_file_idx != dst_file_idx {
            let ep_rank = src.to_index() / 8;
            let ep_sq = unsafe { Square::new((ep_rank * 8 + dst_file_idx) as u8) };
            delta.removed[delta.num_removed] = (
                white_feature(!stm, Piece::Pawn, ep_sq),
                black_feature(!stm, Piece::Pawn, ep_sq),
            );
            delta.num_removed += 1;
            if delta.num_removed > 2 {
                return None;
            }
        }
    }

    let placed_piece = mv.get_promotion().unwrap_or(moving_piece);
    delta.added[delta.num_added] = (
        white_feature(stm, placed_piece, dst),
        black_feature(stm, placed_piece, dst),
    );
    delta.num_added += 1;

    Some(delta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_king_bucket_layout() {
        // King on a1 (sq=0): file=0 < 4, no mirror, half_idx=0*4+0=0, bucket=0
        assert_eq!(king_bucket(0), (0, false));
        // King on e1 (sq=4): file=4 >= 4, mirror, mirrored_file=3, half_idx=0*4+3=3, bucket=3
        assert_eq!(king_bucket(4), (3, true));
        // King on h1 (sq=7): file=7, mirror, mirrored_file=0, half_idx=0, bucket=0
        assert_eq!(king_bucket(7), (0, true));
        // King on d4 (sq=27): file=3, rank=3, no mirror, half_idx=3*4+3=15, bucket=7
        assert_eq!(king_bucket(27), (7, false));
    }

    #[test]
    fn test_feature_indices_bucketed() {
        let board = Board::default();
        let (wf, bf, count) = extract_features_bucketed(&board);
        assert_eq!(count, 32);
        // All features should be in range [0, NUM_INPUT_BUCKETS * 768)
        for i in 0..count {
            assert!(wf[i] < NUM_INPUT_BUCKETS * NUM_FEATURES);
            assert!(bf[i] < NUM_INPUT_BUCKETS * NUM_FEATURES);
        }
    }

    #[test]
    fn test_extract_starting_position() {
        let board = Board::default();
        let (_, _, count) = extract_features(&board);
        assert_eq!(count, 32);
    }
}
