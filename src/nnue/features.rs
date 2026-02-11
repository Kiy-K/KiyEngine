// src/nnue/features.rs
//! NNUE feature extraction from chess::Board
//!
//! Standard 768-feature encoding: 6 piece types × 2 colors × 64 squares.
//! Two perspectives (white/black) with mirrored features for color symmetry.

use chess::{Board, Color, Piece, Square};

/// Total input features: 2 colors × 6 piece types × 64 squares
pub const NUM_FEATURES: usize = 768;

/// Maximum active features in any legal position (at most 32 pieces)
pub const MAX_ACTIVE: usize = 32;

/// Compute feature index for the WHITE perspective.
///
/// white_feature(color, piece_type, square) = color * 384 + piece_type * 64 + square
#[inline]
pub fn white_feature(color: Color, piece: Piece, sq: Square) -> usize {
    let c = if color == Color::White { 0 } else { 1 };
    let p = piece_index(piece);
    c * 384 + p * 64 + sq.to_index()
}

/// Compute feature index for the BLACK perspective (mirror colors + flip ranks).
///
/// black_feature(color, piece_type, square) = (1-color) * 384 + piece_type * 64 + (square ^ 56)
#[inline]
pub fn black_feature(color: Color, piece: Piece, sq: Square) -> usize {
    let c = if color == Color::White { 1 } else { 0 }; // Flipped
    let p = piece_index(piece);
    c * 384 + p * 64 + (sq.to_index() ^ 56) // Flip rank
}

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

/// Extract all active feature indices for both perspectives from a Board.
///
/// Returns (white_features, black_features, count) where count is the number
/// of active features (same for both perspectives).
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

const ALL_PIECES: [Piece; 6] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
];

/// Determine the feature changes when a move is made.
///
/// Returns arrays of (added, removed) feature indices for both perspectives,
/// or None if incremental update is not possible (e.g., castling).
///
/// Each entry: (white_feature_idx, black_feature_idx)
pub struct FeatureDelta {
    pub added: [(usize, usize); 2], // Up to 2 pieces added (promotion + rook in castling)
    pub removed: [(usize, usize); 2], // Up to 2 pieces removed (capture + source)
    pub num_added: usize,
    pub num_removed: usize,
}

/// Compute feature deltas for a move. Returns None for castling (use full refresh).
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
    };

    // Detect castling: king moves more than 1 square
    let src_file = src.to_index() % 8;
    let dst_file = dst.to_index() % 8;
    if moving_piece == Piece::King && ((src_file as i32 - dst_file as i32).abs() > 1) {
        return None; // Use full refresh for castling
    }

    // Remove moving piece from source
    delta.removed[delta.num_removed] = (
        white_feature(stm, moving_piece, src),
        black_feature(stm, moving_piece, src),
    );
    delta.num_removed += 1;

    // Handle capture: remove captured piece
    let captured = board.piece_on(dst);
    if let Some(cap_piece) = captured {
        let cap_color = !stm;
        delta.removed[delta.num_removed] = (
            white_feature(cap_color, cap_piece, dst),
            black_feature(cap_color, cap_piece, dst),
        );
        delta.num_removed += 1;
    }

    // Handle en passant: captured pawn is on a different square
    if moving_piece == Piece::Pawn && captured.is_none() {
        let src_file_idx = src.to_index() % 8;
        let dst_file_idx = dst.to_index() % 8;
        if src_file_idx != dst_file_idx {
            // Diagonal pawn move with no piece on dest = en passant
            let ep_rank = src.to_index() / 8;
            let ep_sq = unsafe { Square::new((ep_rank * 8 + dst_file_idx) as u8) };
            delta.removed[delta.num_removed] = (
                white_feature(!stm, Piece::Pawn, ep_sq),
                black_feature(!stm, Piece::Pawn, ep_sq),
            );
            delta.num_removed += 1;
            // Need 3 removes but we only have 2 slots — use full refresh
            if delta.num_removed > 2 {
                return None;
            }
        }
    }

    // Add piece to destination (possibly promoted)
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
    use std::str::FromStr;

    #[test]
    fn test_feature_indices() {
        // White pawn on e2 (square 12)
        let sq = Square::from_str("e2").unwrap();
        let wf = white_feature(Color::White, Piece::Pawn, sq);
        assert_eq!(wf, 0 * 384 + 0 * 64 + 12); // 12

        // Black perspective: white pawn on e2 → color=1(white→enemy), sq=12^56=52
        let bf = black_feature(Color::White, Piece::Pawn, sq);
        assert_eq!(bf, 1 * 384 + 0 * 64 + 52); // 436
    }

    #[test]
    fn test_extract_starting_position() {
        let board = Board::default();
        let (_, _, count) = extract_features(&board);
        assert_eq!(count, 32); // 32 pieces in starting position
    }
}
