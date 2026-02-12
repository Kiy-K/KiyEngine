// src/nnue/accumulator.rs
//! NNUE Accumulator: the first hidden layer of the feature transformer.
//!
//! Maintains per-perspective (white/black) i16 vectors that are efficiently
//! updatable when pieces move on the board. Supports both full refresh
//! (scan entire board) and incremental updates (add/subtract feature columns).
//!
//! Optimizations:
//! - AVX2 SIMD vectorized add/sub (16 i16 ops per cycle)
//! - Zero-copy split_at_mut for parent→child updates
//! - Fused add-sub operations to reduce loop overhead

use crate::nnue::features::{self, FeatureDelta};
use crate::nnue::NnueNetwork;
use chess::{Board, Color, Piece, Square};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Fixed hidden layer size — must match the NNUE file (asserted at load time).
/// Using a const eliminates heap allocation, bounds checks, and enables aligned SIMD.
pub const HIDDEN_SIZE: usize = 512;

/// Maximum number of output buckets.
const MAX_BUCKETS: usize = 8;

/// NNUE accumulator: stores the feature transformer output for both perspectives.
///
/// Fixed-size, stack-allocated, 32-byte aligned for AVX2.
/// No heap allocation — critical for NPS performance.
#[derive(Clone)]
#[repr(align(64))]
pub struct Accumulator {
    /// White perspective accumulator [HIDDEN_SIZE]
    pub white: [i16; HIDDEN_SIZE],
    /// Black perspective accumulator [HIDDEN_SIZE]
    pub black: [i16; HIDDEN_SIZE],
    /// PSQT scores per bucket for white perspective
    pub psqt_white: [i32; MAX_BUCKETS],
    /// PSQT scores per bucket for black perspective
    pub psqt_black: [i32; MAX_BUCKETS],
    /// Whether this accumulator has been computed
    pub computed: bool,
}

impl Accumulator {
    /// Create an uninitialized accumulator
    #[inline]
    pub fn new(_hidden_size: usize) -> Self {
        Self {
            white: [0i16; HIDDEN_SIZE],
            black: [0i16; HIDDEN_SIZE],
            psqt_white: [0i32; MAX_BUCKETS],
            psqt_black: [0i32; MAX_BUCKETS],
            computed: false,
        }
    }

    /// Full refresh: compute accumulator from scratch by scanning the board.
    ///
    /// Automatically uses king-bucketed features if the network has input buckets > 1.
    pub fn refresh(&mut self, board: &Board, net: &NnueNetwork) {
        let hs = net.hidden_size;
        let nb = net.num_buckets;
        let has_psqt = !net.psqt_weights.is_empty();

        // Start from bias
        self.white[..hs].copy_from_slice(&net.ft_biases[..hs]);
        self.black[..hs].copy_from_slice(&net.ft_biases[..hs]);

        // Initialize PSQT accumulators
        if has_psqt {
            self.psqt_white[..nb].fill(0);
            self.psqt_black[..nb].fill(0);
        }

        // Extract active features (king-bucketed or legacy)
        let (white_feats, black_feats, count) = if net.has_king_buckets() {
            features::extract_features_bucketed(board)
        } else {
            features::extract_features(board)
        };

        // Add each active feature's column to the accumulator
        for idx in 0..count {
            let wf = white_feats[idx];
            let bf = black_feats[idx];
            let w_offset = wf * hs;
            let b_offset = bf * hs;

            vec_add_i16(
                &mut self.white[..hs],
                &net.ft_weights[w_offset..w_offset + hs],
            );
            vec_add_i16(
                &mut self.black[..hs],
                &net.ft_weights[b_offset..b_offset + hs],
            );

            // Accumulate PSQT scores
            if has_psqt {
                let w_psqt_offset = wf * nb;
                let b_psqt_offset = bf * nb;
                for b in 0..nb {
                    self.psqt_white[b] += net.psqt_weights[w_psqt_offset + b] as i32;
                    self.psqt_black[b] += net.psqt_weights[b_psqt_offset + b] as i32;
                }
            }
        }

        self.computed = true;
    }

    /// Incremental update: apply feature deltas from a move.
    ///
    /// Uses fused SIMD operations for the common cases:
    /// - Quiet move (1 rem + 1 add): single-pass copy-sub-add  → 3× fewer memory passes
    /// - Capture   (2 rem + 1 add): single-pass copy-sub2-add → 4× fewer memory passes
    /// Falls back to sequential copy+sub+add for rare edge cases.
    pub fn update_from(&mut self, parent: &Accumulator, delta: &FeatureDelta, net: &NnueNetwork) {
        let hs = net.hidden_size;
        let ftw = &net.ft_weights;

        if delta.num_removed == 1 && delta.num_added == 1 {
            // ── Quiet move: fused copy + sub + add (single pass per perspective) ──
            let (wr, br) = delta.removed[0];
            let (wa, ba) = delta.added[0];
            vec_copy_sub_add_i16(
                &mut self.white[..hs],
                &parent.white[..hs],
                &ftw[wr * hs..wr * hs + hs],
                &ftw[wa * hs..wa * hs + hs],
            );
            vec_copy_sub_add_i16(
                &mut self.black[..hs],
                &parent.black[..hs],
                &ftw[br * hs..br * hs + hs],
                &ftw[ba * hs..ba * hs + hs],
            );
        } else if delta.num_removed == 2 && delta.num_added == 1 {
            // ── Capture: fused copy + sub2 + add (single pass per perspective) ──
            let (wr1, br1) = delta.removed[0];
            let (wr2, br2) = delta.removed[1];
            let (wa, ba) = delta.added[0];
            vec_copy_sub2_add_i16(
                &mut self.white[..hs],
                &parent.white[..hs],
                &ftw[wr1 * hs..wr1 * hs + hs],
                &ftw[wr2 * hs..wr2 * hs + hs],
                &ftw[wa * hs..wa * hs + hs],
            );
            vec_copy_sub2_add_i16(
                &mut self.black[..hs],
                &parent.black[..hs],
                &ftw[br1 * hs..br1 * hs + hs],
                &ftw[br2 * hs..br2 * hs + hs],
                &ftw[ba * hs..ba * hs + hs],
            );
        } else {
            // ── Fallback: sequential copy then add/sub ──
            self.white[..hs].copy_from_slice(&parent.white[..hs]);
            self.black[..hs].copy_from_slice(&parent.black[..hs]);
            for r in 0..delta.num_removed {
                let (wf, bf) = delta.removed[r];
                vec_sub_i16(&mut self.white[..hs], &ftw[wf * hs..wf * hs + hs]);
                vec_sub_i16(&mut self.black[..hs], &ftw[bf * hs..bf * hs + hs]);
            }
            for a in 0..delta.num_added {
                let (wf, bf) = delta.added[a];
                vec_add_i16(&mut self.white[..hs], &ftw[wf * hs..wf * hs + hs]);
                vec_add_i16(&mut self.black[..hs], &ftw[bf * hs..bf * hs + hs]);
            }
        }

        // PSQT incremental update (v3 only, empty for v5 deep networks)
        let nb = net.num_buckets;
        let has_psqt = !net.psqt_weights.is_empty();
        if has_psqt {
            self.psqt_white[..nb].copy_from_slice(&parent.psqt_white[..nb]);
            self.psqt_black[..nb].copy_from_slice(&parent.psqt_black[..nb]);
            for r in 0..delta.num_removed {
                let (wf, bf) = delta.removed[r];
                for b in 0..nb {
                    self.psqt_white[b] -= net.psqt_weights[wf * nb + b] as i32;
                    self.psqt_black[b] -= net.psqt_weights[bf * nb + b] as i32;
                }
            }
            for a in 0..delta.num_added {
                let (wf, bf) = delta.added[a];
                for b in 0..nb {
                    self.psqt_white[b] += net.psqt_weights[wf * nb + b] as i32;
                    self.psqt_black[b] += net.psqt_weights[bf * nb + b] as i32;
                }
            }
        }

        self.computed = true;
    }
}

// ============================================================================
// FINNY TABLE (Accumulator Cache)
// ============================================================================
//
// Idea from Luecx (Koivisto), adopted by Stockfish since SF15.
// Caches accumulator state per (king_square, perspective).
// When a full refresh is needed (e.g. king bucket changed), instead of
// recomputing from all ~32 pieces, diff against the cached state —
// typically only 2-4 pieces have changed, making refreshes ~8× faster.

const NO_PIECE_ENC: u8 = 0;

#[inline]
fn encode_piece(color: Color, piece: Piece) -> u8 {
    let c = if color == Color::White { 0u8 } else { 6u8 };
    c + features::piece_index(piece) as u8 + 1
}

#[inline]
fn decode_piece(encoded: u8) -> Option<(Color, Piece)> {
    if encoded == NO_PIECE_ENC {
        return None;
    }
    let v = (encoded - 1) as usize;
    let color = if v < 6 { Color::White } else { Color::Black };
    Some((color, features::ALL_PIECES[v % 6]))
}

/// Encode the board's piece layout into a compact [u8; 64] array.
fn encode_board(board: &Board) -> ([u8; 64], u64) {
    let mut pieces = [NO_PIECE_ENC; 64];
    let mut bb = 0u64;
    for &color in &[Color::White, Color::Black] {
        let color_bb = board.color_combined(color);
        for &piece in &features::ALL_PIECES {
            let piece_bb = board.pieces(piece) & color_bb;
            for sq in piece_bb {
                let idx = sq.to_index();
                pieces[idx] = encode_piece(color, piece);
                bb |= 1u64 << idx;
            }
        }
    }
    (pieces, bb)
}

/// Find squares where pieces differ between cached and current state.
#[inline]
fn get_changed_squares(old: &[u8; 64], new: &[u8; 64]) -> u64 {
    let mut changed = 0u64;
    for i in 0..64 {
        if old[i] != new[i] {
            changed |= 1u64 << i;
        }
    }
    changed
}

/// Single Finny Table entry: cached accumulator for one (king_sq, perspective).
#[repr(align(64))]
pub struct FinnyEntry {
    accumulation: [i16; HIDDEN_SIZE],
    psqt: [i32; MAX_BUCKETS],
    pieces: [u8; 64],
    piece_bb: u64,
}

impl FinnyEntry {
    fn new(biases: &[i16]) -> Self {
        let mut entry = Self {
            accumulation: [0i16; HIDDEN_SIZE],
            psqt: [0i32; MAX_BUCKETS],
            pieces: [NO_PIECE_ENC; 64],
            piece_bb: 0,
        };
        let hs = biases.len().min(HIDDEN_SIZE);
        entry.accumulation[..hs].copy_from_slice(&biases[..hs]);
        entry
    }
}

/// Finny Table: per-(king_square, perspective) accumulator cache.
/// 64 king squares × 2 perspectives = 128 entries (~144 KB).
///
/// Each entry caches the accumulator state from the last time a position
/// with this king square was evaluated. On refresh, we diff the cached
/// piece layout against the current board and apply only the changes.
pub struct FinnyTable {
    entries: Box<[[FinnyEntry; 2]; 64]>,
}

impl FinnyTable {
    /// Create a new Finny Table initialized with biases (empty board state).
    pub fn new(net: &NnueNetwork) -> Self {
        let biases = &net.ft_biases;
        Self {
            entries: Box::new(std::array::from_fn(|_| {
                [FinnyEntry::new(biases), FinnyEntry::new(biases)]
            })),
        }
    }

    /// Refresh a single perspective's accumulator using the cached entry.
    ///
    /// Instead of recomputing from all pieces (bias + sum of 32 weight columns),
    /// we diff against the cached state and apply only the changes.
    fn refresh_perspective(
        &mut self,
        acc_out: &mut [i16],
        psqt_out: &mut [i32],
        board: &Board,
        king_sq: usize,
        perspective: usize, // 0 = white, 1 = black
        net: &NnueNetwork,
    ) {
        let hs = net.hidden_size;
        let nb = net.num_buckets;
        let has_psqt = !net.psqt_weights.is_empty();
        let king_bucketed = net.has_king_buckets();

        let entry = &mut self.entries[king_sq][perspective];

        // Encode current board state
        let (current_pieces, current_bb) = encode_board(board);

        // Find squares that changed since the cache was last updated
        let changed = get_changed_squares(&entry.pieces, &current_pieces);
        let removed_bb = changed & entry.piece_bb;
        let added_bb = changed & current_bb;

        // Remove old features from cached accumulation
        let mut iter = removed_bb;
        while iter != 0 {
            let sq_idx = iter.trailing_zeros() as usize;
            iter &= iter - 1;

            if let Some((color, piece)) = decode_piece(entry.pieces[sq_idx]) {
                let sq = unsafe { Square::new(sq_idx as u8) };
                let feat = if king_bucketed {
                    if perspective == 0 {
                        features::white_bucketed_feature(king_sq, color, piece, sq)
                    } else {
                        features::black_bucketed_feature(king_sq, color, piece, sq)
                    }
                } else {
                    if perspective == 0 {
                        features::white_feature(color, piece, sq)
                    } else {
                        features::black_feature(color, piece, sq)
                    }
                };

                let offset = feat * hs;
                vec_sub_i16(
                    &mut entry.accumulation[..hs],
                    &net.ft_weights[offset..offset + hs],
                );
                if has_psqt {
                    let psqt_offset = feat * nb;
                    for b in 0..nb {
                        entry.psqt[b] -= net.psqt_weights[psqt_offset + b] as i32;
                    }
                }
            }
        }

        // Add new features to cached accumulation
        iter = added_bb;
        while iter != 0 {
            let sq_idx = iter.trailing_zeros() as usize;
            iter &= iter - 1;

            if let Some((color, piece)) = decode_piece(current_pieces[sq_idx]) {
                let sq = unsafe { Square::new(sq_idx as u8) };
                let feat = if king_bucketed {
                    if perspective == 0 {
                        features::white_bucketed_feature(king_sq, color, piece, sq)
                    } else {
                        features::black_bucketed_feature(king_sq, color, piece, sq)
                    }
                } else {
                    if perspective == 0 {
                        features::white_feature(color, piece, sq)
                    } else {
                        features::black_feature(color, piece, sq)
                    }
                };

                let offset = feat * hs;
                vec_add_i16(
                    &mut entry.accumulation[..hs],
                    &net.ft_weights[offset..offset + hs],
                );
                if has_psqt {
                    let psqt_offset = feat * nb;
                    for b in 0..nb {
                        entry.psqt[b] += net.psqt_weights[psqt_offset + b] as i32;
                    }
                }
            }
        }

        // Update cache with current board state
        entry.pieces = current_pieces;
        entry.piece_bb = current_bb;

        // Copy cached result to output accumulator
        acc_out[..hs].copy_from_slice(&entry.accumulation[..hs]);
        if has_psqt {
            psqt_out[..nb].copy_from_slice(&entry.psqt[..nb]);
        }
    }

    /// Refresh both perspectives of an accumulator using Finny Table caches.
    pub fn refresh(&mut self, acc: &mut Accumulator, board: &Board, net: &NnueNetwork) {
        let wk_sq = board.king_square(Color::White).to_index();
        let bk_sq = board.king_square(Color::Black).to_index();

        self.refresh_perspective(&mut acc.white, &mut acc.psqt_white, board, wk_sq, 0, net);
        self.refresh_perspective(&mut acc.black, &mut acc.psqt_black, board, bk_sq, 1, net);

        acc.computed = true;
    }
}

/// Stack of accumulators indexed by ply for use during search.
///
/// At each ply, the accumulator represents the position after all moves
/// up to that ply have been made. Supports push/pop semantics via ply index.
pub struct AccumulatorStack {
    stack: Vec<Accumulator>,
}

impl AccumulatorStack {
    /// Create a stack with capacity for max_ply levels
    pub fn new(max_ply: usize, hidden_size: usize) -> Self {
        let stack = (0..max_ply)
            .map(|_| Accumulator::new(hidden_size))
            .collect();
        Self { stack }
    }

    /// Get accumulator at given ply (immutable)
    #[inline]
    pub fn get(&self, ply: usize) -> &Accumulator {
        &self.stack[ply]
    }

    /// Get accumulator at given ply (mutable)
    #[inline]
    pub fn get_mut(&mut self, ply: usize) -> &mut Accumulator {
        &mut self.stack[ply]
    }

    /// Refresh accumulator at ply from the board
    pub fn refresh_at(&mut self, ply: usize, board: &Board, net: &NnueNetwork) {
        self.stack[ply].refresh(board, net);
    }

    /// Incrementally update accumulator at child_ply from parent_ply using delta.
    /// Falls back to full refresh if delta is None.
    ///
    /// For king-bucketed networks: if the king moved for a perspective, that
    /// perspective's accumulator is fully refreshed (bucket changed), while the
    /// other perspective is incrementally updated.
    ///
    /// Uses split_at_mut for zero-copy parent→child update (no Vec clone).
    pub fn update_or_refresh(
        &mut self,
        child_ply: usize,
        parent_ply: usize,
        delta: Option<&FeatureDelta>,
        board: &Board,
        net: &NnueNetwork,
    ) {
        if let Some(d) = delta {
            if parent_ply < child_ply && self.stack[parent_ply].computed {
                let king_bucketed = net.has_king_buckets();
                let need_white_refresh = king_bucketed && d.white_king_moved;
                let need_black_refresh = king_bucketed && d.black_king_moved;

                if need_white_refresh || need_black_refresh {
                    // King moved in a king-bucketed network: full refresh
                    // (king bucket changed, ALL features must be recomputed for that perspective)
                    self.stack[child_ply].refresh(board, net);
                    return;
                }

                // Normal incremental update (no king move or non-bucketed network)
                let (left, right) = self.stack.split_at_mut(child_ply);
                let parent = &left[parent_ply];
                let child = &mut right[0];
                child.update_from(parent, d, net);
                return;
            }
        }
        // Fallback: full refresh
        self.stack[child_ply].refresh(board, net);
    }

    /// Mark accumulator at ply as invalid
    pub fn invalidate(&mut self, ply: usize) {
        self.stack[ply].computed = false;
    }
}

// ============================================================================
// SIMD-VECTORIZED i16 ADD/SUB
// ============================================================================

/// Vectorized dst[i] += src[i] for i16 slices.
/// Uses AVX2 when available, scalar fallback otherwise.
/// dst must be from an aligned Accumulator; src (ft_weights) is unaligned.
#[inline(always)]
fn vec_add_i16(dst: &mut [i16], src: &[i16]) {
    debug_assert_eq!(dst.len(), src.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                return vec_add_i16_avx2(dst, src);
            }
        }
    }

    for i in 0..dst.len() {
        dst[i] = dst[i].wrapping_add(src[i]);
    }
}

/// Vectorized dst[i] -= src[i] for i16 slices.
#[inline(always)]
fn vec_sub_i16(dst: &mut [i16], src: &[i16]) {
    debug_assert_eq!(dst.len(), src.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                return vec_sub_i16_avx2(dst, src);
            }
        }
    }

    for i in 0..dst.len() {
        dst[i] = dst[i].wrapping_sub(src[i]);
    }
}

/// Fused: dst[i] = src[i] - sub[i] + add[i]  (quiet move: 3× fewer memory passes)
#[inline(always)]
fn vec_copy_sub_add_i16(dst: &mut [i16], src: &[i16], sub: &[i16], add: &[i16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                return vec_copy_sub_add_i16_avx2(dst, src, sub, add);
            }
        }
    }
    for i in 0..dst.len() {
        dst[i] = src[i].wrapping_sub(sub[i]).wrapping_add(add[i]);
    }
}

/// Fused: dst[i] = src[i] - sub1[i] - sub2[i] + add[i]  (capture: 4× fewer passes)
#[inline(always)]
fn vec_copy_sub2_add_i16(dst: &mut [i16], src: &[i16], sub1: &[i16], sub2: &[i16], add: &[i16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                return vec_copy_sub2_add_i16_avx2(dst, src, sub1, sub2, add);
            }
        }
    }
    for i in 0..dst.len() {
        dst[i] = src[i]
            .wrapping_sub(sub1[i])
            .wrapping_sub(sub2[i])
            .wrapping_add(add[i]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn vec_add_i16_avx2(dst: &mut [i16], src: &[i16]) {
    let n = dst.len();
    let dst_ptr = dst.as_mut_ptr();
    let src_ptr = src.as_ptr();
    let mut i = 0;
    while i + 16 <= n {
        let a = _mm256_load_si256(dst_ptr.add(i) as *const __m256i);
        let b = _mm256_loadu_si256(src_ptr.add(i) as *const __m256i);
        _mm256_store_si256(dst_ptr.add(i) as *mut __m256i, _mm256_add_epi16(a, b));
        i += 16;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn vec_sub_i16_avx2(dst: &mut [i16], src: &[i16]) {
    let n = dst.len();
    let dst_ptr = dst.as_mut_ptr();
    let src_ptr = src.as_ptr();
    let mut i = 0;
    while i + 16 <= n {
        let a = _mm256_load_si256(dst_ptr.add(i) as *const __m256i);
        let b = _mm256_loadu_si256(src_ptr.add(i) as *const __m256i);
        _mm256_store_si256(dst_ptr.add(i) as *mut __m256i, _mm256_sub_epi16(a, b));
        i += 16;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn vec_copy_sub_add_i16_avx2(dst: &mut [i16], src: &[i16], sub: &[i16], add: &[i16]) {
    let n = dst.len();
    let (dp, sp, subp, addp) = (dst.as_mut_ptr(), src.as_ptr(), sub.as_ptr(), add.as_ptr());
    let mut i = 0;
    while i + 16 <= n {
        let s = _mm256_load_si256(sp.add(i) as *const __m256i);
        let r = _mm256_loadu_si256(subp.add(i) as *const __m256i);
        let a = _mm256_loadu_si256(addp.add(i) as *const __m256i);
        _mm256_store_si256(
            dp.add(i) as *mut __m256i,
            _mm256_add_epi16(_mm256_sub_epi16(s, r), a),
        );
        i += 16;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn vec_copy_sub2_add_i16_avx2(
    dst: &mut [i16],
    src: &[i16],
    sub1: &[i16],
    sub2: &[i16],
    add: &[i16],
) {
    let n = dst.len();
    let (dp, sp) = (dst.as_mut_ptr(), src.as_ptr());
    let (s1p, s2p, ap) = (sub1.as_ptr(), sub2.as_ptr(), add.as_ptr());
    let mut i = 0;
    while i + 16 <= n {
        let s = _mm256_load_si256(sp.add(i) as *const __m256i);
        let r1 = _mm256_loadu_si256(s1p.add(i) as *const __m256i);
        let r2 = _mm256_loadu_si256(s2p.add(i) as *const __m256i);
        let a = _mm256_loadu_si256(ap.add(i) as *const __m256i);
        _mm256_store_si256(
            dp.add(i) as *mut __m256i,
            _mm256_add_epi16(_mm256_sub_epi16(_mm256_sub_epi16(s, r1), r2), a),
        );
        i += 16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    /// Create a small random-ish NNUE network for testing Finny Tables.
    fn make_test_net() -> NnueNetwork {
        let hs = 8; // small hidden size for testing
        let nb = 1; // 1 output bucket
        let num_input_buckets = 1; // no king bucketing for simplicity
        let total_features = num_input_buckets * features::NUM_FEATURES;

        // Deterministic "random" weights: feature_idx * 3 + i
        let ft_weights: Vec<i16> = (0..total_features * hs)
            .map(|i| ((i as i32 * 7 + 13) % 200 - 100) as i16)
            .collect();
        let ft_biases: Vec<i16> = (0..hs).map(|i| (i as i16 + 1) * 10).collect();
        let output_weights: Vec<i16> = vec![1; nb * 2 * hs];
        let output_biases: Vec<i16> = vec![0; nb];
        let psqt_weights: Vec<i16> = vec![0; total_features * nb];

        NnueNetwork {
            hidden_size: hs,
            l1_size: 0,
            num_buckets: nb,
            num_input_buckets,
            ft_weights,
            ft_biases,
            output_weights,
            output_biases,
            l1_weights: Vec::new(),
            l1_biases: Vec::new(),
            l2_weights: Vec::new(),
            l2_biases: Vec::new(),
            psqt_weights,
        }
    }

    #[test]
    fn test_finny_table_matches_full_refresh() {
        let net = make_test_net();
        let board = Board::default();

        // Full refresh
        let mut acc_full = Accumulator::new(net.hidden_size);
        acc_full.refresh(&board, &net);

        // Finny Table refresh (first call — cache is empty, so it adds all pieces)
        let mut finny = FinnyTable::new(&net);
        let mut acc_finny = Accumulator::new(net.hidden_size);
        finny.refresh(&mut acc_finny, &board, &net);

        // Both should produce identical results
        assert_eq!(
            &acc_full.white[..net.hidden_size],
            &acc_finny.white[..net.hidden_size],
            "White perspective mismatch on initial refresh"
        );
        assert_eq!(
            &acc_full.black[..net.hidden_size],
            &acc_finny.black[..net.hidden_size],
            "Black perspective mismatch on initial refresh"
        );
        assert!(acc_finny.computed);
    }

    #[test]
    fn test_finny_table_after_move() {
        let net = make_test_net();

        // Start position
        let board1 = Board::default();
        let mut finny = FinnyTable::new(&net);
        let mut acc = Accumulator::new(net.hidden_size);
        finny.refresh(&mut acc, &board1, &net);

        // Make a move: e2e4
        let board2 =
            Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1").unwrap();

        // Full refresh on new position
        let mut acc_full = Accumulator::new(net.hidden_size);
        acc_full.refresh(&board2, &net);

        // Finny Table refresh on new position (should diff against cached start pos)
        let mut acc_finny = Accumulator::new(net.hidden_size);
        finny.refresh(&mut acc_finny, &board2, &net);

        assert_eq!(
            &acc_full.white[..net.hidden_size],
            &acc_finny.white[..net.hidden_size],
            "White perspective mismatch after move"
        );
        assert_eq!(
            &acc_full.black[..net.hidden_size],
            &acc_finny.black[..net.hidden_size],
            "Black perspective mismatch after move"
        );
    }

    #[test]
    fn test_finny_table_multiple_positions() {
        let net = make_test_net();
        let mut finny = FinnyTable::new(&net);

        let positions = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        ];

        for fen in &positions {
            let board = Board::from_str(fen).unwrap();

            let mut acc_full = Accumulator::new(net.hidden_size);
            acc_full.refresh(&board, &net);

            let mut acc_finny = Accumulator::new(net.hidden_size);
            finny.refresh(&mut acc_finny, &board, &net);

            assert_eq!(
                &acc_full.white[..net.hidden_size],
                &acc_finny.white[..net.hidden_size],
                "White mismatch at FEN: {}",
                fen
            );
            assert_eq!(
                &acc_full.black[..net.hidden_size],
                &acc_finny.black[..net.hidden_size],
                "Black mismatch at FEN: {}",
                fen
            );
        }
    }
}
