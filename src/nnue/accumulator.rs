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
use chess::Board;

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
    /// Copies parent accumulator, then adds/subtracts changed feature columns.
    /// Also incrementally updates PSQT shortcut scores.
    pub fn update_from(&mut self, parent: &Accumulator, delta: &FeatureDelta, net: &NnueNetwork) {
        let hs = net.hidden_size;
        let nb = net.num_buckets;
        let has_psqt = !net.psqt_weights.is_empty();

        // Copy parent state
        self.white[..hs].copy_from_slice(&parent.white[..hs]);
        self.black[..hs].copy_from_slice(&parent.black[..hs]);

        // Copy PSQT state
        if has_psqt {
            self.psqt_white[..nb].copy_from_slice(&parent.psqt_white[..nb]);
            self.psqt_black[..nb].copy_from_slice(&parent.psqt_black[..nb]);
        }

        // Subtract removed features
        for r in 0..delta.num_removed {
            let (wf, bf) = delta.removed[r];
            let w_offset = wf * hs;
            let b_offset = bf * hs;
            vec_sub_i16(
                &mut self.white[..hs],
                &net.ft_weights[w_offset..w_offset + hs],
            );
            vec_sub_i16(
                &mut self.black[..hs],
                &net.ft_weights[b_offset..b_offset + hs],
            );
            if has_psqt {
                let w_psqt = wf * nb;
                let b_psqt = bf * nb;
                for b in 0..nb {
                    self.psqt_white[b] -= net.psqt_weights[w_psqt + b] as i32;
                    self.psqt_black[b] -= net.psqt_weights[b_psqt + b] as i32;
                }
            }
        }

        // Add new features
        for a in 0..delta.num_added {
            let (wf, bf) = delta.added[a];
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
            if has_psqt {
                let w_psqt = wf * nb;
                let b_psqt = bf * nb;
                for b in 0..nb {
                    self.psqt_white[b] += net.psqt_weights[w_psqt + b] as i32;
                    self.psqt_black[b] += net.psqt_weights[b_psqt + b] as i32;
                }
            }
        }

        self.computed = true;
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
            unsafe { return vec_add_i16_avx2(dst, src); }
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
            unsafe { return vec_sub_i16_avx2(dst, src); }
        }
    }

    for i in 0..dst.len() {
        dst[i] = dst[i].wrapping_sub(src[i]);
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
