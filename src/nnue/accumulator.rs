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

/// NNUE accumulator: stores the feature transformer output for both perspectives.
///
/// Each perspective has `hidden_size` i16 values representing the sum of
/// active feature columns plus the bias vector.
#[derive(Clone)]
pub struct Accumulator {
    /// White perspective accumulator [hidden_size]
    pub white: Vec<i16>,
    /// Black perspective accumulator [hidden_size]
    pub black: Vec<i16>,
    /// Whether this accumulator has been computed
    pub computed: bool,
}

impl Accumulator {
    /// Create an uninitialized accumulator
    pub fn new(hidden_size: usize) -> Self {
        Self {
            white: vec![0i16; hidden_size],
            black: vec![0i16; hidden_size],
            computed: false,
        }
    }

    /// Full refresh: compute accumulator from scratch by scanning the board.
    ///
    /// acc[i] = bias[i] + sum(ft_weights[active_feature][i]) for all active features
    pub fn refresh(&mut self, board: &Board, net: &NnueNetwork) {
        let hs = net.hidden_size;

        // Start from bias
        self.white[..hs].copy_from_slice(&net.ft_biases[..hs]);
        self.black[..hs].copy_from_slice(&net.ft_biases[..hs]);

        // Extract active features
        let (white_feats, black_feats, count) = features::extract_features(board);

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
        }

        self.computed = true;
    }

    /// Incremental update: apply feature deltas from a move.
    ///
    /// Copies parent accumulator, then adds/subtracts changed feature columns.
    pub fn update_from(&mut self, parent: &Accumulator, delta: &FeatureDelta, net: &NnueNetwork) {
        let hs = net.hidden_size;

        // Copy parent state
        self.white[..hs].copy_from_slice(&parent.white[..hs]);
        self.black[..hs].copy_from_slice(&parent.black[..hs]);

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
                // Zero-copy split borrow: parent immutable, child mutable
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

/// Vectorized dst[i] += src[i] for i16 slices (AVX2 fast path, scalar fallback).
#[inline]
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

/// Vectorized dst[i] -= src[i] for i16 slices (AVX2 fast path, scalar fallback).
#[inline]
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_add_i16_avx2(dst: &mut [i16], src: &[i16]) {
    let n = dst.len();
    let mut i = 0;
    // Process 16 i16 values per iteration (256 bits)
    while i + 16 <= n {
        let a = _mm256_loadu_si256(dst.as_ptr().add(i) as *const __m256i);
        let b = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
        let sum = _mm256_add_epi16(a, b);
        _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, sum);
        i += 16;
    }
    // Scalar remainder
    while i < n {
        dst[i] = dst[i].wrapping_add(src[i]);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_sub_i16_avx2(dst: &mut [i16], src: &[i16]) {
    let n = dst.len();
    let mut i = 0;
    while i + 16 <= n {
        let a = _mm256_loadu_si256(dst.as_ptr().add(i) as *const __m256i);
        let b = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
        let diff = _mm256_sub_epi16(a, b);
        _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, diff);
        i += 16;
    }
    while i < n {
        dst[i] = dst[i].wrapping_sub(src[i]);
        i += 1;
    }
}
