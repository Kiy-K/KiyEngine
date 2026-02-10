// src/engine/bitlinear.rs
//! BitNet 1.58-bit linear layer with bit-packed ternary weights
//!
//! v5.2.0: Ternary weights {-1, 0, +1} stored as F16 in GGUF, packed to dual bitmasks
//! (pos_bits + neg_bits) at load time for 4× memory reduction over i8.
//! Uses SIMD-optimized packed GEMV (AVX-512/AVX2/NEON) for single-token inference.
//! Sequences use F32 matmul with cached transposed weights (candle's tiled GEMM).

use candle_core::{Device, Result, Tensor, D};
use candle_nn::VarBuilder;
use parking_lot::RwLock;

/// RMS normalization
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x_sq = x.sqr()?;
    let mean_sq = x_sq.mean_keepdim(D::Minus1)?;
    let norm = (mean_sq + eps)?.sqrt()?;
    let x_normed = x.broadcast_div(&norm)?;
    x_normed.broadcast_mul(weight)
}

/// RMS normalization on raw f32 slices — zero tensor allocation for single-token SIMD path
#[inline]
pub(crate) fn rms_norm_vec(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / ((sum_sq / n + eps).sqrt());
    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| xi * inv_rms * wi)
        .collect()
}

/// Pack F16 ternary values {-1.0, 0.0, 1.0} into i8 {-1, 0, 1}
pub fn pack_f16_to_ternary_i8(raw: &[f32]) -> Vec<i8> {
    raw.iter()
        .map(|&v| {
            if v > 0.5 {
                1i8
            } else if v < -0.5 {
                -1i8
            } else {
                0i8
            }
        })
        .collect()
}

/// Pack i8 ternary weights into dual bitmasks for SIMD-efficient dot products.
/// Returns (pos_bits, neg_bits, row_bytes) where:
///   pos_bits[row * row_bytes + j/8] bit (j%8) = 1 if weight[row][j] == +1
///   neg_bits[row * row_bytes + j/8] bit (j%8) = 1 if weight[row][j] == -1
pub fn pack_ternary_bits(
    weight_i8: &[i8],
    out_dim: usize,
    in_dim: usize,
) -> (Vec<u8>, Vec<u8>, usize) {
    let row_bytes = (in_dim + 7) / 8;
    let total_bytes = out_dim * row_bytes;
    let mut pos_bits = vec![0u8; total_bytes];
    let mut neg_bits = vec![0u8; total_bytes];

    for row in 0..out_dim {
        let row_offset = row * row_bytes;
        let w_offset = row * in_dim;
        for col in 0..in_dim {
            let w = weight_i8[w_offset + col];
            let byte_idx = row_offset + col / 8;
            let bit_pos = col % 8;
            if w == 1 {
                pos_bits[byte_idx] |= 1 << bit_pos;
            } else if w == -1 {
                neg_bits[byte_idx] |= 1 << bit_pos;
            }
        }
    }

    (pos_bits, neg_bits, row_bytes)
}

/// BitNet ternary linear layer with bit-packed quantization
///
/// Implements BitNet 1.58-bit: weights stored as {-1, 0, +1}.
/// Dual bitmask packing: pos_bits (weight==+1) and neg_bits (weight==-1).
/// Each weight uses 2 bits across the two masks — 4x denser than i8.
/// Forward: y = RMSNorm(x) @ W^T (ternary GEMM with SIMD bit expansion)
pub struct BitLinear {
    /// Quantized weights as i8: -1, 0, or +1 (kept for F32 dequantization cache)
    pub weight_i8: Vec<i8>,
    /// Packed positive bitmask: bit j in byte (row*row_bytes + j/8) = 1 if weight[row][j] == +1
    pub pos_bits: Vec<u8>,
    /// Packed negative bitmask: bit j in byte (row*row_bytes + j/8) = 1 if weight[row][j] == -1
    pub neg_bits: Vec<u8>,
    /// Bytes per row in packed representation: ceil(in_dim / 8)
    pub row_bytes: usize,
    /// Scale factor (1.0 for pure ternary, kept for backward compat)
    pub scale: f32,
    pub in_dim: usize,
    pub out_dim: usize,
    pub rms_norm_weight: Tensor,
    /// Cached f32 norm weights for zero-tensor single-token SIMD path
    pub(crate) rms_norm_weight_f32: Vec<f32>,
    pub eps: f64,
    /// Cached dequantized weights as F32 transposed [in_dim, out_dim] (lazy init, thread-safe)
    dequantized_cache_t: RwLock<Option<Tensor>>,
}

impl BitLinear {
    /// Load BitLinear from VarBuilder using explicit weight and norm key names.
    ///
    /// v5.2.0: Ternary weights stored as F16 in GGUF are packed to i8 at load time.
    /// The norm weight key is the RMSNorm weight inside the BitLinear layer.
    pub fn load_from_vb(
        weight_key: &str,
        norm_key: &str,
        vb: &VarBuilder,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Self> {
        // Load weight tensor (F16 in GGUF, converted to f32 by loader)
        let weight = vb.get((out_dim, in_dim), weight_key)?;
        let raw = weight.flatten_all()?.to_vec1::<f32>()?;
        let weight_i8 = pack_f16_to_ternary_i8(&raw);

        // Load RMS norm weight
        let rms_norm_weight = vb.get((in_dim,), norm_key)?;
        let rms_norm_weight_f32 = rms_norm_weight.flatten_all()?.to_vec1::<f32>()?;

        // Pack ternary weights into dual bitmasks (4x memory reduction)
        let (pos_bits, neg_bits, row_bytes) = pack_ternary_bits(&weight_i8, out_dim, in_dim);

        Ok(Self {
            weight_i8,
            pos_bits,
            neg_bits,
            row_bytes,
            scale: 1.0,
            in_dim,
            out_dim,
            rms_norm_weight,
            rms_norm_weight_f32,
            eps: 1e-5,
            dequantized_cache_t: RwLock::new(None),
        })
    }

    /// Get dequantized weights as F32 transposed [in_dim, out_dim], cached.
    /// On CPU, F32 is faster than F16 (no native F16 compute on x86).
    pub fn get_weights_t(&self, device: &Device) -> Result<Tensor> {
        {
            let cache = self.dequantized_cache_t.read();
            if let Some(ref tensor) = *cache {
                return Ok(tensor.clone());
            }
        }

        let mut cache = self.dequantized_cache_t.write();
        if let Some(ref tensor) = *cache {
            return Ok(tensor.clone());
        }

        // Build F32 weight matrix [out_dim, in_dim], transpose to [in_dim, out_dim], make contiguous
        let weights_f32: Vec<f32> = self
            .weight_i8
            .iter()
            .map(|&w| w as f32 * self.scale)
            .collect();
        let w = Tensor::from_vec(weights_f32, (self.out_dim, self.in_dim), device)?;
        let w_t = w.t()?.contiguous()?;
        *cache = Some(w_t.clone());
        Ok(w_t)
    }

    /// Force dequantization cache refresh (e.g., after device change)
    pub fn clear_cache(&self) {
        let mut cache = self.dequantized_cache_t.write();
        *cache = None;
    }

    /// Fused RMSNorm + matmul: computes RMSNorm(x) @ W^T in one step.
    /// Avoids allocating the intermediate normalized tensor (saves 5 tensor ops).
    fn fused_rms_norm_matmul(&self, x: &Tensor, w_t: &Tensor) -> Result<Tensor> {
        let x_sq = x.sqr()?;
        let mean_sq = x_sq.mean_keepdim(D::Minus1)?;
        let rms = (mean_sq + self.eps)?.sqrt()?;
        let x_unit = x.broadcast_div(&rms)?;
        let x_normed = x_unit.broadcast_mul(&self.rms_norm_weight)?;
        x_normed.broadcast_matmul(w_t)
    }

    /// Dispatch packed GEMV for a single input row.
    /// Uses runtime ISA auto-selection: AVX-512 > AVX2 > NEON > scalar.
    pub(crate) fn dispatch_packed_gemv(&self, input: &[f32], output: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                unsafe {
                    self.packed_gemv_avx512(input, output);
                }
                return;
            }
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    self.packed_gemv_avx2(input, output);
                }
                return;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                unsafe {
                    self.packed_gemv_neon(input, output);
                }
                return;
            }
        }
        self.packed_gemv_scalar(input, output);
    }

    /// Forward pass: y = RMSNorm(x) @ W^T
    ///
    /// Auto-dispatches: single token → SIMD packed GEMV (bit-packed ternary),
    /// sequences → F32 matmul with cached transposed weights (candle's tiled GEMM).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.dims();
        // Single-token auto-dispatch: SIMD packed GEMV is faster than F32 GEMM
        // because ternary bits are 4x denser (better cache) and use add/sub (no multiply)
        let is_single = match shape.len() {
            2 => shape[0] == 1,
            3 => shape[0] == 1 && shape[1] == 1,
            _ => false,
        };
        if is_single {
            return self.forward_single(x);
        }
        let device = x.device();
        let w_t = self.get_weights_t(device)?;
        self.fused_rms_norm_matmul(x, &w_t)
    }

    /// Optimized forward for single token (batch_size=1, seq_len=1)
    /// Zero-tensor hot path: raw f32 RMSNorm + SIMD packed GEMV.
    /// Bit-packed weights are 4x denser than i8, greatly improving cache utilization.
    pub fn forward_single(&self, x: &Tensor) -> Result<Tensor> {
        let device = x.device();
        let orig_rank = x.dims().len();

        // Single copy: tensor → f32 slice
        let x_f32: Vec<f32> = x.flatten_all()?.to_vec1::<f32>()?;

        // RMS norm on raw f32 — zero tensor allocation
        let x_normed = rms_norm_vec(&x_f32, &self.rms_norm_weight_f32, self.eps as f32);

        // SIMD packed GEMV (AVX-512/AVX2/NEON)
        let mut output = vec![0.0f32; self.out_dim];
        self.dispatch_packed_gemv(&x_normed, &mut output);

        // Single copy back: f32 slice → tensor (preserve input rank)
        let out_shape = if orig_rank <= 2 {
            vec![1, self.out_dim]
        } else {
            vec![1, 1, self.out_dim]
        };
        Tensor::from_vec(output, out_shape, device)
    }

    /// Scalar packed GEMV: extract bits and add/subtract input values
    fn packed_gemv_scalar(&self, input: &[f32], output: &mut [f32]) {
        for i in 0..self.out_dim {
            let row_offset = i * self.row_bytes;
            let mut sum = 0.0f32;

            for byte_idx in 0..self.row_bytes {
                let pb = self.pos_bits[row_offset + byte_idx];
                let nb = self.neg_bits[row_offset + byte_idx];
                let base = byte_idx * 8;

                // Process up to 8 weights per byte
                let remaining = self.in_dim.saturating_sub(base).min(8);
                for bit in 0..remaining {
                    let j = base + bit;
                    let mask = 1u8 << bit;
                    if pb & mask != 0 {
                        sum += unsafe { *input.get_unchecked(j) };
                    } else if nb & mask != 0 {
                        sum -= unsafe { *input.get_unchecked(j) };
                    }
                }
            }

            output[i] = sum * self.scale;
        }
    }

    /// AVX2 packed GEMV: expand 1 byte → 8 float masks using bit tricks.
    /// Each byte of pos/neg_bits controls 8 float lanes with just 3 integer ops.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn packed_gemv_avx2(&self, input: &[f32], output: &mut [f32]) {
        use std::arch::x86_64::*;

        // Bit selection constant: [1, 2, 4, 8, 16, 32, 64, 128] for expanding byte→8 lanes
        let bit_select = _mm256_set_epi32(128, 64, 32, 16, 8, 4, 2, 1);

        for i in 0..self.out_dim {
            let row_offset = i * self.row_bytes;
            let mut sum_vec = _mm256_setzero_ps();

            let mut j = 0usize;
            let mut byte_idx = 0usize;

            // Main loop: process 8 floats per iteration (1 byte of pos/neg bits)
            while j + 8 <= self.in_dim {
                let x_vec = _mm256_loadu_ps(input.as_ptr().add(j));

                // Load pos byte, broadcast to all 8 lanes, AND with bit_select, compare
                let pb = *self.pos_bits.get_unchecked(row_offset + byte_idx) as i32;
                let pos_broadcast = _mm256_set1_epi32(pb);
                let pos_anded = _mm256_and_si256(pos_broadcast, bit_select);
                let pos_mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(pos_anded, bit_select));

                // Same for neg byte
                let nb = *self.neg_bits.get_unchecked(row_offset + byte_idx) as i32;
                let neg_broadcast = _mm256_set1_epi32(nb);
                let neg_anded = _mm256_and_si256(neg_broadcast, bit_select);
                let neg_mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(neg_anded, bit_select));

                // Masked add/sub: only accumulate where bits are set
                let pos_vals = _mm256_and_ps(x_vec, pos_mask);
                let neg_vals = _mm256_and_ps(x_vec, neg_mask);
                sum_vec = _mm256_add_ps(sum_vec, pos_vals);
                sum_vec = _mm256_sub_ps(sum_vec, neg_vals);

                j += 8;
                byte_idx += 1;
            }

            // Horizontal sum
            let mut arr = [0.0f32; 8];
            _mm256_storeu_ps(arr.as_mut_ptr(), sum_vec);
            let mut sum: f32 = arr.iter().sum();

            // Tail: remaining < 8 elements
            while j < self.in_dim {
                let bi = row_offset + j / 8;
                let bp = j % 8;
                let mask = 1u8 << bp;
                if *self.pos_bits.get_unchecked(bi) & mask != 0 {
                    sum += *input.get_unchecked(j);
                } else if *self.neg_bits.get_unchecked(bi) & mask != 0 {
                    sum -= *input.get_unchecked(j);
                }
                j += 1;
            }

            *output.get_unchecked_mut(i) = sum * self.scale;
        }
    }

    /// AVX-512 packed GEMV: uses native 16-bit mask registers.
    /// Load 2 bytes → directly use as __mmask16 → masked add/sub.
    /// This is the optimal path: zero mask expansion, just load bits and go.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn packed_gemv_avx512(&self, input: &[f32], output: &mut [f32]) {
        use std::arch::x86_64::*;

        for i in 0..self.out_dim {
            let row_offset = i * self.row_bytes;
            let mut sum_vec = _mm512_setzero_ps();

            let mut j = 0usize;
            let mut byte_idx = 0usize;

            // Main loop: process 16 floats per iteration (2 bytes of pos/neg bits)
            while j + 16 <= self.in_dim {
                let x_vec = _mm512_loadu_ps(input.as_ptr().add(j));

                // Load 2 bytes → 16-bit mask (native AVX512 mask register!)
                let pos_lo = *self.pos_bits.get_unchecked(row_offset + byte_idx) as u16;
                let pos_hi = *self.pos_bits.get_unchecked(row_offset + byte_idx + 1) as u16;
                let pos_mask: __mmask16 = pos_lo | (pos_hi << 8);

                let neg_lo = *self.neg_bits.get_unchecked(row_offset + byte_idx) as u16;
                let neg_hi = *self.neg_bits.get_unchecked(row_offset + byte_idx + 1) as u16;
                let neg_mask: __mmask16 = neg_lo | (neg_hi << 8);

                // Native masked operations — the bits ARE the mask, zero expansion needed
                sum_vec = _mm512_mask_add_ps(sum_vec, pos_mask, sum_vec, x_vec);
                sum_vec = _mm512_mask_sub_ps(sum_vec, neg_mask, sum_vec, x_vec);

                j += 16;
                byte_idx += 2;
            }

            // Handle remaining 8 (if in_dim not divisible by 16)
            if j + 8 <= self.in_dim {
                // Use the lower 8 lanes of a zmm register or fall through to scalar
                let pb = *self.pos_bits.get_unchecked(row_offset + byte_idx) as u16;
                let nb = *self.neg_bits.get_unchecked(row_offset + byte_idx) as u16;
                // Load 8 floats into the low half, upper 8 are zero
                let x_partial = _mm256_loadu_ps(input.as_ptr().add(j));
                let x_vec = _mm512_castps256_ps512(x_partial);
                // Mask only applies to low 8 lanes (upper 8 bits of mask are 0)
                sum_vec = _mm512_mask_add_ps(sum_vec, pb, sum_vec, x_vec);
                sum_vec = _mm512_mask_sub_ps(sum_vec, nb, sum_vec, x_vec);
                j += 8;
            }

            let mut sum = _mm512_reduce_add_ps(sum_vec);

            // Tail: remaining < 8 elements
            while j < self.in_dim {
                let bi = row_offset + j / 8;
                let bp = j % 8;
                let mask = 1u8 << bp;
                if *self.pos_bits.get_unchecked(bi) & mask != 0 {
                    sum += *input.get_unchecked(j);
                } else if *self.neg_bits.get_unchecked(bi) & mask != 0 {
                    sum -= *input.get_unchecked(j);
                }
                j += 1;
            }

            *output.get_unchecked_mut(i) = sum * self.scale;
        }
    }

    /// ARM NEON packed GEMV: expand 4 bits → 4 float masks
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn packed_gemv_neon(&self, input: &[f32], output: &mut [f32]) {
        use std::arch::aarch64::*;

        for i in 0..self.out_dim {
            let row_offset = i * self.row_bytes;
            let mut sum_vec = vdupq_n_f32(0.0);

            let mut j = 0usize;
            let mut byte_idx = 0usize;

            // Process 4 floats at a time (half a byte of bits)
            while j + 8 <= self.in_dim {
                let pb = *self.pos_bits.get_unchecked(row_offset + byte_idx);
                let nb = *self.neg_bits.get_unchecked(row_offset + byte_idx);

                // Low 4 bits → first 4 floats
                {
                    let x_vec = vld1q_f32(input.as_ptr().add(j));
                    let pos_arr: [u32; 4] = [
                        if pb & 1 != 0 { 0xFFFFFFFF } else { 0 },
                        if pb & 2 != 0 { 0xFFFFFFFF } else { 0 },
                        if pb & 4 != 0 { 0xFFFFFFFF } else { 0 },
                        if pb & 8 != 0 { 0xFFFFFFFF } else { 0 },
                    ];
                    let neg_arr: [u32; 4] = [
                        if nb & 1 != 0 { 0xFFFFFFFF } else { 0 },
                        if nb & 2 != 0 { 0xFFFFFFFF } else { 0 },
                        if nb & 4 != 0 { 0xFFFFFFFF } else { 0 },
                        if nb & 8 != 0 { 0xFFFFFFFF } else { 0 },
                    ];
                    let pos_mask: uint32x4_t = vld1q_u32(pos_arr.as_ptr());
                    let neg_mask: uint32x4_t = vld1q_u32(neg_arr.as_ptr());
                    let added = vbslq_f32(pos_mask, x_vec, vdupq_n_f32(0.0));
                    let subbed = vbslq_f32(neg_mask, x_vec, vdupq_n_f32(0.0));
                    sum_vec = vaddq_f32(sum_vec, added);
                    sum_vec = vsubq_f32(sum_vec, subbed);
                }

                // High 4 bits → next 4 floats
                {
                    let x_vec = vld1q_f32(input.as_ptr().add(j + 4));
                    let pos_arr: [u32; 4] = [
                        if pb & 16 != 0 { 0xFFFFFFFF } else { 0 },
                        if pb & 32 != 0 { 0xFFFFFFFF } else { 0 },
                        if pb & 64 != 0 { 0xFFFFFFFF } else { 0 },
                        if pb & 128 != 0 { 0xFFFFFFFF } else { 0 },
                    ];
                    let neg_arr: [u32; 4] = [
                        if nb & 16 != 0 { 0xFFFFFFFF } else { 0 },
                        if nb & 32 != 0 { 0xFFFFFFFF } else { 0 },
                        if nb & 64 != 0 { 0xFFFFFFFF } else { 0 },
                        if nb & 128 != 0 { 0xFFFFFFFF } else { 0 },
                    ];
                    let pos_mask: uint32x4_t = vld1q_u32(pos_arr.as_ptr());
                    let neg_mask: uint32x4_t = vld1q_u32(neg_arr.as_ptr());
                    let added = vbslq_f32(pos_mask, x_vec, vdupq_n_f32(0.0));
                    let subbed = vbslq_f32(neg_mask, x_vec, vdupq_n_f32(0.0));
                    sum_vec = vaddq_f32(sum_vec, added);
                    sum_vec = vsubq_f32(sum_vec, subbed);
                }

                j += 8;
                byte_idx += 1;
            }

            // Horizontal sum
            let mut sum_array = [0.0f32; 4];
            vst1q_f32(sum_array.as_mut_ptr(), sum_vec);
            let mut sum: f32 = sum_array.iter().sum();

            // Tail
            while j < self.in_dim {
                let bi = row_offset + j / 8;
                let bp = j % 8;
                let mask = 1u8 << bp;
                if *self.pos_bits.get_unchecked(bi) & mask != 0 {
                    sum += *input.get_unchecked(j);
                } else if *self.neg_bits.get_unchecked(bi) & mask != 0 {
                    sum -= *input.get_unchecked(j);
                }
                j += 1;
            }

            *output.get_unchecked_mut(i) = sum * self.scale;
        }
    }
}
