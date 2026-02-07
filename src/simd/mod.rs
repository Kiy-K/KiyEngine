// src/simd/mod.rs

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
use std::arch::aarch64::*;

/// BitNet Ternary Gemm optimized for AVX-512.
/// Calculates: output = input * weights_t
/// where weights_t are strict {-1.0, 0.0, 1.0}.
/// This implementation uses SIMD to parallelize the addition/subtraction.
pub fn bitnet_ternary_gemm_avx512(
    input: &[f32],
    weights_t: &[f32],
    output: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    #[cfg(target_feature = "avx512f")]
    unsafe {
        for i in 0..out_dim {
            let mut sum_vec = _mm512_setzero_ps();
            let row_offset = i * in_dim;

            // Process in chunks of 16 floats (512 bits)
            let mut j = 0;
            while j + 15 < in_dim {
                let x_vec = _mm512_loadu_ps(input.as_ptr().add(j));
                let w_vec = _mm512_loadu_ps(weights_t.as_ptr().add(row_offset + j));

                // BitNet optimization: ternary weights {-1.0, 0.0, 1.0}.
                // We use masks to determine where to add or subtract.
                // Thanks to strict quantization in BitLinear::load, we can
                // compare against 0.0 instead of using loose thresholds.
                let pos_mask = _mm512_cmp_ps_mask(w_vec, _mm512_set1_ps(0.0), _CMP_GT_OQ);
                let neg_mask = _mm512_cmp_ps_mask(w_vec, _mm512_set1_ps(0.0), _CMP_LT_OQ);

                // Add where W > 0.5 (W=1)
                sum_vec = _mm512_mask_add_ps(sum_vec, pos_mask, sum_vec, x_vec);
                // Subtract where W < -0.5 (W=-1)
                sum_vec = _mm512_mask_sub_ps(sum_vec, neg_mask, sum_vec, x_vec);

                j += 16;
            }

            // Horizontal sum of the 16 floats in sum_vec
            output[i] += _mm512_reduce_add_ps(sum_vec);

            // Tail cleanup
            while j < in_dim {
                let w = weights_t[row_offset + j];
                if w > 0.0 {
                    output[i] += input[j];
                } else if w < 0.0 {
                    output[i] -= input[j];
                }
                j += 1;
            }
        }
    }

    #[cfg(not(target_feature = "avx512f"))]
    {
        // Fallback to optimized scalar implementation
        for i in 0..out_dim {
            let mut sum = 0.0;
            let row_offset = i * in_dim;
            for j in 0..in_dim {
                let w = weights_t[row_offset + j];
                if w > 0.0 {
                    sum += input[j];
                } else if w < 0.0 {
                    sum -= input[j];
                }
            }
            output[i] += sum;
        }
    }
}

/// Fallback function for non-AVX-512 systems (using AVX2 if available)
#[cfg(target_arch = "x86_64")]
pub fn bitnet_ternary_gemm_avx2(
    input: &[f32],
    weights_t: &[f32],
    output: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    #[cfg(target_feature = "avx2")]
    unsafe {
        for i in 0..out_dim {
            let mut sum_vec = _mm256_setzero_ps();
            let row_offset = i * in_dim;

            let mut j = 0;
            while j + 7 < in_dim {
                let x_vec = _mm256_loadu_ps(input.as_ptr().add(j));
                let w_vec = _mm256_loadu_ps(weights_t.as_ptr().add(row_offset + j));

                // AVX2 doesn't have direct mask add/sub like AVX-512.
                // We calculate masks as floats and multiply. With strictly
                // quantized weights, comparing to 0.0 is sufficient.
                let pos_mask = _mm256_cmp_ps(w_vec, _mm256_set1_ps(0.0), _CMP_GT_OQ);
                let neg_mask = _mm256_cmp_ps(w_vec, _mm256_set1_ps(0.0), _CMP_LT_OQ);

                let added = _mm256_and_ps(x_vec, pos_mask);
                let subbed = _mm256_and_ps(x_vec, neg_mask);

                sum_vec = _mm256_add_ps(sum_vec, added);
                sum_vec = _mm256_sub_ps(sum_vec, subbed);

                j += 8;
            }

            // Horizontal sum
            let mut arr = [0.0f32; 8];
            _mm256_storeu_ps(arr.as_mut_ptr(), sum_vec);
            output[i] += arr.iter().sum::<f32>();

            // Tail cleanup
            while j < in_dim {
                let w = weights_t[row_offset + j];
                if w > 0.0 {
                    output[i] += input[j];
                } else if w < 0.0 {
                    output[i] -= input[j];
                }
                j += 1;
            }
        }
    }

    #[cfg(not(target_feature = "avx2"))]
    {
        // Extreme fallback
        for i in 0..out_dim {
            let mut sum = 0.0;
            let row_offset = i * in_dim;
            for j in 0..in_dim {
                let w = weights_t[row_offset + j];
                if w > 0.0 {
                    sum += input[j];
                } else if w < 0.0 {
                    sum -= input[j];
                }
            }
            output[i] += sum;
        }
    }
}

/// BitNet Ternary Gemm optimized for ARM NEON.
/// Calculates: output = input * weights_t
/// where weights_t are strict {-1.0, 0.0, 1.0}.
#[cfg(target_arch = "aarch64")]
pub fn bitnet_ternary_gemm_neon(
    input: &[f32],
    weights_t: &[f32],
    output: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    #[cfg(target_feature = "neon")]
    unsafe {
        for i in 0..out_dim {
            let mut sum_vec = vdupq_n_f32(0.0);
            let row_offset = i * in_dim;

            // Process in chunks of 4 floats (128 bits)
            let mut j = 0;
            while j + 3 < in_dim {
                let x_vec = vld1q_f32(input.as_ptr().add(j));
                let w_vec = vld1q_f32(weights_t.as_ptr().add(row_offset + j));

                // Create masks for positive and negative weights
                let zero_vec = vdupq_n_f32(0.0);
                let pos_mask = vcgtq_f32(w_vec, zero_vec);
                let neg_mask = vcltq_f32(w_vec, zero_vec);

                // Apply masks: add where positive, subtract where negative
                let added = vbslq_f32(pos_mask, x_vec, vdupq_n_f32(0.0));
                let subbed = vbslq_f32(neg_mask, x_vec, vdupq_n_f32(0.0));

                sum_vec = vaddq_f32(sum_vec, added);
                sum_vec = vsubq_f32(sum_vec, subbed);

                j += 4;
            }

            // Horizontal sum of the 4 floats in sum_vec
            let mut sum_array = [0.0f32; 4];
            vst1q_f32(sum_array.as_mut_ptr(), sum_vec);
            output[i] += sum_array.iter().sum::<f32>();

            // Tail cleanup
            while j < in_dim {
                let w = weights_t[row_offset + j];
                if w > 0.0 {
                    output[i] += input[j];
                } else if w < 0.0 {
                    output[i] -= input[j];
                }
                j += 1;
            }
        }
    }

    #[cfg(not(target_feature = "neon"))]
    {
        // Fallback to scalar implementation
        for i in 0..out_dim {
            let mut sum = 0.0;
            let row_offset = i * in_dim;
            for j in 0..in_dim {
                let w = weights_t[row_offset + j];
                if w > 0.0 {
                    sum += input[j];
                } else if w < 0.0 {
                    sum -= input[j];
                }
            }
            output[i] += sum;
        }
    }
}
