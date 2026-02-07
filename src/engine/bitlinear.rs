// src/engine/bitlinear.rs
//! BitNet 1.58-bit linear layer implementation with i8 quantization and scale factors
//!
//! Uses ternary weights {-1, 0, +1} stored as i8 for extreme speed.
//! Dequantization: W_dequantized = W_i8 * scale (converted to F16 for inference)
//! Memory bandwidth reduced 4x compared to f32, with SIMD-friendly layout.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::VarBuilder;
use half::f16;
use parking_lot::RwLock;
use std::collections::HashMap;

/// RMS normalization
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x_sq = x.sqr()?;
    let mean_sq = x_sq.mean_keepdim(D::Minus1)?;
    let norm = (mean_sq + eps)?.sqrt()?;
    let x_normed = x.broadcast_div(&norm)?;
    x_normed.broadcast_mul(weight)
}

/// Fused RMSNorm + Linear for single token inference
/// Reduces memory bandwidth by avoiding intermediate tensor creation
pub fn rms_norm_linear_fused(
    x: &Tensor,
    rms_weight: &Tensor,
    weight_i8: &[i8],
    scale: f32,
    in_dim: usize,
    out_dim: usize,
    eps: f64,
) -> Result<Tensor> {
    let device = x.device();
    let x_flat = x.flatten_all()?.to_vec1::<f32>()?;

    // Compute RMSNorm inline
    let rms_weight_flat = rms_weight.flatten_all()?.to_vec1::<f32>()?;

    // Calculate mean squared
    let mean_sq: f32 = x_flat.iter().map(|&v| v * v).sum::<f32>() / in_dim as f32;
    let norm_factor = 1.0 / (mean_sq + eps as f32).sqrt();

    // Apply RMSNorm and linear transformation in one pass
    let mut output = vec![0.0f32; out_dim];

    for i in 0..out_dim {
        let mut sum = 0.0f32;
        let row_start = i * in_dim;

        // Fused: RMSNorm(x) @ W^T * scale
        for j in 0..in_dim {
            let x_normed = x_flat[j] * norm_factor * rms_weight_flat[j];
            let w = weight_i8[row_start + j];
            if w == 1 {
                sum += x_normed;
            } else if w == -1 {
                sum -= x_normed;
            }
        }
        output[i] = sum * scale;
    }

    // Return as F32 tensor to match input dtype for residual connections
    Tensor::from_vec(output, (1, 1, out_dim), device)
}

/// BitNet ternary linear layer with i8 quantization and scale dequantization
///
/// Implements BitNet 1.58-bit: weights stored as {-1, 0, +1} in i8 format,
/// with a per-layer scale factor for dequantization to F16.
pub struct BitLinear {
    /// Quantized weights as i8: -1, 0, or +1
    pub weight_i8: Vec<i8>,
    /// Scale factor for dequantization (from safetensors metadata)
    pub scale: f32,
    pub in_dim: usize,
    pub out_dim: usize,
    pub rms_norm_weight: Tensor,
    pub eps: f64,
    /// Cached dequantized weights in F16 (lazy initialization with thread-safe interior mutability)
    dequantized_cache: RwLock<Option<Tensor>>,
}

impl BitLinear {
    /// Load BitLinear layer from VarBuilder with scale from metadata
    ///
    /// # Arguments
    /// * `prefix` - Layer name prefix (e.g., "layers.0.linear")
    /// * `vb` - VarBuilder containing tensor data
    /// * `in_dim` - Input dimension
    /// * `out_dim` - Output dimension
    /// * `metadata` - Safetensors metadata containing scale factors
    pub fn load(
        prefix: &str,
        vb: &VarBuilder,
        in_dim: usize,
        out_dim: usize,
        metadata: &HashMap<String, String>,
    ) -> Result<Self> {
        // Load weights
        let weight_key = format!("{}.weight", prefix);
        let weight = vb.get((out_dim, in_dim), &weight_key)?;

        // Quantize to ternary {-1, 0, +1}
        let raw = weight.to_vec2::<f32>()?;
        let mut weight_i8 = Vec::with_capacity(out_dim * in_dim);
        for row in &raw {
            for &w in row {
                let q = if w > 0.01 {
                    1i8
                } else if w < -0.01 {
                    -1i8
                } else {
                    0i8
                };
                weight_i8.push(q);
            }
        }

        // Parse scale from metadata (key format: {layer_name}_scale)
        let scale_key = format!("{}_scale", prefix);
        let scale = metadata
            .get(&scale_key)
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(1.0); // Default scale if not found

        // Load RMS norm weight
        let norm_key = format!("{}.norm.weight", prefix);
        let rms_norm_weight = vb.get((in_dim,), &norm_key)?;

        Ok(Self {
            weight_i8,
            scale,
            in_dim,
            out_dim,
            rms_norm_weight,
            eps: 1e-5,
            dequantized_cache: RwLock::new(None),
        })
    }

    /// Load without metadata (backward compatibility, scale=1.0)
    pub fn load_legacy(
        prefix: &str,
        vb: &VarBuilder,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Self> {
        Self::load(prefix, vb, in_dim, out_dim, &HashMap::new())
    }

    /// Dequantize weights: W_dequantized = W_i8 * scale
    /// Returns F16 tensor for efficient inference
    fn dequantize_weights(&self, device: &Device) -> Result<Tensor> {
        // Try to read from cache first
        {
            let cache = self.dequantized_cache.read();
            if let Some(ref tensor) = *cache {
                return Ok(tensor.clone());
            }
        }

        // Need to create - upgrade to write lock
        let mut cache = self.dequantized_cache.write();

        // Double-check after acquiring write lock
        if let Some(ref tensor) = *cache {
            return Ok(tensor.clone());
        }

        // Convert i8 weights to f16 and apply scale
        let weights_f16: Vec<f16> = self
            .weight_i8
            .iter()
            .map(|&w| f16::from_f32(w as f32 * self.scale))
            .collect();
        let weights_tensor = Tensor::from_vec(weights_f16, (self.out_dim, self.in_dim), device)?;
        *cache = Some(weights_tensor.clone());

        Ok(weights_tensor)
    }

    /// Force dequantization cache refresh (e.g., after device change)
    pub fn clear_cache(&self) {
        let mut cache = self.dequantized_cache.write();
        *cache = None;
    }

    /// Forward pass: y = RMSNorm(x) @ W^T * scale
    ///
    /// Performs BitNet 1.58-bit linear transformation with dequantization.
    /// Output dtype matches input dtype to preserve compatibility with residual connections.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let device = x.device();
        let input_dtype = x.dtype();

        // Apply RMS normalization
        let x_normed = rms_norm(x, &self.rms_norm_weight, self.eps)?;

        // Compute in F16 for memory efficiency
        let x_f16 = if x_normed.dtype() == DType::F16 {
            x_normed
        } else {
            x_normed.to_dtype(DType::F16)?
        };

        // Get dequantized weights (F16)
        let weights_dequantized = self.dequantize_weights(device)?;
        let w_t = weights_dequantized.t()?;

        // Perform matrix multiplication: y = x @ W^T
        let output = x_f16.broadcast_matmul(&w_t)?;

        // Convert back to input dtype to avoid mismatch in residual add
        if output.dtype() != input_dtype {
            output.to_dtype(input_dtype)
        } else {
            Ok(output)
        }
    }

    /// Optimized forward for single token (batch_size=1, seq_len=1)
    /// Uses SIMD optimizations (AVX-512/AVX2/NEON) for maximum CPU performance
    pub fn forward_single(&self, x: &Tensor) -> Result<Tensor> {
        let device = x.device();
        let shape = x.dims();

        // Ensure single token
        if shape[0] != 1 || (shape.len() > 1 && shape[1] != 1) {
            return self.forward(x);
        }

        // Apply RMS norm
        let x_normed = rms_norm(x, &self.rms_norm_weight, self.eps)?;
        let x_f16 = x_normed.to_dtype(DType::F16)?;

        // Flatten input
        let x_flat = x_f16.flatten_all()?.to_vec1::<f16>()?;
        let x_f32: Vec<f32> = x_flat.iter().map(|&v| v.to_f32()).collect();

        // Fast ternary GEMM with dequantization using SIMD
        let mut output = vec![0.0f32; self.out_dim];

        // Use SIMD-optimized implementation based on CPU features
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                // AVX-512: Process 16 floats at a time
                unsafe {
                    self.ternary_gemm_avx512(&x_f32, &mut output);
                }
            } else if is_x86_feature_detected!("avx2") {
                // AVX2: Process 8 floats at a time
                unsafe {
                    self.ternary_gemm_avx2(&x_f32, &mut output);
                }
            } else {
                // Scalar fallback with unrolled loops
                self.ternary_gemm_scalar(&x_f32, &mut output);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                // ARM NEON: Process 4 floats at a time
                unsafe {
                    self.ternary_gemm_neon(&x_f32, &mut output);
                }
            } else {
                self.ternary_gemm_scalar(&x_f32, &mut output);
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            self.ternary_gemm_scalar(&x_f32, &mut output);
        }

        // Return as F32 tensor to match input dtype for residual connections
        Tensor::from_vec(output, (1, 1, self.out_dim), device)
    }

    /// Scalar implementation with 4x unrolling
    fn ternary_gemm_scalar(&self, input: &[f32], output: &mut [f32]) {
        for i in 0..self.out_dim {
            let mut sum = 0.0f32;
            let row_start = i * self.in_dim;

            // Main loop with 4x unrolling
            let mut j = 0;
            while j + 4 <= self.in_dim {
                unsafe {
                    let w0 = *self.weight_i8.get_unchecked(row_start + j);
                    let w1 = *self.weight_i8.get_unchecked(row_start + j + 1);
                    let w2 = *self.weight_i8.get_unchecked(row_start + j + 2);
                    let w3 = *self.weight_i8.get_unchecked(row_start + j + 3);

                    let x0 = *input.get_unchecked(j);
                    let x1 = *input.get_unchecked(j + 1);
                    let x2 = *input.get_unchecked(j + 2);
                    let x3 = *input.get_unchecked(j + 3);

                    if w0 == 1 {
                        sum += x0;
                    } else if w0 == -1 {
                        sum -= x0;
                    }
                    if w1 == 1 {
                        sum += x1;
                    } else if w1 == -1 {
                        sum -= x1;
                    }
                    if w2 == 1 {
                        sum += x2;
                    } else if w2 == -1 {
                        sum -= x2;
                    }
                    if w3 == 1 {
                        sum += x3;
                    } else if w3 == -1 {
                        sum -= x3;
                    }
                }
                j += 4;
            }

            // Remainder
            while j < self.in_dim {
                unsafe {
                    let w = *self.weight_i8.get_unchecked(row_start + j);
                    let x = *input.get_unchecked(j);
                    if w == 1 {
                        sum += x;
                    } else if w == -1 {
                        sum -= x;
                    }
                }
                j += 1;
            }

            output[i] = sum * self.scale;
        }
    }

    /// AVX-512 optimized ternary GEMM
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn ternary_gemm_avx512(&self, input: &[f32], output: &mut [f32]) {
        use std::arch::x86_64::*;

        for i in 0..self.out_dim {
            let mut sum_vec = _mm512_setzero_ps();
            let row_start = i * self.in_dim;

            // Process in chunks of 16 floats (512 bits)
            let mut j = 0;
            while j + 16 <= self.in_dim {
                let x_vec = _mm512_loadu_ps(input.as_ptr().add(j));

                // Load weights as i8, convert to f32 {-1.0, 0.0, 1.0}
                let w0 = *self.weight_i8.get_unchecked(row_start + j) as f32;
                let w1 = *self.weight_i8.get_unchecked(row_start + j + 1) as f32;
                let w2 = *self.weight_i8.get_unchecked(row_start + j + 2) as f32;
                let w3 = *self.weight_i8.get_unchecked(row_start + j + 3) as f32;
                let w4 = *self.weight_i8.get_unchecked(row_start + j + 4) as f32;
                let w5 = *self.weight_i8.get_unchecked(row_start + j + 5) as f32;
                let w6 = *self.weight_i8.get_unchecked(row_start + j + 6) as f32;
                let w7 = *self.weight_i8.get_unchecked(row_start + j + 7) as f32;
                let w8 = *self.weight_i8.get_unchecked(row_start + j + 8) as f32;
                let w9 = *self.weight_i8.get_unchecked(row_start + j + 9) as f32;
                let w10 = *self.weight_i8.get_unchecked(row_start + j + 10) as f32;
                let w11 = *self.weight_i8.get_unchecked(row_start + j + 11) as f32;
                let w12 = *self.weight_i8.get_unchecked(row_start + j + 12) as f32;
                let w13 = *self.weight_i8.get_unchecked(row_start + j + 13) as f32;
                let w14 = *self.weight_i8.get_unchecked(row_start + j + 14) as f32;
                let w15 = *self.weight_i8.get_unchecked(row_start + j + 15) as f32;

                let w_vec = _mm512_set_ps(
                    w15, w14, w13, w12, w11, w10, w9, w8, w7, w6, w5, w4, w3, w2, w1, w0,
                );

                // Create masks for positive and negative weights
                let pos_mask = _mm512_cmp_ps_mask(w_vec, _mm512_setzero_ps(), _CMP_GT_OQ);
                let neg_mask = _mm512_cmp_ps_mask(w_vec, _mm512_setzero_ps(), _CMP_LT_OQ);

                // Add where W > 0 (W=1), subtract where W < 0 (W=-1)
                sum_vec = _mm512_mask_add_ps(sum_vec, pos_mask, sum_vec, x_vec);
                sum_vec = _mm512_mask_sub_ps(sum_vec, neg_mask, sum_vec, x_vec);

                j += 16;
            }

            // Horizontal sum
            let mut sum = _mm512_reduce_add_ps(sum_vec);

            // Tail cleanup
            while j < self.in_dim {
                let w = *self.weight_i8.get_unchecked(row_start + j);
                let x = *input.get_unchecked(j);
                if w == 1 {
                    sum += x;
                } else if w == -1 {
                    sum -= x;
                }
                j += 1;
            }

            output[i] = sum * self.scale;
        }
    }

    /// AVX2 optimized ternary GEMM
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn ternary_gemm_avx2(&self, input: &[f32], output: &mut [f32]) {
        use std::arch::x86_64::*;

        for i in 0..self.out_dim {
            let mut sum_vec = _mm256_setzero_ps();
            let row_start = i * self.in_dim;

            // Process in chunks of 8 floats (256 bits)
            let mut j = 0;
            while j + 8 <= self.in_dim {
                let x_vec = _mm256_loadu_ps(input.as_ptr().add(j));

                // Load 8 weights and convert to f32
                let w0 = *self.weight_i8.get_unchecked(row_start + j) as f32;
                let w1 = *self.weight_i8.get_unchecked(row_start + j + 1) as f32;
                let w2 = *self.weight_i8.get_unchecked(row_start + j + 2) as f32;
                let w3 = *self.weight_i8.get_unchecked(row_start + j + 3) as f32;
                let w4 = *self.weight_i8.get_unchecked(row_start + j + 4) as f32;
                let w5 = *self.weight_i8.get_unchecked(row_start + j + 5) as f32;
                let w6 = *self.weight_i8.get_unchecked(row_start + j + 6) as f32;
                let w7 = *self.weight_i8.get_unchecked(row_start + j + 7) as f32;

                let w_vec = _mm256_set_ps(w7, w6, w5, w4, w3, w2, w1, w0);

                // Create masks for positive and negative weights
                let pos_mask = _mm256_cmp_ps(w_vec, _mm256_setzero_ps(), _CMP_GT_OQ);
                let neg_mask = _mm256_cmp_ps(w_vec, _mm256_setzero_ps(), _CMP_LT_OQ);

                // Apply masks: add where positive, subtract where negative
                let added = _mm256_and_ps(x_vec, pos_mask);
                let subbed = _mm256_and_ps(x_vec, neg_mask);

                sum_vec = _mm256_add_ps(sum_vec, added);
                sum_vec = _mm256_sub_ps(sum_vec, subbed);

                j += 8;
            }

            // Horizontal sum
            let mut arr = [0.0f32; 8];
            _mm256_storeu_ps(arr.as_mut_ptr(), sum_vec);
            let mut sum: f32 = arr.iter().sum();

            // Tail cleanup
            while j < self.in_dim {
                let w = *self.weight_i8.get_unchecked(row_start + j);
                let x = *input.get_unchecked(j);
                if w == 1 {
                    sum += x;
                } else if w == -1 {
                    sum -= x;
                }
                j += 1;
            }

            output[i] = sum * self.scale;
        }
    }

    /// ARM NEON optimized ternary GEMM
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn ternary_gemm_neon(&self, input: &[f32], output: &mut [f32]) {
        use std::arch::aarch64::*;

        for i in 0..self.out_dim {
            let mut sum_vec = vdupq_n_f32(0.0);
            let row_start = i * self.in_dim;

            // Process in chunks of 4 floats (128 bits)
            let mut j = 0;
            while j + 4 <= self.in_dim {
                let x_vec = vld1q_f32(input.as_ptr().add(j));

                // Load 4 weights and convert to f32
                let w0 = *self.weight_i8.get_unchecked(row_start + j) as f32;
                let w1 = *self.weight_i8.get_unchecked(row_start + j + 1) as f32;
                let w2 = *self.weight_i8.get_unchecked(row_start + j + 2) as f32;
                let w3 = *self.weight_i8.get_unchecked(row_start + j + 3) as f32;

                // Create weight vector
                let w_arr = [w0, w1, w2, w3];
                let w_vec = vld1q_f32(w_arr.as_ptr());

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

            // Horizontal sum
            let mut sum_array = [0.0f32; 4];
            vst1q_f32(sum_array.as_mut_ptr(), sum_vec);
            let mut sum: f32 = sum_array.iter().sum();

            // Tail cleanup
            while j < self.in_dim {
                let w = *self.weight_i8.get_unchecked(row_start + j);
                let x = *input.get_unchecked(j);
                if w == 1 {
                    sum += x;
                } else if w == -1 {
                    sum -= x;
                }
                j += 1;
            }

            output[i] = sum * self.scale;
        }
    }
}

/// BitLinear layer without cache (for immutable contexts)
pub struct BitLinearImmutable {
    pub weight_i8: Vec<i8>,
    pub scale: f32,
    pub in_dim: usize,
    pub out_dim: usize,
    pub rms_norm_weight: Tensor,
    pub eps: f64,
}

impl BitLinearImmutable {
    /// Load from VarBuilder with metadata
    pub fn load(
        prefix: &str,
        vb: &VarBuilder,
        in_dim: usize,
        out_dim: usize,
        metadata: &HashMap<String, String>,
    ) -> Result<Self> {
        // Load weights
        let weight_key = format!("{}.weight", prefix);
        let weight = vb.get((out_dim, in_dim), &weight_key)?;

        // Quantize to ternary {-1, 0, +1}
        let raw = weight.to_vec2::<f32>()?;
        let mut weight_i8 = Vec::with_capacity(out_dim * in_dim);
        for row in &raw {
            for &w in row {
                let q = if w > 0.01 {
                    1i8
                } else if w < -0.01 {
                    -1i8
                } else {
                    0i8
                };
                weight_i8.push(q);
            }
        }

        let scale_key = format!("{}_scale", prefix);
        let scale = metadata
            .get(&scale_key)
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(1.0);

        let norm_key = format!("{}.norm.weight", prefix);
        let rms_norm_weight = vb.get((in_dim,), &norm_key)?;

        Ok(Self {
            weight_i8,
            scale,
            in_dim,
            out_dim,
            rms_norm_weight,
            eps: 1e-5,
        })
    }

    /// Forward pass with on-the-fly dequantization (no caching)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let device = x.device();
        let input_dtype = x.dtype();

        // Apply RMS norm
        let x_normed = rms_norm(x, &self.rms_norm_weight, self.eps)?;
        let x_f16 = x_normed.to_dtype(DType::F16)?;

        // Dequantize weights on-the-fly
        let weights_f16: Vec<f16> = self
            .weight_i8
            .iter()
            .map(|&w| f16::from_f32(w as f32 * self.scale))
            .collect();
        let weights_tensor = Tensor::from_vec(weights_f16, (self.out_dim, self.in_dim), device)?;
        let w_t = weights_tensor.t()?;

        // Matmul and convert back to input dtype
        let output = x_f16.broadcast_matmul(&w_t)?;
        if output.dtype() != input_dtype {
            output.to_dtype(input_dtype)
        } else {
            Ok(output)
        }
    }
}
