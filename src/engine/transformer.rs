// src/engine/transformer.rs
//! v6.0.0 KiyNet_V6 Transformer components:
//! - GQA MultiheadAttention (separate BitLinear Q/K/V/O + RoPE)
//! - LayerScale (learnable per-channel scaling)
//! - BitSwiGLU (gated MLP with ternary BitLinear layers)
//! - TransformerBlock (attention + MLP with pre-norm residual)

use candle_core::{Result, Tensor, D};
use candle_nn::{Module, VarBuilder};

use crate::engine::bitlinear::{rms_norm, rms_norm_vec, BitLinear};

/// KV Cache for incremental transformer inference.
/// Stores per-layer K and V tensors so only new tokens need QKV projection.
/// V6: KV heads = 2 (GQA), so cache is much smaller than v5.
pub struct KVCache {
    /// Per-layer (K, V) caches
    /// K: [batch, num_kv_heads, cached_seq_len, head_dim]
    /// V: [batch, num_kv_heads, cached_seq_len, head_dim]
    pub layers: Vec<Option<(Tensor, Tensor)>>,
    /// Number of tokens currently cached
    pub seq_len: usize,
    /// The token sequence that produced this cache (for validation)
    pub tokens: Vec<usize>,
}

impl KVCache {
    /// Create empty cache for `num_layers` transformer layers
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| None).collect(),
            seq_len: 0,
            tokens: Vec::new(),
        }
    }

    /// Reset the cache (e.g., new game)
    pub fn clear(&mut self) {
        for slot in &mut self.layers {
            *slot = None;
        }
        self.seq_len = 0;
        self.tokens.clear();
    }

    /// Check if new_tokens extend the cached sequence (prefix match)
    pub fn can_reuse(&self, new_tokens: &[usize]) -> bool {
        if self.seq_len == 0 || new_tokens.len() <= self.seq_len {
            return false;
        }
        // The first `seq_len` tokens must match exactly
        new_tokens[..self.seq_len] == self.tokens[..]
    }

    /// Number of new tokens that need processing
    pub fn new_token_count(&self, total_tokens: usize) -> usize {
        total_tokens.saturating_sub(self.seq_len)
    }
}

/// Precompute RoPE frequency pairs for given dim and max sequence length.
/// Returns cos and sin tensors of shape [max_len, dim/2].
pub fn precompute_rope(
    head_dim: usize,
    max_len: usize,
    theta: f64,
    device: &candle_core::Device,
) -> Result<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;
    let mut cos_vals = Vec::with_capacity(max_len * half_dim);
    let mut sin_vals = Vec::with_capacity(max_len * half_dim);
    for pos in 0..max_len {
        for i in 0..half_dim {
            let freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
            let angle = pos as f64 * freq;
            cos_vals.push(angle.cos() as f32);
            sin_vals.push(angle.sin() as f32);
        }
    }
    let cos = Tensor::from_vec(cos_vals, (max_len, half_dim), device)?;
    let sin = Tensor::from_vec(sin_vals, (max_len, half_dim), device)?;
    Ok((cos, sin))
}

/// Apply RoPE to tensor x of shape [batch, heads, seq_len, head_dim]
/// cos/sin are [max_len, half_dim], sliced to [seq_len, half_dim] starting at offset.
fn apply_rope(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    offset: usize,
) -> Result<Tensor> {
    let (_batch, _heads, seq_len, head_dim) = x.dims4()?;
    let half = head_dim / 2;
    // Slice cos/sin for positions [offset .. offset+seq_len]
    let cos_slice = cos.narrow(0, offset, seq_len)?; // [seq_len, half]
    let sin_slice = sin.narrow(0, offset, seq_len)?;
    // Reshape for broadcasting: [1, 1, seq_len, half]
    let cos_b = cos_slice.unsqueeze(0)?.unsqueeze(0)?;
    let sin_b = sin_slice.unsqueeze(0)?.unsqueeze(0)?;
    // Split x into even and odd halves
    let x_even = x.narrow(D::Minus1, 0, half)?;
    let x_odd = x.narrow(D::Minus1, half, half)?;
    // x_rot_even = x_even * cos - x_odd * sin
    // x_rot_odd  = x_even * sin + x_odd * cos
    let rot_even = (x_even.broadcast_mul(&cos_b)? - x_odd.broadcast_mul(&sin_b)?)?;
    let rot_odd = (x_even.broadcast_mul(&sin_b)? + x_odd.broadcast_mul(&cos_b)?)?;
    Tensor::cat(&[&rot_even, &rot_odd], D::Minus1)
}

/// LayerScale: element-wise multiply by learnable gamma vector
pub struct LayerScale {
    pub gamma: Tensor,
}

impl LayerScale {
    pub fn load(key: &str, vb: &VarBuilder, dim: usize) -> Result<Self> {
        let gamma = vb.get((dim,), key)?;
        Ok(Self { gamma })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.broadcast_mul(&self.gamma)
    }
}

/// GQA MultiheadAttention with separate BitLinear Q/K/V/O projections and RoPE
///
/// V6: 8 query heads, 2 KV heads (GQA ratio 4:1), no biases.
/// All projections are BitLinear (ternary weights with RMSNorm).
pub struct GQAttention {
    pub q_proj: BitLinear,
    pub k_proj: BitLinear,
    pub v_proj: BitLinear,
    pub o_proj: BitLinear,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub embed_dim: usize,
    pub scale: f64,
    /// GQA repeat factor: num_heads / num_kv_heads
    pub gqa_repeat: usize,
}

impl GQAttention {
    pub fn load(
        prefix: &str,
        vb: &VarBuilder,
        embed_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let q_dim = num_heads * head_dim;     // 512
        let kv_dim = num_kv_heads * head_dim; // 128

        let q_proj = BitLinear::load_from_vb(
            &format!("{}.q_proj.weight", prefix),
            &format!("{}.q_proj.norm.weight", prefix),
            vb, embed_dim, q_dim,
        )?;
        let k_proj = BitLinear::load_from_vb(
            &format!("{}.k_proj.weight", prefix),
            &format!("{}.k_proj.norm.weight", prefix),
            vb, embed_dim, kv_dim,
        )?;
        let v_proj = BitLinear::load_from_vb(
            &format!("{}.v_proj.weight", prefix),
            &format!("{}.v_proj.norm.weight", prefix),
            vb, embed_dim, kv_dim,
        )?;
        let o_proj = BitLinear::load_from_vb(
            &format!("{}.o_proj.weight", prefix),
            &format!("{}.o_proj.norm.weight", prefix),
            vb, embed_dim, embed_dim,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            embed_dim,
            scale: (head_dim as f64).sqrt(),
            gqa_repeat: num_heads / num_kv_heads,
        })
    }

    /// Expand KV heads to match query heads via repeat_interleave.
    /// [batch, num_kv_heads, seq, head_dim] → [batch, num_heads, seq, head_dim]
    fn expand_kv(&self, kv: &Tensor) -> Result<Tensor> {
        if self.gqa_repeat == 1 {
            return Ok(kv.clone());
        }
        let (batch, _kv_heads, seq, hd) = kv.dims4()?;
        // [batch, kv_heads, 1, seq, hd] → repeat → [batch, kv_heads, repeat, seq, hd]
        let expanded = kv
            .unsqueeze(2)?
            .expand((batch, self.num_kv_heads, self.gqa_repeat, seq, hd))?
            .reshape((batch, self.num_heads, seq, hd))?;
        Ok(expanded)
    }

    /// Forward: GQA self-attention with RoPE and causal mask
    /// x: [batch, seq_len, embed_dim]
    /// rope_cos/sin: precomputed [max_len, half_dim]
    /// causal_mask: [1, 1, max_seq, max_seq] or None for single token
    pub fn forward(
        &self,
        x: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        causal_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        // Separate projections (each does RMSNorm internally)
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape: Q → [batch, num_heads, seq, head_dim]
        let q = q.reshape((batch, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        // K,V → [batch, num_kv_heads, seq, head_dim]
        let k = k.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        // Apply RoPE to Q and K
        let q = apply_rope(&q, rope_cos, rope_sin, 0)?;
        let k = apply_rope(&k, rope_cos, rope_sin, 0)?;

        // Expand KV heads for GQA: [batch, num_heads, seq, head_dim]
        let k = self.expand_kv(&k)?;
        let v = self.expand_kv(&v)?;

        // Scaled dot-product attention
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let attn_weights = q.matmul(&k_t)?;
        let attn_weights = (attn_weights / self.scale)?;

        let attn_weights = if let Some(mask) = causal_mask {
            let mask = mask.narrow(2, 0, seq_len)?.narrow(3, 0, seq_len)?;
            let masked = attn_weights.broadcast_add(&mask)?;
            candle_nn::ops::softmax(&masked, D::Minus1)?
        } else {
            candle_nn::ops::softmax(&attn_weights, D::Minus1)?
        };

        let attn_out = attn_weights.matmul(&v)?;
        let attn_out = attn_out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, self.embed_dim))?;

        // Output projection (BitLinear with RMSNorm)
        self.o_proj.forward(&attn_out)
    }

    /// Forward with KV cache for incremental inference.
    /// Returns (output, new_k, new_v) where new_k/new_v are full KV for cache update.
    /// K,V returned at num_kv_heads dimension (not expanded).
    pub fn forward_cached(
        &self,
        x_new: &Tensor,
        cache_k: Option<&Tensor>,
        cache_v: Option<&Tensor>,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        causal_mask: &Tensor,
        total_seq_len: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, new_tokens, _) = x_new.dims3()?;
        let cached_len = total_seq_len - new_tokens;

        let q = self.q_proj.forward(x_new)?;
        let k = self.k_proj.forward(x_new)?;
        let v = self.v_proj.forward(x_new)?;

        let q = q.reshape((batch, new_tokens, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((batch, new_tokens, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((batch, new_tokens, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        // RoPE with offset for cached positions
        let q = apply_rope(&q, rope_cos, rope_sin, cached_len)?;
        let k = apply_rope(&k, rope_cos, rope_sin, cached_len)?;

        // Concatenate with cache (at num_kv_heads dimension)
        let full_k = if let Some(ck) = cache_k {
            Tensor::cat(&[ck, &k], 2)?
        } else {
            k
        };
        let full_v = if let Some(cv) = cache_v {
            Tensor::cat(&[cv, &v], 2)?
        } else {
            v
        };

        // Expand KV for attention computation
        let k_expanded = self.expand_kv(&full_k)?;
        let v_expanded = self.expand_kv(&full_v)?;

        let k_t = k_expanded.transpose(2, 3)?.contiguous()?;
        let attn_weights = q.matmul(&k_t)?;
        let attn_weights = (attn_weights / self.scale)?;

        // Single-token optimization: the last row of causal mask is all-zero
        // (attends to all previous positions), so skip masking entirely
        let attn_weights = if new_tokens == 1 {
            candle_nn::ops::softmax(&attn_weights, D::Minus1)?
        } else {
            let mask_rows = causal_mask.narrow(2, cached_len, new_tokens)?;
            let mask_cols = mask_rows.narrow(3, 0, total_seq_len)?;
            let attn_weights = attn_weights.broadcast_add(&mask_cols)?;
            candle_nn::ops::softmax(&attn_weights, D::Minus1)?
        };

        let attn_out = attn_weights.matmul(&v_expanded)?;
        let attn_out = attn_out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, new_tokens, self.embed_dim))?;

        let output = self.o_proj.forward(&attn_out)?;

        // Return un-expanded K,V for cache storage (saves memory)
        Ok((output, full_k, full_v))
    }

    /// Fused single-token cached forward: inline RoPE on f32, per-group attention.
    /// Eliminates ~20 tensor allocations vs the generic path.
    /// rope_cos_f32/sin_f32: flat [max_len * half_dim] arrays.
    pub fn forward_cached_single_token(
        &self,
        x_new: &Tensor,
        cache_k: Option<&Tensor>,
        cache_v: Option<&Tensor>,
        rope_cos_f32: &[f32],
        rope_sin_f32: &[f32],
        pos: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let device = x_new.device();
        let half = self.head_dim / 2;

        // Q/K/V projections (auto-dispatches to SIMD GEMV for single token)
        let q_raw = self.q_proj.forward(x_new)?; // [1, 1, q_dim]
        let k_raw = self.k_proj.forward(x_new)?; // [1, 1, kv_dim]
        let v_raw = self.v_proj.forward(x_new)?; // [1, 1, kv_dim]

        // Extract to f32 for inline RoPE
        let mut q_f32: Vec<f32> = q_raw.flatten_all()?.to_vec1::<f32>()?;
        let mut k_f32: Vec<f32> = k_raw.flatten_all()?.to_vec1::<f32>()?;
        let v_f32: Vec<f32> = v_raw.flatten_all()?.to_vec1::<f32>()?;

        // RoPE cos/sin for position `pos`
        let cos_row = &rope_cos_f32[pos * half..(pos + 1) * half];
        let sin_row = &rope_sin_f32[pos * half..(pos + 1) * half];

        // Apply RoPE inline to Q (num_heads × head_dim)
        for h in 0..self.num_heads {
            let base = h * self.head_dim;
            for i in 0..half {
                let even = q_f32[base + i];
                let odd = q_f32[base + half + i];
                q_f32[base + i] = even * cos_row[i] - odd * sin_row[i];
                q_f32[base + half + i] = even * sin_row[i] + odd * cos_row[i];
            }
        }
        // Apply RoPE inline to K (num_kv_heads × head_dim)
        for h in 0..self.num_kv_heads {
            let base = h * self.head_dim;
            for i in 0..half {
                let even = k_f32[base + i];
                let odd = k_f32[base + half + i];
                k_f32[base + i] = even * cos_row[i] - odd * sin_row[i];
                k_f32[base + half + i] = even * sin_row[i] + odd * cos_row[i];
            }
        }

        // Create tensors with correct shapes
        // Q: [1, num_heads, 1, head_dim]
        let q = Tensor::from_vec(q_f32, (1, self.num_heads, 1, self.head_dim), device)?;
        // K: [1, num_kv_heads, 1, head_dim]
        let k = Tensor::from_vec(k_f32, (1, self.num_kv_heads, 1, self.head_dim), device)?;
        // V: [1, num_kv_heads, 1, head_dim]
        let v = Tensor::from_vec(v_f32, (1, self.num_kv_heads, 1, self.head_dim), device)?;

        // Concatenate with cache
        let full_k = if let Some(ck) = cache_k {
            Tensor::cat(&[ck, &k], 2)?
        } else {
            k
        };
        let full_v = if let Some(cv) = cache_v {
            Tensor::cat(&[cv, &v], 2)?
        } else {
            v
        };

        // Per-group attention: avoid expand_kv entirely
        // Each group of gqa_repeat query heads shares one KV head
        let mut head_outputs = Vec::with_capacity(self.num_kv_heads);
        for kv_h in 0..self.num_kv_heads {
            let q_start = kv_h * self.gqa_repeat;
            // Q group: [1, gqa_repeat, 1, head_dim]
            let q_group = q.narrow(1, q_start, self.gqa_repeat)?;
            // K/V for this kv head: [1, 1, N+1, head_dim]
            let k_h = full_k.narrow(1, kv_h, 1)?;
            let v_h = full_v.narrow(1, kv_h, 1)?;

            // scores: [1, gqa_repeat, 1, N+1]
            let k_t = k_h.transpose(2, 3)?;
            let scores = (q_group.matmul(&k_t)? / self.scale)?;
            let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
            // out: [1, gqa_repeat, 1, head_dim]
            let out = probs.matmul(&v_h)?;
            head_outputs.push(out);
        }

        // Concatenate all head outputs: [1, num_heads, 1, head_dim]
        let attn_out = Tensor::cat(&head_outputs, 1)?;
        let attn_out = attn_out.reshape((1, 1, self.embed_dim))?;

        // O projection (auto-dispatches to SIMD for single token)
        let output = self.o_proj.forward(&attn_out)?;

        Ok((output, full_k, full_v))
    }
}

/// BitSwiGLU MLP: SiLU(gate(x)) * val(x) -> out
pub struct BitSwiGLU {
    pub w_gate: BitLinear,
    pub w_val: BitLinear,
    pub w_out: BitLinear,
}

/// Fast f32 SiLU: x * sigmoid(x) = x / (1 + exp(-x))
#[inline]
fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

impl BitSwiGLU {
    pub fn load(
        prefix: &str,
        vb: &VarBuilder,
        embed_dim: usize,
        hidden_dim: usize,
    ) -> Result<Self> {
        let w_gate = BitLinear::load_from_vb(
            &format!("{}.w_gate.weight", prefix),
            &format!("{}.w_gate.norm.weight", prefix),
            vb,
            embed_dim,
            hidden_dim,
        )?;
        let w_val = BitLinear::load_from_vb(
            &format!("{}.w_val.weight", prefix),
            &format!("{}.w_val.norm.weight", prefix),
            vb,
            embed_dim,
            hidden_dim,
        )?;
        let w_out = BitLinear::load_from_vb(
            &format!("{}.w_out.weight", prefix),
            &format!("{}.w_out.norm.weight", prefix),
            vb,
            hidden_dim,
            embed_dim,
        )?;
        Ok(Self {
            w_gate,
            w_val,
            w_out,
        })
    }

    /// Forward: w_out(SiLU(w_gate(x)) * w_val(x))
    ///
    /// Auto-dispatches: single token → fully-fused f32 SIMD path,
    /// sequences → tensor matmul with shared RMS normalization.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.dims();
        let is_single = match shape.len() {
            2 => shape[0] == 1,
            3 => shape[0] == 1 && shape[1] == 1,
            _ => false,
        };
        if is_single {
            return self.forward_single(x);
        }

        // Shared RMS factor: compute once, apply with different norm weights
        let x_sq = x.sqr()?;
        let mean_sq = x_sq.mean_keepdim(D::Minus1)?;
        let rms = (mean_sq + 1e-5)?.sqrt()?;
        let x_unit = x.broadcast_div(&rms)?;

        // Apply different norm weights
        let x_gate = x_unit.broadcast_mul(&self.w_gate.rms_norm_weight)?;
        let x_val = x_unit.broadcast_mul(&self.w_val.rms_norm_weight)?;

        let device = x.device();
        let gate_wt = self.w_gate.get_weights_t(device)?;
        let val_wt = self.w_val.get_weights_t(device)?;

        let gate = x_gate.broadcast_matmul(&gate_wt)?;
        let gate = candle_nn::Activation::Silu.forward(&gate)?;
        let val = x_val.broadcast_matmul(&val_wt)?;
        let gated = (gate * val)?;

        // w_out has its own RMSNorm on the gated output
        self.w_out.forward(&gated)
    }

    /// Zero-tensor single-token MLP: entire pipeline in raw f32 + SIMD GEMV.
    /// Avoids ~15 tensor allocations per call.
    fn forward_single(&self, x: &Tensor) -> Result<Tensor> {
        let device = x.device();
        let orig_rank = x.dims().len();
        let x_f32: Vec<f32> = x.flatten_all()?.to_vec1::<f32>()?;

        // Shared RMS norm factor (computed once, reused for gate and val)
        let n = x_f32.len() as f32;
        let sum_sq: f32 = x_f32.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / ((sum_sq / n + 1e-5f32).sqrt());
        let x_unit: Vec<f32> = x_f32.iter().map(|&xi| xi * inv_rms).collect();

        // Gate: apply norm weights + SIMD GEMV + SiLU
        let x_gate: Vec<f32> = x_unit.iter()
            .zip(self.w_gate.rms_norm_weight_f32.iter())
            .map(|(&x, &w)| x * w).collect();
        let mut gate_out = vec![0.0f32; self.w_gate.out_dim];
        self.w_gate.dispatch_packed_gemv(&x_gate, &mut gate_out);

        // Val: apply norm weights + SIMD GEMV
        let x_val: Vec<f32> = x_unit.iter()
            .zip(self.w_val.rms_norm_weight_f32.iter())
            .map(|(&x, &w)| x * w).collect();
        let mut val_out = vec![0.0f32; self.w_val.out_dim];
        self.w_val.dispatch_packed_gemv(&x_val, &mut val_out);

        // SiLU(gate) * val — fused in-place
        let gated: Vec<f32> = gate_out.iter()
            .zip(val_out.iter())
            .map(|(&g, &v)| silu_f32(g) * v).collect();

        // w_out: RMS norm + SIMD GEMV
        let out_normed = rms_norm_vec(&gated, &self.w_out.rms_norm_weight_f32, 1e-5);
        let mut out = vec![0.0f32; self.w_out.out_dim];
        self.w_out.dispatch_packed_gemv(&out_normed, &mut out);

        let out_shape = if orig_rank <= 2 { vec![1, self.w_out.out_dim] } else { vec![1, 1, self.w_out.out_dim] };
        Tensor::from_vec(out, out_shape, device)
    }
}

/// TransformerBlock: pre-norm GQA attention + pre-norm MLP with LayerScale residuals
pub struct TransformerBlock {
    pub ln1: Tensor,
    pub attn: GQAttention,
    pub ls1: LayerScale,
    pub ln2: Tensor,
    pub mlp: BitSwiGLU,
    pub ls2: LayerScale,
    pub embed_dim: usize,
}

impl TransformerBlock {
    pub fn load(
        layer_idx: usize,
        vb: &VarBuilder,
        embed_dim: usize,
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let prefix = format!("layers.{}", layer_idx);

        let ln1 = vb.get((embed_dim,), &format!("{}.ln1.weight", prefix))?;
        let attn = GQAttention::load(
            &format!("{}.attn", prefix), vb, embed_dim, num_heads, num_kv_heads, head_dim,
        )?;
        // V6: key is "ls1" not "ls1.gamma"
        let ls1 = LayerScale::load(&format!("{}.ls1", prefix), vb, embed_dim)?;

        let ln2 = vb.get((embed_dim,), &format!("{}.ln2.weight", prefix))?;
        let mlp = BitSwiGLU::load(&format!("{}.mlp", prefix), vb, embed_dim, hidden_dim)?;
        let ls2 = LayerScale::load(&format!("{}.ls2", prefix), vb, embed_dim)?;

        Ok(Self {
            ln1,
            attn,
            ls1,
            ln2,
            mlp,
            ls2,
            embed_dim,
        })
    }

    /// Forward with RoPE: x + ls1(attn(ln1(x))) + ls2(mlp(ln2(...)))
    pub fn forward(
        &self,
        x: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        causal_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Pre-norm attention with LayerScale residual
        let x_norm = rms_norm(x, &self.ln1, 1e-5)?;
        let attn_out = self.attn.forward(&x_norm, rope_cos, rope_sin, causal_mask)?;
        let attn_out = self.ls1.forward(&attn_out)?;
        let x = (x + attn_out)?;

        // Pre-norm MLP with LayerScale residual
        let x_norm = rms_norm(&x, &self.ln2, 1e-5)?;
        let mlp_out = self.mlp.forward(&x_norm)?;
        let mlp_out = self.ls2.forward(&mlp_out)?;
        let x = (x + mlp_out)?;

        Ok(x)
    }

    /// Forward with KV cache and RoPE for incremental inference.
    /// Returns (output, updated_k, updated_v) for cache update.
    pub fn forward_cached(
        &self,
        x_new: &Tensor,
        cache_k: Option<&Tensor>,
        cache_v: Option<&Tensor>,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        causal_mask: &Tensor,
        total_seq_len: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let x_norm = rms_norm(x_new, &self.ln1, 1e-5)?;
        let (attn_out, new_k, new_v) = self.attn.forward_cached(
            &x_norm, cache_k, cache_v, rope_cos, rope_sin, causal_mask, total_seq_len,
        )?;
        let attn_out = self.ls1.forward(&attn_out)?;
        let x = (x_new + attn_out)?;

        let x_norm = rms_norm(&x, &self.ln2, 1e-5)?;
        let mlp_out = self.mlp.forward(&x_norm)?;
        let mlp_out = self.ls2.forward(&mlp_out)?;
        let x = (x + mlp_out)?;

        Ok((x, new_k, new_v))
    }

    /// Fused single-token cached forward with inline RoPE and per-group attention.
    pub fn forward_cached_single_token(
        &self,
        x_new: &Tensor,
        cache_k: Option<&Tensor>,
        cache_v: Option<&Tensor>,
        rope_cos_f32: &[f32],
        rope_sin_f32: &[f32],
        pos: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let x_norm = rms_norm(x_new, &self.ln1, 1e-5)?;
        let (attn_out, new_k, new_v) = self.attn.forward_cached_single_token(
            &x_norm, cache_k, cache_v, rope_cos_f32, rope_sin_f32, pos,
        )?;
        let attn_out = self.ls1.forward(&attn_out)?;
        let x = (x_new + attn_out)?;

        let x_norm = rms_norm(&x, &self.ln2, 1e-5)?;
        let mlp_out = self.mlp.forward(&x_norm)?;
        let mlp_out = self.ls2.forward(&mlp_out)?;
        let x = (x + mlp_out)?;

        Ok((x, new_k, new_v))
    }
}
