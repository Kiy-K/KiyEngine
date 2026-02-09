// src/engine/transformer.rs
//! v5.2.0 KiyNet_Ultimate Transformer components:
//! - MultiheadAttention (standard QKV with causal mask)
//! - LayerScale (learnable per-channel scaling)
//! - BitSwiGLU (gated MLP with ternary BitLinear layers)
//! - TransformerBlock (attention + MLP with pre-norm residual)

use candle_core::{Result, Tensor, D};
use candle_nn::{Module, VarBuilder};

use crate::engine::bitlinear::{rms_norm, BitLinear};

/// KV Cache for incremental transformer inference.
/// Stores per-layer K and V tensors so only new tokens need QKV projection.
pub struct KVCache {
    /// Per-layer (K, V) caches
    /// K: [batch, num_heads, cached_seq_len, head_dim]
    /// V: [batch, num_heads, cached_seq_len, head_dim]
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

/// MultiheadAttention with packed QKV projection
pub struct MultiheadAttention {
    /// Packed QKV weight transposed: [embed_dim, 3*embed_dim] (pre-computed, contiguous)
    pub in_proj_weight_t: Tensor,
    /// Packed QKV bias: [3*embed_dim]
    pub in_proj_bias: Tensor,
    /// Output projection weight transposed: [embed_dim, embed_dim] (pre-computed, contiguous)
    pub out_proj_weight_t: Tensor,
    /// Output projection bias: [embed_dim]
    pub out_proj_bias: Tensor,
    pub num_heads: usize,
    pub head_dim: usize,
    pub embed_dim: usize,
    pub scale: f64,
}

impl MultiheadAttention {
    pub fn load(prefix: &str, vb: &VarBuilder, embed_dim: usize, num_heads: usize) -> Result<Self> {
        let head_dim = embed_dim / num_heads;
        let three_dim = 3 * embed_dim;

        // GGUF reversed dims: in_proj_weight is [3*embed_dim, embed_dim] (standard PyTorch layout)
        let in_proj_weight = vb.get(
            (three_dim, embed_dim),
            &format!("{}.in_proj_weight", prefix),
        )?;
        // Pre-compute transpose + contiguous for fast matmul (avoid per-call transpose)
        let in_proj_weight_t = in_proj_weight.t()?.contiguous()?;
        let in_proj_bias = vb.get((three_dim,), &format!("{}.in_proj_bias", prefix))?;

        // out_proj.weight is [embed_dim, embed_dim] — pre-transpose + contiguous
        let out_w = vb.get(
            (embed_dim, embed_dim),
            &format!("{}.out_proj.weight", prefix),
        )?;
        let out_proj_weight_t = out_w.t()?.contiguous()?;
        let out_proj_bias = vb.get((embed_dim,), &format!("{}.out_proj.bias", prefix))?;

        Ok(Self {
            in_proj_weight_t,
            in_proj_bias,
            out_proj_weight_t,
            out_proj_bias,
            num_heads,
            head_dim,
            embed_dim,
            scale: (head_dim as f64).sqrt(),
        })
    }

    /// Forward: self-attention with pre-computed causal mask
    /// x: [batch, seq_len, embed_dim]
    /// causal_mask: [1, 1, max_seq, max_seq] or None for single token
    pub fn forward(&self, x: &Tensor, causal_mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        // QKV projection using pre-transposed weight: x @ W^T + bias
        let qkv = x.broadcast_matmul(&self.in_proj_weight_t)?;
        let qkv = qkv.broadcast_add(&self.in_proj_bias)?;

        // Split into Q, K, V: each [batch, seq_len, embed_dim]
        let q = qkv.narrow(D::Minus1, 0, self.embed_dim)?;
        let k = qkv.narrow(D::Minus1, self.embed_dim, self.embed_dim)?;
        let v = qkv.narrow(D::Minus1, 2 * self.embed_dim, self.embed_dim)?;

        // Reshape to [batch, num_heads, seq_len, head_dim]
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let attn_weights = q.matmul(&k_t)?;
        let attn_weights = (attn_weights / self.scale)?;

        // Apply causal mask if provided (seq_len > 1)
        let attn_weights = if let Some(mask) = causal_mask {
            let mask = mask.narrow(2, 0, seq_len)?.narrow(3, 0, seq_len)?;
            let masked = attn_weights.broadcast_add(&mask)?;
            candle_nn::ops::softmax(&masked, D::Minus1)?
        } else {
            candle_nn::ops::softmax(&attn_weights, D::Minus1)?
        };

        // Attention output → reshape → output projection
        let attn_out = attn_weights.matmul(&v)?;
        let attn_out =
            attn_out
                .transpose(1, 2)?
                .contiguous()?
                .reshape((batch, seq_len, self.embed_dim))?;
        let out = attn_out.broadcast_matmul(&self.out_proj_weight_t)?;
        out.broadcast_add(&self.out_proj_bias)
    }

    /// Forward with KV cache: only processes new tokens, reuses cached K,V.
    /// Returns (output, new_k, new_v) where new_k/new_v are the FULL K,V
    /// (cached + new) for updating the cache.
    ///
    /// x_new: [batch, new_tokens, embed_dim] — only the NEW tokens
    /// cache_k: [batch, num_heads, cached_len, head_dim] (or None)
    /// cache_v: [batch, num_heads, cached_len, head_dim] (or None)
    /// causal_mask: pre-computed [1, 1, max_seq, max_seq]
    /// total_seq_len: cached_len + new_tokens (for causal mask slicing)
    pub fn forward_cached(
        &self,
        x_new: &Tensor,
        cache_k: Option<&Tensor>,
        cache_v: Option<&Tensor>,
        causal_mask: &Tensor,
        total_seq_len: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, new_tokens, _) = x_new.dims3()?;

        // QKV projection for NEW tokens only
        let qkv = x_new.broadcast_matmul(&self.in_proj_weight_t)?;
        let qkv = qkv.broadcast_add(&self.in_proj_bias)?;

        let q_new = qkv.narrow(D::Minus1, 0, self.embed_dim)?;
        let k_new = qkv.narrow(D::Minus1, self.embed_dim, self.embed_dim)?;
        let v_new = qkv.narrow(D::Minus1, 2 * self.embed_dim, self.embed_dim)?;

        // Reshape to [batch, num_heads, new_tokens, head_dim]
        let q_new = q_new
            .reshape((batch, new_tokens, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k_new = k_new
            .reshape((batch, new_tokens, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v_new = v_new
            .reshape((batch, new_tokens, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Concatenate with cached K,V along seq dimension
        let full_k = if let Some(ck) = cache_k {
            Tensor::cat(&[ck, &k_new], 2)?
        } else {
            k_new
        };
        let full_v = if let Some(cv) = cache_v {
            Tensor::cat(&[cv, &v_new], 2)?
        } else {
            v_new
        };

        // Attention: Q_new attends to ALL K (cached + new)
        // Q: [batch, heads, new_tokens, head_dim]
        // K^T: [batch, heads, head_dim, total_seq_len]
        let k_t = full_k.transpose(2, 3)?.contiguous()?;
        let attn_weights = q_new.matmul(&k_t)?;
        let attn_weights = (attn_weights / self.scale)?;

        // Causal mask: only the rows for the new positions
        // New tokens are at positions [cached_len .. total_seq_len)
        let cached_len = total_seq_len - new_tokens;
        let mask_rows = causal_mask.narrow(2, cached_len, new_tokens)?;
        let mask_cols = mask_rows.narrow(3, 0, total_seq_len)?;
        let attn_weights = attn_weights.broadcast_add(&mask_cols)?;
        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;

        // Weighted sum: [batch, heads, new_tokens, head_dim]
        let attn_out = attn_weights.matmul(&full_v)?;
        let attn_out =
            attn_out
                .transpose(1, 2)?
                .contiguous()?
                .reshape((batch, new_tokens, self.embed_dim))?;
        let output = attn_out.broadcast_matmul(&self.out_proj_weight_t)?;
        let output = output.broadcast_add(&self.out_proj_bias)?;

        Ok((output, full_k, full_v))
    }
}

/// BitSwiGLU MLP: SiLU(gate(x)) * val(x) -> out
pub struct BitSwiGLU {
    pub w_gate: BitLinear,
    pub w_val: BitLinear,
    pub w_out: BitLinear,
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
    /// Optimized: gate and val share the same input x, so the RMS normalization
    /// denominator is computed once and reused with both norm weights.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
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
}

/// TransformerBlock: pre-norm attention + pre-norm MLP with LayerScale residuals
pub struct TransformerBlock {
    pub ln1: Tensor,
    pub attn: MultiheadAttention,
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
    ) -> Result<Self> {
        let prefix = format!("layers.{}", layer_idx);

        let ln1 = vb.get((embed_dim,), &format!("{}.ln1.weight", prefix))?;
        let attn = MultiheadAttention::load(&format!("{}.attn", prefix), vb, embed_dim, num_heads)?;
        let ls1 = LayerScale::load(&format!("{}.ls1.gamma", prefix), vb, embed_dim)?;

        let ln2 = vb.get((embed_dim,), &format!("{}.ln2.weight", prefix))?;
        let mlp = BitSwiGLU::load(&format!("{}.mlp", prefix), vb, embed_dim, hidden_dim)?;
        let ls2 = LayerScale::load(&format!("{}.ls2.gamma", prefix), vb, embed_dim)?;

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

    /// Forward: x + ls1(attn(ln1(x))) + ls2(mlp(ln2(x + ls1(attn(ln1(x))))))
    /// causal_mask: pre-computed [1, 1, max_seq, max_seq] or None for single token
    pub fn forward(&self, x: &Tensor, causal_mask: Option<&Tensor>) -> Result<Tensor> {
        // Pre-norm attention with LayerScale residual
        let x_norm = rms_norm(x, &self.ln1, 1e-5)?;
        let attn_out = self.attn.forward(&x_norm, causal_mask)?;
        let attn_out = self.ls1.forward(&attn_out)?;
        let x = (x + attn_out)?;

        // Pre-norm MLP with LayerScale residual
        let x_norm = rms_norm(&x, &self.ln2, 1e-5)?;
        let mlp_out = self.mlp.forward(&x_norm)?;
        let mlp_out = self.ls2.forward(&mlp_out)?;
        let x = (x + mlp_out)?;

        Ok(x)
    }

    /// Forward with KV cache: only processes new tokens through the layer.
    /// Returns (output, updated_k, updated_v) for cache update.
    pub fn forward_cached(
        &self,
        x_new: &Tensor,
        cache_k: Option<&Tensor>,
        cache_v: Option<&Tensor>,
        causal_mask: &Tensor,
        total_seq_len: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // Pre-norm attention with KV cache
        let x_norm = rms_norm(x_new, &self.ln1, 1e-5)?;
        let (attn_out, new_k, new_v) =
            self.attn
                .forward_cached(&x_norm, cache_k, cache_v, causal_mask, total_seq_len)?;
        let attn_out = self.ls1.forward(&attn_out)?;
        let x = (x_new + attn_out)?;

        // MLP is position-wise — only processes new tokens
        let x_norm = rms_norm(&x, &self.ln2, 1e-5)?;
        let mlp_out = self.mlp.forward(&x_norm)?;
        let mlp_out = self.ls2.forward(&mlp_out)?;
        let x = (x + mlp_out)?;

        Ok((x, new_k, new_v))
    }
}
