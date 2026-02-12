// src/engine/mod.rs
//! v6.0.0 KiyNet_V6 neural network engine for chess evaluation
//!
//! Architecture: Embedding → 12× TransformerBlock(GQA+RoPE) → ln_f → PolicyHead / ValueHead
//! TransformerBlock: RMSNorm → GQAttention(BitLinear Q/K/V/O + RoPE) → LayerScale → RMSNorm → BitSwiGLU → LayerScale

pub mod bitlinear;
pub mod gguf;
pub mod heads;
pub mod model;
pub mod transformer;

#[cfg(test)]
mod tests;

pub use bitlinear::{rms_norm, BitLinear};
pub use heads::{PolicyHead, ValueHead};
pub use model::{ChessModel, ModelConfig};
pub use transformer::{precompute_rope, KVCache, TransformerBlock};

use crate::constants::{
    CONTEXT_LENGTH, DEFAULT_MODEL_PATH, D_MODEL, HIDDEN_DIM, NUM_HEADS, NUM_KV_HEADS, NUM_LAYERS,
    ROPE_THETA, VOCAB_SIZE,
};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};
use std::collections::HashMap;

/// Embedded GGUF model (compiled into binary for single-file distribution)
pub static EMBEDDED_GGUF: &[u8] = include_bytes!("../../kiyengine.gguf");

/// Main engine struct containing the neural network
pub struct Engine {
    pub embedding: Embedding,
    pub layers: Vec<TransformerBlock>,
    pub ln_f: Tensor,
    pub policy_head: PolicyHead,
    pub value_head: ValueHead,
    pub device: Device,
    /// Pre-computed causal mask [1, 1, ctx_len, ctx_len] (avoids per-call allocation)
    pub causal_mask: Tensor,
    /// Pre-computed RoPE cos/sin [max_len, head_dim/2]
    pub rope_cos: Tensor,
    pub rope_sin: Tensor,
    /// Cached f32 RoPE tables for zero-tensor single-token path [max_len * half_dim]
    pub rope_cos_f32: Vec<f32>,
    pub rope_sin_f32: Vec<f32>,
    // Runtime config (read from GGUF metadata)
    pub d_model: usize,
    pub context_length: usize,
    pub head_dim: usize,
}

impl Engine {
    /// Create a new engine — loads GGUF model from disk or embedded bytes
    pub fn new() -> anyhow::Result<Self> {
        let device = if cfg!(feature = "cuda") {
            Device::new_cuda(0).unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        };

        if std::path::Path::new(DEFAULT_MODEL_PATH).exists() {
            eprintln!("info string Loading GGUF model: {}", DEFAULT_MODEL_PATH);
            return Self::load_from_gguf(DEFAULT_MODEL_PATH, device);
        }

        // Fall back to embedded GGUF
        eprintln!("info string Loading GGUF model: embedded");
        Self::load_from_embedded(device)
    }

    /// Load model from embedded GGUF bytes compiled into the binary
    pub fn load_from_embedded(device: Device) -> anyhow::Result<Self> {
        use crate::engine::gguf::GgufFile;
        let gguf = GgufFile::from_static_bytes(EMBEDDED_GGUF)?;
        Self::load_from_gguf_parsed(gguf, device)
    }

    /// Load model weights from GGUF file on disk
    pub fn load_from_gguf(model_path: &str, device: Device) -> anyhow::Result<Self> {
        use crate::engine::gguf::GgufFile;
        let gguf = GgufFile::from_path(model_path)?;
        Self::load_from_gguf_parsed(gguf, device)
    }

    /// Load model weights from a parsed GgufFile
    fn load_from_gguf_parsed(
        gguf: crate::engine::gguf::GgufFile,
        device: Device,
    ) -> anyhow::Result<Self> {
        use crate::engine::gguf::GgmlQuantizationType;

        // Extract configuration from GGUF metadata
        let vocab_size = gguf
            .get_u32("kiyengine.vocab_size")
            .unwrap_or(VOCAB_SIZE as u32) as usize;
        let context_length = gguf
            .get_u32("kiyengine.context_length")
            .unwrap_or(CONTEXT_LENGTH as u32) as usize;
        let d_model = gguf
            .get_u32("kiyengine.embedding_length")
            .unwrap_or(D_MODEL as u32) as usize;
        let num_layers = gguf
            .get_u32("kiyengine.block_count")
            .unwrap_or(NUM_LAYERS as u32) as usize;
        let num_heads = gguf
            .get_u32("kiyengine.attention.head_count")
            .or_else(|| gguf.get_u32("kiyengine.head_count"))
            .unwrap_or(NUM_HEADS as u32) as usize;
        let num_kv_heads = gguf
            .get_u32("kiyengine.attention.head_count_kv")
            .unwrap_or(NUM_KV_HEADS as u32) as usize;
        let hidden_dim = gguf
            .get_u32("kiyengine.feed_forward_length")
            .unwrap_or(HIDDEN_DIM as u32) as usize;
        let head_dim = d_model / num_heads;

        println!(
            "  Config: d_model={}, layers={}, heads={}, kv_heads={}, head_dim={}, hidden={}, ctx={}, vocab={}",
            d_model, num_layers, num_heads, num_kv_heads, head_dim, hidden_dim, context_length, vocab_size
        );

        // Convert all GGUF tensors to f32 Candle tensors
        let mut tensors = HashMap::new();

        for tensor_info in &gguf.tensors {
            let tensor_data = gguf.get_tensor_data(tensor_info);
            let shape: Vec<usize> = tensor_info.dimensions.iter().map(|&d| d as usize).collect();

            let tensor = match tensor_info.quantization {
                GgmlQuantizationType::I8 => {
                    let i8_data: &[i8] = bytemuck::cast_slice(tensor_data);
                    let f32_data: Vec<f32> = i8_data.iter().map(|&v| v as f32).collect();
                    Tensor::from_vec(f32_data, shape, &device)?
                }
                GgmlQuantizationType::F16 => {
                    let f16_data: &[half::f16] = bytemuck::cast_slice(tensor_data);
                    let f32_data: Vec<f32> = f16_data.iter().map(|&v| v.to_f32()).collect();

                    Tensor::from_vec(f32_data, shape, &device)?
                }
                GgmlQuantizationType::F32 => {
                    let f32_data: &[f32] = bytemuck::cast_slice(tensor_data);
                    Tensor::from_vec(f32_data.to_vec(), shape, &device)?
                }
                _ => {
                    let f32_data: Vec<f32> = tensor_data.iter().map(|&v| v as f32).collect();
                    Tensor::from_vec(f32_data, shape, &device)?
                }
            };

            tensors.insert(tensor_info.name.clone(), tensor);
        }

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        // Load embedding
        let embed_w = vb.get((vocab_size, d_model), "embed.weight")?;
        let embedding = Embedding::new(embed_w, d_model);

        // V6: No positional embedding — using RoPE instead
        // Precompute RoPE cos/sin for max context length
        let (rope_cos, rope_sin) = precompute_rope(head_dim, context_length, ROPE_THETA, &device)?;
        let rope_cos_f32 = rope_cos.flatten_all()?.to_vec1::<f32>()?;
        let rope_sin_f32 = rope_sin.flatten_all()?.to_vec1::<f32>()?;

        // Load transformer blocks (with GQA params)
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(TransformerBlock::load(
                i,
                &vb,
                d_model,
                hidden_dim,
                num_heads,
                num_kv_heads,
                head_dim,
            )?);
        }

        // Load final layer norm
        let ln_f = vb.get((d_model,), "ln_f.weight")?;

        // Load heads
        let policy_head = PolicyHead::load(&vb, d_model, vocab_size)?;
        let value_head = ValueHead::load(&vb, d_model)?;

        // Pre-compute causal mask for max context length (upper triangle = -inf)
        let causal_mask = {
            let mask: Vec<f32> = (0..context_length)
                .flat_map(|i| {
                    (0..context_length).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 })
                })
                .collect();
            Tensor::from_vec(mask, (1, 1, context_length, context_length), &device)?
        };

        // Pre-warm all BitLinear weight caches (avoids lazy init on first inference)
        for layer in &layers {
            let _ = layer.attn.q_proj.get_weights_t(&device);
            let _ = layer.attn.k_proj.get_weights_t(&device);
            let _ = layer.attn.v_proj.get_weights_t(&device);
            let _ = layer.attn.o_proj.get_weights_t(&device);
            let _ = layer.mlp.w_gate.get_weights_t(&device);
            let _ = layer.mlp.w_val.get_weights_t(&device);
            let _ = layer.mlp.w_out.get_weights_t(&device);
        }
        // Pre-warm head BitLinear caches
        let _ = policy_head.layer.get_weights_t(&device);
        let _ = value_head.layer1.get_weights_t(&device);
        let _ = value_head.layer2.get_weights_t(&device);

        println!(
            "  Model loaded successfully ({} layers, GQA {}/{})",
            num_layers, num_heads, num_kv_heads
        );

        Ok(Self {
            embedding,
            layers,
            ln_f,
            policy_head,
            value_head,
            device,
            causal_mask,
            rope_cos,
            rope_sin,
            rope_cos_f32,
            rope_sin_f32,
            d_model,
            context_length,
            head_dim,
        })
    }

    /// Create a new empty KV cache sized for this model
    pub fn create_kv_cache(&self) -> KVCache {
        KVCache::new(self.layers.len())
    }

    /// Number of transformer layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Run forward pass with KV cache for incremental inference.
    ///
    /// If cache contains a valid prefix of `tokens`, only the new tokens are processed.
    /// Otherwise falls back to full forward pass and populates the cache.
    ///
    /// Returns (policy_logits, value_score) where value_score is in range [-1, 1]
    pub fn forward_with_cache(
        &self,
        tokens: &[usize],
        cache: &mut KVCache,
    ) -> Result<(Tensor, f32)> {
        let padded;
        let tokens = if tokens.is_empty() {
            padded = vec![0usize];
            &padded[..]
        } else {
            tokens
        };

        let seq_len = tokens.len();
        let take_len = seq_len.min(self.context_length);
        let tokens_slice = &tokens[seq_len - take_len..];

        // Check if we can reuse the cache (prefix match)
        if cache.can_reuse(tokens_slice) {
            let cached_len = cache.seq_len;
            let new_count = take_len - cached_len;
            let new_tokens_slice = &tokens_slice[cached_len..];

            // Embed only new tokens (no positional embedding — RoPE handles positions)
            let new_tokens_u32: Vec<u32> = new_tokens_slice.iter().map(|&t| t as u32).collect();
            let new_tensor = Tensor::new(new_tokens_u32.as_slice(), &self.device)?.unsqueeze(0)?;
            let mut x = self.embedding.forward(&new_tensor)?;

            // Process through layers with KV cache + RoPE
            if new_count == 1 {
                // Fused single-token path: inline RoPE on f32, per-group attention
                let pos = cached_len; // position of the new token
                for (i, layer) in self.layers.iter().enumerate() {
                    let (ck, cv) = cache.layers[i]
                        .as_ref()
                        .map(|(k, v)| (Some(k), Some(v)))
                        .unwrap_or((None, None));
                    let (out, new_k, new_v) = layer.forward_cached_single_token(
                        &x,
                        ck,
                        cv,
                        &self.rope_cos_f32,
                        &self.rope_sin_f32,
                        pos,
                    )?;
                    x = out;
                    cache.layers[i] = Some((new_k, new_v));
                }
            } else {
                let total_seq = take_len;
                for (i, layer) in self.layers.iter().enumerate() {
                    let (ck, cv) = cache.layers[i]
                        .as_ref()
                        .map(|(k, v)| (Some(k), Some(v)))
                        .unwrap_or((None, None));
                    let (out, new_k, new_v) = layer.forward_cached(
                        &x,
                        ck,
                        cv,
                        &self.rope_cos,
                        &self.rope_sin,
                        &self.causal_mask,
                        total_seq,
                    )?;
                    x = out;
                    cache.layers[i] = Some((new_k, new_v));
                }
            }

            // Update cache metadata
            cache.seq_len = take_len;
            cache.tokens = tokens_slice.to_vec();

            // Extract last token, apply final layer norm, run heads
            let last_x = if new_count == 1 {
                x.squeeze(1)?
            } else {
                x.narrow(1, new_count - 1, 1)?.squeeze(1)?
            };
            let last_x = rms_norm(&last_x, &self.ln_f, 1e-5)?;

            let policy_logits = self.policy_head.forward(&last_x)?.squeeze(0)?;
            let value_out = self.value_head.forward(&last_x)?.squeeze(0)?.squeeze(0)?;
            let eval_score = value_out.tanh()?.to_scalar::<f32>()?;

            return Ok((policy_logits, eval_score));
        }

        // Cache miss — full forward pass, then populate cache
        cache.clear();

        let tokens_u32: Vec<u32> = tokens_slice.iter().map(|&t| t as u32).collect();
        let tokens_tensor = Tensor::new(tokens_u32.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut x = self.embedding.forward(&tokens_tensor)?;

        // Full forward with cache population (RoPE applied inside each layer)
        for (i, layer) in self.layers.iter().enumerate() {
            let (out, new_k, new_v) = layer.forward_cached(
                &x,
                None,
                None,
                &self.rope_cos,
                &self.rope_sin,
                &self.causal_mask,
                take_len,
            )?;
            x = out;
            cache.layers[i] = Some((new_k, new_v));
        }

        cache.seq_len = take_len;
        cache.tokens = tokens_slice.to_vec();

        let last_token_x = x.narrow(1, take_len - 1, 1)?.squeeze(1)?;
        let last_token_x = rms_norm(&last_token_x, &self.ln_f, 1e-5)?;

        let policy_logits = self.policy_head.forward(&last_token_x)?.squeeze(0)?;
        let value_out = self
            .value_head
            .forward(&last_token_x)?
            .squeeze(0)?
            .squeeze(0)?;
        let eval_score = value_out.tanh()?.to_scalar::<f32>()?;

        Ok((policy_logits, eval_score))
    }

    /// Run forward pass on token sequence (no cache)
    ///
    /// Returns (policy_logits, value_score) where value_score is in range [-1, 1]
    pub fn forward(&self, tokens: &[usize]) -> Result<(Tensor, f32)> {
        let padded;
        let tokens = if tokens.is_empty() {
            padded = vec![0usize];
            &padded[..]
        } else {
            tokens
        };

        let seq_len = tokens.len();
        let take_len = seq_len.min(self.context_length);
        let tokens_slice = &tokens[seq_len - take_len..];

        let tokens_u32: Vec<u32> = tokens_slice.iter().map(|&t| t as u32).collect();
        let tokens_tensor = Tensor::new(tokens_u32.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut x = self.embedding.forward(&tokens_tensor)?;

        // V6: No positional embedding — RoPE applied inside each attention layer
        let mask = if take_len > 1 {
            Some(&self.causal_mask)
        } else {
            None
        };
        for layer in &self.layers {
            x = layer.forward(&x, &self.rope_cos, &self.rope_sin, mask)?;
        }

        // Extract last token, apply final layer norm
        let last_token_x = x.narrow(1, take_len - 1, 1)?.squeeze(1)?;
        let last_token_x = rms_norm(&last_token_x, &self.ln_f, 1e-5)?;

        // Policy and value heads
        let policy_logits = self.policy_head.forward(&last_token_x)?.squeeze(0)?;
        let value_out = self
            .value_head
            .forward(&last_token_x)?
            .squeeze(0)?
            .squeeze(0)?;
        let eval_score = value_out.tanh()?.to_scalar::<f32>()?;

        Ok((policy_logits, eval_score))
    }
}
