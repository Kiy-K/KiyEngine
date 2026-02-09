// src/engine/mod.rs
//! v5.2.0 KiyNet_Ultimate neural network engine for chess evaluation
//!
//! Architecture: Embedding → Positional → 8× TransformerBlock → ln_f → PolicyHead / ValueHead
//! TransformerBlock: RMSNorm → MultiheadAttention → LayerScale → RMSNorm → BitSwiGLU → LayerScale

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
pub use transformer::{KVCache, TransformerBlock};

use crate::constants::{
    CONTEXT_LENGTH, DEFAULT_MODEL_PATH, D_MODEL, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, VOCAB_SIZE,
};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};
use std::collections::HashMap;

/// Main engine struct containing the neural network
pub struct Engine {
    pub embedding: Embedding,
    pub pos_embed: Tensor,
    pub layers: Vec<TransformerBlock>,
    pub ln_f: Tensor,
    pub policy_head: PolicyHead,
    pub value_head: ValueHead,
    pub device: Device,
    /// Pre-computed causal mask [1, 1, ctx_len, ctx_len] (avoids per-call allocation)
    pub causal_mask: Tensor,
    // Runtime config (read from GGUF metadata)
    pub d_model: usize,
    pub context_length: usize,
}

impl Engine {
    /// Create a new engine — loads GGUF model
    pub fn new() -> anyhow::Result<Self> {
        let device = if cfg!(feature = "cuda") {
            Device::new_cuda(0).unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        };

        if std::path::Path::new(DEFAULT_MODEL_PATH).exists() {
            println!("Loading v5.2.0 GGUF model: {}", DEFAULT_MODEL_PATH);
            return Self::load_from_gguf(DEFAULT_MODEL_PATH, device);
        }

        anyhow::bail!("No model file found. Expected '{}'", DEFAULT_MODEL_PATH,)
    }

    /// Load model weights from GGUF file (v5.2.0 KiyNet_Ultimate format)
    pub fn load_from_gguf(model_path: &str, device: Device) -> anyhow::Result<Self> {
        use crate::engine::gguf::GgmlQuantizationType;
        use crate::engine::gguf::GgufFile;

        let gguf = GgufFile::from_path(model_path)?;

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
            .get_u32("kiyengine.head_count")
            .unwrap_or(NUM_HEADS as u32) as usize;
        let hidden_dim = gguf
            .get_u32("kiyengine.feed_forward_length")
            .unwrap_or(HIDDEN_DIM as u32) as usize;

        println!(
            "  Config: d_model={}, layers={}, heads={}, hidden={}, ctx={}, vocab={}",
            d_model, num_layers, num_heads, hidden_dim, context_length, vocab_size
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

        // Load positional embedding [1, ctx_len, d_model]
        // (GGUF loader already reverses dims from GGML order)
        let pos_embed = vb.get((1, context_length, d_model), "pos_embed")?;

        // Load transformer blocks
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(TransformerBlock::load(
                i, &vb, d_model, hidden_dim, num_heads,
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
            let _ = layer.mlp.w_gate.get_weights_t(&device);
            let _ = layer.mlp.w_val.get_weights_t(&device);
            let _ = layer.mlp.w_out.get_weights_t(&device);
        }

        println!("  Model loaded successfully ({} layers)", num_layers);

        Ok(Self {
            embedding,
            pos_embed,
            layers,
            ln_f,
            policy_head,
            value_head,
            device,
            causal_mask,
            d_model,
            context_length,
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

            // Embed only new tokens
            let new_tokens_u32: Vec<u32> = new_tokens_slice.iter().map(|&t| t as u32).collect();
            let new_tensor = Tensor::new(new_tokens_u32.as_slice(), &self.device)?.unsqueeze(0)?;
            let mut x = self.embedding.forward(&new_tensor)?;

            // Positional embedding for new positions only
            let pos_slice = self.pos_embed.narrow(1, cached_len, new_count)?;
            x = x.broadcast_add(&pos_slice)?;

            // Process through layers with KV cache
            let total_seq = take_len;
            for (i, layer) in self.layers.iter().enumerate() {
                let (ck, cv) = cache.layers[i]
                    .as_ref()
                    .map(|(k, v)| (Some(k), Some(v)))
                    .unwrap_or((None, None));
                let (out, new_k, new_v) =
                    layer.forward_cached(&x, ck, cv, &self.causal_mask, total_seq)?;
                x = out;
                cache.layers[i] = Some((new_k, new_v));
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

        let pos_slice = self.pos_embed.narrow(1, 0, take_len)?;
        x = x.broadcast_add(&pos_slice)?;

        // Full forward with cache population
        for (i, layer) in self.layers.iter().enumerate() {
            let (out, new_k, new_v) =
                layer.forward_cached(&x, None, None, &self.causal_mask, take_len)?;
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

        // Add positional embedding
        let pos_slice = self.pos_embed.narrow(1, 0, take_len)?;
        x = x.broadcast_add(&pos_slice)?;

        // Transformer layers (attention + BitSwiGLU MLP)
        let mask = if take_len > 1 {
            Some(&self.causal_mask)
        } else {
            None
        };
        for layer in &self.layers {
            x = layer.forward(&x, mask)?;
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
