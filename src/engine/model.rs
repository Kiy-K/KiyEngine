// src/engine/model.rs
//! ChessModel: High-level BitNet 1.58-bit inference API for chess move prediction
//!
//! Provides a clean interface for:
//! - Model loading with metadata-aware scale factors
//! - Inference with softmax and argmax
//! - Policy and value head outputs

use crate::constants::{CONTEXT_LENGTH, D_MODEL, NUM_LAYERS, VOCAB_SIZE};
use crate::engine::bitlinear::{rms_norm, BitLinear};
use crate::engine::gguf::{GgmlQuantizationType, GgufFile};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{ops::softmax, Embedding, Linear, Module, VarBuilder};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;

/// Configuration for model architecture
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub context_length: usize,
    pub d_model: usize,
    pub num_layers: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: VOCAB_SIZE,
            context_length: CONTEXT_LENGTH,
            d_model: D_MODEL,
            num_layers: NUM_LAYERS,
        }
    }
}

/// Output from the chess model inference
#[derive(Debug)]
pub struct InferenceOutput {
    /// Policy logits for all possible moves [vocab_size]
    pub policy_logits: Tensor,
    /// Value estimate in range [-1, 1] (win probability)
    pub value: f32,
    /// Index of the most likely move (argmax of policy)
    pub best_move_idx: usize,
    /// Confidence score (softmax probability of best move)
    pub confidence: f32,
}

/// ChessModel: Complete BitNet 1.58-bit transformer for chess move prediction
///
/// Architecture:
/// - Token embedding
/// - Positional embedding
/// - N BitLinear layers (BitNet 1.58-bit) with RMSNorm
/// - Policy head (outputs move probabilities)
/// - Value head (outputs position evaluation)
pub struct ChessModel {
    config: ModelConfig,
    device: Device,

    // Embeddings
    token_embed: Embedding,
    pos_embed: Tensor,

    // Transformer layers (mutable for cache management)
    layers: Vec<BitLinear>,

    // Output heads
    policy_norm: Tensor,
    policy_linear: Linear,

    value_norm: Tensor,
    value_linear1: Linear,
    value_linear2: Linear,
}

impl ChessModel {
    /// Load a ChessModel from a safetensors file
    ///
    /// Automatically detects and parses scale factors from metadata
    pub fn from_safetensors<P: AsRef<std::path::Path>>(
        path: P,
        device: Device,
    ) -> anyhow::Result<Self> {
        let file = File::open(path.as_ref())?;

        // SAFETY: Memory mapping is safe for read-only files
        let mmap = unsafe { Mmap::map(&file)? };

        // Load tensors
        let tensors = candle_core::safetensors::load_buffer(&mmap, &device)?;

        // Parse metadata for BitNet scale factors
        let metadata = Self::parse_metadata(&mmap)?;

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        Self::from_varbuilder(vb, device, &metadata, ModelConfig::default())
    }

    /// Load a ChessModel from a GGUF file (V5 format)
    ///
    /// Supports BitNet 1.58-bit quantized models with fast memory-mapped loading
    pub fn from_gguf<P: AsRef<std::path::Path>>(path: P, device: Device) -> anyhow::Result<Self> {
        // Load GGUF file with memory mapping
        let gguf = GgufFile::from_path(path)?;

        // Extract metadata for model configuration
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

        let config = ModelConfig {
            vocab_size,
            context_length,
            d_model,
            num_layers,
        };

        // Build tensors from GGUF
        let mut tensors = HashMap::new();
        let mut metadata: HashMap<String, String> = HashMap::new();

        // Convert each tensor from GGUF format to Candle tensors
        for tensor_info in &gguf.tensors {
            let tensor_data = gguf.get_tensor_data(tensor_info);
            let shape: Vec<usize> = tensor_info.dimensions.iter().map(|&d| d as usize).collect();

            let tensor = match tensor_info.quantization {
                GgmlQuantizationType::I8 => {
                    // BitNet 1.58-bit ternary weights: convert i8 to f32
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
                    // Default: convert bytes to f32
                    let f32_data: Vec<f32> = tensor_data.iter().map(|&v| v as f32).collect();
                    Tensor::from_vec(f32_data, shape, &device)?
                }
            };

            tensors.insert(tensor_info.name.clone(), tensor);
        }

        // Extract scale factors from GGUF metadata
        for i in 0..num_layers {
            let scale_key = format!("kiyengine.layers.{}.linear.scale", i);
            if let Some(scale) = gguf.get_f32(&scale_key) {
                let meta_key = format!("layers.{}.linear_scale", i);
                metadata.insert(meta_key, scale.to_string());
            }
        }

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        Self::from_varbuilder(vb, device, &metadata, config)
    }

    /// Create model from VarBuilder with metadata
    fn from_varbuilder(
        vb: VarBuilder,
        device: Device,
        metadata: &HashMap<String, String>,
        config: ModelConfig,
    ) -> anyhow::Result<Self> {
        // Load token embedding
        let token_embed_w = vb.get((config.vocab_size, config.d_model), "embed.weight")?;
        let token_embed = Embedding::new(token_embed_w, config.d_model);

        // Load positional embedding
        let pos_embed = vb.get((1, config.context_length, config.d_model), "pos_embed")?;

        // Load BitLinear layers with scale factors from metadata
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let prefix = format!("layers.{}.linear", i);
            layers.push(BitLinear::load(
                &prefix,
                &vb,
                config.d_model,
                config.d_model,
                metadata,
            )?);
        }

        // Load policy head
        let policy_norm = vb.get((config.d_model,), "policy_head.0.weight")?;
        let policy_w = vb.get((config.vocab_size, config.d_model), "policy_head.1.weight")?;
        let policy_linear = Linear::new(policy_w, None);

        // Load value head
        let value_norm = vb.get((config.d_model,), "value_head.0.weight")?;
        let value_w1 = vb.get((512, config.d_model), "value_head.1.weight")?;
        let value_linear1 = Linear::new(value_w1, None);
        let value_w2 = vb.get((1, 512), "value_head.3.weight")?;
        let value_linear2 = Linear::new(value_w2, None);

        Ok(Self {
            config,
            device,
            token_embed,
            pos_embed,
            layers,
            policy_norm,
            policy_linear,
            value_norm,
            value_linear1,
            value_linear2,
        })
    }

    /// Parse metadata from safetensors file header
    fn parse_metadata(mmap: &Mmap) -> anyhow::Result<HashMap<String, String>> {
        // Safetensors header: first 8 bytes = header length (u64 LE)
        let header_len = u64::from_le_bytes([
            mmap[0], mmap[1], mmap[2], mmap[3], mmap[4], mmap[5], mmap[6], mmap[7],
        ]) as usize;

        // Parse JSON header
        let header_bytes = &mmap[8..8 + header_len.min(mmap.len() - 8)];
        let header_str = std::str::from_utf8(header_bytes)?;

        // Parse JSON to extract __metadata__
        let doc: serde_json::Value = serde_json::from_str(header_str)?;

        let mut result = HashMap::new();
        if let Some(meta) = doc.get("__metadata__").and_then(|m| m.as_object()) {
            for (key, value) in meta {
                if let Some(s) = value.as_str() {
                    result.insert(key.clone(), s.to_string());
                }
            }
        }

        Ok(result)
    }

    /// Run inference on a sequence of board tokens
    ///
    /// # Arguments
    /// * `tokens` - Slice of token IDs representing the game history
    ///
    /// # Returns
    /// * `InferenceOutput` containing policy logits, value, best move index, and confidence
    pub fn predict(&mut self, tokens: &[usize]) -> Result<InferenceOutput> {
        let seq_len = tokens.len().min(self.config.context_length);

        // Take last CONTEXT_LENGTH tokens
        let start_idx = tokens.len().saturating_sub(self.config.context_length);
        let tokens_slice = &tokens[start_idx..];

        // Convert to tensor [1, seq_len]
        let tokens_u32: Vec<u32> = tokens_slice.iter().map(|&t| t as u32).collect();
        let tokens_tensor = Tensor::new(tokens_u32.as_slice(), &self.device)?.unsqueeze(0)?;

        // Token embedding
        let mut x = self.token_embed.forward(&tokens_tensor)?;

        // Add positional embedding
        let pos_slice = self.pos_embed.narrow(1, 0, seq_len)?;
        x = x.broadcast_add(&pos_slice)?;

        // Transformer layers (with mutable access for caching)
        for layer in &mut self.layers {
            let layer_out = layer.forward(&x)?;
            x = x.broadcast_add(&layer_out)?;
        }

        // Extract last token representation
        let last_token = x.narrow(1, seq_len - 1, 1)?.squeeze(1)?;

        // Policy head: RMSNorm -> Linear
        let policy_normed = rms_norm(&last_token, &self.policy_norm, 1e-5)?;
        let policy_logits = self.policy_linear.forward(&policy_normed)?;

        // Value head: RMSNorm -> Linear -> ReLU -> Linear
        let value_normed = rms_norm(&last_token, &self.value_norm, 1e-5)?;
        let value_hidden = self.value_linear1.forward(&value_normed)?.relu()?;
        let value_out = self.value_linear2.forward(&value_hidden)?;
        let value_scalar = value_out.squeeze(0)?.squeeze(0)?;
        let value = value_scalar.tanh()?.to_scalar::<f32>()?;

        // Get best move index and confidence
        let policy_squeezed = policy_logits.squeeze(0)?;
        let best_move_idx = policy_squeezed.argmax(0)?.to_scalar::<u32>()? as usize;

        // Compute softmax for confidence
        let policy_softmax = softmax(&policy_squeezed, 0)?;
        let confidence = policy_softmax
            .narrow(0, best_move_idx, 1)?
            .to_scalar::<f32>()?;

        Ok(InferenceOutput {
            policy_logits: policy_squeezed,
            value,
            best_move_idx,
            confidence,
        })
    }

    /// Get model device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Clear all layer caches (call after device changes)
    pub fn clear_caches(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.vocab_size, VOCAB_SIZE);
        assert_eq!(config.context_length, CONTEXT_LENGTH);
        assert_eq!(config.d_model, D_MODEL);
    }
}
