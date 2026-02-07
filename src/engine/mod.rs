// src/engine/mod.rs
//! Neural network engine for position evaluation
//!
//! Implements a BitNet 1.58-bit Transformer architecture for chess evaluation.

pub mod bitlinear;
pub mod gguf;
pub mod heads;
pub mod model;

#[cfg(test)]
mod tests;

pub use bitlinear::{rms_norm, BitLinear};
pub use heads::{PolicyHead, ValueHead};
pub use model::{ChessModel, ModelConfig};

use crate::constants::{
    CONTEXT_LENGTH, DEFAULT_MODEL_PATH, D_MODEL, LEGACY_MODEL_PATH, NUM_LAYERS, VOCAB_SIZE,
};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;

/// Model format detection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelFormat {
    /// Original format (v4): embed.weight, layers.0.linear.weight, etc.
    Original,
    /// Fine-tuned format with _orig_mod prefix
    Finetuned,
}

/// Main engine struct containing the neural network
pub struct Engine {
    pub embedding: Embedding,
    pub pos_embed: Tensor,
    pub layers: Vec<BitLinear>,
    pub policy_head: PolicyHead,
    pub value_head: ValueHead,
    pub device: Device,
    pub format: ModelFormat,
}

impl Engine {
    /// Create a new engine with automatic format detection
    ///
    /// Tries V5 GGUF format first, falls back to v4 safetensors
    pub fn new() -> anyhow::Result<Self> {
        let device = if cfg!(feature = "cuda") {
            Device::new_cuda(0).unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        };

        // Try V5 GGUF format first
        if std::path::Path::new(DEFAULT_MODEL_PATH).exists() {
            println!("Loading V5 GGUF model: {}", DEFAULT_MODEL_PATH);
            return Self::load_from_gguf(DEFAULT_MODEL_PATH, device);
        }

        // Fall back to v4 safetensors
        if std::path::Path::new(LEGACY_MODEL_PATH).exists() {
            println!("Loading v4 safetensors model: {}", LEGACY_MODEL_PATH);
            return Self::load_from_safetensors(LEGACY_MODEL_PATH, device);
        }

        anyhow::bail!(
            "No model file found. Expected '{}' or '{}'",
            DEFAULT_MODEL_PATH,
            LEGACY_MODEL_PATH
        )
    }

    /// Detect model format from tensor keys
    fn detect_format(tensors: &HashMap<String, Tensor>) -> ModelFormat {
        // Check if any key starts with _orig_mod
        if tensors.keys().any(|k| k.starts_with("_orig_mod.")) {
            ModelFormat::Finetuned
        } else {
            ModelFormat::Original
        }
    }

    /// Get tensor key with proper prefix based on format
    fn get_key(format: ModelFormat, base: &str) -> String {
        match format {
            ModelFormat::Original => base.to_string(),
            ModelFormat::Finetuned => format!("_orig_mod.{}", base),
        }
    }

    /// Load model weights from safetensors file
    pub fn load_from_safetensors(model_path: &str, device: Device) -> anyhow::Result<Self> {
        let file = File::open(model_path)?;
        // SAFETY: Memory mapping is safe because the file is read-only and
        // we trust the model file format (safetensors has built-in validation)
        let mmap = unsafe { Mmap::map(&file)? };

        // Parse metadata for scale factors
        let metadata = Self::parse_metadata(&mmap)?;

        // Load tensors using safetensors crate with I8 handling
        let tensors = Self::load_tensors_with_i8_support(&mmap, &device)?;

        let format = Self::detect_format(&tensors);

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        // Load embeddings
        let embed_key = Self::get_key(format, "embed.weight");
        let embed_w = vb.get((VOCAB_SIZE, D_MODEL), &embed_key)?;
        let embedding = Embedding::new(embed_w, D_MODEL);

        let pos_key = Self::get_key(format, "pos_embed");
        let pos_embed = vb.get((1, CONTEXT_LENGTH, D_MODEL), &pos_key)?;

        // Load transformer layers with metadata
        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for i in 0..NUM_LAYERS {
            let prefix = Self::get_key(format, &format!("layers.{}.linear", i));
            layers.push(BitLinear::load(&prefix, &vb, D_MODEL, D_MODEL, &metadata)?);
        }

        // Load heads with format
        let policy_head = PolicyHead::load_with_format(&vb, format)?;
        let value_head = ValueHead::load_with_format(&vb, format)?;

        Ok(Self {
            embedding,
            pos_embed,
            layers,
            policy_head,
            value_head,
            device,
            format,
        })
    }

    /// Load model weights from GGUF file (V5 format)
    pub fn load_from_gguf(model_path: &str, device: Device) -> anyhow::Result<Self> {
        use crate::engine::gguf::GgmlQuantizationType;
        use crate::engine::gguf::GgufFile;

        // Load GGUF file
        let gguf = GgufFile::from_path(model_path)?;

        // Extract configuration from metadata
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

        // Convert GGUF tensors to Candle tensors
        let mut tensors = HashMap::new();
        let mut metadata: HashMap<String, String> = HashMap::new();

        for tensor_info in &gguf.tensors {
            let tensor_data = gguf.get_tensor_data(tensor_info);
            let shape: Vec<usize> = tensor_info.dimensions.iter().map(|&d| d as usize).collect();

            let tensor = match tensor_info.quantization {
                GgmlQuantizationType::I8 => {
                    // BitNet 1.58-bit: convert i8 to f32
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

        // Extract scale factors from metadata
        for i in 0..num_layers {
            let scale_key = format!("kiyengine.layers.{}.linear.scale", i);
            if let Some(scale) = gguf.get_f32(&scale_key) {
                let meta_key = format!("layers.{}.linear_scale", i);
                metadata.insert(meta_key, scale.to_string());
            }
        }

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        // Load embeddings
        let embed_w = vb.get((vocab_size, d_model), "embed.weight")?;
        let embedding = Embedding::new(embed_w, d_model);

        let pos_embed = vb.get((1, context_length, d_model), "pos_embed")?;

        // Load transformer layers with metadata
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let prefix = format!("layers.{}.linear", i);
            layers.push(BitLinear::load(&prefix, &vb, d_model, d_model, &metadata)?);
        }

        // Load heads
        let policy_head = PolicyHead::load(&vb)?;
        let value_head = ValueHead::load(&vb)?;

        Ok(Self {
            embedding,
            pos_embed,
            layers,
            policy_head,
            value_head,
            device,
            format: ModelFormat::Original,
        })
    }

    /// Load tensors from safetensors with I8 dtype support
    /// Converts I8 tensors to F32 for Candle compatibility
    fn load_tensors_with_i8_support(
        mmap: &Mmap,
        device: &Device,
    ) -> anyhow::Result<HashMap<String, Tensor>> {
        use safetensors::{Dtype, SafeTensors};

        let st = SafeTensors::deserialize(&mmap)?;
        let mut tensors = HashMap::new();

        for name in st.names() {
            let view = st.tensor(&name)?;
            let shape = view.shape().to_vec();
            let dtype = view.dtype();
            let data = view.data();

            // Convert I8 to F32 for Candle compatibility
            let tensor = if dtype == Dtype::I8 {
                let i8_data: &[i8] = bytemuck::cast_slice(data);
                let f32_data: Vec<f32> = i8_data.iter().map(|&v| v as f32).collect();
                Tensor::from_vec(f32_data, shape, device)?
            } else if dtype == Dtype::F16 {
                let f16_data: &[half::f16] = bytemuck::cast_slice(data);
                let f32_data: Vec<f32> = f16_data.iter().map(|&v| v.to_f32()).collect();
                Tensor::from_vec(f32_data, shape, device)?
            } else if dtype == Dtype::F32 {
                let f32_data: &[f32] = bytemuck::cast_slice(data);
                Tensor::from_vec(f32_data.to_vec(), shape, device)?
            } else if dtype == Dtype::F64 {
                let f64_data: &[f64] = bytemuck::cast_slice(data);
                let f32_data: Vec<f32> = f64_data.iter().map(|&v| v as f32).collect();
                Tensor::from_vec(f32_data, shape, device)?
            } else {
                // Try to convert other dtypes to f32
                let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
                Tensor::from_vec(f32_data, shape, device)?
            };

            tensors.insert(name.to_string(), tensor);
        }

        Ok(tensors)
    }

    /// Parse metadata from safetensors file
    fn parse_metadata(mmap: &Mmap) -> anyhow::Result<HashMap<String, String>> {
        // Get the header length from the first 8 bytes
        let header_len = u64::from_le_bytes([
            mmap[0], mmap[1], mmap[2], mmap[3], mmap[4], mmap[5], mmap[6], mmap[7],
        ]) as usize;

        // Parse the JSON header
        let header_bytes = &mmap[8..8 + header_len];
        let header_str = std::str::from_utf8(header_bytes)?;

        // Parse as JSON to extract metadata
        let metadata: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(header_str)?;

        // Extract __metadata__ if present
        let mut result = HashMap::new();
        if let Some(meta) = metadata.get("__metadata__") {
            if let Some(meta_obj) = meta.as_object() {
                for (key, value) in meta_obj {
                    if let Some(s) = value.as_str() {
                        result.insert(key.clone(), s.to_string());
                    }
                }
            }
        }

        Ok(result)
    }

    /// Run forward pass on token sequence
    ///
    /// Returns (policy_logits, value_score) where value_score is in range [-1, 1]
    pub fn forward(&self, tokens: &[usize]) -> Result<(Tensor, f32)> {
        // Handle empty token sequences (e.g., starting position with no moves)
        // Use token 0 as a BOS/padding token (a1a1 = not a valid chess move)
        let padded;
        let tokens = if tokens.is_empty() {
            padded = vec![0usize];
            &padded[..]
        } else {
            tokens
        };

        let seq_len = tokens.len();

        // Take at most CONTEXT_LENGTH tokens
        let take_len = seq_len.min(CONTEXT_LENGTH);
        let tokens_slice = &tokens[seq_len - take_len..];

        let tokens_u32: Vec<u32> = tokens_slice.iter().map(|&t| t as u32).collect();

        let tokens_tensor = Tensor::new(tokens_u32.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut x = self.embedding.forward(&tokens_tensor)?;

        let pos_slice = self.pos_embed.narrow(1, 0, take_len)?;
        x = x.broadcast_add(&pos_slice)?;

        // Transformer layers (BitLinear handles interior mutability for caching)
        for layer in &self.layers {
            let layer_out = layer.forward(&x)?;
            x = x.broadcast_add(&layer_out)?;
        }

        let last_token_x = x.narrow(1, take_len - 1, 1)?.squeeze(1)?;
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
