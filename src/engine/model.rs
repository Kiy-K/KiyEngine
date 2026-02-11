// src/engine/model.rs
//! ChessModel: High-level inference API delegating to Engine
//!
//! Provides InferenceOutput with argmax/softmax convenience.

use crate::constants::{CONTEXT_LENGTH, D_MODEL, NUM_LAYERS, VOCAB_SIZE};
use crate::engine::Engine;
use candle_core::{Device, Result, Tensor};
use candle_nn::ops::softmax;

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
    pub policy_logits: Tensor,
    pub value: f32,
    pub best_move_idx: usize,
    pub confidence: f32,
}

/// ChessModel: wraps Engine for high-level inference with argmax/softmax
pub struct ChessModel {
    engine: Engine,
}

impl ChessModel {
    pub fn from_gguf<P: AsRef<std::path::Path>>(path: P, device: Device) -> anyhow::Result<Self> {
        let engine = Engine::load_from_gguf(path.as_ref().to_str().unwrap(), device)?;
        Ok(Self { engine })
    }

    pub fn predict(&self, tokens: &[usize]) -> Result<InferenceOutput> {
        let (policy_logits, value) = self.engine.forward(tokens)?;

        let best_move_idx = policy_logits.argmax(0)?.to_scalar::<u32>()? as usize;

        let policy_softmax = softmax(&policy_logits, 0)?;
        let confidence = policy_softmax
            .narrow(0, best_move_idx, 1)?
            .to_scalar::<f32>()?;

        Ok(InferenceOutput {
            policy_logits,
            value,
            best_move_idx,
            confidence,
        })
    }

    pub fn device(&self) -> &Device {
        &self.engine.device
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
