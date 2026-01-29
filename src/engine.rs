// src/engine.rs
//
// The main KiyEngine V3 model combining Mamba-MoE architecture.
// Loads weights from HuggingFace safetensors format.

use crate::mamba::MambaBlock;
use crate::moe::{MoELayer, NUM_EXPERTS};
use crate::safetensor_loader::SafeTensorFile;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module};
use chess::Board;

pub const NUM_LAYERS: usize = 4;
pub const VOCAB_SIZE: usize = 768; // 12 pieces * 64 squares

/// Sequential value head: Linear(384->128) -> ReLU -> Linear(128->1)
pub struct ValueHead {
    pub linear1: Linear, // value_head.0
    pub linear2: Linear, // value_head.2
}

impl ValueHead {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = x.relu()?;
        self.linear2.forward(&x)
    }
}

/// Represents the full MoE-Mamba model using Candle.
pub struct Engine {
    pub embedding: Tensor, // Shape: (VOCAB_SIZE, D_MODEL) - note: singular "embedding"
    pub layers: Vec<MoELayer>,
    pub norm_w: Tensor, // Shape: (D_MODEL) - RMSNorm weight
    pub policy_head: Linear,
    pub value_head: ValueHead,
    pub device: Device,
}

/// Represents the game state for search.
/// Simplified version without Mamba state tracking.
#[derive(Clone)]
pub struct GameState {
    pub board: Board,
    pub move_history: Vec<usize>, // Token history for the model
}

impl GameState {
    pub fn new(board: Board) -> Self {
        Self {
            board,
            move_history: Vec::new(),
        }
    }

    /// Adds a move token to the history.
    pub fn push_token(&mut self, token: usize) {
        self.move_history.push(token);
        // Keep only the last 64 tokens (model's max sequence length)
        if self.move_history.len() > 64 {
            self.move_history.remove(0);
        }
    }

    /// Gets the input sequence for the model (padded to at least 1 token).
    pub fn get_input_sequence(&self) -> Vec<usize> {
        if self.move_history.is_empty() {
            vec![0] // Start token or padding
        } else {
            self.move_history.clone()
        }
    }
}

impl Engine {
    pub fn new() -> anyhow::Result<Self> {
        let device = if cfg!(feature = "cuda") {
            Device::new_cuda(0).unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        };

        if std::path::Path::new("model.safetensors").exists() {
            println!("Loading model from model.safetensors...");
            Self::load_from_safetensors("model.safetensors", device)
        } else {
            anyhow::bail!("model.safetensors not found. Random initialization not supported in this version.")
        }
    }

    fn load_from_safetensors(path: &str, device: Device) -> anyhow::Result<Self> {
        let st = SafeTensorFile::open(path)?;
        let dtype = DType::F32;

        // Load embedding (note: singular, not "embeddings")
        let embedding = st.load_tensor("embedding.weight", dtype, &device)?;

        // Load layers
        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for l in 0..NUM_LAYERS {
            // Router (with bias)
            // Router (with bias) - user specified mapping to skill_emb
            let router_w = st.load_tensor(&format!("layers.{}.skill_emb.weight", l), dtype, &device)?;
            let router_b = st.load_tensor(&format!("layers.{}.skill_emb.bias", l), dtype, &device)?;
            let router = Linear::new(router_w, Some(router_b));

            // Experts
            let mut experts = Vec::with_capacity(NUM_EXPERTS);
            for e in 0..NUM_EXPERTS {
                let prefix = format!("layers.{}.experts.{}", l, e);

                // in_proj: (2*D_INNER, D_MODEL)
                let in_proj_w = st.load_tensor(&format!("{}.in_proj.weight", prefix), dtype, &device)?;
                let in_proj = Linear::new(in_proj_w, None);

                // conv1d: weight (D_INNER, 1, D_CONV), bias (D_INNER)
                let conv1d_w = st.load_tensor(&format!("{}.conv1d.weight", prefix), dtype, &device)?;
                let conv1d_b = st.load_tensor(&format!("{}.conv1d.bias", prefix), dtype, &device)?;

                // x_proj: (D_INNER + 2*D_STATE, D_INNER) - kept for weight compatibility
                let x_proj_w = st.load_tensor(&format!("{}.x_proj.weight", prefix), dtype, &device)?;
                let x_proj = Linear::new(x_proj_w, None);

                // dt_proj: weight (D_INNER, D_INNER), bias (D_INNER)
                let dt_proj_w = st.load_tensor(&format!("{}.dt_proj.weight", prefix), dtype, &device)?;
                let dt_proj_b = st.load_tensor(&format!("{}.dt_proj.bias", prefix), dtype, &device)?;
                let dt_proj = Linear::new(dt_proj_w, Some(dt_proj_b));

                // A_log: (D_INNER, D_STATE) - kept for weight compatibility
                let a_log = st.load_tensor(&format!("{}.A_log", prefix), dtype, &device)?;

                // D: (D_INNER) - used for gating
                let d = st.load_tensor(&format!("{}.D", prefix), dtype, &device)?;

                // out_proj: (D_MODEL, D_INNER)
                let out_proj_w = st.load_tensor(&format!("{}.out_proj.weight", prefix), dtype, &device)?;
                let out_proj = Linear::new(out_proj_w, None);

                experts.push(MambaBlock {
                    in_proj,
                    conv1d_w,
                    conv1d_b,
                    x_proj,
                    dt_proj,
                    a_log,
                    d,
                    out_proj,
                });
            }

            layers.push(MoELayer { router, experts });
        }

        // Norm weight
        let norm_w = st.load_tensor("norm.weight", dtype, &device)?;

        // Policy head: (VOCAB_SIZE, D_MODEL)
        let policy_w = st.load_tensor("policy_head.weight", dtype, &device)?;
        let policy_head = Linear::new(policy_w, None);

        // Value head: Sequential with Linear(384->128) -> ReLU -> Linear(128->1)
        let value_head = ValueHead {
            linear1: Linear::new(
                st.load_tensor("value_head.0.weight", dtype, &device)?,
                Some(st.load_tensor("value_head.0.bias", dtype, &device)?),
            ),
            linear2: Linear::new(
                st.load_tensor("value_head.2.weight", dtype, &device)?,
                Some(st.load_tensor("value_head.2.bias", dtype, &device)?),
            ),
        };

        Ok(Self {
            embedding,
            layers,
            norm_w,
            policy_head,
            value_head,
            device,
        })
    }

    /// Forward pass on a sequence of tokens.
    ///
    /// # Arguments
    /// * `tokens` - Sequence of move tokens
    ///
    /// # Returns
    /// * (policy_logits, value) where policy_logits is (VOCAB_SIZE) and value is a scalar
    pub fn forward(&self, tokens: &[usize]) -> Result<(Tensor, f32)> {
        let seq_len = tokens.len();

        // Look up embeddings for all tokens
        let embeddings: Vec<Tensor> = tokens
            .iter()
            .map(|&t| self.embedding.get(t))
            .collect::<Result<Vec<_>>>()?;

        // Stack to (1, seq_len, d_model)
        let mut x = Tensor::stack(&embeddings, 0)?.unsqueeze(0)?;

        // Forward through layers with residual connections
        // PyTorch: for layer in self.layers: x = x + layer(self.norm(x))
        for layer in &self.layers {
            let x_norm = self.rms_norm(&x)?;
            let layer_output = layer.forward(&x_norm)?;
            x = (x + layer_output)?;
        }

        // Final norm
        x = self.rms_norm(&x)?;

        // Get last token state: (1, d_model)
        let last_token_state = x.get(0)?.get(seq_len - 1)?;

        // Policy logits: (vocab_size)
        let policy_logits = self.policy_head.forward(&last_token_state.unsqueeze(0)?)?
            .squeeze(0)?;

        // Value: tanh(value_head(last_token_state))
        let value_tensor = self.value_head.forward(&last_token_state.unsqueeze(0)?)?
            .squeeze(0)?
            .squeeze(0)?;
        let value = value_tensor.tanh()?.to_scalar::<f32>()?;

        Ok((policy_logits, value))
    }

    /// Single token forward pass for incremental inference.
    ///
    /// # Arguments
    /// * `token` - Single move token
    /// * `state` - Game state with history
    ///
    /// # Returns
    /// * (policy_logits, value)
    pub fn forward_token(&self, token: usize, state: &mut GameState) -> Result<(Tensor, f32)> {
        state.push_token(token);
        let tokens = state.get_input_sequence();
        self.forward(&tokens)
    }

    /// RMSNorm implementation matching PyTorch.
    fn rms_norm(&self, x: &Tensor) -> Result<Tensor> {
        let eps = 1e-5;
        // norm = x.norm(2, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        // return x / (norm + eps) * self.weight

        let x_sq = x.sqr()?;
        let mean_sq = x_sq.mean_keepdim(candle_core::D::Minus1)?;
        let norm = (mean_sq + eps)?.sqrt()?;
        let x_normed = x.broadcast_div(&norm)?;
        x_normed.broadcast_mul(&self.norm_w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_game_state() {
        let mut state = GameState::new(Board::default());
        assert_eq!(state.get_input_sequence(), vec![0]);

        state.push_token(100);
        state.push_token(200);
        assert_eq!(state.get_input_sequence(), vec![100, 200]);
    }
}
