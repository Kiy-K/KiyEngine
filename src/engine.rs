// src/engine.rs
//
// The main KiyEngine V3 model combining Mamba-MoE architecture.
// Optimized for stateful incremental inference.

use crate::moe::{MoELayer, MoEState, NUM_EXPERTS};
use crate::safetensor_loader::SafeTensorFile;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module};
use chess::Board;

pub const NUM_LAYERS: usize = 4;
pub const VOCAB_SIZE: usize = 768; // 12 pieces * 64 squares

/// Sequential value head: `Linear(384->128)` -> `ReLU` -> `Linear(128->1)`
pub struct ValueHead {
    pub linear1: Linear,
    pub linear2: Linear,
}

impl ValueHead {
    /// Forward pass for the value head.
    ///
    /// # Errors
    ///
    /// Returns an error if the linear layers fail.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = x.relu()?;
        self.linear2.forward(&x)
    }
}

/// Full state of the model for a specific search branch.
#[derive(Clone)]
pub struct EngineState {
    pub layer_states: Vec<MoEState>,
}

impl EngineState {
    /// Creates a new `EngineState`.
    ///
    /// # Errors
    ///
    /// Returns an error if the layer states cannot be initialized.
    pub fn new(device: &Device) -> Result<Self> {
        let mut layer_states = Vec::with_capacity(NUM_LAYERS);
        for _ in 0..NUM_LAYERS {
            layer_states.push(MoEState::new(device)?);
        }
        Ok(Self { layer_states })
    }
}

pub struct Engine {
    pub embedding: Tensor, // (VOCAB_SIZE, D_MODEL)
    pub layers: Vec<MoELayer>,
    pub norm_w: Tensor, // RMSNorm weight
    pub policy_head: Linear,
    pub value_head: ValueHead,
    pub device: Device,
}

#[derive(Clone)]
pub struct GameState {
    pub board: Board,
    pub move_history: Vec<usize>,
    pub engine_state: Option<EngineState>,
}

impl GameState {
    #[must_use]
    pub fn new(board: Board) -> Self {
        Self {
            board,
            move_history: Vec::new(),
            engine_state: None,
        }
    }

    pub fn push_token(&mut self, token: usize) {
        self.move_history.push(token);
        if self.move_history.len() > 64 {
            self.move_history.remove(0);
        }
    }

    #[must_use]
    pub fn get_input_sequence(&self) -> Vec<usize> {
        if self.move_history.is_empty() {
            vec![0]
        } else {
            self.move_history.clone()
        }
    }
}

impl Engine {
    /// Creates a new `Engine` and loads weights from `model.safetensors`.
    ///
    /// # Errors
    ///
    /// Returns an error if the model file is missing or corrupted.
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
            anyhow::bail!("model.safetensors not found.")
        }
    }

    fn load_from_safetensors(path: &str, device: Device) -> anyhow::Result<Self> {
        let st = SafeTensorFile::open(path)?;
        let dtype = DType::F32;

        let embedding = st.load_tensor("embedding.weight", dtype, &device)?;

        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for l in 0..NUM_LAYERS {
            let router_w =
                st.load_tensor(&format!("layers.{l}.skill_emb.weight"), dtype, &device)?;
            let router_b =
                st.load_tensor(&format!("layers.{l}.skill_emb.bias"), dtype, &device)?;
            let router = Linear::new(router_w, Some(router_b));

            let mut experts = Vec::with_capacity(NUM_EXPERTS);
            for e in 0..NUM_EXPERTS {
                let prefix = format!("layers.{l}.experts.{e}");
                let in_proj = Linear::new(
                    st.load_tensor(&format!("{prefix}.in_proj.weight"), dtype, &device)?,
                    None,
                );
                let conv1d_w =
                    st.load_tensor(&format!("{prefix}.conv1d.weight"), dtype, &device)?;
                let conv1d_b =
                    st.load_tensor(&format!("{prefix}.conv1d.bias"), dtype, &device)?;
                let x_proj = Linear::new(
                    st.load_tensor(&format!("{prefix}.x_proj.weight"), dtype, &device)?,
                    None,
                );
                let dt_proj = Linear::new(
                    st.load_tensor(&format!("{prefix}.dt_proj.weight"), dtype, &device)?,
                    Some(st.load_tensor(&format!("{prefix}.dt_proj.bias"), dtype, &device)?),
                );
                let a_log = st.load_tensor(&format!("{prefix}.A_log"), dtype, &device)?;
                let d = st.load_tensor(&format!("{prefix}.D"), dtype, &device)?;
                let out_proj = Linear::new(
                    st.load_tensor(&format!("{prefix}.out_proj.weight"), dtype, &device)?,
                    None,
                );

                experts.push(crate::mamba::MambaBlock {
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

        let norm_w = st.load_tensor("norm.weight", dtype, &device)?;
        let policy_head = Linear::new(st.load_tensor("policy_head.weight", dtype, &device)?, None);
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

    /// Incremental forward pass for a single token using `EngineState`.
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
    pub fn forward_incremental(
        &self,
        token: usize,
        state: &mut EngineState,
    ) -> Result<(Tensor, f32)> {
        let x = self.embedding.get(token)?.unsqueeze(0)?.unsqueeze(0)?; // (1, 1, D_MODEL)
        let mut current_x = x;

        for (i, layer) in self.layers.iter().enumerate() {
            let x_norm = self.rms_norm(&current_x)?;
            let layer_output = layer.forward_stateful(&x_norm, &mut state.layer_states[i])?;
            current_x = (current_x + layer_output)?;
        }

        current_x = self.rms_norm(&current_x)?;
        let last_state = current_x.squeeze(0)?.squeeze(0)?; // (D_MODEL)

        let policy_logits = self
            .policy_head
            .forward(&last_state.unsqueeze(0)?)?
            .squeeze(0)?;
        let value_tensor = self
            .value_head
            .forward(&last_state.unsqueeze(0)?)?
            .squeeze(0)?
            .squeeze(0)?;
        let value = value_tensor.tanh()?.to_scalar::<f32>()?;

        Ok((policy_logits, value))
    }

    /// Batch forward pass for a sequence of tokens.
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
    pub fn forward(&self, tokens: &[usize]) -> Result<(Tensor, f32)> {
        let embeddings: Vec<Tensor> = tokens
            .iter()
            .map(|&t| self.embedding.get(t))
            .collect::<Result<Vec<_>>>()?;
        let mut x = Tensor::stack(&embeddings, 0)?.unsqueeze(0)?;

        for layer in &self.layers {
            let x_norm = self.rms_norm(&x)?;
            let layer_output = layer.forward(&x_norm)?;
            x = (x + layer_output)?;
        }

        x = self.rms_norm(&x)?;
        let last_token_state = x.get(0)?.get(tokens.len() - 1)?;
        let policy_logits = self
            .policy_head
            .forward(&last_token_state.unsqueeze(0)?)?
            .squeeze(0)?;
        let value_tensor = self
            .value_head
            .forward(&last_token_state.unsqueeze(0)?)?
            .squeeze(0)?
            .squeeze(0)?;
        let value = value_tensor.tanh()?.to_scalar::<f32>()?;

        Ok((policy_logits, value))
    }

    fn rms_norm(&self, x: &Tensor) -> Result<Tensor> {
        let eps = 1e-5;
        let x_sq = x.sqr()?;
        let mean_sq = x_sq.mean_keepdim(candle_core::D::Minus1)?;
        let norm = (mean_sq + eps)?.sqrt()?;
        x.broadcast_div(&norm)?.broadcast_mul(&self.norm_w)
    }

    /// Computes the initial engine state for a sequence of tokens.
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
    pub fn get_initial_state(&self, tokens: &[usize]) -> Result<(EngineState, Tensor, f32)> {
        let mut state = EngineState::new(&self.device)?;
        let mut last_res = (
            Tensor::zeros((VOCAB_SIZE,), DType::F32, &self.device)?,
            0.0f32,
        );
        for &t in tokens {
            last_res = self.forward_incremental(t, &mut state)?;
        }
        Ok((state, last_res.0, last_res.1))
    }
}
