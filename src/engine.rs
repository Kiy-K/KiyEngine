// src/engine.rs

use crate::mamba::{MambaBlock, MambaState, D_INNER, D_MODEL, D_STATE, D_CONV};
use crate::moe::{MoELayer, NUM_EXPERTS};
use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{Linear, VarBuilder, Module};
use chess::Board;

pub const NUM_LAYERS: usize = 4;
pub const VOCAB_SIZE: usize = 768; // 12 pieces * 64 squares

/// Represents the full MoE-Mamba model using Candle.
pub struct Engine {
    pub embeddings: Tensor, // Shape: (VOCAB_SIZE, D_MODEL)
    pub layers: Vec<MoELayer>,
    pub norm_w: Tensor, // Shape: (D_MODEL)
    pub policy_head: Linear,
    pub value_head: Linear,
    pub device: Device,
}

#[derive(Clone)]
pub struct GameState {
    pub board: Board,
    pub mamba_states: Vec<Vec<MambaState>>, // [layer][expert]
}

impl GameState {
    pub fn new(board: Board, device: &Device) -> Result<Self> {
        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for _ in 0..NUM_LAYERS {
            let mut experts = Vec::with_capacity(NUM_EXPERTS);
            for _ in 0..NUM_EXPERTS {
                experts.push(MambaState::new(device)?);
            }
            layers.push(experts);
        }
        Ok(Self {
            board,
            mamba_states: layers,
        })
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
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device)? };

        let embeddings = vb.get((VOCAB_SIZE, D_MODEL), "embeddings.weight")?;

        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for l in 0..NUM_LAYERS {
            let router_w = vb.get((NUM_EXPERTS, D_MODEL), &format!("layers.{}.router.weight", l))?;
            let router = Linear::new(router_w, None);

            let mut experts = Vec::with_capacity(NUM_EXPERTS);
            for e in 0..NUM_EXPERTS {
                let prefix = format!("layers.{}.experts.{}", l, e);

                let in_proj_w = vb.get((2 * D_INNER, D_MODEL), &format!("{}.in_proj.weight", prefix))?;
                let in_proj = Linear::new(in_proj_w, None);

                let conv1d_w = vb.get((D_INNER, 1, D_CONV), &format!("{}.conv1d.weight", prefix))?;
                let conv1d_b = vb.get(D_INNER, &format!("{}.conv1d.bias", prefix))?;

                let x_proj_w = vb.get((D_INNER + 2 * D_STATE, D_INNER), &format!("{}.x_proj.weight", prefix))?;
                let x_proj = Linear::new(x_proj_w, None);

                let dt_proj_w = vb.get((D_INNER, D_INNER), &format!("{}.dt_proj.weight", prefix))?;
                let dt_proj_b = vb.get(D_INNER, &format!("{}.dt_proj.bias", prefix))?;
                let dt_proj = Linear::new(dt_proj_w, Some(dt_proj_b));

                let a_log = vb.get((D_INNER, D_STATE), &format!("{}.A_log", prefix))?;
                let d = vb.get(D_INNER, &format!("{}.D", prefix))?;

                let out_proj_w = vb.get((D_MODEL, D_INNER), &format!("{}.out_proj.weight", prefix))?;
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

        let norm_w = vb.get(D_MODEL, "norm.weight")?;

        let policy_w = vb.get((VOCAB_SIZE, D_MODEL), "policy_head.weight")?;
        let policy_head = Linear::new(policy_w, None);

        let value_w = vb.get((1, D_MODEL), "value_head.weight")?;
        let value_head = Linear::new(value_w, None);

        Ok(Self {
            embeddings,
            layers,
            norm_w,
            policy_head,
            value_head,
            device,
        })
    }

    pub fn forward(&self, token: usize, state: &mut GameState) -> Result<(Tensor, f32)> {
        let mut x = self.embeddings.get(token)?;

        for i in 0..NUM_LAYERS {
            let residual = x.clone();

            // RMSNorm
            let x_norm = self.rms_norm(&x)?;

            let layer_output = self.layers[i].forward(&x_norm, &mut state.mamba_states[i])?;
            x = (residual + layer_output)?;
        }

        x = self.rms_norm(&x)?;

        let policy_logits = self.policy_head.forward(&x.reshape((1, 1, D_MODEL))?)?.squeeze(0)?.squeeze(0)?;
        let value_tensor = self.value_head.forward(&x.reshape((1, 1, D_MODEL))?)?.squeeze(0)?.squeeze(0)?;
        let value = value_tensor.to_scalar::<f32>()?;

        Ok((policy_logits, value))
    }

    fn rms_norm(&self, x: &Tensor) -> Result<Tensor> {
        let eps = 1e-6;
        let mean_sq = x.sqr()?.mean_all()?;
        let inv_rms = (mean_sq + eps)?.sqrt()?.recip()?;
        x.broadcast_mul(&inv_rms)?.broadcast_mul(&self.norm_w)
    }
}
