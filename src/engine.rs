// src/engine.rs

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Embedding, Linear, Module, VarBuilder};
use std::fs::File;
use memmap2::Mmap;

pub const VOCAB_SIZE: usize = 4608;
pub const CONTEXT_LENGTH: usize = 32;
pub const D_MODEL: usize = 1024;
pub const NUM_LAYERS: usize = 4;

/// Standard RMSNorm implementation.
fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x_sq = x.sqr()?;
    let mean_sq = x_sq.mean_keepdim(D::Minus1)?;
    let norm = (mean_sq + eps)?.sqrt()?;
    let x_normed = x.broadcast_div(&norm)?;
    x_normed.broadcast_mul(weight)
}

/// Custom BitLinear mirroring PyTorch QAT behavior.
pub struct BitLinear {
    pub weight_t: Tensor, // Pre-transposed for performance
    pub rms_norm_weight: Tensor,
    pub eps: f64,
}

impl BitLinear {
    pub fn load(
        prefix: &str,
        vb: &VarBuilder,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Self> {
        let weight = vb.get((out_dim, in_dim), &format!("{}.weight", prefix))?;
        let weight_t = weight.t()?.contiguous()?;
        let rms_norm_weight = vb.get((in_dim,), &format!("{}.norm.weight", prefix))?;
        
        Ok(Self {
            weight_t,
            rms_norm_weight,
            eps: 1e-5,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = rms_norm(x, &self.rms_norm_weight, self.eps)?;
        x.broadcast_matmul(&self.weight_t)
    }
}

pub struct ValueHead {
    pub norm_weight: Tensor,
    pub linear1: Linear,
    pub linear2: Linear,
}

impl ValueHead {
    pub fn load(vb: &VarBuilder) -> Result<Self> {
        let norm_weight = vb.get((D_MODEL,), "value_head.0.weight")?;
        let linear1_w = vb.get((512, D_MODEL), "value_head.1.weight")?;
        let linear2_w = vb.get((1, 512), "value_head.3.weight")?;
        
        Ok(Self { 
            norm_weight,
            linear1: Linear::new(linear1_w, None),
            linear2: Linear::new(linear2_w, None),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = rms_norm(x, &self.norm_weight, 1e-5)?;
        let x = self.linear1.forward(&x)?;
        let x = x.relu()?;
        self.linear2.forward(&x)
    }
}

pub struct PolicyHead {
    pub norm_weight: Tensor,
    pub linear: Linear,
}

impl PolicyHead {
    pub fn load(vb: &VarBuilder) -> Result<Self> {
        let norm_weight = vb.get((D_MODEL,), "policy_head.0.weight")?;
        let linear_w = vb.get((VOCAB_SIZE, D_MODEL), "policy_head.1.weight")?;
        Ok(Self {
            norm_weight,
            linear: Linear::new(linear_w, None),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = rms_norm(x, &self.norm_weight, 1e-5)?;
        self.linear.forward(&x)
    }
}

pub struct Engine {
    pub embedding: Embedding,
    pub pos_embed: Tensor,
    pub layers: Vec<BitLinear>,
    pub policy_head: PolicyHead,
    pub value_head: ValueHead,
    pub device: Device,
}

impl Engine {
    pub fn new() -> anyhow::Result<Self> {
        let device = if cfg!(feature = "cuda") {
            Device::new_cuda(0).unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        };
        Self::load_from_safetensors("kiyengine_v4.safetensors", device)
    }

    pub fn load_from_safetensors(model_path: &str, device: Device) -> anyhow::Result<Self> {
        let file = File::open(model_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let tensors = candle_core::safetensors::load_buffer(&mmap, &device)?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        let embed_w = vb.get((VOCAB_SIZE, D_MODEL), "embed.weight")?;
        let embedding = Embedding::new(embed_w, D_MODEL);
        
        let pos_embed = vb.get((1, CONTEXT_LENGTH, D_MODEL), "pos_embed")?;

        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for i in 0..NUM_LAYERS {
            layers.push(BitLinear::load(&format!("layers.{}.linear", i), &vb, D_MODEL, D_MODEL)?);
        }

        let policy_head = PolicyHead::load(&vb)?;
        let value_head = ValueHead::load(&vb)?;

        Ok(Self {
            embedding,
            pos_embed,
            layers,
            policy_head,
            value_head,
            device,
        })
    }

    pub fn forward(&self, tokens: &[usize]) -> Result<(Tensor, f32)> {
        let seq_len = tokens.len();
        if seq_len == 0 {
            return Err(candle_core::Error::Msg("Empty tokens".to_string()));
        }
        
        let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        let tokens_tensor = Tensor::new(tokens_u32.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut x = self.embedding.forward(&tokens_tensor)?;
        
        let pos_slice = self.pos_embed.narrow(1, 0, seq_len)?;
        x = x.broadcast_add(&pos_slice)?;

        for layer in &self.layers {
            let layer_out = layer.forward(&x)?;
            x = (x + layer_out)?;
        }

        let last_token_x = x.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
        let policy_logits = self.policy_head.forward(&last_token_x)?.squeeze(0)?;
        let value_out = self.value_head.forward(&last_token_x)?.squeeze(0)?.squeeze(0)?;
        let eval_score = value_out.tanh()?.to_scalar::<f32>()?;

        Ok((policy_logits, eval_score))
    }

    pub fn predict(&self, tokens: &[usize]) -> Result<(u32, f32)> {
        let (logits, eval) = self.forward(tokens)?;
        let best_move_id = logits.argmax(0)?.to_scalar::<u32>()?;
        Ok((best_move_id, eval))
    }
}
