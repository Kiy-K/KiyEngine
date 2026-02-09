// src/engine/heads.rs
//! v5.2.0 Neural network heads for policy and value prediction
//!
//! Head weights are continuous FP16 (sandwich strategy: FP16 shell, ternary core).
//! PolicyHead: RMSNorm -> Linear(d_model, vocab_size)
//! ValueHead: RMSNorm -> Linear(d_model, 256) -> SiLU -> RMSNorm -> Linear(256, 1) -> Tanh
//!
//! Input is already ln_f-normalized by Engine before reaching heads.

use crate::engine::bitlinear::rms_norm;
use candle_core::Result;
use candle_core::Tensor;
use candle_nn::{Module, VarBuilder};

/// Value head: RMSNorm -> Linear(d_model, 256) -> SiLU -> RMSNorm -> Linear(256, 1)
/// Weights pre-transposed at load time for faster matmul.
pub struct ValueHead {
    pub norm1: Tensor,
    pub w1_t: Tensor, // [d_model, 256] pre-transposed + contiguous
    pub norm2: Tensor,
    pub w2_t: Tensor, // [256, 1] pre-transposed + contiguous
}

impl ValueHead {
    pub fn load(vb: &VarBuilder, d_model: usize) -> Result<Self> {
        let norm1 = vb.get((d_model,), "value_head.0.norm.weight")?;
        let w1 = vb.get((256, d_model), "value_head.0.weight")?;
        let w1_t = w1.t()?.contiguous()?;

        let norm2 = vb.get((256,), "value_head.2.norm.weight")?;
        let w2 = vb.get((1, 256), "value_head.2.weight")?;
        let w2_t = w2.t()?.contiguous()?;

        Ok(Self {
            norm1,
            w1_t,
            norm2,
            w2_t,
        })
    }

    /// Forward: RMSNorm -> matmul -> SiLU -> RMSNorm -> matmul (caller applies tanh)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = rms_norm(x, &self.norm1, 1e-5)?;
        let x = x.broadcast_matmul(&self.w1_t)?;
        let x = candle_nn::Activation::Silu.forward(&x)?;
        let x = rms_norm(&x, &self.norm2, 1e-5)?;
        x.broadcast_matmul(&self.w2_t)
    }
}

/// Policy head: RMSNorm -> Linear(d_model, vocab_size)
/// Weight pre-transposed at load time for faster matmul.
pub struct PolicyHead {
    pub norm: Tensor,
    pub w_t: Tensor, // [d_model, vocab_size] pre-transposed + contiguous
}

impl PolicyHead {
    pub fn load(vb: &VarBuilder, d_model: usize, vocab_size: usize) -> Result<Self> {
        let norm = vb.get((d_model,), "policy_head.norm.weight")?;
        let w = vb.get((vocab_size, d_model), "policy_head.weight")?;
        let w_t = w.t()?.contiguous()?;
        Ok(Self { norm, w_t })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = rms_norm(x, &self.norm, 1e-5)?;
        x.broadcast_matmul(&self.w_t)
    }
}
