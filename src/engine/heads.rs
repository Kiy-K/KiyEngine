// src/engine/heads.rs
//! v6.0.0 Neural network heads for policy and value prediction
//!
//! V6: ALL head weights are BitLinear (ternary with RMSNorm).
//! PolicyHead: BitLinear(d_model, vocab_size) — SIMD GEMV for single token
//! ValueHead: BitLinear(d_model, 256) -> SiLU -> BitLinear(256, 1) -> Tanh
//!
//! Input is already ln_f-normalized by Engine before reaching heads.

use crate::engine::bitlinear::{rms_norm_vec, BitLinear};
use candle_core::Result;
use candle_core::Tensor;
use candle_nn::{Module, VarBuilder};

/// Value head: BitLinear(d_model, 256) -> SiLU -> BitLinear(256, 1)
/// Uses SIMD packed GEMV for single-token inference.
pub struct ValueHead {
    pub layer1: BitLinear, // d_model -> 256
    pub layer2: BitLinear, // 256 -> 1
}

impl ValueHead {
    pub fn load(vb: &VarBuilder, d_model: usize) -> Result<Self> {
        let layer1 = BitLinear::load_from_vb(
            "value_head.0.weight",
            "value_head.0.norm.weight",
            vb,
            d_model,
            256,
        )?;
        let layer2 = BitLinear::load_from_vb(
            "value_head.2.weight",
            "value_head.2.norm.weight",
            vb,
            256,
            1,
        )?;
        Ok(Self { layer1, layer2 })
    }

    /// Forward: BitLinear -> SiLU -> BitLinear (caller applies tanh)
    /// Single token: zero-tensor f32 path with SIMD GEMV.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.dims();
        let is_single =
            shape[0] == 1 && (shape.len() < 2 || shape.get(1).copied().unwrap_or(1) == 1);

        if is_single {
            return self.forward_single(x);
        }

        let x = self.layer1.forward(x)?;
        let x = candle_nn::Activation::Silu.forward(&x)?;
        self.layer2.forward(&x)
    }

    /// Fully-fused single-token path: raw f32 + SIMD GEMV
    fn forward_single(&self, x: &Tensor) -> Result<Tensor> {
        let device = x.device();
        let x_f32: Vec<f32> = x.flatten_all()?.to_vec1::<f32>()?;

        // Layer 1: RMSNorm + SIMD GEMV
        let normed1 = rms_norm_vec(&x_f32, &self.layer1.rms_norm_weight_f32, 1e-5);
        let mut hidden = vec![0.0f32; self.layer1.out_dim];
        self.layer1.dispatch_packed_gemv(&normed1, &mut hidden);

        // SiLU in-place
        for h in hidden.iter_mut() {
            *h = *h / (1.0 + (-*h).exp());
        }

        // Layer 2: RMSNorm + SIMD GEMV
        let normed2 = rms_norm_vec(&hidden, &self.layer2.rms_norm_weight_f32, 1e-5);
        let mut out = vec![0.0f32; self.layer2.out_dim];
        self.layer2.dispatch_packed_gemv(&normed2, &mut out);

        Tensor::from_vec(out, (1, self.layer2.out_dim), device)
    }
}

/// Policy head: BitLinear(d_model, vocab_size) — SIMD GEMV for single token
pub struct PolicyHead {
    pub layer: BitLinear, // d_model -> vocab_size
}

impl PolicyHead {
    pub fn load(vb: &VarBuilder, d_model: usize, vocab_size: usize) -> Result<Self> {
        let layer = BitLinear::load_from_vb(
            "policy_head.weight",
            "policy_head.norm.weight",
            vb,
            d_model,
            vocab_size,
        )?;
        Ok(Self { layer })
    }

    /// Forward: BitLinear (auto-dispatches to SIMD for single token)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.layer.forward(x)
    }
}
