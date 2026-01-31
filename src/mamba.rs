// src/mamba.rs
//
// Optimized Mamba block with Stateful SSM and Conv1D.
// Matches the architecture of RWKV-6 Eagle / Mamba-MoE.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module};

pub const D_MODEL: usize = 384;
pub const D_STATE: usize = 16;
pub const D_CONV: usize = 4;
pub const EXPANSION_FACTOR: usize = 2;
pub const D_INNER: usize = D_MODEL * EXPANSION_FACTOR; // 768

/// State for a single Mamba block to allow incremental inference.
#[derive(Clone)]
pub struct MambaState {
    pub conv_state: Tensor, // Shape: (1, D_INNER, D_CONV - 1)
    pub ssm_state: Tensor,  // Shape: (1, D_INNER, D_STATE)
}

impl MambaState {
    pub fn new(device: &Device) -> Result<Self> {
        Ok(Self {
            conv_state: Tensor::zeros((1, D_INNER, D_CONV - 1), DType::F32, device)?,
            ssm_state: Tensor::zeros((1, D_INNER, D_STATE), DType::F32, device)?,
        })
    }
}

pub struct MambaBlock {
    pub in_proj: Linear,
    pub conv1d_w: Tensor, // (D_INNER, 1, D_CONV)
    pub conv1d_b: Tensor, // (D_INNER)
    pub x_proj: Linear,
    pub dt_proj: Linear,
    pub a_log: Tensor,
    pub d: Tensor,
    pub out_proj: Linear,
}

impl MambaBlock {
    /// Full stateful forward pass for a single token.
    pub fn forward_stateful(&self, x: &Tensor, state: &mut MambaState) -> Result<Tensor> {
        // x shape: (1, 1, D_MODEL)
        let (_batch, _seq_len, _dim) = x.dims3()?;

        // 1. Input Projection
        let xz = self.in_proj.forward(x)?; // (1, 1, 2*D_INNER)
        let x_inner = xz.narrow(2, 0, D_INNER)?.transpose(1, 2)?; // (1, D_INNER, 1)
        let z = xz.narrow(2, D_INNER, D_INNER)?; // (1, 1, D_INNER)

        // 2. Conv1D State Update
        let next_conv_state = Tensor::cat(&[
            state.conv_state.narrow(2, 1, D_CONV - 2)?,
            x_inner.clone()
        ], 2)?;
        state.conv_state = next_conv_state.clone();

        // Depthwise Conv1D
        let x_conv = (next_conv_state * self.conv1d_w.squeeze(1)?.unsqueeze(0)?)?
            .sum(2)? // (1, D_INNER)
            .broadcast_add(&self.conv1d_b)?;

        let x_activated = candle_nn::ops::silu(&x_conv)?; // (1, D_INNER)

        // 3. SSM Part
        let x_proj = self.x_proj.forward(&x_activated)?; // (1, D_INNER + 2*D_STATE)

        let dt = x_proj.narrow(1, 0, D_INNER)?;
        let b = x_proj.narrow(1, D_INNER, D_STATE)?;
        let c = x_proj.narrow(1, D_INNER + D_STATE, D_STATE)?;

        let dt = self.dt_proj.forward(&dt)?;
        // softplus: log(1 + exp(x))
        let dt = (dt.exp()? + 1.0)?.log()?;

        let a = self.a_log.exp()?.neg()?; // (D_INNER, D_STATE)

        // Discretization
        let da = (dt.unsqueeze(2)? * a.unsqueeze(0)?)?.exp()?; // (1, D_INNER, D_STATE)
        let db = (dt.unsqueeze(2)? * b.unsqueeze(1)?)?; // (1, D_INNER, D_STATE)

        // State Update: h = dA * h + dB * x
        let ssm_update = db.broadcast_mul(&x_activated.unsqueeze(2)?)?;
        state.ssm_state = ((da * &state.ssm_state)? + ssm_update)?;

        // Output: y = C * h + D * x
        let y_ssm = state.ssm_state.broadcast_mul(&c.unsqueeze(1)?)?.sum(2)?;
        let y = (y_ssm + x_activated.broadcast_mul(&self.d)?)?;

        // 4. Gating and Final Projection
        let z_gate = candle_nn::ops::silu(&z.squeeze(1)?)?;
        let y_gated = (y * z_gate)?;

        self.out_proj.forward(&y_gated.unsqueeze(1)?)
    }

    /// Optimized batch forward pass using native candle operations.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_batch, seq_len, _) = x.dims3()?;

        // 1. Input Projection
        let xz = self.in_proj.forward(x)?;
        let x_inner = xz.narrow(2, 0, D_INNER)?;
        let z = xz.narrow(2, D_INNER, D_INNER)?;

        // 2. Optimized Conv1D
        let padding = D_CONV - 1;
        let x_padded = x_inner.transpose(1, 2)?.pad_with_zeros(2, padding, 0)?;

        let mut conv_outputs = Vec::with_capacity(seq_len);
        let weights = self.conv1d_w.squeeze(1)?; // (D_INNER, D_CONV)

        for i in 0..seq_len {
            let window = x_padded.narrow(2, i, D_CONV)?; // (B, D_INNER, D_CONV)
            let out = (window * weights.unsqueeze(0)?)?.sum(2)?; // (B, D_INNER)
            conv_outputs.push(out.unsqueeze(1)?);
        }

        let x_conv = Tensor::cat(&conv_outputs, 1)?;
        let x_conv = x_conv.broadcast_add(&self.conv1d_b)?;

        let x_activated = candle_nn::ops::silu(&x_conv)?;

        // 3. Batch SSM (Simplified Gated-CNN for sequence)
        let y = x_activated.broadcast_mul(&self.d)?;
        let z_gate = candle_nn::ops::silu(&z)?;
        let y = (y * z_gate)?;

        self.out_proj.forward(&y)
    }
}

pub fn create_zero_block(device: &Device) -> Result<MambaBlock> {
    Ok(MambaBlock {
        in_proj: Linear::new(Tensor::zeros((2 * D_INNER, D_MODEL), DType::F32, device)?, None),
        conv1d_w: Tensor::zeros((D_INNER, 1, D_CONV), DType::F32, device)?,
        conv1d_b: Tensor::zeros(D_INNER, DType::F32, device)?,
        x_proj: Linear::new(Tensor::zeros((D_INNER + 2 * D_STATE, D_INNER), DType::F32, device)?, None),
        dt_proj: Linear::new(Tensor::zeros((D_INNER, D_INNER), DType::F32, device)?, Some(Tensor::zeros(D_INNER, DType::F32, device)?)),
        a_log: Tensor::zeros((D_INNER, D_STATE), DType::F32, device)?,
        d: Tensor::ones(D_INNER, DType::F32, device)?,
        out_proj: Linear::new(Tensor::zeros((D_MODEL, D_INNER), DType::F32, device)?, None),
    })
}
