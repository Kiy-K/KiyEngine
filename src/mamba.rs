// src/mamba.rs

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module};
use std::sync::Arc;

// Constants based on the spec
pub const D_MODEL: usize = 384;
pub const D_STATE: usize = 16;
pub const D_CONV: usize = 4;
pub const EXPANSION_FACTOR: usize = 2;
pub const D_INNER: usize = D_MODEL * EXPANSION_FACTOR;

/// Represents the state of a Mamba block that needs to be tracked during search.
#[derive(Clone)]
pub struct MambaState {
    pub conv_state: Arc<Tensor>, // Shape: (D_INNER, D_CONV - 1)
    pub ssm_state: Arc<Tensor>,  // Shape: (D_INNER, D_STATE)
}

impl MambaState {
    /// Creates a new, zero-initialized MambaState.
    pub fn new(device: &Device) -> Result<Self> {
        Ok(Self {
            conv_state: Arc::new(Tensor::zeros((D_INNER, D_CONV - 1), DType::F32, device)?),
            ssm_state: Arc::new(Tensor::zeros((D_INNER, D_STATE), DType::F32, device)?),
        })
    }
}

/// The weights for a single Mamba block.
pub struct MambaBlock {
    pub in_proj: Linear,
    pub conv1d_w: Tensor, // Shape: (D_INNER, 1, D_CONV)
    pub conv1d_b: Tensor, // Shape: (D_INNER)
    pub x_proj: Linear,
    pub dt_proj: Linear,
    pub a_log: Tensor,    // Shape: (D_INNER, D_STATE)
    pub d: Tensor,        // Shape: (D_INNER)
    pub out_proj: Linear,
}

impl MambaBlock {
    /// Performs the forward pass for a single Mamba block on a single token.
    pub fn forward(&self, x: &Tensor, state: &mut MambaState) -> Result<Tensor> {
        // x shape: (D_MODEL)
        // Add batch and sequence dims: (1, 1, D_MODEL)
        let x_reshaped = x.reshape((1, 1, D_MODEL))?;

        // 1. Input Projection
        let xz = self.in_proj.forward(&x_reshaped)?; // (1, 1, 2 * D_INNER)
        let mut x_inner = xz.narrow(2, 0, D_INNER)?.squeeze(0)?.squeeze(0)?; // (D_INNER)
        let mut z = xz.narrow(2, D_INNER, D_INNER)?.squeeze(0)?.squeeze(0)?; // (D_INNER)

        // 2. 1D Convolution (Single token update)
        // Update conv state: [C-1] -> [C-1]
        // state.conv_state: (D_INNER, D_CONV - 1)

        // current_conv_input: (D_INNER, D_CONV)
        let current_conv_input = Tensor::cat(&[state.conv_state.as_ref(), &x_inner.reshape((D_INNER, 1))?], 1)?;

        // Update state for next time (zero-clone update of Arc)
        state.conv_state = Arc::new(current_conv_input.narrow(1, 1, D_CONV - 1)?);

        // conv1d_w: (D_INNER, 1, D_CONV)
        // Single token conv is just element-wise mul and sum
        let conv_weights = self.conv1d_w.squeeze(1)?; // (D_INNER, D_CONV)
        x_inner = (&current_conv_input * &conv_weights)?.sum(1)?; // (D_INNER)
        x_inner = x_inner.broadcast_add(&self.conv1d_b)?;

        // 3. Activation (SiLU)
        x_inner = candle_nn::ops::silu(&x_inner)?;

        // 4. SSM (S6) - Selective Scan
        let (y, new_ssm_state) = self.ssm(&x_inner, state.ssm_state.as_ref())?;
        state.ssm_state = Arc::new(new_ssm_state);

        // 5. Gating
        z = candle_nn::ops::silu(&z)?;
        let output = (y * z)?;

        // 6. Output Projection
        self.out_proj.forward(&output.reshape((1, 1, D_INNER))?)?.squeeze(0)?.squeeze(0)
    }

    fn ssm(&self, x: &Tensor, ssm_state: &Tensor) -> Result<(Tensor, Tensor)> {
        // x: (D_INNER)
        // ssm_state: (D_INNER, D_STATE)

        // Project x to get dt, B, C
        let ssm_params = self.x_proj.forward(&x.reshape((1, 1, D_INNER))?)?.squeeze(0)?.squeeze(0)?; // (D_INNER + 2 * D_STATE)

        let dt_params = ssm_params.narrow(0, 0, D_INNER)?;
        let b = ssm_params.narrow(0, D_INNER, D_STATE)?; // (D_STATE)
        let c = ssm_params.narrow(0, D_INNER + D_STATE, D_STATE)?; // (D_STATE)

        // Discretize dt
        let dt = self.dt_proj.forward(&dt_params.reshape((1, 1, D_INNER))?)?.squeeze(0)?.squeeze(0)?; // (D_INNER)
        // Softplus: log(1 + exp(x))
        let dt = (dt.exp()? + 1.0)?.log()?;

        // Discretize A and B
        // A_bar = exp(dt * A)
        let a = self.a_log.exp()?; // (D_INNER, D_STATE)
        let dt_reshaped = dt.reshape((D_INNER, 1))?;
        let a_bar = (&dt_reshaped * &a)?.exp()?;

        // B_bar = dt * B
        let b_reshaped = b.reshape((1, D_STATE))?;
        let b_bar = (&dt_reshaped * &b_reshaped)?;

        // Update SSM state: h_new = A_bar * h_old + B_bar * x
        let x_reshaped = x.reshape((D_INNER, 1))?;
        let new_ssm_state = ((&a_bar * ssm_state)? + (&b_bar * &x_reshaped)?)?;

        // Calculate output: y = sum(h_new * C, axis=1)
        let c_reshaped = c.reshape((1, D_STATE))?;
        let y = (&new_ssm_state * &c_reshaped)?.sum(1)?;

        // Add residual D connection
        let y = (y + (x * &self.d)?)?;

        Ok((y, new_ssm_state))
    }
}
