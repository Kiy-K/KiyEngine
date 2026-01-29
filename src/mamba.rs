// src/mamba.rs
//
// Simplified Mamba block matching the PyTorch "Gated CNN" variant from HuggingFace.
// This version does NOT use SSM state tracking - it's a stateless sequence model.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module};

// Constants based on the spec (matching config.json)
pub const D_MODEL: usize = 384;
pub const D_STATE: usize = 16;
pub const D_CONV: usize = 4;
pub const EXPANSION_FACTOR: usize = 2;
pub const D_INNER: usize = D_MODEL * EXPANSION_FACTOR; // 768

/// The weights for a single Mamba block.
/// This matches the PyTorch MambaBlock structure exactly.
pub struct MambaBlock {
    /// Input projection: (d_model) -> (2 * d_inner) for x and z
    pub in_proj: Linear,

    /// 1D depthwise convolution weights
    /// PyTorch shape: (D_INNER, 1, D_CONV) with groups=D_INNER
    pub conv1d_w: Tensor,
    pub conv1d_b: Tensor,

    /// These are present in the weights but NOT used in the simplified forward
    /// Kept for weight loading compatibility
    pub x_proj: Linear,
    pub dt_proj: Linear,
    pub a_log: Tensor,

    /// The D parameter used for gating
    pub d: Tensor,

    /// Output projection: (d_inner) -> (d_model)
    pub out_proj: Linear,
}

impl MambaBlock {
    /// Performs the forward pass for a Mamba block.
    ///
    /// This matches the PyTorch implementation which uses a simplified "Gated CNN" approach:
    /// 1. Input projection splits into x_inner and z (gate)
    /// 2. Conv1D on x_inner with causal padding
    /// 3. SiLU activation
    /// 4. Element-wise multiply with D
    /// 5. Gate with SiLU(z)
    /// 6. Output projection
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape (batch, seq_len, d_model)
    ///
    /// # Returns
    /// * Output tensor of shape (batch, seq_len, d_model)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_batch, seq_len, _) = x.dims3()?;

        // 1. Input Projection: (B, L, D_MODEL) -> (B, L, 2*D_INNER)
        let xz = self.in_proj.forward(x)?;

        // Split into x_inner and z (gate)
        let x_inner = xz.narrow(2, 0, D_INNER)?; // (B, L, D_INNER)
        let z = xz.narrow(2, D_INNER, D_INNER)?; // (B, L, D_INNER)

        // 2. Conv1D expects (B, C, L) format
        let x_conv = x_inner.transpose(1, 2)?; // (B, D_INNER, L)

        // Apply depthwise conv1d manually
        // PyTorch uses padding=d_conv-1 and then truncates to [:, :, :L]
        let x_conv = self.depthwise_conv1d(&x_conv, seq_len)?;

        // Transpose back: (B, D_INNER, L) -> (B, L, D_INNER)
        let x_conv = x_conv.transpose(1, 2)?;

        // 3. SiLU activation on conv output
        let x_activated = candle_nn::ops::silu(&x_conv)?;

        // 4. Element-wise multiply with D parameter
        // D shape: (D_INNER) -> broadcast to (B, L, D_INNER)
        let y = x_activated.broadcast_mul(&self.d)?;

        // 5. Gate with SiLU(z)
        let z_gate = candle_nn::ops::silu(&z)?;
        let y = (y * z_gate)?;

        // 6. Output projection: (B, L, D_INNER) -> (B, L, D_MODEL)
        self.out_proj.forward(&y)
    }

    /// Performs depthwise 1D convolution with causal padding.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape (B, D_INNER, L)
    /// * `seq_len` - Original sequence length for truncation
    ///
    /// # Returns
    /// * Output tensor of shape (B, D_INNER, L)
    fn depthwise_conv1d(&self, x: &Tensor, seq_len: usize) -> Result<Tensor> {
        let (batch, channels, _len) = x.dims3()?;

        // Pad input with zeros on the left (causal padding)
        // padding = d_conv - 1 = 3
        let padding = D_CONV - 1;
        let zeros = Tensor::zeros((batch, channels, padding), x.dtype(), x.device())?;
        let x_padded = Tensor::cat(&[&zeros, x], 2)?; // (B, D_INNER, L + padding)

        // Squeeze conv weights: (D_INNER, 1, D_CONV) -> (D_INNER, D_CONV)
        let weights = self.conv1d_w.squeeze(1)?;

        // Manual depthwise convolution
        // For each channel, convolve independently
        let padded_len = x_padded.dim(2)?;
        let out_len = padded_len - D_CONV + 1;

        // Unfold the padded input to create sliding windows
        // We'll compute this efficiently using matrix operations
        let mut output_slices = Vec::with_capacity(out_len);

        for i in 0..out_len {
            // Extract window: (B, D_INNER, D_CONV)
            let window = x_padded.narrow(2, i, D_CONV)?;
            // Element-wise multiply with weights and sum over conv dimension
            // weights: (D_INNER, D_CONV) -> broadcast to (B, D_INNER, D_CONV)
            let weighted = window.broadcast_mul(&weights)?;
            let summed = weighted.sum(2)?; // (B, D_INNER)
            output_slices.push(summed.unsqueeze(2)?); // (B, D_INNER, 1)
        }

        // Concatenate all output positions
        let output = Tensor::cat(&output_slices, 2)?; // (B, D_INNER, out_len)

        // Add bias: (D_INNER) -> broadcast
        let output = output.broadcast_add(&self.conv1d_b.reshape((1, D_INNER, 1))?)?;

        // Truncate to original sequence length (matching PyTorch [:, :, :L])
        output.narrow(2, 0, seq_len)
    }
}

/// Creates a new MambaBlock with zero-initialized weights (for testing).
pub fn create_zero_block(device: &Device) -> Result<MambaBlock> {
    Ok(MambaBlock {
        in_proj: Linear::new(
            Tensor::zeros((2 * D_INNER, D_MODEL), DType::F32, device)?,
            None,
        ),
        conv1d_w: Tensor::zeros((D_INNER, 1, D_CONV), DType::F32, device)?,
        conv1d_b: Tensor::zeros(D_INNER, DType::F32, device)?,
        x_proj: Linear::new(
            Tensor::zeros((D_INNER + 2 * D_STATE, D_INNER), DType::F32, device)?,
            None,
        ),
        dt_proj: Linear::new(
            Tensor::zeros((D_INNER, D_INNER), DType::F32, device)?,
            Some(Tensor::zeros(D_INNER, DType::F32, device)?),
        ),
        a_log: Tensor::zeros((D_INNER, D_STATE), DType::F32, device)?,
        d: Tensor::ones(D_INNER, DType::F32, device)?,
        out_proj: Linear::new(
            Tensor::zeros((D_MODEL, D_INNER), DType::F32, device)?,
            None,
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mamba_block_forward() -> Result<()> {
        let device = Device::Cpu;
        let block = create_zero_block(&device)?;

        // Create dummy input: (batch=1, seq_len=16, d_model=384)
        let x = Tensor::zeros((1, 16, D_MODEL), DType::F32, &device)?;
        let output = block.forward(&x)?;

        assert_eq!(output.dims(), &[1, 16, D_MODEL]);
        Ok(())
    }
}
