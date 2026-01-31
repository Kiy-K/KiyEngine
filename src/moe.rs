// src/moe.rs
//
// Mixture-of-Experts layer with Top-K gated routing.
// Matches the PyTorch MoELayer structure from HuggingFace.

use crate::mamba::{MambaBlock, D_MODEL};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module};

pub const NUM_EXPERTS: usize = 8;
pub const TOP_K: usize = 2;

/// A Mixture-of-Experts layer containing a router and a set of MambaBlock experts.
pub struct MoELayer {
    /// The router projects the input embedding to a score for each expert.
    /// Note: PyTorch's nn.Linear has bias=True by default.
    pub router: Linear,

    /// The array of 8 Mamba block experts.
    pub experts: Vec<MambaBlock>,
}

impl MoELayer {
    /// Performs the forward pass for the MoE layer on a sequence.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape (batch, seq_len, d_model)
    ///
    /// # Returns
    /// * Output tensor of shape (batch, seq_len, d_model)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, channels) = x.dims3()?;

        // Flatten to (B*L, C) for routing
        let x_flat = x.reshape((batch * seq_len, channels))?;

        // 1. Compute router logits and probabilities
        let router_logits = self.router.forward(&x_flat)?; // (B*L, NUM_EXPERTS)
        let router_probs = candle_nn::ops::softmax(&router_logits, 1)?;

        // 2. Select Top-K experts
        // We'll do this on CPU for simplicity since NUM_EXPERTS is small (8)
        let probs_vec: Vec<Vec<f32>> = router_probs.to_dtype(DType::F32)?.to_vec2()?;

        let num_tokens = batch * seq_len;
        let mut all_top_indices: Vec<Vec<usize>> = Vec::with_capacity(num_tokens);
        let mut all_top_weights: Vec<Vec<f32>> = Vec::with_capacity(num_tokens);

        for token_probs in &probs_vec {
            // Get top-k indices
            let mut indexed: Vec<(usize, f32)> = token_probs
                .iter()
                .enumerate()
                .map(|(i, &p)| (i, p))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let top_indices: Vec<usize> = indexed.iter().take(TOP_K).map(|(i, _)| *i).collect();
            let top_weights: Vec<f32> = indexed.iter().take(TOP_K).map(|(_, w)| *w).collect();

            // Renormalize weights
            let weight_sum: f32 = top_weights.iter().sum();
            let normalized: Vec<f32> = top_weights
                .iter()
                .map(|w| w / (weight_sum + 1e-9))
                .collect();

            all_top_indices.push(top_indices);
            all_top_weights.push(normalized);
        }

        // 3. Process each expert and accumulate weighted outputs
        // For efficiency, we batch tokens going to the same expert

        // Initialize output tensor
        let mut final_output = Tensor::zeros((num_tokens, D_MODEL), x.dtype(), x.device())?;

        for expert_idx in 0..NUM_EXPERTS {
            // Find all (token_idx, k_position) pairs where this expert is selected
            let mut token_indices: Vec<usize> = Vec::new();
            let mut token_weights: Vec<f32> = Vec::new();

            for (token_idx, (indices, weights)) in all_top_indices
                .iter()
                .zip(all_top_weights.iter())
                .enumerate()
            {
                for (k_pos, &idx) in indices.iter().enumerate() {
                    if idx == expert_idx {
                        token_indices.push(token_idx);
                        token_weights.push(weights[k_pos]);
                    }
                }
            }

            if token_indices.is_empty() {
                continue;
            }

            // Gather input tokens for this expert
            let expert_input = self.gather_tokens(&x_flat, &token_indices)?;

            // Reshape to (num_selected, 1, D_MODEL) for MambaBlock (which expects sequences)
            let num_selected = token_indices.len();
            let expert_input = expert_input.reshape((num_selected, 1, D_MODEL))?;

            // Forward through expert
            let expert_output = self.experts[expert_idx].forward(&expert_input)?;

            // Squeeze back to (num_selected, D_MODEL)
            let expert_output = expert_output.reshape((num_selected, D_MODEL))?;

            // Apply weights and scatter back to output
            let weight_tensor =
                Tensor::from_vec(token_weights.clone(), (num_selected,), x.device())?
                    .to_dtype(x.dtype())?;

            let weighted_output =
                expert_output.broadcast_mul(&weight_tensor.reshape((num_selected, 1))?)?;

            // Scatter add to final output
            final_output = self.scatter_add(&final_output, &token_indices, &weighted_output)?;
        }

        // Reshape back to (batch, seq_len, d_model)
        final_output.reshape((batch, seq_len, D_MODEL))
    }

    /// Gathers tokens at specified indices from the flattened input.
    fn gather_tokens(&self, x_flat: &Tensor, indices: &[usize]) -> Result<Tensor> {
        let slices: Vec<Tensor> = indices
            .iter()
            .map(|&i| x_flat.get(i))
            .collect::<Result<Vec<_>>>()?;

        if slices.len() == 1 {
            slices[0].unsqueeze(0)
        } else {
            Tensor::stack(&slices, 0)
        }
    }

    /// Scatter-adds weighted expert outputs back to the output tensor.
    fn scatter_add(&self, output: &Tensor, indices: &[usize], values: &Tensor) -> Result<Tensor> {
        // For thread safety and simplicity, we do this via explicit indexing
        // This is not the most efficient but correct for now
        let mut output_vec: Vec<Vec<f32>> = output.to_vec2()?;
        let values_vec: Vec<Vec<f32>> = values.to_vec2()?;

        for (local_idx, &global_idx) in indices.iter().enumerate() {
            for (j, val) in values_vec[local_idx].iter().enumerate() {
                output_vec[global_idx][j] += val;
            }
        }

        // Convert back to tensor
        let flat: Vec<f32> = output_vec.into_iter().flatten().collect();
        Tensor::from_vec(flat, output.dims(), output.device())?.to_dtype(output.dtype())
    }
}

/// Creates a new MoELayer with zero-initialized weights (for testing).
pub fn create_zero_moe_layer(device: &Device) -> Result<MoELayer> {
    let router = Linear::new(
        Tensor::zeros((NUM_EXPERTS, D_MODEL), DType::F32, device)?,
        Some(Tensor::zeros(NUM_EXPERTS, DType::F32, device)?),
    );

    let mut experts = Vec::with_capacity(NUM_EXPERTS);
    for _ in 0..NUM_EXPERTS {
        experts.push(crate::mamba::create_zero_block(device)?);
    }

    Ok(MoELayer { router, experts })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moe_layer_forward() -> Result<()> {
        let device = Device::Cpu;
        let layer = create_zero_moe_layer(&device)?;

        // Create dummy input: (batch=1, seq_len=16, d_model=384)
        let x = Tensor::randn(0.0f32, 1.0, (1, 16, D_MODEL), &device)?;
        let output = layer.forward(&x)?;

        assert_eq!(output.dims(), &[1, 16, D_MODEL]);
        Ok(())
    }
}
