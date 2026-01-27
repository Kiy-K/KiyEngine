// src/moe.rs

use crate::mamba::{MambaBlock, MambaState, D_MODEL};
use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module};

pub const NUM_EXPERTS: usize = 8;
const TOP_K: usize = 2;

/// A Mixture-of-Experts layer containing a router and a set of MambaBlock experts.
pub struct MoELayer {
    /// The router projects the input embedding to a score for each expert.
    pub router: Linear,

    /// The array of 8 Mamba block experts.
    pub experts: Vec<MambaBlock>,
}

impl MoELayer {
    /// Performs the forward pass for the MoE layer.
    pub fn forward(&self, x: &Tensor, states: &mut [MambaState]) -> Result<Tensor> {
        // 1. Route to experts
        let x_reshaped = x.reshape((1, 1, D_MODEL))?;
        let router_logits = self.router.forward(&x_reshaped)?.squeeze(0)?.squeeze(0)?; // (NUM_EXPERTS)

        // 2. Select Top-K experts
        // Since NUM_EXPERTS is small (8), we can do this on the CPU or manually.
        let logits_vec: Vec<f32> = router_logits.to_vec1()?;
        let mut indices: Vec<usize> = (0..NUM_EXPERTS).collect();
        indices.sort_by(|&a, &b| logits_vec[b].partial_cmp(&logits_vec[a]).unwrap());

        let top_indices = &indices[0..TOP_K];

        // Convert to probabilities using softmax only for the selected experts or all?
        // Standard MoE usually does softmax over all and then picks top-k.
        let probs = candle_nn::ops::softmax(&router_logits, 0)?;
        let probs_vec: Vec<f32> = probs.to_vec1()?;

        let mut top_weights: Vec<f32> = top_indices.iter().map(|&i| probs_vec[i]).collect();
        let weight_sum: f32 = top_weights.iter().sum();
        for w in top_weights.iter_mut() {
            *w /= weight_sum;
        }

        // 3. Compute weighted sum of expert outputs
        let mut final_output = Tensor::zeros(D_MODEL, x.dtype(), x.device())?;
        for (i, &expert_idx) in top_indices.iter().enumerate() {
            let expert_output = self.experts[expert_idx].forward(x, &mut states[expert_idx])?;
            let weighted_expert_output = expert_output.affine(top_weights[i] as f64, 0.0)?;
            final_output = (final_output + weighted_expert_output)?;
        }

        Ok(final_output)
    }
}
