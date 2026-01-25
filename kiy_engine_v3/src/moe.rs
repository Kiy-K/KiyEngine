// src/moe.rs

use crate::mamba::{MambaBlock, MambaState, D_MODEL};
use ndarray::{Array1, Array2};
use ndarray_stats::QuantileExt;

/// A Mixture-of-Experts layer containing a router and a set of MambaBlock experts.
pub struct MoELayer {
    /// The router projects the input embedding to a score for each expert.
    pub router_w: Array2<f32>, // Shape: (NUM_EXPERTS, D_MODEL)

    /// The array of 8 Mamba block experts.
    pub experts: [MambaBlock; NUM_EXPERTS],
}

pub const NUM_EXPERTS: usize = 8;
const TOP_K: usize = 2;

impl MoELayer {
    /// Performs the forward pass for the MoE layer.
    ///
    /// # Arguments
    /// * `x` - The input tensor of shape (D_MODEL).
    /// * `states` - The mutable slice of states for each expert in this layer.
    ///
    /// # Returns
    /// The output tensor of shape (D_MODEL).
    pub fn forward(&self, x: &Array1<f32>, states: &mut [MambaState]) -> Array1<f32> {
        // 1. Route to experts
        let router_logits = self.router_w.dot(x);

        // Convert logits to probabilities using softmax
        let max_logit = *router_logits.max().unwrap();
        let exp_logits = router_logits.mapv(|logit| (logit - max_logit).exp());
        let exp_sum = exp_logits.sum();
        let probs = exp_logits / exp_sum;

        // 2. Select Top-K experts
        let mut expert_indices: Vec<usize> = (0..NUM_EXPERTS).collect();
        expert_indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

        let top_k_indices = &expert_indices[0..TOP_K];
        let top_k_probs: Vec<f32> = top_k_indices.iter().map(|&i| probs[i]).collect();
        let top_k_sum = top_k_probs.iter().sum::<f32>();
        let top_k_weights: Vec<f32> = top_k_probs.iter().map(|p| p / top_k_sum).collect();

        // 3. Compute weighted sum of expert outputs
        let mut final_output = Array1::zeros(D_MODEL);
        for (i, &expert_idx) in top_k_indices.iter().enumerate() {
            let expert_output = self.experts[expert_idx].forward(x, &mut states[expert_idx]);
            final_output = final_output + (&expert_output * top_k_weights[i]);
        }

        final_output
    }
}
