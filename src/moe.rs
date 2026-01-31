// src/moe.rs
//
// Optimized Mixture-of-Experts layer for KiyEngine V3.
// Eliminates CPU roundtrips for large tensors and implements stateful inference.

use crate::mamba::{MambaBlock, MambaState, D_MODEL};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module};

pub const NUM_EXPERTS: usize = 8;
pub const TOP_K: usize = 2;

/// State for an MoE layer to allow incremental inference.
#[derive(Clone)]
pub struct MoEState {
    pub expert_states: Vec<MambaState>,
}

impl MoEState {
    pub fn new(device: &Device) -> Result<Self> {
        let mut expert_states = Vec::with_capacity(NUM_EXPERTS);
        for _ in 0..NUM_EXPERTS {
            expert_states.push(MambaState::new(device)?);
        }
        Ok(Self { expert_states })
    }
}

pub struct MoELayer {
    pub router: Linear,
    pub experts: Vec<MambaBlock>,
}

impl MoELayer {
    /// Stateful forward pass for a single token.
    pub fn forward_stateful(&self, x: &Tensor, state: &mut MoEState) -> Result<Tensor> {
        // x: (1, 1, D_MODEL)
        let router_logits = self.router.forward(&x.squeeze(1)?)?; // (1, NUM_EXPERTS)
        let router_probs = candle_nn::ops::softmax(&router_logits, 1)?;

        // For single-token routing, using host-side logic for the 8 expert scores is efficient
        let probs_vec = router_probs.to_vec2::<f32>()?[0].clone();

        let mut indexed: Vec<(usize, f32)> = probs_vec.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_k: Vec<(usize, f32)> = indexed.into_iter().take(TOP_K).collect();
        let weight_sum: f32 = top_k.iter().map(|(_, w)| w).sum();

        let mut final_output = Tensor::zeros(x.dims(), x.dtype(), x.device())?;

        for (expert_idx, weight) in top_k {
            let normalized_weight = weight / (weight_sum + 1e-9);
            let expert_output = self.experts[expert_idx]
                .forward_stateful(x, &mut state.expert_states[expert_idx])?;
            let weighted_output = (expert_output * normalized_weight as f64)?;
            final_output = (final_output + weighted_output)?;
        }

        Ok(final_output)
    }

    /// Batch forward pass using native candle operations and index_add.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, channels) = x.dims3()?;
        let num_tokens = batch * seq_len;
        let x_flat = x.reshape((num_tokens, channels))?;

        let router_logits = self.router.forward(&x_flat)?;
        let router_probs = candle_nn::ops::softmax(&router_logits, 1)?;

        // Grouping tokens by expert for batch processing
        let probs_vec = router_probs.to_vec2::<f32>()?;
        let mut final_output = Tensor::zeros((num_tokens, D_MODEL), x.dtype(), x.device())?;

        for expert_idx in 0..NUM_EXPERTS {
            let mut token_indices = Vec::new();
            let mut token_weights = Vec::new();

            for (t_idx, probs) in probs_vec.iter().enumerate() {
                let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                let top_k = &indexed[0..TOP_K];
                let weight_sum: f32 = top_k.iter().map(|(_, w)| w).sum();

                for &(idx, w) in top_k {
                    if idx == expert_idx {
                        token_indices.push(t_idx as u32);
                        token_weights.push(w / (weight_sum + 1e-9));
                    }
                }
            }

            if token_indices.is_empty() {
                continue;
            }

            let indices_tensor =
                Tensor::from_vec(token_indices.clone(), (token_indices.len(),), x.device())?;
            let expert_input = x_flat.index_select(&indices_tensor, 0)?.unsqueeze(1)?;
            let expert_output = self.experts[expert_idx]
                .forward(&expert_input)?
                .squeeze(1)?;

            let weights_tensor =
                Tensor::from_vec(token_weights, (token_indices.len(), 1), x.device())?
                    .to_dtype(x.dtype())?;
            let weighted_output = expert_output.broadcast_mul(&weights_tensor)?;

            final_output = final_output.index_add(&indices_tensor, &weighted_output, 0)?;
        }

        final_output.reshape((batch, seq_len, D_MODEL))
    }
}

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
