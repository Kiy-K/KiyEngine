// src/moe.rs
//
// Optimized Mixture-of-Experts layer for KiyEngine V3.
// Eliminates CPU roundtrips for large tensors and implements stateful inference.

use crate::mamba::{MambaBlock, MambaState, D_MODEL};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module};

pub const NUM_EXPERTS: usize = 8;
pub const TOP_K: usize = 2;

/// State for an `MoE` layer to allow incremental inference.
#[derive(Clone)]
pub struct MoEState {
    pub expert_states: Vec<MambaState>,
}

impl MoEState {
    /// Creates a new `MoEState`.
    ///
    /// # Errors
    ///
    /// Returns an error if the expert states cannot be initialized.
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
    ///
    /// # Errors
    ///
    /// Returns an error if the router or experts fail.
    ///
    /// # Panics
    ///
    /// Panics if top-k indices are out of range for the experts.
    pub fn forward_stateful(&self, x: &Tensor, state: &mut MoEState) -> Result<Tensor> {
        // x: (1, 1, D_MODEL)
        let router_logits = self.router.forward(&x.squeeze(1)?)?; // (1, NUM_EXPERTS)
        let router_probs = candle_nn::ops::softmax(&router_logits, 1)?;

        // To eliminate host-device transfers and maintain top-k=2 routing, we use device-side masking.
        // 1. Find top-1 mask
        let max1 = router_probs.max_keepdim(1)?;
        let mask1 = router_probs.ge(&max1)?;

        // 2. Find top-2 by masking out top-1
        let mask1_f = mask1.to_dtype(router_probs.dtype())?;
        let masked_probs = router_probs.broadcast_mul(&mask1_f.ones_like()?.sub(&mask1_f)?)?;
        let max2 = masked_probs.max_keepdim(1)?;
        let mask2 = masked_probs.ge(&max2)?;

        let top_k_mask = mask1.add(&mask2)?;
        let top_k_mask_f = top_k_mask.to_dtype(router_probs.dtype())?;
        let top_k_probs = router_probs.broadcast_mul(&top_k_mask_f)?;
        let weight_sum = top_k_probs.sum_keepdim(1)?;
        let normalized_weights = top_k_probs.broadcast_div(&(weight_sum + 1e-9)?)?;

        // Execute all experts and combine on-device to avoid host-device synchronization.
        // We only update the state of selected experts using device-side conditional updates.
        let mut expert_outputs = Vec::with_capacity(NUM_EXPERTS);
        for i in 0..NUM_EXPERTS {
            let old_conv = state.expert_states[i].conv_state.clone();
            let old_ssm = state.expert_states[i].ssm_state.clone();

            let output = self.experts[i].forward_stateful(x, &mut state.expert_states[i])?;
            expert_outputs.push(output);

            let mask_i = top_k_mask.narrow(1, i, 1)?;
            state.expert_states[i].conv_state = mask_i.where_cond(&state.expert_states[i].conv_state, &old_conv)?;
            state.expert_states[i].ssm_state = mask_i.where_cond(&state.expert_states[i].ssm_state, &old_ssm)?;
        }

        let stacked = Tensor::stack(&expert_outputs, 0)?; // (8, 1, 1, D_MODEL)
        let weights = normalized_weights.transpose(0, 1)?.reshape((NUM_EXPERTS, 1, 1, 1))?;

        stacked.broadcast_mul(&weights)?.sum(0)?.squeeze(0)
    }

    /// Batch forward pass using native candle operations.
    ///
    /// # Errors
    ///
    /// Returns an error if the router or experts fail.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, channels) = x.dims3()?;
        let num_tokens = batch * seq_len;
        let x_flat = x.reshape((num_tokens, channels))?;

        let router_logits = self.router.forward(&x_flat)?;
        let router_probs = candle_nn::ops::softmax(&router_logits, 1)?;

        // Device-side top-k=2 mask to eliminate host-device synchronization
        let max1 = router_probs.max_keepdim(1)?;
        let mask1 = router_probs.ge(&max1)?;
        let mask1_f = mask1.to_dtype(router_probs.dtype())?;
        let masked_probs = router_probs.broadcast_mul(&mask1_f.ones_like()?.sub(&mask1_f)?)?;
        let max2 = masked_probs.max_keepdim(1)?;
        let mask2 = masked_probs.ge(&max2)?;

        let top_k_mask_f = mask1.add(&mask2)?.to_dtype(router_probs.dtype())?;
        let top_k_probs = router_probs.broadcast_mul(&top_k_mask_f)?;
        let weight_sum = top_k_probs.sum_keepdim(1)?;
        let normalized_weights = top_k_probs.broadcast_div(&(weight_sum + 1e-9)?)?;

        let mut expert_outputs = Vec::with_capacity(NUM_EXPERTS);
        for i in 0..NUM_EXPERTS {
            expert_outputs.push(self.experts[i].forward(x)?);
        }

        let stacked = Tensor::stack(&expert_outputs, 0)?; // (8, B, T, D_MODEL)
        let weights = normalized_weights.reshape((batch, seq_len, NUM_EXPERTS))?
            .transpose(1, 2)? // (B, 8, T)
            .transpose(0, 1)? // (8, B, T)
            .unsqueeze(3)?; // (8, B, T, 1)

        stacked.broadcast_mul(&weights)?.sum(0)
    }
}

/// Creates a new `MoELayer` with zero weights.
///
/// # Errors
///
/// Returns an error if the layer cannot be initialized.
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
