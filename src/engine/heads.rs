// src/engine/heads.rs
//! Neural network heads for policy and value prediction

use crate::constants::{D_MODEL, VOCAB_SIZE};
use crate::engine::rms_norm;
use crate::engine::ModelFormat;
use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

/// Value head: predicts position evaluation (-1 to +1)
pub struct ValueHead {
    pub norm_weight: Tensor,
    pub linear1: Linear,
    pub linear2: Linear,
}

impl ValueHead {
    /// Load from original format
    pub fn load(vb: &VarBuilder) -> Result<Self> {
        Self::load_with_format(vb, ModelFormat::Original)
    }

    /// Load with specified format
    pub fn load_with_format(vb: &VarBuilder, format: ModelFormat) -> Result<Self> {
        let prefix = match format {
            ModelFormat::Original => "value_head",
            ModelFormat::Finetuned => "_orig_mod.value_head",
        };

        let norm_weight = vb.get((D_MODEL,), &format!("{}.0.weight", prefix))?;
        let linear1_w = vb.get((512, D_MODEL), &format!("{}.1.weight", prefix))?;
        let linear2_w = vb.get((1, 512), &format!("{}.3.weight", prefix))?;
        Ok(Self {
            norm_weight,
            linear1: Linear::new(linear1_w, None),
            linear2: Linear::new(linear2_w, None),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = rms_norm(x, &self.norm_weight, 1e-5)?;
        let x = self.linear1.forward(&x)?;
        let x = x.relu()?;
        self.linear2.forward(&x)
    }
}

/// Policy head: outputs logits for move probabilities
pub struct PolicyHead {
    pub norm_weight: Tensor,
    pub linear: Linear,
}

impl PolicyHead {
    /// Load from original format
    pub fn load(vb: &VarBuilder) -> Result<Self> {
        Self::load_with_format(vb, ModelFormat::Original)
    }

    /// Load with specified format
    pub fn load_with_format(vb: &VarBuilder, format: ModelFormat) -> Result<Self> {
        let prefix = match format {
            ModelFormat::Original => "policy_head",
            ModelFormat::Finetuned => "_orig_mod.policy_head",
        };

        let norm_weight = vb.get((D_MODEL,), &format!("{}.0.weight", prefix))?;
        let linear_w = vb.get((VOCAB_SIZE, D_MODEL), &format!("{}.1.weight", prefix))?;
        Ok(Self {
            norm_weight,
            linear: Linear::new(linear_w, None),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = rms_norm(x, &self.norm_weight, 1e-5)?;
        self.linear.forward(&x)
    }
}
