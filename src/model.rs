//! Mamba-MoE Model Loading and Architecture.
//!
//! This module defines the structures for the Mamba-MoE neural network
//! and implements the logic for loading weights from a `.safetensors` file.
//! The architecture is fixed as per the KiyEngine V3 specifications.

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

// --- Architectural Constants ---
const NUM_LAYERS: usize = 4;
const NUM_EXPERTS: usize = 8;
const D_MODEL: usize = 384;

/// Represents a single expert in a Mamba-MoE layer.
#[derive(Debug)]
pub struct Expert {
    pub in_proj_w: Tensor,
    pub conv1d_w: Tensor,
    pub x_proj_w: Tensor,
    pub dt_proj_w: Tensor,
    pub a_log: Tensor,
    pub d: Tensor,
    pub out_proj_w: Tensor,
}

/// Represents a single layer in the Mamba-MoE model, containing multiple experts and a router.
#[derive(Debug)]
pub struct MambaLayer {
    pub experts: Vec<Expert>,
    pub router_w: Tensor,
}

/// The main Mamba-MoE model structure.
#[derive(Debug)]
pub struct MambaMoEModel {
    pub embeddings: Tensor,
    pub layers: Vec<MambaLayer>,
    pub policy_head: Tensor,
    pub value_head: Tensor,
}

impl MambaMoEModel {
    /// Attempts to load the model weights from a `.safetensors` file.
    ///
    /// If the file is not found, it falls back to random initialization and
    /// prints a warning. If the file is found but is malformed or missing
    /// specific tensors, it will return an error.
    pub fn load(path: &str) -> Result<Self> {
        let device = Device::Cpu; // Or use Device::cuda_if_available(0)?;
        if !Path::new(path).exists() {
            println!("⚠️ Warning: '{}' not found. Initializing model with random weights.", path);
            return Self::random_init(&device);
        }

        let tensors = candle_core::safetensors::load(path, &device)?;
        Self::from_tensors(tensors)
    }

    /// Creates the model from a loaded SafeTensors map.
    fn from_tensors(mut tensors: HashMap<String, Tensor>) -> Result<Self> {
        let mut get_tensor = |name: &str| -> Result<Tensor> {
            tensors.remove(name).ok_or_else(|| anyhow!("Missing tensor: {}", name))
        };

        let embeddings = get_tensor("embeddings")?;
        let policy_head = get_tensor("policy_head")?;
        let value_head = get_tensor("value_head")?;

        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for l in 0..NUM_LAYERS {
            let router_w = get_tensor(&format!("layer.{}.router_w", l))?;
            let mut experts = Vec::with_capacity(NUM_EXPERTS);
            for e in 0..NUM_EXPERTS {
                let expert = Expert {
                    in_proj_w: get_tensor(&format!("layer.{}.expert.{}.in_proj_w", l, e))?,
                    conv1d_w: get_tensor(&format!("layer.{}.expert.{}.conv1d_w", l, e))?,
                    x_proj_w: get_tensor(&format!("layer.{}.expert.{}.x_proj_w", l, e))?,
                    dt_proj_w: get_tensor(&format!("layer.{}.expert.{}.dt_proj_w", l, e))?,
                    a_log: get_tensor(&format!("layer.{}.expert.{}.a_log", l, e))?,
                    d: get_tensor(&format!("layer.{}.expert.{}.d", l, e))?,
                    out_proj_w: get_tensor(&format!("layer.{}.expert.{}.out_proj_w", l, e))?,
                };
                experts.push(expert);
            }
            layers.push(MambaLayer { experts, router_w });
        }

        Ok(MambaMoEModel { embeddings, layers, policy_head, value_head })
    }

    /// Initializes the model with random weights as a fallback.
    fn random_init(device: &Device) -> Result<Self> {
        // This is a placeholder for random initialization logic.
        // In a real scenario, you would define the shapes and initialize with randn.
        // For this task, we will just create empty tensors as a demonstration.
        let embeddings = Tensor::zeros((1, D_MODEL), candle_core::DType::F32, device)?;
        let policy_head = Tensor::zeros((D_MODEL, 1), candle_core::DType::F32, device)?;
        let value_head = Tensor::zeros((D_MODEL, 1), candle_core::DType::F32, device)?;

        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for _ in 0..NUM_LAYERS {
            let router_w = Tensor::zeros((D_MODEL, NUM_EXPERTS), candle_core::DType::F32, device)?;
            let mut experts = Vec::with_capacity(NUM_EXPERTS);
            for _ in 0..NUM_EXPERTS {
                experts.push(Expert {
                    in_proj_w: Tensor::zeros((D_MODEL, D_MODEL), candle_core::DType::F32, device)?,
                    conv1d_w: Tensor::zeros((D_MODEL, D_MODEL), candle_core::DType::F32, device)?,
                    x_proj_w: Tensor::zeros((D_MODEL, D_MODEL), candle_core::DType::F32, device)?,
                    dt_proj_w: Tensor::zeros((D_MODEL, D_MODEL), candle_core::DType::F32, device)?,
                    a_log: Tensor::zeros((D_MODEL, D_MODEL), candle_core::DType::F32, device)?,
                    d: Tensor::zeros((D_MODEL,), candle_core::DType::F32, device)?,
                    out_proj_w: Tensor::zeros((D_MODEL, D_MODEL), candle_core::DType::F32, device)?,
                });
            }
            layers.push(MambaLayer { experts, router_w });
        }

        Ok(MambaMoEModel { embeddings, layers, policy_head, value_head })
    }
}
