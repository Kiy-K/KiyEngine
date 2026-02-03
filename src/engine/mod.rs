// src/engine/mod.rs

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Embedding, Linear, Module, VarBuilder};
use memmap2::Mmap;
use std::fs::File;

pub const VOCAB_SIZE: usize = 4608;
pub const CONTEXT_LENGTH: usize = 32;
pub const D_MODEL: usize = 1024;
pub const NUM_LAYERS: usize = 4;

fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x_sq = x.sqr()?;
    let mean_sq = x_sq.mean_keepdim(D::Minus1)?;
    let norm = (mean_sq + eps)?.sqrt()?;
    let x_normed = x.broadcast_div(&norm)?;
    x_normed.broadcast_mul(weight)
}

pub struct BitLinear {
    pub weight_ternary: Vec<f32>, // Pre-quantized {-1, 0, 1}
    pub in_dim: usize,
    pub out_dim: usize,
    pub rms_norm_weight: Tensor,
    pub eps: f64,
}

impl BitLinear {
    pub fn load(prefix: &str, vb: &VarBuilder, in_dim: usize, out_dim: usize) -> Result<Self> {
        let weight = vb.get((out_dim, in_dim), &format!("{}.weight", prefix))?;
        // Pre-quantize and convert to flat Vec for SIMD optimization
        let weight_ternary = weight.clamp(-1.0, 1.0)?.round()?.to_vec2::<f32>()?
            .into_iter().flatten().collect();
            
        let rms_norm_weight = vb.get((in_dim,), &format!("{}.norm.weight", prefix))?;
        Ok(Self {
            weight_ternary,
            in_dim,
            out_dim,
            rms_norm_weight,
            eps: 1e-5,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = rms_norm(x, &self.rms_norm_weight, self.eps)?;
        
        let device = x.device();
        if device.is_cpu() {
            // Use custom SIMD kernels for CPU
            let x_flat = x.flatten_all()?.to_vec1::<f32>()?;
            let mut output = vec![0.0f32; self.out_dim];
            
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx512f") {
                    crate::simd::bitnet_ternary_gemm_avx512(&x_flat, &self.weight_ternary, &mut output, self.in_dim, self.out_dim);
                } else if is_x86_feature_detected!("avx2") {
                    crate::simd::bitnet_ternary_gemm_avx2(&x_flat, &self.weight_ternary, &mut output, self.in_dim, self.out_dim);
                } else {
                    // Optimized scalar fallback
                    for i in 0..self.out_dim {
                        let mut sum = 0.0;
                        let row_offset = i * self.in_dim;
                        for j in 0..self.in_dim {
                            let w = self.weight_ternary[row_offset + j];
                            if w > 0.5 { sum += x_flat[j]; }
                            else if w < -0.5 { sum -= x_flat[j]; }
                        }
                        output[i] += sum;
                    }
                }
            }
            
            Tensor::from_vec(output, (1, self.out_dim), device)
        } else {
            // Fallback for CUDA/Other devices (use standard MatMul)
            // Note: In a production V4.4, we would also have a CUDA kernel for this.
            let weights_tensor = Tensor::from_vec(self.weight_ternary.clone(), (self.out_dim, self.in_dim), device)?.t()?;
            x.broadcast_matmul(&weights_tensor)
        }
    }
}

pub struct ValueHead {
    pub norm_weight: Tensor,
    pub linear1: Linear,
    pub linear2: Linear,
}

impl ValueHead {
    pub fn load(vb: &VarBuilder) -> Result<Self> {
        let norm_weight = vb.get((D_MODEL,), "value_head.0.weight")?;
        let linear1_w = vb.get((512, D_MODEL), "value_head.1.weight")?;
        let linear2_w = vb.get((1, 512), "value_head.3.weight")?;
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

pub struct PolicyHead {
    pub norm_weight: Tensor,
    pub linear: Linear,
}

impl PolicyHead {
    pub fn load(vb: &VarBuilder) -> Result<Self> {
        let norm_weight = vb.get((D_MODEL,), "policy_head.0.weight")?;
        let linear_w = vb.get((VOCAB_SIZE, D_MODEL), "policy_head.1.weight")?;
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

pub struct Engine {
    pub embedding: Embedding,
    pub pos_embed: Tensor,
    pub layers: Vec<BitLinear>,
    pub policy_head: PolicyHead,
    pub value_head: ValueHead,
    pub device: Device,
}

impl Engine {
    pub fn new() -> anyhow::Result<Self> {
        let device = if cfg!(feature = "cuda") {
            Device::new_cuda(0).unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        };
        Self::load_from_safetensors("kiyengine_v4.safetensors", device)
    }

    pub fn load_from_safetensors(model_path: &str, device: Device) -> anyhow::Result<Self> {
        let file = File::open(model_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let tensors = candle_core::safetensors::load_buffer(&mmap, &device)?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        let embed_w = vb.get((VOCAB_SIZE, D_MODEL), "embed.weight")?;
        let embedding = Embedding::new(embed_w, D_MODEL);
        let pos_embed = vb.get((1, CONTEXT_LENGTH, D_MODEL), "pos_embed")?;

        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for i in 0..NUM_LAYERS {
            layers.push(BitLinear::load(
                &format!("layers.{}.linear", i),
                &vb,
                D_MODEL,
                D_MODEL,
            )?);
        }

        let policy_head = PolicyHead::load(&vb)?;
        let value_head = ValueHead::load(&vb)?;

        Ok(Self {
            embedding,
            pos_embed,
            layers,
            policy_head,
            value_head,
            device,
        })
    }

    pub fn forward(&self, tokens: &[usize]) -> Result<(Tensor, f32)> {
        let seq_len = tokens.len();
        if seq_len == 0 {
            return Err(candle_core::Error::Msg("Empty tokens".to_string()));
        }

        // Take at most CONTEXT_LENGTH tokens
        let take_len = seq_len.min(CONTEXT_LENGTH);
        let tokens_slice = &tokens[seq_len - take_len..];

        let tokens_u32: Vec<u32> = tokens_slice.iter().map(|&t| t as u32).collect();
        let tokens_tensor = Tensor::new(tokens_u32.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut x = self.embedding.forward(&tokens_tensor)?;

        let pos_slice = self.pos_embed.narrow(1, 0, take_len)?;
        x = x.broadcast_add(&pos_slice)?;

        for layer in &self.layers {
            let layer_out = layer.forward(&x)?;
            x = (x + layer_out)?;
        }

        let last_token_x = x.narrow(1, take_len - 1, 1)?.squeeze(1)?;
        let policy_logits = self.policy_head.forward(&last_token_x)?.squeeze(0)?;
        let value_out = self
            .value_head
            .forward(&last_token_x)?
            .squeeze(0)?
            .squeeze(0)?;
        let eval_score = value_out.tanh()?.to_scalar::<f32>()?;

        Ok((policy_logits, eval_score))
    }
}
