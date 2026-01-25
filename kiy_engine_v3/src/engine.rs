// src/engine.rs

use crate::mamba::{MambaBlock, MambaState, D_CONV, D_INNER, D_MODEL, D_STATE};
use crate::moe::{MoELayer, NUM_EXPERTS};
use chess::Board;
use ndarray::{Array1, Array2, Array3, s};
use ndarray_rand::rand_distr::Uniform;
#[cfg(feature = "noise")]
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use safetensors::SafeTensors;
use std::fs::File;
use std::io::Read;

const NUM_LAYERS: usize = 4;
pub const VOCAB_SIZE: usize = 768; // 12 pieces * 64 squares

/// Represents the full MoE-Mamba model, including weights and architecture constants.
///
/// This struct holds all the parameters of the neural network. It can be initialized
/// either with random weights or by loading from a `.safetensors` file.
pub struct Engine {
    /// Token embeddings for each piece-square combination. Maps a piece on a square
    /// to a high-dimensional vector representation.
    pub embeddings: Array2<f32>, // Shape: (VOCAB_SIZE, D_MODEL)

    /// The core of the model, consisting of 4 MoE-Mamba layers.
    pub layers: [MoELayer; NUM_LAYERS],

    /// Weights for the final layer normalization.
    pub norm_w: Array1<f32>, // Shape: (D_MODEL)

    /// Output head for the policy. Projects the final model state to a score for each possible move.
    pub policy_head: Array2<f32>, // Shape: (VOCAB_SIZE, D_MODEL)

    /// Output head for the value. Projects the final model state to a single floating-point
    /// value representing the evaluation of the current position.
    pub value_head: Array1<f32>, // Shape: (D_MODEL)
}

/// Encapsulates the complete state of a chess game at a specific point in time.
///
/// This includes not only the board position but also the sequence of moves made to reach it
/// and the corresponding hidden states of the Mamba model for each layer and expert.
/// The `Clone` trait is crucial for the search algorithm, which needs to create copies
/// of the state to explore different move sequences.
#[derive(Clone)]
pub struct GameState {
    /// The current board position.
    pub board: Board,

    /// The hidden states for each Mamba block in the model.
    /// The outer array is for the layers, the inner for the experts.
    pub mamba_states: [[MambaState; 8]; NUM_LAYERS],
}

impl GameState {
    /// Creates a new GameState from a starting board position.
    pub fn new(board: Board) -> Self {
        Self {
            board,
            mamba_states: std::array::from_fn(|_| std::array::from_fn(|_| MambaState::new())),
        }
    }
}

/// Root Mean Square Normalization
fn rms_norm(x: &mut Array1<f32>, weight: &Array1<f32>) {
    let n = x.len() as f32;
    let norm = (x.mapv(|v| v.powi(2)).sum() / n).sqrt() + 1e-6; // Add epsilon for stability
    *x = (&*x / norm) * weight;
}

impl Engine {
    /// Creates a new engine, loading weights from "model.safetensors" if it exists,
    /// otherwise initializing with random weights.
    pub fn new() -> Self {
        if let Ok(mut f) = File::open("model.safetensors") {
            println!("Loading model from model.safetensors...");
            let mut buffer: Vec<u8> = Vec::new();
            f.read_to_end(&mut buffer).expect("Failed to read safetensors file");
            let tensors = SafeTensors::deserialize(&buffer).expect("Failed to deserialize tensors");
            Self::from_tensors(&tensors)
        } else {
            println!("Warning: 'model.safetensors' not found. Initializing with random weights.");
            Self::new_random()
        }
    }

    /// Creates an engine from a SafeTensors object.
    fn from_tensors(tensors: &SafeTensors) -> Self {
        let embeddings = get_tensor_2d(tensors, "embeddings");

        let layers = std::array::from_fn(|l| MoELayer {
            router_w: get_tensor_2d(tensors, &format!("layer.{}.router_w", l)),
            experts: std::array::from_fn(|e| MambaBlock {
                in_proj_w: get_tensor_2d(tensors, &format!("layer.{}.expert.{}.in_proj_w", l, e)),
                conv1d_w: get_tensor_3d(tensors, &format!("layer.{}.expert.{}.conv1d_w", l, e)),
                conv1d_b: get_tensor_1d(tensors, &format!("layer.{}.expert.{}.conv1d_b", l, e)),
                x_proj_w: get_tensor_2d(tensors, &format!("layer.{}.expert.{}.x_proj_w", l, e)),
                dt_proj_w: get_tensor_1d(tensors, &format!("layer.{}.expert.{}.dt_proj_w", l, e)),
                dt_proj_b: get_tensor_1d(tensors, &format!("layer.{}.expert.{}.dt_proj_b", l, e)),
                a_log: get_tensor_2d(tensors, &format!("layer.{}.expert.{}.a_log", l, e)),
                d: get_tensor_1d(tensors, &format!("layer.{}.expert.{}.d", l, e)),
                out_proj_w: get_tensor_2d(tensors, &format!("layer.{}.expert.{}.out_proj_w", l, e)),
            }),
        });

        let norm_w = get_tensor_1d(tensors, "norm_w");
        let policy_head = get_tensor_2d(tensors, "policy_head");
        let value_head = get_tensor_1d(tensors, "value_head");

        Self {
            embeddings,
            layers,
            norm_w,
            policy_head,
            value_head,
        }
    }

    /// Initializes the engine with random weights.
    fn new_random() -> Self {
        let dist = Uniform::new(-0.1, 0.1).unwrap();

        let embeddings = Array2::random((VOCAB_SIZE, D_MODEL), dist);
        let layers = std::array::from_fn(|_l| MoELayer {
            router_w: Array2::random((NUM_EXPERTS, D_MODEL), dist),
            experts: std::array::from_fn(|_e| MambaBlock {
                in_proj_w: Array2::random((2 * D_INNER, D_MODEL), dist),
                conv1d_w: Array3::random((D_INNER, 1, D_CONV), dist),
                conv1d_b: Array1::random(D_INNER, dist),
                x_proj_w: Array2::random((D_INNER + 2 * D_STATE, D_INNER), dist),
                dt_proj_w: Array1::random(D_INNER, dist),
                dt_proj_b: Array1::random(D_INNER, dist),
                a_log: Array2::random((D_INNER, D_STATE), dist),
                d: Array1::random(D_INNER, dist),
                out_proj_w: Array2::random((D_MODEL, D_INNER), dist),
            }),
        });

        let norm_w = Array1::random(D_MODEL, dist);
        let policy_head = Array2::random((VOCAB_SIZE, D_MODEL), dist);
        let value_head = Array1::random(D_MODEL, dist);

        Self {
            embeddings,
            layers,
            norm_w,
            policy_head,
            value_head,
        }
    }
}

// Helper functions for loading tensors of different ranks
fn get_tensor_1d(tensors: &SafeTensors, name: &str) -> Array1<f32> {
    let tensor_view = tensors.tensor(name).expect(&format!("Tensor {} not found", name));
    let shape = tensor_view.shape();
    let data = tensor_view.data();
    let f32_data: Vec<f32> = data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    Array1::from_shape_vec(shape[0], f32_data).expect("Failed to create Array1")
}

fn get_tensor_2d(tensors: &SafeTensors, name: &str) -> Array2<f32> {
    let tensor_view = tensors.tensor(name).expect(&format!("Tensor {} not found", name));
    let shape = tensor_view.shape();
    let data = tensor_view.data();
    let f32_data: Vec<f32> = data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    Array2::from_shape_vec((shape[0], shape[1]), f32_data).expect("Failed to create Array2")
}

fn get_tensor_3d(tensors: &SafeTensors, name: &str) -> Array3<f32> {
    let tensor_view = tensors.tensor(name).expect(&format!("Tensor {} not found", name));
    let shape = tensor_view.shape();
    let data = tensor_view.data();
    let f32_data: Vec<f32> = data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    Array3::from_shape_vec((shape[0], shape[1], shape[2]), f32_data).expect("Failed to create Array3")
}

impl Engine {
    /// Performs a forward pass of the entire MoE-Mamba model for a single token.
    ///
    /// # Arguments
    /// * `token` - The input token (an index from 0 to VOCAB_SIZE-1).
    /// * `state` - The mutable GameState containing the Mamba hidden states.
    ///
    /// # Returns
    /// A tuple containing (policy_logits, value).
    pub fn forward(&self, token: usize, state: &mut GameState) -> (Array1<f32>, f32) {
        // 1. Embedding
        let mut x = self.embeddings.slice(s![token, ..]).to_owned();

        // 2. (Optional) Gaussian Noise
        #[cfg(feature = "noise")]
        {
            let std_dev = 0.01; // A small amount of noise
            let noise_dist = Normal::new(0.0, std_dev).unwrap();
            let noise = Array1::random(D_MODEL, noise_dist);
            x = x + noise;
        }

        // 3. MoE-Mamba Layers
        for i in 0..NUM_LAYERS {
            let residual = x.clone();
            let mut layer_input = x;

            // Pre-Norm
            rms_norm(&mut layer_input, &self.norm_w);

            let layer_output = self.layers[i].forward(&layer_input, &mut state.mamba_states[i]);
            x = residual + layer_output;
        }

        // 4. Final Normalization
        rms_norm(&mut x, &self.norm_w);

        // 5. Output Heads
        let policy_logits = self.policy_head.dot(&x);
        let value = self.value_head.dot(&x);

        (policy_logits, value)
    }
}
