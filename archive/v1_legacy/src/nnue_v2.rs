use std::fs::File;
use std::io::{self, Read};
use std::mem;
use std::sync::Arc;
use memmap2::{Mmap, MmapOptions};
use std::arch::x86_64::*;

// Structs and basic impls
#[derive(Clone)]
#[repr(align(64))]
pub struct Accumulator {
    pub weights: [i16; 32768],
}

impl Accumulator {
    fn new() -> Box<Self> {
        unsafe {
            let layout = std::alloc::Layout::from_size_align_unchecked(mem::size_of::<Self>(), 64);
            Box::from_raw(std::alloc::alloc_zeroed(layout) as *mut Self)
        }
    }
}

pub struct NeuralNetworkData {
    feature_weights: Mmap,
    output_weights: Mmap,
    output_bias: Mmap,
    quantization_scale: f32,
}

#[derive(Clone)]
pub struct KiyNNUE {
    pub white_acc: Box<Accumulator>,
    pub black_acc: Box<Accumulator>,
    pub weights: Arc<NeuralNetworkData>,
}

pub fn get_feature_indices(piece: crate::defs::PieceType, color: crate::defs::Color, sq: usize) -> (usize, usize) {
    let color_idx = color as usize;
    let piece_idx = piece as usize;
    let white_sq = sq;
    let black_sq = sq ^ 56;
    let w_idx = (color_idx * 6 + piece_idx) * 64 + white_sq;
    let b_idx = (color_idx * 6 + piece_idx) * 64 + black_sq;
    (w_idx, b_idx)
}

#[target_feature(enable = "avx2")]
pub unsafe fn update_accumulator_add(nn_data: &Arc<NeuralNetworkData>, acc: &mut Accumulator, feature_idx: usize) {
    let weights_ptr = nn_data.feature_weights.as_ptr() as *const i16;
    let feature_offset = feature_idx * 32768;
    for i in (0..32768).step_by(16) {
        let acc_ptr = acc.weights.as_mut_ptr().add(i);
        let weights_slice_ptr = weights_ptr.add(feature_offset + i);
        let acc_vec = _mm256_load_si256(acc_ptr as *const _);
        let weights_vec = _mm256_loadu_si256(weights_slice_ptr as *const _);
        let sum = _mm256_add_epi16(acc_vec, weights_vec);
        _mm256_store_si256(acc_ptr as *mut _, sum);
    }
}

#[target_feature(enable = "avx2")]
pub unsafe fn update_accumulator_sub(nn_data: &Arc<NeuralNetworkData>, acc: &mut Accumulator, feature_idx: usize) {
    let weights_ptr = nn_data.feature_weights.as_ptr() as *const i16;
    let feature_offset = feature_idx * 32768;
    for i in (0..32768).step_by(16) {
        let acc_ptr = acc.weights.as_mut_ptr().add(i);
        let weights_slice_ptr = weights_ptr.add(feature_offset + i);
        let acc_vec = _mm256_load_si256(acc_ptr as *const _);
        let weights_vec = _mm256_loadu_si256(weights_slice_ptr as *const _);
        let diff = _mm256_sub_epi16(acc_vec, weights_vec);
        _mm256_store_si256(acc_ptr as *mut _, diff);
    }
}

// Main implementation block
impl KiyNNUE {
    pub fn load(weights_path: &str) -> io::Result<Self> {
        let file = File::open(weights_path)?;
        let mut header_buf = [0u8; 64];
        file.try_clone()?.read_exact(&mut header_buf)?;

        let mut cursor = 0;

        let magic_bytes_raw = &header_buf[cursor..cursor + 16];
        let magic_bytes = magic_bytes_raw.split(|&b| b == 0).next().unwrap_or(&[]);
        if std::str::from_utf8(magic_bytes).unwrap_or("INVALID") != "KIY-NNUE-V2" {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid NNUE magic string"));
        }
        cursor += 16;

        let input_dim = u32::from_le_bytes(header_buf[cursor..cursor+4].try_into().unwrap());
        cursor += 4;
        let hidden_dim = u32::from_le_bytes(header_buf[cursor..cursor+4].try_into().unwrap());
        cursor += 4;
        let output_dim = u32::from_le_bytes(header_buf[cursor..cursor+4].try_into().unwrap());
        cursor += 4;

        if input_dim != 768 || hidden_dim != 32768 || output_dim != 1 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Unexpected NNUE dimensions"));
        }

        let quantization_scale = f32::from_le_bytes(header_buf[cursor..cursor+4].try_into().unwrap());

        let mut current_offset = 64u64;
        let feature_weights_size = (input_dim * hidden_dim) as usize * mem::size_of::<i16>();
        let feature_weights = unsafe {
            MmapOptions::new().offset(current_offset).len(feature_weights_size).map(&file)?
        };
        current_offset += feature_weights_size as u64;

        let output_weights_size = (hidden_dim * output_dim) as usize * mem::size_of::<i16>();
        let output_weights = unsafe {
            MmapOptions::new().offset(current_offset).len(output_weights_size).map(&file)?
        };
        current_offset += output_weights_size as u64;

        let output_bias_size = output_dim as usize * mem::size_of::<i16>();
        let output_bias = unsafe {
            MmapOptions::new().offset(current_offset).len(output_bias_size).map(&file)?
        };

        let nn_data = NeuralNetworkData {
            feature_weights,
            output_weights,
            output_bias,
            quantization_scale,
        };

        Ok(KiyNNUE {
            white_acc: Accumulator::new(),
            black_acc: Accumulator::new(),
            weights: Arc::new(nn_data),
        })
    }

    pub fn evaluate(&self, is_white_turn: bool) -> i32 {
        let (us_acc, them_acc) = if is_white_turn {
            (&self.white_acc, &self.black_acc)
        } else {
            (&self.black_acc, &self.white_acc)
        };

        let mut output: i64 = 0;

        let output_weights: &[i16] = unsafe {
            std::slice::from_raw_parts(self.weights.output_weights.as_ptr() as *const i16, 32768)
        };
        let output_bias: i16 = unsafe {
            *(self.weights.output_bias.as_ptr() as *const i16)
        };

        for i in 0..32768 {
            let white_val = us_acc.weights[i].max(0).min(127) as i64;
            let black_val = them_acc.weights[i].max(0).min(127) as i64;
            let combined_activation = white_val + black_val;
            output += combined_activation * output_weights[i] as i64;
        }

        ((output + output_bias as i64) as f32 / self.weights.quantization_scale).round() as i32
    }

    fn get_halfkp_features(&self, board: &crate::board::Board) -> (Vec<usize>, Vec<usize>) {
        let mut white_indices = Vec::new();
        let mut black_indices = Vec::new();
        for sq in 0..64 {
            if let Some((piece_type, color)) = board.piece_at_sq(sq) {
                let (w_idx, b_idx) = get_feature_indices(piece_type, color, sq);
                white_indices.push(w_idx);
                black_indices.push(b_idx);
            }
        }
        (white_indices, black_indices)
    }

    pub fn refresh_accumulators(&mut self, board: &crate::board::Board) {
        self.white_acc.weights.iter_mut().for_each(|w| *w = 0);
        self.black_acc.weights.iter_mut().for_each(|w| *w = 0);
        let (white_features, black_features) = self.get_halfkp_features(board);

        unsafe {
            for &idx in &white_features {
                update_accumulator_add(&self.weights, &mut self.white_acc, idx);
            }
            for &idx in &black_features {
                update_accumulator_add(&self.weights, &mut self.black_acc, idx);
            }
        }
    }
}
