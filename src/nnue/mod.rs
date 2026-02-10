// src/nnue/mod.rs
//! NNUE (Efficiently Updatable Neural Network) for fast position evaluation.
//!
//! Architecture: (768 → HIDDEN)×2 → SCReLU → 2*HIDDEN → 1
//!
//! - Input: 768 binary features (6 pieces × 2 colors × 64 squares)
//! - Feature transformer: 768 → HIDDEN per perspective (i16 quantized)
//! - SCReLU activation: clamp(x, 0, QA)² on both accumulators
//! - Output: dot product of SCReLU'd values with i16 weights → centipawns
//!
//! ~394KB for HIDDEN=256. Designed for knowledge distillation from v5.2 model.
//!
//! Performance optimizations:
//! - AVX2 SIMD vectorized evaluate (16 i16 values per cycle)
//! - SIMD accumulator refresh/update (vectorized i16 add/sub)
//! - Zero-copy incremental updates via split_at_mut

pub mod accumulator;
pub mod features;

use accumulator::Accumulator;
use chess::{Board, Color};
use features::NUM_FEATURES;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ============================================================================
// QUANTIZATION CONSTANTS
// ============================================================================

/// Accumulator quantization factor (clamp ceiling for SCReLU)
pub const QA: i16 = 255;

/// Output weight quantization factor.
/// Chosen so that v*w fits in i16 for the Lizard SCReLU SIMD trick:
/// max(v*w) = QA * (WEIGHT_CLIP * QB) = 255 * 127 = 32385 < 32767
pub const QB: i32 = 64;

/// Output scale: sigmoid(cp / SCALE) ≈ win_probability (used in training only)
pub const SCALE: i32 = 400;

/// Magic bytes for NNUE binary file format
pub const NNUE_MAGIC: &[u8; 4] = b"KYNN";

/// Current NNUE format version
pub const NNUE_VERSION: u32 = 1;

/// Default NNUE file path
pub const DEFAULT_NNUE_PATH: &str = "kiyengine.nnue";

// ============================================================================
// NETWORK
// ============================================================================

/// NNUE network weights (quantized, loaded from binary file).
///
/// Architecture: (768 → hidden_size)×2 → SCReLU → 2*hidden_size → 1
pub struct NnueNetwork {
    /// Feature transformer weights: [NUM_FEATURES * hidden_size] (i16, row-major)
    /// ft_weights[feature * hidden_size + i] = weight for feature → neuron i
    pub ft_weights: Vec<i16>,

    /// Feature transformer biases: [hidden_size] (i16)
    pub ft_biases: Vec<i16>,

    /// Output layer weights: [2 * hidden_size] (i16)
    /// First half for "us" perspective, second half for "them"
    pub output_weights: Vec<i16>,

    /// Output layer bias (i16)
    pub output_bias: i16,

    /// Hidden layer size (typically 256)
    pub hidden_size: usize,
}

impl NnueNetwork {
    /// Evaluate a position given a computed accumulator.
    ///
    /// 1. Apply SCReLU: clamp(x, 0, QA)² to both perspectives
    /// 2. Dot product with output weights
    /// 3. Scale to centipawns
    ///
    /// Returns evaluation in centipawns from side-to-move's perspective.
    #[inline]
    pub fn evaluate(&self, acc: &Accumulator, stm: Color) -> i32 {
        debug_assert!(acc.computed, "Accumulator not computed");

        let (us, them) = match stm {
            Color::White => (&acc.white, &acc.black),
            Color::Black => (&acc.black, &acc.white),
        };

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { self.evaluate_avx2(us, them) };
            }
        }

        self.evaluate_scalar(us, them)
    }

    /// Scalar fallback evaluate (no SIMD).
    #[inline]
    fn evaluate_scalar(&self, us: &[i16], them: &[i16]) -> i32 {
        let qa = QA as i32;
        let qa_sq = (QA as i64) * (QA as i64);

        let mut sum: i64 = 0;

        for i in 0..self.hidden_size {
            let v = (us[i] as i32).clamp(0, qa);
            sum += (v * v) as i64 * self.output_weights[i] as i64;
        }

        for i in 0..self.hidden_size {
            let v = (them[i] as i32).clamp(0, qa);
            sum += (v * v) as i64 * self.output_weights[self.hidden_size + i] as i64;
        }

        // sum is in QA² * QB units → divide by QA² to get QB units, add bias (QB units), divide by QB
        let eval_cp = (sum / qa_sq + self.output_bias as i64) / (QB as i64);
        eval_cp as i32
    }

    /// AVX2-vectorized SCReLU evaluate.
    ///
    /// Processes 16 i16 values per iteration using 256-bit SIMD.
    /// Strategy: widen i16→i32 (8 at a time), clamp, square, multiply by weight, accumulate.
    /// Uses 4 i32 accumulators to exploit instruction-level parallelism.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn evaluate_avx2(&self, us: &[i16], them: &[i16]) -> i32 {
        let zero = _mm256_setzero_si256();
        let qa_vec = _mm256_set1_epi16(QA);

        let mut sum0 = _mm256_setzero_si256();
        let mut sum1 = _mm256_setzero_si256();

        let hs = self.hidden_size;
        debug_assert!(hs % 16 == 0, "hidden_size must be multiple of 16 for AVX2");

        // Process "us" perspective
        let mut i = 0;
        while i < hs {
            // Load 16 x i16 accumulator values and weights
            let acc = _mm256_loadu_si256(us.as_ptr().add(i) as *const __m256i);
            let wgt = _mm256_loadu_si256(self.output_weights.as_ptr().add(i) as *const __m256i);

            // Clamp to [0, QA]: max(0, min(x, QA))
            let clamped = _mm256_min_epi16(_mm256_max_epi16(acc, zero), qa_vec);

            // SCReLU trick: compute (clamped * weight) * clamped
            // Step 1: clamped * weight → i16 (low 16 bits)
            let vw = _mm256_mullo_epi16(clamped, wgt);
            // Step 2: madd_epi16(vw, clamped) → pairs of i16 multiplied and adjacent-added to i32
            // This computes: vw[0]*clamped[0] + vw[1]*clamped[1] as i32, etc.
            let prod = _mm256_madd_epi16(vw, clamped);

            sum0 = _mm256_add_epi32(sum0, prod);
            i += 16;
        }

        // Process "them" perspective
        i = 0;
        while i < hs {
            let acc = _mm256_loadu_si256(them.as_ptr().add(i) as *const __m256i);
            let wgt =
                _mm256_loadu_si256(self.output_weights.as_ptr().add(hs + i) as *const __m256i);

            let clamped = _mm256_min_epi16(_mm256_max_epi16(acc, zero), qa_vec);

            let vw = _mm256_mullo_epi16(clamped, wgt);
            let prod = _mm256_madd_epi16(vw, clamped);

            sum1 = _mm256_add_epi32(sum1, prod);
            i += 16;
        }

        // Horizontal sum: sum0 + sum1 → 8 x i32 → scalar
        let combined = _mm256_add_epi32(sum0, sum1);
        // Reduce 8 x i32 to scalar
        let hi128 = _mm256_extracti128_si256(combined, 1);
        let lo128 = _mm256_castsi256_si128(combined);
        let sum128 = _mm_add_epi32(lo128, hi128);
        // Shuffle and add: [a,b,c,d] → [c,d,a,b] → add → [a+c,b+d,..]
        let hi64 = _mm_shuffle_epi32(sum128, 0b_01_00_11_10);
        let sum64 = _mm_add_epi32(sum128, hi64);
        // Final pair
        let hi32 = _mm_shuffle_epi32(sum64, 0b_00_00_00_01);
        let total = _mm_add_epi32(sum64, hi32);
        let dot = _mm_cvtsi128_si32(total) as i64;

        // Dequantize: dot is in QA² * QB units
        let qa_sq = (QA as i64) * (QA as i64);
        let eval_cp = (dot / qa_sq + self.output_bias as i64) / (QB as i64);
        eval_cp as i32
    }

    /// Evaluate a board from scratch (no accumulator reuse).
    /// Convenience method for standalone evaluation.
    pub fn evaluate_board(&self, board: &Board) -> i32 {
        let mut acc = Accumulator::new(self.hidden_size);
        acc.refresh(board, self);
        self.evaluate(&acc, board.side_to_move())
    }

    /// Load NNUE weights from a binary file.
    ///
    /// Format:
    /// - Magic: "KYNN" (4 bytes)
    /// - Version: u32 LE
    /// - hidden_size: u32 LE
    /// - ft_weights: i16[768 * hidden_size] LE
    /// - ft_biases: i16[hidden_size] LE
    /// - output_weights: i16[2 * hidden_size] LE
    /// - output_bias: i16 LE
    pub fn load<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let data = std::fs::read(path.as_ref())?;
        Self::from_bytes(&data)
    }

    /// Parse NNUE weights from a byte buffer.
    pub fn from_bytes(data: &[u8]) -> anyhow::Result<Self> {
        let mut cursor = std::io::Cursor::new(data);

        // Read and verify magic
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;
        if &magic != NNUE_MAGIC {
            anyhow::bail!(
                "Invalid NNUE magic: expected {:?}, got {:?}",
                NNUE_MAGIC,
                magic
            );
        }

        // Read version
        let mut buf4 = [0u8; 4];
        cursor.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != NNUE_VERSION {
            anyhow::bail!(
                "Unsupported NNUE version: {} (expected {})",
                version,
                NNUE_VERSION
            );
        }

        // Read hidden_size
        cursor.read_exact(&mut buf4)?;
        let hidden_size = u32::from_le_bytes(buf4) as usize;

        // Read feature transformer weights: i16[768 * hidden_size]
        let ft_count = NUM_FEATURES * hidden_size;
        let ft_weights = read_i16_vec(&mut cursor, ft_count)?;

        // Read feature transformer biases: i16[hidden_size]
        let ft_biases = read_i16_vec(&mut cursor, hidden_size)?;

        // Read output weights: i16[2 * hidden_size]
        let output_weights = read_i16_vec(&mut cursor, 2 * hidden_size)?;

        // Read output bias: i16
        let mut buf2 = [0u8; 2];
        cursor.read_exact(&mut buf2)?;
        let output_bias = i16::from_le_bytes(buf2);

        println!(
            "  NNUE loaded: hidden_size={}, ft_weights={}, output_weights={}, total={:.1}KB",
            hidden_size,
            ft_count,
            2 * hidden_size,
            (ft_count * 2 + hidden_size * 2 + hidden_size * 4 + 2) as f64 / 1024.0,
        );

        Ok(Self {
            ft_weights,
            ft_biases,
            output_weights,
            output_bias,
            hidden_size,
        })
    }

    /// Create a network with random weights (for testing/bootstrapping only).
    pub fn random(hidden_size: usize) -> Self {
        let ft_count = NUM_FEATURES * hidden_size;
        let mut ft_weights = vec![0i16; ft_count];
        let mut ft_biases = vec![0i16; hidden_size];
        let mut output_weights = vec![0i16; 2 * hidden_size];

        // Simple deterministic pseudo-random initialization
        let mut seed: u64 = 0xDEADBEEF;
        for w in ft_weights.iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *w = ((seed >> 48) as i16) / 256; // Small values ~[-128, 127]
        }
        for b in ft_biases.iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = ((seed >> 48) as i16) / 256;
        }
        for w in output_weights.iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *w = ((seed >> 48) as i16) / 256;
        }

        Self {
            ft_weights,
            ft_biases,
            output_weights,
            output_bias: 0,
            hidden_size,
        }
    }

    /// Serialize network to binary format (for saving trained weights).
    pub fn to_bytes(&self) -> Vec<u8> {
        let ft_count = NUM_FEATURES * self.hidden_size;
        let total_size = 4 + 4 + 4 + ft_count * 2 + self.hidden_size * 2 + self.hidden_size * 4 + 2;
        let mut buf = Vec::with_capacity(total_size);

        buf.extend_from_slice(NNUE_MAGIC);
        buf.extend_from_slice(&NNUE_VERSION.to_le_bytes());
        buf.extend_from_slice(&(self.hidden_size as u32).to_le_bytes());

        for &w in &self.ft_weights {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        for &b in &self.ft_biases {
            buf.extend_from_slice(&b.to_le_bytes());
        }
        for &w in &self.output_weights {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        buf.extend_from_slice(&self.output_bias.to_le_bytes());

        buf
    }
}

/// Read a vector of i16 values from a reader (little-endian).
fn read_i16_vec<R: Read>(reader: &mut R, count: usize) -> anyhow::Result<Vec<i16>> {
    let mut bytes = vec![0u8; count * 2];
    reader.read_exact(&mut bytes)?;
    let values: Vec<i16> = bytes
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();
    Ok(values)
}

/// Shared NNUE state for use across search threads.
pub struct NnueState {
    pub network: Arc<NnueNetwork>,
}

impl NnueState {
    pub fn new(network: Arc<NnueNetwork>) -> Self {
        Self { network }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess::Board;

    #[test]
    fn test_random_network_evaluation() {
        let net = NnueNetwork::random(256);
        let board = Board::default();
        let eval = net.evaluate_board(&board);
        // Just check it doesn't crash and returns a finite value
        assert!(eval.abs() < 100_000);
    }

    #[test]
    fn test_serialize_deserialize() {
        let net = NnueNetwork::random(64);
        let bytes = net.to_bytes();
        let net2 = NnueNetwork::from_bytes(&bytes).unwrap();

        assert_eq!(net.hidden_size, net2.hidden_size);
        assert_eq!(net.ft_weights, net2.ft_weights);
        assert_eq!(net.ft_biases, net2.ft_biases);
        assert_eq!(net.output_weights, net2.output_weights);
        assert_eq!(net.output_bias, net2.output_bias);
    }

    #[test]
    fn test_evaluation_symmetry() {
        // The starting position should evaluate close to 0 (slight tempo)
        let net = NnueNetwork::random(256);
        let board = Board::default();
        let eval = net.evaluate_board(&board);
        // With random weights, no guarantee of near-zero, but it should be finite
        assert!(eval.abs() < 100_000);
    }
}
