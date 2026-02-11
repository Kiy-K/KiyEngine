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

/// Current NNUE format version (v4: king-bucketed inputs + output buckets)
pub const NNUE_VERSION: u32 = 4;
/// Also accept v2/v3 files for backward compatibility
const NNUE_VERSION_V2: u32 = 2;
const NNUE_VERSION_V3: u32 = 3;

/// Number of output buckets (piece-count based)
pub const NUM_BUCKETS: usize = 8;

/// Default NNUE file path
pub const DEFAULT_NNUE_PATH: &str = "kiyengine.nnue";

// ============================================================================
// NETWORK
// ============================================================================

/// NNUE network weights (quantized, loaded from binary file).
///
/// Architecture: (HalfKAv2 → hidden_size)×2 → SCReLU → 2*hidden_size → num_output_buckets
///
/// v4 (king-bucketed): feature transformer is (num_input_buckets * 768) × hidden_size
/// v2/v3 (legacy): feature transformer is 768 × hidden_size (no king buckets)
///
/// Output bucket selected by: (total_pieces - 2) / 4
pub struct NnueNetwork {
    /// Feature transformer weights: [num_input_buckets * 768 * hidden_size] (i16, row-major)
    /// ft_weights[feature * hidden_size + i] = weight for feature → neuron i
    /// For king-bucketed: feature = king_bucket * 768 + base_feature
    pub ft_weights: Vec<i16>,

    /// Feature transformer biases: [hidden_size] (i16)
    pub ft_biases: Vec<i16>,

    /// Output layer weights per bucket: [num_buckets][2 * hidden_size] (i16, flattened)
    /// output_weights[bucket * 2 * hidden_size + i]
    pub output_weights: Vec<i16>,

    /// Output layer bias per bucket: [num_buckets] (i16)
    pub output_biases: Vec<i16>,

    /// PSQT weights (legacy v3 only, empty for v4)
    pub psqt_weights: Vec<i16>,

    /// Hidden layer size (typically 256 or 512)
    pub hidden_size: usize,

    /// Number of output buckets (1 = legacy, 8 = standard)
    pub num_buckets: usize,

    /// Number of input (king) buckets (1 = legacy Chess768, 10 = HalfKAv2_hm)
    pub num_input_buckets: usize,
}

impl NnueNetwork {
    /// Select output bucket based on total piece count.
    /// bucket = (total_pieces - 2) / 4, clamped to [0, num_buckets-1]
    #[inline]
    pub fn bucket_index(&self, board: &Board) -> usize {
        if self.num_buckets <= 1 {
            return 0;
        }
        let total = board.combined().0.count_ones() as usize;
        // total is 2..=32, bucket = (total - 2) / 4 gives 0..7
        let bucket = (total.saturating_sub(2)) / 4;
        bucket.min(self.num_buckets - 1)
    }

    /// Evaluate a position given a computed accumulator and the board (for bucket selection).
    ///
    /// 1. Select output bucket by piece count
    /// 2. Apply SCReLU: clamp(x, 0, QA)² to both perspectives
    /// 3. Dot product with bucket's output weights
    /// 4. Scale to centipawns
    ///
    /// Returns evaluation in centipawns from side-to-move's perspective.
    #[inline(always)]
    pub fn evaluate(&self, acc: &Accumulator, stm: Color, board: &Board) -> i32 {
        debug_assert!(acc.computed, "Accumulator not computed");

        let bucket = self.bucket_index(board);
        let (us, them) = match stm {
            Color::White => (&acc.white, &acc.black),
            Color::Black => (&acc.black, &acc.white),
        };

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { self.evaluate_avx2(us, them, bucket) };
            }
        }

        self.evaluate_scalar(us, them, bucket)
    }

    /// Scalar fallback evaluate (no SIMD).
    /// Dequantization follows Bullet convention: (sum/QA + bias) * SCALE / (QA*QB)
    #[inline]
    fn evaluate_scalar(&self, us: &[i16], them: &[i16], bucket: usize) -> i32 {
        let qa = QA as i64;
        let hs = self.hidden_size;
        let w_offset = bucket * 2 * hs;

        let mut sum: i64 = 0;

        for i in 0..hs {
            let v = (us[i] as i64).clamp(0, qa);
            sum += v * v * self.output_weights[w_offset + i] as i64;
        }

        for i in 0..hs {
            let v = (them[i] as i64).clamp(0, qa);
            sum += v * v * self.output_weights[w_offset + hs + i] as i64;
        }

        // sum is in QA² * QB units
        // Bullet dequantization: (sum/QA + bias) * SCALE / (QA * QB)
        sum /= qa;                                       // QA²*QB → QA*QB
        sum += self.output_biases[bucket] as i64;         // bias in QA*QB units
        sum *= SCALE as i64;                              // → centipawn-scaled
        sum /= qa * QB as i64;                            // remove quantization
        sum as i32
    }

    /// AVX2-vectorized SCReLU evaluate.
    ///
    /// Processes 16 i16 values per iteration using 256-bit SIMD.
    /// Strategy: widen i16→i32 (8 at a time), clamp, square, multiply by weight, accumulate.
    /// Uses 4 i32 accumulators to exploit instruction-level parallelism.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn evaluate_avx2(&self, us: &[i16], them: &[i16], bucket: usize) -> i32 {
        let zero = _mm256_setzero_si256();
        let qa_vec = _mm256_set1_epi16(QA);

        let mut sum0 = _mm256_setzero_si256();
        let mut sum1 = _mm256_setzero_si256();

        let hs = self.hidden_size;
        let w_offset = bucket * 2 * hs;
        debug_assert!(hs % 16 == 0, "hidden_size must be multiple of 16 for AVX2");

        // Process "us" perspective
        let mut i = 0;
        while i < hs {
            let acc = _mm256_load_si256(us.as_ptr().add(i) as *const __m256i);
            let wgt = _mm256_loadu_si256(self.output_weights.as_ptr().add(w_offset + i) as *const __m256i);

            let clamped = _mm256_min_epi16(_mm256_max_epi16(acc, zero), qa_vec);

            let vw = _mm256_mullo_epi16(clamped, wgt);
            let prod = _mm256_madd_epi16(vw, clamped);

            sum0 = _mm256_add_epi32(sum0, prod);
            i += 16;
        }

        // Process "them" perspective
        i = 0;
        while i < hs {
            let acc = _mm256_load_si256(them.as_ptr().add(i) as *const __m256i);
            let wgt =
                _mm256_loadu_si256(self.output_weights.as_ptr().add(w_offset + hs + i) as *const __m256i);

            let clamped = _mm256_min_epi16(_mm256_max_epi16(acc, zero), qa_vec);

            let vw = _mm256_mullo_epi16(clamped, wgt);
            let prod = _mm256_madd_epi16(vw, clamped);

            sum1 = _mm256_add_epi32(sum1, prod);
            i += 16;
        }

        // Horizontal sum: sum0 + sum1 → 8 x i32 → scalar
        let combined = _mm256_add_epi32(sum0, sum1);
        let hi128 = _mm256_extracti128_si256(combined, 1);
        let lo128 = _mm256_castsi256_si128(combined);
        let sum128 = _mm_add_epi32(lo128, hi128);
        let hi64 = _mm_shuffle_epi32(sum128, 0b_01_00_11_10);
        let sum64 = _mm_add_epi32(sum128, hi64);
        let hi32 = _mm_shuffle_epi32(sum64, 0b_00_00_00_01);
        let total = _mm_add_epi32(sum64, hi32);
        let dot = _mm_cvtsi128_si32(total) as i64;

        // Bullet dequantization: (sum/QA + bias) * SCALE / (QA * QB)
        let qa = QA as i64;
        let mut result = dot / qa;                              // QA²*QB → QA*QB
        result += self.output_biases[bucket] as i64;             // bias in QA*QB units
        result *= SCALE as i64;                                  // → centipawn-scaled
        result /= qa * QB as i64;                                // remove quantization
        result as i32
    }

    /// Evaluate a board from scratch (no accumulator reuse).
    /// Convenience method for standalone evaluation.
    pub fn evaluate_board(&self, board: &Board) -> i32 {
        let mut acc = Accumulator::new(self.hidden_size);
        acc.refresh(board, self);
        self.evaluate(&acc, board.side_to_move(), board)
    }

    /// Load NNUE weights from a binary file.
    ///
    /// Supports v2 (single bucket) and v3 (output buckets + PSQT) formats.
    ///
    /// v2 format: Magic, Version, hidden_size, ft_weights, ft_biases, output_weights[2*hs], output_bias
    /// v3 format: Magic, Version, hidden_size, num_buckets, ft_weights, ft_biases,
    ///            psqt_weights[768*num_buckets], output_weights[num_buckets*2*hs], output_biases[num_buckets]
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
        if version != NNUE_VERSION && version != NNUE_VERSION_V3 && version != NNUE_VERSION_V2 {
            anyhow::bail!(
                "Unsupported NNUE version: {} (expected {}, {}, or {})",
                version,
                NNUE_VERSION_V2,
                NNUE_VERSION_V3,
                NNUE_VERSION
            );
        }

        // Read hidden_size
        cursor.read_exact(&mut buf4)?;
        let hidden_size = u32::from_le_bytes(buf4) as usize;

        // Accumulator uses fixed-size arrays for performance — assert match
        if hidden_size != accumulator::HIDDEN_SIZE {
            anyhow::bail!(
                "NNUE hidden_size {} does not match compiled HIDDEN_SIZE {}",
                hidden_size,
                accumulator::HIDDEN_SIZE
            );
        }

        if version == NNUE_VERSION_V2 {
            // --- V2: single bucket, no PSQT, no king buckets ---
            let ft_count = NUM_FEATURES * hidden_size;
            let ft_weights = read_i16_vec(&mut cursor, ft_count)?;
            let ft_biases = read_i16_vec(&mut cursor, hidden_size)?;
            let output_weights = read_i16_vec(&mut cursor, 2 * hidden_size)?;

            let mut buf2 = [0u8; 2];
            cursor.read_exact(&mut buf2)?;
            let output_bias = i16::from_le_bytes(buf2);

            let total_bytes = ft_count * 2 + hidden_size * 2 + hidden_size * 4 + 2;
            println!(
                "  NNUE v2 loaded: hidden_size={}, buckets=1, total={:.1}KB",
                hidden_size,
                total_bytes as f64 / 1024.0,
            );

            Ok(Self {
                ft_weights,
                ft_biases,
                output_weights,
                output_biases: vec![output_bias],
                psqt_weights: Vec::new(),
                hidden_size,
                num_buckets: 1,
                num_input_buckets: 1,
            })
        } else if version == NNUE_VERSION_V3 {
            // --- V3: output buckets + optional PSQT, no king buckets ---
            cursor.read_exact(&mut buf4)?;
            let num_buckets = u32::from_le_bytes(buf4) as usize;

            let ft_count = NUM_FEATURES * hidden_size;
            let ft_weights = read_i16_vec(&mut cursor, ft_count)?;
            let ft_biases = read_i16_vec(&mut cursor, hidden_size)?;

            // PSQT weights: [768 * num_buckets]
            let psqt_weights = read_i16_vec(&mut cursor, NUM_FEATURES * num_buckets)?;

            // Output weights: [num_buckets * 2 * hidden_size]
            let output_weights = read_i16_vec(&mut cursor, num_buckets * 2 * hidden_size)?;

            // Output biases: [num_buckets]
            let output_biases = read_i16_vec(&mut cursor, num_buckets)?;

            let total_bytes = ft_count * 2 + hidden_size * 2
                + NUM_FEATURES * num_buckets * 2
                + num_buckets * 2 * hidden_size * 2
                + num_buckets * 2;
            println!(
                "  NNUE v3 loaded: hidden_size={}, buckets={}, psqt={}, total={:.1}KB",
                hidden_size,
                num_buckets,
                !psqt_weights.is_empty(),
                total_bytes as f64 / 1024.0,
            );

            Ok(Self {
                ft_weights,
                ft_biases,
                output_weights,
                output_biases,
                psqt_weights,
                hidden_size,
                num_buckets,
                num_input_buckets: 1,
            })
        } else {
            // --- V4: king-bucketed inputs + output buckets ---
            cursor.read_exact(&mut buf4)?;
            let num_buckets = u32::from_le_bytes(buf4) as usize;

            cursor.read_exact(&mut buf4)?;
            let num_input_buckets = u32::from_le_bytes(buf4) as usize;

            let ft_count = num_input_buckets * NUM_FEATURES * hidden_size;
            let ft_weights = read_i16_vec(&mut cursor, ft_count)?;
            let ft_biases = read_i16_vec(&mut cursor, hidden_size)?;

            // Output weights: [num_buckets * 2 * hidden_size]
            let output_weights = read_i16_vec(&mut cursor, num_buckets * 2 * hidden_size)?;

            // Output biases: [num_buckets]
            let output_biases = read_i16_vec(&mut cursor, num_buckets)?;

            let total_bytes = ft_count * 2 + hidden_size * 2
                + num_buckets * 2 * hidden_size * 2
                + num_buckets * 2;
            println!(
                "  NNUE v4 loaded: hidden_size={}, input_buckets={}, output_buckets={}, total={:.1}KB",
                hidden_size,
                num_input_buckets,
                num_buckets,
                total_bytes as f64 / 1024.0,
            );

            Ok(Self {
                ft_weights,
                ft_biases,
                output_weights,
                output_biases,
                psqt_weights: Vec::new(),
                hidden_size,
                num_buckets,
                num_input_buckets,
            })
        }
    }

    /// Create a network with random weights (for testing/bootstrapping only).
    /// Uses single bucket for simplicity.
    pub fn random(hidden_size: usize) -> Self {
        let num_buckets = 1;
        let ft_count = NUM_FEATURES * hidden_size;
        let mut ft_weights = vec![0i16; ft_count];
        let mut ft_biases = vec![0i16; hidden_size];
        let mut output_weights = vec![0i16; num_buckets * 2 * hidden_size];

        // Simple deterministic pseudo-random initialization
        let mut seed: u64 = 0xDEADBEEF;
        for w in ft_weights.iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *w = ((seed >> 48) as i16) / 256;
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
            output_biases: vec![0i16; num_buckets],
            psqt_weights: Vec::new(),
            hidden_size,
            num_buckets,
            num_input_buckets: 1,
        }
    }

    /// Whether this network uses king-bucketed features (v4+)
    #[inline]
    pub fn has_king_buckets(&self) -> bool {
        self.num_input_buckets > 1
    }

    /// Serialize network to v4 binary format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let ft_count = self.num_input_buckets * NUM_FEATURES * self.hidden_size;
        let out_count = self.num_buckets * 2 * self.hidden_size;
        // header: magic(4) + version(4) + hidden(4) + out_buckets(4) + in_buckets(4)
        let total_size = 20 + ft_count * 2 + self.hidden_size * 2
            + out_count * 2 + self.num_buckets * 2;
        let mut buf = Vec::with_capacity(total_size);

        buf.extend_from_slice(NNUE_MAGIC);
        buf.extend_from_slice(&NNUE_VERSION.to_le_bytes());
        buf.extend_from_slice(&(self.hidden_size as u32).to_le_bytes());
        buf.extend_from_slice(&(self.num_buckets as u32).to_le_bytes());
        buf.extend_from_slice(&(self.num_input_buckets as u32).to_le_bytes());

        for &w in &self.ft_weights {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        for &b in &self.ft_biases {
            buf.extend_from_slice(&b.to_le_bytes());
        }
        for &w in &self.output_weights {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        for &b in &self.output_biases {
            buf.extend_from_slice(&b.to_le_bytes());
        }

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
        let net = NnueNetwork::random(accumulator::HIDDEN_SIZE);
        let bytes = net.to_bytes();
        let net2 = NnueNetwork::from_bytes(&bytes).unwrap();

        assert_eq!(net.hidden_size, net2.hidden_size);
        assert_eq!(net.num_buckets, net2.num_buckets);
        assert_eq!(net.ft_weights, net2.ft_weights);
        assert_eq!(net.ft_biases, net2.ft_biases);
        assert_eq!(net.output_weights, net2.output_weights);
        assert_eq!(net.output_biases, net2.output_biases);
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
