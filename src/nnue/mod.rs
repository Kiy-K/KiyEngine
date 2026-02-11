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

/// L1 hidden layer size for deep network (v5+)
pub const L1_SIZE: usize = 16;

/// L1 weight quantization factor (i8 weights: max representable ≈ 127/Q1 = 1.98)
pub const Q1: i32 = 64;

/// Magic bytes for NNUE binary file format
pub const NNUE_MAGIC: &[u8; 4] = b"KYNN";

/// Current NNUE format version (v5: deep network with L1→CReLU→L2)
pub const NNUE_VERSION: u32 = 5;
/// Also accept older formats for backward compatibility
const NNUE_VERSION_V2: u32 = 2;
const NNUE_VERSION_V3: u32 = 3;
const NNUE_VERSION_V4: u32 = 4;

/// Number of output buckets (piece-count based)
pub const NUM_BUCKETS: usize = 8;

/// Default NNUE file path
pub const DEFAULT_NNUE_PATH: &str = "kiyengine.nnue";

// ============================================================================
// NETWORK
// ============================================================================

/// NNUE network weights (quantized, loaded from binary file).
///
/// v5 (deep): FT → SCReLU → L1(shared) → CReLU → L2(per-bucket) → select
/// v4 (king-bucketed): FT → SCReLU → single output layer per bucket
/// v2/v3 (legacy): simpler feature transformer variants
///
/// Output bucket selected by: (total_pieces - 2) / 4
pub struct NnueNetwork {
    /// Feature transformer weights: [num_input_buckets * 768 * hidden_size] (i16, row-major)
    pub ft_weights: Vec<i16>,

    /// Feature transformer biases: [hidden_size] (i16)
    pub ft_biases: Vec<i16>,

    /// Output layer weights per bucket (v2-v4 only): [num_buckets * 2 * hidden_size] (i16)
    pub output_weights: Vec<i16>,

    /// Output layer bias per bucket (v2-v4 only): [num_buckets] (i16)
    pub output_biases: Vec<i16>,

    /// L1 hidden layer weights (v5, shared across buckets): [l1_size * 2 * hidden_size] (i8)
    /// l1_weights[j * 2*hidden_size + i] = weight from input i to neuron j
    pub l1_weights: Vec<i8>,

    /// L1 hidden layer biases (v5): [l1_size] (i32, in QA*Q1 scale)
    pub l1_biases: Vec<i32>,

    /// L2 output weights (v5, per-bucket): [num_buckets * 2 * l1_size] (i16)
    /// After CReLU doubles L1 output: 2*l1_size inputs per bucket
    pub l2_weights: Vec<i16>,

    /// L2 output biases (v5, per-bucket): [num_buckets] (i16, in QA*Q2 scale)
    pub l2_biases: Vec<i16>,

    /// PSQT weights (legacy v3 only, empty otherwise)
    pub psqt_weights: Vec<i16>,

    /// Hidden layer size (typically 256 or 512)
    pub hidden_size: usize,

    /// L1 hidden layer size (0 = shallow/v2-v4, 16 = deep/v5)
    pub l1_size: usize,

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
    /// Dispatches to deep (v5: L1→CReLU→L2) or shallow (v2-v4: SCReLU→output) path.
    /// Returns evaluation in centipawns from side-to-move's perspective.
    #[inline(always)]
    pub fn evaluate(&self, acc: &Accumulator, stm: Color, board: &Board) -> i32 {
        debug_assert!(acc.computed, "Accumulator not computed");

        let bucket = self.bucket_index(board);
        let (us, them) = match stm {
            Color::White => (&acc.white, &acc.black),
            Color::Black => (&acc.black, &acc.white),
        };

        if self.has_deep_layers() {
            return self.evaluate_deep_scalar(us, them, bucket);
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { self.evaluate_avx2(us, them, bucket) };
            }
        }

        self.evaluate_scalar(us, them, bucket)
    }

    /// Deep network evaluate (v5): FT → SCReLU(v²/QA) → L1 → CReLU → L2 → select(bucket)
    ///
    /// Quantization flow:
    /// - SCReLU: v = clamp(acc, 0, QA), output = v*v/QA → [0, QA=255]
    /// - L1 (i8 weights, shared): accumulate in i32, biases in QA*Q1 scale
    /// - CReLU (rescaled): clamp(±l1/Q1, 0, QA) → [0, 255], doubles dimension
    /// - L2 (i16 weights, per-bucket): accumulate in i32, biases in QA*QB scale
    /// - Dequant: output * SCALE / (QA * QB)
    #[inline]
    fn evaluate_deep_scalar(&self, us: &[i16], them: &[i16], bucket: usize) -> i32 {
        let qa = QA as i32;
        let hs = self.hidden_size;
        let l1s = self.l1_size;
        let in_size = 2 * hs;

        // Step 1: SCReLU — clamp(x, 0, QA)² / QA → [0, QA]
        // This matches Bullet's screlu: clamp(x,0,1)² quantized in QA scale
        let mut l1_out = [0i32; 64]; // max l1_size (padded)

        for j in 0..l1s {
            let mut sum = self.l1_biases[j];
            let w_base = j * in_size;

            // "us" perspective
            for i in 0..hs {
                let v = (us[i] as i32).clamp(0, qa);
                let s = v * v / qa; // SCReLU: v²/QA → [0, QA]
                sum += s * self.l1_weights[w_base + i] as i32;
            }

            // "them" perspective
            for i in 0..hs {
                let v = (them[i] as i32).clamp(0, qa);
                let s = v * v / qa;
                sum += s * self.l1_weights[w_base + hs + i] as i32;
            }

            l1_out[j] = sum;
        }

        // Step 2: CReLU (rescaled) — doubles L1 output dimension
        // positive[j] = clamp(l1/Q1, 0, QA), negative[j] = clamp(-l1/Q1, 0, QA)
        let mut crelu = [0i32; 128]; // max 2 * l1_size
        for j in 0..l1s {
            crelu[j] = (l1_out[j] / Q1).clamp(0, qa);
            crelu[l1s + j] = (-l1_out[j] / Q1).clamp(0, qa);
        }

        // Step 3: L2 — per-bucket output
        let l2_in_size = 2 * l1s;
        let l2_w_base = bucket * l2_in_size;
        let mut output = self.l2_biases[bucket] as i64;
        for j in 0..l2_in_size {
            output += crelu[j] as i64 * self.l2_weights[l2_w_base + j] as i64;
        }

        // Step 4: Dequant — output is in QA*QB scale
        let result = output * SCALE as i64 / (qa as i64 * QB as i64);
        result as i32
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
        if version != NNUE_VERSION && version != NNUE_VERSION_V4 && version != NNUE_VERSION_V3 && version != NNUE_VERSION_V2 {
            anyhow::bail!(
                "Unsupported NNUE version: {} (expected 2, 3, 4, or 5)",
                version,
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
                l1_weights: Vec::new(),
                l1_biases: Vec::new(),
                l2_weights: Vec::new(),
                l2_biases: Vec::new(),
                psqt_weights: Vec::new(),
                hidden_size,
                l1_size: 0,
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
                l1_weights: Vec::new(),
                l1_biases: Vec::new(),
                l2_weights: Vec::new(),
                l2_biases: Vec::new(),
                psqt_weights,
                hidden_size,
                l1_size: 0,
                num_buckets,
                num_input_buckets: 1,
            })
        } else if version == NNUE_VERSION_V4 {
            // --- V4: king-bucketed inputs + output buckets (shallow) ---
            cursor.read_exact(&mut buf4)?;
            let num_buckets = u32::from_le_bytes(buf4) as usize;

            cursor.read_exact(&mut buf4)?;
            let num_input_buckets = u32::from_le_bytes(buf4) as usize;

            let ft_count = num_input_buckets * NUM_FEATURES * hidden_size;
            let ft_weights = read_i16_vec(&mut cursor, ft_count)?;
            let ft_biases = read_i16_vec(&mut cursor, hidden_size)?;

            let output_weights = read_i16_vec(&mut cursor, num_buckets * 2 * hidden_size)?;
            let output_biases = read_i16_vec(&mut cursor, num_buckets)?;

            let total_bytes = ft_count * 2 + hidden_size * 2
                + num_buckets * 2 * hidden_size * 2
                + num_buckets * 2;
            println!(
                "  NNUE v4 loaded: hidden_size={}, input_buckets={}, output_buckets={}, total={:.1}KB",
                hidden_size, num_input_buckets, num_buckets, total_bytes as f64 / 1024.0,
            );

            Ok(Self {
                ft_weights,
                ft_biases,
                output_weights,
                output_biases,
                l1_weights: Vec::new(),
                l1_biases: Vec::new(),
                l2_weights: Vec::new(),
                l2_biases: Vec::new(),
                psqt_weights: Vec::new(),
                hidden_size,
                l1_size: 0,
                num_buckets,
                num_input_buckets,
            })
        } else {
            // --- V5: deep network (FT → SCReLU → L1 → CReLU → L2) ---
            cursor.read_exact(&mut buf4)?;
            let num_buckets = u32::from_le_bytes(buf4) as usize;

            cursor.read_exact(&mut buf4)?;
            let num_input_buckets = u32::from_le_bytes(buf4) as usize;

            cursor.read_exact(&mut buf4)?;
            let l1_size = u32::from_le_bytes(buf4) as usize;

            // Feature transformer
            let ft_count = num_input_buckets * NUM_FEATURES * hidden_size;
            let ft_weights = read_i16_vec(&mut cursor, ft_count)?;
            let ft_biases = read_i16_vec(&mut cursor, hidden_size)?;

            // L1: shared across buckets, i8 weights
            let l1_w_count = l1_size * 2 * hidden_size;
            let l1_weights = read_i8_vec(&mut cursor, l1_w_count)?;
            let l1_biases = read_i32_vec(&mut cursor, l1_size)?;

            // L2: per-bucket, i16 weights (CReLU doubles L1 output)
            let l2_w_count = num_buckets * 2 * l1_size;
            let l2_weights = read_i16_vec(&mut cursor, l2_w_count)?;
            let l2_biases = read_i16_vec(&mut cursor, num_buckets)?;

            let total_bytes = ft_count * 2 + hidden_size * 2
                + l1_w_count + l1_size * 4
                + l2_w_count * 2 + num_buckets * 2;
            println!(
                "  NNUE v5 loaded: hidden_size={}, l1_size={}, input_buckets={}, output_buckets={}, total={:.1}KB",
                hidden_size, l1_size, num_input_buckets, num_buckets, total_bytes as f64 / 1024.0,
            );

            Ok(Self {
                ft_weights,
                ft_biases,
                output_weights: Vec::new(),
                output_biases: Vec::new(),
                l1_weights,
                l1_biases,
                l2_weights,
                l2_biases,
                psqt_weights: Vec::new(),
                hidden_size,
                l1_size,
                num_buckets,
                num_input_buckets,
            })
        }
    }

    /// Create a shallow network with random weights (for testing/bootstrapping only).
    /// Uses single bucket, no deep layers.
    pub fn random(hidden_size: usize) -> Self {
        let num_buckets = 1;
        let ft_count = NUM_FEATURES * hidden_size;
        let mut ft_weights = vec![0i16; ft_count];
        let mut ft_biases = vec![0i16; hidden_size];
        let mut output_weights = vec![0i16; num_buckets * 2 * hidden_size];

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
            l1_weights: Vec::new(),
            l1_biases: Vec::new(),
            l2_weights: Vec::new(),
            l2_biases: Vec::new(),
            psqt_weights: Vec::new(),
            hidden_size,
            l1_size: 0,
            num_buckets,
            num_input_buckets: 1,
        }
    }

    /// Create a deep network with random weights (for testing v5 architecture).
    pub fn random_deep(hidden_size: usize, l1_size: usize, num_buckets: usize) -> Self {
        let ft_count = NUM_FEATURES * hidden_size;
        let mut ft_weights = vec![0i16; ft_count];
        let mut ft_biases = vec![0i16; hidden_size];
        let mut l1_weights = vec![0i8; l1_size * 2 * hidden_size];
        let l1_biases = vec![0i32; l1_size];
        let mut l2_weights = vec![0i16; num_buckets * 2 * l1_size];
        let l2_biases = vec![0i16; num_buckets];

        let mut seed: u64 = 0xCAFEBABE;
        for w in ft_weights.iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *w = ((seed >> 48) as i16) / 256;
        }
        for b in ft_biases.iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = ((seed >> 48) as i16) / 256;
        }
        for w in l1_weights.iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *w = ((seed >> 48) as i8) / 4;
        }
        for w in l2_weights.iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *w = ((seed >> 48) as i16) / 256;
        }

        Self {
            ft_weights,
            ft_biases,
            output_weights: Vec::new(),
            output_biases: Vec::new(),
            l1_weights,
            l1_biases,
            l2_weights,
            l2_biases,
            psqt_weights: Vec::new(),
            hidden_size,
            l1_size,
            num_buckets,
            num_input_buckets: 1,
        }
    }

    /// Whether this network uses king-bucketed features (v4+)
    #[inline]
    pub fn has_king_buckets(&self) -> bool {
        self.num_input_buckets > 1
    }

    /// Whether this network has deep post-FT layers (v5+)
    #[inline]
    pub fn has_deep_layers(&self) -> bool {
        self.l1_size > 0
    }

    /// Serialize network to binary format (v5 if deep, v4 if shallow).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        buf.extend_from_slice(NNUE_MAGIC);

        if self.has_deep_layers() {
            // v5: deep network
            buf.extend_from_slice(&NNUE_VERSION.to_le_bytes());
            buf.extend_from_slice(&(self.hidden_size as u32).to_le_bytes());
            buf.extend_from_slice(&(self.num_buckets as u32).to_le_bytes());
            buf.extend_from_slice(&(self.num_input_buckets as u32).to_le_bytes());
            buf.extend_from_slice(&(self.l1_size as u32).to_le_bytes());

            for &w in &self.ft_weights { buf.extend_from_slice(&w.to_le_bytes()); }
            for &b in &self.ft_biases { buf.extend_from_slice(&b.to_le_bytes()); }
            for &w in &self.l1_weights { buf.push(w as u8); }
            for &b in &self.l1_biases { buf.extend_from_slice(&b.to_le_bytes()); }
            for &w in &self.l2_weights { buf.extend_from_slice(&w.to_le_bytes()); }
            for &b in &self.l2_biases { buf.extend_from_slice(&b.to_le_bytes()); }
        } else {
            // v4: shallow network
            buf.extend_from_slice(&NNUE_VERSION_V4.to_le_bytes());
            buf.extend_from_slice(&(self.hidden_size as u32).to_le_bytes());
            buf.extend_from_slice(&(self.num_buckets as u32).to_le_bytes());
            buf.extend_from_slice(&(self.num_input_buckets as u32).to_le_bytes());

            for &w in &self.ft_weights { buf.extend_from_slice(&w.to_le_bytes()); }
            for &b in &self.ft_biases { buf.extend_from_slice(&b.to_le_bytes()); }
            for &w in &self.output_weights { buf.extend_from_slice(&w.to_le_bytes()); }
            for &b in &self.output_biases { buf.extend_from_slice(&b.to_le_bytes()); }
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

/// Read a vector of i8 values from a reader.
fn read_i8_vec<R: Read>(reader: &mut R, count: usize) -> anyhow::Result<Vec<i8>> {
    let mut bytes = vec![0u8; count];
    reader.read_exact(&mut bytes)?;
    Ok(bytes.into_iter().map(|b| b as i8).collect())
}

/// Read a vector of i32 values from a reader (little-endian).
fn read_i32_vec<R: Read>(reader: &mut R, count: usize) -> anyhow::Result<Vec<i32>> {
    let mut bytes = vec![0u8; count * 4];
    reader.read_exact(&mut bytes)?;
    let values: Vec<i32> = bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
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
    fn test_serialize_deserialize_deep() {
        let net = NnueNetwork::random_deep(accumulator::HIDDEN_SIZE, L1_SIZE, 8);
        assert!(net.has_deep_layers());
        let bytes = net.to_bytes();
        let net2 = NnueNetwork::from_bytes(&bytes).unwrap();

        assert_eq!(net.hidden_size, net2.hidden_size);
        assert_eq!(net.l1_size, net2.l1_size);
        assert_eq!(net.num_buckets, net2.num_buckets);
        assert_eq!(net.ft_weights, net2.ft_weights);
        assert_eq!(net.ft_biases, net2.ft_biases);
        assert_eq!(net.l1_weights, net2.l1_weights);
        assert_eq!(net.l1_biases, net2.l1_biases);
        assert_eq!(net.l2_weights, net2.l2_weights);
        assert_eq!(net.l2_biases, net2.l2_biases);
    }

    #[test]
    fn test_deep_network_evaluation() {
        let net = NnueNetwork::random_deep(accumulator::HIDDEN_SIZE, L1_SIZE, 8);
        let board = Board::default();
        let eval = net.evaluate_board(&board);
        assert!(eval.abs() < 100_000, "Deep eval out of range: {}", eval);
    }

    #[test]
    fn test_evaluation_symmetry() {
        let net = NnueNetwork::random(256);
        let board = Board::default();
        let eval = net.evaluate_board(&board);
        assert!(eval.abs() < 100_000);
    }
}
