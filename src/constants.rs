//! KiyEngine Core Constants
//!
//! Centralized constants for the chess engine to avoid magic numbers
//! and ensure consistency across modules.

/// Engine identification
pub const ENGINE_NAME: &str = "KiyEngine";
pub const ENGINE_VERSION: &str = "5.0.0";
pub const ENGINE_AUTHOR: &str = "Khoi";

/// Default model file path for V5 (GGUF format)
pub const DEFAULT_MODEL_PATH: &str = "kiyengine.gguf";
/// Legacy v4 model path (safetensors)
pub const LEGACY_MODEL_PATH: &str = "kiyengine.safetensors";

/// Search constants
pub const MATE_SCORE: i32 = 30000;
pub const INFINITY: i32 = 32000;
pub const MAX_DEPTH: usize = 64;
pub const MAX_PLY: usize = 128;

/// Default transposition table size in MB
pub const DEFAULT_TT_SIZE_MB: usize = 64;
/// Maximum TT size in MB
pub const MAX_TT_SIZE_MB: usize = 65536;

/// Default thread count
pub const DEFAULT_THREADS: usize = 1;
/// Maximum thread count
pub const MAX_THREADS: usize = 256;

/// Default move overhead in milliseconds
pub const DEFAULT_MOVE_OVERHEAD_MS: u32 = 30;
/// Maximum move overhead in milliseconds
pub const MAX_MOVE_OVERHEAD_MS: u32 = 5000;

/// Neural network architecture constants
pub const VOCAB_SIZE: usize = 4608;
pub const CONTEXT_LENGTH: usize = 32;
pub const D_MODEL: usize = 1024;
pub const NUM_LAYERS: usize = 4;

/// Evaluation constants
pub const PAWN_VALUE: i32 = 100;
pub const KNIGHT_VALUE: i32 = 320;
pub const BISHOP_VALUE: i32 = 330;
pub const ROOK_VALUE: i32 = 500;
pub const QUEEN_VALUE: i32 = 900;
pub const KING_VALUE: i32 = 20000;

/// Tempo bonus (centipawns)
pub const TEMPO_BONUS: i32 = 10;

/// Node check interval for stop flag polling
pub const NODE_CHECK_INTERVAL: u64 = 2047;
