// src/eval/hybrid.rs
//! Hybrid Evaluation System
//! Combines fast material/PST evaluation with deep BitNet inference

use crate::engine::Engine;
use crate::search::tt::AtomicTT;
use chess::{Board, Color, Piece, Square};
use std::sync::Arc;

/// Evaluation levels for the hybrid system
#[derive(Debug, Clone, Copy)]
pub enum EvalLevel {
    Fast, // Material + PST only
    Deep, // BitNet Value Head
}

/// Hybrid evaluation context
pub struct EvaluationContext {
    pub engine: Arc<Engine>,
    pub tt: Arc<AtomicTT>,
    pub bitnet_depth_threshold: u8, // Depth at which to use BitNet
    pub fast_eval_cache: [i32; 64 * 12 * 2], // Piece-square table cache
}

impl EvaluationContext {
    pub fn new(engine: Arc<Engine>, tt: Arc<AtomicTT>) -> Self {
        let mut ctx = Self {
            engine,
            tt,
            bitnet_depth_threshold: 4, // Increased to reduce BitNet usage
            fast_eval_cache: [0; 64 * 12 * 2],
        };
        ctx.init_piece_square_tables();
        ctx
    }

    /// Initialize piece-square tables with standard chess values
    fn init_piece_square_tables(&mut self) {
        // Pawn PST (simplified)
        let pawn_pst = [
            0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 50, 50, 50, 50, 50, 50, 10, 10, 20, 30, 30, 20, 10, 10,
            5, 5, 10, 25, 25, 10, 5, 5, 0, 0, 0, 20, 20, 0, 0, 0, 5, -5, -10, 0, 0, -10, -5, 5, 5,
            10, 10, -20, -20, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0,
        ];

        // Knight PST (simplified)
        let knight_pst = [
            -50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 0, 0, 0, -20, -40, -30, 0, 10, 15,
            15, 10, 0, -30, -30, 5, 15, 20, 20, 15, 5, -30, -30, 0, 15, 20, 20, 15, 0, -30, -30, 5,
            10, 15, 15, 10, 5, -30, -40, -20, 0, 5, 5, 0, -20, -40, -50, -40, -30, -30, -30, -30,
            -40, -50,
        ];

        // Bishop PST (simplified)
        let bishop_pst = [
            -20, -10, -10, -10, -10, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 10, 10,
            5, 0, -10, -10, 5, 5, 10, 10, 5, 5, -10, -10, 0, 10, 10, 10, 10, 0, -10, -10, 10, 5, 0,
            0, 5, 10, -10, -10, 5, 0, 0, 0, 0, 5, -10, -20, -10, -10, -10, -10, -10, -10, -20,
        ];

        // Rook PST (simplified)
        let rook_pst = [
            0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, 10, 10, 10, 10, 5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0,
            0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0,
            0, 0, -5, 0, 0, 0, 5, 5, 0, 0, 0,
        ];

        // Queen PST (simplified)
        let queen_pst = [
            -20, -10, -10, -5, -5, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 5, 5, 5,
            0, -10, -5, 0, 5, 5, 5, 5, 0, -5, 0, 0, 5, 5, 5, 5, 0, -5, -10, 5, 5, 5, 5, 5, 0, -10,
            -10, 0, 5, 0, 0, 0, 0, -10, -20, -10, -10, -5, -5, -10, -10, -20,
        ];

        // King PST (simplified, middle game)
        let king_pst = [
            -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30,
            -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -20, -30,
            -30, -40, -40, -30, -30, -20, -10, -20, -20, -20, -20, -20, -20, -10, 20, 20, 0, 0, 0,
            0, 20, 20, 20, 30, 10, 0, 0, 10, 30, 20,
        ];

        // Store PSTs in cache
        for (i, &value) in pawn_pst.iter().enumerate() {
            self.fast_eval_cache
                [Self::pst_index(Piece::Pawn, unsafe { Square::new(i as u8) }, Color::White)] =
                value;
            self.fast_eval_cache[Self::pst_index(
                Piece::Pawn,
                unsafe { Square::new(63 - i as u8) },
                Color::Black,
            )] = value;
        }

        for (i, &value) in knight_pst.iter().enumerate() {
            self.fast_eval_cache
                [Self::pst_index(Piece::Knight, unsafe { Square::new(i as u8) }, Color::White)] =
                value;
            self.fast_eval_cache[Self::pst_index(
                Piece::Knight,
                unsafe { Square::new(63 - i as u8) },
                Color::Black,
            )] = value;
        }

        for (i, &value) in bishop_pst.iter().enumerate() {
            self.fast_eval_cache
                [Self::pst_index(Piece::Bishop, unsafe { Square::new(i as u8) }, Color::White)] =
                value;
            self.fast_eval_cache[Self::pst_index(
                Piece::Bishop,
                unsafe { Square::new(63 - i as u8) },
                Color::Black,
            )] = value;
        }

        for (i, &value) in rook_pst.iter().enumerate() {
            self.fast_eval_cache
                [Self::pst_index(Piece::Rook, unsafe { Square::new(i as u8) }, Color::White)] =
                value;
            self.fast_eval_cache[Self::pst_index(
                Piece::Rook,
                unsafe { Square::new(63 - i as u8) },
                Color::Black,
            )] = value;
        }

        for (i, &value) in queen_pst.iter().enumerate() {
            self.fast_eval_cache
                [Self::pst_index(Piece::Queen, unsafe { Square::new(i as u8) }, Color::White)] =
                value;
            self.fast_eval_cache[Self::pst_index(
                Piece::Queen,
                unsafe { Square::new(63 - i as u8) },
                Color::Black,
            )] = value;
        }

        for (i, &value) in king_pst.iter().enumerate() {
            self.fast_eval_cache
                [Self::pst_index(Piece::King, unsafe { Square::new(i as u8) }, Color::White)] =
                value;
            self.fast_eval_cache[Self::pst_index(
                Piece::King,
                unsafe { Square::new(63 - i as u8) },
                Color::Black,
            )] = value;
        }
    }

    #[inline]
    fn pst_index(piece: Piece, square: Square, color: Color) -> usize {
        let piece_idx = match piece {
            Piece::Pawn => 0,
            Piece::Knight => 1,
            Piece::Bishop => 2,
            Piece::Rook => 3,
            Piece::Queen => 4,
            Piece::King => 5,
        };
        let color_idx = if color == Color::White { 0 } else { 1 };
        (piece_idx * 64 * 2) + (color_idx * 64) + square.to_index() as usize
    }

    /// Fast evaluation using material count and piece-square tables
    pub fn evaluate_fast(&self, board: &Board) -> i32 {
        let mut score = 0;

        // Material values
        let material_values = [
            (Piece::Pawn, 100),
            (Piece::Knight, 320),
            (Piece::Bishop, 330),
            (Piece::Rook, 500),
            (Piece::Queen, 900),
            (Piece::King, 20000),
        ];

        for color in [Color::White, Color::Black] {
            let color_multiplier = if color == Color::White { 1 } else { -1 };

            for (piece, value) in &material_values {
                let bitboard = board.pieces(*piece) & board.color_combined(color);
                for square in bitboard {
                    let pst_value = self.fast_eval_cache[Self::pst_index(*piece, square, color)];
                    score += color_multiplier * (value + pst_value);
                }
            }
        }

        // Tempo bonus
        score += if board.side_to_move() == Color::White {
            10
        } else {
            -10
        };

        score
    }

    /// Deep evaluation using BitNet Value Head
    pub fn evaluate_deep(&self, _board: &Board, move_history: &[usize]) -> Option<i32> {
        // Run BitNet inference directly (AtomicTT doesn't cache static eval)
        match self.engine.forward(move_history) {
            Ok((_, eval_score)) => {
                // SCALE_FACTOR: Convert tanh output [-1, 1] to centipawns
                // tanh(±1) ≈ ±0.76, so we scale to get reasonable centipawn values
                // Formula: final_score = tanh_output * SCALE_FACTOR * 100
                // Using SCALE_FACTOR = 4.0 to map ~±0.7 to ~±280 cp (realistic range)
                const SCALE_FACTOR: f32 = 4.0;
                let scaled_eval = (eval_score * SCALE_FACTOR * 100.0) as i32;
                // Clamp to reasonable range to prevent inflation
                let clamped_eval = scaled_eval.clamp(-1000, 1000);
                Some(clamped_eval)
            }
            Err(_) => None,
        }
    }

    /// Hybrid evaluation based on depth and position
    pub fn evaluate_hybrid(
        &self,
        board: &Board,
        move_history: &[usize],
        depth: u8,
        ply: usize,
    ) -> i32 {
        // Always use fast evaluation for quiescence search
        if ply >= 100 {
            // Quiescence search indicator
            return self.evaluate_fast(board);
        }

        // Use BitNet only at deeper levels to maintain NPS
        if depth > self.bitnet_depth_threshold {
            if let Some(deep_eval) = self.evaluate_deep(board, move_history) {
                // Blend fast and deep evaluation for stability
                let fast_eval = self.evaluate_fast(board);
                const BLEND_FACTOR: f32 = 0.7; // 70% BitNet, 30% traditional
                let blended = (fast_eval as f32 * (1.0 - BLEND_FACTOR)
                    + deep_eval as f32 * BLEND_FACTOR) as i32;
                return blended;
            }
        }

        // Fallback to fast evaluation
        self.evaluate_fast(board)
    }
}
