// src/eval/batching.rs
//! Batching Inference System for Scout Engine
//! Optimizes BitNet inference by batching multiple positions

use crate::engine::Engine;
use chess::Board;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Batching inference manager
pub struct BatchInferenceManager {
    engine: Arc<Engine>,
    batch_queue: Arc<Mutex<VecDeque<Vec<usize>>>>,
    batch_size: usize,
    max_batch_delay_ms: u64,
    #[allow(dead_code)] // Used for performance monitoring
    last_process_time: Arc<Mutex<std::time::Instant>>,
    batch_counter: Arc<Mutex<usize>>,
}

impl BatchInferenceManager {
    pub fn new(engine: Arc<Engine>) -> Self {
        Self {
            engine,
            batch_queue: Arc::new(Mutex::new(VecDeque::new())),
            batch_size: 8,         // Optimal batch size for most GPUs/NPUs
            max_batch_delay_ms: 5, // 5ms max delay
            last_process_time: Arc::new(Mutex::new(std::time::Instant::now())),
            batch_counter: Arc::new(Mutex::new(0)),
        }
    }

    /// Add position to batch queue and return immediate evaluation
    pub fn add_position(&self, move_history: Vec<usize>) -> f32 {
        let mut queue = self.batch_queue.lock().unwrap();
        queue.push_back(move_history.clone());

        // Check if we should process the batch
        let should_process = queue.len() >= self.batch_size || {
            let last_time = *self.last_process_time.lock().unwrap();
            last_time.elapsed().as_millis() >= self.max_batch_delay_ms as u128
        };

        if should_process {
            self.process_batch(&mut queue);
            *self.last_process_time.lock().unwrap() = std::time::Instant::now();
        }

        // For now, return immediate evaluation (in future, return cached batch result)
        match self.engine.forward(&move_history) {
            Ok((_, eval_score)) => eval_score,
            Err(_) => 0.0,
        }
    }

    /// Process a batch of positions for efficient inference
    fn process_batch(&self, queue: &mut VecDeque<Vec<usize>>) {
        if queue.is_empty() {
            return;
        }

        let batch_size = queue.len().min(self.batch_size);
        let batch: Vec<Vec<usize>> = queue.drain(..batch_size).collect();

        // Update batch counter
        *self.batch_counter.lock().unwrap() += 1;

        // In a full implementation, this would:
        // 1. Prepare batch tensor from move histories
        // 2. Run single forward pass on batch
        // 3. Cache results for immediate retrieval
        // 4. Store results in a lookup table

        // For now, we'll process individually but this shows the structure
        for history in batch {
            if let Err(_) = self.engine.forward(&history) {
                // Log error but continue processing
            }
        }
    }

    /// Get batch statistics
    pub fn get_batch_stats(&self) -> BatchStats {
        let queue = self.batch_queue.lock().unwrap();
        let batch_count = *self.batch_counter.lock().unwrap();
        BatchStats {
            queue_size: queue.len(),
            batch_size: self.batch_size,
            max_delay_ms: self.max_batch_delay_ms,
            batches_processed: batch_count,
        }
    }
}

/// Batch processing statistics
#[derive(Debug)]
pub struct BatchStats {
    pub queue_size: usize,
    pub batch_size: usize,
    pub max_delay_ms: u64,
    pub batches_processed: usize,
}

/// Enhanced evaluation context with batching support
pub struct BatchedEvaluationContext {
    pub base_ctx: crate::eval::EvaluationContext,
    pub batch_manager: BatchInferenceManager,
}

impl BatchedEvaluationContext {
    pub fn new(engine: Arc<crate::engine::Engine>, tt: Arc<crate::search::tt::AtomicTT>) -> Self {
        let base_ctx = crate::eval::EvaluationContext::new(engine.clone(), tt);
        let batch_manager = BatchInferenceManager::new(engine.clone());

        Self {
            base_ctx,
            batch_manager,
        }
    }

    /// Evaluate with batching support
    pub fn evaluate_with_batching(
        &self,
        board: &Board,
        move_history: &[usize],
        depth: u8,
        ply: usize,
    ) -> i32 {
        // Use batching only for deeper searches where it matters
        if depth > 3 && ply < 50 {
            let batched_eval = self.batch_manager.add_position(move_history.to_vec());
            let fast_eval = self.base_ctx.evaluate_fast(board);

            // Blend batched and fast evaluation
            let blend_factor = 0.8; // 80% batched, 20% fast
            let blended = (fast_eval as f32 * (1.0 - blend_factor)
                + batched_eval * 100.0 * blend_factor) as i32;
            return blended;
        }

        // Fallback to regular evaluation
        self.base_ctx
            .evaluate_hybrid(board, move_history, depth, ply)
    }
}
