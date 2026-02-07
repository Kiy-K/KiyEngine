// src/search/mcts_batch.rs
//! Batched inference for MCTS - Optimizes BitNet throughput
//!
//! Collects multiple leaf nodes and processes them in a single batch
//! to maximize GPU/NPU utilization during MCTS expansion.

use crate::engine::Engine;
use crate::search::mcts::Node;
use crate::uci::MoveCodec;
use candle_core::Result;
use chess::{Board, ChessMove, MoveGen};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Result of neural network evaluation
pub struct InferenceResult {
    pub node: Arc<Node>,
    pub policy: Vec<(ChessMove, f32)>,
    pub value: f32,
}

/// Batched inference engine for MCTS
pub struct MCTSBatchedInference {
    /// Neural network engine
    engine: Arc<Engine>,
    /// Pending inference requests
    queue: Arc<Mutex<VecDeque<PendingRequest>>>,
    /// Batch size (number of positions to process together)
    batch_size: usize,
    /// Maximum wait time before processing partial batch
    max_wait: Duration,
    /// Statistics
    stats: Arc<InferenceStats>,
}

/// A pending inference request with completion tracking
struct PendingRequest {
    pub node: Arc<Node>,
    pub move_history: Vec<usize>,
    pub board: Board,
    pub submitted_at: Instant,
}

/// Inference statistics for monitoring
#[derive(Debug, Default)]
pub struct InferenceStats {
    pub total_batches: AtomicUsize,
    pub total_positions: AtomicUsize,
    pub total_collisions: AtomicUsize,
    pub avg_batch_size: Mutex<f32>,
}

impl MCTSBatchedInference {
    /// Create a new batched inference manager
    pub fn new(engine: Arc<Engine>, batch_size: usize, max_wait_ms: u64) -> Self {
        Self {
            engine,
            queue: Arc::new(Mutex::new(VecDeque::new())),
            batch_size,
            max_wait: Duration::from_millis(max_wait_ms),
            stats: Arc::new(InferenceStats::default()),
        }
    }

    /// Queue a position for batched evaluation
    /// Returns the number of positions in queue after adding
    pub fn queue_evaluation(&self, node: Arc<Node>, move_history: Vec<usize>) -> usize {
        let board = node.board.clone();
        let request = PendingRequest {
            node,
            move_history,
            board,
            submitted_at: Instant::now(),
        };

        let mut queue = self.queue.lock();
        queue.push_back(request);
        queue.len()
    }

    /// Check if batch is ready to process
    pub fn should_process(&self) -> bool {
        let queue = self.queue.lock();
        if queue.len() >= self.batch_size {
            return true;
        }

        // Check if oldest request has exceeded max wait
        if let Some(oldest) = queue.front() {
            if oldest.submitted_at.elapsed() >= self.max_wait {
                return true;
            }
        }

        false
    }

    /// Get current queue size
    pub fn queue_size(&self) -> usize {
        self.queue.lock().len()
    }

    /// Process a batch of pending requests
    /// Returns the results of the evaluations
    pub fn process_batch(&self) -> Vec<InferenceResult> {
        let mut queue = self.queue.lock();

        if queue.is_empty() {
            return Vec::new();
        }

        // Collect batch
        let batch_size = queue.len().min(self.batch_size);
        let batch: Vec<PendingRequest> = queue.drain(..batch_size).collect();
        drop(queue); // Release lock during processing

        // Update stats
        self.stats.total_batches.fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_positions
            .fetch_add(batch.len(), Ordering::Relaxed);

        // Process batch
        let results = self.evaluate_batch(&batch);

        // Update average batch size
        let avg = *self.stats.avg_batch_size.lock();
        let count = self.stats.total_batches.load(Ordering::Relaxed);
        let new_avg = (avg * (count - 1) as f32 + batch.len() as f32) / count as f32;
        *self.stats.avg_batch_size.lock() = new_avg;

        results
    }

    /// Evaluate a batch of positions using the neural network
    fn evaluate_batch(&self, batch: &[PendingRequest]) -> Vec<InferenceResult> {
        let mut results = Vec::with_capacity(batch.len());

        // For each position in the batch
        for request in batch {
            // Run individual inference (can be optimized to true batching)
            match self.evaluate_single(&request.move_history, &request.board) {
                Ok((policy, value)) => {
                    results.push(InferenceResult {
                        node: request.node.clone(),
                        policy,
                        value,
                    });
                }
                Err(_) => {
                    // Fallback: uniform policy, neutral value
                    let legal_moves: Vec<ChessMove> = MoveGen::new_legal(&request.board).collect();
                    let n = legal_moves.len();
                    let uniform_prob = if n > 0 { 1.0 / n as f32 } else { 0.0 };

                    let policy: Vec<(ChessMove, f32)> = legal_moves
                        .into_iter()
                        .map(|mv| (mv, uniform_prob))
                        .collect();

                    results.push(InferenceResult {
                        node: request.node.clone(),
                        policy,
                        value: 0.0,
                    });
                }
            }
        }

        results
    }

    /// Evaluate a single position
    fn evaluate_single(
        &self,
        move_history: &[usize],
        board: &Board,
    ) -> Result<(Vec<(ChessMove, f32)>, f32)> {
        // Run neural network forward pass
        let (policy_logits, value) = self.engine.forward(move_history)?;

        // Extract policy for legal moves
        let legal_moves: Vec<ChessMove> = MoveGen::new_legal(board).collect();

        if legal_moves.is_empty() {
            return Ok((Vec::new(), value));
        }

        // Get logits as vector
        let logits = policy_logits.to_vec1::<f32>()?;

        // Compute softmax for legal moves only
        let mut move_logits: Vec<(ChessMove, f32)> = legal_moves
            .into_iter()
            .filter_map(|mv| {
                let token = MoveCodec::move_to_token(&mv) as usize;
                if token < logits.len() {
                    Some((mv, logits[token]))
                } else {
                    None
                }
            })
            .collect();

        // Find max for numerical stability
        let max_logit = move_logits
            .iter()
            .map(|(_, l)| *l)
            .fold(f32::NEG_INFINITY, f32::max);

        // Compute exp and sum
        let mut exp_sum = 0.0f32;
        for (_, logit) in &mut move_logits {
            let exp_val = (*logit - max_logit).exp();
            *logit = exp_val;
            exp_sum += exp_val;
        }

        // Normalize to get probabilities
        if exp_sum > 0.0 {
            for (_, prob) in &mut move_logits {
                *prob /= exp_sum;
            }
        }

        Ok((move_logits, value))
    }

    /// Get current inference statistics
    pub fn get_stats(&self) -> InferenceStatsSnapshot {
        InferenceStatsSnapshot {
            total_batches: self.stats.total_batches.load(Ordering::Relaxed),
            total_positions: self.stats.total_positions.load(Ordering::Relaxed),
            total_collisions: self.stats.total_collisions.load(Ordering::Relaxed),
            avg_batch_size: *self.stats.avg_batch_size.lock(),
            queue_size: self.queue_size(),
        }
    }

    /// Clear the queue (e.g., when search is stopped)
    pub fn clear(&self) {
        let mut queue = self.queue.lock();
        queue.clear();
    }
}

/// Snapshot of inference statistics
#[derive(Debug, Clone)]
pub struct InferenceStatsSnapshot {
    pub total_batches: usize,
    pub total_positions: usize,
    pub total_collisions: usize,
    pub avg_batch_size: f32,
    pub queue_size: usize,
}

/// Thread-safe batch processor for parallel MCTS
pub struct ParallelBatchProcessor {
    inference: Arc<MCTSBatchedInference>,
    workers: Vec<std::thread::JoinHandle<()>>,
    stop_flag: Arc<std::sync::atomic::AtomicBool>,
}

impl ParallelBatchProcessor {
    /// Create parallel batch processor with worker threads
    pub fn new(inference: Arc<MCTSBatchedInference>, _num_workers: usize) -> Self {
        Self {
            inference,
            workers: Vec::new(),
            stop_flag: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Start worker threads
    pub fn start(&mut self) {
        for _ in 0..self.workers.capacity() {
            let inference = self.inference.clone();
            let stop = self.stop_flag.clone();

            let handle = std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    if inference.should_process() {
                        let _ = inference.process_batch();
                    } else {
                        std::thread::sleep(Duration::from_micros(100));
                    }
                }
            });

            self.workers.push(handle);
        }
    }

    /// Stop all workers
    pub fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

impl Drop for ParallelBatchProcessor {
    fn drop(&mut self) {
        self.stop();
    }
}
