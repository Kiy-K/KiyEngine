// src/search/time.rs
//
// Stockfish-inspired time management with soft/hard bounds,
// dynamic adjustment based on search stability and score trends.

use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};
use std::time::Instant;

pub struct TimeManager {
    start_time: Instant,
    /// Optimum time (soft bound) — base allocation in ms
    optimum_ms: f64,
    /// Maximum time (hard bound) — absolute cap in ms
    maximum_ms: f64,
    /// Game ply (half-moves from start) — used during construction for scaling
    _ply: u32,
    // --- Dynamic adjustment factors (updated during search) ---
    /// Best move stability: depth at which current best move was first chosen
    best_move_depth: AtomicU32,
    /// Number of best-move changes accumulated across all threads
    best_move_changes: AtomicU32,
    /// Previous iteration score (for fallingEval)
    prev_score: AtomicI32,
    /// Average score over recent iterations (exponential moving average × 100)
    avg_score_x100: AtomicI32,
    /// Completed depth of last iteration
    completed_depth: AtomicU32,
}

impl TimeManager {
    pub fn new(
        wtime: u32,
        btime: u32,
        winc: u32,
        binc: u32,
        movestogo: u32,
        move_overhead_ms: u32,
        is_white: bool,
        ply: u32,
    ) -> Self {
        let (time_ms, inc_ms) = if is_white {
            (wtime as f64, winc as f64)
        } else {
            (btime as f64, binc as f64)
        };
        let overhead = move_overhead_ms as f64;

        // --- Compute available time pool ---
        // Estimate moves to go if not given (sudden death)
        let mtg = if movestogo > 0 {
            movestogo as f64
        } else {
            // Stockfish-style: fewer estimated moves as game progresses
            (40.0 - (ply as f64 * 0.3)).clamp(15.0, 40.0)
        };

        // Total time available for this move (pool)
        let time_left = (time_ms + inc_ms * (mtg - 1.0) - overhead * mtg).max(1.0);

        // --- Compute optimal and maximum time (Stockfish-aligned) ---
        let (opt_scale, max_scale) = if movestogo == 0 {
            // Sudden death / increment: Stockfish-style opt/max scaling
            // optScale: grows with ply but capped conservatively at 0.039
            let opt_raw =
                (0.0120 + (ply as f64 + 2.43).powf(0.462) * 0.0084).min(0.039);
            let opt = opt_raw * time_left;

            // maxScale: allows extending for unstable positions but capped
            let max_raw =
                (0.054 + (ply as f64 + 2.0).powf(0.442) * 0.074).min(6.0);
            let max_from_pool = max_raw * time_left;
            // Hard cap: never burn more than 25% of remaining clock on one move
            let max = max_from_pool.min(0.25 * time_ms - overhead);

            (opt, max)
        } else {
            // Classical: fixed moves to go
            let opt = 0.90 * time_left / mtg;
            let max = (1.5 + 0.10 * mtg) * opt;
            (opt, max)
        };

        // Ensure minimum time and cap at remaining time
        let optimum = opt_scale.max(10.0).min(0.20 * time_ms); // Never exceed 20% of clock
        let maximum = max_scale.max(optimum).min(time_ms - overhead).max(10.0);

        Self {
            start_time: Instant::now(),
            optimum_ms: optimum,
            maximum_ms: maximum,
            _ply: ply,
            best_move_depth: AtomicU32::new(0),
            best_move_changes: AtomicU32::new(0),
            prev_score: AtomicI32::new(0),
            avg_score_x100: AtomicI32::new(0),
            completed_depth: AtomicU32::new(0),
        }
    }

    /// Elapsed time in milliseconds
    #[inline]
    pub fn elapsed_ms(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64() * 1000.0
    }

    /// Hard stop: always abort search if exceeded. Checked periodically inside search.
    #[inline]
    pub fn should_stop(&self) -> bool {
        self.elapsed_ms() >= self.maximum_ms
    }

    /// Soft stop with dynamic adjustment. Checked between ID iterations by main thread.
    /// Incorporates: fallingEval, bestMoveInstability, bestMoveStability (timeReduction).
    pub fn should_stop_soft(&self) -> bool {
        let elapsed = self.elapsed_ms();
        let depth = self.completed_depth.load(Ordering::Relaxed);

        // --- Factor 1: Best-move stability (timeReduction) ---
        // If best move has been stable since early depth, reduce time.
        // Sigmoid: stable_depth < center → reduction > 1 (use less time)
        let bm_depth = self.best_move_depth.load(Ordering::Relaxed) as f64;
        let center = bm_depth + 10.0;
        let time_reduction = 0.70 + 0.80 / (1.0 + (-0.50 * (depth as f64 - center)).exp());
        let reduction_factor = 1.40 / (2.20 * time_reduction);

        // --- Factor 2: Falling eval ---
        // If score is dropping compared to average, spend more time
        let cur_score = self.prev_score.load(Ordering::Relaxed) as f64;
        let avg_score = self.avg_score_x100.load(Ordering::Relaxed) as f64 / 100.0;
        let falling_eval = ((12.0 + 2.0 * (avg_score - cur_score)) / 100.0).clamp(0.60, 1.70);

        // --- Factor 3: Best-move instability ---
        // More changes → spend more time, but capped to prevent runaway extension
        let changes = self.best_move_changes.load(Ordering::Relaxed) as f64;
        let instability = (1.0 + 1.0 * changes / 10.0).min(1.50);

        // Combined dynamic time — product of all factors, capped at maximum
        let total_time = self.optimum_ms * falling_eval * reduction_factor * instability;

        elapsed >= total_time.min(self.maximum_ms)
    }

    /// Update search state after each completed depth iteration.
    /// Called by main thread from the ID loop.
    pub fn update_iteration(&self, depth: u32, score: i32, best_move_changed: bool) {
        self.completed_depth.store(depth, Ordering::Relaxed);

        // Update falling eval tracking
        if depth <= 1 {
            // Initialize on first iteration
            self.avg_score_x100.store(score * 100, Ordering::Relaxed);
        } else {
            // Exponential moving average: avg = avg * 0.7 + score * 0.3
            let old_avg = self.avg_score_x100.load(Ordering::Relaxed);
            let new_avg = (old_avg as f64 * 0.70 + score as f64 * 100.0 * 0.30) as i32;
            self.avg_score_x100.store(new_avg, Ordering::Relaxed);
        }
        self.prev_score.store(score, Ordering::Relaxed);

        // Update best-move stability tracking
        if best_move_changed {
            self.best_move_depth.store(depth, Ordering::Relaxed);
            self.best_move_changes.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Optimum time in ms (for external queries)
    pub fn optimum(&self) -> f64 {
        self.optimum_ms
    }

    /// Maximum time in ms
    pub fn maximum(&self) -> f64 {
        self.maximum_ms
    }

    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }
}
