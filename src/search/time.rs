// src/search/time.rs

use std::time::{Duration, Instant};

pub struct TimeManager {
    start_time: Instant,
    // Soft limit: target time — can stop here if best move is stable
    soft_limit: Duration,
    // Hard limit: absolute max — always stop here
    hard_limit: Duration,
    move_overhead: Duration,
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
    ) -> Self {
        let (time, inc) = if is_white {
            (wtime, winc)
        } else {
            (btime, binc)
        };
        let overhead = Duration::from_millis(move_overhead_ms as u64);

        // Basic time management formula
        let mtg = if movestogo == 0 { 30 } else { movestogo };
        let base_time = time / mtg;
        let increment_time = inc * 3 / 4; // Use 75% of increment

        let mut allocated = Duration::from_millis((base_time + increment_time) as u64);

        // Always subtract overhead
        if allocated > overhead {
            allocated -= overhead;
        } else {
            allocated = Duration::from_millis(10); // Minimum safety
        }

        // Don't spend more than 20% of total remaining time on one move
        let max_safe = Duration::from_millis((time / 5) as u64);
        if allocated > max_safe {
            allocated = max_safe;
        }

        // Soft limit = base allocation
        // Hard limit = 2.5× soft limit, capped at 33% of remaining time
        let soft = allocated;
        let hard_cap = Duration::from_millis((time / 3) as u64);
        let hard = (soft * 5 / 2).min(hard_cap);

        Self {
            start_time: Instant::now(),
            soft_limit: soft,
            hard_limit: hard,
            move_overhead: overhead,
        }
    }

    /// Check if we should stop searching (hard limit).
    /// Used inside the search for periodic abort checks.
    pub fn should_stop(&self) -> bool {
        self.start_time.elapsed() >= self.hard_limit
    }

    /// Check if we've exceeded the soft limit.
    /// Used in iterative deepening to decide whether to start a new depth.
    pub fn soft_stop(&self) -> bool {
        self.start_time.elapsed() >= self.soft_limit
    }

    /// Scale the soft limit based on best-move stability.
    /// If best move has been stable for many iterations, use less time.
    /// If best move keeps changing, use more time.
    pub fn should_stop_soft(&self, stability: u32) -> bool {
        let factor = match stability {
            0..=1 => 2.0_f64, // Unstable — use 2× soft limit
            2..=3 => 1.5,     // Somewhat stable
            4..=5 => 1.0,     // Normal
            _ => 0.6,         // Very stable — stop early
        };
        let adjusted = Duration::from_secs_f64(self.soft_limit.as_secs_f64() * factor);
        self.start_time.elapsed() >= adjusted
    }

    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    pub fn overhead(&self) -> Duration {
        self.move_overhead
    }
}
