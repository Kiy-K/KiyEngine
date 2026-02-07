// src/search/time.rs

use std::time::{Duration, Instant};

pub struct TimeManager {
    start_time: Instant,
    allocated_time: Duration,
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

        Self {
            start_time: Instant::now(),
            allocated_time: allocated,
            move_overhead: overhead,
        }
    }

    pub fn should_stop(&self) -> bool {
        self.start_time.elapsed() >= self.allocated_time
    }

    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    pub fn overhead(&self) -> Duration {
        self.move_overhead
    }
}
