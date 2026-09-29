//! `Clock`: the time source [`crate::planner::Planner::decide_production`] (D-041's real
//! deadline-driven search, review round 1 finding F1) reads instead of calling
//! `std::time::Instant`/`Date.now` directly. Injected so production-search tests are
//! deterministic (a [`StepClock`] advances by a fixed increment every query, with no real-time
//! flakiness) while live play uses [`WallClock`]. **The parity path
//! (`Planner::decide`/`decide_once`, TS's own `budgetMs`/`hardMs`) never reads a clock at all** —
//! not even `WallClock` — matching the constraint that fixed-iteration mode stays deterministic
//! regardless of wall time (`docs/DECISIONS.md` D-017).

use std::cell::Cell;
use std::time::Instant;

/// A source of "milliseconds since some fixed starting point". Only the differences between two
/// calls matter to any caller here (nothing reads an absolute epoch).
pub trait Clock {
    fn now_ms(&self) -> f64;
}

/// Real wall-clock time (`std::time::Instant`), for live play.
#[derive(Debug)]
pub struct WallClock {
    start: Instant,
}

impl WallClock {
    pub fn new() -> Self {
        WallClock { start: Instant::now() }
    }
}

impl Default for WallClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for WallClock {
    fn now_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
}

/// A deterministic fake clock for tests: `now_ms()` advances by a fixed `step_ms` every time it
/// is *read* (not on a wall-clock timer), so a test can assert exactly how many candidates/ticks
/// a deadline of a given size lets the search get through, with zero real-time flakiness. Interior
/// mutability (`Cell`) so it can be read from `&self` call sites (`Clock::now_ms(&self)`) while
/// still advancing.
#[derive(Debug)]
pub struct StepClock {
    ticks: Cell<f64>,
    step_ms: f64,
}

impl StepClock {
    pub fn new(step_ms: f64) -> Self {
        StepClock {
            ticks: Cell::new(0.0),
            step_ms,
        }
    }
}

impl Clock for StepClock {
    fn now_ms(&self) -> f64 {
        let t = self.ticks.get();
        self.ticks.set(t + self.step_ms);
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wall_clock_is_monotonically_nondecreasing() {
        let c = WallClock::new();
        let a = c.now_ms();
        let b = c.now_ms();
        assert!(b >= a);
    }

    #[test]
    fn step_clock_advances_by_a_fixed_amount_per_read() {
        let c = StepClock::new(0.5);
        assert_eq!(c.now_ms(), 0.0);
        assert_eq!(c.now_ms(), 0.5);
        assert_eq!(c.now_ms(), 1.0);
    }
}
