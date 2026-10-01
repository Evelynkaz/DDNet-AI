//! Decision latency (D-042): snapshot arrival -> input on the wire, per decision, p50/p99, with the
//! brain's share separated from the bot's own overhead.
//!
//! Three clocks per decision (all wall time; on this VM wall time includes the host's ~10 ms pauses,
//! D-045 — `ddai-env`'s report prints the machine's pause baseline next to arena timings, and this
//! one is read against the same baseline):
//!
//! - `total`: from the moment the bot thread takes the snapshot up to `Client::set_input` returning
//!   (queue hop excluded — see `queue` for that);
//! - `brain`: just the `Brain::decide_in` call (plus nothing else);
//! - `overhead` = `total - brain`: everything the bot itself does — collapsing, LiveWorld update,
//!   in-flight input bookkeeping, target selection, prediction, observation, post-filters, encoding.
//!   **The D-042 target is overhead p99 <= 0.5 ms**, so the brain has the budget;
//! - `pick`: target selection alone, a part of `overhead` (reachability floods, seal searches);
//! - `queue`: the driver thread assembling the snapshot -> the bot thread starting on it (channel
//!   hop and waiting behind other events);
//! - `wire`: the driver's report ([`ddai_client::ClientEvent::InputLatency`]) from snapshot
//!   arrival to the first `NETMSG_INPUT` carrying the decision leaving the socket. It contains the
//!   session's own input cadence (one input per predicted tick, up to 20 ms apart), which no
//!   amount of bot speed changes.
//!
//! Samples live in fixed ring buffers (the last [`RING`] per series); percentiles are computed on a
//! sorted copy when asked, never on the hot path.

use std::time::Duration;

/// Samples kept per series.
pub const RING: usize = 1 << 15;

/// A ring of microsecond samples.
#[derive(Debug, Clone)]
pub struct Series {
    samples: Vec<u32>,
    next: usize,
    count: u64,
    max_us: u32,
}

impl Default for Series {
    fn default() -> Self {
        Series {
            samples: Vec::with_capacity(RING),
            next: 0,
            count: 0,
            max_us: 0,
        }
    }
}

/// Percentile summary, microseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Summary {
    pub count: u64,
    pub p50_us: u32,
    pub p90_us: u32,
    pub p99_us: u32,
    pub max_us: u32,
}

impl Series {
    pub fn push(&mut self, d: Duration) {
        let us = u32::try_from(d.as_micros()).unwrap_or(u32::MAX);
        if self.samples.len() < RING {
            self.samples.push(us);
        } else {
            self.samples[self.next] = us;
        }
        self.next = (self.next + 1) % RING;
        self.count += 1;
        self.max_us = self.max_us.max(us);
    }

    pub fn summary(&self) -> Summary {
        if self.samples.is_empty() {
            return Summary::default();
        }
        let mut sorted = self.samples.clone();
        sorted.sort_unstable();
        let at = |p: f64| sorted[((sorted.len() - 1) as f64 * p).round() as usize];
        Summary {
            count: self.count,
            p50_us: at(0.50),
            p90_us: at(0.90),
            p99_us: at(0.99),
            max_us: self.max_us,
        }
    }
}

/// How decisions landed against the input slots (task 4.1): `tick` is the first `NETMSG_INPUT` that
/// carried the decision.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SlotStats {
    /// Decisions whose input went out.
    pub decisions: u64,
    /// ... in the first slot due after their snapshot (`tick == first_slot`).
    pub in_first_slot: u64,
    /// ... one or more slots later (`tick > first_slot`): the decision missed the slot.
    pub missed_first_slot: u64,
    /// ... on exactly the tick the bot predicted (`expected_tick`), which is `first_slot` or one
    /// later when the bot saw the miss coming.
    pub as_predicted: u64,
    /// ... later than predicted (the prediction was a tick short).
    pub later_than_predicted: u64,
    /// ... earlier than predicted.
    pub earlier_than_predicted: u64,
}

impl SlotStats {
    pub fn note(&mut self, tick: i32, first_slot: i32, expected: i32) {
        self.decisions += 1;
        if tick <= first_slot {
            self.in_first_slot += 1;
        } else {
            self.missed_first_slot += 1;
        }
        match tick.cmp(&expected) {
            std::cmp::Ordering::Equal => self.as_predicted += 1,
            std::cmp::Ordering::Greater => self.later_than_predicted += 1,
            std::cmp::Ordering::Less => self.earlier_than_predicted += 1,
        }
    }
}

/// All series.
#[derive(Debug, Clone, Default)]
pub struct LatencyStats {
    pub total: Series,
    pub brain: Series,
    pub overhead: Series,
    /// Target selection alone (part of `overhead`): reachability floods and seal searches live here.
    pub pick: Series,
    pub queue: Series,
    pub wire: Series,
    pub slots: SlotStats,
}

impl LatencyStats {
    /// One decision's bot-side timing.
    pub fn record(&mut self, total: Duration, brain: Duration) {
        self.total.push(total);
        self.brain.push(brain);
        self.overhead.push(total.saturating_sub(brain));
    }

    /// `key=value` lines for the log / report.
    pub fn report(&self) -> String {
        let line = |name: &str, s: Summary| {
            format!(
                "{name}: n={} p50={}us p90={}us p99={}us max={}us",
                s.count, s.p50_us, s.p90_us, s.p99_us, s.max_us
            )
        };
        [
            line("total", self.total.summary()),
            line("brain", self.brain.summary()),
            line("overhead", self.overhead.summary()),
            line("pick", self.pick.summary()),
            line("queue", self.queue.summary()),
            line("wire", self.wire.summary()),
            format!(
                "slots: decisions={} first_slot={} missed_first_slot={} as_predicted={} later_than_predicted={} earlier_than_predicted={}",
                self.slots.decisions,
                self.slots.in_first_slot,
                self.slots.missed_first_slot,
                self.slots.as_predicted,
                self.slots.later_than_predicted,
                self.slots.earlier_than_predicted
            ),
        ]
        .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_of_a_known_distribution() {
        let mut s = Series::default();
        for us in 1..=100u64 {
            s.push(Duration::from_micros(us));
        }
        let sum = s.summary();
        assert_eq!(sum.count, 100);
        assert_eq!(sum.p50_us, 51, "index round(99 * 0.5) = 50 -> value 51");
        assert_eq!(sum.p99_us, 99);
        assert_eq!(sum.max_us, 100);
        assert_eq!(Series::default().summary(), Summary::default());
    }

    #[test]
    fn the_ring_keeps_the_most_recent_samples() {
        let mut s = Series::default();
        for _ in 0..RING {
            s.push(Duration::from_micros(10));
        }
        for _ in 0..RING {
            s.push(Duration::from_micros(500));
        }
        let sum = s.summary();
        assert_eq!(sum.p50_us, 500, "old samples were overwritten");
        assert_eq!(sum.count, 2 * RING as u64);
        assert_eq!(sum.max_us, 500);
    }

    #[test]
    fn overhead_is_total_minus_brain() {
        let mut l = LatencyStats::default();
        l.record(Duration::from_micros(1500), Duration::from_micros(1200));
        l.record(Duration::from_micros(100), Duration::from_micros(400)); // clock skew: saturates
        assert_eq!(l.overhead.summary().max_us, 300);
        assert_eq!(l.overhead.summary().count, 2);
        assert!(l.report().contains("overhead: n=2"));
    }

    #[test]
    fn recording_allocates_nothing_once_the_ring_exists() {
        let mut l = LatencyStats::default();
        let info = allocation_counter::measure(|| {
            for i in 0..1000u64 {
                l.record(Duration::from_micros(100 + i), Duration::from_micros(50));
            }
        });
        // `Vec::with_capacity(RING)` reserved everything up front in `default()`.
        assert_eq!(info.count_total, 0, "{info:?}");
    }
}
