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
//! Samples live in fixed ring buffers (the last [`RING`] per series); percentiles for the status message and
//! the log come from a histogram of the ring (no sort); [`Series::summary`] sorts a copy and is for the final report.

use std::time::Duration;

/// Decisions the [`DecisionEstimator`] looks back over (about 2.5 s at 25 snapshots/s).
pub const ESTIMATE_WINDOW: usize = 64;

/// A conservative estimate of how long the next decision takes (task 4.1b): a **rolling quantile**
/// of the last [`ESTIMATE_WINDOW`] decision times, not a mean. Decision time is skewed and bimodal
/// (the hybrid brain usually needs ~6 ms but extends to 15 ms in danger), and the estimate only
/// picks the input slot a decision is aimed at: a mean sits between the modes and is wrong for both,
/// where a high quantile errs towards one more tick of latency that the driver's hold (the
/// decision goes out for exactly the tick it was predicted for) makes harmless. Allocation-free.
#[derive(Debug, Clone)]
pub struct DecisionEstimator {
    ring: [u32; ESTIMATE_WINDOW],
    len: usize,
    next: usize,
    quantile: f64,
    initial: Duration,
}

impl DecisionEstimator {
    pub fn new(quantile: f64, initial: Duration) -> Self {
        DecisionEstimator {
            ring: [0; ESTIMATE_WINDOW],
            len: 0,
            next: 0,
            quantile: quantile.clamp(0.0, 1.0),
            initial,
        }
    }

    pub fn push(&mut self, d: Duration) {
        self.ring[self.next] = u32::try_from(d.as_micros()).unwrap_or(u32::MAX);
        self.next = (self.next + 1) % ESTIMATE_WINDOW;
        self.len = (self.len + 1).min(ESTIMATE_WINDOW);
    }

    /// The estimate: `initial` before the first decision, the quantile after.
    pub fn estimate(&self) -> Duration {
        if self.len == 0 {
            return self.initial;
        }
        let mut v = [0u32; ESTIMATE_WINDOW];
        v[..self.len].copy_from_slice(&self.ring[..self.len]);
        let v = &mut v[..self.len];
        v.sort_unstable();
        Duration::from_micros(u64::from(v[((self.len - 1) as f64 * self.quantile).ceil() as usize]))
    }
}

/// Samples kept per series.
pub const RING: usize = 1 << 15;

/// A ring of microsecond samples.
#[derive(Debug, Clone)]
pub struct Series {
    samples: Vec<u32>,
    next: usize,
    count: u64,
    max_us: u32,
    /// A histogram of the samples **currently in the ring** (a sample leaving the ring is subtracted), so a percentile
    /// is a short scan of the buckets instead of a copy and a sort of 32 768 values (task 4.4: the status message and
    /// the 10 s log line run on the decision thread, and a sorted copy cost about 0.9 ms per series).
    hist: Vec<u32>,
}

/// Values below this get a bucket each; above it, 128 buckets per power of two (a bucket is at most 1/128 = 0.78% wide,
/// the midpoint is reported: error <= 0.4%).
const EXACT_BELOW: u32 = 256;
const SUB_BITS: u32 = 7;
/// 256 exact buckets plus octaves 2^8 .. 2^31 at 128 buckets each.
const BUCKETS: usize = EXACT_BELOW as usize + 24 * (1 << SUB_BITS);

fn bucket_of(us: u32) -> usize {
    if us < EXACT_BELOW {
        return us as usize;
    }
    let msb = 31 - us.leading_zeros();
    let shift = msb - SUB_BITS;
    let sub = (us >> shift) - (1 << SUB_BITS);
    EXACT_BELOW as usize + ((msb - 8) as usize) * (1 << SUB_BITS) + sub as usize
}

/// The value reported for a bucket: exact below [`EXACT_BELOW`], the midpoint above.
fn bucket_value(idx: usize) -> u32 {
    if idx < EXACT_BELOW as usize {
        return idx as u32;
    }
    let rel = idx - EXACT_BELOW as usize;
    let msb = (rel >> SUB_BITS) as u32 + 8;
    let sub = (rel & ((1 << SUB_BITS) - 1)) as u32;
    let shift = msb - SUB_BITS;
    let lo = u64::from((1u32 << SUB_BITS) + sub) << shift;
    let mid = lo + (1u64 << shift) / 2;
    u32::try_from(mid).unwrap_or(u32::MAX)
}

impl Default for Series {
    fn default() -> Self {
        Series {
            samples: Vec::with_capacity(RING),
            next: 0,
            count: 0,
            max_us: 0,
            hist: vec![0; BUCKETS],
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
            let old = self.samples[self.next];
            self.hist[bucket_of(old)] -= 1;
            self.samples[self.next] = us;
        }
        self.hist[bucket_of(us)] += 1;
        self.next = (self.next + 1) % RING;
        self.count += 1;
        self.max_us = self.max_us.max(us);
    }

    /// The same summary as [`Series::summary`] to within 0.4% (exact below 256 us), from the histogram: a scan of at most
    /// a few thousand counters, no allocation, no sort. This is what the decision thread uses (status message, log line);
    /// `summary` stays exact for the final report and the tests.
    pub fn quick_summary(&self) -> Summary {
        let n = self.samples.len();
        if n == 0 {
            return Summary::default();
        }
        let rank = |p: f64| ((n - 1) as f64 * p).round() as u64;
        let (r50, r90, r99) = (rank(0.50), rank(0.90), rank(0.99));
        let (mut p50, mut p90, mut p99) = (None, None, None);
        let mut seen = 0u64;
        for (idx, &c) in self.hist.iter().enumerate() {
            if c == 0 {
                continue;
            }
            seen += u64::from(c);
            if p50.is_none() && seen > r50 {
                p50 = Some(bucket_value(idx));
            }
            if p90.is_none() && seen > r90 {
                p90 = Some(bucket_value(idx));
            }
            if seen > r99 {
                p99 = Some(bucket_value(idx));
                break;
            }
        }
        Summary {
            count: self.count,
            p50_us: p50.unwrap_or(0).min(self.max_us),
            p90_us: p90.unwrap_or(0).min(self.max_us),
            p99_us: p99.unwrap_or(0).min(self.max_us),
            max_us: self.max_us,
        }
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
    /// Task 3.7a: what the brain reported about its own decisions ([`ddai_brain::PlanTelemetry`]): candidates scored (the
    /// "microseconds" of this series are a count), the proposer's time and the search's time. Only decisions the brain
    /// made are counted.
    pub candidates: Series,
    pub proposal: Series,
    pub search: Series,
    /// `brain` restricted to the decisions the brain made (the calls with a target in reach: the ones that search); the
    /// all-calls `brain` series is diluted by the cheap ones.
    pub brain_made: Series,
    pub slots: SlotStats,
}

impl LatencyStats {
    /// One decision's bot-side timing.
    pub fn record(&mut self, total: Duration, brain: Duration) {
        self.total.push(total);
        self.brain.push(brain);
        self.overhead.push(total.saturating_sub(brain));
    }

    /// What the brain said about one decision it made (see the fields). Only decisions that searched count: a
    /// decision that returned early has no verdict of its own.
    pub fn record_plan(&mut self, p: &ddai_brain::PlanTelemetry, brain: Duration) {
        if !p.searched {
            return;
        }
        self.brain_made.push(brain);
        self.candidates.push(Duration::from_micros(u64::from(p.candidates)));
        self.proposal.push(Duration::from_micros(u64::from(p.proposal_us)));
        self.search.push(Duration::from_micros(u64::from(p.search_us)));
    }

    /// `total`, `brain` and `overhead` summaries for the bridge's status message (histogram scans: well under 0.05 ms).
    pub fn status_summaries(&self) -> (Summary, Summary, Summary) {
        (
            self.total.quick_summary(),
            self.brain.quick_summary(),
            self.overhead.quick_summary(),
        )
    }

    /// `key=value` lines for the log, from the histograms (within 0.4% of the exact percentiles).
    pub fn report(&self) -> String {
        let line = |name: &str, s: Summary| {
            format!(
                "{name}: n={} p50={}us p90={}us p99={}us max={}us",
                s.count, s.p50_us, s.p90_us, s.p99_us, s.max_us
            )
        };
        [
            line("total", self.total.quick_summary()),
            line("brain", self.brain.quick_summary()),
            line("overhead", self.overhead.quick_summary()),
            line("pick", self.pick.quick_summary()),
            line("queue", self.queue.quick_summary()),
            line("wire", self.wire.quick_summary()),
            line("candidates (count)", self.candidates.quick_summary()),
            line("proposal", self.proposal.quick_summary()),
            line("search", self.search.quick_summary()),
            line("brain (decisions made)", self.brain_made.quick_summary()),
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
    fn the_brains_own_account_of_a_decision_feeds_three_series() {
        let mut l = LatencyStats::default();
        for k in 1..=100u32 {
            l.record_plan(
                &ddai_brain::PlanTelemetry {
                    searched: true,
                    candidates: k,
                    proposal_us: 10 * k,
                    search_us: 4000 + k,
                    ..ddai_brain::PlanTelemetry::default()
                },
                Duration::from_micros(5000 + u64::from(k)),
            );
        }
        let (c, p, s) = (l.candidates.summary(), l.proposal.summary(), l.search.summary());
        assert_eq!((c.count, c.p50_us, c.p99_us, c.max_us), (100, 51, 99, 100));
        assert_eq!((p.p99_us, p.max_us), (990, 1000));
        assert_eq!((s.p50_us, s.max_us), (4051, 4100));
        assert_eq!(l.brain_made.summary().max_us, 5100);
        assert_eq!(l.brain.summary().count, 0, "only record() feeds the all-calls series");
        assert!(l.report().contains("candidates (count): n=100"), "{}", l.report());
        assert_eq!(LatencyStats::default().candidates.summary(), Summary::default());
        // A decision that did not search leaves every series as it was.
        l.record_plan(
            &ddai_brain::PlanTelemetry {
                searched: false,
                candidates: 7,
                proposal_us: 7,
                search_us: 7,
                ..ddai_brain::PlanTelemetry::default()
            },
            Duration::from_micros(9),
        );
        assert_eq!((l.candidates.summary().count, l.brain_made.summary().count), (100, 100));
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

    /// A heavy-tailed, deterministic sample stream (microseconds): mostly 50-400, a long tail to ~30 ms.
    fn stream(n: usize) -> Vec<u32> {
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let u = (x % 1_000_000) as f64 / 1_000_000.0;
                (50.0 + 30_000.0 * u.powi(6) + 300.0 * u) as u32
            })
            .collect()
    }

    fn close(quick: u32, exact: u32) -> bool {
        quick == exact || (f64::from(quick) - f64::from(exact)).abs() <= 0.01 * f64::from(exact)
    }

    #[test]
    fn the_histogram_summary_matches_the_sorted_one_within_one_percent_also_after_the_ring_wraps() {
        for n in [10, 1_000, RING, RING + 1, 3 * RING + 123] {
            let mut s = Series::default();
            for us in stream(n) {
                s.push(Duration::from_micros(u64::from(us)));
            }
            let (q, e) = (s.quick_summary(), s.summary());
            assert_eq!((q.count, q.max_us), (e.count, e.max_us), "n={n}");
            assert!(close(q.p50_us, e.p50_us), "n={n} p50 {q:?} vs {e:?}");
            assert!(close(q.p90_us, e.p90_us), "n={n} p90 {q:?} vs {e:?}");
            assert!(close(q.p99_us, e.p99_us), "n={n} p99 {q:?} vs {e:?}");
        }
        assert_eq!(Series::default().quick_summary(), Summary::default());
    }

    #[test]
    fn small_values_are_exact_and_every_value_lands_in_a_bucket_of_at_most_one_percent() {
        let mut s = Series::default();
        for us in 1..=100u64 {
            s.push(Duration::from_micros(us));
        }
        assert_eq!(s.quick_summary(), s.summary(), "below 256 us the histogram is exact");
        for us in [
            255u32,
            256,
            257,
            1_000,
            65_535,
            65_536,
            1_000_000,
            u32::MAX / 2,
            u32::MAX,
        ] {
            let idx = bucket_of(us);
            assert!(idx < BUCKETS, "{us}");
            assert!(
                close(bucket_value(idx), us),
                "{us} -> bucket {idx} -> {}",
                bucket_value(idx)
            );
        }
    }

    /// Task 4.4 acceptance: the status path costs the decision thread at most 0.05 ms per call (it was three sorted copies of
    /// 32 768 samples, about 2.7 ms). Best of 31 batches of 20 calls (the cost without host pauses: the VM is shared).
    #[test]
    fn the_status_summaries_of_full_rings_cost_under_fifty_microseconds() {
        let mut l = LatencyStats::default();
        for us in stream(2 * RING) {
            let d = Duration::from_micros(u64::from(us));
            l.record(d + Duration::from_micros(30), d);
        }
        let mut per_call = Vec::new();
        let mut sink = 0u64;
        for _ in 0..31 {
            let t = std::time::Instant::now();
            for _ in 0..20 {
                let (a, b, c) = l.status_summaries();
                sink += u64::from(a.p99_us + b.p99_us + c.p99_us);
            }
            per_call.push(t.elapsed() / 20);
        }
        let best = per_call.into_iter().min().unwrap();
        eprintln!("status_summaries: {best:?} per call (sink {sink})");
        assert!(best < Duration::from_micros(50), "status_summaries took {best:?}");
    }
}
