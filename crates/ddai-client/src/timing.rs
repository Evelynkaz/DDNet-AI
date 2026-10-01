// Ported from DDNet `src/engine/client/client.cpp` (pinned rev
// c9d208138f85755521f16a0096b6fe036c5c8698, "20.1"): the `NETMSG_INPUTTIMING`/prediction-margin
// feedback loop (`client.cpp:2084-2108`, `2902-2969`, `5429-5437`) and `SendInput`
// (`client.cpp:350-403`), which carry the original Teeworlds zlib-style notice:
//
//   (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information.
//   If you are missing that file, acquire a complete release at teeworlds.com.
//
// This is an *altered* source version: rewritten in safe Rust, deliberately narrowed to the parts
// that matter for a headless client with no rendering/local-prediction of its own — see this
// module's top-level docs for exactly what is and is not ported, cited against the exact lines.
//
//! Input timing: when to send the next `NETMSG_INPUT`, and how to keep [`crate::smooth_time`]'s
//! predicted clock tracking the server via `NETMSG_INPUTTIMING` feedback — task 2.3 acceptance
//! criterion 3, and the "old TS bot ignored INPUTTIMING" gap (decision D-023).
//!
//! # What is ported, and what is deliberately not (read this before touching the algorithm)
//!
//! The real client (`CClient`) maintains **two** [`SmoothTime`](crate::smooth_time::SmoothTime)
//! clocks per connection: `m_aGameTime` (driven by snapshot arrival, `ADJUSTDIRECTION_DOWN`) and
//! `m_PredictedTime` (driven by `NETMSG_INPUTTIMING`, `ADJUSTDIRECTION_UP`). `m_aGameTime` exists
//! *only* to let the renderer smoothly interpolate between the last two displayed snapshots
//! (`client.cpp:2902-2949`'s `SNAP_PREV`/`SNAP_CURRENT` pointer walk, `m_aGameIntraTick*`) — this
//! bot has no renderer, so this port has no `game_time` at all. What is left, and *is* ported
//! faithfully (these are the parts that decide "is it time to send an input yet" and "was that
//! input on time", i.e. everything task acceptance criterion 3/h actually asks for):
//!
//! - `m_PredictedTime` itself (here: [`InputTiming::predicted_time`]), including its two-snapshot
//!   bootstrap (`client.cpp:2307-2320`) and the "too far off, reset" safety net
//!   (`client.cpp:2950-2954`) — the latter compares the newly-computed predicted tick against the
//!   most recently *received* snapshot's tick ([`InputTiming::latest_snapshot_tick`]) rather than
//!   against `SNAP_PREV`'s tick specifically; those two differ by at most one server tick in
//!   practice (`SNAP_PREV` is always either the latest or second-latest received snapshot), well
//!   inside [`max_latency_ticks`]'s margin, so this is a faithful behavioural equivalent, not a
//!   relaxation.
//! - The per-frame predicted-tick advance and `SendInput` trigger (`client.cpp:2939-2963`,
//!   [`InputTiming::advance`]).
//! - The `NETMSG_INPUTTIMING` feedback itself (`client.cpp:2084-2108`, [`InputTiming::on_input_timing`]),
//!   including the 200-entry input history it looks the tick up in (`m_aInputs`, here
//!   [`InputTiming::history`]) — with one intentional, documented micro-difference: C++ guards the
//!   feedback on `if(Target)` (an `int64_t Target = 0` sentinel for "no matching history entry
//!   found"), which would also (mis-)skip a legitimately-computed `Target` of exactly zero; this
//!   port uses `Option<i64>` instead, so a genuine zero is not silently dropped. The two are
//!   observably identical in every case except that vanishingly unlikely exact-zero coincidence.
//! - [`max_latency_ticks`]/[`PredictionMargin`]-equivalent (`client.cpp:5429-5437`).
//!
//! `SendInput`'s per-dummy loop, `m_aCurrentInput` wraparound bookkeeping, and the sixup
//! `PlayerFlags_SixToSeven` translation are all dummy/0.7-only concerns this project's 0.6+DDNet,
//! single-connection scope (see the crate root docs) never exercises, so they are not ported
//! either — [`InputTiming::advance`] returns the one piece [`crate::session::Session`] needs (the
//! predicted tick a `NETMSG_INPUT` should now be sent for), and the session builds/sends that
//! message itself.

use crate::smooth_time::{AdjustDirection, SmoothTime};
use std::collections::VecDeque;

/// `SERVER_TICK_SPEED` (`protocol.h:81`) — fixed for every 0.6+DDNet server; not negotiated.
pub const GAME_TICK_SPEED: i32 = 50;

/// `cl_prediction_margin`'s default (`config_variables.h:27`) — also exactly what
/// `CClient::PredictionMargin` falls back to when the server lacks the `SyncWeaponInput`
/// capability (`client.cpp:5434-5437`), so this single default is correct either way.
pub const DEFAULT_PREDICTION_MARGIN_MS: i32 = 10;

/// How many past `NETMSG_INPUT` sends to remember for `NETMSG_INPUTTIMING` lookups —
/// `m_aInputs[i][200]` (`client.h`).
const INPUT_HISTORY_LEN: usize = 200;

/// How many raw `time_left_ms` samples [`MarginStats`] keeps for percentile estimation. Overall
/// count/late-count/mean are exact and unbounded (never windowed) — only the percentile estimate
/// is over this bounded recent window, so a very long session still reports meaningful recent
/// percentiles without unbounded memory growth (task acceptance criterion 7: bounded memory).
const MARGIN_SAMPLE_WINDOW: usize = 8192;

/// One past `NETMSG_INPUT` send, remembered so a later `NETMSG_INPUTTIMING` for the same
/// `pred_tick` can compute how far off the predicted clock was (`client.cpp:2099-2104`).
#[derive(Debug, Clone, Copy)]
struct HistoryEntry {
    tick: i32,
    /// `m_PredictedTime.Get(Now)` at the moment this input was sent.
    predicted_time_ns: i64,
    /// `Now` (wall-clock) at the moment this input was sent.
    sent_at_ns: i64,
}

/// A snapshot of [`MarginStats`], suitable for logging/reporting (task acceptance criterion h:
/// "report the margin distribution").
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarginSummary {
    /// Total `NETMSG_INPUTTIMING` messages ever observed (unbounded, exact).
    pub count: u64,
    /// How many had `time_left_ms < 0` (the input arrived too late) — unbounded, exact.
    pub late_count: u64,
    /// `late_count as f64 / count as f64`, or `0.0` if `count == 0`.
    pub late_fraction: f64,
    /// Mean `time_left_ms`, over the same unbounded/exact count.
    pub mean_ms: f64,
    /// `min`/`p50`/`p90`/`p99`/`max`, computed over the bounded recent window
    /// ([`MARGIN_SAMPLE_WINDOW`]) — `None` if no samples have been recorded at all.
    pub min_ms: Option<i32>,
    pub p50_ms: Option<i32>,
    pub p90_ms: Option<i32>,
    pub p99_ms: Option<i32>,
    pub max_ms: Option<i32>,
}

/// Running/windowed statistics over every `time_left_ms` this session has observed via
/// `NETMSG_INPUTTIMING` — the raw material for task acceptance criterion h's margin distribution.
#[derive(Debug, Clone, Default)]
pub struct MarginStats {
    count: u64,
    late_count: u64,
    sum_ms: i64,
    recent: VecDeque<i32>,
}

impl MarginStats {
    fn record(&mut self, time_left_ms: i32) {
        self.count += 1;
        if time_left_ms < 0 {
            self.late_count += 1;
        }
        self.sum_ms += i64::from(time_left_ms);
        self.recent.push_back(time_left_ms);
        if self.recent.len() > MARGIN_SAMPLE_WINDOW {
            self.recent.pop_front();
        }
    }

    /// A point-in-time [`MarginSummary`]. `O(n log n)` in the recent-window size (sorts a copy) —
    /// meant to be called occasionally for reporting, not on every sample.
    pub fn summary(&self) -> MarginSummary {
        let mut sorted: Vec<i32> = self.recent.iter().copied().collect();
        sorted.sort_unstable();
        let percentile = |p: f64| -> Option<i32> {
            if sorted.is_empty() {
                return None;
            }
            let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
            sorted.get(idx).copied()
        };
        MarginSummary {
            count: self.count,
            late_count: self.late_count,
            late_fraction: if self.count == 0 {
                0.0
            } else {
                self.late_count as f64 / self.count as f64
            },
            mean_ms: if self.count == 0 {
                0.0
            } else {
                self.sum_ms as f64 / self.count as f64
            },
            min_ms: sorted.first().copied(),
            p50_ms: percentile(0.50),
            p90_ms: percentile(0.90),
            p99_ms: percentile(0.99),
            max_ms: sorted.last().copied(),
        }
    }
}

/// `CClient::MaxLatencyTicks` (`client.cpp:5429-5432`).
pub fn max_latency_ticks(prediction_margin_ms: i32) -> i32 {
    GAME_TICK_SPEED + (prediction_margin_ms * GAME_TICK_SPEED) / 1000
}

/// The input-timing state for one connection — see the module docs for exactly what this is (and
/// is not) a port of.
#[derive(Debug, Clone)]
pub struct InputTiming {
    prediction_margin_ms: i32,
    predicted_time: Option<SmoothTime>,
    received_snapshots: u32,
    /// The most recently *received* snapshot's tick (`client.cpp:2950`'s comparison target — see
    /// the module docs for why this, not `SNAP_PREV`, is used here).
    latest_snapshot_tick: i32,
    /// `m_aPredTick[Conn]` — `0` until initialized (mirrors `SendInput`'s own `<= 0` guard,
    /// `client.cpp:354-355`).
    pred_tick: i32,
    history: VecDeque<HistoryEntry>,
    margin_stats: MarginStats,
}

impl InputTiming {
    pub fn new(prediction_margin_ms: i32) -> Self {
        InputTiming {
            prediction_margin_ms,
            predicted_time: None,
            received_snapshots: 0,
            latest_snapshot_tick: 0,
            pred_tick: 0,
            history: VecDeque::with_capacity(INPUT_HISTORY_LEN),
            margin_stats: MarginStats::default(),
        }
    }

    /// Resets every piece of per-connection state this module owns — call this exactly where
    /// [`ddai_net::assembly::SnapAssembler::reset`] is called (every `NETMSG_ENTERGAME`, first
    /// entry and every later map change): a fresh map means a fresh server tick counter, so the
    /// two-snapshot bootstrap must run again from scratch (mirrors `CClient::OnEnterGame`
    /// zeroing `m_aPredTick`/`m_aReceivedSnapshots`/re-`Init`ing `m_PredictedTime`,
    /// `client.cpp:494-504`). `margin_stats` is deliberately **not** reset — the margin
    /// distribution (acceptance criterion h) is a whole-session metric, not a per-map one.
    pub fn reset(&mut self) {
        self.predicted_time = None;
        self.received_snapshots = 0;
        self.latest_snapshot_tick = 0;
        self.pred_tick = 0;
        self.history.clear();
    }

    /// The predicted tick the next `NETMSG_INPUT` should carry, or `0` before the two-snapshot
    /// bootstrap has happened (`SendInput`'s own guard).
    pub fn pred_tick(&self) -> i32 {
        self.pred_tick
    }

    pub fn margin_summary(&self) -> MarginSummary {
        self.margin_stats.summary()
    }

    /// Task 4.1: how long until [`InputTiming::advance`] next fires (the next `NETMSG_INPUT` goes
    /// out), at `now_ns`; `None` before the two-snapshot bootstrap. `advance` fires once the
    /// predicted clock reaches tick `pred_tick` (`floor(pred_now) + 1 > pred_tick`), so the wait is
    /// `(pred_tick - pred_now_in_ticks) * 20 ms`, never negative. The live bot uses it to know whether
    /// its decision will still make the next input or the one after.
    pub fn next_input_in_ns(&self, now_ns: i64) -> Option<i64> {
        let pred_now = self.predicted_time.as_ref()?.get(now_ns);
        let tick_ns = crate::smooth_time::TIME_FREQ / i64::from(GAME_TICK_SPEED);
        let reaches = i64::from(self.pred_tick) * tick_ns;
        Some((reaches - pred_now).max(0))
    }

    /// Call once for every [`ddai_net::assembly::Event::Snapshot`] the session accepts (in
    /// order) — `client.cpp:2304-2320`'s `m_aReceivedSnapshots[Conn]++` plus the two-snapshot
    /// bootstrap.
    pub fn on_snapshot(&mut self, tick: i32, now_ns: i64) {
        self.received_snapshots += 1;
        self.latest_snapshot_tick = tick;
        if self.received_snapshots == 2 {
            let target = i64::from(tick) * crate::smooth_time::TIME_FREQ / i64::from(GAME_TICK_SPEED);
            let mut predicted = SmoothTime::init(target, now_ns);
            predicted.set_adjust_speed(AdjustDirection::Up, 1000.0);
            predicted.update_margin(i64::from(self.prediction_margin_ms) * crate::smooth_time::TIME_FREQ / 1000);
            self.predicted_time = Some(predicted);
        }
    }

    /// Call periodically (the driver's own poll cadence — DDNet calls this every rendered frame,
    /// we call it every driver tick, e.g. every 5-20ms; a faster cadence only makes the predicted
    /// tick advance *sooner* within its 20ms tick window, it cannot advance further than the
    /// wall clock allows either way). Returns `Some(tick)` exactly when a new predicted tick was
    /// reached and a fresh `NETMSG_INPUT` should be built and sent for it — mirrors
    /// `client.cpp:2939-2963`.
    pub fn advance(&mut self, now_ns: i64) -> Option<i32> {
        let pred_now = self.predicted_time.as_ref()?.get(now_ns);
        let prev_pred_tick = pred_now * i64::from(GAME_TICK_SPEED) / crate::smooth_time::TIME_FREQ;
        let candidate = prev_pred_tick + 1;

        // Sanity reset: predicted tick has drifted too far from where the server actually is —
        // `client.cpp:2950-2954`. See the module docs for `latest_snapshot_tick` vs `SNAP_PREV`.
        //
        // Review finding F8: `CSmoothTime::Init` (`client.cpp:2953`) is called *alone* here — it
        // already resets the adjust speeds to their own defaults (0.3/0.3) and the margin to 0
        // (`SmoothTime::init`'s own doc comment/tests pin this down), so re-applying a margin and
        // a huge adjust speed right afterwards (as this used to) was actively re-undoing part of
        // the reset it had just done, not "restoring" anything real `CSmoothTime::Init` does.
        // And crucially, `client.cpp:2950-2963` never returns early after this reset: it falls
        // straight through to the *same* `if(NewPredTick > m_aPredTick[...])` check below, using
        // `NewPredTick`/`PrevPredTick` computed *before* the reset — so this must fall through to
        // the identical check on `candidate` too, not bail out of this call entirely.
        let max_latency = u64::try_from(max_latency_ticks(self.prediction_margin_ms)).unwrap_or(0);
        if candidate.abs_diff(i64::from(self.latest_snapshot_tick)) > max_latency {
            let cur_tick_start =
                i64::from(self.latest_snapshot_tick) * crate::smooth_time::TIME_FREQ / i64::from(GAME_TICK_SPEED);
            let reset_target = cur_tick_start + 2 * crate::smooth_time::TIME_FREQ / i64::from(GAME_TICK_SPEED);
            let predicted_time = self.predicted_time.as_mut().expect("checked Some via as_ref() above");
            *predicted_time = SmoothTime::init(reset_target, now_ns);
        }

        // Review finding F8 (round 2 — a real regression the round-1 fix introduced):
        // `client.cpp:3129` calls `m_PredictedTime.UpdateMargin(PredictionMargin() * ...)`
        // *unconditionally, every frame* inside `CClient::Update()` — not just once at the
        // two-snapshot bootstrap (`on_snapshot`, above). Without this, `SmoothTime::init`'s
        // reset (which zeroes the margin — by design, matching `CSmoothTime::Init` exactly, see
        // the comment above) would leave the margin at 0 *forever* after any reset, since nothing
        // else ever restores it: every `NETMSG_INPUT` sent from then on would be built `margin`
        // ticks later than intended, silently degrading the "send input a bit early" cushion
        // (confirmed live with a timing simulation: after a stall-induced reset, the late-input
        // fraction rose from ~0.6% to ~6.8% and the mean margin dropped from ~14ms to ~2ms, and
        // stayed that way until the next `ENTERGAME`/`reset()`). Placed *after* the reset above
        // (not before): the whole point is that even a reset earlier in *this exact* call must
        // not leave a stale/zeroed margin behind for this call's own `recorded_predicted_time`
        // read below, let alone every call after it.
        if let Some(predicted_time) = self.predicted_time.as_mut() {
            let margin_ns = i64::from(self.prediction_margin_ms) * crate::smooth_time::TIME_FREQ / 1000;
            predicted_time.update_margin(margin_ns);
        }

        let new_pred_tick = i32::try_from(candidate).unwrap_or(self.pred_tick);
        if new_pred_tick > self.pred_tick {
            self.pred_tick = new_pred_tick;
            let recorded_predicted_time = self.predicted_time.as_ref().expect("still Some").get(now_ns);
            if self.history.len() == INPUT_HISTORY_LEN {
                self.history.pop_front();
            }
            self.history.push_back(HistoryEntry {
                tick: new_pred_tick,
                predicted_time_ns: recorded_predicted_time,
                sent_at_ns: now_ns,
            });
            Some(new_pred_tick)
        } else {
            None
        }
    }

    /// Feeds one `NETMSG_INPUTTIMING` (`pred_tick`, `time_left_ms`) — `client.cpp:2084-2108`. Every
    /// call records a [`MarginStats`] sample regardless of whether a matching history entry was
    /// found (acceptance criterion h asks for the margin distribution of every `INPUTTIMING`
    /// message observed, not just the ones this session can still correlate).
    pub fn on_input_timing(&mut self, pred_tick: i32, time_left_ms: i32, now_ns: i64) {
        self.margin_stats.record(time_left_ms);

        let target = self.history.iter().find(|e| e.tick == pred_tick).map(|entry| {
            let raw = entry.predicted_time_ns + (now_ns - entry.sent_at_ns);
            raw - (i64::from(time_left_ms) * crate::smooth_time::TIME_FREQ) / 1000
        });

        if let (Some(target), Some(predicted_time)) = (target, self.predicted_time.as_mut()) {
            predicted_time.update(target, time_left_ms, AdjustDirection::Up, now_ns);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smooth_time::TIME_FREQ;

    #[test]
    fn pred_tick_is_zero_and_advance_is_none_before_two_snapshots() {
        let mut timing = InputTiming::new(DEFAULT_PREDICTION_MARGIN_MS);
        assert_eq!(timing.pred_tick(), 0);
        assert_eq!(timing.advance(0), None);
        timing.on_snapshot(100, 0);
        assert_eq!(timing.advance(0), None, "still only one snapshot");
    }

    #[test]
    fn second_snapshot_bootstraps_predicted_time_and_advance_starts_producing_ticks() {
        let mut timing = InputTiming::new(DEFAULT_PREDICTION_MARGIN_MS);
        timing.on_snapshot(100, 0);
        timing.on_snapshot(101, 0);
        // Immediately at t=0, pred_now == target == 101 ticks worth of ns, so the first `advance`
        // already sees a new predicted tick (101/50s worth of ticks + 1).
        let first = timing.advance(0);
        assert!(first.is_some());
        assert_eq!(first.unwrap(), timing.pred_tick());
    }

    #[test]
    fn advance_only_returns_some_once_per_new_tick_boundary() {
        // Note: the bootstrap's own `prediction_margin_ms` (`crate::smooth_time`'s `margin` field,
        // added unconditionally by `SmoothTime::get`) means predicted ticks are not spaced at
        // exactly human-round wall-clock offsets from `now == 0` — this test therefore checks the
        // *invariants* (never skips, never double-fires at the same `now`, eventually advances)
        // rather than pinning an exact tick boundary, which would be fragile to the margin's
        // exact value.
        let mut timing = InputTiming::new(DEFAULT_PREDICTION_MARGIN_MS);
        timing.on_snapshot(0, 0);
        timing.on_snapshot(0, 0);
        let first = timing.advance(0).expect("bootstrap tick");

        // Calling again at the exact same instant must not produce a second new tick.
        assert_eq!(timing.advance(0), None);

        let tick_ns = TIME_FREQ / i64::from(GAME_TICK_SPEED);
        let mut last = first;
        for step in 1..=5 {
            let now = step * tick_ns;
            if let Some(tick) = timing.advance(now) {
                assert_eq!(tick, last + 1, "pred_tick must advance by exactly one tick at a time");
                last = tick;
            }
            assert_eq!(timing.advance(now), None, "repeating the same `now` must never re-fire");
        }
        assert!(
            last > first,
            "wall-clock time passing must eventually advance pred_tick"
        );
    }

    /// Task 4.1: `next_input_in_ns` says exactly when `advance` fires next.
    #[test]
    fn next_input_in_matches_when_advance_actually_fires() {
        let mut timing = InputTiming::new(DEFAULT_PREDICTION_MARGIN_MS);
        assert_eq!(timing.next_input_in_ns(0), None, "before the bootstrap");
        timing.on_snapshot(100, 0);
        timing.on_snapshot(101, 0);
        timing.advance(0).expect("the bootstrap tick");
        let wait = timing.next_input_in_ns(0).expect("bootstrapped");
        let tick_ns = TIME_FREQ / i64::from(GAME_TICK_SPEED);
        assert!((0..=tick_ns * 3).contains(&wait), "{wait}");
        // Just before the predicted time: nothing is sent; at it: the next tick is sent.
        if wait > 1_000_000 {
            assert_eq!(timing.advance(wait - 1_000_000), None);
        }
        assert!(timing.advance(wait + 1_000_000).is_some());
        // Right after a send the next one is up to a tick away.
        let now = wait + 1_000_000;
        let again = timing.next_input_in_ns(now).unwrap();
        assert!(again <= tick_ns, "{again}");
    }

    #[test]
    fn on_input_timing_records_margin_sample_even_without_a_matching_history_entry() {
        let mut timing = InputTiming::new(DEFAULT_PREDICTION_MARGIN_MS);
        timing.on_input_timing(12345, -7, 0);
        let summary = timing.margin_summary();
        assert_eq!(summary.count, 1);
        assert_eq!(summary.late_count, 1);
        assert_eq!(summary.min_ms, Some(-7));
    }

    #[test]
    fn on_input_timing_with_matching_history_updates_predicted_time_margin_direction() {
        let mut timing = InputTiming::new(DEFAULT_PREDICTION_MARGIN_MS);
        timing.on_snapshot(0, 0);
        timing.on_snapshot(0, 0);
        let tick = timing.advance(0).expect("bootstrap tick");
        let before = timing.predicted_time.unwrap().get(0);
        timing.on_input_timing(tick, -20, 1_000_000); // "late" feedback
        let after = timing.predicted_time.unwrap().get(1_000_000);
        assert_ne!(
            before, after,
            "predicted_time should have moved in response to feedback"
        );
    }

    #[test]
    fn reset_clears_bootstrap_state_but_not_margin_stats() {
        let mut timing = InputTiming::new(DEFAULT_PREDICTION_MARGIN_MS);
        timing.on_snapshot(0, 0);
        timing.on_snapshot(0, 0);
        timing.advance(0);
        timing.on_input_timing(1, 5, 0);
        assert_ne!(timing.pred_tick(), 0);
        assert_eq!(timing.margin_summary().count, 1);

        timing.reset();
        assert_eq!(timing.pred_tick(), 0);
        assert_eq!(timing.advance(0), None);
        assert_eq!(
            timing.margin_summary().count,
            1,
            "margin distribution survives a map-change reset"
        );
    }

    #[test]
    fn drift_too_far_from_latest_snapshot_resets_predicted_time_but_still_falls_through() {
        let mut timing = InputTiming::new(DEFAULT_PREDICTION_MARGIN_MS);
        timing.on_snapshot(0, 0);
        timing.on_snapshot(0, 0);
        // Jump the wall clock far ahead without any further snapshots arriving: predicted tick
        // would drift arbitrarily far from `latest_snapshot_tick` (still 0).
        let far_future = 100 * TIME_FREQ; // 100 seconds

        // Review finding F8 (`client.cpp:2950-2963`): the reset never returns early — it falls
        // straight through to the *same* tick-advance check, using the candidate tick computed
        // *before* the reset (exactly like the real client's `NewPredTick`, computed before
        // `CSmoothTime::Init` runs) — so this call must still produce a tick, not silently
        // swallow it into `None`.
        let result = timing.advance(far_future);
        assert!(
            result.is_some(),
            "must fall through to the ordinary tick-advance check, not bail out to None"
        );
    }

    /// Review finding F8: the reset must call *only* `SmoothTime::init` (which already resets the
    /// margin to 0 and the adjust speeds to their own defaults, 0.3/0.3 — see that function's own
    /// doc comment) — not re-apply `update_margin`/`set_adjust_speed` right afterwards, which would
    /// silently undo part of the reset it had just done. Checked directly on the freshly-reset
    /// `SmoothTime`'s own `get()`, independent of `pred_tick`'s one-way-ratchet behaviour (which,
    /// exactly like the real client, does not itself reveal this — the previous test covers that
    /// half separately).
    #[test]
    fn drift_reset_still_leaves_the_prediction_margin_present() {
        let margin_ms = 40;
        let mut timing = InputTiming::new(margin_ms);
        timing.on_snapshot(0, 0);
        timing.on_snapshot(0, 0);
        let far_future = 100 * TIME_FREQ;
        let _ = timing.advance(far_future); // triggers the reset branch

        let anchored = timing.predicted_time.as_ref().unwrap().get(far_future);
        // `latest_snapshot_tick` is still 0, so `cur_tick_start == 0` — matches `advance`'s own
        // `reset_target` computation.
        let expected_target = 2 * TIME_FREQ / i64::from(GAME_TICK_SPEED);
        let margin_ns = i64::from(margin_ms) * TIME_FREQ / 1000;
        // Review finding F8 (round 2): `SmoothTime::init` itself still zeroes the margin (that is
        // its own, separately-tested contract, `smooth_time.rs`) — but by the time `advance()` as
        // a whole returns, the margin must already be back (`client.cpp:3129`'s unconditional
        // per-frame `UpdateMargin`, mirrored right after the reset here), not left at that bare
        // zero. Round 1 asserted the opposite of this (a `margin_ms`-too-high failure) — that
        // assertion described a real bug this round fixes, not the intended final behaviour.
        assert_eq!(
            anchored,
            expected_target + margin_ns,
            "get() right after advance() must include the refreshed margin ({margin_ms}ms), not \
             the bare `SmoothTime::init` zero"
        );
    }

    /// Review finding F8 (round 2), timingsim-style: the *effect* margin has on the predicted
    /// clock — `with.get(now) - without.get(now) == margin_ns`, for two otherwise-identical
    /// `InputTiming`s that only differ in `prediction_margin_ms` — must survive a drift-triggered
    /// reset and every call afterwards, not just hold once at the reset instant. Before this
    /// round's fix, the margin stayed zeroed forever after any reset (matching the reviewer's own
    /// `timingsim` probe: a 1.3s stall forcing a reset raised the late-input fraction from ~0.6%
    /// to ~6.8% and dropped the mean margin from ~14ms to ~2ms, until the next `ENTERGAME`).
    #[test]
    fn margin_effect_on_predicted_time_survives_a_drift_reset() {
        let margin_ms = 40i32;
        let margin_ns = i64::from(margin_ms) * TIME_FREQ / 1000;

        let mut with_margin = InputTiming::new(margin_ms);
        let mut without_margin = InputTiming::new(0);
        for timing in [&mut with_margin, &mut without_margin] {
            timing.on_snapshot(0, 0);
            timing.on_snapshot(0, 0);
        }

        // A long stall (no further snapshots) forces the drift-reset in `advance()` for both.
        let far_future = 100 * TIME_FREQ;
        let _ = with_margin.advance(far_future);
        let _ = without_margin.advance(far_future);

        let tick_ns = TIME_FREQ / i64::from(GAME_TICK_SPEED);
        for extra_ticks in [0i64, 1, 5, 50, 500] {
            let now = far_future + extra_ticks * tick_ns;
            let with = with_margin.predicted_time.as_ref().unwrap().get(now);
            let without = without_margin.predicted_time.as_ref().unwrap().get(now);
            assert_eq!(
                with - without,
                margin_ns,
                "the margin's effect on the predicted clock must persist after the reset, not \
                 just at the reset instant (extra_ticks={extra_ticks})"
            );
        }
    }

    #[test]
    fn margin_summary_percentiles_are_monotonic_and_within_range() {
        let mut timing = InputTiming::new(DEFAULT_PREDICTION_MARGIN_MS);
        for ms in -50..50 {
            timing.on_input_timing(0, ms, 0);
        }
        let s = timing.margin_summary();
        assert_eq!(s.count, 100);
        assert_eq!(s.late_count, 50);
        assert!((s.late_fraction - 0.5).abs() < 1e-9);
        let (min, p50, p90, p99, max) = (
            s.min_ms.unwrap(),
            s.p50_ms.unwrap(),
            s.p90_ms.unwrap(),
            s.p99_ms.unwrap(),
            s.max_ms.unwrap(),
        );
        assert!(min <= p50 && p50 <= p90 && p90 <= p99 && p99 <= max);
        assert_eq!(min, -50);
        assert_eq!(max, 49);
    }

    #[test]
    fn margin_window_is_bounded_even_over_many_samples() {
        let mut timing = InputTiming::new(DEFAULT_PREDICTION_MARGIN_MS);
        for i in 0..(MARGIN_SAMPLE_WINDOW * 3) {
            timing.on_input_timing(0, i as i32, 0);
        }
        assert_eq!(timing.margin_stats.recent.len(), MARGIN_SAMPLE_WINDOW);
        // Exact/unbounded counters must still reflect every sample, not just the window.
        assert_eq!(timing.margin_summary().count, (MARGIN_SAMPLE_WINDOW * 3) as u64);
    }
}
