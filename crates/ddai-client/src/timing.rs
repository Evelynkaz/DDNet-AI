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

/// Task 4.1b (D-063, reworked in round 1 of its review): the adaptive prediction margin's bounds and
/// rule — see [`MarginController`].
pub const ADAPTIVE_MARGIN_MIN_MS: i32 = 3;
pub const ADAPTIVE_MARGIN_MAX_MS: i32 = 20;
/// The `time_left` the controller aims the rolling p1 at.
pub const ADAPTIVE_SAFETY_MS: i32 = 2;
/// The rolling window of `time_left` samples (about 5 s at 50 inputs/s).
pub const ADAPTIVE_WINDOW_NS: i64 = 5_000_000_000;
/// Fewest samples in the window before the margin may be lowered (about 3 s at 50 Hz; after every
/// change the window starts over, because the old samples describe the old margin).
pub const ADAPTIVE_MIN_SAMPLES_TO_LOWER: usize = 150;
/// Fewest samples before a p1 below the safety level raises the margin (about 2 s; with fewer the
/// "p1" is the minimum, and one slow sample would move the margin).
pub const ADAPTIVE_MIN_SAMPLES_TO_RAISE: usize = 100;
/// The margin is raised by this much when the window holds [`ADAPTIVE_LATES_TO_RAISE`] late inputs.
pub const ADAPTIVE_LATE_STEP_MS: i32 = 2;
/// Late inputs (not stalls) in the current window that raise the margin. **One late input never
/// does**: on a shared host a scheduler pause makes an input late every ~10 s whatever the margin, and
/// raising on each of them ratcheted the margin to its cap (review F1).
pub const ADAPTIVE_LATES_TO_RAISE: usize = 2;
/// The margin is lowered by at most 1 ms, and not more often than this.
pub const ADAPTIVE_LOWER_EVERY_NS: i64 = 3_000_000_000;
/// Lowering is allowed only after this long without a late input (not a stall): the hold after a
/// raise decays instead of being a fixed 30 s that a late input every 13 s could never outlast.
pub const ADAPTIVE_QUIET_BEFORE_LOWER_NS: i64 = 10_000_000_000;
/// Lowering also needs hysteresis: p1 must be at least `SAFETY + ADAPTIVE_LOWER_HYSTERESIS_MS` above
/// zero, so that after the 1 ms step p1 is still above the safety level instead of sitting exactly on
/// the edge where the next two lates raise the margin again (review round 2, F7: 135 changes in 10 min).
pub const ADAPTIVE_LOWER_HYSTERESIS_MS: i32 = 2;
/// The stall watch looks at windows of this length (one warning per minute at most).
pub const STALL_WATCH_WINDOW_NS: i64 = 60_000_000_000;
/// Fewest inputs in a window before the stall watch judges it (about 10 s at 50 Hz).
pub const STALL_WATCH_MIN_SAMPLES: u64 = 500;
/// Stalls above this share of a window's inputs are reported.
pub const STALL_WATCH_RATE: f64 = 0.005;
/// A late input is a **stall** when it was sent more than `MAX - SAFETY` ms later than the margin in
/// force allowed (`margin - time_left`, the implied send delay): no margin in range would have saved
/// it, so it neither raises the margin nor enters the window. Judged by the implied delay rather than
/// by raw lateness, because a late input at margin `m` means the send stalled by more than `m` ms.
pub const ADAPTIVE_STALL_DELAY_MS: i32 = ADAPTIVE_MARGIN_MAX_MS - ADAPTIVE_SAFETY_MS;

/// The adaptive prediction margin (task 4.1b; D-023's `INPUTTIMING` feedback is the sensor): keeps
/// `margin` just large enough that inputs reach the server `ADAPTIVE_SAFETY_MS` before their tick, so
/// the live bot gets the most decision time per snapshot the connection allows (the first input slot
/// after a snapshot comes `20 ms - margin` later) without late inputs.
///
/// * **Rolling p1** of `time_left` over [`ADAPTIVE_WINDOW_NS`] (instead of the session-wide
///   [`MarginStats`]); the window starts over after every change.
/// * **Raise only on a windowed signal:** the window holds [`ADAPTIVE_LATES_TO_RAISE`] late inputs
///   (by [`ADAPTIVE_LATE_STEP_MS`]), or p1 is below the safety level over at least
///   [`ADAPTIVE_MIN_SAMPLES_TO_RAISE`] samples (by the shortfall).
/// * **Stalls** (see [`ADAPTIVE_STALL_DELAY_MS`]) are ignored.
/// * **Samples of inputs sent before the last change are ignored** (`stale`): they describe the old
///   margin, and counting them wound the margin up while the change was still in flight.
/// * **Lower slowly:** by at most 1 ms, not more often than [`ADAPTIVE_LOWER_EVERY_NS`], only from a
///   full window ([`ADAPTIVE_MIN_SAMPLES_TO_LOWER`]) with p1 at least [`ADAPTIVE_LOWER_HYSTERESIS_MS`]
///   above the safety level, and only after
///   [`ADAPTIVE_QUIET_BEFORE_LOWER_NS`] without a late input.
/// * **Reset** to the initial margin on a new map / connection ([`MarginController::reset`]).
#[derive(Debug, Clone)]
pub struct MarginController {
    initial_ms: i32,
    margin_ms: i32,
    window: VecDeque<(i64, i32)>,
    last_change_ns: i64,
    last_late_ns: i64,
    last_sample_ns: i64,
    first_sample_ns: Option<i64>,
    changes: u32,
    /// When the last few changes happened (for [`MarginController::changes_within_ms`]).
    change_times: VecDeque<i64>,
    /// `(ms since the first sample, new margin)` of every change (capped), for the report.
    trajectory: Vec<(u64, i32)>,
    /// Time (ms) spent at each margin value `0..=MAX`, sample to sample.
    time_at_margin_ms: [i64; ADAPTIVE_MARGIN_MAX_MS as usize + 1],
}

impl MarginController {
    pub fn new(initial_ms: i32) -> Self {
        let initial_ms = initial_ms.clamp(ADAPTIVE_MARGIN_MIN_MS, ADAPTIVE_MARGIN_MAX_MS);
        MarginController {
            initial_ms,
            margin_ms: initial_ms,
            window: VecDeque::with_capacity(512),
            last_change_ns: i64::MIN / 2,
            last_late_ns: i64::MIN / 2,
            last_sample_ns: 0,
            first_sample_ns: None,
            changes: 0,
            change_times: VecDeque::new(),
            trajectory: Vec::new(),
            time_at_margin_ms: [0; ADAPTIVE_MARGIN_MAX_MS as usize + 1],
        }
    }

    pub fn margin_ms(&self) -> i32 {
        self.margin_ms
    }

    /// How many times the margin has been changed (sessions-wide, survives `reset`).
    pub fn changes(&self) -> u32 {
        self.changes
    }

    /// `(ms since the first sample, new margin)` of every change (the first 4096).
    pub fn trajectory(&self) -> &[(u64, i32)] {
        &self.trajectory
    }

    /// Time (ms) spent at each margin value (index = margin in ms).
    pub fn time_at_margin_ms(&self) -> &[i64] {
        &self.time_at_margin_ms
    }

    /// How many times the margin changed in the `ms` before the last sample ("has it settled").
    pub fn changes_within_ms(&self, ms: i64) -> u32 {
        let from = self.last_sample_ns - ms * 1_000_000;
        self.change_times.iter().filter(|&&t| t >= from).count() as u32
    }

    /// How long (ms) the margin has been unchanged as of the last sample; since the controller
    /// started if it never changed.
    pub fn stable_for_ms(&self) -> i64 {
        let since = if self.changes == 0 || self.last_change_ns < 0 {
            0
        } else {
            self.last_change_ns
        };
        (self.last_sample_ns - since).max(0) / 1_000_000
    }

    /// Back to the initial margin with an empty window (new map or connection).
    pub fn reset(&mut self) {
        self.margin_ms = self.initial_ms;
        self.window.clear();
        self.last_change_ns = i64::MIN / 2;
        self.last_late_ns = i64::MIN / 2;
    }

    /// The rolling p1 of `time_left` (ms); `None` for an empty window.
    pub fn p1_ms(&self) -> Option<i32> {
        if self.window.is_empty() {
            return None;
        }
        let mut v: Vec<i32> = self.window.iter().map(|&(_, t)| t).collect();
        v.sort_unstable();
        Some(v[((v.len() - 1) as f64 * 0.01).round() as usize])
    }

    /// Whether `time_left_ms` would be a stall at the margin now in force (see
    /// [`ADAPTIVE_STALL_DELAY_MS`]).
    pub fn is_stall(&self, time_left_ms: i32) -> bool {
        time_left_ms < 0 && self.margin_ms - time_left_ms > ADAPTIVE_STALL_DELAY_MS
    }

    /// One `NETMSG_INPUTTIMING` sample at `now_ns`; `stale` is true for an input sent before the last
    /// margin change. Returns the new margin if it changed.
    pub fn on_sample(&mut self, time_left_ms: i32, now_ns: i64, stale: bool) -> Option<i32> {
        let had_sample = self.first_sample_ns.is_some();
        let first = *self.first_sample_ns.get_or_insert(now_ns);
        if had_sample {
            let dt = (now_ns - self.last_sample_ns).clamp(0, 1_000_000_000) / 1_000_000;
            self.time_at_margin_ms[self.margin_ms as usize] += dt;
        }
        self.last_sample_ns = now_ns;
        let stall = self.is_stall(time_left_ms);
        if time_left_ms < 0 && !stall {
            self.last_late_ns = now_ns;
        }
        if stale || stall {
            return None;
        }
        self.window.push_back((now_ns, time_left_ms));
        while self
            .window
            .front()
            .is_some_and(|&(t, _)| now_ns.saturating_sub(t) > ADAPTIVE_WINDOW_NS)
        {
            self.window.pop_front();
        }
        let old = self.margin_ms;
        let mut new = old;
        let lates = self.window.iter().filter(|&&(_, t)| t < 0).count();
        if lates >= ADAPTIVE_LATES_TO_RAISE {
            new = old + ADAPTIVE_LATE_STEP_MS;
        } else if self.window.len() >= ADAPTIVE_MIN_SAMPLES_TO_RAISE {
            let delta = self.p1_ms().unwrap_or(time_left_ms) - ADAPTIVE_SAFETY_MS;
            if delta < 0 {
                new = old - delta;
            } else if delta >= ADAPTIVE_LOWER_HYSTERESIS_MS
                && self.window.len() >= ADAPTIVE_MIN_SAMPLES_TO_LOWER
                && now_ns.saturating_sub(self.last_change_ns) >= ADAPTIVE_LOWER_EVERY_NS
                && now_ns.saturating_sub(self.last_late_ns) >= ADAPTIVE_QUIET_BEFORE_LOWER_NS
            {
                new = old - 1;
            }
        }
        let new = new.clamp(ADAPTIVE_MARGIN_MIN_MS, ADAPTIVE_MARGIN_MAX_MS);
        if new == old {
            return None;
        }
        self.margin_ms = new;
        self.window.clear();
        self.last_change_ns = now_ns;
        self.changes += 1;
        self.change_times.push_back(now_ns);
        if self.change_times.len() > 64 {
            self.change_times.pop_front();
        }
        if self.trajectory.len() < 4096 {
            self.trajectory
                .push((((now_ns - first).max(0) / 1_000_000) as u64, new));
        }
        Some(new)
    }
}

/// Reports a link whose stalls (see [`ADAPTIVE_STALL_DELAY_MS`]) stay above [`STALL_WATCH_RATE`] of its
/// inputs: no margin in range can absorb them, so this is what the operator should hear about (a link
/// with jitter beyond 18 ms, a starved host). Judged per window of [`STALL_WATCH_WINDOW_NS`], so it
/// speaks at most once per minute, and it carries only counts and rates — no names, no addresses.
#[derive(Debug, Clone, Default)]
pub struct StallWatch {
    window_start_ns: Option<i64>,
    count: u64,
    stalls: u64,
}

impl StallWatch {
    /// One `NETMSG_INPUTTIMING` at `now_ns`. Returns `Some((stalls, count))` when a window just ended
    /// above the threshold.
    pub fn on_sample(&mut self, now_ns: i64, stall: bool) -> Option<(u64, u64)> {
        let start = *self.window_start_ns.get_or_insert(now_ns);
        self.count += 1;
        self.stalls += u64::from(stall);
        if now_ns - start < STALL_WATCH_WINDOW_NS {
            return None;
        }
        let verdict = (self.count >= STALL_WATCH_MIN_SAMPLES
            && self.stalls as f64 > STALL_WATCH_RATE * self.count as f64)
            .then_some((self.stalls, self.count));
        self.window_start_ns = Some(now_ns);
        self.count = 0;
        self.stalls = 0;
        verdict
    }
}

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
#[derive(Debug, Clone, PartialEq)]
pub struct MarginSummary {
    /// Total `NETMSG_INPUTTIMING` messages ever observed (unbounded, exact).
    pub count: u64,
    /// How many had `time_left_ms < 0` (the input arrived too late) — unbounded, exact.
    pub late_count: u64,
    /// Of those, how many were **stalls**: sent more than [`ADAPTIVE_STALL_DELAY_MS`] later than the
    /// margin in force allowed (`margin - time_left`; a paused thread, a route change) — no margin in
    /// the adaptive range could have absorbed them.
    pub stall_count: u64,
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
    /// Task 4.1b: the prediction margin in force when this summary was taken (the fixed value, or
    /// the adaptive controller's current one).
    pub margin_ms: i32,
    /// How many times the adaptive controller changed the margin (0 for a fixed margin).
    pub margin_changes: u32,
    /// Whether the margin is adaptive.
    pub adaptive: bool,
    /// How long (ms) the adaptive margin has been unchanged as of the last `NETMSG_INPUTTIMING`
    /// (0 for a fixed margin): "the margin settled" means this is long.
    pub margin_stable_ms: i64,
    /// How many times the adaptive margin changed in the last 30 s ("settled" = few).
    pub margin_changes_last_30s: u32,
    /// `(ms since the first `NETMSG_INPUTTIMING`, new margin)` of every change of the adaptive margin
    /// (the first 4096): the margin's trajectory.
    pub margin_trajectory: Vec<(u64, i32)>,
    /// Time (ms) the adaptive margin spent at each value (index = margin in ms; empty for a fixed
    /// margin): `time_at_margin_ms[20]` is the time pinned at the cap.
    pub time_at_margin_ms: Vec<i64>,
    /// Task 4.1b (driver): tagged decisions that were replaced by a newer one before their tick
    /// came, so they never went out ([`ClientEvent::MarginSummary`] carries it from the driver).
    pub superseded_decisions: u64,
    /// Task 4.1b (driver): decisions adopted more than 2 ticks after their tick whose fire press was
    /// dropped (the hammer veto only covers the intended tick plus 2).
    pub late_presses_dropped: u64,
    /// Task 3.11: how late the driver actually sent each `NETMSG_INPUT` after the predicted clock made it due, microseconds
    /// (`count`, then percentiles over the recent window): the loop's wake-up granularity. `None` until an input was sent.
    pub send_lag_us: Option<SendLagSummary>,
}

/// Task 3.11: the distribution of [`MarginSummary::send_lag_us`] (send time minus due time).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SendLagSummary {
    pub count: u64,
    pub p50_us: u32,
    pub p90_us: u32,
    pub p99_us: u32,
    pub max_us: u32,
}

/// Running/windowed statistics over every `time_left_ms` this session has observed via
/// `NETMSG_INPUTTIMING` — the raw material for task acceptance criterion h's margin distribution.
#[derive(Debug, Clone, Default)]
pub struct MarginStats {
    count: u64,
    late_count: u64,
    stall_count: u64,
    sum_ms: i64,
    recent: VecDeque<i32>,
}

impl MarginStats {
    fn record(&mut self, time_left_ms: i32, stall: bool) {
        self.count += 1;
        if time_left_ms < 0 {
            self.late_count += 1;
        }
        if stall {
            self.stall_count += 1;
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
        self.summary_with(None, 0)
    }

    fn summary_with(&self, controller: Option<&MarginController>, margin_ms: i32) -> MarginSummary {
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
            stall_count: self.stall_count,
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
            margin_ms,
            margin_changes: controller.map_or(0, MarginController::changes),
            adaptive: controller.is_some(),
            margin_stable_ms: controller.map_or(0, MarginController::stable_for_ms),
            margin_changes_last_30s: controller.map_or(0, |c| c.changes_within_ms(30_000)),
            margin_trajectory: controller.map_or_else(Vec::new, |c| c.trajectory().to_vec()),
            time_at_margin_ms: controller.map_or_else(Vec::new, |c| c.time_at_margin_ms().to_vec()),
            superseded_decisions: 0,
            late_presses_dropped: 0,
            send_lag_us: None,
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
    /// Task 4.1b: `Some` when the margin adapts to the `NETMSG_INPUTTIMING` feedback.
    controller: Option<MarginController>,
    /// Inputs for ticks up to this one were sent before the last margin change (their `INPUTTIMING`
    /// describes the old margin and is ignored by the controller).
    stale_through_tick: i32,
    /// Task 4.1b round 2 (F7): warns about a link whose stalls stay above the threshold.
    stall_watch: StallWatch,
    /// Task 3.11: how late (us) each input was sent after it became due (the recent window), and how many were counted.
    send_lag: VecDeque<u32>,
    send_lag_count: u64,
    send_lag_max_us: u32,
}

impl InputTiming {
    pub fn new(prediction_margin_ms: i32) -> Self {
        InputTiming {
            controller: None,
            ..Self::with_fixed_margin(prediction_margin_ms)
        }
    }

    /// Task 4.1b: a timing whose margin starts at `initial_ms` and follows
    /// [`MarginController`]'s rule.
    pub fn adaptive(initial_ms: i32) -> Self {
        let controller = MarginController::new(initial_ms);
        InputTiming {
            controller: Some(controller.clone()),
            ..Self::with_fixed_margin(controller.margin_ms())
        }
    }

    fn with_fixed_margin(prediction_margin_ms: i32) -> Self {
        InputTiming {
            controller: None,
            stale_through_tick: 0,
            stall_watch: StallWatch::default(),
            prediction_margin_ms,
            predicted_time: None,
            received_snapshots: 0,
            latest_snapshot_tick: 0,
            pred_tick: 0,
            history: VecDeque::with_capacity(INPUT_HISTORY_LEN),
            margin_stats: MarginStats::default(),
            send_lag: VecDeque::new(),
            send_lag_count: 0,
            send_lag_max_us: 0,
        }
    }

    /// Task 4.1b: the runtime margin setter (what `advance`'s per-call `update_margin` reads).
    pub fn set_prediction_margin_ms(&mut self, margin_ms: i32) {
        self.prediction_margin_ms = margin_ms;
    }

    pub fn prediction_margin_ms(&self) -> i32 {
        self.prediction_margin_ms
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
        self.stale_through_tick = 0;
        self.history.clear();
        if let Some(c) = self.controller.as_mut() {
            c.reset();
            self.prediction_margin_ms = c.margin_ms();
        }
    }

    /// The predicted tick the next `NETMSG_INPUT` should carry, or `0` before the two-snapshot
    /// bootstrap has happened (`SendInput`'s own guard).
    pub fn pred_tick(&self) -> i32 {
        self.pred_tick
    }

    pub fn margin_summary(&self) -> MarginSummary {
        let mut s = self
            .margin_stats
            .summary_with(self.controller.as_ref(), self.prediction_margin_ms);
        if !self.send_lag.is_empty() {
            let mut v: Vec<u32> = self.send_lag.iter().copied().collect();
            v.sort_unstable();
            let at = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
            s.send_lag_us = Some(SendLagSummary {
                count: self.send_lag_count,
                p50_us: at(0.50),
                p90_us: at(0.90),
                p99_us: at(0.99),
                max_us: self.send_lag_max_us,
            });
        }
        s
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
            // Task 3.11: the input became due when the predicted clock reached the previous predicted tick's end
            // (`next_input_in_ns`); how long after that it is actually being sent is the loop's wake-up lag.
            if self.pred_tick > 0 {
                let tick_ns = crate::smooth_time::TIME_FREQ / i64::from(GAME_TICK_SPEED);
                let lag_us =
                    u32::try_from(((pred_now - i64::from(self.pred_tick) * tick_ns).max(0)) / 1000).unwrap_or(u32::MAX);
                if self.send_lag.len() == MARGIN_SAMPLE_WINDOW {
                    self.send_lag.pop_front();
                }
                self.send_lag.push_back(lag_us);
                self.send_lag_count += 1;
                self.send_lag_max_us = self.send_lag_max_us.max(lag_us);
            }
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
        let stall = time_left_ms < 0 && self.prediction_margin_ms - time_left_ms > ADAPTIVE_STALL_DELAY_MS;
        self.margin_stats.record(time_left_ms, stall);
        if let Some((stalls, count)) = self.stall_watch.on_sample(now_ns, stall) {
            tracing::warn!(
                stalls,
                inputs = count,
                stall_rate_percent = 100.0 * stalls as f64 / count as f64,
                margin_ms = self.prediction_margin_ms,
                "input timing: more than {}% of the inputs in the last minute were sent over {} ms later than the margin allows; no margin in range absorbs that (jitter beyond the range, or a starved host)",
                STALL_WATCH_RATE * 100.0,
                ADAPTIVE_STALL_DELAY_MS
            );
        }
        let stale = pred_tick <= self.stale_through_tick;
        if let Some(c) = self.controller.as_mut()
            && let Some(new) = c.on_sample(time_left_ms, now_ns, stale)
        {
            self.prediction_margin_ms = new;
            // Inputs already sent (up to `pred_tick`) went out with the old margin.
            self.stale_through_tick = self.pred_tick;
        }

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

    /// Task 3.11: how late each input is sent after it became due is recorded per `advance` that fires.
    #[test]
    fn the_send_lag_is_the_time_between_an_input_becoming_due_and_being_sent() {
        let mut timing = InputTiming::new(DEFAULT_PREDICTION_MARGIN_MS);
        timing.on_snapshot(100, 0);
        timing.on_snapshot(101, 0);
        timing.advance(0).expect("bootstrap tick");
        assert!(
            timing.margin_summary().send_lag_us.is_none(),
            "the bootstrap input has no due time to be late for"
        );
        let wait = timing.next_input_in_ns(0).unwrap();
        let tick_ns = TIME_FREQ / i64::from(GAME_TICK_SPEED);
        // 3 ms late, then exactly on time, then 7.5 ms late.
        let t1 = wait + 3_000_000;
        assert!(timing.advance(t1).is_some());
        let t2 = wait + tick_ns;
        assert!(timing.advance(t2).is_some());
        let t3 = t2 + tick_ns + 7_500_000;
        assert!(timing.advance(t3).is_some());
        let lag = timing.margin_summary().send_lag_us.expect("three samples");
        assert_eq!(lag.count, 3);
        assert!((2_900..=3_100).contains(&lag.p50_us), "{lag:?}");
        assert!((7_400..=7_600).contains(&lag.p99_us), "{lag:?}");
        assert!((7_400..=7_600).contains(&lag.max_us), "{lag:?}");
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

    // --- Task 4.1b: the adaptive margin -----------------------------------------------------------

    const MS: i64 = 1_000_000;

    /// Feeds `n` fresh samples 20 ms apart starting at `*now`, each `left_ms`; returns the last change.
    fn feed(c: &mut MarginController, now: &mut i64, n: usize, left_ms: i32) -> Option<i32> {
        let mut last = None;
        for _ in 0..n {
            *now += 20 * MS;
            if let Some(m) = c.on_sample(left_ms, *now, false) {
                last = Some(m);
            }
        }
        last
    }

    /// One fresh sample 20 ms after the last.
    fn one(c: &mut MarginController, now: &mut i64, left_ms: i32) -> Option<i32> {
        *now += 20 * MS;
        c.on_sample(left_ms, *now, false)
    }

    #[test]
    fn a_quiet_link_lowers_the_margin_one_ms_at_a_time_down_to_the_clamp() {
        let mut c = MarginController::new(10);
        let mut now = 0;
        // time_left follows the margin: always margin + 1 (plenty of slack).
        let mut seen = vec![10];
        for _ in 0..40 {
            let left = c.margin_ms() + 1;
            feed(&mut c, &mut now, 160, left);
            if *seen.last().unwrap() != c.margin_ms() {
                seen.push(c.margin_ms());
            }
        }
        assert_eq!(*seen.last().unwrap(), ADAPTIVE_MARGIN_MIN_MS, "{seen:?}");
        for w in seen.windows(2) {
            assert_eq!(w[0] - w[1], 1, "at most 1 ms at a time: {seen:?}");
        }
    }

    #[test]
    fn lowering_waits_for_a_full_window_and_the_interval() {
        let mut c = MarginController::new(10);
        let mut now = 0;
        assert_eq!(
            feed(&mut c, &mut now, ADAPTIVE_MIN_SAMPLES_TO_LOWER - 10, 11),
            None,
            "too few samples"
        );
        let first = feed(&mut c, &mut now, 20, 11);
        assert_eq!(first, Some(9));
        // The window starts over after a change: no second change until it refills.
        assert_eq!(feed(&mut c, &mut now, 100, 10), None);
    }

    /// Review F1: ONE late input never raises the margin.
    #[test]
    fn a_single_late_input_does_not_raise_the_margin_but_two_in_the_window_do() {
        let mut c = MarginController::new(8);
        let mut now = 0;
        feed(&mut c, &mut now, 10, 8);
        assert_eq!(one(&mut c, &mut now, -3), None, "one late input: nothing");
        assert_eq!(feed(&mut c, &mut now, 30, 8), None);
        assert_eq!(
            one(&mut c, &mut now, -4),
            Some(8 + ADAPTIVE_LATE_STEP_MS),
            "the second one in the window"
        );
        // The window started over: it takes two new lates to raise again.
        assert_eq!(one(&mut c, &mut now, -3), None);
        // ... and two lates farther apart than the window never raise it.
        let mut c = MarginController::new(8);
        let mut now = 0;
        for _ in 0..4 {
            assert_eq!(one(&mut c, &mut now, -2), None);
            feed(&mut c, &mut now, 300, 8); // 6 s later: the first one has left the window
        }
        assert_eq!(c.margin_ms(), 8);
    }

    #[test]
    fn clamped_at_the_top_and_stalls_are_ignored() {
        let mut c = MarginController::new(17);
        let mut now = 0;
        one(&mut c, &mut now, -1);
        assert_eq!(one(&mut c, &mut now, -1), Some(19));
        // From margin 19 on, any late input was sent > 18 ms late (a stall): the cap is reached by
        // the p1 rule only, never by lates.
        one(&mut c, &mut now, -1);
        assert_eq!(one(&mut c, &mut now, -1), None);
        assert_eq!(c.margin_ms(), 19);
        // A stall is judged by the implied send delay (margin - time_left): at margin 10 an input that
        // is 9 ms late was sent 19 ms too late (> 18): a stall; 5 ms late (delay 15) is jitter.
        let mut c = MarginController::new(10);
        assert!(!c.is_stall(-5));
        assert!(c.is_stall(-9));
        assert!(!c.is_stall(3), "an on-time input is never a stall");
        // ... so at margin 3 a 15 ms late input (delay 18) is still absorbable, 16 ms is not.
        let c3 = MarginController::new(3);
        assert!(!c3.is_stall(-15));
        assert!(c3.is_stall(-16));
        // Stalls neither count as lates in the window nor raise the margin.
        let mut now = 0;
        feed(&mut c, &mut now, 5, 11);
        for _ in 0..6 {
            assert_eq!(one(&mut c, &mut now, -30), None);
        }
        assert_eq!(c.margin_ms(), 10);
        assert_eq!(c.p1_ms(), Some(11));
    }

    /// Review F1: samples of inputs sent before the last change do not count.
    #[test]
    fn samples_of_inputs_sent_before_the_last_change_are_ignored() {
        let mut c = MarginController::new(8);
        let mut now = 0;
        one(&mut c, &mut now, -3);
        assert_eq!(one(&mut c, &mut now, -3), Some(10));
        for _ in 0..10 {
            now += 20 * MS;
            assert_eq!(c.on_sample(-4, now, true), None, "stale: the old margin's lateness");
        }
        assert_eq!(c.margin_ms(), 10);
        assert_eq!(c.p1_ms(), None, "and not in the window either");
    }

    #[test]
    fn a_low_p1_over_enough_samples_raises_it_and_an_outlier_among_many_does_not() {
        let mut c = MarginController::new(10);
        let mut now = 0;
        assert_eq!(
            feed(&mut c, &mut now, ADAPTIVE_MIN_SAMPLES_TO_RAISE - 5, 0),
            None,
            "too few samples"
        );
        assert_eq!(feed(&mut c, &mut now, 10, 0), Some(12));
        // One 1 ms sample among 200 healthy ones: p1 stays healthy, nothing changes.
        let mut c = MarginController::new(10);
        let mut now = 0;
        feed(&mut c, &mut now, 100, 3);
        one(&mut c, &mut now, 1);
        assert_eq!(feed(&mut c, &mut now, 20, 3), None);
        assert_eq!(c.margin_ms(), 10);
    }

    /// The hold after a raise decays: lowering is allowed after 10 s without a late input.
    #[test]
    fn lowering_is_allowed_again_after_ten_quiet_seconds() {
        let mut c = MarginController::new(10);
        let mut now = 0;
        feed(&mut c, &mut now, 5, 11);
        one(&mut c, &mut now, -4);
        assert_eq!(one(&mut c, &mut now, -4), Some(12), "two lates: raised");
        let raised_at = now;
        // Plenty of slack, but a late input 6 s in keeps the quiet period from starting over before 16 s.
        let mut lowered_at = None;
        let mut late_done = false;
        while now - raised_at < 40_000 * MS && lowered_at.is_none() {
            let left = if !late_done && now - raised_at >= 6_000 * MS {
                late_done = true;
                -1
            } else {
                c.margin_ms() + 1
            };
            if one(&mut c, &mut now, left) == Some(11) {
                lowered_at = Some(now);
            }
        }
        let after = (lowered_at.expect("lowered") - raised_at) / MS;
        assert!((16_000..=20_000).contains(&after), "lowered {after} ms after the raise");
    }

    #[test]
    fn the_window_is_five_seconds_and_reset_restores_the_initial_margin() {
        let mut c = MarginController::new(10);
        let mut now = 0;
        feed(&mut c, &mut now, 50, 3);
        assert_eq!(c.p1_ms(), Some(3));
        now += 6_000 * MS;
        one(&mut c, &mut now, 9);
        assert_eq!(c.p1_ms(), Some(9), "old samples drop out of the window");
        one(&mut c, &mut now, -4);
        one(&mut c, &mut now, -4); // two lates: raised
        assert_ne!(c.margin_ms(), 10);
        let changes = c.changes();
        c.reset();
        assert_eq!(c.margin_ms(), 10);
        assert_eq!(c.p1_ms(), None);
        assert_eq!(c.changes(), changes, "the change counter is a session total");
    }

    #[test]
    fn input_timing_applies_the_controllers_margin_ignores_stale_samples_and_resets_on_a_new_map() {
        let mut timing = InputTiming::adaptive(10);
        timing.on_snapshot(100, 0);
        timing.on_snapshot(101, 0);
        timing.advance(0).expect("the bootstrap tick");
        let sent = timing.pred_tick();
        timing.on_input_timing(sent, -3, 0);
        assert_eq!(timing.prediction_margin_ms(), 10, "one late input: no change");
        timing.on_input_timing(sent, -3, 20 * MS);
        assert_eq!(timing.prediction_margin_ms(), 10 + ADAPTIVE_LATE_STEP_MS);
        let s = timing.margin_summary();
        assert_eq!(
            (s.margin_ms, s.margin_changes, s.adaptive),
            (10 + ADAPTIVE_LATE_STEP_MS, 1, true)
        );
        assert_eq!(s.margin_trajectory.len(), 1);
        assert_eq!(s.margin_trajectory[0].1, 12);
        // The input for `sent` went out with the old margin: its late reports no longer count.
        for i in 0..10 {
            timing.on_input_timing(sent, -3, (40 + 20 * i) * MS);
        }
        assert_eq!(
            timing.prediction_margin_ms(),
            12,
            "stale samples do not wind the margin up"
        );
        timing.reset();
        assert_eq!(timing.prediction_margin_ms(), 10, "reset on a new map / connection");
        // A fixed timing never moves, and has a runtime setter.
        let mut fixed = InputTiming::new(10);
        fixed.on_input_timing(1, -3, 0);
        fixed.on_input_timing(1, -3, MS);
        assert_eq!(fixed.prediction_margin_ms(), 10);
        assert!(!fixed.margin_summary().adaptive);
        fixed.set_prediction_margin_ms(4);
        assert_eq!(fixed.margin_summary().margin_ms, 4);
    }

    #[test]
    fn changes_within_counts_recent_changes_only_and_time_at_margin_is_tracked() {
        let mut c = MarginController::new(10);
        let mut now = 0;
        one(&mut c, &mut now, -1);
        assert_eq!(one(&mut c, &mut now, -1), Some(12));
        now += 40_000 * MS;
        feed(&mut c, &mut now, 1, 13);
        assert_eq!(c.changes(), 1);
        assert_eq!(c.changes_within_ms(30_000), 0, "the change is 40 s old");
        assert!(c.time_at_margin_ms()[12] >= 1_000, "{:?}", c.time_at_margin_ms());
        assert!(c.time_at_margin_ms()[10] > 0);
    }

    #[test]
    fn stalls_are_late_inputs_sent_too_late_for_any_margin_in_range() {
        let mut timing = InputTiming::new(10);
        // margin 10: delay = 10 - left. 18 is the limit.
        for left in [5, -1, -8, -9, -120] {
            timing.on_input_timing(1, left, 0);
        }
        let s = timing.margin_summary();
        assert_eq!((s.count, s.late_count, s.stall_count), (5, 4, 2));
    }

    /// Review round 2, F7: lowering needs p1 two ms above the safety level, so it stops one step earlier
    /// than before and does not sit on the edge.
    #[test]
    fn lowering_stops_while_p1_is_less_than_the_safety_level_plus_the_hysteresis() {
        let mut c = MarginController::new(10);
        let mut now = 0;
        // p1 = margin - 5: lowers while p1 - 2 >= 2, i.e. margin >= 9.
        for _ in 0..40 {
            let left = c.margin_ms() - 5;
            feed(&mut c, &mut now, 160, left.max(0));
        }
        // At margin 9 p1 = 4: 4 - 2 = 2 still lowers (to 8); at 8, p1 = 3 leaves 1 < 2: it stops with p1 one
        // above the safety level instead of on it.
        assert_eq!(c.margin_ms(), 8);
    }

    #[test]
    fn the_stall_watch_speaks_once_per_window_and_only_above_the_threshold() {
        let mut w = StallWatch::default();
        let mut now = 0i64;
        let mut verdicts = vec![];
        // 3 minutes at 50 inputs/s with 1 stall per 100 inputs (1%): one verdict per minute.
        for i in 0..9000u64 {
            now += 20 * MS;
            if let Some(v) = w.on_sample(now, i % 100 == 0) {
                verdicts.push(v);
            }
        }
        assert_eq!(verdicts.len(), 2, "windows ending inside the 3 minutes: {verdicts:?}");
        assert!(verdicts.iter().all(|&(s, c)| s as f64 > 0.005 * c as f64 && c > 2500));
        // 0.2% stalls: silent. Too few samples: silent.
        let mut w = StallWatch::default();
        let mut now = 0i64;
        for i in 0..6000u64 {
            now += 20 * MS;
            assert_eq!(w.on_sample(now, i % 500 == 0), None);
        }
        let mut w = StallWatch::default();
        assert_eq!(w.on_sample(0, true), None);
        assert_eq!(w.on_sample(61_000 * MS, true), None, "two samples are no evidence");
    }
}
