// Ported from DDNet `src/engine/client/smooth_time.{h,cpp}` (pinned rev
// c9d208138f85755521f16a0096b6fe036c5c8698, "20.1"), which carries the original Teeworlds
// zlib-style notice:
//
//   (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information.
//   If you are missing that file, acquire a complete release at teeworlds.com.
//
// This is an *altered* source version: rewritten in safe Rust, same smoothing/adjust-speed
// arithmetic, adapted to a sans-IO shape (the C++ `CSmoothTime::Get`/`UpdateInt` call `time_get()`
// themselves; here every method that cares about "now" takes it as an explicit parameter, matching
// `ddai_net::conn::Connection`'s own sans-IO convention).
//
//! [`SmoothTime`]: DDNet's `CSmoothTime` — a self-adjusting clock that smoothly nudges its
//! `Current` value towards a `Target` the caller periodically reports (e.g. "here is where the
//! game/predicted tick should be, according to the last snapshot/`NETMSG_INPUTTIMING`"), instead
//! of jumping straight to it — used for both the *game* clock (`m_aGameTime`, driven by snapshot
//! arrival) and the *predicted* clock (`m_PredictedTime`, driven by `NETMSG_INPUTTIMING`) in
//! `crate::timing`.
//!
//! Units: the C++ type uses `int64_t` ticks of `time_freq()` per second (a platform-specific
//! high-resolution counter frequency — irrelevant to the *behavior*, since every quantity here is
//! either a ratio against `time_freq()` or a duration expressed in the same unit). This port fixes
//! the unit to nanoseconds (`time_freq() == 1_000_000_000`), so every `i64` here is directly an
//! `i64` nanosecond count — convertible losslessly to/from [`std::time::Duration`] for any value
//! that stays non-negative (a full [`SmoothTime`] history can go negative internally, e.g. a
//! `Target` set from a tick number multiplied down, so the raw `i64` API is kept internally and
//! [`crate::timing`] is the layer that decides when a value is meaningful as a `Duration`).

/// `time_freq()`: fixed at nanoseconds/second for this port — see the module docs.
pub const TIME_FREQ: i64 = 1_000_000_000;

/// `CSmoothTime::EAdjustDirection`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdjustDirection {
    Down = 0,
    Up = 1,
}

/// A single adjust-speed pair, indexed by [`AdjustDirection`] (`m_aAdjustSpeed[NUM_ADJUSTDIRECTIONS]`).
#[derive(Debug, Clone, Copy, PartialEq)]
struct AdjustSpeeds {
    down: f32,
    up: f32,
}

impl AdjustSpeeds {
    fn get(&self, dir: AdjustDirection) -> f32 {
        match dir {
            AdjustDirection::Down => self.down,
            AdjustDirection::Up => self.up,
        }
    }

    fn get_mut(&mut self, dir: AdjustDirection) -> &mut f32 {
        match dir {
            AdjustDirection::Down => &mut self.down,
            AdjustDirection::Up => &mut self.up,
        }
    }
}

/// `CSmoothTime` (`smooth_time.h`) — see the module docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmoothTime {
    snap: i64,
    current: i64,
    target: i64,
    margin: i64,
    spike_counter: i32,
    adjust_speed: AdjustSpeeds,
}

/// What [`SmoothTime::update`] classified the reported `time_left` as — purely informational
/// (mirrors the graph-coloring branches in `CSmoothTime::Update`, `smooth_time.cpp:55-98`), used
/// by [`crate::timing::MarginStats`] instead of a real on-screen graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateOutcome {
    /// `time_left >= 0`: on time or early: eases the relevant adjust speed back down.
    OnTime,
    /// `time_left < 0` but treated as noise (an isolated spike, `spike_counter` still low):
    /// the timer target is *not* updated this call (`smooth_time.cpp:71-76`).
    IgnoredSpike,
    /// `time_left < 0` and acted on: adjust speed doubled (capped at 30.0), target updated.
    Late,
}

impl SmoothTime {
    /// `CSmoothTime::Init` (`smooth_time.cpp:11-20`). `now` replaces `time_get()`.
    pub fn init(target: i64, now: i64) -> Self {
        SmoothTime {
            snap: now,
            current: target,
            target,
            margin: 0,
            spike_counter: 0,
            adjust_speed: AdjustSpeeds { down: 0.3, up: 0.3 },
        }
    }

    /// `CSmoothTime::SetAdjustSpeed` (`smooth_time.cpp:22-25`).
    pub fn set_adjust_speed(&mut self, direction: AdjustDirection, value: f32) {
        *self.adjust_speed.get_mut(direction) = value;
    }

    /// `CSmoothTime::UpdateMargin` (`smooth_time.cpp:100-103`).
    pub fn update_margin(&mut self, margin: i64) {
        self.margin = margin;
    }

    /// `CSmoothTime::Get` (`smooth_time.cpp:27-45`): the smoothed clock value at `now`.
    pub fn get(&self, now: i64) -> i64 {
        let c = self.current + (now - self.snap);
        let t = self.target + (now - self.snap);

        // "it's faster to adjust upward instead of downward" — smooth_time.cpp:32-37.
        let adjust_speed = if t > c {
            self.adjust_speed.get(AdjustDirection::Up)
        } else {
            self.adjust_speed.get(AdjustDirection::Down)
        };

        let mut a = ((now - self.snap) as f64 / TIME_FREQ as f64) as f32 * adjust_speed;
        if a > 1.0 {
            a = 1.0;
        }

        let r = c + ((t - c) as f64 * a as f64) as i64;
        r + self.margin
    }

    /// `CSmoothTime::UpdateInt` (`smooth_time.cpp:47-53`): re-snapshots `current`/`snap` at `now`
    /// and sets a fresh `target`, without touching the adjust speeds or spike counter (that is
    /// [`SmoothTime::update`]'s job, which calls this internally when appropriate).
    pub fn update_int(&mut self, target: i64, now: i64) {
        self.current = self.get(now) - self.margin;
        self.snap = now;
        self.target = target;
    }

    /// `CSmoothTime::Update` (`smooth_time.cpp:55-98`): the feedback step driven by a
    /// `NETMSG_INPUTTIMING`/snapshot-derived `(target, time_left_ms)` pair. `direction` selects
    /// which of the two adjust speeds this feedback tunes (predicted time only ever adjusts
    /// [`AdjustDirection::Up`]; game time only ever adjusts [`AdjustDirection::Down`] — see
    /// `crate::timing`, matching the two call sites in `client.cpp`).
    ///
    /// Returns which branch was taken (see [`UpdateOutcome`]) so a caller can feed
    /// [`crate::timing::MarginStats`] without re-deriving the same classification.
    pub fn update(&mut self, target: i64, time_left_ms: i32, direction: AdjustDirection, now: i64) -> UpdateOutcome {
        let mut update_timer = true;
        let outcome;

        if time_left_ms < 0 {
            let is_spike = time_left_ms < -50;
            if is_spike {
                self.spike_counter = (self.spike_counter + 5).min(50);
            }

            if is_spike && self.spike_counter < 15 {
                // Ignore this ping spike.
                update_timer = false;
                outcome = UpdateOutcome::IgnoredSpike;
            } else {
                let speed = self.adjust_speed.get_mut(direction);
                if *speed < 30.0 {
                    *speed *= 2.0;
                }
                outcome = UpdateOutcome::Late;
            }
        } else {
            if self.spike_counter > 0 {
                self.spike_counter -= 1;
            }
            let speed = self.adjust_speed.get_mut(direction);
            *speed *= 0.95;
            if *speed < 2.0 {
                *speed = 2.0;
            }
            outcome = UpdateOutcome::OnTime;
        }

        if update_timer {
            self.update_int(target, now);
        }
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_get_returns_target_plus_margin_immediately() {
        let st = SmoothTime::init(1_000_000_000, 0);
        assert_eq!(st.get(0), 1_000_000_000);
    }

    #[test]
    fn get_matches_the_closed_form_of_the_ported_formula() {
        // `Get`'s formula (`smooth_time.cpp:27-45`) is `r = c + (t-c)*a`, `c = current+elapsed`,
        // `t = target+elapsed`, so `t-c = target-current` is an *invariant*, independent of
        // elapsed time — i.e. this is not a simple lerp from `current` to a fixed `target`, it is
        // a control loop whose *gap* (`target-current`) gets scaled down by `a` while both ends
        // drift forward with the wall clock. With `current=0`, `target=TIME_FREQ`, `speed=1.0`:
        // `a = min(1, elapsed/FREQ)`, so `r(elapsed) = elapsed + TIME_FREQ*min(1, elapsed/FREQ)`.
        let mut st = SmoothTime::init(0, 0);
        st.set_adjust_speed(AdjustDirection::Up, 1.0);
        st.update_int(TIME_FREQ, 0); // current=0, target=TIME_FREQ (1s away), snapped at t=0

        // a = 0.5 here: r = TIME_FREQ/2 + TIME_FREQ*0.5 = TIME_FREQ.
        assert_eq!(st.get(TIME_FREQ / 2), TIME_FREQ);
        // a clamps to exactly 1.0 here (elapsed == FREQ/speed): r = TIME_FREQ + TIME_FREQ*1 = 2*TIME_FREQ.
        assert_eq!(st.get(TIME_FREQ), 2 * TIME_FREQ);
        // a stays clamped at 1.0 past that point: r = 2*TIME_FREQ + TIME_FREQ*1 = 3*TIME_FREQ.
        assert_eq!(st.get(2 * TIME_FREQ), 3 * TIME_FREQ);
    }

    #[test]
    fn update_margin_is_a_flat_offset_on_get() {
        let mut st = SmoothTime::init(500, 0);
        st.update_margin(1234);
        assert_eq!(st.get(0), 500 + 1234);
    }

    #[test]
    fn update_on_time_relaxes_adjust_speed_towards_two() {
        let mut st = SmoothTime::init(0, 0);
        st.set_adjust_speed(AdjustDirection::Up, 1000.0);
        let outcome = st.update(TIME_FREQ, 5, AdjustDirection::Up, 0);
        assert_eq!(outcome, UpdateOutcome::OnTime);
        assert!(
            (st.adjust_speed.up - 950.0).abs() < 1e-3,
            "950 = 1000*0.95, got {}",
            st.adjust_speed.up
        );
    }

    #[test]
    fn update_on_time_never_drops_adjust_speed_below_two() {
        let mut st = SmoothTime::init(0, 0);
        st.set_adjust_speed(AdjustDirection::Up, 2.01);
        st.update(0, 0, AdjustDirection::Up, 0);
        assert_eq!(st.adjust_speed.up, 2.0);
    }

    #[test]
    fn update_late_but_not_a_big_spike_doubles_adjust_speed_and_still_updates_target() {
        let mut st = SmoothTime::init(0, 0);
        st.set_adjust_speed(AdjustDirection::Up, 1.0);
        let outcome = st.update(999, -10, AdjustDirection::Up, 5);
        assert_eq!(outcome, UpdateOutcome::Late);
        assert_eq!(st.adjust_speed.up, 2.0);
        assert_eq!(st.target, 999);
        assert_eq!(st.snap, 5);
    }

    #[test]
    fn update_late_doubles_speed_and_can_overshoot_past_thirty() {
        // `smooth_time.cpp:80-82`'s guard is `if(speed < 30.0) speed *= 2.0` — a starting speed
        // just under 30 still doubles once, ending up *above* 30 (not clamped to exactly 30).
        let mut st = SmoothTime::init(0, 0);
        st.set_adjust_speed(AdjustDirection::Up, 25.0);
        st.update(0, -10, AdjustDirection::Up, 0);
        assert_eq!(st.adjust_speed.up, 50.0);
    }

    #[test]
    fn update_late_stops_doubling_once_at_or_above_thirty() {
        let mut st = SmoothTime::init(0, 0);
        st.set_adjust_speed(AdjustDirection::Up, 30.0);
        st.update(0, -10, AdjustDirection::Up, 0);
        assert_eq!(
            st.adjust_speed.up, 30.0,
            "speed already >= 30 must not be doubled further"
        );
    }

    #[test]
    fn a_single_big_spike_is_ignored_and_does_not_move_the_target() {
        let mut st = SmoothTime::init(0, 0);
        let before_target = st.target;
        let outcome = st.update(999_999, -60, AdjustDirection::Up, 5);
        assert_eq!(outcome, UpdateOutcome::IgnoredSpike);
        assert_eq!(st.target, before_target, "an ignored spike must not call UpdateInt");
        assert_eq!(st.spike_counter, 5);
    }

    #[test]
    fn repeated_big_spikes_eventually_stop_being_ignored() {
        let mut st = SmoothTime::init(0, 0);
        let mut last = UpdateOutcome::IgnoredSpike;
        // spike_counter += 5 each big spike, capped at 50; stops being ignored once >= 15,
        // i.e. by the 3rd call (5, 10, 15).
        for _ in 0..3 {
            last = st.update(42, -60, AdjustDirection::Up, 0);
        }
        assert_eq!(last, UpdateOutcome::Late);
        assert_eq!(st.target, 42);
    }

    #[test]
    fn spike_counter_decays_by_one_on_each_on_time_update() {
        let mut st = SmoothTime::init(0, 0);
        st.update(0, -60, AdjustDirection::Up, 0); // spike_counter = 5
        assert_eq!(st.spike_counter, 5);
        st.update(0, 1, AdjustDirection::Up, 0);
        assert_eq!(st.spike_counter, 4);
    }
}
