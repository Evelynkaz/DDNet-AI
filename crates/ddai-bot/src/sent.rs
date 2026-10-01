//! Our own inputs in flight: the log of what `NETMSG_INPUT` carried for which tick
//! (`SessionEvent::InputSent`) and what the server said about its timing
//! (`SessionEvent::InputTiming`), turned into "the input the server actually applies at tick T".
//!
//! This replaces the TS `sent`/`inFlightInputs` with the `lag` heuristic (an eco-measured or
//! ping-derived 0..6 ticks, `bot.ts:4494-4563,4871-4880`) by D-023's exact model: the driver knows
//! every tick an input was sent for, and `NETMSG_INPUTTIMING` says which ones were late. A late
//! input is not dropped by DDNet 20.1's server, it is re-targeted forward: `IntendedTick =
//! max(IntendedTick, Server()->Tick() + 1)` (`server.cpp:1921`) — so a tick `T` reported `time_left
//! < 0` ms lands on `T + ceil(-time_left / 20)`, the first input to claim a tick keeps it, and a tick
//! nobody claims keeps the previous input. This is `ddai_world::retarget_late_inputs` (round 3 of
//! task 2.4's review, cross-validated against a live capture at 1.0000 exact), re-implemented over
//! fixed storage so the per-snapshot path allocates nothing; a test pins the two together.

use ddai_net::generated::objects;
use ddai_physics::core::PlayerInput;
use ddai_world::player_input_from_net;

/// Entries kept: a few seconds of ticks, far more than any in-flight window.
const CAP: usize = 64;
/// One server tick in milliseconds (`NETMSG_INPUTTIMING` reports `time_left` in ms).
const TICK_MS: i32 = 20;

#[derive(Debug, Clone, Copy)]
struct Sent {
    tick: i32,
    input: PlayerInput,
    /// `time_left` in ms, once the server reported it.
    time_left: Option<i32>,
}

/// The ring of sent inputs plus the derived claims.
pub struct SentLog {
    ring: Vec<Sent>,
    /// `(effective tick, input)` sorted by effective tick, rebuilt by [`SentLog::refresh`].
    claims: Vec<(i32, PlayerInput)>,
}

impl Default for SentLog {
    fn default() -> Self {
        Self::new()
    }
}

impl SentLog {
    pub fn new() -> Self {
        SentLog {
            ring: Vec::with_capacity(CAP),
            claims: Vec::with_capacity(CAP),
        }
    }

    pub fn clear(&mut self) {
        self.ring.clear();
        self.claims.clear();
    }

    /// `SessionEvent::InputSent { tick, input }`.
    pub fn on_sent(&mut self, tick: i32, input: &objects::PlayerInput) {
        // Inputs arrive in tick order; a repeated or older tick replaces nothing (first wins).
        if self.ring.last().is_some_and(|s| s.tick >= tick) {
            return;
        }
        if self.ring.len() == CAP {
            self.ring.remove(0);
        }
        self.ring.push(Sent {
            tick,
            input: player_input_from_net(*input),
            time_left: None,
        });
    }

    /// `SessionEvent::InputTiming { tick, time_left }`.
    pub fn on_timing(&mut self, tick: i32, time_left: i32) {
        if let Some(s) = self.ring.iter_mut().rev().find(|s| s.tick == tick) {
            s.time_left = Some(time_left);
        }
    }

    /// Recomputes the claims (call once per snapshot, after the events). `keep_from`: entries more
    /// than a window older than this tick are dropped from the ring.
    pub fn refresh(&mut self, keep_from: i32) {
        self.ring.retain(|s| s.tick >= keep_from - 32);
        self.claims.clear();
        for s in &self.ring {
            let eff = match s.time_left {
                Some(t) if t < 0 => s.tick + (-t + TICK_MS - 1) / TICK_MS,
                _ => s.tick,
            };
            // The first (earliest original tick) input to claim an effective tick keeps it.
            match self.claims.binary_search_by_key(&eff, |c| c.0) {
                Ok(_) => {}
                Err(at) => self.claims.insert(at, (eff, s.input)),
            }
        }
    }

    /// The input in force at `tick` (the last claim at or before it), if any.
    pub fn effective_at(&self, tick: i32) -> Option<PlayerInput> {
        self.claims.iter().rev().find(|c| c.0 <= tick).map(|c| c.1)
    }

    /// Claims with `from < tick <= to`, appended to `out` (cleared first) — the `own_inputs_in_flight`
    /// of [`ddai_world::LiveWorld::predict`].
    pub fn in_flight(&self, from: i32, to: i32, out: &mut Vec<(i32, PlayerInput)>) {
        out.clear();
        out.extend(self.claims.iter().filter(|c| c.0 > from && c.0 <= to).copied());
    }

    pub fn len(&self) -> usize {
        self.ring.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    fn net(direction: i32, fire: i32) -> objects::PlayerInput {
        objects::PlayerInput {
            direction,
            target_x: 10,
            target_y: -5,
            jump: 0,
            fire,
            hook: 0,
            player_flags: 1,
            wanted_weapon: 1,
            next_weapon: 0,
            prev_weapon: 0,
        }
    }

    fn log_of(entries: &[(i32, i32, Option<i32>)]) -> SentLog {
        let mut l = SentLog::new();
        for &(tick, dir, tl) in entries {
            l.on_sent(tick, &net(dir, 0));
            if let Some(t) = tl {
                l.on_timing(tick, t);
            }
        }
        l.refresh(0);
        l
    }

    #[test]
    fn on_time_inputs_apply_on_their_own_tick_and_gaps_hold_the_previous() {
        let l = log_of(&[(10, 1, Some(5)), (11, -1, Some(3)), (13, 0, None)]);
        assert_eq!(l.effective_at(9).map(|i| i.direction), None);
        assert_eq!(l.effective_at(10).unwrap().direction, 1);
        assert_eq!(l.effective_at(11).unwrap().direction, -1);
        assert_eq!(l.effective_at(12).unwrap().direction, -1, "nothing sent for 12: hold");
        assert_eq!(l.effective_at(13).unwrap().direction, 0);
    }

    #[test]
    fn a_late_input_lands_ceil_lateness_over_20ms_ticks_later() {
        // 25 ms late: ceil(25/20) = 2 ticks; 20 ms late: exactly 1 tick; 1 ms late: 1 tick.
        let l = log_of(&[(10, 1, Some(-25)), (20, 1, Some(-20)), (30, 1, Some(-1))]);
        let mut out = Vec::new();
        l.in_flight(0, 100, &mut out);
        assert_eq!(out.iter().map(|c| c.0).collect::<Vec<_>>(), vec![12, 21, 31]);
    }

    #[test]
    fn the_first_input_to_claim_a_tick_keeps_it() {
        // Tick 10 is 25 ms late -> lands on 12; the on-time input for tick 12 arrives later in the
        // log and loses the claim.
        let mut l = SentLog::new();
        l.on_sent(10, &net(1, 0));
        l.on_timing(10, -25);
        l.on_sent(11, &net(2, 0));
        l.on_sent(12, &net(3, 0));
        l.refresh(0);
        assert_eq!(
            l.effective_at(12).unwrap().direction,
            1,
            "tick 10's input claimed 12 first"
        );
        assert_eq!(l.effective_at(11).unwrap().direction, 2);
    }

    #[test]
    fn in_flight_is_the_half_open_window_and_fire_counters_pass_through() {
        let mut l = SentLog::new();
        for t in 10..=15 {
            l.on_sent(t, &net(0, t)); // fire counter = tick, to see it survive
        }
        l.refresh(0);
        let mut out = Vec::new();
        l.in_flight(11, 14, &mut out);
        assert_eq!(
            out.iter().map(|c| (c.0, c.1.fire)).collect::<Vec<_>>(),
            vec![(12, 12), (13, 13), (14, 14)]
        );
        l.in_flight(99, 100, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn a_repeated_or_out_of_order_tick_is_ignored() {
        let mut l = SentLog::new();
        l.on_sent(10, &net(1, 0));
        l.on_sent(10, &net(2, 0));
        l.on_sent(9, &net(3, 0));
        l.refresh(0);
        assert_eq!(l.len(), 1);
        assert_eq!(l.effective_at(10).unwrap().direction, 1);
    }

    #[test]
    fn the_ring_is_bounded_and_old_entries_age_out() {
        let mut l = SentLog::new();
        for t in 0..500 {
            l.on_sent(t, &net(0, 0));
        }
        assert_eq!(l.len(), CAP);
        l.refresh(495);
        assert!(l.len() <= 40, "{}", l.len());
    }

    /// The fixed-storage model is the same function as `ddai_world::retarget_late_inputs`.
    #[test]
    fn agrees_with_the_world_crates_retarget_model() {
        let mut entries = Vec::new();
        let mut timing = BTreeMap::new();
        let mut l = SentLog::new();
        let lateness = [5, 12, -3, 8, -21, 9, 14, -45, 7, 2, -19, 6, 11, -1, 10, 4, 3, -60, 8, 9];
        for (i, &tl) in lateness.iter().enumerate() {
            let tick = 100 + i as i32 + i32::from(i > 12); // one gap
            let input = net((i % 3) as i32 - 1, i as i32);
            l.on_sent(tick, &input);
            l.on_timing(tick, tl);
            entries.push((tick, player_input_from_net(input)));
            timing.insert(tick, tl);
        }
        l.refresh(0);
        let reference = ddai_world::retarget_late_inputs(&entries, &timing);
        for &(tick, want) in &reference {
            assert_eq!(l.effective_at(tick), Some(want), "tick {tick}");
        }
        let _: BTreeSet<i32> = BTreeSet::new();
    }

    #[test]
    fn refresh_and_window_queries_allocate_nothing_once_warm() {
        let mut l = SentLog::new();
        let mut out = Vec::with_capacity(CAP);
        for t in 0..CAP as i32 {
            l.on_sent(t, &net(0, 0));
        }
        l.refresh(0);
        let info = allocation_counter::measure(|| {
            for k in 0..200 {
                l.on_sent(1000 + k, &net(1, k));
                l.on_timing(1000 + k, -(k % 50));
                l.refresh(1000 + k - 40);
                l.in_flight(1000 + k - 4, 1000 + k, &mut out);
                std::hint::black_box(l.effective_at(1000 + k));
            }
        });
        assert_eq!(info.count_total, 0, "{info:?}");
    }
}
