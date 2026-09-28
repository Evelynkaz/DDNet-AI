//! Prediction-accuracy measurement (task spec, acceptance criterion 1's "measurement mode" and
//! criterion 2): logs [`LiveWorld::predict`](crate::LiveWorld::predict)'s predicted positions
//! against the later, *actual* reconstructed positions once a snapshot for that same tick
//! arrives, and summarizes the result — own tee: the bit-exact fraction; everyone else: an
//! error-in-pixels distribution per prediction horizon.
//!
//! Deliberately decoupled from [`crate::LiveWorld`] itself (it takes plain `Vec2<f32>`/`&World`
//! values, never a `&mut LiveWorld`): a caller drives both independently —
//! [`AccuracyTracker::record_prediction`] right after a `predict()` call, then, once the *next*
//! real snapshot(s) confirm each predicted tick, [`AccuracyTracker::record_actual`] once per
//! `on_snapshot` — which keeps this module trivial to unit-test without any network/reckoning
//! machinery at all (see this file's own tests).

use std::collections::VecDeque;

use ddai_physics::vmath::Vec2;
use ddai_physics::world::World;

/// Review round 1, finding F9: an unbounded `samples` `Vec` grows for as long as the tracker
/// runs (the reviewer's own estimate: ~65 MB/hour of continuous 3-horizon logging) — bounded to a
/// generous but fixed window instead (a few tens of minutes of continuous logging at a realistic
/// rate; still comfortably enough samples for [`AccuracyTracker::summarize`]'s percentiles to
/// mean something), oldest-evicted, so long-running use has a fixed memory ceiling.
const MAX_SAMPLES: usize = 50_000;

/// One outstanding "I predicted character `id` would be at `predicted_pos` at tick
/// `target_tick`" — waiting for a real snapshot to confirm or refute it.
#[derive(Debug, Clone, Copy)]
struct PendingEntry {
    base_tick: i32,
    target_tick: i32,
    id: i32,
    predicted_pos: Vec2<f32>,
    /// Review round 3, finding F13: whether `id` was already frozen at `base_tick` (the caller's
    /// own knowledge at `record_prediction` time — see that method's own doc comment).
    base_frozen: bool,
}

/// A pending prediction whose target tick has come and gone without ever being confirmed (the
/// character left, or the caller skipped a snapshot) is dropped rather than kept forever —
/// bounds this tracker's memory to "however many predictions are in flight right now", not
/// "every prediction ever made".
const STALE_AFTER_TICKS: i32 = 200;

/// One resolved prediction-vs-actual comparison.
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    pub id: i32,
    pub is_own: bool,
    /// `target_tick - base_tick` — how many ticks ahead of the base snapshot this was predicted.
    pub horizon_ticks: i32,
    pub predicted: Vec2<f32>,
    pub actual: Vec2<f32>,
    pub error_px: f32,
    /// Bit-exact position match (task spec's own-tee acceptance bar).
    pub exact: bool,
    /// Review round 3, finding F13: was `id` frozen at `base_tick` (prediction time)? A frozen
    /// tee barely moves tick to tick, so a summary that doesn't exclude these is mostly measuring
    /// "predicting a near-motionless tee is easy", not real prediction accuracy — see
    /// [`AccuracyTracker::summarize`]'s own doc comment for where this is actually enforced.
    pub base_frozen: bool,
    /// Same as [`Self::base_frozen`], but at `target_tick` (confirmation time) — a tee that
    /// *becomes* frozen partway through the predicted window is just as trivially "accurate" as
    /// one that started frozen.
    pub target_frozen: bool,
}

/// Logs predictions and, once confirmed, the resulting [`Sample`]s. See the module doc comment
/// for the two-call protocol ([`Self::record_prediction`] / [`Self::record_actual`]).
#[derive(Debug, Clone, Default)]
pub struct AccuracyTracker {
    own_id: i32,
    pending: Vec<PendingEntry>,
    samples: VecDeque<Sample>,
}

impl AccuracyTracker {
    pub fn new(own_id: i32) -> Self {
        AccuracyTracker {
            own_id,
            pending: Vec::new(),
            samples: VecDeque::new(),
        }
    }

    /// Logs one predicted position (task spec: "log predicted-vs-snapshot differences") — call
    /// once per `(character, horizon)` right after a `LiveWorld::predict(base_tick + horizon, ..)`
    /// call, reading `predicted_pos` from that result's `cores.get(id)`.
    ///
    /// `base_frozen` (review round 3, finding F13): whether `id` was frozen at `base_tick` — the
    /// caller already has this (`world.characters[id].freeze_time > 0` on the same base world
    /// `predicted_pos` was derived from); pass it through so [`Self::summarize`] can exclude a
    /// frozen-at-either-end own-tee sample from the task spec's `>= 0.999` bar (a frozen tee is
    /// trivially easy to predict — see [`Sample::base_frozen`]'s own doc comment).
    pub fn record_prediction(
        &mut self,
        base_tick: i32,
        target_tick: i32,
        id: i32,
        predicted_pos: Vec2<f32>,
        base_frozen: bool,
    ) {
        self.pending.push(PendingEntry {
            base_tick,
            target_tick,
            id,
            predicted_pos,
            base_frozen,
        });
    }

    /// Resolves every pending prediction targeting exactly `tick` against `world` (the *base*,
    /// unpredicted [`crate::LiveWorld::base_world`] right after `on_snapshot(tick, ...)`) — a
    /// character no longer present at `tick` is simply dropped (no sample), matching the task
    /// spec's own framing ("report where and why prediction fails", not "count a departed
    /// character as an error"). `target_frozen` (review round 3, finding F13) is read straight off
    /// `world.characters[id].freeze_time` — the same `world` the caller is already resolving
    /// positions against, so no extra parameter is needed for this half of the pair (unlike
    /// `base_frozen`, which was only known back when the *prediction* was logged).
    pub fn record_actual(&mut self, tick: i32, world: &World<f32>) {
        let mut i = 0;
        while i < self.pending.len() {
            let p = self.pending[i];
            if p.target_tick == tick {
                if let Some(core) = world.cores.get(p.id as u8) {
                    let actual = core.pos;
                    let target_frozen = world.characters[p.id as usize]
                        .map(|c| c.freeze_time > 0)
                        .unwrap_or(false);
                    if self.samples.len() >= MAX_SAMPLES {
                        self.samples.pop_front();
                    }
                    self.samples.push_back(Sample {
                        id: p.id,
                        is_own: p.id == self.own_id,
                        horizon_ticks: p.target_tick - p.base_tick,
                        predicted: p.predicted_pos,
                        actual,
                        error_px: ddai_physics::vmath::distance(p.predicted_pos, actual),
                        exact: p.predicted_pos == actual,
                        base_frozen: p.base_frozen,
                        target_frozen,
                    });
                }
                self.pending.swap_remove(i);
            } else if tick - p.target_tick > STALE_AFTER_TICKS {
                self.pending.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    /// Every resolved [`Sample`] still retained (see [`MAX_SAMPLES`] — the oldest ones are
    /// evicted once the tracker has been running long enough to hit it).
    pub fn samples(&self) -> &VecDeque<Sample> {
        &self.samples
    }

    /// How many predictions are still waiting for their target tick to be confirmed.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Summarizes [`Self::samples`] (task spec, acceptance criterion 2): one [`HorizonSummary`]
    /// per distinct `(is_own, horizon_ticks)` pair actually observed, sorted by
    /// `(is_own descending, horizon_ticks ascending)` (own tee's summaries first, since it is the
    /// one with a pass/fail bar; everyone else's grouped by horizon next).
    ///
    /// Review round 3, finding F13: **own-tee** (`is_own`) groups only ever count a sample where
    /// the own tee was unfrozen at *both* `base_tick` and `target_tick` — a frozen tee barely
    /// moves, so a run that spends most of its time frozen would otherwise report a near-trivial,
    /// inflated exact-fraction (confirmed by the reviewer's own measurement: mostly-frozen data
    /// scored `>= 0.999` almost by construction, not because prediction itself is that good).
    /// Other characters' groups are unaffected (freeze filtering was never the concern there —
    /// only the own-tee bar has a pass/fail threshold at all).
    pub fn summarize(&self) -> Vec<HorizonSummary> {
        // A sample "counts" at all iff it isn't an own-tee sample frozen at either end.
        let counts = |s: &Sample| !s.is_own || (!s.base_frozen && !s.target_frozen);

        let mut horizons: Vec<(bool, i32)> = self
            .samples
            .iter()
            .filter(|s| counts(s))
            .map(|s| (s.is_own, s.horizon_ticks))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        horizons.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

        horizons
            .into_iter()
            .map(|(is_own, horizon_ticks)| {
                let matches = |s: &&Sample| s.is_own == is_own && s.horizon_ticks == horizon_ticks && counts(s);
                let mut errors: Vec<f32> = self.samples.iter().filter(matches).map(|s| s.error_px).collect();
                errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let exact_count = self.samples.iter().filter(matches).filter(|s| s.exact).count();
                let n = errors.len();
                let percentile = |p: f64| -> f32 {
                    if n == 0 {
                        return 0.0;
                    }
                    let idx = ((p * (n as f64 - 1.0)).round() as usize).min(n - 1);
                    errors[idx]
                };
                HorizonSummary {
                    is_own,
                    horizon_ticks,
                    count: n,
                    exact_fraction: if n == 0 { 0.0 } else { exact_count as f64 / n as f64 },
                    mean_error_px: if n == 0 {
                        0.0
                    } else {
                        errors.iter().map(|&e| e as f64).sum::<f64>() / n as f64
                    },
                    p50_error_px: percentile(0.50),
                    p90_error_px: percentile(0.90),
                    p99_error_px: percentile(0.99),
                    max_error_px: errors.last().copied().unwrap_or(0.0),
                }
            })
            .collect()
    }
}

/// One `(is_own, horizon_ticks)` group's summary statistics — see [`AccuracyTracker::summarize`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HorizonSummary {
    pub is_own: bool,
    pub horizon_ticks: i32,
    pub count: usize,
    /// Fraction of samples with a bit-exact position match (task spec's own-tee bar: `>= 0.999`).
    pub exact_fraction: f64,
    pub mean_error_px: f64,
    pub p50_error_px: f32,
    pub p90_error_px: f32,
    pub p99_error_px: f32,
    pub max_error_px: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn core_at(pos: Vec2<f32>) -> ddai_physics::core::CharacterCore<f32> {
        let mut c = ddai_physics::core::CharacterCore::<f32>::default();
        c.pos = pos;
        c
    }

    /// `frozen`: review round 3, finding F13 — sets `world.characters[id].freeze_time` so
    /// [`AccuracyTracker::record_actual`]'s `target_frozen` derivation has something real to read;
    /// every pre-F13 test below passes `false` (unfrozen), matching this fn's old, no-`Character`-
    /// at-all behavior exactly (`record_actual`'s `unwrap_or(false)` fallback covered that case the
    /// same way `false` here does).
    fn world_with(id: i32, pos: Vec2<f32>, frozen: bool) -> World<f32> {
        let map = ddai_physics::map::MapData {
            width: 2,
            height: 2,
            game: vec![Default::default(); 4],
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        };
        let mut world: World<f32> = World::from_map(&map, 1);
        world.cores.insert(id as u8, core_at(pos));
        world.characters[id as usize] = Some(ddai_physics::world::Character {
            alive: true,
            freeze_time: if frozen { 10 } else { 0 },
            ..Default::default()
        });
        world
    }

    #[test]
    fn exact_match_is_flagged_exact_with_zero_error() {
        let mut tracker = AccuracyTracker::new(0);
        tracker.record_prediction(10, 11, 0, Vec2::new(100.0, 200.0), false);
        let world = world_with(0, Vec2::new(100.0, 200.0), false);
        tracker.record_actual(11, &world);
        let samples = tracker.samples();
        assert_eq!(samples.len(), 1);
        assert!(samples[0].exact);
        assert_eq!(samples[0].error_px, 0.0);
        assert_eq!(samples[0].horizon_ticks, 1);
        assert!(samples[0].is_own);
    }

    #[test]
    fn mismatch_reports_nonzero_pixel_error_and_not_exact() {
        let mut tracker = AccuracyTracker::new(5);
        tracker.record_prediction(10, 15, 1, Vec2::new(0.0, 0.0), false);
        let world = world_with(1, Vec2::new(3.0, 4.0), false);
        tracker.record_actual(15, &world);
        let samples = tracker.samples();
        assert_eq!(samples.len(), 1);
        assert!(!samples[0].exact);
        assert_eq!(samples[0].error_px, 5.0); // 3-4-5 triangle
        assert!(!samples[0].is_own);
    }

    #[test]
    fn departed_character_is_dropped_without_a_sample() {
        let mut tracker = AccuracyTracker::new(0);
        tracker.record_prediction(10, 11, 7, Vec2::new(0.0, 0.0), false);
        let world = world_with(0, Vec2::new(0.0, 0.0), false); // id 7 never inserted
        tracker.record_actual(11, &world);
        assert!(tracker.samples().is_empty());
        assert_eq!(
            tracker.pending_count(),
            0,
            "resolved (dropped) either way, not left pending forever"
        );
    }

    #[test]
    fn unmatched_prediction_is_pruned_once_far_enough_stale() {
        let mut tracker = AccuracyTracker::new(0);
        tracker.record_prediction(10, 11, 0, Vec2::new(0.0, 0.0), false);
        let world = world_with(0, Vec2::new(0.0, 0.0), false);
        // Never actually confirm tick 11 itself; skip straight past the staleness window.
        tracker.record_actual(11 + STALE_AFTER_TICKS + 1, &world);
        assert_eq!(tracker.pending_count(), 0);
        assert!(tracker.samples().is_empty());
    }

    /// Review round 1, finding F9: `samples` must have a fixed memory ceiling — the oldest
    /// resolved sample is evicted once at capacity, not kept forever.
    #[test]
    fn samples_are_bounded_oldest_evicted_first() {
        let mut tracker = AccuracyTracker::new(0);
        // One `World` reused for every tick (rebuilding it from a map 50,000+ times would just
        // slow the test down for no reason — only its own character's position changes here).
        let mut world = world_with(0, Vec2::new(0.0, 0.0), false);
        for tick in 1..=(MAX_SAMPLES + 10) as i32 {
            tracker.record_prediction(tick - 1, tick, 0, Vec2::new(tick as f32, 0.0), false);
            world.cores.get_mut(0).unwrap().pos = Vec2::new(tick as f32, 0.0);
            tracker.record_actual(tick, &world);
        }
        assert_eq!(tracker.samples().len(), MAX_SAMPLES, "must never exceed the cap");
        // The oldest 10 (predicted_pos.x == 1..=10) must be gone; the newest must still be there.
        assert!(tracker.samples().iter().all(|s| s.predicted.x > 10.0));
        assert_eq!(tracker.samples().back().unwrap().predicted.x, (MAX_SAMPLES + 10) as f32);
    }

    #[test]
    fn summarize_groups_by_own_flag_and_horizon_and_computes_percentiles() {
        let mut tracker = AccuracyTracker::new(0);
        for (i, err) in [1.0f32, 2.0, 3.0, 4.0, 5.0].into_iter().enumerate() {
            tracker.record_prediction(0, 1, 1, Vec2::new(0.0, 0.0), false);
            let world = world_with(1, Vec2::new(err, 0.0), false);
            tracker.record_actual(1, &world);
            let _ = i;
        }
        let summary = tracker.summarize();
        assert_eq!(summary.len(), 1);
        let h = summary[0];
        assert!(!h.is_own);
        assert_eq!(h.horizon_ticks, 1);
        assert_eq!(h.count, 5);
        assert_eq!(h.max_error_px, 5.0);
        assert_eq!(h.mean_error_px, 3.0);
    }

    // --- Review round 3, finding F13 ------------------------------------------------------------

    /// An own-tee sample frozen at `base_tick` must never enter [`AccuracyTracker::summarize`]'s
    /// own-tee accounting at all — not even as a (trivially exact) zero-error sample.
    #[test]
    fn summarize_excludes_an_own_tee_sample_frozen_at_base_tick() {
        let mut tracker = AccuracyTracker::new(0);
        tracker.record_prediction(10, 11, 0, Vec2::new(5.0, 5.0), true); // base_frozen = true.
        let world = world_with(0, Vec2::new(5.0, 5.0), false); // exact match, unfrozen at target.
        tracker.record_actual(11, &world);
        assert_eq!(tracker.samples().len(), 1, "the sample itself is still recorded...");
        assert!(
            tracker.summarize().is_empty(),
            "...but must not surface in summarize()'s own-tee accounting"
        );
    }

    /// Symmetric case: unfrozen at `base_tick` but frozen by the time `target_tick` confirms it —
    /// equally excluded (freezing *during* the predicted window is just as trivially "easy" to
    /// predict as starting frozen).
    #[test]
    fn summarize_excludes_an_own_tee_sample_frozen_at_target_tick() {
        let mut tracker = AccuracyTracker::new(0);
        tracker.record_prediction(10, 11, 0, Vec2::new(5.0, 5.0), false); // base_frozen = false.
        let world = world_with(0, Vec2::new(5.0, 5.0), true); // frozen at target.
        tracker.record_actual(11, &world);
        assert!(tracker.summarize().is_empty());
    }

    /// A *non*-own character's frozen samples are unaffected by F13's filter — freeze exclusion
    /// only ever applies to the own-tee accuracy bar.
    #[test]
    fn summarize_does_not_filter_frozen_samples_for_other_characters() {
        let mut tracker = AccuracyTracker::new(99); // own id is 99, so id 0 below is "other".
        tracker.record_prediction(10, 11, 0, Vec2::new(5.0, 5.0), true);
        let world = world_with(0, Vec2::new(5.0, 5.0), true);
        tracker.record_actual(11, &world);
        let summary = tracker.summarize();
        assert_eq!(summary.len(), 1);
        assert!(!summary[0].is_own);
        assert_eq!(summary[0].count, 1);
    }

    /// A mixed batch: only the unfrozen-at-both-ends own-tee samples are counted; frozen ones are
    /// silently excluded from `count`/`exact_fraction`, not treated as zero-count-but-still-there.
    #[test]
    fn summarize_counts_only_unfrozen_at_both_ends_own_tee_samples() {
        let mut tracker = AccuracyTracker::new(0);
        // Two unfrozen, exact samples...
        for _ in 0..2 {
            tracker.record_prediction(0, 1, 0, Vec2::new(1.0, 1.0), false);
            let world = world_with(0, Vec2::new(1.0, 1.0), false);
            tracker.record_actual(1, &world);
        }
        // ...and three frozen-at-base samples that would otherwise inflate the exact fraction.
        for _ in 0..3 {
            tracker.record_prediction(0, 1, 0, Vec2::new(2.0, 2.0), true);
            let world = world_with(0, Vec2::new(2.0, 2.0), false);
            tracker.record_actual(1, &world);
        }
        let summary = tracker.summarize();
        assert_eq!(summary.len(), 1);
        assert_eq!(summary[0].count, 2, "only the two unfrozen-at-both-ends samples count");
        assert_eq!(summary[0].exact_fraction, 1.0);
    }
}
