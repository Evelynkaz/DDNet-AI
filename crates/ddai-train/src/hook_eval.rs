//! What the focal player does with its hook after the freeze, on the post-freeze starts of the bank (task 8.6): the hook-start rate by own
//! hook state, the timing of the first press, and the aim at the throw, split by start class (V / B / H) and meant to be read next to
//! the planner's on the same starts (60% hook start rate; it opens with a hook in about 80% of its first decisions on V starts).
//!
//! It plays the same episodes as [`crate::es::eval::eval_starts`] (same starts, same replay, the focal brain takes over at the freeze) with
//! a [`RecordingBrain`] around the brain: every decision from the handover tick on is recorded (the observed own hook state, the hook key
//! the brain pressed, the aim of a throw against the bearing to the victim). Nothing the recorder does reaches the game.

use std::sync::{Arc, Mutex};

use ddai_brain::{Action, Brain, HOOK_FLYING, HOOK_GRABBED, HOOK_IDLE, Observation, ResetContext, WorldView};
use ddai_env::EnvError;
use ddai_env::config::Rules;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::bank::{BankStart, play_from_start};
use crate::es::eval::BrainMaker;
use crate::experiment::Env;
use crate::heldblock::EpisodeOutcome;
use crate::types::ring_angle_of_target;

/// One decision of the focal player after the handover.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HookDecision {
    /// Ticks since the freeze (the handover tick).
    pub tick: i32,
    /// The observed own hook state (`HOOK_*`).
    pub state: i32,
    /// The hook key the brain pressed.
    pub hook: bool,
    /// For a throw (the key pressed while the state is idle): the angle between the aim and the bearing to the victim, radians.
    pub throw_err: Option<f32>,
}

/// Records the decisions of a brain from `handover` on, and passes everything through.
pub struct RecordingBrain {
    inner: Box<dyn Brain>,
    handover: i32,
    sink: Arc<Mutex<Vec<HookDecision>>>,
}

impl RecordingBrain {
    pub fn new(inner: Box<dyn Brain>, handover: i32, sink: Arc<Mutex<Vec<HookDecision>>>) -> Self {
        RecordingBrain { inner, handover, sink }
    }
}

fn wrap_pi(a: f32) -> f32 {
    a.sin().atan2(a.cos())
}

impl Brain for RecordingBrain {
    fn reset(&mut self, ctx: &ResetContext) {
        self.inner.reset(ctx);
    }

    fn decide(&mut self, obs: &Observation) -> Action {
        self.decide_in(obs, None)
    }

    fn decide_in(&mut self, obs: &Observation, view: Option<&WorldView<'_>>) -> Action {
        let a = self.inner.decide_in(obs, view);
        if obs.tick >= self.handover {
            let state = obs.self_state.hook_state;
            let throw_err = (a.hook && state == HOOK_IDLE)
                .then(|| {
                    let me = obs.self_state.pos;
                    // The victim is the one other character of a post-freeze episode.
                    let victim = obs.others.first()?;
                    let bearing = (-(victim.pos.y - me.y)).atan2(victim.pos.x - me.x);
                    let aim = ring_angle_of_target([a.target.x, a.target.y]);
                    Some(wrap_pi(aim - bearing).abs())
                })
                .flatten();
            self.sink.lock().expect("sink lock").push(HookDecision {
                tick: obs.tick - self.handover,
                state,
                hook: a.hook,
                throw_err,
            });
        }
        a
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn telemetry(&self) -> Option<String> {
        self.inner.telemetry()
    }
}

/// The start class of the 8.5a review (`victim_escapes_under_idle` / `idle_held` tags): V the victim escapes under an idle blocker, H an
/// idle blocker holds the block, B the rest (the idle blocker itself falls).
pub fn start_class(s: &BankStart) -> char {
    if s.victim_escapes_under_idle == Some(true) {
        'V'
    } else if s.idle_held {
        'H'
    } else {
        'B'
    }
}

/// The recorded episode of one start.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StartRecord {
    pub class: char,
    pub decisions: Vec<HookDecision>,
    pub outcome_held: bool,
    pub outcome_self_out: bool,
}

/// Plays the `starts` (in order) with a recording around every focal brain from `maker`.
pub fn record_starts(
    env: &Env,
    pool: &rayon::ThreadPool,
    starts: &[&BankStart],
    rules: &Rules,
    maker: &BrainMaker<'_>,
    window: i32,
    burn_in: i32,
) -> Result<Vec<StartRecord>, String> {
    let r: Vec<Result<StartRecord, EnvError>> = pool.install(|| {
        starts
            .par_iter()
            .map(|s| {
                let arena = env
                    .arenas
                    .get(&s.arena)
                    .ok_or_else(|| EnvError::new(format!("unknown arena {:?}", s.arena)))?;
                let sink = Arc::new(Mutex::new(Vec::new()));
                let focal = Box::new(RecordingBrain::new(maker()?, s.end_tick, sink.clone()));
                let opponent = env.models.factory()(&ddai_env::config::PlayerSpec::simple("scripted"))?;
                let o: EpisodeOutcome = play_from_start(arena, rules, s, focal, opponent, window, burn_in)?;
                let decisions = std::mem::take(&mut *sink.lock().expect("sink lock"));
                Ok(StartRecord {
                    class: start_class(s),
                    decisions,
                    outcome_held: o.held_block,
                    outcome_self_out: o.focal_out_in_window,
                })
            })
            .collect()
    });
    r.into_iter().collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())
}

/// A rate with its count.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Frac {
    pub k: u64,
    pub n: u64,
}

impl Frac {
    pub fn p(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            self.k as f64 / self.n as f64
        }
    }
}

/// The hook of one start class.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HookSummary {
    pub starts: u64,
    pub decisions: u64,
    /// `P(key pressed | observed own hook state)`: idle is the **start rate**; flying / grabbed are the holds; anything else (retracting).
    pub idle: Frac,
    pub flying: Frac,
    pub grabbed: Frac,
    pub other: Frac,
    /// The hook key in the first `OPENING` decisions of a start (any state): the planner's opening is ~80% on V starts.
    pub opening_key: Frac,
    /// A **throw** (key pressed while idle) in the first `OPENING` decisions.
    pub opening_throw: Frac,
    /// Starts whose first press (any state) comes within 8 / 24 / 50 ticks of the freeze, and ever.
    pub first_press_8: Frac,
    pub first_press_24: Frac,
    pub first_press_50: Frac,
    pub first_press_ever: Frac,
    /// Median and mean tick of the first press over the starts that press at all (`NaN` if none).
    pub first_press_median: f64,
    pub first_press_mean: f64,
    /// Throws per start and the aim error at the throws (degrees): the share within 15 and 45 degrees, median, mean.
    pub throws_per_start: f64,
    pub throw_aim_n: u64,
    pub throw_aim_median_deg: f64,
    pub throw_aim_mean_deg: f64,
    pub throw_aim_within_15: f64,
    pub throw_aim_within_45: f64,
}

/// Decisions counted in the opening: 4, as the `open_hook_v` metric of 8.5b.
pub const OPENING: usize = 4;

pub fn summarize(records: &[&StartRecord]) -> HookSummary {
    let mut s = HookSummary {
        starts: records.len() as u64,
        ..HookSummary::default()
    };
    let mut firsts: Vec<f64> = Vec::new();
    let mut errs: Vec<f32> = Vec::new();
    let mut throws = 0u64;
    for r in records {
        for (i, d) in r.decisions.iter().enumerate() {
            s.decisions += 1;
            let bucket = match d.state {
                HOOK_IDLE => &mut s.idle,
                HOOK_FLYING => &mut s.flying,
                HOOK_GRABBED => &mut s.grabbed,
                _ => &mut s.other,
            };
            bucket.n += 1;
            bucket.k += u64::from(d.hook);
            if i < OPENING {
                s.opening_key.n += 1;
                s.opening_key.k += u64::from(d.hook);
                s.opening_throw.n += 1;
                s.opening_throw.k += u64::from(d.hook && d.state == HOOK_IDLE);
            }
            if d.hook && d.state == HOOK_IDLE {
                throws += 1;
            }
            if let Some(e) = d.throw_err {
                errs.push(e);
            }
        }
        let first = r.decisions.iter().find(|d| d.hook).map(|d| d.tick);
        for (frac, limit) in [
            (&mut s.first_press_8, 8),
            (&mut s.first_press_24, 24),
            (&mut s.first_press_50, 50),
        ] {
            frac.n += 1;
            frac.k += u64::from(first.is_some_and(|t| t <= limit));
        }
        s.first_press_ever.n += 1;
        s.first_press_ever.k += u64::from(first.is_some());
        if let Some(t) = first {
            firsts.push(f64::from(t));
        }
    }
    firsts.sort_by(f64::total_cmp);
    s.first_press_median = if firsts.is_empty() {
        f64::NAN
    } else {
        firsts[firsts.len() / 2]
    };
    s.first_press_mean = if firsts.is_empty() {
        f64::NAN
    } else {
        firsts.iter().sum::<f64>() / firsts.len() as f64
    };
    s.throws_per_start = if records.is_empty() {
        f64::NAN
    } else {
        throws as f64 / records.len() as f64
    };
    s.throw_aim_n = errs.len() as u64;
    if !errs.is_empty() {
        errs.sort_by(f32::total_cmp);
        let deg = |x: f32| f64::from(x).to_degrees();
        s.throw_aim_median_deg = deg(errs[errs.len() / 2]);
        s.throw_aim_mean_deg = errs.iter().map(|&e| deg(e)).sum::<f64>() / errs.len() as f64;
        s.throw_aim_within_15 = errs.iter().filter(|&&e| deg(e) <= 15.0).count() as f64 / errs.len() as f64;
        s.throw_aim_within_45 = errs.iter().filter(|&&e| deg(e) <= 45.0).count() as f64 / errs.len() as f64;
    } else {
        s.throw_aim_median_deg = f64::NAN;
        s.throw_aim_mean_deg = f64::NAN;
        s.throw_aim_within_15 = f64::NAN;
        s.throw_aim_within_45 = f64::NAN;
    }
    s
}

/// Per class (`V`, `B`, `H`) and over all starts.
pub fn summarize_by_class(records: &[StartRecord]) -> Vec<(String, HookSummary)> {
    let mut out = Vec::new();
    for (name, want) in [
        ("V", Some('V')),
        ("B", Some('B')),
        ("H", Some('H')),
        ("all", None::<char>),
    ] {
        let sel: Vec<&StartRecord> = records.iter().filter(|r| want.is_none_or(|c| r.class == c)).collect();
        out.push((name.to_string(), summarize(&sel)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(tick: i32, state: i32, hook: bool, err: Option<f32>) -> HookDecision {
        HookDecision {
            tick,
            state,
            hook,
            throw_err: err,
        }
    }

    #[test]
    fn the_summary_counts_the_start_rate_by_state_the_opening_and_the_first_press() {
        let a = StartRecord {
            class: 'V',
            decisions: vec![
                d(0, HOOK_IDLE, false, None),
                d(2, HOOK_IDLE, true, Some(0.1)),
                d(4, HOOK_FLYING, true, None),
                d(6, HOOK_GRABBED, false, None),
                d(8, HOOK_IDLE, true, Some(0.5)),
            ],
            outcome_held: true,
            outcome_self_out: false,
        };
        let b = StartRecord {
            class: 'V',
            decisions: vec![d(0, HOOK_IDLE, false, None), d(2, HOOK_IDLE, false, None)],
            outcome_held: false,
            outcome_self_out: true,
        };
        let s = summarize(&[&a, &b]);
        assert_eq!((s.starts, s.decisions), (2, 7));
        // Idle decisions: 5 (a: 0, 2, 8; b: 0, 2), pressed: 2 (a's decisions at ticks 2 and 8).
        assert_eq!((s.idle.k, s.idle.n), (2, 5));
        assert_eq!((s.flying.k, s.flying.n), (1, 1));
        assert_eq!((s.grabbed.k, s.grabbed.n), (0, 1));
        // The first four decisions of a (key: no, yes, yes, no) and of b (no, no).
        assert_eq!((s.opening_key.k, s.opening_key.n), (2, 6));
        assert_eq!((s.opening_throw.k, s.opening_throw.n), (1, 6));
        // a presses first at tick 2, b never.
        assert_eq!((s.first_press_8.k, s.first_press_8.n), (1, 2));
        assert_eq!((s.first_press_ever.k, s.first_press_ever.n), (1, 2));
        assert_eq!(s.first_press_median, 2.0);
        assert_eq!(s.throws_per_start, 1.0);
        assert_eq!(s.throw_aim_n, 2);
        assert!(
            (s.throw_aim_within_15 - 0.5).abs() < 1e-12,
            "0.1 rad = 5.7 deg is within 15, 0.5 rad = 28.6 is not"
        );
        assert!((s.throw_aim_within_45 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn an_empty_class_summarises_to_nans_not_a_panic() {
        let s = summarize(&[]);
        assert_eq!(s.starts, 0);
        assert!(s.first_press_median.is_nan() && s.throws_per_start.is_nan() && s.idle.p().is_nan());
    }
}
