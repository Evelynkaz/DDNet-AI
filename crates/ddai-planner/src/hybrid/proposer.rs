//! Proposals (task 3.5, D-041): the "fly proposes" half of the brain. A [`Proposer`] hands the
//! search up to `K` candidate plans in the planner's plan encoding; the search scores them with the
//! same exact rollouts as everything else and keeps one only if it wins. Proposals are candidates,
//! never decisions.
//!
//! * [`NoProposer`]: no proposals (the search on its own).
//! * [`ScriptedProposer`]: the scripted bot rolled forward on the planning world and recorded as a
//!   plan. It stands in for a trained fly (and is the "distillation source" shape a trained fly
//!   will imitate).
//! * `FlyProposer` (in `ddai-fly`, implementing this trait): the fly's action distribution turned
//!   into plans by [`plans_from_distribution`].
//!
//! The plan encoding (`docs/research/orig-plan.md` §1.1): `steps` steps of `plan_step` ticks; each
//! step holds a direction, jump, hook, fire and an aim, relative to the direction to the victim
//! when the planner tracks the aim.

use std::sync::Arc;

use ddai_brain::{Observation, ResetContext};
use ddai_jsmath::Rng;
use ddai_physics::map::MapData;

use crate::fields::wrap_angle;
use crate::hybrid::abs_aim;
use crate::physics_adapter::{PhysicsSavedState, PhysicsWorld};
use crate::plan_world::PlanWorld;
use crate::planner::PlanStep;
use crate::scripted::scripted_action;
use crate::types::PlayerInput;
use ddai_jsmath as js;

/// What a proposer may read to make its proposals.
pub struct ProposeCtx<'a> {
    /// The plain-data observation of this decision (what a fly sees).
    pub obs: &'a Observation,
    /// The planning world at the decision state (lag already rolled in). Read-only: a proposer
    /// that wants to simulate copies it (`saved`).
    pub world: &'a PhysicsWorld,
    /// The same state as a restorable snapshot (for a proposer with its own scratch world).
    pub saved: &'a PhysicsSavedState,
    pub map: &'a Arc<MapData>,
    pub self_id: i32,
    pub victim_id: i32,
    /// Plan length and the tick length of each step.
    pub steps: usize,
    pub step_ticks: &'a [i32],
    /// Absolute angle from us to the victim.
    pub aim_at: f64,
    pub track_aim: bool,
    /// The input we sent last.
    pub prev: PlayerInput,
    /// How many proposals the search wants.
    pub k: usize,
}

/// What the exact search did with a proposer's plans in the decision just made (task 7.4): the viewer's
/// "how often is the fly's proposal the one that is played". Counted by the hybrid brain itself since its
/// last reset, so the numbers do not depend on anybody watching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProposalOutcome {
    /// The plan the search played in this decision came from the proposer.
    pub chosen: bool,
    /// Decisions since the last reset whose played plan came from the proposer.
    pub chosen_total: u32,
    /// Decisions since the last reset.
    pub decisions_total: u32,
}

/// A source of candidate plans. Implementations must be deterministic given their seed.
pub trait Proposer {
    fn name(&self) -> &str;
    /// New episode (`ResetContext::seed` seeds any sampling).
    fn reset(&mut self, _ctx: &ResetContext) {}
    /// Appends up to `ctx.k` plans of exactly `ctx.steps` steps to `out`.
    fn propose(&mut self, ctx: &ProposeCtx<'_>, out: &mut Vec<Vec<PlanStep>>);
    /// Physics ticks this proposer has simulated in total (a work counter, D-045; a proposer that
    /// does not simulate reports `0`, its cost is then only wall time).
    fn work_ticks(&self) -> u64 {
        0
    }
    /// Task 7.4: the static description of the proposer's visualisation stream (see
    /// [`ddai_brain::Brain::viz_meta`]); `None`: it has none (the default).
    fn viz_meta(&self) -> Option<String> {
        None
    }
    /// Task 7.4: the visualisation frame of the proposal made in the latest `propose` call, with what the
    /// search did with it. Called only while somebody watches, once per decision, after the search has
    /// decided; `None` when there is no stream or no new proposal since the last call. Must not change what
    /// the next `propose` returns (the stream is read-only).
    fn viz_frame(&mut self, tick: u32, outcome: Option<ProposalOutcome>) -> Option<&[u8]> {
        let _ = (tick, outcome);
        None
    }
}

/// Proposes nothing.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoProposer;

impl Proposer for NoProposer {
    fn name(&self) -> &str {
        "none"
    }
    fn propose(&mut self, _ctx: &ProposeCtx<'_>, _out: &mut Vec<Vec<PlanStep>>) {}
}

/// The scripted bot (`scriptedAction`) playing for us on a scratch copy of the planning world for
/// the whole plan, recorded step by step. The victim keeps the input the planning world holds.
/// Proposal `i > 0` re-rolls the scripted aim noise with a different draw, so `K` proposals are
/// `K` slightly different scripted lines.
pub struct ScriptedProposer {
    rng_seed: u32,
    scratch: Option<PhysicsWorld>,
    /// Physics ticks simulated while proposing (work counter, D-045).
    pub ticks: u64,
}

impl ScriptedProposer {
    pub fn new() -> Self {
        ScriptedProposer {
            rng_seed: 17,
            scratch: None,
            ticks: 0,
        }
    }
}

impl Default for ScriptedProposer {
    fn default() -> Self {
        Self::new()
    }
}

/// A recorded plan step from a wire-format input: `aim_base` is the direction to the victim when
/// the aim is relative, `0` otherwise.
fn step_from_input(i: &PlayerInput, aim_base: f64) -> PlanStep {
    let angle = js::atan2(i.target_y, i.target_x);
    PlanStep {
        dir: i.direction,
        jump: i32::from(i.jump != 0),
        hook: i32::from(i.hook != 0),
        fire: i32::from((i.fire & 1) != 0),
        aim: wrap_angle(angle - aim_base),
    }
}

impl Proposer for ScriptedProposer {
    fn name(&self) -> &str {
        "scripted"
    }

    fn reset(&mut self, ctx: &ResetContext) {
        self.rng_seed = ctx.seed.wrapping_mul(7919).wrapping_add(17) as u32;
        self.scratch = None;
    }

    fn work_ticks(&self) -> u64 {
        self.ticks
    }

    fn propose(&mut self, ctx: &ProposeCtx<'_>, out: &mut Vec<Vec<PlanStep>>) {
        if ctx.k == 0 || ctx.steps == 0 {
            return;
        }
        if self.scratch.is_none() {
            self.scratch = Some(PhysicsWorld::from_world(ctx.world.inner().clone(), ctx.map.clone()));
        }
        let scratch = self.scratch.as_mut().expect("just set");
        for i in 0..ctx.k {
            scratch.restore_state(ctx.saved);
            let mut rng = Rng::new(self.rng_seed.wrapping_add(i as u32 * 7919));
            let mut input = ctx.prev;
            let mut plan = Vec::with_capacity(ctx.steps);
            for s in 0..ctx.steps {
                let to_victim = match (scratch.get_tee(ctx.self_id), scratch.get_tee(ctx.victim_id)) {
                    (Some(m), Some(v)) => js::atan2(v.pos.y - m.pos.y, v.pos.x - m.pos.x),
                    _ => ctx.aim_at,
                };
                input = scripted_action(&*scratch, ctx.self_id, ctx.victim_id, &input, &mut rng);
                plan.push(step_from_input(&input, if ctx.track_aim { to_victim } else { 0.0 }));
                for _ in 0..ctx.step_ticks[s.min(ctx.step_ticks.len() - 1)] {
                    scratch.set_input(ctx.self_id, input);
                    scratch.step();
                    self.ticks += 1;
                }
            }
            out.push(plan);
        }
        scratch.restore_state(ctx.saved);
    }
}

/// A decoded policy output for one decision: direction probabilities `[left, stop, right]`,
/// jump/hook/fire probabilities and the aim as an absolute angle (y down, `atan2(dy, dx)`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActionDistribution {
    pub direction: [f64; 3],
    pub jump: f64,
    pub hook: f64,
    pub fire: f64,
    pub aim_angle: f64,
}

/// Turns one decision's action distribution into `k` plans (task 3.5 criterion 1). The first plan
/// is the **argmax** plan: the most likely direction held for the whole plan, the jump pressed on
/// the first step if it is likely, the hook held for the first two thirds and the hammer swung
/// from the second step if they are likely, the aim absolute (a policy aims at a place, not
/// relative to the victim). The others **sample** the same heads step by step, with each draw
/// held for three steps so the plan stays a coherent move rather than noise. `uniform` draws from
/// `[0, 1)`: the caller's seeded generator (so no particular RNG crate is imposed on a proposer).
pub fn plans_from_distribution(
    d: &ActionDistribution,
    steps: usize,
    k: usize,
    uniform: &mut dyn FnMut() -> f64,
    out: &mut Vec<Vec<PlanStep>>,
) {
    if k == 0 || steps == 0 {
        return;
    }
    let dir_of = |idx: usize| [-1, 0, 1][idx];
    let argmax = (0..3)
        .max_by(|&a, &b| d.direction[a].total_cmp(&d.direction[b]))
        .unwrap_or(1);
    let aim = abs_aim(d.aim_angle);
    let hook_steps = (2 * steps) / 3;
    out.push(
        (0..steps)
            .map(|s| PlanStep {
                dir: dir_of(argmax),
                jump: i32::from(s == 0 && d.jump >= 0.5),
                hook: i32::from(d.hook >= 0.5 && s < hook_steps),
                fire: i32::from(d.fire >= 0.5 && s >= 1),
                aim,
            })
            .collect(),
    );
    for _ in 1..k {
        let mut plan = Vec::with_capacity(steps);
        let mut cur = PlanStep {
            dir: 0,
            jump: 0,
            hook: 0,
            fire: 0,
            aim,
        };
        for s in 0..steps {
            if s % 3 == 0 {
                let r = uniform();
                let idx = if r < d.direction[0] {
                    0
                } else if r < d.direction[0] + d.direction[1] {
                    1
                } else {
                    2
                };
                cur.dir = dir_of(idx);
                cur.hook = i32::from(uniform() < d.hook);
                cur.fire = i32::from(uniform() < d.fire);
            }
            cur.jump = i32::from(s == 0 && uniform() < d.jump);
            plan.push(cur);
        }
        out.push(plan);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dist() -> ActionDistribution {
        ActionDistribution {
            direction: [0.1, 0.2, 0.7],
            jump: 0.9,
            hook: 0.8,
            fire: 0.1,
            aim_angle: -1.0,
        }
    }

    #[test]
    fn the_first_proposal_is_the_argmax_plan() {
        let mut rng = Rng::new(1);
        let mut out = Vec::new();
        plans_from_distribution(&dist(), 9, 3, &mut || rng.next_float(), &mut out);
        assert_eq!(out.len(), 3);
        for p in &out {
            assert_eq!(p.len(), 9);
        }
        let a = &out[0];
        assert!(a.iter().all(|s| s.dir == 1));
        assert_eq!(a[0].jump, 1);
        assert!(a[1..].iter().all(|s| s.jump == 0));
        assert_eq!(a.iter().filter(|s| s.hook == 1).count(), 6);
        assert!(a.iter().all(|s| s.fire == 0), "fire probability is below one half");
        assert!(crate::hybrid::is_abs_aim(a[0].aim));
        assert!((crate::hybrid::resolve_aim(a[0].aim, 0.0) - -1.0).abs() < 1e-12);
    }

    #[test]
    fn sampling_is_reproducible_and_follows_the_probabilities() {
        let sample = |seed: u32| {
            let mut rng = Rng::new(seed);
            let mut out = Vec::new();
            plans_from_distribution(&dist(), 9, 200, &mut || rng.next_float(), &mut out);
            out
        };
        assert_eq!(
            sample(5).iter().map(|p| p[3].dir).collect::<Vec<_>>(),
            sample(5).iter().map(|p| p[3].dir).collect::<Vec<_>>()
        );
        let right = sample(5).iter().skip(1).filter(|p| p[0].dir == 1).count();
        assert!((110..=165).contains(&right), "right in {right} of 199 samples");
        // A draw is held for three steps.
        for p in sample(9).iter().skip(1) {
            assert_eq!(p[0].dir, p[1].dir);
            assert_eq!(p[1].dir, p[2].dir);
        }
    }

    #[test]
    fn zero_k_or_steps_is_empty() {
        let mut rng = Rng::new(1);
        let mut out = Vec::new();
        plans_from_distribution(&dist(), 9, 0, &mut || rng.next_float(), &mut out);
        plans_from_distribution(&dist(), 0, 3, &mut || rng.next_float(), &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn a_recorded_input_becomes_a_relative_step() {
        let i = PlayerInput {
            direction: -1,
            target_x: 0.0,
            target_y: -300.0,
            jump: 1,
            fire: 3,
            hook: 1,
            ..crate::types::empty_input()
        };
        let s = step_from_input(&i, -std::f64::consts::FRAC_PI_2 + 0.5);
        assert_eq!((s.dir, s.jump, s.hook, s.fire), (-1, 1, 1, 1));
        assert!((s.aim - -0.5).abs() < 1e-9, "aim {}", s.aim);
    }
}
