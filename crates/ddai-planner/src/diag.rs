//! Loss-diagnosis tooling (task 3.7b, E-017): a [`Brain`] that plays exactly like a [`HybridBrain`] while a fixed-iteration
//! planner (the competitor) shadows it from the same seat on the same world, and that records, for every decision, what
//! both of them did and how the hybrid's own evaluator scores the planner's plan.
//!
//! * **Shadow.** At every decision the planner decides on the hybrid's world with the hybrid's last input as its `prev`
//!   (`TeacherPlanner::note_executed`), so "what the fixed planner would have done in this seat" is exact. Its answer
//!   is recorded and never played.
//! * **Evaluator.** The pool of the hybrid's decision (`HybridConfig::debug_pool`) is recorded with its scores, and the
//!   planner's plan is scored by the same evaluator ([`HybridBrain::debug_score`]).
//! * **Truth.** For decisions inside [`DiagOptions::window`] the true world is kept; after the game
//!   [`DiagInner::finalize`] rolls the hybrid's chosen plan and the planner's plan forward in it against the opponent's
//!   *actual* inputs (read from the world at the following decisions) and reports who went out when -- the planner's plan, and
//!   every plan of the hybrid's pool. The decision's context is kept too, so that the hybrid's evaluator can be told the
//!   opponent's actual inputs (`HybridBrain::debug_oracle`): the oracle that separates a wrong opponent model from a wrong
//!   evaluation.
//! * **Opponent as a planner.** With [`DiagOptions::mirror`] a second fixed planner decides from the *opponent's* seat; its plan
//!   becomes open-loop inputs ([`HybridBrain::debug_plan_inputs`]) that the evaluator can be told instead (`oracle_mirror`): the
//!   prototype of the hybrid's opponent model (`HybridConfig::mirror`).
//!
//! Nothing here is on the production path: the hybrid inside decides exactly as it would alone (checked by the hashes of
//! the replayed games against the original run), the diagnostics read its state after the decision.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::Arc;

use ddai_brain::{Action, Brain, Observation, ResetContext, WorldView};
use ddai_physics::world::World;

use crate::brains::{action_from_input, input_from_action, target_of};
use crate::hybrid::HybridBrain;
use crate::hybrid::engine::Ctx;
use crate::hybrid::search::{DebugScore, DecisionTelemetry, TruthOutcome};
use crate::physics_adapter::from_ddnet_input;
use crate::planner::PlanStep;
use crate::teacher::TeacherPlanner;
use crate::types::PlayerInput;

/// What a diagnosing brain does besides playing.
#[derive(Debug, Clone, Copy, Default)]
pub struct DiagOptions {
    /// Keep the true world (and run the truth rollouts) of decisions on ticks `lo..=hi`.
    pub window: Option<(i32, i32)>,
    /// Also run a second fixed planner from the *opponent's* seat on every decision's world (its input held as the opponent's
    /// previous input), to see how well "the opponent is a planner like us" predicts what it really does.
    pub mirror: bool,
}

/// A tee as the diagnosis wants to remember it.
#[derive(Debug, Clone, Copy, Default)]
pub struct TeeSnap {
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub frozen: bool,
    pub hook_state: i32,
    pub hooked_player: i32,
    pub jumps_left: i32,
    pub freeze_ticks_left: i64,
}

/// Everything recorded about one decision.
pub struct DecisionRecord {
    /// Index among the decisions the hybrid searched.
    pub idx: usize,
    pub tick: i32,
    pub self_id: i32,
    pub victim_id: i32,
    pub me: TeeSnap,
    pub victim: TeeSnap,
    /// The hybrid's own answer and the planner's, as played inputs would be (the planner's `prev` was the hybrid's).
    pub hybrid_action: Action,
    pub teacher_action: Action,
    /// The hybrid's telemetry with its whole pool (`debug_pool`).
    pub hybrid: DecisionTelemetry,
    /// The planner's chosen plan and its score under the hybrid's evaluator.
    pub teacher_plan: Option<Vec<PlanStep>>,
    pub teacher_score: Option<DebugScore>,
    /// `DiagOptions::mirror`: the planner's answer from the opponent's seat, and the hold model's prediction (the opponent keeps its
    /// direction, its hook if one is out; no jump, no fire). The opponent's actual action is `DiagInner::opp_action_at`.
    pub mirror_action: Option<Action>,
    pub hold_action: Option<Action>,
    /// The planner's plan from the opponent's seat, the open-loop inputs (per plan step) that plan sends against us holding our
    /// input, and how the hybrid's pool scores when the opponent is told to play them (`oracle_mirror`, like `oracle`).
    pub mirror_plan: Option<Vec<PlanStep>>,
    pub mirror_inputs: Vec<PlayerInput>,
    pub oracle_mirror: Vec<Option<f64>>,
    /// The true world at the decision and our in-flight inputs (inside the window only).
    pub world: Option<Box<World<f32>>>,
    pub in_flight: Vec<ddai_physics::core::PlayerInput>,
    pub prev: PlayerInput,
    /// Truth rollouts (after `finalize`, inside the window): the hybrid's chosen plan and the planner's.
    pub truth_hybrid: Option<TruthOutcome>,
    pub truth_teacher: Option<TruthOutcome>,
    /// The hybrid's context for this decision (inside the window only), kept to score plans in it later.
    pub ctx: Option<Arc<Ctx>>,
    /// After `finalize`: what every plan of the hybrid's pool does in the true world (index-aligned with
    /// `hybrid.pool`) and the score the hybrid's evaluator gives it when it is told the opponent's actual inputs
    /// (pool order, the planner's plan last, `None` = not scored).
    pub truth_pool: Vec<TruthOutcome>,
    pub oracle: Vec<Option<f64>>,
}

/// The state behind a [`DiagBrain`]; the driver keeps an `Rc` to it to read the records after the game.
pub struct DiagInner {
    hybrid: HybridBrain,
    teacher: TeacherPlanner,
    mirror: TeacherPlanner,
    opts: DiagOptions,
    pub records: Vec<DecisionRecord>,
    /// The opponent's input applied at each world tick (read from the world at the next decision).
    opp_applied: BTreeMap<i32, PlayerInput>,
    last_decision_tick: Option<i32>,
    searched: usize,
    /// The input played at the previous decision (the planner's and the hybrid's `prev`).
    played_prev: PlayerInput,
}

impl DiagInner {
    pub fn new(hybrid: HybridBrain, teacher: TeacherPlanner, opts: DiagOptions) -> DiagInner {
        DiagInner {
            hybrid,
            teacher,
            mirror: TeacherPlanner::new(crate::brains::PlannerPreset::Normal),
            opts,
            records: Vec::new(),
            opp_applied: BTreeMap::new(),
            last_decision_tick: None,
            searched: 0,
            played_prev: crate::types::empty_input(),
        }
    }

    /// The opponent's input at world tick `tick` (the nearest recorded one at or before it; before the first, the
    /// first).
    fn opp_at(&self, tick: i32) -> PlayerInput {
        self.opp_applied
            .range(..=tick)
            .next_back()
            .or_else(|| self.opp_applied.iter().next())
            .map_or_else(crate::types::empty_input, |(_, v)| *v)
    }

    /// The action the opponent actually played at the decision made on world tick `tick` (known once a later decision has
    /// seen its input in the world).
    pub fn opp_action_at(&self, tick: i32) -> Option<Action> {
        self.opp_applied.get(&tick).map(action_from_input)
    }

    /// The opponent's applied input at world tick `tick`, if it has been seen.
    pub fn opp_input_at(&self, tick: i32) -> Option<PlayerInput> {
        self.opp_applied.get(&tick).copied()
    }

    fn decide_in_impl(&mut self, obs: &Observation, view: &WorldView<'_>) -> Action {
        let tick = view.world.tick;
        // The opponent's input applied since the previous decision is in the world now.
        let victim_id = target_of(obs).map_or(-1, |t| t.id);
        if victim_id >= 0
            && let Some(Some(c)) = usize::try_from(victim_id)
                .ok()
                .and_then(|i| view.world.characters.get(i))
        {
            let seen = from_ddnet_input(&c.input);
            let from = self.last_decision_tick.unwrap_or(tick - 2);
            for t in from..tick {
                self.opp_applied.insert(t, seen);
            }
        }
        self.last_decision_tick = Some(tick);

        let h_action = self.hybrid.decide_in(obs, Some(view));
        let hybrid_tel = self.hybrid.last_decision().cloned();
        let label = self.teacher.label(obs, view);
        let teacher_plan = self.teacher.last_plan();
        let teacher_score = match (&teacher_plan, &hybrid_tel) {
            (Some(p), Some(_)) => self
                .hybrid
                .debug_score(std::slice::from_ref(p))
                .and_then(|mut v| v.pop()),
            _ => None,
        };

        let mut mirror_plan = None;
        let (mirror_action, hold_action) = if self.opts.mirror && victim_id >= 0 {
            // The opponent's previous action, as the world shows it, is the planner's `prev`.
            let opp_prev = view
                .world
                .characters
                .get(victim_id as usize)
                .and_then(|c| c.as_ref())
                .map(|c| Action::from_player_input(&c.input));
            if let Some(a) = opp_prev {
                self.mirror.note_executed(&a);
            }
            let victim_obs = obs.others.iter().find(|c| c.id == victim_id).cloned();
            let mirror_action = victim_obs.as_ref().map(|v| {
                let opp_obs = Observation {
                    map: obs.map.clone(),
                    tick: obs.tick,
                    self_state: *v,
                    others: vec![obs.self_state],
                    target_id: Some(view.self_id),
                    tuning: obs.tuning,
                };
                let opp_view = WorldView {
                    world: view.world,
                    self_id: victim_id,
                    lag_ticks: 0,
                    in_flight: &[],
                };
                let a = self.mirror.label(&opp_obs, &opp_view).action;
                mirror_plan = self.mirror.last_plan();
                a
            });
            let hold = victim_obs.as_ref().map(|v| Action {
                direction: v.direction,
                jump: false,
                hook: v.hook_state > 0,
                fire: false,
                target: ddai_brain::IVec2::new(0, 0),
                wanted_weapon: None,
            });
            (mirror_action, hold)
        } else {
            (None, None)
        };

        let mut idx = None;
        let played = h_action;
        if hybrid_tel.is_some() {
            idx = Some(self.searched);
            self.searched += 1;
        }
        let prev_before = self.played_prev;
        let played_input = input_from_action(&played, &prev_before);
        // The planner carries on from what was really played (the hybrid's answer).
        self.teacher.note_executed(&played);
        self.played_prev = played_input;

        if let (Some(idx), Some(tel)) = (idx, hybrid_tel) {
            let in_window = self.opts.window.is_some_and(|(lo, hi)| (lo..=hi).contains(&tick));
            let tee = |id: i32| {
                obs.others
                    .iter()
                    .chain(std::iter::once(&obs.self_state))
                    .find(|c| c.id == id)
                    .map(|c| TeeSnap {
                        x: f64::from(c.pos.x),
                        y: f64::from(c.pos.y),
                        vx: f64::from(c.vel.x),
                        vy: f64::from(c.vel.y),
                        frozen: c.is_frozen,
                        hook_state: c.hook_state,
                        hooked_player: c.hooked_player,
                        jumps_left: c.jumps_left,
                        freeze_ticks_left: i64::from(c.freeze_ticks_remaining),
                    })
                    .unwrap_or_default()
            };
            self.records.push(DecisionRecord {
                idx,
                tick,
                self_id: view.self_id,
                victim_id,
                me: tee(view.self_id),
                victim: tee(victim_id),
                hybrid_action: h_action,
                teacher_action: label.action,
                hybrid: tel,
                teacher_plan,
                teacher_score,
                mirror_action,
                hold_action,
                mirror_plan,
                mirror_inputs: Vec::new(),
                oracle_mirror: Vec::new(),
                world: in_window.then(|| Box::new(view.world.clone())),
                in_flight: view.in_flight.to_vec(),
                prev: prev_before,
                truth_hybrid: None,
                truth_teacher: None,
                ctx: in_window.then(|| self.hybrid.debug_ctx()).flatten(),
                truth_pool: Vec::new(),
                oracle: Vec::new(),
            });
        }
        played
    }

    /// Runs the truth rollouts of every record that kept its world. Call once, after the game.
    pub fn finalize(&mut self) {
        for i in 0..self.records.len() {
            let Some(world) = self.records[i].world.take() else {
                continue;
            };
            let (self_id, victim_id, prev) = (self.records[i].self_id, self.records[i].victim_id, self.records[i].prev);
            let tick = self.records[i].tick;
            let in_flight = self.records[i].in_flight.clone();
            let hyb_plan = self.records[i].hybrid.chosen_plan.clone();
            let t_plan = self.records[i].teacher_plan.clone();
            let opp: Vec<PlayerInput> = (0..80).map(|k| self.opp_at(tick + k)).collect();
            let oppf = |k: usize| opp[k.min(opp.len() - 1)];
            let ctx = self.records[i].ctx.take();
            let pool_plans: Vec<Vec<PlanStep>> = self.records[i].hybrid.pool.iter().map(|c| c.plan.clone()).collect();
            let mut plans: Vec<&[PlanStep]> = vec![&hyb_plan];
            let t_idx = t_plan.as_ref().map(|p| {
                plans.push(p);
                plans.len() - 1
            });
            let first_pool = plans.len();
            plans.extend(pool_plans.iter().map(Vec::as_slice));
            let out = self
                .hybrid
                .debug_truth(&world, self_id, victim_id, prev, &in_flight, &plans, &oppf);
            // The evaluator told the opponent's inputs: one per plan step, from the plan's first tick.
            let lag = in_flight.len() as i32;
            let step_ticks = self.hybrid.config().planner.plan_step.max(1);
            let predicted: Vec<PlayerInput> = (0..self.hybrid.config().planner.steps.max(1))
                .map(|s| self.opp_at(tick + lag + s * step_ticks))
                .collect();
            let mut oracle_plans: Vec<&[PlanStep]> = pool_plans.iter().map(Vec::as_slice).collect();
            if let Some(p) = &t_plan {
                oracle_plans.push(p);
            }
            let oracle = ctx
                .as_ref()
                .map_or_else(Vec::new, |c| self.hybrid.debug_oracle(c, &oracle_plans, &predicted));
            // The opponent as a planner like us: its plan's open-loop inputs, and the pool scored against them.
            let mirror_plan = self.records[i].mirror_plan.clone();
            let (mirror_inputs, oracle_mirror) = match (&mirror_plan, ctx.as_ref()) {
                (Some(mp), Some(c)) => {
                    let opp_prev = self.opp_at(tick - 1);
                    let inputs = self.hybrid.debug_plan_inputs(&world, victim_id, self_id, opp_prev, mp);
                    let scores = if inputs.is_empty() {
                        Vec::new()
                    } else {
                        self.hybrid.debug_oracle(c, &oracle_plans, &inputs)
                    };
                    (inputs, scores)
                }
                _ => (Vec::new(), Vec::new()),
            };
            let rec = &mut self.records[i];
            rec.mirror_inputs = mirror_inputs;
            rec.oracle_mirror = oracle_mirror;
            rec.truth_hybrid = out.first().copied();
            rec.truth_teacher = t_idx.and_then(|k| out.get(k).copied());
            rec.truth_pool = out[first_pool.min(out.len())..].to_vec();
            rec.oracle = oracle;
            self.records[i].world = Some(world);
        }
    }
}

/// The brain a diagnosis plays with: share the [`DiagInner`] with the driver (`Rc`), hand the brain to the game.
pub struct DiagBrain(pub Rc<RefCell<DiagInner>>);

impl DiagBrain {
    pub fn new(hybrid: HybridBrain, opts: DiagOptions) -> (DiagBrain, Rc<RefCell<DiagInner>>) {
        let inner = Rc::new(RefCell::new(DiagInner::new(
            hybrid,
            TeacherPlanner::new(crate::brains::PlannerPreset::Normal),
            opts,
        )));
        (DiagBrain(Rc::clone(&inner)), inner)
    }
}

impl Brain for DiagBrain {
    fn reset(&mut self, ctx: &ResetContext) {
        let mut s = self.0.borrow_mut();
        s.hybrid.reset(ctx);
        s.teacher.reset(ctx);
        s.records.clear();
        s.opp_applied.clear();
        s.mirror.reset(ctx);
        s.last_decision_tick = None;
        s.searched = 0;
        s.played_prev = crate::types::empty_input();
    }

    fn decide(&mut self, obs: &Observation) -> Action {
        self.0.borrow_mut().hybrid.decide(obs)
    }

    fn decide_in(&mut self, obs: &Observation, view: Option<&WorldView<'_>>) -> Action {
        match view {
            Some(v) => self.0.borrow_mut().decide_in_impl(obs, v),
            None => self.decide(obs),
        }
    }

    fn name(&self) -> &str {
        "hybrid-diag"
    }
}
