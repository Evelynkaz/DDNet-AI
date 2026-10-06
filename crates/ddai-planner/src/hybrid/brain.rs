//! [`HybridBrain`]: the [`Brain`] of D-041 -- the fly proposes, exact search on the real physics
//! decides. It takes the exact world through [`Brain::decide_in`] (the offline arena's, or the live
//! bot's forward-predicted `LiveWorld`), copies it into its own planning world, models the
//! opponents from what their visible state shows, and hands the decision to
//! [`crate::hybrid::search::HybridSearch`].
//!
//! Like the plain planner brain it never touches the caller's world, and without a view (a replay
//! of recorded observations) it rebuilds an approximate planning world from the observation.

use std::collections::BTreeMap;
use std::sync::Arc;

use ddai_brain::{Action, Brain, LiveContext, Observation, ResetContext, WorldView};
use ddai_physics::map::MapData;

use crate::brains::{ClockKind, PlannerStats, action_from_input, enemy_input_from_tee, target_of};
use crate::clock::{Clock, StepClock, WallClock};
use crate::hybrid::config::{HybridConfig, HybridMode};
use crate::hybrid::proposer::{NoProposer, ProposalOutcome, Proposer};
use crate::hybrid::search::{DecisionInput, DecisionTelemetry, HybridSearch, SOURCE_KINDS, Source, WorkCounters};
use crate::physics_adapter::{PhysicsWorld, from_ddnet_input};
use crate::plan_world::PlanWorld;
use crate::types::{PlayerInput, empty_input};

fn dist_f32(a: ddai_physics::vmath::Vec2<f32>, b: ddai_physics::vmath::Vec2<f32>) -> f64 {
    f64::from(a.x - b.x).hypot(f64::from(a.y - b.y))
}

/// Sums over a brain's decisions since its last reset (the arena JSONL carries this).
#[derive(Debug, Clone, Default)]
pub struct Totals {
    pub decisions: u64,
    /// Decisions that used the adaptive extension (D-042).
    pub extended: u64,
    /// Decisions on which some danger flag fired (the extension was *allowed*).
    pub danger_flagged: u64,
    pub shielded: u64,
    pub shield_incomplete: u64,
    /// Decisions the shield checked, and those whose first escape (the plan remainder) held.
    pub shield_ran: u64,
    pub shield_plan_ok: u64,
    pub shield_skipped: u64,
    pub out_of_time: u64,
    /// Decisions whose chosen plan still ended with us out under some modelled response.
    pub unsafe_choices: u64,
    pub work: WorkCounters,
    pub generated: [u64; SOURCE_KINDS],
    pub evaluated: [u64; SOURCE_KINDS],
    /// How often each source supplied the chosen plan, by [`Source::kind`].
    pub chosen: [u64; SOURCE_KINDS],
    /// How often each technique supplied the chosen plan, by name.
    pub techniques: BTreeMap<&'static str, u64>,
    /// Decisions with at least one modelled threat besides the victim.
    pub with_threats: u64,
    /// Task 3.9 fire counters (see `DecisionTelemetry::polished`).
    pub polished: u64,
    pub wall_cands: u64,
}

impl Totals {
    fn add(&mut self, t: &DecisionTelemetry) {
        self.decisions += 1;
        self.extended += u64::from(t.extended);
        self.danger_flagged += u64::from(t.danger.flagged());
        self.shielded += u64::from(t.shielded);
        self.shield_incomplete += u64::from(t.shield_incomplete);
        self.shield_ran += u64::from(t.shield_ran);
        self.shield_plan_ok += u64::from(t.shield_plan_ok);
        self.shield_skipped += u64::from(t.shield_skipped);
        self.out_of_time += u64::from(t.out_of_time);
        self.unsafe_choices += u64::from(t.unsafe_choice);
        self.with_threats += u64::from(!t.threat_ids.is_empty());
        self.work.add(&t.work);
        self.polished += u64::from(t.polished);
        self.wall_cands += u64::from(t.wall_cands);
        for k in 0..SOURCE_KINDS {
            self.generated[k] += u64::from(t.generated[k]);
            self.evaluated[k] += u64::from(t.evaluated[k]);
        }
        if let Some(src) = t.chosen {
            self.chosen[src.kind()] += 1;
            if let Source::Tech(tech) = src {
                *self.techniques.entry(tech.name()).or_insert(0) += 1;
            }
        }
    }

    pub fn to_json(&self) -> String {
        let per = |a: &[u64; SOURCE_KINDS]| {
            (0..SOURCE_KINDS)
                .map(|k| format!("\"{}\":{}", Source::kind_name(k), a[k]))
                .collect::<Vec<_>>()
                .join(",")
        };
        let techs = self
            .techniques
            .iter()
            .map(|(k, v)| format!("\"{k}\":{v}"))
            .collect::<Vec<_>>()
            .join(",");
        let mut fired = String::new();
        if self.polished > 0 {
            fired.push_str(&format!(",\"polish\":{}", self.polished));
        }
        if self.wall_cands > 0 {
            fired.push_str(&format!(",\"wall\":{}", self.wall_cands));
        }
        let w = &self.work;
        format!(
            "{{\"decisions\":{},\"extended\":{},\"danger_flagged\":{},\"shielded\":{},\"shield_incomplete\":{},\"shield_ran\":{},\"shield_plan_ok\":{},\"shield_skipped\":{},\
\"out_of_time\":{},\"unsafe_choices\":{},\"with_threats\":{},\"generated\":{{{}{}}},\"evaluated\":{{{}}},\"chosen\":{{{}}},\
\"techniques\":{{{}}},\"work\":{{\"ticks\":{},\"lag\":{},\"proposal\":{},\"proposal_units\":{},\"stage1\":{},\"stage2\":{},\"extension\":{},\"shield\":{},\"rays\":{},\
\"rollouts_stage1\":{},\"rollouts_stage2\":{},\"rollouts_extension\":{}{}}}}}",
            self.decisions,
            self.extended,
            self.danger_flagged,
            self.shielded,
            self.shield_incomplete,
            self.shield_ran,
            self.shield_plan_ok,
            self.shield_skipped,
            self.out_of_time,
            self.unsafe_choices,
            self.with_threats,
            per(&self.generated),
            fired,
            per(&self.evaluated),
            per(&self.chosen),
            techs,
            w.total_ticks(),
            w.lag,
            w.proposal,
            w.proposal_units,
            w.stage1,
            w.stage2,
            w.extension,
            w.shield,
            w.rays,
            w.rollouts_stage1,
            w.rollouts_stage2,
            w.rollouts_extension,
            // The optional counters, only when non-zero (a hybrid without the v2 switches prints what it always did).
            {
                let mut tail = String::new();
                if w.mirror > 0 {
                    tail.push_str(&format!(",\"mirror\":{}", w.mirror));
                }
                if w.units > 0 {
                    tail.push_str(&format!(",\"units\":{}", w.units));
                }
                tail
            },
        )
    }
}

enum BrainClock {
    Wall(Arc<WallClock>),
    Step(StepClock),
    Work(crate::hybrid::work::WorkClock),
}

impl Clock for BrainClock {
    fn now_ms(&self) -> f64 {
        match self {
            BrainClock::Wall(c) => c.now_ms(),
            BrainClock::Step(c) => c.now_ms(),
            BrainClock::Work(c) => c.now_ms(),
        }
    }
}

/// The hybrid brain (see the module docs). Not `Send`-required by anything: build it inside the
/// thread that plays the game (its worker threads, if any, are owned by it).
pub struct HybridBrain {
    cfg: HybridConfig,
    clock: BrainClock,
    wall: Arc<WallClock>,
    proposer: Option<Box<dyn Proposer>>,
    meter: Option<Arc<crate::hybrid::work::WorkMeter>>,
    search: Option<HybridSearch>,
    map: Option<Arc<MapData>>,
    prev: PlayerInput,
    totals: Totals,
    last: Option<DecisionTelemetry>,
    stats: PlannerStats,
    name: String,
    /// A `reset` that the search (built lazily, once the map is known) has not seen yet.
    pending_reset: Option<ResetContext>,
    last_reset: Option<ResetContext>,
    /// The live bot's spared tees and travel goal (task 3.5b), handed to the search before every
    /// decision; they hold until replaced.
    spares: Vec<crate::vmath::Vec2>,
    spare_vels: Vec<crate::vmath::Vec2>,
    spare_ids: Vec<i32>,
    travel_goal: Option<crate::vmath::Vec2>,
    /// Task 7.4: the decision just made went through the search with a proposer, so the proposer's frame is of *this*
    /// decision. Cleared at the start of every decision (watched or not), set only when the proposer was consulted: a
    /// viewer that subscribes later never gets an older decision's frame labelled with the current verdict.
    viz_fresh: bool,
}

impl HybridBrain {
    /// `clock`: [`ClockKind::Wall`] in play; a [`ClockKind::Step`] clock makes deadline mode
    /// deterministic for tests (it needs `cfg.workers == 1`, since worker threads read real time).
    pub fn new(cfg: HybridConfig, clock: ClockKind, proposer: Box<dyn Proposer>) -> Result<HybridBrain, String> {
        cfg.validate()?;
        if matches!(clock, ClockKind::Step { .. }) && cfg.workers > 1 {
            return Err("hybrid: a step clock cannot drive worker threads (workers must be 1)".into());
        }
        let wall = Arc::new(WallClock::new());
        let meter = cfg
            .work_clock_us_per_tick
            .map(|_| crate::hybrid::work::WorkMeter::new());
        let bc = match (clock, cfg.work_clock_us_per_tick, &meter) {
            (_, Some(us), Some(m)) => BrainClock::Work(crate::hybrid::work::WorkClock::new(Arc::clone(m), us)),
            (ClockKind::Wall, _, _) => BrainClock::Wall(Arc::clone(&wall)),
            (ClockKind::Step { step_ms }, _, _) => BrainClock::Step(StepClock::new(step_ms)),
        };
        let mode = match cfg.mode {
            HybridMode::Fixed => "fixed".to_string(),
            HybridMode::Deadline { budget_ms } if cfg.work_clock_us_per_tick.is_some() => format!("{budget_ms}msw"),
            HybridMode::Deadline { budget_ms } => format!("{budget_ms}ms"),
        };
        let name = format!(
            "hybrid-{}-{}{}{}",
            proposer.name(),
            mode,
            if cfg.workers > 1 {
                format!("-w{}", cfg.workers)
            } else {
                String::new()
            },
            if cfg.threat_model { "" } else { "-1v1model" },
        );
        Ok(HybridBrain {
            cfg,
            clock: bc,
            wall,
            proposer: Some(proposer),
            meter,
            search: None,
            map: None,
            prev: empty_input(),
            totals: Totals::default(),
            last: None,
            stats: PlannerStats::default(),
            name,
            pending_reset: None,
            last_reset: None,
            spares: Vec::new(),
            spare_vels: Vec::new(),
            spare_ids: Vec::new(),
            travel_goal: None,
            viz_fresh: false,
        })
    }

    /// A hybrid without proposals (`NoProposer`), fixed-work mode.
    pub fn fixed() -> HybridBrain {
        let mut cfg = HybridConfig::fixed();
        cfg.proposals = 0;
        HybridBrain::new(cfg, ClockKind::Wall, Box::new(NoProposer)).expect("valid")
    }

    /// Tees the rope and the hammer must spare (the live bot's friends, ignored, out-of-game and AFK
    /// players, `bot.ts:3009-3018`): positions and velocities in pixels (per tick), as the planner's
    /// `set_spare_bystanders` takes them. A spared tee is never a threat or a victim, a candidate
    /// whose rope would catch one is refused and a hammer swing that would hit one is not fired.
    /// They hold until replaced; the arena never sets them. (`ddai_brain::Brain::set_live_context`
    /// delegates here.)
    pub fn set_spares(&mut self, positions: Vec<crate::vmath::Vec2>, velocities: Vec<crate::vmath::Vec2>) {
        self.spares = positions;
        self.spare_vels = velocities;
    }

    /// Client ids of the spared tees (4.1b: `LiveContext.spare_ids`). They stay in the world as bodies
    /// (tee-tee collision can still deflect us into freeze) but are never a threat, a victim, a target
    /// or a hook target; the swing and hook gates use the positions of [`HybridBrain::set_spares`].
    pub fn set_spare_ids(&mut self, ids: Vec<i32>) {
        self.spare_ids = ids;
    }

    /// The tee to play against: the observation's target unless it is spared, else the nearest tee that
    /// is not (`None` when there is none).
    fn target_id(&self, obs: &Observation) -> Option<i32> {
        let t = target_of(obs)?;
        if !self.spare_ids.contains(&t.id) {
            return Some(t.id);
        }
        let me = obs.self_state.pos;
        obs.others
            .iter()
            .filter(|c| c.id != obs.self_state.id && !self.spare_ids.contains(&c.id))
            .min_by(|a, b| dist_f32(a.pos, me).total_cmp(&dist_f32(b.pos, me)))
            .map(|c| c.id)
    }

    /// An intermediate point to head for when the target is far or behind a wall (`None` = the
    /// target itself); the planner's `set_travel_goal`.
    pub fn set_travel_goal(&mut self, goal: Option<crate::vmath::Vec2>) {
        self.travel_goal = goal;
    }

    pub fn totals(&self) -> &Totals {
        &self.totals
    }

    pub fn last_decision(&self) -> Option<&DecisionTelemetry> {
        self.last.as_ref()
    }

    /// Task 3.7b: scores `plans` with the evaluator of the decision just made (`HybridSearch::debug_score`).
    pub fn debug_score(
        &mut self,
        plans: &[Vec<crate::planner::PlanStep>],
    ) -> Option<Vec<crate::hybrid::search::DebugScore>> {
        self.search.as_mut()?.debug_score(&self.clock, plans)
    }

    /// Task 3.7b: what `plans` do in the true world `world` (the decision's `WorldView`) against the opponent's
    /// recorded inputs: `opp(k)` is its input at world tick `world.tick + k`. Our own inputs through the lag are
    /// `in_flight`. One outcome per plan. Diagnostics only.
    #[allow(clippy::too_many_arguments)]
    pub fn debug_truth(
        &mut self,
        world: &ddai_physics::world::World<f32>,
        self_id: i32,
        victim_id: i32,
        prev: PlayerInput,
        in_flight: &[ddai_physics::core::PlayerInput],
        plans: &[&[crate::planner::PlanStep]],
        opp: &dyn Fn(usize) -> PlayerInput,
    ) -> Vec<crate::hybrid::search::TruthOutcome> {
        let Some(search) = self.search.as_mut() else {
            return Vec::new();
        };
        let pw = search.world_mut();
        pw.sync_from(world);
        pw.set_held_input(self_id, in_flight.first().map_or(prev, from_ddnet_input));
        for other in pw.all_tees() {
            if other.id != self_id {
                pw.set_held_input(other.id, opp(0));
            }
        }
        for (k, wire) in in_flight.iter().enumerate() {
            pw.set_input(self_id, from_ddnet_input(wire));
            pw.set_input(victim_id, opp(k));
            pw.step();
        }
        let lag = in_flight.len();
        let after = in_flight.last().map_or(prev, from_ddnet_input);
        plans
            .iter()
            .map(|p| search.debug_truth(self_id, victim_id, after, p, &|t| opp(lag + t), None))
            .collect()
    }

    /// Task 3.7b: the inputs, one per plan step, that `plan` makes tee `owner` send against `other` holding its input (the
    /// opponent's predicted plan, turned into the open-loop inputs `debug_oracle` takes). The world is synced from `world`;
    /// no input lag is rolled in.
    pub fn debug_plan_inputs(
        &mut self,
        world: &ddai_physics::world::World<f32>,
        owner: i32,
        other: i32,
        prev: PlayerInput,
        plan: &[crate::planner::PlanStep],
    ) -> Vec<PlayerInput> {
        let Some(search) = self.search.as_mut() else {
            return Vec::new();
        };
        let pw = search.world_mut();
        pw.sync_from(world);
        // What a planner in the owner's seat believes of the other tee: it keeps doing what it does now.
        let Some(other_tee) = pw.get_tee(other) else {
            return Vec::new();
        };
        let other_input = enemy_input_from_tee(&other_tee);
        pw.set_held_input(owner, prev);
        pw.set_held_input(other, other_input);
        let mut inputs = Vec::new();
        let _ = search.debug_truth(owner, other, prev, plan, &|_| other_input, Some(&mut inputs));
        inputs
    }

    /// Task 3.7b: the engine's context of the decision just made, for `debug_oracle`.
    pub fn debug_ctx(&self) -> Option<std::sync::Arc<crate::hybrid::engine::Ctx>> {
        self.search.as_ref()?.debug_ctx()
    }

    /// Task 3.7b: scores `plans` in a kept context with the opponent's actual inputs per plan step (`predicted`).
    pub fn debug_oracle(
        &mut self,
        ctx: &crate::hybrid::engine::Ctx,
        plans: &[&[crate::planner::PlanStep]],
        predicted: &[PlayerInput],
    ) -> Vec<Option<f64>> {
        self.search
            .as_mut()
            .map_or_else(Vec::new, |s| s.debug_oracle(ctx, plans, predicted))
    }

    /// What the work clock charges for one proposal of this brain's proposer, in tee-tick equivalents (0 without a
    /// proposer): the price the arena takes off the search budget under `proposal_in_cap` (task 3.7a, D-080).
    pub fn proposer_work_units(&self) -> u64 {
        match (self.search.as_ref(), self.proposer.as_ref()) {
            (Some(search), _) => search.proposer_work_units(),
            (None, Some(proposer)) => proposer.work_units(),
            (None, None) => 0,
        }
    }

    pub fn config(&self) -> &HybridConfig {
        &self.cfg
    }

    /// Planner-shaped counters, so arena tooling written for `PlannerBrain` reads this brain too.
    pub fn stats(&self) -> PlannerStats {
        self.stats
    }

    fn ensure_search(&mut self, map: &Arc<MapData>, template: impl FnOnce() -> ddai_physics::world::World<f32>) {
        let same_map = self.map.as_ref().is_some_and(|m| Arc::ptr_eq(m, map));
        if self.search.is_some() && same_map {
            return;
        }
        let proposer = match self.search.take() {
            Some(old) => old.into_proposer(),
            None => self.proposer.take().unwrap_or_else(|| Box::new(NoProposer)),
        };
        let world = PhysicsWorld::from_world(template(), map.clone());
        let mut search = HybridSearch::new(self.cfg.clone(), proposer, world, Arc::clone(&self.wall));
        search.set_meter(self.meter.clone());
        self.search = Some(search);
        self.map = Some(map.clone());
        // A rebuilt search starts from the last reset too.
        if self.pending_reset.is_none() {
            self.pending_reset = self.last_reset.clone();
        }
    }

    fn apply_pending_reset(&mut self) {
        if let (Some(ctx), Some(search)) = (self.pending_reset.take(), self.search.as_mut()) {
            search.reset(&ctx);
        }
    }

    /// The decision, given the planning world already synced into the search.
    fn plan(
        &mut self,
        self_id: i32,
        target_id: i32,
        in_flight: &[ddai_physics::core::PlayerInput],
        obs: &Observation,
    ) -> Action {
        let search = self.search.as_mut().expect("search built");
        search.set_live(&self.spares, &self.spare_vels, &self.spare_ids, self.travel_goal);
        let world = search.world_mut();
        let (Some(me), Some(target)) = (world.get_tee(self_id), world.get_tee(target_id)) else {
            return action_from_input(&self.prev);
        };
        if !me.alive || !target.alive {
            return action_from_input(&self.prev);
        }
        // Everybody keeps doing what a snapshot shows them doing (`syncPlanningWorld`).
        for other in world.all_tees() {
            if other.id != self_id {
                world.set_held_input(other.id, enemy_input_from_tee(&other));
            }
        }
        let target_input = enemy_input_from_tee(&target);
        world.set_held_input(self_id, in_flight.first().map_or(self.prev, from_ddnet_input));
        let mut roll_ticks = 0u64;
        for wire in in_flight {
            world.set_input(self_id, from_ddnet_input(wire));
            world.set_input(target_id, target_input);
            world.step();
            roll_ticks += 1;
        }
        let (out, tel) = search.decide(
            &self.clock,
            &DecisionInput {
                obs,
                self_id,
                victim_id: target_id,
                prev: self.prev,
                lag_ticks: in_flight.len() as u32,
                roll_ticks,
            },
        );
        self.totals.add(&tel);
        self.stats.decisions += 1;
        self.stats.searched += 1;
        self.stats.out_of_time += u64::from(tel.out_of_time);
        self.stats.shielded += u64::from(tel.shielded);
        self.stats.shield_incomplete += u64::from(tel.shield_incomplete);
        self.stats.candidates += u64::from(tel.evaluated.iter().sum::<u32>());
        self.viz_fresh = self.cfg.proposals > 0;
        self.last = Some(tel);
        self.prev = out;
        action_from_input(&out)
    }
}

impl Brain for HybridBrain {
    fn reset(&mut self, ctx: &ResetContext) {
        self.prev = empty_input();
        self.totals = Totals::default();
        self.stats = PlannerStats::default();
        self.last = None;
        if let Some(p) = self.proposer.as_mut() {
            p.reset(ctx);
        }
        self.pending_reset = Some(ctx.clone());
        self.last_reset = Some(ctx.clone());
        self.spares.clear();
        self.spare_vels.clear();
        self.spare_ids.clear();
        self.travel_goal = None;
    }

    fn decide(&mut self, obs: &Observation) -> Action {
        self.viz_fresh = false;
        // A decision that returns early (no target, dead target) leaves no verdict behind: `last_plan` must not
        // report the previous decision's search as this one's.
        self.last = None;
        let Some(target) = self.target_id(obs) else {
            self.prev = empty_input();
            return Action::neutral();
        };
        // No exact world: rebuild an approximate one from the observation.
        let map = obs.map.clone();
        self.ensure_search(&map, || {
            let mut w = ddai_physics::world::World::<f32>::from_map(&map, 1);
            let _ = w.init(std::iter::empty::<&str>());
            w
        });
        self.apply_pending_reset();
        let rebuilt = crate::brains::planning_world_from_observation(&map, obs);
        *self.search.as_mut().expect("built").world_mut() = rebuilt;
        self.plan(obs.self_state.id, target, &[], obs)
    }

    fn decide_in(&mut self, obs: &Observation, view: Option<&WorldView<'_>>) -> Action {
        self.viz_fresh = false;
        self.last = None;
        let Some(view) = view else {
            return self.decide(obs);
        };
        let Some(target) = self.target_id(obs) else {
            self.prev = empty_input();
            return Action::neutral();
        };
        self.ensure_search(&obs.map, || view.world.clone());
        self.apply_pending_reset();
        self.search.as_mut().expect("built").world_mut().sync_from(view.world);
        self.plan(view.self_id, target, view.in_flight, obs)
    }

    fn set_live_context(&mut self, ctx: &LiveContext<'_>) {
        let to64 = |v: &ddai_physics::vmath::Vec2<f32>| crate::vmath::Vec2 {
            x: f64::from(v.x),
            y: f64::from(v.y),
        };
        // In place: no allocation per decision once the buffers have grown.
        self.spares.clear();
        self.spares.extend(ctx.spares.iter().map(|(p, _)| to64(p)));
        self.spare_vels.clear();
        self.spare_vels.extend(ctx.spares.iter().map(|(_, v)| to64(v)));
        // 4.1b (F8): the spared tees stay in the world as bodies; their ids leave the threat, victim,
        // target and hook-target sets (no allocation once the buffer has grown).
        self.spare_ids.clear();
        self.spare_ids.extend_from_slice(ctx.spare_ids);
        self.set_travel_goal(ctx.travel_goal.as_ref().map(to64));
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn last_plan(&self) -> Option<ddai_brain::PlanTelemetry> {
        let t = self.last.as_ref()?;
        Some(ddai_brain::PlanTelemetry {
            searched: t.evaluated.iter().any(|&n| n > 0),
            out_of_time: t.out_of_time,
            shielded: t.shielded,
            shield_incomplete: t.shield_incomplete,
            candidates: t.evaluated.iter().sum(),
            proposal_us: (t.proposal_ms * 1000.0).clamp(0.0, f64::from(u32::MAX)) as u32,
            search_us: (t.search_ms * 1000.0).clamp(0.0, f64::from(u32::MAX)) as u32,
        })
    }

    fn viz_meta(&self) -> Option<String> {
        match (self.search.as_ref(), self.proposer.as_ref()) {
            (Some(search), _) => search.proposer().viz_meta(),
            (None, Some(proposer)) => proposer.viz_meta(),
            (None, None) => None,
        }
    }

    fn viz_frame(&mut self, tick: u32) -> Option<&[u8]> {
        if !std::mem::take(&mut self.viz_fresh) {
            return None;
        }
        // The search's verdict on the decision just made; the proposer decides whether it has a fresh frame.
        let outcome = self.last.as_ref().map(|t| ProposalOutcome {
            chosen: matches!(t.chosen, Some(Source::Proposal)),
            chosen_total: u32::try_from(self.totals.chosen[Source::Proposal.kind()]).unwrap_or(u32::MAX),
            decisions_total: u32::try_from(self.totals.decisions).unwrap_or(u32::MAX),
        });
        match (self.search.as_mut(), self.proposer.as_mut()) {
            (Some(search), _) => search.proposer_mut().viz_frame(tick, outcome),
            (None, Some(proposer)) => proposer.viz_frame(tick, outcome),
            (None, None) => None,
        }
    }

    fn telemetry(&self) -> Option<String> {
        let last = self
            .last
            .as_ref()
            .map_or("null".to_string(), DecisionTelemetry::to_json);
        Some(format!(
            "{{\"brain\":\"{}\",\"proposer\":\"{}\",\"workers\":{},\"totals\":{},\"last\":{}}}",
            self.name,
            self.search.as_ref().map_or("none", HybridSearch::proposer_name),
            self.cfg.workers,
            self.totals.to_json(),
            last
        ))
    }
}
