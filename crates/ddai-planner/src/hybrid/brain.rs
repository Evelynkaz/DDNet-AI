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

use ddai_brain::{Action, Brain, Observation, ResetContext, WorldView};
use ddai_physics::map::MapData;

use crate::brains::{ClockKind, PlannerStats, action_from_input, enemy_input_from_tee, target_of};
use crate::clock::{Clock, StepClock, WallClock};
use crate::hybrid::config::{HybridConfig, HybridMode};
use crate::hybrid::proposer::{NoProposer, Proposer};
use crate::hybrid::search::{DecisionInput, DecisionTelemetry, HybridSearch, SOURCE_KINDS, Source, WorkCounters};
use crate::physics_adapter::{PhysicsWorld, from_ddnet_input};
use crate::plan_world::PlanWorld;
use crate::types::{PlayerInput, empty_input};

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
}

impl Totals {
    fn add(&mut self, t: &DecisionTelemetry) {
        self.decisions += 1;
        self.extended += u64::from(t.extended);
        self.danger_flagged += u64::from(t.danger.flagged());
        self.shielded += u64::from(t.shielded);
        self.shield_incomplete += u64::from(t.shield_incomplete);
        self.out_of_time += u64::from(t.out_of_time);
        self.unsafe_choices += u64::from(t.unsafe_choice);
        self.with_threats += u64::from(!t.threat_ids.is_empty());
        self.work.add(&t.work);
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
        let w = &self.work;
        format!(
            "{{\"decisions\":{},\"extended\":{},\"danger_flagged\":{},\"shielded\":{},\"shield_incomplete\":{},\
\"out_of_time\":{},\"unsafe_choices\":{},\"with_threats\":{},\"generated\":{{{}}},\"evaluated\":{{{}}},\"chosen\":{{{}}},\
\"techniques\":{{{}}},\"work\":{{\"ticks\":{},\"lag\":{},\"proposal\":{},\"stage1\":{},\"stage2\":{},\"extension\":{},\"shield\":{},\"rays\":{},\
\"rollouts_stage1\":{},\"rollouts_stage2\":{},\"rollouts_extension\":{}}}}}",
            self.decisions,
            self.extended,
            self.danger_flagged,
            self.shielded,
            self.shield_incomplete,
            self.out_of_time,
            self.unsafe_choices,
            self.with_threats,
            per(&self.generated),
            per(&self.evaluated),
            per(&self.chosen),
            techs,
            w.total_ticks(),
            w.lag,
            w.proposal,
            w.stage1,
            w.stage2,
            w.extension,
            w.shield,
            w.rays,
            w.rollouts_stage1,
            w.rollouts_stage2,
            w.rollouts_extension,
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
        })
    }

    /// A hybrid without proposals (`NoProposer`), fixed-work mode.
    pub fn fixed() -> HybridBrain {
        let mut cfg = HybridConfig::fixed();
        cfg.proposals = 0;
        HybridBrain::new(cfg, ClockKind::Wall, Box::new(NoProposer)).expect("valid")
    }

    pub fn totals(&self) -> &Totals {
        &self.totals
    }

    pub fn last_decision(&self) -> Option<&DecisionTelemetry> {
        self.last.as_ref()
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
    }

    fn decide(&mut self, obs: &Observation) -> Action {
        let Some(target) = target_of(obs).map(|c| c.id) else {
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
        let Some(view) = view else {
            return self.decide(obs);
        };
        let Some(target) = target_of(obs).map(|c| c.id) else {
            self.prev = empty_input();
            return Action::neutral();
        };
        self.ensure_search(&obs.map, || view.world.clone());
        self.apply_pending_reset();
        self.search.as_mut().expect("built").world_mut().sync_from(view.world);
        self.plan(view.self_id, target, view.in_flight, obs)
    }

    fn name(&self) -> &str {
        &self.name
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
