//! [`Brain`] adapters over the ported planner and scripted bot (task 8.1): [`ScriptedBrain`]
//! (`scriptedAction`, RNG seeded from `ResetContext.seed`) and [`PlannerBrain`] (the CEM planner
//! over [`crate::physics_adapter::PhysicsWorld`], fixed-iteration or deadline mode, presets
//! normal/low/strong). `IdleBrain` lives in `ddai-brain` and is re-exported here.
//!
//! Both brains prefer the exact world of [`Brain::decide_in`]: they copy it into a private
//! planning world (`PhysicsWorld::sync_from`) and never touch the caller's. Called through plain
//! [`Brain::decide`] (no world, e.g. a replayed observation) they rebuild an approximate planning
//! world from the [`Observation`] the way the TS bot's `syncPlanningWorld` did from a snapshot.
//!
//! Opponent model: like the phase-0 harness (`orig-run.md` §2.1) the target's input is
//! reconstructed from its *observable* state ([`enemy_input_from_tee`]: direction, hook flag, aim
//! angle), not read from the arena, so an arena result reflects what a live bot could know.

pub use ddai_brain::IdleBrain;

use std::sync::Arc;

use ddai_brain::{
    Action, Brain, CharacterObservation, IVec2, LiveContext, MapKnowledge, Observation, ResetContext, WorldView,
};
use ddai_jsmath as js;
use ddai_jsmath::Rng;
use ddai_physics::map::MapData;

use crate::clock::{Clock, StepClock, WallClock};
use crate::config::{
    PlannerConfig, PlannerVersion, preset_live_v2, preset_low_cpu, preset_normal, preset_normal_v2, preset_strong_wb,
    preset_strong_wb_v2, wb_overrides, wb_overrides_v2,
};
use crate::memory::FreezeMemory;
use crate::physics_adapter::{PhysicsWorld, from_ddnet_input};
use crate::plan_world::PlanWorld;
use crate::planner::{DeadZoneGrid, DecisionInfo, Planner};
use crate::scripted::scripted_action;
use crate::types::{PlayerInput, TeeState, blank_tee_state, empty_input};
use crate::vmath::Vec2;

/// `BYSTANDER_PX` (`bot.ts:339`): frozen non-target tees closer than this are passed to the
/// planner as frozen bystanders (`bot.ts:4767-4775`).
pub(crate) const BYSTANDER_PX: f64 = 160.0;

/// `wireAngleRad` (`core/types.ts:83`): the wire aim angle (1/256 rad, `0..2*PI*256`) as radians
/// in `(-PI, PI]`.
fn wire_angle_rad(angle: f64) -> f64 {
    let a = angle / 256.0;
    if a >= std::f64::consts::PI {
        a - 2.0 * std::f64::consts::PI
    } else {
        a
    }
}

/// `enemyInputFromSnapshot` (`livePlan.ts:5`): the input the planner assumes an opponent keeps
/// holding, reconstructed from what a snapshot shows -- direction, "hook is out", aim.
pub fn enemy_input_from_tee(target: &TeeState) -> PlayerInput {
    let mut input = empty_input();
    input.direction = target.direction;
    input.hook = i32::from(target.hook_state > 0);
    let angle = wire_angle_rad(target.angle);
    input.target_x = js::round(js::cos(angle) * 300.0);
    input.target_y = js::round(js::sin(angle) * 300.0);
    input
}

/// The planner's input as the brain-level [`Action`] (levels; `fire` = low bit of the counter).
pub fn action_from_input(i: &PlayerInput) -> Action {
    Action {
        direction: i.direction,
        jump: i.jump != 0,
        hook: i.hook != 0,
        fire: (i.fire & 1) != 0,
        target: IVec2::new(i.target_x.round() as i32, i.target_y.round() as i32),
        wanted_weapon: (i.wanted_weapon > 0).then_some(i.wanted_weapon - 1),
    }
}

/// An [`Action`] as a planner input, continuing `prev`'s fire press-counter (a `true` level is
/// always a fresh press, a `false` level releases -- the same convention as
/// `decodeAction`/`scriptedAction`, and as the arena's).
pub fn input_from_action(a: &Action, prev: &PlayerInput) -> PlayerInput {
    let held = (prev.fire & 1) != 0;
    let fire = match (a.fire, held) {
        (true, true) => prev.fire + 2,
        (true, false) => prev.fire + 1,
        (false, true) => prev.fire + 1,
        (false, false) => prev.fire,
    };
    let (tx, ty) = if a.target == IVec2::new(0, 0) {
        (0, -1)
    } else {
        (a.target.x, a.target.y)
    };
    PlayerInput {
        direction: a.direction,
        target_x: f64::from(tx),
        target_y: f64::from(ty),
        jump: i32::from(a.jump),
        fire,
        hook: i32::from(a.hook),
        player_flags: 0,
        wanted_weapon: a.wanted_weapon.map_or(0, |w| w + 1),
        next_weapon: 0,
        prev_weapon: 0,
    }
}

/// An approximate planner tee from an [`Observation`] entry (used only when no exact world is
/// given). Fields the observation does not carry (aim angle, reload, attack tick) get neutral
/// values.
fn tee_from_observation(c: &CharacterObservation) -> TeeState {
    let mut t = blank_tee_state();
    t.id = c.id;
    t.alive = true;
    t.pos = Vec2 {
        x: f64::from(c.pos.x),
        y: f64::from(c.pos.y),
    };
    t.vel = Vec2 {
        x: f64::from(c.vel.x),
        y: f64::from(c.vel.y),
    };
    t.hook_state = c.hook_state;
    t.hook_pos = Vec2 {
        x: f64::from(c.hook_pos.x),
        y: f64::from(c.hook_pos.y),
    };
    let (dx, dy) = (t.hook_pos.x - t.pos.x, t.hook_pos.y - t.pos.y);
    let len = js::hypot2(dx, dy);
    if len > 0.0 {
        t.hook_dir = Vec2 {
            x: dx / len,
            y: dy / len,
        };
    }
    t.hooked_player = c.hooked_player;
    t.jumped = i32::from(c.jumps_used > 0);
    t.jumps_left = c.jumps_left;
    t.jumped_total = Some(c.jumps_used);
    t.direction = c.direction;
    t.active_weapon = c.weapon;
    t.frozen = c.is_frozen;
    t.freeze_ticks_left = i64::from(c.freeze_ticks_remaining);
    t.deep_frozen = Some(c.is_deep_frozen);
    t
}

/// A planner-config transformation (`wb_overrides`, `preset_strong_wb`, ...).
type OverrideFn = fn(PlannerConfig) -> PlannerConfig;

/// A private planning world, rebuilt lazily for the map of the current episode.
pub(crate) struct Scratch {
    pub(crate) map: Arc<MapData>,
    world: Option<PhysicsWorld>,
}

impl Scratch {
    pub(crate) fn new(map: Arc<MapData>) -> Self {
        Scratch { map, world: None }
    }

    /// Copies the exact world in. The first call clones it (sharing the `Arc`'d collision); later
    /// ones `restore_from` into the same allocation.
    pub(crate) fn sync_exact(&mut self, src: &ddai_physics::world::World<f32>) -> &mut PhysicsWorld {
        match &mut self.world {
            Some(w) => w.sync_from(src),
            None => {
                let mut w = PhysicsWorld::from_world(src.clone(), self.map.clone());
                w.sync_from(src);
                self.world = Some(w);
            }
        }
        self.world.as_mut().expect("just set")
    }

    /// Rebuilds a planning world holding exactly the tees of `obs` (no exact world available).
    pub(crate) fn rebuild_from_observation(&mut self, obs: &Observation) -> &mut PhysicsWorld {
        self.world = Some(planning_world_from_observation(&self.map, obs));
        self.world.as_mut().expect("just set")
    }
}

/// A planning world holding exactly the tees of `obs`, built the way `syncPlanningWorld` builds
/// one from a snapshot (no exact world available).
pub(crate) fn planning_world_from_observation(map: &Arc<MapData>, obs: &Observation) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(map.clone(), 1);
    for c in std::iter::once(&obs.self_state).chain(obs.others.iter()) {
        let st = tee_from_observation(c);
        w.add_tee(c.id, st.pos);
        w.apply_tee_state(c.id, &st);
    }
    w
}

/// Which of the target-finding fields the brain resolves the opponent from.
pub(crate) fn target_of(obs: &Observation) -> Option<&CharacterObservation> {
    obs.target_or_nearest()
}

// ---------------------------------------------------------------------------------------------
// ScriptedBrain

/// The built-in scripted bot (`scriptedAction`, `src/env/scripted.ts`) as a [`Brain`]. Draws its
/// aim-noise RNG from `ResetContext.seed` exactly like the phase-0 harness's
/// `ScriptedController` (`Rng((seed * 7919 + 17) >>> 0)`), so a game is reproducible from its
/// seed alone.
pub struct ScriptedBrain {
    rng: Rng,
    prev: PlayerInput,
    scratch: Option<Scratch>,
    name: String,
}

impl ScriptedBrain {
    pub fn new() -> Self {
        ScriptedBrain {
            rng: Rng::new(17),
            prev: empty_input(),
            scratch: None,
            name: "scripted".to_string(),
        }
    }

    fn act(&mut self, world: &PhysicsWorld, self_id: i32, target: Option<i32>) -> Action {
        // No target: `scriptedAction` with an absent enemy id releases fire and does nothing else.
        let out = scripted_action(world, self_id, target.unwrap_or(-1), &self.prev, &mut self.rng);
        self.prev = out;
        action_from_input(&out)
    }
}

impl Default for ScriptedBrain {
    fn default() -> Self {
        Self::new()
    }
}

impl Brain for ScriptedBrain {
    fn reset(&mut self, ctx: &ResetContext) {
        self.rng = Rng::new(ctx.seed.wrapping_mul(7919).wrapping_add(17) as u32);
        self.prev = empty_input();
        if self.scratch.as_ref().is_none_or(|s| !Arc::ptr_eq(&s.map, &ctx.map)) {
            self.scratch = Some(Scratch::new(ctx.map.clone()));
        }
    }

    fn decide(&mut self, obs: &Observation) -> Action {
        let target = target_of(obs).map(|c| c.id);
        let mut scratch = self.scratch.take().unwrap_or_else(|| Scratch::new(obs.map.clone()));
        let action = self.act(scratch.rebuild_from_observation(obs), obs.self_state.id, target);
        self.scratch = Some(scratch);
        action
    }

    fn decide_in(&mut self, obs: &Observation, view: Option<&WorldView<'_>>) -> Action {
        let Some(view) = view else {
            return self.decide(obs);
        };
        let target = target_of(obs).map(|c| c.id);
        let mut scratch = self.scratch.take().unwrap_or_else(|| Scratch::new(obs.map.clone()));
        let action = self.act(scratch.sync_exact(view.world), view.self_id, target);
        self.scratch = Some(scratch);
        action
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ---------------------------------------------------------------------------------------------
// PlannerBrain

/// Planner configuration presets (`docs/research/orig-plan.md` §1.2): `Normal` is the live
/// default (E-000's baseline), `Low` is `--low-cpu`, `Strong` is `STRONG_WB` over `Normal`.
/// `NormalV2` and `LiveV2` are the competitor's current planner (upstream af49dfb, task 3.8, D-095): `NormalV2` is
/// `Normal` on the v2 defaults (what the phase-0 harness would run), `LiveV2` is its `LIVE_PLANNER_CFG`
/// (`launchExposure 1.5`, `jumplessHazardCost 0.4`: how its live bot plays).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlannerPreset {
    Normal,
    Low,
    Strong,
    NormalV2,
    LiveV2,
    /// Task 3.9: the competitor's strong mode on its current planner, applied everywhere (the competitor applies `!strong on` on the
    /// wayblock only): `NormalV2` with `STRONG_WB`'s search of `population 40`, `iterations 3` (three times the default 20 x 2).
    V2Strong,
    /// Task 3.9: `V2Strong` as the competitor really plays it in a wayblock hall -- on top of `WB_PLAN_OVERRIDES` of af49dfb
    /// ([`preset_strong_wb_v2`]).
    V2StrongWb,
    /// Task 3.9 (review F3, F6): a **hypothetical upper bound**, not a version the competitor plays: its `LIVE_PLANNER_CFG`
    /// (`launchExposure 1.5`, `jumplessHazardCost 0.4`, [`PlannerPreset::LiveV2`]) with `STRONG_WB`'s 40 x 3 search everywhere and no
    /// wall-clock budget. In af49dfb `STRONG_WB` applies only inside a held wayblock hall, on top of `WB_PLAN_OVERRIDES` (which resets both
    /// weights to 1.0 / 0.15); outside it the competitor plays `LiveV2` (20 x 2).
    LiveV2Strong,
    /// Task 3.9 (review F5): [`PlannerPreset::Strong`] (`STRONG_WB` over `WB_PLAN_OVERRIDES` of c3c619d) without its 30 / 36 ms wall-clock
    /// budget, for the arena: fixed iterations, reproducible. (`Strong` itself depends on the machine's load there.)
    StrongFixed,
}

impl PlannerPreset {
    pub fn config(self) -> PlannerConfig {
        match self {
            PlannerPreset::Normal => preset_normal(),
            PlannerPreset::Low => preset_low_cpu(),
            PlannerPreset::Strong => preset_strong_wb(preset_normal()),
            PlannerPreset::NormalV2 => preset_normal_v2(),
            PlannerPreset::LiveV2 => preset_live_v2(),
            // `STRONG_WB`'s search size without its wall-clock budget (30 / 36 ms): the arena plays fixed iterations, a pure function of
            // the state (a deadline would make the opponent weaker on a loaded machine and the games irreproducible, task 3.9).
            PlannerPreset::V2Strong => PlannerConfig {
                population: 40,
                iterations: 3,
                budget_ms: 0.0,
                hard_ms: 0.0,
                ..preset_normal_v2()
            },
            PlannerPreset::LiveV2Strong => PlannerConfig {
                population: 40,
                iterations: 3,
                budget_ms: 0.0,
                hard_ms: 0.0,
                ..preset_live_v2()
            },
            PlannerPreset::StrongFixed => PlannerConfig {
                budget_ms: 0.0,
                hard_ms: 0.0,
                ..preset_strong_wb(preset_normal())
            },
            PlannerPreset::V2StrongWb => PlannerConfig {
                budget_ms: 0.0,
                hard_ms: 0.0,
                ..preset_strong_wb_v2(preset_normal_v2())
            },
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            PlannerPreset::Normal => "normal",
            PlannerPreset::Low => "low",
            PlannerPreset::Strong => "strong",
            PlannerPreset::NormalV2 => "normal-v2",
            PlannerPreset::LiveV2 => "live-v2",
            PlannerPreset::V2Strong => "v2-strong",
            PlannerPreset::V2StrongWb => "v2-strong-wb",
            PlannerPreset::LiveV2Strong => "live-v2-strong",
            PlannerPreset::StrongFixed => "strong-fixed",
        }
    }

    /// The planner version this preset reproduces.
    pub fn version(self) -> PlannerVersion {
        match self {
            PlannerPreset::NormalV2
            | PlannerPreset::LiveV2
            | PlannerPreset::V2Strong
            | PlannerPreset::V2StrongWb
            | PlannerPreset::LiveV2Strong => PlannerVersion::Upstream20261002,
            _ => PlannerVersion::Classic,
        }
    }

    /// Parses a preset name (`normal`, `low`, `strong`, `normal-v2`, `live-v2`, `v2-strong`, `v2-strong-wb`, `live-v2-strong`, `strong-fixed`).
    pub fn parse(name: &str) -> Option<PlannerPreset> {
        Some(match name {
            "normal" => PlannerPreset::Normal,
            "low" => PlannerPreset::Low,
            "strong" => PlannerPreset::Strong,
            "normal-v2" => PlannerPreset::NormalV2,
            "live-v2" => PlannerPreset::LiveV2,
            "v2-strong" => PlannerPreset::V2Strong,
            "v2-strong-wb" => PlannerPreset::V2StrongWb,
            "live-v2-strong" => PlannerPreset::LiveV2Strong,
            "strong-fixed" => PlannerPreset::StrongFixed,
            _ => return None,
        })
    }
}

/// How the planner spends effort per decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlannerMode {
    /// `Planner::decide`: fixed iterations, no clock, deterministic (the TS-parity path).
    Fixed,
    /// `Planner::decide_production`: iterative deepening against a wall-clock deadline (D-041).
    Deadline { budget_ms: f64 },
}

/// The time source of [`PlannerMode::Deadline`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClockKind {
    /// Real time: what a live bot uses. Not reproducible.
    Wall,
    /// A fake clock advancing `step_ms` per read: deterministic, for tests.
    Step { step_ms: f64 },
}

enum BrainClock {
    Wall(WallClock),
    Step(StepClock),
}

impl Clock for BrainClock {
    fn now_ms(&self) -> f64 {
        match self {
            BrainClock::Wall(c) => c.now_ms(),
            BrainClock::Step(c) => c.now_ms(),
        }
    }
}

/// [`PlannerBrain`] settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlannerBrainConfig {
    pub preset: PlannerPreset,
    pub mode: PlannerMode,
    pub clock: ClockKind,
}

impl Default for PlannerBrainConfig {
    fn default() -> Self {
        PlannerBrainConfig {
            preset: PlannerPreset::Normal,
            mode: PlannerMode::Fixed,
            clock: ClockKind::Wall,
        }
    }
}

/// Cumulative planner counters over the brain's lifetime (since the last `reset`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PlannerStats {
    pub decisions: u64,
    pub searched: u64,
    /// The search ran out of its deadline before finishing (deadline mode / `budgetMs`).
    pub out_of_time: u64,
    /// The shield replaced the searched input by a safer one.
    pub shielded: u64,
    /// The shield's own time reserve ran out before it could verify the answer.
    pub shield_incomplete: u64,
    pub candidates: u64,
}

impl PlannerStats {
    fn add(&mut self, info: &DecisionInfo) {
        self.decisions += 1;
        self.searched += u64::from(info.searched);
        self.out_of_time += u64::from(info.out_of_time);
        self.shielded += u64::from(info.shielded);
        self.shield_incomplete += u64::from(info.shield_incomplete);
        self.candidates += info.candidates.max(0) as u64;
    }
}

/// The CEM planner as a [`Brain`]: over `PhysicsWorld` (real DDNet physics). See the module docs
/// for the world/opponent handling. `Send` (the planner shares its hazard fields through `Arc`; a
/// compile-time check in `tests/teacher.rs`), but a game still builds its own brain in the thread that
/// plays it.
pub struct PlannerBrain {
    cfg: PlannerBrainConfig,
    // Boxed: a planner is hundreds of kB, and test threads have 2 MB stacks.
    planner: Box<Planner<PhysicsWorld>>,
    clock: BrainClock,
    prev: PlayerInput,
    scratch: Option<Scratch>,
    stats: PlannerStats,
    name: String,
    /// Task 4.1b: the live bot's spared tees (`LiveContext::spare_ids`). They are in the world as
    /// bodies but are never the target nor a frozen bystander (their own geometric gate is `spares`).
    spare_ids: Vec<i32>,
    /// Task 4.2: which `WB_PLAN_OVERRIDES` variant the planner currently runs with, `(in_hall, strong)`;
    /// `set_overrides` recomputes the config, so it is only called when this changes.
    wb_applied: (bool, bool),
}

impl PlannerBrain {
    pub fn new(cfg: PlannerBrainConfig) -> Self {
        let name = match cfg.mode {
            PlannerMode::Fixed => format!("planner-{}-fixed", cfg.preset.label()),
            PlannerMode::Deadline { budget_ms } => format!("planner-{}-{budget_ms}ms", cfg.preset.label()),
        };
        let clock = match cfg.clock {
            ClockKind::Wall => BrainClock::Wall(WallClock::new()),
            ClockKind::Step { step_ms } => BrainClock::Step(StepClock::new(step_ms)),
        };
        PlannerBrain {
            planner: Box::new(Planner::new(cfg.preset.config())),
            cfg,
            clock,
            prev: empty_input(),
            scratch: None,
            stats: PlannerStats::default(),
            name,
            spare_ids: Vec::new(),
            wb_applied: (false, false),
        }
    }

    pub fn stats(&self) -> PlannerStats {
        self.stats
    }

    /// The planner configuration in effect now (the preset, or the preset with the wayblock hall's
    /// overrides applied by [`Brain::set_live_context`]).
    pub fn current_config(&self) -> PlannerConfig {
        self.planner.config()
    }

    /// The planner and the previous input, for the 8.2 teacher (`crate::teacher`), which labels
    /// states through [`Brain::decide_in`] and reads the planner's search statistics afterwards.
    pub(crate) fn planner_mut(&mut self) -> &mut Planner<PhysicsWorld> {
        &mut self.planner
    }

    pub(crate) fn prev_input(&self) -> PlayerInput {
        self.prev
    }

    pub(crate) fn set_prev_input(&mut self, prev: PlayerInput) {
        self.prev = prev;
    }

    /// The single decision path both entry points share: `world` is the private planning world,
    /// already synced; `lag`/`in_flight` describe this client's unapplied inputs.
    fn plan(
        &mut self,
        world: &mut PhysicsWorld,
        self_id: i32,
        target_id: i32,
        in_flight: &[ddai_physics::core::PlayerInput],
    ) -> Action {
        if self.spare_ids.contains(&target_id) {
            // A spared tee is never a target (the bot's picker does not choose one); do not plan
            // against it if a caller hands one over anyway.
            return action_from_input(&self.prev);
        }
        let (Some(me), Some(target)) = (world.get_tee(self_id), world.get_tee(target_id)) else {
            return action_from_input(&self.prev);
        };
        if !me.alive || !target.alive {
            return action_from_input(&self.prev);
        }
        let enemy_input = enemy_input_from_tee(&target);
        // Everybody else keeps doing what a snapshot shows them doing.
        for other in world.all_tees() {
            if other.id != self_id && other.id != target_id {
                world.set_held_input(other.id, enemy_input_from_tee(&other));
            }
        }
        world.set_held_input(target_id, enemy_input);
        // Roll the private world to the tick the decision will take effect on (`syncPlanningWorld`).
        world.set_held_input(self_id, in_flight.first().map_or(self.prev, from_ddnet_input));
        let start_tick = world.tick();
        for wire in in_flight {
            world.set_input(self_id, from_ddnet_input(wire));
            world.set_input(target_id, enemy_input);
            world.step();
        }
        self.planner.set_live_tick(start_tick);

        let mut frozen = Vec::new();
        let mut frozen_vel = Vec::new();
        for other in world.all_tees() {
            if other.id != self_id
                && other.id != target_id
                && other.alive
                && other.frozen
                && !self.spare_ids.contains(&other.id)
                && js::hypot2(other.pos.x - me.pos.x, other.pos.y - me.pos.y) <= BYSTANDER_PX
            {
                frozen.push(other.pos);
                frozen_vel.push(other.vel);
            }
        }
        self.planner.set_frozen_bystanders(frozen, frozen_vel);

        let out = match self.cfg.mode {
            PlannerMode::Fixed => self.planner.decide(world, self_id, target_id, self.prev, enemy_input),
            PlannerMode::Deadline { budget_ms } => self.planner.decide_production(
                world,
                self_id,
                target_id,
                self.prev,
                enemy_input,
                &self.clock,
                budget_ms,
            ),
        };
        self.stats.add(&self.planner.last_info);
        self.prev = out;
        action_from_input(&out)
    }
}

impl Brain for PlannerBrain {
    fn reset(&mut self, ctx: &ResetContext) {
        self.planner.set_search_seed(ctx.seed as u32);
        self.prev = empty_input();
        self.stats = PlannerStats::default();
        if self.scratch.as_ref().is_none_or(|s| !Arc::ptr_eq(&s.map, &ctx.map)) {
            self.scratch = Some(Scratch::new(ctx.map.clone()));
        }
    }

    fn decide(&mut self, obs: &Observation) -> Action {
        let Some(target) = target_of(obs).map(|c| c.id) else {
            self.prev = input_from_action(&Action::neutral(), &self.prev);
            return Action::neutral();
        };
        let mut scratch = self.scratch.take().unwrap_or_else(|| Scratch::new(obs.map.clone()));
        let world = scratch.rebuild_from_observation(obs);
        let action = self.plan(world, obs.self_state.id, target, &[]);
        self.scratch = Some(scratch);
        action
    }

    fn decide_in(&mut self, obs: &Observation, view: Option<&WorldView<'_>>) -> Action {
        let Some(view) = view else {
            return self.decide(obs);
        };
        let Some(target) = target_of(obs).map(|c| c.id) else {
            self.prev = input_from_action(&Action::neutral(), &self.prev);
            return Action::neutral();
        };
        let mut scratch = self.scratch.take().unwrap_or_else(|| Scratch::new(obs.map.clone()));
        let world = scratch.sync_exact(view.world);
        let action = self.plan(world, view.self_id, target, view.in_flight);
        self.scratch = Some(scratch);
        action
    }

    fn set_live_context(&mut self, ctx: &LiveContext<'_>) {
        // `setSpareBystanders` (`bot.ts:4777-4783`): the tees the rope must not catch. The planner
        // rejects candidate hooks whose line would catch one. (`setThirdTees` is not passed: the live
        // configuration runs `thirdTeeExposure = 0`, `bot.ts:590`, so it would be ignored anyway.)
        let pos: Vec<Vec2> = ctx
            .spares
            .iter()
            .map(|(p, _)| Vec2 {
                x: f64::from(p.x),
                y: f64::from(p.y),
            })
            .collect();
        let vel: Vec<Vec2> = ctx
            .spares
            .iter()
            .map(|(_, v)| Vec2 {
                x: f64::from(v.x),
                y: f64::from(v.y),
            })
            .collect();
        self.planner.set_spare_bystanders(pos, vel);
        self.spare_ids.clear();
        self.spare_ids.extend_from_slice(ctx.spare_ids);
        self.planner.set_travel_goal(ctx.travel_goal.map(|g| Vec2 {
            x: f64::from(g.x),
            y: f64::from(g.y),
        }));
        // Task 4.2: `planner.setOverrides(wbPlanOverrides(self))` / `setBand(wbBand(self))`
        // (`bot.ts:4778-4779`): inside a held wayblock hall the planner runs `WB_PLAN_OVERRIDES` (and
        // `STRONG_WB` on top when the bot is in strong mode and the base population is below 40).
        let want = (ctx.wb.in_hall, ctx.wb.in_hall && ctx.wb.strong);
        if want != self.wb_applied {
            self.wb_applied = want;
            // af49dfb's `WB_PLAN_OVERRIDES` also names `launchExposure 1.0` and `jumplessHazardCost 0.15` (it matters on top of
            // its live config, which raises both); the guard's wall plans (`wallDir`) are not wired here.
            let (wb, strong): (OverrideFn, OverrideFn) = match self.cfg.preset.version() {
                PlannerVersion::Classic => (wb_overrides, preset_strong_wb),
                PlannerVersion::Upstream20261002 => (wb_overrides_v2, preset_strong_wb_v2),
            };
            self.planner.set_overrides(match want {
                (false, _) => None,
                (true, false) => Some(Box::new(wb)),
                (true, true) => Some(Box::new(
                    move |base: PlannerConfig| {
                        if base.population < 40 { strong(base) } else { wb(base) }
                    },
                )),
            });
        }
        self.planner.set_band(
            ctx.wb
                .band
                .filter(|_| ctx.wb.in_hall)
                .map(|(x0, y0, x1, y1)| (f64::from(x0), f64::from(y0), f64::from(x1), f64::from(y1))),
        );
    }

    fn set_map_knowledge(&mut self, k: &MapKnowledge) {
        // `planner.setDeadZone(deadZone)` / the freeze memory (`bot.ts` `memory`, `memoryTrust` from
        // the preset): shared snapshots (no copy), the bot keeps owning, mutating and saving its own.
        self.planner
            .set_dead_zone(k.dead_zone.as_ref().map(|cells| DeadZoneGrid {
                width: k.width,
                cells: Arc::clone(cells),
            }));
        self.planner.set_freeze_memory(k.freeze_memory.as_ref().and_then(|m| {
            FreezeMemory::from_shared(k.width, k.height, Arc::clone(&m.cells), Arc::clone(&m.passes), m.events)
        }));
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn last_plan(&self) -> Option<ddai_brain::PlanTelemetry> {
        let i = &self.planner.last_info;
        Some(ddai_brain::PlanTelemetry {
            searched: i.searched,
            out_of_time: i.out_of_time,
            shielded: i.shielded,
            shield_incomplete: i.shield_incomplete,
            candidates: i.candidates.max(0) as u32,
            proposal_us: 0,
            search_us: 0,
        })
    }

    fn telemetry(&self) -> Option<String> {
        let s = self.stats;
        Some(format!(
            "{{\"decisions\":{},\"searched\":{},\"out_of_time\":{},\"shielded\":{},\"shield_incomplete\":{},\"candidates\":{}}}",
            s.decisions, s.searched, s.out_of_time, s.shielded, s.shield_incomplete, s.candidates
        ))
    }
}
