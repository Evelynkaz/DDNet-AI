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

use ddai_brain::{Action, Brain, CharacterObservation, IVec2, Observation, ResetContext, WorldView};
use ddai_jsmath as js;
use ddai_jsmath::Rng;
use ddai_physics::map::MapData;

use crate::clock::{Clock, StepClock, WallClock};
use crate::config::{PlannerConfig, preset_low_cpu, preset_normal, preset_strong_wb};
use crate::physics_adapter::{PhysicsWorld, from_ddnet_input};
use crate::plan_world::PlanWorld;
use crate::planner::{DecisionInfo, Planner};
use crate::scripted::scripted_action;
use crate::types::{PlayerInput, TeeState, blank_tee_state, empty_input};
use crate::vmath::Vec2;

/// `BYSTANDER_PX` (`bot.ts:339`): frozen non-target tees closer than this are passed to the
/// planner as frozen bystanders (`bot.ts:4767-4775`).
const BYSTANDER_PX: f64 = 160.0;

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

/// A private planning world, rebuilt lazily for the map of the current episode.
struct Scratch {
    map: Arc<MapData>,
    world: Option<PhysicsWorld>,
}

impl Scratch {
    fn new(map: Arc<MapData>) -> Self {
        Scratch { map, world: None }
    }

    /// Copies the exact world in. The first call clones it (sharing the `Arc`'d collision); later
    /// ones `restore_from` into the same allocation.
    fn sync_exact(&mut self, src: &ddai_physics::world::World<f32>) -> &mut PhysicsWorld {
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
    fn rebuild_from_observation(&mut self, obs: &Observation) -> &mut PhysicsWorld {
        let mut w = PhysicsWorld::new(self.map.clone(), 1);
        for c in std::iter::once(&obs.self_state).chain(obs.others.iter()) {
            let st = tee_from_observation(c);
            w.add_tee(c.id, st.pos);
            w.apply_tee_state(c.id, &st);
        }
        self.world = Some(w);
        self.world.as_mut().expect("just set")
    }
}

/// Which of the target-finding fields the brain resolves the opponent from.
fn target_of(obs: &Observation) -> Option<&CharacterObservation> {
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlannerPreset {
    Normal,
    Low,
    Strong,
}

impl PlannerPreset {
    pub fn config(self) -> PlannerConfig {
        match self {
            PlannerPreset::Normal => preset_normal(),
            PlannerPreset::Low => preset_low_cpu(),
            PlannerPreset::Strong => preset_strong_wb(preset_normal()),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            PlannerPreset::Normal => "normal",
            PlannerPreset::Low => "low",
            PlannerPreset::Strong => "strong",
        }
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
/// for the world/opponent handling. Not `Send` (the planner holds `Rc`s): build it inside the
/// thread that plays the game.
pub struct PlannerBrain {
    cfg: PlannerBrainConfig,
    planner: Planner<PhysicsWorld>,
    clock: BrainClock,
    prev: PlayerInput,
    scratch: Option<Scratch>,
    stats: PlannerStats,
    name: String,
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
            planner: Planner::new(cfg.preset.config()),
            cfg,
            clock,
            prev: empty_input(),
            scratch: None,
            stats: PlannerStats::default(),
            name,
        }
    }

    pub fn stats(&self) -> PlannerStats {
        self.stats
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

    fn name(&self) -> &str {
        &self.name
    }

    fn telemetry(&self) -> Option<String> {
        let s = self.stats;
        Some(format!(
            "{{\"decisions\":{},\"searched\":{},\"out_of_time\":{},\"shielded\":{},\"shield_incomplete\":{},\"candidates\":{}}}",
            s.decisions, s.searched, s.out_of_time, s.shielded, s.shield_incomplete, s.candidates
        ))
    }
}
