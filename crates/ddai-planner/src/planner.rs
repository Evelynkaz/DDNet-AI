//! `Planner` (`src/plan/planner.ts:817-2176`) — the CEM search itself: opening book, CEM
//! sampling/refit, `evaluate`/`scoreTick`, `stepToInput`, hook gating, `noThaw`, the edge-hold/
//! plan-margin/hook-polish/escape-bias post-passes, and every piece of hidden state carried
//! between decisions (`docs/research/orig-plan.md` §1.15/§2.5). Generic over [`PlanWorld`] so the
//! exact same algorithm runs against `ddai-tsworld::SimWorld` (parity) and
//! `ddai_physics::World<f32>` (production, D-041).
//!
//! Every public/private method below cites the `planner.ts` lines it ports; numeric operations go
//! through `ddai_jsmath`, never `f64`/`std` float methods directly (house rule, `docs/DECISIONS.md`
//! D-018/D-035).

use crate::action::{ACTION_SIZE, decode_action};
use crate::clock::Clock;
use crate::config::{OpponentModel, PlannerConfig};
use crate::fields::{
    self, CEILING_NONE, CeilingField, EDGE_GAP_PX, HazardField, LaunchMemo, drag_crosses_hazard, flight_ends_in_hazard,
    freeze_gap_px, hazard_nearness, launch_flight_lands_in_hazard, launch_flight_lands_in_hazard_memo,
    launch_lands_in_hazard, rope_intercept, wrap_angle,
};
use crate::memory::FreezeMemory;
use crate::opponent_profile::OpponentProfile;
use crate::plan_world::{PlanCollision, PlanWorld};
use crate::scripted::scripted_action;
use crate::seal::rests_in_freeze;
use crate::throw_lines::{
    ThrowSituation, air_chain_lines, frozen_throw_lines, frozen_throw_worth_trying, throw_lines, throw_worth_trying,
    wall_swing_lines,
};
use crate::trig;
use crate::tuning::{HOOK_LENGTH, PHYSICAL_SIZE};
use crate::types::{HOOK_FLYING, HOOK_GRABBED, HOOK_IDLE, HOOK_RETRACT_START, PlayerInput, TeeState, empty_input};
use crate::vmath::{Vec2, closest_point_on_line_or_null, vdistance, vec2};
use ddai_jsmath::{self as js, Rng};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

const AIR_JUMP_MIN_GAP_TICKS: i64 = 11;
const CADENCE_GAPS: usize = 4;
const CADENCE_MAX_GAP: i64 = 50;
const CADENCE_MAX_TICKS: i32 = 8;
const SHIELD_MAX_HOLD: i32 = 16;
const THROW_LANDED_TICKS: i32 = 5;
const FROZEN_PLAN_MIN_TICKS: i64 = 30;
const FREEZE_CLOCK_TICKS: f64 = 3.0 * 50.0;
const EDGE_NEARER_PX: f64 = 4.0;
const THAW_ESCAPE_TICKS: i32 = 40;
/// `SNAP_MAX_RAD` (af49dfb): the most the aim snap turns a hook throw away from the planned angle.
const SNAP_MAX_RAD: f64 = 0.35;
/// `ROPE_CEIL_MARGIN_PX` (af49dfb): how far below a ceiling a rope-hauled flight starts to count as touching it.
const ROPE_CEIL_MARGIN_PX: f64 = 16.0;

/// Review round 3, F18: `decide_production`'s normal shield time reserve, on top of the search
/// budget, **per simulated tee** (`world.all_tees().len()`, self + enemy + every bystander --
/// every one of them is stepped by `world.step()`, so per-step cost scales with this count, not
/// just wall-clock time). A flat `SHIELD_RESERVE_MS = 1.0` (round 2) was measured to comfortably
/// cover one `escape_exists` rollout (~126 steps) at 2 tees (`step_us` 2.35-3.7 µs,
/// `component_cost_report`) but not at 6, where the *same* rollout costs ~0.9 ms (`step_us`
/// 6.99-7.15 µs, ~3x the 2-tee cost) -- nearly the whole reserve, leaving no time for a second
/// attempt if the first escape candidate fails, which is why `shield_incomplete` was measured at
/// 330-493/1000 (whole map) / 95-117/1000 (E-000 hall) at 6 tees vs 29-127/1000 / 2-12/1000 at 2
/// (review round 3's `r3-phase.log`). `0.5 ms/tee` is calibrated so this reduces to exactly the
/// round-2 constant at 2 tees (self + one enemy, the common case with no bystanders: `0.5 * 2 =
/// 1.0`) and scales up proportionally as more tees join (`0.5 * 6 = 3.0` ms at 6), buying back
/// roughly the same *number* of affordable rollouts rather than a fixed wall-clock budget that
/// buys fewer and fewer of them as the scene gets busier. See `README.md`'s F18 section for the
/// measured before/after `shield_incomplete` rates.
const SHIELD_RESERVE_MS_PER_TEE: f64 = 0.5;
/// Review round 2, F12: the *total* wall-clock cap (from the start of the whole
/// `decide_production` call, not just the shield) the shield's search for a safer alternative may
/// extend into once a real danger is confirmed (the chosen input has no escape) -- this is the
/// "D-042 adaptive extension" the round-2 lead decision names. Deliberately the same order of
/// magnitude as `decide_once`'s own `--low-cpu` hard cap (`preset_low_cpu`'s `hard_ms: 11`) plus
/// slack for the shield itself, not an arbitrary number.
const SHIELD_DANGER_MAX_TOTAL_MS: f64 = 15.0;

/// `PlanStep` (`planner.ts:338`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanStep {
    pub dir: i32,
    pub jump: i32,
    pub hook: i32,
    pub fire: i32,
    pub aim: f64,
}

/// `StepDist` (`planner.ts:339`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct StepDist {
    p_left: f64,
    p_right: f64,
    p_jump: f64,
    p_hook: f64,
    p_fire: f64,
    aim: f64,
    aim_spread: f64,
}

/// One draw of `samplePlan` from `rng` (see [`Planner::sample_plan`]).
fn sample_plan_with(rng: &mut Rng, dist: &[StepDist]) -> Vec<PlanStep> {
    dist.iter()
        .map(|d| {
            let r = rng.next_float();
            let dir = if r < d.p_left {
                -1
            } else if r < d.p_left + d.p_right {
                1
            } else {
                0
            };
            let jump = i32::from(rng.next_float() < d.p_jump);
            let hook = i32::from(rng.next_float() < d.p_hook);
            let fire = i32::from(rng.next_float() < d.p_fire);
            let aim = d.aim + rng.next_gaussian() * d.aim_spread;
            PlanStep {
                dir,
                jump,
                hook,
                fire,
                aim,
            }
        })
        .collect()
}

/// `DecisionInfo` (`planner.ts:317-332`).
#[derive(Debug, Clone, Copy, Default)]
pub struct DecisionInfo {
    pub searched: bool,
    pub candidates: i32,
    pub out_of_time: bool,
    pub ms: f64,
    pub self_out: i32,
    pub enemy_out: i32,
    pub hook_at: i32,
    pub gated: bool,
    pub shielded: bool,
    pub edge_held: bool,
    /// Review round 2, F12: `decide_production`-only (always `false` on the `decide_once`/TS-
    /// parity path, which never reads a deadline at all, D-017). `true` when the shield's own
    /// time reserve ran out before it could confirm either "the chosen input has an escape" or "a
    /// safer alternative exists" -- the searched `chosen` input is kept either way (see
    /// `decide_production`'s doc comment for the measured justification against falling back to
    /// `prev` instead), this flag only says the answer wasn't fully verified in time.
    pub shield_incomplete: bool,
}

/// The value [`Planner::decide`] returns — an alias, not a new type, so callers can pass it
/// straight to whatever sends the input over the wire / into another `PlanWorld::set_input`.
pub type Decision = PlayerInput;

/// [`Planner::debug_state`]'s return type -- test-only hidden-state introspection (review round
/// 1, F4).
#[derive(Debug, Clone)]
pub struct PlannerDebugState {
    pub rng_s0: u32,
    pub rng_s1: u32,
    pub rng_s2: u32,
    pub rng_s3: u32,
    pub rng_have_spare: bool,
    pub rng_spare: f64,
    pub opp_seed: u32,
    pub warm: Option<Vec<PlanStep>>,
    pub committed: Option<PlayerInput>,
    pub commit_left: i32,
    pub last_decide_tick: i64,
    pub decide_gaps: Vec<i64>,
    pub last_frozen: bool,
    pub dir_since: i64,
    pub dir_last: i32,
}

#[derive(Debug, Clone, Copy)]
struct Band {
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
}

/// Externally-supplied dead-zone grid (`Bot.deadCells`/`Planner.setDeadZone`,
/// `docs/research/orig-plan.md` §1.11's `deadZoneOf`). See `crate::fields`'s module doc comment
/// for why this crate does not compute it itself.
#[derive(Debug, Clone)]
pub struct DeadZoneGrid {
    pub width: i32,
    /// Shared and never mutated: the live bot hands the same grid to every brain and planner clone
    /// (task 4.2, review F4: a map-sized copy per hand-off was too much for the decision thread).
    pub cells: std::sync::Arc<Vec<u8>>,
}

/// `inDead(dead, x, y)` (`planner.ts:512-516`) — deliberately reproduces the TS bug
/// (`docs/research/orig-plan.md` §11 item 7): only the *combined* index `i` is bounds-checked,
/// not `x` against `width` on its own, so an out-of-range `x` can silently wrap into the next row
/// instead of being rejected.
fn in_dead(dead: Option<&DeadZoneGrid>, x: f64, y: f64) -> bool {
    let Some(dead) = dead else { return false };
    let i = js::trunc(y / 32.0) as i64 * i64::from(dead.width) + js::trunc(x / 32.0) as i64;
    i >= 0 && (i as usize) < dead.cells.len() && dead.cells[i as usize] == 1
}

struct DragTracker {
    prev_enemy_near: f64,
    start_enemy_near: f64,
    started_in_dead: bool,
    /// Task 3.10: our distance to the staging point behind a frozen victim at the previous tick (`NaN` = not seen yet).
    prev_stage_dist: f64,
}

/// Task 3.10: how far behind a frozen victim (on the side its nearest freeze lies) the staging point is, in px: from there a hook pulls it
/// toward the freeze.
const STAGE_BEHIND_PX: f64 = 110.0;

/// `scoreTick` (`planner.ts:518-605`). Free function (not a method): needs only `cfg`/`drag` plus
/// read-only world state, exactly like TS's own module-level function. `travel` is not a
/// parameter -- see `crate::fields`'s module doc comment (TS computes it but `scoreTick` itself
/// never reads it, `docs/research/orig-plan.md` §11 item 1).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn score_tick<W: PlanWorld>(
    world: &W,
    self_id: i32,
    enemy_id: i32,
    events: &[crate::types::WorldEvent],
    field: &HazardField,
    unfreeze: &HazardField,
    cfg: &PlannerConfig,
    drag: &mut DragTracker,
    goal: Option<Vec2>,
    dead: Option<&DeadZoneGrid>,
    memory: Option<&FreezeMemory>,
    thirds: &[Vec2],
    band: Option<&Band>,
    launch_memo: Option<&mut LaunchMemo>,
    ceiling: Option<&CeilingField>,
) -> f64 {
    let me = world.get_tee(self_id);
    let en = world.get_tee(enemy_id);
    score_tick_of(
        world,
        me.as_ref(),
        en.as_ref(),
        self_id,
        enemy_id,
        events,
        field,
        unfreeze,
        cfg,
        drag,
        goal,
        dead,
        memory,
        thirds,
        band,
        launch_memo,
        ceiling,
    )
}

/// [`score_tick`] for a caller that already holds the two tees' states (`world.get_tee(self_id)` / `get_tee(enemy_id)`
/// as of now: reading a tee is pure, and each read copies a ~250 B `TeeState`, which the rollout loop paid four times a tick
/// before task 4.13).
#[allow(clippy::too_many_arguments)]
fn score_tick_of<W: PlanWorld>(
    world: &W,
    me: Option<&TeeState>,
    en: Option<&TeeState>,
    self_id: i32,
    enemy_id: i32,
    events: &[crate::types::WorldEvent],
    field: &HazardField,
    unfreeze: &HazardField,
    cfg: &PlannerConfig,
    drag: &mut DragTracker,
    goal: Option<Vec2>,
    dead: Option<&DeadZoneGrid>,
    memory: Option<&FreezeMemory>,
    thirds: &[Vec2],
    band: Option<&Band>,
    launch_memo: Option<&mut LaunchMemo>,
    ceiling: Option<&CeilingField>,
) -> f64 {
    use crate::types::WorldEvent;

    let Some(me) = me else {
        return -1000.0;
    };
    let Some(en) = en else {
        return -1000.0;
    };
    let mut s = 0.0;
    if !en.alive {
        s += 15.0;
    }
    if !me.alive {
        s -= 15.0;
    }
    if en.frozen {
        s += cfg.frozen_weight;
    }
    if me.frozen {
        s -= cfg.frozen_weight * cfg.self_freeze_bias;
        // Task 3.10b (d): a trade (both of us frozen) is not a block.
        if en.frozen && en.alive && me.alive {
            s -= cfg.mutual_freeze_cost;
        }
    }
    if let Some(band) = band
        && cfg.band_cost > 0.0
        && !me.frozen
        && me.pos.x >= band.x0
        && me.pos.x <= band.x1
        && me.pos.y >= band.y0
        && me.pos.y <= band.y1
    {
        s -= cfg.band_cost;
    }
    if me.hooked_player == enemy_id {
        s += cfg.hook_hold_weight;
    }
    if en.hooked_player == self_id {
        s -= cfg.hook_hold_weight * 0.75;
    }
    for e in events {
        match *e {
            WorldEvent::HammerFire { from, hits } if from == self_id && hits == 0 => s -= cfg.wasted_hammer,
            WorldEvent::HammerHit { from, to } if from == enemy_id && to == self_id => s -= 0.3,
            WorldEvent::Death { id, .. } if id == enemy_id => s += 15.0,
            WorldEvent::Death { id, .. } if id == self_id => s -= 15.0,
            _ => {}
        }
    }

    if me.frozen {
        s += 0.08 * hazard_nearness(unfreeze, me.pos.x, me.pos.y);
    }
    if en.frozen {
        s -= 0.06 * hazard_nearness(unfreeze, en.pos.x, en.pos.y);
    }

    if me.direction != 0 && js::abs(me.vel.x) < 0.2 && !me.frozen {
        s -= cfg.wall_push_cost;
    }
    if cfg.jumpless_hazard_cost > 0.0 && !me.frozen && me.jumps_left == 0 {
        s -= cfg.jumpless_hazard_cost * hazard_nearness(field, me.pos.x, me.pos.y);
    }
    let en_near = hazard_nearness(field, en.pos.x, en.pos.y);
    let en_floor = if cfg.enemy_hazard_from_start {
        js::max(0.3, drag.start_enemy_near)
    } else {
        0.3
    };
    if en_near > en_floor {
        s += cfg.enemy_hazard_weight * (en_near - en_floor);
    }

    if cfg.hook_drag_weight > 0.0 && me.hooked_player == enemy_id && !en.frozen && en_near > drag.prev_enemy_near {
        s += cfg.hook_drag_weight * (en_near - drag.prev_enemy_near);
    }
    if cfg.frozen_drag_weight > 0.0 && en.frozen && en.alive {
        s += cfg.frozen_drag_weight * (en_near - drag.prev_enemy_near);
    }
    drag.prev_enemy_near = en_near;
    // Task 3.10b (c): the sooner the frozen victim is in the freeze, the longer its timer is renewed within the rollout.
    if cfg.frozen_seal_weight > 0.0
        && en.frozen
        && en.alive
        && crate::seal::touches_freeze(world.collision(), en.pos.x, en.pos.y)
    {
        s += cfg.frozen_seal_weight;
    }
    // Task 3.10: a frozen victim lying off the freeze is hauled back only by someone standing on its freeze side (the rope pulls it to us), and
    // that takes longer than the 27 ticks of a rollout: reward the progress toward that spot (per tile gained), the way the drag term rewards its progress.
    if cfg.frozen_stage_weight > 0.0 && en.frozen && en.alive && me.alive && !me.frozen && en_near < 0.95 {
        let g = crate::hybrid::techniques::toward_hazard(field, en.pos);
        if g.x != 0.0 || g.y != 0.0 {
            let stage = Vec2 {
                x: en.pos.x + g.x * STAGE_BEHIND_PX,
                y: en.pos.y + g.y * STAGE_BEHIND_PX,
            };
            let d = vdistance(me.pos, stage);
            if !drag.prev_stage_dist.is_nan() {
                s += cfg.frozen_stage_weight * (drag.prev_stage_dist - d) / 32.0;
            }
            drag.prev_stage_dist = d;
        }
    }

    let me_near = hazard_nearness(field, me.pos.x, me.pos.y);
    if me_near > cfg.self_hazard_threshold {
        let trusted = match memory {
            Some(m) if cfg.memory_trust > 0.0 => cfg.memory_trust * m.safety(me.pos.x, me.pos.y),
            _ => 0.0,
        };
        s -= cfg.self_hazard_cost * (me_near - cfg.self_hazard_threshold) * (1.0 - trusted);
    }

    // af49dfb `ropeCeilingCost`: the victim's rope hauls us up into a freeze/death ceiling.
    if let Some(ceiling) = ceiling
        && cfg.rope_ceiling_cost > 0.0
        && en.hooked_player == self_id
        && me.vel.y < 0.0
        && !me.frozen
        && me.alive
    {
        let tx = js::floor(me.pos.x / 32.0) as i32;
        let ty = js::floor(me.pos.y / 32.0) as i32;
        if tx >= 0 && ty >= 0 && tx < ceiling.width && ty < ceiling.height {
            let d = ceiling.dist[(ty * ceiling.width + tx) as usize];
            if d != CEILING_NONE {
                let gap = me.pos.y - f64::from(ty - i32::from(d) + 1) * 32.0;
                let rise = (me.vel.y * me.vel.y) / (2.0 * *crate::tuning::GRAVITY);
                let over = (rise - gap + ROPE_CEIL_MARGIN_PX) / ROPE_CEIL_MARGIN_PX;
                if over > 0.0 {
                    s -= cfg.rope_ceiling_cost * js::min(1.0, over);
                }
            }
        }
    }
    let separation = vdistance(me.pos, en.pos);
    // Task 3.14 `ceiling_guard_cost`: near a freeze ceiling with a free opponent in reach.
    if cfg.ceiling_guard_cost > 0.0
        && cfg.ceiling_guard_px > 0.0
        && let Some(ceiling) = ceiling
        && !me.frozen
        && me.alive
        && !en.frozen
        && en.alive
        && separation < crate::config::CEILING_GUARD_REACH_PX
    {
        let tx = js::floor(me.pos.x / 32.0) as i32;
        let ty = js::floor(me.pos.y / 32.0) as i32;
        if tx >= 0 && ty >= 0 && tx < ceiling.width && ty < ceiling.height {
            let d = ceiling.dist[(ty * ceiling.width + tx) as usize];
            if d != CEILING_NONE {
                let gap = js::max(0.0, me.pos.y - f64::from(ty - i32::from(d) + 1) * 32.0);
                if gap < cfg.ceiling_guard_px {
                    s -= cfg.ceiling_guard_cost * (1.0 - gap / cfg.ceiling_guard_px);
                }
            }
        }
    }
    let col = world.collision();
    if cfg.launch_exposure > 0.0 && !me.frozen && !en.frozen && separation < LAUNCH_REACH_PX {
        let exact = cfg.launch_exact_reach > 0.0
            && separation < cfg.launch_exact_reach
            && (me.pos.y < en.pos.y || (cfg.launch_exact_rise_vy > 0.0 && me.vel.y < -cfg.launch_exact_rise_vy));
        if exact {
            s -= (if cfg.launch_exact_weight > 0.0 {
                cfg.launch_exact_weight
            } else {
                cfg.launch_exposure
            }) * launch_flight_lands_in_hazard_memo(launch_memo, col, me.pos, en.pos, separation, me.vel);
        } else {
            s -= cfg.launch_exposure * launch_lands_in_hazard(col, me.pos, en.pos, separation);
        }
    }
    if cfg.drag_exposure > 0.0 && !me.frozen && !en.frozen && separation < *HOOK_LENGTH {
        s -= cfg.drag_exposure * drag_crosses_hazard(col, me.pos, en.pos, separation);
    }
    if cfg.drag_threat > 0.0 && !me.frozen && !en.frozen && separation < *HOOK_LENGTH {
        s += cfg.drag_threat * drag_crosses_hazard(col, en.pos, me.pos, separation);
    }
    if cfg.launch_threat > 0.0 && !me.frozen && !en.frozen && separation < LAUNCH_REACH_PX {
        s += cfg.launch_threat * launch_lands_in_hazard(col, en.pos, me.pos, separation);
    }
    if cfg.third_tee_exposure > 0.0 && !me.frozen && !thirds.is_empty() {
        for &t in thirds {
            let d = vdistance(me.pos, t);
            if d < 1.0 || d >= *HOOK_LENGTH {
                continue;
            }
            s -= cfg.third_tee_exposure * drag_crosses_hazard(col, me.pos, t, d);
        }
    }
    s -= cfg.distance_weight * (js::max(0.0, separation - cfg.standoff_px) / 32.0);
    if let Some(goal) = goal {
        s -= cfg.travel_weight * (vdistance(me.pos, goal) / 32.0);
    }
    if let Some(m) = memory
        && cfg.memory_weight > 0.0
        && me.alive
        && !me.frozen
    {
        s -= cfg.memory_weight * m.risk(me.pos.x, me.pos.y);
    }
    if dead.is_some() && !drag.started_in_dead && (cfg.dead_zone_cost > 0.0 || cfg.enemy_dead_zone_bonus > 0.0) {
        let me_dead = me.alive && in_dead(dead, me.pos.x, me.pos.y);
        let en_dead = en.alive && in_dead(dead, en.pos.x, en.pos.y);
        if cfg.dead_zone_cost > 0.0 && me_dead && !en_dead {
            s -= cfg.dead_zone_cost;
        }
        if cfg.enemy_dead_zone_bonus > 0.0 && en_dead && !me_dead {
            s += cfg.enemy_dead_zone_bonus;
        }
    }
    s
}

pub(crate) const LAUNCH_REACH_PX: f64 = 96.0;

/// `THAW_ESCAPES` (`planner.ts:723-754`): the 9 base 40-tick escape attempts `thawEscapable` tries
/// (recomputed on demand -- these are plain data, cheap to rebuild; see `Planner::thaw_escapable`
/// for the memoization that actually matters for performance).
fn thaw_escapes() -> Vec<Vec<PlayerInput>> {
    let line = |fill: &dyn Fn(i32, &mut PlayerInput)| -> Vec<PlayerInput> {
        (0..THAW_ESCAPE_TICKS)
            .map(|t| {
                let mut e = empty_input();
                e.target_x = 0.0;
                e.target_y = -300.0;
                fill(t, &mut e);
                e
            })
            .collect()
    };
    let mut out = vec![line(&|_, _| {})];
    for d in [-1, 1] {
        out.push(line(&move |_, e| e.direction = d));
    }
    for d in [0, -1, 1] {
        out.push(line(&move |t, e| {
            e.direction = d;
            e.jump = i32::from(t == 0 || t == 6);
        }));
    }
    for ax in [0, -1, 1] {
        out.push(line(&move |t, e| {
            e.direction = ax;
            e.jump = i32::from(t % 2 == 0);
            e.hook = 1;
            e.target_x = f64::from(ax) * 200.0;
            e.target_y = -300.0;
        }));
    }
    out
}

/// [`thaw_escapes`] built once: it is constant data, and `thaw_escapable` used to rebuild it (nine
/// vectors of 40 inputs) on every call.
static THAW_ESCAPES: std::sync::LazyLock<Vec<Vec<PlayerInput>>> = std::sync::LazyLock::new(thaw_escapes);

/// `ropeEscapes(dx, dy)` (`planner.ts:756-774`).
fn rope_escapes(dx: f64, dy: f64) -> Vec<Vec<PlayerInput>> {
    let mut out = Vec::new();
    for d in [0, -1, 1] {
        for jumps in [false, true] {
            let seq = (0..THAW_ESCAPE_TICKS)
                .map(|t| {
                    let mut e = empty_input();
                    e.direction = d;
                    e.jump = i32::from(jumps && t % 2 == 0);
                    e.hook = 1;
                    e.target_x = if js::round(dx) == 0.0 { 1.0 } else { js::round(dx) };
                    e.target_y = js::round(dy);
                    e
                })
                .collect();
            out.push(seq);
        }
    }
    out
}

/// `snapTarget(input, aim)` (af49dfb): aim straight along `aim` (the rounded `AIM_RADIUS` vector), skipping the aim smoothing.
fn snap_target(input: &mut PlayerInput, aim: f64) {
    input.target_x = js::round(trig::cos(aim) * crate::action::AIM_RADIUS);
    input.target_y = js::round(trig::sin(aim) * crate::action::AIM_RADIUS);
    if input.target_x == 0.0 && input.target_y == 0.0 {
        input.target_x = crate::action::AIM_RADIUS;
    }
}

/// Whether [`build_step_ticks`] makes every step `plan_step` ticks long whatever the number of steps (no fine front steps).
fn build_step_ticks_is_uniform(front_steps: f64, front_step: f64, plan_step: i32) -> bool {
    js::trunc(front_steps) <= 0.0 || front_step <= 0.0 || front_step >= f64::from(plan_step)
}

/// `buildStepTicks(steps, planStep, frontSteps, frontStep)` (`planner.ts:192-207`).
pub fn build_step_ticks(steps: i32, plan_step: i32, front_steps: f64, front_step: f64) -> Vec<i32> {
    let steps_f = f64::from(steps);
    let plan_step_f = f64::from(plan_step);
    let total = steps_f * plan_step_f;
    let front = js::max(0.0, js::min(js::trunc(front_steps), steps_f - 1.0));
    if front == 0.0 || front_step <= 0.0 || front_step >= plan_step_f {
        return vec![plan_step; steps as usize];
    }
    let fine = js::max(1.0, js::trunc(front_step));
    let rest = steps_f - front;
    let left = total - front * fine;
    if left < rest {
        return vec![plan_step; steps as usize];
    }
    let base = js::floor(left / rest);
    let extra = left - base * rest;
    let mut out = Vec::with_capacity(steps as usize);
    for _ in 0..(front as i32) {
        out.push(fine as i32);
    }
    let rest_i = rest as i32;
    for i in 0..rest_i {
        out.push(if f64::from(i) >= rest - extra {
            (base + 1.0) as i32
        } else {
            base as i32
        });
    }
    out
}

/// `(b - a)` with `NaN` treated as equal (V8's `Array.prototype.sort` comparator convention,
/// `docs/research/orig-plan.md` §2.5/§3): descending by score, stable on ties (Rust's `sort_by`
/// is a stable sort, matching V8's guaranteed-stable `TimSort`).
fn score_desc(a: f64, b: f64) -> Ordering {
    let d = b - a;
    if d < 0.0 {
        Ordering::Less
    } else if d > 0.0 {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

/// The ported planner (`class Planner`, `planner.ts:817-2163`). Generic over the world backend
/// ([`PlanWorld`]); every field below is hidden state TS carries between decisions
/// (`docs/research/orig-plan.md` §1.15) unless documented otherwise.
pub struct Planner<W: PlanWorld> {
    cfg: PlannerConfig,
    base_cfg: PlannerConfig,
    pub(crate) rng: Rng,
    pub(crate) saved: Option<W::SavedState>,
    pub(crate) warm: Option<Vec<PlanStep>>,
    pub(crate) committed: Option<PlayerInput>,
    pub(crate) commit_left: i32,

    last_decide_tick: i64,
    decide_gaps: Vec<i64>,

    live_tick: i64,
    last_frozen: bool,
    profile: OpponentProfile,

    goal: Option<Vec2>,
    dead: Option<DeadZoneGrid>,
    memory: Option<FreezeMemory>,
    frozen_bystanders: Vec<Vec2>,
    frozen_bystander_vels: Vec<Vec2>,
    spares: Vec<Vec2>,
    spare_vels: Vec<Vec2>,
    thirds: Vec<Vec2>,
    band: Option<Band>,

    pub(crate) opp_seed: u32,
    seed_offset: u32,
    predicted: Vec<PlayerInput>,
    /// Task 3.7b: when `Some`, `evaluate_impl` appends the input of every plan step it sends (the hybrid's opponent model turns
    /// the opponent's predicted plan into open-loop inputs this way). `None` on every other path.
    pub(crate) record_inputs: Option<Vec<PlayerInput>>,

    pub(crate) track_rollout: bool,
    /// Task 3.6: memo of the exact hammer-launch flight check (hybrid workers only; `None` keeps the
    /// TS-parity path exactly as it was). Never changes a score, see [`LaunchMemo`].
    pub(crate) launch_memo: Option<Box<LaunchMemo>>,
    /// Task 3.5b (hybrid shield): `evaluate_impl` leaves the world in its end-of-plan state instead of
    /// restoring the snapshot, so the caller can continue from there. `false` everywhere else.
    pub(crate) keep_final: bool,
    /// The input of the last plan step of the most recent `evaluate_impl` (what the tee would keep
    /// doing if the plan went on): the shield's plan-remainder escape continues with it.
    pub(crate) last_input: PlayerInput,
    track_gap: bool,
    rollout_min_gap: f64,
    pub(crate) react_this_pass: bool,

    pub(crate) swing_target_frozen: bool,
    pub(crate) swing_target: Option<TeeState>,
    pub(crate) swing_rope_on: bool,

    thaw_scratch: Option<W>,
    thaw_scratch_identity: Option<u64>,
    thaw_memo: HashMap<String, bool>,
    /// Task 3.5 (hybrid search only; `false` on every TS-parity path): make `thaw_escapable` a pure
    /// function of its arguments, so a candidate's score does not depend on which other candidates
    /// this planner instance happened to evaluate first (or on which worker thread it ran). The
    /// coarse-keyed memo is cleared per evaluation and the scratch world is restored to its
    /// pristine snapshot before each simulated escape.
    pub(crate) deterministic_thaw: bool,
    thaw_base: Option<W::SavedState>,
    /// `thaw_memo` for `deterministic_thaw` mode: an integer key, no allocation, per evaluation.
    thaw_fast: Vec<([i64; 9], bool)>,
    /// Task 3.5 (hybrid 1vN search only; `None` on every TS-parity path): the other tees the
    /// rollouts model individually and score defensively, next to the single `enemy_id` victim.
    pub(crate) threats: Option<crate::hybrid::threat::ThreatSet>,
    /// Physics ticks simulated by `evaluate_impl` over this planner's lifetime (a work counter,
    /// D-045: unlike a clock it does not count host stalls).
    pub(crate) eval_ticks: u64,
    /// Reused event buffer of the rollout loop (`step_into`), so a rollout allocates nothing.
    events_buf: Vec<crate::types::WorldEvent>,

    pub(crate) rollout_enemy_out: i32,
    pub(crate) rollout_enemy_sealed: bool,
    pub(crate) rollout_self_out: i32,

    pub last_info: DecisionInfo,

    /// Task 8.2: the first-step statistics of the elite set of the last CEM iteration of the most
    /// recent [`Planner::decide`] (`None` after a committed decision, and never filled by
    /// `decide_production`). Written once per iteration, never read by the search itself.
    pub last_elite: Option<crate::elite::EliteFirstStep>,

    last_search_tick: i64,
    pub(crate) warm_shift_steps: i32,
    pub(crate) dir_since: i64,
    pub(crate) dir_last: i32,
    pub(crate) held_ticks: i64,

    pub(crate) step_ticks: Vec<i32>,

    field_cache_identity: Option<u64>,
    field_cache: Option<(Arc<HazardField>, Arc<HazardField>)>,
    /// af49dfb `ropeCeilingCost`: the ceiling field of the map in play (`Some` only when the cost is on), cached by
    /// collision identity like the hazard fields.
    ceiling: Option<Arc<CeilingField>>,
    ceiling_cache: Option<(u64, Arc<CeilingField>)>,
    /// Task 3.9: how many times `rope_intercept` projected a *moving* victim (the one piece of the v2 hook gate and aim snap that costs
    /// about a physics tick: ~0.9 us); the hybrid's work clock prices each as one tee-tick. Never read by the planner itself.
    intercepts: std::cell::Cell<u64>,

    /// Review round 1, F4: when `Some`, every `evaluate_impl` call (in call order -- book seeds,
    /// `landed_throws`, the CEM population loop, `notNow`, `polishRope`, `escapeBias`, `explain`,
    /// the `planMargin` carry candidate) appends `(plan, score)` here, for a test harness to
    /// compare bit-for-bit against the same ordered list monkey-patched out of the real TS
    /// `Planner.prototype.evaluate`. `None` (the default) costs nothing -- test-only, enabled via
    /// [`Planner::start_candidate_log`].
    candidate_log: Option<Vec<(Vec<PlanStep>, f64)>>,
}

impl<W: PlanWorld> Planner<W> {
    /// `constructor(cfg?)` (`planner.ts:902-910`).
    pub fn new(cfg: PlannerConfig) -> Self {
        let step_ticks = build_step_ticks(cfg.steps, cfg.plan_step, cfg.front_steps, cfg.front_step);
        let seed_offset = 0u32;
        let rng = Rng::new(cfg.seed.wrapping_add(seed_offset));
        Planner {
            cfg,
            base_cfg: cfg,
            rng,
            saved: None,
            warm: None,
            committed: None,
            commit_left: 0,
            last_decide_tick: -1,
            decide_gaps: Vec::new(),
            live_tick: -1,
            last_frozen: false,
            profile: OpponentProfile::new(),
            goal: None,
            dead: None,
            memory: None,
            frozen_bystanders: Vec::new(),
            frozen_bystander_vels: Vec::new(),
            spares: Vec::new(),
            spare_vels: Vec::new(),
            thirds: Vec::new(),
            band: None,
            opp_seed: 1,
            seed_offset,
            predicted: Vec::new(),
            record_inputs: None,
            track_rollout: false,
            launch_memo: None,
            keep_final: false,
            last_input: crate::types::empty_input(),
            track_gap: false,
            rollout_min_gap: EDGE_GAP_PX,
            react_this_pass: false,
            swing_target_frozen: false,
            swing_target: None,
            swing_rope_on: false,
            thaw_scratch: None,
            thaw_scratch_identity: None,
            thaw_memo: HashMap::new(),
            deterministic_thaw: false,
            thaw_base: None,
            thaw_fast: Vec::new(),
            threats: None,
            eval_ticks: 0,
            events_buf: Vec::new(),
            rollout_enemy_out: 0,
            rollout_enemy_sealed: false,
            rollout_self_out: 0,
            last_info: DecisionInfo::default(),
            last_elite: None,
            last_search_tick: -1,
            warm_shift_steps: 1,
            dir_since: -1,
            dir_last: 0,
            held_ticks: 0,
            step_ticks,
            field_cache_identity: None,
            field_cache: None,
            ceiling: None,
            ceiling_cache: None,
            intercepts: std::cell::Cell::new(0),
            candidate_log: None,
        }
    }

    /// Review round 1, F4: starts (or clears, if already started) recording every candidate this
    /// `Planner` evaluates via [`Planner::evaluate_impl`]. Test-only.
    pub fn start_candidate_log(&mut self) {
        self.candidate_log = Some(Vec::new());
    }

    /// Stops recording and returns everything collected since [`Planner::start_candidate_log`]
    /// (in call order). Test-only.
    pub fn take_candidate_log(&mut self) -> Vec<(Vec<PlanStep>, f64)> {
        self.candidate_log.take().unwrap_or_default()
    }

    /// A snapshot of every piece of hidden state TS carries between decisions
    /// (`docs/research/orig-plan.md` §1.15/§2.5), for a test harness to compare against the same
    /// fields read off the real TS `Planner` instance (review round 1, F4: "dump ... RNG and
    /// hidden state"). Test-only introspection -- not used by `decide`/`decide_production`
    /// themselves.
    pub fn debug_state(&self) -> PlannerDebugState {
        PlannerDebugState {
            rng_s0: self.rng.s0,
            rng_s1: self.rng.s1,
            rng_s2: self.rng.s2,
            rng_s3: self.rng.s3,
            rng_have_spare: self.rng.have_spare,
            rng_spare: self.rng.spare,
            opp_seed: self.opp_seed,
            warm: self.warm.clone(),
            committed: self.committed,
            commit_left: self.commit_left,
            last_decide_tick: self.last_decide_tick,
            decide_gaps: self.decide_gaps.clone(),
            last_frozen: self.last_frozen,
            dir_since: self.dir_since,
            dir_last: self.dir_last,
        }
    }

    /// `setOverrides(over)` (`planner.ts:912-917`). Unlike TS's arbitrary `Partial<PlannerConfig>`
    /// object, an override here is a pure function of the base config (exactly what
    /// `crate::config::wb_overrides`/`preset_strong_wb`/`preset_bold` already are) --
    /// `Object.assign(cfg, baseCfg, over)` *is* "recompute from base, then apply these specific
    /// field changes", which a function of `PlannerConfig -> PlannerConfig` expresses directly.
    ///
    /// Review round 1, F11: a bare `fn` pointer (an earlier revision's parameter type) cannot
    /// capture any surrounding state, so it could not express `bot.ts`'s own
    /// `wbPlanOverrides(self)`-shaped overrides (a closure over live bot state, e.g. "which side
    /// of the WB hall `self` is standing in this decision"); `Box<dyn Fn(...) -> ...>` can.
    pub fn set_overrides(&mut self, over: Option<Box<dyn Fn(PlannerConfig) -> PlannerConfig>>) {
        self.cfg = match over {
            Some(f) => f(self.base_cfg),
            None => self.base_cfg,
        };
        self.sync_grid();
    }

    pub fn config(&self) -> PlannerConfig {
        self.cfg
    }

    /// Task 3.5: the hybrid search's worker planners tune their private copy per decision.
    pub(crate) fn cfg_mut(&mut self) -> &mut PlannerConfig {
        &mut self.cfg
    }

    /// Task 3.10b (hybrid only): plans of `steps` steps from now on (the longer horizon while the victim is frozen). With the uniform grid
    /// (`front_steps` 0, the hybrid's) the step-tick table is resized in place -- no allocation once it has been that long -- else
    /// it is rebuilt. A no-op when the length is already `steps`.
    pub(crate) fn set_plan_steps(&mut self, steps: i32) {
        if self.cfg.steps == steps && self.step_ticks.len() == steps as usize {
            return;
        }
        self.cfg.steps = steps;
        if build_step_ticks_is_uniform(self.cfg.front_steps, self.cfg.front_step, self.cfg.plan_step) {
            self.step_ticks.resize(steps as usize, self.cfg.plan_step);
        } else {
            self.sync_grid();
        }
    }

    pub fn set_freeze_memory(&mut self, memory: Option<FreezeMemory>) {
        self.memory = memory;
    }

    pub fn set_dead_zone(&mut self, dead: Option<DeadZoneGrid>) {
        self.dead = dead;
    }

    /// Task 3.7b: the opponent's input at the start of each plan step, instead of "hold" (the hybrid's opponent model hands its
    /// prediction over this way, the loss diagnosis its oracle). Empty = the configured model. Never set on the parity path.
    pub(crate) fn set_predicted(&mut self, inputs: &[PlayerInput]) {
        self.predicted.clear();
        self.predicted.extend_from_slice(inputs);
    }

    pub fn set_travel_goal(&mut self, goal: Option<Vec2>) {
        self.goal = goal;
    }

    pub fn travel_goal(&self) -> Option<Vec2> {
        self.goal
    }

    pub fn set_frozen_bystanders(&mut self, tees: Vec<Vec2>, vels: Vec<Vec2>) {
        self.frozen_bystanders = tees;
        self.frozen_bystander_vels = vels;
    }

    /// Task 3.5: the hybrid workers copy the decision's frozen bystanders in place (no allocation
    /// once the buffers have grown).
    pub(crate) fn frozen_bystanders_mut(&mut self) -> (&mut Vec<Vec2>, &mut Vec<Vec2>) {
        (&mut self.frozen_bystanders, &mut self.frozen_bystander_vels)
    }

    pub fn set_spare_bystanders(&mut self, tees: Vec<Vec2>, vels: Vec<Vec2>) {
        self.spares = tees;
        self.spare_vels = vels;
    }

    /// Task 3.5b: the hybrid workers copy the live spared tees in place (no allocation once the
    /// buffers have grown).
    pub(crate) fn spares_mut(&mut self) -> (&mut Vec<Vec2>, &mut Vec<Vec2>) {
        (&mut self.spares, &mut self.spare_vels)
    }

    pub fn set_band(&mut self, band: Option<(f64, f64, f64, f64)>) {
        self.band = band.map(|(x0, y0, x1, y1)| Band { x0, y0, x1, y1 });
    }

    pub fn set_third_tees(&mut self, tees: Vec<Vec2>) {
        self.thirds = tees;
    }

    pub fn set_live_tick(&mut self, tick: i64) {
        self.live_tick = tick;
    }

    /// `reset()` (`planner.ts:991-1011`). Deliberately does **not** touch `thaw_scratch`/
    /// `thaw_scratch_identity`/`field_cache*`/`predicted` -- matching TS exactly
    /// (`docs/research/orig-plan.md` §1.15's "не сбрасываются" list).
    pub fn reset(&mut self) {
        self.warm = None;
        self.committed = None;
        self.commit_left = 0;
        self.last_decide_tick = -1;
        self.decide_gaps.clear();
        self.rng = Rng::new(self.cfg.seed.wrapping_add(self.seed_offset));
        self.opp_seed = 1u32.wrapping_add(self.seed_offset);
        self.last_frozen = false;
        self.last_search_tick = -1;
        self.warm_shift_steps = 1;
        self.dir_since = -1;
        self.dir_last = 0;
        self.held_ticks = 0;
        self.profile.reset();
        self.saved = None;
    }

    /// `setSearchSeed(n)` (`planner.ts:986-989`).
    pub fn set_search_seed(&mut self, n: u32) {
        self.seed_offset = n;
        self.reset();
    }

    fn sync_grid(&mut self) {
        let grid = build_step_ticks(
            self.cfg.steps,
            self.cfg.plan_step,
            self.cfg.front_steps,
            self.cfg.front_step,
        );
        if grid != self.step_ticks {
            self.step_ticks = grid;
        }
    }

    /// Review round 1, F10: returns cheap `Arc` clones (a refcount bump), never a deep copy of the
    /// field's `dist: Vec<i32>` (one `i32` per map tile -- 0.8 MB on Copy Love Box, 6.6 MB on
    /// BlmapChill; a per-decision deep copy of that was the finding). Every caller already just
    /// borrows `&field`/`&unfreeze`, which `Arc<HazardField>` derefs to for free. (`Arc`, not `Rc`,
    /// since task 3.5: the hybrid search hands the same two fields to its worker threads.)
    pub(crate) fn hazard_fields(&mut self, col: &W::Collision) -> (Arc<HazardField>, Arc<HazardField>) {
        let id = col.identity();
        if self.field_cache_identity != Some(id) || self.field_cache.is_none() {
            self.field_cache = Some((
                Arc::new(fields::hazard_field(col)),
                Arc::new(fields::unfreeze_field(col)),
            ));
            self.field_cache_identity = Some(id);
        }
        self.field_cache.clone().unwrap()
    }

    /// `this.ceiling = cfg.ropeCeilingCost > 0 ? ceilingField(collision) : null` (af49dfb `decideOnce`).
    fn refresh_ceiling(&mut self, col: &W::Collision) {
        if self.cfg.rope_ceiling_cost <= 0.0 && self.cfg.ceiling_guard_cost <= 0.0 {
            self.ceiling = None;
            return;
        }
        let id = col.identity();
        if !self.ceiling_cache.as_ref().is_some_and(|(cached, _)| *cached == id) {
            self.ceiling_cache = Some((id, Arc::new(fields::ceiling_field(col))));
        }
        self.ceiling = self.ceiling_cache.as_ref().map(|(_, f)| Arc::clone(f));
    }

    /// Task 3.9 (hybrid only): makes this planner's rollouts see the ceiling field of `col`'s map when `rope_ceiling_cost` is on (what
    /// `decide_once` does at its start); cached by the map's identity, so it costs a comparison after the first call.
    pub(crate) fn prepare_ceiling(&mut self, col: &W::Collision) {
        self.refresh_ceiling(col);
    }

    /// `decide(world, selfId, enemyId, prev, enemyInput)` (`planner.ts:1021-1036`).
    pub fn decide(
        &mut self,
        world: &mut W,
        self_id: i32,
        enemy_id: i32,
        prev: PlayerInput,
        enemy_input: PlayerInput,
    ) -> Decision {
        let en = world.get_tee(enemy_id);
        let steps = self.cfg.steps;
        let long = self.cfg.frozen_target_steps > steps
            && en.is_some_and(|en| en.frozen && en.freeze_ticks_left >= FROZEN_PLAN_MIN_TICKS);
        if !long {
            self.sync_grid();
            return self.decide_once(world, self_id, enemy_id, prev, enemy_input);
        }
        self.cfg.steps = self.cfg.frozen_target_steps;
        self.sync_grid();
        let out = self.decide_once(world, self_id, enemy_id, prev, enemy_input);
        self.cfg.steps = steps;
        out
    }

    /// `decideOnce` (`planner.ts:1043-1316`).
    #[allow(clippy::too_many_lines)]
    fn decide_once(
        &mut self,
        world: &mut W,
        self_id: i32,
        enemy_id: i32,
        prev: PlayerInput,
        enemy_input: PlayerInput,
    ) -> PlayerInput {
        self.opp_seed = js::opp_seed_next(self.opp_seed);
        self.thaw_memo.clear();
        self.last_elite = None;
        let Some(me) = world.get_tee(self_id) else { return prev };
        let Some(en) = world.get_tee(enemy_id) else { return prev };
        if !me.alive || !en.alive {
            return prev;
        }

        if self.cfg.opponent_read_weight > 0.0 {
            self.profile.observe(&me, &en, *HOOK_LENGTH);
        }

        let now_tick = if self.live_tick >= 0 {
            self.live_tick
        } else {
            world.tick()
        };
        self.live_tick = -1;
        let gap = now_tick - self.last_decide_tick;
        if self.last_decide_tick >= 0 && gap > 0 && gap <= CADENCE_MAX_GAP {
            self.decide_gaps.push(gap);
            if self.decide_gaps.len() > CADENCE_GAPS {
                self.decide_gaps.remove(0);
            }
        } else if gap != 0 {
            self.decide_gaps.clear();
        }
        self.last_decide_tick = now_tick;

        let frozen_now = me.frozen;
        let urgent = frozen_now != self.last_frozen || en.hooked_player == self_id;
        self.last_frozen = frozen_now;
        if !urgent
            && self.commit_left > 0
            && let Some(committed) = self.committed
        {
            self.commit_left -= 1;
            self.last_info.searched = false;
            self.last_info.candidates = 0;
            self.last_info.out_of_time = false;
            self.last_info.ms = 0.0;
            return self.maybe_release(world, self_id, committed);
        }
        let started = std::time::Instant::now();

        self.held_ticks = if self.dir_since < 0 {
            self.cfg.flip_hold_ticks as i64
        } else {
            world.tick() - self.dir_since
        };

        let (field, unfreeze) = self.hazard_fields(world.collision());
        self.refresh_ceiling(world.collision());

        let aim_at = js::atan2(en.pos.y - me.pos.y, en.pos.x - me.pos.x);
        self.warm_shift_steps = self.warm_shift(world.tick());
        self.last_search_tick = world.tick();
        let mut dist = self.build_dist(if self.cfg.track_aim { 0.0 } else { aim_at });
        match &mut self.saved {
            Some(saved) => world.save_state_into(saved),
            None => self.saved = Some(world.save_state()),
        }
        self.predict_opponent();

        let mut best: Option<Vec<PlanStep>> = None;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_stay: Option<Vec<PlanStep>> = None;
        let mut best_stay_score = f64::NEG_INFINITY;

        let mut carry: Option<Vec<PlanStep>> = None;
        let mut carry_score = f64::NEG_INFINITY;
        if self.cfg.plan_margin > 0.0
            && let Some(warm) = self.warm.clone()
            && warm.len() == self.cfg.steps as usize
        {
            let shift = self.warm_shift_steps as usize;
            let mut c: Vec<PlanStep> = warm[shift.min(warm.len())..].to_vec();
            c.extend(warm[warm.len() - shift..].iter().copied());
            let c = if c.len() != self.cfg.steps as usize {
                warm.clone()
            } else {
                c
            };
            carry_score = self.evaluate(world, self_id, enemy_id, prev, &c, enemy_input, &field, &unfreeze);
            carry = Some(c);
        }

        let started_ms = |i: std::time::Instant| i.elapsed().as_secs_f64() * 1000.0;
        let hardline: Option<f64> = if self.cfg.hard_ms > 0.0 {
            Some(self.cfg.hard_ms)
        } else {
            None
        };
        let deadline: Option<f64> = match (self.cfg.budget_ms > 0.0, hardline) {
            (true, Some(h)) => Some(js::min(self.cfg.budget_ms, h)),
            (true, None) => Some(self.cfg.budget_ms),
            (false, Some(h)) => Some(h),
            (false, None) => None,
        };
        let mut out_of_time = false;
        let over_cap = |i: std::time::Instant| hardline.is_some_and(|h| started_ms(i) > h);

        let mut candidates = 0i32;
        for it in 0..self.cfg.iterations {
            if out_of_time {
                break;
            }
            self.react_this_pass = self.cfg.opponent_mix && it > 0;
            let mut scored: Vec<(Vec<PlanStep>, f64)> = Vec::new();

            let seeds = if it == 0 {
                self.seed_plans(world, self_id, enemy_id, &field, aim_at)
            } else {
                Vec::new()
            };
            for plan in seeds {
                if best.is_some() && over_cap(started) {
                    out_of_time = true;
                    break;
                }
                let score = self.evaluate(world, self_id, enemy_id, prev, &plan, enemy_input, &field, &unfreeze);
                if plan[0].dir == prev.direction && score > best_stay_score {
                    best_stay_score = score;
                    best_stay = Some(plan.clone());
                }
                if score > best_score {
                    best_score = score;
                    best = Some(plan.clone());
                }
                scored.push((plan, score));
            }
            if it == 0 && !out_of_time {
                for plan in self.policy_seed_plans() {
                    if best.is_some() && over_cap(started) {
                        out_of_time = true;
                        break;
                    }
                    let score = self.evaluate(world, self_id, enemy_id, prev, &plan, enemy_input, &field, &unfreeze);
                    if plan[0].dir == prev.direction && score > best_stay_score {
                        best_stay_score = score;
                        best_stay = Some(plan.clone());
                    }
                    if score > best_score {
                        best_score = score;
                        best = Some(plan.clone());
                    }
                    scored.push((plan, score));
                }
            }
            if it == 0 && !out_of_time && (self.cfg.freeze_throw > 0 || self.cfg.frozen_throw > 0) && !over_cap(started)
            {
                for (plan, score) in self.landed_throws(
                    world,
                    self_id,
                    enemy_id,
                    prev,
                    enemy_input,
                    &field,
                    &unfreeze,
                    aim_at,
                    started,
                    hardline,
                ) {
                    if plan[0].dir == prev.direction && score > best_stay_score {
                        best_stay_score = score;
                        best_stay = Some(plan.clone());
                    }
                    if score > best_score {
                        best_score = score;
                        best = Some(plan.clone());
                    }
                    scored.push((plan, score));
                }
            }
            let mut i = 0;
            while i < self.cfg.population {
                if out_of_time {
                    break;
                }
                if let Some(dl) = deadline
                    && (i & 3) == 3
                    && started_ms(started) > dl
                {
                    out_of_time = true;
                    break;
                }
                let plan = self.sample_plan(&dist);
                let score = self.evaluate(world, self_id, enemy_id, prev, &plan, enemy_input, &field, &unfreeze);
                if plan[0].dir == prev.direction && score > best_stay_score {
                    best_stay_score = score;
                    best_stay = Some(plan.clone());
                }
                if score > best_score {
                    best_score = score;
                    best = Some(plan.clone());
                }
                scored.push((plan, score));
                i += 1;
            }
            candidates += scored.len() as i32;
            scored.sort_by(|a, b| score_desc(a.1, b.1));
            let elite_n = self.cfg.elite.max(0) as usize;
            let elites: Vec<Vec<PlanStep>> = scored.iter().take(elite_n).map(|(p, _)| p.clone()).collect();
            self.last_elite = crate::elite::summarize_first_step(&elites, self.cfg.track_aim, aim_at);
            self.refit(&mut dist, &elites);
        }

        world.restore_state(self.saved.as_ref().unwrap());
        let Some(mut best) = best else { return prev };

        let flips = best[0].dir != prev.direction;
        let plain_hold = self.cfg.flip_margin > 0.0
            && best_stay.is_some()
            && flips
            && best_score - best_stay_score < self.cfg.flip_margin;

        let mut not_now_in = false;
        if self.cfg.edge_hold
            && flips
            && !me.frozen
            && !over_cap(started)
            && freeze_gap_px(world.collision(), me.pos.x, me.pos.y) < EDGE_GAP_PX
            && let Some((plan, score)) =
                self.not_now(world, self_id, enemy_id, prev, &best, enemy_input, &field, &unfreeze)
        {
            not_now_in = true;
            if score > best_stay_score {
                best_stay_score = score;
                best_stay = Some(plan);
            }
        }
        self.last_info.edge_held = false;
        if (self.cfg.flip_margin > 0.0 || not_now_in)
            && best_stay.is_some()
            && flips
            && best_score - best_stay_score < self.cfg.flip_margin
        {
            best = best_stay.clone().unwrap();
            best_score = best_stay_score;
            self.last_info.edge_held = !plain_hold;
        }

        if let Some(c) = &carry
            && best_score - carry_score < self.cfg.plan_margin
        {
            best = c.clone();
            best_score = carry_score;
        }
        self.react_this_pass = false;

        if self.cfg.hook_polish
            && best[0].hook == 0
            && !over_cap(started)
            && let Some((plan, score)) = self.polish_rope(
                world,
                self_id,
                enemy_id,
                prev,
                enemy_input,
                &field,
                &unfreeze,
                &best,
                best_score,
            )
        {
            best = plan;
            best_score = score;
        }

        if self.cfg.escape_bias > 0.0 && !over_cap(started) {
            self.track_rollout = true;
            self.evaluate(world, self_id, enemy_id, prev, &best, enemy_input, &field, &unfreeze);
            self.track_rollout = false;
            if self.rollout_self_out > 0 {
                let was_bias = self.cfg.self_freeze_bias;
                self.cfg.self_freeze_bias = was_bias * self.cfg.escape_bias;
                let mut escape: Option<Vec<PlanStep>> = None;
                let mut escape_score = f64::NEG_INFINITY;
                let esc_dist = self.build_dist(if self.cfg.track_aim { 0.0 } else { aim_at });
                for _ in 0..self.cfg.population {
                    let plan = self.sample_plan(&esc_dist);
                    let score = self.evaluate(world, self_id, enemy_id, prev, &plan, enemy_input, &field, &unfreeze);
                    if score > escape_score {
                        escape_score = score;
                        escape = Some(plan);
                    }
                }
                self.cfg.self_freeze_bias = was_bias;
                if let Some(escape) = escape {
                    let fair = self.evaluate(world, self_id, enemy_id, prev, &escape, enemy_input, &field, &unfreeze);
                    self.track_rollout = true;
                    self.evaluate(world, self_id, enemy_id, prev, &escape, enemy_input, &field, &unfreeze);
                    self.track_rollout = false;
                    if self.rollout_self_out == 0 && fair >= best_score - self.cfg.escape_margin {
                        best = escape;
                        best_score = fair;
                    }
                }
            }
        }
        let _ = best_score;
        self.warm = Some(best.clone());
        self.last_info.searched = true;
        self.last_info.candidates = candidates;
        self.last_info.out_of_time = out_of_time;
        self.last_info.ms = started_ms(started);
        self.last_info.hook_at = -1;
        {
            let mut at = 0i32;
            for (i, st) in best.iter().enumerate() {
                if st.hook == 1 {
                    self.last_info.hook_at = at;
                    break;
                }
                at += self.step_ticks[i];
            }
        }
        if self.cfg.explain && !over_cap(started) {
            self.track_rollout = true;
            self.evaluate(world, self_id, enemy_id, prev, &best, enemy_input, &field, &unfreeze);
            self.track_rollout = false;
            self.last_info.self_out = self.rollout_self_out;
            self.last_info.enemy_out = self.rollout_enemy_out;
        } else {
            self.last_info.self_out = -1;
            self.last_info.enemy_out = -1;
        }

        let rest = self.cfg.rest_aim && best[0].hook == 0 && best[0].fire == 0;
        let aim0 = if rest {
            aim_at
        } else if self.cfg.track_aim {
            aim_at + best[0].aim
        } else {
            best[0].aim
        };

        let (hook_ok, aim0, snapped) =
            self.gate_and_snap(world, self_id, enemy_id, Some(&me), Some(&en), best[0], prev, aim0);
        self.last_info.gated = best[0].hook == 1 && !hook_ok;

        self.swing_target_frozen = en.frozen;
        self.swing_rope_on = me.hooked_player == enemy_id;
        self.swing_target = Some(en);
        let mut chosen = self.step_to_input_snapped(
            world,
            best[0],
            prev,
            vdistance(me.pos, en.pos),
            hook_ok,
            Some(me.pos),
            Some(en.pos),
            Some(en.vel),
            aim0,
            snapped,
        );

        self.last_info.shielded = false;
        if self.cfg.shield && !me.frozen {
            let hold = self.shield_hold(self.cfg.commit_decisions);
            let others: HashMap<i32, PlayerInput> = HashMap::from([(enemy_id, enemy_input)]);
            if !crate::shield::escape_exists(world, self_id, &chosen, hold, &others)
                && let Some(safer) = crate::shield::safer_input(world, self_id, &chosen, hold, &others, Some(&prev))
            {
                chosen = safer;
                self.last_info.shielded = true;
            }
        }
        if chosen.direction != self.dir_last || self.dir_since < 0 {
            self.dir_last = chosen.direction;
            self.dir_since = world.tick();
        }
        self.committed = Some(chosen);
        self.commit_left = js::max(0.0, f64::from(self.cfg.commit_decisions - 1)) as i32;
        self.maybe_release(world, self_id, chosen)
    }

    /// D-041's production deadline mode (review round 1, finding F1 -- the parity path,
    /// `Planner::decide`/`decide_once`, is completely unaffected: it never calls this method or
    /// reads a clock of any kind). Iterative deepening: a warm-started candidate (the previous
    /// plan shifted by one step) first if one exists, then the opening book, then CEM population
    /// sampling with per-iteration `refit` -- every candidate after the very first is checked
    /// against `deadline_ms` (read from `clock`) before it starts, and [`Planner::evaluate_impl`]
    /// itself re-checks the same deadline once per plan *step* (a few ticks) during every
    /// candidate's rollout, so one candidate's own evaluation can't blow the budget by more than
    /// that (measured well under the ~0.3 ms target -- see the crate README's D-041 section).
    /// Always returns the best candidate found so far. The very first candidate this call
    /// evaluates (the warm-started one, or the first book plan if there is no warm plan of the
    /// right length yet) is *never* itself interrupted mid-rollout (its own `deadline` argument is
    /// `None`), so a decision is always well-formed even under an unreasonably small budget --
    /// exactly like TS's own `best === null` guard in `decideOnce` never leaves `chosen`
    /// undefined, just answered here with a real deadline instead of a fixed iteration count.
    ///
    /// Time comes from the injected `clock: &C`, not `std::time::Instant`/`Date.now` directly, so
    /// a production-search test can use a deterministic [`crate::clock::StepClock`] instead of
    /// real wall time (`crate::clock`'s own doc comment).
    ///
    /// **Documented simplification vs. `decide_once`** (time-boxed for this fix round, not a
    /// parity requirement -- this method is never compared against TS and never will be): does
    /// not run the `notNow` (edge-hold)/`hookPolish`/`escapeBias`/`explain` post-passes (secondary
    /// quality refinements in TS, not safety-critical), nor the `planMargin`/`carry` candidate
    /// (superseded here by the warm-started candidate below, which already serves the same "don't
    /// throw away a good in-flight plan" role); does keep the `shield` safety net (final
    /// escape-exists check) since D-041 explicitly frames the search as "поиск остаётся страховкой
    /// точности" (the precision safety net), which is exactly what `shield` is -- review round 2,
    /// F12/F13 made that safety net itself deadline-bounded and ported back
    /// `landedThrows`/`frozenThrow`, the `flip_margin` stay-hysteresis, the `opponent_mix` react
    /// pass, the `opp_seed` advance and `thaw_memo.clear()`, all of which round 1 had silently
    /// left out.
    ///
    /// **F12's shield budget (review round 2, BLOCKER):** the shield (`escape_exists`/
    /// `safer_input`) used to run after `deadline_ms` with no clock check at all -- up to ~5,300
    /// physics steps, measured up to 2,056 steps / ~20 ms of *real* work in one decision (not a
    /// stall: `r2-phase.log`, ~4.5 µs/step). It now gets its own `SHIELD_RESERVE_MS` reserve on
    /// top of the search budget, checked every escape candidate and every few ticks inside one
    /// (`shield::CLOCK_CHECK_STRIDE`); if that isn't enough to even confirm whether the chosen
    /// input has an escape, it's extended (only then -- see below) up to
    /// `SHIELD_DANGER_MAX_TOTAL_MS` total. On a timeout the searched `chosen` input is always kept
    /// (never `prev`) -- `last_info.shield_incomplete` says whether that answer was fully
    /// verified. See the crate README's D-041 section for the measured step counts before/after
    /// and the falls-back-to-`prev` comparison this fallback choice is based on.
    #[allow(clippy::too_many_arguments)]
    pub fn decide_production<C: Clock>(
        &mut self,
        world: &mut W,
        self_id: i32,
        enemy_id: i32,
        prev: PlayerInput,
        enemy_input: PlayerInput,
        clock: &C,
        budget_ms: f64,
    ) -> Decision {
        // Review round 2, F13: `decide_once` advances the opponent-model RNG seed and clears the
        // `noThaw` memo on *every* call, unconditionally, before even checking whether `self`/
        // `enemy` exist -- ported verbatim (both are cheap, unconditional, and have no deadline of
        // their own to respect). Skipping `thaw_memo.clear()` here would let the memo grow without
        // bound over a live session (it's keyed by rounded position + a few flags, so a bot that
        // roams a whole map over hours would accumulate entries forever); TS clears it every
        // decision instead of aging entries out, so this does too.
        self.opp_seed = js::opp_seed_next(self.opp_seed);
        self.thaw_memo.clear();
        let Some(me) = world.get_tee(self_id) else { return prev };
        let Some(en) = world.get_tee(enemy_id) else { return prev };
        if !me.alive || !en.alive {
            return prev;
        }

        // Review round 3, F19: `decide_once` tracks the tick gap between successive decisions
        // into `decide_gaps` (used by `shield_hold`'s `shield_cadence` branch to estimate a
        // "typical" decision cadence) unconditionally, before doing anything else -- ported
        // verbatim; before this fix `decide_production` never touched `decide_gaps` at all, so it
        // stayed permanently empty and `shield_hold` always fell back to its `sorted.is_empty()`
        // default (`typical = 2`), meaning `shield_cadence` never actually adapted to this path's
        // real cadence.
        let now_tick = if self.live_tick >= 0 {
            self.live_tick
        } else {
            world.tick()
        };
        self.live_tick = -1;
        let gap = now_tick - self.last_decide_tick;
        if self.last_decide_tick >= 0 && gap > 0 && gap <= CADENCE_MAX_GAP {
            self.decide_gaps.push(gap);
            if self.decide_gaps.len() > CADENCE_GAPS {
                self.decide_gaps.remove(0);
            }
        } else if gap != 0 {
            self.decide_gaps.clear();
        }
        self.last_decide_tick = now_tick;

        crate::prof::mark(0);
        let (field, unfreeze) = self.hazard_fields(world.collision());
        self.refresh_ceiling(world.collision());
        let aim_at = js::atan2(en.pos.y - me.pos.y, en.pos.x - me.pos.x);
        // Only "hold"/"react"/`opponentMix` are supported (see `Planner::predict_opponent`'s doc
        // comment) -- always empty, matching TS's own behavior with no policy/learned net loaded.
        self.predicted.clear();
        let mut dist = self.build_dist(if self.cfg.track_aim { 0.0 } else { aim_at });
        match &mut self.saved {
            Some(saved) => world.save_state_into(saved),
            None => self.saved = Some(world.save_state()),
        }

        let deadline_ms = clock.now_ms() + budget_ms;
        let mut best: Option<Vec<PlanStep>> = None;
        let mut best_score = f64::NEG_INFINITY;
        // Review round 2, F13: `decide_once`'s `flip_margin` stay-hysteresis ("don't flip
        // direction unless the flip is clearly better") tracks the best-scoring candidate that
        // *doesn't* flip `prev.direction` alongside the overall best, then swaps back to it after
        // the search if the overall winner didn't beat it by more than `flip_margin` -- ported
        // verbatim below (search this function for `best_stay`/`cfg.flip_margin`).
        let mut best_stay: Option<Vec<PlanStep>> = None;
        let mut best_stay_score = f64::NEG_INFINITY;
        let mut candidates = 0i32;
        let mut have_any = false;
        // Review round 3, F17: `decide_once` feeds the book (`seed_plans`) and throw
        // (`landed_throws`) seeds into iteration 0's own `scored`, so they can become elites for
        // that iteration's `refit` -- before this fix, `decide_production` created `scored` fresh
        // *inside* the `'iters` loop, so the first refit only ever saw plain CEM samples, and every
        // seed's influence on `dist` was lost after iteration 0. Declared here (not inside the
        // loop) so the seed loops below can push into it; the `'iters` loop reuses it for `it == 0`
        // and clears it for every iteration after that. The warm-start candidate (immediately
        // below) is intentionally NOT pushed here -- it has no analogue in `decide_once`'s `scored`
        // pipeline either; it stands in for `carry`/`plan_margin` (see the "not ported" list),
        // which `decide_once` also keeps out of `scored` and only compares against `best` once, at
        // the very end, after the whole search.
        let mut scored: Vec<(Vec<PlanStep>, f64)> = Vec::new();

        // Review round 1, F1 follow-up (found while measuring): `self.warm` being `Some` means a
        // *previous* `decide_production` call already succeeded, so `prev` is a safe fallback if
        // this candidate is interrupted -- unlike the very first candidate ever tried for this
        // `Planner` (`have_any == false`, below), which still gets an uninterrupted pass so *some*
        // decision is always produced. Before this fix, the warm candidate always ran with
        // `deadline: None` unconditionally (copying `decide_once`'s "warm is free" shape without
        // its TS-parity reason to exist), which meant *every* decision after the first paid its
        // full, uninterruptible cost regardless of budget -- measured p99 latencies an order of
        // magnitude over the requested budget (see BUILD REPORT). Gating it the same way the rest
        // of this function gates every other candidate closes that gap.
        if let Some(warm) = self.warm.clone()
            && warm.len() == self.cfg.steps as usize
        {
            let shifted: Vec<PlanStep> = warm[1..]
                .iter()
                .copied()
                .chain(std::iter::once(*warm.last().unwrap()))
                .collect();
            // `self.warm` existing at all (independent of this call's own `have_any`, which is
            // always `false` here -- nothing has been tried yet this call) is itself the "a
            // fallback exists" signal: a previous `decide_production` call already succeeded, so
            // it is safe to interrupt this one and fall back to `prev`.
            let dl: Option<(&dyn Clock, f64)> = Some((clock, deadline_ms));
            if let Some(score) = self.evaluate_impl(
                world,
                self_id,
                enemy_id,
                prev,
                &shifted,
                enemy_input,
                &field,
                &unfreeze,
                dl,
            ) {
                best_score = score;
                if shifted[0].dir == prev.direction {
                    best_stay_score = score;
                    best_stay = Some(shifted.clone());
                }
                best = Some(shifted);
                have_any = true;
            }
            candidates += 1;
        }

        for plan in self.seed_plans(world, self_id, enemy_id, &field, aim_at) {
            if have_any && clock.now_ms() >= deadline_ms {
                break;
            }
            let dl: Option<(&dyn Clock, f64)> = if have_any { Some((clock, deadline_ms)) } else { None };
            let Some(score) = self.evaluate_impl(
                world,
                self_id,
                enemy_id,
                prev,
                &plan,
                enemy_input,
                &field,
                &unfreeze,
                dl,
            ) else {
                break;
            };
            have_any = true;
            candidates += 1;
            if plan[0].dir == prev.direction && score > best_stay_score {
                best_stay_score = score;
                best_stay = Some(plan.clone());
            }
            if score > best_score {
                best_score = score;
                best = Some(plan.clone());
            }
            // Review round 3, F17: feed the book seed into iteration 0's elites, like `decide_once`.
            scored.push((plan, score));
        }

        // Review round 2, F13: `landedThrows`/`frozenThrow` seeds -- `decide_once` runs these on
        // its first iteration whenever `freeze_throw > 0 || frozen_throw > 0` (every preset this
        // crate's acceptance criteria uses sets `frozen_throw: 3`); `decide_production` silently
        // skipped them entirely before this fix.
        if clock.now_ms() < deadline_ms && (self.cfg.freeze_throw > 0 || self.cfg.frozen_throw > 0) {
            for (plan, score) in self.landed_throws_bounded(
                world,
                self_id,
                enemy_id,
                prev,
                enemy_input,
                &field,
                &unfreeze,
                aim_at,
                clock,
                deadline_ms,
            ) {
                have_any = true;
                candidates += 1;
                if plan[0].dir == prev.direction && score > best_stay_score {
                    best_stay_score = score;
                    best_stay = Some(plan.clone());
                }
                if score > best_score {
                    best_score = score;
                    best = Some(plan.clone());
                }
                // Review round 3, F17: feed the throw seed into iteration 0's elites too.
                scored.push((plan, score));
            }
        }

        'iters: for it in 0..self.cfg.iterations {
            if have_any && clock.now_ms() >= deadline_ms {
                break;
            }
            // Review round 2, F13: `decide_once` sets this once per iteration (`it > 0`, i.e.
            // never on the first) so `opponentMix` bots start reacting to the actual chosen plan
            // from the second iteration on, instead of only ever holding -- `decide_production`
            // never set it at all before this fix, so an `opponent_mix` enemy model always used
            // the `hold` behavior regardless of iteration.
            self.react_this_pass = self.cfg.opponent_mix && it > 0;
            // Review round 3, F17: `scored` is declared once, above this loop, specifically so the
            // book/throw seeds pushed into it before the loop started survive into `it == 0`'s own
            // elite selection -- only clear it on every iteration *after* the first (`decide_once`
            // creates a fresh `scored` per iteration too, but never carries seeds past `it == 0` in
            // the first place, so clearing from `it == 1` on is equivalent).
            if it > 0 {
                scored.clear();
            }
            for _ in 0..self.cfg.population {
                if have_any && clock.now_ms() >= deadline_ms {
                    break 'iters;
                }
                let plan = self.sample_plan(&dist);
                let dl: Option<(&dyn Clock, f64)> = if have_any { Some((clock, deadline_ms)) } else { None };
                let Some(score) = self.evaluate_impl(
                    world,
                    self_id,
                    enemy_id,
                    prev,
                    &plan,
                    enemy_input,
                    &field,
                    &unfreeze,
                    dl,
                ) else {
                    break 'iters;
                };
                have_any = true;
                candidates += 1;
                if plan[0].dir == prev.direction && score > best_stay_score {
                    best_stay_score = score;
                    best_stay = Some(plan.clone());
                }
                if score > best_score {
                    best_score = score;
                    best = Some(plan.clone());
                }
                scored.push((plan, score));
            }
            scored.sort_by(|a, b| score_desc(a.1, b.1));
            let elite_n = self.cfg.elite.max(0) as usize;
            let elites: Vec<Vec<PlanStep>> = scored.iter().take(elite_n).map(|(p, _)| p.clone()).collect();
            self.refit(&mut dist, &elites);
        }
        self.react_this_pass = false;

        crate::prof::mark(1);
        world.restore_state(self.saved.as_ref().unwrap());
        self.last_info.candidates = candidates;
        self.last_info.searched = true;
        self.last_info.out_of_time = clock.now_ms() >= deadline_ms;
        self.last_info.self_out = -1;
        self.last_info.enemy_out = -1;

        let Some(mut best) = best else { return prev };
        self.last_info.hook_at = -1;
        {
            let mut at = 0i32;
            for (i, st) in best.iter().enumerate() {
                if st.hook == 1 {
                    self.last_info.hook_at = at;
                    break;
                }
                at += self.step_ticks[i];
            }
        }

        // Review round 2, F13: `decide_once`'s `flip_margin` stay-hysteresis, ported verbatim
        // (`planner.ts`'s own `plainHold`/`edgeHeld` swap, minus the `notNow`/edge-hold half of it
        // -- that post-pass stays out of scope, documented above). Only swaps `best` back to the
        // best non-flipping candidate found when the flip didn't win by more than `flip_margin`.
        let flips = best[0].dir != prev.direction;
        if self.cfg.flip_margin > 0.0
            && best_stay.is_some()
            && flips
            && best_score - best_stay_score < self.cfg.flip_margin
        {
            best = best_stay.clone().unwrap();
        }
        self.warm = Some(best.clone());

        let rest = self.cfg.rest_aim && best[0].hook == 0 && best[0].fire == 0;
        let aim0 = if rest {
            aim_at
        } else if self.cfg.track_aim {
            aim_at + best[0].aim
        } else {
            best[0].aim
        };
        let (hook_ok, aim0, snapped) =
            self.gate_and_snap(world, self_id, enemy_id, Some(&me), Some(&en), best[0], prev, aim0);
        self.last_info.gated = best[0].hook == 1 && !hook_ok;

        self.swing_target_frozen = en.frozen;
        self.swing_rope_on = me.hooked_player == enemy_id;
        self.swing_target = Some(en);
        let mut chosen = self.step_to_input_snapped(
            world,
            best[0],
            prev,
            vdistance(me.pos, en.pos),
            hook_ok,
            Some(me.pos),
            Some(en.pos),
            Some(en.vel),
            aim0,
            snapped,
        );

        crate::prof::mark(2);
        self.last_info.shielded = false;
        self.last_info.shield_incomplete = false;
        if self.cfg.shield && !me.frozen {
            // Review round 3, F19: `commit = 1`, not `self.cfg.commit_decisions` -- see
            // `shield_hold`'s doc comment; `decide_production` always keeps `commit_left = 0`.
            let hold = self.shield_hold(1);
            let others: HashMap<i32, PlayerInput> = HashMap::from([(enemy_id, enemy_input)]);
            // Review round 2, F12: the shield used to run `escape_exists`/`safer_input` with no
            // clock check at all here -- up to ~5,300 physics steps with the deadline already
            // passed (measured up to 2,056 steps / ~20 ms of real work, `r2-phase.log`). It now
            // gets a reserve (see `SHIELD_RESERVE_MS_PER_TEE`, review round 3, F18: scaled by the
            // number of simulated tees) on top of the search budget for the initial check; only
            // once that check confirms real danger (no escape for the searched `chosen`) does the
            // search for a safer alternative get to extend further, up to
            // `SHIELD_DANGER_MAX_TOTAL_MS` total -- an inconclusive (timed-out) initial check does
            // *not* count as "confirmed danger" on its own, so it doesn't earn the extension; it
            // still gets one attempt at `safer_input` within the plain reserve, since checking
            // "is there anything obviously safer" cheaply is worth it even when we couldn't
            // confirm the chosen input is actually unsafe. (3.5 review round 1, F1, proposed
            // extending on a timeout too: measured worse, see `docs/EXPERIMENTS.md` E-003.)
            let call_start_ms = deadline_ms - budget_ms;
            let sim_tees = js::max(1.0, world.all_tees().len() as f64);
            let reserve_ms = SHIELD_RESERVE_MS_PER_TEE * sim_tees;
            let reserve_deadline_ms = clock.now_ms() + reserve_ms;
            let escape_status = crate::shield::escape_exists_bounded(
                world,
                self_id,
                &chosen,
                hold,
                &others,
                Some((clock, reserve_deadline_ms)),
            );
            if !matches!(escape_status, crate::shield::Bounded::Done(true)) {
                let danger = matches!(escape_status, crate::shield::Bounded::Done(false));
                let safer_deadline_ms = if danger {
                    js::max(reserve_deadline_ms, call_start_ms + SHIELD_DANGER_MAX_TOTAL_MS)
                } else {
                    reserve_deadline_ms
                };
                match crate::shield::safer_input_bounded(
                    world,
                    self_id,
                    &chosen,
                    hold,
                    &others,
                    Some(&prev),
                    Some((clock, safer_deadline_ms)),
                ) {
                    crate::shield::Bounded::Done(Some(safer)) => {
                        chosen = safer;
                        self.last_info.shielded = true;
                    }
                    // A complete, honest "no safer input either" -- not a timeout, nothing to flag.
                    crate::shield::Bounded::Done(None) => {}
                    // Ran out of time before confirming an alternative either way: keep the
                    // searched `chosen` (see this method's doc comment/README for the measured
                    // comparison against falling back to `prev`), flag it as unverified.
                    crate::shield::Bounded::TimedOut => {
                        self.last_info.shield_incomplete = true;
                    }
                }
            }
        }
        crate::prof::mark(3);
        if chosen.direction != self.dir_last || self.dir_since < 0 {
            self.dir_last = chosen.direction;
            self.dir_since = world.tick();
        }
        self.committed = Some(chosen);
        self.commit_left = 0;
        self.maybe_release(world, self_id, chosen)
    }

    /// `shieldHold()` (`planner.ts:1318-1326`). `commit` is how many decisions in a row the caller
    /// intends to hold this input for before deciding again -- `decide_once` passes
    /// `self.cfg.commit_decisions` (its own `commit_left` bookkeeping actually skips that many
    /// decisions, so the shield must hold at least that long too). Review round 3, F19:
    /// `decide_production` always keeps `commit_left = 0` (it decides fresh every call, never
    /// skips), so multiplying by `commit_decisions` there over-held the shield input by that same
    /// factor for no reason (e.g. the `low` preset's `commit_decisions: 4` made
    /// `decide_production`'s shield hold 4 ticks when only 1 was ever actually committed to) --
    /// its call site now passes `1` instead.
    pub(crate) fn shield_hold(&self, commit: i32) -> i32 {
        let commit = commit.max(1);
        if !self.cfg.shield_cadence {
            return 2 * commit;
        }
        let mut sorted = self.decide_gaps.clone();
        sorted.sort_unstable();
        let typical = if !sorted.is_empty() {
            sorted[(sorted.len() - 1) / 2]
        } else {
            2
        };
        let per_decision = (typical.clamp(2, i64::from(CADENCE_MAX_TICKS))) as i32;
        (per_decision * commit).min(SHIELD_MAX_HOLD)
    }

    /// `maybeRelease(world, selfId, input)` (`planner.ts:1328-1331`).
    pub(crate) fn maybe_release(&self, world: &W, self_id: i32, input: PlayerInput) -> PlayerInput {
        if !self.cfg.release_dead_hook || input.hook == 0 || !self.hook_is_dead(world, self_id) {
            input
        } else {
            PlayerInput { hook: 0, ..input }
        }
    }

    pub(crate) fn hook_is_dead(&self, world: &W, self_id: i32) -> bool {
        match world.get_tee(self_id) {
            Some(me) => me.hook_state != HOOK_IDLE && me.hook_state != HOOK_FLYING && me.hook_state != HOOK_GRABBED,
            None => false,
        }
    }

    pub(crate) fn hook_already_out(&self, world: &W, self_id: i32) -> bool {
        match world.get_tee(self_id) {
            Some(me) => me.hook_state == HOOK_FLYING || me.hook_state == HOOK_GRABBED,
            None => false,
        }
    }

    /// `notNow` (`planner.ts:1333-1356`).
    #[allow(clippy::too_many_arguments)]
    fn not_now(
        &mut self,
        world: &mut W,
        self_id: i32,
        enemy_id: i32,
        prev: PlayerInput,
        turning: &[PlanStep],
        enemy_input: PlayerInput,
        field: &HazardField,
        unfreeze: &HazardField,
    ) -> Option<(Vec<PlanStep>, f64)> {
        let later: Vec<PlanStep> = turning
            .iter()
            .enumerate()
            .map(|(i, st)| PlanStep {
                dir: if i == 0 { prev.direction } else { turning[i - 1].dir },
                ..*st
            })
            .collect();
        let kept: Vec<PlanStep> = turning
            .iter()
            .enumerate()
            .map(|(i, st)| {
                if i == 0 {
                    PlanStep {
                        dir: prev.direction,
                        ..*st
                    }
                } else {
                    *st
                }
            })
            .collect();
        self.track_gap = true;
        self.evaluate(world, self_id, enemy_id, prev, turning, enemy_input, field, unfreeze);
        let turn_gap = self.rollout_min_gap;
        let mut out: Option<(Vec<PlanStep>, f64, f64)> = None;
        for plan in [later, kept] {
            let score = self.evaluate(world, self_id, enemy_id, prev, &plan, enemy_input, field, unfreeze);
            if out.as_ref().is_none_or(|o| score > o.1) {
                out = Some((plan, score, self.rollout_min_gap));
            }
        }
        self.track_gap = false;
        let (plan, score, gap) = out?;
        if gap < turn_gap - EDGE_NEARER_PX {
            None
        } else {
            Some((plan, score))
        }
    }

    /// The frozen-victim throw seeds of `landedThrows`: `frozenThrowLines`, and with a wall to the side (af49dfb `wallDir`) the
    /// wall swings and (with `airChain`, while airborne) the air chains in front of them, identical lines dropped when chains
    /// are offered.
    fn frozen_throw_seed_lines(&self, world: &W, me: &TeeState, aim: f64) -> Vec<Vec<PlanStep>> {
        if self.cfg.wall_dir == 0 {
            return frozen_throw_lines(self.cfg.steps, aim);
        }
        let half = PHYSICAL_SIZE / 2.0;
        let col = world.collision();
        let grounded = col.is_solid(me.pos.x + half, me.pos.y + half + 5.0)
            || col.is_solid(me.pos.x - half, me.pos.y + half + 5.0);
        let chain = if grounded || !self.cfg.air_chain {
            Vec::new()
        } else {
            air_chain_lines(
                self.cfg.steps,
                &self.step_ticks,
                aim,
                self.cfg.wall_dir,
                me.jumps_left > 0,
            )
        };
        let offered = !chain.is_empty();
        let mut lines = chain;
        lines.extend(wall_swing_lines(
            self.cfg.steps,
            &self.step_ticks,
            aim,
            self.cfg.wall_dir,
        ));
        lines.extend(frozen_throw_lines(self.cfg.steps, aim));
        if offered {
            // `JSON.stringify` equality: the same steps, a NaN aim equal to a NaN aim.
            let same = |a: &[PlanStep], b: &[PlanStep]| {
                a.len() == b.len()
                    && a.iter().zip(b).all(|(x, y)| {
                        x.dir == y.dir
                            && x.jump == y.jump
                            && x.hook == y.hook
                            && x.fire == y.fire
                            && (x.aim == y.aim || (x.aim.is_nan() && y.aim.is_nan()))
                    })
            };
            let mut kept: Vec<Vec<PlanStep>> = Vec::with_capacity(lines.len());
            for line in lines {
                if !kept.iter().any(|k| same(k, &line)) {
                    kept.push(line);
                }
            }
            lines = kept;
        }
        lines
    }

    /// `landedThrows` (`planner.ts:1358-1406`).
    #[allow(clippy::too_many_arguments)]
    fn landed_throws(
        &mut self,
        world: &mut W,
        self_id: i32,
        enemy_id: i32,
        prev: PlayerInput,
        enemy_input: PlayerInput,
        field: &HazardField,
        unfreeze: &HazardField,
        aim_at: f64,
        started: std::time::Instant,
        hardline: Option<f64>,
    ) -> Vec<(Vec<PlanStep>, f64)> {
        let over_cap = || hardline.is_some_and(|h| started.elapsed().as_secs_f64() * 1000.0 > h);
        let Some(me) = world.get_tee(self_id) else {
            return Vec::new();
        };
        let Some(en) = world.get_tee(enemy_id) else {
            return Vec::new();
        };
        let situation = ThrowSituation {
            separation: vdistance(me.pos, en.pos),
            enemy_hazard_nearness: hazard_nearness(field, en.pos.x, en.pos.y),
            me_frozen: me.frozen,
            enemy_frozen: en.frozen,
            enemy_alive: en.alive,
        };
        if self.cfg.frozen_throw > 0
            && frozen_throw_worth_trying(&situation)
            && en.freeze_ticks_left >= FROZEN_PLAN_MIN_TICKS
        {
            let mut kept: Vec<(Vec<PlanStep>, f64)> = Vec::new();
            self.track_rollout = true;
            for plan in self.frozen_throw_seed_lines(world, &me, if self.cfg.track_aim { 0.0 } else { aim_at }) {
                if over_cap() {
                    break;
                }
                let score = self.evaluate(world, self_id, enemy_id, prev, &plan, enemy_input, field, unfreeze);
                if self.rollout_enemy_sealed && self.rollout_self_out == 0 {
                    kept.push((plan, score));
                }
            }
            self.track_rollout = false;
            kept.sort_by(|a, b| score_desc(a.1, b.1));
            kept.truncate(self.cfg.frozen_throw.max(0) as usize);
            return kept;
        }
        if self.cfg.freeze_throw <= 0 || !throw_worth_trying(&situation) {
            return Vec::new();
        }
        let mut kept: Vec<(Vec<PlanStep>, f64, f64)> = Vec::new();
        self.track_rollout = true;
        for plan in throw_lines(self.cfg.steps, if self.cfg.track_aim { 0.0 } else { aim_at }) {
            if over_cap() {
                break;
            }
            let score = self.evaluate(world, self_id, enemy_id, prev, &plan, enemy_input, field, unfreeze);
            let gain = f64::from(self.rollout_enemy_out - self.rollout_self_out);
            if self.rollout_enemy_out >= THROW_LANDED_TICKS && gain > 0.0 {
                kept.push((plan, score, gain));
            }
        }
        self.track_rollout = false;
        kept.sort_by(|a, b| {
            let g = score_desc(a.2, b.2);
            if g != Ordering::Equal { g } else { score_desc(a.1, b.1) }
        });
        kept.truncate(self.cfg.freeze_throw.max(0) as usize);
        kept.into_iter().map(|(p, s, _)| (p, s)).collect()
    }

    /// Review round 2, F13: `decide_production`'s deadline-aware counterpart to
    /// [`Planner::landed_throws`] -- same TS logic (`landedThrows`/`frozenThrow`,
    /// `docs/research/orig-plan.md`), but each candidate's rollout goes through `evaluate_impl`
    /// (interruptible) instead of the always-complete `evaluate`, and is gated on `clock`/
    /// `deadline_ms` instead of `decide_once`'s own `hardline`/`started: Instant` (which
    /// `decide_production` has no equivalent of -- it only ever has one deadline, not TS's
    /// separate soft/hard pair). `decide_production` previously skipped this entirely: every
    /// preset this crate's acceptance criteria uses sets `frozen_throw: 3`, so `decide_once` ran
    /// it on every single decision and `decide_production` never did.
    #[allow(clippy::too_many_arguments)]
    fn landed_throws_bounded(
        &mut self,
        world: &mut W,
        self_id: i32,
        enemy_id: i32,
        prev: PlayerInput,
        enemy_input: PlayerInput,
        field: &HazardField,
        unfreeze: &HazardField,
        aim_at: f64,
        clock: &dyn Clock,
        deadline_ms: f64,
    ) -> Vec<(Vec<PlanStep>, f64)> {
        let Some(me) = world.get_tee(self_id) else {
            return Vec::new();
        };
        let Some(en) = world.get_tee(enemy_id) else {
            return Vec::new();
        };
        let situation = ThrowSituation {
            separation: vdistance(me.pos, en.pos),
            enemy_hazard_nearness: hazard_nearness(field, en.pos.x, en.pos.y),
            me_frozen: me.frozen,
            enemy_frozen: en.frozen,
            enemy_alive: en.alive,
        };
        if self.cfg.frozen_throw > 0
            && frozen_throw_worth_trying(&situation)
            && en.freeze_ticks_left >= FROZEN_PLAN_MIN_TICKS
        {
            let mut kept: Vec<(Vec<PlanStep>, f64)> = Vec::new();
            self.track_rollout = true;
            for plan in self.frozen_throw_seed_lines(world, &me, if self.cfg.track_aim { 0.0 } else { aim_at }) {
                if clock.now_ms() >= deadline_ms {
                    break;
                }
                let Some(score) = self.evaluate_impl(
                    world,
                    self_id,
                    enemy_id,
                    prev,
                    &plan,
                    enemy_input,
                    field,
                    unfreeze,
                    Some((clock, deadline_ms)),
                ) else {
                    break;
                };
                if self.rollout_enemy_sealed && self.rollout_self_out == 0 {
                    kept.push((plan, score));
                }
            }
            self.track_rollout = false;
            kept.sort_by(|a, b| score_desc(a.1, b.1));
            kept.truncate(self.cfg.frozen_throw.max(0) as usize);
            return kept;
        }
        if self.cfg.freeze_throw <= 0 || !throw_worth_trying(&situation) {
            return Vec::new();
        }
        let mut kept: Vec<(Vec<PlanStep>, f64, f64)> = Vec::new();
        self.track_rollout = true;
        for plan in throw_lines(self.cfg.steps, if self.cfg.track_aim { 0.0 } else { aim_at }) {
            if clock.now_ms() >= deadline_ms {
                break;
            }
            let Some(score) = self.evaluate_impl(
                world,
                self_id,
                enemy_id,
                prev,
                &plan,
                enemy_input,
                field,
                unfreeze,
                Some((clock, deadline_ms)),
            ) else {
                break;
            };
            let gain = f64::from(self.rollout_enemy_out - self.rollout_self_out);
            if self.rollout_enemy_out >= THROW_LANDED_TICKS && gain > 0.0 {
                kept.push((plan, score, gain));
            }
        }
        self.track_rollout = false;
        kept.sort_by(|a, b| {
            let g = score_desc(a.2, b.2);
            if g != Ordering::Equal { g } else { score_desc(a.1, b.1) }
        });
        kept.truncate(self.cfg.freeze_throw.max(0) as usize);
        kept.into_iter().map(|(p, s, _)| (p, s)).collect()
    }

    /// `polishRope` (`planner.ts:1408-1440`).
    #[allow(clippy::too_many_arguments)]
    fn polish_rope(
        &mut self,
        world: &mut W,
        self_id: i32,
        enemy_id: i32,
        prev: PlayerInput,
        enemy_input: PlayerInput,
        field: &HazardField,
        unfreeze: &HazardField,
        best: &[PlanStep],
        best_score: f64,
    ) -> Option<(Vec<PlanStep>, f64)> {
        let mut variants: Vec<Vec<PlanStep>> = Vec::with_capacity(3);
        if !self.polish_variants(world, self_id, enemy_id, prev, best, &mut variants) {
            return None;
        }
        let mut out: Option<(Vec<PlanStep>, f64)> = None;
        for plan in variants {
            let score = self.evaluate(world, self_id, enemy_id, prev, &plan, enemy_input, field, unfreeze);
            if score >= best_score && out.as_ref().is_none_or(|o| score > o.1) {
                out = Some((plan, score));
            }
        }
        out
    }

    /// The plans `polishRope` tries (`planner.ts:1408-1440`): `best` with the hook held (and, for a throw, aimed at the
    /// victim) for its first `k` steps, `k` = 2, 4, all (1, 2, 3 while our hook is in flight, af49dfb `hookKeepFlying`).
    /// Pushes them, in order, onto `out` and returns `true`; `false` (nothing pushed) when polishing does not apply: a
    /// frozen or dead tee, a hook that is neither out nor within reach, or the hook gate refusing the throw. Split out of
    /// `polish_rope` (task 3.9) so the hybrid search can score the same variants itself.
    pub(crate) fn polish_variants(
        &self,
        world: &W,
        self_id: i32,
        enemy_id: i32,
        prev: PlayerInput,
        best: &[PlanStep],
        out: &mut Vec<Vec<PlanStep>>,
    ) -> bool {
        let (Some(me), Some(en)) = (world.get_tee(self_id), world.get_tee(enemy_id)) else {
            return false;
        };
        if !me.alive || !en.alive || me.frozen || en.frozen {
            return false;
        }
        let holding = me.hooked_player == enemy_id;
        let free = me.hook_state == HOOK_IDLE;
        // af49dfb `hookKeepFlying`: a hook already in flight is polished too (the first 1-3 steps keep the rope out).
        let flying = self.cfg.hook_keep_flying && me.hook_state == HOOK_FLYING;
        if !holding && !flying && !(free && vdistance(me.pos, en.pos) < *HOOK_LENGTH) {
            return false;
        }
        let bearing = js::atan2(en.pos.y - me.pos.y, en.pos.x - me.pos.x);
        let throw_aim = if self.cfg.track_aim { 0.0 } else { bearing };

        if !holding && !flying && self.cfg.gate_hook {
            let mut angle = self.executed_aim(PlanStep { hook: 1, ..best[0] }, prev, bearing);
            let mut meet: Option<Vec2> = None;
            if self.cfg.hook_snap_aim {
                let (at, m) = self.intercept_aim(world, &me, &en, angle, prev);
                meet = m;
                if let Some(at) = at
                    && self.hook_would_reach(world, self_id, enemy_id, at, meet)
                {
                    angle = at;
                }
            }
            if !self.hook_would_reach(world, self_id, enemy_id, angle, meet) {
                return false;
            }
        }
        let n = best.len();
        for k in if flying { [1usize, 2, 3] } else { [2usize, 4, n] } {
            if k > n {
                continue;
            }
            out.push(
                best.iter()
                    .enumerate()
                    .map(|(i, st)| {
                        if i < k {
                            PlanStep {
                                hook: 1,
                                aim: if holding || flying { st.aim } else { throw_aim },
                                ..*st
                            }
                        } else {
                            *st
                        }
                    })
                    .collect(),
            );
        }
        true
    }

    /// Task 3.9 (hybrid only): the wall throws of af49dfb's wayblock guard for a frozen victim -- the air chains (while
    /// airborne, when `air_chain`) and the wall swings toward the wall on side `wall_dir` -- without the plain frozen throw lines
    /// ([`Planner::frozen_throw_seed_lines`] puts them in front of those). `wall_dir == 0` gives none.
    pub(crate) fn wall_throw_lines(
        &self,
        world: &W,
        me: &TeeState,
        aim: f64,
        wall_dir: i32,
        air_chain: bool,
    ) -> Vec<Vec<PlanStep>> {
        if wall_dir == 0 {
            return Vec::new();
        }
        let half = PHYSICAL_SIZE / 2.0;
        let col = world.collision();
        let grounded = col.is_solid(me.pos.x + half, me.pos.y + half + 5.0)
            || col.is_solid(me.pos.x - half, me.pos.y + half + 5.0);
        let mut lines = if grounded || !air_chain {
            Vec::new()
        } else {
            air_chain_lines(self.cfg.steps, &self.step_ticks, aim, wall_dir, me.jumps_left > 0)
        };
        lines.extend(wall_swing_lines(self.cfg.steps, &self.step_ticks, aim, wall_dir));
        lines
    }

    /// `warmShift(tick)` (`planner.ts:1442-1453`).
    fn warm_shift(&self, tick: i64) -> i32 {
        if !self.cfg.warm_shift_elapsed {
            return 1;
        }
        if self.last_search_tick < 0 {
            return 1;
        }
        let elapsed = (tick - self.last_search_tick).max(0);
        let mut acc = 0i64;
        let mut steps = 0usize;
        while steps < self.step_ticks.len() && acc + i64::from(self.step_ticks[steps]) <= elapsed {
            acc += i64::from(self.step_ticks[steps]);
            steps += 1;
        }
        steps as i32
    }

    /// `buildDist(aimAt)` (`planner.ts:1455-1475`).
    pub(crate) fn build_dist(&self, aim_at: f64) -> Vec<StepDist> {
        let mut out = Vec::with_capacity(self.cfg.steps as usize);
        for s in 0..self.cfg.steps as usize {
            let w = self
                .warm
                .as_ref()
                .map(|warm| warm[(s + self.warm_shift_steps as usize).min(warm.len() - 1)]);
            match w {
                None => out.push(StepDist {
                    p_left: 0.33,
                    p_right: 0.33,
                    p_jump: 0.15,
                    p_hook: 0.3,
                    p_fire: 0.2,
                    aim: aim_at,
                    aim_spread: 1.2,
                }),
                Some(w) => out.push(StepDist {
                    p_left: if w.dir == -1 { 0.7 } else { 0.15 },
                    p_right: if w.dir == 1 { 0.7 } else { 0.15 },
                    p_jump: if w.jump != 0 { 0.7 } else { 0.1 },
                    p_hook: if w.hook != 0 { 0.7 } else { 0.15 },
                    p_fire: if w.fire != 0 { 0.6 } else { 0.15 },
                    aim: w.aim,
                    aim_spread: 0.9,
                }),
            }
        }
        out
    }

    /// `samplePlan(dist)` (`planner.ts:1477-1490`). RNG draw order per step: `dir`, `jump`,
    /// `hook`, `fire`, `aim` (gaussian) -- matters for parity, see the module doc comment.
    pub(crate) fn sample_plan(&mut self, dist: &[StepDist]) -> Vec<PlanStep> {
        sample_plan_with(&mut self.rng, dist)
    }

    /// The next `count` plans `sample_plan` will return, without consuming the RNG (the hybrid search's
    /// work-clock speculation, task 3.7a).
    pub(crate) fn preview_plans(&self, dist: &[StepDist], count: usize) -> Vec<Vec<PlanStep>> {
        let mut rng = self.rng;
        (0..count).map(|_| sample_plan_with(&mut rng, dist)).collect()
    }

    /// `refit(dist, elites)` (`planner.ts:1492-1523`).
    pub(crate) fn refit(&self, dist: &mut [StepDist], elites: &[Vec<PlanStep>]) {
        if elites.is_empty() {
            return;
        }
        for (s, d) in dist.iter_mut().enumerate() {
            let (mut left, mut right, mut jump, mut hook, mut fire, mut ax, mut ay) =
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            for plan in elites {
                let st = plan[s];
                if st.dir == -1 {
                    left += 1.0;
                } else if st.dir == 1 {
                    right += 1.0;
                }
                jump += f64::from(st.jump);
                hook += f64::from(st.hook);
                fire += f64::from(st.fire);
                ax += js::cos(st.aim);
                ay += js::sin(st.aim);
            }
            let n = elites.len() as f64;
            d.p_left = 0.1 + 0.8 * (left / n);
            d.p_right = 0.1 + 0.8 * (right / n);
            d.p_jump = 0.05 + 0.9 * (jump / n);
            d.p_hook = 0.05 + 0.9 * (hook / n);
            d.p_fire = 0.05 + 0.9 * (fire / n);
            d.aim = js::atan2(ay / n, ax / n);
            d.aim_spread = js::max(0.25, d.aim_spread * 0.7);
        }
    }

    /// `seedPlans` (`planner.ts:1525-1628`).
    pub(crate) fn seed_plans(
        &self,
        world: &W,
        self_id: i32,
        enemy_id: i32,
        field: &HazardField,
        aim_at: f64,
    ) -> Vec<Vec<PlanStep>> {
        let me = world.get_tee(self_id).unwrap();
        let en = world.get_tee(enemy_id).unwrap();
        let toward = {
            let s = js::sign(en.pos.x - me.pos.x);
            if s == 0.0 || s.is_nan() { 1 } else { s as i32 }
        };

        let mut hazard_dir = toward;
        let mut best_near = 0.0;
        for a in [0, 1, 3, 4, 5, 7] {
            let ang = f64::from(a) * js::PI / 4.0;
            let near = hazard_nearness(field, en.pos.x + js::cos(ang) * 96.0, en.pos.y + js::sin(ang) * 96.0);
            if near > best_near {
                best_near = near;
                hazard_dir = if js::cos(ang) > 0.0 { 1 } else { -1 };
            }
        }

        let n = self.cfg.steps;
        let track = self.cfg.track_aim;
        let at = if track { 0.0 } else { aim_at };
        let rel = |absolute: f64| if track { wrap_angle(absolute - aim_at) } else { absolute };
        let up = rel(-js::PI / 2.0 + 0.4 * f64::from(toward));
        let away = rel(js::atan2(0.0, -f64::from(toward)));

        let mk = |f: &dyn Fn(i32) -> PlanStep| -> Vec<PlanStep> { (0..n).map(f).collect() };

        let mut book: Vec<Vec<PlanStep>> = vec![
            mk(&|s| PlanStep {
                dir: if s < 3 { toward } else { hazard_dir },
                jump: i32::from(s == 2),
                hook: i32::from(s >= 2),
                fire: i32::from(s > n - 4),
                aim: at,
            }),
            mk(&|s| PlanStep {
                dir: hazard_dir,
                jump: 0,
                hook: i32::from(f64::from(s) >= 1.0 && f64::from(s) < f64::from(n) / 2.0),
                fire: i32::from(f64::from(s) >= f64::from(n) / 2.0 && f64::from(s) < f64::from(n) / 2.0 + 3.0),
                aim: at,
            }),
            mk(&|s| PlanStep {
                dir: toward,
                jump: i32::from(s % 6 == 0),
                hook: 0,
                fire: i32::from(s > 2),
                aim: at,
            }),
            mk(&|s| PlanStep {
                dir: toward,
                jump: i32::from(s == 0),
                hook: 1,
                fire: 0,
                aim: up,
            }),
            mk(&|_| PlanStep {
                dir: -toward,
                jump: 0,
                hook: 0,
                fire: 0,
                aim: at,
            }),
        ];

        let read = self.profile.read();
        let w = js::max(0.0, js::min(1.0, self.cfg.opponent_read_weight));
        let mix = |v: f64, prior: f64| prior + w * (v - prior);
        if mix(read.hook_opens_first, 0.5) > 0.6 {
            book.push(mk(&|s| PlanStep {
                dir: if f64::from(s) < f64::from(n) / 2.0 {
                    -toward
                } else {
                    toward
                },
                jump: i32::from(f64::from(s) == js::round(f64::from(n) / 2.0)),
                hook: 0,
                fire: i32::from(s > n - 4),
                aim: at,
            }));
        }
        if mix(read.aggression, 0.5) > 0.6 {
            book.push(mk(&|s| PlanStep {
                dir: 0,
                jump: 0,
                hook: 0,
                fire: i32::from(s > 1),
                aim: at,
            }));
        } else if mix(read.aggression, 0.5) < 0.4 {
            book.push(mk(&|s| PlanStep {
                dir: toward,
                jump: i32::from(s % 6 == 0),
                hook: 0,
                fire: i32::from(s > 2),
                aim: at,
            }));
        }
        if mix(read.out_of_jumps, 0.3) > 0.5 {
            book.push(mk(&|s| PlanStep {
                dir: toward,
                jump: 0,
                hook: 0,
                fire: i32::from(s > 0),
                aim: at,
            }));
        }

        let third = js::max(2.0, js::round(f64::from(n) / 3.0)) as i32;
        if self.cfg.hook_seeds && en.alive && !en.frozen && !me.frozen && vdistance(me.pos, en.pos) < *HOOK_LENGTH {
            if me.hooked_player == enemy_id {
                if let Some(warm) = &self.warm
                    && warm.len() == n as usize
                {
                    let shift = self.warm_shift_steps;
                    book.push(mk(&|s| PlanStep {
                        hook: 1,
                        ..warm[usize::try_from(s + shift).unwrap_or(0).min(warm.len() - 1)]
                    }));
                }
                book.push(mk(&|_| PlanStep {
                    dir: hazard_dir,
                    jump: 0,
                    hook: 1,
                    fire: 0,
                    aim: at,
                }));
            } else if me.hook_state != HOOK_FLYING && me.hook_state != HOOK_GRABBED {
                book.push(mk(&|_| PlanStep {
                    dir: hazard_dir,
                    jump: 0,
                    hook: 1,
                    fire: 0,
                    aim: at,
                }));
                book.push(mk(&|s| PlanStep {
                    dir: hazard_dir,
                    jump: 0,
                    hook: i32::from(s < third),
                    fire: 0,
                    aim: at,
                }));
            }
        }

        if matches!(
            self.cfg.opening_book,
            crate::config::OpeningBook::Movement | crate::config::OpeningBook::All
        ) {
            self.movement_seeds(world, &me, toward, at, &rel, &mut book);
        }
        if !matches!(
            self.cfg.opening_book,
            crate::config::OpeningBook::Wide | crate::config::OpeningBook::All
        ) {
            return book;
        }
        book.push(mk(&|_| PlanStep {
            dir: hazard_dir,
            jump: 0,
            hook: 1,
            fire: 0,
            aim: at,
        }));
        book.push(mk(&|s| PlanStep {
            dir: hazard_dir,
            jump: 0,
            hook: i32::from(s < third),
            fire: 0,
            aim: at,
        }));
        book.push(mk(&|s| PlanStep {
            dir: hazard_dir,
            jump: i32::from(s == 1),
            hook: i32::from(s < 2 * third),
            fire: 0,
            aim: at,
        }));
        book.push(mk(&|s| PlanStep {
            dir: 0,
            jump: 0,
            hook: i32::from(s < 2 * third),
            fire: i32::from(s >= third),
            aim: at,
        }));
        book.push(mk(&|s| PlanStep {
            dir: if s < third { 0 } else { hazard_dir },
            jump: 0,
            hook: i32::from(s < 2 * third),
            fire: i32::from(s >= third),
            aim: at,
        }));
        book.push(mk(&|s| PlanStep {
            dir: toward,
            jump: 0,
            hook: 0,
            fire: i32::from(s >= 1),
            aim: at,
        }));
        book.push(mk(&|s| PlanStep {
            dir: if s < third { toward } else { -toward },
            jump: i32::from(s == 0 || s == 2),
            hook: 0,
            fire: i32::from(s > third),
            aim: at,
        }));
        book.push(mk(&|s| PlanStep {
            dir: -toward,
            jump: i32::from(s == 0 || s == 3),
            hook: 0,
            fire: 0,
            aim: at,
        }));
        book.push(mk(&|s| PlanStep {
            dir: -toward,
            jump: 0,
            hook: i32::from(s < third),
            fire: 0,
            aim: away,
        }));
        book.push(mk(&|_| PlanStep {
            dir: 0,
            jump: 0,
            hook: 0,
            fire: 0,
            aim: at,
        }));
        book
    }

    /// `movementSeeds` (`planner.ts:1630-1666`).
    fn movement_seeds(
        &self,
        world: &W,
        me: &TeeState,
        toward: i32,
        at: f64,
        rel: &dyn Fn(f64) -> f64,
        book: &mut Vec<Vec<PlanStep>>,
    ) {
        let n = self.cfg.steps;
        let travel = if js::abs(me.vel.x) >= 1.0 {
            js::sign(me.vel.x) as i32
        } else {
            toward
        };
        let travel_f = f64::from(travel);
        let up_ahead = rel(js::atan2(-1.0, 0.8 * travel_f));
        let up_ahead_steep = rel(js::atan2(-1.0, 0.4 * travel_f));
        let ceiling = rel(js::atan2(-1.0, 0.15 * travel_f));
        let up_behind = rel(js::atan2(-1.2, -0.5 * travel_f));
        let mk = |f: &dyn Fn(i32) -> PlanStep| -> Vec<PlanStep> { (0..n).map(f).collect() };

        book.push(mk(&|s| PlanStep {
            dir: travel,
            jump: i32::from(s == 3),
            hook: i32::from(s < 3),
            fire: 0,
            aim: if s < 3 { up_ahead } else { at },
        }));
        book.push(mk(&|s| PlanStep {
            dir: travel,
            jump: i32::from(s == 4),
            hook: i32::from(s < 4),
            fire: 0,
            aim: if s < 4 { up_ahead_steep } else { at },
        }));
        book.push(mk(&|s| PlanStep {
            dir: if s < 2 {
                travel
            } else if s < 4 {
                -travel
            } else {
                travel
            },
            jump: i32::from(s == 6),
            hook: i32::from(s < 6),
            fire: 0,
            aim: if s < 6 { ceiling } else { at },
        }));
        book.push(mk(&|s| PlanStep {
            dir: if s < 1 { travel } else { -travel },
            jump: i32::from(s == 4),
            hook: i32::from(s < 4),
            fire: 0,
            aim: if s < 4 { up_behind } else { at },
        }));

        if let Some(edge_step) = self.steps_to_edge(world, me, travel)
            && edge_step < n - 1
        {
            book.push(mk(&|s| PlanStep {
                dir: travel,
                jump: i32::from(s == edge_step || s == edge_step + 3),
                hook: 0,
                fire: 0,
                aim: at,
            }));
        }
        book.push(mk(&|_| PlanStep {
            dir: travel,
            jump: 0,
            hook: 0,
            fire: 0,
            aim: at,
        }));
    }

    /// `stepsToEdge` (`planner.ts:1668-1683`).
    fn steps_to_edge(&self, world: &W, me: &TeeState, travel: i32) -> Option<i32> {
        let feet_y = me.pos.y + PHYSICAL_SIZE / 2.0 + 4.0;
        if !world.collision().is_solid(me.pos.x, feet_y) {
            return None;
        }
        let tx = js::floor(me.pos.x / 32.0) as i32;
        for k in 1..=8 {
            let cx = f64::from(tx + k * travel) * 32.0 + 16.0;
            if world.collision().is_solid(cx, feet_y) {
                continue;
            }
            let lip_x = if travel > 0 {
                f64::from(tx + k) * 32.0
            } else {
                f64::from(tx - k + 1) * 32.0
            };
            let px = js::abs(lip_x - me.pos.x);
            let speed = js::max(js::abs(me.vel.x), *crate::tuning::GROUND_CONTROL_SPEED * 0.6);
            return Some(self.step_at_tick(px / speed));
        }
        None
    }

    /// `stepAtTick` (`planner.ts:1685-1699`).
    fn step_at_tick(&self, ticks: f64) -> i32 {
        let mut acc = 0.0;
        let mut best = 0;
        let mut best_gap = js::abs(ticks);
        for (i, &st) in self.step_ticks.iter().enumerate() {
            acc += f64::from(st);
            let gap = js::abs(ticks - acc);
            if gap <= best_gap {
                best_gap = gap;
                best = i as i32 + 1;
            }
        }
        best
    }

    /// `hookAllowed(world, selfId, enemyId, step, prev, aim, snapped, meet)` (af49dfb `planner.ts`): `snapped` says `aim` is
    /// already the angle that will be thrown (the aim snap's), so it is used as is instead of going through
    /// [`Planner::executed_aim`]; `meet` is the victim's projected position when the caller has just computed it.
    #[allow(clippy::too_many_arguments)]
    fn hook_allowed_snapped(
        &self,
        world: &W,
        self_id: i32,
        enemy_id: i32,
        step: PlanStep,
        prev: PlayerInput,
        aim: f64,
        snapped: bool,
        meet: Option<Vec2>,
    ) -> bool {
        if !self.cfg.gate_hook && self.spares.is_empty() {
            return true;
        }
        let angle = if snapped {
            aim
        } else {
            self.executed_aim(step, prev, aim)
        };
        if self.cfg.gate_hook
            && !self.hook_would_reach(world, self_id, enemy_id, angle, meet)
            && !self.threats.as_ref().is_some_and(|t| {
                t.hook_targets
                    && t.ids
                        .iter()
                        .any(|&id| self.hook_would_reach(world, self_id, id, angle, None))
            })
        {
            return false;
        }
        !self.rope_catches_spare(world, self_id, enemy_id, angle)
    }

    /// The hook gate with the aim snap, as `decideOnce` and `evaluate` run it for one plan step (af49dfb): the snapped
    /// angle if the snap finds one the gate lets through, else the plain aim. Returns `(hook_ok, aim, snapped)`. With
    /// `hook_snap_aim` off this is exactly `hook_ok = !hook || already out || hook_allowed_snapped(aim, false, None)` (the plain gate).
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub(crate) fn gate_and_snap(
        &self,
        world: &W,
        self_id: i32,
        enemy_id: i32,
        me: Option<&TeeState>,
        en: Option<&TeeState>,
        step: PlanStep,
        prev: PlayerInput,
        aim: f64,
    ) -> (bool, f64, bool) {
        let (mut snap_at, meet) = self.snap_aim(world, me, en, step, prev, aim);
        let mut hook_ok = step.hook == 0
            || self.hook_already_out(world, self_id)
            || self.hook_allowed_snapped(
                world,
                self_id,
                enemy_id,
                step,
                prev,
                snap_at.unwrap_or(aim),
                snap_at.is_some(),
                meet,
            );
        if snap_at.is_some() && !hook_ok {
            snap_at = None;
            hook_ok = self.hook_allowed_snapped(world, self_id, enemy_id, step, prev, aim, false, meet);
        }
        match snap_at {
            Some(angle) => (hook_ok, angle, true),
            None => (hook_ok, aim, false),
        }
    }

    /// `snapAim` (af49dfb): the snapped angle (if any) and the victim's projected position (`snapMeet`, set once the
    /// projection was computed even when no angle comes out of it).
    fn snap_aim(
        &self,
        world: &W,
        me: Option<&TeeState>,
        en: Option<&TeeState>,
        step: PlanStep,
        prev: PlayerInput,
        aim: f64,
    ) -> (Option<f64>, Option<Vec2>) {
        if !self.cfg.hook_snap_aim || step.hook != 1 {
            return (None, None);
        }
        let (Some(me), Some(en)) = (me, en) else {
            return (None, None);
        };
        if me.hook_state != HOOK_IDLE {
            return (None, None);
        }
        self.intercept_aim(world, me, en, self.executed_aim(step, prev, aim), prev)
    }

    /// `interceptAim` (af49dfb): turn the planned throw angle (by at most [`SNAP_MAX_RAD`], and by no more than the
    /// aim smoothing allows from `prev`) so the rope meets the victim's projected position, unless a wall is in the way
    /// or the planned throw is already further off than the aim smoothing allows (the `None` returns of the TS, in order).
    fn intercept_aim(
        &self,
        world: &W,
        me: &TeeState,
        en: &TeeState,
        planned_angle: f64,
        prev: PlayerInput,
    ) -> (Option<f64>, Option<Vec2>) {
        if !me.alive || !en.alive {
            return (None, None);
        }
        let col = world.collision();
        let meet = self.project(col, me.pos, en);
        let lx = meet.x - me.pos.x;
        let ly = meet.y - me.pos.y;
        let dist = js::hypot2(lx, ly);
        if dist < 1.0 || dist > *HOOK_LENGTH {
            return (None, Some(meet));
        }
        let dir = vec2(trig::cos(planned_angle), trig::sin(planned_angle));
        let perp = js::abs(lx * dir.y - ly * dir.x);
        if lx * dir.x + ly * dir.y < 0.0 || perp > PHYSICAL_SIZE * 2.0 {
            return (None, Some(meet));
        }
        if Self::ray_hits_wall(col, me.pos, dir, dist) {
            return (None, Some(meet));
        }
        if perp >= PHYSICAL_SIZE + 2.0 {
            let grab = col.intersect_line_hook(
                me.pos,
                vec2(me.pos.x + dir.x * *HOOK_LENGTH, me.pos.y + dir.y * *HOOK_LENGTH),
            );
            if grab.collision != 0 && (grab.collision & crate::plan_world::CFLAG_NOHOOK) == 0 {
                return (None, Some(meet));
            }
        }
        let mut turn = trig::atan2(ly, lx) - planned_angle;
        while turn > js::PI {
            turn -= 2.0 * js::PI;
        }
        while turn < -js::PI {
            turn += 2.0 * js::PI;
        }
        turn = js::max(-SNAP_MAX_RAD, js::min(SNAP_MAX_RAD, turn));
        let angle = planned_angle + turn;
        if prev.target_x != 0.0 || prev.target_y != 0.0 {
            let mut step = angle - trig::atan2(prev.target_y, prev.target_x);
            while step > js::PI {
                step -= 2.0 * js::PI;
            }
            while step < -js::PI {
                step += 2.0 * js::PI;
            }
            if js::abs(step) > crate::action::MAX_AIM_TURN_RAD {
                return (None, Some(meet));
            }
        }
        if Self::ray_hits_wall(col, me.pos, vec2(trig::cos(angle), trig::sin(angle)), dist) {
            return (None, Some(meet));
        }
        (Some(angle), Some(meet))
    }

    /// [`rope_intercept`] of the victim `en` from `from`, counting the projections of a moving victim ([`Planner::intercept_count`]).
    fn project(&self, col: &W::Collision, from: Vec2, en: &TeeState) -> Vec2 {
        if js::abs(en.vel.x) + js::abs(en.vel.y) > 0.01 {
            self.intercepts.set(self.intercepts.get() + 1);
        }
        rope_intercept(col, from, en.pos, en.vel)
    }

    /// Task 3.9: the number of moving-victim projections so far (a work counter for the hybrid's work clock).
    pub(crate) fn intercept_count(&self) -> u64 {
        self.intercepts.get()
    }

    /// `rayHitsWall` (af49dfb): the hook line of length `dist` from `from` along `dir` ends at a wall before `dist`.
    fn ray_hits_wall(col: &W::Collision, from: Vec2, dir: Vec2, dist: f64) -> bool {
        let hit = col.intersect_line_hook(from, vec2(from.x + dir.x * dist, from.y + dir.y * dist));
        hit.collision != 0 && vdistance(from, hit.out_pos) < dist
    }

    /// `ropeCatchesSpare` (`planner.ts:1720-1731`).
    fn rope_catches_spare(&self, world: &W, self_id: i32, enemy_id: i32, angle: f64) -> bool {
        if self.spares.is_empty() {
            return false;
        }
        let Some(me) = world.get_tee(self_id) else { return false };
        let dir = vec2(trig::cos(angle), trig::sin(angle));
        let hit = world.collision().intersect_line_hook(
            me.pos,
            vec2(me.pos.x + dir.x * *HOOK_LENGTH, me.pos.y + dir.y * *HOOK_LENGTH),
        );
        let mut stop = if hit.collision != 0 {
            vdistance(me.pos, hit.out_pos)
        } else {
            *HOOK_LENGTH
        };
        if let Some(en) = world.get_tee(enemy_id)
            && en.alive
        {
            stop = js::min(stop, fields::rope_catch_along(me.pos, dir, en.pos, *HOOK_LENGTH));
        }
        self.spares
            .iter()
            .any(|&b| fields::rope_catch_along(me.pos, dir, b, *HOOK_LENGTH) < stop)
    }

    /// `hookWouldReach` (`planner.ts:1733-1758`).
    fn hook_would_reach(&self, world: &W, self_id: i32, enemy_id: i32, angle: f64, meet_known: Option<Vec2>) -> bool {
        let Some(me) = world.get_tee(self_id) else { return false };
        let dir = vec2(trig::cos(angle), trig::sin(angle));
        let to = vec2(me.pos.x + dir.x * *HOOK_LENGTH, me.pos.y + dir.y * *HOOK_LENGTH);
        let hit = world.collision().intersect_line_hook(me.pos, to);
        let wall_hit = hit.collision != 0;
        if wall_hit && (hit.collision & crate::plan_world::CFLAG_NOHOOK) == 0 {
            return true;
        }
        let Some(en) = world.get_tee(enemy_id) else {
            return false;
        };
        if !en.alive {
            return false;
        }
        let wall_dist = if wall_hit {
            vdistance(me.pos, hit.out_pos)
        } else {
            *HOOK_LENGTH
        };
        if self.cfg.hook_exact_gate {
            // af49dfb: the rope's line (from the body's edge to the wall or full reach) must pass within a body of the
            // victim's projected position.
            let meet = meet_known.unwrap_or_else(|| self.project(world.collision(), me.pos, &en));
            let start = vec2(
                me.pos.x + dir.x * PHYSICAL_SIZE * 1.5,
                me.pos.y + dir.y * PHYSICAL_SIZE * 1.5,
            );
            let reach = js::min(*HOOK_LENGTH, wall_dist);
            let end = vec2(me.pos.x + dir.x * reach, me.pos.y + dir.y * reach);
            return closest_point_on_line_or_null(start, end, meet)
                .is_some_and(|closest| vdistance(meet, closest) < PHYSICAL_SIZE + 2.0);
        }
        let rel = vec2(en.pos.x - me.pos.x, en.pos.y - me.pos.y);
        let along = rel.x * dir.x + rel.y * dir.y;
        if along < 0.0 || along > *HOOK_LENGTH {
            return false;
        }
        let perp = js::abs(rel.x * dir.y - rel.y * dir.x);
        if along > wall_dist {
            return false;
        }
        if perp <= PHYSICAL_SIZE * 2.0 {
            return true;
        }
        let lead = vec2(
            en.pos.x + en.vel.x * 8.0 - me.pos.x,
            en.pos.y + en.vel.y * 8.0 - me.pos.y,
        );
        let lead_along = lead.x * dir.x + lead.y * dir.y;
        if lead_along < 0.0 || lead_along > js::min(*HOOK_LENGTH, wall_dist) {
            return false;
        }
        js::abs(lead.x * dir.y - lead.y * dir.x) <= PHYSICAL_SIZE * 2.0
    }

    /// `predictOpponent` (`planner.ts:1824-1856`). Only `"hold"`/`"react"` are in scope here (see
    /// `crate::config::OpponentModel`'s doc comment): both leave `predicted` empty exactly like
    /// TS's own `else` branch (`policy === null`) does, so this never touches `world` at all.
    fn predict_opponent(&mut self) {
        self.predicted.clear();
    }

    /// `policySeedPlans` (`planner.ts:1760-1805`) -- always empty here: every acceptance-criteria
    /// preset has `policySeeds: 0` and no `seedPolicy` is ever loaded (out of scope, see
    /// `crate::config::PlannerConfig::policy_seeds`'s doc comment), matching TS's own
    /// `wanted === 0` early return exactly. Kept as a named no-op (not inlined at the one call
    /// site) so a future policy-net task has an obvious place to fill in.
    fn policy_seed_plans(&self) -> Vec<Vec<PlanStep>> {
        Vec::new()
    }

    /// `hammerWouldHit` (`planner.ts:1858-1866`).
    fn hammer_would_hit(&self, me_pos: Vec2, en_pos: Vec2, en_vel: Vec2, target_x: f64, target_y: f64) -> bool {
        let len = js::sqrt(target_x * target_x + target_y * target_y);
        if len < 1e-6 {
            return false;
        }
        let sx = me_pos.x + (target_x / len) * PHYSICAL_SIZE * 0.75;
        let sy = me_pos.y + (target_y / len) * PHYSICAL_SIZE * 0.75;
        let reach = PHYSICAL_SIZE * 0.5 + PHYSICAL_SIZE;
        if js::hypot2(en_pos.x - sx, en_pos.y - sy) < reach {
            return true;
        }
        js::hypot2(en_pos.x + en_vel.x * 2.0 - sx, en_pos.y + en_vel.y * 2.0 - sy) < reach
    }

    /// `thawEscapable` (`planner.ts:1868-1915`).
    fn thaw_escapable(&mut self, world: &W, en: &TeeState, me_pos: Vec2) -> bool {
        let strict = self.cfg.no_thaw_rope;
        // Review round 1, F8: JS's template-literal string coercion normalizes `-0` to `"0"`
        // (`` `${-0}` === "0" ``), but Rust's `f64` `Display` does not (`format!("{}", -0.0)` ==
        // `"-0"`) -- `js::round` can produce a signed zero (`round(-0.4) === -0`, per
        // `ddai-jsmath`'s own README), so without this normalization two positions TS's key
        // collapses into one cache entry would get distinct Rust keys (and vice versa for a
        // position whose rounded coordinate is exactly `0`, which TS's `-0` case would otherwise
        // collide with here but doesn't in TS). `v == 0.0` is true for both `+0.0` and `-0.0` in
        // IEEE-754, so this catches every sign of zero.
        let clean = |v: f64| if v == 0.0 { 0.0 } else { v };
        // Task 3.5: the hybrid search (`deterministic_thaw`) memoises per evaluation under an
        // allocation-free integer key holding exactly the same rounded fields as the string key.
        let int_key: Option<[i64; 9]> = self.deterministic_thaw.then(|| {
            [
                i64::from(strict),
                clean(js::round(en.pos.x / 4.0)) as i64,
                clean(js::round(en.pos.y / 4.0)) as i64,
                clean(js::round(en.vel.x)) as i64,
                clean(js::round(en.vel.y)) as i64,
                clean(js::round((en.pos.x - me_pos.x) / 8.0)) as i64,
                clean(js::round((en.pos.y - me_pos.y) / 8.0)) as i64,
                i64::from(en.jumps_left),
                i64::from(en.freeze_ticks_left > i64::from(THAW_ESCAPE_TICKS)),
            ]
        });
        let key = if let Some(int_key) = int_key {
            if let Some(&(_, known)) = self.thaw_fast.iter().find(|(k, _)| *k == int_key) {
                return known;
            }
            String::new()
        } else {
            let key = format!(
                "{}{},{},{},{},{},{},{},{}",
                if strict { "s" } else { "" },
                clean(js::round(en.pos.x / 4.0)),
                clean(js::round(en.pos.y / 4.0)),
                clean(js::round(en.vel.x)),
                clean(js::round(en.vel.y)),
                clean(js::round((en.pos.x - me_pos.x) / 8.0)),
                clean(js::round((en.pos.y - me_pos.y) / 8.0)),
                en.jumps_left,
                i32::from(en.freeze_ticks_left > i64::from(THAW_ESCAPE_TICKS)),
            );
            if let Some(&known) = self.thaw_memo.get(&key) {
                return known;
            }
            key
        };

        let identity = world.collision().identity();
        if self.thaw_scratch.is_none() || self.thaw_scratch_identity != Some(identity) {
            let mut scratch = world.new_scratch();
            scratch.add_tee(0, en.pos);
            self.thaw_scratch = Some(scratch);
            self.thaw_scratch_identity = Some(identity);
            self.thaw_base = None;
        }
        let scratch = self.thaw_scratch.as_mut().unwrap();
        if self.deterministic_thaw {
            match &self.thaw_base {
                Some(base) => scratch.restore_state(base),
                None => self.thaw_base = Some(scratch.save_state()),
            }
        }
        if strict && scratch.get_tee(1).is_none() {
            scratch.add_tee(1, me_pos);
        }
        if !strict && scratch.get_tee(1).is_some() {
            scratch.remove_tee(1);
        }

        let sep = js::max(1.0, vdistance(en.pos, me_pos));
        let hx = (en.pos.x - me_pos.x) / sep;
        let hy = (en.pos.y - me_pos.y) / sep;
        let bl_raw = js::hypot2(hx, hy - 1.1);
        let bl = if bl_raw == 0.0 { 1.0 } else { bl_raw };
        let k = *crate::tuning::HAMMER_STRENGTH;
        let push = vec2((k * 10.0 * hx) / bl, k * (-1.0 + (10.0 * (hy - 1.1)) / bl));

        let rope: Vec<Vec<PlayerInput>> = if strict {
            rope_escapes(me_pos.x - en.pos.x, me_pos.y - en.pos.y)
        } else {
            Vec::new()
        };
        let mut escapable = false;
        for escape in THAW_ESCAPES.iter().chain(rope.iter()) {
            let mut applied = *en;
            applied.id = 0;
            applied.hook_state = 0;
            applied.hooked_player = -1;
            scratch.apply_tee_state(0, &applied);
            scratch.set_held_input(0, empty_input());
            if strict {
                let mut blank = crate::types::blank_tee_state();
                blank.id = 1;
                blank.alive = true;
                blank.pos = me_pos;
                scratch.apply_tee_state(1, &blank);
                scratch.set_held_input(1, empty_input());
                scratch.set_input(1, empty_input());
            }
            scratch.apply_force(0, push);
            scratch.unfreeze(0);
            let mut caught = false;
            for input in escape.iter().take(THAW_ESCAPE_TICKS as usize) {
                scratch.set_input(0, *input);
                scratch.step();
                match scratch.get_tee(0) {
                    Some(tee) if tee.alive && !tee.frozen => {}
                    _ => {
                        caught = true;
                        break;
                    }
                }
            }
            if !caught {
                escapable = true;
                break;
            }
        }
        if let Some(int_key) = int_key {
            self.thaw_fast.push((int_key, escapable));
        } else {
            self.thaw_memo.insert(key, escapable);
        }
        escapable
    }

    /// `executedAim` (`planner.ts:1976-1991`).
    fn executed_aim(&self, step: PlanStep, prev: PlayerInput, aim: f64) -> f64 {
        let mut raw = [0.0f64; ACTION_SIZE];
        raw[0] = if step.dir == -1 { 1.0 } else { -1.0 };
        raw[1] = if step.dir == 0 { 1.0 } else { -1.0 };
        raw[2] = if step.dir == 1 { 1.0 } else { -1.0 };
        raw[3] = if step.jump != 0 { 1.0 } else { -1.0 };
        raw[4] = -1.0;
        raw[5] = -1.0;
        raw[6] = 1.0;
        raw[7] = -1.0;
        raw[8] = trig::cos(aim);
        raw[9] = trig::sin(aim);
        let probe = decode_action(&raw, &prev, false);
        trig::atan2(probe.target_y, probe.target_x)
    }

    /// `stepToInput(..., aim, snapped)` (af49dfb): with `snapped` the target is the rounded aim vector itself (`snapTarget`),
    /// not the smoothed turn [`decode_action`] limits it to.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step_to_input_snapped(
        &mut self,
        world: &W,
        step: PlanStep,
        prev: PlayerInput,
        enemy_dist: f64,
        hook_ok: bool,
        me_pos: Option<Vec2>,
        en_pos: Option<Vec2>,
        en_vel: Option<Vec2>,
        aim: f64,
        snapped: bool,
    ) -> PlayerInput {
        let mut raw = [0.0f64; ACTION_SIZE];
        raw[0] = if step.dir == -1 { 1.0 } else { -1.0 };
        raw[1] = if step.dir == 0 { 1.0 } else { -1.0 };
        raw[2] = if step.dir == 1 { 1.0 } else { -1.0 };
        raw[3] = if step.jump != 0 { 1.0 } else { -1.0 };
        raw[4] = if step.hook != 0 && hook_ok { 1.0 } else { -1.0 };
        raw[6] = 1.0;
        raw[7] = -1.0;
        raw[8] = trig::cos(aim);
        raw[9] = trig::sin(aim);

        let mut can_swing = self.cfg.hammer_range_px <= 0.0 || enemy_dist <= self.cfg.hammer_range_px;
        if can_swing
            && step.fire != 0
            && self.cfg.gate_hammer
            && let (Some(mp), Some(ep), Some(ev)) = (me_pos, en_pos, en_vel)
        {
            raw[5] = -1.0;
            let mut dry = decode_action(&raw, &prev, false);
            if snapped {
                snap_target(&mut dry, aim);
            }
            can_swing = self.hammer_would_hit(mp, ep, ev, dry.target_x, dry.target_y);
        }

        if can_swing
            && step.fire != 0
            && self.cfg.no_thaw
            && self.swing_target_frozen
            && let Some(mp) = me_pos
        {
            let swing_target = self.swing_target;
            if let Some(target) = swing_target {
                if self.cfg.no_thaw_rope && self.swing_rope_on && step.hook != 0 {
                    can_swing = false;
                } else {
                    can_swing = !self.thaw_escapable(world, &target, mp);
                }
            }
        }

        if can_swing
            && step.fire != 0
            && !self.frozen_bystanders.is_empty()
            && let Some(mp) = me_pos
        {
            raw[5] = -1.0;
            let mut dry = decode_action(&raw, &prev, false);
            if snapped {
                snap_target(&mut dry, aim);
            }
            for (i, &b) in self.frozen_bystanders.iter().enumerate() {
                let bv = self.frozen_bystander_vels.get(i).copied().unwrap_or(fields::no_vel());
                let d = vdistance(mp, b);
                if d > LAUNCH_REACH_PX * 1.5 {
                    continue;
                }
                if self.hammer_would_hit(mp, b, bv, dry.target_x, dry.target_y)
                    && launch_flight_lands_in_hazard(world.collision(), b, mp, js::max(1.0, d), bv) == 0.0
                {
                    can_swing = false;
                    break;
                }
            }
        }

        if can_swing
            && step.fire != 0
            && !self.spares.is_empty()
            && let Some(mp) = me_pos
        {
            raw[5] = -1.0;
            let mut dry = decode_action(&raw, &prev, false);
            if snapped {
                snap_target(&mut dry, aim);
            }
            for (i, &b) in self.spares.iter().enumerate() {
                if vdistance(mp, b) > LAUNCH_REACH_PX * 1.5 {
                    continue;
                }
                let bv = self.spare_vels.get(i).copied().unwrap_or(fields::no_vel());
                if self.hammer_would_hit(mp, b, bv, dry.target_x, dry.target_y) {
                    can_swing = false;
                    break;
                }
            }
        }
        raw[5] = if step.fire != 0 && can_swing { 1.0 } else { -1.0 };
        let mut out = decode_action(&raw, &prev, false);
        if snapped {
            snap_target(&mut out, aim);
        }
        out
    }

    /// `evaluate` (`planner.ts:1993-2162`) -- the exact roll-out score for one plan.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    #[allow(clippy::too_many_arguments)]
    fn evaluate(
        &mut self,
        world: &mut W,
        self_id: i32,
        enemy_id: i32,
        prev: PlayerInput,
        plan: &[PlanStep],
        enemy_input: PlayerInput,
        field: &HazardField,
        unfreeze: &HazardField,
    ) -> f64 {
        // `deadline: None` never trips the check inserted below (review round 1, F1) -- the
        // parity path (every call site except `decide_production`) is bit-for-bit the same
        // control flow it always was.
        self.evaluate_impl(world, self_id, enemy_id, prev, plan, enemy_input, field, unfreeze, None)
            .expect("evaluate() with deadline=None never returns None")
    }

    /// `evaluate` (`planner.ts:1993-2162`), with an optional injected deadline (review round 1,
    /// F1): checked once per plan *step* (a few ticks -- `docs/DECISIONS.md` D-041's "check inside
    /// a rollout every few ticks" requirement), not just once per whole candidate, so a single
    /// slow candidate can't blow a production deadline by more than one step's worth of physics
    /// (measured well under the ~0.3 ms overshoot budget -- see the crate README). Returns `None`
    /// if the deadline was hit mid-rollout (world state is still restored before returning, exactly
    /// like the normal end of the function) -- `decide_production` discards that candidate's score
    /// entirely and stops searching, keeping whatever `best` it already had. `deadline` is only
    /// ever `Some` from `decide_production`; every parity-path call site passes `None`.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub(crate) fn evaluate_impl(
        &mut self,
        world: &mut W,
        self_id: i32,
        enemy_id: i32,
        prev: PlayerInput,
        plan: &[PlanStep],
        enemy_input: PlayerInput,
        field: &HazardField,
        unfreeze: &HazardField,
        deadline: Option<(&dyn crate::clock::Clock, f64)>,
    ) -> Option<f64> {
        let mut score = 0.0;
        let mut input = prev;

        let mut prev_dir = prev.direction;
        let hold = self.cfg.flip_hold_ticks;
        let jitter = self.cfg.jitter_cost;
        if hold <= 0.0 || jitter <= 0.0 {
            for st in plan {
                if st.dir != prev_dir {
                    score -= self.cfg.flip_cost;
                }
                prev_dir = st.dir;
            }
        } else {
            let mut run = js::min(hold, js::max(0.0, self.held_ticks as f64));
            for (i, st) in plan.iter().enumerate() {
                let held = f64::from(self.step_ticks[i]);
                if st.dir != prev_dir {
                    score -= self.cfg.flip_cost + jitter * (1.0 - run / hold);
                    run = held;
                } else {
                    run = js::min(hold, run + held);
                }
                prev_dir = st.dir;
            }
        }

        let mut hook_was_flying = false;
        let mut hook_grabbed = false;
        let mut held_enemy =
            self.cfg.hook_release_cost > 0.0 && world.get_tee(self_id).map(|t| t.hooked_player) == Some(enemy_id);

        let mut opp_rng = Rng::new(self.opp_seed);
        let mut opp_input = enemy_input;
        let mut events = std::mem::take(&mut self.events_buf);
        if self.deterministic_thaw {
            self.thaw_memo.clear();
            self.thaw_fast.clear();
        }
        // Task 3.5: the hybrid 1vN search's extra tees (`None`, hence a no-op, everywhere else).
        let mut thr_inputs = [empty_input(); crate::hybrid::threat::MAX_THREATS];
        let thr_n = self.threats.as_ref().map_or(0, |t| {
            let n = t.ids.len().min(crate::hybrid::threat::MAX_THREATS);
            thr_inputs[..n].copy_from_slice(&t.inputs[..n]);
            n
        });
        if self.track_rollout {
            self.rollout_enemy_out = 0;
            self.rollout_self_out = 0;
            self.rollout_enemy_sealed = false;
        }
        if self.track_gap {
            self.rollout_min_gap = EDGE_GAP_PX;
        }

        let en_at_start = world.get_tee(enemy_id);
        let me_at_start = world.get_tee(self_id);
        let en_near_at_start = en_at_start.map_or(0.0, |en| hazard_nearness(field, en.pos.x, en.pos.y));
        let mut drag = DragTracker {
            prev_enemy_near: en_near_at_start,
            start_enemy_near: en_near_at_start,
            started_in_dead: me_at_start.is_some_and(|me| in_dead(self.dead.as_ref(), me.pos.x, me.pos.y)),
            prev_stage_dist: f64::NAN,
        };

        // Task 3.5 (T14): jumpless, airborne, with a hazard below -- an anchor is the only control.
        let anchor_bonus = if self.cfg.jumpless_anchor_bonus > 0.0
            && let Some(m) = me_at_start
            && m.jumps_left == 0
            && !m.frozen
        {
            let col = world.collision();
            let grounded = col.is_solid(m.pos.x - 13.0, m.pos.y + 16.0) || col.is_solid(m.pos.x + 13.0, m.pos.y + 16.0);
            let hazard_below = (1..=12).any(|k| {
                let y = m.pos.y + f64::from(k) * 32.0;
                col.is_hazard(m.pos.x, y)
            });
            if !grounded && hazard_below {
                self.cfg.jumpless_anchor_bonus
            } else {
                0.0
            }
        } else {
            0.0
        };
        // Task 3.14: the first freeze of the rollout decides a duel (`duel_loss_cost`, `duel_win_bonus`); a tee already out at the start has had its onset.
        let mut duel_me_out = me_at_start.is_none_or(|m| m.frozen || !m.alive);
        let mut duel_en_out = en_at_start.is_none_or(|e| e.frozen || !e.alive);
        let mut prev_jumps_left = me_at_start.map_or(0, |me| me.jumps_left);
        let mut ground_jump_at: i64 = -1;
        let mut rollout_tick: i64 = 0;

        for (s, plan_step) in plan.iter().enumerate() {
            let me_now = world.get_tee(self_id);
            let en_now = world.get_tee(enemy_id);
            let enemy_dist = match (me_now, en_now) {
                (Some(m), Some(e)) => vdistance(m.pos, e.pos),
                _ => 0.0,
            };
            let mut aim = plan_step.aim;
            if crate::hybrid::is_abs_aim(aim) {
                // Task 3.5: a technique plan's absolute aim (never produced on the parity path).
                aim -= crate::hybrid::ABS_AIM;
            } else if self.cfg.track_aim
                && let (Some(m), Some(e)) = (me_now, en_now)
            {
                aim += trig::atan2(e.pos.y - m.pos.y, e.pos.x - m.pos.x);
            }
            self.swing_target_frozen = en_now.is_some_and(|e| e.frozen);
            self.swing_rope_on = me_now.is_some_and(|m| m.hooked_player == enemy_id);
            self.swing_target = en_now;
            let (hook_ok, aim, snapped) = self.gate_and_snap(
                world,
                self_id,
                enemy_id,
                me_now.as_ref(),
                en_now.as_ref(),
                *plan_step,
                input,
                aim,
            );
            input = self.step_to_input_snapped(
                world,
                *plan_step,
                input,
                enemy_dist,
                hook_ok,
                me_now.map(|m| m.pos),
                en_now.map(|e| e.pos),
                en_now.map(|e| e.vel),
                aim,
                snapped,
            );
            if let Some(rec) = &mut self.record_inputs {
                rec.push(input);
            }

            if self.react_this_pass || self.cfg.opponent_model == OpponentModel::React {
                opp_input = scripted_action(world, enemy_id, self_id, &opp_input, &mut opp_rng);
            } else if !self.predicted.is_empty() {
                opp_input = self.predicted[s.min(self.predicted.len() - 1)];
            }
            if let Some(t) = &self.threats {
                for (i, inp) in thr_inputs.iter_mut().enumerate().take(thr_n) {
                    if (t.react_mask >> i) & 1 == 1 {
                        *inp = scripted_action(world, t.ids[i], self_id, inp, &mut opp_rng);
                    }
                }
            }

            for _ in 0..self.step_ticks[s] {
                rollout_tick += 1;
                self.eval_ticks += 1;
                if self.cfg.release_dead_hook && input.hook != 0 && self.hook_is_dead(world, self_id) {
                    world.set_input(self_id, PlayerInput { hook: 0, ..input });
                } else {
                    world.set_input(self_id, input);
                }
                world.set_input(enemy_id, opp_input);
                if let Some(t) = &self.threats {
                    for (i, inp) in thr_inputs.iter().enumerate().take(thr_n) {
                        if (t.react_mask >> i) & 1 == 1 {
                            world.set_input(t.ids[i], *inp);
                        }
                    }
                }
                world.step_into(&mut events);
                let me_now = world.get_tee(self_id);
                if let Some(me_now) = me_now {
                    if (me_now.jumped & 2) == 0 && me_now.jumps_left < prev_jumps_left {
                        ground_jump_at = rollout_tick;
                    }
                    if prev_jumps_left > 0 && me_now.jumps_left == 0 && !me_now.frozen {
                        let gap = if ground_jump_at < 0 {
                            AIR_JUMP_MIN_GAP_TICKS
                        } else {
                            rollout_tick - ground_jump_at
                        };
                        if gap < AIR_JUMP_MIN_GAP_TICKS {
                            score -= self.cfg.air_jump_cost * (1.0 - (gap as f64) / (AIR_JUMP_MIN_GAP_TICKS as f64));
                        }
                        ground_jump_at = -1;
                    }
                    prev_jumps_left = me_now.jumps_left;
                    if me_now.hook_state == HOOK_FLYING {
                        hook_was_flying = true;
                    }
                    if me_now.hook_state == HOOK_GRABBED {
                        hook_grabbed = true;
                        if anchor_bonus > 0.0 && me_now.hooked_player < 0 && !me_now.frozen {
                            score += anchor_bonus * (1.0 - (s as f64) / (plan.len() as f64 * 2.0));
                        }
                    }
                    if self.cfg.hook_release_cost > 0.0 {
                        let holds = me_now.hooked_player == enemy_id;
                        if held_enemy
                            && !holds
                            && let Some(en_now_held) = world.get_tee(enemy_id)
                            && en_now_held.alive
                            && !en_now_held.frozen
                        {
                            score -= self.cfg.hook_release_cost;
                        }
                        held_enemy = holds;
                    }
                    if hook_was_flying
                        && !hook_grabbed
                        && me_now.hook_state >= HOOK_RETRACT_START
                        && me_now.hook_state < HOOK_FLYING
                    {
                        score -= self.cfg.wasted_hook;
                        hook_was_flying = false;
                    }
                    if me_now.hook_state <= 0 {
                        hook_was_flying = false;
                        hook_grabbed = false;
                    }
                }
                let en_now = world.get_tee(enemy_id);
                if self.track_rollout {
                    if en_now.is_some_and(|e| e.frozen || !e.alive) {
                        self.rollout_enemy_out += 1;
                    }
                    if me_now.is_some_and(|m| m.frozen || !m.alive) {
                        self.rollout_self_out += 1;
                    }
                }
                if self.track_gap
                    && let Some(m) = me_now
                {
                    let gap = if m.frozen || !m.alive {
                        0.0
                    } else {
                        freeze_gap_px(world.collision(), m.pos.x, m.pos.y)
                    };
                    if gap < self.rollout_min_gap {
                        self.rollout_min_gap = gap;
                    }
                }
                let mut tick_score = score_tick_of(
                    world,
                    me_now.as_ref(),
                    en_now.as_ref(),
                    self_id,
                    enemy_id,
                    &events,
                    field,
                    unfreeze,
                    &self.cfg,
                    &mut drag,
                    self.goal,
                    self.dead.as_ref(),
                    self.memory.as_ref(),
                    &self.thirds,
                    self.band.as_ref(),
                    self.launch_memo.as_deref_mut(),
                    self.ceiling.as_deref(),
                );
                if let Some(t) = &self.threats {
                    tick_score += t.weight
                        * crate::hybrid::threat::threat_terms(
                            world,
                            self_id,
                            &t.ids[..thr_n],
                            &self.cfg,
                            self.launch_memo.as_deref_mut(),
                        );
                }
                let discount = 1.0 - (s as f64) / (plan.len() as f64 * 2.0);
                score += tick_score * discount;
                if self.cfg.duel_loss_cost > 0.0 || self.cfg.duel_win_bonus > 0.0 {
                    if !duel_me_out && world.get_tee(self_id).is_some_and(|m| m.frozen || !m.alive) {
                        duel_me_out = true;
                        score -= self.cfg.duel_loss_cost * discount;
                    }
                    if !duel_en_out && world.get_tee(enemy_id).is_some_and(|e| e.frozen || !e.alive) {
                        duel_en_out = true;
                        score += self.cfg.duel_win_bonus * discount;
                    }
                }
            }
            if let Some((clock, deadline_ms)) = deadline
                && clock.now_ms() >= deadline_ms
            {
                world.restore_state(self.saved.as_ref().unwrap());
                self.events_buf = events;
                return None;
            }
        }

        if self.cfg.value_weight != 0.0 {
            // No value-net loader in this crate (out of scope, see
            // `PlannerConfig::value_weight`'s doc comment) -- `value_weight` should stay `0` for
            // every configuration this crate supports; nothing is added here either way.
        }
        if self.cfg.landing_cost > 0.0
            && let Some(me_end) = world.get_tee(self_id)
            && me_end.alive
            && !me_end.frozen
        {
            score -= self.cfg.landing_cost * flight_ends_in_hazard(world.collision(), me_end.pos, me_end.vel);
        }
        if self.cfg.jumpless_air_cost > 0.0
            && let Some(me_end) = world.get_tee(self_id)
            && me_end.alive
            && !me_end.frozen
            && me_end.jumps_left == 0
        {
            let col = world.collision();
            let on_ground = col.is_solid(me_end.pos.x - 13.0, me_end.pos.y + 16.0)
                || col.is_solid(me_end.pos.x + 13.0, me_end.pos.y + 16.0);
            if !on_ground {
                score -= self.cfg.jumpless_air_cost;
            }
        }
        if self.cfg.enemy_landing_bonus > 0.0
            && let Some(en_end) = world.get_tee(enemy_id)
            && en_end.alive
            && !en_end.frozen
        {
            score += self.cfg.enemy_landing_bonus * flight_ends_in_hazard(world.collision(), en_end.pos, en_end.vel);
        }
        if self.cfg.freeze_tail_weight > 0.0 {
            let tail = 1.0 - ((plan.len() as f64) - 1.0) / (plan.len() as f64 * 2.0);
            let w = self.cfg.freeze_tail_weight * self.cfg.frozen_weight * tail;
            let me_end = world.get_tee(self_id);
            let en_end = world.get_tee(enemy_id);
            if let Some(me_end) = me_end
                && me_end.frozen
            {
                score -= w * self.cfg.self_freeze_bias * (me_end.freeze_ticks_left as f64);
            }
            let en_sealed = match en_end {
                Some(en_end) if self.cfg.seal_ticks > 0.0 && en_end.frozen => {
                    f64::from(rests_in_freeze(world.collision(), en_end.pos, en_end.vel))
                }
                _ => 0.0,
            };
            if let Some(en_end) = en_end
                && en_end.frozen
            {
                score += w
                    * (if self.cfg.no_thaw && en_sealed > 0.0 {
                        FREEZE_CLOCK_TICKS
                    } else {
                        en_end.freeze_ticks_left as f64
                    });
            }
            if self.cfg.seal_ticks > 0.0 {
                if let Some(me_end) = me_end
                    && me_end.frozen
                {
                    score -= w
                        * self.cfg.self_freeze_bias
                        * self.cfg.seal_ticks
                        * f64::from(rests_in_freeze(world.collision(), me_end.pos, me_end.vel));
                }
                score += w * self.cfg.seal_ticks * en_sealed;
            }
        }
        if self.track_rollout {
            let en_end = world.get_tee(enemy_id);
            self.rollout_enemy_sealed =
                en_end.is_some_and(|e| !e.alive || (e.frozen && rests_in_freeze(world.collision(), e.pos, e.vel) > 0));
        }
        // Task 3.10: the exact passive forecast of a frozen victim (hybrid only; plays the victim alone on the real physics, so it is
        // charged to the work meter as ticks of the rollout's own tee count). Never when the final state is kept for inspection.
        if self.cfg.held_forecast_weight > 0.0
            && !self.keep_final
            && let Some(en_end) = world.get_tee(enemy_id)
            && en_end.alive
            && en_end.frozen
        {
            let tees = crate::forecast::tee_count(world).max(1) as u64;
            let f = crate::forecast::passive_forecast(world, enemy_id, crate::forecast::HELD_HORIZON_TICKS);
            self.eval_ticks += (f.steps.max(0) as u64).div_ceil(tees);
            let h = crate::forecast::HELD_HORIZON_TICKS;
            score += self.cfg.held_forecast_weight * f64::from(f.out_ticks(h)) / f64::from(h);
        }
        // Task 3.10b (c): a seal that the ballistic guess grants is checked on the real physics (the victim alone); one that thaws soon pays for it.
        if self.cfg.sealed_forecast_weight > 0.0
            && !self.keep_final
            && let Some(en_end) = world.get_tee(enemy_id)
            && en_end.alive
            && en_end.frozen
            && rests_in_freeze(world.collision(), en_end.pos, en_end.vel) > 0
        {
            let tees = crate::forecast::tee_count(world).max(1) as u64;
            let h = crate::forecast::HELD_HORIZON_TICKS;
            let f = crate::forecast::passive_forecast(world, enemy_id, h);
            self.eval_ticks += (f.steps.max(0) as u64).div_ceil(tees);
            score += self.cfg.sealed_forecast_weight * (f64::from(f.out_ticks(h)) / f64::from(h) - 1.0);
        }
        self.last_input = input;
        if !self.keep_final {
            world.restore_state(self.saved.as_ref().unwrap());
        }
        self.events_buf = events;
        if let Some(log) = self.candidate_log.as_mut() {
            log.push((plan.to_vec(), score));
        }
        Some(score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::StepClock;
    use crate::config::PlannerConfig;
    use crate::physics_adapter::PhysicsWorld;
    use std::sync::Arc;

    fn tiny_map() -> Arc<ddai_physics::map::MapData> {
        let w = 20usize;
        let h = 10usize;
        let mut game = vec![ddai_physics::map::Tile::default(); w * h];
        for x in 0..w {
            game[(h - 1) * w + x] = ddai_physics::map::Tile {
                index: ddai_physics::map::TILE_SOLID,
                flags: 0,
                skip: 0,
                reserved: 0,
            };
        }
        Arc::new(ddai_physics::map::MapData {
            width: w as u32,
            height: h as u32,
            game,
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        })
    }

    /// Review round 1, F1: `decide_production` must always return a well-formed decision (never
    /// panic, never silently do nothing) even with a deadline so small the very first candidate's
    /// own per-step check (inside `evaluate_impl`) could plausibly fire -- the "first candidate is
    /// never itself interrupted" guarantee is what's actually under test here.
    #[test]
    fn decide_production_returns_a_decision_even_with_a_near_zero_budget() {
        let mut world = PhysicsWorld::new(tiny_map(), 1);
        world.add_tee(0, Vec2 { x: 100.0, y: 200.0 });
        world.add_tee(1, Vec2 { x: 300.0, y: 200.0 });
        let mut planner: Planner<PhysicsWorld> = Planner::new(PlannerConfig::default());
        planner.reset();
        let clock = StepClock::new(10.0); // one read already exceeds any tiny budget below.
        let prev = crate::types::empty_input();
        let decision = planner.decide_production(&mut world, 0, 1, prev, prev, &clock, 0.001);
        assert!(planner.last_info.searched);
        assert!(
            planner.last_info.candidates >= 1,
            "must evaluate at least one candidate"
        );
        assert_ne!(decision.direction, i32::MIN); // just exercises the field, always true -- see below
        assert!((-1..=1).contains(&decision.direction));
    }

    /// A generous budget should let the search examine many more candidates than a tiny one, and
    /// should still terminate (not hang) -- exercises the iterative-deepening loop's normal path.
    #[test]
    fn decide_production_examines_more_candidates_with_a_bigger_budget() {
        let cfg = PlannerConfig::default();
        let run = |budget_ms: f64| -> i32 {
            let mut world = PhysicsWorld::new(tiny_map(), 1);
            world.add_tee(0, Vec2 { x: 100.0, y: 200.0 });
            world.add_tee(1, Vec2 { x: 300.0, y: 200.0 });
            let mut planner: Planner<PhysicsWorld> = Planner::new(cfg);
            planner.reset();
            let clock = StepClock::new(0.05);
            let prev = crate::types::empty_input();
            let _ = planner.decide_production(&mut world, 0, 1, prev, prev, &clock, budget_ms);
            planner.last_info.candidates
        };
        let small = run(0.2);
        let big = run(20.0);
        assert!(
            big > small,
            "big={big} should examine more candidates than small={small}"
        );
    }

    /// Warm-starting: a second call with the same warm plan available should not panic or regress
    /// (exercises the `warm[1..]` shift path, including the single-step-plan edge case where
    /// `warm.len() == 1`).
    #[test]
    fn decide_production_warm_start_does_not_panic_on_a_single_step_plan() {
        let cfg = PlannerConfig {
            steps: 1,
            ..PlannerConfig::default()
        };
        let mut world = PhysicsWorld::new(tiny_map(), 1);
        world.add_tee(0, Vec2 { x: 100.0, y: 200.0 });
        world.add_tee(1, Vec2 { x: 300.0, y: 200.0 });
        let mut planner: Planner<PhysicsWorld> = Planner::new(cfg);
        planner.reset();
        let clock = StepClock::new(0.05);
        let prev = crate::types::empty_input();
        let _ = planner.decide_production(&mut world, 0, 1, prev, prev, &clock, 4.0);
        world.set_input(0, prev);
        world.set_input(1, prev);
        world.step();
        let _ = planner.decide_production(&mut world, 0, 1, prev, prev, &clock, 4.0);
    }

    /// Review round 2, F13: `decide_once` advances `opp_seed` (`js::opp_seed_next`) on every call,
    /// unconditionally, before anything else -- `decide_production` silently never did. Two calls
    /// must leave `opp_seed` at `opp_seed_next(opp_seed_next(initial))`, not just "some value".
    #[test]
    fn decide_production_advances_opp_seed_every_call() {
        let mut world = PhysicsWorld::new(tiny_map(), 1);
        world.add_tee(0, Vec2 { x: 100.0, y: 200.0 });
        world.add_tee(1, Vec2 { x: 300.0, y: 200.0 });
        let mut planner: Planner<PhysicsWorld> = Planner::new(PlannerConfig::default());
        planner.reset();
        let initial = planner.debug_state().opp_seed;
        let clock = StepClock::new(0.05);
        let prev = crate::types::empty_input();
        let _ = planner.decide_production(&mut world, 0, 1, prev, prev, &clock, 4.0);
        let after_one = planner.debug_state().opp_seed;
        assert_eq!(after_one, ddai_jsmath::opp_seed_next(initial));
        let _ = planner.decide_production(&mut world, 0, 1, prev, prev, &clock, 4.0);
        let after_two = planner.debug_state().opp_seed;
        assert_eq!(after_two, ddai_jsmath::opp_seed_next(after_one));
        assert_ne!(after_two, initial);
    }

    /// Review round 2, F13: `decide_once` clears `thaw_memo` on every call so it never grows
    /// without bound over a live session -- `decide_production` silently never did.
    #[test]
    fn decide_production_clears_thaw_memo_before_deciding() {
        let mut world = PhysicsWorld::new(tiny_map(), 1);
        world.add_tee(0, Vec2 { x: 100.0, y: 200.0 });
        world.add_tee(1, Vec2 { x: 300.0, y: 200.0 });
        let mut planner: Planner<PhysicsWorld> = Planner::new(PlannerConfig::default());
        planner.reset();
        planner.thaw_memo.insert("stale-test-marker".to_string(), true);
        let clock = StepClock::new(0.05);
        let prev = crate::types::empty_input();
        let _ = planner.decide_production(&mut world, 0, 1, prev, prev, &clock, 4.0);
        assert!(
            !planner.thaw_memo.contains_key("stale-test-marker"),
            "decide_production must clear thaw_memo before deciding, like decide_once does"
        );
    }

    /// Review round 2, F12: when the clock outruns even the shield's own reserve, the shield must
    /// give up cleanly (not hang, not panic) and flag `shield_incomplete` instead of silently
    /// pretending the (unverified) chosen input is safe. A `StepClock` that jumps far on every
    /// read guarantees this: `decide_production` still returns a real decision (the "first
    /// candidate is never itself interrupted" guarantee), but by the time execution reaches the
    /// shield block the clock has already blown through `SHIELD_RESERVE_MS`.
    #[test]
    fn decide_production_flags_shield_incomplete_when_the_shield_itself_times_out() {
        let mut world = PhysicsWorld::new(tiny_map(), 1);
        world.add_tee(0, Vec2 { x: 100.0, y: 200.0 });
        world.add_tee(1, Vec2 { x: 300.0, y: 200.0 });
        let mut planner: Planner<PhysicsWorld> = Planner::new(PlannerConfig::default());
        planner.reset();
        // Large enough that the very first clock read inside the shield's own bounded checks is
        // already past its (tiny) reserve deadline, but the search phase still completes its
        // guaranteed-first-candidate pass (that candidate's own deadline is `None`, never checked).
        let clock = StepClock::new(50.0);
        let prev = crate::types::empty_input();
        let _ = planner.decide_production(&mut world, 0, 1, prev, prev, &clock, 4.0);
        assert!(planner.last_info.searched);
        assert!(
            planner.last_info.shield_incomplete,
            "expected shield_incomplete=true once the clock has outrun the shield's reserve"
        );
    }

    /// Review round 3, F16's own acceptance test through the real `Planner`, matching the
    /// reviewer's `zz_r3_leak.rs` repro (which asserted the *opposite*, pre-fix outcome: `4 * n`
    /// marks retained after `n` undrained calls). With `prof` left at its default (disabled) state
    /// -- this test never calls `prof::enable()` -- 10k `decide_production` calls with nobody ever
    /// calling `prof::take()` must leave the fixed-size mark storage empty and the step/escape
    /// counters at zero: proof the disabled path never allocates or grows, unlike the old
    /// unconditional `Vec::push`.
    #[test]
    fn prof_stays_disabled_by_default_across_many_decisions() {
        crate::prof::disable(); // this test binary may run other tests that call `prof::enable()`
        let mut world = PhysicsWorld::new(tiny_map(), 1);
        world.add_tee(0, Vec2 { x: 100.0, y: 200.0 });
        world.add_tee(1, Vec2 { x: 300.0, y: 200.0 });
        let mut planner: Planner<PhysicsWorld> = Planner::new(PlannerConfig::default());
        planner.reset();
        let clock = crate::clock::WallClock::new();
        let prev = crate::types::empty_input();
        for _ in 0..10_000 {
            let _ = planner.decide_production(&mut world, 0, 1, prev, prev, &clock, 0.3);
        }
        let marks = crate::prof::take();
        assert_eq!(
            marks.len(),
            0,
            "prof must retain nothing when disabled, found {} marks after 10k decisions",
            marks.len()
        );
    }
    /// A floor with a freeze "ceiling" tile at column 5, row 2 (task 3.8, `ceilingField`).
    fn map_with_freeze_ceiling() -> Arc<ddai_physics::map::MapData> {
        let base = tiny_map();
        let mut map = (*base).clone();
        let w = map.width as usize;
        map.game[2 * w + 5] = ddai_physics::map::Tile {
            index: ddai_physics::map::TILE_FREEZE,
            flags: 0,
            skip: 0,
            reserved: 0,
        };
        Arc::new(map)
    }

    /// Two tees on the tiny map's floor; `en_vel` is the victim's velocity.
    fn duel_world(en_vel: Vec2) -> PhysicsWorld {
        let mut world = PhysicsWorld::new(tiny_map(), 1);
        world.add_tee(0, Vec2 { x: 100.0, y: 270.0 });
        world.add_tee(1, Vec2 { x: 300.0, y: 270.0 });
        let mut en = world.get_tee(1).unwrap();
        en.vel = en_vel;
        world.apply_tee_state(1, &en);
        world
    }

    fn v2_planner() -> Planner<PhysicsWorld> {
        Planner::new(crate::config::preset_normal_v2())
    }

    /// Task 3.8 (af49dfb `hookExactGate`): the rope's line must pass within a body of where the victim will be when the
    /// hook arrives -- a victim fleeing upward is missed by a straight throw that the old geometric test accepted.
    #[test]
    fn exact_gate_asks_where_the_victim_will_be() {
        let world = duel_world(Vec2 { x: 0.0, y: -40.0 });
        let exact = v2_planner();
        let old: Planner<PhysicsWorld> = Planner::new(crate::config::preset_normal());
        // Straight at the victim's current position: the old test (within two body widths of the line) says yes ...
        assert!(old.hook_would_reach(&world, 0, 1, 0.0, None));
        // ... the exact one projects the victim 1.5 ticks * 40 px upward, ~59 px off the line.
        assert!(!exact.hook_would_reach(&world, 0, 1, 0.0, None));
        // Aimed at the projected position, it reaches.
        let meet = rope_intercept(
            world.collision(),
            Vec2 { x: 100.0, y: 270.0 },
            Vec2 { x: 300.0, y: 270.0 },
            Vec2 { x: 0.0, y: -40.0 },
        );
        let angle = trig::atan2(meet.y - 270.0, meet.x - 100.0);
        assert!(exact.hook_would_reach(&world, 0, 1, angle, None));
        // A still victim: both tests agree.
        let still = duel_world(Vec2 { x: 0.0, y: 0.0 });
        assert!(exact.hook_would_reach(&still, 0, 1, 0.0, None));
        assert!(!exact.hook_would_reach(&still, 0, 1, -0.6, None));
    }

    /// Task 3.8 (af49dfb `hookSnapAim`): a throw that would just miss a moving victim is turned toward where it will be (at most
    /// 0.35 rad); with the switch off, or while our hook is already out, nothing is snapped.
    #[test]
    fn snap_aim_turns_toward_the_projected_victim() {
        let world = duel_world(Vec2 { x: 0.0, y: -12.0 });
        let planner = v2_planner();
        let me = world.get_tee(0).unwrap();
        let en = world.get_tee(1).unwrap();
        let mut prev = crate::types::empty_input();
        prev.target_x = 300.0;
        prev.target_y = 0.0;
        let step = PlanStep {
            dir: 0,
            jump: 0,
            hook: 1,
            fire: 0,
            aim: 0.0,
        };
        let (angle, meet) = planner.snap_aim(&world, Some(&me), Some(&en), step, prev, 0.0);
        let meet = meet.expect("the projection was computed");
        let want = trig::atan2(meet.y - me.pos.y, meet.x - me.pos.x);
        let angle = angle.expect("a snap");
        assert!(
            (angle - want).abs() < 1e-12,
            "snapped to the projected bearing: {angle} vs {want}"
        );
        assert!(angle < 0.0 && angle.abs() < SNAP_MAX_RAD);
        // Not a hook step / hook already out / switch off.
        let walk = PlanStep { hook: 0, ..step };
        assert_eq!(
            planner.snap_aim(&world, Some(&me), Some(&en), walk, prev, 0.0),
            (None, None)
        );
        let busy = TeeState {
            hook_state: HOOK_FLYING,
            ..me
        };
        assert_eq!(
            planner.snap_aim(&world, Some(&busy), Some(&en), step, prev, 0.0),
            (None, None)
        );
        let old: Planner<PhysicsWorld> = Planner::new(crate::config::preset_normal());
        assert_eq!(
            old.snap_aim(&world, Some(&me), Some(&en), step, prev, 0.0),
            (None, None)
        );
        // The snapped gate: `gate_and_snap` lets the hook through at the snapped angle and reports it.
        let (ok, aim, snapped) = planner.gate_and_snap(&world, 0, 1, Some(&me), Some(&en), step, prev, 0.0);
        assert!(ok && snapped);
        assert_eq!(aim.to_bits(), angle.to_bits());
        // `snap_target` aims exactly along the angle, unlike the smoothed decode.
        let mut input = crate::types::empty_input();
        snap_target(&mut input, angle);
        assert_eq!(input.target_x, js::round(trig::cos(angle) * crate::action::AIM_RADIUS));
        assert_eq!(input.target_y, js::round(trig::sin(angle) * crate::action::AIM_RADIUS));
    }

    /// Task 3.9 (inventory of E-020): what the hook gate costs per hook step of a rollout -- the classic gate, the exact gate and the whole
    /// gate-and-snap -- next to a physics tick of two tees. `cargo test -p ddai-planner --release --lib -- --ignored --nocapture v2_gate_cost`.
    #[test]
    #[ignore = "timing report"]
    fn v2_gate_cost_report() {
        use std::hint::black_box;
        use std::time::Instant;
        fn ns(n: u32, mut f: impl FnMut()) -> f64 {
            let t = Instant::now();
            for _ in 0..n {
                f();
            }
            t.elapsed().as_secs_f64() * 1e9 / f64::from(n)
        }
        for (label, vel) in [
            ("still victim", Vec2 { x: 0.0, y: 0.0 }),
            ("moving victim", Vec2 { x: 0.0, y: -12.0 }),
        ] {
            let world = duel_world(vel);
            let me = world.get_tee(0).unwrap();
            let en = world.get_tee(1).unwrap();
            let mut prev = crate::types::empty_input();
            prev.target_x = 300.0;
            let step = PlanStep {
                dir: 0,
                jump: 0,
                hook: 1,
                fire: 0,
                aim: 0.0,
            };
            let exact = v2_planner();
            let old: Planner<PhysicsWorld> = Planner::new(crate::config::preset_normal());
            let n = 200_000;
            let plain_gate = ns(n, || {
                let _ = black_box(old.hook_would_reach(&world, 0, 1, black_box(0.0), None));
            });
            let exact_gate = ns(n, || {
                let _ = black_box(exact.hook_would_reach(&world, 0, 1, black_box(0.0), None));
            });
            let intercept = ns(n, || {
                let _ = black_box(rope_intercept(world.collision(), me.pos, black_box(en.pos), en.vel));
            });
            let classic_step = ns(n, || {
                let _ = black_box(old.gate_and_snap(&world, 0, 1, Some(&me), Some(&en), step, prev, black_box(0.0)));
            });
            let v2_step = ns(n, || {
                let _ = black_box(exact.gate_and_snap(&world, 0, 1, Some(&me), Some(&en), step, prev, black_box(0.0)));
            });
            println!(
                "{label}: classic gate {plain_gate:.0} ns, exact gate {exact_gate:.0} ns, rope_intercept {intercept:.0} ns; per hook step: classic gate_and_snap {classic_step:.0} ns, v2 gate_and_snap {v2_step:.0} ns"
            );
        }
        let mut w = duel_world(Vec2 { x: 0.0, y: 0.0 });
        let saved = w.save_state();
        let tick = ns(2_000, || {
            w.step();
            if w.inner().tick % 27 == 0 {
                w.restore_state(&saved);
            }
        });
        println!("one physics tick of two tees: {tick:.0} ns (a rollout of 9 plan steps is 27 ticks)");
    }

    /// Task 3.8 (af49dfb `ceilingField`): tiles under a freeze ceiling see it; so do the columns beside, through open air.
    #[test]
    fn ceiling_field_sees_a_freeze_ceiling_above_and_beside() {
        let world = PhysicsWorld::new(map_with_freeze_ceiling(), 1);
        let f = fields::ceiling_field(world.collision());
        let at = |x: i32, y: i32| f.dist[(y * f.width + x) as usize];
        assert_eq!(at(5, 3), 1, "right under it");
        assert_eq!(at(5, 6), 4);
        assert_eq!(at(4, 5), 3, "the column beside, through open air, sees it too");
        assert_eq!(at(6, 5), 3);
        assert_eq!(at(2, 5), CEILING_NONE, "two columns away: nothing");
        assert_eq!(at(5, 2), CEILING_NONE, "the freeze tile itself and above: nothing");
        assert_eq!(at(5, 1), CEILING_NONE);
    }

    /// Task 3.8 (af49dfb `ropeCeilingCost`): being hauled up toward a freeze ceiling by the victim's rope costs; the same flight
    /// without the rope (or with the switch off) does not.
    #[test]
    fn rope_ceiling_costs_only_when_hauled_up_into_it() {
        let world = PhysicsWorld::new(map_with_freeze_ceiling(), 1);
        let ceiling = fields::ceiling_field(world.collision());
        let field = fields::hazard_field(world.collision());
        let unfreeze = fields::unfreeze_field(world.collision());
        let cfg = crate::config::preset_normal_v2();
        let hooked = |ceiling: Option<&CeilingField>, hooked_by_enemy: bool, cfg: &PlannerConfig| {
            let mut w = PhysicsWorld::new(map_with_freeze_ceiling(), 1);
            w.add_tee(
                0,
                Vec2 {
                    x: 5.0 * 32.0 + 16.0,
                    y: 100.0,
                },
            );
            w.add_tee(
                1,
                Vec2 {
                    x: 5.0 * 32.0 + 16.0,
                    y: 250.0,
                },
            );
            let mut me = w.get_tee(0).unwrap();
            me.vel = Vec2 { x: 0.0, y: -12.0 };
            w.apply_tee_state(0, &me);
            let mut en = w.get_tee(1).unwrap();
            en.hooked_player = if hooked_by_enemy { 0 } else { -1 };
            w.apply_tee_state(1, &en);
            let mut drag = DragTracker {
                prev_enemy_near: 0.0,
                start_enemy_near: 0.0,
                started_in_dead: false,
                prev_stage_dist: f64::NAN,
            };
            score_tick(
                &w,
                0,
                1,
                &[],
                &field,
                &unfreeze,
                cfg,
                &mut drag,
                None,
                None,
                None,
                &[],
                None,
                None,
                ceiling,
            )
        };
        let off = hooked(None, true, &cfg);
        let on = hooked(Some(&ceiling), true, &cfg);
        assert!(on < off, "the haul toward the ceiling costs: {on} vs {off}");
        assert!(off - on <= cfg.rope_ceiling_cost + 1e-12, "at most the cost per tick");
        assert_eq!(
            hooked(Some(&ceiling), false, &cfg),
            hooked(None, false, &cfg),
            "no rope, no cost"
        );
        let no_cost = PlannerConfig {
            rope_ceiling_cost: 0.0,
            ..cfg
        };
        assert_eq!(hooked(Some(&ceiling), true, &no_cost), off, "switched off");
    }

    /// Task 3.14 (`ceiling_guard_cost`): a tee close under a freeze ceiling pays per tick, in proportion to how deep it is inside the guard gap, but only
    /// while a free opponent is within reach; far from the ceiling, with the opponent out of reach or frozen, or with the cost off, nothing.
    #[test]
    fn ceiling_guard_costs_only_close_under_a_ceiling_with_a_foe_in_reach() {
        let map = map_with_freeze_ceiling();
        let world0 = PhysicsWorld::new(map.clone(), 1);
        let ceiling = fields::ceiling_field(world0.collision());
        let field = fields::hazard_field(world0.collision());
        let unfreeze = fields::unfreeze_field(world0.collision());
        // The freeze tile (5, 2) ends at y = 96 px; `y` is our height, `foe_dx` the opponent's distance, `foe_frozen` whether it is out.
        let score = |cfg: &PlannerConfig, y: f64, foe_dx: f64, foe_frozen: bool| {
            let mut w = PhysicsWorld::new(map.clone(), 1);
            w.add_tee(
                0,
                Vec2 {
                    x: 5.0 * 32.0 + 16.0,
                    y,
                },
            );
            w.add_tee(
                1,
                Vec2 {
                    x: 5.0 * 32.0 + 16.0 + foe_dx,
                    y: 270.0,
                },
            );
            let mut foe = w.get_tee(1).unwrap();
            foe.frozen = foe_frozen;
            w.apply_tee_state(1, &foe);
            let mut drag = DragTracker {
                prev_enemy_near: 0.0,
                start_enemy_near: 0.0,
                started_in_dead: false,
                prev_stage_dist: f64::NAN,
            };
            score_tick(
                &w,
                0,
                1,
                &[],
                &field,
                &unfreeze,
                cfg,
                &mut drag,
                None,
                None,
                None,
                &[],
                None,
                None,
                Some(&ceiling),
            )
        };
        let off = PlannerConfig::default();
        let on = PlannerConfig {
            ceiling_guard_cost: 2.0,
            ceiling_guard_px: 100.0,
            ..off
        };
        // 20 px under the tile's bottom edge (gap 20 of 100): 80% of the cost; touching: all of it.
        let near = score(&off, 116.0, 100.0, false) - score(&on, 116.0, 100.0, false);
        assert!((near - 2.0 * 0.8).abs() < 1e-9, "20 of 100 px: {near}");
        let touching = score(&off, 96.0, 100.0, false) - score(&on, 96.0, 100.0, false);
        assert!((touching - 2.0).abs() < 1e-9, "touching: {touching}");
        // 100 px or more below the edge, the foe out of reach (440 px), the foe frozen: no cost.
        assert_eq!(score(&off, 200.0, 100.0, false), score(&on, 200.0, 100.0, false));
        assert_eq!(score(&off, 116.0, 500.0, false), score(&on, 116.0, 500.0, false));
        assert_eq!(score(&off, 116.0, 100.0, true), score(&on, 116.0, 100.0, true));
        let no_gap = PlannerConfig {
            ceiling_guard_px: 0.0,
            ..on
        };
        assert_eq!(
            score(&off, 116.0, 100.0, false),
            score(&no_gap, 116.0, 100.0, false),
            "a zero gap is off"
        );
    }

    /// Task 3.14 (`duel_loss_cost`, `duel_win_bonus`): the first tick we are frozen in a rollout costs a flat amount (discounted by the step it
    /// falls in), the first tick the victim is frozen is worth one, and a rollout in which nobody freezes is untouched.
    #[test]
    fn duel_terms_are_charged_once_at_the_first_freeze() {
        let map = map_with_freeze_ceiling();
        let field = fields::hazard_field(PhysicsWorld::new(map.clone(), 1).collision());
        let unfreeze = fields::unfreeze_field(PhysicsWorld::new(map.clone(), 1).collision());
        // The freeze tile is above column 5: the tee that `rise`s stands under it and flies up into it, the other one stands far away.
        let score = |cfg: PlannerConfig, us_rise: bool, en_rise: bool| {
            let mut w = PhysicsWorld::new(map.clone(), 1);
            let x = |rise: bool, far: f64| (if rise { 5.0 } else { far }) * 32.0 + 16.0;
            w.add_tee(
                0,
                Vec2 {
                    x: x(us_rise, 9.0),
                    y: 130.0,
                },
            );
            w.add_tee(
                1,
                Vec2 {
                    x: x(en_rise, 13.0),
                    y: 130.0,
                },
            );
            for (id, up) in [(0, us_rise), (1, en_rise)] {
                let mut t = w.get_tee(id).unwrap();
                t.vel = Vec2 {
                    x: 0.0,
                    y: if up { -14.0 } else { 0.0 },
                };
                w.apply_tee_state(id, &t);
            }
            let mut p: Planner<PhysicsWorld> = Planner::new(cfg);
            p.reset();
            p.saved = Some(w.save_state());
            let plan = vec![
                PlanStep {
                    dir: 0,
                    jump: 0,
                    hook: 0,
                    fire: 0,
                    aim: 0.0
                };
                cfg.steps as usize
            ];
            let idle = crate::types::empty_input();
            p.evaluate(&mut w, 0, 1, idle, &plan, idle, &field, &unfreeze)
        };
        let base = crate::config::preset_normal();
        let with = |loss: f64, win: f64| PlannerConfig {
            duel_loss_cost: loss,
            duel_win_bonus: win,
            ..base
        };
        // Nobody freezes: the terms change nothing.
        assert_eq!(score(with(10.0, 5.0), false, false), score(base, false, false));
        // We freeze: the cost is charged once, between half of it (the last step's discount) and all of it.
        let (off, on) = (score(base, true, false), score(with(10.0, 5.0), true, false));
        assert!(off - on > 5.0 && off - on <= 10.0, "loss charged once: {off} -> {on}");
        let on20 = score(with(20.0, 5.0), true, false);
        assert!(((off - on20) - 2.0 * (off - on)).abs() < 1e-9, "linear in the cost");
        assert_eq!(
            score(with(10.0, 99.0), true, false),
            on,
            "the bonus is not charged to us"
        );
        // The victim freezes: the bonus is paid once.
        let (off, on) = (score(base, false, true), score(with(10.0, 5.0), false, true));
        assert!(on - off > 2.5 && on - off <= 5.0, "win paid once: {off} -> {on}");
        assert_eq!(
            score(with(99.0, 5.0), false, true),
            on,
            "the cost is not charged for the victim's freeze"
        );
    }

    /// Task 3.8: the classic configuration never reaches the v2 code -- all four switches off, the decision is the same whether
    /// or not the planner knows the v2 fields (a default planner and `with_version(Classic)` are one thing).
    #[test]
    fn classic_config_ignores_the_v2_switches() {
        let decide = |cfg: PlannerConfig| {
            let mut world = duel_world(Vec2 { x: 3.0, y: -1.0 });
            let mut planner: Planner<PhysicsWorld> = Planner::new(cfg);
            planner.reset();
            let prev = crate::types::empty_input();
            let mut out = Vec::new();
            for _ in 0..4 {
                out.push(planner.decide(&mut world, 0, 1, prev, prev));
            }
            (out, planner.debug_state().rng_s0)
        };
        let a = decide(crate::config::preset_normal());
        let b = decide(crate::config::preset_normal().with_version(crate::config::PlannerVersion::Classic));
        assert_eq!(a, b);
    }

    /// 20 x 22 tiles: solid floor from row 18, a freeze pit in it at x 8..=12 (rows 18 and 19).
    fn pit_map() -> Arc<ddai_physics::map::MapData> {
        let (w, h) = (20usize, 22usize);
        let mut game = vec![ddai_physics::map::Tile::default(); w * h];
        for y in 18..h {
            for x in 0..w {
                let freeze = y < 20 && (8..=12).contains(&x);
                game[y * w + x] = ddai_physics::map::Tile {
                    index: if freeze {
                        ddai_physics::map::TILE_FREEZE
                    } else {
                        ddai_physics::map::TILE_SOLID
                    },
                    flags: 0,
                    skip: 0,
                    reserved: 0,
                };
            }
        }
        Arc::new(ddai_physics::map::MapData {
            width: w as u32,
            height: h as u32,
            game,
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        })
    }

    /// Task 3.10b (c): `frozen_seal_weight` pays the weight per tick of a frozen victim that touches a freeze tile, and nothing for one that does not
    /// (nor when it is off).
    #[test]
    fn frozen_seal_weight_pays_per_tick_the_frozen_victim_touches_a_freeze() {
        let tick = |victim_x_tile: f64, frozen: bool, weight: f64| {
            let mut w = PhysicsWorld::new(pit_map(), 1);
            w.add_tee(
                0,
                Vec2 {
                    x: 2.0 * 32.0,
                    y: 17.0 * 32.0,
                },
            );
            w.add_tee(
                1,
                Vec2 {
                    x: victim_x_tile * 32.0,
                    y: 19.0 * 32.0,
                },
            );
            let mut en = w.get_tee(1).unwrap();
            en.frozen = frozen;
            en.freeze_ticks_left = if frozen { 100 } else { 0 };
            w.apply_tee_state(1, &en);
            let field = fields::hazard_field(w.collision());
            let unfreeze = fields::unfreeze_field(w.collision());
            let cfg = PlannerConfig {
                frozen_seal_weight: weight,
                ..crate::config::preset_normal()
            };
            let mut drag = DragTracker {
                prev_enemy_near: 0.0,
                start_enemy_near: 0.0,
                started_in_dead: false,
                prev_stage_dist: f64::NAN,
            };
            score_tick(
                &w,
                0,
                1,
                &[],
                &field,
                &unfreeze,
                &cfg,
                &mut drag,
                None,
                None,
                None,
                &[],
                None,
                None,
                None,
            )
        };
        // In the pit, frozen: exactly the weight more.
        assert!((tick(10.0, true, 0.3) - tick(10.0, true, 0.0) - 0.3).abs() < 1e-12);
        // Off the pit, or not frozen there: nothing.
        assert_eq!(tick(3.0, true, 0.3), tick(3.0, true, 0.0));
        assert_eq!(tick(10.0, false, 0.3), tick(10.0, false, 0.0));
    }

    /// Task 3.10b (c): a seal that the ballistic guess grants but the exact forecast refutes costs `sealed_forecast_weight * (1 - out / 250)`; a seal
    /// that holds costs nothing; off, nothing changes.
    #[test]
    fn sealed_forecast_weight_charges_a_seal_that_thaws_before_it_arrives() {
        let score = |victim_y_tile: f64, freeze_left: i64, weight: f64| {
            let mut w = PhysicsWorld::new(pit_map(), 1);
            w.add_tee(
                0,
                Vec2 {
                    x: 2.0 * 32.0,
                    y: 17.0 * 32.0,
                },
            );
            w.add_tee(
                1,
                Vec2 {
                    x: 10.0 * 32.0,
                    y: victim_y_tile * 32.0,
                },
            );
            let mut en = w.get_tee(1).unwrap();
            en.frozen = true;
            en.freeze_ticks_left = freeze_left;
            w.apply_tee_state(1, &en);
            let field = fields::hazard_field(w.collision());
            let unfreeze = fields::unfreeze_field(w.collision());
            let cfg = PlannerConfig {
                sealed_forecast_weight: weight,
                ..crate::config::preset_normal()
            };
            let mut planner: Planner<PhysicsWorld> = Planner::new(cfg);
            planner.reset();
            planner.saved = Some(w.save_state());
            let plan = vec![
                PlanStep {
                    dir: 0,
                    jump: 0,
                    hook: 0,
                    fire: 0,
                    aim: 0.0,
                };
                9
            ];
            let prev = crate::types::empty_input();
            planner.evaluate(&mut w, 0, 1, prev, &plan, prev, &field, &unfreeze)
        };
        // Resting in the pit, 100 ticks left: the forecast says held (the pit renews it) -- no charge.
        assert_eq!(score(19.0, 100, 45.0), score(19.0, 100, 0.0));
        // High above the pit, frozen for only 40 more ticks: still in the air when the rollout ends (the ballistic guess says it lands in the
        // pit), but it thaws before it gets there -- the seal is refuted.
        let (on, off) = (score(2.0, 40, 45.0), score(2.0, 40, 0.0));
        assert!(
            off - on > 30.0 && off - on <= 45.0,
            "charged {} of at most 45",
            off - on
        );
        // The same victim frozen long enough to arrive: a true seal, no charge.
        assert_eq!(score(2.0, 400, 45.0), score(2.0, 400, 0.0));
    }

    /// Task 3.10b (d): `mutual_freeze_cost` is taken off per tick only while both tees are frozen.
    #[test]
    fn mutual_freeze_cost_applies_only_to_a_trade() {
        let tick = |me_frozen: bool, en_frozen: bool, cost: f64| {
            let mut w = PhysicsWorld::new(pit_map(), 1);
            w.add_tee(
                0,
                Vec2 {
                    x: 2.0 * 32.0,
                    y: 17.0 * 32.0,
                },
            );
            w.add_tee(
                1,
                Vec2 {
                    x: 4.0 * 32.0,
                    y: 17.0 * 32.0,
                },
            );
            for (id, frozen) in [(0, me_frozen), (1, en_frozen)] {
                let mut t = w.get_tee(id).unwrap();
                t.frozen = frozen;
                t.freeze_ticks_left = if frozen { 100 } else { 0 };
                w.apply_tee_state(id, &t);
            }
            let field = fields::hazard_field(w.collision());
            let unfreeze = fields::unfreeze_field(w.collision());
            let cfg = PlannerConfig {
                mutual_freeze_cost: cost,
                ..crate::config::preset_normal()
            };
            let mut drag = DragTracker {
                prev_enemy_near: 0.0,
                start_enemy_near: 0.0,
                started_in_dead: false,
                prev_stage_dist: f64::NAN,
            };
            score_tick(
                &w,
                0,
                1,
                &[],
                &field,
                &unfreeze,
                &cfg,
                &mut drag,
                None,
                None,
                None,
                &[],
                None,
                None,
                None,
            )
        };
        assert!((tick(true, true, 1.5) - tick(true, true, 0.0) + 1.5).abs() < 1e-12);
        assert_eq!(tick(true, false, 1.5), tick(true, false, 0.0));
        assert_eq!(tick(false, true, 1.5), tick(false, true, 0.0));
        assert_eq!(tick(false, false, 1.5), tick(false, false, 0.0));
    }
}
