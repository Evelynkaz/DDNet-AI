//! `src/bot/navigate.ts`: the goto/follow navigator. It walks along the BFS field of the goal, falls
//! back to a planned [`RouteRunner`] route when walking stalls (jumps, ropes, a `/kill` respawn, planned
//! freeze crossings), swings through the Copy Love Box freeze tubes with a [`SwingCrosser`], probes
//! teleporters, and gives up with a reason when nothing works. A line-by-line port; its decisions are
//! compared tick by tick with the TS navigator (`tests/parity_nav.rs`, `Navtrace` lines).

use ddai_jsmath as js;
use ddai_planner::fields::{HazardField, travel_field};
use ddai_planner::plan_world::{PlanCollision, PlanWorld};
use ddai_planner::types::{PlayerInput, TeeState, empty_input};
use ddai_planner::vmath::{Vec2, vdistance};
use std::collections::{HashMap, HashSet};

use crate::TILE_PX;
use crate::crossing::{CrossPhase, CrossSmart, Crossing, SwingCrosser, in_any_box};
use crate::route::{MoveKind, RouteOpts, Router};
use crate::runner::{RouteRunner, RunnerState};

const UNREACHABLE: i32 = 0x3fff_ffff;
const LOOKAHEAD_TILES: usize = 4;
const CENTRE_PX: f64 = 8.0;
pub const GROUND_JUMP_RISE_PX: f64 = 182.0;
pub const DOUBLE_JUMP_RISE_PX: f64 = 330.0;
const RISING_VEL: f64 = -0.1;
const STALL_TICKS: i64 = 150;
const PROBE_TICKS: i64 = 100;
const TELEPORT_JUMP_PX: f64 = 96.0;
const MAX_HONEST_PX_PER_TICK: f64 = 12.0;
const MAX_TELE_GOALS: usize = 8;
const MAX_ROUTE_REPLANS: i32 = 3;
const MAX_ROUTE_DROPS: i32 = 3;
const MAX_ALTERNATIVE_ROUTES: i32 = 4;
const MAX_CROSS_TRIES: i32 = 4;
const PLANNED_FREEZE_STEPS: usize = 16;
const CLIMB_MIN_RISE_TILES: i32 = 3;
const CLIMB_ARC_RAYS: i32 = 13;
const CLIMB_RAY_STEP_PX: f64 = 12.0;
const CLIMB_GIVE_UP_TICKS: i64 = 120;
const CLIMB_ARRIVE_PX: f64 = 40.0;
const CLIMB_BRAKE_TICKS: f64 = 6.0;
/// `WALK_BRAKE_TICKS`: ticks a walking brake takes to bite.
pub const WALK_BRAKE_TICKS: f64 = 3.0;
/// `LAG_MARGIN_TICKS`: ticks of margin on top of the lag when braking in front of the goal.
pub const LAG_MARGIN_TICKS: f64 = 2.0;
const HOOK_LENGTH: f64 = 380.0;

/// `NavGoal`.
#[derive(Debug, Clone, PartialEq)]
pub struct NavGoal {
    pub tx: i32,
    pub ty: i32,
    pub label: String,
    /// `(type, number)` of a teleporter entrance goal.
    pub tele: Option<(i32, i32)>,
}

/// `NavPhase`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavPhase {
    Walking,
    Probing,
    Arrived,
    Blocked,
}

impl NavPhase {
    pub fn name(self) -> &'static str {
        match self {
            NavPhase::Walking => "walking",
            NavPhase::Probing => "probing",
            NavPhase::Arrived => "arrived",
            NavPhase::Blocked => "blocked",
        }
    }
}

/// The Navigator's options (`opts`).
#[derive(Debug, Clone)]
pub struct NavOpts {
    pub stall_ticks: i64,
    pub probe_ticks: i64,
    pub through_freeze: bool,
    pub crossings: Vec<Crossing>,
    /// **CHANGE against TS** (off by default, which is the exact TS behaviour the parity tests need):
    /// when a planned route ends within 2 tiles of the goal but still more than [`APPROACH_PX`] from
    /// its centre, walk the last stretch instead of reporting "arrived" (the 64 px of the arrival
    /// metric; up to [`MAX_APPROACH_TRIES`] times).
    pub finish_approach: bool,
}

/// The distance to the goal centre under which a finished route counts as arrived.
pub const APPROACH_PX: f64 = 40.0;
/// How often a finished route may be followed by a walk to the goal centre.
pub const MAX_APPROACH_TRIES: i32 = 2;

impl Default for NavOpts {
    fn default() -> Self {
        NavOpts {
            stall_ticks: STALL_TICKS,
            probe_ticks: PROBE_TICKS,
            through_freeze: true,
            crossings: Vec::new(),
            finish_approach: false,
        }
    }
}

/// What a [`Navigator`] needs from the world each step (borrowed from the caller, not owned, so one
/// map's grids and scratch are shared by every navigator).
pub struct NavCtx<'a, W: PlanWorld> {
    pub col: &'a W::Collision,
    pub router: &'a mut Router,
    /// A fresh private simulation world of the map (for the crossings' rollouts).
    pub make_sim: &'a mut dyn FnMut() -> W,
}

fn tile_of(px: f64) -> i32 {
    (px / f64::from(TILE_PX)).trunc() as i32
}

fn centre_of(tile: i32) -> f64 {
    f64::from(tile * TILE_PX) + f64::from(TILE_PX) / 2.0
}

fn dist_at(field: &HazardField, tx: i32, ty: i32) -> i32 {
    if tx < 0 || ty < 0 || tx >= field.width || ty >= field.height {
        return UNREACHABLE;
    }
    field.dist[(ty * field.width + tx) as usize]
}

fn field_to(col: &impl PlanCollision, tx: i32, ty: i32) -> HazardField {
    travel_field(col, centre_of(tx), centre_of(ty))
}

const NEIGHBOURS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

/// `traceRoute(field, tx, ty, maxSteps)`: the greedy descent of the BFS field from a tile, preferring to
/// keep direction, then sideways, then down.
pub fn trace_route(field: &HazardField, tx: i32, ty: i32, max_steps: usize) -> Vec<(i32, i32)> {
    let mut out = Vec::new();
    let (mut cx, mut cy) = (tx, ty);
    let mut d = dist_at(field, cx, cy);
    if d >= UNREACHABLE {
        return out;
    }
    let (mut last_dx, mut last_dy) = (0, 0);
    let mut step = 0;
    while step < max_steps && d > 0 {
        let (mut best_x, mut best_y, mut best_score) = (-1, -1, -1);
        for (dx, dy) in NEIGHBOURS {
            let nd = dist_at(field, cx + dx, cy + dy);
            if nd >= d {
                continue;
            }
            let score = (if dx == last_dx && dy == last_dy { 4 } else { 0 })
                + (if dy == 0 { 2 } else { 0 })
                + (if dy > 0 { 1 } else { 0 });
            if score > best_score {
                best_score = score;
                best_x = cx + dx;
                best_y = cy + dy;
            }
        }
        if best_score < 0 {
            break;
        }
        last_dx = best_x - cx;
        last_dy = best_y - cy;
        cx = best_x;
        cy = best_y;
        d = dist_at(field, cx, cy);
        out.push((cx, cy));
        step += 1;
    }
    out
}

/// `teleGoals(collision, fromX, fromY, limit)`: up to `limit` teleporter entrances reachable by walking,
/// one per `(type, number)`, nearest first.
pub fn tele_goals(col: &impl PlanCollision, from_x: f64, from_y: f64, limit: usize) -> Vec<NavGoal> {
    if !col.has_tele() {
        return Vec::new();
    }
    let field = travel_field(col, from_x, from_y);
    let mut best: HashMap<(i32, i32), (NavGoal, i32)> = HashMap::new();
    let mut order: Vec<(i32, i32)> = Vec::new();
    for i in 0..col.width() * col.height() {
        let (tx, ty) = (i % col.width(), i / col.width());
        let (t, num) = col.tele_at(centre_of(tx), centre_of(ty));
        if t == 0 {
            continue;
        }
        let d = dist_at(&field, tx, ty);
        if d >= UNREACHABLE {
            continue;
        }
        let key = (t, num);
        if let Some((_, prev)) = best.get(&key)
            && *prev <= d
        {
            continue;
        }
        if !best.contains_key(&key) {
            order.push(key);
        }
        best.insert(
            key,
            (
                NavGoal {
                    tx,
                    ty,
                    label: format!("teleporter {t}#{num} at ({tx},{ty})"),
                    tele: Some((t, num)),
                },
                d,
            ),
        );
    }
    let mut entries: Vec<(NavGoal, i32)> = order.into_iter().filter_map(|k| best.remove(&k)).collect();
    // `Array.prototype.sort` is stable: ties keep insertion order.
    entries.sort_by_key(|e| e.1);
    entries.into_iter().take(limit).map(|e| e.0).collect()
}

/// `MAX_TELE_GOALS`.
pub fn default_tele_goals(col: &impl PlanCollision, from_x: f64, from_y: f64) -> Vec<NavGoal> {
    tele_goals(col, from_x, from_y, MAX_TELE_GOALS)
}

/// `tileGoal(collision, tx, ty)`.
pub fn tile_goal(col: &impl PlanCollision, tx: i32, ty: i32) -> NavGoal {
    let (t, number) = col.tele_at(centre_of(tx), centre_of(ty));
    NavGoal {
        tx,
        ty,
        label: format!("({tx},{ty})"),
        tele: if t == 0 { None } else { Some((t, number)) },
    }
}

/// `class Navigator`.
pub struct Navigator<W: PlanWorld> {
    pub goals: Vec<NavGoal>,
    stall_ticks: i64,
    probe_ticks: i64,
    index: usize,
    field: Option<HazardField>,
    phase: NavPhase,
    reason: String,
    notes: Vec<String>,
    window_best: i32,
    window_ref: i32,
    window_start: i64,
    probe_until: i64,
    last_pos: Option<Vec2>,
    aim: f64,
    climb_anchor: Option<Vec2>,
    climb_start: i64,
    climb_best_y: f64,
    start_tick: i64,
    last_tick: i64,
    steps: i64,
    runner: Option<RouteRunner>,
    replans: i32,
    drops: i32,
    drop_give_up: Option<String>,
    avoid: HashSet<i32>,
    alternatives: i32,
    walk_routed: bool,
    kill_wanted: bool,
    /// Whether a planned route may use a respawn (kill) step; off, a route that needs one is unavailable ([`Navigator::set_allow_kill`]).
    allow_kill: bool,
    crossings: Vec<Crossing>,
    /// Wall-clock budget of the crossing search per step in ms (`crossBudgetMs`; 0 = none).
    pub cross_budget_ms: f64,
    /// `wallRoute`: the crossing takes route 2 (through the wall of the passage) where the tube has one.
    pub wall_route: bool,
    /// Task 3.12b (`--wb-smart`): the crossing's own options (all off: the TS behaviour).
    pub smart: CrossSmart,
    to_crossing: Option<usize>,
    crosser: Option<SwingCrosser<W>>,
    cross_tries: i32,
    crossing_reach: HashMap<usize, bool>,
    lag: i64,
    through_freeze: bool,
    finish_approach: bool,
    approach_tries: i32,
}

impl<W: PlanWorld> Navigator<W> {
    pub fn new(goals: Vec<NavGoal>, opts: NavOpts) -> Navigator<W> {
        let through_freeze = opts.through_freeze;
        let mut nav = Navigator {
            goals,
            stall_ticks: opts.stall_ticks,
            probe_ticks: opts.probe_ticks,
            index: 0,
            field: None,
            phase: NavPhase::Walking,
            reason: String::new(),
            notes: Vec::new(),
            window_best: i32::MAX,
            window_ref: i32::MAX,
            window_start: 0,
            probe_until: -1,
            last_pos: None,
            aim: 0.0,
            climb_anchor: None,
            climb_start: -1,
            climb_best_y: f64::INFINITY,
            start_tick: -1,
            last_tick: 0,
            steps: 0,
            runner: None,
            replans: 0,
            drops: 0,
            drop_give_up: None,
            avoid: HashSet::new(),
            alternatives: 0,
            walk_routed: false,
            kill_wanted: false,
            allow_kill: true,
            crossings: if through_freeze { opts.crossings } else { Vec::new() },
            cross_budget_ms: 0.0,
            wall_route: false,
            smart: CrossSmart::default(),
            to_crossing: None,
            crosser: None,
            cross_tries: 0,
            crossing_reach: HashMap::new(),
            lag: 0,
            through_freeze,
            finish_approach: opts.finish_approach,
            approach_tries: 0,
        };
        if nav.goals.is_empty() {
            nav.phase = NavPhase::Blocked;
            nav.reason = "nowhere to go".to_string();
        }
        nav
    }

    pub fn phase(&self) -> NavPhase {
        self.phase
    }
    pub fn done(&self) -> bool {
        matches!(self.phase, NavPhase::Arrived | NavPhase::Blocked)
    }
    pub fn outcome(&self) -> &str {
        &self.reason
    }
    pub fn goal(&self) -> Option<&NavGoal> {
        self.goals.get(self.index)
    }
    pub fn take_notes(&mut self) -> Vec<String> {
        std::mem::take(&mut self.notes)
    }
    pub fn take_kill(&mut self) -> bool {
        std::mem::take(&mut self.kill_wanted)
    }
    /// Allow (the default) or forbid routes with a respawn step. Forbidden: the planner never picks a route through a spawn (D-102,
    /// `--no-selfkill`); a route already running keeps its steps (the bot does not send the kill, the runner then replans).
    pub fn set_allow_kill(&mut self, allow: bool) {
        self.allow_kill = allow;
    }
    pub fn vetoed(&mut self) {
        if let Some(r) = &mut self.runner {
            r.vetoed();
        }
    }

    /// `progress(self)`: a sentence about where it is.
    pub fn progress(&self, me: Option<&TeeState>) -> String {
        let Some(goal) = self.goal() else {
            return if self.reason.is_empty() {
                "nowhere to go".to_string()
            } else {
                self.reason.clone()
            };
        };
        let left = match (me, &self.field) {
            (Some(m), Some(f)) => dist_at(f, tile_of(m.pos.x), tile_of(m.pos.y)),
            _ => -1,
        };
        let far = if left < 0 {
            String::new()
        } else if left >= UNREACHABLE {
            ", no route from where it is standing".to_string()
        } else {
            format!(", {left} tiles to go")
        };
        let which = if self.goals.len() > 1 {
            format!(" (candidate {} of {})", self.index + 1, self.goals.len())
        } else {
            String::new()
        };
        format!(
            "{} {}{}{}",
            if self.phase == NavPhase::Probing {
                "standing on"
            } else {
                "walking to"
            },
            goal.label,
            far,
            which
        )
    }

    /// `tilesLeft(self)`.
    pub fn tiles_left(&self, me: Option<&TeeState>) -> i32 {
        let (Some(m), Some(f)) = (me, &self.field) else {
            return -1;
        };
        let left = dist_at(f, tile_of(m.pos.x), tile_of(m.pos.y));
        if left >= UNREACHABLE { -1 } else { left }
    }

    fn note(&mut self, text: impl Into<String>) {
        self.notes.push(text.into());
    }

    fn finish(&mut self, phase: NavPhase, reason: String) {
        self.phase = phase;
        self.reason = reason.clone();
        self.note(reason);
    }

    fn next_goal(&mut self, why: String, tick: i64) {
        self.note(why.clone());
        self.index += 1;
        self.field = None;
        self.runner = None;
        self.replans = 0;
        self.drops = 0;
        self.drop_give_up = None;
        self.avoid = HashSet::new();
        self.alternatives = 0;
        self.walk_routed = false;
        self.to_crossing = None;
        self.crosser = None;
        self.cross_tries = 0;
        self.approach_tries = 0;
        self.crossing_reach = HashMap::new();
        self.window_best = i32::MAX;
        self.window_ref = i32::MAX;
        self.window_start = tick;
        self.probe_until = -1;
        if self.index >= self.goals.len() {
            self.phase = NavPhase::Blocked;
            self.reason = if self.goals.len() > 1 {
                format!("{why}; nothing else to try")
            } else {
                why
            };
            if self.goals.len() > 1 {
                self.note("no route to any of them");
            }
        }
    }

    /// `step(self, tick, others, lag)`.
    pub fn step(
        &mut self,
        ctx: &mut NavCtx<'_, W>,
        me: &TeeState,
        tick: i64,
        others: &[TeeState],
        lag: i64,
    ) -> PlayerInput {
        self.lag = lag;
        let since_last = if self.start_tick < 0 {
            1
        } else {
            (tick - self.last_tick).max(1)
        };
        self.last_tick = tick;
        self.steps += 1;
        if self.start_tick < 0 {
            self.start_tick = tick;
            self.window_start = tick;
        }
        if tick < self.window_start {
            self.window_start = tick;
        }
        if self.climb_start > tick {
            self.climb_start = tick;
        }
        if self.probe_until > tick + self.probe_ticks {
            self.probe_until = tick + self.probe_ticks;
        }
        let was = self.last_pos;
        let moved = was.map_or(0.0, |w| vdistance(w, me.pos));
        self.last_pos = Some(Vec2 {
            x: me.pos.x,
            y: me.pos.y,
        });
        if let Some(was) = was
            && moved >= js::max(TELEPORT_JUMP_PX, since_last as f64 * MAX_HONEST_PX_PER_TICK)
            && !self.done()
        {
            let goal = self.goal().cloned();
            let here = format!("tile ({},{})", tile_of(me.pos.x), tile_of(me.pos.y));
            let from_goal = goal.as_ref().map_or(f64::INFINITY, |g| {
                vdistance(
                    was,
                    Vec2 {
                        x: centre_of(g.tx),
                        y: centre_of(g.ty),
                    },
                )
            });
            if let Some(g) = &goal
                && let Some(t) = g.tele
                && from_goal <= f64::from(TILE_PX) * 1.5
            {
                self.finish(
                    NavPhase::Arrived,
                    format!("teleported to {here} -- type {} is an entrance", t.0),
                );
                return empty_input();
            }
            self.window_best = i32::MAX;
            self.window_ref = i32::MAX;
            self.window_start = tick;
            if self.runner.as_ref().is_some_and(RouteRunner::awaiting_kill) {
                self.runner.as_mut().expect("runner").respawned();
                self.note(format!("respawned at {here}"));
            } else if self.crosser.is_some() {
                self.crosser = None;
                self.field = None;
                self.note(format!(
                    "moved to {here} in the middle of the swing; starting over from here"
                ));
            } else if self.runner.as_ref().and_then(|r| r.current()).is_some_and(|s| s.tele) {
                self.note(format!("teleported to {here}"));
            } else if self.runner.is_some() {
                self.drop_route(&format!("moved to {here} off it"));
                self.note(format!(
                    "something moved us to {here} (not the doorway we were walking to); planning again from here"
                ));
            } else {
                self.note(format!(
                    "something moved us to {here} (not the doorway we were walking to); carrying on"
                ));
            }
        }
        if self.done() {
            return empty_input();
        }
        let Some(goal) = self.goal().cloned() else {
            self.finish(NavPhase::Blocked, "nowhere to go".to_string());
            return empty_input();
        };
        if let Some(give_up) = self.drop_give_up.clone() {
            self.next_goal(format!("route to {} broke off: {give_up}", goal.label), tick);
            return empty_input();
        }
        if let Some(crossed) = self.step_crossing(ctx, me, tick, &goal, others) {
            return crossed;
        }
        if self.field.is_none() && me.frozen {
            return empty_input();
        }
        if self.field.is_none() {
            let field = field_to(ctx.col, goal.tx, goal.ty);
            self.window_best = i32::MAX;
            self.window_ref = i32::MAX;
            self.window_start = tick;
            let here = dist_at(&field, tile_of(me.pos.x), tile_of(me.pos.y));
            self.field = Some(field);
            if here >= UNREACHABLE || self.walk_routed {
                let route = ctx.router.find_route(
                    (me.pos.x, me.pos.y),
                    (centre_of(goal.tx), centre_of(goal.ty)),
                    &RouteOpts {
                        near_tiles: 2,
                        allow_kill: self.allow_kill,
                        through_freeze: self.through_freeze,
                        avoid: Some(&self.avoid),
                        ..RouteOpts::default()
                    },
                );
                match route {
                    Some(r) if !r.steps.is_empty() => {
                        let hooks = r.steps.iter().filter(|s| s.kind == MoveKind::Hook).count();
                        let n = r.steps.len();
                        self.runner = Some(RouteRunner::new(
                            r.steps,
                            Some(&ctx.router.grid),
                            !self.crossings.is_empty(),
                        ));
                        self.note(format!(
                            "no walk to {}; going by route: {n} steps{}",
                            goal.label,
                            if hooks > 0 {
                                format!(", {hooks} on the rope")
                            } else {
                                String::new()
                            }
                        ));
                    }
                    _ => {
                        if self.start_crossing(ctx, me, &goal) {
                            return empty_input();
                        }
                        self.next_goal(format!("no route to {}: it is walled off from here", goal.label), tick);
                        return empty_input();
                    }
                }
            } else {
                self.note(format!("heading for {}, {here} tiles away", goal.label));
            }
        }
        if self.runner.is_some() {
            let state = self.runner.as_ref().expect("runner").state;
            if state == RunnerState::Running {
                let runner = self.runner.as_mut().expect("runner");
                let out = runner.step(me, tick, Some(others));
                if runner.take_kill() {
                    self.kill_wanted = true;
                }
                self.steps += 1;
                return out;
            }
            let runner = self.runner.take().expect("runner");
            if runner.state == RunnerState::Arrived {
                if self.to_crossing.is_some() {
                    self.field = None;
                    return empty_input();
                }
                if self.finish_approach
                    && goal.tele.is_none()
                    && self.approach_tries < MAX_APPROACH_TRIES
                    && vdistance(
                        me.pos,
                        Vec2 {
                            x: centre_of(goal.tx),
                            y: centre_of(goal.ty),
                        },
                    ) > APPROACH_PX
                {
                    self.approach_tries += 1;
                    self.walk_routed = false;
                    self.field = None;
                    self.note(format!("the route ended short of {}; walking the rest", goal.label));
                    return empty_input();
                }
                self.finish(NavPhase::Arrived, format!("walked the route to {}", goal.label));
                return empty_input();
            }
            if runner.state == RunnerState::Replan && self.replans < MAX_ROUTE_REPLANS {
                self.replans += 1;
                self.note(format!(
                    "route to {}: {}; planning again from here",
                    goal.label, runner.reason
                ));
                self.field = None;
                return empty_input();
            }
            if let Some(failed) = runner.failed_move()
                && self.alternatives < MAX_ALTERNATIVE_ROUTES
            {
                self.alternatives += 1;
                self.avoid.insert(failed);
                self.note(format!(
                    "route to {}: {}; looking for another way ({}/{})",
                    goal.label, runner.reason, self.alternatives, MAX_ALTERNATIVE_ROUTES
                ));
                self.field = None;
                return empty_input();
            }
            self.next_goal(format!("route to {} broke off: {}", goal.label, runner.reason), tick);
            return empty_input();
        }
        let field = self.field.clone().expect("field");
        let tx = tile_of(me.pos.x);
        let ty = tile_of(me.pos.y);
        let here = dist_at(&field, tx, ty);
        if tx == goal.tx && ty == goal.ty {
            if goal.tele.is_none() {
                self.finish(NavPhase::Arrived, format!("arrived at {}", goal.label));
                // Arrived at a run: brake against the hazard behind the goal, the lag counted.
                let dir = js::sign(me.vel.x);
                let brake_px = 24.0 + js::max(0.0, me.vel.x * dir) * (self.lag as f64 + WALK_BRAKE_TICKS);
                if dir != 0.0 && me.vel.x.abs() > 0.5 && hazard_within(ctx.col, me, dir as i32, brake_px) {
                    let mut brake = empty_input();
                    brake.direction = -(dir as i32);
                    return brake;
                }
                return empty_input();
            }
            if self.probe_until < 0 {
                self.probe_until = tick + self.probe_ticks;
                self.phase = NavPhase::Probing;
                self.note(format!("standing on {} to see whether it moves us", goal.label));
            }
        }
        if self.probe_until >= 0 && tick >= self.probe_until {
            self.phase = NavPhase::Walking;
            let dead = goal.label.clone();
            if self.index + 1 >= self.goals.len() {
                self.finish(
                    NavPhase::Arrived,
                    format!("reached {dead}; it did not move us, and there is no other doorway from here"),
                );
                return empty_input();
            }
            self.next_goal(format!("{dead} did not move us; trying the next one"), tick);
            return empty_input();
        }
        if here < self.window_best {
            self.window_best = here;
        }
        if self.probe_until < 0 && tick - self.window_start >= self.stall_ticks {
            if self.window_best >= self.window_ref && !self.walk_routed {
                self.walk_routed = true;
                let route = ctx.router.find_route(
                    (me.pos.x, me.pos.y),
                    (centre_of(goal.tx), centre_of(goal.ty)),
                    &RouteOpts {
                        near_tiles: 2,
                        allow_kill: self.allow_kill,
                        through_freeze: self.through_freeze,
                        avoid: Some(&self.avoid),
                        ..RouteOpts::default()
                    },
                );
                if let Some(r) = route
                    && !r.steps.is_empty()
                {
                    let hooks = r.steps.iter().filter(|s| s.kind == MoveKind::Hook).count();
                    let n = r.steps.len();
                    self.runner = Some(RouteRunner::new(
                        r.steps,
                        Some(&ctx.router.grid),
                        !self.crossings.is_empty(),
                    ));
                    self.note(format!(
                        "walking to {} stalled {}; going by route: {n} steps{}",
                        goal.label,
                        if here >= UNREACHABLE {
                            "off the flood".to_string()
                        } else {
                            format!("{here} tiles away")
                        },
                        if hooks > 0 {
                            format!(", {hooks} on the rope")
                        } else {
                            String::new()
                        }
                    ));
                    return empty_input();
                }
            }
            if self.window_best >= self.window_ref {
                self.next_goal(
                    format!(
                        "stuck {}: no closer in {}s -- the rest of that route needs more than walking",
                        if here >= UNREACHABLE {
                            "off the route".to_string()
                        } else {
                            format!("{here} tiles from {}", goal.label)
                        },
                        format_fixed0((tick - self.window_start) as f64 / 50.0)
                    ),
                    tick,
                );
                return empty_input();
            }
            self.window_ref = self.window_best;
            self.window_best = here;
            self.window_start = tick;
        }
        self.follow(ctx.col, me, &field, tx, ty, here, tick)
    }

    #[allow(clippy::too_many_arguments)]
    fn follow<C: PlanCollision>(
        &mut self,
        col: &C,
        me: &TeeState,
        field: &HazardField,
        tx: i32,
        ty: i32,
        here: i32,
        tick: i64,
    ) -> PlayerInput {
        let mut input = empty_input();
        let mut route = trace_route(field, tx, ty, LOOKAHEAD_TILES);
        if here >= UNREACHABLE || route.is_empty() {
            let Some(rescue) = nearest_on_route(field, tx, ty) else {
                return input;
            };
            route = vec![rescue];
        }
        let wp = route[LOOKAHEAD_TILES.min(route.len()) - 1];
        let next = route[0];
        let want_x = centre_of(wp.0);
        let mut direction = if me.pos.x < want_x - CENTRE_PX {
            1
        } else if me.pos.x > want_x + CENTRE_PX {
            -1
        } else {
            0
        };
        let mut want_up = next.1 < ty;
        let brake_px = 24.0 + js::max(0.0, me.vel.x * f64::from(direction)) * (self.lag as f64 + WALK_BRAKE_TICKS);
        if direction != 0 && hazard_within(col, me, direction, brake_px) {
            direction = if me.vel.x * f64::from(direction) > 0.5 {
                -direction
            } else {
                0
            };
            want_up = false;
        } else if let Some(target) = self.goals.get(self.index)
            && direction != 0
            && target.tele.is_none()
        {
            // Carried past the goal into a hazard behind it: turn back early (lag and margin counted).
            let carried = js::max(0.0, me.vel.x * f64::from(direction));
            let to_goal = (centre_of(target.tx) - me.pos.x) * f64::from(direction);
            let brake_ticks = self.lag as f64 + WALK_BRAKE_TICKS + LAG_MARGIN_TICKS;
            if carried > 0.5
                && to_goal > 0.0
                && to_goal <= carried * brake_ticks + f64::from(TILE_PX)
                && hazard_within(col, me, direction, to_goal + 24.0 + carried * brake_ticks)
            {
                direction = -direction;
                want_up = false;
            }
        }
        let rise_tiles = ty - route[route.len() - 1].1;
        let want_climb = rise_tiles >= CLIMB_MIN_RISE_TILES && !hazard_within(col, me, direction, 24.0);
        if (self.climb_anchor.is_some() || want_climb)
            && let Some(c) = self.climb(col, me, tick, direction, want_x)
        {
            self.aim_at(c.aim_x, c.aim_y, me, &mut input);
            input.direction = c.direction;
            input.jump = i32::from(c.jump);
            input.hook = i32::from(c.hook);
            return input;
        }
        input.direction = direction;
        input.jump = i32::from(want_up && (me.vel.y < RISING_VEL || self.steps % 2 == 0));
        input.hook = 0;
        input.fire = 0;
        self.aim_at(want_x, centre_of(wp.1), me, &mut input);
        input
    }

    fn aim_at(&mut self, x: f64, y: f64, me: &TeeState, input: &mut PlayerInput) {
        let want = js::atan2(y - me.pos.y, x - me.pos.x);
        let mut d = want - self.aim;
        while d > std::f64::consts::PI {
            d -= 2.0 * std::f64::consts::PI;
        }
        while d < -std::f64::consts::PI {
            d += 2.0 * std::f64::consts::PI;
        }
        self.aim += js::max(-0.12, js::min(0.12, d));
        let ax = js::round(js::cos(self.aim) * 300.0);
        let ay = js::round(js::sin(self.aim) * 300.0);
        input.target_x = if ax == 0.0 && ay == 0.0 { 300.0 } else { ax };
        input.target_y = ay;
    }

    fn climb<C: PlanCollision>(
        &mut self,
        col: &C,
        me: &TeeState,
        tick: i64,
        direction: i32,
        want_x: f64,
    ) -> Option<Climb> {
        if self.climb_anchor.is_none() {
            let anchor = find_anchor(col, me, direction)?;
            self.climb_anchor = Some(anchor);
            self.climb_start = tick;
            self.climb_best_y = me.pos.y;
            self.note(format!(
                "climbing to ({}, {})",
                js::round(anchor.x / f64::from(TILE_PX)),
                js::round(anchor.y / f64::from(TILE_PX))
            ));
        }
        let anchor = self.climb_anchor.expect("anchor");
        if me.pos.y < self.climb_best_y - 1.0 {
            self.climb_best_y = me.pos.y;
        }
        let stalled = tick - self.climb_start > CLIMB_GIVE_UP_TICKS;
        let arrived = me.pos.y <= anchor.y + CLIMB_ARRIVE_PX;
        if arrived || stalled {
            self.climb_anchor = None;
            if stalled {
                self.note("climb stalled, back to walking");
            }
            return Some(Climb {
                aim_x: want_x,
                aim_y: me.pos.y - 200.0,
                direction,
                jump: arrived,
                hook: false,
            });
        }
        let lean = if direction != 0 && hazard_within(col, me, direction, 24.0 + js::abs(me.vel.x) * CLIMB_BRAKE_TICKS)
        {
            if me.vel.x * f64::from(direction) > 0.5 {
                -direction
            } else {
                0
            }
        } else {
            direction
        };
        Some(Climb {
            aim_x: anchor.x,
            aim_y: anchor.y,
            direction: lean,
            jump: false,
            hook: true,
        })
    }

    fn step_crossing(
        &mut self,
        ctx: &mut NavCtx<'_, W>,
        me: &TeeState,
        tick: i64,
        goal: &NavGoal,
        others: &[TeeState],
    ) -> Option<PlayerInput> {
        let ci = self.to_crossing?;
        if self.crosser.is_none() {
            let c = &self.crossings[ci];
            if me.frozen || !in_any_box(&c.from, tile_of(me.pos.x), tile_of(me.pos.y)) {
                return None;
            }
            self.runner = None;
            let mut crosser = SwingCrosser::new((ctx.make_sim)(), c.clone());
            crosser.smart = self.smart;
            self.crosser = Some(crosser);
            let label = c.label.clone();
            self.note(format!(
                "at the start of {label}: swinging through on the rope (try {} of {MAX_CROSS_TRIES})",
                self.cross_tries + 1
            ));
        }
        let label = self.crossings[ci].label.clone();
        let lag = self.lag;
        let budget = self.cross_budget_ms;
        let wall_route = self.wall_route;
        let (out, doing_changed, doing, phase, reason, done) = {
            let crosser = self.crosser.as_mut().expect("crosser");
            crosser.budget_ms = budget;
            crosser.use_wall = wall_route;
            crosser.set_others(me, others);
            let was = crosser.doing();
            let out = crosser.step(ctx.col, me, tick, lag);
            let doing = crosser.doing();
            (
                out,
                doing != was,
                doing,
                crosser.phase(),
                crosser.reason().to_string(),
                crosser.done(),
            )
        };
        if doing_changed && !done && phase != CrossPhase::Approach {
            self.note(format!("{label}: {doing} (lag {lag})"));
        }
        if phase == CrossPhase::Arrived {
            self.note(format!("{reason}; on to {}", goal.label));
            self.crosser = None;
            self.to_crossing = None;
            self.field = None;
            self.runner = None;
            self.walk_routed = false;
            self.window_best = i32::MAX;
            self.window_ref = i32::MAX;
            self.window_start = tick;
            return Some(empty_input());
        }
        if phase == CrossPhase::Failed {
            self.crosser = None;
            self.cross_tries += 1;
            if self.cross_tries >= MAX_CROSS_TRIES {
                self.next_goal(
                    format!(
                        "no way through {label} to {}: {reason}, {} times",
                        goal.label, self.cross_tries
                    ),
                    tick,
                );
                return Some(empty_input());
            }
            self.note(format!("{label}: {reason}; trying again from the spawn"));
            self.field = None;
            return Some(empty_input());
        }
        self.window_start = tick;
        Some(out)
    }

    fn start_crossing(&mut self, ctx: &mut NavCtx<'_, W>, me: &TeeState, goal: &NavGoal) -> bool {
        if self.crossings.is_empty() {
            return false;
        }
        let tx = tile_of(me.pos.x);
        let ty = tile_of(me.pos.y);
        let mut best: Option<usize> = None;
        let mut best_route: Option<crate::route::RouteResult> = None;
        for ci in 0..self.crossings.len() {
            let reach = match self.crossing_reach.get(&ci) {
                Some(&r) => r,
                None => {
                    let c = &self.crossings[ci];
                    let far = c.hall_tile.unwrap_or(c.exit_tile);
                    let on = ctx.router.find_route(
                        (centre_of(far.0), centre_of(far.1)),
                        (centre_of(goal.tx), centre_of(goal.ty)),
                        &RouteOpts {
                            near_tiles: 2,
                            through_freeze: self.through_freeze,
                            ..RouteOpts::default()
                        },
                    );
                    let r = on.is_some();
                    self.crossing_reach.insert(ci, r);
                    r
                }
            };
            if !reach {
                continue;
            }
            let c = &self.crossings[ci];
            if in_any_box(&c.from, tx, ty) {
                best = Some(ci);
                best_route = None;
                break;
            }
            let way = ctx.router.find_route(
                (me.pos.x, me.pos.y),
                (centre_of(c.start.0), centre_of(c.start.1)),
                &RouteOpts {
                    near_tiles: 1,
                    allow_kill: self.allow_kill,
                    through_freeze: self.through_freeze,
                    avoid: Some(&self.avoid),
                    ..RouteOpts::default()
                },
            );
            let Some(way) = way else { continue };
            if way.steps.is_empty() {
                continue;
            }
            if best_route.as_ref().is_none_or(|b| way.cost < b.cost) {
                best = Some(ci);
                best_route = Some(way);
            }
        }
        let Some(best) = best else { return false };
        self.to_crossing = Some(best);
        if let Some(r) = best_route {
            let n = r.steps.len();
            self.runner = Some(RouteRunner::new(
                r.steps,
                Some(&ctx.router.grid),
                !self.crossings.is_empty(),
            ));
            self.note(format!(
                "no way to {} but through {}; going to its start first: {n} steps",
                goal.label, self.crossings[best].label
            ));
        }
        true
    }

    /// `respawned()`: we died (or were killed).
    pub fn respawned(&mut self) {
        self.crosser = None;
        if self.runner.as_ref().is_some_and(RouteRunner::awaiting_kill) {
            self.runner.as_mut().expect("runner").respawned();
        } else if self.runner.is_some() {
            self.drop_route("died on it");
        }
        self.climb_anchor = None;
        self.climb_start = -1;
        self.climb_best_y = f64::INFINITY;
        self.last_pos = None;
        self.window_best = i32::MAX;
        self.window_ref = i32::MAX;
        self.window_start = self.last_tick;
    }

    fn drop_route(&mut self, why: &str) {
        self.runner = None;
        self.field = None;
        self.drops += 1;
        if self.drops > MAX_ROUTE_DROPS {
            self.drop_give_up = Some(format!("{why} {} times", self.drops));
        }
    }

    /// `cancel(why)`.
    pub fn cancel(&mut self, why: &str) {
        self.climb_anchor = None;
        self.climb_start = -1;
        self.climb_best_y = f64::INFINITY;
        if self.done() {
            return;
        }
        self.finish(NavPhase::Blocked, why.to_string());
    }

    /// A swing is running (`crossing`): its input is not checked by the guard.
    pub fn crossing(&self) -> bool {
        self.crosser.is_some()
    }

    /// `crossingState`: the tube the walk is heading for (or crossing), and whether the swing is thrown.
    pub fn crossing_state(&self) -> Option<(&Crossing, bool)> {
        let c = &self.crossings[self.to_crossing?];
        Some((c, self.crosser.as_ref().is_some_and(SwingCrosser::thrown)))
    }

    /// `crossFails`: crossings of the current goal that failed so far.
    pub fn cross_fails(&self) -> i32 {
        self.cross_tries
    }

    /// A planned freeze lies ahead on the route (`plannedFreeze`): not checked by the guard either.
    pub fn planned_freeze(&self) -> bool {
        !self.crossings.is_empty()
            && self
                .runner
                .as_ref()
                .is_some_and(|r| r.freeze_ahead(PLANNED_FREEZE_STEPS))
    }

    pub fn elapsed_ticks(&self) -> i64 {
        if self.start_tick < 0 {
            0
        } else {
            self.last_tick - self.start_tick
        }
    }

    pub fn current_route(&self) -> Option<&RouteRunner> {
        self.runner.as_ref()
    }
}

struct Climb {
    aim_x: f64,
    aim_y: f64,
    direction: i32,
    jump: bool,
    hook: bool,
}

/// `(tick - windowStart) / 50` printed with `toFixed(0)`.
fn format_fixed0(v: f64) -> String {
    // `Number.prototype.toFixed(0)` rounds half away from zero for positive values.
    format!("{}", (v + 0.5).floor())
}

fn nearest_on_route(field: &HazardField, tx: i32, ty: i32) -> Option<(i32, i32)> {
    let mut best: Option<(i32, i32)> = None;
    let mut best_d = UNREACHABLE;
    for dy in -3..=3 {
        for dx in -3..=3 {
            let d = dist_at(field, tx + dx, ty + dy);
            if d < best_d {
                best_d = d;
                best = Some((tx + dx, ty + dy));
            }
        }
    }
    best
}

fn find_anchor<C: PlanCollision>(col: &C, me: &TeeState, direction: i32) -> Option<Vec2> {
    let mut best: Option<Vec2> = None;
    let mut best_score = f64::NEG_INFINITY;
    for r in 0..CLIMB_ARC_RAYS {
        let angle = -std::f64::consts::PI + (std::f64::consts::PI * (f64::from(r) + 0.5)) / f64::from(CLIMB_ARC_RAYS);
        let dx = js::cos(angle);
        let dy = js::sin(angle);
        let mut t = f64::from(TILE_PX);
        while t <= HOOK_LENGTH {
            let x = me.pos.x + dx * t;
            let y = me.pos.y + dy * t;
            if !col.is_solid(x, y) {
                t += CLIMB_RAY_STEP_PX;
                continue;
            }
            if col.is_no_hook(x, y) {
                break;
            }
            let rise = me.pos.y - y;
            if rise < f64::from(TILE_PX) {
                break;
            }
            let bonus = if direction != 0 && js::sign(x - me.pos.x) == f64::from(direction) {
                40.0
            } else {
                0.0
            };
            let score = rise + bonus;
            if score > best_score {
                best_score = score;
                best = Some(Vec2 { x, y });
            }
            break;
        }
    }
    best
}

fn hazard_within<C: PlanCollision>(col: &C, me: &TeeState, direction: i32, px: f64) -> bool {
    hazard_within_px(col, me.pos.x, me.pos.y, direction, px)
}

/// `hazardWithinPx(collision, x0, y0, direction, px)`: freeze or death within `px` ahead of `(x0, y0)`
/// in `direction`, down to four tiles below, before a wall.
pub fn hazard_within_px(col: &impl PlanCollision, x0: f64, y0: f64, direction: i32, px: f64) -> bool {
    let mut offsets: Vec<f64> = Vec::new();
    let mut a = js::min(24.0, px);
    while a < px {
        offsets.push(a);
        a += f64::from(TILE_PX);
    }
    offsets.push(px);
    for ahead in offsets {
        let x = x0 + f64::from(direction) * ahead;
        if col.is_solid(x, y0) {
            return false;
        }
        for dy in 0..=4 {
            let y = y0 + f64::from(dy * TILE_PX);
            if col.is_freeze(x, y) || col.is_death(x, y) {
                return true;
            }
            if col.is_solid(x, y) {
                break;
            }
        }
    }
    false
}
