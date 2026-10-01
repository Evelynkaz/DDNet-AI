//! `src/bot/crossing.ts`: crossing freeze tubes on the rope (`SwingCrosser`). The tee approaches the
//! start of a tube, then searches — by rolling a private simulation forward — for a rope swing (anchor,
//! hold length, push direction), a hop or a drop that gets it through without freezing, with the
//! network lag and small position errors taken into account; the first program that survives every
//! variation is run. Generic over the planner's [`PlanWorld`]: on `ddai-tsworld` the choices are
//! bit-for-bit those of the TS code, on `ddai_physics::World<f32>` they drive the live bot.

use ddai_planner::plan_world::{PlanCollision, PlanWorld};
use ddai_planner::types::{HOOK_GRABBED, PlayerInput, TeeState, empty_input};
use ddai_planner::vmath::Vec2;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::time::Instant;

use crate::TILE_PX;

/// `TileBox` (inclusive tile rectangle).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileBox {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl TileBox {
    pub const fn new(x0: i32, y0: i32, x1: i32, y1: i32) -> TileBox {
        TileBox { x0, y0, x1, y1 }
    }
    pub fn contains(&self, tx: i32, ty: i32) -> bool {
        tx >= self.x0 && tx <= self.x1 && ty >= self.y0 && ty <= self.y1
    }
    pub fn shifted(&self, dx: i32, dy: i32) -> TileBox {
        TileBox::new(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)
    }
}

/// `inAnyBox`.
pub fn in_any_box(boxes: &[TileBox], tx: i32, ty: i32) -> bool {
    boxes.iter().any(|b| b.contains(tx, ty))
}

/// `Crossing`: one tube.
#[derive(Debug, Clone, PartialEq)]
pub struct Crossing {
    pub label: String,
    /// Where the swing may start.
    pub from: Vec<TileBox>,
    /// The freeze chamber.
    pub chamber: TileBox,
    pub start: (i32, i32),
    /// Tiles the rope can grab.
    pub anchors: Vec<(i32, i32)>,
    pub landing: Vec<TileBox>,
    pub exit: Vec<TileBox>,
    pub exit_tile: (i32, i32),
    pub hall: Option<Vec<TileBox>>,
    pub hall_tile: Option<(i32, i32)>,
    /// -1 = left, 1 = right.
    pub toward: i32,
}

impl Crossing {
    /// `shiftCrossing`.
    pub fn shifted(&self, dx: i32, dy: i32) -> Crossing {
        let sh = |v: &[TileBox]| v.iter().map(|b| b.shifted(dx, dy)).collect::<Vec<_>>();
        Crossing {
            label: self.label.clone(),
            from: sh(&self.from),
            chamber: self.chamber.shifted(dx, dy),
            start: (self.start.0 + dx, self.start.1 + dy),
            anchors: self.anchors.iter().map(|a| (a.0 + dx, a.1 + dy)).collect(),
            landing: sh(&self.landing),
            exit: sh(&self.exit),
            exit_tile: (self.exit_tile.0 + dx, self.exit_tile.1 + dy),
            hall: self.hall.as_ref().map(|h| sh(h)),
            hall_tile: self.hall_tile.map(|t| (t.0 + dx, t.1 + dy)),
            toward: self.toward,
        }
    }
}

const HOLDS: [i32; 4] = [10, 16, 24, 40];
const PUSH_AFTER_TICKS: i64 = 30;
const SETTLE_TICKS: i64 = 110;
const HOP_RUNS: [i32; 3] = [10, 20, 40];
const HOP_JUMP_AT: [i32; 7] = [0, 2, 4, 6, 8, 10, 14];
const HOP_JUMP_HOLD: [i32; 2] = [6, 14];
const HOP_TICKS: i64 = 90;
const DROP_RUNS: [i32; 4] = [0, 6, 12, 24];
const DROP_TICKS: i64 = 260;
const AIR_RUNS: [i32; 4] = [4, 10, 20, 40];
const AIR_JUMP_AT: [i32; 3] = [-1, 0, 3];
const MAX_HOPS: i32 = 3;
const SPREAD_TICKS: i64 = 2;
const SPREAD_EARLY_TICKS: i64 = 1;
const NUDGE_PX: f64 = 6.0;
const NUDGE_Y_PX: f64 = 4.0;
const HOP_CLEAR_PX: f64 = 6.0;
const ARRIVED_VX: f64 = 3.0;
const APPROACH_DEPTH_TILES: i32 = 3;
const APPROACH_GIVE_UP_TICKS: i64 = 150;
const HOOK_LENGTH_PX: f64 = 380.0;

/// A rope swing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Swing {
    pub anchor: (i32, i32),
    pub hold: i32,
    pub dir: i32,
}

/// A hop / drop / in-air correction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hop {
    pub what: &'static str,
    pub dir: i32,
    pub run: i32,
    pub jump_at: i32,
    pub jump_hold: i32,
    pub ticks: i64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Move {
    Swing(Swing),
    Hop(Hop),
}

#[derive(Debug, Clone, Copy)]
struct Program {
    mv: Move,
    start_tick: i64,
    froze: bool,
}

/// `CrossPhase`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossPhase {
    Approach,
    Swinging,
    Hopping,
    Arrived,
    Failed,
}

fn centre(t: i32) -> f64 {
    f64::from(t * TILE_PX) + f64::from(TILE_PX) / 2.0
}

fn tile_of(px: f64) -> i32 {
    (px / f64::from(TILE_PX)).trunc() as i32
}

#[derive(Debug, Clone, Copy)]
struct Lap {
    at: usize,
    tried: usize,
}

/// `class SwingCrosser`.
pub struct SwingCrosser<W: PlanWorld> {
    pub crossing: Crossing,
    sim: RefCell<W>,
    tee_added: Cell<bool>,
    program: Option<Program>,
    approach_dir: i32,
    start_tick: i64,
    last_tick: i64,
    cadence: i64,
    hops: i32,
    phase: CrossPhase,
    why: String,
    /// Rollouts run (statistics).
    pub tried: Cell<u64>,
    /// Wall-clock budget per `step` in ms (`0` = none, as in TS `budgetMs`).
    pub budget_ms: f64,
    until: Cell<Option<Instant>>,
    laps: RefCell<HashMap<&'static str, Lap>>,
    out_of_time: Cell<bool>,
    /// How often the budget stopped a search (statistics).
    pub stops: Cell<u64>,
}

impl<W: PlanWorld> SwingCrosser<W> {
    /// `sim` is the crosser's private simulation world of the map (`template.new_scratch()`); the tee
    /// is added to it on first use. The collision is passed to every [`SwingCrosser::step`].
    pub fn new(sim: W, crossing: Crossing) -> SwingCrosser<W> {
        SwingCrosser {
            crossing,
            sim: RefCell::new(sim),
            tee_added: Cell::new(false),
            program: None,
            approach_dir: 0,
            start_tick: -1,
            last_tick: -1,
            cadence: 1,
            hops: 0,
            phase: CrossPhase::Approach,
            why: String::new(),
            tried: Cell::new(0),
            budget_ms: 0.0,
            until: Cell::new(None),
            laps: RefCell::new(HashMap::new()),
            out_of_time: Cell::new(false),
            stops: Cell::new(0),
        }
    }

    pub fn phase(&self) -> CrossPhase {
        self.phase
    }
    pub fn done(&self) -> bool {
        matches!(self.phase, CrossPhase::Arrived | CrossPhase::Failed)
    }
    pub fn reason(&self) -> &str {
        &self.why
    }

    /// `doing`: a short description of the running program.
    pub fn doing(&self) -> String {
        let Some(p) = &self.program else {
            return format!("{:?}", self.phase).to_lowercase();
        };
        match p.mv {
            Move::Hop(h) => {
                let push = if h.dir == 0 || h.run == 0 {
                    String::new()
                } else {
                    format!(
                        " pushing {} {} ticks",
                        if h.dir == self.crossing.toward { "on" } else { "back" },
                        h.run
                    )
                };
                let jump = if h.jump_at < 0 {
                    String::new()
                } else {
                    format!(", jump at {} for {}", h.jump_at, h.jump_hold)
                };
                format!("{}{}{}", h.what, push, jump)
            }
            Move::Swing(s) => format!(
                "rope on ({},{}) for {} ticks{}",
                s.anchor.0,
                s.anchor.1,
                s.hold,
                if s.dir == 0 { ", swinging free" } else { ", pushing on" }
            ),
        }
    }

    /// The program the next `step` would be running (tests / parity).
    pub fn program(&self) -> Option<Move> {
        self.program.map(|p| p.mv)
    }

    // --- predicates -----------------------------------------------------------------------------

    fn supported(&self, col: &W::Collision, me: &TeeState) -> bool {
        col.is_solid(me.pos.x, me.pos.y + 17.0)
            || col.is_solid(me.pos.x - 14.0, me.pos.y + 17.0)
            || col.is_solid(me.pos.x + 14.0, me.pos.y + 17.0)
    }

    fn arrived_at(&self, col: &W::Collision, me: &TeeState) -> bool {
        if me.frozen {
            return false;
        }
        if let Some(hall) = &self.crossing.hall {
            return me.vel.x.abs() <= 2.0
                && self.supported(col, me)
                && in_any_box(hall, tile_of(me.pos.x), tile_of(me.pos.y));
        }
        self.in_passage(me)
    }

    fn in_passage(&self, me: &TeeState) -> bool {
        !me.frozen
            && me.vel.x.abs() <= ARRIVED_VX
            && in_any_box(&self.crossing.exit, tile_of(me.pos.x), tile_of(me.pos.y))
    }

    fn through_at(&self, col: &W::Collision, me: &TeeState) -> bool {
        self.arrived_at(col, me) || self.in_passage(me)
    }

    fn in_freeze(&self, col: &W::Collision, me: &TeeState) -> bool {
        for dx in [-14.0, 14.0] {
            for dy in [-14.0, 14.0] {
                if col.is_freeze(me.pos.x + dx, me.pos.y + dy) {
                    return true;
                }
            }
        }
        false
    }

    fn near_freeze(&self, col: &W::Collision, me: &TeeState, px: f64) -> bool {
        let r = 14.0 + px;
        for dx in [-r, 0.0, r] {
            for dy in [-r, 0.0, r] {
                if col.is_freeze(me.pos.x + dx, me.pos.y + dy) {
                    return true;
                }
            }
        }
        false
    }

    fn in_chamber_floor(&self, me: &TeeState) -> bool {
        let ch = &self.crossing.chamber;
        let tx = tile_of(me.pos.x);
        tx >= ch.x0 && tx <= ch.x1 && tile_of(me.pos.y) > ch.y1
    }

    fn landed_at(&self, col: &W::Collision, me: &TeeState) -> bool {
        !me.frozen
            && me.vel.x.abs() <= 1.0
            && self.supported(col, me)
            && in_any_box(&self.crossing.landing, tile_of(me.pos.x), tile_of(me.pos.y))
    }

    fn in_from(&self, tx: i32, ty: i32) -> bool {
        in_any_box(&self.crossing.from, tx, ty)
    }

    fn near_from(&self, tx: i32, ty: i32) -> bool {
        self.crossing
            .from
            .iter()
            .any(|b| tx >= b.x0 - 2 && tx <= b.x1 + 2 && ty >= b.y0 - 2 && ty <= b.y1 + 2)
    }

    fn fail(&mut self, why: String) -> PlayerInput {
        self.phase = CrossPhase::Failed;
        self.why = why;
        self.program = None;
        empty_input()
    }

    // --- the step -------------------------------------------------------------------------------

    /// `step(self, tick, lag)`.
    pub fn step(&mut self, col: &W::Collision, me: &TeeState, tick: i64, lag: i64) -> PlayerInput {
        let input = empty_input();
        if self.done() {
            return input;
        }
        self.until.set(if self.budget_ms > 0.0 {
            Some(Instant::now() + std::time::Duration::from_secs_f64(self.budget_ms / 1000.0))
        } else {
            None
        });
        self.out_of_time.set(false);
        if self.start_tick < 0 || tick < self.start_tick {
            self.start_tick = tick;
        }
        if self.last_tick >= 0 && tick > self.last_tick {
            self.cadence = (tick - self.last_tick).min(4);
        }
        self.last_tick = tick;
        if !me.alive {
            return self.fail("died on the way".to_string());
        }
        if self.arrived_at(col, me) {
            self.phase = CrossPhase::Arrived;
            self.why = format!("through {}", self.crossing.label);
            return input;
        }
        let tx = tile_of(me.pos.x);
        let ty = tile_of(me.pos.y);
        if me.frozen {
            if ddai_jsmath::hypot2(me.vel.x, me.vel.y) < 0.5
                && self.supported(col, me)
                && (self.in_freeze(col, me) || self.in_chamber_floor(me))
            {
                return self.fail(format!("lies frozen at ({tx},{ty})"));
            }
            if let Some(p) = &mut self.program {
                p.froze = true;
                let p = *p;
                return self.program_input(me, p.mv, tick - p.start_tick, input);
            }
            return input;
        }
        if let Some(p) = self.program
            && !self.supported(col, me)
        {
            let t = tick - p.start_tick;
            let roped = matches!(p.mv, Move::Swing(s) if t < i64::from(s.hold)) && me.hook_state == HOOK_GRABBED;
            if !roped
                && t > 0
                && !self.works(col, me, p.mv, t, lag)
                && let Some(fix) = self.search_hop(col, me, lag, true)
            {
                self.program = Some(Program {
                    mv: Move::Hop(fix),
                    start_tick: tick,
                    froze: p.froze,
                });
            }
        }
        if let Some(p) = self.program {
            let t = tick - p.start_tick;
            match p.mv {
                Move::Swing(s) => {
                    let hold = i64::from(s.hold);
                    if (!p.froze && t > 12 && t < hold && me.hook_state != HOOK_GRABBED)
                        || (t >= hold && (self.landed_at(col, me) || self.in_passage(me)))
                    {
                        self.program = None;
                    } else if t > hold + PUSH_AFTER_TICKS + SETTLE_TICKS {
                        return self.fail(format!("the swing ran out at ({tx},{ty})"));
                    } else if t < hold + PUSH_AFTER_TICKS || !self.in_from(tx, ty) {
                        return self.program_input(me, p.mv, t, input);
                    } else {
                        self.program = None;
                    }
                }
                Move::Hop(h) => {
                    let dropping = h.ticks > HOP_TICKS;
                    if t > 4 && !dropping && (self.landed_at(col, me) || self.in_passage(me)) {
                        self.program = None;
                    } else if t > h.ticks {
                        return self.fail(format!(
                            "the {} ran out at ({tx},{ty})",
                            if dropping { "drop" } else { "hop" }
                        ));
                    } else {
                        return self.program_input(me, p.mv, t, input);
                    }
                }
            }
        }
        if self.crossing.hall.is_some() && self.in_passage(me) {
            let drop = self.search_drop(col, me, lag);
            if drop.is_none() && self.out_of_time.get() {
                return input;
            }
            let Some(drop) = drop else {
                self.phase = CrossPhase::Arrived;
                self.why = format!("through {}, at the foot of the passage", self.crossing.label);
                return input;
            };
            let p = Program {
                mv: Move::Hop(drop),
                start_tick: tick,
                froze: false,
            };
            self.program = Some(p);
            self.phase = CrossPhase::Hopping;
            return self.program_input(me, p.mv, 0, input);
        }
        if self.landed_at(col, me) {
            if self.hops >= MAX_HOPS {
                return self.fail(format!("{} hops and still at ({tx},{ty})", self.hops));
            }
            let hop = self.search_hop(col, me, lag, false);
            if hop.is_none() && self.out_of_time.get() {
                return input;
            }
            let Some(hop) = hop else {
                return self.fail(format!("no hop from ({tx},{ty}) that clears the freeze"));
            };
            self.hops += 1;
            let p = Program {
                mv: Move::Hop(hop),
                start_tick: tick,
                froze: false,
            };
            self.program = Some(p);
            self.phase = CrossPhase::Hopping;
            return self.program_input(me, p.mv, 0, input);
        }
        if tick - self.start_tick > APPROACH_GIVE_UP_TICKS {
            return self.fail("no swing through from where it got to".to_string());
        }
        if !self.near_from(tx, ty) {
            return self.fail(format!("off the start of it at ({tx},{ty})"));
        }
        {
            let ch = self.crossing.chamber;
            let toward = self.crossing.toward;
            let near_edge = if toward < 0 { ch.x0 } else { ch.x1 };
            let depth = (tx - near_edge) * -toward;
            if tx < ch.x0 - 3 || tx > ch.x1 + 3 {
                self.approach_dir = if tx < ch.x0 { 1 } else { -1 };
            } else if depth < APPROACH_DEPTH_TILES && (self.supported(col, me) || self.approach_dir == -toward) {
                self.approach_dir = -toward;
            } else {
                self.approach_dir = toward;
            }
        }
        if let Some(found) = self.search_swing(col, me, lag) {
            let p = Program {
                mv: Move::Swing(found),
                start_tick: tick,
                froze: false,
            };
            self.program = Some(p);
            self.phase = CrossPhase::Swinging;
            return self.program_input(me, p.mv, 0, input);
        }
        self.approach_input(input)
    }

    fn approach_input(&self, mut input: PlayerInput) -> PlayerInput {
        input.direction = self.approach_dir;
        input.hook = 0;
        input.jump = 0;
        input.target_x = f64::from(self.approach_dir) * 300.0;
        input.target_y = 0.0;
        input
    }

    fn program_input(&self, me: &TeeState, mv: Move, t: i64, mut input: PlayerInput) -> PlayerInput {
        let toward = self.crossing.toward;
        match mv {
            Move::Hop(p) => {
                input.direction = if t < i64::from(p.run) { p.dir } else { 0 };
                input.jump =
                    i32::from(p.jump_at >= 0 && t >= i64::from(p.jump_at) && t < i64::from(p.jump_at + p.jump_hold));
                input.hook = 0;
                input.target_x = f64::from(toward) * 300.0;
                input.target_y = 0.0;
                input
            }
            Move::Swing(p) => {
                let holding = t < i64::from(p.hold);
                input.hook = i32::from(holding);
                input.jump = 0;
                input.direction = if t < i64::from(p.hold) + PUSH_AFTER_TICKS {
                    p.dir
                } else {
                    0
                };
                input.target_x = if holding {
                    ddai_jsmath::round(centre(p.anchor.0) - me.pos.x)
                } else {
                    f64::from(toward) * 300.0
                };
                input.target_y = if holding {
                    ddai_jsmath::round(centre(p.anchor.1) - me.pos.y)
                } else {
                    0.0
                };
                if input.target_x == 0.0 && input.target_y == 0.0 {
                    input.target_y = -1.0;
                }
                input
            }
        }
    }

    // --- the search -----------------------------------------------------------------------------

    fn with_sim<R>(&self, me: &TeeState, f: impl FnOnce(&mut W) -> R) -> R {
        let mut sim = self.sim.borrow_mut();
        if !self.tee_added.get() {
            sim.add_tee(
                0,
                Vec2 {
                    x: me.pos.x,
                    y: me.pos.y,
                },
            );
            self.tee_added.set(true);
        }
        f(&mut sim)
    }

    #[allow(clippy::too_many_arguments)]
    fn robust(
        &self,
        col: &W::Collision,
        me: &TeeState,
        mv: Move,
        lag: i64,
        done: &dyn Fn(&TeeState) -> bool,
        ticks: i64,
        approaching: bool,
    ) -> bool {
        let mut lags = vec![lag];
        let mut d = (lag - SPREAD_EARLY_TICKS).max(0);
        while d <= lag + SPREAD_TICKS {
            if d != lag {
                lags.push(d);
            }
            d += 1;
        }
        let mut with_id = *me;
        with_id.id = 0;
        for d in lags {
            let ok = self.with_sim(me, |sim| {
                sim.apply_tee_state(0, &with_id);
                self.rollout(col, sim, mv, d, done, ticks, approaching, 0)
            });
            if !ok {
                return false;
            }
        }
        for (dx, dy) in [(-NUDGE_PX, 0.0), (NUDGE_PX, 0.0), (0.0, -NUDGE_Y_PX), (0.0, NUDGE_Y_PX)] {
            let x = me.pos.x + dx;
            let y = me.pos.y + dy;
            if col.is_solid(x - 14.0, y - 14.0)
                || col.is_solid(x + 14.0, y - 14.0)
                || col.is_solid(x - 14.0, y + 14.0)
                || col.is_solid(x + 14.0, y + 14.0)
            {
                continue;
            }
            let mut st = with_id;
            st.pos = Vec2 { x, y };
            let ok = self.with_sim(me, |sim| {
                sim.apply_tee_state(0, &st);
                self.rollout(col, sim, mv, lag, done, ticks, approaching, 0)
            });
            if !ok {
                return false;
            }
        }
        true
    }

    /// `firstThat(kind, list, works)`: the first entry that works, resuming a search the budget cut short.
    fn first_that<T: Copy>(&self, kind: &'static str, list: &[T], works: &dyn Fn(&T) -> bool) -> Option<T> {
        let n = list.len();
        let lap = self.laps.borrow_mut().remove(kind);
        if n == 0 {
            return None;
        }
        let from = lap.map_or(0, |l| l.at % n);
        let mut tried = lap.map_or(0, |l| l.tried.min(n - 1));
        let mut k = 0usize;
        while tried < n {
            let i = (from + k) % n;
            if k > 0
                && let Some(until) = self.until.get()
                && Instant::now() > until
            {
                self.laps.borrow_mut().insert(kind, Lap { at: i, tried });
                self.out_of_time.set(true);
                self.stops.set(self.stops.get() + 1);
                return None;
            }
            if works(&list[i]) {
                self.laps.borrow_mut().clear();
                return Some(list[i]);
            }
            k += 1;
            tried += 1;
        }
        None
    }

    fn search_swing(&self, col: &W::Collision, me: &TeeState, lag: i64) -> Option<Swing> {
        let toward = self.crossing.toward;
        let done = |m: &TeeState| self.through_at(col, m) || self.landed_at(col, m);
        let mut list: Vec<Swing> = Vec::new();
        for &anchor in &self.crossing.anchors {
            let reach = ddai_jsmath::hypot2(centre(anchor.0) - me.pos.x, centre(anchor.1) - me.pos.y);
            if reach > HOOK_LENGTH_PX + (lag + SPREAD_TICKS) as f64 * 16.0 {
                continue;
            }
            for hold in HOLDS {
                for dir in [toward, 0] {
                    list.push(Swing { anchor, hold, dir });
                }
            }
        }
        self.first_that("swing", &list, &|p: &Swing| {
            self.robust(
                col,
                me,
                Move::Swing(*p),
                lag,
                &done,
                lag + i64::from(p.hold) + SETTLE_TICKS,
                true,
            )
        })
    }

    fn search_hop(&self, col: &W::Collision, me: &TeeState, lag: i64, in_air: bool) -> Option<Hop> {
        let toward = self.crossing.toward;
        let from = me.pos.x;
        let done = move |m: &TeeState| -> bool {
            if in_air {
                self.through_at(col, m) || self.landed_at(col, m)
            } else {
                self.through_at(col, m)
                    || (self.landed_at(col, m) && (m.pos.x - from) * f64::from(toward) > f64::from(TILE_PX))
            }
        };
        let dirs: Vec<i32> = if in_air { vec![toward, 0, -toward] } else { vec![toward] };
        let runs: &[i32] = if in_air { &AIR_RUNS } else { &HOP_RUNS };
        let jumps: &[i32] = if in_air { &AIR_JUMP_AT } else { &HOP_JUMP_AT };
        let passes: Vec<fn(i32) -> bool> = if in_air {
            vec![|j| j < 0, |j| j >= 0]
        } else {
            vec![|_| true]
        };
        let mut list: Vec<Hop> = Vec::new();
        for pass in &passes {
            for &dir in &dirs {
                for &run in runs {
                    for &jump_at in jumps {
                        if !pass(jump_at) || jump_at >= run {
                            continue;
                        }
                        let holds: &[i32] = if jump_at < 0 { &[0] } else { &HOP_JUMP_HOLD };
                        for &jump_hold in holds {
                            list.push(Hop {
                                what: if in_air { "in the air" } else { "hop" },
                                dir,
                                run,
                                jump_at,
                                jump_hold,
                                ticks: HOP_TICKS,
                            });
                        }
                    }
                }
            }
        }
        self.first_that(if in_air { "air" } else { "hop" }, &list, &|p: &Hop| {
            self.robust(col, me, Move::Hop(*p), lag, &done, lag + HOP_TICKS, false)
        })
    }

    fn search_drop(&self, col: &W::Collision, me: &TeeState, lag: i64) -> Option<Hop> {
        let done = |m: &TeeState| self.arrived_at(col, m);
        let toward = self.crossing.toward;
        let mut list: Vec<Hop> = Vec::new();
        for run in DROP_RUNS {
            let dirs: Vec<i32> = if run == 0 { vec![0] } else { vec![-toward, toward] };
            for dir in dirs {
                list.push(Hop {
                    what: "drop into the hall",
                    dir,
                    run,
                    jump_at: -1,
                    jump_hold: 0,
                    ticks: DROP_TICKS,
                });
            }
        }
        self.first_that("drop", &list, &|p: &Hop| {
            self.robust(col, me, Move::Hop(*p), lag, &done, lag + DROP_TICKS, false)
        })
    }

    /// `works(self, p, t, lag)`: does the running program still get through from here?
    fn works(&self, col: &W::Collision, me: &TeeState, mv: Move, t: i64, lag: i64) -> bool {
        let dropping = matches!(mv, Move::Hop(h) if h.ticks > HOP_TICKS);
        let done = |m: &TeeState| {
            if dropping {
                self.arrived_at(col, m)
            } else {
                self.through_at(col, m) || self.landed_at(col, m)
            }
        };
        let ticks = match mv {
            Move::Swing(s) => (lag + 20).max(lag + i64::from(s.hold) + SETTLE_TICKS - t),
            Move::Hop(h) => (lag + 20).max(lag + h.ticks - t),
        };
        let mut with_id = *me;
        with_id.id = 0;
        self.with_sim(me, |sim| {
            sim.apply_tee_state(0, &with_id);
            self.rollout(col, sim, mv, lag, &done, ticks, false, t)
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn rollout(
        &self,
        col: &W::Collision,
        sim: &mut W,
        mv: Move,
        lag: i64,
        done: &dyn Fn(&TeeState) -> bool,
        ticks: i64,
        approaching: bool,
        t0: i64,
    ) -> bool {
        self.tried.set(self.tried.get() + 1);
        let first = sim.get_tee(0);
        let mut held = if approaching {
            self.approach_input(empty_input())
        } else if t0 > 0 && first.is_some() {
            self.program_input(&first.expect("first"), mv, t0 - 1, empty_input())
        } else {
            empty_input()
        };
        let mut pending: std::collections::VecDeque<(i64, PlayerInput)> = std::collections::VecDeque::new();
        for t in 0..ticks {
            let Some(me) = sim.get_tee(0) else { return false };
            if !me.alive {
                return false;
            }
            if t % self.cadence == 0 {
                pending.push_back((t + lag, self.program_input(&me, mv, t0 + t, empty_input())));
            }
            while pending.front().is_some_and(|p| p.0 <= t) {
                held = pending.pop_front().expect("front").1;
            }
            sim.set_input(0, held);
            let _ = sim.step();
            let Some(now) = sim.get_tee(0) else { return false };
            if t > lag + 4 && done(&now) {
                return true;
            }
            if t > lag + 8
                && now.frozen
                && (self.in_chamber_floor(&now)
                    || (ddai_jsmath::hypot2(now.vel.x, now.vel.y) < 0.3 && self.in_freeze(col, &now)))
            {
                return false;
            }
            if let Move::Hop(h) = mv
                && h.ticks <= HOP_TICKS
                && t > lag
                && (now.frozen || (h.what == "in the air" && self.near_freeze(col, &now, HOP_CLEAR_PX)))
            {
                return false;
            }
        }
        false
    }
}
