//! `src/bot/crossing.ts` (upstream `af49dfb`, release 2026-10-02): crossing freeze tubes on the rope (`SwingCrosser`). The tee approaches the
//! start of a tube, then searches — by rolling a private simulation forward — for a rope swing (anchor,
//! hold length, push direction), a hop or a drop that gets it through without freezing, with the
//! network lag and small position errors taken into account; the first program that survives every
//! variation is run. Generic over the planner's [`PlanWorld`]: on `ddai-tsworld` the choices are
//! bit-for-bit those of the TS code, on `ddai_physics::World<f32>` they drive the live bot.

use ddai_planner::plan_world::{PlanCollision, PlanWorld};
use ddai_planner::types::{HOOK_GRABBED, PlayerInput, TeeState, empty_input};
use ddai_planner::vmath::Vec2;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
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

/// `inBox`.
pub fn in_box(b: &TileBox, tx: i32, ty: i32) -> bool {
    b.contains(tx, ty)
}

/// `inAnyBox`.
pub fn in_any_box(boxes: &[TileBox], tx: i32, ty: i32) -> bool {
    boxes.iter().any(|b| b.contains(tx, ty))
}

/// `WallRoute`: "route 2" of a tube, out of the passage through its far wall onto the lower shelf.
#[derive(Debug, Clone, PartialEq)]
pub struct WallRoute {
    /// Tiles of the wall column the rope grabs.
    pub anchors: Vec<(i32, i32)>,
    pub shelf: Vec<TileBox>,
    pub room: Vec<TileBox>,
    pub last_row: i32,
    pub miss_row: i32,
}

impl WallRoute {
    /// `shiftWall`.
    pub fn shifted(&self, dx: i32, dy: i32) -> WallRoute {
        let sh = |v: &[TileBox]| v.iter().map(|b| b.shifted(dx, dy)).collect::<Vec<_>>();
        WallRoute {
            anchors: self.anchors.iter().map(|a| (a.0 + dx, a.1 + dy)).collect(),
            shelf: sh(&self.shelf),
            room: sh(&self.room),
            last_row: self.last_row + dy,
            miss_row: self.miss_row + dy,
        }
    }
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
    /// `directAnchors`: indices into `anchors` of the ones a swing straight into the passage may use.
    pub direct_anchors: Vec<usize>,
    pub landing: Vec<TileBox>,
    pub exit: Vec<TileBox>,
    pub exit_tile: (i32, i32),
    pub hall: Option<Vec<TileBox>>,
    pub hall_tile: Option<(i32, i32)>,
    /// -1 = left, 1 = right.
    pub toward: i32,
    /// `wall`: route 2, when this tube has one.
    pub wall: Option<WallRoute>,
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
            direct_anchors: self.direct_anchors.clone(),
            landing: sh(&self.landing),
            exit: sh(&self.exit),
            exit_tile: (self.exit_tile.0 + dx, self.exit_tile.1 + dy),
            hall: self.hall.as_ref().map(|h| sh(h)),
            hall_tile: self.hall_tile.map(|t| (t.0 + dx, t.1 + dy)),
            toward: self.toward,
            wall: self.wall.as_ref().map(|w| w.shifted(dx, dy)),
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
/// `DIRECT_HOLDS` / `DIRECT_STEER`: the straight-into-the-passage swings, `(push, brake)` pairs.
const DIRECT_HOLDS: [i32; 2] = [32, 40];
const DIRECT_STEER: [(i32, i32); 8] = [(12, 0), (16, 3), (8, 0), (16, 6), (24, 6), (30, 6), (30, 10), (12, 3)];
/// `DIRECT_SAME_TICKS`: the smallest `push`: every direct swing of one anchor and hold sends the same
/// inputs up to `hold + DIRECT_SAME_TICKS`, so the rollouts share their first part.
pub const DIRECT_SAME_TICKS: i64 = 8;
pub const DIRECT_SETTLE_TICKS: i64 = 70;
const DIRECT_ENTER_TICKS: i64 = 50;
const DIRECT_ARRIVE_DEPTH_TILES: i32 = 7;
const DIRECT_WAIT_TICKS: i64 = 45;
const SPREAD_TICKS: i64 = 2;
const SPREAD_EARLY_TICKS: i64 = 1;
const NUDGE_PX: f64 = 6.0;
const NUDGE_Y_PX: f64 = 4.0;
const HOP_CLEAR_PX: f64 = 6.0;
const ARRIVED_VX: f64 = 3.0;
const APPROACH_DEPTH_TILES: i32 = 5;
const APPROACH_DEPTH_CLOCK_TILES: i32 = 3;
const APPROACH_GIVE_UP_TICKS: i64 = 150;
const HOOK_LENGTH_PX: f64 = 380.0;
const WALL_HOLDS: [i32; 3] = [8, 12, 18];
/// `WALL_STEER`: `(dir, push)`, the direction is a multiple of "out of the tube".
const WALL_STEER: [(i32, i32); 3] = [(1, 0), (1, 20), (0, 0)];
const WALL_DRIFT_TICKS: [i32; 4] = [2, 4, 6, 8];
const WALL_SPREAD_TICKS: i64 = 1;
const WALL_FALL_PX: f64 = 20.0;
const WALL_COPY_TICKS: i64 = 140;
const WALL_MIN_VX: f64 = 3.0;
const WALL_SETTLE_TICKS: i64 = 300;
const ROOM_RUNS: [i32; 5] = [6, 12, 24, 48, 80];
const ROOM_BRAKE_VX: [f64; 3] = [f64::INFINITY, 4.0, 1.5];
const ROOM_DROP_TICKS: i64 = 260;

/// A rope swing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Swing {
    pub anchor: (i32, i32),
    pub hold: i32,
    pub dir: i32,
    /// Ticks the direction is held after the rope is let go.
    pub push: i32,
    /// Ticks the direction is then held the other way.
    pub brake: i32,
    /// A swing straight into the passage (`directAnchors`).
    pub direct: bool,
    /// A swing through the wall (route 2).
    pub wall: bool,
    /// Ticks of drifting before the rope goes on (`ropeAt`; wall swings only).
    pub rope_at: i32,
    /// The direction held while drifting (`preDir`).
    pub pre_dir: i32,
}

impl Swing {
    fn rope_from(&self) -> i64 {
        i64::from(self.rope_at)
    }
    fn rope_to(&self) -> i64 {
        self.rope_from() + i64::from(self.hold)
    }
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
    /// A drop from the room floor into the shaft (route 2).
    pub room: bool,
    /// Speed above which the direction is reversed in the air (`brakeVx`; infinity = never).
    pub brake_vx: f64,
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

/// `wallRouteOk(col, c)`: the wall's rope anchors are hookable solid tiles and every shelf and room box
/// has a solid floor under it somewhere (another version of the map is refused).
pub fn wall_route_ok(col: &impl PlanCollision, c: &Crossing) -> bool {
    let Some(w) = &c.wall else { return false };
    for a in &w.anchors {
        if a.0 < 0 || a.1 < 0 || a.0 >= col.width() || a.1 >= col.height() {
            return false;
        }
        let x = f64::from(a.0 * TILE_PX) + f64::from(TILE_PX) / 2.0;
        let y = f64::from(a.1 * TILE_PX) + f64::from(TILE_PX) / 2.0;
        if !col.is_solid(x, y) || col.is_no_hook(x, y) {
            return false;
        }
    }
    for b in w.shelf.iter().chain(w.room.iter()) {
        if b.x0 < 0 || b.y0 < 0 || b.x1 >= col.width() || b.y1 + 1 >= col.height() {
            return false;
        }
        let mut floor = false;
        let mut x = b.x0;
        while x <= b.x1 && !floor {
            floor = col.is_solid(
                f64::from(x * TILE_PX) + f64::from(TILE_PX) / 2.0,
                f64::from((b.y1 + 1) * TILE_PX) + f64::from(TILE_PX) / 2.0,
            );
            x += 1;
        }
        if !floor {
            return false;
        }
    }
    true
}

#[derive(Debug, Clone, Copy)]
struct Lap {
    at: usize,
    tried: usize,
}

/// `Frame`: a rollout's state at one tick (the world, the input being held and the inputs on their way),
/// to resume from.
struct Frame<W: PlanWorld> {
    world: Option<W::SavedState>,
    t: i64,
    held: PlayerInput,
    pending: VecDeque<(i64, PlayerInput)>,
}

impl<W: PlanWorld> Frame<W> {
    fn empty() -> Frame<W> {
        Frame {
            world: None,
            t: -1,
            held: empty_input(),
            pending: VecDeque::new(),
        }
    }
}

/// The frames saved by [`SwingCrosser::search_direct`], by anchor and hold.
type FrameMap<W> = HashMap<(i32, i32, i32), Option<Frame<W>>>;

/// `Done`: `Some(true)` the rollout got there, `Some(false)` not yet, `None` it can no longer.
type Done<'a> = Box<dyn FnMut(&TeeState, i64) -> Option<bool> + 'a>;

/// What a rollout does besides rolling (`opts` of the TS `rollout`).
struct RollOpts<'a, W: PlanWorld> {
    resume: Option<&'a Frame<W>>,
    save: Option<&'a mut Frame<W>>,
    at: i64,
    wobble: Option<i64>,
}

impl<'a, W: PlanWorld> RollOpts<'a, W> {
    fn none() -> RollOpts<'a, W> {
        RollOpts {
            resume: None,
            save: None,
            at: -1,
            wobble: None,
        }
    }
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
    /// The inputs sent in the last ticks (`sent`), replayed into rollouts started in the air.
    sent: Vec<(i64, PlayerInput)>,
    now_tick: i64,
    cadence: i64,
    hops: i32,
    room_drops: i32,
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
    direct_empty: Cell<bool>,
    ran_out: bool,
    /// `useWall`: route 2 is on for this crossing.
    pub use_wall: bool,
    wall_thrown: bool,
    wall_dropped: bool,
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
            sent: Vec::new(),
            now_tick: 0,
            cadence: 1,
            hops: 0,
            room_drops: 0,
            phase: CrossPhase::Approach,
            why: String::new(),
            tried: Cell::new(0),
            budget_ms: 0.0,
            until: Cell::new(None),
            laps: RefCell::new(HashMap::new()),
            out_of_time: Cell::new(false),
            stops: Cell::new(0),
            direct_empty: Cell::new(false),
            ran_out: false,
            use_wall: false,
            wall_thrown: false,
            wall_dropped: false,
        }
    }

    pub fn phase(&self) -> CrossPhase {
        self.phase
    }
    pub fn done(&self) -> bool {
        matches!(self.phase, CrossPhase::Arrived | CrossPhase::Failed)
    }
    /// `thrown`: a program (a swing, a hop) is running.
    pub fn thrown(&self) -> bool {
        self.program.is_some()
    }
    /// `wallTaken`: route 2's swing through the wall was thrown.
    pub fn wall_taken(&self) -> bool {
        self.wall_thrown
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
                if h.room {
                    return format!(
                        "{}: walking to it at most {} ticks{} (route 2)",
                        h.what,
                        h.run,
                        if h.brake_vx == f64::INFINITY {
                            String::new()
                        } else {
                            format!(", then holding back above {} px/tick", h.brake_vx)
                        }
                    );
                }
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
            Move::Swing(s) => {
                if s.wall {
                    let steer = if s.dir == 0 {
                        ", swinging free".to_string()
                    } else {
                        format!(
                            ", pushing out{}",
                            if s.push > 0 {
                                format!(" and {} ticks after", s.push)
                            } else {
                                String::new()
                            }
                        )
                    };
                    let first = if s.rope_from() > 0 {
                        format!("drifting on {} ticks, then ", s.rope_from())
                    } else {
                        String::new()
                    };
                    return format!(
                        "{first}rope on ({},{}) for {} ticks{steer}, through the wall (route 2)",
                        s.anchor.0, s.anchor.1, s.hold
                    );
                }
                let pushed = if s.direct || s.brake > 0 {
                    format!(" {} ticks", s.push)
                } else {
                    String::new()
                };
                let steer = if s.dir == 0 {
                    ", swinging free".to_string()
                } else {
                    format!(
                        ", pushing on{pushed}{}",
                        if s.brake > 0 {
                            format!(", then back {}", s.brake)
                        } else {
                            String::new()
                        }
                    )
                };
                format!(
                    "rope on ({},{}) for {} ticks{steer}{}",
                    s.anchor.0,
                    s.anchor.1,
                    s.hold,
                    if s.direct { ", straight into the passage" } else { "" }
                )
            }
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

    /// `nearFreezeAhead`: like `near_freeze`, but only on our own column and the one toward the passage.
    fn near_freeze_ahead(&self, col: &W::Collision, me: &TeeState, px: f64) -> bool {
        let r = 14.0 + px;
        for dx in [0.0, f64::from(self.crossing.toward) * r] {
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

    fn wall_active(&self) -> bool {
        self.use_wall && self.crossing.wall.is_some()
    }

    fn in_room(&self, col: &W::Collision, me: &TeeState) -> bool {
        let Some(w) = &self.crossing.wall else { return false };
        !me.frozen
            && me.vel.x.abs() <= 1.0
            && self.supported(col, me)
            && in_any_box(&w.room, tile_of(me.pos.x), tile_of(me.pos.y))
    }

    /// `inWallWindow`: falling in the passage above the wall's last row, the rope off.
    fn in_wall_window(&self, col: &W::Collision, me: &TeeState) -> bool {
        let Some(w) = &self.crossing.wall else { return false };
        if me.frozen || me.vel.y < -1.0 || self.supported(col, me) || me.hook_state == HOOK_GRABBED {
            return false;
        }
        let ty = tile_of(me.pos.y);
        ty <= w.last_row && in_any_box(&self.crossing.exit, tile_of(me.pos.x), ty)
    }

    fn wall_stop(&self, col: &W::Collision, me: &TeeState, boxes: &[TileBox]) -> bool {
        me.vel.x.abs() <= 1.0
            && self.supported(col, me)
            && !self.in_freeze(col, me)
            && in_any_box(boxes, tile_of(me.pos.x), tile_of(me.pos.y))
    }

    /// `wallDone`: `Some(true)` on a shelf or the room floor, `None` once the swing missed.
    fn wall_done(&self, col: &W::Collision, me: &TeeState, far: i32) -> Option<bool> {
        let w = self.crossing.wall.as_ref().expect("wall route");
        let tx = tile_of(me.pos.x);
        let ty = tile_of(me.pos.y);
        let out = -self.crossing.toward;
        let from_column = (tx - w.anchors[0].0) * out;
        if from_column < 0 && ty >= w.miss_row {
            return None;
        }
        if me.frozen && (-2..=0).contains(&from_column) && me.vel.x * f64::from(out) < WALL_MIN_VX {
            return None;
        }
        if (tx - far) * out > 0 {
            return None;
        }
        if self.wall_stop(col, me, &w.shelf) || self.wall_stop(col, me, &w.room) {
            return Some(true);
        }
        if !me.frozen && me.vel.x.abs() <= 1.0 && self.supported(col, me) {
            return None;
        }
        Some(false)
    }

    fn wall_far(&self) -> i32 {
        let w = self.crossing.wall.as_ref().expect("wall route");
        if self.crossing.toward < 0 {
            w.room.iter().map(|b| b.x1).max().expect("room")
        } else {
            w.room.iter().map(|b| b.x0).min().expect("room")
        }
    }

    /// `shelfDone`: stopped on a shelf.
    fn shelf_done(&self, col: &W::Collision, me: &TeeState) -> Option<bool> {
        let w = self.crossing.wall.as_ref().expect("wall route");
        if self.wall_stop(col, me, &w.shelf) {
            return Some(true);
        }
        if !me.frozen
            && me.vel.x.abs() <= 1.0
            && self.supported(col, me)
            && !in_any_box(&w.room, tile_of(me.pos.x), tile_of(me.pos.y))
        {
            return None;
        }
        Some(false)
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

    /// `step(self, tick, lag)`: decides, then remembers the input sent (rollouts replay it).
    pub fn step(&mut self, col: &W::Collision, me: &TeeState, tick: i64, lag: i64) -> PlayerInput {
        self.now_tick = tick;
        if self.sent.last().is_some_and(|s| tick < s.0) {
            self.sent.clear();
        }
        let input = self.decide(col, me, tick, lag);
        self.sent.push((tick, input));
        let keep_from = tick - (lag + SPREAD_TICKS + self.cadence);
        let drop = self.sent.iter().take_while(|s| s.0 < keep_from).count();
        self.sent.drain(..drop);
        input
    }

    fn decide(&mut self, col: &W::Collision, me: &TeeState, tick: i64, lag: i64) -> PlayerInput {
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
            let on_shelf = self.wall_thrown
                && self
                    .crossing
                    .wall
                    .as_ref()
                    .is_some_and(|w| in_any_box(&w.shelf, tile_of(me.pos.x), tile_of(me.pos.y)));
            self.why = format!(
                "through {}{}",
                self.crossing.label,
                if !on_shelf {
                    ""
                } else if self.wall_dropped {
                    ", by route 1's drop (the swing through the wall was given up)"
                } else {
                    " and the wall (route 2)"
                }
            );
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
                return self.program_input(col, me, p.mv, tick - p.start_tick, input);
            }
            return input;
        }

        if self.wall_active()
            && self.in_wall_window(col, me)
            && !matches!(self.program, Some(Program { mv: Move::Swing(s), .. }) if s.wall)
            && let Some(wall) = self.search_wall(col, me, lag)
        {
            let p = Program {
                mv: Move::Swing(wall),
                start_tick: tick,
                froze: false,
            };
            self.program = Some(p);
            self.phase = CrossPhase::Swinging;
            self.wall_thrown = true;
            self.wall_dropped = false;
            return self.program_input(col, me, p.mv, 0, input);
        }

        if let Some(p) = self.program
            && !self.supported(col, me)
        {
            let t = tick - p.start_tick;
            let roped = matches!(p.mv, Move::Swing(s) if t >= s.rope_from() && t < s.rope_to())
                && me.hook_state == HOOK_GRABBED;
            if matches!(p.mv, Move::Swing(s) if s.wall) {
                if t > 0 && !self.works(col, me, p.mv, t, lag) {
                    if let Some(again) = self.search_wall(col, me, lag) {
                        self.program = Some(Program {
                            mv: Move::Swing(again),
                            start_tick: tick,
                            froze: p.froze,
                        });
                        self.wall_dropped = false;
                    } else {
                        let drop = if self.out_of_time.get() {
                            None
                        } else {
                            self.search_drop(col, me, lag)
                        };
                        if let Some(drop) = drop {
                            self.program = Some(Program {
                                mv: Move::Hop(drop),
                                start_tick: tick,
                                froze: p.froze,
                            });
                            self.wall_dropped = true;
                        }
                    }
                }
            } else if !roped
                && t > 0
                && !matches!(p.mv, Move::Hop(h) if h.room)
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
                    let settle = if s.wall { WALL_SETTLE_TICKS } else { SETTLE_TICKS };
                    if (!p.froze && t > s.rope_from() + 12 && t < s.rope_to() && me.hook_state != HOOK_GRABBED)
                        || (t >= s.rope_to()
                            && (self.landed_at(col, me) || self.in_passage(me) || (s.wall && self.in_room(col, me))))
                    {
                        self.program = None;
                    } else if t > s.rope_to() + i64::from(s.push) + i64::from(s.brake) + settle {
                        return self.fail(format!("the swing ran out at ({tx},{ty})"));
                    } else if t < s.rope_to() + i64::from(s.push) + i64::from(s.brake) || !self.in_from(tx, ty) {
                        return self.program_input(col, me, p.mv, t, input);
                    } else {
                        self.program = None;
                    }
                }
                Move::Hop(h) => {
                    let dropping = h.ticks > HOP_TICKS;
                    // Done: standing on the room floor after the run (a drop from it), or landed / in the
                    // passage after a hop.
                    if (h.room && t > i64::from(h.run) + 8 && self.in_room(col, me))
                        || (t > 4 && !dropping && (self.landed_at(col, me) || self.in_passage(me)))
                    {
                        self.program = None;
                    } else if t > h.ticks {
                        return self.fail(format!(
                            "the {} ran out at ({tx},{ty})",
                            if dropping { "drop" } else { "hop" }
                        ));
                    } else {
                        return self.program_input(col, me, p.mv, t, input);
                    }
                }
            }
        }

        if self.wall_active() && self.in_room(col, me) {
            if self.room_drops >= MAX_HOPS {
                return self.fail(format!(
                    "{} drops from the room floor and still at ({tx},{ty})",
                    self.room_drops
                ));
            }
            let drop = self.search_room_drop(col, me, lag);
            if drop.is_none() && self.out_of_time.get() {
                return input;
            }
            let Some(drop) = drop else {
                return self.fail(format!("no drop from the room floor at ({tx},{ty}) into the shaft"));
            };
            self.room_drops += 1;
            let p = Program {
                mv: Move::Hop(drop),
                start_tick: tick,
                froze: false,
            };
            self.program = Some(p);
            self.phase = CrossPhase::Hopping;
            return self.program_input(col, me, p.mv, 0, input);
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
            return self.program_input(col, me, p.mv, 0, input);
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
            return self.program_input(col, me, p.mv, 0, input);
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
            let give_way = if self.budget_ms > 0.0 {
                APPROACH_DEPTH_CLOCK_TILES
            } else {
                APPROACH_DEPTH_TILES
            };
            if tx < ch.x0 - 3 || tx > ch.x1 + 3 {
                self.approach_dir = if tx < ch.x0 { 1 } else { -1 };
            } else if depth < give_way && (self.supported(col, me) || self.approach_dir == -toward) {
                self.approach_dir = -toward;
            } else {
                self.approach_dir = toward;
            }
            if self.approach_dir == -toward {
                self.ran_out = true;
            }
        }
        if let Some(found) = self.search_swing(col, me, lag, tick - self.start_tick) {
            let p = Program {
                mv: Move::Swing(found),
                start_tick: tick,
                froze: false,
            };
            self.program = Some(p);
            self.phase = CrossPhase::Swinging;
            return self.program_input(col, me, p.mv, 0, input);
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

    fn program_input(
        &self,
        col: &W::Collision,
        me: &TeeState,
        mv: Move,
        t: i64,
        mut input: PlayerInput,
    ) -> PlayerInput {
        let toward = self.crossing.toward;
        match mv {
            Move::Hop(p) => {
                if p.room {
                    let on_floor = self.supported(col, me);
                    input.direction = if on_floor {
                        if t < i64::from(p.run) { p.dir } else { 0 }
                    } else if me.vel.x * f64::from(p.dir) > p.brake_vx {
                        -p.dir
                    } else {
                        0
                    };
                } else {
                    input.direction = if t < i64::from(p.run) { p.dir } else { 0 };
                }
                input.jump =
                    i32::from(p.jump_at >= 0 && t >= i64::from(p.jump_at) && t < i64::from(p.jump_at + p.jump_hold));
                input.hook = 0;
                input.target_x = f64::from(toward) * 300.0;
                input.target_y = 0.0;
                input
            }
            Move::Swing(p) => {
                let holding = t >= p.rope_from() && t < p.rope_to();
                input.hook = i32::from(holding);
                input.jump = 0;
                input.direction = if t < p.rope_from() {
                    p.pre_dir
                } else if t < p.rope_to() + i64::from(p.push) {
                    p.dir
                } else if t < p.rope_to() + i64::from(p.push) + i64::from(p.brake) {
                    -p.dir
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

    /// `robust(self, p, lag, doneFor, ticks, approaching, resume, late)`: does the program get through
    /// with the lag a little off, the tee nudged, and (for a swing through the wall) the inputs a tick
    /// early or late? `done_for(d)` makes the `done` of the rollout run with lag `d`.
    #[allow(clippy::too_many_arguments)]
    fn robust<'a>(
        &self,
        col: &W::Collision,
        me: &TeeState,
        mv: Move,
        lag: i64,
        done_for: &dyn Fn(i64) -> Done<'a>,
        ticks: i64,
        approaching: bool,
        resume: Option<&Frame<W>>,
        late: i64,
    ) -> bool {
        let mut lags = vec![lag];
        let mut d = (lag - SPREAD_EARLY_TICKS).max(0);
        while d <= lag + late {
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
                let opts = RollOpts {
                    resume: if d == lag { resume } else { None },
                    ..RollOpts::none()
                };
                self.rollout(col, sim, mv, d, done_for(d), ticks, approaching, 0, opts)
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
                self.rollout(
                    col,
                    sim,
                    mv,
                    lag,
                    done_for(lag),
                    ticks,
                    approaching,
                    0,
                    RollOpts::none(),
                )
            });
            if !ok {
                return false;
            }
        }
        if matches!(mv, Move::Swing(s) if s.wall) {
            for wobble in [1i64, -1] {
                if wobble < 0 && lag == 0 {
                    continue;
                }
                let ok = self.with_sim(me, |sim| {
                    sim.apply_tee_state(0, &with_id);
                    let opts = RollOpts {
                        wobble: Some(wobble),
                        ..RollOpts::none()
                    };
                    self.rollout(col, sim, mv, lag, done_for(lag), ticks, approaching, 0, opts)
                });
                if !ok {
                    return false;
                }
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

    /// `directDone(ropeLeft)(d)`: the `done` of a swing straight into the passage (stateful: has the tee
    /// entered the passage yet).
    fn direct_done<'a>(&'a self, col: &'a W::Collision, rope_left: i64) -> impl Fn(i64) -> Done<'a> + 'a {
        let top = self.crossing.exit.iter().map(|b| b.y0).min().expect("exit boxes");
        move |d| {
            let release_at = d + rope_left;
            let mut entered = false;
            Box::new(move |me: &TeeState, t: i64| {
                if t == 0 {
                    entered = false;
                }
                let tx = tile_of(me.pos.x);
                let ty = tile_of(me.pos.y);
                if !entered {
                    if !me.frozen
                        && self.supported(col, me)
                        && (in_any_box(&self.crossing.landing, tx, ty) || t > release_at + 2)
                    {
                        return None;
                    }
                    if !me.frozen
                        && in_any_box(&self.crossing.landing, tx, ty)
                        && self.near_freeze_ahead(col, me, HOP_CLEAR_PX)
                    {
                        return None;
                    }
                    if !in_any_box(&self.crossing.exit, tx, ty) {
                        if t > release_at + DIRECT_ENTER_TICKS {
                            return None;
                        }
                        for b in &self.crossing.exit {
                            if (tx - b.x0) * self.crossing.toward > 0 && (tx - b.x1) * self.crossing.toward > 0 {
                                return None;
                            }
                        }
                        return Some(false);
                    }
                    entered = true;
                }
                if me.frozen || self.near_freeze(col, me, HOP_CLEAR_PX) {
                    return None;
                }
                if ty > top + DIRECT_ARRIVE_DEPTH_TILES {
                    return None;
                }
                Some(me.vel.x.abs() <= ARRIVED_VX)
            })
        }
    }

    fn search_direct(&self, col: &W::Collision, me: &TeeState, lag: i64) -> Option<Swing> {
        self.direct_empty.set(self.crossing.direct_anchors.is_empty());
        let toward = self.crossing.toward;
        let ch = self.crossing.chamber;
        if (tile_of(me.pos.x) - if toward < 0 { ch.x0 } else { ch.x1 }) * -toward < 0 {
            return None;
        }
        let mut list: Vec<Swing> = Vec::new();
        for &i in &self.crossing.direct_anchors {
            let Some(&anchor) = self.crossing.anchors.get(i) else {
                continue;
            };
            let reach = ddai_jsmath::hypot2(centre(anchor.0) - me.pos.x, centre(anchor.1) - me.pos.y);
            if reach > HOOK_LENGTH_PX + (lag + SPREAD_TICKS) as f64 * 16.0 {
                continue;
            }
            for hold in DIRECT_HOLDS {
                for (push, brake) in DIRECT_STEER {
                    list.push(Swing {
                        anchor,
                        hold,
                        dir: toward,
                        push,
                        brake,
                        direct: true,
                        wall: false,
                        rope_at: 0,
                        pre_dir: 0,
                    });
                }
            }
        }
        self.direct_empty.set(list.is_empty());

        // The part of the swing before the steering differs is the same for all of one anchor and
        // hold: it is rolled once and every candidate resumes from there.
        let frames: RefCell<FrameMap<W>> = RefCell::new(HashMap::new());
        self.first_that("direct", &list, &|p: &Swing| {
            let key = (p.anchor.0, p.anchor.1, p.hold);
            if !frames.borrow().contains_key(&key) {
                let mut save = Frame::<W>::empty();
                let mut with_id = *me;
                with_id.id = 0;
                self.with_sim(me, |sim| {
                    sim.apply_tee_state(0, &with_id);
                    let opts = RollOpts {
                        save: Some(&mut save),
                        at: i64::from(p.hold) + DIRECT_SAME_TICKS,
                        ..RollOpts::none()
                    };
                    let _ = self.rollout(
                        col,
                        sim,
                        Move::Swing(*p),
                        lag,
                        self.direct_done(col, i64::from(p.hold))(lag),
                        lag + i64::from(p.hold) + DIRECT_SAME_TICKS + 1,
                        true,
                        0,
                        opts,
                    );
                });
                let frame = if save.t < 0 { None } else { Some(save) };
                frames.borrow_mut().insert(key, frame);
            }
            let frames = frames.borrow();
            let Some(frame) = frames.get(&key).expect("frame").as_ref() else {
                return false;
            };
            let done_for = self.direct_done(col, i64::from(p.hold));
            self.robust(
                col,
                me,
                Move::Swing(*p),
                lag,
                &done_for,
                lag + i64::from(p.hold) + DIRECT_SETTLE_TICKS,
                true,
                Some(frame),
                SPREAD_TICKS,
            )
        })
    }

    fn search_swing(&self, col: &W::Collision, me: &TeeState, lag: i64, waited: i64) -> Option<Swing> {
        let turned_back = self.ran_out && self.approach_dir == self.crossing.toward;
        let late = waited >= DIRECT_WAIT_TICKS || turned_back;
        if self.budget_ms > 0.0 && late {
            let old = self.search_landing(col, me, lag);
            if old.is_some() || self.out_of_time.get() {
                return old;
            }
            return self.search_direct(col, me, lag);
        }
        let direct = self.search_direct(col, me, lag);
        if direct.is_some() || self.out_of_time.get() {
            return direct;
        }
        if !self.direct_empty.get() && !late {
            return None;
        }
        self.search_landing(col, me, lag)
    }

    fn search_landing(&self, col: &W::Collision, me: &TeeState, lag: i64) -> Option<Swing> {
        let toward = self.crossing.toward;
        let mut list: Vec<Swing> = Vec::new();
        for &anchor in &self.crossing.anchors {
            let reach = ddai_jsmath::hypot2(centre(anchor.0) - me.pos.x, centre(anchor.1) - me.pos.y);
            if reach > HOOK_LENGTH_PX + (lag + SPREAD_TICKS) as f64 * 16.0 {
                continue;
            }
            for hold in HOLDS {
                for dir in [toward, 0] {
                    list.push(Swing {
                        anchor,
                        hold,
                        dir,
                        push: PUSH_AFTER_TICKS as i32,
                        brake: 0,
                        direct: false,
                        wall: false,
                        rope_at: 0,
                        pre_dir: 0,
                    });
                }
            }
        }
        let done_for = |_d: i64| -> Done<'_> {
            Box::new(|m: &TeeState, _t: i64| Some(self.through_at(col, m) || self.landed_at(col, m)))
        };
        self.first_that("swing", &list, &|p: &Swing| {
            self.robust(
                col,
                me,
                Move::Swing(*p),
                lag,
                &done_for,
                lag + i64::from(p.hold) + SETTLE_TICKS,
                true,
                None,
                SPREAD_TICKS,
            )
        })
    }

    fn search_hop(&self, col: &W::Collision, me: &TeeState, lag: i64, in_air: bool) -> Option<Hop> {
        let toward = self.crossing.toward;
        let from = me.pos.x;
        let done_for = move |_d: i64| -> Done<'_> {
            Box::new(move |m: &TeeState, _t: i64| {
                Some(if in_air {
                    self.through_at(col, m) || self.landed_at(col, m)
                } else {
                    self.through_at(col, m)
                        || (self.landed_at(col, m) && (m.pos.x - from) * f64::from(toward) > f64::from(TILE_PX))
                })
            })
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
                                room: false,
                                brake_vx: f64::INFINITY,
                            });
                        }
                    }
                }
            }
        }
        self.first_that(if in_air { "air" } else { "hop" }, &list, &|p: &Hop| {
            self.robust(
                col,
                me,
                Move::Hop(*p),
                lag,
                &done_for,
                lag + HOP_TICKS,
                false,
                None,
                SPREAD_TICKS,
            )
        })
    }

    fn search_drop(&self, col: &W::Collision, me: &TeeState, lag: i64) -> Option<Hop> {
        let done_for = |_d: i64| -> Done<'_> { Box::new(|m: &TeeState, _t: i64| Some(self.arrived_at(col, m))) };
        let toward = self.crossing.toward;
        let what = if self.wall_active() && !self.wall_thrown {
            "drop into the hall (route 2: no swing through the wall from here)"
        } else {
            "drop into the hall"
        };
        let mut list: Vec<Hop> = Vec::new();
        for run in DROP_RUNS {
            let dirs: Vec<i32> = if run == 0 { vec![0] } else { vec![-toward, toward] };
            for dir in dirs {
                list.push(Hop {
                    what,
                    dir,
                    run,
                    jump_at: -1,
                    jump_hold: 0,
                    ticks: DROP_TICKS,
                    room: false,
                    brake_vx: f64::INFINITY,
                });
            }
        }
        self.first_that("drop", &list, &|p: &Hop| {
            self.robust(
                col,
                me,
                Move::Hop(*p),
                lag,
                &done_for,
                lag + DROP_TICKS,
                false,
                None,
                SPREAD_TICKS,
            )
        })
    }

    /// `searchWall`: a swing on the wall column that carries the tee out through the wall onto a shelf.
    fn search_wall(&self, col: &W::Collision, me: &TeeState, lag: i64) -> Option<Swing> {
        let w = self.crossing.wall.as_ref().expect("wall route");
        let out = -self.crossing.toward;
        let mut list: Vec<Swing> = Vec::new();
        for &anchor in &w.anchors {
            let reach = ddai_jsmath::hypot2(centre(anchor.0) - me.pos.x, centre(anchor.1) - me.pos.y);
            for rope_at in std::iter::once(0).chain(WALL_DRIFT_TICKS) {
                if reach
                    > HOOK_LENGTH_PX + (lag + WALL_SPREAD_TICKS + i64::from(rope_at)) as f64 * WALL_FALL_PX + NUDGE_PX
                {
                    continue;
                }
                for hold in WALL_HOLDS {
                    for (dir, push) in WALL_STEER {
                        list.push(Swing {
                            anchor,
                            hold,
                            dir: dir * out,
                            push,
                            brake: 0,
                            direct: false,
                            wall: true,
                            rope_at,
                            pre_dir: if rope_at > 0 { -out } else { 0 },
                        });
                    }
                }
            }
        }
        let far = self.wall_far();
        let done_for =
            move |_d: i64| -> Done<'_> { Box::new(move |m: &TeeState, _t: i64| self.wall_done(col, m, far)) };
        self.first_that("wall", &list, &|p: &Swing| {
            self.robust(
                col,
                me,
                Move::Swing(*p),
                lag,
                &done_for,
                lag + p.rope_to() + WALL_COPY_TICKS,
                false,
                None,
                WALL_SPREAD_TICKS,
            )
        })
    }

    /// `searchRoomDrop`: from the room floor into the shaft, onto a shelf.
    fn search_room_drop(&self, col: &W::Collision, me: &TeeState, lag: i64) -> Option<Hop> {
        let toward = self.crossing.toward;
        let mut list: Vec<Hop> = Vec::new();
        for brake_vx in ROOM_BRAKE_VX {
            for run in ROOM_RUNS {
                list.push(Hop {
                    what: "drop into the shaft",
                    dir: toward,
                    run,
                    jump_at: -1,
                    jump_hold: 0,
                    ticks: ROOM_DROP_TICKS,
                    room: true,
                    brake_vx,
                });
            }
        }
        let done_for = |_d: i64| -> Done<'_> { Box::new(|m: &TeeState, _t: i64| self.shelf_done(col, m)) };
        self.first_that("room", &list, &|p: &Hop| {
            self.robust(
                col,
                me,
                Move::Hop(*p),
                lag,
                &done_for,
                lag + ROOM_DROP_TICKS,
                false,
                None,
                SPREAD_TICKS,
            )
        })
    }

    /// `works(self, p, t, lag)`: does the running program still get through from here?
    fn works(&self, col: &W::Collision, me: &TeeState, mv: Move, t: i64, lag: i64) -> bool {
        let dropping = matches!(mv, Move::Hop(h) if h.ticks > HOP_TICKS);
        let far = match mv {
            Move::Swing(s) if s.wall => self.wall_far(),
            _ => 0,
        };
        let done: Done<'_> = match mv {
            Move::Swing(s) if s.direct => self.direct_done(col, i64::from(s.hold) - t)(lag),
            Move::Swing(s) if s.wall => Box::new(move |m: &TeeState, _t: i64| self.wall_done(col, m, far)),
            Move::Hop(h) if h.room => Box::new(move |m: &TeeState, _t: i64| self.shelf_done(col, m)),
            _ => Box::new(move |m: &TeeState, _t: i64| {
                Some(if dropping {
                    self.arrived_at(col, m)
                } else {
                    self.through_at(col, m) || self.landed_at(col, m)
                })
            }),
        };
        let settle = match mv {
            Move::Swing(s) if s.direct => DIRECT_SETTLE_TICKS,
            Move::Swing(s) if s.wall => WALL_COPY_TICKS,
            _ => SETTLE_TICKS,
        };
        let ticks = match mv {
            Move::Swing(s) => (lag + 20).max(lag + s.rope_to() + settle - t),
            Move::Hop(h) => (lag + 20).max(lag + h.ticks - t),
        };
        let mut with_id = *me;
        with_id.id = 0;
        self.with_sim(me, |sim| {
            sim.apply_tee_state(0, &with_id);
            self.rollout(col, sim, mv, lag, done, ticks, false, t, RollOpts::none())
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn rollout(
        &self,
        col: &W::Collision,
        sim: &mut W,
        mv: Move,
        lag: i64,
        mut done: Done<'_>,
        ticks: i64,
        approaching: bool,
        t0: i64,
        mut opts: RollOpts<'_, W>,
    ) -> bool {
        self.tried.set(self.tried.get() + 1);
        let mut held = if approaching {
            self.approach_input(empty_input())
        } else {
            empty_input()
        };
        let mut pending: VecDeque<(i64, PlayerInput)> = VecDeque::new();
        // Inputs already sent but not yet applied by the server are part of the start state.
        if !self.sent.is_empty()
            && (approaching
                || t0 > 0
                || matches!(mv, Move::Hop(h) if h.ticks > HOP_TICKS)
                || matches!(mv, Move::Swing(s) if s.wall))
        {
            for s in &self.sent {
                let at = s.0 + lag - self.now_tick;
                if at <= 0 {
                    held = s.1;
                } else {
                    pending.push_back((at, s.1));
                }
            }
        }
        let mut from = 0;
        if let Some(resume) = opts.resume
            && let Some(world) = &resume.world
        {
            sim.restore_state(world);
            held = resume.held;
            pending = resume.pending.clone();
            from = resume.t;
        }
        let mut t = from;
        while t < ticks {
            if t == opts.at
                && let Some(save) = opts.save.as_mut()
            {
                save.world = Some(sim.save_state());
                save.t = t;
                save.held = held;
                save.pending = pending.clone();
            }
            let Some(me) = sim.get_tee(0) else { return false };
            if !me.alive {
                return false;
            }
            if t % self.cadence == 0 {
                let wobble = match opts.wobble {
                    Some(w) if (t / self.cadence) % 2 == 1 => w,
                    _ => 0,
                };
                let at = if wobble == 0 {
                    t + lag
                } else {
                    (t + (lag + wobble).max(0)).max(pending.back().map_or(0, |p| p.0))
                };
                pending.push_back((at, self.program_input(col, &me, mv, t0 + t, empty_input())));
            }
            while pending.front().is_some_and(|p| p.0 <= t) {
                held = pending.pop_front().expect("front").1;
            }
            sim.set_input(0, held);
            let _ = sim.step();
            let Some(now) = sim.get_tee(0) else { return false };
            let Some(over) = done(&now, t) else { return false };
            if t > lag + 4 && over {
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
            t += 1;
        }
        false
    }
}
