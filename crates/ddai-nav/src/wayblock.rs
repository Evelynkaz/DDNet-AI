//! `src/bot/wayblock.ts` (upstream `af49dfb`, release 2026-10-02): the wayblock (WB) of `Copy Love Box`:
//! hard-coded zones, spots, watch points, rope anchors, tubes and the tubes' walls (route 2), the side
//! chooser and the spot choice. All numbers are the TS ones. `wayblock_for` knows the map by its name
//! and size (the original, JoniTee shifted by `(182, 212)` on 600x600) **or by its tiles**: the hall is
//! searched for in any map (`find_hall_offset`) and the whole definition shifted to where it is found
//! (the Swarfey version is 468x255, `Copy Love Box IN` another).

use ddai_planner::plan_world::PlanCollision;
use ddai_planner::types::TeeState;
use std::sync::OnceLock;

use crate::crossing::{Crossing, TileBox, WallRoute, in_any_box, wall_route_ok};

/// `WbSide`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WbSide {
    Left,
    Right,
}

impl WbSide {
    pub fn other(self) -> WbSide {
        match self {
            WbSide::Left => WbSide::Right,
            WbSide::Right => WbSide::Left,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            WbSide::Left => "left",
            WbSide::Right => "right",
        }
    }
}

/// `WbSideDef`.
#[derive(Debug, Clone, PartialEq)]
pub struct WbSideDef {
    pub zone: Vec<TileBox>,
    pub approach: Vec<TileBox>,
    pub leash: Vec<TileBox>,
    pub spots: Vec<(i32, i32)>,
    pub watch: (i32, i32),
    pub crossing: Crossing,
}

/// `WbDef`.
#[derive(Debug, Clone, PartialEq)]
pub struct WbDef {
    pub name: String,
    pub left: WbSideDef,
    pub right: WbSideDef,
    /// Zones the walk and the trek must stay out of.
    pub avoid: Vec<TileBox>,
    pub crossings: Vec<Crossing>,
    pub size: (i32, i32),
}

pub const WB_LEASH_TILES: i32 = 3;
pub const WB_SWITCH_MARGIN: i32 = 2;
/// `WB_SIDE_HOPPING`: the side is never given up for the other one.
pub const WB_SIDE_HOPPING: bool = false;
pub const WB_PROVISIONAL_TICKS: i64 = 5 * 50;
pub const WB_SWITCH_TICKS: i64 = 5 * 50;
/// Task 3.12 (`--wb-smart`): after a side change the side is kept at least this long (a change costs a walk through a freeze tube).
pub const WB_SMART_SWITCH_COOLDOWN_TICKS: i64 = 40 * 50;
/// Task 3.12: a failed crossing of a side's tube counts against that side for this long, and each one is worth
/// [`WB_FAIL_PENALTY`] blockable targets.
pub const WB_FAIL_MEMORY_TICKS: i64 = 3 * 60 * 50;
pub const WB_FAIL_PENALTY: i32 = 2;
/// Task 3.12 (`--wb-smart`): until this long after the first look at the tees -- and until we first stand in a hall -- the side is only
/// a provisional pick, taken again at every look, with no cooldown. (Who is AFK is not known earlier: the activity clock needs 10 s of
/// unchanged input to call a tee idle, and until then only the tees inside a hall count as blockable.)
pub const WB_SMART_WARMUP_TICKS: i64 = 12 * 50;
/// Task 3.12 review F6: during the warm-up a **different** side than the one picked must lead this long without a break before the
/// pick changes (a count that flips by one at a hall boundary must not flip the side, and cancel the walk, every look).
pub const WB_SMART_WARMUP_HOLD_TICKS: i64 = 50;
/// Task 3.12 review F11: a fight (`WbState::fight_here`) counts for this long after it was last seen, so that a fight test that
/// flickers (a target at the edge of its radius, a hook that goes on and off) neither flaps a committed switch nor cancels it on a blink.
pub const WB_FIGHT_HOLD_TICKS: i64 = 75;
/// `WB_NO_CLIMB_TILES` (`bot.ts:222`).
pub const WB_NO_CLIMB_TILES: i32 = 3;
/// `WB_ZONE_SCORE` (`bot.ts:228`): the target-score bonus for a tee in the WB zone.
pub const WB_ZONE_SCORE: f64 = 300.0;

/// `WB_GUARD` (`wayblock.ts`; on unless the environment variable `DDAI_WB_GUARD` is `0`, TS: `WB_GUARD`):
/// the new spots (the left end of the upper shelf first) and the guard's behaviour in the bot.
pub fn wb_guard() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("DDAI_WB_GUARD").map(|v| v != "0").unwrap_or(true))
}

const MIRROR: i32 = 234;

fn grow(b: &TileBox, n: i32) -> TileBox {
    TileBox::new(b.x0 - n, b.y0 - n, b.x1 + n, b.y1 + n)
}

fn mirror_box(b: &TileBox) -> TileBox {
    TileBox::new(MIRROR - b.x1, b.y0, MIRROR - b.x0, b.y1)
}

fn side(
    zone: Vec<TileBox>,
    approach: Vec<TileBox>,
    spots: Vec<(i32, i32)>,
    watch: (i32, i32),
    crossing: Crossing,
) -> WbSideDef {
    let mut leash: Vec<TileBox> = zone.iter().map(|b| grow(b, WB_LEASH_TILES)).collect();
    leash.extend(approach.iter().copied());
    WbSideDef {
        zone,
        approach,
        leash,
        spots,
        watch,
        crossing,
    }
}

fn left_tube() -> Crossing {
    let from = vec![TileBox::new(103, 29, 131, 35), TileBox::new(107, 36, 127, 50)];
    Crossing {
        label: "the left freeze tube".to_string(),
        from,
        chamber: TileBox::new(108, 36, 126, 50),
        start: (104, 35),
        anchors: vec![
            (107, 37),
            (107, 38),
            (107, 39),
            (105, 40),
            (106, 40),
            (107, 40),
            (103, 39),
            (102, 38),
        ],
        direct_anchors: vec![7, 0, 6],
        landing: vec![TileBox::new(95, 41, 103, 50)],
        exit: vec![TileBox::new(87, 51, 92, 63), TileBox::new(87, 64, 91, 65)],
        exit_tile: (89, 60),
        hall: Some(vec![TileBox::new(79, 67, 104, 79), TileBox::new(78, 79, 104, 87)]),
        hall_tile: Some((90, 79)),
        toward: -1,
        wall: Some(left_wall()),
    }
}

/// `LEFT_WALL`: route 2 of the left tube.
fn left_wall() -> WallRoute {
    WallRoute {
        anchors: vec![(95, 56), (95, 55), (95, 54), (95, 57), (95, 53)],
        shelf: vec![TileBox::new(93, 80, 104, 84)],
        room: vec![TileBox::new(99, 59, 107, 63)],
        last_row: 58,
        miss_row: 64,
    }
}

fn right_tube(left: &Crossing) -> Crossing {
    Crossing {
        label: "the right freeze tube".to_string(),
        from: left.from.clone(),
        chamber: left.chamber,
        start: (130, 35),
        anchors: left.anchors.iter().map(|a| (MIRROR - a.0, a.1)).collect(),
        direct_anchors: left.direct_anchors.clone(),
        landing: left.landing.iter().map(mirror_box).collect(),
        exit: left.exit.iter().map(mirror_box).collect(),
        exit_tile: (MIRROR - left.exit_tile.0, left.exit_tile.1),
        hall: Some(left.hall.as_ref().expect("left hall").iter().map(mirror_box).collect()),
        hall_tile: Some((MIRROR - 90, 79)),
        toward: 1,
        wall: left.wall.as_ref().map(|w| WallRoute {
            anchors: w.anchors.iter().map(|a| (MIRROR - a.0, a.1)).collect(),
            shelf: w.shelf.iter().map(mirror_box).collect(),
            room: w.room.iter().map(mirror_box).collect(),
            last_row: w.last_row,
            miss_row: w.miss_row,
        }),
    }
}

fn copy_love_box() -> WbDef {
    let left_tube = left_tube();
    let right_tube = right_tube(&left_tube);
    let l1 = TileBox::new(79, 67, 104, 79);
    let l2 = TileBox::new(78, 79, 104, 87);
    let left_approach = TileBox::new(84, 41, 103, 66);
    let (left_spots, right_spots) = if wb_guard() {
        (
            vec![(83, 79), (94, 84), (101, 84)],
            vec![(MIRROR - 83, 79), (MIRROR - 94, 84), (MIRROR - 101, 84)],
        )
    } else {
        (
            vec![(94, 84), (82, 79), (101, 84)],
            vec![(152, 79), (MIRROR - 94, 84), (MIRROR - 101, 84)],
        )
    };
    let left_watch = (89, 79);
    WbDef {
        name: "Copy Love Box".to_string(),
        left: side(
            vec![l1, l2],
            vec![left_approach],
            left_spots,
            left_watch,
            left_tube.clone(),
        ),
        right: side(
            vec![mirror_box(&l1), mirror_box(&l2)],
            vec![mirror_box(&left_approach)],
            right_spots,
            (MIRROR - left_watch.0, left_watch.1),
            right_tube.clone(),
        ),
        avoid: vec![TileBox::new(96, 12, 140, 24)],
        crossings: vec![left_tube, right_tube],
        size: (387, 250),
    }
}

fn shift_side(s: &WbSideDef, dx: i32, dy: i32) -> WbSideDef {
    let sh = |v: &[TileBox]| v.iter().map(|b| b.shifted(dx, dy)).collect::<Vec<_>>();
    WbSideDef {
        zone: sh(&s.zone),
        approach: sh(&s.approach),
        leash: sh(&s.leash),
        spots: s.spots.iter().map(|p| (p.0 + dx, p.1 + dy)).collect(),
        watch: (s.watch.0 + dx, s.watch.1 + dy),
        crossing: s.crossing.shifted(dx, dy),
    }
}

fn shift_def(d: &WbDef, name: String, dx: i32, dy: i32, size: (i32, i32)) -> WbDef {
    let left = shift_side(&d.left, dx, dy);
    let right = shift_side(&d.right, dx, dy);
    WbDef {
        name,
        crossings: vec![left.crossing.clone(), right.crossing.clone()],
        left,
        right,
        avoid: d.avoid.iter().map(|b| b.shifted(dx, dy)).collect(),
        size,
    }
}

/// `WAYBLOCKS`.
pub fn wayblocks() -> Vec<WbDef> {
    let clb = copy_love_box();
    let joni = shift_def(&clb, "Copy Love Box JoniTee".to_string(), 182, 212, (600, 600));
    vec![clb, joni]
}

/// `standable(col, tx, ty)`: a free tile with solid ground below.
pub fn standable(col: &impl PlanCollision, tx: i32, ty: i32) -> bool {
    if tx < 0 || ty < 0 || tx >= col.width() || ty + 1 >= col.height() {
        return false;
    }
    let px = f64::from(tx * 32 + 16);
    let py = f64::from(ty * 32 + 16);
    if col.is_solid(px, py) || col.is_freeze(px, py) || col.is_death(px, py) {
        return false;
    }
    col.is_solid(px, py + 32.0)
}

/// `HALL_X0` / `HALL_Y0`: where the core of the hall sits in the original 387x250 map (tiles).
const HALL_X0: i32 = 76;
const HALL_Y0: i32 = 64;

/// `HALL_CORE`: the hall of `Copy Love Box` (`#` solid, `~` freeze, `X` death, `.` anything else), the
/// rows from `HALL_Y0` and the columns from `HALL_X0` of the original map.
const HALL_CORE: [&str; 29] = [
    ".......#~~~.....~~~~....###############~~~~~###############....~~~~.....~~~#.......",
    ".......#~~~.....~~~#....#~~~~~~~~~~~...........~~~~~~~~~~~#....#~~~.....~~~#.......",
    "...########~~~~~####....#~...~~.....................~~...~#....####~~~~~########...",
    "...#~~~~~~~~~~~~~~~#~~~~##...~~.....................~~...##~~~~#~~~~~~~~~~~~~~~#...",
    "...#.........................~~.....................~~.........................#...",
    "...#.........................~~.....................~~.........................#...",
    "...#.........................~~.....................~~.........................#...",
    "####.........................~~.....................~~.........................####",
    "~~~~.........................~~.....................~~.........................~~~~",
    "~~~~.........................~~.....................~~.........................~~~~",
    "~~~~.........................~~.....................~~.........................~~~~",
    ".~~~.........................##.....................##.........................~~~.",
    ".~~~.........................#########################.........................~~~.",
    ".~~~.........................~~~~~~~~.........~~~~~~~~.........................~~~.",
    ".~~~.........................~~~~~~~...~~~~~...~~~~~~~.........................~~~.",
    ".~~~.........................~~~#~...~~~~~~~~~...~#~~~.........................~~~.",
    ".################............~~~#...~~~~~~~~~~~...#~~~............################.",
    "..~#~~~~~~~~~~~~~............~~~~###~~~~~~~~~~~###~~~~............~~~~~~~~~~~~~#~..",
    "..~#~~~~~~~~~~~~~............~~~~~~~~~~~~~~~~~~~~~~~~~............~~~~~~~~~~~~~#~..",
    "..~#~~~~~~~~~~~~~............~~.....................~~............~~~~~~~~~~~~~#~..",
    "..~#~~~~~~~~~~~~~............~~.....................~~............~~~~~~~~~~~~~#~..",
    "..~#########~~~##############~~.....................~~##############~~~#########~..",
    "..~~~~~~~~~#~~~#~~~~~~~~~~~~~~~.....................~~~~~~~~~~~~~~~#~~~#~~~~~~~~~..",
    "..~~~~~~~~##~~~##~~~~~~~~~~~~~~.....................~~~~~~~~~~~~~~##~~~##~~~~~~~~..",
    ".............................~~.....................~~.............................",
    ".............................~~.....................~~.............................",
    ".............................~~.....................~~.............................",
    ".............................~~.....................~~.............................",
    ".............................~~.....................~~.............................",
];

/// `HALL_MATCH`: the share of the hall's tiles that must be equal.
const HALL_MATCH: f64 = 0.95;
/// `SAMPLE_MIN`: the share of the sample that must be equal for an offset to be looked at in full.
const SAMPLE_MIN: f64 = 0.8;

/// `hallClasses` and the sample (`SAMPLE_*`): every 25th non-air tile and every 150th air tile.
struct HallPattern {
    w: usize,
    h: usize,
    classes: Vec<u8>,
    sample_x: Vec<usize>,
    sample_y: Vec<usize>,
    sample_c: Vec<u8>,
}

fn hall_pattern() -> &'static HallPattern {
    static P: OnceLock<HallPattern> = OnceLock::new();
    P.get_or_init(|| {
        let w = HALL_CORE[0].len();
        let h = HALL_CORE.len();
        let classes: Vec<u8> = HALL_CORE
            .iter()
            .flat_map(|r| r.bytes())
            .map(|c| match c {
                b'#' => 1,
                b'~' => 2,
                b'X' => 3,
                _ => 0,
            })
            .collect();
        let (mut non_air, mut air) = (0usize, 0usize);
        let (mut sample_x, mut sample_y, mut sample_c) = (Vec::new(), Vec::new(), Vec::new());
        for (i, &c) in classes.iter().enumerate() {
            let take = if c != 0 {
                let t = non_air % 25 == 0;
                non_air += 1;
                t
            } else {
                let t = air % 150 == 0;
                air += 1;
                t
            };
            if take {
                sample_x.push(i % w);
                sample_y.push(i / w);
                sample_c.push(c);
            }
        }
        HallPattern {
            w,
            h,
            classes,
            sample_x,
            sample_y,
            sample_c,
        }
    })
}

/// The result of [`find_hall_offset`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HallOffset {
    pub dx: i32,
    pub dy: i32,
    /// The share of the hall's tiles that are equal.
    pub matched: f64,
}

/// `findHallOffset(col)`: where in the map the hall of `Copy Love Box` is (its offset from the original's
/// place), if its tiles match the hall's by `HALL_MATCH`. The first best offset wins.
pub fn find_hall_offset(col: &impl PlanCollision) -> Option<HallOffset> {
    let pat = hall_pattern();
    let (w, h) = (col.width().max(0) as usize, col.height().max(0) as usize);
    if w < pat.w || h < pat.h {
        return None;
    }
    let mut grid = vec![0u8; w * h];
    for ty in 0..h {
        for tx in 0..w {
            grid[ty * w + tx] = match col.game_tile(tx as i32, ty as i32) {
                TILE_SOLID | TILE_NOHOOK => 1,
                TILE_FREEZE => 2,
                TILE_DEATH => 3,
                _ => 0,
            };
        }
    }
    let n = pat.sample_c.len();
    let sample_off: Vec<usize> = (0..n).map(|k| pat.sample_y[k] * w + pat.sample_x[k]).collect();
    let sample_limit = (n as f64 * (1.0 - SAMPLE_MIN) + 1e-9).floor() as usize;
    let total = pat.w * pat.h;
    let limit = (total as f64 * (1.0 - HALL_MATCH) + 1e-9).floor() as usize;
    let mut best_mis = limit + 1;
    let mut best: Option<(i32, i32)> = None;
    for oy in 0..=(h - pat.h) {
        for ox in 0..=(w - pat.w) {
            let base = oy * w + ox;
            let mut mis = 0usize;
            let mut k = 0;
            while k < n && mis <= sample_limit {
                if grid[base + sample_off[k]] != pat.sample_c[k] {
                    mis += 1;
                }
                k += 1;
            }
            if mis > sample_limit {
                continue;
            }
            mis = 0;
            let max = best_mis as i64 - 1;
            let mut y = 0;
            while y < pat.h && mis as i64 <= max {
                let g = base + y * w;
                let s = y * pat.w;
                for x in 0..pat.w {
                    if grid[g + x] != pat.classes[s + x] {
                        mis += 1;
                    }
                }
                y += 1;
            }
            if mis as i64 <= max {
                best_mis = mis;
                best = Some((ox as i32 - HALL_X0, oy as i32 - HALL_Y0));
            }
        }
    }
    best.map(|(dx, dy)| HallOffset {
        dx,
        dy,
        matched: 1.0 - best_mis as f64 / total as f64,
    })
}

const TILE_SOLID: u8 = 1;
const TILE_DEATH: u8 = 2;
const TILE_NOHOOK: u8 = 3;
const TILE_FREEZE: u8 = 9;

/// `checkWb(def, col, whole)`: every spot standable, every rope anchor of the tubes a hookable solid tile
/// and, with `whole`, also every tube's start standable and its exit boxes free of solid and freeze.
fn check_wb(def: &WbDef, col: &impl PlanCollision, whole: bool) -> bool {
    for s in [&def.left, &def.right] {
        for p in &s.spots {
            if !standable(col, p.0, p.1) {
                return false;
            }
        }
    }
    for c in &def.crossings {
        for a in &c.anchors {
            if a.0 < 0 || a.1 < 0 || a.0 >= col.width() || a.1 >= col.height() {
                return false;
            }
            let x = f64::from(a.0 * 32 + 16);
            let y = f64::from(a.1 * 32 + 16);
            if !col.is_solid(x, y) || col.is_no_hook(x, y) {
                return false;
            }
        }
        if !whole {
            continue;
        }
        if !standable(col, c.start.0, c.start.1) {
            return false;
        }
        for b in &c.exit {
            if b.x0 < 0 || b.y0 < 0 || b.x1 >= col.width() || b.y1 >= col.height() {
                return false;
            }
            for y in b.y0..=b.y1 {
                for x in b.x0..=b.x1 {
                    let (px, py) = (f64::from(x * 32 + 16), f64::from(y * 32 + 16));
                    if col.is_freeze(px, py) || col.is_solid(px, py) {
                        return false;
                    }
                }
            }
        }
    }
    true
}

/// `checkWalls(def, col)`: a tube whose wall route does not fit this map loses it (route 2 is off for it).
fn check_walls(def: WbDef, col: &impl PlanCollision) -> WbDef {
    if def.crossings.iter().all(|c| c.wall.is_none() || wall_route_ok(col, c)) {
        return def;
    }
    let strip = |sd: &WbSideDef| -> WbSideDef {
        if sd.crossing.wall.is_none() || wall_route_ok(col, &sd.crossing) {
            return sd.clone();
        }
        let mut sd = sd.clone();
        sd.crossing.wall = None;
        sd
    };
    let left = strip(&def.left);
    let right = strip(&def.right);
    WbDef {
        crossings: vec![left.crossing.clone(), right.crossing.clone()],
        left,
        right,
        ..def
    }
}

/// `wayblockFor(mapName, col?)`: the definition for the map named `map_name`, or `None`. With a
/// collision: a map of the named definition's size with standable spots and hookable anchors is that
/// definition (its walls checked); any other map is searched for the hall of `Copy Love Box` by its tiles
/// (`find_hall_offset`) and gets the definition shifted to where the hall is, if that fits (every spot
/// standable, every anchor hookable, every start standable, every exit free).
pub fn wayblock_for<C: PlanCollision>(map_name: &str, col: Option<&C>) -> Option<WbDef> {
    let want = map_name.trim().to_lowercase();
    let def = wayblocks().into_iter().find(|d| d.name.to_lowercase() == want);
    let Some(col) = col else { return def };
    if let Some(def) = def
        && col.width() == def.size.0
        && col.height() == def.size.1
        && check_wb(&def, col, false)
    {
        return Some(check_walls(def, col));
    }
    let hall = find_hall_offset(col)?;
    let sign = |n: i32| if n >= 0 { format!("+{n}") } else { format!("{n}") };
    let found = shift_def(
        &copy_love_box(),
        format!("Copy Love Box hall at {},{}", sign(hall.dx), sign(hall.dy)),
        hall.dx,
        hall.dy,
        (col.width(), col.height()),
    );
    check_wb(&found, col, true).then(|| check_walls(found, col))
}

/// `wayblockFor(name)` without a collision: whether a definition for the name exists.
pub fn has_wayblock_named(map_name: &str) -> bool {
    let want = map_name.trim().to_lowercase();
    wayblocks().iter().any(|d| d.name.to_lowercase() == want)
}

impl WbDef {
    pub fn side(&self, s: WbSide) -> &WbSideDef {
        match s {
            WbSide::Left => &self.left,
            WbSide::Right => &self.right,
        }
    }

    /// `sideAt`: the side whose zone or approach contains the tile.
    pub fn side_at(&self, tx: i32, ty: i32) -> Option<WbSide> {
        for s in [WbSide::Left, WbSide::Right] {
            let d = self.side(s);
            if in_any_box(&d.zone, tx, ty) || in_any_box(&d.approach, tx, ty) {
                return Some(s);
            }
        }
        None
    }

    /// `inWbZone`.
    pub fn in_zone(&self, s: WbSide, tx: i32, ty: i32) -> bool {
        let d = self.side(s);
        in_any_box(&d.zone, tx, ty) || in_any_box(&d.approach, tx, ty)
    }

    /// `inWbHall`: within 3 tiles of a zone box.
    pub fn in_hall(&self, s: WbSide, tx: i32, ty: i32) -> bool {
        self.side(s).zone.iter().any(|b| {
            tx >= b.x0 - WB_LEASH_TILES
                && tx <= b.x1 + WB_LEASH_TILES
                && ty >= b.y0 - WB_LEASH_TILES
                && ty <= b.y1 + WB_LEASH_TILES
        })
    }

    /// `inWbLeash`.
    pub fn in_leash(&self, s: WbSide, tx: i32, ty: i32) -> bool {
        in_any_box(&self.side(s).leash, tx, ty) && !in_any_box(&self.avoid, tx, ty)
    }

    /// `wbWalkAllowed(def, tx, ty)`.
    pub fn walk_allowed(&self, tx: i32, ty: i32) -> bool {
        !in_any_box(&self.avoid, tx, ty)
    }

    /// `wbGuardGeom(def, s)`: the places the guard on the upper shelf of side `s` cares about, all
    /// measured from the side's watch point (the numbers are those of the left hall, mirrored for the right).
    pub fn guard_geom(&self, s: WbSide) -> WbGuardGeom {
        let o = self.side(s).watch;
        let sign = if s == WbSide::Left { 1 } else { -1 };
        let gx = |x: i32| o.0 + sign * (x - 89);
        let gy = |y: i32| o.1 + (y - 79);
        let bx =
            |x0: i32, y0: i32, x1: i32, y1: i32| TileBox::new(gx(x0).min(gx(x1)), gy(y0), gx(x0).max(gx(x1)), gy(y1));
        WbGuardGeom {
            shelf: bx(92, 81, 104, 84),
            column: bx(93, 68, 104, 80),
            landing: bx(85, 76, 92, 79),
            foot: bx(85, 66, 92, 78),
            passage: bx(85, 41, 92, 65),
            corridor: bx(73, 70, 77, 88),
            job: (gx(91), gy(79)),
            step_off: (gx(86), gy(79)),
        }
    }
}

/// `WbGuardGeom`: see [`WbDef::guard_geom`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WbGuardGeom {
    /// The lower shelf, under the spot.
    pub shelf: TileBox,
    /// The column of air over the lower shelf.
    pub column: TileBox,
    /// Where the tube's first route drops them.
    pub landing: TileBox,
    pub foot: TileBox,
    pub passage: TileBox,
    /// The corridor behind the wall.
    pub corridor: TileBox,
    /// Where the guard stands to throw the ones on the lower shelf.
    pub job: (i32, i32),
    /// Where it steps off to when someone in the corridor is in reach of its home.
    pub step_off: (i32, i32),
}

/// `wbWalkAllowed` for a map that may have no WB.
pub fn wb_walk_allowed(def: Option<&WbDef>, tx: i32, ty: i32) -> bool {
    def.is_none_or(|d| d.walk_allowed(tx, ty))
}

/// `WbSideChooser`: which hall to hold. `auto` picks the side with fewer players (ties: the side we
/// stand in, else the nearer), is provisional for 5 s when nobody is around, and never hops.
#[derive(Debug, Clone)]
pub struct WbSideChooser {
    pub side: Option<WbSide>,
    pending_since: i64,
    provisional: bool,
    picked_at: i64,
    /// Task 3.12 (`--wb-smart`, off by default): the side with **more** blockable targets, ties random, with hysteresis
    /// ([`WbSideChooser::update_targets`]).
    smart: bool,
    /// The tie-break generator of the smart chooser (splitmix64; seeded once per session, kept over map changes).
    rng: u64,
    /// The tick of the last change of side (smart).
    changed_at: i64,
    /// Ticks of the failed crossings of the left / right tube (smart), newest last.
    fails: [Vec<i64>; 2],
    /// The tick of the first smart look (warm-up, smart); -1 before it.
    first_tick: i64,
    /// We have stood free in a hall since the map was loaded (smart): the warm-up is over for the side choice and the cooldown runs.
    reached: bool,
    /// A committed smart switch to this side whose hall we have not reached yet: the "standing" override must not undo it.
    leaving: Option<WbSide>,
    /// What the commit of the switch in `leaving` overwrote: `changed_at` and the age of the lead (`pending_since`). A switch that is
    /// cancelled before we leave the zone puts them back (review F7): a cancelled switch starts no cooldown and keeps its lead.
    commit_prev: Option<(i64, i64)>,
}

impl Default for WbSideChooser {
    fn default() -> Self {
        WbSideChooser {
            side: None,
            pending_since: -1,
            provisional: false,
            picked_at: -1,
            smart: false,
            rng: 0,
            changed_at: i64::MIN / 2,
            fails: [Vec::new(), Vec::new()],
            first_tick: -1,
            reached: false,
            leaving: None,
            commit_prev: None,
        }
    }
}

impl WbSideChooser {
    /// A new map: the choice and the failures of the tubes start over. The smart switch and its generator stay (the session's).
    pub fn reset(&mut self) {
        let (smart, rng) = (self.smart, self.rng);
        *self = WbSideChooser {
            smart,
            rng,
            ..WbSideChooser::default()
        };
    }

    /// Task 3.12: choose by blockable targets from now on (`seed` starts the tie-break generator).
    pub fn set_smart(&mut self, on: bool, seed: u64) {
        self.smart = on;
        self.rng = seed;
    }

    pub fn is_smart(&self) -> bool {
        self.smart
    }

    /// The side a committed smart switch is taking us to, until we stand in its hall.
    pub fn leaving(&self) -> Option<WbSide> {
        self.leaving
    }

    /// We stand free in the zone of `side` (smart): the warm-up is over, the cooldown of a change starts, a switch to it is complete.
    pub fn note_reached(&mut self, side: WbSide, tick: i64) {
        if !self.reached {
            self.reached = true;
            self.changed_at = tick;
        }
        if self.leaving == Some(side) {
            self.leaving = None;
            self.commit_prev = None;
        }
    }

    /// What a committed switch changes, to put back a switch that is not to be (see [`WbState::update_side`]).
    fn smart_snapshot(&self) -> (Option<WbSide>, i64, i64, Option<WbSide>) {
        (self.side, self.pending_since, self.changed_at, self.leaving)
    }

    fn smart_restore(&mut self, snap: (Option<WbSide>, i64, i64, Option<WbSide>)) {
        (self.side, self.pending_since, self.changed_at, self.leaving) = snap;
    }

    fn next_bit(&mut self) -> bool {
        // splitmix64
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) & 1 == 1
    }

    /// A walk through `side`'s freeze tube failed at `tick` (the navigator's "...; trying again from the spawn").
    pub fn note_cross_fail(&mut self, side: WbSide, tick: i64) {
        let v = &mut self.fails[usize::from(side == WbSide::Right)];
        v.retain(|&t| t <= tick && tick - t < WB_FAIL_MEMORY_TICKS);
        if v.len() < 8 {
            v.push(tick);
        }
    }

    /// Failed crossings of `side`'s tube in the last [`WB_FAIL_MEMORY_TICKS`].
    pub fn recent_fails(&self, side: WbSide, tick: i64) -> i32 {
        self.fails[usize::from(side == WbSide::Right)]
            .iter()
            .filter(|&&t| t <= tick && tick - t < WB_FAIL_MEMORY_TICKS)
            .count() as i32
    }

    pub fn adopt(&mut self, side: WbSide) {
        self.side = Some(side);
        self.provisional = false;
        self.pending_since = -1;
        self.leaving = None;
        self.commit_prev = None;
    }

    /// `update(counts, here, tick, nearer)`.
    pub fn update(&mut self, counts: (i32, i32), here: Option<WbSide>, tick: i64, nearer: WbSide) -> WbSide {
        let (left, right) = counts;
        if self.provisional
            && (here == self.side || tick < self.picked_at || tick - self.picked_at >= WB_PROVISIONAL_TICKS)
        {
            self.provisional = false;
        }
        if self.side.is_none() || (self.provisional && left + right > 0) {
            let side = if left < right {
                WbSide::Left
            } else if right < left {
                WbSide::Right
            } else {
                here.unwrap_or(nearer)
            };
            self.side = Some(side);
            self.provisional = left + right == 0;
            self.picked_at = tick;
            self.pending_since = -1;
            return side;
        }
        let cur = self.side.expect("side");
        if !WB_SIDE_HOPPING {
            return cur;
        }
        if tick < self.pending_since {
            self.pending_since = tick;
        }
        let other = cur.other();
        let count = |s: WbSide| if s == WbSide::Left { left } else { right };
        let lead = count(other) - count(cur);
        let wants = if here == Some(other) {
            lead >= 0
        } else {
            lead >= WB_SWITCH_MARGIN
        };
        if !wants {
            self.pending_since = -1;
            return cur;
        }
        if self.pending_since < 0 {
            self.pending_since = tick;
        }
        if tick - self.pending_since >= WB_SWITCH_TICKS {
            self.side = Some(other);
            self.pending_since = -1;
        }
        self.side.expect("side")
    }
}

impl WbSideChooser {
    /// Task 3.12: the smart `update`. `counts` are the **blockable** targets of the left and right hall (the caller decides
    /// who is blockable); a side's score is its count less [`WB_FAIL_PENALTY`] per recent failed crossing of its tube.
    ///
    /// - Nothing chosen yet (or the pick is provisional and somebody showed up): the side with the higher score; a tie goes
    ///   to the side we stand in, else **random**. With nobody on either side the pick is provisional for
    ///   [`WB_PROVISIONAL_TICKS`] (as `update`).
    /// - Warm-up: until [`WB_SMART_WARMUP_TICKS`] after the first look, and until [`WbSideChooser::note_reached`], the pick is taken
    ///   again at every look (a tie keeps the side), starts no cooldown and needs no margin.
    /// - Hysteresis: the other side wins only when its score is ahead by [`WB_SWITCH_MARGIN`] (when we stand in it: not behind) for
    ///   [`WB_SWITCH_TICKS`] without a break, and not within [`WB_SMART_SWITCH_COOLDOWN_TICKS`] of the last change (a change made once
    ///   we have been in a hall; the cooldown starts at the first arrival). A committed switch sets [`WbSideChooser::leaving`].
    pub fn update_targets(&mut self, counts: (i32, i32), here: Option<WbSide>, tick: i64) -> WbSide {
        let score = |me: &Self, s: WbSide| {
            (if s == WbSide::Left { counts.0 } else { counts.1 }) - WB_FAIL_PENALTY * me.recent_fails(s, tick)
        };
        if self.first_tick < 0 || tick < self.first_tick {
            self.first_tick = tick;
        }
        let warm = !self.reached && tick - self.first_tick < WB_SMART_WARMUP_TICKS;
        if self.provisional
            && (here == self.side || tick < self.picked_at || tick - self.picked_at >= WB_PROVISIONAL_TICKS)
        {
            self.provisional = false;
        }
        let first = self.side.is_none() || (self.provisional && counts.0 + counts.1 > 0);
        if first || warm {
            let (l, r) = (score(self, WbSide::Left), score(self, WbSide::Right));
            let side = if l > r {
                WbSide::Left
            } else if r > l {
                WbSide::Right
            } else if let Some(h) = here {
                h
            } else if let Some(cur) = self.side {
                cur
            } else if self.next_bit() {
                WbSide::Left
            } else {
                WbSide::Right
            };
            if !first && Some(side) != self.side {
                // Warm-up, a pick that would change the side: it has to lead for WB_SMART_WARMUP_HOLD_TICKS first (review F6).
                if self.pending_since < 0 || tick < self.pending_since {
                    self.pending_since = tick;
                }
                if tick - self.pending_since < WB_SMART_WARMUP_HOLD_TICKS {
                    return self.side.expect("side");
                }
            }
            self.side = Some(side);
            self.provisional = counts.0 + counts.1 == 0;
            self.picked_at = tick;
            self.pending_since = -1;
            return side;
        }
        let cur = self.side.expect("side");
        if tick < self.pending_since {
            self.pending_since = tick;
        }
        if tick < self.changed_at {
            self.changed_at = tick;
        }
        let other = cur.other();
        let lead = score(self, other) - score(self, cur);
        let wants = if here == Some(other) {
            lead >= 0
        } else {
            lead >= WB_SWITCH_MARGIN
        };
        if !wants || tick - self.changed_at < WB_SMART_SWITCH_COOLDOWN_TICKS {
            if !wants {
                self.pending_since = -1;
            }
            return cur;
        }
        if self.pending_since < 0 {
            self.pending_since = tick;
        }
        if tick - self.pending_since >= WB_SWITCH_TICKS {
            self.side = Some(other);
            self.commit_prev = Some((self.changed_at, self.pending_since));
            self.pending_since = -1;
            self.leaving = Some(other);
            if self.reached {
                self.changed_at = tick;
            }
        }
        self.side.expect("side")
    }
}

/// `onWbSpot(here, p)`: within 2 tiles of the spot.
pub fn on_wb_spot(here: (i32, i32), p: (i32, i32)) -> bool {
    (here.0 - p.0).abs() <= 2 && (here.1 - p.1).abs() <= 2
}

/// `wbSpot(ownId, def, side, here?)`: the spot to hold. `is_friend(id)` is true for an unfrozen friend
/// that holds a spot of its own (TS `isFriendId`; the partner of TS is dropped, D-021).
pub fn wb_spot(
    own_id: i32,
    tees: &[TeeState],
    is_friend: &dyn Fn(i32) -> bool,
    def: &WbDef,
    side: WbSide,
    here: Option<(i32, i32)>,
) -> (i32, i32) {
    let all = &def.side(side).spots;
    let inside = here.is_some_and(|h| def.in_hall(side, h.0, h.1));
    let reachable: Vec<(i32, i32)> = if inside {
        let h = here.expect("here");
        all.iter().copied().filter(|p| h.1 - p.1 < WB_NO_CLIMB_TILES).collect()
    } else {
        all.clone()
    };
    let spots: &[(i32, i32)] = if reachable.is_empty() { all } else { &reachable };
    let tile = |t: &TeeState| ((t.pos.x / 32.0).trunc() as i32, (t.pos.y / 32.0).trunc() as i32);
    let taken = |p: (i32, i32)| {
        tees.iter().any(|t| {
            if t.id == own_id || !t.alive {
                return false;
            }
            if !t.frozen && is_friend(t.id) {
                return on_wb_spot(tile(t), p);
            }
            !t.frozen
                && (t.pos.x - f64::from(p.0 * 32 + 16)).abs() < 32.0
                && (t.pos.y - f64::from(p.1 * 32 + 16)).abs() < 32.0
        })
    };
    let friend_holds = |p: (i32, i32)| {
        tees.iter()
            .any(|t| t.id != own_id && t.alive && !t.frozen && is_friend(t.id) && on_wb_spot(tile(t), p))
    };
    spots
        .iter()
        .copied()
        .find(|&p| here.is_some_and(|h| on_wb_spot(h, p)) || !taken(p))
        .or_else(|| spots.iter().copied().find(|&p| !friend_holds(p)))
        .unwrap_or(if inside { here.expect("here") } else { spots[0] })
}

/// `wbBand` (`bot.ts:4799`): the pixel rectangle `(x0, y0, x1, y1)` around the foot of the side's tube,
/// from just under the zone's top row to two rows above its bottom.
pub fn wb_band(def: &WbDef, side: WbSide) -> Option<(f32, f32, f32, f32)> {
    let sd = def.side(side);
    let foot = sd.crossing.exit.last()?;
    let top = sd.zone.first()?;
    Some((
        ((foot.x0 - 1) * 32) as f32,
        ((top.y0 + 1) * 32) as f32,
        ((foot.x1 + 2) * 32) as f32,
        ((top.y1 - 2) * 32) as f32,
    ))
}

/// `wbMode` (`!wb off | left | right | auto`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WbMode {
    Off,
    Auto,
    Left,
    Right,
}

impl WbMode {
    pub fn name(self) -> &'static str {
        match self {
            WbMode::Off => "off",
            WbMode::Auto => "auto",
            WbMode::Left => "left",
            WbMode::Right => "right",
        }
    }

    /// `on` is `auto`, as in `wbCommand`.
    pub fn parse(s: &str) -> Option<WbMode> {
        match s.trim().to_lowercase().as_str() {
            "on" | "auto" => Some(WbMode::Auto),
            "off" => Some(WbMode::Off),
            "left" => Some(WbMode::Left),
            "right" => Some(WbMode::Right),
            _ => None,
        }
    }
}

/// `WB_WALK_MAX_FAILS` (`bot.ts:178`): deaths on the way to the WB in a row before it is left alone.
pub const WB_WALK_MAX_FAILS: i32 = 4;
/// `WB_WALK_PAUSE_MS`: the first pause is 5 minutes, doubling per pause up to 30.
pub const WB_WALK_PAUSE_MS: i64 = 5 * 60_000;
pub const WB_WALK_PAUSE_MAX_MS: i64 = 30 * 60_000;
/// `WB_RETURN_TICKS`: idle this long outside the spot, and the bot walks back to it.
pub const WB_RETURN_TICKS: i64 = 50;

/// A side change reported by [`WbState::update_side`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SideChange {
    pub to: WbSide,
}

/// The wayblock state of one bot (`wbDef`, `wbMode`, `wbChooser`, `wbWalkFails`, `wbPauses`,
/// `wbPausedUntilMs`): everything `wbHolding`/`updateWbSide`/`noteWbWalkDeath` keep, with the clock
/// injected (`now_ms` is real wall time in minutes' resolution; tests pass their own).
#[derive(Debug, Clone)]
pub struct WbState {
    pub def: Option<WbDef>,
    pub mode: WbMode,
    pub chooser: WbSideChooser,
    /// Players seen playing on each side at the last update (`wbCounts`).
    pub counts: (i32, i32),
    /// Task 3.12 (`--wb-smart`): a fight is going on where we stand (set by the caller before `update_side`): a committed switch of side
    /// then waits (the standing override keeps us in this hall) until it is over.
    pub fight_here: bool,
    /// The tick `fight_here` was last true ([`WB_FIGHT_HOLD_TICKS`]).
    fight_last: i64,
    walk_fails: i32,
    pauses: i32,
    paused_until_ms: i64,
}

impl WbState {
    pub fn new(def: Option<WbDef>, mode: WbMode) -> WbState {
        WbState {
            def,
            mode,
            chooser: WbSideChooser::default(),
            counts: (0, 0),
            fight_here: false,
            fight_last: i64::MIN / 2,
            walk_fails: 0,
            pauses: 0,
            paused_until_ms: 0,
        }
    }

    /// A new map: the choice, the counts and the pauses start over.
    pub fn on_map(&mut self, def: Option<WbDef>) {
        self.on_map_keeping(def, false);
    }

    /// A map (re)loaded: with `keep_pauses` (the same map name and size as before: `wbPauseKey`) the failures
    /// of the walk and its pause are kept, so reloading the map does not end a pause.
    pub fn on_map_keeping(&mut self, def: Option<WbDef>, keep_pauses: bool) {
        self.def = def;
        self.chooser.reset();
        self.counts = (0, 0);
        if !keep_pauses {
            self.walk_fails = 0;
            self.pauses = 0;
            self.paused_until_ms = 0;
        }
    }

    /// Deaths on the way in a row (`wbWalkFails`).
    pub fn walk_fails(&self) -> i32 {
        self.walk_fails
    }

    /// `wbHolding()`: the wayblock we are holding now. `fights` is "the mode is fight, or a goto that
    /// returns to fight"; a set home, a duel and a pause switch it off.
    pub fn holding(&self, now_ms: i64, fights: bool, home_set: bool, duel: bool) -> Option<&WbDef> {
        let def = self.def.as_ref()?;
        if self.mode == WbMode::Off || home_set || duel || now_ms < self.paused_until_ms {
            return None;
        }
        fights.then_some(def)
    }

    /// Minutes of pause left, rounded up (`wbCommand`'s report).
    pub fn paused_min(&self, now_ms: i64) -> i64 {
        ((self.paused_until_ms - now_ms) as f64 / 60_000.0).ceil() as i64
    }

    pub fn side(&self) -> Option<WbSide> {
        self.chooser.side
    }

    /// `wbCommand`'s mode change: any mode but `off` ends a pause and forgets the failures.
    pub fn set_mode(&mut self, mode: WbMode) {
        self.mode = mode;
        if mode != WbMode::Off {
            self.paused_until_ms = 0;
            self.walk_fails = 0;
        }
    }

    /// `updateWbSide`: `counts` are the players playing on each side (the caller drops AFK and parked
    /// tees), `own` our tile and whether we stand free (alive, unfrozen). Returns the new side when it
    /// changed while the WB is held (`holding`).
    pub fn update_side(
        &mut self,
        own_tile: (i32, i32),
        own_x_tiles: f64,
        own_free: bool,
        counts: (i32, i32),
        tick: i64,
        holding: bool,
    ) -> Option<SideChange> {
        let def = self.def.as_ref()?;
        self.counts = counts;
        if tick < self.fight_last {
            self.fight_last = i64::MIN / 2; // the clock went back (a new session): forget the old fight
        }
        if self.fight_here {
            self.fight_last = tick;
        }
        let fight = tick - self.fight_last < WB_FIGHT_HOLD_TICKS;
        let before = self.chooser.side;
        let side;
        match self.mode {
            WbMode::Left | WbMode::Right => {
                side = if self.mode == WbMode::Left {
                    WbSide::Left
                } else {
                    WbSide::Right
                };
                self.chooser.side = Some(side);
            }
            _ => {
                let here = def.side_at(own_tile.0, own_tile.1);
                let nearer = if (own_x_tiles - f64::from(def.left.watch.0)).abs()
                    <= (own_x_tiles - f64::from(def.right.watch.0)).abs()
                {
                    WbSide::Left
                } else {
                    WbSide::Right
                };
                let smart = self.chooser.is_smart();
                let snap = self.chooser.smart_snapshot();
                let mut s = if smart {
                    self.chooser.update_targets(counts, here, tick)
                } else {
                    self.chooser.update(counts, here, tick, nearer)
                };
                let standing = if own_free {
                    [WbSide::Left, WbSide::Right]
                        .into_iter()
                        .find(|&sd| in_any_box(&def.side(sd).zone, own_tile.0, own_tile.1))
                } else {
                    None
                };
                if let Some(st) = standing {
                    if st == s {
                        if smart {
                            self.chooser.note_reached(st, tick);
                        }
                    } else if smart && self.chooser.leaving() == Some(s) && !fight {
                        // A committed switch: we leave the hall we stand in for the busier one.
                    } else {
                        // We stay where we stand; a switch committed by this very call is not made (and starts no cooldown).
                        let undone = smart && self.chooser.leaving() == Some(s) && snap.3 != Some(s);
                        // Committed on an earlier look and cancelled now, before we left the zone (review F7): the same, from `commit_prev`.
                        let cancelled = smart && self.chooser.leaving() == Some(s) && snap.3 == Some(s);
                        let prev = self.chooser.commit_prev;
                        if undone {
                            self.chooser.smart_restore(snap);
                        }
                        self.chooser.adopt(st);
                        if undone {
                            // The lead keeps its age: the switch comes when the fight is over, not 250 ticks later.
                            self.chooser.pending_since = snap.1;
                        }
                        if cancelled && let Some((changed_at, pending)) = prev {
                            self.chooser.changed_at = changed_at;
                            self.chooser.pending_since = pending;
                        }
                        if smart {
                            self.chooser.note_reached(st, tick);
                        }
                        s = st;
                    }
                }
                side = s;
            }
        }
        (before.is_some_and(|b| b != side) && holding).then_some(SideChange { to: side })
    }

    /// `noteWbWalkDeath` (a death on the walk to the WB): after [`WB_WALK_MAX_FAILS`] in a row the WB is
    /// left alone for `5 * 2^(pauses - 1)` minutes, at most 30. Returns the pause in ms when one began.
    pub fn note_walk_death(&mut self, now_ms: i64) -> Option<i64> {
        self.walk_fails += 1;
        if self.walk_fails < WB_WALK_MAX_FAILS {
            return None;
        }
        self.walk_fails = 0;
        self.pauses += 1;
        let ms = WB_WALK_PAUSE_MAX_MS.min(WB_WALK_PAUSE_MS * (1i64 << (self.pauses - 1).min(20)));
        self.paused_until_ms = now_ms + ms;
        Some(ms)
    }

    /// Reaching a hall (alive and free) forgives the failures and pauses.
    pub fn arrived_in_hall(&mut self, own_tile: (i32, i32), frozen: bool) {
        if (self.walk_fails > 0 || self.pauses > 0)
            && !frozen
            && let Some(def) = &self.def
            && (def.in_hall(WbSide::Left, own_tile.0, own_tile.1) || def.in_hall(WbSide::Right, own_tile.0, own_tile.1))
        {
            self.walk_fails = 0;
            self.pauses = 0;
        }
    }

    /// `wbWalkCuttable`: we stand in the zone of the held side below its top row, free: the walk to the
    /// spot may be cut short to fight someone.
    pub fn walk_cuttable(&self, own_tile: (i32, i32), frozen: bool, holding: bool) -> bool {
        let (Some(def), Some(side)) = (&self.def, self.chooser.side) else {
            return false;
        };
        if !holding || frozen {
            return false;
        }
        def.side(side)
            .zone
            .iter()
            .any(|b| own_tile.0 >= b.x0 && own_tile.0 <= b.x1 && own_tile.1 > b.y0 && own_tile.1 <= b.y1)
    }

    /// `wbBand` and whether the hall's overrides apply: both only while we stand in the hall of the held
    /// side (`in_hall`). The band is the pixel rectangle `(x0, y0, x1, y1)` around the tube's foot.
    pub fn hall_hints(&self, own_tile: (i32, i32), holding: bool) -> Option<(f32, f32, f32, f32)> {
        let (Some(def), Some(side)) = (&self.def, self.chooser.side) else {
            return None;
        };
        if !holding || !def.in_hall(side, own_tile.0, own_tile.1) {
            return None;
        }
        wb_band(def, side)
    }
}

#[cfg(test)]
mod state_tests {
    use super::*;

    fn clb() -> WbDef {
        wayblocks()
            .into_iter()
            .find(|d| d.name == "Copy Love Box")
            .expect("CLB")
    }

    fn state() -> WbState {
        WbState::new(Some(clb()), WbMode::Auto)
    }

    #[test]
    fn four_deaths_on_the_walk_pause_the_wb_for_five_then_ten_then_twenty_then_thirty_minutes() {
        let mut s = state();
        let mut now = 1_000_000;
        let mut pauses = Vec::new();
        for _ in 0..5 {
            for k in 0..WB_WALK_MAX_FAILS {
                let r = s.note_walk_death(now);
                assert_eq!(
                    r.is_some(),
                    k == WB_WALK_MAX_FAILS - 1,
                    "only the 4th death in a row pauses"
                );
                if let Some(ms) = r {
                    pauses.push(ms / 60_000);
                }
            }
            assert!(s.holding(now, true, false, false).is_none(), "paused");
            now += s.paused_until_ms - now; // wait it out
            assert!(s.holding(now, true, false, false).is_some(), "the pause is over");
        }
        assert_eq!(pauses, vec![5, 10, 20, 30, 30], "5 * 2^(n-1) minutes, at most 30");
    }

    #[test]
    fn reaching_a_hall_forgives_failures_and_pauses_and_a_mode_change_ends_a_pause() {
        let mut s = state();
        for _ in 0..4 {
            s.note_walk_death(0);
        }
        assert_eq!(s.paused_min(0), 5);
        // Frozen in the hall does not count; standing free in the left hall does. As in TS the running
        // pause is not cut short, but the next one starts at 5 minutes again instead of 10.
        let hall = clb().left.zone[0];
        let tile = (hall.x0 + 1, hall.y0 + 2);
        s.arrived_in_hall(tile, true);
        s.arrived_in_hall(tile, false);
        let mut ms = 0;
        for _ in 0..4 {
            ms = s.note_walk_death(10).unwrap_or(ms);
        }
        assert_eq!(ms, WB_WALK_PAUSE_MS, "the pause counter started over");
        assert!(s.holding(11, true, false, false).is_none());
        s.set_mode(WbMode::Left);
        assert!(s.holding(11, true, false, false).is_some(), "`!wb left` ends the pause");
    }

    #[test]
    fn holding_needs_the_wb_a_fighting_mode_and_no_home_or_duel() {
        let s = state();
        assert!(s.holding(0, true, false, false).is_some());
        assert!(s.holding(0, false, false, false).is_none(), "not a fighting mode");
        assert!(s.holding(0, true, true, false).is_none(), "home is set");
        assert!(s.holding(0, true, false, true).is_none(), "a duel");
        let mut off = state();
        off.set_mode(WbMode::Off);
        assert!(off.holding(0, true, false, false).is_none());
        assert!(
            WbState::new(None, WbMode::Auto)
                .holding(0, true, false, false)
                .is_none(),
            "no WB on this map"
        );
    }

    #[test]
    fn auto_picks_the_emptier_side_and_never_hops_while_a_fixed_mode_wins_outright() {
        let mut s = state();
        let change = s.update_side((100, 100), 100.0, true, (3, 1), 0, true);
        assert!(change.is_none(), "the first choice is not a change");
        assert_eq!(s.side(), Some(WbSide::Right), "fewer players on the right");
        // The crowd moves; no hopping.
        assert!(s.update_side((100, 100), 100.0, true, (0, 9), 1000, true).is_none());
        assert_eq!(s.side(), Some(WbSide::Right));
        s.set_mode(WbMode::Left);
        let c = s.update_side((100, 100), 100.0, true, (0, 9), 1001, true);
        assert_eq!(c, Some(SideChange { to: WbSide::Left }));
        assert_eq!(s.side(), Some(WbSide::Left));
    }

    #[test]
    fn the_band_and_the_walk_cut_exist_only_inside_the_held_hall() {
        let mut s = state();
        s.update_side((100, 100), 100.0, true, (1, 0), 0, true);
        let side = s.side().expect("side");
        let def = clb();
        let zone = def.side(side).zone[0];
        let inside = (zone.x0 + 1, zone.y0 + 2);
        let band = s.hall_hints(inside, true).expect("a band in the hall");
        assert!(band.0 < band.2 && band.1 < band.3, "a proper rectangle: {band:?}");
        assert!(s.hall_hints(inside, false).is_none(), "not holding");
        assert!(s.hall_hints((0, 0), true).is_none(), "far outside the hall");
        assert!(s.walk_cuttable(inside, false, true));
        assert!(!s.walk_cuttable(inside, true, true), "frozen");
        assert!(
            !s.walk_cuttable((zone.x0 + 1, zone.y0), false, true),
            "the top row of the zone is excluded"
        );
    }
    // ---- task 3.12: `--wb-smart` -------------------------------------------------------------------------------------

    fn smart(seed: u64) -> WbSideChooser {
        let mut c = WbSideChooser::default();
        c.set_smart(true, seed);
        c
    }

    #[test]
    fn smart_picks_the_side_with_more_blockable_targets_the_reverse_of_the_default() {
        let mut c = smart(1);
        assert_eq!(c.update_targets((1, 4), None, 0), WbSide::Right, "more on the right");
        let mut c = smart(1);
        assert_eq!(c.update_targets((5, 2), None, 0), WbSide::Left, "more on the left");
        // The default chooser still takes the emptier side.
        let mut d = WbSideChooser::default();
        assert_eq!(d.update((1, 4), None, 0, WbSide::Left), WbSide::Left);
    }

    #[test]
    fn smart_ties_are_random_per_seed_deterministic_and_the_side_we_stand_in_wins_them() {
        let pick = |seed: u64, here: Option<WbSide>| smart(seed).update_targets((3, 3), here, 0);
        let sides: Vec<WbSide> = (0..64).map(|s| pick(s, None)).collect();
        assert!(
            sides.contains(&WbSide::Left) && sides.contains(&WbSide::Right),
            "both sides come up"
        );
        let lefts = sides.iter().filter(|&&s| s == WbSide::Left).count();
        assert!((16..=48).contains(&lefts), "roughly even: {lefts} of 64");
        assert_eq!(pick(7, None), pick(7, None), "same seed, same pick");
        assert_eq!(pick(7, Some(WbSide::Left)), WbSide::Left);
        assert_eq!(
            pick(8, Some(WbSide::Right)),
            WbSide::Right,
            "standing in a hall is not a coin toss"
        );
    }

    #[test]
    fn smart_nobody_anywhere_is_provisional_and_is_picked_again_when_somebody_shows_up() {
        let mut c = smart(3);
        let first = c.update_targets((0, 0), None, 0);
        // Somebody on the other side within 5 s: the pick is redone.
        let other = first.other();
        let counts = if other == WbSide::Left { (2, 0) } else { (0, 2) };
        assert_eq!(c.update_targets(counts, None, 100), other);
        // Once we stand in a hall it is a real pick: a smaller lead on the first side does not undo it.
        c.note_reached(other, 120);
        assert_eq!(c.update_targets((counts.1, counts.0), None, 150), other);
    }

    #[test]
    fn smart_hysteresis_needs_the_margin_for_five_seconds_without_a_break_and_obeys_the_cooldown() {
        let mut c = smart(1);
        assert_eq!(c.update_targets((6, 2), None, 0), WbSide::Left);
        c.note_reached(WbSide::Left, 0); // in a hall: the warm-up is over and the cooldown runs from here
        // A lead of one is never enough.
        for t in (1..4000).step_by(25) {
            assert_eq!(c.update_targets((3, 4), None, t), WbSide::Left, "tick {t}");
        }
        // A lead of the margin must hold for WB_SWITCH_TICKS; a dip below it starts the count over. (Past the cooldown.)
        let t0 = WB_SMART_SWITCH_COOLDOWN_TICKS + 10;
        assert_eq!(c.update_targets((1, 3), None, t0), WbSide::Left);
        assert_eq!(
            c.update_targets((1, 3), None, t0 + WB_SWITCH_TICKS - 1),
            WbSide::Left,
            "not yet"
        );
        assert_eq!(
            c.update_targets((2, 3), None, t0 + WB_SWITCH_TICKS),
            WbSide::Left,
            "a dip: the count is cleared"
        );
        assert_eq!(
            c.update_targets((1, 3), None, t0 + WB_SWITCH_TICKS + 10),
            WbSide::Left,
            "starts again"
        );
        assert_eq!(
            c.update_targets((1, 3), None, t0 + 2 * WB_SWITCH_TICKS + 10),
            WbSide::Right,
            "held for the whole window"
        );
        // Straight back is held off by the cooldown, whatever the lead.
        let t1 = t0 + 2 * WB_SWITCH_TICKS + 10;
        for t in (t1 + 1..t1 + WB_SMART_SWITCH_COOLDOWN_TICKS - 1).step_by(50) {
            assert_eq!(c.update_targets((9, 0), None, t), WbSide::Right, "cooldown, tick {t}");
        }
    }

    #[test]
    fn smart_failed_crossings_of_a_tube_count_against_its_side_for_three_minutes() {
        let mut c = smart(1);
        assert_eq!(c.update_targets((3, 3), Some(WbSide::Left), 0), WbSide::Left);
        c.note_cross_fail(WbSide::Left, 10);
        c.note_cross_fail(WbSide::Left, 20);
        assert_eq!(c.recent_fails(WbSide::Left, 30), 2);
        assert_eq!(c.recent_fails(WbSide::Right, 30), 0);
        // 3 vs 3 with two failures on the left: the right is ahead by 4 (2 * WB_FAIL_PENALTY), after the cooldown and the window.
        let t0 = WB_SMART_SWITCH_COOLDOWN_TICKS + 100;
        assert_eq!(c.update_targets((3, 3), None, t0), WbSide::Left);
        assert_eq!(c.update_targets((3, 3), None, t0 + WB_SWITCH_TICKS), WbSide::Right);
        // They are forgotten.
        assert_eq!(c.recent_fails(WbSide::Left, 20 + WB_FAIL_MEMORY_TICKS), 0);
    }

    #[test]
    fn smart_state_uses_the_target_chooser_only_when_asked_and_keeps_it_over_a_new_map() {
        let mut s = state();
        s.chooser.set_smart(true, 5);
        s.update_side((100, 100), 100.0, true, (1, 4), 0, true);
        assert_eq!(s.side(), Some(WbSide::Right), "more blockable targets on the right");
        s.on_map(Some(clb()));
        assert!(s.chooser.is_smart(), "the session's choice survives a map change");
        assert!(s.side().is_none());
        // A fixed mode still wins outright.
        s.set_mode(WbMode::Left);
        s.update_side((100, 100), 100.0, true, (1, 4), 10, true);
        assert_eq!(s.side(), Some(WbSide::Left));
    }
    fn smart_state() -> WbState {
        let mut s = state();
        s.chooser.set_smart(true, 1);
        s
    }

    /// A tile inside the zone of the right hall, away from its top row.
    fn in_right_zone() -> (i32, i32) {
        let z = clb().right.zone[0];
        (z.x0 + 3, z.y0 + 2)
    }

    /// Drives `update_side` for a bot standing free at `tile` from `from` to `to`; returns the ticks at which the side changed.
    fn drive(s: &mut WbState, tile: (i32, i32), counts: (i32, i32), from: i64, to: i64, fight: bool) -> Vec<i64> {
        let mut changes = Vec::new();
        for t in (from..to).step_by(25) {
            s.fight_here = fight;
            if s.update_side(tile, f64::from(tile.0), true, counts, t, true).is_some() {
                changes.push(t);
            }
        }
        changes
    }

    #[test]
    fn smart_a_committed_switch_takes_us_out_of_the_hall_we_stand_in_and_is_not_undone_by_the_standing_override() {
        let mut s = smart_state();
        let tile = in_right_zone();
        assert!(drive(&mut s, tile, (0, 3), 0, 100, false).is_empty());
        assert_eq!(s.side(), Some(WbSide::Right));
        // The left gets the crowd. The cooldown (from the arrival at tick 0) and the 250 ticks of lead come first; then the side is Left
        // and stays Left while we are still standing in the right zone.
        let changes = drive(&mut s, tile, (6, 0), 100, 4000, false);
        assert_eq!(changes.len(), 1, "one change, no flapping: {changes:?}");
        assert!(
            changes[0] >= WB_SMART_SWITCH_COOLDOWN_TICKS,
            "not inside the cooldown: {changes:?}"
        );
        assert!(
            changes[0] <= WB_SMART_SWITCH_COOLDOWN_TICKS + WB_SWITCH_TICKS + 50,
            "and as soon as it is allowed: {changes:?}"
        );
        assert_eq!(s.side(), Some(WbSide::Left));
        assert_eq!(s.chooser.leaving(), Some(WbSide::Left));
        // Arriving in the left zone completes it.
        let z = clb().left.zone[0];
        drive(&mut s, (z.x0 + 3, z.y0 + 2), (6, 0), 4000, 4100, false);
        assert_eq!(s.chooser.leaving(), None);
        assert_eq!(s.side(), Some(WbSide::Left));
    }

    #[test]
    fn smart_a_fight_in_our_hall_holds_a_switch_back_and_starts_no_cooldown() {
        let mut s = smart_state();
        let tile = in_right_zone();
        drive(&mut s, tile, (0, 3), 0, 100, false);
        // The lead is there from tick 100; a fight goes on until tick 3000: no change, however long.
        assert!(drive(&mut s, tile, (6, 0), 100, 3000, true).is_empty());
        assert_eq!(s.side(), Some(WbSide::Right));
        assert_eq!(s.chooser.leaving(), None, "nothing is half done");
        // The fight is over: the switch comes once it has been gone for WB_FIGHT_HOLD_TICKS (a refused switch started no new cooldown, and the 250 ticks of lead have passed).
        let changes = drive(&mut s, tile, (6, 0), 3000, 3200, false);
        let first = changes.first().copied().expect("the switch comes");
        assert!(
            (3000 + WB_FIGHT_HOLD_TICKS - 25..=3000 + WB_FIGHT_HOLD_TICKS + 25).contains(&first),
            "as soon as the fight has been gone for the hold time, and with no new cooldown: {changes:?}"
        );
        assert_eq!(s.side(), Some(WbSide::Left));
    }

    #[test]
    fn smart_the_first_seconds_are_a_provisional_pick_taken_again_at_every_look_with_no_cooldown() {
        let mut s = smart_state();
        let away = (100, 100); // spawn: in no hall
        let f = |s: &mut WbState, counts, t| {
            s.update_side(away, 100.0, true, counts, t, true);
            s.side()
        };
        assert!(f(&mut s, (0, 0), 0).is_some());
        // One hall occupant on the left on the first looks, then the crowd turns out to be on the right: no lock-in.
        assert_eq!(f(&mut s, (1, 0), 2), Some(WbSide::Left));
        assert_eq!(
            f(&mut s, (1, 4), 4),
            Some(WbSide::Left),
            "a pick that changes the side has to hold first (F6)"
        );
        assert_eq!(f(&mut s, (1, 4), 40), Some(WbSide::Left), "still inside the hold");
        assert_eq!(
            f(&mut s, (1, 4), 60),
            Some(WbSide::Right),
            "re-picked freely during the warm-up, without the margin, once the lead has held"
        );
        assert_eq!(f(&mut s, (4, 1), 300), Some(WbSide::Right));
        assert_eq!(
            f(&mut s, (4, 1), 351),
            Some(WbSide::Left),
            "held again for the 50 ticks: taken again"
        );
        assert_eq!(f(&mut s, (4, 4), 400), Some(WbSide::Left), "a tie keeps the side");
        // After the warm-up the margin and the 250 ticks apply -- but no cooldown, as we have not been in a hall.
        let mut changed = None;
        for t in (WB_SMART_WARMUP_TICKS..WB_SMART_WARMUP_TICKS + 600).step_by(25) {
            if s.update_side(away, 100.0, true, (1, 4), t, true).is_some() {
                changed = Some(t);
                break;
            }
        }
        let t = changed.expect("the side changes");
        assert!(
            t <= WB_SMART_WARMUP_TICKS + WB_SWITCH_TICKS + 25,
            "no cooldown on a side nobody stood in: {t}"
        );
        assert_eq!(s.side(), Some(WbSide::Right));
    }

    /// Review F6 (probe): the counts alternate (2,1)/(1,2) every 10 ticks through the whole warm-up: the side does not flap, and no walk is cut.
    #[test]
    fn smart_the_warm_up_does_not_flap_when_the_counts_flip_every_few_ticks() {
        let mut s = smart_state();
        let away = (100, 100);
        let mut changes = 0;
        for (i, t) in (0..WB_SMART_WARMUP_TICKS).step_by(10).enumerate() {
            let counts = if i % 2 == 0 { (2, 1) } else { (1, 2) };
            if s.update_side(away, 100.0, true, counts, t, true).is_some() {
                changes += 1;
            }
        }
        assert_eq!(changes, 0, "no reported side change (each one cancels the walk)");
        assert_eq!(s.side(), Some(WbSide::Left), "the first pick stands");
    }

    /// Review F7 (probe): a switch committed on one look and cancelled on the next by a fight starts no cooldown and keeps its lead:
    /// it is made again as soon as the fight is over.
    #[test]
    fn smart_a_fight_right_after_the_commit_cancels_it_without_the_cooldown() {
        let mut s = smart_state();
        let tile = in_right_zone();
        drive(&mut s, tile, (0, 3), 0, 100, false);
        let mut commit = None;
        for t in (100..4000).step_by(2) {
            s.fight_here = false;
            if s.update_side(tile, f64::from(tile.0), true, (6, 0), t, true).is_some() {
                commit = Some(t);
                break;
            }
        }
        let c = commit.expect("committed");
        assert_eq!(s.chooser.leaving(), Some(WbSide::Left));
        // Next look: still in the right zone, somebody awake comes near.
        s.fight_here = true;
        let back = s.update_side(tile, f64::from(tile.0), true, (6, 0), c + 2, true);
        assert!(
            back.is_some_and(|b| b.to == WbSide::Right),
            "the fight cancels the switch: {back:?}"
        );
        assert_eq!(s.chooser.leaving(), None);
        // The fight is over: the switch is made again once it has been gone for the hold time -- not a cooldown later.
        s.fight_here = false;
        let mut again = None;
        for t in (c + 4..c + 400).step_by(2) {
            if s.update_side(tile, f64::from(tile.0), true, (6, 0), t, true).is_some() {
                again = Some(t);
                break;
            }
        }
        let again = again.expect("made again");
        assert!(
            again <= c + 2 + WB_FIGHT_HOLD_TICKS + 4,
            "made again as soon as the hold is over, with the cooldown not restarted: {again} after a fight at {}",
            c + 2
        );
    }

    /// Review F11 (probe): a fight that flickers on and off after a committed switch (our target at the edge of its 128 px, a hook on and
    /// off) does not flap the side at tick rate: one change (the cancel) and then none while it flickers.
    #[test]
    fn smart_a_flickering_fight_after_the_commit_does_not_flap_the_side() {
        let mut s = smart_state();
        let tile = in_right_zone();
        drive(&mut s, tile, (0, 3), 0, 100, false);
        let mut commit = None;
        for t in (100..4000).step_by(2) {
            s.fight_here = false;
            if s.update_side(tile, f64::from(tile.0), true, (6, 0), t, true).is_some() {
                commit = Some(t);
                break;
            }
        }
        let c = commit.expect("committed");
        let mut changes = 0;
        for (i, t) in (c + 2..c + 502).enumerate() {
            s.fight_here = i % 2 == 0; // on/off every tick
            changes += usize::from(s.update_side(tile, f64::from(tile.0), true, (6, 0), t, true).is_some());
        }
        assert_eq!(
            changes, 1,
            "the cancel, and nothing else, for 500 ticks of a fight that blinks every tick"
        );
        for (i, t) in (c + 600..c + 1100).enumerate() {
            s.fight_here = (i / 10) % 2 == 0; // on/off every 10 ticks
            changes += usize::from(s.update_side(tile, f64::from(tile.0), true, (6, 0), t, true).is_some());
        }
        assert_eq!(changes, 1, "and nothing while it blinks every 10 ticks");
        assert_eq!(s.side(), Some(WbSide::Right), "the side we hold stays");
    }

    #[test]
    fn smart_the_cooldown_starts_when_we_first_stand_in_a_hall() {
        let mut s = smart_state();
        let tile = in_right_zone();
        // Walking in: the pick follows the counts. At tick 500 we stand in the right zone.
        s.update_side((100, 100), 100.0, true, (0, 2), 0, true);
        assert_eq!(s.side(), Some(WbSide::Right));
        assert!(drive(&mut s, tile, (0, 2), 500, 600, false).is_empty());
        // From the arrival on, a lead for the other side must wait out the cooldown.
        let changes = drive(&mut s, tile, (6, 0), 600, 3000, false);
        assert_eq!(changes.len(), 1);
        assert!(changes[0] >= 500 + WB_SMART_SWITCH_COOLDOWN_TICKS, "{changes:?}");
    }
}
