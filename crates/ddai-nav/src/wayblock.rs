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
}

impl Default for WbSideChooser {
    fn default() -> Self {
        WbSideChooser {
            side: None,
            pending_since: -1,
            provisional: false,
            picked_at: -1,
        }
    }
}

impl WbSideChooser {
    pub fn reset(&mut self) {
        *self = WbSideChooser::default();
    }

    pub fn adopt(&mut self, side: WbSide) {
        self.side = Some(side);
        self.provisional = false;
        self.pending_since = -1;
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
                let mut s = self.chooser.update(counts, here, tick, nearer);
                let standing = if own_free {
                    [WbSide::Left, WbSide::Right]
                        .into_iter()
                        .find(|&sd| in_any_box(&def.side(sd).zone, own_tile.0, own_tile.1))
                } else {
                    None
                };
                if let Some(st) = standing
                    && st != s
                {
                    self.chooser.adopt(st);
                    s = st;
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
}
