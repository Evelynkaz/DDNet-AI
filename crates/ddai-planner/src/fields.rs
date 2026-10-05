//! BFS hazard/dead-zone fields (`docs/research/orig-plan.md` §1.11) and the geometry heuristics
//! built on top of them (§1.9): `hazardField`/`unfreezeField`/`travelField`, `hazardNearness`,
//! `dragCrossesHazard`, `launchLandsInHazard`, `launchFlightLandsInHazard`, `flightEndsInHazard`,
//! `freezeGapPx`, `ropeCatchAlong`. Every function is a literal, line-for-line port of its
//! `planner.ts` counterpart (cited by name); numeric ops go through `ddai_jsmath`.
//!
//! **Descoped, on purpose:** `travelDistance`/`travelDistanceSmooth` (exported by TS, called by
//! nothing in `src/plan`/`src/env` — `docs/research/orig-plan.md` §1.11/§11.1) and `route.ts`'s
//! `deadZoneOf`/Dijkstra routing are not ported here. `routeDistance` (feeds `travel` into
//! `scoreTick`) and `deadZoneCost`/`enemyDeadZoneBonus` (feed `dead`) all default to `false`/`0`
//! in every preset this task's acceptance criteria names (normal/low-cpu/strong/WB/bold — none of
//! them touch these fields), so `travel`/`dead` are always `None` for every teacher-forced/parity
//! decision this crate proves against TS, exactly matching what TS itself computes for those
//! configs (the `travel` field it *does* still compute when `routeDistance: true` is a
//! `docs/research/orig-plan.md` §11.1 dead-code bug: the result is threaded into `scoreTick` but
//! never read there — reproducing that unread computation would only cost cycles for zero
//! observable effect). If a future task needs `deadZoneCost > 0` live, `route.ts`'s BFS dead zone
//! needs its own port first; [`crate::planner::Planner::set_dead_zone`] accepts an
//! externally-computed grid in the meantime (same shape TS's `Bot.setDeadZone` passes in).

use crate::plan_world::PlanCollision;
use crate::tuning::{HAMMER_STRENGTH, PHYSICAL_SIZE, TILE_DEATH, TILE_FREEZE, TILE_NOHOOK, TILE_SOLID, TILE_UNFREEZE};
use crate::vmath::{Vec2, vec2};
use ddai_jsmath as js;

const TILE_PX: f64 = 32.0;
const HAZARD_HORIZON_TILES: i32 = 20;

/// `HazardField` (`planner.ts:341`): a full-map BFS distance grid, in tiles, capped at
/// `0x3fffffff` ("unreached").
#[derive(Debug, Clone)]
pub struct HazardField {
    pub width: i32,
    pub height: i32,
    pub dist: Vec<i32>,
}

const UNREACHED: i32 = 0x3fff_ffff;

fn is_wall(t: u8) -> bool {
    t == TILE_SOLID || t == TILE_NOHOOK
}

/// `bfsField(collision, isSource)` (`planner.ts:354-385`): 4-connected BFS over the game-layer
/// tile grid, walls are `TILE_SOLID`/`TILE_NOHOOK` (freeze/death are *not* walls — the field
/// propagates straight through them, matching TS). Neighbour order `(+1,0),(-1,0),(0,+1),(0,-1)`
/// matters for which of several equal-distance predecessors is enqueued first, though the
/// resulting *distance* grid is order-independent (BFS on an unweighted graph).
fn bfs_field(col: &impl PlanCollision, is_source: impl Fn(u8) -> bool) -> HazardField {
    bfs_field_at(col, |x, y| is_source(col.game_tile(x, y)))
}

/// [`bfs_field`] with the sources given by tile coordinates (so a source may depend on more than the game layer).
fn bfs_field_at(col: &impl PlanCollision, is_source: impl Fn(i32, i32) -> bool) -> HazardField {
    let width = col.width();
    let height = col.height();
    let n = (width * height) as usize;
    let mut dist = vec![UNREACHED; n];
    let mut queue: Vec<i32> = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            if is_source(x, y) {
                dist[idx] = 0;
                queue.push(y * width + x);
            }
        }
    }
    let mut qi = 0usize;
    while qi < queue.len() {
        let idx = queue[qi];
        qi += 1;
        let x = idx % width;
        let y = (idx - x) / width;
        let d = dist[idx as usize];
        for k in 0..4 {
            let nx = x + if k == 0 {
                1
            } else if k == 1 {
                -1
            } else {
                0
            };
            let ny = y + if k == 2 {
                1
            } else if k == 3 {
                -1
            } else {
                0
            };
            if nx < 0 || ny < 0 || nx >= width || ny >= height {
                continue;
            }
            let ni = (ny * width + nx) as usize;
            if is_wall(col.game_tile(nx, ny)) {
                continue;
            }
            if dist[ni] > d + 1 {
                dist[ni] = d + 1;
                queue.push(ny * width + nx);
            }
        }
    }
    HazardField { width, height, dist }
}

/// `hazardField(collision)` (`planner.ts:454-490`): sources are `TILE_FREEZE`/`TILE_DEATH` (game
/// layer only — front-layer freeze/deep-freeze/live-freeze are deliberately invisible to this
/// field, matching TS, `docs/research/orig-plan.md` §9/§11 item 9).
pub fn hazard_field(col: &impl PlanCollision) -> HazardField {
    bfs_field(col, |t| t == TILE_FREEZE || t == TILE_DEATH)
}

/// Distance field to the hazards the *physics* sees (task 3.5b, review F5): the freeze/death tiles of the game layer
/// **and the front layer** (`PlanCollision::is_freeze` of the production adapter looks at both; [`hazard_field`]
/// stays game-layer only on purpose, it is the TS-parity field). Used only to skip the shield far from any hazard.
pub fn hazard_field_full(col: &impl PlanCollision) -> HazardField {
    bfs_field_at(col, |x, y| {
        let (cx, cy) = (
            f64::from(x) * TILE_PX + TILE_PX / 2.0,
            f64::from(y) * TILE_PX + TILE_PX / 2.0,
        );
        col.is_freeze(cx, cy) || col.is_death(cx, cy)
    })
}

/// `unfreezeField(collision)` (`planner.ts:346-352`).
pub fn unfreeze_field(col: &impl PlanCollision) -> HazardField {
    bfs_field(col, |t| t == TILE_UNFREEZE)
}

/// `travelField(collision, fromX, fromY)` (`planner.ts:389-425`): single-source BFS from the tile
/// containing `(from_x, from_y)`, walls are solid/nohook/freeze/death. Kept for structural
/// completeness (the module doc comment explains why nothing feeds it into a live decision).
pub fn travel_field(col: &impl PlanCollision, from_x: f64, from_y: f64) -> HazardField {
    let width = col.width();
    let height = col.height();
    let n = (width * height) as usize;
    let mut dist = vec![UNREACHED; n];
    let sx = js::min((width - 1) as f64, js::max(0.0, js::trunc(from_x / TILE_PX))) as i32;
    let sy = js::min((height - 1) as f64, js::max(0.0, js::trunc(from_y / TILE_PX))) as i32;
    let mut queue: Vec<i32> = Vec::new();
    let start = sy * width + sx;
    dist[start as usize] = 0;
    queue.push(start);
    let mut qi = 0usize;
    while qi < queue.len() {
        let idx = queue[qi];
        qi += 1;
        let x = idx % width;
        let y = (idx - x) / width;
        let d = dist[idx as usize];
        for k in 0..4 {
            let nx = x + if k == 0 {
                1
            } else if k == 1 {
                -1
            } else {
                0
            };
            let ny = y + if k == 2 {
                1
            } else if k == 3 {
                -1
            } else {
                0
            };
            if nx < 0 || ny < 0 || nx >= width || ny >= height {
                continue;
            }
            let ni = (ny * width + nx) as usize;
            let nt = col.game_tile(nx, ny);
            if is_wall(nt) || nt == TILE_FREEZE || nt == TILE_DEATH {
                continue;
            }
            if dist[ni] > d + 1 {
                dist[ni] = d + 1;
                queue.push(ny * width + nx);
            }
        }
    }
    HazardField { width, height, dist }
}

/// `CEILING_NONE` (af49dfb `planner.ts`): "no freeze/death ceiling above".
pub const CEILING_NONE: u8 = 255;
const CEILING_MAX_TILES: u8 = 20;

/// `ceilingField(collision)` (af49dfb `planner.ts`, `ropeCeilingCost`): per tile, how many tiles up the nearest
/// freeze/death ceiling is, looking straight up through open air and also one column to either side (`CEILING_NONE` =
/// none within 20 tiles). Game layer only, like [`hazard_field`]. Cached per map by the planner.
#[derive(Debug, Clone)]
pub struct CeilingField {
    pub width: i32,
    pub height: i32,
    pub dist: Vec<u8>,
}

pub fn ceiling_field(col: &impl PlanCollision) -> CeilingField {
    let width = col.width();
    let height = col.height();
    let n = (width * height) as usize;
    let at = |x: i32, y: i32| (y * width + x) as usize;
    // `own[i]`: tiles up to the ceiling straight above tile `i`; `open[i]`: tiles of open air above it (capped).
    let mut own = vec![CEILING_NONE; n];
    let mut open = vec![0u8; n];
    for x in 0..width {
        for y in 1..height {
            let above = col.game_tile(x, y - 1);
            let i = at(x, y);
            if above == TILE_SOLID || above == TILE_NOHOOK {
                continue;
            }
            open[i] = CEILING_MAX_TILES.min(open[at(x, y - 1)] + 1);
            if above == TILE_FREEZE || above == TILE_DEATH {
                own[i] = 1;
            } else if own[at(x, y - 1)] < CEILING_MAX_TILES {
                own[i] = own[at(x, y - 1)] + 1;
            }
        }
    }
    let beside = |x: i32, y: i32| {
        let t = col.game_tile(x, y);
        t == TILE_FREEZE || t == TILE_DEATH
    };
    let mut dist = vec![CEILING_NONE; n];
    for y in 0..height {
        for x in 0..width {
            let i = at(x, y);
            let mut d = own[i];
            if x > 0 && !beside(x - 1, y) {
                let nb = own[at(x - 1, y)];
                if nb < d && nb <= open[i] {
                    d = nb;
                }
            }
            if x + 1 < width && !beside(x + 1, y) {
                let nb = own[at(x + 1, y)];
                if nb < d && nb <= open[i] {
                    d = nb;
                }
            }
            dist[i] = d;
        }
    }
    CeilingField { width, height, dist }
}

/// `collision.moveBox(pos, vel, size, {x: 0, y: 0})` (`collision.ts` `moveBox`) written over [`PlanCollision::test_box`]
/// (`testBoxAt(x, y, size.x * 0.5, size.y * 0.5)` is exactly `testBox({x, y}, size)`): the slide of a box along the
/// walls, with zero elasticity (a blocked velocity component becomes `-0`/`0`).
fn move_box_no_bounce(col: &impl PlanCollision, pos: &mut Vec2, vel: &mut Vec2, size: Vec2) {
    let mut pos_x = pos.x;
    let mut pos_y = pos.y;
    let mut vel_x = vel.x;
    let mut vel_y = vel.y;
    let elasticity = 0.0_f64;
    let distance = js::sqrt(vel_x * vel_x + vel_y * vel_y);
    let max = js::trunc(distance) as i64;
    if distance > 0.00001 {
        let fraction = 1.0 / ((max + 1) as f64);
        for _ in 0..=max {
            if vel_x == 0.0 && vel_y == 0.0 {
                break;
            }
            let mut new_x = pos_x + vel_x * fraction;
            let mut new_y = pos_y + vel_y * fraction;
            if new_x == pos_x && new_y == pos_y {
                break;
            }
            if col.test_box(vec2(new_x, new_y), size) {
                let mut hits = 0;
                if col.test_box(vec2(pos_x, new_y), size) {
                    new_y = pos_y;
                    vel_y *= -elasticity;
                    hits += 1;
                }
                if col.test_box(vec2(new_x, pos_y), size) {
                    new_x = pos_x;
                    vel_x *= -elasticity;
                    hits += 1;
                }
                if hits == 0 {
                    new_y = pos_y;
                    vel_y *= -elasticity;
                    new_x = pos_x;
                    vel_x *= -elasticity;
                }
            }
            pos_x = new_x;
            pos_y = new_y;
        }
    }
    pos.x = pos_x;
    pos.y = pos_y;
    vel.x = vel_x;
    vel.y = vel_y;
}

/// `ropeIntercept(from, en, collision)` (af49dfb `planner.ts`): where a hook thrown from `from` would meet a victim at
/// `pos` moving with `vel`: the victim's position after the rope's flight time, slid along the walls (two passes,
/// the flight time re-estimated from the first). A still victim stays where it is.
pub fn rope_intercept(col: &impl PlanCollision, from: Vec2, pos: Vec2, vel: Vec2) -> Vec2 {
    let mut x = pos.x;
    let mut y = pos.y;
    let moving = js::abs(vel.x) + js::abs(vel.y) > 0.01;
    let fire = *crate::tuning::HOOK_FIRE_SPEED;
    let tee_box = vec2(PHYSICAL_SIZE, PHYSICAL_SIZE);
    let mut i = 0;
    while i < 2 && moving {
        i += 1;
        let t = js::min(
            *crate::tuning::HOOK_LENGTH / fire,
            js::max(
                0.0,
                (js::hypot2(x - from.x, y - from.y) - PHYSICAL_SIZE * 1.5 - fire / 2.0) / fire,
            ),
        );
        let mut p = pos;
        let mut v = vel;
        let mut left = t;
        while left > 0.0 {
            let f = js::min(1.0, left);
            let mut step = vec2(v.x * f, v.y * f);
            move_box_no_bounce(col, &mut p, &mut step, tee_box);
            if f == 1.0 {
                v = step;
            }
            left -= 1.0;
        }
        x = p.x;
        y = p.y;
    }
    vec2(x, y)
}

/// `hazardNearness(field, x, y)` (`planner.ts:500-507`): `floor(x/32)` tile addressing (not the
/// collision's own `indexAt`/`getMapIndex`), out of bounds -> `0`.
pub fn hazard_nearness(field: &HazardField, x: f64, y: f64) -> f64 {
    let tx = js::floor(x / TILE_PX) as i32;
    let ty = js::floor(y / TILE_PX) as i32;
    if tx < 0 || ty < 0 || tx >= field.width || ty >= field.height {
        return 0.0;
    }
    let d = field.dist[(ty * field.width + tx) as usize];
    if d >= HAZARD_HORIZON_TILES {
        return 0.0;
    }
    f64::from(HAZARD_HORIZON_TILES - d) / f64::from(HAZARD_HORIZON_TILES)
}

/// Path distance in tiles (BFS around walls) from `(x, y)` to the nearest hazard tile the field was built
/// from; `i32::MAX` outside the map or when no hazard is reachable. Task 3.5b: the hybrid shield is
/// skipped when this is large (nothing to freeze on within reach of the own path).
pub fn hazard_tiles(field: &HazardField, x: f64, y: f64) -> i32 {
    let tx = js::floor(x / TILE_PX) as i32;
    let ty = js::floor(y / TILE_PX) as i32;
    if tx < 0 || ty < 0 || tx >= field.width || ty >= field.height {
        return i32::MAX;
    }
    let d = field.dist[(ty * field.width + tx) as usize];
    if d >= UNREACHED { i32::MAX } else { d }
}

/// `wrapAngle(a)` (`planner.ts:492-496`): while-loop wrap into `[-pi, pi]`, not a single `rem`
/// (matches TS's own loop exactly, including its behavior for already-huge inputs).
pub fn wrap_angle(mut a: f64) -> f64 {
    while a > js::PI {
        a -= 2.0 * js::PI;
    }
    while a < -js::PI {
        a += 2.0 * js::PI;
    }
    a
}

/// `dragCrossesHazard(collision, at, from, separation)` (`planner.ts:610-621`).
pub fn drag_crosses_hazard(col: &impl PlanCollision, at: Vec2, from: Vec2, separation: f64) -> f64 {
    let dx = (from.x - at.x) / separation;
    let dy = (from.y - at.y) / separation;
    for px in [TILE_PX, 2.0 * TILE_PX, 3.0 * TILE_PX] {
        if px >= separation {
            return 0.0;
        }
        let x = at.x + dx * px;
        let y = at.y + dy * px;
        if col.is_solid(x, y) {
            return 0.0;
        }
        if col.is_hazard(x, y) {
            return 1.0;
        }
    }
    0.0
}

/// `launchLandsInHazard(collision, at, from, separation)` (`planner.ts:623-639`).
pub fn launch_lands_in_hazard(col: &impl PlanCollision, at: Vec2, from: Vec2, separation: f64) -> f64 {
    let hx = if separation > 0.0 {
        (at.x - from.x) / separation
    } else {
        0.0
    };
    let hy = if separation > 0.0 {
        (at.y - from.y) / separation
    } else {
        -1.0
    };
    let bx = hx;
    let by = hy - 1.1;
    let bl_raw = js::hypot2(bx, by);
    let bl = if bl_raw == 0.0 { 1.0 } else { bl_raw };
    let dx = bx / bl;
    let dy = by / bl;
    for px in [3.0 * TILE_PX, 5.0 * TILE_PX, 7.0 * TILE_PX] {
        let x = at.x + dx * px;
        let y = at.y + dy * px;
        if col.is_solid(x, y) {
            return 0.0;
        }
        if col.is_hazard(x, y) {
            return 1.0;
        }
    }
    0.0
}

pub const ROPE_CATCH_PX: f64 = PHYSICAL_SIZE + 6.0;

/// `ropeCatchAlong(from, dir, at)` (`planner.ts:642-648`).
pub fn rope_catch_along(from: Vec2, dir: Vec2, at: Vec2, hook_length: f64) -> f64 {
    let rx = at.x - from.x;
    let ry = at.y - from.y;
    let along = rx * dir.x + ry * dir.y;
    if along < 0.0 || along > hook_length {
        return f64::INFINITY;
    }
    if js::abs(rx * dir.y - ry * dir.x) <= ROPE_CATCH_PX {
        along
    } else {
        f64::INFINITY
    }
}

const LAUNCH_FLIGHT_TICKS: i32 = 50;
const LAUNCH_PROBE_STEP_PX: f64 = TILE_PX / 2.0;
const NO_VEL: Vec2 = vec2(0.0, 0.0);

/// `freeFraction(collision, x, y, dx, dy)` (`planner.ts:697-711`): fraction of `(dx, dy)` the
/// 28x28 tee box can move before it would first overlap solid ground, via 5 rounds of bisection.
fn free_fraction(col: &impl PlanCollision, x: f64, y: f64, dx: f64, dy: f64) -> f64 {
    let box_size = vec2(PHYSICAL_SIZE, PHYSICAL_SIZE);
    if !col.test_box(vec2(x + dx, y + dy), box_size) {
        return 1.0;
    }
    let mut lo = 0.0;
    let mut hi = 1.0;
    for _ in 0..5 {
        let mid = (lo + hi) / 2.0;
        if col.test_box(vec2(x + dx * mid, y + dy * mid), box_size) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    lo
}

/// `launchFlightLandsInHazard(collision, at, from, separation, vel)` (`planner.ts:650-695`): a
/// simplified (not real-physics) 50-tick hammer-launch simulation. Deliberately not real DDNet
/// physics (`docs/research/orig-plan.md` §11 item 13) — reproduced as-is, on the parity path.
pub fn launch_flight_lands_in_hazard(
    col: &impl PlanCollision,
    at: Vec2,
    from: Vec2,
    separation: f64,
    vel: Vec2,
) -> f64 {
    let hx = if separation > 0.0 {
        (at.x - from.x) / separation
    } else {
        0.0
    };
    let hy = if separation > 0.0 {
        (at.y - from.y) / separation
    } else {
        -1.0
    };
    let bx = hx;
    let by = hy - 1.1;
    let bl_raw = js::hypot2(bx, by);
    let bl = if bl_raw == 0.0 { 1.0 } else { bl_raw };
    let k = *HAMMER_STRENGTH;
    let mut vx = vel.x + (k * 10.0 * bx) / bl;
    let mut vy = vel.y + k * (-1.0 + (10.0 * by) / bl);
    let mut x = at.x;
    let mut y = at.y;
    // The tuning constants are `LazyLock`s: read them once, not on every one of the 50 ticks.
    let (gravity, ground_friction, air_friction) = (
        *crate::tuning::GRAVITY,
        *crate::tuning::GROUND_FRICTION,
        *crate::tuning::AIR_FRICTION,
    );
    let (ramp_start, ramp_curvature, ramp_range) = (
        *crate::tuning::VELRAMP_START,
        *crate::tuning::VELRAMP_CURVATURE,
        *crate::tuning::VELRAMP_RANGE,
    );

    let half = PHYSICAL_SIZE / 2.0;
    let mut grounded = col.is_solid(x + half, y + half + 5.0) || col.is_solid(x - half, y + half + 5.0);
    for _t in 0..LAUNCH_FLIGHT_TICKS {
        vy += gravity;
        vx *= if grounded { ground_friction } else { air_friction };
        grounded = false;

        let speed = js::hypot2(vx, vy) * 50.0;
        let ramp = if speed < ramp_start {
            1.0
        } else {
            1.0 / js::pow(ramp_curvature, (speed - ramp_start) / ramp_range)
        };
        let n = js::max(1.0, js::ceil(js::max(js::abs(vx), js::abs(vy)) / LAUNCH_PROBE_STEP_PX)) as i32;
        for _i in 0..n {
            let sx = (vx * ramp) / f64::from(n);
            if sx != 0.0 {
                let f = free_fraction(col, x, y, sx, 0.0);
                x += sx * f;
                if f < 1.0 {
                    vx = 0.0;
                }
            }
            let mut landed = false;
            let sy = vy / f64::from(n);
            if sy != 0.0 {
                let f = free_fraction(col, x, y, 0.0, sy);
                y += sy * f;
                if f < 1.0 {
                    landed = vy > 0.0;
                    vy = 0.0;
                }
            }
            if col.is_hazard(x, y) {
                return 1.0;
            }
            if landed {
                return 0.0;
            }
        }
    }
    0.0
}

/// A direct-mapped memo of [`launch_flight_lands_in_hazard`] (task 3.6). The function is a pure
/// function of `(at, (at - from) / separation, vel)` and the map; inside one hybrid decision every
/// candidate's rollout starts from the same state and many share their first ticks, so the same
/// arguments come back about four times in ten. An entry is only valid in the epoch it was written
/// (the owner bumps [`LaunchMemo::new_epoch`] whenever the map or the decision may have changed), and
/// a hit must match all six argument words bit for bit, so a memoised answer is the very value the
/// function would compute.
pub struct LaunchMemo {
    slots: Vec<MemoSlot>,
    epoch: u64,
}

#[derive(Clone, Copy, Default)]
struct MemoSlot {
    epoch: u64,
    key: [u64; 6],
    value: f64,
}

const LAUNCH_MEMO_SLOTS: usize = 1024;

impl Default for LaunchMemo {
    fn default() -> Self {
        LaunchMemo {
            slots: vec![MemoSlot::default(); LAUNCH_MEMO_SLOTS],
            epoch: 1,
        }
    }
}

impl LaunchMemo {
    /// Invalidates every entry (O(1)): the next lookups recompute.
    pub fn new_epoch(&mut self) {
        self.epoch += 1;
    }
}

/// [`launch_flight_lands_in_hazard`], through `memo` when there is one (`None`: just compute it).
pub fn launch_flight_lands_in_hazard_memo(
    memo: Option<&mut LaunchMemo>,
    col: &impl PlanCollision,
    at: Vec2,
    from: Vec2,
    separation: f64,
    vel: Vec2,
) -> f64 {
    let Some(memo) = memo else {
        return launch_flight_lands_in_hazard(col, at, from, separation, vel);
    };
    // The same `hx`/`hy` the function derives from `from` and `separation` (nothing else of `from` is read).
    let hx = if separation > 0.0 {
        (at.x - from.x) / separation
    } else {
        0.0
    };
    let hy = if separation > 0.0 {
        (at.y - from.y) / separation
    } else {
        -1.0
    };
    let key = [
        at.x.to_bits(),
        at.y.to_bits(),
        hx.to_bits(),
        hy.to_bits(),
        vel.x.to_bits(),
        vel.y.to_bits(),
    ];
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for k in key {
        h = (h ^ k).wrapping_mul(0x0000_0100_0000_01b3);
        h ^= h >> 29;
    }
    let slot = &mut memo.slots[(h >> 40) as usize % LAUNCH_MEMO_SLOTS];
    if slot.epoch == memo.epoch && slot.key == key {
        return slot.value;
    }
    let value = launch_flight_lands_in_hazard(col, at, from, separation, vel);
    *slot = MemoSlot {
        epoch: memo.epoch,
        key,
        value,
    };
    value
}

const FLIGHT_PROBE_TICKS: [f64; 4] = [6.0, 12.0, 18.0, 24.0];

/// `flightEndsInHazard(collision, pos, vel)` (`planner.ts:778-790`).
pub fn flight_ends_in_hazard(col: &impl PlanCollision, pos: Vec2, vel: Vec2) -> f64 {
    let half = PHYSICAL_SIZE / 2.0;
    let grounded = col.is_solid(pos.x + half, pos.y + half + 5.0) || col.is_solid(pos.x - half, pos.y + half + 5.0);
    for t in FLIGHT_PROBE_TICKS {
        let x = pos.x + vel.x * t;
        let y = if grounded {
            pos.y
        } else {
            pos.y + vel.y * t + 0.5 * *crate::tuning::GRAVITY * t * t
        };
        if col.is_solid(x, y) {
            return 0.0;
        }
        if col.is_hazard(x, y) {
            return 1.0;
        }
    }
    0.0
}

const EDGE_GAP_TILES: i32 = 3;
pub const EDGE_GAP_PX: f64 = (EDGE_GAP_TILES as i64 as f64) * TILE_PX;

/// `freezeGapPx(collision, x, y)` (`planner.ts:797-815`).
pub fn freeze_gap_px(col: &impl PlanCollision, x: f64, y: f64) -> f64 {
    let tx = js::floor(x / TILE_PX) as i32;
    let ty = js::floor(y / TILE_PX) as i32;
    let mut best = EDGE_GAP_PX;
    for oy in -EDGE_GAP_TILES..=EDGE_GAP_TILES {
        for ox in -EDGE_GAP_TILES..=EDGE_GAP_TILES {
            let left = f64::from(tx + ox) * TILE_PX;
            let top = f64::from(ty + oy) * TILE_PX;
            let cx = left + TILE_PX / 2.0;
            let cy = top + TILE_PX / 2.0;
            if !col.is_freeze(cx, cy) && !col.is_death(cx, cy) {
                continue;
            }
            let dx = js::max_n(&[left - x, 0.0, x - (left + TILE_PX)]);
            let dy = js::max_n(&[top - y, 0.0, y - (top + TILE_PX)]);
            let d = js::hypot2(dx, dy);
            if d < best {
                best = d;
            }
        }
    }
    best
}

/// Unused function silencer: `NO_VEL` is a documentation-only default used by callers in
/// `crate::planner` (kept here since it's this module's own vocabulary constant).
pub const fn no_vel() -> Vec2 {
    NO_VEL
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan_world::LineHit;
    use crate::vmath::vdistance;

    struct FlatCol {
        w: i32,
        h: i32,
        tiles: Vec<u8>,
    }
    impl PlanCollision for FlatCol {
        fn identity(&self) -> u64 {
            0
        }
        fn width(&self) -> i32 {
            self.w
        }
        fn height(&self) -> i32 {
            self.h
        }
        fn game_tile(&self, tx: i32, ty: i32) -> u8 {
            self.tiles[(ty * self.w + tx) as usize]
        }
        fn is_solid(&self, x: f64, y: f64) -> bool {
            self.tile_at(x, y) == TILE_SOLID
        }
        fn is_death(&self, x: f64, y: f64) -> bool {
            self.tile_at(x, y) == TILE_DEATH
        }
        fn is_freeze(&self, x: f64, y: f64) -> bool {
            self.tile_at(x, y) == TILE_FREEZE
        }
        fn is_un_freeze(&self, x: f64, y: f64) -> bool {
            self.tile_at(x, y) == TILE_UNFREEZE
        }
        fn is_no_hook(&self, x: f64, y: f64) -> bool {
            self.tile_at(x, y) == TILE_NOHOOK
        }
        fn test_box(&self, _pos: Vec2, _size: Vec2) -> bool {
            false
        }
        fn intersect_line(&self, _a: Vec2, _b: Vec2) -> LineHit {
            LineHit {
                collision: 0,
                out_pos: _b,
                out_before_pos: _b,
            }
        }
        fn intersect_line_hook(&self, a: Vec2, b: Vec2) -> LineHit {
            self.intersect_line(a, b)
        }
        fn has_tele(&self) -> bool {
            false
        }
        fn tele_at(&self, _x: f64, _y: f64) -> (i32, i32) {
            (0, 0)
        }
        fn tele_outs_for(&self, _n: i32) -> Vec<Vec2> {
            Vec::new()
        }
    }
    impl FlatCol {
        fn tile_at(&self, x: f64, y: f64) -> u8 {
            let tx = js::floor(x / TILE_PX) as i32;
            let ty = js::floor(y / TILE_PX) as i32;
            if tx < 0 || ty < 0 || tx >= self.w || ty >= self.h {
                return TILE_SOLID;
            }
            self.tiles[(ty * self.w + tx) as usize]
        }
    }

    fn strip_with_freeze() -> FlatCol {
        // 5x3: row 0 air, row 1 freeze in the middle column, row 2 solid floor.
        let w = 5;
        let h = 3;
        let mut tiles = vec![TILE_SOLID; (w * h) as usize];
        for x in 0..w {
            tiles[x as usize] = TILE_AIR_FOR_TEST;
            tiles[(w + x) as usize] = TILE_AIR_FOR_TEST;
        }
        tiles[(w + 2) as usize] = TILE_FREEZE;
        FlatCol { w, h, tiles }
    }
    const TILE_AIR_FOR_TEST: u8 = 0;

    /// Task 3.6: the memoised launch-flight check returns exactly what the function computes, for
    /// arguments that repeat, that collide in the memo and that straddle an epoch change.
    #[test]
    fn launch_flight_memo_returns_the_exact_value() {
        let (w, h) = (40, 30);
        let mut s = 0xA5A5_1234_5678_9ABCu64;
        let mut next = move || {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            s.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        let mut tiles = vec![TILE_AIR_FOR_TEST; (w * h) as usize];
        for (i, tile) in tiles.iter_mut().enumerate() {
            let (x, y) = (i as i32 % w, i as i32 / w);
            *tile = if x == 0 || y == 0 || x == w - 1 || y == h - 1 || next() % 9 == 0 {
                TILE_SOLID
            } else if next() % 30 == 0 {
                TILE_FREEZE
            } else {
                TILE_AIR_FOR_TEST
            };
        }
        let col = FlatCol { w, h, tiles };
        let mut memo = LaunchMemo::default();
        let mut args: Vec<(Vec2, Vec2, Vec2)> = Vec::new();
        let (mut ones, mut zeros) = (0, 0);
        for k in 0..6_000u32 {
            let (at, from, vel) = if k % 2 == 1 && !args.is_empty() {
                args[(next() % args.len() as u64) as usize]
            } else {
                let at = vec2(64.0 + (next() % 1100) as f64, 64.0 + (next() % 800) as f64);
                let from = vec2(at.x + (next() % 120) as f64 - 60.0, at.y + (next() % 120) as f64 - 60.0);
                (
                    at,
                    from,
                    vec2((next() % 400) as f64 / 10.0 - 20.0, (next() % 400) as f64 / 10.0 - 20.0),
                )
            };
            args.push((at, from, vel));
            let sep = vdistance(at, from);
            let want = launch_flight_lands_in_hazard(&col, at, from, sep, vel);
            let got = launch_flight_lands_in_hazard_memo(Some(&mut memo), &col, at, from, sep, vel);
            assert_eq!(got.to_bits(), want.to_bits(), "case {k}");
            assert_eq!(
                launch_flight_lands_in_hazard_memo(None, &col, at, from, sep, vel).to_bits(),
                want.to_bits()
            );
            if want == 1.0 {
                ones += 1;
            } else {
                zeros += 1;
            }
            if k % 997 == 996 {
                memo.new_epoch();
            }
        }
        assert!(ones > 100 && zeros > 100, "both outcomes must occur ({ones}, {zeros})");
    }

    #[test]
    fn hazard_nearness_is_1_at_the_freeze_tile_itself() {
        let col = strip_with_freeze();
        let field = hazard_field(&col);
        // Tile (2,1) center.
        let n = hazard_nearness(&field, 2.0 * TILE_PX + 16.0, 1.0 * TILE_PX + 16.0);
        assert_eq!(n, 1.0);
    }

    #[test]
    fn hazard_nearness_decays_with_distance_and_is_zero_far_away() {
        let col = strip_with_freeze();
        let field = hazard_field(&col);
        let near = hazard_nearness(&field, 2.0 * TILE_PX + 16.0, 0.0 * TILE_PX + 16.0);
        let far = hazard_nearness(&field, 0.0, 0.0);
        assert!(
            near > 0.0 && near < 1.0,
            "adjacent tile should be a partial nearness: {near}"
        );
        assert!(far < near);
    }

    #[test]
    fn wrap_angle_normalizes_into_pi_range() {
        let w = wrap_angle(3.0 * js::PI);
        assert!((-js::PI..=js::PI).contains(&w));
    }

    #[test]
    fn drag_crosses_hazard_detects_freeze_between_two_points() {
        let col = strip_with_freeze();
        let at = vec2(0.0 * TILE_PX + 16.0, 1.0 * TILE_PX + 16.0);
        let from = vec2(4.0 * TILE_PX + 16.0, 1.0 * TILE_PX + 16.0);
        let sep = vdistance(at, from);
        assert_eq!(drag_crosses_hazard(&col, at, from, sep), 1.0);
    }
}
