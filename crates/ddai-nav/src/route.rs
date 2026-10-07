//! `src/plan/route.ts`: the walk/fall/jump/hook/freeze-crossing graph search (`findRoute`), the
//! spawn tiles and the dead zone. A line-by-line port: neighbour order, costs, tie-breaking of the
//! heap and the stamp scheme are kept exactly so that the routes match the TS ones step for step
//! (`tests/parity_nav.rs`).

use ddai_jsmath as js;
use ddai_physics::map::MapData;
use ddai_planner::plan_world::PlanCollision;
use std::collections::HashSet;

use crate::TILE_PX;
use crate::grid::{NavGrid, RAY_DX, RAY_DY};

/// `MoveKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveKind {
    Walk,
    Fall,
    Jump,
    Hook,
    Kill,
}

/// `RouteStep` (`route.ts:6-22`).
#[derive(Debug, Clone, PartialEq)]
pub struct RouteStep {
    pub x: i32,
    pub y: i32,
    pub kind: MoveKind,
    pub anchor: Option<(i32, i32)>,
    pub freeze: bool,
    pub tele: bool,
    pub move_key: i32,
    pub leap: bool,
}

/// `routeMoveKey(index, kind)`.
pub fn route_move_key(index: i32, kind: i32) -> i32 {
    index * 8 + kind
}

/// `RouteResult`.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteResult {
    pub steps: Vec<RouteStep>,
    pub cost: i32,
}

const HOOK_LENGTH_PX: f64 = 380.0; // `TUNING.hookLength`
const HOOK_TILES: i32 = 11; // `Math.floor(TUNING.hookLength / TILE_PX)`

const JUMP_UP_REACH: [f64; 6] = [7.7, 7.3, 6.9, 6.3, 5.6, 4.4];
const JUMP_DOWN_REACH: [f64; 7] = [7.7, 8.1, 8.4, 8.8, 9.1, 9.4, 9.7];
const JUMP_MARGIN: f64 = 0.8;

const COST_WALK: i32 = 1;
const COST_FALL: i32 = 1;
const COST_JUMP: i32 = 3;
const COST_HOOK: i32 = 6;
const COST_KILL: i32 = 60;
const COST_FREEZE_FALL: i32 = 25;
const FREEZE_FALL_TILES: i32 = 3;
const COST_FREEZE_CROSS: i32 = 20;
const FREEZE_CROSS_TILES: i32 = 3;
const MAX_TRAP_TILES: usize = 250;
const COST_DANGER: i32 = 4;
const OVERSHOOT_DX: i32 = 2;
const FREEZE_KIND: u8 = 5;
const ENTITY_SPAWN: u8 = 192;

pub const UNREACHED: i32 = 0x3fff_ffff;

/// `class Heap` (`route.ts:191-245`): binary min-heap of `(index, cost)` with a fixed capacity;
/// `push` into a full heap is silently dropped (a TS quirk kept for parity; counted in `dropped`).
#[derive(Debug, Clone)]
struct Heap {
    idx: Vec<i32>,
    cost: Vec<i32>,
    size: usize,
    dropped: u64,
}

impl Heap {
    fn new(capacity: usize) -> Heap {
        Heap {
            idx: vec![0; capacity],
            cost: vec![0; capacity],
            size: 0,
            dropped: 0,
        }
    }
    fn empty(&self) -> bool {
        self.size == 0
    }
    fn clear(&mut self) {
        self.size = 0;
    }
    fn push(&mut self, index: i32, cost: i32) {
        if self.size >= self.idx.len() {
            self.dropped += 1;
            return;
        }
        let mut i = self.size;
        self.size += 1;
        self.idx[i] = index;
        self.cost[i] = cost;
        while i > 0 {
            let p = (i - 1) >> 1;
            if self.cost[p] <= self.cost[i] {
                break;
            }
            self.swap(p, i);
            i = p;
        }
    }
    fn pop(&mut self) -> (i32, i32) {
        let out = (self.idx[0], self.cost[0]);
        self.size -= 1;
        if self.size > 0 {
            self.idx[0] = self.idx[self.size];
            self.cost[0] = self.cost[self.size];
            let mut i = 0usize;
            loop {
                let l = 2 * i + 1;
                let r = l + 1;
                let mut m = i;
                if l < self.size && self.cost[l] < self.cost[m] {
                    m = l;
                }
                if r < self.size && self.cost[r] < self.cost[m] {
                    m = r;
                }
                if m == i {
                    break;
                }
                self.swap(m, i);
                i = m;
            }
        }
        out
    }
    fn swap(&mut self, a: usize, b: usize) {
        self.idx.swap(a, b);
        self.cost.swap(a, b);
    }
}

fn overshoots_into_hazard(g: &NavGrid, nx: i32, ny: i32, dx: i32) -> bool {
    if dx.abs() < OVERSHOOT_DX {
        return false;
    }
    let bx = nx + dx.signum();
    if bx < 0 || bx >= g.width {
        return false;
    }
    let hazard = |i: usize| g.free[i] == 0 && g.solid[i] == 0;
    if hazard(g.idx(bx, ny)) {
        return true;
    }
    ny + 1 < g.height && g.free[g.idx(bx, ny)] == 1 && hazard(g.idx(bx, ny + 1))
}

#[allow(clippy::too_many_arguments)]
fn jump_clear(g: &NavGrid, x: i32, y: i32, nx: i32, ny: i32, dx: i32, _dy: i32, reach: f64) -> bool {
    if f64::from(dx.abs()) > reach {
        return false;
    }
    let apex = y.min(ny);
    let mut yy = y - 1;
    while yy >= apex {
        if g.free[g.idx(x, yy)] == 0 {
            return false;
        }
        yy -= 1;
    }
    if !g.clear_line(x, apex, nx, apex, true) {
        return false;
    }
    g.clear_line(nx, apex, nx, ny, true)
}

/// `freezeCrossing` (`route.ts:459-473`): crossing a freeze strip of up to 3 tiles from `(x, y)` in
/// direction `dx`; the out tile and the number of freeze tiles.
fn freeze_crossing(g: &NavGrid, x: i32, y: i32, dx: i32, loose: bool) -> Option<(i32, i32)> {
    if !loose && !g.supported(x, y) {
        return None;
    }
    for d in 1..=FREEZE_CROSS_TILES {
        let nx = x + dx * d;
        if nx < 0 || nx >= g.width {
            return None;
        }
        let i = g.idx(nx, y);
        if g.free[i] == 1 {
            return None;
        }
        if g.solid[i] == 1 || g.death[i] == 1 {
            return None;
        }
        if !loose && g.supported(nx, y) {
            return None;
        }
        let out_x = nx + dx;
        if out_x < 0 || out_x >= g.width {
            return None;
        }
        if g.free[g.idx(out_x, y)] == 1 {
            return Some((out_x, d));
        }
    }
    None
}

/// `Visit`: `(nx, ny, cost, kind, anchor)`.
type Visit<'a> = dyn FnMut(i32, i32, i32, u8, i32) + 'a;

/// `expand` (`route.ts:282-366`).
fn expand(
    g: &NavGrid,
    x: i32,
    y: i32,
    visit: &mut Visit<'_>,
    spawns: Option<&[i32]>,
    through_freeze: bool,
    loose: bool,
) {
    if let Some(spawns) = spawns {
        for &si in spawns {
            if si != y * g.width + x {
                visit(si % g.width, si / g.width, COST_KILL, 4, -1);
            }
        }
    }
    for dx in [-1, 1] {
        let nx = x + dx;
        if nx < 0 || nx >= g.width {
            continue;
        }
        if g.free[g.idx(nx, y)] == 0 {
            continue;
        }
        if !g.supported(x, y) && !g.supported(nx, y) {
            continue;
        }
        visit(nx, y, COST_WALK, 0, -1);
    }
    for dx in [0, -1, 1] {
        let nx = x + dx;
        let ny = y + 1;
        if nx < 0 || nx >= g.width || ny >= g.height {
            continue;
        }
        if g.free[g.idx(nx, ny)] == 0 {
            continue;
        }
        if dx != 0 && g.free[g.idx(nx, y)] == 0 {
            continue;
        }
        visit(nx, ny, COST_FALL, 1, -1);
    }
    if through_freeze {
        for d in 1..=FREEZE_FALL_TILES {
            let ny = y + d;
            if ny >= g.height {
                break;
            }
            let i = g.idx(x, ny);
            if g.free[i] == 1 {
                break;
            }
            if g.solid[i] == 1 || g.death[i] == 1 {
                break;
            }
            let out_y = ny + 1;
            if out_y >= g.height {
                break;
            }
            if g.free[g.idx(x, out_y)] == 1 {
                visit(x, out_y, COST_FREEZE_FALL * d, FREEZE_KIND, -1);
                break;
            }
        }
        for dx in [-1, 1] {
            let Some((out_x, tiles)) = freeze_crossing(g, x, y, dx, loose) else {
                continue;
            };
            let cost = if g.unfreeze[g.idx(out_x, y)] == 1 {
                COST_FREEZE_CROSS
            } else {
                COST_FREEZE_CROSS * 2
            };
            visit(out_x, y, cost * tiles, FREEZE_KIND, -1);
        }
    }
    if g.supported(x, y) {
        let lo = -(JUMP_UP_REACH.len() as i32 - 1);
        let hi = JUMP_DOWN_REACH.len() as i32 - 1;
        for dy in lo..=hi {
            let ny = y + dy;
            if ny < 0 || ny >= g.height {
                continue;
            }
            let reach = (if dy <= 0 {
                JUMP_UP_REACH[(-dy) as usize]
            } else {
                JUMP_DOWN_REACH[dy as usize]
            }) * JUMP_MARGIN;
            let span = reach.trunc() as i32;
            for dx in -span..=span {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x + dx;
                if nx < 0 || nx >= g.width {
                    continue;
                }
                if g.free[g.idx(nx, ny)] == 0 {
                    continue;
                }
                if !g.supported(nx, ny) {
                    continue;
                }
                if overshoots_into_hazard(g, nx, ny, dx) {
                    continue;
                }
                if !jump_clear(g, x, y, nx, ny, dx, dy, reach) {
                    continue;
                }
                visit(nx, ny, COST_JUMP + dx.abs() + (-dy).max(0), 2, -1);
            }
        }
    }
    let tele_out = g.tele_out[g.idx(x, y)];
    if tele_out >= 0 {
        visit(tele_out % g.width, tele_out / g.width, COST_WALK, 0, -1);
    }
    let n = (g.width * g.height) as usize;
    for d in 0..8usize {
        let ai = g.first_solid[d * n + g.idx(x, y)];
        if ai < 0 || g.hookable[ai as usize] == 0 {
            continue;
        }
        let ax = ai % g.width;
        let ay = (ai - ax) / g.width;
        let k = (ax - x).abs().max((ay - y).abs());
        if js::hypot2(f64::from(ax - x), f64::from(ay - y)) * f64::from(TILE_PX) > HOOK_LENGTH_PX {
            continue;
        }
        let tx = ax - RAY_DX[d];
        let ty = ay - RAY_DY[d];
        if g.free[g.idx(tx, ty)] == 1 && g.clear_line(x, y, tx, ty, true) {
            visit(tx, ty, COST_HOOK + k, 3, ai);
        }
    }
}

/// `expandBack` (`route.ts:368-457`): the reverse edges (used by [`dead_zone`]).
fn expand_back(g: &NavGrid, x: i32, y: i32, visit: &mut Visit<'_>, loose: bool) {
    for dx in [-1, 1] {
        let ax = x + dx;
        if ax < 0 || ax >= g.width {
            continue;
        }
        if g.free[g.idx(ax, y)] == 0 {
            continue;
        }
        if !g.supported(x, y) && !g.supported(ax, y) {
            continue;
        }
        visit(ax, y, COST_WALK, 0, -1);
    }
    for dx in [0, -1, 1] {
        let ax = x + dx;
        let ay = y - 1;
        if ax < 0 || ax >= g.width || ay < 0 {
            continue;
        }
        if g.free[g.idx(ax, ay)] == 0 {
            continue;
        }
        if dx != 0 && g.free[g.idx(ax, y)] == 0 {
            continue;
        }
        visit(ax, ay, COST_FALL, 1, -1);
    }
    let lo = -(JUMP_UP_REACH.len() as i32 - 1);
    let hi = JUMP_DOWN_REACH.len() as i32 - 1;
    for dy in lo..=hi {
        let ay = y - dy;
        if ay < 0 || ay >= g.height {
            continue;
        }
        let reach = (if dy <= 0 {
            JUMP_UP_REACH[(-dy) as usize]
        } else {
            JUMP_DOWN_REACH[dy as usize]
        }) * JUMP_MARGIN;
        let span = reach.trunc() as i32;
        for dx in -span..=span {
            if dx == 0 && dy == 0 {
                continue;
            }
            let ax = x - dx;
            if ax < 0 || ax >= g.width {
                continue;
            }
            if g.free[g.idx(ax, ay)] == 0 {
                continue;
            }
            if !g.supported(ax, ay) {
                continue;
            }
            if !g.supported(x, y) {
                continue;
            }
            if overshoots_into_hazard(g, x, y, dx) {
                continue;
            }
            if !jump_clear(g, ax, ay, x, y, dx, dy, reach) {
                continue;
            }
            visit(ax, ay, COST_JUMP + dx.abs() + (-dy).max(0), 2, -1);
        }
    }
    for dx in [-1, 1] {
        for d in 1..=FREEZE_CROSS_TILES {
            let nx = x + dx * d;
            if nx < 0 || nx >= g.width {
                break;
            }
            let i = g.idx(nx, y);
            if g.free[i] == 1 {
                break;
            }
            if g.solid[i] == 1 || g.death[i] == 1 {
                break;
            }
            let from_x = nx + dx;
            if from_x < 0 || from_x >= g.width {
                break;
            }
            if g.free[g.idx(from_x, y)] == 1 {
                if let Some((out_x, _)) = freeze_crossing(g, from_x, y, -dx, loose)
                    && out_x == x
                {
                    visit(from_x, y, COST_FREEZE_CROSS * d, 1, -1);
                }
                break;
            }
        }
    }
    for d in 1..=FREEZE_FALL_TILES {
        let ny = y - d;
        if ny < 0 {
            break;
        }
        let i = g.idx(x, ny);
        if g.free[i] == 1 {
            break;
        }
        if g.solid[i] == 1 || g.death[i] == 1 {
            break;
        }
        let from_y = ny - 1;
        if from_y < 0 {
            break;
        }
        if g.free[g.idx(x, from_y)] == 1 {
            visit(x, from_y, COST_FREEZE_FALL * d, 1, -1);
            break;
        }
    }
    if let Some(froms) = g.tele_in.get(&(y * g.width + x)) {
        for &from in froms {
            visit(from % g.width, from / g.width, COST_WALK, 0, -1);
        }
    }
    for d in 0..8usize {
        let (dx, dy) = (RAY_DX[d], RAY_DY[d]);
        let ax = x + dx;
        let ay = y + dy;
        if ax < 0 || ay < 0 || ax >= g.width || ay >= g.height {
            continue;
        }
        if g.hookable[g.idx(ax, ay)] == 0 {
            continue;
        }
        for k in 1..=HOOK_TILES {
            let fx = x - dx * k;
            let fy = y - dy * k;
            if fx < 0 || fy < 0 || fx >= g.width || fy >= g.height {
                break;
            }
            if js::hypot2(f64::from(ax - fx), f64::from(ay - fy)) * f64::from(TILE_PX) > HOOK_LENGTH_PX {
                break;
            }
            if g.free[g.idx(fx, fy)] == 0 {
                break;
            }
            if !g.clear_line(fx, fy, ax, ay, false) {
                break;
            }
            if !g.clear_line(fx, fy, x, y, true) {
                break;
            }
            visit(fx, fy, COST_HOOK + k, 3, ay * g.width + ax);
        }
    }
}

/// `findRoute`'s options (`opts`), with the TS defaults.
#[derive(Debug, Clone)]
pub struct RouteOpts<'a> {
    /// Arrive within this many tiles (Chebyshev) of the goal; default 2.
    pub near_tiles: i32,
    pub max_cost: i32,
    /// Accept the node nearest the goal when it is at least 34% closer than the start.
    pub partial: bool,
    /// Allow `kill` steps (a respawn at a spawn tile).
    pub allow_kill: bool,
    pub max_nodes: usize,
    pub through_freeze: bool,
    /// Moves (`route_move_key`) the search must not use.
    pub avoid: Option<&'a HashSet<i32>>,
}

impl Default for RouteOpts<'_> {
    fn default() -> Self {
        RouteOpts {
            near_tiles: 2,
            max_cost: 4000,
            partial: false,
            allow_kill: false,
            max_nodes: 200_000,
            through_freeze: true,
            avoid: None,
        }
    }
}

/// The per-map route search state: the grids, the spawn tiles and the reusable scratch arrays
/// (`scratchOf`).
#[derive(Debug, Clone)]
pub struct Router {
    pub grid: NavGrid,
    spawns: Vec<i32>,
    dist: Vec<i32>,
    from: Vec<i32>,
    kind: Vec<u8>,
    anchor: Vec<i32>,
    stamp: Vec<i32>,
    heap: Heap,
    gen_: i32,
}

impl Router {
    /// `spawn_tile_px`: the spawn tiles' centres in pixels (`spawnTiles(collision)`), used only for
    /// `allow_kill` routes.
    pub fn new(col: &impl PlanCollision, spawn_tile_px: &[(f64, f64)]) -> Router {
        let grid = NavGrid::new(col);
        let n = (grid.width * grid.height) as usize;
        let spawns = spawn_tile_px
            .iter()
            .map(|&(x, y)| {
                (y / f64::from(TILE_PX)).trunc() as i32 * grid.width + (x / f64::from(TILE_PX)).trunc() as i32
            })
            .collect();
        Router {
            grid,
            spawns,
            dist: vec![0; n],
            from: vec![0; n],
            kind: vec![0; n],
            anchor: vec![0; n],
            stamp: vec![0; n],
            heap: Heap::new(n),
            gen_: 0,
        }
    }

    /// Pushes the heap dropped for being full (a TS quirk, see [`Heap`]); 0 in practice.
    pub fn heap_dropped(&self) -> u64 {
        self.heap.dropped
    }

    /// Task 4.12 (`--selfkill-policy smart`): [`Router::find_route`], but a respawn (`kill`) step is the last resort. With `kill_last`
    /// and `opts.allow_kill`, a full route on foot (walk, fall, jump, hook) is looked for first and taken when there is one, however
    /// much dearer than a respawn it is; only without one is the search of `opts` run (which may use a respawn, or settle for a
    /// partial route when `opts.partial`). Without `kill_last` this is exactly [`Router::find_route`].
    pub fn find_route_kill_last(
        &mut self,
        from: (f64, f64),
        to: (f64, f64),
        opts: &RouteOpts<'_>,
        kill_last: bool,
    ) -> Option<RouteResult> {
        if kill_last && opts.allow_kill {
            let foot = RouteOpts {
                allow_kill: false,
                partial: false,
                ..opts.clone()
            };
            if let Some(r) = self.find_route(from, to, &foot) {
                return Some(r);
            }
        }
        self.find_route(from, to, opts)
    }

    /// `findRoute(collision, from, to, opts)` with positions in pixels.
    pub fn find_route(&mut self, from: (f64, f64), to: (f64, f64), opts: &RouteOpts<'_>) -> Option<RouteResult> {
        let avoid = opts.avoid.filter(|a| !a.is_empty());
        let width = self.grid.width;
        let height = self.grid.height;
        let tile = |v: f64, max: i32| -> i32 {
            js::min((max - 1) as f64, js::max(0.0, (v / f64::from(TILE_PX)).trunc())) as i32
        };
        let sx = tile(from.0, width);
        let sy = tile(from.1, height);
        let gx = tile(to.0, width);
        let gy = tile(to.1, height);
        let near = opts.near_tiles;
        let max_cost = opts.max_cost;
        let mut popped = 0usize;
        self.gen_ += 1;
        let gen_ = self.gen_;
        self.heap.clear();
        let start = sy * width + sx;
        self.dist[start as usize] = 0;
        self.from[start as usize] = -1;
        self.stamp[start as usize] = gen_;
        self.heap.push(start, 0);
        let mut best = -1i32;
        let mut closest = start;
        let mut closest_gap = (sx - gx).abs() + (sy - gy).abs();
        let kill_spawns: Option<Vec<i32>> = if opts.allow_kill {
            Some(self.spawns.clone())
        } else {
            None
        };
        while !self.heap.empty() {
            popped += 1;
            if popped > opts.max_nodes {
                break;
            }
            let (index, cost) = self.heap.pop();
            let iu = index as usize;
            if self.stamp[iu] != gen_ || cost > self.dist[iu] {
                continue;
            }
            let x = index % width;
            let y = (index - x) / width;
            if (x - gx).abs() <= near && (y - gy).abs() <= near {
                best = index;
                break;
            }
            let gap = (x - gx).abs() + (y - gy).abs();
            if gap < closest_gap {
                closest_gap = gap;
                closest = index;
            }
            // Borrow juggling: the visitor writes into the scratch arrays while `expand` reads the grid.
            let Router {
                grid,
                dist,
                from,
                kind,
                anchor,
                stamp,
                heap,
                ..
            } = self;
            let mut relax = |nx: i32, ny: i32, step: i32, k: u8, anc: i32| {
                let ni = ny * width + nx;
                let cost0 = cost + step;
                if let Some(a) = avoid
                    && a.contains(&route_move_key(ni, i32::from(k)))
                {
                    return;
                }
                let c = cost0 + if grid.danger[ni as usize] == 1 { COST_DANGER } else { 0 };
                if c > max_cost {
                    return;
                }
                let niu = ni as usize;
                if stamp[niu] == gen_ && c >= dist[niu] {
                    return;
                }
                stamp[niu] = gen_;
                dist[niu] = c;
                from[niu] = index;
                kind[niu] = k;
                anchor[niu] = anc;
                heap.push(ni, c);
            };
            expand(
                grid,
                x,
                y,
                &mut relax,
                kill_spawns.as_deref(),
                opts.through_freeze,
                false,
            );
        }
        if best < 0 && opts.partial {
            let start_gap = (sx - gx).abs() + (sy - gy).abs();
            if closest != start && f64::from(closest_gap) < f64::from(start_gap) * 0.66 {
                best = closest;
            }
        }
        if best < 0 {
            return None;
        }
        let mut steps: Vec<RouteStep> = Vec::new();
        let kinds = [
            MoveKind::Walk,
            MoveKind::Fall,
            MoveKind::Jump,
            MoveKind::Hook,
            MoveKind::Kill,
            MoveKind::Fall,
        ];
        let mut i = best;
        let mut next = -1i32;
        while i >= 0 && i != start {
            let iu = i as usize;
            let x = i % width;
            let y = (i - x) / width;
            let a = self.anchor[iu];
            let k = self.kind[iu];
            let mut step = RouteStep {
                x,
                y,
                kind: kinds[k as usize],
                anchor: if a >= 0 {
                    Some((a % width, (a - (a % width)) / width))
                } else {
                    None
                },
                freeze: false,
                tele: false,
                move_key: route_move_key(i, i32::from(k)),
                leap: false,
            };
            if k == FREEZE_KIND {
                step.freeze = true;
                let p = self.from[iu];
                if p >= 0 && p % width != x {
                    step.leap = true;
                }
            }
            if next >= 0 && self.grid.tele_out[iu] == next {
                step.tele = true;
            }
            steps.push(step);
            next = i;
            i = self.from[iu];
        }
        steps.reverse();
        Some(RouteResult {
            steps,
            cost: self.dist[best as usize],
        })
    }
}

/// `spawnTiles(collision)` (`route.ts:905-913`): the centres (px) of the spawn entity tiles of the
/// game and front layers, in tile-index order.
pub fn spawn_tiles(map: &MapData) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    for i in 0..(map.width as usize * map.height as usize) {
        let in_game = map.game.get(i).is_some_and(|t| t.index == ENTITY_SPAWN);
        let in_front = map
            .front
            .as_ref()
            .is_some_and(|f| f.get(i).is_some_and(|t| t.index == ENTITY_SPAWN));
        if !in_game && !in_front {
            continue;
        }
        let w = map.width as usize;
        out.push((
            (i % w) as f64 * f64::from(TILE_PX) + f64::from(TILE_PX) / 2.0,
            (i / w) as f64 * f64::from(TILE_PX) + f64::from(TILE_PX) / 2.0,
        ));
    }
    out
}

/// `deadZone(collision, spawns)` (`route.ts:926-984`): tiles that can be reached from a spawn but
/// from which no way leads back (loose reverse flood), minus components larger than 250 tiles.
pub fn dead_zone(grid: &NavGrid, spawns: &[(f64, f64)]) -> Vec<u8> {
    let n = (grid.width * grid.height) as usize;
    let mut queue: Vec<i32> = vec![0; n];
    let mut flood = |back: bool| -> Vec<u8> {
        let mut seen = vec![0u8; n];
        let (mut head, mut tail) = (0usize, 0usize);
        for s in spawns {
            let (x, y) = (
                (s.0 / f64::from(TILE_PX)).trunc() as i32,
                (s.1 / f64::from(TILE_PX)).trunc() as i32,
            );
            if x < 0 || y < 0 || x >= grid.width || y >= grid.height {
                continue;
            }
            let i = (y * grid.width + x) as usize;
            if seen[i] == 1 {
                continue;
            }
            seen[i] = 1;
            queue[tail] = i as i32;
            tail += 1;
        }
        while head < tail {
            let i = queue[head];
            head += 1;
            let x = i % grid.width;
            let y = (i - x) / grid.width;
            let mut mark = |nx: i32, ny: i32, _c: i32, _k: u8, _a: i32| {
                if nx < 0 || ny < 0 || nx >= grid.width || ny >= grid.height {
                    return;
                }
                let j = (ny * grid.width + nx) as usize;
                if seen[j] == 1 {
                    return;
                }
                seen[j] = 1;
                queue[tail] = j as i32;
                tail += 1;
            };
            if back {
                expand_back(grid, x, y, &mut mark, true);
            } else {
                expand(grid, x, y, &mut mark, None, true, true);
            }
        }
        seen
    };
    let can_return = flood(true);
    let can_get_there = flood(false);
    let mut dead = vec![0u8; n];
    for i in 0..n {
        dead[i] = u8::from(grid.free[i] == 1 && can_get_there[i] == 1 && can_return[i] == 0);
    }
    let mut seen = vec![0u8; n];
    for start in 0..n {
        if dead[start] == 0 || seen[start] == 1 {
            continue;
        }
        let (mut head, mut tail) = (0usize, 0usize);
        queue[tail] = start as i32;
        tail += 1;
        seen[start] = 1;
        let mut part: Vec<i32> = Vec::new();
        while head < tail {
            let i = queue[head];
            head += 1;
            part.push(i);
            let x = i % grid.width;
            let y = (i - x) / grid.width;
            for (ox, oy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let (nx, ny) = (x + ox, y + oy);
                if nx < 0 || ny < 0 || nx >= grid.width || ny >= grid.height {
                    continue;
                }
                let j = (ny * grid.width + nx) as usize;
                if dead[j] == 0 || seen[j] == 1 {
                    continue;
                }
                seen[j] = 1;
                queue[tail] = j as i32;
                tail += 1;
            }
        }
        if part.len() > MAX_TRAP_TILES {
            for i in part {
                dead[i as usize] = 0;
            }
        }
    }
    dead
}
