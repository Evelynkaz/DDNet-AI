//! Reachability for the target score (`reachable` / `checkWanted` / `checkReach`, `bot.ts:2828-2897`,
//! `docs/research/orig-bot.md` §7.5).
//!
//! A target at `>= PATH_NEAR_PX` that is not at war and not roped to us gets `-700` unless a route
//! to it exists. The route finder itself is **navigation** — the TS `findRoute` is a 984-line
//! walk/fall/jump/hook graph search (`src/plan/route.ts`) and belongs to task 4.2 — so it is a trait
//! here ([`RouteFinder`], one of the [`crate::hooks::Hooks`]), with a built-in default
//! ([`FloodRoute`]): a bounded flood fill through free air tiles that never enters freeze or death
//! tiles (`throughFreeze: false`) and succeeds when it comes within `near_tiles` (3) of the goal.
//! That is an upper bound on what the real search can reach (it ignores jump reach), and exact for
//! what the score needs: a target sealed off behind a wall or a freeze barrier stays unreachable.
//!
//! The answer cache is the TS one: per target id `{tick, from tile, to tile, ok}`, fresh for
//! `REACH_ANSWER_TICKS` (25) while both tiles are unchanged. To keep the bot's own overhead under
//! the D-042 budget at most `checks_per_snapshot` (default 1) fresh searches run per snapshot — the
//! TS "low-cpu" scheme, always on: a stale answer is returned (or `true` for a never-checked
//! target) and the id is queued in `wanted`; the next snapshot's first query processes the queue
//! oldest-answer-first. The TS capped the cache at 64 entries; here there is one slot per client id
//! (128), so nothing is ever evicted.

use crate::consts::*;
use crate::mapgrid::{MapGrid, TILE_PX};
use crate::players::MAX_CLIENTS;
use crate::tees::Tee;

/// `(tile x, tile y)`.
pub type Tile = (i32, i32);

/// The pixel position's tile.
pub fn tile_of(x: f32, y: f32) -> Tile {
    ((x / TILE_PX as f32).floor() as i32, (y / TILE_PX as f32).floor() as i32)
}

/// Is there a route from `from` to within `near_tiles` of `to`, through at most `max_nodes` nodes?
/// (`findRoute(..) !== null` with `throughFreeze: false`, `partial: false`.)
pub trait RouteFinder {
    fn reachable(&mut self, grid: &MapGrid, from: Tile, to: Tile, near_tiles: i32, max_nodes: usize) -> bool;
}

/// The built-in flood-fill finder (see the module docs).
#[derive(Default)]
pub struct FloodRoute {
    stamp: Vec<u32>,
    generation: u32,
    queue: Vec<u32>,
}

impl RouteFinder for FloodRoute {
    fn reachable(&mut self, grid: &MapGrid, from: Tile, to: Tile, near_tiles: i32, max_nodes: usize) -> bool {
        let (w, h) = (grid.width(), grid.height());
        let cells = (w as usize) * (h as usize);
        if self.stamp.len() != cells {
            self.stamp.clear();
            self.stamp.resize(cells, 0);
            self.generation = 0;
        }
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.stamp.fill(0);
            self.generation = 1;
        }
        let near = |t: Tile| (t.0 - to.0).abs() <= near_tiles && (t.1 - to.1).abs() <= near_tiles;
        let inside = |t: Tile| t.0 >= 0 && t.1 >= 0 && t.0 < w && t.1 < h;
        if !inside(from) {
            return false;
        }
        if near(from) {
            return true;
        }
        self.queue.clear();
        let start = (from.1 * w + from.0) as u32;
        self.stamp[start as usize] = self.generation;
        self.queue.push(start);
        let mut head = 0;
        let mut nodes = 0usize;
        while head < self.queue.len() {
            let cur = self.queue[head];
            head += 1;
            nodes += 1;
            if nodes > max_nodes {
                return false;
            }
            let (cx, cy) = ((cur as i32) % w, (cur as i32) / w);
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let n = (cx + dx, cy + dy);
                if !inside(n) {
                    continue;
                }
                let ni = (n.1 * w + n.0) as usize;
                if self.stamp[ni] == self.generation {
                    continue;
                }
                // Free air only; the goal's neighbourhood counts even when the goal tile itself is
                // solid/freeze (a frozen target lies *in* freeze).
                if grid.tile_solid(n.0, n.1) || grid.tile_hazard(n.0, n.1) {
                    continue;
                }
                if near(n) {
                    return true;
                }
                self.stamp[ni] = self.generation;
                self.queue.push(ni as u32);
            }
        }
        false
    }
}

#[derive(Debug, Clone, Copy)]
struct Answer {
    valid: bool,
    tick: i32,
    from: Tile,
    to: Tile,
    ok: bool,
}

const NO_ANSWER: Answer = Answer {
    valid: false,
    tick: 0,
    from: (0, 0),
    to: (0, 0),
    ok: true,
};

/// The answer cache with the per-snapshot search budget (see the module docs).
pub struct ReachCache {
    answers: Box<[Answer; MAX_CLIENTS]>,
    wanted: Box<[bool; MAX_CLIENTS]>,
    want_tick: i32,
    left: u32,
    checks_per_snapshot: u32,
    /// Fresh searches run since creation (telemetry / tests).
    searches: u64,
}

impl ReachCache {
    pub fn new(checks_per_snapshot: u32) -> Self {
        ReachCache {
            answers: Box::new([NO_ANSWER; MAX_CLIENTS]),
            wanted: Box::new([false; MAX_CLIENTS]),
            want_tick: i32::MIN,
            left: checks_per_snapshot,
            checks_per_snapshot,
            searches: 0,
        }
    }

    pub fn searches(&self) -> u64 {
        self.searches
    }

    /// Forgets answers and the queue (new map / tick reset).
    pub fn clear(&mut self) {
        self.answers.fill(NO_ANSWER);
        self.wanted.fill(false);
        self.want_tick = i32::MIN;
    }

    /// Call once per snapshot before any [`ReachCache::reachable`]: refills the search budget
    /// (`reachLeft = LOW_CPU.reachChecks`, `bot.ts:2356`).
    pub fn begin_snapshot(&mut self) {
        self.left = self.checks_per_snapshot;
    }

    /// `reachable(from, tee)` (`bot.ts:2828-2845`).
    #[allow(clippy::too_many_arguments)]
    pub fn reachable(
        &mut self,
        tick: i32,
        me: &Tee,
        tee: &Tee,
        tees: &crate::tees::TeeSet,
        grid: &MapGrid,
        finder: &mut dyn RouteFinder,
    ) -> bool {
        let a = tile_of(me.pos.x, me.pos.y);
        if tick != self.want_tick {
            self.want_tick = tick;
            self.check_wanted(tick, a, tees, grid, finder);
        }
        let b = tile_of(tee.pos.x, tee.pos.y);
        let Some(i) = usize::try_from(tee.id).ok().filter(|&i| i < MAX_CLIENTS) else {
            return true;
        };
        let seen = self.answers[i];
        if seen.valid && seen.from == a && seen.to == b && tick - seen.tick < REACH_ANSWER_TICKS && tick >= seen.tick {
            return seen.ok;
        }
        // Budget exhausted for this snapshot: queue it, answer from the old answer or optimistically.
        if self.left == 0 {
            self.wanted[i] = true;
            return if seen.valid { seen.ok } else { true };
        }
        self.left -= 1;
        self.check(tick, a, i, b, grid, finder)
    }

    /// `checkWanted` (`bot.ts:2847-2875`): refresh queued ids, oldest answer first (ties by id).
    fn check_wanted(
        &mut self,
        tick: i32,
        a: Tile,
        tees: &crate::tees::TeeSet,
        grid: &MapGrid,
        finder: &mut dyn RouteFinder,
    ) {
        // Collect in a fixed array sorted by (age, id): at most 128 entries, no allocation.
        let mut order = [(0i64, 0usize); MAX_CLIENTS];
        let mut n = 0;
        for i in 0..MAX_CLIENTS {
            if !self.wanted[i] {
                continue;
            }
            self.wanted[i] = false;
            let age = if self.answers[i].valid && self.answers[i].tick <= tick {
                i64::from(self.answers[i].tick)
            } else {
                i64::MIN
            };
            order[n] = (age, i);
            n += 1;
        }
        order[..n].sort_unstable();
        for &(_, i) in &order[..n] {
            if self.left == 0 {
                break;
            }
            let Some(tee) = tees.get(i as i32) else { continue };
            let b = tile_of(tee.pos.x, tee.pos.y);
            let seen = self.answers[i];
            if seen.valid
                && seen.from == a
                && seen.to == b
                && tick - seen.tick < REACH_ANSWER_TICKS
                && tick >= seen.tick
            {
                continue;
            }
            self.left -= 1;
            self.check(tick, a, i, b, grid, finder);
        }
    }

    fn check(&mut self, tick: i32, a: Tile, i: usize, b: Tile, grid: &MapGrid, finder: &mut dyn RouteFinder) -> bool {
        self.searches += 1;
        let ok = finder.reachable(grid, a, b, 3, REACH_MAX_NODES);
        self.answers[i] = Answer {
            valid: true,
            tick,
            from: a,
            to: b,
            ok,
        };
        ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapgrid::test_maps::*;
    use crate::tees::{Tee, TeeSet};
    use ddai_physics::vmath::Vec2;

    fn tee_at(id: i32, tx: i32, ty: i32) -> Tee {
        Tee {
            id,
            alive: true,
            pos: Vec2::new(tx as f32 * 32.0 + 16.0, ty as f32 * 32.0 + 16.0),
            ..Tee::DEAD
        }
    }

    /// A 30x12 room split by a wall at x=15 (full height) — two sealed halves.
    fn split_room() -> MapGrid {
        let wall: Vec<(u32, u32, u8)> = (0..12).map(|y| (15, y, SOLID)).collect();
        MapGrid::new(&room(30, 12, &wall))
    }

    #[test]
    fn flood_fill_reaches_through_open_air_and_not_through_walls() {
        let open = MapGrid::new(&room(30, 12, &[]));
        let mut f = FloodRoute::default();
        assert!(f.reachable(&open, (2, 5), (27, 5), 3, 20_000));
        let split = split_room();
        assert!(
            !f.reachable(&split, (2, 5), (27, 5), 3, 20_000),
            "a wall seals the halves"
        );
        assert!(f.reachable(&split, (2, 5), (13, 5), 3, 20_000), "same half");
        assert!(
            f.reachable(&split, (2, 5), (17, 5), 3, 20_000),
            "within near_tiles=3 of the wall's far side"
        );
        assert!(
            !f.reachable(&split, (2, 5), (19, 5), 3, 20_000),
            "4+ tiles beyond the wall"
        );
    }

    #[test]
    fn freeze_blocks_the_flood_but_a_goal_standing_in_freeze_is_reached_from_beside_it() {
        // A freeze column at x=15 (full height) separates the halves.
        let fz: Vec<(u32, u32, u8)> = (1..11).map(|y| (15, y, FREEZE)).collect();
        let g = MapGrid::new(&room(30, 12, &fz));
        let mut f = FloodRoute::default();
        assert!(!f.reachable(&g, (2, 5), (27, 5), 3, 20_000), "throughFreeze: false");
        assert!(
            f.reachable(&g, (2, 5), (15, 5), 3, 20_000),
            "a tee lying in the freeze is reachable from beside it"
        );
    }

    #[test]
    fn the_node_cap_means_unreachable() {
        let open = MapGrid::new(&room(60, 40, &[]));
        let mut f = FloodRoute::default();
        assert!(
            !f.reachable(&open, (2, 2), (57, 37), 3, 50),
            "50 nodes cannot cross the map"
        );
        assert!(f.reachable(&open, (2, 2), (57, 37), 3, 20_000));
        assert!(
            f.reachable(&open, (2, 2), (4, 3), 3, 50),
            "the goal's neighbourhood is found immediately"
        );
    }

    #[test]
    fn a_start_outside_the_map_is_unreachable() {
        let g = MapGrid::new(&room(10, 10, &[]));
        assert!(!FloodRoute::default().reachable(&g, (-1, 5), (5, 5), 0, 100));
    }

    struct Counting(u32, bool);
    impl RouteFinder for Counting {
        fn reachable(&mut self, _g: &MapGrid, _f: Tile, _t: Tile, _n: i32, _m: usize) -> bool {
            self.0 += 1;
            self.1
        }
    }

    fn set(tees: &[Tee]) -> TeeSet {
        let mut s = TeeSet::new();
        for t in tees {
            s.set_for_test(*t);
        }
        s
    }

    #[test]
    fn answers_are_cached_for_25_ticks_while_the_tiles_stay_and_refreshed_after() {
        let grid = MapGrid::new(&room(30, 12, &[]));
        let (me, a) = (tee_at(0, 2, 5), tee_at(1, 20, 5));
        let tees = set(&[me, a]);
        let mut cache = ReachCache::new(1);
        let mut f = Counting(0, false);
        cache.begin_snapshot();
        assert!(!cache.reachable(100, &me, &a, &tees, &grid, &mut f));
        assert_eq!(f.0, 1);
        for tick in [102, 110, 124] {
            cache.begin_snapshot();
            assert!(!cache.reachable(tick, &me, &a, &tees, &grid, &mut f));
        }
        assert_eq!(f.0, 1, "fresh for 25 ticks");
        f.1 = true;
        cache.begin_snapshot();
        assert!(
            cache.reachable(125, &me, &a, &tees, &grid, &mut f),
            "25 ticks later: re-asked"
        );
        assert_eq!(f.0, 2);
        // A moved target invalidates immediately.
        let a2 = tee_at(1, 22, 5);
        let tees2 = set(&[me, a2]);
        cache.begin_snapshot();
        f.1 = false;
        assert!(!cache.reachable(126, &me, &a2, &tees2, &grid, &mut f));
        assert_eq!(f.0, 3);
    }

    #[test]
    fn the_budget_limits_fresh_searches_per_snapshot_and_the_queue_catches_up() {
        let grid = MapGrid::new(&room(40, 12, &[]));
        let me = tee_at(0, 2, 5);
        let others: Vec<Tee> = (1..=4).map(|i| tee_at(i, 10 + 3 * i, 5)).collect();
        let mut all = vec![me];
        all.extend(&others);
        let tees = set(&all);
        let mut cache = ReachCache::new(1);
        let mut f = Counting(0, false);
        cache.begin_snapshot();
        let first: Vec<bool> = others
            .iter()
            .map(|t| cache.reachable(200, &me, t, &tees, &grid, &mut f))
            .collect();
        assert_eq!(f.0, 1, "one fresh search in the first snapshot");
        assert_eq!(first, vec![false, true, true, true], "the rest answered optimistically");
        // Next snapshots work through the queue, one per snapshot, and the answers turn false.
        for k in 1..=3 {
            cache.begin_snapshot();
            for t in &others {
                cache.reachable(200 + 2 * k, &me, t, &tees, &grid, &mut f);
            }
        }
        assert_eq!(f.0, 4, "each of the four targets searched once");
        cache.begin_snapshot();
        assert!(
            others
                .iter()
                .all(|t| !cache.reachable(208, &me, t, &tees, &grid, &mut f))
        );
        assert_eq!(f.0, 4, "all cached now");
    }

    #[test]
    fn clear_forgets_answers() {
        let grid = MapGrid::new(&room(30, 12, &[]));
        let (me, a) = (tee_at(0, 2, 5), tee_at(1, 20, 5));
        let tees = set(&[me, a]);
        let mut cache = ReachCache::new(1);
        let mut f = Counting(0, true);
        cache.begin_snapshot();
        cache.reachable(10, &me, &a, &tees, &grid, &mut f);
        cache.clear();
        cache.begin_snapshot();
        cache.reachable(12, &me, &a, &tees, &grid, &mut f);
        assert_eq!(f.0, 2);
    }
}
