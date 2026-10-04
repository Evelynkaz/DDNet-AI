//! Target selection — the port of `pickTarget` (`bot.ts:3020-3123`, `docs/research/orig-bot.md`
//! §7.1-§7.2) without the wayblock, duel and partner branches (those are hooks / dropped, D-021).
//!
//! **Fixed target** (`--target <name>`): the player whose folded name equals the wanted one, if it is
//! not us and alive; no other filter (friend, AFK, frozen all ignored — `bot.ts:3021-3027`).
//!
//! **Auto**: every alive tee but us is filtered, then scored; the best strictly-greater score wins.
//!
//! Filters (in TS order): on the friend/ignore/clan-friend list (never); out of game
//! (paused/spectating); not at war and AFK ([`ActivityClock::afk`]); farther than 1600 px; the
//! wayblock hook's verdict; *settled* — `sealed` (frozen, or the current target near freeze, and
//! [`PlanScratch::sealed`]: it cannot get out whatever it tries) or frozen for more than
//! `SETTLED_FREEZE_TICKS` (0) without being a "finishing" current target. A settled current target is
//! remembered (`keepSettled`) and returned when nobody else qualifies.
//!
//! Score (all terms of the §7.1 table that survive the drops): at war `+900`; it hooks us `+1000`; we
//! hook it `+800`; `blockHoldScore` (0) for the frozen current target within 320 px; finishing within
//! 320 px `+600`; aggressor (attacked within 150 ticks and within 500 px) `+500`; attacked a friend
//! within 150 ticks `+450`; approaching (`d < last_seen_dist - 1`) `+200`; the current target gets
//! the hold bonus `400` (full up to 420 px, fading to 0 over the next 400); `-0.25 * d`; out of reach
//! (`d >= 420`, not at war, no rope either way, not [`ReachCache::reachable`]) `-700`; wayblock zone
//! `+300`.
//!
//! **Tie rule.** The TS broke ties by `Map` insertion order (first seen). Here candidates are visited
//! in **ascending client id** and only a strictly greater score replaces the best, so a tie goes to
//! the lowest id — deterministic and independent of join order.

use ddai_physics::world::World;

use crate::activity::ActivityClock;
use crate::bot::Mode;
use crate::consts::*;
use crate::hooks::{HookContext, Hooks};
use crate::mapgrid::MapGrid;
use crate::names::fold_name;
use crate::planning::PlanScratch;
use crate::players::{MAX_CLIENTS, PlayerTable};
use crate::reach::{ReachCache, Tile};
use crate::relations::RelationFlags;
use crate::seal_worker::SealWorker;
use crate::tees::{Tee, TeeSet, dist};

/// What the target selection reads.
pub struct PickCtx<'a> {
    pub tick: i32,
    pub own: &'a Tee,
    pub tees: &'a TeeSet,
    pub players: &'a PlayerTable,
    pub clock: &'a ActivityClock,
    pub grid: &'a MapGrid,
    /// The base world (snapshot tick) the seal check copies.
    pub base: &'a World<f32>,
    pub lag_ticks: i32,
    pub mode: Mode,
}

#[derive(Debug, Clone, Copy)]
struct SealAnswer {
    valid: bool,
    tick: i32,
    sealed: bool,
    tile: (i32, i32),
}

/// One fight score, kept for the telemetry of the chosen target.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Pick {
    pub id: i32,
    pub score: f32,
}

pub struct TargetPicker {
    /// The current target (`this.targetId`, -1 none).
    target: i32,
    /// `lastSeenDist`: the distance at which each id last passed the filters (NaN = never).
    last_seen: Box<[f32; MAX_CLIENTS]>,
    seal: Box<[SealAnswer; MAX_CLIENTS]>,
    /// Fresh seal searches left in this snapshot (`SEAL_CHECKS_PER_SNAPSHOT`).
    seal_left: u32,
    /// `Some` in the live runner: seal searches run on this worker thread (see `seal_worker`).
    worker: Option<SealWorker>,
    /// Bumped on reset: answers of searches started before it are discarded.
    generation: u64,
    /// Async mode: the (tick, tile) a request for each id was started at, while in flight.
    in_flight: Box<[Option<(i32, Tile)>; MAX_CLIENTS]>,
    /// Wall time of each fresh seal search / reachability flood (they are the dearest parts of
    /// target selection; reported next to the latency summaries).
    pub seal_times: crate::latency::Series,
    pub reach_times: crate::latency::Series,
    reach: ReachCache,
    fixed_name: Option<String>,
    last_pick: Pick,
}

impl TargetPicker {
    pub fn new(fixed_target: Option<&str>) -> Self {
        TargetPicker {
            target: -1,
            last_seen: Box::new([f32::NAN; MAX_CLIENTS]),
            seal: Box::new(
                [SealAnswer {
                    valid: false,
                    tick: 0,
                    sealed: false,
                    tile: (0, 0),
                }; MAX_CLIENTS],
            ),
            seal_left: SEAL_CHECKS_PER_SNAPSHOT,
            worker: None,
            generation: 0,
            in_flight: Box::new([None; MAX_CLIENTS]),
            seal_times: Default::default(),
            reach_times: Default::default(),
            reach: ReachCache::new(REACH_CHECKS_PER_SNAPSHOT),
            fixed_name: fixed_target.map(fold_name).filter(|n| !n.is_empty()),
            last_pick: Pick { id: -1, score: 0.0 },
        }
    }

    /// `!target <nick>` / `!target -`: fight only this player (folded), or pick automatically again.
    pub fn set_fixed(&mut self, name: Option<&str>) {
        self.fixed_name = name.map(fold_name).filter(|n| !n.is_empty());
        self.target = -1;
    }

    /// The fixed target's folded name, if one is set.
    pub fn fixed(&self) -> Option<&str> {
        self.fixed_name.as_deref()
    }

    /// The current target id, -1 when none.
    pub fn target(&self) -> i32 {
        self.target
    }

    /// The score of the last auto pick (0 for a fixed target or none).
    pub fn last_score(&self) -> f32 {
        self.last_pick.score
    }

    /// Runs seal searches on `worker` from now on (`None`: back to the synchronous search).
    pub fn set_seal_worker(&mut self, worker: Option<SealWorker>) {
        self.worker = worker;
        self.generation += 1;
        self.in_flight.fill(None);
    }

    pub fn set_target(&mut self, id: i32) {
        self.target = id;
    }

    pub fn reach_searches(&self) -> u64 {
        self.reach.searches()
    }

    /// Forget everything map- or tick-specific (`map_change`: `targetId = -1`, `lastSeenDist.clear()`).
    pub fn reset(&mut self) {
        self.target = -1;
        self.last_seen.fill(f32::NAN);
        for s in self.seal.iter_mut() {
            s.valid = false;
        }
        self.reach.clear();
        self.generation += 1;
        self.in_flight.fill(None);
        self.last_pick = Pick { id: -1, score: 0.0 };
    }

    /// `pickTarget`: the id to fight, or -1. Does **not** store it (the caller decides whether a mode
    /// uses a target at all and then calls [`TargetPicker::set_target`]).
    pub fn pick(&mut self, ctx: &PickCtx<'_>, hooks: &mut Hooks, plan: &mut PlanScratch) -> i32 {
        if let Some(want) = self.fixed_name.as_deref() {
            return self.pick_fixed(ctx, want);
        }
        let hook_ctx = HookContext {
            tick: ctx.tick,
            own: ctx.own,
            tees: ctx.tees,
            players: ctx.players,
            grid: ctx.grid,
            clock: ctx.clock,
            world: ctx.base,
            lag_ticks: ctx.lag_ticks,
            mode: ctx.mode,
            fixed_target: false,
        };
        self.reach.begin_snapshot();
        self.seal_left = SEAL_CHECKS_PER_SNAPSHOT;
        self.collect_seal_results();
        if hooks.wayblock.holding() {
            let (base, tick, target) = (ctx.base, ctx.tick, self.target);
            hooks
                .wayblock
                .begin_pick(&hook_ctx, target, &mut |t: &Tee| self.is_sealed(tick, t, base, plan));
        }
        let (tick, me) = (ctx.tick, ctx.own);
        let mut best = -1;
        let mut best_score = f32::NEG_INFINITY;
        let mut keep_settled = false;
        let default_flags = RelationFlags::default();
        for tee in ctx.tees.iter() {
            if tee.id == me.id {
                continue;
            }
            let slot = ctx.players.get(tee.id);
            let flags = slot.map_or(default_flags, |s| s.flags);
            if flags.never_target() {
                continue;
            }
            let at_war = flags.at_war();
            // A spectator or a paused player whose tee is still on the map is fought like anybody else
            // (`awayInGame`), except in the AFK room (`wbWalkAllowed`); only a tee that is AFK while it
            // plays is skipped.
            if !at_war && ctx.clock.away_in_game(tee.id, tick, ctx.players) {
                continue;
            }
            let not_playing = slot.is_some_and(|s| s.not_playing());
            if !at_war && not_playing {
                let (tx, ty) = crate::reach::tile_of(tee.pos.x, tee.pos.y);
                if !hooks.wayblock.walk_allowed(tx, ty) {
                    continue;
                }
            }
            let d = dist(me.pos, tee.pos);
            if d > TARGET_MAX_PX {
                continue;
            }
            let wb = if hooks.wayblock.holding() {
                hooks.wayblock.filter(&hook_ctx, tee)
            } else {
                Default::default()
            };
            if wb.skip {
                continue;
            }

            let frozen_for = ctx.clock.frozen_for(tee, tick);
            let is_current = tee.id == self.target;
            let near_freeze = ctx.grid.near_freeze(tee.pos.x, tee.pos.y, SEAL_NEAR_TILES);
            // `settled = sealed || (!finishing && frozenFor > S)` with `finishing = cand && !sealed`
            // (`bot.ts:3072-3088`). When the tee has been frozen a while and is not a "finishing"
            // candidate, `settled` is true whatever `sealed` says, so the ~0.5 ms `sealedIn` search is
            // skipped: the result is identical and the search only runs where it can matter (the
            // first frozen tick, a finishing target, a free current target near freeze).
            // `wbFinish` (`bot.ts:3080`): a frozen tee in the WB zone, seen from inside the hall, is a
            // finishing target (unless sealed) however long it has been frozen.
            let wb_finish_candidate = wb.finish_zone && tee.frozen;
            let finishing_candidate =
                wb_finish_candidate || (is_current && tee.frozen && frozen_for <= FINISH_BLOCK_TICKS && near_freeze);
            let settled_anyway = frozen_for > SETTLED_FREEZE_TICKS && !finishing_candidate;
            let sealed = !wb.corridor
                && (tee.frozen || (is_current && near_freeze))
                && !settled_anyway
                && self.is_sealed(tick, tee, ctx.base, plan);
            let finishing = finishing_candidate && !sealed;
            let settled = sealed || (!finishing && frozen_for > SETTLED_FREEZE_TICKS);
            if settled {
                if is_current {
                    keep_settled = true;
                }
                continue;
            }

            let roped = tee.hooked_player == me.id || me.hooked_player == tee.id;
            let out_of_reach = d >= PATH_NEAR_PX
                && !at_war
                && !roped
                && !wb.corridor
                && !{
                    let before = self.reach.searches();
                    let t0 = std::time::Instant::now();
                    let ok = self
                        .reach
                        .reachable(tick, me, tee, ctx.tees, ctx.grid, hooks.route.as_mut());
                    if self.reach.searches() != before {
                        self.reach_times.push(t0.elapsed());
                    }
                    ok
                };

            if out_of_reach && not_playing && ctx.clock.input_idle(tee.id, tick, false) {
                continue;
            }
            let mut score = 0.0f32;
            if at_war {
                score += 900.0;
            }
            if tee.hooked_player == me.id {
                score += 1000.0;
            }
            if me.hooked_player == tee.id {
                score += 800.0;
            }
            if is_current && tee.frozen && d < BLOCKING_RANGE_PX {
                score += BLOCK_HOLD_SCORE;
            }
            if finishing && d < BLOCKING_RANGE_PX {
                score += FINISH_BLOCK_SCORE;
            }
            if tick - tee.attack_tick < AGGRESSOR_MEMORY_TICKS && d < AGGRESSOR_RANGE_PX {
                score += 500.0;
            }
            if ctx.clock.at_friend_within(tee.id, tick, AGGRESSOR_MEMORY_TICKS) {
                score += AT_FRIEND_SCORE;
            }
            let prev = self.last_seen[tee.id as usize];
            if !prev.is_nan() && d < prev - 1.0 {
                score += 200.0;
            }
            if is_current {
                let hold = TARGET_HOLD_SCORE;
                score += if d <= ENGAGED_PX {
                    hold
                } else {
                    hold * (1.0 - (d - ENGAGED_PX) / HOLD_FADE_PX).max(0.0)
                };
            }
            score -= d * TARGET_DIST_WEIGHT;
            if out_of_reach {
                score -= OUT_OF_REACH_SCORE;
            }
            if wb.in_zone {
                score += 300.0;
            }
            if wb.corridor {
                score += crate::wb_guard::WB_CORRIDOR_SCORE;
            }
            self.last_seen[tee.id as usize] = d;
            if score > best_score {
                best_score = score;
                best = tee.id;
            }
        }
        if best == -1 && keep_settled {
            self.last_pick = Pick {
                id: self.target,
                score: 0.0,
            };
            return self.target;
        }
        self.last_pick = Pick {
            id: best,
            score: if best >= 0 { best_score } else { 0.0 },
        };
        best
    }

    /// The fixed-target branch (`bot.ts:3021-3027`).
    fn pick_fixed(&self, ctx: &PickCtx<'_>, want: &str) -> i32 {
        for (id, slot) in ctx.players.present() {
            if slot.name_key == want {
                return if id == ctx.own.id || ctx.tees.get(id).is_none() {
                    -1
                } else {
                    id
                };
            }
        }
        -1
    }

    /// Takes the finished searches of the worker into the answer cache.
    fn collect_seal_results(&mut self) {
        let Some(w) = self.worker.as_ref() else { return };
        while let Some(r) = w.try_recv() {
            let Some(slot) = usize::try_from(r.id).ok().filter(|&i| i < MAX_CLIENTS) else {
                continue;
            };
            if r.generation != self.generation {
                continue; // a search from before a reset
            }
            let Some((tick, tile)) = self.in_flight[slot].take() else {
                continue;
            };
            self.seal_times.push(r.took);
            self.seal[slot] = SealAnswer {
                valid: true,
                tick: tick.max(r.tick),
                sealed: r.sealed,
                tile,
            };
        }
    }

    /// `isSealed` with its answer cache (`bot.ts:3030-3050`: 6 ticks). Two cost controls the TS did not
    /// have (`sealedIn` is up to 4 x 90 physics steps, ~1.5 ms, the dearest thing the bot does
    /// outside the brain): a "sealed" answer is kept up to `SEALED_TRUE_TICKS` while the tee is still
    /// frozen on the same tile, and at most `SEAL_CHECKS_PER_SNAPSHOT` fresh searches run per
    /// snapshot — past that, a stale answer is used, or "not sealed" for a tee never checked (it is
    /// then checked on a later snapshot; the cost is at worst a few ticks of targeting a tee that is
    /// in fact stuck).
    fn is_sealed(&mut self, tick: i32, tee: &Tee, base: &World<f32>, plan: &mut PlanScratch) -> bool {
        let Some(slot) = usize::try_from(tee.id).ok().filter(|&i| i < MAX_CLIENTS) else {
            return false;
        };
        let tile = crate::reach::tile_of(tee.pos.x, tee.pos.y);
        let seen = self.seal[slot];
        if seen.valid && tick >= seen.tick {
            let age = tick - seen.tick;
            let fresh =
                age < SEAL_ANSWER_TICKS || (seen.sealed && tee.frozen && seen.tile == tile && age < SEALED_TRUE_TICKS);
            if fresh {
                return seen.sealed;
            }
        }
        if self.worker.is_some() {
            // Asynchronous: ask (once per tee) and use the old answer until the new one arrives.
            if self.in_flight[slot].is_none()
                && let Some(w) = self.worker.as_ref()
                && w.request(tee.id, tick, self.generation, base)
            {
                self.in_flight[slot] = Some((tick, tile));
            }
            return seen.valid && seen.sealed && seen.tile == tile;
        }
        if self.seal_left == 0 {
            return seen.valid && seen.sealed && seen.tile == tile;
        }
        self.seal_left -= 1;
        let t0 = std::time::Instant::now();
        let sealed = plan.sealed(base, tee.id);
        self.seal_times.push(t0.elapsed());
        self.seal[slot] = SealAnswer {
            valid: true,
            tick,
            sealed,
            tile,
        };
        sealed
    }
}

/// `spared(t)` (`bot.ts:3009-3018`): tees the hook must never catch and the planner steers around —
/// ignored; an unfrozen friend / clan-friend; or (not at war) away in the game (AFK while playing).
pub fn is_spared(tee: &Tee, tick: i32, players: &PlayerTable, clock: &ActivityClock) -> bool {
    let slot = players.get(tee.id);
    let flags = slot.map_or(RelationFlags::default(), |s| s.flags);
    if flags.ignore {
        return true;
    }
    if !tee.frozen && flags.friendly() {
        return true;
    }
    !flags.at_war() && clock.away_in_game(tee.id, tick, players)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hooks::{WayBlock, WbFilter};
    use crate::mapgrid::test_maps::{FREEZE, SOLID, room};
    use crate::players::test_support::player;
    use crate::relations::{ListKind, Relations};
    use ddai_physics::map::MapData;
    use ddai_physics::vmath::Vec2;
    use std::sync::Arc;

    /// A picker fixture over a synthetic room: tees, players, the clock, an (empty) base world.
    struct Fx {
        picker: TargetPicker,
        tees: TeeSet,
        players: PlayerTable,
        clock: ActivityClock,
        grid: MapGrid,
        world: World<f32>,
        plan: PlanScratch,
        hooks: Hooks,
        rel: Relations,
        tick: i32,
    }

    fn tee(id: i32, x: f32) -> Tee {
        Tee {
            id,
            alive: true,
            pos: Vec2::new(x, 500.0),
            attack_tick: -10_000,
            ..Tee::DEAD
        }
    }

    impl Fx {
        fn new(map: MapData, fixed: Option<&str>) -> Fx {
            let map = Arc::new(map);
            let mut world = World::<f32>::from_map(&map, 1);
            let _ = world.init(std::iter::empty::<&str>());
            Fx {
                picker: TargetPicker::new(fixed),
                tees: TeeSet::new(),
                players: PlayerTable::new([0; 16]),
                clock: ActivityClock::new(),
                grid: MapGrid::new(&map),
                world,
                plan: PlanScratch::new(Arc::clone(&map)),
                hooks: Hooks::default(),
                rel: Relations::new(),
                tick: 1000,
            }
        }

        fn open() -> Fx {
            Fx::new(room(120, 40, &[]), None)
        }

        fn set_players(&mut self, names: &[(i32, &str)]) {
            let list: Vec<_> = names
                .iter()
                .map(|&(id, n)| player(id, n, "", id == 0, 0, Some(0)))
                .collect();
            self.players.update(&list, &self.rel);
        }

        fn put(&mut self, t: Tee) {
            self.tees.set_for_test(t);
        }

        /// Updates the clock with the current tees (so activity/attribution state exists) and picks.
        fn pick(&mut self) -> i32 {
            self.clock.update(self.tick, &self.tees, &self.players, 0);
            let own = *self.tees.get(0).unwrap();
            let t = self.picker.pick(
                &PickCtx {
                    tick: self.tick,
                    own: &own,
                    tees: &self.tees,
                    players: &self.players,
                    clock: &self.clock,
                    grid: &self.grid,
                    base: &self.world,
                    lag_ticks: 0,
                    mode: Mode::Fight,
                },
                &mut self.hooks,
                &mut self.plan,
            );
            self.picker.set_target(t);
            t
        }

        fn score(&mut self) -> f32 {
            self.picker.last_score()
        }
    }

    const D200: f32 = 200.0;

    /// One candidate 200 px away, everything else default: the score is exactly `-0.25 * d`.
    fn single() -> Fx {
        let mut f = Fx::open();
        f.set_players(&[(0, "me"), (1, "foe")]);
        f.put(tee(0, 1000.0));
        f.put(tee(1, 1000.0 + D200));
        f
    }

    #[test]
    fn the_base_score_is_minus_a_quarter_per_pixel() {
        let mut f = single();
        assert_eq!(f.pick(), 1);
        assert_eq!(f.score(), -50.0);
    }

    #[test]
    fn war_adds_900_and_ignores_afk() {
        let mut f = single();
        f.rel.add(ListKind::War, "foe");
        f.set_players(&[(0, "me"), (1, "foe")]);
        // 600 ticks of nothing: AFK, but at war.
        for t in (0..=600).step_by(2) {
            f.tick = t;
            f.clock.update(t, &f.tees, &f.players, 0);
        }
        f.tick = 602;
        assert_eq!(f.pick(), 1);
        assert_eq!(f.score(), -50.0 + 900.0);
        let mut g = single();
        for t in (0..=600).step_by(2) {
            g.tick = t;
            g.clock.update(t, &g.tees, &g.players, 0);
        }
        g.tick = 602;
        assert_eq!(g.pick(), -1, "not at war: AFK is out");
    }

    #[test]
    fn a_hook_between_us_adds_1000_when_it_hooks_us_and_800_when_we_hook_it() {
        let mut f = single();
        let mut t = *f.tees.get(1).unwrap();
        t.hooked_player = 0;
        f.put(t);
        assert_eq!(f.pick(), 1);
        assert_eq!(f.score(), -50.0 + 1000.0);
        let mut g = single();
        let mut me = *g.tees.get(0).unwrap();
        me.hooked_player = 1;
        g.put(me);
        assert_eq!(g.pick(), 1);
        assert_eq!(g.score(), -50.0 + 800.0);
    }

    #[test]
    fn the_aggressor_bonus_needs_a_fresh_attack_and_less_than_500_px() {
        let mut f = single();
        let mut t = *f.tees.get(1).unwrap();
        t.attack_tick = f.tick - 149;
        f.put(t);
        assert_eq!(f.pick(), 1);
        assert_eq!(f.score(), -50.0 + 500.0, "attacked 149 ticks ago");
        let mut g = single();
        let mut t = *g.tees.get(1).unwrap();
        t.attack_tick = g.tick - 150;
        g.put(t);
        g.pick();
        assert_eq!(g.score(), -50.0, "150 ticks ago is stale");
        let mut h = Fx::open();
        h.set_players(&[(0, "me"), (1, "foe")]);
        h.put(tee(0, 1000.0));
        let mut t = tee(1, 1500.0);
        t.attack_tick = h.tick - 10;
        h.put(t);
        h.pick();
        assert_eq!(h.score(), -125.0, "exactly 500 px is not < 500");
    }

    #[test]
    fn attacking_a_friend_adds_450() {
        let mut f = Fx::open();
        f.rel.add(ListKind::Friend, "pal");
        f.set_players(&[(0, "me"), (1, "foe"), (2, "pal")]);
        f.put(tee(0, 1000.0));
        f.put(tee(1, 1200.0));
        f.put(tee(2, 1250.0));
        f.tick = 1000;
        f.clock.update(f.tick, &f.tees, &f.players, 0);
        // The foe swings next to the friend; by the next snapshot that is "at a friend".
        let mut foe = *f.tees.get(1).unwrap();
        foe.attack_tick = 1001;
        f.put(foe);
        f.tick = 1002;
        assert_eq!(f.pick(), 1);
        // aggressor (+500, fresh attack within 500 px) and at-friend (+450) on top of -50.
        assert_eq!(f.score(), -50.0 + 500.0 + 450.0);
    }

    #[test]
    fn approaching_adds_200_and_the_hold_bonus_400_within_420_px() {
        let mut f = Fx::open();
        f.set_players(&[(0, "me"), (1, "foe")]);
        f.put(tee(0, 1000.0));
        f.put(tee(1, 1300.0));
        assert_eq!(f.pick(), 1);
        assert_eq!(f.score(), -75.0, "first sight at 300 px: no approach yet, no hold yet");
        f.put(tee(1, 1250.0));
        f.tick += 2;
        assert_eq!(f.pick(), 1);
        // approaching (+200), now the current target: hold 400 (250 px <= 420), d = 250.
        assert_eq!(f.score(), -62.5 + 200.0 + 400.0);
        f.put(tee(1, 1250.5));
        f.tick += 2;
        f.pick();
        assert_eq!(
            f.score(),
            -62.625 + 400.0,
            "0.5 px closer than 1 px margin is not approaching: no +200"
        );
    }

    #[test]
    fn the_hold_bonus_fades_linearly_between_420_and_820_px() {
        let mut f = Fx::open();
        f.set_players(&[(0, "me"), (1, "foe")]);
        f.put(tee(0, 1000.0));
        f.put(tee(1, 1000.0 + 620.0));
        f.picker.set_target(1);
        f.pick();
        // d = 620: hold = 400 * (1 - 200/400) = 200; -155 distance; out of reach? d >= 420, reachable
        // in the open room, so no penalty.
        assert_eq!(f.score(), -155.0 + 200.0);
    }

    #[test]
    fn finishing_a_just_frozen_target_near_freeze_adds_600_within_320_px() {
        let mut f = Fx::new(room(120, 40, &[(31, 15, FREEZE)]), None);
        f.set_players(&[(0, "me"), (1, "foe")]);
        // Everyone around tile (31, 15): x ~ 1008, y ~ 496.
        f.put(Tee {
            pos: Vec2::new(900.0, 500.0),
            ..tee(0, 0.0)
        });
        f.put(Tee {
            pos: Vec2::new(1070.0, 500.0),
            ..tee(1, 0.0)
        });
        f.tick = 1000;
        f.clock.update(f.tick, &f.tees, &f.players, 0);
        f.picker.set_target(1);
        let mut t = *f.tees.get(1).unwrap();
        t.frozen = true;
        t.freeze_ticks_left = 100;
        f.put(t);
        f.tick = 1010;
        let got = f.pick();
        // Frozen for 0 ticks since the clock saw the onset at this update; near freeze (2 tiles),
        // not sealed (the empty base world has no such tee): finishing.
        assert_eq!(got, 1);
        // -0.25*170 (distance) + 400 (hold, current target) + 600 (finishing within 320 px).
        assert_eq!(f.score(), -42.5 + 400.0 + 600.0);
        // One snapshot later it has been frozen for 2 ticks > SETTLED (0) but is still "finishing".
        f.tick = 1012;
        assert_eq!(f.pick(), 1);
        // After 151 ticks frozen it stops being finishing and is settled: dropped, but kept (the only one).
        f.tick = 1161;
        assert_eq!(f.pick(), 1, "keepSettled: nobody else, so the old target stays");
        assert_eq!(f.score(), 0.0);
    }

    #[test]
    fn the_seal_search_only_runs_where_it_can_change_the_verdict() {
        let mut f = single();
        let mut t = *f.tees.get(1).unwrap();
        t.frozen = true;
        t.freeze_ticks_left = 100;
        f.put(t);
        f.tick = 1000;
        f.pick(); // first frozen tick: frozen_for == 0, the seal can matter
        assert_eq!(f.picker.seal_times.summary().count, 1);
        for k in 1..20 {
            f.tick = 1000 + 10 * k; // frozen a while, never the target: settled whatever sealed says
            f.picker.set_target(-1);
            f.pick();
        }
        assert_eq!(
            f.picker.seal_times.summary().count,
            1,
            "no further searches for a long-frozen bystander"
        );
    }

    #[test]
    fn a_frozen_player_who_is_not_the_current_target_is_settled_after_its_first_tick() {
        let mut f = single();
        let mut t = *f.tees.get(1).unwrap();
        t.frozen = true;
        t.freeze_ticks_left = 100;
        f.put(t);
        f.tick = 1000;
        assert_eq!(f.pick(), 1, "first frozen tick: frozen for 0 ticks, still eligible");
        f.tick = 1002;
        f.picker.set_target(-1);
        assert_eq!(f.pick(), -1, "frozen for 2 ticks and not the current target: settled");
    }

    #[test]
    fn a_target_with_no_route_loses_700_but_one_in_reach_or_at_war_or_roped_does_not() {
        // A wall column splits the room; the foe is 500 px away on the far side.
        let wall: Vec<(u32, u32, u8)> = (1..39).map(|y| (40, y, SOLID)).collect();
        let build = || {
            let mut f = Fx::new(room(120, 40, &wall), None);
            f.set_players(&[(0, "me"), (1, "foe")]);
            f.put(Tee {
                pos: Vec2::new(38.0 * 32.0, 500.0),
                ..tee(0, 0.0)
            });
            f.put(Tee {
                pos: Vec2::new(38.0 * 32.0 + 500.0, 500.0),
                ..tee(1, 0.0)
            });
            f
        };
        let mut f = build();
        // The first query searches (budget 1): no route -> -700.
        assert_eq!(f.pick(), 1);
        assert_eq!(f.score(), -125.0 - 700.0);
        // At war: no penalty.
        let mut g = build();
        g.rel.add(ListKind::War, "foe");
        g.set_players(&[(0, "me"), (1, "foe")]);
        g.pick();
        assert_eq!(g.score(), -125.0 + 900.0);
        // Roped to us: no penalty.
        let mut h = build();
        let mut t = *h.tees.get(1).unwrap();
        t.hooked_player = 0;
        h.put(t);
        h.pick();
        assert_eq!(h.score(), -125.0 + 1000.0);
        // Closer than 420 px the route is not even asked.
        let mut i = build();
        i.put(Tee {
            pos: Vec2::new(38.0 * 32.0 + 400.0, 500.0),
            ..tee(1, 0.0)
        });
        i.pick();
        assert_eq!(i.score(), -100.0);
        assert_eq!(i.picker.reach_searches(), 0);
    }

    struct Wb {
        skip: Vec<i32>,
        zone: Vec<i32>,
    }
    impl WayBlock for Wb {
        fn holding(&self) -> bool {
            true
        }
        fn filter(&mut self, _c: &crate::hooks::HookContext<'_>, t: &Tee) -> WbFilter {
            WbFilter {
                skip: self.skip.contains(&t.id),
                in_zone: self.zone.contains(&t.id),
                finish_zone: false,
                corridor: false,
            }
        }
    }

    #[test]
    fn the_wayblock_hook_can_skip_a_candidate_or_add_the_zone_bonus() {
        let mut f = single();
        f.hooks.wayblock = Box::new(Wb {
            skip: vec![1],
            zone: vec![],
        });
        assert_eq!(f.pick(), -1);
        let mut g = single();
        g.hooks.wayblock = Box::new(Wb {
            skip: vec![],
            zone: vec![1],
        });
        g.pick();
        assert_eq!(g.score(), -50.0 + 300.0);
    }

    #[test]
    fn filters_friends_far_or_dead_and_a_tie_is_the_lowest_id_and_a_paused_tee_on_the_map_is_fought() {
        let mut f = Fx::open();
        f.rel.add(ListKind::Friend, "pal");
        let list = [
            player(0, "me", "", true, 0, Some(0)),
            player(1, "pal", "", false, 0, Some(0)),
            player(
                2,
                "x",
                "",
                false,
                0,
                Some(ddai_net::generated::enums::explayerflagflag::PAUSED),
            ),
            player(3, "y", "", false, 0, Some(0)),
            player(4, "z", "", false, 0, Some(0)),
            player(5, "far", "", false, 0, Some(0)),
        ];
        f.players.update(&list, &f.rel);
        f.put(tee(0, 1000.0));
        f.put(tee(1, 1100.0)); // friend
        f.put(tee(2, 1100.0)); // paused, its tee still on the map: fought like anybody (af49dfb)
        f.put(tee(3, 1300.0)); // 300 px right
        f.put(tee(4, 700.0)); // 300 px left: equal score, higher id
        f.put(tee(5, 1000.0 + 1601.0)); // too far
        assert_eq!(
            f.pick(),
            2,
            "the paused tee is the nearest foe; friend and far are filtered"
        );
        // Without the paused tee: the lowest id of the tied pair, after every filter.
        let mut g = Fx::open();
        g.rel.add(ListKind::Friend, "pal");
        g.players.update(&list, &g.rel);
        g.put(tee(0, 1000.0));
        g.put(tee(1, 1100.0));
        g.put(tee(3, 1300.0));
        g.put(tee(4, 700.0));
        g.put(tee(5, 1000.0 + 1601.0));
        assert_eq!(g.pick(), 3, "the lowest id of the tied pair, after every filter");
    }

    #[test]
    fn a_fixed_target_picks_by_folded_name_and_ignores_the_filters() {
        let mut f = Fx::new(room(120, 40, &[]), Some("  The   Foe "));
        f.rel.add(ListKind::Friend, "the foe");
        f.set_players(&[(0, "me"), (1, "THE FOE"), (2, "other")]);
        f.put(tee(0, 1000.0));
        f.put(Tee {
            frozen: true,
            ..tee(1, 1000.0 + 1900.0)
        });
        f.put(tee(2, 1050.0));
        assert_eq!(f.pick(), 1, "a far, frozen friend: still the fixed target");
        f.set_players(&[(0, "me"), (2, "other")]);
        assert_eq!(f.pick(), -1, "not on the server");
        f.set_players(&[(0, "THE FOE"), (2, "other")]);
        assert_eq!(f.pick(), -1, "it is us");
    }

    #[test]
    fn spared_covers_ignore_friends_and_afk_in_the_game_but_not_war_or_the_paused() {
        let mut f = Fx::open();
        f.rel.add(ListKind::Ignore, "ign");
        f.rel.add(ListKind::Friend, "pal");
        f.rel.add(ListKind::War, "afkwar");
        let list = [
            player(0, "me", "", true, 0, Some(0)),
            player(1, "ign", "", false, 0, Some(0)),
            player(2, "pal", "", false, 0, Some(0)),
            player(
                3,
                "paused",
                "",
                false,
                0,
                Some(ddai_net::generated::enums::explayerflagflag::PAUSED),
            ),
            player(4, "plain", "", false, 0, Some(0)),
            player(
                5,
                "afkwar",
                "",
                false,
                0,
                Some(ddai_net::generated::enums::explayerflagflag::AFK),
            ),
            player(
                6,
                "afk",
                "",
                false,
                0,
                Some(ddai_net::generated::enums::explayerflagflag::AFK),
            ),
        ];
        f.players.update(&list, &f.rel);
        for id in 0..=6 {
            f.put(tee(id, 1000.0 + 50.0 * id as f32));
        }
        f.clock.update(1000, &f.tees, &f.players, 0);
        let spared = |id: i32| is_spared(f.tees.get(id).unwrap(), 1000, &f.players, &f.clock);
        assert!(spared(1), "ignored");
        assert!(spared(2), "unfrozen friend");
        assert!(
            !spared(3),
            "paused with its tee on the map: no longer spared (af49dfb), only AFK in the game is"
        );
        assert!(!spared(4), "a plain active player is fair game");
        assert!(!spared(5), "server AFK but at war: AFK does not spare");
        assert!(spared(6), "server AFK, not at war");
        // A *frozen* friend is not spared (the hook may help him... TS `!t.frozen && friend`).
        let mut pal = *f.tees.get(2).unwrap();
        pal.frozen = true;
        assert!(!is_spared(&pal, 1000, &f.players, &f.clock));
    }
}
