//! Task 3.12 (`--wb-smart`, opt-in): who is **worth blocking** on Copy Love Box, for the two owner rules of 2026-10-06 (D-103).
//!
//! - **An AFK player is fought only when he is in the way** ([`Core::afk_in_the_way`]). The port skipped every idle tee
//!   (`awayInGame`: no input change for 10 s, or the server's AFK flag) -- on the live session of that day four such tees stood on
//!   the lower shelf of the halls for the whole 30 s of a clip and were never touched. Now an idle tee is a target when
//!   1. **the hall**: he stands in the hall of the side we hold ([`ddai_nav::wayblock::WbDef::in_hall`]: the zone and 3 tiles around
//!      it) -- it is the place we fight from, his body blocks its spots, ropes and shelves, and the only place where an idle tee can
//!      be thrown into the freeze at no cost;
//!   2. **next to us**: within [`AFK_NEXT_TO_PX`] (two tiles; [`AFK_KEEP_PX`] for the one we are fighting now, so that a shove does not
//!      drop him): he blocks our feet and our hook anywhere;
//!   3. **the route**: within [`AFK_ROUTE_TILES`] tiles of one of the next [`AFK_ROUTE_STEPS`] steps of the walk we are on (or, with no
//!      route yet, of the straight line to its goal) and within [`AFK_ROUTE_REACH_PX`] of us: he would stop the walk. The target
//!      selection does not run while a walk drives, so this only counts when it does (a walk that is not driving), and a walk is cut
//!      short only for the one **next to us** (a tee on the route that cut it would stop counting once the walk is gone).
//!
//!   Never in the AFK room of the map (`avoid`), never a friend / ignored tee, never one who is not playing there (those rules stay
//!   where they were). Any other idle tee -- in the other hall, in the passages, anywhere else -- stays unfought, as before.
//! - **The side of the hall** is chosen by the number of **blockable** targets on each side ([`Core::wb_blockable_counts`]).

use super::*;
use ddai_nav::wayblock::WbSide;

/// "Next to us": two tiles.
pub const AFK_NEXT_TO_PX: f32 = 64.0;
/// The idle tee we are fighting now stays "next to us" up to this far (hysteresis against a shove or a hook moving him).
pub const AFK_KEEP_PX: f32 = 96.0;
/// Review F8: the player we are fighting counts as a fight here within this distance (four tiles: a hook or a hammer away).
pub const FIGHT_TARGET_PX: f32 = 128.0;
/// ... when the steering saw him as the target within this many ticks.
pub const FIGHT_TARGET_FRESH_TICKS: i32 = 25;
/// "On the route": within this many tiles of a step of the route.
pub const AFK_ROUTE_TILES: i32 = 2;
/// ... of the next this many steps.
pub const AFK_ROUTE_STEPS: usize = 12;
/// ... and no farther than this from us.
pub const AFK_ROUTE_REACH_PX: f32 = 8.0 * 32.0;

/// Why an idle tee is in the way (for the log and the tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Way {
    Hall,
    NextTo,
    Route,
}

/// Distance from `p` to the segment `a`-`b`.
fn seg_dist(p: Vec2<f32>, a: Vec2<f32>, b: Vec2<f32>) -> f32 {
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    let len2 = dx * dx + dy * dy;
    let t = if len2 <= f32::EPSILON {
        0.0
    } else {
        (((p.x - a.x) * dx + (p.y - a.y) * dy) / len2).clamp(0.0, 1.0)
    };
    dist(p, Vec2::new(a.x + t * dx, a.y + t * dy))
}

impl Core {
    /// Is `t` within [`AFK_ROUTE_TILES`] of the next [`AFK_ROUTE_STEPS`] steps of the walk we are on (or, with no route yet, of the
    /// straight line to its goal)? No allocation.
    fn near_route(&self, t: &Tee, own: Vec2<f32>) -> bool {
        let Some(n) = &self.nav else { return false };
        let (tx, ty) = tile_of(t.pos);
        if let Some(r) = n.current_route() {
            let all = r.steps();
            let ahead = &all[all.len().saturating_sub(r.remaining())..];
            if !ahead.is_empty() {
                return ahead
                    .iter()
                    .take(AFK_ROUTE_STEPS)
                    .any(|s| (s.x - tx).abs() <= AFK_ROUTE_TILES && (s.y - ty).abs() <= AFK_ROUTE_TILES);
            }
        }
        n.goal().is_some_and(|g| {
            let g = Vec2::new((g.tx * 32 + 16) as f32, (g.ty * 32 + 16) as f32);
            seg_dist(t.pos, own, g) <= (AFK_ROUTE_TILES * 32) as f32
        })
    }

    /// The rule above: `Some(why)` when the idle tee `t` is in the way. (The caller has dealt with the friends and the war list;
    /// this only answers the question for a tee that would otherwise be skipped as "away". `current`: he is our target now.)
    pub(super) fn afk_in_the_way(&self, ctx: &HookContext<'_>, t: &Tee, current: bool) -> Option<Way> {
        if !self.cfg.wb_smart || t.id == ctx.own.id || !t.alive {
            return None;
        }
        let (tx, ty) = tile_of(t.pos);
        if let Some(def) = &self.wb.def
            && !def.walk_allowed(tx, ty)
        {
            return None;
        }
        if let (Some(def), Some(side)) = (&self.wb.def, self.wb.side())
            && self.wb_holding()
            && def.in_hall(side, tx, ty)
        {
            return Some(Way::Hall);
        }
        let d = dist(ctx.own.pos, t.pos);
        if d <= if current { AFK_KEEP_PX } else { AFK_NEXT_TO_PX } {
            return Some(Way::NextTo);
        }
        if d <= AFK_ROUTE_REACH_PX && self.near_route(t, ctx.own.pos) {
            return Some(Way::Route);
        }
        None
    }

    /// Is an idle tee at our feet? (The walk is cut short to fight him, like a player worth a fight. Only for the one next to us: see the
    /// module doc.)
    pub(super) fn afk_blocks_the_walk(&self, ctx: &HookContext<'_>) -> bool {
        if !self.cfg.wb_smart {
            return false;
        }
        ctx.tees.iter().any(|t| {
            if t.id == ctx.own.id || t.frozen {
                return false;
            }
            let slot = ctx.players.get(t.id);
            if slot.is_some_and(|s| s.flags.never_target() || s.not_playing()) {
                return false;
            }
            ctx.clock.away_in_game(t.id, ctx.tick, ctx.players)
                && matches!(self.afk_in_the_way(ctx, t, false), Some(Way::NextTo))
        })
    }

    /// The players each hall counts as **blockable** targets (the side choice of `--wb-smart`): alive, not a friend / ignored,
    /// playing, not parked frozen in a freeze, and in or near the hall -- the zone, the approach and 3 tiles around the zone for a
    /// player who is awake; an idle (AFK) one only inside the hall itself, where he is in the way. The war list always counts.
    pub(super) fn wb_blockable_counts(&self, ctx: &HookContext<'_>) -> (i32, i32) {
        let Some(def) = &self.wb.def else { return (0, 0) };
        let (mut l, mut r) = (0, 0);
        for t in ctx.tees.iter() {
            if t.id == ctx.own.id || !t.alive || Self::parked_in_freeze(ctx, t) {
                continue;
            }
            let slot = ctx.players.get(t.id);
            if slot.is_some_and(|s| s.flags.never_target() || s.not_playing()) {
                continue;
            }
            let (tx, ty) = tile_of(t.pos);
            if !def.walk_allowed(tx, ty) {
                continue;
            }
            let active = !ctx.clock.afk(t.id, ctx.tick, ctx.players, true);
            for (side, n) in [(WbSide::Left, &mut l), (WbSide::Right, &mut r)] {
                let hall = def.in_hall(side, tx, ty);
                if hall || (active && def.in_zone(side, tx, ty)) {
                    *n += 1;
                }
            }
        }
        (l, r)
    }

    /// Is a fight going on where we stand? A real engagement (review F8): we hook somebody who may be a target or somebody hooks us, or
    /// the player we are fighting now (our current target, as the steering last saw it) is within [`FIGHT_TARGET_PX`], or somebody who
    /// may be a target has attacked us in the last [`AGGRESSOR_MEMORY_TICKS`]. A free awake tee that merely stands within 420 px is no
    /// fight: it used to hold a committed switch of hall for good. A committed switch waits for a fight to end (`WbState::fight_here`).
    pub(super) fn fighting_here(&self, ctx: &HookContext<'_>) -> bool {
        let me = ctx.own;
        let (target, seen) = self.target_seen;
        let target = (ctx.tick - seen <= FIGHT_TARGET_FRESH_TICKS).then_some(target);
        ctx.tees.iter().any(|t| {
            if t.id == me.id {
                return false;
            }
            let slot = ctx.players.get(t.id);
            if slot.is_some_and(|s| s.flags.never_target() || s.not_playing()) {
                return false;
            }
            t.hooked_player == me.id
                || me.hooked_player == t.id
                || (target == Some(t.id) && dist(me.pos, t.pos) <= FIGHT_TARGET_PX)
                || (t.alive && ctx.clock.at_us_within(t.id, ctx.tick, AGGRESSOR_MEMORY_TICKS))
        })
    }
}
