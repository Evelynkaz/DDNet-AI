//! The wayblock walk of upstream `af49dfb` (`bot.ts`), the parts that live in the navigation hooks:
//!
//! - **route 2** (`updateWbRoute2`, `wbRoute2Why`): after two failed tries to get in, the crossing takes the way
//!   through the far wall of the passage onto the lower shelf (the crosser's `use_wall`);
//! - the **foe** of the walk (`updateWbFoe`, `wbHits`, `noteWbFreeze`, `crossingApproach`, `wbCrowd`): on the way nobody
//!   is fought but the war list; except the player who is at the tube and acts against us, and the one who froze us
//!   three times on this way;
//! - the **guard** (`wbLowerRole`, `wbGuard`, `wbSpot`): see [`crate::wb_guard`].

use super::*;
use crate::wb_guard::{GuardEnv, WB_FALLING_PX, WbGuardState, WbRole, in_zone_boxes, wb_guard as guard_of};
use ddai_nav::crossing::{in_any_box, in_box};
use ddai_nav::wayblock::{WbDef, WbSide, wb_guard, wb_spot};
use std::collections::HashMap;

/// `HARASS_REACH_PX`: how far a player at the tube may be and still be dealt with first.
const HARASS_REACH_PX: f32 = HOOK_LENGTH_PX + 64.0;
/// `HARASS_AIM_RAD`: his aim within this angle of us counts as "at us".
const HARASS_AIM_RAD: f32 = std::f32::consts::PI / 6.0;
/// `HARASS_ROPE_NEAR_PX`: his rope tip this near counts as "at us".
const HARASS_ROPE_NEAR_PX: f32 = 64.0;
/// `HARASS_MAX_TICKS` (a grudge), `HARASS_FIGHT_TICKS` (a harasser): the longest a fight with him lasts.
const HARASS_MAX_TICKS: i32 = 20 * 50;
const HARASS_FIGHT_TICKS: i32 = 6 * 50;
/// `HARASS_SPARE_TICKS`: after a fight without result he is left alone this long.
const HARASS_SPARE_TICKS: i32 = 15 * 50;
const HARASS_LOST_TICKS: i32 = 5 * 50;
const HARASS_LOST_SPARE_TICKS: i32 = 5 * 50;
/// `GRUDGE_FREEZES` in `GRUDGE_WINDOW_TICKS`: a player who froze us this often on the way is dealt with.
const GRUDGE_FREEZES: usize = 3;
const GRUDGE_WINDOW_TICKS: i32 = 3 * 60 * 50;
const GRUDGE_REACH_PX: f32 = 500.0;
const GRUDGE_LOST_TICKS: i32 = 5 * 50;
/// `WB_ROUTE2_AFTER_FAILS`: failed tries before route 2.
const WB_ROUTE2_AFTER_FAILS: i32 = 2;

/// `WB_ROUTE2` (on unless `DDAI_WB_ROUTE2=0`) and `WB_ROUTE2_CROWD` (off unless `DDAI_WB_ROUTE2_CROWD=1`).
pub(super) fn route2_on() -> bool {
    std::env::var("DDAI_WB_ROUTE2").map(|v| v != "0").unwrap_or(true)
}
pub(super) fn route2_crowd_on() -> bool {
    std::env::var("DDAI_WB_ROUTE2_CROWD").map(|v| v == "1").unwrap_or(false)
}

/// `wbRoute2Why(walkFails, crossFails, crowd)`: why route 2 is taken now, or `None`.
pub fn wb_route2_why(walk_fails: i32, cross_fails: i32, crowd: bool) -> Option<String> {
    let fails = walk_fails.max(cross_fails);
    if fails >= WB_ROUTE2_AFTER_FAILS {
        return Some(format!("the last {fails} tries to get in failed"));
    }
    if crowd {
        return Some("a crowd at the tube".to_string());
    }
    None
}

/// `wbFoe`.
#[derive(Debug, Clone)]
pub(super) struct WbFoe {
    pub id: i32,
    pub name_key: String,
    pub grudge: bool,
    pub since_tick: i32,
    pub lost_since_tick: i32,
    pub crossing: Option<Crossing>,
}

/// A player's freezes of us on the way (`wbHits`).
#[derive(Debug, Clone, Default)]
pub(super) struct WbHits {
    pub name_key: String,
    pub ticks: Vec<i32>,
}

/// The guard's view of the hall at one tick (`wbGuardMemo`).
#[derive(Debug, Clone)]
pub(super) struct GuardMemo {
    pub tick: i32,
    pub side: WbSide,
    pub state: WbGuardState,
}

/// The state of the foe, the route and the guard (`wbFoe`, `wbHits`, `wbFoeSpared`, `wbCrowdSaid`, `wbRoute2*`,
/// `wbLower*`, `wbGuardMemo`).
#[derive(Default)]
pub(super) struct WbExtra {
    pub foe: Option<WbFoe>,
    pub hits: HashMap<i32, WbHits>,
    pub spared: HashMap<i32, i32>,
    pub crowd_said: bool,
    pub route2_said: bool,
    pub route2_crowd: bool,
    /// Who froze us since the last poll (`noteWbFreeze` runs at the next poll, with our position).
    pub blocked_by: Vec<i32>,
    pub role: WbRole,
    pub memo: Option<GuardMemo>,
    /// The role and the guard of this tick's target selection (`begin_pick`).
    pub pick_tick: i32,
    pub pick_lower: bool,
    pub pick_target: i32,
    /// `map|WxH` of the map the pauses of the WB walk belong to (`wbPauseKey`).
    pub pause_key: String,
}

impl WbExtra {
    /// What a new map or tick reset forgets (`onTickReset`).
    pub fn reset(&mut self) {
        let key = std::mem::take(&mut self.pause_key);
        *self = WbExtra::default();
        self.pause_key = key;
    }
}

fn friendly(ctx: &HookContext<'_>, id: i32) -> bool {
    ctx.players.get(id).is_some_and(|s| s.flags.friendly())
}

/// `wbFoeKind(id)`: not a friend, not ignored (the partner of TS is dropped, D-021).
fn foe_kind(ctx: &HookContext<'_>, id: i32) -> bool {
    ctx.players
        .get(id)
        .is_none_or(|s| !s.flags.friendly() && !s.flags.ignore)
}

/// `wbFoeAwake(t)`.
fn foe_awake(ctx: &HookContext<'_>, t: &Tee) -> bool {
    t.id != ctx.own.id
        && t.alive
        && !t.frozen
        && !ctx.clock.afk(t.id, ctx.tick, ctx.players, true)
        && foe_kind(ctx, t.id)
}

/// `isGrounded(world, tee)` (`env/obs.ts`).
fn grounded(ctx: &HookContext<'_>, t: &Tee) -> bool {
    ctx.grid.is_solid(t.pos.x + 14.0, t.pos.y + 19.0) || ctx.grid.is_solid(t.pos.x - 14.0, t.pos.y + 19.0)
}

fn tile(t: &Tee) -> (i32, i32) {
    tile_of(t.pos)
}

impl Core {
    /// The tube whose start we are at, not yet thrown (`crossingApproach`).
    fn crossing_approach(&self, ctx: &HookContext<'_>) -> Option<Crossing> {
        let def = self.wb.def.as_ref()?;
        let (c, thrown) = self.nav.as_ref()?.crossing_state()?;
        let (tx, ty) = tile(ctx.own);
        if def
            .crossings
            .iter()
            .any(|c| in_any_box(&c.landing, tx, ty) || in_any_box(&c.exit, tx, ty))
        {
            return None;
        }
        if thrown || !in_any_box(&c.from, tx, ty) {
            return None;
        }
        Some(c.clone())
    }

    /// Foes awake near the tube's start or in its chamber (`wbCrowd`).
    pub(super) fn wb_crowd(ctx: &HookContext<'_>, crossing: &Crossing) -> Vec<i32> {
        let mut out = Vec::new();
        for him in ctx.tees.iter() {
            if !foe_awake(ctx, him) || dist(ctx.own.pos, him.pos) > HARASS_REACH_PX {
                continue;
            }
            let (hx, hy) = tile(him);
            if in_any_box(&crossing.from, hx, hy) || in_box(&crossing.chamber, hx, hy) {
                out.push(him.id);
            }
        }
        out
    }

    /// `wbFoeActing(t, self)`: he hooks or hits at us, or aims at us after a recent action.
    pub(super) fn foe_acting(&self, ctx: &HookContext<'_>, him: &Tee) -> bool {
        let tick = ctx.tick;
        let own = ctx.own;
        if ctx.clock.at_us_within(him.id, tick, AGGRESSOR_MEMORY_TICKS) {
            return true;
        }
        if him.hook_state >= HOOK_FLYING {
            if him.hooked_player == own.id {
                return true;
            }
            if let Some(st) = self.tee_state(him.id)
                && ddai_planner::vmath::vdistance(
                    st.hook_pos,
                    Vec2d {
                        x: f64::from(own.pos.x),
                        y: f64::from(own.pos.y),
                    },
                ) <= f64::from(HARASS_ROPE_NEAR_PX)
            {
                return true;
            }
        }
        if tick - him.attack_tick >= ACTION_MEMORY_TICKS as i32 {
            return false;
        }
        let (dx, dy) = (own.pos.x - him.pos.x, own.pos.y - him.pos.y);
        let d = dx.hypot(dy);
        let aim = him.aim_rad();
        d < 1.0 || aim.cos() * dx + aim.sin() * dy >= d * HARASS_AIM_RAD.cos()
    }

    /// `noteWbFreeze(by, me, tick)`: a player froze us on the way to the WB; three times in three minutes make
    /// him a grudge.
    pub(super) fn note_wb_freeze(&mut self, ctx: &HookContext<'_>, by: i32) {
        if !self.wb_holding() || !foe_kind(ctx, by) {
            return;
        }
        let Some(def) = self.wb.def.as_ref() else { return };
        let (tx, ty) = tile(ctx.own);
        if in_any_box(&def.left.zone, tx, ty) || in_any_box(&def.right.zone, tx, ty) {
            return;
        }
        let on_the_way = (self.wb_walk && self.nav.is_some())
            || self.x.foe.is_some()
            || def.crossings.iter().any(|c| {
                in_any_box(&c.from, tx, ty)
                    || in_box(&c.chamber, tx, ty)
                    || in_any_box(&c.landing, tx, ty)
                    || in_any_box(&c.exit, tx, ty)
            });
        if !on_the_way {
            return;
        }
        let Some(name_key) = ctx.players.get(by).map(|s| s.name_key.clone()) else {
            return;
        };
        let tick = ctx.tick;
        let seen = self.x.hits.remove(&by);
        let mut ticks: Vec<i32> = match seen {
            Some(h) if h.name_key == name_key => h.ticks,
            _ => Vec::new(),
        };
        ticks.retain(|&x| x <= tick && tick - x <= GRUDGE_WINDOW_TICKS);
        ticks.push(tick);
        self.x.hits.insert(by, WbHits { name_key, ticks });
    }

    /// `updateWbFoe(ownId, self)`: ends the fight with the foe of the walk, or starts one.
    pub(super) fn update_wb_foe(&mut self, ctx: &HookContext<'_>) {
        let tick = ctx.tick;
        let own = ctx.own;
        if let Some(foe) = self.x.foe.clone() {
            if ctx.mode != Mode::Fight || !self.wb_holding() {
                self.x.foe = None;
                return;
            }
            let him = ctx.tees.get(foe.id);
            let named = ctx.players.get(foe.id).is_some_and(|s| s.name_key == foe.name_key);
            let mut why: Option<String> = None;
            match him {
                Some(h) if named => {
                    if h.frozen {
                        why = Some("frozen".to_string());
                    } else if ctx.clock.afk(h.id, tick, ctx.players, false) {
                        why = Some("away".to_string());
                    } else if dist(own.pos, h.pos) > if foe.grudge { GRUDGE_REACH_PX } else { HARASS_REACH_PX } {
                        let lost = self.x.foe.as_ref().map_or(-1, |f| f.lost_since_tick);
                        let lost = if lost < 0 { tick } else { lost };
                        if let Some(f) = &mut self.x.foe {
                            f.lost_since_tick = lost;
                        }
                        if tick - lost
                            >= if foe.grudge {
                                GRUDGE_LOST_TICKS
                            } else {
                                HARASS_LOST_TICKS
                            }
                        {
                            why = Some("out of reach".to_string());
                            self.x.spared.insert(foe.id, tick + HARASS_LOST_SPARE_TICKS);
                        }
                    } else if let Some(f) = &mut self.x.foe {
                        f.lost_since_tick = -1;
                    }
                }
                _ => why = Some("gone".to_string()),
            }
            let cap = if foe.grudge {
                HARASS_MAX_TICKS
            } else {
                HARASS_FIGHT_TICKS
            };
            if why.is_none() && tick - foe.since_tick >= cap {
                why = Some(format!("no result in {}s", cap / 50));
                self.x.spared.insert(foe.id, tick + HARASS_SPARE_TICKS);
            }
            if why.is_none()
                && let Some(c) = &foe.crossing
                && Self::wb_crowd(ctx, c).len() >= 2
            {
                why = Some("crowd at the tube".to_string());
                self.x.crowd_said = true;
                self.x.spared.insert(foe.id, tick + HARASS_SPARE_TICKS);
            }
            let Some(why) = why else { return };
            self.x.foe = None;
            if foe.grudge {
                self.x.hits.remove(&foe.id);
            }
            self.log(&format!("WB walk: done with a foe ({why}); on to the WB"));
            self.walk_to_wb(ctx);
            return;
        }
        if !self.wb_walk || self.nav.is_none() || own.frozen || !own.alive || ctx.fixed_target || !self.wb_holding() {
            return;
        }
        let Some(def) = self.wb.def.as_ref() else { return };
        let (tx, ty) = tile(own);
        let standing = grounded(ctx, own);
        let swinging = self
            .nav
            .as_ref()
            .and_then(|n| n.crossing_state())
            .is_some_and(|(_, thrown)| thrown)
            || def
                .crossings
                .iter()
                .any(|c| in_any_box(&c.landing, tx, ty) || in_any_box(&c.exit, tx, ty));
        let mut pick: Option<(i32, bool)> = None; // (id, grudge)
        if !swinging && standing {
            let mut near = f32::INFINITY;
            for (&id, hits) in &self.x.hits {
                let n = hits
                    .ticks
                    .iter()
                    .filter(|&&x| x <= tick && tick - x <= GRUDGE_WINDOW_TICKS)
                    .count();
                if n < GRUDGE_FREEZES {
                    continue;
                }
                let Some(him) = ctx.tees.get(id) else { continue };
                if !foe_awake(ctx, him) || ctx.players.get(id).is_none_or(|s| s.name_key != hits.name_key) {
                    continue;
                }
                let d = dist(own.pos, him.pos);
                if d > GRUDGE_REACH_PX || d >= near || !self.hook_line_clear(own.pos, him.pos) {
                    continue;
                }
                near = d;
                pick = Some((id, true));
            }
        }
        let crossing = if pick.is_none() && standing {
            self.crossing_approach(ctx)
        } else {
            None
        };
        if let Some(c) = &crossing {
            self.sync(ctx);
            let mut near = f32::INFINITY;
            for him in ctx.tees.iter() {
                if !foe_awake(ctx, him)
                    || !self.foe_acting(ctx, him)
                    || self.x.spared.get(&him.id).is_some_and(|&until| until > tick)
                {
                    continue;
                }
                let d = dist(own.pos, him.pos);
                if d > HARASS_REACH_PX || d >= near {
                    continue;
                }
                let (hx, hy) = tile(him);
                if !in_box(&c.chamber, hx, hy) {
                    continue;
                }
                if !self.hook_line_clear(own.pos, him.pos) {
                    continue;
                }
                near = d;
                pick = Some((him.id, false));
            }
            let crowd = Self::wb_crowd(ctx, c).len() >= 2;
            if pick.is_some() && crowd {
                pick = None;
                if !self.x.crowd_said {
                    self.log("WB walk: crowd at the tube, crossing");
                }
                self.x.crowd_said = true;
            }
            if !crowd {
                self.x.crowd_said = false;
            }
        }
        let Some((id, grudge)) = pick else { return };
        let name_key = ctx.players.get(id).map(|s| s.name_key.clone()).unwrap_or_default();
        self.x.foe = Some(WbFoe {
            id,
            name_key,
            grudge,
            since_tick: tick,
            lost_since_tick: -1,
            crossing: if grudge { None } else { crossing },
        });
        let line = self.cancel_nav("a foe first");
        self.log(&line);
        self.log(&if grudge {
            format!("WB walk: c{id} has frozen us {GRUDGE_FREEZES} times on the way, dealing with him first")
        } else {
            format!("WB walk: c{id} is at the tube, dealing with him first")
        });
    }

    /// `col.intersectLineHook(a, b).collision === 0`: nothing that stops a hook between the two points.
    pub(super) fn hook_line_clear(&self, a: Vec2<f32>, b: Vec2<f32>) -> bool {
        let Some(ms) = &self.ms else { return true };
        ms.world
            .collision()
            .intersect_line_hook(
                Vec2d {
                    x: f64::from(a.x),
                    y: f64::from(a.y),
                },
                Vec2d {
                    x: f64::from(b.x),
                    y: f64::from(b.y),
                },
            )
            .collision
            == 0
    }

    /// `updateWbRoute2(self)`: route 2 for the running walk, when the tube has one and the walk failed before.
    pub(super) fn update_wb_route2(&mut self, ctx: &HookContext<'_>) {
        if self.nav.is_none() {
            return;
        }
        let holding = self.wb_holding();
        if !self.wb_route2_on || !self.wb_walk || !holding {
            if let Some(n) = &mut self.nav {
                n.wall_route = false;
            }
            return;
        }
        let own = ctx.own;
        if self.wb_route2_crowd_on
            && !self.x.route2_crowd
            && !own.frozen
            && grounded(ctx, own)
            && let Some(c) = self.crossing_approach(ctx)
            && Self::wb_crowd(ctx, &c).len() >= 2
        {
            self.x.route2_crowd = true;
        }
        let side = self.wb.side();
        let has_wall = match (&self.wb.def, side) {
            (Some(def), Some(s)) => def.side(s).crossing.wall.is_some(),
            _ => false,
        };
        let cross_fails = self.nav.as_ref().map_or(0, |n| n.cross_fails());
        let why = if has_wall {
            wb_route2_why(self.wb.walk_fails(), cross_fails, self.x.route2_crowd)
        } else {
            None
        };
        if let Some(n) = &mut self.nav {
            n.wall_route = why.is_some();
        }
        if let Some(why) = why
            && !self.x.route2_said
            && !own.frozen
        {
            self.x.route2_said = true;
            self.log(&format!(
                "WB walk: route 2 this time ({why}): out of the passage through its far wall, onto the lower shelf"
            ));
        }
    }

    // ---- the guard -------------------------------------------------------------------------------

    /// Whether the guard logic applies now: the WB is held, the side is chosen.
    fn guard_def(&self) -> Option<(&WbDef, WbSide)> {
        if !wb_guard() || !self.wb_holding() {
            return None;
        }
        Some((self.wb.def.as_ref()?, self.wb.side()?))
    }

    /// `wbLowerRole(ownId, def, side)`.
    pub(super) fn role_lower(&mut self, ctx: &HookContext<'_>) -> bool {
        if !wb_guard() || !self.wb_holding() {
            return false;
        }
        let (Some(def), Some(side)) = (self.wb.def.as_ref(), self.wb.side()) else {
            return false;
        };
        // `holds`: an unfrozen friend, not away in the game.
        let holds = |t: &Tee| !t.frozen && friendly(ctx, t.id) && !ctx.clock.away_in_game(t.id, ctx.tick, ctx.players);
        let (lower, said) = self.x.role.update(ctx.tick, ctx.own, def, side, ctx.tees, &holds);
        if let Some(line) = said {
            tracing::info!(target: "nav", "{line}");
        }
        lower
    }

    /// The guard's view of the hall: with `sealed` for the tees on the lower shelf.
    fn compute_guard(
        &self,
        ctx: &HookContext<'_>,
        def: &WbDef,
        side: WbSide,
        target: i32,
        sealed: &mut dyn FnMut(&Tee) -> bool,
    ) -> WbGuardState {
        let eligible = |t: &Tee| {
            t.alive
                && foe_kind(ctx, t.id)
                && !ctx.clock.away_in_game(t.id, ctx.tick, ctx.players)
                && !ctx.players.get(t.id).is_some_and(|s| s.not_playing())
        };
        let friend = |t: &Tee| friendly(ctx, t.id);
        let hook_clear = |a: Vec2<f32>, b: Vec2<f32>| self.hook_line_clear(a, b);
        guard_of(
            ctx.own,
            def,
            side,
            ctx.tees,
            GuardEnv {
                target,
                eligible: &eligible,
                friend: &friend,
                sealed,
                hook_clear: &hook_clear,
            },
        )
    }

    /// `begin_pick`: the role and the guard of this tick's target selection.
    pub(super) fn begin_pick(&mut self, ctx: &HookContext<'_>, target: i32, sealed: &mut dyn FnMut(&Tee) -> bool) {
        self.x.pick_tick = ctx.tick;
        self.x.pick_lower = false;
        self.x.pick_target = target;
        self.x.memo = None;
        let lower = self.role_lower(ctx);
        self.x.pick_lower = lower;
        let Some((def, side)) = self.guard_def() else {
            return;
        };
        let (ox, oy) = tile(ctx.own);
        if lower || !def.in_hall(side, ox, oy) {
            return;
        }
        let state = self.compute_guard(ctx, def, side, target, sealed);
        self.x.memo = Some(GuardMemo {
            tick: ctx.tick,
            side,
            state,
        });
    }

    /// The guard for the spot choice (`wbGuard` from `wbSpot`): this tick's if the target selection made it,
    /// else computed now without the seal checks (nobody is "sealed" then: the shelf's frozen are jobs).
    fn guard_for_spot(&mut self, ctx: &HookContext<'_>, def: &WbDef, side: WbSide) -> WbGuardState {
        if let Some(m) = &self.x.memo
            && m.side == side
            && ctx.tick - m.tick <= 1
        {
            return m.state.clone();
        }
        let target = self.x.pick_target;
        self.compute_guard(ctx, def, side, target, &mut |_| false)
    }

    /// `wbSpot(ownId, def, side, here)`: the guard's anchor (the job spot, or one step off), else the first free spot.
    pub(super) fn wb_spot_for(
        &mut self,
        ctx: &HookContext<'_>,
        def: &WbDef,
        side: WbSide,
        here: (i32, i32),
    ) -> (i32, i32) {
        if wb_guard() {
            let lower = self.role_lower(ctx);
            if !lower && let Some(a) = self.guard_for_spot(ctx, def, side).anchor {
                return a;
            }
        }
        let tees: Vec<TeeState> = ctx.tees.iter().map(|t| self.tee_state_of(t)).collect();
        let is_friend = |id: i32| friendly(ctx, id);
        wb_spot(ctx.own.id, &tees, &is_friend, def, side, Some(here))
    }

    /// `wbFilter` of the target selection (`pickTarget`'s wayblock block).
    pub(super) fn wb_filter_guard(&self, ctx: &HookContext<'_>, cand: &Tee) -> WbFilter {
        let (Some(def), Some(side)) = (&self.wb.def, self.wb.side()) else {
            return WbFilter::default();
        };
        let own = ctx.own;
        let (tx, ty) = tile(cand);
        let me_in_leash = {
            let (ox, oy) = tile(own);
            def.in_hall(side, ox, oy)
        };
        let flags = ctx.players.get(cand.id).map(|s| s.flags).unwrap_or_default();
        let at_war = flags.at_war();
        let d = dist(own.pos, cand.pos);
        let in_leash = def.in_leash(side, tx, ty);
        let at_us = ctx.clock.at_us_within(cand.id, ctx.tick, AGGRESSOR_MEMORY_TICKS);
        let guarding = wb_guard() && self.x.pick_tick == ctx.tick && !self.x.pick_lower;
        let guard = if guarding && me_in_leash {
            self.x
                .memo
                .as_ref()
                .filter(|m| m.tick == ctx.tick && m.side == side)
                .map(|m| &m.state)
        } else {
            None
        };
        let corridor = guard.is_some_and(|g| g.corridor == Some(cand.id));
        let skip = WbFilter {
            skip: true,
            ..WbFilter::default()
        };
        if !me_in_leash {
            // On the way in nobody is fought but the war list (the foe of the walk is picked by `foe_target`).
            if !at_war {
                return skip;
            }
        } else {
            let roped = cand.hooked_player == own.id || own.hooked_player == cand.id;
            let counter = at_us && d <= HOOK_LENGTH_PX + COUNTER_REACH_PX && !in_leash;
            if !roped && !at_war && !counter && !corridor && !in_leash {
                return WbFilter {
                    leash_only: true,
                    ..skip
                };
            }
        }
        let in_zone = def.in_zone(side, tx, ty);
        let roped = cand.hooked_player == own.id || own.hooked_player == cand.id;
        if guarding && in_zone && !at_war && !roped {
            // Somebody falling past the hall, or a frozen one still falling, is no target yet.
            if !cand.frozen && !at_us && cand.vel.y > 0.0 && !in_zone_boxes(def, side, tx, ty) {
                return skip;
            }
            if cand.frozen && cand.vel.y > WB_FALLING_PX {
                return WbFilter {
                    falling: true,
                    in_zone,
                    finish_zone: me_in_leash && in_zone,
                    ..skip
                };
            }
        }
        if let Some(g) = guard
            && !at_war
            && !roped
            && g.lower.contains(&cand.id)
        {
            // The tees on the lower shelf are thrown in from the job spot, one at a time.
            if g.route2 != Some(cand.id) {
                return skip;
            }
            if !g.reached && cand.id != self.x.pick_target {
                return skip;
            }
        }
        WbFilter {
            skip: false,
            in_zone,
            finish_zone: me_in_leash && in_zone,
            corridor,
            leash_only: false,
            falling: false,
        }
    }
}
