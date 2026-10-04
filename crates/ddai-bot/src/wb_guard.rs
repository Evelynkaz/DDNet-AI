//! The wayblock **guard** of upstream `af49dfb` (`bot.ts`: `wbLowerRole`, `wbGuard`, `WB_GUARD`): on Copy Love Box the
//! bot stands at the left (right) end of the upper shelf of its hall (the first spot) and throws the arrivals into the
//! freeze of the wall under it. Two bots share the halls: one holds the upper shelf, the other the lower one. This
//! module is the decision part that needs nothing but the tees and the hall's geometry
//! ([`ddai_nav::wayblock::WbGuardGeom`]):
//!
//! - [`WbRole`]: upper or lower shelf (a friend that holds the upper strip -> we take the lower one; debounced);
//! - [`wb_guard`]: who lies frozen on the lower shelf (the **job**: stand over it and throw it in), who comes down
//!   the passage (route 1), who is in the corridor behind the wall (to be caught first), and where the guard
//!   should stand instead of its spot.
//!
//! What is *not* here is the planner's part of the TS guard (`wallDir`, `airChain`, `frozenTargetSteps`: the planner
//! tries wall-swing throw lines, task 3.8) and the passive seal check (`sealedIn(..., passive)`, also `seal.ts`).

use ddai_nav::crossing::{in_any_box, in_box};
use ddai_nav::wayblock::{WB_NO_CLIMB_TILES, WbDef, WbSide, on_wb_spot};
use ddai_physics::vmath::Vec2;

use crate::consts::HOOK_LENGTH_PX;
use crate::reach::tile_of;
use crate::tees::{Tee, TeeSet, dist};

/// `WB_STRIP_PAD_X` / `WB_STRIP_PAD_Y`: tiles around the line between the first spot and the job that count as
/// "the upper strip" when looking for a friend that holds it.
pub const WB_STRIP_PAD_X: i32 = 2;
pub const WB_STRIP_PAD_Y: i32 = 3;
/// `WB_ROLE_HOLD_TICKS`: how long the lower role is kept after the friend left the strip.
pub const WB_ROLE_HOLD_TICKS: i32 = 100;
/// `WB_ROLE_DEBOUNCE_TICKS`: how long a change of role must be wanted before it is made.
pub const WB_ROLE_DEBOUNCE_TICKS: i32 = 25;
/// `WB_FALLING_PX`: a downward speed above this is "falling past", not "lying there".
pub const WB_FALLING_PX: f32 = 1.0;
/// `WB_CORRIDOR_STEP_PX`, `WB_CORRIDOR_ROPE_PX`: reach of the home spot / the rope toward the corridor.
pub const WB_CORRIDOR_STEP_PX: f32 = 32.0;
pub const WB_CORRIDOR_ROPE_PX: f32 = HOOK_LENGTH_PX - 8.0;
/// `WB_JOB_REACH_X_PX` / `WB_JOB_REACH_Y_PX`: how close to the job spot counts as "there".
pub const WB_JOB_REACH_X_PX: f32 = 48.0;
pub const WB_JOB_REACH_Y_PX: f32 = 40.0;
/// `WB_CORRIDOR_SCORE`: the target-score bonus of the tee in the corridor that blocks the way.
pub const WB_CORRIDOR_SCORE: f32 = 5000.0;

/// Upper or lower shelf (`wbLower`, `wbLowerTick`, `wbWantSince`, `wbLowerHeldTick`, `wbRoleFlips`).
#[derive(Debug, Clone)]
pub struct WbRole {
    lower: bool,
    tick: i32,
    want_since: i32,
    held_tick: i32,
    /// How often the role changed (statistics).
    pub flips: u32,
}

impl Default for WbRole {
    fn default() -> Self {
        WbRole {
            lower: false,
            tick: -1,
            want_since: -1,
            held_tick: i32::MIN / 2,
            flips: 0,
        }
    }
}

/// What a tee other than ourselves is for the role: `holds(t)` is "an unfrozen friend that is not away in the
/// game" (TS: not frozen, `isFriendId`, not `awayInGame`).
impl WbRole {
    /// `wbLowerRole(ownId, def, side)`, once per tick. Returns the role (`true` = lower shelf) and a line for the
    /// log when it changed.
    pub fn update(
        &mut self,
        tick: i32,
        own: &Tee,
        def: &WbDef,
        side: WbSide,
        tees: &TeeSet,
        holds: &dyn Fn(&Tee) -> bool,
    ) -> (bool, Option<&'static str>) {
        if self.tick == tick {
            return (self.lower, None);
        }
        self.tick = tick;
        let first = def.side(side).spots[0];
        let job = def.guard_geom(side).job;
        let lo = first.0.min(job.0) - WB_STRIP_PAD_X;
        let hi = first.0.max(job.0) + WB_STRIP_PAD_X;
        let here = tile_of(own.pos.x, own.pos.y);
        let on_first = on_wb_spot(here, first);
        let below = def.in_hall(side, here.0, here.1) && here.1 - first.1 >= WB_NO_CLIMB_TILES;
        let mut held = false;
        for t in tees.iter() {
            if t.id == own.id || !t.alive || !holds(t) {
                continue;
            }
            let there = tile_of(t.pos.x, t.pos.y);
            if there.0 >= lo && there.0 <= hi && (there.1 - first.1).abs() <= WB_STRIP_PAD_Y {
                held = true;
            }
        }
        if held && !on_first {
            self.held_tick = tick;
        } else if on_first {
            self.held_tick = i32::MIN / 2;
        }
        let since = tick - self.held_tick;
        let lower = below || (!on_first && (held || (0..WB_ROLE_HOLD_TICKS).contains(&since)));
        let mut said = None;
        if lower == self.lower {
            self.want_since = -1;
        } else {
            if self.want_since < 0 || tick < self.want_since {
                self.want_since = tick;
            }
            if tick - self.want_since >= WB_ROLE_DEBOUNCE_TICKS {
                self.lower = lower;
                self.want_since = -1;
                self.flips += 1;
                said = Some(if lower {
                    "WB guard: lower shelf"
                } else {
                    "WB guard: upper shelf"
                });
            }
        }
        (self.lower, said)
    }

    /// The role as of the last update.
    pub fn lower(&self) -> bool {
        self.lower
    }

    /// A new map or a new life of the hall.
    pub fn reset(&mut self) {
        *self = WbRole::default();
    }
}

/// `WbGuardState`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WbGuardState {
    /// Frozen tees on the lower shelf or falling in its column and not sealed (sorted by the time left frozen).
    pub lower: Vec<i32>,
    /// The one to throw in from the job spot ("route 2": the wall), if any.
    pub route2: Option<i32>,
    /// We stand at the job spot.
    pub reached: bool,
    /// The tee in the corridor behind the wall that we can rope from here.
    pub corridor: Option<i32>,
    /// Where to stand instead of the first spot (the job spot, or one step off when the corridor is near).
    pub anchor: Option<(i32, i32)>,
}

/// What [`wb_guard`] needs to ask the world.
pub struct GuardEnv<'a> {
    /// The tee we are fighting now (`targetId`), -1 for none.
    pub target: i32,
    /// A foe: alive, not a friend, not ignored, not away in the game and in the game (`wbFoeKind`, `awayInGame`,
    /// `notPlaying`).
    pub eligible: &'a dyn Fn(&Tee) -> bool,
    /// `isFriendId`.
    pub friend: &'a dyn Fn(&Tee) -> bool,
    /// Is this frozen tee sealed (`wbSealedNow`)? Asked only for tees on the lower shelf or in its column.
    pub sealed: &'a mut dyn FnMut(&Tee) -> bool,
    /// Is the line between two points free of what stops a hook (`intersectLineHook(a, b).collision === 0`)?
    pub hook_clear: &'a dyn Fn(Vec2<f32>, Vec2<f32>) -> bool,
}

/// `wbGuard(ownId, def, side, me)`.
pub fn wb_guard(own: &Tee, def: &WbDef, side: WbSide, tees: &TeeSet, env: GuardEnv<'_>) -> WbGuardState {
    let GuardEnv {
        target,
        eligible,
        friend,
        sealed,
        hook_clear,
    } = env;
    let g = def.guard_geom(side);
    let first = def.side(side).spots[0];
    let home = Vec2::new((first.0 * 32 + 16) as f32, (first.1 * 32 + 16) as f32);
    let mut lower: Vec<&Tee> = Vec::new();
    let mut corridor_tees: Vec<&Tee> = Vec::new();
    let mut route1 = false;
    for t in tees.iter() {
        if t.id == own.id || !t.alive || !eligible(t) {
            continue;
        }
        let (tx, ty) = tile_of(t.pos.x, t.pos.y);
        if t.frozen {
            if (in_box(&g.shelf, tx, ty) || (in_box(&g.column, tx, ty) && t.vel.y > WB_FALLING_PX)) && !sealed(t) {
                lower.push(t);
            } else if in_box(&g.landing, tx, ty) || (in_box(&g.foot, tx, ty) && t.vel.y > WB_FALLING_PX) {
                route1 = true;
            }
        } else if in_box(&g.corridor, tx, ty) {
            corridor_tees.push(t);
        } else if (in_box(&g.foot, tx, ty) || in_box(&g.passage, tx, ty)) && t.vel.y > 0.0 {
            route1 = true;
        }
    }

    let on_lower = tile_of(own.pos.x, own.pos.y).1 - g.job.1 >= WB_NO_CLIMB_TILES;
    if on_lower {
        lower.clear();
    }
    lower.sort_by_key(|t| t.freeze_ticks_left);
    let ours = lower.iter().find(|t| own.hooked_player == t.id).copied();
    let begun = lower.iter().find(|t| t.id == target).copied();
    let route2: Option<&Tee> = ours.or(if route1 { None } else { begun.or(lower.first().copied()) });

    let roped_with = tees.iter().any(|t| {
        t.alive
            && t.id != own.id
            && !friend(t)
            && !corridor_tees.iter().any(|c| c.id == t.id)
            && (t.hooked_player == own.id || own.hooked_player == t.id)
    });
    let mut corridor: Option<i32> = None;
    let mut step_off = false;
    if !roped_with {
        let mut best = f32::INFINITY;
        for t in &corridor_tees {
            let d = dist(own.pos, t.pos);
            if d <= WB_CORRIDOR_ROPE_PX && hook_clear(own.pos, t.pos) && d < best {
                best = d;
                corridor = Some(t.id);
            }
        }
        if !on_lower {
            let (mut home_rope, mut home_near) = (false, false);
            for t in &corridor_tees {
                let d = dist(home, t.pos);
                if d <= WB_CORRIDOR_ROPE_PX && hook_clear(home, t.pos) {
                    home_rope = true;
                } else if d <= HOOK_LENGTH_PX + WB_CORRIDOR_STEP_PX {
                    home_near = true;
                }
            }
            step_off = home_near && !home_rope;
        }
    }
    let anchor = if route2.is_some() {
        Some(g.job)
    } else if step_off {
        Some(g.step_off)
    } else {
        None
    };
    let reached = route2.is_some()
        && (own.pos.x - (g.job.0 * 32 + 16) as f32).abs() <= WB_JOB_REACH_X_PX
        && (own.pos.y - (g.job.1 * 32 + 16) as f32).abs() <= WB_JOB_REACH_Y_PX;
    WbGuardState {
        lower: lower.iter().map(|t| t.id).collect(),
        route2: route2.map(|t| t.id),
        reached,
        corridor,
        anchor,
    }
}

/// Is a side zone box (`zone`, not the approach) around the tile? (`inAnyBox(sideDef(wb, side).zone, …)`.)
pub fn in_zone_boxes(def: &WbDef, side: WbSide, tx: i32, ty: i32) -> bool {
    in_any_box(&def.side(side).zone, tx, ty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddai_nav::wayblock::wayblocks;

    fn tee(id: i32, tx: i32, ty: i32) -> Tee {
        Tee {
            id,
            alive: true,
            pos: Vec2::new((tx * 32 + 16) as f32, (ty * 32 + 16) as f32),
            attack_tick: -10_000,
            ..Tee::DEAD
        }
    }

    fn set(tees: &[Tee]) -> TeeSet {
        let mut s = TeeSet::new();
        for t in tees {
            s.set_for_test(*t);
        }
        s
    }

    fn def() -> WbDef {
        wayblocks().remove(0)
    }

    fn open() -> impl Fn(Vec2<f32>, Vec2<f32>) -> bool {
        |_, _| true
    }

    #[test]
    fn a_friend_on_the_upper_strip_makes_us_take_the_lower_shelf_after_the_debounce_and_back_after_it_left() {
        let d = def();
        let first = d.left.spots[0];
        let own = tee(0, first.0 + 6, first.1 + 1); // in the hall, off the first spot, above the climb limit
        let friend = tee(1, first.0 + 2, first.1);
        let tees = set(&[own, friend]);
        let mut role = WbRole::default();
        let holds = |t: &Tee| t.id == 1;
        let mut said = Vec::new();
        for tick in 0..60 {
            let (lower, s) = role.update(tick, &own, &d, WbSide::Left, &tees, &holds);
            said.extend(s);
            if tick < WB_ROLE_DEBOUNCE_TICKS {
                assert!(!lower, "not before the debounce (tick {tick})");
            }
        }
        assert!(role.lower(), "the friend holds the upper strip: we are the lower one");
        assert_eq!(said, vec!["WB guard: lower shelf"]);
        // The friend leaves: the lower role is kept for WB_ROLE_HOLD_TICKS, then the upper one is wanted
        // again, and made after the debounce.
        let alone = set(&[own]);
        let mut tick = 60;
        let mut back_at = None;
        while tick < 400 {
            let (lower, _) = role.update(tick, &own, &d, WbSide::Left, &alone, &holds);
            if !lower && back_at.is_none() {
                back_at = Some(tick);
            }
            tick += 1;
        }
        let back = back_at.expect("back to the upper shelf");
        assert!(back >= 60 + WB_ROLE_HOLD_TICKS, "held for the hold time: {back}");
        assert_eq!(role.flips, 2);
    }

    #[test]
    fn below_the_first_spot_in_the_hall_is_the_lower_shelf_and_standing_on_it_is_never() {
        let d = def();
        let first = d.left.spots[0];
        let mut role = WbRole::default();
        let low = tee(0, first.0 + 10, first.1 + WB_NO_CLIMB_TILES + 1);
        let tees = set(&[low]);
        let mut lower = false;
        for tick in 0..40 {
            lower = role.update(tick, &low, &d, WbSide::Left, &tees, &|_| false).0;
        }
        assert!(lower, "below the climb limit in the hall");
        let on = tee(0, first.0, first.1);
        let mut role = WbRole::default();
        let tees = set(&[on, tee(1, first.0 + 1, first.1)]);
        for tick in 0..200 {
            assert!(
                !role.update(tick, &on, &d, WbSide::Left, &tees, &|t| t.id == 1).0,
                "on the first spot: upper"
            );
        }
    }

    #[test]
    fn a_frozen_tee_on_the_lower_shelf_is_the_job_and_the_guard_goes_to_the_job_spot() {
        let d = def();
        let g = d.guard_geom(WbSide::Left);
        let first = d.left.spots[0];
        let own = tee(0, first.0, first.1);
        let mut frozen = tee(1, g.shelf.x0 + 3, g.shelf.y0 + 1);
        frozen.frozen = true;
        frozen.freeze_ticks_left = 100;
        let mut frozen2 = tee(2, g.shelf.x0 + 5, g.shelf.y0 + 1);
        frozen2.frozen = true;
        frozen2.freeze_ticks_left = 40;
        let tees = set(&[own, frozen, frozen2]);
        let mut asked = Vec::new();
        let st = wb_guard(
            &own,
            &d,
            WbSide::Left,
            &tees,
            GuardEnv {
                target: -1,
                eligible: &|_| true,
                friend: &|_| false,
                sealed: &mut |t| {
                    asked.push(t.id);
                    false
                },
                hook_clear: &open(),
            },
        );
        assert_eq!(st.lower, vec![2, 1], "sorted by the time left frozen");
        assert_eq!(st.route2, Some(2), "the one that thaws first");
        assert_eq!(st.anchor, Some(g.job));
        assert!(!st.reached, "we are not at the job spot yet");
        assert_eq!(asked.len(), 2);
        // At the job spot it counts as reached.
        let there = tee(0, g.job.0, g.job.1);
        let tees = set(&[there, frozen, frozen2]);
        let st = wb_guard(
            &there,
            &d,
            WbSide::Left,
            &tees,
            GuardEnv {
                target: 1,
                eligible: &|_| true,
                friend: &|_| false,
                sealed: &mut |_| false,
                hook_clear: &open(),
            },
        );
        assert!(st.reached);
        assert_eq!(st.route2, Some(1), "the one begun with stays the job");
    }

    #[test]
    fn a_sealed_tee_is_no_job_and_a_tee_coming_down_the_passage_holds_the_job_back() {
        let d = def();
        let g = d.guard_geom(WbSide::Left);
        let first = d.left.spots[0];
        let own = tee(0, first.0, first.1);
        let mut frozen = tee(1, g.shelf.x0 + 3, g.shelf.y0 + 1);
        frozen.frozen = true;
        let tees = set(&[own, frozen]);
        let sealed = wb_guard(
            &own,
            &d,
            WbSide::Left,
            &tees,
            GuardEnv {
                target: -1,
                eligible: &|_| true,
                friend: &|_| false,
                sealed: &mut |_| true,
                hook_clear: &open(),
            },
        );
        assert!(sealed.lower.is_empty() && sealed.route2.is_none() && sealed.anchor.is_none());
        // Somebody falls down the passage (route 1 is coming): the shelf job waits.
        let mut falling = tee(2, g.passage.x0 + 2, g.passage.y0 + 5);
        falling.vel.y = 8.0;
        let tees = set(&[own, frozen, falling]);
        let st = wb_guard(
            &own,
            &d,
            WbSide::Left,
            &tees,
            GuardEnv {
                target: -1,
                eligible: &|_| true,
                friend: &|_| false,
                sealed: &mut |_| false,
                hook_clear: &open(),
            },
        );
        assert_eq!(st.lower, vec![1]);
        assert_eq!(st.route2, None, "route 1 first");
        assert_eq!(st.anchor, None);
    }

    #[test]
    fn a_tee_in_the_corridor_is_roped_when_in_reach_and_the_guard_steps_off_when_only_the_home_spot_is_near() {
        let d = def();
        let g = d.guard_geom(WbSide::Left);
        let first = d.left.spots[0];
        let own = tee(0, first.0, first.1);
        let near = tee(5, g.corridor.x0 + 2, first.1 + 2);
        let tees = set(&[own, near]);
        let env = |clear: bool| {
            let tees = &tees;
            let own = &own;
            let d = &d;
            wb_guard(
                own,
                d,
                WbSide::Left,
                tees,
                GuardEnv {
                    target: -1,
                    eligible: &|_| true,
                    friend: &|_| false,
                    sealed: &mut |_| false,
                    hook_clear: &move |_, _| clear,
                },
            )
        };
        let clear = env(true);
        assert_eq!(clear.corridor, Some(5), "in reach and the line is free");
        assert_eq!(clear.anchor, None, "home reaches it too: stay");
        let blocked = env(false);
        assert_eq!(blocked.corridor, None, "a wall between: no rope");
        assert_eq!(
            blocked.anchor,
            Some(g.step_off),
            "the corridor is near but the home spot cannot rope it: step off"
        );
        // Not in reach at all (the corridor's far end, more than a rope from both): nothing.
        let away = set(&[own, tee(5, g.corridor.x0, g.corridor.y1)]);
        let st = wb_guard(
            &own,
            &d,
            WbSide::Left,
            &away,
            GuardEnv {
                target: -1,
                eligible: &|_| true,
                friend: &|_| false,
                sealed: &mut |_| false,
                hook_clear: &|_, _| true,
            },
        );
        assert_eq!(st.corridor, None);
        assert_eq!(st.anchor, None);
    }

    #[test]
    fn nothing_is_asked_about_a_tee_that_is_not_a_foe_and_a_rope_with_somebody_stops_the_corridor_catch() {
        let d = def();
        let g = d.guard_geom(WbSide::Left);
        let first = d.left.spots[0];
        let mut own = tee(0, first.0, first.1);
        let corridor = tee(5, g.corridor.x0 + 2, first.1 + 2);
        let mut opponent = tee(6, first.0 + 6, first.1);
        opponent.hooked_player = 0;
        let tees = set(&[own, corridor, opponent]);
        let st = wb_guard(
            &own,
            &d,
            WbSide::Left,
            &tees,
            GuardEnv {
                target: -1,
                eligible: &|_| true,
                friend: &|_| false,
                sealed: &mut |_| false,
                hook_clear: &open(),
            },
        );
        assert_eq!(st.corridor, None, "roped with the opponent: that fight first");
        own.hooked_player = -1;
        // The same opponent as a friend does not count.
        let st = wb_guard(
            &own,
            &d,
            WbSide::Left,
            &tees,
            GuardEnv {
                target: -1,
                eligible: &|_| true,
                friend: &|t| t.id == 6,
                sealed: &mut |_| false,
                hook_clear: &open(),
            },
        );
        assert_eq!(st.corridor, Some(5));
    }
}
