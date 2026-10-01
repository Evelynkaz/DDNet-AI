//! Follow mode (`gotoPlayer`/`steerFollow`/`followTile`, `bot.ts:1696-1900`): walk to a player and keep
//! re-routing as they move. The navigator does the walking ([`crate::navigator`]); this decides when to
//! re-aim it and when to give up (120 s of walking, 30 s without getting 32 px closer, 3 deaths of
//! either, 3 times "no way", 5 s without their tee). Arrival: within 64 px and not frozen.

use ddai_planner::plan_world::PlanCollision;
use ddai_planner::types::TeeState;
use ddai_planner::vmath::{Vec2, vdistance};

use crate::navigator::NavPhase;

pub const FOLLOW_ARRIVED_PX: f64 = 64.0;
pub const FOLLOW_LOST_TICKS: i64 = 5 * 50;
pub const FOLLOW_MAX_FAILS: i32 = 3;
pub const FOLLOW_RETRY_TICKS: i64 = 50;
pub const FOLLOW_STALL_TICKS: i64 = 30 * 50;
pub const FOLLOW_MAX_TICKS: i64 = 120 * 50;
pub const FOLLOW_MAX_DEATHS: i32 = 3;
pub const FOLLOW_GOAL_TILES: i32 = 3;
pub const FOLLOW_JUMP_PX: f64 = 8.0 * 32.0;
/// `PATH_MOVED_PX` / `PATH_REFRESH_TICKS`: the target moved this far / the route is this old.
const PATH_MOVED_PX: f64 = 96.0;
const PATH_REFRESH_TICKS: i64 = 25;

/// `followTile(pos)`: the nearest free tile within 3 tiles of `pos`, preferring one with a floor.
pub fn follow_tile(col: &impl PlanCollision, pos: Vec2) -> Option<(i32, i32)> {
    let cx = (pos.x / 32.0).trunc() as i32;
    let cy = (pos.y / 32.0).trunc() as i32;
    let free = |tx: i32, ty: i32| -> bool {
        if tx < 0 || ty < 0 || tx >= col.width() || ty >= col.height() {
            return false;
        }
        let px = f64::from(tx * 32 + 16);
        let py = f64::from(ty * 32 + 16);
        !col.is_solid(px, py) && !col.is_freeze(px, py) && !col.is_death(px, py)
    };
    let mut best: Option<(i32, i32)> = None;
    let mut best_d = f64::INFINITY;
    for oy in -FOLLOW_GOAL_TILES..=FOLLOW_GOAL_TILES {
        for ox in -FOLLOW_GOAL_TILES..=FOLLOW_GOAL_TILES {
            let floor = col.is_solid(f64::from((cx + ox) * 32 + 16), f64::from((cy + oy + 1) * 32 + 16));
            let d = f64::from(ox * ox + oy * oy) + if floor { 0.0 } else { 5.0 };
            if d < best_d && free(cx + ox, cy + oy) {
                best_d = d;
                best = Some((cx + ox, cy + oy));
            }
        }
    }
    best
}

/// `this.follow`.
#[derive(Debug, Clone)]
pub struct Follow {
    pub id: i32,
    pub tx: i32,
    pub ty: i32,
    routed_tick: i64,
    seen_tick: i64,
    start_tick: i64,
    best: f64,
    best_tick: i64,
    fails: i32,
    blocked_at: i64,
    deaths: i32,
    self_deaths: i32,
    last: Option<Vec2>,
    pub waiting: bool,
}

/// What [`Follow::steer`] sees this tick.
pub struct FollowCtx<'a> {
    pub tick: i64,
    pub me: &'a TeeState,
    /// The target's tee, `None` when it has none on the map.
    pub target: Option<&'a TeeState>,
    /// The target is out of the game (spectating / paused).
    pub target_away: bool,
    /// The target is still on the server under the same name.
    pub target_on_server: bool,
    pub nav_phase: NavPhase,
    pub nav_outcome: &'a str,
    /// The tile to aim at now (`followTile(target.pos)`).
    pub goal: Option<(i32, i32)>,
}

/// The verdict of one steering step.
#[derive(Debug, Clone, PartialEq)]
pub enum FollowVerdict {
    /// Let the navigator drive.
    Go,
    /// Nothing to drive this tick (waiting for their tee, a retry, or a finished navigator).
    Wait,
    /// Replace the navigator by one for this tile and drive it.
    Reroute((i32, i32)),
    /// Stop following; the text says why.
    End(String),
}

impl Follow {
    pub fn new(id: i32, goal: (i32, i32), my_pos: Vec2, target_pos: Vec2, tick: i64) -> Follow {
        Follow {
            id,
            tx: goal.0,
            ty: goal.1,
            routed_tick: tick,
            seen_tick: tick,
            start_tick: tick,
            best: vdistance(my_pos, target_pos),
            best_tick: tick,
            fails: 0,
            blocked_at: -1,
            deaths: 0,
            self_deaths: 0,
            last: Some(target_pos),
            waiting: false,
        }
    }

    /// A death was announced (`onKill`): theirs counts for them, ours only when we did not `Cl_Kill`.
    pub fn on_kill(&mut self, victim: i32, own_id: i32, tick: i64, last_own_kill_tick: i64) {
        if victim == self.id {
            self.deaths += 1;
        }
        if victim == own_id && tick - last_own_kill_tick > 50 {
            self.self_deaths += 1;
        }
    }

    /// `steerFollow(self)`.
    pub fn steer(&mut self, c: &FollowCtx<'_>) -> FollowVerdict {
        let tick = c.tick;
        if tick < self.start_tick {
            self.start_tick = tick;
            self.routed_tick = tick;
            self.seen_tick = tick;
            self.best_tick = tick;
            if self.blocked_at >= 0 {
                self.blocked_at = tick;
            }
        }
        if tick - self.start_tick > FOLLOW_MAX_TICKS {
            return FollowVerdict::End(format!("gave up after {}s of walking", FOLLOW_MAX_TICKS / 50));
        }
        if self.self_deaths >= FOLLOW_MAX_DEATHS {
            return FollowVerdict::End(format!("died {} times on the way; giving up", self.self_deaths));
        }
        if self.deaths >= FOLLOW_MAX_DEATHS {
            return FollowVerdict::End(format!(
                "they died {} times before it got to them; giving up",
                self.deaths
            ));
        }
        if !c.target_on_server {
            return FollowVerdict::End("they left the server".to_string());
        }
        let Some(tee) = c.target.filter(|t| t.alive && !c.target_away) else {
            if tick - self.seen_tick > FOLLOW_LOST_TICKS {
                return FollowVerdict::End(format!(
                    "they are not in the game ({})",
                    if c.target_away {
                        "spectating or paused"
                    } else {
                        "no tee on the map"
                    }
                ));
            }
            self.waiting = true;
            return FollowVerdict::Wait;
        };
        self.waiting = false;
        let jumped = self.last.is_some_and(|l| vdistance(l, tee.pos) > FOLLOW_JUMP_PX);
        self.seen_tick = tick;
        self.last = Some(Vec2 {
            x: tee.pos.x,
            y: tee.pos.y,
        });
        let d = vdistance(c.me.pos, tee.pos);
        if d <= FOLLOW_ARRIVED_PX && !c.me.frozen {
            return FollowVerdict::End("arrived".to_string());
        }
        if d < self.best - 32.0 {
            self.best = d;
            self.best_tick = tick;
            self.fails = 0;
        } else if tick - self.best_tick > FOLLOW_STALL_TICKS {
            return FollowVerdict::End(format!("no closer in {}s; giving up", FOLLOW_STALL_TICKS / 50));
        }
        let goal = c.goal;
        let moved = goal.is_some_and(|g| {
            ddai_jsmath::hypot2(f64::from(g.0 - self.tx), f64::from(g.1 - self.ty)) * 32.0 >= PATH_MOVED_PX
        });
        let mut reroute = false;
        if c.nav_phase == NavPhase::Blocked {
            if self.blocked_at < 0 {
                self.blocked_at = tick;
                self.fails += 1;
                if self.fails >= FOLLOW_MAX_FAILS {
                    return FollowVerdict::End(format!("no way to them from here: {}", c.nav_outcome));
                }
            }
            if !goal.is_some_and(|g| g.0 != self.tx || g.1 != self.ty) && tick - self.blocked_at < FOLLOW_RETRY_TICKS {
                return FollowVerdict::Wait;
            }
            reroute = true;
        } else if c.nav_phase == NavPhase::Arrived && !jumped && goal.is_some_and(|g| g.0 == self.tx && g.1 == self.ty)
        {
            return FollowVerdict::End("as near as it gets without the freeze".to_string());
        } else if c.nav_phase == NavPhase::Arrived || jumped || (moved && tick - self.routed_tick >= PATH_REFRESH_TICKS)
        {
            reroute = true;
        }
        if reroute && let Some(g) = goal {
            self.tx = g.0;
            self.ty = g.1;
            self.routed_tick = tick;
            self.blocked_at = -1;
            return FollowVerdict::Reroute(g);
        }
        FollowVerdict::Go
    }
}
