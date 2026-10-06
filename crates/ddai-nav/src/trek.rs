//! Trek / seek (`bot.ts:4251-4329`, `4448-4480`, `4330-4400`): going to where the game is.
//!
//! * [`busiest_spot`] / [`game_spot`]: the crowd centre worth walking to (score `near + busy - dist/1500`,
//!   `None` when the best is nearer than 800 px; `game_spot` also refuses spots inside the WB `avoid`
//!   zones).
//! * [`Trek::start`] (`startTrek`): a partial route to the spot over solid ground only
//!   (`throughFreeze: false`, `allowKill`), refused when it breaks off more than 3 tiles from the spot
//!   next to freeze; [`Trek::goal`] (`trekGoal`) hands the planner the next step as its travel goal,
//!   asks for a `Cl_Kill` at a `kill` step, and bans a step that stalls (`TREK_STALL_TICKS`).
//! * [`PathGoal`] (`pathGoal`): while a target is far or behind a wall, a refreshed route toward it.

use ddai_planner::plan_world::PlanCollision;
use ddai_planner::types::{HOOK_FLYING, TeeState};
use ddai_planner::vmath::{Vec2, vdistance};
use std::collections::HashSet;

use crate::TILE_PX;
use crate::route::{MoveKind, RouteOpts, RouteStep, Router};
use crate::wayblock::{WbDef, wb_walk_allowed};

pub const CROWD_RADIUS_PX: f64 = 600.0;
pub const ACTION_MEMORY_TICKS: i64 = 2 * 50;
pub const TARGET_MAX_PX: f64 = 1600.0;
pub const TREK_REACHED_PX: f64 = 56.0;
pub const TREK_STALL_TICKS: i64 = 150;
pub const PATH_NEAR_PX: f64 = 420.0;
pub const PATH_REACHED_PX: f64 = 56.0;
pub const PATH_REFRESH_TICKS: i64 = 25;
pub const PATH_MIN_REFRESH_TICKS: i64 = 6;
pub const PATH_MOVED_PX: f64 = 96.0;
pub const PATH_PROGRESS_WINDOW: usize = 10;
/// The trek's stall ban set is cleared at this size (`trekAvoid.size >= 24`).
pub const TREK_AVOID_MAX: usize = 24;
/// `REACH_MAX_NODES` for the route checks around seeking.
pub const REACH_MAX_NODES: usize = 20_000;

/// `busiestSpot`'s answer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spot {
    pub x: f64,
    pub y: f64,
    pub dist: f64,
    pub tees: i32,
    pub busy: i32,
}

/// `busiestSpot(ownId, from)`: `tees` are all alive tees but us; `awake(t)` is "not AFK and not parked
/// in freeze" (the bot's activity clock).
pub fn busiest_spot(from: Vec2, tees: &[TeeState], tick: i64, awake: &dyn Fn(&TeeState) -> bool) -> Option<Spot> {
    let active = |t: &TeeState| t.hook_state >= HOOK_FLYING || tick - t.attack_tick < ACTION_MEMORY_TICKS;
    let mut best: Option<Spot> = None;
    let mut best_score = f64::NEG_INFINITY;
    for centre in tees {
        if !awake(centre) {
            continue;
        }
        let (mut near, mut busy) = (0, 0);
        for other in tees {
            if vdistance(centre.pos, other.pos) > CROWD_RADIUS_PX {
                continue;
            }
            if !awake(other) {
                continue;
            }
            near += 1;
            if active(other) {
                busy += 1;
            }
        }
        let dist = vdistance(from, centre.pos);
        let score = f64::from(near) + f64::from(busy) - dist / 1500.0;
        if score > best_score {
            best_score = score;
            best = Some(Spot {
                x: centre.pos.x,
                y: centre.pos.y,
                dist,
                tees: near,
                busy,
            });
        }
    }
    let best = best?;
    if best.dist < TARGET_MAX_PX / 2.0 {
        return None;
    }
    Some(best)
}

/// `gameSpot`: [`busiest_spot`] unless the spot lies in a WB `avoid` zone.
pub fn game_spot(
    wb: Option<&WbDef>,
    from: Vec2,
    tees: &[TeeState],
    tick: i64,
    awake: &dyn Fn(&TeeState) -> bool,
) -> Option<Spot> {
    let spot = busiest_spot(from, tees, tick, awake)?;
    wb_walk_allowed(wb, (spot.x / 32.0).trunc() as i32, (spot.y / 32.0).trunc() as i32).then_some(spot)
}

/// `freezeWithin(tx, ty, r)`.
pub fn freeze_within(col: &impl PlanCollision, tx: i32, ty: i32, r: i32) -> bool {
    for oy in -r..=r {
        for ox in -r..=r {
            if col.is_freeze(f64::from((tx + ox) * 32 + 16), f64::from((ty + oy) * 32 + 16)) {
                return true;
            }
        }
    }
    false
}

/// What [`Trek::goal`] asks of the bot this tick.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrekStep {
    /// The point the planner should head for (`setTravelGoal`); `None` when the trek just ended.
    pub goal: Option<Vec2>,
    /// A `kill` step was passed and the cooldown allows it: send `Cl_Kill`.
    pub kill: bool,
    /// The trek is over (arrived, or a step stalled and was banned).
    pub ended: bool,
    /// A note for the log, e.g. a stall.
    pub note: Option<String>,
}

/// `this.trek`.
#[derive(Debug, Clone)]
pub struct Trek {
    pub steps: Vec<RouteStep>,
    pub at: usize,
    pub since: i64,
    best: f64,
    best_tick: i64,
}

impl Trek {
    /// `startTrek(from, to)`. `avoid` is the per-map set of banned moves (cleared by the caller on a new
    /// map); returns the trek or the refusal text.
    pub fn start(
        router: &mut Router,
        col: &impl PlanCollision,
        from: Vec2,
        to: (f64, f64),
        avoid: &HashSet<i32>,
        tick: i64,
    ) -> Result<Trek, String> {
        Self::start_with(router, col, from, to, avoid, tick, true)
    }

    /// [`Trek::start`]; `allow_kill` false: no route with a respawn step (D-102, `--no-selfkill`).
    pub fn start_with(
        router: &mut Router,
        col: &impl PlanCollision,
        from: Vec2,
        to: (f64, f64),
        avoid: &HashSet<i32>,
        tick: i64,
        allow_kill: bool,
    ) -> Result<Trek, String> {
        let mut route = router.find_route(
            (from.x, from.y),
            to,
            &RouteOpts {
                near_tiles: 3,
                partial: true,
                allow_kill,
                through_freeze: false,
                avoid: Some(avoid),
                ..RouteOpts::default()
            },
        );
        if let Some(r) = &route
            && let Some(end) = r.steps.last()
        {
            let gx = (to.0 / 32.0).trunc() as i32;
            let gy = (to.1 / 32.0).trunc() as i32;
            let short = (end.x - gx).abs() > 3 || (end.y - gy).abs() > 3;
            if short && freeze_within(col, end.x, end.y, 2) {
                route = None;
            }
        }
        match route {
            Some(r) if !r.steps.is_empty() => Ok(Trek {
                steps: r.steps,
                at: 0,
                since: tick,
                best: f64::INFINITY,
                best_tick: tick,
            }),
            _ => Err(format!(
                "no route there (to tile {},{})",
                (to.0 / 32.0).trunc() as i32,
                (to.1 / 32.0).trunc() as i32
            )),
        }
    }

    /// A short text: steps and ropes (`walking over: N steps, M on the rope`).
    pub fn describe(&self) -> String {
        let hooks = self.steps.iter().filter(|s| s.kind == MoveKind::Hook).count();
        format!(
            "walking over: {} steps{}",
            self.steps.len(),
            if hooks > 0 {
                format!(", {hooks} on the rope")
            } else {
                String::new()
            }
        )
    }

    /// Steps still to go.
    pub fn remaining(&self) -> usize {
        self.steps.len().saturating_sub(self.at)
    }

    /// `trekGoal(self)`; `kill_ready`: the `Cl_Kill` cooldown allows a kill now.
    pub fn goal(&mut self, me: Vec2, tick: i64, kill_ready: bool, avoid: &mut HashSet<i32>) -> TrekStep {
        let mut out = TrekStep::default();
        while self.at < self.steps.len() {
            let s = self.steps[self.at].clone();
            if s.kind == MoveKind::Kill {
                self.at += 1;
                self.best = f64::INFINITY;
                self.best_tick = tick;
                if kill_ready {
                    out.kill = true;
                }
                continue;
            }
            let p = Vec2 {
                x: f64::from(s.x * TILE_PX + 16),
                y: f64::from(s.y * TILE_PX + 16),
            };
            let d = vdistance(me, p);
            let tele_next = if s.tele { self.steps.get(self.at + 1) } else { None };
            let reached = match tele_next {
                Some(n) => {
                    vdistance(
                        me,
                        Vec2 {
                            x: f64::from(n.x * TILE_PX + 16),
                            y: f64::from(n.y * TILE_PX + 16),
                        },
                    ) < TREK_REACHED_PX
                }
                None => d < TREK_REACHED_PX,
            };
            if reached {
                self.at += 1;
                self.best = f64::INFINITY;
                self.best_tick = tick;
                continue;
            }
            if d < self.best - 8.0 {
                self.best = d;
                self.best_tick = tick;
            } else if tick - self.best_tick > TREK_STALL_TICKS {
                out.note = Some(format!(
                    "the walk stalled {}px from step {}/{} ({}); rethinking without that move",
                    ddai_jsmath::round(d),
                    self.at + 1,
                    self.steps.len(),
                    match s.kind {
                        MoveKind::Walk => "walk",
                        MoveKind::Fall => "fall",
                        MoveKind::Jump => "jump",
                        MoveKind::Hook => "hook",
                        MoveKind::Kill => "kill",
                    }
                ));
                if avoid.len() >= TREK_AVOID_MAX {
                    avoid.clear();
                }
                avoid.insert(s.move_key);
                out.ended = true;
                return out;
            }
            out.goal = Some(p);
            return out;
        }
        avoid.clear();
        out.ended = true;
        out
    }
}

/// `this.path` (`pathGoal` state).
#[derive(Debug, Clone)]
struct PathState {
    steps: Vec<RouteStep>,
    at: i64,
    target: i32,
    to: Vec2,
    done: usize,
}

/// `pathGoal(self, target)`: while the target is `PATH_NEAR_PX` away or not in a clear line, a route to
/// it (refreshed every 25 ticks, earlier when the target moved 96 px or we left the route) gives the
/// planner a point to head for.
#[derive(Debug, Clone, Default)]
pub struct PathGoal {
    path: Option<PathState>,
}

impl PathGoal {
    pub fn reset(&mut self) {
        self.path = None;
    }

    /// `line_clear(a, b)`: sampled every 16 px, no solid/freeze/death between.
    fn line_is_clear(col: &impl PlanCollision, a: Vec2, b: Vec2) -> bool {
        let steps = ddai_jsmath::max(1.0, (vdistance(a, b) / 16.0).trunc()) as i32;
        for i in 1..steps {
            let x = a.x + ((b.x - a.x) * f64::from(i)) / f64::from(steps);
            let y = a.y + ((b.y - a.y) * f64::from(i)) / f64::from(steps);
            if col.is_solid(x, y) || col.is_freeze(x, y) || col.is_death(x, y) {
                return false;
            }
        }
        true
    }

    pub fn goal(
        &mut self,
        router: &mut Router,
        col: &impl PlanCollision,
        me: Vec2,
        target_id: i32,
        target: Vec2,
        tick: i64,
    ) -> Option<Vec2> {
        let d = vdistance(me, target);
        if d < PATH_NEAR_PX && Self::line_is_clear(col, me, target) {
            self.path = None;
            return None;
        }
        let stale = match &self.path {
            None => true,
            Some(p) => {
                let age = tick - p.at;
                p.target != target_id
                    || age > PATH_REFRESH_TICKS
                    || (age >= PATH_MIN_REFRESH_TICKS && (vdistance(target, p.to) > PATH_MOVED_PX || off_path(p, me)))
            }
        };
        if stale {
            let route = router.find_route(
                (me.x, me.y),
                (target.x, target.y),
                &RouteOpts {
                    near_tiles: 3,
                    partial: false,
                    max_nodes: 4000,
                    through_freeze: false,
                    ..RouteOpts::default()
                },
            );
            self.path = Some(PathState {
                steps: route.map_or_else(Vec::new, |r| r.steps),
                at: tick,
                target: target_id,
                to: Vec2 {
                    x: target.x,
                    y: target.y,
                },
                done: 0,
            });
        }
        path_ahead(self.path.as_mut().expect("path"), me)
    }
}

fn step_px(s: &RouteStep) -> Vec2 {
    Vec2 {
        x: f64::from(s.x * TILE_PX + 16),
        y: f64::from(s.y * TILE_PX + 16),
    }
}

fn path_ahead(path: &mut PathState, at: Vec2) -> Option<Vec2> {
    let steps = &path.steps;
    let mut near = path.done;
    let mut near_d = f64::INFINITY;
    for (i, step) in steps
        .iter()
        .enumerate()
        .take(path.done + PATH_PROGRESS_WINDOW)
        .skip(path.done)
    {
        let d = vdistance(at, step_px(step));
        if d < near_d {
            near_d = d;
            near = i;
        }
    }
    path.done = near;
    for s in steps.iter().skip(near) {
        let p = step_px(s);
        if vdistance(at, p) > PATH_REACHED_PX {
            return Some(p);
        }
    }
    None
}

fn off_path(path: &PathState, at: Vec2) -> bool {
    if path.steps.is_empty() {
        return true;
    }
    for i in path.done..path.steps.len().min(path.done + PATH_PROGRESS_WINDOW) {
        if vdistance(at, step_px(&path.steps[i])) <= PATH_MOVED_PX {
            return false;
        }
    }
    true
}
