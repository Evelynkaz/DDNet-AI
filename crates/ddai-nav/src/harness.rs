//! A single-bot, no-opponent navigation run in any [`PlanWorld`]: the harness the arrival-rate
//! comparison with the TS navigator, the follow scenarios and the WB scenarios build on. The tee starts
//! at a tile, a [`Navigator`] drives it, a `Cl_Kill` request respawns it at the next spawn tile (a fixed
//! cycle) with a clean state — the same rules as `tools/ts-trace/gen-nav-dump.mjs`'s `runNav`.

use ddai_planner::plan_world::{PlanCollision, PlanWorld};
use ddai_planner::types::{TeeState, blank_tee_state};
use ddai_planner::vmath::Vec2;

use crate::crossing::Crossing;
use crate::navigator::{NavCtx, NavGoal, NavOpts, NavPhase, Navigator, tile_goal};
use crate::route::Router;

/// The state of a tee that has just respawned at `pos`.
pub fn respawn_state(id: i32, pos: Vec2) -> TeeState {
    let mut st = blank_tee_state();
    st.id = id;
    st.alive = true;
    st.pos = pos;
    st.hook_pos = pos;
    st.jumps_left = 2;
    st.active_weapon = 1;
    st.deep_frozen = Some(false);
    st
}

/// One goto run.
#[derive(Debug, Clone)]
pub struct GotoSpec {
    pub start: (i32, i32),
    pub goal: (i32, i32),
    pub through_freeze: bool,
    pub crossings: Vec<Crossing>,
    pub max_ticks: i64,
    /// Use the CHANGEs against TS ([`NavOpts::finish_approach`]); `false` is the exact TS behaviour.
    pub improved: bool,
    /// Kill the tee after 400 ticks frozen in a row (the live bot's unstick).
    pub unstick: bool,
}

/// `FROZEN_HARD_LIMIT_TICKS` of the unstick rules.
pub const FROZEN_UNSTICK_TICKS: i64 = 400;

/// What a run came to.
#[derive(Debug, Clone)]
pub struct GotoResult {
    pub phase: NavPhase,
    pub outcome: String,
    pub ticks: i64,
    pub kills: Vec<i64>,
    pub freezes: u32,
    /// Where and when (ticks into the run) the tee first froze.
    pub first_freeze: Option<(f64, f64, i64)>,
    pub dist_px: f64,
    pub notes: Vec<String>,
}

impl GotoResult {
    /// Arrival as the comparison counts it: the navigator finished `arrived` and the tee ended within
    /// 64 px of the goal tile's centre.
    pub fn arrived(&self) -> bool {
        self.phase == NavPhase::Arrived && self.dist_px <= 64.0
    }
}

/// Runs one goto in `world` (which must hold no tee 0 yet; the tee is added here).
pub fn run_goto<W: PlanWorld>(
    world: &mut W,
    router: &mut Router,
    spawns: &[(f64, f64)],
    spec: &GotoSpec,
) -> GotoResult {
    let centre = |t: i32| f64::from(t * 32 + 16);
    let start = Vec2 {
        x: centre(spec.start.0),
        y: centre(spec.start.1),
    };
    world.add_tee(0, start);
    let goal: NavGoal = tile_goal(world.collision(), spec.goal.0, spec.goal.1);
    let mut nav: Navigator<W> = Navigator::new(
        vec![goal],
        NavOpts {
            through_freeze: spec.through_freeze,
            crossings: spec.crossings.clone(),
            finish_approach: spec.improved,
            ..NavOpts::default()
        },
    );
    let template = world.new_scratch();
    let mut make = move || template.new_scratch();
    let mut tick = 1000i64;
    let mut kills = Vec::new();
    let mut notes = Vec::new();
    let mut freezes = 0u32;
    let mut first_freeze: Option<(f64, f64, i64)> = None;
    let mut was_frozen = false;
    let mut n_spawn = 0usize;
    let mut frozen_run = 0i64;
    let mut t = 0i64;
    while t < spec.max_ticks {
        let Some(me) = world.get_tee(0) else { break };
        if me.frozen && !was_frozen {
            freezes += 1;
            first_freeze.get_or_insert((me.pos.x, me.pos.y, t));
        }
        was_frozen = me.frozen;
        frozen_run = if me.frozen { frozen_run + 1 } else { 0 };
        if spec.unstick && frozen_run >= FROZEN_UNSTICK_TICKS {
            // like the live bot: frozen for 400 ticks in a row -> Cl_Kill, respawn at the next spawn
            frozen_run = 0;
            kills.push(t);
            let sp = if spawns.is_empty() {
                start
            } else {
                let s = spawns[n_spawn % spawns.len()];
                n_spawn += 1;
                Vec2 { x: s.0, y: s.1 }
            };
            world.apply_tee_state(0, &respawn_state(0, sp));
            nav.respawned();
            t += 1;
            tick += 1;
            continue;
        }
        let inp = {
            let mut ctx = NavCtx {
                col: world.collision(),
                router,
                make_sim: &mut make,
            };
            nav.step(&mut ctx, &me, tick, &[], 0)
        };
        notes.extend(nav.take_notes());
        if nav.take_kill() {
            kills.push(t);
            let sp = if spawns.is_empty() {
                start
            } else {
                let s = spawns[n_spawn % spawns.len()];
                n_spawn += 1;
                Vec2 { x: s.0, y: s.1 }
            };
            world.apply_tee_state(0, &respawn_state(0, sp));
            nav.respawned();
            t += 1;
            tick += 1;
            continue;
        }
        if nav.done() {
            break;
        }
        world.set_input(0, inp);
        let _ = world.step();
        t += 1;
        tick += 1;
    }
    let dist_px = world.get_tee(0).map_or(f64::INFINITY, |me| {
        ddai_jsmath::hypot2(me.pos.x - centre(spec.goal.0), me.pos.y - centre(spec.goal.1))
    });
    GotoResult {
        phase: nav.phase(),
        outcome: nav.outcome().to_string(),
        ticks: t,
        kills,
        freezes,
        first_freeze,
        dist_px,
        notes,
    }
}

/// Wilson score interval (95%) for `k` successes in `n` trials.
pub fn wilson(k: usize, n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let z = 1.96f64;
    let nf = n as f64;
    let p = k as f64 / nf;
    let denom = 1.0 + z * z / nf;
    let centre = (p + z * z / (2.0 * nf)) / denom;
    let half = z * ((p * (1.0 - p) / nf + z * z / (4.0 * nf * nf)).sqrt()) / denom;
    ((centre - half).max(0.0), (centre + half).min(1.0))
}

/// Whether a tile is a free standing tile (used to pick pairs).
pub fn standable(col: &impl PlanCollision, tx: i32, ty: i32) -> bool {
    crate::wayblock::standable(col, tx, ty)
}

/// A scripted moving target for the follow runs: it walks `path` (pixel points) at `speed` px per tick
/// and pauses `dwell` ticks whenever it reaches a path index in `dwell_at`.
#[derive(Debug, Clone)]
pub struct FollowSpec {
    pub start: (i32, i32),
    pub path: Vec<(f64, f64)>,
    pub dwell_at: Vec<usize>,
    pub speed: f64,
    pub dwell: i64,
    pub through_freeze: bool,
    pub crossings: Vec<Crossing>,
    pub max_ticks: i64,
    pub improved: bool,
    pub unstick: bool,
}

/// What a follow run came to.
#[derive(Debug, Clone)]
pub struct FollowResult {
    pub arrived: bool,
    pub ended: String,
    pub ticks: i64,
    pub kills: usize,
    pub freezes: u32,
    /// Where and when (ticks into the run) the follower first froze.
    pub first_freeze: Option<(f64, f64, i64)>,
    pub dist_px: f64,
}

/// Follows a scripted target in `world` (no tees yet): the follower is tee 0, the target tee 1 is moved
/// kinematically. Mirrors `runFollow` of `tools/ts-trace/gen-nav-dump.mjs`.
pub fn run_follow<W: PlanWorld>(
    world: &mut W,
    router: &mut Router,
    spawns: &[(f64, f64)],
    spec: &FollowSpec,
) -> FollowResult {
    use crate::follow::{Follow, FollowCtx, FollowVerdict, follow_tile};
    use ddai_planner::types::empty_input;

    let centre = |t: i32| f64::from(t * 32 + 16);
    let start = Vec2 {
        x: centre(spec.start.0),
        y: centre(spec.start.1),
    };
    world.add_tee(0, start);
    let p0 = spec.path[0];
    world.add_tee(1, Vec2 { x: p0.0, y: p0.1 });
    let t0 = world.get_tee(1).expect("target");
    let goal = follow_tile(world.collision(), t0.pos)
        .unwrap_or(((t0.pos.x / 32.0).trunc() as i32, (t0.pos.y / 32.0).trunc() as i32));
    let opts = || NavOpts {
        through_freeze: spec.through_freeze,
        crossings: spec.crossings.clone(),
        finish_approach: spec.improved,
        ..NavOpts::default()
    };
    let goal_of = |g: (i32, i32)| NavGoal {
        tx: g.0,
        ty: g.1,
        label: format!("p1 at ({},{})", g.0, g.1),
        tele: None,
    };
    let mut nav: Navigator<W> = Navigator::new(vec![goal_of(goal)], opts());
    let mut follow = Follow::new(1, goal, start, t0.pos, 1000);
    let template = world.new_scratch();
    let mut make = move || template.new_scratch();
    let (mut pi, mut pause) = (0usize, 0i64);
    let (mut kills, mut freezes, mut was_frozen, mut n_spawn) = (0usize, 0u32, false, 0usize);
    let mut first_freeze: Option<(f64, f64, i64)> = None;
    let mut frozen_run = 0i64;
    let mut tick = 1000i64;
    let mut t = 0i64;
    let mut ended = String::new();
    while t < spec.max_ticks {
        let tee1 = world.get_tee(1).expect("target");
        if pause > 0 {
            pause -= 1;
        } else if pi + 1 < spec.path.len() {
            let (tx, ty) = spec.path[pi + 1];
            let d = ddai_jsmath::hypot2(tx - tee1.pos.x, ty - tee1.pos.y);
            let mut st = tee1;
            st.vel = Vec2 { x: 0.0, y: 0.0 };
            if d <= spec.speed {
                pi += 1;
                st.pos = Vec2 { x: tx, y: ty };
                world.apply_tee_state(1, &st);
                if spec.dwell_at.contains(&pi) {
                    pause = spec.dwell;
                }
            } else {
                st.pos = Vec2 {
                    x: tee1.pos.x + (tx - tee1.pos.x) / d * spec.speed,
                    y: tee1.pos.y + (ty - tee1.pos.y) / d * spec.speed,
                };
                world.apply_tee_state(1, &st);
            }
        }
        let me = world.get_tee(0).expect("me");
        if me.frozen && !was_frozen {
            freezes += 1;
            first_freeze.get_or_insert((me.pos.x, me.pos.y, t));
        }
        was_frozen = me.frozen;
        frozen_run = if me.frozen { frozen_run + 1 } else { 0 };
        if spec.unstick && frozen_run >= FROZEN_UNSTICK_TICKS {
            frozen_run = 0;
            kills += 1;
            let sp = if spawns.is_empty() {
                start
            } else {
                let s = spawns[n_spawn % spawns.len()];
                n_spawn += 1;
                Vec2 { x: s.0, y: s.1 }
            };
            world.apply_tee_state(0, &respawn_state(0, sp));
            nav.respawned();
            t += 1;
            tick += 1;
            continue;
        }
        let target = world.get_tee(1);
        let verdict = follow.steer(&FollowCtx {
            tick,
            me: &me,
            target: target.as_ref(),
            target_away: false,
            target_on_server: true,
            nav_phase: nav.phase(),
            nav_outcome: nav.outcome(),
            goal: target.as_ref().and_then(|tt| follow_tile(world.collision(), tt.pos)),
        });
        if std::env::var_os("DDAI_FOLLOW_TRACE").is_some() && t < 40 {
            eprintln!(
                "t={t} me=({:.0},{:.0}) hook {} grabbed {} target=({:.0},{:.0}) phase={} outcome={:?} verdict={verdict:?}",
                me.pos.x,
                me.pos.y,
                me.hook_state,
                me.hooked_player,
                target.map_or(0.0, |x| x.pos.x),
                target.map_or(0.0, |x| x.pos.y),
                nav.phase().name(),
                nav.outcome()
            );
        }
        let go = match verdict {
            FollowVerdict::End(why) => {
                ended = why;
                break;
            }
            FollowVerdict::Wait => false,
            FollowVerdict::Reroute(g) => {
                nav = Navigator::new(vec![goal_of(g)], opts());
                !nav.done()
            }
            FollowVerdict::Go => !nav.done(),
        };
        let mut inp = empty_input();
        if go {
            inp = {
                let mut ctx = NavCtx {
                    col: world.collision(),
                    router,
                    make_sim: &mut make,
                };
                nav.step(&mut ctx, &me, tick, &[], 0)
            };
            let notes = nav.take_notes();
            if std::env::var_os("DDAI_FOLLOW_TRACE").is_some() && t < 40 {
                eprintln!("   notes {notes:?}");
            }
            if nav.take_kill() {
                kills += 1;
                let sp = if spawns.is_empty() {
                    start
                } else {
                    let s = spawns[n_spawn % spawns.len()];
                    n_spawn += 1;
                    Vec2 { x: s.0, y: s.1 }
                };
                world.apply_tee_state(0, &respawn_state(0, sp));
                nav.respawned();
                t += 1;
                tick += 1;
                continue;
            }
        }
        world.set_input(0, inp);
        let _ = world.step();
        t += 1;
        tick += 1;
    }
    let me = world.get_tee(0).expect("me");
    let tee1 = world.get_tee(1).expect("target");
    FollowResult {
        arrived: ended == "arrived",
        ended,
        ticks: t,
        kills,
        freezes,
        first_freeze,
        dist_px: ddai_jsmath::hypot2(me.pos.x - tee1.pos.x, me.pos.y - tee1.pos.y),
    }
}

/// One-sided exact McNemar test on paired outcomes: the probability, if both systems were equally good,
/// of the TS navigator winning at least `only_ts` of the `only_ts + only_rust` pairs where exactly one
/// arrived. A small value means Rust is *significantly* worse.
pub fn mcnemar_worse_p(only_ts: usize, only_rust: usize) -> f64 {
    let n = only_ts + only_rust;
    if n == 0 || only_ts <= only_rust {
        return 1.0;
    }
    // P(X >= only_ts), X ~ Binomial(n, 1/2), summed in log space.
    let ln_choose = |n: usize, k: usize| -> f64 {
        (1..=k)
            .map(|i| ddai_libm::log((n - k + i) as f64) - ddai_libm::log(i as f64))
            .sum()
    };
    (only_ts..=n)
        .map(|k| (ln_choose(n, k) - n as f64 * std::f64::consts::LN_2).exp())
        .sum::<f64>()
        .min(1.0)
}

/// Replays the scripted target of `spec` alone (no follower) in `world` (which must hold no tees) and
/// reports whether it ever moves more than 96 px in one tick. A kinematic target that is physically
/// teleported (it grazes a teleporter tile in the live world, where the TS world does not) makes a follow
/// pair incomparable between the two worlds, so such pairs are dropped (`gen-nav-dump.mjs` does the same
/// check in the TS world).
pub fn target_jumps<W: PlanWorld>(world: &mut W, spec: &FollowSpec) -> bool {
    let p0 = spec.path[0];
    world.add_tee(1, Vec2 { x: p0.0, y: p0.1 });
    let (mut pi, mut pause) = (0usize, 0i64);
    let mut last = world.get_tee(1).map_or(Vec2 { x: p0.0, y: p0.1 }, |t| t.pos);
    for _ in 0..spec.max_ticks {
        let Some(tee1) = world.get_tee(1) else { return false };
        if pause > 0 {
            pause -= 1;
        } else if pi + 1 < spec.path.len() {
            let (tx, ty) = spec.path[pi + 1];
            let d = ddai_jsmath::hypot2(tx - tee1.pos.x, ty - tee1.pos.y);
            let mut st = tee1;
            st.vel = Vec2 { x: 0.0, y: 0.0 };
            if d <= spec.speed {
                pi += 1;
                st.pos = Vec2 { x: tx, y: ty };
                if spec.dwell_at.contains(&pi) {
                    pause = spec.dwell;
                }
            } else {
                st.pos = Vec2 {
                    x: tee1.pos.x + (tx - tee1.pos.x) / d * spec.speed,
                    y: tee1.pos.y + (ty - tee1.pos.y) / d * spec.speed,
                };
            }
            world.apply_tee_state(1, &st);
        }
        let _ = world.step();
        let Some(now) = world.get_tee(1) else { return false };
        if ddai_jsmath::hypot2(now.pos.x - last.x, now.pos.y - last.y) > 96.0 {
            return true;
        }
        last = now.pos;
    }
    false
}

#[cfg(test)]
mod mcnemar_tests {
    use super::*;

    #[test]
    fn mcnemar_is_the_exact_one_sided_binomial_tail() {
        assert_eq!(mcnemar_worse_p(0, 0), 1.0);
        assert_eq!(mcnemar_worse_p(3, 5), 1.0, "Rust better");
        assert!((mcnemar_worse_p(4, 0) - 0.0625).abs() < 1e-12, "4 of 4 = 1/16");
        assert!((mcnemar_worse_p(9, 1) - 11.0 / 1024.0).abs() < 1e-12, "(10 + 1) / 2^10");
    }
}
