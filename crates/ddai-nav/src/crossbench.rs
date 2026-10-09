//! Task 3.12b: a walk-to-the-hall benchmark with other tees in the world.
//!
//! [`crate::harness::run_goto`] sends one tee through a tube in an empty world. On Copy Love Box the hall at the end of the tube is
//! where the players are, and the 2026-10-06 session (D-103) lost 64% of its 53 crossings there. This module runs the same walk
//! (spawn, route to the tube, the [`Navigator`]'s crossing, the unstick rules of the live bot) with **other tees in the world**,
//! behaving in a few scripted ways ([`Script`]), the input delay of the live bot (`lag`), and counts what the owner counts: walks that
//! reach the hall, crossings that fail, self-kills.
//!
//! The scripts are a model, not the players: they are calibrated on what the clips of the session show (`ddai-bot`'s `clb_diag`
//! example): nearly every tee that landed frozen in the hall was hooked by somebody within 3 s and flung toward the outer wall.

use std::collections::VecDeque;

use ddai_planner::plan_world::{PlanCollision, PlanWorld};
use ddai_planner::types::{HOOK_GRABBED, PlayerInput, TeeState, empty_input};
use ddai_planner::vmath::Vec2;

use crate::crossing::{Crossing, TileBox, in_any_box};
use crate::harness::respawn_state;
use crate::navigator::{NavCtx, NavGoal, NavOpts, NavPhase, Navigator, tile_goal};
use crate::route::Router;

/// [`WalkSpec::live_ids`]: our id and the first id of the others.
pub const LIVE_OUR_ID: i32 = 12;
pub const LIVE_OTHER_ID_BASE: i32 = 20;

/// What an "other" tee does.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Script {
    /// Stands where it is (an AFK player).
    Idle,
    /// Paces left and right within a few tiles of its home and jumps now and then.
    Wander,
    /// Wanders, and hooks a **frozen** tee of ours that comes within reach (with probability `p`), then walks toward the outer side
    /// of the hall with the hook held, which drags the victim that way, and lets go with a throw of `fling` px/tick outward (the
    /// clips show 12..15: a victim dragged at the hook's speed and let go flies on). `chase`: walks toward a frozen victim in the hall.
    Hunter { p: f64, fling: f64, chase: bool },
    /// Wanders, and hooks **any** tee of ours within reach (a player who plays at us), then pulls it outward.
    Brawler { fling: f64 },
}

/// One other tee.
#[derive(Debug, Clone, Copy)]
pub struct OtherSpec {
    pub home: Vec2,
    pub script: Script,
}

/// A simple deterministic generator (splitmix64).
#[derive(Debug, Clone, Copy)]
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// In `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// In `[0, n)`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }
}

/// The state a script keeps.
#[derive(Debug, Clone, Copy)]
struct OtherState {
    spec: OtherSpec,
    rng: Rng,
    /// Direction of the pacing and the tick it changes.
    dir: i32,
    dir_until: i64,
    /// The hook is held until this tick (a hunter's grip).
    grip_until: i64,
    /// Decided for the current frozen episode of the victim: `Some(true)` hunt it, `Some(false)` leave it.
    decided: Option<bool>,
    /// The next tick a free hunter may start a grip.
    cool_until: i64,
    /// The grip ended this tick and a throw is due.
    throw: bool,
}

/// What a walk is for.
#[derive(Debug, Clone)]
pub struct WalkSpec {
    /// The tube definitions of the map (both sides) and the goal tile.
    pub crossings: Vec<Crossing>,
    pub goal: (i32, i32),
    /// The tube's hall boxes and the side the tee leaves toward (to know the "outer" direction).
    pub hall: Vec<TileBox>,
    pub toward: i32,
    pub others: Vec<OtherSpec>,
    /// The input delay of the live bot in ticks.
    pub lag: i64,
    /// Route 2 after this many failed crossings (the live bot: 2).
    pub route2_after: i32,
    pub max_ticks: i64,
    pub seed: u64,
    /// The crossing's search budget in ms (0: unbounded, deterministic).
    pub budget_ms: f64,
    /// Options of 3.12b (off: the 3.12 behaviour).
    pub smart: crate::crossing::CrossSmart,
    /// Client ids as on a live server (ours [`LIVE_OUR_ID`], the others from [`LIVE_OTHER_ID_BASE`]) instead of 0 and 1..: the
    /// crossing's private world numbers the tees itself, and a hook id copied verbatim would only work when the two coincide.
    pub live_ids: bool,
    /// Print our tee every this many ticks (0: never; a debugging aid of the benchmark).
    pub trace_every: i64,
}

/// What one walk came to.
#[derive(Debug, Clone)]
pub struct WalkResult {
    /// The navigator finished `Arrived` within the limit.
    pub arrived: bool,
    pub ticks: i64,
    pub kills: u32,
    /// Where the tee lay (tile) when it was killed, with the reason.
    pub kill_spots: Vec<(i32, i32, &'static str)>,
    /// Crossings that ended in a "trying again from the spawn" note.
    pub cross_fails: u32,
    /// Crossings started.
    pub cross_starts: u32,
    /// Where each failed crossing ended (the note).
    pub fail_notes: Vec<String>,
    pub notes: Vec<String>,
}

fn tile_of(v: f64) -> i32 {
    (v / 32.0).trunc() as i32
}

/// Unstick rules of the live bot that matter here (`unstick.rs`, `WayBlock::wants_kill`): lying frozen in a freeze tile outside the
/// hall zone for [`WB_LYING`] ticks with nobody hooking us, or frozen for 400 ticks.
const WB_LYING: i64 = 25;
const FROZEN_HARD: i64 = 400;

/// Runs one walk of the tee in `world` (no tee yet). Tee 0 is ours, the others are 1...
pub fn run_walk<W: PlanWorld>(
    world: &mut W,
    router: &mut Router,
    spawns: &[(f64, f64)],
    spawn_at: usize,
    spec: &WalkSpec,
) -> WalkResult {
    let sp = spawns[spawn_at % spawns.len()];
    let start = Vec2 { x: sp.0, y: sp.1 };
    let (us, other_id) = if spec.live_ids {
        (LIVE_OUR_ID, LIVE_OTHER_ID_BASE)
    } else {
        (0, 1)
    };
    world.add_tee(us, start);
    world.apply_tee_state(us, &respawn_state(us, start));
    let mut others: Vec<OtherState> = Vec::new();
    for (i, o) in spec.others.iter().enumerate() {
        let id = i as i32 + other_id;
        world.add_tee(id, o.home);
        world.apply_tee_state(id, &respawn_state(id, o.home));
        others.push(OtherState {
            spec: *o,
            rng: Rng(spec.seed ^ (0xA5A5_0000 + id as u64)),
            dir: 1,
            dir_until: 0,
            grip_until: -1,
            decided: None,
            cool_until: 0,
            throw: false,
        });
    }
    let goal: NavGoal = tile_goal(world.collision(), spec.goal.0, spec.goal.1);
    let mut nav: Navigator<W> = Navigator::new(
        vec![goal],
        NavOpts {
            through_freeze: true,
            crossings: spec.crossings.clone(),
            finish_approach: true,
            ..NavOpts::default()
        },
    );
    nav.cross_budget_ms = spec.budget_ms;
    nav.smart = spec.smart;
    let template = world.new_scratch();
    let mut make = move || template.new_scratch();

    let lag = spec.lag;
    let mut inq: VecDeque<PlayerInput> = (0..lag).map(|_| empty_input()).collect();
    let mut tick = 1000i64;
    let mut res = WalkResult {
        arrived: false,
        ticks: 0,
        kills: 0,
        kill_spots: Vec::new(),
        cross_fails: 0,
        cross_starts: 0,
        fail_notes: Vec::new(),
        notes: Vec::new(),
    };
    let mut n_spawn = spawn_at;
    let (mut frozen_run, mut t) = (0i64, 0i64);
    let mut me_hooked_by: Option<i32>;
    while t < spec.max_ticks {
        let Some(me) = world.get_tee(us) else { break };
        if spec.trace_every > 0 && t % spec.trace_every == 0 {
            println!(
                "t{t:>5} tile({:>3},{:>3}) xy({:>6.0},{:>6.0}) v({:>5.1},{:>5.1}) {} hook{}",
                tile_of(me.pos.x),
                tile_of(me.pos.y),
                me.pos.x,
                me.pos.y,
                me.vel.x,
                me.vel.y,
                if me.frozen { "FROZEN" } else { "      " },
                me.hook_state
            );
        }
        let all = world.all_tees();
        let hooked_now = all
            .iter()
            .any(|o| o.id != us && o.alive && o.hook_state == HOOK_GRABBED && o.hooked_player == us);
        me_hooked_by = hooked_now.then_some(1);
        frozen_run = if me.frozen { frozen_run + 1 } else { 0 };
        // The unstick of the live bot.
        let col = world.collision();
        let in_zone = in_any_box(&spec.hall, tile_of(me.pos.x), tile_of(me.pos.y));
        let in_freeze_tile = [-14.0, 14.0].iter().any(|&dx| {
            [-14.0, 14.0]
                .iter()
                .any(|&dy| col.is_freeze(me.pos.x + dx, me.pos.y + dy))
        });
        let speed = ddai_libm::hypot(me.vel.x, me.vel.y);
        let lying =
            me.frozen && in_freeze_tile && !in_zone && me_hooked_by.is_none() && speed < 0.5 && frozen_run >= WB_LYING;
        let mut kill = lying || frozen_run >= FROZEN_HARD;
        let others_snapshot: Vec<TeeState> = all.iter().filter(|o| o.id != us && o.alive).copied().collect();
        let mut want = empty_input();
        if !kill {
            let mut ctx = NavCtx {
                col: world.collision(),
                router,
                make_sim: &mut make,
            };
            want = nav.step(&mut ctx, &me, tick, &others_snapshot, lag);
            for n in nav.take_notes() {
                if spec.trace_every > 0 {
                    println!("t{t:>5} NOTE {n}");
                }
                if n.contains("at the start of") {
                    res.cross_starts += 1;
                }
                if n.contains("trying again") || n.contains("no way through") {
                    res.cross_fails += 1;
                    res.fail_notes.push(n.clone());
                }
                res.notes.push(n);
            }
            kill = nav.take_kill();
            // route 2 as the live bot: after `route2_after` failed crossings of the walk.
            nav.wall_route = nav.cross_fails() >= spec.route2_after;
            // In: the tee stands free inside the hall (the walk's last metres to the spot are the spot's occupant's business).
            // The live bot also calls the walk off once it stands free in the hall three tiles or more below the spot ("no climb
            // up to it") and goes back to fighting.
            if !me.frozen && in_zone && res.cross_starts > 0 && me.vel.x.abs() < 2.0 && me.vel.y.abs() < 2.0 {
                res.arrived = true;
                break;
            }
            if nav.done() {
                res.arrived = nav.phase() == NavPhase::Arrived;
                break;
            }
        }
        if kill {
            res.kills += 1;
            res.kill_spots.push((
                tile_of(me.pos.x),
                tile_of(me.pos.y),
                if lying {
                    "lying"
                } else if frozen_run >= FROZEN_HARD {
                    "frozen 400"
                } else {
                    "navigator"
                },
            ));
            frozen_run = 0;
            n_spawn += 1;
            let s = spawns[n_spawn % spawns.len()];
            world.apply_tee_state(us, &respawn_state(us, Vec2 { x: s.0, y: s.1 }));
            nav.respawned();
            inq.iter_mut().for_each(|i| *i = empty_input());
            // a kill takes a moment on the live server; the others keep playing meanwhile.
        }
        // the input travels `lag` ticks
        inq.push_back(want);
        let applied = inq.pop_front().unwrap_or_else(empty_input);
        world.set_input(us, applied);
        // the others
        let bot = world.get_tee(us);
        for (i, st) in others.iter_mut().enumerate() {
            let id = i as i32 + other_id;
            let Some(mine) = world.get_tee(id) else { continue };
            if !mine.alive {
                world.apply_tee_state(id, &respawn_state(id, st.spec.home));
                continue;
            }
            let inp = other_input(st, &mine, bot.as_ref(), world.collision(), spec, tick);
            world.set_input(id, inp);
            if st.throw {
                st.throw = false;
                let fling = match st.spec.script {
                    Script::Hunter { fling, .. } | Script::Brawler { fling } => fling,
                    _ => 0.0,
                };
                // the victim is let go while it is being dragged: it flies on at the speed of the drag.
                if bot
                    .as_ref()
                    .is_some_and(|b| b.alive && b.hooked_player != us && fling > 0.0)
                {
                    world.apply_force(
                        us,
                        Vec2 {
                            x: f64::from(spec.toward) * fling,
                            y: -0.4 * fling,
                        },
                    );
                }
            }
        }
        let _ = world.step();
        t += 1;
        tick += 1;
    }
    res.ticks = t;
    res
}

/// What an other tee does this tick.
fn other_input(
    st: &mut OtherState,
    mine: &TeeState,
    bot: Option<&TeeState>,
    col: &impl PlanCollision,
    spec: &WalkSpec,
    tick: i64,
) -> PlayerInput {
    let mut inp = empty_input();
    if mine.frozen {
        return inp;
    }
    let outward = spec.toward;
    // pacing within 6 tiles of home
    let pace = |st: &mut OtherState, inp: &mut PlayerInput| {
        if tick >= st.dir_until {
            st.dir = if st.rng.unit() < 0.5 { -1 } else { 1 };
            st.dir_until = tick + 20 + st.rng.below(50) as i64;
            if st.rng.unit() < 0.3 {
                st.dir = 0;
            }
        }
        let away = mine.pos.x - st.spec.home.x;
        let d = if away > 192.0 {
            -1
        } else if away < -192.0 {
            1
        } else {
            st.dir
        };
        inp.direction = d;
        inp.jump = i32::from(st.rng.unit() < 0.02);
    };
    match st.spec.script {
        Script::Idle => {}
        Script::Wander => pace(st, &mut inp),
        Script::Hunter { .. } | Script::Brawler { .. } => {
            let victim = bot.filter(|b| b.alive);
            let dist = victim.map_or(f64::INFINITY, |b| {
                ddai_jsmath::hypot2(b.pos.x - mine.pos.x, b.pos.y - mine.pos.y)
            });
            let los = victim.is_some_and(|b| col.intersect_line_hook(mine.pos, b.pos).collision == 0);
            let reach = dist <= 330.0 && los;
            let prey = match (st.spec.script, victim) {
                (Script::Hunter { .. }, Some(b)) => b.frozen,
                (Script::Brawler { .. }, Some(_)) => true,
                _ => false,
            };
            if !(prey && dist <= 700.0) {
                st.decided = None;
            }
            if prey && dist <= 700.0 && st.decided.is_none() {
                let p = match st.spec.script {
                    Script::Hunter { p, .. } => p,
                    _ => 1.0,
                };
                st.decided = Some(st.rng.unit() < p);
            }
            let gripping = tick < st.grip_until;
            if st.decided == Some(true) && reach && prey && !gripping && tick >= st.cool_until {
                st.grip_until = tick + 40 + st.rng.below(40) as i64;
                st.cool_until = st.grip_until + 60;
            }
            if gripping && tick + 1 >= st.grip_until {
                st.throw = true;
            }
            if gripping {
                if let Some(b) = victim {
                    inp.hook = 1;
                    inp.target_x = ddai_jsmath::round(b.pos.x - mine.pos.x);
                    inp.target_y = ddai_jsmath::round(b.pos.y - mine.pos.y);
                    if inp.target_x == 0.0 && inp.target_y == 0.0 {
                        inp.target_y = -1.0;
                    }
                }
                // drag toward the outer side
                inp.direction = outward;
                inp.jump = i32::from(st.rng.unit() < 0.15);
            } else if st.decided == Some(true)
                && prey
                && !reach
                && matches!(st.spec.script, Script::Hunter { chase: true, .. })
            {
                // walk toward the frozen victim
                if let Some(b) = victim {
                    inp.direction = if b.pos.x > mine.pos.x { 1 } else { -1 };
                    inp.jump = i32::from(st.rng.unit() < 0.05);
                }
            } else {
                pace(st, &mut inp);
            }
        }
    }
    inp
}
