//! Post-mortem of the joniTee test duel, 2026-10-07: **counterfactuals from live clips**.
//!
//! A clip is the live bot's 30 s ring (`ddai-clip`): snapshot frames every 2 ticks with the wire state of the tees and the inputs we sent. Every frame is
//! fed to a `LiveWorld` as the bot did (`ddai_clip::replay::feed`, as `opp_clips` does); at a start frame some ticks before our freeze the reconstructed
//! world is played forward in exact physics (`PhysicsWorld`) for up to `--horizon` ticks with
//!
//! * **us**: the default hybrid as the arena builds it (`brain = "hybrid", mode = "deadline", clock = "work", budget_ms = B`, optionally the 3.15 window
//!   model `m1`), with input lag `L` (arena convention: decided at `T`, in force from the step that makes `T + 1 + L`); the inputs in flight at the
//!   start are the ones we really sent (the clip's `sent`, ticks `T + 1 ..= T + L`);
//! * **him**: either `live-v2` (the competitor's planner, TS-parity port, fixed iterations) with lag `Lo`, holding his snapshot input (`enemy_input_from_tee`)
//!   during his first `Lo` ticks, or **open loop**: the inputs his snapshots show (direction, jump bit, hook held while the hook state is not idle, aim from
//!   the angle, a fire press on the tick his attack tick moves to), replayed whatever we do.
//!
//! The outcome of a run is who goes out (frozen or dead) first: `we` / `he` / `both` / `none` within the horizon, and whether his hook was on us at our freeze.
//! `--sanity` replays both tees open loop (our real inputs, his inferred ones) and reports how far the reconstruction drifts from the clip.
//!
//! ```text
//! cargo run --release -p ddai-env --example pm_counterfactual -- --clips <dir> [--sanity] [--seeds 16] [--threads 3] [--offsets 40,30,20,10]
//!     [--cells base:0/0:4,base:2/1:4,m1:2/1:4,ol:0:4,...] [--horizon 200] [--jsonl out.jsonl]
//! ```
//! Cell syntax: `base:<our lag>/<his lag>:<budget ms>` (hybrid vs live-v2), `m1:<L>/<Lo>:<B>` (hybrid with the window model vs live-v2),
//! `ol:<L>:<B>` (hybrid vs his open-loop inputs), `olm1:<L>:<B>`.

use std::collections::{BTreeMap, VecDeque};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ddai_brain::{Brain, ResetContext, WorldView};
use ddai_clip::format::{Clip, Frame};
use ddai_clip::replay::feed;
use ddai_env::config::{HybridSpec, PlayerSpec, builtin_brain};
use ddai_env::observe;
use ddai_env::sim::wire_from_action;
use ddai_physics::core::PlayerInput as Wire;
use ddai_physics::map::MapData;
use ddai_physics::world::World;
use ddai_planner::physics_adapter::{PhysicsWorld, from_ddnet_input};
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::types::WorldEvent;
use ddai_world::{LiveWorld, player_input_from_net};
use rayon::prelude::*;
use serde_json::json;

const M1: &str = "~/aiddnet/data/runs/E-028/m1.oppnet";

#[derive(Debug, Clone)]
enum Opp {
    /// live-v2 with this lag.
    Planner(u32),
    /// His inputs as the clip shows them.
    OpenLoop,
}

#[derive(Debug, Clone)]
struct Cell {
    name: String,
    lag: u32,
    budget: f64,
    m1: bool,
    opp: Opp,
}

fn parse_cell(s: &str) -> Result<Cell, String> {
    let p: Vec<&str> = s.split(':').collect();
    let bad = || format!("bad cell {s:?}");
    if p.len() != 3 {
        return Err(bad());
    }
    let budget: f64 = p[2].parse().map_err(|_| bad())?;
    match p[0] {
        "base" | "m1" => {
            let (a, b) = p[1].split_once('/').ok_or_else(bad)?;
            Ok(Cell {
                name: s.to_string(),
                lag: a.parse().map_err(|_| bad())?,
                budget,
                m1: p[0] == "m1",
                opp: Opp::Planner(b.parse().map_err(|_| bad())?),
            })
        }
        "ol" | "olm1" => Ok(Cell {
            name: s.to_string(),
            lag: p[1].parse().map_err(|_| bad())?,
            budget,
            m1: p[0] == "olm1",
            opp: Opp::OpenLoop,
        }),
        _ => Err(bad()),
    }
}

fn find_map(dir: &Path, sha: &[u8; 32]) -> Result<PathBuf, String> {
    let hex: String = sha.iter().map(|b| format!("{b:02x}")).collect();
    for e in std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .flatten()
    {
        if e.file_name().to_string_lossy().ends_with(&format!("{hex}.map")) {
            return Ok(e.path());
        }
    }
    Err(format!("no map {hex} in {}", dir.display()))
}

/// One clip, reconstructed.
struct ClipData {
    name: String,
    map: Arc<MapData>,
    own: i32,
    opp: i32,
    freeze_tick: i32,
    /// Our inputs in force by tick (the clip's `sent`).
    ours: BTreeMap<i32, Wire>,
    /// His inputs as inferred from his snapshots, by tick.
    his: BTreeMap<i32, Wire>,
    /// Recorded positions by frame tick: (our x, y, his x, y).
    truth: BTreeMap<i32, [f32; 4]>,
    /// Start states: (offset asked, frame tick, world).
    starts: Vec<(i32, i32, World<f32>)>,
    /// Tees in the snapshot at each start (to flag crowded frames).
    start_tees: Vec<usize>,
}

fn wire_angle_target(angle: i32) -> (i32, i32) {
    let a = f64::from(angle) / 256.0;
    ((a.cos() * 256.0).round() as i32, (a.sin() * 256.0).round() as i32)
}

/// His inputs from his snapshots: for the ticks `(prev.tick, f.tick]` the state the frame `f` shows (direction, jump bit, hook held, aim); a fire
/// press on the tick his attack tick moved to (released the tick after).
fn infer_his(frames: &[Frame], opp: i32) -> BTreeMap<i32, Wire> {
    let mut out = BTreeMap::new();
    let mut fire = 0i32;
    let mut prev_attack: Option<i32> = None;
    let mut prev_tick: Option<i32> = None;
    let mut prev_rec: Option<ddai_clip::format::TeeRec> = None;
    for f in frames {
        let Some(t) = f.tee(opp) else {
            prev_tick = Some(f.tick);
            prev_attack = None;
            prev_rec = None;
            continue;
        };
        let (tx, ty) = wire_angle_target(t.ch.angle);
        let base = Wire {
            direction: t.ch.direction,
            target_x: tx,
            target_y: ty,
            jump: t.ch.jumped & 1,
            fire: 0,
            hook: i32::from(t.ch.hook_state != 0),
            player_flags: 0,
            wanted_weapon: 0,
            next_weapon: 0,
            prev_weapon: 0,
        };
        let press_at = match prev_attack {
            Some(pa) if t.ch.attack_tick != pa => Some(t.ch.attack_tick),
            _ => None,
        };
        // Refinements for a 2-tick gap (the odd tick is not in any snapshot):
        // * a hook launched in the gap: a flying hook moves 80 px/tick from 42 px off the tee on its launch tick, so a hook nearer than ~160 px was
        //   launched on the frame's own tick (the odd tick had the key up); the aim at launch is the hook direction;
        // * a jump pressed in the gap: the jump sets vy to -13.2 (ground) or -12 (air) on its tick and gravity adds 0.5 the next, so vy right at
        //   an impulse means "pressed on the frame's tick".
        let mut odd_hook: Option<i32> = None;
        let mut odd_jump: Option<i32> = None;
        let mut launch_aim: Option<(i32, i32)> = None;
        if let Some(p) = prev_rec
            && prev_tick == Some(f.tick - 2)
        {
            let launched = (p.ch.hook_state == 0 || p.ch.hook_state == -1) && t.ch.hook_state == 4;
            if launched {
                let d = f64::from((t.ch.hook_x - t.ch.x).pow(2) + (t.ch.hook_y - t.ch.y).pow(2)).sqrt();
                launch_aim = Some((t.ch.hook_dx, t.ch.hook_dy));
                if d < 160.0 || p.ch.hook_state == -1 {
                    odd_hook = Some(0);
                }
            }
            if p.ch.jumped & 1 == 0 && t.ch.jumped & 1 == 1 {
                let vy = f64::from(t.ch.vel_y) / 256.0;
                if (vy + 13.2).abs() < 0.2 || (vy + 12.0).abs() < 0.2 {
                    odd_jump = Some(0);
                }
            }
        }
        let from = prev_tick.map_or(f.tick, |p| p + 1);
        for tick in from..=f.tick {
            let mut w = base;
            if tick < f.tick {
                if let Some(h) = odd_hook {
                    w.hook = h;
                }
                if let Some(j) = odd_jump {
                    w.jump = j;
                }
            }
            if let Some((ax, ay)) = launch_aim
                && (ax != 0 || ay != 0)
            {
                w.target_x = ax;
                w.target_y = ay;
            }
            if press_at == Some(tick) {
                fire = (fire | 1) + if fire & 1 == 1 { 2 } else { 0 };
            } else if fire & 1 == 1 {
                fire += 1;
            }
            w.fire = fire;
            out.insert(tick, w);
        }
        prev_attack = Some(t.ch.attack_tick);
        prev_tick = Some(f.tick);
        prev_rec = Some(*t);
    }
    out
}

fn load_clip(path: &Path, maps: &Path, offsets: &[i32]) -> Result<ClipData, String> {
    let clip = Clip::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let map_file = find_map(maps, &clip.header.map_sha256)?;
    let bytes = std::fs::read(&map_file).map_err(|e| e.to_string())?;
    let map = Arc::new(ddai_map::load_map(&bytes).map_err(|e| format!("{e}"))?.data);
    let own = clip.header.own_id;
    let freeze_tick = clip.header.reason.tick;
    // The opponent: the other tee nearest to us most often in the 100 ticks before the freeze.
    let mut counts: BTreeMap<i32, usize> = BTreeMap::new();
    for f in clip
        .frames
        .iter()
        .filter(|f| f.tick >= freeze_tick - 100 && f.tick <= freeze_tick)
    {
        let Some(me) = f.tee(own) else { continue };
        if let Some(t) = f
            .tees
            .iter()
            .filter(|t| t.id != own)
            .min_by_key(|t| (t.ch.x - me.ch.x).pow(2) + (t.ch.y - me.ch.y).pow(2))
        {
            *counts.entry(t.id).or_default() += 1;
        }
    }
    let opp = *counts.iter().max_by_key(|(_, c)| **c).ok_or("no opponent")?.0;
    let mut ours = BTreeMap::new();
    let mut truth = BTreeMap::new();
    for f in &clip.frames {
        for s in &f.sent {
            ours.insert(s.tick, player_input_from_net(s.input.to_net()));
        }
        if let (Some(a), Some(b)) = (f.tee(own), f.tee(opp)) {
            truth.insert(f.tick, [a.ch.x as f32, a.ch.y as f32, b.ch.x as f32, b.ch.y as f32]);
        }
    }
    let his = infer_his(&clip.frames, opp);
    let want: Vec<i32> = offsets.iter().map(|o| freeze_tick - o).collect();
    let mut lw = LiveWorld::new(Arc::clone(&map), own, clip.header.world_seed);
    let mut starts = Vec::new();
    let mut start_tees = Vec::new();
    for f in &clip.frames {
        feed(&mut lw, &clip, f);
        for (k, &w) in want.iter().enumerate() {
            // The frame at the wanted tick, or the first one after it.
            if f.tick >= w && f.tick < w + 2 && f.own_alive && f.tee(own).is_some() && f.tee(opp).is_some() {
                let world = lw.base_world().clone();
                assert_eq!(world.tick, f.tick, "base world tick");
                starts.push((offsets[k], f.tick, world));
                start_tees.push(f.tees.len());
            }
        }
    }
    Ok(ClipData {
        name: path
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().to_string()),
        map,
        own,
        opp,
        freeze_tick,
        ours,
        his,
        truth,
        starts,
        start_tees,
    })
}

fn in_flight_inputs(current: Wire, pending: &VecDeque<(i32, Wire)>, tick: i32, lag: u32) -> Vec<Wire> {
    let mut out = Vec::with_capacity(lag as usize);
    let mut cur = current;
    let mut it = pending.iter().peekable();
    for k in 0..lag as i32 {
        while let Some(&&(apply, input)) = it.peek() {
            if apply <= tick + k + 1 {
                cur = input;
                it.next();
            } else {
                break;
            }
        }
        out.push(cur);
    }
    out
}

struct Player {
    id: i32,
    target: i32,
    brain: Option<Box<dyn Brain>>,
    /// Open loop: the input in force for each tick.
    script: Option<BTreeMap<i32, Wire>>,
    lag: u32,
    current: Wire,
    last_sent: Wire,
    pending: VecDeque<(i32, Wire)>,
}

fn out_state(w: &World<f32>, id: i32) -> bool {
    observe::is_out(w, id)
}

#[derive(Debug, Clone, serde::Serialize)]
struct RunOut {
    first: &'static str,
    /// Ticks after the start.
    at: i32,
    /// His hook held us at our onset.
    hooked: bool,
    /// He hammered us within 20 ticks before our onset.
    hammered: bool,
    /// We died (not only froze).
    died: bool,
}

fn hold_wire(w: &World<f32>, id: i32) -> Wire {
    let mut out = w.characters[id as usize]
        .as_ref()
        .map_or_else(Wire::default, |c| c.input);
    if let Some(c) = w.cores.get(id as u8) {
        out.direction = c.direction;
        out.hook = i32::from(c.hook_state > 0);
        let (tx, ty) = wire_angle_target(c.angle);
        out.target_x = tx;
        out.target_y = ty;
        out.jump = 0;
    }
    out
}

fn play(cd: &ClipData, start: &World<f32>, cell: &Cell, seed: u64, horizon: i32) -> Result<RunOut, String> {
    let map = Arc::clone(&cd.map);
    let mut pw = PhysicsWorld::from_world(start.clone(), Arc::clone(&map));
    pw.sync_from(start);
    let t0 = start.tick;
    let mk_brain = |hybrid: bool, self_id: i32, s: u64| -> Result<Box<dyn Brain>, String> {
        let mut spec = PlayerSpec::simple(if hybrid { "hybrid" } else { "planner" });
        if hybrid {
            spec.mode = Some("deadline".into());
            spec.clock = Some("work".into());
            spec.budget_ms = Some(cell.budget);
            if cell.m1 {
                spec.hybrid = Some(HybridSpec {
                    window_model: Some(M1.into()),
                    ..Default::default()
                });
            }
        } else {
            spec.mode = Some("fixed".into());
            spec.preset = Some("live-v2".into());
        }
        let mut b = builtin_brain(&spec).map_err(|e| e.to_string())?;
        b.reset(&ResetContext {
            map: Arc::clone(&map),
            self_id,
            seed: s,
        });
        Ok(b)
    };
    // Us.
    let mut us = Player {
        id: cd.own,
        target: cd.opp,
        brain: Some(mk_brain(true, cd.own, seed)?),
        script: None,
        lag: cell.lag,
        current: *cd.ours.get(&t0).ok_or("no own input at start")?,
        last_sent: Wire::default(),
        pending: VecDeque::new(),
    };
    for k in 1..=cell.lag as i32 {
        let w = *cd.ours.get(&(t0 + k)).ok_or("no own input in flight")?;
        us.pending.push_back((t0 + k, w));
    }
    us.last_sent = us.pending.back().map_or(us.current, |p| p.1);
    let him = match cell.opp {
        Opp::Planner(lo) => Player {
            id: cd.opp,
            target: cd.own,
            brain: Some(mk_brain(false, cd.opp, seed.wrapping_add(100_000))?),
            script: None,
            lag: lo,
            current: hold_wire(start, cd.opp),
            last_sent: hold_wire(start, cd.opp),
            pending: VecDeque::new(),
        },
        Opp::OpenLoop => Player {
            id: cd.opp,
            target: cd.own,
            brain: None,
            script: Some(cd.his.clone()),
            lag: 0,
            current: *cd.his.get(&t0).unwrap_or(&hold_wire(start, cd.opp)),
            last_sent: Wire::default(),
            pending: VecDeque::new(),
        },
    };
    let mut players = [us, him];
    let ids = [cd.own, cd.opp];
    let mut last_hammer_on_us = i32::MIN / 2;
    for _ in 0..horizon {
        let tick = pw.inner().tick;
        if tick % 2 == 0 {
            for p in players.iter_mut() {
                let Some(brain) = p.brain.as_mut() else { continue };
                if !observe::is_alive(pw.inner(), p.id) {
                    continue;
                }
                let action = match observe::observation(pw.inner(), &map, p.id, &ids, Some(p.target)) {
                    Some(obs) => {
                        let in_flight = if p.lag > 0 {
                            in_flight_inputs(p.current, &p.pending, tick, p.lag)
                        } else {
                            Vec::new()
                        };
                        let view = WorldView {
                            world: pw.inner(),
                            self_id: p.id,
                            lag_ticks: p.lag,
                            in_flight: &in_flight,
                        };
                        brain.decide_in(&obs, Some(&view))
                    }
                    None => ddai_brain::Action::neutral(),
                };
                let wire = wire_from_action(&action, p.last_sent.fire);
                p.last_sent = wire;
                if p.lag > 0 {
                    p.pending.push_back((tick + 1 + p.lag as i32, wire));
                } else {
                    p.current = wire;
                }
            }
        }
        for p in players.iter_mut() {
            if let Some(s) = &p.script {
                if let Some(w) = s.get(&(tick + 1)) {
                    p.current = *w;
                }
            } else {
                while let Some(&(apply, input)) = p.pending.front() {
                    if apply <= tick + 1 {
                        p.current = input;
                        p.pending.pop_front();
                    } else {
                        break;
                    }
                }
            }
            pw.set_input(p.id, from_ddnet_input(&p.current));
        }
        let events = pw.step();
        let now = pw.inner().tick;
        for e in &events {
            if let WorldEvent::HammerHit { from, to } = e
                && *from == cd.opp
                && *to == cd.own
            {
                last_hammer_on_us = now;
            }
        }
        let (a, b) = (out_state(pw.inner(), cd.own), out_state(pw.inner(), cd.opp));
        if a || b {
            let first = match (a, b) {
                (true, true) => "both",
                (true, false) => "we",
                _ => "he",
            };
            return Ok(RunOut {
                first,
                at: now - t0,
                hooked: a && observe::hooked_player(pw.inner(), cd.opp) == cd.own,
                hammered: a && now - last_hammer_on_us <= 20,
                died: a && !observe::is_alive(pw.inner(), cd.own),
            });
        }
    }
    Ok(RunOut {
        first: "none",
        at: horizon,
        hooked: false,
        hammered: false,
        died: false,
    })
}

/// Both tees open loop from a start: the drift of the reconstruction from the clip.
type SanityRow = (i32, f32, f32, bool, bool);

fn sanity(cd: &ClipData, start: &World<f32>) -> (Vec<SanityRow>, Option<i32>, Option<i32>) {
    let mut pw = PhysicsWorld::from_world(start.clone(), Arc::clone(&cd.map));
    pw.sync_from(start);
    let t0 = start.tick;
    let mut out = Vec::new();
    let (mut first_we, mut first_he) = (None, None);
    for k in 1..=60 {
        let tick = t0 + k;
        if let Some(w) = cd.ours.get(&tick) {
            pw.set_input(cd.own, from_ddnet_input(w));
        }
        if let Some(w) = cd.his.get(&tick) {
            pw.set_input(cd.opp, from_ddnet_input(w));
        }
        pw.step();
        if first_we.is_none() && out_state(pw.inner(), cd.own) {
            first_we = Some(k);
        }
        if first_he.is_none() && out_state(pw.inner(), cd.opp) {
            first_he = Some(k);
        }
        if [10, 20, 30, 40].contains(&k)
            && let Some(t) = cd.truth.get(&tick)
        {
            let w = pw.inner();
            let pos = |id: i32| w.cores.get(id as u8).map(|c| (c.pos.x, c.pos.y));
            let d = |p: Option<(f32, f32)>, x: f32, y: f32| {
                p.map_or(f32::NAN, |(a, b)| ((a - x).powi(2) + (b - y).powi(2)).sqrt())
            };
            out.push((
                k,
                d(pos(cd.own), t[0], t[1]),
                d(pos(cd.opp), t[2], t[3]),
                out_state(w, cd.own),
                out_state(w, cd.opp),
            ));
        }
    }
    (out, first_we, first_he)
}

/// The optional hammer diagnostic: replays the hybrid's decision on every frame of a clip (one brain per seed, fed every frame like the live
/// bot, lag 2 with our real in-flight inputs) and, where he is within 48 px and both are free, counts how often it decides a fresh fire press.
/// Variant `live`: our reload timer as `LiveWorld` builds it (always 0 for the hammer); variant `rebuilt`: from our last swing in the clip's
/// events (16 ticks after a hit, 6 after a miss). "Ready" = at least 16 ticks after our last hit and 7 after our last swing.
fn fire_diag(path: &Path, maps: &Path, seeds: u64, lag: u32) -> Result<String, String> {
    let clip = Clip::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let map_file = find_map(maps, &clip.header.map_sha256)?;
    let bytes = std::fs::read(&map_file).map_err(|e| e.to_string())?;
    let map = Arc::new(ddai_map::load_map(&bytes).map_err(|e| format!("{e}"))?.data);
    let own = clip.header.own_id;
    let opp = 0;
    let mut ours = BTreeMap::new();
    for f in &clip.frames {
        for s in &f.sent {
            ours.insert(s.tick, player_input_from_net(s.input.to_net()));
        }
    }
    let ids = [own, opp];
    // [variant][ready] -> (decisions, fire presses); and the live bot's own presses in the same frames.
    let mut cnt = [[(0u64, 0u64); 2]; 2];
    let mut live = [(0u64, 0u64); 2];
    let mut his = [(0u64, 0u64); 2];
    for (variant, cnt_v) in cnt.iter_mut().enumerate() {
        let mut brains: Vec<Box<dyn Brain>> = Vec::new();
        for s in 0..seeds {
            let mut spec = PlayerSpec::simple("hybrid");
            spec.mode = Some("deadline".into());
            spec.clock = Some("work".into());
            spec.budget_ms = Some(4.0);
            let mut b = builtin_brain(&spec).map_err(|e| e.to_string())?;
            b.reset(&ResetContext {
                map: Arc::clone(&map),
                self_id: own,
                seed: 29_001 + s,
            });
            brains.push(b);
        }
        let mut lw = LiveWorld::new(Arc::clone(&map), own, clip.header.world_seed);
        let (mut last_hit, mut last_swing) = (i32::MIN / 2, i32::MIN / 2);
        let (mut his_last_hit, mut his_last_swing) = (i32::MIN / 2, i32::MIN / 2);
        let mut last_fire: Vec<i32> = vec![0; seeds as usize];
        for (i, f) in clip.frames.iter().enumerate() {
            feed(&mut lw, &clip, f);
            for e in &f.events {
                match e {
                    ddai_clip::format::ClipEvent::HammerHit { from, .. } if *from == own => last_hit = f.tick,
                    ddai_clip::format::ClipEvent::HammerFire { from, .. } if *from == own => last_swing = f.tick,
                    ddai_clip::format::ClipEvent::HammerHit { from, .. } if *from == opp => his_last_hit = f.tick,
                    ddai_clip::format::ClipEvent::HammerFire { from, .. } if *from == opp => his_last_swing = f.tick,
                    _ => {}
                }
            }
            let (Some(me), Some(him)) = (f.tee(own), f.tee(opp)) else {
                continue;
            };
            if !f.own_alive || me.frozen || him.frozen {
                continue;
            }
            let t = f.tick;
            let in_flight: Option<Vec<Wire>> = (1..=lag as i32).map(|k| ours.get(&(t + k)).copied()).collect();
            let Some(in_flight) = in_flight else { continue };
            let mut world = lw.base_world().clone();
            if variant == 1
                && let Some(c) = world.characters[own as usize].as_mut()
            {
                let hit_left = 16 - (t - last_hit);
                let swing_left = 6 - (t - last_swing);
                c.reload_timer = if last_hit >= last_swing {
                    hit_left.max(0)
                } else {
                    swing_left.max(0)
                };
            }
            if variant == 1
                && let Some(c) = world.characters[opp as usize].as_mut()
            {
                let hit_left = 16 - (t - his_last_hit);
                let swing_left = 6 - (t - his_last_swing);
                c.reload_timer = if his_last_hit >= his_last_swing {
                    hit_left.max(0)
                } else {
                    swing_left.max(0)
                };
            }
            let d = f64::from((me.ch.x - him.ch.x).pow(2) + (me.ch.y - him.ch.y).pow(2)).sqrt();
            let near = d <= 48.0;
            let ready = usize::from(t - last_hit >= 16 && t - last_swing >= 7);
            let his_ready = usize::from(t - his_last_hit >= 16 && t - his_last_swing >= 7);
            for (si, b) in brains.iter_mut().enumerate() {
                let Some(obs) = observe::observation(&world, &map, own, &ids, Some(opp)) else {
                    continue;
                };
                let view = WorldView {
                    world: &world,
                    self_id: own,
                    lag_ticks: lag,
                    in_flight: &in_flight,
                };
                let a = b.decide_in(&obs, Some(&view));
                let w = wire_from_action(&a, last_fire[si]);
                last_fire[si] = w.fire;
                if near {
                    cnt_v[ready].0 += 1;
                    cnt_v[ready].1 += u64::from(a.fire);
                }
            }
            if variant == 0 && near {
                // The live bot: a fresh press in the inputs it sent for the two ticks after this frame's decision slot.
                let press = |a: Option<&Wire>, b: Option<&Wire>| match (a, b) {
                    (Some(a), Some(b)) => b.fire > a.fire && b.fire & 1 == 1,
                    _ => false,
                };
                let t3 = t + lag as i32 + 1;
                let p = press(ours.get(&(t3 - 1)), ours.get(&t3)) || press(ours.get(&t3), ours.get(&(t3 + 1)));
                live[ready].0 += 1;
                live[ready].1 += u64::from(p);
                // His swings seen in the next two frames.
                let next_swing = clip.frames[i + 1..clip.frames.len().min(i + 2)].iter().any(|g| {
                    g.events
                        .iter()
                        .any(|e| matches!(e, ddai_clip::format::ClipEvent::HammerFire { from, .. } if *from == opp))
                });
                his[his_ready].0 += 1;
                his[his_ready].1 += u64::from(next_swing);
            }
        }
    }
    let pc = |x: (u64, u64)| format!("{}/{} = {:.1}%", x.1, x.0, 100.0 * x.1 as f64 / x.0.max(1) as f64);
    Ok(format!(
        "{}: frames within 48 px, both free. hybrid replay (lag {lag}, {seeds} seeds, fire decided per decision): reload as live (0): ready {} | not ready {}; reload rebuilt (ours and his): ready {} | not ready {}. live bot presses (per frame): ready {} | not ready {}. his swings in the next frame: his hammer ready {} | not ready {}",
        path.file_name().unwrap_or_default().to_string_lossy(),
        pc(cnt[0][1]),
        pc(cnt[0][0]),
        pc(cnt[1][1]),
        pc(cnt[1][0]),
        pc(live[1]),
        pc(live[0]),
        pc(his[1]),
        pc(his[0])
    ))
}

fn main() -> Result<(), String> {
    let mut clips_dir = PathBuf::new();
    let mut maps = PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("aiddnet/data/maps/cache");
    let mut seeds = 16u64;
    let mut threads = 3usize;
    let mut horizon = 200i32;
    let mut offsets = vec![40, 30, 20, 10];
    let mut do_sanity = false;
    let mut jsonl: Option<PathBuf> = None;
    let mut only: Option<String> = None;
    let mut cells_arg = String::from(
        "base:0/0:4,base:0/1:4,base:2/0:4,base:2/1:4,base:3/0:4,base:3/1:4,base:2/1:2,base:2/1:3,m1:2/1:4,m1:3/1:4,ol:0:4,ol:2:4,ol:3:4",
    );
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--clips" => clips_dir = PathBuf::from(v()?),
            "--maps" => maps = PathBuf::from(v()?),
            "--seeds" => seeds = v()?.parse().map_err(|e| format!("--seeds: {e}"))?,
            "--threads" => threads = v()?.parse().map_err(|e| format!("--threads: {e}"))?,
            "--horizon" => horizon = v()?.parse().map_err(|e| format!("--horizon: {e}"))?,
            "--offsets" => {
                offsets = v()?
                    .split(',')
                    .map(|s| s.parse::<i32>().map_err(|e| format!("--offsets: {e}")))
                    .collect::<Result<_, _>>()?;
            }
            "--cells" => cells_arg = v()?,
            "--sanity" => do_sanity = true,
            "--jsonl" => jsonl = Some(PathBuf::from(v()?)),
            "--only" => only = Some(v()?),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let cells: Vec<Cell> = if cells_arg.is_empty() {
        Vec::new()
    } else {
        cells_arg.split(',').map(parse_cell).collect::<Result<_, _>>()?
    };
    if let Ok(n) = std::env::var("PM_FIRE_DIAG") {
        let seeds: u64 = n.parse().map_err(|e| format!("PM_FIRE_DIAG: {e}"))?;
        let mut files: Vec<PathBuf> = std::fs::read_dir(&clips_dir)
            .map_err(|e| e.to_string())?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "clip"))
            .collect();
        files.sort();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .stack_size(64 << 20)
            .build()
            .map_err(|e| e.to_string())?;
        let out: Vec<Result<String, String>> =
            pool.install(|| files.par_iter().map(|f| fire_diag(f, &maps, seeds, 2)).collect());
        for o in out {
            println!("{}", o?);
        }
        return Ok(());
    }
    if let Ok(p) = std::env::var("PM_TUNING") {
        // The tuning a clip recorded (to compare servers/maps).
        for f in p.split(',') {
            let clip = Clip::read(Path::new(f)).map_err(|e| e.to_string())?;
            for t in &clip.header.tuning {
                println!("{f}: from tick {}: {:?}", t.from_tick, t.values);
            }
        }
        return Ok(());
    }
    if std::env::var("PM_RESPAWNS").is_ok() {
        // Where both tees are a few frames after each of our respawns (the arena's spawn points).
        let mut files: Vec<PathBuf> = std::fs::read_dir(&clips_dir)
            .map_err(|e| e.to_string())?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "clip"))
            .collect();
        files.sort();
        for f in &files {
            let clip = Clip::read(f).map_err(|e| e.to_string())?;
            let own = clip.header.own_id;
            for (i, fr) in clip.frames.iter().enumerate() {
                if fr
                    .events
                    .iter()
                    .any(|e| matches!(e, ddai_clip::format::ClipEvent::Respawn { .. }))
                {
                    for g in clip.frames.iter().skip(i).take(3) {
                        println!(
                            "{} tick {}: {}",
                            f.display(),
                            g.tick,
                            g.tees
                                .iter()
                                .map(|t| format!(
                                    "id {}{} ({},{}) frozen {}",
                                    t.id,
                                    if t.id == own { "*" } else { "" },
                                    t.ch.x,
                                    t.ch.y,
                                    t.frozen
                                ))
                                .collect::<Vec<_>>()
                                .join(" | ")
                        );
                    }
                }
            }
        }
        return Ok(());
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(&clips_dir)
        .map_err(|e| format!("{}: {e}", clips_dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "clip"))
        .filter(|p| only.as_ref().is_none_or(|o| p.to_string_lossy().contains(o.as_str())))
        .collect();
    files.sort();
    let data: Vec<ClipData> = files
        .iter()
        .map(|f| load_clip(f, &maps, &offsets))
        .collect::<Result<_, _>>()?;
    for cd in &data {
        println!(
            "{}: own {} opp {} freeze tick {}; starts {:?}; tees in snapshot at the starts {:?}; his inferred inputs {} ticks",
            cd.name,
            cd.own,
            cd.opp,
            cd.freeze_tick,
            cd.starts.iter().map(|s| (s.0, s.1)).collect::<Vec<_>>(),
            cd.start_tees,
            cd.his.len()
        );
        for (off, tick, w) in &cd.starts {
            let fr = |id: i32| w.characters[id as usize].as_ref().is_some_and(|c| c.freeze_time > 0);
            if fr(cd.own) || fr(cd.opp) {
                println!("  start -{off} (tick {tick}): a tee is frozen there, skipped");
            }
        }
    }
    if do_sanity {
        println!(
            "\n## sanity: both tees open loop (our sent inputs, his inferred inputs) from each start; position error px (us / him) after k ticks; out flags\n"
        );
        for cd in &data {
            for (off, tick, w) in &cd.starts {
                let (s, fw, fh) = sanity(cd, w);
                println!(
                    "{} start -{off} (tick {tick}): open-loop replay: we out at +{} (real +{}), he out at +{}; {}",
                    cd.name,
                    fw.map_or("-".to_string(), |k| k.to_string()),
                    cd.freeze_tick - tick,
                    fh.map_or("-".to_string(), |k| k.to_string()),
                    s.iter()
                        .map(|(k, a, b, oa, ob)| format!(
                            "k={k}: {a:.1}/{b:.1}{}{}",
                            if *oa { " WE-OUT" } else { "" },
                            if *ob { " HE-OUT" } else { "" }
                        ))
                        .collect::<Vec<_>>()
                        .join("; ")
                );
            }
        }
    }
    if cells.is_empty() || seeds == 0 {
        return Ok(());
    }
    // Jobs: (clip, start, cell, seed).
    let mut jobs = Vec::new();
    for (ci, cd) in data.iter().enumerate() {
        for (si, (_, _, w)) in cd.starts.iter().enumerate() {
            let fr = |id: i32| w.characters[id as usize].as_ref().is_some_and(|c| c.freeze_time > 0);
            if fr(cd.own) || fr(cd.opp) {
                continue;
            }
            for (ki, _) in cells.iter().enumerate() {
                for s in 0..seeds {
                    jobs.push((ci, si, ki, 19_001 + s));
                }
            }
        }
    }
    eprintln!("{} runs on {threads} threads", jobs.len());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .stack_size(64 << 20)
        .build()
        .map_err(|e| e.to_string())?;
    let t_start = std::time::Instant::now();
    let results: Vec<Result<RunOut, String>> = pool.install(|| {
        jobs.par_iter()
            .map(|&(ci, si, ki, seed)| play(&data[ci], &data[ci].starts[si].2, &cells[ki], seed, horizon))
            .collect()
    });
    eprintln!("done in {:.0} s", t_start.elapsed().as_secs_f64());
    let mut jf = match &jsonl {
        Some(p) => Some(std::fs::File::create(p).map_err(|e| e.to_string())?),
        None => None,
    };
    // Aggregate: (clip, offset, cell) -> counts.
    #[derive(Default, Clone, Copy)]
    struct Agg {
        n: u32,
        we: u32,
        he: u32,
        both: u32,
        none: u32,
        hooked: u32,
        hammered: u32,
        we_at_sum: i64,
    }
    let mut agg: BTreeMap<(usize, i32, usize), Agg> = BTreeMap::new();
    let mut by_cell: BTreeMap<usize, Agg> = BTreeMap::new();
    for (&(ci, si, ki, seed), r) in jobs.iter().zip(&results) {
        let r = r.as_ref().map_err(|e| e.clone())?;
        let off = data[ci].starts[si].0;
        for a in [agg.entry((ci, off, ki)).or_default(), by_cell.entry(ki).or_default()] {
            a.n += 1;
            match r.first {
                "we" => {
                    a.we += 1;
                    a.we_at_sum += i64::from(r.at);
                }
                "he" => a.he += 1,
                "both" => a.both += 1,
                _ => a.none += 1,
            }
            if r.hooked {
                a.hooked += 1;
            }
            if r.hammered {
                a.hammered += 1;
            }
        }
        if let Some(f) = jf.as_mut() {
            let line = json!({"clip": data[ci].name, "offset": off, "cell": cells[ki].name, "seed": seed, "first": r.first, "at": r.at,
                "hooked": r.hooked, "hammered": r.hammered, "died": r.died});
            writeln!(f, "{line}").map_err(|e| e.to_string())?;
        }
    }
    println!(
        "\n## per clip and start: we froze first / he froze first / both / none (n seeds); of our freezes: on his hook, hammered <= 20 ticks\n"
    );
    println!("| clip | start | cell | we | he | both | none | n | we% | hooked | hammered | mean tick of our freeze |");
    println!("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for (&(ci, off, ki), a) in &agg {
        println!(
            "| {} | -{off} | {} | {} | {} | {} | {} | {} | {:.0} | {} | {} | {:.1} |",
            data[ci].name.trim_end_matches(".clip"),
            cells[ki].name,
            a.we,
            a.he,
            a.both,
            a.none,
            a.n,
            100.0 * f64::from(a.we + a.both) / f64::from(a.n.max(1)),
            a.hooked,
            a.hammered,
            if a.we > 0 {
                a.we_at_sum as f64 / f64::from(a.we)
            } else {
                f64::NAN
            }
        );
    }
    println!("\n## pooled by cell (all clips and starts)\n");
    println!("| cell | we | he | both | none | n | we or both % |");
    println!("|---|---:|---:|---:|---:|---:|---:|");
    for (&ki, a) in &by_cell {
        println!(
            "| {} | {} | {} | {} | {} | {} | {:.1} |",
            cells[ki].name,
            a.we,
            a.he,
            a.both,
            a.none,
            a.n,
            100.0 * f64::from(a.we + a.both) / f64::from(a.n.max(1))
        );
    }
    Ok(())
}
