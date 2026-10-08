//! Post-mortem of the 2026-10-08 duel against a human: **the hybrid's choice on the live clip worlds**, decision by decision.
//!
//! The clip's frames are fed to a `LiveWorld` as the bot did (`ddai_clip::replay::feed`); at every frame of the asked tick windows the default
//! hybrid (as the arena builds it: deadline mode on the work clock, `--finish full` = `frozen_drag_weight` 20, mirror on, duel live context) decides
//! on the snapshot world with our real inputs in flight (`WorldView { lag_ticks: L, in_flight }`, the opponent holds his snapshot input, which is what
//! the live bot had: the target's pre-inputs almost never arrived, the window model was not loaded). One brain per seed keeps its warm start across
//! the frames of a window (it starts deciding `--warm` frames before the window). Prints one JSON line per decision: the chosen plan's source
//! (the "plan family"), the plan, the scores, the danger and the candidates by source, plus the first input.
//!
//! ```text
//! cargo run --release -p ddai-env --example pm1008_replay -- --clip <file.clip> --window <from>..<to> [--window ...] [--rate-us 2.1]
//!     [--budget-ms 4] [--seeds 4] [--lag 2] [--drag 20] [--warm 20] [--opp 0] [--live-path] [--no-mirror] [--dump] [--static-push] [--counter] [--belief P] [--protect] [--finish-push] [--no-hammer] > out.jsonl
//! ```
//! `--rate-us` is the work clock's cost of one tee-tick in microseconds (2.1 = the live-like starved search of E-034 §2.2, 1.25 = the arena default);
//! `--budget-ms 40` with `--rate-us 1.25` is an "oracle" with ten times the search.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ddai_brain::{Brain, LiveContext, ResetContext, WorldView};
use ddai_clip::format::Clip;
use ddai_clip::replay::feed;
use ddai_env::config::{DuelFixSpec, HybridSpec, PlayerSpec, builtin_brain};
use ddai_env::observe;
use ddai_physics::core::PlayerInput as Wire;
use ddai_world::{LiveWorld, player_input_from_net};

fn find_map(dir: &Path, sha: &[u8; 32]) -> Result<PathBuf, String> {
    let hex: String = sha.iter().map(|b| format!("{b:02x}")).collect();
    for e in std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let p = e.map_err(|e| e.to_string())?.path();
        if p.file_name().is_some_and(|n| n.to_string_lossy().contains(&hex)) {
            return Ok(p);
        }
    }
    Err(format!("no map with sha {hex} in {}", dir.display()))
}

fn main() -> Result<(), String> {
    let mut clip_path = PathBuf::new();
    let mut maps = PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("aiddnet/data/maps/cache");
    let mut windows: Vec<(i32, i32)> = Vec::new();
    let mut rate_us = 2.1f64;
    let mut budget_ms = 4.0f64;
    let mut seeds = 4u64;
    let mut lag = 2u32;
    let mut drag = 20.0f64;
    let mut warm = 20usize;
    let mut opp = 0i32;
    let mut live_path = false;
    let mut mirror = true;
    let mut dump = false;
    let mut fixes = DuelFixSpec::default();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--clip" => clip_path = PathBuf::from(val()?),
            "--maps" => maps = PathBuf::from(val()?),
            "--window" => {
                let v = val()?;
                let (x, y) = v.split_once("..").ok_or("--window a..b")?;
                windows.push((
                    x.parse().map_err(|e| format!("{e}"))?,
                    y.parse().map_err(|e| format!("{e}"))?,
                ));
            }
            "--rate-us" => rate_us = val()?.parse().map_err(|e| format!("{e}"))?,
            "--budget-ms" => budget_ms = val()?.parse().map_err(|e| format!("{e}"))?,
            "--seeds" => seeds = val()?.parse().map_err(|e| format!("{e}"))?,
            "--lag" => lag = val()?.parse().map_err(|e| format!("{e}"))?,
            "--drag" => drag = val()?.parse().map_err(|e| format!("{e}"))?,
            "--warm" => warm = val()?.parse().map_err(|e| format!("{e}"))?,
            "--opp" => opp = val()?.parse().map_err(|e| format!("{e}"))?,
            "--live-path" => live_path = true,
            "--no-mirror" => mirror = false,
            "--dump" => dump = true,
            "--static-push" => fixes.static_push = Some(true),
            "--counter" => fixes.counter_release = Some(true),
            "--protect" => fixes.protect_defence = Some(true),
            "--finish-push" => {
                fixes.finish_push = Some(true);
                fixes.finish_approach = Some(6);
            }
            "--no-hammer" => fixes.no_hammer_frozen = Some(true),
            "--belief" => fixes.hooked_belief = Some(val()?.parse().map_err(|e| format!("{e}"))?),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let clip = Clip::read(&clip_path).map_err(|e| format!("{}: {e}", clip_path.display()))?;
    let map_file = find_map(&maps, &clip.header.map_sha256)?;
    let bytes = std::fs::read(&map_file).map_err(|e| e.to_string())?;
    let map = Arc::new(ddai_map::load_map(&bytes).map_err(|e| format!("{e}"))?.data);
    let own = clip.header.own_id;
    let mut ours: BTreeMap<i32, Wire> = BTreeMap::new();
    for f in &clip.frames {
        for s in &f.sent {
            ours.insert(s.tick, player_input_from_net(s.input.to_net()));
        }
    }
    let ids = [own, opp];
    let name = clip_path
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().to_string());
    for &(w0, w1) in &windows {
        // Fresh brains and world per window (as after a respawn); the brains start deciding `warm` frames early.
        let mut brains: Vec<Box<dyn Brain>> = Vec::new();
        for s in 0..seeds {
            let mut spec = PlayerSpec::simple("hybrid");
            spec.mode = Some("deadline".into());
            spec.clock = Some("work".into());
            spec.budget_ms = Some(budget_ms);
            spec.step_ms = Some(rate_us / 1000.0);
            let mut h = HybridSpec::default();
            if drag > 0.0 {
                h.frozen_drag_weight = Some(drag);
            }
            if !mirror {
                h.mirror = Some(false);
            }
            if dump {
                h.debug_dump = Some(true);
            }
            if fixes != DuelFixSpec::default() {
                h.duel_fixes = Some(fixes.clone());
            }
            spec.hybrid = Some(h);
            let mut b = builtin_brain(&spec).map_err(|e| e.to_string())?;
            b.reset(&ResetContext {
                map: Arc::clone(&map),
                self_id: own,
                seed: 31_001 + s,
            });
            brains.push(b);
        }
        let mut lw = LiveWorld::new(Arc::clone(&map), own, clip.header.world_seed);
        let first_idx = clip
            .frames
            .iter()
            .position(|f| f.tick >= w0)
            .unwrap_or(clip.frames.len());
        let warm_from = first_idx.saturating_sub(warm);
        for (i, f) in clip.frames.iter().enumerate() {
            feed(&mut lw, &clip, f);
            if i < warm_from || f.tick > w1 {
                continue;
            }
            let (Some(_me), Some(_him)) = (f.tee(own), f.tee(opp)) else {
                continue;
            };
            if !f.own_alive {
                continue;
            }
            let t = f.tick;
            let in_flight: Option<Vec<Wire>> = (1..=lag as i32).map(|k| ours.get(&(t + k)).copied()).collect();
            let Some(in_flight) = in_flight else { continue };
            // `--live-path`: the bot's step 9 -- LiveWorld rolls the world to the slot the live decision aimed at (`aimed_tick - 1`) with our
            // inputs in flight, the opponent holding; the brain then decides with lag 0 (`bot.rs`, `predict_local_observation`).
            let to_tick = if f.bot.aimed_tick > t {
                f.bot.aimed_tick - 1
            } else {
                t + lag as i32
            };
            let world = if live_path {
                let fl: Vec<(i32, Wire)> = (t + 1..=to_tick)
                    .filter_map(|k| ours.get(&k).map(|w| (k, *w)))
                    .collect();
                let mut keep = [false; ddai_physics::core::MAX_CLIENTS];
                keep[opp as usize] = true;
                let mut obs = lw.build_observation(lw.base_world(), Some(opp));
                lw.predict_local_observation(to_tick, &fl, &keep, Some(opp), &mut obs)
                    .clone()
            } else {
                lw.base_world().clone()
            };
            let (view_lag, view_flight): (u32, &[Wire]) = if live_path { (0, &[]) } else { (lag, &in_flight) };
            for (si, b) in brains.iter_mut().enumerate() {
                let Some(obs) = observe::observation(&world, &map, own, &ids, Some(opp)) else {
                    continue;
                };
                b.set_live_context(&LiveContext {
                    duel: true,
                    ..LiveContext::default()
                });
                let view = WorldView {
                    world: &world,
                    self_id: own,
                    lag_ticks: view_lag,
                    in_flight: view_flight,
                };
                let a = b.decide_in(&obs, Some(&view));
                if t < w0 {
                    continue;
                }
                let tel = b.telemetry().unwrap_or_else(|| "null".into());
                println!(
                    "{{\"clip\":\"{name}\",\"tick\":{t},\"seed\":{si},\"rate_us\":{rate_us},\"budget_ms\":{budget_ms},\"lag\":{lag},\"drag\":{drag},\
\"live_path\":{live_path},\"act\":[{},{},{},{}],\"tel\":{tel}}}",
                    a.direction,
                    u8::from(a.jump),
                    u8::from(a.hook),
                    u8::from(a.fire)
                );
            }
        }
    }
    Ok(())
}
