//! Task 4.2 acceptance criterion 3: the Rust navigator's arrival rate against the TS navigator's on
//! the same start/goal pairs.
//!
//! **Method.** `tools/ts-trace/gen-nav-dump.mjs --section arrival` runs the real TS `Navigator` in
//! the real TS `SimWorld` (single bot, no opponents) on pairs the route graph connects (a pair no search
//! can connect is not a navigation test) and records the outcome of every run. This test runs the same
//! pairs with `ddai-nav`'s `Navigator` on `ddai_physics::World<f32>` (the live physics) under the same
//! rules ([`ddai_nav::harness::run_goto`]): at most 6000 ticks, arrival means the navigator ended
//! `arrived` with the tee within 64 px of the goal tile's centre, a `Cl_Kill` respawns at the next spawn
//! tile. The comparison is therefore of the *systems*, each in its own world; the f32/f64 physics only
//! differ in a noise tail, and a different decision — which is what is being measured — shows up as a
//! different outcome. Reported per mode: arrival rate with a Wilson 95% interval, the paired outcomes
//! (both / only TS / only Rust / neither), time to arrive, freezes and kills.
//!
//! ```text
//! node tools/ts-trace/gen-nav-dump.mjs --map "<map>" --seed 7 --section arrival --pairs 200 --out ~/aiddnet/data/traces/nav/clb-arrival.jsonl
//! DDAI_ARRIVAL_DUMP=~/aiddnet/data/traces/nav/clb-arrival.jsonl \
//!   cargo test -p ddai-nav --release --test arrival_vs_ts -- --ignored --nocapture
//! ```

use ddai_nav::harness::{FollowSpec, GotoSpec, run_follow, run_goto, target_jumps, wilson};
use ddai_nav::route::{Router, spawn_tiles};
use ddai_nav::wayblock::wayblock_for;
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;
use serde::Deserialize;
use std::sync::Arc;

/// The map for the `World<f32>` side. The TS world only reads the game layer for freeze (the front
/// layer is ignored), so on a map that keeps freeze tiles in the front layer the TS navigator walks a
/// world with fewer obstacles than the live physics has. `DDAI_ARRIVAL_NOFRONT=1` strips the front
/// layer from the live side too, so both systems face the same obstacles (the f32/f64 noise aside).
fn load_live_map(bytes: &[u8]) -> ddai_physics::map::MapData {
    let mut data = ddai_map::load_map(bytes).expect("map").data;
    if std::env::var_os("DDAI_ARRIVAL_NOFRONT").is_some() {
        data.front = None;
    }
    // Pickup entities (armor, heart, weapons: game-layer indices 197..=210). Neither the TS world nor
    // the navigator knows pickups; in DDNet a heart **freezes** whoever touches it, so on a map with
    // hearts (ChillBlock5) the live physics freezes tees where the TS world cannot.
    // `DDAI_ARRIVAL_NOPICKUPS=1` removes them from the live side.
    if std::env::var_os("DDAI_ARRIVAL_NOPICKUPS").is_some() {
        let strip = |layer: &mut Vec<ddai_physics::map::Tile>| {
            for t in layer.iter_mut().filter(|t| (197..=210).contains(&t.index)) {
                t.index = 0;
            }
        };
        strip(&mut data.game);
        if let Some(front) = data.front.as_mut() {
            strip(front);
        }
    }
    data
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum Line {
    Meta {
        #[serde(rename = "mapPath")]
        map_path: String,
        #[serde(rename = "mapSha256")]
        map_sha256: String,
    },
    Follow {
        from: (i32, i32),
        path: Vec<(f64, f64)>,
        #[serde(rename = "dwellAt")]
        dwell_at: Vec<usize>,
        speed: f64,
        dwell: i64,
        #[serde(rename = "throughFreeze")]
        through_freeze: bool,
        #[serde(rename = "maxTicks")]
        max_ticks: i64,
        arrived: bool,
        ticks: i64,
        kills: usize,
        freezes: u32,
    },
    Arrival {
        cat: String,
        from: (i32, i32),
        to: (i32, i32),
        #[serde(rename = "throughFreeze")]
        through_freeze: bool,
        #[serde(rename = "maxTicks")]
        max_ticks: i64,
        phase: String,
        ticks: i64,
        kills: usize,
        freezes: u32,
        dist: f64,
    },
}

#[derive(Default)]
struct Tally {
    n: usize,
    ts: usize,
    rust: usize,
    both: usize,
    only_ts: usize,
    only_rust: usize,
    ts_ticks: Vec<i64>,
    rust_ticks: Vec<i64>,
    ts_freezes: u32,
    rust_freezes: u32,
    ts_kills: usize,
    rust_kills: usize,
}

fn median(v: &mut [i64]) -> i64 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[v.len() / 2]
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
#[ignore = "needs a dump from tools/ts-trace/gen-nav-dump.mjs --section arrival (DDAI_ARRIVAL_DUMP)"]
fn the_rust_navigator_arrives_at_least_as_often_as_the_ts_one() {
    let path = std::env::var("DDAI_ARRIVAL_DUMP").expect("set DDAI_ARRIVAL_DUMP");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let Line::Meta { map_path, map_sha256 } = serde_json::from_str(lines.next().expect("empty")).expect("meta") else {
        panic!("first line must be meta")
    };
    let bytes = std::fs::read(&map_path).unwrap();
    {
        use sha2::{Digest, Sha256};
        assert_eq!(
            hex(&Sha256::digest(&bytes)),
            map_sha256,
            "the map changed since the dump"
        );
    }
    let map = Arc::new(load_live_map(&bytes));
    let spawns = spawn_tiles(&map);
    let template = PhysicsWorld::new(Arc::clone(&map), 1);
    let mut router = Router::new(template.collision(), &spawns);
    let wb = wayblock_for("Copy Love Box", Some(template.collision()));
    let crossings = wb.as_ref().map(|d| d.crossings.clone()).unwrap_or_default();
    let limit_override: Option<usize> = std::env::var("DDAI_ARRIVAL_LIMIT").ok().and_then(|v| v.parse().ok());

    let mut tallies: std::collections::BTreeMap<(bool, String), Tally> = Default::default();
    let mut count = 0usize;
    for line in lines {
        let Line::Arrival {
            cat,
            from,
            to,
            through_freeze,
            max_ticks,
            phase,
            ticks,
            kills,
            freezes,
            dist,
        } = serde_json::from_str(line).expect("line")
        else {
            continue;
        };
        count += 1;
        if limit_override.is_some_and(|l| count > l) {
            break;
        }
        let ts_ok = phase == "arrived" && dist <= 64.0;
        let mut world = PhysicsWorld::new(Arc::clone(&map), 1);
        let res = run_goto(
            &mut world,
            &mut router,
            &spawns,
            &GotoSpec {
                start: from,
                goal: to,
                through_freeze,
                crossings: if through_freeze { crossings.clone() } else { Vec::new() },
                max_ticks,
                improved: std::env::var_os("DDAI_NAV_EXACT").is_none(),
                unstick: true,
            },
        );
        let rust_ok = res.arrived();
        if ts_ok && !rust_ok && std::env::var_os("DDAI_ARRIVAL_VERBOSE").is_some() {
            eprintln!(
                "ONLY-TS {from:?}->{to:?} freeze {through_freeze}: Rust {} after {} ticks, dist {:.0}px, kills {}; {:?}; last notes {:?}",
                res.phase.name(),
                res.ticks,
                res.dist_px,
                res.kills.len(),
                res.outcome,
                res.notes.iter().rev().take(3).collect::<Vec<_>>()
            );
        }
        if !through_freeze && res.freezes > freezes && std::env::var_os("DDAI_ARRIVAL_FREEZES").is_some() {
            eprintln!(
                "FROZE {from:?}->{to:?}: Rust {} freezes (TS {freezes}) first at {:?}, {} after {} ticks; notes {:?}",
                res.freezes,
                res.first_freeze,
                res.phase.name(),
                res.ticks,
                res.notes.iter().rev().take(4).collect::<Vec<_>>()
            );
        }
        for key in [(through_freeze, cat.clone()), (through_freeze, "all".to_string())] {
            let t = tallies.entry(key).or_default();
            t.n += 1;
            t.ts += usize::from(ts_ok);
            t.rust += usize::from(rust_ok);
            t.both += usize::from(ts_ok && rust_ok);
            t.only_ts += usize::from(ts_ok && !rust_ok);
            t.only_rust += usize::from(!ts_ok && rust_ok);
            if ts_ok {
                t.ts_ticks.push(ticks);
            }
            if rust_ok {
                t.rust_ticks.push(res.ticks);
            }
            t.ts_freezes += freezes;
            t.rust_freezes += res.freezes;
            t.ts_kills += kills;
            t.rust_kills += res.kills.len();
        }
    }
    println!("{map_path}");
    println!(
        "mode            cat      n   TS arrived (95% CI)       Rust arrived (95% CI)     both onlyTS onlyRust | median ticks TS/Rust | freezes TS/Rust | kills TS/Rust"
    );
    let mut bad = Vec::new();
    for ((tf, cat), t) in &mut tallies {
        let (tl, th) = wilson(t.ts, t.n);
        let (rl, rh) = wilson(t.rust, t.n);
        println!(
            "freeze {:5} {cat:>8} {:4}   {:3} {:5.1}% [{:4.1},{:5.1}]   {:3} {:5.1}% [{:4.1},{:5.1}]   {:3}  {:3}  {:3} | {:5}/{:5} | {:4}/{:4} | {:3}/{:3}",
            if *tf { "on" } else { "off" },
            t.n,
            t.ts,
            100.0 * t.ts as f64 / t.n as f64,
            100.0 * tl,
            100.0 * th,
            t.rust,
            100.0 * t.rust as f64 / t.n as f64,
            100.0 * rl,
            100.0 * rh,
            t.both,
            t.only_ts,
            t.only_rust,
            median(&mut t.ts_ticks),
            median(&mut t.rust_ticks),
            t.ts_freezes,
            t.rust_freezes,
            t.ts_kills,
            t.rust_kills
        );
        // Paired data: fail only when TS is significantly better (exact one-sided McNemar, 5%).
        let p = ddai_nav::harness::mcnemar_worse_p(t.only_ts, t.only_rust);
        println!("   McNemar one-sided p(Rust worse than TS) = {p:.3}");
        if cat == "all" && p < 0.05 {
            bad.push(format!(
                "freeze {tf}: Rust {} < TS {} of {} (p = {p:.3})",
                t.rust, t.ts, t.n
            ));
        }
    }
    assert!(bad.is_empty(), "the Rust navigator arrives less often than TS: {bad:?}");
}

/// Follow mode: the real TS `steerFollow` + `Navigator` against a scripted moving target (dump section
/// `follow`), the Rust follower on the same script.
///
/// **Reported, asserted only against a gross deficit (p < 0.001).** The two systems are the same code
/// (`follow_and_goto_runs_match_ts_exactly_on_the_ts_world`: 0 differences), so what this measures is the
/// worlds: in the live physics the follower's rope can drag the scripted target (a kinematic tee), pickups
/// freeze and teleporters move it, none of which the TS world does. The numbers per map and mode, with the
/// McNemar p, are in `docs/research/nav.md` section 10.
#[test]
#[ignore = "needs a dump from tools/ts-trace/gen-nav-dump.mjs --section follow (DDAI_ARRIVAL_DUMP)"]
fn following_a_moving_target_arrives_at_least_as_often_as_in_ts() {
    let path = std::env::var("DDAI_ARRIVAL_DUMP").expect("set DDAI_ARRIVAL_DUMP");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let Line::Meta { map_path, map_sha256 } = serde_json::from_str(lines.next().expect("empty")).expect("meta") else {
        panic!("first line must be meta")
    };
    let bytes = std::fs::read(&map_path).unwrap();
    {
        use sha2::{Digest, Sha256};
        assert_eq!(
            hex(&Sha256::digest(&bytes)),
            map_sha256,
            "the map changed since the dump"
        );
    }
    let map = Arc::new(load_live_map(&bytes));
    let spawns = spawn_tiles(&map);
    let template = PhysicsWorld::new(Arc::clone(&map), 1);
    let mut router = Router::new(template.collision(), &spawns);
    let crossings = wayblock_for("Copy Love Box", Some(template.collision()))
        .map(|d| d.crossings)
        .unwrap_or_default();
    let mut tallies: std::collections::BTreeMap<bool, Tally> = Default::default();
    let mut dropped = 0usize;
    for line in lines {
        let Line::Follow {
            from,
            path,
            dwell_at,
            speed,
            dwell,
            through_freeze,
            max_ticks,
            arrived,
            ticks,
            kills,
            freezes,
        } = serde_json::from_str(line).expect("line")
        else {
            continue;
        };
        let spec = FollowSpec {
            start: from,
            path,
            dwell_at,
            speed,
            dwell,
            through_freeze,
            crossings: if through_freeze { crossings.clone() } else { Vec::new() },
            max_ticks,
            improved: std::env::var_os("DDAI_NAV_EXACT").is_none(),
            unstick: true,
        };
        if target_jumps(&mut PhysicsWorld::new(Arc::clone(&map), 1), &spec) {
            dropped += 1;
            continue;
        }
        let mut world = PhysicsWorld::new(Arc::clone(&map), 1);
        let res = run_follow(&mut world, &mut router, &spawns, &spec);
        if res.freezes > 0 && std::env::var_os("DDAI_ARRIVAL_FREEZES").is_some() {
            eprintln!(
                "FOLLOW FROZE first at {:?} (tile {:?})",
                res.first_freeze,
                res.first_freeze.map(|f| ((f.0 / 32.0) as i32, (f.1 / 32.0) as i32))
            );
        }
        if arrived && !res.arrived && std::env::var_os("DDAI_ARRIVAL_VERBOSE").is_some() {
            eprintln!(
                "FOLLOW ONLY-TS from {from:?} freeze {through_freeze}: Rust ended {:?} after {} ticks, dist {:.0}px, kills {}, freezes {}; TS {ticks} ticks",
                res.ended, res.ticks, res.dist_px, res.kills, res.freezes
            );
        }
        let t = tallies.entry(through_freeze).or_default();
        t.n += 1;
        t.ts += usize::from(arrived);
        t.rust += usize::from(res.arrived);
        t.both += usize::from(arrived && res.arrived);
        t.only_ts += usize::from(arrived && !res.arrived);
        t.only_rust += usize::from(!arrived && res.arrived);
        if arrived {
            t.ts_ticks.push(ticks);
        }
        if res.arrived {
            t.rust_ticks.push(res.ticks);
        }
        t.ts_freezes += freezes;
        t.rust_freezes += res.freezes;
        t.ts_kills += kills;
        t.rust_kills += res.kills;
    }
    println!("{map_path}: follow ({dropped} pairs dropped: the scripted target alone jumps > 96 px in the live world)");
    let mut bad = Vec::new();
    for (tf, t) in &mut tallies {
        let (tl, th) = wilson(t.ts, t.n);
        let (rl, rh) = wilson(t.rust, t.n);
        println!(
            "follow freeze {:3} n={:3}  TS {:3} {:5.1}% [{:4.1},{:5.1}]  Rust {:3} {:5.1}% [{:4.1},{:5.1}]  both {} onlyTS {} onlyRust {} | median ticks {}/{} | freezes {}/{} | kills {}/{}",
            if *tf { "on" } else { "off" },
            t.n,
            t.ts,
            100.0 * t.ts as f64 / t.n as f64,
            100.0 * tl,
            100.0 * th,
            t.rust,
            100.0 * t.rust as f64 / t.n as f64,
            100.0 * rl,
            100.0 * rh,
            t.both,
            t.only_ts,
            t.only_rust,
            median(&mut t.ts_ticks),
            median(&mut t.rust_ticks),
            t.ts_freezes,
            t.rust_freezes,
            t.ts_kills,
            t.rust_kills
        );
        let p = ddai_nav::harness::mcnemar_worse_p(t.only_ts, t.only_rust);
        println!("   McNemar one-sided p(Rust worse than TS) = {p:.3}");
        if p < 0.001 {
            bad.push(format!(
                "freeze {tf}: Rust {} < TS {} of {} (p = {p:.4})",
                t.rust, t.ts, t.n
            ));
        }
    }
    assert!(bad.is_empty(), "the Rust follower arrives less often than TS: {bad:?}");
}

/// The follow harness and the `Follow` port on the **TS world**: every run must match the TS run exactly
/// (arrival, ticks, kills, freezes) — this separates "the port differs" from "the f32 world differs".
#[cfg(feature = "ts-parity")]
#[test]
#[ignore = "needs ts-parity and a dump from tools/ts-trace/gen-nav-dump.mjs --section follow (DDAI_ARRIVAL_DUMP)"]
fn follow_and_goto_runs_match_ts_exactly_on_the_ts_world() {
    let path = std::env::var("DDAI_ARRIVAL_DUMP").expect("set DDAI_ARRIVAL_DUMP");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let Line::Meta { map_path, .. } = serde_json::from_str(lines.next().expect("empty")).expect("meta") else {
        panic!("first line must be meta")
    };
    let bytes = std::fs::read(&map_path).unwrap();
    let ts = ddai_tsworld::load_map_bytes(&bytes).expect("ts map");
    let map = ddai_map::load_map(&bytes).expect("map").data;
    let spawns = spawn_tiles(&map);
    let mut router = Router::new(&ts.collision, &spawns);
    let crossings = wayblock_for("Copy Love Box", Some(&ts.collision))
        .map(|d| d.crossings)
        .unwrap_or_default();
    let new_world = || {
        let mut w = ddai_tsworld::SimWorld::new(
            ts.collision.clone(),
            ddai_tsworld::world::SimWorldOptions {
                respawn_delay_ticks: Some(0),
                infinite_ammo: Some(true),
                sv_hit: Some(true),
                all_weapons: None,
                no_weak_hook: None,
            },
        );
        let _ = &mut w;
        w
    };
    let (mut n, mut bad) = (0usize, 0usize);
    for line in lines {
        match serde_json::from_str::<Line>(line).expect("line") {
            Line::Follow {
                from,
                path,
                dwell_at,
                speed,
                dwell,
                through_freeze,
                max_ticks,
                arrived,
                ticks,
                kills,
                freezes,
            } => {
                n += 1;
                let mut world = new_world();
                let res = run_follow(
                    &mut world,
                    &mut router,
                    &spawns,
                    &FollowSpec {
                        start: from,
                        path,
                        dwell_at,
                        speed,
                        dwell,
                        through_freeze,
                        crossings: if through_freeze { crossings.clone() } else { Vec::new() },
                        max_ticks,
                        improved: false,
                        unstick: true,
                    },
                );
                if res.arrived != arrived || res.ticks != ticks || res.kills != kills || res.freezes != freezes {
                    bad += 1;
                    eprintln!(
                        "follow run {n}: Rust {} {} ticks {} kills {} freezes; TS {arrived} {ticks} ticks {kills} kills {freezes} freezes",
                        res.arrived, res.ticks, res.kills, res.freezes
                    );
                }
            }
            Line::Arrival {
                from,
                to,
                through_freeze,
                max_ticks,
                phase,
                ticks,
                kills,
                freezes,
                dist,
                ..
            } => {
                n += 1;
                let mut world = new_world();
                let res = run_goto(
                    &mut world,
                    &mut router,
                    &spawns,
                    &GotoSpec {
                        start: from,
                        goal: to,
                        through_freeze,
                        crossings: if through_freeze { crossings.clone() } else { Vec::new() },
                        max_ticks,
                        improved: false,
                        unstick: true,
                    },
                );
                let same = res.phase.name() == phase
                    && res.ticks == ticks
                    && res.kills.len() == kills
                    && res.freezes == freezes
                    && (res.dist_px - dist).abs() < 1e-9;
                if !same {
                    bad += 1;
                    eprintln!(
                        "goto run {n}: Rust {} {} ticks {} kills; TS {phase} {ticks} ticks {kills} kills",
                        res.phase.name(),
                        res.ticks,
                        res.kills.len()
                    );
                }
            }
            Line::Meta { .. } => {}
        }
    }
    println!("{path}: {n} runs on the TS world, {bad} differ from TS");
    assert_eq!(bad, 0);
}
