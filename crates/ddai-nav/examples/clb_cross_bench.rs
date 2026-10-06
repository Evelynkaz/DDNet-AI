//! Task 3.12b: the walk-to-the-hall benchmark on the real Copy Love Box map (D-104).
//!
//! `cargo run --release -p ddai-nav --example clb_cross_bench -- [--walks N] [--side left|right|both] [--profiles a,b,..]
//!    [--others on|off] [--unblock on|off] [--live-ids on|off] [--route2-after N] [--seed S] [--threads T] [--map FILE]`
//!
//! Every walk starts at a spawn of the map, goes to the first wayblock spot of the side through the navigator (route, tube
//! crossing, route 2 after failures, the unstick of the live bot) in a world with other tees ([`Script`]) and the input delay of the
//! live bot. Prints, per profile, how many walks reach the hall, the crossings that failed and the self-kills.

use std::path::PathBuf;
use std::sync::Arc;

use ddai_nav::crossbench::{OtherSpec, Rng, Script, WalkResult, WalkSpec, run_walk};
use ddai_nav::crossing::CrossSmart;
use ddai_nav::route::{Router, spawn_tiles};
use ddai_nav::wayblock::{WbSide, standable, wayblock_for};
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::vmath::Vec2;

fn others_for(
    profile: &str,
    hall: &[ddai_nav::crossing::TileBox],
    spots: &[(i32, i32)],
    start: (i32, i32),
    exit_tile: (i32, i32),
    col: &impl ddai_planner::plan_world::PlanCollision,
    rng: &mut Rng,
) -> Vec<OtherSpec> {
    let mut stand: Vec<(i32, i32)> = Vec::new();
    for b in hall {
        for y in b.y0..=b.y1 {
            for x in b.x0..=b.x1 {
                if standable(col, x, y) {
                    stand.push((x, y));
                }
            }
        }
    }
    let pick = |rng: &mut Rng| {
        let (x, y) = stand[rng.below(stand.len())];
        Vec2 {
            x: f64::from(x * 32 + 16),
            y: f64::from(y * 32 + 18),
        }
    };
    let scatter = |n: usize, script: Script, rng: &mut Rng| -> Vec<OtherSpec> {
        (0..n)
            .map(|_| OtherSpec {
                home: pick(rng),
                script,
            })
            .collect()
    };
    // the three wayblock spots of the side, held by players who hook a frozen arrival (what the clips show)
    let at_spots = |script: Script| -> Vec<OtherSpec> {
        spots
            .iter()
            .take(3)
            .map(|&(x, y)| OtherSpec {
                home: Vec2 {
                    x: f64::from(x * 32 + 16),
                    y: f64::from(y * 32 + 18),
                },
                script,
            })
            .collect()
    };
    let fling: f64 = std::env::var("BENCH_FLING")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12.0);
    let grip: f64 = std::env::var("BENCH_P")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.9);
    // players on the ledge of the tube's start (the top room), where the walk begins
    let on_ledge = |n: usize, script: Script, rng: &mut Rng| -> Vec<OtherSpec> {
        (0..n)
            .map(|_| OtherSpec {
                home: Vec2 {
                    x: f64::from((start.0 + rng.below(7) as i32 - 3) * 32 + 16),
                    y: f64::from(start.1 * 32 + 18),
                },
                script,
            })
            .collect()
    };
    // players on the hall floor under the tube's passage, where the drop lands
    let under_exit = |n: usize, script: Script, rng: &mut Rng| -> Vec<OtherSpec> {
        let floor: Vec<(i32, i32)> = stand
            .iter()
            .copied()
            .filter(|&(x, y)| (x - exit_tile.0).abs() <= 3 && y >= 78)
            .collect();
        (0..n)
            .map(|_| {
                let (x, y) = floor[rng.below(floor.len())];
                OtherSpec {
                    home: Vec2 {
                        x: f64::from(x * 32 + 16),
                        y: f64::from(y * 32 + 18),
                    },
                    script,
                }
            })
            .collect()
    };
    let hunter = Script::Hunter {
        p: grip,
        fling,
        chase: true,
    };
    let mob = Script::Hunter {
        p: grip,
        fling: fling.max(15.0),
        chase: true,
    };
    let keeper = Script::Hunter {
        p: grip,
        fling,
        chase: false,
    };
    match profile {
        "empty" => Vec::new(),
        "idle4" => scatter(4, Script::Idle, rng),
        "wander6" => scatter(6, Script::Wander, rng),
        "hunt1w5" => [scatter(1, hunter, rng), scatter(5, Script::Wander, rng)].concat(),
        "hunt3w3" => [scatter(3, hunter, rng), scatter(3, Script::Wander, rng)].concat(),
        "mob3" => [scatter(3, mob, rng), scatter(3, Script::Wander, rng)].concat(),
        "mob5" => [scatter(5, mob, rng), scatter(3, Script::Wander, rng)].concat(),
        "under2idle" => under_exit(2, Script::Idle, rng),
        "under3wander" => under_exit(3, Script::Wander, rng),
        "ledge2" => on_ledge(2, Script::Wander, rng),
        "ledge3" => on_ledge(3, Script::Wander, rng),
        "ledge2idle" => on_ledge(2, Script::Idle, rng),
        "spots3" => at_spots(keeper),
        "spots3w3" => [at_spots(keeper), scatter(3, Script::Wander, rng)].concat(),
        "brawl2w4" => [
            scatter(2, Script::Brawler { fling: 12.0 }, rng),
            scatter(4, Script::Wander, rng),
        ]
        .concat(),
        other => panic!("unknown profile {other}"),
    }
}

fn category(note: &str) -> &'static str {
    if let Some(i) = note.find("lies frozen at (") {
        let rest = &note[i + "lies frozen at (".len()..];
        let nums: Vec<i32> = rest
            .split([',', ')'])
            .take(2)
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        if nums.len() == 2 {
            let (x, y) = (nums[0], nums[1]);
            let outer = x <= 79 || x >= 155;
            return if y <= 53 {
                "chamber/tube"
            } else if y >= 95 {
                "pit below the hall"
            } else if y >= 81 {
                "lower freeze (shelf side)"
            } else if outer && (68..=80).contains(&y) && !(86..=150).contains(&x) {
                "outer freeze wall"
            } else {
                "in the hall"
            };
        }
    }
    if note.contains("ran out") {
        "ran out"
    } else if note.contains("no swing") {
        "no swing found"
    } else if note.contains("no hop") {
        "no hop found"
    } else {
        "other"
    }
}

fn main() {
    let mut walks = 40usize;
    let mut side = "both".to_string();
    let mut profiles = "empty,idle4,wander6,hunt3w3".to_string();
    let mut smart = CrossSmart::default();
    let mut route2_after = 2;
    let mut seed = 1u64;
    let mut threads = 2usize;
    let mut budget_ms = 0.0;
    let mut max_ticks = 4000i64;
    let mut trace_every = 0i64;
    let mut live_ids = false;
    let mut map_file = PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(
        "aiddnet/data/maps/cache/Copy Love Box_1134dda918002ad0c77376548c208a53b889973cee72b691d51ea030590c6e0a.map",
    );
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = |name: &str| it.next().unwrap_or_else(|| panic!("{name} needs a value"));
        match a.as_str() {
            "--walks" => walks = val("--walks").parse().expect("number"),
            "--side" => side = val("--side"),
            "--profiles" => profiles = val("--profiles"),
            "--others" => smart.others = val("--others") == "on",
            "--unblock" => smart.unblock = val("--unblock") == "on",
            "--route2-after" => route2_after = val("--route2-after").parse().expect("number"),
            "--seed" => seed = val("--seed").parse().expect("number"),
            "--threads" => threads = val("--threads").parse().expect("number"),
            "--budget-ms" => budget_ms = val("--budget-ms").parse().expect("number"),
            "--live-ids" => live_ids = val("--live-ids") == "on",
            "--trace" => trace_every = val("--trace").parse().expect("number"),
            "--max-ticks" => max_ticks = val("--max-ticks").parse().expect("number"),
            "--map" => map_file = PathBuf::from(val("--map")),
            other => panic!("unknown argument {other}"),
        }
    }
    let bytes = std::fs::read(&map_file).expect("map file");
    let map = Arc::new(ddai_map::load_map(&bytes).expect("map").data);
    let spawns = spawn_tiles(&map);
    if std::env::var_os("BENCH_SPAWNS").is_some() {
        for sp in &spawns {
            println!("spawn tile ({},{})", (sp.0 / 32.0) as i32, (sp.1 / 32.0) as i32);
        }
    }
    let template = PhysicsWorld::new(Arc::clone(&map), 1);
    let def = wayblock_for("Copy Love Box", Some(template.collision())).expect("the map has the wayblock");
    let sides: Vec<WbSide> = match side.as_str() {
        "left" => vec![WbSide::Left],
        "right" => vec![WbSide::Right],
        _ => vec![WbSide::Left, WbSide::Right],
    };
    println!(
        "walks per cell {walks}, sides {sides:?}, others-in-rollouts {}, unblock {}, route 2 after {route2_after}, seed {seed}",
        smart.others, smart.unblock
    );
    for profile in profiles.split(',') {
        let mut results: Vec<(WbSide, WalkResult)> = Vec::new();
        for &s in &sides {
            let sd = def.side(s);
            let jobs: Vec<usize> = (0..walks).collect();
            let chunks: Vec<Vec<usize>> = (0..threads)
                .map(|k| jobs.iter().copied().filter(|j| j % threads == k).collect())
                .collect();
            let out: Vec<Vec<(WbSide, WalkResult)>> = std::thread::scope(|sc| {
                let hs: Vec<_> = chunks
                    .into_iter()
                    .map(|chunk| {
                        let (map, def, spawns) = (Arc::clone(&map), def.clone(), spawns.clone());
                        let sd = sd.clone();
                        let profile = profile.to_string();
                        sc.spawn(move || {
                            let mut v = Vec::new();
                            for j in chunk {
                                let mut world = PhysicsWorld::new(Arc::clone(&map), 1);
                                let mut router = Router::new(world.collision(), &spawns);
                                let mut rng = Rng(seed.wrapping_mul(1_000_003).wrapping_add(j as u64 * 7919 + 17));
                                let others = others_for(
                                    &profile,
                                    sd.crossing.hall.as_deref().unwrap_or(&sd.zone),
                                    &sd.spots,
                                    sd.crossing.start,
                                    sd.crossing.exit_tile,
                                    world.collision(),
                                    &mut rng,
                                );
                                let spec = WalkSpec {
                                    crossings: def.crossings.clone(),
                                    goal: sd.spots[0],
                                    hall: sd.crossing.hall.clone().unwrap_or_else(|| sd.zone.clone()),
                                    toward: sd.crossing.toward,
                                    others,
                                    lag: 4,
                                    route2_after,
                                    max_ticks,
                                    seed: seed ^ (j as u64) << 8,
                                    budget_ms,
                                    smart,
                                    trace_every,
                                    live_ids,
                                };
                                let r = run_walk(&mut world, &mut router, &spawns, rng.below(spawns.len()), &spec);
                                v.push((s, r));
                            }
                            v
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().expect("worker")).collect()
            });
            results.extend(out.into_iter().flatten());
        }
        report(profile, &results);
    }
}

fn report(profile: &str, results: &[(WbSide, WalkResult)]) {
    if std::env::var_os("BENCH_SHOW_FAILS").is_some() {
        for (side, r) in results.iter().filter(|r| !r.1.arrived) {
            println!("  NOT ARRIVED ({side:?}, {} ticks, kills {}):", r.ticks, r.kills);
            for n in r.notes.iter().rev().take(6).rev() {
                println!("      {n}");
            }
        }
    }
    let n = results.len() as f64;
    let arrived = results.iter().filter(|r| r.1.arrived).count();
    let kills: u32 = results.iter().map(|r| r.1.kills).sum();
    let starts: u32 = results.iter().map(|r| r.1.cross_starts).sum();
    let fails: u32 = results.iter().map(|r| r.1.cross_fails).sum();
    let ticks: i64 = results.iter().map(|r| r.1.ticks).sum();
    let (lo, hi) = ddai_nav::harness::wilson(arrived, results.len());
    println!(
        "{profile:<10} walks {:>4}  arrived {:>3} = {:>5.1}% (95% {:>4.1}..{:>4.1})  crossings started {:>4} failed {:>4} ({:>4.1}%)  self-kills/walk {:>4.2}  mean ticks {:>6.0}",
        results.len(),
        arrived,
        100.0 * arrived as f64 / n,
        100.0 * lo,
        100.0 * hi,
        starts,
        fails,
        100.0 * f64::from(fails) / f64::from(starts.max(1)),
        f64::from(kills) / n,
        ticks as f64 / n,
    );
    let mut cats = std::collections::BTreeMap::<&str, u32>::new();
    for (_, r) in results {
        for note in &r.fail_notes {
            *cats.entry(category(note)).or_default() += 1;
        }
    }
    println!("           failed crossings by where they ended: {cats:?}");
    let mut kills = std::collections::BTreeMap::<String, u32>::new();
    for (_, r) in results {
        for (x, y, why) in &r.kill_spots {
            let c = if *y <= 53 {
                "chamber/tube"
            } else if *y >= 95 {
                "pit"
            } else if *y >= 81 {
                "y81-94 (shelf level)"
            } else if (68..=80).contains(y) {
                "hall level"
            } else {
                "other"
            };
            let _ = x;
            *kills.entry(format!("{c}/{why}")).or_default() += 1;
        }
    }
    println!("           self-kills by where and why: {kills:?}");
}
