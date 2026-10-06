//! Task 3.12b (D-104): the crossing options of `--wb-smart` on the real Copy Love Box map, in a world with other tees.
//!
//! The map is not in the repository: without it each test says so and passes. Deterministic: the crossing search has no clock
//! (`budget_ms` 0) and the scripts of the other tees are seeded.

use std::path::PathBuf;
use std::sync::Arc;

use ddai_nav::crossbench::{OtherSpec, Script, WalkResult, WalkSpec, run_walk};
use ddai_nav::crossing::CrossSmart;
use ddai_nav::route::{Router, spawn_tiles};
use ddai_nav::wayblock::{WbSide, wayblock_for};
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::vmath::Vec2;

fn clb() -> Option<Arc<ddai_physics::map::MapData>> {
    let path = PathBuf::from(std::env::var("HOME").ok()?).join(
        "aiddnet/data/maps/copy-love-box/Copy Love Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25.map",
    );
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!("skipped: no map {}", path.display());
        return None;
    };
    Some(Arc::new(ddai_map::load_map(&bytes).expect("map").data))
}

/// `walks` walks of the right side from the spawns in turn, with `others` (placed by `place(walk)`), under `smart`.
fn walks(
    map: &Arc<ddai_physics::map::MapData>,
    side: WbSide,
    n: usize,
    smart: CrossSmart,
    place: &dyn Fn(usize, &ddai_nav::wayblock::WbSideDef) -> Vec<OtherSpec>,
) -> Vec<WalkResult> {
    let spawns = spawn_tiles(map);
    let template = PhysicsWorld::new(Arc::clone(map), 1);
    let def = wayblock_for("Copy Love Box", Some(template.collision())).expect("wayblock");
    let sd = def.side(side).clone();
    (0..n)
        .map(|i| {
            let mut world = PhysicsWorld::new(Arc::clone(map), 1);
            let mut router = Router::new(world.collision(), &spawns);
            let spec = WalkSpec {
                crossings: def.crossings.clone(),
                goal: sd.spots[0],
                hall: sd.crossing.hall.clone().unwrap_or_else(|| sd.zone.clone()),
                toward: sd.crossing.toward,
                others: place(i, &sd),
                lag: 4,
                route2_after: 2,
                max_ticks: 1500,
                seed: i as u64 + 1,
                budget_ms: 0.0,
                smart,
                trace_every: 0,
                live_ids: false,
            };
            run_walk(&mut world, &mut router, &spawns, i, &spec)
        })
        .collect()
}

fn failed(r: &[WalkResult]) -> u32 {
    r.iter().map(|w| w.cross_fails).sum()
}

fn on_the_ledge(sd: &ddai_nav::wayblock::WbSideDef, dx: i32) -> OtherSpec {
    OtherSpec {
        home: Vec2 {
            x: f64::from((sd.crossing.start.0 + dx) * 32 + 16),
            y: f64::from(sd.crossing.start.1 * 32 + 18),
        },
        script: Script::Idle,
    }
}

#[test]
fn in_an_empty_world_every_walk_gets_through_with_or_without_the_options() {
    let Some(map) = clb() else { return };
    let all = CrossSmart {
        others: true,
        unblock: true,
    };
    for side in [WbSide::Left, WbSide::Right] {
        for smart in [CrossSmart::default(), all] {
            let r = walks(&map, side, 4, smart, &|_, _| Vec::new());
            assert!(r.iter().all(|w| w.arrived && w.kills == 0), "{side:?} {smart:?}: {r:?}");
            assert_eq!(failed(&r), 0, "{side:?} {smart:?}");
        }
    }
}

#[test]
fn two_idle_tees_on_the_ledge_of_the_start_wall_the_walk_in_and_unblock_gets_past_them() {
    let Some(map) = clb() else { return };
    let place = |_: usize, sd: &ddai_nav::wayblock::WbSideDef| vec![on_the_ledge(sd, -2), on_the_ledge(sd, -1)];
    let off = walks(&map, WbSide::Right, 4, CrossSmart::default(), &place);
    let on = walks(
        &map,
        WbSide::Right,
        4,
        CrossSmart {
            unblock: true,
            ..CrossSmart::default()
        },
        &place,
    );
    println!(
        "failed crossings of 4 walks: default {}, unblock {}",
        failed(&off),
        failed(&on)
    );
    assert!(
        failed(&off) >= 4,
        "the idle tees wall the approach in: {}",
        failed(&off)
    );
    assert!(
        failed(&on) * 3 <= failed(&off),
        "unblock: {} against {}",
        failed(&on),
        failed(&off)
    );
    assert!(on.iter().all(|w| w.arrived));
}

#[test]
fn the_rollouts_with_the_other_tees_still_find_the_way_through_a_crowded_hall() {
    let Some(map) = clb() else { return };
    let place = |_: usize, sd: &ddai_nav::wayblock::WbSideDef| {
        // idle tees on the hall floor under the passage
        (0..3)
            .map(|k| OtherSpec {
                home: Vec2 {
                    x: f64::from((sd.crossing.exit_tile.0 + k - 1) * 32 + 16),
                    y: f64::from(79 * 32 + 18),
                },
                script: Script::Idle,
            })
            .collect()
    };
    let r = walks(
        &map,
        WbSide::Left,
        8,
        CrossSmart {
            others: true,
            ..CrossSmart::default()
        },
        &place,
    );
    assert!(r.iter().all(|w| w.arrived), "{r:?}");
}
