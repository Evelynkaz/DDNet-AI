//! The navigation grids built from the f64 TS world and from the live `World<f32>` must be the same:
//! `NavGrid` and the dead zone are functions of the collision only, so any difference would be a
//! difference in what the two `PlanCollision` backends report for a tile centre.
#![cfg(feature = "ts-parity")]

use std::sync::Arc;

use ddai_nav::grid::NavGrid;
use ddai_nav::route::{dead_zone, spawn_tiles};
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;

#[test]
#[ignore = "needs DDAI_NAV_MAP (a map file)"]
fn the_grids_of_both_worlds_agree() {
    let path = std::env::var("DDAI_NAV_MAP").expect("set DDAI_NAV_MAP");
    let bytes = std::fs::read(&path).expect("map");
    let ts = ddai_tsworld::load_map_bytes(&bytes).expect("ts map");
    let map = Arc::new(ddai_map::load_map(&bytes).expect("map").data);
    let live = PhysicsWorld::new(Arc::clone(&map), 1);
    let (a, b) = (NavGrid::new(&ts.collision), NavGrid::new(live.collision()));
    let spawns = spawn_tiles(&map);
    let report = |name: &str, x: &[u8], y: &[u8]| {
        let diff: Vec<usize> = (0..x.len()).filter(|&i| x[i] != y[i]).collect();
        println!("{name}: {} differences", diff.len());
        for &i in diff.iter().take(5) {
            println!(
                "   tile ({}, {}): ts {} live {}",
                i as i32 % a.width,
                i as i32 / a.width,
                x[i],
                y[i]
            );
        }
        diff.len()
    };
    let mut bad = 0;
    // The TS collision only looks at the game layer for freeze; the live physics (like DDNet) also
    // reads the front layer. A tile that is free for TS but not live, with something in the front
    // layer, is that known difference (BlmapChill has 161 of them); anything else is a bug.
    // Heart pickups (task 4.2, review F2): the live side treats the 3x3 tiles around one as freeze, the TS
    // world knows no pickups.
    let hearts = ddai_physics::map::pickup_freeze_mask(&map);
    let front_only = |i: usize| {
        a.free[i] == 1 && b.free[i] == 0 && (hearts[i] || map.front.as_ref().is_some_and(|f| f[i].index != 0))
    };
    let unexplained: Vec<usize> = (0..a.free.len())
        .filter(|&i| a.free[i] != b.free[i] && !front_only(i))
        .collect();
    let explained = (0..a.free.len())
        .filter(|&i| a.free[i] != b.free[i] && front_only(i))
        .count();
    println!(
        "free: {} differences explained by the front layer or a heart pickup, {} unexplained",
        explained,
        unexplained.len()
    );
    bad += unexplained.len();
    bad += report("hookable", &a.hookable, &b.hookable);
    bad += report("solid", &a.solid, &b.solid);
    bad += report("death", &a.death, &b.death);
    bad += report("unfreeze", &a.unfreeze, &b.unfreeze);
    if explained == 0 {
        bad += report("danger", &a.danger, &b.danger);
    }
    let diff_i32 = |name: &str, x: &[i32], y: &[i32]| {
        let n = (0..x.len()).filter(|&i| x[i] != y[i]).count();
        println!("{name}: {n} differences (of {})", x.len());
        if let Some(i) = (0..x.len()).find(|&i| x[i] != y[i]) {
            println!("   first at index {i}: ts {} live {}", x[i], y[i]);
        }
        n
    };
    bad += diff_i32("first_solid", &a.first_solid, &b.first_solid);
    bad += diff_i32("tele_out", &a.tele_out, &b.tele_out);
    let (da, db) = (dead_zone(&a, &spawns), dead_zone(&b, &spawns));
    if explained == 0 {
        bad += report("dead zone", &da, &db);
    } else {
        println!(
            "dead zone: {} differences (front-layer tiles differ)",
            (0..da.len()).filter(|&i| da[i] != db[i]).count()
        );
    }
    assert_eq!(bad, 0);
}

#[test]
#[ignore = "needs DDAI_NAV_MAP (a map file)"]
fn routes_with_a_respawn_agree_between_the_worlds() {
    use ddai_nav::route::{RouteOpts, Router};
    let path = std::env::var("DDAI_NAV_MAP").expect("set DDAI_NAV_MAP");
    if !path.contains("Copy Love Box") {
        println!("{path}: the fixed query below is for Copy Love Box; skipped");
        return;
    }
    let bytes = std::fs::read(&path).expect("map");
    let ts = ddai_tsworld::load_map_bytes(&bytes).expect("ts map");
    let map = Arc::new(ddai_map::load_map(&bytes).expect("map").data);
    let live = PhysicsWorld::new(Arc::clone(&map), 1);
    let spawns = spawn_tiles(&map);
    let mut ra = Router::new(&ts.collision, &spawns);
    let mut rb = Router::new(live.collision(), &spawns);
    let opts = RouteOpts {
        near_tiles: 2,
        allow_kill: true,
        through_freeze: true,
        ..RouteOpts::default()
    };
    let from = (3952.0, 4912.0);
    let to = (121.0 * 32.0 + 16.0, 22.0 * 32.0 + 16.0);
    let a = ra.find_route(from, to, &opts);
    let b = rb.find_route(from, to, &opts);
    println!("ts: {:?}", a.as_ref().map(|r| r.steps.len()));
    println!("live: {:?}", b.as_ref().map(|r| r.steps.len()));
    assert_eq!(a.map(|r| r.steps.len()), b.map(|r| r.steps.len()));
}
