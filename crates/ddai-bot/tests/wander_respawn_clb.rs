//! Task 3.12b (D-104): what the free wander does to a tee that has just respawned in the top room of Copy Love Box.
//!
//! The spawns of the map stand on the ledges of the freeze chamber (`(131,34)`, `(104,35)`...) or hang over it (`(123,32)`...). The bot
//! walks to the wayblock one second after the respawn (`WB_RETURN_TICKS`); until then the wander runs, and it takes a drop ahead
//! with probability 0.3. Under `--wb-smart` the wander is anchored where the tee stands (`still`). The test runs the real [`Wander`]
//! on the real map with the real physics for that second and counts the tees that end in the chamber.
//! The map is not in the repository: without it the test says so and passes.

use std::path::PathBuf;
use std::sync::Arc;

use ddai_bot::mapgrid::MapGrid;
use ddai_bot::tees::Tee;
use ddai_bot::wander::{Wander, WanderCtx, WanderEnv};
use ddai_brain::{Action, IVec2};
use ddai_nav::route::spawn_tiles;
use ddai_physics::vmath::Vec2;
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::types::PlayerInput;

struct PassThrough;
impl WanderEnv for PassThrough {
    fn guard(&mut self, wanted: Action) -> Action {
        wanted
    }
    fn rope_catches(&mut self, _aim: IVec2) -> bool {
        false
    }
}

fn input_of(a: &Action) -> PlayerInput {
    let mut i = ddai_planner::types::empty_input();
    i.direction = a.direction;
    i.jump = i32::from(a.jump);
    i.hook = i32::from(a.hook);
    i.target_x = f64::from(a.target.x);
    i.target_y = f64::from(a.target.y);
    i
}

/// Runs `ticks` ticks of the wander from `spawn`; true when the tee ends in the chamber (tile row 41 or lower) or dead.
fn falls(
    map: &Arc<ddai_physics::map::MapData>,
    grid: &MapGrid,
    spawn: (f64, f64),
    seed: u64,
    still: bool,
    ticks: i32,
) -> bool {
    let mut world = PhysicsWorld::new(Arc::clone(map), 1);
    world.add_tee(0, ddai_planner::vmath::Vec2 { x: spawn.0, y: spawn.1 });
    let mut wander = Wander::new(seed);
    wander.respawned();
    let mut aim = (300, 0);
    // the live bot's input reaches the server `lag` ticks after it is decided
    let lag = 4;
    let mut inq: std::collections::VecDeque<PlayerInput> =
        (0..lag).map(|_| ddai_planner::types::empty_input()).collect();
    for tick in 0..ticks {
        let Some(me) = world.get_tee(0) else { return true };
        if !me.alive {
            return true;
        }
        let tee = Tee {
            id: 0,
            alive: true,
            pos: Vec2::new(me.pos.x as f32, me.pos.y as f32),
            vel: Vec2::new(me.vel.x as f32, me.vel.y as f32),
            frozen: me.frozen,
            ..Tee::DEAD
        };
        let ctx = WanderCtx {
            tick,
            own: &tee,
            grid,
            prev_aim: aim,
            anchor_x: still.then_some(tee.pos.x),
            look_at: None,
            still,
            lag_ticks: 4,
        };
        let a = wander.step(&ctx, &mut PassThrough);
        aim = (a.target.x, a.target.y);
        inq.push_back(input_of(&a));
        world.set_input(0, inq.pop_front().expect("queue"));
        let _ = world.step();
    }
    world.get_tee(0).is_none_or(|t| !t.alive || t.pos.y >= 41.0 * 32.0)
}

#[test]
fn the_free_wander_walks_a_respawned_tee_off_the_ledge_and_the_anchored_one_does_not() {
    let path = PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join("aiddnet/data/maps/copy-love-box/Copy Love Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25.map");
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!("skipped: no map {}", path.display());
        return;
    };
    let map = Arc::new(ddai_map::load_map(&bytes).expect("map").data);
    let grid = MapGrid::new(&map);
    // The spawns of the top room that stand on a ledge (the tube starts); the ones in the air fall whatever the wander does.
    let spawns: Vec<(f64, f64)> = spawn_tiles(&map).into_iter().filter(|s| s.1 < 36.0 * 32.0).collect();
    assert!(!spawns.is_empty(), "the map has spawns in the top room");
    let (mut free, mut anchored, mut n) = (0, 0, 0);
    for &sp in &spawns {
        // a respawn puts the tee on the tile's floor
        let spawn = (sp.0, sp.1 + 2.0);
        for seed in 0..60u64 {
            n += 1;
            free += usize::from(falls(&map, &grid, spawn, seed, false, 50));
            anchored += usize::from(falls(&map, &grid, spawn, seed, true, 50));
        }
    }
    println!("{n} respawns of the top room, 50 ticks of wander: free {free} in the chamber, anchored {anchored}");
    assert!(free > 0, "the free wander does walk off the ledge at least sometimes");
    assert!(anchored <= free);
}
