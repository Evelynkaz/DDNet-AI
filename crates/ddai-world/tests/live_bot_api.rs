//! Task 4.1's additions to `LiveWorld` for the live bot: `predict_local` (cut far tees before
//! stepping), `fill_observation` / `predict_local_observation` (no-allocation observation), and the
//! allocation behaviour of `on_snapshot` itself in steady state.

use std::sync::Arc;

use ddai_net::generated::enums::playerflagflag;
use ddai_net::generated::objects;
use ddai_net::tuning::DEFAULT_TUNE_PARAMS;
use ddai_net::view::CharacterView;
use ddai_physics::core::{MAX_CLIENTS, PlayerInput};
use ddai_physics::map::{MapData, TILE_SOLID, Tile};
use ddai_world::{LiveWorld, SnapshotInput};

fn room() -> Arc<MapData> {
    let (w, h) = (120u32, 30u32);
    let mut game = vec![Tile::default(); (w * h) as usize];
    for x in 0..w {
        game[x as usize].index = TILE_SOLID;
        game[((h - 1) * w + x) as usize].index = TILE_SOLID;
    }
    for y in 0..h {
        game[(y * w) as usize].index = TILE_SOLID;
        game[(y * w + w - 1) as usize].index = TILE_SOLID;
    }
    Arc::new(MapData {
        width: w,
        height: h,
        game,
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    })
}

fn tee(id: i32, x: i32) -> CharacterView {
    let y = 29 * 32 - 15;
    CharacterView {
        id,
        character: objects::Character {
            tick: 0,
            x,
            y,
            vel_x: 0,
            vel_y: 0,
            angle: 0,
            direction: 0,
            jumped: 0,
            hooked_player: -1,
            hook_state: 0,
            hook_tick: 0,
            hook_x: x,
            hook_y: y,
            hook_dx: 0,
            hook_dy: 0,
            player_flags: playerflagflag::PLAYING,
            health: 10,
            armor: 0,
            ammo_count: -1,
            weapon: 0,
            emote: 0,
            attack_tick: 0,
        },
        ddnet: Some(objects::DDNetCharacter {
            flags: 0,
            freeze_end: 0,
            jumps: 2,
            tele_checkpoint: -1,
            strong_weak_id: id,
            jumped_total: -1,
            ninja_activation_tick: -1,
            freeze_start: -1,
            target_x: 0,
            target_y: 0,
            tune_zone_override: -1,
        }),
    }
}

fn run_right() -> PlayerInput {
    PlayerInput {
        direction: 1,
        target_y: -1,
        player_flags: 1,
        ..Default::default()
    }
}

#[test]
fn predict_local_drops_unkept_tees_from_the_prediction_but_not_from_the_base_world() {
    let chars = [tee(0, 1000), tee(1, 1100), tee(2, 1200), tee(3, 3000)];
    let mut live = LiveWorld::new(room(), 0, 1);
    live.on_snapshot(SnapshotInput::new(500, &chars, DEFAULT_TUNE_PARAMS));
    let mut keep = [false; MAX_CLIENTS];
    keep[1] = true;
    let p = live.predict_local(503, &[(501, run_right())], &keep);
    let present: Vec<usize> = (0..MAX_CLIENTS).filter(|&i| p.characters[i].is_some()).collect();
    assert_eq!(present, vec![0, 1], "ourselves and the kept tee only");
    assert!(p.entity_order.iter().all(|&i| i == 0 || i == 1), "{:?}", p.entity_order);
    assert_eq!(p.tick, 503);
    // The base world still has everybody, and a plain predict still steps everybody.
    assert_eq!(live.base_world().characters.iter().flatten().count(), 4);
    assert_eq!(live.predict(503, &[]).characters.iter().flatten().count(), 4);
}

#[test]
fn predict_local_gives_our_own_tee_the_same_trajectory_as_the_full_prediction_for_distant_tees() {
    let chars = [tee(0, 1000), tee(3, 3000)];
    let mut live = LiveWorld::new(room(), 0, 1);
    live.on_snapshot(SnapshotInput::new(500, &chars, DEFAULT_TUNE_PARAMS));
    let inputs: Vec<(i32, PlayerInput)> = (501..=506).map(|t| (t, run_right())).collect();
    let full = live.predict(506, &inputs).cores.get(0).unwrap().pos;
    let keep = [false; MAX_CLIENTS];
    let local = live.predict_local(506, &inputs, &keep).cores.get(0).unwrap().pos;
    assert_eq!(full, local, "a tee 2000 px away cannot have touched us");
    assert!(local.x > 1000.0, "and the in-flight input moved us");
}

#[test]
fn fill_observation_equals_build_observation_and_reuses_the_others_buffer() {
    let chars = [tee(0, 1000), tee(1, 1100), tee(2, 1200)];
    let mut live = LiveWorld::new(room(), 0, 1);
    live.on_snapshot(SnapshotInput::new(500, &chars, DEFAULT_TUNE_PARAMS));
    let world = live.base_world().clone();
    let built = live.build_observation(&world, Some(1));
    let mut obs = live.build_observation(&world, None);
    live.fill_observation(&world, Some(1), &mut obs);
    assert_eq!(obs.others, built.others);
    assert_eq!(obs.self_state, built.self_state);
    assert_eq!(
        (obs.tick, obs.target_id, obs.tuning),
        (built.tick, built.target_id, built.tuning)
    );
    assert!(Arc::ptr_eq(&obs.map, live.map()));
    let (cap, ptr) = (obs.others.capacity(), obs.others.as_ptr());
    live.fill_observation(&world, None, &mut obs);
    assert_eq!(
        (obs.others.capacity(), obs.others.as_ptr()),
        (cap, ptr),
        "no reallocation"
    );
    assert_eq!(obs.target_id, None);
}

#[test]
fn predict_local_observation_builds_the_observation_of_the_returned_world() {
    let chars = [tee(0, 1000), tee(1, 1100), tee(2, 1200)];
    let mut live = LiveWorld::new(room(), 0, 1);
    live.on_snapshot(SnapshotInput::new(500, &chars, DEFAULT_TUNE_PARAMS));
    let mut obs = live.build_observation(&live.base_world().clone(), None);
    let mut keep = [false; MAX_CLIENTS];
    keep[2] = true;
    let inputs = [(501, run_right()), (502, run_right())];
    let (world_tick, self_x) = {
        let w = live.predict_local_observation(503, &inputs, &keep, Some(2), &mut obs);
        (w.tick, w.cores.get(0).unwrap().pos.x)
    };
    assert_eq!(obs.tick, world_tick);
    assert_eq!(
        obs.self_state.pos.x, self_x,
        "the observation is of the predicted world"
    );
    assert_eq!(
        obs.others.iter().map(|o| o.id).collect::<Vec<_>>(),
        vec![2],
        "only the kept tee"
    );
    assert_eq!(obs.target_id, Some(2));
}

#[test]
fn a_steady_state_snapshot_and_prediction_allocate_nothing() {
    let chars: Vec<CharacterView> = (0..12).map(|i| tee(i, 1000 + 60 * i)).collect();
    let mut live = LiveWorld::new(room(), 0, 1);
    let mut obs = live.build_observation(&live.base_world().clone(), None);
    let mut keep = [false; MAX_CLIENTS];
    keep[1] = true;
    let inputs = [(501, run_right()), (502, run_right())];
    for tick in (500..600).step_by(2) {
        live.on_snapshot(SnapshotInput::new(tick, &chars, DEFAULT_TUNE_PARAMS));
        let _ = live.predict_local_observation(tick + 3, &inputs, &keep, Some(1), &mut obs);
    }
    let info = allocation_counter::measure(|| {
        for tick in (600..800).step_by(2) {
            live.on_snapshot(SnapshotInput::new(tick, &chars, DEFAULT_TUNE_PARAMS));
            let w = live.predict_local_observation(tick + 3, &inputs, &keep, Some(1), &mut obs);
            std::hint::black_box(w.tick);
        }
    });
    assert_eq!(info.count_total, 0, "{info:?}");
}
