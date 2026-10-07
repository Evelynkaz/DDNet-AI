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

// Reviewer scratch: the fire counter of a pre-input at the first rolled tick.
fn msg(fire: i32) -> PlayerInput {
    PlayerInput {
        direction: 0,
        target_x: 100,
        target_y: -1,
        player_flags: 1,
        fire,
        ..Default::default()
    }
}

fn attack_tick_after(msgs: &[(i32, i32)]) -> i32 {
    let chars = [tee(0, 1000), tee(1, 1300)];
    let mut live = LiveWorld::new(room(), 0, 1);
    live.on_snapshot(SnapshotInput::new(500, &chars, DEFAULT_TUNE_PARAMS));
    let mut obs = live.build_observation(&live.base_world().clone(), None);
    let keep = [true; MAX_CLIENTS];
    live.set_preinput(true);
    for &(t, f) in msgs {
        live.on_pre_input(1, t, msg(f));
    }
    let w = live.predict_local_observation(504, &[], &keep, Some(1), &mut obs);
    w.characters[1].as_ref().unwrap().attack_tick
}

#[test]
fn a_pre_input_press_never_changes_the_predicted_swing_because_fire_is_not_replayed() {
    // Review round 2, F6/F9: the server fires when an input ARRIVES, so a press may already be in the snapshot; fire is taken from the assumed
    // input only. Presses at the first and at a later rolled tick, with and without an earlier message, give the prediction of no message at all.
    let none = attack_tick_after(&[]);
    for msgs in [
        &[(500, 4), (502, 5)][..],
        &[(500, 4), (501, 5)][..],
        &[(501, 5)][..],
        &[(501, 5), (503, 6)][..],
    ] {
        assert_eq!(attack_tick_after(msgs), none, "{msgs:?}");
    }
}

#[test]
fn known_ahead_counts_each_snapshot_once() {
    let chars = [tee(0, 1000), tee(1, 1300), tee(2, 1500)];
    let mut live = LiveWorld::new(room(), 0, 1);
    for tick in 500..510 {
        live.on_snapshot(SnapshotInput::new(tick, &chars, DEFAULT_TUNE_PARAMS));
        live.on_pre_input(1, tick + 2, msg(0));
        live.on_pre_input(2, tick + 2, msg(0));
    }
    // Owners 1 and 2 have messages from the second snapshot on: 9 snapshots x 2 owners, each counted once.
    let sum: u64 = live.pre_inputs().counts().known_ahead.iter().sum();
    assert_eq!(sum, 18);
}

#[test]
fn a_stale_hook_message_is_not_trusted_against_a_snapshot_that_shows_the_hook_idle() {
    // Review round 2, F7: the release message was lost; the old "hook down" would start a phantom hook over a correct hold.
    let chars = [tee(0, 1000), tee(1, 1300)];
    let mut live = LiveWorld::new(room(), 0, 1);
    live.on_snapshot(SnapshotInput::new(500, &chars, DEFAULT_TUNE_PARAMS));
    let mut obs = live.build_observation(&live.base_world().clone(), None);
    let keep = [true; MAX_CLIENTS];
    let hook_state = |live: &mut LiveWorld, obs: &mut _| {
        live.predict_local_observation(504, &[], &keep, Some(1), obs)
            .cores
            .get(1)
            .unwrap()
            .hook_state
    };
    live.set_preinput(true);
    let held = hook_state(&mut live, &mut obs);
    live.on_pre_input(1, 495, PlayerInput { hook: 1, ..msg(0) });
    live.on_pre_input(1, 503, PlayerInput { hook: 1, ..msg(0) });
    assert_eq!(
        hook_state(&mut live, &mut obs),
        held,
        "the snapshot shows the hook idle: the stale state is distrusted"
    );
    assert_eq!(live.pre_inputs().counts().distrusted, 1);
}
