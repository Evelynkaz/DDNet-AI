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

/// A tee in mid-air at `x`, as the wire shows it: `wire_tick` is the character's `m_Tick` (0 = exact at the snapshot tick, else the tick of the
/// server's last full resync, which `LiveWorld` dead-reckons forward from).
fn air_tee(id: i32, x: i32, wire_tick: i32, jumped: i32, hook_state: i32) -> CharacterView {
    let mut cv = tee(id, x);
    cv.character.tick = wire_tick;
    cv.character.y = 10 * 32;
    cv.character.hook_y = 10 * 32;
    cv.character.jumped = jumped;
    cv.character.hook_state = hook_state;
    cv
}

/// Counters after one roll over a snapshot at 500 whose owner-1 core is `owner`, with the messages `msgs` (`(tick, input)`).
fn roll_counts(owner: CharacterView, msgs: &[(i32, PlayerInput)]) -> ddai_world::preinput::PreInputCounts {
    let chars = [tee(0, 1000), owner];
    let mut live = LiveWorld::new(room(), 0, 1);
    live.on_snapshot(SnapshotInput::new(500, &chars, DEFAULT_TUNE_PARAMS));
    let mut obs = live.build_observation(&live.base_world().clone(), None);
    let keep = [true; MAX_CLIENTS];
    live.set_preinput(true);
    for &(t, m) in msgs {
        live.on_pre_input(1, t, m);
    }
    live.predict_local_observation(504, &[], &keep, Some(1), &mut obs);
    live.pre_inputs().counts()
}

#[test]
fn a_dead_reckoned_character_is_distrusted_for_a_stale_hook_or_jump_bit_like_an_exact_one_and_counted_apart() {
    // Review F11 (F10 withdrawn): the server resyncs the wire core on any difference (`character.cpp:953-968`), so a reckoned core's `jumped` and
    // `hook_state` are as reliable as an exact core's. Held jump key at 496 in the wire, released at 497 in the pre-inputs (the snapshot at 500 still
    // shows it held: the release is not what the snapshot says), a later message at 502.
    let released = PlayerInput { jump: 0, ..msg(0) };
    let pressed_again = PlayerInput { jump: 1, ..msg(0) };
    let msgs = [(497, released), (502, pressed_again)];
    let reckoned = roll_counts(air_tee(1, 1300, 496, 1, 0), &msgs);
    assert_eq!(
        (
            reckoned.checked,
            reckoned.checked_reckoned,
            reckoned.distrusted,
            reckoned.distrusted_reckoned
        ),
        (1, 1, 1, 1),
        "{reckoned:?}"
    );
    assert_eq!(reckoned.used, 0, "{reckoned:?}");
    // The same wire core exact at the snapshot tick (`m_Tick == 0`), at it, or after it: distrusted the same, counted as exact.
    for t in [0, 500, 503] {
        let exact = roll_counts(air_tee(1, 1300, t, 1, 0), &msgs);
        assert_eq!(
            (
                exact.checked,
                exact.checked_reckoned,
                exact.distrusted,
                exact.distrusted_reckoned
            ),
            (1, 0, 1, 0),
            "m_Tick {t}: {exact:?}"
        );
        assert_eq!(exact.used, 0, "{exact:?}");
    }
    // The same for the hook: the wire shows the hook key down at 496 (state FLYING = 1), the message in force at the snapshot says released.
    let hook_msgs = [
        (497, PlayerInput { hook: 0, ..msg(0) }),
        (502, PlayerInput { hook: 1, ..msg(0) }),
    ];
    let reckoned = roll_counts(air_tee(1, 1300, 496, 0, 1), &hook_msgs);
    assert_eq!(
        (reckoned.distrusted, reckoned.distrusted_reckoned),
        (1, 1),
        "{reckoned:?}"
    );
    let exact = roll_counts(air_tee(1, 1300, 0, 0, 1), &hook_msgs);
    assert_eq!((exact.distrusted, exact.distrusted_reckoned), (1, 0), "{exact:?}");
    // A reckoned character whose messages agree with its bits is trusted and its messages play.
    let held = [(497, PlayerInput { jump: 1, ..msg(0) }), (502, released)];
    let ok = roll_counts(air_tee(1, 1300, 496, 1, 0), &held);
    assert_eq!((ok.checked_reckoned, ok.distrusted), (1, 0), "{ok:?}");
    assert!(ok.used > 0, "{ok:?}");
}

#[test]
fn the_direction_of_a_dead_reckoned_character_is_checked_and_counted_apart() {
    // The wire's direction is kept by the evolution: a message that says "right" while the reckoned core shows "left" is a lost message.
    let right = PlayerInput { direction: 1, ..msg(0) };
    let msgs = [(497, right), (502, right)];
    let mut left = air_tee(1, 1300, 496, 0, 0);
    left.character.direction = -1;
    let c = roll_counts(left, &msgs);
    assert_eq!(
        (
            c.checked,
            c.checked_reckoned,
            c.distrusted,
            c.distrusted_reckoned,
            c.used
        ),
        (1, 1, 1, 1, 0),
        "{c:?}"
    );
}

#[test]
fn the_decision_metric_counts_predictions_for_a_target_the_pre_inputs_reached() {
    // Task 3.20b: `decisions` = predictions for a target while the pre-inputs are played; `decisions_real` = those in which the target had a step played
    // from a real pre-input. The share is the A/B metric of the live protocol.
    let chars = [tee(0, 1000), tee(1, 1300), tee(2, 1500)];
    let mut live = LiveWorld::new(room(), 0, 1);
    live.on_snapshot(SnapshotInput::new(500, &chars, DEFAULT_TUNE_PARAMS));
    let mut obs = live.build_observation(&live.base_world().clone(), None);
    let keep = [true; MAX_CLIENTS];
    let mut roll = |live: &mut LiveWorld, target: Option<i32>| {
        live.predict_local_observation(504, &[], &keep, target, &mut obs);
        let c = live.pre_inputs().counts();
        (c.decisions, c.decisions_real)
    };
    // Switched off: nothing is counted, whatever the messages.
    live.on_pre_input(1, 502, msg(0));
    assert_eq!(roll(&mut live, Some(1)), (0, 0));
    live.set_preinput(true);
    // Owner 1 has a message inside the rolled ticks (501..=504), owner 2 has none: the target decides which one counts.
    assert_eq!(roll(&mut live, Some(1)), (1, 1));
    assert_eq!(
        roll(&mut live, Some(2)),
        (2, 1),
        "no message for the target: a decision, not a real one"
    );
    // No target (the roll is not for a decision about anyone) or an id outside the table: not counted.
    assert_eq!(roll(&mut live, None), (2, 1));
    assert_eq!(roll(&mut live, Some(500)), (2, 1));
    // A message behind the snapshot is held beyond itself (the last real keys), but it tells nothing about the rolled ticks: not a real decision.
    let mut live = LiveWorld::new(room(), 0, 1);
    live.on_snapshot(SnapshotInput::new(500, &chars, DEFAULT_TUNE_PARAMS));
    live.set_preinput(true);
    live.on_pre_input(1, 499, msg(0));
    let mut obs2 = live.build_observation(&live.base_world().clone(), None);
    live.predict_local_observation(504, &[], &keep, Some(1), &mut obs2);
    let c = live.pre_inputs().counts();
    assert_eq!((c.decisions, c.decisions_real, c.used > 0), (1, 0, true), "{c:?}");
    // One for the first rolled tick (501) is the first real one.
    live.on_pre_input(1, 501, msg(0));
    live.predict_local_observation(504, &[], &keep, Some(1), &mut obs2);
    let c = live.pre_inputs().counts();
    assert_eq!((c.decisions, c.decisions_real), (2, 1), "{c:?}");
}
