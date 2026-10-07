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

// ---- task 3.17: the opponent's inputs from a model in the roll -------------------------------------------

fn walk(direction: i32) -> PlayerInput {
    PlayerInput {
        direction,
        target_y: -1,
        player_flags: 1,
        ..Default::default()
    }
}

#[test]
fn without_an_override_the_prediction_is_the_old_one_bit_for_bit() {
    let chars = [tee(0, 1000), tee(1, 1100), tee(2, 1200)];
    let mut live = LiveWorld::new(room(), 0, 1);
    live.on_snapshot(SnapshotInput::new(500, &chars, DEFAULT_TUNE_PARAMS));
    let mut obs = live.build_observation(&live.base_world().clone(), None);
    let keep = [true; MAX_CLIENTS];
    let inputs = [(501, run_right()), (503, run_right())];
    let plain = live
        .predict_local_observation(505, &inputs, &keep, Some(1), &mut obs)
        .clone();
    let none = live
        .predict_local_observation_with(505, &inputs, &keep, Some(1), &mut obs, None)
        .clone();
    for id in 0..3u8 {
        assert_eq!(plain.cores.get(id), none.cores.get(id));
    }
    // An empty override and an override for a tee that is not there change nothing either.
    let empty: [PlayerInput; 0] = [];
    let e = live
        .predict_local_observation_with(505, &inputs, &keep, Some(1), &mut obs, Some((1, &empty)))
        .clone();
    let absent = live
        .predict_local_observation_with(505, &inputs, &keep, Some(1), &mut obs, Some((9, &[walk(1)])))
        .clone();
    for id in 0..3u8 {
        assert_eq!(plain.cores.get(id), e.cores.get(id));
        assert_eq!(plain.cores.get(id), absent.cores.get(id));
    }
}

#[test]
fn the_overridden_tee_plays_its_inputs_by_step_and_holds_afterwards() {
    let chars = [tee(0, 1000), tee(1, 1300), tee(2, 1500)];
    let mut live = LiveWorld::new(room(), 0, 1);
    live.on_snapshot(SnapshotInput::new(500, &chars, DEFAULT_TUNE_PARAMS));
    let mut obs = live.build_observation(&live.base_world().clone(), None);
    let keep = [true; MAX_CLIENTS];
    // Held, tee 1 stands (direction 0 in the snapshot). Walking left for 2 of the 4 steps moves it left of where it started; the other
    // tee, which has no override, and ourselves are untouched by the override.
    let held = live
        .predict_local_observation(504, &[], &keep, Some(1), &mut obs)
        .clone();
    let over = [walk(-1), walk(-1)];
    let played = live
        .predict_local_observation_with(504, &[], &keep, Some(1), &mut obs, Some((1, &over)))
        .clone();
    let x = |w: &ddai_physics::world::World<f32>, id| w.cores.get(id).unwrap().pos.x;
    assert!(x(&played, 1) < x(&held, 1), "{} vs {}", x(&played, 1), x(&held, 1));
    assert_eq!(x(&played, 0), x(&held, 0));
    assert_eq!(x(&played, 2), x(&held, 2));
    // Step 0 of the override alone moves tee 1 in the first step already.
    let first = live
        .predict_local_observation_with(501, &[], &keep, Some(1), &mut obs, Some((1, &over)))
        .clone();
    let still = live
        .predict_local_observation(501, &[], &keep, Some(1), &mut obs)
        .clone();
    assert!(first.cores.get(1).unwrap().vel.x < still.cores.get(1).unwrap().vel.x);
}

#[test]
fn own_inputs_over_are_the_inputs_the_prediction_uses() {
    let chars = [tee(0, 1000), tee(1, 1100)];
    let mut live = LiveWorld::new(room(), 0, 1);
    live.on_snapshot(SnapshotInput::new(500, &chars, DEFAULT_TUNE_PARAMS));
    let mut out = Vec::new();
    // Nothing in flight: the held input (ours, first sighting: derived) for every step.
    live.own_inputs_over(504, &[], &mut out);
    assert_eq!(out.len(), 4);
    assert!(out.iter().all(|i| *i == out[0]));
    // Claims at 502 and 504: step k runs into tick 501 + k, so steps 0 is held, 1 and 2 take the 502 input, 3 the 504 one.
    let (a, b) = (walk(1), walk(-1));
    live.own_inputs_over(504, &[(502, a), (504, b)], &mut out);
    assert_eq!(out.len(), 4);
    assert_eq!((out[1], out[2], out[3]), (a, a, b));
    assert_ne!(out[0], a);
    // Not ahead of the snapshot: nothing.
    live.own_inputs_over(500, &[], &mut out);
    assert!(out.is_empty());
    live.own_inputs_over(400, &[], &mut out);
    assert!(out.is_empty());
    // The held input of another tee is what a plain prediction holds; an unknown id has none.
    assert_eq!(live.held_input_of(1).map(|i| i.direction), Some(0));
    assert_eq!(live.held_input_of(7), None);
    assert_eq!(live.held_input_of(-1), None);
}

// ---- task 3.20: the server's pre-inputs in the roll ---------------------------------------------------------

fn pre(direction: i32) -> PlayerInput {
    PlayerInput {
        direction,
        target_x: 100,
        target_y: -1,
        player_flags: 1,
        ..Default::default()
    }
}

fn pos_x(w: &ddai_physics::world::World<f32>, id: u8) -> f32 {
    w.cores.get(id).unwrap().pos.x
}

fn pre_rig() -> (LiveWorld, ddai_brain::Observation, [bool; MAX_CLIENTS]) {
    let chars = [tee(0, 1000), tee(1, 1300), tee(2, 2500)];
    let mut live = LiveWorld::new(room(), 0, 1);
    live.on_snapshot(SnapshotInput::new(500, &chars, DEFAULT_TUNE_PARAMS));
    let obs = live.build_observation(&live.base_world().clone(), None);
    (live, obs, [true; MAX_CLIENTS])
}

#[test]
fn a_pre_input_moves_the_predicted_tee_exactly_as_the_input_would_and_only_when_switched_on() {
    let (mut live, mut obs, keep) = pre_rig();
    let held = pos_x(live.predict_local_observation(504, &[], &keep, Some(1), &mut obs), 1);
    // The server says tee 1 walks left from tick 501 on (heard ahead: the newest message is for 503).
    live.on_pre_input(1, 501, pre(-1));
    live.on_pre_input(1, 503, pre(-1));
    // Stored, not used: off is the old prediction, bit for bit.
    assert_eq!(
        pos_x(live.predict_local_observation(504, &[], &keep, Some(1), &mut obs), 1),
        held
    );
    live.set_preinput(true);
    let with = pos_x(live.predict_local_observation(504, &[], &keep, Some(1), &mut obs), 1);
    assert!(with < held, "{with} vs {held}");
    // The same motion as playing that input by hand: the override with the same inputs on the same ticks (501..=503 walk left from the messages; 504 is beyond
    // the newest message, where the last real input persists, still walking left).
    live.set_preinput(false);
    let by_hand = pos_x(
        live.predict_local_observation_with(
            504,
            &[],
            &keep,
            Some(1),
            &mut obs,
            Some((1, &[pre(-1), pre(-1), pre(-1), pre(-1)])),
        ),
        1,
    );
    assert_eq!(with, by_hand);
    // Other tees are untouched.
    live.set_preinput(true);
    let w = live.predict_local_observation(504, &[], &keep, Some(1), &mut obs);
    assert_eq!(pos_x(w, 2), 2500.0);
    assert!(live.pre_inputs().counts().used >= 3);
}

#[test]
fn a_pre_input_beats_the_window_models_override_and_the_model_plays_where_nothing_is_known() {
    let (mut live, mut obs, keep) = pre_rig();
    live.set_preinput(true);
    // Truth for ticks 501-502: right. The model says left for all four steps.
    live.on_pre_input(1, 501, pre(1));
    live.on_pre_input(1, 502, pre(1));
    let left = [pre(-1); 4];
    let x_model_only = {
        live.set_preinput(false);
        let v = pos_x(
            live.predict_local_observation_with(504, &[], &keep, Some(1), &mut obs, Some((1, &left))),
            1,
        );
        live.set_preinput(true);
        v
    };
    let x_both = pos_x(
        live.predict_local_observation_with(504, &[], &keep, Some(1), &mut obs, Some((1, &left))),
        1,
    );
    assert!(
        x_both > x_model_only,
        "the truth for the first ticks turned it around: {x_both} vs {x_model_only}"
    );
}

#[test]
fn stale_foreign_and_own_pre_inputs_change_nothing() {
    let (mut live, mut obs, keep) = pre_rig();
    live.set_preinput(true);
    let base = pos_x(live.predict_local_observation(504, &[], &keep, Some(1), &mut obs), 1);
    // For our own id (ignored), for a tick the snapshot already confirmed with nothing after it, for a tick beyond the horizon of the roll.
    live.on_pre_input(0, 502, pre(-1));
    live.on_pre_input(1, 499, pre(-1));
    live.on_pre_input(1, 530, pre(-1));
    let x = pos_x(live.predict_local_observation(504, &[], &keep, Some(1), &mut obs), 1);
    // The message at 499 says "left" while the snapshot at 500 shows the tee standing: a message went missing in between, so its state is not trusted
    // (the other message, at 530, is beyond the roll) and the prediction is the old one.
    assert_eq!(x, base);
    assert_eq!(live.pre_inputs().counts().distrusted, 1);
    let c = live.pre_inputs().counts();
    assert_eq!(c.invalid, 1, "our own id: counted, stored nowhere");
}

#[test]
fn a_tee_that_changes_team_or_leaves_loses_its_pre_inputs() {
    let (mut live, mut obs, keep) = pre_rig();
    live.set_preinput(true);
    live.on_pre_input(1, 501, pre(-1));
    live.on_pre_input(1, 504, pre(-1));
    assert!(live.pre_inputs().newest(1) == 504);
    // The next snapshot has tee 1 in another DDRace team.
    let chars = [tee(0, 1000), tee(1, 1300), tee(2, 2500)];
    let mut teams = ddai_net::tuning::TeamsState {
        teams: [0; 128],
        received: 3,
    };
    teams.teams[1] = 5;
    let mut input = SnapshotInput::new(502, &chars, DEFAULT_TUNE_PARAMS);
    input.teams = Some(&teams);
    live.on_snapshot(input);
    assert_eq!(live.pre_inputs().newest(1), -1, "forgotten with the team change");
    let _ = (&mut obs, &keep);
    // A tee that is gone from the snapshot is forgotten too.
    live.on_pre_input(2, 505, pre(1));
    live.on_snapshot(SnapshotInput::new(504, &chars[..2], DEFAULT_TUNE_PARAMS));
    assert_eq!(live.pre_inputs().newest(2), -1);
}

// ---- task 4.3: carrying our own tee through a clip replay --------------------------------------------

#[test]
fn own_state_exported_after_a_prediction_and_imported_after_a_snapshot_resumes_the_same_trajectory() {
    // Snapshot at 500, free-run our tee to 506 on a held input; then a "new snapshot" that knows nothing
    // of where we got to (the other tee moved, ours is stale) and our imported state: predicting on to
    // 510 must equal predicting 500 -> 510 straight through.
    let inputs: Vec<(i32, PlayerInput)> = (501..=510).map(|t| (t, run_right())).collect();
    let chars = [tee(0, 1000), tee(1, 2000)];
    let mut straight = LiveWorld::new(room(), 0, 1);
    straight.on_snapshot(SnapshotInput::new(500, &chars, DEFAULT_TUNE_PARAMS));
    let want = *straight.predict(510, &inputs).cores.get(0).unwrap();

    let mut live = LiveWorld::new(room(), 0, 1);
    live.on_snapshot(SnapshotInput::new(500, &chars, DEFAULT_TUNE_PARAMS));
    let _ = live.predict(506, &inputs[..6]);
    let own = live.export_own_predicted().expect("our tee is there");
    // The next snapshot (tick 506) says we are back at the start: it is overruled by the import.
    let stale = [tee(0, 1000), tee(1, 2050)];
    let mut resumed = LiveWorld::new(room(), 0, 1);
    resumed.on_snapshot(SnapshotInput::new(500, &chars, DEFAULT_TUNE_PARAMS));
    resumed.on_snapshot(SnapshotInput::new(506, &stale, DEFAULT_TUNE_PARAMS));
    assert!(resumed.import_own(&own));
    let got = *resumed.predict(510, &inputs[6..]).cores.get(0).unwrap();
    assert_eq!(
        got.write(),
        want.write(),
        "our tee carried over the snapshot equals the straight run"
    );
    assert_eq!(got.pos, want.pos);
}

#[test]
fn import_own_refuses_when_our_tee_is_not_in_the_world_and_export_finds_none_then() {
    let chars = [tee(0, 1000)];
    let mut live = LiveWorld::new(room(), 0, 1);
    live.on_snapshot(SnapshotInput::new(500, &chars, DEFAULT_TUNE_PARAMS));
    let own = live.export_own(live.base_world()).expect("ours");
    // A snapshot without us (a death): nothing to import into, nothing to export.
    live.on_snapshot(SnapshotInput::new(502, &[tee(1, 2000)], DEFAULT_TUNE_PARAMS));
    assert!(!live.import_own(&own));
    assert!(live.export_own(live.base_world()).is_none());
}
