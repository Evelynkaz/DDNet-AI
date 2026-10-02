//! The replay on clips produced by the physics itself acting as the server (it quantizes every core
//! each tick like `CCharacter::TickDeferred`): a clip of a scripted walk, jump and hook on another tee
//! must replay exactly in both modes; a tampered clip must report its first divergence with a cause.
//! The real proof — clips from the live server — is the e2e of `ddai-bot`.

use std::sync::Arc;

use ddai_clip::ClipEvent;
use ddai_clip::format::*;
use ddai_clip::replay::{Cause, Mode, replay};
use ddai_physics::core::PlayerInput;
use ddai_physics::map::{MapData, TILE_SOLID, Tile};
use ddai_physics::vmath::Vec2;
use ddai_physics::world::{TickInput, World, spawn_character};

fn room() -> Arc<MapData> {
    let (w, h) = (60u32, 20u32);
    let mut game = vec![Tile::default(); (w * h) as usize];
    for x in 0..w {
        game[x as usize].index = TILE_SOLID;
        game[((h - 1) * w + x) as usize].index = TILE_SOLID;
    }
    for y in 0..h {
        game[(y * w) as usize].index = TILE_SOLID;
        game[(y * w + w - 1) as usize].index = TILE_SOLID;
    }
    for x in 20..40 {
        game[(12 * w + x) as usize].index = TILE_SOLID; // a platform to hook onto
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

fn input(direction: i32, jump: i32, hook: i32, tx: i32, ty: i32) -> PlayerInput {
    PlayerInput {
        direction,
        target_x: tx,
        target_y: ty,
        jump,
        fire: 0,
        hook,
        player_flags: 1,
        wanted_weapon: 0,
        next_weapon: 0,
        prev_weapon: 0,
    }
}

fn tee_rec(w: &World<f32>, id: i32) -> Option<TeeRec> {
    let c = w.cores.get(id as u8)?;
    let core = c.write();
    // What a DDNet 20.x server adds to every character: the jump counters, the weapons it holds, no freeze.
    let dd = DdRec {
        flags: (1 << 14) | (1 << 15), // WEAPON_HAMMER | WEAPON_GUN
        freeze_end: 0,
        jumps: c.jumps,
        tele_checkpoint: 0,
        strong_weak_id: w.characters[id as usize].map_or(id, |c| c.strong_weak_id),
        jumped_total: c.jumped_total,
        ninja_activation_tick: -1,
        freeze_start: -1,
        target_x: 0,
        target_y: 0,
        tune_zone_override: -1,
    };
    Some(TeeRec {
        id,
        ch: CharRec {
            tick: w.tick,
            x: core.x,
            y: core.y,
            vel_x: core.vel_x,
            vel_y: core.vel_y,
            angle: core.angle,
            direction: core.direction,
            jumped: core.jumped,
            hooked_player: core.hooked_player,
            hook_state: core.hook_state,
            hook_tick: core.hook_tick,
            hook_x: core.hook_x,
            hook_y: core.hook_y,
            hook_dx: core.hook_dx,
            hook_dy: core.hook_dy,
            player_flags: 1,
            health: 10,
            armor: 0,
            ammo_count: 0,
            weapon: 1,
            emote: 0,
            attack_tick: 0,
        },
        dd: Some(dd),
        frozen: false,
        deep_frozen: false,
        freeze_left: 0,
    })
}

/// A scripted 400-tick game of two tees; our tee (0) follows `script`, tee 1 walks left and right. A frame
/// every second tick (one every `gap` ticks where `gap` is given), like the 25 Hz snapshots.
fn record(map: &Arc<MapData>, script: impl Fn(i32) -> PlayerInput, gap: i32, ticks: i32) -> Clip {
    record_with(map, script, gap, ticks, false, 50, 90)
}

/// `wander`: tee 1 walks left and right (changing its input every 90 ticks), else it stands.
fn record_with(
    map: &Arc<MapData>,
    script: impl Fn(i32) -> PlayerInput,
    gap: i32,
    ticks: i32,
    wander: bool,
    tee1_tile: i32,
    period: i32,
) -> Clip {
    let mut w = World::<f32>::from_map(map, 7);
    let _ = w.init(std::iter::empty::<&str>());
    w.projectiles.clear();
    spawn_character(&mut w, 0, Vec2::new(10.0 * 32.0 + 16.0, 18.0 * 32.0 + 16.0));
    spawn_character(&mut w, 1, Vec2::new(tee1_tile as f32 * 32.0 + 16.0, 18.0 * 32.0 + 16.0));
    let mut frames = Vec::new();
    let mut pending_sent: Vec<SentRec> = Vec::new();
    for _ in 0..ticks {
        let t = w.tick + 1;
        let mine = script(t);
        let theirs = input(
            if wander {
                if (t / period) % 2 == 0 { 1 } else { -1 }
            } else {
                0
            },
            0,
            0,
            100,
            0,
        );
        w.step(&[
            TickInput {
                id: 0,
                input: mine,
                kill: false,
            },
            TickInput {
                id: 1,
                input: theirs,
                kill: false,
            },
        ]);
        pending_sent.push(SentRec {
            tick: w.tick,
            input: InputRec::from_net(&ddai_world::player_input_to_net(mine)),
            timing_known: true,
        });
        if w.tick % gap == 0 {
            let mut tees = vec![tee_rec(&w, 0).unwrap()];
            tees.extend(tee_rec(&w, 1));
            frames.push(Frame {
                tick: w.tick,
                own_alive: true,
                tees,
                sent: std::mem::take(&mut pending_sent),
                ..Frame::default()
            });
        }
    }
    Clip {
        header: ClipHeader {
            map_name: "room".into(),
            map_sha256: [0; 32],
            own_id: 0,
            brain: "script".into(),
            reason: ClipReason {
                kind: "manual".into(),
                severity: 0,
                tick: 0,
                note: String::new(),
            },
            labels: vec![String::new()],
            players: vec![],
            tuning: vec![],
            teams: vec![],
            world_seed: 7,
        },
        frames,
    }
}

fn script(t: i32) -> PlayerInput {
    match t % 200 {
        0..=59 => input(1, 0, 0, 100, -100),
        60..=64 => input(1, 1, 0, 100, -100),
        65..=99 => input(1, 0, 1, 200, -300), // hook up and right
        100..=139 => input(-1, 0, 0, -100, -100),
        140..=149 => input(-1, 1, 0, -100, -100),
        150..=179 => input(1, 0, 1, 300, 0),
        _ => input(0, 0, 0, 100, 0),
    }
}

#[test]
fn a_clip_of_the_physics_replays_exactly_in_both_modes() {
    let map = room();
    for gap in [2, 2, 4] {
        let clip = record(&map, script, gap, 600);
        assert!(clip.frames.len() >= 150);
        for mode in [Mode::Resync, Mode::FreeRun] {
            let r = replay(&clip, Arc::clone(&map), mode);
            assert!(r.steps > 100, "{mode:?}: {}", r.steps);
            assert!(
                r.is_exact(),
                "{mode:?} gap {gap}: first divergence {:?}",
                r.first_divergence
            );
        }
    }
}

#[test]
fn the_clip_survives_the_file_and_replays_the_same() {
    let map = room();
    let clip = record(&map, script, 2, 300);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.clip");
    clip.write(&path).unwrap();
    let back = Clip::read(&path).unwrap();
    assert_eq!(back, clip);
    assert!(replay(&back, map, Mode::FreeRun).is_exact());
    assert!(!dir.path().join("x.clip.tmp").exists(), "written atomically");
}

#[test]
fn a_tampered_frame_is_the_first_divergence_with_its_cause() {
    let map = room();
    let mut clip = record(&map, script, 2, 400);
    // The server "corrects" our x by 3 px at frame 80, nobody around.
    let at = 80;
    clip.frames[at].tees[0].ch.x += 3;
    let r = replay(&clip, Arc::clone(&map), Mode::FreeRun);
    let d = r.first_divergence.clone().expect("a divergence");
    assert_eq!(d.frame, at);
    assert_eq!(d.field, "x");
    assert_eq!(d.recorded - d.replayed, 3);
    assert!(
        matches!(d.cause, Cause::OtherTee { .. } | Cause::ServerCorrection),
        "{:?}",
        d.cause
    );
    assert_eq!(r.exact, at - 1, "everything before it was exact");
    // Resync mode reports it and goes on: the step after the tampered frame starts from the tampered state.
    let r2 = replay(&clip, Arc::clone(&map), Mode::Resync);
    assert_eq!(r2.first_divergence.as_ref().map(|d| d.frame), Some(at));
    assert!(
        r2.divergences.len() <= 2,
        "{:?}",
        r2.divergences.iter().map(|d| d.frame).collect::<Vec<_>>()
    );

    // With the other tee far away and a projectile-free, input-confirmed step it is a server correction.
    let mut far = record(&map, script, 2, 400);
    far.frames[at].tees[0].ch.y -= 2;
    for f in &mut far.frames {
        f.tees.truncate(1); // nobody else recorded
    }
    let d = replay(&far, map, Mode::FreeRun).first_divergence.expect("a divergence");
    assert_eq!((d.frame, d.field), (at, "y"));
    assert_eq!(d.cause, Cause::ServerCorrection);
}

#[test]
fn an_unconfirmed_input_timing_is_named_as_the_cause() {
    let map = room();
    let mut clip = record(&map, script, 2, 300);
    // The step into frame 70 used an input whose timing report had not arrived, and it was wrong.
    let at = 70;
    for f in &mut clip.frames {
        f.tees.truncate(1);
    }
    clip.frames[at].sent[0].timing_known = false;
    clip.frames[at].sent[0].input.direction += 1;
    clip.frames[at].sent[0].input.jump = 1;
    clip.frames[at].sent[1].input.direction += 1;
    let r = replay(&clip, map, Mode::FreeRun);
    // The altered inputs change our state in that step (or the next): whatever the first divergence is,
    // it is caused by the timing-unconfirmed step when it falls on that frame.
    let d = r.first_divergence.expect("the altered input changes the physics");
    assert!(d.frame >= at, "{d:?}");
    if d.frame == at {
        assert_eq!(d.cause, Cause::InputTiming);
    }
}

#[test]
fn deaths_are_skipped_and_the_replay_restarts_from_the_recording() {
    let map = room();
    let mut clip = record(&map, script, 2, 300);
    // Our tee is absent for 5 frames (a death), then it respawns elsewhere.
    for f in clip.frames[40..45].iter_mut() {
        f.own_alive = false;
        f.tees.retain(|t| t.id != 0);
    }
    clip.frames[45].tees[0].ch.x = 400;
    clip.frames[45].tees[0].ch.y = 400;
    let r = replay(&clip, map, Mode::Resync);
    assert!(r.skipped_deaths >= 5);
    // The frames after the respawn replay from the recorded respawn state; the one step that crosses it is
    // skipped, the rest need not be exact (the scripted game did not respawn) but the replay does not crash.
    assert!(r.steps > 100);
}

#[test]
fn a_divergence_with_a_neighbour_in_reach_is_blamed_on_it() {
    let map = room();
    // We stand still with tee 1 standing a tile away (in reach); the server "corrects" us at frame 30.
    let mut clip = record_with(&map, |_| input(0, 0, 0, 100, 0), 2, 200, false, 12, 90);
    assert!(
        replay(&clip, Arc::clone(&map), Mode::FreeRun).is_exact(),
        "an unchanged neighbour is no problem"
    );
    clip.frames[30].tees[0].ch.x += 2;
    let d = replay(&clip, Arc::clone(&map), Mode::FreeRun)
        .first_divergence
        .expect("tampered");
    assert_eq!((d.frame, d.field), (30, "x"));
    assert_eq!(d.cause, Cause::OtherTee { id: 1 });
}

#[test]
fn a_respawn_and_a_teleport_are_named_not_physics_and_the_replay_goes_on_from_the_recording() {
    let map = room();
    // We die and reappear 300 px away at frame 60 (a kill message for us is in that frame): a respawn.
    let mut clip = record(&map, script, 2, 400);
    for f in &mut clip.frames[60..] {
        f.tees[0].ch.x += 300;
    }
    clip.frames[60].events.push(ClipEvent::Kill {
        killer: -1,
        victim: 0,
        weapon: -1,
    });
    let r = replay(&clip, Arc::clone(&map), Mode::Resync);
    // (The shifted frames after it need not all be physically consistent with the room's walls; only the
    // step with the kill message is a respawn.)
    assert!(r.respawn_steps >= 1);
    assert_eq!((r.divergences[0].frame, &r.divergences[0].cause), (60, &Cause::Respawn));
    assert!(
        r.first_divergence.as_ref().is_none_or(|d| d.frame != 60),
        "a respawn is not 'the first divergence'"
    );
    assert!(!r.is_exact());
    // The free run restarts our tee there and keeps going: the later frames are shifted by 300 px, so the
    // physics (walls) no longer matches them, but the point is that it did not stop at the respawn.
    let free = replay(&clip, Arc::clone(&map), Mode::FreeRun);
    assert!(free.steps > 60, "the free run went past the respawn: {}", free.steps);

    // The same jump with no death and no teleporter anywhere is NOT excused: the server changed our state for
    // no reason the clip shows (review F3).
    let mut jump = record(&map, script, 2, 400);
    for f in &mut jump.frames[60..] {
        f.tees[0].ch.x += 300;
    }
    let r = replay(&jump, Arc::clone(&map), Mode::Resync);
    assert_eq!(
        r.divergences[0].cause,
        Cause::ServerCorrection,
        "{:?}",
        r.divergences[0]
    );
    assert!(r.first_divergence.is_some(), "and it stays the first divergence");
    let free = replay(&jump, map, Mode::FreeRun);
    assert_eq!(
        free.first_divergence.as_ref().map(|d| d.frame),
        Some(60),
        "the free run stops there"
    );
}

/// The room with a tele layer: a tele-in under the tee's start and a tele-out far away.
fn room_with_tele(tele_in: (u32, u32), tele_out: (u32, u32)) -> Arc<MapData> {
    use ddai_physics::map::{TILE_TELEIN, TILE_TELEOUT, TeleTile};
    let base = room();
    let mut m = (*base).clone();
    let mut tele = vec![TeleTile::default(); (m.width * m.height) as usize];
    tele[(tele_in.1 * m.width + tele_in.0) as usize] = TeleTile {
        number: 1,
        kind: TILE_TELEIN,
    };
    tele[(tele_out.1 * m.width + tele_out.0) as usize] = TeleTile {
        number: 1,
        kind: TILE_TELEOUT,
    };
    m.tele = Some(tele);
    Arc::new(m)
}

#[test]
fn a_jump_is_a_teleport_only_next_to_a_tele_in_or_on_a_tele_out() {
    let plain = room();
    let mut clip = record(&plain, |_| input(0, 0, 0, 100, 0), 2, 200);
    for f in &mut clip.frames[40..] {
        f.tees[0].ch.x += 600;
    }
    // Tele-in at our tile (10, 18): the jump is the teleporter's random pick.
    let m = room_with_tele((10, 18), (40, 5));
    let r = replay(&clip, Arc::clone(&m), Mode::Resync);
    assert_eq!(r.divergences[0].cause, Cause::Teleport, "{:?}", r.divergences[0]);
    // Landing on a tele-out (x = 336 + 600 = 936 -> tile 29, y tile 18) with the tele-in elsewhere (far).
    let m = room_with_tele((50, 3), (29, 18));
    let r = replay(&clip, m, Mode::Resync);
    assert_eq!(r.divergences[0].cause, Cause::Teleport, "{:?}", r.divergences[0]);
    // A small correction next to the tele tiles (the CLB spawn has tele-outs two tiles above it) is no teleport,
    // with the tele-out two tiles above or a tele-in right next to the tee (review F5).
    for m in [room_with_tele((50, 3), (10, 16)), room_with_tele((11, 18), (50, 3))] {
        let mut small = record(&plain, |_| input(0, 0, 0, 100, 0), 2, 200);
        small.frames[40].tees[0].ch.x += 2;
        let r = replay(&small, m, Mode::Resync);
        assert_eq!(
            r.divergences[0].cause,
            Cause::ServerCorrection,
            "{:?}",
            r.divergences[0]
        );
        assert!(r.unexplained().count() >= 1);
    }
    // A jump next to a tee within reach is a fight, not a teleport, even with a tele tile there.
    let mut brawl = record_with(&plain, |_| input(0, 0, 0, 100, 0), 2, 200, false, 12, 90);
    for f in &mut brawl.frames[40..] {
        f.tees[0].ch.x += 600;
        f.tees[1].ch.x += 600;
    }
    let r = replay(&brawl, room_with_tele((10, 18), (40, 5)), Mode::Resync);
    assert_eq!(
        r.divergences[0].cause,
        Cause::OtherTee { id: 1 },
        "{:?}",
        r.divergences[0]
    );
    // A tele layer somewhere else on the map excuses nothing.
    let m = room_with_tele((50, 3), (55, 3));
    let r = replay(&clip, m, Mode::Resync);
    assert_eq!(
        r.divergences[0].cause,
        Cause::ServerCorrection,
        "{:?}",
        r.divergences[0]
    );
}

#[test]
fn the_reviewers_tampered_cases_are_server_corrections_not_excuses() {
    let map = room();
    let at = 30;
    let stand = |_: i32| input(0, 0, 0, 100, 0);
    let cause_of_first = |clip: &Clip| {
        let r = replay(clip, Arc::clone(&map), Mode::Resync);
        r.divergences.first().map(|d| (d.frame, d.cause.clone()))
    };
    // +2 px with another tee 13 tiles (416 px) away and not linked: out of reach, so a server correction.
    let mut far_tee = record_with(&map, stand, 2, 120, false, 23, 90);
    far_tee.frames[at].tees[0].ch.x += 2;
    assert_eq!(cause_of_first(&far_tee), Some((at, Cause::ServerCorrection)));
    // The same with the other tee 2 tiles away (within reach): blamed on it.
    let mut near_tee = record_with(&map, stand, 2, 120, false, 12, 90);
    near_tee.frames[at].tees[0].ch.x += 2;
    assert_eq!(cause_of_first(&near_tee), Some((at, Cause::OtherTee { id: 1 })));
    // Far, but its hook is on us: linked, blamed on it.
    let mut hooked = record_with(&map, stand, 2, 120, false, 23, 90);
    hooked.frames[at].tees[0].ch.x += 2;
    hooked.frames[at].tees[1].ch.hooked_player = 0;
    assert_eq!(cause_of_first(&hooked), Some((at, Cause::OtherTee { id: 1 })));
    // Far, with its hook tip at our position: it may have caught us for a tick.
    let mut tip = record_with(&map, stand, 2, 120, false, 23, 90);
    tip.frames[at].tees[0].ch.x += 2;
    let (mx, my) = (tip.frames[at].tees[0].ch.x, tip.frames[at].tees[0].ch.y);
    tip.frames[at].tees[1].ch.hook_state = 4;
    tip.frames[at].tees[1].ch.hook_x = mx + 10;
    tip.frames[at].tees[1].ch.hook_y = my;
    assert_eq!(cause_of_first(&tip), Some((at, Cause::OtherTee { id: 1 })));
    // A projectile 800 px away: nothing to do with us.
    let mut far_proj = record_with(&map, stand, 2, 120, false, 50, 90);
    far_proj.frames[at].tees[0].ch.x += 2;
    let (mx, my) = (far_proj.frames[at].tees[0].ch.x, far_proj.frames[at].tees[0].ch.y);
    for k in [at - 1, at] {
        far_proj.frames[k].projectiles.push(ddai_clip::ProjRec {
            id: 1,
            kind: 2,
            v: [mx + 800, my, 0, 0, 3, 0, 5, 0, 0, 0],
        });
    }
    assert_eq!(cause_of_first(&far_proj), Some((at, Cause::ServerCorrection)));
    // A projectile 60 px away could have hit or exploded on us.
    let mut near_proj = record_with(&map, stand, 2, 120, false, 50, 90);
    near_proj.frames[at].tees[0].ch.x += 2;
    let (mx, my) = (near_proj.frames[at].tees[0].ch.x, near_proj.frames[at].tees[0].ch.y);
    near_proj.frames[at].projectiles.push(ddai_clip::ProjRec {
        id: 1,
        kind: 2,
        v: [mx + 60, my, 0, 0, 3, 0, 5, 0, 0, 0],
    });
    assert_eq!(cause_of_first(&near_proj), Some((at, Cause::Projectile)));
    // Our freeze state changing in the step is a named reason (the thaw tick is only known to the snapshot).
    let mut thaw = record_with(&map, stand, 2, 120, false, 50, 90);
    thaw.frames[at].tees[0].ch.x += 2;
    thaw.frames[at].tees[0].frozen = true;
    assert_eq!(cause_of_first(&thaw), Some((at, Cause::FreezeChange)));
    // Tees left out of the frame explain nothing when the farthest recorded one is far (the nearest are kept).
    let mut dropped = record_with(&map, stand, 2, 120, false, 23, 90);
    dropped.frames[at].tees[0].ch.x += 2;
    dropped.frames[at].tees_dropped = 3;
    assert_eq!(cause_of_first(&dropped), Some((at, Cause::ServerCorrection)));
}

#[test]
fn steps_with_nobody_near_are_counted_apart_as_the_pure_test_of_our_own_physics() {
    let map = room();
    // The other tee stands 40 tiles away at first: nobody can touch us for a good part of the walk.
    let clip = record(&map, script, 2, 400);
    let r = replay(&clip, Arc::clone(&map), Mode::Resync);
    assert!(r.isolated_steps >= 100, "{r:?}");
    assert_eq!(r.isolated_exact, r.isolated_steps, "{r:?}");
    // The same clip with the other tee 3 tiles from us is not isolated at all.
    let close = record_with(&map, |_| input(0, 0, 0, 100, 0), 2, 100, false, 13, 90);
    let r = replay(&close, Arc::clone(&map), Mode::Resync);
    assert_eq!(r.isolated_steps, 0);
    // A server correction with nobody near is the one thing that stays unexplained.
    let mut bad = record(&map, script, 2, 300);
    bad.frames[50].tees[0].ch.vel_x += 77;
    for f in &mut bad.frames {
        f.tees.truncate(1);
    }
    let r = replay(&bad, map, Mode::Resync);
    let un: Vec<_> = r.unexplained().collect();
    assert_eq!(un.len(), 1, "{:?}", r.divergences);
    assert_eq!((un[0].frame, un[0].field), (50, "vel_x"));
    assert_eq!(r.by_cause().get("server-correction"), Some(&1));
}
