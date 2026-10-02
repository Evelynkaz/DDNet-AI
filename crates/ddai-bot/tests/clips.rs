//! The bot's clip recording through the whole pipeline (task 4.3): frames per snapshot, the inputs
//! actually sent, tags instead of nicknames, real events, the death tail, the autoclip (inline and on the
//! worker thread), the cross-fail clip and pruning. No network. The bit-exact replay of clips recorded
//! against the real server is the gated e2e (`tests/e2e_commands.rs`).

mod support;

use std::path::Path;

use ddai_bot::clipper::ClipConfig;
use ddai_bot::hooks::{Hooks, Navigator};
use ddai_bot::{Bot, BotEvent, BrainKind, Relations};
use ddai_clip::{Clip, ClipEvent};
use support::*;

fn clip_cfg(dir: &Path, autoclip: bool, async_save: bool) -> ddai_bot::BotConfig {
    let mut c = cfg(BrainKind::Idle);
    c.clips = ClipConfig {
        dir: Some(dir.to_path_buf()),
        autoclip,
        async_save,
    };
    c
}

fn bot_in(dir: &Path, autoclip: bool, async_save: bool, hooks: Hooks) -> (Bot, Scenario) {
    let mut bot = Bot::new(
        clip_cfg(dir, autoclip, async_save),
        Box::new(ddai_brain::IdleBrain),
        hooks,
        Relations::new(),
    );
    let map = room(&[]);
    bot.set_map_ident(ddai_bot::hooks::MapIdent {
        name: "room".to_string(),
        sha256: [9; 32],
    });
    bot.on_map_loaded(std::sync::Arc::clone(&map));
    let sc = Scenario::new(map, vec![tee(0, 1000), tee(1, 1300)]);
    (bot, sc)
}

fn read_all(dir: &Path) -> Vec<(String, Clip)> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let p = e.path();
        if p.extension().is_some_and(|x| x == "clip") {
            let stem = p.file_stem().unwrap().to_string_lossy().to_string();
            out.push((stem, Clip::read(&p).unwrap()));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn events_of(clip: &Clip) -> Vec<ClipEvent> {
    clip.frames.iter().flat_map(|f| f.events.iter().copied()).collect()
}

#[test]
fn every_snapshot_is_a_frame_with_the_inputs_sent_and_the_clip_holds_tags_not_names() {
    big_stack(|| {
        let dir = tempfile::tempdir().unwrap();
        let (mut bot, mut sc) = bot_in(dir.path(), false, false, Hooks::default());
        for i in 0..30 {
            let mut snap = sc.snapshot();
            if i >= 20 {
                snap.tuning.ground_friction = 100; // a tuning change mid-clip
            }
            let out = bot.on_snapshot(&snap);
            if let Some(input) = out.input {
                bot.on_input_sent(snap.pred_tick + 1, &input);
            }
            sc.tick += 2;
        }
        assert_eq!(bot.clip_frames(), 30);
        let saved = bot.save_clip("first try").expect("saved");
        assert!(
            saved.path.file_name().unwrap().to_string_lossy().starts_with("manual-"),
            "{:?}",
            saved.path
        );
        assert!(
            saved.path.to_string_lossy().ends_with("first_try.clip"),
            "{:?}",
            saved.path
        );
        let clip = Clip::read(&saved.path).unwrap();
        assert_eq!(clip.frames.len(), 30);
        assert_eq!(clip.header.map_name, "room");
        assert_eq!(clip.header.map_sha256, [9; 32]);
        assert_eq!(clip.header.own_id, 0);
        assert_eq!(clip.header.brain, "idle");
        assert_eq!(clip.header.reason.kind, "manual");
        assert_eq!(clip.header.reason.note, "first try");
        assert_eq!(clip.header.tuning.len(), 2, "the initial tuning and the change");
        assert_eq!(clip.header.tuning[1].from_tick, clip.frames[20].tick);
        // The ring's frames are in tick order, both tees recorded, and sent inputs are there once the
        // first decisions have gone out.
        assert!(clip.frames.windows(2).all(|w| w[0].tick < w[1].tick));
        assert!(clip.frames.iter().all(|f| f.tees.len() == 2 && f.own_alive));
        let with_input = clip.frames.iter().filter(|f| f.own_input().is_some()).count();
        assert!(with_input >= 25, "{with_input}");
        for f in clip.frames.iter().filter(|f| f.own_input().is_some()) {
            assert_eq!(
                f.own_input().unwrap().tick,
                f.tick,
                "the last input is the one in force at the frame"
            );
        }
        // Tags only: the scenario names are `p<id>`.
        assert!(
            clip.header
                .players
                .iter()
                .all(|p| p.tag.starts_with(&format!("c{}-", p.id)))
        );
        let bytes = std::fs::read(&saved.path).unwrap();
        let raw = ddai_clip::format::Clip::decode(&bytes).unwrap();
        let dbg = format!("{raw:?}");
        assert!(
            !dbg.contains("\"p0\"") && !dbg.contains("\"p1\""),
            "no nickname in the clip"
        );
    });
}

#[test]
fn real_events_hook_hammer_freeze_and_kill_land_in_the_frames() {
    big_stack(|| {
        let dir = tempfile::tempdir().unwrap();
        let (mut bot, mut sc) = bot_in(dir.path(), false, false, Hooks::default());
        sc.tee_mut(1).x = 1030; // within the hammer's reach of us
        let step = |bot: &mut Bot, sc: &mut Scenario| {
            let snap = sc.snapshot();
            let out = bot.on_snapshot(&snap);
            if let Some(input) = out.input {
                bot.on_input_sent(snap.pred_tick + 1, &input);
            }
            sc.tick += 2;
        };
        for _ in 0..6 {
            step(&mut bot, &mut sc);
        }
        // Our hook grabs tee 1 for three snapshots.
        {
            let t = sc.tee_mut(0);
            t.hook_state = 5;
            t.hooked_player = 1;
        }
        for _ in 0..3 {
            step(&mut bot, &mut sc);
        }
        {
            let t = sc.tee_mut(0);
            t.hook_state = 0;
            t.hooked_player = -1;
        }
        step(&mut bot, &mut sc);
        // Tee 1 swings its hammer at us (it points left, towards us).
        {
            let t = sc.tee_mut(1);
            t.angle = 804;
            t.attack_tick = 5;
        }
        step(&mut bot, &mut sc);
        // Tee 1 freezes.
        sc.tee_mut(1).frozen = true;
        step(&mut bot, &mut sc);
        // A kill message and then one more snapshot.
        bot.on_kill_message(0, 1, 0);
        step(&mut bot, &mut sc);
        let clip = Clip::read(&bot.save_clip("").unwrap().path).unwrap();
        let ev = events_of(&clip);
        assert!(ev.contains(&ClipEvent::HookAttach { id: 0, target: 1 }), "{ev:?}");
        let release = ev.iter().find_map(|e| match e {
            ClipEvent::HookRelease { id: 0, target: 1, held } => Some(*held),
            _ => None,
        });
        assert_eq!(release, Some(6), "held for three snapshots of two ticks: {ev:?}");
        assert!(ev.contains(&ClipEvent::HammerFire { from: 1, hits: 1 }), "{ev:?}");
        assert!(ev.contains(&ClipEvent::HammerHit { from: 1, to: 0 }), "{ev:?}");
        assert!(ev.contains(&ClipEvent::FreezeOnset { id: 1 }), "{ev:?}");
        assert!(
            ev.contains(&ClipEvent::Kill {
                killer: 0,
                victim: 1,
                weapon: 0
            }),
            "{ev:?}"
        );
        // Each exactly once.
        assert_eq!(
            ev.iter()
                .filter(|e| matches!(e, ClipEvent::FreezeOnset { id: 1 }))
                .count(),
            1
        );
        assert_eq!(
            ev.iter().filter(|e| matches!(e, ClipEvent::HookAttach { .. })).count(),
            1
        );
    });
}

#[test]
fn a_crowd_records_our_tee_the_tees_linked_by_a_hook_and_the_nearest_up_to_eight() {
    big_stack(|| {
        let dir = tempfile::tempdir().unwrap();
        let map = room(&[]);
        let mut bot = Bot::new(
            clip_cfg(dir.path(), false, false),
            Box::new(ddai_brain::IdleBrain),
            Hooks::default(),
            Relations::new(),
        );
        bot.on_map_loaded(std::sync::Arc::clone(&map));
        let mut tees = vec![tee(0, 1000)];
        for i in 1..=11 {
            tees.push(tee(i, 1000 + 60 * i));
        }
        // Tee 12 is far away but its hook is on us.
        let mut far = tee(12, 3000);
        far.hooked_player = 0;
        far.hook_state = 5;
        tees.push(far);
        let mut sc = Scenario::new(map, tees);
        for _ in 0..4 {
            let snap = sc.snapshot();
            bot.on_snapshot(&snap);
            sc.tick += 2;
        }
        let clip = Clip::read(&bot.save_clip("").unwrap().path).unwrap();
        let f = clip.frames.last().unwrap();
        let ids: Vec<i32> = f.tees.iter().map(|t| t.id).collect();
        assert_eq!(f.tees[0].id, 0, "ours first");
        assert!(ids.contains(&12), "the far tee hooked to us is in: {ids:?}");
        assert_eq!(ids.len(), 9, "us, the hooked one and the nearest seven: {ids:?}");
        for id in 1..=7 {
            assert!(ids.contains(&id), "{id} is among the nearest: {ids:?}");
        }
        assert_eq!(f.tees_dropped, 4, "tees 8..=11 were left out");
    });
}

#[test]
fn the_death_is_recorded_for_a_few_frames_and_then_the_ring_waits_for_the_respawn() {
    big_stack(|| {
        let dir = tempfile::tempdir().unwrap();
        let (mut bot, mut sc) = bot_in(dir.path(), false, false, Hooks::default());
        let step = |bot: &mut Bot, sc: &mut Scenario| {
            let snap = sc.snapshot();
            bot.on_snapshot(&snap);
            sc.tick += 2;
        };
        for _ in 0..10 {
            step(&mut bot, &mut sc);
        }
        let own = sc.tees.remove(0);
        for _ in 0..12 {
            step(&mut bot, &mut sc);
        }
        sc.tees.insert(0, own);
        for _ in 0..3 {
            step(&mut bot, &mut sc);
        }
        let clip = Clip::read(&bot.save_clip("").unwrap().path).unwrap();
        let dead = clip.frames.iter().filter(|f| !f.own_alive).count();
        assert_eq!(
            dead,
            ddai_bot::clipper::TAIL_FRAMES as usize,
            "a few frames after the tee is gone"
        );
        assert_eq!(
            clip.frames.len(),
            10 + dead + 3,
            "nothing recorded while we are gone for long"
        );
        assert!(events_of(&clip).contains(&ClipEvent::Respawn { id: 0 }));
        // The death incident sees the tee vanish.
        let found = ddai_clip::find_incidents(&clip.frames, 0, 40);
        assert!(found.iter().any(|i| i.kind == "death"), "{found:?}");
    });
}

/// Frames of a long freeze that we walked into at frame 76 (a self-freeze of severity > 180 at the
/// 150th frame's scan).
fn walk_into_a_freeze(bot: &mut Bot, sc: &mut Scenario, frames: usize) {
    for _ in 0..frames {
        walk_one_frame(bot, sc);
    }
}

fn walk_one_frame(bot: &mut Bot, sc: &mut Scenario) {
    let i = ((sc.tick - 1000) / 2) as usize;
    sc.tee_mut(0).x = 1000 + 8 * i as i32;
    sc.tee_mut(0).frozen = i >= 76;
    sc.tee_mut(1).x = 200; // far behind: the opponent is free and not closing in
    let snap = sc.snapshot();
    let out = bot.on_snapshot(&snap);
    if let Some(input) = out.input {
        bot.on_input_sent(snap.pred_tick + 1, &input);
    }
    sc.tick += 2;
}

#[test]
fn the_autoclip_saves_the_worst_incident_of_the_second_half_once_per_cooldown() {
    big_stack(|| {
        let dir = tempfile::tempdir().unwrap();
        let (mut bot, mut sc) = bot_in(dir.path(), true, false, Hooks::default());
        walk_into_a_freeze(&mut bot, &mut sc, 149);
        assert!(read_all(dir.path()).is_empty(), "the scan comes at the 150th frame");
        walk_into_a_freeze(&mut bot, &mut sc, 1);
        let clips = read_all(dir.path());
        assert_eq!(clips.len(), 1, "{:?}", clips.iter().map(|c| &c.0).collect::<Vec<_>>());
        let (name, clip) = &clips[0];
        assert!(name.starts_with("self-freeze-"), "{name}");
        assert_eq!(clip.header.reason.kind, "self-freeze");
        assert!(clip.header.reason.severity >= 180);
        assert_eq!(clip.frames.len(), 150);
        assert!(
            events_of(clip)
                .iter()
                .any(|e| matches!(e, ClipEvent::FreezeOnset { id: 0 }))
        );
        let evs: Vec<BotEvent> = bot.drain_events().collect();
        assert!(
            evs.iter()
                .any(|e| matches!(e, BotEvent::ClipSaved { kind, .. } if kind == "self-freeze")),
            "{evs:?}"
        );
        // Another 100 frames of the same freeze: the 45 s cooldown holds.
        walk_into_a_freeze(&mut bot, &mut sc, 100);
        assert_eq!(read_all(dir.path()).len(), 1, "cooldown");
    });
}

#[test]
fn no_autoclip_writes_nothing_and_manual_clips_still_work() {
    big_stack(|| {
        let dir = tempfile::tempdir().unwrap();
        let (mut bot, mut sc) = bot_in(dir.path(), false, false, Hooks::default());
        walk_into_a_freeze(&mut bot, &mut sc, 160);
        assert!(read_all(dir.path()).is_empty(), "--no-autoclip");
        bot.save_clip("by hand").unwrap();
        assert_eq!(read_all(dir.path()).len(), 1);
        // No directory at all: the ring records, saving says why not.
        let mut plain = bot_with(Box::new(ddai_brain::IdleBrain), cfg(BrainKind::Idle), Relations::new());
        plain.on_map_loaded(room(&[]));
        assert!(plain.save_clip("x").is_err());
    });
}

#[test]
fn the_worker_thread_saves_the_same_clip_and_shutdown_waits_for_it() {
    big_stack(|| {
        let dir = tempfile::tempdir().unwrap();
        let (mut bot, mut sc) = bot_in(dir.path(), true, true, Hooks::default());
        for _ in 0..150 {
            walk_one_frame(&mut bot, &mut sc);
            bot.flush_clips(); // the scenario runs far faster than real time: wait for the scans of 50 and 100
        }
        bot.shutdown();
        let clips = read_all(dir.path());
        assert_eq!(clips.len(), 1, "{:?}", clips.iter().map(|c| &c.0).collect::<Vec<_>>());
        assert!(clips[0].0.starts_with("self-freeze-"));
        let evs: Vec<BotEvent> = bot.drain_events().collect();
        assert!(evs.iter().any(|e| matches!(e, BotEvent::ClipSaved { .. })), "{evs:?}");
    });
}

/// Reports a failed crossing at its `at`-th poll.
struct DelayedFail {
    frames: u32,
    at: u32,
    note: String,
}

impl Navigator for DelayedFail {
    fn take_cross_fail(&mut self) -> Option<String> {
        self.frames += 1;
        (self.frames == self.at).then(|| self.note.clone())
    }
}

fn failing_at(at: u32, note: &str) -> Hooks {
    Hooks {
        navigator: Box::new(DelayedFail {
            frames: 0,
            at,
            note: note.to_string(),
        }),
        ..Hooks::default()
    }
}

#[test]
fn a_failed_crossing_saves_a_cross_fail_clip_when_enough_frames_are_in() {
    big_stack(|| {
        // Too early: fewer than 50 frames, nothing is saved.
        let dir = tempfile::tempdir().unwrap();
        let (mut bot, mut sc) = bot_in(
            dir.path(),
            true,
            false,
            failing_at(10, "tube: it did not hold; trying again from the spawn"),
        );
        for _ in 0..20 {
            let snap = sc.snapshot();
            bot.on_snapshot(&snap);
            sc.tick += 2;
        }
        assert!(read_all(dir.path()).is_empty(), "fewer than 50 frames");

        let dir = tempfile::tempdir().unwrap();
        let (mut bot, mut sc) = bot_in(
            dir.path(),
            true,
            false,
            failing_at(60, "no way through the tube to (3,4): blocked, 2 times"),
        );
        for _ in 0..70 {
            let snap = sc.snapshot();
            bot.on_snapshot(&snap);
            sc.tick += 2;
        }
        let clips = read_all(dir.path());
        assert_eq!(clips.len(), 1);
        assert!(
            clips[0].0.starts_with("cross-fail-") && clips[0].0.ends_with("-s0"),
            "{}",
            clips[0].0
        );
        assert_eq!(clips[0].1.header.reason.kind, "cross-fail");
        assert!(clips[0].1.header.reason.note.starts_with("no way through"));
        assert!(clips[0].1.frames.len() >= 50);

        // With the autoclip off, a failed crossing saves nothing either.
        let dir = tempfile::tempdir().unwrap();
        let (mut bot, mut sc) = bot_in(
            dir.path(),
            false,
            false,
            failing_at(60, "x; trying again from the spawn"),
        );
        for _ in 0..70 {
            let snap = sc.snapshot();
            bot.on_snapshot(&snap);
            sc.tick += 2;
        }
        assert!(read_all(dir.path()).is_empty());
    });
}

#[test]
fn the_frames_say_what_the_brain_decided_and_how_long_it_took() {
    big_stack(|| {
        let dir = tempfile::tempdir().unwrap();
        for (kind, code) in [(BrainKind::Planner, 1u8), (BrainKind::Hybrid, 0u8)] {
            let mut c = clip_cfg(dir.path(), false, false);
            c.brain = kind;
            let brain = ddai_bot::make_brain(kind, &ddai_bot::BrainOptions::default()).expect("brain");
            let mut bot = Bot::new(c, brain, Hooks::default(), Relations::new());
            let map = room(&[]);
            bot.on_map_loaded(std::sync::Arc::clone(&map));
            let mut sc = Scenario::new(map, vec![tee(0, 1000), tee(1, 1200)]);
            for i in 0..30 {
                sc.tee_mut(1).angle = (i * 37) % 1000; // not AFK
                let snap = sc.snapshot();
                let out = bot.on_snapshot(&snap);
                if let Some(input) = out.input {
                    bot.on_input_sent(snap.pred_tick + 1, &input);
                }
                sc.tick += 2;
            }
            let clip = Clip::read(&bot.save_clip(kind.name()).unwrap().path).unwrap();
            assert_eq!(clip.header.brain, bot.brain_name());
            let decided: Vec<_> = clip
                .frames
                .iter()
                .filter(|f| f.bot.has(ddai_clip::BotRec::BIT_BRAIN_DECIDED))
                .collect();
            assert!(
                decided.len() >= 10,
                "{kind:?}: {} frames with a brain decision",
                decided.len()
            );
            for f in &decided {
                assert_eq!(f.bot.target, 1, "{kind:?}: the target");
                assert_eq!(f.bot.brain, code, "{kind:?}: the brain kind");
                assert!(f.bot.aimed_tick > f.tick, "{kind:?}: aimed at a later tick");
                assert!(
                    f.bot.brain_us > 0 && f.bot.total_us >= f.bot.brain_us,
                    "{kind:?}: timing"
                );
            }
            assert!(
                decided
                    .iter()
                    .any(|f| f.bot.candidates > 0 && f.bot.has(ddai_clip::BotRec::BIT_SEARCHED)),
                "{kind:?}: the plan summary (candidates, searched) reaches the clip"
            );
        }
    });
}
