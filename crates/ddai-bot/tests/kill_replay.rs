//! Task 4.12 (D-108): the offline replay of a clip's self-kills under the smart policy (`ddai_bot::kill_replay`), on clips made here
//! (a pit, a freeze that runs out) and, when the data is there, on the real clips of 2026-10-06 (read only).

mod support;

use std::path::PathBuf;
use std::sync::Arc;

use ddai_bot::kill_replay::{Fate, Outcome, analyse};
use ddai_bot::smartkill::SmartWhy;
use ddai_clip::format::{
    BotRec, CharRec, Clip, ClipEvent, ClipHeader, ClipReason, DdRec, Frame, KillWhy, PlayerTag, TeeRec,
};
use ddai_physics::map::MapData;
use support::{FLOOR_Y, FREEZE, room};

fn header() -> ClipHeader {
    ClipHeader {
        map_name: "room".into(),
        map_sha256: [0; 32],
        own_id: 0,
        brain: "test".into(),
        reason: ClipReason {
            kind: "manual".into(),
            severity: 0,
            tick: 0,
            note: String::new(),
        },
        labels: vec![String::new()],
        players: vec![
            PlayerTag {
                id: 0,
                tag: "c0-aaaaaaaa".into(),
            },
            PlayerTag {
                id: 1,
                tag: "c1-bbbbbbbb".into(),
            },
        ],
        tuning: Vec::new(),
        teams: Vec::new(),
        world_seed: 1,
    }
}

/// Our tee at `(x, y)` px at `tick`, frozen until `freeze_end` (`0`: free, `-1`: deep).
fn tee_at(tick: i32, x: i32, y: i32, freeze_end: i32) -> TeeRec {
    let frozen = freeze_end != 0;
    TeeRec {
        id: 0,
        ch: CharRec {
            tick,
            x,
            y,
            player_flags: 1,
            health: 10,
            ammo_count: -1,
            hook_x: x,
            hook_y: y,
            ..CharRec::default()
        },
        dd: Some(DdRec {
            freeze_end,
            jumps: 2,
            tele_checkpoint: -1,
            jumped_total: -1,
            ninja_activation_tick: -1,
            freeze_start: if frozen { tick } else { -1 },
            tune_zone_override: -1,
            ..DdRec::default()
        }),
        frozen,
        deep_frozen: freeze_end == -1,
        freeze_left: if freeze_end > tick { freeze_end - tick } else { 0 },
    }
}

fn frame(tee: TeeRec, kill: Option<KillWhy>) -> Frame {
    Frame {
        tick: tee.ch.tick,
        own_alive: true,
        tees: vec![tee],
        events: kill
            .map(|why| ClipEvent::KillSent { why: why as u8 })
            .into_iter()
            .collect(),
        bot: BotRec::default(),
        ..Frame::default()
    }
}

fn clip(frames: Vec<Frame>) -> Clip {
    Clip {
        header: header(),
        frames,
    }
}

fn map(extra: &[(u32, u32, u8)]) -> Arc<MapData> {
    room(extra)
}

fn run(clip: &Clip, m: &Arc<MapData>) -> Vec<ddai_bot::kill_replay::Case> {
    let c = clip.clone();
    let m = Arc::clone(m);
    // The planner's helpers copy whole worlds by value: a big stack, like the bot's own thread.
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || analyse(&c, &m, "synthetic"))
        .unwrap()
        .join()
        .unwrap()
}

/// A tee frozen for 300 ticks in a freeze pit (the freeze renewed every second) and the legacy kill at 200: the smart policy kills at 50.
#[test]
fn a_pit_kill_is_sent_150_ticks_sooner() {
    let m = map(&[(35, 38, FREEZE), (35, 37, FREEZE)]);
    let (x, y) = (35 * 32 + 16, 37 * 32 + 16);
    let frames: Vec<Frame> = (500..650)
        .map(|i| {
            let tick = i * 2;
            let kill = (tick == 1000 + 200).then_some(KillWhy::Unstick);
            frame(tee_at(tick, x, y, tick + 100), kill)
        })
        .collect();
    // Frozen from tick 1000 (the first frozen frame): make the earlier frames free.
    let mut frames = frames;
    for f in frames.iter_mut().filter(|f| f.tick < 1000) {
        f.tees[0] = tee_at(f.tick, x, y - 40, 0);
    }
    let cases = run(&clip(frames), &m);
    assert_eq!(cases.len(), 1, "{cases:?}");
    let c = &cases[0];
    assert_eq!((c.tick, c.why_name()), (1200, "unstick"));
    assert_eq!(c.frozen_for, 200);
    match &c.verdict {
        Outcome::Earlier { ticks, why } => {
            assert_eq!(*why, SmartWhy::NoExit);
            assert!((148..=152).contains(ticks), "{ticks}: 200 - 50");
        }
        other => panic!("{other:?}\n{:#?}", c.trace),
    }
}

/// A freeze that ends by itself 70 ticks after a (wayblock) kill was sent: the smart policy skips the kill, the tee thaws on its own.
#[test]
fn a_kill_in_a_freeze_that_runs_out_is_skipped_and_the_tee_thaws() {
    let m = map(&[]);
    let frames: Vec<Frame> = (500..620)
        .map(|i| {
            let tick = i * 2;
            let end = 1130;
            let freeze_end = if tick < end { end } else { 0 };
            let kill = (tick == 1060).then_some(KillWhy::WayBlockLying);
            frame(tee_at(tick, 1000, FLOOR_Y, freeze_end), kill)
        })
        .collect();
    let mut frames = frames;
    for f in frames.iter_mut().filter(|f| f.tick < 1000) {
        f.tees[0] = tee_at(f.tick, 1000, FLOOR_Y, 0);
    }
    // Frozen from tick 1000 for 130 ticks: the freeze_end of those frames is 1130, `freeze_start` 1000.
    let cases = run(&clip(frames), &m);
    assert_eq!(cases.len(), 1, "{cases:?}");
    match &cases[0].verdict {
        Outcome::Skipped {
            why,
            fate: Fate::Thawed { after },
        } => {
            assert_eq!(*why, Some(SmartWhy::ThawSoon));
            assert!(
                (60..=80).contains(after),
                "thawed {after} ticks after the kill that was skipped"
            );
        }
        other => panic!("{other:?}\n{:#?}", cases[0].trace),
    }
}

/// The same freeze, but the clip ends before it does: the replay says so (still frozen) instead of guessing.
#[test]
fn a_skipped_kill_whose_outcome_the_clip_cannot_show_is_reported_as_such() {
    let m = map(&[]);
    let frames: Vec<Frame> = (500..560)
        .map(|i| {
            let tick = i * 2;
            let kill = (tick == 1060).then_some(KillWhy::WayBlockLying);
            frame(tee_at(tick, 1000, FLOOR_Y, if tick < 1000 { 0 } else { 1130 }), kill)
        })
        .collect();
    let cases = run(&clip(frames), &m);
    match &cases[0].verdict {
        Outcome::Skipped {
            fate: Fate::StillFrozen { ticks },
            ..
        } => assert!(*ticks <= 60, "{ticks}"),
        other => panic!("{other:?}"),
    }
}

/// A deep freeze lasts forever: the smart policy kills as soon as it is sure, the legacy timer only at 400.
#[test]
fn a_deep_freeze_kill_is_sent_350_ticks_sooner() {
    let m = map(&[]);
    let frames: Vec<Frame> = (500..750)
        .map(|i| {
            let tick = i * 2;
            let kill = (tick == 1400).then_some(KillWhy::Unstick);
            frame(tee_at(tick, 1000, FLOOR_Y, if tick < 1000 { 0 } else { -1 }), kill)
        })
        .collect();
    let cases = run(&clip(frames), &m);
    match &cases[0].verdict {
        Outcome::Earlier { ticks, why } => {
            assert_eq!(*why, SmartWhy::DeepFreeze);
            assert!((348..=352).contains(ticks), "{ticks}");
        }
        other => panic!("{other:?}"),
    }
}

/// A route's respawn step and the owner's `!kill` are listed, not judged.
#[test]
fn navigation_and_console_kills_are_not_judged() {
    let m = map(&[]);
    let frames: Vec<Frame> = (500..520)
        .map(|i| {
            let tick = i * 2;
            let kill = match tick {
                1010 => Some(KillWhy::Navigation),
                1020 => Some(KillWhy::Console),
                1030 => Some(KillWhy::Trek),
                _ => None,
            };
            frame(tee_at(tick, 1000, FLOOR_Y, 0), kill)
        })
        .collect();
    let cases = run(&clip(frames), &m);
    assert_eq!(cases.len(), 3);
    assert!(
        cases.iter().all(|c| matches!(c.verdict, Outcome::NotJudged(_))),
        "{cases:?}"
    );
}

// ---- the real clips of 2026-10-06 (read only; skipped when the data is not there) ----------------------------------------------------

fn real_clips() -> Vec<PathBuf> {
    let home = PathBuf::from(std::env::var_os("HOME").unwrap_or_default());
    let dir = std::env::var_os("DDAI_BOT_CLIPS").map_or_else(|| home.join("aiddnet/data/bot/clips"), PathBuf::from);
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "clip"))
        .collect();
    v.sort();
    v
}

/// What the replay of the real clips must keep true whatever their number: every judged kill is one of same / earlier / skipped, the smart
/// policy is never later than the legacy upper bounds, and no skipped kill leaves the tee frozen past them.
#[test]
fn on_the_real_clips_no_skipped_kill_leaves_the_bot_frozen_past_the_upper_bounds() {
    let home = PathBuf::from(std::env::var_os("HOME").unwrap_or_default());
    let cache = home.join("aiddnet/data/maps/cache");
    let (mut judged, mut earlier, mut skipped) = (0, 0, 0);
    for p in real_clips() {
        let c = Clip::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        let Some(bytes) = ddai_client::map_cache::read_cached(&cache, &c.header.map_name, &c.header.map_sha256) else {
            continue;
        };
        let m = Arc::new(ddai_map::load_map(&bytes).expect("the cached map parses").data);
        for case in run(&c, &m) {
            match case.verdict {
                Outcome::Same { .. } => judged += 1,
                Outcome::Earlier { ticks, .. } => {
                    judged += 1;
                    earlier += 1;
                    assert!(ticks > 0 && ticks <= 400, "{}: {ticks}", case.clip);
                }
                Outcome::Skipped { fate, .. } => {
                    judged += 1;
                    skipped += 1;
                    if let Fate::StillFrozen { ticks } = fate {
                        // The clip ends while the bot waits: the upper bounds (400 frozen) are still ahead, never exceeded.
                        assert!(
                            ticks < 400,
                            "{}: still frozen {ticks} ticks after the skipped kill",
                            case.clip
                        );
                    }
                }
                Outcome::NotJudged(_) | Outcome::LegacyNotReproduced => {}
            }
        }
    }
    eprintln!("real clips: judged {judged}, earlier {earlier}, skipped {skipped}");
}
