//! Each incident kind on a synthetic frame sequence (task 4.3 acceptance criterion 2), with the real
//! events the TS never had live.

use ddai_clip::format::*;
use ddai_clip::incidents::{Incident, find_incidents, merge_overlapping, summarise};

const ME: i32 = 0;
const FOE: i32 = 1;
/// `HOOK_IDLE`, `HOOK_FLYING`, `HOOK_GRABBED` of the physics.
const IDLE: i32 = 0;
const FLYING: i32 = 4;
const GRABBED: i32 = 5;

fn tee(id: i32, x: i32, y: i32) -> TeeRec {
    TeeRec {
        id,
        ch: CharRec {
            x,
            y,
            hooked_player: -1,
            hook_state: IDLE,
            ..CharRec::default()
        },
        ..TeeRec::default()
    }
}

fn frozen(mut t: TeeRec) -> TeeRec {
    t.frozen = true;
    t.freeze_left = 150;
    t
}

fn with_vel(mut t: TeeRec, vx: f64, vy: f64) -> TeeRec {
    t.ch.vel_x = (vx * 256.0) as i32;
    t.ch.vel_y = (vy * 256.0) as i32;
    t
}

fn hook(mut t: TeeRec, state: i32, on: i32) -> TeeRec {
    t.ch.hook_state = state;
    t.ch.hooked_player = on;
    t
}

fn input(direction: i32, fire: i32) -> SentRec {
    SentRec {
        tick: 0,
        input: InputRec {
            direction,
            fire,
            ..InputRec::default()
        },
        timing_known: true,
    }
}

/// Frames two ticks apart from tick 1000; `make(i)` gives the frame's tees, our input and events.
fn frames(
    n: usize,
    mut make: impl FnMut(usize) -> (Vec<TeeRec>, Option<SentRec>, Vec<ClipEvent>, BotRec),
) -> Vec<Frame> {
    (0..n)
        .map(|i| {
            let (tees, sent, events, bot) = make(i);
            let tick = 1000 + 2 * i as i32;
            Frame {
                tick,
                own_alive: tees.iter().any(|t| t.id == ME),
                tees,
                sent: sent
                    .map(|mut s| {
                        s.tick = tick;
                        s
                    })
                    .into_iter()
                    .collect(),
                events,
                bot,
                ..Frame::default()
            }
        })
        .collect()
}

fn plain(n: usize, me: impl Fn(usize) -> TeeRec, foe: impl Fn(usize) -> TeeRec) -> Vec<Frame> {
    frames(n, |i| {
        (vec![me(i), foe(i)], Some(input(0, 0)), vec![], BotRec::default())
    })
}

fn kinds(found: &[Incident]) -> Vec<&'static str> {
    found.iter().map(|i| i.kind).collect()
}

fn of<'a>(found: &'a [Incident], kind: &str) -> Option<&'a Incident> {
    found.iter().find(|i| i.kind == kind)
}

#[test]
fn self_freeze_is_our_freeze_with_a_free_opponent() {
    // We walk (vx 4) for 10 frames, then freeze for 6 frames (150 ticks thaw is longer than the clip).
    let f = plain(
        30,
        |i| {
            let t = with_vel(tee(ME, 100 + 8 * i as i32, 500), 4.0, 0.0);
            if i >= 10 { frozen(t) } else { t }
        },
        |_| tee(FOE, -3000, 500), // behind us: walking away, not closing in
    );
    let found = find_incidents(&f, ME, 40);
    let inc = of(&found, "self-freeze").expect("self-freeze");
    assert_eq!(inc.tick, 1000 + 2 * 10);
    // held = ticks from the onset to the last frozen frame (19 frames * 2 ticks) + 40 (the opponent is free).
    assert_eq!(inc.severity, 2 * 19 + 40);
    assert!(
        inc.note.contains("walked into it") && inc.note.contains("opponent was free"),
        "{}",
        inc.note
    );
    assert_eq!((inc.from, inc.to), (inc.tick - 40, inc.tick + 40));
}

#[test]
fn an_also_frozen_opponent_and_their_hook_lower_the_severity() {
    let f = plain(
        20,
        |i| {
            let t = with_vel(tee(ME, 100 + 8 * i as i32, 500), 4.0, 0.0);
            if i >= 10 { frozen(t) } else { t }
        },
        |_| frozen(hook(tee(FOE, 400, 500), GRABBED, ME)),
    );
    let inc = find_incidents(&f, ME, 40)
        .into_iter()
        .find(|i| i.kind == "self-freeze")
        .unwrap();
    assert_eq!(
        inc.severity,
        2 * 9 - 30,
        "no +40 for a frozen opponent, -30 for being on their hook"
    );
    assert!(inc.note.contains("on their hook") && inc.note.contains("also frozen"));
}

#[test]
fn chased_into_freeze_needs_the_same_free_opponent_closing_by_40_px() {
    // The opponent runs at us from 600 px to 100 px over 30 frames; we stand still then freeze at frame 30.
    let f = plain(
        40,
        |i| {
            let t = with_vel(tee(ME, 1000, 500), 0.5, 0.0);
            if i >= 30 { frozen(t) } else { t }
        },
        |i| tee(FOE, 1000 + 600 - (i as i32 * 17).min(500), 500),
    );
    let found = find_incidents(&f, ME, 40);
    let inc = of(&found, "chased-into-freeze").unwrap_or_else(|| panic!("{:?}", kinds(&found)));
    assert!(inc.note.contains("closing on the opponent"));
}

#[test]
fn a_freeze_while_a_walk_label_is_set_is_goto_into_freeze() {
    let f = frames(20, |i| {
        let t = with_vel(tee(ME, 100 + 8 * i as i32, 500), 4.0, 0.0);
        let me = if i >= 10 { frozen(t) } else { t };
        let bot = BotRec {
            walk: 1,
            ..BotRec::default()
        };
        (vec![me, tee(FOE, 2000, 500)], Some(input(1, 0)), vec![], bot)
    });
    let found = find_incidents(&f, ME, 40);
    assert!(of(&found, "goto-into-freeze").is_some(), "{:?}", kinds(&found));
    assert!(of(&found, "self-freeze").is_none());
}

#[test]
fn a_planned_freeze_a_hammer_push_and_a_teleport_are_not_incidents() {
    let base = |i: usize| {
        let t = with_vel(tee(ME, 100 + 8 * i as i32, 500), 4.0, 0.0);
        if i >= 10 { frozen(t) } else { t }
    };
    // Planned (crossing / plannedFreeze flag on the frame or the one before).
    let planned = frames(20, |i| {
        let bot = BotRec {
            flags: if i == 9 { BotRec::BIT_PLANNED_FREEZE } else { 0 },
            ..BotRec::default()
        };
        (vec![base(i), tee(FOE, 2000, 500)], Some(input(1, 0)), vec![], bot)
    });
    assert!(
        of(&find_incidents(&planned, ME, 40), "self-freeze").is_none(),
        "planned freeze"
    );
    // Pushed by a hammer: the hit is in the frame before the freeze (the 25 Hz sampling puts them apart).
    let pushed = frames(20, |i| {
        let ev = if i == 9 {
            vec![ClipEvent::HammerHit { from: FOE, to: ME }]
        } else {
            vec![]
        };
        (
            vec![base(i), tee(FOE, 300, 500)],
            Some(input(1, 0)),
            ev,
            BotRec::default(),
        )
    });
    assert!(
        of(&find_incidents(&pushed, ME, 40), "self-freeze").is_none(),
        "hammer push"
    );
    // A teleport: we jump 800 px in the frame we froze.
    let tele = plain(
        20,
        |i| {
            let t = with_vel(tee(ME, if i >= 10 { 900 } else { 100 }, 500), 4.0, 0.0);
            if i >= 10 { frozen(t) } else { t }
        },
        |_| tee(FOE, 2000, 500),
    );
    assert!(of(&find_incidents(&tele, ME, 40), "self-freeze").is_none(), "teleport");
    // Standing still when frozen (nothing moved, nothing was moving): a thaw-and-refreeze, not an incident.
    let still = plain(
        20,
        |i| {
            if i >= 10 {
                frozen(tee(ME, 100, 500))
            } else {
                tee(ME, 100, 500)
            }
        },
        |_| tee(FOE, 2000, 500),
    );
    assert!(
        of(&find_incidents(&still, ME, 40), "self-freeze").is_none(),
        "standing still"
    );
}

#[test]
fn slow_rehook_is_a_free_rope_with_both_free_and_them_in_reach_for_over_30_ticks() {
    // Frames 0-4 rope out (flying then idle again), foe 200 px away and free; the rope stays in 60 ticks, then goes out.
    let f = plain(
        60,
        |i| {
            let state = if (2..4).contains(&i) || i >= 40 { FLYING } else { IDLE };
            hook(tee(ME, 100, 500), state, -1)
        },
        |_| tee(FOE, 300, 500),
    );
    let found = find_incidents(&f, ME, 40);
    let inc = of(&found, "slow-rehook").unwrap_or_else(|| panic!("{:?}", kinds(&found)));
    assert!(
        inc.severity > 30 - 10 && inc.note.contains("ticks with the rope free"),
        "{inc:?}"
    );
    // A quick re-hook (10 ticks) is human speed: no incident.
    let quick = plain(
        20,
        |i| {
            hook(
                tee(ME, 100, 500),
                if (2..4).contains(&i) || i >= 9 { FLYING } else { IDLE },
                -1,
            )
        },
        |_| tee(FOE, 300, 500),
    );
    assert!(of(&find_incidents(&quick, ME, 40), "slow-rehook").is_none());
}

#[test]
fn short_hold_is_letting_a_free_tee_go_within_8_ticks() {
    let f = plain(
        20,
        |i| {
            if (4..7).contains(&i) {
                hook(tee(ME, 100, 500), GRABBED, FOE)
            } else {
                hook(tee(ME, 100, 500), IDLE, -1)
            }
        },
        |_| tee(FOE, 250, 500),
    );
    let found = find_incidents(&f, ME, 40);
    let inc = of(&found, "short-hold").unwrap_or_else(|| panic!("{:?}", kinds(&found)));
    assert_eq!(inc.severity, 17 - 6, "HUMAN_HOLD_TICKS - held (6 ticks)");
    // Holding a frozen tee briefly is fine.
    let frozen_foe = plain(
        20,
        |i| {
            if (4..7).contains(&i) {
                hook(tee(ME, 100, 500), GRABBED, FOE)
            } else {
                tee(ME, 100, 500)
            }
        },
        |_| frozen(tee(FOE, 250, 500)),
    );
    assert!(of(&find_incidents(&frozen_foe, ME, 40), "short-hold").is_none());
}

#[test]
fn thawed_the_enemy_is_a_hammer_press_on_a_frozen_foe_that_is_free_15_ticks_later() {
    let f = frames(30, |i| {
        let foe = if i < 4 {
            frozen(tee(FOE, 150, 500))
        } else {
            tee(FOE, 150, 500)
        };
        let press = if i == 3 { 1 } else { 0 };
        (
            vec![tee(ME, 100, 500), foe],
            Some(input(0, press)),
            vec![],
            BotRec::default(),
        )
    });
    let found = find_incidents(&f, ME, 40);
    let inc = of(&found, "thawed-the-enemy").unwrap_or_else(|| panic!("{:?}", kinds(&found)));
    assert_eq!((inc.tick, inc.severity), (1006, 60));
    // Still frozen 20 ticks later: it did not thaw.
    let still = frames(30, |i| {
        let press = if i == 3 { 1 } else { 0 };
        (
            vec![tee(ME, 100, 500), frozen(tee(FOE, 150, 500))],
            Some(input(0, press)),
            vec![],
            BotRec::default(),
        )
    });
    assert!(of(&find_incidents(&still, ME, 40), "thawed-the-enemy").is_none());
}

#[test]
fn swing_at_air_comes_from_a_real_hammer_event_with_no_hit_and_nobody_in_reach() {
    let f = frames(10, |i| {
        let ev = match i {
            3 => vec![ClipEvent::HammerFire { from: ME, hits: 0 }],
            5 => vec![ClipEvent::HammerFire { from: ME, hits: 1 }],
            6 => vec![ClipEvent::HammerFire { from: FOE, hits: 0 }],
            _ => vec![],
        };
        (
            vec![tee(ME, 100, 500), tee(FOE, 400, 500)],
            Some(input(0, 0)),
            ev,
            BotRec::default(),
        )
    });
    let found = find_incidents(&f, ME, 40);
    let air: Vec<_> = found.iter().filter(|i| i.kind == "swing-at-air").collect();
    assert_eq!(air.len(), 1, "only our own empty swing: {found:?}");
    assert_eq!((air[0].tick, air[0].severity), (1006, 5));
    // An empty swing with the foe within the 96 px reach is just a miss.
    let near = frames(10, |i| {
        let ev = if i == 3 {
            vec![ClipEvent::HammerFire { from: ME, hits: 0 }]
        } else {
            vec![]
        };
        (
            vec![tee(ME, 100, 500), tee(FOE, 150, 500)],
            Some(input(0, 0)),
            ev,
            BotRec::default(),
        )
    });
    assert!(of(&find_incidents(&near, ME, 40), "swing-at-air").is_none());
}

#[test]
fn wall_grind_is_a_held_direction_without_moving_for_20_ticks() {
    let f = frames(30, |i| {
        let dir = if (5..20).contains(&i) { 1 } else { 0 };
        (
            vec![with_vel(tee(ME, 100, 500), 0.0, 0.0), tee(FOE, 2000, 500)],
            Some(input(dir, 0)),
            vec![],
            BotRec::default(),
        )
    });
    let found = find_incidents(&f, ME, 40);
    let inc = of(&found, "wall-grind").unwrap_or_else(|| panic!("{:?}", kinds(&found)));
    assert_eq!((inc.tick, inc.severity), (1010, 30), "15 frames of 2 ticks");
    // Moving at speed is not grinding; neither is a short press.
    let moving = frames(30, |_| {
        (
            vec![with_vel(tee(ME, 100, 500), 5.0, 0.0), tee(FOE, 2000, 500)],
            Some(input(1, 0)),
            vec![],
            BotRec::default(),
        )
    });
    assert!(of(&find_incidents(&moving, ME, 40), "wall-grind").is_none());
}

#[test]
fn jitter_is_six_direction_changes_within_25_ticks() {
    let f = frames(30, |i| {
        let dir = if i < 4 {
            0
        } else if i < 14 {
            if i % 2 == 0 { 1 } else { -1 }
        } else {
            1
        };
        (
            vec![
                with_vel(tee(ME, 100 + 50 * i as i32, 500), 5.0, 0.0),
                tee(FOE, 4000, 500),
            ],
            Some(input(dir, 0)),
            vec![],
            BotRec::default(),
        )
    });
    let found = find_incidents(&f, ME, 40);
    let inc = of(&found, "jitter").unwrap_or_else(|| panic!("{:?}", kinds(&found)));
    assert!(
        inc.note.starts_with("6 direction changes in") && inc.severity >= 6,
        "{inc:?}"
    );
    let calm = frames(30, |i| {
        (
            vec![
                with_vel(tee(ME, 100 + 50 * i as i32, 500), 5.0, 0.0),
                tee(FOE, 4000, 500),
            ],
            Some(input(if i % 10 == 0 { -1 } else { 1 }, 0)),
            vec![],
            BotRec::default(),
        )
    });
    assert!(
        of(&find_incidents(&calm, ME, 40), "jitter").is_none(),
        "a flip pair every 20 ticks is not jitter"
    );
}

#[test]
fn death_comes_from_the_kill_message_and_from_the_tee_vanishing_once_each() {
    // The kill message names us.
    let by_event = frames(20, |i| {
        let ev = if i == 12 {
            vec![ClipEvent::Kill {
                killer: -1,
                victim: ME,
                weapon: -2,
            }]
        } else {
            vec![]
        };
        let mut tees = vec![tee(FOE, 400, 500)];
        if i < 12 {
            tees.push(tee(ME, 100, 500));
        }
        (tees, Some(input(0, 0)), ev, BotRec::default())
    });
    let found = find_incidents(&by_event, ME, 40);
    let deaths: Vec<_> = found.iter().filter(|i| i.kind == "death").collect();
    assert_eq!(
        deaths.len(),
        1,
        "the message and the vanishing are the same death: {deaths:?}"
    );
    assert_eq!(deaths[0].severity, 150);
    // Someone else's death is not ours.
    let other = frames(20, |i| {
        let ev = if i == 12 {
            vec![ClipEvent::Kill {
                killer: ME,
                victim: FOE,
                weapon: 0,
            }]
        } else {
            vec![]
        };
        (
            vec![tee(ME, 100, 500), tee(FOE, 400, 500)],
            Some(input(0, 0)),
            ev,
            BotRec::default(),
        )
    });
    assert!(of(&find_incidents(&other, ME, 40), "death").is_none());
    // Only the vanishing (no message, e.g. a lost one).
    let vanish = frames(20, |i| {
        let mut tees = vec![tee(FOE, 400, 500)];
        if i < 12 {
            tees.push(tee(ME, 100, 500));
        }
        (tees, None, vec![], BotRec::default())
    });
    assert!(of(&find_incidents(&vanish, ME, 40), "death").is_some());
}

#[test]
fn incidents_sort_by_severity_merge_when_close_and_summarise_by_weight() {
    let mk = |kind: &'static str, tick, severity| Incident {
        kind,
        tick,
        from: tick - 40,
        to: tick + 40,
        severity,
        note: String::new(),
    };
    let merged = merge_overlapping(vec![
        mk("death", 100, 150),
        mk("jitter", 110, 20),
        mk("jitter", 112, 9),
        mk("wall-grind", 300, 40),
    ]);
    assert_eq!(merged.len(), 2);
    assert_eq!(merged[0].note, "; also jitter", "the swallowed kind once");
    let rows = summarise(&[
        mk("death", 1, 150),
        mk("jitter", 2, 20),
        mk("jitter", 3, 30),
        mk("wall-grind", 4, 40),
    ]);
    assert_eq!(rows[0].kind, "death");
    assert_eq!((rows[1].kind, rows[1].count, rows[1].worst), ("jitter", 2, 30));
}
