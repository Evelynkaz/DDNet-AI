//! Incidents: the moments of a clip worth a human's look — a port of `src/watch/incidents.ts` with
//! **real events** (task 4.3). The TS fed them `events: []` live, so `swing-at-air` and the event
//! based `death` never fired and `death` by "alive turned false" was impossible (frames were written
//! only while alive). Here the frames carry hammer hits and swings, hook attaches/releases, freeze onsets
//! and the deaths announced by the kill messages, and the bot keeps recording a few frames after its tee
//! is gone, so every kind can fire.
//!
//! Severities and thresholds are the TS ones. Deliberate differences (each also in
//! `docs/research/clips.md`):
//!
//! 1. A hammer push into the freeze is recognised from the freeze frame **or the one before it** (at
//!    25 Hz the hit and the freeze onset usually fall in different frames; the TS looked at one frame).
//! 2. `death` fires from the kill message (`Kill { victim: us }`) and, as a fallback, from our tee
//!    disappearing from the frames, once per death.
//! 3. `walking` is "the frame (or the one before) has a walk label": the nav state is always recorded.
//! 4. `swing-at-air` and `thawed-the-enemy` read the real swings; `thawed-the-enemy` uses our sent
//!    fire counter (odd = pressed) like the TS.

use std::collections::HashMap;

use ddai_physics::core::{HOOK_FLYING, HOOK_GRABBED};

use crate::format::{BotRec, ClipEvent, Frame, TeeRec};

/// `TUNING.hookLength`.
const HOOK_LENGTH: f64 = 380.0;
const SWING_REACH_PX: f64 = 96.0;
const HUMAN_REHOOK_TICKS: i32 = 10;
const HUMAN_HOLD_TICKS: i32 = 17;
const TELEPORT_PX: f64 = 96.0;
const TELEPORT_SLACK_PX: f64 = 8.0;
const THAW_LOOK_TICKS: i32 = 30;

/// The kinds, in the order the TS finds them.
pub const KINDS: [&str; 10] = [
    "self-freeze",
    "chased-into-freeze",
    "goto-into-freeze",
    "slow-rehook",
    "short-hold",
    "thawed-the-enemy",
    "swing-at-air",
    "wall-grind",
    "jitter",
    "death",
];

/// One incident.
#[derive(Debug, Clone, PartialEq)]
pub struct Incident {
    pub kind: &'static str,
    pub tick: i32,
    pub from: i32,
    pub to: i32,
    pub severity: i32,
    pub note: String,
}

fn dist(a: &TeeRec, b: &TeeRec) -> f64 {
    let (ax, ay) = a.pos();
    let (bx, by) = b.pos();
    ddai_libm::hypot(ax - bx, ay - by)
}

fn speed(t: &TeeRec) -> f64 {
    let (vx, vy) = t.vel();
    ddai_libm::hypot(vx, vy)
}

/// The nearest other tee (`free`: not frozen).
fn foe_of(f: &Frame, self_id: i32, free: bool) -> Option<&TeeRec> {
    let me = f.tee(self_id)?;
    let mut best: Option<&TeeRec> = None;
    let mut best_d = f64::INFINITY;
    for t in &f.tees {
        if t.id == self_id || (free && t.frozen) {
            continue;
        }
        let d = dist(t, me);
        if d < best_d {
            best_d = d;
            best = Some(t);
        }
    }
    best
}

fn hit_us(f: &Frame, self_id: i32) -> bool {
    f.events
        .iter()
        .any(|e| matches!(e, ClipEvent::HammerHit { to, .. } if *to == self_id))
}

/// Our input in force at the frame's tick.
fn own_input(f: &Frame) -> Option<&crate::format::InputRec> {
    f.sent.last().map(|s| &s.input)
}

/// Finds the incidents of `frames` for `self_id` (`context` ticks of lead-in and tail around each).
pub fn find_incidents(frames: &[Frame], self_id: i32, context: i32) -> Vec<Incident> {
    let mut out: Vec<Incident> = Vec::new();
    let mut clip = |tick: i32, severity: i32, kind: &'static str, note: String| {
        out.push(Incident {
            kind,
            tick,
            from: tick - context,
            to: tick + context,
            severity,
            note,
        });
    };
    let n = frames.len();
    let walking = |i: usize| -> bool { frames[i].bot.walk != 0 || (i > 0 && frames[i - 1].bot.walk != 0) };

    // 1. our own freezes: self-freeze / chased-into-freeze / goto-into-freeze.
    for i in 1..n {
        let (Some(me), Some(was)) = (frames[i].tee(self_id), frames[i - 1].tee(self_id)) else {
            continue;
        };
        if !me.frozen || was.frozen {
            continue;
        }
        let foe = foe_of(&frames[i], self_id, false);
        if hit_us(&frames[i], self_id) || hit_us(&frames[i - 1], self_id) {
            continue;
        }
        let moved = dist(me, was);
        let ticks = f64::from((frames[i].tick - frames[i - 1].tick).max(1));
        let sp = speed(was).max(speed(me));
        if moved > sp * ticks + TELEPORT_PX {
            continue;
        }
        if moved > sp * ticks + TELEPORT_SLACK_PX && speed(me) <= 1.0 {
            continue;
        }
        if moved < 1.0 && speed(was) < 0.1 {
            continue;
        }
        if frames[i].bot.has(BotRec::BIT_PLANNED_FREEZE) || frames[i - 1].bot.has(BotRec::BIT_PLANNED_FREEZE) {
            continue;
        }
        let mut held = 0;
        for f in &frames[i..] {
            match f.tee(self_id) {
                Some(t) if t.frozen => held = f.tick - frames[i].tick,
                _ => break,
            }
        }
        let foe_frozen = foe.is_some_and(|f| f.frozen);

        let back = i.saturating_sub(25);
        let mut closing = false;
        if let (Some(before), Some(chased), Some(chased_before)) = (
            frames[back].tee(self_id),
            foe_of(&frames[i], self_id, true),
            foe_of(&frames[back], self_id, true),
        ) && chased.id == chased_before.id
        {
            closing = dist(before, chased_before) - dist(me, chased) > 40.0;
        }
        let kind = if walking(i) {
            "goto-into-freeze"
        } else if closing {
            "chased-into-freeze"
        } else {
            "self-freeze"
        };
        let hooked_by_foe = frames[i]
            .tees
            .iter()
            .any(|t| t.id != self_id && t.ch.hooked_player == self_id);
        let (_, was_vy) = was.vel();
        let how = if hooked_by_foe {
            "on their hook"
        } else if was_vy < -0.5 {
            "jumped into it"
        } else if was_vy > 3.0 {
            "fell into it"
        } else {
            "walked into it"
        };
        clip(
            frames[i].tick,
            held + if foe_frozen { 0 } else { 40 } + if hooked_by_foe { -30 } else { 0 },
            kind,
            format!(
                "froze itself for {held} ticks, {how}{}; opponent was {}",
                if closing { ", while closing on the opponent" } else { "" },
                if foe_frozen { "also frozen" } else { "free" }
            ),
        );
    }

    // 2. the rope: slow re-hook, short hold.
    let mut out_since = -1;
    let mut grabbed_tee = false;
    let mut in_reach_at_throw = false;
    let mut tee_grab_start = -1;
    let mut empty_return_tick = -1;
    let mut chance_ticks = 0;
    let mut was_out = false;
    for i in 0..n {
        let Some(me) = frames[i].tee(self_id) else { continue };
        let foe = foe_of(&frames[i], self_id, false);
        let out_now = me.ch.hook_state == HOOK_FLYING || me.ch.hook_state == HOOK_GRABBED;
        let thrown = out_now && !was_out;
        was_out = out_now;
        if empty_return_tick >= 0 && i > 0 && !out_now {
            let can_throw = !me.frozen && foe.is_some_and(|f| !f.frozen && dist(me, f) <= HOOK_LENGTH);
            if can_throw {
                chance_ticks += frames[i].tick - frames[i - 1].tick;
            }
        }
        if thrown && out_since < 0 {
            out_since = frames[i].tick;
            grabbed_tee = false;
            in_reach_at_throw = foe.is_some_and(|f| dist(me, f) <= HOOK_LENGTH);
            if empty_return_tick >= 0 {
                if chance_ticks > 3 * HUMAN_REHOOK_TICKS {
                    clip(
                        empty_return_tick,
                        chance_ticks - HUMAN_REHOOK_TICKS,
                        "slow-rehook",
                        format!(
                            "{chance_ticks} ticks with the rope free, both of us free and them in reach before the next hook went out; players take {HUMAN_REHOOK_TICKS}"
                        ),
                    );
                }
                empty_return_tick = -1;
                chance_ticks = 0;
            }
        }
        if me.ch.hook_state == HOOK_GRABBED && me.ch.hooked_player >= 0 && tee_grab_start < 0 {
            tee_grab_start = frames[i].tick;
            grabbed_tee = true;
        }
        if tee_grab_start >= 0 && me.ch.hooked_player < 0 {
            let held = frames[i].tick - tee_grab_start;
            if held < HUMAN_HOLD_TICKS / 2 && foe.is_some_and(|f| !f.frozen) {
                clip(
                    tee_grab_start,
                    HUMAN_HOLD_TICKS - held,
                    "short-hold",
                    format!("let a tee off the hook after {held} ticks; players hold {HUMAN_HOLD_TICKS}"),
                );
            }
            tee_grab_start = -1;
        }
        if out_since >= 0 && !out_now {
            if !grabbed_tee && in_reach_at_throw {
                empty_return_tick = frames[i].tick;
                chance_ticks = 0;
            }
            out_since = -1;
        }
    }

    // 3. hammering a frozen opponent thaws them.
    for i in 0..n {
        let (Some(me), Some(foe), Some(inp)) = (
            frames[i].tee(self_id),
            foe_of(&frames[i], self_id, false),
            own_input(&frames[i]),
        ) else {
            continue;
        };
        if inp.fire & 1 == 0 || !foe.frozen || me.frozen {
            continue;
        }
        if dist(me, foe) > SWING_REACH_PX {
            continue;
        }
        if i > 0 && own_input(&frames[i - 1]).is_some_and(|b| b.fire & 1 != 0) {
            continue;
        }
        let (mut free_after, mut still_frozen) = (false, false);
        let mut j = i + 1;
        while j < n && frames[j].tick - frames[i].tick <= THAW_LOOK_TICKS {
            if let Some(later) = frames[j].tee(foe.id) {
                if later.frozen {
                    still_frozen = true;
                } else if frames[j].tick - frames[i].tick >= THAW_LOOK_TICKS / 2 {
                    free_after = true;
                }
            }
            j += 1;
        }
        if free_after && !still_frozen {
            clip(
                frames[i].tick,
                60,
                "thawed-the-enemy",
                format!(
                    "hammered a FROZEN opponent at {:.0}px and they were free {THAW_LOOK_TICKS} ticks later -- a hammer unfreezes its target",
                    dist(me, foe)
                ),
            );
        }
    }

    // 4. swings at nothing (real hammer events).
    for f in frames {
        let Some(me) = f.tee(self_id) else { continue };
        let foe = foe_of(f, self_id, false);
        for e in &f.events {
            let ClipEvent::HammerFire { from, hits } = e else {
                continue;
            };
            if *from != self_id || *hits != 0 {
                continue;
            }
            let d = foe.map_or(f64::INFINITY, |t| dist(me, t));
            if d > SWING_REACH_PX {
                let away = if d.is_finite() {
                    format!("{d:.0}px")
                } else {
                    "nowhere".to_string()
                };
                clip(
                    f.tick,
                    5,
                    "swing-at-air",
                    format!("swung with the opponent {away} away, hammer reaches {SWING_REACH_PX}px"),
                );
            }
        }
    }

    // 5. wall grind: a direction held without moving.
    let mut stuck_from = -1;
    for f in frames {
        let me = f.tee(self_id);
        let inp = own_input(f);
        let stuck = matches!((me, inp), (Some(m), Some(i)) if i.direction != 0 && m.vel().0.abs() < 0.2 && !m.frozen);
        if stuck && stuck_from < 0 {
            stuck_from = f.tick;
        }
        if !stuck && stuck_from >= 0 {
            let held = f.tick - stuck_from;
            if held >= 20 {
                clip(
                    stuck_from,
                    held,
                    "wall-grind",
                    format!("held a direction for {held} ticks without moving"),
                );
            }
            stuck_from = -1;
        }
    }

    // 6. jitter: six direction changes within 25 ticks.
    {
        let mut flips: Vec<i32> = Vec::new();
        let mut last: Option<i32> = None;
        for f in frames {
            let Some(inp) = own_input(f) else { continue };
            if last.is_some_and(|l| inp.direction != l) {
                flips.push(f.tick);
            }
            last = Some(inp.direction);
        }
        let mut i = 0;
        while i + 5 < flips.len() {
            let span = flips[i + 5] - flips[i];
            if span <= 25 {
                clip(
                    flips[i],
                    6 + (25 - span),
                    "jitter",
                    format!("6 direction changes in {span} ticks"),
                );
                i += 5;
            }
            i += 1;
        }
    }

    // 7. deaths: the kill message, then our tee vanishing from the frames.
    for f in frames {
        for e in &f.events {
            if let ClipEvent::Kill { victim, .. } = e
                && *victim == self_id
            {
                clip(f.tick, 150, "death", "died".to_string());
            }
        }
    }
    for i in 1..n {
        let (was, now) = (frames[i - 1].tee(self_id), frames[i].tee(self_id));
        let already = out
            .iter()
            .any(|o| o.kind == "death" && (o.tick - frames[i].tick).abs() < 10);
        if !already && was.is_some() && now.is_none() {
            out.push(Incident {
                kind: "death",
                tick: frames[i].tick,
                from: frames[i].tick - context,
                to: frames[i].tick + context,
                severity: 150,
                note: "died".to_string(),
            });
        }
    }

    out.sort_by(|a, b| b.severity.cmp(&a.severity).then(a.tick.cmp(&b.tick)));
    out
}

/// `mergeOverlapping`: incidents closer than 25 ticks are one; the kinds of the swallowed ones go in the note.
pub fn merge_overlapping(incidents: Vec<Incident>) -> Vec<Incident> {
    let mut kept: Vec<Incident> = Vec::new();
    for inc in incidents {
        match kept.iter_mut().find(|k| (k.tick - inc.tick).abs() < 25) {
            None => kept.push(inc),
            Some(clash) => {
                if clash.kind != inc.kind && !clash.note.contains(&format!("also {}", inc.kind)) {
                    clash.note.push_str(&format!("; also {}", inc.kind));
                }
            }
        }
    }
    kept
}

/// One row of [`summarise`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Summary {
    pub kind: &'static str,
    pub count: usize,
    pub worst: i32,
}

/// `summarise`: per kind, by `count * worst` descending.
pub fn summarise(incidents: &[Incident]) -> Vec<Summary> {
    let mut by: HashMap<&'static str, Summary> = HashMap::new();
    for i in incidents {
        let row = by.entry(i.kind).or_insert(Summary {
            kind: i.kind,
            count: 0,
            worst: 0,
        });
        row.count += 1;
        row.worst = row.worst.max(i.severity);
    }
    let mut rows: Vec<Summary> = by.into_values().collect();
    rows.sort_by(|a, b| {
        (b.count as i64 * i64::from(b.worst))
            .cmp(&(a.count as i64 * i64::from(a.worst)))
            .then(a.kind.cmp(b.kind))
    });
    rows
}
