//! Real human inputs from demos (task 3.24, E-039): the `Sv_PreInput` messages the DDNet client
//! records ([`crate::ingest::PreInputMsg`]) as a per-player timeline that the physics replay of
//! [`crate::pipeline`] can use instead of the inputs reconstructed from snapshots.
//!
//! **What a message is** (`server.cpp:1934-1980`): the server forwards a client's input to the
//! other clients of its team when it *changes* (direction, jump, fire counter, hook, weapon
//! fields; the aim is attached to a message that is sent anyway, never a reason to send one), with
//! the tick it will be applied at. So a player's input at tick `t` is the newest message with
//! `intended_tick <= t`; between messages the input is unchanged, and the aim is the aim of the
//! last message (exact at the moments it matters: a hook or a swing starts with a message).
//! Messages are not vital: a lost one leaves the old state in force until the next change, which
//! the physics replay then shows as a mismatch (the dataset keeps only replay-confirmed samples).
//!
//! No names: tracks are keyed by the anonymous per-demo label.

use std::collections::{BTreeMap, HashMap};

use ddai_demo::Demo;
use serde::{Deserialize, Serialize};

use crate::ingest::{FrameSource, LabelTracker, PreInputMsg};
use ddai_recorder::format::Frame;

/// One input state of a player, valid from `intended` until the next event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputEvent {
    /// The tick the input is applied at (`IntendedTick`).
    pub intended: i32,
    /// Demo tick the message was recorded at (`intended - arrived` is the lead).
    pub arrived: i32,
    pub direction: i8,
    pub jump: bool,
    pub hook: bool,
    /// The press counter (odd = held), as sent.
    pub fire: i32,
    pub aim: [i32; 2],
}

/// How many ticks before a (re)appearance an event may still count as made after it (messages
/// are sent in the snapshot's own tick or a little earlier).
pub const RESTART_SLACK: i32 = 4;

/// The input timeline of one player: events sorted by `intended` (stable in arrival order).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Track {
    pub events: Vec<InputEvent>,
    /// Ticks of the snapshots in which the player (re)appeared after being absent (clipped out of
    /// view, dead, away), ascending. The server forwards inputs only while the owner is in view,
    /// so what happened while it was away is unknown: an event older than the latest restart says
    /// nothing about the state after it.
    pub restarts: Vec<i32>,
    /// Ticks at which the snapshot's own core contradicted the input in force (its direction differed from the
    /// event's, twice running, for a free player): a message of that player was lost, so that event says nothing
    /// from there on ([`snapshot_trust`]). Ascending.
    #[serde(default)]
    pub desyncs: Vec<i32>,
}

impl Track {
    /// The input in force at world tick `t`: the newest event with `intended <= t`, `None` before
    /// the first one and when the player has re-appeared since that event ([`Track::restarts`]):
    /// then the real input is unknown until its next message.
    pub fn at(&self, t: i32) -> Option<&InputEvent> {
        let n = self.events.partition_point(|e| e.intended <= t);
        let e = self.events.get(n.checked_sub(1)?)?;
        let r = self.restarts.partition_point(|&r| r <= t);
        if let Some(restart) = r.checked_sub(1).map(|i| self.restarts[i])
            && e.intended < restart - RESTART_SLACK
        {
            return None;
        }
        // the event in force at a desync tick was wrong there; it stays wrong until a newer one arrives
        let d = self.desyncs.partition_point(|&d| d <= t);
        match d.checked_sub(1).map(|i| self.desyncs[i]) {
            Some(desync) if e.intended <= desync => None,
            _ => Some(e),
        }
    }
}

/// What [`true_table`] saw.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrueStats {
    /// `Sv_PreInput` messages in the demo.
    pub messages: u64,
    /// Messages whose owner had a label (a character or player record) when they were resolved.
    pub labelled: u64,
    /// Messages that arrived out of the order of their intended ticks.
    pub out_of_order: u64,
    /// Events dropped because an earlier arrival already holds their intended tick (the server applies only the first).
    pub same_tick_dropped: u64,
    /// `intended - arrived`, clamped to `-8..=8` (index `lead + 8`).
    pub lead: [u64; 17],
}

/// The inputs of every player of a demo, by anonymous label.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TrueTable {
    pub tracks: HashMap<u16, Track>,
    pub stats: TrueStats,
    /// Calibration of the application tick: the step into tick `t` uses the input in force at
    /// `t - tick_shift`. `0` is the server's rule (`server.cpp:3581`: the input whose `GameTick` is
    /// `Tick()` is applied in that tick); other values only exist to measure that rule.
    pub tick_shift: i32,
}

impl TrueTable {
    pub fn track(&self, label: u16) -> Option<&Track> {
        self.tracks.get(&label)
    }

    /// Labels with at least one event, ascending.
    pub fn labels(&self) -> Vec<u16> {
        let mut v: Vec<u16> = self.tracks.keys().copied().collect();
        v.sort_unstable();
        v
    }
}

fn event_of(m: &PreInputMsg) -> InputEvent {
    InputEvent {
        intended: m.intended_tick,
        arrived: m.arrived,
        direction: m.direction.clamp(-1, 1) as i8,
        jump: m.jump != 0,
        hook: m.hook != 0,
        fire: m.fire,
        aim: m.target,
    }
}

/// [`true_table_raw`] plus the trust pass of [`snapshot_trust`]: the tracks a consumer should use.
pub fn true_table(demo: &Demo<'_>) -> TrueTable {
    let mut t = true_table_raw(demo);
    snapshot_trust(demo, &mut t);
    t
}

/// The trust pass: messages are not vital, so a lost one leaves the old input in force in the table while the player has moved on. The
/// snapshot's own core says what the player's direction was (a fresh core of a free player: its `direction` is the input of its tick);
/// when that differs from the event in force at the core's tick in two consecutive snapshots, that event is marked wrong
/// ([`Track::desyncs`]) and [`Track::at`] answers `None` until a newer event arrives. Same idea as `ddai_world::preinput`'s "distrusted" on
/// the live bot (D-112).
pub fn snapshot_trust(demo: &Demo<'_>, table: &mut TrueTable) {
    use ddai_net::generated::enums::characterflagflag as cf;
    let src = FrameSource::new(demo);
    let mut tracker = LabelTracker::new();
    // per label: (intended tick of the event found wrong, tick of the first mismatch)
    let mut pending: HashMap<u16, (i32, i32)> = HashMap::new();
    for (frame, _) in src {
        let Frame::Snapshot { tick, characters, .. } = &frame else {
            continue;
        };
        let row = tracker.push(&frame);
        for (c, &(_, label)) in characters.iter().zip(&row) {
            let ch = &c.character;
            let core_tick = if ch.tick == 0 { *tick } else { ch.tick };
            if core_tick < *tick - 1 {
                continue; // stale core: it says nothing about this tick
            }
            let frozen = c.ddnet.as_ref().map_or_else(
                || crate::pipeline::infer_freeze(ch.weapon),
                |d| d.freeze_end != 0 || d.flags & cf::MOVEMENTS_DISABLED != 0,
            );
            if frozen {
                pending.remove(&label);
                continue;
            }
            let Some(track) = table.tracks.get_mut(&label) else {
                continue;
            };
            let Some(e) = track.at(core_tick) else {
                pending.remove(&label);
                continue;
            };
            if i32::from(e.direction) == ch.direction.clamp(-1, 1) {
                pending.remove(&label);
                continue;
            }
            let e_intended = e.intended;
            match pending.get(&label) {
                Some(&(pe, first)) if pe == e_intended => {
                    track.desyncs.push(first);
                    pending.remove(&label);
                }
                _ => {
                    pending.insert(label, (e_intended, core_tick));
                }
            }
        }
    }
    for track in table.tracks.values_mut() {
        track.desyncs.sort_unstable();
        track.desyncs.dedup();
    }
}

/// Reads the demo once and builds the timeline of every labelled player, without the trust pass. The labels are the ones
/// [`FrameSource`] / [`crate::pipeline`] use (same anonymiser, same order), so a track can be looked
/// up by `CharRec::player`. A message is resolved with the frame that follows it (a player who
/// appears in that frame already has a label); one whose owner never had a label is not kept.
pub fn true_table_raw(demo: &Demo<'_>) -> TrueTable {
    let mut src = FrameSource::new(demo).collect_pre_inputs();
    let mut tracker = LabelTracker::new();
    let mut table = TrueTable::default();
    let mut by_label: BTreeMap<u16, Vec<InputEvent>> = BTreeMap::new();
    let mut restarts: BTreeMap<u16, Vec<i32>> = BTreeMap::new();
    let mut present: std::collections::HashSet<u16> = std::collections::HashSet::new();
    let mut take = |tracker: &LabelTracker, msgs: Vec<PreInputMsg>, table: &mut TrueTable| {
        for m in msgs {
            table.stats.messages += 1;
            let lead = (m.intended_tick - m.arrived).clamp(-8, 8);
            table.stats.lead[(lead + 8) as usize] += 1;
            let Some(label) = tracker.label_of(m.owner) else {
                continue;
            };
            table.stats.labelled += 1;
            by_label.entry(label).or_default().push(event_of(&m));
        }
    };
    while let Some((frame, _)) = src.next() {
        let tick = match &frame {
            ddai_recorder::format::Frame::Snapshot { tick, .. } => *tick,
            _ => continue,
        };
        let row = tracker.push(&frame);
        let now: std::collections::HashSet<u16> = row.iter().map(|&(_, l)| l).collect();
        for &l in &now {
            if !present.contains(&l) {
                restarts.entry(l).or_default().push(tick);
            }
        }
        present = now;
        let msgs = src.take_pre_inputs();
        take(&tracker, msgs, &mut table);
    }
    let rest = src.take_pre_inputs();
    take(&tracker, rest, &mut table);
    for (label, mut events) in by_label {
        let before = events.windows(2).filter(|w| w[1].intended < w[0].intended).count() as u64;
        table.stats.out_of_order += before;
        events.sort_by_key(|e| e.intended); // stable: ties stay in arrival order
        // Two inputs for one tick (the server moves a late input to `Tick() + 1`): `server.cpp:3557-3584` applies the first one it finds
        // for the tick, the earliest to have arrived, and never the others. Keep that one.
        let n = events.len();
        events.dedup_by(|later, earlier| later.intended == earlier.intended);
        table.stats.same_tick_dropped += (n - events.len()) as u64;
        let restarts = restarts.remove(&label).unwrap_or_default();
        table.tracks.insert(
            label,
            Track {
                events,
                restarts,
                desyncs: Vec::new(),
            },
        );
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(intended: i32, direction: i8, fire: i32) -> InputEvent {
        InputEvent {
            intended,
            arrived: intended - 1,
            direction,
            jump: false,
            hook: false,
            fire,
            aim: [0, -1],
        }
    }

    #[test]
    fn the_input_in_force_is_the_newest_event_not_after_the_tick() {
        let t = Track {
            events: vec![ev(10, 1, 0), ev(20, -1, 1), ev(20, 0, 2), ev(30, 0, 3)],
            restarts: Vec::new(),
            ..Track::default()
        };
        assert_eq!(t.at(9), None);
        assert_eq!(t.at(10).map(|e| e.direction), Some(1));
        assert_eq!(t.at(19).map(|e| e.direction), Some(1));
        // `Track::at` takes the last event of a tick; a table never holds two (see `two_inputs_for_one_tick_...`)
        assert_eq!(t.at(20).map(|e| e.fire), Some(2));
        assert_eq!(t.at(29).map(|e| e.fire), Some(2));
        assert_eq!(t.at(1000).map(|e| e.fire), Some(3));
        assert_eq!(Track::default().at(5), None);
    }

    #[test]
    fn an_event_from_before_the_player_re_appeared_is_unknown_state() {
        let t = Track {
            events: vec![ev(10, 1, 0), ev(60, -1, 1)],
            restarts: vec![8, 40],
            ..Track::default()
        };
        assert_eq!(t.at(30).map(|e| e.direction), Some(1), "inside the first stint");
        assert_eq!(
            t.at(40),
            None,
            "left at 10..40: the event of tick 10 is stale after the restart at 40"
        );
        assert_eq!(t.at(59), None);
        assert_eq!(
            t.at(60).map(|e| e.direction),
            Some(-1),
            "the next message makes the state known again"
        );
        // an event a hair before the restart tick still counts (messages precede their snapshot)
        let u = Track {
            events: vec![ev(38, 1, 0)],
            restarts: vec![40],
            ..Track::default()
        };
        assert_eq!(u.at(45).map(|e| e.direction), Some(1));
    }

    use crate::synth::{self, SynthChar};
    use crate::testutil::wire_character;
    use ddai_net::generated::messages::SvPreInput;

    fn pre(
        owner: i32,
        intended: i32,
        direction: i32,
        jump: i32,
        fire: i32,
        hook: i32,
        target: (i32, i32),
    ) -> SvPreInput {
        SvPreInput {
            direction,
            target_x: target.0,
            target_y: target.1,
            jump,
            fire,
            hook,
            wanted_weapon: 0,
            next_weapon: 0,
            prev_weapon: 0,
            owner,
            intended_tick: intended,
        }
    }

    fn synthetic_demo() -> Vec<u8> {
        let map = synth::map_bytes(30, 12, (10, 20));
        let ch = |id: i32, tick: i32| SynthChar {
            id,
            name: format!("ZzSecret{id}"),
            clan: "ZzClan".into(),
            wire: wire_character(tick, 100 + 40 * id, 338),
        };
        // player 1 is out of view in the snapshot of tick 106
        let snaps: Vec<(i32, Vec<SynthChar>)> = (0..7)
            .map(|i| {
                let t = 100 + 2 * i;
                (
                    t,
                    if t == 106 {
                        vec![ch(0, t)]
                    } else {
                        vec![ch(0, t), ch(1, t)]
                    },
                )
            })
            .collect();
        let msgs = [
            (100, pre(0, 101, 1, 0, 0, 0, (0, -1))),
            (102, pre(1, 103, 0, 1, 0, 0, (5, 6))),
            (102, pre(7, 104, 1, 0, 0, 0, (0, 0))), // an owner that never appears
            (104, pre(0, 106, 1, 0, 1, 1, (100, -50))),
            (112, pre(1, 114, -1, 0, 0, 0, (7, 8))),
        ];
        synth::demo_bytes_with_pre_inputs(&map, true, &snaps, &msgs)
    }

    #[test]
    fn a_demo_s_pre_inputs_become_per_label_tracks_without_names() {
        let bytes = synthetic_demo();
        let demo = Demo::parse(&bytes).unwrap();
        let t = true_table_raw(&demo);
        assert_eq!(t.stats.messages, 5);
        assert_eq!(t.stats.labelled, 4, "the owner that never appeared has no label");
        assert_eq!(
            t.stats.lead[(1 + 8) as usize],
            2,
            "two messages were 1 tick ahead of their chunk"
        );
        assert_eq!(t.stats.lead[(2 + 8) as usize], 3, "three were 2 ticks ahead");
        // the synthetic player 1 leaves the player list for one snapshot and comes back as a new
        // stint, i.e. a new label; its inputs go to the label of the stint they belong to
        assert_eq!(t.labels(), vec![0, 1, 2]);
        let a = t.track(0).unwrap();
        assert_eq!(a.events.len(), 2);
        assert_eq!(a.events[1].fire, 1);
        assert!(a.events[1].hook);
        assert_eq!(a.events[1].aim, [100, -50]);
        assert_eq!(a.events[1].intended, 106);
        assert_eq!(a.events[1].arrived, 104);
        assert_eq!(a.at(105).map(|e| e.direction), Some(1));
        assert_eq!(a.restarts, vec![100]);
        let b = t.track(1).unwrap();
        assert_eq!(b.events.len(), 1);
        assert!(b.events[0].jump);
        assert_eq!(b.events[0].aim, [5, 6]);
        let c = t.track(2).unwrap();
        assert_eq!(c.events.len(), 1);
        assert_eq!(c.events[0].direction, -1);
        assert_eq!(c.restarts, vec![108]);
        // nothing but numbers: no name anywhere in the table
        assert!(!format!("{t:?}").contains("ZzSecret"));
    }

    #[test]
    fn pre_inputs_are_counted_but_not_collected_unless_asked() {
        let bytes = synthetic_demo();
        let demo = Demo::parse(&bytes).unwrap();
        let mut src = FrameSource::new(&demo);
        for _ in &mut src {}
        assert_eq!(src.pre_input_messages, 5);
        assert!(src.take_pre_inputs().is_empty());
        let mut src = FrameSource::new(&demo).collect_pre_inputs();
        for _ in &mut src {}
        assert_eq!(src.take_pre_inputs().len(), 5);
        assert!(src.take_pre_inputs().is_empty(), "taken once");
    }

    #[test]
    fn a_lost_message_is_found_by_the_snapshot_core_and_the_event_is_distrusted() {
        let map = synth::map_bytes(30, 12, (10, 20));
        // player 0 really walks right (core direction 1) from the start; player 1 stands (0)
        let ch = |id: i32, tick: i32| {
            let mut w = wire_character(tick, 100 + 40 * id, 338);
            w.direction = if id == 0 { 1 } else { 0 };
            SynthChar {
                id,
                name: format!("ZzSecret{id}"),
                clan: "c".into(),
                wire: w,
            }
        };
        let snaps: Vec<(i32, Vec<SynthChar>)> = (0..8)
            .map(|i| (100 + 2 * i, vec![ch(0, 100 + 2 * i), ch(1, 100 + 2 * i)]))
            .collect();
        // the table has "player 0 stands" (the message that said it started walking was lost); player 1 is right; a later message of player 0 is right
        let msgs = [
            (100, pre(0, 101, 0, 0, 0, 0, (0, -1))),
            (100, pre(1, 101, 0, 0, 0, 0, (0, -1))),
            (110, pre(0, 111, 1, 0, 0, 0, (0, -1))),
        ];
        let bytes = synth::demo_bytes_with_pre_inputs(&map, true, &snaps, &msgs);
        let demo = Demo::parse(&bytes).unwrap();
        let raw = true_table_raw(&demo);
        assert!(
            raw.track(0).unwrap().at(105).is_some(),
            "the raw table believes the lost-message state"
        );
        let t = true_table(&demo);
        assert_eq!(
            t.track(0).unwrap().desyncs,
            vec![102],
            "found at the first of two mismatching snapshots (the core of tick 102)"
        );
        assert_eq!(
            t.track(0).unwrap().at(101).map(|e| e.direction),
            Some(0),
            "before the first mismatch it is still believed"
        );
        assert!(t.track(0).unwrap().at(105).is_none(), "wrong from the mismatch on");
        assert_eq!(
            t.track(0).unwrap().at(112).map(|e| e.direction),
            Some(1),
            "a newer message makes the state known again"
        );
        assert!(t.track(1).unwrap().desyncs.is_empty());
        assert_eq!(t.track(1).unwrap().at(105).map(|e| e.direction), Some(0));
    }

    #[test]
    fn two_inputs_for_one_tick_keep_the_earliest_arrival_like_the_server() {
        let map = synth::map_bytes(30, 12, (10, 20));
        let ch = |id: i32, tick: i32| SynthChar {
            id,
            name: format!("ZzSecret{id}"),
            clan: "c".into(),
            wire: wire_character(tick, 100 + 40 * id, 338),
        };
        let snaps: Vec<(i32, Vec<SynthChar>)> = (0..4)
            .map(|i| (100 + 2 * i, vec![ch(0, 100 + 2 * i), ch(1, 100 + 2 * i)]))
            .collect();
        // a late input is moved to the next tick by the server, so two messages can carry the same intended tick (103): the one that arrived
        // first (chunk tick 100, direction -1) is the one the server applies, the second (chunk tick 102, direction +1) never is
        let msgs = [
            (100, pre(0, 103, -1, 0, 0, 0, (0, -1))),
            (102, pre(0, 103, 1, 0, 0, 0, (0, -1))),
            (102, pre(0, 104, 0, 0, 0, 0, (0, -1))),
        ];
        let bytes = synth::demo_bytes_with_pre_inputs(&map, true, &snaps, &msgs);
        let demo = Demo::parse(&bytes).unwrap();
        let t = true_table_raw(&demo);
        assert_eq!(t.stats.same_tick_dropped, 1);
        let tr = t.track(0).unwrap();
        assert_eq!(tr.events.iter().map(|e| e.intended).collect::<Vec<_>>(), vec![103, 104]);
        assert_eq!(
            tr.at(103).map(|e| e.direction),
            Some(-1),
            "the earliest arrival holds tick 103"
        );
    }
}
