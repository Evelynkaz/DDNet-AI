//! Demo -> anonymised recorder frames.
//!
//! A `.demo` is walked once; every *fresh* snapshot (the reader replays the previous snapshot on
//! ticks where the server sent nothing, those are skipped) becomes a `ddai_recorder` snapshot
//! [`Frame`] - the same shape the observer recorder writes (8.4a), so the recorder's input
//! reconstruction is reused unchanged. Two deliberate adaptations, both forced by the difference
//! between a live recording and a demo:
//!
//! - `character.tick == 0` ("this already is the current state", sent for freshly spawned
//!   characters) is rewritten to the snapshot tick. The recorder deduplicates samples by
//!   `character.tick`, and a run of literal `0`s would be dropped as "repeats" of the first;
//! - **nicknames are removed at the door**: every frame goes through the recorder's own
//!   [`Anonymizer`] (labels `player_N`, keyed by `(client id, stint)`, numbered in order of first
//!   appearance *per demo*, never a hash of the name) before anything else sees it. Nothing
//!   downstream of this module can leak a nickname because nothing downstream ever holds one.

use std::sync::Arc;

use ddai_demo::Demo;
use ddai_net::generated::messages::ExGameMsg;
use ddai_net::message::Msg;
use ddai_net::snapshot::Snapshot;
use ddai_net::tuning::{DEFAULT_TUNE_PARAMS, TuneParams};
use ddai_net::view::View;
use ddai_recorder::anonymize::Anonymizer;
use ddai_recorder::format::{CharacterRecord, Frame, PlayerRecord};

/// A kill message (`Sv_KillMsg`): the block mod credits the last toucher as `killer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KillEvent {
    /// Tick of the snapshot the message arrived with.
    pub tick: i32,
    pub killer: i32,
    pub victim: i32,
    pub weapon: i32,
}

/// One `Sv_PreInput` message (task 3.24): a *real* input change of another player, forwarded by the
/// server before its tick and recorded by the client into the demo (`client.cpp:2415`). Only the
/// owner's client id is known here (the label comes from the frames, see
/// [`crate::humaninput::true_table`]); no name is involved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreInputMsg {
    /// Demo tick of the chunk the message was recorded in.
    pub arrived: i32,
    /// Client id of the player whose input this is.
    pub owner: i32,
    pub intended_tick: i32,
    pub direction: i32,
    pub jump: i32,
    pub hook: i32,
    pub fire: i32,
    pub target: [i32; 2],
    pub wanted_weapon: i32,
    pub next_weapon: i32,
    pub prev_weapon: i32,
}

/// The anonymised content of one demo.
#[derive(Debug, Clone, Default)]
pub struct Ingested {
    /// Fresh snapshots, in order (`Frame::Snapshot` only).
    pub frames: Vec<Frame>,
    /// The latest `Sv_TuneParams` seen at or before each frame.
    pub tunes: Vec<TuneParams>,
    pub kills: Vec<KillEvent>,
    /// Snapshots that were replays of the previous one (skipped).
    pub repeated_snapshots: usize,
    /// A fatal decode error that ended the demo early (the frames before it are kept).
    pub decode_error: Option<String>,
    /// Characters that had no `ClientInfo` in their frame (they get a carried-forward label).
    pub chars_without_info: usize,
    /// Frames whose tuning differed from DDNet's defaults ([`tune_is_default`]).
    pub tune_nondefault_frames: usize,
    /// Frames whose tuning differed in a movement-relevant field ([`tune_moves_differently`]).
    pub tune_movement_frames: usize,
}

/// A tune message that differs from DDNet's compile-time defaults in a field that was received.
pub fn tune_is_default(t: &TuneParams) -> bool {
    let mut a = *t;
    let mut b = DEFAULT_TUNE_PARAMS;
    a.received = 0;
    b.received = 0;
    a == b
}

/// Whether the tuning differs from the defaults in a field that changes how tees move or how
/// strongly weapons push (as opposed to weapon speeds, lifetimes, fire delays and the jetpack).
pub fn tune_moves_differently(t: &TuneParams) -> bool {
    let d = DEFAULT_TUNE_PARAMS;
    let f = |x: &TuneParams| {
        [
            x.ground_control_speed,
            x.ground_control_accel,
            x.ground_friction,
            x.ground_jump_impulse,
            x.air_jump_impulse,
            x.air_control_speed,
            x.air_control_accel,
            x.air_friction,
            x.hook_length,
            x.hook_fire_speed,
            x.hook_drag_accel,
            x.hook_drag_speed,
            x.gravity,
            x.velramp_start,
            x.velramp_range,
            x.velramp_curvature,
            x.player_collision,
            x.player_hooking,
            x.shotgun_strength,
            x.explosion_strength,
            x.hammer_strength,
            x.hook_duration,
            x.ground_elasticity_x,
            x.ground_elasticity_y,
        ]
    };
    f(t) != f(&d)
}

/// `View` -> recorder frame: a mechanical field-for-field copy (same as the observer's
/// `snapshot_to_frame`, which lives in the `ddnet-ai` binary and cannot be called from here), with
/// the `tick == 0` adaptation described in the module docs.
pub fn snapshot_to_frame(tick: i32, snapshot: &Snapshot) -> Frame {
    let view = View::new(snapshot);
    let characters = view
        .characters()
        .into_iter()
        .map(|c| {
            let mut character = c.character;
            if character.tick == 0 {
                character.tick = tick;
            }
            CharacterRecord {
                id: c.id,
                character,
                ddnet: c.ddnet,
            }
        })
        .collect();
    let players = view
        .players()
        .into_iter()
        .map(|p| PlayerRecord {
            id: p.id,
            info: p.info,
            client_info: p.client_info,
            ddnet: p.ddnet,
        })
        .collect();
    Frame::Snapshot {
        tick,
        characters,
        players,
    }
}

/// One fresh snapshot of a demo, anonymised, with the tuning in force.
pub type IngestedFrame = (Frame, TuneParams);

/// Walks the demo one fresh snapshot at a time (constant memory: the demo's bytes aside, only the
/// previous snapshot, the anonymiser state and the kill list are kept). Never panics; a decode
/// error ends the stream and is reported in [`FrameSource::decode_error`]. The counters and
/// `kills` are complete once the iterator is exhausted.
pub struct FrameSource<'d> {
    ticks: ddai_demo::TickIter<'d>,
    anon: Anonymizer,
    prev: Option<Arc<Snapshot>>,
    tune: TuneParams,
    done: bool,
    /// Every kill message seen so far, in order.
    pub kills: Vec<KillEvent>,
    /// Snapshots that were replays of the previous one (skipped).
    pub repeated_snapshots: usize,
    /// A fatal decode error that ended the demo early (the frames before it are kept).
    pub decode_error: Option<String>,
    /// Frames yielded whose tuning differed from DDNet's defaults ([`tune_is_default`]).
    pub tune_nondefault_frames: usize,
    /// Frames yielded whose tuning differed in a movement-relevant field ([`tune_moves_differently`]).
    pub tune_movement_frames: usize,
    /// `Sv_PreInput` messages seen so far and not yet taken ([`FrameSource::take_pre_inputs`]);
    /// only filled when collection was switched on with [`FrameSource::collect_pre_inputs`].
    pre_inputs: Vec<PreInputMsg>,
    collect_pre_inputs: bool,
    /// `Sv_PreInput` messages seen in total (counted whether or not they are collected).
    pub pre_input_messages: u64,
    /// Fresh snapshots closer than this many ticks to the last one yielded are not yielded (they are
    /// still anonymised, so labels do not depend on it); 0 = yield all.
    min_spacing: i32,
    last_yielded: Option<i32>,
    /// Fresh snapshots left out because of `min_spacing`.
    pub decimated_snapshots: usize,
}

impl<'d> FrameSource<'d> {
    pub fn new(demo: &Demo<'d>) -> Self {
        FrameSource {
            ticks: demo.ticks(),
            anon: Anonymizer::new(),
            prev: None,
            tune: DEFAULT_TUNE_PARAMS,
            done: false,
            kills: Vec::new(),
            repeated_snapshots: 0,
            decode_error: None,
            tune_nondefault_frames: 0,
            tune_movement_frames: 0,
            pre_inputs: Vec::new(),
            collect_pre_inputs: false,
            pre_input_messages: 0,
            min_spacing: 0,
            last_yielded: None,
            decimated_snapshots: 0,
        }
    }

    /// Skips fresh snapshots less than `ticks` after the last one yielded. Servers that send a
    /// snapshot every tick (50 Hz; several of the owner's duel demos) then look like the usual
    /// 25 Hz stream the pipeline's two-tick steps are made for; the players' real inputs are per
    /// tick anyway ([`crate::humaninput`]).
    pub fn min_spacing(mut self, ticks: i32) -> Self {
        self.min_spacing = ticks;
        self
    }

    /// Switches the collection of `Sv_PreInput` messages on (off by default: nothing else needs
    /// them and they would pile up).
    pub fn collect_pre_inputs(mut self) -> Self {
        self.collect_pre_inputs = true;
        self
    }

    /// The pre-input messages recorded since the previous call, in file order.
    pub fn take_pre_inputs(&mut self) -> Vec<PreInputMsg> {
        std::mem::take(&mut self.pre_inputs)
    }
}

impl Iterator for FrameSource<'_> {
    type Item = IngestedFrame;

    fn next(&mut self) -> Option<IngestedFrame> {
        if self.done {
            return None;
        }
        loop {
            let tick = match self.ticks.next() {
                None => {
                    self.done = true;
                    return None;
                }
                Some(Ok(t)) => t,
                Some(Err(e)) => {
                    self.decode_error = Some(e.to_string());
                    self.done = true;
                    return None;
                }
            };
            for msg in &tick.messages {
                match msg {
                    Msg::TuneParams(t) => self.tune = *t,
                    Msg::Game(ddai_net::generated::messages::GameMsg::SvKillMsg(k)) => {
                        self.kills.push(KillEvent {
                            tick: tick.tick,
                            killer: k.killer,
                            victim: k.victim,
                            weapon: k.weapon,
                        });
                    }
                    Msg::ExGame(ExGameMsg::SvPreInput(p)) => {
                        self.pre_input_messages += 1;
                        if self.collect_pre_inputs {
                            self.pre_inputs.push(PreInputMsg {
                                arrived: tick.tick,
                                owner: p.owner,
                                intended_tick: p.intended_tick,
                                direction: p.direction,
                                jump: p.jump,
                                hook: p.hook,
                                fire: p.fire,
                                target: [p.target_x, p.target_y],
                                wanted_weapon: p.wanted_weapon,
                                next_weapon: p.next_weapon,
                                prev_weapon: p.prev_weapon,
                            });
                        }
                    }
                    _ => {}
                }
            }
            let Some(snap) = &tick.snapshot else { continue };
            let fresh = self.prev.as_ref().is_none_or(|p| !Arc::ptr_eq(p, snap));
            self.prev = Some(Arc::clone(snap));
            if !fresh {
                self.repeated_snapshots += 1;
                continue;
            }
            let mut frame = snapshot_to_frame(tick.tick, snap);
            self.anon.anonymize_frame(&mut frame);
            if self.min_spacing > 0 && self.last_yielded.is_some_and(|l| tick.tick - l < self.min_spacing) {
                self.decimated_snapshots += 1;
                continue;
            }
            self.last_yielded = Some(tick.tick);
            if !tune_is_default(&self.tune) {
                self.tune_nondefault_frames += 1;
            }
            if tune_moves_differently(&self.tune) {
                self.tune_movement_frames += 1;
            }
            return Some((frame, self.tune));
        }
    }
}

impl Ingested {
    /// The frames with their tuning. `tunes` is aligned with `frames`; a missing entry means the
    /// defaults (never a silent truncation of the frames).
    pub fn frames_with_tunes(&self) -> impl Iterator<Item = IngestedFrame> + '_ {
        self.frames
            .iter()
            .enumerate()
            .map(|(k, f)| (f.clone(), self.tunes.get(k).copied().unwrap_or(DEFAULT_TUNE_PARAMS)))
    }
}

/// Walks the whole demo into memory. Only for tests and small tools: the pipeline streams
/// ([`FrameSource`]) so that memory does not grow with the length of the demo.
pub fn ingest(demo: &Demo<'_>) -> Ingested {
    let mut src = FrameSource::new(demo);
    let mut out = Ingested::default();
    for (frame, tune) in &mut src {
        out.frames.push(frame);
        out.tunes.push(tune);
    }
    out.kills = src.kills;
    out.repeated_snapshots = src.repeated_snapshots;
    out.decode_error = src.decode_error;
    out.tune_nondefault_frames = src.tune_nondefault_frames;
    out.tune_movement_frames = src.tune_movement_frames;
    out
}

/// The label number of an anonymised `player_N` name.
pub fn label_number(name: &str) -> Option<u16> {
    name.strip_prefix("player_")?
        .parse::<u32>()
        .ok()
        .and_then(|n| u16::try_from(n).ok())
}

/// Carries the `(client id -> label number)` map from frame to frame: a character whose
/// `ClientInfo` is missing in a frame keeps the label it had in the previous frame; one that never
/// had any gets a label above `u16::MAX / 2` so it can never collide with an anonymiser label.
#[derive(Debug, Default)]
pub struct LabelTracker {
    carried: std::collections::HashMap<i32, u16>,
    /// Characters that never had a label of their own.
    pub missing: usize,
}

impl LabelTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// The label number the client id carries right now (`None` before it was ever seen).
    pub fn label_of(&self, id: i32) -> Option<u16> {
        self.carried.get(&id).copied()
    }

    /// `(client id, label number)` of every character of `frame`, in the frame's order (empty for
    /// a frame that is not a snapshot).
    pub fn push(&mut self, frame: &Frame) -> Vec<(i32, u16)> {
        let Frame::Snapshot {
            characters, players, ..
        } = frame
        else {
            return Vec::new();
        };
        for p in players {
            if let Some(n) = p.client_info.as_ref().and_then(|ci| label_number(&ci.name)) {
                self.carried.insert(p.id, n);
            }
        }
        let mut row = Vec::with_capacity(characters.len());
        for c in characters {
            let n = match self.carried.get(&c.id) {
                Some(&n) => n,
                None => {
                    self.missing += 1;
                    let n = 0x8000u16.saturating_add(u16::try_from(c.id.clamp(0, 0x7f00)).unwrap_or(0));
                    self.carried.insert(c.id, n);
                    n
                }
            };
            row.push((c.id, n));
        }
        row
    }
}

/// Per-frame `(client id -> label number)` maps for an anonymised frame list (see
/// [`LabelTracker`]), and the number of characters that never had a label.
pub fn labels(frames: &[Frame]) -> (Vec<Vec<(i32, u16)>>, usize) {
    let mut tracker = LabelTracker::new();
    let out = frames.iter().map(|f| tracker.push(f)).collect();
    (out, tracker.missing)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{client_info as ci, player_info, wire_character};

    #[test]
    fn label_number_parses_only_anonymiser_labels() {
        assert_eq!(label_number("player_7"), Some(7));
        assert_eq!(label_number("player_"), None);
        assert_eq!(label_number("Alice"), None);
        assert_eq!(label_number("player_70000"), None);
    }

    #[test]
    fn default_tune_is_default_and_a_change_is_not() {
        assert!(tune_is_default(&DEFAULT_TUNE_PARAMS));
        let mut t = DEFAULT_TUNE_PARAMS;
        t.hook_length += 100;
        assert!(!tune_is_default(&t));
    }

    #[test]
    fn weapon_only_tuning_changes_do_not_count_as_movement_changes() {
        let mut t = DEFAULT_TUNE_PARAMS;
        t.gun_speed = 1;
        t.hammer_hit_fire_delay = 0;
        assert!(!tune_is_default(&t));
        assert!(!tune_moves_differently(&t));
        t.gravity += 10;
        assert!(tune_moves_differently(&t));
        let mut e = DEFAULT_TUNE_PARAMS;
        e.ground_elasticity_x = 12500;
        assert!(tune_moves_differently(&e));
    }

    #[test]
    fn labels_carry_forward_and_never_collide() {
        let chr = |id: i32| CharacterRecord {
            id,
            character: wire_character(0, 0, 0),
            ddnet: None,
        };
        let player = |id: i32, name: Option<&str>| PlayerRecord {
            id,
            info: player_info(id),
            client_info: name.map(ci),
            ddnet: None,
        };
        let frames = vec![
            Frame::Snapshot {
                tick: 2,
                characters: vec![chr(0), chr(1)],
                players: vec![player(0, Some("player_3")), player(1, None)],
            },
            Frame::Snapshot {
                tick: 4,
                characters: vec![chr(0)],
                players: vec![player(0, None)],
            },
        ];
        let (rows, missing) = labels(&frames);
        assert_eq!(rows[0][0], (0, 3));
        assert!(rows[0][1].1 >= 0x8000, "unlabelled ids get a reserved label");
        assert_eq!(rows[1][0], (0, 3), "label carried forward when ClientInfo is absent");
        assert_eq!(missing, 1);
    }

    #[test]
    fn a_zero_character_tick_becomes_the_snapshot_tick_and_other_ticks_are_kept() {
        use ddai_net::snapshot::{Snapshot, SnapshotItem};
        let item = |id: i32, tick: i32| {
            let c = wire_character(tick, 10 * id, 20);
            SnapshotItem {
                key: (ddai_net::generated::objects::Character::ID << 16) | id,
                data: vec![
                    c.tick,
                    c.x,
                    c.y,
                    c.vel_x,
                    c.vel_y,
                    c.angle,
                    c.direction,
                    c.jumped,
                    c.hooked_player,
                    c.hook_state,
                    c.hook_tick,
                    c.hook_x,
                    c.hook_y,
                    c.hook_dx,
                    c.hook_dy,
                    c.player_flags,
                    c.health,
                    c.armor,
                    c.ammo_count,
                    c.weapon,
                    c.emote,
                    c.attack_tick,
                ],
            }
        };
        let snap = Snapshot {
            items: vec![item(0, 0), item(1, 4990)],
        };
        let Frame::Snapshot { characters, tick, .. } = snapshot_to_frame(5000, &snap) else {
            panic!("a snapshot frame")
        };
        assert_eq!(tick, 5000);
        assert_eq!(characters.len(), 2);
        assert_eq!(characters[0].character.tick, 5000);
        assert_eq!(characters[1].character.tick, 4990);
    }

    #[test]
    fn ingest_synthetic_demo_yields_frames_and_no_error() {
        let bytes = ddai_demo::testutil::build_synthetic_demo(6);
        let demo = Demo::parse(&bytes).unwrap();
        let ing = ingest(&demo);
        assert!(ing.decode_error.is_none());
        // Both ticks of the synthetic demo carry a snapshot chunk (the second one an empty delta),
        // so both are fresh snapshots; the synthetic snapshot has no character objects.
        assert_eq!(ing.frames.len(), 2);
        assert_eq!(ing.repeated_snapshots, 0);
        assert_eq!(ing.tunes.len(), 2);
        assert_eq!(ing.kills.len(), 0);
    }

    #[test]
    fn min_spacing_thins_a_50_hz_stream_to_two_ticks_and_keeps_the_labels() {
        use crate::synth::{self, SynthChar};
        let map = synth::map_bytes(30, 12, (10, 20));
        let ch = |id: i32, tick: i32| SynthChar {
            id,
            name: format!("ZzName{id}"),
            clan: "c".into(),
            wire: wire_character(tick, 100 + 40 * id, 338),
        };
        // a snapshot every tick, then a 3-tick gap
        let ticks = [100, 101, 102, 103, 104, 105, 108, 109, 110];
        let snaps: Vec<(i32, Vec<SynthChar>)> = ticks.iter().map(|&t| (t, vec![ch(0, t), ch(1, t)])).collect();
        let bytes = synth::demo_bytes(&map, true, &snaps);
        let demo = Demo::parse(&bytes).unwrap();
        let all: Vec<i32> = FrameSource::new(&demo).map(|(f, _)| frame_tick(&f)).collect();
        assert_eq!(all, ticks);
        let mut thin = FrameSource::new(&demo).min_spacing(2);
        let kept: Vec<(i32, Vec<String>)> = (&mut thin).map(|(f, _)| (frame_tick(&f), names(&f))).collect();
        assert_eq!(
            kept.iter().map(|k| k.0).collect::<Vec<_>>(),
            vec![100, 102, 104, 108, 110]
        );
        assert_eq!(thin.decimated_snapshots, 4);
        // the labels of the thinned stream are the labels of the full one
        let full: Vec<(i32, Vec<String>)> = FrameSource::new(&demo)
            .map(|(f, _)| (frame_tick(&f), names(&f)))
            .collect();
        for (t, n) in &kept {
            assert_eq!(full.iter().find(|f| f.0 == *t).map(|f| &f.1), Some(n));
        }
        assert_eq!(kept[0].1, vec!["player_0", "player_1"]);
    }

    fn frame_tick(f: &Frame) -> i32 {
        match f {
            Frame::Snapshot { tick, .. } => *tick,
            _ => unreachable!(),
        }
    }

    fn names(f: &Frame) -> Vec<String> {
        match f {
            Frame::Snapshot { players, .. } => players
                .iter()
                .map(|p| p.client_info.as_ref().map(|c| c.name.clone()).unwrap_or_default())
                .collect(),
            _ => unreachable!(),
        }
    }
}
