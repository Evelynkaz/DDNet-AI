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
        }
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
}
