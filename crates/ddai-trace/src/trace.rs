//! Trace v1: the applied input and resulting `CCharacterCore` state for every character, every
//! tick. This is what the C++ oracle writes and what task 1.3's Rust physics port will be
//! checked against. See `docs/formats.md` for the exact byte layout and the field list.

use crate::hash::Fnv1a64;
use crate::io::{FormatError, Reader, Writer};
use crate::scenario::PlayerInput;
use serde::{Deserialize, Serialize};
use std::fmt;

const MAGIC: &[u8; 4] = b"TRC1";
const VERSION: u32 = 1;

/// JSON metadata embedded in a trace file's header. Every field is a plain struct/`Vec`
/// (never a `HashMap`), and `serde_json` serializes struct fields in declaration order, so
/// `to_json_string` is deterministic byte-for-byte.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceMetadata {
    pub producer: Producer,
    pub ddnet: DdnetRef,
    /// Lowercase hex sha256 of the rawmap v1 bytes of the map the scenario ran on.
    pub map_sha256: String,
    pub scenario: ScenarioRef,
    /// `(name, type)` pairs, `type` one of `"i32"`/`"f32"`/`"f64"`, in the exact order the
    /// "input applied" half of every trace row is written in. See [`crate::scenario::PlayerInput`].
    pub input_schema: Vec<(String, String)>,
    /// `(name, type)` pairs for the "state after the tick" half of every trace row, in the
    /// exact order [`CharacterCoreState`] is written in — and the order the canonical per-tick
    /// hash (see [`crate::hash`]) concatenates fields in.
    pub state_schema: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Producer {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DdnetRef {
    pub tag: String,
    pub commit: String,
}

/// How to reproduce the scenario the trace ran, redundantly: both the generator call that
/// should reproduce it byte-for-byte, and the scenario file's own sha256 as a check that still
/// works when the scenario wasn't generator-produced (e.g. a hand-built rawmap scenario).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenarioRef {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generator: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Lowercase hex sha256 of the scenario v1 bytes.
    pub scenario_sha256: String,
}

/// The 10 `CNetObj_PlayerInput` fields, always `i32`, in wire order.
pub fn input_schema() -> Vec<(String, String)> {
    [
        "direction",
        "target_x",
        "target_y",
        "jump",
        "fire",
        "hook",
        "player_flags",
        "wanted_weapon",
        "next_weapon",
        "prev_weapon",
    ]
    .iter()
    .map(|n| (n.to_string(), "i32".to_string()))
    .collect()
}

/// The [`CharacterCoreState`] fields, in declaration/write order. Kept in one place so the JSON
/// metadata's `state_schema` can never silently drift from what `CharacterCoreState::write`
/// actually writes (a test in this module checks the byte count these types imply against
/// `CharacterCoreState`'s actual serialized size).
pub fn state_schema() -> Vec<(String, String)> {
    let f32_fields = [
        "pos_x",
        "pos_y",
        "vel_x",
        "vel_y",
        "hook_pos_x",
        "hook_pos_y",
        "hook_dir_x",
        "hook_dir_y",
        "hook_tele_base_x",
        "hook_tele_base_y",
    ];
    let i32_fields = [
        "hook_tick",
        "hook_state",
        "hooked_player",
        "active_weapon",
        "new_hook",
        "jumped",
        "jumped_total",
        "jumps",
        "direction",
        "angle",
        "triggered_events",
        "colliding",
        "left_wall",
        "move_restrictions",
        "solo",
        "collision_disabled",
        "endless_hook",
        "hook_hit_disabled",
    ];
    f32_fields
        .iter()
        .map(|n| (n.to_string(), "f32".to_string()))
        .chain(i32_fields.iter().map(|n| (n.to_string(), "i32".to_string())))
        .collect()
}

/// The full `CCharacterCore` state recorded after a tick (i.e. after `Move()` + `Quantize()`),
/// for one character.
///
/// Fields are exactly the public members of DDNet 20.1's `CCharacterCore` (`src/game/gamecore.h`)
/// that the core-level tick (no `CCharacter`/DDRace logic — see the task spec) can change, with
/// two deliberate omissions documented in `docs/formats.md`: `m_AttachedPlayers` (a
/// `std::set<int>`, fully derivable from every character's `hooked_player` this same tick, so it
/// carries no information a fixed-width schema would gain from repeating) and the
/// weapon/ninja/telegun/freeze members (never touched by the core-only tick this oracle runs;
/// they are Oracle B's job).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CharacterCoreState {
    pub pos_x: f32,
    pub pos_y: f32,
    pub vel_x: f32,
    pub vel_y: f32,
    pub hook_pos_x: f32,
    pub hook_pos_y: f32,
    pub hook_dir_x: f32,
    pub hook_dir_y: f32,
    pub hook_tele_base_x: f32,
    pub hook_tele_base_y: f32,
    pub hook_tick: i32,
    pub hook_state: i32,
    /// `CCharacterCore::HookedPlayer()`, `-1` if not hooking anyone.
    pub hooked_player: i32,
    pub active_weapon: i32,
    /// `bool m_NewHook`, as `0`/`1`.
    pub new_hook: i32,
    pub jumped: i32,
    pub jumped_total: i32,
    pub jumps: i32,
    pub direction: i32,
    pub angle: i32,
    pub triggered_events: i32,
    pub colliding: i32,
    /// `bool m_LeftWall`, as `0`/`1`.
    pub left_wall: i32,
    /// `CCollision::GetMoveRestrictions` result the core computed for this tick (`m_MoveRestrictions`
    /// is private with no getter; the oracle recomputes the identical value — see
    /// `tools/ddnet-oracle/oracle_core.cpp` and `docs/formats.md`).
    pub move_restrictions: i32,
    /// `bool m_Solo`, as `0`/`1`.
    pub solo: i32,
    /// `bool m_CollisionDisabled`, as `0`/`1`.
    pub collision_disabled: i32,
    /// `bool m_EndlessHook`, as `0`/`1`.
    pub endless_hook: i32,
    /// `bool m_HookHitDisabled`, as `0`/`1`.
    pub hook_hit_disabled: i32,
}

/// Number of `f32` fields at the front of [`CharacterCoreState`] (must match [`state_schema`]).
const STATE_F32_COUNT: usize = 10;
/// Number of `i32` fields after the `f32` ones (must match [`state_schema`]).
const STATE_I32_COUNT: usize = 18;
/// Serialized size in bytes of one [`CharacterCoreState`].
pub const STATE_BYTES: usize = STATE_F32_COUNT * 4 + STATE_I32_COUNT * 4;
/// Serialized size in bytes of one [`PlayerInput`] (10 `i32` fields).
pub const INPUT_BYTES: usize = 10 * 4;

impl CharacterCoreState {
    fn write(&self, w: &mut Writer) {
        w.f32(self.pos_x);
        w.f32(self.pos_y);
        w.f32(self.vel_x);
        w.f32(self.vel_y);
        w.f32(self.hook_pos_x);
        w.f32(self.hook_pos_y);
        w.f32(self.hook_dir_x);
        w.f32(self.hook_dir_y);
        w.f32(self.hook_tele_base_x);
        w.f32(self.hook_tele_base_y);
        w.i32(self.hook_tick);
        w.i32(self.hook_state);
        w.i32(self.hooked_player);
        w.i32(self.active_weapon);
        w.i32(self.new_hook);
        w.i32(self.jumped);
        w.i32(self.jumped_total);
        w.i32(self.jumps);
        w.i32(self.direction);
        w.i32(self.angle);
        w.i32(self.triggered_events);
        w.i32(self.colliding);
        w.i32(self.left_wall);
        w.i32(self.move_restrictions);
        w.i32(self.solo);
        w.i32(self.collision_disabled);
        w.i32(self.endless_hook);
        w.i32(self.hook_hit_disabled);
    }

    fn read(r: &mut Reader) -> Result<Self, FormatError> {
        Ok(CharacterCoreState {
            pos_x: r.f32("pos_x")?,
            pos_y: r.f32("pos_y")?,
            vel_x: r.f32("vel_x")?,
            vel_y: r.f32("vel_y")?,
            hook_pos_x: r.f32("hook_pos_x")?,
            hook_pos_y: r.f32("hook_pos_y")?,
            hook_dir_x: r.f32("hook_dir_x")?,
            hook_dir_y: r.f32("hook_dir_y")?,
            hook_tele_base_x: r.f32("hook_tele_base_x")?,
            hook_tele_base_y: r.f32("hook_tele_base_y")?,
            hook_tick: r.i32("hook_tick")?,
            hook_state: r.i32("hook_state")?,
            hooked_player: r.i32("hooked_player")?,
            active_weapon: r.i32("active_weapon")?,
            new_hook: r.i32("new_hook")?,
            jumped: r.i32("jumped")?,
            jumped_total: r.i32("jumped_total")?,
            jumps: r.i32("jumps")?,
            direction: r.i32("direction")?,
            angle: r.i32("angle")?,
            triggered_events: r.i32("triggered_events")?,
            colliding: r.i32("colliding")?,
            left_wall: r.i32("left_wall")?,
            move_restrictions: r.i32("move_restrictions")?,
            solo: r.i32("solo")?,
            collision_disabled: r.i32("collision_disabled")?,
            endless_hook: r.i32("endless_hook")?,
            hook_hit_disabled: r.i32("hook_hit_disabled")?,
        })
    }

    /// The field values in schema order, named, for diffing/printing.
    pub fn named_fields(&self) -> [(&'static str, FieldValue); STATE_F32_COUNT + STATE_I32_COUNT] {
        [
            ("pos_x", FieldValue::F32(self.pos_x)),
            ("pos_y", FieldValue::F32(self.pos_y)),
            ("vel_x", FieldValue::F32(self.vel_x)),
            ("vel_y", FieldValue::F32(self.vel_y)),
            ("hook_pos_x", FieldValue::F32(self.hook_pos_x)),
            ("hook_pos_y", FieldValue::F32(self.hook_pos_y)),
            ("hook_dir_x", FieldValue::F32(self.hook_dir_x)),
            ("hook_dir_y", FieldValue::F32(self.hook_dir_y)),
            ("hook_tele_base_x", FieldValue::F32(self.hook_tele_base_x)),
            ("hook_tele_base_y", FieldValue::F32(self.hook_tele_base_y)),
            ("hook_tick", FieldValue::I32(self.hook_tick)),
            ("hook_state", FieldValue::I32(self.hook_state)),
            ("hooked_player", FieldValue::I32(self.hooked_player)),
            ("active_weapon", FieldValue::I32(self.active_weapon)),
            ("new_hook", FieldValue::I32(self.new_hook)),
            ("jumped", FieldValue::I32(self.jumped)),
            ("jumped_total", FieldValue::I32(self.jumped_total)),
            ("jumps", FieldValue::I32(self.jumps)),
            ("direction", FieldValue::I32(self.direction)),
            ("angle", FieldValue::I32(self.angle)),
            ("triggered_events", FieldValue::I32(self.triggered_events)),
            ("colliding", FieldValue::I32(self.colliding)),
            ("left_wall", FieldValue::I32(self.left_wall)),
            ("move_restrictions", FieldValue::I32(self.move_restrictions)),
            ("solo", FieldValue::I32(self.solo)),
            ("collision_disabled", FieldValue::I32(self.collision_disabled)),
            ("endless_hook", FieldValue::I32(self.endless_hook)),
            ("hook_hit_disabled", FieldValue::I32(self.hook_hit_disabled)),
        ]
    }
}

/// One field's value, tagged by type — used for schema-driven diffing/printing, never for the
/// binary encoding itself (that's fixed per-field in [`CharacterCoreState::write`]/`read`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FieldValue {
    I32(i32),
    F32(f32),
}

impl FieldValue {
    /// Bit-exact equality: `f32` values are compared via `to_bits()`, so e.g. `+0.0 != -0.0` and
    /// two `NaN`s with the same payload are equal — what a physics-parity comparison needs,
    /// unlike IEEE `==`.
    pub fn bit_eq(&self, other: &FieldValue) -> bool {
        match (self, other) {
            (FieldValue::I32(a), FieldValue::I32(b)) => a == b,
            (FieldValue::F32(a), FieldValue::F32(b)) => a.to_bits() == b.to_bits(),
            _ => false,
        }
    }
}

impl fmt::Display for FieldValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FieldValue::I32(v) => write!(f, "{v}"),
            FieldValue::F32(v) => write!(f, "{v} (0x{:08x})", v.to_bits()),
        }
    }
}

fn input_named_fields(input: &PlayerInput) -> [(&'static str, FieldValue); 10] {
    let f = input.to_fields();
    [
        ("direction", FieldValue::I32(f[0])),
        ("target_x", FieldValue::I32(f[1])),
        ("target_y", FieldValue::I32(f[2])),
        ("jump", FieldValue::I32(f[3])),
        ("fire", FieldValue::I32(f[4])),
        ("hook", FieldValue::I32(f[5])),
        ("player_flags", FieldValue::I32(f[6])),
        ("wanted_weapon", FieldValue::I32(f[7])),
        ("next_weapon", FieldValue::I32(f[8])),
        ("prev_weapon", FieldValue::I32(f[9])),
    ]
}

/// One character's recorded row for one tick: the input the oracle applied, and the resulting
/// state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TraceRow {
    pub input: PlayerInput,
    pub state: CharacterCoreState,
}

/// A full trace: metadata plus one [`TraceRow`] per (tick, character).
#[derive(Debug, Clone, PartialEq)]
pub struct Trace {
    pub metadata: TraceMetadata,
    /// Character ids, in the fixed order every tick's rows are stored in (matches the
    /// scenario's `characters` order).
    pub character_ids: Vec<u32>,
    /// `rows[tick][character_slot]`.
    pub rows: Vec<Vec<TraceRow>>,
}

impl Trace {
    pub fn ticks(&self) -> usize {
        self.rows.len()
    }

    /// The canonical per-tick state hash (see `docs/formats.md`): FNV-1a 64 over every
    /// character's state fields, in character order then schema order, little-endian bytes.
    /// Returns one hash per tick.
    pub fn tick_hashes(&self) -> Vec<u64> {
        self.rows
            .iter()
            .map(|tick_rows| {
                let mut h = Fnv1a64::new();
                for row in tick_rows {
                    let mut w = Writer::new();
                    row.state.write(&mut w);
                    h.update(&w.into_bytes());
                }
                h.finish()
            })
            .collect()
    }

    pub fn write_bytes(&self) -> Vec<u8> {
        let metadata_json = serde_json::to_string(&self.metadata).expect("TraceMetadata always serializes");
        let mut w = Writer::new();
        w.bytes(MAGIC);
        w.u32(VERSION);
        w.string32(&metadata_json);
        w.u32(self.character_ids.len() as u32);
        for id in &self.character_ids {
            w.u32(*id);
        }
        w.u32(self.rows.len() as u32);
        for tick in &self.rows {
            assert_eq!(
                tick.len(),
                self.character_ids.len(),
                "trace invariant violated: one row per character per tick"
            );
            for row in tick {
                for v in row.input.to_fields() {
                    w.i32(v);
                }
                row.state.write(&mut w);
            }
        }
        w.into_bytes()
    }

    pub fn read_bytes(bytes: &[u8]) -> Result<Self, FormatError> {
        let mut r = Reader::new(bytes);
        r.expect_magic(MAGIC)?;
        let version = r.u32("version")?;
        if version != VERSION {
            return Err(FormatError::UnsupportedVersion {
                format: "trace",
                version,
            });
        }
        let metadata_json = r.string32("metadata json")?;
        let metadata: TraceMetadata = serde_json::from_str(&metadata_json).map_err(|_| FormatError::InvalidJson {
            context: "trace metadata",
        })?;
        // The body below is decoded with a hard-coded layout (`PlayerInput`'s 10 fields,
        // `CharacterCoreState`'s 28), not one driven by `metadata.input_schema`/`state_schema` —
        // those exist so a file is self-describing to a human/other tool, but this reader must
        // still refuse to silently misinterpret a file written to a schema it doesn't actually
        // implement (review round 1, finding F4).
        if metadata.input_schema != input_schema() {
            return Err(FormatError::SchemaMismatch {
                context: "trace input_schema",
            });
        }
        if metadata.state_schema != state_schema() {
            return Err(FormatError::SchemaMismatch {
                context: "trace state_schema",
            });
        }
        let char_count = r.u32("character count")?;
        let mut character_ids = Vec::with_capacity(char_count as usize);
        for _ in 0..char_count {
            character_ids.push(r.u32("character id")?);
        }
        let tick_count = r.u32("tick count")?;
        let mut rows = Vec::with_capacity(tick_count as usize);
        for _ in 0..tick_count {
            let mut tick_rows = Vec::with_capacity(character_ids.len());
            for _ in 0..character_ids.len() {
                let mut fields = [0i32; 10];
                for slot in &mut fields {
                    *slot = r.i32("trace input field")?;
                }
                let input = PlayerInput::from_fields(fields);
                let state = CharacterCoreState::read(&mut r)?;
                tick_rows.push(TraceRow { input, state });
            }
            rows.push(tick_rows);
        }
        r.expect_eof()?;
        Ok(Trace {
            metadata,
            character_ids,
            rows,
        })
    }
}

/// A single mismatching field found by [`diff`].
#[derive(Debug, Clone, PartialEq)]
pub struct Mismatch {
    pub tick: usize,
    pub character_id: u32,
    /// `"input.<field>"` or `"state.<field>"`.
    pub field: String,
    pub value_a: FieldValue,
    pub value_b: FieldValue,
}

impl fmt::Display for Mismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "tick {} character {} field {}: {} != {}",
            self.tick, self.character_id, self.field, self.value_a, self.value_b
        )
    }
}

/// Why two traces couldn't be compared field-by-field at all.
#[derive(Debug, Clone, PartialEq)]
pub enum DiffShapeError {
    TickCountMismatch { a: usize, b: usize },
    CharacterCountMismatch { a: usize, b: usize },
    CharacterIdMismatch { slot: usize, a: u32, b: u32 },
}

impl fmt::Display for DiffShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiffShapeError::TickCountMismatch { a, b } => write!(f, "tick count differs: {a} vs {b}"),
            DiffShapeError::CharacterCountMismatch { a, b } => write!(f, "character count differs: {a} vs {b}"),
            DiffShapeError::CharacterIdMismatch { slot, a, b } => {
                write!(f, "character id at slot {slot} differs: {a} vs {b}")
            }
        }
    }
}

/// The result of comparing two traces field-by-field.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffResult {
    /// The first mismatch encountered, scanning ticks in order, characters in trace order, then
    /// input fields before state fields, both in schema order. `None` means the traces are
    /// identical.
    pub first_mismatch: Option<Mismatch>,
    /// Total number of individual field mismatches across the whole trace (`0` iff identical).
    pub mismatch_count: u64,
}

/// Compares two traces field-by-field, reporting the first mismatching `(tick, character,
/// field)` (with both values) and the total mismatch count.
///
/// Returns `Err` instead if the traces don't even have the same shape (tick count, character
/// count/ids) — in that case no meaningful field-by-field comparison is possible.
pub fn diff(a: &Trace, b: &Trace) -> Result<DiffResult, DiffShapeError> {
    if a.rows.len() != b.rows.len() {
        return Err(DiffShapeError::TickCountMismatch {
            a: a.rows.len(),
            b: b.rows.len(),
        });
    }
    if a.character_ids.len() != b.character_ids.len() {
        return Err(DiffShapeError::CharacterCountMismatch {
            a: a.character_ids.len(),
            b: b.character_ids.len(),
        });
    }
    for (slot, (ia, ib)) in a.character_ids.iter().zip(b.character_ids.iter()).enumerate() {
        if ia != ib {
            return Err(DiffShapeError::CharacterIdMismatch { slot, a: *ia, b: *ib });
        }
    }

    let mut first_mismatch = None;
    let mut mismatch_count: u64 = 0;
    for (tick, (rows_a, rows_b)) in a.rows.iter().zip(b.rows.iter()).enumerate() {
        for (slot, (row_a, row_b)) in rows_a.iter().zip(rows_b.iter()).enumerate() {
            let character_id = a.character_ids[slot];
            for ((name, va), (_, vb)) in input_named_fields(&row_a.input)
                .into_iter()
                .zip(input_named_fields(&row_b.input))
            {
                if !va.bit_eq(&vb) {
                    mismatch_count += 1;
                    if first_mismatch.is_none() {
                        first_mismatch = Some(Mismatch {
                            tick,
                            character_id,
                            field: format!("input.{name}"),
                            value_a: va,
                            value_b: vb,
                        });
                    }
                }
            }
            for ((name, va), (_, vb)) in row_a.state.named_fields().into_iter().zip(row_b.state.named_fields()) {
                if !va.bit_eq(&vb) {
                    mismatch_count += 1;
                    if first_mismatch.is_none() {
                        first_mismatch = Some(Mismatch {
                            tick,
                            character_id,
                            field: format!("state.{name}"),
                            value_a: va,
                            value_b: vb,
                        });
                    }
                }
            }
        }
    }
    Ok(DiffResult {
        first_mismatch,
        mismatch_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metadata() -> TraceMetadata {
        TraceMetadata {
            producer: Producer {
                name: "ddai-trace-oracle-a".to_string(),
                version: "0.1.0".to_string(),
            },
            ddnet: DdnetRef {
                tag: "20.1".to_string(),
                commit: "c9d208138f85755521f16a0096b6fe036c5c8698".to_string(),
            },
            map_sha256: "0".repeat(64),
            scenario: ScenarioRef {
                generator: Some("random-v1".to_string()),
                seed: Some(42),
                scenario_sha256: "1".repeat(64),
            },
            input_schema: input_schema(),
            state_schema: state_schema(),
        }
    }

    fn sample_state(x: f32) -> CharacterCoreState {
        CharacterCoreState {
            pos_x: x,
            pos_y: 100.0,
            vel_x: 0.0,
            vel_y: 0.5,
            hook_pos_x: x,
            hook_pos_y: 100.0,
            hook_dir_x: 0.0,
            hook_dir_y: 0.0,
            hook_tele_base_x: 0.0,
            hook_tele_base_y: 0.0,
            hook_tick: 0,
            hook_state: 0,
            hooked_player: -1,
            active_weapon: 0,
            new_hook: 0,
            jumped: 0,
            jumped_total: 0,
            jumps: 2,
            direction: 0,
            angle: 0,
            triggered_events: 0,
            colliding: 0,
            left_wall: 0,
            move_restrictions: 0,
            solo: 0,
            collision_disabled: 0,
            endless_hook: 0,
            hook_hit_disabled: 0,
        }
    }

    fn sample_trace() -> Trace {
        Trace {
            metadata: sample_metadata(),
            character_ids: vec![0, 1],
            rows: vec![
                vec![
                    TraceRow {
                        input: PlayerInput {
                            direction: 1,
                            ..Default::default()
                        },
                        state: sample_state(100.0),
                    },
                    TraceRow {
                        input: PlayerInput::default(),
                        state: sample_state(200.0),
                    },
                ],
                vec![
                    TraceRow {
                        input: PlayerInput::default(),
                        state: sample_state(101.0),
                    },
                    TraceRow {
                        input: PlayerInput::default(),
                        state: sample_state(200.0),
                    },
                ],
            ],
        }
    }

    #[test]
    fn state_byte_size_matches_schema() {
        assert_eq!(state_schema().len(), STATE_F32_COUNT + STATE_I32_COUNT);
        assert_eq!(STATE_BYTES, STATE_F32_COUNT * 4 + STATE_I32_COUNT * 4);
        let mut w = Writer::new();
        sample_state(0.0).write(&mut w);
        assert_eq!(w.into_bytes().len(), STATE_BYTES);
    }

    #[test]
    fn input_byte_size_matches_schema() {
        assert_eq!(input_schema().len(), 10);
        assert_eq!(INPUT_BYTES, 40);
    }

    #[test]
    fn round_trips() {
        let t = sample_trace();
        let bytes = t.write_bytes();
        assert_eq!(Trace::read_bytes(&bytes).unwrap(), t);
    }

    #[test]
    fn write_bytes_is_deterministic() {
        assert_eq!(sample_trace().write_bytes(), sample_trace().write_bytes());
    }

    #[test]
    fn rejects_mismatched_input_schema() {
        // Review round 1, finding F4: the reader used to decode the body with a hard-coded
        // layout regardless of what the header's schema said — silently ignoring a header that
        // doesn't describe the bytes that actually follow.
        let mut t = sample_trace();
        t.metadata.input_schema.pop();
        let bytes = t.write_bytes();
        assert_eq!(
            Trace::read_bytes(&bytes),
            Err(FormatError::SchemaMismatch {
                context: "trace input_schema"
            })
        );
    }

    #[test]
    fn rejects_mismatched_state_schema() {
        let mut t = sample_trace();
        t.metadata.state_schema[0].1 = "f64".to_string(); // pos_x is really f32
        let bytes = t.write_bytes();
        assert_eq!(
            Trace::read_bytes(&bytes),
            Err(FormatError::SchemaMismatch {
                context: "trace state_schema"
            })
        );
    }

    #[test]
    fn rejects_invalid_metadata_json_with_a_dedicated_error_not_invalid_utf8() {
        let mut w = Writer::new();
        w.bytes(MAGIC);
        w.u32(VERSION);
        w.string32("{ not valid json"); // well-formed UTF-8, not well-formed JSON
        w.u32(0); // character count
        w.u32(0); // tick count
        let bytes = w.into_bytes();
        assert_eq!(
            Trace::read_bytes(&bytes),
            Err(FormatError::InvalidJson {
                context: "trace metadata"
            })
        );
    }

    #[test]
    fn tick_hashes_are_stable_and_change_with_state() {
        let t = sample_trace();
        let hashes = t.tick_hashes();
        assert_eq!(hashes.len(), 2);
        assert_ne!(hashes[0], hashes[1]);
        // Re-hashing the same trace must give the same numbers (determinism).
        assert_eq!(hashes, t.tick_hashes());
    }

    #[test]
    fn tick_hash_depends_on_character_order() {
        let mut t = sample_trace();
        let original = t.tick_hashes();
        t.rows[0].swap(0, 1);
        let swapped = t.tick_hashes();
        assert_ne!(
            original[0], swapped[0],
            "swapping character order within a tick must change its hash"
        );
    }

    #[test]
    fn diff_reports_no_mismatch_for_identical_traces() {
        let t = sample_trace();
        let result = diff(&t, &t).unwrap();
        assert_eq!(result.mismatch_count, 0);
        assert_eq!(result.first_mismatch, None);
    }

    #[test]
    fn diff_reports_first_mismatch_and_total_count() {
        let a = sample_trace();
        let mut b = sample_trace();
        b.rows[0][0].state.pos_x = 999.0;
        b.rows[1][1].state.vel_y = -1.0;
        let result = diff(&a, &b).unwrap();
        assert_eq!(result.mismatch_count, 2);
        let m = result.first_mismatch.unwrap();
        assert_eq!(m.tick, 0);
        assert_eq!(m.character_id, 0);
        assert_eq!(m.field, "state.pos_x");
        assert_eq!(m.value_a, FieldValue::F32(100.0));
        assert_eq!(m.value_b, FieldValue::F32(999.0));
    }

    #[test]
    fn diff_distinguishes_positive_and_negative_zero() {
        let a = sample_trace();
        let mut b = sample_trace();
        b.rows[0][0].state.vel_x = -0.0;
        // sample state has vel_x = 0.0; -0.0 must count as a mismatch (bit-exact comparison).
        let result = diff(&a, &b).unwrap();
        assert_eq!(result.mismatch_count, 1);
    }

    #[test]
    fn diff_reports_shape_mismatch_for_different_tick_counts() {
        let a = sample_trace();
        let mut b = sample_trace();
        b.rows.pop();
        assert_eq!(diff(&a, &b), Err(DiffShapeError::TickCountMismatch { a: 2, b: 1 }));
    }

    #[test]
    fn diff_reports_shape_mismatch_for_different_character_ids() {
        let a = sample_trace();
        let mut b = sample_trace();
        b.character_ids[1] = 7;
        assert_eq!(
            diff(&a, &b),
            Err(DiffShapeError::CharacterIdMismatch { slot: 1, a: 1, b: 7 })
        );
    }

    #[test]
    fn metadata_json_round_trips_through_serde() {
        let m = sample_metadata();
        let json = serde_json::to_string(&m).unwrap();
        let back: TraceMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(m, back);
    }
}
