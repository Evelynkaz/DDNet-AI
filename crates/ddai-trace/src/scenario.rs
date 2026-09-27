//! Scenario v2: a map reference, world/tuning setup, and explicit per-tick inputs for a fixed
//! set of characters. See `docs/formats.md` for the exact binary layout.
//!
//! v2 (review round 1, finding F3) replaces v1's static per-tick `(target_x, target_y)` with an
//! "aim mode" ([`ScenarioInput::aim_slot`]) so a character can aim at another character's *live*
//! position instead of a value baked in at generation time — see [`resolve_input`].

use crate::io::{FormatError, Reader, Writer};

const MAGIC: &[u8; 4] = b"SCN1";
const VERSION: u32 = 2;

/// DDNet's `MAX_CLIENTS` (`src/engine/shared/protocol.h`): the size of
/// `CWorldCore::m_apCharacters` and the exclusive upper bound on a valid character id. A
/// scenario with a character id `>= MAX_CLIENTS`, or two characters sharing an id, is rejected
/// by [`Scenario::read_bytes`] (review round 1, finding F5).
pub const MAX_CLIENTS: u32 = 128;

/// Which map a [`Scenario`] runs on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MapRef {
    /// A `ddai_trace::synthetic` recipe name (e.g. `"arena"`).
    Recipe { name: String },
    /// A path to a rawmap v1 file, relative to whatever the reader considers its map root.
    RawmapFile { path: String },
}

/// A single character's fixed starting position. Characters never respawn within a scenario
/// (Oracle A is core-only, see the task spec) — a scenario's character list and spawn positions
/// are exactly the initial state of `CWorldCore::m_apCharacters`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CharacterSpawn {
    /// Character id (`0..MAX_CLIENTS`), used as the array index into `CWorldCore::m_apCharacters`
    /// and for tuning `CTeamsCore`/player-vs-player checks. Must be `< MAX_CLIENTS` and unique
    /// within a scenario — `Scenario::read_bytes` rejects violations.
    pub id: u32,
    /// Spawn X, in map pixels (not tiles).
    pub spawn_x: i32,
    /// Spawn Y, in map pixels (not tiles).
    pub spawn_y: i32,
}

/// One character's *resolved* input for one tick — what actually gets written into
/// `CCharacterCore::m_Input` (via `CNetObj_PlayerInput`). Field names and order match
/// `CNetObj_PlayerInput` (`src/generated/protocol.h`, generated from `datasrc/network.py`).
///
/// This is what a trace v1 file's "input applied" record is (see `docs/formats.md` §6) — always
/// a concrete, absolute target, never an aim-mode reference. [`ScenarioInput`] is the *stored*,
/// pre-resolution form used by the scenario format; [`resolve_input`] converts one to the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PlayerInput {
    pub direction: i32,
    pub target_x: i32,
    pub target_y: i32,
    pub jump: i32,
    pub fire: i32,
    pub hook: i32,
    pub player_flags: i32,
    pub wanted_weapon: i32,
    pub next_weapon: i32,
    pub prev_weapon: i32,
}

impl PlayerInput {
    /// The 10 fields in wire order, as `i32`s — used by the trace binary format (a trace's
    /// "input applied" record is a `PlayerInput`).
    pub fn to_fields(self) -> [i32; 10] {
        [
            self.direction,
            self.target_x,
            self.target_y,
            self.jump,
            self.fire,
            self.hook,
            self.player_flags,
            self.wanted_weapon,
            self.next_weapon,
            self.prev_weapon,
        ]
    }

    pub fn from_fields(f: [i32; 10]) -> Self {
        PlayerInput {
            direction: f[0],
            target_x: f[1],
            target_y: f[2],
            jump: f[3],
            fire: f[4],
            hook: f[5],
            player_flags: f[6],
            wanted_weapon: f[7],
            next_weapon: f[8],
            prev_weapon: f[9],
        }
    }
}

/// One character's *stored* input recipe for one tick, as scenario v2 files hold it. Use
/// [`resolve_input`] to turn this into the concrete [`PlayerInput`] that actually gets applied on
/// a given tick (the resolution depends on where characters *were* at the end of the previous
/// tick, so it cannot be done once at scenario-generation time — see [`resolve_input`]'s doc
/// comment and `docs/formats.md` §2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScenarioInput {
    pub direction: i32,
    /// Explicit target (`aim_slot == -1`) or integer aiming noise added to the live
    /// vector-to-`aim_slot` (`aim_slot >= 0`) — see [`resolve_input`].
    pub target_x: i32,
    pub target_y: i32,
    /// `-1`: `(target_x, target_y)` is an explicit target, used directly.
    /// `0..characters.len()`: aim at that character slot's previous-tick position, offset by
    /// `(target_x, target_y)` as noise. See [`resolve_input`].
    pub aim_slot: i32,
    pub jump: i32,
    pub fire: i32,
    pub hook: i32,
    pub player_flags: i32,
    pub wanted_weapon: i32,
    pub next_weapon: i32,
    pub prev_weapon: i32,
}

impl Default for ScenarioInput {
    fn default() -> Self {
        // `aim_slot: -1` (explicit-target mode), not `0` (aim at character 0) — `i32`'s own
        // `Default` (`0`) would silently mean something completely different here.
        ScenarioInput {
            direction: 0,
            target_x: 0,
            target_y: 0,
            aim_slot: -1,
            jump: 0,
            fire: 0,
            hook: 0,
            player_flags: 0,
            wanted_weapon: 0,
            next_weapon: 0,
            prev_weapon: 0,
        }
    }
}

impl ScenarioInput {
    fn to_fields(self) -> [i32; 11] {
        [
            self.direction,
            self.target_x,
            self.target_y,
            self.aim_slot,
            self.jump,
            self.fire,
            self.hook,
            self.player_flags,
            self.wanted_weapon,
            self.next_weapon,
            self.prev_weapon,
        ]
    }

    fn from_fields(f: [i32; 11]) -> Self {
        ScenarioInput {
            direction: f[0],
            target_x: f[1],
            target_y: f[2],
            aim_slot: f[3],
            jump: f[4],
            fire: f[5],
            hook: f[6],
            player_flags: f[7],
            wanted_weapon: f[8],
            next_weapon: f[9],
            prev_weapon: f[10],
        }
    }

    fn write(&self, w: &mut Writer) {
        for v in self.to_fields() {
            w.i32(v);
        }
    }

    fn read(r: &mut Reader) -> Result<Self, FormatError> {
        let mut f = [0i32; 11];
        for slot in &mut f {
            *slot = r.i32("scenario input field")?;
        }
        Ok(ScenarioInput::from_fields(f))
    }
}

/// Resolves one character's stored [`ScenarioInput`] for one tick into the concrete
/// [`PlayerInput`] that gets applied to `CCharacterCore::m_Input` — the shared implementation of
/// scenario v2's "aim mode" (review round 1, finding F3; see `docs/formats.md` §2.1). Both this
/// function and `tools/ddnet-oracle/oracle_core.cpp`'s tick loop implement the exact same
/// algorithm independently; task 1.3's Rust physics port should call this function directly
/// rather than re-deriving it.
///
/// - `input.aim_slot == -1`: `(input.target_x, input.target_y)` is used directly, as an explicit
///   target.
/// - `input.aim_slot == k` (`0 <= k < prev_positions.len()`): the target is
///   `(pos[k] - pos[self_slot]) + (input.target_x, input.target_y)`, where `pos[..]` are the
///   *previous* tick's post-`Quantize()` integer positions (`prev_positions` — at tick 0 there is
///   no previous tick, so callers pass each character's spawn position instead) and
///   `(input.target_x, input.target_y)` is reinterpreted as small integer aiming noise on top of
///   that live vector. If the result is exactly `(0, 0)` (character `k` is exactly on top of
///   `self_slot` and the noise happens to cancel out), falls back to
///   `(input.target_x, input.target_y)` verbatim — exactly the `aim_slot == -1` behavior.
///   Callers/generators must make sure that fallback value is itself never `(0, 0)`: DDNet's
///   `CNetObj_PlayerInput` forbids aiming exactly at the center (see `docs/formats.md`).
///
/// All arithmetic is plain `i32` (never float): every position involved is always exactly an
/// integer (a spawn position, or a `Quantize()`d one — `Quantize()` rounds `m_Pos` to whole
/// pixels), so there is nothing for float rounding to get subtly wrong between this and
/// `oracle_core.cpp`'s reimplementation.
///
/// # Panics
///
/// Panics if `aim_slot` is neither `-1` nor a valid index into `prev_positions`, or if
/// `self_slot` is out of bounds for `prev_positions`. Both readers (`Scenario::read_bytes`, the
/// C++ oracle) reject an out-of-range `aim_slot` before it can ever reach this function, so a
/// panic here means a caller built a `ScenarioInput`/`prev_positions` pair by hand without
/// validating it first — a programmer error, not a data error.
pub fn resolve_input(input: &ScenarioInput, self_slot: usize, prev_positions: &[(i32, i32)]) -> PlayerInput {
    let (target_x, target_y) = if input.aim_slot >= 0 {
        let k = input.aim_slot as usize;
        assert!(k < prev_positions.len(), "resolve_input: aim_slot {k} out of bounds");
        assert!(
            self_slot < prev_positions.len(),
            "resolve_input: self_slot {self_slot} out of bounds"
        );
        let (px, py) = prev_positions[k];
        let (sx, sy) = prev_positions[self_slot];
        let bx = px - sx + input.target_x;
        let by = py - sy + input.target_y;
        if bx == 0 && by == 0 {
            (input.target_x, input.target_y)
        } else {
            (bx, by)
        }
    } else {
        (input.target_x, input.target_y)
    };
    PlayerInput {
        direction: input.direction,
        target_x,
        target_y,
        jump: input.jump,
        fire: input.fire,
        hook: input.hook,
        player_flags: input.player_flags,
        wanted_weapon: input.wanted_weapon,
        next_weapon: input.next_weapon,
        prev_weapon: input.prev_weapon,
    }
}

/// A named tuning override, applied on top of `CTuningParams::DEFAULT` before any tick runs.
///
/// Stored as `(name, value * 100)` — the same fixed-point representation `CTuneParam` uses
/// internally (`m_Value = (int)(v * 100.0f)`) — rather than as a float, so a scenario file's
/// bytes (and its sha256) never depend on how some writer chose to round a float to text or
/// back; `name` matches a `CTuningParams` script name (e.g. `"gravity"`, `"hook_length"`, see
/// `src/game/tuning.h`) and is resolved with `CTuningParams::Set(pName, Value)` by the oracle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuningOverride {
    pub name: String,
    pub value_x100: i32,
}

/// A fully-specified scenario: reproducible inputs for a fixed cast of characters over a fixed
/// number of ticks, on a fixed map.
#[derive(Debug, Clone, PartialEq)]
pub struct Scenario {
    pub map_ref: MapRef,
    /// sha256 of the rawmap v1 bytes of the map `map_ref` resolves to, pinned here so a reader
    /// can catch a stale/mismatched map without re-running whatever generated it.
    pub map_sha256: [u8; 32],
    /// Mirrors `sv_no_weak_hook`: when set, `CGameWorld::Tick` runs a separate `PreTick` pass
    /// over every character before any character's deferred (player-vs-player) tick.
    pub no_weak_hook: bool,
    pub tuning_overrides: Vec<TuningOverride>,
    /// Characters in the fixed order used for both world-tick processing (see
    /// `docs/formats.md` for why the oracle ticks characters newest-spawned-first) and for the
    /// per-tick input rows below. Every id must be `< MAX_CLIENTS` and unique.
    pub characters: Vec<CharacterSpawn>,
    /// `inputs[tick][character_slot]`, `character_slot` indexing into `characters` (not into
    /// `characters[..].id`). Every `aim_slot` must be `-1` or a valid index into `characters`.
    pub inputs: Vec<Vec<ScenarioInput>>,
}

impl Scenario {
    pub fn ticks(&self) -> usize {
        self.inputs.len()
    }

    /// Each character's spawn position, as the `prev_positions` [`resolve_input`] expects for
    /// tick 0 (there is no tick -1, so tick 0 resolves against spawn positions instead).
    pub fn spawn_positions(&self) -> Vec<(i32, i32)> {
        self.characters.iter().map(|c| (c.spawn_x, c.spawn_y)).collect()
    }

    pub fn write_bytes(&self) -> Vec<u8> {
        self.validate()
            .expect("Scenario must be internally consistent before it can be serialized");

        let mut w = Writer::new();
        w.bytes(MAGIC);
        w.u32(VERSION);
        match &self.map_ref {
            MapRef::Recipe { name } => {
                w.u8(0);
                w.string16(name);
            }
            MapRef::RawmapFile { path } => {
                w.u8(1);
                w.string16(path);
            }
        }
        w.bytes(&self.map_sha256);
        w.u8(self.no_weak_hook as u8);
        w.u32(self.tuning_overrides.len() as u32);
        for t in &self.tuning_overrides {
            w.string16(&t.name);
            w.i32(t.value_x100);
        }
        w.u32(self.characters.len() as u32);
        for c in &self.characters {
            w.u32(c.id);
            w.i32(c.spawn_x);
            w.i32(c.spawn_y);
        }
        w.u32(self.inputs.len() as u32);
        for tick in &self.inputs {
            for input in tick {
                input.write(&mut w);
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
                format: "scenario",
                version,
            });
        }
        let map_ref_tag = r.u8("map ref tag")?;
        let map_ref = match map_ref_tag {
            0 => MapRef::Recipe {
                name: r.string16("map ref recipe name")?,
            },
            1 => MapRef::RawmapFile {
                path: r.string16("map ref rawmap path")?,
            },
            other => {
                return Err(FormatError::InvalidValue {
                    context: "map ref tag",
                    value: other as i64,
                });
            }
        };
        let map_sha256 = r.array32::<32>("map sha256")?;
        let no_weak_hook = r.u8("no_weak_hook")? != 0;
        let override_count = r.u32("tuning override count")?;
        let mut tuning_overrides = Vec::with_capacity(override_count as usize);
        for _ in 0..override_count {
            let name = r.string16("tuning override name")?;
            let value_x100 = r.i32("tuning override value")?;
            tuning_overrides.push(TuningOverride { name, value_x100 });
        }
        let char_count = r.u32("character count")?;
        let mut characters = Vec::with_capacity(char_count as usize);
        for _ in 0..char_count {
            let id = r.u32("character id")?;
            let spawn_x = r.i32("character spawn x")?;
            let spawn_y = r.i32("character spawn y")?;
            characters.push(CharacterSpawn { id, spawn_x, spawn_y });
        }
        let tick_count = r.u32("tick count")?;
        let mut inputs = Vec::with_capacity(tick_count as usize);
        for _ in 0..tick_count {
            let mut tick = Vec::with_capacity(characters.len());
            for _ in 0..characters.len() {
                tick.push(ScenarioInput::read(&mut r)?);
            }
            inputs.push(tick);
        }
        r.expect_eof()?;
        let scenario = Scenario {
            map_ref,
            map_sha256,
            no_weak_hook,
            tuning_overrides,
            characters,
            inputs,
        };
        scenario
            .validate()
            .map_err(|(context, value)| FormatError::InvalidValue { context, value })?;
        Ok(scenario)
    }

    /// Checks the invariants [`write_bytes`](Self::write_bytes)/[`read_bytes`](Self::read_bytes)
    /// both rely on: every tick has exactly one input per character (review round 1 kept this
    /// invariant from v1), every character id is `< MAX_CLIENTS` and unique (finding F5), and
    /// every input's `aim_slot` is `-1` or a valid character slot index (finding F3/F5).
    pub fn validate(&self) -> Result<(), (&'static str, i64)> {
        for c in &self.characters {
            if c.id >= MAX_CLIENTS {
                return Err(("character id must be < MAX_CLIENTS (128)", c.id as i64));
            }
        }
        for (i, a) in self.characters.iter().enumerate() {
            for b in &self.characters[..i] {
                if a.id == b.id {
                    return Err(("duplicate character id", a.id as i64));
                }
            }
        }
        for tick in &self.inputs {
            if tick.len() != self.characters.len() {
                return Err((
                    "every tick must have exactly one input per character",
                    tick.len() as i64,
                ));
            }
            for input in tick {
                if input.aim_slot < -1 || input.aim_slot >= self.characters.len() as i32 {
                    return Err((
                        "aim_slot must be -1 or a valid character slot index",
                        input.aim_slot as i64,
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Scenario {
        Scenario {
            map_ref: MapRef::Recipe {
                name: "arena".to_string(),
            },
            map_sha256: [0x42; 32],
            no_weak_hook: true,
            tuning_overrides: vec![
                TuningOverride {
                    name: "gravity".to_string(),
                    value_x100: 25,
                },
                TuningOverride {
                    name: "hook_length".to_string(),
                    value_x100: 38000,
                },
            ],
            characters: vec![
                CharacterSpawn {
                    id: 0,
                    spawn_x: 100,
                    spawn_y: 200,
                },
                CharacterSpawn {
                    id: 1,
                    spawn_x: 132,
                    spawn_y: 200,
                },
            ],
            inputs: vec![
                vec![
                    ScenarioInput {
                        direction: 1,
                        target_x: 10,
                        target_y: -5,
                        ..Default::default()
                    },
                    ScenarioInput {
                        hook: 1,
                        aim_slot: 0,
                        target_x: -3,
                        target_y: 7,
                        ..Default::default()
                    },
                ],
                vec![ScenarioInput::default(), ScenarioInput::default()],
            ],
        }
    }

    #[test]
    fn round_trips() {
        let s = sample();
        let bytes = s.write_bytes();
        let back = Scenario::read_bytes(&bytes).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn write_bytes_is_deterministic() {
        assert_eq!(sample().write_bytes(), sample().write_bytes());
    }

    #[test]
    fn rawmap_file_ref_round_trips() {
        let mut s = sample();
        s.map_ref = MapRef::RawmapFile {
            path: "maps/foo.rawmap".to_string(),
        };
        let bytes = s.write_bytes();
        assert_eq!(Scenario::read_bytes(&bytes).unwrap().map_ref, s.map_ref);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = sample().write_bytes();
        bytes[0] = b'X';
        assert!(Scenario::read_bytes(&bytes).is_err());
    }

    #[test]
    fn rejects_old_v1_version_number() {
        let mut bytes = sample().write_bytes();
        bytes[4] = 1; // version is the first LE u32 after the 4-byte magic
        assert_eq!(
            Scenario::read_bytes(&bytes),
            Err(FormatError::UnsupportedVersion {
                format: "scenario",
                version: 1
            })
        );
    }

    #[test]
    fn rejects_truncated_input() {
        let bytes = sample().write_bytes();
        let truncated = &bytes[..bytes.len() - 1];
        assert!(Scenario::read_bytes(truncated).is_err());
    }

    #[test]
    #[should_panic(expected = "one input per character")]
    fn write_bytes_panics_on_malformed_scenario() {
        let mut s = sample();
        s.inputs[0].pop();
        s.write_bytes();
    }

    #[test]
    fn empty_ticks_round_trip() {
        let mut s = sample();
        s.inputs.clear();
        let bytes = s.write_bytes();
        assert_eq!(Scenario::read_bytes(&bytes).unwrap(), s);
    }

    #[test]
    fn rejects_character_id_at_or_above_max_clients() {
        let mut s = sample();
        s.characters[1].id = MAX_CLIENTS;
        let bytes = {
            // Bypass `write_bytes`' own validation (it would panic first) to build bytes that
            // exercise `read_bytes`' independent validation, as an external/adversarial file
            // would.
            let mut w = Writer::new();
            w.bytes(MAGIC).u32(VERSION);
            w.u8(0).string16("arena");
            w.bytes(&s.map_sha256);
            w.u8(0).u32(0);
            w.u32(s.characters.len() as u32);
            for c in &s.characters {
                w.u32(c.id).i32(c.spawn_x).i32(c.spawn_y);
            }
            w.u32(0);
            w.into_bytes()
        };
        assert_eq!(
            Scenario::read_bytes(&bytes),
            Err(FormatError::InvalidValue {
                context: "character id must be < MAX_CLIENTS (128)",
                value: MAX_CLIENTS as i64
            })
        );
    }

    #[test]
    fn rejects_duplicate_character_ids() {
        let mut s = sample();
        s.characters[1].id = s.characters[0].id;
        assert_eq!(s.validate(), Err(("duplicate character id", s.characters[0].id as i64)));
    }

    #[test]
    fn rejects_out_of_range_aim_slot() {
        let mut s = sample();
        s.inputs[0][0].aim_slot = 2; // only slots 0 and 1 exist
        assert_eq!(
            s.validate(),
            Err(("aim_slot must be -1 or a valid character slot index", 2))
        );
    }

    #[test]
    fn accepts_aim_slot_minus_one_and_valid_slots() {
        let mut s = sample();
        s.inputs[0][0].aim_slot = -1;
        s.inputs[0][1].aim_slot = 0;
        assert!(s.validate().is_ok());
    }

    #[test]
    fn spawn_positions_matches_characters() {
        let s = sample();
        assert_eq!(s.spawn_positions(), vec![(100, 200), (132, 200)]);
    }
}

#[cfg(test)]
mod resolve_input_tests {
    use super::*;

    #[test]
    fn explicit_target_mode_uses_target_verbatim() {
        let input = ScenarioInput {
            target_x: 123,
            target_y: -45,
            aim_slot: -1,
            ..Default::default()
        };
        let resolved = resolve_input(&input, 0, &[(0, 0), (500, 500)]);
        assert_eq!((resolved.target_x, resolved.target_y), (123, -45));
    }

    #[test]
    fn aim_slot_mode_targets_the_live_vector_plus_noise() {
        // self at (100, 100), target character at (150, 80): live vector is (50, -20).
        let input = ScenarioInput {
            aim_slot: 1,
            target_x: 5,
            target_y: 5,
            ..Default::default()
        };
        let resolved = resolve_input(&input, 0, &[(100, 100), (150, 80)]);
        assert_eq!((resolved.target_x, resolved.target_y), (55, -15));
    }

    #[test]
    fn aim_slot_mode_tracks_live_position_across_ticks() {
        let input = ScenarioInput {
            aim_slot: 1,
            target_x: 0,
            target_y: 0,
            ..Default::default()
        };
        // Tick 0: target character at (200, 100), self at (100, 100) -> vector (100, 0).
        let r0 = resolve_input(&input, 0, &[(100, 100), (200, 100)]);
        assert_eq!((r0.target_x, r0.target_y), (100, 0));
        // Tick 1: target character has moved to (100, 300); the SAME stored input now resolves
        // to a completely different vector, without the scenario needing to change at all.
        let r1 = resolve_input(&input, 0, &[(100, 100), (100, 300)]);
        assert_eq!((r1.target_x, r1.target_y), (0, 200));
    }

    #[test]
    fn falls_back_to_explicit_target_when_live_vector_plus_noise_is_exactly_zero() {
        // self and target character at the same position, noise (0, 0) -> would resolve to
        // (0, 0), which is forbidden; must fall back to the stored (target_x, target_y).
        let input = ScenarioInput {
            aim_slot: 1,
            target_x: 7,
            target_y: -3,
            ..Default::default()
        };
        let resolved = resolve_input(&input, 0, &[(50, 50), (50, 50)]);
        assert_eq!((resolved.target_x, resolved.target_y), (7, -3));
    }

    #[test]
    fn other_fields_pass_through_unchanged() {
        let input = ScenarioInput {
            direction: -1,
            jump: 1,
            fire: 3,
            hook: 1,
            player_flags: 2,
            wanted_weapon: 4,
            next_weapon: 5,
            prev_weapon: 6,
            aim_slot: -1,
            target_x: 1,
            target_y: 1,
        };
        let resolved = resolve_input(&input, 0, &[(0, 0)]);
        assert_eq!(resolved.direction, -1);
        assert_eq!(resolved.jump, 1);
        assert_eq!(resolved.fire, 3);
        assert_eq!(resolved.hook, 1);
        assert_eq!(resolved.player_flags, 2);
        assert_eq!(resolved.wanted_weapon, 4);
        assert_eq!(resolved.next_weapon, 5);
        assert_eq!(resolved.prev_weapon, 6);
    }

    #[test]
    #[should_panic(expected = "aim_slot")]
    fn panics_on_out_of_bounds_aim_slot() {
        let input = ScenarioInput {
            aim_slot: 5,
            ..Default::default()
        };
        resolve_input(&input, 0, &[(0, 0)]);
    }
}
