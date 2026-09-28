//! A streaming reader for Oracle B `trace-b` v2 files (task 1.5, `docs/formats.md` §11/§12).
//!
//! Deliberately streams tick-by-tick from a buffered file handle rather than reading the whole
//! file into memory first (the task's own instruction: "the Oracle B corpus is large — stream
//! traces; never load whole"). Peak extra memory per read is one fixed-size record (at most a
//! few hundred bytes), never the file's full size (individual files in this corpus run up to
//! ~33 MiB).
//!
//! This reader only extracts the fields the live map view actually renders (position, aim,
//! hook, freeze state, a handful of DDRace flags) — every other documented field (weapon ammo,
//! ninja state, tele-gun flags, ...) is read and discarded to keep the stream position correct,
//! never skipped via a wrong byte count. Field offsets are exact byte-for-byte quotes of
//! `tools/ddnet-oracle/server/oracle_server.cpp`'s `CoreStateFields`/`DDRaceStateFields::Write`
//! (cited inline below), not a re-derivation from the prose in `docs/formats.md` alone.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use ddai_trace::io::{FormatError, Reader};

const MAGIC: &[u8; 4] = b"TRB1";
const SUPPORTED_VERSION: u32 = 2;

/// DDNet's `MAX_CLIENTS` (see `ddai_trace::scenario::MAX_CLIENTS`) — the same bound applied here
/// to `character_count` before it is ever used to size a `Vec` or drive a read loop, so a
/// corrupt/adversarial header can reject early instead of attempting a huge allocation or a
/// near-infinite per-tick read loop.
const MAX_CLIENTS: u32 = 128;
/// Sanity cap on `metadata_len` — the metadata JSON in every real trace is a few hundred bytes;
/// this is generous headroom, not a tight fit, while still bounding a corrupt file's claimed
/// length.
const MAX_METADATA_BYTES: u32 = 1024 * 1024;
/// Sanity cap on `switch_team_count * switch_highest_number` (the per-tick switch table size) —
/// bounds both the multiplication itself (checked, see [`TraceBReader::open`]) and the amount of
/// data one `next_tick()` call will skip over.
const MAX_SWITCH_ENTRIES: u64 = 10_000_000;
/// Sanity cap on one tick's `entity_count` (projectiles/lasers/map fixtures) — bounds the amount
/// of data one `next_tick()` call will skip over for a single tick's entity list.
const MAX_ENTITIES_PER_TICK: u32 = 1_000_000;

const SWITCH_ENTRY_BYTES: u64 = 16; // 4 x i32 (see `SwitchEntry::Write` in oracle_server.cpp)
const ENTITY_RECORD_BYTES: u64 = 36; // 9 x 4 bytes (see `EntityRecord::Write`)
/// `PlayerInput` (11 x i32) + `CoreStateFields` (28 fields: 10 f32 + 18 i32) +
/// `DDRaceStateFields` (54 fields: 52 i32 + 2 f32), all i32/f32 fields being 4 bytes.
const CHARACTER_ROW_BYTES: usize = 11 * 4 + 28 * 4 + 54 * 4;

#[derive(Debug, thiserror::Error)]
pub enum TraceBError {
    #[error("format error: {0}")]
    Format(#[from] FormatError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("unsupported trace-b version {0} (only {SUPPORTED_VERSION} is supported)")]
    UnsupportedVersion(u32),
    #[error("character_count {0} exceeds the {MAX_CLIENTS} MAX_CLIENTS cap")]
    TooManyCharacters(u32),
    #[error("switch_team_count * switch_highest_number ({0}) exceeds the {MAX_SWITCH_ENTRIES} cap")]
    TooManySwitchEntries(u64),
    #[error("tick {tick}'s entity_count ({count}) exceeds the {MAX_ENTITIES_PER_TICK} cap")]
    TooManyEntities { tick: u32, count: u32 },
    #[error("metadata is not valid JSON: {0}")]
    InvalidMetadataJson(String),
    #[error("metadata is missing or has the wrong type for field {0:?}")]
    MissingMetadataField(&'static str),
    #[error("metadata's map_sha256 is not 64 hex characters")]
    InvalidMapSha256,
}

/// The subset of a trace-b file's metadata JSON (`docs/formats.md` §11.1) this reader needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceBMetadata {
    pub map_sha256: [u8; 32],
    /// `"real-map"` or `"rawmap-scenario"` (`docs/formats.md` §11.1).
    pub mode: String,
    /// Only present when `mode == "real-map"`. An absolute path from whatever machine originally
    /// ran the oracle — **never used as a filesystem path by this reader or any caller**; see
    /// `crate::live::map_resolve`'s doc comment for why, and how it's actually used (only its
    /// filename, resolved inside a configured directory).
    pub real_map_path: Option<String>,
}

fn parse_metadata(bytes: &[u8]) -> Result<TraceBMetadata, TraceBError> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| TraceBError::InvalidMetadataJson(e.to_string()))?;
    let map_sha256_hex = value
        .get("map_sha256")
        .and_then(|v| v.as_str())
        .ok_or(TraceBError::MissingMetadataField("map_sha256"))?;
    let map_sha256 = decode_hex_sha256(map_sha256_hex)?;
    let mode = value
        .get("mode")
        .and_then(|v| v.as_str())
        .ok_or(TraceBError::MissingMetadataField("mode"))?
        .to_string();
    let real_map_path = value
        .get("real_map_path")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    Ok(TraceBMetadata {
        map_sha256,
        mode,
        real_map_path,
    })
}

/// Review finding F1 (round 1): the previous version of this function sliced the `&str` by byte
/// range (`&hex[i*2..i*2+2]`) — `str` indexing panics if a range boundary doesn't fall on a UTF-8
/// character boundary, which a `map_sha256` field containing any multi-byte character can trigger
/// even though `hex.len()` (a *byte* count) still equals 64 (e.g. `"a\u{e9}" + "0".repeat(61)`:
/// 1 + 2 + 61 = 64 bytes, but byte offset 2 lands inside the 2-byte `\u{e9}`). Untrusted trace
/// metadata must never be able to panic this task (acceptance criterion 2) — this now works
/// entirely on `hex.as_bytes()` (always safe to index/slice, `u8` has no "boundary" concept) and
/// explicitly rejects any non-ASCII-hex byte instead of assuming the input is well-formed ASCII.
fn decode_hex_sha256(hex: &str) -> Result<[u8; 32], TraceBError> {
    let bytes = hex.as_bytes();
    if bytes.len() != 64 {
        return Err(TraceBError::InvalidMapSha256);
    }
    fn hex_val(b: u8) -> Result<u8, TraceBError> {
        match b {
            b'0'..=b'9' => Ok(b - b'0'),
            b'a'..=b'f' => Ok(b - b'a' + 10),
            b'A'..=b'F' => Ok(b - b'A' + 10),
            _ => Err(TraceBError::InvalidMapSha256),
        }
    }
    let mut out = [0u8; 32];
    for (i, out_byte) in out.iter_mut().enumerate() {
        let high = hex_val(bytes[i * 2])?;
        let low = hex_val(bytes[i * 2 + 1])?;
        *out_byte = (high << 4) | low;
    }
    Ok(out)
}

/// A trace-b file's fixed header (everything before the first tick's data).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceBHeader {
    pub metadata: TraceBMetadata,
    /// Character ids in file order — a [`TraceBTick`]'s `characters` are in this same order.
    pub character_ids: Vec<u32>,
    pub switch_highest_number: u32,
    pub switch_team_count: u32,
    pub tick_count: u32,
}

/// One character's rendering-relevant fields for one tick, decoded from its 372-byte row. Field
/// names mirror `docs/formats.md`'s tables (§6.2 for the `core_*`-prefixed ones, §11.3 for the
/// `ddrace_*`-prefixed ones) rather than `crate::live::source::CharacterState`'s names directly —
/// converting to a `CharacterState` (deciding `frozen`/`alive`/etc. from these raw values) is
/// `crate::live::replay`'s job, one layer up, so this module stays a faithful, semantics-free
/// transcription of the file format alone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TraceBCharacterRow {
    pub input_target_x: i32,
    pub input_target_y: i32,
    pub core_pos_x: f32,
    pub core_pos_y: f32,
    pub core_hook_pos_x: f32,
    pub core_hook_pos_y: f32,
    pub core_hook_state: i32,
    pub core_hooked_player: i32,
    pub core_active_weapon: i32,
    pub core_triggered_events: i32,
    pub ddrace_alive: i32,
    pub ddrace_died_this_tick: i32,
    pub ddrace_respawned_this_tick: i32,
    pub ddrace_is_in_freeze: i32,
    pub ddrace_deep_frozen: i32,
    pub ddrace_live_frozen: i32,
    pub ddrace_team: i32,
}

/// One tick: the game tick number and every character's row, in header `character_ids` order.
/// Per-tick switch/entity data is read (to keep the stream position correct) but not retained —
/// see this module's doc comment.
#[derive(Debug, Clone, PartialEq)]
pub struct TraceBTick {
    pub game_tick: i32,
    pub characters: Vec<TraceBCharacterRow>,
}

/// `COREEVENT_HOOK_ATTACH_PLAYER` (`gamecore.h`) — a bit in `core_triggered_events`, exposed here
/// (not decoded into a bool field above) so callers needing it don't need to hardcode the bit
/// value themselves.
pub const COREEVENT_HOOK_ATTACH_PLAYER: i32 = 0x08;

#[derive(Debug)]
pub struct TraceBReader {
    file: BufReader<File>,
    header: TraceBHeader,
    ticks_read: u32,
}

impl TraceBReader {
    pub fn open(path: &Path) -> Result<Self, TraceBError> {
        let file = File::open(path)?;
        let mut file = BufReader::new(file);

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(FormatError::BadMagic {
                expected: MAGIC,
                actual: magic.to_vec(),
            }
            .into());
        }
        let version = read_u32(&mut file)?;
        if version != SUPPORTED_VERSION {
            return Err(TraceBError::UnsupportedVersion(version));
        }
        let metadata_len = read_u32(&mut file)?;
        if metadata_len > MAX_METADATA_BYTES {
            return Err(FormatError::InvalidValue {
                context: "trace-b metadata_len",
                value: metadata_len as i64,
            }
            .into());
        }
        let mut metadata_bytes = vec![0u8; metadata_len as usize];
        file.read_exact(&mut metadata_bytes)?;
        let metadata = parse_metadata(&metadata_bytes)?;

        let character_count = read_u32(&mut file)?;
        if character_count > MAX_CLIENTS {
            return Err(TraceBError::TooManyCharacters(character_count));
        }
        let mut character_ids = Vec::with_capacity(character_count as usize);
        for _ in 0..character_count {
            character_ids.push(read_u32(&mut file)?);
        }

        let switch_highest_number = read_u32(&mut file)?;
        let switch_team_count = read_u32(&mut file)?;
        let switch_entry_count = (switch_highest_number as u64)
            .checked_mul(switch_team_count as u64)
            .ok_or(TraceBError::TooManySwitchEntries(u64::MAX))?;
        if switch_entry_count > MAX_SWITCH_ENTRIES {
            return Err(TraceBError::TooManySwitchEntries(switch_entry_count));
        }
        for _ in 0..switch_team_count {
            let _team_id = read_i32(&mut file)?;
        }

        let tick_count = read_u32(&mut file)?;

        let header = TraceBHeader {
            metadata,
            character_ids,
            switch_highest_number,
            switch_team_count,
            tick_count,
        };
        Ok(TraceBReader {
            file,
            header,
            ticks_read: 0,
        })
    }

    pub fn header(&self) -> &TraceBHeader {
        &self.header
    }

    /// Bytes to skip per tick for the switch table (fixed per file, computed once).
    fn switch_bytes_per_tick(&self) -> u64 {
        (self.header.switch_highest_number as u64) * (self.header.switch_team_count as u64) * SWITCH_ENTRY_BYTES
    }

    /// Reads and decodes the next tick, or `Ok(None)` at end of file. A malformed tick (a read
    /// past EOF mid-record, an out-of-range `entity_count`) returns `Err`, never a panic
    /// (acceptance criterion 2: "a malformed trace gives an error event, not a panic") — callers
    /// (`crate::live::replay`) turn this into a [`crate::live::source::SourceEvent::Error`] and
    /// move on to the next trace file, rather than propagating a panic into the whole server.
    pub fn next_tick(&mut self) -> Result<Option<TraceBTick>, TraceBError> {
        if self.ticks_read >= self.header.tick_count {
            return Ok(None);
        }
        let game_tick = match read_i32(&mut self.file) {
            Ok(v) => v,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.into()),
        };

        // Switch table: skip (see this module's doc comment — not needed for rendering yet).
        let switch_bytes = self.switch_bytes_per_tick();
        seek_forward(&mut self.file, switch_bytes)?;

        let entity_count = read_u32(&mut self.file)?;
        if entity_count > MAX_ENTITIES_PER_TICK {
            return Err(TraceBError::TooManyEntities {
                tick: game_tick as u32,
                count: entity_count,
            });
        }
        seek_forward(&mut self.file, entity_count as u64 * ENTITY_RECORD_BYTES)?;

        let mut characters = Vec::with_capacity(self.header.character_ids.len());
        let mut row_buf = [0u8; CHARACTER_ROW_BYTES];
        for _ in 0..self.header.character_ids.len() {
            self.file.read_exact(&mut row_buf)?;
            characters.push(decode_character_row(&row_buf)?);
        }

        self.ticks_read += 1;
        Ok(Some(TraceBTick { game_tick, characters }))
    }
}

fn seek_forward(file: &mut BufReader<File>, bytes: u64) -> Result<(), std::io::Error> {
    if bytes == 0 {
        return Ok(());
    }
    file.seek(SeekFrom::Current(bytes as i64)).map(|_| ())
}

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_i32(r: &mut impl Read) -> std::io::Result<i32> {
    read_u32(r).map(|v| v as i32)
}

fn decode_character_row(buf: &[u8; CHARACTER_ROW_BYTES]) -> Result<TraceBCharacterRow, TraceBError> {
    let mut r = Reader::new(buf);

    // PlayerInput (11 x i32): direction, target_x, target_y, jump, fire, hook, player_flags,
    // wanted_weapon, next_weapon, prev_weapon, kill.
    let _direction = r.i32("input direction")?;
    let input_target_x = r.i32("input target_x")?;
    let input_target_y = r.i32("input target_y")?;
    let _jump = r.i32("input jump")?;
    let _fire = r.i32("input fire")?;
    let _hook = r.i32("input hook")?;
    let _player_flags = r.i32("input player_flags")?;
    let _wanted_weapon = r.i32("input wanted_weapon")?;
    let _next_weapon = r.i32("input next_weapon")?;
    let _prev_weapon = r.i32("input prev_weapon")?;
    let _kill = r.i32("input kill")?;

    // CoreStateFields: 10 f32 then 18 i32, in exactly this order (see this module's doc comment
    // for the citation).
    let core_pos_x = r.f32("core pos_x")?;
    let core_pos_y = r.f32("core pos_y")?;
    let _vel_x = r.f32("core vel_x")?;
    let _vel_y = r.f32("core vel_y")?;
    let core_hook_pos_x = r.f32("core hook_pos_x")?;
    let core_hook_pos_y = r.f32("core hook_pos_y")?;
    let _hook_dir_x = r.f32("core hook_dir_x")?;
    let _hook_dir_y = r.f32("core hook_dir_y")?;
    let _hook_tele_base_x = r.f32("core hook_tele_base_x")?;
    let _hook_tele_base_y = r.f32("core hook_tele_base_y")?;
    let _hook_tick = r.i32("core hook_tick")?;
    let core_hook_state = r.i32("core hook_state")?;
    let core_hooked_player = r.i32("core hooked_player")?;
    let core_active_weapon = r.i32("core active_weapon")?;
    let _new_hook = r.i32("core new_hook")?;
    let _jumped = r.i32("core jumped")?;
    let _jumped_total = r.i32("core jumped_total")?;
    let _jumps = r.i32("core jumps")?;
    let _direction2 = r.i32("core direction")?;
    let _angle = r.i32("core angle")?;
    let core_triggered_events = r.i32("core triggered_events")?;
    let _colliding = r.i32("core colliding")?;
    let _left_wall = r.i32("core left_wall")?;
    let _move_restrictions = r.i32("core move_restrictions")?;
    let _solo = r.i32("core solo")?;
    let _collision_disabled = r.i32("core collision_disabled")?;
    let _endless_hook = r.i32("core endless_hook")?;
    let _hook_hit_disabled = r.i32("core hook_hit_disabled")?;

    // DDRaceStateFields (54 fields, table order — see this module's doc comment).
    let ddrace_alive = r.i32("ddrace alive")?;
    let ddrace_died_this_tick = r.i32("ddrace died_this_tick")?;
    let ddrace_respawned_this_tick = r.i32("ddrace respawned_this_tick")?;
    let _freeze_time = r.i32("ddrace freeze_time")?;
    let ddrace_is_in_freeze = r.i32("ddrace is_in_freeze")?;
    let ddrace_deep_frozen = r.i32("ddrace deep_frozen")?;
    let ddrace_live_frozen = r.i32("ddrace live_frozen")?;
    let _frozen_last_tick = r.i32("ddrace frozen_last_tick")?;
    let _reload_timer = r.i32("ddrace reload_timer")?;
    let _attack_tick = r.i32("ddrace attack_tick")?;
    let _queued_weapon = r.i32("ddrace queued_weapon")?;
    let _last_weapon = r.i32("ddrace last_weapon")?;
    let _weapon_got_mask = r.i32("ddrace weapon_got_mask")?;
    for _ in 0..6 {
        let _ = r.i32("ddrace weapon_ammo")?;
    }
    for _ in 0..6 {
        let _ = r.i32("ddrace weapon_ammo_regen_start")?;
    }
    let _ninja_activation_tick = r.i32("ddrace ninja_activation_tick")?;
    let _ninja_current_move_time = r.i32("ddrace ninja_current_move_time")?;
    let _ninja_old_vel_amount = r.i32("ddrace ninja_old_vel_amount")?;
    let _ninja_activation_dir_x = r.f32("ddrace ninja_activation_dir_x")?;
    let _ninja_activation_dir_y = r.f32("ddrace ninja_activation_dir_y")?;
    let _tele_checkpoint = r.i32("ddrace tele_checkpoint")?;
    let _endless_jump = r.i32("ddrace endless_jump")?;
    let _jetpack = r.i32("ddrace jetpack")?;
    let _super = r.i32("ddrace super")?;
    let _invincible = r.i32("ddrace invincible")?;
    let _hammer_hit_disabled = r.i32("ddrace hammer_hit_disabled")?;
    let _grenade_hit_disabled = r.i32("ddrace grenade_hit_disabled")?;
    let _laser_hit_disabled = r.i32("ddrace laser_hit_disabled")?;
    let _shotgun_hit_disabled = r.i32("ddrace shotgun_hit_disabled")?;
    let _has_telegun_gun = r.i32("ddrace has_telegun_gun")?;
    let _has_telegun_grenade = r.i32("ddrace has_telegun_grenade")?;
    let _has_telegun_laser = r.i32("ddrace has_telegun_laser")?;
    let ddrace_team = r.i32("ddrace team")?;
    let _strong_weak_id = r.i32("ddrace strong_weak_id")?;
    let _freeze_start = r.i32("ddrace freeze_start")?;
    let _freeze_end = r.i32("ddrace freeze_end")?;
    let _tune_zone = r.i32("ddrace tune_zone")?;
    let _num_inputs = r.i32("ddrace num_inputs")?;
    let _last_refill_jumps = r.i32("ddrace last_refill_jumps")?;
    let _ddrace_state = r.i32("ddrace ddrace_state")?;
    let _start_time = r.i32("ddrace start_time")?;
    let _die_tick = r.i32("ddrace die_tick")?;
    let _spawning = r.i32("ddrace spawning")?;
    let _previous_die_tick = r.i32("ddrace previous_die_tick")?;

    r.expect_eof()?;

    Ok(TraceBCharacterRow {
        input_target_x,
        input_target_y,
        core_pos_x,
        core_pos_y,
        core_hook_pos_x,
        core_hook_pos_y,
        core_hook_state,
        core_hooked_player,
        core_active_weapon,
        core_triggered_events,
        ddrace_alive,
        ddrace_died_this_tick,
        ddrace_respawned_this_tick,
        ddrace_is_in_freeze,
        ddrace_deep_frozen,
        ddrace_live_frozen,
        ddrace_team,
    })
}

/// A minimal, hand-built trace-b v2 writer for tests — an independent construction from
/// `next_tick`'s own reading (same discipline `ddai-map`'s `testutil` module documents for the
/// same reason: a bug in one direction is unlikely to be mirrored, and equally wrong, in the
/// other). Exposed beyond this crate's own unit tests (`test-util` feature, same convention
/// `ddai-map`/`ddai-web`'s own `[dev-dependencies]` self-reference already use) so integration
/// tests in `tests/*.rs` — which need a real trace-b file on disk to exercise the WS `map`/`live`
/// messages against a running server — don't have to duplicate this byte-layout knowledge.
#[cfg(any(test, feature = "test-util"))]
pub mod testutil {
    use super::{COREEVENT_HOOK_ATTACH_PLAYER, MAGIC};
    use std::io::Write;
    use std::path::Path;

    /// Writes a trace-b v2 file at `path`: `character_count` characters, `tick_count` ticks, no
    /// switches, no entities. Character 0 is frozen on every tick (`is_in_freeze = 1`); every
    /// character's `core_pos_x`/`core_pos_y` advances linearly with `tick` so a reader can tell
    /// ticks apart. `real_map_path` is only written into the metadata JSON when `mode ==
    /// "real-map"` (matching real trace-b files, see `docs/formats.md` §11.1).
    pub fn write_trace_b_fixture(
        path: &Path,
        character_count: u32,
        tick_count: u32,
        mode: &str,
        map_sha256: [u8; 32],
        real_map_path: Option<&str>,
    ) {
        let sha256_hex: String = map_sha256.iter().map(|b| format!("{b:02x}")).collect();
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&2u32.to_le_bytes());
        let metadata = if mode == "real-map" {
            format!(
                r#"{{"map_sha256":"{sha256_hex}","mode":"real-map","real_map_path":"{}"}}"#,
                real_map_path.unwrap_or("/tmp/scratch/map.map")
            )
        } else {
            format!(r#"{{"map_sha256":"{sha256_hex}","mode":"rawmap-scenario"}}"#)
        };
        buf.extend_from_slice(&(metadata.len() as u32).to_le_bytes());
        buf.extend_from_slice(metadata.as_bytes());
        buf.extend_from_slice(&character_count.to_le_bytes());
        for id in 0..character_count {
            buf.extend_from_slice(&id.to_le_bytes());
        }
        buf.extend_from_slice(&0u32.to_le_bytes()); // switch_highest_number
        buf.extend_from_slice(&0u32.to_le_bytes()); // switch_team_count (0 team ids follow)
        buf.extend_from_slice(&tick_count.to_le_bytes());

        for tick in 0..tick_count {
            buf.extend_from_slice(&(tick as i32).to_le_bytes()); // game_tick
            // no switches (team_count = 0)
            buf.extend_from_slice(&0u32.to_le_bytes()); // entity_count = 0
            for slot in 0..character_count {
                write_character_row(&mut buf, tick, slot);
            }
        }

        let mut file = std::fs::File::create(path).expect("create trace-b fixture");
        file.write_all(&buf).expect("write trace-b fixture");
    }

    fn write_character_row(buf: &mut Vec<u8>, tick: u32, slot: u32) {
        // PlayerInput (11 i32)
        buf.extend_from_slice(&0i32.to_le_bytes()); // direction
        buf.extend_from_slice(&(100 + slot as i32).to_le_bytes()); // target_x
        buf.extend_from_slice(&(200 + slot as i32).to_le_bytes()); // target_y
        for _ in 0..8 {
            buf.extend_from_slice(&0i32.to_le_bytes()); // jump,fire,hook,flags,wanted,next,prev,kill
        }
        // CoreStateFields: 10 f32
        buf.extend_from_slice(&(tick as f32 * 10.0).to_le_bytes()); // pos_x
        buf.extend_from_slice(&(tick as f32 * 20.0).to_le_bytes()); // pos_y
        for _ in 0..2 {
            buf.extend_from_slice(&0f32.to_le_bytes()); // vel_x, vel_y
        }
        buf.extend_from_slice(&500f32.to_le_bytes()); // hook_pos_x
        buf.extend_from_slice(&600f32.to_le_bytes()); // hook_pos_y
        for _ in 0..4 {
            buf.extend_from_slice(&0f32.to_le_bytes()); // hook_dir x/y, hook_tele_base x/y
        }
        // 18 i32
        buf.extend_from_slice(&0i32.to_le_bytes()); // hook_tick
        buf.extend_from_slice(&5i32.to_le_bytes()); // hook_state = HOOK_GRABBED
        buf.extend_from_slice(&((slot as i32 + 1) % 2).to_le_bytes()); // hooked_player
        buf.extend_from_slice(&1i32.to_le_bytes()); // active_weapon
        for _ in 0..5 {
            buf.extend_from_slice(&0i32.to_le_bytes()); // new_hook,jumped,jumped_total,jumps,direction
        }
        buf.extend_from_slice(&0i32.to_le_bytes()); // angle
        // triggered_events: the HOOK_ATTACH_PLAYER bit means "the hook attached THIS tick" (an
        // edge), not "is currently attached" (that's `hook_state`/`hooked_player`, set above,
        // constant across every tick here — a long-held hook is realistic on its own). Setting
        // this edge bit on every tick for every character (an earlier version of this fixture
        // did) would make every consumer of trace-b data see an unrealistic, constant stream of
        // "just grabbed" events no real trace ever produces (confirmed against the real corpus,
        // see `tests/replay_real_corpus.rs`: 86 hook grabs across 9000 character-ticks, ~1%, not
        // 100%) — restricted to a single, one-time synthetic edge (tick 0, character 0) instead.
        let triggered_events = if tick == 0 && slot == 0 {
            COREEVENT_HOOK_ATTACH_PLAYER
        } else {
            0
        };
        buf.extend_from_slice(&triggered_events.to_le_bytes());
        for _ in 0..7 {
            // colliding, left_wall, move_restrictions, solo, collision_disabled, endless_hook,
            // hook_hit_disabled
            buf.extend_from_slice(&0i32.to_le_bytes());
        }
        // DDRaceStateFields (54 fields)
        buf.extend_from_slice(&1i32.to_le_bytes()); // alive
        buf.extend_from_slice(&0i32.to_le_bytes()); // died_this_tick
        buf.extend_from_slice(&0i32.to_le_bytes()); // respawned_this_tick
        buf.extend_from_slice(&0i32.to_le_bytes()); // freeze_time
        buf.extend_from_slice(&(if slot == 0 { 1i32 } else { 0i32 }).to_le_bytes()); // is_in_freeze
        buf.extend_from_slice(&0i32.to_le_bytes()); // deep_frozen
        buf.extend_from_slice(&0i32.to_le_bytes()); // live_frozen
        for _ in 0..2 {
            buf.extend_from_slice(&0i32.to_le_bytes()); // frozen_last_tick, reload_timer
        }
        buf.extend_from_slice(&0i32.to_le_bytes()); // attack_tick
        buf.extend_from_slice(&(-1i32).to_le_bytes()); // queued_weapon
        buf.extend_from_slice(&0i32.to_le_bytes()); // last_weapon
        buf.extend_from_slice(&0i32.to_le_bytes()); // weapon_got_mask
        for _ in 0..12 {
            buf.extend_from_slice(&0i32.to_le_bytes()); // weapon_ammo x6, weapon_ammo_regen_start x6
        }
        for _ in 0..3 {
            buf.extend_from_slice(&0i32.to_le_bytes()); // ninja_activation_tick/current_move_time/old_vel_amount
        }
        buf.extend_from_slice(&0f32.to_le_bytes()); // ninja_activation_dir_x
        buf.extend_from_slice(&0f32.to_le_bytes()); // ninja_activation_dir_y
        buf.extend_from_slice(&0i32.to_le_bytes()); // tele_checkpoint
        for _ in 0..4 {
            buf.extend_from_slice(&0i32.to_le_bytes()); // endless_jump, jetpack, super, invincible
        }
        for _ in 0..4 {
            buf.extend_from_slice(&0i32.to_le_bytes()); // 4 hit_disabled
        }
        for _ in 0..3 {
            buf.extend_from_slice(&0i32.to_le_bytes()); // 3 has_telegun
        }
        buf.extend_from_slice(&3i32.to_le_bytes()); // team
        buf.extend_from_slice(&0i32.to_le_bytes()); // strong_weak_id
        buf.extend_from_slice(&0i32.to_le_bytes()); // freeze_start
        buf.extend_from_slice(&0i32.to_le_bytes()); // freeze_end
        buf.extend_from_slice(&0i32.to_le_bytes()); // tune_zone
        buf.extend_from_slice(&0i32.to_le_bytes()); // num_inputs
        buf.extend_from_slice(&0i32.to_le_bytes()); // last_refill_jumps
        buf.extend_from_slice(&0i32.to_le_bytes()); // ddrace_state
        buf.extend_from_slice(&0i32.to_le_bytes()); // start_time
        buf.extend_from_slice(&0i32.to_le_bytes()); // die_tick
        buf.extend_from_slice(&0i32.to_le_bytes()); // spawning
        buf.extend_from_slice(&0i32.to_le_bytes()); // previous_die_tick
    }
} // mod testutil

#[cfg(test)]
mod tests {
    use super::*;

    /// This crate's own unit tests use fixed sha256/path values throughout — a thin wrapper over
    /// [`testutil::write_trace_b_fixture`] so each test below doesn't repeat them.
    fn write_trace_b(path: &Path, character_count: u32, tick_count: u32, mode: &str) {
        let sha256 = if mode == "real-map" { [0xab; 32] } else { [0xcd; 32] };
        let real_map_path = (mode == "real-map").then_some("/tmp/scratch/BlmapChill.map");
        testutil::write_trace_b_fixture(path, character_count, tick_count, mode, sha256, real_map_path);
    }

    #[test]
    fn reads_header_metadata_and_character_ids() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("t.trb");
        write_trace_b(&path, 3, 2, "real-map");

        let reader = TraceBReader::open(&path).expect("open");
        let header = reader.header();
        assert_eq!(header.character_ids, vec![0, 1, 2]);
        assert_eq!(header.tick_count, 2);
        assert_eq!(header.metadata.mode, "real-map");
        assert_eq!(
            header.metadata.real_map_path.as_deref(),
            Some("/tmp/scratch/BlmapChill.map")
        );
        assert_eq!(header.metadata.map_sha256, [0xab; 32]);
    }

    #[test]
    fn reads_every_tick_and_stops_cleanly_at_the_end() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("t.trb");
        write_trace_b(&path, 2, 3, "rawmap-scenario");

        let mut reader = TraceBReader::open(&path).expect("open");
        let mut ticks = Vec::new();
        while let Some(tick) = reader.next_tick().expect("next_tick") {
            ticks.push(tick);
        }
        assert_eq!(ticks.len(), 3);
        assert!(reader.next_tick().expect("next_tick after end").is_none());

        // Spot-check the decoded fields for the last tick.
        let last = ticks.last().unwrap();
        assert_eq!(last.game_tick, 2);
        assert_eq!(last.characters.len(), 2);
        assert_eq!(last.characters[0].core_pos_x, 2.0 * 10.0);
        assert_eq!(last.characters[1].input_target_x, 101);
        assert_eq!(last.characters[0].ddrace_is_in_freeze, 1);
        assert_eq!(last.characters[1].ddrace_is_in_freeze, 0);
        assert_eq!(last.characters[0].core_hook_state, 5);
        assert_eq!(last.characters[0].core_hooked_player, 1);
        assert_eq!(last.characters[0].ddrace_team, 3);

        // The one-time synthetic hook-attach edge (see `write_character_row`'s doc comment) is on
        // tick 0, character 0 only — not on every tick, unlike `hook_state`/`hooked_player` above.
        assert_eq!(
            ticks[0].characters[0].core_triggered_events & COREEVENT_HOOK_ATTACH_PLAYER,
            COREEVENT_HOOK_ATTACH_PLAYER
        );
        assert_eq!(
            last.characters[0].core_triggered_events & COREEVENT_HOOK_ATTACH_PLAYER,
            0
        );
    }

    #[test]
    fn rejects_bad_magic() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("t.trb");
        std::fs::write(&path, b"XXXX\x02\x00\x00\x00").expect("write");
        let err = TraceBReader::open(&path).expect_err("should reject bad magic");
        assert!(matches!(err, TraceBError::Format(FormatError::BadMagic { .. })));
    }

    #[test]
    fn rejects_unsupported_version() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("t.trb");
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&1u32.to_le_bytes()); // v1, not supported
        std::fs::write(&path, &buf).expect("write");
        let err = TraceBReader::open(&path).expect_err("should reject v1");
        assert!(matches!(err, TraceBError::UnsupportedVersion(1)));
    }

    #[test]
    fn rejects_a_character_count_above_max_clients() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("t.trb");
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&2u32.to_le_bytes());
        let metadata = format!(r#"{{"map_sha256":"{}","mode":"rawmap-scenario"}}"#, "00".repeat(32));
        buf.extend_from_slice(&(metadata.len() as u32).to_le_bytes());
        buf.extend_from_slice(metadata.as_bytes());
        buf.extend_from_slice(&999u32.to_le_bytes()); // character_count, no ids follow
        std::fs::write(&path, &buf).expect("write");
        let err = TraceBReader::open(&path).expect_err("should reject");
        assert!(matches!(err, TraceBError::TooManyCharacters(999)));
    }

    #[test]
    fn a_truncated_file_gives_an_error_not_a_panic() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("t.trb");
        write_trace_b(&path, 2, 5, "real-map");
        let full = std::fs::read(&path).expect("read");
        // Cut it off partway through the third tick's data.
        let truncated = &full[..full.len() - 50];
        std::fs::write(&path, truncated).expect("write truncated");

        let mut reader = TraceBReader::open(&path).expect("header should still parse");
        let mut saw_error = false;
        loop {
            match reader.next_tick() {
                Ok(Some(_)) => continue,
                Ok(None) => break,
                Err(_) => {
                    saw_error = true;
                    break;
                }
            }
        }
        assert!(
            saw_error,
            "a truncated tick must be reported as an error, not silently stop"
        );
    }

    #[test]
    fn invalid_metadata_json_is_an_error_not_a_panic() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("t.trb");
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&2u32.to_le_bytes());
        let metadata = b"not json at all";
        buf.extend_from_slice(&(metadata.len() as u32).to_le_bytes());
        buf.extend_from_slice(metadata);
        std::fs::write(&path, &buf).expect("write");
        let err = TraceBReader::open(&path).expect_err("should reject invalid JSON");
        assert!(matches!(err, TraceBError::InvalidMetadataJson(_)));
    }

    /// Regression test for review round 1, finding F1: a `map_sha256` field containing a
    /// multi-byte UTF-8 character used to panic (`decode_hex_sha256` sliced the `&str` by byte
    /// range, which panics on a non-char-boundary index) even though `hex.len()` — a *byte*
    /// count — still equalled 64. Exact repro from the finding: `"a" + "\u{e9}" (2 bytes) +
    /// "0"*61` = 1 + 2 + 61 = 64 bytes.
    #[test]
    fn a_multi_byte_utf8_map_sha256_is_an_error_not_a_panic() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("t.trb");
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&2u32.to_le_bytes());
        let bad_sha256 = format!("a\u{e9}{}", "0".repeat(61));
        assert_eq!(bad_sha256.len(), 64, "byte length must still pass the length check");
        let metadata = format!(r#"{{"map_sha256":"{bad_sha256}","mode":"rawmap-scenario"}}"#);
        buf.extend_from_slice(&(metadata.len() as u32).to_le_bytes());
        buf.extend_from_slice(metadata.as_bytes());
        std::fs::write(&path, &buf).expect("write");

        // Must not panic — `TraceBReader::open` reports a normal `Err` instead.
        let err = TraceBReader::open(&path).expect_err("should reject, not panic");
        assert!(matches!(err, TraceBError::InvalidMapSha256), "{err:?}");
    }

    #[test]
    fn decode_hex_sha256_rejects_non_hex_ascii_without_panicking() {
        // A plain non-hex ASCII byte (no multi-byte UTF-8 involved) must also be rejected, not
        // just tolerated by accident — covers the `hex_val` match's `_` arm directly.
        assert!(matches!(
            decode_hex_sha256(&"g".repeat(64)),
            Err(TraceBError::InvalidMapSha256)
        ));
    }
}
