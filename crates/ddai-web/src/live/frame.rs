//! The binary `live` WS frame format (acceptance criterion 1: "Binary frame format versioned and
//! documented in docs/formats.md"). See `docs/formats.md`'s new section for the byte-for-byte
//! layout this module implements; this file is the single source of truth the doc transcribes.
//!
//! Design notes (why this isn't a 1:1 copy of `docs/research/orig-web.md` §7.3's sketch, which
//! the task spec explicitly only asks to follow "roughly"):
//! - **Aim** is sent as the applied input's raw `target_x`/`target_y` (clamped to `i16`), not a
//!   reinterpretation of `CCharacterCore::m_Angle`'s internal fixed-point scale. `target_x/y` is
//!   DDNet's own wire representation of "where the cursor is, relative to the tee" — it needs no
//!   DDNet-internals knowledge to decode correctly, and Oracle B's trace-b format already records
//!   it as the *applied* per-tick input (see `crate::live::replay::trace_b`).
//! - **Direction/jumped/emote** are omitted (a v1 scope cut, not an oversight) — the acceptance
//!   criteria's own minimum ("position, aim, hook line, hook state, freeze state and name") does
//!   not need them for a first correct render; a v2 frame version can add them as trailing fields
//!   without breaking v1 decoders (this format is deliberately front-loaded: version first, so an
//!   unrecognized version is rejected before anything else is parsed).

pub const MAGIC: &[u8; 4] = b"DWLF"; // "DDai Web Live Frame"
pub const VERSION: u8 = 1;

/// Fixed size of one [`crate::live::source::CharacterState`]'s wire record, in bytes. See this
/// module's `encode_character`/`decode_character` for the exact field layout.
pub const CHARACTER_RECORD_BYTES: usize = 26;

/// `CCharacterCore::m_HookState`'s `HOOK_FLYING`/`HOOK_GRABBED` values (`gamecore.h`) — see
/// `flags` bit 5 below.
const HOOK_FLYING: i8 = 4;
const HOOK_GRABBED: i8 = 5;

mod flag_bits {
    pub const ALIVE: u8 = 1 << 0;
    pub const FROZEN: u8 = 1 << 1;
    pub const DEEP_FROZEN: u8 = 1 << 2;
    pub const LIVE_FROZEN: u8 = 1 << 3;
    pub const HOOK_VISIBLE: u8 = 1 << 4;
}

use super::source::{CharacterState, WorldFrame};

/// Errors [`decode`] reports — always a data problem, never a panic (the same bounded-parsing
/// discipline `ddai-trace`/`ddai-map` already follow, extended here to our own new format).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DecodeError {
    #[error("payload shorter than the {0}-byte header")]
    Truncated(usize),
    #[error("bad magic: expected {DWLF:?}", DWLF = MAGIC)]
    BadMagic,
    #[error("unsupported live frame version {0}")]
    UnsupportedVersion(u8),
    #[error("declared character_count ({0}) exceeds the {1} MAX_CLIENTS cap")]
    TooManyCharacters(u16, u16),
    #[error("payload has {extra} trailing byte(s) past the last character record")]
    TrailingBytes { extra: usize },
}

/// DDNet's `MAX_CLIENTS` (`engine/shared/protocol.h`) — the same bound `ddai-trace::scenario`
/// enforces on a character id; [`decode`] rejects a `character_count` above it up front, before
/// even trying to read that many 26-byte records, so a corrupt/adversarial `character_count`
/// (read from an otherwise-untrusted byte stream) can never make this allocate more than a
/// `MAX_CLIENTS`-sized `Vec` no matter what it claims.
const MAX_CLIENTS: u16 = 128;

const HEADER_BYTES: usize = 4 + 1 + 1 + 4 + 2; // magic + version + reserved + tick + char_count

/// Encodes `frame` as a `live` WS binary message (acceptance criterion 1).
///
/// Panics only if `frame.characters.len()` exceeds `u16::MAX` (far beyond `MAX_CLIENTS`, so this
/// can only happen if a caller builds a `WorldFrame` by hand with a nonsensical length — every
/// real `FrameSource` in this crate caps character counts at `MAX_CLIENTS` well before this
/// point) — matches this crate's convention of reserving `Result` for *data* problems (a caller's
/// own malformed value is a programming error, not a runtime data error).
pub fn encode(frame: &WorldFrame) -> Vec<u8> {
    let char_count: u16 = frame
        .characters
        .len()
        .try_into()
        .expect("WorldFrame.characters.len() must fit in u16");
    let mut out = Vec::with_capacity(HEADER_BYTES + frame.characters.len() * CHARACTER_RECORD_BYTES);
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    out.push(0); // reserved, always 0
    out.extend_from_slice(&frame.tick.to_le_bytes());
    out.extend_from_slice(&char_count.to_le_bytes());
    for c in &frame.characters {
        encode_character(&mut out, c);
    }
    out
}

fn encode_character(out: &mut Vec<u8>, c: &CharacterState) {
    let mut flags = 0u8;
    if c.alive {
        flags |= flag_bits::ALIVE;
    }
    if c.frozen {
        flags |= flag_bits::FROZEN;
    }
    if c.deep_frozen {
        flags |= flag_bits::DEEP_FROZEN;
    }
    if c.live_frozen {
        flags |= flag_bits::LIVE_FROZEN;
    }
    if c.hook_state == HOOK_FLYING || c.hook_state == HOOK_GRABBED {
        flags |= flag_bits::HOOK_VISIBLE;
    }

    out.push(c.id);
    out.push(flags);
    out.push(c.team);
    out.push(c.weapon);
    out.extend_from_slice(&c.x.to_le_bytes());
    out.extend_from_slice(&c.y.to_le_bytes());
    out.extend_from_slice(&clamp_i16(c.aim_x).to_le_bytes());
    out.extend_from_slice(&clamp_i16(c.aim_y).to_le_bytes());
    out.extend_from_slice(&c.hook_x.to_le_bytes());
    out.extend_from_slice(&c.hook_y.to_le_bytes());
    out.push(c.hooked_id.map(|id| id as i8 as u8).unwrap_or(0xFF)); // 0xFF = -1 = none
    out.push(0); // reserved, always 0
}

fn clamp_i16(v: i32) -> i16 {
    v.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

/// Decodes a `live` WS binary message back into a [`WorldFrame`] (acceptance criterion 5: "frame
/// encode/decode round trip"). `hook_state`/`frozen`/etc. are reconstructed from the encoded
/// flags/values, not byte-identical to whatever produced the original frame (this format is
/// intentionally lossy about a few things — see this module's doc comment) — round-trip tests
/// compare against what `encode` itself would produce for the decoded value, not against an
/// arbitrary original [`CharacterState`].
pub fn decode(bytes: &[u8]) -> Result<WorldFrame, DecodeError> {
    if bytes.len() < HEADER_BYTES {
        return Err(DecodeError::Truncated(HEADER_BYTES));
    }
    if &bytes[0..4] != MAGIC {
        return Err(DecodeError::BadMagic);
    }
    let version = bytes[4];
    if version != VERSION {
        return Err(DecodeError::UnsupportedVersion(version));
    }
    // bytes[5] is reserved, ignored.
    let tick = u32::from_le_bytes(bytes[6..10].try_into().unwrap());
    let char_count = u16::from_le_bytes(bytes[10..12].try_into().unwrap());
    if char_count > MAX_CLIENTS {
        return Err(DecodeError::TooManyCharacters(char_count, MAX_CLIENTS));
    }

    let mut characters = Vec::with_capacity(char_count as usize);
    let mut offset = HEADER_BYTES;
    for _ in 0..char_count {
        let record = bytes
            .get(offset..offset + CHARACTER_RECORD_BYTES)
            .ok_or(DecodeError::Truncated(offset + CHARACTER_RECORD_BYTES))?;
        characters.push(decode_character(record));
        offset += CHARACTER_RECORD_BYTES;
    }
    if offset != bytes.len() {
        return Err(DecodeError::TrailingBytes {
            extra: bytes.len() - offset,
        });
    }
    Ok(WorldFrame { tick, characters })
}

fn decode_character(record: &[u8]) -> CharacterState {
    let flags = record[1];
    let hooked_raw = record[24] as i8;
    CharacterState {
        id: record[0],
        alive: flags & flag_bits::ALIVE != 0,
        team: record[2],
        weapon: record[3],
        x: i32::from_le_bytes(record[4..8].try_into().unwrap()),
        y: i32::from_le_bytes(record[8..12].try_into().unwrap()),
        aim_x: i16::from_le_bytes(record[12..14].try_into().unwrap()) as i32,
        aim_y: i16::from_le_bytes(record[14..16].try_into().unwrap()) as i32,
        hook_x: i32::from_le_bytes(record[16..20].try_into().unwrap()),
        hook_y: i32::from_le_bytes(record[20..24].try_into().unwrap()),
        hooked_id: if hooked_raw < 0 { None } else { Some(hooked_raw as u8) },
        // The wire format only round-trips *whether* the hook is visible (flag bit), not the
        // precise `HOOK_STATE` enum value — `HOOK_FLYING` is as good a representative as any
        // "visible" state to reconstruct, since nothing downstream distinguishes flying/grabbed
        // once decoded other than `hooked_id` (grabbed-a-player vs grabbed-a-wall), which is
        // carried separately and correctly either way.
        hook_state: if flags & flag_bits::HOOK_VISIBLE != 0 {
            HOOK_FLYING
        } else {
            0
        },
        frozen: flags & flag_bits::FROZEN != 0,
        deep_frozen: flags & flag_bits::DEEP_FROZEN != 0,
        live_frozen: flags & flag_bits::LIVE_FROZEN != 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_character(id: u8) -> CharacterState {
        CharacterState {
            id,
            alive: true,
            x: 12345,
            y: -6789,
            aim_x: 1000,
            aim_y: -500,
            hook_state: HOOK_GRABBED,
            hook_x: 100,
            hook_y: 200,
            hooked_id: Some(7),
            weapon: 3,
            team: 2,
            frozen: true,
            deep_frozen: false,
            live_frozen: false,
        }
    }

    #[test]
    fn character_record_size_matches_the_constant() {
        let mut out = Vec::new();
        encode_character(&mut out, &sample_character(0));
        assert_eq!(out.len(), CHARACTER_RECORD_BYTES);
    }

    #[test]
    fn round_trip_preserves_every_field_a_v1_client_can_render() {
        let frame = WorldFrame {
            tick: 42,
            characters: vec![sample_character(0), sample_character(1)],
        };
        let bytes = encode(&frame);
        let decoded = decode(&bytes).expect("decode");
        assert_eq!(decoded.tick, 42);
        assert_eq!(decoded.characters.len(), 2);
        for (original, back) in frame.characters.iter().zip(&decoded.characters) {
            assert_eq!(back.id, original.id);
            assert_eq!(back.alive, original.alive);
            assert_eq!(back.x, original.x);
            assert_eq!(back.y, original.y);
            assert_eq!(back.aim_x, original.aim_x);
            assert_eq!(back.aim_y, original.aim_y);
            assert_eq!(back.hook_x, original.hook_x);
            assert_eq!(back.hook_y, original.hook_y);
            assert_eq!(back.hooked_id, original.hooked_id);
            assert_eq!(back.weapon, original.weapon);
            assert_eq!(back.team, original.team);
            assert_eq!(back.frozen, original.frozen);
            assert_eq!(back.deep_frozen, original.deep_frozen);
            assert_eq!(back.live_frozen, original.live_frozen);
        }
    }

    #[test]
    fn round_trip_with_zero_characters() {
        let frame = WorldFrame {
            tick: 7,
            characters: vec![],
        };
        let decoded = decode(&encode(&frame)).expect("decode");
        assert_eq!(decoded, frame);
    }

    #[test]
    fn no_hooked_id_round_trips_as_none() {
        let mut c = sample_character(3);
        c.hooked_id = None;
        c.hook_state = 0;
        let frame = WorldFrame {
            tick: 1,
            characters: vec![c],
        };
        let decoded = decode(&encode(&frame)).expect("decode");
        assert_eq!(decoded.characters[0].hooked_id, None);
    }

    #[test]
    fn aim_is_clamped_to_i16_range_not_silently_wrapped() {
        let mut c = sample_character(0);
        c.aim_x = 100_000; // way past i16::MAX
        c.aim_y = -100_000;
        let frame = WorldFrame {
            tick: 0,
            characters: vec![c],
        };
        let decoded = decode(&encode(&frame)).expect("decode");
        assert_eq!(decoded.characters[0].aim_x, i16::MAX as i32);
        assert_eq!(decoded.characters[0].aim_y, i16::MIN as i32);
    }

    #[test]
    fn hook_visible_flag_reflects_flying_or_grabbed_only() {
        for (state, visible) in [
            (-1i8, false),
            (0, false),
            (1, false),
            (2, false),
            (3, false),
            (4, true),
            (5, true),
        ] {
            let mut c = sample_character(0);
            c.hook_state = state;
            let frame = WorldFrame {
                tick: 0,
                characters: vec![c],
            };
            let bytes = encode(&frame);
            let flags = bytes[HEADER_BYTES + 1];
            assert_eq!(
                flags & flag_bits::HOOK_VISIBLE != 0,
                visible,
                "hook_state {state} should have hook-visible = {visible}"
            );
        }
    }

    #[test]
    fn decode_rejects_bad_magic() {
        let mut bytes = encode(&WorldFrame {
            tick: 0,
            characters: vec![],
        });
        bytes[0] = b'X';
        assert_eq!(decode(&bytes), Err(DecodeError::BadMagic));
    }

    #[test]
    fn decode_rejects_unsupported_version() {
        let mut bytes = encode(&WorldFrame {
            tick: 0,
            characters: vec![],
        });
        bytes[4] = 99;
        assert_eq!(decode(&bytes), Err(DecodeError::UnsupportedVersion(99)));
    }

    #[test]
    fn decode_rejects_truncated_header() {
        assert_eq!(decode(&[1, 2, 3]), Err(DecodeError::Truncated(HEADER_BYTES)));
    }

    #[test]
    fn decode_rejects_truncated_character_record() {
        let mut bytes = encode(&WorldFrame {
            tick: 0,
            characters: vec![sample_character(0)],
        });
        bytes.truncate(bytes.len() - 1); // one byte short of the last record
        assert!(matches!(decode(&bytes), Err(DecodeError::Truncated(_))));
    }

    #[test]
    fn decode_rejects_trailing_bytes() {
        let mut bytes = encode(&WorldFrame {
            tick: 0,
            characters: vec![],
        });
        bytes.push(0xAA);
        assert_eq!(decode(&bytes), Err(DecodeError::TrailingBytes { extra: 1 }));
    }

    #[test]
    fn decode_rejects_a_character_count_above_max_clients() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.push(VERSION);
        bytes.push(0);
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&200u16.to_le_bytes()); // > MAX_CLIENTS, and no records follow
        assert_eq!(decode(&bytes), Err(DecodeError::TooManyCharacters(200, MAX_CLIENTS)));
    }

    /// Golden byte fixture (acceptance criterion 5): a hand-computed, hard-coded expected byte
    /// sequence for one known `WorldFrame`, so an accidental future change to the wire layout
    /// (e.g. a field reordering) is caught even if it happens to keep passing every *round-trip*
    /// test (which would happily "pass" against a self-consistent but now-different format).
    #[test]
    fn golden_fixture_matches_the_documented_byte_layout() {
        let frame = WorldFrame {
            tick: 0x0102_0304,
            characters: vec![CharacterState {
                id: 5,
                alive: true,
                x: 100,
                y: -200,
                aim_x: 300,
                aim_y: -400,
                hook_state: HOOK_GRABBED,
                hook_x: 500,
                hook_y: 600,
                hooked_id: Some(9),
                weapon: 1,
                team: 0,
                frozen: true,
                deep_frozen: true,
                live_frozen: false,
            }],
        };
        let bytes = encode(&frame);
        #[rustfmt::skip]
        let expected: Vec<u8> = vec![
            // magic
            b'D', b'W', b'L', b'F',
            // version, reserved
            1, 0,
            // tick = 0x0102_0304, little-endian
            0x04, 0x03, 0x02, 0x01,
            // char_count = 1, little-endian
            0x01, 0x00,
            // --- character record ---
            5,                                   // id
            0b0001_0111,                         // flags: ALIVE|FROZEN|DEEP_FROZEN|HOOK_VISIBLE
            0,                                   // team
            1,                                   // weapon
            0x64, 0x00, 0x00, 0x00,               // x = 100 (i32 LE)
            0x38, 0xff, 0xff, 0xff,               // y = -200 (i32 LE)
            0x2c, 0x01,                           // aim_x = 300 (i16 LE)
            0x70, 0xfe,                           // aim_y = -400 (i16 LE)
            0xf4, 0x01, 0x00, 0x00,               // hook_x = 500
            0x58, 0x02, 0x00, 0x00,               // hook_y = 600
            9,                                    // hooked_id = 9
            0,                                    // reserved
        ];
        assert_eq!(
            bytes, expected,
            "byte layout changed — update docs/formats.md too if intentional"
        );
    }
}
