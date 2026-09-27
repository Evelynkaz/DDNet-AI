// Ported from DDNet `src/engine/shared/{uuid_manager,protocol_ex,protocol_ex_msgs}.{h,cpp}`
// (pinned rev c9d208138f85755521f16a0096b6fe036c5c8698, "20.1"). Unlike `huffman.cpp`/
// `network.cpp`, these particular files (DDNet-specific additions, not part of the original
// Teeworlds engine core) carry no *per-file* copyright header of their own — they fall under the
// repository-wide notice instead (`license.txt` at the repo root):
//
//   Teeworlds Copyright (C) 2007-2014 Magnus Auvinen
//   DDRace    Copyright (C) 2010-2011 Shereef Marzouk
//   DDNet     Copyright (C)           Dennis Felsing
//
//   This software is provided 'as-is', without any express or implied warranty. [...]
//   Permission is granted to anyone to use this software for any purpose, [...] and to alter it
//   and redistribute it freely, subject to the following restrictions: [...]
//   2. Altered source versions must be plainly marked as such, and must not be misrepresented as
//      being the original software.
//
// This is an *altered* source version: rewritten in safe Rust, same UUID v3 (namespace + MD5)
// derivation and `NETMSG_EX` wire encoding, so it stays byte-for-byte compatible with the wire
// format every DDNet 0.6+DDNet client/server speaks. See docs/formats.md for the byte layout.
//
//! DDNet UUID-based extended message ids (`NETMSG_EX`).
//!
//! DDNet identifies "extended" system messages by a stable UUID instead of a small numbered id,
//! so third-party mods and future protocol versions can add messages without colliding with each
//! other. The UUID is a version-3 (namespace + MD5) UUID: `md5(namespace_uuid_bytes ++ name)`
//! with the version/variant bits overwritten, exactly RFC 4122 §4.3 (`uuid_manager.cpp:38-61`).
//!
//! Wire encoding (`client.cpp:179-188`, `protocol_ex.cpp:20-51`): the leading varint of every
//! message is `(msg_id << 1) | sys`. When the decoded `msg_id` is `0`, the message is
//! `NETMSGTYPE_EX`/an extended object: the *next* 16 raw bytes are the UUID, and the message's
//! real identity is whatever [`UuidRegistry::lookup`] resolves that UUID to (or "unknown" if it
//! isn't registered — DDNet ignores unknown extended messages rather than erroring the
//! connection, `protocol_ex.cpp:36-39`).

use crate::packer::{Packer, Unpacker};
use md5::{Digest, Md5};
use std::fmt;

/// A 16-byte UUID, without the string-formatting/parsing frills DDNet's `CUuid` doesn't need
/// either — this is purely a value type for [`UuidRegistry`] and the packer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Uuid(pub [u8; 16]);

/// `UUID_ZEROED` (`uuid_manager.cpp:18-20`).
pub const ZEROED: Uuid = Uuid([0; 16]);

impl fmt::Display for Uuid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let b = &self.0;
        write!(
            f,
            "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]
        )
    }
}

/// DDNet's fixed namespace UUID `e05ddaaa-c4e6-4cfb-b642-5d48e80c0029` (`uuid_manager.cpp:14-16`),
/// used as the RFC 4122 namespace for every extended-message name.
const TEEWORLDS_NAMESPACE: [u8; 16] = [
    0xe0, 0x5d, 0xda, 0xaa, 0xc4, 0xe6, 0x4c, 0xfb, 0xb6, 0x42, 0x5d, 0x48, 0xe8, 0x0c, 0x00, 0x29,
];

/// Computes the version-3 UUID for an extended-message `name`, exactly as `CalculateUuid`
/// (`uuid_manager.cpp:38-62`): `md5(namespace ++ name)`, then set the version nibble (byte 6's
/// high nibble) to `0011` and the variant bits (byte 8's top two bits) to `10`.
pub fn calculate_uuid(name: &str) -> Uuid {
    let mut hasher = Md5::new();
    hasher.update(TEEWORLDS_NAMESPACE);
    hasher.update(name.as_bytes());
    let digest = hasher.finalize();

    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&digest);
    bytes[6] = (bytes[6] & 0x0f) | 0x30; // version 3
    bytes[8] = (bytes[8] & 0x3f) | 0x80; // variant 1 (RFC 4122)
    Uuid(bytes)
}

/// The extended-message names needed so far (task 2.2a's scope), in the exact order DDNet 20.1's
/// `protocol_ex_msgs.h` lists them — order matters: [`UuidRegistry::new`] assigns ids by
/// registration order, exactly like `RegisterUuids`/`CUuidManager::RegisterName`
/// (`uuid_manager.cpp:123-137`), so changing this order would change every id (though not the
/// UUIDs themselves, which only depend on the name).
pub const REGISTERED_NAMES: &[&str] = &[
    "what-is@ddnet.tw",
    "it-is@ddnet.tw",
    "i-dont-know@ddnet.tw",
    "rcon-type@ddnet.tw",
    "map-details@ddnet.tw",
    "capabilities@ddnet.tw",
    "clientver@ddnet.tw",
    "ping@ddnet.tw",
    "pong@ddnet.tw",
    "checksum-request@ddnet.tw",
    "checksum-response@ddnet.tw",
    "checksum-error@ddnet.tw",
    "redirect@ddnet.org",
    "rcon-cmd-group-start@ddnet.org",
    "rcon-cmd-group-end@ddnet.org",
    "map-reload@ddnet.org",
    "reconnect@ddnet.org",
    "sv-maplist-add@ddnet.org",
    "sv-maplist-start@ddnet.org",
    "sv-maplist-end@ddnet.org",
];

/// `OFFSET_UUID` (`uuid_manager.h:13`): the first numbered id used for UUID-backed messages, so
/// that they never collide with the small hand-numbered ids of the ~25 non-extended system
/// messages (`protocol.h`, out of scope here — task 2.2b).
pub const OFFSET_UUID: i32 = 1 << 16;

/// Maps extended-message names to their UUIDs and small integer ids, and back — `CUuidManager`
/// (`uuid_manager.h:52-69`).
///
/// Ids are assigned sequentially starting at [`OFFSET_UUID`] in registration order; looking a
/// UUID up that was never registered returns `None` rather than erroring (DDNet: silently ignore
/// unknown extended messages/objects, since a peer running a newer/different protocol version or
/// a third-party mod is expected to send ids we don't know).
#[derive(Debug, Clone)]
pub struct UuidRegistry {
    /// Index `i` holds the UUID for id `OFFSET_UUID + i`; `names[i]` its human-readable name.
    entries: Vec<(Uuid, &'static str)>,
}

impl Default for UuidRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl UuidRegistry {
    /// Builds the registry from [`REGISTERED_NAMES`] (task 2.2a's system/engine `ex` messages
    /// only), computing each UUID via [`calculate_uuid`]. Panics only on a programmer error (a
    /// duplicate name/UUID in the fixed list above), never on anything runtime/network-controlled
    /// — this never runs on attacker-controlled input.
    pub fn new() -> Self {
        Self::from_names(REGISTERED_NAMES)
    }

    /// Builds a registry from an arbitrary ordered list of names, assigning ids sequentially
    /// starting at [`OFFSET_UUID`] in `names`' order — the general form [`UuidRegistry::new`]
    /// calls with [`REGISTERED_NAMES`]. Task 2.2b uses this to build one combined registry
    /// covering the system/engine `ex` messages *and* the game-level `ex` objects/messages
    /// (`crate::generated::objects::EX_NAMES`, `crate::generated::messages::EX_NAMES`) — see
    /// `crate::message`. Panics only on a programmer error (a duplicate name/UUID in `names`),
    /// never on anything runtime/network-controlled.
    pub fn from_names(names: &[&'static str]) -> Self {
        let mut entries = Vec::with_capacity(names.len());
        for &name in names {
            let uuid = calculate_uuid(name);
            debug_assert!(
                !entries.iter().any(|(u, _)| *u == uuid),
                "duplicate uuid for name {name}"
            );
            entries.push((uuid, name));
        }
        UuidRegistry { entries }
    }

    /// The id assigned to `uuid`, or `None` if it was never registered.
    pub fn lookup(&self, uuid: Uuid) -> Option<i32> {
        self.entries
            .iter()
            .position(|(u, _)| *u == uuid)
            .map(|idx| OFFSET_UUID + idx as i32)
    }

    /// The UUID for a previously registered `id`, or `None` if `id` is out of range.
    pub fn uuid(&self, id: i32) -> Option<Uuid> {
        let idx = usize::try_from(id - OFFSET_UUID).ok()?;
        self.entries.get(idx).map(|(u, _)| *u)
    }

    /// The registered name for `id`, or `None` if `id` is out of range.
    pub fn name(&self, id: i32) -> Option<&'static str> {
        let idx = usize::try_from(id - OFFSET_UUID).ok()?;
        self.entries.get(idx).map(|(_, name)| *name)
    }

    /// Number of registered names.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry has no entries (never true for [`UuidRegistry::new`]).
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Appends `uuid`'s 16 raw bytes to `packer` (`CUuidManager::PackUuid`, `uuid_manager.cpp:184-188`).
pub fn pack_uuid(packer: &mut Packer, uuid: Uuid) {
    packer.add_raw(&uuid.0);
}

/// Reads a 16-byte UUID from `unpacker`, returning `None` (and poisoning the unpacker) if fewer
/// than 16 bytes remain (`CUuidManager::UnpackUuid`, `uuid_manager.cpp:173-182`).
pub fn unpack_uuid(unpacker: &mut Unpacker) -> Option<Uuid> {
    let raw = unpacker.get_raw(16)?;
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(raw);
    Some(Uuid(bytes))
}

/// A message's identity as read off the wire: either a small numbered id (a "normal" 0.6/DDNet
/// system or game message, meaningful to a layer above this one that knows the id table — task
/// 2.2b) or an extended message identified by UUID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsgId {
    /// A non-extended message id (always `< OFFSET_UUID`).
    Numbered(i32),
    /// `NETMSGTYPE_EX`: identified by UUID. `resolved` is the small id [`UuidRegistry::lookup`]
    /// mapped it to, if it was registered — callers that only care about known extended messages
    /// can match on this without re-deriving it.
    Ex { uuid: Uuid, resolved: Option<i32> },
}

/// Why [`unpack_msg_id`] failed. Mirrors the `UNPACKMESSAGE_ERROR` return of `UnpackMessageId`
/// (`protocol_ex.cpp:20-51`) — DDNet folds "no more data" and "id out of the valid numbered
/// range" into the same generic decode failure, dropping the packet either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum UnpackMsgIdError {
    #[error("could not read the leading message-id varint")]
    NoLeadingInt,
    #[error("message id out of the valid range (must be < OFFSET_UUID)")]
    OutOfRange,
    #[error("NETMSGTYPE_EX marker but no 16-byte UUID followed")]
    MissingUuid,
}

/// Reads the leading `(id << 1) | sys` varint and, for `NETMSGTYPE_EX` (decoded id `0`), the
/// following 16-byte UUID — `UnpackMessageId`'s decode half (`protocol_ex.cpp:20-51`), minus the
/// `NETMSG_WHATIS`/`IDONTKNOW`/`ITIS` answer-message construction, which needs a `CMsgPacker`
/// equivalent that belongs to the game-message layer (task 2.2b), not here.
///
/// Returns the message id and whether it is a system (`true`) or game (`false`) message.
pub fn unpack_msg_id(unpacker: &mut Unpacker, registry: &UuidRegistry) -> Result<(MsgId, bool), UnpackMsgIdError> {
    let raw = unpacker.get_int();
    if unpacker.error() {
        return Err(UnpackMsgIdError::NoLeadingInt);
    }
    let id = raw >> 1;
    let sys = raw & 1 != 0;
    if !(0..OFFSET_UUID).contains(&id) {
        return Err(UnpackMsgIdError::OutOfRange);
    }
    if id != 0 {
        return Ok((MsgId::Numbered(id), sys));
    }
    let Some(uuid) = unpack_uuid(unpacker) else {
        return Err(UnpackMsgIdError::MissingUuid);
    };
    let resolved = registry.lookup(uuid);
    Ok((MsgId::Ex { uuid, resolved }, sys))
}

/// Writes the leading `(id << 1) | sys` varint, followed by the 16-byte UUID for an extended
/// message — the encode half of [`unpack_msg_id`] (`client.cpp:179-188` builds this the same way
/// on the sending side, via `CMsgPacker`'s constructor).
pub fn pack_msg_id(packer: &mut Packer, id: MsgId, sys: bool) {
    match id {
        MsgId::Numbered(id) => {
            packer.add_int((id << 1) | i32::from(sys));
        }
        MsgId::Ex { uuid, .. } => {
            packer.add_int(i32::from(sys)); // id 0, i.e. NETMSGTYPE_EX
            pack_uuid(packer, uuid);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independently computed with Python's `hashlib.md5` (a separate MD5 implementation from
    /// the `md5` crate we use) following the exact algorithm in `uuid_manager.cpp:38-62`:
    /// `md5(bytes.fromhex("e05ddaaac4e64cfbb6425d48e80c0029") + name.encode())`, then
    /// `digest[6] = (digest[6] & 0x0F) | 0x30; digest[8] = (digest[8] & 0x3F) | 0x80`. This is
    /// the same computation DDNet's own `tools/uuid.cpp` performs; we can't build that C++ tool
    /// here without a full engine build, so an independent MD5 implementation plus the
    /// documented bit-twiddling serves as the oracle instead (task acceptance criterion 2's
    /// "differential test against a reference implementation", applied to the one piece of this
    /// module libtw2 does not implement at all — it has no `CalculateUuid` equivalent).
    const EXPECTED: &[(&str, &str)] = &[
        ("what-is@ddnet.tw", "245e5097-9fe0-39d6-bf7d-9a29e1691e4c"),
        ("it-is@ddnet.tw", "6954847e-2e87-3603-b562-36da29ed1aca"),
        ("i-dont-know@ddnet.tw", "416911b5-7973-33bf-8d52-7bf01e519cf0"),
        ("rcon-type@ddnet.tw", "12810e1f-a1db-3378-b4fb-164ed6505926"),
        ("map-details@ddnet.tw", "f9117b3c-8039-3416-9fc0-aef2bcb75c03"),
        ("capabilities@ddnet.tw", "f621a5a1-f585-3775-8e73-41beee79f2b2"),
        ("clientver@ddnet.tw", "8c001304-8461-3e47-8787-f672b3835bd4"),
        ("ping@ddnet.tw", "bcb43bf5-427c-36d8-b5b8-7975c8c06aa1"),
        ("pong@ddnet.tw", "d8295530-14a7-3a0a-b02e-b2cee08d2033"),
        ("checksum-request@ddnet.tw", "60a7cef1-2ecc-3ed4-b138-00fd0c8f5994"),
        ("checksum-response@ddnet.tw", "88fc61ec-5a3c-3fc3-8dfa-fd3b715db9e0"),
        ("checksum-error@ddnet.tw", "090960d1-4000-3fd5-9670-4976ae702a6a"),
        ("redirect@ddnet.org", "4efe406a-7774-33f1-bfde-1806ff6d1528"),
        ("rcon-cmd-group-start@ddnet.org", "85f67ffe-f1b1-3af3-98c4-26dbf77111b7"),
        ("rcon-cmd-group-end@ddnet.org", "5e02c980-6ca1-3c99-a9af-4650ae956252"),
        ("map-reload@ddnet.org", "9a9b28a3-19b0-37d9-b1f4-2cccfba05bac"),
        ("reconnect@ddnet.org", "5f4d5db7-3947-3711-b04e-07a1ff23c970"),
        ("sv-maplist-add@ddnet.org", "ca956101-b034-3339-92ca-aa104b20d770"),
        ("sv-maplist-start@ddnet.org", "d2fafec0-5cd2-319a-a84d-480f2072dee4"),
        ("sv-maplist-end@ddnet.org", "43fd0a8b-8b23-350d-b3f6-0de549246a70"),
    ];

    #[test]
    fn calculate_uuid_matches_independent_md5_oracle() {
        for &(name, expected) in EXPECTED {
            let uuid = calculate_uuid(name);
            assert_eq!(uuid.to_string(), expected, "for name {name}");
        }
    }

    #[test]
    fn calculate_uuid_sets_version_and_variant_bits() {
        for &name in REGISTERED_NAMES {
            let uuid = calculate_uuid(name);
            assert_eq!(uuid.0[6] >> 4, 0x3, "version nibble for {name}");
            assert_eq!(uuid.0[8] >> 6, 0b10, "variant bits for {name}");
        }
    }

    #[test]
    fn registry_assigns_sequential_ids_in_declaration_order() {
        let registry = UuidRegistry::new();
        assert_eq!(registry.len(), REGISTERED_NAMES.len());
        for (i, &name) in REGISTERED_NAMES.iter().enumerate() {
            let id = OFFSET_UUID + i as i32;
            assert_eq!(registry.name(id), Some(name));
            assert_eq!(registry.uuid(id), Some(calculate_uuid(name)));
            assert_eq!(registry.lookup(calculate_uuid(name)), Some(id));
        }
    }

    #[test]
    fn registry_rejects_unregistered_uuid_and_out_of_range_id() {
        let registry = UuidRegistry::new();
        assert_eq!(registry.lookup(calculate_uuid("not-a-real-message@example.com")), None);
        assert_eq!(registry.uuid(OFFSET_UUID + 9999), None);
        assert_eq!(registry.name(OFFSET_UUID - 1), None);
        assert_eq!(registry.uuid(0), None);
    }

    #[test]
    fn msg_id_roundtrip_numbered_system_message() {
        let registry = UuidRegistry::new();
        let mut buf = [0u8; 32];
        let mut packer = Packer::new(&mut buf);
        pack_msg_id(&mut packer, MsgId::Numbered(4), true);
        assert!(!packer.error());

        let mut unpacker = Unpacker::new(packer.data());
        let (id, sys) = unpack_msg_id(&mut unpacker, &registry).unwrap();
        assert!(sys);
        assert_eq!(id, MsgId::Numbered(4));
    }

    #[test]
    fn msg_id_roundtrip_known_extended_message() {
        let registry = UuidRegistry::new();
        let uuid = calculate_uuid("map-details@ddnet.tw");
        let mut buf = [0u8; 32];
        let mut packer = Packer::new(&mut buf);
        pack_msg_id(&mut packer, MsgId::Ex { uuid, resolved: None }, true);
        assert!(!packer.error());

        let mut unpacker = Unpacker::new(packer.data());
        let (id, sys) = unpack_msg_id(&mut unpacker, &registry).unwrap();
        assert!(sys);
        match id {
            MsgId::Ex { uuid: got, resolved } => {
                assert_eq!(got, uuid);
                assert_eq!(resolved, registry.lookup(uuid));
                assert!(resolved.is_some());
            }
            MsgId::Numbered(_) => panic!("expected an extended message id"),
        }
    }

    #[test]
    fn msg_id_unknown_extended_message_resolves_to_none_not_an_error() {
        // DDNet ignores unknown extended messages rather than treating them as malformed input —
        // a third-party mod or newer server sending a UUID we don't know is normal, not hostile.
        let registry = UuidRegistry::new();
        let unknown = calculate_uuid("some-future-message@example.com");
        let mut buf = [0u8; 32];
        let mut packer = Packer::new(&mut buf);
        pack_msg_id(
            &mut packer,
            MsgId::Ex {
                uuid: unknown,
                resolved: None,
            },
            false,
        );

        let mut unpacker = Unpacker::new(packer.data());
        let (id, sys) = unpack_msg_id(&mut unpacker, &registry).unwrap();
        assert!(!sys);
        match id {
            MsgId::Ex { uuid: got, resolved } => {
                assert_eq!(got, unknown);
                assert_eq!(resolved, None);
            }
            MsgId::Numbered(_) => panic!("expected an extended message id"),
        }
    }

    #[test]
    fn unpack_msg_id_rejects_out_of_range_and_truncated_input() {
        let registry = UuidRegistry::new();

        // Empty input: no leading int at all.
        let mut unpacker = Unpacker::new(&[]);
        assert_eq!(
            unpack_msg_id(&mut unpacker, &registry),
            Err(UnpackMsgIdError::NoLeadingInt)
        );

        // `id` decodes to exactly OFFSET_UUID (out of range: must be strictly less).
        let mut buf = [0u8; 8];
        let mut packer = Packer::new(&mut buf);
        packer.add_int(OFFSET_UUID << 1);
        let mut unpacker = Unpacker::new(packer.data());
        assert_eq!(
            unpack_msg_id(&mut unpacker, &registry),
            Err(UnpackMsgIdError::OutOfRange)
        );

        // Negative id: also out of range.
        let mut buf2 = [0u8; 8];
        let mut packer2 = Packer::new(&mut buf2);
        packer2.add_int(-2);
        let mut unpacker2 = Unpacker::new(packer2.data());
        assert_eq!(
            unpack_msg_id(&mut unpacker2, &registry),
            Err(UnpackMsgIdError::OutOfRange)
        );

        // NETMSGTYPE_EX marker (id 0) with no UUID bytes following.
        let mut buf3 = [0u8; 8];
        let mut packer3 = Packer::new(&mut buf3);
        packer3.add_int(0);
        let mut unpacker3 = Unpacker::new(packer3.data());
        assert_eq!(
            unpack_msg_id(&mut unpacker3, &registry),
            Err(UnpackMsgIdError::MissingUuid)
        );
    }

    #[test]
    fn uuid_display_matches_ddnet_format() {
        assert_eq!(ZEROED.to_string(), "00000000-0000-0000-0000-000000000000");
        let uuid = Uuid([
            0x8d, 0x30, 0x0e, 0xcf, 0x58, 0x73, 0x42, 0x97, 0xbe, 0xe5, 0x95, 0x66, 0x8f, 0xdf, 0xf3, 0x20,
        ]);
        // uuid_test.cpp: Uuid.FromToString
        assert_eq!(uuid.to_string(), "8d300ecf-5873-4297-bee5-95668fdff320");
    }
}
