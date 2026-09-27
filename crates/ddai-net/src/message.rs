// Ported from DDNet `src/engine/shared/protocol_ex.cpp` (the `NETMSG_WHATIS`/`IDONTKNOW`/`ITIS`
// answer construction) and the ex-message decode call sites in `src/engine/client/client.cpp`
// (`MAP_DETAILS`/`CAPABILITIES`/`PINGEX`/`CHECKSUM_*`/`REDIRECT`/`RECONNECT`, none of which have a
// `datasrc/network.py` entry — pinned rev c9d208138f85755521f16a0096b6fe036c5c8698, "20.1"), which
// carries the original Teeworlds zlib-style notice:
//
//   (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information.
//   If you are missing that file, acquire a complete release at teeworlds.com.
//
// This is an *altered* source version: rewritten in safe Rust, same ids/byte layouts. See
// docs/formats.md for the byte layout.
//
//! Top-level message dispatch: combines [`crate::sysmsg`] (numbered system), [`ExSysMsg`] (UUID
//! system, this module), and [`crate::generated::messages`] (numbered/UUID game) behind one
//! [`decode`]/[`encode`] pair keyed on the wire's leading `(id << 1) | sys` varint + UUID
//! extension (`crate::uuid`) — task 2.2b acceptance criterion 2.
//!
//! [`Registry`] is the one place this crate combines *every* UUID name it knows (engine system ex
//! messages, `crate::uuid::REGISTERED_NAMES`, plus game ex messages,
//! `crate::generated::messages::EX_NAMES`) into a single [`crate::uuid::UuidRegistry`] — needed
//! because [`decode`] must resolve one UUID off the wire not knowing in advance which of those two
//! disjoint sets it belongs to (the `sys` bit, read separately, then says which decode table to
//! use — see [`decode`]). Game *object* UUID names (`crate::generated::objects::EX_NAMES`) are
//! deliberately not part of this registry: those only ever appear inside a snapshot item's
//! type-descriptor, resolved by `crate::view` against its own, separate registry — never at this
//! (message) layer.

use crate::generated::messages as msgs;
use crate::packer::{Packer, SanitizeMode, Unpacker};
use crate::sysmsg::{self, SysMsg};
use crate::uuid::{self, MsgId, Uuid, UuidRegistry};

/// One combined [`crate::uuid::UuidRegistry`] for every UUID this crate resolves at the message
/// (not snapshot-object) layer — see the module docs.
pub struct Registry {
    uuids: UuidRegistry,
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry {
    pub fn new() -> Self {
        let mut names: Vec<&'static str> = Vec::with_capacity(uuid::REGISTERED_NAMES.len() + msgs::EX_NAMES.len());
        names.extend_from_slice(uuid::REGISTERED_NAMES);
        names.extend_from_slice(msgs::EX_NAMES);
        Registry {
            uuids: UuidRegistry::from_names(&names),
        }
    }

    fn name_of(&self, uuid: Uuid) -> Option<&'static str> {
        let id = self.uuids.lookup(uuid)?;
        self.uuids.name(id)
    }

    /// The full inner registry, for callers that need [`crate::uuid::unpack_msg_id`] directly
    /// (e.g. to resolve `resolved: Option<i32>` themselves).
    pub fn uuids(&self) -> &UuidRegistry {
        &self.uuids
    }
}

/// A decoded UUID (`ex`) *system* message — `sys` bit set, id resolved via UUID rather than a
/// small number. Unlike [`crate::generated::messages`] (mechanically generated from
/// `datasrc/network.py`), these have no datasrc description (same situation as
/// [`crate::sysmsg::SysMsg`]) — hand-typed from `protocol_ex_msgs.h` + the C++ decode call sites.
/// [`ExSysMsg::Unhandled`] covers every registered name this module does not otherwise type out
/// (`rcon-type@ddnet.tw`, `map-reload@ddnet.org`, the `sv-maplist-*@ddnet.org` trio,
/// `rcon-cmd-group-*@ddnet.org`) plus, from [`decode_ex`], any name outside this crate's registry
/// entirely — raw payload preserved either way, matching this crate's "never drop, never panic"
/// philosophy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExSysMsg {
    /// S->C.
    MapDetails {
        name: String,
        sha256: [u8; 32],
        crc: i32,
        size: i32,
        url: String,
    },
    /// S->C.
    Capabilities {
        version: i32,
        flags: i32,
    },
    /// C->S.
    ClientVer {
        connection_id: [u8; 16],
        ddnet_version: i32,
        version_str: String,
    },
    /// Either direction: answer with [`ExSysMsg::PongEx`] carrying the same `id`.
    PingEx {
        id: [u8; 16],
    },
    PongEx {
        id: [u8; 16],
    },
    /// S->C: `uuid` names which checksum module is being requested (the rest of that submodule's
    /// own payload — out of this crate's scope, D-029/goal: the bot never needs to answer this
    /// truthfully to play — is kept raw).
    ChecksumRequest {
        uuid: [u8; 16],
        rest: Vec<u8>,
    },
    /// C->S.
    ChecksumResponse {
        uuid: [u8; 16],
        sha256: [u8; 32],
    },
    /// C->S.
    ChecksumError {
        uuid: [u8; 16],
        error: i32,
    },
    /// S->C: reconnect to the same address, different port.
    Redirect {
        port: i32,
    },
    /// S->C: reconnect to the same address.
    Reconnect,
    /// Either direction: "what does UUID `uuid` mean to you?" — see [`respond_to_whatis`].
    WhatIs {
        uuid: Uuid,
    },
    ItIs {
        uuid: Uuid,
        name: String,
    },
    IDontKnow {
        uuid: Uuid,
    },
    /// A registered name this module does not decode further (see the enum docs), or an
    /// unregistered/unresolvable UUID (from [`decode_ex`]) — `name` is `None` in the latter case.
    Unhandled {
        name: Option<&'static str>,
        raw: Vec<u8>,
    },
}

fn decode_typed_ex(name: &str, unpacker: &mut Unpacker) -> Option<ExSysMsg> {
    Some(match name {
        "map-details@ddnet.tw" => {
            let mode = SanitizeMode {
                sanitize: false,
                sanitize_cc: true,
                skip_start_whitespace: true,
            };
            let map_name = unpacker.get_string(mode);
            let sha256 = unpacker.get_raw(32)?.try_into().ok()?;
            let crc = unpacker.get_int();
            let size = unpacker.get_int();
            if unpacker.error() {
                return None;
            }
            // `url` is read tolerantly — a missing/malformed url falls back to "", matching
            // `client.cpp:1699-1700`'s own `if(Unpacker.Error()) pMapUrl = "";` (it does not
            // invalidate the rest of the already-successfully-read message).
            let url_mode = SanitizeMode {
                sanitize: false,
                sanitize_cc: true,
                skip_start_whitespace: false,
            };
            let url = unpacker.get_string(url_mode);
            let url = if unpacker.error() { String::new() } else { url };
            return Some(ExSysMsg::MapDetails {
                name: map_name,
                sha256,
                crc,
                size,
                url,
            });
        }
        "capabilities@ddnet.tw" => ExSysMsg::Capabilities {
            version: unpacker.get_int(),
            flags: unpacker.get_int(),
        },
        "clientver@ddnet.tw" => ExSysMsg::ClientVer {
            connection_id: unpacker.get_raw(16)?.try_into().ok()?,
            ddnet_version: unpacker.get_int(),
            version_str: unpacker.get_string(SanitizeMode::SANITIZE),
        },
        "ping@ddnet.tw" => ExSysMsg::PingEx {
            id: unpacker.get_raw(16)?.try_into().ok()?,
        },
        "pong@ddnet.tw" => ExSysMsg::PongEx {
            id: unpacker.get_raw(16)?.try_into().ok()?,
        },
        "checksum-request@ddnet.tw" => ExSysMsg::ChecksumRequest {
            uuid: unpacker.get_raw(16)?.try_into().ok()?,
            rest: unpacker.get_rest().to_vec(),
        },
        "checksum-response@ddnet.tw" => ExSysMsg::ChecksumResponse {
            uuid: unpacker.get_raw(16)?.try_into().ok()?,
            sha256: unpacker.get_raw(32)?.try_into().ok()?,
        },
        "checksum-error@ddnet.tw" => ExSysMsg::ChecksumError {
            uuid: unpacker.get_raw(16)?.try_into().ok()?,
            error: unpacker.get_int(),
        },
        "redirect@ddnet.org" => ExSysMsg::Redirect {
            port: unpacker.get_int(),
        },
        "reconnect@ddnet.org" => ExSysMsg::Reconnect,
        "what-is@ddnet.tw" => ExSysMsg::WhatIs {
            uuid: uuid::unpack_uuid(unpacker)?,
        },
        "it-is@ddnet.tw" => {
            let uuid = uuid::unpack_uuid(unpacker)?;
            ExSysMsg::ItIs {
                uuid,
                name: unpacker.get_string(SanitizeMode::SANITIZE_CC),
            }
        }
        "i-dont-know@ddnet.tw" => ExSysMsg::IDontKnow {
            uuid: uuid::unpack_uuid(unpacker)?,
        },
        _ => return None,
    })
}

/// Decodes an ex (UUID) system message body, given its already-resolved `name` (`None` if the
/// UUID was not one this crate's [`Registry`] knows at all — always [`ExSysMsg::Unhandled`] in
/// that case). Never panics; a registered-but-not-specifically-decoded name, or one whose payload
/// fails to decode, falls back to [`ExSysMsg::Unhandled`] with the raw remaining payload.
pub fn decode_ex(name: Option<&'static str>, unpacker: &mut Unpacker) -> ExSysMsg {
    let raw_on_failure = unpacker.remaining().to_vec();
    match name.and_then(|n| decode_typed_ex(n, unpacker)) {
        Some(msg) if !unpacker.error() => msg,
        _ => ExSysMsg::Unhandled {
            name,
            raw: raw_on_failure,
        },
    }
}

/// Encodes an ex system message's body (leading `(0<<1)|sys` varint + 16-byte UUID is
/// [`encode`]'s job, not this function's). [`ExSysMsg::Unhandled`] cannot be re-encoded (its
/// original UUID is not necessarily even in [`Registry`]) and is a no-op.
pub fn encode_ex(msg: &ExSysMsg, packer: &mut Packer) {
    match msg {
        ExSysMsg::MapDetails {
            name,
            sha256,
            crc,
            size,
            url,
        } => {
            packer.add_string(name, 0, true);
            packer.add_raw(sha256);
            packer.add_int(*crc);
            packer.add_int(*size);
            packer.add_string(url, 0, true);
        }
        ExSysMsg::Capabilities { version, flags } => {
            packer.add_int(*version);
            packer.add_int(*flags);
        }
        ExSysMsg::ClientVer {
            connection_id,
            ddnet_version,
            version_str,
        } => {
            packer.add_raw(connection_id);
            packer.add_int(*ddnet_version);
            packer.add_string(version_str, 0, true);
        }
        ExSysMsg::PingEx { id } | ExSysMsg::PongEx { id } => packer.add_raw(id),
        ExSysMsg::ChecksumRequest { uuid, rest } => {
            packer.add_raw(uuid);
            packer.add_raw(rest);
        }
        ExSysMsg::ChecksumResponse { uuid, sha256 } => {
            packer.add_raw(uuid);
            packer.add_raw(sha256);
        }
        ExSysMsg::ChecksumError { uuid, error } => {
            packer.add_raw(uuid);
            packer.add_int(*error);
        }
        ExSysMsg::Redirect { port } => packer.add_int(*port),
        ExSysMsg::Reconnect => {}
        ExSysMsg::WhatIs { uuid } => uuid::pack_uuid(packer, *uuid),
        ExSysMsg::ItIs { uuid, name } => {
            uuid::pack_uuid(packer, *uuid);
            packer.add_string(name, 0, true);
        }
        ExSysMsg::IDontKnow { uuid } => uuid::pack_uuid(packer, *uuid),
        ExSysMsg::Unhandled { .. } => {}
    }
}

/// `UnpackMessageId`'s `NETMSG_WHATIS` handling (`protocol_ex.cpp:55-73`): a peer asked what
/// `uuid` means to us — answers truthfully with [`ExSysMsg::ItIs`] if `registry` knows it (from
/// *either* the system-ex or game-ex namespace, since a peer may ask about any UUID we might
/// plausibly use) or [`ExSysMsg::IDontKnow`] otherwise. The bot always answers honestly here —
/// this is protocol metadata (what names a peer supports), not gameplay or chat, so it is not
/// covered by D-007's "never send chat" restriction.
pub fn respond_to_whatis(uuid: Uuid, registry: &Registry) -> ExSysMsg {
    match registry.name_of(uuid) {
        Some(name) => ExSysMsg::ItIs {
            uuid,
            name: name.to_string(),
        },
        None => ExSysMsg::IDontKnow { uuid },
    }
}

/// One fully decoded message, whichever of the four decode tables it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)] // see the same justification on `crate::generated::messages::GameMsg`
pub enum Msg {
    Sys(SysMsg),
    ExSys(ExSysMsg),
    Game(msgs::GameMsg),
    ExGame(msgs::ExGameMsg),
    /// `Sv_TuneParams` (`NETMSGTYPE_SV_TUNEPARAMS`), hand-decoded (`crate::tuning`; task 2.2b
    /// review round 1, finding F2) — `datasrc/network.py` declares this message with zero
    /// fields (DDNet reads it by hand, `gameclient.cpp:1058-1090`), so [`decode`] intercepts this
    /// id itself and returns the real tuning values here; `Game(GameMsg::SvTuneParams(_))` (the
    /// mechanically generated, always-empty struct) is never returned for this id.
    TuneParams(crate::tuning::TuneParams),
    /// `Sv_TeamsState`/`Sv_TeamsStateLegacy`, hand-decoded (`crate::tuning`; same situation as
    /// `TuneParams`, `gameclient.cpp:1174-1193`) — covers both the numbered legacy id and the
    /// `teamsstate@netmsg.ddnet.tw` ex id.
    TeamsState(crate::tuning::TeamsState),
    /// The leading id/UUID itself failed to decode at all (`crate::uuid::UnpackMsgIdError`) —
    /// the whole chunk payload is unusable, matching DDNet's own `UNPACKMESSAGE_ERROR` (the
    /// message, not just this decode call, is dropped).
    Invalid,
}

/// Decodes one chunk payload into a [`Msg`] (the leading `(id << 1) | sys` varint, and the 16-byte
/// UUID extension if present, per `crate::uuid::unpack_msg_id`) plus whether it answers a
/// `NETMSG_WHATIS` (in which case the caller should send that answer back, vital — matches
/// `UNPACKMESSAGE_ANSWER`, `protocol_ex.cpp:20-31`, which this crate's higher layer, not
/// `crate::uuid::unpack_msg_id` itself, is responsible for constructing — see
/// [`respond_to_whatis`]). Never panics: every failure mode decodes to [`Msg::Invalid`] or a
/// `*::Unhandled` variant instead.
pub fn decode(payload: &[u8], registry: &Registry) -> (Msg, Option<ExSysMsg>) {
    let mut unpacker = Unpacker::new(payload);
    let Ok((id, sys)) = uuid::unpack_msg_id(&mut unpacker, registry.uuids()) else {
        return (Msg::Invalid, None);
    };
    match id {
        MsgId::Numbered(numbered_id) => {
            if sys {
                (Msg::Sys(sysmsg::decode(numbered_id, &mut unpacker)), None)
            } else if numbered_id == msgs::id::NETMSGTYPE_SV_TUNEPARAMS {
                // F2: intercept before the generated (always-empty) dispatch — see `crate::tuning`.
                (
                    Msg::TuneParams(crate::tuning::decode_sv_tune_params(&mut unpacker)),
                    None,
                )
            } else if numbered_id == msgs::id::NETMSGTYPE_SV_TEAMSSTATELEGACY {
                (Msg::TeamsState(crate::tuning::decode_teams_state(&mut unpacker)), None)
            } else {
                match msgs::decode_game_msg(numbered_id, &mut unpacker) {
                    Some(m) => (Msg::Game(m), None),
                    None => (Msg::Invalid, None),
                }
            }
        }
        MsgId::Ex { uuid: _, resolved } => {
            let name = resolved.and_then(|id| registry.uuids().name(id));
            if sys {
                let msg = decode_ex(name, &mut unpacker);
                let answer = if let ExSysMsg::WhatIs { uuid: asked } = &msg {
                    Some(respond_to_whatis(*asked, registry))
                } else {
                    None
                };
                (Msg::ExSys(msg), answer)
            } else if name == Some("teamsstate@netmsg.ddnet.tw") {
                // F2: same interception, ex (UUID) id for the non-legacy `Sv_TeamsState`.
                (Msg::TeamsState(crate::tuning::decode_teams_state(&mut unpacker)), None)
            } else {
                match name.and_then(|n| msgs::decode_ex_game_msg(n, &mut unpacker)) {
                    Some(m) => (Msg::ExGame(m), None),
                    None => (Msg::Invalid, None),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_full(sys: bool, uuid_name: Option<&str>, body: impl Fn(&mut Packer)) -> Vec<u8> {
        let mut buf = [0u8; 4096];
        let mut packer = Packer::new(&mut buf);
        match uuid_name {
            None => panic!("use encode_numbered for numbered ids"),
            Some(name) => {
                let uuid = crate::uuid::calculate_uuid(name);
                uuid::pack_msg_id(&mut packer, MsgId::Ex { uuid, resolved: None }, sys);
            }
        }
        body(&mut packer);
        packer.data().to_vec()
    }

    fn encode_numbered(sys: bool, id: i32, body: impl Fn(&mut Packer)) -> Vec<u8> {
        let mut buf = [0u8; 4096];
        let mut packer = Packer::new(&mut buf);
        uuid::pack_msg_id(&mut packer, MsgId::Numbered(id), sys);
        body(&mut packer);
        packer.data().to_vec()
    }

    #[test]
    fn decodes_numbered_system_message() {
        let registry = Registry::new();
        let payload = encode_numbered(true, sysmsg::id::READY, |_| {});
        let (msg, answer) = decode(&payload, &registry);
        assert_eq!(msg, Msg::Sys(SysMsg::Ready));
        assert_eq!(answer, None);
    }

    #[test]
    fn decodes_numbered_game_message() {
        let registry = Registry::new();
        let payload = encode_numbered(false, msgs::id::NETMSGTYPE_SV_READYTOENTER, |_| {});
        let (msg, _) = decode(&payload, &registry);
        assert_eq!(msg, Msg::Game(msgs::GameMsg::SvReadyToEnter(msgs::SvReadyToEnter {})));
    }

    #[test]
    fn decodes_ex_system_message() {
        let registry = Registry::new();
        let payload = encode_full(true, Some("reconnect@ddnet.org"), |_| {});
        let (msg, answer) = decode(&payload, &registry);
        assert_eq!(msg, Msg::ExSys(ExSysMsg::Reconnect));
        assert_eq!(answer, None);
    }

    #[test]
    fn decodes_ex_game_message() {
        let registry = Registry::new();
        let payload = encode_full(false, Some("showothers@netmsg.ddnet.tw"), |p| p.add_int(1));
        let (msg, _) = decode(&payload, &registry);
        assert_eq!(
            msg,
            Msg::ExGame(msgs::ExGameMsg::ClShowOthers(msgs::ClShowOthers { show: 1 }))
        );
    }

    #[test]
    fn whatis_known_uuid_answers_itis() {
        let registry = Registry::new();
        let asked = crate::uuid::calculate_uuid("reconnect@ddnet.org");
        let payload = encode_full(true, Some("what-is@ddnet.tw"), |p| uuid::pack_uuid(p, asked));
        let (msg, answer) = decode(&payload, &registry);
        assert_eq!(msg, Msg::ExSys(ExSysMsg::WhatIs { uuid: asked }));
        assert_eq!(
            answer,
            Some(ExSysMsg::ItIs {
                uuid: asked,
                name: "reconnect@ddnet.org".to_string()
            })
        );
    }

    #[test]
    fn whatis_unknown_uuid_answers_idontknow() {
        let registry = Registry::new();
        let asked = crate::uuid::calculate_uuid("totally-made-up@example.com");
        let payload = encode_full(true, Some("what-is@ddnet.tw"), |p| uuid::pack_uuid(p, asked));
        let (_, answer) = decode(&payload, &registry);
        assert_eq!(answer, Some(ExSysMsg::IDontKnow { uuid: asked }));
    }

    #[test]
    fn unknown_ex_name_is_unhandled_not_dropped_and_not_panicking() {
        let registry = Registry::new();
        let payload = encode_full(true, Some("rcon-type@ddnet.tw"), |p| p.add_int(1));
        let (msg, _) = decode(&payload, &registry);
        assert!(matches!(
            msg,
            Msg::ExSys(ExSysMsg::Unhandled {
                name: Some("rcon-type@ddnet.tw"),
                ..
            })
        ));
    }

    #[test]
    fn unregistered_uuid_is_invalid_not_panicking() {
        let registry = Registry::new();
        let payload = encode_full(true, Some("never-registered@example.com"), |p| p.add_int(1));
        let (msg, _) = decode(&payload, &registry);
        assert!(matches!(msg, Msg::ExSys(ExSysMsg::Unhandled { name: None, .. })));
    }

    #[test]
    fn empty_payload_is_invalid_not_panicking() {
        let registry = Registry::new();
        let (msg, answer) = decode(&[], &registry);
        assert_eq!(msg, Msg::Invalid);
        assert_eq!(answer, None);
    }

    #[test]
    fn garbage_payload_never_panics() {
        let registry = Registry::new();
        for len in 0..40 {
            let payload = vec![0xffu8; len];
            let _ = decode(&payload, &registry); // must not panic
        }
    }

    #[test]
    fn map_details_roundtrip() {
        let registry = Registry::new();
        let msg = ExSysMsg::MapDetails {
            name: "Copy Love Box".to_string(),
            sha256: [7u8; 32],
            crc: 123,
            size: 456,
            url: "https://example.com/map".to_string(),
        };
        let mut buf = [0u8; 4096];
        let mut packer = Packer::new(&mut buf);
        let uuid = crate::uuid::calculate_uuid("map-details@ddnet.tw");
        uuid::pack_msg_id(&mut packer, MsgId::Ex { uuid, resolved: None }, true);
        encode_ex(&msg, &mut packer);
        let (decoded, _) = decode(packer.data(), &registry);
        assert_eq!(decoded, Msg::ExSys(msg));
    }

    #[test]
    fn checksum_request_preserves_unparsed_remainder() {
        let registry = Registry::new();
        let mut buf = [0u8; 4096];
        let mut packer = Packer::new(&mut buf);
        let uuid = crate::uuid::calculate_uuid("checksum-request@ddnet.tw");
        uuid::pack_msg_id(&mut packer, MsgId::Ex { uuid, resolved: None }, true);
        packer.add_raw(&[9u8; 16]); // the "which module" uuid
        packer.add_raw(&[1, 2, 3, 4]); // opaque module-specific payload
        let (decoded, _) = decode(packer.data(), &registry);
        assert_eq!(
            decoded,
            Msg::ExSys(ExSysMsg::ChecksumRequest {
                uuid: [9u8; 16],
                rest: vec![1, 2, 3, 4],
            })
        );
    }

    // --- F2 (review round 1): `Sv_TuneParams`/`Sv_TeamsState*` must decode to real values via
    // the top-level `decode`, never the generated always-empty struct.

    #[test]
    fn decode_intercepts_sv_tune_params_with_real_values() {
        let registry = Registry::new();
        let payload = encode_numbered(false, msgs::id::NETMSGTYPE_SV_TUNEPARAMS, |p| {
            for i in 0..crate::tuning::NUM_TUNE_PARAMS {
                p.add_int(1000 + i as i32);
            }
        });
        let (decoded, _) = decode(&payload, &registry);
        match decoded {
            Msg::TuneParams(params) => {
                assert_eq!(params.received, crate::tuning::NUM_TUNE_PARAMS);
                assert_eq!(params.ground_control_speed, 1000); // index 0
                assert_eq!(params.gravity, 1012); // index 12
            }
            other => panic!("expected Msg::TuneParams, got {other:?}"),
        }
    }

    #[test]
    fn decode_intercepts_sv_teams_state_legacy_numbered_id() {
        let registry = Registry::new();
        let payload = encode_numbered(false, msgs::id::NETMSGTYPE_SV_TEAMSSTATELEGACY, |p| {
            p.add_int(2);
            p.add_int(-1); // stop here
        });
        let (decoded, _) = decode(&payload, &registry);
        match decoded {
            Msg::TeamsState(state) => {
                assert_eq!(state.received, 1);
                assert_eq!(state.teams[0], 2);
            }
            other => panic!("expected Msg::TeamsState, got {other:?}"),
        }
    }

    #[test]
    fn decode_intercepts_sv_teams_state_ex_name() {
        let registry = Registry::new();
        let payload = encode_full(false, Some("teamsstate@netmsg.ddnet.tw"), |p| {
            p.add_int(5);
        });
        let (decoded, _) = decode(&payload, &registry);
        match decoded {
            Msg::TeamsState(state) => {
                assert_eq!(state.received, 1);
                assert_eq!(state.teams[0], 5);
            }
            other => panic!("expected Msg::TeamsState, got {other:?}"),
        }
    }
}
