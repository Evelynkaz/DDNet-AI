// Ported from DDNet `src/engine/shared/protocol.h` (numbered ids) and the decode call sites in
// `src/engine/client/client.cpp` / `src/engine/server/server.cpp` (field lists — these hand-typed
// messages have no `datasrc/network.py` entry of their own, unlike the game messages in
// `crate::generated::messages`; pinned rev c9d208138f85755521f16a0096b6fe036c5c8698, "20.1"),
// which carries the original Teeworlds zlib-style notice:
//
//   (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information.
//   If you are missing that file, acquire a complete release at teeworlds.com.
//
// This is an *altered* source version: rewritten in safe Rust, same message ids and byte layouts,
// so it stays byte-for-byte compatible with the wire format every DDNet 0.6+DDNet client/server
// speaks. See docs/formats.md for the byte layout.
//
//! System messages (`NETMSG_*`, `sys` bit set) — task 2.2b acceptance criterion 2.
//!
//! Unlike `crate::generated::messages` (game messages, mechanically generated from
//! `datasrc/network.py`), these ids and field lists are hand-typed from `protocol.h` and the C++
//! decode call sites, because DDNet defines them by hand too (there is no datasrc description of
//! the system message layer). [`decode`]/[`decode_ex`] never panic; an id/name this module does
//! not have a specific variant for decodes to [`SysMsg::Unhandled`]/`ExSysMsg::Unhandled`
//! (raw payload preserved) rather than being dropped, so message counts (task acceptance
//! criterion 5) stay accurate even for messages this module does not fully type out.

use crate::packer::{Packer, SanitizeMode, Unpacker};

/// Numbered system message ids (`NETMSG_*`, `protocol.h:32-76`). `NETMSG_EX` (0) is not listed
/// here — that is [`crate::uuid::MsgId::Ex`], handled one layer down in `crate::message`.
pub mod id {
    pub const INFO: i32 = 1;
    pub const MAP_CHANGE: i32 = 2;
    pub const MAP_DATA: i32 = 3;
    pub const CON_READY: i32 = 4;
    pub const SNAP: i32 = 5;
    pub const SNAPEMPTY: i32 = 6;
    pub const SNAPSINGLE: i32 = 7;
    pub const SNAPSMALL: i32 = 8;
    pub const INPUTTIMING: i32 = 9;
    pub const RCON_AUTH_STATUS: i32 = 10;
    pub const RCON_LINE: i32 = 11;
    pub const UNUSED1: i32 = 12;
    pub const UNUSED2: i32 = 13;
    pub const READY: i32 = 14;
    pub const ENTERGAME: i32 = 15;
    pub const INPUT: i32 = 16;
    pub const RCON_CMD: i32 = 17;
    pub const RCON_AUTH: i32 = 18;
    pub const REQUEST_MAP_DATA: i32 = 19;
    pub const UNUSED3: i32 = 20;
    pub const UNUSED4: i32 = 21;
    pub const PING: i32 = 22;
    pub const PING_REPLY: i32 = 23;
    pub const UNUSED5: i32 = 24;
    pub const RCON_CMD_ADD: i32 = 25;
    pub const RCON_CMD_REM: i32 = 26;
}

/// Maximum input size DDNet accepts, in `i32`s (`MAX_INPUT_SIZE`, `protocol.h:106`) — bounds
/// [`decode`]'s read of [`SysMsg::Input`]'s `data` regardless of what the wire claims `size` is.
pub const MAX_INPUT_INTS: usize = 128;

/// A decoded numbered system message. [`SysMsg::Unhandled`] covers every numbered id this module
/// does not otherwise type out (`RCON_CMD`/`RCON_AUTH`/the `UNUSED*` placeholders, and any future
/// id we do not yet know) — the raw payload (after the leading id varint) is preserved so nothing
/// is silently dropped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SysMsg {
    /// C->S. `netversion` is `"0.6 626fce9a778df4d4"` for 0.6+DDNet.
    Info { netversion: String, password: String },
    /// S->C.
    MapChange { name: String, crc: i32, size: i32 },
    /// S->C.
    MapData {
        last: i32,
        crc: i32,
        chunk: i32,
        data: Vec<u8>,
    },
    /// S->C: connection ready, client should send `StartInfo`.
    ConReady,
    /// S->C: one part of a multi-part snapshot (`crate::snapshot` assembles these).
    Snap {
        tick: i32,
        delta_tick: i32,
        num_parts: i32,
        part: i32,
        crc: i32,
        data: Vec<u8>,
    },
    /// S->C: the delta against `delta_tick` was empty (nothing changed).
    SnapEmpty { tick: i32, delta_tick: i32 },
    /// S->C: a whole (single-part) snapshot.
    SnapSingle {
        tick: i32,
        delta_tick: i32,
        crc: i32,
        data: Vec<u8>,
    },
    /// S->C. Reserved id, unused by real DDNet servers; decoded the same shape as `SnapSingle`
    /// for tolerance (`protocol.h:47`: "not used").
    SnapSmall {
        tick: i32,
        delta_tick: i32,
        crc: i32,
        data: Vec<u8>,
    },
    /// S->C: how far off `pred_tick`'s `NETMSG_INPUT` was.
    InputTiming { pred_tick: i32, time_left: i32 },
    /// S->C.
    RconAuthStatus { result: i32, use_temp_rcon_commands: i32 },
    /// S->C.
    RconLine { line: String },
    /// C->S: client has loaded the map, mod should send its own init next.
    Ready,
    /// C->S: tells the server to start sending snapshots.
    EnterGame,
    /// C->S, not vital: `ack_game_tick`/`pred_tick` plus `data[0..size/4]` raw `CNetObj_PlayerInput`
    /// ints (`crate::generated::objects::PlayerInput`), capped at [`MAX_INPUT_INTS`] regardless of
    /// what `size` (attacker/peer-controlled) claims.
    Input {
        ack_game_tick: i32,
        pred_tick: i32,
        size: i32,
        data: Vec<i32>,
    },
    /// C->S.
    RequestMapData { chunk: i32 },
    /// S->C.
    Ping,
    /// S/C, whichever answers a [`SysMsg::Ping`].
    PingReply,
    /// S->C: a `rcon_cmd`-style console command becomes available.
    RconCmdAdd { name: String, help: String, params: String },
    /// S->C.
    RconCmdRem { name: String },
    /// C->S: a console command to run (only meaningful once authed via [`SysMsg::RconAuth`]) —
    /// the bot has no rcon password and never sends this; decoded/encoded for completeness.
    RconCmd { command: String },
    /// C->S: rcon login attempt. `one` is always `1` in every real client (`client.cpp:311-319`
    /// always calls `Msg.AddInt(1)`); kept as a field rather than hard-coded so decode is exact
    /// about whatever a real (possibly non-standard) peer actually sent.
    RconAuth { name: String, password: String, one: i32 },
    /// Every numbered id not decoded into a specific variant above (raw payload, without the
    /// leading id varint).
    Unhandled { msg_id: i32, raw: Vec<u8> },
}

/// Decodes a numbered system message payload (`Unpacker` positioned right after the leading
/// `(id<<1)|sys` varint — see `crate::message`). Never panics; malformed input for a message this
/// module type-decodes falls back to [`SysMsg::Unhandled`] with whatever raw bytes remain, exactly
/// like the rest of this crate's "poison, don't panic, don't drop" philosophy.
pub fn decode(msg_id: i32, unpacker: &mut Unpacker) -> SysMsg {
    let rest_on_failure = unpacker.remaining().to_vec();
    let decoded = decode_typed(msg_id, unpacker);
    match decoded {
        Some(msg) if !unpacker.error() => msg,
        _ => SysMsg::Unhandled {
            msg_id,
            raw: rest_on_failure,
        },
    }
}

fn decode_typed(msg_id: i32, unpacker: &mut Unpacker) -> Option<SysMsg> {
    Some(match msg_id {
        id::INFO => SysMsg::Info {
            netversion: unpacker.get_string(SanitizeMode::SANITIZE),
            password: unpacker.get_string(SanitizeMode::SANITIZE),
        },
        id::MAP_CHANGE => {
            let mode = SanitizeMode {
                sanitize: false,
                sanitize_cc: true,
                skip_start_whitespace: true,
            };
            let name = unpacker.get_string(mode);
            let crc = unpacker.get_int();
            let size = unpacker.get_int();
            SysMsg::MapChange { name, crc, size }
        }
        id::MAP_DATA => {
            let last = unpacker.get_int();
            let crc = unpacker.get_int();
            let chunk = unpacker.get_int();
            let size = unpacker.get_int();
            let data = get_raw_bounded(unpacker, size)?;
            SysMsg::MapData { last, crc, chunk, data }
        }
        id::CON_READY => SysMsg::ConReady,
        id::SNAP => {
            let tick = unpacker.get_int();
            let delta_tick = tick.wrapping_sub(unpacker.get_int());
            let num_parts = unpacker.get_int();
            let part = unpacker.get_int();
            let crc = unpacker.get_int();
            let size = unpacker.get_int();
            let data = get_raw_bounded(unpacker, size)?;
            SysMsg::Snap {
                tick,
                delta_tick,
                num_parts,
                part,
                crc,
                data,
            }
        }
        id::SNAPEMPTY => {
            let tick = unpacker.get_int();
            let delta_tick = tick.wrapping_sub(unpacker.get_int());
            SysMsg::SnapEmpty { tick, delta_tick }
        }
        id::SNAPSINGLE => {
            let tick = unpacker.get_int();
            let delta_tick = tick.wrapping_sub(unpacker.get_int());
            let crc = unpacker.get_int();
            let size = unpacker.get_int();
            let data = get_raw_bounded(unpacker, size)?;
            SysMsg::SnapSingle {
                tick,
                delta_tick,
                crc,
                data,
            }
        }
        id::SNAPSMALL => {
            let tick = unpacker.get_int();
            let delta_tick = tick.wrapping_sub(unpacker.get_int());
            let crc = unpacker.get_int();
            let size = unpacker.get_int();
            let data = get_raw_bounded(unpacker, size)?;
            SysMsg::SnapSmall {
                tick,
                delta_tick,
                crc,
                data,
            }
        }
        id::INPUTTIMING => SysMsg::InputTiming {
            pred_tick: unpacker.get_int(),
            time_left: unpacker.get_int(),
        },
        id::RCON_AUTH_STATUS => SysMsg::RconAuthStatus {
            result: unpacker.get_int(),
            use_temp_rcon_commands: unpacker.get_int_or_default(0),
        },
        id::RCON_LINE => SysMsg::RconLine {
            line: unpacker.get_string(SanitizeMode::SANITIZE),
        },
        id::READY => SysMsg::Ready,
        id::ENTERGAME => SysMsg::EnterGame,
        id::INPUT => {
            let ack_game_tick = unpacker.get_int();
            let pred_tick = unpacker.get_int();
            let size = unpacker.get_int();
            let count = (size.max(0) as usize / 4).min(MAX_INPUT_INTS);
            let mut data = Vec::with_capacity(count);
            for _ in 0..count {
                data.push(unpacker.get_int());
            }
            SysMsg::Input {
                ack_game_tick,
                pred_tick,
                size,
                data,
            }
        }
        id::REQUEST_MAP_DATA => SysMsg::RequestMapData {
            chunk: unpacker.get_int(),
        },
        id::PING => SysMsg::Ping,
        id::PING_REPLY => SysMsg::PingReply,
        id::RCON_CMD_ADD => SysMsg::RconCmdAdd {
            name: unpacker.get_string(SanitizeMode::SANITIZE_CC),
            help: unpacker.get_string(SanitizeMode::SANITIZE_CC),
            params: unpacker.get_string(SanitizeMode::SANITIZE_CC),
        },
        id::RCON_CMD_REM => SysMsg::RconCmdRem {
            name: unpacker.get_string(SanitizeMode::SANITIZE_CC),
        },
        id::RCON_CMD => SysMsg::RconCmd {
            command: unpacker.get_string(SanitizeMode::SANITIZE),
        },
        id::RCON_AUTH => SysMsg::RconAuth {
            name: unpacker.get_string(SanitizeMode::SANITIZE),
            password: unpacker.get_string(SanitizeMode::SANITIZE),
            one: unpacker.get_int_or_default(1),
        },
        _ => return None,
    })
}

/// Reads `size` raw bytes, but never a negative or absurd amount — `size` comes straight off the
/// wire, so this is the one bounds check standing between a hostile peer's `size` field and an
/// attempted multi-gigabyte allocation. Returns `None` (falls back to [`SysMsg::Unhandled`]) if
/// `size` is negative or larger than a packet could ever legitimately carry.
fn get_raw_bounded(unpacker: &mut Unpacker, size: i32) -> Option<Vec<u8>> {
    if !(0..=crate::packet::MAX_CHUNK_DATA_SIZE as i32).contains(&size) {
        return None;
    }
    unpacker.get_raw(size as usize).map(<[u8]>::to_vec)
}

/// Encodes a numbered system message payload (leading id varint is `crate::message`'s job, not
/// this function's — matches [`decode`]'s scope). `Unhandled` cannot be re-encoded (its original
/// id/sys bit were never recorded) and is skipped (no-op) rather than panicking.
pub fn encode(msg: &SysMsg, packer: &mut Packer) {
    match msg {
        SysMsg::Info { netversion, password } => {
            packer.add_string(netversion, 0, true);
            packer.add_string(password, 0, true);
        }
        SysMsg::MapChange { name, crc, size } => {
            packer.add_string(name, 0, true);
            packer.add_int(*crc);
            packer.add_int(*size);
        }
        SysMsg::MapData { last, crc, chunk, data } => {
            packer.add_int(*last);
            packer.add_int(*crc);
            packer.add_int(*chunk);
            packer.add_int(data.len() as i32);
            packer.add_raw(data);
        }
        SysMsg::ConReady | SysMsg::Ready | SysMsg::EnterGame | SysMsg::Ping | SysMsg::PingReply => {}
        SysMsg::Snap {
            tick,
            delta_tick,
            num_parts,
            part,
            crc,
            data,
        } => {
            packer.add_int(*tick);
            packer.add_int(tick.wrapping_sub(*delta_tick));
            packer.add_int(*num_parts);
            packer.add_int(*part);
            packer.add_int(*crc);
            packer.add_int(data.len() as i32);
            packer.add_raw(data);
        }
        SysMsg::SnapEmpty { tick, delta_tick } => {
            packer.add_int(*tick);
            packer.add_int(tick.wrapping_sub(*delta_tick));
        }
        SysMsg::SnapSingle {
            tick,
            delta_tick,
            crc,
            data,
        }
        | SysMsg::SnapSmall {
            tick,
            delta_tick,
            crc,
            data,
        } => {
            packer.add_int(*tick);
            packer.add_int(tick.wrapping_sub(*delta_tick));
            packer.add_int(*crc);
            packer.add_int(data.len() as i32);
            packer.add_raw(data);
        }
        SysMsg::InputTiming { pred_tick, time_left } => {
            packer.add_int(*pred_tick);
            packer.add_int(*time_left);
        }
        SysMsg::RconAuthStatus {
            result,
            use_temp_rcon_commands,
        } => {
            packer.add_int(*result);
            packer.add_int(*use_temp_rcon_commands);
        }
        SysMsg::RconLine { line } => packer.add_string(line, 0, true),
        SysMsg::Input {
            ack_game_tick,
            pred_tick,
            size,
            data,
        } => {
            packer.add_int(*ack_game_tick);
            packer.add_int(*pred_tick);
            packer.add_int(*size);
            for &v in data {
                packer.add_int(v);
            }
        }
        SysMsg::RequestMapData { chunk } => packer.add_int(*chunk),
        SysMsg::RconCmdAdd { name, help, params } => {
            packer.add_string(name, 0, true);
            packer.add_string(help, 0, true);
            packer.add_string(params, 0, true);
        }
        SysMsg::RconCmdRem { name } => packer.add_string(name, 0, true),
        SysMsg::RconCmd { command } => packer.add_string(command, 0, true),
        SysMsg::RconAuth { name, password, one } => {
            packer.add_string(name, 0, true);
            packer.add_string(password, 0, true);
            packer.add_int(*one);
        }
        SysMsg::Unhandled { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(msg: SysMsg) {
        let mut buf = [0u8; 4096];
        let mut packer = Packer::new(&mut buf);
        encode(&msg, &mut packer);
        assert!(!packer.error());
        let mut unpacker = Unpacker::new(packer.data());
        let msg_id = match &msg {
            SysMsg::Info { .. } => id::INFO,
            SysMsg::MapChange { .. } => id::MAP_CHANGE,
            SysMsg::MapData { .. } => id::MAP_DATA,
            SysMsg::ConReady => id::CON_READY,
            SysMsg::Snap { .. } => id::SNAP,
            SysMsg::SnapEmpty { .. } => id::SNAPEMPTY,
            SysMsg::SnapSingle { .. } => id::SNAPSINGLE,
            SysMsg::SnapSmall { .. } => id::SNAPSMALL,
            SysMsg::InputTiming { .. } => id::INPUTTIMING,
            SysMsg::RconAuthStatus { .. } => id::RCON_AUTH_STATUS,
            SysMsg::RconLine { .. } => id::RCON_LINE,
            SysMsg::Ready => id::READY,
            SysMsg::EnterGame => id::ENTERGAME,
            SysMsg::Input { .. } => id::INPUT,
            SysMsg::RequestMapData { .. } => id::REQUEST_MAP_DATA,
            SysMsg::Ping => id::PING,
            SysMsg::PingReply => id::PING_REPLY,
            SysMsg::RconCmdAdd { .. } => id::RCON_CMD_ADD,
            SysMsg::RconCmdRem { .. } => id::RCON_CMD_REM,
            SysMsg::RconCmd { .. } => id::RCON_CMD,
            SysMsg::RconAuth { .. } => id::RCON_AUTH,
            SysMsg::Unhandled { msg_id, .. } => *msg_id,
        };
        assert_eq!(decode(msg_id, &mut unpacker), msg);
    }

    #[test]
    fn roundtrip_info() {
        roundtrip(SysMsg::Info {
            netversion: "0.6 626fce9a778df4d4".to_string(),
            password: "".to_string(),
        });
    }

    #[test]
    fn roundtrip_map_change() {
        roundtrip(SysMsg::MapChange {
            name: "Copy Love Box".to_string(),
            crc: 0x1234,
            size: 999,
        });
    }

    #[test]
    fn roundtrip_map_data() {
        roundtrip(SysMsg::MapData {
            last: 0,
            crc: 7,
            chunk: 3,
            data: vec![1, 2, 3, 4, 5],
        });
    }

    #[test]
    fn roundtrip_con_ready_ready_entergame_ping_pingreply() {
        roundtrip(SysMsg::ConReady);
        roundtrip(SysMsg::Ready);
        roundtrip(SysMsg::EnterGame);
        roundtrip(SysMsg::Ping);
        roundtrip(SysMsg::PingReply);
    }

    #[test]
    fn roundtrip_snap() {
        roundtrip(SysMsg::Snap {
            tick: 1000,
            delta_tick: 998,
            num_parts: 2,
            part: 0,
            crc: 42,
            data: vec![9, 9, 9],
        });
    }

    #[test]
    fn roundtrip_snapempty() {
        roundtrip(SysMsg::SnapEmpty {
            tick: 1000,
            delta_tick: -1,
        });
    }

    #[test]
    fn roundtrip_snapsingle() {
        roundtrip(SysMsg::SnapSingle {
            tick: 500,
            delta_tick: 498,
            crc: 1,
            data: vec![1],
        });
    }

    #[test]
    fn roundtrip_inputtiming() {
        roundtrip(SysMsg::InputTiming {
            pred_tick: 10,
            time_left: 20,
        });
    }

    #[test]
    fn roundtrip_rcon_auth_status() {
        roundtrip(SysMsg::RconAuthStatus {
            result: 1,
            use_temp_rcon_commands: 0,
        });
    }

    #[test]
    fn roundtrip_rcon_line() {
        roundtrip(SysMsg::RconLine {
            line: "hello".to_string(),
        });
    }

    #[test]
    fn roundtrip_input() {
        roundtrip(SysMsg::Input {
            ack_game_tick: -1,
            pred_tick: 5,
            size: 40,
            data: vec![0; 10],
        });
    }

    #[test]
    fn roundtrip_request_map_data() {
        roundtrip(SysMsg::RequestMapData { chunk: 3 });
    }

    #[test]
    fn roundtrip_rcon_cmd_add_rem() {
        roundtrip(SysMsg::RconCmdAdd {
            name: "ban".to_string(),
            help: "bans a player".to_string(),
            params: "s".to_string(),
        });
        roundtrip(SysMsg::RconCmdRem {
            name: "ban".to_string(),
        });
    }

    #[test]
    fn roundtrip_rcon_cmd_and_rcon_auth() {
        roundtrip(SysMsg::RconCmd {
            command: "status".to_string(),
        });
        roundtrip(SysMsg::RconAuth {
            name: "admin".to_string(),
            password: "secret".to_string(),
            one: 1,
        });
    }

    #[test]
    fn unknown_id_decodes_to_unhandled_not_panic() {
        let mut unpacker = Unpacker::new(&[1, 2, 3]);
        let msg = decode(9999, &mut unpacker);
        assert_eq!(
            msg,
            SysMsg::Unhandled {
                msg_id: 9999,
                raw: vec![1, 2, 3]
            }
        );
    }

    #[test]
    fn map_data_with_hostile_size_falls_back_to_unhandled_not_panic() {
        // A `size` far larger than any real packet could ever carry: must not attempt to
        // allocate/read that many bytes, must not panic — falls back to `Unhandled`.
        let mut buf = [0u8; 64];
        let mut packer = Packer::new(&mut buf);
        packer.add_int(0); // last
        packer.add_int(0); // crc
        packer.add_int(0); // chunk
        packer.add_int(i32::MAX); // size: hostile
        let mut unpacker = Unpacker::new(packer.data());
        let msg = decode(id::MAP_DATA, &mut unpacker);
        assert!(matches!(msg, SysMsg::Unhandled { msg_id, .. } if msg_id == id::MAP_DATA));
    }

    #[test]
    fn map_data_negative_size_falls_back_to_unhandled_not_panic() {
        let mut buf = [0u8; 64];
        let mut packer = Packer::new(&mut buf);
        packer.add_int(0);
        packer.add_int(0);
        packer.add_int(0);
        packer.add_int(-1);
        let mut unpacker = Unpacker::new(packer.data());
        let msg = decode(id::MAP_DATA, &mut unpacker);
        assert!(matches!(msg, SysMsg::Unhandled { msg_id, .. } if msg_id == id::MAP_DATA));
    }

    #[test]
    fn truncated_payload_never_panics() {
        for id in [
            id::INFO,
            id::MAP_CHANGE,
            id::MAP_DATA,
            id::SNAP,
            id::SNAPEMPTY,
            id::SNAPSINGLE,
            id::INPUTTIMING,
            id::RCON_AUTH_STATUS,
            id::RCON_LINE,
            id::INPUT,
            id::REQUEST_MAP_DATA,
            id::RCON_CMD_ADD,
            id::RCON_CMD_REM,
            id::RCON_CMD,
            id::RCON_AUTH,
        ] {
            for len in 0..8 {
                let bytes = vec![0xffu8; len];
                let mut unpacker = Unpacker::new(&bytes);
                let _ = decode(id, &mut unpacker); // must not panic
            }
        }
    }

    #[test]
    fn input_data_length_is_capped_regardless_of_claimed_size() {
        let mut buf = [0u8; 4096];
        let mut packer = Packer::new(&mut buf);
        packer.add_int(-1); // ack_game_tick
        packer.add_int(1); // pred_tick
        packer.add_int(i32::MAX); // size: hostile, would imply > 500M ints
        // Actually supply MAX_INPUT_INTS real ints, so decode succeeds — this proves the read is
        // capped at MAX_INPUT_INTS rather than trying to honour the hostile `size` field, without
        // also exercising truncated-input poisoning (covered separately below).
        for i in 0..MAX_INPUT_INTS {
            packer.add_int(i as i32);
        }
        assert!(!packer.error());
        let mut unpacker = Unpacker::new(packer.data());
        let msg = decode(id::INPUT, &mut unpacker);
        match msg {
            SysMsg::Input { data, .. } => {
                assert_eq!(data.len(), MAX_INPUT_INTS);
                assert_eq!(data, (0..MAX_INPUT_INTS as i32).collect::<Vec<_>>());
            }
            other => panic!("expected Input, got {other:?}"),
        }
    }

    #[test]
    fn input_truncated_shorter_than_claimed_size_degrades_to_unhandled_not_panic() {
        let mut buf = [0u8; 64];
        let mut packer = Packer::new(&mut buf);
        packer.add_int(-1);
        packer.add_int(1);
        packer.add_int(i32::MAX); // claims far more data than actually follows
        packer.add_int(7); // one real int, nowhere near enough
        let mut unpacker = Unpacker::new(packer.data());
        let msg = decode(id::INPUT, &mut unpacker); // must not panic
        assert!(matches!(msg, SysMsg::Unhandled { msg_id, .. } if msg_id == id::INPUT));
    }
}
