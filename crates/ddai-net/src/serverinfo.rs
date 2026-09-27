// Ported from DDNet `src/engine/shared/{masterserver,server}.cpp` (`SendServerInfo`,
// `CacheServerInfo` — the `SERVERINFO_VANILLA`/`_64_LEGACY`/`_EXTENDED` response formats, pinned
// rev c9d208138f85755521f16a0096b6fe036c5c8698, "20.1"), which carries the original Teeworlds
// zlib-style notice:
//
//   (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information.
//   If you are missing that file, acquire a complete release at teeworlds.com.
//
// This is an *altered* source version: rewritten in safe Rust, same byte layout, so it stays
// byte-for-byte compatible with what a real DDNet 20.x server sends. See docs/formats.md.
//
//! `SERVERINFO` — the connectionless server-browser response (task 2.2b acceptance criterion 2's
//! explicit mention of `SERVERINFO`). Unlike every other message in this crate, this one is not
//! carried by a connection at all: it rides DDNet's legacy connectionless packet framing
//! (`crate::packet::unpack_connless_packet`), *inside* which it has its own second magic — 4×
//! `0xFF` followed by a 4-ASCII-character tag (`SERVERBROWSE_INFO*`, `masterserver.cpp:3-10`) —
//! before the actual fields start. Every field after that magic is a NUL-terminated **string**,
//! including the ones that are logically integers (`ADD_INT` in the C++ reference is
//! `str_format` + `AddString`, never a packed int) — this module mirrors that exactly, parsing
//! those back with [`str::parse`].
//!
//! Only the two variants a client actually needs to ask a server "who's playing" —
//! [`Variant::Extended`] (what a DDNet 20.x server answers with, unprompted piggy-backing aside)
//! and [`Variant::Vanilla`] — are decoded with their per-client player list; `SERVERBROWSE_INFO_64_LEGACY`
//! and the `_EXTENDED_MORE` ("iex+") continuation-only packet (no header of its own, just more
//! players, `server.cpp:2517-2527`) are recognised (so callers can count/log them) but not further
//! decoded — the bot only ever needs this for a live server's current player count/names, which
//! the primary `_EXTENDED`/`_VANILLA` response already carries as far as this crate cares.

use crate::packer::{SanitizeMode, Unpacker};

/// Which `SERVERBROWSE_INFO*` response this is (`masterserver.cpp:3-10`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// `"inf3"` — the original 16-player format.
    Vanilla,
    /// `"dtsf"` — the 64-player format, paginated in groups of 24 (`server.cpp:2510-2537`).
    Legacy64,
    /// `"iext"` — DDNet's own extended format, effectively unbounded player count.
    Extended,
    /// `"iex+"` — a continuation of an `Extended` response that did not fit one packet; carries
    /// more players only, no header (not decoded further by this module, see the module docs).
    ExtendedMore,
}

/// One player/spectator/bot entry in a [`ServerInfo`] response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerInfoClient {
    pub name: String,
    pub clan: String,
    pub country: i32,
    pub score: i32,
    /// `true` for an actual player, `false` for a spectator (`CacheServerInfo`'s "is player?"
    /// field, `server.cpp`: `GameServer()->IsClientPlayer(i) ? 1 : 0`).
    pub is_player: bool,
}

/// A decoded `Variant::Vanilla`/`Extended` server-info response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerInfo {
    pub variant: Variant,
    pub token: i32,
    pub version: String,
    pub name: String,
    pub map: String,
    /// `Some` only for [`Variant::Extended`] (`SendServerInfo`'s `Type == SERVERINFO_EXTENDED`
    /// branch adds these two fields right after `map`).
    pub map_crc: Option<i32>,
    pub map_size: Option<i32>,
    pub game_type: String,
    pub flags: i32,
    pub num_players: i32,
    pub max_players: i32,
    pub num_clients: i32,
    pub max_clients: i32,
    /// Every client entry this response actually carried — may be fewer than `num_clients`
    /// claims (a truncated/multi-packet response, or simply a hostile server) and is never
    /// treated as an error: whatever parsed cleanly before the data ran out or a field failed to
    /// parse is returned as-is.
    pub clients: Vec<ServerInfoClient>,
}

const MAGIC_LEN: usize = 8;

/// Recognises which `SERVERBROWSE_INFO*` magic (if any) `data` starts with — `data` is the
/// connectionless packet's payload *after* `crate::packet::unpack_connless_packet` has already
/// stripped the outer 6-byte connectionless framing (see the module docs). Returns `None` if
/// `data` does not start with one of the four known magics at all (not a server-info response).
pub fn recognise(data: &[u8]) -> Option<Variant> {
    if data.len() < MAGIC_LEN || data[0..4] != [0xFF, 0xFF, 0xFF, 0xFF] {
        return None;
    }
    match &data[4..8] {
        b"inf3" => Some(Variant::Vanilla),
        b"dtsf" => Some(Variant::Legacy64),
        b"iext" => Some(Variant::Extended),
        b"iex+" => Some(Variant::ExtendedMore),
        _ => None,
    }
}

fn get_int_string(unpacker: &mut Unpacker) -> Option<i32> {
    let s = unpacker.get_string(SanitizeMode::NONE);
    if unpacker.error() {
        return None;
    }
    s.parse().ok()
}

fn get_plain_string(unpacker: &mut Unpacker) -> Option<String> {
    let s = unpacker.get_string(SanitizeMode::NONE);
    if unpacker.error() {
        return None;
    }
    Some(s)
}

/// Decodes a [`Variant::Vanilla`] or [`Variant::Extended`] response's fields (header +
/// however many client entries parsed cleanly before the data ran out) — `data` including the
/// 8-byte magic (see [`recognise`], which the caller should use first to route here only for
/// those two variants). Returns `None` only if a *header* field failed to decode (a `Legacy64`/
/// `ExtendedMore` payload passed in by mistake will almost always hit this, harmlessly); a
/// per-client field failing simply ends the client list early rather than failing outright — this
/// is metadata about other players on a server, not something whose absence should ever be
/// treated as an error by a caller.
pub fn decode(data: &[u8]) -> Option<ServerInfo> {
    let variant = recognise(data)?;
    if !matches!(variant, Variant::Vanilla | Variant::Extended) {
        return None;
    }
    let extended = variant == Variant::Extended;
    let mut unpacker = Unpacker::new(&data[MAGIC_LEN..]);

    let token = get_int_string(&mut unpacker)?;
    let version = get_plain_string(&mut unpacker)?;
    let name = get_plain_string(&mut unpacker)?;
    let map = get_plain_string(&mut unpacker)?;
    let (map_crc, map_size) = if extended {
        (
            Some(get_int_string(&mut unpacker)?),
            Some(get_int_string(&mut unpacker)?),
        )
    } else {
        (None, None)
    };
    let game_type = get_plain_string(&mut unpacker)?;
    let flags = get_int_string(&mut unpacker)?;
    let num_players = get_int_string(&mut unpacker)?;
    let max_players = get_int_string(&mut unpacker)?;
    let num_clients = get_int_string(&mut unpacker)?;
    let max_clients = get_int_string(&mut unpacker)?;
    if extended {
        get_plain_string(&mut unpacker)?; // "extra info, reserved" — always empty, discarded
    }

    let mut clients = Vec::new();
    loop {
        if unpacker.remaining().is_empty() {
            break;
        }
        let Some(client_name) = get_plain_string(&mut unpacker) else {
            break;
        };
        let Some(clan) = get_plain_string(&mut unpacker) else {
            break;
        };
        let Some(country) = get_int_string(&mut unpacker) else {
            break;
        };
        let Some(score) = get_int_string(&mut unpacker) else {
            break;
        };
        let Some(is_player) = get_int_string(&mut unpacker) else {
            break;
        };
        if extended && get_plain_string(&mut unpacker).is_none() {
            break; // per-client "extra info, reserved"
        }
        clients.push(ServerInfoClient {
            name: client_name,
            clan,
            country,
            score,
            is_player: is_player != 0,
        });
    }

    Some(ServerInfo {
        variant,
        token,
        version,
        name,
        map,
        map_crc,
        map_size,
        game_type,
        flags,
        num_players,
        max_players,
        num_clients,
        max_clients,
        clients,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packer::Packer;

    fn build_extended(num_clients: usize) -> Vec<u8> {
        let mut buf = [0u8; 4096];
        let mut p = Packer::new(&mut buf);
        p.add_raw(&[0xFF, 0xFF, 0xFF, 0xFF]);
        p.add_raw(b"iext");
        for s in [
            "12345",
            "0.6 626fce9a778df4d4",
            "Test Server",
            "Copy Love Box",
            "0",
            "1000",
        ] {
            p.add_string(s, 0, true);
        }
        p.add_string("DDraceNetwork", 0, true); // gametype
        for s in ["0", "3", "64", &num_clients.to_string(), "64"] {
            p.add_string(s, 0, true);
        }
        p.add_string("", 0, true); // reserved
        for i in 0..num_clients {
            p.add_string(&format!("Player{i}"), 0, true);
            p.add_string("", 0, true);
            p.add_string("-1", 0, true);
            p.add_string("0", 0, true);
            p.add_string("1", 0, true);
            p.add_string("", 0, true);
        }
        p.data().to_vec()
    }

    #[test]
    fn recognise_all_four_magics() {
        assert_eq!(recognise(b"\xff\xff\xff\xffinf3rest"), Some(Variant::Vanilla));
        assert_eq!(recognise(b"\xff\xff\xff\xffdtsfrest"), Some(Variant::Legacy64));
        assert_eq!(recognise(b"\xff\xff\xff\xffiextrest"), Some(Variant::Extended));
        assert_eq!(recognise(b"\xff\xff\xff\xffiex+rest"), Some(Variant::ExtendedMore));
        assert_eq!(recognise(b"not a server info packet"), None);
        assert_eq!(recognise(b"\xff\xff\xff\xff"), None); // too short for even the magic
        assert_eq!(recognise(b""), None);
    }

    #[test]
    fn decode_extended_with_three_clients() {
        let data = build_extended(3);
        let info = decode(&data).expect("well-formed extended response");
        assert_eq!(info.variant, Variant::Extended);
        assert_eq!(info.token, 12345);
        assert_eq!(info.name, "Test Server");
        assert_eq!(info.map, "Copy Love Box");
        assert_eq!(info.map_crc, Some(0));
        assert_eq!(info.map_size, Some(1000));
        assert_eq!(info.game_type, "DDraceNetwork");
        assert_eq!(info.num_clients, 3);
        assert_eq!(info.clients.len(), 3);
        assert_eq!(info.clients[1].name, "Player1");
        assert!(info.clients[1].is_player);
    }

    #[test]
    fn decode_extended_with_zero_clients() {
        let data = build_extended(0);
        let info = decode(&data).unwrap();
        assert!(info.clients.is_empty());
    }

    #[test]
    fn truncated_client_list_yields_partial_list_not_none() {
        let mut data = build_extended(2);
        data.truncate(data.len() - 3); // cut off mid-way through the last client's fields
        let info = decode(&data).expect("header is intact, only the client list is truncated");
        assert_eq!(
            info.clients.len(),
            1,
            "the incomplete trailing client must be dropped, not error out"
        );
    }

    #[test]
    fn truncated_header_yields_none_not_panic() {
        let data = build_extended(1);
        for len in 0..20 {
            let _ = decode(&data[..len]); // must not panic, whatever it returns
        }
    }

    #[test]
    fn legacy64_and_extended_more_are_recognised_but_not_decoded() {
        assert_eq!(decode(b"\xff\xff\xff\xffdtsfrest"), None);
        assert_eq!(decode(b"\xff\xff\xff\xffiex+rest"), None);
    }

    #[test]
    fn garbage_never_panics() {
        for len in 0..40 {
            let data = vec![0xffu8; len];
            let _ = decode(&data);
        }
    }
}
