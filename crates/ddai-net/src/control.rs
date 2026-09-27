// Ported from DDNet `src/engine/shared/{network,network_conn,network_client}.{h,cpp}` (pinned
// rev c9d208138f85755521f16a0096b6fe036c5c8698, "20.1"), which carries the original Teeworlds
// zlib-style notice:
//
//   (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information.
//   If you are missing that file, acquire a complete release at teeworlds.com.
//
// This is an *altered* source version: rewritten in safe Rust, same control-message byte layouts
// and TKEN handshake sequencing, so it stays byte-for-byte compatible with the wire format every
// DDNet 0.6+DDNet client/server speaks. See docs/formats.md for the byte layout.
//
//! Control messages and the DDNet `TKEN` security-token handshake — pure byte-layout encode/
//! decode, no connection state. [`crate::conn::Connection`] is what actually drives the
//! handshake; this module only knows how to build and parse the five control-message payloads.
//!
//! # Byte layouts (see also `docs/formats.md`)
//!
//! A control message is a connection-oriented packet with [`crate::packet::packet_flags::CONTROL`]
//! set, zero chunks, and a payload of `[ctrl_msg::* ] ++ extra`, to which the packet layer
//! appends the trailing 4-byte security token whenever one is being negotiated (see
//! `crate::packet::build_packet`'s docs) — *even on these control packets themselves*
//! (`network.cpp:217-223`: the token is appended to every non-connless packet, control messages
//! included).
//!
//! ```text
//! KEEPALIVE:      [00]
//! CONNECT:        [01] 'T' 'K' 'E' 'N'    (+ trailing token, initially unknown, 0xffffffff)
//! CONNECTACCEPT:  [02] 'T' 'K' 'E' 'N'    (+ trailing token = the token being handed to the client)
//! ACCEPT:         [03]                    (+ trailing token, now known)
//! CLOSE:          [04] <reason:NUL-terminated string>?  (+ trailing token, if known)
//! ```
//!
//! DDNet's client/server TKEN handshake (`network_conn.cpp:204-212,356-478`;
//! `network_server.cpp:513-522`), the 0.6+DDNet path only (no sixup/vanilla-0.6.5 anti-spoof):
//!
//! ```text
//! C→S  CONTROL CONNECT       ["TKEN"]     (+ trailing token = UNKNOWN, 0xffffffff)
//! S→C  CONTROL CONNECTACCEPT ["TKEN"]     (+ trailing token = the server's freshly generated token)
//! C→S  CONTROL ACCEPT        []           (+ trailing token = that same token)
//! C→S  … every further packet …           (+ trailing token = that same token)
//! ```
//!
//! **The token is carried exactly once**, as the ordinary trailing token every packet gets once
//! a connection has one (`network_server.cpp:519`: `SendControl(..., SECURITY_TOKEN_MAGIC, 4,
//! Token)` — only the 4-byte *magic* is passed as message data; `Token` is the `SendPacket`
//! `SecurityToken` argument, i.e. the trailing copy). There is no *second*, data-embedded copy —
//! an earlier draft of this module (and of `docs/research/ddnet-protocol.md`) assumed DDNet
//! duplicated it, which a real capture against the local DDNet 20.1 server disproved (see
//! `tests/capture.rs`): the captured `CONNECTACCEPT` datagram is exactly `3 + 1 + 4 + 4 = 12`
//! bytes, not `3 + 1 + 4 + 4 + 4 = 16`.
//!
//! The client can read this trailing token as if it were message data *only* because, while its
//! own `security_token` is still `Unknown`, [`crate::conn::Connection::feed`] does not yet
//! strip/verify a trailing token (see that module) — so the bytes right after `"TKEN"` are, from
//! the client's point of view, indistinguishable data-vs-trailer, and
//! [`crate::conn::Connection`] is what actually reads the numeric token value out of them (this
//! module's [`decode`] only reports whether the magic is present, via
//! [`ControlMsg::ConnectAccept`]'s `has_tken_magic`, deliberately *not* a token value — there
//! usually isn't one to read as "message data" once the connection already knows its token and
//! the trailing bytes were stripped before `decode` ever saw them).

/// Control message type ids (`NET_CTRLMSG_*`, `network.h:97-101`).
pub mod ctrl_msg {
    pub const KEEPALIVE: u8 = 0;
    pub const CONNECT: u8 = 1;
    pub const CONNECTACCEPT: u8 = 2;
    pub const ACCEPT: u8 = 3;
    pub const CLOSE: u8 = 4;
}

/// `SECURITY_TOKEN_MAGIC` (`network.cpp:19`).
pub const SECURITY_TOKEN_MAGIC: [u8; 4] = crate::packet::SECURITY_TOKEN_MAGIC;

/// Sentinel security-token wire values (`NET_SECURITY_TOKEN_UNKNOWN`/`_UNSUPPORTED`,
/// `network.h:129-133`); see [`crate::conn::SecurityToken`] for the connection-state-aware
/// wrapper that decides which of these applies when.
pub const TOKEN_UNKNOWN: u32 = 0xffff_ffff;
pub const TOKEN_UNSUPPORTED: u32 = 0;

/// A decoded control message payload (everything after the packet header, before any trailing
/// security token — the caller is expected to have already stripped/verified that separately, or
/// to know it is not present yet, exactly like `CNetConnection::Feed` does before dispatching on
/// `pPacket->m_aChunkData[0]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlMsg {
    KeepAlive,
    /// `extra` is normally exactly [`SECURITY_TOKEN_MAGIC`] for a DDNet-aware peer, but is kept
    /// as raw bytes since a vanilla (non-DDNet) 0.6 peer sends a bare `[01]` with no magic at
    /// all — out of scope for us to *speak* (constraints: DDNet servers only), but not something
    /// [`decode`] should panic or error out on.
    Connect {
        extra: Vec<u8>,
    },
    /// Whether the payload carried the `TKEN` magic right after the message type byte — a bare
    /// vanilla (non-DDNet) `CONNECTACCEPT` has `has_tken_magic: false`. There is no token *value*
    /// here to decode: see the module docs for why, and [`crate::conn::Connection`] for where the
    /// numeric token is actually read from a fresh `CONNECTACCEPT` while still negotiating one.
    ConnectAccept {
        has_tken_magic: bool,
    },
    Accept,
    /// `reason` is the sanitised, UTF-8-checked close reason, or `None` if the payload carried
    /// none (`network_conn.cpp:411-421`; on invalid UTF-8 DDNet substitutes a fixed message
    /// rather than rejecting the whole `CLOSE` — mirrored here as `Some("(Invalid error
    /// message)".to_string())`).
    Close {
        reason: Option<String>,
    },
}

/// Reason [`decode`] rejected a control payload outright (as opposed to falling back to a
/// tolerant default, like an unrecognised `CONNECTACCEPT` extra payload).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ControlDecodeError {
    #[error("empty control payload (missing the message-type byte)")]
    Empty,
    #[error("unrecognised control message type byte")]
    UnknownType,
}

/// `str_utf8_fix_truncation` (`base/str.cpp`, called from `str_copy` after every hard-length
/// truncation): backs `bytes` off to the last complete UTF-8 character boundary, by trimming a
/// trailing multi-byte sequence that a fixed byte cap cut short. Never panics; if `bytes` is not
/// valid UTF-8 for reasons *unrelated* to truncation (already-malformed input), this may still
/// trim a byte or two off the very end — matching DDNet's own unconditional call, which has no way
/// to tell "truncated" apart from "already invalid" either — the caller's own UTF-8 validation
/// still catches whatever is left over.
fn fix_utf8_truncation(bytes: &[u8]) -> &[u8] {
    let len = bytes.len();
    let max_back = len.min(4); // a complete UTF-8 sequence is at most 4 bytes long
    for back in 1..=max_back {
        let b = bytes[len - back];
        if b & 0xC0 == 0x80 {
            // A continuation byte: keep scanning backwards for the lead byte that started this
            // sequence, unless we have already scanned as far as we usefully can.
            if back == max_back {
                return bytes;
            }
            continue;
        }
        let expected_len: usize = if b < 0x80 {
            1 // ASCII
        } else if b & 0xE0 == 0xC0 {
            2
        } else if b & 0xF0 == 0xE0 {
            3
        } else if b & 0xF8 == 0xF0 {
            4
        } else {
            1 // not a valid UTF-8 lead byte either way; leave validation to the caller
        };
        return if expected_len > back {
            &bytes[..len - back]
        } else {
            bytes
        };
    }
    bytes
}

/// Decodes a control message payload (`pPacket->m_aChunkData[0..]` in the C++ reference, i.e.
/// *without* any trailing security token still attached — strip that first). Never panics; a
/// payload that is merely too short for what a message type would ideally carry decodes with
/// tolerant defaults (`None`/empty) rather than erroring, matching how permissive
/// `CNetConnection::Feed` is about e.g. a `CLOSE` with no reason string.
pub fn decode(payload: &[u8]) -> Result<ControlMsg, ControlDecodeError> {
    let &[msg_type, ref rest @ ..] = payload else {
        return Err(ControlDecodeError::Empty);
    };
    match msg_type {
        ctrl_msg::KEEPALIVE => Ok(ControlMsg::KeepAlive),
        ctrl_msg::CONNECT => Ok(ControlMsg::Connect { extra: rest.to_vec() }),
        ctrl_msg::CONNECTACCEPT => {
            let has_tken_magic =
                rest.len() >= SECURITY_TOKEN_MAGIC.len() && rest[..SECURITY_TOKEN_MAGIC.len()] == SECURITY_TOKEN_MAGIC;
            Ok(ControlMsg::ConnectAccept { has_tken_magic })
        }
        ctrl_msg::ACCEPT => Ok(ControlMsg::Accept),
        ctrl_msg::CLOSE => {
            if rest.is_empty() {
                return Ok(ControlMsg::Close { reason: None });
            }
            // `network_conn.cpp:411-421`: copies into a fixed `char aStr[256]` (`str_copy`, so at
            // most 255 bytes survive before the NUL DDNet itself adds), NUL-terminates, sanitizes
            // control characters (`SANITIZE_CC`), and substitutes a fixed message if the result
            // is not valid UTF-8.
            const MAX_REASON_BYTES: usize = 255;
            let capped = &rest[..rest.len().min(MAX_REASON_BYTES)];
            let raw = match capped.iter().position(|&b| b == 0) {
                Some(nul) => &capped[..nul],
                None => capped,
            };
            // F8 (2.2a review carry-over): `str_copy`'s `MAX_REASON_BYTES` cap can land in the
            // middle of a multi-byte UTF-8 character (e.g. a 254-byte name plus one Cyrillic
            // character straddling the cut). DDNet's `str_copy` always calls
            // `str_utf8_fix_truncation` after truncating, backing off to the last complete
            // character boundary so the *valid* prefix survives — without this, a long multibyte
            // reason decodes as invalid UTF-8 as a whole and gets replaced by the fixed fallback
            // message below, even though only the last character was ever in question.
            let raw = fix_utf8_truncation(raw);
            let mut sanitized: Vec<u8> = raw.iter().map(|&b| if b < 32 { b' ' } else { b }).collect();
            let reason = match String::from_utf8(std::mem::take(&mut sanitized)) {
                Ok(s) => s,
                Err(_) => "(Invalid error message)".to_string(),
            };
            Ok(ControlMsg::Close { reason: Some(reason) })
        }
        _ => Err(ControlDecodeError::UnknownType),
    }
}

/// Encodes a control message's payload (everything after the packet header; the caller still
/// has to build the surrounding packet and append the trailing security token, if any — see
/// `crate::packet::build_packet`).
pub fn encode(msg: &ControlMsg) -> Vec<u8> {
    match msg {
        ControlMsg::KeepAlive => vec![ctrl_msg::KEEPALIVE],
        ControlMsg::Connect { extra } => {
            let mut out = vec![ctrl_msg::CONNECT];
            out.extend_from_slice(extra);
            out
        }
        ControlMsg::ConnectAccept { has_tken_magic } => {
            let mut out = vec![ctrl_msg::CONNECTACCEPT];
            if *has_tken_magic {
                out.extend_from_slice(&SECURITY_TOKEN_MAGIC);
            }
            out
        }
        ControlMsg::Accept => vec![ctrl_msg::ACCEPT],
        ControlMsg::Close { reason } => {
            let mut out = vec![ctrl_msg::CLOSE];
            if let Some(reason) = reason {
                out.extend_from_slice(reason.as_bytes());
                out.push(0);
            }
            out
        }
    }
}

/// Builds the DDNet-aware `CONNECT` payload: `[01, 'T', 'K', 'E', 'N']`
/// (`network_conn.cpp:204-212`: `SendControl(..., SECURITY_TOKEN_MAGIC, 4, ...)`).
pub fn connect_payload() -> ControlMsg {
    ControlMsg::Connect {
        extra: SECURITY_TOKEN_MAGIC.to_vec(),
    }
}

/// Builds the DDNet-aware `CONNECTACCEPT` message payload: `[02, 'T', 'K', 'E', 'N']`
/// (`network_server.cpp:519`: `SendControl(CONNECTACCEPT, SECURITY_TOKEN_MAGIC, 4, Token)` — only
/// the magic is passed as message data; the caller is responsible for passing `Some(token)` as
/// `crate::packet::build_packet`'s trailing-security-token argument so `token` actually reaches
/// the wire — see the module docs for why there is no token parameter here).
pub fn connect_accept_payload() -> ControlMsg {
    ControlMsg::ConnectAccept { has_tken_magic: true }
}

/// Whether `extra` (the bytes after the `CONNECT` message-type byte) carries the DDNet TKEN
/// magic, i.e. whether this is a DDNet-aware `CONNECT` as opposed to a bare vanilla-0.6 one
/// (`network_server.cpp:597-614`'s `data[1..5] == "TKEN"` check — we only ever speak the DDNet
/// path, see the module docs, but recognising the vanilla shape lets a caller reject/ignore it
/// cleanly instead of misreading garbage as a token).
pub fn is_ddnet_connect(extra: &[u8]) -> bool {
    extra.len() >= SECURITY_TOKEN_MAGIC.len() && extra[..SECURITY_TOKEN_MAGIC.len()] == SECURITY_TOKEN_MAGIC
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keepalive_roundtrip() {
        let msg = ControlMsg::KeepAlive;
        let bytes = encode(&msg);
        assert_eq!(bytes, [ctrl_msg::KEEPALIVE]);
        assert_eq!(decode(&bytes).unwrap(), msg);
    }

    #[test]
    fn connect_roundtrip_carries_tken_magic() {
        let msg = connect_payload();
        let bytes = encode(&msg);
        assert_eq!(bytes, [ctrl_msg::CONNECT, b'T', b'K', b'E', b'N']);
        assert_eq!(decode(&bytes).unwrap(), msg);
        match decode(&bytes).unwrap() {
            ControlMsg::Connect { extra } => assert!(is_ddnet_connect(&extra)),
            _ => panic!("expected Connect"),
        }
    }

    #[test]
    fn connect_accept_roundtrip_carries_tken_magic() {
        // Matches a real captured DDNet 20.1 `CONNECTACCEPT`: header(3) + this(5) + trailing
        // token(4) = 12 bytes on the wire, not 16 — see the module docs.
        let msg = connect_accept_payload();
        let bytes = encode(&msg);
        assert_eq!(bytes, [ctrl_msg::CONNECTACCEPT, b'T', b'K', b'E', b'N']);
        assert_eq!(decode(&bytes).unwrap(), msg);
    }

    #[test]
    fn connect_accept_without_magic_decodes_false() {
        // A vanilla (non-DDNet) CONNECTACCEPT: no TKEN magic.
        let bytes = [ctrl_msg::CONNECTACCEPT];
        assert_eq!(
            decode(&bytes).unwrap(),
            ControlMsg::ConnectAccept { has_tken_magic: false }
        );
    }

    #[test]
    fn connect_accept_truncated_magic_decodes_false() {
        let mut bytes = vec![ctrl_msg::CONNECTACCEPT];
        bytes.extend_from_slice(&SECURITY_TOKEN_MAGIC[..2]);
        assert_eq!(
            decode(&bytes).unwrap(),
            ControlMsg::ConnectAccept { has_tken_magic: false }
        );
    }

    #[test]
    fn connect_accept_magic_with_trailing_bytes_still_decodes_true() {
        // While the connection's security token is still Unknown, `Connection::feed` does not
        // strip a trailing token before calling `decode` — so `decode` must tolerate (and ignore)
        // extra bytes after the magic, exactly like real captured traffic has.
        let mut bytes = vec![ctrl_msg::CONNECTACCEPT];
        bytes.extend_from_slice(&SECURITY_TOKEN_MAGIC);
        bytes.extend_from_slice(&[0xaa, 0xbb, 0xcc, 0xdd]);
        assert_eq!(
            decode(&bytes).unwrap(),
            ControlMsg::ConnectAccept { has_tken_magic: true }
        );
    }

    #[test]
    fn accept_roundtrip() {
        let msg = ControlMsg::Accept;
        let bytes = encode(&msg);
        assert_eq!(bytes, [ctrl_msg::ACCEPT]);
        assert_eq!(decode(&bytes).unwrap(), msg);
    }

    #[test]
    fn close_roundtrip_with_reason() {
        let msg = ControlMsg::Close {
            reason: Some("bye".to_string()),
        };
        let bytes = encode(&msg);
        assert_eq!(bytes, [ctrl_msg::CLOSE, b'b', b'y', b'e', 0]);
        assert_eq!(decode(&bytes).unwrap(), msg);
    }

    #[test]
    fn close_roundtrip_without_reason() {
        let msg = ControlMsg::Close { reason: None };
        let bytes = encode(&msg);
        assert_eq!(bytes, [ctrl_msg::CLOSE]);
        assert_eq!(decode(&bytes).unwrap(), msg);
    }

    #[test]
    fn close_sanitizes_control_characters_in_reason() {
        let bytes = [ctrl_msg::CLOSE, b'a', 0x01, b'b', 0];
        assert_eq!(
            decode(&bytes).unwrap(),
            ControlMsg::Close {
                reason: Some("a b".to_string())
            }
        );
    }

    #[test]
    fn close_substitutes_fixed_message_on_invalid_utf8() {
        let bytes = [ctrl_msg::CLOSE, 0x80, 0x80, 0];
        assert_eq!(
            decode(&bytes).unwrap(),
            ControlMsg::Close {
                reason: Some("(Invalid error message)".to_string())
            }
        );
    }

    #[test]
    fn close_reason_backs_off_to_last_complete_char_on_multibyte_truncation() {
        // F8 (2.2a review carry-over): 254 × 'a' + "ä" (2-byte UTF-8) is 256 bytes, one over the
        // 255-byte cap — the cap lands exactly on "ä"'s lead byte (0xC3), which must be trimmed
        // off entirely (backing off to the last complete character) rather than surviving as a
        // dangling lead byte that turns the *whole* reason into "(Invalid error message)".
        let mut bytes = vec![ctrl_msg::CLOSE];
        bytes.extend(std::iter::repeat_n(b'a', 254));
        bytes.extend("ä".as_bytes()); // 0xC3 0xA4
        let ControlMsg::Close { reason } = decode(&bytes).unwrap() else {
            panic!("expected Close");
        };
        let reason = reason.unwrap();
        assert_eq!(reason, "a".repeat(254));
        assert_eq!(reason.len(), 254);
    }

    #[test]
    fn fix_utf8_truncation_backs_off_various_multibyte_widths() {
        // 3-byte sequence (e.g. '€' = E2 82 AC) cut after 1 or 2 bytes.
        let three_byte = "€".as_bytes();
        assert_eq!(fix_utf8_truncation(&three_byte[..1]), b"");
        assert_eq!(fix_utf8_truncation(&three_byte[..2]), b"");
        assert_eq!(fix_utf8_truncation(three_byte), three_byte);

        // 4-byte sequence (e.g. '𝄞' U+1D11E) cut after 1..3 bytes.
        let four_byte = "𝄞".as_bytes();
        assert_eq!(four_byte.len(), 4);
        for cut in 1..4 {
            assert_eq!(fix_utf8_truncation(&four_byte[..cut]), b"");
        }
        assert_eq!(fix_utf8_truncation(four_byte), four_byte);

        // A complete character followed by a truncated one: only the truncated tail is dropped.
        let mixed = "a€".as_bytes(); // 'a' + first byte of '€' only, if we cut it
        assert_eq!(fix_utf8_truncation(&mixed[..2]), b"a");

        // Plain ASCII, no multibyte anywhere: never touched.
        assert_eq!(fix_utf8_truncation(b"hello"), b"hello");

        // Empty input: must not panic (no bytes to look at).
        assert_eq!(fix_utf8_truncation(b""), b"");
    }

    #[test]
    fn close_without_nul_terminator_still_decodes() {
        // Malformed (no NUL), but must not panic — takes the whole remainder as the reason.
        let bytes = [ctrl_msg::CLOSE, b'h', b'i'];
        assert_eq!(
            decode(&bytes).unwrap(),
            ControlMsg::Close {
                reason: Some("hi".to_string())
            }
        );
    }

    #[test]
    fn close_reason_capped_at_255_bytes_like_reference_char_astr_256() {
        // `network_conn.cpp:411`: `char aStr[256]`, filled via `str_copy` (at most 255 bytes
        // survive before the NUL DDNet's own code adds) — no NUL in our input at all here, so
        // without the cap the whole 500-byte remainder would become the reason.
        let mut bytes = vec![ctrl_msg::CLOSE];
        bytes.extend(std::iter::repeat_n(b'a', 500));
        let ControlMsg::Close { reason } = decode(&bytes).unwrap() else {
            panic!("expected Close");
        };
        let reason = reason.unwrap();
        assert_eq!(reason.len(), 255);
        assert!(reason.chars().all(|c| c == 'a'));
    }

    #[test]
    fn close_reason_cap_and_nul_terminator_compose_correctly() {
        // A NUL within the first 255 bytes still ends the reason there, same as without the cap.
        let mut bytes = vec![ctrl_msg::CLOSE];
        bytes.extend(std::iter::repeat_n(b'a', 10));
        bytes.push(0);
        bytes.extend(std::iter::repeat_n(b'b', 500)); // must never leak into the reason
        assert_eq!(
            decode(&bytes).unwrap(),
            ControlMsg::Close {
                reason: Some("a".repeat(10))
            }
        );
    }

    #[test]
    fn decode_empty_payload_errors() {
        assert_eq!(decode(&[]).unwrap_err(), ControlDecodeError::Empty);
    }

    #[test]
    fn decode_unknown_type_errors() {
        assert_eq!(decode(&[0xEE]).unwrap_err(), ControlDecodeError::UnknownType);
    }

    #[test]
    fn is_ddnet_connect_rejects_vanilla_connect() {
        assert!(!is_ddnet_connect(&[]));
        assert!(!is_ddnet_connect(b"XXXX"));
        assert!(is_ddnet_connect(b"TKEN"));
        assert!(is_ddnet_connect(b"TKENextra"));
    }
}
