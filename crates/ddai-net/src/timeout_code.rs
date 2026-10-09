//! The third chat-channel message the bot may ever send: DDNet's timeout-protection command `/timeout <code>` (task 4.10, D-100).
//!
//! D-007 stands for everything automatic. The owner allowed three kinds of `Cl_Say`: the typed `/kill` fallback
//! ([`crate::server_command::ServerCommand`], D-078), a line the owner typed on the website ([`crate::owner_chat`], D-094), and this
//! one (2026-10-06): the stock timeout code, sent **once per join** exactly as the official DDNet 20.1 client sends it, so that a
//! reconnect from a new address (a SOCKS5 association that was lost and re-made) takes the old tee back instead of joining as
//! `(1)Name` next to a ghost (the server prefixes a duplicate name).
//!
//! **What the official client does** (`engine/client/client.cpp`, 20.1), and what is ported here:
//!
//! - `CClient::Connect` calls `GenerateTimeoutCodes(addrs)`: the code is `generate_password` over the 16 bytes of
//!   `MD5("normal\0" ++ seed ++ "\0" ++ raw NETADDR bytes of every connect address)`, read as eight native-endian `u16`
//!   ([`TimeoutCode::derive`]). The seed is `cl_timeout_seed`, 16 characters of `secure_random_password`, made once and kept in the
//!   client's config ([`TimeoutSeed`]; the bot keeps it in a 0600 file in its data dir, never logged).
//! - `CClient::OnPostConnect` runs when more than `GameTickSpeed()` (50) snapshots arrived since `EnterGame` and the server announced the
//!   `CHATTIMEOUTCODE` capability; it sends `Cl_Say{team 0, "/timeout <code>"}` ([`TimeoutCommand`]). The session (`ddai-client`) does the same.
//!
//! **What the server does** (`game/server/ddracechat.cpp` `ConTimeout`, `engine/server/server.cpp` `SetTimedOut`): `/timeout <code>`
//! looks for another player with the same code whose connection is *already in the timeout state* (`HasErrored`: the server has not
//! heard from it for `conn_timeout` seconds, 100 by default); if there is one, the new connection takes over that player's slot (name,
//! tee, state) and the new connection's own player is dropped. If the old connection has **not** timed out yet, nothing is taken over:
//! the new client is only marked timeout-protected under the code. So the takeover works for a reconnect made after the server has
//! noticed the old connection is dead; see `docs/DECISIONS.md` D-100 for what that means for a fast reconnect.
//!
//! **The guarantee is in the types, like `/kill`'s.** A [`TimeoutCode`] is made only by [`TimeoutCode::derive`] (from a seed and a
//! server address), a [`TimeoutCommand`] only from a code, and a [`TimeoutPayload`] only by [`TimeoutCommand::payload`]. There is no
//! function that makes a `Cl_Say` from a string: `encode_cl_say` stays `pub(crate)`. The outgoing allow-list (`ddai-client::allowlist`)
//! lets this `Cl_Say` through only against a one-shot authorisation for the exact bytes **and** only when they are structurally
//! `/timeout ` plus 16 characters of DDNet's password alphabet ([`TimeoutCommand::is_canonical`]). The owner chat refuses every `/` line that
//! mentions `timeout` (`OwnerText`, [`crate::owner_chat`]), so the website cannot send `/timeout` with another code (or none:
//! `/timeout` without an argument matches any timed-out player whose code is empty).
//!
//! The code is a secret of the same kind as the seed: whoever knows it can take over the tee while the old connection is timed out.
//! `Debug` shows only the length; the only thing logged is `timeout code sent (len N)`.
//!
//! ```compile_fail,E0603
//! // `TimeoutCode` has a private field: a code cannot be made from a string, only derived from a seed and an address.
//! let _ = ddai_net::timeout_code::TimeoutCode([b'A'; 16]);
//! ```
//!
//! ```compile_fail,E0603
//! // `TimeoutPayload` has a private field: an authorisation cannot be granted for hand-built bytes.
//! let _ = ddai_net::timeout_code::TimeoutPayload(vec![0u8; 4]);
//! ```

use std::fmt;
use std::net::{IpAddr, SocketAddr};

use md5::{Digest, Md5};

use crate::generated::messages::{self as msgs, ClSay};
use crate::packer::{Packer, Unpacker};
use crate::uuid::{MsgId, pack_msg_id};

/// DDNet's `generate_password` alphabet (`base/secure.cpp`): 46 characters without the look-alikes (`I`, `O`, `Q`, `i`, `l`, `1`, `0`, ...).
const VALUES: &[u8; 46] = b"ABCDEFGHKLMNPRSTUVWXYZabcdefghjkmnopqt23456789";

/// Length of a timeout code and of the seed: 8 `u16` values, two characters each.
pub const CODE_LEN: usize = 16;
/// Length of the seed (`secure_random_password(cl_timeout_seed, .., 16)`).
pub const SEED_LEN: usize = 16;

/// The chat command word and its argument separator, exactly as `OnPostConnect` formats them (`"/timeout %s"`).
const PREFIX: &str = "/timeout ";

/// `generate_password` for eight random `u16` values: `random % 2048`, then two alphabet characters (`/ 46`, `% 46`).
fn generate_password(random: &[u16; 8]) -> [u8; CODE_LEN] {
    let mut out = [0u8; CODE_LEN];
    for (i, r) in random.iter().enumerate() {
        let n = usize::from(r % 2048);
        out[2 * i] = VALUES[n / VALUES.len()];
        out[2 * i + 1] = VALUES[n % VALUES.len()];
    }
    out
}

fn is_alphabet(bytes: &[u8]) -> bool {
    bytes.iter().all(|b| VALUES.contains(b))
}

/// The persistent random seed (`cl_timeout_seed`): 16 characters of DDNet's password alphabet. A secret: `Debug` hides it and there is
/// no `Display`; [`TimeoutSeed::as_str`] is for writing the seed to its file only.
#[derive(Clone, PartialEq, Eq)]
pub struct TimeoutSeed([u8; SEED_LEN]);

impl TimeoutSeed {
    /// `secure_random_password(.., 16)`: the 16 bytes of OS randomness, read as eight native-endian (little-endian) `u16`, through
    /// `generate_password`. The caller supplies the randomness (the bot reads `/dev/urandom`), which keeps this crate free of I/O.
    pub fn from_random(random: [u8; SEED_LEN]) -> TimeoutSeed {
        let mut shorts = [0u16; 8];
        for (i, s) in shorts.iter_mut().enumerate() {
            *s = u16::from_le_bytes([random[2 * i], random[2 * i + 1]]);
        }
        TimeoutSeed(generate_password(&shorts))
    }

    /// A seed read back from its file (surrounding whitespace such as a final newline is ignored). `None` unless it is exactly
    /// [`SEED_LEN`] characters of the alphabet.
    pub fn parse(text: &str) -> Option<TimeoutSeed> {
        let bytes = text.trim().as_bytes();
        if bytes.len() != SEED_LEN || !is_alphabet(bytes) {
            return None;
        }
        let mut seed = [0u8; SEED_LEN];
        seed.copy_from_slice(bytes);
        Some(TimeoutSeed(seed))
    }

    /// The seed's text, for writing it to its file. Never log it.
    pub fn as_str(&self) -> &str {
        // The bytes are ASCII from the alphabet (checked by `parse`, made by `generate_password`).
        std::str::from_utf8(&self.0).unwrap_or("")
    }
}

impl fmt::Debug for TimeoutSeed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TimeoutSeed(..)")
    }
}

/// The timeout code of one connection: derived from the seed and the server's address, the same for the same pair (so a reconnect to the
/// same server presents the same code) and different for another server. `Debug` shows only the length.
#[derive(Clone, PartialEq, Eq)]
pub struct TimeoutCode([u8; CODE_LEN]);

impl TimeoutCode {
    /// `GenerateTimeoutCode(.., Dummy = false)` with one connect address: `generate_password` over
    /// `MD5("normal\0" ++ seed ++ "\0" ++ NETADDR)`, where `NETADDR` is DDNet's 24-byte struct as it sits in memory:
    /// `type: u32` (1 = IPv4, 2 = IPv6; little-endian host), `ip: [u8; 16]` (an IPv4 address in the first four bytes, the rest zero),
    /// `port: u16` (host order), two bytes of padding (zero here; the C++ struct leaves them indeterminate, and the server only ever
    /// compares the resulting string, so the value cannot matter to it). The address is the **game server's**, never a proxy's.
    pub fn derive(seed: &TimeoutSeed, server: SocketAddr) -> TimeoutCode {
        let mut netaddr = [0u8; 24];
        match server.ip().to_canonical() {
            IpAddr::V4(v4) => {
                netaddr[0..4].copy_from_slice(&1u32.to_le_bytes());
                netaddr[4..8].copy_from_slice(&v4.octets());
            }
            IpAddr::V6(v6) => {
                netaddr[0..4].copy_from_slice(&2u32.to_le_bytes());
                netaddr[4..20].copy_from_slice(&v6.octets());
            }
        }
        netaddr[20..22].copy_from_slice(&server.port().to_le_bytes());

        let mut md5 = Md5::new();
        md5.update(b"normal\0");
        md5.update(seed.0);
        md5.update([0u8]);
        md5.update(netaddr);
        let digest = md5.finalize();
        let mut shorts = [0u16; 8];
        for (i, s) in shorts.iter_mut().enumerate() {
            *s = u16::from_le_bytes([digest[2 * i], digest[2 * i + 1]]);
        }
        TimeoutCode(generate_password(&shorts))
    }

    /// Length in bytes (the only thing that may be logged about a code).
    #[allow(clippy::len_without_is_empty)] // a code is never empty
    pub fn len(&self) -> usize {
        CODE_LEN
    }

    fn as_str(&self) -> &str {
        std::str::from_utf8(&self.0).unwrap_or("")
    }
}

impl fmt::Debug for TimeoutCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TimeoutCode(len {CODE_LEN})")
    }
}

/// The one chat command of this module: `/timeout <code>`, all-chat (team 0, as the real client sends a typed command). It carries a
/// [`TimeoutCode`] and nothing else: there is no way to put other text in it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeoutCommand {
    code: TimeoutCode,
}

/// The wire payload of a [`TimeoutCommand`] (`(NETMSGTYPE_CL_SAY << 1)` varint, team 0, the NUL-terminated text), as
/// `Connection::send_chunk` takes it. Only [`TimeoutCommand::payload`] builds one, so holding one is the proof that the bytes came from a
/// derived code; a session records its authorisation from it. `Debug` hides the bytes.
#[derive(Clone, PartialEq, Eq)]
pub struct TimeoutPayload(Vec<u8>);

impl TimeoutPayload {
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.0
    }
}

impl fmt::Debug for TimeoutPayload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TimeoutPayload({} bytes)", self.0.len())
    }
}

impl TimeoutCommand {
    pub fn new(code: TimeoutCode) -> TimeoutCommand {
        TimeoutCommand { code }
    }

    /// The code's length, for the log line `timeout code sent (len N)`.
    pub fn code_len(&self) -> usize {
        self.code.len()
    }

    /// The payload. Fails closed: should the buffer ever not fit (it always does), the payload is empty and the allow-list refuses it
    /// as undecodable.
    pub fn payload(&self) -> TimeoutPayload {
        TimeoutPayload(Self::encode(self.code.as_str()))
    }

    fn encode(text: &str) -> Vec<u8> {
        let mut buf = [0u8; 64];
        let mut packer = Packer::new(&mut buf);
        pack_msg_id(&mut packer, MsgId::Numbered(msgs::id::NETMSGTYPE_CL_SAY), false);
        msgs::encode_cl_say(
            &ClSay {
                team: 0,
                message: format!("{PREFIX}{text}"),
            },
            &mut packer,
        );
        if packer.error() {
            return Vec::new();
        }
        packer.data().to_vec()
    }

    /// Whether `payload` is, byte for byte, the encoding of *some* timeout command: a non-system `Cl_Say`, team 0, the text `/timeout `
    /// followed by exactly [`CODE_LEN`] characters of the password alphabet, nothing before, between or after (no other case, no
    /// trailing space or NUL, no other command, no `;` chaining, no team chat). This is the structural half of the allow-list's check;
    /// the other half is the one-shot authorisation for the exact bytes. A yes/no only.
    pub fn is_canonical(payload: &[u8]) -> bool {
        let mut u = Unpacker::new(payload);
        if u.get_int() != (msgs::id::NETMSGTYPE_CL_SAY << 1) || u.error() {
            return false;
        }
        let Some(say) = msgs::decode_cl_say(&mut u) else {
            return false;
        };
        if say.team != 0 {
            return false;
        }
        let Some(code) = say.message.strip_prefix(PREFIX) else {
            return false;
        };
        code.len() == CODE_LEN && is_alphabet(code.as_bytes()) && Self::encode(code) == payload
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uuid::unpack_msg_id;

    fn seed(s: &str) -> TimeoutSeed {
        TimeoutSeed::parse(s).expect("a valid seed")
    }

    fn addr(s: &str) -> SocketAddr {
        s.parse().unwrap()
    }

    fn code_str(seed_text: &str, server: &str) -> String {
        TimeoutCode::derive(&seed(seed_text), addr(server)).as_str().to_string()
    }

    /// Reference values produced by compiling DDNet 20.1's own code: `GenerateTimeoutCode` (`engine/client/client.cpp`) and
    /// `generate_password` (`base/secure.cpp`) verbatim, DDNet's bundled MD5 (`engine/external/md5`), and the real `NETADDR`
    /// (`sizeof` 24: `type`, `ip[16]`, `port`, two bytes of padding, the array zeroed first as `CClient::Connect` does with `mem_zero`).
    /// Each line: seed, server, the code the C++ printed (the harness is `tools/ddnet-vectors/timeout_code.cpp`).
    const CPP_VECTORS: &[(&str, &str, &str)] = &[
        ("ABCDEFGHKLMNPRST", "127.0.0.1:8443", "KbCS2mj3DjD2YRE2"),
        ("ABCDEFGHKLMNPRST", "192.0.2.35:8308", "4HkFtCM5kgLapeeN"),
        ("n2e9mUWqk3HdGPt8", "192.0.2.35:8308", "RN6gTPkCHcnZnp5E"),
        ("n2e9mUWqk3HdGPt8", "127.0.0.1:8444", "YjPtdL5fPKf2mGbZ"),
        ("n2e9mUWqk3HdGPt8", "[::1]:8303", "K2Xj3daZftB2oKme"),
    ];

    #[test]
    fn the_code_matches_ddnets_own_algorithm_on_the_cpp_vectors() {
        for (seed_text, server, expected) in CPP_VECTORS {
            assert_eq!(&code_str(seed_text, server), expected, "{seed_text} {server}");
        }
    }

    #[test]
    fn the_code_is_deterministic_and_depends_on_seed_and_address_only() {
        let a = code_str("ABCDEFGHKLMNPRST", "127.0.0.1:8443");
        assert_eq!(
            a,
            code_str("ABCDEFGHKLMNPRST", "127.0.0.1:8443"),
            "same inputs, same code"
        );
        assert_ne!(a, code_str("ABCDEFGHKLMNPRSU", "127.0.0.1:8443"), "another seed");
        assert_ne!(a, code_str("ABCDEFGHKLMNPRST", "127.0.0.1:8444"), "another port");
        assert_ne!(a, code_str("ABCDEFGHKLMNPRST", "127.0.0.2:8443"), "another address");
        // an IPv4-mapped IPv6 address is the IPv4 address (how a dual-stack socket reports it)
        assert_eq!(a, code_str("ABCDEFGHKLMNPRST", "[::ffff:127.0.0.1]:8443"));
        assert_eq!(a.len(), CODE_LEN);
        assert!(is_alphabet(a.as_bytes()));
    }

    /// `secure_random_password(.., 16)` over known bytes (the table is the C++ `generate_password` applied by hand, little-endian).
    #[test]
    fn a_seed_from_randomness_follows_generate_password() {
        let mut bytes = [0u8; 16];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i as u8;
        }
        assert_eq!(TimeoutSeed::from_random(bytes).as_str(), "FeUof63EFoU6gE3P");
        assert_eq!(TimeoutSeed::from_random([255; 16]).as_str(), "8b8b8b8b8b8b8b8b");
        assert_eq!(TimeoutSeed::from_random([0; 16]).as_str(), "AAAAAAAAAAAAAAAA");
    }

    #[test]
    fn a_seed_is_parsed_strictly_and_never_shown() {
        assert!(TimeoutSeed::parse("ABCDEFGHKLMNPRST").is_some());
        assert!(
            TimeoutSeed::parse("ABCDEFGHKLMNPRST\n").is_some(),
            "a final newline is fine"
        );
        for bad in [
            "",
            "ABCDEFGHKLMNPRS",
            "ABCDEFGHKLMNPRSTU",
            "ABCDEFGHKLMNPRSl",
            "ABCDEFGHKLMNPRS0",
            "ABCDEFGHKLMNPR T",
            "ABCDEFGHKLMNPRS\u{e9}",
        ] {
            assert!(TimeoutSeed::parse(bad).is_none(), "{bad:?}");
        }
        let s = seed("ABCDEFGHKLMNPRST");
        assert_eq!(format!("{s:?}"), "TimeoutSeed(..)");
        let c = TimeoutCode::derive(&s, addr("127.0.0.1:8443"));
        assert_eq!(format!("{c:?}"), "TimeoutCode(len 16)");
        let cmd = TimeoutCommand::new(c);
        assert!(!format!("{cmd:?}").contains("KbCS"), "{cmd:?}");
        assert!(!format!("{:?}", cmd.payload()).contains("KbCS"));
    }

    fn say(team: i32, text: &str) -> Vec<u8> {
        let mut buf = [0u8; 2048];
        let mut packer = Packer::new(&mut buf);
        pack_msg_id(&mut packer, MsgId::Numbered(msgs::id::NETMSGTYPE_CL_SAY), false);
        msgs::encode_cl_say(
            &ClSay {
                team,
                message: text.to_string(),
            },
            &mut packer,
        );
        packer.data().to_vec()
    }

    #[test]
    fn the_command_is_a_cl_say_of_slash_timeout_and_the_code_in_all_chat() {
        let cmd = TimeoutCommand::new(TimeoutCode::derive(&seed("ABCDEFGHKLMNPRST"), addr("127.0.0.1:8443")));
        let payload = cmd.payload();
        let mut u = Unpacker::new(payload.as_bytes());
        let registry = crate::message::Registry::new();
        let (id, sys) = unpack_msg_id(&mut u, registry.uuids()).unwrap();
        assert_eq!(id, MsgId::Numbered(msgs::id::NETMSGTYPE_CL_SAY));
        assert!(!sys);
        let decoded = msgs::decode_cl_say(&mut u).expect("decodes");
        assert_eq!(decoded.team, 0);
        assert_eq!(decoded.message, "/timeout KbCS2mj3DjD2YRE2");
        assert!(!u.error());
        // the tail of the wire bytes, spelled out: team 0, the text, NUL
        let mut tail = vec![0u8];
        tail.extend_from_slice(b"/timeout KbCS2mj3DjD2YRE2");
        tail.push(0);
        assert!(payload.as_bytes().ends_with(&tail));
        assert_eq!(
            payload.as_bytes()[0],
            0x22,
            "varint(NETMSGTYPE_CL_SAY << 1), no sys bit"
        );
        assert_eq!(
            payload.as_bytes().len(),
            1 + tail.len(),
            "docs/formats.md §38.2 shows these bytes"
        );
        assert_eq!(cmd.code_len(), 16);
        assert!(TimeoutCommand::is_canonical(payload.as_bytes()));
    }

    #[test]
    fn only_the_exact_shape_is_canonical() {
        let good = "/timeout KbCS2mj3DjD2YRE2";
        assert!(TimeoutCommand::is_canonical(&say(0, good)));
        let long = format!("{good}{}", "A".repeat(300));
        let refused = [
            say(1, good), // team chat
            say(2, good),
            say(-1, good),
            say(0, "/timeout"),
            say(0, "/timeout "),
            say(0, "/timeout KbCS2mj3DjD2YRE"),   // 15
            say(0, "/timeout KbCS2mj3DjD2YRE22"), // 17
            say(0, "/timeout KbCS2mj3DjD2YRE2 "),
            say(0, " /timeout KbCS2mj3DjD2YRE2"),
            say(0, "/Timeout KbCS2mj3DjD2YRE2"),
            say(0, "/TIMEOUT KbCS2mj3DjD2YRE2"),
            say(0, "/timeout  KbCS2mj3DjD2YRE2"),
            say(0, "/timeout KbCS2mj3DjD2YRE2;kill"),
            say(0, "/timeout KbCS2mj3DjD2YRE;"),
            say(0, "/timeout KbCS2mj3DjD2YR l"),
            say(0, "/timeout KbCS2mj3DjD2YR1"), // `1` is not in the alphabet
            say(0, "/timeout KbCS2mj3DjD2YR0"),
            say(0, "/timeout KbCS2mj3DjD2YRI"),
            say(0, "timeout KbCS2mj3DjD2YRE2"),
            say(0, "/mc;timeout KbCS2mj3DjD2YRE2"),
            say(0, "/kill"),
            say(0, "hello"),
            say(0, ""),
            say(0, &long),
        ];
        for (i, p) in refused.iter().enumerate() {
            assert!(!TimeoutCommand::is_canonical(p), "refused[{i}]");
        }
        // trailing byte, cut short, a system-bit id, garbage
        let mut longer = say(0, good);
        longer.push(b'x');
        assert!(!TimeoutCommand::is_canonical(&longer));
        let mut shorter = say(0, good);
        shorter.pop();
        assert!(!TimeoutCommand::is_canonical(&shorter));
        let mut sys = say(0, good);
        sys[0] |= 1;
        assert!(!TimeoutCommand::is_canonical(&sys));
        assert!(!TimeoutCommand::is_canonical(&[]));
        assert!(!TimeoutCommand::is_canonical(&[0xff; 6]));
    }

    /// The one chat-sending type with a fixed shape: no variant or field carries text other than the code.
    #[test]
    fn the_command_holds_nothing_but_the_code() {
        let TimeoutCommand { code } =
            TimeoutCommand::new(TimeoutCode::derive(&seed("ABCDEFGHKLMNPRST"), addr("127.0.0.1:1")));
        assert_eq!(std::mem::size_of::<TimeoutCommand>(), std::mem::size_of_val(&code));
    }
}
