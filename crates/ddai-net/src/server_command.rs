//! The one chat-channel message the bot may ever send: the server command `/kill` (task 4.6, D-078).
//!
//! D-007 stands: the bot never writes to the game chat. The owner allowed exactly one exception, because DDNet
//! drops the protocol `Cl_Kill` silently once a life is older than `sv_kill_protection` minutes (default 20,
//! `gamecontext.cpp:2977` in 20.1) and the server command `/kill` is then the only way to die. The command travels
//! as `Cl_Say`, but other players do not see it (the server consumes a leading `/`).
//!
//! **The guarantee is in the type.** [`ServerCommand`] has one variant, [`ServerCommand::Kill`], and its encoder
//! writes the constant text `/kill` with the team flag 0 (all-chat, as the real client sends a typed command).
//! There is no function anywhere that takes a string and makes a `Cl_Say` of it: the generated `encode_cl_say`
//! stays `pub(crate)`, and this module is its only caller. The outgoing allow-list (`ddai-client::allowlist`) is the
//! second, independent check: it lets a `Cl_Say` through only when the payload is **byte-identical** to
//! [`ServerCommand::payload`].

//!
//! The two claims, checked by the compiler (these doctests must fail to compile; a regression that makes either compile
//! fails `cargo test`):
//!
//! ```compile_fail,E0603
//! // `encode_cl_say` is not public: there is no way to build a `Cl_Say` from a string outside this crate.
//! let mut buf = [0u8; 64];
//! let mut packer = ddai_net::packer::Packer::new(&mut buf);
//! let say = ddai_net::generated::messages::ClSay { team: 0, message: "hello".to_string() };
//! ddai_net::generated::messages::encode_cl_say(&say, &mut packer);
//! ```
//!
//! ```compile_fail,E0599
//! // `ServerCommand` has exactly one variant, `Kill`: no variant carries text.
//! let _ = ddai_net::server_command::ServerCommand::Say("hello".to_string());
//! ```

use crate::generated::messages::{self as msgs, ClSay};
use crate::packer::Packer;
use crate::uuid::{MsgId, pack_msg_id};

/// A server command the bot may send. One variant, on purpose: adding another is a decision for the owner (D-078), not a
/// code change that can slip through a review.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerCommand {
    /// `/kill`: ends our own life, like the protocol `Cl_Kill`, but is not subject to kill protection.
    Kill,
}

/// The exact chat text of [`ServerCommand::Kill`].
const KILL_TEXT: &str = "/kill";

impl ServerCommand {
    /// The command as it is typed in chat (a constant of the type: never built from input).
    pub const fn text(self) -> &'static str {
        match self {
            ServerCommand::Kill => KILL_TEXT,
        }
    }

    /// The whole game-message payload (`(NETMSGTYPE_CL_SAY << 1)` varint, team, the NUL-terminated text), ready for
    /// `Connection::send_chunk`.
    pub fn payload(self) -> Vec<u8> {
        let mut buf = [0u8; 64];
        let mut packer = Packer::new(&mut buf);
        pack_msg_id(&mut packer, MsgId::Numbered(msgs::id::NETMSGTYPE_CL_SAY), false);
        msgs::encode_cl_say(
            &ClSay {
                team: 0,
                message: self.text().to_string(),
            },
            &mut packer,
        );
        packer.data().to_vec()
    }

    /// Whether `payload` is, byte for byte, the encoding of a server command (nothing else qualifies: another text, a
    /// trailing space or NUL, another case, team chat, extra bytes).
    pub fn recognise(payload: &[u8]) -> Option<ServerCommand> {
        [ServerCommand::Kill].into_iter().find(|c| c.payload() == payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packer::Unpacker;
    use crate::uuid::unpack_msg_id;

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

    /// A new variant breaks this match: adding one is a decision for the owner (D-078), and the allow-list, the audit label and
    /// the tests have to be revisited with it. (The doctests only rule out some names; this rules out any addition.)
    #[test]
    fn kill_is_the_only_variant() {
        match ServerCommand::Kill {
            ServerCommand::Kill => {}
        }
        assert_eq!(std::mem::size_of::<ServerCommand>(), 0, "no variant carries data");
    }

    #[test]
    fn kill_is_a_cl_say_of_exactly_slash_kill_in_all_chat() {
        let payload = ServerCommand::Kill.payload();
        let mut u = Unpacker::new(&payload);
        let registry = crate::message::Registry::new();
        let (id, sys) = unpack_msg_id(&mut u, registry.uuids()).unwrap();
        assert_eq!(id, MsgId::Numbered(msgs::id::NETMSGTYPE_CL_SAY));
        assert!(!sys);
        let decoded = msgs::decode_cl_say(&mut u).expect("decodes");
        assert_eq!(decoded.team, 0);
        assert_eq!(decoded.message, "/kill");
        assert!(!u.error());
        // the tail of the wire bytes, spelled out: team 0, "/kill", NUL
        assert!(payload.ends_with(&[0, b'/', b'k', b'i', b'l', b'l', 0]));
        assert_eq!(ServerCommand::Kill.text(), "/kill");
    }

    #[test]
    fn only_the_byte_identical_payload_is_recognised() {
        assert_eq!(
            ServerCommand::recognise(&ServerCommand::Kill.payload()),
            Some(ServerCommand::Kill)
        );
        let long = "a".repeat(400);
        let refused = [
            say(0, "/kill "),
            say(0, " /kill"),
            say(0, "/KILL"),
            say(0, "/Kill"),
            say(0, "/kill\0"),
            say(0, "/kill\0x"),
            say(0, "/kill/kill"),
            say(0, "/kills"),
            say(0, "/ki11"),
            say(0, "kill"),
            say(0, "hello"),
            say(0, ""),
            say(0, "/"),
            say(0, &long),
            say(1, "/kill"), // team chat
            say(2, "/kill"),
            say(-1, "/kill"),
        ];
        for p in &refused {
            assert_eq!(ServerCommand::recognise(p), None, "{p:?}");
        }
        // extra bytes after, bytes missing, empty
        let mut longer = ServerCommand::Kill.payload();
        longer.push(0);
        assert_eq!(ServerCommand::recognise(&longer), None);
        let mut shorter = ServerCommand::Kill.payload();
        shorter.pop();
        assert_eq!(ServerCommand::recognise(&shorter), None);
        assert_eq!(ServerCommand::recognise(&[]), None);
    }
}
