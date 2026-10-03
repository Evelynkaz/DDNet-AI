//! The outgoing-message allow-list guard — task 2.3 acceptance criterion 6g, closing the
//! "residual finding" flagged at the end of task 2.2b's review (`docs/formats.md` §13.9): the
//! public low-level `ddai_net::conn::Connection::send_chunk` plus `ddai_net::packer::Packer` can
//! carry *any* bytes, including a hand-built `Cl_Say` — nothing at that layer (by design: it is a
//! transport-independent protocol crate, not a policy layer) stops it. D-007 ("the bot never
//! sends chat") is enforced at the type level everywhere [`crate::session::Session`] itself builds
//! an outgoing message (there is simply no code path that constructs a `Cl_Say` payload) — but
//! defence in depth means a *second*, independent, runtime check on the raw bytes, right before
//! they reach [`ddai_net::conn::Connection::send_chunk`], so that even a future bug that
//! accidentally packs the wrong message id (a typo'd constant, a copy-pasted branch, …) is caught
//! here rather than silently reaching the wire.
//!
//! [`Session`](crate::session::Session) funnels *every* outgoing chunk it ever sends through
//! exactly one function, [`crate::session::Session::send_game_chunk`] (game/ex-game messages,
//! guarded by this module) or the small, fixed set of protocol-level sends the join sequence and
//! keep-alive/timing machinery build directly (`Cl_StartInfo`'s sibling system messages —
//! `NETMSG_INFO`/`READY`/`ENTERGAME`/`INPUT`/`REQUEST_MAP_DATA`/`PING_REPLY`/`CLIENTVER`/`PONGEX`/
//! `CHECKSUM_ERROR`/the `WHATIS` answers — none of which can ever be a chat message: the `sys`
//! bit distinguishes DDNet's *system* message namespace, entirely disjoint from the *game*
//! message namespace `NETMSGTYPE_CL_SAY` lives in, so this module does not need to re-check them;
//! see [`Guard::check`]'s doc comment for exactly where that split is enforced).

use ddai_net::message::Registry;
use ddai_net::packer::Unpacker;
use ddai_net::server_command::ServerCommand;
use ddai_net::uuid::{MsgId, unpack_msg_id};

/// Numbered (non-UUID) game message ids this session is allowed to ever send —
/// `datasrc/network.py`'s declaration order, task spec 6g's list minus `Cl_ChangeInfo`/
/// `Cl_Emoticon` (kept here anyway, "if used" per the spec — nothing in this crate sends them
/// yet, but a caller of a future public setter must not have to touch this guard) and
/// `Cl_IsDDNetLegacy` (hand-decoded on the wire —
/// [`Session::send_post_enter_extras`](crate::session::Session::send_post_enter_extras)'s doc
/// comment explains why — but still a numbered id that must pass this same gate).
const ALLOWED_NUMBERED_IDS: &[i32] = &[
    ddai_net::generated::messages::id::NETMSGTYPE_CL_SETTEAM,
    ddai_net::generated::messages::id::NETMSGTYPE_CL_STARTINFO,
    ddai_net::generated::messages::id::NETMSGTYPE_CL_CHANGEINFO,
    ddai_net::generated::messages::id::NETMSGTYPE_CL_KILL,
    ddai_net::generated::messages::id::NETMSGTYPE_CL_EMOTICON,
    ddai_net::generated::messages::id::NETMSGTYPE_CL_ISDDNETLEGACY,
];

/// UUID (`ex`) game message names this session is allowed to ever send.
// Note: `PingEx`/`PongEx` (`ping@ddnet.tw`) are *system* ex-messages, sent through
// `Session::send_ex_system_chunk` (the `sys` bit set) — that path never reaches this guard at all
// (see the module docs' "why only non-system messages need checking" section), so they are
// deliberately not listed here, unlike the genuinely *game* ex-messages below.
const ALLOWED_EX_NAMES: &[&str] = &[
    "show-distance@netmsg.ddnet.tw",           // Cl_ShowDistance
    "showothers@netmsg.ddnet.tw",              // Cl_ShowOthers
    "camera-info@netmsg.ddnet.org",            // Cl_CameraInfo
    "enable-spectator-count@netmsg.ddnet.org", // Cl_EnableSpectatorCount
];

/// Why [`Guard::check`] refused to let a payload reach the wire.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GuardError {
    #[error("outgoing payload's leading message id could not even be decoded")]
    Undecodable,
    #[error("refused a system message on the game guard path (should never happen: sys messages bypass this guard)")]
    UnexpectedSystemMessage,
    #[error(
        "outgoing numbered game message id {0} is not on the allow-list (D-007: the bot never sends chat; D-078: only `/kill`, byte for byte)"
    )]
    NumberedIdNotAllowed(i32),
    #[error("outgoing ex game message '{0}' is not on the allow-list (D-007: the bot never sends chat)")]
    ExNameNotAllowed(String),
    #[error("outgoing ex game message has an unresolved/unregistered UUID — refused, not on the allow-list")]
    UnresolvedExName,
}

/// Peeks the leading `(id<<1)|sys` varint (and UUID, if extended) of `payload` — the exact same
/// bytes about to be handed to [`ddai_net::conn::Connection::send_chunk`] — and refuses it unless
/// it is a **non-system** (`sys == false`) message whose id/name is on this module's allow-list.
///
/// Why only non-system messages need checking here: DDNet's own wire format puts every chat
/// message (`NETMSGTYPE_CL_SAY`) in the *game* message namespace (`sys` bit clear), disjoint by
/// construction from the *system* namespace (`sys` bit set, `protocol.h`'s `NETMSG_*`) that
/// `Session`'s join-sequence/keep-alive code builds directly — there is no numbered or UUID
/// system message that means "chat" (see `crate::sysmsg`/`crate::message`'s exhaustive id
/// tables). [`Session::send_system_chunk`](crate::session::Session::send_system_chunk) (the other
/// half of the "single outgoing path" this guard's module doc refers to) therefore skips this
/// check entirely rather than special-casing "sys messages always pass" inside this function —
/// keeping this guard's allow-list *exclusively* about the one namespace that can ever carry
/// chat, so a reviewer auditing this file never has to reason about the (much larger, and
/// deliberately not enumerated here) set of legitimate system message ids too.
pub fn check(payload: &[u8], registry: &Registry) -> Result<(), GuardError> {
    let mut unpacker = Unpacker::new(payload);
    let (id, sys) = unpack_msg_id(&mut unpacker, registry.uuids()).map_err(|_| GuardError::Undecodable)?;
    if sys {
        return Err(GuardError::UnexpectedSystemMessage);
    }
    match id {
        MsgId::Numbered(numbered) => {
            // D-078: `Cl_Say` is allowed for exactly one payload, the typed `/kill` command, byte for byte (no other text, no
            // other case, no trailing byte, no team chat). Everything else with this id is chat and stays refused.
            if numbered == ddai_net::generated::messages::id::NETMSGTYPE_CL_SAY {
                return if ServerCommand::recognise(payload) == Some(ServerCommand::Kill) {
                    Ok(())
                } else {
                    Err(GuardError::NumberedIdNotAllowed(numbered))
                };
            }
            if ALLOWED_NUMBERED_IDS.contains(&numbered) {
                Ok(())
            } else {
                Err(GuardError::NumberedIdNotAllowed(numbered))
            }
        }
        MsgId::Ex { resolved, .. } => {
            let name = resolved.and_then(|id| registry.uuids().name(id));
            match name {
                Some(name) if ALLOWED_EX_NAMES.contains(&name) => Ok(()),
                Some(name) => Err(GuardError::ExNameNotAllowed(name.to_string())),
                None => Err(GuardError::UnresolvedExName),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddai_net::generated::messages as msgs;
    use ddai_net::packer::Packer;
    use ddai_net::uuid::{calculate_uuid, pack_msg_id};

    fn numbered_payload(sys: bool, id: i32, body: impl Fn(&mut Packer)) -> Vec<u8> {
        let mut buf = [0u8; 4096];
        let mut packer = Packer::new(&mut buf);
        pack_msg_id(&mut packer, MsgId::Numbered(id), sys);
        body(&mut packer);
        packer.data().to_vec()
    }

    fn ex_payload(sys: bool, name: &str, body: impl Fn(&mut Packer)) -> Vec<u8> {
        let mut buf = [0u8; 4096];
        let mut packer = Packer::new(&mut buf);
        let id = calculate_uuid(name);
        pack_msg_id(
            &mut packer,
            MsgId::Ex {
                uuid: id,
                resolved: None,
            },
            sys,
        );
        body(&mut packer);
        packer.data().to_vec()
    }

    #[test]
    fn cl_say_is_rejected() {
        let registry = Registry::new();
        let mut buf = [0u8; 4096];
        let mut packer = Packer::new(&mut buf);
        pack_msg_id(&mut packer, MsgId::Numbered(msgs::id::NETMSGTYPE_CL_SAY), false);
        // `msgs::encode_cl_say` is `pub(crate)` inside `ddai-net` (D-007: not reachable from
        // outside that crate at all) — hand-pack the exact same wire shape here instead, since
        // this test is specifically about a *hand-built* Cl_Say slipping past the guard.
        packer.add_int(0); // team
        packer.add_string("hi", 0, true); // message
        let err = check(packer.data(), &registry).unwrap_err();
        assert_eq!(err, GuardError::NumberedIdNotAllowed(msgs::id::NETMSGTYPE_CL_SAY));
    }

    /// D-078: the typed `/kill` is the one `Cl_Say` the guard lets through.
    #[test]
    fn the_typed_slash_kill_is_the_only_cl_say_allowed() {
        let registry = Registry::new();
        assert!(check(&ServerCommand::Kill.payload(), &registry).is_ok());
        let say = |team: i32, text: &str| {
            let mut buf = [0u8; 2048];
            let mut packer = Packer::new(&mut buf);
            pack_msg_id(&mut packer, MsgId::Numbered(msgs::id::NETMSGTYPE_CL_SAY), false);
            packer.add_int(team);
            packer.add_string(text, 0, true);
            packer.data().to_vec()
        };
        let long = "x".repeat(300);
        for (team, text) in [
            (0, "hi"),
            (0, "/kill "),
            (0, " /kill"),
            (0, "/KILL"),
            (0, "/Kill"),
            (0, "/kill\0x"),
            (0, "/kills"),
            (0, "/kill /kill"),
            (0, "/help"),
            (0, ""),
            (0, long.as_str()),
            (1, "/kill"),
            (1, "hi"),
            (2, "/kill"),
        ] {
            assert_eq!(
                check(&say(team, text), &registry).unwrap_err(),
                GuardError::NumberedIdNotAllowed(msgs::id::NETMSGTYPE_CL_SAY),
                "team {team} text {text:?}"
            );
        }
        // the right text with trailing garbage, or cut short
        let mut longer = ServerCommand::Kill.payload();
        longer.extend_from_slice(b"x");
        assert!(check(&longer, &registry).is_err());
        let mut shorter = ServerCommand::Kill.payload();
        shorter.pop();
        assert!(check(&shorter, &registry).is_err());
        // a NUL inside the text: `add_string` stops at it, so build the bytes by hand
        let mut nul = ServerCommand::Kill.payload();
        let n = nul.len();
        nul[n - 2] = 0; // "/kil\0\0"
        assert!(check(&nul, &registry).is_err());
        // the same id as a *system* message is refused by the sys check, never a way around it
        let mut sys = ServerCommand::Kill.payload();
        sys[0] |= 1;
        assert!(check(&sys, &registry).is_err());
    }

    #[test]
    fn cl_start_info_is_allowed() {
        let registry = Registry::new();
        let payload = numbered_payload(false, msgs::id::NETMSGTYPE_CL_STARTINFO, |_| {});
        assert!(check(&payload, &registry).is_ok());
    }

    #[test]
    fn cl_set_team_kill_emoticon_change_info_and_is_ddnet_legacy_are_allowed() {
        let registry = Registry::new();
        for id in [
            msgs::id::NETMSGTYPE_CL_SETTEAM,
            msgs::id::NETMSGTYPE_CL_KILL,
            msgs::id::NETMSGTYPE_CL_EMOTICON,
            msgs::id::NETMSGTYPE_CL_CHANGEINFO,
            msgs::id::NETMSGTYPE_CL_ISDDNETLEGACY,
        ] {
            let payload = numbered_payload(false, id, |_| {});
            assert!(check(&payload, &registry).is_ok(), "id {id} should be allowed");
        }
    }

    #[test]
    fn show_distance_show_others_camera_info_and_enable_spectator_count_are_allowed() {
        let registry = Registry::new();
        for name in [
            "show-distance@netmsg.ddnet.tw",
            "showothers@netmsg.ddnet.tw",
            "camera-info@netmsg.ddnet.org",
            "enable-spectator-count@netmsg.ddnet.org",
        ] {
            let payload = ex_payload(false, name, |_| {});
            assert!(check(&payload, &registry).is_ok(), "{name} should be allowed");
        }
    }

    /// Review finding F3: the module docs previously claimed `PingEx`/`PongEx` bypass this guard
    /// via the `sys` bit — this pins that claim down as an actual test, not just prose.
    #[test]
    fn ping_ex_as_a_system_message_bypasses_this_guard_via_the_sys_bit() {
        let registry = Registry::new();
        let payload = ex_payload(true, "ping@ddnet.tw", |_| {});
        assert_eq!(
            check(&payload, &registry).unwrap_err(),
            GuardError::UnexpectedSystemMessage
        );
    }

    #[test]
    fn an_arbitrary_other_ex_game_name_is_rejected() {
        let registry = Registry::new();
        // A real, registered ex game name that is not on the allow-list.
        let payload = ex_payload(false, "ddrace-time@netmsg.ddnet.tw", |_| {});
        let err = check(&payload, &registry).unwrap_err();
        assert_eq!(
            err,
            GuardError::ExNameNotAllowed("ddrace-time@netmsg.ddnet.tw".to_string())
        );
    }

    #[test]
    fn an_unregistered_ex_uuid_is_rejected() {
        let registry = Registry::new();
        let payload = ex_payload(false, "totally-made-up@example.com", |_| {});
        let err = check(&payload, &registry).unwrap_err();
        assert_eq!(err, GuardError::UnresolvedExName);
    }

    #[test]
    fn a_system_message_is_rejected_by_this_guard_specifically() {
        // Session must never route a sys message through this function at all (it has its own
        // `send_system_chunk` path) — this pins down that *if* it ever did by mistake, the guard
        // still refuses rather than silently accepting it.
        let registry = Registry::new();
        let payload = numbered_payload(true, ddai_net::sysmsg::id::READY, |_| {});
        assert_eq!(
            check(&payload, &registry).unwrap_err(),
            GuardError::UnexpectedSystemMessage
        );
    }

    #[test]
    fn garbage_payload_is_rejected_not_panicking() {
        let registry = Registry::new();
        for len in 0..8 {
            let payload = vec![0xffu8; len];
            let _ = check(&payload, &registry); // must not panic
        }
        assert_eq!(check(&[], &registry).unwrap_err(), GuardError::Undecodable);
    }

    #[test]
    fn unregistered_numbered_id_far_outside_any_known_range_is_rejected() {
        let registry = Registry::new();
        let payload = numbered_payload(false, 9999, |_| {});
        assert_eq!(
            check(&payload, &registry).unwrap_err(),
            GuardError::NumberedIdNotAllowed(9999)
        );
    }
}
