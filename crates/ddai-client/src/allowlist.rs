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
use ddai_net::owner_chat::{OwnerPayload, OwnerSay};
use ddai_net::packer::Unpacker;
use ddai_net::server_command::ServerCommand;
use ddai_net::timeout_code::{TimeoutCommand, TimeoutPayload};
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
        "outgoing numbered game message id {0} is not on the allow-list (D-007: the bot never sends chat; D-078: only `/kill`, byte for byte; D-094: a Cl_Say only against a one-shot authorisation for an owner's website line; D-100: and the one `/timeout <code>` of the join, against its own authorisation)"
    )]
    NumberedIdNotAllowed(i32),
    #[error("outgoing ex game message '{0}' is not on the allow-list (D-007: the bot never sends chat)")]
    ExNameNotAllowed(String),
    #[error("outgoing ex game message has an unresolved/unregistered UUID — refused, not on the allow-list")]
    UnresolvedExName,
}

/// One-shot authorisations for owner chat (task 4.9, D-094): the exact payload bytes of `Cl_Say` lines that the session itself encoded
/// from an [`OwnerSay`] and is about to send. [`check_authorised`] **consumes** an entry when it lets a `Cl_Say` through, so replaying
/// the same bytes is refused, and the session revokes whatever is left as soon as its send is done.
///
/// There is no way to make an entry from raw bytes: [`OwnerSayAuth::grant`] takes an [`OwnerPayload`], which only
/// `OwnerSay::payload` (in `ddai-net`, from a validated `OwnerText`) can build. A forged authorisation therefore cannot be
/// written, and [`check_authorised`] additionally requires the bytes to be the canonical encoding of an [`OwnerSay`].
#[derive(Default)]
pub struct OwnerSayAuth {
    granted: Vec<Vec<u8>>,
}

impl std::fmt::Debug for OwnerSayAuth {
    /// The count only: the bytes are a user's text.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OwnerSayAuth({} pending)", self.granted.len())
    }
}

impl OwnerSayAuth {
    pub fn new() -> OwnerSayAuth {
        OwnerSayAuth::default()
    }

    /// Records an authorisation for exactly these bytes, good for one `Cl_Say`.
    pub fn grant(&mut self, payload: &OwnerPayload) {
        self.granted.push(payload.as_bytes().to_vec());
    }

    /// Drops every authorisation that was not used.
    pub fn revoke_all(&mut self) {
        self.granted.clear();
    }

    /// How many authorisations are waiting.
    pub fn pending(&self) -> usize {
        self.granted.len()
    }

    /// Removes and reports the authorisation for these exact bytes.
    fn take(&mut self, payload: &[u8]) -> bool {
        match self.granted.iter().position(|g| g == payload) {
            Some(i) => {
                self.granted.swap_remove(i);
                true
            }
            None => false,
        }
    }
}

/// The one-shot authorisation of the bot's own `/timeout <code>` (task 4.10, D-100): the exact payload bytes the session built from its
/// [`TimeoutCommand`] and is about to send. Like [`OwnerSayAuth`], and separate from it: a `Cl_Say` is let through as a timeout command
/// only against **this** authorisation (never against an owner line's), and only when the bytes are also structurally `/timeout ` plus a
/// 16-character code ([`TimeoutCommand::is_canonical`]). There is no way to make an entry from raw bytes: [`TimeoutAuth::grant`] takes a
/// [`TimeoutPayload`], which only [`TimeoutCommand::payload`] builds. It holds at most one entry, and [`check_with`] consumes it.
#[derive(Default)]
pub struct TimeoutAuth {
    granted: Option<Vec<u8>>,
}

impl std::fmt::Debug for TimeoutAuth {
    /// Whether one is pending, never the bytes (they hold the code).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TimeoutAuth({} pending)", self.pending())
    }
}

impl TimeoutAuth {
    pub fn new() -> TimeoutAuth {
        TimeoutAuth::default()
    }

    /// Records an authorisation for exactly these bytes, good for one `Cl_Say` (replacing an unused one).
    pub fn grant(&mut self, payload: &TimeoutPayload) {
        self.granted = Some(payload.as_bytes().to_vec());
    }

    /// Drops the authorisation if it was not used.
    pub fn revoke_all(&mut self) {
        self.granted = None;
    }

    /// How many authorisations are waiting (0 or 1).
    pub fn pending(&self) -> usize {
        usize::from(self.granted.is_some())
    }

    /// Removes and reports the authorisation for these exact bytes.
    fn take(&mut self, payload: &[u8]) -> bool {
        if self.granted.as_deref() == Some(payload) {
            self.granted = None;
            true
        } else {
            false
        }
    }
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
    check_authorised(payload, registry, &mut OwnerSayAuth::new())
}

/// [`check`] with the session's one-shot owner-chat authorisations (task 4.9, D-094): a `Cl_Say` also passes when `auth` holds an
/// authorisation for exactly these bytes **and** the bytes are the canonical encoding of an [`OwnerSay`] (so a validated line:
/// trimmed, no control or invisible character, at most 255 bytes, team 0 or 1; a leading `/` is allowed since 4.9b, the owner's
/// decision of 2026-10-05). The authorisation is consumed either way, also when the bytes are the typed `/kill` of D-078.
pub fn check_authorised(payload: &[u8], registry: &Registry, auth: &mut OwnerSayAuth) -> Result<(), GuardError> {
    check_with(payload, registry, auth, &mut TimeoutAuth::new())
}

/// [`check_authorised`] with the session's authorisation for its own `/timeout <code>` as well (task 4.10, D-100): a `Cl_Say` also passes
/// when `timeout_auth` holds an authorisation for exactly these bytes **and** the bytes are structurally a timeout command
/// ([`TimeoutCommand::is_canonical`]: team 0, `/timeout ` and 16 characters of the code alphabet). Both authorisations are consumed by any
/// `Cl_Say` that matches them, also when the check then refuses, so neither can be used for another message later.
pub fn check_with(
    payload: &[u8],
    registry: &Registry,
    auth: &mut OwnerSayAuth,
    timeout_auth: &mut TimeoutAuth,
) -> Result<(), GuardError> {
    let mut unpacker = Unpacker::new(payload);
    let (id, sys) = unpack_msg_id(&mut unpacker, registry.uuids()).map_err(|_| GuardError::Undecodable)?;
    if sys {
        return Err(GuardError::UnexpectedSystemMessage);
    }
    match id {
        MsgId::Numbered(numbered) => {
            // D-078: `Cl_Say` is allowed for the typed `/kill` command, byte for byte (no other text, no other case, no trailing
            // byte, no team chat). D-094: and for an owner's website line, only against a one-shot authorisation the session
            // recorded for these very bytes. Everything else with this id is chat and stays refused.
            if numbered == ddai_net::generated::messages::id::NETMSGTYPE_CL_SAY {
                // `take` first, so the authorisation is spent even when the structural check then refuses, and also when these are
                // the bytes of the typed `/kill` (the owner's own `/kill`, 4.9b, has the same bytes and must not leave one behind).
                let authorised = auth.take(payload);
                let timeout_authorised = timeout_auth.take(payload);
                if ServerCommand::recognise(payload) == Some(ServerCommand::Kill) {
                    return Ok(());
                }
                return if (authorised && OwnerSay::is_canonical(payload))
                    || (timeout_authorised && TimeoutCommand::is_canonical(payload))
                {
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

    fn owner_say(team: bool, text: &str) -> OwnerSay {
        OwnerSay::new(
            team,
            ddai_net::owner_chat::OwnerText::new(&ddai_net::owner_chat::OwnerChannel::mint_for_tests(), text)
                .expect("valid owner text"),
        )
    }

    fn hand_built_say(team: i32, text: &str) -> Vec<u8> {
        let mut buf = [0u8; 2048];
        let mut packer = Packer::new(&mut buf);
        pack_msg_id(&mut packer, MsgId::Numbered(msgs::id::NETMSGTYPE_CL_SAY), false);
        packer.add_int(team);
        packer.add_string(text, 0, true);
        packer.data().to_vec()
    }

    /// Task 4.9 (D-094): an owner's line passes with its authorisation, in all chat and in team chat, and only once.
    #[test]
    fn an_owner_line_passes_once_against_its_authorisation_and_a_replay_is_refused() {
        let registry = Registry::new();
        for team in [false, true] {
            let say = owner_say(team, "gg wp");
            let payload = say.payload();
            let mut auth = OwnerSayAuth::new();
            // without an authorisation the very same bytes are refused (it is a hand-built Cl_Say as far as the guard knows)
            assert_eq!(
                check_authorised(payload.as_bytes(), &registry, &mut auth).unwrap_err(),
                GuardError::NumberedIdNotAllowed(msgs::id::NETMSGTYPE_CL_SAY),
                "team {team}: no authorisation"
            );
            assert_eq!(
                check(payload.as_bytes(), &registry).unwrap_err(),
                GuardError::NumberedIdNotAllowed(msgs::id::NETMSGTYPE_CL_SAY),
                "the plain `check` has no authorisations at all"
            );
            auth.grant(&payload);
            assert_eq!(auth.pending(), 1);
            assert!(
                check_authorised(payload.as_bytes(), &registry, &mut auth).is_ok(),
                "team {team}"
            );
            assert_eq!(auth.pending(), 0, "consumed");
            // replay
            assert_eq!(
                check_authorised(payload.as_bytes(), &registry, &mut auth).unwrap_err(),
                GuardError::NumberedIdNotAllowed(msgs::id::NETMSGTYPE_CL_SAY),
                "team {team}: a replay of an authorised payload"
            );
        }
    }

    /// Task 4.9b: an owner's command line passes only against its authorisation, like any owner line; the owner's `/kill` has the bytes of
    /// the D-078 fallback, passes by either path, and the authorisation is spent so none is left behind.
    #[test]
    fn an_owner_command_passes_only_against_its_authorisation() {
        let registry = Registry::new();
        for (team, text) in [
            (false, "/emote happy"),
            (false, "/spec"),
            (true, "/w Name hi"),
            (true, "/kill"),
        ] {
            let payload = owner_say(team, text).payload();
            assert!(check(payload.as_bytes(), &registry).is_err(), "{text}: unauthorised");
            let mut auth = OwnerSayAuth::new();
            auth.grant(&payload);
            assert!(
                check_authorised(payload.as_bytes(), &registry, &mut auth).is_ok(),
                "{text}"
            );
            assert_eq!(auth.pending(), 0, "{text}: consumed");
            assert!(
                check_authorised(payload.as_bytes(), &registry, &mut auth).is_err(),
                "{text}: a replay is refused"
            );
        }
        // the owner's all-chat `/kill` is the very bytes of the fallback: it passes with or without an authorisation, and spends it
        let kill = owner_say(false, "/kill").payload();
        assert_eq!(kill.as_bytes(), ServerCommand::Kill.payload().as_slice());
        let mut auth = OwnerSayAuth::new();
        auth.grant(&kill);
        assert!(check_authorised(kill.as_bytes(), &registry, &mut auth).is_ok());
        assert_eq!(
            auth.pending(),
            0,
            "the authorisation is spent, not left to a later `/kill`"
        );
        assert!(
            check_authorised(kill.as_bytes(), &registry, &mut auth).is_ok(),
            "the fallback's own /kill"
        );
    }

    /// An authorisation is for exact bytes: another text, the other team flag or a longer line is not covered by it.
    #[test]
    fn an_authorisation_covers_only_its_exact_bytes() {
        let registry = Registry::new();
        let mut auth = OwnerSayAuth::new();
        auth.grant(&owner_say(false, "hello").payload());
        for (team, text) in [
            (1, "hello"),  // the other team flag
            (0, "hello!"), // longer
            (0, "hell"),
            (0, "Hello"),
            (0, "hello "),
            (0, "/kill "),
            (0, "bye"),
        ] {
            assert!(
                check_authorised(&hand_built_say(team, text), &registry, &mut auth).is_err(),
                "team {team} text {text:?}"
            );
        }
        assert_eq!(
            auth.pending(),
            1,
            "refusals of other bytes leave the authorisation alone"
        );
        assert!(check_authorised(&hand_built_say(0, "hello"), &registry, &mut auth).is_ok());
        assert_eq!(auth.pending(), 0);
    }

    /// Two lines authorised in one flush are each good once, in any order.
    #[test]
    fn several_authorisations_are_independent_and_revocable() {
        let registry = Registry::new();
        let (a, b) = (owner_say(false, "one").payload(), owner_say(true, "two").payload());
        let mut auth = OwnerSayAuth::new();
        auth.grant(&a);
        auth.grant(&b);
        assert!(check_authorised(b.as_bytes(), &registry, &mut auth).is_ok());
        assert!(check_authorised(b.as_bytes(), &registry, &mut auth).is_err());
        auth.revoke_all();
        assert_eq!(auth.pending(), 0);
        assert!(check_authorised(a.as_bytes(), &registry, &mut auth).is_err(), "revoked");
    }

    /// Defence in depth: even an authorisation entry for bytes that are not a valid owner line (it cannot be made from outside this
    /// module, a test builds it by hand) does not let them through.
    #[test]
    fn an_authorised_payload_must_still_be_a_canonical_owner_line() {
        let registry = Registry::new();
        let long = "x".repeat(300);
        for bytes in [
            hand_built_say(0, "/kill "),
            hand_built_say(0, "\u{200B}/w someone secret"),
            hand_built_say(0, " padded"),
            hand_built_say(0, ""),
            hand_built_say(0, "a\u{1}b"),
            hand_built_say(2, "hi"),
            hand_built_say(0, &long),
        ] {
            let mut auth = OwnerSayAuth {
                granted: vec![bytes.clone()],
            };
            assert!(check_authorised(&bytes, &registry, &mut auth).is_err(), "{bytes:?}");
            assert_eq!(auth.pending(), 0, "spent even though refused");
        }
    }

    /// An authorisation never opens any other message id: a granted payload with the system bit or another id is refused as before.
    #[test]
    fn an_authorisation_does_not_widen_anything_but_cl_say() {
        let registry = Registry::new();
        let payload = owner_say(false, "hello").payload();
        let mut auth = OwnerSayAuth::new();
        auth.grant(&payload);
        let mut sys = payload.as_bytes().to_vec();
        sys[0] |= 1;
        assert_eq!(
            check_authorised(&sys, &registry, &mut auth).unwrap_err(),
            GuardError::UnexpectedSystemMessage
        );
        let kill = numbered_payload(false, msgs::id::NETMSGTYPE_CL_KILL, |_| {});
        assert!(
            check_authorised(&kill, &registry, &mut auth).is_ok(),
            "Cl_Kill is as allowed as ever"
        );
        let vote = numbered_payload(false, msgs::id::NETMSGTYPE_CL_VOTE, |_| {});
        assert!(
            check_authorised(&vote, &registry, &mut auth).is_err(),
            "Cl_Vote is still not"
        );
        assert_eq!(auth.pending(), 1, "none of that touched the authorisation");
    }

    fn timeout_command(server: &str) -> TimeoutCommand {
        let seed = ddai_net::timeout_code::TimeoutSeed::parse("ABCDEFGHKLMNPRST").unwrap();
        TimeoutCommand::new(ddai_net::timeout_code::TimeoutCode::derive(
            &seed,
            server.parse().unwrap(),
        ))
    }

    /// Task 4.10 (D-100): the bot's own `/timeout <code>` passes once against its own authorisation, and a hand-built copy, a replay, or
    /// the same bytes without the authorisation are refused.
    #[test]
    fn the_timeout_command_passes_once_against_its_authorisation_and_never_without_it() {
        let registry = Registry::new();
        let payload = timeout_command("127.0.0.1:8443").payload();
        let mut owner = OwnerSayAuth::new();
        let mut auth = TimeoutAuth::new();
        // no authorisation: the very same bytes are a hand-built Cl_Say as far as the guard knows
        assert_eq!(
            check_with(payload.as_bytes(), &registry, &mut owner, &mut auth).unwrap_err(),
            GuardError::NumberedIdNotAllowed(msgs::id::NETMSGTYPE_CL_SAY)
        );
        assert!(
            check(payload.as_bytes(), &registry).is_err(),
            "the plain check has none"
        );
        assert!(check_authorised(payload.as_bytes(), &registry, &mut owner).is_err());
        auth.grant(&payload);
        assert_eq!(auth.pending(), 1);
        assert!(check_with(payload.as_bytes(), &registry, &mut owner, &mut auth).is_ok());
        assert_eq!(auth.pending(), 0, "consumed");
        assert!(
            check_with(payload.as_bytes(), &registry, &mut owner, &mut auth).is_err(),
            "a replay is refused"
        );
        // revocable
        auth.grant(&payload);
        auth.revoke_all();
        assert!(check_with(payload.as_bytes(), &registry, &mut owner, &mut auth).is_err());
    }

    /// The authorisation is for exact bytes: another code, another case, the other team flag, a trailing byte are not covered, and those
    /// refusals leave it alone.
    #[test]
    fn a_timeout_authorisation_covers_only_its_exact_bytes() {
        let registry = Registry::new();
        let mut owner = OwnerSayAuth::new();
        let mut auth = TimeoutAuth::new();
        auth.grant(&timeout_command("127.0.0.1:8443").payload());
        let other_code = timeout_command("127.0.0.1:8444").payload();
        assert_ne!(
            other_code.as_bytes(),
            timeout_command("127.0.0.1:8443").payload().as_bytes()
        );
        assert!(check_with(other_code.as_bytes(), &registry, &mut owner, &mut auth).is_err());
        for (team, text) in [
            (0, "/timeout"),
            (0, "/timeout KbCS2mj3DjD2YRE2 "),
            (0, "/Timeout KbCS2mj3DjD2YRE2"),
            (0, "/timeout KbCS2mj3DjD2YRE2;kill"),
            (1, "/timeout KbCS2mj3DjD2YRE2"),
            (0, "/kill "),
            (0, "hello"),
        ] {
            assert!(
                check_with(&hand_built_say(team, text), &registry, &mut owner, &mut auth).is_err(),
                "team {team} text {text:?}"
            );
        }
        let mut longer = timeout_command("127.0.0.1:8443").payload().into_bytes();
        longer.push(b'x');
        assert!(check_with(&longer, &registry, &mut owner, &mut auth).is_err());
        assert_eq!(
            auth.pending(),
            1,
            "refusals of other bytes leave the authorisation alone"
        );
        assert!(
            check_with(
                timeout_command("127.0.0.1:8443").payload().as_bytes(),
                &registry,
                &mut owner,
                &mut auth
            )
            .is_ok()
        );
        assert_eq!(auth.pending(), 0);
    }

    /// Defence in depth: an authorisation entry for bytes that are not a timeout command (a test builds it by hand; it cannot be made
    /// from outside this module) does not let them through, and is spent.
    #[test]
    fn an_authorised_payload_must_still_be_a_canonical_timeout_command() {
        let registry = Registry::new();
        for bytes in [
            hand_built_say(0, "/timeout"),
            hand_built_say(0, "/timeout x"),
            hand_built_say(0, "/timeout KbCS2mj3DjD2YRE2;kill"),
            hand_built_say(1, "/timeout KbCS2mj3DjD2YRE2"),
            hand_built_say(0, "hello"),
            hand_built_say(0, "/kill "),
        ] {
            let mut auth = TimeoutAuth {
                granted: Some(bytes.clone()),
            };
            assert!(
                check_with(&bytes, &registry, &mut OwnerSayAuth::new(), &mut auth).is_err(),
                "{bytes:?}"
            );
            assert_eq!(auth.pending(), 0, "spent even though refused");
        }
    }

    /// The two capabilities do not stand in for each other: an owner authorisation for `/timeout`-shaped bytes (hand-made here; the owner
    /// chat refuses to make one) does not pass as a timeout command, and a timeout authorisation does not pass an owner line.
    #[test]
    fn the_owner_authorisation_and_the_timeout_authorisation_are_separate() {
        let registry = Registry::new();
        let timeout = timeout_command("127.0.0.1:8443").payload();
        let mut owner = OwnerSayAuth {
            granted: vec![timeout.as_bytes().to_vec()],
        };
        assert!(
            check_with(timeout.as_bytes(), &registry, &mut owner, &mut TimeoutAuth::new()).is_err(),
            "an owner authorisation never lets a /timeout through (it is not an owner line)"
        );
        let say = owner_say(false, "hello").payload();
        let mut auth = TimeoutAuth::new();
        auth.grant(&timeout);
        assert!(check_with(say.as_bytes(), &registry, &mut OwnerSayAuth::new(), &mut auth).is_err());
        assert_eq!(
            auth.pending(),
            1,
            "an owner line's refusal does not touch the timeout authorisation"
        );
        // both pending: each line passes against its own
        let mut owner = OwnerSayAuth::new();
        owner.grant(&say);
        assert!(check_with(say.as_bytes(), &registry, &mut owner, &mut auth).is_ok());
        assert!(check_with(timeout.as_bytes(), &registry, &mut owner, &mut auth).is_ok());
        assert_eq!((owner.pending(), auth.pending()), (0, 0));
    }

    /// A timeout authorisation does not widen anything but `Cl_Say`: other ids and the system bit are decided as before.
    #[test]
    fn a_timeout_authorisation_does_not_widen_anything_but_cl_say() {
        let registry = Registry::new();
        let payload = timeout_command("127.0.0.1:8443").payload();
        let mut auth = TimeoutAuth::new();
        auth.grant(&payload);
        let mut sys = payload.as_bytes().to_vec();
        sys[0] |= 1;
        assert_eq!(
            check_with(&sys, &registry, &mut OwnerSayAuth::new(), &mut auth).unwrap_err(),
            GuardError::UnexpectedSystemMessage
        );
        let vote = numbered_payload(false, msgs::id::NETMSGTYPE_CL_VOTE, |_| {});
        assert!(check_with(&vote, &registry, &mut OwnerSayAuth::new(), &mut auth).is_err());
        assert_eq!(auth.pending(), 1, "none of that touched the authorisation");
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
