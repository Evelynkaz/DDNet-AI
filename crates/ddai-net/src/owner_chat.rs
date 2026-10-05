//! The second chat-channel message the bot may ever send: a line the **owner typed on the authenticated website** (task 4.9, D-094).
//!
//! D-007 stands for everything automatic: no auto-replies, no LLM, no periodic messages, no orders read from the game chat. The owner
//! allowed exactly two kinds of `Cl_Say`: the typed `/kill` fallback ([`crate::server_command::ServerCommand`], D-078) and this one.
//!
//! **The guarantee is in the types (a capability), and a census test backs it.**
//!
//! 1. [`OwnerText::new`] needs a `&`[`OwnerChannel`]: a capability that is not `Clone`, not `Copy`, cannot be built outside this module
//!    and is **minted once per process** ([`OwnerChannel::claim`] returns `Some` only the first time). The bot's control-socket
//!    dispatcher (`ddai-bot::control`) claims it and keeps it: that is the only place that turns a validated `say` request into an
//!    [`OwnerText`]. Code that only has to *judge* a text (the web unit, the protocol crate) uses [`OwnerText::check`], which returns no
//!    value. So nothing else in the process, however it gets hold of a string (game chat, a name, an LLM), can make an [`OwnerText`], and
//!    without one nothing can make an [`OwnerSay`], an [`OwnerPayload`] or an authorisation.
//! 2. [`OwnerSay`] (`team`, [`OwnerText`]) is the only thing that encodes into a `Cl_Say` payload: [`OwnerSay::payload`] returns an
//!    [`OwnerPayload`], a value only this crate can build. The generated `encode_cl_say` stays `pub(crate)`.
//! 3. The outgoing allow-list (`ddai-client::allowlist`) accepts a `Cl_Say` with free text only against a **one-shot authorisation**
//!    that the session recorded from an [`OwnerPayload`] for those exact bytes, and the bytes must also be the canonical encoding of an
//!    [`OwnerSay`] ([`OwnerSay::is_canonical`]). A hand-built `Cl_Say`, or a replay of an authorised one, is refused.
//!
//! The census test `ddai-bot/tests/owner_chat_census.rs` scans the workspace sources and fails when the call sites of these constructors
//! appear in a file outside a short allow-list: a new route to the chat has to be added there, in the open, in review.
//!
//! **What [`OwnerText::new`] checks** (the same rules back [`OwnerText::check`]): the text is trimmed (Rust whitespace and DDNet's
//! `str_utf8_isspace` set), not empty, at most [`MAX_OWNER_TEXT_BYTES`] bytes, free of control characters, line breaks, and of the
//! invisible and direction-changing format characters (zero-width and bidi marks, the word joiner, the BOM, the Hangul fillers), and is
//! not the one string 20.1 uses as a bot trap (`xd sure chillerbot.png is lyfe`).
//!
//! **A leading `/` is allowed (task 4.9b, the owner's decision of 2026-10-05, amending D-094).** D-094 first refused it so that no server
//! command could come from the site; the owner now wants to use the server's commands (`/spec`, `/pause`, `/emote`, `/w`, `/team`, ...)
//! from the website. Nothing else changed: only owner-typed lines, the capability, the one-shot authorisation, the pacing, the length
//! limit, the refusals above and the length-only logs. The text is still trimmed (DDNet's whitespace), so what is sent starts with the
//! `/` the owner typed; a `/` hidden behind an invisible character is refused as before (the invisible character is). The owner's `/kill`
//! is an owner line like any other (audit label `Cl_Say(owner)`); the bot's own typed `/kill` fallback (D-078) is a separate path
//! ([`crate::server_command::ServerCommand`], label `Cl_Say(/kill)`).
//!
//! The byte limit: 255 bytes is the smaller of DDNet 20.1's two. The server keeps `MAX_CHAT_LENGTH` (256) code points minus the
//! terminator (`gamecontext.cpp`, `OnSayNetMessage`: it cuts the line at the 256th code point); the client's chat box is a
//! `CLineInputBuffered<256>`, 255 bytes plus the NUL (`chat.h`, `lineinput.h`). 255 bytes can never be more than 255 code points, so the
//! byte limit is the binding one.
//!
//! ```compile_fail,E0603
//! // `encode_cl_say` is not public: there is no way to build a `Cl_Say` from a string outside this crate.
//! let mut buf = [0u8; 64];
//! let mut packer = ddai_net::packer::Packer::new(&mut buf);
//! let say = ddai_net::generated::messages::ClSay { team: 0, message: "hello".to_string() };
//! ddai_net::generated::messages::encode_cl_say(&say, &mut packer);
//! ```
//!
//! ```compile_fail,E0603
//! // `OwnerText` has a private field: it cannot be built without `OwnerText::new`'s checks.
//! let _ = ddai_net::owner_chat::OwnerText("hello".to_string());
//! ```
//!
//! ```compile_fail,E0603
//! // `OwnerPayload` has a private field too: authorisation cannot be granted for hand-built bytes.
//! let _ = ddai_net::owner_chat::OwnerPayload(vec![0u8; 4]);
//! ```
//!
//! ```compile_fail,E0603
//! // `OwnerChannel` cannot be built by hand: the only way to get one is `OwnerChannel::claim()`, once per process.
//! let _ = ddai_net::owner_chat::OwnerChannel(());
//! ```
//!
//! ```compile_fail,E0061
//! // An `OwnerText` cannot be constructed without the channel.
//! let _ = ddai_net::owner_chat::OwnerText::new("hello");
//! ```
//!
//! ```compile_fail,E0599
//! // The channel is not `Clone`: whoever holds it holds the only one.
//! let channel = ddai_net::owner_chat::OwnerChannel::claim().unwrap();
//! let _second = channel.clone();
//! ```

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::generated::messages::{self as msgs, ClSay};
use crate::packer::{Packer, Unpacker};
use crate::uuid::{MsgId, pack_msg_id};

/// The longest owner text, in UTF-8 bytes: the smaller of the server's and the client's limits (see the module docs).
pub const MAX_OWNER_TEXT_BYTES: usize = 255;

/// Room for the payload of the longest owner line: message id varint (1-2 bytes), the team varint, the text, the NUL.
const PAYLOAD_BUF: usize = 512;

/// The one line 20.1's `OnSayNetMessage` takes as a bot marker (the player's finishes stop counting): never said.
const SERVER_BOT_TRAP: &str = "xd sure chillerbot.png is lyfe";

/// Set when the process's one [`OwnerChannel`] has been handed out.
static CLAIMED: AtomicBool = AtomicBool::new(false);

/// The capability to make an [`OwnerText`]: the right to put a line in the bot's mouth. Not `Clone`, not `Copy`, not constructible
/// outside this module. There is **one per process**: [`OwnerChannel::claim`] returns it the first time and `None` afterwards, and the
/// bot's control-socket dispatcher is the claimant (`ddai-bot::control::ControlServer::start`). Holding it is the proof of being that
/// dispatcher. Its `Debug` shows nothing.
pub struct OwnerChannel(());

impl OwnerChannel {
    /// The process's one channel, or `None` when it has been claimed already.
    pub fn claim() -> Option<OwnerChannel> {
        CLAIMED
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .ok()
            .map(|_| OwnerChannel(()))
    }

    /// A channel for a test, outside the once-per-process rule (tests build many dispatchers in one process). Exists only in this
    /// crate's own tests and behind the `test-util` feature, which only dev-dependencies turn on; the census test forbids it in
    /// production sources.
    #[cfg(any(test, feature = "test-util"))]
    pub fn mint_for_tests() -> OwnerChannel {
        OwnerChannel(())
    }
}

impl fmt::Debug for OwnerChannel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OwnerChannel")
    }
}

/// Why [`OwnerText::new`] / [`OwnerText::check`] refused a text. Never carries the text itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum OwnerTextError {
    #[error("the message is empty")]
    Empty,
    #[error("the message is longer than {MAX_OWNER_TEXT_BYTES} bytes")]
    TooLong,
    #[error("the message has a control character, a line break or an invisible formatting character")]
    Control,
    #[error("the message is one the server treats as a bot marker")]
    Reserved,
}

/// A chat line the owner typed. Only [`OwnerText::new`] makes one, and only for the holder of the [`OwnerChannel`].
#[derive(Clone, PartialEq, Eq)]
pub struct OwnerText(String);

/// Characters that end a line without being `char::is_control` (Unicode line and paragraph separators).
fn is_line_break(c: char) -> bool {
    matches!(c, '\u{2028}' | '\u{2029}')
}

/// Invisible and direction-changing characters (category Cf and the Hangul fillers) that would let a line look like something it is not
/// (a hidden `/kill`, a reordered display) or hide behind DDNet's own idea of "space".
fn is_invisible_format(c: char) -> bool {
    matches!(c,
        '\u{200B}'..='\u{200F}'
        | '\u{202A}'..='\u{202E}'
        | '\u{2060}'..='\u{2064}'
        | '\u{2066}'..='\u{2069}'
        | '\u{FEFF}'
        | '\u{180E}'
        | '\u{115F}' | '\u{1160}' | '\u{3164}' | '\u{FFA0}')
}

/// DDNet 20.1's `str_utf8_isspace` (`base/str.cpp`): what the server calls a space when it trims a line and what a client that skips
/// leading whitespace skips.
fn is_ddnet_space(c: char) -> bool {
    let code = u32::from(c);
    code <= 0x0020
        || matches!(
            code,
            0x0085 | 0x00A0 | 0x034F | 0x115F | 0x1160 | 0x1680 | 0x180E | 0x2800 | 0x3000 | 0x3164 | 0xFEFF | 0xFFA0
        )
        || (0x2000..=0x200F).contains(&code)
        || (0x2028..=0x202F).contains(&code)
        || (0x205F..=0x2064).contains(&code)
        || (0x206A..=0x206F).contains(&code)
        || (0xFE00..=0xFE0F).contains(&code)
        || (0xFFF9..=0xFFFC).contains(&code)
}

/// What the trim cuts: Rust whitespace, and DDNet's spaces other than the control codes (a stray `\u{1}` at the end is not "space", it is
/// refused).
fn is_space(c: char) -> bool {
    c.is_whitespace() || (is_ddnet_space(c) && !c.is_control())
}

/// The rules, shared by [`OwnerText::new`], [`OwnerText::check`] and the allow-list's canonical-form test. Returns the trimmed text.
fn validate(raw: &str) -> Result<&str, OwnerTextError> {
    // Invisible formatting is refused anywhere, also at the ends where the trim below would hide it.
    if raw.chars().any(is_invisible_format) {
        return Err(OwnerTextError::Control);
    }
    let text = raw.trim_matches(is_space);
    if text.is_empty() {
        return Err(OwnerTextError::Empty);
    }
    if text.len() > MAX_OWNER_TEXT_BYTES {
        return Err(OwnerTextError::TooLong);
    }
    if text.chars().any(|c| c.is_control() || is_line_break(c)) {
        return Err(OwnerTextError::Control);
    }
    if text == SERVER_BOT_TRAP {
        return Err(OwnerTextError::Reserved);
    }
    Ok(text)
}

impl OwnerText {
    /// Checks and normalises `raw` (see the module docs for the rules) and makes the text, for the holder of the [`OwnerChannel`].
    pub fn new(_channel: &OwnerChannel, raw: &str) -> Result<OwnerText, OwnerTextError> {
        validate(raw).map(|t| OwnerText(t.to_string()))
    }

    /// Whether `raw` would pass [`OwnerText::new`]. Makes no value: the web unit and the protocol crate use it to refuse early without
    /// ever holding a sendable text.
    pub fn check(raw: &str) -> Result<(), OwnerTextError> {
        validate(raw).map(|_| ())
    }

    /// [`OwnerText::check`] that also shows the trimmed text it judged, as a plain `&str` (not a sendable value), so the web unit can
    /// pass on the short form instead of the padded one.
    pub fn normalise(raw: &str) -> Result<&str, OwnerTextError> {
        validate(raw)
    }

    /// The text, for the owner's own screen and for encoding. Never log it.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Length in UTF-8 bytes (the only thing that may be logged about it).
    #[allow(clippy::len_without_is_empty)] // an `OwnerText` is never empty
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Length in code points (DDNet's own server-side spam rule counts these).
    pub fn char_count(&self) -> usize {
        self.0.chars().count()
    }
}

impl fmt::Debug for OwnerText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OwnerText(len {})", self.0.len())
    }
}

/// A `Cl_Say` the owner asked for: all-chat (`team == false`) or the team's chat.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnerSay {
    pub team: bool,
    pub text: OwnerText,
}

/// The wire payload of an [`OwnerSay`] (`(NETMSGTYPE_CL_SAY << 1)` varint, team, the NUL-terminated text), as
/// `Connection::send_chunk` takes it. Only [`OwnerSay::payload`] builds one: holding an `OwnerPayload` is the proof that the bytes came
/// from a validated [`OwnerText`], and it is what a session records an authorisation from. `Debug` hides the bytes.
#[derive(Clone, PartialEq, Eq)]
pub struct OwnerPayload(Vec<u8>);

impl OwnerPayload {
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.0
    }
}

impl fmt::Debug for OwnerPayload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OwnerPayload({} bytes)", self.0.len())
    }
}

impl OwnerSay {
    pub fn new(team: bool, text: OwnerText) -> OwnerSay {
        OwnerSay { team, text }
    }

    /// The payload. Fails closed: should the buffer ever not fit (it always does for a valid [`OwnerText`]), the payload is empty and
    /// the allow-list refuses it as undecodable.
    pub fn payload(&self) -> OwnerPayload {
        let mut buf = [0u8; PAYLOAD_BUF];
        let mut packer = Packer::new(&mut buf);
        pack_msg_id(&mut packer, MsgId::Numbered(msgs::id::NETMSGTYPE_CL_SAY), false);
        msgs::encode_cl_say(
            &ClSay {
                team: i32::from(self.team),
                message: self.text.0.clone(),
            },
            &mut packer,
        );
        if packer.error() {
            return OwnerPayload(Vec::new());
        }
        OwnerPayload(packer.data().to_vec())
    }

    /// The `OwnerSay` that `payload` encodes, when it is byte for byte the canonical encoding of one. Private on purpose: it makes an
    /// [`OwnerText`] out of bytes, which only a yes/no ([`OwnerSay::is_canonical`]) may be shown to the rest of the world.
    fn decode(payload: &[u8]) -> Option<OwnerSay> {
        let mut u = Unpacker::new(payload);
        if u.get_int() != (msgs::id::NETMSGTYPE_CL_SAY << 1) || u.error() {
            return None;
        }
        let say = msgs::decode_cl_say(&mut u)?;
        let text = OwnerText(validate(&say.message).ok()?.to_string());
        let candidate = OwnerSay {
            team: say.team == 1,
            text,
        };
        // Re-encoding must give the very same bytes: this rejects sanitised control bytes, untrimmed text, trailing bytes, a
        // non-minimal varint, and anything else that is not exactly what `payload` writes.
        (candidate.payload().0 == payload).then_some(candidate)
    }

    /// Whether `payload` is, byte for byte, the canonical encoding of an [`OwnerSay`]: a non-system `Cl_Say` with team 0 or 1 and a text
    /// that passes the [`OwnerText`] rules unchanged (already trimmed), nothing before, between or after. This is the structural half of
    /// the allow-list's check; the other half is the one-shot authorisation. A yes/no only: it hands out no value.
    pub fn is_canonical(payload: &[u8]) -> bool {
        OwnerSay::decode(payload).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests mint their own channels: the process's one real channel belongs to the control dispatcher.
    fn new(raw: &str) -> Result<OwnerText, OwnerTextError> {
        OwnerText::new(&OwnerChannel::mint_for_tests(), raw)
    }

    fn say_bytes(team: i32, text: &str) -> Vec<u8> {
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
    fn plain_text_is_accepted_and_trimmed() {
        assert_eq!(new("hello").unwrap().as_str(), "hello");
        assert_eq!(new("  hello world \t").unwrap().as_str(), "hello world");
        assert_eq!(new("привет, мир").unwrap().as_str(), "привет, мир");
        assert_eq!(
            new("gg\n").unwrap().as_str(),
            "gg",
            "a trailing newline is whitespace, trimmed"
        );
        assert_eq!(new("a  b").unwrap().as_str(), "a  b", "inner spaces stay");
        assert_eq!(new("hi /kill").unwrap().as_str(), "hi /kill");
        assert_eq!(new("!kill").unwrap().as_str(), "!kill");
    }

    #[test]
    fn empty_and_blank_are_refused() {
        for s in ["", " ", "   \t  ", "\n", "\u{a0}", "\u{3000}"] {
            assert_eq!(new(s).unwrap_err(), OwnerTextError::Empty, "{s:?}");
        }
    }

    #[test]
    fn the_length_limit_is_255_bytes_not_characters() {
        let at_limit = "a".repeat(MAX_OWNER_TEXT_BYTES);
        assert_eq!(new(&at_limit).unwrap().len(), MAX_OWNER_TEXT_BYTES);
        assert_eq!(
            new(&"a".repeat(MAX_OWNER_TEXT_BYTES + 1)).unwrap_err(),
            OwnerTextError::TooLong
        );
        // 127 two-byte characters = 254 bytes: fine; 128 = 256 bytes: too long although only 128 characters.
        assert!(new(&"я".repeat(127)).is_ok());
        assert_eq!(new(&"я".repeat(128)).unwrap_err(), OwnerTextError::TooLong);
        // whitespace around the text does not count, and cannot push a short text over the limit
        let padded = format!("{}{}{}", " ".repeat(500), "ok", " ".repeat(500));
        assert_eq!(new(&padded).unwrap().as_str(), "ok");
        // 255 bytes of 4-byte characters (63 of them = 252 bytes) fits; the encoding of the longest line fits the buffer
        let longest = new(&"😀".repeat(63)).unwrap();
        assert!(OwnerSay::new(false, longest).payload().as_bytes().len() > 250);
        let at_limit = new(&at_limit).unwrap();
        let payload = OwnerSay::new(true, at_limit).payload();
        assert!(!payload.as_bytes().is_empty(), "the longest line encodes");
    }

    #[test]
    fn control_characters_and_line_breaks_are_refused() {
        for s in [
            "a\nb",
            "a\rb",
            "a\tb",
            "a\0b",
            "\u{1}a",
            "a\u{7f}",
            "a\u{85}b", // C1: NEL
            "a\u{9b}b",
            "a\u{2028}b",
            "a\u{2029}b",
            "line1\nline2",
            "a\u{1b}[31mred",
        ] {
            assert_eq!(new(s).unwrap_err(), OwnerTextError::Control, "{s:?}");
        }
    }

    /// Task 4.9b: a leading `/` is allowed (the owner's decision of 2026-10-05): server commands from the site, text unchanged.
    #[test]
    fn a_leading_slash_is_accepted_and_the_text_is_kept_as_typed() {
        for s in [
            "/spec",
            "/pause",
            "/emote happy",
            "/w Name hi",
            "/team 1",
            "/me waves",
            "/rank",
            "/kill",
            "/",
            "//kill",
            "/ kill",
        ] {
            assert_eq!(new(s).unwrap().as_str(), s, "{s:?}");
        }
        // the trim is DDNet's: leading and trailing spaces go, the slash is the first character that is sent
        assert_eq!(new("  /kill").unwrap().as_str(), "/kill");
        assert_eq!(new("\t/w someone hi  ").unwrap().as_str(), "/w someone hi");
        assert_eq!(new("\n/spec").unwrap().as_str(), "/spec");
        assert_eq!(new("\u{3000}/pause").unwrap().as_str(), "/pause");
        // every command is still a team-flag line and still goes through the length limit
        let long = format!("/w x {}", "a".repeat(MAX_OWNER_TEXT_BYTES));
        assert_eq!(new(&long).unwrap_err(), OwnerTextError::TooLong);
        assert_eq!(new("/w x ").unwrap().as_str(), "/w x");
    }

    /// A command is refused for the same reasons any other line is: control and invisible characters (also hiding in front of the
    /// slash), line breaks inside it, and the trap string. A slash does not make any of these acceptable.
    #[test]
    fn a_command_is_refused_for_the_reasons_any_line_is() {
        for s in [
            "\u{200B}/kill",
            "\u{FEFF}/spec",
            "\u{202E}/pause",
            "\u{2060}/w x y",
            "\u{3164}/kill",
            "/ki\u{200B}ll",
            "/kill\u{200B}",
            "/w x\nhi",
            "/w x\r\n/kill",
            "/emote\0",
            "/kill\u{1b}[31m",
            "/a\u{2028}b",
        ] {
            assert_eq!(new(s).unwrap_err(), OwnerTextError::Control, "{s:?}");
        }
        assert_eq!(new("/").unwrap().len(), 1);
        assert_eq!(new("  /  ").unwrap().as_str(), "/");
        assert_eq!(new("   ").unwrap_err(), OwnerTextError::Empty);
        assert_eq!(
            new(&format!("/{}", "a".repeat(MAX_OWNER_TEXT_BYTES))).unwrap_err(),
            OwnerTextError::TooLong
        );
        assert_eq!(
            new(&format!("/{}", "a".repeat(MAX_OWNER_TEXT_BYTES - 1)))
                .unwrap()
                .len(),
            MAX_OWNER_TEXT_BYTES
        );
    }

    #[test]
    fn debug_never_shows_the_text() {
        let t = new("a secret line").unwrap();
        assert_eq!(format!("{t:?}"), "OwnerText(len 13)");
        let s = OwnerSay::new(true, t);
        let shown = format!("{s:?}");
        assert!(!shown.contains("secret"), "{shown}");
        let p = format!("{:?}", s.payload());
        assert!(!p.contains("secret") && p == "OwnerPayload(16 bytes)", "{p}");
    }

    /// The channel is minted once per process: the first `claim()` is `Some`, every later one `None`. (This is the only test in this
    /// crate's test binary that claims; the production claimant is the control dispatcher of `ddai-bot`.)
    #[test]
    fn the_channel_can_be_claimed_only_once() {
        {
            let first = OwnerChannel::claim();
            assert!(first.is_some(), "nobody else claimed it in this test process");
            assert!(OwnerChannel::claim().is_none(), "a second claim gets nothing");
            assert!(OwnerChannel::claim().is_none());
        }
        assert!(
            OwnerChannel::claim().is_none(),
            "the channel going out of scope does not give it back"
        );
        // tests mint their own, outside the rule
        assert!(new("still works").is_ok());
        assert_eq!(format!("{:?}", OwnerChannel::mint_for_tests()), "OwnerChannel");
    }

    /// `check` is `new` without the value: it agrees with it on every input and holds nothing.
    #[test]
    fn check_agrees_with_new() {
        for s in [
            "hello",
            "",
            "   ",
            "/kill",
            "a\nb",
            "\u{200B}x",
            "xd sure chillerbot.png is lyfe",
            "\u{2800}/w x",
            "ok \u{3164}",
            "я",
            "gg\n",
        ] {
            assert_eq!(OwnerText::check(s), new(s).map(|_| ()), "{s:?}");
        }
        assert_eq!(OwnerText::check(&"a".repeat(256)), Err(OwnerTextError::TooLong));
    }

    /// F3: invisible formatting characters are refused anywhere, also where the trim would hide them.
    #[test]
    fn invisible_and_direction_characters_are_refused_anywhere() {
        let mut bad: Vec<char> = Vec::new();
        bad.extend('\u{200B}'..='\u{200F}');
        bad.extend('\u{202A}'..='\u{202E}');
        bad.extend('\u{2060}'..='\u{2064}');
        bad.extend('\u{2066}'..='\u{2069}');
        bad.extend(['\u{FEFF}', '\u{180E}', '\u{115F}', '\u{1160}', '\u{3164}', '\u{FFA0}']);
        for c in bad {
            for s in [
                format!("{c}/kill"),
                format!("{c}/w E2eSay hi"),
                format!("hi{c}there"),
                format!("hi{c}"),
                format!("{c}hi"),
                format!("{c}"),
            ] {
                assert_eq!(
                    new(&s).unwrap_err(),
                    OwnerTextError::Control,
                    "U+{:04X} in {s:?}",
                    u32::from(c)
                );
            }
        }
        // harmless look-alikes and combining marks stay allowed (a fullwidth slash is not a command)
        assert!(new("／kill").is_ok());
        assert!(new("e\u{301}").is_ok());
        assert!(new("😀 ok").is_ok());
    }

    /// F3 (4.9b): DDNet's own spaces in front of a command are trimmed, so the line that is sent starts with the `/`; trailing ones are
    /// cut as the server cuts them. Nothing hides behind them: a space-only line is empty.
    #[test]
    fn ddnet_spaces_around_a_command_are_trimmed() {
        for c in [
            '\u{0085}', '\u{00A0}', '\u{034F}', '\u{1680}', '\u{2000}', '\u{200A}', '\u{2028}', '\u{202F}', '\u{205F}',
            '\u{206A}', '\u{206F}', '\u{2800}', '\u{3000}', '\u{FE00}', '\u{FE0F}', '\u{FFF9}', '\u{FFFC}', ' ', '\t',
            '\n',
        ] {
            for (raw, want) in [
                (format!("{c}/kill"), "/kill"),
                (format!("{c}{c}/w someone hi"), "/w someone hi"),
                (format!(" {c} /"), "/"),
                (format!("/spec{c}"), "/spec"),
            ] {
                assert_eq!(new(&raw).unwrap().as_str(), want, "U+{:04X} in {raw:?}", u32::from(c));
            }
            // and trailing ones are cut, as the server cuts them
            assert_eq!(new(&format!("gg{c}")).unwrap().as_str(), "gg", "U+{:04X}", u32::from(c));
        }
        // only spaces: nothing left to say
        for c in ['\u{2800}', '\u{034F}', '\u{FE0F}', '\u{FFFC}'] {
            assert_eq!(
                new(&format!("{c}{c}")).unwrap_err(),
                OwnerTextError::Empty,
                "U+{:04X}",
                u32::from(c)
            );
        }
    }

    /// N3: the one string 20.1 takes as a bot marker is never said (in any chat, with or without padding).
    #[test]
    fn the_servers_bot_trap_line_is_refused() {
        for s in [
            "xd sure chillerbot.png is lyfe",
            "  xd sure chillerbot.png is lyfe\n",
            "\u{3000}xd sure chillerbot.png is lyfe",
        ] {
            assert_eq!(new(s).unwrap_err(), OwnerTextError::Reserved, "{s:?}");
        }
        assert!(
            new("xd sure chillerbot.png is lyfe!").is_ok(),
            "only the exact line is the trap"
        );
        assert!(new("XD sure chillerbot.png is lyfe").is_ok());
    }

    #[test]
    fn the_payload_is_a_cl_say_with_the_team_flag_and_the_exact_text() {
        for (team, flag) in [(false, 0u8), (true, 1u8)] {
            let say = OwnerSay::new(team, new("gg wp").unwrap());
            let payload = say.payload();
            let bytes = payload.as_bytes();
            let mut u = Unpacker::new(bytes);
            assert_eq!(
                u.get_int(),
                msgs::id::NETMSGTYPE_CL_SAY << 1,
                "game namespace, numbered id"
            );
            let decoded = msgs::decode_cl_say(&mut u).expect("decodes");
            assert_eq!(decoded.team, i32::from(flag));
            assert_eq!(decoded.message, "gg wp");
            assert!(!u.error());
            // the tail of the wire bytes, spelled out: team, "gg wp", NUL
            let mut tail = vec![flag];
            tail.extend_from_slice(b"gg wp");
            tail.push(0);
            assert!(bytes.ends_with(&tail), "{bytes:?}");
            assert_eq!(OwnerSay::decode(bytes), Some(say));
        }
    }

    #[test]
    fn is_canonical_takes_only_the_canonical_encoding() {
        let ok = OwnerSay::new(false, new("hello").unwrap()).payload().into_bytes();
        assert!(OwnerSay::is_canonical(&ok));
        // a command is a canonical owner line too (4.9b), in either chat
        for (team, text) in [(0, "/spec"), (0, "/emote happy"), (1, "/w Name hi"), (0, "/kill")] {
            assert!(OwnerSay::is_canonical(&say_bytes(team, text)), "{text}");
        }
        // what a hand-built Cl_Say could look like instead
        let refused: Vec<(&str, Vec<u8>)> = vec![
            ("untrimmed command", say_bytes(0, " /kill")),
            ("untrimmed command end", say_bytes(1, "/w x y ")),
            ("hidden command", say_bytes(0, "\u{200B}/kill")),
            ("the bot trap", say_bytes(0, SERVER_BOT_TRAP)),
            ("untrimmed", say_bytes(0, " hello")),
            ("untrimmed end", say_bytes(0, "hello ")),
            ("empty", say_bytes(0, "")),
            ("newline", say_bytes(0, "a\nb")),
            ("control", say_bytes(0, "a\u{1}b")),
            ("team 2", say_bytes(2, "hello")),
            ("team -1", say_bytes(-1, "hello")),
            ("too long", say_bytes(0, &"x".repeat(MAX_OWNER_TEXT_BYTES + 1))),
        ];
        for (what, bytes) in &refused {
            assert!(!OwnerSay::is_canonical(bytes), "{what}");
        }
        let mut trailing = ok.clone();
        trailing.push(0);
        assert!(!OwnerSay::is_canonical(&trailing), "a trailing byte");
        let mut cut = ok.clone();
        cut.pop();
        assert!(!OwnerSay::is_canonical(&cut), "the NUL missing");
        let mut sys = ok.clone();
        sys[0] |= 1;
        assert!(!OwnerSay::is_canonical(&sys), "the system bit");
        let mut other_id = ok.clone();
        other_id[0] = (msgs::id::NETMSGTYPE_CL_KILL << 1) as u8;
        assert!(!OwnerSay::is_canonical(&other_id), "another message id");
        assert!(!OwnerSay::is_canonical(&[]));
        assert!(!OwnerSay::is_canonical(&[0xff; 8]));
    }

    /// 4.9b: the owner may type `/kill`, and its wire bytes are the very bytes of the bot's own `/kill` fallback (D-078): the two are told
    /// apart by the path (the type that made the payload and the audit label `Cl_Say(owner)` / `Cl_Say(/kill)`), not by the bytes.
    #[test]
    fn the_owners_slash_kill_has_the_fallbacks_bytes_but_is_made_apart() {
        let fallback = crate::server_command::ServerCommand::Kill.payload();
        let owner = OwnerSay::new(false, new("/kill").unwrap()).payload().into_bytes();
        assert_eq!(owner, fallback, "the same Cl_Say bytes");
        assert!(OwnerSay::is_canonical(&fallback));
        assert_eq!(
            crate::server_command::ServerCommand::recognise(&owner),
            Some(crate::server_command::ServerCommand::Kill)
        );
        // with the team flag it is an owner line only
        let team_kill = OwnerSay::new(true, new("/kill").unwrap()).payload().into_bytes();
        assert_ne!(team_kill, fallback);
        assert_eq!(crate::server_command::ServerCommand::recognise(&team_kill), None);
        assert!(OwnerSay::is_canonical(&team_kill));
    }

    /// Whatever bytes arrive, `decode` never panics, and what it accepts re-encodes to itself.
    #[test]
    fn decode_is_total_over_arbitrary_bytes() {
        let mut x: u64 = 0x1234_5678_9abc_def0;
        for round in 0..20_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let len = (x % 40) as usize;
            let mut bytes: Vec<u8> = (0..len)
                .map(|i| (x.rotate_left((i * 7) as u32 % 63) & 0xff) as u8)
                .collect();
            if round % 2 == 0 && !bytes.is_empty() {
                bytes[0] = (msgs::id::NETMSGTYPE_CL_SAY << 1) as u8;
            }
            if let Some(say) = OwnerSay::decode(&bytes) {
                assert_eq!(say.payload().as_bytes(), &bytes[..]);
            }
        }
    }
}
