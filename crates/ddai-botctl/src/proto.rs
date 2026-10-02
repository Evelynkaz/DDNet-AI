//! The control-channel protocol v1 (`docs/formats.md` §26): what the web unit may ask of the running bot.
//!
//! **Transport.** A Unix stream socket (the bot's `control.sock`, mode `0600` in a `0700` directory), one JSON
//! document per line (`\n`-terminated, at most [`MAX_REQUEST_BYTES`] / [`MAX_REPLY_BYTES`] including the newline):
//! a [`ControlRequest`] from the web, one [`ControlReply`] back, strictly in turn.
//!
//! **No chat, by construction.** [`ControlCommand`] is a closed enum. Not one variant carries text to the game
//! server; the only free text is [`ControlCommand::Clip`]'s note, which goes into a clip *file* (and its name) on
//! the bot's disk. The bot maps a command to its own `BotCommand` (the 4.3 console API), which has no way to reach
//! the chat either. Unknown fields and unknown command types are refused at parse time (`deny_unknown_fields`), so
//! a newer or hostile client cannot smuggle anything in.
//!
//! **No nicknames.** Nothing in a request or in [`ControlCommand::tag`] is a nickname, which is what the audit log
//! relies on. The friend / war / ignore lists are not sent through this channel at all: the web edits the lists
//! file and sends [`ControlCommand::ReloadRelations`]; the bot reads the file itself.

use serde::{Deserialize, Serialize};

/// Protocol version in every request and reply.
pub const VERSION: u32 = 1;
/// Longest request line the bot reads (newline included): a clip note is capped far below this.
pub const MAX_REQUEST_BYTES: usize = 2048;
/// Longest reply line the web reads (newline included). Reply texts are clipped to [`MAX_REPLY_TEXT_CHARS`].
pub const MAX_REPLY_BYTES: usize = 8192;
/// Longest reply text, in characters.
pub const MAX_REPLY_TEXT_CHARS: usize = 600;
/// Longest clip note, in characters (it ends up in a file name that is meant to be shared).
pub const MAX_NOTE_CHARS: usize = 60;
/// `goto` tile coordinates are accepted within +-this (DDNet maps are far smaller).
pub const GOTO_LIMIT: i32 = 100_000;
/// Longest session tag.
pub const MAX_SESSION_CHARS: usize = 32;

/// `fight` / `passive` / `hold`. (`goto` is a state the navigation enters by itself, never asked for.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModeArg {
    Fight,
    Passive,
    Hold,
}

impl ModeArg {
    pub fn name(self) -> &'static str {
        match self {
            ModeArg::Fight => "fight",
            ModeArg::Passive => "passive",
            ModeArg::Hold => "hold",
        }
    }
}

/// The wayblock mode: `auto` / `left` / `right` / `off`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WbArg {
    Auto,
    Left,
    Right,
    Off,
}

impl WbArg {
    pub fn name(self) -> &'static str {
        match self {
            WbArg::Auto => "auto",
            WbArg::Left => "left",
            WbArg::Right => "right",
            WbArg::Off => "off",
        }
    }
}

/// The brains the bot can swap to live.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BrainArg {
    Hybrid,
    Planner,
    Scripted,
    Idle,
    Fly,
}

impl BrainArg {
    pub fn name(self) -> &'static str {
        match self {
            BrainArg::Hybrid => "hybrid",
            BrainArg::Planner => "planner",
            BrainArg::Scripted => "scripted",
            BrainArg::Idle => "idle",
            BrainArg::Fly => "fly",
        }
    }
}

/// Everything the web may ask. Closed: see the module docs. (The argument-less commands are `{}` variants, not unit
/// variants, because serde's `deny_unknown_fields` is not applied to unit variants of an internally tagged enum.)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ControlCommand {
    /// `!mode fight|passive|hold`.
    Mode { mode: ModeArg },
    /// `!stop`: stand still (or end a walk).
    Stop {},
    /// `!go`: play again.
    Go {},
    /// `!wb auto|left|right|off`.
    Wb { mode: WbArg },
    /// `!brain ...`: swap the brain live.
    Brain { brain: BrainArg },
    /// `!kill`: kill and respawn (the bot's own 500-tick cooldown applies).
    Kill {},
    /// `!clip [note]`: save the last 30 s. The note goes into the clip file and its name.
    Clip { note: String },
    /// `!goto <x> <y>` (tiles).
    Goto { x: i32, y: i32 },
    /// `!spec`: go to the spectators.
    Spec {},
    /// `!join`: back into the game.
    Join {},
    /// Re-read the friend / war / ignore lists file the web editor has just written.
    ReloadRelations {},
}

/// Why a request is not acceptable (a static text: it never echoes the request).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum Invalid {
    #[error("unsupported protocol version")]
    Version,
    #[error("bad session tag")]
    Session,
    #[error("the clip note is too long or has control characters")]
    Note,
    #[error("goto coordinates out of range")]
    Goto,
}

impl ControlCommand {
    /// A short, fixed-shape label for logs and the audit trail: the command and its closed-set argument, **never**
    /// free text (a clip's note is not in it) and never a nickname (there is none to put in).
    pub fn tag(&self) -> String {
        match self {
            ControlCommand::Mode { mode } => format!("mode:{}", mode.name()),
            ControlCommand::Stop {} => "stop".to_string(),
            ControlCommand::Go {} => "go".to_string(),
            ControlCommand::Wb { mode } => format!("wb:{}", mode.name()),
            ControlCommand::Brain { brain } => format!("brain:{}", brain.name()),
            ControlCommand::Kill {} => "kill".to_string(),
            ControlCommand::Clip { .. } => "clip".to_string(),
            ControlCommand::Goto { x, y } => format!("goto:{x},{y}"),
            ControlCommand::Spec {} => "spec".to_string(),
            ControlCommand::Join {} => "join".to_string(),
            ControlCommand::ReloadRelations {} => "relations:reload".to_string(),
        }
    }

    /// Checks the arguments the type system cannot.
    pub fn validate(&self) -> Result<(), Invalid> {
        match self {
            ControlCommand::Clip { note } => {
                if note.chars().count() > MAX_NOTE_CHARS || note.chars().any(char::is_control) {
                    return Err(Invalid::Note);
                }
            }
            ControlCommand::Goto { x, y } => {
                let range = -GOTO_LIMIT..=GOTO_LIMIT;
                if !range.contains(x) || !range.contains(y) {
                    return Err(Invalid::Goto);
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// A request line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlRequest {
    pub v: u32,
    /// An opaque tag of the web session that asked (lower-case hex, see [`valid_session_tag`]): for the audit log only. It is
    /// **not** the session cookie and cannot be turned back into it.
    pub session: String,
    pub cmd: ControlCommand,
}

/// Whether `s` is an acceptable session tag: 1..=[`MAX_SESSION_CHARS`] lower-case hex digits.
pub fn valid_session_tag(s: &str) -> bool {
    !s.is_empty() && s.len() <= MAX_SESSION_CHARS && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl ControlRequest {
    pub fn new(session: impl Into<String>, cmd: ControlCommand) -> ControlRequest {
        ControlRequest {
            v: VERSION,
            session: session.into(),
            cmd,
        }
    }

    pub fn validate(&self) -> Result<(), Invalid> {
        if self.v != VERSION {
            return Err(Invalid::Version);
        }
        if !valid_session_tag(&self.session) {
            return Err(Invalid::Session);
        }
        self.cmd.validate()
    }
}

/// Why a reply is not a plain answer from the bot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplyCode {
    /// The request did not parse or did not validate.
    BadRequest,
    /// Too many commands lately.
    RateLimited,
    /// Too many open connections to the control socket.
    Busy,
    /// The bot did not answer in time (it answers between two snapshots).
    Timeout,
    /// The bot is stopping.
    Gone,
}

/// A reply line: the bot's `CommandReply` (the 4.3 console API) in wire form, or a refusal with a [`ReplyCode`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlReply {
    pub v: u32,
    /// The command was understood and done (or queued).
    pub ok: bool,
    /// The bot's answer, for the owner's screen only (the bot's reply texts name other players by tag).
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<ReplyCode>,
    /// Structured detail of the answer, for commands that have some. Today only `reload_relations`:
    /// `{"counts":{"friend":n,...},"digest":"<16 hex>"}` — counts and a fingerprint of the lists the bot now holds
    /// (`Relations::digest`), never a name. The web compares the digest with what it wrote to tell whether the bot
    /// reloaded the same file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl ControlReply {
    pub fn answer(ok: bool, text: &str) -> ControlReply {
        ControlReply {
            v: VERSION,
            ok,
            text: text.chars().take(MAX_REPLY_TEXT_CHARS).collect(),
            code: None,
            data: None,
        }
    }

    pub fn refused(code: ReplyCode, text: &str) -> ControlReply {
        ControlReply {
            v: VERSION,
            ok: false,
            text: text.chars().take(MAX_REPLY_TEXT_CHARS).collect(),
            code: Some(code),
            data: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_commands() -> Vec<ControlCommand> {
        vec![
            ControlCommand::Mode { mode: ModeArg::Fight },
            ControlCommand::Mode { mode: ModeArg::Passive },
            ControlCommand::Mode { mode: ModeArg::Hold },
            ControlCommand::Stop {},
            ControlCommand::Go {},
            ControlCommand::Wb { mode: WbArg::Auto },
            ControlCommand::Wb { mode: WbArg::Left },
            ControlCommand::Wb { mode: WbArg::Right },
            ControlCommand::Wb { mode: WbArg::Off },
            ControlCommand::Brain {
                brain: BrainArg::Hybrid,
            },
            ControlCommand::Brain {
                brain: BrainArg::Planner,
            },
            ControlCommand::Brain {
                brain: BrainArg::Scripted,
            },
            ControlCommand::Brain { brain: BrainArg::Idle },
            ControlCommand::Brain { brain: BrainArg::Fly },
            ControlCommand::Kill {},
            ControlCommand::Clip {
                note: "nice save".into(),
            },
            ControlCommand::Goto { x: -3, y: 40 },
            ControlCommand::Spec {},
            ControlCommand::Join {},
            ControlCommand::ReloadRelations {},
        ]
    }

    #[test]
    fn every_command_round_trips_through_one_json_line() {
        for cmd in all_commands() {
            let req = ControlRequest::new("00ff", cmd.clone());
            let line = serde_json::to_string(&req).unwrap();
            assert!(!line.contains('\n'), "one line: {line}");
            assert!(line.len() < MAX_REQUEST_BYTES);
            let back: ControlRequest = serde_json::from_str(&line).unwrap();
            assert_eq!(back, req);
            assert_eq!(back.validate(), Ok(()));
        }
    }

    #[test]
    fn the_wire_form_is_the_documented_one() {
        let req = ControlRequest::new("a1b2", ControlCommand::Wb { mode: WbArg::Left });
        assert_eq!(
            serde_json::to_string(&req).unwrap(),
            r#"{"v":1,"session":"a1b2","cmd":{"type":"wb","mode":"left"}}"#
        );
        let req = ControlRequest::new("a1b2", ControlCommand::ReloadRelations {});
        assert_eq!(
            serde_json::to_string(&req).unwrap(),
            r#"{"v":1,"session":"a1b2","cmd":{"type":"reload_relations"}}"#
        );
        let reply = ControlReply::refused(ReplyCode::RateLimited, "slow down");
        assert_eq!(
            serde_json::to_string(&reply).unwrap(),
            r#"{"v":1,"ok":false,"text":"slow down","code":"rate_limited"}"#
        );
        assert_eq!(
            serde_json::to_string(&ControlReply::answer(true, "ok")).unwrap(),
            r#"{"v":1,"ok":true,"text":"ok"}"#
        );
        let mut with_data = ControlReply::answer(true, "ok");
        with_data.data = Some(serde_json::json!({"digest": "00ff"}));
        assert_eq!(
            serde_json::to_string(&with_data).unwrap(),
            r#"{"v":1,"ok":true,"text":"ok","data":{"digest":"00ff"}}"#
        );
        let back: ControlReply = serde_json::from_str(&serde_json::to_string(&with_data).unwrap()).unwrap();
        assert_eq!(back, with_data);
    }

    #[test]
    fn anything_that_is_not_a_known_command_is_refused_at_parse_time() {
        for bad in [
            // The things a chat path would need.
            r#"{"v":1,"session":"a","cmd":{"type":"say","text":"hi"}}"#,
            r#"{"v":1,"session":"a","cmd":{"type":"chat","message":"hi"}}"#,
            r#"{"v":1,"session":"a","cmd":{"type":"quit"}}"#,
            r#"{"v":1,"session":"a","cmd":{"type":"target","name":"x"}}"#,
            // Smuggled fields.
            r#"{"v":1,"session":"a","cmd":{"type":"stop","say":"hi"}}"#,
            r#"{"v":1,"session":"a","cmd":{"type":"kill"},"extra":1}"#,
            r#"{"v":1,"session":"a","cmd":{"type":"mode","mode":"goto"}}"#,
            r#"{"v":1,"session":"a","cmd":{"type":"clip","note":"x","say":"hi"}}"#,
            // Wrong shapes.
            r#"{"v":1,"session":"a","cmd":"stop"}"#,
            r#"{"v":1,"session":"a"}"#,
            r#"[]"#,
            r#"null"#,
            "",
        ] {
            assert!(serde_json::from_str::<ControlRequest>(bad).is_err(), "accepted: {bad}");
        }
    }

    #[test]
    fn validation_checks_version_session_note_and_goto() {
        let ok = ControlRequest::new("abc123", ControlCommand::Stop {});
        assert_eq!(ok.validate(), Ok(()));
        let mut v = ok.clone();
        v.v = 2;
        assert_eq!(v.validate(), Err(Invalid::Version));
        for bad in ["", "ABC", "xyz", "a b", &"a".repeat(MAX_SESSION_CHARS + 1)] {
            let mut r = ok.clone();
            r.session = bad.to_string();
            assert_eq!(r.validate(), Err(Invalid::Session), "{bad:?}");
        }
        let note = |n: &str| ControlCommand::Clip { note: n.to_string() };
        assert_eq!(note(&"x".repeat(MAX_NOTE_CHARS)).validate(), Ok(()));
        assert_eq!(note(&"x".repeat(MAX_NOTE_CHARS + 1)).validate(), Err(Invalid::Note));
        assert_eq!(note("a\nb").validate(), Err(Invalid::Note));
        assert_eq!(note("a\u{7}b").validate(), Err(Invalid::Note));
        assert_eq!(note("").validate(), Ok(()));
        assert_eq!(
            ControlCommand::Goto {
                x: GOTO_LIMIT,
                y: -GOTO_LIMIT
            }
            .validate(),
            Ok(())
        );
        assert_eq!(
            ControlCommand::Goto {
                x: GOTO_LIMIT + 1,
                y: 0
            }
            .validate(),
            Err(Invalid::Goto)
        );
        assert_eq!(
            ControlCommand::Goto { x: i32::MIN, y: 0 }.validate(),
            Err(Invalid::Goto),
            "abs() of MIN must not wrap into range"
        );
    }

    #[test]
    fn a_tag_never_carries_free_text() {
        let tag = ControlCommand::Clip {
            note: "SECRET-NOTE-xyz".into(),
        }
        .tag();
        assert_eq!(tag, "clip");
        let tags: Vec<String> = all_commands().iter().map(ControlCommand::tag).collect();
        for t in &tags {
            assert!(
                t.chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || ":,-".contains(c)),
                "{t}"
            );
        }
        assert!(tags.contains(&"goto:-3,40".to_string()));
        assert!(tags.contains(&"relations:reload".to_string()));
    }

    #[test]
    fn replies_clip_their_text() {
        let long = "я".repeat(MAX_REPLY_TEXT_CHARS * 2);
        let r = ControlReply::answer(true, &long);
        assert_eq!(r.text.chars().count(), MAX_REPLY_TEXT_CHARS);
        let line = serde_json::to_string(&r).unwrap();
        assert!(line.len() < MAX_REPLY_BYTES, "{} bytes", line.len());
    }
}
