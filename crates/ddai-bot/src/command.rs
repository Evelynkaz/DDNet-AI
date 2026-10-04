//! The bot's commands as data (task 4.3): [`BotCommand`] is what the console (stdin) and, later, the web
//! control ask of a running bot, [`parse_line`] is the console syntax (`!cmd args` or `?cmd args`), and
//! [`CommandBus`] carries them to the bot's thread. The bot applies one with `Bot::command`.
//!
//! **No chat from the console, by construction.** A line that does not begin with `!` or `?` is a [`ParseError::NoPrefix`] and goes
//! nowhere (the TS bot sent it to the game chat, `handleConsole` `bot.ts:1314-1322`); `!say`, `!owner`, `!llm`, the dummy and the vote
//! commands parse to [`BotCommand::Unsupported`] with the reason. [`parse_line`] never returns [`BotCommand::Say`]: that variant comes
//! from one place only, the web control channel (`crate::control`, task 4.9, D-094), and it carries an [`OwnerText`], a line the
//! owner typed on the authenticated website that passed the one validating constructor. `tests/commands.rs` fuzzes the parser to
//! prove no console input ever produces anything but an enum value, and never a `Say`.

use std::sync::mpsc;
use std::time::Duration;

use crate::bot::Mode;
use crate::brains::BrainKind;
use crate::nav_hooks::WbMode;
use crate::relations::ListKind;
use ddai_net::owner_chat::OwnerText;

/// `!goto` forms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GotoArg {
    /// `!goto`: the progress of the walk, or a hint.
    Status,
    /// `!goto stop | - | off`.
    Stop,
    /// `!goto tele | teleport | tp`.
    Tele,
    /// `!goto <x> <y>` (tiles).
    Tile { x: i32, y: i32 },
    /// `!goto <nick>` / `!goto @nick`: follow that player.
    Player(String),
}

/// `!target` forms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetArg {
    /// `!target`: say what is set.
    Query,
    /// `!target -`.
    Clear,
    /// `!target <nick>`.
    Name(String),
}

/// What a list command does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ListArg {
    /// No argument: show the list.
    Show,
    /// `off`: clear it.
    Clear,
    /// A name: toggle it on the list.
    Toggle(String),
}

/// `!home` forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HomeArg {
    /// `!home`: where the tee stands now.
    Here,
    /// `!home <x> <y>`.
    Tile { x: i32, y: i32 },
    /// `!home off`.
    Off,
}

/// Everything the operator (or the web) can ask of the bot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BotCommand {
    Help,
    /// `!mode [fight|passive|hold]`; `None` says the current one.
    Mode(Option<Mode>),
    Stop,
    Go,
    Goto(GotoArg),
    Target(TargetArg),
    List {
        kind: ListKind,
        arg: ListArg,
    },
    Home(HomeArg),
    /// `!wb [off|left|right|auto|on]`; `None` reports.
    Wb(Option<WbMode>),
    /// `!clip [note]`.
    Clip(String),
    Stats,
    Where,
    /// `!brain [hybrid|planner|scripted|idle|fly]`; `None` says the current one.
    Brain(Option<BrainKind>),
    /// `!low [on|off]`.
    Low(Option<bool>),
    /// `!strong [on|off]`.
    Strong(Option<bool>),
    Spec,
    Join,
    /// `!kill` / `!reset`.
    Kill,
    /// Re-read the lists file (the web editor changed it): the lists in memory become the file's. Not a console
    /// command (the console edits the lists itself); it comes from the web control channel (task 5.6).
    ReloadRelations,
    Quit,
    /// Say a line in the game chat (task 4.9, D-094): what the owner typed on the website, already validated. The only variant
    /// that carries text for the server; the console cannot make one. Applied by the runner's `OwnerChat` (rate limits, queue,
    /// in-game only), not by `Bot::command`.
    Say {
        team: bool,
        text: OwnerText,
    },
    /// A command the old bot had that this one deliberately does not (with the reason).
    Unsupported(&'static str),
}

/// Why a line is not a command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// Blank line.
    Empty,
    /// No `!` or `?` in front: **nothing is sent anywhere** (the TS bot said it in the game chat).
    NoPrefix,
    /// `!` / `?` with nothing after it.
    NoCommand,
    Unknown(String),
    /// A known command with an argument it cannot read: the usage line.
    Usage(&'static str),
}

impl ParseError {
    /// The console's answer.
    pub fn message(&self) -> String {
        match self {
            ParseError::Empty => String::new(),
            ParseError::NoPrefix => {
                "not a command: start with ! or ? (try !help). The console never writes in the game chat, so a line without a prefix goes nowhere".to_string()
            }
            ParseError::NoCommand => "empty command -- try !help".to_string(),
            ParseError::Unknown(c) => format!("unknown command '{c}' -- try !help"),
            ParseError::Usage(u) => (*u).to_string(),
        }
    }
}

pub const COMMAND_PREFIXES: [char; 2] = ['!', '?'];

fn on_off(arg: &str) -> Result<Option<bool>, ()> {
    match arg.to_lowercase().as_str() {
        "" => Ok(None),
        "on" => Ok(Some(true)),
        "off" => Ok(Some(false)),
        _ => Err(()),
    }
}

fn int(s: &str) -> Option<i32> {
    s.parse::<i32>().ok()
}

/// Parses one console line.
pub fn parse_line(line: &str) -> Result<BotCommand, ParseError> {
    let trimmed = line.trim();
    let Some(first) = trimmed.chars().next() else {
        return Err(ParseError::Empty);
    };
    if !COMMAND_PREFIXES.contains(&first) {
        return Err(ParseError::NoPrefix);
    }
    let body = &trimmed[first.len_utf8()..];
    let mut words = body.split_whitespace();
    let Some(cmd) = words.next() else {
        return Err(ParseError::NoCommand);
    };
    let rest: Vec<&str> = words.collect();
    let arg = rest.join(" ");
    let low = arg.to_lowercase();
    let list = |kind: ListKind| -> BotCommand {
        BotCommand::List {
            kind,
            arg: match low.as_str() {
                "" => ListArg::Show,
                "off" => ListArg::Clear,
                _ => ListArg::Toggle(arg.clone()),
            },
        }
    };
    Ok(match cmd.to_lowercase().as_str() {
        "help" | "h" => BotCommand::Help,
        "mode" => match low.as_str() {
            "" => BotCommand::Mode(None),
            "fight" | "passive" | "hold" => BotCommand::Mode(Mode::parse(&low)),
            _ => return Err(ParseError::Usage("!mode fight | passive | hold")),
        },
        "stop" => BotCommand::Stop,
        "go" => BotCommand::Go,
        "goto" => BotCommand::Goto(parse_goto(&arg, &low)),
        "target" => BotCommand::Target(match arg.as_str() {
            "" => TargetArg::Query,
            "-" => TargetArg::Clear,
            _ => TargetArg::Name(arg.clone()),
        }),
        "war" => list(ListKind::War),
        "friend" => list(ListKind::Friend),
        "ignore" => list(ListKind::Ignore),
        "clanwar" => list(ListKind::ClanWar),
        "clanfriend" => list(ListKind::ClanFriend),
        "home" => {
            let parts: Vec<&str> = arg
                .split(|c: char| c.is_whitespace() || c == ',')
                .filter(|p| !p.is_empty())
                .collect();
            if low == "off" {
                BotCommand::Home(HomeArg::Off)
            } else if parts.is_empty() {
                BotCommand::Home(HomeArg::Here)
            } else if let [x, y] = parts[..]
                && let (Some(x), Some(y)) = (int(x), int(y))
            {
                BotCommand::Home(HomeArg::Tile { x, y })
            } else {
                return Err(ParseError::Usage(
                    "!home            mark where the tee is standing\n!home <x> <y>    mark a tile\n!home off        forget it",
                ));
            }
        }
        "wb" => match low.as_str() {
            "" => BotCommand::Wb(None),
            "on" => BotCommand::Wb(Some(WbMode::Auto)),
            other => match WbMode::parse(other) {
                Some(m) => BotCommand::Wb(Some(m)),
                None => return Err(ParseError::Usage("!wb off | left | right | auto")),
            },
        },
        "clip" => BotCommand::Clip(arg),
        "stats" => BotCommand::Stats,
        "where" => BotCommand::Where,
        "brain" => match low.as_str() {
            "" => BotCommand::Brain(None),
            // The TS name for the fly was `net`.
            "net" => BotCommand::Brain(Some(BrainKind::Fly)),
            other => match BrainKind::parse(other) {
                Some(k) => BotCommand::Brain(Some(k)),
                None => return Err(ParseError::Usage("!brain hybrid | planner | scripted | idle | fly")),
            },
        },
        "low" => BotCommand::Low(on_off(&low).map_err(|()| ParseError::Usage("!low on | off"))?),
        "strong" => BotCommand::Strong(on_off(&low).map_err(|()| ParseError::Usage("!strong on | off"))?),
        "spec" => BotCommand::Spec,
        "join" => BotCommand::Join,
        "kill" | "reset" => BotCommand::Kill,
        "quit" | "exit" => BotCommand::Quit,
        // Dropped on purpose (task 4.3 / D-007): each says why instead of "unknown command".
        "say" => BotCommand::Unsupported(
            "say: the console never writes in the game chat (D-007); the owner types chat lines on the website (D-094)",
        ),
        "owner" | "llm" => BotCommand::Unsupported("owner / llm: chat orders are gone with the chat (D-007)"),
        "d" => BotCommand::Unsupported("d: there is no dummy"),
        "emote" => BotCommand::Unsupported("emote: the bot sends no emotes"),
        "yes" | "f3" | "no" | "f4" | "votes" | "vote" => BotCommand::Unsupported(
            "votes: the bot does not vote or call votes (moderation and votes are the server's)",
        ),
        "duel" | "style" => {
            BotCommand::Unsupported("duel / style: the duel mode was dropped; the bot plays one way (use !wb)")
        }
        "try" => BotCommand::Unsupported(
            "try: the planner's experimental presets are not offered live (use !brain, !low, !strong)",
        ),
        "lang" | "log" => BotCommand::Unsupported("lang / log: the console is English; the web unit filters the log"),
        other => return Err(ParseError::Unknown(other.to_string())),
    })
}

fn parse_goto(arg: &str, low: &str) -> GotoArg {
    let named = arg.starts_with('@');
    if arg.is_empty() {
        return GotoArg::Status;
    }
    if !named && matches!(low, "stop" | "-" | "off") {
        return GotoArg::Stop;
    }
    if named {
        return GotoArg::Player(arg[1..].trim().to_string());
    }
    if matches!(low, "tele" | "teleport" | "tp") {
        return GotoArg::Tele;
    }
    let parts: Vec<&str> = arg
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|p| !p.is_empty())
        .collect();
    if let [x, y] = parts[..]
        && let (Some(x), Some(y)) = (int(x), int(y))
    {
        return GotoArg::Tile { x, y };
    }
    GotoArg::Player(arg.to_string())
}

/// What a command answers.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CommandReply {
    /// The text for the operator's console. May hold nicknames the operator typed or that the bot has on
    /// its roster: it is for the local console only, never for a log (the runner prints it directly).
    pub text: String,
    /// The command was understood and done (or queued); false for a usage error, a refusal, a failure.
    pub ok: bool,
    /// The operator asked the bot to quit.
    pub quit: bool,
    /// Structured detail for the web control (`ReloadRelations`: counts and a digest); `None` for everything else.
    pub data: Option<serde_json::Value>,
}

impl CommandReply {
    pub fn ok(text: impl Into<String>) -> CommandReply {
        CommandReply {
            text: text.into(),
            ok: true,
            quit: false,
            data: None,
        }
    }

    pub fn err(text: impl Into<String>) -> CommandReply {
        CommandReply {
            text: text.into(),
            ok: false,
            quit: false,
            data: None,
        }
    }
}

/// A command on its way to the bot's thread, with where to send the answer.
pub struct CommandRequest {
    pub cmd: BotCommand,
    pub reply: mpsc::Sender<CommandReply>,
}

/// The bot-side end: the runner drains it between snapshots.
pub struct CommandInbox {
    rx: mpsc::Receiver<CommandRequest>,
}

impl CommandInbox {
    /// The next waiting request, if any.
    pub fn try_next(&self) -> Option<CommandRequest> {
        self.rx.try_recv().ok()
    }
}

/// The sender side: `Send + Clone`, for the stdin thread and the future web control.
#[derive(Clone)]
pub struct CommandSender {
    tx: mpsc::Sender<CommandRequest>,
}

/// The sender could not reach the bot, or the bot did not answer in time.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BusError {
    #[error("the bot has stopped")]
    Gone,
    #[error("the bot did not answer in time")]
    Timeout,
}

impl CommandSender {
    /// Sends `cmd` and waits for the answer (the bot answers between two snapshots, so within tens of
    /// milliseconds).
    pub fn send(&self, cmd: BotCommand, timeout: Duration) -> Result<CommandReply, BusError> {
        let (reply, rx) = mpsc::channel();
        self.tx
            .send(CommandRequest { cmd, reply })
            .map_err(|_| BusError::Gone)?;
        rx.recv_timeout(timeout).map_err(|e| match e {
            mpsc::RecvTimeoutError::Timeout => BusError::Timeout,
            mpsc::RecvTimeoutError::Disconnected => BusError::Gone,
        })
    }

    /// Parses `line` and sends it: the console's whole path. A line that is not a command is answered here,
    /// without touching the bot.
    pub fn send_line(&self, line: &str, timeout: Duration) -> Result<CommandReply, BusError> {
        match parse_line(line) {
            Ok(cmd) => self.send(cmd, timeout),
            Err(e) => Ok(CommandReply::err(e.message())),
        }
    }
}

/// A command channel: [`CommandBus::open`] gives the inbox for the runner and a sender to clone.
pub struct CommandBus;

impl CommandBus {
    pub fn open() -> (CommandSender, CommandInbox) {
        let (tx, rx) = mpsc::channel();
        (CommandSender { tx }, CommandInbox { rx })
    }
}
