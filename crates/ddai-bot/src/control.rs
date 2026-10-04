//! The web control channel, bot side (task 5.6, D-070, `docs/formats.md` §26): a Unix socket next to the read-only
//! live bridge (`crate::bridge`) over which the web unit asks the running bot for the same things the console can —
//! through the same API (`BotCommand` on the `CommandBus`, `CommandReply` back).
//!
//! - **Private.** `control.sock` is created `0600` in a `0700` directory ([`crate::bridge::bind_private_socket`]); that
//!   file mode is the access control (same user only), as for the bridge.
//! - **Typed and closed.** A request is a `ddai_botctl::proto::ControlRequest`; its command is a closed enum of
//!   what the web needs (mode, stop/go, wb, brain, kill, clip, goto x y, spec/join, reload the lists, and since task 4.9 `say`: a
//!   line the owner typed on the website). [`to_bot_command`] maps it onto a `BotCommand` with an exhaustive `match` (no wildcard);
//!   for `say` it validates the text **again** and makes the `OwnerText` with the process's one `OwnerChannel`, which only the
//!   [`Dispatcher`] holds (`ControlServer::start` claims it): no other code can make an `OwnerText`, so a `BotCommand::Say` always
//!   holds a validated line that came through this socket. Without the channel (`--no-owner-chat`, or `owner_chat = false` in the
//!   settings, or the channel already taken) a `say` is refused (`chat_disabled`) and everything else works. Unknown commands and fields are refused when parsing; `quit` and `target` by name do not exist in the
//!   protocol. The bot's own pacing of chat lines (`crate::ownerchat`) comes after that.
//! - **Rate-limited.** One token bucket for the whole socket ([`RATE_BURST`], [`RATE_PER_SEC`]) counts every request line,
//!   well-formed or not; an empty bucket answers `rate_limited` without bothering the bot. At most [`MAX_CONNECTIONS`]
//!   connections are served at once.
//! - **Audited, without names.** Every command that reaches the bot — and every refusal — becomes one line in the audit
//!   log: unix milliseconds, the web session's opaque tag, the command's tag (`ControlCommand::tag`: a fixed shape, no
//!   free text, no nickname) and the outcome. Nothing else is written: not the clip note, not the reply text (the
//!   bot's replies may name players). The audit type ([`AuditEntry`]) has no field that could hold a name.
//!
//! The server runs on its own threads; the bot's thread only drains the `CommandBus` between snapshots, as for the
//! console, so a slow or hostile client can never stall a decision.

use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ddai_botctl::proto::{
    BrainArg, ControlCommand, ControlReply, ControlRequest, Invalid, MAX_REQUEST_BYTES, ModeArg, ReplyCode, WbArg,
};

use crate::bot::Mode;
use crate::brains::BrainKind;
use crate::command::{BotCommand, BusError, CommandSender, GotoArg};
use crate::nav_hooks::WbMode;
use ddai_net::owner_chat::{OwnerChannel, OwnerText};

/// File name of the socket inside the bot's directory.
pub const SOCKET_NAME: &str = "control.sock";
/// Commands in a burst before the bucket is empty.
pub const RATE_BURST: f64 = 8.0;
/// Commands per second the bucket refills with (a person at a keyboard).
pub const RATE_PER_SEC: f64 = 2.0;
/// Connections served at once; further ones are told `busy` and closed.
pub const MAX_CONNECTIONS: usize = 4;
/// A connection that sends nothing for this long is closed.
pub const IDLE_TIMEOUT: Duration = Duration::from_secs(15);
/// How long the bot has to answer a command (it answers between two snapshots).
pub const ANSWER_TIMEOUT: Duration = crate::console::ANSWER_TIMEOUT;
/// The audit file is moved to `<name>.1` once it passes this size.
pub const AUDIT_MAX_BYTES: u64 = 8 << 20;

/// What a control command asks of the bot, as the 4.3 console's own type. Exhaustive on purpose: a command added to the
/// protocol must be given its meaning here. Fails only for a `say`: its text is refused by `OwnerText::new` (the bot's own check of
/// the owner's line, independent of the web's), or there is no `owner` channel (the owner chat is switched off).
pub fn to_bot_command(cmd: &ControlCommand, owner: Option<&OwnerChannel>) -> Result<BotCommand, Invalid> {
    Ok(match cmd {
        ControlCommand::Mode { mode } => BotCommand::Mode(Some(match mode {
            ModeArg::Fight => Mode::Fight,
            ModeArg::Passive => Mode::Passive,
            ModeArg::Hold => Mode::Hold,
        })),
        ControlCommand::Stop {} => BotCommand::Stop,
        ControlCommand::Go {} => BotCommand::Go,
        ControlCommand::Wb { mode } => BotCommand::Wb(Some(match mode {
            WbArg::Auto => WbMode::Auto,
            WbArg::Left => WbMode::Left,
            WbArg::Right => WbMode::Right,
            WbArg::Off => WbMode::Off,
        })),
        ControlCommand::Brain { brain } => BotCommand::Brain(Some(match brain {
            BrainArg::Hybrid => BrainKind::Hybrid,
            BrainArg::Planner => BrainKind::Planner,
            BrainArg::Scripted => BrainKind::Scripted,
            BrainArg::Idle => BrainKind::Idle,
            BrainArg::Fly => BrainKind::Fly,
        })),
        ControlCommand::Kill {} => BotCommand::Kill,
        ControlCommand::Clip { note } => BotCommand::Clip(note.clone()),
        ControlCommand::Goto { x, y } => BotCommand::Goto(GotoArg::Tile { x: *x, y: *y }),
        ControlCommand::Spec {} => BotCommand::Spec,
        ControlCommand::Join {} => BotCommand::Join,
        ControlCommand::ReloadRelations {} => BotCommand::ReloadRelations,
        ControlCommand::Say { team, text } => {
            let channel = owner.ok_or(Invalid::ChatDisabled)?;
            BotCommand::Say {
                team: *team,
                text: OwnerText::new(channel, text.as_str()).map_err(Invalid::Say)?,
            }
        }
    })
}

/// Gives up the owner chat for this process: claims the process's one `OwnerChannel` and lets it go, so that no later `claim()` gets
/// it (task 4.9, D-094). Called when the chat is switched off (`--no-owner-chat`, `owner_chat = false`, the marker file, a settings
/// file that cannot be trusted) and when there is no control socket at all (`--no-control`): the capability must not be left lying
/// free exactly in the modes where the owner asked for no chat.
pub fn forgo_owner_chat() {
    let _burnt = OwnerChannel::claim();
}

// ---- the rate limit ---------------------------------------------------------------------------------

/// A token bucket with an injectable clock.
#[derive(Debug, Clone)]
pub struct RateLimiter {
    capacity: f64,
    per_sec: f64,
    tokens: f64,
    last: Instant,
}

impl RateLimiter {
    /// A full bucket of `capacity` tokens, refilled at `per_sec` tokens per second, as of `now`.
    pub fn new(capacity: f64, per_sec: f64, now: Instant) -> RateLimiter {
        RateLimiter {
            capacity,
            per_sec,
            tokens: capacity,
            last: now,
        }
    }

    /// Takes one token if there is one.
    pub fn try_acquire(&mut self, now: Instant) -> bool {
        let dt = now.saturating_duration_since(self.last).as_secs_f64();
        self.last = self.last.max(now);
        self.tokens = (self.tokens + dt * self.per_sec).min(self.capacity);
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

// ---- the audit log ----------------------------------------------------------------------------------

/// How a request ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The bot applied it (or queued it) and said ok.
    Ok,
    /// The bot answered, but refused or failed (a cooldown, no navigation, no clip directory ...).
    Failed,
    RateLimited,
    BadRequest,
    /// The bot did not answer in time.
    Timeout,
    /// The bot is stopping.
    Gone,
}

impl Outcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Outcome::Ok => "ok",
            Outcome::Failed => "failed",
            Outcome::RateLimited => "rate_limited",
            Outcome::BadRequest => "bad_request",
            Outcome::Timeout => "timeout",
            Outcome::Gone => "gone",
        }
    }
}

/// One audit line. The fields are the whole record: there is nowhere to put a nickname, a clip note or a reply text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditEntry {
    /// Unix milliseconds.
    pub ts_ms: u64,
    /// The web session's opaque tag (`-` when the request had none that validated).
    pub session: String,
    /// `ControlCommand::tag` (`-` when the request did not parse).
    pub cmd: String,
    pub outcome: Outcome,
}

impl AuditEntry {
    /// The JSON line (without the newline). Built from the validated parts only: a session tag is hex, a command tag is
    /// `[a-z0-9:,-]`, so nothing needs escaping.
    pub fn to_line(&self) -> String {
        format!(
            r#"{{"ts_ms":{},"session":"{}","cmd":"{}","outcome":"{}"}}"#,
            self.ts_ms,
            self.session,
            self.cmd,
            self.outcome.as_str()
        )
    }
}

/// Where audit entries go.
pub trait AuditSink: Send + Sync {
    fn record(&self, entry: &AuditEntry);
}

/// An in-memory sink (tests).
#[derive(Default)]
pub struct MemoryAudit(pub Mutex<Vec<AuditEntry>>);

impl AuditSink for MemoryAudit {
    fn record(&self, entry: &AuditEntry) {
        if let Ok(mut v) = self.0.lock() {
            v.push(entry.clone());
        }
    }
}

/// An append-only file (`0600`), one JSON line per entry, moved aside to `<name>.1` past [`AUDIT_MAX_BYTES`].
pub struct FileAudit {
    path: PathBuf,
    file: Mutex<File>,
    max_bytes: u64,
}

fn open_audit(path: &Path) -> io::Result<File> {
    OpenOptions::new().create(true).append(true).mode(0o600).open(path)
}

impl FileAudit {
    pub fn open(path: &Path) -> io::Result<FileAudit> {
        FileAudit::open_with_cap(path, AUDIT_MAX_BYTES)
    }

    pub fn open_with_cap(path: &Path, max_bytes: u64) -> io::Result<FileAudit> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        Ok(FileAudit {
            path: path.to_path_buf(),
            file: Mutex::new(open_audit(path)?),
            max_bytes,
        })
    }
}

impl AuditSink for FileAudit {
    fn record(&self, entry: &AuditEntry) {
        let Ok(mut file) = self.file.lock() else { return };
        if file.metadata().is_ok_and(|m| m.len() > self.max_bytes) {
            let mut old = self.path.as_os_str().to_owned();
            old.push(".1");
            if std::fs::rename(&self.path, PathBuf::from(old)).is_ok()
                && let Ok(fresh) = open_audit(&self.path)
            {
                *file = fresh;
            }
        }
        let mut line = entry.to_line();
        line.push('\n');
        if let Err(e) = file.write_all(line.as_bytes()) {
            tracing::warn!(error = %e, "could not write the control audit log");
        }
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

// ---- one request ------------------------------------------------------------------------------------

struct Limits {
    bucket: RateLimiter,
    /// When a `rate_limited` entry was last written (a flood must not flood the audit log).
    last_limited_audit: Option<Instant>,
}

/// Turns request lines into replies: parse, validate, rate-limit, hand to the bot, audit. No sockets in here, so every
/// rule is testable without any.
pub struct Dispatcher {
    sender: CommandSender,
    audit: Arc<dyn AuditSink>,
    limits: Mutex<Limits>,
    answer_timeout: Duration,
    /// The process's one `OwnerChannel` (task 4.9, D-094): the only thing that can turn a `say` request into an `OwnerText`. `None`:
    /// the owner chat is off and every `say` is refused.
    owner: Option<OwnerChannel>,
}

impl Dispatcher {
    pub fn new(sender: CommandSender, audit: Arc<dyn AuditSink>) -> Dispatcher {
        Dispatcher::with_limits(sender, audit, RATE_BURST, RATE_PER_SEC, ANSWER_TIMEOUT)
    }

    pub fn with_limits(
        sender: CommandSender,
        audit: Arc<dyn AuditSink>,
        burst: f64,
        per_sec: f64,
        answer_timeout: Duration,
    ) -> Dispatcher {
        Dispatcher {
            sender,
            audit,
            limits: Mutex::new(Limits {
                bucket: RateLimiter::new(burst, per_sec, Instant::now()),
                last_limited_audit: None,
            }),
            answer_timeout,
            owner: None,
        }
    }

    /// Hands the dispatcher the process's `OwnerChannel`: from now on it turns validated `say` requests into `OwnerText`s. The only
    /// production caller is [`ControlServer::start`], with the channel from `OwnerChannel::claim()`.
    pub fn with_owner_channel(mut self, channel: OwnerChannel) -> Dispatcher {
        self.owner = Some(channel);
        self
    }

    /// Whether this dispatcher can pass a chat line on.
    pub fn owner_chat_enabled(&self) -> bool {
        self.owner.is_some()
    }

    fn audit(&self, session: &str, cmd: &str, outcome: Outcome) {
        let entry = AuditEntry {
            ts_ms: now_ms(),
            session: session.to_string(),
            cmd: cmd.to_string(),
            outcome,
        };
        tracing::info!(target: "control", session, cmd, outcome = outcome.as_str(), "control command");
        self.audit.record(&entry);
    }

    /// Answers one request line (no trailing newline).
    pub fn handle_line(&self, line: &[u8]) -> ControlReply {
        let now = Instant::now();
        {
            let mut limits = self.limits.lock().unwrap_or_else(|e| e.into_inner());
            if !limits.bucket.try_acquire(now) {
                let log_it = limits
                    .last_limited_audit
                    .is_none_or(|t| now.saturating_duration_since(t) >= Duration::from_secs(1));
                if log_it {
                    limits.last_limited_audit = Some(now);
                }
                drop(limits);
                if log_it {
                    self.audit("-", "-", Outcome::RateLimited);
                }
                return ControlReply::refused(ReplyCode::RateLimited, "too many commands: wait a moment");
            }
        }
        let req: ControlRequest = match serde_json::from_slice(line) {
            Ok(r) => r,
            Err(_) => {
                self.audit("-", "-", Outcome::BadRequest);
                return ControlReply::refused(ReplyCode::BadRequest, "bad request");
            }
        };
        let tag = req.cmd.tag();
        if let Err(e) = req.validate() {
            let session = if ddai_botctl::proto::valid_session_tag(&req.session) {
                req.session.as_str()
            } else {
                "-"
            };
            self.audit(session, &tag, Outcome::BadRequest);
            return ControlReply::refused(ReplyCode::BadRequest, &e.to_string());
        }
        // The bot's own check of a chat line (the web has validated it too); nothing else can fail here.
        let bot_cmd = match to_bot_command(&req.cmd, self.owner.as_ref()) {
            Ok(c) => c,
            Err(Invalid::ChatDisabled) => {
                self.audit(&req.session, &tag, Outcome::Failed);
                let mut reply = ControlReply::answer(false, "refused: the owner chat is switched off on the bot");
                reply.data = Some(serde_json::json!({ "reason": "chat_disabled" }));
                return reply;
            }
            Err(e) => {
                self.audit(&req.session, &tag, Outcome::BadRequest);
                return ControlReply::refused(ReplyCode::BadRequest, &e.to_string());
            }
        };
        let (reply, outcome) = match self.sender.send(bot_cmd, self.answer_timeout) {
            Ok(r) => {
                let mut reply = ControlReply::answer(r.ok, &r.text);
                reply.data = r.data;
                (reply, if r.ok { Outcome::Ok } else { Outcome::Failed })
            }
            Err(BusError::Timeout) => (
                ControlReply::refused(ReplyCode::Timeout, "the bot did not answer in time"),
                Outcome::Timeout,
            ),
            Err(BusError::Gone) => (
                ControlReply::refused(ReplyCode::Gone, "the bot is stopping"),
                Outcome::Gone,
            ),
        };
        self.audit(&req.session, &tag, outcome);
        reply
    }
}

// ---- the socket -------------------------------------------------------------------------------------

/// What reading one request line found.
#[derive(Debug, PartialEq, Eq)]
pub enum LineRead {
    /// A line (without the newline) is in the buffer.
    Line,
    /// The peer closed the connection.
    Eof,
    /// The line passed the limit without a newline: the stream cannot be resynchronised.
    TooLong,
}

/// Reads up to the next `\n`, never holding more than `max` bytes of it (newline included).
pub fn read_line_bounded<R: BufRead>(reader: &mut R, buf: &mut Vec<u8>, max: usize) -> io::Result<LineRead> {
    buf.clear();
    let n = reader.by_ref().take(max as u64 + 1).read_until(b'\n', buf)?;
    if n == 0 {
        return Ok(LineRead::Eof);
    }
    if buf.last() == Some(&b'\n') {
        buf.pop();
        if n > max {
            return Ok(LineRead::TooLong);
        }
        return Ok(LineRead::Line);
    }
    if n > max {
        return Ok(LineRead::TooLong);
    }
    // EOF in the middle of a line: not a request.
    Ok(LineRead::Eof)
}

fn write_reply(stream: &mut UnixStream, reply: &ControlReply) -> io::Result<()> {
    let mut line = serde_json::to_vec(reply).map_err(io::Error::other)?;
    line.push(b'\n');
    stream.write_all(&line)
}

struct ConnGuard(Arc<AtomicUsize>);

impl Drop for ConnGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

fn serve_connection(mut stream: UnixStream, dispatcher: &Dispatcher) {
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(IDLE_TIMEOUT));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));
    let Ok(read_half) = stream.try_clone() else { return };
    let mut reader = BufReader::new(read_half);
    let mut buf = Vec::with_capacity(256);
    loop {
        match read_line_bounded(&mut reader, &mut buf, MAX_REQUEST_BYTES) {
            Ok(LineRead::Line) => {
                let reply = dispatcher.handle_line(&buf);
                if write_reply(&mut stream, &reply).is_err() {
                    return;
                }
            }
            Ok(LineRead::TooLong) => {
                let _ = write_reply(
                    &mut stream,
                    &ControlReply::refused(ReplyCode::BadRequest, "request too long"),
                );
                return;
            }
            Ok(LineRead::Eof) | Err(_) => return,
        }
    }
}

/// The running control socket. Dropping it stops accepting and removes the socket file.
pub struct ControlServer {
    path: PathBuf,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl ControlServer {
    /// Binds `path` (mode `0600` in a `0700` directory; a live bot's socket is not stolen, a stale one is replaced) and
    /// starts serving it on a thread named `ddai-botctl`.
    ///
    /// **Claims the process's one `OwnerChannel`** and moves it into the dispatcher (task 4.9, D-094): this is the only place a chat line
    /// can enter the bot. If the channel is already taken the server still starts, with the owner chat off.
    pub fn start(path: &Path, sender: CommandSender, audit: Arc<dyn AuditSink>) -> io::Result<ControlServer> {
        ControlServer::start_with_owner_chat(path, sender, audit, true)
    }

    /// [`ControlServer::start`] with the owner chat on or off: off (`--no-owner-chat`, the emergency switch) gives the dispatcher no
    /// channel and **burns** the process's one channel ([`forgo_owner_chat`]), so every `say` is refused while the rest of the control
    /// channel works and nothing else can claim it later.
    pub fn start_with_owner_chat(
        path: &Path,
        sender: CommandSender,
        audit: Arc<dyn AuditSink>,
        owner_chat: bool,
    ) -> io::Result<ControlServer> {
        let mut dispatcher = Dispatcher::new(sender, audit);
        if owner_chat {
            match OwnerChannel::claim() {
                Some(channel) => dispatcher = dispatcher.with_owner_channel(channel),
                None => {
                    tracing::warn!("the owner chat channel was already taken in this process: chat lines are refused")
                }
            }
        } else {
            forgo_owner_chat();
        }
        ControlServer::start_with(path, dispatcher)
    }

    pub fn start_with(path: &Path, dispatcher: Dispatcher) -> io::Result<ControlServer> {
        let listener = crate::bridge::bind_private_socket(path)?;
        listener.set_nonblocking(true)?;
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let stop = Arc::clone(&stop);
            let dispatcher = Arc::new(dispatcher);
            thread::Builder::new()
                .name("ddai-botctl".into())
                .spawn(move || accept_loop(&listener, &stop, &dispatcher))?
        };
        Ok(ControlServer {
            path: path.to_path_buf(),
            stop,
            thread: Some(thread),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

fn accept_loop(listener: &UnixListener, stop: &AtomicBool, dispatcher: &Arc<Dispatcher>) {
    let active = Arc::new(AtomicUsize::new(0));
    while !stop.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((mut stream, _)) => {
                if active.fetch_add(1, Ordering::SeqCst) >= MAX_CONNECTIONS {
                    active.fetch_sub(1, Ordering::SeqCst);
                    let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
                    let _ = write_reply(
                        &mut stream,
                        &ControlReply::refused(ReplyCode::Busy, "too many connections"),
                    );
                    continue;
                }
                let guard = ConnGuard(Arc::clone(&active));
                let dispatcher = Arc::clone(dispatcher);
                let spawned = thread::Builder::new().name("ddai-botctl-conn".into()).spawn(move || {
                    let _guard = guard;
                    serve_connection(stream, &dispatcher);
                });
                if let Err(e) = spawned {
                    // The guard moved into the closure that was never run is dropped with it.
                    tracing::warn!(error = %e, "could not start a control connection thread");
                }
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => thread::sleep(Duration::from_millis(25)),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => {
                tracing::warn!(error = %e, "the control socket stopped accepting");
                return;
            }
        }
    }
}

impl Drop for ControlServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::{CommandBus, CommandInbox, CommandReply};
    use std::os::unix::fs::PermissionsExt;

    const SESSION: &str = "0a1b2c3d4e5f6071";

    fn line(cmd: &ControlCommand) -> Vec<u8> {
        serde_json::to_vec(&ControlRequest::new(SESSION, cmd.clone())).unwrap()
    }

    /// A stand-in bot thread: answers every command with `answer(cmd)` until `stop`, recording what reached it.
    struct FakeBot {
        seen: Arc<Mutex<Vec<BotCommand>>>,
        stop: Arc<AtomicBool>,
        thread: Option<thread::JoinHandle<()>>,
    }

    impl FakeBot {
        fn start(inbox: CommandInbox, answer: impl Fn(&BotCommand) -> CommandReply + Send + 'static) -> FakeBot {
            let seen = Arc::new(Mutex::new(Vec::new()));
            let stop = Arc::new(AtomicBool::new(false));
            let (s2, st2) = (Arc::clone(&seen), Arc::clone(&stop));
            let thread = thread::spawn(move || {
                while !st2.load(Ordering::SeqCst) {
                    while let Some(req) = inbox.try_next() {
                        let reply = answer(&req.cmd);
                        s2.lock().unwrap().push(req.cmd);
                        let _ = req.reply.send(reply);
                    }
                    thread::sleep(Duration::from_millis(2));
                }
            });
            FakeBot {
                seen,
                stop,
                thread: Some(thread),
            }
        }

        fn seen(&self) -> Vec<BotCommand> {
            self.seen.lock().unwrap().clone()
        }
    }

    impl Drop for FakeBot {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::SeqCst);
            if let Some(t) = self.thread.take() {
                let _ = t.join();
            }
        }
    }

    fn setup(
        burst: f64,
        per_sec: f64,
        answer: impl Fn(&BotCommand) -> CommandReply + Send + 'static,
    ) -> (Dispatcher, Arc<MemoryAudit>, FakeBot) {
        let (sender, inbox) = CommandBus::open();
        let audit = Arc::new(MemoryAudit::default());
        let d = Dispatcher::with_limits(
            sender,
            Arc::clone(&audit) as Arc<dyn AuditSink>,
            burst,
            per_sec,
            Duration::from_secs(2),
        );
        (d, audit, FakeBot::start(inbox, answer))
    }

    #[test]
    fn the_token_bucket_allows_a_burst_then_the_refill_rate() {
        let t0 = Instant::now();
        let mut b = RateLimiter::new(3.0, 2.0, t0);
        assert!(b.try_acquire(t0) && b.try_acquire(t0) && b.try_acquire(t0));
        assert!(!b.try_acquire(t0), "burst of 3 is spent");
        assert!(!b.try_acquire(t0 + Duration::from_millis(400)), "0.8 tokens");
        assert!(
            b.try_acquire(t0 + Duration::from_millis(500)),
            "1 token after 500 ms at 2/s"
        );
        assert!(!b.try_acquire(t0 + Duration::from_millis(500)));
        // A long pause refills to the capacity and no further.
        let later = t0 + Duration::from_secs(60);
        assert!(b.try_acquire(later) && b.try_acquire(later) && b.try_acquire(later));
        assert!(!b.try_acquire(later));
        // A clock that goes backwards neither panics nor mints tokens.
        assert!(!b.try_acquire(t0));
    }

    #[test]
    fn every_protocol_command_becomes_the_console_command_it_names() {
        use ddai_botctl::proto::*;
        let channel = OwnerChannel::mint_for_tests();
        let all: Vec<(ControlCommand, BotCommand)> = vec![
            (
                ControlCommand::Mode { mode: ModeArg::Fight },
                BotCommand::Mode(Some(Mode::Fight)),
            ),
            (
                ControlCommand::Mode { mode: ModeArg::Passive },
                BotCommand::Mode(Some(Mode::Passive)),
            ),
            (
                ControlCommand::Mode { mode: ModeArg::Hold },
                BotCommand::Mode(Some(Mode::Hold)),
            ),
            (ControlCommand::Stop {}, BotCommand::Stop),
            (ControlCommand::Go {}, BotCommand::Go),
            (
                ControlCommand::Wb { mode: WbArg::Auto },
                BotCommand::Wb(Some(WbMode::Auto)),
            ),
            (
                ControlCommand::Wb { mode: WbArg::Left },
                BotCommand::Wb(Some(WbMode::Left)),
            ),
            (
                ControlCommand::Wb { mode: WbArg::Right },
                BotCommand::Wb(Some(WbMode::Right)),
            ),
            (
                ControlCommand::Wb { mode: WbArg::Off },
                BotCommand::Wb(Some(WbMode::Off)),
            ),
            (
                ControlCommand::Brain {
                    brain: BrainArg::Hybrid,
                },
                BotCommand::Brain(Some(BrainKind::Hybrid)),
            ),
            (
                ControlCommand::Brain {
                    brain: BrainArg::Planner,
                },
                BotCommand::Brain(Some(BrainKind::Planner)),
            ),
            (
                ControlCommand::Brain {
                    brain: BrainArg::Scripted,
                },
                BotCommand::Brain(Some(BrainKind::Scripted)),
            ),
            (
                ControlCommand::Brain { brain: BrainArg::Idle },
                BotCommand::Brain(Some(BrainKind::Idle)),
            ),
            (
                ControlCommand::Brain { brain: BrainArg::Fly },
                BotCommand::Brain(Some(BrainKind::Fly)),
            ),
            (ControlCommand::Kill {}, BotCommand::Kill),
            (ControlCommand::Clip { note: "n".into() }, BotCommand::Clip("n".into())),
            (
                ControlCommand::Goto { x: 1, y: -2 },
                BotCommand::Goto(GotoArg::Tile { x: 1, y: -2 }),
            ),
            (ControlCommand::Spec {}, BotCommand::Spec),
            (ControlCommand::Join {}, BotCommand::Join),
            (ControlCommand::ReloadRelations {}, BotCommand::ReloadRelations),
            (
                ControlCommand::Say {
                    team: false,
                    text: ddai_botctl::proto::SayText::new("  hello  "),
                },
                BotCommand::Say {
                    team: false,
                    text: OwnerText::new(&channel, "hello").unwrap(),
                },
            ),
            (
                ControlCommand::Say {
                    team: true,
                    text: ddai_botctl::proto::SayText::new("gg"),
                },
                BotCommand::Say {
                    team: true,
                    text: OwnerText::new(&channel, "gg").unwrap(),
                },
            ),
        ];
        for (c, want) in all {
            assert_eq!(to_bot_command(&c, Some(&channel)), Ok(want.clone()), "{c:?}");
            assert!(
                !matches!(want, BotCommand::Quit | BotCommand::Unsupported(_)),
                "no control command quits or is a dropped one"
            );
        }
    }

    /// Task 4.9: the bot's own check of a chat line, independent of `ControlRequest::validate`, which the web ran first.
    #[test]
    fn the_bot_checks_a_chat_line_itself_when_it_maps_it() {
        use ddai_botctl::proto::SayText;
        use ddai_net::owner_chat::OwnerTextError;
        let channel = OwnerChannel::mint_for_tests();
        for (text, why) in [
            ("", OwnerTextError::Empty),
            ("/kill", OwnerTextError::Command),
            ("a\nb", OwnerTextError::Control),
            (&"x".repeat(256), OwnerTextError::TooLong),
            ("\u{200B}/kill", OwnerTextError::Control),
            ("\u{2800}/w someone hi", OwnerTextError::Command),
            ("xd sure chillerbot.png is lyfe", OwnerTextError::Reserved),
        ] {
            let c = ControlCommand::Say {
                team: false,
                text: SayText::new(text),
            };
            assert_eq!(to_bot_command(&c, Some(&channel)), Err(Invalid::Say(why)), "{text:?}");
            // and without the channel the bot refuses before it even looks at the text
            assert_eq!(to_bot_command(&c, None), Err(Invalid::ChatDisabled), "{text:?}");
        }
    }

    /// Task 4.9: a chat line goes through the dispatcher like any command, is trimmed, reaches the bot as a typed `BotCommand::Say`,
    /// and its text is nowhere in the audit trail; one the bot refuses never reaches it.
    #[test]
    fn a_chat_line_reaches_the_bot_typed_and_its_text_is_not_audited() {
        use ddai_botctl::proto::SayText;
        let (d, audit, bot) = setup(1000.0, 1000.0, |c| match c {
            BotCommand::Say { .. } => {
                let mut r = CommandReply::ok("accepted: it is being said now");
                r.data = Some(serde_json::json!({"reason": "none"}));
                r
            }
            _ => CommandReply::err("unexpected"),
        });
        let d = d.with_owner_channel(OwnerChannel::mint_for_tests());
        let say = |team: bool, text: &str| {
            line(&ControlCommand::Say {
                team,
                text: SayText::new(text),
            })
        };
        let ok = d.handle_line(&say(true, "  SECRET-LINE-777 "));
        assert!(ok.ok && ok.code.is_none(), "{ok:?}");
        assert_eq!(
            ok.data,
            Some(serde_json::json!({"reason": "none"})),
            "the bot's data is relayed"
        );
        for bad in ["/kill", "", "a\nb", &"x".repeat(300)] {
            let r = d.handle_line(&say(false, bad));
            assert!(!r.ok);
            assert_eq!(r.code, Some(ReplyCode::BadRequest), "{bad:?}");
            assert!(!r.text.contains("kill") || bad.is_empty(), "{}", r.text);
        }
        thread::sleep(Duration::from_millis(30));
        let seen = bot.seen();
        assert_eq!(seen.len(), 1, "only the valid line reached the bot: {seen:?}");
        let BotCommand::Say { team, text } = &seen[0] else {
            panic!("{seen:?}")
        };
        assert!(*team);
        assert_eq!(text.as_str(), "SECRET-LINE-777");
        let entries = audit.0.lock().unwrap().clone();
        assert_eq!(entries[0].cmd, "say:team");
        assert_eq!(entries[0].outcome, Outcome::Ok);
        for e in &entries {
            assert!(!e.to_line().contains("SECRET"), "{}", e.to_line());
            assert!(e.cmd == "say:team" || e.cmd == "say:all", "{}", e.cmd);
        }
        assert!(entries[1..].iter().all(|e| e.outcome == Outcome::BadRequest));
    }

    /// F4 (task 4.9): the emergency switch. Without the owner channel every `say` is refused with `chat_disabled` (a plain refusal, not
    /// a bad request) and never reaches the bot, while every other command works.
    #[test]
    fn without_the_owner_channel_chat_is_refused_and_everything_else_works() {
        use ddai_botctl::proto::SayText;
        let (d, audit, bot) = setup(1000.0, 1000.0, |_| CommandReply::ok("did it"));
        assert!(!d.owner_chat_enabled());
        let r = d.handle_line(&line(&ControlCommand::Say {
            team: false,
            text: SayText::new("hello there"),
        }));
        assert!(!r.ok && r.code.is_none(), "{r:?}");
        assert_eq!(r.data, Some(serde_json::json!({"reason": "chat_disabled"})));
        assert!(
            r.text.contains("switched off") && !r.text.contains("hello"),
            "{}",
            r.text
        );
        // the others are unaffected
        assert!(d.handle_line(&line(&ControlCommand::Stop {})).ok);
        assert!(d.handle_line(&line(&ControlCommand::Kill {})).ok);
        thread::sleep(Duration::from_millis(30));
        assert_eq!(
            bot.seen(),
            vec![BotCommand::Stop, BotCommand::Kill],
            "the chat line never reached the bot"
        );
        let entries = audit.0.lock().unwrap().clone();
        assert_eq!(
            (entries[0].cmd.as_str(), entries[0].outcome),
            ("say:all", Outcome::Failed)
        );
        // with the channel it works
        let (d, _audit, bot) = setup(1000.0, 1000.0, |_| CommandReply::ok("did it"));
        let d = d.with_owner_channel(OwnerChannel::mint_for_tests());
        assert!(d.owner_chat_enabled());
        assert!(
            d.handle_line(&line(&ControlCommand::Say {
                team: true,
                text: SayText::new("hello there"),
            }))
            .ok
        );
        thread::sleep(Duration::from_millis(30));
        assert_eq!(bot.seen().len(), 1);
    }

    #[test]
    fn a_command_reaches_the_bot_and_its_reply_comes_back_and_is_audited() {
        let (d, audit, bot) = setup(8.0, 1.0, |c| CommandReply::ok(format!("did {c:?}")));
        let reply = d.handle_line(&line(&ControlCommand::Mode {
            mode: ddai_botctl::proto::ModeArg::Hold,
        }));
        assert!(reply.ok && reply.code.is_none());
        assert_eq!(reply.text, "did Mode(Some(Hold))");
        assert_eq!(bot.seen(), vec![BotCommand::Mode(Some(Mode::Hold))]);
        let entries = audit.0.lock().unwrap().clone();
        assert_eq!(entries.len(), 1);
        assert_eq!(
            (entries[0].session.as_str(), entries[0].cmd.as_str(), entries[0].outcome),
            (SESSION, "mode:hold", Outcome::Ok)
        );
        assert!(entries[0].ts_ms > 1_600_000_000_000, "a real timestamp");
        drop(bot);
    }

    #[test]
    fn a_refusal_by_the_bot_is_relayed_and_audited_as_failed() {
        let (d, audit, _bot) = setup(8.0, 1.0, |_| CommandReply::err("reset is on cooldown"));
        let reply = d.handle_line(&line(&ControlCommand::Kill {}));
        assert!(!reply.ok && reply.code.is_none());
        assert_eq!(reply.text, "reset is on cooldown");
        assert_eq!(audit.0.lock().unwrap()[0].outcome, Outcome::Failed);
    }

    #[test]
    fn nothing_that_is_not_a_known_command_reaches_the_bot() {
        let (d, audit, bot) = setup(1000.0, 1000.0, |_| CommandReply::ok("x"));
        for bad in [
            &br#"{"v":1,"session":"0a","cmd":{"type":"say","text":"hello everyone"}}"#[..],
            br#"{"v":1,"session":"0a","cmd":{"type":"quit"}}"#,
            br#"{"v":1,"session":"0a","cmd":{"type":"target","name":"someone"}}"#,
            br#"{"v":1,"session":"0a","cmd":{"type":"stop","say":"hi"}}"#,
            br#"{"v":2,"session":"0a","cmd":{"type":"stop"}}"#,
            br#"{"v":1,"session":"NOT HEX","cmd":{"type":"stop"}}"#,
            br#"{"v":1,"session":"0a","cmd":{"type":"goto","x":99999999,"y":0}}"#,
            br#"{"v":1,"session":"0a","cmd":{"type":"clip","note":"a\nb"}}"#,
            b"!say hi",
            b"hello",
            b"",
            b"\xff\xfe\x00",
            b"{",
        ] {
            let r = d.handle_line(bad);
            assert!(!r.ok, "{}", String::from_utf8_lossy(bad));
            assert_eq!(r.code, Some(ReplyCode::BadRequest));
        }
        thread::sleep(Duration::from_millis(30));
        assert!(bot.seen().is_empty(), "nothing reached the bot: {:?}", bot.seen());
        assert!(audit.0.lock().unwrap().iter().all(|e| e.outcome == Outcome::BadRequest));
    }

    #[test]
    fn commands_are_rate_limited_and_a_flood_does_not_flood_the_audit_log() {
        // Burst 3, no refill within the test.
        let (d, audit, bot) = setup(3.0, 0.000_001, |_| CommandReply::ok("x"));
        let cmd = line(&ControlCommand::Go {});
        let started = Instant::now();
        let replies: Vec<ControlReply> = (0..50).map(|_| d.handle_line(&cmd)).collect();
        let secs = started.elapsed().as_secs() as usize;
        assert_eq!(replies.iter().filter(|r| r.ok).count(), 3);
        assert!(
            replies[3..]
                .iter()
                .all(|r| !r.ok && r.code == Some(ReplyCode::RateLimited))
        );
        assert_eq!(bot.seen().len(), 3, "the refused ones never reached the bot");
        let entries = audit.0.lock().unwrap().clone();
        assert_eq!(entries.iter().filter(|e| e.outcome == Outcome::Ok).count(), 3);
        // At most one line per second of wall time (normally 1: the 47 refusals take microseconds; a starved CI runner
        // may stretch the loop over a second or two).
        let limited = entries.iter().filter(|e| e.outcome == Outcome::RateLimited).count();
        assert!(
            (1..=1 + secs).contains(&limited),
            "47 refusals over {secs} s left {limited} audit lines"
        );
        // Garbage costs tokens too (a malformed flood cannot be cheaper than a command flood).
        let (d, _a, bot) = setup(2.0, 0.000_001, |_| CommandReply::ok("x"));
        assert_eq!(d.handle_line(b"junk").code, Some(ReplyCode::BadRequest));
        assert_eq!(d.handle_line(b"junk").code, Some(ReplyCode::BadRequest));
        assert_eq!(d.handle_line(&cmd).code, Some(ReplyCode::RateLimited));
        assert!(bot.seen().is_empty());
    }

    #[test]
    fn the_audit_log_carries_tags_only_never_a_note_a_name_or_a_reply() {
        let (d, audit, _bot) = setup(100.0, 100.0, |_| {
            CommandReply::ok("saved 30 s to /x/SECRET-NICK-in-reply.clip")
        });
        d.handle_line(&line(&ControlCommand::Clip {
            note: "SECRET-NOTE with SECRET-NICK".into(),
        }));
        d.handle_line(&line(&ControlCommand::Goto { x: 5, y: 6 }));
        d.handle_line(&line(&ControlCommand::ReloadRelations {}));
        d.handle_line(br#"{"v":1,"session":"0a","cmd":{"type":"say","text":"SECRET-NICK says hi"}}"#);
        d.handle_line(b"SECRET-NICK");
        let entries = audit.0.lock().unwrap().clone();
        assert_eq!(entries.len(), 5);
        let all: String = entries.iter().map(|e| e.to_line() + "\n").collect();
        assert!(!all.contains("SECRET"), "{all}");
        assert!(
            all.contains(r#""cmd":"clip""#) && all.contains(r#""cmd":"goto:5,6""#),
            "{all}"
        );
        assert!(all.contains(r#""cmd":"relations:reload""#), "{all}");
        for e in &entries {
            let v: serde_json::Value = serde_json::from_str(&e.to_line()).expect("every line is JSON");
            assert!(v["ts_ms"].is_u64() && v["session"].is_string() && v["cmd"].is_string());
            assert_eq!(v.as_object().unwrap().len(), 4, "exactly ts, session, cmd, outcome");
        }
    }

    #[test]
    fn a_bot_that_does_not_answer_or_has_stopped_is_reported_not_waited_for_forever() {
        let (sender, inbox) = CommandBus::open();
        let audit = Arc::new(MemoryAudit::default());
        let d = Dispatcher::with_limits(
            sender,
            Arc::clone(&audit) as Arc<dyn AuditSink>,
            100.0,
            100.0,
            Duration::from_millis(60),
        );
        let started = Instant::now();
        let r = d.handle_line(&line(&ControlCommand::Go {})); // the inbox is never drained
        assert_eq!(r.code, Some(ReplyCode::Timeout));
        assert!(started.elapsed() < Duration::from_secs(20));
        drop(inbox);
        let r = d.handle_line(&line(&ControlCommand::Go {}));
        assert_eq!(r.code, Some(ReplyCode::Gone));
        let outcomes: Vec<Outcome> = audit.0.lock().unwrap().iter().map(|e| e.outcome).collect();
        assert_eq!(outcomes, vec![Outcome::Timeout, Outcome::Gone]);
    }

    #[test]
    fn lines_are_read_with_a_hard_size_limit() {
        let mut buf = Vec::new();
        let mut r = io::Cursor::new(b"abc\ndef\n".to_vec());
        assert_eq!(read_line_bounded(&mut r, &mut buf, 16).unwrap(), LineRead::Line);
        assert_eq!(buf, b"abc");
        assert_eq!(read_line_bounded(&mut r, &mut buf, 16).unwrap(), LineRead::Line);
        assert_eq!(buf, b"def");
        assert_eq!(read_line_bounded(&mut r, &mut buf, 16).unwrap(), LineRead::Eof);
        // Exactly at the limit (newline included) is fine, one more is not.
        let mut r = io::Cursor::new(b"abcd\n".to_vec());
        assert_eq!(read_line_bounded(&mut r, &mut buf, 5).unwrap(), LineRead::Line);
        let mut r = io::Cursor::new(b"abcde\n".to_vec());
        assert_eq!(read_line_bounded(&mut r, &mut buf, 5).unwrap(), LineRead::TooLong);
        // No newline within the limit, however long the stream is: never buffers more than max + 1.
        let mut r = io::Cursor::new(vec![b'x'; 1 << 20]);
        assert_eq!(read_line_bounded(&mut r, &mut buf, 64).unwrap(), LineRead::TooLong);
        assert!(buf.len() <= 65);
        // A partial line at EOF is not a request.
        let mut r = io::Cursor::new(b"abc".to_vec());
        assert_eq!(read_line_bounded(&mut r, &mut buf, 16).unwrap(), LineRead::Eof);
    }

    fn roundtrip(stream: &mut UnixStream, line: &[u8]) -> ControlReply {
        stream.write_all(line).unwrap();
        stream.write_all(b"\n").unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut reply = String::new();
        reader.read_line(&mut reply).unwrap();
        serde_json::from_str(reply.trim_end()).unwrap_or_else(|e| panic!("reply {reply:?}: {e}"))
    }

    #[test]
    fn the_socket_is_private_serves_requests_refuses_oversize_and_cleans_up() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bot").join(SOCKET_NAME);
        let (sender, inbox) = CommandBus::open();
        let _bot = FakeBot::start(inbox, |_| CommandReply::ok("fine"));
        let audit = Arc::new(MemoryAudit::default());
        let server = ControlServer::start(&path, sender.clone(), Arc::clone(&audit) as Arc<dyn AuditSink>).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        assert_eq!(
            std::fs::metadata(path.parent().unwrap()).unwrap().permissions().mode() & 0o777,
            0o700
        );
        // Two requests on one connection, strictly in turn.
        let mut c = UnixStream::connect(&path).unwrap();
        c.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
        let r = roundtrip(&mut c, &line(&ControlCommand::Stop {}));
        assert!(r.ok && r.text == "fine");
        let r = roundtrip(&mut c, br#"{"v":1,"session":"0a","cmd":{"type":"say","text":"hi"}}"#);
        assert_eq!(r.code, Some(ReplyCode::BadRequest));
        // Too long: refused, and the connection is closed (it cannot be resynchronised).
        let mut c2 = UnixStream::connect(&path).unwrap();
        c2.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
        let mut big = vec![b'x'; MAX_REQUEST_BYTES + 100];
        big.push(b'\n');
        c2.write_all(&big).unwrap();
        let mut reader = BufReader::new(c2.try_clone().unwrap());
        let mut reply = String::new();
        reader.read_line(&mut reply).unwrap();
        let r: ControlReply = serde_json::from_str(reply.trim_end()).unwrap();
        assert_eq!(r.code, Some(ReplyCode::BadRequest));
        reply.clear();
        assert_eq!(
            reader.read_line(&mut reply).unwrap(),
            0,
            "closed after the oversize line"
        );
        // A second bot does not steal the live socket.
        let (s2, _i2) = CommandBus::open();
        let err = ControlServer::start(&path, s2, Arc::clone(&audit) as Arc<dyn AuditSink>)
            .err()
            .unwrap();
        assert_eq!(err.kind(), io::ErrorKind::AddrInUse);
        drop(c);
        drop(server);
        assert!(!path.exists(), "the socket file is removed when the server stops");
    }

    #[test]
    fn only_a_few_connections_are_served_at_once() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(SOCKET_NAME);
        let (sender, inbox) = CommandBus::open();
        let _bot = FakeBot::start(inbox, |_| CommandReply::ok("fine"));
        let audit = Arc::new(MemoryAudit::default());
        let _server = ControlServer::start(&path, sender, audit as Arc<dyn AuditSink>).unwrap();
        let held: Vec<UnixStream> = (0..MAX_CONNECTIONS)
            .map(|_| {
                let mut s = UnixStream::connect(&path).unwrap();
                s.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
                // Proves each one is being served (and so counted).
                assert!(roundtrip(&mut s, &line(&ControlCommand::Go {})).ok);
                s
            })
            .collect();
        let mut extra = UnixStream::connect(&path).unwrap();
        extra.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
        let mut reply = String::new();
        BufReader::new(&mut extra).read_line(&mut reply).unwrap();
        let r: ControlReply = serde_json::from_str(reply.trim_end()).unwrap();
        assert_eq!(r.code, Some(ReplyCode::Busy));
        // Freeing a slot lets a new connection in.
        drop(held);
        let mut ok = false;
        for _ in 0..100 {
            let mut s = UnixStream::connect(&path).unwrap();
            s.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
            if roundtrip(&mut s, &line(&ControlCommand::Go {})).ok {
                ok = true;
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }
        assert!(ok, "a slot is free again after the others closed");
    }

    #[test]
    fn the_audit_file_is_private_append_only_and_rotates() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("logs").join("audit.log");
        let log = FileAudit::open_with_cap(&path, 300).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        let entry = |n: u64| AuditEntry {
            ts_ms: n,
            session: SESSION.to_string(),
            cmd: "go".to_string(),
            outcome: Outcome::Ok,
        };
        for n in 0..10 {
            log.record(&entry(n));
        }
        let now = std::fs::read_to_string(&path).unwrap();
        let old = std::fs::read_to_string(dir.path().join("logs").join("audit.log.1")).unwrap();
        assert!(!old.is_empty() && !now.is_empty(), "rotated once the cap passed");
        let total = old.lines().count() + now.lines().count();
        assert!((5..=10).contains(&total), "{total}");
        assert!(
            now.lines()
                .chain(old.lines())
                .all(|l| serde_json::from_str::<serde_json::Value>(l).is_ok())
        );
        assert_eq!(
            std::fs::metadata(dir.path().join("logs").join("audit.log.1"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        // Reopening appends.
        drop(log);
        let again = FileAudit::open(&path).unwrap();
        again.record(&entry(99));
        assert!(std::fs::read_to_string(&path).unwrap().lines().count() > now.lines().count());
    }
}
