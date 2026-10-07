//! The read-only live-state feed to the web unit (`ddai-web`, D-036: the bot and the web server are
//! separate units) — the "bot live bridge v1", `docs/formats.md` §21.
//!
//! The bot **listens** on a Unix socket (default `~/aiddnet/data/bot/live.sock`, mode `0600`, the
//! directory `0700`); the web unit connects and only ever reads. The bot never reads from the
//! socket: there is no control path here (the web control is a separate socket, `crate::control`,
//! task 5.6), and a client that writes is simply ignored. Messages are length-prefixed:
//! `u32 LE len | u8 kind | payload` where `len` counts the kind byte and the payload (at most
//! [`MAX_MESSAGE`]):
//!
//! | kind | name | payload |
//! |---|---|---|
//! | 1 | `HELLO` | `"DDBL"`, `u8` version (1) |
//! | 2 | `MAP` | JSON `{"name","sha256","w","h"}` (`sha256` hex of the `.map` file) |
//! | 3 | `PLAYERS` | JSON `{"own":id,"list":[{"id","name","team"}]}` |
//! | 4 | `FRAME` | a `DWLF` v1 live frame, byte for byte the web's binary `live` message (`docs/formats.md` §15.3) |
//! | 5 | `STATUS` | JSON, at most ~5 Hz: target, mode, brain, counters, latency percentiles, brain telemetry, and (5.6) connection, server, identity, wayblock, kill cooldown |
//! | 6 | `FLYMETA` | (7.4) JSON, the layout of the fly's visualisation stream (`docs/formats.md` §27.2); empty: the brain has none. Only to a client subscribed to the fly stream |
//! | 7 | `FLY` | (7.4) one binary `DFLY` v1 frame (`docs/formats.md` §27.1), decimated by the brain. Only to a client subscribed to the fly stream |
//! | 8 | `CHAT` | (5.10) JSON `{"team","cid","name","text"}`: one line of the server's chat, for display only (`docs/formats.md` §35). Never stored here |
//! | 9 | `PLAYERINFO` | (5.10) JSON `{"list":[{"id","clan","skin","cc","cb","cf","country","score","ping"}]}`: how each player looks and their numbers; resent while they change |
//!
//! **What a client may say (7.4).** The bridge is still read-only in effect: there is no control path. The one
//! thing the bot reads from a client is a subscription, `u32 LE len | u8 kind 1 | u8 mask` (bit 0: the fly stream),
//! so that the fly's frames are built only while somebody watches (`Bridge::fly_wanted`). Since 5.7 bit 1 of the mask
//! says "the site is open" (`Bridge::watched` is true while any client has any bit set), which the offline demo uses to
//! pause itself. Anything else a client writes is read and discarded; a message longer than [`MAX_CLIENT_MESSAGE`]
//! drops the client.
//!
//! **Names.** `PLAYERS.name` is the salted-hash tag (`c12-9f3a01bc`) unless the bot was started with
//! `--web-names`; real nicknames never leave the process otherwise (D-040, `CLAUDE.md`). The same holds for `CHAT` (the
//! sender is the tag, and known names and clans inside the text become tags) and for `PLAYERINFO.clan` (empty without
//! `--web-names`).
//!
//! **Flow control.** Each client has a bounded output buffer; the socket is non-blocking. A client
//! that cannot keep up — its buffer would pass [`MAX_PENDING`] — is dropped (it reconnects and gets
//! `HELLO`, the current `MAP` and `PLAYERS` again). A slow web unit can therefore never stall the
//! bot's decisions: every call here is non-blocking and bounded.

use std::io::{self, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};

use serde::Serialize;

/// `u8` kinds.
pub mod kind {
    pub const HELLO: u8 = 1;
    pub const MAP: u8 = 2;
    pub const PLAYERS: u8 = 3;
    pub const FRAME: u8 = 4;
    pub const STATUS: u8 = 5;
    pub const FLYMETA: u8 = 6;
    pub const FLY: u8 = 7;
    pub const CHAT: u8 = 8;
    pub const PLAYERINFO: u8 = 9;
}

/// What a client may send: `kind`s of its messages.
pub mod client_kind {
    /// Payload: one byte, a bit mask of the streams the client wants (bit 0: the fly's).
    pub const SUBSCRIBE: u8 = 1;
}

/// Bit of the `SUBSCRIBE` mask for the fly stream.
pub const SUBSCRIBE_FLY: u8 = 1;
/// Bit of the `SUBSCRIBE` mask (task 5.7): somebody has the site open, so the game itself (map, players, frames) is
/// being watched. The live bot does nothing with it; the offline demo (`ddnet-ai fly watch --pause-idle`) pauses its
/// game while no client says either bit.
pub const SUBSCRIBE_VIEW: u8 = 2;
/// Longest client message (kind + payload) the bot reads; a longer one drops the client.
pub const MAX_CLIENT_MESSAGE: usize = 16;

/// Protocol version in `HELLO`.
pub const VERSION: u8 = 1;
/// Largest message (kind + payload) either side accepts.
pub const MAX_MESSAGE: usize = 1 << 20;
/// Bytes a client may have queued before it is dropped.
pub const MAX_PENDING: usize = 512 * 1024;

const MAGIC: &[u8; 4] = b"DWLF";
const FRAME_VERSION: u8 = 1;
const HOOK_FLYING: i32 = 4;
const HOOK_GRABBED: i32 = 5;

/// One character of a `FRAME` (the web's `CharacterState`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameChar {
    pub id: u8,
    pub alive: bool,
    pub frozen: bool,
    pub deep_frozen: bool,
    pub live_frozen: bool,
    pub team: u8,
    pub weapon: u8,
    pub x: i32,
    pub y: i32,
    pub aim_x: i32,
    pub aim_y: i32,
    pub hook_state: i32,
    pub hook_x: i32,
    pub hook_y: i32,
    /// -1 = none.
    pub hooked_id: i32,
}

/// Encodes a `DWLF` v1 frame (`docs/formats.md` §15.3): 12-byte header, then 26 bytes per character.
pub fn encode_frame(tick: u32, chars: &[FrameChar], out: &mut Vec<u8>) {
    out.clear();
    out.extend_from_slice(MAGIC);
    out.push(FRAME_VERSION);
    out.push(0);
    out.extend_from_slice(&tick.to_le_bytes());
    out.extend_from_slice(&(chars.len().min(u16::MAX as usize) as u16).to_le_bytes());
    for c in chars {
        let mut flags = 0u8;
        if c.alive {
            flags |= 1;
        }
        if c.frozen {
            flags |= 1 << 1;
        }
        if c.deep_frozen {
            flags |= 1 << 2;
        }
        if c.live_frozen {
            flags |= 1 << 3;
        }
        if c.hook_state == HOOK_FLYING || c.hook_state == HOOK_GRABBED {
            flags |= 1 << 4;
        }
        out.push(c.id);
        out.push(flags);
        out.push(c.team);
        out.push(c.weapon);
        out.extend_from_slice(&c.x.to_le_bytes());
        out.extend_from_slice(&c.y.to_le_bytes());
        out.extend_from_slice(&(c.aim_x.clamp(i16::MIN.into(), i16::MAX.into()) as i16).to_le_bytes());
        out.extend_from_slice(&(c.aim_y.clamp(i16::MIN.into(), i16::MAX.into()) as i16).to_le_bytes());
        out.extend_from_slice(&c.hook_x.to_le_bytes());
        out.extend_from_slice(&c.hook_y.to_le_bytes());
        out.push(if c.hooked_id >= 0 {
            c.hooked_id as i8 as u8
        } else {
            0xFF
        });
        out.push(0);
    }
}

/// `MAP` payload.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct MapMessage {
    pub name: String,
    pub sha256: String,
    pub w: u32,
    pub h: u32,
}

/// One `PLAYERS` entry.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PlayerEntry {
    pub id: i32,
    pub name: String,
    pub team: i32,
}

/// `PLAYERS` payload.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PlayersMessage {
    pub own: i32,
    pub list: Vec<PlayerEntry>,
}

/// One `PLAYERINFO` entry (task 5.10): how a player looks (skin and colours, from `ClientInfo`) and their live numbers
/// (`PlayerInfo`). `clan` is empty unless real names are sent.
#[derive(Debug, Clone, Serialize, PartialEq, Eq, Default)]
pub struct PlayerInfoEntry {
    pub id: i32,
    pub clan: String,
    pub skin: String,
    /// `use_custom_color`.
    pub cc: bool,
    /// `color_body`, `color_feet`: DDNet's packed HSL.
    pub cb: i32,
    pub cf: i32,
    pub country: i32,
    pub score: i32,
    pub ping: i32,
}

/// `PLAYERINFO` payload.
#[derive(Debug, Clone, Serialize, PartialEq, Eq, Default)]
pub struct PlayerInfoMessage {
    pub list: Vec<PlayerInfoEntry>,
}

/// Most bytes of a chat line's text or name that leave the bot (DDNet's own limit is 256).
pub const MAX_CHAT_TEXT: usize = 512;
pub const MAX_CHAT_NAME: usize = 64;

/// `s` cut to at most `max` bytes, at a character boundary.
pub fn cut_at(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// `CHAT` payload (task 5.10): one line the server's chat showed, for the web unit to display. **Display only**: the bot
/// never writes chat (D-007; the one exception is the typed `/kill`, D-078), and nothing here is stored or logged.
/// `team`: 0 all, 1 team, 2 whisper sent, 3 whisper received (`CNetMsg_Sv_Chat::m_Team`); `cid`: the sender's client id,
/// -1 for the server itself. `name` is the sender's tag (or nickname with `--web-names`), empty for the server.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub team: i32,
    pub cid: i32,
    pub name: String,
    pub text: String,
}

/// `STATUS` payload.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct StatusMessage {
    pub tick: i32,
    pub own: i32,
    pub target: i32,
    pub mode: String,
    pub brain: String,
    pub alive: bool,
    pub frozen: bool,
    pub blocks: u32,
    pub blocked_by: u32,
    pub self_kills: u64,
    pub decisions: u64,
    pub collapsed: u64,
    pub decide_p50_us: u32,
    pub decide_p99_us: u32,
    pub brain_p99_us: u32,
    pub overhead_p99_us: u32,
    /// The brain's own telemetry, parsed JSON (or null).
    pub telemetry: Option<serde_json::Value>,
    // Additive since task 5.6 (the web status panel).
    /// The session is in the game.
    pub connected: bool,
    /// The game server's address, `ip:port`, and the map being played.
    pub server: String,
    pub map: String,
    /// The bot's own nickname, clan and skin.
    pub name: String,
    pub clan: String,
    pub skin: String,
    /// The target's tag (`c<id>-<hash>`), never a nickname.
    pub target_tag: Option<String>,
    /// The wayblock line and the walk's progress (empty when none).
    pub wb: String,
    pub goto: String,
    pub deaths: u64,
    pub clips_saved: u64,
    /// Ticks until `Cl_Kill` is allowed again, in server ticks (50 per second: the cooldown is 500 ticks = 10 s); 0: now.
    pub kill_cooldown_ticks: i32,
    /// Additive since task 4.9b: the server has paused the bot (its own `DDNetPlayer` flag `SPEC`/`PAUSED`: the owner's `/pause` or
    /// `/spec`); the bot idles until the flag clears. The site shows «на паузе».
    pub paused: bool,
    /// Additive since task 5.13: the finishing mode the process was started with (`ddnet-ai play --finish`, D-097): `off`, `target`
    /// or `full`. `full`'s drag shaping only acts while the brain is the hybrid; the target rule acts with every brain. The site's
    /// «Бот» card shows it, so a launch that did not take (an old unit without `--finish`) is visible.
    pub finish: String,
    /// Additive since task 4.11 (D-102): `"off"` while the duel switch is on (`--no-selfkill` or the marker `bot/selfkill.off`: the bot
    /// never kills itself), else `"on"` (the default: the unstick and the other self-kills work).
    pub selfkill: String,
    /// Additive since task 5.15 (D-103, D-104): `"on"` while the smart wayblock (`--wb-smart on`) is on, else `"off"` (the default). The
    /// site's «Бот» card shows it, so a launch that did not take (an old unit without `--wb-smart`) is visible.
    pub wb_smart: String,
}

/// Binds a Unix socket at `path` that only its owner can reach: replaces a stale socket file (but refuses to touch
/// anything that is not a socket, and never steals a live bot's socket), with the directory `0700` and the socket
/// `0600`. Shared by the live bridge and the control channel (`crate::control`), which live side by side.
pub fn bind_private_socket(path: &Path) -> io::Result<UnixListener> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
    }
    match std::fs::symlink_metadata(path) {
        Ok(m) => {
            use std::os::unix::fs::FileTypeExt;
            if !m.file_type().is_socket() {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    format!("{} exists and is not a socket", path.display()),
                ));
            }
            // A live bot already listening there? Do not steal its socket.
            if UnixStream::connect(path).is_ok() {
                return Err(io::Error::new(
                    io::ErrorKind::AddrInUse,
                    format!("another bot is already serving {}", path.display()),
                ));
            }
            std::fs::remove_file(path)?;
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    // The socket file is created with the process umask; close the window in which it is looser than 0600.
    let listener = UnixListener::bind(path)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    Ok(listener)
}

struct Client {
    stream: UnixStream,
    pending: Vec<u8>,
    /// Bytes read from the client not yet forming a whole message.
    inbox: Vec<u8>,
    /// Subscribed to the fly stream.
    fly: bool,
    /// The site is open (task 5.7).
    view: bool,
}

/// The publisher.
pub struct Bridge {
    listener: UnixListener,
    path: PathBuf,
    clients: Vec<Client>,
    map_msg: Option<Vec<u8>>,
    players_msg: Option<Vec<u8>>,
    /// The last `PLAYERINFO` message (task 5.10), what a client that connects later is greeted with after the roster.
    player_info_msg: Option<Vec<u8>>,
    /// The `FLYMETA` message (an empty payload when the brain has no stream), sent to a client when it subscribes.
    fly_meta_msg: Option<Vec<u8>>,
    scratch: Vec<u8>,
    frame_buf: Vec<u8>,
    /// Clients dropped for being too slow (telemetry).
    dropped: u64,
}

fn message(kind: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(5 + payload.len());
    out.extend_from_slice(&((payload.len() + 1) as u32).to_le_bytes());
    out.push(kind);
    out.extend_from_slice(payload);
    out
}

impl Bridge {
    /// Binds `path` ([`bind_private_socket`]) and starts listening.
    pub fn bind(path: &Path) -> io::Result<Bridge> {
        let listener = bind_private_socket(path)?;
        listener.set_nonblocking(true)?;
        Ok(Bridge {
            listener,
            path: path.to_path_buf(),
            clients: Vec::new(),
            map_msg: None,
            players_msg: None,
            player_info_msg: None,
            fly_meta_msg: None,
            scratch: Vec::with_capacity(8192),
            frame_buf: Vec::with_capacity(8192),
            dropped: 0,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn clients(&self) -> usize {
        self.clients.len()
    }

    pub fn dropped_clients(&self) -> u64 {
        self.dropped
    }

    /// Accepts waiting clients (non-blocking) and greets them with `HELLO`, `MAP`, `PLAYERS`.
    pub fn accept_pending(&mut self) {
        loop {
            match self.listener.accept() {
                Ok((stream, _)) => {
                    if stream.set_nonblocking(true).is_err() {
                        continue;
                    }
                    let mut client = Client {
                        stream,
                        pending: Vec::new(),
                        inbox: Vec::new(),
                        fly: false,
                        view: false,
                    };
                    let mut hello = MAGIC_HELLO.to_vec();
                    hello.push(VERSION);
                    client.pending.extend_from_slice(&message(kind::HELLO, &hello));
                    if let Some(m) = &self.map_msg {
                        client.pending.extend_from_slice(m);
                    }
                    if let Some(p) = &self.players_msg {
                        client.pending.extend_from_slice(p);
                    }
                    if let Some(p) = &self.player_info_msg {
                        client.pending.extend_from_slice(p);
                    }
                    if flush(&mut client) {
                        self.clients.push(client);
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(_) => break,
            }
        }
        self.poll_clients();
    }

    /// Whether some client is subscribed to the fly stream: the brain builds frames only then.
    pub fn fly_wanted(&self) -> bool {
        self.clients.iter().any(|c| c.fly)
    }

    /// Whether some client has subscribed to anything (the fly stream or the plain "the site is open"): somebody may be
    /// looking. A client that is connected but has said nothing (a web unit with no browser open) does not count.
    pub fn watched(&self) -> bool {
        self.clients.iter().any(|c| c.fly || c.view)
    }

    /// Reads (non-blocking, bounded) what the clients said: subscriptions. A subscriber gets the current `FLYMETA` at once.
    fn poll_clients(&mut self) {
        let meta = self.fly_meta_msg.as_deref();
        let mut dropped = 0;
        self.clients.retain_mut(|c| {
            let keep = read_client(c, meta);
            if !keep {
                dropped += 1;
            }
            keep
        });
        self.dropped += dropped;
    }

    /// The fly stream's description changed (a brain switch, or the first one): a client subscribed to the stream is told
    /// now, one that subscribes later when it does. `None`: the brain has no stream.
    pub fn set_fly_meta(&mut self, meta: Option<&str>) {
        let msg = message(kind::FLYMETA, meta.map_or(&[][..], str::as_bytes));
        self.fly_meta_msg = Some(msg.clone());
        self.broadcast_fly(&msg);
    }

    /// One `FLY` frame to the subscribed clients (a few Hz, decimated by the brain).
    pub fn send_fly(&mut self, frame: &[u8]) {
        if !self.fly_wanted() || frame.len() + 1 > MAX_MESSAGE {
            return;
        }
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.clear();
        scratch.extend_from_slice(&((frame.len() + 1) as u32).to_le_bytes());
        scratch.push(kind::FLY);
        scratch.extend_from_slice(frame);
        self.broadcast_fly(&scratch);
        self.scratch = scratch;
    }

    fn broadcast_fly(&mut self, msg: &[u8]) {
        let mut dropped = 0;
        self.clients.retain_mut(|c| {
            if !c.fly {
                return true;
            }
            if c.pending.len() + msg.len() > MAX_PENDING {
                dropped += 1;
                return false;
            }
            c.pending.extend_from_slice(msg);
            let keep = flush(c);
            if !keep {
                dropped += 1;
            }
            keep
        });
        self.dropped += dropped;
    }

    fn broadcast(&mut self, msg: &[u8]) {
        let mut dropped = 0;
        self.clients.retain_mut(|c| {
            if c.pending.len() + msg.len() > MAX_PENDING {
                dropped += 1;
                return false;
            }
            c.pending.extend_from_slice(msg);
            let keep = flush(c);
            if !keep {
                dropped += 1;
            }
            keep
        });
        self.dropped += dropped;
    }

    /// The map changed (also what late joiners are greeted with).
    pub fn send_map(&mut self, m: &MapMessage) {
        let Ok(json) = serde_json::to_vec(m) else { return };
        let msg = message(kind::MAP, &json);
        self.map_msg = Some(msg.clone());
        // A new map invalidates the roster message of the old one.
        self.players_msg = None;
        self.player_info_msg = None;
        self.broadcast(&msg);
    }

    /// The roster changed.
    pub fn send_players(&mut self, p: &PlayersMessage) {
        let Ok(json) = serde_json::to_vec(p) else { return };
        let msg = message(kind::PLAYERS, &json);
        self.players_msg = Some(msg.clone());
        self.broadcast(&msg);
    }

    /// How each player looks and their numbers (task 5.10). The last one is kept for a client that connects later (like the
    /// roster), since the runner sends it only when something differs.
    pub fn send_player_info(&mut self, m: &PlayerInfoMessage) {
        let Ok(json) = serde_json::to_vec(m) else { return };
        if json.len() + 1 > MAX_MESSAGE {
            return;
        }
        let msg = message(kind::PLAYERINFO, &json);
        self.player_info_msg = Some(msg.clone());
        self.broadcast(&msg);
    }

    /// One chat line of the server's chat, to display (task 5.10). Text and name are cut at the caps; nothing is kept.
    pub fn send_chat(&mut self, line: &ChatMessage) {
        if self.clients.is_empty() {
            return;
        }
        let line = ChatMessage {
            team: line.team,
            cid: line.cid,
            name: cut_at(&line.name, MAX_CHAT_NAME).to_string(),
            text: cut_at(&line.text, MAX_CHAT_TEXT).to_string(),
        };
        let Ok(json) = serde_json::to_vec(&line) else { return };
        let msg = message(kind::CHAT, &json);
        self.broadcast(&msg);
    }

    /// One frame (every snapshot; the web side throttles what it forwards to browsers).
    pub fn send_frame(&mut self, tick: u32, chars: &[FrameChar]) {
        if self.clients.is_empty() {
            return;
        }
        let mut frame = std::mem::take(&mut self.frame_buf);
        encode_frame(tick, chars, &mut frame);
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.clear();
        scratch.extend_from_slice(&((frame.len() + 1) as u32).to_le_bytes());
        scratch.push(kind::FRAME);
        scratch.extend_from_slice(&frame);
        self.broadcast(&scratch);
        self.frame_buf = frame;
        self.scratch = scratch;
    }

    /// The bot's own status (a few Hz).
    pub fn send_status(&mut self, s: &StatusMessage) {
        if self.clients.is_empty() {
            return;
        }
        let Ok(json) = serde_json::to_vec(s) else { return };
        self.send_status_json(&json);
    }

    /// A `STATUS` message with a payload the caller has already built (a JSON object): the offline demo describes
    /// itself with its own small object (`docs/formats.md` §28) instead of the bot's [`StatusMessage`].
    pub fn send_status_json(&mut self, json: &[u8]) {
        if self.clients.is_empty() || json.len() + 1 > MAX_MESSAGE {
            return;
        }
        let msg = message(kind::STATUS, json);
        self.broadcast(&msg);
    }
}

const MAGIC_HELLO: &[u8; 4] = b"DDBL";

/// Reads at most this many chunks of a client per poll (a client that writes without pause cannot hold the decision thread).
const MAX_READS_PER_POLL: usize = 64;

/// Reads what `c` has sent so far and applies its subscriptions, message by message as they arrive (a burst of small
/// subscriptions between two polls is fine; only a *partial* message is kept between reads, and it is at most
/// `4 + MAX_CLIENT_MESSAGE` bytes). `false` when the client is gone, misbehaves, or its queue is past [`MAX_PENDING`]:
/// the cap is checked for every client on every poll and before every append here, not only when the bot sends something.
fn read_client(c: &mut Client, fly_meta: Option<&[u8]>) -> bool {
    use std::io::Read;
    if c.pending.len() > MAX_PENDING {
        return false;
    }
    let mut buf = [0u8; 64];
    for _ in 0..MAX_READS_PER_POLL {
        match c.stream.read(&mut buf) {
            Ok(0) => return false,
            Ok(n) => {
                c.inbox.extend_from_slice(&buf[..n]);
                if !apply_messages(c, fly_meta) || c.inbox.len() > 4 + MAX_CLIENT_MESSAGE {
                    return false;
                }
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => return false,
        }
    }
    flush(c) && c.pending.len() <= MAX_PENDING
}

/// Applies every whole message in `c.inbox`; `false` for a malformed one or a queue that would pass [`MAX_PENDING`].
fn apply_messages(c: &mut Client, fly_meta: Option<&[u8]>) -> bool {
    while c.inbox.len() >= 4 {
        let len = u32::from_le_bytes([c.inbox[0], c.inbox[1], c.inbox[2], c.inbox[3]]) as usize;
        if len == 0 || len > MAX_CLIENT_MESSAGE {
            return false;
        }
        if c.inbox.len() < 4 + len {
            break;
        }
        let kind = c.inbox[4];
        let payload_is_one_byte = len == 2;
        let mask = c.inbox.get(5).copied().unwrap_or(0);
        c.inbox.drain(..4 + len);
        if kind == client_kind::SUBSCRIBE && payload_is_one_byte {
            c.view = mask & SUBSCRIBE_VIEW != 0;
            let want = mask & SUBSCRIBE_FLY != 0;
            if want && !c.fly {
                // A new subscriber learns the layout before its first frame.
                if let Some(meta) = fly_meta {
                    if c.pending.len() + meta.len() > MAX_PENDING {
                        return false;
                    }
                    c.pending.extend_from_slice(meta);
                }
            }
            c.fly = want;
        }
    }
    true
}

impl Drop for Bridge {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Writes as much of `c.pending` as the socket takes; false when the client is gone.
fn flush(c: &mut Client) -> bool {
    while !c.pending.is_empty() {
        match c.stream.write(&c.pending) {
            Ok(0) => return false,
            Ok(n) => {
                c.pending.drain(..n);
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => return true,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => return false,
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn read_message(s: &mut UnixStream) -> (u8, Vec<u8>) {
        let mut len = [0u8; 4];
        s.read_exact(&mut len).unwrap();
        let len = u32::from_le_bytes(len) as usize;
        let mut body = vec![0u8; len];
        s.read_exact(&mut body).unwrap();
        (body[0], body[1..].to_vec())
    }

    fn sample_char() -> FrameChar {
        FrameChar {
            id: 3,
            alive: true,
            frozen: true,
            deep_frozen: false,
            live_frozen: false,
            team: 1,
            weapon: 0,
            x: 0x0102_0304,
            y: -2,
            aim_x: 300,
            aim_y: -70000,
            hook_state: 5,
            hook_x: 5,
            hook_y: 6,
            hooked_id: 7,
        }
    }

    /// The same golden bytes as `ddai-web`'s `live::frame` test of the documented layout: the two
    /// implementations are independent and must agree byte for byte.
    #[test]
    fn a_frame_is_the_documented_dwlf_v1_layout() {
        let mut out = Vec::new();
        encode_frame(0x0A0B_0C0D, &[sample_char()], &mut out);
        let mut want: Vec<u8> = Vec::new();
        want.extend_from_slice(b"DWLF");
        want.extend_from_slice(&[1, 0]); // version, reserved
        want.extend_from_slice(&0x0A0B_0C0Du32.to_le_bytes());
        want.extend_from_slice(&1u16.to_le_bytes());
        want.extend_from_slice(&[3, 0b1_0011, 1, 0]); // id, ALIVE|FROZEN|HOOK_VISIBLE, team, weapon
        want.extend_from_slice(&0x0102_0304i32.to_le_bytes());
        want.extend_from_slice(&(-2i32).to_le_bytes());
        want.extend_from_slice(&300i16.to_le_bytes());
        want.extend_from_slice(&i16::MIN.to_le_bytes()); // clamped
        want.extend_from_slice(&5i32.to_le_bytes());
        want.extend_from_slice(&6i32.to_le_bytes());
        want.extend_from_slice(&[7, 0]); // hooked id, reserved
        assert_eq!(out, want);
        assert_eq!(out.len(), 12 + 26);
        // No one hooked, no hook out.
        let mut c = sample_char();
        c.hooked_id = -1;
        c.hook_state = 0;
        encode_frame(1, &[c], &mut out);
        assert_eq!(out[12 + 1] & 0b1_0000, 0, "no HOOK_VISIBLE");
        assert_eq!(out[12 + 24], 0xFF, "-1 is 0xFF");
    }

    #[test]
    fn clients_get_hello_map_and_players_then_frames_and_status() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("b").join("live.sock");
        let mut bridge = Bridge::bind(&path).unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "the socket is private");
        assert_eq!(
            std::fs::metadata(path.parent().unwrap()).unwrap().permissions().mode() & 0o777,
            0o700
        );
        bridge.send_map(&MapMessage {
            name: "Copy Love Box".into(),
            sha256: "ab".repeat(32),
            w: 10,
            h: 20,
        });
        bridge.send_players(&PlayersMessage {
            own: 0,
            list: vec![PlayerEntry {
                id: 0,
                name: "c0-deadbeef".into(),
                team: 0,
            }],
        });
        let mut client = UnixStream::connect(&path).unwrap();
        client
            .set_read_timeout(Some(std::time::Duration::from_secs(2)))
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        bridge.accept_pending();
        assert_eq!(bridge.clients(), 1);
        let (k, p) = read_message(&mut client);
        assert_eq!((k, p.as_slice()), (kind::HELLO, &b"DDBL\x01"[..]));
        let (k, p) = read_message(&mut client);
        assert_eq!(k, kind::MAP);
        assert!(String::from_utf8(p).unwrap().contains("Copy Love Box"));
        let (k, p) = read_message(&mut client);
        assert_eq!(k, kind::PLAYERS);
        assert!(String::from_utf8(p).unwrap().contains("c0-deadbeef"));
        bridge.send_frame(77, &[sample_char()]);
        let (k, p) = read_message(&mut client);
        assert_eq!(k, kind::FRAME);
        assert_eq!(&p[..4], b"DWLF");
        assert_eq!(u32::from_le_bytes(p[6..10].try_into().unwrap()), 77);
        bridge.send_status(&StatusMessage {
            tick: 77,
            own: 0,
            target: 3,
            mode: "fight".into(),
            brain: "planner".into(),
            alive: true,
            frozen: false,
            blocks: 1,
            blocked_by: 0,
            self_kills: 0,
            decisions: 5,
            collapsed: 0,
            decide_p50_us: 100,
            decide_p99_us: 900,
            brain_p99_us: 800,
            overhead_p99_us: 100,
            telemetry: Some(serde_json::json!({"x": 1})),
            connected: true,
            server: "127.0.0.1:8303".into(),
            map: "Copy Love Box".into(),
            name: "bot".into(),
            clan: "Neuroset".into(),
            skin: "default".into(),
            target_tag: Some("c3-deadbeef".into()),
            wb: "WB: off".into(),
            goto: String::new(),
            deaths: 2,
            clips_saved: 1,
            kill_cooldown_ticks: 120,
            paused: false,
            finish: "target".into(),
            selfkill: "off".into(),
            wb_smart: "on".into(),
        });
        let (k, p) = read_message(&mut client);
        assert_eq!(k, kind::STATUS);
        let v: serde_json::Value = serde_json::from_slice(&p).unwrap();
        assert_eq!(v["target"], 3);
        assert_eq!(v["telemetry"]["x"], 1);
        assert_eq!(v["target_tag"], "c3-deadbeef");
        assert_eq!(v["kill_cooldown_ticks"], 120);
        assert_eq!(v["finish"], "target");
        assert_eq!(v["selfkill"], "off");
        assert_eq!(v["wb_smart"], "on");
    }

    #[test]
    fn a_client_that_never_reads_is_dropped_instead_of_blocking_the_bot() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("live.sock");
        let mut bridge = Bridge::bind(&path).unwrap();
        let _slow = UnixStream::connect(&path).unwrap(); // connects and never reads
        std::thread::sleep(std::time::Duration::from_millis(50));
        bridge.accept_pending();
        assert_eq!(bridge.clients(), 1);
        let chars: Vec<FrameChar> = (0..64).map(|i| FrameChar { id: i, ..sample_char() }).collect();
        let started = std::time::Instant::now();
        for tick in 0..2000 {
            bridge.send_frame(tick, &chars);
        }
        assert!(started.elapsed() < std::time::Duration::from_secs(2), "never blocks");
        assert_eq!(bridge.clients(), 0, "dropped once its buffer passed the cap");
        assert_eq!(bridge.dropped_clients(), 1);
    }

    fn write_client(c: &mut UnixStream, kind: u8, payload: &[u8]) {
        use std::io::Write as _;
        let mut m = ((payload.len() + 1) as u32).to_le_bytes().to_vec();
        m.push(kind);
        m.extend_from_slice(payload);
        c.write_all(&m).unwrap();
    }

    fn connect(bridge: &mut Bridge, path: &Path) -> UnixStream {
        let mut client = UnixStream::connect(path).unwrap();
        client
            .set_read_timeout(Some(std::time::Duration::from_secs(2)))
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(30));
        bridge.accept_pending();
        let (k, _) = read_message(&mut client);
        assert_eq!(k, kind::HELLO);
        client
    }

    fn settle(bridge: &mut Bridge) {
        std::thread::sleep(std::time::Duration::from_millis(30));
        bridge.accept_pending();
    }

    #[test]
    fn the_fly_stream_goes_only_to_subscribers_and_only_while_there_is_one() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("live.sock");
        let mut bridge = Bridge::bind(&path).unwrap();
        bridge.set_fly_meta(Some("{\"v\":1}"));
        let mut watcher = connect(&mut bridge, &path);
        let mut other = connect(&mut bridge, &path);
        assert!(!bridge.fly_wanted(), "nobody subscribed yet");
        // Nothing is queued without a subscriber.
        bridge.send_fly(b"DFLYxx");
        assert!(bridge.clients.iter().all(|c| c.pending.is_empty()));

        write_client(&mut watcher, client_kind::SUBSCRIBE, &[SUBSCRIBE_FLY]);
        settle(&mut bridge);
        assert!(bridge.fly_wanted());
        // The subscriber is told the layout first, then gets frames; the other client gets neither.
        let (k, p) = read_message(&mut watcher);
        assert_eq!((k, p.as_slice()), (kind::FLYMETA, &b"{\"v\":1}"[..]));
        bridge.send_fly(b"DFLYframe");
        let (k, p) = read_message(&mut watcher);
        assert_eq!((k, p.as_slice()), (kind::FLY, &b"DFLYframe"[..]));
        // The other client has received nothing since its HELLO.
        other
            .set_read_timeout(Some(std::time::Duration::from_millis(100)))
            .unwrap();
        let mut one = [0u8; 1];
        let e = std::io::Read::read(&mut other, &mut one).unwrap_err();
        assert!(
            matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut),
            "{e:?}"
        );

        // A brain switch re-sends the layout to subscribers (an empty one: no stream).
        bridge.set_fly_meta(None);
        let (k, p) = read_message(&mut watcher);
        assert_eq!((k, p.len()), (kind::FLYMETA, 0));

        // Unsubscribing stops the frames and `fly_wanted`.
        write_client(&mut watcher, client_kind::SUBSCRIBE, &[0]);
        settle(&mut bridge);
        assert!(!bridge.fly_wanted());
        bridge.send_fly(b"DFLYlate");
        assert!(bridge.clients.iter().all(|c| c.pending.is_empty()));
    }

    /// Task 5.7: `watched` is true while any client has any subscription bit set; the view bit alone does not subscribe the
    /// client to the fly stream, and a client that is only connected does not count.
    #[test]
    fn watched_means_some_client_said_it_is_looking_and_the_view_bit_is_not_the_fly_stream() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("demo.sock");
        let mut bridge = Bridge::bind(&path).unwrap();
        let mut web = connect(&mut bridge, &path);
        assert!(!bridge.watched(), "connected but silent: nobody is looking");
        write_client(&mut web, client_kind::SUBSCRIBE, &[SUBSCRIBE_VIEW]);
        settle(&mut bridge);
        assert!(bridge.watched());
        assert!(!bridge.fly_wanted(), "the view bit is not the fly stream");
        bridge.send_fly(b"DFLYxx");
        assert!(
            bridge.clients.iter().all(|c| c.pending.is_empty()),
            "no fly frame for it"
        );
        write_client(&mut web, client_kind::SUBSCRIBE, &[SUBSCRIBE_VIEW | SUBSCRIBE_FLY]);
        settle(&mut bridge);
        assert!(bridge.watched() && bridge.fly_wanted());
        write_client(&mut web, client_kind::SUBSCRIBE, &[SUBSCRIBE_FLY]);
        settle(&mut bridge);
        assert!(bridge.watched(), "the fly alone is watching too");
        write_client(&mut web, client_kind::SUBSCRIBE, &[0]);
        settle(&mut bridge);
        assert!(!bridge.watched());
        // A closed client stops counting.
        write_client(&mut web, client_kind::SUBSCRIBE, &[SUBSCRIBE_VIEW]);
        settle(&mut bridge);
        assert!(bridge.watched());
        drop(web);
        settle(&mut bridge);
        assert!(!bridge.watched());
    }

    /// Task 5.7: the demo's own status object goes out as a plain `STATUS` message, and nothing is queued without a client.
    #[test]
    fn a_prebuilt_status_object_is_sent_as_a_status_message() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("demo.sock");
        let mut bridge = Bridge::bind(&path).unwrap();
        bridge.send_status_json(b"{\"demo\":true}");
        let mut c = connect(&mut bridge, &path);
        bridge.send_status_json(b"{\"demo\":true}");
        let (k, p) = read_message(&mut c);
        assert_eq!((k, p.as_slice()), (kind::STATUS, &b"{\"demo\":true}"[..]));
    }

    #[test]
    fn a_closed_subscriber_stops_the_demand_and_other_client_messages_are_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("live.sock");
        let mut bridge = Bridge::bind(&path).unwrap();
        let mut c = connect(&mut bridge, &path);
        // An unknown kind, a short subscribe and a subscribe with a wrong payload size change nothing.
        write_client(&mut c, 99, &[1, 2, 3]);
        write_client(&mut c, client_kind::SUBSCRIBE, &[]);
        write_client(&mut c, client_kind::SUBSCRIBE, &[1, 1]);
        settle(&mut bridge);
        assert_eq!(bridge.clients(), 1);
        assert!(!bridge.fly_wanted());
        write_client(&mut c, client_kind::SUBSCRIBE, &[0xFF]);
        settle(&mut bridge);
        assert!(bridge.fly_wanted());
        drop(c);
        settle(&mut bridge);
        assert_eq!(bridge.clients(), 0, "a closed client is noticed on the read");
        assert!(!bridge.fly_wanted());
    }

    #[test]
    fn a_client_that_sends_an_oversized_message_or_floods_is_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("live.sock");
        let mut bridge = Bridge::bind(&path).unwrap();
        let mut c = connect(&mut bridge, &path);
        use std::io::Write as _;
        c.write_all(&(1000u32).to_le_bytes()).unwrap();
        c.write_all(&[1; 8]).unwrap();
        settle(&mut bridge);
        assert_eq!(bridge.clients(), 0, "a message longer than the cap");
        let mut c = connect(&mut bridge, &path);
        c.write_all(&[0u8; 4096]).unwrap();
        settle(&mut bridge);
        assert_eq!(bridge.clients(), 0, "a zero length is not a message");
    }

    #[test]
    fn a_client_that_toggles_its_subscription_and_never_reads_is_dropped_not_grown_without_bound() {
        use std::io::Write as _;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("live.sock");
        let mut bridge = Bridge::bind(&path).unwrap();
        // A layout as big as a real one (~5 KB): every re-subscription queues a copy for a client that does not read.
        bridge.set_fly_meta(Some(&format!("{{\"v\":1,\"pad\":\"{}\"}}", "x".repeat(5000))));
        let mut c = connect(&mut bridge, &path);
        let toggle: Vec<u8> = [[2u8, 0, 0, 0, 1, 1], [2, 0, 0, 0, 1, 0]].concat();
        let mut polls = 0;
        // 6 KB of toggles per round; with a 5 KB copy per subscription this passes MAX_PENDING (512 KiB) plus what the
        // socket buffers hold in a few hundred rounds at the very most.
        while bridge.clients() == 1 && polls < 2000 {
            let mut round = Vec::new();
            for _ in 0..4 {
                round.extend_from_slice(&toggle);
            }
            if c.write_all(&round).is_err() {
                break;
            }
            bridge.accept_pending();
            polls += 1;
            // Whatever it asked for, the queue never passed the cap while the client was still there.
            assert!(bridge.clients.iter().all(|k| k.pending.len() <= MAX_PENDING));
        }
        assert_eq!(bridge.clients(), 0, "dropped after {polls} polls");
        assert_eq!(bridge.dropped_clients(), 1);
        assert!(polls < 2000);
    }

    #[test]
    fn a_burst_of_small_subscriptions_between_two_polls_is_not_a_violation() {
        use std::io::Write as _;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("live.sock");
        let mut bridge = Bridge::bind(&path).unwrap();
        bridge.set_fly_meta(Some("{\"v\":1}"));
        let mut c = connect(&mut bridge, &path);
        // 40 messages of 6 bytes (240 bytes, well past the old 80-byte inbox limit), ending subscribed.
        let mut burst = Vec::new();
        for i in 0..40 {
            burst.extend_from_slice(&[2, 0, 0, 0, 1, u8::from(i % 2 == 1)]);
        }
        c.write_all(&burst).unwrap();
        settle(&mut bridge);
        assert_eq!(bridge.clients(), 1, "a legitimate burst keeps the client");
        assert!(bridge.fly_wanted());
        // A message split across two writes is put together.
        write_client(&mut c, client_kind::SUBSCRIBE, &[0]);
        settle(&mut bridge);
        assert!(!bridge.fly_wanted());
        c.write_all(&[2, 0, 0]).unwrap();
        settle(&mut bridge);
        c.write_all(&[0, 1, 1]).unwrap();
        settle(&mut bridge);
        assert!(bridge.fly_wanted(), "the two halves made one subscription");
    }

    #[test]
    fn a_second_bot_does_not_steal_a_live_socket_but_a_stale_one_is_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("live.sock");
        let first = Bridge::bind(&path).unwrap();
        let err = Bridge::bind(&path).err().expect("in use");
        assert_eq!(err.kind(), io::ErrorKind::AddrInUse);
        drop(first);
        // A stale socket file (process gone without cleanup): bind again works.
        let stale = UnixListener::bind(&path).unwrap();
        drop(stale); // the file stays, nobody listens
        assert!(Bridge::bind(&path).is_ok());
        // Something that is not a socket is never replaced.
        let file = dir.path().join("regular");
        std::fs::write(&file, b"x").unwrap();
        assert_eq!(Bridge::bind(&file).err().unwrap().kind(), io::ErrorKind::AlreadyExists);
    }

    /// Task 5.10: a chat line reaches every client as a `CHAT` message, is cut at the caps (on a character boundary),
    /// is not remembered for a client that connects later, and is not queued when nobody is connected.
    #[test]
    fn chat_lines_are_forwarded_cut_to_the_caps_and_never_kept() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("live.sock");
        let mut bridge = Bridge::bind(&path).unwrap();
        let line = |text: &str| ChatMessage {
            team: 1,
            cid: 4,
            name: "c4-deadbeef".into(),
            text: text.into(),
        };
        bridge.send_chat(&line("nobody hears this")); // no client: nothing queued, nothing remembered
        let mut early = connect(&mut bridge, &path);
        bridge.send_chat(&line("<img src=x onerror=alert(1)> \u{202e}hi"));
        let (k, p) = read_message(&mut early);
        assert_eq!(k, kind::CHAT);
        let v: serde_json::Value = serde_json::from_slice(&p).unwrap();
        assert_eq!((v["team"].as_i64(), v["cid"].as_i64()), (Some(1), Some(4)));
        assert_eq!(v["name"], "c4-deadbeef");
        assert_eq!(
            v["text"], "<img src=x onerror=alert(1)> \u{202e}hi",
            "the bridge forwards text as it is"
        );
        // Over the cap: cut, and never in the middle of a character (each "я" is 2 bytes; 511 bytes would split one).
        bridge.send_chat(&line(&"я".repeat(400)));
        let (_, p) = read_message(&mut early);
        let v: serde_json::Value = serde_json::from_slice(&p).unwrap();
        let text = v["text"].as_str().unwrap();
        assert!(text.len() <= MAX_CHAT_TEXT && text.chars().all(|c| c == 'я') && text.len() >= MAX_CHAT_TEXT - 1);
        // A late client is greeted with HELLO only: chat is not replayed by the bot.
        let late = UnixStream::connect(&path).unwrap();
        late.set_read_timeout(Some(std::time::Duration::from_millis(200)))
            .unwrap();
        settle(&mut bridge);
        let mut late = late;
        let (k, _) = read_message(&mut late);
        assert_eq!(k, kind::HELLO);
        let mut one = [0u8; 1];
        let e = std::io::Read::read(&mut late, &mut one).unwrap_err();
        assert!(
            matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut),
            "{e:?}"
        );
    }

    #[test]
    fn player_info_reaches_clients_as_one_json_message() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("live.sock");
        let mut bridge = Bridge::bind(&path).unwrap();
        let mut c = connect(&mut bridge, &path);
        bridge.send_player_info(&PlayerInfoMessage {
            list: vec![PlayerInfoEntry {
                id: 3,
                clan: String::new(),
                skin: "coala".into(),
                cc: true,
                cb: 0x00ff_8040,
                cf: 7,
                country: 276,
                score: -2,
                ping: 31,
            }],
        });
        let (k, p) = read_message(&mut c);
        assert_eq!(k, kind::PLAYERINFO);
        let v: serde_json::Value = serde_json::from_slice(&p).unwrap();
        assert_eq!(v["list"][0]["skin"], "coala");
        assert_eq!(v["list"][0]["cc"], true);
        assert_eq!(v["list"][0]["cb"], 0x00ff_8040);
        assert_eq!(v["list"][0]["score"], -2);
        // A client that connects later is greeted with the last one (the runner resends only on a difference).
        let mut late = connect(&mut bridge, &path);
        let (k, p) = read_message(&mut late);
        assert_eq!(k, kind::PLAYERINFO, "after HELLO");
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&p).unwrap()["list"][0]["skin"],
            "coala"
        );
    }

    #[test]
    fn the_socket_file_is_removed_when_the_bridge_is_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("live.sock");
        drop(Bridge::bind(&path).unwrap());
        assert!(!path.exists());
    }
}
