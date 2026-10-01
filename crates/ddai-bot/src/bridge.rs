//! The read-only live-state feed to the web unit (`ddai-web`, D-036: the bot and the web server are
//! separate units) — the "bot live bridge v1", `docs/formats.md` §21.
//!
//! The bot **listens** on a Unix socket (default `~/aiddnet/data/bot/live.sock`, mode `0600`, the
//! directory `0700`); the web unit connects and only ever reads. The bot never reads from the
//! socket: there is no control path (web control is a later task), and a client that writes is
//! simply ignored. Messages are length-prefixed: `u32 LE len | u8 kind | payload` where `len` counts
//! the kind byte and the payload (at most [`MAX_MESSAGE`]):
//!
//! | kind | name | payload |
//! |---|---|---|
//! | 1 | `HELLO` | `"DDBL"`, `u8` version (1) |
//! | 2 | `MAP` | JSON `{"name","sha256","w","h"}` (`sha256` hex of the `.map` file) |
//! | 3 | `PLAYERS` | JSON `{"own":id,"list":[{"id","name","team"}]}` |
//! | 4 | `FRAME` | a `DWLF` v1 live frame, byte for byte the web's binary `live` message (`docs/formats.md` §15.3) |
//! | 5 | `STATUS` | JSON, at most ~5 Hz: target, mode, brain, counters, latency percentiles, brain telemetry |
//!
//! **Names.** `PLAYERS.name` is the salted-hash tag (`c12-9f3a01bc`) unless the bot was started with
//! `--web-names`; real nicknames never leave the process otherwise (D-040, `CLAUDE.md`).
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
}

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
}

struct Client {
    stream: UnixStream,
    pending: Vec<u8>,
}

/// The publisher.
pub struct Bridge {
    listener: UnixListener,
    path: PathBuf,
    clients: Vec<Client>,
    map_msg: Option<Vec<u8>>,
    players_msg: Option<Vec<u8>>,
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
    /// Binds `path`, replacing a stale socket file (but refusing to touch anything that is not a
    /// socket), with the directory `0700` and the socket `0600`.
    pub fn bind(path: &Path) -> io::Result<Bridge> {
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
        let listener = UnixListener::bind(path)?;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
        listener.set_nonblocking(true)?;
        Ok(Bridge {
            listener,
            path: path.to_path_buf(),
            clients: Vec::new(),
            map_msg: None,
            players_msg: None,
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
                    if flush(&mut client) {
                        self.clients.push(client);
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(_) => break,
            }
        }
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
        self.broadcast(&msg);
    }

    /// The roster changed.
    pub fn send_players(&mut self, p: &PlayersMessage) {
        let Ok(json) = serde_json::to_vec(p) else { return };
        let msg = message(kind::PLAYERS, &json);
        self.players_msg = Some(msg.clone());
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
        let msg = message(kind::STATUS, &json);
        self.broadcast(&msg);
    }
}

const MAGIC_HELLO: &[u8; 4] = b"DDBL";

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
        });
        let (k, p) = read_message(&mut client);
        assert_eq!(k, kind::STATUS);
        let v: serde_json::Value = serde_json::from_slice(&p).unwrap();
        assert_eq!(v["target"], 3);
        assert_eq!(v["telemetry"]["x"], 1);
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

    #[test]
    fn the_socket_file_is_removed_when_the_bridge_is_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("live.sock");
        drop(Bridge::bind(&path).unwrap());
        assert!(!path.exists());
    }
}
