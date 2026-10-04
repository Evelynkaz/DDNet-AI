//! The live bot as a [`FrameSource`] (task 4.1, `docs/formats.md` §21): connects to the bot
//! process's Unix socket (`ddai_bot::bridge`), reads its messages — never writes — and turns them
//! into [`SourceEvent`]s for the hub, so the "Игра" page shows the real game instead of replays.
//!
//! **Read-only, with one exception (task 7.4).** The web control path is a separate socket; this one carries nothing
//! but the bot's output, and the only thing this source ever writes to it is a **subscription**
//! (`u32 LE len | u8 1 | u8 mask`, bit 0: the fly stream; bit 1, task 5.7: a browser is connected at all) so that the bot
//! builds the fly's frames only while a browser watches them and the offline demo can pause while nobody looks
//! (`docs/formats.md` §21.2, §27, §28). The `replay{...}` commands of the page are drained and ignored.
//!
//! **Link (task 5.7).** The source tells whoever reads it (the multiplexer that puts the demo behind the live bot,
//! [`super::mux`]) when the connection is *usable*: [`SourceEvent::Link`]`(true)` after a valid greeting, `false` when the
//! connection ends. A socket that connects but does not speak the protocol never counts as a bot.
//!
//! **Trust.** The socket is a `0600` file in a `0700` directory of the same user, but the bytes are
//! still parsed defensively: message length capped at [`MAX_MESSAGE`], unknown kinds skipped, a bad
//! frame / JSON reported as one [`SourceEvent::Error`] (never a panic), a wrong protocol version
//! disconnects. A map is resolved **only** from the configured maps directories, by a filename
//! built from the sanitised name and the claimed sha256, and the file's real sha256 must match
//! (`crate::live::map_resolve`, the same rule as the replay source) — nothing on the wire names a path.
//!
//! **Events.** The bot sends frames, not events; freeze / unfreeze / death / respawn / hook-grab are
//! derived from consecutive frames here (a character appearing is a respawn, disappearing a death).
//!
//! **Reconnect.** A missing socket or a dropped connection is retried every second; the page is told
//! once per outage.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::net::unix::OwnedReadHalf;
use tokio::sync::{mpsc, watch};

use super::map_resolve::{self, MapCache};
use super::source::{
    CharacterState, FrameSource, GameEvent, MapMeta, PlayerMeta, ReplayControl, SourceEvent, WorldFrame,
};

/// Largest message the source accepts (`ddai_bot::bridge::MAX_MESSAGE`).
pub const MAX_MESSAGE: usize = 1 << 20;
/// Protocol version this reader speaks.
pub const VERSION: u8 = 1;

mod kind {
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

/// What this source says to the bot: a subscription (`ddai_bot::bridge::client_kind::SUBSCRIBE`).
const SUBSCRIBE: u8 = 1;
const SUBSCRIBE_FLY: u8 = 1;
const SUBSCRIBE_VIEW: u8 = 2;
/// A write of a few bytes to the bot must not hang the source.
const WRITE_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Debug, Deserialize)]
struct MapMsg {
    name: String,
    sha256: String,
    w: u32,
    h: u32,
}

#[derive(Debug, Deserialize)]
struct PlayerEntry {
    id: i32,
    name: String,
    #[serde(default)]
    team: i32,
}

#[derive(Debug, Deserialize)]
struct PlayersMsg {
    #[serde(default)]
    list: Vec<PlayerEntry>,
}

/// One `PLAYERINFO` entry (task 5.10): how a player looks and their numbers. Every field but the id may be missing.
#[derive(Debug, Clone, Default, Deserialize)]
struct InfoEntry {
    id: i32,
    #[serde(default)]
    clan: String,
    #[serde(default)]
    skin: String,
    #[serde(default)]
    cc: bool,
    #[serde(default)]
    cb: i32,
    #[serde(default)]
    cf: i32,
    #[serde(default)]
    country: i32,
    #[serde(default)]
    score: i32,
    #[serde(default)]
    ping: i32,
}

#[derive(Debug, Deserialize)]
struct InfoMsg {
    #[serde(default)]
    list: Vec<InfoEntry>,
}

/// `CHAT`: one line of the server's chat (task 5.10), display only.
#[derive(Debug, Deserialize)]
struct ChatMsg {
    #[serde(default)]
    team: i32,
    #[serde(default = "no_sender")]
    cid: i32,
    #[serde(default)]
    name: String,
    #[serde(default)]
    text: String,
}

fn no_sender() -> i32 {
    -1
}

fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// A roster entry with the look and numbers of `info` applied (the skin is kept to a plausible length; the page validates the name).
fn apply_info(meta: &mut PlayerMeta, info: &InfoEntry) {
    meta.clan = info.clan.chars().take(32).collect();
    meta.skin = info.skin.chars().take(32).collect();
    meta.custom_color = info.cc;
    meta.color_body = info.cb;
    meta.color_feet = info.cf;
    meta.country = info.country;
    meta.score = info.score;
    meta.ping = info.ping;
}

/// The bot-socket frame source.
pub struct BotSource {
    socket: PathBuf,
    search_dirs: Vec<PathBuf>,
    map_cache: Arc<MapCache>,
    /// Whether a browser watches the fly (set by the hub).
    fly_demand: Option<watch::Receiver<bool>>,
    /// Whether a browser is connected at all (set by the hub).
    view_demand: Option<watch::Receiver<bool>>,
}

impl BotSource {
    pub fn new(socket: PathBuf, search_dirs: Vec<PathBuf>, map_cache: Arc<MapCache>) -> Self {
        BotSource {
            socket,
            search_dirs,
            map_cache,
            fly_demand: None,
            view_demand: None,
        }
    }
}

/// Per-connection bookkeeping: the greeting and the fly stream.
#[derive(Default)]
struct ConnState {
    /// A valid greeting was seen and [`SourceEvent::Link`]`(true)` sent (so `false` is sent when the connection ends).
    linked: bool,
    /// A layout was passed on (so the page is told when it ends).
    meta_told: bool,
    /// A bad frame or layout was reported (one report per connection, not one per frame).
    bad_told: bool,
    /// Task 5.10: the roster as last built from `PLAYERS`, and the newest `PLAYERINFO` per id; the two arrive separately
    /// and a roster is only ever sent whole (a `PLAYERINFO` for an id that is not on the roster is remembered, not shown).
    roster: Vec<PlayerMeta>,
    info: HashMap<u8, InfoEntry>,
}

/// One message of the bot: its body (kind byte and payload), the read half handed back for the next one.
/// A future of its own, polled across `select!` rounds, so a message that is half read is never lost to another branch.
async fn read_message(mut rd: OwnedReadHalf) -> (OwnedReadHalf, std::io::Result<Vec<u8>>) {
    let mut len_buf = [0u8; 4];
    let result = async {
        rd.read_exact(&mut len_buf).await?;
        let len = u32::from_le_bytes(len_buf) as usize;
        if len == 0 || len > MAX_MESSAGE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("message length {len} is out of range"),
            ));
        }
        let mut body = vec![0u8; len];
        rd.read_exact(&mut body).await?;
        Ok(body)
    }
    .await;
    (rd, result)
}

impl FrameSource for BotSource {
    fn attach_fly_demand(&mut self, demand: watch::Receiver<bool>) {
        self.fly_demand = Some(demand);
    }

    fn attach_view_demand(&mut self, demand: watch::Receiver<bool>) {
        self.view_demand = Some(demand);
    }

    fn spawn(
        self: Box<Self>,
        events_tx: mpsc::Sender<SourceEvent>,
        control_rx: mpsc::Receiver<ReplayControl>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move { self.run(events_tx, control_rx).await })
    }
}

fn parse_sha256(hex: &str) -> Option<[u8; 32]> {
    if hex.len() != 64 || !hex.is_ascii() {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

/// Why one connection ended.
enum End {
    /// The hub went away: stop for good.
    HubClosed,
    /// Disconnected or bad stream: reconnect.
    Lost(String),
}

impl BotSource {
    async fn run(self: Box<Self>, events_tx: mpsc::Sender<SourceEvent>, mut control_rx: mpsc::Receiver<ReplayControl>) {
        let mut told_outage = false;
        loop {
            match UnixStream::connect(&self.socket).await {
                Ok(stream) => {
                    told_outage = false;
                    match self.session(stream, &events_tx, &mut control_rx).await {
                        End::HubClosed => return,
                        End::Lost(why) => {
                            if events_tx
                                .send(SourceEvent::Error(format!("the bot connection ended: {why}")))
                                .await
                                .is_err()
                            {
                                return;
                            }
                        }
                    }
                }
                Err(_) => {
                    if !told_outage {
                        told_outage = true;
                        if events_tx
                            .send(SourceEvent::Error(
                                "the bot is not running (no live socket)".to_string(),
                            ))
                            .await
                            .is_err()
                        {
                            return;
                        }
                    }
                }
            }
            // Wait to retry, draining the page's replay commands (there is nothing to control).
            let sleep = tokio::time::sleep(Duration::from_secs(1));
            tokio::pin!(sleep);
            loop {
                tokio::select! {
                    _ = &mut sleep => break,
                    cmd = control_rx.recv() => {
                        if cmd.is_none() {
                            return;
                        }
                    }
                }
            }
        }
    }

    async fn session(
        &self,
        stream: UnixStream,
        events_tx: &mpsc::Sender<SourceEvent>,
        control_rx: &mut mpsc::Receiver<ReplayControl>,
    ) -> End {
        let mut fly = ConnState::default();
        let end = self.session_inner(stream, events_tx, control_rx, &mut fly).await;
        // The stream ended with the connection: a page showing it is told there is none.
        if fly.meta_told && events_tx.send(SourceEvent::FlyMeta(None)).await.is_err() {
            return End::HubClosed;
        }
        if fly.linked && events_tx.send(SourceEvent::Link(false)).await.is_err() {
            return End::HubClosed;
        }
        end
    }

    async fn session_inner(
        &self,
        stream: UnixStream,
        events_tx: &mpsc::Sender<SourceEvent>,
        control_rx: &mut mpsc::Receiver<ReplayControl>,
        fly: &mut ConnState,
    ) -> End {
        let mut prev: HashMap<u8, CharacterState> = HashMap::new();
        let (rd, mut wr) = stream.into_split();
        let mut reader = Box::pin(read_message(rd));
        let mut fly_demand = self.fly_demand.clone();
        let mut view_demand = self.view_demand.clone();
        let mut want = Want::default();
        if let Some(d) = fly_demand.as_mut() {
            want.fly = *d.borrow_and_update();
        }
        if let Some(d) = view_demand.as_mut() {
            want.view = *d.borrow_and_update();
        }
        // Whoever already watches (a page open before the bot started, or a reconnect) is announced at once.
        if want.any()
            && let Err(e) = write_subscription(&mut wr, want).await
        {
            return End::Lost(e);
        }
        loop {
            // One message: the bot's next, raced against the page's commands and the demand for the fly and the view.
            tokio::select! {
                (rd, result) = &mut reader => {
                    let body = match result {
                        Ok(b) => b,
                        Err(e) => return End::Lost(e.to_string()),
                    };
                    reader = Box::pin(read_message(rd));
                    let (kind, payload) = (body[0], &body[1..]);
                    match self.handle(kind, payload, &mut prev, events_tx, fly).await {
                        Ok(()) => {}
                        Err(End::HubClosed) => return End::HubClosed,
                        Err(End::Lost(why)) => return End::Lost(why),
                    }
                }
                cmd = control_rx.recv() => {
                    if cmd.is_none() {
                        return End::HubClosed;
                    }
                    // read-only: commands are ignored
                }
                changed = demand_changed(&mut fly_demand) => {
                    if changed {
                        want.fly = *fly_demand.as_mut().expect("it just changed").borrow_and_update();
                        if let Err(e) = write_subscription(&mut wr, want).await {
                            return End::Lost(e);
                        }
                    } else {
                        fly_demand = None; // the hub is going away; nothing more will change
                    }
                }
                changed = demand_changed(&mut view_demand) => {
                    if changed {
                        want.view = *view_demand.as_mut().expect("it just changed").borrow_and_update();
                        if let Err(e) = write_subscription(&mut wr, want).await {
                            return End::Lost(e);
                        }
                    } else {
                        view_demand = None;
                    }
                }
            }
        }
    }

    async fn handle(
        &self,
        kind: u8,
        payload: &[u8],
        prev: &mut HashMap<u8, CharacterState>,
        events_tx: &mpsc::Sender<SourceEvent>,
        fly: &mut ConnState,
    ) -> Result<(), End> {
        let send = |e: SourceEvent| async move { events_tx.send(e).await.map_err(|_| End::HubClosed) };
        match kind {
            kind::HELLO => {
                if payload.len() < 5 || &payload[..4] != b"DDBL" || payload[4] != VERSION {
                    return Err(End::Lost(format!(
                        "unsupported bot protocol (expected DDBL v{VERSION})"
                    )));
                }
                if !fly.linked {
                    fly.linked = true;
                    send(SourceEvent::Link(true)).await?;
                }
            }
            kind::MAP => match serde_json::from_slice::<MapMsg>(payload) {
                Ok(m) => {
                    prev.clear();
                    fly.roster.clear();
                    fly.info.clear();
                    let Some(sha) = parse_sha256(&m.sha256) else {
                        return send(SourceEvent::Error("the bot sent a malformed map hash".to_string())).await;
                    };
                    // The bot's map cache names files `<name>_<sha256>.map`; only a filename is
                    // built (and sanitised by `resolve_by_sha256`), never a path.
                    let hint = format!("{}_{}.map", m.name, m.sha256);
                    let dirs = self.search_dirs.clone();
                    let cache = Arc::clone(&self.map_cache);
                    let resolved = tokio::task::spawn_blocking(move || match cache.get(&sha) {
                        Some(_) => Ok(()),
                        None => map_resolve::resolve_by_sha256(&dirs, &hint, sha).map(|(path, scene)| {
                            cache.insert(sha, scene);
                            cache.insert_path(sha, path);
                        }),
                    })
                    .await;
                    match resolved {
                        Ok(Ok(())) => {
                            send(SourceEvent::MapChanged(MapMeta {
                                sha256: sha,
                                name: m.name,
                                width: m.w,
                                height: m.h,
                            }))
                            .await?;
                        }
                        Ok(Err(e)) => send(SourceEvent::Error(format!("could not load the bot's map: {e}"))).await?,
                        Err(_) => send(SourceEvent::Error("the map loader failed".to_string())).await?,
                    }
                }
                Err(e) => send(SourceEvent::Error(format!("bad MAP message: {e}"))).await?,
            },
            kind::PLAYERS => match serde_json::from_slice::<PlayersMsg>(payload) {
                Ok(p) => {
                    let list: Vec<PlayerMeta> = p
                        .list
                        .into_iter()
                        .filter_map(|e| {
                            let id = u8::try_from(e.id).ok()?;
                            let mut meta = PlayerMeta {
                                id,
                                name: e.name.chars().take(64).collect(),
                                team: u8::try_from(e.team.clamp(0, 255)).unwrap_or(0),
                                ..PlayerMeta::default()
                            };
                            if let Some(info) = fly.info.get(&id) {
                                apply_info(&mut meta, info);
                            }
                            Some(meta)
                        })
                        .collect();
                    fly.roster = list.clone();
                    send(SourceEvent::Players(list)).await?;
                }
                Err(e) => send(SourceEvent::Error(format!("bad PLAYERS message: {e}"))).await?,
            },
            kind::PLAYERINFO => match serde_json::from_slice::<InfoMsg>(payload) {
                Ok(m) => {
                    let mut changed = false;
                    for e in m.list {
                        let Ok(id) = u8::try_from(e.id) else { continue };
                        if let Some(meta) = fly.roster.iter_mut().find(|r| r.id == id) {
                            let before = meta.clone();
                            apply_info(meta, &e);
                            changed |= *meta != before;
                        }
                        fly.info.insert(id, e);
                    }
                    if changed {
                        send(SourceEvent::Players(fly.roster.clone())).await?;
                    }
                }
                Err(e) => send(SourceEvent::Error(format!("bad PLAYERINFO message: {e}"))).await?,
            },
            kind::CHAT => {
                // A bad chat line is dropped without a word: reporting it would put its bytes into an error message.
                if let Ok(m) = serde_json::from_slice::<ChatMsg>(payload)
                    // The text is hostile input: cleaned and capped here (`super::chat`), never stored beyond the in-memory ring.
                    && let Some(line) = super::chat::ChatLine::from_bridge(m.team, m.cid, &m.name, &m.text, unix_ms())
                {
                    send(SourceEvent::Chat(line)).await?;
                }
            }
            kind::FRAME => match super::frame::decode(payload) {
                Ok(frame) => {
                    let events = diff_events(prev, &frame);
                    prev.clear();
                    prev.extend(frame.characters.iter().map(|c| (c.id, *c)));
                    let tick = frame.tick;
                    send(SourceEvent::Frame(frame)).await?;
                    if !events.is_empty() {
                        send(SourceEvent::Events { tick, events }).await?;
                    }
                }
                Err(e) => send(SourceEvent::Error(format!("bad FRAME: {e}"))).await?,
            },
            kind::STATUS => {
                // Validated as JSON here so the hub/WS layers only ever see well-formed text.
                if serde_json::from_slice::<serde_json::Value>(payload).is_ok()
                    && let Ok(text) = String::from_utf8(payload.to_vec())
                {
                    send(SourceEvent::BotStatus(text)).await?;
                }
            }
            kind::FLYMETA => match super::fly::validate_meta(payload) {
                Ok(meta) => {
                    fly.meta_told = meta.is_some();
                    send(SourceEvent::FlyMeta(meta)).await?;
                }
                Err(e) => report_bad_fly(fly, &send, e).await?,
            },
            kind::FLY => match super::fly::validate_frame(payload) {
                Ok(()) => send(SourceEvent::FlyFrame(payload.to_vec())).await?,
                Err(e) => report_bad_fly(fly, &send, e).await?,
            },
            _ => {} // forward-compatible: an unknown kind is skipped
        }
        Ok(())
    }
}

/// Tells the page once per connection that the bot's fly stream is malformed (the frame itself is dropped).
async fn report_bad_fly<F, Fut>(fly: &mut ConnState, send: &F, why: String) -> Result<(), End>
where
    F: Fn(SourceEvent) -> Fut,
    Fut: std::future::Future<Output = Result<(), End>>,
{
    if fly.bad_told {
        return Ok(());
    }
    fly.bad_told = true;
    send(SourceEvent::Error(format!("the bot's fly stream is malformed: {why}"))).await
}

/// What the page wants from the bot right now (the bits of the subscription mask).
#[derive(Clone, Copy, Default)]
struct Want {
    fly: bool,
    view: bool,
}

impl Want {
    fn any(self) -> bool {
        self.fly || self.view
    }

    fn mask(self) -> u8 {
        (if self.fly { SUBSCRIBE_FLY } else { 0 }) | (if self.view { SUBSCRIBE_VIEW } else { 0 })
    }
}

/// Resolves `true` when the demand changed, `false` when its sender is gone; never resolves without a demand.
pub(super) async fn demand_changed(demand: &mut Option<watch::Receiver<bool>>) -> bool {
    match demand.as_mut() {
        Some(d) => d.changed().await.is_ok(),
        None => std::future::pending().await,
    }
}

/// `u32 LE len | u8 SUBSCRIBE | u8 mask`.
async fn write_subscription(wr: &mut tokio::net::unix::OwnedWriteHalf, want: Want) -> Result<(), String> {
    let msg = [2, 0, 0, 0, SUBSCRIBE, want.mask()];
    match tokio::time::timeout(WRITE_TIMEOUT, wr.write_all(&msg)).await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => Err("the bot did not take a subscription in time".to_string()),
    }
}

/// Freeze / unfreeze / death / respawn / hook-grab between two consecutive frames.
pub fn diff_events(prev: &HashMap<u8, CharacterState>, now: &WorldFrame) -> Vec<GameEvent> {
    let mut out = Vec::new();
    if prev.is_empty() {
        return out;
    }
    for c in &now.characters {
        match prev.get(&c.id) {
            None => out.push(GameEvent::Respawn { id: c.id }),
            Some(p) => {
                let (was, is) = (p.frozen || p.deep_frozen, c.frozen || c.deep_frozen);
                if !was && is {
                    out.push(GameEvent::Freeze { id: c.id });
                } else if was && !is {
                    out.push(GameEvent::Unfreeze { id: c.id });
                }
                if c.hooked_id.is_some() && c.hooked_id != p.hooked_id {
                    out.push(GameEvent::HookGrab {
                        id: c.id,
                        target: c.hooked_id,
                    });
                }
            }
        }
    }
    for id in prev.keys() {
        if !now.characters.iter().any(|c| c.id == *id) {
            out.push(GameEvent::Death { id: *id });
        }
    }
    out.sort_by_key(|e| match e {
        GameEvent::Freeze { id }
        | GameEvent::Unfreeze { id }
        | GameEvent::Death { id }
        | GameEvent::Respawn { id }
        | GameEvent::HookGrab { id, .. } => *id,
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::os::unix::net::UnixListener;

    fn char_state(id: u8, frozen: bool, hooked: Option<u8>) -> CharacterState {
        CharacterState {
            id,
            alive: true,
            x: 100,
            y: 200,
            aim_x: 1,
            aim_y: 2,
            hook_state: 0,
            hook_x: 0,
            hook_y: 0,
            hooked_id: hooked,
            weapon: 0,
            team: 0,
            frozen,
            deep_frozen: false,
            live_frozen: false,
        }
    }

    fn frame(tick: u32, chars: Vec<CharacterState>) -> WorldFrame {
        WorldFrame {
            tick,
            characters: chars,
        }
    }

    #[test]
    fn events_are_derived_from_consecutive_frames() {
        let first = frame(1, vec![char_state(0, false, None), char_state(1, false, None)]);
        let prev: HashMap<u8, CharacterState> = first.characters.iter().map(|c| (c.id, *c)).collect();
        let second = frame(2, vec![char_state(0, true, None), char_state(2, false, None)]);
        let ev = diff_events(&prev, &second);
        assert_eq!(
            ev,
            vec![
                GameEvent::Freeze { id: 0 },
                GameEvent::Death { id: 1 },
                GameEvent::Respawn { id: 2 }
            ]
        );
        let third_prev: HashMap<u8, CharacterState> = second.characters.iter().map(|c| (c.id, *c)).collect();
        let third = frame(3, vec![char_state(0, false, Some(2)), char_state(2, false, None)]);
        assert_eq!(
            diff_events(&third_prev, &third),
            vec![
                GameEvent::Unfreeze { id: 0 },
                GameEvent::HookGrab { id: 0, target: Some(2) }
            ]
        );
        assert!(
            diff_events(&HashMap::new(), &second).is_empty(),
            "the first frame has no history"
        );
    }

    #[test]
    fn hashes_parse_strictly() {
        assert_eq!(parse_sha256(&"ab".repeat(32)), Some([0xab; 32]));
        assert_eq!(parse_sha256("zz"), None);
        assert_eq!(
            parse_sha256(&"é".repeat(32)),
            None,
            "non-ascii never slices a char boundary"
        );
        assert_eq!(parse_sha256(&"a".repeat(63)), None);
    }

    fn message(kind: u8, payload: &[u8]) -> Vec<u8> {
        let mut out = ((payload.len() + 1) as u32).to_le_bytes().to_vec();
        out.push(kind);
        out.extend_from_slice(payload);
        out
    }

    /// The bot's wire format, written by hand here (independent of `ddai-bot`): a scripted "bot" on
    /// a real Unix socket feeds the source; the source must produce the hub events.
    #[tokio::test]
    async fn a_scripted_bot_socket_produces_map_players_frames_events_and_status() {
        use crate::live::frame;
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("live.sock");
        let maps = dir.path().join("maps");
        std::fs::create_dir_all(&maps).unwrap();
        // A real, loadable map file named the way the bot's cache names it.
        let map_bytes = {
            use ddai_map::testutil::{MapWriter, TILESLAYERFLAG_GAME, TileLayerSpec, TilemapShape, game_layer_data};
            let mut w = MapWriter::new(4);
            w.add_version_item(1);
            w.add_tile_layer(&TileLayerSpec {
                shape: TilemapShape::Full,
                item_version: 3,
                width: 2,
                height: 2,
                flags: TILESLAYERFLAG_GAME,
                data: &game_layer_data(2, 2),
            });
            w.add_single_group_with_all_layers();
            w.finish()
        };
        use sha2::{Digest, Sha256};
        let sha: [u8; 32] = Sha256::digest(&map_bytes).into();
        let hex: String = sha.iter().map(|b| format!("{b:02x}")).collect();
        std::fs::write(maps.join(format!("Test Map_{hex}.map")), &map_bytes).unwrap();

        let listener = UnixListener::bind(&sock).unwrap();
        let hex2 = hex.clone();
        let server = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            s.write_all(&message(kind::HELLO, b"DDBL\x01")).unwrap();
            s.write_all(&message(
                kind::MAP,
                format!(r#"{{"name":"Test Map","sha256":"{hex2}","w":2,"h":2}}"#).as_bytes(),
            ))
            .unwrap();
            s.write_all(&message(
                kind::PLAYERS,
                br#"{"own":0,"list":[{"id":0,"name":"c0-aaaaaaaa","team":0},{"id":1,"name":"c1-bbbbbbbb","team":0}]}"#,
            ))
            .unwrap();
            let f1 = frame::encode(&frame(10, vec![char_state(0, false, None), char_state(1, false, None)]));
            let f2 = frame::encode(&frame(12, vec![char_state(0, false, None), char_state(1, true, None)]));
            s.write_all(&message(kind::FRAME, &f1)).unwrap();
            s.write_all(&message(kind::FRAME, &f2)).unwrap();
            s.write_all(&message(kind::STATUS, br#"{"target":1,"mode":"fight"}"#))
                .unwrap();
            s.write_all(&message(99, b"unknown kinds are skipped")).unwrap();
            std::thread::sleep(Duration::from_millis(300));
        });

        let cache = Arc::new(MapCache::new());
        let source = Box::new(BotSource::new(sock, vec![maps], Arc::clone(&cache)));
        let (tx, mut rx) = mpsc::channel(64);
        let (_ctl_tx, ctl_rx) = mpsc::channel(4);
        let handle = source.spawn(tx, ctl_rx);

        let mut seen_map = false;
        let mut players = 0;
        let mut frames = Vec::new();
        let mut events = Vec::new();
        let mut status = None;
        while let Ok(Some(ev)) = tokio::time::timeout(Duration::from_secs(3), rx.recv()).await {
            match ev {
                SourceEvent::MapChanged(m) => {
                    assert_eq!((m.name.as_str(), m.width, m.height, m.sha256), ("Test Map", 2, 2, sha));
                    seen_map = true;
                }
                SourceEvent::Players(p) => players = p.len(),
                SourceEvent::Frame(f) => frames.push(f.tick),
                SourceEvent::Events { tick, events: e } => events.push((tick, e)),
                SourceEvent::BotStatus(s) => status = Some(s),
                SourceEvent::Error(e) => {
                    if !e.contains("connection ended") {
                        panic!("unexpected error: {e}");
                    }
                    break;
                }
                _ => {}
            }
        }
        handle.abort();
        server.join().unwrap();
        assert!(seen_map, "the map was resolved from the maps directory");
        assert!(
            cache.get(&sha).is_some(),
            "and its scene cached for GET /api/map/<sha256>"
        );
        assert_eq!(players, 2);
        assert_eq!(frames, vec![10, 12]);
        assert_eq!(events, vec![(12, vec![GameEvent::Freeze { id: 1 }])]);
        assert!(status.unwrap().contains("fight"));
    }

    /// Task 5.7: the link is up only after a valid greeting and down when the connection ends; a socket that connects and
    /// says something else never links, whatever it goes on to send.
    #[tokio::test]
    async fn the_link_follows_a_valid_greeting_and_a_bad_one_never_links() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("live.sock");
        let listener = UnixListener::bind(&sock).unwrap();
        listener.set_nonblocking(false).unwrap();
        let server = std::thread::spawn(move || {
            // First connection: not the protocol (wrong magic), then plausible messages.
            let (mut s, _) = listener.accept().unwrap();
            s.write_all(&message(kind::HELLO, b"NOPE\x01")).unwrap();
            s.write_all(&message(kind::STATUS, br#"{"mode":"fight"}"#)).unwrap();
            std::thread::sleep(Duration::from_millis(100));
            drop(s);
            // Second connection: a valid greeting, a status, then it goes away.
            let (mut s, _) = listener.accept().unwrap();
            s.write_all(&message(kind::HELLO, b"DDBL\x01")).unwrap();
            s.write_all(&message(kind::HELLO, b"DDBL\x01")).unwrap(); // a second greeting is not a second link
            s.write_all(&message(kind::STATUS, br#"{"mode":"fight"}"#)).unwrap();
            std::thread::sleep(Duration::from_millis(150));
        });
        let source = Box::new(BotSource::new(sock, vec![], Arc::new(MapCache::new())));
        let (tx, mut rx) = mpsc::channel(64);
        let (_ctl, ctl_rx) = mpsc::channel(4);
        let handle = source.spawn(tx, ctl_rx);
        let mut seen: Vec<String> = Vec::new();
        while let Ok(Some(ev)) = tokio::time::timeout(Duration::from_secs(5), rx.recv()).await {
            let tag = match &ev {
                SourceEvent::Link(up) => format!("link {up}"),
                SourceEvent::BotStatus(_) => "status".to_string(),
                SourceEvent::Error(e) if e.contains("unsupported bot protocol") => "bad greeting".to_string(),
                SourceEvent::Error(e) if e.contains("connection ended") => "ended".to_string(),
                _ => continue,
            };
            seen.push(tag);
            if seen.last().is_some_and(|t| t == "ended") && seen.iter().any(|t| t == "link false") {
                break;
            }
        }
        handle.abort();
        server.join().unwrap();
        assert_eq!(
            seen,
            ["bad greeting", "link true", "status", "link false", "ended"],
            "the first connection never linked; the second linked once, and unlinked before its end was reported"
        );
    }

    /// Task 5.7: the subscription carries the fly bit and the view bit, and is re-sent when either changes.
    // Multi-threaded: the test blocks on a std channel while the source runs.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_subscription_carries_the_fly_and_view_bits() {
        use std::io::Read;
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("demo.sock");
        let listener = UnixListener::bind(&sock).unwrap();
        let (got_tx, got_rx) = std::sync::mpsc::channel::<u8>();
        let server = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            s.write_all(&message(kind::HELLO, b"DDBL\x01")).unwrap();
            s.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
            let mut buf = [0u8; 6];
            while s.read_exact(&mut buf).is_ok() {
                assert_eq!(buf[..5], [2, 0, 0, 0, 1], "a subscription message");
                got_tx.send(buf[5]).unwrap();
            }
        });
        let mut source = Box::new(BotSource::new(sock, vec![], Arc::new(MapCache::new())));
        let (fly_tx, fly_rx) = watch::channel(false);
        let (view_tx, view_rx) = watch::channel(true); // a browser was there before the bot
        source.attach_fly_demand(fly_rx);
        source.attach_view_demand(view_rx);
        let (tx, mut rx) = mpsc::channel(64);
        let (_ctl, ctl_rx) = mpsc::channel(4);
        let handle = source.spawn(tx, ctl_rx);
        let next = |what: &str| {
            got_rx
                .recv_timeout(Duration::from_secs(3))
                .unwrap_or_else(|e| panic!("no subscription for {what}: {e}"))
        };
        assert_eq!(next("connect"), SUBSCRIBE_VIEW, "announced at once on connect");
        fly_tx.send(true).unwrap();
        assert_eq!(next("fly on"), SUBSCRIBE_VIEW | SUBSCRIBE_FLY);
        view_tx.send(false).unwrap();
        assert_eq!(next("view off"), SUBSCRIBE_FLY);
        fly_tx.send(false).unwrap();
        assert_eq!(next("fly off"), 0, "nobody: unsubscribed");
        while rx.try_recv().is_ok() {}
        handle.abort();
        drop(server); // detached: it ends when the connection closes
    }

    #[tokio::test]
    async fn a_missing_socket_is_reported_once_and_retried_and_a_hostile_length_disconnects() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("none.sock");
        let source = Box::new(BotSource::new(sock.clone(), vec![], Arc::new(MapCache::new())));
        let (tx, mut rx) = mpsc::channel(8);
        let (_ctl, ctl_rx) = mpsc::channel(4);
        let handle = source.spawn(tx, ctl_rx);
        let first = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(first, SourceEvent::Error(e) if e.contains("not running")));
        // A server appears and sends a length beyond the cap: the source drops the connection.
        let listener = UnixListener::bind(&sock).unwrap();
        let server = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            s.write_all(&(MAX_MESSAGE as u32 + 1).to_le_bytes()).unwrap();
            std::thread::sleep(Duration::from_millis(200));
        });
        let next = tokio::time::timeout(Duration::from_secs(4), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(
            matches!(&next, SourceEvent::Error(e) if e.contains("out of range")),
            "{next:?}"
        );
        handle.abort();
        server.join().unwrap();
    }
}
