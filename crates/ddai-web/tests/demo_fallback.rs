//! The offline demo behind the live bot (task 5.7, D-075, `docs/formats.md` §28) against a real server and two scripted FAKE
//! bridges on real Unix sockets (the wire format is written here by hand, independent of `ddai-bot`): the live bot has
//! priority, the demo fills in while it is absent and gives way the moment it is up, the page is told which is on show,
//! commands never reach anything while only the demo is up, the demo is asked for viewers only while a browser is open and
//! the bot is not up, and what the demo sends is validated like what the bot sends.

// Task 5.5a: talks to the bot over a Unix-domain socket, which Windows has no counterpart of yet (ddai_os::ipc, task 5.5b).
#![cfg(unix)]

mod support;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ddai_web::live::frame;
use ddai_web::live::source::{CharacterState, WorldFrame};
use futures_util::{SinkExt, StreamExt};
use sha2::{Digest, Sha256};
use support::{Req, TestServer, send};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::{ClientRequestBuilder, Message};

type Ws = tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

const LIVE_TICK0: u32 = 1_000_000;
const DEMO_TICK0: u32 = 1;
const META: &str = r#"{"v":1,"role":"fly","name":"fly-test","bundle":{"name":"run/final","sha256":"abcd"},"rays":4,"bins":2,"groups":[],"dn":[],"channels":[],"scalars":[]}"#;

fn msg(kind: u8, payload: &[u8]) -> Vec<u8> {
    let mut m = ((payload.len() + 1) as u32).to_le_bytes().to_vec();
    m.push(kind);
    m.extend_from_slice(payload);
    m
}

/// A `.map` file of `size` x `size` tiles, written under the name the bridge's `MAP` message gives: `<name>_<sha256>.map`.
fn write_map(dir: &Path, name: &str, size: i32) -> String {
    use ddai_map::testutil::{MapWriter, TILESLAYERFLAG_GAME, TileLayerSpec, TilemapShape, game_layer_data};
    let mut w = MapWriter::new(4);
    w.add_version_item(1);
    w.add_tile_layer(&TileLayerSpec {
        shape: TilemapShape::Full,
        item_version: 3,
        width: size,
        height: size,
        flags: TILESLAYERFLAG_GAME,
        data: &game_layer_data(size, size),
    });
    w.add_single_group_with_all_layers();
    let bytes = w.finish();
    let sha: [u8; 32] = Sha256::digest(&bytes).into();
    let hex: String = sha.iter().map(|b| format!("{b:02x}")).collect();
    std::fs::write(dir.join(format!("{name}_{hex}.map")), &bytes).unwrap();
    hex
}

/// A `DFLY` frame of the documented layout (2 groups, 3 DN, 4 rays x 2 bins x 7 channels, 3 scalars).
fn fly_frame(seq: u32) -> Vec<u8> {
    let mut f = vec![0u8; 44 + 2 + 3 + 56 + 3];
    f[0..4].copy_from_slice(b"DFLY");
    f[4] = 1;
    f[12..16].copy_from_slice(&seq.to_le_bytes());
    f[34..36].copy_from_slice(&2u16.to_le_bytes());
    f[36..38].copy_from_slice(&3u16.to_le_bytes());
    f[38..40].copy_from_slice(&4u16.to_le_bytes());
    f[40] = 2;
    f[41] = 7;
    f[42] = 3;
    f
}

fn world_frame(tick: u32) -> Vec<u8> {
    frame::encode(&WorldFrame {
        tick,
        characters: vec![CharacterState {
            id: 0,
            alive: true,
            x: 100,
            y: 100,
            aim_x: 1,
            aim_y: 0,
            hook_state: 0,
            hook_x: 0,
            hook_y: 0,
            hooked_id: None,
            weapon: 0,
            team: 0,
            frozen: false,
            deep_frozen: false,
            live_frozen: false,
        }],
    })
}

// ---- a scripted bridge that can come and go ---------------------------------------------------------

#[derive(Default)]
struct BridgeLog {
    /// Subscription masks in the order they arrived (across connections).
    masks: Vec<u8>,
    /// Anything that was not a well-formed subscription.
    junk: usize,
    connects: usize,
}

#[derive(Clone)]
struct Script {
    /// First tick of the frames it sends (tells the live bot's frames from the demo's).
    tick0: u32,
    map_name: &'static str,
    map_sha: String,
    map_size: i32,
    /// `STATUS` JSON sent with every fourth frame.
    status: String,
    /// Every n-th fly frame is malformed (0: none).
    bad_fly_every: u32,
}

struct Bridge {
    path: PathBuf,
    script: Script,
    log: Arc<Mutex<BridgeLog>>,
    tasks: Vec<tokio::task::JoinHandle<()>>,
    conns: Arc<Mutex<Vec<tokio::task::JoinHandle<()>>>>,
}

impl Bridge {
    fn new(path: PathBuf, script: Script) -> Bridge {
        Bridge {
            path,
            script,
            log: Arc::default(),
            tasks: Vec::new(),
            conns: Arc::default(),
        }
    }

    /// Binds the socket and serves (the bot coming up).
    fn up(&mut self) {
        let _ = std::fs::remove_file(&self.path);
        let listener = UnixListener::bind(&self.path).unwrap();
        let (script, log, conns) = (self.script.clone(), Arc::clone(&self.log), Arc::clone(&self.conns));
        self.tasks.push(tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    return;
                };
                log.lock().unwrap().connects += 1;
                let (script, log) = (script.clone(), Arc::clone(&log));
                let conn = tokio::spawn(serve(stream, script, log));
                conns.lock().unwrap().push(conn);
            }
        }));
    }

    /// Closes everything and removes the socket (the bot going away).
    fn down(&mut self) {
        for t in self.tasks.drain(..) {
            t.abort();
        }
        for c in self.conns.lock().unwrap().drain(..) {
            c.abort();
        }
        let _ = std::fs::remove_file(&self.path);
    }

    fn masks(&self) -> Vec<u8> {
        self.log.lock().unwrap().masks.clone()
    }

    fn junk(&self) -> usize {
        self.log.lock().unwrap().junk
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        self.down();
    }
}

async fn serve(stream: tokio::net::UnixStream, script: Script, log: Arc<Mutex<BridgeLog>>) {
    let (mut rd, mut wr) = stream.into_split();
    let _ = wr.write_all(&msg(1, b"DDBL\x01")).await;
    let map = format!(
        r#"{{"name":"{}","sha256":"{}","w":{},"h":{}}}"#,
        script.map_name, script.map_sha, script.map_size, script.map_size
    );
    let _ = wr.write_all(&msg(2, map.as_bytes())).await;
    let _ = wr
        .write_all(&msg(3, br#"{"own":0,"list":[{"id":0,"name":"c0-aaaaaaaa","team":0}]}"#))
        .await;
    let (mut fly, mut tick, mut seq) = (false, script.tick0, 0u32);
    let mut every = tokio::time::interval(Duration::from_millis(50));
    let mut buf = [0u8; 6];
    loop {
        tokio::select! {
            r = rd.read_exact(&mut buf) => {
                if r.is_err() { return; }
                if buf[..5] == [2, 0, 0, 0, 1] {
                    log.lock().unwrap().masks.push(buf[5]);
                    let want = buf[5] & 1 != 0;
                    if want && !fly {
                        let _ = wr.write_all(&msg(6, META.as_bytes())).await;
                    }
                    fly = want;
                } else {
                    log.lock().unwrap().junk += 1;
                }
            }
            _ = every.tick() => {
                tick += 1;
                let _ = wr.write_all(&msg(4, &world_frame(tick))).await;
                if tick % 4 == 0 {
                    let _ = wr.write_all(&msg(5, script.status.as_bytes())).await;
                }
                if fly {
                    seq += 1;
                    let mut f = fly_frame(seq);
                    if script.bad_fly_every > 0 && seq.is_multiple_of(script.bad_fly_every) {
                        f.truncate(f.len() - 1);
                    }
                    let _ = wr.write_all(&msg(7, &f)).await;
                }
            }
        }
    }
}

// ---- the control socket of the bot (commands go here, never to a demo) ------------------------------

struct ControlBot {
    lines: Arc<Mutex<Vec<String>>>,
    task: tokio::task::JoinHandle<()>,
}

impl ControlBot {
    fn start(socket: &Path) -> ControlBot {
        std::fs::create_dir_all(socket.parent().unwrap()).unwrap();
        let listener = UnixListener::bind(socket).unwrap();
        let lines = Arc::new(Mutex::new(Vec::new()));
        let lines2 = Arc::clone(&lines);
        let task = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    return;
                };
                let lines = Arc::clone(&lines2);
                tokio::spawn(async move {
                    let (r, mut w) = stream.into_split();
                    let mut reader = BufReader::new(r);
                    let mut line = String::new();
                    while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                        lines.lock().unwrap().push(line.trim_end().to_string());
                        let mut out =
                            serde_json::to_vec(&ddai_botctl::proto::ControlReply::answer(true, "done")).unwrap();
                        out.push(b'\n');
                        let _ = w.write_all(&out).await;
                        line.clear();
                    }
                });
            }
        });
        ControlBot { lines, task }
    }

    fn received(&self) -> usize {
        self.lines.lock().unwrap().len()
    }
}

impl Drop for ControlBot {
    fn drop(&mut self) {
        self.task.abort();
    }
}

// ---- the rig ----------------------------------------------------------------------------------------

struct Rig {
    server: TestServer,
    live: Bridge,
    demo: Bridge,
    _dir: tempfile::TempDir,
}

fn live_status() -> String {
    serde_json::json!({
        "tick": 1, "own": 0, "target": 1, "mode": "fight", "brain": "hybrid", "alive": true, "frozen": false,
        "blocks": 0, "blocked_by": 0, "self_kills": 0, "decisions": 1, "collapsed": 0, "decide_p50_us": 1,
        "decide_p99_us": 2, "brain_p99_us": 1, "overhead_p99_us": 1, "telemetry": null, "connected": true,
        "server": "127.0.0.1:8303", "name": "bot", "clan": "c", "skin": "s", "target_tag": null, "wb": "WB: off",
        "goto": "", "deaths": 0, "clips_saved": 0, "kill_cooldown_ticks": 0
    })
    .to_string()
}

const DEMO_STATUS: &str = r#"{"demo":true,"arena":"clb-left","bundle":"e005-fly/final"}"#;

/// `live_up` / `demo_up`: which bridges exist when the web unit starts.
async fn rig(live_up: bool, demo_up: bool, bad_demo_fly_every: u32) -> Rig {
    rig_with_capacity(live_up, demo_up, bad_demo_fly_every, None).await
}

/// `event_capacity`: the hub's event channel size (tiny: connections lose events and must be told the state again).
async fn rig_with_capacity(
    live_up: bool,
    demo_up: bool,
    bad_demo_fly_every: u32,
    event_capacity: Option<usize>,
) -> Rig {
    let dir = tempfile::tempdir().unwrap();
    let maps = dir.path().join("maps");
    std::fs::create_dir_all(&maps).unwrap();
    let live_sha = write_map(&maps, "Live Map", 2);
    let demo_sha = write_map(&maps, "Demo Arena", 3);
    let (live_sock, demo_sock) = (dir.path().join("live.sock"), dir.path().join("demo.sock"));
    let mut live = Bridge::new(
        live_sock.clone(),
        Script {
            tick0: LIVE_TICK0,
            map_name: "Live Map",
            map_sha: live_sha,
            map_size: 2,
            status: live_status(),
            bad_fly_every: 0,
        },
    );
    let mut demo = Bridge::new(
        demo_sock.clone(),
        Script {
            tick0: DEMO_TICK0,
            map_name: "Demo Arena",
            map_sha: demo_sha,
            map_size: 3,
            status: DEMO_STATUS.to_string(),
            bad_fly_every: bad_demo_fly_every,
        },
    );
    if live_up {
        live.up();
    }
    if demo_up {
        demo.up();
    }
    let server = TestServer::start_with(move |c| {
        c.bot_socket = Some(live_sock);
        c.demo_socket = Some(demo_sock);
        c.map_search_dirs = vec![maps];
        if let Some(n) = event_capacity {
            c.event_broadcast_capacity = n;
        }
    })
    .await;
    Rig {
        server,
        live,
        demo,
        _dir: dir,
    }
}

async fn connect(server: &TestServer) -> Ws {
    let (cookie, _) = server.login();
    let req = ClientRequestBuilder::new(format!("ws://{}/ws", server.addr).parse().unwrap())
        .with_header("Cookie", cookie)
        .with_header("Origin", server.origin())
        .into_client_request()
        .unwrap();
    let (mut ws, _) = tokio_tungstenite::connect_async(req).await.expect("upgrade");
    let hello = next_text(&mut ws).await;
    assert_eq!(hello["type"], "hello");
    ws
}

/// The next text message that is not the 1 Hz `status`.
async fn next_text(ws: &mut Ws) -> serde_json::Value {
    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        let left = deadline
            .checked_duration_since(Instant::now())
            .expect("a message in time");
        match tokio::time::timeout(left, ws.next())
            .await
            .expect("a message in time")
            .expect("open")
            .expect("ok")
        {
            Message::Text(t) => {
                let v: serde_json::Value = serde_json::from_str(&t).unwrap();
                if v["type"] != "status" {
                    return v;
                }
            }
            Message::Ping(_) | Message::Pong(_) | Message::Binary(_) => {}
            other => panic!("unexpected {other:?}"),
        }
    }
}

/// Reads until a `source` message of `kind` arrives; the `source` messages passed on the way are returned too.
async fn wait_source(ws: &mut Ws, kind: &str) -> serde_json::Value {
    loop {
        let m = next_text(ws).await;
        if m["type"] == "source" && m["kind"] == kind {
            return m;
        }
    }
}

/// Everything that arrives within `window`: the ticks of the `DWLF` frames, and the text messages (but the 1 Hz status).
async fn collect(ws: &mut Ws, window: Duration) -> (Vec<u32>, Vec<serde_json::Value>, usize) {
    let end = Instant::now() + window;
    let (mut ticks, mut texts, mut fly) = (Vec::new(), Vec::new(), 0);
    while let Some(left) = end.checked_duration_since(Instant::now()) {
        match tokio::time::timeout(left, ws.next()).await {
            Ok(Some(Ok(Message::Binary(b)))) => {
                if b.starts_with(b"DWLF") {
                    ticks.push(u32::from_le_bytes([b[6], b[7], b[8], b[9]]));
                } else if b.starts_with(b"DFLY") {
                    fly += 1;
                }
            }
            Ok(Some(Ok(Message::Text(t)))) => {
                let v: serde_json::Value = serde_json::from_str(&t).unwrap();
                if v["type"] != "status" {
                    texts.push(v);
                }
            }
            Ok(Some(Ok(_))) => {}
            Ok(_) | Err(_) => break,
        }
    }
    (ticks, texts, fly)
}

/// What a browser sees, in the order it sees it: a `source` message or a `DWLF` frame (by the tick it carries).
#[derive(Debug, PartialEq)]
enum Seen {
    Source(String),
    Frame(u32),
}

async fn collect_ordered(ws: &mut Ws, window: Duration) -> Vec<Seen> {
    let end = Instant::now() + window;
    let mut out = Vec::new();
    while let Some(left) = end.checked_duration_since(Instant::now()) {
        match tokio::time::timeout(left, ws.next()).await {
            Ok(Some(Ok(Message::Binary(b)))) if b.starts_with(b"DWLF") => {
                out.push(Seen::Frame(u32::from_le_bytes([b[6], b[7], b[8], b[9]])));
            }
            Ok(Some(Ok(Message::Text(t)))) => {
                let v: serde_json::Value = serde_json::from_str(&t).unwrap();
                if v["type"] == "source" {
                    out.push(Seen::Source(v["kind"].as_str().unwrap().to_string()));
                }
            }
            Ok(Some(Ok(_))) => {}
            Ok(_) | Err(_) => break,
        }
    }
    out
}

/// Every frame is of the source the last `source` message named (a frame before any such message is not checked).
fn assert_frames_match_the_badge(seen: &[Seen]) {
    let mut shown: Option<&str> = None;
    for (i, item) in seen.iter().enumerate() {
        match item {
            Seen::Source(kind) => shown = Some(kind.as_str()),
            Seen::Frame(tick) => {
                let owner = if *tick >= LIVE_TICK0 { "live" } else { "demo" };
                if let Some(shown) = shown {
                    assert_eq!(
                        owner, shown,
                        "frame {tick} (message {i}) is drawn under the badge {shown}: {seen:?}"
                    );
                }
            }
        }
    }
}

async fn wait_for(what: &str, mut f: impl FnMut() -> bool) {
    for _ in 0..160 {
        if f() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("timed out waiting for {what}");
}

struct Login {
    cookie: String,
    csrf: String,
}

fn login(server: &TestServer) -> Login {
    let (cookie, csrf) = server.login();
    Login { cookie, csrf }
}

fn api_status(server: &TestServer, l: &Login) -> serde_json::Value {
    send(server.addr, Req::new("GET", "/api/bot/status").cookie(&l.cookie)).json()
}

fn post(server: &TestServer, l: &Login, path: &str, body: &serde_json::Value) -> support::RawResponse {
    send(
        server.addr,
        Req::new("POST", path)
            .cookie(&l.cookie)
            .header("Origin", &server.origin())
            .header("X-CSRF-Token", &l.csrf)
            .json_body(body),
    )
}

// ---- tests ------------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_demo_stands_in_while_the_bot_is_absent_and_the_bot_takes_over_and_gives_back() {
    let mut r = rig(false, true, 0).await;
    let l = login(&r.server);
    let mut ws = connect(&r.server).await;

    // Only the demo is there: it is shown, with its description and its own map and frames.
    let mut s = wait_source(&mut ws, "demo").await;
    while !s["info"].is_object() {
        // The description may follow the switch by a status interval.
        s = wait_source(&mut ws, "demo").await;
    }
    assert_eq!(
        s["info"],
        serde_json::json!({"arena": "clb-left", "bundle": "e005-fly/final"})
    );
    let (ticks, texts, _) = collect(&mut ws, Duration::from_millis(700)).await;
    assert!(ticks.len() >= 5, "demo frames flow: {ticks:?}");
    assert!(ticks.iter().all(|t| *t < LIVE_TICK0), "only demo frames: {ticks:?}");
    let maps: Vec<&str> = texts
        .iter()
        .filter(|m| m["type"] == "map")
        .map(|m| m["name"].as_str().unwrap())
        .collect();
    assert!(maps.iter().all(|n| *n == "Demo Arena"), "the demo's map: {maps:?}");
    // The bot panel's status is about the bot: there is none.
    let st = api_status(&r.server, &l);
    assert_eq!(
        (
            st["source"].as_str(),
            st["live"].as_bool(),
            st["demo_configured"].as_bool()
        ),
        (Some("demo"), Some(false), Some(true))
    );
    assert_eq!(
        st["status"],
        serde_json::Value::Null,
        "the demo's STATUS is not the bot's"
    );

    // The bot comes up: the page is told, the map is the bot's, only its frames come, and its status is the bot panel's.
    r.live.up();
    wait_source(&mut ws, "live").await;
    // Whatever was in flight from the demo is gone after a moment; from then on only live frames.
    let _ = collect(&mut ws, Duration::from_millis(250)).await;
    let (ticks, texts, _) = collect(&mut ws, Duration::from_millis(700)).await;
    assert!(ticks.len() >= 5, "live frames flow: {ticks:?}");
    assert!(ticks.iter().all(|t| *t >= LIVE_TICK0), "only live frames: {ticks:?}");
    assert!(
        !texts.iter().any(|m| m["type"] == "source" && m["kind"] == "demo"),
        "no flapping back to the demo: {texts:?}"
    );
    wait_for("the bot's status", || {
        let s = api_status(&r.server, &l);
        s["source"] == "live" && s["live"] == true && s["status"]["mode"] == "fight"
    })
    .await;

    // The bot goes away: the demo is back, with its map again.
    r.live.down();
    wait_source(&mut ws, "demo").await;
    let _ = collect(&mut ws, Duration::from_millis(250)).await;
    let (ticks, texts, _) = collect(&mut ws, Duration::from_millis(700)).await;
    assert!(
        ticks.len() >= 5 && ticks.iter().all(|t| *t < LIVE_TICK0),
        "demo frames again: {ticks:?}"
    );
    let _ = texts;
    let st = api_status(&r.server, &l);
    assert_eq!(
        (st["source"].as_str(), st["live"].as_bool()),
        (Some("demo"), Some(false))
    );

    // Neither: the page is told there is nothing.
    r.demo.down();
    wait_source(&mut ws, "none").await;
    let st = api_status(&r.server, &l);
    assert_eq!(
        (st["source"].as_str(), st["live"].as_bool()),
        (Some("none"), Some(false))
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_browser_that_connects_later_is_told_what_is_on_show_before_anything_else() {
    let r = rig(false, true, 0).await;
    let l = login(&r.server);
    wait_for("the demo on show", || api_status(&r.server, &l)["source"] == "demo").await;
    let mut ws = connect(&r.server).await;
    // The first messages after the hello: the source, then the map.
    let first = next_text(&mut ws).await;
    assert_eq!(
        (first["type"].as_str(), first["kind"].as_str()),
        (Some("source"), Some("demo")),
        "{first}"
    );
    let mut seen_map = false;
    for _ in 0..4 {
        let m = next_text(&mut ws).await;
        if m["type"] == "map" {
            assert_eq!(m["name"], "Demo Arena");
            seen_map = true;
            break;
        }
    }
    assert!(seen_map, "the demo's map follows the source");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn commands_are_refused_while_there_is_only_the_demo_and_reach_nothing_and_work_with_a_bot() {
    let mut r = rig(false, true, 0).await;
    let l = login(&r.server);
    wait_for("the demo on show", || api_status(&r.server, &l)["source"] == "demo").await;
    assert_eq!(api_status(&r.server, &l)["control_socket"], false);

    // Only the demo is up: no bot, no control socket anywhere.
    for body in [
        serde_json::json!({"type": "stop"}),
        serde_json::json!({"type": "mode", "mode": "passive"}),
        serde_json::json!({"type": "kill"}),
        serde_json::json!({"nonsense": true}),
    ] {
        let resp = post(&r.server, &l, "/api/bot/command", &body);
        assert_eq!(
            (resp.status, resp.json()["error"].as_str().map(str::to_string)),
            (503, Some("demo_only".to_string())),
            "{body}"
        );
    }
    assert!(
        !r.server.config.control_socket.exists(),
        "the web never creates a control socket"
    );
    // The refusals of 5.6 still come first: no CSRF, no Origin, no session.
    let no_csrf = send(
        r.server.addr,
        Req::new("POST", "/api/bot/command")
            .cookie(&l.cookie)
            .header("Origin", &r.server.origin())
            .json_body(&serde_json::json!({"type": "stop"})),
    );
    assert_eq!(
        (no_csrf.status, no_csrf.json()["error"].as_str().map(str::to_string)),
        (403, Some("missing_csrf".to_string()))
    );
    let foreign = send(
        r.server.addr,
        Req::new("POST", "/api/bot/command")
            .cookie(&l.cookie)
            .header("Origin", "http://evil.example")
            .header("X-CSRF-Token", &l.csrf)
            .json_body(&serde_json::json!({"type": "stop"})),
    );
    assert_eq!(foreign.status, 403);
    let anon = send(
        r.server.addr,
        Req::new("POST", "/api/bot/command")
            .header("Origin", &r.server.origin())
            .header("X-CSRF-Token", "AAAA")
            .json_body(&serde_json::json!({"type": "stop"})),
    );
    assert_eq!(anon.status, 401);

    // The lists still edit (a file), but nothing is sent to a bot: there is none to tell.
    let edit = post(
        &r.server,
        &l,
        "/api/bot/relations",
        &serde_json::json!({"op": "add", "kind": "friend", "name": "someone"}),
    );
    assert_eq!(edit.status, 200);
    assert_eq!(edit.json()["applied"], "unavailable");
    // And the demo's socket only ever got subscriptions, never anything that looks like a command.
    assert_eq!(r.demo.junk(), 0);

    // A bot that is running but whose bridge is away (`--no-bridge`, or the site's bridge connection reconnecting) still has
    // its control socket: it stays commandable although the demo is what the page shows.
    let control = ControlBot::start(&r.server.config.control_socket);
    wait_for("the control socket to be seen", || {
        api_status(&r.server, &l)["control_socket"] == true
    })
    .await;
    assert_eq!(api_status(&r.server, &l)["source"], "demo");
    let resp = post(&r.server, &l, "/api/bot/command", &serde_json::json!({"type": "stop"}));
    assert_eq!(resp.status, 200, "{}", String::from_utf8_lossy(&resp.body));
    assert_eq!(
        control.received(),
        1,
        "it reached the bot's control socket, and nowhere else"
    );
    assert_eq!(r.demo.junk(), 0);

    // The bot's bridge comes up: the same command goes through.
    r.live.up();
    wait_for("the bot on show", || api_status(&r.server, &l)["source"] == "live").await;
    let resp = post(&r.server, &l, "/api/bot/command", &serde_json::json!({"type": "stop"}));
    assert_eq!(resp.status, 200, "{}", String::from_utf8_lossy(&resp.body));
    assert_eq!(control.received(), 2);
    assert_eq!(r.demo.junk(), 0);
}

/// Review F2: while frames flow from both bridges, switching back and forth never draws one source's frame under the other's
/// badge: in the order the browser receives things, every frame belongs to the last `source` message.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_frame_of_one_source_never_follows_the_badge_of_the_other() {
    let mut r = rig(false, true, 0).await;
    let mut ws = connect(&r.server).await;
    wait_source(&mut ws, "demo").await;
    let mut all = Vec::new();
    for round in 0..3 {
        r.live.up();
        let seen = collect_ordered(&mut ws, Duration::from_millis(1500)).await;
        assert!(
            seen.contains(&Seen::Source("live".to_string())),
            "round {round}: switched to the bot: {seen:?}"
        );
        assert!(
            seen.iter().any(|s| matches!(s, Seen::Frame(t) if *t >= LIVE_TICK0)),
            "round {round}: its frames came"
        );
        all.extend(seen);
        r.live.down();
        let seen = collect_ordered(&mut ws, Duration::from_millis(4500)).await;
        assert!(
            seen.contains(&Seen::Source("demo".to_string())),
            "round {round}: back to the demo: {seen:?}"
        );
        assert!(
            seen.iter().any(|s| matches!(s, Seen::Frame(t) if *t < LIVE_TICK0)),
            "round {round}: its frames came"
        );
        all.extend(seen);
    }
    assert_frames_match_the_badge(&all);
}

/// Review F4: a connection that lags past events (here: a channel of one message, so any burst loses some, a `source` among them)
/// is told the state again, so the frames of the new source are let through under the right badge and the page does not freeze.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_connection_that_lost_events_is_told_what_is_on_show_and_its_frames_resume() {
    let mut r = rig_with_capacity(false, true, 0, Some(1)).await;
    let mut ws = connect(&r.server).await;
    wait_source(&mut ws, "demo").await;
    let mut all = Vec::new();
    for round in 0..4 {
        r.live.up();
        let seen = collect_ordered(&mut ws, Duration::from_millis(2000)).await;
        assert!(
            seen.contains(&Seen::Source("live".to_string())),
            "round {round}: told about the bot: {seen:?}"
        );
        assert!(
            seen.iter().any(|s| matches!(s, Seen::Frame(t) if *t >= LIVE_TICK0)),
            "round {round}: the bot's frames were not dropped for good"
        );
        all.extend(seen);
        r.live.down();
        let seen = collect_ordered(&mut ws, Duration::from_millis(4500)).await;
        assert!(
            seen.contains(&Seen::Source("demo".to_string())),
            "round {round}: told about the demo: {seen:?}"
        );
        assert!(
            seen.iter().any(|s| matches!(s, Seen::Frame(t) if *t < LIVE_TICK0)),
            "round {round}: the demo's frames are back"
        );
        all.extend(seen);
    }
    assert_frames_match_the_badge(&all);
}

/// Review F3: a bot whose bridge connection is only briefly away does not flip the page to the demo.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_short_break_of_the_bots_bridge_does_not_flip_the_page_to_the_demo() {
    let mut r = rig(true, true, 0).await;
    let mut ws = connect(&r.server).await;
    wait_source(&mut ws, "live").await;
    r.live.down();
    tokio::time::sleep(Duration::from_millis(600)).await;
    r.live.up();
    let seen = collect_ordered(&mut ws, Duration::from_millis(3500)).await;
    assert!(
        !seen.contains(&Seen::Source("demo".to_string())),
        "no flip to the demo: {seen:?}"
    );
    assert!(
        seen.iter().any(|s| matches!(s, Seen::Frame(t) if *t >= LIVE_TICK0)),
        "the bot's frames are back: {seen:?}"
    );
    assert_frames_match_the_badge(&seen);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_demo_is_asked_for_viewers_only_while_a_browser_is_open_and_the_bot_is_not_up() {
    let mut r = rig(false, true, 0).await;
    tokio::time::sleep(Duration::from_millis(400)).await;
    assert!(r.demo.masks().is_empty(), "no browser: the demo is asked for nothing");

    // A browser opens the site: the demo is told somebody is looking (bit 1).
    let ws = connect(&r.server).await;
    wait_for("the view bit at the demo", || r.demo.masks() == [2]).await;

    // The bot comes up while the browser is there: the bot is told, the demo is told to rest.
    r.live.up();
    wait_for("the demo to be released", || r.demo.masks() == [2, 0]).await;
    wait_for("the bot to be told", || r.live.masks() == [2]).await;

    // The browser leaves: the bot is told nobody is there; the demo hears nothing more.
    drop(ws);
    wait_for("the bot to hear nobody", || r.live.masks() == [2, 0]).await;
    assert_eq!(r.demo.masks(), [2, 0]);

    // The bot goes; with nobody watching the demo is not asked; a browser opening asks it again.
    r.live.down();
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(r.demo.masks(), [2, 0], "nobody watches: the demo stays at rest");
    let _ws = connect(&r.server).await;
    wait_for("the demo to be asked again", || r.demo.masks() == [2, 0, 2]).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_fly_stream_of_the_demo_is_validated_like_the_bots() {
    // Every second fly frame of the demo is malformed.
    let r = rig(false, true, 2).await;
    let mut ws = connect(&r.server).await;
    wait_source(&mut ws, "demo").await;
    ws.send(Message::Text(r#"{"type":"fly","hz":15}"#.into()))
        .await
        .unwrap();
    let (_, texts, fly) = collect(&mut ws, Duration::from_millis(1500)).await;
    assert!(fly >= 3, "the good frames arrive: {fly}");
    assert!(
        texts
            .iter()
            .any(|m| m["type"] == "fly_meta" && m["meta"]["bundle"]["name"] == "run/final")
    );
    let errors: Vec<_> = texts.iter().filter(|m| m["type"] == "live_error").collect();
    assert_eq!(
        errors.len(),
        1,
        "one report of the malformed stream, not one per frame: {errors:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_demo_socket_without_a_live_socket_or_equal_to_it_is_refused_at_start() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = ddai_web::WebConfig::new("127.0.0.1:0".parse().unwrap(), dir.path().to_path_buf());
    cfg.demo_socket = Some(dir.path().join("demo.sock"));
    let e = ddai_web::bind(cfg.clone()).await.err().expect("refused");
    assert!(e.to_string().contains("needs the live bot's socket"), "{e}");
    cfg.bot_socket = cfg.demo_socket.clone();
    let e = ddai_web::bind(cfg).await.err().expect("refused");
    assert!(e.to_string().contains("never be the live socket"), "{e}");
}
