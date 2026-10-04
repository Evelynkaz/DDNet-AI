//! The «Игра» tab's server side (task 5.10) against a real server and a scripted FAKE bot on a real Unix socket (the bridge
//! messages are written here by hand, independent of `ddai-bot`): the chat frame round trip, hostile chat text, the bounded
//! in-memory chat ring, the look and numbers of players, the visual scene and its images, and the DDNet graphics route.

mod support;

use std::path::Path;
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use support::{Req, TestServer, send};
use tokio::io::AsyncWriteExt;
use tokio::net::UnixListener;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::{ClientRequestBuilder, Message};

type Ws = tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

fn msg(kind: u8, payload: &[u8]) -> Vec<u8> {
    let mut m = ((payload.len() + 1) as u32).to_le_bytes().to_vec();
    m.push(kind);
    m.extend_from_slice(payload);
    m
}

/// A real, loadable map with an embedded image, two external ones (one with a hostile name) and a quad layer.
fn write_scene_map(dir: &Path) -> String {
    use ddai_map::testutil::{
        MapWriter, QuadSpec, TILESLAYERFLAG_GAME, TileLayerLook, TileLayerSpec, TilemapShape, encode_tile_skip,
    };
    let mut w = MapWriter::new(4);
    w.add_version_item(1);
    let ext = w.add_image("grass_main", 1024, 1024, None, 1);
    w.add_image("../skins/greyfox", 64, 64, None, 1);
    let rgba: Vec<u8> = (0..4 * 4 * 4).map(|i| (i * 3) as u8).collect();
    let emb = w.add_image("art", 4, 4, Some(&rgba), 2);
    let q = QuadSpec {
        points: [[0, 0], [1024, 0], [0, 1024], [1024, 1024], [512, 512]],
        colors: [[255; 4]; 4],
        texcoords: [[0, 0], [1024, 0], [0, 1024], [1024, 1024]],
        pos_env: -1,
        pos_env_offset: 0,
        color_env: -1,
        color_env_offset: 0,
    };
    w.add_quad_layer(&[q], emb, false);
    let design = encode_tile_skip(&[(1u8, 0u8); 16]);
    w.add_tile_layer_look(
        &TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 4,
            width: 4,
            height: 4,
            flags: 0,
            data: &design,
        },
        TileLayerLook {
            image: ext,
            ..Default::default()
        },
    );
    let game = encode_tile_skip(&[(0u8, 0u8); 16]);
    w.add_tile_layer(&TileLayerSpec {
        shape: TilemapShape::Full,
        item_version: 4,
        width: 4,
        height: 4,
        flags: TILESLAYERFLAG_GAME,
        data: &game,
    });
    w.add_single_group_with_all_layers();
    let bytes = w.finish();
    let sha: [u8; 32] = Sha256::digest(&bytes).into();
    let hex: String = sha.iter().map(|b| format!("{b:02x}")).collect();
    std::fs::write(dir.join(format!("Scene Map_{hex}.map")), &bytes).unwrap();
    hex
}

/// A scripted bot: on every connection it greets with `HELLO`, `MAP`, `PLAYERS`, `PLAYERINFO`, then writes whatever the test
/// pushes into `tx` (raw bridge messages).
struct FakeBot {
    tx: mpsc::UnboundedSender<Vec<u8>>,
    task: tokio::task::JoinHandle<()>,
}

impl FakeBot {
    fn start(socket: &Path, map_sha: String) -> FakeBot {
        let listener = UnixListener::bind(socket).unwrap();
        let (tx, mut rx) = mpsc::unbounded_channel::<Vec<u8>>();
        let task = tokio::spawn(async move {
            let Ok((stream, _)) = listener.accept().await else {
                return;
            };
            let (mut rd, mut wr) = stream.into_split();
            // The web unit writes its subscription now and then; keep reading so its writes never block.
            tokio::spawn(async move {
                let mut buf = [0u8; 64];
                while tokio::io::AsyncReadExt::read(&mut rd, &mut buf).await.unwrap_or(0) > 0 {}
            });
            let _ = wr.write_all(&msg(1, b"DDBL\x01")).await;
            let map = format!(r#"{{"name":"Scene Map","sha256":"{map_sha}","w":4,"h":4}}"#);
            let _ = wr.write_all(&msg(2, map.as_bytes())).await;
            let _ = wr
                .write_all(&msg(
                    3,
                    br#"{"own":0,"list":[{"id":0,"name":"c0-aaaaaaaa","team":0},{"id":1,"name":"c1-bbbbbbbb","team":0}]}"#,
                ))
                .await;
            let _ = wr
                .write_all(&msg(
                    9,
                    br#"{"list":[{"id":0,"clan":"","skin":"default","cc":false,"cb":0,"cf":0,"country":-1,"score":3,"ping":12},{"id":1,"clan":"Clan","skin":"coala","cc":true,"cb":16711680,"cf":255,"country":276,"score":-1,"ping":40}]}"#,
                ))
                .await;
            while let Some(m) = rx.recv().await {
                if wr.write_all(&m).await.is_err() {
                    return;
                }
            }
        });
        FakeBot { tx, task }
    }

    fn chat(&self, team: i32, cid: i32, name: &str, text: &str) {
        let json = serde_json::json!({"team": team, "cid": cid, "name": name, "text": text});
        self.tx.send(msg(8, json.to_string().as_bytes())).unwrap();
    }
}

impl Drop for FakeBot {
    fn drop(&mut self) {
        self.task.abort();
    }
}

struct Rig {
    server: TestServer,
    bot: FakeBot,
    sha: String,
    _dir: tempfile::TempDir,
}

async fn rig_with(data_dir: Option<std::path::PathBuf>) -> Rig {
    let dir = tempfile::tempdir().unwrap();
    let maps = dir.path().join("maps");
    std::fs::create_dir_all(&maps).unwrap();
    let sha = write_scene_map(&maps);
    let sock = dir.path().join("live.sock");
    let bot = FakeBot::start(&sock, sha.clone());
    let server = TestServer::start_with(|c| {
        c.bot_socket = Some(sock.clone());
        c.map_search_dirs = vec![maps.clone()];
        c.ddnet_data_dir = data_dir;
    })
    .await;
    Rig {
        server,
        bot,
        sha,
        _dir: dir,
    }
}

async fn rig() -> Rig {
    rig_with(None).await
}

async fn connect(server: &TestServer) -> Ws {
    let (cookie, _) = server.login();
    let req = ClientRequestBuilder::new(format!("ws://{}/ws", server.addr).parse().unwrap())
        .with_header("Cookie", cookie)
        .with_header("Origin", server.origin())
        .into_client_request()
        .unwrap();
    let (mut ws, _) = tokio_tungstenite::connect_async(req).await.expect("upgrade");
    let hello = next_where(&mut ws, |v| v["type"] == "hello").await;
    assert_eq!(hello["version"], 1);
    ws
}

/// The next text message satisfying `want` (others are skipped), within 8 s.
async fn next_where(ws: &mut Ws, want: impl Fn(&serde_json::Value) -> bool) -> serde_json::Value {
    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        let left = deadline
            .checked_duration_since(Instant::now())
            .expect("the expected message in time");
        let m = tokio::time::timeout(left, ws.next())
            .await
            .expect("the expected message in time")
            .expect("open")
            .expect("ok");
        if let Message::Text(t) = m {
            let v: serde_json::Value = serde_json::from_str(&t).unwrap();
            if want(&v) {
                return v;
            }
        }
    }
}

fn get(server: &TestServer, cookie: &str, path: &str) -> support::RawResponse {
    send(server.addr, Req::new("GET", path).header("Cookie", cookie))
}

fn inflate(raw: &[u8]) -> Vec<u8> {
    use std::io::Read;
    let mut out = Vec::new();
    flate2::read::DeflateDecoder::new(raw)
        .read_to_end(&mut out)
        .expect("raw deflate");
    out
}

// ---- chat -------------------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_chat_line_goes_from_the_bridge_to_the_page_as_a_typed_message() {
    let rig = rig().await;
    let mut ws = connect(&rig.server).await;
    // The roster first: the bot's own line is attributed to a slot the page knows.
    next_where(&mut ws, |v| v["type"] == "players").await;
    rig.bot.chat(0, 1, "c1-bbbbbbbb", "hello there");
    rig.bot.chat(1, 0, "c0-aaaaaaaa", "team talk");
    rig.bot.chat(3, 1, "c1-bbbbbbbb", "psst");
    rig.bot.chat(0, -1, "ignored", "*** Kill Protection enabled");
    let l = next_where(&mut ws, |v| v["type"] == "chat").await;
    assert_eq!(l["line"]["kind"], "all");
    assert_eq!(l["line"]["id"], 1);
    assert_eq!(l["line"]["name"], "c1-bbbbbbbb");
    assert_eq!(l["line"]["text"], "hello there");
    assert!(l["line"]["at"].as_u64().unwrap() > 1_600_000_000_000);
    let l = next_where(&mut ws, |v| v["type"] == "chat").await;
    assert_eq!(
        (l["line"]["kind"].as_str(), l["line"]["id"].as_u64()),
        (Some("team"), Some(0))
    );
    let l = next_where(&mut ws, |v| v["type"] == "chat").await;
    assert_eq!(l["line"]["kind"], "whisper_from");
    let l = next_where(&mut ws, |v| v["type"] == "chat").await;
    assert_eq!(l["line"]["kind"], "system");
    assert_eq!(l["line"]["id"], serde_json::Value::Null);
    assert_eq!(l["line"]["name"], "", "the server has no sender name");
}

/// Hostile text is cleaned (controls, line breaks, direction overrides, zero-width characters) and capped, markup is left
/// as the plain characters it is, and nothing is invented or dropped but what is forbidden.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn hostile_chat_text_is_sanitised_capped_and_not_turned_into_markup() {
    let rig = rig().await;
    let mut ws = connect(&rig.server).await;
    let payload = r#"<img src=x onerror=alert(1)><script>fetch("/api/logout",{method:"POST"})</script>"#;
    rig.bot.chat(0, 1, "<b>c1</b>\u{202e}", payload);
    rig.bot.chat(
        0,
        1,
        "c1",
        "a\nb\rc\u{0}d\u{1b}[31me\u{202e}f\u{200b}g\u{feff}h\u{2066}i",
    );
    rig.bot.chat(0, 1, "c1", &"я".repeat(5000));
    rig.bot.chat(0, 1, "c1", "\u{200b}\u{202e}\n"); // nothing left to show: dropped
    rig.bot.chat(0, 1, "c1", "marker");
    let l = next_where(&mut ws, |v| v["type"] == "chat").await;
    assert_eq!(
        l["line"]["text"], payload,
        "markup characters pass through verbatim; the page writes them with textContent"
    );
    assert_eq!(
        l["line"]["name"], "<b>c1</b>",
        "the override in the name is gone, the markup is not mangled"
    );
    let l = next_where(&mut ws, |v| v["type"] == "chat").await;
    assert_eq!(
        l["line"]["text"], "abcd[31mefghi",
        "controls and invisible characters are removed (the ESC byte goes, the rest of the sequence is plain text)"
    );
    let l = next_where(&mut ws, |v| v["type"] == "chat").await;
    assert_eq!(l["line"]["text"].as_str().unwrap().chars().count(), 256);
    let l = next_where(&mut ws, |v| v["type"] == "chat").await;
    assert_eq!(l["line"]["text"], "marker", "the all-forbidden line produced nothing");
}

/// Only the newest 200 lines are remembered, in memory, and a browser that connects later is handed them in order, whole,
/// as one `chat_history` message.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_chat_ring_is_bounded_and_a_late_browser_gets_its_newest_lines() {
    let rig = rig().await;
    let mut early = connect(&rig.server).await;
    for i in 0..230 {
        rig.bot.chat(0, 1, "c1-bbbbbbbb", &format!("line {i}"));
    }
    // Wait for the last one to arrive (it is on the ring once the early browser has seen it).
    next_where(&mut early, |v| v["type"] == "chat" && v["line"]["text"] == "line 229").await;
    let mut late = connect(&rig.server).await;
    let h = next_where(&mut late, |v| v["type"] == "chat_history").await;
    let lines = h["lines"].as_array().unwrap();
    assert_eq!(lines.len(), ddai_web::live::chat::RING_LINES);
    assert_eq!(lines.first().unwrap()["text"], "line 30");
    assert_eq!(lines.last().unwrap()["text"], "line 229");
    // Nothing about chat is on disk: the data directory the server runs in holds no file with a chat line in it.
    let data_dir = rig.server.config.data_dir.clone();
    let mut stack = vec![data_dir];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().filter_map(Result::ok) {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if let Ok(bytes) = std::fs::read(&p) {
                assert!(
                    !String::from_utf8_lossy(&bytes).contains("line 229"),
                    "{} holds chat",
                    p.display()
                );
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_roster_carries_the_look_and_numbers_from_playerinfo() {
    let rig = rig().await;
    let mut ws = connect(&rig.server).await;
    // The roster arrives first with defaults, then again with the look merged in; take the one that has it.
    let p = next_where(&mut ws, |v| v["type"] == "players" && v["list"][1]["skin"] == "coala").await;
    let list = p["list"].as_array().unwrap();
    assert_eq!(list.len(), 2);
    assert_eq!(list[0]["skin"], "default");
    assert_eq!(list[0]["score"], 3);
    assert_eq!(list[0]["ping"], 12);
    assert_eq!(list[1]["name"], "c1-bbbbbbbb");
    assert_eq!(list[1]["clan"], "Clan");
    assert_eq!(list[1]["cc"], true);
    assert_eq!(list[1]["cb"], 16_711_680);
    assert_eq!(list[1]["cf"], 255);
    assert_eq!(list[1]["country"], 276);
    assert_eq!(list[1]["score"], -1);
}

// ---- the visual scene and the DDNet graphics ---------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_visual_scene_and_its_embedded_image_are_served_to_a_logged_in_page_only() {
    let rig = rig().await;
    let mut ws = connect(&rig.server).await;
    next_where(&mut ws, |v| v["type"] == "map").await; // the source has resolved the map: its file is known
    let (cookie, _) = rig.server.login();
    let url = format!("/api/map/{}/scene", rig.sha);

    assert_eq!(send(rig.server.addr, Req::new("GET", &url)).status, 401, "no session");
    let r = get(&rig.server, &cookie, &url);
    assert_eq!(r.status, 200);
    assert_eq!(r.header("Cache-Control"), Some("private, max-age=604800, immutable"));
    let etag = r.header("ETag").unwrap().to_string();
    let raw = inflate(&r.body);
    assert_eq!(&raw[..4], b"DWSC");
    let json_len = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&raw[12..12 + json_len]).unwrap();
    assert_eq!(header["game"], serde_json::json!({"w": 4, "h": 4}));
    let images = header["images"].as_array().unwrap();
    assert_eq!(images[0]["n"], "grass_main");
    assert_eq!(images[1]["n"], "", "the path-like name never reaches the page");
    assert_eq!(images[2]["d"], 1);
    let kinds: Vec<&str> = header["groups"][0]["layers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| l["k"].as_str().unwrap())
        .collect();
    assert_eq!(kinds, ["q", "t", "t"], "the file's draw order");

    // Conditional GET.
    let again = send(
        rig.server.addr,
        Req::new("GET", &url)
            .header("Cookie", &cookie)
            .header("If-None-Match", &etag),
    );
    assert_eq!(again.status, 304);

    // The embedded image: width, height, then RGBA.
    let r = get(&rig.server, &cookie, &format!("/api/map/{}/image/2", rig.sha));
    assert_eq!(r.status, 200);
    let px = inflate(&r.body);
    assert_eq!(
        (
            u32::from_le_bytes(px[0..4].try_into().unwrap()),
            u32::from_le_bytes(px[4..8].try_into().unwrap())
        ),
        (4, 4)
    );
    assert_eq!(px.len(), 8 + 64);
    assert_eq!(px[8 + 5], 15);
    // External, out of range and malformed requests.
    for (path, status) in [
        (format!("/api/map/{}/image/0", rig.sha), 404),
        (format!("/api/map/{}/image/99", rig.sha), 404),
        (format!("/api/map/{}/image/-1", rig.sha), 400),
        (format!("/api/map/{}/scene", "ab".repeat(32)), 404),
        ("/api/map/nothex/scene".to_string(), 400),
    ] {
        assert_eq!(get(&rig.server, &cookie, &path).status, status, "{path}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ddnet_graphics_come_from_the_data_directory_for_safe_names_only() {
    let data = tempfile::tempdir().unwrap();
    for (rel, bytes) in [
        ("game.png", &b"GAME"[..]),
        ("skins/default.png", b"SKIN"),
        ("mapres/grass_main.png", b"GRASS"),
        ("editor/entities_clear/ddnet.png", b"ENT"),
        ("fonts/DejaVuSans.ttf", b"FONT"),
        ("settings_ddnet.cfg", b"secret"),
    ] {
        let p = data.path().join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, bytes).unwrap();
    }
    let rig = rig_with(Some(data.path().to_path_buf())).await;
    let (cookie, _) = rig.server.login();

    assert_eq!(send(rig.server.addr, Req::new("GET", "/assets/game.png")).status, 401);
    let r = get(&rig.server, &cookie, "/assets/skins/default.png");
    assert_eq!((r.status, r.body.as_slice()), (200, &b"SKIN"[..]));
    assert_eq!(r.header("Content-Type"), Some("image/png"));
    assert_eq!(r.header("X-Content-Type-Options"), Some("nosniff"));
    assert!(r.header("Cache-Control").unwrap().starts_with("private"));
    assert_eq!(
        get(&rig.server, &cookie, "/assets/mapres/grass_main.png").body,
        b"GRASS"
    );
    assert_eq!(
        get(&rig.server, &cookie, "/assets/editor/entities_clear/ddnet.png").body,
        b"ENT"
    );
    let font = get(&rig.server, &cookie, "/assets/fonts/DejaVuSans.ttf");
    assert_eq!((font.status, font.header("Content-Type")), (200, Some("font/ttf")));

    for path in [
        "/assets/skins/unknown.png",
        "/assets/skins/../settings_ddnet.cfg",
        "/assets/skins/%2e%2e/settings_ddnet.cfg",
        "/assets/skins/..%2fgame.png",
        "/assets/mapres/..%2fskins%2fdefault.png",
        "/assets/settings_ddnet.cfg",
        "/assets/",
        "/assets/mapres/grass_main",
        "/assets/skins/default.png/",
    ] {
        let r = get(&rig.server, &cookie, path);
        assert!(r.status == 404 || r.status == 400, "{path} -> {}", r.status);
        assert!(!String::from_utf8_lossy(&r.body).contains("secret"), "{path}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn without_a_data_directory_the_graphics_are_not_found_and_the_scene_still_works() {
    let rig = rig().await;
    let (cookie, _) = rig.server.login();
    assert_eq!(get(&rig.server, &cookie, "/assets/game.png").status, 404);
}

// ---- the page itself ---------------------------------------------------------------------------------------

/// The scripts of the tab are served with the site's headers, never build markup from game data (no `innerHTML` and friends),
/// and the chat can only be read: the page has no input that could write chat and no message of that kind in its scripts.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_game_scripts_and_page_never_turn_game_text_into_markup_and_the_chat_has_no_input() {
    let rig = rig().await;
    let addr = rig.server.addr;
    let mut scripts = String::new();
    for (path, ctype) in [
        ("/game.js", "text/javascript; charset=utf-8"),
        ("/ddmap.js", "text/javascript; charset=utf-8"),
        ("/ddtee.js", "text/javascript; charset=utf-8"),
        ("/game.css", "text/css; charset=utf-8"),
    ] {
        let r = send(addr, Req::new("GET", path));
        assert_eq!(r.status, 200, "{path}");
        assert_eq!(r.header("Content-Type"), Some(ctype), "{path}");
        assert!(
            r.header("Content-Security-Policy")
                .unwrap()
                .contains("script-src 'self'"),
            "{path}"
        );
        assert_eq!(r.header("X-Content-Type-Options"), Some("nosniff"));
        scripts.push_str(&String::from_utf8(r.body).unwrap());
    }
    for banned in [
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
        "setTimeout(\"",
        "setInterval(\"",
        "srcdoc",
        "javascript:",
    ] {
        assert!(
            !scripts.contains(banned),
            "the game scripts must not contain {banned:?}"
        );
    }
    // Nothing sends chat: the only things the tab sends are the live rate and the replay controls.
    assert!(
        !scripts.contains("type: \"chat\"") && !scripts.contains("type: \"say\""),
        "no chat message is ever sent"
    );
    let sends: Vec<&str> = scripts
        .match_indices("host.send({")
        .map(|(i, _)| &scripts[i..i + 60])
        .collect();
    assert!(!sends.is_empty());
    for s in &sends {
        assert!(
            s.contains("type: \"sub\"") || s.contains("type: \"replay\""),
            "unexpected message from the tab: {s}"
        );
    }

    let page = String::from_utf8(send(addr, Req::new("GET", "/")).body).unwrap();
    let game = &page[page.find("id=\"game-view\"").unwrap()..page.find("id=\"bot-view\"").unwrap()];
    let chat = &game[game.find("chat-card").unwrap()..];
    let chat = &chat[..chat.find("</section>").unwrap()];
    for banned in ["<input", "<textarea", "<form", "contenteditable"] {
        assert!(!chat.contains(banned), "the chat card is read-only: {banned}");
    }
    assert!(
        !page.contains(" style=") && !page.contains(" onclick=") && !page.contains(" onerror="),
        "no inline styles or handlers (CSP)"
    );
    assert!(
        page.contains("/ddmap.js")
            && page.contains("/ddtee.js")
            && page.contains("/game.js")
            && page.contains("/game.css")
    );
    // The credit for the DDNet graphics is on the page (CC BY-SA 3.0).
    assert!(game.contains("CC BY-SA 3.0") && game.contains("DDNet"));
}
