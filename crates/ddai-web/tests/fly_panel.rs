//! The «Муха» tab's server side (task 7.4, `docs/formats.md` §27) against a real server and a scripted FAKE bot on a Unix
//! socket (written here from the documented wire format, independent of `ddai-bot`): the WebSocket route is behind the
//! session and the Origin check, a browser's `{"type":"fly"}` makes the web unit subscribe at the bot (and only while someone
//! watches), the layout and the frames come back as `fly_meta` and binary `DFLY` messages, malformed frames never reach a
//! browser, and the per-connection rate is capped.

mod support;

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures_util::{SinkExt, StreamExt};
use support::{Req, TestServer, send};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixListener;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::{ClientRequestBuilder, Message};

type Ws = tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

const META: &str = r#"{"v":1,"role":"fly","name":"fly-test","bundle":{"name":"run/final","sha256":"abcd"},"rays":4,"bins":2,"groups":[],"dn":[],"channels":[],"scalars":[]}"#;

fn msg(kind: u8, payload: &[u8]) -> Vec<u8> {
    let mut m = ((payload.len() + 1) as u32).to_le_bytes().to_vec();
    m.push(kind);
    m.extend_from_slice(payload);
    m
}

/// A frame of the documented layout (2 groups, 3 DN, 4 rays x 2 bins x 7 channels, 3 scalars), `seq` in the header.
fn frame(seq: u32) -> Vec<u8> {
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

/// What the fake bot saw from the web unit, and what it does.
#[derive(Default)]
struct BotLog {
    /// Subscription masks in the order they arrived.
    masks: Vec<u8>,
    /// Anything that was not a well-formed subscription.
    junk: usize,
}

/// A fake bot: accepts connections, greets with HELLO, and while a client is subscribed to the fly sends the layout once and
/// then `frames_per_s` frames a second (`bad_every`: every n-th one is malformed). Never sends a thing otherwise.
fn fake_bot(socket: &Path, frames_per_s: u32, bad_every: u32) -> (tokio::task::JoinHandle<()>, Arc<Mutex<BotLog>>) {
    let log = Arc::new(Mutex::new(BotLog::default()));
    let listener = UnixListener::bind(socket).unwrap();
    let log2 = Arc::clone(&log);
    let task = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                return;
            };
            let log = Arc::clone(&log2);
            tokio::spawn(async move {
                let (mut rd, mut wr) = stream.into_split();
                let _ = wr.write_all(&msg(1, b"DDBL\x01")).await;
                let mut subscribed = false;
                let mut seq = 0u32;
                let mut tick = tokio::time::interval(Duration::from_micros(1_000_000 / u64::from(frames_per_s.max(1))));
                let mut buf = [0u8; 6];
                loop {
                    tokio::select! {
                        r = rd.read_exact(&mut buf) => {
                            if r.is_err() { return; }
                            // `u32 LE len = 2 | u8 kind 1 | u8 mask`
                            if buf[..4] == [2, 0, 0, 0] && buf[4] == 1 {
                                log.lock().unwrap().masks.push(buf[5]);
                                let want = buf[5] & 1 != 0;
                                if want && !subscribed {
                                    let _ = wr.write_all(&msg(6, META.as_bytes())).await;
                                }
                                subscribed = want;
                            } else {
                                log.lock().unwrap().junk += 1;
                            }
                        }
                        _ = tick.tick(), if subscribed => {
                            seq += 1;
                            let mut f = frame(seq);
                            if bad_every > 0 && seq.is_multiple_of(bad_every) {
                                f.truncate(f.len() - 1); // counts say one byte more
                            }
                            let _ = wr.write_all(&msg(7, &f)).await;
                        }
                    }
                }
            });
        }
    });
    (task, log)
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
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let left = deadline
            .checked_duration_since(Instant::now())
            .expect("a message within 5 s");
        match tokio::time::timeout(left, ws.next())
            .await
            .expect("a message within 5 s")
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

/// Binary messages that arrive within `window`.
async fn binaries_for(ws: &mut Ws, window: Duration) -> Vec<Vec<u8>> {
    let end = Instant::now() + window;
    let mut out = Vec::new();
    while let Some(left) = end.checked_duration_since(Instant::now()) {
        match tokio::time::timeout(left, ws.next()).await {
            Ok(Some(Ok(Message::Binary(b)))) => out.push(b.to_vec()),
            Ok(Some(Ok(_))) => {}
            Ok(_) => break,
            Err(_) => break,
        }
    }
    out
}

async fn wait_for(mut f: impl FnMut() -> bool) {
    for _ in 0..100 {
        if f() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("timed out");
}

async fn bot_server(
    frames_per_s: u32,
    bad_every: u32,
    customize: impl FnOnce(&mut ddai_web::config::WebConfig) + Send + 'static,
) -> (
    TestServer,
    Arc<Mutex<BotLog>>,
    tokio::task::JoinHandle<()>,
    tempfile::TempDir,
) {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("live.sock");
    let (task, log) = fake_bot(&sock, frames_per_s, bad_every);
    let server = TestServer::start_with(move |c| {
        c.bot_socket = Some(sock);
        customize(c);
    })
    .await;
    (server, log, task, dir)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_route_is_behind_the_session_and_the_origin_check() {
    let server = TestServer::start().await;
    let (cookie, _) = server.login();
    let uri = format!("ws://{}/ws", server.addr);
    for (cookie, origin, status) in [
        (None, Some(server.origin()), 401u16),
        (Some(cookie.clone()), Some("http://evil.example".to_string()), 403),
        (Some(cookie), None, 403),
    ] {
        let mut b = ClientRequestBuilder::new(uri.parse().unwrap());
        if let Some(c) = cookie {
            b = b.with_header("Cookie", c);
        }
        if let Some(o) = origin {
            b = b.with_header("Origin", o);
        }
        let err = tokio_tungstenite::connect_async(b.into_client_request().unwrap())
            .await
            .expect_err("refused");
        match err {
            tokio_tungstenite::tungstenite::Error::Http(r) => assert_eq!(r.status().as_u16(), status),
            other => panic!("{other:?}"),
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_browser_that_watches_makes_the_web_unit_subscribe_and_gets_the_layout_and_frames() {
    let (server, log, task, _dir) = bot_server(40, 0, |_| {}).await;
    let mut ws = connect(&server).await;
    // Nobody watches yet: the bot has been told nothing and sends nothing.
    tokio::time::sleep(Duration::from_millis(400)).await;
    assert!(
        log.lock().unwrap().masks.is_empty(),
        "no subscription before a browser asks"
    );
    assert!(binaries_for(&mut ws, Duration::from_millis(200)).await.is_empty());

    ws.send(Message::Text(r#"{"type":"fly","hz":50}"#.into()))
        .await
        .unwrap();
    // The layout (as it is now: none known yet is also fine), then the real one as the bot announces it.
    let mut meta = next_text(&mut ws).await;
    if meta["meta"].is_null() {
        meta = next_text(&mut ws).await;
    }
    assert_eq!(meta["type"], "fly_meta");
    assert_eq!(meta["meta"]["name"], "fly-test");
    assert_eq!(meta["meta"]["bundle"]["name"], "run/final");
    wait_for(|| log.lock().unwrap().masks == [1]).await;
    let frames = binaries_for(&mut ws, Duration::from_millis(600)).await;
    assert!(frames.len() >= 5, "{} frames", frames.len());
    assert!(
        frames.iter().all(|f| f.len() == 108 && &f[..4] == b"DFLY"),
        "byte for byte what the bot sent"
    );
    let seqs: Vec<u32> = frames
        .iter()
        .map(|f| u32::from_le_bytes(f[12..16].try_into().unwrap()))
        .collect();
    assert!(seqs.windows(2).all(|w| w[0] < w[1]), "in order: {seqs:?}");

    // Stopping stops the demand; the frames stop.
    ws.send(Message::Text(r#"{"type":"fly","hz":0}"#.into())).await.unwrap();
    wait_for(|| log.lock().unwrap().masks == [1, 0]).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(binaries_for(&mut ws, Duration::from_millis(300)).await.is_empty());
    // And again.
    ws.send(Message::Text(r#"{"type":"fly","hz":10}"#.into()))
        .await
        .unwrap();
    wait_for(|| log.lock().unwrap().masks == [1, 0, 1]).await;
    assert_eq!(log.lock().unwrap().junk, 0);
    task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_demand_follows_the_connections_and_ends_when_the_last_one_closes() {
    let (server, log, task, _dir) = bot_server(40, 0, |_| {}).await;
    let mut a = connect(&server).await;
    let mut b = connect(&server).await;
    a.send(Message::Text(r#"{"type":"fly","hz":10}"#.into())).await.unwrap();
    b.send(Message::Text(r#"{"type":"fly","hz":10}"#.into())).await.unwrap();
    wait_for(|| log.lock().unwrap().masks == [1]).await;
    // Both get frames; one leaving does not stop the other's.
    assert!(!binaries_for(&mut a, Duration::from_millis(500)).await.is_empty());
    drop(a);
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(log.lock().unwrap().masks, [1], "one left, one still watches");
    assert!(!binaries_for(&mut b, Duration::from_millis(500)).await.is_empty());
    // Closing the page (no unsubscribe message) ends the demand.
    drop(b);
    wait_for(|| log.lock().unwrap().masks == [1, 0]).await;
    task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_connection_gets_at_most_the_rate_it_asked_for_and_never_more_than_the_cap() {
    let (server, _log, task, _dir) = bot_server(40, 0, |c| c.max_fly_hz = 5.0).await;
    let mut ws = connect(&server).await;
    ws.send(Message::Text(r#"{"type":"fly","hz":1000}"#.into()))
        .await
        .unwrap();
    let _ = next_text(&mut ws).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let n = binaries_for(&mut ws, Duration::from_secs(2)).await.len();
    assert!(
        (6..=12).contains(&n),
        "5 Hz cap over 2 s gave {n} frames from a 40 Hz source"
    );
    ws.send(Message::Text(r#"{"type":"fly","hz":1}"#.into())).await.unwrap();
    tokio::time::sleep(Duration::from_millis(300)).await;
    let n = binaries_for(&mut ws, Duration::from_secs(3)).await.len();
    assert!((2..=5).contains(&n), "1 Hz asked, over 3 s gave {n}");
    task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_rate_a_little_below_the_sources_does_not_halve_the_frames() {
    // A 25 Hz source (40 ms apart) and the default cap of 15 Hz (66 ms): a minimum-gap rule would pass every second
    // frame only (12.5 Hz); an average-rate limiter passes 15 a second.
    let (server, _log, task, _dir) = bot_server(25, 0, |_| {}).await;
    let mut ws = connect(&server).await;
    ws.send(Message::Text(r#"{"type":"fly","hz":15}"#.into()))
        .await
        .unwrap();
    let _ = next_text(&mut ws).await; // the layout: the subscription is in place
    let n = binaries_for(&mut ws, Duration::from_secs(3)).await.len();
    assert!(
        (40..=50).contains(&n),
        "{n} frames in 3 s at 15 Hz from a 25 Hz source (a minimum gap would give ~38)"
    );
    task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn malformed_frames_never_reach_a_browser_and_are_reported_once() {
    let (server, _log, task, _dir) = bot_server(40, 3, |_| {}).await;
    let mut ws = connect(&server).await;
    ws.send(Message::Text(r#"{"type":"fly","hz":50}"#.into()))
        .await
        .unwrap();
    // Everything that arrives in 1.5 s, binary and text.
    let end = Instant::now() + Duration::from_millis(1500);
    let (mut frames, mut reports) = (Vec::new(), 0);
    while let Some(left) = end.checked_duration_since(Instant::now()) {
        match tokio::time::timeout(left, ws.next()).await {
            Ok(Some(Ok(Message::Binary(b)))) => frames.push(b.to_vec()),
            Ok(Some(Ok(Message::Text(t)))) => {
                let v: serde_json::Value = serde_json::from_str(&t).unwrap();
                if v["type"] == "live_error" && v["message"].as_str().unwrap_or("").contains("fly stream is malformed")
                {
                    reports += 1;
                }
            }
            Ok(Some(Ok(_))) => {}
            _ => break,
        }
    }
    assert!(frames.len() >= 10, "{} frames", frames.len());
    assert!(frames.iter().all(|f| f.len() == 108), "the short ones were dropped");
    // One report on the event channel, however many bad frames came (every third of 40 a second).
    assert_eq!(reports, 1);
    task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn without_a_bot_the_layout_is_null_and_odd_rates_are_harmless() {
    let server = TestServer::start().await; // no live hub at all
    let mut ws = connect(&server).await;
    for hz in ["5", "0", "-3", "1e308", "0.0000001"] {
        ws.send(Message::Text(format!(r#"{{"type":"fly","hz":{hz}}}"#).into()))
            .await
            .unwrap();
    }
    let m = next_text(&mut ws).await;
    assert_eq!(
        (m["type"].clone(), m["meta"].clone()),
        ("fly_meta".into(), serde_json::Value::Null)
    );
    // Garbage of the same type does not close the connection.
    ws.send(Message::Text(r#"{"type":"fly","hz":"fast"}"#.into()))
        .await
        .unwrap();
    ws.send(Message::Text(r#"{"type":"ping"}"#.into())).await.unwrap();
    // (each positive rate with no bot to watch is answered with the null layout; the connection lives on)
    loop {
        let m = next_text(&mut ws).await;
        if m["type"] == "pong" {
            break;
        }
        assert_eq!(m["type"], "fly_meta");
    }

    // A bot socket that does not exist: the same, and the page learns the bot is not there.
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("none.sock");
    let server = TestServer::start_with(move |c| c.bot_socket = Some(sock)).await;
    let mut ws = connect(&server).await;
    ws.send(Message::Text(r#"{"type":"fly","hz":5}"#.into())).await.unwrap();
    let m = next_text(&mut ws).await;
    assert_eq!(
        (m["type"].clone(), m["meta"].clone()),
        ("fly_meta".into(), serde_json::Value::Null)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_page_and_its_script_are_served_with_the_site_headers() {
    let server = TestServer::start().await;
    let page = send(server.addr, Req::new("GET", "/"));
    assert_eq!(page.status, 200);
    let html = String::from_utf8(page.body).unwrap();
    assert!(html.contains(r#"id="tab-fly""#) && html.contains(r#"id="fly-view""#) && html.contains("Муха"));
    assert!(html.contains(r#"src="/fly.js""#));
    assert!(!html.contains("style=\""), "the CSP forbids inline styles");
    let js = send(server.addr, Req::new("GET", "/fly.js"));
    assert_eq!(js.status, 200);
    assert!(js.header("content-type").unwrap().starts_with("text/javascript"));
    assert!(
        js.header("content-security-policy")
            .unwrap()
            .contains("script-src 'self'")
    );
    assert_eq!(js.header("x-content-type-options"), Some("nosniff"));
    let src = String::from_utf8(js.body).unwrap();
    assert!(!src.contains(".innerHTML"), "dynamic text is set with textContent only");
    assert!(!src.contains("eval("));
}
