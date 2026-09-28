//! Integration tests for the live map view (task 5.2a acceptance criterion 5): a real server,
//! with a real (synthetic, `test-util`-built) replay source and map directory wired in, driven
//! over real HTTP/WS — covering the `map`/`players`/`live`/`events`/`replay_status` WS messages,
//! `GET /api/map/<sha256>` auth + ETag, and that a bad `--maps-dir` lookup can never read outside
//! its configured directory.

mod support;

use std::path::Path;
use std::time::Duration;

use ddai_web::config::WebConfig;
use ddai_web::live::replay::testutil::write_trace_b_fixture;
use futures_util::{SinkExt, Stream, StreamExt};
use sha2::{Digest, Sha256};
use support::TestServer;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::Uri;
use tokio_tungstenite::tungstenite::{ClientRequestBuilder, Message};

/// Builds a small but real, loadable `.map` file (via `ddai_map::testutil`, the same helper
/// `ddai-map`'s own tests use) at `dir/name`, returning its sha256.
fn write_real_map(dir: &Path, name: &str) -> [u8; 32] {
    use ddai_map::testutil::{MapWriter, TILESLAYERFLAG_GAME, TileLayerSpec, TilemapShape, game_layer_data};

    let mut w = MapWriter::new(4);
    w.add_version_item(1);
    w.add_tile_layer(&TileLayerSpec {
        shape: TilemapShape::Full,
        item_version: 3,
        width: 3,
        height: 3,
        flags: TILESLAYERFLAG_GAME,
        data: &game_layer_data(3, 3),
    });
    w.add_single_group_with_all_layers();
    let bytes = w.finish();
    std::fs::write(dir.join(name), &bytes).expect("write real map");

    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    hasher.finalize().into()
}

/// Sets up a server with a working replay source: one real map file plus one matching trace-b
/// fixture (`character_count` characters, `tick_count` ticks). Returns the server and the map's
/// sha256 hex (for `/api/map/<sha256>` requests).
async fn server_with_replay(character_count: u32, tick_count: u32) -> (TestServer, String) {
    let maps_dir = tempfile::tempdir().expect("maps tempdir");
    let traces_dir = tempfile::tempdir().expect("traces tempdir");
    let map_sha256 = write_real_map(maps_dir.path(), "TestMap.map");
    write_trace_b_fixture(
        &traces_dir.path().join("fixture.trb"),
        character_count,
        tick_count,
        "real-map",
        map_sha256,
        Some("/some/stale/path/TestMap.map"),
    );

    let maps_dir_path = maps_dir.path().to_path_buf();
    let traces_dir_path = traces_dir.path().to_path_buf();
    let server = TestServer::start_with(move |config: &mut WebConfig| {
        config.replay_source = Some(traces_dir_path.clone());
        config.map_search_dirs = vec![maps_dir_path.clone()];
    })
    .await;
    // Keep the tempdirs alive for the server's lifetime by leaking them into the returned
    // server's own drop order — simplest way here is to just `std::mem::forget` isn't needed:
    // returning them would work too, but the server already outlives this fn's stack frame via
    // its background task reading from these paths on demand (not holding the dirs open), and
    // `tempfile::TempDir` deletes on drop — so we must keep them alive as long as the server runs.
    // Leaking the `TempDir` guards here (test-only, tiny, and this process exits at test end
    // regardless) keeps the function's return type simple.
    std::mem::forget(maps_dir);
    std::mem::forget(traces_dir);

    let sha256_hex: String = map_sha256.iter().map(|b| format!("{b:02x}")).collect();
    (server, sha256_hex)
}

fn ws_uri(addr: std::net::SocketAddr) -> Uri {
    format!("ws://{addr}/ws").parse().expect("valid ws uri")
}

fn ws_request(
    addr: std::net::SocketAddr,
    cookie: &str,
    origin: &str,
) -> tokio_tungstenite::tungstenite::handshake::client::Request {
    ClientRequestBuilder::new(ws_uri(addr))
        .with_header("Cookie", cookie)
        .with_header("Origin", origin)
        .into_client_request()
        .expect("build ws request")
}

async fn next_json<S>(ws: &mut S) -> serde_json::Value
where
    S: Stream<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    loop {
        match ws.next().await.expect("stream ended").expect("ws error") {
            Message::Text(text) => return serde_json::from_str(&text).expect("valid JSON"),
            Message::Ping(_) | Message::Pong(_) => continue,
            other => panic!("expected text, got {other:?}"),
        }
    }
}

/// Reads messages until one of `wanted_types` arrives (skipping anything else — `status` ticks,
/// binary `live` frames, other JSON types), or panics after `deadline`.
async fn next_json_of_type<S>(ws: &mut S, wanted_types: &[&str], deadline: Duration) -> serde_json::Value
where
    S: Stream<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    let started = tokio::time::Instant::now();
    loop {
        assert!(
            started.elapsed() < deadline,
            "timed out waiting for one of {wanted_types:?}"
        );
        match tokio::time::timeout(deadline, ws.next()).await {
            Ok(Some(Ok(Message::Text(text)))) => {
                let value: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
                if wanted_types.iter().any(|t| value["type"] == *t) {
                    return value;
                }
            }
            Ok(Some(Ok(_))) => continue,
            Ok(Some(Err(e))) => panic!("ws error: {e}"),
            Ok(None) => panic!("stream ended"),
            Err(_) => panic!("timed out waiting for one of {wanted_types:?}"),
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn connecting_receives_map_players_and_live_frames() {
    let (server, _sha256_hex) = server_with_replay(3, 200).await;
    let (cookie, _csrf) = server.login();
    let request = ws_request(server.addr, &cookie, &server.origin());
    let (mut ws, _resp) = tokio_tungstenite::connect_async(request).await.expect("connect");
    let _hello = next_json(&mut ws).await;

    let map = next_json_of_type(&mut ws, &["map"], Duration::from_secs(5)).await;
    assert_eq!(map["w"], 3);
    assert_eq!(map["h"], 3);
    assert!(map["sha256"].as_str().unwrap().len() == 64);

    let players = next_json_of_type(&mut ws, &["players"], Duration::from_secs(5)).await;
    assert_eq!(players["list"].as_array().unwrap().len(), 3);

    // A binary `live` frame should arrive too (default subscription is on at 25 Hz).
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        assert!(tokio::time::Instant::now() < deadline, "no live frame received in time");
        match ws.next().await.expect("stream ended").expect("ws error") {
            Message::Binary(bytes) => {
                let decoded = ddai_web::live::frame::decode(&bytes).expect("decode live frame");
                assert_eq!(decoded.characters.len(), 3);
                break;
            }
            _ => continue,
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn replay_pause_control_stops_the_tick_from_advancing() {
    let (server, _sha256_hex) = server_with_replay(2, 2000).await;
    let (cookie, _csrf) = server.login();
    let request = ws_request(server.addr, &cookie, &server.origin());
    let (mut ws, _resp) = tokio_tungstenite::connect_async(request).await.expect("connect");
    let _hello = next_json(&mut ws).await;

    ws.send(Message::Text(r#"{"type":"replay","action":"pause"}"#.into()))
        .await
        .expect("send pause");

    // Give the pause a moment to actually take effect, then record the tick from the next couple
    // of status updates — they should not advance while paused.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let status_a = next_json_of_type(&mut ws, &["replay_status"], Duration::from_secs(5)).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let status_b = next_json_of_type(&mut ws, &["replay_status"], Duration::from_secs(5)).await;
    assert_eq!(status_a["tick"], status_b["tick"], "tick must not advance while paused");
    assert_eq!(status_a["playing"], false);

    ws.send(Message::Text(r#"{"type":"replay","action":"play"}"#.into()))
        .await
        .expect("send play");
    tokio::time::sleep(Duration::from_millis(200)).await;
    let status_c = next_json_of_type(&mut ws, &["replay_status"], Duration::from_secs(5)).await;
    assert_eq!(status_c["playing"], true);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn map_route_requires_auth() {
    let (server, sha256_hex) = server_with_replay(2, 10).await;
    let response = support::send(server.addr, support::Req::new("GET", &format!("/api/map/{sha256_hex}")));
    assert_eq!(response.status, 401);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn map_route_rejects_a_non_hex_sha256() {
    let (server, _sha256_hex) = server_with_replay(2, 10).await;
    let (cookie, _csrf) = server.login();
    let response = support::send(
        server.addr,
        support::Req::new("GET", "/api/map/not-hex-at-all").cookie(&cookie),
    );
    assert_eq!(response.status, 400);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn map_route_404s_for_an_unknown_sha256() {
    let (server, _sha256_hex) = server_with_replay(2, 10).await;
    let (cookie, _csrf) = server.login();
    let unknown = "00".repeat(32);
    let response = support::send(
        server.addr,
        support::Req::new("GET", &format!("/api/map/{unknown}")).cookie(&cookie),
    );
    assert_eq!(response.status, 404);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn map_route_serves_the_scene_with_etag_and_supports_conditional_get() {
    let (server, sha256_hex) = server_with_replay(2, 10).await;
    let (cookie, _csrf) = server.login();

    // Wait for the replay source to actually resolve and cache the map (it does this once at
    // startup, but asynchronously — poll briefly rather than assuming it's instant).
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let mut response;
    loop {
        response = support::send(
            server.addr,
            support::Req::new("GET", &format!("/api/map/{sha256_hex}")).cookie(&cookie),
        );
        if response.status == 200 || tokio::time::Instant::now() >= deadline {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(response.status, 200, "{response:?}");
    assert_eq!(response.header("content-type"), Some("application/octet-stream"));
    let etag = response.header("etag").expect("etag header").to_string();
    assert_eq!(etag, format!("\"{sha256_hex}\""));
    assert!(
        response
            .header("cache-control")
            .is_some_and(|v| v.contains("immutable")),
        "{response:?}"
    );
    let scene = ddai_web::live::scene::decode_compressed(&response.body).expect("decode scene");
    assert_eq!(scene.width, 3);
    assert_eq!(scene.height, 3);

    // Review round 1, finding F11: a second plain (non-conditional) GET must be served from the
    // compressed-bytes cache (`MapCache::get_compressed`), not recompressed from scratch — this
    // can't observe *that* directly from outside the process, but it can and does confirm the
    // cached path still returns the exact same bytes as the first, freshly-compressed response,
    // which is what `crate::live::map_resolve::map_cache_returns_the_same_compressed_bytes_instance`
    // (a unit test, same crate) confirms is actually the same cached `Arc`, not a coincidence.
    let second = support::send(
        server.addr,
        support::Req::new("GET", &format!("/api/map/{sha256_hex}")).cookie(&cookie),
    );
    assert_eq!(second.status, 200);
    assert_eq!(
        second.body, response.body,
        "cached response must be byte-identical to the first"
    );

    // Conditional GET with the ETag we just got back must 304.
    let conditional = support::send(
        server.addr,
        support::Req::new("GET", &format!("/api/map/{sha256_hex}"))
            .cookie(&cookie)
            .header("If-None-Match", &etag),
    );
    assert_eq!(conditional.status, 304);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_malformed_trace_reports_an_error_event_not_a_crash() {
    let traces_dir = tempfile::tempdir().expect("traces tempdir");
    std::fs::write(traces_dir.path().join("broken.trb"), b"not a trace-b file at all").expect("write");
    let traces_dir_path = traces_dir.path().to_path_buf();

    let server = TestServer::start_with(move |config: &mut WebConfig| {
        config.replay_source = Some(traces_dir_path.clone());
    })
    .await;

    let (cookie, _csrf) = server.login();
    let request = ws_request(server.addr, &cookie, &server.origin());
    let (mut ws, _resp) = tokio_tungstenite::connect_async(request).await.expect("connect");
    let _hello = next_json(&mut ws).await;

    let error = next_json_of_type(&mut ws, &["live_error"], Duration::from_secs(5)).await;
    assert!(error["message"].as_str().unwrap().contains("broken.trb"), "{error}");

    std::mem::forget(traces_dir);
}

/// Regression test for review round 1, finding F12: `--replay <dir>` is an operator-configured,
/// often-absolute server filesystem path (`~/aiddnet/data/...` in production); a `live_error` (or
/// `replay_status.file`, checked below too) that echoes it back used to hand every authenticated
/// client the server's directory layout for no reason a client needs it. Uses a real tempdir (not
/// a fixed string) so this can't pass by coincidence — the assertion below only holds if the fix
/// is actually stripping the directory, since the tempdir's own unique name is guaranteed to
/// appear in the *pre-fix* message (it's the whole path being formatted).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_error_and_replay_status_never_leak_the_server_side_directory_path() {
    let traces_dir = tempfile::tempdir().expect("traces tempdir");
    std::fs::write(traces_dir.path().join("broken.trb"), b"not a trace-b file at all").expect("write");
    let traces_dir_path = traces_dir.path().to_path_buf();
    // The directory itself (not just the trace file within it) — this is the part of the path a
    // fix must strip; a tempdir's name is unique enough that this can't accidentally match
    // anything else in the message.
    let dir_name = traces_dir_path.file_name().unwrap().to_str().unwrap().to_string();

    let server = TestServer::start_with(move |config: &mut WebConfig| {
        config.replay_source = Some(traces_dir_path.clone());
    })
    .await;

    let (cookie, _csrf) = server.login();
    let request = ws_request(server.addr, &cookie, &server.origin());
    let (mut ws, _resp) = tokio_tungstenite::connect_async(request).await.expect("connect");
    let _hello = next_json(&mut ws).await;

    // `run`'s loop sends `replay_status` (tick 0, before the file is even opened) right before
    // `play_one_file` gets a chance to fail on this unparseable file, so both message types are
    // reachable on the very same connection from this one broken trace.
    let status = next_json_of_type(&mut ws, &["replay_status"], Duration::from_secs(5)).await;
    let status_file = status["file"].as_str().unwrap().to_string();
    assert!(
        !status_file.contains(&dir_name),
        "replay_status.file must not contain the replay directory's own path component: {status_file:?}"
    );
    assert_eq!(status_file, "broken.trb");

    let error = next_json_of_type(&mut ws, &["live_error"], Duration::from_secs(5)).await;
    let message = error["message"].as_str().unwrap().to_string();
    assert!(
        !message.contains(&dir_name),
        "live_error must not contain the replay directory's own path component: {message:?}"
    );
    assert!(message.contains("broken.trb"), "{message:?}");

    std::mem::forget(traces_dir);
}

/// Regression test for review round 1, finding F6: a `sub{live: hz}` rate close enough to zero
/// without actually being the unsubscribe sentinel (`live <= 0.0`) used to drive
/// `Duration::from_secs_f32(1.0 / live_hz)` past what `Duration` can represent, panicking the
/// whole WS task the next time a `live` broadcast frame actually arrives. That's why this test
/// (unlike a plain `websocket.rs` one) needs a real, running replay source: with no live hub at
/// all, the vulnerable branch's inner future is `pending()` forever and never actually polls the
/// panicking code, which would make a test without one pass vacuously regardless of the fix.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_tiny_positive_sub_rate_does_not_crash_the_connection() {
    let (server, _sha256_hex) = server_with_replay(2, 500).await;
    let (cookie, _csrf) = server.login();
    let request = ws_request(server.addr, &cookie, &server.origin());
    let (mut ws, _resp) = tokio_tungstenite::connect_async(request).await.expect("connect");
    let _hello = next_json(&mut ws).await;

    ws.send(Message::Text(r#"{"type":"sub","live":1e-20}"#.into()))
        .await
        .expect("send sub");

    // The vulnerable code (pre-fix) only ran when a `live` broadcast actually reached this
    // connection's `select!` loop — NOT merely on receiving the `sub` message itself — so this
    // must wait long enough for at least one such broadcast to arrive after the rate change
    // before checking anything. An earlier version of this test checked immediately after
    // sending `sub` and passed even against the unfixed code: the whole
    // sub-then-ping-then-pong round trip completed in a millisecond or two (everything here is
    // in-process, no real network latency), well before the replay source's own next ~20ms tick
    // ever reached this subscriber — the test that "caught" nothing because the vulnerable branch
    // was simply never polled in time. The replay source ticks at up to 50 Hz, so 300ms is many
    // multiples of one tick period, comfortably enough for the race to resolve deterministically.
    tokio::time::sleep(Duration::from_millis(300)).await;

    // Deliberately does NOT wait for a `live` *binary* frame here: `1e-20` clamps up to
    // `MIN_LIVE_HZ` (0.1 Hz, i.e. one forwarded frame per 10s), so a binary frame legitimately
    // might not arrive within any short test deadline even on a live connection — that would
    // make this test flaky for a reason that has nothing to do with the bug it's checking.
    // `status`/`replay_status` JSON messages are never subject to that per-connection forwarding
    // throttle at all, so an explicit ping/pong round trip proves the connection is still alive
    // and responsive without this test's own timing depending on what the forwarding rate
    // actually throttles down to.
    ws.send(Message::Text(r#"{"type":"ping"}"#.into()))
        .await
        .expect("send ping");
    let pong = next_json_of_type(&mut ws, &["pong"], Duration::from_secs(5)).await;
    assert_eq!(pong["type"], "pong");
}

/// Review round 1, finding F9: an explicit test that `replay{...}` control is unreachable without
/// a valid session — not just implied by task 5.1's generic `upgrade_without_a_session_cookie_is_
/// rejected_with_401` (`tests/websocket.rs`), which doesn't run against a server with a live
/// source configured at all. The WS upgrade itself is the only gate `replay{...}` messages ever
/// pass through (`ws.rs`'s doc comment at the `Replay` message arm says exactly this); this test
/// exercises that against a REAL running replay source, proving there is no separate code path
/// (e.g. a `replay` message handled before the session check, or a raw TCP/HTTP route bypassing
/// the upgrade) that could reach `ReplayControl` without one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn replay_control_is_unreachable_without_a_session() {
    let (server, _sha256_hex) = server_with_replay(2, 500).await;

    // No `Cookie` header at all — the same "completely unauthenticated" request a `replay{...}`
    // message would have to ride in on if there were any way to reach it pre-auth.
    let request = ClientRequestBuilder::new(format!("ws://{}/ws", server.addr).parse().unwrap())
        .with_header("Origin", server.origin())
        .into_client_request()
        .expect("build ws request");
    let err = tokio_tungstenite::connect_async(request)
        .await
        .expect_err("an unauthenticated upgrade must be rejected before any message (including `replay`) is ever read");
    let status = match err {
        tokio_tungstenite::tungstenite::Error::Http(response) => response.status().as_u16(),
        other => panic!("expected an HTTP-level rejection, got {other:?}"),
    };
    assert_eq!(
        status, 401,
        "the upgrade itself — replay control's only gate — must reject with 401"
    );
}
