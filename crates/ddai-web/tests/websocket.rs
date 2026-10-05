//! `GET /ws` integration tests against a real server, using `tokio-tungstenite` as the client
//! (acceptance criteria 4, 5).

mod support;

use futures_util::{SinkExt, Stream, StreamExt};
use support::TestServer;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::Uri;
use tokio_tungstenite::tungstenite::{ClientRequestBuilder, Error as WsError, Message};

fn ws_uri(addr: std::net::SocketAddr) -> Uri {
    format!("ws://{addr}/ws").parse().expect("valid ws uri")
}

fn request_with(
    addr: std::net::SocketAddr,
    cookie: Option<&str>,
    origin: Option<&str>,
) -> tokio_tungstenite::tungstenite::handshake::client::Request {
    let mut builder = ClientRequestBuilder::new(ws_uri(addr));
    if let Some(cookie) = cookie {
        builder = builder.with_header("Cookie", cookie);
    }
    if let Some(origin) = origin {
        builder = builder.with_header("Origin", origin);
    }
    builder.into_client_request().expect("build ws request")
}

fn http_status_of(err: &WsError) -> Option<u16> {
    match err {
        WsError::Http(response) => Some(response.status().as_u16()),
        _ => None,
    }
}

async fn next_json<S>(ws: &mut S) -> serde_json::Value
where
    S: Stream<Item = Result<Message, WsError>> + Unpin,
{
    loop {
        let message = ws
            .next()
            .await
            .expect("stream ended unexpectedly")
            .expect("websocket error");
        match message {
            Message::Text(text) => return serde_json::from_str(&text).expect("valid JSON message"),
            Message::Ping(_) | Message::Pong(_) => continue,
            other => panic!("unexpected message: {other:?}"),
        }
    }
}

/// Like [`next_json`], but also skips any `status` message it sees along the way — the 1 Hz
/// status ticker can race with anything else the test is waiting for (its first tick fires
/// immediately on connect, see `ws.rs`), so tests that care about a specific message type other
/// than `status` should use this instead of assuming a fixed message order.
async fn next_json_skipping_status<S>(ws: &mut S) -> serde_json::Value
where
    S: Stream<Item = Result<Message, WsError>> + Unpin,
{
    loop {
        let value = next_json(ws).await;
        if value["type"] != "status" {
            return value;
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn upgrade_without_a_session_cookie_is_rejected_with_401() {
    let server = TestServer::start().await;
    let request = request_with(server.addr, None, Some(&server.origin()));
    let err = tokio_tungstenite::connect_async(request)
        .await
        .expect_err("should be rejected");
    assert_eq!(http_status_of(&err), Some(401), "{err:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn upgrade_with_wrong_origin_is_rejected_with_403_even_with_a_valid_cookie() {
    let server = TestServer::start().await;
    let (cookie, _csrf) = server.login();
    let request = request_with(server.addr, Some(&cookie), Some("http://evil.example"));
    let err = tokio_tungstenite::connect_async(request)
        .await
        .expect_err("should be rejected");
    assert_eq!(http_status_of(&err), Some(403), "{err:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn upgrade_with_missing_origin_is_rejected_with_403() {
    let server = TestServer::start().await;
    let (cookie, _csrf) = server.login();
    let request = request_with(server.addr, Some(&cookie), None);
    let err = tokio_tungstenite::connect_async(request)
        .await
        .expect_err("should be rejected");
    assert_eq!(http_status_of(&err), Some(403), "{err:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn authenticated_upgrade_succeeds_and_sends_hello_then_status() {
    let server = TestServer::start().await;
    let (cookie, _csrf) = server.login();
    let request = request_with(server.addr, Some(&cookie), Some(&server.origin()));
    let (mut ws, response) = tokio_tungstenite::connect_async(request)
        .await
        .expect("upgrade should succeed");
    assert_eq!(response.status().as_u16(), 101);

    let hello = next_json(&mut ws).await;
    assert_eq!(hello["type"], "hello");
    assert_eq!(hello["version"], ddai_web::ws::PROTOCOL_VERSION);
    assert!(hello["server_time"].as_u64().is_some());

    let status = next_json(&mut ws).await;
    assert_eq!(status["type"], "status");
    // a site with no bridge has no bot (task 5.11: the word comes from the live bridge, it used to be a constant "idle")
    assert_eq!(status["bot_state"], "stopped");
    assert!(status["uptime_s"].as_u64().is_some());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn client_ping_receives_server_pong() {
    let server = TestServer::start().await;
    let (cookie, _csrf) = server.login();
    let request = request_with(server.addr, Some(&cookie), Some(&server.origin()));
    let (mut ws, _response) = tokio_tungstenite::connect_async(request)
        .await
        .expect("upgrade should succeed");
    let _hello = next_json(&mut ws).await;

    ws.send(Message::Text(r#"{"type":"ping"}"#.into()))
        .await
        .expect("send ping");
    let pong = next_json_skipping_status(&mut ws).await;
    assert_eq!(pong["type"], "pong");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_client_message_is_ignored_not_fatal() {
    let server = TestServer::start().await;
    let (cookie, _csrf) = server.login();
    let request = request_with(server.addr, Some(&cookie), Some(&server.origin()));
    let (mut ws, _response) = tokio_tungstenite::connect_async(request)
        .await
        .expect("upgrade should succeed");
    let _hello = next_json(&mut ws).await;

    ws.send(Message::Text(r#"{"type":"something_unknown"}"#.into()))
        .await
        .expect("send unknown message");
    // The connection should stay alive and keep sending status ticks.
    let status = next_json(&mut ws).await;
    assert_eq!(status["type"], "status");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn oversized_message_closes_the_connection() {
    let server = TestServer::start_with(|c| c.max_ws_message_bytes = 64).await;
    let (cookie, _csrf) = server.login();
    let request = request_with(server.addr, Some(&cookie), Some(&server.origin()));
    let (mut ws, _response) = tokio_tungstenite::connect_async(request)
        .await
        .expect("upgrade should succeed");
    let _hello = next_json(&mut ws).await;

    let oversized = "x".repeat(1024);
    // Sending a frame larger than the server's configured max must not silently succeed. Skip
    // over any interleaved `status`/ping frames the server may still emit before it notices and
    // closes the connection.
    let _ = ws.send(Message::Text(oversized.into())).await;
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        assert!(
            tokio::time::Instant::now() < deadline,
            "connection never closed after an oversized message"
        );
        match ws.next().await {
            None => break,         // connection closed
            Some(Err(_)) => break, // protocol error surfaced
            Some(Ok(Message::Close(_))) => break,
            Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
            Some(Ok(Message::Text(text))) => {
                let value: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();
                assert_ne!(value["type"], "hello", "should not get a second hello");
                continue; // a `status` tick that raced with the close is fine
            }
            Some(Ok(other)) => panic!("expected the connection to close, got {other:?}"),
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_ws_connections_per_session_are_capped() {
    let server = TestServer::start_with(|c| c.max_ws_per_session = 1).await;
    let (cookie, _csrf) = server.login();

    let first_request = request_with(server.addr, Some(&cookie), Some(&server.origin()));
    let (_first_ws, _resp) = tokio_tungstenite::connect_async(first_request)
        .await
        .expect("first connection should succeed");

    let second_request = request_with(server.addr, Some(&cookie), Some(&server.origin()));
    let err = tokio_tungstenite::connect_async(second_request)
        .await
        .expect_err("second connection should be refused");
    assert_eq!(http_status_of(&err), Some(429), "{err:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_a_connection_frees_its_slot() {
    let server = TestServer::start_with(|c| c.max_ws_per_session = 1).await;
    let (cookie, _csrf) = server.login();

    let first_request = request_with(server.addr, Some(&cookie), Some(&server.origin()));
    let (first_ws, _resp) = tokio_tungstenite::connect_async(first_request)
        .await
        .expect("first connection should succeed");
    drop(first_ws);

    // Give the server a moment to observe the TCP close and release the slot.
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let second_request = request_with(server.addr, Some(&cookie), Some(&server.origin()));
    let result = tokio_tungstenite::connect_async(second_request).await;
    assert!(result.is_ok(), "slot should have been freed: {result:?}");
}

/// Reads messages until the connection closes (`Close`, an error, or the stream ending), skipping
/// over any interleaved `status`/ping frames along the way, or panics if `deadline` passes first.
/// Review finding F2's whole point: a WebSocket must not stay open past its session's lifetime.
async fn expect_close_within<S>(ws: &mut S, deadline: std::time::Duration, context: &str)
where
    S: Stream<Item = Result<Message, WsError>> + Unpin,
{
    let started = tokio::time::Instant::now();
    loop {
        assert!(
            started.elapsed() < deadline,
            "{context}: connection did not close within {deadline:?}"
        );
        let remaining = deadline.saturating_sub(started.elapsed());
        match tokio::time::timeout(remaining, ws.next()).await {
            Err(_) => panic!("{context}: connection did not close within {deadline:?}"),
            Ok(None) => return,
            Ok(Some(Err(_))) => return,
            Ok(Some(Ok(Message::Close(_)))) => return,
            Ok(Some(Ok(_))) => continue, // status tick / ping / etc. — keep waiting for the close
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn websocket_closes_promptly_after_logout() {
    let server = TestServer::start().await;
    let (cookie, csrf) = server.login();

    let request = request_with(server.addr, Some(&cookie), Some(&server.origin()));
    let (mut ws, _resp) = tokio_tungstenite::connect_async(request)
        .await
        .expect("upgrade should succeed");
    let _hello = next_json(&mut ws).await;

    // Log out via a plain HTTP call while the WebSocket stays open — this must reach the
    // connection almost immediately via the invalidation broadcast (see `state.rs`,
    // `ws.rs`), not just eventually via the once-a-second backstop check.
    let logout_response = support::send(
        server.addr,
        support::Req::new("POST", "/api/logout")
            .header("Origin", &server.origin())
            .header("X-CSRF-Token", &csrf)
            .cookie(&cookie),
    );
    assert_eq!(logout_response.status, 200, "{logout_response:?}");

    expect_close_within(&mut ws, std::time::Duration::from_secs(2), "after logout").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn websocket_closes_after_session_idle_expiry() {
    let server = TestServer::start_with(|c| {
        c.idle_timeout = std::time::Duration::from_millis(200);
    })
    .await;
    let (cookie, _csrf) = server.login();

    let request = request_with(server.addr, Some(&cookie), Some(&server.origin()));
    let (mut ws, _resp) = tokio_tungstenite::connect_async(request)
        .await
        .expect("upgrade should succeed");
    let _hello = next_json(&mut ws).await;

    // No logout event fires for plain time-based expiry — this exercises the per-status-tick
    // backstop check instead (STATUS_INTERVAL is 1s, so this must close within a few ticks of
    // the 200ms idle timeout elapsing).
    expect_close_within(&mut ws, std::time::Duration::from_secs(5), "after idle expiry").await;
}
