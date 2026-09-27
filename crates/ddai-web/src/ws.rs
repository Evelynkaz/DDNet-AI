//! `GET /ws` (acceptance criterion 5): authenticated WebSocket carrying small, versioned JSON
//! control messages. Binary frames are reserved for the live game view (a later task); this
//! skeleton only ever proves the auth/WS path itself works.

use axum::body::Bytes;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum_extra::extract::cookie::CookieJar;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::time::{interval, interval_at};

use crate::auth::session::SessionId;
use crate::session_guard::{current_session, request_is_same_origin};
use crate::state::SharedState;

/// Versioned so a future incompatible change to this JSON protocol can be detected by the
/// client from `hello.version` (acceptance criterion 5: "versioned JSON for now").
pub const PROTOCOL_VERSION: u32 = 1;

/// How often the server sends a `status` message while connected.
const STATUS_INTERVAL: Duration = Duration::from_secs(1);
/// How often the server pings an otherwise-quiet connection.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(15);
/// If nothing at all has been received from the client (app messages, pings, or the pongs a real
/// browser answers our pings with automatically) for this long, the connection is considered
/// dead and closed.
const IDLE_CLOSE_AFTER: Duration = Duration::from_secs(45);

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ServerMessage {
    Hello { version: u32, server_time: u64 },
    Status { uptime_s: u64, bot_state: &'static str },
    Pong,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClientMessage {
    Ping,
}

fn unix_time_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub async fn ws_handler(
    State(state): State<SharedState>,
    headers: HeaderMap,
    jar: CookieJar,
    ws: WebSocketUpgrade,
) -> Response {
    if !request_is_same_origin(&headers) {
        return (StatusCode::FORBIDDEN, "cross-origin request").into_response();
    }
    let Some((session_id, _info)) = current_session(&state, &jar) else {
        return (StatusCode::UNAUTHORIZED, "unauthenticated").into_response();
    };
    if !state.try_reserve_ws_slot(session_id) {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            "too many concurrent websocket connections for this session",
        )
            .into_response();
    }

    let max_size = state.config.max_ws_message_bytes;
    ws.max_message_size(max_size)
        .max_frame_size(max_size)
        .on_upgrade(move |socket| handle_socket(socket, state, session_id))
}

struct WsSlotGuard {
    state: SharedState,
    session_id: SessionId,
}

impl Drop for WsSlotGuard {
    fn drop(&mut self) {
        self.state.release_ws_slot(self.session_id);
    }
}

async fn send_json(socket: &mut WebSocket, message: &ServerMessage) -> Result<(), ()> {
    let text = serde_json::to_string(message).expect("ServerMessage always serializes to JSON");
    socket.send(Message::Text(text.into())).await.map_err(|_| ())
}

async fn handle_socket(mut socket: WebSocket, state: SharedState, session_id: SessionId) {
    let _slot_guard = WsSlotGuard {
        state: state.clone(),
        session_id,
    };
    // Review finding F2: without this, a WebSocket opened before a logout (or before the
    // session's idle/absolute timeout elapses) just kept running — the session was only ever
    // checked once, at upgrade time. Subscribing here means a logout (which publishes on this
    // channel) closes the socket within one `recv()` wakeup instead of staying open forever; the
    // per-status-tick `is_valid` check below is the backstop for plain time-based expiry, which
    // has no discrete event to subscribe to.
    let mut invalidated_rx = state.session_invalidated.subscribe();

    let hello = ServerMessage::Hello {
        version: PROTOCOL_VERSION,
        server_time: unix_time_secs(),
    };
    if send_json(&mut socket, &hello).await.is_err() {
        return;
    }

    let mut status_ticker = interval(STATUS_INTERVAL);
    // `interval()`'s first tick fires immediately, which is what we want for `status_ticker`
    // (the client gets a status right after `hello`, without waiting a full second) but not for
    // the heartbeat: its whole point is to notice a *quiet* connection, so the first ping should
    // only fire after a full `HEARTBEAT_INTERVAL` of silence, not the instant the connection
    // opens.
    let mut heartbeat_ticker = interval_at(tokio::time::Instant::now() + HEARTBEAT_INTERVAL, HEARTBEAT_INTERVAL);
    let mut last_activity = Instant::now();

    loop {
        tokio::select! {
            invalidated = invalidated_rx.recv() => {
                match invalidated {
                    Ok(id) if id == session_id => {
                        let _ = socket.send(Message::Close(None)).await;
                        break;
                    }
                    Ok(_) => {} // a different session logged out; irrelevant to us
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                        // We may have missed our own invalidation notice while lagging behind a
                        // burst of others. Fall back to an active check rather than assuming.
                        if !state.sessions.is_valid(&session_id) {
                            let _ = socket.send(Message::Close(None)).await;
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        // Sender only drops with the whole AppState; the process is shutting
                        // down anyway. Nothing to do — the connection will end with it.
                    }
                }
            }
            _ = status_ticker.tick() => {
                // Backstop for plain time-based expiry (idle/absolute timeout), which — unlike a
                // logout — has no discrete event to subscribe to. Deliberately `is_valid`, not
                // `touch`: an open socket sitting idle must not itself keep extending the idle
                // timeout (see `SessionStore::is_valid`'s doc comment).
                if !state.sessions.is_valid(&session_id) {
                    let _ = socket.send(Message::Close(None)).await;
                    break;
                }
                let status = ServerMessage::Status {
                    uptime_s: state.uptime_secs(),
                    bot_state: "idle",
                };
                if send_json(&mut socket, &status).await.is_err() {
                    break;
                }
            }
            _ = heartbeat_ticker.tick() => {
                if last_activity.elapsed() > IDLE_CLOSE_AFTER {
                    let _ = socket.send(Message::Close(None)).await;
                    break;
                }
                if socket.send(Message::Ping(Bytes::new())).await.is_err() {
                    break;
                }
            }
            incoming = socket.recv() => {
                match incoming {
                    None => break,
                    Some(Err(_)) => break,
                    Some(Ok(Message::Close(_))) => break,
                    Some(Ok(Message::Text(text))) => {
                        last_activity = Instant::now();
                        if let Ok(ClientMessage::Ping) = serde_json::from_str::<ClientMessage>(&text)
                            && send_json(&mut socket, &ServerMessage::Pong).await.is_err()
                        {
                            break;
                        }
                        // Any other/unknown message type is ignored (forward-compatible: a newer
                        // client may send message types this build doesn't know about yet).
                    }
                    Some(Ok(Message::Binary(_))) => {
                        // Reserved for the live game view (later task); ignored for now.
                        last_activity = Instant::now();
                    }
                    Some(Ok(Message::Ping(_) | Message::Pong(_))) => {
                        last_activity = Instant::now();
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_serializes_with_type_tag() {
        let json = serde_json::to_value(ServerMessage::Hello {
            version: 1,
            server_time: 42,
        })
        .unwrap();
        assert_eq!(json["type"], "hello");
        assert_eq!(json["version"], 1);
        assert_eq!(json["server_time"], 42);
    }

    #[test]
    fn status_serializes_with_type_tag() {
        let json = serde_json::to_value(ServerMessage::Status {
            uptime_s: 7,
            bot_state: "idle",
        })
        .unwrap();
        assert_eq!(json["type"], "status");
        assert_eq!(json["uptime_s"], 7);
        assert_eq!(json["bot_state"], "idle");
    }

    #[test]
    fn client_ping_deserializes() {
        let msg: ClientMessage = serde_json::from_str(r#"{"type":"ping"}"#).unwrap();
        assert!(matches!(msg, ClientMessage::Ping));
    }

    #[test]
    fn unknown_client_message_type_fails_to_deserialize() {
        assert!(serde_json::from_str::<ClientMessage>(r#"{"type":"nonsense"}"#).is_err());
    }
}
