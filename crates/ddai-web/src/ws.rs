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
use tokio::sync::broadcast;
use tokio::time::{interval, interval_at};

use crate::auth::session::SessionId;
use crate::live::hub::{FlySubscription, HubEvent, LiveHub};
use crate::live::source::{GameEvent, MapMeta, PlayerMeta, ReplayControl, ReplayStatus, SourceKind};
use crate::session_guard::{current_session, request_is_same_origin};
use crate::state::SharedState;

/// Versioned so a future incompatible change to this JSON protocol can be detected by the
/// client from `hello.version` (acceptance criterion 5: "versioned JSON for now").
pub const PROTOCOL_VERSION: u32 = 1;

/// Lower bound on a client-requested `sub{live: hz}` rate (acceptance criterion 1's "≤ 50 Hz;
/// default 25 Hz, 10 Hz эконом" only ever names an upper bound — review round 1, finding F6: a
/// client-supplied rate close enough to 0 without actually *being* 0 (the unsubscribe sentinel,
/// handled separately below) drove `1.0 / live_hz` up past what `Duration` can represent,
/// panicking the whole WS task on the next `select!` iteration. `0.1` Hz (one frame per 10s) is
/// comfortably impractical-but-safe: the important thing is it keeps `1.0 / live_hz` (and
/// therefore every `Duration` built from it) bounded, not that it's a particularly useful rate to
/// actually request.
const MIN_LIVE_HZ: f32 = 0.1;
/// How often the server sends a `status` message while connected.
const STATUS_INTERVAL: Duration = Duration::from_secs(1);
/// How often the server pings an otherwise-quiet connection.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(15);
/// If nothing at all has been received from the client (app messages, pings, or the pongs a real
/// browser answers our pings with automatically) for this long, the connection is considered
/// dead and closed.
const IDLE_CLOSE_AFTER: Duration = Duration::from_secs(45);

/// A `Map`/`Players`/`Events`/`ReplayStatus`/`Error` message's JSON shape (acceptance criterion
/// 1's WS topics). Wire field names are deliberately terse (`w`/`h`/`t`) to match the task's own
/// sketch (`docs/research/orig-web.md` §7.3) and keep these — sent once per map/roster change, so
/// bandwidth is not the concern `live`'s binary format exists for — still small and obvious.
#[derive(Debug, Serialize)]
struct MapMsg {
    sha256: String,
    name: String,
    w: u32,
    h: u32,
}

impl From<MapMeta> for MapMsg {
    fn from(m: MapMeta) -> Self {
        MapMsg {
            // `ddai_trace::hash::to_hex` (already a dependency for `crate::live::replay`) rather
            // than pulling in a `hex` crate just for this one call site.
            sha256: ddai_trace::hash::to_hex(&m.sha256),
            name: m.name,
            w: m.width,
            h: m.height,
        }
    }
}

#[derive(Debug, Serialize)]
struct PlayerMsg {
    id: u8,
    name: String,
    team: u8,
}

impl From<PlayerMeta> for PlayerMsg {
    fn from(p: PlayerMeta) -> Self {
        PlayerMsg {
            id: p.id,
            name: p.name,
            team: p.team,
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum EventMsg {
    Freeze { id: u8 },
    Unfreeze { id: u8 },
    Death { id: u8 },
    Respawn { id: u8 },
    HookGrab { id: u8, target: Option<u8> },
}

impl From<GameEvent> for EventMsg {
    fn from(e: GameEvent) -> Self {
        match e {
            GameEvent::Freeze { id } => EventMsg::Freeze { id },
            GameEvent::Unfreeze { id } => EventMsg::Unfreeze { id },
            GameEvent::Death { id } => EventMsg::Death { id },
            GameEvent::Respawn { id } => EventMsg::Respawn { id },
            GameEvent::HookGrab { id, target } => EventMsg::HookGrab { id, target },
        }
    }
}

#[derive(Debug, Serialize)]
struct ReplayStatusMsg {
    file: String,
    tick: u32,
    tick_count: u32,
    playing: bool,
    speed: f32,
}

impl From<ReplayStatus> for ReplayStatusMsg {
    fn from(s: ReplayStatus) -> Self {
        ReplayStatusMsg {
            file: s.file,
            tick: s.tick,
            tick_count: s.tick_count,
            playing: s.playing,
            speed: s.speed,
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ServerMessage {
    Hello {
        version: u32,
        server_time: u64,
    },
    Status {
        uptime_s: u64,
        bot_state: &'static str,
    },
    Pong,
    Map(MapMsg),
    Players {
        list: Vec<PlayerMsg>,
    },
    Events {
        tick: u32,
        events: Vec<EventMsg>,
    },
    ReplayStatus(ReplayStatusMsg),
    /// Task 4.1: the live bot's status (`docs/formats.md` §21): target, mode, counters, latency.
    Bot {
        status: serde_json::Value,
    },
    /// Task 7.4: the layout of the fly's visualisation stream (`docs/formats.md` §27.2); `null`: none (the bot's brain
    /// has no stream, or the bot is not there). Sent to a connection that subscribed with `{"type":"fly","hz":N}`, when it
    /// does and whenever the layout changes; the frames follow as binary messages starting with `DFLY`.
    FlyMeta {
        meta: serde_json::Value,
    },
    /// Task 5.7: what the site shows, `kind` `live` / `demo` / `none` (`docs/formats.md` §28). Sent on connect and whenever it
    /// changes; the page then drops what it drew of the previous source. `info` (demo only): `{"arena","bundle"}` or `null`.
    Source {
        kind: &'static str,
        info: serde_json::Value,
    },
    /// A live-source problem (acceptance criterion 2: "a malformed trace gives an error event,
    /// not a panic") — reported to the client, distinct from any HTTP-level error.
    LiveError {
        message: String,
    },
}

fn hub_event_to_server_message(event: &HubEvent) -> ServerMessage {
    match event {
        HubEvent::Map(meta) => ServerMessage::Map(meta.clone().into()),
        HubEvent::Players(players) => ServerMessage::Players {
            list: players.iter().cloned().map(PlayerMsg::from).collect(),
        },
        HubEvent::Events { tick, events } => ServerMessage::Events {
            tick: *tick,
            events: events.iter().copied().map(EventMsg::from).collect(),
        },
        HubEvent::ReplayStatus(status) => ServerMessage::ReplayStatus(status.clone().into()),
        HubEvent::BotStatus(json) => ServerMessage::Bot {
            status: serde_json::from_str(json).unwrap_or(serde_json::Value::Null),
        },
        HubEvent::FlyMeta(meta) => ServerMessage::FlyMeta {
            meta: fly_meta_value(meta.as_deref()),
        },
        HubEvent::Source(kind, info, _) => source_message(*kind, info.as_deref()),
        HubEvent::Error(message) => ServerMessage::LiveError {
            message: message.clone(),
        },
    }
}

fn source_message(kind: SourceKind, info: Option<&str>) -> ServerMessage {
    ServerMessage::Source {
        kind: kind.as_str(),
        info: info
            .and_then(|i| serde_json::from_str(i).ok())
            .unwrap_or(serde_json::Value::Null),
    }
}

fn fly_meta_value(meta: Option<&str>) -> serde_json::Value {
    meta.and_then(|m| serde_json::from_str(m).ok())
        .unwrap_or(serde_json::Value::Null)
}

/// A client `replay{...}` command's JSON shape (acceptance criterion 2: "WS messages
/// `replay{play|pause|speed|seek|next}`").
#[derive(Debug, Deserialize)]
struct ReplayCommand {
    action: String,
    #[serde(default)]
    value: Option<f32>,
    #[serde(default)]
    tick: Option<u32>,
}

impl ReplayCommand {
    /// `None` for an action this build doesn't recognize — ignored, same convention as an
    /// unknown top-level message type (forward-compatible with a newer client).
    fn into_control(self) -> Option<ReplayControl> {
        match self.action.as_str() {
            "play" => Some(ReplayControl::Play),
            "pause" => Some(ReplayControl::Pause),
            "speed" => self.value.map(ReplayControl::SetSpeed),
            "seek" => self.tick.map(ReplayControl::Seek),
            "next" => Some(ReplayControl::Next),
            _ => None,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClientMessage {
    Ping,
    /// `{"type":"sub","live":25}` (acceptance criterion 1: "the client subscribes
    /// (`sub{live: hz}`)"). `live <= 0` unsubscribes (no more `live` binary frames until the next
    /// `sub` with a positive rate).
    Sub {
        live: f32,
    },
    Replay(ReplayCommand),
    /// Task 7.4: `{"type":"fly","hz":8}` watches the fly's stream at up to `hz` frames per second (clamped to
    /// `config.max_fly_hz`); `hz <= 0` stops. The source is told that somebody watches while at least one connection does.
    Fly {
        hz: f32,
    },
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

/// What a connection is told about the state of the hub: the source on show (setting `shown_generation` to its generation, so the
/// frames of that source are let through), then the map and the roster. Sent when the connection starts, and again when it has
/// lagged behind the event channel and may have missed any of it (a missed `source` would otherwise leave every frame dropped).
async fn send_snapshot(socket: &mut WebSocket, hub: &LiveHub, shown_generation: &mut u64) -> Result<(), ()> {
    // What is on show comes first: the page drops what it drew of another source when this changes it.
    if let Some((kind, info, generation)) = hub.latest_source_tagged() {
        *shown_generation = generation;
        send_json(socket, &source_message(kind, info.as_deref())).await?;
    }
    if let Some(map) = hub.latest_map() {
        send_json(socket, &ServerMessage::Map(map.into())).await?;
    }
    let players = hub.latest_players();
    if !players.is_empty() {
        send_json(
            socket,
            &ServerMessage::Players {
                list: players.into_iter().map(PlayerMsg::from).collect(),
            },
        )
        .await?;
    }
    Ok(())
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

    // Task 5.2a: a newly-connecting client is handed the CURRENT map/roster directly, rather
    // than waiting for the next change to be broadcast — see `LiveHub`'s doc comment on why the
    // broadcast channel alone can't guarantee that for a connection that joins between two
    // (rare) map-change events.
    // Task 5.7: a browser is connected: the source is told (the offline demo plays only while somebody is here).
    let _viewer = state.live_hub.as_ref().map(|hub| hub.viewer());
    let mut live_rx = state.live_hub.as_ref().map(|hub| hub.subscribe_live());
    let mut event_rx = state.live_hub.as_ref().map(|hub| hub.subscribe_events());
    // Task 5.7: the source generation of the `source` message this connection sent last. Binary frames of any other generation
    // (the previous source's last ones still queued, or the next one's first before its badge went out) are not forwarded.
    let mut shown_generation: u64 = 0;
    if let Some(hub) = &state.live_hub
        && send_snapshot(&mut socket, hub, &mut shown_generation).await.is_err()
    {
        return;
    }
    // Live frames flow at this rate by default (acceptance criterion 1: "default 25 Hz") until
    // the client sends its own `sub{live: hz}` — clamped to `config.max_live_hz` either way.
    let mut live_hz: f32 = 25.0f32.min(state.config.max_live_hz);
    // So the very first live frame that arrives is sent immediately rather than waiting out a
    // full `1/live_hz` interval — `checked_sub` (not a bare `-`) since `Instant` subtraction
    // panics on underflow, which a process that has been up for well under a second could
    // otherwise hit here.
    let mut last_live_sent = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .unwrap_or_else(Instant::now);

    // Task 7.4: this connection's subscription to the fly stream and its own frame-rate cap.
    let mut fly_sub: Option<FlySubscription> = None;
    let mut fly_hz: f32 = 0.0;
    let mut fly_next_due = Instant::now();

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
            // Task 5.2a, acceptance criterion 1's backpressure requirement: `recv()` on a
            // `broadcast` channel this connection has fallen behind on returns `Lagged` rather
            // than the oldest still-buffered frame, so a slow connection naturally skips ahead to
            // the newest frame instead of this loop ever building its own unbounded queue —
            // `Lagged` is deliberately just `continue`d here, not logged as an error, since it is
            // the intended behavior under load, not a bug. Only polled when subscribed
            // (`live_hz > 0.0`); see `live_rx`'s construction above for why `None` (no live hub
            // configured at all) never fires this branch either.
            live_bytes = async {
                match live_rx.as_mut() {
                    Some(rx) => rx.recv().await,
                    None => std::future::pending().await,
                }
            }, if live_hz > 0.0 => {
                match live_bytes {
                    Ok(frame) if frame.generation != shown_generation => {} // another source's: not under this badge
                    Ok(frame) => {
                        let bytes = frame.bytes;
                        // Per-connection downsampling to this connection's own requested rate —
                        // independent of the hub's own (much higher) production rate and of any
                        // other connection's rate. `live_hz` is already bounded to
                        // `[MIN_LIVE_HZ, max_live_hz]` by the `Sub` handler below (finding F6), so
                        // `try_from_secs_f32` should never actually hit its `Err` arm here — kept
                        // as `try_from` (not the panicking `from_secs_f32`) anyway, on the same
                        // "audit every float-to-duration conversion" principle the finding asked
                        // for: a *future* change to that bound must not turn back into a panic.
                        let min_interval = Duration::try_from_secs_f32((1.0 / live_hz).max(0.0))
                            .unwrap_or(Duration::from_secs_f32(1.0 / 25.0));
                        if last_live_sent.elapsed() >= min_interval {
                            last_live_sent = Instant::now();
                            if socket.send(Message::Binary(Bytes::from((*bytes).clone()))).await.is_err() {
                                break;
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {} // see the comment above
                    Err(broadcast::error::RecvError::Closed) => {} // hub shut down; nothing to do
                }
            }
            // Task 7.4: the fly's binary frames (`DFLY`), lossy and throttled per connection like `live`.
            fly_bytes = async {
                match fly_sub.as_mut() {
                    Some(sub) => sub.rx.recv().await,
                    None => std::future::pending().await,
                }
            } => {
                match fly_bytes {
                    Ok(frame) if frame.generation != shown_generation => {} // another source's: not under this badge
                    Ok(frame) => {
                        let bytes = frame.bytes;
                        // An average-rate limiter, not a minimum gap: a source at 12.5 Hz asked for 12 must not lose
                        // every other frame to gaps of 80 ms against 83 ms. The schedule advances by one period per
                        // frame sent (and restarts from now after a pause), so the long-run rate is the requested one.
                        let period = Duration::try_from_secs_f32((1.0 / fly_hz.max(MIN_LIVE_HZ)).max(0.0))
                            .unwrap_or(Duration::from_millis(100));
                        let now = Instant::now();
                        if now >= fly_next_due {
                            fly_next_due = if now.duration_since(fly_next_due) > period {
                                now + period
                            } else {
                                fly_next_due + period
                            };
                            if socket.send(Message::Binary(Bytes::from((*bytes).clone()))).await.is_err() {
                                break;
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {} // skip ahead, like `live`
                    Err(broadcast::error::RecvError::Closed) => fly_sub = None,
                }
            }
            event = async {
                match event_rx.as_mut() {
                    Some(rx) => rx.recv().await,
                    None => std::future::pending().await,
                }
            } => {
                match event {
                    Ok(event) => {
                        // The layout of the fly's stream is for the connections that watch it.
                        if matches!(&*event, HubEvent::FlyMeta(_)) && fly_sub.is_none() {
                            continue;
                        }
                        if let HubEvent::Source(_, _, generation) = &*event {
                            shown_generation = *generation;
                        }
                        let message = hub_event_to_server_message(&event);
                        if send_json(&mut socket, &message).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        // Events were lost. Rare, and mostly harmless (see `LiveHub`'s doc comment), except that a lost `source`
                        // would leave `shown_generation` stale and every frame dropped: say again what is on show, the map and
                        // the roster (idempotent for the page), and the fly's layout to a connection that watches the fly.
                        if let Some(hub) = &state.live_hub {
                            if send_snapshot(&mut socket, hub, &mut shown_generation).await.is_err() {
                                break;
                            }
                            if fly_sub.is_some() {
                                let message = ServerMessage::FlyMeta {
                                    meta: fly_meta_value(hub.latest_fly_meta().as_deref()),
                                };
                                if send_json(&mut socket, &message).await.is_err() {
                                    break;
                                }
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => {}
                }
            }
            incoming = socket.recv() => {
                match incoming {
                    None => break,
                    Some(Err(_)) => break,
                    Some(Ok(Message::Close(_))) => break,
                    Some(Ok(Message::Text(text))) => {
                        last_activity = Instant::now();
                        match serde_json::from_str::<ClientMessage>(&text) {
                            Ok(ClientMessage::Ping) => {
                                if send_json(&mut socket, &ServerMessage::Pong).await.is_err() {
                                    break;
                                }
                            }
                            Ok(ClientMessage::Sub { live }) => {
                                // Review round 1, finding F6: `live <= 0.0` (including NaN, which
                                // every comparison against is `false`) unsubscribes; anything
                                // else is clamped into `[MIN_LIVE_HZ, max_live_hz]` — never a
                                // tiny-but-positive value that would later blow up
                                // `Duration::from_secs_f32(1.0 / live_hz)`.
                                live_hz = if live > 0.0 {
                                    live.clamp(MIN_LIVE_HZ, state.config.max_live_hz)
                                } else {
                                    0.0
                                };
                            }
                            Ok(ClientMessage::Fly { hz }) => {
                                // Same rule as `sub{live}`: NaN and anything <= 0 stop; a positive rate is clamped.
                                if hz > 0.0 {
                                    fly_hz = hz.clamp(MIN_LIVE_HZ, state.config.max_fly_hz);
                                    if fly_sub.is_none() {
                                        // Subscribed (the source is told) before the layout is read, so a layout
                                        // announced in between reaches this connection through the event channel.
                                        fly_sub = state.live_hub.as_ref().map(|hub| hub.subscribe_fly());
                                        let meta = state.live_hub.as_ref().and_then(|hub| hub.latest_fly_meta());
                                        let message = ServerMessage::FlyMeta {
                                            meta: fly_meta_value(meta.as_deref()),
                                        };
                                        if send_json(&mut socket, &message).await.is_err() {
                                            break;
                                        }
                                    }
                                } else {
                                    fly_hz = 0.0;
                                    fly_sub = None;
                                }
                            }
                            Ok(ClientMessage::Replay(command)) => {
                                // Acceptance criterion 2: "allowed only for authenticated
                                // sessions, like everything else on the WS" — already guaranteed
                                // here, since this whole connection only exists because the
                                // upgrade above required a valid session; there is no separate
                                // per-message check to add.
                                if let (Some(hub), Some(control)) = (&state.live_hub, command.into_control()) {
                                    hub.send_control(control);
                                }
                            }
                            Err(_) => {
                                // Any other/unknown message type is ignored (forward-compatible:
                                // a newer client may send message types this build doesn't know
                                // about yet).
                            }
                        }
                    }
                    Some(Ok(Message::Binary(_))) => {
                        // The client never sends binary frames in this protocol; ignored.
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
