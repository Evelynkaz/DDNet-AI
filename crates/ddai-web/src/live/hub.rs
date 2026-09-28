//! The live-view hub: owns the one running [`FrameSource`], turns its [`SourceEvent`]s into two
//! fan-out channels every WebSocket connection can subscribe to, and forwards client `replay{...}`
//! commands back to it.
//!
//! **Two channels, not one, and why.** `live` frames (25 Hz by default) are exactly the thing
//! acceptance criterion 1's backpressure requirement is about: "if the client's send queue grows
//! beyond N frames, drop frames rather than buffering unboundedly." A [`tokio::sync::broadcast`]
//! channel already has precisely this behavior built in — a slow subscriber that doesn't call
//! `recv()` often enough gets `Lagged(n)` and resumes from the oldest still-buffered message, so
//! `ws.rs`'s per-connection task can just treat a `Lagged` as "skip ahead, don't panic, don't
//! block anyone else." But that same drop-on-lag behavior would be a correctness bug for the
//! *other* kind of message this hub emits — a `map`/`players` change is not "the next one will
//! do", a client that missed one is looking at stale data until the next such (rare) event. So:
//! - [`LiveHub::subscribe_live`]: pre-encoded binary frame bytes, lossy under load (by design).
//! - [`LiveHub::subscribe_events`]: `map`/`players`/`events`/`replay-status`/`error`, on a
//!   separate, larger-capacity channel. A pathologically slow connection could in principle still
//!   lag far enough to drop one of these — mitigated (not eliminated) by throttling the one
//!   highest-frequency member of this set ([`SourceEvent::ReplayStatus`], see
//!   `STATUS_THROTTLE`) and by [`LiveHub::latest_map`]/[`LiveHub::latest_players`] handing a
//!   freshly-subscribing connection the current snapshot directly, so it is never left with NO
//!   map/players at all even if it joined between two broadcasts. A full solution (a durable
//!   per-connection queue with real flow control) is a larger feature than this task's scope; see
//!   `crates/ddai-web/README.md`'s "known limitations" for this trade-off spelled out for a
//!   reviewer.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::{broadcast, mpsc};

use super::map_resolve::MapCache;
use super::source::{FrameSource, GameEvent, MapMeta, PlayerMeta, ReplayControl, ReplayStatus, SourceEvent};

/// Capacity of the `mpsc` channel a [`FrameSource`] sends [`SourceEvent`]s into. Small: the hub's
/// pump task drains it essentially immediately (its own work per event is cheap — encode a frame,
/// or a `Vec` clone), so this only needs to smooth over scheduling jitter, not act as a real
/// buffer.
const SOURCE_EVENTS_CHANNEL_CAPACITY: usize = 32;
/// Capacity of the `mpsc` channel client `replay{...}` commands are forwarded through.
const CONTROL_CHANNEL_CAPACITY: usize = 16;
/// Capacity of the lossy `live` binary-frame broadcast channel (acceptance criterion 1's "N" —
/// see this module's doc comment). At the default 25 Hz this is a little over 2.5 seconds of
/// frames before a lagging connection starts skipping ahead — long enough to absorb an ordinary
/// GC-style pause, short enough that "dropping" actually means something under sustained load.
const LIVE_FRAME_BROADCAST_CAPACITY: usize = 64;
/// Capacity of the `map`/`players`/`events`/`replay-status`/`error` broadcast channel — much
/// larger than the live-frame one because these are comparatively rare (a map change roughly
/// once a minute, a roster change on connect, an events batch on a small fraction of ticks).
const EVENT_BROADCAST_CAPACITY: usize = 512;
/// Minimum real time between two [`SourceEvent::ReplayStatus`] updates actually broadcast — a
/// source may report one every tick (this task's [`super::replay::ReplaySource`] does, so the
/// client's tick counter stays live), but the client only needs this a few times a second, and
/// keeping this channel's volume low is itself part of the mitigation this module's doc comment
/// describes for the "rare but must-not-drop" messages sharing the same channel.
const STATUS_THROTTLE: Duration = Duration::from_millis(400);

/// A `map`/`players`/`events`/`replay-status`/`error` update, fanned out on
/// [`LiveHub::subscribe_events`]. `ws.rs` converts each variant into its own tagged JSON `t`
/// message — this type deliberately carries no serde attributes of its own, keeping the wire
/// protocol's tags owned entirely by `ws.rs`.
#[derive(Debug, Clone)]
pub enum HubEvent {
    Map(MapMeta),
    Players(Vec<PlayerMeta>),
    Events { tick: u32, events: Vec<GameEvent> },
    ReplayStatus(ReplayStatus),
    Error(String),
}

#[derive(Default)]
struct Latest {
    map: Option<MapMeta>,
    players: Vec<PlayerMeta>,
}

pub struct LiveHub {
    live_tx: broadcast::Sender<Arc<Vec<u8>>>,
    event_tx: broadcast::Sender<Arc<HubEvent>>,
    latest: Arc<Mutex<Latest>>,
    control_tx: mpsc::Sender<ReplayControl>,
    /// Shared with whatever `FrameSource` this hub was started with (e.g.
    /// `crate::live::replay::ReplaySource` fills it in as it resolves each trace's map) — `GET
    /// /api/map/<sha256>` (`crate::http::map`) reads from this same cache.
    pub map_cache: Arc<MapCache>,
    // Kept alive for as long as the hub is: dropping either would stop the source/pump task by
    // closing the channel the other side is reading from.
    _source_task: tokio::task::JoinHandle<()>,
    _pump_task: tokio::task::JoinHandle<()>,
}

impl LiveHub {
    /// Spawns `source`'s background task and this hub's own event-pump task. Must be called from
    /// within a running Tokio runtime (both `bind()`, the real entry point, and every test that
    /// constructs one run inside `#[tokio::main]`/`#[tokio::test]`). `map_cache` should be the
    /// same instance `source` itself was built with (see `crate::live::replay::ReplaySource::new`)
    /// so `GET /api/map/<sha256>` can serve scenes the source has already resolved.
    pub fn start(source: Box<dyn FrameSource>, map_cache: Arc<MapCache>) -> Self {
        let (events_tx, events_rx) = mpsc::channel(SOURCE_EVENTS_CHANNEL_CAPACITY);
        let (control_tx, control_rx) = mpsc::channel(CONTROL_CHANNEL_CAPACITY);
        let source_task = source.spawn(events_tx, control_rx);

        let (live_tx, _) = broadcast::channel(LIVE_FRAME_BROADCAST_CAPACITY);
        let (event_tx, _) = broadcast::channel(EVENT_BROADCAST_CAPACITY);
        let latest = Arc::new(Mutex::new(Latest::default()));

        let pump_task = tokio::spawn(Self::pump(events_rx, live_tx.clone(), event_tx.clone(), latest.clone()));

        LiveHub {
            live_tx,
            event_tx,
            latest,
            control_tx,
            map_cache,
            _source_task: source_task,
            _pump_task: pump_task,
        }
    }

    async fn pump(
        mut events_rx: mpsc::Receiver<SourceEvent>,
        live_tx: broadcast::Sender<Arc<Vec<u8>>>,
        event_tx: broadcast::Sender<Arc<HubEvent>>,
        latest: Arc<Mutex<Latest>>,
    ) {
        // Not `Instant::now()`: the very first status update should always go out immediately,
        // not wait a full `STATUS_THROTTLE` after the hub starts.
        let mut last_status_sent = Instant::now().checked_sub(STATUS_THROTTLE).unwrap_or_else(Instant::now);
        while let Some(event) = events_rx.recv().await {
            match event {
                SourceEvent::MapChanged(meta) => {
                    latest.lock().expect("live hub mutex poisoned").map = Some(meta.clone());
                    let _ = event_tx.send(Arc::new(HubEvent::Map(meta)));
                }
                SourceEvent::Players(players) => {
                    latest.lock().expect("live hub mutex poisoned").players = players.clone();
                    let _ = event_tx.send(Arc::new(HubEvent::Players(players)));
                }
                SourceEvent::Frame(frame) => {
                    let bytes = super::frame::encode(&frame);
                    let _ = live_tx.send(Arc::new(bytes));
                }
                SourceEvent::Events { tick, events } => {
                    if !events.is_empty() {
                        let _ = event_tx.send(Arc::new(HubEvent::Events { tick, events }));
                    }
                }
                SourceEvent::ReplayStatus(status) => {
                    let now = Instant::now();
                    if now.duration_since(last_status_sent) >= STATUS_THROTTLE {
                        last_status_sent = now;
                        let _ = event_tx.send(Arc::new(HubEvent::ReplayStatus(status)));
                    }
                }
                SourceEvent::Error(message) => {
                    let _ = event_tx.send(Arc::new(HubEvent::Error(message)));
                }
            }
        }
    }

    pub fn subscribe_live(&self) -> broadcast::Receiver<Arc<Vec<u8>>> {
        self.live_tx.subscribe()
    }

    pub fn subscribe_events(&self) -> broadcast::Receiver<Arc<HubEvent>> {
        self.event_tx.subscribe()
    }

    pub fn latest_map(&self) -> Option<MapMeta> {
        self.latest.lock().expect("live hub mutex poisoned").map.clone()
    }

    pub fn latest_players(&self) -> Vec<PlayerMeta> {
        self.latest.lock().expect("live hub mutex poisoned").players.clone()
    }

    /// Forwards a client `replay{...}` command to the running source (acceptance criterion 2).
    /// Best-effort: if the control channel is momentarily full (bounded, see
    /// `CONTROL_CHANNEL_CAPACITY`) the command is dropped rather than blocking the caller (a WS
    /// connection's own recv loop) — a client that resends (e.g. holding a seek slider) will
    /// simply have its next command land instead.
    pub fn send_control(&self, control: ReplayControl) {
        let _ = self.control_tx.try_send(control);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live::source::{CharacterState, WorldFrame, test_util::ScriptedSource};
    use std::time::Duration;

    fn sample_frame(tick: u32) -> WorldFrame {
        WorldFrame {
            tick,
            characters: vec![CharacterState {
                id: 0,
                alive: true,
                x: 0,
                y: 0,
                aim_x: 0,
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
        }
    }

    #[tokio::test]
    async fn map_and_players_reach_a_subscriber_and_update_latest() {
        let map = MapMeta {
            sha256: [1; 32],
            name: "Test".to_string(),
            width: 4,
            height: 4,
        };
        let players = vec![PlayerMeta {
            id: 0,
            name: "Игрок 0".to_string(),
            team: 0,
        }];
        let source = ScriptedSource {
            events: vec![
                SourceEvent::MapChanged(map.clone()),
                SourceEvent::Players(players.clone()),
            ],
        };
        let hub = LiveHub::start(Box::new(source), Arc::new(MapCache::new()));
        let mut events = hub.subscribe_events();

        let first = events.recv().await.expect("map event");
        assert!(matches!(&*first, HubEvent::Map(m) if *m == map));
        let second = events.recv().await.expect("players event");
        assert!(matches!(&*second, HubEvent::Players(p) if *p == players));

        // Allow the pump task's own update to `latest` to land (it happens before the broadcast
        // send in program order, but `recv().await` above only guarantees the *send*, not that a
        // caller reading `latest` from a different task sees it — give the scheduler a moment).
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(hub.latest_map(), Some(map));
        assert_eq!(hub.latest_players(), players);
    }

    #[tokio::test]
    async fn live_frames_are_encoded_and_reach_a_subscriber() {
        let source = ScriptedSource {
            events: vec![SourceEvent::Frame(sample_frame(7))],
        };
        let hub = LiveHub::start(Box::new(source), Arc::new(MapCache::new()));
        let mut live = hub.subscribe_live();
        let bytes = live.recv().await.expect("live frame");
        let decoded = super::super::frame::decode(&bytes).expect("decode");
        assert_eq!(decoded.tick, 7);
    }

    #[tokio::test]
    async fn a_slow_live_subscriber_drops_frames_instead_of_blocking_the_hub() {
        let frame_count = LIVE_FRAME_BROADCAST_CAPACITY * 4;
        let events: Vec<SourceEvent> = (0..frame_count as u32)
            .map(|t| SourceEvent::Frame(sample_frame(t)))
            .collect();
        let source = ScriptedSource { events };
        let hub = LiveHub::start(Box::new(source), Arc::new(MapCache::new()));
        let mut live = hub.subscribe_live();

        // Deliberately don't read for a while, so the source (which sends as fast as the
        // scripted list allows, no per-frame delay) races far ahead of this subscriber and the
        // broadcast channel's bounded capacity is exceeded many times over.
        tokio::time::sleep(Duration::from_millis(200)).await;

        // The channel must have dropped older frames (a `Lagged` on the very next recv) rather
        // than having grown to hold all `frame_count` of them — this is the acceptance
        // criterion's "drop frames rather than buffering unboundedly", observed from the
        // subscriber's side.
        let first = live.recv().await;
        assert!(
            matches!(first, Err(broadcast::error::RecvError::Lagged(_))),
            "expected a Lagged error from a subscriber that fell behind, got {first:?}"
        );
    }

    #[tokio::test]
    async fn events_are_only_broadcast_when_non_empty() {
        let source = ScriptedSource {
            events: vec![
                SourceEvent::Events {
                    tick: 1,
                    events: vec![],
                },
                SourceEvent::Events {
                    tick: 2,
                    events: vec![GameEvent::Death { id: 0 }],
                },
            ],
        };
        let hub = LiveHub::start(Box::new(source), Arc::new(MapCache::new()));
        let mut events = hub.subscribe_events();
        let first = events.recv().await.expect("should get the non-empty one");
        match &*first {
            HubEvent::Events { tick, events } => {
                assert_eq!(*tick, 2);
                assert_eq!(events, &vec![GameEvent::Death { id: 0 }]);
            }
            other => panic!("expected Events, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn error_events_reach_subscribers() {
        let source = ScriptedSource {
            events: vec![SourceEvent::Error("malformed trace".to_string())],
        };
        let hub = LiveHub::start(Box::new(source), Arc::new(MapCache::new()));
        let mut events = hub.subscribe_events();
        let event = events.recv().await.expect("error event");
        assert!(matches!(&*event, HubEvent::Error(m) if m == "malformed trace"));
    }

    #[tokio::test]
    async fn send_control_never_blocks_even_if_the_source_never_reads_it() {
        // `ScriptedSource` drains `control_rx` in a loop after its scripted events run out, so
        // this also exercises that the channel doesn't fill up and start refusing sends under
        // completely ordinary use.
        let source = ScriptedSource { events: vec![] };
        let hub = LiveHub::start(Box::new(source), Arc::new(MapCache::new()));
        for _ in 0..(CONTROL_CHANNEL_CAPACITY * 2) {
            hub.send_control(ReplayControl::Play);
        }
    }
}
