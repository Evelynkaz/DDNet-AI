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

use tokio::sync::{broadcast, mpsc, watch};

use super::map_resolve::MapCache;
use super::source::{
    FrameSource, GameEvent, MapMeta, PlayerMeta, ReplayControl, ReplayStatus, SourceEvent, SourceKind,
};

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
pub const DEFAULT_EVENT_BROADCAST_CAPACITY: usize = 512;
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
    Events {
        tick: u32,
        events: Vec<GameEvent>,
    },
    ReplayStatus(ReplayStatus),
    /// Task 5.10: one line of the game server's chat (also kept in the in-memory ring, see [`LiveHub::latest_chat`]).
    Chat(super::chat::ChatLine),
    /// Task 4.1: the live bot's status JSON (see [`SourceEvent::BotStatus`]).
    BotStatus(String),
    /// Task 7.4: the layout of the fly's stream changed (`None`: no stream any more).
    FlyMeta(Option<String>),
    /// Task 5.7: the source on show changed, or the demo described itself (`info`: the demo's JSON description). The last
    /// field is the source generation (see [`TaggedFrame`]) from which this is true.
    Source(SourceKind, Option<String>, u64),
    Error(String),
}

#[derive(Default)]
struct Latest {
    map: Option<MapMeta>,
    players: Vec<PlayerMeta>,
    /// Task 5.6: the newest `STATUS` JSON of the live bot and when it arrived (`GET /api/bot/status`).
    bot_status: Option<(Instant, String)>,
    /// Task 7.4: the newest fly stream layout (a viewer that connects later is handed it).
    fly_meta: Option<String>,
    /// Task 5.7: which source is on show (`None` until a multiplexer says; a plain replay or a lone bot source never does).
    source: Option<SourceKind>,
    /// Task 5.7: the demo's description while it is the source on show.
    demo_info: Option<String>,
    /// Task 5.7: the source generation (bumped at every switch; frames are tagged with it).
    generation: u64,
    /// Task 5.10: the last chat lines, in memory only (`super::chat`); emptied when the source changes.
    chat: super::chat::ChatRing,
}

/// A binary frame as it travels to the browser connections, tagged with the **source generation** it was made in (task 5.7):
/// the hub bumps the generation at every source switch ([`SourceEvent::Active`]), and a connection forwards only frames of the
/// generation of the `source` message it last sent, so a frame of the previous source (or a first frame of the next one, before
/// the badge changed) is never drawn under the wrong badge.
#[derive(Debug, Clone)]
pub struct TaggedFrame {
    pub generation: u64,
    pub bytes: Arc<Vec<u8>>,
}

/// Capacity of the lossy fly-frame broadcast channel: ~5 s of frames at the bot's 12.5 Hz.
const FLY_FRAME_BROADCAST_CAPACITY: usize = 64;

/// Who watches something (the fly, or the page at all). The count and the demand flag change together, under one lock (an
/// increment racing a decrement must not leave the flag false with a watcher present).
struct Demand {
    watchers: Mutex<usize>,
    tx: watch::Sender<bool>,
}

impl Demand {
    fn new() -> (Arc<Demand>, watch::Receiver<bool>) {
        let (tx, rx) = watch::channel(false);
        (
            Arc::new(Demand {
                watchers: Mutex::new(0),
                tx,
            }),
            rx,
        )
    }

    /// One more watcher; `before` runs under the lock, ahead of the announcement.
    fn join(&self, before: impl FnOnce()) {
        let mut n = self.watchers.lock().expect("demand mutex poisoned");
        *n += 1;
        before();
        self.tx.send_if_modified(|v| !std::mem::replace(v, true));
    }

    fn leave(&self) {
        let mut n = self.watchers.lock().expect("demand mutex poisoned");
        *n = n.saturating_sub(1);
        let want = *n > 0;
        self.tx.send_if_modified(|v| std::mem::replace(v, want) != want);
    }

    fn count(&self) -> usize {
        *self.watchers.lock().expect("demand mutex poisoned")
    }
}

/// A browser connection's subscription to the fly stream: dropping it ends the demand.
pub struct FlySubscription {
    /// Pre-validated `DFLY` frames, lossy under load like the live ones.
    pub rx: broadcast::Receiver<TaggedFrame>,
    demand: Arc<Demand>,
}

impl Drop for FlySubscription {
    fn drop(&mut self) {
        self.demand.leave();
    }
}

/// Task 5.7: a browser connection exists. While at least one does, the source is told the page is open (the offline demo
/// pauses itself otherwise); dropping the last one tells it so.
pub struct ViewerGuard {
    demand: Arc<Demand>,
}

impl Drop for ViewerGuard {
    fn drop(&mut self) {
        self.demand.leave();
    }
}

pub struct LiveHub {
    live_tx: broadcast::Sender<TaggedFrame>,
    fly_tx: broadcast::Sender<TaggedFrame>,
    fly_demand: Arc<Demand>,
    view_demand: Arc<Demand>,
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
        Self::start_with_event_capacity(source, map_cache, DEFAULT_EVENT_BROADCAST_CAPACITY)
    }

    /// [`LiveHub::start`] with the capacity of the event channel given (a connection that falls further behind than this loses
    /// events and is re-told the state, see `ws.rs`); tests make it small to force that.
    pub fn start_with_event_capacity(
        mut source: Box<dyn FrameSource>,
        map_cache: Arc<MapCache>,
        event_capacity: usize,
    ) -> Self {
        let (events_tx, events_rx) = mpsc::channel(SOURCE_EVENTS_CHANNEL_CAPACITY);
        let (control_tx, control_rx) = mpsc::channel(CONTROL_CHANNEL_CAPACITY);
        let (fly_demand, fly_rx) = Demand::new();
        let (view_demand, view_rx) = Demand::new();
        source.attach_fly_demand(fly_rx);
        source.attach_view_demand(view_rx);
        let source_task = source.spawn(events_tx, control_rx);

        let (live_tx, _) = broadcast::channel(LIVE_FRAME_BROADCAST_CAPACITY);
        let (fly_tx, _) = broadcast::channel(FLY_FRAME_BROADCAST_CAPACITY);
        let (event_tx, _) = broadcast::channel(event_capacity.max(1));
        let latest = Arc::new(Mutex::new(Latest::default()));

        let pump_task = tokio::spawn(Self::pump(
            events_rx,
            live_tx.clone(),
            fly_tx.clone(),
            event_tx.clone(),
            latest.clone(),
        ));

        LiveHub {
            live_tx,
            fly_tx,
            fly_demand,
            view_demand,
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
        live_tx: broadcast::Sender<TaggedFrame>,
        fly_tx: broadcast::Sender<TaggedFrame>,
        event_tx: broadcast::Sender<Arc<HubEvent>>,
        latest: Arc<Mutex<Latest>>,
    ) {
        // Not `Instant::now()`: the very first status update should always go out immediately,
        // not wait a full `STATUS_THROTTLE` after the hub starts.
        let mut last_status_sent = Instant::now().checked_sub(STATUS_THROTTLE).unwrap_or_else(Instant::now);
        // Bumped at every source switch; the frames made after it carry the new number.
        let mut generation: u64 = 0;
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
                    let _ = live_tx.send(TaggedFrame {
                        generation,
                        bytes: Arc::new(bytes),
                    });
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
                SourceEvent::Chat(line) => {
                    latest.lock().expect("live hub mutex poisoned").chat.push(line.clone());
                    let _ = event_tx.send(Arc::new(HubEvent::Chat(line)));
                }
                SourceEvent::BotStatus(json) => {
                    latest.lock().expect("live hub mutex poisoned").bot_status = Some((Instant::now(), json.clone()));
                    let _ = event_tx.send(Arc::new(HubEvent::BotStatus(json)));
                }
                SourceEvent::FlyMeta(meta) => {
                    latest.lock().expect("live hub mutex poisoned").fly_meta = meta.clone();
                    let _ = event_tx.send(Arc::new(HubEvent::FlyMeta(meta)));
                }
                SourceEvent::FlyFrame(bytes) => {
                    let _ = fly_tx.send(TaggedFrame {
                        generation,
                        bytes: Arc::new(bytes),
                    });
                }
                SourceEvent::Link(_) => {} // the multiplexer's business (`super::mux`)
                SourceEvent::Active(kind) => {
                    // Everything kept of the previous source is stale now; the new one's state follows.
                    generation += 1;
                    *latest.lock().expect("live hub mutex poisoned") = Latest {
                        source: Some(kind),
                        generation,
                        ..Latest::default()
                    };
                    let _ = event_tx.send(Arc::new(HubEvent::Source(kind, None, generation)));
                    // A page watching the fly must not keep the layout of the previous source.
                    let _ = event_tx.send(Arc::new(HubEvent::FlyMeta(None)));
                }
                SourceEvent::DemoInfo(info) => {
                    // Only while the demo is on show, and only when it says something new.
                    let changed = {
                        let mut l = latest.lock().expect("live hub mutex poisoned");
                        if l.source != Some(SourceKind::Demo) || l.demo_info.as_deref() == Some(info.as_str()) {
                            false
                        } else {
                            l.demo_info = Some(info.clone());
                            true
                        }
                    };
                    if changed {
                        let _ = event_tx.send(Arc::new(HubEvent::Source(SourceKind::Demo, Some(info), generation)));
                    }
                }
                SourceEvent::Error(message) => {
                    let _ = event_tx.send(Arc::new(HubEvent::Error(message)));
                }
            }
        }
    }

    /// Task 7.4: subscribes a browser connection to the fly stream. While at least one subscription lives, the source
    /// is told that somebody watches (the bot then builds frames); dropping the last one tells it to stop.
    pub fn subscribe_fly(&self) -> FlySubscription {
        // Subscribe to the frames before announcing the demand, so the first frame cannot be missed.
        let mut rx = None;
        self.fly_demand.join(|| rx = Some(self.fly_tx.subscribe()));
        FlySubscription {
            rx: rx.expect("the closure ran"),
            demand: Arc::clone(&self.fly_demand),
        }
    }

    /// Task 5.7: a browser connection is open (see [`ViewerGuard`]).
    pub fn viewer(&self) -> ViewerGuard {
        self.view_demand.join(|| {});
        ViewerGuard {
            demand: Arc::clone(&self.view_demand),
        }
    }

    /// The newest fly stream layout (`None` before the bot announced one, or when its brain has no stream).
    pub fn latest_fly_meta(&self) -> Option<String> {
        self.latest.lock().expect("live hub mutex poisoned").fly_meta.clone()
    }

    /// How many browser connections watch the fly now.
    pub fn fly_watchers(&self) -> usize {
        self.fly_demand.count()
    }

    /// How many browser connections are open now.
    pub fn viewers(&self) -> usize {
        self.view_demand.count()
    }

    /// Task 5.7: which source is on show and, for the demo, its description; `None` when no multiplexer feeds this hub.
    pub fn latest_source(&self) -> Option<(SourceKind, Option<String>)> {
        self.latest_source_tagged().map(|(k, info, _)| (k, info))
    }

    /// [`LiveHub::latest_source`] with the source generation it belongs to, read together (see [`TaggedFrame`]); a hub with no
    /// multiplexer has none, and its frames are all generation 0.
    pub fn latest_source_tagged(&self) -> Option<(SourceKind, Option<String>, u64)> {
        let l = self.latest.lock().expect("live hub mutex poisoned");
        l.source.map(|k| (k, l.demo_info.clone(), l.generation))
    }

    pub fn subscribe_live(&self) -> broadcast::Receiver<TaggedFrame> {
        self.live_tx.subscribe()
    }

    pub fn subscribe_events(&self) -> broadcast::Receiver<Arc<HubEvent>> {
        self.event_tx.subscribe()
    }

    pub fn latest_map(&self) -> Option<MapMeta> {
        self.latest.lock().expect("live hub mutex poisoned").map.clone()
    }

    /// The chat lines the web process still remembers (at most [`super::chat::RING_LINES`]), oldest first.
    pub fn latest_chat(&self) -> Vec<super::chat::ChatLine> {
        self.latest.lock().expect("live hub mutex poisoned").chat.snapshot()
    }

    pub fn latest_players(&self) -> Vec<PlayerMeta> {
        self.latest.lock().expect("live hub mutex poisoned").players.clone()
    }

    /// The newest bot `STATUS` JSON and how long ago it arrived (task 5.6); `None` before the first one.
    pub fn latest_bot_status(&self) -> Option<(Duration, String)> {
        self.latest
            .lock()
            .expect("live hub mutex poisoned")
            .bot_status
            .as_ref()
            .map(|(at, json)| (at.elapsed(), json.clone()))
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
            ..PlayerMeta::default()
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
    async fn the_newest_bot_status_is_kept_with_its_age() {
        let source = ScriptedSource {
            events: vec![
                SourceEvent::BotStatus(r#"{"tick":1}"#.to_string()),
                SourceEvent::BotStatus(r#"{"tick":2}"#.to_string()),
            ],
        };
        let hub = LiveHub::start(Box::new(source), Arc::new(MapCache::new()));
        let mut seen = None;
        for _ in 0..200 {
            if let Some((age, json)) = hub.latest_bot_status()
                && json == r#"{"tick":2}"#
            {
                seen = Some(age);
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(seen.expect("the newest status is kept") < Duration::from_secs(5));
    }

    #[tokio::test]
    async fn live_frames_are_encoded_and_reach_a_subscriber() {
        let source = ScriptedSource {
            events: vec![SourceEvent::Frame(sample_frame(7))],
        };
        let hub = LiveHub::start(Box::new(source), Arc::new(MapCache::new()));
        let mut live = hub.subscribe_live();
        let bytes = live.recv().await.expect("live frame");
        let decoded = super::super::frame::decode(&bytes.bytes).expect("decode");
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

    /// A source that keeps the demand receiver the hub hands it, and emits what the test pushes.
    struct DemandSource {
        demand: Option<watch::Receiver<bool>>,
        seen: Arc<Mutex<Vec<bool>>>,
        script: Vec<SourceEvent>,
    }

    impl FrameSource for DemandSource {
        fn attach_fly_demand(&mut self, demand: watch::Receiver<bool>) {
            self.demand = Some(demand);
        }
        fn spawn(
            mut self: Box<Self>,
            events_tx: mpsc::Sender<SourceEvent>,
            mut control_rx: mpsc::Receiver<ReplayControl>,
        ) -> tokio::task::JoinHandle<()> {
            tokio::spawn(async move {
                let mut demand = self.demand.take().expect("the hub attached the demand");
                for e in std::mem::take(&mut self.script) {
                    let _ = events_tx.send(e).await;
                }
                loop {
                    tokio::select! {
                        changed = demand.changed() => {
                            if changed.is_err() { return; }
                            self.seen.lock().unwrap().push(*demand.borrow_and_update());
                        }
                        c = control_rx.recv() => if c.is_none() { return; },
                    }
                }
            })
        }
    }

    #[tokio::test]
    async fn the_source_is_told_while_somebody_watches_the_fly_and_only_then() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let source = DemandSource {
            demand: None,
            seen: Arc::clone(&seen),
            script: vec![
                SourceEvent::FlyMeta(Some(r#"{"v":1}"#.to_string())),
                SourceEvent::FlyFrame(b"DFLY-frame".to_vec()),
            ],
        };
        let hub = LiveHub::start(Box::new(source), Arc::new(MapCache::new()));
        assert_eq!(hub.fly_watchers(), 0);
        let settle = || tokio::time::sleep(Duration::from_millis(60));
        settle().await;
        assert!(seen.lock().unwrap().is_empty(), "no demand before anybody subscribes");
        assert_eq!(
            hub.latest_fly_meta().as_deref(),
            Some(r#"{"v":1}"#),
            "the layout is kept for late viewers"
        );

        let a = hub.subscribe_fly();
        let b = hub.subscribe_fly();
        settle().await;
        assert_eq!(hub.fly_watchers(), 2);
        assert_eq!(*seen.lock().unwrap(), [true], "one announcement for two watchers");
        drop(a);
        settle().await;
        assert_eq!(*seen.lock().unwrap(), [true], "one left: still wanted");
        drop(b);
        settle().await;
        assert_eq!(hub.fly_watchers(), 0);
        assert_eq!(
            *seen.lock().unwrap(),
            [true, false],
            "the last one leaving ends the demand"
        );
        let c = hub.subscribe_fly();
        settle().await;
        assert_eq!(*seen.lock().unwrap(), [true, false, true]);
        drop(c);
    }

    #[tokio::test]
    async fn fly_frames_reach_subscribers_lossily_and_a_layout_change_is_an_event() {
        let frames = FLY_FRAME_BROADCAST_CAPACITY * 3;
        let mut script: Vec<SourceEvent> = vec![SourceEvent::FlyMeta(Some(r#"{"v":1,"a":1}"#.to_string()))];
        script.extend((0..frames).map(|i| SourceEvent::FlyFrame(vec![i as u8])));
        script.push(SourceEvent::FlyMeta(None));
        let source = DemandSource {
            demand: None,
            seen: Arc::new(Mutex::new(Vec::new())),
            script,
        };
        let hub = LiveHub::start(Box::new(source), Arc::new(MapCache::new()));
        let mut events = hub.subscribe_events();
        let mut sub = hub.subscribe_fly();
        tokio::time::sleep(Duration::from_millis(200)).await;
        // A subscriber that did not read falls behind and skips ahead instead of buffering everything.
        assert!(matches!(
            sub.rx.recv().await,
            Err(broadcast::error::RecvError::Lagged(_))
        ));
        // The layout changes were events on the reliable channel, in order.
        let first = events.recv().await.unwrap();
        assert!(matches!(&*first, HubEvent::FlyMeta(Some(m)) if m.contains("\"a\":1")));
        let second = events.recv().await.unwrap();
        assert!(matches!(&*second, HubEvent::FlyMeta(None)));
        assert_eq!(hub.latest_fly_meta(), None);
    }

    #[tokio::test]
    async fn a_source_switch_drops_what_was_kept_of_the_previous_one_and_is_announced() {
        let map = MapMeta {
            sha256: [2; 32],
            name: "Arena".to_string(),
            width: 4,
            height: 4,
        };
        let source = ScriptedSource {
            events: vec![
                SourceEvent::Active(SourceKind::Live),
                SourceEvent::MapChanged(map.clone()),
                SourceEvent::Players(vec![PlayerMeta {
                    id: 1,
                    name: "c1-aaaaaaaa".to_string(),
                    team: 0,
                    ..PlayerMeta::default()
                }]),
                SourceEvent::BotStatus(r#"{"tick":1}"#.to_string()),
                SourceEvent::FlyMeta(Some(r#"{"v":1}"#.to_string())),
                // The demo takes over: nothing of the bot is kept.
                SourceEvent::Active(SourceKind::Demo),
                SourceEvent::DemoInfo(r#"{"arena":"a","bundle":"b"}"#.to_string()),
                // The same description again is not news.
                SourceEvent::DemoInfo(r#"{"arena":"a","bundle":"b"}"#.to_string()),
                SourceEvent::Link(true),
            ],
        };
        let hub = LiveHub::start(Box::new(source), Arc::new(MapCache::new()));
        let mut events = hub.subscribe_events();
        let mut seen = Vec::new();
        // Source, FlyMeta(None) for each Active; Map, Players, BotStatus, FlyMeta in between; one demo description.
        while seen.len() < 9 {
            let e = tokio::time::timeout(Duration::from_secs(2), events.recv())
                .await
                .expect("events")
                .unwrap();
            seen.push(e);
        }
        assert!(matches!(&*seen[0], HubEvent::Source(SourceKind::Live, None, 1)));
        assert!(matches!(&*seen[1], HubEvent::FlyMeta(None)));
        assert!(matches!(&*seen[2], HubEvent::Map(_)));
        assert!(matches!(&*seen[5], HubEvent::FlyMeta(Some(_))));
        assert!(matches!(&*seen[6], HubEvent::Source(SourceKind::Demo, None, 2)));
        assert!(matches!(&*seen[7], HubEvent::FlyMeta(None)));
        assert!(matches!(&*seen[8], HubEvent::Source(SourceKind::Demo, Some(i), 2) if i.contains("\"a\"")));
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            events.try_recv().is_err(),
            "the repeated description and the link are not events"
        );
        assert_eq!(hub.latest_map(), None, "the live bot's map is gone");
        assert!(hub.latest_players().is_empty());
        assert!(hub.latest_bot_status().is_none(), "the demo is not a bot with a status");
        assert_eq!(hub.latest_fly_meta(), None);
        let (kind, info) = hub.latest_source().expect("a source was announced");
        assert_eq!(kind, SourceKind::Demo);
        assert!(info.unwrap().contains("bundle"));
    }

    /// Task 5.7: a frame carries the generation of the switch it was made after, and the `source` event of that switch carries
    /// the same number, so a connection can tell which frames belong under which badge.
    #[tokio::test]
    async fn frames_carry_the_generation_of_the_switch_they_were_made_after() {
        let source = ScriptedSource {
            events: vec![
                SourceEvent::Frame(sample_frame(1)),
                SourceEvent::Active(SourceKind::Live),
                SourceEvent::Frame(sample_frame(2)),
                SourceEvent::Active(SourceKind::Demo),
                SourceEvent::Frame(sample_frame(3)),
            ],
        };
        let hub = LiveHub::start(Box::new(source), Arc::new(MapCache::new()));
        let mut live = hub.subscribe_live();
        let mut events = hub.subscribe_events();
        let mut frames = Vec::new();
        for _ in 0..3 {
            let f = tokio::time::timeout(Duration::from_secs(2), live.recv())
                .await
                .unwrap()
                .unwrap();
            frames.push((super::super::frame::decode(&f.bytes).unwrap().tick, f.generation));
        }
        assert_eq!(frames, [(1, 0), (2, 1), (3, 2)]);
        let mut gens = Vec::new();
        while gens.len() < 2 {
            let e = tokio::time::timeout(Duration::from_secs(2), events.recv())
                .await
                .unwrap()
                .unwrap();
            if let HubEvent::Source(kind, _, generation) = &*e {
                gens.push((*kind, *generation));
            }
        }
        assert_eq!(gens, [(SourceKind::Live, 1), (SourceKind::Demo, 2)]);
        assert_eq!(
            hub.latest_source_tagged().map(|(k, _, g)| (k, g)),
            Some((SourceKind::Demo, 2))
        );
    }

    #[tokio::test]
    async fn a_demo_description_is_ignored_unless_the_demo_is_on_show() {
        let source = ScriptedSource {
            events: vec![
                SourceEvent::Active(SourceKind::Live),
                SourceEvent::DemoInfo(r#"{"arena":"a","bundle":"b"}"#.to_string()),
            ],
        };
        let hub = LiveHub::start(Box::new(source), Arc::new(MapCache::new()));
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(hub.latest_source(), Some((SourceKind::Live, None)));
    }

    #[tokio::test]
    async fn a_hub_without_a_multiplexer_names_no_source() {
        let hub = LiveHub::start(Box::new(ScriptedSource { events: vec![] }), Arc::new(MapCache::new()));
        assert_eq!(hub.latest_source(), None);
    }

    #[tokio::test]
    async fn the_source_is_told_while_a_browser_is_connected_and_only_then() {
        struct ViewSource {
            demand: Option<watch::Receiver<bool>>,
            seen: Arc<Mutex<Vec<bool>>>,
        }
        impl FrameSource for ViewSource {
            fn attach_view_demand(&mut self, demand: watch::Receiver<bool>) {
                self.demand = Some(demand);
            }
            fn spawn(
                mut self: Box<Self>,
                _events_tx: mpsc::Sender<SourceEvent>,
                mut control_rx: mpsc::Receiver<ReplayControl>,
            ) -> tokio::task::JoinHandle<()> {
                tokio::spawn(async move {
                    let mut demand = self.demand.take().expect("the hub attached the demand");
                    loop {
                        tokio::select! {
                            changed = demand.changed() => {
                                if changed.is_err() { return; }
                                self.seen.lock().unwrap().push(*demand.borrow_and_update());
                            }
                            c = control_rx.recv() => if c.is_none() { return; },
                        }
                    }
                })
            }
        }
        let seen = Arc::new(Mutex::new(Vec::new()));
        let hub = LiveHub::start(
            Box::new(ViewSource {
                demand: None,
                seen: Arc::clone(&seen),
            }),
            Arc::new(MapCache::new()),
        );
        let settle = || tokio::time::sleep(Duration::from_millis(60));
        settle().await;
        assert!(seen.lock().unwrap().is_empty(), "nobody connected: nothing announced");
        let a = hub.viewer();
        let b = hub.viewer();
        settle().await;
        assert_eq!(hub.viewers(), 2);
        assert_eq!(*seen.lock().unwrap(), [true], "one announcement for two browsers");
        drop(a);
        settle().await;
        assert_eq!(*seen.lock().unwrap(), [true]);
        drop(b);
        settle().await;
        assert_eq!(hub.viewers(), 0);
        assert_eq!(*seen.lock().unwrap(), [true, false]);
        assert_eq!(hub.fly_watchers(), 0, "a viewer is not a fly watcher");
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
