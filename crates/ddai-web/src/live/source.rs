//! `FrameSource`: the abstraction (acceptance criterion 1) between "something that produces a
//! sequence of game ticks" and the WebSocket hub that fans those ticks out to browsers. The live
//! bot itself doesn't exist yet (tasks 2.3/2.4/4.x) — [`crate::live::replay::ReplaySource`] is
//! the one real implementation today, replaying Oracle B traces; a future live-session source, an
//! arena source, or "the fly" would each implement this same trait and plug into the same hub
//! unchanged.
//!
//! A `FrameSource` does not return a `Stream` or expose an `async fn` directly (which would force
//! either `async-trait` — an extra dependency — or giving up `dyn FrameSource` entirely): instead
//! [`FrameSource::spawn`] is a plain, object-safe method that starts its own background task and
//! communicates purely through two channels, [`SourceEvent`] out and [`ReplayControl`] in. Tests
//! (`crate::live::hub`'s own tests) exercise the trait itself with a tiny scripted
//! implementation, independent of the real replay source.

use tokio::sync::mpsc;

/// Where a `WorldFrame`'s current map came from, and just enough to fetch its rendered scene
/// (acceptance criterion 1: "tick, map id, per-character state, players, events").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapMeta {
    pub sha256: [u8; 32],
    pub name: String,
    pub width: u32,
    pub height: u32,
}

/// A player slot's display metadata (acceptance criterion 1: "players (id, name, team)"). Sent
/// as its own WS message only when it changes, not on every frame (acceptance criterion "Binary
/// frame format ... Names/teams go in a separate JSON `players` message on change").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlayerMeta {
    pub id: u8,
    pub name: String,
    pub team: u8,
}

/// One character's rendering-relevant state for one tick (acceptance criterion 1). Field
/// semantics deliberately reuse `ddai-net`'s `CharacterView`/`DDNetCharacter` naming (task 2.2b)
/// where the same concept exists in both, per the task's context note ("reuse its
/// character/player field semantics for the frame format so the live source in task 2.4 maps
/// 1:1") — see `crate::live::frame`'s doc comment for the exact field-by-field mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CharacterState {
    pub id: u8,
    pub alive: bool,
    /// Position, DDNet game units (32 units = 1 tile), matching `Character::x`/`y`.
    pub x: i32,
    pub y: i32,
    /// The applied input's `target_x`/`target_y` (cursor position relative to the tee) — the
    /// direct, unambiguous source for "which way is this character aiming", rather than
    /// reinterpreting `CCharacterCore::m_Angle`'s internal fixed-point scale (see `frame.rs`'s
    /// doc comment for why).
    pub aim_x: i32,
    pub aim_y: i32,
    /// `gamecore.h`'s `HOOK_STATE` enum (`HOOK_RETRACTED = -1` .. `HOOK_GRABBED = 5`).
    pub hook_state: i8,
    pub hook_x: i32,
    pub hook_y: i32,
    /// The character id this one's hook is attached to, or `None`.
    pub hooked_id: Option<u8>,
    pub weapon: u8,
    pub team: u8,
    pub frozen: bool,
    pub deep_frozen: bool,
    pub live_frozen: bool,
}

/// One tick's worth of every character's state (acceptance criterion 1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorldFrame {
    pub tick: u32,
    pub characters: Vec<CharacterState>,
}

/// A discrete gameplay event for the HUD/event log (acceptance criterion 1: "events
/// (freeze/unfreeze/death/respawn/hammer hit/hook grab/tele)"). This task's replay source
/// implements the four events it can derive *exactly* from Oracle B's own already-computed
/// per-tick fields, with no heuristic guessing (see `crate::live::replay`'s doc comment for the
/// two the spec also lists, `HammerHit`/`Teleport`, that are deliberately deferred).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameEvent {
    Freeze { id: u8 },
    Unfreeze { id: u8 },
    Death { id: u8 },
    Respawn { id: u8 },
    HookGrab { id: u8, target: Option<u8> },
}

/// Something a [`FrameSource`] reports as it runs. The hub (`crate::live::hub`) turns each of
/// these into the matching WS message(s) — see that module for the mapping.
#[derive(Debug, Clone, PartialEq)]
pub enum SourceEvent {
    /// The active map changed (including "a new source just started"). Always sent before any
    /// [`SourceEvent::Frame`]/[`SourceEvent::Events`] that reference the new map.
    MapChanged(MapMeta),
    /// The player roster changed (new/renamed/re-teamed slots).
    Players(Vec<PlayerMeta>),
    Frame(WorldFrame),
    /// Zero or more events that happened on `tick`. The hub only forwards this when non-empty.
    Events {
        tick: u32,
        events: Vec<GameEvent>,
    },
    /// A source-specific status update the hub exposes to the client verbatim (e.g. the replay
    /// source's play/pause/speed/current-file state — acceptance criterion 2).
    ReplayStatus(ReplayStatus),
    /// Task 4.1: the live bot's own status (target, mode, brain telemetry, latency) as one JSON
    /// object — only the bot-socket source produces it (`docs/formats.md` §21, `STATUS`). Opaque to
    /// the hub: forwarded to browsers as the `bot` WS message.
    BotStatus(String),
    /// Task 7.4: the layout of the fly's visualisation stream (one JSON object, `docs/formats.md` §27.2), or `None`
    /// when the bot's brain has none (or the bot is gone). Validated by the source.
    FlyMeta(Option<String>),
    /// Task 7.4: one `DFLY` frame (`docs/formats.md` §27.1), already checked by [`crate::live::fly::validate_frame`].
    FlyFrame(Vec<u8>),
    /// Something went wrong that the source can recover from (acceptance criterion 2: "parsing
    /// is bounded, and a malformed trace gives an error event, not a panic") — reported to the
    /// client, never a panic or a silently-dropped frame stream.
    Error(String),
}

/// The replay source's own status, mirrored to the client so its UI can show which trace is
/// playing, at what speed, paused or not (acceptance criterion 2/3).
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayStatus {
    pub file: String,
    pub tick: u32,
    pub tick_count: u32,
    pub playing: bool,
    pub speed: f32,
}

/// A client `replay{...}` WS message (acceptance criterion 2: "Speed, pause and seek are
/// controllable from the page: WS messages `replay{play|pause|speed|seek|next}`"). Not every
/// `FrameSource` implementation has to honor every variant (a future live-session source has no
/// use for `Seek`, say) — an implementation that doesn't understand a command is free to ignore
/// it, matching the WS layer's own "unknown message types are ignored" convention (`ws.rs`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReplayControl {
    Play,
    Pause,
    /// Playback speed multiplier (1.0 = real time, i.e. ticks advance at the map's tick rate).
    /// Clamped by the receiving source to a sane range — see `crate::live::replay`.
    SetSpeed(f32),
    /// Seek to an absolute tick within the currently-playing trace.
    Seek(u32),
    /// Skip immediately to the next trace in the corpus.
    Next,
}

/// The abstraction every frame source implements (acceptance criterion 1). Object-safe on
/// purpose (`Box<dyn FrameSource>` works) — see this module's doc comment for why `spawn` is
/// plain and synchronous rather than an `async fn`.
pub trait FrameSource: Send + 'static {
    /// Starts a background task that drives this source until `control_rx`'s sender is dropped
    /// (the hub shutting down) or the source itself decides it's done, sending every
    /// [`SourceEvent`] it produces to `events_tx` and reading [`ReplayControl`] commands from
    /// `control_rx` as they arrive. Consumes `self` (owns the task for its whole lifetime) —
    /// callers that need to keep talking to a *running* source do so purely through the channels
    /// they already hold, never through the (now-moved-away) `self` value.
    fn spawn(
        self: Box<Self>,
        events_tx: mpsc::Sender<SourceEvent>,
        control_rx: mpsc::Receiver<ReplayControl>,
    ) -> tokio::task::JoinHandle<()>;

    /// Task 7.4: whether some browser watches the fly (`true`) or none does. Called once by the hub before
    /// [`FrameSource::spawn`]; a source that has a fly stream (the bot's) tells the bot when the value changes, so the
    /// fly builds frames only while someone looks. The default ignores it (a replay has no fly).
    fn attach_fly_demand(&mut self, demand: tokio::sync::watch::Receiver<bool>) {
        let _ = demand;
    }
}

#[cfg(test)]
pub(crate) mod test_util {
    //! A tiny scripted [`FrameSource`] used by `crate::live::hub`'s own tests, so those tests
    //! don't need a real trace file on disk to exercise the hub's fan-out/backpressure/control
    //! wiring.
    use super::*;

    pub struct ScriptedSource {
        pub events: Vec<SourceEvent>,
    }

    impl FrameSource for ScriptedSource {
        fn spawn(
            self: Box<Self>,
            events_tx: mpsc::Sender<SourceEvent>,
            mut control_rx: mpsc::Receiver<ReplayControl>,
        ) -> tokio::task::JoinHandle<()> {
            tokio::spawn(async move {
                for event in self.events {
                    if events_tx.send(event).await.is_err() {
                        return;
                    }
                }
                // Keep draining control messages (so a sender never blocks/panics on a closed
                // channel mid-test) until the hub drops its sender.
                while control_rx.recv().await.is_some() {}
            })
        }
    }
}
