//! Two producers behind one [`FrameSource`], with a priority (task 5.7, D-075): the **live bot** first, and the offline
//! **demo** (`ddnet-ai fly watch`, the fly playing an arena) only while the live bot is not there.
//!
//! Both are ordinary bridge sources ([`super::bot_source::BotSource`] on two sockets). Each runs all the time and tells
//! the multiplexer, in its own event stream, when its connection is usable ([`SourceEvent::Link`]). The multiplexer
//! picks the source on show: live if its link is up, else the demo if its link is up, else none. Only the chosen
//! source's events reach the hub; the others are read and dropped, so a demo cannot be mistaken for the bot, and the demo's
//! `STATUS` never becomes the bot's status (it becomes [`SourceEvent::DemoInfo`], rebuilt from two short strings).
//!
//! **Switching.** The hub is told [`SourceEvent::Active`] first (it drops everything it kept of the previous source), then
//! the new source's own state is replayed from a per-source cache (map, roster, fly layout, demo description), so that a
//! demo that has been connected all along appears at once when the bot goes away, and the next frame is already the new
//! source's. Frames are never cached.
//!
//! **Hysteresis.** Switching *to* the live bot is immediate; falling *back* to the demo waits [`FALLBACK_DELAY`] after the bot's link
//! drops, so a short break of the site's own bridge connection (a reconnect, a bot restart) does not flip the whole page to the
//! demo and back. A link that returns within the delay changes nothing.
//!
//! **Demand.** What the hub says about its viewers (the fly stream, a browser connected at all) goes to the live source
//! as it is. The demo gets it only while the live bot is not up: with the bot running nobody asks the demo for anything,
//! so it stays paused (`ddnet-ai fly watch --pause-idle`) and costs nothing.
//!
//! **Not forwarded.** The page's `replay{...}` commands: a bot source ignores them anyway. Nothing the site sends as a
//! command goes through here (the control socket is a different path, and the demo has none).

use std::time::Duration;

use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tokio::time::Instant;

use super::bot_source::demand_changed;
use super::source::{FrameSource, MapMeta, PlayerMeta, ReplayControl, SourceEvent, SourceKind};

/// Capacity of the channel each sub-source reports into (the hub's own is 32).
const CHANNEL: usize = 32;
/// How long the live bot's link may stay down before the page falls back to the demo.
pub const FALLBACK_DELAY: Duration = Duration::from_millis(2500);
/// Longest string of the demo's description let through.
const MAX_INFO_STR: usize = 64;

pub struct MuxSource {
    live: Box<dyn FrameSource>,
    demo: Option<Box<dyn FrameSource>>,
    fly_demand: Option<watch::Receiver<bool>>,
    view_demand: Option<watch::Receiver<bool>>,
    fallback_delay: Duration,
}

impl MuxSource {
    /// `demo: None` is a plain live source with the switching messages (the site then shows the live bot or "none").
    pub fn new(live: Box<dyn FrameSource>, demo: Option<Box<dyn FrameSource>>) -> Self {
        MuxSource {
            live,
            demo,
            fly_demand: None,
            view_demand: None,
            fallback_delay: FALLBACK_DELAY,
        }
    }

    /// How long the live bot's link may be down before the demo takes over (default [`FALLBACK_DELAY`]; zero: at once).
    pub fn with_fallback_delay(mut self, delay: Duration) -> Self {
        self.fallback_delay = delay;
        self
    }
}

impl FrameSource for MuxSource {
    fn attach_fly_demand(&mut self, demand: watch::Receiver<bool>) {
        self.fly_demand = Some(demand);
    }

    fn attach_view_demand(&mut self, demand: watch::Receiver<bool>) {
        self.view_demand = Some(demand);
    }

    fn spawn(
        self: Box<Self>,
        events_tx: mpsc::Sender<SourceEvent>,
        control_rx: mpsc::Receiver<ReplayControl>,
    ) -> JoinHandle<()> {
        tokio::spawn(run(*self, events_tx, control_rx))
    }
}

/// Aborts the task when the multiplexer ends (dropping a `JoinHandle` alone would leave the sub-source running).
struct AbortOnDrop(JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// The demand the demo is shown: the hub's, but closed while the live bot is up.
struct Gate {
    upstream: Option<watch::Receiver<bool>>,
    out: watch::Sender<bool>,
}

impl Gate {
    fn new(upstream: Option<watch::Receiver<bool>>) -> (Gate, watch::Receiver<bool>) {
        let (out, rx) = watch::channel(false);
        (Gate { upstream, out }, rx)
    }

    fn apply(&self, open: bool) {
        let want = open && self.upstream.as_ref().is_some_and(|u| *u.borrow());
        self.out.send_if_modified(|v| std::mem::replace(v, want) != want);
    }
}

/// The state of one source that the multiplexer replays when it becomes the one on show.
#[derive(Default)]
struct Side {
    link: bool,
    map: Option<MapMeta>,
    players: Option<Vec<PlayerMeta>>,
    fly_meta: Option<String>,
    /// The demo's description (always `None` on the live side).
    info: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Which {
    Live,
    Demo,
}

/// The multiplexer's whole state (the loop in [`run`] only feeds it events).
struct Mux {
    live: Side,
    demo: Side,
    has_demo: bool,
    /// `None` until the first announcement.
    active: Option<SourceKind>,
    out: mpsc::Sender<SourceEvent>,
    fly_gate: Gate,
    view_gate: Gate,
    fallback_delay: Duration,
    /// The live bot's link is down but the page still shows it until this moment (the hysteresis).
    live_grace_until: Option<Instant>,
}

/// The hub is gone.
struct HubClosed;

impl Mux {
    fn side(&mut self, which: Which) -> &mut Side {
        match which {
            Which::Live => &mut self.live,
            Which::Demo => &mut self.demo,
        }
    }

    fn is_active(&self, which: Which) -> bool {
        self.active
            == Some(match which {
                Which::Live => SourceKind::Live,
                Which::Demo => SourceKind::Demo,
            })
    }

    /// The live bot counts as there while its link is up, or is down for less than the fallback delay.
    fn live_effective(&self) -> bool {
        self.live.link || self.live_grace_until.is_some()
    }

    async fn send(&self, e: SourceEvent) -> Result<(), HubClosed> {
        self.out.send(e).await.map_err(|_| HubClosed)
    }

    /// Chooses the source on show and, if that changed, announces it and replays its state.
    async fn reselect(&mut self) -> Result<(), HubClosed> {
        let want = if self.live_effective() {
            SourceKind::Live
        } else if self.has_demo && self.demo.link {
            SourceKind::Demo
        } else {
            SourceKind::None
        };
        // The demo is asked for anything only while the live bot is not up.
        self.fly_gate.apply(!self.live_effective());
        self.view_gate.apply(!self.live_effective());
        if self.active == Some(want) {
            return Ok(());
        }
        self.active = Some(want);
        self.send(SourceEvent::Active(want)).await?;
        let side = match want {
            SourceKind::Live => &self.live,
            SourceKind::Demo => &self.demo,
            SourceKind::None => return Ok(()),
        };
        let (map, players, fly_meta, info) = (
            side.map.clone(),
            side.players.clone(),
            side.fly_meta.clone(),
            side.info.clone(),
        );
        if let Some(m) = map {
            self.send(SourceEvent::MapChanged(m)).await?;
        }
        if let Some(p) = players {
            self.send(SourceEvent::Players(p)).await?;
        }
        if let Some(m) = fly_meta {
            self.send(SourceEvent::FlyMeta(Some(m))).await?;
        }
        if let Some(i) = info {
            self.send(SourceEvent::DemoInfo(i)).await?;
        }
        Ok(())
    }

    async fn on_event(&mut self, from: Which, event: SourceEvent) -> Result<(), HubClosed> {
        let active = self.is_active(from);
        match event {
            SourceEvent::Link(up) => {
                let side = self.side(from);
                side.link = up;
                if !up {
                    *side = Side::default();
                }
                if from == Which::Live {
                    // Back at once; gone only after the delay (and only if the demo is there to take over).
                    self.live_grace_until = if !up && self.has_demo && active && !self.fallback_delay.is_zero() {
                        Some(Instant::now() + self.fallback_delay)
                    } else {
                        None
                    };
                }
                self.reselect().await?;
            }
            // State that is cached for a later switch, and shown now if this source is on show.
            SourceEvent::MapChanged(m) => {
                self.side(from).map = Some(m.clone());
                if active {
                    self.send(SourceEvent::MapChanged(m)).await?;
                }
            }
            SourceEvent::Players(p) => {
                self.side(from).players = Some(p.clone());
                if active {
                    self.send(SourceEvent::Players(p)).await?;
                }
            }
            SourceEvent::FlyMeta(m) => {
                self.side(from).fly_meta = m.clone();
                if active {
                    self.send(SourceEvent::FlyMeta(m)).await?;
                }
            }
            SourceEvent::BotStatus(text) => match from {
                // The bot's status is the bot's, shown only while it is the source on show.
                Which::Live if active => self.send(SourceEvent::BotStatus(text)).await?,
                Which::Live => {}
                // The demo's is never a bot status: its description is rebuilt from a few short fields.
                Which::Demo => {
                    if let Some(info) = demo_info(&text) {
                        self.demo.info = Some(info.clone());
                        if active {
                            self.send(SourceEvent::DemoInfo(info)).await?;
                        }
                    }
                }
            },
            // A problem is shown for the source on show; a live bot that is not there says so when nothing else is.
            SourceEvent::Error(e) => {
                if active || (from == Which::Live && self.active == Some(SourceKind::None)) {
                    self.send(SourceEvent::Error(e)).await?;
                }
            }
            // Streams: only from the source on show.
            e @ (SourceEvent::Frame(_)
            | SourceEvent::Events { .. }
            | SourceEvent::ReplayStatus(_)
            | SourceEvent::FlyFrame(_)) => {
                if active {
                    self.send(e).await?;
                }
            }
            // A source does not choose itself: these are the multiplexer's own words.
            SourceEvent::Active(_) | SourceEvent::DemoInfo(_) => {}
        }
        Ok(())
    }
}

/// The demo's `STATUS` as the site may show it: `{"demo":true,"arena":"...","bundle":"..."}` rebuilt with only those two
/// strings (control characters removed, at most [`MAX_INFO_STR`] characters). Anything else is not a demo description.
fn demo_info(text: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(text).ok()?;
    if v.get("demo") != Some(&serde_json::Value::Bool(true)) {
        return None;
    }
    let field = |key: &str| -> String {
        v.get(key)
            .and_then(serde_json::Value::as_str)
            .map(|s| s.chars().filter(|c| !c.is_control()).take(MAX_INFO_STR).collect())
            .unwrap_or_default()
    };
    Some(serde_json::json!({ "arena": field("arena"), "bundle": field("bundle") }).to_string())
}

/// Receives from a sub-source that may have ended: `None` once, then never resolves.
async fn recv(rx: &mut Option<mpsc::Receiver<SourceEvent>>) -> Option<SourceEvent> {
    match rx.as_mut() {
        Some(r) => {
            let e = r.recv().await;
            if e.is_none() {
                *rx = None;
            }
            e
        }
        None => std::future::pending().await,
    }
}

async fn run(mut m: MuxSource, out: mpsc::Sender<SourceEvent>, mut control_rx: mpsc::Receiver<ReplayControl>) {
    // The live source gets the hub's demand as it is.
    if let Some(d) = &m.fly_demand {
        m.live.attach_fly_demand(d.clone());
    }
    if let Some(d) = &m.view_demand {
        m.live.attach_view_demand(d.clone());
    }
    let (live_tx, live_rx) = mpsc::channel(CHANNEL);
    let (_live_ctl_tx, live_ctl_rx) = mpsc::channel(1);
    let _live_task = AbortOnDrop(m.live.spawn(live_tx, live_ctl_rx));
    let mut live_rx = Some(live_rx);

    // The demo gets a gated copy, closed while the live bot is up.
    let (fly_gate, demo_fly) = Gate::new(m.fly_demand.clone());
    let (view_gate, demo_view) = Gate::new(m.view_demand.clone());
    let has_demo = m.demo.is_some();
    let mut demo_rx = None;
    let mut _demo_task = None;
    let mut _demo_ctl_tx = None;
    if let Some(mut demo) = m.demo.take() {
        demo.attach_fly_demand(demo_fly);
        demo.attach_view_demand(demo_view);
        let (tx, rx) = mpsc::channel(CHANNEL);
        let (ctl_tx, ctl_rx) = mpsc::channel(1);
        _demo_task = Some(AbortOnDrop(demo.spawn(tx, ctl_rx)));
        _demo_ctl_tx = Some(ctl_tx);
        demo_rx = Some(rx);
    }

    let mut mux = Mux {
        live: Side::default(),
        demo: Side::default(),
        has_demo,
        active: None,
        out,
        fly_gate,
        view_gate,
        fallback_delay: m.fallback_delay,
        live_grace_until: None,
    };
    // The hub's demand changes are watched here too: the demo's gate follows them.
    let mut fly_up = m.fly_demand.take();
    let mut view_up = m.view_demand.take();
    if mux.reselect().await.is_err() {
        return;
    }
    loop {
        let grace = mux.live_grace_until;
        let result = tokio::select! {
            _ = async {
                match grace {
                    Some(at) => tokio::time::sleep_until(at).await,
                    None => std::future::pending().await,
                }
            } => {
                // The live bot stayed away for the whole delay: the demo (or nothing) takes over.
                mux.live_grace_until = None;
                mux.reselect().await
            }
            e = recv(&mut live_rx) => match e {
                Some(e) => mux.on_event(Which::Live, e).await,
                // The live source ended on its own: it is gone for good.
                None => mux.on_event(Which::Live, SourceEvent::Link(false)).await,
            },
            e = recv(&mut demo_rx) => match e {
                Some(e) => mux.on_event(Which::Demo, e).await,
                None => mux.on_event(Which::Demo, SourceEvent::Link(false)).await,
            },
            c = control_rx.recv() => {
                if c.is_none() {
                    return;
                }
                Ok(())
            }
            changed = demand_changed(&mut fly_up) => {
                if !changed {
                    fly_up = None;
                }
                mux.fly_gate.apply(!mux.live_effective());
                Ok(())
            }
            changed = demand_changed(&mut view_up) => {
                if !changed {
                    view_up = None;
                }
                mux.view_gate.apply(!mux.live_effective());
                Ok(())
            }
        };
        if result.is_err() {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live::source::{CharacterState, WorldFrame};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    /// A source the test drives by hand: what it is fed comes out of its event stream; the demand it is shown is logged.
    struct Fake {
        feed: mpsc::UnboundedReceiver<SourceEvent>,
        fly: Option<watch::Receiver<bool>>,
        view: Option<watch::Receiver<bool>>,
        log: Arc<Mutex<Demand>>,
    }

    #[derive(Default)]
    struct Demand {
        fly: Vec<bool>,
        view: Vec<bool>,
    }

    struct Handle {
        tx: mpsc::UnboundedSender<SourceEvent>,
        log: Arc<Mutex<Demand>>,
    }

    impl Handle {
        fn send(&self, e: SourceEvent) {
            let _ = self.tx.send(e); // a source that is gone takes nothing
        }
        fn fly(&self) -> Vec<bool> {
            self.log.lock().unwrap().fly.clone()
        }
        fn view(&self) -> Vec<bool> {
            self.log.lock().unwrap().view.clone()
        }
    }

    fn fake() -> (Box<Fake>, Handle) {
        let (tx, feed) = mpsc::unbounded_channel();
        let log = Arc::new(Mutex::new(Demand::default()));
        (
            Box::new(Fake {
                feed,
                fly: None,
                view: None,
                log: Arc::clone(&log),
            }),
            Handle { tx, log },
        )
    }

    impl FrameSource for Fake {
        fn attach_fly_demand(&mut self, d: watch::Receiver<bool>) {
            self.fly = Some(d);
        }
        fn attach_view_demand(&mut self, d: watch::Receiver<bool>) {
            self.view = Some(d);
        }
        fn spawn(
            mut self: Box<Self>,
            events_tx: mpsc::Sender<SourceEvent>,
            mut control_rx: mpsc::Receiver<ReplayControl>,
        ) -> JoinHandle<()> {
            tokio::spawn(async move {
                let mut fly = self.fly.take();
                let mut view = self.view.take();
                if let Some(d) = fly.as_mut() {
                    self.log.lock().unwrap().fly.push(*d.borrow_and_update());
                }
                if let Some(d) = view.as_mut() {
                    self.log.lock().unwrap().view.push(*d.borrow_and_update());
                }
                loop {
                    tokio::select! {
                        e = self.feed.recv() => match e {
                            Some(e) => { if events_tx.send(e).await.is_err() { return; } }
                            None => return,
                        },
                        ok = demand_changed(&mut fly) => {
                            if ok { let v = *fly.as_mut().unwrap().borrow_and_update(); self.log.lock().unwrap().fly.push(v); } else { fly = None; }
                        }
                        ok = demand_changed(&mut view) => {
                            if ok { let v = *view.as_mut().unwrap().borrow_and_update(); self.log.lock().unwrap().view.push(v); } else { view = None; }
                        }
                        c = control_rx.recv() => if c.is_none() { return; },
                    }
                }
            })
        }
    }

    struct Rig {
        live: Handle,
        demo: Handle,
        rx: mpsc::Receiver<SourceEvent>,
        fly: watch::Sender<bool>,
        view: watch::Sender<bool>,
        _ctl: mpsc::Sender<ReplayControl>,
        _task: JoinHandle<()>,
    }

    fn rig(with_demo: bool) -> Rig {
        rig_with_delay(with_demo, Duration::ZERO)
    }

    fn rig_with_delay(with_demo: bool, delay: Duration) -> Rig {
        let (live, live_h) = fake();
        let (demo, demo_h) = fake();
        let mut mux =
            MuxSource::new(live, with_demo.then_some(demo as Box<dyn FrameSource>)).with_fallback_delay(delay);
        let (fly, fly_rx) = watch::channel(false);
        let (view, view_rx) = watch::channel(false);
        mux.attach_fly_demand(fly_rx);
        mux.attach_view_demand(view_rx);
        let (tx, rx) = mpsc::channel(64);
        let (ctl, ctl_rx) = mpsc::channel(4);
        let task = Box::new(mux).spawn(tx, ctl_rx);
        Rig {
            live: live_h,
            demo: demo_h,
            rx,
            fly,
            view,
            _ctl: ctl,
            _task: task,
        }
    }

    async fn next(rx: &mut mpsc::Receiver<SourceEvent>) -> SourceEvent {
        tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("an event within 2 s")
            .expect("the multiplexer is alive")
    }

    /// Nothing more arrives (for a short while): what a source on the bench must not cause.
    async fn quiet(rx: &mut mpsc::Receiver<SourceEvent>) {
        if let Ok(e) = tokio::time::timeout(Duration::from_millis(150), rx.recv()).await {
            panic!("unexpected event {e:?}");
        }
    }

    fn map(name: &str) -> MapMeta {
        MapMeta {
            sha256: [name.len() as u8; 32],
            name: name.to_string(),
            width: 4,
            height: 4,
        }
    }

    fn players(name: &str) -> Vec<PlayerMeta> {
        vec![PlayerMeta {
            id: 0,
            name: name.to_string(),
            team: 0,
        }]
    }

    fn frame(tick: u32) -> WorldFrame {
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

    const DEMO_STATUS: &str = r#"{"demo":true,"arena":"clb-left","bundle":"e005-fly/final"}"#;
    const DEMO_INFO: &str = r#"{"arena":"clb-left","bundle":"e005-fly/final"}"#;

    #[tokio::test]
    async fn the_demo_stands_in_while_there_is_no_bot_and_the_bot_takes_over_and_gives_back() {
        let mut r = rig(true);
        assert!(matches!(next(&mut r.rx).await, SourceEvent::Active(SourceKind::None)));

        // The demo comes up while the bot is absent: it is shown.
        r.demo.send(SourceEvent::Link(true));
        r.demo.send(SourceEvent::MapChanged(map("arena")));
        r.demo.send(SourceEvent::Players(players("fly")));
        r.demo.send(SourceEvent::BotStatus(DEMO_STATUS.to_string()));
        r.demo.send(SourceEvent::Frame(frame(1)));
        assert!(matches!(next(&mut r.rx).await, SourceEvent::Active(SourceKind::Demo)));
        assert!(matches!(next(&mut r.rx).await, SourceEvent::MapChanged(m) if m.name == "arena"));
        assert!(matches!(next(&mut r.rx).await, SourceEvent::Players(p) if p[0].name == "fly"));
        assert!(matches!(next(&mut r.rx).await, SourceEvent::DemoInfo(i) if i == DEMO_INFO));
        assert!(matches!(next(&mut r.rx).await, SourceEvent::Frame(f) if f.tick == 1));

        // The bot comes up: it takes over, the demo's frames are no longer shown.
        r.live.send(SourceEvent::Link(true));
        assert!(matches!(next(&mut r.rx).await, SourceEvent::Active(SourceKind::Live)));
        r.live.send(SourceEvent::MapChanged(map("real map")));
        r.demo.send(SourceEvent::Frame(frame(2)));
        r.demo.send(SourceEvent::MapChanged(map("another arena")));
        r.live.send(SourceEvent::Frame(frame(100)));
        r.live.send(SourceEvent::BotStatus(r#"{"mode":"fight"}"#.to_string()));
        assert!(matches!(next(&mut r.rx).await, SourceEvent::MapChanged(m) if m.name == "real map"));
        assert!(matches!(next(&mut r.rx).await, SourceEvent::Frame(f) if f.tick == 100));
        assert!(matches!(next(&mut r.rx).await, SourceEvent::BotStatus(s) if s.contains("fight")));
        quiet(&mut r.rx).await;

        // The bot goes away: the demo is back at once, with its latest state (what it sent while it was not on show).
        r.live.send(SourceEvent::Link(false));
        assert!(matches!(next(&mut r.rx).await, SourceEvent::Active(SourceKind::Demo)));
        assert!(matches!(next(&mut r.rx).await, SourceEvent::MapChanged(m) if m.name == "another arena"));
        assert!(matches!(next(&mut r.rx).await, SourceEvent::Players(p) if p[0].name == "fly"));
        assert!(matches!(next(&mut r.rx).await, SourceEvent::DemoInfo(_)));
        r.demo.send(SourceEvent::Frame(frame(3)));
        assert!(matches!(next(&mut r.rx).await, SourceEvent::Frame(f) if f.tick == 3));
        // The bot's frames do not show while it is away.
        r.live.send(SourceEvent::Frame(frame(101)));
        quiet(&mut r.rx).await;

        // And with neither: none.
        r.demo.send(SourceEvent::Link(false));
        assert!(matches!(next(&mut r.rx).await, SourceEvent::Active(SourceKind::None)));
    }

    #[tokio::test]
    async fn the_bot_wins_even_if_the_demo_was_there_first_and_a_demo_that_comes_late_does_not_displace_it() {
        let mut r = rig(true);
        assert!(matches!(next(&mut r.rx).await, SourceEvent::Active(SourceKind::None)));
        r.live.send(SourceEvent::Link(true));
        assert!(matches!(next(&mut r.rx).await, SourceEvent::Active(SourceKind::Live)));
        r.demo.send(SourceEvent::Link(true));
        r.demo.send(SourceEvent::Frame(frame(5)));
        r.demo.send(SourceEvent::BotStatus(DEMO_STATUS.to_string()));
        quiet(&mut r.rx).await;
        // Both go down at once: the demo's link going down alone does not matter either.
        r.demo.send(SourceEvent::Link(false));
        quiet(&mut r.rx).await;
        r.live.send(SourceEvent::Link(false));
        assert!(matches!(next(&mut r.rx).await, SourceEvent::Active(SourceKind::None)));
    }

    #[tokio::test]
    async fn the_demo_is_never_the_bots_status_and_its_description_is_rebuilt() {
        let mut r = rig(true);
        let _ = next(&mut r.rx).await;
        r.demo.send(SourceEvent::Link(true));
        assert!(matches!(next(&mut r.rx).await, SourceEvent::Active(SourceKind::Demo)));
        // Extra fields, control characters and a long string: only arena and bundle survive, cleaned and cut.
        let long = "x".repeat(200);
        r.demo.send(SourceEvent::BotStatus(format!(
            r#"{{"demo":true,"arena":"a\u0007b","bundle":"{long}","server":"1.2.3.4:8303","name":"Somebody"}}"#
        )));
        match next(&mut r.rx).await {
            SourceEvent::DemoInfo(i) => {
                let v: serde_json::Value = serde_json::from_str(&i).unwrap();
                assert_eq!(v["arena"], "ab");
                assert_eq!(v["bundle"].as_str().unwrap().len(), MAX_INFO_STR);
                assert_eq!(v.as_object().unwrap().len(), 2, "{i}");
            }
            other => panic!("{other:?}"),
        }
        // A STATUS that is not a demo's (no flag, not JSON) is dropped, never forwarded as a bot status.
        r.demo
            .send(SourceEvent::BotStatus(r#"{"mode":"fight","target":1}"#.to_string()));
        r.demo.send(SourceEvent::BotStatus("not json".to_string()));
        quiet(&mut r.rx).await;
    }

    #[tokio::test]
    async fn problems_are_shown_for_the_source_on_show_and_a_missing_bot_only_when_nothing_else_is() {
        let mut r = rig(true);
        let _ = next(&mut r.rx).await;
        // Nothing on show: the live source's "not running" reaches the page.
        r.live.send(SourceEvent::Error("the bot is not running".to_string()));
        assert!(matches!(next(&mut r.rx).await, SourceEvent::Error(e) if e.contains("not running")));
        r.demo.send(SourceEvent::Error("the demo is not running".to_string()));
        quiet(&mut r.rx).await;
        // The demo on show: the bot's "not running" is not news; the demo's own problems are.
        r.demo.send(SourceEvent::Link(true));
        let _ = next(&mut r.rx).await;
        r.live.send(SourceEvent::Error("the bot is not running".to_string()));
        quiet(&mut r.rx).await;
        r.demo.send(SourceEvent::Error("bad frame".to_string()));
        assert!(matches!(next(&mut r.rx).await, SourceEvent::Error(e) if e == "bad frame"));
    }

    #[tokio::test]
    async fn without_a_demo_the_live_source_passes_through_with_its_switch_messages() {
        let mut r = rig(false);
        assert!(matches!(next(&mut r.rx).await, SourceEvent::Active(SourceKind::None)));
        r.live.send(SourceEvent::Link(true));
        r.live.send(SourceEvent::Frame(frame(1)));
        assert!(matches!(next(&mut r.rx).await, SourceEvent::Active(SourceKind::Live)));
        assert!(matches!(next(&mut r.rx).await, SourceEvent::Frame(_)));
        r.live.send(SourceEvent::Link(false));
        assert!(matches!(next(&mut r.rx).await, SourceEvent::Active(SourceKind::None)));
        // A "demo" socket that was never configured cannot appear.
        r.demo.send(SourceEvent::Link(true));
        quiet(&mut r.rx).await;
    }

    #[tokio::test]
    async fn a_source_that_claims_to_be_the_multiplexer_is_ignored() {
        let mut r = rig(true);
        let _ = next(&mut r.rx).await;
        r.demo.send(SourceEvent::Active(SourceKind::Live));
        r.demo.send(SourceEvent::DemoInfo("{}".to_string()));
        r.live.send(SourceEvent::Active(SourceKind::Demo));
        quiet(&mut r.rx).await;
    }

    #[tokio::test]
    async fn a_source_that_ends_counts_as_gone() {
        let mut r = rig(true);
        let _ = next(&mut r.rx).await;
        r.demo.send(SourceEvent::Link(true));
        assert!(matches!(next(&mut r.rx).await, SourceEvent::Active(SourceKind::Demo)));
        // The demo's task ends (its feed closes): the page falls back to nothing.
        let Rig { demo, mut rx, .. } = r;
        drop(demo);
        assert!(matches!(next(&mut rx).await, SourceEvent::Active(SourceKind::None)));
    }

    #[tokio::test]
    async fn the_demo_is_asked_for_viewers_only_while_the_bot_is_not_up() {
        let mut r = rig(true);
        let _ = next(&mut r.rx).await;
        let settle = || tokio::time::sleep(Duration::from_millis(80));
        settle().await;
        assert_eq!(
            r.demo.view(),
            [false],
            "nobody watches: the demo is told nothing is wanted"
        );
        assert_eq!(r.live.view(), [false]);

        // A browser connects; the bot is absent: both are asked (the demo is the one that matters).
        r.view.send(true).unwrap();
        r.fly.send(true).unwrap();
        settle().await;
        assert_eq!(r.live.view(), [false, true]);
        assert_eq!(r.demo.view(), [false, true]);
        assert_eq!(r.demo.fly(), [false, true]);

        // The bot comes up: the demo is no longer asked (so it pauses); the bot still is.
        r.live.send(SourceEvent::Link(true));
        let _ = next(&mut r.rx).await;
        settle().await;
        assert_eq!(r.demo.view(), [false, true, false]);
        assert_eq!(r.demo.fly(), [false, true, false]);
        assert_eq!(r.live.view(), [false, true]);

        // The viewer leaves and returns while the bot is up: the demo stays unasked.
        r.view.send(false).unwrap();
        r.view.send(true).unwrap();
        settle().await;
        assert_eq!(r.demo.view(), [false, true, false]);

        // The bot goes: the demo is asked again at once.
        r.live.send(SourceEvent::Link(false));
        let _ = next(&mut r.rx).await;
        settle().await;
        assert_eq!(r.demo.view(), [false, true, false, true]);
        assert_eq!(r.demo.fly(), [false, true, false, true]);
    }

    /// Task 5.7 (review F3): falling back waits out the delay and a link that returns in time changes nothing; going to the
    /// live bot is at once; the demo is not asked for viewers while the bot is only briefly away.
    #[tokio::test]
    async fn the_fall_back_to_the_demo_waits_but_the_switch_to_the_bot_does_not() {
        let delay = Duration::from_millis(500);
        let mut r = rig_with_delay(true, delay);
        let _ = next(&mut r.rx).await;
        r.view.send(true).unwrap();
        // The demo is up first and shown; the bot arriving takes over at once, not after the delay.
        r.demo.send(SourceEvent::Link(true));
        assert!(matches!(next(&mut r.rx).await, SourceEvent::Active(SourceKind::Demo)));
        let t = std::time::Instant::now();
        r.live.send(SourceEvent::Link(true));
        assert!(matches!(next(&mut r.rx).await, SourceEvent::Active(SourceKind::Live)));
        assert!(
            t.elapsed() < Duration::from_millis(300),
            "the bot wins at once: {:?}",
            t.elapsed()
        );

        // A short break: nothing changes on the page, and the demo is not asked to play.
        r.live.send(SourceEvent::Link(false));
        tokio::time::sleep(Duration::from_millis(200)).await;
        r.live.send(SourceEvent::Link(true));
        quiet(&mut r.rx).await;
        assert_eq!(
            r.demo.view().last(),
            Some(&false),
            "the demo stays at rest during a short break"
        );

        // A real loss: the demo comes after the delay, not before.
        let t = std::time::Instant::now();
        r.live.send(SourceEvent::Link(false));
        tokio::time::sleep(Duration::from_millis(250)).await;
        assert_eq!(r.demo.view().last(), Some(&false), "still at rest inside the delay");
        assert!(matches!(next(&mut r.rx).await, SourceEvent::Active(SourceKind::Demo)));
        assert!(
            t.elapsed() >= Duration::from_millis(450),
            "after the delay: {:?}",
            t.elapsed()
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(r.demo.view().last(), Some(&true), "now the demo is asked to play");
    }

    #[test]
    fn a_demo_description_needs_the_demo_flag_and_keeps_two_strings() {
        assert_eq!(demo_info(DEMO_STATUS).as_deref(), Some(DEMO_INFO));
        assert!(demo_info(r#"{"arena":"x"}"#).is_none());
        assert!(demo_info(r#"{"demo":"yes","arena":"x"}"#).is_none());
        assert!(demo_info("[]").is_none());
        // Missing or wrong-typed fields become empty strings, never a failure of the stream.
        let v: serde_json::Value = serde_json::from_str(&demo_info(r#"{"demo":true,"arena":5}"#).unwrap()).unwrap();
        assert_eq!((v["arena"].as_str(), v["bundle"].as_str()), (Some(""), Some("")));
    }
}
