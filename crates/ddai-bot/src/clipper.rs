//! The bot's side of the clips (task 4.3): every snapshot becomes a frame of the 30 s ring
//! ([`ddai_clip::Recorder`], no allocation in the steady state), real events are derived from what the
//! snapshots show, and the autoclip scan / cross-fail clip / pruning run on a small worker thread so the
//! decision path never waits for them (D-042: 5 ms p99).
//!
//! **What a frame holds** (`docs/formats.md` §24): our own tee, every tee hooked to or by us and the nearest
//! others up to [`ddai_clip::MAX_TEES`] (the count left out is recorded, the replay names it as a possible
//! cause), the projectiles and switch states, the inputs we actually SENT for the ticks since the previous
//! frame, what the bot knew and did ([`BotRec`]) and the events.
//!
//! **Sent inputs, one frame late.** The server's timing reports (`NETMSG_INPUTTIMING`) for the inputs of
//! frame `i` arrive after frame `i` was recorded, and they decide which tick a late input really took
//! effect on ([`crate::sent`]). So frame `i`'s inputs are written provisionally when it is recorded and
//! **re-read when frame `i + 1` is recorded** (`Recorder::amend_last_sent`), by which time the reports
//! are in. Only the newest frame of a saved clip can still hold a provisional input; its
//! `timing_known` flag says so.
//!
//! **Real events.** The TS recorded none live (`orig-bot.md` §13 bug 8). Here they are derived from the
//! tee states of consecutive snapshots with the same rules the activity clock uses: a hook attach and
//! release, a hammer swing (the attack tick moved while holding the hammer) and the tees its 56 px reach
//! covers, a freeze onset; plus `SV_KILLMSG` and our own `Cl_Kill`s, pushed by the bot.
//!
//! **Death.** A frame is recorded for [`TAIL_FRAMES`] snapshots after our tee disappears (`own_alive: false`),
//! so the `death` incident can see the tee vanish.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use ddai_client::LiveWorldSnapshot;
use ddai_clip::format::{
    BotRec, Clip, ClipEvent, ClipReason, DdRec, InputRec, MAX_EVENTS, MAX_SWITCHES, MAX_TEES, PlayerTag, RING_FRAMES,
    SentRec, SwitchRec, TeeRec,
};
use ddai_clip::record::{ClipMeta, Recorder};
use ddai_clip::store::{self, AutoClip};
use ddai_physics::core::MAX_CLIENTS;
use ddai_physics::vmath::Vec2;

use crate::consts::{HAMMER_REACH_AHEAD_PX, HAMMER_REACH_PX};
use crate::players::PlayerTable;
use crate::sent::SentLog;
use crate::tees::{HOOK_GRABBED, Tee, TeeSet, dist};

/// Frames recorded after our own tee is gone (a death), so `death` can see it vanish.
pub const TAIL_FRAMES: u8 = 6;

/// Two round ends closer than this are one (a round is at least the 150-tick countdown long).
pub const DUEL_ROUND_GAP_TICKS: i32 = 100;

/// A duel round-loss clip needs at least this many frames in the ring (the autoclip's 50 would drop the early rounds of a session).
pub const DUEL_MIN_FRAMES: usize = 20;

/// Where clips go and whether the autoclip runs.
#[derive(Debug, Clone)]
pub struct ClipConfig {
    /// The clip directory; `None` writes nothing (the ring still records: `!clip` then says it cannot save).
    pub dir: Option<PathBuf>,
    /// The automatic clips (incidents, cross-fail).
    pub autoclip: bool,
    /// Scan and save on a worker thread (the live runner does; the sans-IO scenario tests scan inline).
    pub async_save: bool,
}

impl Default for ClipConfig {
    fn default() -> Self {
        ClipConfig {
            dir: None,
            autoclip: true,
            async_save: false,
        }
    }
}

/// A clip that reached the disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SavedClip {
    pub path: PathBuf,
    /// An incident kind, `cross-fail` or `manual`.
    pub kind: String,
    pub severity: i32,
    pub note: String,
    pub tick: i32,
}

/// What the previous snapshot showed of one tee.
#[derive(Debug, Clone, Copy, Default)]
struct Prev {
    valid: bool,
    frozen: bool,
    hook_state: i32,
    hooked: i32,
    attack: i32,
    hook_since: i32,
}

// ---- the worker -----------------------------------------------------------------------------------

enum Job {
    /// The ring as a clip: scan it for an incident worth saving.
    Scan(Box<Clip>),
    /// The ring as a clip, saved because a navigation crossing failed (`note`).
    Cross(Box<Clip>, String),
    /// Task 3.19: the ring as a clip, saved because the bot ended a duel round frozen (no cooldown; bounded by the caller and by `store::prune_duel`).
    Duel(Box<Clip>),
}

/// The state of the automatic saving: the cooldowns and the directory. Lives on the worker thread (or
/// inline in the synchronous mode).
struct AutoState {
    dir: PathBuf,
    auto: AutoClip,
}

impl AutoState {
    fn handle(&mut self, job: Job) -> Option<SavedClip> {
        match job {
            Job::Scan(mut clip) => {
                let tick = clip.frames.last()?.tick;
                if !self.auto.ready(tick) {
                    return None;
                }
                let inc = store::scan(&clip)?;
                clip.header.reason = ClipReason {
                    kind: inc.kind.to_string(),
                    severity: inc.severity,
                    tick: inc.tick,
                    note: inc.note.clone(),
                };
                let name = store::auto_name(inc.kind, inc.tick, inc.severity);
                let path = self.write(&clip, &name)?;
                self.auto.clipped(tick);
                Some(SavedClip {
                    path,
                    kind: inc.kind.to_string(),
                    severity: inc.severity,
                    note: inc.note,
                    tick: inc.tick,
                })
            }
            Job::Duel(mut clip) => {
                let tick = clip.frames.last()?.tick;
                clip.header.reason = ClipReason {
                    kind: store::DUEL_KIND.to_string(),
                    severity: 0,
                    tick,
                    note: "a duel round ended with us frozen".to_string(),
                };
                let path = match store::save(&self.dir, &clip, &store::duel_name(tick)) {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::warn!(error = %e, "could not write the duel clip");
                        return None;
                    }
                };
                if let Err(e) = store::prune_duel(&self.dir) {
                    tracing::warn!(error = %e, "pruning the duel clips failed");
                }
                Some(SavedClip {
                    path,
                    kind: store::DUEL_KIND.to_string(),
                    severity: 0,
                    note: "round lost".to_string(),
                    tick,
                })
            }
            Job::Cross(mut clip, note) => {
                let tick = clip.frames.last()?.tick;
                if !self.auto.cross_ready(tick) || clip.frames.len() < store::MIN_FRAMES {
                    return None;
                }
                clip.header.reason = ClipReason {
                    kind: "cross-fail".to_string(),
                    severity: 0,
                    tick,
                    note: note.clone(),
                };
                let name = store::auto_name("cross-fail", tick, 0);
                let path = self.write(&clip, &name)?;
                self.auto.cross_clipped(tick);
                Some(SavedClip {
                    path,
                    kind: "cross-fail".to_string(),
                    severity: 0,
                    note,
                    tick,
                })
            }
        }
    }

    fn write(&self, clip: &Clip, name: &str) -> Option<PathBuf> {
        match store::save(&self.dir, clip, name) {
            Ok(p) => {
                if let Err(e) = store::prune(&self.dir) {
                    tracing::warn!(error = %e, "pruning the clips failed");
                }
                Some(p)
            }
            Err(e) => {
                tracing::warn!(error = %e, "could not write the clip");
                None
            }
        }
    }
}

struct Worker {
    jobs: mpsc::SyncSender<Job>,
    saved: mpsc::Receiver<SavedClip>,
    /// A job is queued or running (set by the sender, cleared by the worker).
    busy: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Worker {
    fn spawn(state: AutoState) -> Option<Worker> {
        let (jobs, rx) = mpsc::sync_channel::<Job>(1);
        let (tx, saved) = mpsc::channel::<SavedClip>();
        let busy = Arc::new(AtomicBool::new(false));
        let worker_busy = Arc::clone(&busy);
        let handle = thread::Builder::new()
            .name("clip-saver".into())
            .spawn(move || {
                let mut state = state;
                for job in rx {
                    if let Some(s) = state.handle(job) {
                        let _ = tx.send(s);
                    }
                    worker_busy.store(false, Ordering::Release);
                }
            })
            .ok()?;
        Some(Worker {
            jobs,
            saved,
            busy,
            handle: Some(handle),
        })
    }
}

enum Saver {
    Off,
    Inline(AutoState),
    Thread(Worker),
}

// ---- the clipper ----------------------------------------------------------------------------------

/// What the bot hands over for one snapshot.
pub struct FrameInput<'a> {
    pub snap: &'a LiveWorldSnapshot,
    pub tees: &'a TeeSet,
    pub players: &'a PlayerTable,
    pub sent: &'a SentLog,
    pub own_id: i32,
    /// The bot's record without the walk label (the clipper interns it).
    pub bot: BotRec,
    /// The navigation's walk label ("" when not walking).
    pub walk_label: &'a str,
    /// Task 3.19: an F-DDrace duel is detected ([`crate::duel`]): a round that ends with us frozen is saved as a clip, whatever the cooldown.
    pub duel: bool,
}

/// The recorder, the event derivation and the automatic saving.
pub struct Clipper {
    cfg: ClipConfig,
    rec: Recorder,
    scan: AutoClip,
    saver: Saver,
    prev: Box<[Prev; MAX_CLIENTS]>,
    /// Events the bot reported (kill messages, our `Cl_Kill`s), waiting for the next frame.
    pending: Vec<ClipEvent>,
    own_was_alive: bool,
    tail_left: u8,
    /// Ticks of the newest and the previous recorded frame.
    last_tick: Option<i32>,
    before_last_tick: Option<i32>,
    last_saved_tick: i32,
    /// A scan was due while the worker was still busy: try again at the next frame.
    retry_scan: bool,
    label: String,
    label_idx: u16,
    map_name: String,
    map_sha256: [u8; 32],
    brain: String,
    sent_scratch: Vec<SentRec>,
    /// Clips that reached the disk since the last [`Clipper::take_saved`].
    saved: Vec<SavedClip>,
    /// Task 3.19: our tee was frozen in the last frame it was alive in (a round ends with the respawn of both tees).
    own_frozen_last: bool,
    /// The round-loss clips this run has had (capped at [`store::DUEL_SESSION_MAX`]; a new duel does not reset it) and the tick of the last round end clipped.
    duel_saved: usize,
    last_duel_clip_tick: i32,
    /// A round-loss clip the worker was too busy to take: offered again at the next frame.
    duel_pending: Option<Box<Clip>>,
    /// Task 3.19: the hammer hits the snapshots showed, by us and on us (the clips' rule: a swing with the other tee in its reach); for the duel journal.
    hits_by_own: u64,
    hits_on_own: u64,
}

impl Clipper {
    pub fn new(cfg: ClipConfig, world_seed: u64) -> Clipper {
        let mut rec = Recorder::new(RING_FRAMES);
        rec.set_world_seed(world_seed);
        let saver = match (&cfg.dir, cfg.autoclip) {
            (Some(dir), true) => {
                let state = AutoState {
                    dir: dir.clone(),
                    auto: AutoClip::default(),
                };
                if cfg.async_save {
                    Worker::spawn(state).map_or(Saver::Off, Saver::Thread)
                } else {
                    Saver::Inline(state)
                }
            }
            _ => Saver::Off,
        };
        let mut scan = AutoClip::default();
        scan.enabled = cfg.autoclip;
        Clipper {
            cfg,
            rec,
            scan,
            saver,
            prev: Box::new([Prev::default(); MAX_CLIENTS]),
            pending: Vec::with_capacity(MAX_EVENTS),
            own_was_alive: false,
            tail_left: 0,
            last_tick: None,
            before_last_tick: None,
            last_saved_tick: i32::MIN / 2,
            retry_scan: false,
            label: String::new(),
            label_idx: 0,
            map_name: String::new(),
            map_sha256: [0; 32],
            brain: String::new(),
            sent_scratch: Vec::with_capacity(8),
            saved: Vec::new(),
            own_frozen_last: false,
            duel_saved: 0,
            last_duel_clip_tick: i32::MIN / 2,
            duel_pending: None,
            hits_by_own: 0,
            hits_on_own: 0,
        }
    }

    pub fn config(&self) -> &ClipConfig {
        &self.cfg
    }

    pub fn dir(&self) -> Option<&Path> {
        self.cfg.dir.as_deref()
    }

    pub fn frames(&self) -> usize {
        self.rec.len()
    }

    /// Task 3.19: the hammer hits seen so far, by us and on us (counted from the recorded frames' events).
    pub fn hammer_hits(&self) -> (u64, u64) {
        (self.hits_by_own, self.hits_on_own)
    }

    pub fn autoclip_enabled(&self) -> bool {
        self.scan.enabled
    }

    /// A new map, a tick reset or a lost connection: the ring and everything derived from it start over.
    pub fn reset(&mut self) {
        self.rec.clear();
        self.scan.reset();
        for p in self.prev.iter_mut() {
            *p = Prev::default();
        }
        self.pending.clear();
        self.own_was_alive = false;
        self.tail_left = 0;
        self.last_tick = None;
        self.before_last_tick = None;
        self.own_frozen_last = false;
        self.duel_pending = None;
        self.last_duel_clip_tick = i32::MIN / 2;
    }

    /// Something the bot did or heard that belongs in the next frame (`SV_KILLMSG`, our `Cl_Kill`).
    pub fn push_event(&mut self, e: ClipEvent) {
        if self.pending.len() < MAX_EVENTS {
            self.pending.push(e);
        }
    }

    /// The map the ring belongs to (named in every clip; its bytes never are).
    pub fn set_map(&mut self, name: &str, sha256: [u8; 32]) {
        self.map_name.clear();
        self.map_name.push_str(name);
        self.map_sha256 = sha256;
    }

    /// The brain's name for the clip header.
    pub fn set_brain(&mut self, name: &str) {
        self.brain.clear();
        self.brain.push_str(name);
    }

    fn collect_saved(&mut self) {
        if let Saver::Thread(w) = &self.saver {
            while let Ok(s) = w.saved.try_recv() {
                if s.kind != store::DUEL_KIND {
                    self.last_saved_tick = self.last_saved_tick.max(s.tick);
                }
                self.saved.push(s);
            }
        }
    }

    /// Waits for the worker to finish what it was given and keeps its results (the run is over).
    pub fn finish(&mut self) {
        // A duel round-loss clip still waiting for the worker goes first.
        self.flush();
        if let Saver::Thread(w) = std::mem::replace(&mut self.saver, Saver::Off) {
            let Worker {
                jobs, saved, handle, ..
            } = w;
            drop(jobs);
            if let Some(h) = handle {
                let _ = h.join();
            }
            while let Ok(s) = saved.try_recv() {
                self.last_saved_tick = self.last_saved_tick.max(s.tick);
                self.saved.push(s);
            }
        }
    }

    /// Clips saved by the automatic path since the last call.
    pub fn take_saved(&mut self) -> Vec<SavedClip> {
        self.collect_saved();
        std::mem::take(&mut self.saved)
    }

    /// Records one snapshot. Returns nothing: automatic clips come back through [`Clipper::take_saved`].
    pub fn record(&mut self, f: &FrameInput<'_>) {
        let tick = f.snap.tick;
        let own = f.tees.get(f.own_id).copied();
        let own_alive = own.is_some();
        if self.last_tick.is_some_and(|t| tick <= t) {
            return; // a repeated tick: nothing new to record
        }
        // Our tee is gone: record a few more frames (the death), then stop until it is back.
        if own_alive {
            self.tail_left = TAIL_FRAMES;
        } else if self.tail_left == 0 {
            // Keep the deriver in step without recording.
            derive_events(&mut self.prev, tick, f.own_id, f.tees, &mut |_| {});
            self.own_was_alive = false;
            return;
        } else {
            self.tail_left -= 1;
        }

        // The previous frame's inputs, now that their timing reports are in.
        if let (Some(last), false) = (self.last_tick, self.rec.is_empty()) {
            let from = self.before_last_tick.unwrap_or(last - 1);
            Self::sent_between(f.sent, from, last, &mut self.sent_scratch);
            self.rec.amend_last_sent(&self.sent_scratch);
        }

        self.rec.note_tuning(tick, &f.snap.tuning);
        if let Some(t) = &f.snap.teams {
            self.rec.note_teams(tick, t);
        }

        // Events first (they borrow `self.prev`), into a small stack buffer.
        let mut evs = [ClipEvent::Respawn { id: 0 }; MAX_EVENTS];
        let mut n_ev = 0;
        let respawn = own_alive && !self.own_was_alive && self.last_tick.is_some();
        // Task 3.19: a duel round ends with both tees killed and respawned in the same server step (`KillParticipants`), so a snapshot without our tee need
        // not exist: the round end is our `Kill` message (the 07.10 clips: two of four round ends show no dead frame) or, when the tee did go missing, its
        // return. The one that ends it frozen is the loser (or it is a draw): that round is clipped. Rounds are at least a countdown (150 ticks) apart, so a
        // second signal of the same round (the message, then the respawn) is no second clip.
        let own_killed = self
            .pending
            .iter()
            .any(|e| matches!(e, ClipEvent::Kill { victim, .. } if *victim == f.own_id));
        let lost_round = (respawn || own_killed)
            && f.duel
            && self.own_frozen_last
            && tick.saturating_sub(self.last_duel_clip_tick) > DUEL_ROUND_GAP_TICKS;
        {
            let mut push = |e: ClipEvent| {
                if n_ev < MAX_EVENTS {
                    evs[n_ev] = e;
                    n_ev += 1;
                }
            };
            if respawn {
                push(ClipEvent::Respawn { id: f.own_id });
            }
            for e in &self.pending {
                push(*e);
            }
            derive_events(&mut self.prev, tick, f.own_id, f.tees, &mut push);
        }
        self.pending.clear();
        self.own_was_alive = own_alive;

        if f.walk_label != self.label {
            self.label.clear();
            self.label.push_str(f.walk_label);
            self.label_idx = self.rec.intern(f.walk_label);
        }
        let mut bot = f.bot;
        bot.walk = if f.walk_label.is_empty() { 0 } else { self.label_idx };

        let from = self.last_tick.unwrap_or(tick - 1);
        Self::sent_between(f.sent, from, tick, &mut self.sent_scratch);

        let mut b = self.rec.begin(tick, own_alive);
        let mut recorded = [-1i32; MAX_TEES];
        let n = select_tees(own.as_ref(), f.tees, &mut recorded);
        for &id in &recorded[..n] {
            if let (Some(cv), Some(t)) = (f.snap.characters.iter().find(|c| c.id == id), f.tees.get(id)) {
                b.add_tee(TeeRec {
                    id,
                    ch: ddai_clip::CharRec::from_net(&cv.character),
                    dd: cv.ddnet.as_ref().map(DdRec::from_net),
                    frozen: t.frozen,
                    deep_frozen: t.deep_frozen,
                    freeze_left: t.freeze_ticks_left,
                });
            }
        }
        b.set_tees_dropped(f.snap.characters.len().saturating_sub(n));
        for (id, p) in &f.snap.projectiles {
            b.add_projectile(ddai_clip::ProjRec::from_view(*id, p));
        }
        for (team, s) in f.snap.switch_states.iter().take(MAX_SWITCHES) {
            b.add_switch(SwitchRec::from_net(*team, s));
        }
        for r in &self.sent_scratch {
            b.add_sent(*r);
        }
        for e in &evs[..n_ev] {
            if let ClipEvent::HammerHit { from, to } = *e {
                self.hits_by_own += u64::from(from == f.own_id);
                self.hits_on_own += u64::from(to == f.own_id);
            }
            b.add_event(*e);
        }
        b.set_bot(bot);
        b.finish();
        self.before_last_tick = self.last_tick;
        self.last_tick = Some(tick);
        if own_alive {
            self.own_frozen_last = own.is_some_and(|t| t.frozen);
        }
        if lost_round {
            self.last_duel_clip_tick = tick;
            self.queue_duel_clip(f.own_id, f.players);
        }
        self.offer_duel_clip();

        self.autoscan(tick, f.own_id, f.players);
    }

    /// Task 3.19: the ring, as the clip of the duel round we just lost. At most [`store::DUEL_SESSION_MAX`] per duel; no cooldown.
    fn queue_duel_clip(&mut self, own_id: i32, players: &PlayerTable) {
        if matches!(self.saver, Saver::Off) || !self.cfg.autoclip || self.duel_saved >= store::DUEL_SESSION_MAX {
            return;
        }
        if self.rec.len() < DUEL_MIN_FRAMES {
            return;
        }
        if let Some(clip) = self.rec.to_clip(self.meta(own_id, players, store::DUEL_KIND), None) {
            self.duel_pending = Some(Box::new(clip));
        }
    }

    /// Hands a waiting round-loss clip to the saver; a busy worker keeps it for the next frame (a duel clip is never dropped for a scan).
    fn offer_duel_clip(&mut self) {
        let Some(clip) = self.duel_pending.take() else {
            return;
        };
        match &mut self.saver {
            Saver::Off => {}
            Saver::Inline(state) => {
                self.duel_saved += 1;
                if let Some(s) = state.handle(Job::Duel(clip)) {
                    self.saved.push(s);
                }
            }
            Saver::Thread(w) => {
                if w.busy.swap(true, Ordering::AcqRel) {
                    self.duel_pending = Some(clip);
                    return;
                }
                match w.jobs.try_send(Job::Duel(clip)) {
                    Ok(()) => self.duel_saved += 1,
                    Err(mpsc::TrySendError::Full(Job::Duel(c)) | mpsc::TrySendError::Disconnected(Job::Duel(c))) => {
                        w.busy.store(false, Ordering::Release);
                        self.duel_pending = Some(c);
                    }
                    Err(_) => w.busy.store(false, Ordering::Release),
                }
            }
        }
    }

    /// The sent inputs for the ticks `(from, to]` (the last [`ddai_clip::MAX_SENT`] of them).
    fn sent_between(sent: &SentLog, from: i32, to: i32, out: &mut Vec<SentRec>) {
        out.clear();
        let first = (from + 1).max(to - (ddai_clip::MAX_SENT as i32 - 1));
        for t in first..=to {
            if let Some((input, known)) = sent.effective_with_timing(t) {
                out.push(SentRec {
                    tick: t,
                    input: InputRec::from_net(&ddai_world::player_input_to_net(input)),
                    timing_known: known,
                });
            }
        }
    }

    fn autoscan(&mut self, tick: i32, own_id: i32, players: &PlayerTable) {
        let due = self.scan.frame() || self.retry_scan;
        if !due || matches!(self.saver, Saver::Off) {
            self.retry_scan = false;
            return;
        }
        if let Saver::Thread(w) = &self.saver
            && w.busy.load(Ordering::Acquire)
        {
            self.retry_scan = true;
            return;
        }
        self.retry_scan = false;
        self.collect_saved();
        let cooling = tick >= self.last_saved_tick && tick - self.last_saved_tick < store::COOLDOWN_TICKS;
        if cooling || self.rec.len() < store::MIN_FRAMES {
            return;
        }
        let Some(clip) = self.rec.to_clip(self.meta(own_id, players, "scan"), None) else {
            return;
        };
        self.submit(Job::Scan(Box::new(clip)));
    }

    fn submit(&mut self, job: Job) {
        match &mut self.saver {
            Saver::Off => {}
            Saver::Inline(state) => {
                if let Some(s) = state.handle(job) {
                    self.last_saved_tick = self.last_saved_tick.max(s.tick);
                    self.saved.push(s);
                }
            }
            Saver::Thread(w) => {
                // A busy worker drops this job (a scan is retried at the next frame, a cross-fail
                // clip has its minute of cooldown anyway).
                if w.busy.swap(true, Ordering::AcqRel) {
                    return;
                }
                if w.jobs.try_send(job).is_err() {
                    w.busy.store(false, Ordering::Release);
                }
            }
        }
    }

    /// Waits (at most 5 s) until the worker has nothing queued: tests that need a deterministic order,
    /// and the end of a run.
    pub fn flush(&mut self) {
        let t0 = Instant::now();
        loop {
            // A round-loss clip the worker was too busy for goes out as soon as it can.
            self.offer_duel_clip();
            let busy = matches!(&self.saver, Saver::Thread(w) if w.busy.load(Ordering::Acquire));
            if !(busy || self.duel_pending.is_some()) || t0.elapsed() >= Duration::from_secs(5) {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        self.collect_saved();
    }

    /// A navigation crossing failed (`note`): save the ring (subject to the 60 s cooldown, in the saver).
    pub fn cross_fail(&mut self, note: &str, own_id: i32, players: &PlayerTable) {
        if matches!(self.saver, Saver::Off) || self.rec.len() < store::MIN_FRAMES {
            return;
        }
        let Some(clip) = self.rec.to_clip(self.meta(own_id, players, "cross-fail"), None) else {
            return;
        };
        self.submit(Job::Cross(Box::new(clip), note.to_string()));
    }

    /// `!clip [note]`: the ring (the newest `last_frames`, all by default) saved as `manual-<tick>[-<note>]`.
    pub fn save_manual(
        &mut self,
        note: &str,
        own_id: i32,
        players: &PlayerTable,
        last_frames: Option<usize>,
    ) -> Result<SavedClip, String> {
        let Some(dir) = self.cfg.dir.clone() else {
            return Err("no clip directory is set (--clips-dir)".to_string());
        };
        let tick = self.last_tick.ok_or("nothing recorded yet")?;
        let mut meta = self.meta(own_id, players, "manual");
        meta.reason.tick = tick;
        meta.reason.note = note.trim().to_string();
        let clip = self.rec.to_clip(meta, last_frames).ok_or("nothing recorded yet")?;
        let path = store::save(&dir, &clip, &store::manual_name(tick, note)).map_err(|e| e.to_string())?;
        Ok(SavedClip {
            path,
            kind: "manual".to_string(),
            severity: 0,
            note: note.trim().to_string(),
            tick,
        })
    }

    fn meta(&self, own_id: i32, players: &PlayerTable, kind: &str) -> ClipMeta {
        ClipMeta {
            map_name: self.map_name.clone(),
            map_sha256: self.map_sha256,
            own_id,
            brain: self.brain.clone(),
            reason: ClipReason {
                kind: kind.to_string(),
                severity: 0,
                tick: self.last_tick.unwrap_or(0),
                note: String::new(),
            },
            players: players
                .present()
                .map(|(id, _)| PlayerTag {
                    id,
                    tag: players.tag(id).to_string(),
                })
                .collect(),
        }
    }
}

/// The tees a frame records: ours, the nearest seven others, and every other tee linked to us by a hook
/// (either way: physics depends on those whatever their distance), up to [`MAX_TEES`]. Returns how many
/// ids were written to `out`.
fn select_tees(own: Option<&Tee>, tees: &TeeSet, out: &mut [i32; MAX_TEES]) -> usize {
    let mut n = 0;
    let mut chosen = [false; MAX_CLIENTS];
    let mut take = |id: i32, n: &mut usize, chosen: &mut [bool; MAX_CLIENTS]| {
        if *n < MAX_TEES && id >= 0 && (id as usize) < MAX_CLIENTS && !chosen[id as usize] {
            chosen[id as usize] = true;
            out[*n] = id;
            *n += 1;
        }
    };
    let Some(own) = own else {
        // No tee of ours (a death frame): the lowest ids, there is nothing to be near.
        for t in tees.iter().take(NEAREST + 1) {
            take(t.id, &mut n, &mut chosen);
        }
        return n;
    };
    take(own.id, &mut n, &mut chosen);
    for _ in 0..NEAREST {
        let mut best: Option<(f32, i32)> = None;
        for t in tees.iter() {
            if chosen[t.id as usize] {
                continue;
            }
            let d = dist(t.pos, own.pos);
            if best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, t.id)); // ascending ids: the first of equals wins
            }
        }
        let Some((_, id)) = best else { break };
        take(id, &mut n, &mut chosen);
    }
    for t in tees.iter() {
        if t.id != own.id && (t.hooked_player == own.id || own.hooked_player == t.id) {
            take(t.id, &mut n, &mut chosen);
        }
    }
    n
}

/// How many nearest others a frame records besides the tees linked by a hook.
const NEAREST: usize = 7;

/// Events from the difference of two snapshots' tees: hook attach / release, hammer swings, freeze onsets.
/// Our own tee first, then ascending ids (the frame's event list is capped, the newest lose).
fn derive_events(
    prev: &mut [Prev; MAX_CLIENTS],
    tick: i32,
    own_id: i32,
    tees: &TeeSet,
    push: &mut impl FnMut(ClipEvent),
) {
    let order = tees
        .get(own_id)
        .into_iter()
        .chain(tees.iter().filter(|t| t.id != own_id));
    for a in order {
        let i = a.id as usize;
        let p = prev[i];
        if p.valid {
            let grabbed = a.hook_state == HOOK_GRABBED;
            let was_grabbed = p.hook_state == HOOK_GRABBED;
            if was_grabbed && (!grabbed || a.hooked_player != p.hooked) {
                push(ClipEvent::HookRelease {
                    id: a.id,
                    target: p.hooked,
                    held: tick - p.hook_since,
                });
            }
            if grabbed && (!was_grabbed || a.hooked_player != p.hooked) {
                push(ClipEvent::HookAttach {
                    id: a.id,
                    target: a.hooked_player,
                });
            }
            if a.frozen && !p.frozen {
                push(ClipEvent::FreezeOnset { id: a.id });
            }
            if a.attack_tick != p.attack && a.holding_hammer() {
                let r = a.aim_rad();
                let point = Vec2::new(
                    a.pos.x + r.cos() * HAMMER_REACH_AHEAD_PX,
                    a.pos.y + r.sin() * HAMMER_REACH_AHEAD_PX,
                );
                let mut hits = 0u8;
                for b in tees.iter() {
                    if b.id != a.id && dist(point, b.pos) < HAMMER_REACH_PX {
                        hits = hits.saturating_add(1);
                        push(ClipEvent::HammerHit { from: a.id, to: b.id });
                    }
                }
                push(ClipEvent::HammerFire { from: a.id, hits });
            }
        }
        let grabbed = a.hook_state == HOOK_GRABBED;
        let hook_since = if grabbed && (!p.valid || p.hook_state != HOOK_GRABBED || p.hooked != a.hooked_player) {
            tick
        } else {
            p.hook_since
        };
        prev[i] = Prev {
            valid: true,
            frozen: a.frozen,
            hook_state: a.hook_state,
            hooked: a.hooked_player,
            attack: a.attack_tick,
            hook_since,
        };
    }
    // Gone tees: a hook they held ends with them.
    for (i, p) in prev.iter_mut().enumerate() {
        if p.valid && tees.get(i as i32).is_none() {
            if p.hook_state == HOOK_GRABBED {
                push(ClipEvent::HookRelease {
                    id: i as i32,
                    target: p.hooked,
                    held: tick - p.hook_since,
                });
            }
            *p = Prev::default();
        }
    }
}

impl Drop for Clipper {
    fn drop(&mut self) {
        self.finish();
    }
}
