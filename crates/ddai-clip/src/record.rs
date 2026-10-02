//! The ring recorder: the last [`RING_FRAMES`](crate::format::RING_FRAMES) snapshot frames of the bot,
//! kept in fixed storage so that recording a frame **allocates nothing** in the steady state (a test
//! pins that with an allocation counter). Strings (walk labels) are interned once; tuning and teams are
//! stored when they change.
//!
//! The recorder is the bot's side of the format: the bot opens a frame with [`Recorder::begin`], fills
//! it, and [`FrameBuilder::finish`] commits it; [`Recorder::to_clip`] turns the ring into a [`Clip`]
//! (that one allocates, it runs when a clip is saved).

use ddai_net::tuning::{TeamsState, TuneParams};

use crate::format::*;

const NO_EVENT: ClipEvent = ClipEvent::Respawn { id: 0 };

/// One ring slot: a frame in fixed arrays.
#[derive(Clone)]
struct Slot {
    tick: i32,
    own_alive: bool,
    n_tees: u8,
    tees: [TeeRec; MAX_TEES],
    n_proj: u8,
    dropped: u8,
    tees_dropped: u8,
    projectiles: [ProjRec; MAX_PROJECTILES],
    n_sw: u8,
    switches: [SwitchRec; MAX_SWITCHES],
    n_sent: u8,
    sent: [SentRec; MAX_SENT],
    n_ev: u8,
    events: [ClipEvent; MAX_EVENTS],
    bot: BotRec,
}

impl Slot {
    fn empty() -> Slot {
        Slot {
            tick: 0,
            own_alive: false,
            n_tees: 0,
            tees: [TeeRec::default(); MAX_TEES],
            n_proj: 0,
            dropped: 0,
            tees_dropped: 0,
            projectiles: [ProjRec::default(); MAX_PROJECTILES],
            n_sw: 0,
            switches: [SwitchRec::default(); MAX_SWITCHES],
            n_sent: 0,
            sent: [SentRec::default(); MAX_SENT],
            n_ev: 0,
            events: [NO_EVENT; MAX_EVENTS],
            bot: BotRec::default(),
        }
    }

    fn reset(&mut self, tick: i32, own_alive: bool) {
        self.tick = tick;
        self.own_alive = own_alive;
        self.n_tees = 0;
        self.n_proj = 0;
        self.dropped = 0;
        self.tees_dropped = 0;
        self.n_sw = 0;
        self.n_sent = 0;
        self.n_ev = 0;
        self.bot = BotRec::default();
    }

    fn to_frame(&self) -> Frame {
        Frame {
            tick: self.tick,
            own_alive: self.own_alive,
            tees: self.tees[..usize::from(self.n_tees)].to_vec(),
            projectiles: self.projectiles[..usize::from(self.n_proj)].to_vec(),
            projectiles_dropped: self.dropped,
            tees_dropped: self.tees_dropped,
            switches: self.switches[..usize::from(self.n_sw)].to_vec(),
            sent: self.sent[..usize::from(self.n_sent)].to_vec(),
            events: self.events[..usize::from(self.n_ev)].to_vec(),
            bot: self.bot,
        }
    }
}

/// What [`Recorder::to_clip`] needs to know besides the frames.
#[derive(Debug, Clone)]
pub struct ClipMeta {
    pub map_name: String,
    pub map_sha256: [u8; 32],
    pub own_id: i32,
    pub brain: String,
    pub reason: ClipReason,
    pub players: Vec<PlayerTag>,
}

/// The ring.
pub struct Recorder {
    slots: Vec<Slot>,
    next: usize,
    filled: usize,
    labels: Vec<String>,
    tuning: Vec<TuneChange>,
    last_tuning: Option<TuneParams>,
    teams: Vec<TeamsChange>,
    last_teams: Option<(usize, [i32; 128])>,
    world_seed: u64,
}

impl Recorder {
    /// A recorder of `capacity` frames (use [`RING_FRAMES`]).
    pub fn new(capacity: usize) -> Recorder {
        let capacity = capacity.max(1);
        Recorder {
            slots: vec![Slot::empty(); capacity],
            next: 0,
            filled: 0,
            labels: vec![String::new()],
            tuning: Vec::new(),
            last_tuning: None,
            teams: Vec::new(),
            last_teams: None,
            world_seed: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Frames in the ring now.
    pub fn len(&self) -> usize {
        self.filled
    }

    pub fn is_empty(&self) -> bool {
        self.filled == 0
    }

    /// The tick of the newest frame.
    pub fn last_tick(&self) -> Option<i32> {
        (self.filled > 0).then(|| self.slots[(self.next + self.slots.len() - 1) % self.slots.len()].tick)
    }

    /// Replaces the sent inputs of the newest frame (the bot re-reads its late-corrected inputs one frame
    /// later, once the server's timing reports have arrived). Allocates nothing.
    pub fn amend_last_sent(&mut self, recs: &[SentRec]) -> bool {
        if self.filled == 0 {
            return false;
        }
        let at = (self.next + self.slots.len() - 1) % self.slots.len();
        let s = &mut self.slots[at];
        s.n_sent = 0;
        for r in recs.iter().take(MAX_SENT) {
            s.sent[usize::from(s.n_sent)] = *r;
            s.n_sent += 1;
        }
        true
    }

    /// Forgets the frames (a new map, a tick reset); the interned labels stay.
    pub fn clear(&mut self) {
        self.next = 0;
        self.filled = 0;
        self.tuning.clear();
        self.last_tuning = None;
        self.teams.clear();
        self.last_teams = None;
    }

    pub fn set_world_seed(&mut self, seed: u64) {
        self.world_seed = seed;
    }

    /// The index of `label` in the label table (0 for the empty label). A new label allocates once.
    pub fn intern(&mut self, label: &str) -> u16 {
        if label.is_empty() {
            return 0;
        }
        if let Some(i) = self.labels.iter().position(|l| l == label) {
            return i as u16;
        }
        if self.labels.len() >= usize::from(u16::MAX) {
            return 0;
        }
        self.labels.push(label.to_string());
        (self.labels.len() - 1) as u16
    }

    /// Notes the tuning in force at `tick`; stored only when it differs from the last one.
    pub fn note_tuning(&mut self, tick: i32, t: &TuneParams) {
        if self.last_tuning.as_ref() == Some(t) {
            return;
        }
        self.last_tuning = Some(*t);
        self.tuning.push(TuneChange {
            from_tick: tick,
            received: t.received as u32,
            values: t.to_array().to_vec(),
        });
    }

    /// Notes the teams state in force at `tick` (stored when it changes).
    pub fn note_teams(&mut self, tick: i32, t: &TeamsState) {
        let n = t.received.min(128);
        if self
            .last_teams
            .as_ref()
            .is_some_and(|(rn, rt)| *rn == n && rt[..n] == t.teams[..n])
        {
            return;
        }
        self.last_teams = Some((n, t.teams));
        self.teams.push(TeamsChange {
            from_tick: tick,
            received: n as u32,
            teams: t.teams[..n].to_vec(),
        });
    }

    /// Opens the next frame of the ring. Nothing is stored until [`FrameBuilder::finish`].
    pub fn begin(&mut self, tick: i32, own_alive: bool) -> FrameBuilder<'_> {
        let at = self.next;
        self.slots[at].reset(tick, own_alive);
        FrameBuilder { rec: self, at }
    }

    /// The frames, oldest first, as a [`Clip`] (allocates: this runs when a clip is saved).
    /// `last_frames` limits the clip to the newest frames; `None` takes the whole ring.
    pub fn to_clip(&self, meta: ClipMeta, last_frames: Option<usize>) -> Option<Clip> {
        if self.filled == 0 {
            return None;
        }
        let take = last_frames.unwrap_or(self.filled).min(self.filled);
        let cap = self.slots.len();
        let start = (self.next + cap - take) % cap;
        let frames: Vec<Frame> = (0..take).map(|i| self.slots[(start + i) % cap].to_frame()).collect();
        let (first, last) = (frames.first()?.tick, frames.last()?.tick);
        // The state in force at the first frame, then every later change.
        let pick = |changes: &[(i32, usize)]| -> Vec<usize> {
            let mut keep = Vec::new();
            let at_first = changes.iter().rposition(|&(t, _)| t <= first);
            if let Some(i) = at_first {
                keep.push(i);
            }
            keep.extend(
                changes
                    .iter()
                    .enumerate()
                    .filter(|&(_, &(t, _))| t > first && t <= last)
                    .map(|(i, _)| i),
            );
            keep
        };
        let tune_idx: Vec<(i32, usize)> = self.tuning.iter().enumerate().map(|(i, c)| (c.from_tick, i)).collect();
        let tuning = pick(&tune_idx)
            .into_iter()
            .map(|k| {
                let mut c = self.tuning[k].clone();
                c.from_tick = c.from_tick.max(first);
                c
            })
            .collect();
        let teams_idx: Vec<(i32, usize)> = self.teams.iter().enumerate().map(|(i, c)| (c.from_tick, i)).collect();
        let teams = pick(&teams_idx)
            .into_iter()
            .map(|k| {
                let mut c = self.teams[k].clone();
                c.from_tick = c.from_tick.max(first);
                c
            })
            .collect();
        Some(Clip {
            header: ClipHeader {
                map_name: meta.map_name,
                map_sha256: meta.map_sha256,
                own_id: meta.own_id,
                brain: meta.brain,
                reason: meta.reason,
                labels: self.labels.clone(),
                players: meta.players,
                tuning,
                teams,
                world_seed: self.world_seed,
            },
            frames,
        })
    }

    /// The frames, oldest first (allocates; for the incident search and the tests).
    pub fn frames(&self) -> Vec<Frame> {
        let cap = self.slots.len();
        let start = (self.next + cap - self.filled) % cap;
        (0..self.filled)
            .map(|i| self.slots[(start + i) % cap].to_frame())
            .collect()
    }

    /// The label table.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }
}

/// A frame being filled; [`FrameBuilder::finish`] commits it.
pub struct FrameBuilder<'a> {
    rec: &'a mut Recorder,
    at: usize,
}

impl FrameBuilder<'_> {
    fn slot(&mut self) -> &mut Slot {
        &mut self.rec.slots[self.at]
    }

    /// Adds a tee; false (and nothing added) when the frame is full.
    pub fn add_tee(&mut self, t: TeeRec) -> bool {
        let s = self.slot();
        if usize::from(s.n_tees) >= MAX_TEES {
            return false;
        }
        s.tees[usize::from(s.n_tees)] = t;
        s.n_tees += 1;
        true
    }

    pub fn add_projectile(&mut self, p: ProjRec) {
        let s = self.slot();
        if usize::from(s.n_proj) >= MAX_PROJECTILES {
            s.dropped = s.dropped.saturating_add(1);
            return;
        }
        s.projectiles[usize::from(s.n_proj)] = p;
        s.n_proj += 1;
    }

    pub fn add_switch(&mut self, sw: SwitchRec) {
        let s = self.slot();
        if usize::from(s.n_sw) < MAX_SWITCHES {
            s.switches[usize::from(s.n_sw)] = sw;
            s.n_sw += 1;
        }
    }

    pub fn add_sent(&mut self, r: SentRec) {
        let s = self.slot();
        if usize::from(s.n_sent) < MAX_SENT {
            s.sent[usize::from(s.n_sent)] = r;
            s.n_sent += 1;
        }
    }

    /// Adds an event; events past the cap of the frame are dropped (the newest are the ones lost).
    pub fn add_event(&mut self, e: ClipEvent) {
        let s = self.slot();
        if usize::from(s.n_ev) < MAX_EVENTS {
            s.events[usize::from(s.n_ev)] = e;
            s.n_ev += 1;
        }
    }

    /// How many tees of the snapshot were left out of the frame.
    pub fn set_tees_dropped(&mut self, n: usize) {
        self.slot().tees_dropped = u8::try_from(n).unwrap_or(u8::MAX);
    }

    pub fn set_bot(&mut self, b: BotRec) {
        self.slot().bot = b;
    }

    /// Commits the frame to the ring.
    pub fn finish(self) {
        let cap = self.rec.slots.len();
        self.rec.next = (self.at + 1) % cap;
        self.rec.filled = (self.rec.filled + 1).min(cap);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tee(id: i32, x: i32) -> TeeRec {
        TeeRec {
            id,
            ch: CharRec {
                x,
                ..CharRec::default()
            },
            ..TeeRec::default()
        }
    }

    fn meta() -> ClipMeta {
        ClipMeta {
            map_name: "m".into(),
            map_sha256: [1; 32],
            own_id: 0,
            brain: "planner".into(),
            reason: ClipReason {
                kind: "manual".into(),
                severity: 0,
                tick: 0,
                note: String::new(),
            },
            players: vec![],
        }
    }

    #[test]
    fn the_ring_keeps_the_newest_frames_oldest_first() {
        let mut r = Recorder::new(4);
        for t in 0..10 {
            let mut f = r.begin(t * 2, true);
            f.add_tee(tee(0, t));
            f.finish();
        }
        assert_eq!(r.len(), 4);
        let ticks: Vec<i32> = r.frames().iter().map(|f| f.tick).collect();
        assert_eq!(ticks, vec![12, 14, 16, 18]);
        let clip = r.to_clip(meta(), Some(2)).unwrap();
        assert_eq!(clip.frames.iter().map(|f| f.tick).collect::<Vec<_>>(), vec![16, 18]);
        assert_eq!(r.last_tick(), Some(18));
    }

    #[test]
    fn an_unfinished_frame_is_not_stored_and_full_arrays_drop_the_overflow() {
        let mut r = Recorder::new(3);
        {
            let mut f = r.begin(2, true);
            f.add_tee(tee(0, 1));
            // dropped without finish
        }
        assert!(r.is_empty());
        let mut f = r.begin(4, true);
        for id in 0..12 {
            f.add_tee(tee(id, id));
            f.add_projectile(ProjRec::default());
            f.add_event(ClipEvent::FreezeOnset { id });
        }
        f.finish();
        let fr = &r.frames()[0];
        assert_eq!(fr.tees.len(), MAX_TEES);
        assert_eq!(fr.projectiles.len(), MAX_PROJECTILES.min(12));
        assert_eq!(fr.events.len(), 12);
        let mut f = r.begin(6, true);
        for _ in 0..40 {
            f.add_projectile(ProjRec::default());
        }
        f.finish();
        assert_eq!(r.frames()[1].projectiles_dropped as usize, 40 - MAX_PROJECTILES);
    }

    #[test]
    fn labels_tuning_and_teams_are_stored_once_and_the_clip_takes_the_state_in_force() {
        let mut r = Recorder::new(8);
        assert_eq!(r.intern(""), 0);
        let a = r.intern("walk to (3,4)");
        assert_eq!(r.intern("walk to (3,4)"), a, "interned");
        let mut tune = ddai_net::tuning::DEFAULT_TUNE_PARAMS;
        r.note_tuning(0, &tune);
        r.note_tuning(2, &tune);
        tune.gravity += 5;
        for t in 0..6 {
            if t == 3 {
                r.note_tuning(t * 2, &tune);
            }
            r.begin(t * 2, true).finish();
        }
        let clip = r.to_clip(meta(), Some(3)).unwrap(); // frames 6, 8, 10
        assert_eq!(
            clip.header.tuning.len(),
            1,
            "only the gravity change, in force from the first frame"
        );
        assert_eq!(clip.header.tuning[0].from_tick, 6);
        assert_eq!(clip.header.labels, vec![String::new(), "walk to (3,4)".to_string()]);
        let all = r.to_clip(meta(), None).unwrap();
        assert_eq!(all.header.tuning.len(), 2);
    }
}
