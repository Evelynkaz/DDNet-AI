//! Frames -> per-snapshot world states, reconstructed inputs and physics replay.
//!
//! For every fresh snapshot `k` (25 Hz, two ticks apart) the pipeline
//!
//! 1. builds the exact server-tick state of every visible character with [`LiveWorld`]
//!    (reckoning-core extrapolation, `ddai-world` 2.4), after giving characters that carry no
//!    `DDNetCharacter` extension a *synthetic* one (freeze state inferred, see
//!    [`infer_freeze`]);
//! 2. reconstructs each character's input for the interval `[tick_{k-1}, tick_k)` (two steps) with the
//!    recorder's `reconstruct` (rec v1 semantics, reused unchanged);
//! 3. replays the interval with [`World<f32>`] from the state at `k-1` and compares it with the
//!    state at `k` ([`crate::replay`]).
//!
//! The `Action` attached to the *observation at `k-1`* is the input of that interval, i.e. the
//! decision taken at snapshot `k-1` and held until the next snapshot.
//!
//! **Streaming (task 8.4d).** [`build_stream`] takes the anonymised frames one at a time and hands
//! every finished frame (with the steps of the interval that follows it) to a sink; memory is
//! constant in the length of the demo. Everything above is causal except the input reconstruction:
//! the recorder registers a fire event only when a character's dead-reckoning tick advances (a
//! stale core can delay that by minutes: 1000 s seen), and its `(client id, tick)` lookup gives
//! the *last* registration of a pair over the whole demo. So a first, cheap pass over the demo
//! ([`recon_table`]: decode, anonymise, reconstruct - no physics) collects exactly those two
//! whole-demo facts in a [`ReconTable`] (the fire ticks of every track, about 4 bytes per fire,
//! and the few re-registered pairs), and the second pass ([`build_stream`]) is then causal.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;

use ddai_net::generated::enums::characterflagflag as cf;
use ddai_net::generated::objects;
use ddai_net::view::CharacterView;
use ddai_physics::core::{CharacterCore, PlayerInput};
use ddai_physics::map::MapData;
use ddai_physics::world::{TickInput, World};
use ddai_recorder::format::Frame;
use ddai_recorder::reconstruct::{CharacterSample, StreamReconstructor};
use ddai_world::{LiveWorld, SnapshotInput, character_observation};

use crate::config::Config;
use crate::ingest::{Ingested, LabelTracker};
use crate::replay::{Channel, Diff, ReplayStats, diff};
use crate::types::{ActionRec, CharRec, FrameRec, ReplayClass, char_flags};

const WEAPON_HAMMER: i32 = 0;
const WEAPON_NINJA: i32 = 5;

/// What is known about the interval that follows snapshot `k`, per character of frame `k`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepInfo {
    pub action: ActionRec,
    pub replay: ReplayClass,
    pub pos_err: f32,
    /// A non-neutral action, or the character was moving, or hooking.
    pub active: bool,
    /// The wire core at the next snapshot was fresh (the inputs were observable).
    pub next_fresh: bool,
}

/// Output of [`build`]: the whole demo in memory (tests and small inputs; the pipeline streams
/// through [`build_stream`]).
#[derive(Debug, Clone, Default)]
pub struct Built {
    pub frames: Vec<FrameRec>,
    /// `steps[k][slot]`, aligned with `frames[k].chars`; `None` when there is no usable next
    /// snapshot for that character.
    pub steps: Vec<Vec<Option<StepInfo>>>,
    /// Replay statistics per anonymous player label.
    pub replay: BTreeMap<u16, ReplayStats>,
    pub counters: BuildCounters,
    pub summary: BuildSummary,
}

/// One finished frame: the record and, for each of its characters, what is known about the
/// interval that follows it (see [`StepInfo`]).
#[derive(Debug, Clone)]
pub struct FrameOut {
    pub frame: FrameRec,
    pub steps: Vec<Option<StepInfo>>,
}

/// What [`build_stream`] returns besides the frames.
#[derive(Debug, Clone, Default)]
pub struct BuildSummary {
    /// Replay statistics per anonymous player label.
    pub replay: BTreeMap<u16, ReplayStats>,
    pub counters: BuildCounters,
    /// Frames emitted.
    pub frames: usize,
    /// Facts from the reconstruction pass.
    pub recon: ReconStats,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BuildCounters {
    /// Character-snapshots with the `DDNetCharacter` extension / without it.
    pub ext_chars: u64,
    pub inferred_chars: u64,
    /// Character-snapshots that are frozen (of all).
    pub frozen_chars: u64,
    pub chars: u64,
    /// Consecutive snapshots that were not `decision_ticks` apart.
    pub gaps: u64,
    pub chars_without_info: u64,
    /// Character-snapshots whose wire core was fresh.
    pub fresh_chars: u64,
    /// Check of the ninja-weapon freeze convention on characters that carry the real
    /// `DDNetCharacter` (`freeze_end != 0` or movements disabled): both, only the extension,
    /// only the weapon, neither.
    pub conv_both: u64,
    pub conv_ext_only: u64,
    pub conv_weapon_only: u64,
    pub conv_neither: u64,
}

/// Freeze inference for a character without the `DDNetCharacter` extension: servers of that era
/// send `WEAPON_NINJA` as the active weapon of a frozen tee to clients that do not announce the
/// extension (`character.cpp` `SnapCharacter`: "use ninja graphic for old clients if player is
/// frozen"). Nobody carries a real ninja in block, so `weapon == NINJA` means frozen. Measured on
/// the extension-carrying part of the archive: precision 98.4 %, recall 87.4 % of that convention
/// (docs/formats.md section 20 / E-004).
pub fn infer_freeze(weapon: i32) -> bool {
    weapon == WEAPON_NINJA
}

/// Synthetic `DDNetCharacter` for a character that has none: everything `read_ddnet` consumes.
fn synth_ddnet(
    id: i32,
    tick: i32,
    frozen: bool,
    frozen_for: i32,
    active_weapon: i32,
    target: (i32, i32),
    cfg: &Config,
) -> objects::DDNetCharacter {
    let weapon_bit = match active_weapon {
        2 => cf::WEAPON_SHOTGUN,
        3 => cf::WEAPON_GRENADE,
        4 => cf::WEAPON_LASER,
        _ => 0,
    };
    let remaining = (cfg.freeze_ticks - frozen_for).max(1);
    objects::DDNetCharacter {
        flags: cf::WEAPON_HAMMER | cf::WEAPON_GUN | weapon_bit,
        freeze_end: if frozen { tick + remaining } else { 0 },
        jumps: 2,
        tele_checkpoint: 0,
        strong_weak_id: id,
        jumped_total: -1,
        ninja_activation_tick: -1,
        freeze_start: -1,
        target_x: target.0,
        target_y: target.1,
        tune_zone_override: -1,
    }
}

fn i8c(v: i32) -> i8 {
    v.clamp(i32::from(i8::MIN), i32::from(i8::MAX)) as i8
}
fn i16c(v: i32) -> i16 {
    v.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

/// The `Action` the human chose, as far as the recorder's reconstruction can tell, for the
/// interval `[prev_tick, tick)`.
struct Derived {
    action: ActionRec,
    input: PlayerInput,
    /// The estimated sample lies inside the interval (inputs were observable).
    fresh: bool,
    /// For each step of the interval: the press applied in that step (`attack_tick == prev + step`).
    fire_at: Vec<bool>,
}

/// What the reconstructor knows about one character of a frame: its current sample and the fire
/// events of its track.
#[derive(Clone, Copy)]
struct Recon<'a> {
    sample: &'a CharacterSample,
    fire_ticks: &'a [i32],
}

fn derive(rec: Option<Recon<'_>>, prev_tick: i32, tick: i32) -> Derived {
    let n = (tick - prev_tick).max(0) as usize;
    let Some(r) = rec else {
        let action = ActionRec {
            direction: 0,
            jump: false,
            hook: false,
            fire: false,
            aim: [0, -1],
        };
        return Derived {
            action,
            input: to_input(&action),
            fresh: false,
            fire_at: vec![false; n],
        };
    };
    let est = &r.sample.input;
    let traj = &r.sample.trajectory;
    let fresh = est.tick > prev_tick;
    let jump = (traj.jumped & 1) != 0 || (fresh && est.jump);
    // `attack_tick` is the world tick *before* the step in which the press was applied
    // (`FireWeapon` runs in the early-input phase, before the tick counter increments), so the
    // step `prev + s -> prev + s + 1` produces `attack_tick == prev + s`.
    let mut fire_at = vec![false; n];
    for &t in r.fire_ticks.iter().filter(|&&t| t >= prev_tick && t < tick) {
        fire_at[(t - prev_tick) as usize] = true;
    }
    let fire = fire_at.iter().any(|&f| f);
    let aim = if est.aim_x == 0 && est.aim_y == 0 {
        [0, -1]
    } else {
        [est.aim_x, est.aim_y]
    };
    let action = ActionRec {
        direction: i8c(est.direction),
        jump,
        hook: est.hook,
        fire,
        aim,
    };
    Derived {
        action,
        input: to_input(&action),
        fresh,
        fire_at,
    }
}

fn to_input(a: &ActionRec) -> PlayerInput {
    PlayerInput {
        direction: i32::from(a.direction),
        target_x: a.aim[0],
        target_y: a.aim[1],
        jump: i32::from(a.jump),
        fire: i32::from(a.fire),
        hook: i32::from(a.hook),
        player_flags: 0,
        wanted_weapon: 0,
        next_weapon: 0,
        prev_weapon: 0,
    }
}

fn neutral_channel(input: &PlayerInput, ch: Channel) -> PlayerInput {
    let mut i = *input;
    match ch {
        Channel::Direction => i.direction = 0,
        Channel::Jump => i.jump = 0,
        Channel::Hook => i.hook = 0,
        Channel::Fire => i.fire = 0,
        Channel::Aim => {
            i.target_x = -i.target_x;
            i.target_y = -i.target_y;
        }
    }
    i
}

fn channel_active(input: &PlayerInput, ch: Channel) -> bool {
    match ch {
        Channel::Direction => input.direction != 0,
        Channel::Jump => input.jump != 0,
        Channel::Hook => input.hook != 0,
        Channel::Fire => input.fire != 0,
        Channel::Aim => input.hook != 0 || input.fire != 0,
    }
}

/// What one replayed character ended with.
#[derive(Clone, Copy)]
struct Outcome {
    core: CharacterCore<f32>,
    /// `Character::attack_tick` of the replayed character (set by `fire_weapon`).
    attack_tick: i32,
}

/// The per-tick inputs of an interval: `inputs[t]` for tick `t + 1` after the start. The fire
/// counter follows DDNet's press-counter rules: a press makes it odd (one press counted by
/// `count_input_presses`), a tick without a press releases it. Presses happen on the exact
/// `attack_tick`, never spread over the interval.
fn interval_inputs(ids: &[u8], derived: &[Derived], ticks: usize) -> Vec<Vec<TickInput>> {
    let mut counters = vec![0i32; ids.len()];
    (0..ticks)
        .map(|t| {
            ids.iter()
                .zip(derived)
                .zip(counters.iter_mut())
                .map(|((&id, d), c)| {
                    let press = d.fire_at.get(t).copied().unwrap_or(false);
                    if press {
                        *c += if *c % 2 == 0 { 1 } else { 2 };
                    } else if *c % 2 == 1 {
                        *c += 1;
                    }
                    let mut input = d.input;
                    input.fire = *c;
                    TickInput { id, input, kill: false }
                })
                .collect()
        })
        .collect()
}

/// Replays the interval from `start`, one tick at a time with that tick's inputs, and returns the
/// outcome of each requested id.
fn run(scratch: &mut World<f32>, start: &World<f32>, inputs: &[Vec<TickInput>], ids: &[u8]) -> Vec<Option<Outcome>> {
    scratch.restore_from(start);
    // The replay world starts without projectiles. Since task 2.4b `LiveWorld` itself holds none
    // unless its caller passes the snapshot's projectile items (`SnapshotInput::projectiles`); this
    // pipeline does not, on purpose: the recording client strips the DDNet extra info from legacy
    // projectile items (`gameclient.cpp:1409-1413`), leaving plain owner-less items with no
    // bounce/freeze/explosive flags (most of the archive's projectile items are of that
    // shape, counted with `View::projectiles()`) — replaying those would fly inert bullets, not
    // the cannons' freeze hazard. Projectiles fired by players inside the replay window are still
    // simulated.
    // (Before 2.4b the map-native cannons spawned at `start_tick 0` were cleared here because
    // stepping them at a demo's server tick of millions took 0.2-1 s per `step` on BlmapChill;
    // the clear stays as a guard that costs nothing.)
    scratch.projectiles.clear();
    for tick_inputs in inputs {
        scratch.step(tick_inputs);
    }
    ids.iter()
        .map(|&id| {
            let core = scratch.cores.get(id).copied()?;
            let attack_tick = scratch.characters[id as usize].as_ref()?.attack_tick;
            Some(Outcome { core, attack_tick })
        })
        .collect()
}

fn diffs(outcomes: &[Option<Outcome>], target: &World<f32>, ids: &[u8], within_px: f32) -> Vec<Option<Diff>> {
    ids.iter()
        .zip(outcomes)
        .map(|(&id, o)| {
            let a = &o.as_ref()?.core;
            let b = target.cores.get(id)?;
            Some(diff(a, b, within_px))
        })
        .collect()
}

/// The replay of one interval, computed while `live` still holds the state at the interval's start.
struct Replayed {
    ids: Vec<u8>,
    /// Index into the previous frame's `chars` for each id.
    slots: Vec<usize>,
    derived: Vec<Derived>,
    base: Vec<Option<Outcome>>,
    /// Per channel: the replay with that channel neutralised for everyone (`None` = not run).
    ablated: [Option<Vec<Option<Outcome>>>; 5],
    /// For each id, the indices (into `ids`) of the other characters within 100 px at the start:
    /// a swing is judged by what happens to them, not only by the swinger's own core.
    neighbours: Vec<Vec<usize>>,
}

/// How long [`recon_table`] remembers a registered `(client id, character tick)` pair to notice a
/// second registration of it (65 536 ticks, 22 min). Measured on the real archives: the longest gap
/// between two registrations of a pair is 132 ticks, and a character's tick is at most ~150 ticks
/// old when it is registered (the server refreshes a core at least every 3 s), so the window is
/// more than 400x what is needed. The assumption is checked, not just made: a second registration
/// of a pair happens at a snapshot no later than `registration tick - character tick` ticks after
/// the first (a character's tick is never ahead of its snapshot), so if
/// [`ReconStats::max_tick_age`] stays within the window no repetition can have been missed;
/// [`ReconStats::beyond_window`] counts registrations that break that (0 on all real data).
pub const PAIR_WINDOW_TICKS: i32 = 1 << 16;

/// Facts about the input reconstruction of a demo.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReconStats {
    /// Fire events registered.
    pub fire_events: u64,
    /// The longest time, in ticks, between a fire event and the snapshot that revealed it.
    pub max_fire_delay_ticks: i32,
    /// `(client id, character tick)` pairs registered more than once (a slot's stint changed while
    /// its stale core kept the tick).
    pub repeated_pairs: u64,
    /// The largest age, in ticks, of a character's tick at the snapshot that registered it.
    pub max_tick_age: i32,
    /// Registrations older than [`PAIR_WINDOW_TICKS`]: each could hide an unmerged repetition.
    pub beyond_window: u64,
}

/// What the whole-demo `reconstruct` knows that a causal pass cannot: the complete fire list of
/// every track (`fire_ticks` is final only at the end of the demo) and, for each
/// `(client id, character tick)` pair that was registered more than once, the last registration
/// (the batch lookup overwrites, so every frame using the pair gets the last one).
#[derive(Debug, Default)]
pub struct ReconTable {
    fires: HashMap<(i32, u32), Vec<i32>>,
    last_registration: HashMap<(i32, i32), CharacterSample>,
    pub stats: ReconStats,
}

impl ReconTable {
    /// The sample the whole-demo reconstruction would hand each character of a frame: the one the
    /// streaming reconstructor just produced, unless its `(client id, character tick)` pair was
    /// registered again later.
    fn resolve(
        &self,
        characters: &[ddai_recorder::format::CharacterRecord],
        samples: Vec<CharacterSample>,
    ) -> Vec<CharacterSample> {
        characters
            .iter()
            .zip(samples)
            .map(|(c, s)| {
                self.last_registration
                    .get(&(c.id, c.character.tick))
                    .copied()
                    .unwrap_or(s)
            })
            .collect()
    }

    fn fire_ticks(&self, key: (i32, u32)) -> &[i32] {
        self.fires.get(&key).map_or(&[], Vec::as_slice)
    }
}

/// First pass: decode, anonymise and reconstruct the demo, keeping only the [`ReconTable`]. Memory:
/// the table (fire ticks) plus a window of recently registered pairs.
pub fn recon_table(frames: impl Iterator<Item = (Frame, ddai_net::tuning::TuneParams)>) -> ReconTable {
    let mut recon = StreamReconstructor::new();
    let mut table = ReconTable::default();
    // Tick of the last registration of each pair within PAIR_WINDOW_TICKS, and the registrations
    // in order of age.
    let mut recent: HashMap<(i32, i32), i32> = HashMap::new();
    let mut queue: VecDeque<(i32, (i32, i32))> = VecDeque::new();
    for (frame, _) in frames {
        let Frame::Snapshot { tick, characters, .. } = &frame else {
            continue;
        };
        let tick = *tick;
        let pushed = recon.push(&frame, i32::MIN);
        for &(_, t) in &pushed.fires {
            table.stats.fire_events += 1;
            table.stats.max_fire_delay_ticks = table.stats.max_fire_delay_ticks.max(tick - t);
        }
        for (c, sample) in characters.iter().zip(&pushed.samples) {
            if !sample.registered {
                continue;
            }
            let pair = (c.id, c.character.tick);
            let age = tick - c.character.tick;
            table.stats.max_tick_age = table.stats.max_tick_age.max(age);
            if age > PAIR_WINDOW_TICKS {
                table.stats.beyond_window += 1;
            }
            if recent.contains_key(&pair) && table.last_registration.insert(pair, *sample).is_none() {
                table.stats.repeated_pairs += 1;
            }
            recent.insert(pair, tick);
            queue.push_back((tick, pair));
        }
        while queue.front().is_some_and(|&(t, _)| t < tick - PAIR_WINDOW_TICKS) {
            let (t, pair) = queue.pop_front().expect("front checked");
            if recent.get(&pair) == Some(&t) {
                recent.remove(&pair);
            }
        }
    }
    // The fire lists, complete.
    table.fires = recon.into_fire_ticks();
    table
}

/// Builds states, inputs and replay for a whole in-memory demo (tests and small inputs).
pub fn build(cfg: &Config, map: &Arc<MapData>, ing: &Ingested) -> Built {
    let mut out = Built::default();
    let frames = || ing.frames_with_tunes();
    let table = recon_table(frames());
    let summary = build_stream::<std::convert::Infallible>(cfg, map, &table, frames(), |fo| {
        out.frames.push(fo.frame);
        out.steps.push(fo.steps);
        Ok(())
    })
    .unwrap_or_else(|e| match e {});
    out.replay = summary.replay.clone();
    out.counters = summary.counters;
    out.summary = summary;
    out
}

/// Second pass: builds states, inputs and replay for a demo, one frame at a time. `frames` yields
/// the anonymised snapshots (with the tuning in force), the same sequence [`recon_table`] saw;
/// `sink` receives every finished frame in order. Memory does not depend on the number of frames.
pub fn build_stream<E>(
    cfg: &Config,
    map: &Arc<MapData>,
    table: &ReconTable,
    frames: impl Iterator<Item = (Frame, ddai_net::tuning::TuneParams)>,
    mut sink: impl FnMut(FrameOut) -> Result<(), E>,
) -> Result<BuildSummary, E> {
    let mut b = Builder::new(cfg, map);
    // Nothing older than the current fire is needed from the reconstructor here: the fire lists
    // come from the table.
    let mut recon = StreamReconstructor::new();
    let mut labels = LabelTracker::new();
    for (frame, tune) in frames {
        let Frame::Snapshot { characters, .. } = &frame else {
            continue;
        };
        let row = labels.push(&frame);
        let pushed = recon.push(&frame, i32::MAX);
        let samples = table.resolve(characters, pushed.samples);
        b.step(&frame, &row, tune, &samples, table, &mut sink)?;
    }
    b.finish(labels.missing as u64, &mut sink).map(|mut s| {
        s.recon = table.stats;
        s
    })
}

/// The per-frame state of the pipeline.
struct Builder<'a> {
    cfg: &'a Config,
    live: LiveWorld,
    /// Boxed: see `LiveWorld`'s fields (a by-value `World` blew the 2 MiB stack of test threads).
    scratch: Box<World<f32>>,
    last_weapon: HashMap<u16, i32>,
    frozen_since: HashMap<u16, i32>,
    /// Tick and record of frame k-1 (its steps are completed when frame k is processed).
    prev: Option<(i32, FrameRec)>,
    out: BuildSummary,
}

impl<'a> Builder<'a> {
    fn new(cfg: &'a Config, map: &Arc<MapData>) -> Self {
        let live = LiveWorld::new(Arc::clone(map), -1, 0);
        let scratch = Box::new(live.base_world().clone());
        Builder {
            cfg,
            live,
            scratch,
            last_weapon: HashMap::new(),
            frozen_since: HashMap::new(),
            prev: None,
            out: BuildSummary::default(),
        }
    }

    /// Tick of the newest processed frame (`i32::MIN` before the first).
    fn last_tick(&self) -> i32 {
        self.prev.as_ref().map_or(i32::MIN, |(t, _)| *t)
    }

    /// Completes the last frame (no successor) and returns the totals.
    fn finish<E>(
        mut self,
        chars_without_info: u64,
        sink: &mut impl FnMut(FrameOut) -> Result<(), E>,
    ) -> Result<BuildSummary, E> {
        if let Some((_, rec)) = self.prev.take() {
            let steps = vec![None; rec.chars.len()];
            for pc in &rec.chars {
                self.out.replay.entry(pc.player).or_default().no_next += 1;
            }
            sink(FrameOut { frame: rec, steps })?;
            self.out.frames += 1;
        }
        self.out.counters.chars_without_info = chars_without_info;
        Ok(self.out)
    }

    /// Processes one frame: `labels` and `samples` are aligned with its characters.
    fn step<E>(
        &mut self,
        frame: &Frame,
        labels: &[(i32, u16)],
        tune: ddai_net::tuning::TuneParams,
        samples: &[CharacterSample],
        table: &ReconTable,
        sink: &mut impl FnMut(FrameOut) -> Result<(), E>,
    ) -> Result<(), E> {
        let cfg = self.cfg;
        let Frame::Snapshot { tick, characters, .. } = frame else {
            return Ok(());
        };
        let tick = *tick;
        let recon_of = |ci: usize| -> Option<Recon<'_>> {
            samples.get(ci).map(|sample| Recon {
                sample,
                fire_ticks: table.fire_ticks(sample.key),
            })
        };

        // --- replay of the interval [prev_tick, tick) from the state at k-1 (still in `live`) ---
        let mut replayed: Option<Replayed> = None;
        if let Some((prev_tick, prev_frame)) = &self.prev {
            let prev_tick = *prev_tick;
            if tick - prev_tick == cfg.decision_ticks {
                let mut both: Vec<(u8, usize)> = Vec::new();
                for (slot, pc) in prev_frame.chars.iter().enumerate() {
                    if characters.iter().any(|c| c.id == i32::from(pc.id)) {
                        both.push((pc.id, slot));
                    }
                }
                both.sort_unstable();
                let mut r = Replayed {
                    ids: Vec::new(),
                    slots: Vec::new(),
                    derived: Vec::new(),
                    base: Vec::new(),
                    ablated: Default::default(),
                    neighbours: Vec::new(),
                };
                for (id, slot) in both {
                    let ci = characters
                        .iter()
                        .position(|c| c.id == i32::from(id))
                        .expect("present in both frames");
                    r.ids.push(id);
                    r.slots.push(slot);
                    r.derived.push(derive(recon_of(ci), prev_tick, tick));
                }
                let ticks = cfg.decision_ticks.max(0) as usize;
                let inputs = interval_inputs(&r.ids, &r.derived, ticks);
                r.base = run(&mut self.scratch, self.live.base_world(), &inputs, &r.ids);
                for ch in Channel::ALL {
                    if !r.derived.iter().any(|d| channel_active(&d.input, ch)) {
                        continue;
                    }
                    let alt: Vec<Vec<TickInput>> = inputs
                        .iter()
                        .map(|tick_inputs| {
                            tick_inputs
                                .iter()
                                .map(|t| TickInput {
                                    id: t.id,
                                    input: neutral_channel(&t.input, ch),
                                    kill: false,
                                })
                                .collect()
                        })
                        .collect();
                    r.ablated[ch as usize] = Some(run(&mut self.scratch, self.live.base_world(), &alt, &r.ids));
                }
                let starts: Vec<[f32; 2]> = r.slots.iter().map(|&s| prev_frame.chars[s].pos).collect();
                r.neighbours = (0..starts.len())
                    .map(|a| {
                        (0..starts.len())
                            .filter(|&b| {
                                b != a && (starts[a][0] - starts[b][0]).hypot(starts[a][1] - starts[b][1]) <= 100.0
                            })
                            .collect()
                    })
                    .collect();
                replayed = Some(r);
            } else {
                self.out.counters.gaps += 1;
            }
        }

        // --- advance the live world to snapshot k ---
        let mut views: Vec<CharacterView> = Vec::with_capacity(characters.len());
        for (ci, c) in characters.iter().enumerate() {
            let label = labels[ci].1;
            let mut character = c.character;
            let ddnet = match c.ddnet {
                Some(d) => Some(d),
                None => {
                    let frozen = infer_freeze(character.weapon);
                    let since = *self.frozen_since.entry(label).or_insert(tick);
                    if !frozen {
                        self.frozen_since.remove(&label);
                    }
                    let active = if frozen {
                        *self.last_weapon.get(&label).unwrap_or(&WEAPON_HAMMER)
                    } else {
                        self.last_weapon.insert(label, character.weapon);
                        character.weapon
                    };
                    character.weapon = active;
                    let target = aim_from_angle(character.angle);
                    Some(synth_ddnet(
                        c.id,
                        tick,
                        frozen,
                        if frozen { tick - since } else { 0 },
                        active,
                        target,
                        cfg,
                    ))
                }
            };
            views.push(CharacterView {
                id: c.id,
                character,
                ddnet,
            });
        }
        // No projectiles: the replay deliberately does not simulate them (see `docs/formats.md` §20.2).
        self.live.on_snapshot(SnapshotInput::new(tick, &views, tune));
        let world = self.live.base_world();

        // --- records for frame k ---
        let mut chars_out: Vec<CharRec> = Vec::with_capacity(characters.len());
        let prev_tick_for_fire = self.last_tick();
        let had_prev = self.prev.is_some();
        for (ci, c) in characters.iter().enumerate() {
            let Some(obs) = character_observation(world, c.id) else {
                continue;
            };
            let core = world.cores.get(c.id as u8);
            let label = labels[ci].1;
            let rec = recon_of(ci);
            let aim = match rec {
                Some(r) => [r.sample.input.aim_x, r.sample.input.aim_y],
                None => [0, -1],
            };
            let fired =
                had_prev && rec.is_some_and(|r| r.fire_ticks.iter().any(|&t| t >= prev_tick_for_fire && t < tick));
            let fresh = tick - c.character.tick <= 1;
            let mut flags = 0u16;
            if obs.is_frozen {
                flags |= char_flags::FROZEN;
                self.out.counters.frozen_chars += 1;
            }
            if obs.is_deep_frozen {
                flags |= char_flags::DEEP_FROZEN;
            }
            if obs.is_live_frozen {
                flags |= char_flags::LIVE_FROZEN;
            }
            if obs.grounded {
                flags |= char_flags::GROUNDED;
            }
            if let Some(d) = &c.ddnet {
                self.out.counters.ext_chars += 1;
                let ext_frozen = d.freeze_end != 0 || d.flags & cf::MOVEMENTS_DISABLED != 0;
                match (ext_frozen, infer_freeze(c.character.weapon)) {
                    (true, true) => self.out.counters.conv_both += 1,
                    (true, false) => self.out.counters.conv_ext_only += 1,
                    (false, true) => self.out.counters.conv_weapon_only += 1,
                    (false, false) => self.out.counters.conv_neither += 1,
                }
            } else {
                flags |= char_flags::FREEZE_INFERRED;
                self.out.counters.inferred_chars += 1;
            }
            if fired {
                flags |= char_flags::FIRED;
            }
            if fresh {
                flags |= char_flags::FRESH;
                self.out.counters.fresh_chars += 1;
            }
            if core.is_some_and(|c| c.jumped & 2 != 0) {
                flags |= char_flags::AIR_JUMP_USED;
            }
            if core.is_some_and(|c| c.jumped & 1 != 0) {
                flags |= char_flags::JUMP_HELD;
            }
            self.out.counters.chars += 1;
            chars_out.push(CharRec {
                id: c.id as u8,
                player: label,
                team: i16c(obs.team),
                pos: [obs.pos.x, obs.pos.y],
                vel: [obs.vel.x, obs.vel.y],
                hook_state: i8c(obs.hook_state),
                hook_pos: [obs.hook_pos.x, obs.hook_pos.y],
                hooked_player: i16c(obs.hooked_player),
                flags,
                freeze_ticks: i16c(obs.freeze_ticks_remaining),
                jumps_left: i8c(obs.jumps_left),
                jumps_used: i8c(obs.jumps_used),
                weapon: i8c(obs.weapon),
                direction: i8c(obs.direction),
                aim,
            });
        }
        let rec_k = FrameRec { tick, chars: chars_out };

        // --- compare the replay with the state at k, complete frame k-1 ---
        if let Some((prev_tick_for_stats, prev_rec)) = self.prev.take() {
            let n_prev = prev_rec.chars.len();
            let mut steps: Vec<Option<StepInfo>> = vec![None; n_prev];
            if let Some(r) = replayed {
                let base = diffs(&r.base, world, &r.ids, cfg.within_px);
                let ablated: [Option<Vec<Option<Diff>>>; 5] =
                    std::array::from_fn(|i| r.ablated[i].as_ref().map(|c| diffs(c, world, &r.ids, cfg.within_px)));
                for (i, &slot) in r.slots.iter().enumerate() {
                    let pc = prev_rec.chars[slot];
                    let label = pc.player;
                    let stats = self.out.replay.entry(label).or_default();
                    let Some(d) = base[i] else {
                        stats.no_next += 1;
                        continue;
                    };
                    let dv = &r.derived[i];
                    let moving = pc.vel[0].abs().max(pc.vel[1].abs()) >= cfg.active_speed;
                    let active = dv.action.direction != 0
                        || dv.action.jump
                        || dv.action.hook
                        || dv.action.fire
                        || moving
                        || pc.hook_state != 0;
                    stats.add_sample(d.class, active, dv.fresh, d.pos_err, d.vel_err);
                    // Fire validated by the exact `attack_tick`: did the replayed weapon fire on the
                    // same tick as in the demo?
                    if let Some(last) = dv.fire_at.iter().rposition(|&f| f) {
                        stats.fire_events += 1;
                        let expected = prev_tick_for_stats + last as i32;
                        if r.base[i].is_some_and(|o| o.attack_tick == expected) {
                            stats.fire_tick_match += 1;
                        }
                    }
                    for ch in Channel::ALL {
                        if !channel_active(&dv.input, ch) {
                            continue;
                        }
                        let Some(Some(ad)) = ablated[ch as usize].as_ref().map(|v| v[i]) else {
                            continue;
                        };
                        // A swing is judged by the swinger and by the characters next to it (the
                        // hammer never moves the swinger's own core).
                        let (mut base_score, mut abl_score) = (d.score, ad.score);
                        if ch == Channel::Fire {
                            for &nb in &r.neighbours[i] {
                                if let (Some(b), Some(a)) =
                                    (base[nb], ablated[ch as usize].as_ref().and_then(|v| v[nb]))
                                {
                                    base_score += b.score;
                                    abl_score += a.score;
                                }
                            }
                        }
                        let cs = &mut stats.channels[ch as usize];
                        cs.active += 1;
                        if dv.fresh {
                            cs.fresh += 1;
                        }
                        if abl_score > base_score + 0.5 {
                            cs.confirmed += 1;
                        } else if abl_score < base_score - 0.5 {
                            cs.contradicted += 1;
                        } else {
                            cs.unconstrained += 1;
                        }
                    }
                    steps[slot] = Some(StepInfo {
                        action: dv.action,
                        replay: d.class,
                        pos_err: d.pos_err,
                        active,
                        next_fresh: dv.fresh,
                    });
                }
                // Characters of frame k-1 that are absent now.
                for (slot, pc) in prev_rec.chars.iter().enumerate() {
                    if steps[slot].is_none() && !r.slots.contains(&slot) {
                        self.out.replay.entry(pc.player).or_default().no_next += 1;
                    }
                }
            } else {
                for pc in &prev_rec.chars {
                    self.out.replay.entry(pc.player).or_default().no_next += 1;
                }
            }
            sink(FrameOut { frame: prev_rec, steps })?;
            self.out.frames += 1;
        }
        self.prev = Some((tick, rec_k));
        Ok(())
    }
}

/// Direction of the wire angle (1/256 rad) as a nominal-magnitude integer vector.
fn aim_from_angle(angle: i32) -> (i32, i32) {
    let a = f64::from(angle) / 256.0;
    ((a.cos() * 1000.0).round() as i32, (a.sin() * 1000.0).round() as i32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{arena_with_pit, client_info, player_info, wire_character};
    use ddai_net::generated::objects;
    use ddai_physics::core::HOOK_GRABBED;
    use ddai_physics::vmath::Vec2;
    use ddai_physics::world::{self, Player};
    use ddai_recorder::format::{CharacterRecord, PlayerRecord};

    fn map() -> Arc<MapData> {
        Arc::new(arena_with_pit(30, 12, (10, 20)))
    }

    /// The wire `Character` + `DDNetCharacter` a server would send for `id` in `world`.
    fn wire_of(world: &World<f32>, id: u8, aim: (i32, i32)) -> (objects::Character, objects::DDNetCharacter) {
        let core = world.cores.get(id).expect("character exists");
        let ch = world.characters[id as usize].as_ref().expect("character exists");
        let n = core.write();
        let character = objects::Character {
            tick: world.tick,
            x: n.x,
            y: n.y,
            vel_x: n.vel_x,
            vel_y: n.vel_y,
            angle: n.angle,
            direction: n.direction,
            jumped: n.jumped,
            hooked_player: n.hooked_player,
            hook_state: n.hook_state,
            hook_tick: n.hook_tick,
            hook_x: n.hook_x,
            hook_y: n.hook_y,
            hook_dx: n.hook_dx,
            hook_dy: n.hook_dy,
            player_flags: 0,
            health: 0,
            armor: 0,
            ammo_count: 0,
            weapon: core.active_weapon,
            emote: 0,
            attack_tick: ch.attack_tick,
        };
        let ddnet = objects::DDNetCharacter {
            flags: cf::WEAPON_HAMMER | cf::WEAPON_GUN,
            freeze_end: if ch.freeze_time > 0 {
                world.tick + ch.freeze_time
            } else {
                0
            },
            jumps: core.jumps,
            tele_checkpoint: 0,
            strong_weak_id: i32::from(id),
            jumped_total: -1,
            ninja_activation_tick: -1,
            freeze_start: -1,
            target_x: aim.0,
            target_y: aim.1,
            tune_zone_override: -1,
        };
        (character, ddnet)
    }

    /// Runs the real physics world with scripted per-interval inputs and returns the frames a
    /// perfect server would have sent (every character fresh at every snapshot).
    fn scripted_run(intervals: &[[PlayerInput; 2]]) -> Vec<Frame> {
        scripted_run_at(intervals, [(200.0, 338.0), (880.0, 338.0)])
    }

    fn scripted_run_at(intervals: &[[PlayerInput; 2]], spawns: [(f32, f32); 2]) -> Vec<Frame> {
        let m = map();
        let mut world: World<f32> = World::from_map(&m, 1);
        let _ = world.init(std::iter::empty::<&str>());
        world.players[0] = Some(Player::new(0));
        world.players[1] = Some(Player::new(0));
        world::spawn_character(&mut world, 0, Vec2::new(spawns[0].0, spawns[0].1));
        world::spawn_character(&mut world, 1, Vec2::new(spawns[1].0, spawns[1].1));
        // Settle on the ground first.
        let idle = PlayerInput {
            target_y: -1,
            ..Default::default()
        };
        for _ in 0..30 {
            world.step(&[
                TickInput {
                    id: 0,
                    input: idle,
                    kill: false,
                },
                TickInput {
                    id: 1,
                    input: idle,
                    kill: false,
                },
            ]);
        }
        if world.tick % 2 != 0 {
            world.step(&[
                TickInput {
                    id: 0,
                    input: idle,
                    kill: false,
                },
                TickInput {
                    id: 1,
                    input: idle,
                    kill: false,
                },
            ]);
        }
        let snap = |world: &World<f32>, aims: [(i32, i32); 2]| {
            let mut characters = Vec::new();
            let mut players = Vec::new();
            for id in 0..2u8 {
                let (character, ddnet) = wire_of(world, id, aims[id as usize]);
                characters.push(CharacterRecord {
                    id: i32::from(id),
                    character,
                    ddnet: Some(ddnet),
                });
                players.push(PlayerRecord {
                    id: i32::from(id),
                    info: player_info(i32::from(id)),
                    client_info: Some(client_info(&format!("player_{id}"))),
                    ddnet: None,
                });
            }
            Frame::Snapshot {
                tick: world.tick,
                characters,
                players,
            }
        };
        let mut frames = vec![snap(&world, [(0, -1), (0, -1)])];
        for iv in intervals {
            for _ in 0..2 {
                world.step(&[
                    TickInput {
                        id: 0,
                        input: iv[0],
                        kill: false,
                    },
                    TickInput {
                        id: 1,
                        input: iv[1],
                        kill: false,
                    },
                ]);
            }
            frames.push(snap(
                &world,
                [(iv[0].target_x, iv[0].target_y), (iv[1].target_x, iv[1].target_y)],
            ));
        }
        frames
    }

    fn input(direction: i32, jump: bool, hook: bool, aim: (i32, i32)) -> PlayerInput {
        PlayerInput {
            direction,
            target_x: aim.0,
            target_y: aim.1,
            jump: i32::from(jump),
            hook: i32::from(hook),
            ..Default::default()
        }
    }

    fn ingested(frames: Vec<Frame>) -> Ingested {
        let n = frames.len();
        Ingested {
            frames,
            tunes: vec![ddai_net::tuning::DEFAULT_TUNE_PARAMS; n],
            ..Ingested::default()
        }
    }

    #[test]
    fn a_perfect_server_run_replays_exactly_and_recovers_the_scripted_inputs() {
        let idle = input(0, false, false, (0, -1));
        let mut script: Vec<[PlayerInput; 2]> = Vec::new();
        for _ in 0..6 {
            script.push([input(1, false, false, (100, -30)), idle]);
        }
        script.push([input(1, true, false, (100, -30)), idle]); // jump
        script.push([input(1, false, false, (100, -30)), idle]);
        for _ in 0..6 {
            script.push([input(-1, false, false, (-100, 20)), idle]);
        }
        let ing = ingested(scripted_run(&script));
        let built = build(&Config::default(), &map(), &ing);
        assert_eq!(built.frames.len(), script.len() + 1);
        assert_eq!(built.counters.ext_chars, 2 * (script.len() as u64 + 1));
        for (k, iv) in script.iter().enumerate() {
            for (slot, c) in built.frames[k].chars.iter().enumerate() {
                let st = built.steps[k][slot].unwrap_or_else(|| panic!("no step for frame {k} slot {slot}"));
                assert_eq!(st.replay, ReplayClass::Exact, "frame {k} player {}: {st:?}", c.player);
                let truth = &iv[c.id as usize];
                assert_eq!(i32::from(st.action.direction), truth.direction, "direction, frame {k}");
                assert_eq!(st.action.jump, truth.jump != 0, "jump, frame {k}");
                assert!(st.next_fresh);
            }
        }
        assert!(
            built.steps.last().unwrap().iter().all(Option::is_none),
            "the last frame has no next state"
        );
        let stats: ReplayStats = built.replay.values().fold(ReplayStats::default(), |mut a, b| {
            a.merge(b);
            a
        });
        assert_eq!(stats.by_class[ReplayClass::Exact as usize], 2 * script.len() as u64);
        assert_eq!(stats.by_class[ReplayClass::Off as usize], 0);
        // Walking right and left is confirmed by the ablation (without it the replay is worse).
        let dir = &stats.channels[Channel::Direction as usize];
        assert!(dir.active > 0 && dir.confirmed > 0 && dir.contradicted == 0, "{dir:?}");
    }

    #[test]
    fn a_hammer_hit_is_replayed_on_the_exact_attack_tick_and_confirmed_through_the_victim() {
        let idle = input(0, false, false, (0, -1));
        // Player 0 swings the hammer at player 1 standing 36 px to its right. The fire counter only
        // ever grows (press = odd value): 1 = press, 2 = release, 3 = press again.
        let swing = |fire: i32| PlayerInput {
            fire,
            ..input(0, false, false, (100, 0))
        };
        let mut script: Vec<[PlayerInput; 2]> = Vec::new();
        // The first interval selects the hammer (spawned characters hold the gun): wanted_weapon = 1.
        script.push([
            PlayerInput {
                wanted_weapon: 1,
                ..idle
            },
            idle,
        ]);
        for _ in 0..3 {
            script.push([idle, idle]);
        }
        script.push([swing(1), idle]);
        script.push([swing(2), idle]);
        for _ in 0..12 {
            script.push([swing(2), idle]);
        }
        script.push([swing(3), idle]);
        script.push([swing(4), idle]);
        for _ in 0..4 {
            script.push([swing(4), idle]);
        }
        let frames = scripted_run_at(&script, [(200.0, 338.0), (236.0, 338.0)]);
        let built = build(&Config::default(), &map(), &ingested(frames));
        let stats = built.replay.get(&0).expect("player 0 has stats");
        assert_eq!(stats.fire_events, 2, "two presses were made");
        assert_eq!(
            stats.fire_tick_match, 2,
            "the replayed weapon fired on the exact attack ticks"
        );
        let fire = &stats.channels[Channel::Fire as usize];
        assert_eq!(fire.active, 2);
        assert!(
            fire.confirmed >= 1,
            "the victim is thrown only when the swing is replayed: {fire:?}"
        );
        assert_eq!(fire.contradicted, 0, "{fire:?}");
        // The whole run, victim included, replays exactly.
        assert!(
            built
                .steps
                .iter()
                .flatten()
                .flatten()
                .all(|st| st.replay == ReplayClass::Exact)
        );
    }

    #[test]
    fn a_terrain_hook_is_recovered_and_confirmed() {
        let idle = input(0, false, false, (0, -1));
        // Player 1 stands at x = 880 next to the right wall (x = 928) and hooks it.
        let mut script: Vec<[PlayerInput; 2]> = Vec::new();
        for _ in 0..4 {
            script.push([idle, input(0, false, false, (1000, 0))]);
        }
        for _ in 0..12 {
            script.push([idle, input(0, false, true, (1000, 0))]);
        }
        let frames = scripted_run(&script);
        // The hook really grabbed terrain in the perfect run, otherwise this test proves nothing.
        let grabbed = frames.iter().any(|f| match f {
            Frame::Snapshot { characters, .. } => characters[1].character.hook_state == HOOK_GRABBED,
            _ => false,
        });
        assert!(grabbed, "scenario must produce a grabbed hook");
        let built = build(&Config::default(), &map(), &ingested(frames));
        let hook_steps: Vec<_> = built
            .steps
            .iter()
            .enumerate()
            .filter_map(|(k, s)| s.get(1).copied().flatten().map(|st| (k, st)))
            .filter(|(_, st)| st.action.hook)
            .collect();
        assert!(hook_steps.len() >= 10, "hook held for most of the second phase");
        assert!(hook_steps.iter().all(|(_, st)| st.replay == ReplayClass::Exact));
        let stats = built.replay.get(&1).expect("player 1 has stats");
        let h = &stats.channels[Channel::Hook as usize];
        assert!(h.active >= 10 && h.confirmed > 0, "{h:?}");
    }

    fn ext_free_frame(tick: i32, weapon: i32, x: i32) -> Frame {
        let mut c = wire_character(tick, x, 300);
        c.weapon = weapon;
        Frame::Snapshot {
            tick,
            characters: vec![CharacterRecord {
                id: 0,
                character: c,
                ddnet: None,
            }],
            players: vec![PlayerRecord {
                id: 0,
                info: player_info(0),
                client_info: Some(client_info("player_0")),
                ddnet: None,
            }],
        }
    }

    #[test]
    fn freeze_is_inferred_from_the_ninja_convention_when_there_is_no_extension() {
        assert!(infer_freeze(5));
        assert!(!infer_freeze(0));
        let frames = vec![
            ext_free_frame(10, 0, 100),
            ext_free_frame(12, 5, 100),
            ext_free_frame(14, 5, 100),
            ext_free_frame(16, 1, 100),
        ];
        let built = build(&Config::default(), &map(), &ingested(frames));
        let frozen: Vec<bool> = built.frames.iter().map(|f| f.chars[0].frozen()).collect();
        assert_eq!(frozen, vec![false, true, true, false]);
        assert!(built.frames[1].chars[0].has(char_flags::FREEZE_INFERRED));
        assert_eq!(built.counters.inferred_chars, 4);
        assert_eq!(built.counters.frozen_chars, 2);
        // The real weapon is restored while frozen (the ninja weapon is only a marker).
        assert_eq!(built.frames[1].chars[0].weapon, 0);
        assert_eq!(built.frames[3].chars[0].weapon, 1);
        // Remaining freeze time shrinks along the run.
        assert!(built.frames[1].chars[0].freeze_ticks > built.frames[2].chars[0].freeze_ticks);
        assert!(built.frames[2].chars[0].freeze_ticks > 0);
    }

    #[test]
    fn the_ninja_convention_is_measured_against_the_real_extension() {
        let mk = |tick: i32, id: i32, weapon: i32, frozen: bool| {
            let mut c = wire_character(tick, 100 + 40 * id, 300);
            c.weapon = weapon;
            let ext = objects::DDNetCharacter {
                flags: cf::WEAPON_HAMMER | cf::WEAPON_GUN,
                freeze_end: if frozen { tick + 50 } else { 0 },
                jumps: 2,
                tele_checkpoint: 0,
                strong_weak_id: id,
                jumped_total: -1,
                ninja_activation_tick: -1,
                freeze_start: -1,
                target_x: 0,
                target_y: -1,
                tune_zone_override: -1,
            };
            (
                CharacterRecord {
                    id,
                    character: c,
                    ddnet: Some(ext),
                },
                PlayerRecord {
                    id,
                    info: player_info(id),
                    client_info: Some(client_info(&format!("player_{id}"))),
                    ddnet: None,
                },
            )
        };
        let mut characters = Vec::new();
        let mut players = Vec::new();
        for (id, weapon, frozen) in [(0, 5, true), (1, 0, true), (2, 5, false), (3, 0, false), (4, 5, true)] {
            let (c, p) = mk(10, id, weapon, frozen);
            characters.push(c);
            players.push(p);
        }
        let frames = vec![Frame::Snapshot {
            tick: 10,
            characters,
            players,
        }];
        let built = build(&Config::default(), &map(), &ingested(frames));
        let c = built.counters;
        assert_eq!(
            (c.conv_both, c.conv_ext_only, c.conv_weapon_only, c.conv_neither),
            (2, 1, 1, 1)
        );
        assert_eq!(c.ext_chars, 5);
        assert_eq!(c.inferred_chars, 0);
    }

    #[test]
    fn a_gap_in_the_snapshots_yields_no_samples_and_is_counted() {
        let frames = vec![
            ext_free_frame(10, 0, 100),
            ext_free_frame(12, 0, 100),
            ext_free_frame(20, 0, 100), // dropped snapshots
            ext_free_frame(22, 0, 100),
        ];
        let built = build(&Config::default(), &map(), &ingested(frames));
        assert_eq!(built.counters.gaps, 1);
        assert!(built.steps[0][0].is_some());
        assert!(built.steps[1][0].is_none(), "no next state across the gap");
        assert!(built.steps[2][0].is_some());
        assert!(built.steps[3][0].is_none());
        let s = built.replay.get(&0).unwrap();
        assert_eq!(s.samples, 2);
        assert_eq!(s.no_next, 2);
    }

    #[test]
    fn a_character_that_leaves_view_gets_no_sample_for_the_step_it_misses() {
        let mut frames = vec![
            ext_free_frame(10, 0, 100),
            ext_free_frame(12, 0, 100),
            ext_free_frame(14, 0, 100),
        ];
        if let Frame::Snapshot { characters, .. } = &mut frames[1] {
            characters.clear();
        }
        let built = build(&Config::default(), &map(), &ingested(frames));
        assert!(built.steps[0][0].is_none());
        assert!(built.frames[1].chars.is_empty());
        assert!(built.steps[2][0].is_none());
    }

    /// A pseudo-random recording full of the awkward cases: stale cores (the tick repeats) with an
    /// `attack_tick` that moves meanwhile, a slot flickering out of `players` (new stint, same stale
    /// tick: the same `(id, tick)` registered twice), slots reused, characters coming and going.
    fn awkward_frames(seed: u64, n: usize) -> Vec<Frame> {
        let mut x = seed;
        let mut rnd = move |m: u64| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (x >> 33) % m
        };
        let mut core_tick = [0i32; 5];
        let mut attack = [0i32; 5];
        let mut frames = Vec::new();
        for f in 0..n {
            let tick = 1000 + 2 * f as i32;
            let mut characters = Vec::new();
            let mut players = Vec::new();
            for id in 0..5usize {
                if rnd(10) == 0 {
                    continue; // out of view (and sometimes out of `players`: a new stint)
                }
                match rnd(4) {
                    0 => core_tick[id] = tick,                        // a fresh core
                    1 => core_tick[id] = core_tick[id].max(tick - 6), // a slightly old one
                    _ => {}                                           // stale: same tick again
                }
                if rnd(5) == 0 {
                    attack[id] = tick - rnd(4) as i32 - 1;
                }
                let mut c = wire_character(core_tick[id].max(1), 100 + 40 * id as i32 + rnd(3) as i32, 300);
                c.attack_tick = attack[id];
                c.jumped = rnd(4) as i32;
                c.hook_state = if rnd(6) == 0 { 4 } else { 0 };
                characters.push(CharacterRecord {
                    id: id as i32,
                    character: c,
                    ddnet: (rnd(3) != 0).then(|| objects::DDNetCharacter {
                        flags: cf::WEAPON_HAMMER | cf::WEAPON_GUN,
                        freeze_end: 0,
                        jumps: 2,
                        tele_checkpoint: 0,
                        strong_weak_id: id as i32,
                        jumped_total: -1,
                        ninja_activation_tick: -1,
                        freeze_start: -1,
                        target_x: rnd(200) as i32 - 100,
                        target_y: rnd(200) as i32 - 100,
                        tune_zone_override: -1,
                    }),
                });
                if rnd(6) != 0 {
                    players.push(PlayerRecord {
                        id: id as i32,
                        info: player_info(id as i32),
                        client_info: Some(client_info(&format!("player_{}", id + 5 * (rnd(2) as usize)))),
                        ddnet: None,
                    });
                }
            }
            frames.push(Frame::Snapshot {
                tick,
                characters,
                players,
            });
        }
        frames
    }

    /// The streamed reconstruction (table + causal pass) hands every character exactly what the
    /// whole-demo `reconstruct` + `(client id, tick)` lookup of the original pipeline did.
    #[test]
    fn streamed_reconstruction_equals_the_whole_demo_lookup() {
        let mut repeated = 0;
        for seed in 1..=40u64 {
            let frames = awkward_frames(seed, 400);
            // Reference: the original pipeline's batch reconstruction and lookup.
            let recs = ddai_recorder::reconstruct::reconstruct(&frames);
            let mut lookup: HashMap<(i32, i32), (usize, usize)> = HashMap::new();
            for (ri, r) in recs.iter().enumerate() {
                for (si, s) in r.trajectory.iter().enumerate() {
                    lookup.insert((r.client_id, s.tick), (ri, si));
                }
            }
            // Streamed.
            let table = recon_table(
                frames
                    .iter()
                    .cloned()
                    .map(|f| (f, ddai_net::tuning::DEFAULT_TUNE_PARAMS)),
            );
            repeated += table.stats.repeated_pairs;
            let mut recon = StreamReconstructor::new();
            for frame in &frames {
                let Frame::Snapshot { characters, .. } = frame else {
                    unreachable!()
                };
                let pushed = recon.push(frame, i32::MAX);
                let samples = table.resolve(characters, pushed.samples);
                for (c, s) in characters.iter().zip(&samples) {
                    let &(ri, si) = lookup
                        .get(&(c.id, c.character.tick))
                        .expect("every character has a sample");
                    let r = &recs[ri];
                    assert_eq!((s.key.0, s.key.1), (r.client_id, r.stint), "seed {seed}");
                    assert_eq!(s.input, r.inputs[si], "seed {seed}");
                    assert_eq!(s.trajectory, r.trajectory[si], "seed {seed}");
                    assert_eq!(table.fire_ticks(s.key), r.fire_ticks.as_slice(), "seed {seed}");
                }
            }
        }
        assert!(
            repeated > 0,
            "the generator must produce re-registered pairs, or this proves nothing"
        );
    }

    #[test]
    fn a_wire_core_of_the_snapshot_tick_is_flagged_fresh() {
        let frames = vec![ext_free_frame(10, 0, 100), ext_free_frame(12, 0, 100)];
        let built = build(&Config::default(), &map(), &ingested(frames));
        assert!(built.frames[0].chars[0].has(char_flags::FRESH));
    }
}
