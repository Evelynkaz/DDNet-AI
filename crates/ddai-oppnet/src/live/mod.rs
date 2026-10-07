//! Task 3.17 (D-111): the opponent-input predictor in the **live** bot.
//!
//! [`LiveOpp`] sits between the bot's `LiveWorld` snapshots and the planning world. Per snapshot with a target it
//!
//! 1. **resolves** the windows it predicted earlier against what this snapshot shows of the opponent (direction, hook state, aim, jump
//!    events) and scores model against hold ([`sample_cost`]); resolved windows feed the [`guard::Guard`] and the compact JSONL log;
//! 2. **feeds the history ring** of the predictor (every snapshot with a target, not only those a brain decision follows: the training
//!    data had a frame at every decision, 2 ticks apart);
//! 3. at a brain decision, **predicts the window** (the ticks between the snapshot and the tick our input lands on) from the snapshot-tick
//!    world and our own inputs for those ticks, and hands back the opponent's inputs for the roll when the guard allows the model.
//!
//! The predictor and its features are exactly the arena's (`OppPredictor::predict` on a `World<f32>` at the snapshot tick): LiveWorld
//! rebuilds that world from the snapshot (see the feature-equality test in `tests/live_features.rs`). Everything here is allocation-free
//! after construction, the log lines are formatted into a buffer the runner drains once a second (the file is written by another thread:
//! [`writer::LogWriter`]).

pub mod analyze;
pub mod guard;
pub mod writer;

use std::io::Write as _;

use ddai_physics::core::PlayerInput as WireInput;
use ddai_physics::world::World;
use ddai_planner::hybrid::window::{PredictedInput, WindowCtx, WindowModel};

use crate::feature::{HORIZON, IF_SLOTS, wrap_angle};
use crate::frame::TeeFrame;
use crate::predictor::OppPredictor;
use guard::{Guard, GuardConfig, GuardState, GuardStatus, Transition};

/// The longest window the model is asked about (its in-flight slots and its one-hot length).
pub const WINDOW_MAX: usize = IF_SLOTS;
/// Windows tracked at once (a window is open for at most 4 snapshots; decisions come once per snapshot).
const PENDING: usize = 8;
/// Most bytes the log buffer holds before new lines are dropped (a stalled writer must not grow it without bound).
const LOG_BUF_LIMIT: usize = 1 << 20;
/// The aim error that saturates the aim cost (radians): half a radian is ~50 px at 100 px distance, more than the hit radius.
const AIM_COST_SATURATION: f32 = 0.5;

/// The log's schema version (the `v` of the header line).
pub const LOG_VERSION: u32 = 1;

/// A short fixed string for the opponent's tag (`c<id>-<8 hex>`), no allocation.
#[derive(Clone, Copy, PartialEq, Eq)]
struct TagBuf {
    buf: [u8; 24],
    len: u8,
}

impl TagBuf {
    const EMPTY: TagBuf = TagBuf { buf: [0; 24], len: 0 };

    fn new(s: &str) -> TagBuf {
        let mut t = TagBuf::EMPTY;
        // Only printable ASCII goes into the log; anything else is dropped (a tag is `c12-0a1b2c3d`).
        for b in s.bytes().filter(|b| b.is_ascii_alphanumeric() || *b == b'-').take(24) {
            t.buf[usize::from(t.len)] = b;
            t.len += 1;
        }
        t
    }

    fn as_str(&self) -> &str {
        std::str::from_utf8(&self.buf[..usize::from(self.len)]).unwrap_or("")
    }
}

/// What the model said for one tick of the window (the parts the guard and the log compare).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Pm {
    pub dir: i8,
    pub hook: bool,
    pub jump: bool,
    /// Absolute aim, radians in `[0, 2 pi)`.
    pub aim: f32,
}

/// What "hold" says for every tick of the window: what the snapshot at the window's start shows.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Hold {
    pub dir: i8,
    pub hook: bool,
    pub aim: f32,
}

/// What a snapshot shows of the opponent after the step it ends (the observable part of its input).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Actual {
    pub dir: i8,
    /// The hook is out (state above idle) -- the same reading the hold baseline makes of the window's first snapshot.
    pub hook: bool,
    pub aim: f32,
    /// A jump was used in the two steps since the previous snapshot (the `jumped` bits gained one).
    pub jump_event: bool,
}

fn norm_angle(a: f64) -> f32 {
    let two_pi = 2.0 * std::f64::consts::PI;
    let r = a.rem_euclid(two_pi);
    r as f32
}

/// The absolute difference of two angles (radians), in `[0, pi]`.
fn angle_gap(a: f32, b: f32) -> f32 {
    wrap_angle(f64::from(a) - f64::from(b)).abs() as f32
}

/// The costs of one sample: `(model, hold)`. Each is the sum of the direction miss (0 or 1), the hook miss (0 or 1) and the aim miss
/// (the error in radians over [`AIM_COST_SATURATION`], at most 1). Jump events are logged, not scored: the jump *input* is a level the
/// snapshots do not show, only the events it causes.
pub fn sample_cost(model: &Pm, hold: &Hold, actual: &Actual) -> (f32, f32) {
    let aim = |a: f32| (angle_gap(a, actual.aim) / AIM_COST_SATURATION).min(1.0);
    let c_model =
        f32::from(u8::from(model.dir != actual.dir)) + f32::from(u8::from(model.hook != actual.hook)) + aim(model.aim);
    let c_hold =
        f32::from(u8::from(hold.dir != actual.dir)) + f32::from(u8::from(hold.hook != actual.hook)) + aim(hold.aim);
    (c_model, c_hold)
}

/// The opponent's input for one tick of the roll from a prediction: the held input with the predicted direction, jump, hook and aim,
/// and the fire counter moved on by a press (`fire` is the opponent's own counter carried from tick to tick: a press is a move to the next odd
/// value, never a phantom press, never a release). The same rule as `ddai_planner::hybrid::window::input_from_prediction`, in wire types.
pub fn victim_input(p: &PredictedInput, hold: &WireInput, fire: &mut i32) -> WireInput {
    let mut input = *hold;
    input.direction = p.direction.clamp(-1, 1);
    input.jump = i32::from(p.jump);
    input.hook = i32::from(p.hook);
    input.target_x = ddai_jsmath::round(ddai_jsmath::cos(p.aim) * 300.0) as i32;
    input.target_y = ddai_jsmath::round(ddai_jsmath::sin(p.aim) * 300.0) as i32;
    if p.press {
        *fire += if *fire & 1 != 0 { 2 } else { 1 };
    }
    input.fire = *fire;
    input
}

#[derive(Clone, Copy, Default)]
struct Sample {
    k: u8,
    actual: Actual,
    model: Pm,
    hold: Hold,
}

/// A decision's window waiting for the snapshots that show what the opponent did.
#[derive(Clone, Copy)]
struct Pending {
    live: bool,
    tick: i32,
    lag: u8,
    used: bool,
    tag: TagBuf,
    hold: Hold,
    model: [Pm; HORIZON],
    samples: [Sample; HORIZON],
    n: u8,
    /// The guard's share of the costs: only the samples of ticks the roll uses (`k < max(w, 2)`), see [`guard_ticks`].
    g_model: f32,
    g_hold: f32,
    g_n: u8,
}

impl Pending {
    const EMPTY: Pending = Pending {
        live: false,
        tick: 0,
        lag: 0,
        used: false,
        tag: TagBuf::EMPTY,
        hold: Hold {
            dir: 0,
            hook: false,
            aim: 0.0,
        },
        model: [Pm {
            dir: 0,
            hook: false,
            jump: false,
            aim: 0.0,
        }; HORIZON],
        samples: [Sample {
            k: 0,
            actual: Actual {
                dir: 0,
                hook: false,
                aim: 0.0,
                jump_event: false,
            },
            model: Pm {
                dir: 0,
                hook: false,
                jump: false,
                aim: 0.0,
            },
            hold: Hold {
                dir: 0,
                hook: false,
                aim: 0.0,
            },
        }; HORIZON],
        n: 0,
        g_model: 0.0,
        g_hold: 0.0,
        g_n: 0,
    };
}

/// Counters since the start, for STATUS and the end-of-run line.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LiveCounts {
    /// Windows the model was run for (the shadow included).
    pub predicted: u64,
    /// Of them, windows where the model's inputs drove the roll.
    pub used: u64,
    /// Decisions with no model call: the window was empty or longer than the model's.
    pub skipped_window: u64,
    /// Decisions with no model call: the opponent is frozen for longer than the window (the game ignores its inputs).
    pub skipped_frozen: u64,
    /// Decisions with no model call: our own tee is frozen (the model never saw such states: training games end at the first freeze).
    pub skipped_own_frozen: u64,
    /// Decisions with no model call: outside the regime the model was trained in ([`RegimeGate`]).
    pub skipped_regime: u64,
    /// Snapshots whose frame fed the history.
    pub observed: u64,
    /// Log lines dropped because the buffer was full.
    pub log_dropped: u64,
}

/// What [`LiveOpp::window`] decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowUse {
    /// The roll plays the inputs in `victim` (the model drives).
    Model,
    /// The roll holds (the guard benched the model; or no model call: see [`LiveCounts::skipped_window`]).
    Hold,
}

/// The pair of tees of one snapshot: the world at the snapshot tick (every tee as the snapshot shows it), us, the opponent, and the opponent's tag
/// (`c<id>-<hash>`, never a nickname).
#[derive(Clone, Copy)]
pub struct Pair<'a> {
    pub world: &'a World<f32>,
    pub self_id: i32,
    pub target: i32,
    pub tag: &'a str,
}

/// The ticks of a window of length `lag` that the guard judges: `k < max(lag, 2)`. The roll plays ticks `0..lag`; the snapshots show only the odd ones
/// (`k = 1, 3, ...`), so a window shorter than 2 would never be judged -- it is judged on `k = 1`.
pub fn guard_ticks(lag: usize) -> usize {
    lag.max(2)
}

/// The situations the model drives in: the ones it was trained in (an arena duel: two tees, about 100 px apart, nobody else near). Outside them the
/// model is not called for the roll and not scored: on a crowd server it is out of distribution (clips, `docs/research/opponent-predictor-live.md` §4a).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegimeGate {
    /// `false`: no gate (measurements).
    pub enabled: bool,
    /// The target must be within this many pixels of us.
    pub max_target_dist_px: f32,
    /// Other live tees (neither us nor the target) counted within this many pixels of us or of the target ...
    pub others_radius_px: f32,
    /// ... must be at most this many.
    pub max_others: u32,
}

impl Default for RegimeGate {
    fn default() -> Self {
        RegimeGate {
            enabled: true,
            max_target_dist_px: 480.0,
            others_radius_px: 1000.0,
            max_others: 0,
        }
    }
}

impl RegimeGate {
    pub fn off() -> RegimeGate {
        RegimeGate {
            enabled: false,
            ..RegimeGate::default()
        }
    }

    /// Whether the situation of `world` (us `me`, the opponent `opp`) is one the model drives in.
    pub fn admits(&self, world: &World<f32>, self_id: i32, target: i32, me: [f32; 2], opp: [f32; 2]) -> bool {
        if !self.enabled {
            return true;
        }
        let d2 = |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2);
        if d2(me, opp) > self.max_target_dist_px.powi(2) {
            return false;
        }
        let r2 = self.others_radius_px.powi(2);
        let mut others = 0u32;
        for id in 0..ddai_physics::core::MAX_CLIENTS as i32 {
            if id == self_id || id == target {
                continue;
            }
            let (Some(ch), Some(c)) = (world.characters[id as usize].as_ref(), world.cores.get(id as u8)) else {
                continue;
            };
            let p = [c.pos.x, c.pos.y];
            if ch.alive && (d2(p, me) <= r2 || d2(p, opp) <= r2) {
                others += 1;
            }
        }
        others <= self.max_others
    }
}

pub struct LiveOpp {
    pred: OppPredictor,
    guard: Guard,
    gate: RegimeGate,
    sha: [u8; 32],
    target: i32,
    tag: TagBuf,
    last_tick: i32,
    /// Whether the snapshot of `last_tick` showed a live pair (a second look at the same tick gets the same answer).
    last_ok: bool,
    pending: [Pending; PENDING],
    /// The previous snapshot's `jumped` bits of the target and its tick, for the jump events.
    prev_jumped: Option<(i32, u8)>,
    out: [Option<PredictedInput>; HORIZON],
    transitions: Vec<Transition>,
    log: Vec<u8>,
    counts: LiveCounts,
}

impl LiveOpp {
    pub fn new(pred: OppPredictor, cfg: GuardConfig, sha: [u8; 32]) -> Result<LiveOpp, String> {
        Ok(LiveOpp {
            pred,
            guard: Guard::new(cfg)?,
            gate: RegimeGate::default(),
            sha,
            target: -1,
            tag: TagBuf::EMPTY,
            last_tick: i32::MIN,
            last_ok: false,
            pending: [Pending::EMPTY; PENDING],
            prev_jumped: None,
            out: [None; HORIZON],
            transitions: Vec::with_capacity(4),
            log: Vec::with_capacity(1 << 16),
            counts: LiveCounts::default(),
        })
    }

    /// Another regime gate (the default is [`RegimeGate::default`]; [`RegimeGate::off`] for measurements).
    pub fn with_gate(mut self, gate: RegimeGate) -> LiveOpp {
        self.gate = gate;
        self
    }

    /// Forgets the opponent pair: the history and the open windows (a respawn, the kill switch lifted). The guard keeps its judging span and its state:
    /// they are about the model, not about one life.
    pub fn reset(&mut self) {
        self.pred.reset();
        self.target = -1;
        self.tag = TagBuf::EMPTY;
        self.last_tick = i32::MIN;
        self.prev_jumped = None;
        for p in &mut self.pending {
            p.live = false;
        }
    }

    pub fn using_model(&self) -> bool {
        self.guard.using_model()
    }

    pub fn guard_status(&self) -> GuardStatus {
        self.guard.status()
    }

    pub fn counts(&self) -> LiveCounts {
        self.counts
    }

    pub fn sha256(&self) -> &[u8; 32] {
        &self.sha
    }

    pub fn guard_config(&self) -> &GuardConfig {
        self.guard.config()
    }

    /// The guard transitions since the last call (the runner logs them).
    pub fn take_transitions(&mut self, out: &mut Vec<Transition>) {
        out.append(&mut self.transitions);
    }

    /// The bytes of the log lines formatted since the last call, replaced by an empty buffer. Call it outside the decision path (the runner
    /// does once a second).
    pub fn take_log(&mut self) -> Vec<u8> {
        if self.log.is_empty() {
            return Vec::new();
        }
        std::mem::replace(&mut self.log, Vec::with_capacity(1 << 16))
    }

    /// The line that opens every run in the log file (and every new file after a rotation): what produced the lines after it. `start_unix` is the
    /// start time (seconds), `server` the server's address (`ip:port`; anything but letters, digits and `.:-_[]` is dropped: no nickname gets in).
    pub fn header_line(&self, start_unix: u64, server: &str) -> String {
        let c = self.guard.config();
        let server: String = server
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || ".:-_[]".contains(*c))
            .take(64)
            .collect();
        let g = &self.gate;
        format!(
            "{{\"ev\":\"open\",\"v\":{LOG_VERSION},\"start\":{start_unix},\"server\":\"{server}\",\"model\":\"{}\",\"gate\":{{\"on\":{},\"dist\":{},\"radius\":{},\"others\":{}}},\"guard\":{{\"windows\":{},\"min\":{},\"margin\":{},\"retry_after\":{},\"retry_margin\":{}}},\"fields\":\"k,dir(a,m,h),hook(a,m,h),aim_mrad(a,m,h),jump(a,m)\"}}\n",
            hex(&self.sha),
            g.enabled,
            g.max_target_dist_px,
            g.others_radius_px,
            g.max_others,
            c.windows,
            c.min_windows,
            c.margin,
            c.retry_after,
            c.retry_margin
        )
    }

    /// One snapshot with a target: scores the open windows against it and feeds the history.
    pub fn observe(&mut self, pair: &Pair<'_>) {
        self.observe_inner(pair);
    }

    fn observe_inner(&mut self, pair: &Pair<'_>) -> bool {
        let Pair {
            world,
            self_id,
            target,
            tag,
        } = *pair;
        let tick = world.tick;
        let tag = TagBuf::new(tag);
        if target != self.target || tag != self.tag {
            // Another opponent (or the same slot with another player): the open windows cannot be resolved any more.
            self.finish_all();
            self.pred.reset();
            self.prev_jumped = None;
            self.target = target;
            self.tag = tag;
            self.last_tick = i32::MIN;
        }
        if tick == self.last_tick {
            return self.last_ok;
        }
        if tick < self.last_tick {
            // The tick went back (a map restart): nothing open can be right any more.
            self.finish_all();
            self.pred.reset();
            self.prev_jumped = None;
        }
        self.last_tick = tick;
        self.last_ok = false;
        let (Some(me), Some(opp)) = (
            TeeFrame::from_world(world, self_id, target),
            TeeFrame::from_world(world, target, self_id),
        ) else {
            self.finish_all();
            return false;
        };
        if !me.alive || !opp.alive {
            self.finish_all();
            self.prev_jumped = None;
            return false;
        }
        // The jump event since the previous snapshot of this opponent (two steps at the usual 25 Hz).
        let jump_event = match self.prev_jumped {
            Some((t, j)) if tick - t <= 4 && tick > t => (opp.jumped & !j) != 0,
            _ => false,
        };
        self.prev_jumped = Some((tick, opp.jumped));
        self.resolve(tick, &opp, me.freeze_left > 0, jump_event);
        // The history: the predictor records the pair's frame when asked with an empty window.
        let mut none: [Option<PredictedInput>; 0] = [];
        self.pred.predict(
            &WindowCtx {
                world,
                self_id,
                victim_id: target,
                in_flight: &[],
            },
            &mut none,
        );
        self.counts.observed += 1;
        self.last_ok = true;
        true
    }

    /// A brain decision at the snapshot tick: predicts the opponent's inputs for the `own.len()` ticks of the window and scores them later.
    ///
    /// `own` are our inputs for those ticks (`LiveWorld::own_inputs_over`), `hold` the opponent's held input in the live world (its
    /// `LiveWorld::held_input_of`). `victim` is cleared and, when the model drives ([`WindowUse::Model`]), filled with the opponent's input for
    /// each tick of the window. The model is run (and scored in the shadow) also while the guard has benched it.
    pub fn window(
        &mut self,
        pair: &Pair<'_>,
        own: &[WireInput],
        hold: &WireInput,
        victim: &mut Vec<WireInput>,
    ) -> WindowUse {
        victim.clear();
        if !self.observe_inner(pair) {
            return WindowUse::Hold;
        }
        let Pair {
            world, self_id, target, ..
        } = *pair;
        let lag = own.len();
        if lag == 0 || lag > WINDOW_MAX {
            self.counts.skipped_window += 1;
            return WindowUse::Hold;
        }
        let Some(opp) = TeeFrame::from_world(world, target, self_id) else {
            return WindowUse::Hold;
        };
        // Our own freeze: states the model never saw (the training games end at the first freeze), and where it is clearly worse than hold.
        let Some(me) = TeeFrame::from_world(world, self_id, target) else {
            return WindowUse::Hold;
        };
        if me.freeze_left > 0 {
            self.counts.skipped_own_frozen += 1;
            return WindowUse::Hold;
        }
        if !self.gate.admits(world, self_id, target, me.pos, opp.pos) {
            self.counts.skipped_regime += 1;
            return WindowUse::Hold;
        }
        // A frozen opponent plays no input (the game zeroes it) for as long as the freeze lasts: nothing to predict, nothing to score.
        if usize::try_from(opp.freeze_left).unwrap_or(0) > lag {
            self.counts.skipped_frozen += 1;
            return WindowUse::Hold;
        }
        self.out = [None; HORIZON];
        self.pred.predict(
            &WindowCtx {
                world,
                self_id,
                victim_id: target,
                in_flight: own,
            },
            &mut self.out,
        );
        self.counts.predicted += 1;
        // The window is only scored when the opponent can act through all of it.
        let used = self.guard.using_model();
        if opp.freeze_left == 0 {
            self.open_window(world.tick, lag as u8, used, &opp);
        }
        if !used {
            return WindowUse::Hold;
        }
        self.counts.used += 1;
        let mut fire = hold.fire;
        for slot in self.out.iter().take(lag) {
            match slot {
                Some(p) => victim.push(victim_input(p, hold, &mut fire)),
                // No prediction for this tick: the held input, as without a model.
                None => victim.push(*hold),
            }
        }
        WindowUse::Model
    }

    fn open_window(&mut self, tick: i32, lag: u8, used: bool, opp: &TeeFrame) {
        // A free slot, else the oldest (finished first, so its samples are not lost).
        let slot = match self.pending.iter().position(|p| !p.live) {
            Some(i) => i,
            None => {
                let oldest = (0..PENDING).min_by_key(|&i| self.pending[i].tick).unwrap_or(0);
                self.finish(oldest);
                oldest
            }
        };
        let hold = Hold {
            dir: opp.direction,
            hook: opp.hook_state > 0,
            aim: norm_angle(f64::from(opp.angle)),
        };
        let mut model = [Pm::default(); HORIZON];
        for (m, o) in model.iter_mut().zip(&self.out) {
            *m = match o {
                Some(p) => Pm {
                    dir: p.direction.clamp(-1, 1) as i8,
                    hook: p.hook,
                    jump: p.jump,
                    aim: norm_angle(p.aim),
                },
                // No prediction: the model said hold.
                None => Pm {
                    dir: hold.dir,
                    hook: hold.hook,
                    jump: false,
                    aim: hold.aim,
                },
            };
        }
        self.pending[slot] = Pending {
            live: true,
            tick,
            lag,
            used,
            tag: self.tag,
            hold,
            model,
            n: 0,
            g_model: 0.0,
            g_hold: 0.0,
            g_n: 0,
            ..Pending::EMPTY
        };
    }

    /// Compares what the snapshot at `tick` shows with every open window. The snapshot shows the state after the step into `tick`, which ran
    /// the input of window tick `k = tick - window_tick - 1`.
    fn resolve(&mut self, tick: i32, opp: &TeeFrame, me_frozen: bool, jump_event: bool) {
        // A snapshot that shows either of us frozen is not scored: the opponent's input is ignored while it is frozen, and we are outside the regime.
        let frozen = opp.freeze_left > 0 || me_frozen;
        let actual = Actual {
            dir: opp.direction,
            hook: opp.hook_state > 0,
            aim: norm_angle(f64::from(opp.angle)),
            jump_event,
        };
        for i in 0..PENDING {
            let p = &mut self.pending[i];
            if !p.live || p.tick >= tick {
                continue;
            }
            let age = tick - p.tick;
            if (1..=HORIZON as i32).contains(&age) && !frozen && p.tag == self.tag {
                let k = (age - 1) as usize;
                let m = p.model[k];
                // The model's jump over the two steps the snapshot interval covers.
                let mj = m.jump || (k > 0 && p.model[k - 1].jump);
                let (cm, ch) = sample_cost(&m, &p.hold, &actual);
                // The guard judges the ticks the roll uses (the window's own); the log keeps every tick.
                if k < guard_ticks(usize::from(p.lag)) {
                    p.g_model += cm;
                    p.g_hold += ch;
                    p.g_n += 1;
                }
                let n = usize::from(p.n);
                p.samples[n] = Sample {
                    k: k as u8,
                    actual,
                    model: Pm { jump: mj, ..m },
                    hold: p.hold,
                };
                p.n += 1;
            }
            // Nothing more can be learned from a window older than its horizon.
            if age >= HORIZON as i32 {
                self.finish(i);
            }
        }
    }

    fn finish_all(&mut self) {
        for i in 0..PENDING {
            if self.pending[i].live {
                self.finish(i);
            }
        }
    }

    /// Closes window `i`: the guard sees its costs, the log its samples.
    fn finish(&mut self, i: usize) {
        let p = self.pending[i];
        self.pending[i].live = false;
        if p.n == 0 {
            return;
        }
        if let Some(t) = self.guard.push(p.g_model, p.g_hold, u16::from(p.g_n)) {
            self.transitions.push(t);
            self.write_transition(p.tick, &t);
        }
        self.write_window(&p);
    }

    fn room(&mut self) -> bool {
        if self.log.len() >= LOG_BUF_LIMIT {
            self.counts.log_dropped += 1;
            return false;
        }
        true
    }

    fn write_transition(&mut self, tick: i32, t: &Transition) {
        if !self.room() {
            return;
        }
        let _ = writeln!(
            self.log,
            "{{\"ev\":\"guard\",\"t\":{tick},\"to\":\"{}\",\"model\":{:.4},\"hold\":{:.4},\"samples\":{},\"windows\":{}}}",
            t.to.name(),
            t.model_cost,
            t.hold_cost,
            t.samples,
            t.windows
        );
    }

    fn write_window(&mut self, p: &Pending) {
        if !self.room() {
            return;
        }
        let mrad = |a: f32| (f64::from(a) * 1000.0).round() as i32;
        let _ = write!(
            self.log,
            "{{\"v\":{LOG_VERSION},\"t\":{},\"w\":{},\"u\":{},\"o\":\"{}\",\"s\":[",
            p.tick,
            p.lag,
            u8::from(p.used),
            p.tag.as_str()
        );
        for (j, s) in p.samples[..usize::from(p.n)].iter().enumerate() {
            if j > 0 {
                let _ = self.log.write_all(b",");
            }
            let _ = write!(
                self.log,
                "[{},{},{},{},{},{},{},{},{},{},{},{}]",
                s.k,
                s.actual.dir,
                s.model.dir,
                s.hold.dir,
                u8::from(s.actual.hook),
                u8::from(s.model.hook),
                u8::from(s.hold.hook),
                mrad(s.actual.aim),
                mrad(s.model.aim),
                mrad(s.hold.aim),
                u8::from(s.actual.jump_event),
                u8::from(s.model.jump),
            );
        }
        let _ = self.log.write_all(b"]}\n");
    }
}

/// Lower-case hex of a digest.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

impl std::fmt::Debug for LiveOpp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveOpp")
            .field("target", &self.target)
            .field("guard", &self.guard.status())
            .field("counts", &self.counts)
            .finish_non_exhaustive()
    }
}

/// What the bot shows of the guard state in STATUS: `"on"` while the model drives the window, `"hold"` while the guard has benched it.
pub fn state_word(s: GuardState) -> &'static str {
    match s {
        GuardState::Active => "on",
        GuardState::Fallback => "hold",
    }
}

#[cfg(test)]
mod tests;
