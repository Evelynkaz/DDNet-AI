//! The offline replay of a clip: the bot's own physics, from the clip alone, bit for bit.
//!
//! **Why it can be exact.** The server quantizes every character core each tick
//! (`CCharacter::TickDeferred`: `m_Core.Move(); m_Core.Quantize()`), so the snapshot's integers *are*
//! the state of the next tick, and `ddai-physics` is a bit-exact port of that physics. Starting from the
//! recorded state of a frame and stepping it with the inputs we actually sent (the ones the server
//! applied, [`crate::SentRec`]) must give the recorded state of the next frame, integer for integer.
//!
//! **Two modes** ([`Mode`]):
//!
//! * [`Mode::Resync`]: every frame starts from the recorded state of *all* tees; one step to the next
//!   frame is compared. Finds where the physics and the recording disagree, one step at a time.
//! * [`Mode::FreeRun`]: the spec's replay. Our own tee starts from the first frame and **runs free** on
//!   the recorded inputs; the other tees are taken from their recorded state at each frame (they are
//!   stepped on their guessed inputs only between two frames). Our position, velocity and hook state
//!   must match the recorded frames tick for tick; the first frame where they do not is reported with
//!   its cause, and the replay stops there ([`Report::first_divergence`]).
//!
//! **Causes** need a physical reason in the clip (review F3): a death or `Cl_Kill` around the step
//! ([`Cause::Respawn`]), a tele-in we stood at or a tele-out we landed on ([`Cause::Teleport`]), another tee
//! hook-linked to us, with its hook tip at us, or within body / hammer reach (100 px; [`Cause::OtherTee`]), a
//! projectile within 140 px ([`Cause::Projectile`]), an input whose timing the server had not confirmed
//! ([`Cause::InputTiming`]), or our own freeze state changing in the step ([`Cause::FreezeChange`]). Anything else is [`Cause::ServerCorrection`]: the server changed our state and
//! nothing in the clip says why.

use std::sync::Arc;

use ddai_net::generated::objects;
use ddai_net::tuning::{TeamsState, TuneParams, from_array};
use ddai_net::view::CharacterView;
use ddai_physics::core::{NetCharacterCore, PlayerInput};
use ddai_physics::map::MapData;
use ddai_world::{LiveWorld, OwnState, SnapshotInput, player_input_from_net};

use crate::format::{Clip, ClipEvent, Frame, TeeRec};

/// What the replay carries between frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Every frame from the recorded state, one step each.
    Resync,
    /// Our own tee free-running from the first frame.
    FreeRun,
}

/// Why a step did not reproduce the recorded frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cause {
    /// Another tee was hook-linked to us (either way), had its hook in flight on a line that reaches us within the step
    /// ([`flying_hook_reaches`]), or was within body / hammer reach ([`BODY_REACH_PX`]): its input between the two frames
    /// is a guess (the replay holds the last input it knew, so a hook the tee let go of in between is still flying
    /// and catches us in the replay, but not on the server).
    OtherTee { id: i32 },
    /// A projectile was within hit / explosion reach ([`PROJECTILE_REACH_PX`]).
    Projectile,
    /// We died or asked for `Cl_Kill` around the step (a kill message, a `Cl_Kill` of ours, a respawn event within
    /// a frame of the step): the tee reappeared at a spawn the server chose. Not a physics step, by design not
    /// reproduced; the replay restarts our tee from the recording there.
    Respawn,
    /// A jump physics cannot produce (beyond the speed cap times the step plus 64 px) with a tele-in swept by the
    /// hitbox or a tele-out landed on: a teleporter with several exits picks one with the server's own random
    /// number. A small difference next to a tele tile, or a jump with no tele tile, is not this (review F5).
    Teleport,
    /// Our freeze state changed in the step (a freeze or a thaw). Not excused by design (it stops a free run and
    /// counts against exactness); named because `LiveWorld` rebuilds the tee from a snapshot without its
    /// `prev_pos` (snapped to the current position, so the anti-skip tile walk is lost) and without
    /// `frozen_last_tick`, and at freeze / unfreeze tile edges both decide the order. `m_FreezeEnd` itself is
    /// absolute and reproduced exactly. A known `ddai-world` gap (follow-up 2.4c), not a property of the server.
    /// Measured: about three quarters of such steps still reproduce exactly (`Report::freeze_exact`).
    FreezeChange,
    /// The replay applies our sent inputs on the ticks they were tagged for; the server did not always. Either an input of
    /// the step was not confirmed by the server's timing report when it was recorded (or the step is longer than the inputs
    /// the frame holds), or it is **proved** that the server applied an input late and said nothing (it reports the timing
    /// of the newest input of a packet only): (1) the replay with every input of the step applied one or two ticks later, or the newest input of the step applied to every tick of it, or
    /// with one tick's input never applied (the previous input holds), reproduces the recorded state exactly, or (2) a stale, unfrozen own core contradicts the sent input
    /// ([`stale_core_contradicts_sent`]). Not a property of our physics.
    InputTiming,
    /// Nothing the clip knows explains it: the server changed our state in a way the physics does not model
    /// or the frame does not hold (a correction).
    ServerCorrection,
}

impl Cause {
    /// Short name for tables.
    pub fn name(&self) -> &'static str {
        match self {
            Cause::OtherTee { .. } => "other-tee",
            Cause::Projectile => "projectile",
            Cause::Respawn => "respawn",
            Cause::Teleport => "teleport",
            Cause::InputTiming => "input-timing",
            Cause::FreezeChange => "freeze-change",
            Cause::ServerCorrection => "server-correction",
        }
    }

    /// Not physics: the server's own choice (a spawn after a death, a teleporter exit), by design not
    /// reproduced from a clip.
    pub fn is_by_design(&self) -> bool {
        matches!(self, Cause::Respawn | Cause::Teleport)
    }
}

/// A recorded integer and what the physics gave.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Divergence {
    /// Index of the frame whose recorded state was not reproduced.
    pub frame: usize,
    pub tick: i32,
    /// Which field of the core (`x`, `y`, `vel_x`, `vel_y`, `hook_state`, ...).
    pub field: &'static str,
    pub recorded: i32,
    pub replayed: i32,
    /// All the fields that differed in this step.
    /// Every field that differed: `(name, reconstructed, replayed)`.
    pub fields: Vec<(&'static str, i32, i32)>,
    pub cause: Cause,
    /// The recorded core was fresh (its tick is the frame's): `recorded` is then the server's own integer.
    pub fresh: bool,
    /// Our freeze state changed in this step (a freeze or a thaw).
    pub freeze_change: bool,
    /// Nobody who could touch us was around (see [`Report::isolated_steps`]).
    pub isolated: bool,
    /// `cause` is [`Cause::InputTiming`] by proof (a late input the server applied without a report), not by a missing report.
    pub input_timing_proved: bool,
}

/// What a replay found.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Report {
    pub frames: usize,
    /// Frame pairs compared (both have our tee).
    pub steps: usize,
    /// Of them, reproduced exactly (every field of the core).
    pub exact: usize,
    /// Steps into a frame whose own core was fresh (sent by the server for that very tick), and how many of
    /// those were reproduced exactly: the strictest evidence, free of the client-side reckoning.
    pub fresh_steps: usize,
    pub fresh_exact: usize,
    /// Steps with nobody who could touch us (no other tee within 420 px or linked by a hook in either frame, no
    /// projectile, no tee left out of the clip), and how many of those were reproduced exactly. With no one
    /// else involved the clip holds everything the physics needs, so these are the pure test of our own physics.
    pub isolated_steps: usize,
    pub isolated_exact: usize,
    /// Steps where our freeze state changed (a freeze or a thaw), and how many of those reproduced exactly; and the
    /// same for the ones into a fresh core. Input gating switches at the freeze timer's tick, which a snapshot
    /// only knows to the tick.
    pub freeze_steps: usize,
    pub freeze_exact: usize,
    pub freeze_fresh_steps: usize,
    pub freeze_fresh_exact: usize,
    /// Steps skipped because our tee is absent from one of the frames (a death and its respawn).
    pub skipped_deaths: usize,
    /// Steps over a respawn or a teleport (see [`Cause::Respawn`], [`Cause::Teleport`]): counted in
    /// [`Report::steps`] and listed in [`Report::divergences`], but not unexplained.
    pub respawn_steps: usize,
    /// The first step no death explains, with its cause (`None`: everything reproduced but the respawns).
    pub first_divergence: Option<Divergence>,
    /// Every divergent step (in [`Mode::Resync`]; [`Mode::FreeRun`] stops at the first).
    pub divergences: Vec<Divergence>,
}

impl Report {
    /// Every compared step was exact.
    pub fn is_exact(&self) -> bool {
        self.steps > 0 && self.exact == self.steps
    }

    /// Steps with nobody near that differ for a reason that is the server's choice or timing, not our physics: respawns,
    /// teleports and proved late inputs.
    pub fn isolated_designed(&self) -> usize {
        self.divergences
            .iter()
            .filter(|d| {
                d.isolated && (d.cause.is_by_design() || (d.cause == Cause::InputTiming && d.input_timing_proved))
            })
            .count()
    }

    /// Every compared step was exact except the ones over a respawn or a teleport.
    pub fn is_exact_but_respawns(&self) -> bool {
        self.steps > 0 && self.exact + self.respawn_steps == self.steps
    }

    /// The divergences nothing the clip knows explains: the server changed our state and no death, teleport,
    /// other tee, projectile, unrecorded tee or unconfirmed input timing is around ([`Cause::ServerCorrection`]).
    pub fn unexplained(&self) -> impl Iterator<Item = &Divergence> {
        self.divergences.iter().filter(|d| d.cause == Cause::ServerCorrection)
    }

    /// Our freeze-change steps by outcome: `exact`, or the cause name of the divergence.
    pub fn freeze_breakdown(&self) -> std::collections::BTreeMap<&'static str, usize> {
        let mut m = std::collections::BTreeMap::new();
        if self.freeze_exact > 0 {
            m.insert("exact", self.freeze_exact);
        }
        for d in self.divergences.iter().filter(|d| d.freeze_change) {
            *m.entry(d.cause.name()).or_default() += 1;
        }
        m
    }

    /// The divergences by cause.
    pub fn by_cause(&self) -> std::collections::BTreeMap<&'static str, usize> {
        let mut m = std::collections::BTreeMap::new();
        for d in &self.divergences {
            *m.entry(d.cause.name()).or_default() += 1;
        }
        m
    }

    /// The number of exact steps before the first divergence (all of them when there is none).
    pub fn exact_prefix(&self) -> usize {
        self.first_divergence
            .as_ref()
            .map_or(self.exact, |d| d.frame.saturating_sub(1))
    }
}

/// The tuning in force at `tick`.
fn tuning_at(clip: &Clip, tick: i32) -> TuneParams {
    let mut t = ddai_net::tuning::DEFAULT_TUNE_PARAMS;
    for c in &clip.header.tuning {
        if c.from_tick <= tick && c.values.len() == ddai_net::tuning::NUM_TUNE_PARAMS {
            let mut a = [0i32; ddai_net::tuning::NUM_TUNE_PARAMS];
            a.copy_from_slice(&c.values);
            t = from_array(c.received as usize, a);
        }
    }
    t
}

fn teams_at(clip: &Clip, tick: i32) -> Option<TeamsState> {
    let mut out = None;
    for c in &clip.header.teams {
        if c.from_tick <= tick {
            let mut teams = [0i32; 128];
            for (i, t) in c.teams.iter().take(128).enumerate() {
                teams[i] = *t;
            }
            out = Some(TeamsState {
                teams,
                received: (c.received as usize).min(128),
            });
        }
    }
    out
}

fn views(f: &Frame) -> Vec<CharacterView> {
    f.tees
        .iter()
        .map(|t| CharacterView {
            id: t.id,
            character: t.ch.to_net(),
            ddnet: t.dd.map(|d| d.to_net()),
        })
        .collect()
}

/// Feeds `frame` to the world the way the bot did (public for the offline tools that read a clip's frames as worlds, task 3.15).
pub fn feed(lw: &mut LiveWorld, clip: &Clip, frame: &Frame) {
    let characters = views(frame);
    let projectiles: Vec<_> = frame.projectiles.iter().map(|p| p.to_view()).collect();
    let switches: Vec<(i32, objects::SwitchState)> = frame.switches.iter().map(|s| s.to_net()).collect();
    let teams = teams_at(clip, frame.tick);
    lw.on_snapshot(SnapshotInput {
        tick: frame.tick,
        characters: &characters,
        tuning: tuning_at(clip, frame.tick),
        switch_states: &switches,
        teams: teams.as_ref(),
        own_input_at_tick: frame.sent.last().map(|s| player_input_from_net(s.input.to_net())),
        projectiles: &projectiles,
    });
}

/// The inputs of the step into `next`: `(tick, input)` for every tick since the previous frame.
fn inputs_of(next: &Frame) -> Vec<(i32, PlayerInput)> {
    next.sent
        .iter()
        .map(|s| (s.tick, player_input_from_net(s.input.to_net())))
        .collect()
}

/// Compares the replayed own core with the reconstructed one; the differing fields as
/// `(name, reconstructed, replayed)`, in the order of the core's own fields.
fn compare(replayed: &NetCharacterCore, recon: &NetCharacterCore) -> Vec<(&'static str, i32, i32)> {
    let mut d = Vec::new();
    macro_rules! cmp {
        ($name:literal, $f:ident) => {
            if replayed.$f != recon.$f {
                d.push(($name, recon.$f, replayed.$f));
            }
        };
    }
    cmp!("x", x);
    cmp!("y", y);
    cmp!("vel_x", vel_x);
    cmp!("vel_y", vel_y);
    cmp!("hook_state", hook_state);
    cmp!("hooked_player", hooked_player);
    cmp!("hook_x", hook_x);
    cmp!("hook_y", hook_y);
    cmp!("hook_dx", hook_dx);
    cmp!("hook_dy", hook_dy);
    cmp!("hook_tick", hook_tick);
    cmp!("jumped", jumped);
    cmp!("direction", direction);
    cmp!("angle", angle);
    d
}

fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    ddai_libm::hypot(a.0 - b.0, a.1 - b.1)
}

/// Nobody the clip cannot see into could touch us in the step (see [`Report::isolated_steps`]).
fn is_isolated(prev: &Frame, next: &Frame, own_id: i32) -> bool {
    if !prev.projectiles.is_empty() || !next.projectiles.is_empty() || prev.tees_dropped > 0 || next.tees_dropped > 0 {
        return false;
    }
    [prev, next].iter().all(|f| {
        let Some(me) = f.tee(own_id) else { return false };
        f.tees.iter().filter(|t| t.id != own_id).all(|t| {
            let linked = t.ch.hooked_player == own_id || me.ch.hooked_player == t.id;
            !linked && dist(t.pos(), me.pos()) >= ISOLATED_PX
        })
    })
}

/// "Nobody near" for [`Report::isolated_steps`]: further than a hook is long (380) with a margin.
const ISOLATED_PX: f64 = 420.0;
/// Another tee closer than this can touch us without a hook: bodies (28 px), the hammer (about 56-96 px).
pub const BODY_REACH_PX: f64 = 100.0;
/// A projectile closer than this can hit us or explode on us (a grenade's blast reaches about 135 px).
pub const PROJECTILE_REACH_PX: f64 = 140.0;
/// An other tee's flying hook that passes this close to us may have caught us (the server's own test is 30 px,
/// `PhysSize + 2`, from the segment the hook covers in a tick; the rest is slack for positions that are a tick or two old).
const HOOK_TIP_PX: f64 = 60.0;
/// A flying hook covers `hook_fire_speed` per tick (`tuning`'s default 80 px) along its direction.
const HOOK_FIRE_SPEED_PX: f64 = 80.0;
/// How far from its owner a hook can be: `hook_length` (380 px) and the hitbox (28 px).
const HOOK_OWNER_RADIUS_PX: f64 = 380.0 + 28.0;

/// Shortest distance from `p` to the segment `a..b`.
fn point_segment_dist(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (abx, aby) = (b.0 - a.0, b.1 - a.1);
    let len2 = abx * abx + aby * aby;
    let t = if len2 > 0.0 {
        (((p.0 - a.0) * abx + (p.1 - a.1) * aby) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    dist(p, (a.0 + abx * t, a.1 + aby * t))
}

/// Whether the flying hook of `tee` (as `frame` holds it, with `window` the tick span of the step) was, or could have
/// been, on a line that reaches `me` within the step, whatever the tee did with it in between (the clip holds the
/// tee's hook only at the two frames). The hook flies [`HOOK_FIRE_SPEED_PX`] per tick along `hook_dx / hook_dy` (the
/// unit direction times 256): from the tip as the earlier frame holds it it flies on; the tip in the later frame came
/// from behind. The tee's core may be a few ticks older than its frame (a snapshot only carries what changed), so the
/// reach counts those ticks too. A hook that is not in flight reaches nobody.
fn flying_hook_reaches(frame: &Frame, later: bool, tee: &TeeRec, me: (f64, f64), window: i32) -> bool {
    if tee.ch.hook_state != ddai_physics::core::HOOK_FLYING {
        return false;
    }
    let tip = (f64::from(tee.ch.hook_x), f64::from(tee.ch.hook_y));
    let (dx, dy) = (f64::from(tee.ch.hook_dx) / 256.0, f64::from(tee.ch.hook_dy) / 256.0);
    let norm = ddai_libm::hypot(dx, dy);
    let (ux, uy) = if norm > 1e-9 {
        (dx / norm, dy / norm)
    } else {
        (0.0, 0.0)
    };
    let lag = (frame.tick - tee.ch.tick).max(0);
    let mut reach = HOOK_FIRE_SPEED_PX * f64::from(window.max(1) + lag + 1);
    let sign = if later { -1.0 } else { 1.0 };
    // A flying hook never gets further than `hook_length` (380 px) from its owner (plus the hitbox): a core many ticks old
    // must not stretch the line over the map. Clip the line to that disc around the owner.
    let owner = tee.pos();
    let (wx, wy) = (tip.0 - owner.0, tip.1 - owner.1);
    let (b, c) = (
        wx * ux * sign + wy * uy * sign,
        wx * wx + wy * wy - HOOK_OWNER_RADIUS_PX * HOOK_OWNER_RADIUS_PX,
    );
    if c > 0.0 {
        reach = 0.0; // the tip is already beyond the owner's reach: only the tip itself counts
    } else {
        reach = reach.min(-b + (b * b - c).max(0.0).sqrt());
    }
    let end = (tip.0 + ux * reach * sign, tip.1 + uy * reach * sign);
    point_segment_dist(me, tip, end) < HOOK_TIP_PX
}

/// The tele tiles on our way: a tele-in the hitbox swept over between `from` and where physics would have taken
/// us (`to`), or a tele-out the hitbox touches at the recorded end (`landed`). The server then picked the exit
/// with its own random number.
fn tele_involved(map: &MapData, from: (f64, f64), to: (f64, f64), landed: (f64, f64)) -> bool {
    use ddai_physics::map::{
        TILE_TELECHECKIN, TILE_TELECHECKINEVIL, TILE_TELECHECKOUT, TILE_TELEIN, TILE_TELEINEVIL, TILE_TELEOUT,
    };
    let Some(tele) = map.tele.as_ref() else { return false };
    let (w, h) = (i64::from(map.width), i64::from(map.height));
    let kind_at = |x: f64, y: f64| -> u8 {
        let (tx, ty) = ((x / 32.0).floor() as i64, (y / 32.0).floor() as i64);
        if tx < 0 || ty < 0 || tx >= w || ty >= h {
            return 0;
        }
        tele.get((ty * w + tx) as usize).map_or(0, |t| t.kind)
    };
    // The hitbox (28 px) along the segment, every 8 px: its four corners and its centre.
    let corners = [(0.0, 0.0), (-14.0, -14.0), (14.0, -14.0), (-14.0, 14.0), (14.0, 14.0)];
    let len = dist(from, to);
    let steps = (len / 8.0).ceil().max(1.0) as i32;
    for k in 0..=steps {
        let t = f64::from(k) / f64::from(steps);
        let (x, y) = (from.0 + (to.0 - from.0) * t, from.1 + (to.1 - from.1) * t);
        if corners.iter().any(|(dx, dy)| {
            matches!(
                kind_at(x + dx, y + dy),
                TILE_TELEIN | TILE_TELEINEVIL | TILE_TELECHECKIN | TILE_TELECHECKINEVIL
            )
        }) {
            return true;
        }
    }
    corners
        .iter()
        .any(|(dx, dy)| matches!(kind_at(landed.0 + dx, landed.1 + dy), TILE_TELEOUT | TILE_TELECHECKOUT))
}

/// Our own sent input at `tick`, looked up in the frames up to `last` (the clip keeps the inputs of the ticks since the
/// previous frame in each frame).
fn sent_at(frames: &[Frame], last: usize, tick: i32) -> Option<crate::format::InputRec> {
    frames[last.saturating_sub(64)..=last]
        .iter()
        .rev()
        .flat_map(|f| f.sent.iter())
        .find(|s| s.tick == tick)
        .map(|s| s.input)
}

/// Proof that the server applied one of our inputs late (see [`Cause::InputTiming`]). The server compares, every tick, the
/// character it would send against the client's own prediction (`CCharacter::TickDeferred`: the written `CNetObj_Character`,
/// which holds `m_Direction` and `m_Angle`) and resends the core on any difference. So an own core that is **older than its
/// frame** says the server's direction and aim stayed those of the core for every tick since. If an input we sent for one of
/// those ticks says otherwise (a different direction, or a target whose angle differs), the server had not applied it yet.
/// Looked at in frame `i - 1` (where the step starts) and in frame `i` (where it ends). Freeze zeroes a tee's direction, jump and hook
/// but keeps its target (`character.cpp:2248-2250`), and the core's angle follows the target every tick (`gamecore.cpp:211-223`),
/// so a frozen tee is judged by the angle only.
fn stale_core_contradicts_sent(frames: &[Frame], i: usize, own_id: i32) -> bool {
    [i - 1, i].into_iter().any(|j| {
        let f = &frames[j];
        let Some(t) = f.tee(own_id) else { return false };
        if t.ch.tick >= f.tick {
            return false;
        }
        (t.ch.tick + 1..=f.tick).any(|k| {
            sent_at(frames, j, k).is_some_and(|s| {
                // The server never aims at the centre: (0, 0) becomes (0, -1) (`CCharacter::OnPredictedInput`).
                let (tx, ty) = if (s.target_x, s.target_y) == (0, 0) {
                    (0, -1)
                } else {
                    (s.target_x, s.target_y)
                };
                // Freeze zeroes the direction, jump and hook (`character.cpp:2248-2250`) but keeps the target, and
                // `CCharacterCore::Tick` recomputes the angle from it every tick, frozen or not (`gamecore.cpp:211-223`):
                // so the angle is judged for a frozen tee too, the direction only for a free one.
                ddai_physics::core::angle_from_target(tx, ty) != t.ch.angle
                    || (!t.frozen && s.direction != t.ch.direction)
            })
        })
    })
}

/// Whether a step is proved to be a late or lost input of ours. The reproduction proof (some alternative input assumption gives the
/// recorded core bit for bit) is accepted only when the end frame's own core is **fresh**, the server's own integers: with a stale
/// end core the recording is `LiveWorld`'s reckoning and a handful of fake corrections were "reproduced" by chance (review F8). A stale
/// core is the contradiction proof's business ([`stale_core_contradicts_sent`]).
fn input_timing_proved_by(end_core_fresh: bool, reproduced: bool, contradicted: bool) -> bool {
    (end_core_fresh && reproduced) || contradicted
}

/// What was around at the step `prev -> next` that the replay cannot know. Each named cause needs a physical
/// reason in the clip; with none, the step is [`Cause::ServerCorrection`] (review F3).
pub(crate) fn cause_of(map: &MapData, frames: &[Frame], i: usize, own_id: i32, replayed: (f64, f64)) -> Cause {
    let (prev, next) = (&frames[i - 1], &frames[i]);
    let (Some(a), Some(b)) = (prev.tee(own_id), next.tee(own_id)) else {
        return Cause::ServerCorrection;
    };
    // A death announced for us (or our respawn) within a frame of the step: the server put us at a spawn,
    // however near. Our own `Cl_Kill` a few frames earlier counts too, when the tee has moved away.
    let near = &frames[i.saturating_sub(1)..(i + 2).min(frames.len())];
    let earlier = &frames[i.saturating_sub(8)..=i];
    let announced = near.iter().flat_map(|f| f.events.iter()).any(|e| {
        matches!(e, ClipEvent::Kill { victim, .. } if *victim == own_id)
            || matches!(e, ClipEvent::Respawn { id } if *id == own_id)
    });
    let asked = earlier
        .iter()
        .flat_map(|f| f.events.iter())
        .any(|e| matches!(e, ClipEvent::KillSent { .. }));
    if announced || (asked && dist(a.pos(), b.pos()) > 64.0) {
        return Cause::Respawn;
    }
    // Another tee that could have touched us: linked by a hook, within body / hammer reach, or with its hook
    // in flight on a line that reaches us. Its inputs between the two frames are not in the clip.
    let window = next.tick - prev.tick;
    let mut other: Option<(f64, i32)> = None;
    for (f, later) in [(prev, false), (next, true)] {
        let Some(me) = f.tee(own_id) else { continue };
        for t in &f.tees {
            if t.id == own_id {
                continue;
            }
            let linked = t.ch.hooked_player == own_id || me.ch.hooked_player == t.id;
            let d = dist(t.pos(), me.pos());
            let tip = flying_hook_reaches(f, later, t, me.pos(), window);
            let reach = if linked || tip { 0.0 } else { d };
            if (linked || tip || d < BODY_REACH_PX) && other.is_none_or(|(nd, _)| reach < nd) {
                other = Some((reach, t.id));
            }
        }
    }
    if let Some((_, id)) = other {
        return Cause::OtherTee { id };
    }
    // A teleporter (after `OtherTee`: a fight next to one is a fight). It needs a jump physics cannot produce
    // (the recorded move, or the gap between the replayed and the recorded end, beyond what a tee can travel in
    // the step plus 64 px) AND a tele tile on the way: a tele-in the hitbox swept over, or a tele-out it landed on.
    let reach = 64.0 + 30.0 * f64::from((next.tick - prev.tick).max(1));
    let moved = dist(a.pos(), b.pos());
    let gap = dist(replayed, b.pos());
    if (moved > reach || gap > reach) && tele_involved(map, a.pos(), replayed, b.pos()) {
        return Cause::Teleport;
    }
    // A projectile that could have hit or exploded on us.
    for f in [prev, next] {
        let Some(me) = f.tee(own_id) else { continue };
        if f.projectiles
            .iter()
            .any(|p| dist((f64::from(p.v[0]), f64::from(p.v[1])), me.pos()) < PROJECTILE_REACH_PX)
        {
            return Cause::Projectile;
        }
    }
    let span = next.tick - prev.tick;
    let covered = next.sent.len() as i32;
    if span > covered || next.sent.iter().any(|s| !s.timing_known) {
        return Cause::InputTiming;
    }
    if a.frozen != b.frozen {
        return Cause::FreezeChange;
    }
    // Tees left out of the frame (`tees_dropped`) explain nothing: the nearest are the ones kept, so a left-out
    // tee is never nearer than the farthest one recorded, and none of those was within reach.
    Cause::ServerCorrection
}

/// How the server may have applied our inputs of the step other than on the ticks they were tagged for (see
/// [`Cause::InputTiming`]): every input one or two ticks later, the newest input on every tick, or one tick's input never applied. Our own core
/// after the step into frame `i` under each such assumption, from the same base as the normal prediction; an assumption that
/// changes nothing (the inputs of the step are all the same as the one before it) is left out.
fn alternative_input_cores(
    lw: &mut LiveWorld,
    frames: &[Frame],
    i: usize,
    own_id: i32,
    inputs: &[(i32, PlayerInput)],
) -> Vec<NetCharacterCore> {
    let Some(before) = frames[i - 1].sent.last() else {
        return Vec::new();
    };
    let before = player_input_from_net(before.input.to_net());
    let mut variants: Vec<Vec<(i32, PlayerInput)>> = Vec::new();
    // Every input `d` ticks later than it was sent for (the input in force then, from the sent history).
    for d in 1..=2 {
        let late: Vec<_> = inputs
            .iter()
            .map(|(tick, _)| {
                let rec = sent_at(frames, i, tick - d);
                (*tick, rec.map_or(before, |r| player_input_from_net(r.to_net())))
            })
            .collect();
        variants.push(late);
    }
    // The newest input of the step applied to every tick of it: a tick nobody claimed gets the latest *received* input
    // (`m_aLastPlayerInput`, `CGameContext::OnClientPredictedInput`), which can be the one tagged for a later tick.
    if let Some((_, newest)) = inputs.last() {
        variants.push(inputs.iter().map(|(tick, _)| (*tick, *newest)).collect());
    }
    // One tick's input never applied (its claim was lost): the input in force before it holds, the rest as sent. Covers a
    // one-tick pulse of a button the server did not see as well as a change of direction that skipped a tick.
    for j in 0..inputs.len() {
        let mut dropped = inputs.to_vec();
        dropped[j].1 = if j == 0 { before } else { inputs[j - 1].1 };
        variants.push(dropped);
    }
    variants
        .into_iter()
        .filter(|v| v.as_slice() != inputs)
        .filter_map(|v| {
            lw.predict(frames[i].tick, &v)
                .cores
                .get(own_id as u8)
                .map(|c| c.write())
        })
        .collect()
}

/// Replays `clip` on `map` in `mode`.
pub fn replay(clip: &Clip, map: Arc<MapData>, mode: Mode) -> Report {
    replay_with(clip, map, mode, false)
}

/// [`replay`] with the choice of reconstruction: `legacy_prev_pos` restores the pre-2.4c `m_PrevPos` snap in the
/// `LiveWorld` (A/B measurements of task 2.4c only).
pub fn replay_with(clip: &Clip, map: Arc<MapData>, mode: Mode, legacy_prev_pos: bool) -> Report {
    let own_id = clip.header.own_id;
    let mut report = Report {
        frames: clip.frames.len(),
        ..Report::default()
    };
    if clip.frames.len() < 2 {
        return report;
    }
    let mut lw = LiveWorld::new(Arc::clone(&map), own_id, clip.header.world_seed);
    lw.set_snap_prev_pos(legacy_prev_pos);
    // Our own tee carried from the previous prediction (free-run only).
    let mut carried: Option<OwnState> = None;
    feed(&mut lw, clip, &clip.frames[0]);
    for i in 1..clip.frames.len() {
        let (prev, next) = (&clip.frames[i - 1], &clip.frames[i]);
        let have = prev.tee(own_id).is_some() && next.tee(own_id).is_some() && prev.own_alive && next.own_alive;
        if !have {
            // A death (our tee absent from a frame): nothing to compare; start over from the recording.
            report.skipped_deaths += 1;
            carried = None;
            feed(&mut lw, clip, next);
            continue;
        }
        if mode == Mode::FreeRun
            && let Some(own) = &carried
        {
            lw.import_own(own);
        }
        let inputs = inputs_of(next);
        let own_core = lw
            .predict(next.tick, &inputs)
            .cores
            .get(own_id as u8)
            .map(|c| c.write());
        let exported = lw.export_own_predicted();
        let alternatives = alternative_input_cores(&mut lw, &clip.frames, i, own_id, &inputs);
        // The recorded frame as the bot saw it: the snapshot's own core, evolved ("reckoned") from its own tick
        // to the snapshot's, which is what `on_snapshot` builds. Where the core is fresh (its tick is the
        // snapshot's) these are the server's own integers.
        feed(&mut lw, clip, next);
        let recon = lw.base_world().cores.get(own_id as u8).map(|c| c.write());
        let (Some(core), Some(recon)) = (own_core, recon) else {
            report.skipped_deaths += 1;
            carried = None;
            continue;
        };
        let fresh = next.tee(own_id).is_some_and(|t| t.ch.tick == next.tick);
        let diffs = compare(&core, &recon);
        report.steps += 1;
        report.fresh_steps += usize::from(fresh);
        let isolated = is_isolated(prev, next, own_id);
        report.isolated_steps += usize::from(isolated);
        let freeze_change = prev.tee(own_id).map(|t| t.frozen) != next.tee(own_id).map(|t| t.frozen);
        report.freeze_steps += usize::from(freeze_change);
        report.freeze_fresh_steps += usize::from(freeze_change && fresh);
        if freeze_change && diffs.is_empty() {
            report.freeze_exact += 1;
            report.freeze_fresh_exact += usize::from(fresh);
        }
        if diffs.is_empty() {
            report.exact += 1;
            report.fresh_exact += usize::from(fresh);
            report.isolated_exact += usize::from(isolated);
        } else {
            let mut cause = cause_of(&map, &clip.frames, i, own_id, (f64::from(core.x), f64::from(core.y)));
            let mut input_timing_proved = false;
            // Proofs for a step nothing else explains (or only a freeze change): the server did not apply our inputs as
            // sent and said nothing. (1) With the inputs one or two ticks late, the newest input on every tick, or one tick's input never
            // applied, the recorded state comes out exactly. (2) A stale own core contradicts the sent input.
            // A step already named `InputTiming` for want of a timing report is proved here too (it counts as designed only then).
            if matches!(
                cause,
                Cause::ServerCorrection | Cause::FreezeChange | Cause::InputTiming
            ) && input_timing_proved_by(
                fresh,
                alternatives.iter().any(|c| compare(c, &recon).is_empty()),
                stale_core_contradicts_sent(&clip.frames, i, own_id),
            ) {
                cause = Cause::InputTiming;
                input_timing_proved = true;
            }
            // Our own hook grabbed (or missed) a tee in the replay that the recording says otherwise: that tee's real place
            // between the frames is not in the clip.
            if cause == Cause::ServerCorrection
                && let Some(&(_, recorded_id, replayed_id)) = diffs.iter().find(|d| d.0 == "hooked_player")
            {
                let id = if replayed_id != -1 { replayed_id } else { recorded_id };
                if id != -1 && id != own_id {
                    cause = Cause::OtherTee { id };
                }
            }
            let (field, recorded, got) = diffs[0];
            let d = Divergence {
                frame: i,
                tick: next.tick,
                field,
                recorded,
                replayed: got,
                fields: diffs.clone(),
                cause,
                fresh,
                freeze_change,
                isolated,
                input_timing_proved,
            };
            let respawn = d.cause.is_by_design();
            if !respawn && report.first_divergence.is_none() {
                report.first_divergence = Some(d.clone());
            }
            report.respawn_steps += usize::from(respawn);
            report.divergences.push(d);
            if respawn {
                // Not physics: the tee is somewhere new. Start our tee again from the recording.
                carried = None;
                continue;
            }
            if mode == Mode::FreeRun {
                return report;
            }
        }
        carried = exported;
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{CharRec, InputRec, SentRec};
    use ddai_physics::core::angle_from_target;

    /// Frames with only our tee (id 0): a stale core of tick `core_tick` holding `dir` and the aim at `target`, and our sent
    /// inputs `sent` (tick, direction, target) in the frame at tick 110.
    fn frames(
        core_tick: i32,
        dir: i32,
        target: (i32, i32),
        frozen: bool,
        sent: &[(i32, i32, (i32, i32))],
    ) -> Vec<Frame> {
        let tee = TeeRec {
            id: 0,
            ch: CharRec {
                tick: core_tick,
                direction: dir,
                angle: angle_from_target(target.0, target.1),
                ..CharRec::default()
            },
            frozen,
            ..TeeRec::default()
        };
        let mk = |tick: i32, sent: Vec<SentRec>| Frame {
            tick,
            own_alive: true,
            tees: vec![tee],
            sent,
            ..Frame::default()
        };
        let sent = sent
            .iter()
            .map(|&(tick, direction, (target_x, target_y))| SentRec {
                tick,
                input: InputRec {
                    direction,
                    target_x,
                    target_y,
                    ..InputRec::default()
                },
                timing_known: true,
            })
            .collect();
        vec![mk(108, Vec::new()), mk(110, sent)]
    }

    #[test]
    fn the_reproduction_proof_needs_a_fresh_end_core_the_contradiction_proof_does_not() {
        // Reproduced by chance against a stale (reckoned) end core: not a proof (review F8: fake corrections were "explained").
        assert!(!input_timing_proved_by(false, true, false));
        assert!(input_timing_proved_by(true, true, false));
        assert!(!input_timing_proved_by(true, false, false));
        // A stale core that contradicts the sent input proves it on its own.
        assert!(input_timing_proved_by(false, false, true));
    }

    #[test]
    fn a_zero_target_is_the_servers_minus_one_in_the_angle_check() {
        // A sent target (0, 0) is aimed at (0, -1) by the server: a core with that angle does not contradict it.
        let up = (0, -1);
        assert!(!stale_core_contradicts_sent(
            &frames(108, 1, up, false, &[(110, 1, (0, 0))]),
            1,
            0
        ));
        assert!(stale_core_contradicts_sent(
            &frames(108, 1, (1, 0), false, &[(110, 1, (0, 0))]),
            1,
            0
        ));
    }

    #[test]
    fn a_stale_unfrozen_core_that_contradicts_a_sent_input_proves_a_late_input() {
        let aim = (100, -50);
        // Core of tick 108, direction 1 and this aim; we sent direction -1 for tick 110: the server had not applied it.
        assert!(stale_core_contradicts_sent(
            &frames(108, 1, aim, false, &[(109, 1, aim), (110, -1, aim)]),
            1,
            0
        ));
        // Only the aim differs.
        assert!(stale_core_contradicts_sent(
            &frames(108, 1, aim, false, &[(110, 1, (100, -80))]),
            1,
            0
        ));
        // Everything sent agrees with the core: nothing to prove.
        assert!(!stale_core_contradicts_sent(
            &frames(108, 1, aim, false, &[(109, 1, aim), (110, 1, aim)]),
            1,
            0
        ));
        // A fresh core is the server's own integer for the frame's tick: not stale.
        assert!(!stale_core_contradicts_sent(
            &frames(110, 1, aim, false, &[(110, -1, aim)]),
            1,
            0
        ));
        // A frozen tee's direction is zeroed by the server: a different sent direction proves nothing...
        assert!(!stale_core_contradicts_sent(
            &frames(108, 1, aim, true, &[(110, -1, aim)]),
            1,
            0
        ));
        // ...but its target is kept and the angle follows it: an aim change contradicts a stale frozen core.
        assert!(stale_core_contradicts_sent(
            &frames(108, 0, aim, true, &[(110, 0, (100, -80))]),
            1,
            0
        ));
        assert!(!stale_core_contradicts_sent(
            &frames(108, 0, aim, true, &[(110, 0, aim)]),
            1,
            0
        ));
        // An input sent for a tick at or before the core's own tick is not about the reckoned ticks.
        assert!(!stale_core_contradicts_sent(
            &frames(108, 1, aim, false, &[(108, -1, aim)]),
            1,
            0
        ));
    }
}
