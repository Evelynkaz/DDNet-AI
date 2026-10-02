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

use crate::format::{Clip, ClipEvent, Frame};

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
    /// Another tee was hook-linked to us (either way), had its hook tip at us, or was within body / hammer reach
    /// ([`BODY_REACH_PX`]): its input between the two frames is a guess.
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
    /// An input of the step was not confirmed by the server's timing report when it was recorded, or the
    /// step is longer than the inputs the frame holds.
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

    /// Every compared step was exact except the ones over a respawn or a teleport.
    pub fn is_exact_but_respawns(&self) -> bool {
        self.steps > 0 && self.exact + self.respawn_steps == self.steps
    }

    /// The divergences nothing the clip knows explains: the server changed our state and no death, teleport,
    /// other tee, projectile, unrecorded tee or unconfirmed input timing is around ([`Cause::ServerCorrection`]).
    pub fn unexplained(&self) -> impl Iterator<Item = &Divergence> {
        self.divergences.iter().filter(|d| d.cause == Cause::ServerCorrection)
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

/// Feeds `frame` to the world the way the bot did.
fn feed(lw: &mut LiveWorld, clip: &Clip, frame: &Frame) {
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
    (a.0 - b.0).hypot(a.1 - b.1)
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
/// An other tee's flying hook whose tip is this close to us may have caught us.
const HOOK_TIP_PX: f64 = 60.0;

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
    // tip at us. Its inputs between the two frames are not in the clip.
    let mut other: Option<(f64, i32)> = None;
    for f in [prev, next] {
        let Some(me) = f.tee(own_id) else { continue };
        for t in &f.tees {
            if t.id == own_id {
                continue;
            }
            let linked = t.ch.hooked_player == own_id || me.ch.hooked_player == t.id;
            let d = dist(t.pos(), me.pos());
            let tip = t.ch.hook_state == ddai_physics::core::HOOK_FLYING
                && dist((f64::from(t.ch.hook_x), f64::from(t.ch.hook_y)), me.pos()) < HOOK_TIP_PX;
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

/// Replays `clip` on `map` in `mode`.
pub fn replay(clip: &Clip, map: Arc<MapData>, mode: Mode) -> Report {
    let own_id = clip.header.own_id;
    let mut report = Report {
        frames: clip.frames.len(),
        ..Report::default()
    };
    if clip.frames.len() < 2 {
        return report;
    }
    let mut lw = LiveWorld::new(Arc::clone(&map), own_id, clip.header.world_seed);
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
            let cause = cause_of(&map, &clip.frames, i, own_id, (f64::from(core.x), f64::from(core.y)));
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
