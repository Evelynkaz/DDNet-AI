//! The human behaviour study (task 3.24, E-039): what players do in the seconds after they freeze an
//! opponent ("after the block"), how a block comes about (the hook from above), how they use the
//! hammer and how long they hold the hook - measured on demos with the players' **real** inputs
//! ([`crate::humaninput`]).
//!
//! Everything is computed per demo from a [`Timeline`] (frames), the real-input table and the
//! detectors of [`crate::analysis`] ([`FreezeEntry`], [`HookEpisode`], [`Hit`], [`Attributed`]); the
//! records are plain data (`serde`), the aggregation into the tables of the report is
//! [`aggregate`]. No names: players are the per-demo anonymous labels.
//!
//! Conventions: positions in px, `y` grows downwards (so "above" is a smaller `y`); `dy` is always
//! `actor.y - victim.y` (negative = the actor is above the victim). Ticks are 50 Hz.

use serde::{Deserialize, Serialize};

use ddai_physics::world::count_input_presses;

use crate::analysis::{Attributed, HOOK_GRABBED, Hit, HookEpisode, Tiles, Timeline, TouchKind};
use crate::humaninput::TrueTable;
use crate::types::{CharRec, char_flags};

/// The "after the block" window: 3 s.
pub const WINDOW_TICKS: i32 = 150;
/// Hammer reach used by the duel post-mortem (`fight_stats.py`).
pub const REACH_PX: f32 = 56.0;
/// The hammer cannot be used again for this many ticks after a swing (`fight_stats.py`: ready when
/// `t - last_swing >= 7`) and for 16 after a hit.
pub const SWING_COOLDOWN: i32 = 7;
pub const HIT_COOLDOWN: i32 = 16;
/// The actor counts as "above" the victim when it is higher by more than this (px).
pub const ABOVE_PX: f32 = 16.0;
/// Hook holds of at most this many ticks are "taps".
pub const TAP_TICKS: i32 = 4;
/// A hook episode is "short" below this many ticks, otherwise "long".
pub const LONG_HOOK_TICKS: i32 = 10;

/// What happened to the victim of a block by the end of the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VictimEnd {
    /// Still frozen when the window ended.
    StillFrozen,
    /// Thawed (and alive) within the window.
    Thawed,
    /// A kill message for the victim during the frozen run.
    Killed,
    /// Left the view while frozen.
    Vanished,
}

/// One credited freeze (the actor's hook or hammer touched the victim within the attribution
/// window before it froze) and the next [`WINDOW_TICKS`] ticks of the actor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AfterBlock {
    pub demo: u16,
    pub tick: i32,
    pub actor: u16,
    pub victim: u16,
    pub hook: bool,
    /// The D-030 block (credited, and the victim stayed frozen >= 1 s or died frozen).
    pub block: bool,
    pub killed: bool,
    /// Characters in the frame of the freeze.
    pub n_chars: u8,
    /// Ticks of the window that could be observed (frames contiguous, both present).
    pub observed: i32,
    pub frozen_ticks: i32,
    pub end: VictimEnd,
    /// `actor.y - victim.y` at the freeze.
    pub dy: f32,
    /// Actor-victim distance at +0 / +50 / +100 / +150 ticks (`NaN` when not observed).
    pub dist: [f32; 4],
    /// Victim displacement from its freeze position at +50 / +100 / +150 ticks (`NaN` unobserved).
    pub victim_dx: [f32; 3],
    pub victim_dy: [f32; 3],
    /// Chebyshev distance (px) from the victim to the nearest freeze tile at +0 and at the end of
    /// the observed window (`NaN` = none within 160 px).
    pub freeze_dist_start: f32,
    pub freeze_dist_end: f32,
    /// Share of the observed frames in which the actor's hook is attached to the victim.
    pub hook_on_victim: f32,
    /// Ticks from the freeze to the first frame with the actor hooked to the victim (`-1` = never).
    pub first_hook: i32,
    /// Real-input hook holds of the actor that are on at some tick of the window: `(start offset, length)` in
    /// ticks (the offset is negative for a hold that began before the freeze).
    pub holds: Vec<(i16, i16)>,
    /// How many of those holds grabbed the victim at some frame.
    pub holds_on_victim: u16,
    /// Hammer swings of the actor (offsets in ticks) and how many were at a frozen victim within reach.
    pub swings: Vec<i16>,
    pub swings_at_frozen: u16,
    /// Hits on the victim (detector of [`crate::analysis::hammer_hits`]).
    pub hits: u16,
    /// Share of the observed ticks with a neutral real input of the actor (no key pressed).
    pub neutral_input: f32,
    /// Share of the observed ticks the actor's real input is known (a track in force).
    pub known_input: f32,
    /// The actor froze itself within the window.
    pub actor_frozen: bool,
    /// Mean actor speed (px/tick) over the observed frames.
    pub actor_speed: f32,
}

/// A hook episode of an actor on a free victim, with what became of it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HookRec {
    pub demo: u16,
    pub tick: i32,
    pub actor: u16,
    pub victim: u16,
    pub n_chars: u8,
    /// Length of the episode in ticks (frames x 2).
    pub ticks: i32,
    /// `actor.y - victim.y` at the start and at the end of the episode.
    pub dy_start: f32,
    pub dy_end: f32,
    /// Victim vertical velocity (px/tick, negative = up) at the end of the episode.
    pub victim_vy_end: f32,
    /// The victim froze with this actor credited within the attribution window after the episode
    /// (from its first tick to `attribution_ticks` after its last), and the ticks from the end of
    /// the episode to the freeze (`-1` = no freeze).
    pub froze: bool,
    pub release_to_freeze: i32,
    /// The freeze was a D-030 block.
    pub block: bool,
    /// Chebyshev distance (px) from the victim to the nearest freeze tile at the end (`NaN` = none within 160 px).
    pub freeze_dist_end: f32,
    /// Real jump presses of the victim / the actor in the 30 ticks before the end.
    pub victim_jumps: u8,
    pub actor_jumps: u8,
}

/// Counters of the fights (two free players near each other), additive over demos.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Fight {
    /// Frames and player-frames (a pair of players counts two).
    pub frames: u64,
    pub player_frames: u64,
    /// Ticks they cover (frames are 2 ticks apart, more where snapshots are missing).
    pub ticks: u64,
    /// Hammer swings (the weapon went off: `FIRED` frames with the hammer in hand), hits (detector of
    /// [`crate::analysis::hammer_hits`]) and presses of the fire key (the real counter, hammer in hand; a press while the
    /// weapon reloads does nothing, so presses >= swings).
    pub swings: u64,
    pub hits: u64,
    pub presses: u64,
    /// Player-frames with the hammer ready and the other player within reach, and those followed by a
    /// swing within 1-4 ticks.
    pub ready_close: u64,
    pub ready_close_swung: u64,
    /// Player-frames with the hook attached to the other player.
    pub hook_on_other: u64,
    /// Real-input hook holds that grabbed the other player: lengths in ticks, in a histogram
    /// `[<=4, <=10, <=25, <=50, <=100, more]`, and the sum / count for the mean.
    pub hold_hist: [u64; 6],
    pub hold_ticks: u64,
    pub holds: u64,
    /// Real jump presses (rising edge of the jump key).
    pub jumps: u64,
    /// Player-frames in which the other player is above by more than [`ABOVE_PX`].
    pub other_above: u64,
    /// Player-frames within 100 px of the other player.
    pub near: u64,
    /// Player-frames of which the real input is known.
    pub known: u64,
    /// All holds lengths (kept for the median), capped in count by the caller.
    pub hold_lengths: Vec<i16>,
}

impl Fight {
    pub fn merge(&mut self, o: &Fight) {
        self.frames += o.frames;
        self.player_frames += o.player_frames;
        self.ticks += o.ticks;
        self.swings += o.swings;
        self.hits += o.hits;
        self.presses += o.presses;
        self.ready_close += o.ready_close;
        self.ready_close_swung += o.ready_close_swung;
        self.hook_on_other += o.hook_on_other;
        for i in 0..6 {
            self.hold_hist[i] += o.hold_hist[i];
        }
        self.hold_ticks += o.hold_ticks;
        self.holds += o.holds;
        self.jumps += o.jumps;
        self.other_above += o.other_above;
        self.near += o.near;
        self.known += o.known;
        self.hold_lengths.extend_from_slice(&o.hold_lengths);
    }
}

/// The study of one demo.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DemoStudy {
    pub after: Vec<AfterBlock>,
    pub hooks: Vec<HookRec>,
    /// Fights between exactly two characters in view / among at most four.
    pub fight2: Fight,
    pub fight4: Fight,
    /// Frames by number of characters (index = count, last = "more").
    pub chars_hist: [u64; 9],
    /// Freeze entries of the demo, those credited to a toucher, and blocks.
    pub freeze_entries: u32,
    pub credited: u32,
    pub blocks: u32,
    /// Frames in which some character's hook is attached to another character.
    pub hook_frames: u64,
}

fn hold_bucket(len: i32) -> usize {
    match len {
        ..=4 => 0,
        5..=10 => 1,
        11..=25 => 2,
        26..=50 => 3,
        51..=100 => 4,
        _ => 5,
    }
}

/// The key state of a real input, or `None` when none is in force.
fn key_at(table: &TrueTable, label: u16, t: i32) -> Option<crate::humaninput::InputEvent> {
    table.track(label)?.at(t - table.tick_shift).copied()
}

/// No key is down: no direction, jump or hook, and the fire counter is even (not held).
fn neutral(e: &crate::humaninput::InputEvent) -> bool {
    e.direction == 0 && !e.jump && !e.hook && e.fire & 1 == 0
}

fn dist(a: &CharRec, b: &CharRec) -> f32 {
    ((a.pos[0] - b.pos[0]).powi(2) + (a.pos[1] - b.pos[1]).powi(2)).sqrt()
}

fn freeze_dist(tiles: &Tiles<'_>, c: &CharRec) -> f32 {
    tiles
        .nearest_freeze(c.pos[0], c.pos[1], 160.0)
        .map_or(f32::NAN, |(dx, dy)| dx.abs().max(dy.abs()))
}

/// Real hook holds of `label` in `[from, to)`: `(start, length)` in ticks, a hold that is still on at
/// `to` is cut there. Unknown input ends a hold.
fn hold_runs(table: &TrueTable, label: u16, from: i32, to: i32) -> Vec<(i32, i32)> {
    let mut out = Vec::new();
    let mut start: Option<i32> = None;
    for t in from..to {
        let on = key_at(table, label, t).is_some_and(|e| e.hook);
        match (on, start) {
            (true, None) => start = Some(t),
            (false, Some(s)) => {
                out.push((s, t - s));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push((s, to - s));
    }
    out
}

/// Ticks in `[from, to)` at which the real jump key goes down.
fn jump_presses(table: &TrueTable, label: u16, from: i32, to: i32) -> Vec<i32> {
    let mut out = Vec::new();
    let mut prev = key_at(table, label, from - 1).is_some_and(|e| e.jump);
    for t in from..to {
        let now = key_at(table, label, t).is_some_and(|e| e.jump);
        if now && !prev {
            out.push(t);
        }
        prev = now;
    }
    out
}

/// Presses of the fire key of `label` in `[from, to)` (the counter passes an odd value) with the
/// hammer in hand; `weapon_at` gives the active weapon around a tick.
fn presses(table: &TrueTable, label: u16, from: i32, to: i32, weapon_at: &dyn Fn(i32) -> Option<i8>) -> Vec<i32> {
    let mut out: Vec<i32> = Vec::new();
    let mut prev = key_at(table, label, from - 1).map(|e| e.fire);
    for t in from..to {
        let now = key_at(table, label, t).map(|e| e.fire);
        if let (Some(p), Some(n)) = (prev, now)
            && count_input_presses(p, n) != 0
            && weapon_at(t) == Some(crate::analysis::WEAPON_HAMMER)
        {
            out.push(t);
        }
        prev = now;
    }
    out
}

/// Hammer swings of `label` in `[from, to)`: the presses of [`presses`], dropping those within
/// [`SWING_COOLDOWN`] ticks of the previous swing (the weapon is not ready).
fn swings(table: &TrueTable, label: u16, from: i32, to: i32, weapon_at: &dyn Fn(i32) -> Option<i8>) -> Vec<i32> {
    let mut out: Vec<i32> = Vec::new();
    let mut prev = key_at(table, label, from - 1).map(|e| e.fire);
    for t in from..to {
        let now = key_at(table, label, t).map(|e| e.fire);
        if let (Some(p), Some(n)) = (prev, now)
            && count_input_presses(p, n) != 0
            && weapon_at(t) == Some(crate::analysis::WEAPON_HAMMER)
            && out.last().is_none_or(|&l| t - l >= SWING_COOLDOWN)
        {
            out.push(t);
        }
        prev = now;
    }
    out
}

/// Runs the whole study over one demo. `demo` is the index stored in the records, `table` the real
/// inputs (tracks keyed by the same labels as the frames).
pub fn study_demo(
    demo: u16,
    tl: &Timeline<'_>,
    table: &TrueTable,
    hooks: &[HookEpisode],
    hits: &[Hit],
    attr: &[Attributed],
) -> DemoStudy {
    let mut out = DemoStudy::default();
    let n = tl.len();
    if n == 0 {
        return out;
    }
    let frames: Vec<crate::store::Item<crate::types::FrameRec>> = (0..n).map(|k| tl.frame(k)).collect();
    out.freeze_entries = attr.len() as u32;
    out.credited = attr.iter().filter(|a| a.toucher.is_some()).count() as u32;
    out.blocks = attr.iter().filter(|a| a.block).count() as u32;
    for f in &frames {
        out.chars_hist[f.chars.len().min(8)] += 1;
        out.hook_frames += u64::from(
            f.chars
                .iter()
                .any(|c| c.hook_state == HOOK_GRABBED && c.hooked_player >= 0),
        );
    }
    let weapon_of = |k: usize, label: u16| frames[k].by_label(label).map(|c| c.weapon);
    // weapon around a tick: the frame at or before it
    let weapon_around = |label: u16| {
        move |t: i32| {
            let k = tl.frame_at_or_before(t)?;
            weapon_of(k, label)
        }
    };

    // ---- after the block ----
    for a in attr {
        let Some((actor, kind)) = a.toucher else { continue };
        let e = a.entry;
        let victim = e.player;
        let t0 = e.tick;
        let (Some(c0), Some(v0)) = (frames[e.k].by_label(actor), frames[e.k].by_label(victim)) else {
            continue;
        };
        let (c0, v0) = (*c0, *v0);
        // observed frames of the window
        let mut observed = 0;
        let mut last_k = e.k;
        let mut hooked_frames = 0u32;
        let mut first_hook = -1;
        let mut speed_sum = 0.0f32;
        let mut frames_seen = 0u32;
        let mut actor_frozen = false;
        let mut dists = [f32::NAN; 4];
        let mut vdx = [f32::NAN; 3];
        let mut vdy = [f32::NAN; 3];
        let mut grabbed_ticks: Vec<i32> = Vec::new();
        let mut k = e.k;
        while k < n {
            let tk = tl.tick(k);
            if tk - t0 > WINDOW_TICKS || (k > e.k && !tl.contiguous(k - 1)) {
                break;
            }
            let (Some(c), Some(v)) = (frames[k].by_label(actor), frames[k].by_label(victim)) else {
                break;
            };
            // a respawn (the victim was killed) teleports it: the story of the block ends there
            if k > e.k && frames[k - 1].by_label(victim).is_some_and(|pv| dist(pv, v) > 200.0) {
                break;
            }
            observed = tk - t0 + tl.cfg.decision_ticks;
            last_k = k;
            frames_seen += 1;
            speed_sum += c.vel[0].hypot(c.vel[1]);
            actor_frozen |= c.frozen() && k > e.k;
            let on = c.hook_state == HOOK_GRABBED && c.hooked_player == i16::from(v.id);
            if on {
                hooked_frames += 1;
                grabbed_ticks.push(tk);
                if first_hook < 0 {
                    first_hook = tk - t0;
                }
            }
            let off = tk - t0;
            for (i, at) in [0, 50, 100, 150].into_iter().enumerate() {
                if off >= at && dists[i].is_nan() {
                    dists[i] = dist(c, v);
                }
            }
            for (i, at) in [50, 100, 150].into_iter().enumerate() {
                if off >= at && vdx[i].is_nan() {
                    vdx[i] = v.pos[0] - v0.pos[0];
                    vdy[i] = v.pos[1] - v0.pos[1];
                }
            }
            k += 1;
        }
        let end_t = t0 + observed.max(1);
        // holds that are on at some tick of the window; one that began before the freeze keeps its
        // true start (a negative offset), one still on at the end of the window is cut there
        let holds: Vec<(i16, i16)> = hold_runs(table, actor, t0 - 150, end_t)
            .into_iter()
            .filter(|&(s, l)| s + l > t0)
            .map(|(s, l)| ((s - t0) as i16, l.min(i32::from(i16::MAX)) as i16))
            .collect();
        let holds_on_victim = holds
            .iter()
            .filter(|&&(s, l)| {
                let (a0, a1) = (t0 + i32::from(s), t0 + i32::from(s) + i32::from(l) + 2);
                grabbed_ticks.iter().any(|&g| g >= a0 && g <= a1)
            })
            .count() as u16;
        let sw = swings(table, actor, t0, end_t, &weapon_around(actor));
        let swings_at_frozen = sw
            .iter()
            .filter(|&&t| {
                tl.frame_at_or_before(t).is_some_and(|k| {
                    matches!(
                        (frames[k].by_label(actor), frames[k].by_label(victim)),
                        (Some(c), Some(v)) if v.frozen() && dist(c, v) <= REACH_PX + 14.0
                    )
                })
            })
            .count() as u16;
        let mut neutral_ticks = 0;
        let mut known_ticks = 0;
        for t in t0..end_t {
            if let Some(ev) = key_at(table, actor, t) {
                known_ticks += 1;
                neutral_ticks += i32::from(neutral(&ev));
            }
        }
        let span = (end_t - t0).max(1) as f32;
        let end = if a.killed {
            VictimEnd::Killed
        } else if e.vanished && e.exit_k.is_some_and(|x| tl.tick(x) <= t0 + WINDOW_TICKS) {
            VictimEnd::Vanished
        } else if e.exit_k.is_some_and(|x| tl.tick(x) <= t0 + WINDOW_TICKS) {
            VictimEnd::Thawed
        } else {
            VictimEnd::StillFrozen
        };
        let hits_n = hits
            .iter()
            .filter(|h| h.actor == actor && h.victim == victim && h.tick >= t0 && h.tick < end_t)
            .count() as u16;
        let v_end = frames[last_k].by_label(victim).copied().unwrap_or(v0);
        out.after.push(AfterBlock {
            demo,
            tick: t0,
            actor,
            victim,
            hook: kind == TouchKind::Hook,
            block: a.block,
            killed: a.killed,
            n_chars: frames[e.k].chars.len().min(255) as u8,
            observed,
            frozen_ticks: e.frozen_ticks,
            end,
            dy: c0.pos[1] - v0.pos[1],
            dist: dists,
            victim_dx: vdx,
            victim_dy: vdy,
            freeze_dist_start: freeze_dist(&tl.tiles, &v0),
            freeze_dist_end: freeze_dist(&tl.tiles, &v_end),
            hook_on_victim: hooked_frames as f32 / frames_seen.max(1) as f32,
            first_hook,
            holds,
            holds_on_victim,
            swings: sw.iter().map(|&t| (t - t0) as i16).collect(),
            swings_at_frozen,
            hits: hits_n,
            neutral_input: neutral_ticks as f32 / span,
            known_input: known_ticks as f32 / span,
            actor_frozen,
            actor_speed: speed_sum / frames_seen.max(1) as f32,
        });
    }

    // ---- hook episodes ----
    for h in hooks {
        let (Some(c0), Some(v0)) = (
            frames[h.start_k].by_label(h.actor),
            frames[h.start_k].by_label(h.victim),
        ) else {
            continue;
        };
        let (Some(c1), Some(v1)) = (frames[h.end_k].by_label(h.actor), frames[h.end_k].by_label(h.victim)) else {
            continue;
        };
        if v0.frozen() {
            continue; // a hook on an already frozen victim is not a way to freeze it
        }
        let t_start = tl.tick(h.start_k);
        let t_end = tl.tick(h.end_k) + tl.cfg.decision_ticks;
        // the victim froze with this actor credited, from the start of the episode to the attribution
        // window after its end
        let freeze = attr.iter().find(|a| {
            a.entry.player == h.victim
                && a.toucher.is_some_and(|(t, k)| t == h.actor && k == TouchKind::Hook)
                && a.entry.tick >= t_start
                && a.entry.tick <= t_end + tl.cfg.attribution_ticks
        });
        out.hooks.push(HookRec {
            demo,
            tick: t_start,
            actor: h.actor,
            victim: h.victim,
            n_chars: frames[h.start_k].chars.len().min(255) as u8,
            ticks: t_end - t_start,
            dy_start: c0.pos[1] - v0.pos[1],
            dy_end: c1.pos[1] - v1.pos[1],
            victim_vy_end: v1.vel[1],
            froze: freeze.is_some(),
            release_to_freeze: freeze.map_or(-1, |a| a.entry.tick - t_end),
            block: freeze.is_some_and(|a| a.block),
            freeze_dist_end: freeze_dist(&tl.tiles, v1),
            victim_jumps: jump_presses(table, h.victim, t_end - 30, t_end).len().min(255) as u8,
            actor_jumps: jump_presses(table, h.actor, t_end - 30, t_end).len().min(255) as u8,
        });
    }

    // ---- fights ----
    let first_tick = tl.tick(0);
    let last_tick = tl.tick(n - 1) + tl.cfg.decision_ticks;
    let mut labels: Vec<u16> = Vec::new();
    for f in &frames {
        for c in &f.chars {
            if !labels.contains(&c.player) {
                labels.push(c.player);
            }
        }
    }
    let mut press_ticks: std::collections::HashMap<u16, Vec<i32>> = std::collections::HashMap::new();
    let mut hold_cache: std::collections::HashMap<u16, Vec<(i32, i32)>> = std::collections::HashMap::new();
    let mut jump_cache: std::collections::HashMap<u16, Vec<i32>> = std::collections::HashMap::new();
    for &l in labels.iter().filter(|&&l| table.track(l).is_some()) {
        press_ticks.insert(l, presses(table, l, first_tick, last_tick, &weapon_around(l)));
        hold_cache.insert(l, hold_runs(table, l, first_tick, last_tick));
        jump_cache.insert(l, jump_presses(table, l, first_tick, last_tick));
    }
    // swings: frames in which the weapon went off (hammer in hand)
    let mut fired: std::collections::HashSet<(u16, usize)> = std::collections::HashSet::new();
    for (k, f) in frames.iter().enumerate() {
        for c in &f.chars {
            if c.has(char_flags::FIRED) && c.weapon == crate::analysis::WEAPON_HAMMER {
                fired.insert((c.player, k));
            }
        }
    }
    // hits by (actor, frame after the hit) and the ticks of each actor's hits (sorted: `hits` is by frame)
    let mut hits_in: std::collections::HashMap<(u16, usize), u64> = std::collections::HashMap::new();
    let mut hit_ticks: std::collections::HashMap<u16, Vec<i32>> = std::collections::HashMap::new();
    for h in hits {
        *hits_in.entry((h.actor, h.k)).or_default() += 1;
        hit_ticks.entry(h.actor).or_default().push(h.tick);
    }
    let last_before = |v: Option<&Vec<i32>>, t: i32| -> i32 {
        v.and_then(|v| {
            let i = v.partition_point(|&x| x < t);
            i.checked_sub(1).map(|i| v[i])
        })
        .unwrap_or(i32::MIN / 2)
    };
    for k in 0..n {
        let tk = tl.tick(k);
        let f = &frames[k];
        let nc = f.chars.len();
        if !(2..=4).contains(&nc) || !tl.contiguous(k) {
            continue;
        }
        let step = tl.tick(k + 1) - tk;
        let in_frame = |t: i32| t >= tk && t < tk + step;
        let mut counted_frame = [false; 2];
        for (ai, a) in f.chars.iter().enumerate() {
            if a.frozen() {
                continue;
            }
            // the nearest free other, within 400 px (the duel regime)
            let other = f
                .chars
                .iter()
                .enumerate()
                .filter(|&(bi, b)| bi != ai && !b.frozen())
                .min_by(|x, y| dist(a, x.1).total_cmp(&dist(a, y.1)));
            let Some((_, b)) = other else { continue };
            let d = dist(a, b);
            if d > 400.0 {
                continue;
            }
            // Only players whose real input is in force: the recording player's own inputs are not in the demo (the server
            // forwards the others' to it), so in a demo it plays in half of the frames have no real input.
            let known = table
                .track(a.player)
                .and_then(|t| t.at(tk - table.tick_shift))
                .is_some();
            if !known {
                continue;
            }
            let prs = press_ticks.get(&a.player);
            let grabbing_other = a.hook_state == HOOK_GRABBED && a.hooked_player == i16::from(b.id);
            for (which, fight) in [&mut out.fight2, &mut out.fight4].into_iter().enumerate() {
                if which == 0 && nc != 2 {
                    continue;
                }
                fight.player_frames += 1;
                fight.ticks += step as u64;
                if !counted_frame[which] {
                    counted_frame[which] = true;
                    fight.frames += 1;
                }
                fight.known += u64::from(known);
                fight.near += u64::from(d <= 100.0);
                fight.other_above += u64::from(b.pos[1] < a.pos[1] - ABOVE_PX);
                fight.hook_on_other += u64::from(grabbing_other);
                fight.hits += hits_in.get(&(a.player, k)).copied().unwrap_or(0);
                fight.swings += u64::from(fired.contains(&(a.player, k)));
                // hammer in hand, within reach, not swung in the last frames or hit in the last 16 ticks, and a
                // swing in the next two frames (1-4 ticks later)
                let ready = a.weapon == crate::analysis::WEAPON_HAMMER
                    && d <= REACH_PX
                    && !(k.saturating_sub(3)..=k).any(|j| fired.contains(&(a.player, j)))
                    && tk - last_before(hit_ticks.get(&a.player), tk + 1) >= HIT_COOLDOWN;
                if ready && k + 2 < n {
                    fight.ready_close += 1;
                    if fired.contains(&(a.player, k + 1)) || fired.contains(&(a.player, k + 2)) {
                        fight.ready_close_swung += 1;
                    }
                }
                if a.weapon == crate::analysis::WEAPON_HAMMER {
                    fight.presses += prs.map_or(0, |v| v.iter().filter(|&&t| in_frame(t)).count() as u64);
                }
                fight.jumps += jump_cache
                    .get(&a.player)
                    .map_or(0, |v| v.iter().filter(|&&t| in_frame(t)).count() as u64);
                // hook holds that start in this frame's interval and grab another player
                if let Some(runs) = hold_cache.get(&a.player) {
                    for &(s, l) in runs.iter().filter(|&&(s, _)| in_frame(s)) {
                        let k0 = k;
                        let k1 = tl.frame_at_or_before(s + l + 2).unwrap_or(k0).max(k0);
                        let grabbed = (k0..=k1).any(|j| {
                            frames[j].by_label(a.player).is_some_and(|c| {
                                c.hook_state == HOOK_GRABBED
                                    && frames[j]
                                        .chars
                                        .iter()
                                        .any(|o| i16::from(o.id) == c.hooked_player && o.player != a.player)
                            })
                        });
                        if grabbed {
                            fight.holds += 1;
                            fight.hold_ticks += l as u64;
                            fight.hold_hist[hold_bucket(l)] += 1;
                            if fight.hold_lengths.len() < 200_000 {
                                fight.hold_lengths.push(l.min(i32::from(i16::MAX)) as i16);
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------------------------

/// Median of a slice, ignoring `NaN` (`NaN` when nothing is left).
pub fn median(v: &[f32]) -> f32 {
    let mut v: Vec<f32> = v.iter().copied().filter(|x| !x.is_nan()).collect();
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// Quantile `q` in `0..=1` (`NaN` when nothing is left).
pub fn quantile(v: &[f32], q: f32) -> f32 {
    let mut v: Vec<f32> = v.iter().copied().filter(|x| !x.is_nan()).collect();
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f32 * q).round() as usize]
}

fn pct(a: usize, b: usize) -> f32 {
    if b == 0 { f32::NAN } else { 100.0 * a as f32 / b as f32 }
}

fn fmt(x: f32, digits: usize) -> String {
    if x.is_nan() {
        "-".into()
    } else {
        format!("{x:.digits$}")
    }
}

/// Summary of a set of [`AfterBlock`] records.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AfterSummary {
    pub n: usize,
    pub blocks: usize,
    pub killed_pct: f32,
    pub thawed_pct: f32,
    pub still_frozen_pct: f32,
    /// Median length (ticks) of the frozen run.
    pub frozen_median: f32,
    /// Share of the records in which the actor hooked the victim at all in the window / median first hook (ticks).
    pub hooked_pct: f32,
    pub first_hook_median: f32,
    /// Mean share of frames with the hook on the victim, median over records.
    pub hook_share_median: f32,
    pub hold_median: f32,
    pub hold_tap_pct: f32,
    pub holds_per_block: f32,
    /// Hammer: share of records with at least one swing, mean swings per record, share with a swing at
    /// the frozen victim, mean hits per record.
    pub swung_pct: f32,
    pub swings_mean: f32,
    pub swung_at_frozen_pct: f32,
    pub hits_mean: f32,
    /// Share of records in which the actor's key state is neutral for more than half of the window.
    pub idle_pct: f32,
    pub neutral_mean: f32,
    pub actor_frozen_pct: f32,
    pub dist_median: [f32; 4],
    /// Victim displacement at +150 ticks: median |dx|, median dy (negative = up).
    pub victim_dx_abs_median: f32,
    pub victim_dy_median: f32,
    /// Median distance to the nearest freeze tile at the start / end of the window.
    pub freeze_dist_start_median: f32,
    pub freeze_dist_end_median: f32,
    /// Share of records in which the victim ended nearer to a freeze tile than it began.
    pub nearer_freeze_pct: f32,
    /// Share of records in which the actor was above the victim at the freeze.
    pub above_pct: f32,
}

/// Summarises `recs`.
pub fn summarize_after(recs: &[&AfterBlock]) -> AfterSummary {
    let n = recs.len();
    let mut s = AfterSummary {
        n,
        blocks: recs.iter().filter(|r| r.block).count(),
        ..Default::default()
    };
    if n == 0 {
        return s;
    }
    let count = |f: &dyn Fn(&AfterBlock) -> bool| recs.iter().filter(|r| f(r)).count();
    s.killed_pct = pct(count(&|r| r.end == VictimEnd::Killed), n);
    s.thawed_pct = pct(count(&|r| r.end == VictimEnd::Thawed), n);
    s.still_frozen_pct = pct(count(&|r| r.end == VictimEnd::StillFrozen), n);
    s.frozen_median = median(&recs.iter().map(|r| r.frozen_ticks as f32).collect::<Vec<_>>());
    s.hooked_pct = pct(count(&|r| r.first_hook >= 0), n);
    s.first_hook_median = median(
        &recs
            .iter()
            .filter(|r| r.first_hook >= 0)
            .map(|r| r.first_hook as f32)
            .collect::<Vec<_>>(),
    );
    s.hook_share_median = median(&recs.iter().map(|r| r.hook_on_victim).collect::<Vec<_>>());
    let holds: Vec<f32> = recs
        .iter()
        .flat_map(|r| r.holds.iter().filter(|_| r.known_input > 0.5).map(|&(_, l)| l as f32))
        .collect();
    s.hold_median = median(&holds);
    s.hold_tap_pct = pct(holds.iter().filter(|&&l| l <= TAP_TICKS as f32).count(), holds.len());
    let known: Vec<&&AfterBlock> = recs.iter().filter(|r| r.known_input > 0.5).collect();
    s.holds_per_block = known.iter().map(|r| r.holds.len() as f32).sum::<f32>() / known.len().max(1) as f32;
    s.swung_pct = pct(known.iter().filter(|r| !r.swings.is_empty()).count(), known.len());
    s.swings_mean = known.iter().map(|r| r.swings.len() as f32).sum::<f32>() / known.len().max(1) as f32;
    s.swung_at_frozen_pct = pct(known.iter().filter(|r| r.swings_at_frozen > 0).count(), known.len());
    s.hits_mean = recs.iter().map(|r| f32::from(r.hits)).sum::<f32>() / n as f32;
    s.idle_pct = pct(known.iter().filter(|r| r.neutral_input > 0.5).count(), known.len());
    s.neutral_mean = known.iter().map(|r| r.neutral_input).sum::<f32>() / known.len().max(1) as f32;
    s.actor_frozen_pct = pct(count(&|r| r.actor_frozen), n);
    for i in 0..4 {
        s.dist_median[i] = median(&recs.iter().map(|r| r.dist[i]).collect::<Vec<_>>());
    }
    s.victim_dx_abs_median = median(&recs.iter().map(|r| r.victim_dx[2].abs()).collect::<Vec<_>>());
    s.victim_dy_median = median(&recs.iter().map(|r| r.victim_dy[2]).collect::<Vec<_>>());
    s.freeze_dist_start_median = median(&recs.iter().map(|r| r.freeze_dist_start).collect::<Vec<_>>());
    s.freeze_dist_end_median = median(&recs.iter().map(|r| r.freeze_dist_end).collect::<Vec<_>>());
    let both: Vec<&&AfterBlock> = recs
        .iter()
        .filter(|r| !r.freeze_dist_start.is_nan() && !r.freeze_dist_end.is_nan())
        .collect();
    s.nearer_freeze_pct = pct(
        both.iter().filter(|r| r.freeze_dist_end < r.freeze_dist_start).count(),
        both.len(),
    );
    s.above_pct = pct(count(&|r| r.dy < -ABOVE_PX), n);
    s
}

/// Rows of the "after the block" table for markdown.
pub fn after_rows(label: &str, s: &AfterSummary) -> String {
    let cells = [
        s.n.to_string(),
        s.blocks.to_string(),
        format!(
            "{} / {} / {}",
            fmt(s.killed_pct, 0),
            fmt(s.thawed_pct, 0),
            fmt(s.still_frozen_pct, 0)
        ),
        fmt(s.frozen_median, 0),
        format!("{} / {}", fmt(s.hooked_pct, 0), fmt(s.first_hook_median, 0)),
        fmt(s.hook_share_median * 100.0, 0),
        format!(
            "{} / {} / {}",
            fmt(s.hold_median, 0),
            fmt(s.hold_tap_pct, 0),
            fmt(s.holds_per_block, 1)
        ),
        format!(
            "{} / {} / {} / {}",
            fmt(s.swung_pct, 0),
            fmt(s.swings_mean, 1),
            fmt(s.swung_at_frozen_pct, 0),
            fmt(s.hits_mean, 2)
        ),
        format!("{} / {}", fmt(s.idle_pct, 0), fmt(s.neutral_mean * 100.0, 0)),
        format!("{} / {}", fmt(s.victim_dx_abs_median, 0), fmt(s.victim_dy_median, 0)),
        fmt(s.nearer_freeze_pct, 0),
        format!(
            "{} / {}",
            fmt(s.freeze_dist_start_median, 0),
            fmt(s.freeze_dist_end_median, 0)
        ),
    ];
    format!("| {label} | {} |", cells.join(" | "))
}

/// Hook episodes by the actor's height over the victim at the start and by length: how many froze
/// the victim (within the attribution window).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HookCell {
    pub n: usize,
    pub froze: usize,
    pub blocks: usize,
    pub release_to_freeze_median: f32,
}

/// `[above, level, below] x [short, long]` cells of [`HookCell`].
pub fn hook_cells(recs: &[&HookRec]) -> [[HookCell; 2]; 3] {
    let mut out: [[HookCell; 2]; 3] = Default::default();
    let mut rel: [[Vec<f32>; 2]; 3] = Default::default();
    for r in recs {
        let h = if r.dy_start < -ABOVE_PX {
            0
        } else if r.dy_start > ABOVE_PX {
            2
        } else {
            1
        };
        let l = usize::from(r.ticks >= LONG_HOOK_TICKS);
        let c = &mut out[h][l];
        c.n += 1;
        c.froze += usize::from(r.froze);
        c.blocks += usize::from(r.block);
        if r.froze {
            rel[h][l].push(r.release_to_freeze as f32);
        }
    }
    for h in 0..3 {
        for l in 0..2 {
            out[h][l].release_to_freeze_median = median(&rel[h][l]);
        }
    }
    out
}

/// Markdown for the fight counters: rates per 30 s of fight per player.
pub fn fight_row(label: &str, f: &Fight) -> String {
    let secs = f.ticks as f64 / 50.0;
    let per30 = |x: u64| if secs > 0.0 { x as f64 * 30.0 / secs } else { f64::NAN };
    let holds: Vec<f32> = f.hold_lengths.iter().map(|&l| l as f32).collect();
    format!(
        "| {label} | {:.0} | {:.1} | {:.1} | {:.1} | {:.1} | {} ({}/{}) | {:.0} | {} / {} / {} / {} | {:.0} | {:.1} | {:.0} |",
        secs / 30.0,
        per30(f.hits),
        per30(f.swings),
        per30(f.presses),
        per30(f.jumps),
        fmt(pct(f.ready_close_swung as usize, f.ready_close as usize), 0),
        f.ready_close_swung,
        f.ready_close,
        100.0 * f.hook_on_other as f64 / f.player_frames.max(1) as f64,
        fmt(quantile(&holds, 0.25), 0),
        fmt(median(&holds), 0),
        fmt(quantile(&holds, 0.75), 0),
        fmt(quantile(&holds, 0.9), 0),
        pct(f.hold_hist[0] as usize, f.holds as usize),
        per30(f.holds),
        100.0 * f.other_above as f64 / f.player_frames.max(1) as f64,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::fixtures::*;
    use crate::analysis::{attribute, freeze_entries, hammer_hits, hook_episodes};
    use crate::config::Config;
    use crate::humaninput::{InputEvent, Track};
    use crate::ingest::KillEvent;
    use crate::testutil::arena_with_pit;
    use crate::types::char_flags;

    fn ev(intended: i32, hook: bool, fire: i32, jump: bool) -> InputEvent {
        InputEvent {
            intended,
            arrived: intended - 1,
            direction: 0,
            jump,
            hook,
            fire,
            aim: [1, 0],
        }
    }

    /// Player 1 (actor, label 1) hooks player 2 (victim, label 2) from above; the victim freezes at
    /// tick 20; the actor keeps the hook on for 10 more ticks, then lets go, waits, and swings once.
    #[test]
    fn after_the_block_and_the_hook_from_above_are_measured_from_the_real_inputs() {
        let map = arena_with_pit(20, 10, (8, 12));
        let cfg = Config::default();
        let kills: Vec<KillEvent> = Vec::new();
        let mut frames = Vec::new();
        for i in 0..90 {
            let t = 10 + 2 * i;
            let hooking_phase = (4..13).contains(&i); // ticks 18..34
            let actor = ch(1, 1, 100.0, 100.0);
            let mut actor = if hooking_phase { hooking(actor, 2) } else { actor };
            if t >= 100 {
                actor.flags |= char_flags::FIRED;
            }
            let victim = ch(2, 2, 140.0, 130.0);
            let victim = if (5..60).contains(&i) { frozen(victim) } else { victim };
            frames.push(frame(t, vec![actor, victim]));
        }
        let store = leak_store(&frames);
        let tl = Timeline::new(&cfg, store, &kills, &map);
        let entries = freeze_entries(&tl);
        let hooks = hook_episodes(&tl);
        let hits = hammer_hits(&tl);
        let attr = attribute(&tl, &entries, &hooks, &hits);
        assert_eq!(attr.len(), 1);
        assert_eq!(attr[0].toucher.map(|(a, _)| a), Some(1));
        // the actor's real input: hook from tick 18 to 33, a press of the fire key at tick 60
        let mut table = TrueTable::default();
        table.tracks.insert(
            1,
            Track {
                events: vec![
                    ev(10, false, 0, false),
                    ev(18, true, 0, false),
                    ev(34, false, 0, false),
                    ev(60, false, 1, false),
                ],
                restarts: vec![],
                ..Track::default()
            },
        );
        let st = study_demo(0, &tl, &table, &hooks, &hits, &attr);
        assert_eq!(st.after.len(), 1);
        let a = &st.after[0];
        assert!(a.hook && a.actor == 1 && a.victim == 2);
        assert!(a.dy < 0.0, "the actor is above: dy = {}", a.dy);
        assert_eq!(a.observed, 152);
        assert!(a.first_hook >= 0);
        assert!(a.hook_on_victim > 0.0 && a.hook_on_victim < 0.3);
        assert_eq!(a.holds.len(), 1, "one real hold: {:?}", a.holds);
        assert_eq!(
            a.holds[0],
            (-2, 16),
            "began 2 ticks before the freeze and keeps its true length"
        );
        assert_eq!(a.holds_on_victim, 1);
        assert!((a.known_input - 1.0).abs() < 0.05, "known {}", a.known_input);
        assert_eq!(a.swings.len(), 1, "one swing, at tick 60: {:?}", a.swings);
        assert_eq!(a.swings[0], 60 - a.tick as i16);
        assert_eq!(st.hooks.len(), 1);
        let h = &st.hooks[0];
        assert!(h.dy_start < -ABOVE_PX, "hooked from above");
        assert!(h.froze, "the hook froze the victim");
        let cells = hook_cells(&[h]);
        assert_eq!(cells[0][1].n, 1);
        assert_eq!(cells[0][1].froze, 1);
        assert_eq!(
            cells[1][0].n + cells[1][1].n + cells[2][0].n + cells[2][1].n + cells[0][0].n,
            0
        );
        // summaries do not choke
        let s = summarize_after(&[a]);
        assert_eq!(s.n, 1);
        assert!(after_rows("x", &s).starts_with("| x | 1 |"));
        assert_eq!(st.chars_hist[2], 90);
        assert!(st.fight2.player_frames > 0);
    }

    #[test]
    fn median_and_quantile_ignore_nan() {
        assert_eq!(median(&[3.0, f32::NAN, 1.0, 2.0]), 2.0);
        assert!(median(&[]).is_nan());
        assert_eq!(quantile(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.0), 1.0);
        assert_eq!(quantile(&[1.0, 2.0, 3.0, 4.0, 5.0], 1.0), 5.0);
        assert_eq!(hold_bucket(4), 0);
        assert_eq!(hold_bucket(5), 1);
        assert_eq!(hold_bucket(101), 5);
    }

    #[test]
    fn hold_runs_and_jump_presses_come_from_the_real_input_only() {
        let mut table = TrueTable::default();
        table.tracks.insert(
            3,
            Track {
                events: vec![
                    ev(5, true, 0, false),
                    ev(9, false, 0, true),
                    ev(12, true, 0, false),
                    ev(14, false, 0, true),
                ],
                restarts: vec![],
                ..Track::default()
            },
        );
        assert_eq!(hold_runs(&table, 3, 0, 20), vec![(5, 4), (12, 2)]);
        assert_eq!(
            hold_runs(&table, 3, 0, 7),
            vec![(5, 2)],
            "a hold still on at the end is cut there"
        );
        assert_eq!(jump_presses(&table, 3, 0, 20), vec![9, 14]);
        assert!(hold_runs(&table, 4, 0, 20).is_empty(), "no track, no hold");
    }
}
