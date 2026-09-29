//! Technique detectors (D-048, catalogue T1-T18 of `docs/research/block-knowledge.md` §2.1) over
//! a reconstructed timeline. Each detector emits [`TechniqueEvent`]s: an *attempt* with its
//! outcome (`success`). The outcome definitions follow the catalogue's own checks - "did the
//! attack freeze the target" / "did the escape avoid freeze" - measured on the snapshots.
//!
//! Detectors see only what a client demo shows: positions, velocities, hook and jump state, fire
//! events, freeze state. They are heuristics with documented thresholds ([`TechniqueConfig`]);
//! `docs/formats.md` §20 lists each rule. `T6`, `T7`, `T16`, `T17`, `T18` have no detector (see
//! [`Technique::not_detected_reason`]).

use serde::{Deserialize, Serialize};

use crate::analysis::{Attributed, FreezeEntry, HOOK_FLYING, HOOK_GRABBED, Hit, HookEpisode, Timeline};
use crate::config::TechniqueConfig;
use crate::tags::Technique;
use crate::types::{CharRec, char_flags};

/// One detected use (or, for the defensive techniques, exposure) of a technique.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TechniqueEvent {
    pub technique: Technique,
    /// The player who performs the technique (the defender for T9/T10/T12/T14/WH).
    pub actor: u16,
    /// The other party (victim of an attack, hooker/hitter of a defence), if any.
    pub other: Option<u16>,
    /// First decision tick belonging to the event (the snapshot *before* the visible change).
    pub start_tick: i32,
    pub end_tick: i32,
    /// Attack: the target entered freeze credited to the actor. Defence: the actor stayed free.
    pub success: bool,
    /// The technique was actually applied. `false` marks an at-risk baseline (e.g. hooked near
    /// freeze without any counter-measure): it is reported but its samples are not tagged.
    pub active: bool,
    /// Technique-specific variant bits, see each detector.
    pub detail: u8,
}

/// Everything the detectors share.
pub struct Ctx<'a, 'b> {
    pub tl: &'b Timeline<'a>,
    pub entries: &'b [FreezeEntry],
    pub hooks: &'b [HookEpisode],
    pub hits: &'b [Hit],
    pub attr: &'b [Attributed],
    pub tc: &'b TechniqueConfig,
}

impl Ctx<'_, '_> {
    fn dec(&self) -> i32 {
        self.tl.cfg.decision_ticks
    }

    /// The victim entered freeze with `actor` as the credited toucher, within `(from, from + window]`.
    fn credited(&self, victim: u16, actor: u16, from: i32, window: i32) -> bool {
        self.attr.iter().any(|a| {
            a.entry.player == victim
                && a.toucher.is_some_and(|(t, _)| t == actor)
                && a.entry.tick > from
                && a.entry.tick <= from + window
        })
    }

    /// `label` is frozen at some snapshot in `(from, from + window]`.
    fn frozen_in(&self, label: u16, from: i32, window: i32) -> bool {
        self.tl.frozen_within(label, from, window)
    }

    fn dist(a: &CharRec, b: &CharRec) -> f32 {
        ((a.pos[0] - b.pos[0]).powi(2) + (a.pos[1] - b.pos[1]).powi(2)).sqrt()
    }
}

fn unit_towards(from: &CharRec, to: &CharRec) -> (f32, f32) {
    let (dx, dy) = (to.pos[0] - from.pos[0], to.pos[1] - from.pos[1]);
    let n = (dx * dx + dy * dy).sqrt();
    if n < 1e-3 { (0.0, 0.0) } else { (dx / n, dy / n) }
}

fn air_jump_used(c: &CharRec) -> bool {
    c.has(char_flags::AIR_JUMP_USED)
}

/// Runs every detector.
pub fn detect(ctx: &Ctx<'_, '_>) -> Vec<TechniqueEvent> {
    let mut out = Vec::new();
    out.extend(t1_t4_hook_attacks(ctx));
    out.extend(t2_hammer_into_wall(ctx));
    out.extend(t3_swing_up(ctx));
    out.extend(t5_body_push(ctx));
    out.extend(t8_hammer_frozen(ctx));
    out.extend(t9_escape_hook(ctx));
    out.extend(t10_save_thrown_up(ctx));
    out.extend(t11_edge_stance(ctx));
    out.extend(t12_second_jump_save(ctx));
    out.extend(t13_regain_jump(ctx));
    out.extend(t14_panic_hook(ctx));
    out.extend(t15_hook_duel(ctx));
    out.extend(wall_hook(ctx));
    out.sort_by_key(|e| (e.start_tick, e.technique, e.actor, e.other));
    out
}

/// T1 (hook drag through the edge into freeze) and T4 (pull down a jumper), both hook episodes
/// of an attacker who stands on the ground:
/// - T1: the victim is free and a freeze tile is within `freeze_near_px` of it, and it is dragged
///   at least 16 px towards the attacker;
/// - T4: the victim is airborne and falling faster than `pull_victim_vy` with freeze below it.
///
/// Success: the victim entered freeze credited to the attacker (D-030) before the hook ended plus
/// the attribution window. `detail & 1`: the attacker froze himself within two seconds.
fn t1_t4_hook_attacks(ctx: &Ctx<'_, '_>) -> Vec<TechniqueEvent> {
    let (tl, tc) = (ctx.tl, ctx.tc);
    let mut out = Vec::new();
    for ep in ctx.hooks {
        let dur = (ep.end_k - ep.start_k + 1) as i32 * ctx.dec();
        if dur < tc.min_hook_ticks {
            continue;
        }
        let (Some(a0), Some(v0)) = (tl.ch(ep.start_k, ep.actor), tl.ch(ep.start_k, ep.victim)) else {
            continue;
        };
        if a0.frozen() || v0.frozen() || !a0.grounded() {
            continue;
        }
        let start_tick = tl.tick(ep.start_k) - ctx.dec();
        let end_tick = tl.tick(ep.end_k);
        let window = end_tick - start_tick + tl.cfg.attribution_ticks;
        let success = ctx.credited(ep.victim, ep.actor, start_tick, window);
        let self_froze = success && ctx.frozen_in(ep.actor, start_tick, window + 100);
        let detail = u8::from(self_froze);

        let at_stake = tl
            .tiles
            .nearest_freeze(v0.pos[0], v0.pos[1], tc.freeze_near_px)
            .is_some();
        let v_end = tl.ch(ep.end_k, ep.victim).unwrap_or(v0);
        let (ux, uy) = unit_towards(v0, a0);
        let pulled = (v_end.pos[0] - v0.pos[0]) * ux + (v_end.pos[1] - v0.pos[1]) * uy;
        if at_stake && pulled >= 16.0 {
            out.push(TechniqueEvent {
                technique: Technique::T1,
                actor: ep.actor,
                other: Some(ep.victim),
                start_tick,
                end_tick,
                success,
                active: true,
                detail,
            });
        }
        if !v0.grounded()
            && v0.vel[1] >= tc.pull_victim_vy
            && tl.tiles.freeze_below(v0.pos[0], v0.pos[1], 192.0).is_some()
        {
            out.push(TechniqueEvent {
                technique: Technique::T4,
                actor: ep.actor,
                other: Some(ep.victim),
                start_tick,
                end_tick,
                success,
                active: true,
                detail,
            });
        }
    }
    out
}

/// T2: a hammer hit that throws a free target sideways (|vx| >= 3 px/tick afterwards) towards a
/// freeze wall or pit within `freeze_near_px` on that side. Success: credited freeze within the
/// attribution window.
fn t2_hammer_into_wall(ctx: &Ctx<'_, '_>) -> Vec<TechniqueEvent> {
    let (tl, tc) = (ctx.tl, ctx.tc);
    let mut out = Vec::new();
    for h in ctx.hits {
        if h.victim_frozen_before {
            continue;
        }
        let (Some(v0), Some(v1)) = (tl.ch(h.k - 1, h.victim), tl.ch(h.k, h.victim)) else {
            continue;
        };
        if v1.vel[0].abs() < 3.0 {
            continue;
        }
        let dir = if v1.vel[0] > 0.0 { 1 } else { -1 };
        if tl
            .tiles
            .freeze_side(v0.pos[0], v0.pos[1], tc.freeze_near_px, dir)
            .is_none()
        {
            continue;
        }
        let start = tl.tick(h.k - 1);
        out.push(TechniqueEvent {
            technique: Technique::T2,
            actor: h.actor,
            other: Some(h.victim),
            start_tick: start,
            end_tick: h.tick,
            success: ctx.credited(h.victim, h.actor, start, tl.cfg.attribution_ticks + ctx.dec()),
            active: true,
            detail: 0,
        });
    }
    out
}

/// T3 (swing up): the attacker hooks a victim from above (>= `swing_min_above_px`) while airborne
/// and finishes with a hammer hit within 24 ticks after the hook, while the victim moves sideways
/// with |vx| >= `swing_victim_vx`; freeze must be near the victim.
fn t3_swing_up(ctx: &Ctx<'_, '_>) -> Vec<TechniqueEvent> {
    let (tl, tc) = (ctx.tl, ctx.tc);
    let mut out = Vec::new();
    for ep in ctx.hooks {
        let (Some(a0), Some(v0)) = (tl.ch(ep.start_k, ep.actor), tl.ch(ep.start_k, ep.victim)) else {
            continue;
        };
        if a0.frozen() || v0.frozen() || a0.grounded() || a0.pos[1] > v0.pos[1] - tc.swing_min_above_px {
            continue;
        }
        let end_tick = tl.tick(ep.end_k);
        let Some(hit) = ctx.hits.iter().find(|h| {
            h.actor == ep.actor && h.victim == ep.victim && h.tick >= tl.tick(ep.start_k) && h.tick <= end_tick + 24
        }) else {
            continue;
        };
        let Some(v_before) = tl.ch(hit.k - 1, hit.victim) else {
            continue;
        };
        if v_before.vel[0].abs() < tc.swing_victim_vx
            || tl
                .tiles
                .nearest_freeze(v_before.pos[0], v_before.pos[1], tc.freeze_near_px)
                .is_none()
        {
            continue;
        }
        let start_tick = tl.tick(ep.start_k) - ctx.dec();
        out.push(TechniqueEvent {
            technique: Technique::T3,
            actor: ep.actor,
            other: Some(ep.victim),
            start_tick,
            end_tick: hit.tick,
            success: ctx.credited(
                hit.victim,
                ep.actor,
                hit.tick - ctx.dec(),
                tl.cfg.attribution_ticks + ctx.dec(),
            ),
            active: true,
            detail: 0,
        });
    }
    out
}

/// T5 (body push at the edge): two characters closer than `push_dist_px`, the pusher closing in at
/// at least 2 px/tick, the target free, grounded, with freeze within `push_edge_px`, and neither
/// hook nor hammer involved. Success: the target entered freeze within 25 ticks. Consecutive
/// contact snapshots form one event.
fn t5_body_push(ctx: &Ctx<'_, '_>) -> Vec<TechniqueEvent> {
    let (tl, tc) = (ctx.tl, ctx.tc);
    let mut out: Vec<TechniqueEvent> = Vec::new();
    for k in 1..tl.frames.len() {
        if !tl.contiguous(k - 1) {
            continue;
        }
        for a in &tl.frames[k].chars {
            for v in &tl.frames[k].chars {
                if a.player == v.player {
                    continue;
                }
                let (Some(a0), Some(v0)) = (tl.ch(k - 1, a.player), tl.ch(k - 1, v.player)) else {
                    continue;
                };
                if a0.frozen() || v0.frozen() || !v0.grounded() {
                    continue;
                }
                if Ctx::dist(a0, v0) >= tc.push_dist_px {
                    continue;
                }
                let (ux, uy) = unit_towards(a0, v0);
                let closing = (a0.vel[0] - v0.vel[0]) * ux + (a0.vel[1] - v0.vel[1]) * uy;
                if closing < 2.0 {
                    continue;
                }
                if a.has(char_flags::FIRED) || a0.hook_state != 0 || v0.hook_state != 0 {
                    continue;
                }
                if ctx
                    .hooks
                    .iter()
                    .any(|h| h.victim == v.player && h.start_k <= k && k <= h.end_k + 1)
                {
                    continue;
                }
                if tl.tiles.nearest_freeze(v0.pos[0], v0.pos[1], tc.push_edge_px).is_none() {
                    continue;
                }
                let start = tl.tick(k - 1);
                let success = ctx
                    .entries
                    .iter()
                    .any(|e| e.player == v.player && e.tick > start && e.tick <= start + 25);
                // Merge with the previous event for the same pair when adjacent.
                if let Some(last) = out
                    .iter_mut()
                    .rev()
                    .find(|e| e.actor == a.player && e.other == Some(v.player))
                    && last.end_tick >= start
                {
                    last.end_tick = tl.tick(k);
                    last.success |= success;
                    continue;
                }
                out.push(TechniqueEvent {
                    technique: Technique::T5,
                    actor: a.player,
                    other: Some(v.player),
                    start_tick: start,
                    end_tick: tl.tick(k),
                    success,
                    active: true,
                    detail: 0,
                });
            }
        }
    }
    out
}

/// T8: a hammer hit on a frozen enemy. The catalogue treats it as a mistake unless it sends the
/// target on into a freeze ceiling; success here = the target is frozen again (or still) within
/// 25 ticks, i.e. the hit did not rescue it.
fn t8_hammer_frozen(ctx: &Ctx<'_, '_>) -> Vec<TechniqueEvent> {
    let tl = ctx.tl;
    ctx.hits
        .iter()
        .filter(|h| h.victim_frozen_before)
        .map(|h| {
            let refrozen = tl.ch(h.k, h.victim).is_some_and(|c| c.frozen())
                || ctx
                    .entries
                    .iter()
                    .any(|e| e.player == h.victim && e.tick > h.tick && e.tick <= h.tick + 25);
            TechniqueEvent {
                technique: Technique::T8,
                actor: h.actor,
                other: Some(h.victim),
                start_tick: tl.tick(h.k - 1),
                end_tick: h.tick,
                success: refrozen,
                active: true,
                detail: 0,
            }
        })
        .collect()
}

/// T9 (escape from being hooked). Exposure: a free character hooked for >= `min_hook_ticks` with
/// freeze within `freeze_near_px`. Counter-measure bits over the hook: 1 = pushes against the pull
/// ("stay strong"), 2 = hooks terrain, 4 = uses the air jump, 8 = hooks the hooker back. No bit =
/// passive baseline (`active == false`). Success: not frozen until 25 ticks after the hook ended.
fn t9_escape_hook(ctx: &Ctx<'_, '_>) -> Vec<TechniqueEvent> {
    let (tl, tc) = (ctx.tl, ctx.tc);
    let mut out = Vec::new();
    for ep in ctx.hooks {
        let dur = (ep.end_k - ep.start_k + 1) as i32 * ctx.dec();
        if dur < tc.min_hook_ticks {
            continue;
        }
        let Some(x0) = tl.ch(ep.start_k, ep.victim) else {
            continue;
        };
        if x0.frozen()
            || tl
                .tiles
                .nearest_freeze(x0.pos[0], x0.pos[1], tc.freeze_near_px)
                .is_none()
        {
            continue;
        }
        let mut detail = 0u8;
        for k in ep.start_k..=ep.end_k {
            let (Some(a), Some(x)) = (tl.ch(k, ep.actor), tl.ch(k, ep.victim)) else {
                continue;
            };
            let toward = a.pos[0] - x.pos[0];
            if x.direction != 0 && toward.abs() >= 16.0 && (f32::from(x.direction) * toward) < 0.0 {
                detail |= 1;
            }
            if x.hook_state == HOOK_GRABBED && x.hooked_player < 0 {
                detail |= 2;
            }
            if k > ep.start_k
                && let Some(xp) = tl.ch(k - 1, ep.victim)
                && ((air_jump_used(x) && !air_jump_used(xp)) || x.vel[1] - xp.vel[1] <= -5.0)
            {
                detail |= 4;
            }
            if i32::from(x.hooked_player) == i32::from(a.id) && x.hook_state == HOOK_GRABBED {
                detail |= 8;
            }
        }
        let start_tick = tl.tick(ep.start_k) - ctx.dec();
        let end_tick = tl.tick(ep.end_k);
        out.push(TechniqueEvent {
            technique: Technique::T9,
            actor: ep.victim,
            other: Some(ep.actor),
            start_tick,
            end_tick,
            success: !ctx.frozen_in(ep.victim, start_tick, end_tick - start_tick + 25),
            active: detail != 0,
            detail,
        });
    }
    out
}

/// T10 (save when thrown up): a hammer hit sends a free character upwards (vy <= `thrown_up_vy`)
/// under a freeze ceiling within `thrown_ceiling_px`. Counter-measure within 25 ticks: 1 = hooks
/// the hitter below, 2 = hooks terrain, 4 = uses the air jump. Success: not frozen for 50 ticks.
fn t10_save_thrown_up(ctx: &Ctx<'_, '_>) -> Vec<TechniqueEvent> {
    let (tl, tc) = (ctx.tl, ctx.tc);
    let mut out = Vec::new();
    for h in ctx.hits {
        if h.victim_frozen_before {
            continue;
        }
        let Some(x1) = tl.ch(h.k, h.victim) else { continue };
        if x1.vel[1] > tc.thrown_up_vy
            || tl
                .tiles
                .freeze_above(x1.pos[0], x1.pos[1], tc.thrown_ceiling_px)
                .is_none()
        {
            continue;
        }
        let mut detail = 0u8;
        let mut last_k = h.k;
        for k in h.k..tl.frames.len() {
            if tl.tick(k) > h.tick + 25 {
                break;
            }
            last_k = k;
            let Some(x) = tl.ch(k, h.victim) else { break };
            if x.hooked_player >= 0
                && x.hook_state == HOOK_GRABBED
                && let Some(a) = tl.ch(k, h.actor)
                && i32::from(x.hooked_player) == i32::from(a.id)
                && a.pos[1] > x.pos[1]
            {
                detail |= 1;
            }
            if x.hook_state == HOOK_GRABBED && x.hooked_player < 0 {
                detail |= 2;
            }
            if k > h.k
                && let Some(xp) = tl.ch(k - 1, h.victim)
                && air_jump_used(x)
                && !air_jump_used(xp)
            {
                detail |= 4;
            }
        }
        out.push(TechniqueEvent {
            technique: Technique::T10,
            actor: h.victim,
            other: Some(h.actor),
            start_tick: tl.tick(h.k - 1),
            end_tick: tl.tick(last_k),
            success: !ctx.frozen_in(h.victim, tl.tick(h.k - 1), 50),
            active: detail != 0,
            detail,
        });
    }
    out
}

/// T11 (edge stance): a grounded, motionless, free character stands >= `edge_stand_ticks` with its
/// centre within `edge_dist_px` of a freeze tile on the floor row beside it while an enemy is in
/// hook range. Success: not frozen during the stance and 25 ticks after.
fn t11_edge_stance(ctx: &Ctx<'_, '_>) -> Vec<TechniqueEvent> {
    let (tl, tc) = (ctx.tl, ctx.tc);
    let mut out = Vec::new();
    // Per label: run start frame.
    let mut runs: std::collections::BTreeMap<u16, usize> = std::collections::BTreeMap::new();
    let on_edge = |c: &CharRec, others: &[CharRec]| -> bool {
        if !c.grounded() || c.frozen() || c.vel[0].abs() > 0.6 || c.vel[1].abs() > 0.6 {
            return false;
        }
        let (cx, _) = tl.tiles.cell(c.pos[0], c.pos[1]);
        let feet_row = tl.tiles.cell(c.pos[0], c.pos[1] + 30.0).1;
        let near_edge = [
            (-1, c.pos[0] - cx as f32 * 32.0),
            (1, (cx + 1) as f32 * 32.0 - c.pos[0]),
        ]
        .iter()
        .any(|&(dir, gap)| tl.tiles.is_freeze(cx + dir, feet_row) && gap <= tc.edge_dist_px);
        near_edge
            && others
                .iter()
                .any(|o| o.player != c.player && !o.frozen() && Ctx::dist(o, c) <= tc.hook_range)
    };
    let finish = |label: u16, from: usize, to: usize, out: &mut Vec<TechniqueEvent>| {
        let ticks = tl.tick(to) - tl.tick(from) + ctx.dec();
        if ticks < tc.edge_stand_ticks {
            return;
        }
        let start = tl.tick(from) - ctx.dec();
        out.push(TechniqueEvent {
            technique: Technique::T11,
            actor: label,
            other: None,
            start_tick: start,
            end_tick: tl.tick(to),
            success: !ctx.frozen_in(label, start, tl.tick(to) - start + 25),
            active: true,
            detail: 0,
        });
    };
    for k in 0..tl.frames.len() {
        let mut continuing: Vec<u16> = Vec::new();
        for c in &tl.frames[k].chars {
            if on_edge(c, &tl.frames[k].chars) && (k == 0 || tl.contiguous(k - 1)) {
                continuing.push(c.player);
                runs.entry(c.player).or_insert(k);
            }
        }
        let ended: Vec<u16> = runs.keys().filter(|l| !continuing.contains(l)).copied().collect();
        for l in ended {
            let from = runs.remove(&l).expect("key from the map");
            finish(l, from, k - 1, &mut out);
        }
    }
    let rest: Vec<(u16, usize)> = runs.into_iter().collect();
    for (l, from) in rest {
        finish(l, from, tl.frames.len() - 1, &mut out);
    }
    out
}

/// T12 (second-jump save): a falling free character without its air jump used, freeze within
/// `save_freeze_below_px` below, uses the air jump (bit rises, vy drops by >= `save_jump_dvy`).
/// Success: not frozen for 50 ticks.
fn t12_second_jump_save(ctx: &Ctx<'_, '_>) -> Vec<TechniqueEvent> {
    let (tl, tc) = (ctx.tl, ctx.tc);
    let mut out = Vec::new();
    for k in 1..tl.frames.len() {
        if !tl.contiguous(k - 1) {
            continue;
        }
        for x1 in &tl.frames[k].chars {
            let Some(x0) = tl.ch(k - 1, x1.player) else { continue };
            if x0.frozen() || x0.grounded() || air_jump_used(x0) || !air_jump_used(x1) || x1.frozen() {
                continue;
            }
            if x0.vel[1] < 0.0 || x1.vel[1] - x0.vel[1] > -tc.save_jump_dvy {
                continue;
            }
            if tl
                .tiles
                .freeze_below(x0.pos[0], x0.pos[1], tc.save_freeze_below_px)
                .is_none()
            {
                continue;
            }
            let start = tl.tick(k - 1);
            out.push(TechniqueEvent {
                technique: Technique::T12,
                actor: x1.player,
                other: None,
                start_tick: start,
                end_tick: tl.tick(k),
                success: !ctx.frozen_in(x1.player, start, 50),
                active: true,
                detail: 0,
            });
        }
    }
    out
}

/// T13 (regain the double jump before the fight): lands (air jump used -> not used, grounded) with a
/// free enemy 300-500 px away. Success: not frozen for the next 50 ticks.
fn t13_regain_jump(ctx: &Ctx<'_, '_>) -> Vec<TechniqueEvent> {
    let tl = ctx.tl;
    let mut out = Vec::new();
    for k in 1..tl.frames.len() {
        if !tl.contiguous(k - 1) {
            continue;
        }
        for x1 in &tl.frames[k].chars {
            let Some(x0) = tl.ch(k - 1, x1.player) else { continue };
            if !(air_jump_used(x0) && !air_jump_used(x1) && x1.grounded() && !x1.frozen()) {
                continue;
            }
            let enemy = tl.frames[k]
                .chars
                .iter()
                .any(|o| o.player != x1.player && !o.frozen() && (300.0..=500.0).contains(&Ctx::dist(o, x1)));
            if !enemy {
                continue;
            }
            let start = tl.tick(k - 1);
            out.push(TechniqueEvent {
                technique: Technique::T13,
                actor: x1.player,
                other: None,
                start_tick: start,
                end_tick: tl.tick(k),
                success: !ctx.frozen_in(x1.player, start, 50),
                active: true,
                detail: 0,
            });
        }
    }
    out
}

/// T14 (panic hook): a free, airborne character with the air jump already used starts a hook (idle
/// -> flying/grabbed on terrain) while freeze is below it. `detail & 1`: the hook attached to
/// terrain within three snapshots. Success: not frozen for 50 ticks.
fn t14_panic_hook(ctx: &Ctx<'_, '_>) -> Vec<TechniqueEvent> {
    let tl = ctx.tl;
    let mut out = Vec::new();
    for k in 1..tl.frames.len() {
        if !tl.contiguous(k - 1) {
            continue;
        }
        for x1 in &tl.frames[k].chars {
            let Some(x0) = tl.ch(k - 1, x1.player) else { continue };
            let started = x0.hook_state <= 0 && (x1.hook_state == HOOK_FLYING || x1.hook_state == HOOK_GRABBED);
            if !started || x1.hooked_player >= 0 || x0.frozen() || x0.grounded() || !air_jump_used(x0) {
                continue;
            }
            if tl.tiles.freeze_below(x0.pos[0], x0.pos[1], 192.0).is_none() {
                continue;
            }
            let attached = (k..(k + 3).min(tl.frames.len())).any(|j| {
                tl.ch(j, x1.player).is_some_and(|c| {
                    c.hook_state == HOOK_GRABBED
                        && c.hooked_player < 0
                        && tl.tiles.hook_on_terrain(c.hook_pos[0], c.hook_pos[1])
                })
            });
            let start = tl.tick(k - 1);
            out.push(TechniqueEvent {
                technique: Technique::T14,
                actor: x1.player,
                other: None,
                start_tick: start,
                end_tick: tl.tick(k),
                success: !ctx.frozen_in(x1.player, start, 50),
                active: true,
                detail: u8::from(attached),
            });
        }
    }
    out
}

/// T15 (hook duel): two characters hooked onto each other for >= two snapshots with freeze near
/// either. One event per side; success = that side stayed free until 50 ticks after the duel.
fn t15_hook_duel(ctx: &Ctx<'_, '_>) -> Vec<TechniqueEvent> {
    let (tl, tc) = (ctx.tl, ctx.tc);
    let mut out = Vec::new();
    for a in ctx.hooks {
        for b in ctx.hooks {
            if !(a.actor == b.victim && a.victim == b.actor) || a.actor > a.victim {
                continue;
            }
            let s = a.start_k.max(b.start_k);
            let e = a.end_k.min(b.end_k);
            if e < s || e - s < 1 {
                continue;
            }
            let (Some(p), Some(q)) = (tl.ch(s, a.actor), tl.ch(s, b.actor)) else {
                continue;
            };
            if p.frozen() || q.frozen() {
                continue;
            }
            let at_stake = tl.tiles.nearest_freeze(p.pos[0], p.pos[1], tc.freeze_near_px).is_some()
                || tl.tiles.nearest_freeze(q.pos[0], q.pos[1], tc.freeze_near_px).is_some();
            if !at_stake {
                continue;
            }
            let start = tl.tick(s) - ctx.dec();
            let end = tl.tick(e);
            for (me, other) in [(a.actor, b.actor), (b.actor, a.actor)] {
                out.push(TechniqueEvent {
                    technique: Technique::T15,
                    actor: me,
                    other: Some(other),
                    start_tick: start,
                    end_tick: end,
                    success: !ctx.frozen_in(me, start, end - start + 50),
                    active: true,
                    detail: 0,
                });
            }
        }
    }
    out
}

/// The no-hook counterfactual of a character: its state at `seed` with the hook idle, evolved
/// with the client's isolated-world `Evolve` (real map collision, gravity and friction, the
/// direction it was holding, no other player) to `target_tick`.
fn ballistic(
    tl: &Timeline<'_>,
    seed: &CharRec,
    seed_tick: i32,
    target_tick: i32,
) -> ddai_physics::core::CharacterCore<f32> {
    let wire = ddai_net::generated::objects::Character {
        tick: seed_tick,
        x: seed.pos[0].round() as i32,
        y: seed.pos[1].round() as i32,
        vel_x: (seed.vel[0] * 256.0).round() as i32,
        vel_y: (seed.vel[1] * 256.0).round() as i32,
        angle: 0,
        direction: i32::from(seed.direction),
        jumped: if air_jump_used(seed) { 2 } else { 0 },
        hooked_player: -1,
        hook_state: 0,
        hook_tick: 0,
        hook_x: seed.pos[0].round() as i32,
        hook_y: seed.pos[1].round() as i32,
        hook_dx: 0,
        hook_dy: 0,
        player_flags: 0,
        health: 0,
        armor: 0,
        ammo_count: 0,
        weapon: i32::from(seed.weapon),
        emote: 0,
        attack_tick: 0,
    };
    ddai_world::reckoning::evolve_character_core(&wire, target_tick, tl.collision())
}

/// D-048: a wall or ceiling hook used to change trajectory **under threat**.
///
/// A run of >= two snapshots with the hook grabbed to terrain (`hooked_player < 0`, hook point on
/// a hookable tile), the hook point above or beside (not below) the character, the character free
/// at the snapshot before the run. Two conditions must then hold, both judged against the
/// **no-hook ballistic path** from the snapshot before the hook (`ballistic`, not against raw
/// velocity change, which gravity alone produces):
///
/// - *the hook changed the trajectory*: at some snapshot of the run the velocity differs from the
///   ballistic path by >= `wall_hook_min_dv` px/tick, or the position by >= `wall_hook_min_dpos` px;
/// - *there was a threat*, at least one of: an enemy closing in (a free tee within hook range that moved
///   towards where we were by >= `threat_closing_speed` px/tick since the previous snapshot), bit 4;
///   an enemy hook aimed at us (hooking us already, or flying towards us), bit 32; freeze on our
///   own ballistic path (the centre would enter a freeze tile within the run plus
///   `ballistic_horizon_ticks`), bit 8. A free tee merely being somewhere within 380 px is not a
///   threat.
///
/// `detail` bits: 1 ceiling, 2 wall, 4 enemy closing, 8 freeze on the ballistic path, 16 airborne
/// with the air jump used, 32 enemy hook aimed at us. Success: not frozen until 50 ticks after the
/// run; with bit 8 set that means the hook saved the character from a freeze it was heading for.
fn wall_hook(ctx: &Ctx<'_, '_>) -> Vec<TechniqueEvent> {
    let (tl, tc) = (ctx.tl, ctx.tc);
    let mut out = Vec::new();
    let mut open: std::collections::BTreeMap<u16, (usize, usize)> = std::collections::BTreeMap::new();
    let terrain = |c: &CharRec| {
        c.hook_state == HOOK_GRABBED && c.hooked_player < 0 && tl.tiles.hook_on_terrain(c.hook_pos[0], c.hook_pos[1])
    };
    let close = |label: u16, s: usize, e: usize, out: &mut Vec<TechniqueEvent>| {
        if e <= s || s == 0 || !tl.contiguous(s - 1) {
            return;
        }
        let (Some(x0), Some(seed)) = (tl.ch(s, label), tl.ch(s - 1, label)) else {
            return;
        };
        if seed.frozen() {
            return;
        }
        let (dx, dy) = (x0.hook_pos[0] - x0.pos[0], x0.hook_pos[1] - x0.pos[1]);
        let ceiling = dy < 0.0 && dy.abs() > dx.abs();
        let wall = dx.abs() >= dy.abs() && dy <= dx.abs();
        if !(ceiling || wall) {
            return; // a floor hook
        }
        let seed_tick = tl.tick(s - 1);

        // Did the hook change the trajectory compared with not hooking?
        let mut changed = false;
        for k in s..=e {
            let Some(c) = tl.ch(k, label) else { continue };
            let b = ballistic(tl, seed, seed_tick, tl.tick(k));
            let dv = ((c.vel[0] - b.vel.x).powi(2) + (c.vel[1] - b.vel.y).powi(2)).sqrt();
            let dp = ((c.pos[0] - b.pos.x).powi(2) + (c.pos[1] - b.pos.y).powi(2)).sqrt();
            if dv >= tc.wall_hook_min_dv || dp >= tc.wall_hook_min_dpos {
                changed = true;
                break;
            }
        }
        if !changed {
            return;
        }

        // Threats.
        let mut threat = 0u8;
        let my_id = i16::from(seed.id);
        for o in tl.frames[s].chars.iter().filter(|o| o.player != label && !o.frozen()) {
            // The enemy's own approach: how much *its* distance to where we were shrank, so our
            // own movement (the hook pulling us towards a still enemy) does not count.
            if Ctx::dist(o, x0) <= tc.hook_range
                && let Some(o_prev) = tl.ch(s - 1, o.player)
            {
                let (d_prev, d_now) = (Ctx::dist(o_prev, seed), Ctx::dist(o, seed));
                if (d_prev - d_now) / ctx.dec() as f32 >= tc.threat_closing_speed {
                    threat |= 4;
                }
            }
            // An enemy hook aimed at us: already hooking us, or flying towards us.
            let hooks_us = o.hook_state == HOOK_GRABBED && o.hooked_player == my_id;
            let flying_at_us = o.hook_state == HOOK_FLYING && {
                let (hx, hy) = (o.hook_pos[0] - o.pos[0], o.hook_pos[1] - o.pos[1]);
                let (ux, uy) = (x0.pos[0] - o.pos[0], x0.pos[1] - o.pos[1]);
                let (hn, un) = ((hx * hx + hy * hy).sqrt(), (ux * ux + uy * uy).sqrt());
                hn > 1.0 && un <= tc.hook_range + 50.0 && (hx * ux + hy * uy) / (hn * un.max(1.0)) >= 0.9
            };
            if hooks_us || flying_at_us {
                threat |= 32;
            }
        }
        // Freeze on our own ballistic path.
        let horizon = tl.tick(e) - seed_tick + tc_horizon(tc);
        let mut t = 2;
        while t <= horizon {
            let b = ballistic(tl, seed, seed_tick, seed_tick + t);
            let (cx, cy) = tl.tiles.cell(b.pos.x, b.pos.y);
            if tl.tiles.is_freeze(cx, cy) {
                threat |= 8;
                break;
            }
            t += 2;
        }
        if threat == 0 {
            return;
        }
        let detail = (u8::from(ceiling))
            | (u8::from(!ceiling) << 1)
            | threat
            | (u8::from(!x0.grounded() && air_jump_used(x0)) << 4);
        let start = seed_tick;
        let end = tl.tick(e);
        out.push(TechniqueEvent {
            technique: Technique::WallHook,
            actor: label,
            other: None,
            start_tick: start,
            end_tick: end,
            success: !ctx.frozen_in(label, start, end - start + 50),
            active: true,
            detail,
        });
    };
    for k in 0..tl.frames.len() {
        let mut here: Vec<u16> = Vec::new();
        if k == 0 || tl.contiguous(k - 1) {
            for c in &tl.frames[k].chars {
                if terrain(c) {
                    here.push(c.player);
                    open.entry(c.player).and_modify(|(_, e)| *e = k).or_insert((k, k));
                }
            }
        }
        let ended: Vec<u16> = open.keys().filter(|l| !here.contains(l)).copied().collect();
        for l in ended {
            let (s, e) = open.remove(&l).expect("key from the map");
            close(l, s, e, &mut out);
        }
    }
    let rest: Vec<(u16, (usize, usize))> = open.into_iter().collect();
    for (l, (s, e)) in rest {
        close(l, s, e, &mut out);
    }
    out
}

fn tc_horizon(tc: &TechniqueConfig) -> i32 {
    tc.ballistic_horizon_ticks.clamp(0, 100)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::fixtures::*;
    use crate::analysis::{attribute, freeze_entries, hammer_hits, hook_episodes};
    use crate::config::Config;
    use crate::ingest::KillEvent;
    use crate::testutil::{arena_with_pit, arena_with_pit_and_ceiling};
    use crate::types::FrameRec;
    use ddai_physics::map::{MapData, TILE_SOLID, Tile};

    /// 30 x 12 tiles; floor row 11 (surface y = 352, standing tee centre y = 338); freeze pit in
    /// the floor at x = 320..640; solid side walls at x < 32 and x >= 928.
    fn pit_map() -> MapData {
        arena_with_pit(30, 12, (10, 20))
    }

    fn run(map: &MapData, frames: &[FrameRec], kills: &[KillEvent]) -> Vec<TechniqueEvent> {
        run_cfg(&Config::default(), map, frames, kills)
    }

    fn run_cfg(cfg: &Config, map: &MapData, frames: &[FrameRec], kills: &[KillEvent]) -> Vec<TechniqueEvent> {
        let tl = Timeline::new(cfg, frames, kills, map);
        let entries = freeze_entries(&tl);
        let hooks = hook_episodes(&tl);
        let hits = hammer_hits(&tl);
        let attr = attribute(&tl, &entries, &hooks, &hits);
        let ctx = Ctx {
            tl: &tl,
            entries: &entries,
            hooks: &hooks,
            hits: &hits,
            attr: &attr,
            tc: &cfg.technique,
        };
        detect(&ctx)
    }

    fn of(events: &[TechniqueEvent], t: Technique) -> Vec<TechniqueEvent> {
        events.iter().filter(|e| e.technique == t).copied().collect()
    }

    fn fired(mut c: CharRec, aim: [i32; 2]) -> CharRec {
        c.flags |= char_flags::FIRED;
        c.aim = aim;
        c
    }

    fn air_jump_used_flag(mut c: CharRec) -> CharRec {
        c.flags |= char_flags::AIR_JUMP_USED;
        c
    }

    fn hook_terrain(mut c: CharRec, at: [f32; 2]) -> CharRec {
        c.hook_state = HOOK_GRABBED;
        c.hooked_player = -1;
        c.hook_pos = at;
        c
    }

    // Labels: attacker/defender A = 1 (id 0), other player V = 2 (id 1).
    fn a(x: f32, y: f32) -> CharRec {
        ch(0, 1, x, y)
    }
    fn v(x: f32, y: f32) -> CharRec {
        ch(1, 2, x, y)
    }

    /// A hooks V from the ground over the pit, drags it >= 16 px and V freezes: T1 success.
    fn t1_scenario(v_freezes: bool, attacker_grounded: bool, v_x: f32) -> Vec<FrameRec> {
        let att = |hook: bool| {
            let c = if attacker_grounded {
                grounded(a(250.0, 338.0))
            } else {
                a(250.0, 338.0)
            };
            if hook { hooking(c, 1) } else { c }
        };
        let mut frames = vec![
            frame(10, vec![att(false), v(v_x, 300.0)]),
            frame(12, vec![att(true), v(v_x - 15.0, 300.0)]),
            frame(14, vec![att(true), v(v_x - 35.0, 300.0)]),
            frame(16, vec![att(true), v(v_x - 55.0, 304.0)]),
        ];
        let last_v = if v_freezes {
            frozen(v(v_x - 75.0, 345.0))
        } else {
            v(v_x - 75.0, 345.0)
        };
        frames.push(frame(18, vec![att(false), last_v]));
        frames.push(frame(20, vec![att(false), last_v]));
        frames
    }

    #[test]
    fn t1_hook_drag_into_freeze_succeeds_when_the_victim_freezes() {
        let ev = run(&pit_map(), &t1_scenario(true, true, 420.0), &[]);
        let t1 = of(&ev, Technique::T1);
        assert_eq!(t1.len(), 1, "{ev:?}");
        assert_eq!((t1[0].actor, t1[0].other), (1, Some(2)));
        assert!(t1[0].success && t1[0].active);
        assert_eq!(t1[0].detail, 0, "the attacker stayed free");
        assert!(t1[0].start_tick < 12 && t1[0].end_tick == 16);
        assert!(of(&ev, Technique::T4).is_empty(), "victim was not falling");
    }

    #[test]
    fn t1_is_an_attempt_that_fails_when_the_victim_does_not_freeze() {
        let ev = run(&pit_map(), &t1_scenario(false, true, 420.0), &[]);
        let t1 = of(&ev, Technique::T1);
        assert_eq!(t1.len(), 1);
        assert!(!t1[0].success);
    }

    #[test]
    fn t1_needs_the_victim_to_be_dragged() {
        // The hook holds but the victim stays where it is (strong resistance): nothing was dragged.
        let mut frames = t1_scenario(true, true, 420.0);
        for f in frames.iter_mut().take(4) {
            f.chars[1].pos = [420.0, 300.0];
        }
        assert!(of(&run(&pit_map(), &frames, &[]), Technique::T1).is_empty());
    }

    #[test]
    fn t1_needs_freeze_at_stake_and_an_attacker_on_the_ground() {
        // Victim far from the pit: nothing at stake.
        assert!(of(&run(&pit_map(), &t1_scenario(false, true, 150.0), &[]), Technique::T1).is_empty());
        // Attacker in the air: that is another technique (or none).
        assert!(of(&run(&pit_map(), &t1_scenario(true, false, 420.0), &[]), Technique::T1).is_empty());
    }

    #[test]
    fn t1_records_that_the_attacker_froze_himself_too() {
        let mut frames = t1_scenario(true, true, 420.0);
        frames.push(frame(22, vec![frozen(a(250.0, 338.0)), frozen(v(345.0, 345.0))]));
        let ev = run(&pit_map(), &frames, &[]);
        let t1 = of(&ev, Technique::T1);
        assert!(t1[0].success);
        assert_eq!(t1[0].detail & 1, 1);
    }

    #[test]
    fn t4_pull_down_a_falling_jumper() {
        let mut frames = t1_scenario(true, true, 420.0);
        frames[1].chars[1].vel = [0.0, 5.0]; // falling at the hook start (episode starts at frame 1)
        let ev = run(&pit_map(), &frames, &[]);
        let t4 = of(&ev, Technique::T4);
        assert_eq!(t4.len(), 1, "{ev:?}");
        assert!(t4[0].success);
        // A grounded, slow victim is not a jumper.
        let ev2 = run(&pit_map(), &t1_scenario(true, true, 420.0), &[]);
        assert!(of(&ev2, Technique::T4).is_empty());
    }

    fn hammer_scenario(throw_vx: f32, v_freezes: bool) -> Vec<FrameRec> {
        let hit_frame_v = with_vel(v(277.0, 330.0), throw_vx, -8.4);
        let mut frames = vec![
            frame(10, vec![grounded(a(230.0, 338.0)), grounded(v(270.0, 338.0))]),
            frame(12, vec![fired(grounded(a(230.0, 338.0)), [1, 0]), hit_frame_v]),
            frame(14, vec![a(230.0, 338.0), v(290.0, 322.0)]),
        ];
        let last = if v_freezes {
            frozen(v(330.0, 350.0))
        } else {
            v(330.0, 350.0)
        };
        frames.push(frame(16, vec![a(230.0, 338.0), last]));
        frames.push(frame(18, vec![a(230.0, 338.0), last]));
        frames
    }

    #[test]
    fn t2_hammer_throw_into_a_side_freeze_wall() {
        let ev = run(&pit_map(), &hammer_scenario(6.7, true), &[]);
        let t2 = of(&ev, Technique::T2);
        assert_eq!(t2.len(), 1, "{ev:?}");
        assert!(t2[0].success);
        assert_eq!((t2[0].actor, t2[0].other), (1, Some(2)));
        // Thrown away from the pit: the wall is on the other side.
        assert!(of(&run(&pit_map(), &hammer_scenario(-6.7, false), &[]), Technique::T2).is_empty());
        // Thrown towards the pit but the victim survives: the attempt fails.
        let ev_fail = run(&pit_map(), &hammer_scenario(6.7, false), &[]);
        assert!(!of(&ev_fail, Technique::T2)[0].success);
        // A too gentle throw (|vx| < 3) is no side throw.
        let mut soft = hammer_scenario(6.7, true);
        soft[1].chars[1].vel = [2.0, -8.0];
        assert!(of(&run(&pit_map(), &soft, &[]), Technique::T2).is_empty());
    }

    #[test]
    fn t3_swing_up_is_hook_from_above_then_a_hammer_on_a_sideways_moving_victim() {
        let att = |x: f32, y: f32, hook: bool| {
            let c = a(x, y);
            if hook { hooking(c, 1) } else { c }
        };
        let frames = vec![
            frame(10, vec![att(270.0, 300.0, false), with_vel(v(300.0, 338.0), 3.0, 0.0)]),
            frame(12, vec![att(270.0, 300.0, true), with_vel(v(300.0, 338.0), 3.0, 0.0)]),
            frame(14, vec![att(280.0, 305.0, true), with_vel(v(300.0, 338.0), 3.0, 0.0)]),
            frame(
                16,
                vec![
                    fired(att(285.0, 310.0, false), [1, 2]),
                    with_vel(v(306.0, 330.0), 6.0, -8.0),
                ],
            ),
            frame(18, vec![a(285.0, 310.0), v(330.0, 350.0)]),
            frame(20, vec![a(285.0, 310.0), frozen(v(330.0, 350.0))]),
        ];
        let ev = run(&pit_map(), &frames, &[]);
        let t3 = of(&ev, Technique::T3);
        assert_eq!(t3.len(), 1, "{ev:?}");
        assert!(t3[0].success, "victim froze credited to the swinger");
        // A victim that barely moves sideways is not thrown by the finishing hammer.
        let mut slow = frames.clone();
        for f in slow.iter_mut().take(3) {
            f.chars[1].vel = [1.0, 0.0];
        }
        assert!(of(&run(&pit_map(), &slow, &[]), Technique::T3).is_empty());
        // Attacker below the victim: not a swing.
        let mut low = frames.clone();
        for f in low.iter_mut().take(3) {
            f.chars[0].pos[1] = 360.0;
        }
        assert!(of(&run(&pit_map(), &low, &[]), Technique::T3).is_empty());
    }

    #[test]
    fn t5_body_push_at_the_edge() {
        let mk = |closing: f32, freezes: bool| {
            let last = if freezes {
                frozen(v(325.0, 345.0))
            } else {
                v(325.0, 345.0)
            };
            vec![
                frame(
                    10,
                    vec![
                        with_vel(grounded(a(275.0, 338.0)), closing, 0.0),
                        grounded(v(300.0, 338.0)),
                    ],
                ),
                frame(
                    12,
                    vec![
                        with_vel(a(285.0, 338.0), closing, 0.0),
                        with_vel(v(310.0, 338.0), 3.0, 0.0),
                    ],
                ),
                frame(14, vec![a(295.0, 338.0), last]),
                frame(16, vec![a(295.0, 338.0), last]),
            ]
        };
        let ev = run(&pit_map(), &mk(4.0, true), &[]);
        let t5 = of(&ev, Technique::T5);
        assert_eq!(t5.len(), 1, "{ev:?}");
        assert!(t5[0].success);
        assert!(!of(&run(&pit_map(), &mk(4.0, false), &[]), Technique::T5)[0].success);
        // Barely moving: no push.
        assert!(of(&run(&pit_map(), &mk(0.5, true), &[]), Technique::T5).is_empty());
    }

    #[test]
    fn t8_hammering_a_frozen_enemy_is_reported_with_its_outcome() {
        let mk = |still_frozen: bool| {
            let v1 = with_vel(v(277.0, 330.0), 6.7, -8.4);
            vec![
                frame(10, vec![grounded(a(230.0, 338.0)), frozen(grounded(v(270.0, 338.0)))]),
                frame(
                    12,
                    vec![
                        fired(a(230.0, 338.0), [1, 0]),
                        if still_frozen { frozen(v1) } else { v1 },
                    ],
                ),
                frame(
                    14,
                    vec![
                        a(230.0, 338.0),
                        if still_frozen {
                            frozen(v(290.0, 322.0))
                        } else {
                            v(290.0, 322.0)
                        },
                    ],
                ),
                frame(16, vec![a(230.0, 338.0), v(300.0, 330.0)]),
            ]
        };
        let ev = run(&pit_map(), &mk(true), &[]);
        let t8 = of(&ev, Technique::T8);
        assert_eq!(t8.len(), 1);
        assert!(t8[0].success, "still frozen right after the hit");
        assert!(of(&ev, Technique::T2).is_empty(), "a frozen victim is not a T2 throw");
        assert!(!of(&run(&pit_map(), &mk(false), &[]), Technique::T8)[0].success);
    }

    fn hooked_x(counter: u8) -> Vec<FrameRec> {
        // A (id 0, label 1) on the left hooks X (id 1, label 2) over the pit.
        let mut frames = Vec::new();
        for (i, t) in [10, 12, 14, 16].into_iter().enumerate() {
            let hook = i >= 1;
            let att = if hook {
                hooking(grounded(a(200.0, 338.0)), 1)
            } else {
                grounded(a(200.0, 338.0))
            };
            let mut x = v(330.0 - 8.0 * i as f32, 300.0);
            if hook && counter & 1 != 0 {
                x.direction = 1; // pushes right, against the pull to the left
            }
            if hook && counter & 2 != 0 {
                x = hook_terrain(x, [928.0, 300.0]);
            }
            frames.push(frame(t, vec![att, x]));
        }
        frames.push(frame(18, vec![grounded(a(200.0, 338.0)), v(300.0, 300.0)]));
        frames
    }

    #[test]
    fn t9_escape_counts_the_counter_measure_and_the_outcome() {
        let ev = run(&pit_map(), &hooked_x(1), &[]);
        let t9 = of(&ev, Technique::T9);
        assert_eq!(t9.len(), 1, "{ev:?}");
        assert_eq!((t9[0].actor, t9[0].other), (2, Some(1)), "the defender is the actor");
        assert!(t9[0].active && t9[0].success);
        assert_eq!(t9[0].detail & 1, 1, "stay strong");
        let ev_wall = run(&pit_map(), &hooked_x(2), &[]);
        assert_eq!(of(&ev_wall, Technique::T9)[0].detail & 2, 2, "terrain hook");
        // Passive: hooked near freeze without doing anything is the at-risk baseline.
        let ev_passive = run(&pit_map(), &hooked_x(0), &[]);
        let p = of(&ev_passive, Technique::T9);
        assert_eq!(p.len(), 1);
        assert!(!p[0].active && p[0].detail == 0);
        // The same escape that fails: X freezes while hooked.
        let mut failing = hooked_x(1);
        failing[3].chars[1] = frozen(failing[3].chars[1]);
        let f = of(&run(&pit_map(), &failing, &[]), Technique::T9);
        assert!(!f[0].success);
    }

    #[test]
    fn t9_counts_an_upward_kick_against_the_pull_as_a_jump() {
        // No air-jump bit (the jump was a ground jump), but the vertical speed changes by >= 5.
        let mut frames = hooked_x(0);
        frames[2].chars[1].vel = [0.0, 2.0];
        frames[3].chars[1].vel = [0.0, -8.0];
        let t9 = of(&run(&pit_map(), &frames, &[]), Technique::T9);
        assert_eq!(t9.len(), 1);
        assert_eq!(t9[0].detail & 4, 4);
        assert!(t9[0].active);
    }

    #[test]
    fn t9_ignores_a_hook_far_from_any_freeze() {
        let mut frames = hooked_x(1);
        for f in frames.iter_mut() {
            f.chars[1].pos[0] = 100.0;
            f.chars[0].pos[0] = 60.0;
        }
        assert!(of(&run(&pit_map(), &frames, &[]), Technique::T9).is_empty());
    }

    #[test]
    fn t10_saving_yourself_when_thrown_up_under_a_freeze_ceiling() {
        let map = arena_with_pit_and_ceiling(30, 12, (10, 20), (5, 25));
        let mk = |save: bool, refreeze: bool| {
            let mut x2 = with_vel(v(400.0, 170.0), 0.0, -9.0);
            if save {
                x2 = hooking(x2, 0); // hooks A (id 0) below itself
            }
            let last = if refreeze {
                frozen(v(400.0, 40.0))
            } else {
                v(400.0, 120.0)
            };
            vec![
                frame(10, vec![grounded(a(400.0, 260.0)), v(400.0, 240.0)]),
                frame(
                    12,
                    vec![fired(a(400.0, 260.0), [0, -1]), with_vel(v(400.0, 200.0), 0.0, -10.0)],
                ),
                frame(14, vec![a(400.0, 260.0), x2]),
                frame(16, vec![a(400.0, 260.0), last]),
            ]
        };
        let ev = run(&map, &mk(true, false), &[]);
        let t10 = of(&ev, Technique::T10);
        assert_eq!(t10.len(), 1, "{ev:?}");
        assert_eq!((t10[0].actor, t10[0].other), (2, Some(1)));
        assert!(t10[0].active && t10[0].success);
        assert_eq!(t10[0].detail & 1, 1);
        let passive = of(&run(&map, &mk(false, true), &[]), Technique::T10);
        assert!(!passive[0].active && !passive[0].success);
        // A hit that does not send the victim upwards fast enough is no throw-up.
        let mut soft = mk(true, false);
        soft[1].chars[1].vel = [3.0, -6.0]; // still a hit (|dv| >= 4), but not thrown up
        assert!(of(&run(&map, &soft, &[]), Technique::T10).is_empty());
        // Without a ceiling there is nothing to be saved from.
        assert!(of(&run(&pit_map(), &mk(true, false), &[]), Technique::T10).is_empty());
    }

    #[test]
    fn t11_edge_stance_needs_a_long_still_stand_next_to_freeze_with_an_enemy_in_range() {
        let mk = |n: usize, enemy_x: f32| -> Vec<FrameRec> {
            (0..n)
                .map(|i| {
                    frame(
                        10 + 2 * i as i32,
                        vec![grounded(v(310.0, 338.0)), grounded(a(enemy_x, 338.0))],
                    )
                })
                .collect()
        };
        let ev = run(&pit_map(), &mk(14, 100.0), &[]);
        let t11 = of(&ev, Technique::T11);
        // Both stand still; only the character next to the pit qualifies.
        assert_eq!(t11.len(), 1, "{ev:?}");
        assert_eq!(t11[0].actor, 2);
        assert!(t11[0].success);
        assert!(
            of(&run(&pit_map(), &mk(6, 100.0), &[]), Technique::T11).is_empty(),
            "too short"
        );
        assert!(
            of(&run(&pit_map(), &mk(14, 900.0), &[]), Technique::T11).is_empty(),
            "no enemy in hook range"
        );
        let mut moving = mk(14, 100.0);
        for f in moving.iter_mut() {
            f.chars[0].vel = [3.0, 0.0];
        }
        assert!(
            of(&run(&pit_map(), &moving, &[]), Technique::T11).is_empty(),
            "not standing still"
        );
    }

    #[test]
    fn t12_second_jump_over_the_pit() {
        let frames = vec![
            frame(10, vec![with_vel(v(400.0, 300.0), 0.0, 4.0)]),
            frame(12, vec![air_jump_used_flag(with_vel(v(400.0, 308.0), 0.0, -5.0))]),
            frame(14, vec![air_jump_used_flag(with_vel(v(400.0, 296.0), 0.0, -4.0))]),
            frame(16, vec![air_jump_used_flag(with_vel(v(400.0, 292.0), 0.0, -3.0))]),
        ];
        let ev = run(&pit_map(), &frames, &[]);
        let t12 = of(&ev, Technique::T12);
        assert_eq!(t12.len(), 1, "{ev:?}");
        assert!(t12[0].success);
        // Over solid ground the jump saves nothing.
        let mut safe = frames.clone();
        for f in safe.iter_mut() {
            f.chars[0].pos[0] = 100.0;
        }
        assert!(of(&run(&pit_map(), &safe, &[]), Technique::T12).is_empty());
        // The air-jump bit rises but the fall is not really stopped: not a save.
        let mut weak = frames.clone();
        weak[1].chars[0].vel = [0.0, 3.5];
        assert!(of(&run(&pit_map(), &weak, &[]), Technique::T12).is_empty());
        // The jump was already used before: nothing new happens.
        let mut used = frames.clone();
        used[0].chars[0] = air_jump_used_flag(used[0].chars[0]);
        assert!(of(&run(&pit_map(), &used, &[]), Technique::T12).is_empty());
        // Fails when the character freezes anyway.
        let mut fail = frames.clone();
        fail[3].chars[0] = frozen(fail[3].chars[0]);
        assert!(!of(&run(&pit_map(), &fail, &[]), Technique::T12)[0].success);
    }

    #[test]
    fn t13_landing_with_the_double_jump_back_before_an_enemy_arrives() {
        let mk = |enemy_x: f32| {
            vec![
                frame(10, vec![air_jump_used_flag(v(300.0, 300.0)), a(enemy_x, 338.0)]),
                frame(12, vec![grounded(v(300.0, 338.0)), a(enemy_x, 338.0)]),
                frame(14, vec![grounded(v(300.0, 338.0)), a(enemy_x, 338.0)]),
            ]
        };
        assert_eq!(of(&run(&pit_map(), &mk(700.0), &[]), Technique::T13).len(), 1);
        assert!(
            of(&run(&pit_map(), &mk(340.0), &[]), Technique::T13).is_empty(),
            "enemy too close"
        );
    }

    #[test]
    fn t14_panic_hook_to_the_wall_when_out_of_jumps_above_freeze() {
        let x = |state: i8, pos: [f32; 2]| {
            let mut c = air_jump_used_flag(v(600.0, 250.0));
            c.hook_state = state;
            c.hook_pos = pos;
            c
        };
        let frames = vec![
            frame(10, vec![x(0, [600.0, 250.0])]),
            frame(12, vec![x(HOOK_FLYING, [700.0, 250.0])]),
            frame(14, vec![hook_terrain(x(0, [0.0, 0.0]), [928.0, 250.0])]),
            frame(16, vec![hook_terrain(x(0, [0.0, 0.0]), [928.0, 250.0])]),
        ];
        let ev = run(&pit_map(), &frames, &[]);
        let t14 = of(&ev, Technique::T14);
        assert_eq!(t14.len(), 1, "{ev:?}");
        assert_eq!(t14[0].detail & 1, 1, "attached to terrain");
        assert!(t14[0].success);
        // Still has its air jump: not a panic.
        let mut fresh = frames.clone();
        for f in fresh.iter_mut() {
            f.chars[0].flags &= !char_flags::AIR_JUMP_USED;
        }
        assert!(of(&run(&pit_map(), &fresh, &[]), Technique::T14).is_empty());
        // Hooking a player is not a wall hook.
        let mut p = frames.clone();
        p[1].chars[0].hooked_player = 3;
        assert!(of(&run(&pit_map(), &p, &[]), Technique::T14).is_empty());
    }

    #[test]
    fn t15_hook_duel_yields_one_event_per_side() {
        let both = |x: f32| {
            frame(
                0,
                vec![
                    hooking(ch(0, 1, 300.0 + x, 300.0), 1),
                    hooking(ch(1, 2, 420.0 - x, 300.0), 0),
                ],
            )
        };
        let mut frames = vec![both(0.0), both(2.0), both(4.0), both(6.0)];
        for (i, f) in frames.iter_mut().enumerate() {
            f.tick = 10 + 2 * i as i32;
        }
        let ev = run(&pit_map(), &frames, &[]);
        let t15 = of(&ev, Technique::T15);
        assert_eq!(t15.len(), 2, "{ev:?}");
        assert!(t15.iter().all(|e| e.success));
        let actors: Vec<u16> = t15.iter().map(|e| e.actor).collect();
        assert!(actors.contains(&1) && actors.contains(&2));
        // A one-sided hook is no duel.
        let mut one = frames.clone();
        for f in one.iter_mut() {
            f.chars[1].hook_state = 0;
            f.chars[1].hooked_player = -1;
        }
        assert!(of(&run(&pit_map(), &one, &[]), Technique::T15).is_empty());
    }

    fn ceiling_map() -> MapData {
        let mut m = pit_map();
        for x in 0..30usize {
            m.game[30 + x] = Tile {
                index: TILE_SOLID,
                ..Default::default()
            };
        }
        m
    }

    /// No-hook ballistic state of a character at `(x, y)` with velocity `vel` at tick 10, at `tick`.
    fn ballistic_at(map: &MapData, pos: [f32; 2], vel: [f32; 2], tick: i32) -> ([f32; 2], [f32; 2]) {
        let seed = with_vel(ch(0, 1, pos[0], pos[1]), vel[0], vel[1]);
        let c = ballistic(&Timeline::new(&Config::default(), &[], &[], map), &seed, 10, tick);
        ([c.pos.x, c.pos.y], [c.vel.x, c.vel.y])
    }

    /// Player 1 (id 0, "X") starts at `x0` with velocity `v0` at tick 10, hooks terrain at `hook_at`
    /// during ticks 12..=16 (`bump` extra px/tick of velocity per snapshot, added to the ballistic
    /// path: 0 = the hook changes nothing) and lets go at tick 18. `enemy(k)` gives player 2's
    /// character in snapshot `k` (0-based), if any.
    fn wall_hook_frames(
        map: &MapData,
        x0: [f32; 2],
        v0: [f32; 2],
        hook_at: [f32; 2],
        bump: f32,
        enemy: impl Fn(usize) -> Option<CharRec>,
    ) -> Vec<FrameRec> {
        (0..5)
            .map(|k| {
                let tick = 10 + 2 * k as i32;
                let (mut p, mut v) = ballistic_at(map, x0, v0, tick);
                let hooked = (1..=3).contains(&k);
                if hooked {
                    v[0] += bump * k as f32;
                    p[0] += bump * k as f32 * 2.0;
                }
                let mut x = with_vel(ch(0, 1, p[0], p[1]), v[0], v[1]);
                x.direction = 0;
                if hooked {
                    x = hook_terrain(x, hook_at);
                }
                let mut chars = vec![x];
                chars.extend(enemy(k));
                frame(tick, chars)
            })
            .collect()
    }

    fn wh(ev: &[TechniqueEvent]) -> Vec<TechniqueEvent> {
        of(ev, Technique::WallHook)
    }

    #[test]
    fn wall_hook_with_freeze_on_the_ballistic_path_is_a_threat_and_a_save() {
        // X falls towards the pit with the wall hook changing its path: the hook saved it.
        let m = ceiling_map();
        let frames = wall_hook_frames(&m, [600.0, 250.0], [0.0, 0.0], [928.0, 250.0], 3.0, |_| None);
        let ev = run(&m, &frames, &[]);
        let w = wh(&ev);
        assert_eq!(w.len(), 1, "{ev:?}");
        assert_eq!(w[0].detail & 8, 8, "freeze on the ballistic path");
        assert_eq!(w[0].detail & 2, 2, "wall");
        assert_eq!(w[0].detail & (4 | 32), 0, "no enemy involved");
        assert!(w[0].success);
        assert_eq!(
            w[0].start_tick, 10,
            "the decision snapshot is the one before the hook attached"
        );
        // The same run ending in a freeze is an escape that failed.
        let mut failed = frames.clone();
        failed[4].chars[0] = frozen(failed[4].chars[0]);
        assert!(!wh(&run(&m, &failed, &[]))[0].success);
    }

    #[test]
    fn freeze_on_the_ballistic_path_only_counts_within_the_horizon() {
        let m = ceiling_map();
        // From y = 250 the free fall reaches the pit within ~20 ticks: a threat.
        let near = wall_hook_frames(&m, [600.0, 250.0], [0.0, 0.0], [928.0, 250.0], 3.0, |_| None);
        assert_eq!(wh(&run(&m, &near, &[])).len(), 1);
        // From y = 80 it needs ~33 ticks, more than the run plus the 25-tick horizon: no threat yet.
        let far = wall_hook_frames(&m, [600.0, 80.0], [0.0, 0.0], [928.0, 80.0], 3.0, |_| None);
        assert!(wh(&run(&m, &far, &[])).is_empty());
        // The horizon is a config value: with 50 ticks the same hook counts.
        let mut cfg = Config::default();
        cfg.technique.ballistic_horizon_ticks = 50;
        let w = wh(&run_cfg(&cfg, &m, &far, &[]));
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].detail & 8, 8);
    }

    #[test]
    fn ceiling_hook_is_classified() {
        let m = ceiling_map();
        let frames = wall_hook_frames(&m, [600.0, 250.0], [0.0, 0.0], [600.0, 50.0], 3.0, |_| None);
        let w = wh(&run(&m, &frames, &[]));
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].detail & 1, 1, "ceiling");
        assert_eq!(w[0].detail & 2, 0);
    }

    #[test]
    fn a_closing_enemy_or_an_enemy_hook_at_us_is_a_threat_without_any_freeze() {
        let m = ceiling_map();
        // X stands high above the solid floor on the far left: nothing to freeze in.
        let x0 = [100.0, 100.0];
        let closing = |k: usize| Some(v(400.0 - 20.0 * k as f32, 338.0));
        let frames = wall_hook_frames(&m, x0, [0.0, 0.0], [32.0, 100.0], 3.0, closing);
        let w = wh(&run(&m, &frames, &[]));
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].detail & 4, 4, "closing enemy");
        assert_eq!(w[0].detail & 8, 0, "no freeze on the path");
        // A hook flying at us.
        let flying = |_k: usize| {
            let mut e = v(300.0, 100.0);
            e.hook_state = HOOK_FLYING;
            e.hook_pos = [250.0, 100.0];
            Some(e)
        };
        let frames = wall_hook_frames(&m, x0, [0.0, 0.0], [32.0, 100.0], 3.0, flying);
        let w = wh(&run(&m, &frames, &[]));
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].detail & 32, 32);
        assert_eq!(w[0].detail & 4, 0, "the enemy stands still");
        // A hook flying somewhere else is not aimed at us.
        let elsewhere = |_k: usize| {
            let mut e = v(300.0, 100.0);
            e.hook_state = HOOK_FLYING;
            e.hook_pos = [350.0, 100.0];
            Some(e)
        };
        let frames = wall_hook_frames(&m, x0, [0.0, 0.0], [32.0, 100.0], 3.0, elsewhere);
        assert!(wh(&run(&m, &frames, &[])).is_empty());
        // ... and one already hooked onto us.
        let hooking_us = |_k: usize| Some(hooking(v(300.0, 100.0), 0));
        let frames = wall_hook_frames(&m, x0, [0.0, 0.0], [32.0, 100.0], 3.0, hooking_us);
        assert_eq!(wh(&run(&m, &frames, &[]))[0].detail & 32, 32);
    }

    #[test]
    fn a_free_tee_standing_within_hook_range_is_not_a_threat() {
        // The old rule: any free tee within 380 px. Nothing closes in, nothing hooks, no freeze.
        let m = ceiling_map();
        let still = |_k: usize| Some(v(250.0, 338.0));
        let frames = wall_hook_frames(&m, [100.0, 100.0], [0.0, 0.0], [32.0, 100.0], 3.0, still);
        assert!(wh(&run(&m, &frames, &[])).is_empty());
        // An enemy moving away is not closing in either.
        let leaving = |k: usize| Some(v(250.0 + 20.0 * k as f32, 338.0));
        let frames = wall_hook_frames(&m, [100.0, 100.0], [0.0, 0.0], [32.0, 100.0], 3.0, leaving);
        assert!(wh(&run(&m, &frames, &[])).is_empty());
        // A frozen enemy closing in is no threat.
        let frozen_enemy = |k: usize| Some(frozen(v(400.0 - 20.0 * k as f32, 338.0)));
        let frames = wall_hook_frames(&m, [100.0, 100.0], [0.0, 0.0], [32.0, 100.0], 3.0, frozen_enemy);
        assert!(wh(&run(&m, &frames, &[])).is_empty());
    }

    #[test]
    fn a_hook_that_leaves_the_ballistic_path_alone_is_not_a_trajectory_change() {
        // Under threat (freeze below) but the recorded path equals free fall: gravity alone is not
        // "used the hook to change trajectory".
        let m = ceiling_map();
        let frames = wall_hook_frames(&m, [600.0, 250.0], [0.0, 0.0], [928.0, 250.0], 0.0, |_| None);
        assert!(wh(&run(&m, &frames, &[])).is_empty());
        // The same with a real change is an event.
        let frames = wall_hook_frames(&m, [600.0, 250.0], [0.0, 0.0], [928.0, 250.0], 3.0, |_| None);
        assert_eq!(wh(&run(&m, &frames, &[])).len(), 1);
    }

    #[test]
    fn wall_hook_needs_terrain_a_side_or_top_hook_and_a_free_character() {
        let m = ceiling_map();
        // Hook point below the character: a floor hook.
        let frames = wall_hook_frames(&m, [600.0, 250.0], [0.0, 0.0], [600.0, 336.0], 3.0, |_| None);
        assert!(wh(&run(&m, &frames, &[])).is_empty());
        // Hook point in the air (not terrain).
        let frames = wall_hook_frames(&m, [600.0, 250.0], [0.0, 0.0], [700.0, 250.0], 3.0, |_| None);
        assert!(wh(&run(&m, &frames, &[])).is_empty());
        // Frozen at the snapshot before the hook: a frozen tee cannot hook.
        let mut frames = wall_hook_frames(&m, [600.0, 250.0], [0.0, 0.0], [928.0, 250.0], 3.0, |_| None);
        frames[0].chars[0] = frozen(frames[0].chars[0]);
        assert!(wh(&run(&m, &frames, &[])).is_empty());
    }

    #[test]
    fn events_are_sorted_and_deterministic() {
        let frames = t1_scenario(true, true, 420.0);
        let a1 = run(&pit_map(), &frames, &[]);
        let a2 = run(&pit_map(), &frames, &[]);
        assert_eq!(a1, a2);
        assert!(a1.windows(2).all(|w| w[0].start_tick <= w[1].start_tick));
    }
}
