//! The technique library (task 3.5, D-048): explicit candidate generators ("macro-plans") for
//! the moves of `docs/research/block-knowledge.md` §2.1, expressed in the planner's plan encoding
//! and scored by the same exact rollouts as every other candidate. A generator never decides
//! anything: it only makes sure the move is *in the pool* when the situation calls for it, and the
//! search keeps it only if it scores best under the modelled responses.
//!
//! Every generator has a cheap geometric/state trigger, so a calm decision produces a handful of
//! plans and a dangerous one produces the escape families first. Plans that hook a wall use an
//! absolute aim ([`crate::hybrid::abs_aim`]); plans that hook or hit the victim use the planner's
//! victim-relative aim.
//!
//! | Tag | Family | Trigger |
//! |---|---|---|
//! | T9 | walk against the pull, or anchor to the floor/wall and walk against it | somebody hooks us |
//! | T10 | hook the tee below (or the floor) while flying up toward a hazard | rising, hazard above |
//! | T12 | freeze jump: second jump timed just before the hazard, sideways | falling onto a hazard, jump left |
//! | T13 | steer to a platform and refill the jump | no jumps, airborne, ground within 6 tiles |
//! | T14 | panic hook: hook a wall or ceiling and hang or swing | no jumps, airborne, hazard below |
//! | T15 | leave a hook duel we would lose (weak hook) | weak side, opponent in hook range |
//! | T17 | leave a free third tee's hook line that drags us over a hazard | third tee, drag crosses a hazard |
//! | T1 | hook drag through the edge into the hazard | hazard on the drag line or near the victim |
//! | T2/T3 | hammer the victim into a side wall / up into a ceiling | victim in reach, hazard near it |
//! | T3 | swing: hook an anchor above, then hammer | anchor above, victim close |
//! | T4 | pull down a jumper | victim airborne over a hazard |
//! | T5 | body push at the edge, brake before going in | victim within 320 px (run-in), both grounded, hazard beside it |
//! | T7 | push a frozen victim deeper, never go in ourselves | victim frozen, hazard near |
//! | T8 | hands off a frozen victim (no hammer) | victim frozen and close |
//! | T29 | jump before a freeze ahead when running or falling sideways fast | `|vx| > 2.2`, freeze 1-3 tiles ahead |
//! | T1b | hook the victim, jump past it, pull it on toward the hazard behind it | hazard beyond the victim, away from us |
//! | T42 | hook the victim along a ray that clears the corner blocking the direct line | victim in rope range, direct line blocked |

use crate::fields::{HazardField, hazard_nearness};
use crate::hybrid::abs_aim;
use crate::hybrid::anchors::{Anchor, AnchorKind};
use crate::plan_world::PlanCollision;
use crate::planner::PlanStep;
use crate::tuning::HOOK_LENGTH;
use crate::types::{HOOK_FLYING, HOOK_GRABBED, TeeState};
use crate::vmath::{Vec2, vdistance};
use ddai_jsmath as js;

/// A technique, as named in telemetry ("T14 panic hook").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tech {
    /// A wall/ceiling hook chosen by geometry alone (danger, but none of the T-triggers fits).
    AnchorEscape,
    T1,
    T2,
    T3,
    T4,
    T5,
    T7,
    T8,
    T9,
    T10,
    T12,
    T13,
    T14,
    T15,
    T17,
    /// Far anchor against a player's hook (R1 catalogue T19).
    T19,
    /// Ground stand against a distant hook: release the direction and let friction hold (T20).
    T20,
    /// Counter-steer at once after a sideways hammer hit (anti-T2, T21).
    T21,
    /// Release our own hook when holding it drags us into a hazard (T28).
    T28,
    /// Jump before the freeze ahead at speed (R1 catalogue T29).
    T29,
    /// Hook aimed past the corner that blocks the direct line to the victim (T42).
    T42,
    /// Hook the victim, jump past it and pull it on toward the hazard behind it (leapfrog, a T1 variant).
    T1b,
}

impl Tech {
    pub fn name(self) -> &'static str {
        match self {
            Tech::AnchorEscape => "anchor escape",
            Tech::T1 => "T1 hook drag",
            Tech::T2 => "T2 hammer throw into wall",
            Tech::T3 => "T3 swing up",
            Tech::T4 => "T4 pull down a jumper",
            Tech::T5 => "T5 body push",
            Tech::T7 => "T7 finish a frozen opponent",
            Tech::T8 => "T8 hands off a frozen opponent",
            Tech::T9 => "T9 escape the hook",
            Tech::T10 => "T10 hook the opponent below",
            Tech::T12 => "T12 freeze jump",
            Tech::T13 => "T13 refill the jump",
            Tech::T14 => "T14 panic hook",
            Tech::T15 => "T15 avoid the hook duel",
            Tech::T17 => "T17 leave the third tee's hook range",
            Tech::T19 => "T19 far anchor against the hook",
            Tech::T20 => "T20 friction stand against a far hook",
            Tech::T21 => "T21 counter-steer after a hammer hit",
            Tech::T28 => "T28 release the hook that drags us in",
            Tech::T29 => "T29 jump before the freeze ahead",
            Tech::T42 => "T42 hook past the corner",
            Tech::T1b => "T1b hook leapfrog",
        }
    }

    /// Survival techniques (the D-048 gate group plus the other defensive ones).
    pub fn is_defensive(self) -> bool {
        matches!(
            self,
            Tech::AnchorEscape
                | Tech::T9
                | Tech::T10
                | Tech::T12
                | Tech::T13
                | Tech::T14
                | Tech::T15
                | Tech::T17
                | Tech::T19
                | Tech::T20
                | Tech::T21
                | Tech::T28
                | Tech::T29
        )
    }
}

/// One generated candidate.
#[derive(Debug, Clone)]
pub struct TechPlan {
    pub tech: Tech,
    pub plan: Vec<PlanStep>,
}

/// Everything a generator reads.
pub struct TechCtx<'a, C: PlanCollision> {
    pub col: &'a C,
    pub field: &'a HazardField,
    pub me: &'a TeeState,
    pub victim: &'a TeeState,
    /// Free threats besides the victim, nearest first.
    pub threats: &'a [TeeState],
    /// A tee whose hook holds us, if any.
    pub hooked_by: Option<&'a TeeState>,
    pub steps: usize,
    /// Absolute angle from us to the victim.
    pub aim_at: f64,
    /// `Some(false)` when the victim's hook is the strong one (spawned earlier): a hook duel with
    /// it is one we would lose.
    pub me_strong: Option<bool>,
}

/// How many plans each family may emit.
#[derive(Debug, Clone, Copy)]
pub struct TechCaps {
    /// Anchors used by the escape families.
    pub escape_anchors: usize,
    /// Anchors used by the offensive swing.
    pub swing_anchors: usize,
    /// Emit the generic wall/ceiling escape even when no T-trigger fits (danger without a cause).
    pub generic_escape: bool,
}

impl Default for TechCaps {
    fn default() -> Self {
        TechCaps {
            escape_anchors: 4,
            swing_anchors: 2,
            generic_escape: false,
        }
    }
}

/// The two priority-ordered candidate lists.
#[derive(Debug, Clone, Default)]
pub struct Generated {
    pub defence: Vec<TechPlan>,
    pub offence: Vec<TechPlan>,
}

fn st(dir: i32, jump: bool, hook: bool, fire: bool, aim: f64) -> PlanStep {
    PlanStep {
        dir,
        jump: i32::from(jump),
        hook: i32::from(hook),
        fire: i32::from(fire),
        aim,
    }
}

fn plan(n: usize, f: impl Fn(usize) -> PlanStep) -> Vec<PlanStep> {
    (0..n).map(f).collect()
}

fn sign_dir(v: f64, dead: f64) -> i32 {
    if v > dead {
        1
    } else if v < -dead {
        -1
    } else {
        0
    }
}

fn hazard_at(col: &impl PlanCollision, x: f64, y: f64) -> bool {
    col.is_freeze(x, y) || col.is_death(x, y)
}

/// The number of tiles from `pos` (`dir` = `+1` down, `-1` up) to the first freeze/death tile
/// before any solid one, over three columns across the tee; `None` if a wall comes first.
pub fn hazard_vertical(col: &impl PlanCollision, pos: Vec2, dir: f64, max_tiles: i32) -> Option<i32> {
    let mut blocked = [false; 3];
    for k in 1..=max_tiles {
        for (i, dx) in [-12.0, 0.0, 12.0].into_iter().enumerate() {
            if blocked[i] {
                continue;
            }
            let (x, y) = (pos.x + dx, pos.y + dir * f64::from(k) * 32.0);
            if col.is_solid(x, y) {
                blocked[i] = true;
            } else if hazard_at(col, x, y) {
                return Some(k);
            }
        }
        if blocked.iter().all(|b| *b) {
            return None;
        }
    }
    None
}

/// Whether a freeze/death tile lies on the segment `from -> to` or up to two tiles under it
/// before a solid floor (a tee dragged along the segment falls into it). The scan of the segment
/// starts 40 px from `from`: the victim itself does not count as standing over the pit.
pub fn hazard_under_line(col: &impl PlanCollision, from: Vec2, to: Vec2) -> bool {
    let len = vdistance(from, to);
    if len < 1.0 {
        return false;
    }
    let (ux, uy) = ((to.x - from.x) / len, (to.y - from.y) / len);
    let mut t = 40.0;
    while t < len - 20.0 {
        let (x, y) = (from.x + ux * t, from.y + uy * t);
        for dy in [0.0, 32.0, 64.0] {
            if col.is_solid(x, y + dy) {
                break;
            }
            if hazard_at(col, x, y + dy) {
                return true;
            }
        }
        t += 24.0;
    }
    false
}

/// Whether the tee stands on something (the same probe the shield uses).
pub fn grounded(col: &impl PlanCollision, pos: Vec2) -> bool {
    col.is_solid(pos.x - 13.0, pos.y + 16.0) || col.is_solid(pos.x + 13.0, pos.y + 16.0)
}

/// The unit direction in which the hazard gets nearer (zero when standing at a local maximum).
pub fn toward_hazard(field: &HazardField, pos: Vec2) -> Vec2 {
    let here = hazard_nearness(field, pos.x, pos.y);
    let mut best = here + 1e-9;
    let mut out = Vec2 { x: 0.0, y: 0.0 };
    for k in 0..16 {
        let a = f64::from(k) * js::PI / 8.0;
        let (dx, dy) = (js::cos(a), js::sin(a));
        let near = hazard_nearness(field, pos.x + dx * 64.0, pos.y + dy * 64.0);
        if near > best {
            best = near;
            out = Vec2 { x: dx, y: dy };
        }
    }
    out
}

/// Ticks until a tee falling from `pos` with `vy` reaches the top of the hazard tile `k` tiles
/// below (a vertical-only forecast: gravity 0.5 per tick).
fn ticks_to_hazard(pos: Vec2, vy: f64, k: i32) -> Option<i32> {
    let top = (js::floor(pos.y / 32.0) + f64::from(k)) * 32.0;
    let (mut y, mut v) = (pos.y, vy);
    for t in 1..=90 {
        v += 0.5;
        y += v;
        if y >= top {
            return Some(t);
        }
    }
    None
}

/// How good an anchor is for getting away from the hazard: pulls away from it, upward, and is
/// close; a ceiling beats a floor when the hazard is below.
fn escape_utility(a: &Anchor, hz: Vec2, hazard_below: bool) -> f64 {
    let (ux, uy) = (js::cos(a.angle), js::sin(a.angle));
    let mut u = -(ux * hz.x + uy * hz.y) + 0.6 * (-uy) - 0.3 * (a.dist / *HOOK_LENGTH);
    if hazard_below {
        u += match a.kind {
            AnchorKind::Ceiling => 0.3,
            AnchorKind::Wall => 0.2,
            AnchorKind::Floor => -0.4,
        };
    }
    u
}

fn ranked_escape_anchors(anchors: &[Anchor], hz: Vec2, hazard_below: bool, k: usize) -> Vec<&Anchor> {
    let mut v: Vec<&Anchor> = anchors.iter().collect();
    v.sort_by(|a, b| escape_utility(b, hz, hazard_below).total_cmp(&escape_utility(a, hz, hazard_below)));
    v.truncate(k);
    v
}

/// Hook the anchor and hang from it (`hold` steps), optionally walking with `dir`, optionally
/// releasing early to carry the momentum.
fn hook_plan(n: usize, a: &Anchor, dir: i32, jump_at: Option<usize>, hold: usize) -> Vec<PlanStep> {
    plan(n, |s| st(dir, jump_at == Some(s), s < hold, false, abs_aim(a.angle)))
}

/// The panic-hook / wall-escape plans of the best anchors: hang from each of the top ones, and
/// swing (walk toward the anchor's side and release two thirds through) from the top two.
fn escape_plans(
    ctx: &TechCtx<'_, impl PlanCollision>,
    anchors: &[Anchor],
    hz: Vec2,
    hazard_below: bool,
    k: usize,
    tech: Tech,
    out: &mut Vec<TechPlan>,
) {
    let n = ctx.steps;
    let top = ranked_escape_anchors(anchors, hz, hazard_below, k);
    for (i, a) in top.iter().enumerate() {
        out.push(TechPlan {
            tech,
            plan: hook_plan(n, a, 0, None, n),
        });
        if i < 2 {
            let toward = sign_dir(a.point.x - ctx.me.pos.x, 8.0);
            out.push(TechPlan {
                tech,
                plan: hook_plan(n, a, toward, (ctx.me.jumps_left > 0).then_some(1), (2 * n) / 3),
            });
        }
    }
}

/// All generators for the current situation, each list in priority order.
pub fn generate<C: PlanCollision>(ctx: &TechCtx<'_, C>, anchors: &[Anchor], caps: &TechCaps) -> Generated {
    let mut g = Generated::default();
    defence(ctx, anchors, caps, &mut g.defence);
    offence(ctx, anchors, caps, &mut g.offence);
    g
}

fn defence<C: PlanCollision>(ctx: &TechCtx<'_, C>, anchors: &[Anchor], caps: &TechCaps, out: &mut Vec<TechPlan>) {
    let (me, n) = (ctx.me, ctx.steps);
    if me.frozen || !me.alive {
        return;
    }
    let hz = toward_hazard(ctx.field, me.pos);
    let air = !grounded(ctx.col, me.pos);
    let below = hazard_vertical(ctx.col, me.pos, 1.0, 12);
    let hazard_below = below.is_some();
    let mut specific = false;

    // T9/T19/T20: somebody hooks us. Walk against the pull (T9, near hooker); on the ground with the
    // hooker far away let friction hold instead (T20: walking makes the drag worse beyond about
    // 170 px); anchor to the floor or, best, a far wall on the other side and steer to it (T19: works
    // up to ~367 px, the pull is weaker than a wall hook plus steering).
    if let Some(h) = ctx.hooked_by {
        specific = true;
        let pull = Vec2 {
            x: h.pos.x - me.pos.x,
            y: h.pos.y - me.pos.y,
        };
        let dist = vdistance(h.pos, me.pos);
        let against = -sign_dir(pull.x, 12.0);
        let stand = !air && dist > 170.0;
        if stand {
            out.push(TechPlan {
                tech: Tech::T20,
                plan: plan(n, |_| st(0, false, false, false, 0.0)),
            });
        }
        out.push(TechPlan {
            tech: Tech::T9,
            plan: plan(n, |_| st(against, false, false, false, 0.0)),
        });
        // Anchors on the far side from the hooker first.
        let (hx, hy) = (-pull.x / dist.max(1.0), -pull.y / dist.max(1.0));
        let mut ranked = ranked_escape_anchors(anchors, hz, hazard_below, 4);
        ranked.sort_by(|a, b| {
            let far = |a: &Anchor| js::cos(a.angle) * hx + js::sin(a.angle) * hy;
            far(b).total_cmp(&far(a))
        });
        for a in ranked.into_iter().take(3) {
            let far_side = js::cos(a.angle) * hx + js::sin(a.angle) * hy > 0.2;
            let steer = if far_side {
                sign_dir(a.point.x - me.pos.x, 8.0)
            } else {
                against
            };
            let tech = if far_side && dist < 367.0 { Tech::T19 } else { Tech::T9 };
            out.push(TechPlan {
                tech,
                plan: hook_plan(n, a, steer, None, n),
            });
            out.push(TechPlan {
                tech,
                plan: hook_plan(n, a, steer, (me.jumps_left > 0).then_some(0), n),
            });
        }
        out.push(TechPlan {
            tech: Tech::T9,
            plan: plan(n, |s| st(against, s == 0 && me.jumps_left > 0, false, false, 0.0)),
        });
    }

    // T29: fast sideways toward a freeze one to three tiles ahead at our level: a jump over it (from the
    // ground, or the air jump when falling onto it) beats running or falling in.
    if me.vel.x.abs() > 2.2 && me.jumps_left > 0 {
        let dir = sign_dir(me.vel.x, 0.0);
        let ahead = (1..=3).any(|k| {
            let x = me.pos.x + f64::from(dir) * f64::from(k) * 32.0;
            hazard_at(ctx.col, x, me.pos.y + 24.0) || hazard_at(ctx.col, x, me.pos.y + 56.0)
        });
        if ahead && (!air || me.vel.y > 1.0) {
            specific = true;
            out.push(TechPlan {
                tech: Tech::T29,
                plan: plan(n, |s| st(dir, s == 0, false, false, 0.0)),
            });
            out.push(TechPlan {
                tech: Tech::T29,
                plan: plan(n, |s| st(dir, s == 1, false, false, 0.0)),
            });
        }
    }

    // T28: holding our own hook drags us toward a hazard: let go.
    if (me.hook_state == HOOK_GRABBED || me.hook_state == HOOK_FLYING)
        && !specific
        && (hazard_below || hz.x != 0.0 || hz.y != 0.0)
    {
        out.push(TechPlan {
            tech: Tech::T28,
            plan: plan(n, |_| st(0, false, false, false, 0.0)),
        });
        let away = if hz.x > 0.1 {
            -1
        } else if hz.x < -0.1 {
            1
        } else {
            0
        };
        if away != 0 {
            out.push(TechPlan {
                tech: Tech::T28,
                plan: plan(n, |_| st(away, false, false, false, 0.0)),
            });
        }
    }

    // T21: flying sideways toward a hazard (a hammer hit sends a tee off at ~6.7 px/tick): counter-steer
    // from the very first tick.
    if me.vel.x.abs() > 5.0 && hz.x.abs() > 0.3 && sign_dir(me.vel.x, 0.0) == sign_dir(hz.x, 0.0) {
        let counter = -sign_dir(me.vel.x, 0.0);
        out.push(TechPlan {
            tech: Tech::T21,
            plan: plan(n, |_| st(counter, false, false, false, 0.0)),
        });
    }

    // T10: flying up toward a hazard with a tee below: hook it (or the floor) and pull ourselves down.
    if me.vel.y < -3.0 && hazard_vertical(ctx.col, me.pos, -1.0, 10).is_some() {
        let mut below_tees: Vec<&TeeState> = std::iter::once(ctx.victim)
            .chain(ctx.threats.iter())
            .filter(|t| t.alive && t.pos.y > me.pos.y + 24.0 && vdistance(t.pos, me.pos) < *HOOK_LENGTH - 10.0)
            .collect();
        below_tees.sort_by(|a, b| vdistance(a.pos, me.pos).total_cmp(&vdistance(b.pos, me.pos)));
        if let Some(t) = below_tees.first() {
            specific = true;
            let ang = js::atan2(t.pos.y - me.pos.y, t.pos.x - me.pos.x);
            let side = sign_dir(t.pos.x - me.pos.x, 12.0);
            out.push(TechPlan {
                tech: Tech::T10,
                plan: plan(n, |_| st(0, false, true, false, abs_aim(ang))),
            });
            out.push(TechPlan {
                tech: Tech::T10,
                plan: plan(n, |s| st(side, s == 1 && me.jumps_left > 0, true, false, abs_aim(ang))),
            });
        }
        let floors: Vec<Anchor> = anchors
            .iter()
            .copied()
            .filter(|a| a.kind != AnchorKind::Ceiling && js::sin(a.angle) > 0.2)
            .collect();
        for a in floors.iter().take(2) {
            specific = true;
            out.push(TechPlan {
                tech: Tech::T10,
                plan: hook_plan(n, a, 0, None, n),
            });
        }
    }

    // T14: over a hazard with no jumps left: hook a wall or ceiling and hang.
    if air && me.jumps_left == 0 && hazard_below {
        specific = true;
        escape_plans(ctx, anchors, hz, true, caps.escape_anchors, Tech::T14, out);
    }

    // T12: falling onto a hazard with a jump left: jump sideways just before touching it.
    if air
        && me.jumps_left > 0
        && me.vel.y > -1.0
        && let Some(k) = below
        && let Some(ttc) = ticks_to_hazard(me.pos, me.vel.y, k)
    {
        specific = true;
        let s0 = (((ttc - 2) as f64) / 3.0).floor().max(0.0) as usize;
        let s0 = s0.min(n - 1);
        let away = if hz.x > 0.1 {
            -1
        } else if hz.x < -0.1 {
            1
        } else {
            sign_dir(me.vel.x, 0.5).max(-1)
        };
        let away = if away == 0 { 1 } else { away };
        let mut steps_to_try = vec![s0];
        if s0 > 0 {
            steps_to_try.push(s0 - 1);
        }
        if s0 + 1 < n {
            steps_to_try.push(s0 + 1);
        }
        for sj in steps_to_try {
            for dir in [away, -away] {
                out.push(TechPlan {
                    tech: Tech::T12,
                    plan: plan(n, |s| st(dir, s == sj, false, false, 0.0)),
                });
            }
        }
    }

    // Generic wall/ceiling escape: danger without a specific cause.
    if !specific && caps.generic_escape && air {
        escape_plans(
            ctx,
            anchors,
            hz,
            hazard_below,
            caps.escape_anchors,
            Tech::AnchorEscape,
            out,
        );
    }

    // T15: a hook duel the strong side would win: leave it.
    if ctx.me_strong == Some(false) {
        let opp = std::iter::once(ctx.victim)
            .chain(ctx.threats.iter())
            .filter(|t| t.alive && !t.frozen && vdistance(t.pos, me.pos) < *HOOK_LENGTH + 40.0)
            .min_by(|a, b| vdistance(a.pos, me.pos).total_cmp(&vdistance(b.pos, me.pos)));
        if let Some(o) = opp {
            let away = if me.pos.x >= o.pos.x { 1 } else { -1 };
            out.push(TechPlan {
                tech: Tech::T15,
                plan: plan(n, |_| st(away, false, false, false, 0.0)),
            });
            out.push(TechPlan {
                tech: Tech::T15,
                plan: plan(n, |s| st(away, s == 0 && me.jumps_left > 0, false, false, 0.0)),
            });
            if let Some(a) = ranked_escape_anchors(anchors, hz, hazard_below, 1).first() {
                out.push(TechPlan {
                    tech: Tech::T15,
                    plan: hook_plan(n, a, away, None, n),
                });
            }
        }
    }

    // T17: a free third tee whose hook would drag us across a hazard: step out of its line.
    for t in ctx.threats.iter().take(2) {
        let sep = vdistance(t.pos, me.pos);
        if t.alive && !t.frozen && sep >= 1.0 && sep < *HOOK_LENGTH + 40.0 && hazard_under_line(ctx.col, me.pos, t.pos)
        {
            let away = if me.pos.x >= t.pos.x { 1 } else { -1 };
            out.push(TechPlan {
                tech: Tech::T17,
                plan: plan(n, |_| st(away, false, false, false, 0.0)),
            });
            out.push(TechPlan {
                tech: Tech::T17,
                plan: plan(n, |s| st(away, s == 0 && me.jumps_left > 0, false, false, 0.0)),
            });
            break;
        }
    }

    // T13: no jumps, in the air, ground within reach: steer onto it and refill the jump.
    if air && me.jumps_left == 0 {
        for ox in [0i32, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6] {
            let x = me.pos.x + f64::from(ox) * 32.0;
            let mut land: Option<Vec2> = None;
            for k in 1..=12 {
                let y = me.pos.y + f64::from(k) * 32.0;
                if col_solid_or_hazard(ctx.col, x, y) {
                    if ctx.col.is_solid(x, y) {
                        land = Some(Vec2 { x, y });
                    }
                    break;
                }
            }
            if let Some(p) = land {
                let dir = sign_dir(f64::from(ox), 0.0);
                out.push(TechPlan {
                    tech: Tech::T13,
                    plan: plan(n, |_| st(dir, false, false, false, 0.0)),
                });
                if let Some(a) = anchors
                    .iter()
                    .filter(|a| vdistance(a.point, p) < 96.0)
                    .min_by(|a, b| vdistance(a.point, p).total_cmp(&vdistance(b.point, p)))
                {
                    out.push(TechPlan {
                        tech: Tech::T13,
                        plan: hook_plan(n, a, dir, None, n),
                    });
                }
                break;
            }
        }
    }
}

fn col_solid_or_hazard(col: &impl PlanCollision, x: f64, y: f64) -> bool {
    col.is_solid(x, y) || hazard_at(col, x, y)
}

fn offence<C: PlanCollision>(ctx: &TechCtx<'_, C>, anchors: &[Anchor], caps: &TechCaps, out: &mut Vec<TechPlan>) {
    let (me, v, n) = (ctx.me, ctx.victim, ctx.steps);
    if me.frozen || !me.alive || !v.alive {
        return;
    }
    let sep = vdistance(me.pos, v.pos);
    let toward = if v.pos.x >= me.pos.x { 1 } else { -1 };
    let v_near = hazard_nearness(ctx.field, v.pos.x, v.pos.y);

    if v.frozen {
        // T8: hands off (never thaw a victim that is already going to stay frozen).
        if sep < 96.0 {
            out.push(TechPlan {
                tech: Tech::T8,
                plan: plan(n, |_| st(0, false, false, false, 0.0)),
            });
        }
        // T7: push it deeper with the rope or the body; never step in ourselves.
        if v_near >= 0.3 && sep < *HOOK_LENGTH {
            let away = -toward;
            out.push(TechPlan {
                tech: Tech::T7,
                plan: plan(n, |_| st(away, false, true, false, 0.0)),
            });
            out.push(TechPlan {
                tech: Tech::T7,
                plan: plan(n, |s| st(away, false, s < (2 * n) / 3, false, 0.0)),
            });
            out.push(TechPlan {
                tech: Tech::T7,
                plan: plan(n, |s| st(if s < 3 { toward } else { 0 }, false, false, false, 0.0)),
            });
        }
        return;
    }

    // T1: hook drag through the edge into the hazard -- only when a hazard lies on (or just under)
    // the line the rope drags the victim along; a victim standing between us and the hazard is
    // dragged *away* from it.
    if (50.0..*HOOK_LENGTH - 25.0).contains(&sep) && hazard_under_line(ctx.col, v.pos, me.pos) {
        let away = -toward;
        out.push(TechPlan {
            tech: Tech::T1,
            plan: plan(n, |_| st(away, false, true, false, 0.0)),
        });
        out.push(TechPlan {
            tech: Tech::T1,
            plan: plan(n, |_| st(0, false, true, false, 0.0)),
        });
        out.push(TechPlan {
            tech: Tech::T1,
            plan: plan(n, |s| {
                st(away, s == 1 && me.jumps_left > 0, s < (2 * n) / 3, false, 0.0)
            }),
        });
    }

    // T2/T3: hammer the victim into a wall of freeze (T2) or up into a ceiling of it (T3).
    if sep < 120.0 && v_near >= 0.3 {
        let above = hazard_vertical(ctx.col, v.pos, -1.0, 8).is_some();
        let tech = if above { Tech::T3 } else { Tech::T2 };
        out.push(TechPlan {
            tech,
            plan: plan(n, |_| st(toward, false, false, true, 0.0)),
        });
        out.push(TechPlan {
            tech,
            plan: plan(n, |s| st(if s < 2 { toward } else { 0 }, false, false, s >= 2, 0.0)),
        });
        out.push(TechPlan {
            tech,
            plan: plan(n, |s| st(toward, s == 0 && me.jumps_left > 0, false, s >= 2, 0.0)),
        });
        out.push(TechPlan {
            tech,
            plan: plan(n, |s| st(0, false, false, s >= 1, 0.0)),
        });
        // Stand and swing at once: when the hammer is already out (or the victim is in reach) a plan that
        // waits one step first is re-decided every two ticks and may wait for ever (E-007: T18 0%).
        out.push(TechPlan {
            tech,
            plan: plan(n, |_| st(0, false, false, true, 0.0)),
        });
    }

    // T3 (swing): hook an anchor above us, swing toward the victim, hammer.
    if sep < 220.0 {
        let above: Vec<&Anchor> = anchors
            .iter()
            .filter(|a| js::sin(a.angle) < -0.3 && a.kind != AnchorKind::Floor)
            .collect();
        for a in above.into_iter().take(caps.swing_anchors) {
            out.push(TechPlan {
                tech: Tech::T3,
                plan: plan(n, |s| {
                    if s < 2 {
                        st(toward, false, true, false, abs_aim(a.angle))
                    } else {
                        st(toward, false, true, s >= 3, 0.0)
                    }
                }),
            });
        }
    }

    // T4: pull down a jumper: the victim is in the air over a hazard and within the rope. (Tried: also while
    // it is still rising, vy > -3: 60% instead of 85% on the T4 scenario, E-007.)
    if sep < *HOOK_LENGTH - 30.0
        && !grounded(ctx.col, v.pos)
        && v.vel.y > 1.0
        && hazard_vertical(ctx.col, v.pos, 1.0, 10).is_some()
    {
        let lead = Vec2 {
            x: v.pos.x + v.vel.x * 4.0,
            y: v.pos.y + v.vel.y * 4.0,
        };
        let ang = js::atan2(lead.y - me.pos.y, lead.x - me.pos.x);
        out.push(TechPlan {
            tech: Tech::T4,
            plan: plan(n, |_| st(0, false, true, false, abs_aim(ang))),
        });
        out.push(TechPlan {
            tech: Tech::T4,
            plan: plan(n, |s| st(0, false, s < (2 * n) / 3, false, abs_aim(ang))),
        });
        out.push(TechPlan {
            tech: Tech::T4,
            plan: plan(n, |_| st(0, false, true, false, 0.0)),
        });
    }

    // T1b: leapfrog. The hazard lies beyond the victim, so a plain drag pulls it *away* from the hazard (T1 needs
    // the hazard between us). Hook it, jump over and past it, and the rope that now points back across it
    // pulls it on toward the hazard (hook held to the end, or let go after two thirds).
    if me.jumps_left > 0 && sep < *HOOK_LENGTH - 40.0 && v_near >= 0.4 && (v.pos.y - me.pos.y).abs() < 96.0 {
        let beyond = toward_hazard(ctx.field, v.pos);
        if beyond.x * f64::from(toward) > 0.5 {
            out.push(TechPlan {
                tech: Tech::T1b,
                plan: plan(n, |s| st(toward, s == 0, true, false, 0.0)),
            });
            out.push(TechPlan {
                tech: Tech::T1b,
                plan: plan(n, |s| st(toward, s == 0, s < (2 * n) / 3, false, 0.0)),
            });
        }
    }

    // T42: the direct hook line to the victim is cut by a solid corner, but the hook box is generous
    // (the rope catches a tee whose centre is within 30 px of it): aim a little past the corner, up
    // or down the victim's body, where the line is clear.
    if sep > 60.0 && sep < *HOOK_LENGTH - 20.0 {
        let direct = ctx.col.intersect_line_hook(me.pos, v.pos);
        if direct.collision != 0 {
            let ang0 = js::atan2(v.pos.y - me.pos.y, v.pos.x - me.pos.x);
            // Offsets of 10, 18 and 26 px at the victim's end (the rope catches within 30 px), up or
            // down, smallest first; the first clear one of each sign is one plan.
            'signs: for sign in [1.0, -1.0] {
                for px in [10.0, 18.0, 26.0] {
                    let ang = ang0 + sign * js::atan2(px, sep);
                    let end = Vec2 {
                        x: me.pos.x + js::cos(ang) * sep,
                        y: me.pos.y + js::sin(ang) * sep,
                    };
                    if ctx.col.intersect_line_hook(me.pos, end).collision == 0 {
                        out.push(TechPlan {
                            tech: Tech::T42,
                            plan: plan(n, |_| st(0, false, true, false, abs_aim(ang))),
                        });
                        continue 'signs;
                    }
                }
            }
        }
    }

    // T5: body push at the edge: run in, then stop before we go over ourselves; the push needs
    // speed and keeps pushing while in contact, so the run lasts 2, 3, 5 or all steps (the
    // rollouts tell which one stops in time).
    if sep < 320.0 && v_near >= 0.5 && grounded(ctx.col, me.pos) && grounded(ctx.col, v.pos) {
        for k in [2usize, 3, 5, 9] {
            if k <= n {
                out.push(TechPlan {
                    tech: Tech::T5,
                    plan: plan(n, |s| st(if s < k { toward } else { 0 }, false, false, false, 0.0)),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan_world::LineHit;
    use crate::types::blank_tee_state;

    /// Grid: `#` solid, `f` freeze, `.` air (32 px tiles), everything outside solid.
    struct Grid {
        rows: Vec<Vec<u8>>,
        /// Whether `intersect_line_hook` stops at solid tiles (off for the older tests).
        blocking: bool,
    }

    impl Grid {
        fn new(rows: &[&str]) -> Grid {
            Grid {
                rows: rows.iter().map(|r| r.bytes().collect()).collect(),
                blocking: false,
            }
        }
        fn blocking(rows: &[&str]) -> Grid {
            Grid {
                blocking: true,
                ..Grid::new(rows)
            }
        }
        fn at(&self, x: f64, y: f64) -> u8 {
            let (tx, ty) = (js::floor(x / 32.0) as i32, js::floor(y / 32.0) as i32);
            if ty < 0 || tx < 0 || ty as usize >= self.rows.len() || tx as usize >= self.rows[0].len() {
                return b'#';
            }
            self.rows[ty as usize][tx as usize]
        }
    }

    impl PlanCollision for Grid {
        fn identity(&self) -> u64 {
            0
        }
        fn width(&self) -> i32 {
            self.rows[0].len() as i32
        }
        fn height(&self) -> i32 {
            self.rows.len() as i32
        }
        fn game_tile(&self, tx: i32, ty: i32) -> u8 {
            match self.at(f64::from(tx) * 32.0 + 1.0, f64::from(ty) * 32.0 + 1.0) {
                b'#' => 1,
                b'f' => 9,
                _ => 0,
            }
        }
        fn is_solid(&self, x: f64, y: f64) -> bool {
            self.at(x, y) == b'#'
        }
        fn is_death(&self, _: f64, _: f64) -> bool {
            false
        }
        fn is_freeze(&self, x: f64, y: f64) -> bool {
            self.at(x, y) == b'f'
        }
        fn is_un_freeze(&self, _: f64, _: f64) -> bool {
            false
        }
        fn is_no_hook(&self, _: f64, _: f64) -> bool {
            false
        }
        fn test_box(&self, _: Vec2, _: Vec2) -> bool {
            false
        }
        fn intersect_line(&self, a: Vec2, b: Vec2) -> LineHit {
            self.intersect_line_hook(a, b)
        }
        fn intersect_line_hook(&self, a: Vec2, b: Vec2) -> LineHit {
            if self.blocking {
                let len = vdistance(a, b).max(1.0);
                let steps = (len / 4.0).ceil() as i32;
                let mut before = a;
                for i in 1..=steps {
                    let t = f64::from(i) / f64::from(steps);
                    let p = Vec2 {
                        x: a.x + (b.x - a.x) * t,
                        y: a.y + (b.y - a.y) * t,
                    };
                    if self.is_solid(p.x, p.y) {
                        return LineHit {
                            collision: 1,
                            out_pos: p,
                            out_before_pos: before,
                        };
                    }
                    before = p;
                }
            }
            LineHit {
                collision: 0,
                out_pos: b,
                out_before_pos: a,
            }
        }
        fn has_tele(&self) -> bool {
            false
        }
        fn tele_at(&self, _: f64, _: f64) -> (i32, i32) {
            (0, 0)
        }
        fn tele_outs_for(&self, _: i32) -> Vec<Vec2> {
            Vec::new()
        }
    }

    fn tee(x: f64, y: f64) -> TeeState {
        let mut t = blank_tee_state();
        t.alive = true;
        t.pos = Vec2 { x, y };
        t.jumps_left = 2;
        t
    }

    fn pit() -> Grid {
        Grid::new(&[
            "####################",
            "#..................#",
            "#..................#",
            "#..................#",
            "#..................#",
            "#..................#",
            "#..................#",
            "#####ffffffffffff####",
            "####################",
        ])
    }

    fn ctx<'a>(col: &'a Grid, field: &'a HazardField, me: &'a TeeState, victim: &'a TeeState) -> TechCtx<'a, Grid> {
        TechCtx {
            col,
            field,
            me,
            victim,
            threats: &[],
            hooked_by: None,
            steps: 9,
            aim_at: 0.0,
            me_strong: None,
        }
    }

    fn anchor(angle: f64, dist: f64, kind: AnchorKind, from: Vec2) -> Anchor {
        Anchor {
            point: Vec2 {
                x: from.x + js::cos(angle) * dist,
                y: from.y + js::sin(angle) * dist,
            },
            angle,
            dist,
            kind,
        }
    }

    #[test]
    fn hazard_scans_stop_at_walls_and_find_freeze_below() {
        let col = pit();
        assert_eq!(hazard_vertical(&col, Vec2 { x: 320.0, y: 100.0 }, 1.0, 12), Some(4));
        assert_eq!(hazard_vertical(&col, Vec2 { x: 100.0, y: 100.0 }, 1.0, 12), None);
        assert_eq!(hazard_vertical(&col, Vec2 { x: 320.0, y: 100.0 }, -1.0, 12), None);
    }

    #[test]
    fn t14_panic_hook_needs_no_jumps_air_and_a_hazard_below() {
        let col = pit();
        let field = crate::fields::hazard_field(&col);
        let mut me = tee(320.0, 100.0);
        me.jumps_left = 0;
        let victim = tee(500.0, 100.0);
        let from = me.pos;
        let anchors = [
            anchor(-2.5, 120.0, AnchorKind::Wall, from),
            anchor(-1.5, 100.0, AnchorKind::Ceiling, from),
        ];
        let g = generate(&ctx(&col, &field, &me, &victim), &anchors, &TechCaps::default());
        let t14: Vec<_> = g.defence.iter().filter(|p| p.tech == Tech::T14).collect();
        assert!(!t14.is_empty(), "no panic hook generated");
        // Every panic-hook plan hooks the wall with an absolute aim on the first step.
        for p in &t14 {
            assert_eq!(p.plan.len(), 9);
            assert_eq!(p.plan[0].hook, 1);
            assert!(crate::hybrid::is_abs_aim(p.plan[0].aim));
        }
        // With a jump left it is a freeze jump (T12) instead, not a panic hook.
        me.jumps_left.clone_from(&1);
        let mut me2 = me;
        me2.jumps_left = 1;
        me2.vel.y = 2.0;
        let g2 = generate(&ctx(&col, &field, &me2, &victim), &anchors, &TechCaps::default());
        assert!(g2.defence.iter().all(|p| p.tech != Tech::T14));
        assert!(g2.defence.iter().any(|p| p.tech == Tech::T12));
    }

    #[test]
    fn t12_jumps_sideways_and_only_when_a_jump_is_left() {
        let col = pit();
        let field = crate::fields::hazard_field(&col);
        let mut me = tee(320.0, 120.0);
        me.vel.y = 3.0;
        let victim = tee(500.0, 100.0);
        let g = generate(&ctx(&col, &field, &me, &victim), &[], &TechCaps::default());
        let t12: Vec<_> = g.defence.iter().filter(|p| p.tech == Tech::T12).collect();
        assert!(t12.len() >= 2);
        for p in &t12 {
            assert_eq!(p.plan.iter().filter(|s| s.jump == 1).count(), 1);
            assert_ne!(p.plan[0].dir, 0);
        }
        me.jumps_left = 0;
        let g = generate(&ctx(&col, &field, &me, &victim), &[], &TechCaps::default());
        assert!(g.defence.iter().all(|p| p.tech != Tech::T12));
    }

    #[test]
    fn t9_walks_against_the_pull() {
        let col = pit();
        let field = crate::fields::hazard_field(&col);
        let me = tee(200.0, 100.0);
        let hooker = tee(500.0, 100.0);
        let mut c = ctx(&col, &field, &me, &hooker);
        c.hooked_by = Some(&hooker);
        let g = generate(&c, &[], &TechCaps::default());
        let first = g.defence.iter().find(|p| p.tech == Tech::T9).unwrap();
        assert!(first.plan.iter().all(|s| s.dir == -1), "walk away from the hooker");
        assert!(first.plan.iter().all(|s| s.hook == 0));
    }

    #[test]
    fn t10_hooks_the_tee_below_when_flying_up_into_a_hazard() {
        let col = Grid::new(&[
            "####################",
            "#..ffffffffffff....#",
            "#..................#",
            "#..................#",
            "#..................#",
            "#..................#",
            "#..................#",
            "####################",
        ]);
        let field = crate::fields::hazard_field(&col);
        let mut me = tee(320.0, 100.0);
        me.vel.y = -9.0;
        let below = tee(320.0, 190.0);
        let g = generate(&ctx(&col, &field, &me, &below), &[], &TechCaps::default());
        let t10 = g.defence.iter().find(|p| p.tech == Tech::T10).expect("no T10");
        assert_eq!(t10.plan[0].hook, 1);
        // Aim points down at the tee below.
        let aim = t10.plan[0].aim - crate::hybrid::ABS_AIM;
        assert!(aim > 1.0 && aim < 2.2, "aim {aim}");
    }

    #[test]
    fn offensive_families_fire_only_when_their_trigger_holds() {
        let col = pit();
        let field = crate::fields::hazard_field(&col);
        // Victim on the edge next to the freeze pit, close by: hammer and body push apply.
        let me = tee(100.0, 209.0);
        let v = tee(150.0, 209.0);
        let g = generate(&ctx(&col, &field, &me, &v), &[], &TechCaps::default());
        assert!(g.offence.iter().any(|p| p.tech == Tech::T2));
        assert!(g.offence.iter().any(|p| p.tech == Tech::T5));
        // Far away and calm: nothing offensive is triggered.
        let me = tee(60.0, 60.0);
        let v = tee(560.0, 60.0);
        let g = generate(&ctx(&col, &field, &me, &v), &[], &TechCaps::default());
        assert!(g.offence.iter().all(|p| p.tech != Tech::T2 && p.tech != Tech::T5));
    }

    #[test]
    fn a_frozen_victim_is_never_hammered() {
        let col = pit();
        let field = crate::fields::hazard_field(&col);
        let me = tee(150.0, 200.0);
        let mut v = tee(190.0, 200.0);
        v.frozen = true;
        v.freeze_ticks_left = 200;
        let g = generate(&ctx(&col, &field, &me, &v), &[], &TechCaps::default());
        assert!(g.offence.iter().any(|p| p.tech == Tech::T8));
        assert!(g.offence.iter().all(|p| p.plan.iter().all(|s| s.fire == 0)));
    }

    #[test]
    fn t15_only_on_the_weak_side() {
        let col = pit();
        let field = crate::fields::hazard_field(&col);
        let me = tee(200.0, 100.0);
        let v = tee(400.0, 100.0);
        let mut c = ctx(&col, &field, &me, &v);
        c.me_strong = Some(true);
        assert!(
            generate(&c, &[], &TechCaps::default())
                .defence
                .iter()
                .all(|p| p.tech != Tech::T15)
        );
        c.me_strong = Some(false);
        assert!(
            generate(&c, &[], &TechCaps::default())
                .defence
                .iter()
                .any(|p| p.tech == Tech::T15)
        );
    }

    #[test]
    fn t20_stands_still_on_the_ground_against_a_far_hook_and_t9_walks_against_a_near_one() {
        let col = pit();
        let field = crate::fields::hazard_field(&col);
        // On the floor (row 6 is the last air row above the floor row 7), far hooker: friction.
        let me = tee(100.0, 209.0);
        let far = tee(400.0, 209.0);
        let mut c = ctx(&col, &field, &me, &far);
        c.hooked_by = Some(&far);
        let g = generate(&c, &[], &TechCaps::default());
        let first = g.defence.first().unwrap();
        assert_eq!(first.tech, Tech::T20, "{}", first.tech.name());
        assert!(first.plan.iter().all(|s| s.dir == 0 && s.hook == 0));
        // A near hooker: walk against the pull, no friction stand.
        let near = tee(220.0, 209.0);
        let mut c = ctx(&col, &field, &me, &near);
        c.hooked_by = Some(&near);
        let g = generate(&c, &[], &TechCaps::default());
        assert!(g.defence.iter().all(|p| p.tech != Tech::T20));
        assert!(
            g.defence
                .iter()
                .any(|p| p.tech == Tech::T9 && p.plan.iter().all(|s| s.dir == -1))
        );
    }

    #[test]
    fn t19_anchors_on_the_far_side_from_the_hooker_and_steers_to_the_anchor() {
        let col = pit();
        let field = crate::fields::hazard_field(&col);
        let me = tee(300.0, 120.0);
        let hooker = tee(500.0, 120.0);
        let mut c = ctx(&col, &field, &me, &hooker);
        c.hooked_by = Some(&hooker);
        // One anchor behind us (away from the hooker), one on the hooker's side.
        let behind = anchor(std::f64::consts::PI, 200.0, AnchorKind::Wall, me.pos);
        let ahead = anchor(0.3, 150.0, AnchorKind::Wall, me.pos);
        let g = generate(&c, &[ahead, behind], &TechCaps::default());
        let t19: Vec<_> = g.defence.iter().filter(|p| p.tech == Tech::T19).collect();
        assert!(!t19.is_empty(), "no T19 plan");
        for p in &t19 {
            // Hook the far anchor and steer to it (left).
            assert!(p.plan.iter().all(|s| s.hook == 1 && s.dir == -1));
            let aim = p.plan[0].aim - crate::hybrid::ABS_AIM;
            assert!((aim.abs() - std::f64::consts::PI).abs() < 0.01, "aim {aim}");
        }
    }

    #[test]
    fn t28_lets_go_of_our_own_hook_next_to_a_hazard() {
        let col = pit();
        let field = crate::fields::hazard_field(&col);
        let mut me = tee(320.0, 190.0);
        me.hook_state = HOOK_GRABBED;
        let victim = tee(500.0, 100.0);
        let g = generate(&ctx(&col, &field, &me, &victim), &[], &TechCaps::default());
        let t28: Vec<_> = g.defence.iter().filter(|p| p.tech == Tech::T28).collect();
        assert!(!t28.is_empty());
        assert!(t28.iter().all(|p| p.plan.iter().all(|s| s.hook == 0)));
        // No hook out: nothing to release.
        let me = tee(320.0, 190.0);
        let g = generate(&ctx(&col, &field, &me, &victim), &[], &TechCaps::default());
        assert!(g.defence.iter().all(|p| p.tech != Tech::T28));
    }

    #[test]
    fn t21_counter_steers_when_flying_sideways_toward_a_hazard() {
        let col = Grid::new(&[
            "####################",
            "#..................#",
            "#..................#",
            "#..................#",
            "#..................#",
            "#..................#",
            "#..................#",
            "#######fffffff######",
            "####################",
        ]);
        let field = crate::fields::hazard_field(&col);
        let mut me = tee(140.0, 190.0);
        me.vel.x = 6.7;
        let victim = tee(60.0, 190.0);
        let g = generate(&ctx(&col, &field, &me, &victim), &[], &TechCaps::default());
        let t21 = g.defence.iter().find(|p| p.tech == Tech::T21).expect("no T21");
        assert!(t21.plan.iter().all(|s| s.dir == -1 && s.hook == 0));
        // Flying away from the hazard: no counter-steer.
        me.vel.x = -6.7;
        let g = generate(&ctx(&col, &field, &me, &victim), &[], &TechCaps::default());
        assert!(g.defence.iter().all(|p| p.tech != Tech::T21));
    }

    #[test]
    fn plans_have_the_planner_step_count() {
        let col = pit();
        let field = crate::fields::hazard_field(&col);
        let mut me = tee(320.0, 100.0);
        me.jumps_left = 0;
        let v = tee(380.0, 100.0);
        let from = me.pos;
        let anchors = [anchor(-2.0, 100.0, AnchorKind::Wall, from)];
        let mut c = ctx(&col, &field, &me, &v);
        c.steps = 7;
        let g = generate(&c, &anchors, &TechCaps::default());
        assert!(!g.defence.is_empty() || !g.offence.is_empty());
        for p in g.defence.iter().chain(g.offence.iter()) {
            assert_eq!(p.plan.len(), 7, "{}", p.tech.name());
        }
    }

    #[test]
    fn t29_jumps_before_a_freeze_ahead_when_fast_and_has_a_jump() {
        // A floor with a freeze strip two tiles to the right of us.
        let col = Grid::new(&[
            "####################",
            "#..................#",
            "#..................#",
            "#..................#",
            "###fff##############",
            "####################",
        ]);
        let field = crate::fields::hazard_field(&col);
        let mut me = tee(48.0, 112.0);
        me.vel = Vec2 { x: 4.0, y: 0.0 };
        let v = tee(500.0, 100.0);
        let g = generate(&ctx(&col, &field, &me, &v), &[], &TechCaps::default());
        assert!(g.defence.iter().any(|p| p.tech == Tech::T29), "no T29 plan");
        let t29 = g.defence.iter().find(|p| p.tech == Tech::T29).unwrap();
        assert_eq!(t29.plan[0].jump, 1);
        assert_eq!(t29.plan[0].dir, 1);
        // Slow, or no jumps left: no T29.
        let mut slow = me;
        slow.vel.x = 1.0;
        assert!(
            !generate(&ctx(&col, &field, &slow, &v), &[], &TechCaps::default())
                .defence
                .iter()
                .any(|p| p.tech == Tech::T29)
        );
        let mut spent = me;
        spent.jumps_left = 0;
        assert!(
            !generate(&ctx(&col, &field, &spent, &v), &[], &TechCaps::default())
                .defence
                .iter()
                .any(|p| p.tech == Tech::T29)
        );
    }

    #[test]
    fn t42_aims_past_the_corner_that_blocks_the_direct_hook_line() {
        // A solid stub sits on the line between us and the victim, two tiles in front of it.
        let col = Grid::blocking(&[
            "####################",
            "#..................#",
            "#..................#",
            "#..................#",
            "#...........#......#",
            "####################",
        ]);
        let field = crate::fields::hazard_field(&col);
        // Us at tile (3, 4), the victim at tile (14, 4): the stub at x = 12 blocks the level line.
        let me = tee(3.0 * 32.0 + 16.0, 4.0 * 32.0 + 16.0);
        let v = tee(14.0 * 32.0 + 16.0, 4.0 * 32.0 + 16.0);
        let g = generate(&ctx(&col, &field, &me, &v), &[], &TechCaps::default());
        let t42: Vec<_> = g.offence.iter().filter(|p| p.tech == Tech::T42).collect();
        assert!(!t42.is_empty(), "an aim that clears the stub");
        let aim = t42[0].plan[0].aim - crate::hybrid::ABS_AIM;
        assert!(aim.abs() > 0.01 && aim.abs() < 0.3, "aim offset {aim}");
        assert_eq!(t42[0].plan[0].hook, 1);
        // With a clear line there is nothing to fix.
        let clear = Grid::blocking(&[
            "####################",
            "#..................#",
            "#..................#",
            "#..................#",
            "#..................#",
            "####################",
        ]);
        let field2 = crate::fields::hazard_field(&clear);
        let g = generate(&ctx(&clear, &field2, &me, &v), &[], &TechCaps::default());
        assert!(!g.offence.iter().any(|p| p.tech == Tech::T42));
    }
}
