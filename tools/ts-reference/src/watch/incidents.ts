import { HOOK_FLYING, HOOK_GRABBED } from "../core/characterCore.ts";
import { TUNING } from "../core/tuning.ts";
import type { RecFrame, Recording } from "./recording.ts";

export type Incident = {
  kind: string;

  tick: number;
  from: number;
  to: number;

  severity: number;
  note: string;
};

const HOOK_LENGTH = TUNING.hookLength;

const SWING_REACH_PX = 96;

const HUMAN_REHOOK_TICKS = 10;
const HUMAN_HOLD_TICKS = 17;

const WALK_FRAMES = 12;

const TELEPORT_PX = 96;
const TELEPORT_SLACK_PX = 8;

function teeOf(f: RecFrame, id: number): RecFrame["tees"][number] | undefined {
  return f.tees.find((t) => t.id === id);
}

function foeOf(f: RecFrame, selfId: number, free = false): RecFrame["tees"][number] | undefined {
  const me = teeOf(f, selfId);
  if (me === undefined) return undefined;
  let best: RecFrame["tees"][number] | undefined;
  let bestD = Infinity;
  for (const t of f.tees) {
    if (t.id === selfId || !t.alive || (free && t.frozen)) continue;
    const d = Math.hypot(t.x - me.x, t.y - me.y);
    if (d < bestD) {
      bestD = d;
      best = t;
    }
  }
  return best;
}

function dist(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export function findIncidents(rec: Recording, opts: { selfId?: number; context?: number } = {}): Incident[] {
  const selfId = opts.selfId ?? rec.selfId ?? 0;
  const context = opts.context ?? 40;
  const F = rec.frames;
  const out: Incident[] = [];
  const clip = (tick: number, severity: number, kind: string, note: string): void => {
    out.push({ kind, tick, from: tick - context, to: tick + context, severity, note });
  };

  const recordsWalks = F.some((f) => f.walk !== undefined);
  const plans = F.some((f) => f.plan !== undefined);
  const walking = (i: number): boolean => {
    if (recordsWalks) return F[i].walk !== undefined || F[i - 1]?.walk !== undefined;
    if (!plans) return false;
    for (let j = Math.max(0, i - WALK_FRAMES); j <= i; j++) if (F[j].plan !== undefined) return false;
    return true;
  };

  for (let i = 1; i < F.length; i++) {
    const me = teeOf(F[i], selfId);
    const was = teeOf(F[i - 1], selfId);
    const foe = foeOf(F[i], selfId);
    if (me === undefined || was === undefined || !me.frozen || was.frozen) continue;

    const pushed = F[i].events.some((e) => e.includes('"hammerHit"') && e.includes(`"to":${selfId}`));
    if (pushed) continue;

    const moved = dist(me, was);
    const ticks = Math.max(1, F[i].tick - F[i - 1].tick);
    const speed = Math.max(Math.hypot(was.vx, was.vy), Math.hypot(me.vx, me.vy));
    if (moved > speed * ticks + TELEPORT_PX) continue;
    if (moved > speed * ticks + TELEPORT_SLACK_PX && Math.hypot(me.vx, me.vy) <= 1) continue;
    if (moved < 1 && Math.hypot(was.vx, was.vy) < 0.1) continue;

    if (F[i].plannedFreeze === true || F[i - 1].plannedFreeze === true) continue;
    let held = 0;
    for (let j = i; j < F.length; j++) {
      const t = teeOf(F[j], selfId);
      if (t === undefined || !t.frozen) break;
      held = F[j].tick - F[i].tick;
    }
    const foeFrozen = foe !== undefined && foe.frozen;

    let closing = false;
    const back = Math.max(0, i - 25);
    const before = teeOf(F[back], selfId);
    const chased = foeOf(F[i], selfId, true);
    const chasedBefore = foeOf(F[back], selfId, true);
    if (before !== undefined && chased !== undefined && chasedBefore !== undefined && chased.id === chasedBefore.id) {
      closing = dist(before, chasedBefore) - dist(me, chased) > 40;
    }
    const kind = walking(i) ? "goto-into-freeze" : closing ? "chased-into-freeze" : "self-freeze";

    const hookedByFoe = F[i].tees.some((t) => t.id !== selfId && t.hookedPlayer === selfId);
    const rising = was.vy < -0.5;
    const falling = was.vy > 3;
    const how = hookedByFoe ? "on their hook" : rising ? "jumped into it" : falling ? "fell into it" : "walked into it";
    clip(
      F[i].tick,
      held + (foeFrozen ? 0 : 40) + (hookedByFoe ? -30 : 0),
      kind,
      `froze itself for ${held} ticks, ${how}${closing ? ", while closing on the opponent" : ""}; opponent was ${foeFrozen ? "also frozen" : "free"}`,
    );
  }

  let outSince = -1;
  let grabbedTee = false;
  let inReachAtThrow = false;
  let teeGrabStart = -1;
  let emptyReturnTick = -1;

  let chanceTicks = 0;

  let wasOut = false;
  for (let i = 0; i < F.length; i++) {
    const me = teeOf(F[i], selfId);
    const foe = foeOf(F[i], selfId);
    if (me === undefined) continue;
    const out = me.hookState === HOOK_FLYING || me.hookState === HOOK_GRABBED;
    const thrown = out && !wasOut;
    wasOut = out;
    if (emptyReturnTick >= 0 && i > 0 && !out) {
      const canThrow = !me.frozen && foe !== undefined && foe.alive && !foe.frozen && dist(me, foe) <= HOOK_LENGTH;
      if (canThrow) chanceTicks += F[i].tick - F[i - 1].tick;
    }
    if (thrown && outSince < 0) {
      outSince = F[i].tick;
      grabbedTee = false;
      inReachAtThrow = foe !== undefined && dist(me, foe) <= HOOK_LENGTH;
      if (emptyReturnTick >= 0) {
        if (chanceTicks > 3 * HUMAN_REHOOK_TICKS) {
          clip(
            emptyReturnTick,
            chanceTicks - HUMAN_REHOOK_TICKS,
            "slow-rehook",
            `${chanceTicks} ticks with the rope free, both of us free and them in reach before the next hook went out; players take ${HUMAN_REHOOK_TICKS}`,
          );
        }
        emptyReturnTick = -1;
        chanceTicks = 0;
      }
    }
    if (me.hookState === HOOK_GRABBED && me.hookedPlayer >= 0 && teeGrabStart < 0) {
      teeGrabStart = F[i].tick;
      grabbedTee = true;
    }
    if (teeGrabStart >= 0 && me.hookedPlayer < 0) {
      const held = F[i].tick - teeGrabStart;

      if (held < HUMAN_HOLD_TICKS / 2 && foe !== undefined && !foe.frozen) {
        clip(teeGrabStart, HUMAN_HOLD_TICKS - held, "short-hold", `let a tee off the hook after ${held} ticks; players hold ${HUMAN_HOLD_TICKS}`);
      }
      teeGrabStart = -1;
    }
    if (outSince >= 0 && !out) {
      if (!grabbedTee && inReachAtThrow) {
        emptyReturnTick = F[i].tick;
        chanceTicks = 0;
      }
      outSince = -1;
    }
  }

  {
    const THAW_LOOK_TICKS = 30;
    for (let i = 0; i < F.length; i++) {
      const me = teeOf(F[i], selfId);
      const foe = foeOf(F[i], selfId);
      const inp = F[i].inputs.find((x) => x.id === selfId);
      if (me === undefined || foe === undefined || inp === undefined) continue;
      if ((inp.fire & 1) === 0 || !foe.frozen || me.frozen) continue;
      if (dist(me, foe) > SWING_REACH_PX) continue;

      const before = F[i - 1]?.inputs.find((x) => x.id === selfId);
      if (before !== undefined && (before.fire & 1) !== 0) continue;
      let freeAfter = false;
      let stillFrozen = false;
      for (let j = i + 1; j < F.length && F[j].tick - F[i].tick <= THAW_LOOK_TICKS; j++) {
        const later = teeOf(F[j], selfId === 0 ? foe.id : foe.id);
        if (later === undefined) continue;
        if (later.frozen) stillFrozen = true;
        else if (F[j].tick - F[i].tick >= THAW_LOOK_TICKS / 2) freeAfter = true;
      }
      if (freeAfter && !stillFrozen) {
        clip(F[i].tick, 60, "thawed-the-enemy", `hammered a FROZEN opponent at ${dist(me, foe).toFixed(0)}px and they were free ${THAW_LOOK_TICKS} ticks later -- a hammer unfreezes its target`);
      }
    }
  }

  for (const f of F) {
    const me = teeOf(f, selfId);
    const foe = foeOf(f, selfId);
    if (me === undefined) continue;
    for (const e of f.events) {
      if (!e.includes('"hammerFire"') || !e.includes(`"from":${selfId}`) || !e.includes('"hits":0')) continue;
      const d = foe === undefined ? Infinity : dist(me, foe);
      if (d > SWING_REACH_PX) clip(f.tick, 5, "swing-at-air", `swung with the opponent ${Number.isFinite(d) ? `${d.toFixed(0)}px` : "nowhere"} away, hammer reaches ${SWING_REACH_PX}px`);
    }
  }

  let stuckFrom = -1;
  for (let i = 0; i < F.length; i++) {
    const me = teeOf(F[i], selfId);
    const inp = F[i].inputs.find((x) => x.id === selfId);
    const stuck = me !== undefined && inp !== undefined && inp.direction !== 0 && Math.abs(me.vx) < 0.2 && !me.frozen && me.alive;
    if (stuck && stuckFrom < 0) stuckFrom = F[i].tick;
    if (!stuck && stuckFrom >= 0) {
      const held = F[i].tick - stuckFrom;
      if (held >= 20) clip(stuckFrom, held, "wall-grind", `held a direction for ${held} ticks without moving`);
      stuckFrom = -1;
    }
  }

  {
    const flips: number[] = [];
    let last: number | null = null;
    for (const f of F) {
      const inp = f.inputs.find((x) => x.id === selfId);
      if (inp === undefined) continue;
      if (last !== null && inp.direction !== last) flips.push(f.tick);
      last = inp.direction;
    }
    for (let i = 0; i + 5 < flips.length; i++) {
      const span = flips[i + 5] - flips[i];
      if (span <= 25) {
        clip(flips[i], 6 + (25 - span), "jitter", `6 direction changes in ${span} ticks`);
        i += 5;
      }
    }
  }

  for (const f of F) {
    for (const e of f.events) {
      if (e.includes('"death"') && e.includes(`"id":${selfId}`)) clip(f.tick, 150, "death", "died");
    }
  }
  for (let i = 1; i < F.length; i++) {
    const me = teeOf(F[i], selfId);
    const was = teeOf(F[i - 1], selfId);
    const already = out.some((o) => o.kind === "death" && Math.abs(o.tick - F[i].tick) < 10);
    if (!already && was?.alive && me !== undefined && !me.alive) clip(F[i].tick, 150, "death", "died");
  }

  out.sort((a, b) => b.severity - a.severity || a.tick - b.tick);
  return out;
}

export function mergeOverlapping(incidents: Incident[]): Incident[] {
  const kept: Incident[] = [];
  for (const inc of incidents) {
    const clash = kept.find((k) => Math.abs(k.tick - inc.tick) < 25);
    if (clash === undefined) kept.push(inc);
    else if (clash.kind !== inc.kind && !clash.note.includes(`also ${inc.kind}`)) clash.note += `; also ${inc.kind}`;
  }
  return kept;
}

export function summarise(incidents: Incident[]): { kind: string; count: number; worst: number }[] {
  const by = new Map<string, { kind: string; count: number; worst: number }>();
  for (const i of incidents) {
    const row = by.get(i.kind) ?? { kind: i.kind, count: 0, worst: 0 };
    row.count++;
    row.worst = Math.max(row.worst, i.severity);
    by.set(i.kind, row);
  }
  return [...by.values()].sort((a, b) => b.count * b.worst - a.count * a.worst);
}
