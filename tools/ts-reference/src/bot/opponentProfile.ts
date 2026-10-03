import type { TeeState } from "../core/types.ts";
import { vdistance } from "../core/vmath.ts";
import { HOOK_FLYING, HOOK_GRABBED } from "../core/characterCore.ts";

const DECISIONS_PER_SEC = 25;

const FAST_HALFLIFE = 6 * DECISIONS_PER_SEC;

const SLOW_HALFLIFE = 30 * DECISIONS_PER_SEC;

const decay = (halfLife: number): number => Math.pow(0.5, 1 / halfLife);

class Rate {
  private hits = 0;
  private total = 0;
  private readonly k: number;
  private readonly minSamples: number;

  constructor(halfLife: number, minSamples = 8) {
    this.k = decay(halfLife);
    this.minSamples = minSamples;
  }

  observe(hit: boolean): void {
    this.hits = this.hits * this.k + (hit ? 1 : 0);
    this.total = this.total * this.k + 1;
  }

  idle(): void {
    this.hits *= this.k;
    this.total *= this.k;
  }

  value(prior: number): number {
    if (this.total < this.minSamples) return prior;
    return this.hits / this.total;
  }

  get confidence(): number {
    return Math.min(1, this.total / (this.minSamples * 2));
  }

  reset(): void {
    this.hits = 0;
    this.total = 0;
  }
}

export type OpponentRead = {

  aggression: number;

  hookOpensFirst: number;

  hookSuccess: number;

  outOfJumps: number;

  confidence: number;
};

const DEFAULT_READ: OpponentRead = {
  aggression: 0.5,
  hookOpensFirst: 0.5,
  hookSuccess: 0.57,
  outOfJumps: 0.3,
  confidence: 0,
};

export class OpponentProfile {
  private readonly aggression = new Rate(FAST_HALFLIFE);
  private readonly hookFirst = new Rate(SLOW_HALFLIFE, 4);
  private readonly hookSuccess = new Rate(SLOW_HALFLIFE, 4);
  private readonly outOfJumps = new Rate(FAST_HALFLIFE);

  private prevDist = -1;
  private theirHookFlying = false;
  private theirHookGrabbed = false;
  private ourHookFlying = false;

  private theyOpened: boolean | null = null;
  private engaged = false;

  observe(me: TeeState, them: TeeState, hookRangePx: number): void {
    if (!me.alive || !them.alive) {
      this.endEngagement();
      return;
    }
    const dist = vdistance(me.pos, them.pos);

    if (!them.frozen) {
      if (this.prevDist >= 0) this.aggression.observe(dist < this.prevDist - 1);
      this.outOfJumps.observe(them.jumpsLeft === 0);
    } else {
      this.aggression.idle();
      this.outOfJumps.idle();
    }
    this.prevDist = dist;

    const inRange = dist <= hookRangePx;
    if (inRange && !this.engaged) {
      this.engaged = true;
      this.theyOpened = null;
    }

    const theirFlyingNow = them.hookState === HOOK_FLYING;
    if (theirFlyingNow && !this.theirHookFlying) {
      this.theirHookFlying = true;
      this.theirHookGrabbed = false;
      if (this.engaged && this.theyOpened === null) this.theyOpened = true;
    }
    if (them.hookState === HOOK_GRABBED) this.theirHookGrabbed = true;
    if (this.theirHookFlying && them.hookState <= 0) {
      this.hookSuccess.observe(this.theirHookGrabbed);
      this.theirHookFlying = false;
      this.theirHookGrabbed = false;
    }

    const ourFlyingNow = me.hookState === HOOK_FLYING;
    if (ourFlyingNow && !this.ourHookFlying) {
      this.ourHookFlying = true;
      if (this.engaged && this.theyOpened === null) this.theyOpened = false;
    }
    if (!ourFlyingNow) this.ourHookFlying = false;

    if (!inRange && this.engaged) this.endEngagement();
  }

  private endEngagement(): void {
    if (this.engaged && this.theyOpened !== null) this.hookFirst.observe(this.theyOpened);
    this.engaged = false;
    this.theyOpened = null;
    this.prevDist = -1;
  }

  read(): OpponentRead {
    const blend = (rate: Rate, prior: number): number => prior + rate.confidence * (rate.value(prior) - prior);
    return {
      aggression: blend(this.aggression, DEFAULT_READ.aggression),
      hookOpensFirst: blend(this.hookFirst, DEFAULT_READ.hookOpensFirst),
      hookSuccess: blend(this.hookSuccess, DEFAULT_READ.hookSuccess),
      outOfJumps: blend(this.outOfJumps, DEFAULT_READ.outOfJumps),

      confidence: Math.max(
        this.aggression.confidence,
        this.hookFirst.confidence,
        this.hookSuccess.confidence,
        this.outOfJumps.confidence,
      ),
    };
  }

  reset(): void {
    this.aggression.reset();
    this.hookFirst.reset();
    this.hookSuccess.reset();
    this.outOfJumps.reset();
    this.prevDist = -1;
    this.theirHookFlying = false;
    this.theirHookGrabbed = false;
    this.ourHookFlying = false;
    this.engaged = false;
    this.theyOpened = null;
  }
}
