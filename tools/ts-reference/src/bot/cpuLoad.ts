export const SNAPSHOT_PERIOD_MS = 40;

export type CpuSettings = {

  budgetMs: number;
  hardMs: number;

  commit: number;

  explain: boolean;

  navMs: number;

  reachChecks: number;
};

export const LOW_CPU: Readonly<CpuSettings> = { budgetMs: 6, hardMs: 11, commit: 2, explain: false, navMs: 10, reachChecks: 2 };

export const STRONG_WB = { population: 40, iterations: 3, budgetMs: 30, hardMs: 36 } as const;

export function lowCpuWanted(v: unknown): boolean {
  return v === true || v === 1 || (typeof v === "string" && ["on", "true", "yes", "1"].includes(v.trim().toLowerCase()));
}

export const strongWanted = lowCpuWanted;

const WINDOW_MS = 1000;

const WARMUP_MS = 60_000;

const WAITING_GAP_MS = 16;

const BUSY_WORK_MS = SNAPSHOT_PERIOD_MS;

const BEHIND_SKIPS = 2;

export const HINT_AFTER_MS = 5000;

const CLEAR_AFTER_MS = 30_000;

const STALE_MS = 5000;

export type LagSummary = {

  hint: boolean;

  workMs: number;

  skipped: number;

  behindMs: number;
};

export class LagWatch {
  private windowStart = -1;
  private work = 0;
  private processed = 0;
  private busySkips = 0;
  private prevEnd = -Infinity;
  private prevWork = 0;
  private runStart = -1;
  private goodSince = -1;

  private firstMs = -1;
  private hintValue = false;
  private last: LagSummary = { hint: false, workMs: 0, skipped: 0, behindMs: 0 };

  private readonly warmupMs: number;

  constructor(warmupMs = WARMUP_MS) {
    this.warmupMs = warmupMs;
  }

  get hint(): boolean {
    return this.hintValue;
  }

  summary(): LagSummary {
    return { ...this.last, hint: this.hintValue };
  }

  note(firstReadMs: number, arrived: number, startMs: number, endMs: number): string | null {
    if (this.windowStart < 0 || startMs - this.windowStart > STALE_MS) {
      this.startWindow(startMs);
      this.runStart = -1;
      this.firstMs = startMs;
    }
    const waiting = firstReadMs - this.prevEnd <= WAITING_GAP_MS && this.prevWork >= BUSY_WORK_MS;
    if (arrived > 1 && waiting) this.busySkips += arrived - 1;
    this.prevEnd = endMs;
    this.prevWork = endMs - startMs;
    this.work += endMs - startMs;
    this.processed++;
    if (endMs - this.windowStart < WINDOW_MS) return null;
    return this.judge(endMs);
  }

  private startWindow(nowMs: number): void {
    this.windowStart = nowMs;
    this.work = 0;
    this.processed = 0;
    this.busySkips = 0;
  }

  private judge(nowMs: number): string | null {
    const workMs = this.processed > 0 ? this.work / this.processed : 0;
    const skipped = this.busySkips;
    const behind = skipped >= BEHIND_SKIPS && nowMs - this.firstMs >= this.warmupMs;
    if (behind) {
      if (this.runStart < 0) this.runStart = this.windowStart;
      this.goodSince = -1;
    } else {
      this.runStart = -1;
      if (this.goodSince < 0) this.goodSince = this.windowStart;
    }
    const behindMs = this.runStart < 0 ? 0 : nowMs - this.runStart;
    this.last = { hint: this.hintValue, workMs, skipped, behindMs };
    this.startWindow(nowMs);
    if (!this.hintValue && behindMs > HINT_AFTER_MS) {
      this.hintValue = true;
      return `this PC is not keeping up: a snapshot takes ${workMs.toFixed(0)} ms of the ${SNAPSHOT_PERIOD_MS} there are and ${skipped} a second are skipped; the mode for a weak PC may help (!low on)`;
    }
    if (this.hintValue && this.goodSince >= 0 && nowMs - this.goodSince >= CLEAR_AFTER_MS) this.hintValue = false;
    return null;
  }
}
