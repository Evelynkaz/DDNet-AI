import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

export type AutoChatRule = { on: boolean; match: string; reply: string };
export type AutoChatConfig = {
  periodic: { on: boolean; text: string; everySec: number };
  mention: { on: boolean; reply: string };
  keywords: AutoChatRule[];
};

export const AUTOCHAT_MAX_TEXT = 200;
export const AUTOCHAT_MAX_RULES = 12;

export const AUTOCHAT_MIN_PERIOD_S = 15;

export const AUTOCHAT_RULE_COOLDOWN_MS = 5000;

export const AUTOCHAT_SERVER_COOLDOWN_MS = 15000;

export const AUTOCHAT_MENTION_PER_PLAYER_MS = 30000;

const PENDING_MS = 6000;

export const AUTOCHAT_DEFAULTS: AutoChatConfig = {
  periodic: { on: false, text: "", everySec: 60 },
  mention: { on: false, reply: "" },
  keywords: [],
};

const str = (v: unknown, max = AUTOCHAT_MAX_TEXT): string => (typeof v === "string" ? v.replace(/[\r\n\t]+/g, " ").trim().slice(0, max) : "");

export function sanitizeAutoChat(raw: unknown): AutoChatConfig {
  const o = raw !== null && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
  const p = o.periodic !== null && typeof o.periodic === "object" ? (o.periodic as Record<string, unknown>) : {};
  const m = o.mention !== null && typeof o.mention === "object" ? (o.mention as Record<string, unknown>) : {};
  const every = Number(p.everySec);
  const rules = Array.isArray(o.keywords) ? o.keywords : [];
  return {
    periodic: {
      on: p.on === true,
      text: str(p.text),
      everySec: Number.isFinite(every) ? Math.min(3600, Math.max(AUTOCHAT_MIN_PERIOD_S, Math.round(every))) : AUTOCHAT_DEFAULTS.periodic.everySec,
    },
    mention: { on: m.on === true, reply: str(m.reply) },
    keywords: rules
      .slice(0, AUTOCHAT_MAX_RULES)
      .map((r) => (r !== null && typeof r === "object" ? (r as Record<string, unknown>) : {}))
      .map((r) => ({ on: r.on !== false, match: str(r.match, 60), reply: str(r.reply) }))
      .filter((r) => r.match !== "" || r.reply !== ""),
  };
}

function words(match: string): string[] {
  return match
    .split(/[|,]/)
    .map((w) => w.trim().toLowerCase())
    .filter((w) => w !== "");
}

function fill(reply: string, name: string, me: string): string {
  let safeName = name.replace(/^[\s/\\]+/u, "");

  if (reply.trimStart().startsWith("/") && /[\s"]/u.test(safeName)) safeName = `"${safeName.replace(/["\\]/g, "\\$&")}"`;
  let out = reply.replace(/\{name\}/g, () => safeName).replace(/\{me\}/g, () => me);
  if (!reply.trimStart().startsWith("/")) out = out.replace(/^[\s/\\]+/u, "");
  return out.slice(0, AUTOCHAT_MAX_TEXT);
}

function invitedBy(text: string): string {
  const name = /'\/\S+ (.{1,40}?)(?: \d+)*'(?=[\s,.!;:?]|$)/u.exec(text)?.[1] ?? "";
  return name !== "" && text.includes(`'${name}'`) ? name : "";
}

export class AutoChat {
  private cfg: AutoChatConfig = sanitizeAutoChat(AUTOCHAT_DEFAULTS);
  private lastPeriodicMs = -Infinity;
  private lastRuleMs = new Map<string, number>();
  private pending: { text: string; until: number } | null = null;
  private readonly file: string | undefined;

  constructor(file?: string) {
    this.file = file;
    if (file === undefined) return;
    try {
      this.cfg = sanitizeAutoChat(JSON.parse(readFileSync(file, "utf8")));
    } catch {

    }
  }

  config(): AutoChatConfig {
    return this.cfg;
  }

  set(raw: unknown, nowMs = Date.now()): AutoChatConfig {
    this.cfg = sanitizeAutoChat(raw);
    this.lastPeriodicMs = nowMs;
    if (this.file !== undefined) {
      try {
        mkdirSync(dirname(this.file), { recursive: true });
        writeFileSync(this.file, JSON.stringify(this.cfg, null, 1));
      } catch {

      }
    }
    return this.cfg;
  }

  onChat(line: { from: string; text: string; server: boolean; me: string }, nowMs = Date.now()): string | null {
    const text = line.text.toLowerCase();
    for (let i = 0; i < this.cfg.keywords.length; i++) {
      const r = this.cfg.keywords[i];
      if (!r.on || r.reply === "") continue;
      const ws = words(r.match);
      if (ws.length === 0 || !ws.some((w) => text.includes(w))) continue;
      const from = line.server ? invitedBy(line.text) : line.from;

      if (r.reply.includes("{name}") && from.replace(/^[\s/\\]+/u, "").trim() === "") continue;
      if (!this.ready(`k${i}:${r.match}`, nowMs, line.server ? AUTOCHAT_SERVER_COOLDOWN_MS : AUTOCHAT_RULE_COOLDOWN_MS)) return null;
      return fill(r.reply, from, line.me);
    }
    if (!line.server && this.cfg.mention.on && this.cfg.mention.reply !== "" && line.me !== "" && text.includes(line.me.toLowerCase())) {
      const key = `mention:${line.from.toLowerCase()}`;
      if (nowMs - (this.lastRuleMs.get(key) ?? -Infinity) < AUTOCHAT_MENTION_PER_PLAYER_MS) return null;
      if (!this.ready("mention", nowMs, AUTOCHAT_RULE_COOLDOWN_MS)) return null;
      this.lastRuleMs.set(key, nowMs);
      return fill(this.cfg.mention.reply, line.from, line.me);
    }
    return null;
  }

  due(me: string, nowMs = Date.now()): string | null {
    if (this.pending !== null) {
      if (nowMs <= this.pending.until) return this.pending.text;
      this.pending = null;
    }
    const p = this.cfg.periodic;
    if (!p.on || p.text === "") return null;
    if (nowMs - this.lastPeriodicMs < p.everySec * 1000) return null;
    this.lastPeriodicMs = nowMs;
    return fill(p.text, "", me);
  }

  sent(text: string, ok: boolean, nowMs = Date.now()): void {
    if (ok) {
      if (this.pending !== null && this.pending.text === text) this.pending = null;
      return;
    }
    if (this.pending === null) this.pending = { text, until: nowMs + PENDING_MS };
  }

  private ready(key: string, nowMs: number, cooldownMs: number): boolean {
    const last = this.lastRuleMs.get(key) ?? -Infinity;
    if (nowMs - last < cooldownMs) return false;
    this.lastRuleMs.set(key, nowMs);
    return true;
  }
}
