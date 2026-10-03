import { EN } from "./i18n-en.ts";

export { EN };

export type Lang = "ru" | "en";

export function isLang(v: unknown): v is Lang {
  return v === "ru" || v === "en";
}

export function systemLang(env: NodeJS.ProcessEnv = process.env, locale: string = Intl.DateTimeFormat().resolvedOptions().locale): Lang {
  for (const v of [env.LC_ALL, env.LC_MESSAGES, env.LANG]) {
    if (typeof v !== "string" || v === "" || /^(C|POSIX)([._@]|$)/.test(v)) continue;
    return /^ru/i.test(v) ? "ru" : "en";
  }
  return /^ru/i.test(locale) ? "ru" : "en";
}

export function detectLang(env: NodeJS.ProcessEnv = process.env, saved?: unknown): Lang {
  if (isLang(env.DDNET_AI_LANG)) return env.DDNET_AI_LANG;
  if (isLang(saved)) return saved;
  return systemLang(env);
}

type Params = Record<string, unknown>;
export type Translator = { t: (s: string, p?: Params) => string; tr: (s: string, depth?: number) => string };

export function makeT(dict: Record<string, string>, lang: string): Translator {
  const en = lang === "en";
  const has = (s: string): boolean => Object.prototype.hasOwnProperty.call(dict, s);
  const fill = (s: string, p?: Params): string =>
    p === undefined ? s : s.replace(/\{(\w+)\}/g, (m: string, k: string) => (p[k] === undefined ? m : String(p[k])));
  const t = (s: string, p?: Params): string => fill(en && has(s) ? dict[s] : s, p);
  let pats: { re: RegExp; keys: string[]; out: string; lit: number }[] | null = null;
  const tr = (s: string, depth = 0): string => {
    if (!en || typeof s !== "string" || !/[А-Яа-яЁё]/.test(s)) return s;
    if (has(s)) return dict[s];
    if (pats === null) {
      pats = [];
      for (const k of Object.keys(dict)) {
        if (!/\{\w+\}/.test(k)) continue;
        const keys: string[] = [];
        const src = k
          .split(/(\{\w+\})/)
          .map((part: string) => {
            const m = /^\{(\w+)\}$/.exec(part);
            if (m !== null) {
              keys.push(m[1]);
              return "([\\s\\S]*?)";
            }
            return part.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
          })
          .join("");
        pats.push({ re: new RegExp("^" + src + "$"), keys, out: dict[k], lit: k.replace(/\{\w+\}/g, "").length });
      }

      pats.sort((a, b) => b.lit - a.lit);
    }
    for (const p of pats) {
      const m = p.re.exec(s);
      if (m === null) continue;
      const vals: Record<string, string> = {};
      p.keys.forEach((k: string, i: number) => {
        vals[k] = depth < 2 ? tr(m[i + 1], depth + 1) : m[i + 1];
      });
      return fill(p.out, vals);
    }
    return s;
  };
  return { t, tr };
}

let current: Lang = "ru";
let cached: Translator | null = null;

export function setLang(l: Lang): void {
  if (l === current) return;
  current = l;
  cached = null;
}

export function getLang(): Lang {
  return current;
}

function translator(): Translator {
  if (cached === null) cached = makeT(EN, current);
  return cached;
}

export function t(s: string, p?: Params): string {
  return translator().t(s, p);
}

export function tr(s: string): string {
  return translator().tr(s);
}
