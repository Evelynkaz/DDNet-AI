import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

export const MASTERS = [
  "https://master1.ddnet.org/ddnet/15/servers.json",
  "https://master2.ddnet.org/ddnet/15/servers.json",
  "https://master3.ddnet.org/ddnet/15/servers.json",
  "https://master4.ddnet.org/ddnet/15/servers.json",
];

export type ServerRow = {
  address: string;
  name: string;
  map: string;
  gameType: string;
  location: string;
  passworded: boolean;

  players: number;
  clients: number;
  maxClients: number;
};

export function isAuto(server: unknown): boolean {
  return typeof server === "string" && /^(auto|авто)?$/i.test(server.trim());
}

const ADDR_RE = /^(tw-0\.6\+udp|tw-0\.7\+udp|ddnet\+udp|udp):\/\/(\d{1,3}(?:\.\d{1,3}){3}):(\d{1,5})$/;

function pickAddress(addresses: unknown): string | null {
  if (!Array.isArray(addresses)) return null;
  let fallback: string | null = null;
  for (const a of addresses) {
    if (typeof a !== "string") continue;
    const m = ADDR_RE.exec(a.trim());
    if (m === null) continue;
    const port = Number(m[3]);
    if (m[2].split(".").some((o) => Number(o) > 255) || port < 1 || port > 65535) continue;
    const addr = `${m[2]}:${port}`;
    if (m[1] === "tw-0.6+udp") return addr;
    fallback ??= addr;
  }
  return fallback;
}

const str = (v: unknown, max = 128): string => (typeof v === "string" ? v.slice(0, max) : "");

export function parseMaster(json: unknown): ServerRow[] {
  const list = json !== null && typeof json === "object" && Array.isArray((json as { servers?: unknown }).servers) ? (json as { servers: unknown[] }).servers : [];
  const seen = new Set<string>();
  const rows: ServerRow[] = [];
  for (const s of list) {
    if (s === null || typeof s !== "object") continue;
    const e = s as { addresses?: unknown; location?: unknown; info?: Record<string, unknown> };
    const address = pickAddress(e.addresses);
    if (address === null || seen.has(address)) continue;
    seen.add(address);
    const info = e.info !== null && typeof e.info === "object" ? e.info : {};
    const clients = Array.isArray(info.clients) ? (info.clients as unknown[]).filter((c): c is Record<string, unknown> => c !== null && typeof c === "object") : [];
    const map = info.map !== null && typeof info.map === "object" ? (info.map as { name?: unknown }).name : info.map;
    rows.push({
      address,
      name: str(info.name) || address,
      map: str(map, 64),
      gameType: str(info.game_type, 32),
      location: str(e.location, 16),
      passworded: info.passworded === true,
      players: clients.filter((c) => c.is_player !== false).length,
      clients: clients.length,
      maxClients: typeof info.max_clients === "number" && Number.isFinite(info.max_clients) ? info.max_clients : 0,
    });
  }
  return rows;
}

const BLOCK_RE = /block|blmap|copy (love|the) box|love box/i;

export function isBlock(r: ServerRow): boolean {
  return BLOCK_RE.test(r.name) || BLOCK_RE.test(r.map) || BLOCK_RE.test(r.gameType);
}

export function pickBlockServer(rows: ServerRow[], opts: { avoid?: string | Iterable<string>; lang?: string } = {}): ServerRow | null {
  const avoid = new Set(typeof opts.avoid === "string" ? [opts.avoid] : (opts.avoid ?? []));
  let best: ServerRow | null = null;
  let bestScore = -Infinity;
  for (const r of rows) {
    if (!isBlock(r) || r.passworded || avoid.has(r.address)) continue;
    if (r.maxClients > 0 && r.clients >= r.maxClients - 1) continue;
    if (r.players < 2) continue;
    const score = r.players + (opts.lang === "ru" && /:ru$/i.test(r.location) ? 6 : 0);
    if (score > bestScore) {
      best = r;
      bestScore = score;
    }
  }
  return best;
}

export async function fetchMaster(fetchImpl: typeof fetch = fetch): Promise<ServerRow[]> {
  let last: unknown = null;
  for (const url of MASTERS) {
    try {
      const res = await fetchImpl(url, { signal: AbortSignal.timeout(8000), headers: { "user-agent": "ddnet-ai" } });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const rows = parseMaster(await res.json());
      if (rows.length > 0) return rows;
      last = new Error("empty list");
    } catch (err) {
      last = err;
    }
  }
  throw last instanceof Error ? last : new Error(String(last));
}

export function readAvoid(file: string, now = Date.now()): string[] {
  try {
    const all = JSON.parse(readFileSync(file, "utf8")) as Record<string, unknown>;
    return Object.entries(all)
      .filter(([, until]) => typeof until === "number" && until > now)
      .map(([addr]) => addr);
  } catch {
    return [];
  }
}

export function addAvoid(file: string, address: string, forMs: number, now = Date.now()): void {
  let all: Record<string, number> = {};
  try {
    all = JSON.parse(readFileSync(file, "utf8")) as Record<string, number>;
  } catch {
    all = {};
  }
  for (const [k, v] of Object.entries(all)) if (typeof v !== "number" || v <= now) delete all[k];
  all[address] = now + forMs;
  mkdirSync(dirname(file), { recursive: true });
  writeFileSync(file, JSON.stringify(all));
}
