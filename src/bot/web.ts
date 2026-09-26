import * as http from "node:http";
import { createReadStream, readFileSync, statSync, writeFileSync } from "node:fs";
import { extname, join, resolve, sep } from "node:path";
import { crc32 } from "node:zlib";
import { CONTENT_TYPES, DATA_CACHE_DIR, SKIN_CACHE_DIR, chosenAssets, dataDirCandidates, downloadData, downloadSkin, downloadableData, findDownloadedMap, firstExisting, pickDataDir, readIfSmall, safeSkinName, scanForUnpacked, skinFiles, steamLibraryData, typedDataDir, userDirCandidates, wavFromPcm } from "./webAssets.ts";
import decodeWavpack from "./wavpack/decode-wavpack.js";
import { SCENE_CACHE_DIR, parseSceneAsync } from "./webMap.ts";
import type { ParsedScene } from "./webMap.ts";
import { pageScript } from "./webPage.ts";
import { isAuto } from "./serverPick.ts";
import { EN, getLang, isLang, makeT, t } from "../i18n.ts";
import type { Lang } from "../i18n.ts";

const LAUNCH_FILE = "settings.json";

const ASSET_KINDS = new Set([".wav", ".ogg", ".mp3", ".png", ".ttf", ".otf"]);

function assetRoot(): string | null {
  const set = readLaunch().ddnetData;
  if (typeof set === "string" && set.trim() !== "") {
    const typed = typedDataDir(set);
    if (typed.dir !== null) return typed.dir;
  }
  return defaultAssetRoot();
}

let cachedRoot: string | null | undefined;

function defaultAssetRoot(): string | null {
  if (cachedRoot !== undefined) return cachedRoot;
  const cands = [...dataDirCandidates(process.env)];
  cands.splice(cands.length - 2, 0, ...steamLibraryData(process.env), ...scanForUnpacked(process.env));
  cachedRoot = pickDataDir(cands);
  return cachedRoot;
}

function readLaunch(): Record<string, string> {
  try {

    const raw = JSON.parse(readFileSync(LAUNCH_FILE, "utf8").replace(/^\uFEFF/, "")) as Record<string, string>;
    return typeof raw === "object" && raw !== null && !Array.isArray(raw) ? raw : {};
  } catch {
    return {};
  }
}

function launchServerKey(server: unknown): string | null {
  if (typeof server !== "string" || isAuto(server)) return null;
  const s = server.trim();
  const at = s.lastIndexOf(":");
  if (at <= 0) return `${s.toLowerCase()}:8303`;

  if (!/^\d{1,5}$/.test(s.slice(at + 1))) return null;
  return `${s.slice(0, at).toLowerCase()}:${Number(s.slice(at + 1))}`;
}

function sameLaunchServer(a: unknown, b: unknown): boolean {
  const x = launchServerKey(a);
  return x !== null && x === launchServerKey(b);
}

const downloadsOn = (): boolean => readLaunch().skinDownload !== "off";

let chosen: { at: number; map: Map<string, string> } | null = null;
function chosenNow(): Map<string, string> {
  if (chosen === null || Date.now() - chosen.at > 5000) chosen = { at: Date.now(), map: chosenAssets(userDirCandidates(process.env)) };
  return chosen.map;
}

async function findAsset(rel: string): Promise<string | null> {
  const picked = chosenNow().get(rel);
  if (picked !== undefined && isFile(picked)) return picked;
  for (const root of [assetRoot(), DATA_CACHE_DIR]) {
    if (root === null) continue;
    const full = resolve(root, rel);
    if (full.startsWith(resolve(root) + sep) && isFile(full)) return full;
  }
  if (downloadsOn() && downloadableData(rel)) return downloadData(rel, DATA_CACHE_DIR);
  return null;
}

const CORE_DATA = ["game.png", "emoticons.png", "extras.png", "hud.png", "arrow.png", "particles.png", "editor/entities_clear/ddnet.png", "editor/speed_arrow.png", "fonts/DejaVuSans.ttf"];
let fetching: Promise<void> | null = null;
function haveGraphics(): boolean {
  if (chosenNow().has("game.png")) return true;
  const root = assetRoot();
  return (root !== null && isFile(join(root, "game.png"))) || isFile(join(DATA_CACHE_DIR, "game.png"));
}
function fetchMissingData(): void {
  if (fetching !== null || !downloadsOn() || haveGraphics()) return;
  fetching = (async () => {
    for (const rel of CORE_DATA) await findAsset(rel);
  })().finally(() => {
    if (haveGraphics()) return;
    setTimeout(() => {
      fetching = null;
    }, 10 * 60_000).unref?.();
  });
}

const wavs = new Map<string, Promise<Buffer | null>>();
function wavOf(file: string): Promise<Buffer | null> {
  let job = wavs.get(file);
  if (job === undefined) {
    job = (async () => {
      try {
        const { channelData, sampleRate } = await decodeWavpack(readFileSync(file));
        return channelData.length > 0 && sampleRate > 0 ? wavFromPcm(channelData, sampleRate) : null;
      } catch {
        return null;
      }
    })();
    wavs.set(file, job);
  }
  return job;
}
import type { BotLine, BotStatus, DuelRow, LiveFrame, LiveMap } from "./bot.ts";

export type WebBot = {
  status: () => BotStatus;
  statsLine: () => string;

  handleConsole: (line: string) => string | Promise<string>;

  liveMap: () => LiveMap | null;
  liveFrame: () => LiveFrame | null;

  clipList?: () => { name: string; size: number; when: number }[];
  duelList?: () => DuelRow[];
  clipPath?: (name: string) => string | null;
  configInfo?: () => unknown;
  commandNames?: () => string[];

  voteOptions?: () => string[];

  relationsInfo?: () => Record<string, string[]>;
  setRelation?: (list: "war" | "friend" | "ignore", name: string, on: boolean) => string;

  llmInfo?: () => unknown;
  setLlm?: (raw: unknown) => string;
  autoChatInfo?: () => unknown;
  setAutoChat?: (raw: unknown) => unknown;
  checkUpdate?: () => Promise<string>;
  knobs?: () => { key: string; value: unknown; def: unknown; changed: boolean }[];
  setKnob?: (key: string, value: unknown) => string;
  resetKnobs?: () => string;

  mapData?: () => { name: string; bytes: Uint8Array } | null;

  emoticons?: () => { id: number; e: number; age: number }[];
};

type ClientLike = {
  map?: { mapBuffer?: Uint8Array; map_name?: string; downloading?: boolean; crc?: number; map_details?: { map_sha256?: Uint8Array } };
  on?: (event: string, fn: (m: { client_id: number; emoticon: number }) => void) => unknown;

  SnapshotUnpacker?: { on?: (event: string, fn: (e: { sound_id?: number; common?: { x?: number; y?: number } }) => void) => unknown };
};
function clientOf(bot: WebBot): ClientLike | null {
  const c = (bot as { client?: unknown }).client;
  return typeof c === "object" && c !== null ? (c as ClientLike) : null;
}

const MAX_LINES = 200;

export type WebUi = { port: number; push: (line: BotLine) => void; close: () => void };

const OVERLAY_PAGE = `<!doctype html><html><head><meta charset="utf-8"><title>DDNet AI duel</title><style>
html,body{margin:0;background:transparent;color:#fff;font:800 56px/1.15 system-ui,"Segoe UI",sans-serif;text-shadow:0 2px 8px #000,0 0 3px #000}
#box{display:inline-flex;flex-direction:column;gap:4px;padding:14px 22px;border-radius:14px}
#box.bg{background:rgba(10,14,20,.72)}
#box[hidden]{display:none}
#row{display:flex;gap:.45em;align-items:baseline;white-space:nowrap}
.n{font-size:.62em;font-weight:700;max-width:9em;overflow:hidden;text-overflow:ellipsis}
#sc{font-variant-numeric:tabular-nums}
#sub{font:600 22px/1.2 system-ui,sans-serif;opacity:.85}
#box.idle{opacity:.7}
</style></head><body><div id="box"><div id="row"><span class="n" id="me"></span><span id="sc"></span><span class="n" id="op"></span></div><div id="sub"></div></div>
<script>
const $=(id)=>document.getElementById(id);
if(new URLSearchParams(location.search).get("bg")==="1")$("box").classList.add("bg");
async function pull(){
 let d=null;try{d=await(await fetch("/api/duelnow",{cache:"no-store"})).json()}catch{d=null}
 const box=$("box");
 if(!d){box.hidden=true;return}
 const cur=d.now,last=d.last,show=cur||last;
 box.hidden=!show;if(!show)return;
 box.classList.toggle("idle",!cur);
 $("me").textContent=cur?d.me:(last.by||d.me);
 $("op").textContent=cur?cur.name:last.opponent;
 $("sc").textContent=(cur?cur.ours:last.ours)+" : "+(cur?cur.theirs:last.theirs);
 $("sub").textContent=cur?"duel":"last duel";
}
pull();setInterval(pull,500);
</script></body></html>`;

const DUEL_ROWS = 50;

export function startWebUi(bot: WebBot, port: number, version: string): Promise<WebUi> {

  const lines: (BotLine & { seq: number })[] = [];

  const boot = `${process.pid}-${Date.now().toString(36)}`;
  let seq = 0;

  const emotes = new Map<number, { e: number; at: number }>();
  let heard: ClientLike | null = null;

  const heardSounds: { s: number; id: number; x: number; y: number; at: number }[] = [];
  let soundSeq = 0;
  const soundList = (): { s: number; id: number; x: number; y: number }[] => {
    const now = Date.now();
    return heardSounds.filter((h) => now - h.at < 1000).map(({ s, id, x, y }) => ({ s, id, x, y }));
  };

  let heardSnap: unknown = null;
  const listen = setInterval(() => {
    const c = clientOf(bot);
    if (c === null || typeof c.on !== "function") return;
    if (c !== heard) {
      heard = c;
      try {
        c.on("emote", (m) => {
          if (typeof m?.client_id === "number" && m.client_id >= 0) emotes.set(m.client_id, { e: m.emoticon, at: Date.now() });
        });
      } catch {

      }
    }

    const snap = c.SnapshotUnpacker;
    if (snap === undefined || snap === null || snap === heardSnap || typeof snap.on !== "function") return;
    heardSnap = snap;
    try {
      snap.on("sound_world", (e) => {
        const id = e?.sound_id;
        if (typeof id !== "number" || id < 0 || id > 64) return;
        heardSounds.push({ s: ++soundSeq, id, x: Math.round(e.common?.x ?? 0), y: Math.round(e.common?.y ?? 0), at: Date.now() });
        if (heardSounds.length > 64) heardSounds.splice(0, heardSounds.length - 64);
      });
    } catch {

    }
  }, 250);
  listen.unref?.();
  const emoticonList = (): { id: number; e: number; age: number }[] => {
    if (bot.emoticons) return bot.emoticons();
    const now = Date.now();
    const out: { id: number; e: number; age: number }[] = [];
    for (const [id, v] of emotes) {
      if (now - v.at > 2500) emotes.delete(id);
      else out.push({ id, e: v.e, age: now - v.at });
    }
    return out;
  };

  let scene: { name: string; parsed: ParsedScene | null; at: number; job: Promise<ParsedScene | null> | null } | null = null;
  const mapBytes = (): { name: string; bytes: Uint8Array } | null => {
    const given = bot.mapData?.();
    if (given) return given;
    const c = clientOf(bot);
    const name = c?.map?.map_name ?? bot.liveFrame()?.map ?? "";
    if (c?.map?.mapBuffer && c.map.mapBuffer.length > 0 && !c.map.downloading) return { name, bytes: c.map.mapBuffer };
    if (name === "" || name === "?") return null;
    const safe = name.replace(/[\\/]/g, "_");
    const sha = c?.map?.map_details?.map_sha256;
    const hashes = {
      crc: typeof c?.map?.crc === "number" ? c.map.crc : undefined,
      sha256: sha instanceof Uint8Array && sha.length === 32 && sha.some((b) => b !== 0) ? Buffer.from(sha).toString("hex") : undefined,
    };
    const file = firstExisting([join("maps", `${safe}.map`)]) ?? findDownloadedMap(safe, userDirCandidates(process.env), undefined, hashes);
    const bytes = file === null ? null : readIfSmall(file);
    return bytes === null ? null : { name, bytes };
  };
  const sceneName = (): string => clientOf(bot)?.map?.map_name ?? bot.mapData?.()?.name ?? bot.liveFrame()?.map ?? "";

  const sceneKey = (): string => {
    const name = sceneName();
    const crc = clientOf(bot)?.map?.crc;
    if (typeof crc === "number") return `${name}#${(crc >>> 0).toString(16)}`;
    const given = bot.mapData?.();
    return given ? `${name}#${(crc32(given.bytes) >>> 0).toString(16)}` : name;
  };
  const currentScene = (): Promise<ParsedScene | null> => {
    const want = sceneKey();
    if (scene !== null && scene.name === want) {
      if (scene.job !== null) return scene.job;
      if (scene.parsed !== null || Date.now() - scene.at < 5000) return Promise.resolve(scene.parsed);
    }
    const entry: { name: string; parsed: ParsedScene | null; at: number; job: Promise<ParsedScene | null> | null } = { name: want, parsed: null, at: Date.now(), job: null };
    scene = entry;
    const src = mapBytes();
    if (src === null) return Promise.resolve(null);
    entry.job = parseSceneAsync(src.bytes, src.name, 4, SCENE_CACHE_DIR)
      .catch(() => null)
      .then((parsed) => {
        entry.parsed = parsed;
        entry.at = Date.now();
        entry.job = null;
        return parsed;
      });
    return entry.job;
  };

  const sceneFor = (url: URL): Promise<ParsedScene | "other" | null> => {
    const m = url.searchParams.get("m");
    return currentScene().then((sc) => (sc !== null && m !== null && m !== sc.scene.name ? "other" : sc));
  };
  const push = (line: BotLine): void => {
    lines.push({ ...line, seq: ++seq });
    if (lines.length > MAX_LINES) lines.shift();
  };

  const ownHost = (h: string | undefined): boolean => {
    const addr = server.address();
    const p = typeof addr === "object" && addr !== null ? addr.port : port;
    return h === `127.0.0.1:${p}` || h === `localhost:${p}`;
  };
  const allowed = (req: http.IncomingMessage): boolean => {
    if (!ownHost(req.headers.host)) return false;
    if (req.method === "GET" || req.method === "HEAD") return true;
    if (req.headers["sec-fetch-site"] === "cross-site") return false;
    const origin = req.headers.origin;
    if (origin === undefined) return true;
    try {
      const o = new URL(origin);
      return o.protocol === "http:" && ownHost(o.host);
    } catch {
      return false;
    }
  };

  const server = http.createServer((req, res) => {
    if (!allowed(req)) {
      res.writeHead(403, { "content-type": "text/plain; charset=utf-8" });
      res.end(t("чужой запрос"));
      return;
    }

    try {
      route(req, res);
    } catch (err) {
      if (!res.headersSent) res.writeHead(500, { "content-type": "text/plain; charset=utf-8" });
      res.end(err instanceof Error ? err.message : String(err));
    }
  });

  const route = (req: http.IncomingMessage, res: http.ServerResponse): void => {
    const url = new URL(req.url ?? "/", "http://localhost");
    if (url.pathname === "/api") {
      res.writeHead(200, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
      res.end(JSON.stringify({ status: bot.status(), stats: bot.statsLine(), version, boot, lines: lines.slice(-120) }));
      return;
    }

    if (url.pathname === "/api/map") {
      const map = bot.liveMap();
      res.writeHead(200, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
      res.end(JSON.stringify(map));
      return;
    }

    if (url.pathname === "/api/live") {
      res.writeHead(200, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
      const f = bot.liveFrame();
      res.end(JSON.stringify(f === null ? null : { ...f, emoticons: emoticonList(), sounds: soundList() }));
      return;
    }

    const tilesAt = /^\/api\/scene\/tiles\/(\d+)$/.exec(url.pathname);
    const imageAt = /^\/api\/scene\/image\/(\d+)$/.exec(url.pathname);
    if (url.pathname === "/api/scene" || tilesAt !== null || imageAt !== null) {
      void sceneFor(url)
        .then(async (sc) => {
          if (sc === "other") {
            res.writeHead(409, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
            res.end("null");
            return;
          }
          if (tilesAt === null && imageAt === null) {
            res.writeHead(200, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
            res.end(JSON.stringify(sc === null ? null : sc.scene));
            return;
          }
          if (tilesAt !== null) {
            const data = sc?.tiles.get(Number(tilesAt[1]));
            if (data === undefined) {
              res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
              res.end(t("нет такого слоя"));
              return;
            }
            res.writeHead(200, { "content-type": "application/octet-stream", "cache-control": "no-store" });
            res.end(Buffer.from(data.buffer, data.byteOffset, data.byteLength));
            return;
          }
          const png = sc === null || imageAt === null ? null : await sc.png(Number(imageAt[1]));
          if (png === null) {
            res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
            res.end(t("нет такой картинки"));
            return;
          }
          res.writeHead(200, { "content-type": "image/png", "cache-control": "no-store" });
          res.end(png);
        })
        .catch((err: unknown) => {
          if (!res.headersSent) res.writeHead(500, { "content-type": "text/plain; charset=utf-8" });
          res.end(err instanceof Error ? err.message : String(err));
        });
      return;
    }

    if (url.pathname.startsWith("/skins/") && url.pathname.endsWith(".png")) {
      let name: string | null = null;
      try {
        name = safeSkinName(decodeURIComponent(url.pathname.slice("/skins/".length, -".png".length)));
      } catch {
        name = null;
      }
      const notFound = (): void => {
        res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
        res.end(t("нет такого скина"));
      };
      if (name === null) {
        notFound();
        return;
      }
      const send = (file: string): void => sendFile(res, file, { "content-type": "image/png", "cache-control": "max-age=3600" });
      const local = firstExisting(skinFiles(name, assetRoot(), userDirCandidates(process.env), SKIN_CACHE_DIR));
      if (local !== null) {
        send(local);
        return;
      }

      if (readLaunch().skinDownload === "off" || req.headers["sec-fetch-site"] === "cross-site") {
        notFound();
        return;
      }
      void downloadSkin(name, SKIN_CACHE_DIR).then((file) => (file === null ? notFound() : send(file)));
      return;
    }

    if (url.pathname === "/overlay") {
      res.writeHead(200, { "content-type": "text/html; charset=utf-8", "cache-control": "no-store" });
      res.end(OVERLAY_PAGE);
      return;
    }
    if (url.pathname === "/api/duelnow") {
      res.writeHead(200, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
      const st = bot.status();
      const last = bot.duelList?.()[0] ?? null;

      const second = (st as { dummy?: { name?: string; duelScore?: { name: string; ours: number; theirs: number } | null } }).dummy;
      const mine = st.panel?.duelScore ?? null;
      const now = mine ?? second?.duelScore ?? null;
      const me = mine === null && now !== null ? (second?.name ?? "") : (st.name ?? "");
      res.end(JSON.stringify({ me, now, last: last === null ? null : { opponent: last.opponent, ours: last.ours, theirs: last.theirs, at: last.at, by: last.by ?? null } }));
      return;
    }

    if (url.pathname === "/api/duels") {
      res.writeHead(200, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });

      const all = bot.duelList?.() ?? [];
      const sum = (k: "ours" | "theirs"): number => all.reduce((a, d) => a + (Number.isFinite(d[k]) ? d[k] : 0), 0);
      res.end(JSON.stringify({ n: all.length, ours: sum("ours"), theirs: sum("theirs"), list: all.slice(0, DUEL_ROWS) }));
      return;
    }

    if (url.pathname === "/api/clips") {
      res.writeHead(200, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
      res.end(JSON.stringify(bot.clipList?.() ?? []));
      return;
    }
    if (url.pathname.startsWith("/clips/")) {
      let name: string;
      try {
        name = decodeURIComponent(url.pathname.slice("/clips/".length));
      } catch {
        res.writeHead(400, { "content-type": "text/plain; charset=utf-8" });
        res.end(t("плохое имя"));
        return;
      }
      const file = bot.clipPath?.(name) ?? null;
      if (file === null) {
        res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
        res.end(t("нет такой записи"));
        return;
      }
      sendFile(res, file, { "content-type": "application/json; charset=utf-8", "content-disposition": `attachment; filename="${name}"` });
      return;
    }

    if (url.pathname === "/api/knobs" && req.method !== "POST") {
      res.writeHead(200, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
      res.end(JSON.stringify(bot.knobs?.() ?? []));
      return;
    }
    if (url.pathname === "/api/knobs" && req.method === "POST") {
      let raw = "";
      req.on("data", (c) => { raw += String(c); });
      req.on("end", () => {
        let reply = "";
        try {
          const body = JSON.parse(raw) as { key: string; value: unknown; reset?: boolean };
          reply = body.reset === true ? (bot.resetKnobs?.() ?? t("правка настроек недоступна")) : (bot.setKnob?.(body.key, body.value) ?? t("правка настроек недоступна"));
        } catch (err) {
          reply = err instanceof Error ? err.message : String(err);
        }
        res.writeHead(200, { "content-type": "application/json; charset=utf-8" });
        res.end(JSON.stringify({ reply }));
      });
      return;
    }

    if (url.pathname === "/api/launch") {
      if (req.method === "POST") {
        let raw = "";
        req.on("data", (c) => { raw += String(c); });
        req.on("end", () => {
          let reply = t("сохранено, применится после перезапуска");
          try {
            const body = JSON.parse(raw) as Record<string, unknown>;
            const cur = readLaunch();

            if (typeof body.server === "string" && typeof cur.password === "string" && cur.password !== "" && !sameLaunchServer(cur.server, body.server)) {
              cur.password = "";
            }
            for (const k of ["server", "name", "clan", "skin", "ddnetData", "skinDownload", "dummy", "dummyName"]) {
              if (typeof body[k] === "string") cur[k] = body[k] as string;
            }
            writeFileSync(LAUNCH_FILE, JSON.stringify(cur, null, 2));

            if (typeof body.ddnetData === "string") cachedRoot = undefined;
          } catch (err) {
            reply = err instanceof Error ? err.message : String(err);
          }
          res.writeHead(200, { "content-type": "application/json; charset=utf-8" });
          res.end(JSON.stringify({ reply }));
        });
        return;
      }
      const cur = readLaunch();
      delete cur.password;

      delete cur.llm;

      const typed = typeof cur.ddnetData === "string" ? typedDataDir(cur.ddnetData) : { dir: null, note: "" };
      cur.ddnetDataNote = typed.note;
      if (!cur.ddnetData || typed.dir === null) cur.ddnetDataFound = defaultAssetRoot() ?? "";

      fetchMissingData();
      cur.ddnetGraphics = haveGraphics() ? "yes" : "";
      if (fetching !== null && cur.ddnetGraphics === "") cur.ddnetFetching = "yes";
      res.writeHead(200, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
      res.end(JSON.stringify(cur));
      return;
    }

    if (url.pathname.startsWith("/assets/")) {
      let rel: string;
      try {
        rel = decodeURIComponent(url.pathname.slice("/assets/".length));
      } catch {
        res.writeHead(400, { "content-type": "text/plain; charset=utf-8" });
        res.end(t("плохое имя"));
        return;
      }
      const ext = extname(rel).toLowerCase();
      const missing = (): void => {
        res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
        res.end(t("нет такого файла"));
      };
      if (!ASSET_KINDS.has(ext) || rel.includes("\\") || rel.startsWith("/")) {
        missing();
        return;
      }
      void (async () => {
        const headers: http.OutgoingHttpHeaders = { "content-type": CONTENT_TYPES[ext] ?? "application/octet-stream", "cache-control": "max-age=3600" };

        if (ext === ".ttf" || ext === ".otf") headers["access-control-allow-origin"] = "*";
        const file = await findAsset(rel);
        if (file !== null) {
          sendFile(res, file, headers);
          return;
        }

        if (ext === ".wav" && rel.startsWith("audio/")) {
          const wv = await findAsset(rel.slice(0, -".wav".length) + ".wv");
          const wav = wv === null ? null : await wavOf(wv);
          if (wav !== null) {
            res.writeHead(200, { ...headers, "content-length": wav.length });
            res.end(wav);
            return;
          }
        }
        missing();
      })().catch(() => {
        if (!res.headersSent) missing();
      });
      return;
    }
    if (url.pathname === "/api/update" && req.method === "POST") {
      void (async () => {
        let reply = t("обновление недоступно: бот запущен без него");
        try {
          reply = (await bot.checkUpdate?.()) ?? reply;
        } catch (err) {
          reply = err instanceof Error ? err.message : String(err);
        }
        push({ kind: "event", text: t("обновление: {reply}", { reply }) });
        res.writeHead(200, { "content-type": "application/json; charset=utf-8" });
        res.end(JSON.stringify({ reply }));
      })();
      return;
    }
    if (url.pathname === "/api/votes") {
      res.writeHead(200, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
      res.end(JSON.stringify(bot.voteOptions?.() ?? []));
      return;
    }

    if (url.pathname === "/api/relations" && req.method !== "POST") {
      res.writeHead(200, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
      res.end(JSON.stringify(bot.relationsInfo?.() ?? {}));
      return;
    }
    if (url.pathname === "/api/relation" && req.method === "POST") {
      let raw = "";
      req.on("data", (c) => { raw += String(c); });
      req.on("end", () => {
        let reply = "";
        try {
          const body = JSON.parse(raw) as { list?: unknown; name?: unknown; on?: unknown };
          const list = body.list === "war" || body.list === "friend" || body.list === "ignore" ? body.list : null;
          if (list !== null && typeof body.name === "string" && body.name.length <= 64) reply = bot.setRelation?.(list, body.name, body.on === true) ?? "";
        } catch (err) {
          reply = err instanceof Error ? err.message : String(err);
        }
        if (reply !== "") push({ kind: "log", text: reply });
        res.writeHead(200, { "content-type": "application/json; charset=utf-8" });
        res.end(JSON.stringify({ reply, lists: bot.relationsInfo?.() ?? {} }));
      });
      return;
    }
    if (url.pathname === "/api/llm" && req.method !== "POST") {
      res.writeHead(200, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
      res.end(JSON.stringify(bot.llmInfo?.() ?? null));
      return;
    }

    if (url.pathname === "/api/llm" && req.method === "POST") {
      let raw = "";
      req.on("data", (c) => {
        raw += String(c);
        if (raw.length > 4000) req.destroy();
      });
      req.on("end", () => {
        let reply = "";
        try {
          reply = bot.setLlm?.(JSON.parse(raw)) ?? "";
        } catch {
          reply = "";
        }
        res.writeHead(reply === "" ? 400 : 200, { "content-type": "application/json; charset=utf-8" });
        res.end(JSON.stringify({ reply }));
      });
      return;
    }
    if (url.pathname === "/api/autochat" && req.method !== "POST") {
      res.writeHead(200, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
      res.end(JSON.stringify(bot.autoChatInfo?.() ?? null));
      return;
    }
    if (url.pathname === "/api/autochat" && req.method === "POST") {
      let raw = "";
      req.on("data", (c) => {
        raw += String(c);
        if (raw.length > 20000) req.destroy();
      });
      req.on("end", () => {
        let cfg: unknown = null;
        try {
          cfg = bot.setAutoChat?.(JSON.parse(raw)) ?? null;
        } catch {
          cfg = null;
        }
        res.writeHead(cfg === null ? 400 : 200, { "content-type": "application/json; charset=utf-8" });
        res.end(JSON.stringify(cfg));
      });
      return;
    }
    if (url.pathname === "/api/commands") {
      res.writeHead(200, { "content-type": "application/json; charset=utf-8", "cache-control": "max-age=60" });
      res.end(JSON.stringify(bot.commandNames?.() ?? []));
      return;
    }
    if (url.pathname === "/api/config") {
      res.writeHead(200, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
      res.end(JSON.stringify(bot.configInfo?.() ?? {}));
      return;
    }
    if (url.pathname === "/cmd" && req.method === "POST") {
      let raw = "";
      req.on("data", (c) => { raw += String(c); });
      req.on("end", async () => {
        let reply = "";
        let line = "";
        try {
          line = String((JSON.parse(raw) as { line?: string }).line ?? "");
          reply = (await bot.handleConsole(line)) ?? "";
        } catch (err) {
          reply = err instanceof Error ? err.message : String(err);
        }

        if (line !== "") push({ kind: "log", text: `> ${line}` });
        for (const l of reply.split("\n")) if (l.trim() !== "") push({ kind: "log", text: l });
        res.writeHead(200, { "content-type": "application/json; charset=utf-8" });
        res.end(JSON.stringify({ reply }));
      });
      return;
    }

    const asked = url.searchParams.get("lang");
    res.writeHead(200, { "content-type": "text/html; charset=utf-8", "cache-control": "no-store" });
    res.end(pageFor(isLang(asked) ? asked : getLang()));
  };

  return new Promise((resolve, reject) => {
    server.once("error", reject);

    server.listen(port, "127.0.0.1", () => {
      const addr = server.address();
      resolve({
        port: typeof addr === "object" && addr !== null ? addr.port : port,
        push,
        close: () => {
          clearInterval(listen);
          server.close();
        },
      });
    });
  });
}

function isFile(p: string): boolean {
  try {
    return statSync(p).isFile();
  } catch {
    return false;
  }
}

function sendFile(res: http.ServerResponse, file: string, headers: http.OutgoingHttpHeaders): void {
  const stream = createReadStream(file);
  stream.on("error", () => {
    if (!res.headersSent) {
      res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
      res.end(t("нет такого файла"));
    } else res.destroy();
  });
  stream.once("open", () => {
    res.writeHead(200, headers);
    stream.pipe(res);
  });
  res.once("close", () => stream.destroy());
}

const PAGE_DIR = new URL("./page/", import.meta.url);
const pages = new Map<Lang, string>();

function pageFor(lang: Lang): string {
  const have = pages.get(lang);
  if (have !== undefined) return have;
  const read = (name: string): string => readFileSync(new URL(name, PAGE_DIR), "utf8");

  let news: unknown = [];
  try {
    news = JSON.parse(read("whatsnew.json"));
  } catch {

  }
  const script = `const LANG=${JSON.stringify(lang)};\nconst EN=${JSON.stringify(EN)};\nconst NEWS=${JSON.stringify(news)};\nconst {t,tr}=(${makeT.toString()})(EN,LANG);\n${pageScript()}\n${read("page.js")}`;
  const page = read("index.html")
    .replace('<html lang="ru">', () => `<html lang="${lang}"${lang === "en" ? ' class="i18n-wait"' : ""}>`)
    .replace("</head>", () => `<style>${read("page.css")}</style></head>`)
    .replace("</body>", () => `<script>${script.replace(/<\/script/gi, "<\\/script")}</script></body>`);
  pages.set(lang, page);
  return page;
}
