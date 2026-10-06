// Screenshots of every tab at 1440x900 and 390x844 for a design review (task 5.14). Local only: a test web on 127.0.0.1:7797 (scratch
// data dir) fed by the scripted fake bot of gamebot.mjs (game, bot, status, training from the real read-only runs dir), a second web on
// 127.0.0.1:7798 fed by `ddnet-ai fly watch` (the «Муха» tab), and synthetic answers for the launcher/servers APIs (those need root
// helpers that do not exist in a scratch instance). No game server, no production unit.
//
//   node support/design-shots.mjs <prefix> [outDir]        e.g. 5.14-before   (files: <outDir>/<prefix>-<tab>-<desk|phone>.png)
//
// DDAI_DEV_ASSETS=1 serves index.html / *.css / *.js from the working tree instead of the copies built into the binary.

import { spawn, execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";
import { startGameBot } from "./gamebot.mjs";
import { BINARY as DEFAULT_BINARY, DDNET_DATA, MAPS, clbOptions, REPO_ROOT } from "./stack.mjs";

const BINARY = process.env.DDAI_BIN ?? DEFAULT_BINARY;

const PORT_A = Number(process.env.DDAI_SHOTS_PORT ?? 7797); // the second web (the «Муха» tab) takes the next port
const PREFIX = process.argv[2] ?? "5.14-shot";
const OUT = process.argv[3] ?? path.join(os.homedir(), "aiddnet", "data", "screenshots");
const DATA = path.join(os.homedir(), "aiddnet", "data");
const BUNDLE = path.join(DATA, "runs", "E-005", "e005-fly", "checkpoints", "final.bundle");
const ASSETS = path.join(REPO_ROOT, "crates", "ddai-web", "assets");
const CSP = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self' wss:; base-uri 'none'; form-action 'self'; frame-ancestors 'none'";
const MIME = { ".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".woff2": "font/woff2" };

fs.mkdirSync(OUT, { recursive: true });
const procs = [];
const cleanups = [];

function spawnLogged(cmd, args, label, opts = {}) {
  const p = spawn(cmd, args, { stdio: ["ignore", "pipe", "pipe"], ...opts });
  p.stderr.on("data", (c) => process.stderr.write(`[${label}] ${c}`));
  procs.push(p);
  return p;
}

function listening(p) {
  return new Promise((resolve, reject) => {
    let buf = "";
    const t = setTimeout(() => reject(new Error("web did not start")), 20000);
    p.stdout.on("data", (c) => {
      buf += c.toString();
      const m = buf.match(/listening on (http:\/\/\S+)/);
      if (m) {
        clearTimeout(t);
        resolve(m[1]);
      }
    });
  });
}

function mkData() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "ddai-shots-"));
  fs.mkdirSync(path.join(dir, "bot"), { recursive: true });
  cleanups.push(() => fs.rmSync(dir, { recursive: true, force: true }));
  const out = execFileSync(BINARY, ["web-passwd", "--data-dir", dir, "--show"], { encoding: "utf8" });
  return { dir, password: out.match(/^password: (\S+)$/m)[1] };
}

// ---- synthetic API answers (the launcher and the server browser need root helpers) ------------------------------------------
const now = Math.floor(Date.now() / 1000);
const MOCK = {
  "/api/bot/launch": {
    enabled: true,
    pending: false,
    launcher_down: false,
    bundle_present: true,
    bundle: "E-005/e005-fly",
    max_sparring: 3,
    servers: [
      { id: "local", kind: "local" },
      { id: "203.0.113.7:8308", kind: "favourite", name: "Swarfey", blocked: false },
    ],
    status: { state: "started", server: "local", brain: "hybrid-fly", duration: "60m", sparring: 2, finish: "target", bundle: "E-005/e005-fly" },
  },
  "/api/servers": {
    enabled: true,
    fetched_at: now - 540,
    age_s: 540,
    master: 4312,
    refresh: { ok: true },
    servers: [
      ["Swarfey | Block | 24/7", "203.0.113.7:8308", "Chill Block", "Block", "eu:de", 18, 32, false, true, true],
      ["GER Block Deluxe", "198.51.100.4:8303", "BlmapChill", "Block", "eu:de", 14, 24, false, true, true],
      ["Hungry Tee's DDRace", "198.51.100.9:8303", "Multeasymap", "DDraceNetwork", "eu:de", 9, 64, false, true, false],
      ["Block Arena (password)", "192.0.2.15:8303", "ctf_arena", "Block", "na:us", 6, 16, true, true, true],
      ["[0.7 only] Zombie Cats", "192.0.2.31:8303", "zomb_cats", "Zombie", "as:sg", 5, 16, false, false, false],
      ["Kog Fast Gores", "198.51.100.77:8303", "Aip-Gores", "Gores", "eu:fi", 3, 32, false, true, false],
    ].map((r) => ({ name: r[0], address: r[1], map: r[2], game_type: r[3], location: r[4], players: r[5], max_clients: r[6], passworded: r[7], v06: r[8], block: r[9] })),
  },
  "/api/favourites": {
    enabled: true,
    favourites: [
      { name: "Swarfey", address: "203.0.113.7:8308", nick: "Муха", connection: "direct", consent_at: now - 86400 * 3, notes: "блок-сервер владельца", blocked: null },
      { name: "Тестовый (закрыт)", address: "198.51.100.200:8303", nick: "Муха", connection: "proxy:eu-1", consent_at: now - 86400 * 9, notes: "", blocked: { at: now - 3600 } },
    ],
  },
  "/api/proxies": {
    enabled: true,
    pending: false,
    last_check_id: null,
    check: null,
    proxies: [
      { name: "eu-1", managed: true, host: "proxy.example.net", port: 1080, relay: "proxy-host-only", session_pick: 3, has_credentials: true, usable: true },
      { name: "file-proxy", managed: false, relay: "public", session_pick: 0, has_credentials: false, usable: false, problem: "bad_mode" },
    ],
  },
};

async function installRoutes(context) {
  await context.route(/^http:\/\/127\.0\.0\.1:\d+\/.*$/, (route) => {
    const u = new URL(route.request().url());
    if (route.request().method() === "GET" && MOCK[u.pathname] && process.env.DDAI_SHOTS_REAL_API !== "1") {
      return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(MOCK[u.pathname]) });
    }
    if (process.env.DDAI_DEV_ASSETS && route.request().method() === "GET") {
      const name = u.pathname === "/" ? "index.html" : u.pathname.slice(1);
      const f = path.join(ASSETS, name);
      if (/^(fonts\/)?[\w.-]+$/.test(name) && fs.existsSync(f)) {
        return route.fulfill({ status: 200, contentType: `${MIME[path.extname(f)] ?? "application/octet-stream"}; charset=utf-8`, body: fs.readFileSync(f), headers: { "content-security-policy": CSP } });
      }
    }
    return route.continue();
  });
}

const VIEWS = [
  ["desk", { width: 1440, height: 900 }],
  ["phone", { width: 390, height: 844 }],
].filter(([v]) => !process.env.DDAI_SHOTS_VIEWS || process.env.DDAI_SHOTS_VIEWS.split(",").includes(v));

// DDAI_SHOTS_ONLY=fly,train limits the run to those shots (names as in the file names, a prefix match: «train» takes «train-run» too).
const ONLY = (process.env.DDAI_SHOTS_ONLY ?? "").split(",").filter(Boolean);
const wanted = (name) => ONLY.length === 0 || ONLY.some((o) => name === o || name.startsWith(o + "-"));

// DDAI_AXE=/path/to/axe.min.js: run axe-core (colour contrast and the basic name/label rules) on every shot and print the violations.
async function axeCheck(page, name, view) {
  if (!process.env.DDAI_AXE) return;
  try {
    await page.addScriptTag({ path: process.env.DDAI_AXE });
    const res = await page.evaluate(async () =>
      // eslint-disable-next-line no-undef
      (await axe.run(document, { runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21aa"] }, resultTypes: ["violations"] })).violations.map((v) => ({
        id: v.id,
        impact: v.impact,
        nodes: v.nodes.slice(0, 5).map((n) => n.target.join(" ") + " :: " + (n.any[0]?.message ?? n.failureSummary ?? "").slice(0, 110)),
        count: v.nodes.length,
      })),
    );
    for (const v of res) console.log(`AXE ${name}-${view}: ${v.id} (${v.impact}) x${v.count}\n   ` + v.nodes.join("\n   "));
    if (res.length === 0) console.log(`AXE ${name}-${view}: clean`);
  } catch (e) {
    console.error("AXE failed", name, String(e.message).split("\n")[0]);
  }
}

async function shot(page, name, view, full) {
  if (!wanted(name)) return;
  await axeCheck(page, name, view);
  const file = path.join(OUT, `${PREFIX}-${name}-${view}.png`);
  try {
    if (full) {
      // A tall viewport instead of Playwright's full-page stitching: the fixed navigation then stays where the user sees it (top bar /
      // bottom dock) instead of appearing in the middle of the picture.
      const size = page.viewportSize();
      const h = await page.evaluate(() => Math.max(document.documentElement.scrollHeight, 0));
      await page.setViewportSize({ width: size.width, height: Math.min(Math.max(h, size.height), 7000) });
      await page.waitForTimeout(600);
      await page.screenshot({ path: file, timeout: 120000, animations: "disabled" });
      await page.setViewportSize(size);
      console.log("wrote", file);
      return;
    }
    await page.screenshot({ path: file, timeout: 120000, animations: "disabled" });
    console.log("wrote", file);
  } catch (e) {
    console.error("FAILED", file, String(e.message).split("\n")[0]);
  }
}

async function login(page, base, password) {
  await page.goto(base);
  await page.locator("#password").waitFor();
  return async () => {
    await page.locator("#password").fill(password);
    await page.locator('#login-form button[type="submit"]').click();
    await page.locator("#tabbar").waitFor({ state: "visible", timeout: 40000 });
  };
}

async function main() {
  // Web A: fake bot (game, bot, status), the real runs directory read-only (training).
  const a = mkData();
  const gb = startGameBot({ sock: path.join(a.dir, "bot", "live.sock"), controlSock: path.join(a.dir, "bot", "control.sock"), ...clbOptions().bot });
  cleanups.push(() => gb.close());
  const webA = spawnLogged(
    BINARY,
    ["web", "--listen", `127.0.0.1:${PORT_A}`, "--data-dir", a.dir, "--bot-socket", path.join(a.dir, "bot", "live.sock"), "--maps-dir", MAPS, "--ddnet-data", DDNET_DATA, "--runs-dir", path.join(DATA, "runs")],
    "web-a",
  );
  const baseA = await listening(webA);

  // Web B: `fly watch` (the «Муха» tab).
  let baseB = null;
  let passB = null;
  if (fs.existsSync(BUNDLE)) {
    const b = mkData();
    passB = b.password;
    const sock = path.join(b.dir, "bot", "fly-watch.sock");
    const watch = spawnLogged(BINARY, ["fly", "watch", "--bundle", BUNDLE, "--arena", "clb-left", "--bridge", sock, "--seed", "3"], "fly-watch", { cwd: REPO_ROOT });
    await new Promise((resolve, reject) => {
      const t = setTimeout(() => reject(new Error("fly watch did not start")), 25000);
      watch.stderr.on("data", (c) => {
        if (c.toString().includes("serving the fly's stream")) {
          clearTimeout(t);
          resolve();
        }
      });
    });
    const webB = spawnLogged(BINARY, ["web", "--listen", `127.0.0.1:${PORT_A + 1}`, "--data-dir", b.dir, "--bot-socket", sock], "web-b");
    baseB = await listening(webB);
  }

  const browser = await chromium.launch({
    headless: true,
    args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--disable-features=LocalNetworkAccessChecks,PrivateNetworkAccessPermissionPrompt,BlockInsecurePrivateNetworkRequests"],
  });
  try {
    for (const [view, size] of VIEWS) {
      const ctx = await browser.newContext({ viewport: size, deviceScaleFactor: view === "phone" ? 2 : 1, locale: "ru-RU", reducedMotion: "reduce", bypassCSP: !!process.env.DDAI_AXE });
      await installRoutes(ctx);
      const page = await ctx.newPage();
      page.on("pageerror", (e) => console.error("PAGEERROR", e.message));
      const doLogin = await login(page, baseA, a.password);
      await page.waitForTimeout(500);
      await shot(page, "login", view, false);
      // a wrong password, for the error state
      await page.locator("#password").fill("nope");
      await page.locator('#login-form button[type="submit"]').click();
      await page.locator("#login-error").waitFor({ state: "visible" });
      await shot(page, "login-error", view, false);
      await doLogin();
      await page.waitForTimeout(1500);
      await shot(page, "status", view, false);

      if (wanted("game")) {
      await page.locator("#tab-game").click();
      await page.waitForTimeout(4000);
      await shot(page, "game", view, false);
      if (view === "desk") {
        await page.locator("#btn-board").click();
        await page.waitForTimeout(600);
        await shot(page, "game-board", view, false);
        await page.locator("#btn-board").click();
      }
      }

      if (wanted("bot")) {
      await page.locator("#tab-bot").click();
      await page.waitForTimeout(3000);
      await shot(page, "bot", view, true);
      }

      if (wanted("servers")) {
      await page.locator("#tab-servers").click();
      await page.waitForTimeout(2000);
      await shot(page, "servers", view, true);
      // the add-to-favourites form of a list row
      await page.locator(".sv-row button", { hasText: "В избранное" }).first().click().catch(() => {});
      await page.waitForTimeout(800);
      await shot(page, "servers-add", view, true);
      }

      if (wanted("train")) {
      await page.locator("#tab-train").click();
      await page.waitForTimeout(2500);
      await shot(page, "train", view, true);
      // open the first run
      const run = page.locator("button.tr-name").first();
      if (await run.count()) {
        await run.click();
        await page.waitForTimeout(2500);
        await shot(page, "train-run", view, true);
      }
      }
      await ctx.close();

      if (baseB && wanted("fly")) {
        const c2 = await browser.newContext({ viewport: size, deviceScaleFactor: view === "phone" ? 2 : 1, locale: "ru-RU", reducedMotion: "reduce", bypassCSP: !!process.env.DDAI_AXE });
        await installRoutes(c2);
        const p2 = await c2.newPage();
        const doLogin2 = await login(p2, baseB, passB);
        await doLogin2();
        await p2.locator("#tab-fly").click();
        await p2.waitForTimeout(9000);
        await shot(p2, "fly", view, true);
        await c2.close();
      }
    }
  } finally {
    await browser.close();
  }
}

try {
  await main();
} finally {
  for (const p of procs) p.kill();
  for (const c of cleanups) {
    try {
      c();
    } catch {}
  }
}
process.exit(0);
