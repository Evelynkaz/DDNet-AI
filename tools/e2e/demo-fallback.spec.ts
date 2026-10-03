// Real-browser test of the offline demo behind the live bot (task 5.7, D-075): the real `ddnet-ai fly watch --pause-idle` (the
// trained fly playing `clb-left`, on its own socket) and the real `ddnet-ai web --bot-socket <live> --demo-socket <demo>` (scratch
// data directory, ephemeral port — never the production unit), plus a FAKE live bot written here (it can come and go).
//
// What it shows: with no live bot the «Игра» tab draws the arena with the fly on it and says «Показ: муха на арене (не настоящая
// игра)» with the map and the weights; the «Бот» tab says there is no bot and its commands are off; the «Муха» tab works; when the
// fake bot comes up the page switches to «Живой бот» (its map, its status, its commands on) and, when it goes, back to the demo.
// The demo's process rests while no browser is open. Desktop and phone.
//
// Needs a trained bundle (default: E-005's final), the S graph and the Copy Love Box map on disk; without them the tests skip.
// No game server. Screenshots: ~/aiddnet/data/screenshots/5.7-*.png (never in git). How to run: README.md.

import { test, expect, type Page } from "@playwright/test";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { mkdtempSync, mkdirSync, existsSync, rmSync, readFileSync } from "node:fs";
import { tmpdir, homedir } from "node:os";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, "..", "..");
const BINARY = path.join(REPO_ROOT, "target", "debug", "ddnet-ai");
const DATA = path.join(homedir(), "aiddnet", "data");
const SCREENSHOT_DIR = path.join(DATA, "screenshots");
const BUNDLE = process.env.DDAI_FLY_BUNDLE ?? path.join(DATA, "runs", "E-005", "e005-fly", "checkpoints", "final.bundle");
const FLYG = path.join(DATA, "connectome", "compiled", "fly-S-v1.flyg");
const CLB_DIR = path.join(DATA, "maps", "copy-love-box");
const CACHE_DIR = path.join(DATA, "maps", "cache");
// The fake live bot plays on another map than the demo, so the page visibly changes.
const LIVE_MAP = { name: "ChillBlock5", sha256: "44f8343a686a05afc8f61e7ca90d95bf8fc97f4074b0f3f6d5f8cff60d1cb378", w: 1000, h: 1000 };

const DEMO_TITLE = "Показ: муха на арене (не настоящая игра)";
const LIVE_TITLE = "Живой бот";

let dataDir: string;
let password: string;
let web: ChildProcessWithoutNullStreams;
let watch: ChildProcessWithoutNullStreams;
let baseUrl: string;
let liveSock: string;
let demoSock: string;

// ---- a fake live bot (the bridge and the control socket), started and stopped by the tests --------------------------------

function message(kind: number, payload: Buffer): Buffer {
  const head = Buffer.alloc(5);
  head.writeUInt32LE(payload.length + 1, 0);
  head.writeUInt8(kind, 4);
  return Buffer.concat([head, payload]);
}

/** A `DWLF` v1 frame (docs/formats.md §15.3): 12-byte header, 26 bytes per character. */
function worldFrame(tick: number, chars: { id: number; x: number; y: number }[]): Buffer {
  const b = Buffer.alloc(12 + 26 * chars.length);
  b.write("DWLF", 0, "latin1");
  b[4] = 1;
  b.writeUInt32LE(tick, 6);
  b.writeUInt16LE(chars.length, 10);
  chars.forEach((c, i) => {
    const o = 12 + 26 * i;
    b[o] = c.id;
    b[o + 1] = 1; // alive
    b.writeInt32LE(c.x, o + 4);
    b.writeInt32LE(c.y, o + 8);
    b.writeInt16LE(120, o + 12);
    b[o + 24] = 0xff; // nobody hooked
  });
  return b;
}

const liveStatus = {
  tick: 1234, own: 0, target: 1, mode: "fight", brain: "hybrid", alive: true, frozen: false, blocks: 7, blocked_by: 2,
  self_kills: 1, decisions: 99, collapsed: 0, decide_p50_us: 800, decide_p99_us: 4100, brain_p99_us: 3900, overhead_p99_us: 200,
  telemetry: null, connected: true, server: "127.0.0.1:8303", map: "ChillBlock5", name: "bot", clan: "Neuroset", skin: "pinky",
  target_tag: "c1-0a1b2c3d", wb: "WB: auto", goto: "", deaths: 4, clips_saved: 2, kill_cooldown_ticks: 0,
};

class FakeLive {
  private conns = new Set<net.Socket>();
  private bridge: net.Server | null = null;
  private control: net.Server | null = null;
  private timer: NodeJS.Timeout | null = null;
  /** Control requests the fake bot received (commands that reached it). */
  received: any[] = [];

  async up() {
    let tick = 100_000;
    this.bridge = net.createServer((conn) => {
      this.conns.add(conn);
      conn.on("close", () => this.conns.delete(conn));
      conn.on("error", () => this.conns.delete(conn));
      conn.write(message(1, Buffer.from("DDBL\x01", "latin1")));
      conn.write(message(2, Buffer.from(JSON.stringify(LIVE_MAP))));
      conn.write(
        message(3, Buffer.from(JSON.stringify({ own: 0, list: [{ id: 0, name: "c0-aaaaaaaa", team: 0 }, { id: 1, name: "c1-0a1b2c3d", team: 0 }] }))),
      );
    });
    this.timer = setInterval(() => {
      tick += 2;
      const t = tick / 25;
      const frame = worldFrame(tick, [
        { id: 0, x: Math.round(1800 + 300 * Math.cos(t)), y: Math.round(1800 + 300 * Math.sin(t)) },
        { id: 1, x: Math.round(1800 - 300 * Math.cos(t)), y: Math.round(1800 - 300 * Math.sin(t)) },
      ]);
      const status = message(5, Buffer.from(JSON.stringify({ ...liveStatus, tick })));
      for (const c of this.conns) {
        c.write(message(4, frame));
        if (tick % 10 === 0) c.write(status);
      }
    }, 40);
    this.control = net.createServer((conn) => {
      let buffer = "";
      conn.on("data", (chunk) => {
        buffer += chunk.toString();
        let nl: number;
        while ((nl = buffer.indexOf("\n")) >= 0) {
          const req = JSON.parse(buffer.slice(0, nl));
          buffer = buffer.slice(nl + 1);
          this.received.push(req);
          conn.write(JSON.stringify({ v: 1, ok: true, text: `did ${req.cmd.type}` }) + "\n");
        }
      });
    });
    await Promise.all([
      new Promise<void>((r) => this.bridge!.listen(liveSock, () => r())),
      new Promise<void>((r) => this.control!.listen(path.join(dataDir, "bot", "control.sock"), () => r())),
    ]);
  }

  async down() {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    for (const c of this.conns) c.destroy();
    this.conns.clear();
    await Promise.all([this.bridge, this.control].map((s) => new Promise<void>((r) => (s ? s.close(() => r()) : r()))));
    this.bridge = this.control = null;
    // `close` removes the socket files on Linux; be sure (the web must see the live socket absent).
    for (const p of [liveSock, path.join(dataDir, "bot", "control.sock")]) rmSync(p, { force: true });
  }
}

const live = new FakeLive();

// ---- the real processes -------------------------------------------------------------------------------------------------

function runCli(args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const proc = spawn(BINARY, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    proc.stdout.on("data", (c) => (stdout += c.toString()));
    proc.stderr.on("data", (c) => (stderr += c.toString()));
    proc.on("error", reject);
    proc.on("exit", (code) => (code === 0 ? resolve(stdout) : reject(new Error(`${args.join(" ")} exited ${code}: ${stderr}`))));
  });
}

/** CPU time (user + system, in clock ticks of 10 ms) the process has used so far. */
function cpuTicks(pid: number): number {
  const stat = readFileSync(`/proc/${pid}/stat`, "utf8");
  const rest = stat.slice(stat.lastIndexOf(")") + 2).split(" ");
  return Number(rest[11]) + Number(rest[12]); // utime, stime (fields 14 and 15)
}

test.describe.configure({ mode: "serial" });
test.setTimeout(90_000);

test.beforeAll(async () => {
  test.skip(
    !existsSync(BUNDLE) || !existsSync(FLYG) || !existsSync(CLB_DIR) || !existsSync(BINARY),
    "needs a trained bundle, the S graph, the Copy Love Box map and a built ddnet-ai",
  );
  mkdirSync(SCREENSHOT_DIR, { recursive: true });
  dataDir = mkdtempSync(path.join(tmpdir(), "ddai-web-demo-"));
  mkdirSync(path.join(dataDir, "bot"), { recursive: true });
  const passwd = await runCli(["web-passwd", "--data-dir", dataDir, "--show"]);
  const m = passwd.match(/^password: (\S+)$/m);
  if (!m) throw new Error("no password in web-passwd output");
  password = m[1];
  liveSock = path.join(dataDir, "bot", "live.sock");
  demoSock = path.join(dataDir, "flydemo", "fly-demo.sock");

  // The demo, as the unit runs it: its own socket, real time, resting while nobody looks.
  watch = spawn(BINARY, ["fly", "watch", "--bundle", BUNDLE, "--arena", "clb-left", "--bridge", demoSock, "--pause-idle", "--seed", "3"], {
    stdio: ["ignore", "pipe", "pipe"],
    cwd: REPO_ROOT,
  });
  watch.stderr.on("data", (c) => process.stderr.write(`[fly watch] ${c}`));
  await new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("fly watch did not start")), 30_000);
    watch.stderr.on("data", (c) => {
      if (c.toString().includes("serving the fly's stream")) {
        clearTimeout(timer);
        resolve();
      }
    });
  });

  web = spawn(
    BINARY,
    ["web", "--listen", "127.0.0.1:0", "--data-dir", dataDir, "--bot-socket", liveSock, "--demo-socket", demoSock, "--maps-dir", CLB_DIR, "--maps-dir", CACHE_DIR],
    { stdio: ["ignore", "pipe", "pipe"] },
  );
  web.stderr.on("data", (c) => process.stderr.write(`[ddnet-ai web] ${c}`));
  baseUrl = await new Promise((resolve, reject) => {
    let buffer = "";
    const timer = setTimeout(() => reject(new Error(`no listening address within 10s: ${buffer}`)), 10_000);
    const onData = (chunk: Buffer) => {
      buffer += chunk.toString();
      const mm = buffer.match(/listening on (http:\/\/\S+)/);
      if (mm) {
        clearTimeout(timer);
        web.stdout.off("data", onData);
        resolve(mm[1]);
      }
    };
    web.stdout.on("data", onData);
  });
});

test.afterAll(async () => {
  await live.down();
  web?.kill();
  watch?.kill();
  if (dataDir) rmSync(dataDir, { recursive: true, force: true });
});

// ---- helpers ------------------------------------------------------------------------------------------------------------

async function signIn(page: Page) {
  await page.goto(baseUrl);
  await page.locator("#password").fill(password);
  await page.locator("#login-form button[type=submit]").click();
  await expect(page.locator("#tabbar")).toBeVisible();
}

async function openTab(page: Page, tab: "game" | "bot" | "fly") {
  await page.locator(`#tab-${tab}`).click();
  await expect(page.locator(`#${tab}-view`)).toBeVisible();
}

const gameState = (page: Page) => page.evaluate(() => (window as any).__ddaiDebug.getState());

/** Canvas pixels that differ from the "air" background: the map and the tees are drawn. */
async function drawnPixels(page: Page): Promise<number> {
  return page.evaluate(() => {
    const c = document.getElementById("game-canvas") as HTMLCanvasElement;
    const d = c.getContext("2d")!.getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let i = 0; i < d.length; i += 4) if (d[i + 3] !== 0 && (d[i] > 40 || d[i + 1] > 40 || d[i + 2] > 40)) n++;
    return n;
  });
}

async function expectDemoOnGameTab(page: Page) {
  const badge = page.locator("#game-source");
  await expect(badge).toBeVisible({ timeout: 20_000 });
  await expect(badge).toHaveClass(/source-demo/);
  await expect(badge.locator(".source-title")).toHaveText(DEMO_TITLE);
  await expect(badge.locator(".source-detail")).toContainText("карта: Copy Love Box", { timeout: 15_000 });
  await expect(badge.locator(".source-detail")).toContainText("арена: clb-left");
  await expect(badge.locator(".source-detail")).toContainText("веса: e005-fly/final");
  // The arena is drawn and two tees (the fly and its opponent) are in the frame, moving.
  await expect.poll(async () => (await gameState(page)).scene !== null, { timeout: 20_000 }).toBe(true);
  await expect.poll(async () => (await gameState(page)).latestFrame?.characters.length ?? 0, { timeout: 20_000 }).toBe(2);
  const s = await gameState(page);
  expect(s.mapMeta.name).toBe("Copy Love Box");
  const tick1 = s.latestFrame.tick;
  await page.waitForTimeout(1_000);
  expect((await gameState(page)).latestFrame.tick).toBeGreaterThan(tick1);
  expect(await drawnPixels(page), "the map is drawn").toBeGreaterThan(2_000);
  // The camera is on the fly (slot 0), close enough to see the hall, not on the whole map.
  await expect.poll(async () => (await gameState(page)).followId, { timeout: 10_000 }).toBe(0);
  // The player list names slots, never nicknames.
  await expect(page.locator("#player-list .name")).toHaveText(["муха", "соперник 1"]);
}

async function expectLiveOnGameTab(page: Page) {
  const badge = page.locator("#game-source");
  await expect(badge).toHaveClass(/source-live/, { timeout: 20_000 });
  await expect(badge.locator(".source-title")).toHaveText(LIVE_TITLE);
  await expect(badge.locator(".source-detail")).toContainText("карта: ChillBlock5", { timeout: 15_000 });
  await expect(badge).not.toContainText("Показ");
  await expect.poll(async () => (await gameState(page)).mapMeta?.name, { timeout: 15_000 }).toBe("ChillBlock5");
  // Only the live bot's frames (ticks from 100000), none of the demo's.
  await expect.poll(async () => (await gameState(page)).latestFrame?.tick ?? 0, { timeout: 15_000 }).toBeGreaterThan(100_000);
  await expect(page.locator("#player-list .name")).toHaveText(["c0-aaaaaaaa", "c1-0a1b2c3d"]);
  // The demo's framing does not carry over: the live view starts on the whole map.
  expect((await gameState(page)).followId).toBeNull();
}

// ---- the tests ----------------------------------------------------------------------------------------------------------

test("desktop: the demo stands in, the live bot takes over and gives back", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await signIn(page);

  // 1. No live bot: the demo is on the «Игра» tab.
  await openTab(page, "game");
  await expectDemoOnGameTab(page);
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "5.7-demo-desktop.png") });

  // 2. The «Бот» tab: there is no bot, and every command is off.
  await openTab(page, "bot");
  await expect(page.locator("#bot-source")).toHaveClass(/source-demo/);
  await expect(page.locator("#bot-source .source-title")).toHaveText(DEMO_TITLE);
  await expect(page.locator("#bot-conn-text")).toContainText("бот не запущен", { timeout: 6_000 });
  await expect(page.locator("#bot-conn-text")).toContainText("показ");
  await expect(page.locator("#bs-server")).toHaveText("—");
  await expect(page.locator("#cmd-demo-note")).toBeVisible();
  for (const sel of ['[data-cmd="stop"]', '[data-cmd="mode"][data-mode="fight"]', "#cmd-brain-apply", "#cmd-kill", "#cmd-clip", "#cmd-goto", "#cmd-spec", "#rel-reload"]) {
    await expect(page.locator(sel), `${sel} is disabled`).toBeDisabled();
  }
  // A command forced out of the page anyway is refused by the server and reaches nothing.
  const refused = await page.evaluate(async () => {
    const me = await (await fetch("/api/me", { credentials: "same-origin" })).json();
    const r = await fetch("/api/bot/command", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", "X-CSRF-Token": me.csrf_token },
      body: JSON.stringify({ type: "stop" }),
    });
    return { status: r.status, body: await r.json() };
  });
  expect(refused).toEqual({ status: 503, body: { error: "demo_only" } });
  expect(live.received).toEqual([]);
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "5.7-demo-bot-desktop.png"), fullPage: true });

  // 3. The «Муха» tab works on the demo (the bundle is the demo's).
  await openTab(page, "fly");
  await expect(page.locator("#fly-source")).toHaveClass(/source-demo/);
  await expect(page.locator("#fly-state")).toHaveText("муха работает", { timeout: 20_000 });
  await expect(page.locator("#fly-bundle")).toHaveText("e005-fly/final");

  // 4. The live bot comes up: the page switches by itself.
  await live.up();
  await openTab(page, "game");
  await expectLiveOnGameTab(page);
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "5.7-live-desktop.png") });
  await openTab(page, "bot");
  await expect(page.locator("#bot-source")).toHaveClass(/source-live/);
  await expect(page.locator("#bot-source .source-title")).toHaveText(LIVE_TITLE);
  await expect(page.locator("#bot-conn-text")).toHaveText("В игре", { timeout: 6_000 });
  await expect(page.locator("#bs-map")).toHaveText("ChillBlock5");
  await expect(page.locator("#cmd-demo-note")).toBeHidden();
  await expect(page.locator('[data-cmd="stop"]')).toBeEnabled();
  await page.locator('[data-cmd="stop"]').click();
  await expect(page.locator("#cmd-result")).toContainText("did stop");
  expect(live.received.map((r) => r.cmd.type)).toEqual(["stop"]);
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "5.7-live-bot-desktop.png"), fullPage: true });
  // The «Муха» tab says the live bot has no fly stream (the fake has none), not the demo's layout.
  await openTab(page, "fly");
  await expect(page.locator("#fly-source")).toHaveClass(/source-live/);
  await expect(page.locator("#fly-content")).toBeHidden({ timeout: 10_000 });
  await expect(page.locator("#fly-empty")).toBeVisible();

  // 5. The live bot goes away: back to the demo, map and all.
  await live.down();
  await openTab(page, "game");
  await expectDemoOnGameTab(page);
  await openTab(page, "bot");
  await expect(page.locator("#bot-conn-text")).toContainText("бот не запущен", { timeout: 6_000 });
  await expect(page.locator('[data-cmd="stop"]')).toBeDisabled();
  expect(live.received.map((r) => r.cmd.type)).toEqual(["stop"]);
  // The fly panel comes back with the demo too (its layout and frames are asked for again from the demo).
  await openTab(page, "fly");
  await expect(page.locator("#fly-source")).toHaveClass(/source-demo/);
  await expect(page.locator("#fly-state")).toHaveText("муха работает", { timeout: 20_000 });
  await expect(page.locator("#fly-bundle")).toHaveText("e005-fly/final");
  expect(errors).toEqual([]);
});

test("phone: demo and live, readable, no horizontal scroll", async ({ page }) => {
  await page.setViewportSize({ width: 360, height: 740 });
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  const noOverflow = async () =>
    expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);
  await signIn(page);

  await openTab(page, "game");
  await expectDemoOnGameTab(page);
  // The badge is wholly inside the screen and does not run under the round buttons at the right edge.
  const box = await page.locator("#game-source").boundingBox();
  const buttons = await page.locator(".view-controls").boundingBox();
  expect(box!.x).toBeGreaterThanOrEqual(0);
  expect(box!.x + box!.width).toBeLessThanOrEqual(buttons!.x);
  await noOverflow();
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "5.7-demo-phone.png") });

  await openTab(page, "bot");
  await expect(page.locator("#cmd-demo-note")).toBeVisible();
  await noOverflow();
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "5.7-demo-bot-phone.png"), fullPage: true });

  await openTab(page, "fly");
  await expect(page.locator("#fly-state")).toHaveText("муха работает", { timeout: 20_000 });
  await noOverflow();

  await live.up();
  await openTab(page, "game");
  await expectLiveOnGameTab(page);
  await noOverflow();
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "5.7-live-phone.png") });
  await openTab(page, "bot");
  await expect(page.locator("#bot-conn-text")).toHaveText("В игре", { timeout: 6_000 });
  await noOverflow();
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "5.7-live-bot-phone.png"), fullPage: true });
  await live.down();
  expect(errors).toEqual([]);
});

test("the demo's game rests while no browser is open and plays while one is", async ({ browser }) => {
  const pid = watch.pid!;
  // Nobody is connected now (the earlier pages are closed): the process uses next to no CPU.
  await new Promise((r) => setTimeout(r, 3_000)); // let the web unit tell the demo that the last browser left
  const idle0 = cpuTicks(pid);
  await new Promise((r) => setTimeout(r, 6_000));
  const idle = cpuTicks(pid) - idle0;

  // A browser opens the site (on a tab that shows nothing of the game): the demo plays.
  const page = await browser.newPage();
  await signIn(page);
  await expect.poll(async () => cpuTicks(pid) - idle0 > idle + 3, { timeout: 10_000 }).toBe(true);
  const busy0 = cpuTicks(pid);
  await new Promise((r) => setTimeout(r, 6_000));
  const busy = cpuTicks(pid) - busy0;
  console.log(`demo CPU over 6 s (10 ms ticks): resting ${idle}, playing ${busy}`);
  expect(busy, "it plays while a browser is open").toBeGreaterThan(idle * 4 + 3);
  expect(idle, "it rests while nobody is").toBeLessThanOrEqual(6);
  await page.close();
});
