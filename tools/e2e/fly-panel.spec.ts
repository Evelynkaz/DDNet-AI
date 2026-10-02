// Real-browser test of the «Муха» tab (task 7.4): the offline watch source (`ddnet-ai fly watch`, a trained bundle playing an
// arena game) feeds a real `ddnet-ai web` process (scratch data directory, ephemeral port — never the production unit) through
// the bridge socket; the page must show the group heat strips, the polar eye, DN bars, the action and the time series.
// Needs a trained bundle (default: E-005's final) and the S graph on disk; without them the tests skip. No game server.
//
// Screenshots: ~/aiddnet/data/screenshots/7.4-desktop.png and 7.4-phone.png (7.4-hybrid-* with DDAI_FLY_HYBRID=1; never in git). How to run: README.md.

import { test, expect, type Page } from "@playwright/test";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { mkdtempSync, mkdirSync, existsSync, rmSync } from "node:fs";
import { tmpdir, homedir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, "..", "..");
const BINARY = path.join(REPO_ROOT, "target", "debug", "ddnet-ai");
const DATA = path.join(homedir(), "aiddnet", "data");
const SCREENSHOT_DIR = path.join(DATA, "screenshots");
const BUNDLE = process.env.DDAI_FLY_BUNDLE ?? path.join(DATA, "runs", "E-005", "e005-fly", "checkpoints", "final.bundle");
const FLYG = path.join(DATA, "connectome", "compiled", "fly-S-v1.flyg");
const ARENA = process.env.DDAI_FLY_ARENA ?? "clb-left";
// DDAI_FLY_HYBRID=1: the fly as the hybrid's proposer (the tab then shows how often its proposal is played); screenshots get a "hybrid-" prefix.
const HYBRID = process.env.DDAI_FLY_HYBRID === "1";
const SHOT = HYBRID ? "7.4-hybrid" : "7.4";

let dataDir: string;
let password: string;
let web: ChildProcessWithoutNullStreams;
let watch: ChildProcessWithoutNullStreams;
let baseUrl: string;

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

test.describe.configure({ mode: "serial" });
test.setTimeout(60_000); // the desktop test waits for the first frame, fills the history and polls: a slow start must not trip the default 30 s

test.beforeAll(async () => {
  test.skip(!existsSync(BUNDLE) || !existsSync(FLYG) || !existsSync(BINARY), "needs a trained bundle, the S graph and a built ddnet-ai");
  mkdirSync(SCREENSHOT_DIR, { recursive: true });
  dataDir = mkdtempSync(path.join(tmpdir(), "ddai-web-fly-"));
  mkdirSync(path.join(dataDir, "bot"), { recursive: true });
  const passwd = await runCli(["web-passwd", "--data-dir", dataDir, "--show"]);
  const m = passwd.match(/^password: (\S+)$/m);
  if (!m) throw new Error("no password in web-passwd output");
  password = m[1];

  const sock = path.join(dataDir, "bot", "fly-watch.sock");
  watch = spawn(BINARY, ["fly", "watch", "--bundle", BUNDLE, "--arena", ARENA, "--bridge", sock, "--seed", "3", ...(HYBRID ? ["--hybrid"] : [])], {
    stdio: ["ignore", "pipe", "pipe"],
    cwd: REPO_ROOT,
  });
  watch.stderr.on("data", (c) => process.stderr.write(`[fly watch] ${c}`));
  await new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("fly watch did not start")), 20_000);
    watch.stderr.on("data", (c) => {
      if (c.toString().includes("serving the fly's stream")) {
        clearTimeout(timer);
        resolve();
      }
    });
  });

  web = spawn(BINARY, ["web", "--listen", "127.0.0.1:0", "--data-dir", dataDir, "--bot-socket", sock], { stdio: ["ignore", "pipe", "pipe"] });
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
  web?.kill();
  watch?.kill();
  if (dataDir) rmSync(dataDir, { recursive: true, force: true });
});

async function openFlyTab(page: Page) {
  await page.goto(baseUrl);
  await page.locator("#password").fill(password);
  await page.locator("#login-form button[type=submit]").click();
  await expect(page.locator("#tabbar")).toBeVisible();
  await page.locator("#tab-fly").click();
  await expect(page.locator("#fly-view")).toBeVisible();
}

/** Number of canvas pixels that are not fully transparent (a blank canvas has none). */
async function paintedPixels(page: Page, id: string): Promise<number> {
  return page.evaluate((canvasId) => {
    const c = document.getElementById(canvasId) as HTMLCanvasElement;
    const d = c.getContext("2d")!.getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let i = 3; i < d.length; i += 4) if (d[i] !== 0) n++;
    return n;
  }, id);
}

async function expectLivePanel(page: Page) {
  await expect(page.locator("#fly-state")).toHaveText("муха работает", { timeout: 15_000 });
  await expect(page.locator("#fly-content")).toBeVisible();
  // Bundle: name and hash only.
  await expect(page.locator("#fly-bundle")).toHaveText("e005-fly/final");
  await expect(page.locator("#fly-hash")).toHaveText(/^[0-9a-f]{16}$/);
  await expect(page.locator("#fly-brain")).toContainText(HYBRID ? "предлагает" : "fly-");
  if (HYBRID) {
    await expect(page.locator("#fly-proposer")).toBeVisible();
    await expect(page.locator("#fly-prop-share")).toContainText("%");
    await expect(page.locator("#fly-ts-share-box")).toBeVisible();
  } else {
    await expect(page.locator("#fly-proposer")).toBeHidden();
  }
  // Let the history fill a little, then every drawing must have paint on it.
  await page.waitForTimeout(6_000);
  for (const id of ["fly-eye", "fly-groups", "fly-dn", "fly-ts-fam", "fly-ts-act", "fly-ts-lat"]) {
    expect(await paintedPixels(page, id), `${id} is drawn`).toBeGreaterThan(500);
  }
  // The group strips: one labelled row per group in the layout.
  const frame = await page.evaluate(() => (window as any).FlyPanel._last());
  expect(frame.groups.length).toBeGreaterThanOrEqual(20);
  expect(frame.dn.length).toBe(100);
  expect(frame.eye.length).toBe(7);
  await expect(page.locator("#fly-dir")).toHaveText(/влево|стоп|вправо/);
  await expect(page.locator("#fly-aim")).toContainText("°");
  await expect(page.locator("#fly-act-jump-p")).toContainText("%");
}

test("desktop: groups, eye, DN bars, action and series are live; the stream stops with the tab", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 1000 });
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await openFlyTab(page);
  await expectLivePanel(page);

  // The polar eye changes as the fly plays (two snapshots of the canvas differ).
  const a = await page.locator("#fly-eye").screenshot();
  await page.waitForTimeout(1_500);
  const b = await page.locator("#fly-eye").screenshot();
  const g1 = await page.locator("#fly-groups").screenshot();
  await page.waitForTimeout(1_500);
  const g2 = await page.locator("#fly-groups").screenshot();
  expect(a.equals(b) && g1.equals(g2), "something on the page moves").toBe(false);

  // A channel switch redraws without the page erroring.
  await page.locator("#fly-eye-legend .chip").first().click();
  await expect(page.locator("#fly-eye-legend .chip").first()).toHaveAttribute("aria-pressed", "false");

  // The text table lists every group.
  await page.locator("#fly-table summary").click();
  await expect(page.locator("#fly-table-body tr").first()).toBeVisible();

  await page.screenshot({ path: path.join(SCREENSHOT_DIR, `${SHOT}-desktop.png`), fullPage: true });

  // Leaving the tab unsubscribes: the frame counter stops.
  await page.locator("#tab-status").click();
  await page.waitForTimeout(700);
  const seq1 = await page.evaluate(() => (window as any).FlyPanel._last().seq);
  await page.waitForTimeout(1_500);
  const seq2 = await page.evaluate(() => (window as any).FlyPanel._last().seq);
  expect(seq2).toBe(seq1);
  await page.locator("#tab-fly").click();
  // (the sequence number restarts with every game, so "moves on" means "differs")
  await expect.poll(async () => page.evaluate(() => (window as any).FlyPanel._last().seq), { timeout: 8_000 }).not.toBe(seq2);
  expect(errors).toEqual([]);
});

test("phone: the same panel, no horizontal scroll, readable", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await openFlyTab(page);
  await expectLivePanel(page);
  // The phone asks for fewer frames by default.
  await expect(page.locator("#fly-hz")).toHaveValue("6");
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(0);
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, `${SHOT}-phone.png`), fullPage: true });
  expect(errors).toEqual([]);
});
