// Real-browser tests for the live map view (task 5.2a). Drives a real `ddnet-ai web` process
// replaying a synthetic trace-b fixture (`ddai-web`'s `make_test_trace` example, built against
// the REAL `~/aiddnet/data/ddnet-server/maps/BlmapChill.map`'s actual sha256 — see that example's
// doc comment for why the corpus itself isn't used directly: no real Oracle B scenario has 8+
// characters on one map, and this task's performance measurement needs that).
//
// How to run: see README.md in this folder. Needs `~/aiddnet/data/ddnet-server/maps/BlmapChill.map`
// on disk (the local DDNet server's map, task 2.1) — skipped (not failed) if it's missing.

import { test, expect, type Page } from "@playwright/test";
import { spawn, execFileSync, type ChildProcessWithoutNullStreams } from "node:child_process";
import { mkdtempSync, mkdirSync, rmSync, existsSync } from "node:fs";
import { tmpdir, homedir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, "..", "..");
const BINARY = path.join(REPO_ROOT, "target", "debug", "ddnet-ai");
const SCREENSHOT_DIR = path.join(homedir(), "aiddnet", "data", "screenshots");
const REAL_MAPS_DIR = path.join(homedir(), "aiddnet", "data", "ddnet-server", "maps");
const REAL_MAP = path.join(REAL_MAPS_DIR, "BlmapChill.map");
const CHARACTER_COUNT = 8;

let dataDir: string;
let tracesDir: string;
let password: string;
let serverProcess: ChildProcessWithoutNullStreams;
let baseUrl: string;

function runCli(args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const proc = spawn(BINARY, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    proc.stdout.on("data", (chunk) => (stdout += chunk.toString()));
    proc.stderr.on("data", (chunk) => (stderr += chunk.toString()));
    proc.on("error", reject);
    proc.on("exit", (code) => {
      if (code === 0) resolve(stdout);
      else reject(new Error(`${BINARY} ${args.join(" ")} exited ${code}: ${stderr}`));
    });
  });
}

test.beforeAll(async () => {
  test.skip(!existsSync(REAL_MAP), `${REAL_MAP} not present on this machine — see task 2.1's setup`);

  mkdirSync(SCREENSHOT_DIR, { recursive: true });
  dataDir = mkdtempSync(path.join(tmpdir(), "ddai-web-e2e-live-"));
  tracesDir = mkdtempSync(path.join(tmpdir(), "ddai-web-e2e-live-traces-"));

  execFileSync(
    "cargo",
    ["build", "-p", "ddai-web", "--features", "test-util", "--example", "make_test_trace"],
    { cwd: REPO_ROOT, stdio: "inherit" }
  );
  const exampleBinary = path.join(REPO_ROOT, "target", "debug", "examples", "make_test_trace");
  execFileSync(
    exampleBinary,
    [
      "--map",
      REAL_MAP,
      "--out",
      // Named to match the real corpus's `realmap_<Name>__seed<N>.trb` convention
      // (`crate::live::replay::map_display_name`) so the HUD shows "BlmapChill", not a raw
      // filename stem.
      path.join(tracesDir, "realmap_BlmapChill__seed99999.trb"),
      "--characters",
      String(CHARACTER_COUNT),
      "--ticks",
      "6000", // 6000 ticks @ 50/s = 120s of real-time playback — comfortably longer than any test below
    ],
    { stdio: "inherit" }
  );

  const passwdOutput = await runCli(["web-passwd", "--data-dir", dataDir, "--show"]);
  const passwordMatch = passwdOutput.match(/^password: (\S+)$/m);
  if (!passwordMatch) {
    throw new Error(`could not find the generated password in web-passwd output:\n${passwdOutput}`);
  }
  password = passwordMatch[1];

  serverProcess = spawn(
    BINARY,
    ["web", "--listen", "127.0.0.1:0", "--data-dir", dataDir, "--replay", tracesDir, "--maps-dir", REAL_MAPS_DIR],
    { stdio: ["ignore", "pipe", "pipe"] }
  );
  serverProcess.stderr.on("data", (chunk) => process.stderr.write(`[ddnet-ai web] ${chunk}`));

  baseUrl = await new Promise((resolve, reject) => {
    let buffer = "";
    const timer = setTimeout(() => {
      reject(new Error(`server did not print its listening address within 10s; got so far: ${buffer}`));
    }, 10_000);
    const onData = (chunk: Buffer) => {
      buffer += chunk.toString();
      const addressMatch = buffer.match(/listening on (http:\/\/\S+)/);
      if (addressMatch) {
        clearTimeout(timer);
        serverProcess.stdout.off("data", onData);
        resolve(addressMatch[1]);
      }
    };
    serverProcess.stdout.on("data", onData);
  });
});

test.afterAll(async () => {
  serverProcess?.kill();
  if (dataDir) rmSync(dataDir, { recursive: true, force: true });
  if (tracesDir) rmSync(tracesDir, { recursive: true, force: true });
});

async function loginAndOpenGameTab(page: Page) {
  await page.goto(baseUrl);
  await page.locator("#password").fill(password);
  await page.locator("#login-form button[type=submit]").click();
  await expect(page.locator("#status-view")).toBeVisible();
  await page.locator("#tab-game").click();
  await expect(page.locator("#game-view")).toBeVisible();
  // Wait until a map has actually loaded (HUD map name stops being the placeholder) before any
  // test starts asserting on rendered content.
  await expect(page.locator("#hud-map-name")).toHaveText("BlmapChill", { timeout: 10_000 });
  // At least one live frame received (tick moved off the placeholder "—").
  await expect(page.locator("#hud-tick")).not.toHaveText("—", { timeout: 10_000 });
}

test("desktop: logs in, sees the map drawn and players moving, screenshot saved", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await loginAndOpenGameTab(page);

  await page.waitForTimeout(300); // let a couple of chunks/characters actually paint
  const before = await page.locator("#game-canvas").screenshot();
  await page.waitForTimeout(1000);
  const after = await page.locator("#game-canvas").screenshot();
  expect(before.equals(after), "canvas pixels should differ 1s apart while characters are moving").toBe(false);

  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "5.2a-desktop.png") });
});

test("phone (360x740): logs in, sees the map and player sheet, screenshot saved", async ({ page }) => {
  await page.setViewportSize({ width: 360, height: 740 });
  await loginAndOpenGameTab(page);
  await expect(page.locator(".player-sheet")).toBeVisible();
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "5.2a-phone.png") });
});

test("following a player keeps the camera on them across 3s", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await loginAndOpenGameTab(page);

  // Follow character 0 via the player list (acceptance criterion 3: "tap a player or the list").
  await page.locator("#player-list .player-row").first().click();

  const rect = await page.locator("#game-canvas").evaluate((el) => {
    const r = el.getBoundingClientRect();
    return { width: r.width, height: r.height };
  });
  const center = { x: rect.width / 2, y: rect.height / 2 };

  const samples: Array<{ x: number; y: number }> = [];
  for (let i = 0; i < 4; i++) {
    await page.waitForTimeout(800);
    const state = await page.evaluate(() => (window as any).__ddaiDebug.getState());
    expect(state.followId).not.toBeNull();
    const screen = await page.evaluate(
      (id) => (window as any).__ddaiDebug.screenPositionOf(id),
      state.followId
    );
    expect(screen).not.toBeNull();
    samples.push(screen);
  }

  // The followed character's screen position should stay close to the canvas center the whole
  // time (a generous tolerance — this is smoothed/lerped follow, not a rigid lock, and the
  // synthetic fixture's characters move at a plodding, constant per-tick rate).
  for (const sample of samples) {
    expect(Math.hypot(sample.x - center.x, sample.y - center.y)).toBeLessThan(rect.width * 0.25);
  }
});

test("pause stops the tick, resume continues it", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await loginAndOpenGameTab(page);
  await expect(page.locator("#replay-bar")).toBeVisible({ timeout: 10_000 });

  await page.locator("#replay-pause").click();
  await page.waitForTimeout(200); // let the pause actually reach the source
  const tickA = await page.locator("#hud-tick").textContent();
  await page.waitForTimeout(600);
  const tickB = await page.locator("#hud-tick").textContent();
  expect(tickB).toBe(tickA);

  await page.locator("#replay-play").click();
  await page.waitForTimeout(600);
  const tickC = await page.locator("#hud-tick").textContent();
  expect(tickC).not.toBe(tickB);
});

// ---------------------------------------------------------------------------------------------
// Review round 1, finding F5: a WS close caused by the SESSION ending server-side (not this
// page's own logout button) must show the login form, not retry forever while still showing the
// authenticated UI. Reproduced with two pages sharing one browser context (same session cookie):
// logging out from the SECOND page invalidates the session both pages share, so the FIRST page's
// WS closes for a reason it didn't initiate itself — exactly the scenario the finding describes
// (idle/absolute timeout server-side would close it the same uninitiated way; this is the
// reproduction "logout" names directly, and needs no new server-side test-only configuration).
// ---------------------------------------------------------------------------------------------

test("a session ending on another tab shows the login form here too, without retrying forever", async ({
  browser,
}) => {
  const context = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  const firstPage = await context.newPage();
  await loginAndOpenGameTab(firstPage);

  const secondPage = await context.newPage();
  await secondPage.goto(baseUrl);
  // The second page loads already-authenticated (shared session cookie) — straight to the status
  // view, per `refresh()`'s own `/api/me` check.
  await expect(secondPage.locator("#status-view")).toBeVisible();
  await secondPage.locator("#logout-button").click();
  await expect(secondPage.locator("#login-view")).toBeVisible();

  // Back on the FIRST page: it never clicked logout itself, and must not just sit there retrying
  // — the login form should appear here too, within a few reconnect-backoff cycles.
  await expect(firstPage.locator("#login-view")).toBeVisible({ timeout: 15_000 });
  await expect(firstPage.locator("#game-view")).toBeHidden();

  await context.close();
});

// ---------------------------------------------------------------------------------------------
// Layout (review round 1, finding F4): every core control must actually be the element a real
// click at its own center would hit — not merely "present in the DOM" (which the earlier,
// visibility/text-based assertions elsewhere in this file already covered, and would keep passing
// even with another element painted on top of it). `elementFromPoint` is the direct way to ask
// the browser "what's actually clickable here", independent of z-index/position bugs like the
// ones this finding described.
// ---------------------------------------------------------------------------------------------

async function assertClickTargetIsWithin(page: Page, locatorSelector: string) {
  const box = await page.locator(locatorSelector).boundingBox();
  expect(box, `${locatorSelector} must have a bounding box (be visible/laid out)`).not.toBeNull();
  const cx = box!.x + box!.width / 2;
  const cy = box!.y + box!.height / 2;
  const isWithin = await page.evaluate(
    ({ x, y, selector }) => {
      const target = document.querySelector(selector);
      const hit = document.elementFromPoint(x, y);
      return target !== null && hit !== null && (hit === target || target.contains(hit));
    },
    { x: cx, y: cy, selector: locatorSelector }
  );
  expect(isWithin, `${locatorSelector}'s own center point must hit itself, not something painted over it`).toBe(true);
}

async function layoutChecks(page: Page) {
  await loginAndOpenGameTab(page);

  // The view controls and legend must be reachable — not hidden under `.bottom-stack`/the player
  // sheet (finding F4's "btn-follow, btn-fit and btn-names are covered by the sheet").
  await assertClickTargetIsWithin(page, "#btn-fit");
  await assertClickTargetIsWithin(page, "#btn-follow");
  await assertClickTargetIsWithin(page, "#btn-names");
  await page.locator("#btn-fit").click(); // exercises the real click path, not just the geometry check
  await page.locator("#btn-names").click();
  await expect(page.locator("#btn-names")).not.toHaveClass(/active/);

  // The tab bar must be reachable while `#game-view` (full-viewport, `position: fixed`) is showing
  // — finding F4's core repro: "#tab-status is covered, so the user can't return to «Статус» or
  // log out."
  await assertClickTargetIsWithin(page, "#tab-status");
  await page.locator("#tab-status").click();
  await expect(page.locator("#status-view")).toBeVisible();
  await expect(page.locator("#logout-button")).toBeVisible();

  // Full round trip: logout must actually work from here too, not just be visible.
  await page.locator("#logout-button").click();
  await expect(page.locator("#login-view")).toBeVisible({ timeout: 5_000 });
}

test("desktop layout: tab bar, view controls and logout are all actually clickable", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await layoutChecks(page);
});

test("phone layout: tab bar, view controls and logout are all actually clickable, player sheet is at the bottom", async ({
  page,
}) => {
  await page.setViewportSize({ width: 360, height: 740 });
  await loginAndOpenGameTab(page);

  // Finding F4: "Phone: the player list sits at the top (y=48), not in a bottom sheet." — assert
  // it directly: the sheet's bottom edge must be near the viewport's bottom (just above the tab
  // bar), not sitting right under the HUD near the top.
  const sheetBox = await page.locator(".player-sheet").boundingBox();
  expect(sheetBox).not.toBeNull();
  expect(sheetBox!.y).toBeGreaterThan(740 * 0.5); // well into the lower half of a 740px-tall viewport
  expect(sheetBox!.y + sheetBox!.height).toBeLessThan(740); // and fully above the tab bar

  await assertClickTargetIsWithin(page, "#btn-fit");
  await assertClickTargetIsWithin(page, "#btn-follow");
  await assertClickTargetIsWithin(page, "#btn-names");
  await assertClickTargetIsWithin(page, "#tab-status");
  await page.locator("#tab-status").click();
  await expect(page.locator("#status-view")).toBeVisible();
  await page.locator("#logout-button").click();
  await expect(page.locator("#login-view")).toBeVisible({ timeout: 5_000 });
});

// ---------------------------------------------------------------------------------------------
// Performance (acceptance criterion 4): CPU-throttled render-loop frame time, phone viewport,
// on the largest local map (BlmapChill) with 8+ characters. Reports p50/p95 honestly — see the
// BUILD REPORT for the actual numbers measured on this machine, including if the ≤33ms p95 target
// is missed.
// ---------------------------------------------------------------------------------------------

test("phone performance: CPU x4 throttled render-loop frame time on BlmapChill (8 characters)", async ({
  page,
}) => {
  await page.setViewportSize({ width: 360, height: 740 });
  const client = await page.context().newCDPSession(page);
  await client.send("Emulation.setCPUThrottlingRate", { rate: 4 });

  await loginAndOpenGameTab(page);
  // Let the throttled render loop run for several seconds to accumulate a meaningful sample.
  await page.waitForTimeout(6000);

  const frameTimes: number[] = await page.evaluate(() => (window as any).__ddaiRenderTimes.slice());
  await client.send("Emulation.setCPUThrottlingRate", { rate: 1 });

  expect(frameTimes.length).toBeGreaterThan(30);
  const sorted = [...frameTimes].sort((a, b) => a - b);
  const percentile = (p: number) => sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))];
  const p50 = percentile(0.5);
  const p95 = percentile(0.95);

  console.log(
    `[5.2a perf] BlmapChill, ${CHARACTER_COUNT} characters, CPU x4, phone 360x740: ` +
      `p50=${p50.toFixed(2)}ms p95=${p95.toFixed(2)}ms n=${frameTimes.length}`
  );
  test.info().annotations.push({
    type: "perf",
    description: `p50=${p50.toFixed(2)}ms p95=${p95.toFixed(2)}ms n=${frameTimes.length}`,
  });

  // Reported honestly either way (see the annotation/console line above) — not asserted as a
  // hard failure, per the task's own instruction ("Report honestly if it's missed, with a
  // profile") rather than making the suite red on a machine-dependent timing target.
});

// -------------------------------------------------------------------------------------------
// F3 (review round 1): a slow `/api/map/<sha256>` response for a map the user has since
// navigated away from must never overwrite the scene for the map they're actually looking at
// now. Own server/traces (two real maps, `BlmapChill` + `Blockdale`), separate from the shared
// one above, so delaying one specific map's route doesn't affect any other test in this file.
// -------------------------------------------------------------------------------------------

test.describe("map-load race (finding F3)", () => {
  const SECOND_MAP = path.join(REAL_MAPS_DIR, "Blockdale.map");
  let raceDataDir: string;
  let raceTracesDir: string;
  let raceServerProcess: ChildProcessWithoutNullStreams;
  let raceBaseUrl: string;
  let racePassword: string;

  test.beforeAll(async () => {
    test.skip(!existsSync(SECOND_MAP), `${SECOND_MAP} not present on this machine — see task 2.1's setup`);

    raceDataDir = mkdtempSync(path.join(tmpdir(), "ddai-web-e2e-race-"));
    raceTracesDir = mkdtempSync(path.join(tmpdir(), "ddai-web-e2e-race-traces-"));
    const exampleBinary = path.join(REPO_ROOT, "target", "debug", "examples", "make_test_trace");
    // Sorted-filename playback order (`ReplaySource`, `docs/formats.md` §15.5) — `BlmapChill`
    // must come first, `Blockdale` second, matching the reviewer's own repro order exactly.
    execFileSync(
      exampleBinary,
      [
        "--map",
        REAL_MAP,
        "--out",
        path.join(raceTracesDir, "realmap_BlmapChill__seed1.trb"),
        "--characters",
        "2",
        "--ticks",
        "3000",
      ],
      { stdio: "inherit" }
    );
    execFileSync(
      exampleBinary,
      [
        "--map",
        SECOND_MAP,
        "--out",
        path.join(raceTracesDir, "realmap_Blockdale__seed1.trb"),
        "--characters",
        "2",
        "--ticks",
        "3000",
      ],
      { stdio: "inherit" }
    );

    const passwdOutput = await runCli(["web-passwd", "--data-dir", raceDataDir, "--show"]);
    const passwordMatch = passwdOutput.match(/^password: (\S+)$/m);
    if (!passwordMatch) throw new Error(`could not find the generated password:\n${passwdOutput}`);
    racePassword = passwordMatch[1];

    raceServerProcess = spawn(
      BINARY,
      ["web", "--listen", "127.0.0.1:0", "--data-dir", raceDataDir, "--replay", raceTracesDir, "--maps-dir", REAL_MAPS_DIR],
      { stdio: ["ignore", "pipe", "pipe"] }
    );
    raceServerProcess.stderr.on("data", (chunk) => process.stderr.write(`[race server] ${chunk}`));
    raceBaseUrl = await new Promise((resolve, reject) => {
      let buffer = "";
      const timer = setTimeout(() => reject(new Error(`race server didn't start: ${buffer}`)), 10_000);
      const onData = (chunk: Buffer) => {
        buffer += chunk.toString();
        const m = buffer.match(/listening on (http:\/\/\S+)/);
        if (m) {
          clearTimeout(timer);
          raceServerProcess.stdout.off("data", onData);
          resolve(m[1]);
        }
      };
      raceServerProcess.stdout.on("data", onData);
    });
  });

  test.afterAll(() => {
    raceServerProcess?.kill();
    if (raceDataDir) rmSync(raceDataDir, { recursive: true, force: true });
    if (raceTracesDir) rmSync(raceTracesDir, { recursive: true, force: true });
  });

  test("a slow response for the PREVIOUS map never overwrites the CURRENT one", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 });

    // Discover BlmapChill's sha256 hex the same way the app itself does: read the `map` WS
    // message. Simplest way to get it from the test side without duplicating hashing logic is to
    // just watch the page's own network traffic for the first `/api/map/<sha256>` request.
    let firstMapSha256: string | null = null;
    await page.route("**/api/map/*", async (route) => {
      const url = new URL(route.request().url());
      const sha256 = url.pathname.split("/").pop()!;
      if (firstMapSha256 === null) {
        firstMapSha256 = sha256;
      }
      if (sha256 === firstMapSha256) {
        // Delay ONLY the first map's (BlmapChill's) own response — long enough that the test
        // below presses "next" (switching to Blockdale, whose OWN faster response should win)
        // well before this one finally arrives.
        await new Promise((resolve) => setTimeout(resolve, 2500));
      }
      await route.continue();
    });

    await page.goto(raceBaseUrl);
    await page.locator("#password").fill(racePassword);
    await page.locator("#login-form button[type=submit]").click();
    await expect(page.locator("#status-view")).toBeVisible();
    await page.locator("#tab-game").click();
    await expect(page.locator("#game-view")).toBeVisible();
    await expect(page.locator("#hud-map-name")).toHaveText("BlmapChill", { timeout: 10_000 });

    // Press "next" well before BlmapChill's delayed scene response can possibly land.
    await expect(page.locator("#replay-bar")).toBeVisible({ timeout: 10_000 });
    await page.locator("#replay-next").click();
    await expect(page.locator("#hud-map-name")).toHaveText("Blockdale", { timeout: 10_000 });

    // Blockdale's own (undelayed) scene should already be showing well before BlmapChill's
    // delayed one could possibly land at 2.5s. Checked via the *rendered scene's own* dimensions
    // (`window.__ddaiDebug`), not a screenshot: a raw pixel diff would also change from players
    // simply moving between the two checks, which has nothing to do with the bug being tested —
    // BlmapChill is 1244x667 tiles, Blockdale is 300x300, so the two are trivially distinguishable
    // by dimensions alone regardless of camera/animation state.
    await expect
      .poll(() => page.evaluate(() => (window as any).__ddaiDebug.getState().scene), { timeout: 10_000 })
      .toEqual({ width: 300, height: 300 });

    // Now wait past the 2.5s delay, so BlmapChill's stale response has definitely arrived (or
    // been dropped, if the fix works) at the client, and confirm the scene is STILL Blockdale's —
    // before the fix, the last-response-wins behavior overwrote it back to BlmapChill's
    // (1244x667) once that stale response finally landed.
    await page.waitForTimeout(2500);
    const state = await page.evaluate(() => (window as any).__ddaiDebug.getState());
    expect(state.mapMeta.name).toBe("Blockdale");
    expect(state.scene, "the rendered scene must still be Blockdale's, not reverted to BlmapChill's").toEqual({
      width: 300,
      height: 300,
    });
  });
});

test("bandwidth: measures WS bytes/s at 25 Hz and at 10 Hz (эконом)", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });

  // Bandwidth is measured entirely on the Node side via Playwright's own WS frame inspection
  // (`page.on("websocket", ...)`) — the page's own JS has no byte-counting hook, and doesn't need
  // one, since Playwright can already see every frame's raw payload on the connection the app
  // itself opened. Registered BEFORE navigating: `page.on("websocket", ...)` only fires for
  // connections opened after the listener exists, and the app opens its one WS connection during
  // login/page load, inside `loginAndOpenGameTab` below.
  const totalBytes = { value: 0 };
  page.on("websocket", (ws) => {
    ws.on("framereceived", (frame) => {
      totalBytes.value += typeof frame.payload === "string" ? Buffer.byteLength(frame.payload) : frame.payload.length;
    });
  });

  await loginAndOpenGameTab(page);

  async function measure(hz: number): Promise<number> {
    await page.evaluate((h) => (window as any).__ddaiDebug.setLiveHz(h), hz);
    await page.waitForTimeout(500); // let the new rate settle
    totalBytes.value = 0;
    const started = Date.now();
    await page.waitForTimeout(3000);
    const elapsedSeconds = (Date.now() - started) / 1000;
    return totalBytes.value / elapsedSeconds;
  }

  const bytesAt25 = await measure(25);
  const bytesAt10 = await measure(10);
  console.log(`[5.2a bandwidth] 25 Hz: ${bytesAt25.toFixed(0)} B/s; 10 Hz: ${bytesAt10.toFixed(0)} B/s`);
  test.info().annotations.push({
    type: "bandwidth",
    description: `25Hz=${bytesAt25.toFixed(0)}B/s 10Hz=${bytesAt10.toFixed(0)}B/s`,
  });
  // 10 Hz must use meaningfully less bandwidth than 25 Hz — the actual numbers are reported above
  // and in the BUILD REPORT, this just guards against the subscription rate silently not doing
  // anything at all.
  expect(bytesAt10).toBeLessThan(bytesAt25);
});
