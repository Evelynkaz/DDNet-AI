// The «Игра» tab (task 5.10) in a real browser: the real Copy Love Box map drawn with its own layers, tilesets and quads, real
// tees from the stock skins, the owner's bot highlighted and followed by default, a chat panel that treats hostile text as
// text, the scoreboard, the entities overlay, free camera, and the phone layout. Task 5.11: the owner's chat input under the chat
// panel (Enter sends, the fake bot's control socket takes the line and the game "repeats" it, refusals show inline, the phone
// layout) and the real state of the bot on the «Статус» tab.
//
// Drives a real `ddnet-ai web` (built beforehand: `cargo build -p ddnet-ai`) on an ephemeral port in a scratch data directory,
// fed by the scripted FAKE bot of support/gamebot.mjs over a real Unix socket. The DDNet graphics come from the local DDNet
// 20.1 data directory (`~/aiddnet/build/ddnet-20.1/build/data`), the maps from `~/aiddnet/data/maps/cache`; without them the
// tests that need them are skipped. No game server, no public server, no production unit.
//
// WebGL runs in software (SwiftShader) in a headless machine, so the frame times it reports are a pessimistic floor, not what
// the owner's GPU does. Screenshots: `~/aiddnet/data/screenshots/5.10-*.png` (never in git).
//
// How to run: see README.md in this folder (`npx playwright test game-view.spec.ts`).

import { test, expect, type Browser, type BrowserContextOptions, type Page } from "@playwright/test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
// @ts-expect-error plain JS helper modules
import { startStack, clbOptions, findMap, BINARY, DDNET_DATA } from "./support/stack.mjs";

const SHOTS = path.join(os.homedir(), "aiddnet", "data", "screenshots");
const haveAll = fs.existsSync(BINARY) && fs.existsSync(DDNET_DATA) && !!findMap("Copy Love Box_6e79ef");

test.skip(!haveAll, "needs target/debug/ddnet-ai, the DDNet data directory and the cached Copy Love Box map");
test.use({
  launchOptions: {
    args: [
      "--use-gl=angle",
      "--use-angle=swiftshader",
      "--enable-unsafe-swiftshader",
      "--ignore-gpu-blocklist",
      "--disable-features=LocalNetworkAccessChecks,PrivateNetworkAccessPermissionPrompt,BlockInsecurePrivateNetworkRequests",
    ],
  },
});
// Software WebGL is slow and this machine is shared: generous timeouts, and clicks do not wait for the page to be "stable"
// (that needs two animation frames, which take seconds when the whole map is rasterised on the CPU).
test.describe.configure({ mode: "serial", timeout: 600_000 });

type Stack = Awaited<ReturnType<typeof startStack>>;
let stack: Stack;

// One login for the whole file: the server rate-limits logins from one address, and every test would otherwise log in again.
let authState: Awaited<ReturnType<import("@playwright/test").BrowserContext["storageState"]>>;

test.beforeAll(async ({ browser }) => {
  fs.mkdirSync(SHOTS, { recursive: true });
  stack = await startStack(clbOptions());
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  await page.goto(stack.baseUrl);
  await page.locator("#password").fill(stack.password);
  await page.locator("#login-form button[type=submit]").click({ force: true });
  await expect(page.locator("#tabbar")).toBeVisible({ timeout: 40_000 });
  authState = await ctx.storageState();
  await ctx.close();
});

// The server allows four WebSocket connections per session: every test closes the pages it opened.
const opened: Page[] = [];
test.afterEach(async () => {
  for (const p of opened.splice(0)) await p.context().close().catch(() => {});
});

async function newPage(browser: Browser, options: BrowserContextOptions = {}): Promise<Page> {
  // Reduced motion: the page's drifting backdrop would otherwise repaint the whole window all the time in software compositing.
  const ctx = await browser.newContext({ storageState: authState, reducedMotion: "reduce", ...options });
  const page = await ctx.newPage();
  opened.push(page);
  return page;
}
test.afterAll(() => stack?.stop());

// Development convenience: DDAI_DEV_ASSETS=1 serves the page's own files (index.html, *.js, *.css) from the working tree
// instead of the ones built into the binary, so an edit needs no rebuild. The CSP is the server's own, copied here.
const DEV_ASSETS = process.env.DDAI_DEV_ASSETS ? path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "crates", "ddai-web", "assets") : null;
const DEV_FILES: Record<string, [string, string]> = {
  "/": ["index.html", "text/html"], "/app.css": ["app.css", "text/css"], "/game.css": ["game.css", "text/css"],
  "/app.js": ["app.js", "text/javascript"], "/game.js": ["game.js", "text/javascript"], "/ddmap.js": ["ddmap.js", "text/javascript"],
  "/ddtee.js": ["ddtee.js", "text/javascript"], "/fly.js": ["fly.js", "text/javascript"], "/train.js": ["train.js", "text/javascript"],
  "/say.js": ["say.js", "text/javascript"], "/say.css": ["say.css", "text/css"], "/launch.js": ["launch.js", "text/javascript"], "/launch.css": ["launch.css", "text/css"],
};
const CSP = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self' wss:; base-uri 'none'; form-action 'self'; frame-ancestors 'none'";

const problems: string[] = [];
function watch(page: Page, label: string) {
  if (DEV_ASSETS) {
    void page.route(/^http:\/\/127\.0\.0\.1:\d+\/[^?]*$/, (route) => {
      const hit = DEV_FILES[new URL(route.request().url()).pathname];
      if (!hit) return route.continue();
      return route.fulfill({ status: 200, contentType: `${hit[1]}; charset=utf-8`, body: fs.readFileSync(path.join(DEV_ASSETS, hit[0])), headers: { "content-security-policy": CSP } });
    });
  }
  page.on("pageerror", (e) => problems.push(`${label}: ${e.message}`));
  page.on("console", (m) => {
    // The pixel-sampling test hook reads the buffer back, which the driver warns about (a performance note, not a problem).
    if ((m.type() === "error" || m.type() === "warning") && !m.text().includes("GPU stall")) problems.push(`${label}: ${m.type()}: ${m.text()}`);
  });
  page.on("response", (r) => {
    if (r.status() >= 400 && !r.url().includes("/api/me")) problems.push(`${label}: HTTP ${r.status()} ${r.url()}`);
  });
}

async function openGame(page: Page, base: string, password?: string) {
  await page.goto(base);
  if (password !== undefined) {
    await page.locator("#password").fill(password);
    await page.locator("#login-form button[type=submit]").click({ force: true });
  }
  // (the password hash is the production one: slow on a busy machine)
  await expect(page.locator("#tabbar")).toBeVisible({ timeout: 40_000 });
  await page.locator("#tab-game").click({ force: true });
  await expect(page.locator("#game-view")).toBeVisible();
}

async function untilReady(page: Page, fixScale = true) {
  // The page lowers the map's resolution when it cannot keep up (software WebGL always cannot); screenshots want it sharp.
  if (fixScale) await page.waitForFunction(() => !!(window as any).__ddaiDebug).then(() => page.evaluate(() => (window as any).__ddaiDebug.fixScale()));
  await expect
    .poll(async () => page.evaluate(() => (window as any).__ddaiDebug.getState().mapStatus), { timeout: 60_000 })
    .toBe("ready");
  // every image of the map has come (embedded ones from the server, external ones from the DDNet data directory)
  await expect
    .poll(
      async () =>
        page.evaluate(() => {
          const s = (window as any).__ddaiDebug.getState().sceneImages as string[];
          return s.length > 0 && s.every((x) => x === "ready");
        }),
      { timeout: 60_000 },
    )
    .toBe(true);
  await expect.poll(async () => page.evaluate(() => !!(window as any).__ddaiDebug.getState().latestFrame), { timeout: 20_000 }).toBe(true);
}

async function stats(page: Page) {
  return page.evaluate(() => (window as any).__ddaiDebug.snapshotStats());
}

test("desktop: the real map and tees, the bot highlighted and followed, chat as plain text", async ({ browser }) => {
  const page = await newPage(browser);
  watch(page, "desktop");
  await page.setViewportSize({ width: 1440, height: 900 });
  await openGame(page, stack.baseUrl);
  await untilReady(page);
  await page.waitForTimeout(2500);

  const st = await page.evaluate(() => (window as any).__ddaiDebug.getState());
  expect(st.scene).toEqual({ width: 387, height: 250 }); // the game layer of Copy Love Box
  expect(st.assetsOk).toBe(true);
  expect(st.ownId).toBe(0);
  expect(st.follow).toBe("bot");

  // The picture is a map, not a flat colour: many distinct colours over most of the canvas.
  const s1 = await stats(page);
  expect(s1.nonBlack).toBeGreaterThan(0.9);
  expect(s1.colors).toBeGreaterThan(40);

  // Following the bot: its tee is in the middle of the stage.
  const box = await page.locator("#stage").boundingBox();
  const own = await page.evaluate(() => (window as any).__ddaiDebug.screenPositionOf(0));
  expect(Math.abs(own.x - box!.width / 2)).toBeLessThan(box!.width * 0.12);
  expect(Math.abs(own.y - box!.height / 2)).toBeLessThan(box!.height * 0.12);

  // The roster: five players, the bot marked, tee icons painted from their skins.
  await expect(page.locator("#player-list .player-row")).toHaveCount(5);
  await expect(page.locator("#player-list .player-row.own .pr-badge")).toHaveText("БОТ");
  await expect(page.locator("#player-list .player-row.own .pr-name")).toHaveText("Муха");
  await expect
    .poll(async () =>
      page.evaluate(() => {
        let painted = 0;
        document.querySelectorAll<HTMLCanvasElement>("#player-list canvas.tee-icon").forEach((c) => {
          const d = c.getContext("2d")!.getImageData(0, 0, c.width, c.height).data;
          for (let i = 3; i < d.length; i += 4) if (d[i] > 0) {
            painted++;
            break;
          }
        });
        return painted;
      }),
      { timeout: 60_000 },
    )
    .toBe(5);

  // The bot card, the links to the other tabs.
  await expect(page.locator("#gb-state")).toHaveText("в игре");
  await expect(page.locator("#gb-grid")).toContainText("hybrid");
  await expect(page.locator("#gb-fly")).toBeVisible();

  // Chat: five lines, hostile text is text. No element came out of it, no script ran, the direction override is gone.
  await expect(page.locator("#chat-log .chat-line")).toHaveCount(5);
  const chat = await page.locator("#chat-log").innerText();
  expect(chat).toContain("<img src=x onerror=alert(1)> <b>bold</b> &amp; gnirts");
  expect(chat).toContain("*** 'Kasper' entered the game");
  expect(await page.locator("#chat-log img, #chat-log b, #chat-float img, #chat-float b").count()).toBe(0);
  expect(await page.evaluate(() => (document.getElementById("chat-log")!.textContent || "").includes("‮"))).toBe(false);
  await expect(page.locator(".chat-system")).toHaveCount(1);
  await expect(page.locator(".chat-team")).toHaveCount(1);
  // New lines arrive live and also show over the map for a while.
  stack.bot.say(0, 2, "brainless", "live line <script>1</script>");
  await expect(page.locator("#chat-log .chat-line")).toHaveCount(6);
  await expect(page.locator("#chat-float .chat-line")).toHaveCount(1);
  expect(await page.locator("#chat-float script, #chat-log script").count()).toBe(0);
  stack.bot.say(0, 1, "Kasper", "gg wp");
  stack.bot.say(1, 4, "nameless tee", "team: пошли вниз");
  await expect(page.locator("#chat-float .chat-line")).toHaveCount(3);

  await page.screenshot({ path: path.join(SHOTS, "5.10-desktop-game.png") });
});

async function readyPage(browser: Browser, label: string, width = 1280, height = 800): Promise<Page> {
  const page = await newPage(browser);
  watch(page, label);
  await page.setViewportSize({ width, height });
  await openGame(page, stack.baseUrl);
  await untilReady(page);
  await page.waitForTimeout(1500);
  return page;
}

test("the entities overlay: three views, each a different picture", async ({ browser }) => {
  const page = await readyPage(browser, "entities");
  const plain = await stats(page);
  await page.locator("#btn-entities").click({ force: true });
  await expect(page.locator("#btn-entities")).toHaveText("Вид: карта + сущности");
  const both = await stats(page);
  expect(both.nonBlack).toBeGreaterThan(0.5);
  await page.screenshot({ path: path.join(SHOTS, "5.10-desktop-entities-both.png") });
  await page.locator("#btn-entities").click({ force: true });
  await expect(page.locator("#btn-entities")).toHaveText("Вид: сущности");
  const only = await stats(page);
  expect(only.colors).not.toBe(plain.colors);
  await page.screenshot({ path: path.join(SHOTS, "5.10-desktop-entities.png") });
  await page.locator("#btn-entities").click({ force: true });
  await expect(page.locator("#btn-entities")).toHaveText("Вид: карта");
  expect(problems.filter((p) => p.startsWith("entities"))).toEqual([]);
});

test("the scoreboard: five rows by score, the bot marked, button and held Tab", async ({ browser }) => {
  const page = await readyPage(browser, "board");
  await page.locator("#btn-board").click({ force: true });
  await expect(page.locator("#board")).toBeVisible();
  await expect(page.locator("#board-title")).toHaveText("Copy Love Box");
  await expect(page.locator("#board-rows tr")).toHaveCount(5);
  await expect(page.locator("#board-rows tr").first().locator("td.nm")).toContainText("nameless tee");
  await expect(page.locator("#board-rows tr.own")).toContainText("Муха");
  await page.screenshot({ path: path.join(SHOTS, "5.10-desktop-scoreboard.png") });
  await page.locator("#btn-board").click({ force: true });
  await expect(page.locator("#board")).toBeHidden();
  await page.keyboard.down("Tab");
  await expect(page.locator("#board")).toBeVisible();
  await page.keyboard.up("Tab");
  await expect(page.locator("#board")).toBeHidden();
  expect(problems.filter((p) => p.startsWith("board"))).toEqual([]);
});

test("zoom, the free camera, the follow list, the whole map", async ({ browser }) => {
  const page = await readyPage(browser, "camera");
  const z0 = await page.evaluate(() => (window as any).__ddaiDebug.getState().camera.scale);
  await page.locator("#stage").hover();
  await page.mouse.wheel(0, -300);
  await expect.poll(async () => page.evaluate(() => (window as any).__ddaiDebug.getState().camera.scale)).toBeLessThan(z0);

  // Free camera: dragging the map takes the camera off the bot and moves it.
  const c0 = await page.evaluate(() => (window as any).__ddaiDebug.getState().camera);
  const box = (await page.locator("#stage").boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 160, box.y + box.height / 2 + 60, { steps: 6 });
  await page.mouse.up();
  const after = await page.evaluate(() => (window as any).__ddaiDebug.getState());
  expect(after.follow).toBeNull();
  expect(after.camera.x).toBeLessThan(c0.x);
  await expect(page.locator("#btn-free")).toHaveAttribute("aria-pressed", "true");
  // The way back is the follow list; another player from the list; the whole map.
  await page.locator("#sel-follow").selectOption("bot");
  expect((await page.evaluate(() => (window as any).__ddaiDebug.getState())).follow).toBe("bot");
  await page.locator("#player-list .player-row", { hasText: "Kasper" }).click({ force: true });
  expect((await page.evaluate(() => (window as any).__ddaiDebug.getState())).follow).toBe(1);
  // The camera comes to that player (the tees of the fake bot walk slowly; the bound is generous for slow software frames).
  const stage = (await page.locator("#stage").boundingBox())!;
  await expect
    .poll(
      async () => {
        const p = await page.evaluate(() => (window as any).__ddaiDebug.screenPositionOf(1));
        return p ? Math.hypot(p.x - stage.width / 2, p.y - stage.height / 2) / stage.width : 9;
      },
      { timeout: 60_000 },
    )
    .toBeLessThan(0.3);
  await page.locator("#btn-fit").click({ force: true });
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(SHOTS, "5.10-desktop-whole-map.png") });
  expect(problems.filter((p) => p.startsWith("camera"))).toEqual([]);
});

test("phone: one column, no sideways scroll, the map and the panels", async ({ browser }) => {
  const page = await newPage(browser, { viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  const ctx = page.context();
  watch(page, "phone");
  await openGame(page, stack.baseUrl);
  await untilReady(page);
  await page.waitForTimeout(2000);
  const s = await stats(page);
  expect(s.nonBlack).toBeGreaterThan(0.9);
  const overflow = await page.evaluate(() => ({ doc: document.documentElement.scrollWidth, win: window.innerWidth, view: document.getElementById("game-view")!.scrollWidth, viewW: document.getElementById("game-view")!.clientWidth }));
  expect(overflow.doc).toBeLessThanOrEqual(overflow.win);
  expect(overflow.view).toBeLessThanOrEqual(overflow.viewW + 1);
  await page.screenshot({ path: path.join(SHOTS, "5.10-phone-game.png") });
  // The chat and the roster are below the map: scroll to them.
  await page.locator("#chat-log").scrollIntoViewIfNeeded();
  await expect(page.locator("#chat-log")).toBeVisible();
  await page.screenshot({ path: path.join(SHOTS, "5.10-phone-chat.png") });
  // The tab bar sits at the bottom and reaches every tab.
  const bar = await page.locator("#tabbar").boundingBox();
  expect(bar!.y + bar!.height).toBeGreaterThan(800);
  await ctx.close();
});

test("the other tabs keep working under the new theme (generic cards, forms, status)", async ({ browser }) => {
  const page = await newPage(browser);
  watch(page, "theme");
  await page.setViewportSize({ width: 1280, height: 860 });
  // (the game tab is not opened here: its software-GL loop would only slow the screenshots of the other tabs down)
  await page.goto(stack.baseUrl);
  await expect(page.locator("#tabbar")).toBeVisible({ timeout: 40_000 });
  for (const [tab, view] of [["#tab-bot", "#bot-view"], ["#tab-fly", "#fly-view"], ["#tab-train", "#train-view"], ["#tab-status", "#status-view"]]) {
    await page.locator(tab).click({ force: true });
    await expect(page.locator(view)).toBeVisible();
    const sideways = await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth);
    expect(sideways).toBe(true);
    await page.screenshot({ path: path.join(SHOTS, `5.10-theme-${tab.slice(5)}.png`) });
  }
  // The tab bar is across the top of a desktop window.
  const bar = await page.locator("#tabbar").boundingBox();
  expect(bar!.y).toBeLessThan(2);
});

test("what a frame costs: draw calls and the page's own time (software WebGL here: a floor, not the GPU)", async ({ browser }) => {
  const page = await newPage(browser);
  watch(page, "perf");
  await page.setViewportSize({ width: 1280, height: 720 });
  await openGame(page, stack.baseUrl);
  await untilReady(page, false);
  await page.waitForTimeout(12000);
  const ms = await page.evaluate(() => (window as any).__ddaiDebug.renderMs());
  const scale = await page.evaluate(() => (window as any).__ddaiDebug.renderScale());
  console.log(`one frame of Copy Love Box at 1280x720 in SOFTWARE WebGL on a shared machine: ${JSON.stringify(ms)}; the page settled on ${scale.toFixed(2)} of the full resolution`);
  expect(ms.n).toBeGreaterThan(0);
  // The GPU's work per frame is small and bounded by the map, not by the page: a few dozen draw calls and a few thousand vertices.
  expect(ms.frame.draws).toBeLessThan(120);
  expect(ms.frame.verts).toBeLessThan(40_000);
  await page.screenshot({ path: path.join(SHOTS, "5.10-desktop-720p.png") });
});

test("a page that cannot find the DDNet graphics still draws the map's own layers and flat tees", async ({ browser }) => {
  const bare = await startStack({ ...clbOptions(), ddnetData: null });
  try {
    const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 } });
    const page = await ctx.newPage();
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await openGame(page, bare.baseUrl, bare.password);
    await expect.poll(async () => page.evaluate(() => (window as any).__ddaiDebug.getState().mapStatus), { timeout: 60_000 }).toBe("ready");
    await expect.poll(async () => page.evaluate(() => !!(window as any).__ddaiDebug.getState().latestFrame), { timeout: 20_000 }).toBe(true);
    await page.waitForTimeout(2500);
    // Embedded images came from the server; the external ones (grass_main, moon) are missing, so those layers are not drawn.
    const st = await page.evaluate(() => (window as any).__ddaiDebug.getState());
    expect(st.sceneImages.filter((x: string) => x === "bad").length).toBeGreaterThan(0);
    expect((await stats(page)).nonBlack).toBeGreaterThan(0.5);
    await expect(page.locator("#player-list .player-row")).toHaveCount(5);
    await page.screenshot({ path: path.join(SHOTS, "5.10-desktop-no-ddnet-data.png") });
    expect(errors).toEqual([]);
    await ctx.close();
  } finally {
    bare.stop();
  }
});

test("a map that names an external image by a path never makes the page ask for one", async ({ browser }) => {
  const royal = findMap("blmapV3ROYAL");
  test.skip(!royal, "needs the cached blmapV3ROYAL map");
  const opts = clbOptions();
  opts.bot.mapFile = royal;
  opts.bot.mapName = "blmapV3ROYAL";
  const s = await startStack(opts);
  try {
    const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 } });
    const page = await ctx.newPage();
    const urls: string[] = [];
    page.on("request", (r) => urls.push(r.url()));
    await openGame(page, s.baseUrl, s.password);
    await expect.poll(async () => page.evaluate(() => (window as any).__ddaiDebug.getState().mapStatus), { timeout: 60_000 }).toBe("ready");
    await page.waitForTimeout(4000);
    expect(urls.filter((u) => u.includes("..") || u.includes("%2e") || u.includes("/skins/greyfox") || u.includes("%2F"))).toEqual([]);
    expect(urls.some((u) => u.includes("/assets/mapres/generic_deathtiles.png"))).toBe(true);
    await ctx.close();
  } finally {
    s.stop();
  }
});

test("when the map's layers cannot be built the coarse geometry is drawn and the page says so", async ({ browser }) => {
  const page = await newPage(browser);
  watch(page, "fallback");
  // The scene route fails (as for a map the source has no file for); the older kinds route still works.
  await page.route("**/api/map/*/scene", (route) => route.fulfill({ status: 500, body: "" }));
  await page.setViewportSize({ width: 1280, height: 800 });
  await openGame(page, stack.baseUrl);
  await expect.poll(async () => page.evaluate(() => (window as any).__ddaiDebug.getState().mapStatus), { timeout: 60_000 }).toBe("fallback");
  await expect(page.locator("#stage-msg")).toContainText("показана только геометрия");
  await expect.poll(async () => page.evaluate(() => !!(window as any).__ddaiDebug.getState().latestFrame), { timeout: 20_000 }).toBe(true);
  await page.waitForTimeout(2000);
  const s = await stats(page);
  expect(s.colors).toBeGreaterThan(5);
  await page.screenshot({ path: path.join(SHOTS, "5.10-desktop-geometry-fallback.png") });
  // The 500s are the point of this test, not problems.
  problems.splice(0, problems.length, ...problems.filter((p) => !p.startsWith("fallback:")));
});

// Review round 1 (F3): name plates keep a constant size on the screen and never overlap, however close the tees are or how far
// the view is zoomed in. The followed tee (the owner's bot) shows its clan; the others are faint.
test("name plates: constant size, no overlap, zoomed in near several tees (desktop and phone)", async ({ browser }) => {
  const page = await readyPage(browser, "plates");
  const plates = () => page.evaluate(() => (window as any).__ddaiDebug.getState().plates as { l: number; r: number; t: number; b: number; fs: number }[]);
  const noOverlap = (boxes: { l: number; r: number; t: number; b: number }[]) => {
    for (let i = 0; i < boxes.length; i++)
      for (let j = i + 1; j < boxes.length; j++) {
        const a = boxes[i], b = boxes[j];
        expect(a.l < b.r && a.r > b.l && a.t < b.b && a.b > b.t, `plates ${i} and ${j} overlap: ${JSON.stringify([a, b])}`).toBe(false);
      }
  };
  await page.screenshot({ path: path.join(SHOTS, "5.10-r1-desktop-game.png") });
  await page.locator("#stage").hover();
  // Zoom in on the bot (Kasper stands 90 units from it): the plates must not grow, nor pile up.
  for (let i = 0; i < 4; i++) await page.mouse.wheel(0, -400);
  await expect.poll(async () => page.evaluate(() => (window as any).__ddaiDebug.getState().camera.scale), { timeout: 30_000 }).toBeLessThan(0.8);
  await page.waitForTimeout(1500);
  let boxes = await plates();
  expect(boxes.length).toBeGreaterThan(1);
  for (const b of boxes) expect(b.fs).toBeLessThanOrEqual(16);
  noOverlap(boxes);
  await page.screenshot({ path: path.join(SHOTS, "5.10-r1-desktop-zoomed-tees.png") });
  // Zoomed out a step: close tees stack their plates instead of overlapping.
  for (let i = 0; i < 3; i++) await page.mouse.wheel(0, 400);
  await page.waitForTimeout(1500);
  boxes = await plates();
  noOverlap(boxes);
  await page.screenshot({ path: path.join(SHOTS, "5.10-r1-desktop-plates-stacked.png") });

  const phone = await newPage(browser, { viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  watch(phone, "plates-phone");
  await openGame(phone, stack.baseUrl);
  await untilReady(phone);
  await phone.waitForTimeout(1500);
  noOverlap(await phone.evaluate(() => (window as any).__ddaiDebug.getState().plates));
  await phone.screenshot({ path: path.join(SHOTS, "5.10-r1-phone-game.png") });
  expect(problems.filter((p) => p.startsWith("plates"))).toEqual([]);
});

// Review round 1 (F4): "Вся карта" fits the big maps too (BlmapChill needs a zoom of about 27).
test("the whole map fits a big map (BlmapChill)", async ({ browser }) => {
  test.skip(!findMap("BlmapChill_"), "needs the cached BlmapChill map");
  const opts = clbOptions();
  const big = await startStack({ bot: { ...opts.bot, mapFile: findMap("BlmapChill_"), mapName: "BlmapChill", mapW: 1244, mapH: 667 } });
  try {
    const ctx = await browser.newContext({ reducedMotion: "reduce", viewport: { width: 1280, height: 800 } });
    const page = await ctx.newPage();
    opened.push(page);
    watch(page, "whole-big");
    await openGame(page, big.baseUrl, big.password);
    await untilReady(page);
    await page.locator("#btn-fit").click({ force: true });
    await page.waitForTimeout(1500);
    const st = await page.evaluate(() => (window as any).__ddaiDebug.getState());
    expect(st.scene).toEqual({ width: 1244, height: 667 });
    // The whole map is in view: zoomed out by the ratio that fits it (about 27), not stopped at the old ceiling of 12.
    expect(st.camera.scale).toBeGreaterThan(12);
    expect(st.camera.scale).toBeLessThanOrEqual(50);
    await page.screenshot({ path: path.join(SHOTS, "5.10-r1-desktop-whole-blmapchill.png") });
    expect(problems.filter((p) => p.startsWith("whole-big"))).toEqual([]);
  } finally {
    big.stop();
  }
});

// Merge with 5.9: the «Бот» tab with the launcher card under the new theme, desktop and phone, no sideways scroll.
test("the «Бот» tab with the launcher card renders under the theme (desktop and phone)", async ({ browser }) => {
  for (const [name, options] of [
    ["desktop", { viewport: { width: 1280, height: 900 } }],
    ["phone", { viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true }],
  ] as const) {
    const page = await newPage(browser, options);
    watch(page, `bot-${name}`);
    await page.goto(stack.baseUrl);
    await expect(page.locator("#tabbar")).toBeVisible({ timeout: 40_000 });
    await page.locator("#tab-bot").click({ force: true });
    await expect(page.locator("#bot-view")).toBeVisible();
    await expect(page.locator(".launch-card")).toBeVisible({ timeout: 15_000 });
    await page.waitForTimeout(1200);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await page.screenshot({ path: path.join(SHOTS, `5.10-r1-${name}-bot-launcher.png`), fullPage: name === "phone" });
    expect(problems.filter((p) => p.startsWith(`bot-${name}`))).toEqual([]);
  }
});

// ---- task 5.11: the owner's chat input under the chat panel (D-094) and the bot's real state on «Статус» ------------------------
// The site limits the owner's lines itself (two in 3 s, ten a minute), so these tests send few lines, three seconds apart.

const SAY_GAP_MS = 3200;

async function sayLine(page: Page, text: string, how: "enter" | "button" = "enter") {
  const input = page.locator("#game-say-mount .say-input");
  await input.fill(text);
  if (how === "enter") await input.press("Enter");
  else await page.locator("#game-say-mount .say-send").click({ force: true });
}

test("the chat input sits under the chat panel; Enter sends, the line reaches the bot and comes back in the log", async ({ browser }) => {
  const page = await newPage(browser);
  watch(page, "say");
  await page.setViewportSize({ width: 1440, height: 900 });
  await openGame(page, stack.baseUrl);
  const mount = page.locator("#game-say-mount");
  await expect(mount.locator("form.say-form .say-input")).toBeVisible();
  await expect(mount.locator(".say-send")).toBeDisabled(); // nothing typed yet
  await expect(mount.locator(".say-team")).toContainText("Командный чат");
  await expect(mount.locator(".say-count")).toHaveText("0 / 255");
  // inside the chat card, below the log; the hint says what is true now
  await expect(page.locator(".chat-card #game-say-mount")).toHaveCount(1);
  const log = await page.locator("#chat-log").boundingBox();
  const inp = await mount.locator(".say-input").boundingBox();
  expect(inp!.y).toBeGreaterThan(log!.y + log!.height - 1);
  const hint = await page.locator(".chat-card .hint").first().innerText();
  expect(hint).toContain("Бот сам в чат не пишет");
  expect(hint).not.toContain("Только чтение");
  // The «Бот» tab no longer carries a second input, only a pointer back here.
  expect(await page.locator("#bot-view .say-input").count()).toBe(0);

  const before = stack.bot.control.length;
  await mount.locator(".say-input").fill("привет <b>всем</b> & co");
  await expect(mount.locator(".say-count")).toHaveText(/^\d+ \/ 255$/);
  await expect(mount.locator(".say-send")).toBeEnabled();
  await mount.locator(".say-input").press("Enter");
  await expect(mount.locator(".say-result.ok")).toContainText("Принято");
  await expect(mount.locator(".say-input")).toHaveValue("");
  expect(stack.bot.control.length).toBe(before + 1);
  const cmd = stack.bot.control[stack.bot.control.length - 1].cmd;
  expect(cmd).toEqual({ type: "say", team: false, text: "привет <b>всем</b> & co" });
  // The bot's line is in the panel only when the game server repeats it (the fake bot does, a moment later), as text.
  await expect(page.locator("#chat-log .chat-line", { hasText: "привет <b>всем</b> & co" })).toHaveCount(1);
  await expect(page.locator("#chat-log .chat-line", { hasText: "Муха" }).last()).toBeVisible();
  expect(await page.locator("#chat-log b").count()).toBe(0);

  // The team toggle; the button works as well as Enter.
  await page.waitForTimeout(SAY_GAP_MS);
  await mount.locator(".say-team input").check();
  await sayLine(page, "держим левый вб", "button");
  await expect(mount.locator(".say-result.ok")).toContainText("Принято");
  expect(stack.bot.control[stack.bot.control.length - 1].cmd).toEqual({ type: "say", team: true, text: "держим левый вб" });
  await expect(page.locator("#chat-log .chat-team", { hasText: "держим левый вб" })).toHaveCount(1);

  // Task 4.9b: a line that starts with "/" is a server command the owner typed: it is sent as typed, like any line (the page says what
  // /spec and /pause do under the input). The server does not repeat a command in the chat, so only the request is checked.
  await expect(mount.locator(".say-note")).toHaveText("/spec и /pause ставят бота на паузу; повторите команду — продолжит");
  await page.waitForTimeout(SAY_GAP_MS);
  await mount.locator(".say-team input").uncheck();
  await sayLine(page, "  /emote happy ");
  await expect(mount.locator(".say-result.ok")).toContainText("Принято");
  await expect(mount.locator(".say-input")).toHaveValue("");
  expect(stack.bot.control[stack.bot.control.length - 1].cmd).toEqual({ type: "say", team: false, text: "/emote happy" });

  // A line the page itself refuses (an invisible character hiding in front of a command) says why, stays in the box, and never reaches the bot.
  const n = stack.bot.control.length;
  await sayLine(page, "\u200b/kill");
  await expect(mount.locator(".say-result.bad")).toContainText("невидимые");
  await expect(mount.locator(".say-input")).toHaveValue("\u200b/kill");
  await page.waitForTimeout(300);
  expect(stack.bot.control.length).toBe(n);
  await page.screenshot({ path: path.join(SHOTS, "5.11-desktop-say.png") });
  expect(problems.filter((p) => p.startsWith("say:"))).toEqual([]);
});

test("refusals show inline: not in game, rate limited, queue full, and chat disabled locks the field until the tab is shown again", async ({ browser }) => {
  const page = await newPage(browser);
  watch(page, "say-refuse");
  await page.setViewportSize({ width: 1280, height: 800 });
  await openGame(page, stack.baseUrl);
  const mount = page.locator("#game-say-mount");
  const input = mount.locator(".say-input");
  const result = mount.locator(".say-result");
  try {
    for (const [mode, text] of [
      ["not_in_game", "не в игре"],
      ["rate_limited", "Слишком часто"],
      ["queue_full", "Уже ждут три сообщения"],
    ] as const) {
      stack.bot.sayMode = mode;
      await sayLine(page, `проба ${mode}`);
      await expect(result).toHaveClass(/bad/);
      await expect(result).toContainText(text);
      await expect(input).toHaveValue(`проба ${mode}`); // kept for a retry
      await expect(input).toBeEnabled();
      await page.waitForTimeout(SAY_GAP_MS);
    }
    // chat_disabled: the bot runs with --no-owner-chat. The field says so and is locked.
    stack.bot.sayMode = "chat_disabled";
    await sayLine(page, "ещё одна");
    await expect(result).toContainText("выключен");
    await expect(input).toBeDisabled();
    await expect(mount.locator(".say-send")).toBeDisabled();
    await expect(mount.locator(".say-team input")).toBeDisabled();
    await page.screenshot({ path: path.join(SHOTS, "5.11-desktop-say-disabled.png") });
    // Back on the tab later (the bot may have been restarted): the field is usable again, and a line goes through once the bot takes them.
    stack.bot.sayMode = "ok";
    await page.locator("#tab-bot").click({ force: true });
    await expect(page.locator("#bot-view")).toBeVisible();
    await page.locator("#bot-to-chat").click({ force: true }); // the pointer on the «Бот» tab leads back here
    await expect(page.locator("#game-view")).toBeVisible();
    await expect(input).toBeEnabled();
    await expect(result).toHaveText("");
    await page.waitForTimeout(SAY_GAP_MS);
    await sayLine(page, "снова можно");
    await expect(result).toHaveClass(/ok/);
  } finally {
    stack.bot.sayMode = "ok";
  }
  // Each refused request (409, 429) shows up twice, as a response and as a console error: those four are the point of this test, anything else is a problem.
  const refusals = problems.filter((p) => p.startsWith("say-refuse") && /HTTP (409|429) .*\/api\/bot\/say|status of (409|429)/.test(p));
  expect(refusals).toHaveLength(8);
  problems.splice(0, problems.length, ...problems.filter((p) => !refusals.includes(p)));
  expect(problems.filter((p) => p.startsWith("say-refuse"))).toEqual([]);
});

test("phone: the chat input fits the screen, the field is 16 px (no zoom on focus), no sideways scroll", async ({ browser }) => {
  const page = await newPage(browser, { viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  watch(page, "say-phone");
  await openGame(page, stack.baseUrl);
  const mount = page.locator("#game-say-mount");
  await mount.scrollIntoViewIfNeeded();
  await expect(mount.locator(".say-input")).toBeVisible();
  const f = await mount.locator(".say-input").evaluate((n) => parseFloat(getComputedStyle(n).fontSize));
  expect(f).toBeGreaterThanOrEqual(16);
  const view = page.viewportSize()!;
  for (const sel of [".say-input", ".say-send", ".say-team", ".say-count"]) {
    const b = await mount.locator(sel).boundingBox();
    expect(b!.x, sel).toBeGreaterThanOrEqual(0);
    expect(b!.x + b!.width, sel).toBeLessThanOrEqual(view.width);
  }
  const bt = await mount.locator(".say-send").boundingBox();
  expect(bt!.height).toBeGreaterThanOrEqual(34); // a thumb-sized target
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  const sayWait = SAY_GAP_MS;
  await page.waitForTimeout(sayWait);
  await mount.locator(".say-input").fill("с телефона");
  await mount.locator(".say-input").press("Enter");
  await expect(mount.locator(".say-result.ok")).toBeVisible();
  await expect(page.locator("#chat-log .chat-line", { hasText: "с телефона" })).toHaveCount(1);
  await mount.scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(SHOTS, "5.11-phone-say.png") });
  expect(problems.filter((p) => p.startsWith("say-phone"))).toEqual([]);
});

test("«Статус» shows the real state of the bot, not a constant", async ({ browser }) => {
  const page = await newPage(browser);
  watch(page, "state");
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto(stack.baseUrl);
  await expect(page.locator("#tabbar")).toBeVisible({ timeout: 40_000 });
  await page.locator("#tab-status").click({ force: true });
  await expect(page.locator("#status-view")).toBeVisible();
  await expect(page.locator("#ws-state")).toHaveText("подключено");
  // The fake bot's STATUS says it is in the game.
  await expect(page.locator("#bot-state")).toHaveText("в игре");
  // It keeps running but leaves the game (a lost connection): the tab follows within a few seconds.
  try {
    stack.bot.connected = false;
    await expect(page.locator("#bot-state")).toHaveText("запущен, не в игре", { timeout: 15_000 });
    await page.screenshot({ path: path.join(SHOTS, "5.11-status-connecting.png") });
  } finally {
    stack.bot.connected = true;
  }
  await expect(page.locator("#bot-state")).toHaveText("в игре", { timeout: 15_000 });
  expect(problems.filter((p) => p.startsWith("state:"))).toEqual([]);
});

// Review 5.11 (F1): in a short desktop window the side column's cards are not cut (the bot card keeps its buttons, the roster scrolls
// inside itself) and the chat input under the log stays inside the window; the chat log is what gives way.
test("short desktop windows: no card of the side column is clipped and the chat input stays in view", async ({ browser }) => {
  const page = await newPage(browser);
  watch(page, "short");
  await openGame(page, stack.baseUrl);
  await expect(page.locator("#player-list .player-row")).toHaveCount(5);
  await expect(page.locator("#game-say-mount .say-input")).toBeVisible();
  for (const [w, h] of [[1280, 720], [1280, 800], [1440, 900], [1920, 1080]] as const) {
    await page.setViewportSize({ width: w, height: h });
    await page.waitForTimeout(800);
    const r = await page.evaluate(() => {
      const cards = [...document.querySelectorAll<HTMLElement>(".side-col > .card")].map((c) => ({ cls: c.className, client: c.clientHeight, content: c.scrollHeight }));
      const links = document.querySelector(".gb-links")!.getBoundingClientRect();
      const bot = document.querySelector(".gb-card")!.getBoundingClientRect();
      const mount = document.getElementById("game-say-mount")!.getBoundingClientRect();
      const chat = document.querySelector(".chat-card")!.getBoundingClientRect();
      return { cards, linksInBot: links.bottom <= bot.bottom + 1, mountInChat: mount.bottom <= chat.bottom + 1, mountBottom: mount.bottom, vh: window.innerHeight };
    });
    for (const c of r.cards) expect(c.content, `${w}x${h} ${c.cls} is clipped`).toBeLessThanOrEqual(c.client + 1);
    expect(r.linksInBot, `${w}x${h}: the bot card's buttons`).toBe(true);
    expect(r.mountInChat, `${w}x${h}: the input is inside the chat card`).toBe(true);
    expect(r.mountBottom, `${w}x${h}: the input is in the window`).toBeLessThanOrEqual(r.vh);
    await page.screenshot({ path: path.join(SHOTS, `5.11-side-col-${w}x${h}.png`) });
  }
  expect(problems.filter((p) => p.startsWith("short:"))).toEqual([]);
});
