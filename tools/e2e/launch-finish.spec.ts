// Real-browser end-to-end of the «Дожим» (finishing) control on the «Запуск» card (task 5.13, D-097), on THIS machine, against a PRIVATE DDNet
// server on 127.0.0.1:8463 only (never 8303, never a public server). `tools/e2e/finish-e2e.sh` builds the stack this spec drives: a test web
// instance (a ddnet-ai with the loopback-favourites feature) with the private server as its one favourite, and the helper and the bot
// through the TEST path (launcher-sim.mjs runs the REAL `ddnet-ai launch apply`, a fake systemctl, run-bot.sh = the bot unit's command line
// with `--finish ${BOT_FINISH}`). Nothing of the production units, /etc, Caddy or ~/aiddnet/data/bot is used.
//
// What it checks:
//   1. the card: the control exists, is «выкл» by default, offers «цель (рекомендуется)» and «полный (не рекомендуется)» with a hint that
//      says so, is hidden for the pure fly, and the request the page sends carries `finish` (and none for the fly) — on the local server
//      choice too, with the POST answered by the test itself so that no bot is ever started on 8303;
//   2. the real run on the private favourite: `target` -> the helper's env file has BOT_FINISH="target", the bot's own log line says it,
//      the bot's STATUS (`/api/bot/status`) says `target`, and the «Бот» card and the «Запуск» card show it; then `off` (the default of
//      the control) and `full`;
//   3. the pure fly with finishing on, as a request file: the helper refuses it (`finish_hybrid_only`) and the card says why.
//
// Needs the env of finish-e2e.sh (else it skips). Screenshots: <scratch dir of the run>/screenshots/5.13-*.png (never in git).

import { test, expect, type Page } from "@playwright/test";
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const BASE = process.env.DDAI_FINISH_E2E_URL ?? "";
const PASSWORD_FILE = process.env.DDAI_FINISH_E2E_PASSWORD_FILE ?? "";
const DIR = process.env.DDAI_FINISH_E2E_DIR ?? "";
const GAME = process.env.DDAI_FINISH_E2E_GAME_ADDR ?? "127.0.0.1:8463";
const ECON_PORT = process.env.DDAI_FINISH_E2E_ECON_PORT ?? "8464";
const ECON_CFG = process.env.DDAI_FINISH_E2E_ECON_CFG ?? "";
// Next to the run's own scratch data, never over screenshots of an earlier run (a copy for the report is a manual step).
const SHOTS = path.join(DIR || homedir(), "screenshots");
const ECON = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "ddnet-server", "econ.py");

test.skip(!BASE || !PASSWORD_FILE || !DIR || !existsSync(PASSWORD_FILE), "run tools/e2e/finish-e2e.sh (see README.md)");
test.describe.configure({ mode: "serial" });

function econ(...cmd: string[]): string {
  return execFileSync("python3", [ECON, "--host", "127.0.0.1", "--port", ECON_PORT, "--password-file", ECON_CFG, ...cmd], { encoding: "utf8" });
}

function muhaIds(): string[] {
  return econ("status")
    .split("\n")
    .filter((l) => /name='Muha'/.test(l))
    .map((l) => /id=(\d+)/.exec(l)?.[1] ?? "");
}

/** The helper's memory: the real-time start interval is over (the ban memory, empty here, is untouched). */
function ageHelperState(seconds: number) {
  const p = path.join(DIR, "var", "state.json");
  const st = JSON.parse(readFileSync(p, "utf8"));
  st.last_start_at = Math.max(0, (st.last_start_at ?? 0) - seconds);
  if (st.last_exit) st.last_exit.at = Math.max(0, st.last_exit.at - seconds);
  writeFileSync(p, JSON.stringify(st));
}

async function login(page: Page) {
  await page.goto(BASE + "/");
  await page.locator("#password").fill(readFileSync(PASSWORD_FILE, "utf8").trim());
  await page.locator('#login-form button[type="submit"]').click();
  await expect(page.locator("#tabbar")).toBeVisible();
}

async function openBotTab(page: Page) {
  await page.locator("#tab-bot").click();
  await expect(page.locator("#bot-view")).toBeVisible();
}

const card = (page: Page) => page.locator(".launch-card");
const field = (page: Page, label: string) =>
  card(page).locator("label.lc-field", { hasText: new RegExp("^" + label) }).locator("select");

async function noHorizontalScroll(page: Page) {
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(0);
}

/** The bot's STATUS as the site serves it. */
async function botStatus(page: Page): Promise<any> {
  const r = await page.request.get(BASE + "/api/bot/status");
  expect(r.status()).toBe(200);
  return (await r.json()).status;
}

const envFile = () => readFileSync(path.join(DIR, "etc", "bot-launch.env"), "utf8");
const botLog = () => readFileSync(path.join(DIR, "bot.log"), "utf8");

test.beforeAll(() => mkdirSync(SHOTS, { recursive: true }));

test("the card: «Дожим» is off by default, offers «цель» (recommended) and «полный» (not recommended), is hidden for the fly, and the request carries it", async ({ page }) => {
  test.setTimeout(120_000);
  await login(page);
  // Answer the POST here: nothing may start on the local server's address (8303 belongs to the production server).
  const posted: any[] = [];
  await page.route("**/api/bot/launch", async (route) => {
    if (route.request().method() === "POST") {
      posted.push(JSON.parse(route.request().postData() ?? "{}"));
      await route.fulfill({ status: 429, contentType: "application/json", body: JSON.stringify({ error: "rate_limited" }) });
    } else {
      await route.continue();
    }
  });
  await openBotTab(page);
  await expect(card(page).locator(".lc-state-text")).toHaveText("Бот остановлен", { timeout: 20_000 });

  const finish = field(page, "Дожим");
  await expect(finish).toBeVisible();
  await expect(finish).toHaveValue("off");
  expect(await finish.locator("option").allTextContents()).toEqual(["выкл", "цель (рекомендуется)", "полный (не рекомендуется)"]);
  const hint = card(page).locator(".lc-finish-hint");
  await expect(hint).toContainText("Дожим выключен");
  await expect(hint).not.toHaveClass(/lc-finish-warn/);
  await finish.selectOption("target");
  await expect(hint).toContainText("Рекомендуется");
  await page.screenshot({ path: path.join(SHOTS, "5.13-card-target.png"), fullPage: true });
  await finish.selectOption("full");
  await expect(hint).toContainText("Не рекомендуется");
  await expect(hint).toHaveClass(/lc-finish-warn/);
  await page.screenshot({ path: path.join(SHOTS, "5.13-card-full.png"), fullPage: true });

  // The fly has no finishing: the control and its hint are hidden, and the request has no `finish`.
  await field(page, "Мозг").selectOption("hybrid");
  await finish.selectOption("target");
  await field(page, "Сервер").selectOption("local");
  await card(page).locator(".lc-start").click();
  await expect.poll(() => posted.length).toBe(1);
  expect(posted[0]).toMatchObject({ action: "start", brain: "hybrid", server: "local", finish: "target" });

  // «выкл» sends no `finish` at all (no field = off): a helper that does not know the field yet still takes the default start.
  await finish.selectOption("off");
  await card(page).locator(".lc-start").click();
  await expect.poll(() => posted.length).toBe(2);
  expect(posted[1]).toMatchObject({ action: "start", brain: "hybrid", server: "local" });
  expect(Object.keys(posted[1])).not.toContain("finish");

  // The pure fly: the control and its hint go away (the e2e data directory has a placeholder bundle file so that the fly is offered; it is
  // never run here).
  await field(page, "Мозг").selectOption("fly");
  await expect(finish).toBeHidden();
  await expect(hint).toBeHidden();
  await expect(field(page, "Предсказание соперника")).toBeHidden(); // hidden for the fly since 5.9, but `display: flex` kept it on screen until 5.13
  await card(page).locator(".lc-start").click();
  await expect.poll(() => posted.length).toBe(3);
  expect(posted[2].brain).toBe("fly");
  expect(Object.keys(posted[2])).not.toContain("finish");
  expect(Object.keys(posted[2])).not.toContain("mirror");

  // Back to a hybrid brain: the control returns with the value it had.
  await field(page, "Мозг").selectOption("hybrid");
  await expect(finish).toBeVisible();
  await expect(finish).toHaveValue("off");

  // Phone width: no horizontal scroll with the control and its (longest) hint.
  await page.setViewportSize({ width: 360, height: 740 });
  await finish.selectOption("full");
  await noHorizontalScroll(page);
  await page.screenshot({ path: path.join(SHOTS, "5.13-card-phone.png"), fullPage: true });
});

async function startOnPrivateServer(page: Page, finish: "off" | "target" | "full") {
  await openBotTab(page);
  await expect(field(page, "Сервер")).toBeVisible();
  await expect(card(page).locator(".lc-start")).toBeEnabled({ timeout: 30_000 });
  await field(page, "Сервер").selectOption(GAME);
  await expect(field(page, "Сервер").locator("option:checked")).toContainText("Private e2e");
  await field(page, "Мозг").selectOption("hybrid");
  await field(page, "Длительность").selectOption("15m");
  await field(page, "Дожим").selectOption(finish);
  page.once("dialog", (d) => d.accept());
  await card(page).locator(".lc-start").click();
  await expect(card(page).locator(".lc-result")).toContainText("запрос отправлен");
  await expect(card(page).locator(".lc-state-text")).toHaveText("В игре", { timeout: 150_000 });
}

async function stopFromPage(page: Page) {
  // The site takes a request at most every 2 s and the bot can be in the game within a second of the start: a stop clicked inside that gap
  // is answered 429 and nothing is written. Wait the gap out, and click again if the page still says «Слишком часто».
  await page.waitForTimeout(2_500);
  for (let attempt = 0; attempt < 4; attempt += 1) {
    await card(page).locator(".lc-stop").click();
    const result = card(page).locator(".lc-result");
    await expect(result).not.toHaveText(/…$/);
    if (!/Слишком часто/.test(await result.innerText())) break;
    await page.waitForTimeout(2_500);
  }
  await expect(card(page).locator(".lc-state-text")).toHaveText("Бот остановлен", { timeout: 60_000 });
  expect(muhaIds()).toHaveLength(0);
}

test("the real run on the private server: `--finish target` reaches the bot (env, log line, STATUS, both cards), then off and full", async ({ page }) => {
  test.setTimeout(600_000);
  await login(page);
  expect(muhaIds()).toHaveLength(0);

  // 1. target
  await startOnPrivateServer(page, "target");
  expect(muhaIds()).toHaveLength(1);
  expect(envFile()).toContain(`BOT_SERVER="${GAME}"`);
  expect(envFile()).toContain('BOT_FINISH="target"');
  const sysLog = readFileSync(path.join(DIR, "systemctl.log"), "utf8");
  expect(sysLog).toContain("start ddnet-ai-bot.service");
  // The bot's own log line and its STATUS.
  await expect.poll(() => botLog(), { timeout: 30_000 }).toContain("finish blocks: target");
  await expect.poll(async () => (await botStatus(page))?.finish, { timeout: 30_000 }).toBe("target");
  // The helper's status names it, the «Запуск» card shows it, the «Бот» card has the row.
  expect(JSON.parse(readFileSync(path.join(DIR, "status", "status.json"), "utf8")).finish).toBe("target");
  await expect(card(page).locator(".lc-detail")).toContainText("дожим: цель");
  await expect(page.locator("#bs-finish")).toHaveText("цель", { timeout: 15_000 });
  await page.screenshot({ path: path.join(SHOTS, "5.13-bot-target.png"), fullPage: true });
  await stopFromPage(page);

  // 2. the control's default: off. No log line is printed for off, STATUS says off, the row says «выкл».
  ageHelperState(900);
  await startOnPrivateServer(page, "off");
  expect(envFile()).toContain('BOT_FINISH="off"');
  await expect.poll(async () => (await botStatus(page))?.finish, { timeout: 30_000 }).toBe("off");
  await expect(page.locator("#bs-finish")).toHaveText("выкл", { timeout: 15_000 });
  expect(botLog().match(/finish blocks: /g)).toHaveLength(1); // the first run's line only
  await expect(card(page).locator(".lc-detail")).not.toContainText("дожим");
  await stopFromPage(page);

  // 3. full (offered behind «не рекомендуется»): STATUS says full.
  ageHelperState(900);
  await startOnPrivateServer(page, "full");
  expect(envFile()).toContain('BOT_FINISH="full"');
  await expect.poll(() => botLog(), { timeout: 30_000 }).toContain("finish blocks: full");
  await expect.poll(async () => (await botStatus(page))?.finish, { timeout: 30_000 }).toBe("full");
  await expect(page.locator("#bs-finish")).toContainText("полный", { timeout: 15_000 });
  await page.screenshot({ path: path.join(SHOTS, "5.13-bot-full.png"), fullPage: true });
  await stopFromPage(page);

  // Three starts, no more; the bot never ran anywhere but the private server (the helper's own list of what it started).
  const starts = readFileSync(path.join(DIR, "systemctl.log"), "utf8")
    .split("\n")
    .filter((l) => l.startsWith("start ddnet-ai-bot"));
  expect(starts).toHaveLength(3);
  expect(readFileSync(path.join(DIR, "etc", "50-launch.conf"), "utf8")).not.toContain("IPAddressDeny");
});

test("the pure fly with finishing on is refused by the helper (the request file the page never sends) and the card says why", async ({ page }) => {
  test.setTimeout(120_000);
  await login(page);
  ageHelperState(900);
  await openBotTab(page);
  await expect(card(page).locator(".lc-state-text")).toHaveText("Бот остановлен", { timeout: 20_000 });
  const before = readFileSync(path.join(DIR, "systemctl.log"), "utf8");
  const req = {
    v: 1,
    id: "00000000000000ff",
    ts: Math.floor(Date.now() / 1000),
    action: "start",
    brain: "fly",
    server: GAME,
    duration: "15m",
    sparring: 0,
    finish: "target",
  };
  writeFileSync(path.join(DIR, "data", "launch", "request.json"), JSON.stringify(req), { mode: 0o644 });
  await expect(card(page).locator(".lc-state-text")).toHaveText("Запрос отклонён", { timeout: 30_000 });
  await expect(card(page).locator(".lc-detail")).toContainText("Дожим бывает только у гибридных мозгов");
  const status = JSON.parse(readFileSync(path.join(DIR, "status", "status.json"), "utf8"));
  expect(status).toMatchObject({ state: "refused", reason: "finish_hybrid_only" });
  // Nothing was started or changed.
  expect(readFileSync(path.join(DIR, "systemctl.log"), "utf8")).toBe(before);
  expect(muhaIds()).toHaveLength(0);
  await page.screenshot({ path: path.join(SHOTS, "5.13-fly-refused.png"), fullPage: true });
});
