// Real-browser end-to-end of the «Умный ВБ» (`--wb-smart`) and «Без самоубийств (дуэль)» (`--no-selfkill`) controls on the «Запуск» card
// (task 5.15; tasks 3.12/3.12b D-103/D-104 and 4.11 D-102), on THIS machine, against a PRIVATE DDNet server on 127.0.0.1:8463 only (never
// 8303, never a public server). `tools/e2e/options-e2e.sh` builds the stack this spec drives, like the one of launch-finish.spec.ts (task
// 5.13): a test web instance (a ddnet-ai with the loopback-favourites feature) with the private server as its one favourite, and the helper
// and the bot through the TEST path (launcher-sim.mjs runs the REAL `ddnet-ai launch apply`, a fake systemctl, run-bot.sh = the bot unit's
// command line with `--wb-smart ${BOT_WB_SMART} --no-selfkill=${BOT_NO_SELFKILL}`). Nothing of the production units, /etc, Caddy or
// ~/aiddnet/data/bot is used.
//
// What it checks:
//   1. the card: both controls exist and are «выкл» by default with their hints; the duel switch's hint turns into a warning while it is
//      on and says «для 1vs1 F-DDrace ... на обычных серверах не включать»; the request the page sends carries `wb_smart: "on"` /
//      `no_selfkill: true` only when they are on (nothing when off, so an old helper still takes a default start); the pure fly keeps both
//      controls (they are not the brain's) while the finishing control is hidden for it; the POST is answered by the test itself, so that no
//      bot is ever started on 8303;
//   2. the real run on the private favourite: both on -> the helper's env file has BOT_WB_SMART="on" and BOT_NO_SELFKILL="true", the bot's
//      own log says «wb smart: on» and «self-kill: off (flag)», its STATUS (`/api/bot/status`) says `wb_smart: on` and `selfkill: off`, and
//      the «Бот» and «Запуск» cards show it; then the defaults (nothing new in the log, STATUS off / on), then the duel switch alone.
//
// Needs the env of options-e2e.sh (else it skips). Screenshots: <scratch dir of the run>/screenshots/5.15-*.png (never in git).

import { test, expect, type Dialog, type Page } from "@playwright/test";
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const BASE = process.env.DDAI_OPTIONS_E2E_URL ?? "";
const PASSWORD_FILE = process.env.DDAI_OPTIONS_E2E_PASSWORD_FILE ?? "";
const DIR = process.env.DDAI_OPTIONS_E2E_DIR ?? "";
const GAME = process.env.DDAI_OPTIONS_E2E_GAME_ADDR ?? "127.0.0.1:8463";
const ECON_PORT = process.env.DDAI_OPTIONS_E2E_ECON_PORT ?? "8464";
const ECON_CFG = process.env.DDAI_OPTIONS_E2E_ECON_CFG ?? "";
// Next to the run's own scratch data, never over screenshots of an earlier run (a copy for the report is a manual step).
const SHOTS = path.join(DIR || homedir(), "screenshots");
const ECON = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "ddnet-server", "econ.py");

test.skip(!BASE || !PASSWORD_FILE || !DIR || !existsSync(PASSWORD_FILE), "run tools/e2e/options-e2e.sh (see README.md)");
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

test("the card: «Умный ВБ» and «Без самоубийств (дуэль)» are off by default, have their hints, the duel one warns while on, and the request carries them only when on", async ({ page }) => {
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

  const wb = field(page, "Умный ВБ");
  const duel = field(page, "Без самоубийств \\(дуэль\\)");
  const wbHint = card(page).locator(".lc-wb-hint");
  const duelHint = card(page).locator(".lc-selfkill-hint");
  await expect(wb).toBeVisible();
  await expect(duel).toBeVisible();
  await expect(wb).toHaveValue("off");
  await expect(duel).toHaveValue("off");
  expect(await wb.locator("option").allTextContents()).toEqual(["выкл", "вкл"]);
  expect(await duel.locator("option").allTextContents()).toEqual(["выкл", "вкл (дуэль)"]);
  await expect(wbHint).toContainText("Умный ВБ выключен");
  await expect(duelHint).toContainText("вкл (дуэль)");
  // Off: one short line each (the long texts are for the «вкл» state, so the phone view stays short).
  expect((await wbHint.innerText()).length).toBeLessThan(80);
  expect((await duelHint.innerText()).length).toBeLessThan(80);
  await expect(duelHint).not.toHaveClass(/warn/);

  // On: the hints say what the options do; the duel switch's turns into a warning (the design system's `.hint.warn`).
  await wb.selectOption("on");
  await expect(wbHint).toContainText("только если он мешает");
  await expect(wbHint).toContainText("по числу целей");
  await expect(wbHint).toContainText("Переходы труб");
  await expect(wbHint).toContainText("не бродит, а сразу начинает путь");
  await duel.selectOption("on");
  await expect(duelHint).toHaveClass(/warn/);
  await expect(duelHint).toContainText("Для 1vs1 F-DDrace: любая смерть бота даёт очко сопернику");
  await expect(duelHint).toContainText("На обычных серверах не включать");
  // The «Убить» button of «Команды» works under the switch too (KillWhy::Console), so the hint names it first.
  await expect(duelHint).toContainText("Убить бота можно только вручную: кнопкой «Убить» в «Командах» (или строкой /kill на вкладке «Игра»)");
  await page.screenshot({ path: path.join(SHOTS, "5.15-card-on.png"), fullPage: true });

  // Both on, on the local server (the POST is answered here): the request carries both, and only them.
  await field(page, "Мозг").selectOption("hybrid");
  await field(page, "Сервер").selectOption("local");
  await card(page).locator(".lc-start").click();
  await expect.poll(() => posted.length).toBe(1);
  expect(posted[0]).toMatchObject({ action: "start", brain: "hybrid", server: "local", wb_smart: "on", no_selfkill: true });
  expect(Object.keys(posted[0])).not.toContain("finish");

  // Off sends neither field at all (no field = off): a helper that does not know them yet still takes the default start.
  await wb.selectOption("off");
  await duel.selectOption("off");
  await expect(duelHint).not.toHaveClass(/warn/);
  await card(page).locator(".lc-start").click();
  await expect.poll(() => posted.length).toBe(2);
  expect(Object.keys(posted[1])).not.toContain("wb_smart");
  expect(Object.keys(posted[1])).not.toContain("no_selfkill");

  // Each alone.
  await wb.selectOption("on");
  await card(page).locator(".lc-start").click();
  await expect.poll(() => posted.length).toBe(3);
  expect(posted[2]).toMatchObject({ wb_smart: "on" });
  expect(Object.keys(posted[2])).not.toContain("no_selfkill");
  await wb.selectOption("off");
  await duel.selectOption("on");
  await card(page).locator(".lc-start").click();
  await expect.poll(() => posted.length).toBe(4);
  expect(posted[3]).toMatchObject({ no_selfkill: true });
  expect(Object.keys(posted[3])).not.toContain("wb_smart");

  // The pure fly: neither option is the brain's, so both stay (and are sent); the finishing control, which is the hybrid's, goes away.
  await wb.selectOption("on");
  await field(page, "Мозг").selectOption("fly");
  await expect(field(page, "Дожим")).toBeHidden();
  await expect(wb).toBeVisible();
  await expect(duel).toBeVisible();
  await expect(wbHint).toBeVisible();
  await expect(duelHint).toBeVisible();
  await card(page).locator(".lc-start").click();
  await expect.poll(() => posted.length).toBe(5);
  expect(posted[4]).toMatchObject({ brain: "fly", wb_smart: "on", no_selfkill: true });
  expect(Object.keys(posted[4])).not.toContain("finish");

  // Phone width: no horizontal scroll with both controls and their (longest) hints.
  await page.setViewportSize({ width: 360, height: 740 });
  await noHorizontalScroll(page);
  await page.screenshot({ path: path.join(SHOTS, "5.15-card-phone.png"), fullPage: true });
});

async function startOnPrivateServer(page: Page, opts: { wb: "off" | "on"; duel: "off" | "on" }) {
  await openBotTab(page);
  await expect(field(page, "Сервер")).toBeVisible();
  await expect(card(page).locator(".lc-start")).toBeEnabled({ timeout: 30_000 });
  await field(page, "Сервер").selectOption(GAME);
  await expect(field(page, "Сервер").locator("option:checked")).toContainText("Private e2e");
  await field(page, "Мозг").selectOption("hybrid");
  await field(page, "Длительность").selectOption("15m");
  await field(page, "Умный ВБ").selectOption(opts.wb);
  await field(page, "Без самоубийств \\(дуэль\\)").selectOption(opts.duel);
  // The site takes at most 6 requests a minute (4 starts and 4 stops come close): when the page says «Слишком часто», wait and click again.
  const accept = (d: Dialog) => void d.accept();
  page.on("dialog", accept);
  const result = card(page).locator(".lc-result");
  for (let attempt = 0; attempt < 8; attempt += 1) {
    await card(page).locator(".lc-start").click();
    await expect(result).not.toHaveText(/…$/);
    if (!/Слишком часто/.test(await result.innerText())) break;
    await page.waitForTimeout(10_000);
  }
  page.off("dialog", accept);
  await expect(result).toContainText("запрос отправлен");
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

const count = (text: string, needle: string) => text.split(needle).length - 1;

test("the real run on the private server: `--wb-smart on` and `--no-selfkill=true` reach the bot (env, log lines, STATUS, both cards), then the defaults, then the duel switch alone", async ({ page }) => {
  test.setTimeout(600_000);
  await login(page);
  expect(muhaIds()).toHaveLength(0);

  // 1. both on
  await startOnPrivateServer(page, { wb: "on", duel: "on" });
  expect(muhaIds()).toHaveLength(1);
  expect(envFile()).toContain(`BOT_SERVER="${GAME}"`);
  expect(envFile()).toContain('BOT_WB_SMART="on"');
  expect(envFile()).toContain('BOT_NO_SELFKILL="true"');
  expect(readFileSync(path.join(DIR, "systemctl.log"), "utf8")).toContain("start ddnet-ai-bot.service");
  // The bot's own log lines and its STATUS.
  await expect.poll(() => botLog(), { timeout: 30_000 }).toContain("wb smart: on");
  await expect.poll(() => botLog(), { timeout: 30_000 }).toContain("self-kill: off (flag)");
  await expect.poll(async () => (await botStatus(page))?.wb_smart, { timeout: 30_000 }).toBe("on");
  expect((await botStatus(page)).selfkill).toBe("off");
  // The helper's status names both, the «Запуск» card shows them, the «Бот» card has the rows (the duel row as a warning).
  const st = JSON.parse(readFileSync(path.join(DIR, "status", "status.json"), "utf8"));
  expect([st.wb_smart, st.no_selfkill]).toEqual(["on", true]);
  await expect(card(page).locator(".lc-detail")).toContainText("умный ВБ");
  await expect(card(page).locator(".lc-detail")).toContainText("без самоубийств");
  await expect(page.locator("#bs-wbsmart")).toHaveText("вкл", { timeout: 15_000 });
  await expect(page.locator("#bs-selfkill")).toHaveText("выкл (флажок)", { timeout: 15_000 });
  await expect(page.locator("#bs-selfkill")).toHaveClass(/kv-warn/);
  await page.screenshot({ path: path.join(SHOTS, "5.15-bot-on.png"), fullPage: true });
  await stopFromPage(page);

  // 2. the controls' defaults: off. The same lines are not printed again, STATUS says off / on, the rows say so.
  ageHelperState(900);
  await startOnPrivateServer(page, { wb: "off", duel: "off" });
  expect(envFile()).toContain('BOT_WB_SMART="off"');
  expect(envFile()).toContain('BOT_NO_SELFKILL="false"');
  await expect.poll(async () => (await botStatus(page))?.wb_smart, { timeout: 30_000 }).toBe("off");
  expect((await botStatus(page)).selfkill).toBe("on");
  await expect(page.locator("#bs-wbsmart")).toHaveText("выкл", { timeout: 15_000 });
  await expect(page.locator("#bs-selfkill")).toHaveText("вкл", { timeout: 15_000 });
  await expect(page.locator("#bs-selfkill")).not.toHaveClass(/kv-warn/);
  expect(count(botLog(), "wb smart: on")).toBe(1); // the first run's line only
  expect(count(botLog(), "self-kill: off (flag)")).toBe(1);
  await expect(card(page).locator(".lc-detail")).not.toContainText("умный ВБ");
  await expect(card(page).locator(".lc-detail")).not.toContainText("без самоубийств");
  await stopFromPage(page);

  // 3. the duel switch alone: independent of the smart wayblock.
  ageHelperState(900);
  await startOnPrivateServer(page, { wb: "off", duel: "on" });
  expect(envFile()).toContain('BOT_WB_SMART="off"');
  expect(envFile()).toContain('BOT_NO_SELFKILL="true"');
  await expect.poll(() => count(botLog(), "self-kill: off (flag)"), { timeout: 30_000 }).toBe(2);
  await expect.poll(async () => (await botStatus(page))?.selfkill, { timeout: 30_000 }).toBe("off");
  expect((await botStatus(page)).wb_smart).toBe("off");
  expect(count(botLog(), "wb smart: on")).toBe(1);
  await stopFromPage(page);

  // 4. the smart wayblock alone.
  ageHelperState(900);
  await startOnPrivateServer(page, { wb: "on", duel: "off" });
  expect(envFile()).toContain('BOT_WB_SMART="on"');
  expect(envFile()).toContain('BOT_NO_SELFKILL="false"');
  await expect.poll(() => count(botLog(), "wb smart: on"), { timeout: 30_000 }).toBe(2);
  await expect.poll(async () => (await botStatus(page))?.wb_smart, { timeout: 30_000 }).toBe("on");
  expect((await botStatus(page)).selfkill).toBe("on");
  expect(count(botLog(), "self-kill: off (flag)")).toBe(2);
  await stopFromPage(page);

  // Four starts, no more; the bot never ran anywhere but the private server (the helper's own list of what it started).
  const starts = readFileSync(path.join(DIR, "systemctl.log"), "utf8")
    .split("\n")
    .filter((l) => l.startsWith("start ddnet-ai-bot"));
  expect(starts).toHaveLength(4);
  expect(readFileSync(path.join(DIR, "etc", "50-launch.conf"), "utf8")).not.toContain("IPAddressDeny");
});

test("the card: «Предсказатель соперника (эксперимент)» is off by default, hidden for the pure fly, says what it is, and is sent only when on", async ({ page }) => {
  test.setTimeout(120_000);
  await login(page);
  // Task 3.17 (D-111). The POST is answered here: nothing may start (the placeholder file is not a model).
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

  const model = field(page, "Предсказатель соперника \\(эксперимент\\)");
  const hint = card(page).locator(".lc-model-hint");
  await expect(model).toBeVisible();
  await expect(model).toHaveValue("off");
  expect(await model.locator("option").allTextContents()).toEqual(["выкл", "вкл (эксперимент)"]);
  await expect(hint).toContainText("Предсказатель выключен");
  // On: the hint says what it is and is honest about the evidence.
  await field(page, "Мозг").selectOption("hybrid");
  await model.selectOption("on");
  await expect(hint).toContainText("маленькая сеть предсказывает ввод соперника");
  await expect(hint).toContainText("на живых клипах при окне 2 пользы не видно");
  await expect(hint).toContainText("предохранитель сам отключает её");
  await page.screenshot({ path: path.join(SHOTS, "3.17-card-on.png"), fullPage: true });
  await noHorizontalScroll(page);

  await field(page, "Сервер").selectOption("local");
  await card(page).locator(".lc-start").click();
  await expect.poll(() => posted.length).toBe(1);
  expect(posted[0]).toMatchObject({ action: "start", brain: "hybrid", server: "local", window_model: true });
  // The request names no file.
  expect(JSON.stringify(posted[0])).not.toContain("oppnet");

  // Off sends no field at all; the pure fly hides the control and never sends it, even if it was on.
  await model.selectOption("off");
  await card(page).locator(".lc-start").click();
  await expect.poll(() => posted.length).toBe(2);
  expect(Object.keys(posted[1])).not.toContain("window_model");
  await model.selectOption("on");
  await field(page, "Мозг").selectOption("fly");
  await expect(model).toBeHidden();
  await expect(hint).toBeHidden();
  await card(page).locator(".lc-start").click();
  await expect.poll(() => posted.length).toBe(3);
  expect(Object.keys(posted[2])).not.toContain("window_model");
});
