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
//   3. (task 3.17) «Предсказатель соперника (эксперимент)»: off by default, hidden for the pure fly, sent as `window_model: true` only when on;
//   4. (task 3.20b) «Дожим» offers «ВБ (эксперимент)» (`finish: "wb"`) and «Настоящие ходы соперника от сервера (эксперимент)» sends `preinput: true`
//      only when on, both with honest hints and hidden / refused for the pure fly; the real run with both on (env `BOT_FINISH="wb"` and
//      `BOT_PREINPUT="on"`, the bot's log lines, STATUS, the «Бот» card rows «Дожим» and «Ходы от сервера»), then the defaults.
//
//   5. (task 5.16, D-120) the «Дуэль» preset: one click fills brain «Гибрид», «Дожим: полный», «Без самоубийств (дуэль)» on, «Настоящие ходы соперника
//      от сервера» on, the predictor and the smart wayblock off, sends nothing by itself, leaves server / duration / sparring alone, and the request it then
//      leads to has exactly the old fields; «полный» is labelled for the duel and not for crowds; the «Бот» card's «Машина» rows and warning (load above 6, or
//      fewer than 20 candidates per decision over at least 25 decisions) against scripted STATUS answers, and the real run's STATUS carries `search_window`.
//
//   6. (task 5.17, D-125) «Потоки поиска» (1–4): 1 by default, an honest hint (the numbers of 4.13, the quiet machine only, strength not measured), hidden for
//      the pure fly, `search_threads` sent only above 1; the «Дуэль» preset sets 3 (and the request it leads to carries `search_threads: 3`); the real run with 3
//      (env `BOT_SEARCH_THREADS="3"`, the bot's log line, STATUS `search_threads`, the «Бот» row «Потоки поиска», the «Запуск» detail), then the default 1.
//
//   7. (task 5.18, D-129) «Исправления дуэли» (off / finish / static,finish; `counter` and `all` are not offered): off by default, an honest hint (the numbers of
//      D-121, «работает только в распознанной дуэли; вживую не проверено», the preset leaves it off), hidden for the pure fly, `duel_fixes` sent only when not off;
//      the «Бот» row «Исправления дуэли» against scripted STATUS answers; the real run with `static,finish` (env `BOT_DUEL_FIXES="static,finish"`, the bot's log
//      line, STATUS `duel_fixes`, the row, the «Запуск» detail), then the default off.
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

async function startOnPrivateServer(
  page: Page,
  opts: { wb: "off" | "on"; duel: "off" | "on"; finish?: "off" | "target" | "wb" | "full"; preinput?: "off" | "on"; threads?: "1" | "2" | "3" | "4"; duelFixes?: "off" | "finish" | "static,finish" },
) {
  await openBotTab(page);
  await expect(field(page, "Сервер")).toBeVisible();
  await expect(card(page).locator(".lc-start")).toBeEnabled({ timeout: 30_000 });
  await field(page, "Сервер").selectOption(GAME);
  await expect(field(page, "Сервер").locator("option:checked")).toContainText("Private e2e");
  await field(page, "Мозг").selectOption("hybrid");
  await field(page, "Длительность").selectOption("15m");
  await field(page, "Умный ВБ").selectOption(opts.wb);
  await field(page, "Без самоубийств \\(дуэль\\)").selectOption(opts.duel);
  if (opts.finish) await field(page, "Дожим").selectOption(opts.finish);
  if (opts.preinput) await field(page, "Настоящие ходы соперника от сервера \\(эксперимент\\)").selectOption(opts.preinput);
  if (opts.threads) await field(page, "Потоки поиска").selectOption(opts.threads);
  if (opts.duelFixes) await field(page, "Исправления дуэли").selectOption(opts.duelFixes);
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

test("the card: «Дожим» offers «ВБ (эксперимент)» with an honest hint, and «Настоящие ходы соперника от сервера (эксперимент)» is off by default, hidden for the pure fly, says what it needs, and both are sent only when on", async ({ page }) => {
  test.setTimeout(120_000);
  await login(page);
  // Task 3.20b (D-112) and the `wb` value of task 3.18 (D-114). The POST is answered here: nothing may start.
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
  await field(page, "Мозг").selectOption("hybrid");
  await field(page, "Сервер").selectOption("local");

  // «Дожим»: the new value, in the list after «цель», with a hint that says what it is and that it is below the bar.
  const finish = field(page, "Дожим");
  const finishHint = card(page).locator(".lc-finish-hint");
  expect(await finish.locator("option").allTextContents()).toEqual(["выкл", "цель (рекомендуется)", "ВБ (эксперимент)", "полный (только дуэль 1 на 1)"]);
  expect(await finish.locator("option").evaluateAll((os) => os.map((o) => (o as HTMLOptionElement).value))).toEqual(["off", "target", "wb", "full"]);
  await finish.selectOption("wb");
  await expect(finishHint).toContainText("Эксперимент для игры на ВБ, не для дуэли");
  await expect(finishHint).toContainText("36,1% → 39,7%");
  await expect(finishHint).toContainText("планки (+4,0 п.п.)");
  await expect(finishHint).toContainText("вживую не проверено");
  await expect(finishHint).not.toHaveClass(/lc-finish-warn/);
  await card(page).locator(".lc-start").click();
  await expect.poll(() => posted.length).toBe(1);
  expect(posted[0]).toMatchObject({ action: "start", brain: "hybrid", server: "local", finish: "wb" });
  await finish.selectOption("off");

  // The pre-inputs: off by default, a hint for each state, hidden for the pure fly.
  const pre = field(page, "Настоящие ходы соперника от сервера \\(эксперимент\\)");
  const hint = card(page).locator(".lc-preinput-hint");
  await expect(pre).toBeVisible();
  await expect(pre).toHaveValue("off");
  expect(await pre.locator("option").allTextContents()).toEqual(["выкл", "вкл (эксперимент)"]);
  await expect(hint).toContainText("в предсказании не использует");
  await pre.selectOption("on");
  await expect(hint).toContainText("Помогает, только если сервер присылает эти ходы заранее");
  await expect(hint).toContainText("при запасе предсказания соперника по умолчанию (10 мс) решение почти ничего не узнаёт заранее");
  await expect(hint).toContainText("7,6% пар «снапшот, соперник» (joniTee, 08.10), 8,0% (GER, 07.10) и ≈ 0,2%");
  await expect(hint).toContainText("сколько это даёт в силе, вживую не измерено");
  await expect(hint).not.toContainText("24%");
  await expect(hint).toContainText("серверу ничего не отправляет");
  await page.screenshot({ path: path.join(SHOTS, "3.20b-card-on.png"), fullPage: true });
  await noHorizontalScroll(page);
  await card(page).locator(".lc-start").click();
  await expect.poll(() => posted.length).toBe(2);
  expect(posted[1]).toMatchObject({ action: "start", brain: "hybrid", server: "local", preinput: true });
  expect(Object.keys(posted[1])).not.toContain("finish");
  expect(Object.keys(posted[1])).not.toContain("window_model");
  // Off sends no field at all; the pure fly hides the control and never sends it, even if it was on.
  await pre.selectOption("off");
  await card(page).locator(".lc-start").click();
  await expect.poll(() => posted.length).toBe(3);
  expect(Object.keys(posted[2])).not.toContain("preinput");
  await pre.selectOption("on");
  await field(page, "Мозг").selectOption("fly");
  await expect(pre).toBeHidden();
  await expect(hint).toBeHidden();
  await card(page).locator(".lc-start").click();
  await expect.poll(() => posted.length).toBe(4);
  expect(Object.keys(posted[3])).not.toContain("preinput");
  // Phone width: nothing scrolls sideways with the new hints.
  await field(page, "Мозг").selectOption("hybrid");
  await finish.selectOption("wb");
  await page.setViewportSize({ width: 360, height: 740 });
  await noHorizontalScroll(page);
  await page.screenshot({ path: path.join(SHOTS, "3.20b-card-phone.png"), fullPage: true });
});

test("the real run on the private server: `--finish wb`, `--preinput on`, `--search-threads 3` and `--duel-fixes static,finish` reach the bot (env, log lines, STATUS, both cards), then the defaults", async ({ page }) => {
  test.setTimeout(600_000);
  await login(page);
  expect(muhaIds()).toHaveLength(0);
  const startsBefore = readFileSync(path.join(DIR, "systemctl.log"), "utf8")
    .split("\n")
    .filter((l) => l.startsWith("start ddnet-ai-bot")).length;
  // The bot's log is shared with the earlier real runs of this file (their default is one thread): count against what is there now.
  const oneThreadBefore = count(botLog(), "hybrid search threads: 1 (--search-threads)");
  const threeThreadsBefore = count(botLog(), "hybrid search threads: 3 (--search-threads)");
  const duelFixesBefore = count(botLog(), "duel fixes: static,finish (--duel-fixes; D-121)");

  // 1. both on
  ageHelperState(900);
  await startOnPrivateServer(page, { wb: "off", duel: "off", finish: "wb", preinput: "on", threads: "3", duelFixes: "static,finish" });
  expect(muhaIds()).toHaveLength(1);
  expect(envFile()).toContain('BOT_FINISH="wb"');
  expect(envFile()).toContain('BOT_PREINPUT="on"');
  // Task 5.17 (D-125): the threads reach the bot as the unit's `--search-threads ${BOT_SEARCH_THREADS}`.
  expect(envFile()).toContain('BOT_SEARCH_THREADS="3"');
  await expect.poll(() => botLog(), { timeout: 30_000 }).toContain("hybrid search threads: 3 (--search-threads)");
  await expect.poll(async () => (await botStatus(page))?.search_threads, { timeout: 30_000 }).toBe(3);
  // Task 5.18 (D-129): the duel fixes reach the bot as the unit's `--duel-fixes ${BOT_DUEL_FIXES}` (the comma stays inside one word).
  expect(envFile()).toContain('BOT_DUEL_FIXES="static,finish"');
  await expect.poll(() => botLog(), { timeout: 30_000 }).toContain("duel fixes: static,finish (--duel-fixes; D-121)");
  await expect.poll(async () => (await botStatus(page))?.duel_fixes, { timeout: 30_000 }).toBe("static,finish");
  await expect(page.locator("#bs-duelfixes")).toHaveText("стоячая цель + добивание", { timeout: 15_000 });
  await expect(card(page).locator(".lc-detail")).toContainText("исправления дуэли: стоячая цель и добивание");
  expect(JSON.parse(readFileSync(path.join(DIR, "status", "status.json"), "utf8")).duel_fixes).toBe("static,finish");
  await expect.poll(() => botLog(), { timeout: 30_000 }).toContain("finish blocks: wb (--finish; D-097)");
  await expect.poll(() => botLog(), { timeout: 30_000 }).toContain("pre-inputs: on (--preinput");
  await expect.poll(async () => (await botStatus(page))?.preinput, { timeout: 30_000 }).toBe("on");
  expect((await botStatus(page)).finish).toBe("wb");
  const st = JSON.parse(readFileSync(path.join(DIR, "status", "status.json"), "utf8"));
  expect([st.finish, st.preinput, st.search_threads]).toEqual(["wb", true, 3]);
  await expect(card(page).locator(".lc-detail")).toContainText("потоки поиска: 3");
  await expect(page.locator("#bs-searchthreads")).toHaveText("3", { timeout: 15_000 });
  await expect(card(page).locator(".lc-detail")).toContainText("дожим: ВБ");
  await expect(card(page).locator(".lc-detail")).toContainText("ходы соперника от сервера");
  await expect(page.locator("#bs-finish")).toHaveText("цель + удержание ВБ", { timeout: 15_000 });
  await expect(page.locator("#bs-preinput")).toContainText("вкл", { timeout: 15_000 });
  await expect(page.locator("#bs-preinput")).toContainText("пришло");
  // Task 5.16: the bot's STATUS carries the 30 s search window (additive; no opponent on the private server, so it may well be empty), the site
  // adds the host's load, and the «Машина» rows show both.
  const sw = (await botStatus(page)).search_window;
  expect(sw).toBeTruthy();
  expect(sw.window_s).toBe(30);
  expect(typeof sw.decisions).toBe("number");
  expect(sw.candidates_mean === null || typeof sw.candidates_mean === "number").toBe(true);
  expect(sw.brain_p90_us === null || typeof sw.brain_p90_us === "number").toBe(true);
  expect((sw.decisions === 0) === (sw.candidates_mean === null)).toBe(true);
  const host = (await (await page.request.get(BASE + "/api/bot/status")).json()).host;
  expect(host.load1).toBeGreaterThanOrEqual(0);
  expect(host.cpus).toBeGreaterThanOrEqual(1);
  await expect(page.locator("#bs-load")).toHaveText(/^\d+,\d \/ \d+,\d \/ \d+,\d \(1 \/ 5 \/ 15 мин\), ядер: \d+$/, { timeout: 15_000 });
  await expect(page.locator("#bs-search")).toHaveText(/^(нет решений с поиском за 30 с|\d+,\d кандидата на решение · .*решений: \d+ за 30 с.*)$/, { timeout: 15_000 });
  await page.screenshot({ path: path.join(SHOTS, "3.20b-bot-on.png"), fullPage: true });
  await stopFromPage(page);

  // 2. the defaults: the env says off, the bot counts without playing, the row says so.
  ageHelperState(900);
  await startOnPrivateServer(page, { wb: "off", duel: "off", finish: "off", preinput: "off", threads: "1", duelFixes: "off" });
  expect(envFile()).toContain('BOT_FINISH="off"');
  expect(envFile()).toContain('BOT_DUEL_FIXES="off"');
  await expect.poll(async () => (await botStatus(page))?.duel_fixes, { timeout: 30_000 }).toBe("off");
  await expect(page.locator("#bs-duelfixes")).toHaveText("выкл", { timeout: 15_000 });
  await expect(card(page).locator(".lc-detail")).not.toContainText("исправления дуэли");
  expect(count(botLog(), "duel fixes: static,finish (--duel-fixes; D-121)")).toBe(duelFixesBefore + 1); // the first run's line only
  expect(envFile()).toContain('BOT_PREINPUT="off"');
  expect(envFile()).toContain('BOT_SEARCH_THREADS="1"');
  await expect.poll(async () => (await botStatus(page))?.search_threads, { timeout: 30_000 }).toBe(1);
  await expect(page.locator("#bs-searchthreads")).toHaveText("1", { timeout: 15_000 });
  await expect(card(page).locator(".lc-detail")).not.toContainText("потоки поиска");
  expect(count(botLog(), "hybrid search threads: 3 (--search-threads)")).toBe(threeThreadsBefore + 1); // the first run's line only
  expect(count(botLog(), "hybrid search threads: 1 (--search-threads)")).toBe(oneThreadBefore + 1);
  await expect.poll(async () => (await botStatus(page))?.preinput, { timeout: 30_000 }).toBe("off");
  expect((await botStatus(page)).finish).toBe("off");
  await expect(page.locator("#bs-preinput")).toContainText("выкл (только счёт)", { timeout: 15_000 });
  await expect(card(page).locator(".lc-detail")).not.toContainText("ходы соперника от сервера");
  expect(count(botLog(), "pre-inputs: on (--preinput")).toBe(1); // the first run's line only
  expect(count(botLog(), "finish blocks: wb (--finish; D-097)")).toBe(1);
  await stopFromPage(page);

  const starts = readFileSync(path.join(DIR, "systemctl.log"), "utf8")
    .split("\n")
    .filter((l) => l.startsWith("start ddnet-ai-bot"));
  expect(starts).toHaveLength(startsBefore + 2);
});

test("the card: the «Дуэль» preset fills the form in one click (three search threads included), sends nothing by itself, leaves server / duration / sparring alone, and the request it leads to has only the request's own fields", async ({ page }) => {
  test.setTimeout(120_000);
  await login(page);
  // Task 5.16 (D-120). The POST is answered here: nothing may start.
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

  const preset = card(page).locator(".lc-preset-btn");
  const done = card(page).locator(".lc-preset-done");
  const finishHint = card(page).locator(".lc-finish-hint");
  await expect(preset).toBeVisible();
  await expect(preset).toHaveText("Дуэль");
  await expect(preset).toBeEnabled();
  await expect(card(page).locator(".lc-preset-note")).toContainText("Запускает только кнопка «Запустить»");

  // «полный» is labelled for the duel, with the numbers and the caveats; it is no longer «не рекомендуется».
  const finish = field(page, "Дожим");
  expect(await finish.locator("option").allTextContents()).toEqual(["выкл", "цель (рекомендуется)", "ВБ (эксперимент)", "полный (только дуэль 1 на 1)"]);

  // A form that is as far from the preset as it gets (and server / duration / sparring set to values the preset must not touch).
  await field(page, "Сервер").selectOption("local");
  await field(page, "Мозг").selectOption("hybrid-fly");
  await finish.selectOption("target");
  await field(page, "Умный ВБ").selectOption("on");
  await field(page, "Без самоубийств \\(дуэль\\)").selectOption("off");
  await field(page, "Предсказатель соперника \\(эксперимент\\)").selectOption("on");
  await field(page, "Настоящие ходы соперника от сервера \\(эксперимент\\)").selectOption("on");
  await field(page, "Длительность").selectOption("60m");
  await field(page, "Спарринг \\(только локальный сервер\\)").selectOption("1");
  await field(page, "Предсказание соперника").selectOption("off");
  await field(page, "Потоки поиска").selectOption("4");
  await field(page, "Исправления дуэли").selectOption("static,finish");
  await expect(preset).not.toHaveClass(/current/);
  await expect(preset).toHaveAttribute("aria-pressed", "false");
  await expect(done).toBeHidden();

  await preset.click();
  await expect(field(page, "Мозг")).toHaveValue("hybrid");
  await expect(finish).toHaveValue("full");
  await expect(field(page, "Без самоубийств \\(дуэль\\)")).toHaveValue("on");
  await expect(field(page, "Настоящие ходы соперника от сервера \\(эксперимент\\)")).toHaveValue("off");
  await expect(field(page, "Предсказатель соперника \\(эксперимент\\)")).toHaveValue("off");
  await expect(field(page, "Умный ВБ")).toHaveValue("off");
  await expect(field(page, "Потоки поиска")).toHaveValue("3");
  // Task 5.18 (D-129): the preset leaves the duel fixes off (until a live A/B has been played), also from a form that had them on.
  await expect(field(page, "Исправления дуэли")).toHaveValue("off");
  // Not touched: the server, the duration, the sparring and the hybrid's own opponent model.
  await expect(field(page, "Сервер")).toHaveValue("local");
  await expect(field(page, "Длительность")).toHaveValue("60m");
  await expect(field(page, "Спарринг \\(только локальный сервер\\)")).toHaveValue("1");
  await expect(field(page, "Предсказание соперника")).toHaveValue("off");
  // It filled the form and sent nothing; it says so and lights up while the form IS the preset.
  await expect(done).toContainText("Форма заполнена для дуэли");
  await expect(done).toContainText("само ничего не запускается");
  await expect(preset).toHaveClass(/current/);
  await expect(preset).toHaveAttribute("aria-pressed", "true");
  await page.waitForTimeout(500);
  expect(posted).toHaveLength(0);
  await expect(card(page).locator(".lc-state-text")).toHaveText("Бот остановлен");

  // Every value has its hint, with the numbers and the caveats.
  await expect(finishHint).toContainText("Для дуэли 1 на 1, не для толпы");
  await expect(finishHint).toContainText("+4,3 ± 2,2 п.п. побед (p 0,0001)");
  await expect(finishHint).toContainText("+7,5 ± 4,1 п.п. (600 пар, p 0,0005)");
  await expect(finishHint).toContainText("плечо выбрано после просмотра таблицы");
  await expect(finishHint).toContainText("вживую не проверено");
  await expect(finishHint).toContainText("там берите «цель»");
  await expect(finishHint).not.toContainText("не рекомендуется");
  await expect(finishHint).not.toHaveClass(/lc-finish-warn/);
  await expect(card(page).locator(".lc-selfkill-hint")).toContainText("Для 1vs1 F-DDrace: любая смерть бота даёт очко сопернику");
  await expect(card(page).locator(".lc-preinput-hint")).toContainText("Выключено: ходы соперника сервер присылает");
  await expect(card(page).locator(".lc-model-hint")).toContainText("Предсказатель выключен");
  await expect(card(page).locator(".lc-wb-hint")).toContainText("Умный ВБ выключен");
  await expect(card(page).locator(".lc-search-threads-hint")).toContainText("Три потока: около 38 кандидатов на решение на тихой машине. Это ставит «Дуэль»");
  const info = card(page).locator(".lc-preset-info");
  await expect(info).not.toHaveAttribute("open", "");
  await info.locator("summary").click();
  const items = card(page).locator(".lc-preset-list li");
  await expect(items).toHaveCount(9);
  const text = await card(page).locator(".lc-preset-list").innerText();
  for (const needle of [
    "Мозг: Гибрид",
    "Дожим: полный",
    "+4,3 ± 2,2 п.п. побед (1800 пар, p 0,0001)",
    "+7,5 ± 4,1 п.п. (600 пар)",
    "соперник в арене один, вживую не проверено",
    "Без самоубийств (дуэль): вкл",
    "Настоящие ходы соперника от сервера: выкл",
    "пользы против соперника с запасом по умолчанию (10 мс) нет: вживую ход известен заранее в ~0,2–8% случаев",
    "включать только для опыта",
    "Предсказатель соперника: выкл",
    "Умный ВБ: выкл",
    "Исправления дуэли: выкл",
    "умолчание не меняем, пока вживую не сыграно сравнение плеч",
    "Потоки поиска: 3",
    "кандидатов на решение вживую 24–25 / 29 / 38 / 44 при 1 / 2 / 3 / 4 потоках",
    "Под нагрузкой потоки отнимают процессор у сборок и могут не помочь: выберите 1",
    "Тихая машина",
    "14,2 кандидата на решение при нагрузке 18–30 против ≈ 27 в арене при часах тихой машины (вживую на тихой машине ещё не измерено)",
  ]) {
    expect(text).toContain(needle);
  }
  await page.screenshot({ path: path.join(SHOTS, "5.16-preset.png"), fullPage: true });

  // «Запустить»: the request is exactly the old fields (the off switches are absent, the on ones are the closed values).
  await card(page).locator(".lc-start").click();
  await expect.poll(() => posted.length).toBe(1);
  expect(posted[0]).toEqual({
    action: "start",
    brain: "hybrid",
    server: "local",
    duration: "60m",
    sparring: 1,
    mirror: "off",
    finish: "full",
    no_selfkill: true,
    search_threads: 3,
  });
  expect(Object.keys(posted[0])).not.toContain("duel_fixes");

  // «полный» without the duel switch is flagged as not looking like a duel, and a change by hand takes the preset's mark and message back.
  await field(page, "Без самоубийств \\(дуэль\\)").selectOption("off");
  await expect(finishHint).toHaveClass(/lc-finish-warn/);
  await expect(finishHint).toContainText("похоже, это не дуэль 1 на 1");
  await expect(preset).not.toHaveClass(/current/);
  await expect(done).toBeHidden();
  await expect(done).toHaveText("");
  // Again: the click puts everything back, also from the pure fly (which hides the hybrid's controls).
  await field(page, "Мозг").selectOption("fly");
  await expect(finish).toBeHidden();
  await preset.click();
  await expect(field(page, "Мозг")).toHaveValue("hybrid");
  await expect(finish).toBeVisible();
  await expect(finish).toHaveValue("full");
  await expect(finishHint).not.toHaveClass(/lc-finish-warn/);

  // Phone width: nothing scrolls sideways with the preset, its list and the longest hint.
  await page.setViewportSize({ width: 360, height: 740 });
  await noHorizontalScroll(page);
  await page.screenshot({ path: path.join(SHOTS, "5.16-preset-phone.png"), fullPage: true });
});

test("the card: «Потоки поиска» is 1 by default, says what the threads buy and what they cost, is hidden for the pure fly, and is sent only above 1", async ({ page }) => {
  test.setTimeout(120_000);
  await login(page);
  // Task 5.17 (D-125). The POST is answered here: nothing may start.
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
  await field(page, "Мозг").selectOption("hybrid");
  await field(page, "Сервер").selectOption("local");

  const threads = field(page, "Потоки поиска");
  const hint = card(page).locator(".lc-search-threads-hint");
  await expect(threads).toBeVisible();
  await expect(threads).toHaveValue("1");
  expect(await threads.locator("option").evaluateAll((os) => os.map((o) => (o as HTMLOptionElement).value))).toEqual(["1", "2", "3", "4"]);
  expect(await threads.locator("option").allTextContents()).toEqual(["1 (по умолчанию)", "2", "3", "4"]);
  // The hint of each value carries the numbers of the quiet machine and the cost under load, and does not promise strength.
  const wanted: Record<string, string> = { "1": "24–25", "2": "29", "3": "38", "4": "44" };
  for (const v of ["1", "2", "3", "4"]) {
    await threads.selectOption(v);
    await expect(hint).toContainText("около " + wanted[v] + " кандидатов на решение на тихой машине");
    await expect(hint).toContainText("24–25 / 29 / 38 / 44 при 1 / 2 / 3 / 4 потоках");
    await expect(hint).toContainText("каждый помощник занимает около 5% ядра");
    await expect(hint).toContainText("пул конечен (≈ 55–62 кандидата)");
    await expect(hint).toContainText("Только на тихой машине: под нагрузкой больше потоков отнимает процессор у сборок и может не помочь");
    await expect(hint).toContainText("Больше кандидатов не значит больше побед");
  }
  await threads.selectOption("3");
  await expect(hint).toContainText("Это ставит «Дуэль»");
  await page.screenshot({ path: path.join(SHOTS, "5.17-card-3.png"), fullPage: true });
  await noHorizontalScroll(page);

  // 1 is the absence of the field (an older helper with a strict format still takes a default start); 2 to 4 are JSON integers.
  await threads.selectOption("1");
  await card(page).locator(".lc-start").click();
  await expect.poll(() => posted.length).toBe(1);
  expect(Object.keys(posted[0])).not.toContain("search_threads");
  for (const [i, v] of [[1, 2], [2, 4]] as const) {
    await threads.selectOption(String(v));
    await card(page).locator(".lc-start").click();
    await expect.poll(() => posted.length).toBe(i + 1);
    expect(posted[i]).toMatchObject({ action: "start", brain: "hybrid", server: "local", search_threads: v });
    expect(typeof posted[i].search_threads).toBe("number");
  }
  // The pure fly does not search: the control and its hint are hidden and nothing is sent, even if a value was chosen. Back to the hybrid, it is still chosen.
  await threads.selectOption("4");
  await field(page, "Мозг").selectOption("fly");
  await expect(threads).toBeHidden();
  await expect(hint).toBeHidden();
  await card(page).locator(".lc-start").click();
  await expect.poll(() => posted.length).toBe(4);
  expect(Object.keys(posted[3])).not.toContain("search_threads");
  await field(page, "Мозг").selectOption("hybrid-fly");
  await expect(threads).toBeVisible();
  await expect(threads).toHaveValue("4");
  // A change by hand takes the preset's mark back; the preset sets 3 again.
  await field(page, "Мозг").selectOption("hybrid");
  await card(page).locator(".lc-preset-btn").click();
  await expect(threads).toHaveValue("3");
  await expect(card(page).locator(".lc-preset-btn")).toHaveClass(/current/);
  await threads.selectOption("2");
  await expect(card(page).locator(".lc-preset-btn")).not.toHaveClass(/current/);
  // Phone width: the longest hint does not scroll the page sideways.
  await page.setViewportSize({ width: 360, height: 740 });
  await noHorizontalScroll(page);
  await page.screenshot({ path: path.join(SHOTS, "5.17-card-phone.png"), fullPage: true });
});

/** A STATUS of a live bot, as the site serves it, with the host's load and the bot's search window of the case under test. */
function liveStatusAnswer(host: any, searchWindow: any | undefined, live = true): any {
  const status: any = {
    tick: 5000, own: 0, target: 1, mode: "fight", brain: "hybrid", alive: true, frozen: false, blocks: 1, blocked_by: 0, self_kills: 0,
    decisions: 100, collapsed: 0, decide_p50_us: 800, decide_p99_us: 4100, brain_p99_us: 3900, overhead_p99_us: 200, telemetry: null,
    connected: true, server: "127.0.0.1:8463", map: "Copy Love Box", name: "Muha", clan: "", skin: "default", target_tag: "c1-deadbeef", wb: "WB: off",
    goto: "", deaths: 0, clips_saved: 0, kill_cooldown_ticks: 0, paused: false, finish: "full", selfkill: "off", wb_smart: "off", duel: true,
    window_model: "off", preinput: "on",
  };
  if (searchWindow !== undefined) status.search_window = searchWindow;
  return {
    bridge: true, source: live ? "live" : "none", demo_configured: false, live, age_ms: 100, status: live ? status : null, control_socket: false, host,
  };
}

test("the «Бот» card: the machine's quietness rows and the warning (load above 6, or fewer than 20 candidates per decision), and the same line under the preset", async ({ page }) => {
  test.setTimeout(120_000);
  await login(page);
  // Task 5.16 (D-120). The status answers are scripted here (the real machine's load is not ours to set); the launch routes are left alone.
  let answer: any = null;
  await page.route("**/api/bot/status", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(answer) });
  });
  const sw = (decisions: number, mean: number | null, p90: number | null) => ({ window_s: 30, decisions, candidates_mean: mean, brain_p90_us: p90 });
  const host = (l1: number) => ({ load1: l1, load5: 1.8, load15: 1.5, cpus: 8 });
  const load = page.locator("#bs-load");
  const search = page.locator("#bs-search");
  const warn = page.locator("#bs-quiet-warn");
  const quiet = card(page).locator(".lc-quiet");
  answer = liveStatusAnswer(host(2.1), sw(612, 27.34, 4125));
  await openBotTab(page);

  // 1. quiet: the numbers, no warning. The decimal comma, the p90 in ms, the count of decisions.
  await expect(load).toHaveText("2,1 / 1,8 / 1,5 (1 / 5 / 15 мин), ядер: 8", { timeout: 15_000 });
  await expect(search).toHaveText("27,3 кандидата на решение · p90 решения 4,13 мс · решений: 612 за 30 с");
  await expect(warn).toBeHidden();
  await expect(load).not.toHaveClass(/kv-warn/);
  await expect(search).not.toHaveClass(/kv-warn/);
  await expect(page.locator("#bs-quiet-note")).toContainText("порог 20");
  await expect(page.locator("#bs-quiet-note")).toContainText("в толпе число может быть другим (не измерено)");
  await expect(page.locator("#bs-quiet-note")).toContainText("никуда не отправляются");
  await expect(quiet).toContainText("Машина сейчас: нагрузка 2,1 / 1,8 / 1,5");
  await expect(quiet).toContainText("Поиск бота: 27,3 кандидата на решение");
  await expect(quiet).not.toHaveClass(/warn/);
  await page.screenshot({ path: path.join(SHOTS, "5.16-quiet.png"), fullPage: true });

  // 2. load above 6: the warning says it, with the reason; the load row is flagged, the search row is not.
  answer = liveStatusAnswer(host(7.4), sw(612, 27.3, 4000));
  await expect(warn).toBeVisible({ timeout: 15_000 });
  await expect(warn).toContainText("машина загружена — бот думает хуже");
  await expect(warn).toContainText("нагрузка 7,4 выше 6");
  await expect(warn).not.toContainText("кандидатов");
  await expect(load).toHaveClass(/kv-warn/);
  await expect(search).not.toHaveClass(/kv-warn/);
  await expect(quiet).toHaveClass(/warn/);
  await expect(quiet).toContainText("машина загружена — бот думает хуже");
  await page.screenshot({ path: path.join(SHOTS, "5.16-quiet-load.png"), fullPage: true });

  // 3. few candidates on a quiet machine: the warning names the candidates only.
  answer = liveStatusAnswer(host(1.0), sw(600, 14.2, 6000));
  await expect(warn).toBeVisible({ timeout: 15_000 });
  await expect(warn).toContainText("машина загружена — бот думает хуже");
  await expect(warn).toContainText("кандидатов на решение 14,2 меньше 20");
  await expect(warn).not.toContainText("нагрузка");
  await expect(search).toHaveClass(/kv-warn/);
  await expect(load).not.toHaveClass(/kv-warn/);
  // ... both at once name both.
  answer = liveStatusAnswer(host(18.5), sw(600, 11.0, 9000));
  await expect(warn).toContainText("нагрузка 18,5 выше 6; кандидатов на решение 11,0 меньше 20", { timeout: 15_000 });

  // 4. the boundaries: load 6 and 20 candidates are not warnings (the rule is above 6 and below 20); and a mean over fewer than 25 decisions is not judged.
  answer = liveStatusAnswer(host(6), sw(600, 20, 4000));
  await expect(warn).toBeHidden({ timeout: 15_000 });
  await expect(load).toHaveText(/^6,00 \//);
  answer = liveStatusAnswer(host(1), sw(24, 3.5, 4000));
  await expect(search).toContainText("(мало, не оцениваем)", { timeout: 15_000 });
  await expect(warn).toBeHidden();
  answer = liveStatusAnswer(host(1), sw(25, 19.9, 4000));
  await expect(warn).toBeVisible({ timeout: 15_000 });

  // 5. nothing searched in the last 30 s; a slow bin; a bot of an older build (no `search_window`); an unreadable load.
  answer = liveStatusAnswer(host(1), sw(0, null, null));
  await expect(search).toHaveText("нет решений с поиском за 30 с", { timeout: 15_000 });
  await expect(warn).toBeHidden();
  answer = liveStatusAnswer(host(1), sw(100, 25, 20000));
  await expect(search).toContainText("p90 решения от 20 мс", { timeout: 15_000 });
  answer = liveStatusAnswer(host(1), undefined);
  await expect(search).toHaveText("—", { timeout: 15_000 });
  await expect(warn).toBeHidden();
  // The candidate threshold is the hybrid's: another brain's count is shown and not judged.
  const planner = liveStatusAnswer(host(1), sw(600, 5.0, 4000));
  planner.status.brain = "planner-normal-5ms";
  answer = planner;
  await expect(search).toContainText("(не гибрид: порог кандидатов к этому мозгу не применяется)", { timeout: 15_000 });
  await expect(warn).toBeHidden();
  answer = liveStatusAnswer(null, sw(100, 25, 4000));
  await expect(load).toHaveText("—", { timeout: 15_000 });
  await expect(search).toContainText("25,0 кандидата на решение");
  await expect(quiet).toContainText("Нагрузку машины сайт сейчас прочитать не может");

  // 6. no bot at all: the load is still shown (before pressing «Запустить») and warns; the bot's search is «—».
  answer = liveStatusAnswer(host(9.9), undefined, false);
  await expect(load).toHaveText("9,9 / 1,8 / 1,5 (1 / 5 / 15 мин), ядер: 8", { timeout: 15_000 });
  await expect(search).toHaveText("—");
  await expect(warn).toBeVisible();
  await expect(warn).toContainText("нагрузка 9,9 выше 6");
  await expect(quiet).toContainText("Внимание: машина загружена — бот думает хуже");
  await expect(quiet).not.toContainText("Поиск бота");
  // ... and the judgement is a pure function the page exposes: the same numbers.
  const verdicts = await page.evaluate(() => {
    const q = (window as any).LaunchCard.quietness;
    const sw = (d: number, m: number) => ({ window_s: 30, decisions: d, candidates_mean: m, brain_p90_us: 1000 });
    return [
      q({ load1: 6, cpus: 8 }, sw(100, 20), "hybrid").warn,
      q({ load1: 6.01, cpus: 8 }, sw(100, 20), "hybrid").warn,
      q({ load1: 0, cpus: 8 }, sw(100, 19.99), "hybrid").warn,
      q({ load1: 0, cpus: 8 }, sw(24, 1), "hybrid").warn,
      q({ load1: 0, cpus: 8 }, sw(100, 5), "planner-normal-5ms").warn,
      q({ load1: 0, cpus: 8 }, sw(100, 5)).warn,
      q({ load1: 6.01, cpus: 8 }, null, "hybrid").reasons.join(),
      q(null, null).warn,
      q({ load1: "7" }, null).warn,
    ];
  });
  expect(verdicts).toEqual([false, true, true, false, false, false, "нагрузка 6,01 выше 6", false, false]);

  // Phone width: the rows and the warning do not scroll sideways.
  answer = liveStatusAnswer(host(18.5), sw(600, 11.0, 9000));
  await expect(warn).toBeVisible({ timeout: 15_000 });
  await page.setViewportSize({ width: 360, height: 740 });
  await noHorizontalScroll(page);
  await page.screenshot({ path: path.join(SHOTS, "5.16-quiet-phone.png"), fullPage: true });
});

test("the «Бот» card and the «Запуск» card follow the search threads: the candidate threshold scales with them, the count stands beside the candidates, the fly shows «—», and the thread hint warns under load", async ({ page }) => {
  test.setTimeout(120_000);
  await login(page);
  // Task 5.17 (D-125), review round 1 (F1, F2, F4). The status answers are scripted (the real load is not ours to set).
  let answer: any = null;
  await page.route("**/api/bot/status", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(answer) });
  });
  const sw = (decisions: number, mean: number | null) => ({ window_s: 30, decisions, candidates_mean: mean, brain_p90_us: 4000 });
  const host = (l1: number) => ({ load1: l1, load5: 1.8, load15: 1.5, cpus: 8 });
  const withThreads = (h: any, w: any, threads: any, brain = "hybrid") => {
    const a = liveStatusAnswer(h, w);
    a.status.search_threads = threads;
    a.status.brain = brain;
    return a;
  };
  const search = page.locator("#bs-search");
  const warn = page.locator("#bs-quiet-warn");
  const rowThreads = page.locator("#bs-searchthreads");
  answer = withThreads(host(1), sw(600, 28), 1);
  await openBotTab(page);

  // 1. F2: 28 candidates is fine for one thread (threshold 20) and low for three (31): the same number, another verdict; the count stands beside it.
  await expect(search).toHaveText("28,0 кандидата на решение · p90 решения 4,00 мс · решений: 600 за 30 с · потоков поиска: 1", { timeout: 15_000 });
  await expect(warn).toBeHidden();
  await expect(rowThreads).toHaveText("1");
  answer = withThreads(host(1), sw(600, 28), 3);
  await expect(warn).toBeVisible({ timeout: 15_000 });
  await expect(warn).toContainText("кандидатов на решение 28,0 меньше 31");
  await expect(search).toContainText("потоков поиска: 3");
  await expect(rowThreads).toHaveText("3");
  // The thresholds of 2 and 4 threads, and the edges (the rule is strictly below).
  for (const [n, edge] of [[2, 24], [3, 31], [4, 36]] as const) {
    answer = withThreads(host(1), sw(600, edge), n);
    await expect(search).toContainText("потоков поиска: " + n, { timeout: 15_000 });
    await expect(warn).toBeHidden();
    answer = withThreads(host(1), sw(600, edge - 0.1), n);
    await expect(warn).toContainText("меньше " + edge, { timeout: 15_000 });
  }
  // A bot that does not report the count (older build) or a hand-started 8 keeps the threshold of 20 (and no count is claimed for the unreported one).
  answer = liveStatusAnswer(host(1), sw(600, 22));
  await expect(search).toHaveText("22,0 кандидата на решение · p90 решения 4,00 мс · решений: 600 за 30 с", { timeout: 15_000 });
  await expect(warn).toBeHidden();
  await expect(rowThreads).toHaveText("—");
  answer = withThreads(host(1), sw(600, 22), 8);
  await expect(search).toContainText("потоков поиска: 8", { timeout: 15_000 });
  await expect(warn).toBeHidden();

  // 2. F4: the pure fly does not search: «—» in the row, no thread count beside the candidates.
  answer = withThreads(host(1), sw(600, 22), 1, "fly");
  await expect(rowThreads).toHaveText("—", { timeout: 15_000 });
  await expect(search).not.toContainText("потоков поиска");
  answer = withThreads(host(1), sw(600, 22), 1, "hybrid");
  await expect(rowThreads).toHaveText("1", { timeout: 15_000 });

  // 3. F1: the thread hint of the «Запуск» card. Quiet machine: the research's condition, no load sentence. Loaded machine and more than one thread: the
  // sentence and the warning look; one thread: no sentence. Nothing is blocked either way.
  const hint = card(page).locator(".lc-search-threads-hint");
  const threads = field(page, "Потоки поиска");
  answer = withThreads(host(0.5), sw(600, 40), 3);
  await expect(card(page).locator(".lc-quiet")).toContainText("нагрузка 0,5", { timeout: 15_000 });
  await field(page, "Мозг").selectOption("hybrid");
  await threads.selectOption("3");
  await expect(hint).toContainText("Тихая — это нагрузка < 2 и не меньше 4 свободных ядер");
  await expect(hint).toContainText("медиана, один прогон на значение");
  await expect(hint).not.toContainText("в среднем");
  await expect(hint).not.toContainText("Машина загружена");
  await expect(hint).not.toHaveClass(/warn/);
  answer = withThreads(host(14.1), sw(600, 40), 3);
  await expect(hint).toContainText("Машина загружена: потоки поиска сверх 1 отнимают процессор у сборок и могут не помочь — выберите 1", { timeout: 15_000 });
  await expect(hint).toHaveClass(/warn/);
  await page.screenshot({ path: path.join(SHOTS, "5.17-card-loaded.png"), fullPage: true });
  await threads.selectOption("1");
  await expect(hint).not.toContainText("Машина загружена");
  await expect(hint).not.toHaveClass(/warn/);
  await threads.selectOption("2");
  await expect(hint).toContainText("Машина загружена");
  // The preset still sets 3 under load (nothing is blocked), and the sentence follows.
  await card(page).locator(".lc-preset-btn").click();
  await expect(threads).toHaveValue("3");
  await expect(hint).toContainText("Машина загружена");
  await card(page).locator(".lc-preset-info summary").click();
  await expect(card(page).locator(".lc-preset-list")).toContainText("Тихая машина — нагрузка < 2 и не меньше 4 свободных ядер");
  await expect(card(page).locator(".lc-preset-list")).toContainText("медиана, один прогон на значение");
  // Back to quiet: the sentence goes.
  answer = withThreads(host(0.5), sw(600, 40), 3);
  await expect(hint).not.toContainText("Машина загружена", { timeout: 15_000 });
  await page.setViewportSize({ width: 360, height: 740 });
  await noHorizontalScroll(page);
});

test("the card: «Исправления дуэли» is off by default, offers three arms and no `counter` or `all`, says what the numbers are and are not, is hidden for the pure fly, and is sent only when not off", async ({ page }) => {
  test.setTimeout(120_000);
  await login(page);
  // Task 5.18 (D-129). The POST is answered here: nothing may start.
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
  await field(page, "Мозг").selectOption("hybrid");
  await field(page, "Сервер").selectOption("local");

  const fixes = field(page, "Исправления дуэли");
  const hint = card(page).locator(".lc-duel-fixes-hint");
  await expect(fixes).toBeVisible();
  await expect(fixes).toHaveValue("off");
  expect(await fixes.locator("option").evaluateAll((os) => os.map((o) => (o as HTMLOptionElement).value))).toEqual(["off", "finish", "static,finish"]);
  expect(await fixes.locator("option").allTextContents()).toEqual(["выкл (по умолчанию)", "добивание", "стоячая цель и добивание"]);
  await expect(hint).toContainText("Выключено (умолчание)");
  // The hint of each arm carries the numbers of D-121 and says that it works only in a recognised duel and has not been tried live.
  await fixes.selectOption("finish");
  await expect(hint).toContainText("удержание 57,7% → 90,4% (+32,7 п.п.)");
  await expect(hint).toContainText("+1,7 ± 2,0 п.п. побед, p 0,13; таймауты те же");
  await expect(hint).toContainText("силу это не доказывает");
  await expect(hint).toContainText("Работает только в распознанной дуэли; вживую не проверено");
  await expect(hint).toContainText("Пресет «Дуэль» ставит «выкл»");
  await fixes.selectOption("static,finish");
  await expect(hint).toContainText("планку по букве не взяла (87% при 6 мкс вместо 90%");
  await expect(hint).toContainText("−0,7 ± 3,3 п.п. (p 0,77; задним числом");
  await expect(hint).toContainText("Работает только в распознанной дуэли; вживую не проверено");
  await expect(hint).not.toContainText("counter");
  await page.screenshot({ path: path.join(SHOTS, "5.18-card.png"), fullPage: true });
  await noHorizontalScroll(page);

  // off is the absence of the field (an older helper with a strict format still takes a default start); the other two are the select's own words.
  await fixes.selectOption("off");
  await card(page).locator(".lc-start").click();
  await expect.poll(() => posted.length).toBe(1);
  expect(Object.keys(posted[0])).not.toContain("duel_fixes");
  for (const [i, v] of [[1, "finish"], [2, "static,finish"]] as const) {
    await fixes.selectOption(v);
    await card(page).locator(".lc-start").click();
    await expect.poll(() => posted.length).toBe(i + 1);
    expect(posted[i]).toMatchObject({ action: "start", brain: "hybrid", server: "local", duel_fixes: v });
  }
  // The pure fly has none: the control and its hint are hidden and nothing is sent, even if a value was chosen. Back to the hybrid, it is still chosen.
  await field(page, "Мозг").selectOption("fly");
  await expect(fixes).toBeHidden();
  await expect(hint).toBeHidden();
  await card(page).locator(".lc-start").click();
  await expect.poll(() => posted.length).toBe(4);
  expect(Object.keys(posted[3])).not.toContain("duel_fixes");
  await field(page, "Мозг").selectOption("hybrid-fly");
  await expect(fixes).toBeVisible();
  await expect(fixes).toHaveValue("static,finish");
  // A change by hand takes the preset's mark back, and the preset puts the duel fixes off again.
  await field(page, "Мозг").selectOption("hybrid");
  await card(page).locator(".lc-preset-btn").click();
  await expect(fixes).toHaveValue("off");
  await expect(card(page).locator(".lc-preset-btn")).toHaveClass(/current/);
  await fixes.selectOption("finish");
  await expect(card(page).locator(".lc-preset-btn")).not.toHaveClass(/current/);
  // Phone width: the longest hint does not scroll the page sideways.
  await fixes.selectOption("static,finish");
  await page.setViewportSize({ width: 360, height: 740 });
  await noHorizontalScroll(page);
  await page.screenshot({ path: path.join(SHOTS, "5.18-card-phone.png"), fullPage: true });
});

test("the «Бот» card: the row «Исправления дуэли» names the fixes the hybrid runs with, and is «—» for the pure fly, an older bot and a list it does not know", async ({ page }) => {
  test.setTimeout(120_000);
  await login(page);
  // Task 5.18 (D-129). The status answers are scripted.
  let answer: any = null;
  await page.route("**/api/bot/status", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(answer) });
  });
  const host = { load1: 0.5, load5: 0.5, load15: 0.5, cpus: 8 };
  const withFixes = (fixes: any, brain = "hybrid") => {
    const a = liveStatusAnswer(host, undefined);
    a.status.brain = brain;
    if (fixes !== undefined) a.status.duel_fixes = fixes;
    return a;
  };
  const row = page.locator("#bs-duelfixes");
  answer = withFixes("off");
  await openBotTab(page);
  await expect(row).toHaveText("выкл", { timeout: 15_000 });
  for (const [list, text] of [
    ["finish", "добивание"],
    ["static,finish", "стоячая цель + добивание"],
    ["static,counter,finish", "стоячая цель + хук сверху (не рекомендуется) + добивание"],
  ] as const) {
    answer = withFixes(list);
    await expect(row).toHaveText(text, { timeout: 15_000 });
  }
  answer = withFixes("finish", "hybrid-fly");
  await expect(row).toHaveText("добивание", { timeout: 15_000 });
  answer = withFixes("finish", "fly");
  await expect(row).toHaveText("—", { timeout: 15_000 });
  answer = withFixes(undefined);
  await expect(row).toHaveText("—", { timeout: 15_000 });
  for (const bad of ["all", "static,sometimes", "", 3, null]) {
    answer = withFixes("finish");
    await expect(row).toHaveText("добивание", { timeout: 15_000 });
    answer = withFixes(bad);
    await expect(row).toHaveText("—", { timeout: 15_000 });
  }
});
