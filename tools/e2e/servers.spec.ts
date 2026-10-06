// Real-browser end-to-end of the server browser (task 5.12, D-099), on THIS machine, against a PRIVATE DDNet server on 127.0.0.1:8463 only.
// `tools/e2e/servers-e2e.sh` builds the stack this spec drives: a test web instance (a ddnet-ai with the loopback-favourites feature) on its
// own port and data directory, the helper and the bot through the TEST path (launcher-sim.mjs, fake systemctl, run-bot.sh: nothing of the
// production units, /etc, Caddy or ~/aiddnet/data/bot is used), a SOCKS5 stand-in for «Проверить», and the real master list (a read-only
// HTTPS fetch) as the cache the «Серверы» tab shows. The bot never connects to any public server: the only server it plays on is the
// private one, added as a favourite by address.
//
// What it checks: the list (search, filters, sort, phone width), the owner's consent for a favourite, malicious addresses refused, the proxy
// profiles with a write-only password (it is in no page text and no API answer), «Проверить» (ok, then «логин или пароль не приняты»), the
// private server as a favourite played from the «Серверы» tab through the card, exactly one bot on the server, a kick closing the favourite
// (the card and the tab say so, a changed proxy does not open it), «Открыть снова», and a second run.
//
// Needs the env of servers-e2e.sh (else it skips). Screenshots: ~/aiddnet/data/screenshots/5.12-*.png (never in git).

import { test, expect, type Page, type Response } from "@playwright/test";
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const BASE = process.env.DDAI_SERVERS_E2E_URL ?? "";
const PASSWORD_FILE = process.env.DDAI_SERVERS_E2E_PASSWORD_FILE ?? "";
const DIR = process.env.DDAI_SERVERS_E2E_DIR ?? "";
const GAME = process.env.DDAI_SERVERS_E2E_GAME_ADDR ?? "127.0.0.1:8463";
const ECON_PORT = process.env.DDAI_SERVERS_E2E_ECON_PORT ?? "8464";
const ECON_CFG = process.env.DDAI_SERVERS_E2E_ECON_CFG ?? "";
const SOCKS = process.env.DDAI_SERVERS_E2E_SOCKS ?? "127.0.0.1:8466";
const SOCKS_USER = process.env.DDAI_SERVERS_E2E_SOCKS_USER ?? "";
const SOCKS_PASS = process.env.DDAI_SERVERS_E2E_SOCKS_PASS ?? "";
const SHOTS = path.join(homedir(), "aiddnet", "data", "screenshots");
const ECON = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "ddnet-server", "econ.py");

test.skip(!BASE || !PASSWORD_FILE || !DIR || !existsSync(PASSWORD_FILE), "run tools/e2e/servers-e2e.sh (see README.md)");
test.describe.configure({ mode: "serial" });

// Every /api answer of the whole run, to prove the proxy password and user name are in none of them.
const apiBodies: string[] = [];

function econ(...cmd: string[]): string {
  return execFileSync("python3", [ECON, "--host", "127.0.0.1", "--port", ECON_PORT, "--password-file", ECON_CFG, ...cmd], { encoding: "utf8" });
}

function muhaIds(): string[] {
  // `status` prints one line per client: `id=0 addr=127.0.0.1:port ... name='Muha' ...`
  return econ("status")
    .split("\n")
    .filter((l) => /name='Muha'/.test(l))
    .map((l) => /id=(\d+)/.exec(l)?.[1] ?? "");
}

/** The helper's memory: the real-time cool-down and start interval are over (as `launch_apply.rs` does), the ban memory is untouched. */
function ageHelperState(seconds: number) {
  const p = path.join(DIR, "var", "state.json");
  const st = JSON.parse(readFileSync(p, "utf8"));
  st.last_start_at = Math.max(0, (st.last_start_at ?? 0) - seconds);
  if (st.last_exit) st.last_exit.at = Math.max(0, st.last_exit.at - seconds);
  writeFileSync(p, JSON.stringify(st));
}

async function login(page: Page) {
  page.on("response", (r: Response) => {
    if (r.url().includes("/api/")) {
      r.text().then((t) => apiBodies.push(t)).catch(() => {});
    }
  });
  await page.goto(BASE + "/");
  await page.locator("#password").fill(readFileSync(PASSWORD_FILE, "utf8").trim());
  await page.locator('#login-form button[type="submit"]').click();
  await expect(page.locator("#tabbar")).toBeVisible();
}

// The three panels of the tab (only one is visible at a time); every locator is scoped to its panel.
const listPanel = (page: Page) => page.locator('#servers-mount > section[aria-label="Список серверов"]');
const favPanel = (page: Page) => page.locator('#servers-mount > section[aria-label="Избранное"]');
const proxyPanel = (page: Page) => page.locator('#servers-mount > section[aria-label="Прокси"]');

async function noHorizontalScroll(page: Page) {
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(0);
}

async function openPanel(page: Page, name: "Список" | "Избранное" | "Прокси") {
  await page.locator("#tab-servers").click();
  await expect(page.locator("#servers-view")).toBeVisible();
  await page.locator(".sv-seg-btn", { hasText: name }).click();
}

test.beforeAll(() => mkdirSync(SHOTS, { recursive: true }));

test("the list: real master list from the cache, search, filters, sort, phone width", async ({ page }) => {
  await login(page);
  await page.locator("#tab-servers").click();
  await expect(page.locator("#servers-view")).toBeVisible();
  const info = page.locator(".sv-info");
  await expect(info).not.toHaveText("загрузка…");
  test.skip(!(await info.innerText()).startsWith("Серверов:"), "the master list could not be fetched on this run");
  await expect(info).toContainText("сайт сам в сеть не ходит");
  const rows = listPanel(page).locator(".sv-rows .sv-row");
  await expect(rows.first()).toBeVisible();
  const blockOnly = await rows.count();
  expect(blockOnly).toBeGreaterThan(0);
  expect(blockOnly).toBeLessThanOrEqual(40);

  // «Только блок-карты» is on by default; switching it off shows more (the list has many non-block servers).
  const all = page.locator('.sv-check:has-text("Только блок-карты") input');
  await expect(all).toBeChecked();
  await all.uncheck();
  await expect(page.locator(".sv-more")).toBeVisible();
  await all.check();

  // Search: nothing matches a nonsense string; a real one finds rows whose text contains it.
  const q = page.locator('.servers-view input[type="search"]');
  await q.fill("zzzzqqqq-no-such-server");
  await expect(page.locator(".sv-empty")).toContainText("Ничего не найдено");
  await q.fill("");
  await all.uncheck();
  const firstName = (await rows.first().locator(".sv-name").innerText()).trim();
  const word = firstName.split(/\s+/).find((w) => w.length >= 4) ?? firstName;
  await q.fill(word);
  const hit = await rows.count();
  expect(hit).toBeGreaterThan(0);
  await expect(rows.first()).toContainText(new RegExp(word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i"));
  await q.fill("");

  // Sort by name: the visible names are in order.
  await page.locator('.servers-view select').nth(3).selectOption("name");
  const names = await rows.locator(".sv-name").allTextContents();
  // Sorted with the browser's own collation (Node's ICU orders punctuation differently from Chromium's).
  const sorted = await page.evaluate((list) => [...list].sort((a, b) => a.localeCompare(b)), names);
  expect(names).toEqual(sorted);
  await page.locator('.servers-view select').nth(3).selectOption("players");
  const players = (await rows.locator(".sv-players").allInnerTexts()).map((t) => parseInt(/Игроков: (\d+)/.exec(t)?.[1] ?? "0", 10));
  expect(players).toEqual([...players].sort((a, b) => b - a));

  // The «Обновить список» request goes to the refresh route and says what it did.
  await page.locator(".sv-head button").click();
  await expect(page.locator(".sv-result").first()).toContainText(/Запрос отправлен|Список свежий/);

  // Phone width: no horizontal scroll, the six tabs fit.
  await page.setViewportSize({ width: 360, height: 740 });
  await noHorizontalScroll(page);
  for (const id of ["tab-status", "tab-game", "tab-bot", "tab-servers", "tab-fly", "tab-train"]) {
    const box = await page.locator("#" + id).boundingBox();
    expect(box && box.x >= 0 && box.x + box.width <= 360).toBeTruthy();
  }
  await page.screenshot({ path: path.join(SHOTS, "5.12-list-phone.png"), fullPage: false });
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.screenshot({ path: path.join(SHOTS, "5.12-list-desktop.png"), fullPage: false });
});

test("a favourite needs the owner's consent; malicious addresses are refused", async ({ page }) => {
  await login(page);
  await page.locator("#tab-servers").click();
  await expect(page.locator(".sv-info")).not.toHaveText("загрузка…");
  const rows = listPanel(page).locator(".sv-rows .sv-row");
  test.skip(!(await page.locator(".sv-info").innerText()).startsWith("Серверов:"), "no list on this run");
  await expect(rows.first()).toBeVisible();
  const row = rows.first();
  const name = (await row.locator(".sv-name").innerText()).trim();
  await row.getByRole("button", { name: "В избранное" }).click();
  const form = row.locator(".sv-form");
  await expect(form).toBeVisible();
  await form.getByRole("button", { name: "Добавить в избранное" }).click();
  await expect(form.locator(".sv-result")).toContainText("подтвердить");
  await form.locator('.sv-check input').check();
  await form.getByRole("button", { name: "Добавить в избранное" }).click();
  await openPanel(page, "Избранное");
  await expect(favPanel(page).locator(".sv-row .sv-name", { hasText: name }).first()).toBeVisible();
  await expect(favPanel(page).getByText("разрешение подтверждено").first()).toBeVisible();

  // By address: private, loopback-name, junk and a missing consent are all refused with a reason; nothing is added.
  await page.locator(".sv-manual summary").click();
  const manual = page.locator(".sv-manual .sv-form");
  const addr = manual.locator('input[placeholder^="IP:порт"]');
  const consent = manual.locator(".sv-check input");
  const before = await favPanel(page).locator(".sv-row").count();
  await consent.check();
  for (const bad of ["10.0.0.5:8303", "localhost:8303", "example.com:8303", "45.141.57.35", "8.8.8.8:0", "169.254.169.254:80", "8.8.8.8:8303/../x"]) {
    await addr.fill(bad);
    await manual.getByRole("button", { name: "Добавить в избранное" }).click();
    await expect(manual.locator(".sv-result")).toContainText("Адрес должен быть вида IP:порт", { timeout: 10_000 });
  }
  expect(await favPanel(page).locator(".sv-row").count()).toBe(before);

  // Remove the list favourite again (nothing public stays in the test data).
  page.once("dialog", (d) => d.accept());
  await favPanel(page).locator(".sv-row", { hasText: name }).first().getByRole("button", { name: "Удалить" }).click();
  await expect(favPanel(page).locator(".sv-row .sv-name", { hasText: name })).toHaveCount(0);
});

test("proxy profiles: the password is write-only; «Проверить» says ok, then that the login was not accepted", async ({ page }) => {
  test.setTimeout(180_000);
  await login(page);
  await openPanel(page, "Прокси");
  const form = page.locator(".sv-proxy-form");
  const [host, port] = SOCKS.split(":");
  await form.locator('input[placeholder^="имя"]').fill("stub");
  await form.locator('input[placeholder^="публичный IP"]').fill(host);
  await form.locator('input[type="number"]').fill(port);
  await form.locator('input[placeholder="логин"]').fill(SOCKS_USER);
  await form.locator('input[type="password"]').fill(SOCKS_PASS);
  await form.getByRole("button", { name: "Сохранить" }).click();
  await expect(form.locator(".sv-result").first()).toContainText("Сохранено");
  const row = proxyPanel(page).locator(".sv-row", { hasText: "stub" });
  await expect(row).toContainText("с сайта");
  await expect(row).toContainText("логин и пароль заданы");
  // The password is nowhere in the page, not even in the form after saving.
  expect(await page.content()).not.toContain(SOCKS_PASS);
  expect(await form.locator('input[type="password"]').inputValue()).toBe("");

  // «Проверить»: the request goes through the check unit (the real `launch check-proxy`) to the SOCKS5 stand-in.
  await row.getByRole("button", { name: "Проверить" }).click();
  await expect(row).toContainText("Проверка: прокси работает", { timeout: 40_000 });
  await page.screenshot({ path: path.join(SHOTS, "5.12-proxy-ok-phone.png") });

  // Edit with a WRONG password (the field is write-only: nothing was prefilled), check again: the stand-in refuses the login.
  await row.getByRole("button", { name: "Изменить" }).click();
  await expect(form.locator('input[placeholder="пусто = оставить прежний"]').nth(1)).toHaveValue("");
  await form.locator('input[type="password"]').fill("definitely-not-the-password");
  await form.getByRole("button", { name: "Сохранить" }).click();
  await expect(form.locator(".sv-result").first()).toContainText("Сохранено");
  await expect(row.getByRole("button", { name: "Проверить" })).toBeEnabled();
  // The site's own limit is one check per 8 s (the route tests cover the refusal): wait it out.
  await page.waitForTimeout(8_500);
  await row.getByRole("button", { name: "Проверить" }).click();
  await expect(row).toContainText("логин или пароль не приняты", { timeout: 40_000 });
  // Put the right one back: an empty login field keeps the stored login.
  await row.getByRole("button", { name: "Изменить" }).click();
  await form.locator('input[type="password"]').fill(SOCKS_PASS);
  await form.getByRole("button", { name: "Сохранить" }).click();
  await expect(form.locator(".sv-result").first()).toContainText("Сохранено");

  // Malicious fields are refused with a reason.
  await form.locator('input[placeholder^="имя"]').fill("../evil");
  await form.locator('input[placeholder^="публичный IP"]').fill("10.0.0.1");
  await form.locator('input[type="number"]').fill("1080");
  await form.locator('input[placeholder="логин"]').fill("u");
  await form.locator('input[type="password"]').fill("p");
  await form.getByRole("button", { name: "Сохранить" }).click();
  await expect(form.locator(".sv-result").first()).not.toHaveText("");
  expect((await form.locator(".sv-result").first().innerText()).length).toBeGreaterThan(5);
  expect(existsSync(path.join(DIR, "data", "secrets", "..-evil-proxy.toml"))).toBeFalsy();
  await page.setViewportSize({ width: 360, height: 740 });
  await noHorizontalScroll(page);
  await page.screenshot({ path: path.join(SHOTS, "5.12-proxies-phone.png"), fullPage: true });
});

test("the private server as a favourite: played from the tab, one bot, a kick closes it, a proxy switch does not open it, «Открыть снова»", async ({ page }) => {
  test.setTimeout(420_000);
  await login(page);
  expect(muhaIds()).toHaveLength(0);

  // Add the private server by address (a loopback favourite exists only in this test build).
  await openPanel(page, "Избранное");
  await page.locator(".sv-manual summary").click();
  const manual = page.locator(".sv-manual .sv-form");
  await manual.locator('input[placeholder^="IP:порт"]').fill(GAME);
  await manual.locator('input[placeholder="название"]').fill("Private e2e");
  await manual.locator(".sv-check input").check();
  await manual.getByRole("button", { name: "Добавить в избранное" }).click();
  await expect(manual.locator(".sv-result")).toContainText("Добавлено");
  const fav = favPanel(page).locator(".sv-row", { hasText: "Private e2e" });
  await expect(fav).toContainText(GAME);

  // «Играть здесь» opens the «Бот» tab with this server chosen in the «Запуск» card.
  await fav.getByRole("button", { name: "Играть здесь" }).click();
  await expect(page.locator("#bot-view")).toBeVisible();
  const card = page.locator(".launch-card");
  const field = (label: string) => card.locator("label.lc-field", { hasText: new RegExp("^" + label) }).locator("select");
  await expect(field("Сервер")).toHaveValue(GAME);
  await expect(field("Сервер").locator("option:checked")).toContainText("Private e2e");
  await expect(card.locator(".lc-state-text")).toHaveText("Бот остановлен", { timeout: 20_000 });
  await field("Мозг").selectOption("hybrid");
  await field("Длительность").selectOption("15m");
  page.once("dialog", (d) => d.accept());
  await card.locator(".lc-start").click();
  await expect(card.locator(".lc-result")).toContainText("запрос отправлен");
  await expect(card.locator(".lc-state-text")).toHaveText("В игре", { timeout: 150_000 });
  await page.screenshot({ path: path.join(SHOTS, "5.12-bot-playing.png"), fullPage: true });

  // What the REAL helper wrote for the test path: the favourite's address and nick, loopback + the server's IP only, no proxy.
  const env = readFileSync(path.join(DIR, "etc", "bot-launch.env"), "utf8");
  expect(env).toContain(`BOT_SERVER="${GAME}"`);
  expect(env).toContain('BOT_NAME="Muha"');
  const dropin = readFileSync(path.join(DIR, "etc", "50-launch.conf"), "utf8");
  expect(dropin).toContain("IPAddressAllow=127.0.0.1");
  expect(dropin).not.toContain("IPAddressDeny");
  // One bot on the server (the private server's own console says so).
  expect(muhaIds()).toHaveLength(1);

  // A kick: the bot stops (exit 3) and the favourite is closed. Nothing switches a proxy or an IP by itself.
  const id = muhaIds()[0];
  econ("kick", id, "e2e kick");
  await expect(card.locator(".lc-state-text")).toHaveText("Бот остановился с ошибкой", { timeout: 60_000 });
  await expect(card.locator(".lc-detail")).toContainText("кикнул или забанил");
  await expect(card.locator(".lc-start")).toBeDisabled();
  const blocked = JSON.parse(readFileSync(path.join(DIR, "status", "blocked.json"), "utf8"));
  expect(blocked.blocked.map((b: any) => b.address)).toContain(GAME);
  expect(muhaIds()).toHaveLength(0);

  await openPanel(page, "Избранное");
  await expect(fav).toContainText("закрыт после кика/бана");
  await expect(fav.getByRole("button", { name: "Играть здесь" })).toBeDisabled();
  await page.screenshot({ path: path.join(SHOTS, "5.12-favourite-closed.png"), fullPage: true });

  // The owner assigns a proxy: the favourite stays closed (the page says so, the helper's memory is unchanged).
  await fav.locator("select").selectOption("proxy:stub");
  await expect(favPanel(page).locator(".sv-result").first()).toContainText("всё ещё закрыт");
  await expect(fav.getByRole("button", { name: "Играть здесь" })).toBeDisabled();
  expect(JSON.parse(readFileSync(path.join(DIR, "var", "state.json"), "utf8")).blocked[GAME]).toBeTruthy();
  await fav.locator("select").selectOption("direct");
  await expect(favPanel(page).locator(".sv-result").first()).toContainText("Подключение изменено");

  // «Открыть снова»: an explicit, confirmed act. Then (after the real-time cool-down, which the test shortens in the helper's own
  // memory) the bot plays again, on the same direct connection.
  page.once("dialog", (d) => d.accept());
  await fav.getByRole("button", { name: "Открыть снова" }).click();
  await expect(fav).not.toContainText("закрыт после кика/бана", { timeout: 20_000 });
  ageHelperState(900);
  await fav.getByRole("button", { name: "Играть здесь" }).click();
  await expect(page.locator("#bot-view")).toBeVisible();
  await expect(card.locator(".lc-start")).toBeEnabled({ timeout: 20_000 });
  await field("Мозг").selectOption("hybrid");
  page.once("dialog", (d) => d.accept());
  await card.locator(".lc-start").click();
  await expect(card.locator(".lc-state-text")).toHaveText("В игре", { timeout: 150_000 });
  expect(muhaIds()).toHaveLength(1);
  const calls = readFileSync(path.join(DIR, "systemctl.log"), "utf8").split("\n").filter((l) => l.startsWith("start ddnet-ai-bot"));
  expect(calls).toHaveLength(2);
  expect(readFileSync(path.join(DIR, "etc", "50-launch.conf"), "utf8")).not.toContain("198.");
  // The ban stays in the helper's memory (it is never forgotten); it is inert because the re-opening is newer than it.
  const after = JSON.parse(readFileSync(path.join(DIR, "status", "blocked.json"), "utf8"));
  const favs = JSON.parse(readFileSync(path.join(DIR, "data", "launch", "favourites.json"), "utf8"));
  expect(after.blocked).toHaveLength(1);
  expect(favs.favourites[0].reopened_at).toBeGreaterThan(after.blocked[0].at);

  // Stop from the page.
  await card.locator(".lc-stop").click();
  await expect(card.locator(".lc-state-text")).toHaveText("Бот остановлен", { timeout: 60_000 });
  expect(muhaIds()).toHaveLength(0);
});

test("no API answer of the whole run ever held the proxy password or user name", async () => {
  expect(apiBodies.length).toBeGreaterThan(20);
  for (const body of apiBodies) {
    expect(body).not.toContain(SOCKS_PASS);
    expect(body).not.toContain("definitely-not-the-password");
    expect(body).not.toContain(SOCKS_USER);
  }
  // The profile file is 0600 and holds it; nothing else of the run printed it.
  const file = readFileSync(path.join(DIR, "data", "secrets", "stub-proxy.toml"), "utf8");
  expect(file).toContain(SOCKS_PASS);
  for (const log of ["web.log", "sim.log", "bot.log", "systemctl.log"]) {
    const p = path.join(DIR, log);
    if (existsSync(p)) {
      const text = readFileSync(p, "utf8");
      expect(text).not.toContain(SOCKS_PASS);
      expect(text).not.toContain("definitely-not-the-password");
    }
  }
});
