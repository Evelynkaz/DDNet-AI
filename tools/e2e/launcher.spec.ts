// Real-browser end-to-end of the web launcher (task 5.9, D-089), on THIS machine, LOCAL server only: a test web instance (not the
// production one on 7788: a copy of the web unit on another port, same hardening, see deploy/README.md "Проверка запуска") talks to
// the REAL launcher units (ddnet-ai-launch.path -> ddnet-ai-launch.service -> ddnet-ai-bot.service + ddnet-ai-sparring@N). The test
// logs in, picks hybrid + fly with 2 sparring opponents on the local server, starts the bot from the «Бот» tab, sees it play on
// the «Игра» tab («Живой бот»), then stops it from the page. It never picks any public server (the Swarfey entry may be offered in
// the list when its `ready` is true: the test only checks that it is listed as a choice, never selects it).
//
// Needs (else it skips): DDAI_LAUNCH_E2E_URL (the test instance, e.g. http://127.0.0.1:7789) and DDAI_LAUNCH_E2E_PASSWORD_FILE, the
// installed launcher units (deploy/install-launcher.sh), the local server running, and the bot unit stopped. Leaves the bot and the
// sparring units stopped. Screenshots: ~/aiddnet/data/screenshots/5.9-*.png (never in git). How to run: README.md.

import { test, expect, type Page } from "@playwright/test";
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, existsSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";

const BASE = process.env.DDAI_LAUNCH_E2E_URL ?? "";
const PASSWORD_FILE = process.env.DDAI_LAUNCH_E2E_PASSWORD_FILE ?? "";
const SCREENSHOT_DIR = path.join(homedir(), "aiddnet", "data", "screenshots");

function active(unit: string): string {
  try {
    return execFileSync("systemctl", ["is-active", unit], { encoding: "utf8" }).trim();
  } catch (e: any) {
    return String(e.stdout ?? "inactive").trim();
  }
}

function botCommandLine(): string {
  const pid = execFileSync("systemctl", ["show", "-p", "MainPID", "--value", "ddnet-ai-bot.service"], { encoding: "utf8" }).trim();
  return readFileSync(`/proc/${pid}/cmdline`, "utf8").split("\0").join(" ");
}

async function login(page: Page) {
  await page.goto(BASE + "/");
  await page.locator("#password").fill(readFileSync(PASSWORD_FILE, "utf8").trim());
  await page.locator('#login-form button[type="submit"]').click();
  await expect(page.locator("#tabbar")).toBeVisible();
}

test.skip(!BASE || !PASSWORD_FILE || !existsSync(PASSWORD_FILE), "set DDAI_LAUNCH_E2E_URL and DDAI_LAUNCH_E2E_PASSWORD_FILE (see README.md)");

test("start hybrid + fly with 2 sparring from the site, watch it play, stop it", async ({ page }) => {
  test.setTimeout(240_000);
  mkdirSync(SCREENSHOT_DIR, { recursive: true });
  expect(active("ddnet-ai-bot.service")).not.toBe("active");
  expect(active("ddnet-local.service")).toBe("active");

  await login(page);
  await page.locator("#tab-bot").click();
  const card = page.locator(".launch-card");
  await expect(card).toBeVisible();
  await expect(card.locator(".lc-state-text")).toHaveText("Бот остановлен", { timeout: 15_000 });

  // The choices: the local server first; the bundle is named by run only.
  const servers = await card.locator("select").first().locator("option").allTextContents();
  expect(servers[0]).toBe("Локальный сервер");
  await expect(card.locator(".lc-bundle")).toContainText("E-005/e005-fly");
  await expect(card.locator(".lc-bundle")).not.toContainText("/home");
  // By label, not by position: the card gained fields (the opponent model, then «Дожим») between the old positions.
  const field = (label: string) => card.locator("label.lc-field", { hasText: new RegExp("^" + label) }).locator("select");
  await expect(field("Сервер")).toHaveValue("local");
  await field("Мозг").selectOption("hybrid-fly");
  await field("Длительность").selectOption("15m");
  await field("Спарринг").selectOption("2");

  // Phone width: the card fits without a horizontal scroll.
  await page.setViewportSize({ width: 360, height: 740 });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(0);
  await card.screenshot({ path: path.join(SCREENSHOT_DIR, "5.9-card-phone.png") });
  await page.setViewportSize({ width: 1280, height: 900 });

  await card.locator(".lc-start").click();
  await expect(card.locator(".lc-result")).toContainText("запрос отправлен");
  await expect(card.locator(".lc-state-text")).toHaveText("В игре", { timeout: 90_000 });
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "5.9-bot-tab-playing.png"), fullPage: true });

  // What systemd really runs: the bot with the chosen brain, bundle and time, and exactly two sparring opponents.
  expect(active("ddnet-ai-bot.service")).toBe("active");
  expect(active("ddnet-ai-sparring@1.service")).toBe("active");
  expect(active("ddnet-ai-sparring@2.service")).toBe("active");
  expect(active("ddnet-ai-sparring@3.service")).not.toBe("active");
  const cmd = botCommandLine();
  expect(cmd).toContain("--server 127.0.0.1:8303");
  expect(cmd).toContain("--brain hybrid");
  expect(cmd).toContain("--duration 900");
  expect(cmd).toContain("--fly-bundle");

  // The game tab shows the live game.
  await page.locator("#tab-game").click();
  await expect(page.locator("#bot-source")).toContainText("Живой бот", { timeout: 30_000 }).catch(async () => {
    await expect(page.locator("body")).toContainText("Живой бот", { timeout: 30_000 });
  });
  await page.waitForTimeout(4000);
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "5.9-game-live.png") });

  // Stop from the page.
  await page.locator("#tab-bot").click();
  await expect(card.locator(".lc-stop")).toBeEnabled();
  await card.locator(".lc-stop").click();
  await expect(card.locator(".lc-state-text")).toHaveText("Бот остановлен", { timeout: 60_000 });
  await expect(card.locator(".lc-detail")).toContainText("Остановлен по вашей просьбе");
  for (const u of ["ddnet-ai-bot.service", "ddnet-ai-sparring@1.service", "ddnet-ai-sparring@2.service", "ddnet-ai-sparring@3.service"]) {
    expect(active(u)).not.toBe("active");
  }
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "5.9-bot-tab-stopped.png"), fullPage: true });
});

test("a run that ends by itself (the bot gets SIGTERM behind the launcher's back) is shown on the page", async ({ page }) => {
  test.setTimeout(240_000);
  // The helper allows one start per 30 s; the first test started a moment ago.
  await page.waitForTimeout(32_000);
  expect(active("ddnet-ai-bot.service")).not.toBe("active");
  await login(page);
  await page.locator("#tab-bot").click();
  const card = page.locator(".launch-card");
  await expect(card.locator(".lc-state-text")).toHaveText("Бот остановлен", { timeout: 15_000 });
  const field = (label: string) => card.locator("label.lc-field", { hasText: new RegExp("^" + label) }).locator("select");
  await field("Мозг").selectOption("hybrid-fly");
  await field("Длительность").selectOption("15m");
  await field("Спарринг").selectOption("1");
  await card.locator(".lc-start").click();
  await expect(card.locator(".lc-state-text")).toHaveText("В игре", { timeout: 90_000 });

  // Not the page's Stop: the bot is told to end (SIGTERM, exit 0) behind the launcher's back, so only the exit hook can report it.
  execFileSync("sudo", ["systemctl", "kill", "--signal=SIGTERM", "ddnet-ai-bot.service"]);
  await expect(card.locator(".lc-state-text")).toHaveText("Бот остановлен (не по кнопке)", { timeout: 60_000 });
  await expect(card.locator(".lc-detail")).toContainText("не по кнопке");
  expect(active("ddnet-ai-bot.service")).not.toBe("active");
  expect(active("ddnet-ai-sparring@1.service")).not.toBe("active");
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "5.9-self-ended.png"), fullPage: true });
});
