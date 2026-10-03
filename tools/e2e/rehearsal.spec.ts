// Task 4.5, the dress rehearsal: the owner-facing tabs on the REAL production site (https://89-58-7-133.sslip.io) while the
// real bot runs through the production unit (soak.sh --real-data). Not part of CI and skipped without DDAI_E2E_BASE_URL.
//
//   DDAI_E2E_BASE_URL=https://89-58-7-133.sslip.io npx playwright test rehearsal.spec.ts -g "Бот"
//   DDAI_E2E_BASE_URL=... DDAI_REHEARSAL_HOLD_S=600 npx playwright test rehearsal.spec.ts -g "Муха"   # the bot runs with --fly-bundle
//
// The owner password is read from ~/aiddnet/data/secrets/web-password.txt at the moment of the run and typed into the login form: it is
// never printed, logged or put in a screenshot (the login page is never captured after typing). The bot tab test sends harmless commands
// (mode, wb, one clip with a harmless note), adds a SYNTHETIC tag-like name to the friends list and removes it again, and checks that
// ~/aiddnet/data/bot/relations.json is back to what it was (the same lists; a file that was absent before comes back as the site's own
// file with empty lists: nothing in data/bot is ever deleted). Screenshots go to ~/aiddnet/data/screenshots/4.5-*.png (never in git);
// pages show the bot's own identity and player TAGS only, never a real nickname.

import { test, expect, type Page } from "@playwright/test";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";

const BASE = process.env.DDAI_E2E_BASE_URL ?? "";
const DATA = path.join(homedir(), "aiddnet", "data");
const SHOTS = path.join(DATA, "screenshots");
const RELATIONS = path.join(DATA, "bot", "relations.json");
const CLIPS = path.join(DATA, "bot", "clips");
// A synthetic, tag-like name: not a real player.
const TEST_FRIEND = "c999-45rehearsal";
// A label for the screenshots of the lists editor: "prod" (the production site) or "twin" (web_twin.py).
const TAG = process.env.DDAI_REHEARSAL_TAG ?? "prod";

test.describe.configure({ mode: "serial" });
test.setTimeout(15 * 60_000);

// DDAI_E2E_PASSWORD_FILE names another password FILE (never the password itself): for the "twin" web unit of web_twin.py, which has its own.
function password(): string {
  const file = process.env.DDAI_E2E_PASSWORD_FILE ?? path.join(DATA, "secrets", "web-password.txt");
  return readFileSync(file, "utf8").trim();
}

async function login(page: Page, tab: "#tab-bot" | "#tab-fly", view: "#bot-view" | "#fly-view") {
  await page.goto(BASE);
  // Not `fill(password)`: Playwright prints the value of a failed `fill` in its error log. `evaluate` arguments are never printed.
  await page.locator("#password").evaluate((el, value) => {
    (el as HTMLInputElement).value = value;
    el.dispatchEvent(new Event("input", { bubbles: true }));
  }, password());
  await page.locator("#login-form button[type=submit]").click();
  await expect(page.locator("#tabbar")).toBeVisible();
  await page.locator(tab).click();
  await expect(page.locator(view)).toBeVisible();
}

/** What the owner's lists are: absent, or the parsed lists plus a sha256 of the bytes (never the names themselves in the report). */
function relationsState(): { exists: boolean; sha?: string; counts?: Record<string, number> } {
  if (!existsSync(RELATIONS)) return { exists: false };
  const raw = readFileSync(RELATIONS);
  const j = JSON.parse(raw.toString("utf8"));
  const counts: Record<string, number> = {};
  for (const k of ["friend", "war", "ignore", "clanWar", "clanFriend"]) counts[k] = (j[k] ?? []).length;
  return { exists: true, sha: createHash("sha256").update(raw).digest("hex").slice(0, 16), counts };
}

function clipFiles(): string[] {
  return existsSync(CLIPS) ? readdirSync(CLIPS).filter((f) => f.endsWith(".clip")) : [];
}

test.beforeAll(() => {
  test.skip(!BASE, "set DDAI_E2E_BASE_URL (the production site) to run the rehearsal");
  mkdirSync(SHOTS, { recursive: true });
});

test("Бот: status and commands with effects in the bot", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 1000 });
  page.on("dialog", (d) => d.accept());
  const clipsBefore = new Set(clipFiles());
  await login(page, "#tab-bot", "#bot-view");

  // Status: «В игре», the server, the map, the mode, hybrid, the production identity.
  await expect(page.locator("#bot-conn-text")).toHaveText("В игре", { timeout: 15_000 });
  await expect(page.locator("#bs-server")).toHaveText("127.0.0.1:8303");
  await expect(page.locator("#bs-map")).toHaveText("Copy Love Box");
  await expect(page.locator("#bs-brain")).toContainText("hybrid");
  await expect(page.locator("#bs-identity")).toContainText("Muha");
  await expect(page.locator("#bs-identity")).toContainText("Neuroset");
  await expect(page.locator('[data-cmd="mode"][data-mode="fight"]')).toHaveClass(/current/, { timeout: 10_000 });
  await expect(page.locator('[data-cmd="wb"][data-mode="auto"]')).toHaveClass(/current/);
  console.log("status ok:", await page.locator("#bs-identity").innerText(), "|", await page.locator("#bs-mode").innerText(), "|", await page.locator("#bs-wb").innerText());
  await page.screenshot({ path: path.join(SHOTS, "4.5-bot-status.png"), fullPage: true, mask: [page.locator("#rel-lists")] });

  const result = page.locator("#cmd-result");
  const pause = () => page.waitForTimeout(1_200); // the control channel allows 2 requests a second

  // mode passive, then fight: the bot's own status (read back through the bridge) shows it.
  await page.locator('[data-cmd="mode"][data-mode="passive"]').click();
  await expect(result).toContainText(/passive|пассив/i, { timeout: 10_000 });
  await expect(page.locator('[data-cmd="mode"][data-mode="passive"]')).toHaveClass(/current/, { timeout: 10_000 });
  await expect(page.locator("#bs-mode")).toHaveText("пассивный");
  console.log("mode passive ->", await result.innerText());
  await page.screenshot({ path: path.join(SHOTS, "4.5-bot-mode-passive.png"), fullPage: true, mask: [page.locator("#rel-lists")] });
  await pause();
  await page.locator('[data-cmd="mode"][data-mode="fight"]').click();
  await expect(page.locator('[data-cmd="mode"][data-mode="fight"]')).toHaveClass(/current/, { timeout: 10_000 });
  await expect(page.locator("#bs-mode")).toHaveText("бой");
  console.log("mode fight ->", await result.innerText());
  await pause();

  // wb off, then auto.
  await page.locator('[data-cmd="wb"][data-mode="off"]').click();
  await expect(page.locator('[data-cmd="wb"][data-mode="off"]')).toHaveClass(/current/, { timeout: 10_000 });
  await expect(page.locator("#bs-wb")).toContainText("WB: off");
  console.log("wb off ->", await result.innerText());
  await pause();
  await page.locator('[data-cmd="wb"][data-mode="auto"]').click();
  await expect(page.locator('[data-cmd="wb"][data-mode="auto"]')).toHaveClass(/current/, { timeout: 10_000 });
  await expect(page.locator("#bs-wb")).toContainText("WB: auto");
  console.log("wb auto ->", await result.innerText());
  await pause();

  // One clip with a harmless note: the counter moves and a manual clip file appears.
  const clipsShown = Number(await page.locator("#bs-clips").innerText());
  await page.locator("#cmd-clip-note").fill("rehearsal 4.5");
  await page.locator("#cmd-clip").click();
  await expect(result).toContainText(/clip|клип/i, { timeout: 10_000 });
  console.log("clip ->", await result.innerText());
  await expect.poll(() => clipFiles().filter((f) => !clipsBefore.has(f) && f.startsWith("manual-") && f.includes("rehearsal")).length, { timeout: 15_000 }).toBeGreaterThan(0);
  await expect.poll(async () => Number(await page.locator("#bs-clips").innerText()), { timeout: 15_000 }).toBeGreaterThan(clipsShown);
  await expect(page.locator("#bot-conn-text")).toHaveText("В игре");
});

test("Бот: a synthetic friend is added and removed, relations.json is as it was", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 1000 });
  page.on("dialog", (d) => d.accept());
  const consoleText: string[] = [];
  page.on("console", (m) => consoleText.push(m.text()));
  const before = relationsState();
  // Real friends would be names on a screenshot (and this test edits the lists): only ever run it against empty lists. The lists are
  // masked in every Бот-tab screenshot anyway.
  test.skip(
    before.exists && Object.values(before.counts ?? {}).some((n) => n > 0),
    "the owner's lists are not empty: this test adds and removes a synthetic name and never runs over real ones",
  );
  await login(page, "#tab-bot", "#bot-view");
  await expect(page.locator("#bot-conn-text")).toHaveText("В игре", { timeout: 15_000 });
  const pause = () => page.waitForTimeout(1_200);

  // A synthetic friend: «применено к работающему боту (бот перечитал тот же файл)», shown in the list, in the file.
  await page.locator("#rel-kind").selectOption("friend");
  await page.locator("#rel-name").fill(TEST_FRIEND);
  await expect(page.locator("#rel-preview")).toContainText(TEST_FRIEND);
  await page.locator("#rel-form button[type=submit]").click();
  await expect(page.locator("#rel-result")).not.toHaveText("", { timeout: 10_000 });
  await page.screenshot({ path: path.join(SHOTS, `4.5-bot-friend-${TAG}-result.png`), fullPage: true, mask: [page.locator("#rel-lists")] });
  await expect(page.locator("#rel-result")).toContainText("применено к работающему боту", { timeout: 10_000 });
  await expect(page.locator("#rel-result")).toContainText("тот же файл");
  console.log("friend added ->", await page.locator("#rel-result").innerText());
  await expect(page.locator(".rel-group", { hasText: /^Друзья \(\d+\)/ })).toContainText(TEST_FRIEND);
  const added = JSON.parse(readFileSync(RELATIONS, "utf8"));
  expect(added.friend).toContain(TEST_FRIEND);
  await page.screenshot({ path: path.join(SHOTS, `4.5-bot-friend-${TAG}-added.png`), fullPage: true, mask: [page.locator("#rel-lists")] });
  await pause();

  // Removed again: the lists are back to what they were.
  await page.locator(`button[aria-label*="${TEST_FRIEND}"]`).click();
  await expect(page.locator("#rel-lists")).not.toContainText(TEST_FRIEND, { timeout: 10_000 });
  const after = relationsState();
  console.log("relations.json before:", JSON.stringify(before), "after:", JSON.stringify(after));
  if (before.exists) expect(after).toEqual(before);
  else {
    // The site's own file with empty lists (the harness never deletes anything in data/bot).
    expect(after.exists).toBe(true);
    expect(Object.values(after.counts ?? {}).every((n) => n === 0)).toBe(true);
  }
  await page.screenshot({ path: path.join(SHOTS, `4.5-bot-friend-${TAG}-removed.png`), fullPage: true, mask: [page.locator("#rel-lists")] });

  // The synthetic name never reached the browser console as a real nickname would; the page still says «В игре».
  await expect(page.locator("#bot-conn-text")).toHaveText("В игре");
  expect(consoleText.join("\n")).not.toContain("Error");

  // A phone viewport of the same tab.
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(800);
  expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);
  await page.screenshot({ path: path.join(SHOTS, "4.5-bot-phone.png"), fullPage: true, mask: [page.locator("#rel-lists")] });
});

test("Муха: live frames on the production site while the bot plays with --fly-bundle", async ({ page }) => {
  test.skip(process.env.DDAI_REHEARSAL_FLY !== "1", "set DDAI_REHEARSAL_FLY=1 when the bot runs with --fly-bundle");
  const holdS = Number(process.env.DDAI_REHEARSAL_HOLD_S ?? "120");
  await page.setViewportSize({ width: 1280, height: 1000 });
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  if (process.env.DDAI_REHEARSAL_WB === "off") {
    // With the wayblock on (the default) the bot answers most snapshots without the brain, and the panel only gets a frame when the
    // brain decided: sparse. Fight instead (wb off) so the fly's stream is continuous; wb auto again at the end of the test.
    await login(page, "#tab-bot", "#bot-view");
    await expect(page.locator("#bot-conn-text")).toHaveText("В игре", { timeout: 15_000 });
    await page.locator('[data-cmd="wb"][data-mode="off"]').click();
    await expect(page.locator('[data-cmd="wb"][data-mode="off"]')).toHaveClass(/current/, { timeout: 10_000 });
    await page.locator("#tab-fly").click();
    await expect(page.locator("#fly-view")).toBeVisible();
  } else {
    await login(page, "#tab-fly", "#fly-view");
  }
  await expect(page.locator("#fly-state")).toHaveText("муха работает", { timeout: 60_000 });
  await expect(page.locator("#fly-content")).toBeVisible();
  await expect(page.locator("#fly-bundle")).toHaveText("e005-fly/final");
  await expect(page.locator("#fly-hash")).toHaveText(/^[0-9a-f]{16}$/);
  await expect(page.locator("#fly-brain")).toContainText("предлагает");
  await expect(page.locator("#fly-proposer")).toBeVisible();
  await expect(page.locator("#fly-prop-share")).toContainText("%", { timeout: 30_000 });
  await page.waitForTimeout(6_000);
  const frame = await page.evaluate(() => (window as any).FlyPanel._last());
  expect(frame.groups.length).toBeGreaterThanOrEqual(20);
  expect(frame.dn.length).toBe(100);
  expect(frame.eye.length).toBe(7);
  await page.screenshot({ path: path.join(SHOTS, "4.5-fly-desktop.png"), fullPage: true });

  // Hold the tab: the frame counter keeps moving (live frames), the proposer's share is read now and then.
  const samples: { t: number; seq: number; share: string; latency: string }[] = [];
  let stalls = 0;
  let last = -1;
  const start = Date.now();
  while ((Date.now() - start) / 1000 < holdS) {
    await page.waitForTimeout(15_000);
    const seq = await page.evaluate(() => (window as any).FlyPanel._last().seq);
    if (seq === last) stalls++;
    last = seq;
    samples.push({
      t: Math.round((Date.now() - start) / 1000),
      seq,
      share: (await page.locator("#fly-prop-share").innerText()).replace(/\s+/g, " "),
      latency: (await page.locator("#fly-latency").innerText()).replace(/\s+/g, " "),
    });
  }
  console.log("fly samples:", JSON.stringify(samples));
  console.log("fly stalls (a 15 s sample with an unchanged frame counter):", stalls);
  expect(stalls).toBeLessThanOrEqual(1); // a game restart renumbers frames; one equal pair at most
  await page.screenshot({ path: path.join(SHOTS, "4.5-fly-desktop-later.png"), fullPage: true });

  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(3_000);
  expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);
  await page.screenshot({ path: path.join(SHOTS, "4.5-fly-phone.png"), fullPage: true });
  if (process.env.DDAI_REHEARSAL_WB === "off") {
    await page.locator("#tab-bot").click();
    await page.locator('[data-cmd="wb"][data-mode="auto"]').click();
    await expect(page.locator('[data-cmd="wb"][data-mode="auto"]')).toHaveClass(/current/, { timeout: 10_000 });
  }
  expect(errors).toEqual([]);
});
