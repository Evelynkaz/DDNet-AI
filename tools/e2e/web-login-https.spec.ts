// Real-browser smoke test for the bot's web UI over the real, publicly-issued HTTPS deployment
// (task 5.3, acceptance criterion 5) — everything web-login.spec.ts (task 5.1) already covers
// (login -> connected status -> logout -> back to login form), but through Caddy's real
// Let's-Encrypt certificate instead of a locally-spawned plain-HTTP instance, which is the one
// thing 5.1's own e2e structurally cannot exercise (it needs an already-deployed server, a real
// DNS name, and the production password file).
//
// Skipped entirely unless DDAI_E2E_BASE_URL is set (so a plain `npx playwright test` — e.g. in
// CI, or a contributor with no deployment at all — still only runs web-login.spec.ts and stays
// green; see README.md). Reads the production password from
// ~/aiddnet/data/secrets/web-password.txt at runtime; never logs or prints it.
//
// How to run: see README.md in this folder.

import { test, expect, type Page } from "@playwright/test";
import { mkdirSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";

const BASE_URL = process.env.DDAI_E2E_BASE_URL;
const SCREENSHOT_DIR = path.join(homedir(), "aiddnet", "data", "screenshots");
const PASSWORD_FILE = path.join(homedir(), "aiddnet", "data", "secrets", "web-password.txt");

test.skip(
  !BASE_URL,
  "DDAI_E2E_BASE_URL is not set; this spec only runs against a real deployment (see README.md)",
);

test.beforeAll(() => {
  mkdirSync(SCREENSHOT_DIR, { recursive: true });
});

function readPassword(): string {
  return readFileSync(PASSWORD_FILE, "utf8").trim();
}

async function loginStatusLogout(page: Page, screenshotPath: string) {
  const password = readPassword();
  await page.goto(BASE_URL!);

  await expect(page.locator("#login-view")).toBeVisible();
  await expect(page.locator("#status-view")).toBeHidden();

  await page.locator("#password").fill(password);
  await page.locator("#login-form button[type=submit]").click();

  // Generous timeouts vs. web-login.spec.ts's defaults: this is a real network round trip to a
  // real TLS-terminating proxy, not a loopback connection to a freshly-spawned process.
  await expect(page.locator("#status-view")).toBeVisible({ timeout: 15_000 });
  await expect(page.locator("#ws-state")).toHaveText("подключено", { timeout: 15_000 });
  // whatever the real deployment is doing right now: a state of the bot (task 5.11; "idle" is a deployment older than that)
  await expect(page.locator("#bot-state")).toHaveText(/^(в игре|запущен, не в игре|не запущен|показ \(не настоящая игра\)|idle)$/, { timeout: 15_000 });

  await page.screenshot({ path: screenshotPath });

  await page.locator("#logout-button").click();
  await expect(page.locator("#login-view")).toBeVisible();
  await expect(page.locator("#status-view")).toBeHidden();
}

test("desktop: login over the real HTTPS deployment shows connected status, then logout returns to the login form", async ({
  page,
}) => {
  test.setTimeout(60_000);
  await page.setViewportSize({ width: 1280, height: 800 });
  await loginStatusLogout(page, path.join(SCREENSHOT_DIR, "5.3-desktop.png"));
});

test("phone (360x740): login over the real HTTPS deployment shows connected status, then logout returns to the login form", async ({
  page,
}) => {
  test.setTimeout(60_000);
  await page.setViewportSize({ width: 360, height: 740 });
  await loginStatusLogout(page, path.join(SCREENSHOT_DIR, "5.3-phone.png"));
});
