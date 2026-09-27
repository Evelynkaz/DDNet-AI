// Real-browser smoke test for `ddai-web`'s login/status/logout flow (task 5.1, acceptance
// criterion 9). Drives a real `ddnet-ai web` process (built beforehand with
// `cargo build -p ddnet-ai`) with headless Chromium via Playwright: log in with the password
// `ddnet-ai web-passwd` generates, wait for the status view to report the WebSocket connected,
// log out, and confirm we're back at the login form — once at a desktop viewport and once at a
// phone-sized viewport, saving a screenshot of each to `~/aiddnet/data/screenshots/`.
//
// How to run: see README.md in this folder.

import { test, expect, type Page } from "@playwright/test";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { mkdtempSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir, homedir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, "..", "..");
const BINARY = path.join(REPO_ROOT, "target", "debug", "ddnet-ai");
const SCREENSHOT_DIR = path.join(homedir(), "aiddnet", "data", "screenshots");

let dataDir: string;
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
  mkdirSync(SCREENSHOT_DIR, { recursive: true });
  dataDir = mkdtempSync(path.join(tmpdir(), "ddai-web-e2e-"));

  const passwdOutput = await runCli(["web-passwd", "--data-dir", dataDir, "--show"]);
  const passwordMatch = passwdOutput.match(/^password: (\S+)$/m);
  if (!passwordMatch) {
    throw new Error(`could not find the generated password in web-passwd output:\n${passwdOutput}`);
  }
  password = passwordMatch[1];

  serverProcess = spawn(BINARY, ["web", "--listen", "127.0.0.1:0", "--data-dir", dataDir], {
    stdio: ["ignore", "pipe", "pipe"],
  });
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
  if (dataDir) {
    rmSync(dataDir, { recursive: true, force: true });
  }
});

async function loginStatusLogout(page: Page, screenshotPath: string) {
  await page.goto(baseUrl);

  await expect(page.locator("#login-view")).toBeVisible();
  await expect(page.locator("#status-view")).toBeHidden();

  await page.locator("#password").fill(password);
  await page.locator("#login-form button[type=submit]").click();

  await expect(page.locator("#status-view")).toBeVisible();
  await expect(page.locator("#ws-state")).toHaveText("подключено", { timeout: 5_000 });
  await expect(page.locator("#bot-state")).toHaveText("idle", { timeout: 5_000 });

  await page.screenshot({ path: screenshotPath });

  await page.locator("#logout-button").click();
  await expect(page.locator("#login-view")).toBeVisible();
  await expect(page.locator("#status-view")).toBeHidden();
}

test("desktop: login shows connected status, then logout returns to the login form", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await loginStatusLogout(page, path.join(SCREENSHOT_DIR, "5.1-desktop.png"));
});

test("phone (360x740): login shows connected status, then logout returns to the login form", async ({ page }) => {
  await page.setViewportSize({ width: 360, height: 740 });
  await loginStatusLogout(page, path.join(SCREENSHOT_DIR, "5.1-phone.png"));
});
