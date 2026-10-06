// Real-browser smoke test of the bot control tab (task 5.6): status panel, commands, friends/war/ignore editor.
// Drives a real `ddnet-ai web` process (built beforehand with `cargo build -p ddnet-ai`) in a scratch data directory on
// an ephemeral port — never the production unit — against a scripted FAKE bot written here in Node: a control socket
// (`<data-dir>/bot/control.sock`, the protocol of docs/formats.md §26) and a read-only bridge (`live.sock`, §21.2) that
// sends HELLO and STATUS. No real bot, no game server, no DDNet connection of any kind.
//
// How to run: see README.md in this folder.

import { test, expect, type Page } from "@playwright/test";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { mkdtempSync, mkdirSync, readFileSync, existsSync, rmSync } from "node:fs";
import net from "node:net";
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
let controlServer: net.Server;
let bridgeServer: net.Server;
let bridgeTimer: NodeJS.Timeout;

// ---- the fake bot ---------------------------------------------------------------------------------

const received: any[] = [];
const fakeStatus: Record<string, unknown> = {
  tick: 1234, own: 3, target: 5, mode: "fight", brain: "hybrid-none-4ms", alive: true, frozen: false,
  blocks: 7, blocked_by: 2, self_kills: 1, decisions: 99, collapsed: 0,
  decide_p50_us: 800, decide_p99_us: 4100, brain_p99_us: 3900, overhead_p99_us: 200, telemetry: null,
  connected: true, server: "127.0.0.1:8303", map: "Copy Love Box", name: "bot", clan: "Neuroset", skin: "pinky",
  target_tag: "c5-0a1b2c3d", wb: "WB: auto, holding the left; playing: 1 on the left, 2 on the right", goto: "",
  deaths: 4, clips_saved: 2, kill_cooldown_ticks: 120, selfkill: "on",
};

// The same fingerprint as `Relations::digest` (FNV-1a 64 over kind name, 0x1e, entries each followed by 0x1f, 0x1d), computed
// here independently from the file the site wrote, like the real bot does after loading it.
function listsDigest(): string {
  const file = JSON.parse(readFileSync(path.join(dataDir, "bot", "relations.json"), "utf8"));
  const kinds: [string, string[]][] = [
    ["friend", file.friend ?? []], ["war", file.war ?? []], ["ignore", file.ignore ?? []],
    ["clanwar", file.clanWar ?? []], ["clanfriend", file.clanFriend ?? []],
  ];
  let h = 0xcbf29ce484222325n;
  const eat = (bytes: Uint8Array) => {
    for (const b of bytes) h = ((h ^ BigInt(b)) * 0x100000001b3n) & 0xffffffffffffffffn;
  };
  for (const [name, entries] of kinds) {
    eat(Buffer.from(name));
    eat(Uint8Array.of(0x1e));
    for (const e of [...entries].map((x) => Buffer.from(x)).sort(Buffer.compare)) {
      eat(e);
      eat(Uint8Array.of(0x1f));
    }
    eat(Uint8Array.of(0x1d));
  }
  return h.toString(16).padStart(16, "0");
}

function startFakeControl(socketPath: string): Promise<net.Server> {
  return new Promise((resolve) => {
    const server = net.createServer((conn) => {
      let buffer = "";
      conn.on("data", (chunk) => {
        buffer += chunk.toString();
        let nl: number;
        while ((nl = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, nl);
          buffer = buffer.slice(nl + 1);
          const req = JSON.parse(line);
          received.push(req);
          const cmd = req.cmd;
          if (cmd.type === "mode") fakeStatus.mode = cmd.mode;
          if (cmd.type === "brain") fakeStatus.brain = cmd.brain;
          const reply: any = { v: 1, ok: true, text: `did ${cmd.type}${cmd.mode ? ":" + cmd.mode : ""}` };
          if (cmd.type === "reload_relations") {
            reply.text = "lists reloaded (counts)";
            reply.data = { digest: listsDigest() };
          }
          conn.write(JSON.stringify(reply) + "\n");
        }
      });
    });
    server.listen(socketPath, () => resolve(server));
  });
}

function message(kind: number, payload: Buffer): Buffer {
  const head = Buffer.alloc(5);
  head.writeUInt32LE(payload.length + 1, 0);
  head.writeUInt8(kind, 4);
  return Buffer.concat([head, payload]);
}

function startFakeBridge(socketPath: string): Promise<net.Server> {
  return new Promise((resolve) => {
    const clients = new Set<net.Socket>();
    const server = net.createServer((conn) => {
      clients.add(conn);
      conn.on("close", () => clients.delete(conn));
      conn.on("error", () => clients.delete(conn));
      conn.write(message(1, Buffer.from("DDBL\x01", "latin1")));
    });
    bridgeTimer = setInterval(() => {
      const payload = Buffer.from(JSON.stringify(fakeStatus));
      for (const c of clients) c.write(message(5, payload));
    }, 200);
    server.listen(socketPath, () => resolve(server));
  });
}

// ---- the web process ------------------------------------------------------------------------------

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

test.describe.configure({ mode: "serial" });

test.beforeAll(async () => {
  mkdirSync(SCREENSHOT_DIR, { recursive: true });
  dataDir = mkdtempSync(path.join(tmpdir(), "ddai-web-botctl-"));
  mkdirSync(path.join(dataDir, "bot"), { recursive: true });

  const passwdOutput = await runCli(["web-passwd", "--data-dir", dataDir, "--show"]);
  const passwordMatch = passwdOutput.match(/^password: (\S+)$/m);
  if (!passwordMatch) throw new Error("could not find the generated password in web-passwd output");
  password = passwordMatch[1];

  const botDir = path.join(dataDir, "bot");
  controlServer = await startFakeControl(path.join(botDir, "control.sock"));
  bridgeServer = await startFakeBridge(path.join(botDir, "live.sock"));

  serverProcess = spawn(
    BINARY,
    ["web", "--listen", "127.0.0.1:0", "--data-dir", dataDir, "--bot-socket", path.join(botDir, "live.sock")],
    { stdio: ["ignore", "pipe", "pipe"] },
  );
  serverProcess.stderr.on("data", (chunk) => process.stderr.write(`[ddnet-ai web] ${chunk}`));
  baseUrl = await new Promise((resolve, reject) => {
    let buffer = "";
    const timer = setTimeout(() => reject(new Error(`no listening address within 10s: ${buffer}`)), 10_000);
    const onData = (chunk: Buffer) => {
      buffer += chunk.toString();
      const m = buffer.match(/listening on (http:\/\/\S+)/);
      if (m) {
        clearTimeout(timer);
        serverProcess.stdout.off("data", onData);
        resolve(m[1]);
      }
    };
    serverProcess.stdout.on("data", onData);
  });
});

test.afterAll(async () => {
  serverProcess?.kill();
  clearInterval(bridgeTimer);
  controlServer?.close();
  bridgeServer?.close();
  if (dataDir) rmSync(dataDir, { recursive: true, force: true });
});

async function openBotTab(page: Page) {
  await page.goto(baseUrl);
  await page.locator("#password").fill(password);
  await page.locator("#login-form button[type=submit]").click();
  await expect(page.locator("#tabbar")).toBeVisible();
  await page.locator("#tab-bot").click();
  await expect(page.locator("#bot-view")).toBeVisible();
}

function lastCommand(): any {
  return received[received.length - 1].cmd;
}

test("desktop: status, commands, and the lists editor round-trip", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  const consoleLines: string[] = [];
  page.on("console", (m) => consoleLines.push(m.text()));
  page.on("dialog", (d) => d.accept());
  await openBotTab(page);

  // Status panel.
  await expect(page.locator("#bot-conn-text")).toHaveText("В игре", { timeout: 5_000 });
  await expect(page.locator("#bs-server")).toHaveText("127.0.0.1:8303");
  await expect(page.locator("#bs-map")).toHaveText("Copy Love Box");
  await expect(page.locator("#bs-brain")).toHaveText("hybrid-none-4ms");
  // The select holds the kind of the descriptive name the bot reports (it used to stay blank).
  await expect(page.locator("#cmd-brain")).toHaveValue("hybrid");
  await expect(page.locator("#bs-target")).toHaveText("c5-0a1b2c3d");
  await expect(page.locator("#bs-blocks")).toHaveText("7 / 2");
  await expect(page.locator("#bs-clips")).toHaveText("2");
  await expect(page.locator("#bs-identity")).toContainText("Neuroset");
  // Task 4.11 (D-102): the duel switch. «вкл» is the usual state; a marker or flag that switches the bot's own kills off is flagged, and a
  // status without the field (an older bot) is «—», never «вкл».
  await expect(page.locator("#bs-selfkill")).toHaveText("вкл");
  await expect(page.locator("#bs-selfkill")).not.toHaveClass(/kv-warn/);
  fakeStatus.selfkill = "off";
  await expect(page.locator("#bs-selfkill")).toHaveText("выкл (флажок)", { timeout: 5_000 });
  await expect(page.locator("#bs-selfkill")).toHaveClass(/kv-warn/);
  delete fakeStatus.selfkill;
  await expect(page.locator("#bs-selfkill")).toHaveText("—", { timeout: 5_000 });
  fakeStatus.selfkill = "on";
  await expect(page.locator("#bs-selfkill")).toHaveText("вкл", { timeout: 5_000 });
  await expect(page.locator('[data-cmd="mode"][data-mode="fight"]')).toHaveClass(/current/);
  await expect(page.locator('[data-cmd="wb"][data-mode="auto"]')).toHaveClass(/current/);

  // The kill button waits out its cooldown (120 server ticks at 50 per second = 2.4 s, shown rounded up as 3 s) and says so.
  await expect(page.locator("#cmd-kill")).toBeDisabled();
  await expect(page.locator("#kill-cooldown")).toHaveText("доступно через 3 с");

  // A command reaches the (fake) bot and its reply is shown.
  await page.locator('[data-cmd="mode"][data-mode="hold"]').click();
  await expect(page.locator("#cmd-result")).toContainText("did mode:hold");
  expect(lastCommand()).toEqual({ type: "mode", mode: "hold" });
  expect(received[received.length - 1].session).toMatch(/^[0-9a-f]{16}$/);
  await expect(page.locator('[data-cmd="mode"][data-mode="hold"]')).toHaveClass(/current/, { timeout: 5_000 });

  await page.locator("#cmd-brain").selectOption("planner");
  await page.locator("#cmd-brain-apply").click();
  await expect(page.locator("#cmd-result")).toContainText("did brain");
  expect(lastCommand()).toEqual({ type: "brain", brain: "planner" });

  // The clip warning is on the page, and the note goes out.
  await expect(page.locator(".hint.warn")).toContainText("без ников");
  await page.locator("#cmd-clip-note").fill("nice save");
  await page.locator("#cmd-clip").click();
  await expect(page.locator("#cmd-result")).toContainText("did clip");
  expect(lastCommand()).toEqual({ type: "clip", note: "nice save" });

  await page.locator("#cmd-goto-x").fill("12");
  await page.locator("#cmd-goto-y").fill("34");
  await page.locator("#cmd-goto").click();
  await expect(page.locator("#cmd-result")).toContainText("did goto");
  expect(lastCommand()).toEqual({ type: "goto", x: 12, y: 34 });

  // Kill once the cooldown is over.
  fakeStatus.kill_cooldown_ticks = 0;
  await expect(page.locator("#kill-cooldown")).toHaveText("готово", { timeout: 5_000 });
  await expect(page.locator("#cmd-kill")).toBeEnabled();
  await page.locator("#cmd-kill").click();
  await expect(page.locator("#cmd-result")).toContainText("did kill");

  // The lists editor: normalised preview, add, persisted to the file, shown, removed.
  await page.locator("#rel-kind").selectOption("war");
  await page.locator("#rel-name").fill("  Some   NICK ");
  await expect(page.locator("#rel-preview")).toHaveText("будет сохранено как: «some nick»");
  await page.locator("#rel-form button[type=submit]").click();
  await expect(page.locator("#rel-result")).toContainText("применено к работающему боту");
  await expect(page.locator("#rel-result")).toContainText("Ответ бота");
  await expect(page.locator(".rel-group", { hasText: /^Война \(1\)/ })).toContainText("some nick");
  expect(received.some((r) => r.cmd.type === "reload_relations")).toBe(true);
  const file = path.join(dataDir, "bot", "relations.json");
  expect(JSON.parse(readFileSync(file, "utf8")).war).toEqual(["some nick"]);

  // Markup in a name is text, never markup.
  await page.locator("#rel-kind").selectOption("ignore");
  await page.locator("#rel-name").fill("<b>bold</b>");
  await page.locator("#rel-form button[type=submit]").click();
  await expect(page.locator(".rel-group", { hasText: /^Игнор \(1\)/ })).toContainText("<b>bold</b>");
  expect(await page.locator("#rel-lists b").count()).toBe(0);

  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "5.6-desktop.png"), fullPage: true });

  await page.locator('button[aria-label*="some nick"]').click();
  // Anchored: a plain substring would also match "Клан-война (0)".
  await expect(page.locator(".rel-group", { hasText: /^Война \(0\)/ })).toContainText("пусто");
  await expect.poll(() => JSON.parse(readFileSync(file, "utf8")).war).toEqual([]);

  // A POST without the CSRF token is refused by the real server, from a real browser.
  const status = await page.evaluate(async () => {
    const r = await fetch("/api/bot/command", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type: "stop" }),
    });
    return r.status;
  });
  expect(status).toBe(403);

  // The nicknames I typed are nowhere in the browser console.
  expect(consoleLines.join("\n").toLowerCase()).not.toContain("some nick");
});

test("phone (360x740): the tab fits and works", async ({ page }) => {
  await page.setViewportSize({ width: 360, height: 740 });
  await openBotTab(page);
  await expect(page.locator("#bot-conn-text")).toHaveText("В игре", { timeout: 5_000 });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(0);
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "5.6-phone.png"), fullPage: true });
  // Logging out closes the tab's routes.
  await page.locator("#tab-status").click();
  await page.locator("#logout-button").click();
  await expect(page.locator("#login-view")).toBeVisible();
  await expect(page.locator("#bot-view")).toBeHidden();
  expect(existsSync(path.join(dataDir, "bot", "control.sock"))).toBe(true);
});
