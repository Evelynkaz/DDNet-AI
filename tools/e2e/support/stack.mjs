// A real `ddnet-ai web` on an ephemeral loopback port in a scratch data directory, fed by the scripted FAKE bot of gamebot.mjs
// (task 5.10). Never the production unit; no public server. `startStack()` resolves with the base URL, the password and a
// `stop()`; run this file directly (`node support/stack.mjs`) to get one to look at by hand.

import { spawn, execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { startGameBot } from "./gamebot.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = path.resolve(HERE, "..", "..", "..");
export const BINARY = path.join(REPO_ROOT, "target", "debug", "ddnet-ai");
export const MAPS = path.join(os.homedir(), "aiddnet", "data", "maps", "cache");
export const DDNET_DATA = path.join(os.homedir(), "aiddnet", "build", "ddnet-20.1", "build", "data");

export function findMap(prefix) {
  const f = fs.readdirSync(MAPS).find((n) => n.startsWith(prefix) && n.endsWith(".map"));
  return f ? path.join(MAPS, f) : null;
}

export async function startStack(opts = {}) {
  const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), "ddai-game-"));
  fs.mkdirSync(path.join(dataDir, "bot"), { recursive: true });
  const out = execFileSync(BINARY, ["web-passwd", "--data-dir", dataDir, "--show"], { encoding: "utf8" });
  const password = out.match(/^password: (\S+)$/m)[1];
  const sock = path.join(dataDir, "bot", "live.sock");
  const bot = startGameBot({ sock, ...opts.bot });
  const args = ["web", "--listen", "127.0.0.1:0", "--data-dir", dataDir, "--bot-socket", sock, "--maps-dir", MAPS];
  if (opts.ddnetData !== null) args.push("--ddnet-data", opts.ddnetData ?? DDNET_DATA);
  const proc = spawn(BINARY, args, { stdio: ["ignore", "pipe", "pipe"] });
  proc.stderr.on("data", (c) => process.stderr.write(`[web] ${c}`));
  const baseUrl = await new Promise((resolve, reject) => {
    let buf = "";
    const timer = setTimeout(() => reject(new Error("web did not start")), 15000);
    proc.stdout.on("data", (c) => {
      buf += c.toString();
      const m = buf.match(/listening on (http:\/\/\S+)/);
      if (m) {
        clearTimeout(timer);
        resolve(m[1]);
      }
    });
  });
  return {
    baseUrl,
    password,
    bot,
    dataDir,
    stop() {
      proc.kill();
      bot.close();
      fs.rmSync(dataDir, { recursive: true, force: true });
    },
  };
}

/** The scene most screenshots use: Copy Love Box, five tees with stock skins, one of them the owner's bot. */
export function clbOptions() {
  const mapFile = findMap("Copy Love Box_6e79ef");
  // Standing places in the arena of Copy Love Box (game layer rows 18 and 20: the platform in the middle, the two ledges).
  const tees = [
    { id: 0, name: "Муха", clan: "Neuroset", skin: "default", base: { x: 3740, y: 626 }, amp: 50, period: 7, weapon: 0, hookTo: 1, score: 14, ping: 12, aimBase: 0.3, jumpEvery: 4, jump: 60 },
    { id: 1, name: "Kasper", clan: "", skin: "greensward", base: { x: 3830, y: 626 }, amp: 30, period: 5, weapon: 1, freezeEvery: 6, phase: 0.2, score: 9, ping: 33 },
    { id: 2, name: "brainless", clan: "Gores", skin: "coala", cc: true, cb: 0x00ff6090, cf: 0x00ffc0a0, base: { x: 3250, y: 562 }, amp: 40, period: 6, weapon: 2, phase: 0.4, score: 3, ping: 41, aimBase: 3.0 },
    { id: 3, name: "sasun", clan: "[D]", skin: "x_ninja", base: { x: 4300, y: 562 }, amp: 40, period: 4, weapon: 5, phase: 0.6, score: 0, ping: 58, aimBase: 2.9 },
    { id: 4, name: "nameless tee", clan: "AI", skin: "pinky", base: { x: 4400, y: 562 }, amp: 30, period: 8, weapon: 3, deep: false, phase: 0.8, score: 21, ping: 26, aimBase: 3.3 },
  ];
  return {
    bot: {
      mapFile,
      mapName: "Copy Love Box",
      mapW: 387,
      mapH: 250,
      tees,
      ownId: 0,
      chat: [
        [0, 1, "Kasper", "привет, кто на блок?"],
        [0, 2, "brainless", "gg"],
        [1, 4, "nameless tee", "team: держим левый вб"],
        [0, -1, "", "'Kasper' entered the game"],
        [0, 3, "sasun", "<img src=x onerror=alert(1)> <b>bold</b> &amp; ‮gnirts"],
      ],
    },
  };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const s = await startStack(clbOptions());
  console.log(`url ${s.baseUrl}\npassword ${s.password}`);
  process.on("SIGINT", () => {
    s.stop();
    process.exit(0);
  });
}
