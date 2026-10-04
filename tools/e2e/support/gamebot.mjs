// A scripted FAKE bot for the «Игра» tab tests (task 5.10): speaks the bot's bridge protocol (docs/formats.md §21.2) on a Unix
// socket, written here by hand and independent of `ddai-bot`. It plays a map it is given (real maps are read from the local
// map cache, never from git) with a few tees that walk, jump, hook, freeze and aim, says some chat, and sends the status.
// No game server, no DDNet connection of any kind.
//
// Used by game-view.spec.ts and by game-stack.mjs (a stack for looking at the page by hand).

import crypto from "node:crypto";
import fs from "node:fs";
import net from "node:net";

const TICK_MS = 20;

function message(kind, payload) {
  const head = Buffer.alloc(5);
  head.writeUInt32LE(payload.length + 1, 0);
  head.writeUInt8(kind, 4);
  return Buffer.concat([head, payload]);
}

/** One 26-byte character record of a DWLF v1 frame. */
function charRecord(c) {
  const b = Buffer.alloc(26);
  let flags = 1; // alive
  if (c.frozen) flags |= 2;
  if (c.deep) flags |= 4;
  if (c.hookOut) flags |= 16;
  b.writeUInt8(c.id, 0);
  b.writeUInt8(flags, 1);
  b.writeUInt8(c.team ?? 0, 2);
  b.writeUInt8(c.weapon ?? 1, 3);
  b.writeInt32LE(Math.round(c.x), 4);
  b.writeInt32LE(Math.round(c.y), 8);
  b.writeInt16LE(Math.round(c.aimX), 12);
  b.writeInt16LE(Math.round(c.aimY), 14);
  b.writeInt32LE(Math.round(c.hookX ?? 0), 16);
  b.writeInt32LE(Math.round(c.hookY ?? 0), 20);
  b.writeInt8(c.hooked ?? -1, 24);
  return b;
}

export function frameBytes(tick, chars) {
  const head = Buffer.alloc(12);
  head.write("DWLF", 0, "latin1");
  head.writeUInt8(1, 4);
  head.writeUInt32LE(tick, 6);
  head.writeUInt16LE(chars.length, 10);
  return Buffer.concat([head, ...chars.map(charRecord)]);
}

/**
 * options:
 *   sock        path of the Unix socket to listen on
 *   mapFile     path of the .map file (its sha256 is computed); mapName its display name (the file is `<name>_<sha>.map` in the web's maps dir)
 *   mapW, mapH  the map's size in tiles (only informational)
 *   tees        [{id, name, skin, cc, cb, cf, clan, base:{x,y}, amp, period, weapon, ...}]
 *   chat        [[team, cid, name, text], ...] sent once, a moment after connect
 *   ownId       the owner's bot (the status says so)
 */
export function startGameBot(options) {
  const { sock, mapFile, mapName, mapW, mapH, tees, chat = [], ownId = 0, hz = 25 } = options;
  const mapBytes = fs.readFileSync(mapFile);
  const sha = crypto.createHash("sha256").update(mapBytes).digest("hex");
  const clients = new Set();
  const received = { subscriptions: [] };
  let tick = 100000;
  const t0 = Date.now();

  function pose(t, now) {
    // Walk along the platform, jump now and then, aim around, freeze for a second every few seconds.
    const s = (now - t0) / 1000;
    const phase = t.phase ?? 0;
    const w = (s / (t.period ?? 6) + phase) * Math.PI * 2;
    const x = t.base.x + Math.sin(w) * (t.amp ?? 100);
    const jumpT = ((s + phase * 7) % (t.jumpEvery ?? 3.2)) / 0.9;
    const hop = jumpT < 1 ? Math.sin(jumpT * Math.PI) * (t.jump ?? 80) : 0;
    const y = t.base.y - hop;
    const frozen = t.freezeEvery ? ((s + phase * 3) % t.freezeEvery) < 1.4 : !!t.frozen;
    const aimA = (t.aimBase ?? 0) + Math.sin(s * 0.8 + phase * 5) * (t.aimSwing ?? 1.2);
    const rec = {
      id: t.id,
      x,
      y,
      frozen,
      deep: !!t.deep,
      weapon: frozen ? 0 : t.weapon ?? 1,
      team: 0,
      aimX: Math.cos(aimA) * 200,
      aimY: Math.sin(aimA) * 200,
      hookOut: false,
      hooked: -1,
    };
    if (t.hookTo !== undefined && !frozen) {
      const target = tees.find((o) => o.id === t.hookTo);
      if (target) {
        const phaseH = (s * 0.5 + phase) % 1;
        if (phaseH < 0.55) {
          const tp = pose(target, now, true);
          const k = Math.min(1, phaseH / 0.3);
          rec.hookOut = true;
          rec.hookX = rec.x + (tp.x - rec.x) * k;
          rec.hookY = rec.y + (tp.y - rec.y) * k;
          rec.aimX = tp.x - rec.x;
          rec.aimY = tp.y - rec.y;
          if (k >= 1) rec.hooked = target.id;
        }
      }
    }
    return rec;
  }

  const players = Buffer.from(
    JSON.stringify({ own: ownId, list: tees.map((t) => ({ id: t.id, name: t.name ?? `c${t.id}-0000000${t.id}`, team: 0 })) }),
  );
  const info = Buffer.from(
    JSON.stringify({
      list: tees.map((t) => ({
        id: t.id,
        clan: t.clan ?? "",
        skin: t.skin ?? "default",
        cc: !!t.cc,
        cb: t.cb ?? 0,
        cf: t.cf ?? 0,
        country: t.country ?? -1,
        score: t.score ?? 0,
        ping: t.ping ?? 20,
      })),
    }),
  );
  const mapMsg = Buffer.from(JSON.stringify({ name: mapName, sha256: sha, w: mapW ?? 0, h: mapH ?? 0 }));

  function statusJson(now) {
    const own = tees.find((t) => t.id === ownId);
    const p = own ? pose(own, now) : { frozen: false };
    return Buffer.from(
      JSON.stringify({
        tick, own: ownId, target: options.targetId ?? 1, mode: "fight", brain: "hybrid", alive: true, frozen: p.frozen,
        blocks: 7, blocked_by: 2, self_kills: 0, decisions: 99, collapsed: 0, decide_p50_us: 800, decide_p99_us: 4100,
        brain_p99_us: 3900, overhead_p99_us: 200, telemetry: null, connected: true, server: "127.0.0.1:8303", map: mapName,
        name: "bot", clan: "Neuroset", skin: "default", target_tag: "c1-0a1b2c3d", wb: "WB: auto", goto: "", deaths: 4,
        clips_saved: 2, kill_cooldown_ticks: 0,
      }),
    );
  }

  const server = net.createServer((conn) => {
    clients.add(conn);
    conn.on("close", () => clients.delete(conn));
    conn.on("error", () => clients.delete(conn));
    conn.on("data", (d) => {
      // The web unit's subscription: u32 len | 1 | mask
      for (let i = 0; i + 6 <= d.length; i += 6) if (d[i + 4] === 1) received.subscriptions.push(d[i + 5]);
    });
    conn.write(message(1, Buffer.from("DDBL\x01", "latin1")));
    conn.write(message(2, mapMsg));
    conn.write(message(3, players));
    conn.write(message(9, info));
    setTimeout(() => {
      for (const [team, cid, name, text] of chat) {
        conn.write(message(8, Buffer.from(JSON.stringify({ team, cid, name, text }))));
      }
    }, 1200);
  });
  fs.rmSync(sock, { force: true });
  server.listen(sock);

  const timer = setInterval(() => {
    const now = Date.now();
    tick += Math.round(1000 / hz / TICK_MS);
    const chars = tees.map((t) => pose(t, now));
    const frame = message(4, frameBytes(tick, chars));
    for (const c of clients) c.write(frame);
  }, 1000 / hz);
  const statusTimer = setInterval(() => {
    const m = message(5, statusJson(Date.now()));
    for (const c of clients) c.write(m);
  }, 200);

  return {
    sha,
    received,
    say(team, cid, name, text) {
      const m = message(8, Buffer.from(JSON.stringify({ team, cid, name, text })));
      for (const c of clients) c.write(m);
    },
    close() {
      clearInterval(timer);
      clearInterval(statusTimer);
      for (const c of clients) c.destroy();
      server.close();
    },
  };
}
