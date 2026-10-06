// The path units of the server browser and the launcher, emulated for the local e2e (task 5.12): a poll every 250 ms (no shell loop, no
// process search) that does what ddnet-ai-launch.path, ddnet-ai-proxycheck.path and ddnet-ai-servers.path do on this machine's production
// units, but against the e2e directory and the e2e fake `systemctl`:
//   data/launch/request.json            -> `ddnet-ai launch apply`     (the REAL helper, the fake systemctl)
//   data/launch/proxycheck-request.json -> `ddnet-ai launch check-proxy`
//   data/launch/servers-refresh changed -> `ddnet-ai servers-cache`    (the REAL master list, read-only HTTPS)
// Usage: node launcher-sim.mjs <e2e-dir> <ddnet-ai binary>. Stops on SIGTERM.
import { spawnSync } from "node:child_process";
import { existsSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";

const [, , dir, bin] = process.argv;
if (!dir || !bin) {
  console.error("usage: launcher-sim.mjs <e2e-dir> <ddnet-ai>");
  process.exit(2);
}
const data = path.join(dir, "data");
const launch = path.join(data, "launch");
const env = {
  ...process.env,
  E2E_DIR: dir,
  E2E_BIN: bin,
  PATH: `${path.join(dir, "shim")}:${process.env.PATH}`,
  NO_COLOR: "1",
};
const log = (...a) => console.log(new Date().toISOString(), ...a);

const applyArgs = [
  "launch", "apply", "--data-dir", data, "--status-dir", path.join(dir, "status"), "--config", path.join(dir, "none.toml"),
  "--env-file", path.join(dir, "etc", "bot-launch.env"), "--dropin", path.join(dir, "etc", "50-launch.conf"),
  "--state", path.join(dir, "var", "state.json"),
];

function mtime(p) {
  try {
    return statSync(p).mtimeMs;
  } catch {
    return 0;
  }
}

let lastRefresh = mtime(path.join(launch, "servers-refresh"));
let busy = false;

function run(label, args) {
  const r = spawnSync(bin, args, { env, encoding: "utf8", timeout: 120_000 });
  log(label, "exit", r.status, (r.stderr || "").trim().split("\n").slice(-1)[0] ?? "");
}

function tick() {
  if (busy) {
    return;
  }
  busy = true;
  try {
    if (existsSync(path.join(launch, "request.json"))) {
      run("launch apply", applyArgs);
    }
    if (existsSync(path.join(launch, "proxycheck-request.json"))) {
      run("check-proxy", ["launch", "check-proxy", "--data-dir", data]);
    }
    const m = mtime(path.join(launch, "servers-refresh"));
    if (m !== lastRefresh) {
      lastRefresh = m;
      run("servers-cache", ["servers-cache", "--data-dir", data]);
    }
  } finally {
    busy = false;
  }
}

writeFileSync(path.join(dir, "sim.pid"), String(process.pid));
const timer = setInterval(tick, 250);
process.on("SIGTERM", () => {
  clearInterval(timer);
  process.exit(0);
});
log("launcher-sim up", dir);
