// Real-browser test of the «Обучение» tab (task 5.8): the list of experiments and runs, a run's curves with DAgger markers,
// the arena table with confidence intervals, the checkpoint list and the 2-4 run comparison, on desktop and on a phone.
//
// Two parts, both against a real `ddnet-ai web` process (built beforehand with `cargo build -p ddnet-ai`) on an ephemeral port
// in a scratch data directory, never the production unit:
//   1. a SCRATCH runs directory with synthetic data written here (including a run that "trains" while the page is open, to
//      check the low-rate refresh), screenshots `~/aiddnet/data/screenshots/5.8-*.png`;
//   2. one READ-ONLY check on the real `~/aiddnet/data/runs` (E-005 / E-008): the page shows them, and nothing in that tree
//      changes (sizes and mtimes of every entry are compared before and after).
// Screenshots are never in git. How to run: see README.md in this folder.

import { test, expect, type Page } from "@playwright/test";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { mkdtempSync, mkdirSync, writeFileSync, appendFileSync, existsSync, rmSync, readdirSync, statSync, utimesSync } from "node:fs";
import { tmpdir, homedir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, "..", "..");
const BINARY = path.join(REPO_ROOT, "target", "debug", "ddnet-ai");
const SCREENSHOT_DIR = path.join(homedir(), "aiddnet", "data", "screenshots");
const REAL_RUNS = path.join(homedir(), "aiddnet", "data", "runs");

type Web = { proc: ChildProcessWithoutNullStreams; baseUrl: string; dataDir: string; password: string };

function runCli(args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const proc = spawn(BINARY, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    proc.stdout.on("data", (c) => (stdout += c.toString()));
    proc.stderr.on("data", (c) => (stderr += c.toString()));
    proc.on("error", reject);
    proc.on("exit", (code) => (code === 0 ? resolve(stdout) : reject(new Error(`${args.join(" ")} exited ${code}: ${stderr}`))));
  });
}

async function startWeb(runsDir: string): Promise<Web> {
  const dataDir = mkdtempSync(path.join(tmpdir(), "ddai-web-train-"));
  const out = await runCli(["web-passwd", "--data-dir", dataDir, "--show"]);
  const m = out.match(/^password: (\S+)$/m);
  if (!m) throw new Error("no password in web-passwd output");
  const proc = spawn(BINARY, ["web", "--listen", "127.0.0.1:0", "--data-dir", dataDir, "--runs-dir", runsDir], {
    stdio: ["ignore", "pipe", "pipe"],
  });
  proc.stderr.on("data", (c) => process.stderr.write(`[ddnet-ai web] ${c}`));
  const baseUrl: string = await new Promise((resolve, reject) => {
    let buffer = "";
    const timer = setTimeout(() => reject(new Error(`no listening address within 10s: ${buffer}`)), 10_000);
    const onData = (chunk: Buffer) => {
      buffer += chunk.toString();
      const mm = buffer.match(/listening on (http:\/\/\S+)/);
      if (mm) {
        clearTimeout(timer);
        proc.stdout.off("data", onData);
        resolve(mm[1]);
      }
    };
    proc.stdout.on("data", onData);
  });
  return { proc, baseUrl, dataDir, password: m[1] };
}

function stopWeb(web?: Web) {
  web?.proc.kill();
  if (web) rmSync(web.dataDir, { recursive: true, force: true });
}

async function openTrainTab(page: Page, web: Web) {
  await page.goto(web.baseUrl);
  await page.locator("#password").fill(web.password);
  await page.locator("#login-form button[type=submit]").click();
  await expect(page.locator("#tabbar")).toBeVisible();
  await page.locator("#tab-train").click();
  await expect(page.locator("#train-view")).toBeVisible();
}

// ---- synthetic runs ---------------------------------------------------------------------------------

function lcg(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32);
}

function wilson(k: number, n: number): [number, number, number] {
  const z = 1.959963984540054;
  const p = k / n;
  const d = 1 + (z * z) / n;
  const c = (p + (z * z) / (2 * n)) / d;
  const h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [p, Math.max(0, c - h), Math.min(1, c + h)];
}

const ARENAS: [string, number][] = [
  ["clb-left", 0.12],
  ["pit", 0.3],
  ["platform", 0.4],
];

function phaseName(i: number): string {
  return i === 0 ? "bc" : `dagger-${i}`;
}

/** metrics.jsonl lines of a synthetic run: `phases` phases of `perPhase` steps, logged every 50. */
function synthMetrics(opts: { seed: number; phases: number; perPhase: number; strength: number }): string[] {
  const rnd = lcg(opts.seed);
  const lines: string[] = [];
  let step = 0;
  for (let ph = 0; ph < opts.phases; ph++) {
    const phase = phaseName(ph);
    for (let k = 1; k <= opts.perPhase / 50; k++) {
      step += 50;
      const decay = Math.exp(-step / 2500);
      const noise = () => (rnd() - 0.5) * 0.06;
      lines.push(
        JSON.stringify({
          kind: "train", phase, step, grad_norm: 3 + noise(), lr_mult: 1, skipped: 0, unix_s: 1790000000 + step,
          loss: {
            total: 1.6 + 1.8 * decay + noise() - 0.1 * opts.strength, dir: 0.85 + 0.2 * decay + noise(), jump: 0.9 + noise(),
            hook: 0.55 + 0.1 * decay + noise(), fire: 0.9 + noise(), aim: -1.5 + 1.2 * decay + noise(),
          },
        }),
      );
    }
    if (ph >= 1) {
      lines.push(JSON.stringify({ kind: "hook_play", round: ph, report: {
        start_student: 0.3 + 0.03 * ph * opts.strength, start_teacher: 0.4, release_student: 0.02 + 0.02 * ph * opts.strength, release_teacher: 0.2, steps: 49000,
      } }));
    }
    for (const set of ph === 0 ? ["teacher-val"] : ["teacher-val", "dagger-val"]) {
      lines.push(JSON.stringify({ kind: "eval", phase, set, step, unix_s: 1790000000 + step, report: {
        dir: { accuracy: 0.66 + 0.02 * ph * opts.strength + (set === "dagger-val" ? -0.03 : 0), top2_accuracy: 0.92 },
        hook: { auroc: 0.7 + 0.025 * ph * opts.strength + (set === "dagger-val" ? -0.04 : 0) },
        jump: { auroc: 0.67 }, fire: { auroc: 0.73 }, aim: { within_15deg: 0.39 },
        hook_by_state: { start_accuracy: 0.48, release_accuracy: 0.09 },
      } }));
    }
    lines.push(JSON.stringify({ kind: "thresholds", phase, step, jump: 0.54, hook: 0.62, fire: 0.53, sets: ["teacher-val"], unix_s: 1790000000 + step }));
    for (const [arena, base] of ARENAS) {
      const games = 300;
      const w = Math.round(games * Math.min(0.95, base + 0.05 * ph * opts.strength + rnd() * 0.02));
      const cw = Math.round(w * 0.5);
      lines.push(JSON.stringify({ kind: "arena", phase, step, eval: {
        arena, games, w, l: games - w - 20, d: 4, t: 16, credited_w: cw, credited_win_rate: wilson(cw, games),
        win_rate: wilson(w, w + (games - w - 20) + 4), win_rate_all: wilson(w, games), blocks_per_min: 3.1 * opts.strength, self_freezes_per_min: 3.8,
        opponents: ["scripted"], decide_us_p50: 230, decide_us_p99: 310,
      } }));
    }
    if (ph < opts.phases - 1) {
      lines.push(JSON.stringify({ kind: "collect", round: ph + 1, beta: 0.5 - 0.1 * ph, jobs_resumed: 0, jobs: [
        { arena: "clb-left", d: 2, games: 250, l: 26, opponents: 1, steps: 38735, t: 0, w: 222 },
        { arena: "pit", d: 0, games: 60, l: 0, opponents: 1, steps: 34277, t: 31, w: 29 },
      ] }));
    }
  }
  lines.push(JSON.stringify({ kind: "selection", phase: phaseName(opts.phases - 1), arenas: ["clb-left", "pit", "platform"],
    table: Array.from({ length: opts.phases }, (_, i) => [phaseName(i), 0.02 * i * opts.strength]) }));
  return lines;
}

function writeRun(root: string, exp: string, name: string, o: {
  kind: "fly" | "mlp"; seed: number; strength: number; phases: number; perPhase: number; state: "done" | "running" | "stalled"; ownHook: string;
  /** Junk lines (~1 KB each) appended to metrics.jsonl, so reading and parsing a run takes real time. */
  pad?: number;
}) {
  const dir = path.join(root, exp, name);
  mkdirSync(path.join(dir, "checkpoints"), { recursive: true });
  mkdirSync(path.join(dir, "rounds"), { recursive: true });
  const lines = synthMetrics({ seed: o.seed, phases: o.phases, perPhase: o.perPhase, strength: o.strength });
  for (let i = 0; i < (o.pad ?? 0); i++) lines.push(JSON.stringify({ kind: "debug", i, pad: "x".repeat(1000) }));
  writeFileSync(path.join(dir, "metrics.jsonl"), lines.join("\n") + "\n");
  const total = o.phases * o.perPhase;
  const planned = 2500 + 5 * 800;
  writeFileSync(
    path.join(dir, "config.toml"),
    `name = "${name}"\nbc_steps = 2500\n[model]\nkind = "${o.kind}"\nhidden = ${o.kind === "mlp" ? 64 : 0}\n[train]\nseed = ${o.seed}\nhuman_fraction = 0.1\n[train.own_hook]\nmode = "${o.ownHook}"\n[dagger]\nbetas = [0.5, 0.4, 0.3, 0.2, 0.15]\nsteps_per_round = 800\neval_games = 300\neval_arenas = ["clb-left", "pit", "platform"]\n`,
  );
  writeFileSync(
    path.join(dir, "status.json"),
    o.state === "done" ? JSON.stringify({ phase: "done", step: total }) : JSON.stringify({ phase: phaseName(o.phases - 1), step: total, phase_step: 400, phase_steps: o.perPhase, loss: 1.7, elapsed_s: 123.4, unix_s: 1790000000 }),
  );
  for (const f of ["last", "final", `step-${String(total).padStart(8, "0")}`]) {
    writeFileSync(path.join(dir, "checkpoints", `${f}.bundle`), Buffer.alloc(24000 + (f.length % 5) * 100, f.length));
  }
  for (let r = 0; r < o.phases; r++) writeFileSync(path.join(dir, "rounds", `round-${r}.bundle`), Buffer.alloc(23900, r + 1));
  if (o.state === "stalled") {
    const old = new Date(Date.now() - 5 * 3600 * 1000);
    for (const f of ["status.json", "metrics.jsonl"]) utimesSync(path.join(dir, f), old, old);
  }
}

function writeEvalSummary(root: string, exp: string, run: string) {
  const dir = path.join(root, exp, "eval", run, "arena");
  mkdirSync(dir, { recursive: true });
  const cond = (name: string, arena: string, w: number, cw: number, games: number) => ({
    name, arena, arena_tag: "train", games, tally: { w, l: games - w - 30, d: 5, t: 25 },
    win_rate: { p: w / (games - 25), lo: 0.5, hi: 0.58 }, win_rate_all: { p: w / games, lo: 0.47, hi: 0.53 },
    credited_win_rate: { p: cw / games, lo: wilson(cw, games)[1], hi: wilson(cw, games)[2] }, credited_w: cw,
    blocks_per_min: 0.19, self_freezes_per_min: 2.48, players: [{ name: "synthetic" }],
  });
  writeFileSync(path.join(dir, "summary.json"), JSON.stringify({
    meta: { run_name: "synthetic evaluation", git_commit: "0123456789abcdef0123", git_dirty: false, base_seed: 1 },
    conditions: [
      cond("clb-left vs scripted", "clb-left", 502, 28, 1000),
      cond("clb-right vs scripted", "clb-right", 430, 9, 1000),
      cond("chillblock5-ruler vs scripted", "chillblock5-ruler", 610, 61, 1000),
    ],
  }));
}

let web: Web | undefined;
let scratch: string;

test.describe.configure({ mode: "serial" });

test.beforeAll(async () => {
  mkdirSync(SCREENSHOT_DIR, { recursive: true });
  scratch = mkdtempSync(path.join(tmpdir(), "ddai-runs-"));
  writeRun(scratch, "E-101", "e101-fly-base-s1", { kind: "fly", seed: 1, strength: 1.0, phases: 6, perPhase: 800, state: "done", ownHook: "off", pad: 1500 });
  writeRun(scratch, "E-101", "e101-fly-maskhook-s1", { kind: "fly", seed: 1, strength: 1.4, phases: 6, perPhase: 800, state: "done", ownHook: "mask_hook_head", pad: 1500 });
  writeRun(scratch, "E-101", "e101-mlpw-base-s1", { kind: "mlp", seed: 1, strength: 0.8, phases: 6, perPhase: 1000, state: "done", ownHook: "off", pad: 1500 });
  writeRun(scratch, "E-101", "e101-mlpw-maskhook-s1", { kind: "mlp", seed: 1, strength: 1.2, phases: 6, perPhase: 1000, state: "done", ownHook: "mask_hook_head", pad: 1500 });
  writeRun(scratch, "E-101", "e101-mlpw-maskhook-s2", { kind: "mlp", seed: 2, strength: 1.1, phases: 5, perPhase: 1000, state: "running", ownHook: "mask_hook_head" });
  writeRun(scratch, "E-100", "old-crashed", { kind: "fly", seed: 7, strength: 0.5, phases: 3, perPhase: 800, state: "stalled", ownHook: "off" });
  writeEvalSummary(scratch, "E-101", "e101-fly-base-s1");
  // Things that are not runs: ignored.
  mkdirSync(path.join(scratch, "E-101", "eval", "scratch-dir"), { recursive: true });
  writeFileSync(path.join(scratch, "E-101", "notes.txt"), "x");
  web = await startWeb(scratch);
});

test.afterAll(() => {
  stopWeb(web);
  if (scratch) rmSync(scratch, { recursive: true, force: true });
});

function noHorizontalScroll(page: Page) {
  return page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1);
}

async function runScenario(page: Page, label: "desktop" | "phone") {
  const consoleErrors: string[] = [];
  page.on("console", (m) => m.type() === "error" && consoleErrors.push(m.text()));
  page.on("pageerror", (e) => consoleErrors.push(String(e)));
  await openTrainTab(page, web!);

  // --- the list
  const list = page.locator("#train-list");
  await expect(list.locator(".tr-run")).toHaveCount(6);
  await expect(page.locator("#train-state")).toContainText("Запусков: 6, идёт: 1");
  await expect(list.locator(".tr-state.running")).toHaveCount(1);
  await expect(list.locator(".tr-state.done")).toHaveCount(4);
  await expect(list.locator(".tr-state.stalled")).toHaveCount(1);
  await expect(list).toContainText("e101-mlpw-maskhook-s2");
  expect(await noHorizontalScroll(page)).toBe(true);
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, `5.8-${label}-list.png`), fullPage: true });

  // --- a run: curves with DAgger markers, tables
  await list.getByRole("button", { name: "e101-fly-base-s1" }).click();
  const detail = page.locator("#train-detail");
  await expect(detail.locator("h2").first()).toContainText("E-101 / e101-fly-base-s1");
  const charts = detail.locator(".tr-chart svg.tr-svg");
  await expect(charts).toHaveCount(5);
  for (let i = 0; i < 5; i++) {
    await expect(charts.nth(i).locator("path.tr-line").first()).toBeVisible();
  }
  // Loss chart: six series, five DAgger round markers.
  await expect(charts.first().locator("path.tr-line")).toHaveCount(6);
  await expect(charts.first().locator("line.tr-mark")).toHaveCount(5);
  await expect(charts.first().locator("text", { hasText: "D1" })).toHaveCount(1);
  // Tooltip on hover.
  await charts.first().scrollIntoViewIfNeeded();
  const box = (await charts.first().boundingBox())!;
  await charts.first().hover({ position: { x: box.width * 0.6, y: box.height * 0.5 } });
  await expect(detail.locator(".tr-tip").first()).toBeVisible();
  await expect(detail.locator(".tr-tip").first()).toContainText("шаг");
  // Arena table: the headline metric with a Wilson interval, the eval summary with three conditions.
  const arena = detail.locator("table.tr-arena").first();
  await expect(arena).toContainText("clb-left");
  await expect(arena).toContainText(/\d+,\d%\s\[\d+,\d–\d+,\d\]/);
  await expect(arena.locator("svg.tr-ci-svg")).toHaveCount(3);
  await detail.locator("details.tr-sub summary").click();
  await expect(detail.locator("details.tr-sub table")).toContainText("chillblock5-ruler vs scripted");
  // DAgger rounds table.
  await expect(detail).toContainText("Раунды DAgger");
  await expect(detail.locator("table", { hasText: "β (доля учителя)" }).locator("tbody tr")).toHaveCount(5);
  // Checkpoints: names, sizes, hash prefixes (16 hex), never bytes.
  const ck = detail.locator("table", { hasText: "sha256 (16)" });
  await expect(ck.locator("tbody tr")).toHaveCount(3 + 6);
  await expect(ck).toContainText("final.bundle");
  await expect(ck.locator("tbody tr").first()).toContainText(/[0-9a-f]{16}/);
  await detail.locator("section.card").nth(1).screenshot({ path: path.join(SCREENSHOT_DIR, `5.8-${label}-curves.png`) });
  expect(await noHorizontalScroll(page)).toBe(true);
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, `5.8-${label}-run.png`), fullPage: true });
  // A table view under a chart.
  await charts.first().locator("xpath=ancestor::div[contains(@class,'tr-chart')]").locator("details.tr-tbl summary").click();
  await expect(detail.locator("details.tr-tbl table").first()).toBeVisible();

  // --- compare 4 runs, a 5th is refused
  const boxes = list.locator(".tr-pick input");
  for (const name of ["e101-fly-base-s1", "e101-fly-maskhook-s1", "e101-mlpw-base-s1", "e101-mlpw-maskhook-s1"]) {
    await list.getByLabel(`Сравнить: ${name}`).check();
  }
  await list.getByLabel("Сравнить: e101-mlpw-maskhook-s2").click();
  await expect(list.getByLabel("Сравнить: e101-mlpw-maskhook-s2")).not.toBeChecked();
  await expect(page.locator("#train-pick-note")).toContainText("Не больше 4");
  await expect(page.locator("#train-compare-btn")).toBeEnabled();
  await page.locator("#train-compare-btn").click();
  const cmp = page.locator("#train-compare");
  await expect(cmp.locator("h2").first()).toContainText("Сравнение запусков (4)");
  const cmpCharts = cmp.locator(".tr-chart svg.tr-svg");
  await expect(cmpCharts).toHaveCount(4);
  for (let i = 0; i < 4; i++) {
    await expect(cmpCharts.nth(i).locator("path.tr-line")).toHaveCount(4);
  }
  await expect(cmp.locator("table", { hasText: "Последние значения" }).or(cmp.locator("section.card table")).first()).toContainText("e101-mlpw-base-s1");
  // The arena / set selectors redraw the overlays.
  await cmp.locator("select").nth(1).selectOption("pit");
  await expect(cmp.locator(".tr-chart h3").nth(3)).toContainText("pit");
  expect(await noHorizontalScroll(page)).toBe(true);
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, `5.8-${label}-compare.png`), fullPage: true });
  await cmp.locator("section.card").nth(1).screenshot({ path: path.join(SCREENSHOT_DIR, `5.8-${label}-compare-charts.png`) });
  await cmp.locator("section.card").nth(2).screenshot({ path: path.join(SCREENSHOT_DIR, `5.8-${label}-compare-table.png`) });
  // Two is the minimum.
  await page.locator("#train-compare-clear").click();
  await expect(page.locator("#train-compare-btn")).toBeDisabled();

  expect(consoleErrors).toEqual([]);
}

test("desktop: list, run curves with DAgger markers, arena CIs, checkpoints, compare overlay", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await runScenario(page, "desktop");
});

test("phone: the same on 360x740 without horizontal scroll", async ({ page }) => {
  await page.setViewportSize({ width: 360, height: 740 });
  await runScenario(page, "phone");
  // The six-tab bar fits (task 5.12 added «Серверы» to the five this test was written for).
  const tabs = page.locator("#tabbar button");
  await expect(tabs).toHaveCount(6);
  for (let i = 0; i < 6; i++) {
    const b = (await tabs.nth(i).boundingBox())!;
    expect(b.x + b.width).toBeLessThanOrEqual(361);
    expect(await tabs.nth(i).evaluate((n) => n.scrollWidth <= n.clientWidth + 1)).toBe(true);
    // The label is a span that clips its own overflow with an ellipsis, so the button alone cannot tell: measure the span (task 5.14, review F2).
    expect(await tabs.nth(i).locator("span").evaluate((n) => n.scrollWidth <= n.clientWidth), `tab ${i}: the label is cut`).toBe(true);
  }
});

const FOUR = ["e101-fly-base-s1", "e101-fly-maskhook-s1", "e101-mlpw-base-s1", "e101-mlpw-maskhook-s1"];

async function openRunAndCompareFour(page: Page) {
  await openTrainTab(page, web!);
  const list = page.locator("#train-list");
  await list.getByRole("button", { name: FOUR[0] }).click();
  await expect(page.locator("#train-detail h2").first()).toContainText(FOUR[0]);
  for (const name of FOUR) await list.getByLabel(`Сравнить: ${name}`).check();
  await page.locator("#train-compare-btn").click();
  await expect(page.locator("#train-compare h2").first()).toContainText("Сравнение запусков (4)");
}

test("re-showing the tab with an open run and a 4-run comparison keeps all four, one request at a time", async ({ page }) => {
  test.setTimeout(90_000);
  await page.setViewportSize({ width: 1280, height: 900 });
  await openRunAndCompareFour(page);
  let inflight = 0;
  let peak = 0;
  const statuses: number[] = [];
  page.on("request", (r) => {
    if (r.url().includes("/api/train/")) peak = Math.max(peak, ++inflight);
  });
  page.on("requestfinished", (r) => r.url().includes("/api/train/") && inflight--);
  page.on("requestfailed", (r) => r.url().includes("/api/train/") && inflight--);
  page.on("response", (r) => r.url().includes("/api/train/") && statuses.push(r.status()));
  for (let round = 0; round < 6; round++) {
    statuses.length = 0;
    await page.locator("#tab-status").click();
    await page.locator("#tab-train").click();
    // list + open run + 4 comparison runs, all answered
    await expect.poll(() => statuses.length, { timeout: 15_000 }).toBeGreaterThanOrEqual(6);
    await expect.poll(() => inflight, { timeout: 15_000 }).toBe(0);
    expect(statuses, `round ${round}`).toEqual(Array(statuses.length).fill(200));
    await expect(page.locator("#train-compare h2").first()).toContainText("Сравнение запусков (4)");
    await expect(page.locator("#train-compare .tr-legend").first().locator(".tr-legend-item")).toHaveCount(4);
    await expect(page.locator("#train-compare .tr-note")).toHaveCount(0);
  }
  expect(peak, "the page asks for one thing at a time").toBeLessThanOrEqual(2);
});

test("a 503 is retried once; a run that stays unreadable does not recolour the others", async ({ page }) => {
  test.setTimeout(60_000);
  await page.setViewportSize({ width: 1280, height: 900 });
  await openRunAndCompareFour(page);
  const second = `run=${FOUR[1]}`;
  // One 503 for the second run, then normal answers: the retry makes up for it.
  let refused = 0;
  await page.route("**/api/train/run?*", async (route) => {
    if (route.request().url().includes(second) && refused < 1) {
      refused++;
      await route.fulfill({ status: 503, contentType: "application/json", body: '{"error":"busy"}' });
    } else {
      await route.fallback();
    }
  });
  await page.locator("#tab-status").click();
  await page.locator("#tab-train").click();
  await expect.poll(() => refused).toBe(1);
  await expect(page.locator("#train-compare h2").first()).toContainText("Сравнение запусков (4)");
  await expect(page.locator("#train-compare .tr-note")).toHaveCount(0);
  await page.unrouteAll({ behavior: "wait" });

  // The second run stays unreadable: three runs remain, numbered and coloured by their place among the picked four.
  await page.route("**/api/train/run?*", async (route) => {
    if (route.request().url().includes(second)) {
      await route.fulfill({ status: 503, contentType: "application/json", body: '{"error":"busy"}' });
    } else {
      await route.fallback();
    }
  });
  // A fresh comparison (the old data of the second run would otherwise be kept): re-open the tab page.
  await page.reload();
  await page.locator("#tab-train").click();
  const list = page.locator("#train-list");
  for (const name of FOUR) await list.getByLabel(`Сравнить: ${name}`).check();
  await page.locator("#train-compare-btn").click();
  await expect(page.locator("#train-compare h2").first()).toContainText("Сравнение запусков (3)", { timeout: 15_000 });
  await expect(page.locator("#train-compare .tr-note")).toContainText(`#2 E-101/${FOUR[1]}`);
  const items = page.locator("#train-compare .tr-legend").first().locator(".tr-legend-item");
  await expect(items).toHaveCount(3);
  await expect(items.nth(0)).toContainText("#1 ");
  await expect(items.nth(1)).toContainText("#3 ");
  await expect(items.nth(2)).toContainText("#4 ");
  await expect(items.nth(1).locator("line.tr-line")).toHaveClass(/\bs3\b/);
  await expect(items.nth(2).locator("line.tr-line")).toHaveClass(/\bs4\b/);
  await page.unrouteAll({ behavior: "wait" });
});

test("a running job is refreshed on the page at a low rate", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await openTrainTab(page, web!);
  const requests: string[] = [];
  page.on("request", (r) => r.url().includes("/api/train/") && requests.push(r.url()));
  await page.locator("#train-list").getByRole("button", { name: "e101-mlpw-maskhook-s2" }).click();
  const detail = page.locator("#train-detail");
  await expect(detail).toContainText("Идёт обучение");
  await expect(detail.locator("dl.kv").first()).toContainText("5000 из 6500");
  // The job moves on: one more train row and a new status.
  const run = path.join(scratch, "E-101", "e101-mlpw-maskhook-s2");
  appendFileSync(path.join(run, "metrics.jsonl"), JSON.stringify({ kind: "train", phase: "dagger-4", step: 5050, loss: { total: 1.23, dir: 0.8 } }) + "\n");
  writeFileSync(path.join(run, "status.json"), JSON.stringify({ phase: "dagger-4", step: 5050, phase_step: 50, phase_steps: 1000, loss: 1.23, unix_s: 1790000100 }));
  // Within a couple of polls (10 s each) the page shows it, without any reload.
  await expect(detail.locator("dl.kv").first()).toContainText("5050 из 6500", { timeout: 25_000 });
  await expect(page.locator("#train-list")).toContainText("шаг 5050");
  // Low rate: nothing like a flood of requests (open run + list; two ticks at most in this window, a few requests each).
  const polls = requests.filter((u) => u.includes("poll=1")).length;
  expect(polls).toBeGreaterThan(0);
  expect(polls).toBeLessThanOrEqual(8);
  // Leaving the tab stops the polling.
  await page.locator("#tab-status").click();
  const before = requests.length;
  await page.waitForTimeout(12_000);
  expect(requests.length).toBe(before);
});

test("after logout the page cannot read runs", async ({ page }) => {
  await openTrainTab(page, web!);
  expect(await page.evaluate(async () => (await fetch("/api/train/runs", { credentials: "same-origin" })).status)).toBe(200);
  await page.locator("#tab-status").click();
  await page.locator("#logout-button").click();
  await expect(page.locator("#login-view")).toBeVisible();
  expect(await page.evaluate(async () => (await fetch("/api/train/runs", { credentials: "same-origin" })).status)).toBe(401);
  expect(await page.evaluate(async () => (await fetch("/api/train/run?exp=E-101&run=e101-fly-base-s1", { credentials: "same-origin" })).status)).toBe(401);
});

// ---- the real runs, read-only ----------------------------------------------------------------------

function snapshot(root: string): string[] {
  const out: string[] = [];
  const walk = (dir: string) => {
    for (const name of readdirSync(dir).sort()) {
      const p = path.join(dir, name);
      let st;
      try {
        st = statSync(p);
      } catch {
        continue;
      }
      if (st.isDirectory()) {
        out.push(`${p}|dir`);
        walk(p);
      } else {
        out.push(`${p}|${st.size}|${st.mtimeMs}`);
      }
    }
  };
  walk(root);
  return out;
}

test.describe("real runs (read-only)", () => {
  test.skip(!existsSync(path.join(REAL_RUNS, "E-008")) || !existsSync(path.join(REAL_RUNS, "E-005")), "no real runs on this machine");

  test("E-005 and E-008 are shown, and nothing under the runs tree changes", async ({ page }) => {
    test.setTimeout(120_000);
    // Only the experiments of interest are snapshotted (the rest of the tree has multi-GB files and live writers).
    const watched = ["E-005", "E-008"].map((e) => path.join(REAL_RUNS, e));
    // Entries that existed before must be byte-for-byte as they were (size, mtime), except the run a job is writing right now;
    // new entries (a queued job starting) are not our business.
    const live = "/E-008/e008-p2-mlpw-d2-maskhook-s2";
    const before = watched.flatMap(snapshot);
    const real = await startWeb(REAL_RUNS);
    try {
      await page.setViewportSize({ width: 1280, height: 900 });
      const errors: string[] = [];
      page.on("pageerror", (e) => errors.push(String(e)));
      await openTrainTab(page, real);
      const list = page.locator("#train-list");
      await expect(list.locator("details.tr-exp").filter({ hasText: "E-008" })).toHaveCount(1);
      await expect(list.locator("details.tr-exp").filter({ hasText: "E-005" })).toHaveCount(1);
      await list.locator("details.tr-exp").filter({ hasText: "E-005" }).locator("summary").click();
      await list.locator("details.tr-exp").filter({ hasText: "E-008" }).evaluate((d) => ((d as HTMLDetailsElement).open = true));
      await expect(list.getByRole("button", { name: "e008-p1-fly-drop-s1" })).toBeVisible();
      await expect(list.getByRole("button", { name: "e005-fly", exact: true })).toBeVisible();
      await expect(list.getByRole("button", { name: "e008-p2-mlpw-d2-maskhook-s2", exact: true })).toBeVisible();
      await page.screenshot({ path: path.join(SCREENSHOT_DIR, "5.8-real-list.png"), fullPage: true });

      await list.getByRole("button", { name: "e008-p1-fly-drop-s1" }).click();
      const detail = page.locator("#train-detail");
      await expect(detail.locator("h2").first()).toContainText("E-008 / e008-p1-fly-drop-s1");
      await expect(detail.locator(".tr-chart svg.tr-svg")).toHaveCount(5);
      await expect(detail.locator(".tr-chart svg.tr-svg").first().locator("line.tr-mark")).toHaveCount(5);
      await expect(detail.locator("table.tr-arena").first()).toContainText("clb-left");
      await expect(detail.locator("table", { hasText: "sha256 (16)" }).locator("tbody tr")).toHaveCount(6 + 6);
      await expect(detail.locator("details.tr-sub").first()).toContainText("Сводка арены");
      await page.screenshot({ path: path.join(SCREENSHOT_DIR, "5.8-real-run.png"), fullPage: true });
      await detail.locator("section.card").nth(1).screenshot({ path: path.join(SCREENSHOT_DIR, "5.8-real-curves.png") });
      await detail.locator("section.card").nth(2).screenshot({ path: path.join(SCREENSHOT_DIR, "5.8-real-rounds.png") });
      await detail.locator("section.card").nth(3).screenshot({ path: path.join(SCREENSHOT_DIR, "5.8-real-arena.png") });

      // The run that trains right now (when it still does), and a comparison of three real runs.
      for (const name of ["e008-p1-fly-drop-s1", "e008-p1-fly-base-s1", "e008-p1-fly-maskhook-s1"]) {
        await list.getByLabel(`Сравнить: ${name}`).check();
      }
      await page.locator("#train-compare-btn").click();
      await expect(page.locator("#train-compare h2").first()).toContainText("Сравнение запусков (3)");
      await expect(page.locator("#train-compare .tr-chart svg.tr-svg")).toHaveCount(4);
      await page.screenshot({ path: path.join(SCREENSHOT_DIR, "5.8-real-compare.png"), fullPage: true });
      // The reviewer's scenario on the real tree: an open run plus a comparison, the tab re-shown six times: all three stay.
      const statuses: number[] = [];
      page.on("response", (r) => r.url().includes("/api/train/") && statuses.push(r.status()));
      for (let round = 0; round < 6; round++) {
        statuses.length = 0;
        await page.locator("#tab-status").click();
        await page.locator("#tab-train").click();
        await expect.poll(() => statuses.length, { timeout: 20_000 }).toBeGreaterThanOrEqual(5);
        await page.waitForTimeout(500);
        expect(statuses, `round ${round}`).toEqual(Array(statuses.length).fill(200));
        await expect(page.locator("#train-compare h2").first()).toContainText("Сравнение запусков (3)");
      }
      expect(errors).toEqual([]);
    } finally {
      stopWeb(real);
    }
    const after = watched.flatMap(snapshot);
    const afterSet = new Set(after);
    const missingOrChanged = before.filter((l) => !l.includes(live) && !afterSet.has(l));
    expect(missingOrChanged).toEqual([]);
  });
});
