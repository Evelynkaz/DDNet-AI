// A side-by-side contact sheet of the before/after screenshots of task 5.14 (no ImageMagick on this machine: a page of <img> tags,
// rendered by Playwright). node support/contact-sheet.mjs <dir> <before-prefix> <after-prefix> <out.png>
import { chromium } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [dir, beforeP, afterP, out] = process.argv.slice(2);
const names = fs.readdirSync(dir).filter((f) => f.startsWith(afterP + "-") && f.endsWith(".png")).map((f) => f.slice(afterP.length + 1, -4));
const groups = ["desk", "phone"].map((v) => names.filter((n) => n.endsWith("-" + v)).sort());
const url = (p) => pathToFileURL(path.join(dir, p)).href;
const cell = (n, w) => {
  const b = `${beforeP}-${n}.png`;
  const a = `${afterP}-${n}.png`;
  if (!fs.existsSync(path.join(dir, b)) || !fs.existsSync(path.join(dir, a))) return "";
  return `<div class="row"><h3>${n}</h3><div class="pair"><figure><figcaption>до</figcaption><img src="${url(b)}" style="width:${w}px"></figure><figure><figcaption>после</figcaption><img src="${url(a)}" style="width:${w}px"></figure></div></div>`;
};
const html = `<!doctype html><meta charset="utf-8"><style>body{background:#101218;color:#ccd;font:14px sans-serif;margin:24px}h2{margin:32px 0 8px}h3{margin:16px 0 6px;font-weight:600}.pair{display:flex;gap:16px;align-items:flex-start}figure{margin:0}figcaption{opacity:.6;margin-bottom:4px}img{display:block;border:1px solid #333}</style>
<h2>1440×900</h2>${groups[0].map((n) => cell(n, 700)).join("")}<h2>390×844</h2><div style="display:flex;flex-wrap:wrap;gap:24px">${groups[1].map((n) => cell(n, 260)).join("")}</div>`;
const tmp = path.join(dir, ".contact.html");
fs.writeFileSync(tmp, html);
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1500, height: 900 } });
await p.goto(pathToFileURL(tmp).href);
await p.waitForLoadState("load");
await p.screenshot({ path: out, fullPage: true, timeout: 120000 });
await b.close();
fs.rmSync(tmp);
console.log("wrote", out);
