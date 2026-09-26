"use strict";

const api = window.ddnet;
const $ = (s) => document.querySelector(s);

const LANG = new URLSearchParams(location.search).get("lang") === "en" ? "en" : "ru";
const { t, tr } = I18N.makeT(I18N.EN, LANG);
const LOCALE = LANG === "en" ? "en-GB" : "ru-RU";
document.documentElement.lang = LANG;
I18N.translateDom(document.body, t, LANG);
document.documentElement.classList.remove("i18n-wait");

{
  let tip = null;
  let cur = null;
  let timer = 0;
  const hide = () => {
    clearTimeout(timer);
    cur = null;
    if (tip) tip.hidden = true;
  };

  const moveTitle = (target) => {
    const v = target.getAttribute("title") || "";
    target.removeAttribute("title");
    target.dataset.tip = v;
    if (!target.textContent.trim()) target.setAttribute("aria-label", v);
    return v;
  };
  const place = (target) => {
    const r = target.getBoundingClientRect();
    const w = tip.offsetWidth;
    const h = tip.offsetHeight;
    const x = Math.max(6, Math.min(innerWidth - w - 6, r.left + r.width / 2 - w / 2));
    const y = r.bottom + 8 + h <= innerHeight - 6 ? r.bottom + 8 : Math.max(6, r.top - h - 8);
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
  };
  document.addEventListener("mouseover", (e) => {
    const target = e.target instanceof Element ? e.target.closest("[title],[data-tip]") : null;
    if (target && target.hasAttribute("title")) moveTitle(target);
    if (target === cur) return;
    hide();
    if (!target || !target.dataset.tip) return;
    cur = target;
    timer = setTimeout(() => {
      if (cur !== target || !target.isConnected) return;
      if (!tip) {
        tip = el("div", "tip");
        document.body.appendChild(tip);
      }
      tip.textContent = target.dataset.tip;
      tip.hidden = false;
      place(target);

      if (target.closest("#titlebar")) setTimeout(() => cur === target && hide(), 3000);
    }, 350);
  });
  document.addEventListener("mouseout", (e) => {
    if (!e.relatedTarget) hide();
  });
  for (const ev of ["mousedown", "keydown", "wheel"]) document.addEventListener(ev, hide, true);
  addEventListener("blur", hide);

  new MutationObserver((ms) => {
    for (const m of ms) {
      const target = m.target;
      if (!target.hasAttribute("title") || !("tip" in target.dataset)) continue;
      const v = moveTitle(target);
      if (target !== cur || !tip || tip.hidden) continue;
      if (v) {
        tip.textContent = v;
        place(target);
      } else hide();
    }
  }).observe(document.documentElement, { subtree: true, attributes: true, attributeFilter: ["title"] });
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function icon(name) {
  return el("i", `ic ic-${name}`);
}

let state = null;
let frameKey = "";
let frameMini = false;
let everReady = false;
let setupMode = "first";
let logItems = [];
let logFilter = "";

$("#w-min").addEventListener("click", () => api.window.minimize());
$("#w-max").addEventListener("click", () => api.window.toggleMaximize());
$("#w-close").addEventListener("click", () => api.window.close());
$("#titlebar").addEventListener("dblclick", (e) => {
  if (e.target.closest("button")) return;
  api.window.toggleMaximize();
});
$("#b-pause").addEventListener("click", () => api.bot.togglePause());
$("#b-mini").addEventListener("click", () => api.window.setMini(!(state && state.mini)));
$("#b-pin").addEventListener("click", () => api.window.setOnTop(!(state && state.onTop)));
$("#b-servers").addEventListener("click", () => toggleDrawer("servers"));
$("#b-settings").addEventListener("click", () => toggleDrawer("settings"));
$("#b-log").addEventListener("click", () => toggleLog());

function pillFor(s) {
  if (s.screen === "noroot") return ["off", t("папка бота не найдена")];
  if (s.screen === "setup") return ["warn", t("ждёт настройки")];
  if (s.screen === "start") return ["warn", t("ждёт старта")];
  if (s.botState !== "running") return ["warn pulse", s.botState === "waiting" ? t("перезапуск") : t("запуск бота")];
  if (s.paused) return ["warn", t("пауза · {name}", { name: s.name || t("бот") })];
  if (s.phase === "online") {
    return ["on", s.target ? t("{name} на {server} · против {target}", { name: s.name, server: s.server, target: s.target }) : t("{name} на {server}", { name: s.name, server: s.server })];
  }
  if (s.phase === "connecting") return ["warn pulse", t("подключается к {server}", { server: s.server })];
  return ["off", s.offlineReason ? t("не на сервере: {reason}", { reason: s.offlineReason }) : t("не на сервере")];
}

function render(s) {
  state = s;
  document.body.classList.toggle("mini", s.mini);
  const [dotCls, text] = pillFor(s);
  $("#pill-dot").className = `dot ${dotCls}`;
  $("#pill-text").textContent = text;
  $("#pill").title = text;

  const running = s.botState === "running";
  const pause = $("#b-pause");
  pause.disabled = !running;
  pause.classList.toggle("paused", s.paused);
  pause.querySelector(".ic").className = `ic ic-${s.paused ? "play" : "pause"}`;
  $("#b-pause-lbl").textContent = s.paused ? t("Играть") : t("Пауза");
  pause.title = `${s.paused ? t("Продолжить игру") : t("Пауза: бот встанет на месте")} (${prettyKey(s.hotkey)})`;
  $("#b-pin").classList.toggle("on", s.onTop);
  $("#b-pin").querySelector(".ic").className = `ic ic-${s.onTop ? "pin-off" : "pin"}`;
  $("#b-mini").title = s.mini ? t("Обычное окно") : t("Мини-режим поверх игры");
  $("#w-max-ic").className = `ic ic-${s.maximized ? "copy" : "square"}`;
  $("#w-max").title = s.maximized ? t("Восстановить") : t("Развернуть");
  for (const id of ["#b-servers", "#b-log", "#b-settings"]) $(id).disabled = s.screen === "noroot";

  const editing = setupMode === "edit" && !$("#screen-setup").hidden;
  $("#screen-noroot").hidden = s.screen !== "noroot";

  if (s.screen === "start" && !startOpen && !openingStart) void openStart();
  if (s.screen !== "start" && startOpen) closeStart();
  if (startOpen) $("#screen-start").hidden = editing;
  if (s.screen === "setup" && $("#screen-setup").hidden && !openingSetup) void openSetup("first");
  if (s.screen !== "setup" && setupMode === "first" && !$("#screen-setup").hidden) $("#screen-setup").hidden = true;
  const showLoading = s.screen === "app" && !everReady && !(running && s.readyCount > 0);
  $("#screen-loading").hidden = !showLoading || editing;
  $("#loading-err").hidden = !s.lastError;
  $("#loading-err").textContent = s.lastError ? tr(s.lastError) : "";
  $("#loading-actions").hidden = !s.lastError;
  $("#loading-title").textContent = s.lastError ? t("Бот не запускается") : t("Запускаю бота");

  if (running && s.port > 0) {
    const frame = $("#bot");
    const base = `http://127.0.0.1:${s.port}/`;

    const page = `${base}?lang=${LANG}&run=${s.readyCount}`;
    const key = `${s.port}|${s.readyCount}`;
    if (key !== frameKey) {
      frameKey = key;
      loadGameFont(base);
      frameMini = s.mini;
      frame.classList.add("hidden");
      frame.src = page + (s.mini ? "#mini" : "#full");
    } else if (frameMini !== s.mini) {
      frameMini = s.mini;
      frame.src = page + (s.mini ? "#mini" : "#full");
    }
    everReady = true;
  }
  const banner = everReady && !running && s.screen === "app";
  $("#banner").hidden = !banner;
  $("#banner-text").textContent = s.botState === "waiting" ? t("Бот перезапускается...") : t("Бот запускается...");

  $("#st-root").textContent = s.root || "";
  $("#st-root").title = s.root || "";
  $("#st-shortcut").hidden = s.platform !== "win32";
  renderAbout(s);
}

$("#bot").addEventListener("load", () => {
  if ($("#bot").src !== "about:blank") $("#bot").classList.remove("hidden");
});

let startOpen = false;
let openingStart = false;
let startTimer = null;
let startLeft = 0;
let startData = null;

const brainLabel = (b) => (b === "bold" ? t("Экспериментальный") : b === "scripted" ? t("Скриптовый") : t("Планировщик"));
const serverName = (h) => (h.server === "auto" || !h.server ? t("сервер сам") : h.label ? `${h.label} (${h.server})` : h.server);
const whoName = (h) => h.name + (h.clan ? ` [${h.clan}]` : "");
function playedFor(ms) {
  const m = Math.floor((ms || 0) / 60000);
  if (m < 1) return t("меньше минуты");
  if (m < 60) return t("{m} мин", { m });
  return t("{h} ч {m} мин", { h: Math.floor(m / 60), m: m % 60 });
}
const whenPlayed = (ms) => (ms > 0 ? new Date(ms).toLocaleString(LOCALE, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) : "");

async function openStart() {
  openingStart = true;
  try {
    startData = await api.start.get();
  } catch {
    startData = null;
  } finally {
    openingStart = false;
  }
  if (startData === null || !startData.gate) return;
  startOpen = true;
  renderStart();
  $("#screen-start").hidden = false;
  $("#screen-loading").hidden = true;
  startLeft = startData.countdown;
  clearInterval(startTimer);

  if (location.hash === "#settings" || document.querySelector(".drawer:not([hidden])") || (setupMode === "edit" && !$("#screen-setup").hidden)) {
    stopCountdown(t("Автозапуск отменён"));
  } else {
    startTimer = setInterval(() => tickStart(false), 1000);
    tickStart(true);
  }
  setTimeout(() => $("#start-play").focus(), 50);
}

function closeStart() {
  startOpen = false;
  stopCountdown("");
  $("#screen-start").hidden = true;
}

function tickStart(first) {
  if (!first) startLeft--;
  if (startLeft <= 0) {
    stopCountdown("");
    void play();
    return;
  }
  const box = $("#start-auto");
  box.textContent = t("Запуск через {n} с", { n: startLeft });
  const stay = el("button", "linkbtn", t("подождать"));
  stay.type = "button";
  stay.addEventListener("click", () => stopCountdown(t("Автозапуск отменён")));
  box.append(stay);
}

function stopCountdown(text) {
  if (startTimer !== null) clearInterval(startTimer);
  startTimer = null;
  $("#start-auto").textContent = text;
}

function renderStart() {
  const c = startData.current;
  const card = $("#start-current");
  card.replaceChildren();
  for (const [k, v] of [
    [t("Сервер"), serverName({ server: c.server, label: "" })],
    [t("Ник"), whoName(c)],
    [t("Скин"), c.skin],
    [t("Чем играть"), brainLabel(c.brain)],
  ]) {
    card.append(el("span", "k", k), el("span", "v", v));
  }
  const list = Array.isArray(startData.history) ? startData.history : [];
  $("#start-hist-wrap").hidden = list.length === 0;
  const box = $("#start-hist");
  box.replaceChildren();
  list.forEach((h, i) => {
    const now = h.server === (c.server || "auto") && h.name === c.name && h.clan === c.clan && h.skin === c.skin && h.brain === c.brain;
    const row = el("div", `hrow${now ? " now" : ""}`);
    const main = el("div", "h-main");
    main.append(el("b", "", whoName(h)), el("span", "", serverName(h)));
    const sub = el("div", "h-sub", [h.skin, brainLabel(h.brain), whenPlayed(h.last), t("сыграно {t}", { t: playedFor(h.playedMs) })].filter(Boolean).join(" · "));
    const go = el("button", "btn");
    go.type = "button";
    go.append(icon("play"), el("span", "", t("Играть так")));
    go.addEventListener("click", () => void play(i));
    const forget = el("button", "icon-btn");
    forget.type = "button";
    forget.title = t("Убрать из истории");
    forget.append(icon("x"));
    forget.addEventListener("click", async () => {
      const r = await api.start.forget(i);
      if (r && r.ok) {
        startData.history = r.history;
        renderStart();
      }
    });
    row.append(main, sub, go, forget);
    box.append(row);
  });
}

async function play(index) {
  stopCountdown("");
  $("#start-play").disabled = true;
  try {
    const res = await api.start.play(index);
    if (!res || !res.ok) {
      toast({ text: tr((res && res.error) || t("не вышло")), kind: "error" });
      return;
    }
    closeStart();
  } finally {
    $("#start-play").disabled = false;
  }
}

$("#start-play").addEventListener("click", () => void play());
$("#start-edit").addEventListener("click", () => {
  stopCountdown(t("Автозапуск отменён"));
  void openSetup("edit");
});

for (const ev of ["pointerdown", "keydown", "wheel"]) {
  $("#screen-start").addEventListener(ev, (e) => {
    if (e.target && e.target.closest && e.target.closest("#start-play")) return;
    if (startTimer !== null) stopCountdown(t("Автозапуск отменён"));
  });
}

$("#loading-log").addEventListener("click", () => toggleLog(true));
$("#loading-restart").addEventListener("click", () => api.bot.restart());
$("#noroot-pick").addEventListener("click", () => api.action("chooseRoot"));

function renderTail() {
  const tail = logItems.slice(-7).map((l) => l.text).join("\n");
  $("#loading-tail").textContent = tail;
}

let openingSetup = false;
async function openSetup(mode) {

  if (startTimer !== null) stopCountdown(t("Автозапуск отменён"));
  setupMode = mode;
  closeDrawers();
  openingSetup = true;
  let data;
  try {
    data = await api.setup.get();
  } finally {
    openingSetup = false;
  }
  const s = data.settings;
  $("#f-server").value = s.server === "auto" ? "" : s.server;
  $("#f-name").value = s.name;
  $("#f-clan").value = s.clan;
  $("#f-skin").value = s.skin;
  $("#f-password").value = "";
  passClear = false;
  passSaved = s.hasPassword;
  passServer = s.server;
  renderPassHint();
  for (const r of document.querySelectorAll("input[name=brain]")) r.checked = r.value === s.brain;
  $("#f-dummy").checked = s.dummy === true;
  $("#f-dummyName").value = s.dummyName || "";
  $("#f-lowCpu").checked = s.lowCpu === true;
  $("#f-strong").checked = s.strong === true && s.lowCpu !== true;

  const test = s.brain !== "planner";
  $(".brains").classList.toggle("all", test);
  $("#brains-more").hidden = test;
  const first = mode === "first";
  $("#setup-title").textContent = first ? t("Первый запуск") : t("Бот: сервер, ник, скин");
  $("#setup-sub").textContent = first
    ? t("Пара полей, и бот пойдёт играть. Потом всё это меняется в настройках.")
    : t("После сохранения бот перезапустится с новыми настройками.");
  $("#setup-save-lbl").textContent = first ? t("Сохранить и запустить") : t("Сохранить и перезапустить");
  $("#setup-cancel").hidden = first;
  clearErrors();
  $("#screen-setup").hidden = false;
  $("#screen-loading").hidden = true;
  setTimeout(() => (first ? $("#f-server") : $("#f-name")).focus(), 50);
}

let passClear = false;
let passSaved = false;
let passServer = "";
function renderPassHint() {
  const other = passSaved && $("#f-server").value.trim() !== passServer;
  const keep = passSaved && !passClear && !other;
  $("#f-password").placeholder = keep ? t("сохранён, пусто = не менять") : "";
  $("#f-pass-hint").textContent = passClear ? t("будет стёрт") : other ? t("другой сервер: старый сотрётся") : passSaved ? t("сохранён") : t("можно пусто");
  $("#f-pass-clear").hidden = !keep;
}
$("#f-pass-clear").addEventListener("click", () => {
  passClear = true;
  renderPassHint();
});
$("#f-server").addEventListener("input", () => renderPassHint());

$("#f-lowCpu").addEventListener("change", () => {
  if ($("#f-lowCpu").checked) $("#f-strong").checked = false;
});
$("#f-strong").addEventListener("change", () => {
  if ($("#f-strong").checked) $("#f-lowCpu").checked = false;
});

function clearErrors() {
  for (const e of document.querySelectorAll(".ferr[data-for]")) e.textContent = "";
  for (const i of document.querySelectorAll(".setup input")) i.classList.remove("bad");
}

function readForm() {
  const form = {
    server: $("#f-server").value,
    name: $("#f-name").value,
    clan: $("#f-clan").value,
    skin: $("#f-skin").value,
    brain: (document.querySelector("input[name=brain]:checked") || { value: "planner" }).value,
    dummy: $("#f-dummy").checked ? "on" : "off",
    dummyName: $("#f-dummyName").value,
    lowCpu: $("#f-lowCpu").checked ? "on" : "off",
    strong: $("#f-strong").checked ? "on" : "off",
  };
  const pass = $("#f-password").value;

  if (pass !== "" || setupMode === "first" || passClear) form.password = pass;
  return form;
}

$("#setup-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  clearErrors();
  $("#setup-save").disabled = true;
  try {
    const res = await api.setup.save(readForm());
    if (!res.ok) {
      for (const [k, msg] of Object.entries(res.errors || {})) {
        const slot = document.querySelector(`.ferr[data-for="${k}"]`);
        if (slot) slot.textContent = tr(msg);
        const input = $(`#f-${k}`);
        if (input) input.classList.add("bad");
      }
      return;
    }
    $("#screen-setup").hidden = true;
    if (setupMode === "edit") toast({ text: t("Сохранено, бот перезапускается"), kind: "ok" });
    setupMode = "done";
  } finally {
    $("#setup-save").disabled = false;
  }
});
$("#setup-cancel").addEventListener("click", () => {
  $("#screen-setup").hidden = true;
  setupMode = "done";
  if (startOpen) $("#screen-start").hidden = false;
});
$("#f-pick").addEventListener("click", () => openDrawer("servers"));
$("#brains-more").addEventListener("click", () => {
  $(".brains").classList.add("all");
  $("#brains-more").hidden = true;
});

function openDrawer(name) {

  if (startTimer !== null) stopCountdown(t("Автозапуск отменён"));
  closeDrawers();
  $(`#drawer-${name}`).hidden = false;
  if (name === "servers") {
    $(`#b-servers`).classList.add("on");
    void loadServers(false);
    setTimeout(() => $("#sv-q").focus(), 30);
  }
  if (name === "settings") {
    $(`#b-settings`).classList.add("on");
    void loadPrefs();
  }
}
function closeDrawers() {
  stopRecording();
  for (const d of document.querySelectorAll(".drawer")) d.hidden = true;
  $("#b-servers").classList.remove("on");
  $("#b-settings").classList.remove("on");
}
function toggleDrawer(name) {
  if ($(`#drawer-${name}`).hidden) openDrawer(name);
  else closeDrawers();
}
for (const b of document.querySelectorAll("[data-close]")) b.addEventListener("click", closeDrawers);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    if (recordingHotkey) return;
    if (!document.querySelector(".drawer:not([hidden])")) return;
    closeDrawers();
  }
});

const svFilters = { hideEmpty: true, hideFull: false, hideLocked: true, onlyFavorites: false };
let svRows = [];
let svFavs = new Set();
let svRecent = [];
let svSelected = null;
let svLoading = false;
let svPassFor = "";

for (const c of document.querySelectorAll(".chip[data-f]")) {
  c.addEventListener("click", () => {
    svFilters[c.dataset.f] = !svFilters[c.dataset.f];
    c.classList.toggle("on", svFilters[c.dataset.f]);
    renderServers();
  });
}
$("#sv-q").addEventListener("input", () => renderServers());
$("#sv-refresh").addEventListener("click", () => loadServers(true));

async function loadServers(force) {
  if (svLoading) return;
  svLoading = true;
  $("#sv-refresh").querySelector(".ic").classList.add("spin");
  if (svRows.length === 0) showEmpty(t("Загружаю список с мастер-сервера DDNet..."));
  try {
    const [res, prefs] = await Promise.all([api.servers.list(force), api.prefs.get()]);
    svFavs = new Set(prefs.favorites);
    svRecent = prefs.recent;
    if (!res.ok) {
      svRows = [];
      showEmpty(tr(res.error));
      $("#sv-count").textContent = "";
    } else {
      svRows = res.rows;
      renderServers();
    }
    renderRecent();
  } catch (err) {
    showEmpty(t("Не загрузилось: {err}", { err: err.message }));
  } finally {
    svLoading = false;
    $("#sv-refresh").querySelector(".ic").classList.remove("spin");
  }
}

function showEmpty(text) {
  $("#sv-rows").textContent = "";
  $("#sv-empty").hidden = false;
  $("#sv-empty").textContent = text;
}

function matches(r, q, opts) {
  if (opts.onlyFavorites && !svFavs.has(r.address)) return false;
  if (opts.hideEmpty && r.clients === 0) return false;
  if (opts.hideFull && r.maxClients > 0 && r.clients >= r.maxClients) return false;
  if (opts.hideLocked && r.passworded) return false;
  if (q === "") return true;
  return (
    r.name.toLowerCase().includes(q) ||
    r.map.toLowerCase().includes(q) ||
    r.gameType.toLowerCase().includes(q) ||
    r.address.includes(q) ||
    r.names.some((n) => n.toLowerCase().includes(q))
  );
}

function renderServers() {
  const q = $("#sv-q").value.trim().toLowerCase();

  const rows = svRows.filter((r) => matches(r, q, svFilters)).sort((a, b) => Number(svFavs.has(b.address)) - Number(svFavs.has(a.address)));
  const players = svRows.reduce((n, r) => n + r.players, 0);
  $("#sv-count").textContent = svRows.length ? t("{n} из {total} · {players} игроков онлайн", { n: rows.length, total: svRows.length, players }) : "";
  const body = $("#sv-rows");
  body.textContent = "";
  if (rows.length === 0) {
    showEmpty(svRows.length ? t("Ничего не нашлось. Попробуй снять фильтры.") : t("Список пуст."));
    return;
  }
  $("#sv-empty").hidden = true;
  const current = state && state.server;
  const frag = document.createDocumentFragment();
  for (const r of rows.slice(0, 600)) {
    const tr = el("tr");
    if (svSelected && svSelected.address === r.address) tr.classList.add("sel");
    if (current === r.address) tr.classList.add("cur");
    const fav = el("td", "c-fav");
    const star = el("button", `star${svFavs.has(r.address) ? " on" : ""}`);
    star.type = "button";
    star.title = svFavs.has(r.address) ? t("Убрать из избранного") : t("В избранное");
    star.append(icon("star"));
    star.addEventListener("click", async (e) => {
      e.stopPropagation();
      svFavs = new Set(await api.servers.favorite(r.address, !svFavs.has(r.address)));
      renderServers();
    });
    fav.append(star);
    const name = el("td");
    const nm = el("div", "sv-name");
    if (r.passworded) nm.append(icon("lock"));
    nm.append(el("span", "", r.name));
    name.append(nm);
    name.title = `${r.name}\n${r.gameType} · ${r.address}`;
    const map = el("td", "", r.map);
    map.title = r.map;
    const num = el("td", "c-num");
    const bar = el("span", "bar-full");
    const fill = el("i");
    fill.style.width = `${r.maxClients ? Math.min(100, (r.clients / r.maxClients) * 100) : 0}%`;
    bar.append(fill);
    const cell = el("span", "num-cell");
    cell.append(bar, el("span", "num-txt", `${r.clients}/${r.maxClients}`));
    num.append(cell);
    const reg = el("td", "c-reg", regionOf(r.location));
    tr.append(fav, name, map, num, reg);
    tr.addEventListener("click", () => selectServer(r));
    tr.addEventListener("dblclick", () => playServer(r));
    frag.append(tr);
  }
  body.append(frag);
}

function regionOf(loc) {
  const m = /^([a-z]{2,3})(?::([a-z]{2}))?/i.exec(loc || "");
  if (!m) return "";
  return m[2] ? `${m[1].toUpperCase()} ${m[2].toUpperCase()}` : m[1].toUpperCase();
}

function selectServer(r) {
  svSelected = r;
  $("#sv-detail").hidden = false;
  $("#sv-d-name").textContent = r.name;
  $("#sv-d-meta").textContent = `${r.gameType || "?"} · ${r.map || "?"} · ${r.address} · ${r.clients}/${r.maxClients}${r.passworded ? ` · ${t("с паролем")}` : ""}`;
  const passBox = $("#sv-passbox");
  if (svPassFor !== r.address) {
    $("#sv-pass").value = "";
    svPassFor = r.address;
  }
  passBox.hidden = !r.passworded;
  $("#sv-pass").classList.remove("bad");
  const q = $("#sv-q").value.trim().toLowerCase();
  const box = $("#sv-d-players");
  box.textContent = "";

  const hit = (n) => q !== "" && n.toLowerCase().includes(q);
  const names = [...r.names.filter(hit), ...r.names.filter((n) => !hit(n))];
  const SHOW = 14;
  for (const n of names.slice(0, SHOW)) box.append(el("span", hit(n) ? "hit" : "", n));
  if (names.length > SHOW) box.append(el("span", "more", t("и ещё {n}", { n: names.length - SHOW })));
  if (names.length === 0) box.append(el("span", "", t("никого")));
  for (const tr of document.querySelectorAll("#sv-rows tr")) tr.classList.remove("sel");
  renderServers();
}

async function playServer(r) {

  if (!$("#screen-setup").hidden) {
    $("#f-server").value = r.address;
    closeDrawers();
    return;
  }
  let password;
  if (r.passworded) {
    password = $("#sv-pass").value;
    if (password === "") {
      $("#sv-pass").classList.add("bad");
      $("#sv-pass").focus();
      toast({ text: t("У этого сервера пароль: впиши его"), kind: "warn" });
      return;
    }
  }
  const res = await api.servers.play(r.address, r.name, password);
  if (res && res.setupOnly) {
    $("#f-server").value = r.address;
  }
  closeDrawers();
}
$("#sv-play").addEventListener("click", () => svSelected && playServer(svSelected));
$("#sv-copy").addEventListener("click", async () => {
  if (!svSelected) return;
  await api.copy(svSelected.address);
  toast({ text: t("Скопировано: {addr}", { addr: svSelected.address }), kind: "ok" });
});
$("#sv-manual").addEventListener("submit", (e) => {
  e.preventDefault();
  const v = $("#sv-addr").value.trim();
  if (!/^\d{1,3}(\.\d{1,3}){3}:\d{1,5}$/.test(v)) {
    $("#sv-addr").classList.add("bad");
    toast({ text: t("Нужен адрес вида 1.2.3.4:8303"), kind: "warn" });
    return;
  }
  $("#sv-addr").classList.remove("bad");
  void playServer({ address: v, name: "" });
});

function renderRecent() {
  const box = $("#sv-recent");
  box.textContent = "";
  if (!svRecent.length) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  box.append(el("span", "muted small", t("Недавние:")));
  for (const r of svRecent.slice(0, 6)) {
    const c = el("button", "chip");
    c.type = "button";
    c.title = r.address;
    c.append(icon("clock"), el("span", "", r.name || r.address));
    c.addEventListener("click", () => {
      const row = svRows.find((x) => x.address === r.address) || { address: r.address, name: r.name, gameType: "", map: "", clients: 0, maxClients: 0, names: [], passworded: false, location: "" };
      selectServer(row);
    });
    box.append(c);
  }
}

let prefsCache = null;
let recordingHotkey = false;

async function loadPrefs() {
  prefsCache = await api.prefs.get();
  $("#st-notify").checked = prefsCache.notifications;
  $("#st-start").checked = prefsCache.startScreen !== false;
  $("#st-tray").checked = prefsCache.closeToTray;
  $("#st-gpu").checked = prefsCache.gpu !== false;
  $("#st-tray").disabled = !prefsCache.trayAvailable;
  $("#st-login").checked = prefsCache.openAtLogin;
  $("#st-login-row").hidden = !prefsCache.loginSupported;
  $("#st-hotkey").textContent = prettyKey(prefsCache.hotkey);
  $("#st-hotkey-err").textContent = state && state.hotkeyError ? tr(state.hotkeyError) : "";
  $("#st-lang").value = prefsCache.lang;
  const data = await api.setup.get();
  const s = data.settings;
  const extra = [s.dummy ? t("второй бот {name}", { name: s.dummyName || t("вкл") }) : "", s.lowCpu ? t("режим для слабого ПК") : "", s.strong && !s.lowCpu ? t("сильный режим") : ""].filter((x) => x !== "");
  $("#st-edit-sub").textContent = [`${s.name}${s.clan ? ` [${s.clan}]` : ""}`, s.server === "auto" ? t("сервер сам") : s.server, brainName(s.brain), ...extra].join(" · ");
}

function brainName(b) {
  return b === "bold" ? t("экспериментальный") : b === "scripted" ? t("скриптовый") : t("планировщик");
}

async function setPref(patch) {
  const res = await api.prefs.set(patch);
  if (!res.ok) toast({ text: tr(res.error), kind: "error" });
  return res;
}
$("#st-notify").addEventListener("change", (e) => setPref({ notifications: e.target.checked }));
$("#st-start").addEventListener("change", (e) => setPref({ startScreen: e.target.checked }));
$("#st-tray").addEventListener("change", (e) => setPref({ closeToTray: e.target.checked }));
$("#st-gpu").addEventListener("change", async (e) => {
  const res = await setPref({ gpu: e.target.checked });
  if (res.ok) toast({ text: t("Применится после перезапуска окна"), kind: "info" });
});
$("#st-login").addEventListener("change", (e) => setPref({ openAtLogin: e.target.checked }));

$("#st-lang").addEventListener("change", (e) => setPref({ lang: e.target.value }));
$("#st-edit").addEventListener("click", () => openSetup("edit"));
$("#st-restart").addEventListener("click", () => {
  closeDrawers();
  api.bot.restart();
});
$("#st-browser").addEventListener("click", () => api.action("openInBrowser"));
$("#st-archive").addEventListener("click", () => api.action("collectArchive"));
$("#st-clips").addEventListener("click", () => api.action("openClips"));
$("#st-project").addEventListener("click", () => api.action("openProject"));
$("#st-shortcut").addEventListener("click", () => api.action("desktopShortcut"));

function prettyKey(accel) {
  return String(accel || "").replace("CommandOrControl", "Ctrl").replace("CmdOrCtrl", "Ctrl").split("+").join(" + ");
}

function stopRecording() {
  if (!recordingHotkey) return;
  recordingHotkey = false;
  $("#st-hotkey").classList.remove("rec");
  $("#st-hotkey").textContent = prettyKey(prefsCache ? prefsCache.hotkey : state && state.hotkey);
}
$("#st-hotkey").addEventListener("click", () => {
  recordingHotkey = true;
  $("#st-hotkey").classList.add("rec");
  $("#st-hotkey").textContent = t("нажми сочетание...");
  $("#st-hotkey").focus();
});
$("#st-hotkey").addEventListener("blur", () => stopRecording());
document.addEventListener("keydown", async (e) => {
  if (!recordingHotkey) return;
  if (document.activeElement !== $("#st-hotkey")) {
    stopRecording();
    return;
  }
  e.preventDefault();
  if (e.key === "Escape") {
    stopRecording();
    return;
  }
  const key = keyName(e);
  if (key === null) return;
  const mods = [];
  if (e.ctrlKey) mods.push("CommandOrControl");
  if (e.altKey) mods.push("Alt");
  if (e.shiftKey) mods.push("Shift");
  if (e.metaKey) mods.push("Super");
  const accel = [...mods, key].join("+");
  stopRecording();
  const res = await setPref({ hotkey: accel });
  $("#st-hotkey-err").textContent = res.ok ? "" : tr(res.error);
  await loadPrefs();
  if (res.ok) toast({ text: t("Пауза теперь на {key}", { key: prettyKey(accel) }), kind: "ok" });
});

function keyName(e) {
  const c = e.code;
  if (/^Key[A-Z]$/.test(c)) return c.slice(3);
  if (/^Digit\d$/.test(c)) return c.slice(5);
  if (/^F([1-9]|1\d|2[0-4])$/.test(c)) return c;
  if (/^Numpad\d$/.test(c)) return `num${c.slice(6)}`;
  const map = { Space: "Space", Insert: "Insert", Delete: "Delete", Home: "Home", End: "End", PageUp: "PageUp", PageDown: "PageDown", Pause: "Pause",
    ArrowUp: "Up", ArrowDown: "Down", ArrowLeft: "Left", ArrowRight: "Right", Backquote: "`", Minus: "-", Equal: "=", BracketLeft: "[", BracketRight: "]",
    Semicolon: ";", Quote: "'", Comma: ",", Period: ".", Slash: "/", Backslash: "\\" };
  return map[c] || null;
}

let fontFrom = "";
function loadGameFont(base) {
  if (fontFrom === base || typeof FontFace !== "function") return;
  fontFrom = base;
  try {
    const face = new FontFace("DejaVu Sans DDNet", `url(${base}assets/fonts/DejaVuSans.ttf)`);
    face.load().then((f) => document.fonts.add(f)).catch(() => {});
  } catch {

  }
}

function renderAbout(s) {
  const box = $("#st-about");
  box.textContent = "";
  const add = (k, v, wrap = false) => {
    const dd = el("dd", wrap ? "wrap" : "", v || "-");
    dd.title = v || "";
    box.append(el("dt", "", k), dd);
  };
  add(t("Приложение"), s.appVersion);
  add(t("Версия бота"), s.botVersion && /^[0-9a-f]{7,}$/.test(s.botVersion) ? s.botVersion.slice(0, 7) : s.botVersion || t("неизвестна"));
  add(t("Чем запущен"), s.runtime ? tr(s.runtime.label) : "");
  add(t("Страница бота"), s.port ? `127.0.0.1:${s.port}` : "");
  add(t("Папка"), s.root || "");
  add(t("Графика"), t("DDNet / Teeworlds (data: CC-BY-SA 3.0; скины, шрифты и ассеты под своими лицензиями). В бота не входит: окно берёт её из твоей установки DDNet, а недостающие скины качает с skins.ddnet.org в runs/skincache (выключается в настройках окна). Код отрисовки частично по исходникам DDNet (zlib)."), true);
}

function toggleLog(force) {
  const dock = $("#logdock");
  const open = typeof force === "boolean" ? force : dock.hidden;
  dock.hidden = !open;
  $("#b-log").classList.toggle("on", open);
  void api.prefs.set({ logOpen: open });
  if (open) renderLog();
}
$("#ld-close").addEventListener("click", () => toggleLog(false));
$("#ld-q").addEventListener("input", (e) => {
  logFilter = e.target.value.toLowerCase();
  renderLog();
});
$("#ld-copy").addEventListener("click", async () => {
  await api.copy(logItems.map((l) => `[${l.s}] ${l.text}`).join("\n").slice(-9000));
  toast({ text: t("Лог скопирован"), kind: "ok" });
});

function logRow(l) {
  const d = el("div", l.s === "err" ? "err" : l.s === "app" ? "app" : "");
  const at = new Date(l.t);
  d.append(el("time", "", at.toLocaleTimeString(LOCALE)), document.createTextNode(l.text));
  return d;
}

function renderLog() {
  const body = $("#ld-body");
  body.textContent = "";
  const frag = document.createDocumentFragment();
  for (const l of logItems) if (!logFilter || l.text.toLowerCase().includes(logFilter)) frag.append(logRow(l));
  body.append(frag);
  body.scrollTop = body.scrollHeight;
}

function appendLog(l) {
  logItems.push(l);
  if (logItems.length > 4000) logItems.splice(0, logItems.length - 4000);
  if (!$("#logdock").hidden && (!logFilter || l.text.toLowerCase().includes(logFilter))) {
    const body = $("#ld-body");
    const stick = body.scrollTop + body.clientHeight >= body.scrollHeight - 30;
    body.append(logRow(l));
    while (body.childElementCount > 4000) body.firstChild.remove();
    if (stick) body.scrollTop = body.scrollHeight;
  }
  if (!$("#screen-loading").hidden) renderTail();
}

(() => {
  const dock = $("#logdock");
  const frame = $("#bot");
  let startY = 0;
  let startH = 0;
  let dragging = null;
  const stop = () => {
    if (dragging === null) return;
    try {
      dock.releasePointerCapture(dragging);
    } catch {

    }
    dragging = null;
    frame.style.pointerEvents = "";
    document.body.classList.remove("resizing");
    void api.prefs.set({ logHeight: dock.offsetHeight });
  };
  dock.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || e.offsetY > 4 || e.target !== dock) return;
    e.preventDefault();
    startY = e.clientY;
    startH = dock.offsetHeight;
    dragging = e.pointerId;
    dock.setPointerCapture(e.pointerId);
    frame.style.pointerEvents = "none";
    document.body.classList.add("resizing");
  });
  dock.addEventListener("pointermove", (e) => {
    if (dragging === null || e.pointerId !== dragging) return;
    if (e.buttons === 0) return stop();
    const h = Math.max(120, Math.min(window.innerHeight - 160, startH + (startY - e.clientY)));
    dock.style.height = `${h}px`;
  });
  dock.addEventListener("pointerup", stop);
  dock.addEventListener("pointercancel", stop);
  dock.addEventListener("lostpointercapture", stop);
  window.addEventListener("blur", stop);
})();

let progressToast = null;
function toast({ text, kind = "info" }) {
  const box = el("div", `toast ${kind}`);
  box.append(icon(kind === "ok" ? "check" : kind === "error" || kind === "warn" ? "circle-alert" : "bell"), el("span", "", text));
  $("#toasts").append(box);
  setTimeout(() => {
    box.classList.add("out");
    setTimeout(() => box.remove(), 300);
  }, kind === "error" ? 7000 : 3800);
}

function progress({ done, total }) {
  if (progressToast === null) {
    progressToast = el("div", "toast info");
    const bar = el("div", "progress");
    bar.append(el("i"));
    progressToast.append(icon("archive"), el("span", "", t("Архив")), bar);
    $("#toasts").append(progressToast);
  }
  progressToast.querySelector(".progress i").style.width = `${Math.round((done / Math.max(1, total)) * 100)}%`;
  progressToast.querySelector("span").textContent = t("Архив: {done} из {total}", { done, total });
  if (done >= total) {
    const t = progressToast;
    progressToast = null;
    setTimeout(() => t.remove(), 600);
  }
}

api.on("state", render);
api.on("log", appendLog);
api.on("toast", toast);
api.on("progress", progress);

(async () => {
  render(await api.state());
  logItems = await api.log.get();
  const prefs = await api.prefs.get();
  if (prefs.logHeight) $("#logdock").style.height = `${Math.max(120, Math.min(window.innerHeight - 160, prefs.logHeight))}px`;
  if (prefs.logOpen) toggleLog(true);
  renderTail();

  if (location.hash === "#settings") openDrawer("settings");
})();

window.__debug = {
  open: (n) => openDrawer(n),
  close: () => {
    closeDrawers();
    if (setupMode === "edit") {
      $("#screen-setup").hidden = true;
      setupMode = "done";
    }
  },
  toggleLog: (on) => toggleLog(on),
  editSetup: () => openSetup("edit"),
  serverSearch: (q) => {
    $("#sv-q").value = q;
    renderServers();
    const first = document.querySelector("#sv-rows tr");
    if (first) first.click();
  },
  fillSetup: (v) => {
    for (const k of ["server", "name", "clan", "skin"]) if (typeof v[k] === "string") $(`#f-${k}`).value = v[k];
    for (const r of document.querySelectorAll("input[name=brain]")) r.checked = r.value === v.brain;
  },
  submitSetup: () => $("#setup-form").requestSubmit(),
  selectLocked: () => {
    const r = svRows.find((x) => x.passworded);
    if (r) selectServer(r);
    return r ? r.address : null;
  },
  filters: (f) => {
    Object.assign(svFilters, f);
    for (const c of document.querySelectorAll(".chip[data-f]")) c.classList.toggle("on", svFilters[c.dataset.f]);
    renderServers();
  },
};
