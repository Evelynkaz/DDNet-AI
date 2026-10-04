(function () {
  "use strict";

  // -----------------------------------------------------------------------------------------
  // Task 5.1: login / status (unchanged in spirit; extended with the WS reconnect-with-backoff
  // acceptance criterion 3 asks for, and to dispatch task 5.2a's new message types to GameView).
  // -----------------------------------------------------------------------------------------

  var shellEl = document.querySelector(".shell");
  var loginView = document.getElementById("login-view");
  var statusView = document.getElementById("status-view");
  var loginForm = document.getElementById("login-form");
  var loginError = document.getElementById("login-error");
  var passwordInput = document.getElementById("password");
  var logoutButton = document.getElementById("logout-button");
  var connDot = document.getElementById("conn-dot");
  var wsStateEl = document.getElementById("ws-state");
  var botStateEl = document.getElementById("bot-state");
  var uptimeEl = document.getElementById("uptime");
  var tabbar = document.getElementById("tabbar");
  var tabStatusButton = document.getElementById("tab-status");
  var tabGameButton = document.getElementById("tab-game");
  var gameViewEl = document.getElementById("game-view");
  var tabBotButton = document.getElementById("tab-bot");
  var botViewEl = document.getElementById("bot-view");
  var tabFlyButton = document.getElementById("tab-fly");
  var flyViewEl = document.getElementById("fly-view");
  var tabTrainButton = document.getElementById("tab-train");
  var trainViewEl = document.getElementById("train-view");

  var csrfToken = null;
  var socket = null;
  var reconnectAttempts = 0;
  var reconnectTimer = null;
  var explicitlyClosed = true;
  var currentTab = "status";

  function showLogin() {
    shellEl.hidden = false;
    loginView.hidden = false;
    statusView.hidden = true;
    tabbar.hidden = true;
    gameViewEl.hidden = true;
    botViewEl.hidden = true;
    flyViewEl.hidden = true;
    trainViewEl.hidden = true;
    BotPanel.onHidden();
    FlyPanel.onHidden();
    TrainPanel.onHidden();
    setConnected(false);
  }

  function showTab(name) {
    currentTab = name;
    // Task 5.10: the window around login and status is only there for those two (the other tabs have windows of their own).
    shellEl.hidden = name !== "status";
    loginView.hidden = true;
    statusView.hidden = name !== "status";
    gameViewEl.hidden = name !== "game";
    botViewEl.hidden = name !== "bot";
    flyViewEl.hidden = name !== "fly";
    trainViewEl.hidden = name !== "train";
    tabTrainButton.classList.toggle("active", name === "train");
    tabFlyButton.classList.toggle("active", name === "fly");
    tabStatusButton.classList.toggle("active", name === "status");
    tabGameButton.classList.toggle("active", name === "game");
    tabBotButton.classList.toggle("active", name === "bot");
    if (name === "game") {
      GameView.onShown();
    }
    if (name === "bot") {
      BotPanel.onShown();
    } else {
      BotPanel.onHidden();
    }
    if (name === "fly") {
      FlyPanel.onShown();
    } else {
      FlyPanel.onHidden();
    }
    if (name === "train") {
      TrainPanel.onShown();
    } else {
      TrainPanel.onHidden();
    }
  }

  function showAuthenticated() {
    tabbar.hidden = false;
    showTab(
      currentTab === "game" || currentTab === "bot" || currentTab === "fly" || currentTab === "train" ? currentTab : "status",
    );
  }

  function setConnected(on) {
    connDot.classList.toggle("dot-on", on);
    connDot.classList.toggle("dot-off", !on);
    wsStateEl.textContent = on ? "подключено" : "не подключено";
    if (!on) {
      SourceBadge.hide();
    }
    GameView.onConnectionChanged(on);
    FlyPanel.onConnectionChanged(on);
  }

  function pad2(value) {
    return String(value).padStart(2, "0");
  }

  function formatUptime(seconds) {
    var total = Math.max(0, Math.floor(seconds));
    var h = Math.floor(total / 3600);
    var m = Math.floor((total % 3600) / 60);
    var s = total % 60;
    return pad2(h) + ":" + pad2(m) + ":" + pad2(s);
  }

  function sendJson(message) {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
    }
  }

  function scheduleReconnect() {
    if (explicitlyClosed) {
      return;
    }
    // Exponential backoff, capped at 10s (acceptance criterion 3: "reconnects automatically with
    // backoff and resubscribes").
    var delay = Math.min(10000, 500 * Math.pow(2, reconnectAttempts));
    reconnectAttempts += 1;
    reconnectTimer = setTimeout(function () {
      reconnectTimer = null;
      connectWs();
    }, delay);
  }

  function connectWs() {
    if (socket) {
      return;
    }
    explicitlyClosed = false;
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    var proto = location.protocol === "https:" ? "wss:" : "ws:";
    socket = new WebSocket(proto + "//" + location.host + "/ws");
    // Binary `live` frames arrive as `ArrayBuffer` (not `Blob`) so decoding never needs an extra
    // async round trip on the hot 25 Hz path.
    socket.binaryType = "arraybuffer";

    socket.addEventListener("open", function () {
      reconnectAttempts = 0;
    });
    socket.addEventListener("message", function (event) {
      if (event.data instanceof ArrayBuffer) {
        // Binary messages: `DFLY` (the fly's frame, task 7.4) or the live map's `DWLF`.
        var head = new Uint8Array(event.data, 0, Math.min(4, event.data.byteLength));
        if (head.length === 4 && head[0] === 0x44 && head[1] === 0x46) {
          FlyPanel.onFrame(event.data);
        } else {
          GameView.onLiveFrame(event.data);
        }
        return;
      }
      var msg;
      try {
        msg = JSON.parse(event.data);
      } catch (parseError) {
        return;
      }
      switch (msg.type) {
        case "hello":
          // Connection state is confirmed once the app-level `hello` message arrives, which
          // proves both the WebSocket handshake and our own protocol worked, not merely `onopen`.
          setConnected(true);
          break;
        case "status":
          botStateEl.textContent = msg.bot_state;
          uptimeEl.textContent = formatUptime(msg.uptime_s);
          break;
        case "map":
          GameView.onMap(msg);
          SourceBadge.setMap(msg.name);
          break;
        case "players":
          GameView.onPlayers(msg.list);
          break;
        case "events":
          GameView.onEvents(msg);
          break;
        case "replay_status":
          GameView.onReplayStatus(msg);
          break;
        case "bot":
          GameView.onBotStatus(msg.status);
          break;
        case "fly_meta":
          FlyPanel.onMeta(msg.meta);
          break;
        case "source":
          onSource(msg);
          break;
        case "chat":
          GameView.onChat(msg.line);
          break;
        case "chat_history":
          GameView.onChatHistory(msg.lines);
          break;
        case "live_error":
          GameView.onLiveError(msg.message);
          break;
        default:
          break; // forward-compatible: ignore anything this build doesn't know about
      }
    });
    socket.addEventListener("close", function () {
      setConnected(false);
      socket = null;
      // Review round 1, finding F5: a WS close after the session itself ended server-side
      // (logout on another tab, idle/absolute timeout, a password rotation) used to retry forever
      // at up to a 10s backoff while still showing the authenticated UI — the upgrade would just
      // keep getting a 401, and the user never saw the login form again without a manual reload.
      // `/api/me` is the same cheap, already-existing check `refresh()` uses at page load; only
      // reconnect if it says the session is still actually valid (a real transient network blip),
      // otherwise show the login form and stop retrying outright.
      if (explicitlyClosed) {
        return; // our own `disconnectWs()` call (e.g. the logout button) — nothing more to do
      }
      fetch("/api/me", { credentials: "same-origin" })
        .then(function (response) {
          return response.json();
        })
        .then(function (data) {
          if (data.authenticated) {
            scheduleReconnect();
          } else {
            csrfToken = null;
            explicitlyClosed = true; // stop any already-pending reconnect timer too
            if (reconnectTimer) {
              clearTimeout(reconnectTimer);
              reconnectTimer = null;
            }
            showLogin();
          }
        })
        .catch(function () {
          // `/api/me` itself is unreachable (the more likely "genuine network blip" case `/api/me`
          // was meant to rule OUT) — fall back to the old retry-with-backoff behavior rather than
          // giving up on a merely-offline connection.
          scheduleReconnect();
        });
    });
    socket.addEventListener("error", function () {
      setConnected(false);
    });
  }

  function disconnectWs() {
    explicitlyClosed = true;
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    if (socket) {
      socket.close();
      socket = null;
    }
    setConnected(false);
  }

  function refresh() {
    fetch("/api/me", { credentials: "same-origin" })
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        if (data.authenticated) {
          csrfToken = data.csrf_token;
          showAuthenticated();
          connectWs();
        } else {
          csrfToken = null;
          showLogin();
        }
      })
      .catch(function () {
        showLogin();
      });
  }

  loginForm.addEventListener("submit", function (event) {
    event.preventDefault();
    loginError.hidden = true;
    fetch("/api/login", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: passwordInput.value }),
    })
      .then(function (response) {
        if (!response.ok) {
          throw new Error("login failed");
        }
        return response.json();
      })
      .then(function (data) {
        csrfToken = data.csrf_token;
        passwordInput.value = "";
        showAuthenticated();
        connectWs();
      })
      .catch(function () {
        loginError.textContent = "Неверный пароль или сервер недоступен.";
        loginError.hidden = false;
      });
  });

  logoutButton.addEventListener("click", function () {
    fetch("/api/logout", {
      method: "POST",
      credentials: "same-origin",
      headers: { "X-CSRF-Token": csrfToken || "" },
    }).finally(function () {
      disconnectWs();
      csrfToken = null;
      showLogin();
    });
  });

  tabStatusButton.addEventListener("click", function () {
    showTab("status");
  });
  tabGameButton.addEventListener("click", function () {
    showTab("game");
  });
  tabBotButton.addEventListener("click", function () {
    showTab("bot");
  });
  tabFlyButton.addEventListener("click", function () {
    showTab("fly");
  });
  tabTrainButton.addEventListener("click", function () {
    showTab("train");
  });
  FlyPanel.attach(sendJson);
  GameView.attach({ send: sendJson, goTab: showTab });

  // -----------------------------------------------------------------------------------------
  // Task 5.7: what the site shows. The live bot has priority; while it is not running the offline demo (the fly playing an
  // arena, not a real game) stands in, and the server says which one is on show (`{"type":"source","kind":...}`). The badge
  // is on the «Игра», «Бот» and «Муха» tabs; every string is set with `textContent`.
  // -----------------------------------------------------------------------------------------

  var SourceBadge = (function () {
    var TITLES = {
      live: "Живой бот",
      demo: "Показ: муха на арене (не настоящая игра)",
      none: "Нет источника: бот не запущен",
    };
    var badges = ["game-source", "bot-source", "fly-source"]
      .map(function (id) {
        return document.getElementById(id);
      })
      .filter(Boolean);
    var kind = null;
    var info = null;
    var mapName = null;

    function detail() {
      var parts = [];
      if (kind === "live" || kind === "demo") {
        if (mapName) {
          parts.push("карта: " + mapName);
        }
      }
      if (kind === "demo" && info) {
        if (info.arena) {
          parts.push("арена: " + info.arena);
        }
        if (info.bundle) {
          parts.push("веса: " + info.bundle);
        }
      }
      return parts.join(" · ");
    }

    function render() {
      badges.forEach(function (el) {
        if (!kind || !TITLES[kind]) {
          el.hidden = true;
          return;
        }
        el.hidden = false;
        el.className = "source-badge source-" + kind;
        el.querySelector(".source-title").textContent = TITLES[kind];
        el.querySelector(".source-detail").textContent = detail();
      });
    }

    return {
      set: function (newKind, newInfo) {
        kind = newKind;
        info = newInfo && typeof newInfo === "object" ? newInfo : null;
        render();
      },
      setMap: function (name) {
        mapName = name || null;
        render();
      },
      // The connection to the site dropped: nothing is known about what is shown until it is back.
      hide: function () {
        kind = null;
        render();
      },
    };
  })();

  // What the server last said is on show (`live` / `demo` / `none`), kept across a reconnect so that a change that happened
  // while the connection was down still clears the previous source's drawing.
  var currentSource = null;

  function onSource(msg) {
    var kind = typeof msg.kind === "string" ? msg.kind : null;
    if (kind !== "live" && kind !== "demo" && kind !== "none") {
      return;
    }
    if (currentSource !== null && currentSource !== kind) {
      // Another source: what was drawn of the previous one (map, players, frames) is not this one's.
      GameView.reset();
      SourceBadge.setMap(null);
    }
    currentSource = kind;
    GameView.setSource(kind);
    GameView.setDemo(kind === "demo");
    SourceBadge.set(kind, msg.info);
    BotPanel.setDemo(kind === "demo");
  }

  // -----------------------------------------------------------------------------------------
  // Task 5.10: the live map view lives in game.js (`window.GameView`, with ddmap.js and ddtee.js): the real DDNet map and
  // tees, the camera, the scoreboard and the read-only chat. This file hands it the WebSocket messages.
  // -----------------------------------------------------------------------------------------

  // -----------------------------------------------------------------------------------------
  // Task 5.6: bot control (owner only). Status panel, commands, friends/war/ignore editor.
  // Every request carries the session cookie and the CSRF token; a 401 means the session ended
  // (back to the login form). Everything dynamic goes through `textContent`, never `innerHTML`,
  // so a nickname (or a bot reply) can never be interpreted as markup. Nothing is written to the
  // console, to storage or to the URL: names stay on this page.
  // -----------------------------------------------------------------------------------------

  var BotPanel = (function () {
    var STATUS_POLL_MS = 2000;
    var TICKS_PER_SECOND = 50; // DDNet server ticks (SERVER_TICK_SPEED); the kill cooldown is 500 ticks = 10 s
    var timer = null;
    var shown = false;
    var lastStatus = null;
    var busy = false;
    // Task 5.7: the site shows the offline demo, not a bot. The status is then "not running"; and when no bot's control socket
    // is there either, the commands are off (the server refuses them as well: `demo_only`). Nothing is ever sent to the demo.
    // A bot that runs without a bridge keeps its control socket, so it stays commandable under the demo's badge.
    var demoShown = false;
    var controlUp = false;

    var KIND_LABELS = {
      friend: "Друзья",
      war: "Война",
      ignore: "Игнор",
      clanfriend: "Клан-друзья",
      clanwar: "Клан-война",
    };
    var KIND_ORDER = ["friend", "war", "ignore", "clanfriend", "clanwar"];
    var MODE_LABELS = { fight: "бой", passive: "пассивный", hold: "стоит", goto: "идёт в точку" };
    var APPLIED_TEXT = {
      applied: "применено к работающему боту (бот перечитал тот же файл)",
      mismatch: "ВНИМАНИЕ: бот перечитал другой файл списков, не тот, что правит сайт (проверьте --relations у бота и сайта): бот может не щадить ваших друзей",
      unverified: "бот ответил, но сверить, что он прочитал тот же файл, не удалось",
      unchanged: "без изменений",
      unavailable: "бот не запущен: список сохранён и вступит в силу при запуске",
      rate_limited: "сохранено, но бот пока не принял (слишком частые команды): нажмите «Перечитать списки в боте»",
      busy: "сохранено, но бот занят: нажмите «Перечитать списки в боте»",
      timeout: "сохранено, но бот не ответил вовремя: нажмите «Перечитать списки в боте»",
      bot_refused: "сохранено, но бот не принял перечитывание списков",
      error: "сохранено, но ответ бота не понят",
    };
    var ERROR_TEXT = {
      unauthenticated: "Сессия закончилась.",
      cross_origin: "Запрос отклонён проверкой источника.",
      missing_csrf: "Нет токена защиты: обновите страницу.",
      bad_csrf: "Токен защиты не подошёл: обновите страницу.",
      bot_unavailable: "Бот не запущен (нет сокета управления).",
      demo_only: "Бот не запущен: сейчас на сайте показ, команды отключены.",
      bot_timeout: "Бот не ответил вовремя.",
      bot_protocol: "Ответ бота не понят.",
      bad_request: "Некорректный запрос.",
      invalid: "Недопустимое значение.",
      invalid_name: "Недопустимое имя.",
      list_full: "Список заполнен.",
      relations_unreadable: "Файл списков не читается (повреждён). Исправьте его вручную: сайт его не перезапишет.",
      relations_write_failed: "Не удалось записать файл списков.",
      json_required: "Нужен JSON.",
    };
    var NAME_DETAIL = {
      empty: "имя пустое после нормализации",
      too_long: "имя длиннее 64 байт",
      control: "в имени управляющие символы",
    };

    function el(id) {
      return document.getElementById(id);
    }

    function setText(id, text) {
      el(id).textContent = text;
    }

    function show(id, text, ok) {
      var node = el(id);
      node.textContent = text;
      node.classList.toggle("ok", ok === true);
      node.classList.toggle("bad", ok === false);
    }

    // Same normalisation as the bot's `fold_name` (trim, collapse whitespace, lower-case, drop one leading "(<digits>)"),
    // for the preview only: the server's answer is what is stored and what the bot matches on.
    function previewFold(name) {
      var folded = name.trim().split(/\s+/).join(" ").toLowerCase();
      // Every leading "(<digits>)", like the bot (the fold must be idempotent: the file is folded again on load).
      var stripped = folded.replace(/^\(\d+\)\s*/, "");
      while (stripped !== folded) {
        folded = stripped;
        stripped = folded.replace(/^\(\d+\)\s*/, "");
      }
      return folded.trim();
    }

    function api(method, path, body) {
      var headers = {};
      var options = { method: method, credentials: "same-origin", headers: headers };
      if (method !== "GET") {
        headers["Content-Type"] = "application/json";
        headers["X-CSRF-Token"] = csrfToken || "";
        options.body = JSON.stringify(body || {});
      }
      return fetch(path, options).then(function (response) {
        return response.json().catch(function () { return {}; }).then(function (data) {
          if (response.status === 401) {
            csrfToken = null;
            disconnectWs();
            showLogin();
          }
          return { status: response.status, data: data };
        });
      });
    }

    function errorText(res) {
      var d = res.data || {};
      if (d.error === "invalid_name" && NAME_DETAIL[d.detail]) {
        return "Недопустимое имя: " + NAME_DETAIL[d.detail] + ".";
      }
      if (d.error === "relations_write_failed" && d.detail === "read_only") {
        return "Файл списков для веб-юнита только для чтения: каталог data/bot создан заново или перенесён после его запуска. Перезапустите юнит: sudo systemctl restart ddnet-ai-web.";
      }
      if (d.code === "rate_limited" || res.status === 429) {
        return "Слишком много команд подряд: подождите секунду.";
      }
      if (d.text) {
        return d.text;
      }
      return ERROR_TEXT[d.error] || "Ошибка " + res.status + ".";
    }

    // ---- status panel ---------------------------------------------------------------------

    function fmtUs(us) {
      if (typeof us !== "number") {
        return "—";
      }
      return us >= 1000 ? (us / 1000).toFixed(2) + " мс" : us + " мкс";
    }

    var lastInfo = null;

    function renderStatus(info) {
      lastInfo = info;
      var dot = el("bot-conn-dot");
      var s = info && info.live ? info.status : null;
      lastStatus = s;
      var cooldownEl = el("kill-cooldown");
      if (!s) {
        dot.classList.remove("dot-on");
        dot.classList.add("dot-off");
        var why = !info
          ? "нет связи с сервером"
          : !info.bridge
            ? "мост к боту не подключён (сайт запущен без --bot-socket)"
            : demoShown && controlUp
              ? "живого статуса нет (моста бота не видно), на сайте показ; сокет управления на месте, команды идут боту"
              : demoShown
                ? "бот не запущен: сейчас на сайте показ (муха на арене), настоящей игры нет"
                : "бот не запущен (нет живого статуса)";
        setText("bot-conn-text", why);
        ["bs-server", "bs-map", "bs-mode", "bs-brain", "bs-target", "bs-wb", "bs-blocks", "bs-deaths", "bs-clips", "bs-latency", "bs-latency2", "bs-identity", "bs-tick"].forEach(function (id) {
          setText(id, "—");
        });
        cooldownEl.textContent = "—";
        updateButtons();
        return;
      }
      dot.classList.toggle("dot-on", !!s.connected);
      dot.classList.toggle("dot-off", !s.connected);
      setText("bot-conn-text", s.connected ? "В игре" : "Бот запущен, но не в игре (подключается или ждёт)");
      setText("bs-server", s.server || "—");
      setText("bs-map", s.map || "—");
      setText("bs-mode", MODE_LABELS[s.mode] || s.mode || "—");
      setText("bs-brain", s.brain || "—");
      setText("bs-target", s.target_tag || "нет");
      setText("bs-wb", (s.wb || "—") + (s.goto ? " · идёт: " + s.goto : ""));
      setText("bs-blocks", (s.blocks | 0) + " / " + (s.blocked_by | 0));
      setText("bs-deaths", (s.deaths | 0) + " / " + (s.self_kills | 0));
      setText("bs-clips", String(s.clips_saved | 0));
      setText("bs-latency", fmtUs(s.decide_p50_us) + " / " + fmtUs(s.decide_p99_us));
      setText("bs-latency2", fmtUs(s.brain_p99_us) + " / " + fmtUs(s.overhead_p99_us));
      setText("bs-identity", [s.name, s.clan, s.skin].filter(Boolean).join(" · ") || "—");
      setText("bs-tick", String(s.tick | 0));
      var cd = s.kill_cooldown_ticks | 0;
      cooldownEl.textContent = cd > 0 ? "доступно через " + Math.ceil(cd / TICKS_PER_SECOND) + " с" : "готово";
      updateButtons();
    }

    function updateButtons() {
      var s = lastStatus;
      var mode = s ? s.mode : null;
      // The bot's line is "WB: <auto|left|right|off>, ..." (docs/formats.md §23).
      var wbMatch = s && s.wb ? /^WB: (auto|left|right|off)\b/.exec(s.wb) : null;
      var wbMode = wbMatch ? wbMatch[1] : null;
      document.querySelectorAll('[data-cmd="mode"]').forEach(function (b) {
        b.classList.toggle("current", mode === b.getAttribute("data-mode"));
      });
      document.querySelectorAll('[data-cmd="wb"]').forEach(function (b) {
        b.classList.toggle("current", wbMode === b.getAttribute("data-mode"));
      });
      var brain = el("cmd-brain");
      if (s && s.brain && document.activeElement !== brain) {
        // The bot reports a descriptive name ("hybrid-none-4ms", "planner-normal-5ms", "hybrid:fly-..."): the select holds the kind.
        brain.value = String(s.brain).split(/[-:]/)[0];
      }
      el("cmd-kill").disabled = locked() || busy || !s || (s.kill_cooldown_ticks | 0) > 0;
    }

    // No bot to command: the demo is on show and no control socket is there.
    function locked() {
      return demoShown && !controlUp;
    }

    // The commands are on only while there is a bot to take them (and while no command is in flight).
    function applyDemoLock() {
      el("cmd-demo-note").hidden = !locked();
      document.querySelectorAll("#bot-cmd-card button, #bot-cmd-card input, #bot-cmd-card select").forEach(function (c) {
        c.disabled = locked() || busy;
      });
      // "Перечитать списки в боте" is a command to the bot; the lists editor itself is a file and stays.
      el("rel-reload").disabled = locked() || busy;
      updateButtons();
    }

    function setControl(up) {
      up = !!up;
      if (up === controlUp) {
        return;
      }
      controlUp = up;
      applyDemoLock();
    }

    function setDemo(on) {
      on = !!on;
      if (on === demoShown) {
        return;
      }
      demoShown = on;
      applyDemoLock();
      if (shown && !lastStatus) {
        renderStatus(lastInfo);
      }
    }

    function pollStatus() {
      api("GET", "/api/bot/status")
        .then(function (res) {
          if (res.status === 200 && res.data && typeof res.data.source === "string") {
            setDemo(res.data.source === "demo"); // the same answer as the WebSocket's, for when that is down
            setControl(res.data.control_socket);
          }
          if (shown) {
            renderStatus(res.status === 200 ? res.data : null);
          }
        })
        .catch(function () {
          if (shown) {
            renderStatus(null);
          }
        });
    }

    // ---- commands -------------------------------------------------------------------------

    function setBusy(on) {
      busy = on;
      document.querySelectorAll("#bot-view button").forEach(function (b) {
        // The «Запуск» card (launch.js) and the chat input (say.js) manage their own buttons.
        if (b.id !== "cmd-kill" && !b.closest("#launch-mount") && !b.closest("#say-mount")) {
          b.disabled = on;
        }
      });
      applyDemoLock();
    }

    function sendCommand(body, label) {
      if (locked()) {
        show("cmd-result", label + ": бот не запущен, на сайте показ — команды отключены.", false);
        return Promise.resolve();
      }
      setBusy(true);
      show("cmd-result", label + "…", null);
      return api("POST", "/api/bot/command", body)
        .then(function (res) {
          var d = res.data || {};
          if (res.status === 200 && d.ok) {
            show("cmd-result", label + ": " + (d.text || "готово"), true);
          } else {
            show("cmd-result", label + ": " + errorText(res), false);
          }
          pollStatus();
        })
        .catch(function () {
          show("cmd-result", label + ": нет связи с сайтом.", false);
        })
        .then(function () {
          setBusy(false);
        });
    }

    function onCommandClick(event) {
      var target = event.target.closest("[data-cmd]");
      if (!target) {
        return;
      }
      var cmd = target.getAttribute("data-cmd");
      var mode = target.getAttribute("data-mode");
      var label = target.textContent.trim();
      if (cmd === "mode" || cmd === "wb") {
        sendCommand({ type: cmd, mode: mode }, label);
      } else {
        sendCommand({ type: cmd }, label);
      }
    }

    function bindCommands() {
      el("bot-view").addEventListener("click", onCommandClick);
      el("cmd-brain-apply").addEventListener("click", function () {
        sendCommand({ type: "brain", brain: el("cmd-brain").value }, "Мозг " + el("cmd-brain").value);
      });
      el("cmd-kill").addEventListener("click", function () {
        if (window.confirm("Убить бота и возродить? Следующее убийство — не раньше чем через 10 с.")) {
          sendCommand({ type: "kill" }, "Убить");
        }
      });
      el("cmd-clip").addEventListener("click", function () {
        sendCommand({ type: "clip", note: el("cmd-clip-note").value.trim() }, "Клип").then(function () {
          el("cmd-clip-note").value = "";
        });
      });
      el("cmd-goto").addEventListener("click", function () {
        var x = parseInt(el("cmd-goto-x").value, 10);
        var y = parseInt(el("cmd-goto-y").value, 10);
        if (isNaN(x) || isNaN(y)) {
          show("cmd-result", "Идти: введите целые x и y (в тайлах).", false);
          return;
        }
        sendCommand({ type: "goto", x: x, y: y }, "Идти в " + x + ", " + y);
      });
      el("cmd-spec").addEventListener("click", function () {
        if (window.confirm("Отправить бота в наблюдатели? Вернуть можно кнопкой «В игру».")) {
          sendCommand({ type: "spec" }, "В наблюдатели");
        }
      });
    }

    // ---- relations editor -----------------------------------------------------------------

    function renderLists(lists) {
      var root = el("rel-lists");
      while (root.firstChild) {
        root.removeChild(root.firstChild);
      }
      KIND_ORDER.forEach(function (kind) {
        var group = document.createElement("div");
        group.className = "rel-group";
        var title = document.createElement("h3");
        var names = (lists && lists[kind]) || [];
        title.textContent = KIND_LABELS[kind] + " (" + names.length + ")";
        group.appendChild(title);
        if (names.length === 0) {
          var empty = document.createElement("span");
          empty.className = "rel-empty";
          empty.textContent = "пусто";
          group.appendChild(empty);
        } else {
          var ul = document.createElement("ul");
          ul.className = "rel-entries";
          names.forEach(function (name) {
            var li = document.createElement("li");
            var label = document.createElement("span");
            label.textContent = name;
            var remove = document.createElement("button");
            remove.type = "button";
            remove.className = "alt";
            remove.textContent = "×";
            remove.setAttribute("aria-label", "Убрать из списка «" + KIND_LABELS[kind] + "»: " + name);
            remove.addEventListener("click", function () {
              editRelations("remove", kind, name);
            });
            li.appendChild(label);
            li.appendChild(remove);
            ul.appendChild(li);
          });
          group.appendChild(ul);
        }
        root.appendChild(group);
      });
    }

    function loadLists() {
      return api("GET", "/api/bot/relations").then(function (res) {
        if (res.status === 200) {
          renderLists(res.data.lists);
        } else {
          show("rel-result", errorText(res), false);
        }
      });
    }

    function editRelations(op, kind, name) {
      show("rel-result", "…", null);
      return api("POST", "/api/bot/relations", { op: op, kind: kind, name: name })
        .then(function (res) {
          var d = res.data || {};
          if (res.status !== 200 || !d.ok) {
            show("rel-result", errorText(res), false);
            return;
          }
          renderLists(d.lists);
          var what = op === "add" ? "Добавлено" : d.changed ? "Убрано" : "Такого имени в списке не было";
          var text = what + " («" + d.normalised + "», список «" + KIND_LABELS[d.kind] + "»)";
          if (d.moved_from && d.moved_from.length) {
            text += "; снято из: " + d.moved_from.map(function (k) { return KIND_LABELS[k]; }).join(", ");
          }
          text += " — " + (APPLIED_TEXT[d.applied] || d.applied) + ".";
          if (d.applied_text && d.applied !== "unchanged") {
            text += " Ответ бота: " + d.applied_text + ".";
          }
          show("rel-result", text, d.applied === "applied" || d.applied === "unchanged");
          if (op === "add") {
            el("rel-name").value = "";
            updatePreview();
          }
        })
        .catch(function () {
          show("rel-result", "Нет связи с сайтом.", false);
        });
    }

    function updatePreview() {
      var raw = el("rel-name").value;
      var folded = previewFold(raw);
      setText("rel-preview", raw.trim() === "" ? "" : folded === "" ? "после нормализации имя пустое" : "будет сохранено как: «" + folded + "»");
    }

    function bindRelations() {
      el("rel-name").addEventListener("input", updatePreview);
      el("rel-form").addEventListener("submit", function (event) {
        event.preventDefault();
        var name = el("rel-name").value;
        if (previewFold(name) === "") {
          show("rel-result", "Введите непустое имя.", false);
          return;
        }
        editRelations("add", el("rel-kind").value, name);
      });
      el("rel-reload").addEventListener("click", function () {
        sendCommand({ type: "reload_relations" }, "Перечитать списки").then(function () {
          show("rel-result", el("cmd-result").textContent, el("cmd-result").classList.contains("ok"));
        });
      });
    }

    function onShown() {
      shown = true;
      LaunchCard.onShown();
      pollStatus();
      loadLists();
      if (!timer) {
        timer = setInterval(pollStatus, STATUS_POLL_MS);
      }
    }

    function onHidden() {
      shown = false;
      LaunchCard.onHidden();
      if (timer) {
        clearInterval(timer);
        timer = null;
      }
    }

    LaunchCard.mount(el("launch-mount"), api);
    SayCard.mount(el("say-mount"), api);
    bindCommands();
    bindRelations();
    return { onShown: onShown, onHidden: onHidden, setDemo: setDemo };
  })();

  refresh();
})();
