// Task 5.9 (D-089): the «Запуск» card of the «Бот» tab — start and stop the bot from the site, choose its brain.
//
// The page only asks: `POST /api/bot/launch` writes a small request file; a root helper (ddnet-ai launch apply) re-checks every value
// against fixed allow-lists and does the rest; the page then shows what the helper wrote (`GET /api/bot/launch`: the choices, the
// helper's status, whether a request still waits) next to the live bridge state (`GET /api/bot/status`). Nothing here is a path or a
// command line: the server is "local" or the address of a ready entry of the owner's allow-list, taken from the choices the site
// itself returned. All text is set with `textContent`; styles are in launch.css (the CSP forbids inline styles).
//
// Integration: app.js calls `LaunchCard.mount(rootElement, api)` once (`api(method, path, body)` resolves `{status, data}` and
// carries the session and the CSRF token) and `onShown()` / `onHidden()` with the tab.
(function () {
  "use strict";

  var POLL_MS = 2500;
  var POLL_FAST_MS = 1000;
  var PENDING_WARN_S = 15;

  var BRAIN_LABEL = { hybrid: "Гибрид", "hybrid-fly": "Гибрид + муха", fly: "Муха" };
  var DURATION_LABEL = { "15m": "15 мин", "60m": "1 час", unlimited: "До остановки" };

  // What a refusal or an ending means, in words for the owner. The helper only ever sends these codes.
  var REASON_TEXT = {
    stopped_by_owner: "Остановлен по вашей просьбе.",
    finished: "Время вышло, бот остановлен.",
    ended: "Бот остановлен не по кнопке «Остановить» (сигнал или остановка вручную; время ещё не вышло).",
    kicked_or_banned:
      "Сервер кикнул или забанил бота (код 3). Бот остановился и не пытается обойти бан. Этот сервер закрыт для запуска с сайта, пока владелец не откроет запись заново.",
    join_failed: "Не удалось войти на сервер или связь потеряна (код 4).",
    crashed: "Бот завершился с ошибкой; systemd перезапустит его, если это сбой.",
    bad_request: "Запрос не принят: неверный формат.",
    request_too_large: "Запрос слишком большой.",
    request_not_regular: "Запрос отклонён: это не обычный файл.",
    request_unreadable: "Запрос не удалось прочитать.",
    server_not_allowed: "Этот сервер не в списке разрешённых.",
    server_not_ready: "Сервер в списке, но владелец ещё не открыл его (ready).",
    server_ambiguous: "В списке серверов противоречивые записи для этого адреса.",
    server_bad_entry: "Запись сервера в списке разрешённых некорректна.",
    sparring_local_only: "Спарринг-боты бывают только на локальном сервере.",
    bundle_missing: "Файл мухи (bundle) не найден.",
    bundle_bad_path: "Путь к bundle в конфиге недопустим.",
    config_bad: "Конфиг запуска не читается.",
    config_untrusted: "Конфиг запуска доступен на запись не только root: отказ.",
    blocked_after_ban:
      "Этот сервер закрыт после кика или бана: запуск возможен, только когда владелец заново откроет запись в списке серверов.",
    state_unreadable: "Память запускателя повреждена: запуск закрыт, пока владелец её не проверит.",
    cooldown: "После кика, бана или ошибки входа запуск закрыт на 2 минуты.",
    rate_limited: "Слишком часто: не больше одного запуска в 30 секунд.",
    request_stale: "Запрос устарел (старше минуты или из будущего) и отброшен: отправьте заново.",
    already_running: "Бот уже запущен: сначала остановите его.",
    local_server_down: "Локальный сервер DDNet не запущен.",
    unit_overridden: "Юнит бота переопределён вручную (лишний drop-in): запуск закрыт, пока он не убран.",
    proxy_error: "Не удалось подготовить прокси для этого сервера.",
    live_servers_unreadable: "Список серверов не читается.",
    write_failed: "Не удалось записать настройки запуска.",
    state_write_failed: "Не удалось записать память запускателя.",
    systemctl_failed: "systemd не выполнил команду.",
    sparring_failed: "Спарринг-боты не запустились; бот остановлен.",
    internal: "Внутренняя ошибка запускателя.",
  };

  var HTTP_TEXT = {
    unauthenticated: "Сессия закончилась: войдите снова.",
    cross_origin: "Запрос отклонён: чужой источник.",
    missing_csrf: "Нет токена защиты: обновите страницу.",
    bad_csrf: "Токен защиты не подошёл: обновите страницу.",
    json_required: "Нужен JSON.",
    bad_request: "Запрос не принят: неверный формат.",
    server_not_allowed: "Этот сервер не в списке разрешённых.",
    sparring_local_only: "Спарринг-боты бывают только на локальном сервере.",
    bundle_missing: "Файл мухи (bundle) не найден.",
    rate_limited: "Слишком часто: подождите несколько секунд (не больше 6 запросов в минуту).",
    pending: "Предыдущий запрос ещё не обработан.",
    launcher_unavailable: "Запуск с сайта не установлен на сервере (нет каталога data/launch).",
    launch_write_failed: "Не удалось записать запрос.",
  };

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) {
      n.className = cls;
    }
    if (text !== undefined && text !== null) {
      n.textContent = text;
    }
    return n;
  }

  function clear(node) {
    while (node.firstChild) {
      node.removeChild(node.firstChild);
    }
  }

  function field(labelText, control) {
    var wrap = el("label", "lc-field");
    wrap.appendChild(el("span", "lc-label", labelText));
    wrap.appendChild(control);
    return wrap;
  }

  function select(options) {
    var s = document.createElement("select");
    options.forEach(function (o) {
      var opt = document.createElement("option");
      opt.value = o.value;
      opt.textContent = o.text;
      s.appendChild(opt);
    });
    return s;
  }

  function serverLabel(id) {
    return id === "local" ? "Локальный сервер" : "Сервер " + id;
  }

  var LaunchCard = (function () {
    var api = null;
    var root = null;
    var ui = {};
    var shown = false;
    var timer = null;
    var busy = false;
    var fastUntil = 0;
    var info = null; // the last GET /api/bot/launch
    var bridge = null; // the last GET /api/bot/status

    function build() {
      clear(root);
      var card = el("section", "card launch-card");
      card.setAttribute("aria-labelledby", "launch-title");
      var title = el("h2", null, "Запуск");
      title.id = "launch-title";
      card.appendChild(title);

      var state = el("p", "lc-state");
      ui.dot = el("span", "dot dot-off");
      ui.dot.setAttribute("aria-hidden", "true");
      ui.stateText = el("span", "lc-state-text", "загрузка…");
      state.appendChild(ui.dot);
      state.appendChild(document.createTextNode(" "));
      state.appendChild(ui.stateText);
      card.appendChild(state);
      ui.detail = el("p", "hint lc-detail");
      card.appendChild(ui.detail);

      ui.server = select([{ value: "local", text: serverLabel("local") }]);
      ui.brain = select([
        { value: "hybrid", text: BRAIN_LABEL.hybrid },
        { value: "hybrid-fly", text: BRAIN_LABEL["hybrid-fly"] },
        { value: "fly", text: BRAIN_LABEL.fly },
      ]);
      ui.duration = select([
        { value: "15m", text: DURATION_LABEL["15m"] },
        { value: "60m", text: DURATION_LABEL["60m"] },
        { value: "unlimited", text: DURATION_LABEL.unlimited },
      ]);
      ui.sparring = select([
        { value: "0", text: "Без спарринга" },
        { value: "1", text: "1 спарринг-бот" },
        { value: "2", text: "2 спарринг-бота" },
        { value: "3", text: "3 спарринг-бота" },
      ]);
      ui.mirror = select([
        { value: "on", text: "вкл" },
        { value: "off", text: "выкл" },
      ]);
      var form = el("div", "lc-form");
      form.appendChild(field("Сервер", ui.server));
      form.appendChild(field("Мозг", ui.brain));
      // The hybrid's opponent model (D-090); the fly brain alone has none, so the toggle is only shown for the hybrid brains.
      ui.mirrorField = field("Предсказание соперника", ui.mirror);
      form.appendChild(ui.mirrorField);
      form.appendChild(field("Длительность", ui.duration));
      ui.sparringField = field("Спарринг (только локальный сервер)", ui.sparring);
      form.appendChild(ui.sparringField);
      card.appendChild(form);
      ui.bundle = el("p", "hint lc-bundle");
      card.appendChild(ui.bundle);

      var buttons = el("div", "lc-buttons");
      ui.start = el("button", "lc-start", "Запустить");
      ui.start.type = "button";
      ui.stop = el("button", "lc-stop danger", "Остановить");
      ui.stop.type = "button";
      ui.watch = el("button", "lc-watch alt", "Смотреть игру");
      ui.watch.type = "button";
      ui.watch.hidden = true;
      buttons.appendChild(ui.start);
      buttons.appendChild(ui.stop);
      buttons.appendChild(ui.watch);
      card.appendChild(buttons);
      ui.result = el("p", "lc-result");
      ui.result.setAttribute("role", "status");
      card.appendChild(ui.result);
      card.appendChild(
        el(
          "p",
          "hint",
          "Бот не пишет в игровой чат и не обходит кики и баны. Запуск проверяет отдельная root-программа: сайт только присылает запрос.",
        ),
      );
      root.appendChild(card);

      ui.server.addEventListener("change", syncSparring);
      ui.brain.addEventListener("change", syncMirror);
      syncMirror();
      ui.start.addEventListener("click", onStart);
      ui.stop.addEventListener("click", onStop);
      ui.watch.addEventListener("click", function () {
        var tab = document.getElementById("tab-game");
        if (tab) {
          tab.click();
        }
      });
    }

    function syncMirror() {
      ui.mirrorField.hidden = ui.brain.value === "fly";
    }

    function syncSparring() {
      var local = ui.server.value === "local";
      ui.sparring.disabled = !local || busy;
      if (!local) {
        ui.sparring.value = "0";
      }
    }

    function setResult(text, ok) {
      ui.result.textContent = text || "";
      ui.result.classList.toggle("ok", ok === true);
      ui.result.classList.toggle("bad", ok === false);
    }

    function errorText(res) {
      var d = res.data || {};
      if (res.status === 429 && !d.error) {
        return HTTP_TEXT.rate_limited;
      }
      return HTTP_TEXT[d.error] || REASON_TEXT[d.error] || "Ошибка " + res.status + ".";
    }

    function applyChoices(i) {
      var current = ui.server.value;
      clear(ui.server);
      (i.servers || []).forEach(function (s) {
        var opt = document.createElement("option");
        opt.value = s.id;
        opt.textContent = serverLabel(s.id);
        ui.server.appendChild(opt);
      });
      ui.server.value = current;
      if (ui.server.value !== current) {
        ui.server.value = "local";
      }
      var fly = i.bundle_present;
      [ui.brain.options[1], ui.brain.options[2]].forEach(function (o) {
        o.disabled = !fly;
      });
      if (!fly && ui.brain.value !== "hybrid") {
        ui.brain.value = "hybrid";
      }
      syncMirror();
      ui.bundle.textContent = fly
        ? "Муха: " + i.bundle + " (run)."
        : "Муха недоступна: файл bundle (" + i.bundle + ") не найден.";
      var max = i.max_sparring || 0;
      for (var n = 0; n < ui.sparring.options.length; n += 1) {
        ui.sparring.options[n].hidden = n > max;
      }
      syncSparring();
    }

    function describe(status) {
      var parts = [];
      if (status.server) {
        parts.push(serverLabel(status.server));
      }
      if (status.brain) {
        parts.push(BRAIN_LABEL[status.brain] || status.brain);
      }
      if (status.duration) {
        parts.push(DURATION_LABEL[status.duration] || status.duration);
      }
      if (status.sparring) {
        parts.push("спарринг: " + status.sparring);
      }
      if (status.bundle && status.brain && status.brain !== "hybrid") {
        parts.push("муха " + status.bundle);
      }
      return parts.join(" · ");
    }

    function render() {
      if (!info) {
        return;
      }
      var enabled = !!info.enabled;
      var status = info.status || null;
      var live = !!(bridge && bridge.live && bridge.source === "live");
      var dot = "dot-off";
      var text;
      var detail = "";
      if (enabled && info.launcher_down) {
        text = "Запуск не отвечает";
        detail =
          "Прошлый запрос никто не обработал (его убрали). Проверьте юнит ddnet-ai-launch.path: sudo systemctl status ddnet-ai-launch.path; после сбоя sudo systemctl restart ddnet-ai-launch.path.";
      } else if (!enabled) {
        text = "Запуск с сайта не установлен";
        detail = "На сервере нет каталога data/launch: запустите deploy/install-launcher.sh.";
      } else if (info.pending) {
        dot = "dot-wait";
        text = "Запрос отправлен, ждём обработки…";
        if (info.pending_age_s >= PENDING_WARN_S) {
          detail = "Запрос не обработан уже " + info.pending_age_s + " с: запуск не отвечает? Проверьте юнит ddnet-ai-launch.path.";
        }
      } else if (live) {
        dot = "dot-on";
        text = "В игре";
        detail = status && status.state === "started" ? describe(status) : "";
      } else if (!status) {
        text = "Бот остановлен";
      } else if (status.state === "started") {
        dot = "dot-wait";
        text = "Запущен, входит в игру…";
        detail = describe(status);
      } else if (status.state === "stopped") {
        text = status.reason === "ended" ? "Бот остановлен (не по кнопке)" : "Бот остановлен";
        detail = REASON_TEXT[status.reason] || "";
      } else if (status.state === "failed") {
        text = "Бот остановился с ошибкой";
        detail = REASON_TEXT[status.reason] || "";
        if (typeof status.exit_code === "number" && status.reason === "crashed") {
          detail += " (код " + status.exit_code + ")";
        }
      } else if (status.state === "refused") {
        text = "Запрос отклонён";
        detail = REASON_TEXT[status.reason] || "Причина: " + status.reason;
      } else {
        text = "Ошибка запуска";
        detail = REASON_TEXT[status.reason] || "Причина: " + status.reason;
      }
      ui.dot.className = "dot " + dot;
      ui.stateText.textContent = text;
      ui.detail.textContent = detail;
      ui.start.disabled = busy || !enabled || info.pending || live;
      ui.stop.disabled = busy || !enabled || info.pending;
      ui.watch.hidden = !live;
      [ui.server, ui.brain, ui.duration, ui.mirror].forEach(function (c) {
        c.disabled = busy || !enabled;
      });
      syncSparring();
    }

    function poll() {
      if (!shown || !api) {
        return;
      }
      api("GET", "/api/bot/launch")
        .then(function (res) {
          if (res.status === 200 && res.data) {
            info = res.data;
            applyChoices(info);
          }
          return api("GET", "/api/bot/status");
        })
        .then(function (res) {
          bridge = res.status === 200 ? res.data : null;
          render();
        })
        .catch(function () {
          ui.stateText.textContent = "нет связи с сайтом";
        })
        .then(schedule);
    }

    function schedule() {
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
      if (shown) {
        timer = setTimeout(poll, Date.now() < fastUntil ? POLL_FAST_MS : POLL_MS);
      }
    }

    function send(body, label) {
      busy = true;
      render();
      setResult(label + "…", null);
      return api("POST", "/api/bot/launch", body)
        .then(function (res) {
          if (res.status === 202) {
            setResult(label + ": запрос отправлен, ждём ответ запускателя.", true);
            fastUntil = Date.now() + 20000;
          } else {
            setResult(label + ": " + errorText(res), false);
          }
        })
        .catch(function () {
          setResult(label + ": нет связи с сайтом.", false);
        })
        .then(function () {
          busy = false;
          if (timer) {
            clearTimeout(timer);
          }
          poll();
        });
    }

    function onStart() {
      var body = {
        action: "start",
        brain: ui.brain.value,
        server: ui.server.value,
        duration: ui.duration.value,
        sparring: ui.server.value === "local" ? parseInt(ui.sparring.value, 10) || 0 : 0,
      };
      if (ui.brain.value !== "fly") {
        body.mirror = ui.mirror.value === "off" ? "off" : "on";
      }
      if (body.server !== "local") {
        var ok = window.confirm("Запустить бота на публичном сервере? Бот не пишет в чат и не обходит баны.");
        if (!ok) {
          return;
        }
      }
      send(body, "Запуск");
    }

    function onStop() {
      send({ action: "stop" }, "Остановка");
    }

    function mount(rootEl, apiFn) {
      root = rootEl;
      api = apiFn;
      build();
    }

    function onShown() {
      shown = true;
      if (api) {
        poll();
      }
    }

    function onHidden() {
      shown = false;
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
    }

    return { mount: mount, onShown: onShown, onHidden: onHidden };
  })();

  window.LaunchCard = LaunchCard;
})();
