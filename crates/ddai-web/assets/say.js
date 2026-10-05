// Task 4.9 (D-094): the chat input — the owner types a line on the website and the bot says it in the game chat.
//
// The only chat the bot may send is the typed /kill fallback and this. The page only asks: `POST /api/bot/say` ({team, text}) checks the
// session, the CSRF token and the Origin, validates the text, and hands it to the bot, which validates it again, paces it (3 s apart, at most
// 10 a minute, a queue of 3) and says nothing while it is not in the game. Everything below only helps the owner: the same checks are made
// by the server. All text is set with `textContent`; styles are in say.css (the CSP forbids inline styles). The line is never stored by
// this page: a refused line stays in the box for a retry, a taken one is cleared.
//
// Integration: `SayCard.mount(rootElement, api, options)`; `api(method, path, body)` resolves `{status, data}` and carries the session and
// the CSRF token (app.js's own helper). `options.embedded: true` draws the bare form without a card and a heading, for placing it under
// the chat panel of the «Игра» tab (docs: crates/ddai-web/README.md, "Чат: строка владельца идёт в игровой чат"). `onShown()` / `onHidden()` are hooks for the tab.
// Task 5.11: when the bot answers `chat_disabled` (it runs with `--no-owner-chat`) the form is locked and says so until the tab is shown again
// (the flag only changes with a restart of the bot, so asking again on every line would be noise).
(function () {
  "use strict";

  var DDNET_SPACE = /^[\s\u0085\u00a0\u034f\u1680\u2000-\u200f\u2028-\u202f\u205f-\u2064\u206a-\u206f\u2800\u3000\ufe00-\ufe0f\ufff9-\ufffc]+/;
  var MAX_BYTES = 255; // the smaller of DDNet 20.1's server and client limits (ddai_net::owner_chat::MAX_OWNER_TEXT_BYTES)

  var REASON_TEXT = {
    not_in_game: "Бот сейчас не в игре: сообщение не отправлено.",
    chat_disabled: "Чат с сайта выключен на боте (--no-owner-chat или owner_chat = false).",
    queue_full: "Уже ждут три сообщения: дождитесь, пока они уйдут.",
    rate_limited: "Слишком часто: не чаще одного сообщения в 3 секунды и десяти в минуту.",
  };

  var ERROR_TEXT = {
    unauthenticated: "Сессия закончилась: войдите снова.",
    cross_origin: "Запрос отклонён: чужой источник.",
    missing_csrf: "Нет токена защиты: обновите страницу.",
    bad_csrf: "Токен защиты не подошёл: обновите страницу.",
    json_required: "Нужен JSON.",
    bad_request: "Запрос не принят: неверный формат.",
    rate_limited: "Слишком часто: подождите несколько секунд.",
    demo_only: "Сейчас на сайте показ мухи, а не живой бот: писать некому.",
    bot_unavailable: "Бот не запущен или не принимает команды.",
    bot_timeout: "Бот не ответил вовремя.",
    bot_protocol: "Ответ бота не понят.",
  };

  var INVALID_TEXT = {
    empty: "Пустое сообщение.",
    too_long: "Слишком длинное: не больше " + MAX_BYTES + " байт.",
    control: "В сообщении есть управляющие, невидимые символы или перевод строки.",
    command: "Команды (строки с «/» в начале) отсюда не отправляются.",
    reserved: "Эта строка у сервера служебная (признак бота): её сказать нельзя.",
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

  var encoder = typeof TextEncoder === "function" ? new TextEncoder() : null;

  function byteLength(s) {
    if (encoder) {
      return encoder.encode(s).length;
    }
    return unescape(encodeURIComponent(s)).length;
  }

  // The page's own pre-check, the same rules as the server's (the server decides): returns a detail code or null.
  function check(raw) {
    var text = raw.trim();
    if (text.length === 0) {
      return "empty";
    }
    if (byteLength(text) > MAX_BYTES) {
      return "too_long";
    }
    // eslint-disable-next-line no-control-regex
    if (/[\u0000-\u001f\u007f-\u009f\u2028\u2029\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff\u180e\u115f\u1160\u3164\uffa0]/.test(text)) {
      return "control";
    }
    // DDNet's own "space" (str_utf8_isspace) is skipped before the slash is looked for, as the server does.
    if (text.replace(DDNET_SPACE, "").charAt(0) === "/") {
      return "command";
    }
    if (text === "xd sure chillerbot.png is lyfe") {
      return "reserved";
    }
    return null;
  }

  function reply(res) {
    var d = res.data || {};
    if (res.status === 200 && d.ok) {
      return { ok: true, text: "Принято: бот скажет это в чат." + (d.text && /about (\d+)/.test(d.text) ? " Очередь: примерно через " + /about (\d+)/.exec(d.text)[1] + " с." : "") };
    }
    if (d.error === "invalid_text") {
      return { ok: false, text: INVALID_TEXT[d.detail] || "Сообщение не принято." };
    }
    if (d.reason && REASON_TEXT[d.reason]) {
      var more = d.reason === "rate_limited" && d.text && /about (\d+)/.test(d.text) ? " Попробуйте через " + /about (\d+)/.exec(d.text)[1] + " с." : "";
      return { ok: false, text: REASON_TEXT[d.reason] + more };
    }
    if (d.error && ERROR_TEXT[d.error]) {
      return { ok: false, text: ERROR_TEXT[d.error] };
    }
    return { ok: false, text: "Не отправлено (ошибка " + res.status + ")." };
  }

  var SayCard = (function () {
    var api = null;
    var root = null;
    var ui = {};
    var busy = false;
    var locked = false; // the bot says it takes no lines from the site (`chat_disabled`)

    function setResult(text, ok) {
      ui.result.textContent = text || "";
      ui.result.className = "say-result" + (text ? (ok ? " ok" : " bad") : "");
    }

    function updateCount() {
      var n = byteLength(ui.input.value.trim());
      ui.count.textContent = n + " / " + MAX_BYTES;
      ui.count.className = "say-count" + (n > MAX_BYTES ? " bad" : "");
    }

    function updateButton() {
      ui.send.disabled = busy || locked || ui.input.value.trim().length === 0;
    }

    // The form shows the bot's refusal to take lines as a state, not only as the answer to one line.
    function setLocked(on, text) {
      locked = on;
      ui.input.disabled = on;
      ui.team.disabled = on;
      ui.form.className = "say-form" + (on ? " say-locked" : "");
      if (on) {
        setResult(text, false);
      } else if (ui.result.className.indexOf("bad") >= 0) {
        setResult("", true);
      }
      updateButton();
    }

    function send() {
      if (busy) {
        return;
      }
      var raw = ui.input.value;
      var problem = check(raw);
      if (problem) {
        setResult(INVALID_TEXT[problem], false);
        return;
      }
      busy = true;
      updateButton();
      setResult("Отправка…", true);
      api("POST", "/api/bot/say", { team: ui.team.checked, text: raw.trim() })
        .then(function (res) {
          var r = reply(res);
          setResult(r.text, r.ok);
          if (res.data && res.data.reason === "chat_disabled") {
            setLocked(true, r.text);
          }
          if (r.ok) {
            ui.input.value = "";
            updateCount();
          }
        })
        .catch(function () {
          setResult("Нет связи с сайтом.", false);
        })
        .then(function () {
          busy = false;
          updateButton();
          ui.input.focus();
        });
    }

    function build(embedded) {
      clear(root);
      var card = el("section", embedded ? "say-card say-embedded" : "card say-card");
      if (!embedded) {
        card.setAttribute("aria-labelledby", "say-title");
        var title = el("h2", null, "Чат");
        title.id = "say-title";
        card.appendChild(title);
        card.appendChild(
          el(
            "p",
            "hint",
            "То, что вы напишете здесь, бот скажет в игровом чате. Сам он ничего не пишет. Не чаще одного сообщения в 3 секунды, не больше десяти в минуту."
          )
        );
      }
      var form = el("form", "say-form");
      ui.form = form;
      form.setAttribute("autocomplete", "off");
      form.addEventListener("submit", function (ev) {
        ev.preventDefault();
        send();
      });
      ui.input = el("input", "say-input");
      ui.input.type = "text";
      ui.input.setAttribute("aria-label", "Сообщение в игровой чат");
      ui.input.placeholder = "Сообщение в игровой чат";
      ui.input.maxLength = MAX_BYTES; // UTF-16 units: never more than MAX_BYTES bytes' worth of ASCII; the byte check is exact
      ui.input.setAttribute("enterkeyhint", "send");
      ui.input.addEventListener("input", function () {
        updateCount();
        updateButton();
        if (ui.result.className.indexOf("bad") >= 0) {
          setResult("", true);
        }
      });
      ui.send = el("button", "say-send", "Сказать");
      ui.send.type = "submit";
      ui.send.disabled = true;
      var row = el("div", "say-row");
      row.appendChild(ui.input);
      row.appendChild(ui.send);
      form.appendChild(row);

      var opts = el("div", "say-opts");
      var teamLabel = el("label", "say-team");
      ui.team = document.createElement("input");
      ui.team.type = "checkbox";
      teamLabel.appendChild(ui.team);
      teamLabel.appendChild(document.createTextNode(" Командный чат"));
      ui.count = el("span", "say-count", "0 / " + MAX_BYTES);
      opts.appendChild(teamLabel);
      opts.appendChild(ui.count);
      form.appendChild(opts);

      ui.result = el("p", "say-result");
      ui.result.setAttribute("role", "status");
      form.appendChild(ui.result);
      card.appendChild(form);
      root.appendChild(card);
      updateCount();
    }

    function mount(rootEl, apiFn, options) {
      root = rootEl;
      api = apiFn;
      busy = false;
      locked = false;
      build(!!(options && options.embedded));
    }

    function onShown() {
      if (locked) {
        setLocked(false);
      }
    }

    function onHidden() {}

    return { mount: mount, onShown: onShown, onHidden: onHidden, check: check };
  })();

  window.SayCard = SayCard;
})();
