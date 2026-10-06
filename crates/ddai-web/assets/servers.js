// Task 5.12 (D-099): the «Серверы» tab — the DDNet server list, the owner's favourites and the proxy profiles.
//
// The page only asks. The list is a cache another unit fetched (the site itself has no network); favourites and proxy profiles are
// files the site writes after the owner's explicit action; a root helper re-checks a favourite before the bot is ever pointed at it,
// and a server the bot was kicked or banned from stays closed until «Открыть снова». Nothing here picks, switches or falls back to
// another proxy. The proxy password is write-only: this page never receives it (nor the user name), and an empty field on edit keeps
// the stored value. All text is set with `textContent` (server names and map names come from the internet); styles are in servers.css
// (the CSP forbids inline styles).
//
// Integration: app.js calls `ServersPanel.mount(rootElement, api, hooks)` once (`api(method, path, body)` resolves `{status, data}`
// and carries the session and the CSRF token; `hooks.play(address)` opens the «Бот» tab with that server chosen) and
// `onShown()` / `onHidden()` with the tab.
(function () {
  "use strict";

  var DEFAULT_NICK = "Muha";
  var PAGE = 40;
  var FAV_POLL_MS = 8000;
  var CHECK_POLL_MS = 1500;

  var ERROR_TEXT = {
    unauthenticated: "Сессия закончилась: войдите снова.",
    cross_origin: "Запрос отклонён: чужой источник.",
    missing_csrf: "Нет токена защиты: обновите страницу.",
    bad_csrf: "Токен защиты не подошёл: обновите страницу.",
    json_required: "Нужен JSON.",
    bad_request: "Запрос не принят: неверный формат.",
    rate_limited: "Слишком часто: подождите немного.",
    consent_required: "Нужно подтвердить, что администратор сервера разрешает бота.",
    confirm_required: "Нужно подтверждение.",
    bad_address: "Адрес должен быть вида IP:порт с публичным IP-адресом (без имён, без локальных и частных адресов).",
    bad_name: "Имя: от 1 до 64 обычных символов, без переводов строк.",
    bad_nick: "Ник бота: от 1 до 15 символов из латиницы, цифр, «_» и «-».",
    bad_connection: "Подключение: «напрямую» или через прокси из списка.",
    bad_notes: "Заметка: до 200 обычных символов, без переводов строк.",
    proxy_unknown: "Такого прокси нет в списке.",
    duplicate: "Этот сервер уже в избранном (или в списке разрешённых владельца).",
    too_many: "Слишком много записей.",
    not_found: "Записи уже нет.",
    not_blocked: "Сервер не закрыт: открывать нечего.",
    favourites_invalid: "Файл избранного повреждён: сайт его не трогает, пока вы его не проверите (data/launch/favourites.json).",
    favourites_unreadable: "Файл избранного не читается (не обычный файл или слишком большой).",
    favourites_write_failed: "Не удалось записать избранное.",
    launcher_unavailable: "Каталог data/launch не создан на сервере: запустите deploy/install-launcher.sh.",
    bad_host: "Адрес прокси: только публичный IP-адрес (имена хостов и локальные адреса не принимаются).",
    bad_port: "Порт: от 1 до 65535.",
    bad_user: "Логин: от 1 до 255 печатных ASCII-символов.",
    bad_pass: "Пароль: от 1 до 255 печатных ASCII-символов (логин и пароль задаются вместе).",
    bad_relay: "Ретранслятор: «только хост прокси» или «публичный».",
    bad_session_pick: "Подбор сессии: выкл или 2–4.",
    proxy_invalid: "Профиль не проходит проверку формата (подбор сессии требует «{session}» в логине и наоборот).",
    proxy_exists: "Прокси с таким именем уже есть.",
    proxy_not_found: "Такого прокси нет.",
    proxy_not_managed: "Этот файл сделан вручную: сайт его не меняет и не удаляет.",
    too_many_proxies: "Слишком много прокси.",
    proxy_write_failed: "Не удалось записать профиль.",
    proxy_in_use: "Этот прокси назначен избранному серверу: сначала смените подключение там.",
    pending: "Предыдущая проверка ещё идёт.",
    check_write_failed: "Не удалось отправить запрос на проверку.",
    refresh_write_failed: "Не удалось отправить запрос на обновление списка.",
  };

  var CHECK_TEXT = {
    ok: "прокси работает",
    udp_not_supported: "UDP не поддерживается (только TCP): для игры не годится",
    auth_failed: "логин или пароль не приняты",
    no_auth_method: "прокси не принял способ входа (нужен логин и пароль?)",
    timeout: "нет ответа (таймаут)",
    connect_failed: "не удалось подключиться к прокси",
    refused: "прокси отказал",
    relay_refused: "адрес ретранслятора отклонён правилами",
    probe_failed: "ретранслятор не пропускает UDP",
    protocol_error: "прокси ответил непонятно",
    proxy_missing: "файл профиля не найден",
    proxy_file_bad: "файл профиля неверный (права должны быть 0600)",
    request_stale: "запрос устарел: нажмите ещё раз",
    bad_request: "запрос не принят",
  };

  var REFRESH_TEXT = {
    no_master: "ни один мастер-сервер не ответил",
    bad_list: "мастер-сервер вернул негодный список",
    write_failed: "не удалось записать кэш",
  };

  var RELAY_LABEL = { "proxy-host-only": "только хост прокси", public: "публичный ретранслятор" };
  var RELAY_NOTE = {
    same_host: "ретранслятор на хосте прокси",
    substituted: "ретранслятор объявлен на другом хосте (не доверяем, используется хост прокси)",
    remote: "ретранслятор на другом хосте",
  };
  var REGION_LABEL = {
    eu: "Европа", na: "Северная Америка", as: "Азия", sa: "Южная Америка", oc: "Океания", af: "Африка",
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

  function field(labelText, control, cls) {
    var wrap = el("label", "sv-field" + (cls ? " " + cls : ""));
    wrap.appendChild(el("span", "sv-label", labelText));
    wrap.appendChild(control);
    return wrap;
  }

  function input(type, placeholder, maxlength) {
    var i = document.createElement("input");
    i.type = type;
    if (placeholder) {
      i.placeholder = placeholder;
    }
    if (maxlength) {
      i.maxLength = maxlength;
    }
    i.autocomplete = "off";
    i.spellcheck = false;
    return i;
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

  function button(text, cls, onClick) {
    var b = el("button", cls || "alt", text);
    b.type = "button";
    if (onClick) {
      b.addEventListener("click", onClick);
    }
    return b;
  }

  function errorText(res) {
    var d = res.data || {};
    if (res.status === 429 && !d.error) {
      return ERROR_TEXT.rate_limited;
    }
    return ERROR_TEXT[d.error] || "Ошибка " + res.status + ".";
  }

  function when(ts) {
    if (!ts) {
      return "—";
    }
    var d = new Date(ts * 1000);
    function p(n) {
      return String(n).padStart(2, "0");
    }
    return d.getFullYear() + "-" + p(d.getMonth() + 1) + "-" + p(d.getDate()) + " " + p(d.getHours()) + ":" + p(d.getMinutes());
  }

  function ago(seconds) {
    if (seconds < 90) {
      return seconds + " с";
    }
    if (seconds < 5400) {
      return Math.round(seconds / 60) + " мин";
    }
    return Math.round(seconds / 3600) + " ч";
  }

  function regionOf(location) {
    var i = (location || "").indexOf(":");
    return i < 0 ? location || "" : location.slice(0, i);
  }

  var ServersPanel = (function () {
    var api = null;
    var hooks = null;
    var root = null;
    var ui = {};
    var shown = false;
    var favTimer = null;
    var checkTimer = null;

    var list = { loaded: false, problem: null, servers: [], fetched_at: 0, age_s: 0, refresh: null, master: 0, enabled: true };
    var favs = { loaded: false, error: null, items: [], enabled: true };
    var proxies = { loaded: false, items: [], check: null, last_check_id: null, pending: false, enabled: true };
    var shownCount = PAGE;
    var panel = "list";
    var editingProxy = null; // the name of the profile being edited, or null for a new one
    var addingFor = null; // the row (address) whose «В избранное» form is open
    var checkingName = null; // the profile whose check the page is waiting for
    var favSig = ""; // the last favourites answer, to re-draw only on a change

    // ------------------------------------------------------------------------------------------------ build

    function build() {
      clear(root);
      var intro = el("section", "card sv-intro");
      intro.appendChild(
        el(
          "p",
          "hint",
          "Выбирая сервер, вы подтверждаете, что его администратор разрешает бота. Бот сам не пишет в игровой чат, не обходит кики и баны и не меняет прокси или IP сам: после кика или бана сервер закрыт, пока вы не нажмёте «Открыть снова».",
        ),
      );
      var seg = el("div", "sv-seg");
      seg.setAttribute("role", "tablist");
      ui.segButtons = {};
      [
        ["list", "Список"],
        ["fav", "Избранное"],
        ["proxy", "Прокси"],
      ].forEach(function (p) {
        var b = el("button", "alt sv-seg-btn", p[1]);
        b.type = "button";
        b.setAttribute("role", "tab");
        b.addEventListener("click", function () {
          showPanel(p[0]);
        });
        ui.segButtons[p[0]] = b;
        seg.appendChild(b);
      });
      intro.appendChild(seg);
      root.appendChild(intro);

      ui.listPanel = buildList();
      ui.favPanel = buildFav();
      ui.proxyPanel = buildProxy();
      root.appendChild(ui.listPanel);
      root.appendChild(ui.favPanel);
      root.appendChild(ui.proxyPanel);
      showPanel("list");
    }

    function showPanel(name) {
      panel = name;
      ui.listPanel.hidden = name !== "list";
      ui.favPanel.hidden = name !== "fav";
      ui.proxyPanel.hidden = name !== "proxy";
      Object.keys(ui.segButtons).forEach(function (k) {
        ui.segButtons[k].classList.toggle("current", k === name);
        ui.segButtons[k].setAttribute("aria-selected", k === name ? "true" : "false");
      });
      if (shown) {
        if (name === "proxy") {
          loadProxies();
        }
        if (name === "fav" || name === "list") {
          loadFavs();
        }
      }
    }

    // ------------------------------------------------------------------------------------------------ list

    function buildList() {
      var card = el("section", "card sv-card");
      card.setAttribute("aria-label", "Список серверов");
      var head = el("div", "sv-head");
      ui.listInfo = el("p", "hint sv-info", "загрузка…");
      head.appendChild(ui.listInfo);
      ui.refreshBtn = button("Обновить список", "alt", onRefresh);
      head.appendChild(ui.refreshBtn);
      card.appendChild(head);
      ui.listResult = el("p", "sv-result");
      ui.listResult.setAttribute("role", "status");
      card.appendChild(ui.listResult);

      var filters = el("div", "sv-filters");
      ui.q = input("search", "Поиск: имя, карта, адрес", 64);
      ui.q.setAttribute("aria-label", "Поиск");
      filters.appendChild(field("Поиск", ui.q, "sv-wide"));
      ui.gameType = select([{ value: "", text: "Любой режим" }]);
      filters.appendChild(field("Режим", ui.gameType));
      ui.region = select([{ value: "", text: "Любой регион" }]);
      filters.appendChild(field("Регион", ui.region));
      ui.minPlayers = select([
        { value: "0", text: "Любое число игроков" },
        { value: "1", text: "от 1 игрока" },
        { value: "2", text: "от 2 игроков" },
        { value: "4", text: "от 4 игроков" },
        { value: "8", text: "от 8 игроков" },
      ]);
      filters.appendChild(field("Игроков", ui.minPlayers));
      ui.sort = select([
        { value: "players", text: "По игрокам (больше сверху)" },
        { value: "name", text: "По названию" },
        { value: "map", text: "По карте" },
        { value: "region", text: "По региону" },
      ]);
      filters.appendChild(field("Порядок", ui.sort));
      card.appendChild(filters);

      var checks = el("div", "sv-checks");
      ui.fBlock = checkbox("Только блок-карты", true);
      ui.fNoPw = checkbox("Без пароля", true);
      ui.fV06 = checkbox("Только протокол 0.6 (бот другого не знает)", true);
      checks.appendChild(ui.fBlock.wrap);
      checks.appendChild(ui.fNoPw.wrap);
      checks.appendChild(ui.fV06.wrap);
      card.appendChild(checks);

      ui.rows = el("ul", "sv-rows");
      card.appendChild(ui.rows);
      ui.more = button("Показать ещё", "alt sv-more", function () {
        shownCount += PAGE;
        renderRows();
      });
      ui.more.hidden = true;
      card.appendChild(ui.more);

      [ui.q, ui.gameType, ui.region, ui.minPlayers, ui.sort].forEach(function (c) {
        c.addEventListener(c === ui.q ? "input" : "change", onFilter);
      });
      [ui.fBlock, ui.fNoPw, ui.fV06].forEach(function (c) {
        c.input.addEventListener("change", onFilter);
      });
      return card;
    }

    function checkbox(text, checked) {
      var wrap = el("label", "sv-check");
      var i = document.createElement("input");
      i.type = "checkbox";
      i.checked = !!checked;
      wrap.appendChild(i);
      wrap.appendChild(el("span", null, text));
      return { wrap: wrap, input: i };
    }

    function onFilter() {
      shownCount = PAGE;
      renderRows();
    }

    function onRefresh() {
      ui.refreshBtn.disabled = true;
      setText(ui.listResult, "Запрос отправлен…", null);
      api("POST", "/api/servers/refresh", {})
        .then(function (res) {
          if (res.status === 202) {
            setText(
              ui.listResult,
              res.data && res.data.asked === false ? "Список свежий (младше минуты): не запрашиваем." : "Запрос отправлен: список обновится в течение полминуты.",
              true,
            );
            setTimeout(loadList, 6000);
            setTimeout(loadList, 15000);
          } else {
            setText(ui.listResult, errorText(res), false);
          }
        })
        .catch(function () {
          setText(ui.listResult, "Нет связи с сайтом.", false);
        })
        .then(function () {
          setTimeout(function () {
            ui.refreshBtn.disabled = false;
          }, 3000);
        });
    }

    function setText(node, text, ok) {
      node.textContent = text || "";
      node.classList.toggle("ok", ok === true);
      node.classList.toggle("bad", ok === false);
    }

    function loadList() {
      return api("GET", "/api/servers")
        .then(function (res) {
          if (res.status !== 200 || !res.data) {
            return;
          }
          var d = res.data;
          list = {
            loaded: true,
            enabled: d.enabled !== false,
            problem: d.problem || null,
            servers: d.servers || [],
            fetched_at: d.fetched_at || 0,
            age_s: d.age_s || 0,
            refresh: d.refresh || null,
            master: d.master || 0,
          };
          fillFilterOptions();
          renderList();
        })
        .catch(function () {
          ui.listInfo.textContent = "Нет связи с сайтом.";
        });
    }

    function fillFilterOptions() {
      var counts = {};
      var regions = {};
      list.servers.forEach(function (s) {
        counts[s.game_type] = (counts[s.game_type] || 0) + 1;
        regions[regionOf(s.location)] = true;
      });
      var types = Object.keys(counts)
        .filter(function (t) {
          return t;
        })
        .sort(function (a, b) {
          return counts[b] - counts[a];
        })
        .slice(0, 40);
      var cur = ui.gameType.value;
      clear(ui.gameType);
      ui.gameType.appendChild(opt("", "Любой режим"));
      types.forEach(function (t) {
        ui.gameType.appendChild(opt(t, t + " (" + counts[t] + ")"));
      });
      ui.gameType.value = cur;
      if (ui.gameType.value !== cur) {
        ui.gameType.value = "";
      }
      var curR = ui.region.value;
      clear(ui.region);
      ui.region.appendChild(opt("", "Любой регион"));
      Object.keys(regions)
        .filter(function (r) {
          return r;
        })
        .sort()
        .forEach(function (r) {
          ui.region.appendChild(opt(r, REGION_LABEL[r] || r));
        });
      ui.region.value = curR;
      if (ui.region.value !== curR) {
        ui.region.value = "";
      }
    }

    function opt(value, text) {
      var o = document.createElement("option");
      o.value = value;
      o.textContent = text;
      return o;
    }

    function renderList() {
      var info;
      if (!list.enabled) {
        info = "Кэш списка не установлен: запустите deploy/install-launcher.sh.";
      } else if (list.problem === "missing") {
        info = "Список ещё не загружался. Нажмите «Обновить список».";
      } else if (list.problem === "invalid") {
        info = "Кэш списка повреждён: нажмите «Обновить список».";
      } else {
        info =
          "Серверов: " + list.servers.length + ". Обновлено " + ago(list.age_s) + " назад (мастер " + list.master + "). Список лежит в кэше: сайт сам в сеть не ходит.";
      }
      if (list.refresh && list.refresh.ok === false) {
        info += " Последнее обновление не удалось: " + (REFRESH_TEXT[list.refresh.reason] || "ошибка") + ".";
      }
      ui.listInfo.textContent = info;
      renderRows();
    }

    function filtered() {
      var q = ui.q.value.trim().toLowerCase();
      var type = ui.gameType.value;
      var region = ui.region.value;
      var minP = parseInt(ui.minPlayers.value, 10) || 0;
      var out = list.servers.filter(function (s) {
        if (ui.fBlock.input.checked && !s.block) {
          return false;
        }
        if (ui.fNoPw.input.checked && s.passworded) {
          return false;
        }
        if (ui.fV06.input.checked && !s.v06) {
          return false;
        }
        if (type && s.game_type !== type) {
          return false;
        }
        if (region && regionOf(s.location) !== region) {
          return false;
        }
        if (s.players < minP) {
          return false;
        }
        if (q && (s.name + " " + s.map + " " + s.address + " " + s.game_type).toLowerCase().indexOf(q) < 0) {
          return false;
        }
        return true;
      });
      var key = ui.sort.value;
      out.sort(function (a, b) {
        if (key === "name") {
          return a.name.localeCompare(b.name);
        }
        if (key === "map") {
          return a.map.localeCompare(b.map) || b.players - a.players;
        }
        if (key === "region") {
          return a.location.localeCompare(b.location) || b.players - a.players;
        }
        return b.players - a.players || a.name.localeCompare(b.name);
      });
      return out;
    }

    function favOf(address) {
      for (var i = 0; i < favs.items.length; i += 1) {
        if (favs.items[i].address === address) {
          return favs.items[i];
        }
      }
      return null;
    }

    function renderRows() {
      clear(ui.rows);
      if (!list.servers.length) {
        ui.more.hidden = true;
        return;
      }
      var rows = filtered();
      if (!rows.length) {
        ui.rows.appendChild(el("li", "sv-empty hint", "Ничего не найдено: ослабьте фильтры."));
        ui.more.hidden = true;
        return;
      }
      rows.slice(0, shownCount).forEach(function (s) {
        ui.rows.appendChild(serverRow(s));
      });
      ui.more.hidden = rows.length <= shownCount;
      if (!ui.more.hidden) {
        ui.more.textContent = "Показать ещё (" + (rows.length - shownCount) + ")";
      }
    }

    function serverRow(s) {
      var li = el("li", "sv-row");
      var top = el("div", "sv-row-top");
      top.appendChild(el("strong", "sv-name", s.name));
      var badges = el("span", "sv-badges");
      if (s.passworded) {
        badges.appendChild(el("span", "sv-badge warn", "пароль"));
      }
      if (!s.v06) {
        badges.appendChild(el("span", "sv-badge warn", "только 0.7"));
      }
      if (s.block) {
        badges.appendChild(el("span", "sv-badge", "блок"));
      }
      var f = favOf(s.address);
      if (f) {
        badges.appendChild(el("span", "sv-badge fav", "в избранном"));
      }
      top.appendChild(badges);
      li.appendChild(top);
      li.appendChild(el("div", "sv-meta", s.address + " · " + (s.map || "—") + " · " + (s.game_type || "—") + " · " + (s.location || "—")));
      li.appendChild(el("div", "sv-players", "Игроков: " + s.players + (s.max_clients ? " / " + s.max_clients : "")));
      var actions = el("div", "sv-actions");
      if (f) {
        actions.appendChild(
          button("Играть здесь", "sv-play", function () {
            hooks.play(s.address);
          }),
        );
      } else {
        actions.appendChild(button("В избранное", "alt", function () {
          openAdd(li, s, false);
        }));
        var play = button("Играть здесь", "sv-play", function () {
          openAdd(li, s, true);
        });
        play.disabled = !s.v06;
        play.title = s.v06 ? "" : "Сервер только на протоколе 0.7: бот его не знает";
        actions.appendChild(play);
      }
      li.appendChild(actions);
      if (addingFor && addingFor.address === s.address) {
        li.appendChild(addForm(s, addingFor.play));
      }
      return li;
    }

    function openAdd(li, s, play) {
      addingFor = { address: s.address, play: play };
      renderRows();
      loadProxies();
    }

    function proxyOptions() {
      var opts = [{ value: "direct", text: "Напрямую" }];
      proxies.items.forEach(function (p) {
        opts.push({ value: "proxy:" + p.name, text: "Через прокси «" + p.name + "»" + (p.usable ? "" : " (не работает)") });
      });
      return opts;
    }

    function addForm(s, play) {
      var form = el("div", "sv-form");
      var nick = input("text", DEFAULT_NICK, 15);
      nick.value = DEFAULT_NICK;
      var conn = select(proxyOptions());
      var notes = input("text", "заметка (необязательно)", 200);
      var consent = checkbox("Администратор сервера разрешает бота (я отвечаю за это разрешение)", false);
      var result = el("p", "sv-result");
      result.setAttribute("role", "status");
      form.appendChild(field("Ник бота", nick));
      form.appendChild(field("Подключение", conn));
      form.appendChild(field("Заметка", notes, "sv-wide"));
      form.appendChild(consent.wrap);
      var buttons = el("div", "sv-actions");
      var ok = button(play ? "Добавить и играть" : "Добавить в избранное", "sv-play", function () {
        if (!consent.input.checked) {
          setText(result, ERROR_TEXT.consent_required, false);
          return;
        }
        ok.disabled = true;
        api("POST", "/api/favourites/add", {
          address: s.address,
          name: s.name,
          nick: nick.value.trim() || DEFAULT_NICK,
          connection: conn.value,
          consent: true,
          notes: notes.value.trim(),
        })
          .then(function (res) {
            if (res.status === 201) {
              addingFor = null;
              return loadFavs().then(function () {
                renderRows();
                if (play) {
                  hooks.play(s.address);
                }
              });
            }
            setText(result, errorText(res), false);
            ok.disabled = false;
          })
          .catch(function () {
            setText(result, "Нет связи с сайтом.", false);
            ok.disabled = false;
          });
      });
      buttons.appendChild(ok);
      buttons.appendChild(
        button("Отмена", "alt", function () {
          addingFor = null;
          renderRows();
        }),
      );
      form.appendChild(buttons);
      form.appendChild(result);
      return form;
    }

    // ------------------------------------------------------------------------------------------------ favourites

    function buildFav() {
      var card = el("section", "card sv-card");
      card.setAttribute("aria-label", "Избранное");
      card.appendChild(el("h3", "sv-h3", "Избранное"));
      ui.favState = el("p", "hint", "загрузка…");
      card.appendChild(ui.favState);
      ui.favResult = el("p", "sv-result");
      ui.favResult.setAttribute("role", "status");
      card.appendChild(ui.favResult);
      ui.favRows = el("ul", "sv-rows");
      card.appendChild(ui.favRows);

      var manual = el("details", "sv-manual");
      manual.appendChild(el("summary", null, "Добавить сервер по адресу"));
      var form = el("div", "sv-form");
      var addr = input("text", "IP:порт, например 45.141.57.35:8308", 64);
      var name = input("text", "название", 64);
      var nick = input("text", DEFAULT_NICK, 15);
      nick.value = DEFAULT_NICK;
      var consent = checkbox("Администратор сервера разрешает бота (я отвечаю за это разрешение)", false);
      var res = el("p", "sv-result");
      res.setAttribute("role", "status");
      form.appendChild(field("Адрес", addr, "sv-wide"));
      form.appendChild(field("Название", name));
      form.appendChild(field("Ник бота", nick));
      form.appendChild(consent.wrap);
      var go = button("Добавить в избранное", "sv-play", function () {
        if (!consent.input.checked) {
          setText(res, ERROR_TEXT.consent_required, false);
          return;
        }
        api("POST", "/api/favourites/add", {
          address: addr.value.trim(),
          name: name.value.trim() || addr.value.trim(),
          nick: nick.value.trim() || DEFAULT_NICK,
          connection: "direct",
          consent: true,
        })
          .then(function (r) {
            if (r.status === 201) {
              setText(res, "Добавлено.", true);
              addr.value = "";
              name.value = "";
              consent.input.checked = false;
              loadFavs();
            } else {
              setText(res, errorText(r), false);
            }
          })
          .catch(function () {
            setText(res, "Нет связи с сайтом.", false);
          });
      });
      form.appendChild(go);
      form.appendChild(res);
      manual.appendChild(form);
      card.appendChild(manual);
      return card;
    }

    function loadFavs() {
      return api("GET", "/api/favourites")
        .then(function (res) {
          if (res.status !== 200 || !res.data) {
            return;
          }
          var next = { loaded: true, enabled: res.data.enabled !== false, error: res.data.error || null, items: res.data.favourites || [] };
          // Re-draw only when something changed (the page polls): a form or a choice the owner is in the middle of must not vanish.
          var sig = JSON.stringify(next);
          if (sig === favSig) {
            return;
          }
          favSig = sig;
          favs = next;
          renderFavs();
          if (list.loaded && panel === "list" && !addingFor) {
            renderRows();
          }
        })
        .catch(function () {
          ui.favState.textContent = "Нет связи с сайтом.";
        });
    }

    function renderFavs() {
      clear(ui.favRows);
      if (!favs.enabled) {
        ui.favState.textContent = "Каталог data/launch не создан на сервере: запустите deploy/install-launcher.sh.";
      } else if (favs.error) {
        ui.favState.textContent = ERROR_TEXT[favs.error] || "Файл избранного не читается.";
      } else if (!favs.items.length) {
        ui.favState.textContent = "Пока пусто. Добавьте сервер из списка или по адресу.";
      } else {
        ui.favState.textContent = "Серверов: " + favs.items.length + ". Запускать бота можно здесь или на вкладке «Бот».";
      }
      favs.items.forEach(function (f) {
        ui.favRows.appendChild(favRow(f));
      });
    }

    function favRow(f) {
      var li = el("li", "sv-row");
      var top = el("div", "sv-row-top");
      top.appendChild(el("strong", "sv-name", f.name));
      var badges = el("span", "sv-badges");
      if (f.blocked) {
        badges.appendChild(el("span", "sv-badge bad", "закрыт после кика/бана"));
      }
      top.appendChild(badges);
      li.appendChild(top);
      li.appendChild(el("div", "sv-meta", f.address + " · ник «" + f.nick + "» · разрешение подтверждено " + when(f.consent_at)));
      if (f.notes) {
        li.appendChild(el("div", "sv-meta", "Заметка: " + f.notes));
      }
      if (f.blocked) {
        li.appendChild(
          el(
            "p",
            "sv-blocked",
            "Бота кикнули или забанили (" + when(f.blocked.at) + "). Он остановился и не пытается обойти бан: ни прокси, ни другой IP не подставляются сами. Запуск закрыт, пока вы не откроете сервер снова.",
          ),
        );
      }
      var conn = select(proxyOptions());
      conn.value = f.connection;
      if (conn.value !== f.connection) {
        // A proxy that is gone from the list: show it as it is stored, never silently as «Напрямую».
        conn.appendChild(opt(f.connection, f.connection + " (профиля нет)"));
        conn.value = f.connection;
      }
      conn.addEventListener("change", function () {
        changeConnection(f, conn);
      });
      li.appendChild(field("Подключение", conn));
      var actions = el("div", "sv-actions");
      var play = button("Играть здесь", "sv-play", function () {
        hooks.play(f.address);
      });
      play.disabled = !!f.blocked;
      actions.appendChild(play);
      if (f.blocked) {
        actions.appendChild(
          button("Открыть снова", "danger", function () {
            reopen(f);
          }),
        );
      }
      actions.appendChild(
        button("Удалить", "alt", function () {
          if (window.confirm("Убрать «" + f.name + "» из избранного?")) {
            mutate("/api/favourites/remove", { address: f.address }, "Удалено.");
          }
        }),
      );
      li.appendChild(actions);
      return li;
    }

    function changeConnection(f, conn) {
      var value = conn.value;
      mutate("/api/favourites/update", { address: f.address, connection: value }, "Подключение изменено.").then(function () {
        if (f.blocked) {
          setText(ui.favResult, "Подключение изменено, но сервер всё ещё закрыт: бан не снимается сменой прокси.", null);
        }
      });
    }

    function reopen(f) {
      var ok = window.confirm(
        "Открыть «" + f.name + "» снова? Бота уже кикали или банили на этом сервере. Вы сами решаете, что администратор по-прежнему согласен.",
      );
      if (ok) {
        mutate("/api/favourites/reopen", { address: f.address, confirm: true }, "Сервер открыт: можно запускать.");
      }
    }

    function mutate(path, body, okText) {
      setText(ui.favResult, "…", null);
      return api("POST", path, body)
        .then(function (res) {
          if (res.status >= 200 && res.status < 300) {
            setText(ui.favResult, okText, true);
          } else {
            setText(ui.favResult, errorText(res), false);
          }
          return loadFavs();
        })
        .catch(function () {
          setText(ui.favResult, "Нет связи с сайтом.", false);
        });
    }

    // ------------------------------------------------------------------------------------------------ proxies

    function buildProxy() {
      var card = el("section", "card sv-card");
      card.setAttribute("aria-label", "Прокси");
      card.appendChild(el("h3", "sv-h3", "Прокси"));
      card.appendChild(
        el(
          "p",
          "hint",
          "Профили хранятся в файлах с правами 0600. Пароль только записывается: сайт его никогда не показывает, пустое поле при правке оставляет прежний. Какой прокси использовать для сервера (или никакого), вы выбираете в избранном; бот сам прокси не меняет.",
        ),
      );
      ui.proxyState = el("p", "hint", "загрузка…");
      card.appendChild(ui.proxyState);
      ui.proxyResult = el("p", "sv-result");
      ui.proxyResult.setAttribute("role", "status");
      card.appendChild(ui.proxyResult);
      ui.proxyRows = el("ul", "sv-rows");
      card.appendChild(ui.proxyRows);

      var form = el("div", "sv-form sv-proxy-form");
      ui.pfTitle = el("h4", "sv-h4", "Новый прокси");
      form.appendChild(ui.pfTitle);
      ui.pfName = input("text", "имя, например hproxy", 64);
      ui.pfHost = input("text", "публичный IP, например 93.184.216.34", 45);
      ui.pfPort = input("number", "1080", 5);
      ui.pfPort.min = "1";
      ui.pfPort.max = "65535";
      ui.pfPort.inputMode = "numeric";
      ui.pfUser = input("text", "логин", 255);
      ui.pfPass = input("password", "пароль", 255);
      ui.pfPass.autocomplete = "new-password";
      ui.pfRelay = select([
        { value: "proxy-host-only", text: "Только хост прокси (обычный)" },
        { value: "public", text: "Публичный ретранслятор на другом хосте" },
      ]);
      ui.pfPick = select([
        { value: "0", text: "Выкл" },
        { value: "2", text: "2 сессии" },
        { value: "3", text: "3 сессии" },
        { value: "4", text: "4 сессии" },
      ]);
      form.appendChild(field("Имя", ui.pfName));
      form.appendChild(field("Адрес (IP)", ui.pfHost));
      form.appendChild(field("Порт", ui.pfPort));
      form.appendChild(field("Логин", ui.pfUser));
      form.appendChild(field("Пароль (только запись)", ui.pfPass));
      form.appendChild(field("Ретранслятор UDP", ui.pfRelay));
      form.appendChild(field("Подбор липкой сессии", ui.pfPick));
      form.appendChild(
        el("p", "hint sv-wide", "Подбор сессии: в логине должно быть «{session}», клиент попробует 2–4 сессии и оставит с лучшим UDP."),
      );
      var buttons = el("div", "sv-actions sv-wide");
      ui.pfSave = button("Сохранить", "sv-play", onSaveProxy);
      ui.pfCancel = button("Отмена правки", "alt", resetProxyForm);
      ui.pfCancel.hidden = true;
      buttons.appendChild(ui.pfSave);
      buttons.appendChild(ui.pfCancel);
      form.appendChild(buttons);
      ui.pfResult = el("p", "sv-result sv-wide");
      ui.pfResult.setAttribute("role", "status");
      form.appendChild(ui.pfResult);
      card.appendChild(form);
      return card;
    }

    function resetProxyForm() {
      editingProxy = null;
      ui.pfTitle.textContent = "Новый прокси";
      ui.pfName.disabled = false;
      [ui.pfName, ui.pfHost, ui.pfPort, ui.pfUser, ui.pfPass].forEach(function (i) {
        i.value = "";
      });
      ui.pfUser.placeholder = "логин";
      ui.pfPass.placeholder = "пароль";
      ui.pfRelay.value = "proxy-host-only";
      ui.pfPick.value = "0";
      ui.pfCancel.hidden = true;
    }

    function onSaveProxy() {
      var body = {
        create: editingProxy === null,
        name: ui.pfName.value.trim(),
        host: ui.pfHost.value.trim(),
        port: parseInt(ui.pfPort.value, 10) || 0,
        user: ui.pfUser.value,
        pass: ui.pfPass.value,
        relay: ui.pfRelay.value,
        session_pick: parseInt(ui.pfPick.value, 10) || 0,
      };
      ui.pfSave.disabled = true;
      api("POST", "/api/proxies/save", body)
        .then(function (res) {
          if (res.status === 200) {
            setText(ui.pfResult, "Сохранено. Нажмите «Проверить» у профиля.", true);
            resetProxyForm();
            return loadProxies();
          }
          setText(ui.pfResult, errorText(res), false);
        })
        .catch(function () {
          setText(ui.pfResult, "Нет связи с сайтом.", false);
        })
        .then(function () {
          ui.pfSave.disabled = false;
          // The password never stays in the page longer than the request.
          ui.pfPass.value = "";
        });
    }

    function loadProxies() {
      return api("GET", "/api/proxies")
        .then(function (res) {
          if (res.status !== 200 || !res.data) {
            return;
          }
          var d = res.data;
          proxies = { loaded: true, enabled: d.enabled !== false, items: d.proxies || [], check: d.check || null, last_check_id: d.last_check_id || null, pending: !!d.pending };
          if (checkingName && proxies.check && proxies.check.proxy === checkingName) {
            checkingName = null;
          }
          renderProxies();
          scheduleCheckPoll();
        })
        .catch(function () {
          ui.proxyState.textContent = "Нет связи с сайтом.";
        });
    }

    function scheduleCheckPoll() {
      if (checkTimer) {
        clearTimeout(checkTimer);
        checkTimer = null;
      }
      if (shown && (checkingName || proxies.pending)) {
        checkTimer = setTimeout(loadProxies, CHECK_POLL_MS);
      }
    }

    function renderProxies() {
      clear(ui.proxyRows);
      if (!proxies.enabled) {
        ui.proxyState.textContent = "Каталог data/launch не создан на сервере: проверка прокси недоступна (deploy/install-launcher.sh).";
      } else if (!proxies.items.length) {
        ui.proxyState.textContent = "Профилей пока нет.";
      } else {
        ui.proxyState.textContent = "Профилей: " + proxies.items.length + ".";
      }
      proxies.items.forEach(function (p) {
        ui.proxyRows.appendChild(proxyRow(p));
      });
    }

    function proxyRow(p) {
      var li = el("li", "sv-row");
      var top = el("div", "sv-row-top");
      top.appendChild(el("strong", "sv-name", p.name));
      var badges = el("span", "sv-badges");
      badges.appendChild(el("span", "sv-badge", p.managed ? "с сайта" : "из файла"));
      if (!p.usable) {
        badges.appendChild(el("span", "sv-badge bad", p.problem === "bad_mode" ? "права файла не 0600" : "файл не читается"));
      }
      top.appendChild(badges);
      li.appendChild(top);
      var meta = [];
      if (p.managed && p.host) {
        meta.push(p.host + ":" + p.port);
      } else {
        meta.push("адрес скрыт (файл сделан вручную)");
      }
      meta.push("ретранслятор: " + (RELAY_LABEL[p.relay] || p.relay));
      if (p.session_pick) {
        meta.push("сессий: " + p.session_pick);
      }
      meta.push(p.has_credentials ? "логин и пароль заданы" : "без пароля");
      li.appendChild(el("div", "sv-meta", meta.join(" · ")));
      var chk = proxies.check && proxies.check.proxy === p.name ? proxies.check : null;
      if (checkingName === p.name && !chk) {
        li.appendChild(el("p", "sv-result", "Проверяем…"));
      } else if (chk) {
        li.appendChild(checkLine(chk));
      }
      var actions = el("div", "sv-actions");
      var check = button("Проверить", "sv-play", function () {
        onCheck(p.name, check);
      });
      check.disabled = !p.usable || !!checkingName || proxies.pending;
      actions.appendChild(check);
      if (p.managed) {
        actions.appendChild(
          button("Изменить", "alt", function () {
            editingProxy = p.name;
            ui.pfTitle.textContent = "Правка «" + p.name + "»";
            ui.pfName.value = p.name;
            ui.pfName.disabled = true;
            ui.pfHost.value = p.host || "";
            ui.pfPort.value = p.port ? String(p.port) : "";
            ui.pfUser.value = "";
            ui.pfPass.value = "";
            ui.pfUser.placeholder = "пусто = оставить прежний";
            ui.pfPass.placeholder = "пусто = оставить прежний";
            ui.pfRelay.value = p.relay;
            ui.pfPick.value = String(p.session_pick || 0);
            ui.pfCancel.hidden = false;
            ui.pfHost.focus();
          }),
        );
        actions.appendChild(
          button("Удалить", "alt", function () {
            if (!window.confirm("Удалить профиль «" + p.name + "»? Файл с паролем будет стёрт.")) {
              return;
            }
            api("POST", "/api/proxies/remove", { name: p.name })
              .then(function (res) {
                setText(ui.proxyResult, res.status === 200 ? "Профиль удалён." : errorText(res), res.status === 200);
                return loadProxies();
              })
              .catch(function () {
                setText(ui.proxyResult, "Нет связи с сайтом.", false);
              });
          }),
        );
      }
      li.appendChild(actions);
      return li;
    }

    function checkLine(c) {
      var parts = [CHECK_TEXT[c.code] || "ошибка " + c.code];
      if (c.ok) {
        if (typeof c.udp_rtt_ms === "number") {
          parts.push("UDP: " + c.udp_rtt_ms + " мс" + (typeof c.probe_replies === "number" ? " (ответили " + c.probe_replies + " из " + c.probe_sent + ")" : ""));
        }
        if (c.relay) {
          parts.push(RELAY_NOTE[c.relay] || c.relay);
        }
      }
      var p = el("p", "sv-result " + (c.ok ? "ok" : "bad"), "Проверка: " + parts.join(" · ") + " (" + when(c.at) + ")");
      return p;
    }

    function onCheck(name, btn) {
      btn.disabled = true;
      api("POST", "/api/proxies/check", { name: name })
        .then(function (res) {
          if (res.status === 202) {
            checkingName = name;
            setText(ui.proxyResult, "Проверка запущена: ответ через несколько секунд.", null);
            return loadProxies();
          }
          setText(ui.proxyResult, errorText(res), false);
          btn.disabled = false;
        })
        .catch(function () {
          setText(ui.proxyResult, "Нет связи с сайтом.", false);
          btn.disabled = false;
        });
    }

    // ------------------------------------------------------------------------------------------------ lifecycle

    function mount(rootEl, apiFn, hooksObj) {
      root = rootEl;
      api = apiFn;
      hooks = hooksObj || { play: function () {} };
      build();
    }

    function onShown() {
      shown = true;
      if (!api) {
        return;
      }
      loadList();
      loadFavs();
      loadProxies();
      if (!favTimer) {
        favTimer = setInterval(function () {
          if (panel === "fav" || panel === "list") {
            loadFavs();
          }
        }, FAV_POLL_MS);
      }
    }

    function onHidden() {
      shown = false;
      if (favTimer) {
        clearInterval(favTimer);
        favTimer = null;
      }
      if (checkTimer) {
        clearTimeout(checkTimer);
        checkTimer = null;
      }
      // The password is never kept in the form while the tab is away.
      if (ui.pfPass) {
        ui.pfPass.value = "";
      }
    }

    return { mount: mount, onShown: onShown, onHidden: onHidden };
  })();

  window.ServersPanel = ServersPanel;
})();
