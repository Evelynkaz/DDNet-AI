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
  // Task 5.13 (D-097): the finishing switch. «цель» is the recommended live A/B. Task 5.16 (D-120): «полный» is the duel 1 on 1 choice (D-116, E-034),
  // not a crowd one; its hint carries the numbers and the caveats.
  var FINISH_LABEL = { off: "выкл", target: "цель", wb: "ВБ", full: "полный" };
  var FINISH_HINT = {
    off: "Дожим выключен: замороженную цель бот отпускает, как раньше.",
    target:
      "Рекомендуется для живой проверки. Бот держит замороженную цель, пока её не удержали, не запечатали или она не погибла. На арене в толпе удержанных блоков на 32% больше, первые заморозки те же. Включайте на одном запуске и сравнивайте с выключенным.",
    wb:
      "Эксперимент для игры на ВБ, не для дуэли (задача 3.18): к «цели» добавляется удержание жертвы в зале ВБ. Роль верхней полки качает замороженную жертву к фриз-стене зала, а замороженную цель не бросает, пока её добивают. На арене удержанных блоков на 3,6 п.п. больше (36,1% → 39,7%), но заранее объявленной планки (+4,0 п.п.) это не берёт; вживую не проверено. Качели работают только в зале ВБ и в роли верхней полки, в остальном это «цель».",
    full:
      "Для дуэли 1 на 1, не для толпы. К «цели» добавляется подтягивание замороженной жертвы к фризу. В арене дуэли (правила раунда F-DDrace, 1800 парных игр): +4,3 ± 2,2 п.п. побед (p 0,0001); на тихой машине +7,5 ± 4,1 п.п. (600 пар, p 0,0005). Оговорки: плечо выбрано после просмотра таблицы; соперник в арене один (live-v2); часть «цель» в арене не моделировалась; вживую не проверено, и счёт за 15 минут этого не покажет. В толпе подтягивание само по себе ничего не добавляло (257 против 259 удержанных блоков, 600 игр): там берите «цель».",
  };
  // «полный» chosen while the duel switch is off: the card says so (a crowd is where it was never shown to help).
  var FINISH_FULL_NOT_DUEL =
    " Сейчас «Без самоубийств (дуэль)» выключено: похоже, это не дуэль 1 на 1.";

  // Task 5.15 (D-103/D-104): the smart wayblock. For every brain: it is the bot's navigation and target choice, not the brain's decision.
  var WB_HINT = {
    off: "Умный ВБ выключен: правила AFK и стороны ВБ прежние.",
    on: "Простаивающего (AFK) игрока бот бьёт, только если он мешает: стоит в удерживаемом зале, рядом или на маршруте. Сторону ВБ выбирает по числу целей, которых можно блокировать. Переходы труб на Copy Love Box безопаснее: после возрождения не бродит, а сразу начинает путь, поиск перехода учитывает других игроков, неподвижного на ступеньке перепрыгивает. Работает со всеми мозгами; рассчитан на Copy Love Box.",
  };
  // Task 5.15 (D-102): the duel switch. On, it is a warning: the bot then loses its own way out of a stuck spot.
  var SELFKILL_HINT = {
    off: "Выключено. Для дуэли 1vs1 в F-DDrace выберите «вкл (дуэль)».",
    on: "Для 1vs1 F-DDrace: любая смерть бота даёт очко сопернику, поэтому бот сам себя не убивает. На обычных серверах не включать: бот лишается выхода из застревания (самоубийство при застревании и в заморозке). Убить бота можно только вручную: кнопкой «Убить» в «Командах» (или строкой /kill на вкладке «Игра»).",
  };

  // Task 3.17 (D-111): the opponent-input predictor (an experiment, hybrid brains only).
  var MODEL_HINT = {
    off: "Предсказатель выключен: в окне лага соперник «держит то, что показывает снапшот».",
    on: "Эксперимент: маленькая сеть предсказывает ввод соперника в окне лага (3.15). В арене помогала на дальних тиках окна, на живых клипах при окне 2 пользы не видно; предохранитель сам отключает её, если она хуже «держит» (карточка «Бот», строка «Предсказатель»). Журнал — data/bot/oppnet-live.jsonl. Вне дуэли 1 на 1 не работает. Выключить на ходу: файл data/bot/window-model.off.",
  };

  // Task 3.20b (D-112): the server's pre-inputs played in the prediction (an experiment, hybrid brains only).
  var PREINPUT_HINT = {
    off: "Выключено: ходы соперника сервер присылает, бот их считает (карточка «Бот», строка «Ходы от сервера»), но в предсказании не использует.",
    on: "Эксперимент: в предсказании соперник ходит так, как сервер заранее прислал его настоящий ввод. Помогает, только если сервер присылает эти ходы заранее (раньше снапшота): при запасе предсказания соперника по умолчанию (10 мс) решение почти ничего не узнаёт заранее. Вживую ход был известен хотя бы на тик вперёд у 7,6% пар «снапшот, соперник» (joniTee, 08.10), 8,0% (GER, 07.10) и ≈ 0,2% (дуэль с человеком: его клиент шлёт с запасом по умолчанию, снапшоты обгоняют сообщения); сколько это даёт в силе, вживую не измерено. Сколько раз это было на самом деле, видно в карточке «Бот» (доля снапшотов, где ход известен хотя бы на тик вперёд). Бот только принимает, серверу ничего не отправляет. Выключить на ходу: файл data/bot/preinput.off.",
  };

  // Task 5.17 (D-125): the hybrid's search threads (docs/research/perf-4.13.md, D-123). The numbers are the live bench on a QUIET machine (candidates scored per
  // decision, median, with `--finish full`); strength per thread was not measured, and under load the helpers compete with the builds and the agents (D-080, D-124).
  var SEARCH_THREADS_CANDIDATES = { "1": "24–25", "2": "29", "3": "38", "4": "44" };
  var SEARCH_THREADS_TAIL =
    " Числа с тихой машины (живой стенд, задача 4.13; медиана, один прогон на значение): кандидатов на решение 24–25 / 29 / 38 / 44 при 1 / 2 / 3 / 4 потоках; p99 времени решения не хуже; каждый помощник занимает около 5% ядра; пул конечен (≈ 55–62 кандидата). Только на тихой машине: под нагрузкой больше потоков отнимает процессор у сборок и может не помочь. Тихая — это нагрузка < 2 и не меньше 4 свободных ядер (perf-4.13.md, раздел 6). Больше кандидатов не значит больше побед: силу от числа потоков вживую и в арене не измеряли.";
  // Shown (and the hint turns into a warning) while the host's load is above the quietness threshold and more than one thread is chosen.
  var SEARCH_THREADS_LOADED =
    " Машина загружена: потоки поиска сверх 1 отнимают процессор у сборок и могут не помочь — выберите 1.";
  function searchThreadsHint(v) {
    var c = SEARCH_THREADS_CANDIDATES[v];
    if (!c) {
      return "";
    }
    var head =
      v === "1"
        ? "Один поток (умолчание бота): около " + c + " кандидатов на решение на тихой машине; помощников нет, процессор не делится."
        : v === "3"
          ? "Три потока: около " + c + " кандидатов на решение на тихой машине. Это ставит «Дуэль» (рекомендация сборщика 4.13 для дуэли на тихой машине)."
          : v === "2"
            ? "Два потока: около " + c + " кандидатов на решение на тихой машине."
            : "Четыре потока: около " + c + " кандидатов на решение на тихой машине.";
    return head + SEARCH_THREADS_TAIL;
  }

  // Task 5.18 (D-129): the 3.23 duel fixes (docs/research/duel-fixes-3.23.md, D-121, E-038), hybrid brains only. A closed list of the three arms of the live protocol
  // (section 7): off, finish, static+finish. `counter` (no-go: it raises timeouts) and `all` are not offered. The numbers are the arena's and the scenarios', not live ones.
  var DUEL_FIXES_LABEL = { off: "выкл", finish: "добивание", "static,finish": "стоячая цель и добивание" };
  var DUEL_FIXES_TAIL =
    " Работает только в распознанной дуэли; вживую не проверено. Силу вживую на сотнях раундов не измерить: сравнивайте плечи по журналу (протокол 3.23, раздел 7). Пресет «Дуэль» ставит «выкл» и остаётся таким, пока вживую не сыграют сравнение плеч.";
  var DUEL_FIXES_HINT = {
    off: "Выключено (умолчание): бот ведёт себя, как до задачи 3.23.",
    finish:
      "Добивание: замороженный соперник лежит вне фриза, а бот не стоит, а действует (подходит, подтягивает; молотом замороженного не бьёт). Заранее объявленные планки взяты: в сценариях добивания удержание 57,7% → 90,4% (+32,7 п.п.), нулевой ввод 34% → 15%; в арене дуэли против «live-v2» не хуже базы (+1,7 ± 2,0 п.п. побед, p 0,13; таймауты те же), силу это не доказывает." +
      DUEL_FIXES_TAIL,
    "static,finish":
      "Добивание плюс стоячая цель: соперник дуэли не отбрасывается как АФК, а против стоящего бот выбирает план, который действует. Стоячая цель — большой эффект в сценарии (ввод «ничего не нажато» 100% → 6,8%; блок 59% → 87% при 6 мкс на ти-тик), но планку по букве не взяла (87% при 6 мкс вместо 90%; в ячейке 2/2 арены −4,5 п.п., в шуме); в арене сочетание вместе −0,7 ± 3,3 п.п. (p 0,77; задним числом, не заранее объявленное плечо). Против активного человека пользы не доказано, нужна при настоящем АФК-партнёре." +
      DUEL_FIXES_TAIL,
  };

  // Task 5.16 (D-120): the «Дуэль» preset. It only fills the form (the owner still presses «Запустить»); every value below says why, with the numbers
  // and the caveats of docs/research/duel-3.19.md (D-116, E-034), preinput.md (D-112) and lag-shave.md (D-115).
  var PRESET_NOTE =
    "Заполняет форму для дуэли 1 на 1 (F-DDrace): мозг, дожим, самоубийства, ходы сервера (выкл), предсказатель, умный ВБ, исправления дуэли (выкл) и потоки поиска (3). Сервер, длительность и спарринг остаются вашими. Запускает только кнопка «Запустить».";
  var PRESET_ITEMS = [
    ["Мозг: Гибрид", "у чистой мухи нет ни дожима, ни ходов сервера."],
    [
      "Дожим: полный",
      "в арене дуэли +4,3 ± 2,2 п.п. побед (1800 пар, p 0,0001), на тихой машине +7,5 ± 4,1 п.п. (600 пар). Выбран после просмотра таблицы, соперник в арене один, вживую не проверено.",
    ],
    [
      "Без самоубийств (дуэль): вкл",
      "в 1vs1 F-DDrace любая смерть бота даёт очко сопернику. Цена: бот теряет самоубийство при застревании; убить его можно кнопкой «Убить».",
    ],
    [
      "Настоящие ходы соперника от сервера: выкл",
      "пользы против соперника с запасом по умолчанию (10 мс) нет: вживую ход известен заранее в ~0,2–8% случаев (дуэль с человеком ≈ 0,2%, joniTee 7,6%, GER 8,0%); включать только для опыта.",
    ],
    [
      "Предсказатель соперника: выкл",
      "в арене он помогал на дальних тиках окна, на живых клипах при окне 2 пользы не видно.",
    ],
    [
      "Умный ВБ: выкл",
      "нужен для толпы на Copy Love Box; в распознанной дуэли бот ВБ не держит и сам.",
    ],
    [
      "Исправления дуэли: выкл",
      "умолчание не меняем, пока вживую не сыграно сравнение плеч «выкл» / «добивание» / «стоячая цель и добивание» (3.23, раздел 7): сценарии и арена показывают, что добивание не вредит, но вживую его не проверяли. Включается отдельным выбором.",
    ],
    [
      "Потоки поиска: 3",
      "на тихой машине кандидатов на решение вживую 24–25 / 29 / 38 / 44 при 1 / 2 / 3 / 4 потоках (4.13; медиана, один прогон на значение); три — рекомендация сборщика для дуэли. Помощник занимает около 5% ядра. Тихая машина — нагрузка < 2 и не меньше 4 свободных ядер (perf-4.13.md, раздел 6). Под нагрузкой потоки отнимают процессор у сборок и могут не помочь: выберите 1. Силу от числа потоков не измеряли.",
    ],
    [
      "Тихая машина",
      "под нагрузкой бот оценивает вдвое меньше вариантов: вживую 14,2 кандидата на решение при нагрузке 18–30 против ≈ 27 в арене при часах тихой машины (вживую на тихой машине ещё не измерено) (3.19, 3.16). Нагрузка должна быть не выше 6 (карточка «Состояние», строки про нагрузку и поиск).",
    ],
  ];

  // Task 5.16: whether the machine is quiet enough, from the host's load average (`host` of `GET /api/bot/status`) and the bot's search of the last 30 s
  // (`search_window` of its STATUS). The thresholds are the owner's: load above 6, or fewer than 20 candidates per decision. A mean over fewer than
  // MIN_DECISIONS decisions is not judged. Pure: the «Бот» card (app.js) and this card use the same function.
  var LOAD_WARN = 6;
  var CANDIDATES_WARN = 20;
  var MIN_DECISIONS = 25;
  // Task 5.17 (D-125): the 20 was calibrated with ONE search thread. With N threads the quiet baseline is higher (4.13, live bench, median of one run per N:
  // 24–25 / 29 / 38 / 44 for N = 1 / 2 / 3 / 4), so the threshold is 20 scaled by that ratio and rounded: 20 / 24 / 31 / 36. Unverified live for N > 1.
  // A thread count that is not 1 to 4 (a hand-started bot) or not reported (an older bot) keeps 20.
  var CANDIDATES_WARN_BY_THREADS = { "1": 20, "2": 24, "3": 31, "4": 36 };

  function candidatesWarn(threads) {
    return isNum(threads) && Object.prototype.hasOwnProperty.call(CANDIDATES_WARN_BY_THREADS, String(threads)) ? CANDIDATES_WARN_BY_THREADS[String(threads)] : CANDIDATES_WARN;
  }

  function num1(x) {
    return x.toFixed(1).replace(".", ",");
  }

  function fmtMs(us) {
    return us >= 20000 ? "от 20 мс" : (us / 1000).toFixed(2).replace(".", ",") + " мс";
  }

  function isNum(x) {
    return typeof x === "number" && isFinite(x);
  }

  // The load with two decimals next to the threshold (6,01 must not read «6,0 выше 6»), one elsewhere.
  function numLoad(x) {
    return Math.abs(x - LOAD_WARN) < 0.5 ? x.toFixed(2).replace(".", ",") : num1(x);
  }

  // `brain` is the bot's brain name (STATUS `brain`): the candidate threshold was calibrated on the hybrid, so another brain's count is shown and not judged.
  // `threads` is the bot's STATUS `search_threads` (the threshold follows it; none or unknown: 20).
  function quietness(host, sw, brain, threads) {
    var warnAt = candidatesWarn(threads);
    var out = { load: "—", search: "—", loadHigh: false, searchLow: false, warn: false, reasons: [] };
    if (host && isNum(host.load1)) {
      out.load = numLoad(host.load1) + " / " + (isNum(host.load5) ? num1(host.load5) : "—") + " / " + (isNum(host.load15) ? num1(host.load15) : "—") + " (1 / 5 / 15 мин)";
      if (isNum(host.cpus) && host.cpus > 0) {
        out.load += ", ядер: " + host.cpus;
      }
      if (host.load1 > LOAD_WARN) {
        out.loadHigh = true;
        out.reasons.push("нагрузка " + numLoad(host.load1) + " выше " + LOAD_WARN);
      }
    }
    if (sw && typeof sw === "object") {
      if (!isNum(sw.decisions) || sw.decisions <= 0 || !isNum(sw.candidates_mean)) {
        out.search = "нет решений с поиском за " + (isNum(sw.window_s) ? sw.window_s : 30) + " с";
      } else {
        out.search = num1(sw.candidates_mean) + " кандидата на решение";
        if (isNum(sw.brain_p90_us)) {
          out.search += " · p90 решения " + fmtMs(sw.brain_p90_us);
        }
        out.search += " · решений: " + sw.decisions + " за " + (isNum(sw.window_s) ? sw.window_s : 30) + " с";
        // The thread count next to the candidates: the same number means something else with another count (known only for a hybrid; the fly has no search).
        if (isNum(threads) && threads >= 1 && typeof brain === "string" && brain.indexOf("hybrid") === 0) {
          out.search += " · потоков поиска: " + threads;
        }
        if (typeof brain !== "string" || brain.indexOf("hybrid") !== 0) {
          out.search += " (не гибрид: порог кандидатов к этому мозгу не применяется)";
        } else if (sw.decisions < MIN_DECISIONS) {
          out.search += " (мало, не оцениваем)";
        } else if (sw.candidates_mean < warnAt) {
          out.searchLow = true;
          out.reasons.push("кандидатов на решение " + num1(sw.candidates_mean) + " меньше " + warnAt);
        }
      }
    }
    out.warn = out.loadHigh || out.searchLow;
    return out;
  }

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
    server_not_allowed: "Этого сервера нет ни в избранном, ни в списке разрешённых.",
    server_not_ready: "Сервер в списке, но владелец ещё не открыл его (ready).",
    favourites_invalid: "Файл избранного повреждён: избранные серверы недоступны, пока он не исправлен.",
    favourites_unreadable: "Файл избранного не читается: избранные серверы недоступны.",
    bad_address: "Адрес избранного сервера не прошёл проверку (нужен публичный IP:порт).",
    bad_nick: "Ник избранного сервера не прошёл проверку.",
    bad_connection: "Подключение избранного сервера не прошло проверку.",
    consent_required: "У избранного сервера нет подтверждения администратора.",
    server_ambiguous: "В списке серверов противоречивые записи для этого адреса.",
    server_bad_entry: "Запись сервера в списке разрешённых некорректна.",
    sparring_local_only: "Спарринг-боты бывают только на локальном сервере.",
    bundle_missing: "Файл мухи (bundle) не найден.",
    finish_hybrid_only: "Дожим бывает только у гибридных мозгов: для «Мухи» он выключен.",
    window_model_missing: "Файл предсказателя соперника (data/bot/models/opp-m1.oppnet) не найден: положите его туда.",
    window_model_hybrid_only: "Предсказатель соперника бывает только у гибридных мозгов: для «Мухи» его нет.",
    window_model_bad_path: "Путь к файлу предсказателя недопустим.",
    preinput_hybrid_only: "Настоящие ходы соперника от сервера бывают только у гибридных мозгов: для «Мухи» их нет.",
    search_threads_hybrid_only: "Потоки поиска бывают только у гибридных мозгов: «Муха» не ищет, для неё остаётся один.",
    duel_fixes_hybrid_only: "Исправления дуэли бывают только у гибридных мозгов: для «Мухи» их нет.",
    bundle_bad_path: "Путь к bundle в конфиге недопустим.",
    config_bad: "Конфиг запуска не читается.",
    config_untrusted: "Конфиг запуска доступен на запись не только root: отказ.",
    blocked_after_ban:
      "Этот сервер закрыт после кика или бана (все порты на этом IP): запуск возможен, только когда вы сами откроете его снова (вкладка «Серверы», «Избранное», «Открыть снова»; запись списка разрешённых — правкой файла). Прокси и IP бот сам не меняет.",
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
    server_not_allowed: "Этого сервера нет ни в избранном, ни в списке разрешённых.",
    blocked_after_ban:
      "Сервер закрыт после кика или бана: откройте его снова на вкладке «Серверы» (кнопка «Открыть снова»).",
    sparring_local_only: "Спарринг-боты бывают только на локальном сервере.",
    bundle_missing: "Файл мухи (bundle) не найден.",
    finish_hybrid_only: "Дожим бывает только у гибридных мозгов: для «Мухи» он выключен.",
    window_model_missing: "Файл предсказателя соперника (data/bot/models/opp-m1.oppnet) не найден: положите его туда.",
    window_model_hybrid_only: "Предсказатель соперника бывает только у гибридных мозгов: для «Мухи» его нет.",
    window_model_bad_path: "Путь к файлу предсказателя недопустим.",
    preinput_hybrid_only: "Настоящие ходы соперника от сервера бывают только у гибридных мозгов: для «Мухи» их нет.",
    search_threads_hybrid_only: "Потоки поиска бывают только у гибридных мозгов: «Муха» не ищет, для неё остаётся один.",
    duel_fixes_hybrid_only: "Исправления дуэли бывают только у гибридных мозгов: для «Мухи» их нет.",
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

  // What the last answer said about each choice (`kind`: local, allowlist or favourite; a favourite's name and whether the bot's ban
  // memory has it closed).
  var serverMeta = {};

  function serverLabel(id) {
    if (id === "local") {
      return "Локальный сервер";
    }
    var m = serverMeta[id];
    if (m && m.kind === "favourite") {
      return "★ " + m.name + " (" + id + ")" + (m.blocked ? " — закрыт после кика/бана" : "");
    }
    return "Сервер " + id;
  }

  var LaunchCard = (function () {
    var api = null;
    var root = null;
    var ui = {};
    var shown = false;
    var timer = null;
    var busy = false;
    var fastUntil = 0;
    var wanted = null; // a server the «Серверы» tab asked for («Играть здесь»), chosen when the choices have it
    var info = null; // the last GET /api/bot/launch
    var bridge = null; // the last GET /api/bot/status
    var pressed = null; // the button whose request is in flight (it shows a spinner)

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

      // Task 5.16: the «Дуэль» preset (it only fills the form below) with the reason for each value, and the machine's quietness.
      var preset = el("div", "lc-preset");
      ui.preset = el("button", "lc-preset-btn alt", "Дуэль");
      ui.preset.type = "button";
      ui.preset.setAttribute("aria-pressed", "false");
      preset.appendChild(ui.preset);
      preset.appendChild(el("p", "hint lc-preset-note", PRESET_NOTE));
      var more = el("details", "lc-preset-info");
      more.appendChild(el("summary", null, "Что ставит «Дуэль» и почему"));
      var list = el("ul", "lc-preset-list");
      PRESET_ITEMS.forEach(function (item) {
        var li = el("li");
        li.appendChild(el("strong", null, item[0] + ". "));
        li.appendChild(document.createTextNode(item[1]));
        list.appendChild(li);
      });
      more.appendChild(list);
      preset.appendChild(more);
      ui.presetDone = el("p", "hint lc-preset-done");
      ui.presetDone.setAttribute("role", "status");
      preset.appendChild(ui.presetDone);
      ui.quiet = el("p", "hint lc-quiet");
      ui.quiet.setAttribute("role", "status");
      preset.appendChild(ui.quiet);
      card.appendChild(preset);

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
      // The finishing switch (task 5.13): off by default; the fly brain has none (the helper refuses it), so it is hidden for the fly. «полный» is the duel's (task 5.16).
      ui.finish = select([
        { value: "off", text: "выкл" },
        { value: "target", text: "цель (рекомендуется)" },
        { value: "wb", text: "ВБ (эксперимент)" },
        { value: "full", text: "полный (только дуэль 1 на 1)" },
      ]);
      // Task 5.15: two more switches, both off by default and sent only when on. Neither is brain-specific, so both stay shown for the fly.
      ui.wbSmart = select([
        { value: "off", text: "выкл" },
        { value: "on", text: "вкл" },
      ]);
      ui.noSelfkill = select([
        { value: "off", text: "выкл" },
        { value: "on", text: "вкл (дуэль)" },
      ]);
      // Task 3.17: the predictor of the opponent's input, off by default and sent only when on; the hybrid brains only (the control is hidden for the fly).
      ui.windowModel = select([
        { value: "off", text: "выкл" },
        { value: "on", text: "вкл (эксперимент)" },
      ]);
      // Task 3.20b: the server's real opponent inputs, off by default and sent only when on; the hybrid brains only (the control is hidden for the fly).
      ui.preinput = select([
        { value: "off", text: "выкл" },
        { value: "on", text: "вкл (эксперимент)" },
      ]);
      // Task 5.17: the hybrid's search threads, 1 to 4; 1 is the bot's default and is not sent (the absence of the field); the pure fly does not search (hidden).
      ui.searchThreads = select([
        { value: "1", text: "1 (по умолчанию)" },
        { value: "2", text: "2" },
        { value: "3", text: "3" },
        { value: "4", text: "4" },
      ]);
      // Task 5.18: the 3.23 duel fixes, off by default and sent only when not off; a closed list of three; the hybrid brains only (hidden for the fly).
      ui.duelFixes = select([
        { value: "off", text: "выкл (по умолчанию)" },
        { value: "finish", text: "добивание" },
        { value: "static,finish", text: "стоячая цель и добивание" },
      ]);
      var form = el("div", "lc-form");
      form.appendChild(field("Сервер", ui.server));
      form.appendChild(field("Мозг", ui.brain));
      // The hybrid's opponent model (D-090); the fly brain alone has none, so the toggle is only shown for the hybrid brains.
      ui.mirrorField = field("Предсказание соперника", ui.mirror);
      form.appendChild(ui.mirrorField);
      ui.finishField = field("Дожим", ui.finish);
      form.appendChild(ui.finishField);
      form.appendChild(field("Умный ВБ", ui.wbSmart));
      form.appendChild(field("Без самоубийств (дуэль)", ui.noSelfkill));
      ui.windowModelField = field("Предсказатель соперника (эксперимент)", ui.windowModel);
      form.appendChild(ui.windowModelField);
      ui.preinputField = field("Настоящие ходы соперника от сервера (эксперимент)", ui.preinput);
      form.appendChild(ui.preinputField);
      ui.searchThreadsField = field("Потоки поиска", ui.searchThreads);
      form.appendChild(ui.searchThreadsField);
      ui.duelFixesField = field("Исправления дуэли", ui.duelFixes);
      form.appendChild(ui.duelFixesField);
      form.appendChild(field("Длительность", ui.duration));
      ui.sparringField = field("Спарринг (только локальный сервер)", ui.sparring);
      form.appendChild(ui.sparringField);
      card.appendChild(form);
      ui.finishHint = el("p", "hint lc-finish-hint");
      card.appendChild(ui.finishHint);
      ui.wbHint = el("p", "hint lc-opt-hint lc-wb-hint");
      card.appendChild(ui.wbHint);
      ui.selfkillHint = el("p", "hint lc-opt-hint lc-selfkill-hint");
      card.appendChild(ui.selfkillHint);
      ui.modelHint = el("p", "hint lc-opt-hint lc-model-hint");
      card.appendChild(ui.modelHint);
      ui.preinputHint = el("p", "hint lc-opt-hint lc-preinput-hint");
      card.appendChild(ui.preinputHint);
      ui.searchThreadsHint = el("p", "hint lc-opt-hint lc-search-threads-hint");
      card.appendChild(ui.searchThreadsHint);
      ui.duelFixesHint = el("p", "hint lc-opt-hint lc-duel-fixes-hint");
      card.appendChild(ui.duelFixesHint);
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
          "Бот сам не пишет в игровой чат (строку набираете вы, на вкладке «Игра») и не обходит кики и баны. Запуск проверяет отдельная root-программа: сайт только присылает запрос.",
        ),
      );
      root.appendChild(card);

      ui.server.addEventListener("change", function () {
        syncSparring();
        render();
      });
      ui.brain.addEventListener("change", syncMirror);
      ui.finish.addEventListener("change", syncMirror);
      ui.wbSmart.addEventListener("change", syncOptions);
      ui.noSelfkill.addEventListener("change", syncOptions);
      ui.windowModel.addEventListener("change", syncMirror);
      ui.preinput.addEventListener("change", syncMirror);
      ui.searchThreads.addEventListener("change", syncMirror);
      ui.duelFixes.addEventListener("change", syncMirror);
      // A change by hand takes back "the form is filled for the duel"; the preset button is lit only while the form IS the preset.
      [ui.brain, ui.finish, ui.wbSmart, ui.noSelfkill, ui.windowModel, ui.preinput, ui.searchThreads, ui.duelFixes].forEach(function (c) {
        c.addEventListener("change", function () {
          ui.presetDone.textContent = "";
          syncPreset();
        });
      });
      ui.preset.addEventListener("click", applyPreset);
      syncMirror();
      syncOptions();
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
      var fly = ui.brain.value === "fly";
      ui.mirrorField.hidden = fly;
      // Finishing is for the hybrid brains only: the fly's choice is always «выкл» and the control is not shown.
      ui.finishField.hidden = fly;
      ui.finishHint.hidden = fly;
      syncFinishHint();
      // The predictor lives in the hybrid's lag window: not offered to the pure fly.
      ui.windowModelField.hidden = fly;
      ui.modelHint.hidden = fly;
      var present = !info || info.window_model_present !== false;
      ui.windowModel.options[1].disabled = !present;
      if (!present && ui.windowModel.value === "on") {
        ui.windowModel.value = "off";
      }
      ui.modelHint.textContent = present
        ? MODEL_HINT[ui.windowModel.value] || ""
        : "Файл предсказателя не найден: положите его в data/bot/models/opp-m1.oppnet (в git его нет).";
      // The pre-inputs sit in the hybrid's prediction too: not offered to the pure fly.
      ui.preinputField.hidden = fly;
      ui.preinputHint.hidden = fly;
      ui.preinputHint.textContent = PREINPUT_HINT[ui.preinput.value] || "";
      // The search threads belong to the hybrid's search: the pure fly does not search, so the choice is not offered to it.
      ui.searchThreadsField.hidden = fly;
      ui.searchThreadsHint.hidden = fly;
      syncThreadsHint();
      // The duel fixes are the hybrid's too: the pure fly has none, so the choice is not offered to it.
      ui.duelFixesField.hidden = fly;
      ui.duelFixesHint.hidden = fly;
      ui.duelFixesHint.textContent = Object.prototype.hasOwnProperty.call(DUEL_FIXES_HINT, ui.duelFixes.value) ? DUEL_FIXES_HINT[ui.duelFixes.value] : "";
    }

    // The hint of the chosen thread count; while the host's load is above the threshold and more than one thread is chosen it adds the warning (the choice is
    // never blocked: the owner may know better, and «Дуэль» still sets 3).
    function syncThreadsHint() {
      var loaded = !!(bridge && bridge.host && quietness(bridge.host, null, null).loadHigh) && ui.searchThreads.value !== "1";
      ui.searchThreadsHint.textContent = searchThreadsHint(ui.searchThreads.value) + (loaded ? SEARCH_THREADS_LOADED : "");
      ui.searchThreadsHint.classList.toggle("warn", loaded);
    }

    // «полный» is the duel's choice: with the duel switch off the hint says it does not look like a duel (and turns into a warning).
    function syncFinishHint() {
      var full = ui.finish.value === "full";
      var odd = full && ui.noSelfkill.value !== "on";
      ui.finishHint.textContent = (FINISH_HINT[ui.finish.value] || "") + (odd ? FINISH_FULL_NOT_DUEL : "");
      ui.finishHint.classList.toggle("lc-finish-warn", odd);
    }

    // The values the «Дуэль» preset sets, as the form's own words.
    var PRESET_VALUES = [
      ["brain", "hybrid"],
      ["finish", "full"],
      ["noSelfkill", "on"],
      ["preinput", "off"],
      ["windowModel", "off"],
      ["wbSmart", "off"],
      ["searchThreads", "3"],
      ["duelFixes", "off"],
    ];

    function presetIsSet() {
      return PRESET_VALUES.every(function (kv) {
        return ui[kv[0]].value === kv[1];
      });
    }

    function syncPreset() {
      var on = presetIsSet();
      ui.preset.classList.toggle("current", on);
      ui.preset.setAttribute("aria-pressed", on ? "true" : "false");
    }

    // Task 5.16: fills the form for the duel and nothing else: no request is sent, the owner still presses «Запустить».
    function applyPreset() {
      PRESET_VALUES.forEach(function (kv) {
        ui[kv[0]].value = kv[1];
      });
      syncMirror();
      syncOptions();
      syncPreset();
      ui.presetDone.textContent = "Форма заполнена для дуэли. Проверьте и нажмите «Запустить»: само ничего не запускается.";
    }

    // One line under the preset: the load of the machine (and the bot's search while it plays), warning when it is not quiet.
    function renderQuiet() {
      var host = bridge ? bridge.host : null;
      var live = !!(bridge && bridge.live && bridge.source === "live");
      var q = quietness(
        host,
        live && bridge.status ? bridge.status.search_window : null,
        live && bridge.status ? bridge.status.brain : null,
        live && bridge.status ? bridge.status.search_threads : null,
      );
      var text;
      if (!host || q.load === "—") {
        text = "Нагрузку машины сайт сейчас прочитать не может.";
      } else {
        text = "Машина сейчас: нагрузка " + q.load + ".";
        if (live && q.search !== "—") {
          text += " Поиск бота: " + q.search + ".";
        }
        text += q.warn ? " Внимание: машина загружена — бот думает хуже (" + q.reasons.join("; ") + ")." : " Порог тишины: нагрузка не выше " + LOAD_WARN + ".";
      }
      ui.quiet.textContent = text;
      ui.quiet.classList.toggle("warn", q.warn);
      syncThreadsHint();
    }

    // The hints of the two switches that every brain has; the duel switch turns its hint into a warning while it is on.
    function syncOptions() {
      ui.wbHint.textContent = WB_HINT[ui.wbSmart.value] || "";
      var duel = ui.noSelfkill.value === "on";
      ui.selfkillHint.textContent = SELFKILL_HINT[duel ? "on" : "off"];
      ui.selfkillHint.classList.toggle("warn", duel);
      syncFinishHint();
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
      serverMeta = {};
      (i.servers || []).forEach(function (s) {
        serverMeta[s.id] = s;
      });
      clear(ui.server);
      (i.servers || []).forEach(function (s) {
        var opt = document.createElement("option");
        opt.value = s.id;
        opt.textContent = serverLabel(s.id);
        ui.server.appendChild(opt);
      });
      if (wanted !== null && serverMeta[wanted]) {
        current = wanted;
        wanted = null;
      }
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
      if (status.finish && status.finish !== "off") {
        parts.push("дожим: " + (FINISH_LABEL[status.finish] || status.finish));
      }
      if (status.wb_smart === "on") {
        parts.push("умный ВБ");
      }
      if (status.no_selfkill === true) {
        parts.push("без самоубийств");
      }
      if (status.window_model === true) {
        parts.push("предсказатель соперника");
      }
      if (status.preinput === true) {
        parts.push("ходы соперника от сервера");
      }
      if (typeof status.search_threads === "number" && status.search_threads > 1) {
        parts.push("потоки поиска: " + status.search_threads);
      }
      if (typeof status.duel_fixes === "string" && Object.prototype.hasOwnProperty.call(DUEL_FIXES_LABEL, status.duel_fixes) && status.duel_fixes !== "off") {
        parts.push("исправления дуэли: " + DUEL_FIXES_LABEL[status.duel_fixes]);
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
      var chosen = serverMeta[ui.server.value];
      var closed = !!(chosen && chosen.kind === "favourite" && chosen.blocked);
      if (closed && !live && !info.pending && !detail) {
        detail =
          "Выбранный сервер закрыт после кика или бана. Бот не пытается обойти бан: откройте сервер снова на вкладке «Серверы» (Избранное, «Открыть снова»).";
      }
      if (info.favourites_error && !detail) {
        detail = REASON_TEXT[info.favourites_error] || "";
      }
      ui.dot.className = "dot " + dot;
      ui.stateText.textContent = text;
      ui.detail.textContent = detail;
      ui.start.disabled = busy || !enabled || info.pending || live || closed;
      ui.stop.disabled = busy || !enabled || info.pending;
      // Task 5.14: the button that was pressed shows a spinner while its request is in flight.
      ui.start.classList.toggle("is-loading", busy && ui.start === pressed);
      ui.stop.classList.toggle("is-loading", busy && ui.stop === pressed);
      ui.watch.hidden = !live;
      [ui.server, ui.brain, ui.duration, ui.mirror, ui.finish, ui.wbSmart, ui.noSelfkill, ui.windowModel, ui.preinput, ui.searchThreads, ui.duelFixes].forEach(function (c) {
        c.disabled = busy || !enabled;
      });
      ui.preset.disabled = busy || !enabled;
      syncPreset();
      renderQuiet();
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

    function send(body, label, button) {
      busy = true;
      pressed = button || null;
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
        // Only a mode that is on is sent: «off» is the absence of the field, so an older (or rolled-back) helper, whose strict request format
        // does not know `finish`, still takes every default start.
        if (ui.finish.value === "target" || ui.finish.value === "wb" || ui.finish.value === "full") {
          body.finish = ui.finish.value;
        }
      }
      // Task 5.15: only a switch that is on is sent («off» is the absence of the field), for every brain including the pure fly.
      if (ui.wbSmart.value === "on") {
        body.wb_smart = "on";
      }
      if (ui.noSelfkill.value === "on") {
        body.no_selfkill = true;
      }
      // Task 3.17: the predictor is sent only when on and never for the pure fly (the helper refuses that anyway).
      if (ui.windowModel.value === "on" && ui.brain.value !== "fly") {
        body.window_model = true;
      }
      // Task 3.20b: the same for the server's pre-inputs: sent only when on, never for the pure fly (the helper refuses that anyway).
      if (ui.preinput.value === "on" && ui.brain.value !== "fly") {
        body.preinput = true;
      }
      // Task 5.17: the search threads are sent only above the default (1 is the absence of the field, so an older helper still takes every default start) and never for
      // the pure fly (the helper refuses that anyway). The value is a JSON integer, 2 to 4, from the select's own closed list.
      var threads = parseInt(ui.searchThreads.value, 10);
      if (ui.brain.value !== "fly" && threads >= 2 && threads <= 4) {
        body.search_threads = threads;
      }
      // Task 5.18: the duel fixes are sent only when not off (the absence of the field is off, so an older helper still takes every default start), never for the pure fly,
      // and only one of the select's own two words.
      if (ui.brain.value !== "fly" && (ui.duelFixes.value === "finish" || ui.duelFixes.value === "static,finish")) {
        body.duel_fixes = ui.duelFixes.value;
      }
      if (body.server !== "local") {
        var ok = window.confirm(
          "Запустить бота на сервере " + serverLabel(body.server) + "? Администратор должен разрешать бота. Бот не пишет в чат и не обходит кики и баны (ни прокси, ни IP сам не меняет).",
        );
        if (!ok) {
          return;
        }
      }
      send(body, "Запуск", ui.start);
    }

    function onStop() {
      send({ action: "stop" }, "Остановка", ui.stop);
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

    // Task 5.12: «Играть здесь» on the «Серверы» tab: choose this server in the card (the card still asks to be started).
    function preselect(address) {
      wanted = typeof address === "string" ? address : null;
      if (info) {
        applyChoices(info);
        render();
      }
    }

    function onHidden() {
      shown = false;
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
    }

    return { mount: mount, onShown: onShown, onHidden: onHidden, preselect: preselect, quietness: quietness };
  })();

  window.LaunchCard = LaunchCard;
})();
