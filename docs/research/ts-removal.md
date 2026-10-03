# Удаление унаследованного Electron/TS-рантайма (задача 5.4): инвентаризация

Состояние «до»: коммит `0311695` (последний коммит с полным TS-деревом; он остаётся в истории, `git show 0311695:<путь>`
открывает любой удалённый файл). Правило владельца и `CLAUDE.md`: старый TS-код удаляется, когда Rust-замена прошла
паритет; генератор эталонных трасс живёт до конца проекта. Модели в git не хранятся.

## 1. Как определено, что нужно оставить

1. Статический обход импортов (скрипт: разбор `import … from`, `import()`, `require`, без `import type`) от всех файлов
   `tools/ts-trace/*.mjs` и `tools/jsmath-oracle/*.mjs` — получается замкнутое множество из **40 TS-файлов** (17 121
   строка из 23 196 строк TS в `src/`). Файлы, до которых ведёт только `import type` (`src/nn/mlp.ts`), Node при type
   stripping не загружает, их в множество не включали.
2. Поиск потребителей вне `tools/`: `grep` по `crates/`, `.github/`, `deploy/`, `configs/`, `Cargo.toml`, `deny.toml` на
   пути `src/*.ts`, `app/`, `start.mjs`, `run.sh`, `opponent.json`, `package.json`, `assets/`, `manifests/`.
   Результат: ни один `Cargo.toml`, `build.rs`, `include_str!/include_bytes!`, тест или CI-шаг **не читает** ни один из
   удаляемых файлов (`ci.yml` не содержит ни `node`, ни `npm`). Упоминания в `crates/*/src/*.rs` — только комментарии-
   цитаты вида `src/plan/planner.ts:1044` (происхождение логики), они не исполняются.
3. Проверка опытом: удалены все файлы вне множества, все генераторы запущены заново (раздел 5) — байтовое совпадение.

## 2. Верхний уровень репозитория

| Элемент | Что это | Кто использовал до 5.4 | Решение |
|---|---|---|---|
| `app/` (74 файла, 488 КБ: `main.js`, `boot.js`, `preload.js`, `lib/*` — supervisor, botProcess, settings, zip…; `ui/`; `icons/` с Lucide) | Electron-окно старого бота | `tools/packageApp.mjs`, `README*`, `docs/ORIGINAL.md`. Rust, CI, deploy — нет | **удалён** (заменён веб-интерфейсом `ddai-web`, фаза 5) |
| `tools/packageApp.mjs` | упаковка Windows-приложения (Electron) | `app/package.json` | **удалён** |
| `start.mjs`, `run.sh` | лаунчеры TS-бота (проверка Node, `settings.json`, выбор сервера) | `README*`, `docs/ORIGINAL.md`, `docs/formats.md` (как ссылка на формат настроек) | **удалены** |
| `package.json`, `package-lock.json` (корень) | зависимости TS-бота: `ink`, `react`, `teeworlds`, TS-типы | `start.mjs`, `bot.ts` | **удалены**; нужная одна зависимость `teeworlds@2.6.1` перенесена в `tools/ts-reference/package.json` + свой lock (тот же `integrity`) |
| `opponent.json` (1,3 МБ) | MLP-модель оппонента старого бота (`shape 224→192→96→10`) | только `src/bot/dummyWorker.ts`/`src/nn/mlp.ts` (удаляются). В Rust не читается (поиск `opponent.json` в `crates/` пуст; `OpponentModel::Policy/Learned` в `ddai-planner` не поддержаны намеренно) | **удалён** (по требованию владельца: моделей в git нет; `tools/ci/no-weights.sh` его не ловил — 1,3 МБ < порога 2 МБ и имя не из списка, — поэтому `opponent.json` добавлен в запрещённые имена скрипта (и случай в `no-weights-test.sh`) и в `.gitignore`; старые записи `policy.json`/`value*.json` сохранены) |
| `src/` (64 файла, 1,09 МБ) | TS-бот: ядро физики, планировщик, навигация, UI | `tools/ts-trace`, `tools/jsmath-oracle` (подмножество); остальное — никто | **40 файлов перенесены** в `tools/ts-reference/src/`, **24 удалены** (раздел 3) |
| `assets/` (14 файлов, 3,2 МБ) | скриншоты Electron-окна (RU/EN), `logo.png`, `social.png` | `README*` | скриншоты и `social.png` **удалены**; `assets/logo.png` **оставлен** (шапка README) |
| `manifests/connectome.toml` | закреплённый список файлов коннектома | **Rust**: `ddai-connectome fetch --manifest manifests/connectome.toml` (README крейта, `manifest.rs`, `fetch.rs`) | **оставлен** (не наследие) |
| `configs/`, `deploy/`, `docs/`, `crates/`, `tools/{ci,ddnet-oracle,ddnet-server,e2e,e005,jsmath-oracle,ts-trace,…}` | Rust-проект | — | не трогались (кроме путей в доках и `tools/{ts-trace,jsmath-oracle}`) |
| `runs/` | каталог данных старого бота | не отслеживается (в `.gitignore`) | не в git; запись в `.gitignore` оставлена |
| `README.md`, `README.en.md` | описывали Electron/TS-версию | — | **переписаны** под Rust (бот, веб, обучение) |
| `LICENSE`, `NOTICE` | GPL-3.0 и уведомления | требование лицензий | **оставлены**; в `NOTICE` обновлены пути (`tools/ts-reference/…`) и убраны утверждения про удалённое (Electron/ink/react, значки Lucide, графика окна) |
| `.github/ISSUE_TEMPLATE/bug.yml` | форма ошибки: «значок в трее → собрать отчёт» (Electron) | GitHub | переписана без Electron |

Уведомления лицензий: `NOTICE` (7(b), Wranked1), zlib-уведомление DDNet для `src/core` — сохранены; `src/bot/wavpack/LICENSE*`
(BSD, audiojs/Bryant) и `app/icons/lucide/LICENSE` (ISC) уходили вместе с кодом, который они описывали (в истории
остаются). Из Rust-кода их никто не использует (`grep -ri wavpack\|lucide crates/` пуст).

## 3. `src/`: что куда

**Перенесено** (`git mv src/<p> tools/ts-reference/src/<p>`, содержимое побайтно прежнее) — 40 файлов:

- `core/`: `characterCore`, `collision`, `projectile`, `tuning`, `types`, `vmath`, `world` — TS-мир (`ddai-tsworld` его порт);
- `map/`: `datafile`, `loadMap`;
- `plan/`: `planner`, `route`, `seal`, `shield`, `throwLines`, `memory`, `livePlan`;
- `env/`: `action`, `obs`, `scripted`;
- `nn/`: `rng`, `gru`, `params`;
- `bot/`: `wayblock`, `crossing`, `navigate`, `bot` (методы `wbSpot`/`steerFollow` вызывает `gen-nav-dump.mjs`),
  `liveWorld`, `opponentProfile`, `netPatch`, `autoChat`, `cpuLoad`, `ownerOrders`, `serverPick`;
- `demo/`: `reckoning`, `snapshot`; `watch/`: `incidents`, `recording`; `train/humanImitate`; `i18n.ts`, `i18n-en.ts`.

Файлы `demo/`, `watch/`, `train/`, `i18n*` и часть `bot/` в дампах не участвуют, но **статически импортируются**
файлом `src/bot/bot.ts`, а он нужен `gen-nav-dump.mjs` (метод `DdnetBot.prototype.wbSpot`, `steerFollow`) — убрать их без
правки TS нельзя, а правка эталона запрещена (смысл паритета — «настоящий, неизменённый TS»).

**Удалено** (24 файла): `bot/main.ts` (точка входа), `bot/ui.ts`, `console.ts`, `terminalSafe.ts`, `mascot.ts`,
`autoUpdate.ts`, `dummyThread.ts`, `dummyWorker.ts`, `web.ts`, `webAssets.ts`, `webDraw.ts`, `webMap.ts`, `webPage.ts`,
`webView.ts`, `page/*` (HTML/CSS/JS старого веб-окна, `whatsnew.json`), `wavpack/*` (вендоренный декодер звука + лицензии),
`nn/mlp.ts` (MLP оппонента).

## 4. Транзитивные импорты генераторов паритета

Прямые импорты (все из `src/`, относительно `tools/ts-reference/`):

| Генератор | Прямые импорты |
|---|---|
| `lib.mjs` (общий) | `core/world`, `core/collision`, `core/types`, `map/loadMap` |
| `gen-planner-dump.mjs` | `plan/planner`, `env/scripted`, `nn/rng`, `core/types`, `plan/memory` |
| `gen-planner-freerun.mjs` | `plan/planner`, `env/scripted`, `nn/rng`, `core/types` |
| `gen-component-dump.mjs` | `env/scripted`, `plan/{seal,throwLines,shield,planner}`, `nn/rng`, `core/types` |
| `gen-nav-dump.mjs` | `plan/route`, `nn/rng`, `core/types`, `bot/{wayblock,crossing,navigate,bot}` |
| `gen-episode`, `gen-opscript`, `gen-regression`, `bench-node` | только `lib.mjs` |
| `tools/jsmath-oracle/rng_probe.mjs` | `nn/rng` |

Замыкание — 40 файлов из раздела 3; единственный внешний пакет — `teeworlds` (через `bot/bot.ts` и `bot/netPatch.ts`;
`ink`, `react`, `electron`, типы не нужны). Без `node_modules` работают все генераторы, кроме `gen-nav-dump.mjs`.

## 5. Решение: перенос в `tools/ts-reference/`, а не «оставить на месте»

Выбран перенос с сохранением внутренней раскладки (`tools/ts-reference/src/…`):

- корень Rust-репозитория перестаёт быть Node-проектом (нет `package.json`/`src/`), нет путаницы «где рабочий код»;
- `git mv` сохраняет историю (`git log --follow`); внутри каталога относительные импорты TS не меняются;
- цитаты `src/plan/planner.ts:NNNN` в сотнях комментариев Rust остаются осмысленными (читать относительно
  `tools/ts-reference/`, это записано в README каталога) — Rust-файлы не правились, конфликтов с параллельными задачами нет;
- цена: константа `TS_REF` и пути в 5 генераторах + `rng_probe.mjs`; `tsCoreCommit` стал закреплённой константой
  (`dd8c1e3d…`, последний коммит, менявший `core/`+`map/`), иначе `git log -- src/core src/map` после переноса вернул бы
  коммит удаления и заголовки дампов разошлись бы с уже существующими корпусами.

## 6. Доказательства (все — из рабочего каталога `~/aiddnet/wt/task-5.4`)

Исходные корпуса — `~/aiddnet/data/traces/*` (не в git, не менялись). «Новый» = генератор из `tools/ts-trace` после переноса.

| Дамп | Команда (параметры из заголовка корпусного файла) | Результат |
|---|---|---|
| планировщик `planner/arena_normal_hold.jsonl` | `gen-planner-dump.mjs --map synthetic:arena --seed 13 --cases 220 --preset normal --opponent hold --scenario baseline` | `cmp` с корпусом: **идентичен** (221 строка) |
| компоненты `planner-components/clb.jsonl` | `gen-component-dump.mjs --map <CLB> --seed 1 --cases 400` | **идентичен** (2394 строки) |
| навигация `nav/clb-follow.jsonl`, `nav/clb-arrival.jsonl` | `gen-nav-dump.mjs --section follow --seed 9 --pairs 200` и `--section arrival --seed 7 --pairs 200` | оба **идентичны** корпусу |
| навигация, полный набор секций (`--section all --seed 1 --routes 1000`) | генератор до переноса и после | **идентичны** друг другу; первые 1003 строки совпадают с корпусным `nav/clb.jsonl` (корпус снят до появления поздних секций `wb`/`cross`/`nav`) |
| `gen-episode`, `gen-opscript`, `gen-regression` (4 случая), `gen-planner-freerun`, `rng_probe` | дерево до (`git archive 0311695`) против дерева после | идентичны (кроме поля `tsCoreCommit`, которое «до» из архива без `.git` печатало `unknown`; регрессионный `f2` совпадает и с зафиксированной фикстурой `crates/ddai-tsworld/tests/fixtures/`) |

Паритетные тесты Rust (`--features ts-parity --release -- --ignored`, команды — в `crates/ddai-planner/README.md` и
`crates/ddai-nav/README.md`), прогнаны на рабочем дереве после переноса:

| Тест | Данные | Результат |
|---|---|---|
| `ddai-planner` `parity_planner` | `DDAI_PLANNER_DUMP_DIR=~/aiddnet/data/traces/planner` | 19 360 решений, 70 файлов, **0 расхождений** |
| `ddai-planner` `parity_components` | `planner-components/clb.jsonl` | seal/throwLines/shield/scripted/touchesFreeze — **0 расхождений** |
| `ddai-planner` `parity_planner_freerun` | `planner-freerun/` | 3 604 решения, 10 игр, **0 расхождений** |
| `ddai-nav` `parity_nav` | `nav/{clb,blmapchill,chillblock5}.jsonl` | по 1500 маршрутов, **0 расхождений** |
| `ddai-nav` `parity_nav` | свежий дамп, снятый перенесённым генератором (`--section all`, CLB) | маршруты, ВБ-зоны, навигатор, вейблок, переходы — **0 расхождений** |

Остальные проверки: `cargo fmt --all --check`, `cargo clippy --workspace --all-targets --locked -- -D warnings` (и с
`ts-parity` для `ddai-planner`/`ddai-nav`), `cargo test --workspace --locked` (2720 прошли, 0 упали, 76 `#[ignore]`),
`cargo deny check`, `tools/ci/no-weights.sh` — зелёные.

## 7. Что осталось за рамками и последствия

- Внешний офлайн-стенд `~/aiddnet/data/research/harness` (вне репозитория) читает `DDNET_AI_SRC`, по умолчанию
  `~/aiddnet/DDNet-AI/src`. После слияния 5.4 путь надо задать `DDNET_AI_SRC=~/aiddnet/DDNet-AI/tools/ts-reference/src`
  (стенд использует только перенесённые файлы либо `bot/main.ts`, которого больше нет — тогда брать коммит `0311695`).
- В репозитории потребитель — комментарий-инструкция `crates/ddai-env/tests/fixtures/gen_harness_spawns.mjs`: он
  запускается через этот же внешний стенд, поэтому перед командой нужен `DDNET_AI_SRC=<repo>/tools/ts-reference/src`.
- Исторические исследования (`docs/ORIGINAL.md`, `docs/research/orig-*.md`, `docs/research/*`) описывают оригинальное
  дерево; их ссылки на `app/…`, `src/bot/main.ts`, `start.mjs` указывают на коммит `0311695`.
- В доках по e2e с локальным сервером (`tools/ddnet-server/README.md`, `docs/formats.md` §9) старый TS-бот запускался как
  клиент; процедура помечена исторической, живой клиент — `ddnet-ai play`.
