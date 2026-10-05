# tools/ts-reference — замороженный TS-эталон для паритетных дампов

Здесь лежат **только те TypeScript-файлы старого бота** (`Wranked1/DDNet-AI`, GPL-3.0), которые реально исполняют
генераторы паритета `tools/ts-trace/*.mjs` и `tools/jsmath-oracle/rng_probe.mjs`. Это не рабочий бот: Electron-окно,
TS-рантайм бота, веб-страница старого бота, `start.mjs`, `run.sh` и модель `opponent.json` удалены задачей 5.4
(полный перечень и обоснование — `docs/research/ts-removal.md`). Файлы **не редактируются** и не копируются: перенесены
через `git mv`, `git log --follow <файл>` ведёт в историю оригинала. Последний коммит с полным TS-деревом —
`0311695` (родитель коммита удаления).

Зачем они нужны: Rust-порты (`ddai-tsworld`, `ddai-planner`, `ddai-nav`, `ddai-jsmath`) доказывают свою верность тем,
что принимают **те же решения, что и настоящий TS**. Эталон — это сам TS-код, выполняемый на Node 24 (type stripping,
без компиляции). Пока паритетные тесты (`--ignored`) живы, эталон живёт в репозитории.

## Раскладка

```
tools/ts-reference/
  package.json, package-lock.json   # одна зависимость: teeworlds 2.6.1 (её импортирует src/bot/bot.ts)
  src/
    core/      # физика TS-мира (порт DDNet 20 на TS): vmath, tuning, types, collision, characterCore, projectile, world
    map/       # datafile.ts, loadMap.ts — чтение .map
    plan/      # planner, route, seal, shield, throwLines, memory, livePlan
    env/       # scripted, action, obs
    nn/        # rng (Rng на splitmix32 с Гауссом), gru, params
    bot/       # wayblock, crossing, navigate, bot (методы wbSpot/steerFollow), liveWorld, netPatch, ...
    demo/, watch/, train/, i18n*.ts   # транзитивные импорты bot.ts (в дампах не участвуют)
```

Корень `tools/ts-reference/` — это «корень репозитория старого проекта». Поэтому комментарии в Rust-коде и в
`docs/**`, которые ссылаются на `src/plan/planner.ts:1044`, `src/bot/bot.ts:4748` и т. п., остаются верными: читать
такие пути надо **относительно `tools/ts-reference/`** (номера строк не менялись, файлы побайтно те же).

## Почему перенос, а не «оставить `src/` на месте»

- В корне Rust-проекта не остаётся `package.json`/`src/` от другого проекта: ни `npm`, ни IDE не принимают корень
  за Node-проект; `npm ci` нужен только здесь и только для `teeworlds`.
- Каталог самодостаточен и очевиден: удалить эталон (когда паритет станет не нужен) = удалить одну папку.
- Цена — одна строка в `tools/ts-trace/lib.mjs` (`TS_REF`) и пути импортов в генераторах; ссылки `src/…` в Rust-коде
  не трогали (см. выше), чтобы не конфликтовать с параллельными задачами в `crates/*`.

## Как пользоваться

```bash
cd tools/ts-reference && npm ci --ignore-scripts      # один раз; нужен Node >= 24 (проверено на v24.21.0)
cd ../ts-trace
node gen-planner-dump.mjs --map synthetic:arena --seed 13 --cases 220 --preset normal --opponent hold \
  --scenario baseline --out ~/aiddnet/data/traces/planner/arena_normal_hold.jsonl
```

`node_modules` нужен **любому** запуску `gen-nav-dump.mjs` (он импортирует `src/bot/bot.ts` → `teeworlds`) и ничему
больше: планировщик, компоненты, `tsworld`-генераторы и `rng_probe.mjs` работают на одном Node без `npm ci`.

**Второй эталон (задача 3.8).** Переменная `DDAI_TS_REF=<каталог с src/>` направляет генераторы `tools/ts-trace/*.mjs` на другую выгрузку
апстрима (по умолчанию — эта, закреплённая на `c3c619d`; без переменной вывод генераторов прежний). Для планировщика релиза 2026-10-02
(`af49dfb`): `git -C ~/aiddnet/ref/DDNet-AI-upstream archive af49dfb src package.json package-lock.json | tar -x -C ~/aiddnet/data/scratch/ts-af49dfb`,
затем `tools/ts-trace/run-planner-corpus-v2.sh` (см. «Версии планировщика» в `crates/ddai-planner/README.md`).

Полные команды перегенерации корпусов и запуска паритетных тестов:

- планировщик, компоненты, free-run — `crates/ddai-planner/README.md` («Как перегенерировать корпус паритета»);
- навигация — `crates/ddai-nav/README.md` («Проверки»);
- физика TS-мира — `crates/ddai-tsworld/README.md`, `docs/formats.md` §16;
- Rng/`Math.*` — `tools/jsmath-oracle/README.md`.

`tsCoreCommit` в заголовках дампов — это константа (`TS_CORE_COMMIT` в `tools/ts-trace/lib.mjs`): последний коммит,
менявший `src/core` и `src/map` до переноса. Раньше значение брал `git log`; после `git mv` оно изменилось бы и
заголовки перестали бы совпадать побайтно с уже сгенерированными корпусами.

## Лицензии

Файлы `src/core/*.ts` и `src/map/loadMap.ts` — порт физики DDNet 20 (zlib-уведомление Teeworlds/DDRace/DDNet) и
остаются под условиями, описанными в корневом `NOTICE` (раздел «Third-party code»). Остальные файлы — код
Wranked1/DDNet-AI (GPL-3.0-only). `teeworlds` (npm) — MIT, ставится из реестра, в репозиторий не вендорится.
