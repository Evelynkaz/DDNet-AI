# ddai-nav

Задача 4.2. **Навигация, переходы через фриз-трубы, вейблок (ВБ) на Copy Love Box, trek/seek/home и память о
фризах** — порт `src/plan/route.ts`, `src/bot/navigate.ts`, `crossing.ts`, `wayblock.ts`, `src/plan/memory.ts` и
навигационных кусков `bot.ts`. **Задача 4.8** довела порт до релиза апстрима `af49dfb` (2026-10-02): поиск зала по тайлам,
споты гарда, «маршрут 2» через стену, прямой замах, торможение на цели (`docs/research/upstream-2026-10-02.md`). Построение заметки: `docs/research/nav.md` (таблица «функция TS → Rust, PORT / CHANGE /
DROP»). Крейт общий для двух миров: `PlanWorld`/`PlanCollision` из `ddai-planner` — побитно-точный `World<f32>` (живой бот,
арена) и f64-мир `ddai-tsworld` (паритет с TS, фича `ts-parity`).

Проводка в бота — `ddai_bot::nav_hooks` (крючки 4.1: `Navigator`, `WayBlock`, `Trek`, `RouteFinder`); здесь только логика,
которая ни от клиента, ни от снапшотов не зависит.

## Модули

| Модуль | Что внутри | TS |
|---|---|---|
| `grid` | `NavGrid`: свободно / хукаемо / твёрдо / смерть / unfreeze / опасно, телепорты, первый твёрдый вдоль 8 лучей | `gridOf` |
| `route` | `Router::find_route` — поиск маршрута по графу ход / падение / прыжок / хук / респавн (`allow_kill`) / телепорт, `through_freeze`, `avoid`, `partial`; `spawn_tiles`; `dead_zone` («из клетки нет пути обратно в игру») | `route.ts` |
| `runner` | `RouteRunner`: идёт по шагам маршрута, замечает «застрял», «замёрз по пути», «вето щита» | `navigate.ts` |
| `crossing` | `SwingCrosser`: поиск перехода трубы (качание на крюке, прыжок), устойчивый к лагу и подталкиванию; бюджет по стенному времени. С 4.8: **прямой замах** в проход (`direct_anchors`, общий кадр симуляции кандидатов), **маршрут 2** через дальнюю стену прохода на нижнюю полку (`WallRoute`, `use_wall`), история отправленных вводов в прогонах из воздуха | `crossing.ts` |
| `navigator` | `Navigator`: поле BFS до цели → маршрут → переходы труб → зондирование телепортов; `NavGoal`, `tile_goal`, `tele_goals`. С 4.8: торможение на цели и перед ней против опасности за целью (с лагом), `wall_route`, `crossing_state`, `cross_fails`, `hazard_within_px` | `navigate.ts` |
| `follow` | `Follow`: идти за движущимся игроком (`steerFollow`), `follow_tile` | `bot.ts` |
| `trek` | `busiest_spot` / `game_spot`, `Trek` (идти туда, где игра), `PathGoal` (точка пути к цели для планировщика) | `bot.ts` |
| `home` | `Home`: домашняя точка, забывается на другой карте | `bot.ts` |
| `wayblock` | зоны / споты / якоря / трубы CLB и JoniTee, **поиск зала по тайлам (`find_hall_offset`)**, `wayblock_for` (имя и размер **или** зал в любой карте), стены «маршрута 2», `WbGuardGeom`, `WbSideChooser`, `wb_spot`, `WbState` (режим, сторона, пауза после смертей), `wb_band` | `wayblock.ts`, `bot.ts` |
| `memory` | `MemoryStore`: файл памяти фризов по sha256 карты, распад, атомарная запись | `memory.ts` |
| `harness` | гонки одиночных проходов для сравнения с TS (`run_goto`, `run_follow`), Уилсон, Макнемар | — |

## Вейблок на Copy Love Box

Числа зон, спотов, якорей и труб — как в `wayblock.ts` (карта 387×250; вариант JoniTee — сдвиг `(182, 212)` на карте
600×600). `wayblock_for(имя, collision)` узнаёт карту **двумя путями** (с 4.8, как апстрим `af49dfb`):

1. **по имени и размеру** — стоячие споты, хукаемые якоря; именованная версия (387×250, JoniTee 600×600);
2. **по тайлам** — иначе зал ищется в *любой* карте независимо от имени: образец `HALL_CORE` (83×29 тайла: твёрдый / фриз /
   смерть / остальное) сравнивается во всех сдвигах, сначала по выборке (каждый 25-й непустой и 150-й пустой тайл, ≥ 80 %), потом
   целиком (≥ 95 %); побеждает первый наилучший сдвиг. Найденное определение — то же, сдвинутое на `(dx, dy)`, имя
   `Copy Love Box hall at +dx,+dy`; оно принимается, если споты стоячие, якоря хукаемые, старты труб стоячие и выходы труб
   свободны от твёрдого и фриза. Так на Swarfey (468×255, зал на `(0, 0)`, совпадение 0,994) и на копиях с залом в другом месте
   бот держит ВБ; раньше там играл обычный блок (триггер задачи 4.8).

Стена «маршрута 2» (`Crossing::wall`) проверяется отдельно (`wall_route_ok`): не подошла — труба остаётся без неё.
`DDAI_WB_GUARD=0` возвращает прежние споты (в TS `WB_GUARD`); по умолчанию первый спот — левый край верхней полки.

- **Сторона** (`WbSideChooser`, `auto`): та, где играет меньше людей; при равенстве — где мы стоим, иначе ближняя;
  без игроков выбор «провизорный» 5 с; **без перескоков** (`WB_SIDE_HOPPING = false`). `!wb left|right|off|auto`.
- **Пауза после смертей по дороге:** 4 смерти подряд на пути к ВБ → ВБ не держится `5 · 2^(n−1)` минут, не больше 30; вход в
  зал живым прощает счётчики. Часы — настоящие, как в TS, но передаются как `now_ms` (проверяется в тестах).
- **Планировщик в зале:** `WB_PLAN_OVERRIDES` = `noThawRope`, `frozenThrow 3`, `airJumpCost 0.3`, `launchExactReach 100`
  (`ddai_planner::config::wb_overrides`), в режиме «strong» при популяции < 40 — ещё `STRONG_WB`. Бот передаёт их мозгу как
  `LiveContext::wb` (`WbHints { in_hall, strong, band }`); `PlannerBrain` применяет их через `set_overrides`/`set_band`. **Гибрид:**
  у `HybridConfig::planner` те же поля; динамическое включение «только в зале» требует метода в `HybridBrain` (там идёт 3.5b), поэтому
  пока `WbHints` гибрид не читает (проводка — одна строка в его `set_live_context`).
- **Выбор цели на ВБ:** вес `+300` для тия в зоне, «поводок» (`in_leash`), встречная атака («counter»), `wbFinish` — замороженный
  тий в зоне, пока мы в зале, остаётся целью, если не запечатан (`WayBlock::filter`).
- **«Лежим во фризе на ВБ»:** `WB_LYING 25` тиков, кулдаун 100 — `WayBlock::wants_kill` с настоящей длительностью заморозки.

## Что отличается от TS (CHANGE)

Подробно с обоснованиями — `docs/research/nav.md` §9. Главное: память фризов привязана к **sha256 карты**, а не к имени; после
выхода «почти у цели» (маршрут кончился в 2 тайлах, но дальше 40 px от центра) навигатор докручивает последнюю часть пешком
(`NavOpts::finish_approach`, по умолчанию выкл. — в паритетных тестах — и вкл. в живом боте); часы паузы ВБ и бюджет перехода
инъектируются; партнёр / дуэль / «спасти друга» вырезаны (D-021). Две ошибки живого адаптера мира (`PhysicsWorld::tele_at` и
`tele_outs_for`), найденные при сверке сеток, исправлены в `ddai-planner`.

## Проверки

```bash
source ~/.cargo/env
cargo test -p ddai-nav                                          # модульные тесты (память, дом, ВБ, поиск, версии карт)
# паритет с настоящим TS (нужны Node 24, карты в ~/aiddnet/data/maps и один раз `npm ci` в tools/ts-reference — пакет teeworlds для bot.ts):
(cd tools/ts-reference && npm ci --ignore-scripts)
node tools/ts-trace/gen-nav-dump.mjs --map "<карта>" --seed 1 --out ~/aiddnet/data/traces/nav/clb.jsonl
DDAI_NAV_DUMP=~/aiddnet/data/traces/nav/clb.jsonl cargo test -p ddai-nav --features ts-parity --release --test parity_nav -- --ignored --nocapture
# сравнение силы (доля прибытий, Уилсон, Макнемар) на трёх картах:
tools/ts-trace/run-nav-arrival.sh 200
```

**Два эталона (4.8).** `tools/ts-reference/` — замороженная копия `c3c619d` (не редактируется). Дампы нового релиза делаются из
извлечённого каталога `af49dfb`, на который указывает `DDAI_TS_REF`:

```bash
mkdir -p ~/aiddnet/data/scratch/ts-af49dfb && cd ~/aiddnet/ref/DDNet-AI-upstream \
  && git archive af49dfb src package.json package-lock.json | tar -x -C ~/aiddnet/data/scratch/ts-af49dfb
mkdir -p ~/aiddnet/data/scratch/ts-af49dfb/node_modules && cp -r ~/aiddnet/DDNet-AI/node_modules/teeworlds ~/aiddnet/data/scratch/ts-af49dfb/node_modules/
export DDAI_TS_REF=~/aiddnet/data/scratch/ts-af49dfb
for s in deadzone routes wb cross nav; do   # секции генератора; routes --routes 600, cross --crossings 60, nav --navs 40
  node tools/ts-trace/gen-nav-dump.mjs --map "<карта>" --seed 1 --section $s --out ~/aiddnet/data/traces/nav-af49dfb/<имя>-$s.jsonl
done
DDAI_NAV_DUMP=<дамп> cargo test -p ddai-nav --features ts-parity --release --test parity_nav -- --ignored --nocapture
```

Дамп `af49dfb` (в заголовке `hall: true`) проверяет всё: `findHallOffset`, определение целиком (`wbdef`), геометрию гарда,
зоны, `wbSpot`, переходы (`crosstrace`, в т. ч. «маршрут 2») и навигатор (`navtrace`, в т. ч. `wallRoute`). Дамп старого
эталона проигрывается частично (маршруты, мёртвая зона, спавны, выбор стороны, зоны): переходы, споты и навигатор `af49dfb`
изменил намеренно. Результаты по картам (Swarfey 468×255, оригинал 387×250, версия с залом на +152, BlmapChill, ChillBlock5) — в
`docs/research/upstream-2026-10-02.md` §4. Версии карт на живом адаптере (`PhysicsWorld`), включая JoniTee, сделанный из
оригинала (карты 600×600 нет), — `tests/wayblock_versions.rs`.

Результаты паритета и сравнения — в конце `docs/research/nav.md` и в записи решения `docs/DECISIONS.md`.
