# Навигация, переходы, вейблок, trek/home и память фризов: TS → Rust (задача 4.2)

Исходники (читаются только на чтение, как в D-021): `src/plan/route.ts` (984 строки), `src/bot/navigate.ts` (784),
`src/bot/crossing.ts` (483), `src/bot/wayblock.ts` (254), `src/plan/memory.ts` (130) и куски `src/bot/bot.ts`
(goto/follow 1696-2023, ВБ 4024-4211, trek/seek/home 4251-4329 и 4448-4480, цикл снапшота 2425-2600, `pickTarget` с ВБ
3030-3123, `planAction` 4695-4830). Общая картина — `docs/research/orig-bot.md` §8.2-§8.5, §10.3.

Статус в таблицах: **PORT** — переносится как есть (по возможности бит-в-бит, см. «Паритет»), **CHANGE** — переносится с
изменением (причина указана), **DROP** — не переносится.

## 0. Как устроен порт

* Новый крейт **`ddai-nav`** (чистая логика, без сети и без `ddai-bot`): сетка маршрутов, `find_route`, `dead_zone`,
  `RouteRunner`, `Navigator`, `SwingCrosser`, определения ВБ и выбор стороны/споты, `busiest_spot`/`start_trek`/`trek`-цель,
  `home`, постоянная память фризов. Обобщён по `ddai_planner::plan_world::{PlanCollision, PlanWorld}` — как планировщик
  (3.2): на `ddai-tsworld` (f64, `ts-parity`) он доказывается сравнением с TS, на `ddai_physics::World<f32>` играет вживую.
* `ddai-bot` зависит от `ddai-nav` и реализует крючки 4.1 (`Navigator`, `WayBlock`, `Trek`, `RouteFinder`) тонкими обёртками
  (`ddai-bot/src/nav_hooks.rs`); `ddai-nav` ничего не знает о боте, снапшотах и игроках.
* `ddai-planner` не меняется, кроме мелких публичных аксессоров к `FreezeMemory` (сохранение/загрузка). `ddai-brain`
  получает в `LiveContext` необязательную подсказку ВБ (полоса и переопределения конфига). Код гибрида (3.5b) не трогается.
* Где TS берёт время (`Date.now()`), в Rust — явные часы: бюджет кроссера в миллисекундах (для паритета 0 = без
  бюджета), пауза ВБ после смертей — часы бота (минуты реального времени).

## 1. `src/plan/route.ts`

| TS | Rust (`ddai-nav`) | Статус | Примечания |
|---|---|---|---|
| `Grid`, `gridOf` (free/hookable/solid/death/unfreeze/danger/teleOut/teleIn/firstSolid) | `grid::NavGrid::new(&impl PlanCollision)` | PORT | `free` = не стена, не фриз, не смерть, плюс вход телепорта (`TELEIN`/`TELEINEVIL`); `danger` = 3×3 содержит фриз/смерть; `firstSolid` — по 8 лучам; кэш по `identity()` карты вместо `WeakMap` |
| `supported`, `clearLine` (Брезенхэм), `overshootsIntoHazard`, `jumpClear`, `freezeCrossing` | `grid::*` | PORT | порядок обхода и целочисленная арифметика сохраняются |
| `Heap` (бинарная куча, ёмкость n, тихий отказ при переполнении) | `route::Heap` | PORT | порядок выхода при равных стоимостях зависит от точной реализации sift — копируется 1:1, включая `if size >= capacity return` (TS-особенность, на практике не срабатывает; документируется, в Rust — тот же предел) |
| `expand` (walk/fall/freeze-fall/freeze-cross/jump/tele/hook, спавны `kill`) | `route::expand` | PORT | константы `COST_*`, `JUMP_UP_REACH`/`JUMP_DOWN_REACH`, `JUMP_MARGIN 0.8` |
| `expandBack` (обратное ребро для `deadZone`) | `route::expand_back` | PORT | |
| `findRoute(collision, from, to, {nearTiles, maxCost, partial, allowKill, maxNodes, throughFreeze, avoid})` | `route::find_route` | PORT | Дейкстра с `stamp`/`gen`; `COST_DANGER` добавляется к стоимости входа; `avoid` — набор `routeMoveKey`; частичный маршрут, если ближайшая точка на 34% ближе |
| `RouteStep`, `routeMoveKey(index, kind)=index*8+kind` | `route::RouteStep`, `route_move_key` | PORT | |
| `RouteRunner` (ход по шагам, замер застоя, `kill`-шаг, `freezePlanned`, `earlyFreeze`, `hopClear`, `teeInTheWay`) | `route::RouteRunner` | PORT | `decisions % 2` для прыжка сохраняется; `others` — срез `TeeState` |
| `routeField` | — | **DROP** | нигде не используется (мёртвый код, `grep` по `src/`) |
| `spawnTiles` (тайлы `ENTITY_SPAWN 192` в игровом и front-слое) | `route::spawn_tiles(&MapData)` | PORT | `PlanCollision` не отдаёт front-слой, поэтому берётся из `MapData` |
| `deadZone`, `deadZoneOf` (прямой и обратный залив от спавнов, компоненты > 250 тайлов сбрасываются) | `route::dead_zone`, `DeadZone` | PORT | кэш по карте; отдаётся и планировщику (`set_dead_zone`), и unstick-правилу 4.1 (`Navigator::in_dead_zone`) |

## 2. `src/bot/navigate.ts`

| TS | Rust | Статус | Примечания |
|---|---|---|---|
| `NavGoal`, `tileGoal`, `teleGoals` (до 8 телепортов по `travelField`) | `navigator::{NavGoal, tile_goal, tele_goals}` | PORT | |
| `traceRoute`, `distAt`, `fieldTo` (через `travelField`) | `navigator::*` | PORT | `travel_field` уже в `ddai-planner::fields` |
| `Navigator.step` (окно застоя 150 тиков, зондирование телепорта 100, прыжок позиции ≥96 px / `12 px/тик` = телепорт/смерть/перемещение, `findRoute` при застое, `MAX_ROUTE_REPLANS 3`, `MAX_ROUTE_DROPS 3`, `MAX_ALTERNATIVE_ROUTES 4`, `avoid`) | `navigator::Navigator::step` | PORT | возвращает ввод + флаги (`crossing`, `planned_freeze`, kill); заметки (`takeNotes`) → события |
| `follow` (идём по 4-тайловому «лучу» поля, торможение у фриза, подъём `climb` на крюке, `aimAt` с лимитом 0.12 рад/тик) | `navigator::Navigator::follow` | PORT | `climb`, `findAnchor`, `hazardWithin` — как есть |
| `startCrossing`, `stepCrossing`, `crossingReach` (переход труб ВБ) | `navigator::*` | PORT | `MAX_CROSS_TRIES 4`, `PLANNED_FREEZE_STEPS 16` |
| `respawned`, `cancel`, `vetoed`, `crossing`/`plannedFreeze`, `tilesLeft`, `progress`/`brief` | то же | PORT | `progress`/`brief` — тексты для будущих консольных команд (4.3), без ников |
| `crossBudgetMs` (бюджет времени кроссера на кадр, `LOW_CPU.navMs = 10`) | `Navigator::set_cross_budget` | CHANGE | `Instant` вместо `Date.now`; 0 = без бюджета (паритет) |

## 3. `src/bot/crossing.ts` (`SwingCrosser`, переходы через фриз-трубы)

| TS | Rust | Статус | Примечания |
|---|---|---|---|
| `TileBox`, `Crossing`, `inBox`/`inAnyBox`/`shiftBox`/`shiftCrossing` | `crossing::*` | PORT | |
| `SwingCrosser.step` (approach → swinging/hopping → arrived/failed) | `crossing::SwingCrosser<W: PlanWorld>::step` | PORT | |
| поиск: `searchSwing`/`searchHop`/`searchDrop`, `robust` (разброс лага −1..+2, сдвиги ±6/±4 px), `rollout` | то же | PORT | мир-симулятор — `PlanWorld::new_scratch` (на `ts-parity` бит-в-бит с TS `SimWorld`); `firstThat` с «кругами» при нехватке бюджета |
| `HOLDS`, `HOP_RUNS`, `HOP_JUMP_AT`, `DROP_RUNS`, `AIR_*`, `SETTLE_TICKS 110` и др. | `crossing::consts` | PORT | |

## 4. `src/bot/wayblock.ts`

| TS | Rust | Статус | Примечания |
|---|---|---|---|
| `WAYBLOCKS` (Copy Love Box 387×250; JoniTee — сдвиг 182,212, 600×600), зоны/подходы/споты/watch/трубы/якоря | `wayblock::{WbDef, wayblocks()}` | PORT | числа 1:1 |
| `wayblockFor(name, col)` (имя + размеры + `standable` споты + якоря крюка) | `wayblock::wayblock_for` | PORT | карта опознаётся и по имени, и по проверкам геометрии (другие версии карты не берутся) |
| `sideAt`, `inWbZone`, `inWbHall`, `inWbLeash`, `wbWalkAllowed`, `standable` | то же | PORT | |
| `WbSideChooser.update` (`auto`, провизорный выбор, `WB_SIDE_HOPPING = false`) | `wayblock::WbSideChooser` | PORT | |

## 5. Части `bot.ts`

| TS | Rust | Статус | Примечания |
|---|---|---|---|
| `wbHolding`, `updateWbSide`, `wbSpot` (занятые споты, друзья/партнёр), `walkToWb`, `wbWalkCuttable` | `wayblock::WbState` + `ddai-bot::nav_hooks` | PORT | партнёр (`isPartnerNow`) — DROP (D-021, партнёр/тиммейт вырезан); друзья считаются через `PlayerTable` бота |
| `noteWbWalkDeath` (4 смерти по пути → пауза `5·2^(n−1)` мин, максимум 30) | `wayblock::WbState::note_walk_death` | PORT | часы — минуты реального времени (инъекция `now_ms`) |
| `wbBand`, `wbPlanOverrides` (`WB_PLAN_OVERRIDES = {noThawRope, frozenThrow 3, airJumpCost 0.3, launchExactReach 100}`, `STRONG_WB` при population < 40) | `ddai_brain::WbHints` в `LiveContext`; `PlannerBrain` применяет через `set_band`/`set_overrides` | CHANGE | переменная среды `WB_PLAN` (JSON) не переносится; для гибрида отображение описано в README `ddai-nav` |
| `wander` у споты ВБ (`anchorX`, `lookAt = watch`) | `ddai-bot::wander` (уже в 4.1) + якорь из `WbState` | PORT | |
| `maybeUnstick`: «лежим во фризе на ВБ» (`WB_LYING 25`, кулдаун 100) | `WayBlock::wants_kill(ctx, frozen_for)` (крючок 4.1) | PORT | реальная длительность фриза уже передаётся (4.1, F6) |
| `pickTarget`: вес `+300` в зоне ВБ, `inWbLeash`, «counter», `wbFinish` | `WayBlock::filter` (крючок 4.1; тело в 4.2) | PORT | |
| `busiestSpot`, `gameSpot` (исключает зоны `avoid`), `crowdAt` | `trek::{busiest_spot, game_spot}` | PORT | «не AFK и не запаркован во фризе > 250 тиков» берёт `PlayerTable`/часы активности бота |
| `startTrek` (`findRoute(partial, allowKill, throughFreeze:false, avoid=trekAvoid)`, отказ при обрыве > 3 тайлов у фриза), `trekGoal` (шаг `kill`, `TREK_REACHED_PX 56`, `TREK_STALL_TICKS 150`, запрет хода ≤ 24), `endTrek` | `trek::{start_trek, Trek}` | PORT | `kill`-шаг → `Cl_Kill` с кулдауном 500 тиков |
| `pathGoal` (`PATH_NEAR_PX 420`, обновление раз в 25 тиков, `maxNodes 4000`) | `trek::path_goal` | PORT | включается, если конфиг планировщика `pathToTarget` (по умолчанию как в TS) |
| `home` (`!home`, `GO_HOME_AFTER_TICKS 200`, забывается при смене карты, ВБ не держится пока `home` задан) | `home::Home` | PORT | команды консоли — 4.3, здесь API |
| `driveNav` (шаги nav через `guard`, кроме `crossing`/`plannedFreeze`; `takeKill` → кулдаун 500) | `ddai-bot::nav_hooks::NavHook::drive` | PORT | |
| follow-режим (`gotoPlayer`, `steerFollow`, `followTile`, константы `FOLLOW_*`) | `follow::Follow` | PORT | цель — клиентский id (CLI `--follow <id>`), не ник |
| `gotoCommand` (текстовый парсер `tele`/`x y`/`@ник`) | API `NavApi::goto_tile/goto_tele/follow` | CHANGE | разбор текста — 4.3 |

## 6. `src/plan/memory.ts` (`FreezeMemory`)

| TS | Rust | Статус | Примечания |
|---|---|---|---|
| `note`, `notePass`, `safety`, `risk`, `SPREAD 0.4` | `ddai_planner::memory::FreezeMemory` (3.2) | PORT (готово) | |
| `save` (`DECAY 0.97` на каждое сохранение, отсечки `< 0.01`/`< 0.05`), `load` (несовпадение размеров → пустая) | `ddai_nav::memory::MemoryStore` | PORT | формат JSON `{width,height,events,idx,val,pidx,pval}` тот же |
| ключ файла `runs/memory/<имя карты>.json` (без CRC) | `~/aiddnet/data/bot/memory/<sha256>.json` | **CHANGE** | ключ — sha256 карты (две версии одной карты не смешиваются; ошибка D-/orig-bot §13.7), каталог вне git |
| сохранение: каждые 20 заморозок, при смене карты, при остановке | то же | PORT | + атомарная запись `.tmp` → rename |
| использование: `memoryTrust 0.9` в штрафе `selfHazard` | `PlannerBrain` (уже умеет `set_freeze_memory`) | PORT | гибрид — см. README (память читает только `Planner`; у гибрида пока нет такого входа) |

## 7. Паритет с TS

Генераторы `tools/ts-trace/gen-nav-dump.mjs` (запускают настоящие `src/plan/route.ts`, `src/bot/crossing.ts`,
`src/bot/wayblock.ts` через type-stripping Node 24, как 3.1a/3.2) пишут JSONL; тесты `ddai-nav/tests/parity_*.rs`
(`ts-parity`, `#[ignore]`) сравнивают: `findRoute` (шаги и стоимость; с `throughFreeze` и без, `avoid`, `partial`,
`allowKill`), `deadZone`, выбор перехода (`searchSwing`/`searchHop`/`searchDrop` на состояниях у труб), выбор стороны ВБ и
споты. Карты: Copy Love Box, BlmapChill, ChillBlock5. ≥ 1000 запросов на функцию.

## 8. Сравнение силы навигации

Способ: **TS-навигатор запускается в собственном мире** (Node-харнесс `tools/ts-trace/nav-arrival.mjs`, настоящий
`Navigator` + `SimWorld` f64, одиночный бот, без соперников) на тех же парах старт/цель, что и Rust (`ddai-nav` на
`World<f32>`). Это даёт сравнение «вся система целиком», а не копию решений: физика f32 против f64 различается лишь в
шумовом хвосте, а различие в решениях и есть то, что измеряется. Прибытие ≤ 64 px за лимит TS (`STALL`-окна и
`FOLLOW_MAX_TICKS`); отчёт — доля прибытий, доверительный интервал Уилсона, время, фризы, `Cl_Kill`. Пары: ≥ 200 на карту
(Copy Love Box: оба зала ВБ и трубы; BlmapChill; ChillBlock5), с `throughFreeze` и без.

## 9. Решения, найденные при чтении TS (ошибки/странности — вносятся по мере порта)

1. `routeField` не используется (DROP).
2. Память по имени карты без CRC (CHANGE, sha256).
3. `Heap.push` молча отбрасывает вставку при переполнении — переносится 1:1 ради паритета; на практике не достигается
   (ёмкость = число тайлов, `maxNodes ≤ 200000`); в Rust добавляется счётчик «отброшено» для диагностики.
4. Время в `SwingCrosser` — стена (`Date.now()`), недетерминированно; в Rust `Instant`, для паритета бюджет 0.
5. Пауза ВБ после смертей — стена (`Date.now()`); в Rust — инъекция часов.

6. **Мир `PhysicsWorld` (3.2) врал про телепорты** (найдено сверкой сеток TS-мира и `World<f32>`, `ddai-nav/tests/grid_worlds.rs`):
   `tele_at` возвращал `(номер, номер)` вместо `(тип, номер)`, а `tele_outs_for(n)` смотрел таблицу, ключом которой служит
   `n − 1` (как в `CCollision`), со сдвигом на единицу. На картах с телепортёрами (Copy Love Box) маршруты через телепорт
   в живом мире «замуровывались», а живой `sealed_in` не видел телепорты. Исправлено в `ddai-planner::physics_adapter`; сетки,
   `tele_out` и мёртвая зона теперь совпадают с TS на всех трёх картах (0 расхождений).
   **Сознательное отличие от `teleAt` TS:** живой `tele_at` сообщает только входы (`TELEIN`, `TELEINEVIL`) — по ним строятся `NavGrid` и `sealed_in`, больше
   никто не спрашивает. TS `teleAt`/`teleGoals` отдают любой тип телепорта, поэтому живой `?goto tele` не перечисляет выходы (`TELEOUT`), а `seal` не возвращает −1 для
   телепорт-чекпоинтов; в мире TS (паритет) всё как раньше.
7. **Слой front и пикапы-сердца.** Коллизия TS смотрит на freeze только в игровом слое; живая физика (как DDNet) — и во фронтальном (BlmapChill: 161 тайл).
   В DDNet **сердце-пикап замораживает** коснувшегося (`gamecontroller.cpp:288`: `ENTITY_HEALTH_1` -> `POWERUP_FREEZE`; `pickup.cpp:54`), радиус 48 px (20 + 28);
   `World<f32>` это моделирует сущностью, а тайловые проверки — нет. Сердец: Copy Love Box 16 (столбец x = 163..165, y = 104..119 за внешней стеной — на ходьбу не влияет),
   BlmapChill 26 (23 во фронтальном слое), blmapV5_ddpp 5 (фронт), ChillBlock5 20. **Сделано** (ревью F2): `ddai_physics::map::pickup_freeze_mask` (сердце в игровом или
   фронтальном слое, блок 3×3 тайла вокруг — тий на центре тайла в пределах 48 px) читают `PlanCollision::is_freeze` живого мира (отсюда `NavGrid` free / danger и
   `seal::rests_in_freeze`) и `MapGrid` бота (щит, анстик, достижимость). Только живой бэкенд: мир TS пикапов не знает, паритет не затронут. Сравнение силы на этих картах
   делается в двух вариантах — настоящая карта и карта без фронта и пикапов (`DDAI_ARRIVAL_NOFRONT=1 DDAI_ARRIVAL_NOPICKUPS=1`), где препятствия у обеих систем одинаковы.
8. **«Прибыл» ≠ «у цели».** `Navigator` объявляет «arrived», когда маршрут прошёл до конца, а конец маршрута (`nearTiles 2`)
   может быть в 2 тайлах (до 90 px) от центра цели; критерий приёмки — ≤ 64 px. CHANGE: `NavOpts::finish_approach` (в живом боте
   включён, в паритетных тестах выключен): если после маршрута мы дальше 40 px от центра, до двух раз идём пешком на последнюю
   часть. Видно в сравнении: без него на ChillBlock5 Rust проигрывал TS (52% против 57%), с ним выигрывает.
   **Всё преимущество Rust над TS в goto даёт именно он**: с `DDAI_NAV_EXACT=1` (без `finish_approach`) Rust совпадает с TS (CLB: 57,5% / 74,0% против 56,5% / 74,0%).
   Цена — время: медиана прибытия на CLB с фризом 116 тиков против 72 у TS (+61%).
9. **Размораживание по таймеру в гонках:** живой бот убивает себя (`Cl_Kill`) после 400 тиков во фризе подряд (анстик). Гонки
   обеих сторон делают то же (`FROZEN_UNSTICK_TICKS = 400`), иначе заморозка в яме была бы концом прогона.
10. **`trapCare`** (`deadZoneCost > 0`) выключен в живом пресете; правило «не целиться в тия из мёртвой зоны, когда мы сами не там»
    не переносится (мёртвая зона нужна анстику и передаётся планировщику как `DeadZoneGrid`).
11. **`rescueFriend`** (спасение друзей) — DROP (D-021: друзья-тиммейты вырезаны); `pickTarget` по-прежнему не трогает друзей.
12. **`someoneWorthFighting` при обрыве «поиска игры»:** TS вызывает ещё и `pickTarget`; здесь «кого-то стоит бить» уже исключает
    друзей, игнорируемых и вышедших из игры (то, что `pickTarget` и так пропускает), а полный подбор цели — только когда ход
    закончился.

13. **Передача знаний о карте мозгу** (ревью F4): мёртвая зона — один `Arc<Vec<u8>>` на карту; память фризов — глубокая копия (на карте в 1 млн тайлов около 8 МБ) **один раз в тик
    замерзания** и при загрузке карты, мозг и его клоны делят её через `Arc` (копирование при записи в `FreezeMemory`). Раньше копировалось дважды на каждые 500 тиков и после каждого
    замерзания (p50 4,7 мс, максимум 34 мс на ChillBlock5). Теперь опрос в стационарном режиме — p50 90 нс, p99 3,8 мкс; передача при замерзании — p50 7,4 мс, максимум 22 мс
    (VM нагружена тремя потоками), но замёрзший тий решения всё равно не принимает.
14. **Свои `Cl_Kill` в слежении** (ревью F5): `Navigator::kill_sent(tick, by_route)` вызывается для каждого `Cl_Kill` бота (анстик, trek, маршрут), как `lastKillTick` TS;
    смерть в пределах 50 тиков после собственного убийства не считается «смертью по дороге» (`self_deaths`); `routeKillTick` (смерть на ходе к ВБ) — только убийства самого маршрута.
15. **Щит на ходу** (ревью F6): ввод навигатора и `wander` проверяются щитом на мире, в котором он подействует (свои уже отправленные вводы применены до тика вступления в силу,
    `predict_own`), как `guard` TS (`lagTicks` тиков с `prevInput`) и путь мозга; раньше — мир тика снапшота. Сценарий с лагом у края фриза: `tests/scenarios.rs`.
16. **Пары слежения с телепортирующейся целью** (ревью F3): скриптованная цель-кинематика может задеть тайл телепорта и в живом мире физически переноситься (в мире TS нет). Такие
    пары (цель одна, без преследователя, скачок > 96 px за тик в любом из миров) отбрасываются: в генераторе (мир TS) и в Rust-тесте (`harness::target_jumps`, живой мир).

Остальные расхождения добавляются сюда с обоснованием по мере реализации (D-021: «чиним ошибки TS сознательно»).

## 10. Результаты (2026-10-01)

### 10.1 Паритет с настоящим TS (0 расхождений везде)

Генератор — `tools/ts-trace/gen-nav-dump.mjs` (настоящие `route.ts`, `navigate.ts`, `crossing.ts`, `wayblock.ts`, `bot.ts` через type-stripping
Node 24), тесты — `crates/ddai-nav/tests/parity_nav.rs` (`ts-parity`, `#[ignore]`, `DDAI_NAV_DUMP=<дамп>`), итоги последнего прогона:

| Функция | Copy Love Box | BlmapChill | ChillBlock5 |
|---|---|---|---|
| `findRoute` (шаги, стоимость, виды ходов; `throughFreeze` и без, `avoid`, `partial`, `allowKill`) | 1500 запросов (421 с маршрутом: 164 хука, 14 фриза, 81 респавн, 55 телепорт) | 1500 (290: 150 / 38 / 77 / 2) | 1500 (533: 1017 хуков / 0 / 31 / 0) |
| `deadZone` целиком (все тайлы) + `spawnTiles` | 96 750 тайлов (974 мёртвых) | 829 748 (51) | 1 013 725 (0) |
| зоны ВБ (`sideAt`, `inWbZone`, `inWbHall`, `inWbLeash`, `wbWalkAllowed`) | 4000 тайлов × 2 варианта (CLB, JoniTee) | — | — |
| `WbSideChooser` (последовательности) / `wbSpot` | 1500 / 2000 | — | — |
| выбор перехода трубы (`SwingCrosser`, поиск и ввод по тикам) | 160 трасс (97 прошли в TS), ~43 тыс. решений | — | — |
| целый `Navigator` (ввод каждого тика, ноты, фазы, исход) | 136 прогонов | 56 | 81 |
| `Heap`: вставки, отброшенные из-за переполнения | 0 | 0 | 0 |

Сверка «сетки двух миров» (`tests/grid_worlds.rs`): `NavGrid`, `tele_out`, `first_solid`, мёртвая зона, маршрут с респавном совпадают у TS-мира и
`World<f32>` на всех трёх картах, кроме 161 тайла фронтального слоя на BlmapChill (пункт 7 выше). Паритет планировщика не тронут
(`planner.rs` не менялся; `parity_planner`: 19 360 решений, 0 расхождений).

Те же `Navigator`/`Follow`/харнесс Rust **в мире TS** (`follow_and_goto_runs_match_ts_exactly_on_the_ts_world`) дают исходы настоящего TS
побитово: 2308 прогонов (400 + 400 + 400 + 308 + 400 + 400 по дампам goto и follow на трёх картах), 0 расхождений (прибытие, тики, `Cl_Kill`, фризы).

### 10.2 Сила навигации: доля прибытий (≤ 64 px, 6000 тиков), 200 пар на режим

Метод — раздел 8. Rust идёт в живой физике (`World<f32>`, с маской сердец и фронтальным слоем), TS — в своём мире; фриз на 400 тиков подряд кончается `Cl_Kill` у обоих;
Уилсон 95%, парный тест Макнемара (одностороннее p того, что Rust хуже TS). Скрипт — `tools/ts-trace/run-nav-arrival.sh 200` (после ревью 1: маска сердец, фильтр пар слежения).

**goto** (настоящие карты):

| Карта | без фриза: TS / Rust | с фризом: TS / Rust | p(Rust хуже) |
|---|---|---|---|
| Copy Love Box (залы и трубы) | 56,5% [49,6; 63,2] / **61,5%** [54,6; 68,0] | 74,0% [67,5; 79,6] / **80,0%** [73,9; 85,0] | 1,00 / 1,00 |
| BlmapChill | 74,0% [67,5; 79,6] / **82,0%** [76,1; 86,7] | 65,5% [58,7; 71,7] / **72,0%** [65,4; 77,8] | 1,00 / 1,00 |
| ChillBlock5 | 57,0% [50,1; 63,7] / **64,5%** [57,7; 70,8] | 58,5% [51,6; 65,1] / **66,0%** [59,2; 72,2] | 1,00 / 1,00 |
| ChillBlock5 без фронта и пикапов | 57,0% / **65,5%** [58,7; 71,7] | 58,5% / **68,0%** [61,2; 74,1] | 1,00 / 1,00 |

На BlmapChill без фронта и пикапов числа те же. Категории Copy Love Box «с фризом»: `wb-in` и `wb-out` (36 пар каждая, вход в зал и выход из зала по трубе) — **100% у обоих**.
**Всё превосходство Rust над TS в goto даёт `finish_approach`** (пункт 8 раздела 9): с `DDAI_NAV_EXACT=1` Rust равен TS. Цена — время: медиана прибытия на CLB с фризом 116 тиков
против 72 (+61%); на ChillBlock5 132 / 168 против 97 / 122. Фризы и `Cl_Kill` сопоставимы; на ChillBlock5 настоящей карты у Rust 1-3 фриза на 200 пар (до маски сердец было 12 и 5)
против 1 и 0 у TS.

**follow** (скриптованная цель-кинематика идёт по маршруту `findRoute` между тремя тайлами, пауза 80 тиков). Отброшены пары, где цель **одна**, без преследователя, прыгает > 96 px за тик
в мире TS (генератор: 17 скриптов CLB) или в живом мире (Rust-тест; цель задевает `TELEIN` #1 / #3 и переносится физически — в TS-мире нет; ревью F3). Результат не утверждается жёстко
(докстринг `tests/arrival_vs_ts.rs`: код тот же, исходы совпадают побитово в мире TS, разница — миры):

| Карта | без фриза: TS / Rust (p) | с фризом: TS / Rust (p) |
|---|---|---|
| Copy Love Box | 40,5% / **43,0%** (1,00) | **49,5%** / 46,5% (0,073) |
| BlmapChill | 68,5% / 65,7% (0,125; n = 108) | 21,5% / 21,5% (1,00) |
| ChillBlock5 | **36,5%** / 31,0% (0,035) | **36,0%** / 34,0% (0,27) |
| ChillBlock5 без пикапов | 36,5% / **38,5%** (0,86) | 36,0% / **42,0%** (1,00) |

Отставание на Copy Love Box с фризом, которое в первом раунде выходило значимым (p = 0,011), было артефактом харнесса (телепортируемая цель) и после фильтра **не значимо** (p = 0,073).
Остаётся ChillBlock5 без фриза (p = 0,035): живой преследователь замерзает на сердцах 18-21 раз на 200 прогонов (маска сняла 49-51 до 18-21; остаток — падение и качание сквозь
зону 3×3 в открытой пещере, где сердца висят в воздухе), у TS в его мире сердец нет (0 фризов); без пикапов отставания нет.

### 10.3 Вейблок-сценарии (`ddai-env`, `configs/arena/wb-hold.toml`, 200 партий на условие, `ddnet-ai arena run`; без `swap`, ревью F7)

Арены `clb-wb-left` (train) и `clb-wb-right` (holdout, зеркало): слот 0 стоит на первом споте ВБ зала, нарушители (`scripted`) входят из боксов зала на 3-14 тайлов; сторонами партии **не меняются**
(чередуется порядок спавна). `+wb` = планировщику сообщают `WB_PLAN_OVERRIDES` и полосу, пока он в зале (`WbHintBrain`). Главная метрика — credited-победы (D-059).

| Условие | credited-побед, % [95% ДИ] | W:L:D:T | самозаморозок/мин | в полосе, % |
|---|---|---|---|---|
| idle против 1 нарушителя (база) | 0,0 [0,0; 1,9] | 71:93:24:12 | 7,10 | 4,5 |
| scripted против 1 (база) | 5,5 [3,1; 9,6] | 11:123:0:66 | 2,94 | 0,2 |
| planner против 1 | 99,0 [96,4; 99,7] | 198:2:0:0 | 0,17 | 5,9 |
| **planner+wb против 1** | 99,0 [96,4; 99,7] | 198:2:0:0 | 0,18 | 4,5 |
| idle против 2 (база) | 0,0 [0,0; 1,9] | 0:149:0:51 | 4,31 | 3,5 |
| scripted против 2 (база) | 32,5 [26,4; 39,3] | 65:107:1:27 | 3,44 | 3,8 |
| planner против 2 | 95,5 [91,7; 97,6] | 191:7:0:2 | 0,48 | 4,6 |
| **planner+wb против 2** | 96,5 [93,0; 98,3] | 193:3:0:4 | 0,17 | 5,5 |
| holdout planner против 1 | 79,5 [73,4; 84,5] | 196:4:0:0 | 1,08 | 32,1 |
| holdout planner+wb против 1 | 80,0 [73,9; 85,0] | 197:3:0:0 | 0,63 | 26,8 |
| holdout planner против 2 | 95,5 [91,7; 97,6] | 191:9:0:0 | 1,51 | 32,5 |
| holdout planner+wb против 2 | 94,5 [90,4; 96,9] | 189:9:1:1 | 1,30 | 29,8 |

Вывод: арена против скриптованного нарушителя **насыщена** (планировщик выигрывает 95-99% партий и без подсказок), `WB_PLAN_OVERRIDES` разницы в credited-победах не дают (ДИ перекрываются во всех
парах); самозаморозки с подсказками ниже в трёх парах из четырёх планировщика (0,48 -> 0,17 против 2; 1,08 -> 0,63; 1,51 -> 1,30), но на 200 партиях это не значимо. «Время в зале»
(`inWbHall`) — 100% везде: партии решаются внутри зала. «Время в полосе» (`wbBand`; штраф только при `bandCost > 0`, по умолчанию 0) — до 6% слева и около 30% справа (первый спот
справа ближе к трубе). Первый прогон (до F7) смешивал стороны в половине партий; выводы те же.

### 10.4 Локальный сервер (Copy Love Box)

См. отчёт сборки раунда 1: `e2e_nav` (goto в 3 точки 12 из 12; ВБ 125 с, 83-95% времени в зале), `e2e_local_server` (после F1 ботам задан `WbMode::Off`), `tools/e2e/session.sh`.

## 11. Обновление 4.8: релиз апстрима 2026-10-02 (`af49dfb`)

`wayblock.ts`, `crossing.ts` и `navigate.ts` перенесены заново с `af49dfb` (поиск зала по тайлам, споты гарда, «маршрут 2» через стену, прямой замах,
торможение на цели); таблица «что изменилось → решение», паритет по картам и ссылки на дампы — `docs/research/upstream-2026-10-02.md`, измерения — E-018,
решение — D-092. Исходные пункты этого документа (`PORT / CHANGE / DROP`) остаются верными для кода, который 4.8 не менял.
