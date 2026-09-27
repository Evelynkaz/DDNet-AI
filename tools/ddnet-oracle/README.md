# Oracle A — C++-эталон физики DDNet (core-уровень)

Собирает и запускает реальные `gamecore.cpp`/`collision.cpp`/`layers.cpp`/`teamscore.cpp`/
`prng.cpp` из исходников DDNet 20.1 (тег `20.1`, коммит
`c9d208138f85755521f16a0096b6fe036c5c8698`) против сценария (карта + вводы), core-уровень
только: `CCharacterCore::Tick`/`TickDeferred`/`Move`/`Quantize`, без DDRace-логики персонажа
(без фриза, тайловых эффектов, оружия — это Oracle B, будущая задача). Пишет побайтово
детерминированный трейс (`docs/formats.md`, раздел 6). Задача 1.3 (Rust-порт
`CCharacterCore`/`CCollision`) должен совпасть с этими трейсами без единого расхождения.

Исходники DDNet **не коммитятся** в этот репозиторий — `fetch.sh` каждый раз скачивает их в
`build/` (в `.gitignore`). Файлы этого каталога (`oracle_core.cpp`, `sha256.h`, `*.sh`, `*.py`)
— наш собственный код (GPL-3.0-only, как весь репозиторий); ни строчки DDNet сюда не
скопировано, только вызовы через заголовки после сборки объектных файлов из скачанного дерева.

## Файлы

| Файл | Назначение |
|---|---|
| `fetch.sh` | shallow+sparse `git clone` DDNet на теге `20.1`, проверка коммита, генерация `build/generated/protocol.h` (`datasrc/compile.py network_header`, как в реальной сборке DDNet). |
| `build.sh` | Компилирует `game/{gamecore,collision,layers,teamscore,prng}.cpp` из скачанного дерева + `oracle_core.cpp` в `build/oracle_core`. |
| `oracle_core.cpp` | Сам харнесс: свои небольшие ридеры rawmap v1/scenario v2, сборка `IMap`/`CLayers`/`CCollision`, разрешение «режима прицела» (`ResolveInput`, docs/formats.md §2.1), цикл тика, запись trace v1. |
| `sha256.h` | Маленькая своя реализация SHA-256 (для проверки `map_sha256` сценария и для `scenario_sha256` в метаданных трейса) — не код DDNet. |
| `selftest.sh` | Собирает оракул, генерирует ≥ 20 сценариев (`ddnet-ai trace gen-scenario`, все 4 рецепта), гоняет оракул по два раза на каждый, сверяет sha256, печатает throughput. |
| `gen_fixtures.sh` + `build_fixture.py` | Пересобирает золотые фикстуры `crates/ddai-trace/tests/fixtures/*.json` (см. `docs/formats.md` §7), включая пару `no_weak_hook` / tuning-override (раунд 2, находка F11: у tuning-фикстуры теперь реалистичные переопределения, не гасящие взаимодействие персонажей). |
| `bulk_run.sh` | Прогоняет ≥ 200 сценариев (3000 тиков × 3 персонажа) плюс два отдельных ≥ 20-сценарных слайса — только `no_weak_hook` (обычный тюнинг) и только tuning-переопределения (обычный `no_weak_hook=false`) — раунд 2, находка F11 — итого ≥ 240, в `~/aiddnet/data/traces/oracle-a/v1/` (вне репозитория) для задачи 1.3. |
| `build/` | Игнорируется git; скачанные исходники DDNet, сгенерированный заголовок, объектные файлы, бинарник, рабочие файлы селфтеста. |

## Использование

```bash
cd tools/ddnet-oracle
./fetch.sh                 # один раз (и повторно — идемпотентно, если коммит уже верный)
./build.sh                 # собирает build/oracle_core

# Карту и сценарий готовит Rust CLI (crates/ddnet-ai), не оракул:
../../target/release/ddnet-ai trace export-map --recipe arena --out /tmp/arena.rawmap
../../target/release/ddnet-ai trace gen-scenario --recipe arena --seed 1 --ticks 3000 --chars 3 --out /tmp/s.scn

build/oracle_core /tmp/arena.rawmap /tmp/s.scn /tmp/out.trace --generator random-v1 --seed 1
# stderr: "oracle_core: N ticks, M characters, K tee-ticks/s, wrote ... (... bytes)"

../../target/release/ddnet-ai trace diff /tmp/out.trace /tmp/out2.trace
../../target/release/ddnet-ai trace hashes /tmp/out.trace --out /tmp/hashes.json

# no_weak_hook / tuning-переопределения — отдельные флаги gen-scenario (не параметры
# random-v1, см. docs/formats.md §2): один сценарий может задать оба сразу, хотя bulk_run.sh
# (ниже) прогоняет их как два отдельных слайса, не смешивая (раунд 2, находка F11).
../../target/release/ddnet-ai trace gen-scenario --recipe arena --seed 1 --ticks 3000 --chars 3 \
    --no-weak-hook --tune gravity=40 --out /tmp/s_nwh.scn

./selftest.sh               # полная проверка детерминизма + throughput
./gen_fixtures.sh           # пересборка золотых фикстур (после изменения генератора/оракула)
./bulk_run.sh                # ≥ 240 сценариев в ~/aiddnet/data/traces/oracle-a/v1/
```

`--generator`/`--seed` у `oracle_core` — необязательная аннотация в JSON-метаданных трейса
(`docs/formats.md` §6.1), сама симуляция от них не зависит: оракул воспроизводит ровно то, что
записано в файле сценария (и проверяет его `map_sha256` перед запуском).

## Почему так собрано

- **Не патчим DDNet.** `m_MoveRestrictions` у `CCharacterCore` приватное и без геттера — вместо
  патча заголовка оракул повторяет тот же (чистый, `const`) запрос к `CCollision` в нужный
  момент тика (см. `docs/formats.md` §5.5) и получает то же самое значение доказуемо, а не
  предположительно.
- **Флаги сборки** — `-std=c++20 -O2 -fno-fast-math -ffp-contract=off -fsigned-char`, без
  `-march`: тот же набор, что и в прототипе фазы 0
  (`data/research/physics-scratch/run.sh`) и что подтверждено экспериментом
  (`docs/research/ddnet-physics.md` §4) не давать FMA-контракции на x86-64 без `-march`, что
  важно для побитового совпадения с Rust `f32`.
- **Список компилируемых `.cpp`** — `gamecore`, `collision`, `layers`, `teamscore`, `prng`;
  `mapitems.cpp` не компилируется (её свободные функции не вызываются ни из одного из
  вышеперечисленных файлов — проверено `grep`); `prng.cpp` нужен, даже когда `CWorldCore::m_pPrng
  == nullptr` (что здесь всегда так — оракул никогда не вызывает `InitSwitchers`/не задаёт PRNG):
  `CWorldCore::RandomOr0` — не шаблон, содержит вызов `m_pPrng->RandomBits()` под `if`, поэтому
  символ нужен линкеру независимо от того, выполнится ли эта ветка в рантайме.
- **`CTuningParams` без раунд-трипа через float.** Переопределения тюнинга применяются через
  `NetworkArray()[index] = value_x100` в обход `CTuningParams::Set(name, float)` — тот сам умножает
  на 100, а обратное деление/умножение float может не восстановить исходное целое (см.
  `docs/formats.md` §2).
- **Явный zero-init `CCharacterCore`.** `Reset()` не трогает `m_ActiveWeapon`/`m_Colliding`/
  `m_LeftWall`/`m_MoveRestrictions`/`m_Id` — на реальном сервере их устанавливает код вокруг
  `CCharacter`, которого здесь нет. `std::vector<CCharacterCore>(N)` (value-initialization)
  гарантирует, что все они стартуют с `0`/`false` — см. `docs/formats.md` §5.6, это часть
  контракта для задачи 1.3.
- **Валидация id/`aim_slot` (раунд 1, находка F5).** `ParseScenario` отклоняет `id >=
  MAX_CLIENTS` (реальная константа DDNet — **128**, `engine/shared/protocol.h`; черновик
  находки F5 в отчёте ревью называл 64 — это `LEGACY_MAX_CLIENTS`, не тот предел, который тут
  важен, см. build report), дублирующиеся `id`, и `aim_slot` вне `-1..characters.size()` — до
  того, как эти значения дошли бы до `World.m_apCharacters[Id]` (реальный размер массива).
- **`resolve_input`/`aim_slot` (раунд 1, находка F3).** Оракул разрешает «прицел» персонажа
  заново каждый тик, читая `Cores[k].m_Pos` (целочисленное после `Quantize()`) в `int32_t`
  арифметике — см. `docs/formats.md` §2.1 для полного алгоритма; здесь важно только то, что это
  ОТДЕЛЬНАЯ от Rust-реализации (`crates/ddai-trace/src/scenario.rs::resolve_input`), но
  алгоритмически идентичная копия — обе стороны специфицированы одним и тем же текстом в
  `docs/formats.md`, а не одна вызывает другую.

## Известные ограничения (вне охвата этой задачи)

Нет weapons/freeze/tiles/switch-дверей/tune-зон — это `CCharacter`-уровень (Oracle B). Свитчеры
(`switch`-слой) в этом харнессе всегда неактивны (`m_vSwitchers` не заполняется) — двери не
работают, но `STOP`/`STOPS`/`STOPA` из front/game-слоя от свитчеров не зависят и проверяются
рецептом `front` напрямую.

## `map2raw` (задача 1.4) — чтение настоящих `.map`-файлов

Отдельный от `oracle_core` инструмент: читает настоящий DDNet `.map` (datafile v3/v4) через
РЕАЛЬНЫЙ, не переписанный код DDNet (`engine/shared/datafile.cpp` + `engine/shared/map.cpp` +
`game/layers.cpp`, из тех же скачанных `fetch.sh` исходников) и пишет физические слои
(game/front/tele/speedup/switch/tune, обрезанные до размера game-слоя — как это фактически
делает `CCollision::Init`) + Settings в формате rawmap v1 (`docs/formats.md` §1) — то же самое,
что производит `ddnet-ai trace export-map --map <file> --out <file.rawmap>` (крейт `ddai-map`).
Задача 1.4: `crates/ddai-map/src/loader.rs` должен совпасть с этим побайтово на всём корпусе карт.

```bash
./build.sh                      # собирает build/map2raw вместе с build/oracle_core
build/map2raw input.map out.rawmap

# Паритет на корпусе (см. tools/ddnet-oracle/map-corpus-check.sh):
cd ../.. && cargo build --release -p ddnet-ai && cd tools/ddnet-oracle
./map-corpus-check.sh ~/aiddnet/data/research/proto-scratch/ddnet-maps \
    ~/aiddnet/data/maps/copy-love-box ~/aiddnet/data/research/physics-scratch/maps

# Дифференциальный мутационный фаззинг (раунд 1 ревью — настоящие карты в корпусе выше все
# ПРИНИМАЮТСЯ обеими сторонами, так что сам корпус не проверяет паритет на ОТКЛОНЕНИЕ; этот режим
# портит случайные поля внутри LAYER-item'ов настоящей карты и сравнивает результат):
./map-corpus-check.sh --mutate ~/aiddnet/data/research/physics-scratch/maps/BlockField.map \
    --seed 42 --iterations-per-map 1000
```

### Песочница по памяти/времени (раунд 2 ревью, находка F7)

И `map-corpus-check.sh`, и `map_mutation_fuzz.py` запускают **оба** инструмента — `build/map2raw`
И `../../target/release/ddnet-ai trace export-map --map` — под жёстким лимитом виртуальной памяти
и таймаутом: `prlimit --as=2147483648` (2 ГиБ) + `timeout --signal=KILL 20s`. Причина: `map2raw`
линкует настоящий, немодифицированный код DDNet, у которого (как и у самого DDNet) нет гарантий
ограниченной памяти — намеренно испорченная (мутационным фаззером) карта может попросить у
`ReadSettings`/`std::vector` сотни миллионов записей и упасть в `std::bad_alloc`/`abort()` (реальный
инцидент раунда 1: ~6.8 ГиБ RSS на карте с испорченной v4 declared-size таблицей для Settings-блоба,
`settings_bomb_256m.map` в отчёте о билде). Классификация кода выхода под лимитом: `0` — приняла,
`1` — собственный чистый отказ инструмента (`Fail()`/код ошибки `ddai-map`), что угодно другое
(`124`/`137` от `timeout`, `128+сигнал` от `prlimit`) — инструмент упёрся в СВОЙ лимит, а не дал
осмысленный ответ. Для `map2raw` это репортится отдельно (`cpp (map2raw) resource-limited`) и НЕ
считается расхождением — ожидаемая, документированная форма (`ddai-map`'s bounded reads, критерий
приёмки №2 задачи 1.4, — именно то, чего у real-DDNet-кода нет). Для `ddnet-ai`/`ddai-map` тот же
код выхода — это всегда баг (`RUST/ddai-map crashed`) и всегда проваливает прогон: критерий №2
обязывает `ddai-map` никогда не падать и никогда не выделять память пропорционально заявленному
(а не реальному) размеру данных, так что сам Rust-инструмент не должен упираться в этот лимит на
файле любого размера, каким бы испорченным он ни был.

`map2raw.cpp` — своя (не DDNet) обвязка: минимальный `IStorage` (открывает путь к файлу напрямую
через `io_open`, без реальной подсистемы хранения), заглушки для `g_UuidManager`/`CUuid` (все
типы карточных item'ов, которые нас интересуют — VERSION/INFO/GROUP/LAYER — намного меньше
`OFFSET_UUID`, так что реальный код `g_UuidManager` никогда не вызывается для них, см. комментарий
в файле) и `log_log`/`log_log_color` (не нужен вывод DDNet, ошибки видны через код возврата
`CMap::Load`). Компилирует РЕАЛЬНЫЕ `base/{str,mem,io,hash_libtomcrypt,bytes,unicode/tolower*}.cpp`
— не переизобретает их — подробности и обоснование каждой заглушки в шапке `map2raw.cpp`.

`map2raw` повторяет «ленивый» `nullptr`-фолбэк DDNet (`CCollision::Init` оставляет указатель
`nullptr`, если данные КОНКРЕТНОГО не-game физического слоя не декомпрессировались, — карта всё
равно загружается, этот слой просто отсутствует) — `GetGameLayerData` (жёсткий отказ, как у
настоящего `CMap::Load` для game-слоя) отдельно от `GetOptionalPhysicsLayerData` (мягкий, как у
`CCollision::Init`, для всех остальных). `ddai-map`'s единственный оставшийся bounded-allocation
предел сверх собственных проверок DDNet — см. `crates/ddai-map/src/lib.rs`'s top-level doc
comment, `docs/formats.md` §10.2 и отчёт о билде задачи 1.4.
