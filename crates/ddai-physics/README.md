# ddai-physics

Побитно точный порт физики **DDNet 20.1** на Rust: ядро персонажа (задача 1.3) и серверный мир
`World<R>` (задача 1.6, стадии A и B). Всё обобщено по скаляру `R: Real` (`f32` — боевой, побитно равен
C++; `f64` — собирается и работает, паритета с C++ не требует). `unsafe` нет, зависимостей нет
(`ddai-trace`, `criterion`, `allocation-counter`, `serde_json` — только dev).

Правила точности (D-002/D-003/D-004): порядок операций и типы (`float`/`double`/`int`) копируются из
C++ буквально; математика — `std` (glibc: `powf`, `sin`, `cos`, `atan`, `log`); никакого `mul_add`,
fast-math, `libm`, `target-cpu`. В каждом портированном файле — zlib-уведомление DDNet, пометка
«altered» и ссылки `файл:строка` на эталон (`~/aiddnet/build/ddnet-20.1/src/src/game/...`).

## Что внутри

| Модуль | Что |
|---|---|
| `real`, `vmath`, `prng`, `tuning` | скаляр, `vec2`, PCG-генератор DDNet, 47 параметров тюнинга |
| `map`, `collision` | плоские слои карты; `CCollision` (линии, `MoveBox`, тайлы, тележки, свитчи) |
| `core`, `core_world` | `CCharacterCore`/`CWorldCore`/`CTeamsCore`; шаг мира уровня Oracle A |
| `switch` | слой switch: двери, таймеры |
| `world` | `World<R>`: серверный мир целиком (ниже) |
| `world/laser.rs` | `CLaser` (лазер и дробовик), `CInteractions::CanHit`, список `LaserList` |
| `world/fixtures.rs` | `CDragger`/`CDraggerBeam`, `CGun`/`CPlasma`, `CLight` |
| `world/ninja.rs` | `HandleNinja`, активация рывка в `FireWeapon` |

## `World<R>`: что воспроизводится

Стадия A: все DDRace-тайлы (фриз, глубокий и live-фриз, телепорты всех видов, спидапы, стопперы, switch-слой
с дверями и таймерами, tune-зоны, endless hook/jump, refill/walljump, NPC/NPH/HIT, solo, команды), молоток,
`CProjectile` (пуля, граната, crazy-shotgun карты), пикапы, спавн/смерть/респавн `CPlayer`, порядок ввода сервера.

**Стадия B (задача 1.6b):**

- `CLaser` — лазерная винтовка **и дробовик** (в DDNet 20.1 дробовик — это лазер): отскоки, `m_Energy`/`m_Bounces`,
  `laser_reach`/`laser_bounce_*`/`shotgun_strength`, попадание (разморозка, подтяжка, «баг дробовика»), телепорт
  через `TILE_TELEINWEAPON` (`RandomOr0`), телеган-лазер (`m_TeleportCancelled`, `ALLOW_TELE_GUN`),
  `sv_old_laser`, `sv_destroy_lasers_on_death`, `CInteractions::CanHit` (команды, solo, отключённые удары).
- `CDragger` (слабый/нормальный/сильный, с обходом стен и без), `CDraggerBeam`: цель на команду, solo отдельно,
  привязка к switch-слою, движение по тайлам-конвейерам.
- Турель `CGun` + `CPlasma` (заморозка / разморозка / взрыв): частота `sv_plasma_per_sec`, дальность
  `sv_plasma_range`, цели по командам, два взрыва за тик, как в C++.
- `CLight`: вращение, раскрывающиеся/закрывающиеся лучи, заморозка, привязка к switch-слою.
- Ниндзя: пикап, рывок на 10 тиков со скоростью 50, удары по пути, 15 секунд, возврат оружия.
- **`m_Pos` персонажа** (`Character::pos`) — позиция *сущности*, отдельная от позиции ядра: её копирует только
  `Spawn()` и `TickDeferred()`, поэтому рывок ниндзя или телепорт внутри тика не видны ни остальным сущностям,
  ни тайлам до конца тика. `World::step` в начале тика пересинхронизирует её с ядром.
- **Порядок сущностей** — как у `CGameWorld::Tick`: снаряды → список `ENTTYPE_LASER` → пикапы → персонажи →
  `TickDeferred`. `ENTTYPE_LASER` один на лазеры, лучи драггеров, выстрелы турелей, драггеры, турели, свет
  и двери; новые всегда вставляются в голову списка, поэтому динамические (`World::lasers`, хранятся «старые
  первыми», обход с конца) всегда идут впереди статических (`World::fixtures`, порядок создания, новые первыми).
- Без аллокаций на горячем пути: ёмкость `lasers`/`projectiles`/`active_timed_switchers` резервируется
  один раз при создании (`LASER_CAPACITY`, `PROJECTILE_HEADROOM`); `LaserList` сохраняет резерв при `Clone`.
  Единственная аллокация, которую может сделать `step`, — удвоение `lasers` при > 64 живых лазеров/лучей/выстрелов
  одновременно (карта с сотней турелей и толпой).

Не воспроизводится (недостижимо из сценариев Oracle B и без видимого эффекта в сравниваемых полях): `/rescue`,
`/pause`, блокировка/флок команд, practice-режим, сохранение команд.

## Паритет с настоящим сервером

Эталон — Oracle B (`tools/ddnet-oracle/server/`, настоящий серверный код DDNet 20.1 в процессе). Формат трассы
`trb` v3 и сценариев — `docs/formats.md` §11, §12, §30.

```bash
source ~/.cargo/env
# Корпус v1 (650 трасс, 5,97 млн тик-персонажей) — каждый тик, без обрезки:
cargo test -p ddai-physics --release --test parity_oracle_b -- --ignored replays_full_oracle_b_corpus --nocapture
# Корпус стадии B (~/aiddnet/data/traces/oracle-b/v2-stageb/: турельные карты + наведённые сценарии):
cargo test -p ddai-physics --release --test parity_oracle_b -- --ignored replays_stage_b_corpus --nocapture
# Золотые фикстуры в репозитории (10 штук, ~600 КБ, работают без корпуса):
cargo test -p ddai-physics --test parity_oracle_b_stage_b_fixtures
```

`DDAI_PARITY_QUIET=1` печатает только итоги по картам и расхождения; `DDAI_DEBUG_TICKS=a-b` печатает для
окна тиков состояния персонажей «наше / эталон» бок о бок; `DDAI_STAGE_B_DIR` подменяет каталог корпуса стадии B.

Сравнивается каждое поле схемы: 28 полей ядра и 54 поля DDRace каждого персонажа (`to_bits` для `f32`),
свитчи по командам, снаряды, весь список `ENTTYPE_LASER` **по порядку** (лазеры, лучи с адресатом, выстрелы
турелей, затем позиции дверей/драггеров/турелей и параметры света) и `died/respawned_this_tick`.
Прежнее правило обрезки стадии A (`docs/formats.md` §17) осталось только как диагностика: тест всё так же
считает, сколько тик-персонажей лежит за ним.

## Производительность

`cargo run --release -p ddai-physics --example world_step_cpu_time` — минимум по тысячам коротких партий
(устойчиво на загруженной машине, где `criterion` даёт разброс в сотни процентов). Цифры стадии B — в
`docs/EXPERIMENTS.md`-стиле отчёте задачи (BUILD REPORT 1.6b) и в `docs/research/ddnet-physics.md` §11.
Быстрые пути для лучей драггеров и выстрелов турелей: `Collision::intersect_no_laser[_no_walls]` пропускают
участки отрезка, где таблица накопленных сумм доказывает отсутствие блокирующих клеток
(дифференциальный тест — `collision::tests::intersect_no_laser_fast_paths_match_the_plain_loops`).

## Тесты

- `tests/world_stage_b.rs` — семантика каждой сущности стадии B на крошечных картах + проверка отсутствия
  аллокаций со всеми сущностями сразу.
- `tests/parity_oracle_b_stage_b_fixtures.rs` — золотые фикстуры (хэш на тик по персонажам и всем сущностям).
- `tests/world_no_alloc.rs` — ноль аллокаций в `step` (включая повтор фикстуры стадии B).
- `tests/parity_oracle_b.rs` — корпуса (см. выше).
