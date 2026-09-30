# ddai-dataset — датасет «человеческая игра» (задача 8.4c)

Превращает клиентские демки DDNet (`ddai-demo`, 8.4b) в обучающие данные для мухи: пары `(Observation, Action)` с
метаданными, тегами приёмов блока (T1–T18, D-048), сигналами мастерства (D-030) и **измеренным** качеством
восстановления ввода. Источник — архив ChillerDragon (D-040): читается только локально, ники нигде не сохраняются.

Отдельный крейт, а не модуль `ddai-brain` или `ddai-fly`: конвейеру нужны `ddai-demo`, `ddai-map`, `ddai-world`
(LiveWorld), `ddai-recorder` (реконструкция ввода) и `ddai-brain` (типы), а контракт `ddai-brain` намеренно лёгкий
(«no heavy deps»); тренер (8.2) зависит только от читателя этого крейта. Общие крейты почти не тронуты
(см. «Что изменено вне крейта» ниже).

## Конвейер

```
.demo ──ingest──► анонимные кадры recorder ──build──► LiveWorld + ввод + реплей ──► FrameRec / SampleRec
                                                        │
                       analysis (фризы, хуки, удары, D-030) ──► skill (мастерство), technique (T1–T18)
                                                        │
                       demo::process ──► run::from_demos ──► dataset (чанки postcard+zstd, manifest, players, report)
```

| Модуль | Что делает |
|---|---|
| `ingest` | демка → кадры `ddai_recorder::format::Frame` по одному (`FrameSource`; только свежие снапшоты); `character.tick == 0` → тик снапшота; **сразу** через анонимайзер 8.4a (`player_N` по `(client id, stint)`, клан пуст) |
| `store` | постраничное хранилище таймлайна демки во временном файле (страница 256 кадров, кэш 8 страниц): память не зависит от длины демки (8.4d) |
| `pipeline` | потоково (`recon_table` + `build_stream`): по каждому кадру `LiveWorld` строит точное состояние всех персонажей; вывод фриза без `DDNetCharacter`; ввод за интервал `[t_k, t_{k+1})` (шаги `t_k → t_k+1`, `t_k+1 → t_k+2`) из `ddai_recorder::reconstruct`; реплей `World<f32>` на 2 тика и сравнение со следующим кадром; абляция каналов |
| `replay` | классы `Exact/Within1px/Off`, статистика по каналам, гистограмма ошибки |
| `analysis` | входы во фриз, эпизоды хука, удары молотом, атрибуция D-030 («последний коснувшийся за 50 тиков») |
| `skill` | блоки, самофризы, время во фризе, выживание; рейтинг и корзины Top/Mid/Low/Unranked |
| `technique` | детекторы T1–T5, T8–T15 и настенный/потолочный хук (D-048) |
| `demo` | один демо целиком → `DemoOutput`; правило `target_id` |
| `dataset` | запись чанков, манифест, читатель для тренера с фильтрами |
| `report` | сводный отчёт (JSON + текст) |
| `run` | `from-demos`: обход каталога, карты (встроенная → локальный кэш по crc), пул потоков, детерминированная сборка |
| `synth` (за `test-util`) | синтетические `.map` и `.demo` для сквозных тестов |

Формат, правила и пороги — `docs/formats.md` §20. Итоги на архиве — `docs/EXPERIMENTS.md` E-004.

## Память (8.4d)

Память не растёт с длиной демки и числом игроков: кадры идут потоком в страницы временного файла (`store`), анализ читает их
через маленький кэш, образцы пишутся без тегов, а теги и корзина навешиваются при записи чанков. Результат побайтно равен
результату прежнего конвейера, который держал весь таймлайн в памяти (на демке в 295 тыс. снапшотов — 14,5 ГБ).
Подробности, допущения и доказательство — `docs/formats.md` §20.10. Временные файлы: 77 байт на кадр на демке с немногими игроками и 758 на занятой (223 МБ на демке в 3,3 ч), живут до записи чанков, так что пик диска — сумма по всем демкам (≈272 МБ для v2); каталог —
`--spill-dir` (по умолчанию временный каталог системы), файл удаляется сразу после создания.

## Использование

```bash
# сборка датасета (тяжёлая, вручную; не входит в cargo test)
ddnet-ai dataset from-demos \
    --demos ~/aiddnet/data/demos/chillerdragon/block-06 \
    --maps ~/aiddnet/data/maps \
    --out ~/aiddnet/data/datasets/human/chillerdragon-block06-v1 \
    --name chillerdragon-block06-v1 --source "TwDemosMain/block-06 @ac5e545" --threads 6 [--spill-dir DIR]

ddnet-ai dataset info <dir> --verify       # отчёт + проверка sha256 чанков
ddnet-ai dataset show <dir> <sha-prefix> --tick T --window 30   # кадры вокруг попадания из отчёта
ddnet-ai dataset locate --demos <dir> <sha-prefix>              # найти файл демки (только локально)
```

Читатель для тренера (`Filter` умеет выбирать демки — для разбиения train/val без утечки; `chunks_for` и
`samples_in_chunk` — для перемешивания и параллельного чтения; `DatasetReader: Sync`):

```rust
let reader = ddai_dataset::dataset::DatasetReader::open(dir)?;
let filter = Filter { any_tags: Technique::T9.bit() | Technique::WallHook.bit(),
                      min_skill: SkillBucket::Mid, min_replay: ReplayClass::Within1px,
                      exclude_frozen: true, ..Filter::default() };
for sample in reader.samples(&filter) {
    let s = sample?;          // s.observation: ddai_brain::Observation, s.action: Action, s.meta, s.quality
}
```

## Ники и приватность

Ники, кланы и имена файлов демок не попадают ни в датасет, ни в отчёты: демки называются sha256, игроки — номерами
`player_N` внутри демки (порядок появления, не хэш от ника). Сквозной тест `tests/e2e_synthetic.rs` ищет ники,
клан и имена файлов во всех выходных файлах (в том числе внутри распакованных чанков).

## Что изменено вне крейта

- `ddai-world`: `character_observation` стала `pub` (читать `CharacterObservation` из мира без своего `LiveWorld` на
  каждую точку зрения); в `reckoning::evolve_character_core` исправлена паника при хуке на клиента с id 0
  (`can_keep_hook(0, -1)`, псевдоним слота одно-слотового мира): слот теперь не равен `hooked_player`, и, как в DDNet, хук
  на игрока во временном мире всегда отпускается; найдено на реальных демках, тест проверяет результат DDNet.
- `ddnet-ai`: подкоманда `dataset` (`src/dataset_cmd.rs`), зависимость от этого крейта.

## Тесты

```bash
cargo test -p ddai-dataset            # юнит-тесты и сквозные тесты на синтетических данных
cargo test -p ddai-dataset --release --test e2e_synthetic -- --ignored --nocapture stress   # пик памяти от длины демки
```

Реальные демки в тестах не используются.
