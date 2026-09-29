# ddai-recorder

Формат записи наблюдателя rec v1 (задача 8.4a): собственный, версионированный, с zstd-сжатыми
чанками и sha256-проверками (`docs/formats.md` §16 — там подробности алгоритмов и живые числа,
здесь — только карта модулей). Живой клиент-наблюдатель (`ddnet-ai record`) и офлайн-CLI
(`ddnet-ai rec inspect|reconstruct|anonymize`) живут в крейте `ddnet-ai`
(`src/record_cmd.rs`/`src/rec_cmd.rs`) — этот крейт сам по себе не делает сетевого ввода-вывода и
не парсит аргументы командной строки, только формат и офлайн-анализ.

## Модули

| Модуль | Что |
|---|---|
| `format` | rec v1: `Header`, `Frame` (`Snapshot`/`GameEvent`), `CharacterRecord`/`PlayerRecord`, `RecordedGameMessage` — кодирование/декодирование одной записи |
| `binio` | Маленькие little-endian примитивы (`Writer`/`Reader`), на которых стоит `format` — не универсальный сериализатор, формат достаточно простой и фиксированный, чтобы `serde`/`bincode` не окупились |
| `writer` | `RecordingWriter` — потоковая запись на диск: буферизует кадры, сбрасывает zstd-сжатый чанк с sha256 по достижении порога, пишет sha256-сайдкар всего файла при `finish()` |
| `reader` | `RecordingReader` — обратное чтение по чанкам (не грузит весь файл в память), плюс `verify_whole_file_sha256` |
| `anonymize` | `Anonymizer` — задача 8.4a, критерий 2: `--anonymize`-экспорт, стабильные `player_N` id вместо ников, текст чата вычищен |
| `reconstruct` | Задача 8.4a, критерий 3: траектория по `character.tick` + оценённый ввод (`direction`/`aim` точно, `jump`/`hook`/`fire` — оценка, каждое поле помечено `Confidence::{Exact,Estimated}`) |

## Тесты

Модульные тесты в каждом файле (round-trip кодирования, порча байт не паникует, дедупликация
повторных `character.tick`, фриз обнуляет `direction`/`jump` — см. `reconstruct`'s собственные
тесты). Живая валидация на реальном сервере — `tools/e2e/record.sh`/
`crates/ddnet-ai/tests/e2e_record.rs` (задача 8.4a).
