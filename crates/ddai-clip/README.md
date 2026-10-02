# ddai-clip

Задача 4.3. **Клипы живого бота**: кольцо последних 30 секунд (750 кадров), формат файла, инциденты с настоящими событиями,
автоклип с чисткой каталога и **офлайн-реплей бит в бит** физики бота на `ddai-world` / `ddai-physics`.

| Модуль | Что |
|---|---|
| `format` | формат v1: `magic + zstd(postcard(Clip))`, версия в magic; типы кадров, тиев, вводов, событий |
| `record` | кольцо в фиксированных слотах (запись кадра не выделяет память), `to_clip` |
| `incidents` | `find_incidents` (10 видов, пороги TS), `merge_overlapping`, `summarise` |
| `store` | имена, `pick_incident`, автоклип (кулдауны), чистка каталога (24 / 16, `manual-*` не трогается) |
| `replay` | `replay(clip, map, Resync \| FreeRun)` → `Report` с первым расхождением и его причиной |

Ников в клипе нет (теги 4.1), карты тоже нет (имя + sha256). Клипы в git не попадают. Формат — `docs/formats.md` §24, решение — D-067,
отличия от TS — `docs/research/clips.md`. Просмотр: `ddnet-ai clip info|incidents|dump|replay <файл>`.

```
cargo test -p ddai-clip      # формат, кольцо без аллокаций, 10 видов инцидентов, реплей на клипах, записанных самой физикой
```

Проверка на настоящем сервере — `DDAI_E2E=1 cargo test -p ddai-bot --test e2e_commands -- --ignored --nocapture --test-threads=1`.
