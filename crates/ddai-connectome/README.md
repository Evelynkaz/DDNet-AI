# ddai-connectome

Утилита предобработки коннектома MaleCNS v1.0: скачивание и проверка «минимального набора» из
3 файлов Arrow Feather (docs/research/fly-data.md §6.1), потоковое чтение по батчам и сборка
компактных внутренних таблиц (нейроны, типы, нейромедиаторы, рёбра) + статистический отчёт.

Отдельный крейт/бинарник, **не зависимость `ddnet-ai`**: тяжёлый `arrow` (≈100 транзитивных
пакетов) не должен линковаться в бота. Сеть нужна только команде `fetch`; `inspect`,
`build-tables` и `stats` работают полностью офлайн.

## Почему `reqwest::blocking`

`fetch` скачивает ровно 3 файла последовательно (проверка HEAD → докачка Range → проверка md5 и
sha256 → атомарный rename). Конкурентности эксплуатировать негде, а больше никакая часть этого
бинарника не нуждается в асинхронном рантайме — блокирующий клиент даёт прямой, линейный код без
`tokio::main` и без ручного управления футурами. Другие бинарники проекта (веб-интерфейс бота,
D-012/rust-stack.md §1) используют `tokio`/`axum`; это не противоречит выбору здесь, так как это
разные бинарники с разными требованиями.

## Команды

```bash
# 1. Скачать и проверить (сеть). Второй запуск ничего не скачивает, если файлы уже верны.
ddai-connectome fetch --manifest manifests/connectome.toml --dest ~/aiddnet/data/connectome/raw

# то же самое, но дополнительно дописать в манифест sha256 файлов, для которых он ещё не известен
ddai-connectome fetch --manifest manifests/connectome.toml --dest ~/aiddnet/data/connectome/raw --update-sha256

# 2. Посмотреть схему/число строк/сжатие любого .feather (офлайн)
ddai-connectome inspect ~/aiddnet/data/connectome/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather

# 3. Собрать компактные таблицы (офлайн; читает только --raw, сеть не нужна)
ddai-connectome build-tables --raw ~/aiddnet/data/connectome/raw --out ~/aiddnet/data/connectome/tables

# 4. Статистика по компактным таблицам, отчёт в Markdown на русском (офлайн)
ddai-connectome stats --tables ~/aiddnet/data/connectome/tables --out ~/aiddnet/data/connectome/tables/report.md
```

`build-tables` ожидает в `--raw` файлы с точными именами из `manifests/connectome.toml`:
`body-annotations-male-cns-v1.0-minconf-0.5.feather`, `body-neurotransmitters-male-cns-v1.0.feather`,
`connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather`.

## Манифест (`manifests/connectome.toml`)

Каждая запись закрепляет файл по GCS `generation` (перезапись объекта = 404, а не молча новые
данные), размеру и md5 (из HEAD), плюс наш собственный sha256, посчитанный после первой
успешной проверенной загрузки. Пустая строка `sha256 = ""` означает «пока не закреплён» — именно
её заполняет `--update-sha256`, никогда не перезаписывая уже проставленное значение.

## Компактные таблицы

`build-tables` пишет один файл `<out>/connectome.tables` (postcard, сжатый zstd, версионированный
заголовок с sha256 всех трёх входных файлов): `neurons` (плотный индекс, отсортирован по
`bodyId`), `types`, `neuron_nt` (predicted_nt + уверенность на нейрон, параллельно `neurons`),
`edges` (только между `Traced`-нейронами, отсортированы по `(post_idx, pre_idx)`, автапсы
отброшены и посчитаны отдельно). Все словари строк и порядок вывода детерминированы (сортировка,
без зависимости от порядка обхода `HashMap`).

## Тесты

`cargo test -p ddai-connectome` — полностью офлайн. Модульные тесты (`src/*.rs`) проверяют
хеширование, разбор `x-goog-hash`, разбор/точечное обновление манифеста, обнаружение испорченного
локального файла (несовпадение размера/md5/sha256) и точность преобразования `double → i64`.
Интеграционный тест (`tests/build_tables.rs`) строит крошечные Feather-файлы тем же писателем
`arrow` со сжатием LZ4 (тот же кодек, что у настоящих файлов) и прогоняет через них весь
`build-tables`, сверяя каждое поле с посчитанными вручную ожиданиями.
