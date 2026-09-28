# ddai-demo — чтец демок DDNet `.demo` (задача 8.4b)

Безопасный (без `unsafe`), точный ридер клиентских и серверных демок DDNet 0.6+DDNet (версии
заголовка 4, 5, 6 — версия 3 читается тем же кодом как бесплатное следствие, но вне корпуса
задачи). Порт `engine/shared/demo.{h,cpp}` + `engine/shared/snapshot.{h,cpp}` DDNet 20.1
(пиннутый коммит `c9d208138f85755521f16a0096b6fe036c5c8698`) поверх низкоуровневого протокола
`ddai-net` (Хаффман, вариант-инт пак, снапшот-дельты, типизированный view, сообщения — задачи
2.2a/2.2b). Полное описание формата с цитатами строк C++-оригинала — `docs/formats.md` §18.

Главная цель — превратить записи чужой игры (например, архив ChillerDragon,
`~/aiddnet/data/demos/chillerdragon/block-06/`, разрешение D-040) в потиковую последовательность
полностью раскрытых снапшотов и игровых сообщений для дальнейшего восстановления траекторий и
вводов (задача 8.4a).

## Пример

```rust
let bytes = std::fs::read("game.demo")?;
let demo = ddai_demo::Demo::parse(&bytes)?;
println!("map: {} ({} bytes)", demo.map.name, demo.map.size);

let map = ddai_map::load_map(demo.map_bytes())?; // задача 1.4

for tick in demo.ticks() {
    let tick = tick?;
    if let Some(snap) = &tick.snapshot {
        let view = ddai_net::view::View::new(snap);
        for c in view.characters() {
            println!("tick {}: id={} pos=({},{})", tick.tick, c.id, c.character.x, c.character.y);
        }
    }
}
```

## Устройство

- `header.rs` — `CDemoHeader` (176 байт) + `CTimelineMarkers` + SHA256-расширение; определяет,
  где в файле начинаются байты вложенной карты.
- `reader.rs` — `TickIter`: ленивый итератор по потоку чанков (тик-маркеры, снапшоты, дельты,
  сообщения), Хаффман + вариант-инт распаковка через `ddai_net::{huffman,packer}`, дельты через
  `ddai_net::delta::unpack_delta` (демка — своя, «локальная» база дельт, не путать с сетевой).
- `rawsnapshot.rs` — разбор *сырого* байтового макета `CSnapshot` (полные снапшоты в демке
  хранятся не как дельта, а как дамп памяти настоящей C++-структуры) в
  `ddai_net::snapshot::Snapshot`.
- `error.rs` — `HeaderError`/`TickIterError`: везде `Result`, ни одной паники на враждебном вводе.
  Фатальная ошибка чанка не роняет уже раскрытые данные ЭТОГО тика (сначала частичный `Tick`,
  ошибка — следующим вызовом итератора, `TickIterError`'s doc comment).
- `testutil.rs` (за `#[cfg(any(test, feature = "test-util"))]`) — синтетические байты демки,
  собранные собственными энкодерами `ddai-net` (никаких сторонних файлов), для юнит-тестов этого
  крейта и фаззинга `tests/robustness.rs`/`tests/hostile.rs`.

## Точность

Сверено побайтово (элементы снапшотов: ключ + сырые `i32` данные, в порядке файла) с настоящим
чтецом демок DDNet 20.1, слинкованным напрямую (`tools/ddnet-oracle/demo2json`,
`docs/formats.md` §18.5) — `tools/ddnet-oracle/parity_check_demo.sh`. Результаты по всему архиву
ChillerDragon (228 демок) и образцам `public-samples/` — в отчёте о билде задачи 8.4b.

## CLI

```bash
ddnet-ai demo info <file>                       # заголовок, карта, диапазон тиков
ddnet-ai demo dump <file> [--anonymize | --raw]  # потиковый вывод (типизированный или машинный)
ddnet-ai demo stats <file-or-dir> [--anonymize]  # агрегаты по одной демке или каталогу рекурсивно
```

Ники видны только в локальном выводе; `--anonymize` заменяет их псевдонимом, пронумерованным в
порядке первого появления (не обратимым хэшем — устойчивый между запусками хэш с фиксированным
ключом вскрывается перебором списка ников), D-040. `--raw` печатает сырые элементы снапшота как
есть (включая упакованные `ClientInfo`), поэтому с `--anonymize` несовместим — `clap` откажет
сразу, а не молча проигнорирует флаг.

## Тесты

```bash
cargo test -p ddai-demo --features ddai-demo/test-util                                 # юнит-тесты
cargo test -p ddai-demo --features ddai-demo/test-util --test hostile                  # регрессии F2/F5
cargo test -p ddai-demo --release --test robustness --features ddai-demo/test-util     # фаззинг синт. демки, ~10^6 случаев
cargo test -p ddai-demo --release --test fuzz_real_bytes --features ddai-demo/test-util \
    -- --ignored --nocapture                                                           # фаззинг РЕАЛЬНЫХ байт, ~10^6 случаев
```

Тесты на реальном корпусе — `#[ignore]`, требуют данные на диске (не в git — `.gitignore`):
демки ChillerDragon и `~/aiddnet/data/demos/public-samples/`.
