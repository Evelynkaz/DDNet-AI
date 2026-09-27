# ddai-net

Транспортно-независимый низкоуровневый слой протокола Teeworlds 0.6 + DDNet, как его говорят
серверы DDNet 20.x: Huffman-сжатие, упаковка целых/строк, кадрирование пакетов и чанков,
control-сообщения и рукопожатие DDNet «TKEN», sans-IO стейт-машина надёжной доставки
(`Connection`), UUID расширенных сообщений (`NETMSG_EX`).

Снапшоты, сгенерированные игровые сообщения и полноценная клиентская сессия — задачи 2.2b/2.3, не
здесь. Крейт ничего не знает про сокеты и системные часы: время передаётся вызывающим кодом
(`now: Duration`), сеть — байтами (`feed`/`flush`).

Байтовые раскладки задокументированы в `docs/formats.md` (раздел «Протокол 0.6+DDNet: низкий
уровень»); там же — как получена фикстура реального трафика в `tests/fixtures/`.

## Решение D-029: своя реализация, а не вендоринг

DDNet 20.1 (C++, `~/aiddnet/build/ddnet-20.1/src`, пиннутый коммит
`c9d208138f85755521f16a0096b6fe036c5c8698`) и `libtw2` (MIT/Apache) — эталоны и тест-оракулы, не
код для копирования. `libtw2` подключён только как dev-зависимость (differential-тесты Huffman и
варинта, `tests/oracle_libtw2.rs`) и никогда не линкуется в обычную сборку.

Если в файле портирован алгоритм/таблица из DDNet C++ (таблица частот Huffman, битовые раскладки),
в начале файла — оригинальное zlib-уведомление Teeworlds/DDNet + пометка «altered» (см.
`src/huffman.rs`, `src/packet.rs`).

## Модули

| Модуль | Что |
|---|---|
| `huffman` | Фиксированное Huffman-дерево DDNet; `compress`/`decompress` с ограничением по размеру буфера, не зацикливается на испорченном входе |
| `packer` | Variable-length int (упаковка/распаковка), строки с санитайзингом (`SANITIZE`/`SANITIZE_CC`/`SKIP_START_WHITESPACES`), сырые данные; `Packer`/`Unpacker` — как `CPacker`/`CUnpacker`, без паник |
| `packet` | Заголовок пакета (флаги/ack/num_chunks), заголовок чанка, connless-кадры, `NET_MAX_*`; кодирование/декодирование в обе стороны |
| `control` | Control-сообщения (`KEEPALIVE`/`CONNECT`/`CONNECTACCEPT`/`ACCEPT`/`CLOSE`) и байтовая раскладка рукопожатия TKEN — чистые функции, без состояния соединения |
| `conn` | Sans-IO `Connection`: `connect`/`accept`/`feed`/`flush`/`send_chunk`/`disconnect`; ack/seq wraparound, resend-буфер, keepalive/timeout по переданным часам |
| `uuid` | UUID v3 (md5 неймспейса + имени) для `NETMSG_EX`, таблица зарегистрированных имён, кодирование id сообщения |

## Тесты

- Модульные тесты в каждом файле, включая перенесённые тест-векторы DDNet
  (`src/test/{huffman,compression,packer,chunk_header,network}_test.cpp`).
- `tests/oracle_libtw2.rs` — дифференциальные тесты против `libtw2-huffman`/`libtw2-packer`
  (≥ 100k случайных входов на каждое свойство, `cargo test --release` — быстрее).
- `tests/robustness.rs` — фаз-тесты (> 10⁶ случайных/искажённых входов суммарно) в декодер
  пакетов, `Huffman::decompress`, `Unpacker`, `Connection::feed`: без паник.
- `tests/capture.rs` — реальный захват трафика (см. `docs/formats.md` §8).
- `tests/lossy_link.rs` — симуляция ненадёжной сети (потери/переупорядочивание/дублирование) между
  двумя `Connection` (роли клиента и сервера), с детерминированным псевдослучайным генератором и
  поддельными часами.

```bash
cargo test -p ddai-net                     # всё, кроме тяжёлых прогонов на release-скорости
cargo test -p ddai-net --release --test oracle_libtw2   # 100k×N случаев быстрее в release
cargo clippy -p ddai-net --all-targets -- -D warnings
cargo fmt -p ddai-net --check
```
