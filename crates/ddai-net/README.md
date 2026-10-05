# ddai-net

Транспортно-независимый низкоуровневый слой протокола Teeworlds 0.6 + DDNet, как его говорят
серверы DDNet 20.x: Huffman-сжатие, упаковка целых/строк, кадрирование пакетов и чанков,
control-сообщения и рукопожатие DDNet «TKEN», sans-IO стейт-машина надёжной доставки
(`Connection`), UUID расширенных сообщений (`NETMSG_EX`).

Второй слой (задача 2.2b, тот же крейт): сгенерированные из `datasrc/network.py` объекты снапшота
и игровые сообщения (`generated/`), hand-written системные сообщения (`sysmsg`, `ExSysMsg` в
`message`), полная обработка снапшотов (мультичастная сборка, дельта, CRC, хранилище по тику —
`delta`/`snapshot`/`assembly`) и типизированный view-API (`view`). Полноценная клиентская сессия
(вход, докачка карты, тайминг ввода) — задача 2.3, не здесь. Крейт ничего не знает про сокеты и
системные часы: время передаётся вызывающим кодом (`now: Duration`), сеть — байтами
(`feed`/`flush`).

Байтовые раскладки задокументированы в `docs/formats.md` (§9 «Протокол 0.6+DDNet: низкий уровень»,
§13 «Протокол DDNet 20.1: сообщения и снапшоты»); там же — как получена фикстура реального трафика
в `tests/fixtures/` и как перегенерировать `generated/` (`tools/ddnet-protocol-gen/`).

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
| `generated` | **Сгенерированный** (`tools/ddnet-protocol-gen/generate.py`, не редактировать руками) код из `datasrc/network.py`: `enums`/`objects`/`messages` — все объекты снапшота и игровые сообщения DDNet 20.1, decode/encode/валидация |
| `sysmsg` | Hand-written системные сообщения (`NETMSG_*` без описания в `datasrc/`): `INFO`/`MAP_CHANGE`/`MAP_DATA`/`SNAP*`/`INPUTTIMING`/`INPUT`/`RCON_*`/… |
| `intstr` | DDNet-кодирование «int-string» (`CNetObj_ClientInfo`'s имя/клан/скин) — 4 байта на `i32`, побайтовый сдвиг +128 |
| `delta` | Дельта снапшота: `unpack_delta` (порт `CSnapshotDelta::UnpackDelta`, включая бюджет `CSnapshot::MAX_SIZE` — 64 КиБ, ревью раунда 1 F3), `create_delta` (только для тестов), статические размеры типов |
| `snapshot` | `Snapshot`/`SnapshotItem`, CRC, резолв `ex`-типов через дескрипторы `NETOBJTYPE_EX` (≥ 4 int, как у самого DDNet — ревью раунда 1 F6), `SnapshotStorage` по тику |
| `assembly` | Мультичастная сборка `NETMSG_SNAP`/`SNAPSINGLE`/`SNAPEMPTY` в готовый `Snapshot` (`SNAPSMALL` намеренно не участвует — настоящий клиент DDNet его тоже игнорирует, ревью раунда 1 F7), CRC-проверка, ресинк, `SnapAssembler::reset()` — сбросить хранилище/тики при смене карты (вызывать на каждом `ENTERGAME`, не только первом — ревью раунда 1 F1; без этого снапшоты новой карты навсегда «протухшие») |
| `tuning` | `Sv_TuneParams`/`Sv_TeamsState(Legacy)` — в `datasrc/network.py` у них нет полей (сам DDNet-клиент разбирает их руками, `gameclient.cpp`), поэтому это hand-written порт: 47 именованных полей `CTuningParams` в порядке `tuning.h`, их дефолты (бит-в-бит те же `float`→fixed-point, что и `CTuneParam::operator=`), останов на первой ошибке пакера с сохранением дефолтов на непрочитанный хвост; аналогично для команд по флоку/DDrace-командам (ревью раунда 1 F2) |
| `message` | Расширенные (`ex`) системные сообщения (`MAP_DETAILS`/`CAPABILITIES`/`CLIENTVER`/`WHATIS`/…), единый `Registry` UUID-имён, `decode`/`encode` верхнего уровня — перехватывает `Sv_TuneParams`/`Sv_TeamsState(Legacy)` (числовой id и `ex`-имя `teamsstate@netmsg.ddnet.tw`) перед сгенерированной таблицей и отдаёт их через `tuning` (см. выше, F2) |
| `view` | Типизированный view над `Snapshot`: персонажи (слиты с `DDNetCharacter`), игроки, снаряды/лазеры/пикапы, игровая инфа |
| `server_command`, `owner_chat` | Единственный исходящий чат (`Cl_Say`): `ServerCommand::Kill` (`/kill`, D-078, задача 4.6) и `OwnerSay { team, text: OwnerText }` — строка, набранная владельцем на сайте (D-094, задача 4.9; `OwnerText::new(&OwnerChannel, …)`: способность `OwnerChannel` выдаётся один раз на процесс; не пусто, ≤ 255 байт, без управляющих и невидимых символов, не строка-ловушка 20.1; **с 4.9b начало с `/` разрешено** (серверные команды владельца, D-094); `OwnerText::check` — только проверка); `encode_cl_say` остаётся `pub(crate)`, байты Cl_Say строятся только здесь (`OwnerPayload`), `compile_fail`-доктесты это проверяют |
| `serverinfo` | Connectionless `SERVERINFO` (серверный браузер): `"iext"`/`"inf3"` целиком (включая список игроков), `"dtsf"`/`"iex+"` — только распознавание |

## Тесты

- Модульные тесты в каждом файле, включая перенесённые тест-векторы DDNet
  (`src/test/{huffman,compression,packer,chunk_header,network}_test.cpp`).
- `tests/oracle_libtw2.rs` — дифференциальные тесты против `libtw2-huffman`/`libtw2-packer`
  (≥ 100k случайных входов на каждое свойство, `cargo test --release` — быстрее).
- `tests/robustness.rs` — фаз-тесты (> 10⁶ случайных/искажённых входов суммарно) в декодер
  пакетов, `Huffman::decompress`, `Unpacker`, `Connection::feed`: без паник.
- `tests/capture.rs` — реальный захват трафика, уровень пакетов/чанков (см. `docs/formats.md` §9.8).
- `tests/lossy_link.rs` — симуляция ненадёжной сети (потери/переупорядочивание/дублирование) между
  двумя `Connection` (роли клиента и сервера), с детерминированным псевдослучайным генератором и
  поддельными часами.
- `tests/oracle_libtw2_snapshot.rs` — дифференциальные тесты против `libtw2-snapshot` (по 10k
  случаев в обе стороны: наш декодер против их кодировщика и наоборот, см. `docs/formats.md` §13.7).
- `tests/robustness_2_2b.rs` — фаз-тесты (> 10⁶ случайных/искажённых входов суммарно) в декодер
  сообщений, распаковку дельты и сборку `NETMSG_SNAP`: без паник.
- `tests/real_traffic.rs` — та же фикстура 2.2a, но на уровень выше: чанки → сообщения → снапшоты,
  0 несовпадений CRC, траектория собственного тика бота (см. `docs/formats.md` §13.8).
- `tests/map_change_real_traffic.rs` — компактная (< 500 КБ) собственная фикстура с настоящей сменой
  карты (`Copy Love Box` → `BlmapChill` → `Copy Love Box`) на локальном сервере: без `reset()` на
  `ENTERGAME` бо́льшая часть снапшотов новой карты — `Event::Stale` (воспроизводит находку F1), с
  `reset()` — восстанавливаются все, 0 несовпадений CRC; там же проверены реальные значения
  `Sv_TuneParams` этого сервера (F2).
- `tests/regenerate_check.rs` — `#[ignore]`d (нужно настоящее, некоммитящееся дерево исходников
  DDNet 20.1): перегенерировать `generated/` и побайтово сравнить с закоммиченным (ревью раунда 1
  F5, критерий приёмки №1). Запуск: `cargo test -p ddai-net --test regenerate_check -- --ignored`.
- `tests/no_third_party_map_bytes.rs` — сканирует оба закоммиченных `.dat`-фикстуры и проверяет,
  что ни в одном `MAP_DATA`-чанке не осталось ни одного не-нулевого байта (ревью раунда 2,
  находка F10 — постоянная CI-проверка результата `examples/strip_map_data.rs`, а не разовый
  прогон).

```bash
cargo test -p ddai-net                     # всё, кроме тяжёлых прогонов на release-скорости
cargo test -p ddai-net --release --test oracle_libtw2   # 100k×N случаев быстрее в release
cargo clippy -p ddai-net --all-targets -- -D warnings
cargo fmt -p ddai-net --check
```
