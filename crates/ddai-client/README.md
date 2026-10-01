# ddai-client

Клиентская сессия DDNet 20.x (задача 2.3): sans-IO `Session` (последовательность входа, снапшоты,
тайминг ввода, единственный проверяемый исходящий путь для игровых сообщений) над `ddai-net`
(2.2a/2.2b) плюс `driver` — реальный поток с UDP-сокетом (лимит попыток подключения, backoff,
переход по redirect, кэш карт на диске, публичный `Client`).

Подробное описание алгоритмов, join-последовательности с цитатами file:line и живых результатов —
`docs/formats.md` §14 (задача 2.3) и §16 (задача 8.4a — наблюдатель, `Cl_SetTeam`, `live_servers`).
Здесь — только карта модулей.

## Модули

| Модуль | Что |
|---|---|
| `session` | Sans-IO `Session`: `connect`/`feed`/`flush`/`take_events`/`set_input`/`disconnect`, последовательность входа (TKEN → `CLIENTVER`/`INFO` → `CAPABILITIES`/`MAP_DETAILS`/`MAP_CHANGE` → карта → `READY`/`CON_READY` → `Cl_StartInfo` → `Sv_ReadyToEnter` → `ENTERGAME` → снапшоты), `ClientConfig`, `SessionEvent`, приобретение карты (`supply_cached_map`/протокольная докачка), единственный охраняемый путь отправки игровых сообщений; задача 8.4a добавила `request_team` (`Cl_SetTeam`), `SessionEvent::SnapshotData`/`InputSent` (оба опциональны/аддитивны — см. `docs/formats.md` §16.2.1) |
| `driver` | Реальный поток: `Client::connect`/`set_input`/`disconnect`/`events`, лимит 5 попыток/20 с на адрес (на весь процесс), backoff 1с→30с после потери соединения, redirect не больше одного раза, никогда не переподключаться после кика/бана, кэш карт на диске (файловый I/O — здесь, не в `Session`); задача 8.4a добавила `Client::set_team`/`ClientEvent::OwnTeam` |
| `timing` | Тайминг ввода: `InputTiming` (продвижение предсказанного тика, обратная связь `NETMSG_INPUTTIMING`), `MarginStats`/`MarginSummary` (распределение задержек) |
| `smooth_time` | Порт `CSmoothTime` (`smooth_time.{h,cpp}`) в sans-IO форме |
| `allowlist` | Список разрешённых исходящих игровых сообщений (защита от `Cl_Say`, D-007) — единственная функция-охрана `check`, работает на сырых байтах, не на типе Rust |
| `map_cache` | Путь/чтение/запись кэша карт (`<cache_dir>/<имя>_<sha256-hex>.map`), валидация имени карты (`str_valid_filename`) |
| `live_servers` | Задача 8.4a, D-027/D-038: `~/aiddnet/data/live-servers.toml` (вне git) — предохранитель «живая игра и запись разрешены только на этих не-loopback серверах, под этим ником»; loopback разрешён всегда без записи в файле |

## CLI

- `ddnet-ai play --server <адрес> --name <имя> --brain idle|circle|random-scripted --duration <с>`
  (крейт `ddnet-ai`, `src/play_cmd.rs`) — тонкая обвязка, которой это всё проверено против
  локального сервера DDNet 20.1.
- `ddnet-ai record --server <адрес> --name Muha --duration <с> --out <каталог>` (задача 8.4a,
  `src/record_cmd.rs`) — клиент-наблюдатель: заходит спектатором, не шлёт ничего, кроме
  нейтрального ввода, записывает в rec v1 (`ddai-recorder`, `docs/formats.md` §16).

## Тесты

- Модульные тесты в каждом файле, включая сквозной `session::tests::
  full_join_sequence_reaches_in_game_and_first_snapshot` — вся последовательность входа плюс
  первый снапшот, над рукописным сервером-двойником в памяти (без реальных сокетов).
- `tests/redirect_double.rs` — переход по `redirect@ddnet.org` и защита от петли, над настоящими
  loopback UDP-сокетами (свой маленький сервер-двойник на Rust, не настоящий DDNet-Server — у
  DDNet 20.1 нет админ-команды, которая шлёт `NETMSG_REDIRECT`).
- `tools/e2e/session.sh` — восемь сценариев (a-h) задачи 2.3 против настоящего локального сервера
  (`ddnet-local.service`); результаты и живые логи — `docs/formats.md` §14.9 и BUILD REPORT.
- `driver::event_channel::tests::control_events_are_never_evicted_when_the_queue_is_full_of_control_events_only`
  — задача 8.4a, перенос находки ревью 2.3 (F15): управляющие события не вытесняются, даже когда
  очередь целиком состоит из них (не только когда среди них есть хоть одно вытесняемое).

## Решения/находки, зафиксированные в коде (см. doc-комментарии на местах)

- `Cl_IsDDNetLegacy` расшифрован сервером и клиентом вручную (как `Sv_TuneParams`/`Sv_TeamsState`
  из задачи 2.2b) — сгенерированная в `ddai-net` структура пуста по ошибке генератора, здесь
  собран правильный payload вручную (`session::Session::send_post_enter_extras`).
- HTTPS-докачка карты не реализована (только in-protocol) — правило живой игры этой задачи
  (только 127.0.0.1), сервер сам её и не предлагает.
- `systemctl restart` шлёт клиентам настоящий `NETMSG_CLOSE("Server shutdown")` до выхода процесса
  — с точки зрения провода не отличить от кика; для проверки именно пути «таймаут → реконнект»
  нужен `kill -9` по основному процессу юнита, не `restart`.

## Дополнения задачи 4.1 (живой бот `ddai-bot`)

- `Client::kill()` / `Session::request_kill` — сообщение протокола `Cl_Kill` (в белом списке с 2.3); `Client::show_distance(x, y)`
  / `Session::request_show_distance` — `Cl_ShowDistance` на лету. Чата по-прежнему нет ни в одном пути.
- `ClientConfig::emit_outgoing_audit` -> `SessionEvent::OutgoingGame { label, accepted }` на каждое исходящее игровое
  сообщение (выключено по умолчанию): аудит «нет чата» в e2e.
- `LiveWorldSnapshot` получил `players`, `pred_tick` (тик последнего отправленного `NETMSG_INPUT`, `Session::pred_tick()`)
  и `arrived` (момент сборки снапшота).
- `Client::set_input_for_snapshot(input, arrived)` + `ClientEvent::InputLatency { tick, since_snapshot }`: драйвер сообщает
  задержку «снапшот пришёл -> ввод ушёл в сокет» для первого ввода с этим решением.
- `POLL_TIMEOUT` драйвера 10 мс -> 2 мс: решение бота подхватывалось не чаще раза за `recv`, и `advance()` срабатывал
  до 10 мс позже. Замер — `docs/formats.md` §21.6.
