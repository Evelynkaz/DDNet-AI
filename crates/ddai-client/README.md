# ddai-client

Клиентская сессия DDNet 20.x (задача 2.3): sans-IO `Session` (последовательность входа, снапшоты,
тайминг ввода, единственный проверяемый исходящий путь для игровых сообщений) над `ddai-net`
(2.2a/2.2b) плюс `driver` — реальный поток с UDP-сокетом (лимит попыток подключения, backoff,
переход по redirect, кэш карт на диске, публичный `Client`).

Подробное описание алгоритмов, join-последовательности с цитатами file:line и живых результатов —
`docs/formats.md` §14. Здесь — только карта модулей.

## Модули

| Модуль | Что |
|---|---|
| `session` | Sans-IO `Session`: `connect`/`feed`/`flush`/`take_events`/`set_input`/`disconnect`, последовательность входа (TKEN → `CLIENTVER`/`INFO` → `CAPABILITIES`/`MAP_DETAILS`/`MAP_CHANGE` → карта → `READY`/`CON_READY` → `Cl_StartInfo` → `Sv_ReadyToEnter` → `ENTERGAME` → снапшоты), `ClientConfig`, `SessionEvent`, приобретение карты (`supply_cached_map`/протокольная докачка), единственный охраняемый путь отправки игровых сообщений |
| `driver` | Реальный поток: `Client::connect`/`set_input`/`disconnect`/`events`, лимит 5 попыток/20 с на адрес (на весь процесс), backoff 1с→30с после потери соединения, redirect не больше одного раза, никогда не переподключаться после кика/бана, кэш карт на диске (файловый I/O — здесь, не в `Session`) |
| `timing` | Тайминг ввода: `InputTiming` (продвижение предсказанного тика, обратная связь `NETMSG_INPUTTIMING`), `MarginStats`/`MarginSummary` (распределение задержек) |
| `smooth_time` | Порт `CSmoothTime` (`smooth_time.{h,cpp}`) в sans-IO форме |
| `allowlist` | Список разрешённых исходящих игровых сообщений (защита от `Cl_Say`, D-007) — единственная функция-охрана `check`, работает на сырых байтах, не на типе Rust |
| `map_cache` | Путь/чтение/запись кэша карт (`<cache_dir>/<имя>_<sha256-hex>.map`), валидация имени карты (`str_valid_filename`) |

## CLI

`ddnet-ai play --server <адрес> --name <имя> --brain idle|circle --duration <с>` (крейт `ddnet-ai`,
`src/play_cmd.rs`) — тонкая обвязка, которой это всё проверено против локального сервера DDNet 20.1.

## Тесты

- Модульные тесты в каждом файле, включая сквозной `session::tests::
  full_join_sequence_reaches_in_game_and_first_snapshot` — вся последовательность входа плюс
  первый снапшот, над рукописным сервером-двойником в памяти (без реальных сокетов).
- `tests/redirect_double.rs` — переход по `redirect@ddnet.org` и защита от петли, над настоящими
  loopback UDP-сокетами (свой маленький сервер-двойник на Rust, не настоящий DDNet-Server — у
  DDNet 20.1 нет админ-команды, которая шлёт `NETMSG_REDIRECT`).
- `tools/e2e/session.sh` — восемь сценариев (a-h) задачи 2.3 против настоящего локального сервера
  (`ddnet-local.service`); результаты и живые логи — `docs/formats.md` §14.9 и BUILD REPORT.

## Решения/находки, зафиксированные в коде (см. doc-комментарии на местах)

- `Cl_IsDDNetLegacy` расшифрован сервером и клиентом вручную (как `Sv_TuneParams`/`Sv_TeamsState`
  из задачи 2.2b) — сгенерированная в `ddai-net` структура пуста по ошибке генератора, здесь
  собран правильный payload вручную (`session::Session::send_post_enter_extras`).
- HTTPS-докачка карты не реализована (только in-protocol) — правило живой игры этой задачи
  (только 127.0.0.1), сервер сам её и не предлагает.
- `systemctl restart` шлёт клиентам настоящий `NETMSG_CLOSE("Server shutdown")` до выхода процесса
  — с точки зрения провода не отличить от кика; для проверки именно пути «таймаут → реконнект»
  нужен `kill -9` по основному процессу юнита, не `restart`.
