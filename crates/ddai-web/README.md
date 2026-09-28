# `ddai-web` — веб-сервер бота

axum + WebSocket, только `127.0.0.1` (Caddy — единственная публичная точка, задача 5.3). Вход по
паролю (argon2id), серверные сессии, CSRF/Origin-проверки, заголовки безопасности (задача 5.1) —
подробности в `src/auth/*`, `src/http/login.rs`, `src/headers.rs`.

## Живая карта (задача 5.2a)

Вкладка «Игра»: сервер шлёт карту один раз (классифицированную в маленькую палитру «kind» —
`src/live/scene.rs`) и дальше — поток компактных бинарных кадров мира по уже открытому,
аутентифицированному WebSocket. Живого бота ещё нет (задачи 2.3/2.4/4.x), поэтому источник кадров
абстрактный (`crate::live::source::FrameSource`), с одной настоящей реализацией сегодня —
воспроизведение трасс Oracle B (`crate::live::replay::ReplaySource`, задача 1.5).

Архитектура и байтовые форматы — `docs/formats.md`, раздел 14 (после раздела о протоколе DDNet
20.1, задача 2.2b). Коротко:

- `src/live/source.rs` — `FrameSource`, `WorldFrame`/`CharacterState`/`GameEvent`.
- `src/live/scene.rs` — классификация тайлов в 16 «kind» (воздух/стена/фриз/телепорт/…) с
  задокументированным приоритетом game+front+спец-слоёв; сжатие DEFLATE (`flate2`, без
  C-зависимости — тот же выбор, что и у `ddai-map`).
- `src/live/frame.rs` — бинарный формат кадра `live` (версия 1, 12 байт заголовок + 26 байт на
  персонажа); золотая байтовая фикстура + раунд-трип-тесты.
- `src/live/map_resolve.rs` — резолюция реальной `.map`-карты по sha256 **только** внутри
  настроенных `--maps-dir` (никогда не открывает путь из недоверенных метаданных трассы напрямую —
  берётся только имя файла, и байты всё равно перепроверяются по sha256).
- `src/live/replay/` — потоковый ридер trace-b v2 (не грузит файл целиком) + сам источник
  воспроизведения (управление play/pause/speed/seek/next через WS).
- `src/live/hub.rs` — фан-аут на все WS-подключения: `live`-кадры через lossy
  `tokio::sync::broadcast` (дроп при отставании клиента — критерий приёмки «не буферизовать
  безгранично»), остальные сообщения (`map`/`players`/`events`/`replay_status`/`live_error`) —
  через отдельный, менее интенсивный канал.
- `src/http/map.rs` — `GET /api/map/<sha256>` (аутентифицировано; `ETag` + недельный
  `Cache-Control: private, max-age=604800, immutable` — контент неизменен для данного sha256).

**Известное ограничение** (осознанный выбор объёма, не находка): текущее состояние
свитчей/дверей (открыт/закрыт) не транслируется — `MapScene` показывает только «здесь есть
свитч», без динамики; события `HammerHit`/`Teleport` из списка задачи тоже отложены (см. doc
comment в `src/live/replay/mod.rs`) — оба добавляются позже без breaking change формата.

## Запуск

```bash
# Однократно: пароль
cargo run -p ddnet-ai -- web-passwd --data-dir <scratch-dir> --show

# Без живой карты (только вход/статус, задача 5.1):
cargo run -p ddnet-ai -- web --listen 127.0.0.1:7788 --data-dir <scratch-dir>

# С живой картой — воспроизведение корпуса Oracle B (или одного .trb-файла):
cargo run -p ddnet-ai -- web --listen 127.0.0.1:7788 --data-dir <scratch-dir> \
    --replay ~/aiddnet/data/traces/oracle-b/v1/ \
    --maps-dir ~/aiddnet/data/ddnet-server/maps
```

`--maps-dir` можно указывать несколько раз. Никогда не запускать на проде (`deploy/install.sh` —
отдельный, управляемый процесс, задача 5.3) — только на отдельном loopback-порту со scratch
`--data-dir` для тестирования.

## Тесты

```bash
cargo test -p ddai-web                                  # юнит + интеграционные (без реального корпуса)
cargo test -p ddai-web --test replay_real_corpus -- --ignored   # сверка с реальным корпусом Oracle B
                                                                   # (нужен ~/aiddnet/data/traces/oracle-b/v1/)
```

`tools/e2e/live-map.spec.ts` (Playwright, реальный браузер) — см. `tools/e2e/README.md`.
