# Кроссплатформенность: инвентаризация (задача 5.5a, D-127)

Сырые заметки задачи: что в воркспейсе зависит от ОС и от C-библиотеки, и как это решено. Сетка поиска — весь воркспейс на 2026-10-09
(`b483c70`, 26 крейтов), включая крейты, которых не было, когда писалась спека (`ddai-oppnet`, веб-запускатель и др.).

## 1. Математика

### 1.1. Что было нужно физике (сервер DDNet 20.1, glibc)

| Вызов в C++ DDNet | Где в Rust | Тип | Теперь |
|---|---|---|---|
| `std::sin/cos(float)` (`direction`, лазеры, пушки, двери) | `Real::sin/cos` в `vmath.rs`, `world.rs`, `world/fixtures.rs`; `ddai-world::projectiles` | `f32` | `ddai_libm::sinf/cosf` |
| `std::atan(float)` (`angle`, старое ускорение) | `Real::atan` в `vmath.rs`, `world.rs` | `f32` | `ddai_libm::atanf` |
| `std::atan2` (в `Tick`: `std::atan2(int, int)`, обе в `double`) | `core::angle_from_target` | `f64` | `ddai_libm::atan2` |
| `std::pow(float, float)` (`VelocityRamp`) | `Real::powf` в `core.rs` | `f32` | `ddai_libm::powf` |
| `log(double)` (`character.cpp:1580`) | `world::max_ramp_speed` | `f64` | `ddai_libm::log` |
| `std::atan2(float, float)` | `Real::atan2` (нигде не вызывается физикой, API) | `f32` | `ddai_libm::atan2f` |
| `pow` в `Math.pow` V8 (паритет планировщика) | `ddai-jsmath::pow` | `f64` | `ddai_libm::pow` |

Полный список функций `ddai-libm`: `sinf`, `cosf`, `atanf`, `atan2f`, `powf`, `hypotf`, `log`, `atan2`, `pow`, `hypot`. `hypot`/`hypotf` физике не
нужны: ими считают расстояния бот, планировщик и клипы (решения, не паритет с сервером).

### 1.2. Остальное в воркспейсе (не `ddai-libm`), по крейтам

Прямые вызовы `std` вне тестовых частей после правок (счётчик вхождений; `census.rs` следит только за первыми 11 крейтами, остальные не
охвачены намеренно):

| Крейт | Вызовы | Почему не заменены |
|---|---|---|
| `ddai-fly` | `ln` 25, `exp` 25, `sin` 26, `cos` 24, `atan2` 9, `ln_1p` 3, `powf` 3, `acos` 2, `tanh` 2, `exp_m1` 2, `cosh` 1 | нейросеть; крейт правит 8.9; `exp`/`tanh`/`ln_1p`/`exp_m1`/`cosh`/`acos` портов нет |
| `ddai-train` | `cos` 6, `ln` 6, `sin` 5, `atan2` 5, `exp` 3, `tanh` 2 | обучение, только Linux |
| `ddai-oppnet` | `exp` 10, `ln` 4, `ln_1p` 2, `cos` 2 | обучение/сеть соперника |
| `ddai-controls` | `tanh` 2 | контрольные сети |
| `ddai-planner::elite`, `ddai-trace::generator`, `ddai-recorder::reconstruct`, `ddai-dataset::pipeline` | `f64` `sin`/`cos` | порта `f64` `sin`/`cos` нет (IBM, большой код и таблицы); для `generator`, `reconstruct`, `pipeline` округление до целого устойчиво (тесты `*_does_not_depend_on_the_last_bit_of_the_math_library`) |
| `ddai-nav::harness` | `exp` | статистика стенда |

`powi`, `sqrt`, `floor`, `round`, `abs`, `mul_add` и деление — точные по IEEE-754 (`powi` — цикл умножений из `compiler-builtins`) и на всех
платформах одинаковы.

## 2. Код, привязанный к Unix

Поиск: `std::os::unix`, `/proc`, `/tmp`, `~/`, `HOME`, `libc`, `mode(0o`, `PermissionsExt`, `O_NOFOLLOW`, сокеты Unix, `systemctl`, `flock`,
сигналы, жёсткие `/`. Каждое вхождение и его решение:

| Что | Где | Решение |
|---|---|---|
| права `0600`/`0700`, `set_permissions`, `.mode(...)` | `ddai-web` `secrets.rs`, `auth/device.rs`; `ddai-client` `timeout_seed.rs`; `ddai-botctl` `relations.rs`; `ddai-oppnet` `live/writer.rs`; `ddai-bot` `control.rs`, `bridge.rs`; `ddai-nav` `memory.rs` | `ddai_os::private`: Unix — те же биты, Windows — ACL «только текущий пользователь» через `icacls` (проверка: ровно одна запись и это пользователь), при неудаче предупреждение, права папки профиля; не доступно всем |
| `O_NOFOLLOW \| O_NONBLOCK`, `ELOOP` (`libc`) | `ddai-client` `safe_file.rs`, `proxy.rs`; `ddai-web` `launch.rs`, `http/servers.rs`, `training/read.rs` | `ddai_os::nofollow`; на Windows `FILE_FLAG_OPEN_REPARSE_POINT` + проверка `is_file()`; `libc` из `ddai-web` и `ddai-client` убран |
| `/dev/urandom` | `ddai-client` `timeout_seed.rs` | `ddai_os::random::fill` (`getrandom`) |
| `/proc/loadavg` | `ddai-bot` `brains.rs`, `ddnet-ai` `fly_cmd.rs` | `ddai_os::host` (Windows: `None`, пропускается) |
| `/proc/stat`, `/proc/self/status` | `ddai-fly` `batched/cores.rs`, `ddai-connectome` | уже необязательны (`None` при отсутствии); тесты-замеры — `#[ignore]` |
| `~/aiddnet/data/...`, `$HOME` (≈20 мест) | `ddai-client`, `ddai-bot`, `ddai-clip`, `ddai-nav`, `ddai-botctl`, `ddai-env`, `ddnet-ai` | `ddai_os::dirs` (`DDNET_AI_DATA_DIR`, `%USERPROFILE%\ddnet-ai\data`, `--data-dir` как был) |
| `UnixListener`/`UnixStream` (std и tokio) | `ddai-bot` `control.rs`, `bridge.rs`; `ddai-web` `control/client.rs`, `live/bot_source.rs` | `ddai_os::ipc` и `ddai_web::local_socket`: на Windows типы-заглушки (`Unsupported`), «бота нет на связи»; решение 5.5b |
| единственный экземпляр на сервер | `ddai-client` `single_instance.rs` | `fd-lock` (кроссплатформенный, `LockFileEx`) — без изменений; `File::try_lock` в `relations.rs` — `std`, тоже |
| сигналы SIGINT/SIGTERM | `ddnet-ai` `play_cmd.rs`, `record_cmd.rs` | `ctrlc` с `termination` уже кроссплатформенен: на Windows Ctrl-C, Ctrl-Break, закрытие консоли, выход из сеанса, выключение |
| systemd, `systemctl`, `/etc/ddnet-ai`, `/run/ddnet-ai`, `/var/lib/ddnet-ai`, `/proc/self` (uid) | `ddnet-ai` `launch_cmd.rs`; умолчания `ddai-web` `launch.rs` | корневой помощник запуска — только Unix: модуль и подкоманда `launch` под `cfg(unix)`; пути в `ddai-web` — умолчания конфигурации, на Windows запускателя нет (каталога запроса нет) |
| юниты systemd | `deploy/` | остаются файлами развёртывания Linux |
| `/` в путях | карты (`map_resolve.rs`), клипы | `Path` понимает оба разделителя; в `map_resolve` на Windows дополнительно отвергаются `\` и `:` (поток данных NTFS); перепроверено: `ddai-client/map_cache.rs` уже отвергает `a\b` |
| 1 МиБ стека главного потока Windows | `ddnet-ai` | `build.rs` просит у линкера 8 МиБ (как Linux) |
| `\r\n` при checkout на Windows | текстовые фикстуры, хэши | `.gitattributes`: `* text=auto eol=lf` |
| UDP `ConnectionReset` на Windows (ICMP) | `ddai-client` `driver.rs` | уже терпится в цикле приёма (`ConnectionRefused \| ConnectionReset`) |
| тесты: `symlink`, `PermissionsExt`, `python3`, `bash`, `sh`, `kill`, `chrt`, `mkfifo`, сокеты Unix | ≈35 файлов тестов | `cfg(unix)`/`cfg(target_os = "linux")` с причиной в файле; кроссплатформенные аналоги — в `ddai-os` (`icacls` на CI) |

## 3. Как проверено без Windows

* `cargo check` и `cargo clippy --workspace --all-targets -- -D warnings` для `x86_64-pc-windows-gnu` (Zig как C-компилятор, `docs/SETUP.md`).
* Ничего из этого не заменяет прогон тестов на Windows: он будет первым на `windows-latest`.
