# Локальный DDNet-сервер (20.1)

Локальный dedicated-сервер DDNet 20.1, собранный из исходников, только для 127.0.0.1. Нужен, чтобы Rust-бот мог
играть «по-настоящему» (join → play → смена карты → реконнект → redirect) без обращения к публичным серверам, и
как основа для будущего Oracle B (in-process сервер для сверки физики, см. `docs/PLAN.md` фаза 2 / `docs/DECISIONS.md`
D-005, D-006).

**Никогда не смотрит в интернет**: `bindaddr 127.0.0.1`, `sv_register 0`, econ только на loopback, ufw не менялся
(только 22/tcp).

## Layout

Всё, что не является кодом/конфигом репозитория, лежит вне репо:

| Что | Где |
|---|---|
| Исходники DDNet 20.1 + build-директория cmake | `~/aiddnet/build/ddnet-20.1/{src,build}` |
| Бинарник сервера | `~/aiddnet/build/ddnet-20.1/build/DDNet-Server` |
| Runtime-хранилище DDNet (storage.cfg, стабильная копия local.cfg, maps/, teehistorian/) | `~/aiddnet/data/ddnet-server/` |
| Логи (stdout/stderr юнита + `logfile`) | `~/aiddnet/data/logs/ddnet-server/` |
| Секреты (rcon/econ-пароли, режим 0600) | `~/aiddnet/data/secrets/ddnet-server-secrets.cfg` |
| Логи end-to-end теста с ботом | `~/aiddnet/data/logs/phase2.1-e2e/` |

В репозитории (`tools/ddnet-server/`):

- `build.sh` — собирает `DDNet-Server` из исходников (тег 20.1, коммит проверяется).
- `setup-runtime.sh` — создаёт `~/aiddnet/data/ddnet-server/{maps,teehistorian}`, `storage.cfg`, копирует карты,
  вызывает `gen-secrets.sh`. Идемпотентен.
- `gen-secrets.sh` — генерирует `~/aiddnet/data/secrets/ddnet-server-secrets.cfg` (rcon+econ пароли, 0600). Не
  перезаписывает существующий файл без `--force`.
- `local.cfg` — коммитится, без секретов; секреты грузятся отдельным `-f` (см. "Runtime-раскладка и секреты"), не
  через `exec`.
- `install-service.sh` — копирует `local.cfg` в стабильный путь, ставит и (по умолчанию) запускает systemd-юнит
  `ddnet-local.service` (fail-closed на отсутствующий конфиг/секреты, IP-сэндбокс - см. "Установка и запуск").
- `econ.py` — клиент econ (stdlib python3): подключается, авторизуется секретом, шлёт одну команду, печатает ответ,
  переживает "занят"-ответы econ ретраями (см. "Смена карт и админ-команды").
- `check-maps.sh` — проверяет, что данная карта грузится в vanilla DDNet-Server 20.1 (стартует сервер в изолированной
  scratch-директории и смотрит, не упал ли процесс).

## Сборка

```bash
tools/ddnet-server/build.sh
```

Делает (см. комментарии в файле для деталей и обоснования каждого пакета):

1. Ставит apt-пакеты, которые нужны именно для **server-only** сборки (`-DCLIENT=OFF -DTOOLS=OFF`): `build-essential`,
   `cmake`, `ninja-build`, `pkg-config`, `git`, `python3`, `libcurl4-openssl-dev`, `libsqlite3-dev`, `libssl-dev`,
   `zlib1g-dev`. Список получен не угадыванием, а разбором `CMakeLists.txt` тега 20.1 (какие `find_package`
   безусловно нужны через `message(SEND_ERROR ...)`, а какие — только при `CLIENT`/`TOOLS`) и подтверждён реальной
   сборкой ровно с этим набором и без него.
2. Использует уже установленный rustup/cargo 1.98.1 из `~/.cargo` (`source ~/.cargo/env`) — **не ставит** второй Rust.
   MSRV DDNet 20.1 (`cmake/FindRust.cmake`) — 1.85.0, у нас 1.98.1, ок.
3. Клонирует DDNet в `~/aiddnet/build/ddnet-20.1/src` (если такой директории ещё нет), делает `git fetch --tags`,
   проверяет, что тег `20.1` резолвится **именно** в коммит `c9d208138f85755521f16a0096b6fe036c5c8698`, и только
   после этого чекаутит его. Если тег указывает на другой коммит или рабочее дерево не чистое — скрипт падает, а не
   собирает непроверенный код.
4. Конфигурирует cmake (`Release`, `-DCLIENT=OFF -DTOOLS=OFF -DDOWNLOAD_GTEST=OFF`) и собирает таргет `DDNet-Server`
   с ограниченным числом потоков (`DDNET_BUILD_JOBS`, по умолчанию 4), измеряя пиковую RSS сборки (сумма RSS
   компилятора/линковщика/cargo/rustc/ninja каждые 0,5 с) и пиковую занятую память системы — обе цифры печатаются и
   пишутся в `~/aiddnet/build/ddnet-20.1/build-mem-summary-<UTC>.txt` (+ обновляемый `...-latest.txt`).

**Идемпотентность**: повторный запуск не клонирует заново (директория уже есть), `git checkout 20.1` — no-op, а
`cmake --build` (ninja) пересобирает только то, что изменилось. Проверено: чистая сборка с нуля — 151/151 файлов,
~45 с, пик RSS сборки ≈ **1,8 ГБ**, пик занятой памяти системы ≈ **2,5 ГБ** (из 15 ГиБ) при `-j4`, сэмплинг раз в
0,5 с (более грубый сэмплинг раз в 2 с в первом прогоне давал заниженный пик ~780 МБ — короткий всплеск ближе к концу
линковки/компиляции просто не попадал в выборку); повторный запуск сразу после — 3/3 файла (только перегенерированный
`git_revision.cpp`), доли секунды. Каждый запуск `build.sh` пишет СВОЙ файл с меткой времени
(`build-mem-summary-<UTC>.txt`), плюс всегда обновляемый `build-mem-summary-latest.txt` — так замер чистой сборки не
затирается тривиальными числами инкрементального перезапуска.

**О `-j4` vs `-j$(nproc)`**: сборка настолько лёгкая (server-only, без клиентского GUI/аудио/видео-кода), что при
измеренном пике ~2,5 ГБ на 15 ГиБ RAM `-j$(nproc)` (8) был бы безопасен и в одиночку. Дефолт всё равно `-j4`
(ограничено, как просит спека), потому что параллельно на этой машине идёт компиляция Rust-крейтов других
агентов-билдеров — `-j4` оставляет им запас по CPU/RAM. Переопределяется через `DDNET_BUILD_JOBS=$(nproc)`.

**Пакеты apt с версиями** (Ubuntu 24.04.5, для `docs/SETUP.md` — этот файл вне `tools/ddnet-server/`, поэтому сам
не редактируется этой задачей, см. BUILD REPORT). Колонка «нужен?» — обязательный (`message(SEND_ERROR...)` в
CMakeLists.txt при отсутствии, для server-only сборки) или зависимость самого apt-пакета (автоматически подтягивается
`apt-get install`, ставить отдельно не нужно):

| Пакет | Версия | Нужен? |
|---|---|---|
| build-essential | 12.10ubuntu1 | обязателен (компилятор) |
| cmake | 3.28.3-1build7 | обязателен |
| ninja-build | 1.11.1-2 | обязателен (backend для cmake) |
| pkg-config | 1.8.1-2build1 | обязателен (`find_package(Curl)` и др. через `pkg_check_modules`) |
| git | 1:2.43.0-1ubuntu7.3 | обязателен |
| python3 | 3.12.3-0ubuntu2.1 | обязателен (кодогенераторы протокола) |
| libcurl4-openssl-dev | 8.5.0-2ubuntu10.15 | обязателен (`SEND_ERROR` без него) |
| libsqlite3-dev | 3.45.1-1ubuntu2.8 | обязателен (`SEND_ERROR` без него) |
| libssl-dev | 3.0.13-0ubuntu3.15 | **опционален** — у DDNet есть чистый bundled-фолбэк для SHA-256/MD5 без OpenSSL (`base/hash_libtomcrypt.cpp` + `base/hash_bundled.cpp`, включаются `#if !defined(CONF_OPENSSL)`, а `CONF_OPENSSL` определяется cmake'ом только когда `CRYPTO_FOUND`); ставим всё равно — 3-строчный dev-пакет, риска в нём нет, а он даёт настоящий OpenSSL SHA-256 вместо самодельного |
| zlib1g-dev | 1:1.3.dfsg-3.1ubuntu2.2 | **не проверено однозначно** — у DDNet в `cmake/FindZLIB.cmake` есть `add_library(zlib ...)` bundled-фолбэк из `src/engine/external/zlib/` (полный исходник zlib), который выглядит подключённым к `DDNet-Server` через `${DEPS}`. Попытка проверить это напрямую (`-DCMAKE_DISABLE_FIND_PACKAGE_ZLIB=ON`) провалилась линковкой (`undefined reference to crc32/uncompress/...`) — но эта проверка сама по себе некорректна: `CMAKE_DISABLE_FIND_PACKAGE_<X>` по документации CMake полностью пропускает выполнение Find-модуля (в т.ч. кастомного `cmake/FindZLIB.cmake` DDNet со всей его bundled-веткой), а не имитирует «системного zlib нет» — так что этот провал ничего не доказывает ни в одну сторону. Более аккуратный способ (спрятать системный zlib только от cmake, не трогая реальные файлы на общей VPS, где от `libz.so` зависят curl/sqlite/python) не найден в отведённое время. Пакет остаётся в списке как заведомо рабочая, протестированная конфигурация — если кто-то захочет подтвердить, что можно и без него, тестировать в одноразовом контейнере, а не на этой машине |
| *(автоматически, зависимости выше)* cmake-data 3.28.3-1build7, pkgconf/pkgconf-bin/libpkgconf3 1.8.1-2build1, libjsoncpp25 1.9.5-6build1, librhash0 1.4.3-3build1 | — | подтягиваются apt сами |

Также по ходу работы был поставлен `shellcheck` 0.9.0-1 — это только для проверки самих `.sh`-скриптов этой задачи,
рантайму сервера он не нужен.

## Runtime-раскладка и секреты

```bash
tools/ddnet-server/setup-runtime.sh
```

Создаёт `~/aiddnet/data/ddnet-server/storage.cfg` с двумя путями поиска (в порядке приоритета — первый и есть
"save path", куда сервер сам создаёт `teehistorian/`, `demos/` и т.д.):

1. `~/aiddnet/data/ddnet-server` — наш save path (карты-копии, `teehistorian/`).
2. `~/aiddnet/build/ddnet-20.1/src/data` — штатный data-каталог DDNet 20.1 (встроенные карты, скины, ...), только
   для чтения.

Каталог секретов **не** входит в storage-пути и не `exec`'ится из `local.cfg`. Секреты грузятся отдельным,
абсолютным `-f`-аргументом на командной строке юнита (`install-service.sh`):

```
DDNet-Server -f .../local.cfg -f ~/aiddnet/data/secrets/ddnet-server-secrets.cfg ...
```

**Почему так, а не через storage-path** (как было в первой версии этого файла): команда `exec` в консоли DDNet
сознательно отказывается открывать файл по абсолютному пути («don't escape base directory»,
`engine/shared/storage.cpp`), поэтому напрямую `exec /home/.../ddnet-server-secrets.cfg` из `local.cfg` не работает.
Первое решение — добавить `~/aiddnet/data/secrets` как storage-path с низким приоритетом и делать
`exec ddnet-server-secrets.cfg` относительным именем — работало, но при этом **весь** каталог секретов оказывался на
общем пути поиска сервера, то есть любой файл, который туда попадёт в будущем, стал бы `exec`-абельным через
rcon/econ. `-f` (в отличие от `exec`) поддерживает абсолютные пути официально (`IStorage::TYPE_ABSOLUTE`,
`console.cpp` `ParseArguments`), поэтому вместо обхода через storage-path секреты просто передаются вторым `-f` —
никакого специального пути поиска для каталога секретов не требуется вовсе.

Для ручного запуска без systemd (то же самое делает юнит) нужны оба `-f`:

```bash
DDNet-Server -f tools/ddnet-server/local.cfg -f ~/aiddnet/data/secrets/ddnet-server-secrets.cfg
```

Копируемые карты (в `~/aiddnet/data/ddnet-server/maps/`):

- `Copy Love Box.map` — копия `~/aiddnet/data/maps/copy-love-box/Copy Love Box_6e79ef43....map` (основной файл,
  sha256 сверяется скриптом), используется как **дефолтная блок-карта** (`sv_map` в `local.cfg`).
- Все 6 карт из `~/aiddnet/data/research/physics-scratch/maps/` (`BlmapChill`, `blmapV3multistarbox`,
  `blmapV5_ddpp`, `Blockdale`, `BlockField`, `ChillBlock5`) — **все шесть** грузятся в vanilla DDNet-Server 20.1 без
  ошибок (проверено `check-maps.sh`, ни одна не провалилась). Список карт-исключений тут пуст, но
  `LOADABLE_PHYSICS_SCRATCH_MAPS` в `setup-runtime.sh` — единственное место, которое трогать, если это изменится.

Плюс встроенные карты DDNet 20.1 (доступны через storage-path №2, доступны в votes/maplist, не «блок»-карты —
ctf/dm/арена): `coverage`, `ctf1`–`ctf7`, `dm1`, `dm2`, `dm6`–`dm9`, `Gold Mine`, `LearnToPlay`, `Sunny Side Up`
(дефолт ванильного DDNet), `Tsunami`, `Tutorial`.

```bash
tools/ddnet-server/gen-secrets.sh            # только если файла ещё нет
tools/ddnet-server/gen-secrets.sh --force    # пересоздать пароли (старые сессии/копии перестанут работать)
```

Пароли — hex(24 байт) через `openssl rand -hex 24`, без спецсимволов (не нужно эскейпить в `.cfg`). Файл никогда не
печатается скриптами на stdout/stderr; `econ.py` читает его сам.

## Установка и запуск (systemd)

```bash
tools/ddnet-server/install-service.sh              # ставит юнит, enable, (re)start
tools/ddnet-server/install-service.sh --no-start   # только установить/enable
```

Юнит **системный** (`/etc/systemd/system/ddnet-local.service`), запускается от текущего пользователя (`ubuntu`) —
не user-юнит с `loginctl enable-linger`. Почему:

- Не нужен linger и активная сессия — юнит стартует при загрузке системы независимо от того, залогинен ли кто-то.
- На этой машине уже есть passwordless sudo и другие системные unit'ы — дополнительная системная служба не меняет
  модель угроз, а user-юнит с linger добавил бы отдельный механизм (linger), который тут не нужен ничем другим.
- Проще проверять (`systemctl status` / `journalctl -u`, без `--user` и без входа в сессию пользователя).

```bash
sudo systemctl status ddnet-local.service
sudo systemctl restart ddnet-local.service
sudo systemctl stop ddnet-local.service
journalctl -u ddnet-local.service -f
```

`Restart=on-failure`, `StandardOutput`/`StandardError` — `append:~/aiddnet/data/logs/ddnet-server/{stdout,stderr}.log`.

**Юнит не зависит от этого чекаута.** `install-service.sh` копирует `tools/ddnet-server/local.cfg` в
`~/aiddnet/data/ddnet-server/local.cfg` (стабильный путь) и указывает `-f` именно туда, а не в репозиторий — так
служба продолжает работать, даже если этот worktree/чекаут потом удалят. Перезапустите `install-service.sh` после
правок `local.cfg`, чтобы обновить копию.

**Fail closed, а не тихий откат к небезопасным дефолтам.** Если стабильная копия конфига или файл секретов
отсутствуют/нечитаемы, `ExecStartPre=/usr/bin/test -r ...` для каждого из них проваливается — юнит вообще не
запускает `DDNet-Server` (проверено: `mv` конфига в сторону → `systemctl restart` → юнит уходит в
`activating (auto-restart)`, ничего не слушает; `mv` обратно → сервис поднимается штатно). Без этой проверки
отсутствующий `-f`-файл не считается фатальной ошибкой для самого DDNet: он просто логирует
`console: failed to open '...'` и продолжает работу на дефолтах штатного `data/autoexec_server.cfg` — а туда
`bindaddr` не входит вовсе, то есть без явного `-f` сервер слушал бы `0.0.0.0`/`::`. Поэтому же ExecStart повторяет
`bindaddr 127.0.0.1`, `sv_register 0` и `sv_ipv4only 1` прямо в командной строке, отдельными кавычными аргументами
(каждый разбирается `ParseArguments`/`ExecuteLine` как обычная строка конфига) — второй, независимый слой защиты
сверху `local.cfg`.

**Ядерный бэкстоп: `IPAddressAllow=127.0.0.0/8 ::1` + `IPAddressDeny=any`.** Это cgroup-eBPF-фильтр systemd, никак не
связанный с конфигом DDNet: пакет от/к любому адресу, кроме loopback, для процессов этого юнита отбрасывается ядром,
даже если приложение по какой-то причине всё-таки слушало бы шире. Проверено: `systemctl show ... -p IPAddressAllow
-p IPAddressDeny` показывает применённые значения, `bpftool cgroup show` — реально навешанные `sd_fw_ingress`/
`sd_fw_egress` фильтры на cgroup юнита; loopback-трафик (econ, игра бота) при этом работает как обычно.

Проверено вживую: `ss -ulnp` показывает **только** `127.0.0.1:8303` (UDP), `ss -tlnp` — **только** `127.0.0.1:8304`
(TCP, econ); ничего на `0.0.0.0`/`::`. `sudo ufw status` — не изменился, только `22/tcp`.

### Локальные лимиты подключений

Штатный `data/autoexec_server.cfg` (грузится автоматически до `local.cfg`) рассчитан на публичный сервер:
`sv_max_clients_per_ip 4`, `sv_connlimit`/`sv_connlimit_time` (5 подключений/20 с с одного IP), `sv_test_cmds 1`
(rcon-читы, тип сервера "TestDDraceNetwo") и несколько votes, включая "Option: Shutdown server" (доступен любому
подключившемуся). `local.cfg` переопределяет это для локального тестирования:

- `sv_max_clients_per_ip 12` (= `sv_max_clients`) — иначе больше 4 ботов с 127.0.0.1 одновременно не пускает
  (`Only N players with the same IP are allowed`). Проверено: 8 одновременных TS-ботов с разными именами подключились
  все сразу (`status` показал id=0..7).
- `sv_connlimit_time 0` — иначе быстрые повторные подключения с одного IP (наши же скриптовые реконнекты/тесты)
  ловят `Too many connections in a short time`.
- `sv_test_cmds 0` — оставляем вэнильное (не-читерское) поведение для физической достоверности.
- `clear_votes` — на приватном бот-only сервере голосования не нужны, включая опасное «Shutdown server».

## Смена карт и админ-команды (econ)

Econ слушает `127.0.0.1:8304`. `econ.py` — одна команда за один запуск (подключился → авторизовался → выполнил →
вывел ответ → отключился):

```bash
python3 tools/ddnet-server/econ.py status
python3 tools/ddnet-server/econ.py change_map "BlmapChill"
python3 tools/ddnet-server/econ.py broadcast "map changing soon"
python3 tools/ddnet-server/econ.py shutdown
```

Пароль по умолчанию читается из `~/aiddnet/data/secrets/ddnet-server-secrets.cfg` (флаг `--password-file`, чтобы
указать другой; `--password`, чтобы передать явно). `status` при отсутствии игроков печатает пусто (это поведение
самого DDNet — `ConStatus` просто ничего не выводит для пустых слотов, не баг клиента).

**Почему econ.py ждёт до ~1,5 с на каждый обмен, а не отвечает мгновенно.** Когда на сервере нет игроков,
`CServer::Run()` (`server.cpp`) ставит весь тик-луп спать в `net_socket_read_wait(GameSocket, 1s)` и перепроверяет
econ (отдельный TCP-сокет, к этому вызову не относящийся) только когда этот сон закончится — то есть даже самое
первое приветствие ("Enter password:") на пустом сервере может прийти почти через секунду тишины. `econ.py` ждёт
"первый байт" до `--first-byte-timeout` (по умолчанию 1.5 с, сознательно выше секунды простоя DDNet), и только ПОСЛЕ
того как ответ начал приходить — использует короткое окно `--quiet-time` (0.6 с), чтобы понять, что многострочный
ответ закончился. Раньше (без разделения этих двух таймаутов) короткий общий тайм-аут иногда обрывал ожидание
приветствия до того, как сервер успевал его отправить, и `econ.py` ошибочно репортовал это как "неверный пароль".

**Одно подключение на IP.** `CNetConsole::AcceptClient` (`network_console.cpp`) пускает только одно TCP-подключение
econ с одного IP одновременно; второе подряд получает `only one client per IP allowed` и закрывается — а закрытие
*предыдущего* соединения сервер тоже замечает только на следующей прокрутке того же тик-лупа (тот же простой на
пустом сервере). Поэтому `econ.py`: (1) после команды сам шлёт `logout` (обрабатывается синхронно, пока сервер уже
не спит, а разбирает нашу строку — освобождает слот сразу, а не при следующем пробуждении) и только потом закрывает
соединение; (2) если следующий запуск всё же встретил `only one client per IP allowed`/`no free slot available` (или
вообще пустой ответ на баннер), повторяет попытку до `--retries` раз (по умолчанию 4) с паузой `--retry-delay`
(0.3 с). Проверено: **20 вызовов `econ.py status` подряд** (без пауз, на полностью пустом сервере — то есть в самом
медленном режиме) — все 20 успешны.

## Логи и teehistorian

- Лог сервиса (stdout/stderr systemd) и внутренний `logfile` DDNet — оба в `~/aiddnet/data/logs/ddnet-server/`.
  `logfile` не прописан в `local.cfg` (это закоммиченный файл — не должен зашивать один конкретный абсолютный путь):
  его передаёт `install-service.sh` тем же механизмом, что и `bindaddr`/`sv_register` (отдельным аргументом
  ExecStart), используя свой `$LOG_DIR` (учитывает `DDNET_DATA_ROOT`).
- Teehistorian (`sv_tee_historian 1` — **важно**: спека и `docs/research/ddnet-physics.md` называют это
  "`sv_teehistorian`", но в исходниках DDNet 20.1 (`engine/shared/config_variables.h`) переменная называется
  `sv_tee_historian`, с подчёркиванием; в `local.cfg` используется настоящее имя) пишет по файлу на игровую сессию
  (join первого игрока/смена карты/старт сервера) в `~/aiddnet/data/ddnet-server/teehistorian/<game-uuid>.teehistorian`.

## End-to-end тест с ботом

Исторически (фаза 2.1) сюда подключали старый TS-бот; он удалён задачей 5.4 (последний коммит с ним — `0311695`:
`git show 0311695:src/bot/main.ts`). Теперь клиент — Rust:

```bash
cd ~/aiddnet/DDNet-AI
target/release/ddnet-ai play --server 127.0.0.1:8303 --brain scripted --duration 30
# исходная команда фазы 2.1: node src/bot/main.ts --server 127.0.0.1:8303 --scripted --duration 30 --no-console --verbose
```

Наблюдения ниже — фазы 2.1 и относятся к старому TS-боту (его таймаут молчания ~15 с); у Rust-клиента таймаут по умолчанию
100 с (как у клиента DDNet), для проверки реконнекта задаётся `--timeout-secs`.

Посередине — смена карты через econ (`change_map`), бот следует за ней автоматически (видно `map change: ...` в
логе бота). Реконнект после падения сервера: `sudo systemctl restart ddnet-local.service` во время `--duration 60`
прогона — бот детектит таймаут (~15 с без пакетов), сам переподключается через ~1 с и продолжает играть. Логи обоих
прогонов и релевантные строки серверного лога сохранены в `~/aiddnet/data/logs/phase2.1-e2e/` (см. BUILD REPORT для
цитат).

## Безопасность

- `bindaddr 127.0.0.1`, `sv_port 8303` (UDP), econ `ec_bindaddr 127.0.0.1`, `ec_port 8304` (TCP) — оба **только**
  loopback, проверено `ss`.
- `sv_register 0` — сервер никогда не регистрируется на мастер-серверах DDNet и не появляется в браузере серверов.
- `sudo ufw status` не менялся этой задачей — открыт только `22/tcp` (SSH), 8303/8304 наружу не пробрасывались и не
  должны.
- `sv_rcon_password`/`ec_password` — только в `~/aiddnet/data/secrets/ddnet-server-secrets.cfg` (режим 0600,
  генерируется `gen-secrets.sh`), никогда в `local.cfg`, никогда в командной строке юнита как обычный аргумент и
  никогда не коммитятся; юнит грузит их отдельным `-f`, не через storage-путь (см. "Runtime-раскладка").
- `sv_ipv4only 1` — явно IPv4-only: без него DDNet при `bindaddr 127.0.0.1` всё равно пытается создать ещё и IPv6-
  сокет (`server.cpp`: `BindAddr.type = sv_ipv4only ? NETTYPE_IPV4 : NETTYPE_ALL`, независимо от того, что
  `BindAddr` уже содержит чисто IPv4-адрес), эта попытка на этой машине проваливается с "Cannot assign requested
  address" — безобидная, но лишняя строка в логе. `sv_ipv4only 1` убирает саму попытку. **Это никак не меняет** факт
  того, что сервер слушает только loopback — `bindaddr 127.0.0.1` гарантирует это независимо; `sv_ipv4only` просто
  убирает не относящуюся к делу ошибку из лога.
- Fail-closed при отсутствии конфига/секретов и `IPAddressAllow`/`IPAddressDeny` на уровне ядра — см.
  "Установка и запуск".

## Известные безобидные строки в логе

- `E console: failed to open 'myServerconfig.cfg'` — сам штатный `data/autoexec_server.cfg` DDNet пытается
  подключить пользовательский файл, которого нет; это ожидаемо и ни на что не влияет.
- `E sixup: couldn't load map maps7/...` — попытка подгрузить 0.7-версию карты для legacy-клиентов; у наших карт её
  нет, sixup-совместимость просто отключается, 0.6-клиенты (бот, реальные тизы) не затронуты.
