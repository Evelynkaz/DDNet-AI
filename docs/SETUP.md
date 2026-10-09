# SETUP: как поднять сервер заново

Всё, что ставится на VPS, записывается сюда (что, зачем, какой командой, какая версия). Порядок разделов = порядок
установки. Проверено на Ubuntu 24.04.5 LTS (kernel 6.8.0-142), x86_64.

## 0. Железо и ОС (факты, 2026-09-27)

- netcup VPS 2000 G12.5, KVM. `nproc` = 8; `lscpu`: **AMD EPYC-Rome Processor** (так гостю отдаёт гипервизор; флаги до
  AVX2/FMA/BMI2/SHA, **без AVX-512**); L3 128 МБ (8 × 16 МБ). 15 ГиБ ОЗУ + 8 ГиБ swap. Диск 251 ГБ (занято 12 ГБ).
- Сеть: скачивание ~2,0–2,2 Гбит/с (Hetzner nbg1 100 МБ за 0,42 с, fsn1 1 ГБ за 3,8 с), RTT до 1.1.1.1 ≈ 3,6 мс.
- Хост ограничивает непрерывный счёт: паузы ~10 мс примерно 10 раз в секунду на vCPU (в steal не видны) —
  `research/rust-stack.md` §4. Поток инференса держать на отдельном ядре.
- Пользователь `ubuntu`, sudo без пароля. ufw: открыт только 22 (80/443 откроются в фазе 5).

## 1. Папки

```bash
mkdir -p ~/aiddnet/{ref,data}
mkdir -p ~/aiddnet/data/{demos,clips,logs,connectome,checkpoints,secrets,traces,maps}
chmod 700 ~/aiddnet/data/secrets
```

## 2. Репозиторий и справочные клоны

```bash
cd ~/aiddnet && git clone https://github.com/Evelynkaz/DDNet-AI.git
cd ~/aiddnet/ref
git clone --depth 1 https://github.com/ddnet/ddnet.git            # фаза 0: 9576fd6 (master, 20.2-dev, 2026-09-27)
git clone https://github.com/heinrich5991/libtw2.git                # 060e4b6 (2026-09-02), MIT/Apache
git clone https://gitlab.com/Patiga/twmap.git                       # 7e5e620 (2026-07-07), AGPL — только читать
git clone --depth 1 https://github.com/Wranked1/DDNet-AI.git DDNet-AI-upstream   # c3c619d
```

Эталон физики — тег **20.1** (см. DECISIONS D-005); оракулы в фазе 1 берут именно его.

## 3. Rust

```bash
curl -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh
sh /tmp/rustup-init.sh -y --profile default --default-toolchain stable
source ~/.cargo/env
rustc --version   # rustc 1.98.1 (48a229cea 2026-09-01); компоненты: cargo, clippy, rustfmt, rust-docs
```

Версия закрепляется в `rust-toolchain.toml` репозитория (фаза 1).

## 4. Node 24 (только для генераторов TS-паритета: `tools/ts-trace`, `tools/ts-reference`)

Официальный архив с проверкой sha256, в `/usr/local`:

```bash
cd /tmp
curl -sL https://nodejs.org/dist/latest-v24.x/SHASUMS256.txt -o SHASUMS256.txt
F=$(grep -o 'node-v24[^ ]*-linux-x64.tar.xz' SHASUMS256.txt | head -1)   # node-v24.21.0-linux-x64.tar.xz
curl -sLO https://nodejs.org/dist/latest-v24.x/$F
grep " $F\$" SHASUMS256.txt | sha256sum -c -
sudo tar -xJf $F -C /usr/local --strip-components=1 --exclude CHANGELOG.md --exclude LICENSE --exclude README.md
node --version    # v24.21.0 ; npm 11.19.0
cd ~/aiddnet/DDNet-AI/tools/ts-reference && npm ci --ignore-scripts --no-audit --no-fund   # node_modules в .gitignore; нужен только gen-nav-dump.mjs (пакет teeworlds)
```

## 5. Исследовательские инструменты фазы 0 (не нужны для работы проекта)

- Python venv с `pyarrow` 25.0.1 и `pandas` 3.0.6 для разведки Feather-файлов: `~/aiddnet/data/research/venv`
  (создан `python3 -m venv` + `pip install pyarrow pandas`). Проект читает Feather на Rust (D-012).
- Стенд старой TS-версии: `~/aiddnet/data/research/harness/` (`run.mjs`, `trace.mjs`, `determinism.mjs`, …).
- Прототипы физики/протокола/бенчмарка: `~/aiddnet/data/research/{physics-scratch,proto-scratch,bench-sparse}/`.
- Блок-карты для разведки: `~/aiddnet/data/maps/copy-love-box/` (источник и sha256 — `SOURCES.txt`),
  `~/aiddnet/data/research/physics-scratch/maps/` (6 карт из github.com/DDNetPP/maps).

## 5a. Инструменты проверки (фаза 1, задача 1.1)

```bash
cargo install --locked cargo-deny            # 0.20.2 → ~/.cargo/bin/cargo-deny
# gitleaks 8.30.1 (MIT) → ~/.local/bin/gitleaks, архив проверен по sha256
# 551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb и по checksums-файлу релиза
# actionlint 1.7.12 → ~/.local/bin/actionlint, архив проверен по sha256 из релиза
```

Локальный полный прогон как в CI:

```bash
cargo fmt --all --check && cargo clippy --workspace --all-targets --locked -- -D warnings \
  && cargo test --workspace --locked && cargo deny check \
  && gitleaks git --redact --exit-code 1 --config .gitleaks.toml . && actionlint .github/workflows/ci.yml
```

## 5a'. Прочие утилиты

- `tcpdump` (apt) — захват трафика локального сервера для тестов сети (задача 2.2a):
  `sudo tcpdump -i lo -w <file>.pcap udp port 8303`.

## 5a''. Playwright (задача 5.1, e2e-тесты веба)

`tools/e2e/` — Node-проект (Playwright 1.63, lock-файл с хэшами): `cd tools/e2e && npm ci && npx playwright install
chromium-headless-shell` (кэш `~/.cache/ms-playwright/`, Chrome Headless Shell 153.0.8010.12); системные зависимости
Chromium ставились `sudo npx playwright install-deps chromium` (≈ 67 apt-пакетов: библиотеки X11/GTK/NSS/шрифты) и
`ffmpeg`. Запуск: `npx playwright test` (см. `tools/e2e/README.md`).

## 5b. Параллельные задачи: git worktree

Для одновременных задач /duo создаются отдельные рабочие копии без веток:
`git worktree add --detach ~/aiddnet/wt/task-<N> HEAD`. После одобрения изменения переносятся патчем в основной
`main` (`git -C <wt> add -A && git -C <wt> diff --cached --binary > p.patch; git apply --index p.patch`), рабочая
копия удаляется `git worktree remove`.

## 5c. Локальный DDNet-сервер 20.1 (задача 2.1)

apt-пакеты (Ubuntu 24.04.5): `cmake` 3.28.3-1build7, `ninja-build` 1.11.1-2, `pkg-config` 1.8.1-2build1,
`libcurl4-openssl-dev` 8.5.0-2ubuntu10.15, `libsqlite3-dev` 3.45.1-1ubuntu2.8, `libssl-dev` 3.0.13-0ubuntu3.15
(опционален), `zlib1g-dev` 1:1.3.dfsg-3.1ubuntu2.2 (возможно опционален), `shellcheck` 0.9.0-1 (линтер скриптов);
автоматически: cmake-data, pkgconf, pkgconf-bin, libpkgconf3, libjsoncpp25, librhash0. Уже были: build-essential, git,
python3. Rust для частей DDNet — существующий rustup 1.98.1.

```bash
tools/ddnet-server/build.sh            # исходники тега 20.1 (c9d20813…) + сборка → ~/aiddnet/build/ddnet-20.1/ (~45 с, пик ~1,8 ГБ)
tools/ddnet-server/setup-runtime.sh    # ~/aiddnet/data/ddnet-server/, карты, секреты (0600)
tools/ddnet-server/install-service.sh  # systemd ddnet-local.service; запускать от ubuntu, НЕ через sudo
python3 tools/ddnet-server/econ.py status        # админ-консоль (127.0.0.1:8304)
```

Сервер слушает только 127.0.0.1:8303 (UDP), econ — 127.0.0.1:8304 (TCP), `sv_register 0`; юнит не стартует без
конфига/секретов, systemd ограничивает трафик loopback'ом. После правки `tools/ddnet-server/local.cfg` — заново
`install-service.sh` (он копирует конфиг в `~/aiddnet/data/ddnet-server/`). Подробно — `tools/ddnet-server/README.md`.

## 5d. Локальный DDNet-сервер 18.5 (задача 2.3b, разовое расследование)

Разовый (не systemd-юнит) сервер DDNet **18.5** для локального воспроизведения инцидента Swarfey —
`tools/ddnet-server/build.sh` уже параметризован под тег/коммит/путь, отдельная сборка не понадобилась:

```bash
DDNET_BUILD_ROOT=~/aiddnet/data/ddnet-18.5 DDNET_TAG=18.5 \
  DDNET_COMMIT=bae736293a2bd1921dfc121e9c730c3d8e15bc70 DDNET_BUILD_JOBS=4 \
  tools/ddnet-server/build.sh
```

**Новый пакет по сравнению с 20.1**: `libpng-dev` 1.6.43-5ubuntu0.6. У тега 18.5 `CMakeLists.txt` требует
`PNG_FOUND` **безусловно** (`if(NOT(PNG_FOUND)) message(SEND_ERROR ...)`, без `CLIENT`-гейта — в отличие от
Freetype/Ogg/Opus/Opusfile/SDL2, все — `if(CLIENT AND NOT(...))`), т.е. даже server-only (`-DCLIENT=OFF
-DTOOLS=OFF`) сборка 18.5 не конфигурируется без libpng; в 20.1 (текущий пиннутый коммит) это ограничение уже
снято. Собрано без `DDNET_SKIP_APT` до этой находки (упало на `cmake configure`), затем `sudo apt-get install -y
libpng-dev` и повтор с `DDNET_SKIP_APT=1`. Бинарник: `~/aiddnet/data/ddnet-18.5/build/DDNet-Server` (7,1 МБ),
сборка ~46 с, пик RSS сборки ≈ 1,47 ГБ.

Рантайм — вручную, не через `setup-runtime.sh`/`install-service.sh` (те жёстко привязаны к путям 20.1 и к
systemd-юниту `ddnet-local.service`, который принадлежит другой задаче): `~/aiddnet/data/ddnet-18.5-runtime/`
(`storage.cfg`, `local.cfg`, `maps/Copy Love Box.map` — та же карта, что у 20.1, скопирована туда же,
`teehistorian/`). Порт игры **8306**, econ **8307** (пароли — временные, только для этого расследования, не
секреты продукта). Запуск вручную в фоне (`nohup ... &`), лог — `~/aiddnet/data/logs/ddnet-18.5-runtime.log`;
**не** systemd-юнит и **не** предназначен жить долго — техлид может удалить `~/aiddnet/data/ddnet-18.5-runtime/` и
`~/aiddnet/data/ddnet-18.5/` после того, как забирёт результаты задачи 2.3b, либо оставить для последующих задач
(проверено, что и `ddnet-ai record`, и `ddnet-ai play --brain idle` заходят на него без единой правки клиента).
`bindaddr 127.0.0.1`, `sv_register 0` — тот же loopback-only принцип, что и у 20.1.

**Второй экземпляр 18.5 с эмуляцией «reconnect-петли»** (порт **8309**, econ 8310, рантайм
`~/aiddnet/data/ddnet-18.5-runtime-flood/`, лог `~/aiddnet/data/logs/ddnet-18.5-flood-runtime.log`, `debug 1`).
Патч `~/aiddnet/data/ddnet-18.5/reconnect-flood-emulation.patch` (в репозиторий не входит; 18.5 не знает
`reconnect@ddnet.org` — патч добавляет `UUID(NETMSG_RECONNECT, ...)` в `protocol_ex_msgs.h` и в
`CServer::ProcessClientPacket` ветку `NETMSG_INFO`: при `DDAI_TEST_RECONNECT_FLOOD=1` отвечать этим сообщением вместо
карты). Один и тот же бинарник, без переменной ведёт себя как ванильный 18.5:

```bash
cd ~/aiddnet/data/ddnet-18.5/src && git apply ../reconnect-flood-emulation.patch   # один раз
cd ../build && ninja DDNet-Server
cd ~/aiddnet/data/ddnet-18.5-runtime-flood && DDAI_TEST_RECONNECT_FLOOD=1 nohup ../ddnet-18.5/build/DDNet-Server -f local.cfg &
```

Третий экземпляр — «полный сервер» (порт **8311**, econ 8312, рантайм `~/aiddnet/data/ddnet-18.5-runtime-full/`, тот же
`local.cfg`, но `sv_max_clients 1`; слот занимает сам тест).

E2E против них (`#[ignore]` + env-гейт, обычный `cargo test` их не запускает):

```bash
DDAI_E2E_185=1 DDAI_E2E_185_FLOOD_ADDR=127.0.0.1:8309 DDAI_E2E_185_FULL_ADDR=127.0.0.1:8311 \
  cargo test -p ddai-client --test e2e_ddnet185 -- --ignored --test-threads=1     # 8306 по умолчанию, DDAI_E2E_185_ADDR
```

Оба сервера слушают только 127.0.0.1; после задачи 2.3b их можно остановить (`kill` по pid из `pgrep -x DDNet-Server`,
осторожно: `ddnet-local.service` 20.1 — тоже `DDNet-Server`).

**Зависший sccache.** Во время задачи 2.3b общий `sccache`-сервер (`~/.cargo/bin/sccache`, `SCCACHE_DIR=
~/aiddnet/data/cache/sccache`) завис: сборки стояли на «Compiling libc», нагрузка ≈ 0. Лечение — убить процесс
сервера (`kill -9 <pid sccache без аргументов>`) и его дочерние `sccache /home/.../rustc`; следующий `cargo` поднимет
новый.

**Мод «DDFightNet fng» (сервер Swarfey) — исходники не найдены** (повторный поиск 2026-09-29). Что нашлось:
- `ddfight.net` — «DDFightNetwork», неофициальный хаб блок-серверов; ссылок на код, лицензии и версию DDNet нет.
- GitHub `swarfeya` (владелец сети): `ddnet-maps` (карты, без лицензии), `teeworlds-library-ts` (клиентская
  библиотека для ботов), форки `ddnet` (0 коммитов сверх апстрима, ветки — PR-ветки), `F-DDrace`, `libtw2`; кода мода нет.
  `gh search` по «ddfight» — только README `TaterClient/ddnet-custom-communities` и `Predatorr0/TClient-AI-Bot`.
- Строка «DDFightNet fng» есть в списке цветов game type (`ddnet/ddnet#12760`, `#12188`) — и всё.
- Публичные, но **не тождественные** «fng» на DDNet: `Inateblig/dfng`, `ddnet-insta/ddnet-insta`,
  `Jupeyy/teeworlds-fng2-mod` (не DDNet), `DDNetPP/DDNetPP`. Ни один не заявлен как источник DDFightNet.

Вывод: мод закрытый, собрать его нельзя, лицензия неизвестна. Поведение Swarfey воспроизводится только эмуляцией
на ванильном 18.5 (см. `docs/formats.md` §14.12).

## 6. Caddy + HTTPS + ufw (задача 5.3)

Бот (`ddnet-ai web`, задача 5.1) слушает только `127.0.0.1:7788`; наружу его публикует **Caddy**,
терминирующий HTTPS на `https://89-58-7-133.sslip.io` (публичный IP `89.58.7.133`). Всё
воспроизводится одним `deploy/install.sh` (идемпотентен) — детали и операции см.
`deploy/README.md`. Здесь — что именно было установлено.

### Установка Caddyfile: реальная валидация, не только синтаксис (находка ревью F11)

`deploy/install.sh` перед установкой `Caddyfile` копирует кандидата во временный файл прямо под
`/etc/caddy/` и гоняет `sudo -u caddy caddy validate` (не `caddy adapt`) — `validate` реально
прогружает конфиг (открывает писатель лога и т.п.), а не только парсит синтаксис, так что ловит
конфиг, который парсится, но не грузится (права на путь лога, ошибка матчера, TLS). Запускается
именно от пользователя `caddy` (не root/sudo как в первой версии этого скрипта — тогда файл лога
оставался root-owned и недоступен настоящему сервису) — `/var/log/caddy` создаётся раньше этого
шага специально для этого. Перед `(re)start`/`reload` скрипт сравнивает admin-адрес СТАРОГО
установленного `Caddyfile` с НОВЫМ (`caddy adapt | python3 -c "..."`): если адрес меняется (как в
находке F2 при переходе TCP→unix-сокет) — идёт прямой `restart` (reload физически не достучится до
старого адреса); если не меняется — только `reload`, и его неудача теперь **фатальна** (`die`,
старый конфиг продолжает отвечать), а не тихий фоллбэк на `restart` при ЛЮБОЙ причине отказа.

### Caddy: пакет, версия, репозиторий, ключ

Официальный (не Ubuntu universe — `2.6.2-6ubuntu0.24.04.3`/`2.6.2-6`, вышел в 2022, без ~2 лет
патчей безопасности) apt-репозиторий Cloudsmith:

```bash
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https
curl -fsSL 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' -o /tmp/caddy-gpg.key
sudo gpg --dearmor -o /tmp/caddy-stable-archive-keyring.gpg /tmp/caddy-gpg.key
sudo install -o root -g root -m 0644 /tmp/caddy-stable-archive-keyring.gpg \
  /etc/apt/keyrings/caddy-stable-archive-keyring.gpg
# /etc/apt/sources.list.d/caddy-stable.list:
#   deb [signed-by=/etc/apt/keyrings/caddy-stable-archive-keyring.gpg] \
#     https://dl.cloudsmith.io/public/caddy/stable/deb/debian any-version main
sudo apt-get update && sudo apt-get install -y caddy
```

- **Пакет**: `caddy` **2.11.4** (2026-06-03), из `https://dl.cloudsmith.io/public/caddy/stable/deb/debian`.
- Предварительные apt-пакеты (Ubuntu universe/main, не Caddy-репозиторий): `debian-keyring`
  **2023.12.24**, `debian-archive-keyring` **2023.4ubuntu1**, `apt-transport-https` **2.8.3**
  (переходный пакет, HTTPS-транспорт уже встроен в современный apt).
- **Ключ**: `/etc/apt/keyrings/caddy-stable-archive-keyring.gpg` (не штатный `/usr/share/keyrings/`
  из инструкции caddyserver.com — по спеке задачи 5.3, тот же файл, тот же эффект). Отпечаток
  (`rsa4096/155B6D79CA56EA34`, "Caddy Web Server <contact@caddyserver.com>"), проверен вручную
  после `gpg --dearmor` и сверяется автоматически при каждом запуске `deploy/install.sh`:
  `6576 0C51 EDEA 2017 CEA2 CA15 155B 6D79 CA56 EA34`.
  **Находка ревью F1**: старая проверка сверяла отпечаток только первого ключа в файле, а apt
  доверяет ВСЕМ ключам файла-кольца — файл с настоящим ключом Caddy плюс второй, подложенный,
  раньше проходил проверку. Теперь `deploy/install.sh` требует ровно один `pub`-ключ в файле
  (`gpg --show-keys --with-colons | grep -c '^pub:'`) и только потом сверяет отпечаток; ключ
  скачивается и проверяется во временном `mktemp -d` и копируется в `/etc/apt/keyrings` только
  после успешной проверки (раньше файл сначала копировался, потом проверялся). Репро ревьюера
  (кольцо с настоящим ключом Caddy + самодельным "Evil"-ключом) подтверждён и исправлен: до фикса
  проверка проходила, после — падает с `expected exactly 1 public key ... found 2`.
- Пакет сам ставит `caddy.service` (пользователь `caddy`, `HOME=/var/lib/caddy`, `ProtectSystem=full`,
  `AmbientCapabilities=CAP_NET_ADMIN CAP_NET_BIND_SERVICE` — из штатного юнита пакета, не менялся).
  Автоматически `enable`+`start` при установке пакета.

### Caddyfile (`deploy/caddy/Caddyfile` → `/etc/caddy/Caddyfile`)

- Сайт `89-58-7-133.sslip.io`: `reverse_proxy 127.0.0.1:7788` (WebSocket — автоматически),
  `request_body { max_size 16KiB }` (тот же лимит 16 384 байт, что у бота — **`KiB`, не `KB`**:
  находка ревью F6, `16KB` у go-humanize это 16 000 десятичных байт, на 384 байта меньше нужного),
  `encode zstd gzip` только для `/`, `/app.css`, `/app.js` (не для `/api/*`/`/ws` — именованный
  REQUEST-матчер `@static`, а не собственный response-матчер `encode`'а по умолчанию, который
  иначе сжал бы и `application/json`), заголовки `Strict-Transport-Security: max-age=31536000` и
  `-Server` (остальные security-заголовки — от самого бота, проходят через проксю без изменений —
  проверено `curl -I`, см. BUILD REPORT).
- **Находка ревью F6: ответы, которые генерирует сам Caddy** (413 от `request_body`, и в общем
  случае любая ошибка, до которой `reverse_proxy` не доходит) **не проходят через блок `header`**
  — живой репро ревьюера подтверждён (413 без HSTS и с `server: Caddy`). Добавлен `handle_errors`
  с тем же `header{}` (HSTS + `-Server`) и `respond "{err.status_code} {err.status_text}"` вместо
  тела ошибки по умолчанию. Отдельно: запрос с правильным SNI (значит, TLS проходит с настоящим
  сертификатом), но с несовпадающим HTTP `Host` — раньше падал на пустой `200 OK` без единого
  заголовка; добавлен безымянный `:443 { abort }` в конце файла (без своего SNI/сертификата —
  просто закрывает соединение для всего, что не совпало ни с одним настоящим сайтом).
- `servers { protocols h1 h2 }` (глобальная опция) — только HTTP/1.1 и HTTP/2, HTTP/3 выключен:
  без этого Caddy рекламирует `Alt-Svc`/слушает QUIC на UDP/443, а ufw держит UDP/443 закрытым.
  Проверено: `ss -ulpn | grep caddy` — пусто (нет UDP-слушателя вообще).
- `trusted_proxies` не задан нигде (ни глобально, ни в `reverse_proxy`) — это то самое "не
  доверять клиентскому `X-Forwarded-For`": без него Caddy сам вычисляет реальный IP клиента и
  **заменяет** любой присланный клиентом `X-Forwarded-For` им (не добавляет в конец списка) —
  это и делает `ddnet-ai web --trust-proxy` (доверяет XFF только когда TCP-peer — сам loopback,
  т.е. Caddy) безопасным. Проверено живьём (BUILD REPORT): 6 неудачных попыток входа со
  спуфленным `X-Forwarded-For: 6.6.6.6` — в аудит-логе бота у всех настоящий IP, не `6.6.6.6`.
- **Admin API — находка ревью F2: заменён на unix-сокет.** Было `admin localhost:2019` (дефолт
  Caddy) — ревьюер живьём показал, что это доступно ЛЮБОМУ локальному процессу, включая
  гипотетически скомпрометированный `ddnet-ai-web`: его `IPAddressAllow=127.0.0.0/8` не исключает
  127.0.0.1:2019 (это про то, к каким *адресам* можно обращаться, не про то, какие локальные
  сервисы разрешены), а сам admin API без аутентификации и может целиком переписать конфиг Caddy
  (например, открыть econ 127.0.0.1:8304 наружу). Стало: `admin unix//var/lib/caddy/admin.sock|0600`
  — `/var/lib/caddy` уже существует (0750, `caddy:caddy`, HOME пакета), доступ к сокету только у
  пользователя `caddy` (и root); `ubuntu`, под которым работает `ddnet-ai-web`, не входит в группу
  `caddy` и не может даже войти в `/var/lib/caddy` (0750 — permission denied на сам каталог, до
  проверки прав самого файла сокета). Репро ревьюера повторено до/после:
  `systemd-run -p User=ubuntu -p IPAddressAllow="127.0.0.0/8 ::1" -p IPAddressDeny=any curl
  http://localhost:2019/config/` — до фикса 200, после фикса `Connection refused` (порт вообще не
  слушает; `ss -tulpn` больше не показывает `127.0.0.1:2019`). `systemctl reload caddy` продолжает
  работать: `caddy reload` без `--address` читает адрес admin API из ЦЕЛЕВОГО конфига, который
  грузит — см. `deploy/install.sh`, при первом переключении на сокет использован разовый ручной
  шаг с `--address localhost:2019` (старый адрес — единственный способ достучаться до ещё не
  переключившегося процесса), автообновления такого шага уже не требуют.
- Access-лог: JSON, `/var/log/caddy/access.log` (каталог создаётся `deploy/install.sh`, владелец
  `caddy:caddy`, права `0750`), ротация `roll_size 20MiB` / `roll_keep 10` / `roll_keep_for 720h`.
  `Cookie`/`Set-Cookie`/`Authorization`/`Proxy-Authorization` редактируются в `"REDACTED"` самим
  Caddy по умолчанию (опция `log_credentials`, которая это отключает, не задана) — проверено
  живым логом.
- ACME: HTTP-01, автоматически (Caddy default `acme_ca` = Let's Encrypt, с фоллбэком на ZeroSSL
  при неудаче — не понадобился). `email` не задан — не хотели класть личный e-mail владельца в
  файл, который коммитится в открытый репозиторий; Let's Encrypt выпускает и продлевает и без
  него. ACME-состояние — штатный каталог Caddy (`/var/lib/caddy/.local/share/caddy`), не менялся.
  **`sslip.io` не входит в Public Suffix List** (проверено вручную загрузкой
  `publicsuffix.org/list/public_suffix_list.dat` 2026-09-27 — нет строки `sslip.io`; см. также
  `docs/research/rust-stack.md` §5). **Исправление (ревью, находка F4): предыдущая версия этого
  абзаца было наоборот.** Раз `sslip.io` НЕ в PSL, Let's Encrypt считает "registered domain" сам
  `sslip.io` целиком (апекс), а не каждое поддомен-имя вида `89-58-7-133.sslip.io` отдельно —
  значит по умному (per-registered-domain) бюджету лимитов ВСЕ пользователи sslip.io в мире делят
  ОДИН общий бюджет, а не получают отдельный на каждое имя. Именно из-за этого общего эффекта
  sslip.io/nip.io договорились с Let's Encrypt об отдельном повышенном лимите для всего домена
  (250 000 сертификатов/неделю вместо обычных 50, по данным страницы nip.io) — это смягчение
  общего риска, а не его отсутствие. Продления (не первый выпуск) освобождены от лимита заказов
  через ARI, так что риск в основном разовый (первый выпуск на новое имя), не при продлении —
  наш случай именно такой. Получилось **за 1 попытку** (см. BUILD REPORT:
  `certificate obtained successfully` в журнале, issuer/expiry через `openssl`).

### Юнит бота (`deploy/systemd/ddnet-ai-web.service` → `/etc/systemd/system/`)

`ddnet-ai web --listen 127.0.0.1:7788 --data-dir ~/aiddnet/data --trust-proxy --cookie-secure`,
пользователь `ubuntu`. Жёсткая песочница (systemd hardening — полный список в самом юните):
`NoNewPrivileges`, `ProtectSystem=strict` + `ProtectHome=read-only` с исключениями только
`~/aiddnet/data/{secrets,logs}` (`ReadWritePaths`), `PrivateTmp`, `RestrictAddressFamilies=AF_INET
AF_INET6 AF_UNIX`, `MemoryMax=512M`, `Restart=on-failure`, плюс ядерный (cgroup eBPF) бэкстоп
`IPAddressAllow=127.0.0.0/8 ::1` + `IPAddressDeny=any` (тот же приём, что у `ddnet-local.service`,
раздел 5c) — на случай, если приложение всё же слушало бы шире, чем `--listen 127.0.0.1:7788`
(которое само по себе уже отказывается стартовать на не-loopback-адресе, задача 5.1 — проверено
живьём в BUILD REPORT: `--listen 0.0.0.0:...` → бинарник завершается с ошибкой, ничего не слушает).

### Доверенные устройства: `~/aiddnet/data/secrets/web-devices.toml` (находки ревью F8a, F9, F10)

Файл (`0600`, каталог `secrets/` — `0700`) хранит устройства, которым разрешён обход общего
(global) лимита входа — по одной строке: `<base64url sha256(device_id)> <expires_at_unix_s>
<last_seen_unix_s> <ключ-фингерпринт пароля>`. Не настоящий TOML (см. комментарий в самом файле);
управляется автоматически, руками не редактировать, в репозиторий не коммитить (`CLAUDE.md`).

- **F8a (сохранение между перезапусками).** Раньше жил только в памяти — перезапуск/redeploy
  сбрасывал единственный способ владельца обойти общий лимит именно в момент, когда флуд с многих
  адресов вероятнее всего и идёт. Теперь загружается при старте, пишется при каждом изменении.
  Проверено живьём реальным `systemctl restart ddnet-ai-web`: `bypassed_global_rate_limit=true` в
  аудит-логе до и после перезапуска для одного и того же реального cookie.
- **F9 (найдено ревью раунда 2): украденная копия старого device-cookie не должна снова получать
  обход после того, как владелец просто зайдёт с новым паролем.** Раньше `login.rs` переиспользовал
  presented-id даже если он не был доверен (например, после смены пароля) — "чтобы браузер не
  копил новые записи на каждый вход" — но именно это переиспользование заново подтверждало ID (и
  значит, HMAC-подпись cookie, которая от смены пароля не меняется) под новым паролем, что заодно
  реактивировало ЛЮБУЮ другую копию того же значения cookie, включая украденную. Исправлено в трёх
  местах:
  1. `login.rs`: presented-id переиспользуется ТОЛЬКО если он уже был доверен (`bypass_global ==
     true` на момент запроса); иначе — свежий случайный id. Никогда не "воскрешаем" недоверенный id.
  2. `secrets::generate_and_store_password` (`ddnet-ai web-passwd`) теперь дополнительно очищает
     `web-devices.toml` целиком при каждой смене пароля.
  3. Хранится не сырой argon2-хэш, а `HMAC-SHA256(session_key, hash_phc)` (`auth::device::
     password_fingerprint`) — утечка одного этого файла больше не выдаёт материал хэша пароля.
  Точечный 7-шаговый сценарий ревьюера воспроизведён как тест
  (`reviewer_scenario_a_saved_old_cookie_copy_never_regains_bypass` в `device.rs` +
  `stolen_device_cookie_copy_does_not_regain_bypass_after_a_password_change_and_relogin` —
  полный HTTP end-to-end в `login_and_sessions.rs`) — оба зелёные.
  **Документируется поведение (пункт (d) находки):** `web-passwd` — отдельный короткоживущий CLI-
  процесс; он чистит файл на диске, но НЕ трогает память уже запущенного `ddnet-ai-web` — та
  таблица независима и при следующем изменении (например, обычный вход другого браузера) снова
  запишет файл, включая свои старые записи. Это безопасно: у такой записи фингерпринт всегда под
  СТАРЫМ паролем, а `is_trusted` сравнивает с ТЕКУЩИМ — совпасть он уже не может никогда, так что
  само присутствие такой записи инертно, а не живой credential.
- **F10 (неограниченный рост).** Каждый вход без cookie устройства (новый браузер, curl,
  Playwright) добавлял запись на 90 дней без ограничения — на проде реально накопилось 6+ записей
  за время тестирования. Добавлен потолок `MAX_TRACKED_DEVICES = 32` (вытесняется наименее давно
  использованная запись), и запись на диск теперь строится под мьютексом (дешёвый снимок), а сам
  fsync выполняется уже ПОСЛЕ его освобождения — раньше конкурентный `is_trusted`/`confirm` от
  другого запроса ждал бы за тем же мьютексом на всё время fsync на одном и том же async-потоке.

- **Смена ключа сессий** (`web-session-key.toml`: удалить или сгенерировать заново) делает недоверенными все устройства сразу: их cookie подписаны этим ключом, и отпечаток пароля в файле — HMAC на нём же. Все сессии тоже завершаются. Так и задумано; пользоваться этим, если есть подозрение, что ключ утёк (ревью 5.3, F13).

### Автообновления Caddy (unattended-upgrades, находка ревью F5)

Ubuntu's own `50unattended-upgrades` только разрешает origin'ы `Ubuntu:*`/`UbuntuESM*:*` —
Cloudsmith-репозиторий Caddy туда не попадает, значит без явного разрешения интернет-смотрящий
Caddy никогда не получал бы автоматических патчей безопасности. `deploy/install.sh` ставит
`/etc/apt/apt.conf.d/51unattended-upgrades-caddy`:

```
Unattended-Upgrade::Origins-Pattern {
	"origin=cloudsmith/caddy/stable";
};
```

Строка `origin=...` сверена не угадыванием, а живым `apt-cache policy caddy`/файлом Release
репозитория (`Origin: cloudsmith/caddy/stable`, `unattended-upgrade` матчит `fnmatch` по полю
`origin.origin` — `usr/bin/unattended-upgrade:match_whitelist_string`). Списки `Origins-Pattern`
из разных файлов `apt.conf.d/*` складываются (не перезатирают друг друга), так что штатные
Ubuntu-origin'ы остаются как есть. Проверено живьём: `sudo unattended-upgrade --dry-run --debug
2>&1 | grep -i caddy` → `Allowed origins are: ..., origin=cloudsmith/caddy/stable` (строка
появилась после установки дроп-ина, отсутствовала до).

### Бинарник

`cargo build --release -p ddnet-ai` из этого чекаута → `~/aiddnet/bin/ddnet-ai` (стабильный путь,
не зависит от рабочей копии/ветки). Обновление — повторный явный запуск `deploy/install.sh`, без
автообновления.

### ufw

Было (до задачи 5.3): только `22/tcp` (v4+v6). Стало: `+ 80/tcp`, `+ 443/tcp` (оба — v4 и v6,
одно правило на порт покрывает оба семейства, IPv6 включён в `/etc/default/ufw`), `22/tcp` не
трогался. Полный `sudo ufw status verbose` до/после — в BUILD REPORT. UDP не открывался (HTTP/3
выключен, см. выше).

### Playwright HTTPS e2e (задача 5.3)

`tools/e2e/web-login-https.spec.ts` — тот же сценарий входа/статуса/выхода, что у 5.1, но через
настоящий `https://89-58-7-133.sslip.io` (см. `tools/e2e/README.md`); паролем из
`~/aiddnet/data/secrets/web-password.txt`. Пропускается без `DDAI_E2E_BASE_URL`, так что обычный
`npx playwright test` (без окружения) остаётся зелёным и гоняет только локальный 5.1-сценарий.

## 7. Будет добавлено по фазам

- Фаза 1: C++-оракулы собираются теми же пакетами, что и сервер (раздел 5c).
- Фаза 2: сетевой клиент (без новых системных пакетов).
- Фаза 6: данные коннектома по манифесту.

### sccache (общий кэш компиляции для worktree, 2026-09-28)

- Бинарник `sccache` v0.18.0 (musl, из релизов github.com/mozilla/sccache) установлен в `~/.cargo/bin/sccache`,
  sha256 архива `45f1447fbe231e3037bde351ef70677dd212216c8d62ae7ca409fecc4d6acc89` (проверен по файлу `.sha256` из
  релиза).
- **Глобально не включён**: смена `RUSTC_WRAPPER` меняет отпечатки cargo и заставила бы уже собранные worktree
  пересобраться. Включается на сборку: `RUSTC_WRAPPER=sccache SCCACHE_DIR=~/aiddnet/data/cache/sccache
  SCCACHE_CACHE_SIZE=20G cargo build`. Кэшируются в основном зависимости (для крейтов воркспейса в dev-профиле
  cargo использует инкрементальную сборку, а её sccache не кэширует).
- Статистика: `sccache --show-stats`, остановить сервер: `sccache --stop-server`.

## Windows: что поставлено для проверки сборки под Windows с Linux (задача 5.5a, 2026-10-09)

Через `apt` ничего не ставилось. В пользовательском каталоге и во временной папке агента:

- `rustup target add x86_64-pc-windows-gnu x86_64-pc-windows-msvc x86_64-unknown-linux-musl --toolchain 1.98.1` (только `rust-std`, без линкера):
  даёт `cargo check/clippy --target x86_64-pc-windows-gnu --workspace --all-targets` (проверка, что всё компилируется под Windows и нет предупреждений)
  и `x86_64-unknown-linux-musl` как «не glibc libm» для проверки, что тесты не зависят от битов glibc.
- Для крейтов с C-кодом (`aws-lc-sys`, `zstd-sys`) нужен кросс-компилятор C. Использован `zig cc` (Zig 0.13.0 с ziglang.org, распакован в
  временную папку, не установлен в систему) под именами `x86_64-w64-mingw32-gcc/-ar/-g++`: обёртки отбрасывают `--target=...`, который добавляет
  `cc-rs`, и зовут `zig cc -target x86_64-windows-gnu`. Для `aws-lc-sys` ещё `AWS_LC_SYS_NO_JITTER_ENTROPY=1` (в mingw-заголовках Zig нет `sched.h`).
  Это нужно только для проверки с Linux; настоящая сборка идёт на `windows-latest` (MSVC) в CI-задаче `windows`.
- Запустить Windows-бинарники здесь нельзя (нет Wine): запуск тестов под Windows происходит только в CI.

Сборка на самой Windows: `rustup` + Visual Studio Build Tools (MSVC) + Git; для `aws-lc-sys` либо NASM в `PATH`, либо `AWS_LC_SYS_PREBUILT_NASM=1`.
Подробности для пользователя — в README, раздел «Запуск на Windows».

## Пакеты, поставленные 2026-10-01

- `git-lfs` 3.4.1 (`sudo apt-get install -y git-lfs`). Нужен, чтобы скачивать LFS-архивы демок ChillerDragon по
  D-057 в `~/aiddnet/data/demos/chillerdragon/`.
