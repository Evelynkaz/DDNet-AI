# Развёртывание: Caddy + HTTPS + веб-бот (задача 5.3)

Публичный вход: `https://89-58-7-133.sslip.io`. Caddy терминирует HTTPS и проксирует на бота
(`ddnet-ai web`, задача 5.1), который слушает **только** `127.0.0.1:7788`. ufw открыт только на
`22/tcp` (SSH), `80/tcp` и `443/tcp`. Подробности того, что именно установлено (версии, ключи,
Caddyfile, юнит) — `docs/SETUP.md` §6.

## Установка / обновление

```bash
deploy/install.sh
```

Идемпотентен — безопасно гонять повторно (в т.ч. после `git pull`, чтобы обновить и Caddy-конфиг,
и юнит, и пересобранный бинарник бота). Что делает: ставит/обновляет пакет Caddy из официального
репозитория (с проверкой ключа — ровно один ключ в кольце, отпечаток совпадает), ставит дроп-ин
`unattended-upgrades` для автообновлений Caddy, открывает `80`/`443` в ufw (никогда не
трогает/не отключает ufw целиком), собирает `ddnet-ai` (`cargo build --release`) и кладёт в
`~/aiddnet/bin/ddnet-ai`, ставит `deploy/systemd/ddnet-ai-web.service` и `deploy/caddy/Caddyfile`,
(пере)запускает оба сервиса.

**Не делает** (осознанно): не генерирует и не меняет пароль — это отдельный, явный шаг (см. ниже),
и не печатает никаких секретов.

Полезные флаги:

```bash
deploy/install.sh --skip-build   # переустановить Caddy/юнит/Caddyfile без пересборки бинарника
deploy/install.sh --no-ufw       # не трогать ufw вообще (если он уже настроен как нужно)
```

### Первое развёртывание с нуля

После первого `deploy/install.sh` бот стартует, но без пароля (`web-auth.toml` ещё нет) —
скрипт сам это печатает как напоминание. Пароль генерируется отдельно (см. ниже) **один раз** —
дальше он переживает любой `deploy/install.sh`/`systemctl restart`.

## Управление ботом с сайта (задача 5.6, D-070): что меняется при развёртывании

Вкладка «Бот» (статус, команды, редактор друзей / войны / игнора) работает через три файла в `~/aiddnet/data/bot/` (каталог `0700`):
`live.sock` (бот → сайт; сайт пишет в него только подписку на поток мухи, задача 7.4, `docs/formats.md` §21.2/§27.3, управления через него нет), `control.sock` (сайт → бот: команды), `relations.json` (списки; пишет сайт, читает бот).

Что изменено в юните `deploy/systemd/ddnet-ai-web.service` (ревью 5.6, F2):

- `ReadWritePaths` получил `/home/ubuntu/aiddnet/data/bot`. Без этого (`ProtectHome=read-only`) каждое добавление в список давало бы 500
  `relations_write_failed`. Сам сокет `control.sock` юнит только открывает на соединение, это разрешено (`AF_UNIX`).
- `--replay …/traces/oracle-b/v1` заменён на `--bot-socket /home/ubuntu/aiddnet/data/bot/live.sock`: эти флаги несовместимы (источник кадров один).
  **Цена:** пока бот не запущен, вкладка «Игра» не показывает реплеи корпуса Oracle B, а пишет, что бота нет. Вернуть реплеи: в юните заменить
  `--bot-socket …` на `--replay /home/ubuntu/aiddnet/data/traces/oracle-b/v1` (тогда у вкладки «Бот» нет статуса, редактор и команды работают).
  Совмещать «живой бот, а при его отсутствии реплеи» не стали: это второй источник кадров с переключением, а не флаг.
- `deploy/install.sh` создаёт `~/aiddnet/data/bot` с `0700` до старта юнита. Если каталога нет, юнит не поднимется (`status=226/NAMESPACE`):
  `mkdir -p -m 700 ~/aiddnet/data/bot`.

Точные шаги (делает лид после ревью):

1. Влить ветку; `deploy/install.sh` (пересборка `~/aiddnet/bin/ddnet-ai`, новый юнит, `daemon-reload`, перезапуск `ddnet-ai-web` и `caddy`). Пароль и секреты не трогаются; сессии сбрасываются перезапуском.
2. Проверить юнит: `systemctl show ddnet-ai-web -p ReadWritePaths` (должен содержать `data/bot`) и `systemctl is-active ddnet-ai-web`.
3. Бота запускать **с теми же путями**: `~/aiddnet/bin/ddnet-ai play … --data-dir ~/aiddnet/data` (по умолчанию `bot/live.sock`, `bot/control.sock`, `bot/relations.json`, аудит команд
   `logs/bot/control-audit.log`). Если у бота свой `--relations` или `relations =` в `settings.toml`, сайт после правки напишет «бот перечитал другой файл списков»: привести пути к одному файлу.
4. Войти на сайт, вкладка «Бот»: статус «В игре» (если бот в игре), кнопки, добавить тестовое имя в «Друзья», убедиться в ответе «применено к работающему боту (бот перечитал тот же файл)», потом убрать его.
   `ls -l ~/aiddnet/data/bot/relations.json` — `-rw-------`.
5. Журнал: ники в `~/aiddnet/data/logs/web/` и в журнале бота не пишутся; в `control-audit.log` — только метки.

Откат: `git checkout <старый коммит> -- deploy/` и `deploy/install.sh --skip-build` (вернёт `--replay` и прежние `ReadWritePaths`).

## Пароль владельца

```bash
~/aiddnet/bin/ddnet-ai web-passwd
```

Генерирует новый случайный пароль (≥ 20 символов, `OsRng`), пишет **только** его argon2id-хэш в
`~/aiddnet/data/secrets/web-auth.toml` и сам пароль (открытым текстом, единственный раз) в
`~/aiddnet/data/secrets/web-password.txt` (оба файла — `0600`, каталог `secrets/` — `0700`).
Ничего не печатает на экран, кроме путей к файлам (флаг `--show` печатает и сам пароль — не
использовать в присутствии посторонних глаз/логов). Новый пароль подхватывается на следующей
попытке входа без перезапуска (файл читается на каждый `POST /api/login`, а не один раз при
старте) — но **старая сессия — не то же самое, что старый пароль**:

```bash
sudo systemctl restart ddnet-ai-web
```

Старый пароль после `web-passwd` больше не работает, но **активная сессия, вошедшая по старому
паролю, сама по себе продолжает работать** до истечения таймаута (12 ч простоя / 7 дней
максимум) — сессии живут в памяти процесса и не привязаны к текущему хэшу пароля при каждой
проверке (в отличие от пароля). Если пароль меняется именно потому, что старая сессия могла
утечь (а не просто "для порядка") — обязательно выполнить `systemctl restart` выше сразу после
`web-passwd`: перезапуск сбрасывает таблицу сессий целиком (все, включая утёкшую, требуют
повторного входа), а не только проверку пароля. Сама команда `ddnet-ai web-passwd` печатает это
же напоминание после генерации.

`web-passwd` также очищает `~/aiddnet/data/secrets/web-devices.toml` (доверенные устройства —
обход общего лимита входа, задача 5.3, находка ревью F9) на диске; подробности и почему это
безопасно даже при работающем процессе — `docs/SETUP.md` §6.

## Логи

| Что | Где |
|---|---|
| Аудит-лог входов бота (IP, успех/неудача, без пароля) | `~/aiddnet/data/logs/web/web.log.<дата>` и `journalctl -u ddnet-ai-web.service` |
| Access-лог Caddy (JSON, ротация) | `/var/log/caddy/access.log` (нужен `sudo`; читает `caddy:caddy`, `0750`) |
| Системный лог Caddy (запуск, ACME, ошибки конфига) | `journalctl -u caddy.service` |

```bash
journalctl -u ddnet-ai-web.service -f
sudo journalctl -u caddy.service -f
sudo tail -f /var/log/caddy/access.log
```

## Остановить всё

```bash
sudo systemctl stop ddnet-ai-web caddy
```

Порты `80`/`443` в ufw при этом остаются открытыми (см. ниже, как их закрыть отдельно) — просто
никто на них больше не слушает: ufw пропускает TCP-SYN до самого хоста, а раз слушателя нет, ядро
сразу отвечает RST — снаружи это выглядит как обычное "соединение отклонено" (`curl`: *Connection
refused*), а не молчаливое "сервер не отвечает" (таймаут). Запустить обратно: `sudo systemctl
start caddy ddnet-ai-web` (в этом порядке — не критично, `ddnet-ai-web.service` не требует Caddy
для собственного старта, но так Caddy сразу находит бота живым).

## Закрыть порты 80/443 снова

```bash
sudo ufw delete allow 80/tcp
sudo ufw delete allow 443/tcp
sudo ufw status verbose   # убедиться, что 22/tcp остался, 80/443 ушли
```

`22/tcp` этой командой не трогается — ufw никогда не переиспользует одно правило для нескольких
портов. Само по себе это не останавливает Caddy (он продолжит слушать локально) — сочетайте с
`systemctl stop caddy`, если нужно и то, и другое.

## Сертификат (Let's Encrypt)

Автоматический выпуск и продление — не требует никаких действий. Проверить:

```bash
curl -I https://89-58-7-133.sslip.io                       # без -k: обрывается, если сертификат невалиден
echo | openssl s_client -connect 89-58-7-133.sslip.io:443 -servername 89-58-7-133.sslip.io 2>/dev/null \
  | openssl x509 -noout -issuer -dates
sudo journalctl -u caddy.service | grep -i "certificate obtained\|renew"
```

Caddy продлевает автоматически заметно раньше истечения (сертификаты Let's Encrypt живут 90
дней); собственный внутренний таймер Caddy сам решает, когда — вручную запускать продление не
нужно. Если сертификат когда-то не выпустился (ACME-ошибка) — смотреть точный текст ошибки в
`journalctl -u caddy.service` и **не** гонять `deploy/install.sh`/`systemctl reload caddy` в
цикле: у Let's Encrypt есть лимиты (5 неудачных попыток в час на хост, 5 дублей сертификата за
7 дней) — см. `docs/research/rust-stack.md` §5.

## Откат

- **Откатить только конфиг/юнит** (не бинарник): `git checkout <старый коммит> -- deploy/` в
  основном репозитории, затем `deploy/install.sh --skip-build`.
- **Откатить бинарник бота**: переключиться на нужный коммит всего репозитория и
  `deploy/install.sh` (пересоберёт из этого коммита). Старый бинарник по пути `~/aiddnet/bin/`
  не хранит историю версий — если нужен конкретный старый бинарник без пересборки, сохранить его
  копию заранее (`cp ~/aiddnet/bin/ddnet-ai ~/aiddnet/bin/ddnet-ai.bak` до обновления).
- **Полностью выключить и убрать с публики** (не удаляя ничего): `sudo systemctl stop ddnet-ai-web
  caddy` + закрыть порты (см. выше). Пароль/секреты/сертификаты не трогаются — повторный
  `deploy/install.sh` + `sudo systemctl start` возвращает всё как было.

## Безопасность (admin API, автообновления)

- **Admin API Caddy** — unix-сокет `/var/lib/caddy/admin.sock` (режим `0600`, владелец `caddy`),
  не TCP. Диагностика (нужен root, т.к. `ubuntu` не входит в группу `caddy`):
  ```bash
  sudo curl --unix-socket /var/lib/caddy/admin.sock http://localhost/config/ | head -c 200
  ```
- **Автообновления Caddy** — `/etc/apt/apt.conf.d/51unattended-upgrades-caddy` разрешает
  `unattended-upgrades` origin `cloudsmith/caddy/stable` (Ubuntu-репозитории и так разрешены по
  умолчанию). Проверить, что патчи безопасности реально применяются:
  ```bash
  sudo unattended-upgrade --dry-run --debug 2>&1 | grep -i caddy
  ```
  Ничего вручную запускать не нужно — это часть штатного таймера `apt-daily-upgrade.timer`.

## Проверка, что всё поднимается при перезагрузке

```bash
systemctl is-enabled ddnet-ai-web.service caddy.service   # оба должны быть "enabled"
```

`deploy/install.sh` сам делает `systemctl enable` для обоих юнитов при каждом прогоне. Полный
тест перезагрузки не входит в задачу 5.3 (см. `docs/SETUP.md`/BUILD REPORT) — `is-enabled`
достаточен, так как оба юнита не завязаны на что-либо, зависящее от порядка старта, кроме
`network.target`/друг друга (`Before=caddy.service`, не `Requires=`).

## Бот под systemd (задача 4.4)

`deploy/systemd/ddnet-ai-bot.service` — сам бот («Муха», `ddnet-ai play --brain hybrid`). **`deploy/install.sh` его не
ставит и не включает**, и юнит нельзя включить командой `systemctl enable` (в нём нет `WantedBy`): после перезагрузки сам он
не стартует, запускать его всегда явно. В поставляемом виде он играет **только на локальном сервере** `127.0.0.1:8303`
(D-043): `IPAddressDeny=any` с разрешённым loopback — это ограничение на уровне cgroup, и оно сильнее любой правки `--server`.

### Установка и запуск

```bash
mkdir -p -m 700 ~/aiddnet/data/bot      # уже есть, если стоит веб-юнит (его создаёт deploy/install.sh); юнит бота требует, чтобы он был
mkdir -p ~/aiddnet/data/run ~/aiddnet/data/logs/play ~/aiddnet/data/logs/bot ~/aiddnet/data/maps/cache
sudo install -m 0644 deploy/systemd/ddnet-ai-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start ddnet-ai-bot            # НЕ enable
journalctl -u ddnet-ai-bot -f                # лог (нужен sudo или группа systemd-journal)
sudo systemctl stop ddnet-ai-bot             # вежливое отключение (SIGTERM), отчёт пишется при остановке
```

Бинарник — `~/aiddnet/bin/ddnet-ai` (его же запускает веб-юнит; `deploy/install.sh` кладёт его туда). Новую сборку бота кладут
туда же (`install -m 0755 target/release/ddnet-ai ~/aiddnet/bin/`) и перезапускают юниты, которые им пользуются.

Что юнит пишет, и только это (`ProtectSystem=strict`): `~/aiddnet/data/bot` (память о фризах `memory/<sha256 карты>.json`,
клипы `clips/`, сокеты `live.sock` (мост, бот → сайт) и `control.sock` (сайт → бот), списки `relations.json` (их правит сайт), `settings.toml`,
`last-report.json`), `~/aiddnet/data/logs` (`play/play.log.<дата>`, аудит команд сайта `bot/control-audit.log`),
`~/aiddnet/data/maps/cache` (карта, скачанная с сервера), `~/aiddnet/data/run` (замки «один бот на сервер»). Остальное только
на чтение; `MemoryMax=2G`, `TasksMax=256`, `MemoryDenyWriteExecute`, без привилегий, адреса — только AF_INET/AF_INET6/AF_UNIX.

### Что делает systemd при выходе бота

| Код выхода | Значит | Юнит |
|---|---|---|
| `0` | остановлен по просьбе (`systemctl stop`, SIGTERM/SIGINT) | не перезапускается |
| `3` | кик, бан, или бот переведён в зрители после игры (модерация, D-016) | **не перезапускается** (`RestartPreventExitStatus=3 4`), юнит остаётся `failed` |
| `4` | не смог войти, исчерпаны попытки, или больше 3 обрывов связи в игре за 600 с (D-050, D-058) | **не перезапускается**, `failed` |
| `1`, паника, сигнал | сбой | `Restart=on-failure` через 10 с; больше 3 запусков за 10 минут — стоп (`StartLimitBurst`) |

После `3` или `4` сначала прочитать журнал (`journalctl -u ddnet-ai-bot -n 100`) и записать случай в `docs/STATUS.md` (CLAUDE.md:
кик и бан не обходим), только потом `sudo systemctl reset-failed ddnet-ai-bot` и новый `start`.

### Журнал: что в нём есть и чего нет

- Консоль выключена: `StandardInput=null` и `--no-console`. **Канал управления с сайта включён** (задача 5.6, D-070): бот открывает `control.sock`
  (`--no-control` в юнит не добавлен), и владелец командует ботом через вкладку «Бот» боевого веб-юнита (закрытый список команд, чата по построению нет,
  лимит 2 в секунду, аудит с метками в `logs/bot/control-audit.log`). Править списки друзей / войны / игнора можно только там же: файл `relations.json` общий у бота и сайта.
- Свободный текст сервера (причина кика или отключения) бот пишет в журнал, заменив известные ему ники и кланы игроков на теги, и добавляет длину оригинала
  (`reason_len`); клиентский слой пишет о причине только длину.
- Других игроков в журнале называют **теги** `c<id>-<хэш>` (соль меняется при каждом запуске); настоящие ники только с
  `--debug-names`/`--console-names`/`--web-names`, в юнит они не добавлены. Свой ник (`Muha`) бот в лог тоже не пишет (он виден только в `systemctl status`,
  в строке команды). Проверка: soak 4.4 искал по журналу юнита ники трёх скриптовых ботов и `Muha` (см. E-009), нашёл 0.
- В игровой чат бот не пишет ничего (D-007): `last-report.json` содержит аудит исходящих сообщений (`outgoing_game_messages`):
  только служебные при входе и `Cl_Kill`.
- Цвета в журнале выключены (`NO_COLOR=1`), уровень `info`.

### Веб-юнит и бот: общий каталог `data/bot`

Боевой `ddnet-ai-web.service` уже запущен с `--bot-socket /home/ubuntu/aiddnet/data/bot/live.sock` и имеет `/home/ubuntu/aiddnet/data/bot` в
`ReadWritePaths` (5.6, раздел «Управление ботом с сайта» выше): мост он только читает (плюс подписка на поток мухи), `control.sock` открывает на соединение,
`relations.json` пишет. Поэтому юниту бота ничего менять в вебе не нужно: достаточно запустить бота с теми же путями (`--data-dir ~/aiddnet/data`, как в юните).
Каталог `data/bot` обязан существовать **до** перезапуска любого из двух юнитов: без него юнит не поднимется (`status=226/NAMESPACE`). Карту веб берёт из
`--maps-dir ~/aiddnet/data/maps/cache`, куда бот кладёт скачанные карты. Песочница веб-юнита (`ProtectHome=read-only`) подключаться к сокету бота не мешает: soak 4.4 проверил это на отдельном
веб-юните (порт 7790) с теми же свойствами (E-009).

### Публичный сервер (когда владелец разрешит, D-043, D-052)

Три явных шага, ни один не делается сам:

1. В `~/aiddnet/data/live-servers.toml` у записи сервера `ready = true` (иначе клиент откажется подключаться: D-067).
2. `sudo systemctl edit ddnet-ai-bot` — drop-in с адресом сервера и **разрешением адреса в cgroup**:
   ```
   [Service]
   ExecStart=
   ExecStart=/home/ubuntu/aiddnet/bin/ddnet-ai play --server <ip>:<port> --name Muha --brain hybrid --duration 0 --no-console --data-dir /home/ubuntu/aiddnet/data --report /home/ubuntu/aiddnet/data/bot/last-report.json
   IPAddressAllow=<ip>
   ```
3. Первый запуск короткий и при владельце (D-068: 10-15 минут), `journalctl -u ddnet-ai-bot -f` открыт.

Один бот на сервер (CLAUDE.md): не запускать одновременно юнит и ручной `ddnet-ai play` на тот же внешний адрес.

### Проверка (soak 4.4)

`tools/e2e/soak.sh --label D-wbauto-unit --unit` ставит этот юнит на время прогона (drop-in подменяет путь бинарника на свежую сборку и даёт **частный каталог данных**
`<прогон>/botdata` со своими `--data-dir`, `--bridge`, `--control`, `--settings`, `--relations` и `ReadWritePaths` только на него и на `data/run`), гонит час игры на
локальном сервере и убирает юнит. **`~/aiddnet/data/bot` стенд не использует, не переносит и не удаляет**: в конце он сверяет, что каталог остался таким, как был
(список, права, размеры, время изменения), и пишет результат в вердикт. Результаты, таблицы и вердикт — `docs/EXPERIMENTS.md`, E-009. Для ≥ 6-часового прогона без присмотра
(он нужен до многодневного запуска) стенд пока не рассчитан: гейт памяти считает базу на 25-й минуте и требует ≥ 35 минут.

## Показ: муха на арене, пока бота нет (задача 5.7, D-075)

Пока живой бот не запущен, вкладки «Игра» и «Муха» показывают **демо**: обученная муха играет арену `clb-left` против скриптового соперника (`ddnet-ai fly watch`, без игрового
сервера). Плашка говорит словами: «Показ: муха на арене (не настоящая игра)» (карта, арена, веса). Когда бот поднимается, сайт сам переключается на него («Живой бот»), когда бот пропадает, возвращается
к демо. Вкладка «Бот» относится **только к живому боту**: со показом там «бот не запущен»; если при этом нет и сокета управления бота (`control.sock`), кнопки команд выключены, а сервер отвечает на команду `503 demo_only` и ничего никуда не шлёт (у демо нет сокета
управления, `control.sock` принадлежит боту). Бот, который работает без моста (`--no-bridge`) или чей мост у сайта переподключается, свой `control.sock` сохраняет и остаётся управляемым, даже пока на сайте показ. Возврат к показу после обрыва моста бота идёт с задержкой 2,5 с (переход на бота мгновенный). Формат — `docs/formats.md` §28.

Два юнита: **`deploy/systemd/ddnet-ai-flydemo.service`** (новый, демо; `deploy/install.sh` его не ставит) и **`ddnet-ai-web.service`** (в репозитории получил `--demo-socket
/home/ubuntu/aiddnet/data/flydemo/fly-demo.sock`; ставит его `deploy/install.sh`).

- **Свой сокет в своём каталоге.** Демо слушает `~/aiddnet/data/flydemo/fly-demo.sock` (каталог `0700`), не в `data/bot` и никогда не `live.sock`; веб отказывается стартовать, если `--demo-socket` совпадает с `--bot-socket`. Вебу достаточно подключаться к сокету (это работает и при `ProtectHome=read-only`), поэтому его `ReadWritePaths` не менялись.
- **Железо.** `Nice=10`, реальное время (50 тиков в секунду), песочница как у веба и бота плюс **без сети вообще** (`PrivateNetwork=true`, только `AF_UNIX`, `IPAddressDeny=any`), пишет только в `data/flydemo`,
  а **весь** `data/bot` для юнита недоступен (`InaccessiblePaths=/home/ubuntu/aiddnet/data/bot`, без `-`: каталог есть всегда). Проверено временными системными юнитами с теми же свойствами: при отсутствующих и при существующих `live.sock` / `control.sock`, и после их
  пересоздания хозяином, демо не может ни подключиться к ним, ни удалить и занять, ни перечислить каталог (маска отдельных сокетов с `-` этого не давала: она пропускает отсутствующий файл и пропадает, когда бот пересоздаёт сокет), `MALLOC_MMAP_THRESHOLD_=131072`, `MemoryMax=512M`, `TasksMax=64`.
- **Цена (замер, release, `nice -n 10`, общая машина с load average ~7, процент одного ядра по `utime+stime`):** нет зрителей — **0,05%** (игра стоит: `--pause-idle`, юнит просыпается 10 раз в секунду);
  открыта страница («Игра») — **2,6%** (154 тика по 10 мс за 60 с); открыта ещё и вкладка «Муха» — **2,9%** (171 тик за 60 с); RSS 18 МиБ, 2 потока. Пока бот работает, демо ни о чём не просят, и оно стоит.
  «Зритель» — любое открытое WS-соединение сайта (в том числе забытая вкладка): пока она открыта, демо играет, если бота нет.
- **Веса настраиваются** без правки юнита: по умолчанию `FLYDEMO_BUNDLE=/home/ubuntu/aiddnet/data/runs/E-005/e005-fly/checkpoints/final.bundle`; лучший bundle (задача 8.2b) — строка
  `FLYDEMO_BUNDLE=/путь/к/new.bundle` в необязательном `~/aiddnet/data/flydemo.env` и `sudo systemctl restart ddnet-ai-flydemo`. `ExecStartPre` проверяет, что файл читается (иначе юнит сразу падает с понятной строкой).
  Bundle формата 3 (E-008) читается только сборкой с 8.2b.

Точные шаги (делает лид после ревью; ничего из этого ветка не выполняла, боевые юниты и Caddy не трогались):

1. Влить ветку; собрать и поставить бинарник и веб-юнит: `deploy/install.sh` (пересборка `~/aiddnet/bin/ddnet-ai`, новый `ddnet-ai-web.service` с `--demo-socket`, `daemon-reload`, перезапуск `ddnet-ai-web` и `caddy`;
   сессии сбрасываются). Если бот (`ddnet-ai-bot`) или 4.5 сейчас работает на `live.sock`, ему перезапуск веба не мешает (мост переподключается), но бинарник бота остаётся прежним до его собственного перезапуска.
2. Каталоги: `~/aiddnet/data/bot` существует (`0700`, он нужен веб-юниту и боту); **создать `mkdir -p -m 700 ~/aiddnet/data/flydemo`** до первого запуска демо (без него юнит не поднимется, `status=226/NAMESPACE`). Проверка файлов: `ls ~/aiddnet/data/runs/E-005/e005-fly/checkpoints/final.bundle`,
   `ls ~/aiddnet/data/maps/copy-love-box/'Copy Love Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25.map'`, `ls ~/aiddnet/data/connectome/compiled/fly-S-v1.flyg`.
3. Поставить и включить демо (в отличие от юнита бота, он включается на загрузку: он безвреден и сети не имеет):
   ```bash
   sudo install -m 0644 deploy/systemd/ddnet-ai-flydemo.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now ddnet-ai-flydemo
   systemctl is-active ddnet-ai-flydemo && journalctl -u ddnet-ai-flydemo -n 5   # "serving the fly's stream on …/fly-demo.sock"
   ls -l ~/aiddnet/data/flydemo/fly-demo.sock                                      # srw------- (0600)
   systemd-analyze security ddnet-ai-flydemo --no-pager | tail -1                 # exposure level ~2 (OK)
   ```
4. Проверить на сайте (бот не запущен): «Игра» — плашка «Показ: …», карта Copy Love Box, муха и «соперник 1» двигаются, камера на мухе; «Бот» — «бот не запущен», кнопки серые; «Муха» — «муха работает»,
   веса `e005-fly/final`. `top -b -n1 -p $(systemctl show -p MainPID --value ddnet-ai-flydemo)` — порядка 3% на открытой странице, ~0% после закрытия всех вкладок (через пару секунд).
5. Переключение: запустить бота (`sudo systemctl start ddnet-ai-bot`, или идёт 4.5) — плашка «Живой бот», его карта и статус «В игре»; остановить — демо возвращается за секунды. Пока бот работает, `top` на демо ~0%.
6. Если на сайте «Нет источника: бот не запущен» при работающем демо: `ls -l ~/aiddnet/data/flydemo/fly-demo.sock`, `journalctl -u ddnet-ai-flydemo`, `systemctl show ddnet-ai-web -p ExecStart | grep demo-socket`.

Откат: `sudo systemctl disable --now ddnet-ai-flydemo` и `git checkout <старый коммит> -- deploy/systemd/ddnet-ai-web.service && deploy/install.sh --skip-build` (веб без `--demo-socket`: вкладки снова пишут, что бота нет). Новый
бинарник веба с `--demo-socket` без юнита демо работает как раньше (источник «нет», пока сокета нет).
