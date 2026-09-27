# DDNet-AI (TypeScript) — runtime-часть: заметки для порта на Rust

Источник: `~/aiddnet/DDNet-AI` (read-only). Все ссылки вида `файл:строка` относятся к этому дереву;
`lib/…` = `node_modules/teeworlds/lib/…` (teeworlds v2.6.1). Прочитаны целиком: `src/bot/bot.ts` (5004 стр.),
`liveWorld.ts`, `main.ts`, `console.ts`, `serverPick.ts`, `netPatch.ts`, `opponentProfile.ts`, `cpuLoad.ts`, `ui.ts`,
`mascot.ts`, `terminalSafe.ts`, `autoChat.ts`, `autoUpdate.ts`, `ownerOrders.ts`, `dummyThread.ts`, `dummyWorker.ts`,
`src/plan/livePlan.ts`, `src/plan/memory.ts`, `src/watch/incidents.ts`, `src/watch/recording.ts`, `src/i18n.ts`,
`start.mjs`, `run.sh`, `README*.md`, библиотека `teeworlds` (`client.js`, `snapshot.js`, `components/*`, `huffman.js`,
`MsgPacker/MsgUnpacker`, `UUIDManager`). Для контекста просмотрены (не целиком): `src/demo/reckoning.ts`,
`src/core/types.ts`, `src/core/tuning.ts`, `src/plan/seal.ts`, `src/plan/shield.ts`, `src/plan/planner.ts` (только
`PLANNER_DEFAULTS`), `src/bot/wayblock.ts`, `src/map/loadMap.ts` (грепом), `src/core/world.ts:380-470`.
Помечено «не проверено» то, что не удалось подтвердить по коду.

---

## 0. Главное в двух абзацах

Бот — UDP-клиент **протокола Teeworlds 0.6 с расширениями DDNet** (токен-хендшейк `TKEN`, UUID-сообщения
`NETMSG_EX`, `CLIENTVER` с версией 19000, ex-объекты снапшота DDNet). Сетевую часть даёт npm-библиотека `teeworlds`,
которую `netPatch.ts` исправляет в трёх местах (защита huffman от бесконечного цикла, обработка `NETMSG_REDIRECT`,
линейный вместо O(n²) декодер дельт снапшота). Карта скачивается с сервера (HTTP по `map_details` или чанками по UDP)
и парсится собственным парсером `src/map/loadMap.ts`; фоллбэк — локальная папка карт. **Тюнинг с сервера не
читается** (`SV_TUNE_PARAMS` игнорируется), используются дефолты DDNet из `src/core/tuning.ts`.

На каждый **собранный снапшот (25 Гц, шаг 2 серверных тика)** бот один раз принимает решение
(`bot.ts:2352-2604`): обновляет `LiveWorld`, считает «эхо»-задержку по собственному прицелу, выбирает цель
(`pickTarget`), при необходимости идёт навигацией/на ВБ/бродит, иначе зовёт планировщик, которому передаётся
симуляционный мир, прокрученный вперёд на `lag` тиков (0..6) с уже отправленными («летящими») вводами. Ввод
отправляется сразу после решения (`client.sendInput()`), плюс библиотека сама переотправляет текущий ввод каждые
50 мс. Всё остальное — периферия: клипы (кольцевой буфер 30 с), память фризов по карте, списки
friend/war/ignore, авто-выбор сервера через master-сервер DDNet, /kill при зависании во фризе.

Навигация по вопросам: (1) файлы — §1; (2) конвейер/тайминг/ввод/состояние — §4 (+§2.2); (3) эхо-лаг, пинг,
прокрутка — §5; (4) LiveWorld — §6; (5) pickTarget/списки/sealed — §7; (6) режимы/команды/CLI/settings — §8, §9;
(7) клипы/инциденты/FreezeMemory — §10; (8) serverPick — §11; (9) реконнект/карта/redirect/версия — §2.4, §3, §2.1;
(10) CPU — §12; (11) баги/хардкод — §13; риски порта — §14.

---

## 1. Таблица файлов

| Файл | Назначение | Главные типы/функции | Зависимости | Статус порта |
|---|---|---|---|---|
| `src/bot/bot.ts` | Весь runtime-бот: соединение, конвейер снапшота, выбор цели, лаг-компенсация, навигация/ВБ/тrek, unstick, клипы, консоль-команды, списки, дуэли, чат, LLM-приказы, партнёр-дамми | `class DdnetBot` (`bot.ts:694`), `onSnapshot` (2352), `pickTarget` (3020), `planAction` (4695), `applyInput` (4828), `handleConsole` (1310), `measureEcho/echoLagTicks/lagTicks` (4494-4563), `maybeUnstick` (4574), `guard` (2801), `refreshInputClock` (2916) | teeworlds, liveWorld, livePlan, planner, seal, shield, route, memory, recording, incidents, navigate, crossing, wayblock, cpuLoad, netPatch, autoChat, ownerOrders, i18n | **PARTIAL**: порт ядра; DROP чат/автоответы/brush-off/LLM/owner/duel-accept/партнёры/rescue друзей/эмоции (опц.) |
| `src/bot/liveWorld.ts` | Мир из снапшота: декодирование персонажей, ex-объектов DDNet, снарядов, лазеров, флагов игроков; загрузка коллизии карты | `LiveWorld` (195), `decodeCharacter` (131), `jumpsLeftOf` (121), `projectileFromItem` (91), `exTypeId` (80), `mapCollisionFromClient` (339) | reckoning, collision, projectile, loadMap, demo/snapshot (UUID) | **PORT** |
| `src/bot/main.ts` | Альтернативный CLI-вход (без меню/веба/авто-сервера) | `main()` (57), `USAGE` (11) | bot, console, gru | **PARTIAL** (флаги как референс) |
| `src/bot/console.ts` | Простая TTY-консоль: строка статуса, ввод команд | `BotConsole` (53) | bot, mascot, terminalSafe | **PARTIAL** (консоль нужна, оформление — нет) |
| `src/bot/serverPick.ts` | Выбор «живого» блок-сервера из master-списка DDNet; файл-«чёрный список» | `MASTERS` (4), `parseMaster` (48), `isBlock` (78), `pickBlockServer` (82), `fetchMaster` (99), `readAvoid/addAvoid` (115/126) | fetch, fs | **PORT** |
| `src/bot/netPatch.ts` | Патчи библиотеки: huffman, redirect, снапшот-декодер; глобальный «сторож» исключений | `patchHuffman` (21), `installNetworkGuard` (93), `patchRedirect` (114), `patchSnapshotDecoder` (189) | teeworlds internals | **PORT по смыслу** (в Rust свой клиент; перенести исправления) |
| `src/bot/opponentProfile.ts` | Онлайн-статистика соперника (агрессия, кто первым хукает, успех хука, без прыжков) | `OpponentProfile` (70), `Rate` (13) | types | **DROP/опц.**: используется планировщиком только при `opponentReadWeight>0` (дефолт 0, `planner.ts:273,1050`) |
| `src/bot/cpuLoad.ts` | Константы режимов low/strong и детектор «ПК не успевает» | `LOW_CPU` (17), `STRONG_WB` (19), `LagWatch` (54) | — | **PORT** (константы; детектор — опц.) |
| `src/bot/ui.ts` | Ink/React TUI | `startInkUi` | ink, react | **DROP** (или своя TUI) |
| `src/bot/mascot.ts` | ASCII-мордочка для консоли | `Mascot` | — | **DROP** |
| `src/bot/terminalSafe.ts` | Вырезание управляющих символов из строк (ники!) | `terminalSafe` | — | **PORT** (тривиально, но нужно: ники с ESC) |
| `src/bot/autoChat.ts` | Автосообщения в чат по таймеру/ключевым словам/упоминанию | `AutoChat` | fs | **DROP** (чат) |
| `src/bot/autoUpdate.ts` | Автообновление с GitHub (tarball), рестарт кодом 75 | `startAutoUpdate`, `apply` | fetch, tar | **DROP** |
| `src/bot/ownerOrders.ts` | Приказы «хозяина» из чата, регэкспы + LLM (OpenAI-совм. API) | `addressed`, `parseOrder`, `modelOrder`, `LLM_PRESETS` | fetch | **DROP** |
| `src/bot/dummyThread.ts` / `dummyWorker.ts` | Второй бот («дамми») в worker_thread, синхронизация списков и partnerId | `DummyThread`, `shareableLists`, `listUpdates` | worker_threads, bot | **DROP** |
| `src/plan/livePlan.ts` | Синхронизация планировочного SimWorld со снапшотом + прокрутка на лаг; угадывание ввода чужих | `enemyInputFromSnapshot` (5), `syncPlanningWorld` (16), `syncPlanningWorldLegacy` (42), `syncOthers` (69) | world, types | **PORT** |
| `src/plan/memory.ts` | «Память фризов» по тайлам карты (фризы/проходы), персистентная | `FreezeMemory` (10) | fs | **PORT** |
| `src/watch/incidents.ts` | Поиск «инцидентов» в записи (самофриз, медленный рехук, дрожание и т.д.) для автоклипов | `findIncidents` (52), `mergeOverlapping` (259), `summarise` (269) | recording, tuning | **PORT** (для автоклипов) |
| `src/watch/recording.ts` | Формат записи/клипа и кольцевой буфер | `RecFrame`, `Recording`, `RingRecorder` (148), `snapTee` (78), `snapInput` (144), `recTeeState` (104) | collision | **PORT** (формат можно упростить/сменить) |
| `src/i18n.ts` (+`i18n-en.ts`) | Строки ru→en словарём, `t()` с `{param}` | `t`, `tr`, `setLang`, `detectLang` | — | **PARTIAL/DROP** (лог-строки; решать отдельно) |
| `start.mjs` | Главный вход: мастер-вопросы, settings.json, авто-сервер, веб-UI, дамми, автообновление, выбор мозга | `main()` | всё | **PARTIAL** (логика запуска/настроек/авто-сервера) |
| `run.sh` | Цикл перезапуска: пока код выхода 75 — перезапуск `node start.mjs` | — | — | **PORT идеи** (рестарт при смене сервера) |
| `node_modules/teeworlds` | 0.6+DDNet клиент | `Client`, `Snapshot`, `SnapshotWrapper`, `Movement`, `Game`, `TwMap`, `Huffman` | dgram, zlib, crypto | **PORT (переписать)** |

Не из списка, но задействованы runtime-ом (порт, изучать отдельно): `src/bot/navigate.ts` (Navigator, goto),
`src/bot/crossing.ts` (перелёт через фриз-трубы на верёвке), `src/bot/wayblock.ts` (ВБ Copy Love Box),
`src/plan/route.ts` (`findRoute`, `deadZone`, `spawnTiles`), `src/plan/seal.ts`, `src/plan/shield.ts`,
`src/plan/planner.ts`, `src/core/*`, `src/demo/reckoning.ts`, `src/map/*`. Веб-UI `src/bot/web*.ts` и `app/` (Electron) — DROP.

---

## 2. Сетевой стек: библиотека `teeworlds` 2.6.1 и патчи

### 2.1 Протокол
* **0.6**, не 0.7: строка версии `"0.6 626fce9a778df4d4"` в `NETMSG_INFO` (`lib/client.js:440`); коды системных
  сообщений 0.6: `INFO=1, MAP_CHANGE=2, MAP_DATA=3, CON_READY=4, SNAP=5, SNAPEMPTY=6, SNAPSINGLE=7, READY=14,
  ENTERGAME=15, INPUT=16, RCON_CMD=17, REQUEST_MAP_DATA=19, PING=22, PING_REPLY=23` (`client.js:517-646`).
  Таблица имён `messageTypes` (`client.js:33-36`) сдвинута и неточна — используется только для подписей.
* **Токен DDNet (0.6 security token)**: `TKEN` изначально `ff ff ff ff` (`client.js:73`), connect шлёт control-msg 1
  с текстом `"TKEN"` и токеном (`client.js:390`, `SendControlMsg` `216-232`), повтор каждые 500 мс (`391-398`).
  Ответ сервера распознаётся как `packet[0]==0x10` и (`includes("TKEN")` или `packet[3]==2`); токен = последние 4 байта
  пакета (`432-435`), затем control-msg 3 (accept) и состояние LOADING (`436-437`). **Токен дописывается в конец
  каждого исходящего пакета** (`client.js:221, 286, 317`).
* Заголовок пакета: `[(flags<<4)|(ack>>8), ack&0xff, numChunks]` (`client.js:273`); флаги пакета: 1=control,
  2=connless, 4=resend, 8=compression (сжатие только на приёме, `180-184`). Заголовок чанка 2 байта (не-vital) или 3
  (vital, с seq 10 бит) (`253-268`). ID сообщения в чанке: `2*msg + sys` (`MsgPacker.js`).
* Сразу после хендшейка шлются три vital-сообщения (`client.js:439-457`):
  1. `NETMSG_EX` с UUID `i-am-npm-package@swarfey.gitlab.io` + URL npm (**бот представляется npm-библиотекой!**);
  2. `NETMSG_EX` `clientver@ddnet.tw` (сырой UUID `8c00130484613e478787f672b3835bd4`), случайный 16-байтный UUID,
     `int version`, строка `"DDNet <release>; https://www.npmjs.com/package/teeworlds/v/2.6.1"`; версия из опции
     `ddnet_version` — бот передаёт `{version: 19000, release_version: "19.0"}` (`bot.ts:548-549, 989-992`), либо
     `--protocol-version N` → `{N, (N/1000).toFixed(1)}`. Без опции библиотека шлёт 16050. Обоснование автора
     (`main.ts:31-32`): «default 19000, needed for 128-player servers; 603 is the old, very conservative value».
     Что именно меняется на сервере от версии (ex-объекты, маппинг id >64, и т.п.) — в коде бота не описано, не проверено;
  3. `NETMSG_INFO` (версия 0.6 + пароль).
* `CON_READY` → `CL_STARTINFO` (game msg 20: name, clan, country, skin, use_custom_color, color_body, color_feet) и
  `RCON_CMD "crashmeplx"` (легаси-признак поддержки 64 игроков) (`client.js:623-646`).
* `SV_READYTOENTER` (game 8) → `NETMSG_ENTERGAME`, сброс снапшот-состояния `OnEnterGame` (`867-871`, `106-115`).
* **`connected` эмитится только после 2-го полученного снапшота** (`client.js:647-653`), тогда же STATE_ONLINE.
* UUID-сообщения (сис.): `WHATIS`→`ITIS`/`IDONTKNOW` (`699-715`), `MAP_DETAILS` (имя, sha256, crc, size, url;
  `716-729`), `CAPABILITIES` (`730-762`, только эмит события), `PINGEX`→`PONGEX` (`763-767`), `RECONNECT` →
  `_Disconnect(false).then(connect)` (`768-771`). `REDIRECT` (65548) библиотека **не** обрабатывает — это делает патч.
* Игровые сообщения, которые разбираются: `SV_MOTD`, `SV_BROADCAST`, `SV_CHAT`, `SV_KILLMSG` (killer, victim, weapon,
  special_mode; инфо только для id<64), `SV_EMOTICON`, голосования (`SV_VOTE_*`), `SV_TEAMSSTATE` (64 int),
  `SV_KILLMSGTEAM` (`774-864`). **`SV_TUNE_PARAMS` не разбирается** → серверный тюнинг неизвестен. `NETMSG_INPUTTIMING`
  тоже игнорируется.
* Надёжность (vital): `ack` клиента обновляется при seq==ack+1, иначе при «не старом» seq выставляется
  `requestResend` → флаг 4 в следующем пакете (`491-514`). Отправленные vital-чанки хранятся в `sentChunkQueue`
  (удаляются по ack сервера, `479-490`), переотправка при флаге resend от сервера (`ResendAfter`, `162-171`) или по
  таймеру 1 с при простое >900 мс (`407-413`, фактически не срабатывает, т.к. ввод шлётся каждые 50 мс).
* Keepalive (control 0) шлётся при приёме пакета, если `now - this.time >= 1000` (`874-882`), но `inputInterval`
  каждые 50 мс ставит `this.time = now` (`403`) → в онлайне keepalive фактически не уходит (трафик ввода его заменяет).
* Таймаут: 15 с без входящих пакетов → `disconnect("Timed Out…", false)` (`414-424`). Проверка каждые 5 с **без
  проверки состояния**, поэтому после таймаута событие повторяется каждые 5 с до следующего `connect()`.
* Входящие пакеты с чужого адреса/порта отбрасываются (`429`). DNS-резолв асинхронный, результат пишется в
  `this.host` без ожидания (`376-382`) — гонка для доменных имён (бот передаёт IP, так что обычно не важно).
* Игровые сообщения (`Say`, `SetTeam`, `Kill`, `Emote`, `ChangePlayerInfo`, `Vote`, `CallVote`) — vital, **ставятся в
  очередь** (`QueueChunkEx`, `components/game.js:11-15`) и уходят вместе со следующим пакетом ввода.

### 2.2 Ввод (`sendInput`, `client.js:887-905`)
`NETMSG_INPUT` (sys 16, **не vital**, flag 0): `AckGameTick, PredGameTick, size=40, direction, targetX, targetY,
jump, fire, hook, playerFlags, wantedWeapon, nextWeapon, prevWeapon`.
* `AckGameTick` = `recvTick` последнего успешно декодированного снапшота (`client.js:689`); `-1` при ошибке базы.
* `PredGameTick` — **счётчик по настенным часам**: `setInterval` 20 мс увеличивает его, если `AckGameTick>0`
  (`384-389`); если `|Pred-Ack|>10` → `Pred=Ack+1` (+ немедленный `sendInput`) (`654-655, 690-693`). Синхронизации с
  сервером (INPUTTIMING) нет. Практически `Pred ≈ Ack+1..3`, что меньше текущего тика сервера на одностороннюю задержку,
  поэтому сервер DDNet подменяет intended tick на `Tick()+1` (по памяти кода DDNet `CServer::ProcessClientPacket`,
  не проверено в этом репо). Итог: ввод применяется ≈ через one-way latency + ≤1 тик после отправки.
* Автоотправка: `inputInterval` 50 мс (20 Гц) шлёт текущий `movement.input`, если не `lightweight` (`399-406`);
  бот `lightweight` не ставит → ввод уходит **и по таймеру 20 Гц, и сразу после каждого решения** (≈25 Гц).
* `Movement` (`components/movement.js`): начальное `m_PlayerFlags=1` (PLAYING), `m_WantedWeapon=1` (хаммер+1),
  `Fire()` = `m_Fire++` (счётчик; нечётное = нажато), флаги: 1 playing, 2 in menu, 4 chatting, 8 scoreboard, 16 hookline.

### 2.3 Снапшоты (`client.js:647-697`, `snapshot.js`)
* Части `NETMSG_SNAP` (NumParts, Part) склеиваются по `GameTick`; `SNAPSINGLE`, `SNAPEMPTY`; валидация
  `NumParts 1..64, PartSize ≤ 900` (`673`). `DeltaTick = GameTick - int`.
* Формат дельты: `num_removed, num_items, _zero, removed_keys[], items(type_id, id, [size], data[])`; размер
  известен для `type_id 1..20` (таблица 0.6: `[0,10,6,5,4,3,8,4,15,22,5,17,3,2,2,2,2,3,3,3,3]`), иначе читается;
  ключ `(type_id<<16)|id`; undiff = поэлементное сложение с базой (`snapshot.js:585-598`); CRC = сумма всех int
  данных всех объектов `& 0xffffffff` (`391-402`). При несовпадении CRC: вернуть старые `deltas`, `recvTick=deltatick`
  (ack старой базы), после >5 ошибок — полный сброс и `recvTick=-1` (`555-570`).
* Ex-объекты (type_id 0): data = 4 int UUID, `id` = назначенный сервером тип. Поддерживаемые имена
  (`snapshot.js:84-91`): `my-own-object@heinrich5991.de`, `character@netobj.ddnet.tw`, `player@netobj.ddnet.tw`,
  `gameinfo@netobj.ddnet.tw`, **`projectile@netobj.ddnet.tw` (ошибочное имя, в DDNet — `ddnet-projectile@…`)**,
  `laser@netobj.ddnet.tw`. Разбор DDNetCharacter: `m_Flags, m_FreezeEnd, m_Jumps, m_TeleCheckpoint, m_StrongWeakID,
  m_JumpedTotal, m_NinjaActivationTick, m_FreezeStart, m_TargetX, m_TargetY` (`121-137`).
* Обёртка `SnapshotWrapper` (`components/snapshot.js`): `getParsed/getAll` — линейный поиск по `deltas` при каждом
  вызове (11-27); `OwnID` = `client_id` у `PlayerInfo` с `local=1` (138).
* Событие `snapshot` эмитится **до** обновления `AckGameTick` (`688-689`); `currentSnapshotGameTick` = GameTick
  собранного снапшота.

### 2.4 Карта (`client.js:527-622`, `components/twmap.js`)
* `MAP_CHANGE(name, crc, size)`: эмит `map_change`, `Flush()`. При `downloadMap:true` (бот ставит, `bot.ts:993`):
  отказ, если `size<0||>1 GiB` или имя содержит `/` или `\` (тогда READY не шлётся — клиент зависает). Если ранее
  пришёл `MAP_DETAILS` с тем же (crc, name, size) — качает `map_url` по HTTP (`fetch`), проверяет CRC32 (64 KiB
  блоками), иначе фоллбэк на UDP. UDP: `REQUEST_MAP_DATA(chunk)` **по одному чанку за раз** (следующий запрос после
  прихода текущего), `MAP_DATA(last, crc, index, size, data)`; при `last` проверка CRC — **при несовпадении `throw`
  внутри обработчика сокета** (`614-616`) → неперехваченное исключение. По завершении `parseMap()` (собственный парсер
  библиотеки, ботом не используется) и `NETMSG_READY`.
* Бот берёт `client.map.mapBuffer`, пишет во временный файл и парсит **своим** `loadMapCollision`
  (`liveWorld.ts:339-369`); фоллбэк — `<mapDir>/<name>.map` (`start.mjs:311` → `./maps`; `main.ts:9` →
  `vendor/DDNet-20.0-linux_x86_64/data/maps`). Попытка не чаще 1 раза в 2 с (`COLLISION_RETRY_MS`, `bot.ts:553, 2242-2254`).
  Т.к. READY шлётся только после загрузки, снапшоты (и `connected`) приходят уже со скачанной картой.

### 2.5 Что патчит `netPatch.ts` и зачем
1. **`patchHuffman`** (`netPatch.ts:21-79`): заменяет `Huffman.prototype.decompress`. Оригинал (`huffman.js:138-185`)
   на битом пакете без EOF-символа крутится бесконечно, наращивая выход (зависание/OOM). Патч: лимит выхода
   `1<<16` байт, `throw NetDecodeError` при «кончились биты» и «пакет кончился без EOF» (60, 66-70).
2. **`installNetworkGuard`** (93-109): глобальный `process.on("uncaughtException")`: ошибки декодирования
   (`NetDecodeError` или сообщения `/Invalid array length|No more bits|huffman produced|Invalid typed array length/` со
   стеком из `teeworlds/lib/`) — проглатываются и считаются (`stats.errors++`, событие); остальные — **re-throw**
   (крэш). Бот логирует на 1-й и каждый 50-й (`bot.ts:967-975`).
3. **`patchRedirect`** (111-139): оборачивает `Client.prototype.Unpack`; для sys-чанка с msgid 65548
   (`redirect@ddnet.org`) читает int порт и эмитит `redirect(port)` (порт 1..65535).
4. **`patchSnapshotDecoder`** (141-319): переписывает `Snapshot.prototype.unpackSnapshot` той же семантикой, но на
   `Map<tick, Map<key, data>>` (`held`) вместо массива `eSnapHolder` с `find/filter` (O(n²)) — «на слабом ПК
   отстаёт навсегда» (`bot.ts:966`). Детали: база = `held.get(deltatick)`; тики `< deltatick` выкидываются (214);
   `deltatick==-1` → сброс; пустой снапшот копирует базу в `recvTick` (226-233); нет базы при `deltatick>=0` →
   `{items:[], recvTick:-1}` (241) — при этом `this.deltas` уже очищен → обёртка видит пустой мир на этот кадр;
   переиспользует `parsed` старых объектов при равных данных (272, 297); регистрирует UUID ex-типов (278-288);
   CRC-логика как в оригинале (301-310); события 13..20 эмитятся в `SnapshotWrapper` (311).
   Замечание: при ошибке CRC неверные данные остаются в `held[recvTick]` (не удаляются) — безвредно, пока сервер
   не возьмёт этот тик базой (не проверено, возможна редкая порча).

---

## 3. Жизненный цикл соединения

* `start()` (`bot.ts:962-1042`): патчи → `watchServer()` (если `autoServer`) → `new Client(host, port, name,
  {identity, password?, ddnet_version, downloadMap:true})` → подписки: `connected`, `disconnect`, `redirect`,
  `snapshot`→`queueSnapshot`, `kill`, `message`, `map_change` → `connect()`.
* `connect()` (2025-2036): только из `STATE_OFFLINE`; ошибка `connect()` → `onDisconnect("connect failed…")`.
* `onConnected()` (2049-2077): phase online, backoff=1000, сброс `ownId/targetId/lastPos/wasAlive/wantSpectate`,
  дуэль-состояния; `FlagPlaying(true)`, `SetAim(0,-1)`, `idle()`; **`client.game.Say("/showall 1")`** (2071) —
  серверная чат-команда DDNet, чтобы получать всех игроков вне экрана; затем `refreshCollision()`.
  Внимание: при смене карты в той же сессии библиотека не переэмитит `connected` (`client.js:650` — состояние
  остаётся ONLINE), т.е. `/showall 1` повторно не шлётся (сохраняется ли ShowAll на сервере после смены карты — не
  проверено).
* `onDisconnect(reason, fromServer)` (2079-2118): чистит таймеры ответов, `planSim=null`, `sent=[]`; если
  `fromServer && autoServer && /\bban/i` → адрес в avoid на 60 мин и запрос смены сервера; если уже offline —
  выход (подавляет повторные события таймаута); иначе `disconnects++` и `scheduleReconnect()` (или reject старта при
  `reconnect:false`).
* `scheduleReconnect()` (2038-2047): задержка `reconnectDelayMs`, затем ×2 до `RECONNECT_MAX_MS=30000`; старт
  `RECONNECT_MIN_MS=1000` (555-556), сбрасывается в `onConnected` и при redirect.
* **Redirect** (999-1007): если порт отличается — `cfg.port = client.port = port`, backoff=1000, `client.Disconnect()`
  → событие `disconnect("")` → реконнект через 1 с на новый порт (хост тот же).
* **NETMSG_RECONNECT** — внутри библиотеки (`client.js:768-771`), без события `disconnect`.
* **map_change** (1011-1037): `collisionReady=false`, `onTickReset(0)`, `targetId=-1`, `lastSeenDist.clear()`,
  отмена `nav`, `home` забывается если карта другая (`homeMap`), `wbDef=null`, `wbChooser.reset()`, `navPending`
  заново из `--goto`. Коллизия подгружается из `onSnapshot` при `!collisionReady`.
* `refreshCollision()` (2242-2286): `mapCollisionFromClient` → `world.setCollision` → `wbDef = wayblockFor(map,
  collision)` (проверка размеров/спотов/якорей, `wayblock.ts:149-164`) → сброс счётчиков WB → `clipRing.clear()` →
  `deadCells = deadZone(collision, spawnTiles(collision))` (клетки, откуда нет пути к игре) → `planner.setDeadZone` →
  сохранить старую память, загрузить `FreezeMemory` для новой карты → `planner.setFreezeMemory`.
* `onTickReset(newTick)` (1258-1293): при уменьшении game tick (рестарт карты/раунда) очищает всё тик-зависимое
  (`sent`, таймеры wander, stuck-якорь, `frozenSince`, `planSim`, trek/path, все per-id карты, rescue-состояние).
* `stop()` (1044-1072): `endDuelScore`, `stopping`, снять serverWatch/guard, `saveMemory()`, `Disconnect()` c гонкой
  2500 мс.
* `maybeJoinGame` (2606-2618): если свой `PlayerInfo.team === -1` (спектатор) и не `!spec` — `SetTeam(0)` не чаще
  3 с (`JOIN_RETRY_MS`) + `ChangePlayerInfo` (identity).
* Смена сервера (авто): `switchRequested` → `start.mjs` `stop(75)` → `run.sh` перезапускает процесс (`run.sh:19-23`),
  который заново выбирает сервер с учётом avoid-файла. Ручная смена сервера в рантайме (кроме redirect) отсутствует.

---

## 4. Конвейер обработки снапшота

### 4.1 Тайминг
* Сервер DDNet: 50 тиков/с; снапшоты по умолчанию каждые 2 тика = **25 Гц** (`SNAPSHOTS_PER_SECOND=25`,
  `bot.ts:567`; `SNAPSHOT_PERIOD_MS=40`, `cpuLoad.ts:1`). `world.tick` = game tick снапшота (обычно чётный шаг 2).
* `queueSnapshot()` (2288-2308): на событие `snapshot` запоминает `snapTick = client.currentSnapshotGameTick`,
  считает `snapArrived`, и через `setImmediate` один раз вызывает `onSnapshot({tick})` — **несколько пришедших подряд
  снапшотов схлопываются в один** (обрабатывается последний, т.к. обёртка всегда даёт последнее состояние). Время
  обработки и число «пропущенных» идут в `LagWatch.note` (раздел 12).
* Одно решение на снапшот (в low-cpu планировщик коммитит решение на 2 снапшота, `commitDecisions=2`).

### 4.2 Точный порядок операций `onSnapshot` (`bot.ts:2352-2604`)
1. `stats.ticks++`; `reachLeft = lowCpu ? LOW_CPU.reachChecks(2) : -1` (2356-2357, 2336-2338).
2. Если `!collisionReady` → `refreshCollision()` (с троттлингом 2 с) (2358).
3. `ownId = snap.OwnID`; нет → `idle()`, выход (2360-2366).
4. [DROP] `autoChat.due()` → автосообщение (2368-2369).
5. `tick = saved.tick ?? currentSnapshotGameTick`; если `tick < world.tick` → `onTickReset(tick)` (2371-2372).
6. `world.updateFromSnapshot(snap, ownId, tick, rawSnapUnpacker.deltas)` (2373) — раздел 6.
7. `refreshInputClock()` (2375) — трекинг активности/AFK/атрибуции блоков (раздел 7.4).
8. `updateDuel(ownId)` (2377) — автодетект дуэли (раздел 8.8).
9. `self = world.getTee(ownId)`; нет/мёртв → `lastPos=null, wasAlive=false, wasFrozen=false`, `maybeJoinGame`,
   `idle()`, выход (2379-2389).
10. Первый живой кадр после смерти (`!wasAlive`): сброс policy, `prevInput=emptyInput()`, `sent=[]`, таймеры wander;
    запуск отложенного `pendingGoto`; `nav.respawned()` (2390-2409).
11. `trackTravel(self.pos)` (статистика пути) (2410).
12. FreezeMemory: при смене тайла и `!frozen` → `memory.notePass(x,y)` (2412-2419).
13. `measureEcho(self)` — замер задержки по эхо прицела (2420; раздел 5).
14. `recordFrame(self)` — кадр в кольцевой буфер клипа; каждые 50 кадров `maybeClip` (2421; раздел 10).
15. `maybeLogStatus` (verbose, раз в 5 с) (2422).
16. `maybeUnstick(client, self)` — может сделать `/kill` (2423; раздел 8.6).
17. Переход frozen↔free: эмоция (DROP/опц.), при заморозке `memory.note(x,y)`, сохранение каждые 20 событий
    (2425-2434).
18. `!collisionReady || !acting` → `idle()`, выход (2436-2439).
19. `updateWbSide(ownId, self)` — выбор стороны ВБ (2441).
20. Если идём на ВБ и уже в зале ниже спота на ≥3 тайла → отмена похода (2443-2452).
21. Сброс счётчиков неудач похода на ВБ, если уже в зале (2454-2461).
22. `navPending` → `gotoCommand(cfg.goto)` (`--goto`) (2463-2467).
23. Trek (поход «туда, где игра») прерывается, если в 500 px есть с кем драться (2469-2472).
24. [DROP] Отмена похода к другу, которому уже не нужна помощь (2476-2480).
25. Если `nav !== null`: если это поиск игры (`seekingGame`), ВБ-поход можно прервать, рядом (500 px) кто-то
    активный и `pickTarget != -1` → конец навигации; иначе `driveNav()` и **выход** (2481-2496).
26. `targetId = mode==="fight" ? pickTarget() : -1` (2498-2503).
27. «Seek/trek»: в fight, есть цель, seek включён, ВБ не держим, нет nav, прошло >250 тиков с прошлого похода, мы не
    вовлечены (`engagedNow`) и рядом (500 px) нет активных: сравнить толпу здесь (`crowdAt`, 600 px) и лучшее место
    (`gameSpot`); если там `tees+busy ≥ здесь + 3` дольше 200 тиков → `startTrek()` (2505-2534).
28. Если `targetId === -1` (2536-2579): `endTrek`; `idleSinceTick`; [DROP] `rescueFriend`; если не держим ВБ, нет nav,
    не дуэль и прошло >250 тиков → `gameSpot`; если туда есть маршрут без фриза (`findRoute … maxNodes 20000`) →
    `gotoCommand("x y", {throughFreeze:false})`, `seekingGame=true`; если `home` и простой >200 тиков и мы дальше 2
    тайлов → идти домой; иначе если держим ВБ и простой >50 тиков → `walkToWb`; затем `wander()` (брожение,
    привязанное к споту ВБ при нахождении в зале, с «взглядом» на `watch`), выход.
29. Цель есть: `idleSinceTick=-1`; [DROP] `rescueFriend(…, fighting=true)` (2580-2582).
30. Планировщик (`cfg.planner`): `input = planAction(ownId, targetId, self)` → `applyInput` → выход (2584-2589).
    Иначе (net/scripted): `encodeObs` → `scriptedAction` или `policy.act`+`decodeAction` → **`guard(self,input)`** →
    `applyInput` (2590-2599). Выход планировщика `guard` не проходит (у планировщика свой `shield: true`,
    `planner.ts` defaults).
31. Любое исключение → `stats.errors++`, лог (2600-2603).

### 4.3 `planAction` (`bot.ts:4695-4797`)
1. Персистентный `planSim: SimWorld(collision, {svHit:true, respawnDelayTicks:0, infiniteAmmo:true})`; пересоздаётся
   при смене ownId/коллизии (тогда `planner.reset()`); при смене цели старая цель удаляется, новая добавляется.
2. `enemyInput = enemyInputFromSnapshot(target)`; `lag = lagTicks()`.
3. `planOthers` (дефолт 0): до N ближайших живых (≤500 px, `PLAN_OTHERS_PX`) добавляются в sim с «удерживаемым»
   угаданным вводом (`syncOthers`).
4. Гранаты (тип `WEAPON_GRENADE`) в радиусе 600 px (`GRENADE_WATCH_PX`) → `sim.setGrenades(owner, spawnPos, dir,
   ageTicks)`.
5. `liveTransfer==="legacy"` → `syncPlanningWorldLegacy`; иначе `inFlightInputs(tick, lag)` → `syncPlanningWorld`
   (раздел 5.3).
6. `planner.setTravelGoal(trek ? trekGoal(self) : pathGoal(self, target))` — промежуточная точка маршрута к цели,
   если цель далеко/за стеной (4331-4351: если `d<420` и линия чиста — нет; маршрут `findRoute(…maxNodes 4000)`
   обновляется при устаревании: >25 тиков, или ≥6 тиков и цель сдвинулась >96 px / мы сошли с пути; `pathAhead`
   ищет ближайший шаг в окне 10 и берёт первый дальше 56 px).
7. `setThirdTees` (если `thirdTeeExposure>0`, в live 0 — `LIVE_PLANNER_CFG`, 590), `setFrozenBystanders` (замороженные
   не-друзья ≤160 px, `BYSTANDER_PX`), `setSpareBystanders` (≤ hookLength+64: игнор/друзья/вне игры/AFK — `spared()`),
   `setOverrides(wbPlanOverrides)` (в зале ВБ: `WB_PLAN_OVERRIDES` или `WB_PLAN_STRONG`), `setBand(wbBand)`,
   `setLiveTick(tick)`.
8. `out = planner.decide(sim, ownId, targetId, prevInput, enemyInput)`.
9. Вето хука: если `out.hook` и (есть spare-наблюдатели или мы кого-то держим): держим spare/друга → `hook=0`; или
   хук не выпущен и верёвка по направлению прицела зацепит spare раньше цели/стены (`ropeCatches`, 4817-4826) → `hook=0`.
10. `lastPlan = {target, lag, lagPing, others, arm:tryName, trek, ...planner.lastInfo}` для клипа.

### 4.4 Отправка ввода `applyInput` (`bot.ts:4828-4864`)
* `direction<0 → RunLeft, >0 → RunRight, 0 → RunStop`; `Jump(jump!=0)`; `Hook(hook!=0)` (+`hooksFired` на фронте);
  `SetAim(targetX, targetY)`; `noteAim(input)` (для эхо).
* `FlagScoreboard(tick % 50 < 2)` — раз в секунду на один снапшот включается флаг «таблица очков» (цель в коде не
  объяснена; гипотеза — «признак активности» для AFK-детекта сервера, не проверено).
* `WantedWeapon(input.wantedWeapon===0 ? WEAPON_HAMMER+1 : input.wantedWeapon)` (оружие 1-based, всегда хаммер по
  умолчанию).
* Огонь — счётчик `m_Fire`: если `firePressed(prev,cur)` (`cur.fire != prev.fire && cur.fire нечётно`, 690-692):
  если в библиотеке сейчас нечётно — `Fire()` (отпустить), затем `Fire()` (нажать), иначе один `Fire()`; иначе если
  `cur.fire` чётно, а в библиотеке нечётно — `Fire()` (отпустить) (4847-4853).
* `client.sendInput()` (в try), `prevInput = {...input}`, `recordSent(input)` (`sent` — последние 16 решений с тиком
  снапшота, 4866-4869).
* `idle()` (4973-4986): RunStop, Jump(false), Hook(false), отпустить огонь, обнулить dir/jump/hook в `prevInput`,
  `recordSent` — **без `sendInput()`** (уйдёт по 50-мс таймеру библиотеки).
* `wander()` (4882-4971) шлёт `sendInput()` сам; прицел поворачивается не быстрее 0.12 рад/решение, радиус 300.

### 4.5 Состояние между снапшотами (поля `DdnetBot`)
`prevInput`; `sent[≤16]`; `aimLog[≤8]`, `echoMiss/echoSeen[11]`, `echoSamples`, `snapshotGap`, `lastSnapTick`;
`targetId`; `lastSeenDist` (id→px); `lastInputById` (id→{name, angle, attack, keys, at, firstSeen, changed,
settleUntil}); `atUsById`, `atFriendById` (id→tick); `frozenSinceById`, `thawTickById`, `lastTouch` (id→{by,tick}),
`lastPosById`; кэши `reachAnswers` (≤64, 25 тиков), `reachWanted`, `sealAnswers` (≤64, 6 тиков); симуляторы
`planSim`, `sealSim`, `shieldSim`, `pullSim` (по одному на коллизию); `planner` (внутреннее состояние, коммит решений,
профиль соперника); `path`, `trek`, `trekAvoid(≤24)`; `nav`, `follow`, `pendingGoto`, `navPending`, `navReturnMode`,
`seekingGame`; `mode`, `acting`; WB: `wbDef`, `wbMode`, `wbChooser`, `wbCounts`, `wbWalk`, `wbWalkFails`, `wbPauses`,
`wbPausedUntilMs`; `home`/`homeMap`; stuck: `stuckAnchor*`, `frozenSince`, `lastKillTick`, `routeKillTick`;
`wasFrozen`, `wasAlive`; wander-таймеры; `idleSinceTick`, `travelSinceTick`, `dullSinceTick`; `memory`,
`memoryDirty`, `lastTileIndex`; `clipRing`, `lastPlan`, `lastClipTick`, `framesSinceScan`; дуэль-состояние; счётчики
`stats`.

---

## 5. Лаг-компенсация

### 5.1 Эхо собственного прицела (`bot.ts:4509-4563`)
Идея: решение, принятое на снапшоте тика T, становится видно в снапшоте тика T+d как угол нашего персонажа
(`character_core.angle` = угол×256). Самое раннее d, при котором наблюдаемый угол совпадает с отправленным, и есть
задержка «решение→видимость».
* `noteAim(input)` (4558-4563): при каждом отправленном вводе (applyInput, wander) в начало `aimLog` кладётся
  `{tick: world.tick, x, y}` — единичный вектор прицела; хранится `ECHO_LOG=8` последних.
* `measureEcho(self)` (4535-4556), каждый снапшот (только если живы):
  * `snapshotGap = now - lastSnapTick`, если `0 < разница ≤ 4` (обычно 2).
  * Для каждого `dec = aimLog[i]`: `d = now - dec.tick`, берутся только `1 ≤ d ≤ ECHO_MAX_TICKS (10)`; нужен
    предыдущий `before = aimLog[i+1]`, и прицел должен был **заметно измениться**: `dot(dec, before) ≤ 0.98`
    (≈>11.5°). Тогда `echoMiss[d] += 1 - dot(наблюдаемый_единичный_угол, dec)`, `echoSeen[d]++`; если `d∈{1,2}` —
    `echoSamples++`.
* `echoLagTicks()` (4509-4533): если `echoSamples < ECHO_MIN_SAMPLES (40)` → -1. Среди d=1..10 с
  `echoSeen[d] ≥ 40` выбрать `best` с минимальным средним `miss`; если `bestMiss > ECHO_MATCH_MAX (0.3)` → -1. Затем
  взять **наименьшее** `d < best`, у которого `miss ≤ bestMiss + ECHO_MATCH_GAP (0.05)`. Результат
  `max(0, best - snapshotGap)`.
* Статистика копится **без затухания и без сброса** (ни при реконнекте, ни при смене карты/redirect) — адаптация к
  изменению пинга медленная (баг/особенность).
### 5.2 Фоллбэк и итог
* `pingLagTicks()` (4501-4507): `round(PlayerInfo(own).latency_ms / 20)` (latency из снапшота; 0 если нет).
* `lagTicks()` (4494-4499): `cfg.lagCompensation===false → 0` (никогда не выставляется в start.mjs/main.ts), иначе
  `min(MAX_LAG_TICKS=6, echo ≥ 0 ? echo : ping)`.
* Использование `lag`: прокрутка планировочного мира (5.3), `guard`/shield (4811-4814: `lag` тиков с `prevInput`),
  `pullLine` (rescue, DROP), `Navigator.step(…, lagTicks())` (1984-1989), лог `noteWbWalkDeath`.

### 5.3 Входы «в полёте» и прокрутка (`bot.ts:4871-4880`, `livePlan.ts:16-40`)
* Модель: решение, принятое на снапшоте тика D, применяется сервером на тике `D + lag + 1`.
* `inFlightInputs(tick, lag)`: `at(k)` = последний элемент `sent` с `d.tick ≤ tick + k - lag - 1` (иначе
  `prevInput`); `held = at(0)` — ввод, который сервер держит на тике снапшота; `inFlight = [at(1)…at(lag)]` — ввод на
  серверных тиках `tick+1 … tick+lag`.
* `syncPlanningWorld(sim, own, self, targetId, target, prevInput, enemyInput, lag, inFlight, held)`:
  `applyTeeState(own)`, `applyTeeState(target)` (в `world.ts:353-397`: prevPos = pos − vel, attackTick из
  `sinceAttack`, `freezeStart` из `frozenFor`, deepFrozen, reload), `setHeldInput(own, held)`,
  `setHeldInput(target, enemyInput)` (`world.ts:400-407`: ввод и «предыдущий ввод для фронтов», пустой прицел →
  (0,−1)); затем `lag` раз: `setInput(own, inFlight[t] ?? prevInput)`, `setInput(target, enemyInput)`, `step()`.
  Итого **прокрутка ровно `lag` тиков (0..6)**; планировщик ищет решение уже от состояния «сейчас на сервере».
* Прочие тии (`planOthers`, дефолт 0) — только `applyTeeState` + `setHeldInput` без явного `setInput` на каждом шаге
  (по-видимому шагают с удерживаемым вводом; детально не проверено).
* `syncPlanningWorldLegacy` (`livePlan.ts:42-67`, `!try oldcopy`): старая схема — обнуляет новые поля TeeState,
  `prevPos = pos` и крутит `lag` тиков с `prevInput`.

---

## 6. LiveWorld: мир из снапшота (`liveWorld.ts`)

### 6.1 Используемые объекты снапшота
| Объект | Откуда | Что берётся |
|---|---|---|
| `OBJ_CHARACTER` (9, 22 int) | `AllObjCharacter` | core (tick, x, y, vel/256, angle, direction, jumped, hooked_player, hook_state, hook_tick, hook_x/y, hook_dx/dy/256), weapon, attack_tick, emote, player_flags (для веба) |
| ex `character@netobj.ddnet.tw` | `getObjExDDNetCharacter(id)` | `m_FreezeEnd`, `m_FreezeStart`, `m_Jumps`, `m_JumpedTotal`, `m_Flags` (m_TargetX/Y **не используются**) |
| ex `player@netobj.ddnet.tw` | сырые `deltas` по UUID (`liveWorld.ts:292-298`) | `m_Flags`: `AFK=1`, `PAUSED=2`, `SPEC=4` (76-78) |
| `OBJ_PROJECTILE` (2) | `AllProjectiles` | x, y, vel/100 (dir), type, start_tick; owner неизвестен (-1) |
| ex `ddnet-projectile@netobj.ddnet.tw` | сырые `deltas` (своё UUID-сопоставление, т.к. в библиотеке имя неверное) | pos/100, dir (нормализация при `PROJECTILEFLAG_NORMALIZE_VEL`, иначе /1e6), weapon, start_tick, owner, explosive (`91-117`) |
| `OBJ_LASER` (3) и ex `laser@netobj.ddnet.tw` | `AllObjLaser` / сырые | to, from, start_tick, owner (ex) |
| `OBJ_PLAYER_INFO` (10) | `getObjPlayerInfo` | `local` (OwnID), `team` (−1 = спектатор → `maybeJoinGame`), `latency` (ping-лаг), score (веб) |
| `OBJ_CLIENT_INFO` (11) | `AllObjClientInfo` | name (4 int), clan (3), country, skin (6), цвета — для списков, `!target`, клипов |
| `OBJ_GAME_INFO` (6) / ex gameinfo | только веб (`liveFrame`) | round_start_tick, флаги |
Не используются: `OBJ_CHARACTER_CORE` (8), `OBJ_PLAYER_INPUT` (1), `SPECTATOR_INFO`, события снапшота (13-20).
UUID ex-типа ищется в сырых `deltas` среди объектов `type_id==0` с данными = 4 int UUID (`exTypeId`, 80-86;
`uuidInts` через `calculateUuid` из `src/demo/snapshot.ts`).

### 6.2 `updateFromSnapshot` (`liveWorld.ts:213-299`)
* `tick = gameTick ?? (tick+1, или максимум core.tick)`.
* Все известные тии помечаются `alive=false`, затем для каждого `OBJ_CHARACTER` — `decodeCharacter`. **Записи не
  удаляются** (мёртвые/ушедшие остаются с `alive=false`); вызывающий код фильтрует по `alive`.
* Снаряды и лазеры пересобираются каждый кадр (id = индекс).
* `flagsById` (DDNetPlayer) пересобирается каждый кадр.
* Методы: `playerFlags`, `serverAfk` (AFK), `spectating` (SPEC), `notPlaying` (PAUSED|SPEC).

### 6.3 `decodeCharacter` (`liveWorld.ts:131-193`)
* **Эволюция (reckoning)**: если `core.tick != 0 && core.tick < tick && tick-core.tick ≤ 150` — ядро прокручивается
  `CharacterCore.tick(false); move(); quantize()` в пустом мире (без других игроков) до `tick`
  (`reckoning.ts:45-63`) — как `Evolve` в клиенте DDNet (сервер шлёт «reckoning core» со своим тиком).
* Заморозка: `m_FreezeEnd == -1` → deep freeze, `freezeTicksLeft = 150` (`DEEP_FREEZE_TICKS`), `deepFrozen=true`;
  `>0` → `max(1, freezeEnd - tick)`; `0` → не заморожен. `frozenFor` = `tick - m_FreezeStart` (если >0), иначе
  `150 - freezeTicksLeft`.
* Оружие: сервер показывает замороженным NINJA → подменяется на предыдущее оружие или GUN (150-151).
* Прыжки: `jumps = m_Jumps ?? 2`; `jumpsLeftOf` (121-129): на земле (пиксели (x±14, y+14+5) твёрдые) → `jumps`;
  иначе если `jumped & 2` → 0; иначе `max(0, jumps - 1 - jumpedTotal)`.
* `reloadTicks = max(0, attack_tick + trunc(125*50/1000)=6 - tick)`; `sinceAttack = tick - attack_tick`;
  `hookTick`, `ddnetFlags`.
* Хук: `hookState`, `hookPos`, `hookDir` (/256), `hookedPlayer` — прямо из core.

### 6.4 Угадывание ввода чужих (`livePlan.ts:5-14`)
`enemyInputFromSnapshot(t)`: `direction = core.direction` (сервер шлёт реальное направление ввода), `hook =
hookState > 0 ? 1 : 0` (держит хук, пока он летит/зацеплен), прицел `= (cos, sin)(wireAngleRad(angle)) * 300`
(округл.), `jump=0, fire=0`. Этот ввод считается постоянным в прокрутке и в модели соперника планировщика (дефолт
`opponentModel: "hold"`; есть `react/policy/learned` и `opponentDirNet` из `opponent.json` — область планировщика, не
проверено). `wireAngleRad(a) = a/256`, с переносом в (−π, π] (`types.ts`).

### 6.5 Тюнинг и карта
* Тюнинг: константы DDNet по умолчанию (`src/core/tuning.ts`: `hookLength 380, gravity 0.5, groundJumpImpulse 13.2,
  airJumpImpulse 12, hookFireSpeed 80, hookDragAccel 3, hookDragSpeed 15, hammerFireDelay 125` и т.д.). Серверные
  `SV_TUNE_PARAMS` и tune-зоны карты **не применяются** (в `loadMap.ts` tune-слой лишь отмечается флагом
  `layers.tune`). Риск на серверах с изменённым тюнингом.
* Карта: см. 2.4; `loadMapCollision` читает game-слой, tele, speedup, front (+флаги switch/tune) (`loadMap.ts:108-211`).

---

## 7. Выбор цели

### 7.1 `pickTarget(snap, ownId, selfPos)` (`bot.ts:3020-3123`)
**Режим фиксированной цели** (`cfg.targetName`, `!target`/`--target`): точное совпадение имени после
`foldName` (trim, сжатие пробелов, lower) среди ClientInfo; не найден/это мы/тии нет/мёртв → −1; иначе id. **Никаких
других фильтров** (друг, AFK, заморожен — всё равно).

**Автовыбор** (иначе):
0. `refreshInputClock()` (повторно; второй вызов в кадре почти ничего не меняет). `wb = wbHolding()`, `wbSide`;
   `meInLeash = wb && side && inWbHall(wb, side, свой тайл)`; `ropeOnUs` = нас держит хуком не-друг.
1. Для каждого живого `tee ≠ me` — **фильтры (continue)**:
   * в списке `friend` или `ignore` по имени (`onList`), или клан в `clanFriend` (исключение: соперник по текущей
     дуэли никогда не «друг/игнор», 3655-3667);
   * `outOfGame`: в дуэли — только флаг SPEC; иначе PAUSED|SPEC;
   * не `atWar` (war по имени или clanWar) и не дуэль и `afk(tee)` (раздел 7.4);
   * `d > TARGET_MAX_PX (1600)`;
   * при удержании ВБ: `roped` (он держит нас или мы его), `atUs` (атаковал/хукал нас за последние 150 тиков),
     `counter = meInLeash && atUs && d ≤ 380+64 && tee вне «поводка»`; пропуск если `!roped && !atWar && !counter &&
     (meInLeash ? tee вне поводка : !atUs)`. `inWb = inWbZone(tee)`;
   * `trapCare()` (`deadZoneCost > 0`; дефолт 0 → выключено): tee в мёртвой зоне, а мы нет;
   * **sealed/settled** (см. 7.2).
2. `outOfReach = d ≥ 420 (PATH_NEAR_PX) && !atWar && нет верёвки между нами && !reachable(self, tee)`.
3. **Очки** (3091-3113):
   | Терм | Вес |
   |---|---|
   | `atWar` | +900 |
   | tee держит нас хуком | +1000 |
   | мы держим tee хуком | +800 |
   | текущая цель, заморожена, `d < 320` (`BLOCKING_RANGE_PX`) | +`blockHoldScore` (дефолт **0**) |
   | `finishing && d < 320` | +600 (`FINISH_BLOCK_SCORE`) |
   | `wbFinish && WB_URGENCY>0` (env, дефолт 0) | +`WB_URGENCY*(1 - min(1, freezeTicksLeft/150))` |
   | «агрессор»: `tick - tee.attackTick < 150` и `d < 500` | +500 |
   | атаковал друга за последние 150 тиков (`atFriendById`) | +450 (`AT_FRIEND_SCORE`) |
   | [DROP] держит хуком нашего партнёра (не дуэль, нас никто не держит) | +700 (`TEAM_HELP` env) |
   | приближается: `d < lastSeenDist - 1` | +200 |
   | удержание текущей цели | `+ (d ≤ 420 ? H : H*max(0, 1-(d-420)/400))`, `H = targetHold` (**400**) |
   | расстояние | `−0.25·d` |
   | `outOfReach` | −700 |
   | tee в зоне ВБ (`inWb`) | +300 |
   `lastSeenDist[tee] = d` обновляется только для прошедших фильтры. Лучший — строго больший счёт (при равенстве —
   первый в порядке вставки в `Map` мира).
4. Если никого не выбрали, но текущая цель была отброшена как settled → вернуть текущую (`keepSettled`, 3121).

### 7.2 «Sealed»/«settled» (`bot.ts:3076-3088`, `2899-2914`, `seal.ts:84-107`)
* `nearFreeze(pos)`: фриз-тайл в квадрате ±2 тайла (`SEAL_NEAR_TILES=2`, 2791-2799).
* `sealed = (tee.frozen || (tee — текущая цель && nearFreeze)) && isSealed(tee)`.
  `isSealed`: кэш 6 тиков (`SEAL_ANSWER_TICKS`), отдельный `sealSim` только с этим тии; `sealedIn(sim, id, state,
  held=enemyInputFromSnapshot)`: если заморожен и `freezeTicksLeft ≥ 90` — пробуется только `held`, иначе `held` +
  3 «побега» (jump+hook вверх, направление −1/0/+1, прицел (ax·200, −300)); каждый прогон 90 тиков (прыжок нажимается
  через тик); если в конце жив и (не заморожен или не касается фриза/смерти) → **не** sealed. Все провалились → sealed.
* `wbFinish = WB_FINISH (env WB_FINISH != "0", по умолчанию true) && держим ВБ && meInLeash && tee.frozen && !sealed && inWb`.
* `finishing = wbFinish || (tee — текущая цель && frozen && !sealed && frozenFor ≤ 150 && nearFreeze)`.
* `settled = sealed || (!finishing && frozenFor > settledFreezeTicks)`; дефолт `settledFreezeTicks = 0`
  (`planner.ts:235`) → **любой замороженный, кроме «добиваемой» текущей цели (или ВБ-случая), пропускается**. Смысл:
  замороженного бить бессмысленно/вредно (хаммер размораживает), лучше переключиться на свободного, но если никого
  нет — держать прежнюю цель.

### 7.3 Списки friend/war/ignore/clanWar/clanFriend
* Хранение: `runs/relations.json` = `{v, war[], friend[], clanWar[], clanFriend[], ignore[]}` (отображаемые имена),
  в памяти `Map<lowercase trim, display>` (`bot.ts:788-795, 3355-3365, 3557-3580`). `v` — версия (миграция тиммейтов,
  DROP).
* Сопоставление `listed(list, nameKey)` (276-286): точное совпадение ключа, **или `nameKey.includes(key)`
  (подстрока!)**, или равенство после снятия префикса дубликата `^\(\d+\)` (DDNet-префикс «(1)nick»). Для клана то же.
* `!friend/!war/!ignore <ник>` (3584-3626): ник дополняется по игрокам на сервере (подстрока; >1 совпадение → просьба
  уточнить); повторная команда с тем же ником **удаляет** его; `off` очищает список; взаимоисключения: war удаляет из
  friend/ignore; friend удаляет из war; ignore удаляет из war; clanWar↔clanFriend.
* Семантика в бою: friend/ignore/clanFriend — никогда не цель (`pickTarget`), «spare» для вето хука и планировщика,
  не считаются «хукающими нас» в `ropeOnUs`, не атрибутируются блоки; friend (оригинал) ещё «спасать из фриза» —
  **DROP** по заданию; ignore дополнительно — не отвечать в чате (DROP). war — +900 и игнор AFK/сравнения ВБ.
* Для порта: friend = «не трогать»; war/ignore — по желанию, в простом виде. Подстрочное совпадение стоит заменить
  на точное (или явный флаг).

### 7.4 Активность/AFK и атрибуция (`refreshInputClock`, `bot.ts:2916-2991`; 2993-3018)
* `inputKeysOf(t) = (direction+1) | (jumped&1)<<2 | (hookState≠IDLE)<<3` (321-323); у замороженных keys = −1.
* На каждый тик для живого tee: при первом появлении/смене имени — запись (`changed=-1`, `firstSeen=tick`);
  `changed` = изменился угол, `attackTick` или keys (оба ≥0); фиксируется, если `tick ≥ settleUntil` (после смерти
  `settleUntil = tick+50`, `INPUT_SETTLE_TICKS`, чтобы респаун не считался активностью, 2126-2127).
* `inputIdle(t, strict)`: нет записи → `strict`; `changed<0` → `strict || tick-firstSeen > 500`; иначе
  `tick-changed > 500` (`AFK_TICKS=10 с`). `afk = serverAfk || notPlaying || inputIdle`.
* `atUsById[tee] = tick`, если `attackTick` изменился и tee ≤128 px от нас (`SWING_AT_US_PX`) или tee держит нас хуком;
  `atFriendById` аналогично для незамороженных друзей.
* Атрибуция блоков (статистика): `lastTouch[victim] = {by, tick}` при хуке (сохраняет прежнего `by`, если он всё ещё
  среди хукающих, иначе min id) и при ударе хаммером (точка `pos + dir(angle)*21`, жертвы в 56 px); сброс при
  телепорте >200 px за ≤4 тика и смерти. `onFreezeOnset` (3773-3788; не считается повторная заморозка в течение 6
  тиков после разморозки) → `blocks`/`blockedBy` если касание ≤50 тиков назад.
* `spared(t)` (3009-3018) для планировщика: ignore; незамороженный friend/clanFriend; вне игры; (не war, не дуэль, AFK).

### 7.5 Достижимость (`reachable`, 2828-2897)
`findRoute(collision, from, tee.pos, {nearTiles:3, partial:false, maxNodes:20000, throughFreeze:false}) !== null`;
кэш по (id, тайл-откуда, тайл-куда) 25 тиков (`REACH_ANSWER_TICKS`), ≤64 записей (вытеснение старейших). В low-cpu
не более 2 проверок на снапшот (`LOW_CPU.reachChecks`), остальные — из кэша или оптимистично `true`, с очередью
по давности.

---

## 8. Режимы и поведенческие ветви

### 8.1 Режимы (`mode`)
`fight` (по умолчанию: выбирать цель и драться), `passive` (не выбирать цель: брожение/ВБ/дом), `hold` (`acting=false`:
`idle()`), `goto` (идёт навигация; по завершении возврат в `navReturnMode`). `!stop`: если идёт nav — отменить его,
иначе `hold`; `!go` → `fight` (1365-1388).
### 8.2 Навигация / goto / follow (`bot.ts:1696-2023`)
`!goto tele|<x> <y>|<ник>|@ник|stop` → `Navigator(collision, goals, {throughFreeze?, crossings?})`; на картах с ВБ
по умолчанию передаются `wbDef.crossings` (перелёты через фриз-трубы). `driveNav` (1966-2023): `nav.step(self, tick,
others, lagTicks())` → если не `crossing`/`plannedFreeze` — через `guard` (при вето — `nav.vetoed()`); `nav.takeKill()` →
`/kill` (кулдаун 500 тиков); `applyInput`. Follow игрока: цель пересчитывается при движении ≥96 px и ≥25 тиков, при
«arrived»/телепорте; отказ после 120 с, 30 с без прогресса (32 px), 3 смертей его/своих, 3 «blocked», 5 с без тии;
прибытие ≤64 px (константы 264-273). Цель-тайл — ближайший свободный в ±3 тайла, предпочтительно с полом (1803-1825).
### 8.3 Trek / seek (4251-4329, 4448-4480)
`busiestSpot`: для каждого «бодрствующего» (не AFK, не «запаркован» во фризе >250 тиков или deep) — соседи в 600 px,
`busy` = хук летит/держит или атаковал ≤100 тиков; счёт `near + busy - dist/1500`; `null` если лучший ближе 800 px.
`gameSpot` исключает зоны `avoid` ВБ. `startTrek`: `findRoute(partial, allowKill, !throughFreeze, avoid=trekAvoid)`;
если маршрут обрывается >3 тайлов от цели рядом с фризом — отказ. `trekGoal` ведёт планировщик по шагам (шаг `kill`
→ `/kill`; достигнут ≤56 px; застой 150 тиков без улучшения на 8 px → запрет этого хода (≤24) и конец).
### 8.4 Wayblock (ВБ; `wayblock.ts`, `bot.ts:4024-4211`)
Только карты `Copy Love Box` и `Copy Love Box JoniTee` (сдвиг 182,212; 600×600), жёстко заданные зоны/споты/якоря.
`wbHolding()` = есть def, `wbMode≠off`, нет `home`, не дуэль, нет паузы; режим fight (или goto с возвратом в fight).
Сторона: `auto` — меньше игроков (не AFK/не запаркованы), при равенстве — где стоим/ближе к `watch`; без
«перепрыгивания» (`WB_SIDE_HOPPING=false`); если стоим в зоне другой стороны — принять её. 4 смерти подряд на пути →
пауза 5·2^(n−1) мин (макс 30) (`noteWbWalkDeath`, 4090-4100). В зале — `WB_PLAN_OVERRIDES` (`{noThawRope:true,
frozenThrow:3, airJumpCost:0.3, launchExactReach:100}` или из env `WB_PLAN`), `wbBand` — полоса допустимых позиций.
### 8.5 Home
`!home [x y|off]`: при отсутствии цели >200 тиков (`GO_HOME_AFTER_TICKS`) и удалении >2 тайлов — goto домой; пока
`home` задан, ВБ не держится; забывается при смене карты.
### 8.6 Unstick / самоубийство (`maybeUnstick`, 4574-4693)
Константы: `STUCK_FROZEN 450`, `STUCK_WEDGED 200`, `STUCK_RADIUS 48 px`, `FROZEN_HARD_LIMIT 400`, `FROZEN_IN_TILE 200`,
`TRAPPED 75`, `HELPED_LIMIT 1500`, `WB_LYING 25`, `WB_KILL_COOLDOWN 100`, `KILL_COOLDOWN 500` тиков.
* `overdue = (frozenFor ≥ 400 || (углы тела во фризе && нас не держат && ≥200) || (в мёртвой зоне && не держат &&
  ≥75)) && (!helper рядом (друг/игнор ≤140 px) || ≥1500)` → `/kill` (кулдаун 500).
* На ВБ: лежим во фризе вне зон ВБ, скорость <0.5, не держат, нет помощника, ≥25 тиков → `/kill` (кулдаун 100).
* Якорь: если за 450 (заморожен) / 200 (свободен, **только при наличии цели**) тиков сдвинулись <48 px → `/kill`, кроме:
  нас держат хуком; помощник рядом (<1500); центр не в фризе; (свободен) цель заморожена и дуэль/держим её/в
  досягаемости хука.
### 8.7 Wander и guard
`wander` (4882-4971): направление ±1/стоп (22%), период 25..199 тиков, разворот у стены/фриза/смерти/провала
(60% при обрыве), прыжок с вероятностью 0.03 (3..10 тиков), хук 0.02 (15..49 тиков) — вне ВБ; всё через `guard`;
хук снимается, если держим кого-то или верёвка зацепит тии. `guard(self,input)` (2801-2820): одиночный `shieldSim`,
прокрутка `lag` тиков с `prevInput`, затем `escapeExists(sim, id, input, 2)` (2 тика ввода + 36 тиков одного из
«побегов» + до 90 тиков доводки без фриза) → иначе `saferInput` (поворот прицела ≤1.5 рад).
### 8.8 Дуэль (4030-4088) — PARTIAL
`duelNow = duelMode=="on" || (auto && duelSeen)`. Автодетект требует принятого приглашения (`/accept` в чате за
последние 120 с) → без чата работает только `!duel on`. В дуэли: нет ВБ/походов/seek, AFK-фильтр отключён, счёт в
`runs/duels.json`.
### 8.9 [DROP] Спасение друга (`rescueFriend`, `pullLine`, 3794-4022), партнёр/тиммейт, эмоции на события, чат.

---

## 9. Команды, CLI, настройки

### 9.1 Консольные команды (`handleConsole`, `bot.ts:1310-1666`), префикс `!` или `?`
Строка **без префикса отправляется в игровой чат** (1314-1322) — **DROP** (бот не должен писать в чат).
| Команда | Семантика | Порт |
|---|---|---|
| `help` | справка | PORT |
| `mode [fight\|passive\|hold]` | режим; не-fight сбрасывает цель; прерывает nav | PORT |
| `stop` / `go` | отменить nav или hold / fight | PORT |
| `goto [tele\|x y\|nick\|@nick\|stop\|-\|off]` | навигация; без аргумента — прогресс/подсказка; до спавна/карты — в очередь | PORT |
| `target <nick>\|-` | фиксированная цель (дополнение по подстроке) / снять | PORT |
| `war/friend/ignore [name\|off]`, `clanwar/clanfriend [clan\|off]` | списки (7.3) | PORT (упростить) |
| `home [x y\|off]` | точка возврата | PORT |
| `wb [off\|left\|right\|auto\|on]` | ВБ (`on`=auto) | PORT |
| `duel [on\|off\|auto]` | дуэль-режим | PARTIAL (без auto через чат) |
| `style default\|wb\|duel` | пресеты: default = duel auto + wb off; wb = duel auto + wb; duel = `!duel on` | PARTIAL |
| `clip [note]` | сохранить буфер 30 с → `manual-<tick>[-note].json` | PORT |
| `stats`, `where` | счётчики; позиция/состояние/цель/ВБ/тик/прогресс goto | PORT |
| `brain planner\|net\|scripted` | смена мозга на лету | PARTIAL (net→fly) |
| `try <name>\|off` | экспериментальные наборы планировщика `TRY_SETTINGS` (17-41) | PARTIAL/отладка |
| `low on\|off`, `strong on\|off` | CPU-режимы, сохраняются в settings.json | PORT |
| `lang ru\|en` | язык, сохраняется | PARTIAL |
| `log on\|off` | в боте только событие; фильтр делает UI | PARTIAL |
| `yes/f3`, `no/f4`, `votes`, `vote <n\|text>` | голосования | опц. |
| `spec` / `join` | SetTeam(−1/0) | PORT |
| `kill` / `reset` | `/kill` (кулдаун 500 тиков) | PORT |
| `emote <name>` | эмоция | опц. |
| `quit` | выход | PORT |
| `say <text>` | в чат | **DROP** |
| `owner`, `llm` | приказы хозяина / LLM | **DROP** |
| `d <cmd>` (в start.mjs) | команда дамми | **DROP** |
| `seek` | есть в `commandNames` (3280-3282), обработчика нет → «unknown command» | — |
Чатовые команды при `--chat` (`!bot, !stop, !go, !stats, !try, !hi/!hello/!привет, !reset`, 2193-2231) — **DROP**.
Веб-UI дополнительно: `setKnob/resetKnobs` (правка любого ключа `PLANNER_DEFAULTS`, 3285-3333), relations, autochat,
llm, launch-настройки — DROP вместе с веб-UI (возможно, аналог «knobs» полезен для отладки).

### 9.2 CLI
* `main.ts` (11-38, 60-88): `--server ip:port | --host --port`, `--name` (AI-Tee), `--clan`, `--no-emotes`,
  `--planner | --policy <file> | --scripted` (ровно один), `--console`, `--no-console`, `--brush-off` (DROP),
  `--protocol-version <n>`, `--password`, `--skin`, `--country`, `--target`, `--goto`, `--map-dir`, `--chat` (DROP),
  `--no-reconnect`, `--verbose`, `--low-cpu`, `--strong`, `--duration <sec>`.
* `start.mjs` (свободный разбор `--key [value]`, 23-30): `--setup`, `--lang`, `--server`, `--name`, `--clan`, `--skin`,
  `--password`, `--policy`, `--scripted`, `--planner`, `--bold`, `--low-cpu [off]`, `--strong [off]`,
  `--protocol-version`, `--goto`, `--dummy [name|off]` (DROP), `--no-web`, `--web-port` (7777), `--ready-line`,
  `--no-open`, `--no-update`, `--console`, `--no-console`, `--ink`, `--plain`.
* Env: `DDNET_AI_LANG`, `WB_FINISH` (≠"0"), `WB_URGENCY` (0), `WB_PLAN` (JSON), `TEAM_HELP` (700, DROP),
  `GITHUB_TOKEN`/`DDNET_AI_TOKEN`/`DDNET_AI_UPDATE_API` (DROP).

### 9.3 `settings.json` (корень)
`server` ("auto"/"авто"/пусто = авто, иначе ip:port), `name`, `clan`, `skin` (дефолт "cammostripes"), `password`,
`brain` ("planner"|"bold"|"scripted"), `lang`, `lowCpu` ("on"/"off"), `strong` ("on"/"off"); DROP: `dummy`,
`dummyName`, `owner`, `llm {url,model,key}`, `llmOff`, `ddnetData`, `skinDownload`. Бот пишет в файл только если в нём
есть строковый `server` (3372, 3385, 3533). Повреждённый файл переименовывается в `.bad-<ts>` (`start.mjs:242-246`).

---

## 10. Клипы, инциденты, память, прочие файлы

### 10.1 Буфер и клипы
* `RingRecorder(30*25 = 750)` кадров (`bot.ts:838`); кадр пишется **на каждый обработанный снапшот, пока свой тии
  жив** и карта задана (`recordFrame`, 3134-3162): `tick`; `tees` = свой + 3 ближайших живых (`snapTee`: id, x, y (окр.),
  vx, vy (3 знака), angle, hookState, hookX/Y, hookedPlayer, frozen, alive, weapon, direction, jumped, jumpsLeft,
  freezeTicksLeft, hookDx/Dy, hookTick, frozenFor); `inputs` = [`prevInput` своего] (решение предыдущего снапшота);
  `events: []` (**всегда пусто вживую**); `plan` = `lastPlan`; `walk` = метка цели nav; `plannedFreeze`.
* Файл `Recording` (JSON, `recording.ts:60-76`): `{map, controller, seed:0, width, height, tiles: [весь массив
  тайлов коллизии], tele?: [[index,type,number]], selfId, label?, players?: [{id,name,clan,skin,cc,cb,cf}], frames}`.
  Путь `runs/clips/<name>.json` (`DEFAULT_CLIP_DIR`, 577); имя: `manual-<tick>[-<note>]`, `<kind>-<tick>-s<severity>`,
  `cross-fail-<tick>`; символы вне `[a-zA-Z0-9_-]` → `_` (3192).
* Автоклип (`maybeClip`, 3173-3186): каждые 50 кадров, кулдаун 45 с игры (2250 тиков), ≥50 кадров;
  `mergeOverlapping(findIncidents(rec))`, порог severity 250 (180 для `self-freeze`, `chased-into-freeze`,
  `goto-into-freeze`), инцидент только во второй половине буфера, берётся максимальный.
* `clipCrossFail` (3164-3171): при заметках навигатора «…trying again from the spawn»/«no way through» — клип, кулдаун
  60 с игры.
* Чистка (`pruneClips`, 3213-3239): авто-клипы по шаблону `^(.*)-\d+-s(\d+)\.json$`, новые первыми: максимум 24 всего
  и 16 на вид; лишние удаляются (+ `.html`). `manual-*` не трогаются; `cross-fail-*` не матчится шаблоном → **не
  чистится никогда**.
### 10.2 Инциденты (`incidents.ts`)
Виды и severity: `self-freeze/chased-into-freeze/goto-into-freeze` — момент нашей заморозки (исключая телепорт,
удар хаммером, неподвижность, запланированный фриз), severity = длительность фриза + 40 (если соперник свободен) − 30
(если нас держал хук); «chased», если за 25 кадров сблизились >40 px с тем же свободным соперником; «goto», если шла
навигация. `slow-rehook` (свободная верёвка при соперике в досягаемости > 30 тиков; sev = тики − 10),
`short-hold` (отпустил тии < 8.5 тиков), `thawed-the-enemy` (60), `swing-at-air` (5, по событиям — вживую не
срабатывает), `wall-grind` (≥20 тиков направление без движения; sev = тики), `jitter` (6 смен направления ≤25 тиков),
`death` (150; вживую невозможен: кадры пишутся только пока живы). `mergeOverlapping` склеивает инциденты ближе 25 тиков.
### 10.3 FreezeMemory (`memory.ts`)
* Файл `runs/memory/<map>.json`, `<map>` = имя карты с `[^\w.-]+ → _` (`bot.ts:4413-4416`; ключ **без CRC**).
* Формат: `{width, height, events, idx[], val[], pidx[], pval[]}` — разреженные `cells` (фризы) и `passes` (проходы).
* `note(x,y)` при заморозке себя: +1 в тайл, +0.4 (`SPREAD`) в 8 соседей, `events++`. `notePass` при входе в новый
  тайл свободным: +1.
* `save`: сначала **умножение всего на `DECAY=0.97`** (т.е. затухание на каждое сохранение, частота зависит от
  активности), обнуление `<0.01` (cells) и `<0.05` (passes), запись. Сохранение: каждые 20 заморозок, при смене карты,
  при остановке.
* `load`: несовпадение размеров → пустая. `risk = v/(1+v)`; `safety = clean·good/(good+10)`,
  `clean = good/(good+15·bad)`. Использование в планировщике (`planner.ts:558-563`): штраф близости к опасности
  `selfHazardCost·(meNear − selfHazardThreshold)` умножается на `(1 − memoryTrust·safety(me.pos))`, где
  `memoryTrust = 0.9` из `LIVE_PLANNER_CFG` (`bot.ts:590`); т.е. часто проходимые без фриза тайлы штрафуются меньше.
  `risk()` и `memoryWeight` (дефолт 0) в live — не проверено, вероятно не используются.
### 10.4 Прочие файлы в `runs/`
`relations.json` (7.3), `duels.json` (`{at, seconds, opponent, ours, theirs}`, до 2000, атомарная запись через `.tmp`),
`autochat.json` (DROP), `server-avoid.json` (`{addr: untilMs}`), `dummy/` (DROP).

---

## 11. Выбор сервера (`serverPick.ts`, `start.mjs:151-173`, `bot.ts:1104-1162`)
* Master: `https://master{1..4}.ddnet.org/ddnet/15/servers.json`, по очереди, таймаут 8 с, `user-agent: ddnet-ai`
  (`serverPick.ts:4-9, 99-113`). JSON `{servers:[{addresses[], location, info{name, map{name}|string, game_type,
  passworded, max_clients, clients[{is_player,…}]}}]}`.
* Адрес: из `addresses` вида `(tw-0.6+udp|tw-0.7+udp|ddnet+udp|udp)://ip:port`; предпочитается `tw-0.6+udp`, иначе
  **первый валидный любого вида** (может оказаться 0.7-only — баг) (28-44). Дедуп по адресу.
* «Блок-сервер»: `/block|blmap|copy (love|the) box|love box/i` в имени, карте или game_type (76-80).
* `pickBlockServer`: пропустить не-блок, с паролем, из avoid, полные (`clients ≥ max_clients-1`), `players < 2`;
  счёт `players + (lang=="ru" && location ~ /:ru$/i ? 6 : 0)`; максимум (82-97).
* На старте (`start.mjs`): при `server = auto` — цикл выбора, при неудаче повтор через 30 с.
* В игре (`checkServer`, каждые 30 с, только `autoServer`): offline >120 с → avoid 30 мин, смена; онлайн и «других»
  (`ClientInfo.length-1`, включая спектаторов) <2 в течение 180 с → выбрать новый (avoid = текущий + файл), сменить,
  только если там `players ≥ others+3`, текущий в avoid на 10 мин; бан при дисконнекте → avoid 60 мин + смена. Смена =
  выход с кодом 75 и перезапуск процесса.

---

## 12. CPU-бюджет (`cpuLoad.ts`, `bot.ts:2310-2350, 4809-4815`)
* Базово: `budgetMs = LIVE_BUDGET_MS = 18` на решение, `explain: true` (43, 2311); дефолт планировщика population 20,
  iterations 2 (`planner.ts:242-244`). `PLANNER_BOLD` (brain "bold"): population 64, iterations 3, 18 мс (45).
* `--low-cpu`/`!low on` (`LOW_CPU`, `cpuLoad.ts:17`): `budgetMs = min(база, 6)`, `hardMs = min(база, 11)` или 11,
  `commitDecisions = max(база, 2)` (новый план раз в 2 снапшота), `explain:false`, `shieldCadence:true`; навигация
  `crossBudgetMs = 10`; ≤2 проверки достижимости за снапшот; в rescue поиск раз в 3 снапшота. Включение low выключает
  strong.
* `--strong`/`!strong on` (`STRONG_WB`, `cpuLoad.ts:19`): **только в зале ВБ** — `WB_PLAN_STRONG = WB_PLAN_OVERRIDES +
  {population 40, iterations 3, budgetMs 30, hardMs 36}`, если у мозга population < 40. Несовместим с low.
* Смена low/strong/try пересоздаёт `planner` (2329, 1491, 1499).
* `LagWatch` (54-126): окно 1 с; `workMs` — среднее время `onSnapshot`; «пропуск» засчитывается, если пришло >1
  снапшота, предыдущая обработка ≥40 мс и новый пришёл в пределах 16 мс после её конца; «отстаём», если ≥2 пропусков
  за окно после 60 с прогрева; подсказка «!low on» после >5 с подряд, снимается после 30 с нормы; окно сбрасывается,
  если между обработками >5 с. Только подсказка, автопереключения нет.

---

## 13. Баги, мёртвый код, странности, хардкод

1. `idle()` не вызывает `sendInput()` (4973-4986) — ввод уйдёт с таймером библиотеки (до 50 мс позже).
2. `!brain planner`: проверка `this.planner === undefined` никогда не истинна (поле `null`) (1466).
3. `seek` в `commandNames` без обработчика (3281).
4. `!log` в боте ничего не переключает (1566-1572).
5. `listed()` — подстрочное совпадение имён/кланов (283): короткий ник в friend/war затрагивает многих.
6. Эхо-статистика без затухания/сброса (5.1); `MAX_LAG_TICKS=6` жёстко.
7. `FreezeMemory`: затухание на каждое сохранение (частотно-зависимое); память по имени карты без CRC.
8. Клипы: `events` всегда `[]` → событийные инциденты (`swing-at-air`, событийный `death`) вживую не работают;
   `death` по alive невозможен; `cross-fail-*` не чистятся; `tiles` пишется целиком в каждый клип (сотни КБ).
9. `refreshInputClock()` вызывается дважды за кадр (2375, 3034).
10. `pickTarget` с `targetName` игнорирует friend/AFK/frozen.
11. `checkServer` считает спектаторов «игроками» (1134).
12. `pickAddress` может вернуть 0.7-адрес для 0.6-клиента.
13. Библиотека: повтор `disconnect` каждые 5 с после таймаута; keepalive/таймерный resend фактически мёртвы;
    `Flush()` ставит `ack = lastCheckedChunkAck` (может подтвердить непоследовательный seq, `client.js:353`);
    карта по UDP по одному чанку за RTT; CRC-ошибка карты — `throw` в обработчике (крэш через guard);
    `SV_VOTE_OPTION_REMOVE` портит список (`VoteList = VoteList.splice(index,1)`, `client.js:798`); kill-инфо только для
    id<64 и teamsstate на 64 при анонсе 19000 (128 игроков?) — не проверено, как сервер маппит id; неверное имя UUID
    `projectile@netobj.ddnet.tw`; `PredGameTick` по настенным часам без INPUTTIMING; `SnapshotWrapper` — линейные поиски.
14. Бот раскрывает себя серверу: `NETMSG_EX i-am-npm-package@swarfey.gitlab.io` и строка версии с URL npm
    (`client.js:448, 454-457`); также `crashmeplx`.
15. `/showall 1` шлётся через `CL_SAY` при каждом `connected` (2071) — это серверная команда, а не видимое сообщение;
    после смены карты в сессии не повторяется.
16. `FlagScoreboard(tick%50<2)` — непрокомментированный трюк (4843).
17. Эмоции на смерть/убийство/заморозку (2141-2147, 2426) и «хаммер-подтверждение» приказов — видимы игрокам (не чат).
18. Хардкод: ВБ-геометрия только для Copy Love Box (`wayblock.ts:139`); `DEFAULT_MAP_DIR` vendor/DDNet-20.0;
    `DEFAULT_PROTOCOL_VERSION 19000`; `CLIENT_VERSION.release_version "19.0"`; skin `cammostripes`; nick `AI-Tee`; веб
    порт 7777; мастер-URL; все веса `pickTarget`.
19. `OpponentProfile` фактически не используется (вес 0).
20. `LiveWorld` не удаляет записи ушедших игроков (растёт до числа когда-либо виденных id; id переиспользуются).
21. При ошибке базы снапшота обёртка на один кадр видит пустой мир → `OwnID` undefined → `idle()`; при `SNAPEMPTY`
    объекты — от предыдущего обработанного снапшота, а не от базы.
22. `onDisconnect` чистит `sent`, но не `aimLog` (безвредно из-за проверки d).
23. `syncOthers` не шагает «других» явным вводом (зависит от `setHeldInput`).
24. `mapCollisionFromClient` создаёт временный каталог на каждую попытку (раз в 2 с до успеха).

---

## 14. Риски и рекомендации для порта

* **Протокол**: реализовать 0.6+DDNet самим (или взять за основу libtw2-подобную схему): токен, vital/ack/resend,
  huffman (с лимитами!), дельта-снапшоты с CRC, ex-UUID; **не** слать `i-am-npm-package`; решить, какую версию
  анонсировать (19000 vs 603) — от неё зависит поведение сервера (DDNet-объекты, 128 игроков); добавить
  `NETMSG_INPUTTIMING` для правильного intended tick (можно уменьшить эффективный лаг); разобрать `SV_TUNE_PARAMS`
  (и по возможности tune-зоны) — иначе физика расходится на нестандартных серверах; обрабатывать REDIRECT и RECONNECT.
* **Тайминг**: сохранить семантику «решение раз в снапшот» + повтор ввода 20–50 Гц; ввод сразу после решения.
  При переходе на честный intended tick пересчитать модель `inFlightInputs` (`D + lag + 1`).
* **Лаг**: эхо-метод работает только при заметных сменах прицела (>11.5°) и ≥40 выборках; добавить затухание/сброс
  при реконнекте. Для «мухи» (нейросеть-решатель) важно сохранить прокрутку мира на `lag` тиков с летящими вводами.
* **Чат**: единственная необходимая «чат»-отправка — `/showall 1` (серверная команда через `CL_SAY`); всё прочее
  (строки без префикса в консоли, `!say`, автоответы, `/accept`) выкинуть. Уточнить у пользователя, допустим ли
  `/showall` (иначе дальние игроки не видны, а `TARGET_MAX_PX=1600`). Эмоции — отдельный вопрос (не чат, но видимы).
* **Friend** = только «не трогать» (фильтр в `pickTarget` + spare для вето хука/планировщика); rescue удалить.
  Автодуэль без чата не срабатывает — оставить ручной `duel on` или убрать.
* **Точность LiveWorld**: сохранить эволюцию reckoning-ядра (≤150 тиков), deep-freeze (−1), подмену ninja,
  `jumpsLeftOf`; можно использовать `m_TargetX/Y` из DDNetCharacter для лучшего прицела соперника.
* **Карты**: HTTP-загрузка по `map_details` + параллельное окно UDP-чанков; кэш по (имя, CRC/sha256); память фризов
  лучше ключевать по CRC.
* **CPU**: в Rust бюджеты 18/6/30 мс, вероятно, можно сохранить как «время на решение», но population/iterations
  станут больше; `LagWatch` можно заменить прямым измерением.
