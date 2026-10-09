# DDNet: протокол, карты, демо, мастер‑сервер — факты для Rust‑переписывания DDNet-AI

Фаза 0, исследование. Дата: 2026‑09‑27. Автор: research‑subagent (+3 параллельных fork‑исследования: онлайн/крейты, аудит libtw2 с живыми тестами, twmap + npm `teeworlds`).
Исходники: DDNet master `~/aiddnet/ref/ddnet` (`9576fd6`, 2026‑09‑27, версия 20.2/20020), libtw2 `~/aiddnet/ref/libtw2` (`060e4b6`, 2026‑09‑02), twmap `~/aiddnet/ref/twmap` (`7e5e620`, 2026‑07‑07), npm `teeworlds` 2.6.1 в `~/aiddnet/DDNet-AI/node_modules`.
Непроверенные утверждения помечены «не проверено». Идентификаторы — как в оригинале.

## 0. Кратко

| Компонент | Вердикт |
|---|---|
| Huffman, varint/packer | взять libtw2 как есть |
| UDP‑соединение 0.6+TKEN | форк libtw2-net (`Connection`, sans‑IO) + своя обёртка; мелкие правки (panic→Result, таймауты) |
| Клиентская сессия (вход, карта, INPUT‑тайминг, REDIRECT/RECONNECT, анти‑бот) | писать своё — готового нет |
| Сообщения/объекты DDNet | форк libtw2-gamenet-ddnet, регенерация под 20.2 (сейчас 19.6; нет `MapDetails.size/url`, `Sv_MapInfo`, `MapBestTime`…) |
| Снапшоты | взять libtw2-snapshot |
| Карты (game/front/tele/speedup/switch/tune) | форк libtw2-datafile+map (MIT; 2423/2428 карт ок; патч «несколько game‑слоёв»); twmap (AGPL‑3.0‑only) — только dev‑оракул |
| Демо | взять libtw2-demo (v3–v6, SHA256‑ext, ex‑объекты; проверено на реальных v6); восстановление вводов — своё поверх физики |
| Мастер‑сервер | своё (HTTP JSON, ~200 строк) |

Ключевые внешние риски (проверено вживую): популярные блок‑сервера и официальные DDNet банят дата‑центровые IP («bad ip», «VPN detected»); капча‑лобби через REDIRECT; `CHECKSUM_REQUEST` от модов; `sv_connlimit` 5/20 с.

## 1. Протокол DDNet 0.6 (+DDNet‑расширения) — то, что нужно клиенту

Источник истины — DDNet master (shallow clone `~/aiddnet/ref/ddnet`, коммит `9576fd6` от 2026‑09‑27, `DDNET_VERSION_NUMBER 20020` / «20.2», `src/game/version.h:8-12`). Текущий релиз — 20.1 (2026‑09‑26), официальные сервера уже на master «0.6, 20.2 <hash>» (данные fork A: github releases/ddnet.org и поле `version` в servers.json). `GAME_NETVERSION = "0.6 626fce9a778df4d4"` (`version.h:20`) не менялся годами. Все пути ниже — относительно `~/aiddnet/ref/ddnet`.

Хорошее вспомогательное описание формата — `~/aiddnet/ref/libtw2/doc/{packet,protocol,connection,snapshot,huffman,int,demo,datafile,map,quirks}.md` (MIT/Apache).

### 1.1 Пакет (UDP, ≤1400 байт)

`src/engine/shared/network.h:20-50, 67-111`, `network.cpp:193-413`.

Заголовок соединения (connection‑oriented) — 3 байта:

```
byte0 = (flags << 2) | ((ack >> 8) & 3)
byte1 = ack & 0xff               // ack: 10 бит (последний in-order vital seq, полученный от пира)
byte2 = num_chunks
flags (6 бит, network.h:85-92):
  1<<0 UNUSED  (в 0.6 = 0; установлен ⇒ пакет 0.7/sixup)
  1<<1 TOKEN   (0.6.5, DDNet не использует)
  1<<2 CONTROL
  1<<3 CONNLESS
  1<<4 RESEND  (просьба пиру переслать все неподтверждённые vital-чанки)
  1<<5 COMPRESSION (payload сжат Huffman)
```

* Payload (после заголовка) сжимается Huffman, если это даёт выигрыш (`network.cpp:225-244`); control‑пакеты никогда не сжимаются (`network.cpp:141-147`).
* **DDNet security token (0.6+DDNet):** после установления соединения к *каждому* пакету (в т.ч. control, keepalive, close) в конец несжатого payload дописываются 4 байта токена big‑endian, и только потом всё сжимается (`network.cpp:217-223`). Приёмник отрезает последние 4 байта и сверяет (`network_conn.cpp:365-377`); при несовпадении пакет молча отбрасывается. Если токен «UNSUPPORTED» (=0) — не дописывается.
* Валидация (`network.cpp:134-155`): flags кроме CONTROL|RESEND|COMPRESSION запрещены; control ⇒ 0 чанков, ≥1 байт, без сжатия; обычный пакет ⇒ ≥1 чанк (0 допустимо только с RESEND), ≤255 чанков.

Чанк (`network.h:46-49`, `network.cpp:443-469`), split = 4 для 0.6:

```
b0 = (flags & 3) << 6 | (size >> 4) & 0x3f      // flags: bit6 = VITAL(1), bit7 = RESEND(2)
b1 = size & 0x0f | ((seq >> 2) & 0xf0)          // только если VITAL
b2 = seq & 0xff                                  // только если VITAL
size ≤ 1023 (NET_MAX_CHUNK_SIZE), seq — 10 бит (NET_MAX_SEQUENCE=1024), биты 6..7 seq дублируются (quirk)
```

Connless (`network.cpp:157-191, 313-340`): 6 байт `0xFF` (или `'x','e'` + 4 байта extra‑data — «extended», используется браузером серверов) + данные. Для запроса инфо сервера: `ff ff ff ff 'g' 'i' 'e' '3' <token>` → ответ `inf3`/`iext`/`iex+` (`src/engine/shared/masterserver.cpp:3-10`). Боту для выбора сервера это не нужно (достаточно HTTP‑мастера, §5), но полезно для пинга.

### 1.2 Надёжная доставка (vital), ack, resend

`network_conn.cpp:101-245, 356-577`, `network.cpp:57-127, 471-488`.

* Отправитель: у vital‑чанка `m_Sequence = (m_Sequence+1) % 1024`; копия кладётся в resend‑буфер (32 КиБ ring) (`network_conn.cpp:143-202`).
* Каждый исходящий пакет несёт `ack = m_Ack` — последний принятый *по порядку* vital seq пира (`Flush`, `network_conn.cpp:121-141`).
* Приёмник: vital‑чанк принимается, только если `seq == (m_Ack+1) % 1024`; если seq «в прошлом» (`IsSeqInBackroom`: в окне 512 назад) — молча дубликат; иначе — `SignalResend()` (выставить флаг RESEND в следующем исходящем пакете) и чанк пропускается (`network.cpp:94-117`). Т.е. **никакой буферизации out‑of‑order — только go‑back‑N**.
* Приход пакета с ack: из resend‑буфера удаляются все чанки с seq ≤ ack (в «заднем» окне) (`AckChunks`). Пришёл флаг RESEND ⇒ переслать весь буфер (с rate‑limit `conn_resend_requests_per_second`, новое в 20.x; `network_conn.cpp:233-245`).
* Таймеры (`Update`, `network_conn.cpp:504-577`): чанк без ack > 1 с — переслать (только первый в буфере); без ack дольше `conn_timeout` (100 с по умолчанию, `config_variables.h:650`) — ошибка «Too weak connection»; нет входящих `conn_timeout` c — «Timeout». В ONLINE: flush очереди каждые 500 мс, KEEPALIVE если ничего не слали 1 с; в CONNECT — повтор CONNECT каждые 500 мс.
* Отдельно: ack в заголовке пакета проверяется на валидность `m_PeerAck ≤ ack ≤ m_Sequence` (с wrap) — иначе пакет отбрасывается (`network_conn.cpp:382-393`).

### 1.3 Control‑сообщения и рукопожатие 0.6+DDNet («TKEN»)

`NET_CTRLMSG_KEEPALIVE=0, CONNECT=1, CONNECTACCEPT=2, ACCEPT=3, CLOSE=4` (`network.h:97-101`). Магия `SECURITY_TOKEN_MAGIC = "TKEN"` (`network.cpp:19`).

Последовательность, которую ждёт современный DDNet‑сервер (клиент: `network_conn.cpp:204-212, 455-480`; сервер: `network_server.cpp:513-542, 597-614`):

```
C→S  CONTROL: [01]['T''K''E''N'][ff ff ff ff]        // токен UNKNOWN(-1) дописан как обычный «хвост»
S→C  CONTROL: [02]['T''K''E''N'][token BE u32][token]  // ответ содержит токен дважды: как данные и как хвост
C→S  CONTROL: [03][token]                              // ACCEPT; «хвост» = токен ⇒ сервер делает TryAcceptClient
C→S  далее все пакеты: ... + [token]
```

* Сервер распознаёт DDNet‑CONNECT по `data[0]==CONNECT && size ≥ 9 && data[1..5]=="TKEN"` (`network_server.cpp:597-614`); токен — детерминированный хэш адреса (`GetToken(Addr)`), поэтому повторный CONNECT безопасен.
* CONNECT без «TKEN» (ванильный 0.6) ⇒ ветка anti‑spoof (`network_server.cpp:342-511`): сервер шлёт CONNECTACCEPT без токена + в одном пакете `MAP_CHANGE("dummy")`, `MAP_DATA`, `CON_READY` и 3×`SNAPEMPTY`, где «game tick» = vanilla‑токен; клиент обязан вернуть его как AckGameTick в `NETMSG_INPUT`. **Нам этот путь не нужен — использовать DDNet‑токен.** (npm‑`teeworlds`, которым пользуется TS‑бот, тоже идёт TKEN‑путём: `node_modules/teeworlds/lib/client.js:390, 433-435`.)
* CLOSE: `[04][reason\0]` (+ токен). Сервер на бан/переполнение отвечает CLOSE с текстом.
* 0.7/sixup: другой формат (токен в заголовке, `NET_CTRLMSG_TOKEN`, чанк‑split 6, иные id сообщений, `protocol7.h`). Боту не нужен: у 1363 из 1368 серверов есть `tw-0.6` адрес (fork A).

Новые анти‑DoS ограничения 20.x, важные для бота (fork A, PR #12609/#12635/#12865): `sv_connlimit` (5 подключений за 20 с с одного IP), лимит resend‑запросов, бюджет декомпрессии для адресов без слота. ⇒ частые реконнекты с одного IP рискованны.

### 1.4 Сообщения: кодирование, системные vs игровые, UUID‑расширения

* Payload чанка = packer‑поток. Первое число `(MsgId << 1) | Sys` (`client.cpp:179-188`). Целые — varint Teeworlds (`compression.cpp:9-35`: 1‑й байт `[ext:1][sign:1][6 бит]`, далее `[ext:1][7 бит]`…, до 5 байт, знак через `^= -sign`). Строки — NUL‑terminated, с санацией (`packer.h:26,63-73`). Raw — как есть.
* `MsgId == 0` ⇒ `NETMSG_EX`/`NETMSGTYPE_EX`: далее 16 байт UUID (`protocol_ex.cpp:20-51`). UUID = MD5(namespace `e05ddaaa-c4e6-4cfb-b642-5d48e80c0029` ‖ имя) с битами version 3 / variant 1 (`uuid_manager.cpp:14,38-58`). Неизвестный UUID ⇒ сообщение игнорируется (UNPACKMESSAGE_ERROR). Сервер может прислать `NETMSG_WHATIS` — нужно ответить `NETMSG_ITIS`/`IDONTKNOW` (`protocol_ex.cpp:55-77`) — желательно, но не критично.
* Список системных (sys=1) id (`protocol.h:32-76`): 1 INFO, 2 MAP_CHANGE, 3 MAP_DATA, 4 CON_READY, 5 SNAP, 6 SNAPEMPTY, 7 SNAPSINGLE, 8 SNAPSMALL (не используется), 9 INPUTTIMING, 10 RCON_AUTH_STATUS, 11 RCON_LINE, 14 READY, 15 ENTERGAME, 16 INPUT, 17 RCON_CMD, 18 RCON_AUTH, 19 REQUEST_MAP_DATA, 22 PING, 23 PING_REPLY, 25 RCON_CMD_ADD, 26 RCON_CMD_REM.
* Системные UUID‑сообщения (`protocol_ex_msgs.h:27-47`): what-is/it-is/i-dont-know@ddnet.tw, rcon-type, **map-details@ddnet.tw**, **capabilities@ddnet.tw**, **clientver@ddnet.tw**, ping/pong@ddnet.tw (PINGEX/PONGEX), checksum-request/response/error@ddnet.tw, **redirect@ddnet.org**, rcon-cmd-group-start/end, map-reload, **reconnect@ddnet.org**, sv-maplist-add/start/end.
* Сервер отбрасывает не‑vital сообщения, кроме `INPUT`, `PING`, `PINGEX` (`server.cpp:1771-1786`); и наоборот **`NETMSG_INPUT`, пришедший vital, отбрасывается** (`server.cpp:1862-1865`).

### 1.5 Полный сценарий входа (как делает DDNet‑клиент 20.2)

| # | Направление | Сообщение (flags) | Поля / смысл | Ссылка |
|---|---|---|---|---|
| 1 | C→S | handshake §1.3 | | |
| 2 | C→S | `NETMSG_CLIENTVER` (sys, UUID, vital) | `ConnectionId` (16 байт случайный UUID), `DDNetVersion` int, `VersionStr` string (напр. `"DDNet 20.2"`) | `client.cpp:240-246`, сервер `server.cpp:2053-2066` |
| 3 | C→S | `NETMSG_INFO` (sys, vital+flush) | `"0.6 626fce9a778df4d4"`, password | `client.cpp:255-261`; сервер кикает при другой строке `server.cpp:2073-2080` |
| 4 | S→C | `NETMSG_RCONTYPE`, `NETMSG_CAPABILITIES` (version=5, flags DDNET\|CHATTIMEOUTCODE\|ANYPLAYERFLAG\|PINGEX\|ALLOWDUMMY\|SYNCWEAPONINPUT) | | `server.cpp:1373-1391`, `protocol_ex.h:77-86` |
| 5 | S→C | `NETMSG_MAP_DETAILS` (UUID) | name, sha256 (32 raw), crc, size, url (может быть пустой) | `server.cpp:1393-1411` |
| 6 | S→C | `NETMSG_MAP_CHANGE` (vital+flush) | name, crc, size | `server.cpp:1412-1424` |
| 7 | C | карта есть в кэше (crc+sha256) ⇒ сразу `READY`; иначе загрузка (§1.6) | | `client.cpp:1717-1804` |
| 8 | C→S | `NETMSG_READY` (vital+flush) | | `client.cpp:270-274` |
| 9 | S→C | `NETMSG_CON_READY` (+ игровые: motd, …) | | `server.cpp:2119-2149` |
| 10 | C→S | `Cl_StartInfo` (game, vital+flush) | name, clan, country, skin, use_custom_color, color_body, color_feet | `gameclient.cpp:3234-3247`, `network.py:496-504` |
| 11 | S→C | `Sv_VoteClearOptions`, **`Sv_TuneParams`**, `Sv_ReadyToEnter` | | `gamecontext.cpp:3021-3061` |
| 12 | C→S | `NETMSG_ENTERGAME` (vital+flush) | сервер требует `IsClientReady` (т.е. StartInfo принят) | `client.cpp:513-527`, `server.cpp:2151-2185` |
| 13 | S→C | снапшоты: сначала раз в 10 тиков (SNAPRATE_INIT) до первого ack во `NETMSG_INPUT`, потом каждый «глобальный» тик | | `server.cpp:1043-1057` |
| 14 | C→S | `NETMSG_INPUT` каждый тик предсказания (не vital, flush) | §1.8 | `client.cpp:350-397` |
| 15 | C→S | после первого снапшота с local player: `Cl_IsDDNetLegacy` (game, vital) c int версии — совместимость со старыми модами | | `gameclient.cpp:2288-2302` |
| 16 | C→S | (опц.) через 50 снапшотов, если сервер умеет `CHATTIMEOUTCODE`: `Cl_Say "/timeout <code>"` — защита от таймаута/перезахода | | `client.cpp:530-540` |

Какую версию объявлять: `GetMaxClients` (`server.cpp:2980-2993`): DDNetVersion ≥ 19000 (`VERSION_DDNET_128_PLAYERS`) ⇒ реальные id 0..127 без маппинга; ≥ 2 ⇒ 64 слота с «player mapping» (сервер показывает ближайших 63); иначе ванильные 16. Ещё пороги (`protocol.h:142-169`): REDIRECT 17020, RECONNECT 18090, PREINPUT 19040, 128_TEAMS 20000, PICKUP_FREEZE 20010. **Рекомендация: объявлять 20020 (как текущий клиент) и корректно игнорировать неизвестные UUID‑сообщения/объекты.** Это даёт 128 id, Sv_PreInput, NETMSG_REDIRECT/RECONNECT.

### 1.6 Загрузка карты: HTTPS vs in‑protocol

* Если `MAP_DETAILS` пришёл и совпадает с `MAP_CHANGE` (name/size/crc), клиент качает по HTTPS: URL из `MAP_DETAILS` (сервер строит `sv_maps_base_url + "<name>_<sha256>.map"`, `server.cpp:3182-3191`), иначе `cl_map_download_url` (по умолчанию `https://maps.ddnet.org`, `config_variables.h:149`, либо `map-download-url` из `https://info.ddnet.org/info`, `client.cpp:2613-2617`) + `/<name>_<sha256hex>.map`; с `ExpectSha256` и `MaxResponseSize` (`client.cpp:1784-1797`). При любой ошибке HTTP — откат на in‑protocol (`client.cpp:3046-3058`).
* **In‑protocol** (`client.cpp:276-291, 1805-1881`; `server.cpp:1430-1471, 1813-1850`): клиент шлёт `REQUEST_MAP_DATA(chunk=0)` (vital+flush); сервер отвечает `MAP_DATA{last, crc, chunk, size, raw[size]}`, chunk size = 1023−128 = 895 байт; при `sv_fast_download=1` сервер сразу высылает окно `sv_map_window=15` чанков и далее по одному на каждый запрос (конвейер); клиент запрашивает `chunk+1` после каждого принятого. Сервер ограничивает 2×N запросов чанков. Проверки клиента: crc и номер чанка совпадают; по завершении `LoadMap` сверяет CRC (и SHA256, если известен). Кэш: `downloadedmaps/<name>_<sha256>.map` (или `_<crc08x>`).
* Факт (fork A): 4 из 5 самых людных Block‑карт **отсутствуют** на maps.ddnet.org (404) ⇒ in‑protocol загрузка обязательна; HTTPS — оптимизация.
* `MAP_CHANGE` может прийти в любой момент (смена карты) ⇒ клиент сбрасывает состояние и повторяет шаги 7–12.

### 1.7 Снапшоты

Сервер (`server.cpp:1020-1172`), клиент (`client.cpp:2110-2360`), формат (`snapshot.h`, `snapshot.cpp`).

* Частота: снапшоты только на «глобальных» тиках — **каждый 2‑й тик (25 Гц)**, если не `sv_high_bandwidth` (`server.cpp:1024`); исключение — наблюдатель с force high bandwidth. Тик сервера = 50 Гц (`SERVER_TICK_SPEED`).
* Сообщения: `SNAPSINGLE{tick, tick−delta_tick, crc, size, data}`; `SNAP{tick, tick−delta_tick, num_parts, part, crc, size, data}` (части по ≤900 байт `MAX_SNAPSHOT_PACKSIZE`, ≤64 частей); `SNAPEMPTY{tick, tick−delta_tick}` (дельта пустая). Все — не vital, flush. `SNAPSMALL` не используется. `delta_tick = −1` ⇒ база — пустой снапшот.
* Данные = varint‑сжатый массив int32 дельты (без отдельного Huffman — Huffman на уровне пакета). Формат дельты (`snapshot.cpp:517-630`): `[num_deleted][num_updated][num_temp=0]`, затем `num_deleted` ключей, затем для каждого обновлённого: `type, id, [size_in_ints — только если размер типа не статический], diff[size]`; значение = старое + diff (wrapping), новый элемент — копия.
* **Статические размеры** (`SetStaticsize`, `gamecontext.cpp:4144-4147`, `gameclient.cpp:360`): для типов 0..20 (не‑ex объекты из `datasrc/network.py`) размер не передаётся — клиент обязан знать ровно те же размеры, что сервер. Тип 0 (NETOBJTYPE_EX) имеет размер 0 ⇒ передаётся. Порядок id (`datasrc/compile.py:64-68`): 1 PlayerInput(10 int), 2 Projectile(6), 3 Laser(5), 4 Pickup(4), 5 Flag(3), 6 GameInfo(8), 7 GameData(4), 8 CharacterCore(15), 9 Character(22), 10 PlayerInfo(5), 11 ClientInfo(17), 12 SpectatorInfo(3), 13 Common(2), 14 Explosion(2), 15 Spawn(2), 16 HammerHit(2), 17 Death(3), 18 SoundGlobal(3), 19 SoundWorld(3), 20 DamageInd(3). (Размеры посчитаны по полям `network.py:111-370` и совпадают с таблицей npm‑`teeworlds` `lib/snapshot.js:8-28`, которая работает с живыми серверами.)
* **Ex‑объекты (UUID‑типы)**: в снапшоте есть элемент type=0, id=`внутренний тип` (0x4000..0x7fff; строитель раздаёт с 0x7fff вниз), data = UUID как 4 int big‑endian; элементы ex‑типа имеют этот внутренний тип (`snapshot.cpp:43-61, 825-845`). Сопоставление живёт внутри каждого снапшота. Ex‑объекты из `network.py:242-405`: `DDNetCharacter` (character@netobj.ddnet.tw, **переменный размер**, `validate_size=False`: Flags, FreezeEnd, Jumps, TeleCheckpoint, StrongWeakId, JumpedTotal, NinjaActivationTick, FreezeStart, **TargetX, TargetY**, TuneZoneOverride), `DDNetPlayer`, `GameInfoEx` (переменный), `DDRaceProjectile`, `DDNetLaser`, `DDNetProjectile`, `DDNetPickup`, `DDNetSpectatorInfo`, `SpectatorCount`, события `Birthday`, `Finish`, `MapSoundWorld`, `SpecChar`, `SwitchState` (переменный), `EntityEx`, `MapBestTime`.
* Ключ элемента = `(type << 16) | id`; id = client id для Character/PlayerInfo/ClientInfo/DDNetCharacter/DDNetPlayer.
* CRC = сумма (wrapping u32) всех int данных всех элементов (без ключей) (`snapshot.cpp:112-124`). При несовпадении клиент выбрасывает снапшот; >10 ошибок подряд ⇒ `AckGameTick = −1` (сервер пришлёт полный). Нет базового снапшота для delta_tick ⇒ тоже ack −1 (`client.cpp:2170-2190, 2221-2234`).
* Сборка частей: битовая маска частей по тику, берутся только тики > последнего ack (`client.cpp:2143-2160`).
* Хранилище: клиент держит снапшоты по тикам и чистит всё старее delta_tick (`client.cpp:2240-2247`); сервер дельтит против `m_LastAckedSnapshot` (из `NETMSG_INPUT`) и хранит до 3 с (`server.cpp:1087-1113`).
* **Dead reckoning (важно для траекторий)**: `Character` другого игрока содержит не текущее ядро, а `m_SendCore` на тике `m_Tick` (`character.cpp:1087-1104`); сервер пересылает новое ядро, только если «предсказание без ввода» (`m_ReckoningCore.Tick(false)` в пустом мире с **дефолтным** тюнингом) разошлось с реальностью или прошло 3 с (`character.cpp:856-868, 953-969`). Клиент обязан прогнать физику от `m_Tick` до тика снапшота. Поля `m_Direction`, `m_Weapon`, `m_AttackTick`, `m_PlayerFlags` пишутся текущими (`character.cpp:1143-1170`).
* Network clipping: сервер шлёт объекты только в радиусе `ShowDistance/2 + 320 (+dyncam)` от камеры игрока (`player.cpp:1118-1141`, `entity.cpp:95-103`); `ShowDistance` задаёт клиент сообщением `Cl_ShowDistance{x,y}` без серверной проверки (`gamecontext.cpp:2778-2782`), по умолчанию 1200×800.

### 1.8 Ввод: `NETMSG_INPUT`

`client.cpp:350-397`, сервер `server.cpp:1860-1990`.

```
NETMSG_INPUT (sys, НЕ vital, flush):
  AckGameTick   int   // последний целиком принятый и проверенный снапшот (или −1)
  PredTick      int   // «intended tick»: на каком тике сервер должен применить ввод
  Size          int   // байт; 40 для CNetObj_PlayerInput (10 int), допустимо 10..128 int
  data[Size/4]  int   // m_Direction(-1/0/1), m_TargetX, m_TargetY (относительно тии, ≠(0,0)),
                      // m_Jump(0/1), m_Fire (счётчик нажатий+отпусканий, &0x3f, нечётный = зажат),
                      // m_Hook(0/1), m_PlayerFlags (PLAYING=1, IN_MENU=2, CHATTING=4, SCOREBOARD=8, AIM=16, …),
                      // m_WantedWeapon (оружие+1, 0 = не менять), m_NextWeapon, m_PrevWeapon (счётчики как Fire)
```

* Сервер: `LastAckedSnapshot > Tick()` или вне диапазона ⇒ отброс; хранит 200 входов; применяет на тике `max(IntendedTick, Tick()+1)`; «сырой» ввод применяется сразу через `OnClientDirectInput` (флаги, огонь по краю). Для новых `IntendedTick` отвечает `NETMSG_INPUTTIMING{IntendedTick, TimeLeft_ms}` (не vital) (`server.cpp:1900-1917`).
* Клиент: `PredTick` вычисляется из сглаженного «предсказанного времени» (`m_PredictedTime`, старт = тик снапшота + margin), корректируется по `INPUTTIMING` так, чтобы `TimeLeft ≈ cl_prediction_margin` (10 мс по умолчанию) (`client.cpp:2084-2108, 2305-2316, 2934-2965`). Ввод отправляется при каждом приращении PredTick (≈50 Гц).
* Семантика счётчиков (`gamecore.h:296-313` `CountInput`, `controls.cpp:80-90`): нажатие/отпускание = +1, чётность = состояние.
* `Sv_PreInput` (UUID‑сообщение, `network.py:633-647`, `server.cpp:1925-1978`, `gamecontext.cpp:1530-1561`): при `sv_preinput=1` (по умолчанию) сервер пересылает **чужие вводы** (Direction, Target, Jump, Fire, Hook, оружие, Owner, IntendedTick) клиентам с версией ≥19040 из той же DDRace‑команды в радиусе видимости, не AFK. Для бота‑противника это прямой источник вводов соперников (с `MSGFLAG_NORECORD` — в демки не пишется).

### 1.9 Прочие нужные сообщения

Системные (`client.cpp:1925-2110`):

* `PING` → ответить `PING_REPLY` (тот же vital‑флаг); `PINGEX{uuid}` → `PONGEX{uuid}`.
* `REDIRECT{port:int}` (≥17020): переподключиться к тому же IP на новый порт (`client.cpp:1991-2020`, сервер `server.cpp:570-596`; старым клиентам — кик «Redirect unsupported»).
* `RECONNECT` (≥18090): переподключиться к тому же адресу.
* `MAP_RELOAD`: касается dummy; `CHECKSUM_REQUEST`: ванильный сервер не шлёт (grep пуст), моды/антибот могут (не проверено).
* `RCON_*`: боту не нужно.

Игровые (`datasrc/network.py:408-680`; id = порядок не‑ex сообщений начиная с 1):

* `Sv_TuneParams` (id 6): последовательность int = параметры тюнинга ×100 (fixed‑point) в порядке `src/game/tuning.h` (47 параметров, последние — `ground_elasticity_x/y`); старые сервера шлют меньше — читать до ошибки распаковки (`gameclient.cpp:1063-1095`). Сервер шлёт при входе и при каждой смене tune‑зоны персонажа/фейк‑тюнинге (`gamecontext.cpp:1108-1180`, `character.cpp:1127-1133`). Для зон есть также `DDNetCharacter.m_TuneZoneOverride`.
* `Sv_KillMsg{killer, victim, weapon, mode_special}` (id 4); `Sv_KillMsgTeam` (ex). Кого считать «убийцей» при блок‑килле — зависит от мода (не проверено для каждого).
* `Sv_Broadcast{msg}` (id 2), `Sv_Chat{team, client_id, msg}` (id 3, игнорировать), `Sv_Motd` (id 1), `Sv_ReadyToEnter` (id 8), `Sv_Emoticon`, `Sv_WeaponPickup`, голосования.
* Ex: `Sv_DDRaceTime`, `Sv_Record`, `Sv_TeamsState`, `Sv_YourVote`, `Sv_RaceFinish`, `Sv_CommandInfo*`, `Sv_ChangeInfoCooldown`, `Sv_MapSoundGlobal`, **`Sv_PreInput`**, `Sv_SaveCode`, `Sv_ServerAlert`, `Sv_ModeratorAlert`, `Sv_MapInfo`.
* Клиентские: `Cl_Say`, `Cl_SetTeam`, `Cl_StartInfo`, `Cl_ChangeInfo`, `Cl_Kill`, `Cl_Emoticon`, `Cl_Vote`, `Cl_IsDDNetLegacy`, ex: `Cl_ShowDistance`, `Cl_ShowOthers`, `Cl_CameraInfo`, `Cl_EnableSpectatorCount`.

### 1.10 Huffman и packer

* Huffman: статическое дерево из фиксированной таблицы частот 257 символов (256 + EOF) (`src/engine/shared/huffman.cpp`); quirk — лишний байт, если конец ровно на границе байта (`libtw2/doc/quirks.md`).
* Packer/Unpacker (`packer.cpp`): varint, строки с санацией (`SANITIZE_CC`), raw; строки в ClientInfo — «int‑strings» (4 байта в int, со сдвигом −128), `NetTwIntString` (`network.py:226-234`).

### 1.11 Что меняется у DDNet и чем грозит

* Добавление полей в конец ex‑объектов с `validate_size=False` (DDNetCharacter получил TargetX/TargetY/TuneZoneOverride) — ридер обязан принимать и короче, и длиннее.
* Новые UUID‑сообщения появляются почти в каждом релизе — неизвестные надо игнорировать.
* Статические размеры 0.6‑объектов не менялись с 0.6 (менять их нельзя без поломки всех клиентов) — стабильная часть.
* 20.0: 128 игроков; 20.x: лимиты анти‑DoS (fork A). Изменений рукопожатия TKEN за 2025–26 не найдено (fork A: правки control‑токенов касались только 0.7, #10568/#10545); с какого года оно неизменно — не проверено.
## 2. libtw2 — покомпонентный вердикт

`~/aiddnet/ref/libtw2` (github.com/heinrich5991/libtw2), копия для экспериментов — `proto-scratch/libtw2`. Аудит — fork B (сборка, тесты, живой клиент), выборочная перепроверка — мной.

### 2.1 Лицензия, публикация, сопровождение

* Все 36 крейтов — **MIT/Apache‑2.0** (`LICENSE-MIT`, `LICENSE-APACHE`, поле `license` в Cargo.toml); транзитивные зависимости — только пермиссивные (MIT/Apache/BSD/Zlib/ISC). **Совместимо с GPL‑3.0‑only** без оговорок.
* crates.io: опубликованы под префиксом **`pre-rfc3243-libtw2-*`** (проверено API 2026‑09‑27): `-net` 0.2.0 (2026‑04‑18), `-snapshot` 0.2.0, `-packer`, `-huffman`, `-common`… 0.2.0, `-gamenet-ddnet` 0.2.1 (2026‑09‑07), `-demo` 0.3.0 (2026‑09‑03). **`datafile`, `map`, `gamenet-teeworlds-0-6`, `event-loop` — не опубликованы** (только git). Имена `libtw2-*` без префикса — пустые заглушки. Соответствует ли `gamenet-ddnet 0.2.1` спецификации новее 19.6 — не проверено (в git master генерация по `ddnet-19.6.py`, `gamenet/generate_all:9`).
* Активность: коммиты 2023 — 34, 2024 — 93, 2025 — 23, 2026 — 46; последний 2026‑09‑02. Фактически один автор — heinrich5991 (727 коммитов; он же core‑разработчик DDNet), всего 20 авторов. CI (`.github/workflows/build.yaml`): build/test/bench на 3 ОС, stable/nightly/1.63, проверка сгенерированного кода и rustfmt. Спека DDNet в gamenet обновляется раз в 6–12 мес (19.1 — 2025‑05, 19.6 — 2025‑12).
* Реальное применение: ddnet-rs `game/legacy-proxy` (MIT/Apache) — рабочий 0.6/DDNet‑клиент поверх `libtw2-net`, `-gamenet-ddnet`, `-packer`, `-snapshot` (форк Jupeyy/libtw2 с 4 патчами: дефолт для неизвестных tune, толерантность к новым полям DDNetCharacter, «send connect twice»). DDNet в 2026‑04 вливал Rust‑снапшоты на libtw2 (#11957) и откатил (#12340) из‑за производительности серверного builder'а — для клиента не критично (fork A).

### 2.2 Сборка и тесты (rustc 1.98.1)

* `cargo build -j4 --locked` (net, packer, huffman, snapshot, gamenet ddnet/0.6/0.7, demo, datafile, map, event-loop, tools, downloader, stats-browser): 0 ошибок, ~35 с. Единственное предупреждение — `binrw 0.11.1` (через demo) будет отвергнут будущими rustc (future‑incompat).
* `cargo test`: 169/169 (net 123, packer 26, huffman 8 — сверка с C++‑эталоном, snapshot 6, serverbrowse 6). У demo/datafile/map/gamenet юнит‑тестов нет.
* MSRV 1.63, edition 2021.

### 2.3 Компоненты

| Компонент | Что есть | Проблемы для DDNet 20.x | Вердикт |
|---|---|---|---|
| **huffman** | статическое дерево DDNet, тесты против C++ | — | **as‑is** |
| **packer** | varint, строки как сырые `&[u8]`, `sanitize` отдельно | санацию вызывать самим | **as‑is** |
| **net** (`net/src/connection.rs`) | sans‑IO `Connection` c трейтом `Callback{send,time,secure_random}` (`:26`), TKEN‑рукопожатие (`:621`, `protocol.rs:54`), ack/vital seq/resend (1 с или по флагу), keepalive, Huffman, connless, sixup (`connection7.rs`) | нет таймаута приёма и лимита очереди (TODO `connection.rs:23-24`); `send()`/`flush()` в не‑online состоянии — **panic `state not online`** (`State::assert_online`, `connection.rs:99-104`; это контракт API, но пробник упал на нём при REDIRECT — обёртка обязана проверять состояние); `socket`/`event-loop` на mio 0.6/net2 — не брать | **нужны небольшие изменения** (Result вместо panic, таймауты) |
| **snapshot** | сборка частей SNAP/SINGLE/EMPTY (`receiver.rs`), storage на 100 тиков, дельты, CRC (`storage.rs:127-129`), UUID‑типы через элементы type 0 (`format.rs:36`, `snap.rs:449-476`), размер неизвестных типов из потока (`snap.rs:796-802`) | — | **as‑is** |
| **gamenet-ddnet** | сгенерированные системные/игровые сообщения и snap‑объекты; все системные EX‑UUID есть (CAPABILITIES, CLIENTVER, PINGEX/PONGEX, CHECKSUM_*, REDIRECT, RECONNECT, RCON_CMD_GROUP_*, MAP_RELOAD, MAPLIST_*); `DDNetCharacter` полный (TargetX/Y, TuneZoneOverride, дефолты для отсутствующих хвостов) | генерация по **19.6**: нет `Sv_MapInfo`, `Cl_PracticeTeleport`, `MapBestTime` (на живом 20.0 пришли UnknownId); теряются новые поля `DDNetPlayer.FinishTime*`, `GameInfoEx.Min/MaxTeamSize/NumDDRaceTeams`; **`MapDetails` без `size` и `url`** (`msg/system.rs:1217-1226` vs `server.cpp:1397-1410`); `Sv_TuneParams` декодер строгий (меньше 47 параметров ⇒ ошибка, DDNet — мягко); `unsafe` transmute `repr(C)` (37 мест); регенерация под 20.2 требует правки генератора (`NetTickStrict`, `FinishTime::*`) ~1–3 ч (не проверено до конца) | **форк + регенерация** |
| **demo** | версии 3–6 (`format.rs:40-45`), timeline markers, SHA256‑расширение, встроенная карта, writer, seek по keyframe; high‑level `ddnet::DemoReader<P>` отдаёт тики, сообщения, декодированные объекты; неизвестные объекты — warning (`ddnet/reader.rs:244`) | на 5 реальных DDNet v6‑демках (из тестов twgame, `proto-scratch/demos/`) всё читается, `Character`/`DDNetCharacter` с `target_x/y`; ~4200 безвредных UnknownId на системных сообщениях в серверных per‑player демках; зависит от устаревшего `binrw 0.11` | **as‑is** (с форком gamenet для новых объектов) |
| **datafile** | v3/v4, zlib через `libz-sys` (C), writer | 23 `unsafe` | **as‑is** |
| **map** | `game_layers()` → game, front, tele, speedup, switch, tune с типами тайлов | корпус ddnet-maps (2428 карт, 21 с release): 2423 ок; **5 падают `TooManyGameLayers`** (`map/reader.rs:744`), DDNet берёт последний game‑слой (`layers.cpp:33-36`); `unsafe vec::transmute` тайлов (`reader.rs:828,855`); нет разбора `Info.settings` (не проверено) | **небольшая правка** |
| клиент целиком | `downloader` (0.6, map download, input) на `event-loop` | без CLIENTVER/EX; на официальном DDNet‑сервере получил REDIRECT/RECONNECT и завис (fork B) | **нет — писать своё** |
| мастер‑сервер | `serverbrowse`, `stats-browser`, `register` — legacy UDP‑мастера/серверная сторона | HTTP `servers.json` клиента нет | **нет — писать своё** (тривиально) |
| teehistorian | парсер есть (`teehistorian/`) | не проверялся | справочно |

### 2.4 Живые проверки

* fork B: минимальный клиент на libtw2 (`proto-scratch/netprobe`) прошёл с неофициальным DDNet 20.0 весь путь: TKEN → CAPABILITIES/MAP_DETAILS → MAP_CHANGE → CON_READY → StartInfo/EnterGame → 60 декодированных снапшотов, сервер отвечал INPUTTIMING на наши INPUT.
* Мной (тот же пробник, 2026‑09‑27, ≤20 с на сервер, ник `research-probe`) (адреса ниже заменены на документационные, RFC 5737):
  * **GameUp Block (192.0.2.176:8311, DDNet 19.7‑мод, «Copy Love Box Mega»)** — полный вход, 60 снапшотов, ~51 игрок (Character 32/снап, ClientInfo 51/снап — 128‑id работает), `Sv_TuneParams`, `Sv_CommandInfo*`, **2× `NETMSG_CHECKSUM_REQUEST`** (моды реально шлют — ответа от нас нет, последствий за 20 с не было; что делает сервер с неответившими — не проверено). Сервер пустил в игру, **не проверяя, что карта скачана** (READY сразу).
  * **TeeFusion (192.0.2.103:51010)** — сразу после рукопожатия `NETMSG_REDIRECT port=51000` → на 51000 карта **`Captcha_CAP`, size=10 000 000, crc=f2159e6e** — анти‑бот капча‑лобби; пробник на libtw2 упал на `assert_online` при отправке после редиректа (ошибка обёртки, но показательно: API паникует вместо Result).
  * **TeeUnion (192.0.2.148:8500, самый людный)** — после MAP_CHANGE: «You have been banned for 10 minutes (bad ip)». fork B на официальном DDNet: «banned (VPN detected)». ⇒ **IP этой (дата‑центровой) машины отвергается популярными серверами; боту нужен «жилой» IP.**
* На официальном сервере fork B видел REDIRECT/RECONNECT на тот же порт сразу после INFO; в открытом коде DDNet `ReconnectClient`/`RedirectClient` (`server.cpp:544-596`) никем не вызываются — вероятно, закрытый antibot‑модуль (не проверено).

### 2.5 Качество кода

`unsafe`: net 1, huffman 14, snapshot 10, datafile 23, map 35, gamenet-ddnet 37, demo 0. Устаревшие зависимости: arrayvec 0.5, log 0.3, itertools 0.4, zerocopy 0.7, binrw 0.11, syn 1, uuid 0.8, mio 0.6/net2 (только socket), clap 2 (tools). Размер дерева (`cargo tree`, строк): net 26, snapshot 33, gamenet-ddnet 32, map 38, demo 46. Старый стиль, паттерн `Warn<W>` повсюду, публичный API почти без rustdoc, но отличная протокольная документация в `doc/`.

### 2.6 Какие крейты брать

`libtw2-huffman`, `-packer`, `-snapshot`, `-net` (только `Connection`), `-gamenet-ddnet` (форк, регенерация под 20.2 + MapDetails), `-demo`, `-datafile`, `-map` (+ транзитивно `common`, `buffer`, `warn`, `gamenet-common`, `gamenet-snap`, `zlib-minimal`). Не брать: `socket`, `event-loop`, `downloader`, `tools`.
## 3. twmap

Репозиторий `~/aiddnet/ref/twmap` (origin gitlab.com/Patiga/twmap; по fork A проект переехал в gitlab.com/ddnet-rs/twmap). Проверено fork C (сборка/тесты/64 реальные карты) и выборочно мной (лицензия).

### 3.1 Лицензия — AGPL‑3.0‑only

* `twmap/Cargo.toml:7`: `license = "AGPL-3.0-only"`; то же у `twmap-tools`, `twmap-web`; корневой `LICENSE` — полный текст GNU AGPL v3. Правообладатель — Patiga (≈495 из ≈510 коммитов).
* История: коммит `d846c9c` (2022‑08‑17) «LICENSE CHANGE: AGPL-3.0-only for both crates». На crates.io 0.4.0–0.7.0 (2021‑09…2022‑05) опубликованы как **LGPL‑3.0‑only**; 0.8.0+ — AGPL. (Старую LGPL‑версию можно форкнуть, но она на 4 года старее — совместимость с современными картами не проверена.)
* Тот же автор/группа ddnet-rs: `twgame` (физика DDNet на Rust), `twsnap`, `twstorage`, `teehistorian-replayer` — тоже AGPL‑3.0‑only (fork A). Крейт `teehistorian` — LGPL‑3.0.

### 3.2 Совместимость AGPL‑3.0‑only с нашим GPL‑3.0‑only (DDNet-AI: `package.json:3`, `LICENSE`)

* GPL‑3.0 §13 (`DDNet-AI/LICENSE:552-561`): «you have permission to link or combine any covered work with a work licensed under **version 3** of the GNU Affero General Public License … The terms of this License will continue to apply to the part which is the covered work, but the special requirements of the GNU Affero General Public License, section 13, concerning interaction through a network **will apply to the combination as such**.»
* AGPL‑3.0 §13 абз. 2 (`twmap/LICENSE:553-559`): зеркальное разрешение объединять с работой под GPL **version 3**; часть под GPL остаётся под GPL.
* Обе ссылки — ровно на «version 3», поэтому пара GPL‑3.0‑only + AGPL‑3.0‑only **совместима** (с GPL‑2.0‑only было бы нельзя; «-only» ничего не ломает, т.к. ссылки и так на v3).
* Цена: AGPL §13 абз. 1 (`twmap/LICENSE:542-551`): «if you modify the Program, your modified version must prominently offer **all users interacting with it remotely through a computer network** … an opportunity to receive the Corresponding Source … This Corresponding Source shall include the Corresponding Source for any work covered by version 3 of the GNU General Public License that is incorporated». По GPL §13 это требование распространяется на **комбинацию целиком**, т.е. на весь бинарник бота, включая наш GPL‑код.
* Применимость к боту (консервативно): триггер — модификация + удалённое взаимодействие пользователей с программой. Бот сам — клиент игрового сервера; но игроки общаются с ним через сервер (приказы владельца, автоответы в чате: `src/bot/ownerOrders.ts`, `autoChat.ts` по fork C), а веб‑UI локальный (`src/bot/web.ts:731`, 127.0.0.1). Мы, скорее всего, будем модифицировать форк twmap (или «комбинацию»), значит **надо исходить из того, что §13 применяется**. Выполнить дёшево — проект и так открыт: публичная ссылка на точный исходник каждой сборки + чат‑команда `!source`/строка в UI. Это моя трактовка, юридически не проверено.
* Итог: использовать twmap **можно**, но это добавляет сетевую copyleft‑обязанность ко всему бинарнику и привязывает нас к AGPL‑экосистеме; если хочется оставить «чистый» GPL‑3.0 без §13 — свой ридер (~400–600 строк; в TS‑боте уже есть `src/map/datafile.ts` 304 строки + `loadMap.ts` 251 строка) либо MIT/Apache‑код (libtw2 `datafile`/`map`, ddnet-rs `legacy-map`).

### 3.3 Возможности и качество

* Все слои: `Layer::{Game, Tiles, Quads, Front, Tele, Speedup, Switch, Tune, Sounds}` (`twmap/src/map/mod.rs:522`), envelopes, images, sounds, automapper, `Info.settings` (серверные команды карты — там `tune_zone …`, `sv_…`), форматы DDNet06/Teeworlds07/MapDir. Структуры тайлов совпадают с DDNet (`mod.rs:369-411`: `Tele{number,id}`, `Speedup{force,max_speed,id,pad,angle:i16}`, `Switch{number,id,flags,delay}`, `Tune{number,id}`).
* API: `TwMap::parse(&[u8])` / `parse_unchecked` (`src/map/parse.rs:41,47`); `map.find_physics_layer::<GameLayer|TeleLayer|…>()` (`src/map/edit/mod.rs:78`); `layer.tiles_mut().load()` → `ndarray::Array2<T>`. Ленивая распаковка: можно распаковать только физические слои.
* Строгость: `parse()` вызывает `check()` и **отвергает реальные карты** (fork C: `ctf4.map` — «Image 'jungle_doodads_old' is not a valid external image for DDNet06»); `parse_unchecked` прошёл 64/64 реальные карты; `physics_group()` делает `unwrap()` (`impls.rs:71`) — паника на карте без game‑группы. Выбор слоя отличается от DDNet (twmap: первая физ.группа, последний слой типа; DDNet: последний game‑слой по всем группам, `ddnet src/game/layers.cpp:33-36`) — на корректных картах не важно.
* Производительность (release): типичная карта 1–10 мс разбор + 1–20 мс физ.слои; большие (4170×1060) ~0.2+0.35 с. Block‑карты (Copy Love Box 387×250) 1–5 мс.
* Зависимости: feature‑флагов нет; дерево ~64 крейта: `image`(png), `ndarray`, `fixed`, `vek`, `serde`+`serde_json`, `flate2`, `opus_headers`, `twstorage`, `regex` (через sanitize-filename), `getrandom`, `thiserror`, `bytemuck`, `structview`, `bitflags`, `az`. Для задачи «прочитать физику» это тяжеловато, но терпимо (build debug ~30 с).
* Сопровождение: активен (2025 — 63 коммита, последний 2026‑07‑07, crates.io 0.15.0 от 2026‑06‑07), но фактически один автор. Тесты 20/20 на Rust 1.98.1.

### 3.4 Вердикт по twmap

Функционально — лучший готовый ридер карт на Rust (все нужные слои, проверен на реальных картах через `parse_unchecked`). Юридически — совместим, но тянет AGPL §13 на весь бинарник. Технически — лишние тяжёлые зависимости (image, ndarray, serde) ради ~5 массивов тайлов. **Рекомендация: не брать как зависимость; написать свой ридер physics‑слоёв (datafile v3/v4 + zlib + 6 слоёв + Info.settings), используя twmap/DDNet/libtw2‑doc только как справку по формату; twmap держать как dev‑dependency/инструмент для тестов‑оракулов (сравнение результатов на корпусе карт) — это не распространяется в бинарнике.** (Если команда сознательно принимает AGPL §13 — twmap можно брать как есть, с `parse_unchecked`.)
## 4. Демо (.demo v6 + DDNet) и извлечение вводов/траекторий

### 4.1 Формат файла

`src/engine/demo.h:15-55`, `src/engine/shared/demo.cpp:20-40, 130-360, 486-560, 800-880`; также `libtw2/doc/demo.md`, `demo.ksy`.

```
CDemoHeader (180 байт):
  marker[7] = "TWDEMO\0", version u8 (DDNet пишет 6; читает 3..6),
  netversion[64] ("0.6 626fce9a778df4d4" или "0.7 …"), map_name[64],
  map_size u32be, map_crc u32be, type[8] ("client"/"server"), length u32be (сек), timestamp[20]
CTimelineMarkers (260 байт, если version > 3): num u32be + 64×u32be тиков
SHA256‑расширение (version ≥ 6): UUID 6be6da4a-cebd-380c-9b5b-1289c842d780 ("demoitem-sha256@ddnet.tw") + 32 байта sha256 карты
                                  (при чтении — опционально: проверяется наличие UUID)
map data [map_size] — карта целиком встроена (если не слишком большая)
чанки до EOF:
  tick marker:  1xxxxxxx; bit6 = keyframe; bit5 = «сжатый тик» (v≥5): tick += low5bits;
                иначе далее u32be абсолютный тик (v<5: low6bits — дельта)
  data chunk:   0 TT SSSSS; type 1=SNAPSHOT (полный), 2=MESSAGE, 3=DELTA; size<30 — в байте,
                30 ⇒ +1 байт, 31 ⇒ +2 байта LE
  payload: Huffman → varint‑int32 (данные дополнены до кратности 4)
```

* SNAPSHOT — сырой `CSnapshot` (data_size, num_items, offsets[], items[key, data…]) раз в ~5 с (keyframe, `demo.cpp:311-322`); DELTA — дельта против *последнего записанного* снапшота в том же формате, что и сетевая (§1.7) (`demo.cpp:323-341`); пустая дельта не пишется, значит «тик без снапшота» = снапшот не изменился.
* MESSAGE — сырое сетевое сообщение (packer‑поток, как payload чанка): у серверных демок — всё, что сервер отправил (кроме `MSGFLAG_NORECORD`, напр. `Sv_PreInput`); у клиентских — всё принятое + локально записанное (`MSGFLAG_RECORD`), например клиент дописывает `Sv_TuneParams` (`gameclient.cpp:2282-2286`).
* Статические размеры объектов и ex‑UUID‑типы — как в сети (демо‑плеер настраивает `SetStaticsize` так же, `src/tools/demo_extract_chat.cpp:248`). Для 0.7‑демок (netversion "0.7") — другие таблицы.

### 4.2 Какие бывают демки и что в них видно

* **Серверные** (`sv_auto_demo_record`, команда `record`): снапшот для `SERVER_DEMO_CLIENT` — **без network clipping, все игроки** (`entity.cpp:97`), полные health/armor/ammo всех (`character.cpp:1143-1149`). Пишутся только на глобальных тиках ⇒ **25 Гц** (`server.cpp:1024-1040`).
* **Клиентские** (ручные, авто‑демки, race‑демки, replay `cl_replays`): только то, что пришло этому клиенту — игроки в радиусе видимости (§1.7), 25 Гц (или 50 Гц у спектатора с high bandwidth); ammo/health только у себя/наблюдаемого.
* **Teehistorian** (не демо!): серверный лог с *точными вводами каждого игрока на каждый тик* (+ join/leave, команды). libtw2 и отдельный крейт `teehistorian` (LGPL‑3.0) умеют его читать. Если удастся получить teehistorian‑файлы — это идеальный источник для имитационного обучения. Официальные сервера DDNet его пишут, но публично не раздают (не проверено, есть ли открытые дампы).

### 4.3 Что можно восстановить per tick из снапшотов

Из `Character` (тип 9, id = client id; поля `network.py:184-215`) + `DDNetCharacter` (ex) + `PlayerInfo`/`ClientInfo`/`DDNetPlayer`:

| Величина | Источник | Точность |
|---|---|---|
| Позиция, скорость | `m_X, m_Y` (px), `m_VelX, m_VelY` (×256) на тике `m_Tick` (dead reckoning!) | Точно на `m_Tick`; на тике снапшота — прогоном физики «без ввода» от `m_Tick` (сервер гарантирует совпадение, иначе переслал бы ядро; см. §1.7). Без физики — только в тиках пересылки ядра. |
| Direction (←/→) | `Character.m_Direction` = `m_Input.m_Direction` на тике снапшота | Точно (на чётных тиках). |
| Прицел | `DDNetCharacter.m_TargetX/Y` = `m_Core.m_Input.m_TargetX/Y` (`character.cpp:1372-1373`) — **точный вектор прицела**; только у новых серверов (поле добавлено в конец, у старых отсутствует); иначе `m_Angle` (угол×256, квантован) | Точно / ≈ |
| Jump | нет поля ввода. Бит0 `m_Jumped` ставится при *успешном* прыжке и держится, пока кнопка зажата; бит1 — использован air‑jump; `DDNetCharacter.m_JumpedTotal`, `m_Jumps`; скачок `VelY` на −импульс | Нажатия, давшие прыжок, — да; «пустые» нажатия и точная длительность удержания в воздухе без прыжков — нет |
| Hook | `m_HookState` (−1 RETRACTED, 0 IDLE, 1..3 RETRACT, 4 FLYING, 5 GRABBED), `m_HookTick`, `m_HookX/Y`, `m_HookedPlayer` | Удержание крюка ≈ `m_HookState != HOOK_IDLE` (при отпускании ядро сразу ставит IDLE, `gamecore.cpp`); погрешность ~1 тик, плюс 25 Гц |
| Fire | `m_AttackTick` (тик последнего выстрела/удара) + события/снаряды (`DDNetProjectile.m_Owner`, `DDNetLaser.m_Owner`, `HammerHit`) | Выстрелы — да; нажатия без выстрела (перезарядка, фриз) — нет; удержание для автоматического оружия — по серии выстрелов |
| Оружие | `m_Weapon` (активное), `DDNetCharacter.m_Flags` (какое оружие есть), | WantedWeapon ≈ смена `m_Weapon` |
| Фриз/состояния | `DDNetCharacter.m_FreezeEnd/m_FreezeStart/m_Flags` (IN_FREEZE, SOLO, JETPACK, …) | Точно |
| Смерти/килл | `Sv_KillMsg` (сообщения в демке) + событие `Death` | Точно (атрибуция убийцы зависит от мода) |
| Тюнинг | `Sv_TuneParams` в сообщениях демки (для клиентских — записанный клиентом) | Точно для записывавшего игрока; tune‑зоны других — через карту (`Info.settings` → `tune_zone …`) и `m_TuneZoneOverride` |

Ограничения:

1. **25 Гц**: вводы на нечётных тиках не наблюдаются. Восстанавливаются перебором: для каждого пропущенного тика ищем ввод (dir∈{−1,0,1} × jump × hook, прицел — интерполяция), при котором детерминированная физика DDNet переводит состояние t в t+2 (нужен наш порт gamecore, он всё равно нужен боту). Неоднозначность остаётся, когда ввод не влияет на физику (например, jump без доступных прыжков).
2. Dead reckoning: без физики позиция «между пересылками ядра» неизвестна; с физикой — точна (ядро пересылается при любом расхождении, а также раз в 3 с).
3. Клиентские демки: только ближайшие игроки; остальные «исчезают».
4. Точные вводы (включая «холостые» нажатия и прицел на старых серверах) есть только в teehistorian или через `Sv_PreInput` в живой игре.
5. Id игроков: в клиентских демках старых клиентов (<19000) — «маппированные» id (до 64); в 20.x‑демках — до 128.
6. Смена карты внутри демки не бывает (одна карта на файл); карта встроена — можно читать её тем же парсером карт.

### 4.4 Что нужно для парсера

Минимально: huffman + varint + snapshot delta/ex‑UUID (всё равно нужно для сети) + разбор заголовка/чанков (~300–500 строк) + таблицы объектов 0.6/DDNet. Поддержка libtw2‑demo — см. §2.
## 5. Мастер‑сервер и авто‑выбор живого Block‑сервера

### 5.1 URL и поведение клиента

* Список по умолчанию (`src/engine/client/serverbrowser_http.cpp:543-548`): `https://master1.ddnet.org/ddnet/15/servers.json` … `master4`. Все 4 отвечают 200, ~1.3 МБ, за Cloudflare, `cache-control: max-age=1` (проверено fork A 2026‑09‑27 12:09 UTC; копии: `proto-scratch/servers*.json`). Клиент выбирает мастер HEAD+GET по скорости/свежести; список переопределяется `ddnet-serverlist-urls.cfg`.
* Регистрация серверов: `POST https://master1.ddnet.org/ddnet/15/register` (`config_variables.h:475`); мастер — `src/mastersrv` (Rust), отдаёт JSON «как зарегистрировал сервер» (`src/mastersrv/src/main.rs:175-210`).
* Дополнительно `https://info.ddnet.org/info` (`DDNET_INFO_URL`): официальные сервера по странам с категориями, **есть категория "Block"** (официальные RUS/EUR/USA блок‑сервера), `map-download-url`, `version`, `stun-servers-*` (копия `proto-scratch/ddnet-info.json`).

### 5.2 Схема servers.json (проверено на копии: 1368 серверов)

```jsonc
{
  "communities": [ { "id", "name", "has_finishes", "icon": {"sha256","url"}, "contact_urls": [] } ],
  "servers": [ {
    "addresses": ["tw-0.6+udp://1.2.3.4:8303", "tw-0.7+udp://1.2.3.4:8303"],  // 1363/1368 имеют tw-0.6
    "location": "eu:ru",            // continent[:country], есть у всех
    "community": "ddnet",           // опционально (766)
    "info": {
      "max_clients", "max_players", "passworded", "game_type", "name", "version",   // обязательные
      "map": { "name", "sha256"?, "size"?, "url"?, "tw_crc"? },
      "client_score_kind"? ("time"|"points"), "requires_login"?, "country"?, "flags"?, "flag"?, "identity_key"?,
      "clients": [ { "name", "clan", "country", "score", "is_player", "afk"?, "team"?, "skin"? } ]
    }
  } ]
}
```

Генерация `info` — `CServer::UpdateRegisterServerInfo` (`server.cpp:2820-2940`) + `CGameContext::OnUpdatePlayerServerInfo` (skin, afk, team; `gamecontext.cpp:5310-5370`). Клиент требует обязательные поля (`serverinfo.cpp:62-90`, по fork A) и предпочитает 0.6‑адрес (`serverbrowser_http.cpp:487-520`).

### 5.3 Снимок Block‑серверов (2026‑09‑27)

Фильтр `/block|blmap|love box|copy the box/i` по name/game_type/map: 70 серверов (66 — по «block» в name/game_type), все с 0.6‑адресом; игроки сосредоточены в eu:ru. Моды разные — это важно для совместимости клиента (версия/тип, взвешено по игрокам):

| игроков | version | game_type |
|---|---|---|
| 219 | 0.6, 20.2 … | Block |
| 118 | 0.6, 20.2 … | 0XF |
| 88 | 0.6.4, 19.7 | GameUp |
| 81 | 0.6* / 0.7 | S-DDRaceX |
| 80 | 0.6, 20.2 … | Block |
| 26+25 | 0.6.4, 19.6/19.7 | BW (Block Worlds) |
| 23+12 | 0.6/0.7, 26.8.7 | F-DDrace / M-DDrace |
| 18 | 0.6.4, 18.5 | DDFightNet fng |

Карты: Copy Love Box (321 игрок), Copy The Box TF (108), Copy Love Box 2s (81), Copy Love Box Mega (80), Unstable (32)… Большинства нет на maps.ddnet.org ⇒ in‑protocol загрузка (§1.6).

### 5.4 Что нужно для «auto‑pick live block server»

1. GET servers.json с master1..4 (таймаут ~5–8 с, по очереди/параллельно; кэшировать ≥30 с — список обновляется мастерами постоянно, Cloudflare кэширует на 1 с).
2. Фильтр: есть `tw-0.6+udp://` адрес; `passworded == false`; `requires_login != true`; признак блока: `game_type`/`name`/`map.name` по regex (как в TS: `serverPick.ts:76`) и/или адрес ∈ категории "Block" из info.ddnet.org.
3. Живость: `clients.filter(is_player && !afk).len() ≥ N`; свободно `clients.len() < max_clients − 1` (резервные слоты: `max_players` ≤ `max_clients`).
4. Приоритет: регион (`location`), пинг (опц. connless `gie3`‑запрос или просто UDP RTT при подключении), число игроков, avoid‑list (кик/бан/ошибки) с TTL.
5. Для карты: `info.map.sha256/size` позволяет заранее проверить кэш и попробовать HTTPS (`info.map.url` или maps.ddnet.org), иначе in‑protocol.
6. Учитывать `sv_connlimit` (5 подключений/20 с с IP) и возможные кики «bot»/антибот — не долбить один сервер.

Всё это — ~150–250 строк Rust (reqwest/ureq + serde_json). Готовый крейт не нужен.
## 6. Рекомендации по компонентам: брать / форкать / писать

Оценки — для одного разработчика, знакомого с Rust, «чистое» время с тестами.

| # | Компонент | Решение | Почему | Трудоёмкость | Риски | Лицензия |
|---|---|---|---|---|---|---|
| 1 | Huffman + varint/packer | **Брать** `libtw2-huffman`, `libtw2-packer` (pin git rev или `pre-rfc3243-libtw2-* 0.2.0`) | Сверены с C++‑эталоном, 34 теста, стабильный формат | 0.5 дня | Минимальные; странный префикс имён на crates.io может смениться — пиновать | MIT/Apache ✔ |
| 2 | UDP‑соединение (пакеты, TKEN, ack/resend, keepalive) | **Форк (vendoring)** `libtw2-net` — только `Connection` (sans‑IO) + своя обёртка на tokio/std | 123 теста, TKEN‑путь совпадает с `network_conn.cpp`, проверено на живых серверах; нужно: Result вместо panic в не‑online, таймаут приёма, лимит очереди | 2–3 дня (обёртка+правки) | bus factor 1 у апстрима; правки держим у себя | MIT/Apache ✔ |
| 3 | Клиентская сессия (CLIENTVER→INFO→MAP→READY→StartInfo→ENTERGAME, INPUT/PredTick по INPUTTIMING, PING/PINGEX, WHATIS, REDIRECT/RECONNECT, MAP_CHANGE посреди игры, `Cl_IsDDNetLegacy`, `/timeout`, обнаружение капча‑лобби/банов) | **Писать своё** | Готового нет ни в libtw2 (downloader устарел и виснет на REDIRECT), ни в crates; ddnet-rs legacy-proxy — как справка (MIT) | 1.5–2 недели | Точность PredTick (TS‑бот её не делал), анти‑бот меры, поведение модов | — |
| 4 | Сообщения и snap‑объекты DDNet | **Форк** `libtw2-gamenet-ddnet`: регенерация по `datasrc/network.py` 20.2 (починить генератор: `NetTickStrict`, `FinishTime::*`), `MapDetails{size,url}`, мягкий `Sv_TuneParams`, толерантность к неизвестным UUID | Типобезопасный кодек ~100 сообщений/объектов; demo‑ридер libtw2 на нём завязан | 2–4 дня + ~0.5 дня на каждое обновление DDNet | Дрейф спеки каждые 1–3 мес.; `unsafe` transmute в encode; альтернатива — свой генератор из `network.py` (3–5 дней) | MIT/Apache ✔ |
| 5 | Снапшоты (сборка частей, дельта, CRC, ex‑UUID, storage) | **Брать** `libtw2-snapshot` | Полная реализация, проверена на живых 19.7/20.0 | 1–2 дня интеграции | Низкие | MIT/Apache ✔ |
| 6 | Карты: datafile + game/front/tele/speedup/switch/tune (+ `Info.settings` для tune‑зон/`sv_*`) | **Форк** `libtw2-datafile` + `libtw2-map` (патч «несколько game‑слоёв → последний», добавить Info.settings, при желании flate2 вместо libz-sys); twmap — **только dev‑оракул** в тестах на корпусе ddnet-maps | libtw2‑map прочитал 2423/2428 реальных карт со всеми 6 слоями, MIT; twmap функционально хорош, но AGPL §13 на весь бинарник + тяжёлые зависимости (image/ndarray/serde) | 1–2 дня (форк) / 3–5 дней (своё с нуля, если решим избавиться от `unsafe`) | Карты с нестандартной структурой (legacy v2 tilemap, tileskip v4 — у DDNet обработаны в `map.cpp:500-700`) — гонять корпус | MIT/Apache ✔ (twmap AGPL — не линковать) |
| 7 | HTTPS‑загрузка карты | **Своё** (reqwest/ureq + sha2), fallback in‑protocol с окном `sv_map_window` | 50–100 строк; большинство блок‑карт всё равно 404 на maps.ddnet.org | 0.5 дня | — | MIT/Apache ✔ |
| 8 | Демо (.demo v3–v6, DDNet ex) | **Брать** `libtw2-demo` (с форкнутым gamenet) + **своё** восстановление траекторий/вводов поверх нашего порта физики | Ридер покрывает формат, проверен на 5 реальных v6‑демках; извлечение вводов — задача физики, не формата | 1–2 дня ридер; 1–2 недели реконструкция вводов (после порта gamecore) | 25 Гц, dead reckoning, клиентские демки видят не всех, нет «холостых» нажатий; `binrw 0.11` future‑incompat (можно заменить при форке) | MIT/Apache ✔ |
| 9 | Мастер‑сервер / авто‑выбор | **Своё** (serde_json) | Схема простая (§5), в libtw2 нет HTTP‑клиента | 0.5–1 день | Смена схемы маловероятна (путь `/ddnet/15/` стабилен; с какого года — не проверено); эвристика «block» по имени | — |
| 10 | (справка) Физика DDNet | вне рамок этого отчёта; есть `twgame` (AGPL‑3.0‑only, fork A) и порт в TS‑боте | та же AGPL‑дилемма, что и twmap | — | — | AGPL ⚠ |

Итого на сетевой стек + карты + демо‑ридер + мастер: **~3–4 недели** до паритета с TS‑ботом, из них основная часть — собственная клиентская сессия (#3) и обновлённый gamenet (#4).

### 6.1 Лицензионные выводы

* libtw2, ddnet-rs, npm `teeworlds`, flate2/miniz_oxide, reqwest, tokio, serde — MIT/Apache: совместимы с GPL‑3.0‑only, требуют только сохранения уведомлений.
* DDNet C++ — zlib‑подобная лицензия (`ref/ddnet/license.txt`): переносить код/таблицы можно, с пометкой «altered source» и сохранением уведомления; ассеты `data/` — CC‑BY‑SA 3.0 (нам не нужны).
* twmap, twgame, twsnap, twstorage, ddnet_protocol (C, MilkeeyCat) — **AGPL‑3.0‑only**: совместимы (GPL‑3.0 §13 / AGPL §13 — оба ссылаются на «version 3»), но AGPL §13 (предоставить исходник всем, кто взаимодействует по сети) распространяется на комбинацию целиком. Рекомендация — не линковать в распространяемый бинарник; как dev‑инструменты — можно.
* crate `teehistorian` — LGPL‑3.0: совместим.

### 6.2 Главные риски (не решаются выбором библиотек)

1. **Репутация IP**: популярные блок‑сервера и официальные DDNet банят дата‑центровые/VPN IP («bad ip», «VPN detected») — проверено вживую. Бот должен работать с «жилого» IP пользователя.
2. **Анти‑бот**: капча‑лобби через REDIRECT (TeeFusion → `Captcha_CAP`), `NETMSG_CHECKSUM_REQUEST` от модов, закрытый antibot на официальных серверах (REDIRECT/RECONNECT сразу после INFO). Нужно: корректно обрабатывать REDIRECT/RECONNECT, детектировать капча‑карты/кики и помечать сервер в avoid‑list.
3. **Лимиты 20.x**: `sv_connlimit` 5/20 с на IP, лимит resend, бюджет декомпрессии — реконнект‑циклы приведут к банам.
4. **Дрейф протокола**: новые UUID‑сообщения/объекты и хвостовые поля каждые 1–3 месяца; разнообразие модов (DDNet 18.5…20.2, F‑DDrace 26.8.7, S‑DDRaceX, BW). Нужны тесты‑записи трафика с каждого целевого сервера.
5. **Тайминг ввода**: без синхронизации PredTick по INPUTTIMING ввод приходит «не в тот тик» (TS‑бот использовал `setInterval(20ms)`).

## 7. Заметки о текущем TS‑боте (что учесть при переписывании)

По fork C (`~/aiddnet/DDNet-AI`, не изменялся):

* npm `teeworlds` 2.6.1 (MIT, github swarfeya/teeworlds-library-ts): только 0.6+DDNet; TKEN‑рукопожатие (`lib/client.js:390, 433-435`); CLIENTVER — бот объявляет 19000/"DDNet 19.0" (`src/bot/bot.ts:548-549`); шлёт нестандартное `i-am-npm-package@swarfey.gitlab.io` (в Rust не повторять); после CON_READY — rcon `crashmeplx`.
* INPUT по таймеру 50 мс + на каждом кадре, PredTick растёт `setInterval(20ms)`, **INPUTTIMING игнорируется**.
* Снапшоты: SNAP/SINGLE/EMPTY, 21 тип 0.6 (`lib/snapshot.js:8-28`), ex‑объекты частично; бот доразбирает сам (`src/demo/snapshot.ts`, `liveWorld.ts`).
* Карта: in‑protocol по одному чанку без окна, проверка только CRC; HTTPS через MAP_DETAILS url, SHA256 закомментирован.
* **`Sv_TuneParams` не разбирается** — бот использует фиксированный тюнинг (`src/core/tuning.ts`) ⇒ ошибка на серверах с нестандартным тюнингом/tune‑зонами. В Rust — обязательно разбирать.
* `netPatch.ts`: безопасный Huffman (лимит 64 КиБ, EOF), глушение ошибок декодера, **обработка REDIRECT** (`patchRedirect`, L114-139, реконнект `bot.ts:999-1007`), O(n) снапшот‑декодер вместо O(n²).
* Баг библиотеки: `ping@ddnet.tw` зарегистрирован с неверным id ⇒ PINGEX не обрабатывается (безвредно).
* Выбор сервера: `src/bot/serverPick.ts` — master1..4, приоритет `tw-0.6`, regex `/block|blmap|copy (love|the) box|love box/i`, ≥2 игроков, не полный, не запароленный, бонус `:ru`, avoid‑list с TTL.
* Карты: собственный `src/map/datafile.ts` (v3/v4, zlib) + `loadMap.ts` (game с tileskip, front, tele, speedup, `sv_no_weak_hook` из Info.settings); switch/tune **не читаются**.
* Парсера `.demo` нет; формат датасета `DDAIHUM1` ссылается на демки, генератор в дереве отсутствует (не проверено, где он).

## 8. Артефакты исследования (`~/aiddnet/data/research/proto-scratch/`)

* `servers.json`, `servers_m1..4.json`, `ddnet-info.json` — снимки мастеров/info 2026‑09‑27.
* `libtw2/` — рабочая копия со сборкой; `build.log`, `test.log`.
* `netprobe/` — минимальный живой клиент на libtw2 (`netprobe <ip:port>`; ≤20 с, 60 снапшотов).
* `demos/` (5 DDNet v6 демок из тестов twgame), `demotest/` — чтение через libtw2‑demo.
* `maptest/` — прогон libtw2‑map по корпусу; `twmap-probe/`, `maps/` — прогон twmap по 64 картам.
* `ddnet-maps/` — весь репозиторий карт DDNet (~2 ГБ), выкачан для корпуса — можно удалить.
* `legacy-proxy-lib.rs` — исходник ddnet-rs legacy-proxy (справка), `pr12018.diff` — PR «128 players».
* `sec*.md` — черновики разделов этого документа.
