# DDNet-AI: веб-UI, Electron-оболочка и ассеты DDNet — заметки для порта на Rust

Фаза 0, исследование. Репозиторий: `~/aiddnet/DDNet-AI` (только чтение). DDNet: `~/aiddnet/ref/ddnet` (shallow master, коммит 9576fd6, 2026-09-27).
Дата: 2026-09-27. Всё, что не проверено руками, помечено «не проверено».
Пути ниже относительны корню DDNet-AI, если не сказано иное. Формат ссылок `файл:строка`.

---

## 0. Коротко

- Сервер страницы: голый `node:http`, **только 127.0.0.1**, порт 7777 по умолчанию (`start.mjs:433`, флаг `--web-port`), **без авторизации**; защита только проверкой `Host` и `Origin`/`Sec-Fetch-Site` (`src/bot/web.ts:314-331`). TLS нет.
- Транспорт: **только HTTP-поллинг**, WebSocket/SSE нет. Горячий путь: `GET /api/live` каждые 40 мс (25 Гц) и `GET /api` раз в секунду (`src/bot/page/page.js:919`, `:665`).
- Рисование: **целиком на клиенте**, Canvas 2D. Сервер отдаёт сырые данные: разобранную карту (JSON + бинарные слои + PNG встроенных картинок), кадры с позициями ти и ассеты DDNet. Код рисования написан на TS (`webView.ts`, `webDraw.ts`) и **сериализуется в страницу через `Function.prototype.toString()`** (`src/bot/webPage.ts:38-76`).
- Ассеты DDNet не лежат в репозитории: ищутся в установке DDNet, иначе качаются с `raw.githubusercontent.com/ddnet/ddnet/20.0/data/` (`src/bot/webAssets.ts:108`) в `runs/ddnet-data`, скины с `skins.ddnet.org` в `runs/skincache`. Звуки `.wv` декодируются на сервере в WAV через wasm-wavpack (BSD-3).
- Electron-оболочка (`app/`) даёт то, чего нет на странице: браузер серверов, первый запуск/стартовый экран с историей, пауза/перезапуск, лог процесса, отчёт-zip, трей/мини-режим/горячая клавиша.
- Лицензии: данные DDNet (кроме шрифтов/скинов/assets/languages) **CC-BY-SA 3.0**, скины по-разному (zlib, CC-BY, CC-BY-SA, CC0; в базе skins.ddnet.org ещё NC и «unknown»), DejaVuSans под лицензией Bitstream Vera/Arev. Скачивать в рантайме с официального источника и показывать владельцу в его браузере можно; нужна страница с атрибуцией; в git ассеты не класть.
- **Важно для всего проекта:** `NOTICE` исходного DDNet-AI содержит дополнительное условие GPLv3 §7(b): сохранять NOTICE и атрибуцию «DDNet AI / Wranked1», изменённые версии помечать как изменённые (`NOTICE:1-14`). Порт на Rust это производная работа (не проверено юридически, но разумно считать так).

---

## 1. Веб-сервер

### 1.1 Как поднимается
- `startWebUi(bot, port, version)` (`src/bot/web.ts:199`) вызывается из `start.mjs:431-433`: порт `flags["web-port"] ?? 7777`. Флаги `--no-web` (`start.mjs:429`), `--no-open` (не открывать браузер, `:455-462`), `--ready-line` (печатать `WEBUI_READY <port>` / `WEBUI_FAIL ...` для Electron, `:452`, `:465`).
- `http.createServer` (`web.ts:333`), `server.listen(port, "127.0.0.1")` (`web.ts:731`). IPv6 и внешние интерфейсы не слушаются.
- Авторизации нет. Фильтр `allowed()` (`web.ts:319-331`):
  - `Host` обязан быть ровно `127.0.0.1:<port>` или `localhost:<port>` (`ownHost`, `web.ts:314-318`). Это защита от DNS-rebinding.
  - `GET`/`HEAD` пропускаются всегда.
  - Остальные методы: отказ при `Sec-Fetch-Site: cross-site`; если есть `Origin`, он должен быть `http://127.0.0.1:<port>`/`http://localhost:<port>`; **без `Origin` пропускается** (curl, любой локальный процесс).
  - Отказ: 403 «чужой запрос» (`web.ts:334-337`).
- Ошибки обработчика отдаются текстом `err.message` с кодом 500 (`web.ts:342-345`).
- HTML страницы собирается один раз на язык: `index.html` + `<style>page.css</style>` + один inline `<script>`: `LANG`, словарь `EN`, `NEWS` (whatsnew.json), исходник `makeT`, `pageScript()` (сериализованные функции рисования) и `page.js` (`web.ts:768-789`). Любой неизвестный путь отдаёт эту страницу (`web.ts:723-725`), так работают `/stream` и `/?view=mini`.

### 1.2 Маршруты (все в `web.ts:348-726`)

| Метод, путь | Что отдаёт/делает | Строки |
|---|---|---|
| GET `/api` | `{status: BotStatus, stats: string, version, boot, lines: BotLine&{seq}[] (последние 120)}` | 350-354 |
| GET `/api/map` | `LiveMap`: `{name, width, height, kinds: base64(u8[w*h]), traps?: base64(u8[w*h])}` или `null` | 356-361; сборка `bot.ts:2638-2669` |
| GET `/api/live` | `LiveFrame` + `emoticons` + `sounds` или `null` | 363-368; сборка `bot.ts:2671-2771` |
| GET `/api/scene?m=<map>` | JSON сцены карты (группы, слои, картинки, огибающие); 409 если `m` не совпал с текущей картой | 370-384 |
| GET `/api/scene/tiles/<layerId>?m=` | бинарь слоя (`application/octet-stream`) | 385-395 |
| GET `/api/scene/image/<i>?m=&g=` | PNG встроенной в карту картинки | 396-403 |
| GET `/skins/<name>.png` | скин: локально или скачать с skins.ddnet.org (кэш `max-age=3600`) | 412-440 |
| GET `/overlay` | отдельная HTML-страница счёта дуэли для OBS (поллит `/api/duelnow` каждые 500 мс) | 168-195, 442-446 |
| GET `/api/duelnow` | `{me, now: {name,ours,theirs}|null, last: {opponent,ours,theirs,at,by}|null}` | 447-458 |
| GET `/api/duels` | `{n, ours, theirs, list: DuelRow[≤50]}` | 460-467 |
| GET `/api/clips` | `[{name,size,when}]` (≤200, `bot.ts:3255-3269`) | 469-473 |
| GET `/clips/<name>` | файл клипа JSON как attachment; имя проверяется `^[\w.-]+\.json$` (`bot.ts:3271-3276`) | 474-491 |
| GET `/api/knobs` | `[{key,value,def,changed}]` параметры планировщика | 493-497 |
| POST `/api/knobs` | `{key,value}` или `{reset:true}` → `{reply}` | 498-513 |
| GET `/api/launch` | содержимое `settings.json` **без** `password` и `llm` + `ddnetDataNote`, `ddnetDataFound`, `ddnetGraphics`, `ddnetFetching`; побочный эффект: запускает докачку графики | 542-556 |
| POST `/api/launch` | пишет в `settings.json` строки `server,name,clan,skin,ddnetData,skinDownload,dummy,dummyName`; при смене сервера стирает пароль | 515-541 |
| GET `/assets/<rel>` | файл данных DDNet (`.png .wav .ogg .mp3 .ttf .otf`); `audio/*.wav` декодируется из `.wv`; для шрифтов `Access-Control-Allow-Origin: *` | 559-601 |
| POST `/api/update` | проверить обновление (git-автообновление бота) | 602-615 |
| GET `/api/votes` | список голосований сервера `string[]` | 616-620 |
| GET `/api/relations` | `{war,friend,ignore,clanWar,clanFriend,partner: string[]}` (`bot.ts:3628`) | 622-626 |
| POST `/api/relation` | `{list:"war"|"friend"|"ignore", name(≤64), on}` → `{reply, lists}` | 627-644 |
| GET/POST `/api/llm` | `{off,url,model,hasKey,presets}` / сохранить `{url,model,key}` или `null` (тело ≤4000) | 645-668; `bot.ts:3492-3517` |
| GET/POST `/api/autochat` | конфиг автосообщений (тело ≤20000) | 669-691 |
| GET `/api/commands` | имена команд (`bot.ts:3278-3283`) | 692-696 |
| GET `/api/config` | `{map, planner, memory:{events,map}|null, traps}` (`bot.ts:3335-3349`) | 697-701 |
| POST `/cmd` | `{line}` → `bot.handleConsole(line)` → `{reply}`; строку и ответ кладёт в лог | 702-721 |
| прочее | HTML страницы (`?lang=ru|en`) | 723-725 |

### 1.3 Модель данных

`BotLine = {kind: "log"|"chat"|"event"|"whisper", text, from?, sys?}` (`src/bot/bot.ts:368`); на сервере добавляется `seq`, буфер 200 строк (`web.ts:164`, `:309-312`), в `/api` уходят последние 120. `boot = "<pid>-<time36>"` (`web.ts:203`) нужен странице, чтобы понять перезапуск бота.

`BotStatus` (`bot.ts:449-481`): `phase ("offline"|"connecting"|"online")`, `acting`, `brain ("planner"|"net"|"scripted")`, `mode ("fight"|"passive"|"hold"|"goto")`, `server`, `name`, `targetName`, `targetDist`, `walk`, `offlineReason`, `wb?`, `panel? {wbMode, duelMode, inDuel, spectating, pinnedTarget, home, tryName, duelScore?}`, `frozen`, `tick`, `stats: BotStats`, `lowCpu?`, `owner?`, `llm?`, `strong?`, `lag? {hint, workMs, skipped, behindMs}` (`src/bot/cpuLoad.ts:43-52`). При включённом втором боте добавляется `dummy: {name, phase, frozen, acting, mode, wb, target, id, duelScore}` (`start.mjs:387-390`).

`stats` в `/api` это **строка** `statsLine()` вида `ticks=.. brain=.. weapon=.. try=.. kills=.. deaths=.. selfKills=.. clips=.. hammerFires=.. hooksFired=.. blocks=.. blockedBy=.. disconnects=.. errors=..` (`bot.ts:1074-1080`); страница парсит её регэкспом (`page.js:462`). В Rust отдавать структурой.

`LiveFrame` (`bot.ts:428-447`):
- `tick, selfId, target, map, mapKey ("<map>#<crc32 hex>")`
- `tees: LiveTee[]` только живые (`bot.ts:2680`). Поля `LiveTee` (`bot.ts:382-411`, заполнение `bot.ts:2686-2716`): `id, name, x, y (px, округлены), frozen, hook (hookState), hx, hy, hooked (id или -1), clan, skin, cc (custom color), cb, cf (цвета HSL packed), aim (рад, 2 знака), wp (оружие 0..5), emote, vx, vy (3 знака), dir, jumped (биты), atk (тиков с атаки), fz (freezeTicksLeft), pf (player_flags), jl (jumpsLeft), fzf (frozenFor), deep, xf (DDNetCharacter m_Flags), jt (m_Jumps)`.
- `players: LivePlayer[]` (`bot.ts:413-425`): `id, name, clan, score, ping, team, skin, cc, cb, cf`.
- `doing` (локализованная строка «что делает»), `goal {x,y}|null`, `route [{x,y,kind}]` (≤40 шагов, клетки), `cursor {x,y}` (прицел, ≤400 px), `roundStart`, `timeScore`.
- Дополнительно в `/api/live`: `emoticons [{id,e,age}]` (эмоции за 2.5 с, `web.ts:246-255`) и `sounds [{s,id,x,y}]` (события `sound_world` за 1 с, id 0..64, ≤64 шт., `web.ts:209-244`).

Размер кадра (оценка синтетикой, не замер): 8 ти ≈ 5 КБ, 32 ти ≈ 16 КБ, 64 ти ≈ 30 КБ JSON. При 25 Гц это **~125 / ~390 / ~740 КиБ/с** без сжатия; сжатия сервер не делает. Для телефона на мобильной сети это много.

`/api/map`: `kinds` 1 байт на клетку: 0 пусто, 1 solid (хук цепляется), 2 freeze, 3 death, 4 unfreeze, 5 nohook solid, 6 tele (`bot.ts:2638-2669`); `traps` это «мёртвые клетки» (карманы без выхода). Для карты 500×300 это ~200 КБ base64; грузится при смене `mapKey` (`page.js:901`, `:706-712`).

`/api/scene` (формат `src/bot/webMap.ts:38-65`):
- `groups[]: {ox, oy, px, py (параллакс, %), clip [x,y,w,h]|null, layers[]}`
- слой тайлов: `{kind:"tiles", id (индекс слоя в файле), w, h, color [r,g,b,a], image (-1 для служебных), detail, role: "visual"|"game"|"front"|"tele"|"speedup"|"switch", env, envOff}`
- слой квадов: `{kind:"quads", id, image, detail, quads: number[]}`, по 38 чисел на квад (`QUAD_REC`, `webMap.ts:36`): 4 точки (8), 16 цветов (4 угла RGBA), 8 текстурных координат, центр (2), posEnv, posEnvOff, colorEnv, colorEnvOff (`webMap.ts:379-389`).
- `images[]: {name, w, h, external}`; `envelopes[]: {c: каналов, p: плоский массив по 6 на точку [timeMs, curve, v0..v3 /1024]}` (`webMap.ts:471-497`).
- Тайлы `/api/scene/tiles/<id>`: для visual/game/front 2 байта на клетку `(index, flags)` без сжатия (`webMap.ts:84-103`); для tele/speedup/switch упакованный список непустых клеток: varint-разрыв + поля (tele 2: type, number; switch 4: type, flags, number, delay; speedup 5: type, force, maxSpeed, angle lo, hi) (`webMap.ts:114-156`, распаковка `webDraw.ts:363-409`). Берётся только **последний** слой каждого особого типа (`webMap.ts:275-288`, `:321`), tune-слой пропускается (`:338`).
- Встроенные картинки кодируются в PNG на сервере (RGB→RGBA, deflate level 1), кэш `runs/scene-cache/<crc32>-<len>/<i>.png`, хранится 12 карт (`webMap.ts:397-440`, `:527-558`).
- Откуда байты карты: `client.map.mapBuffer`, иначе `maps/<name>.map`, иначе `downloadedmaps/<name>_<crc|sha256>.map` в папке пользователя DDNet (`web.ts:258-274`, `webAssets.ts:381-406`).

### 1.4 Частоты опроса на странице
- `/api/live` каждые 40 мс, не чаще одного запроса в полёте; в `lowCpu` не чаще раза в 120 мс; при `document.hidden` не опрашивает (`page.js:887-906`, `:919`).
- `/api` раз в 1000 мс (`page.js:665`) плюс сразу после каждой команды (`page.js:135`).
- `/api/relations` раз в 5000 мс (`page.js:876`).
- `/api/duelnow` раз в 500 мс только в `/overlay` и `/stream` (`web.ts:194`, `page.js:1138`).
- Остальное по действиям: вкладка настроек тянет `/api/config`, `/api/knobs`, `/api/launch`, `/api/autochat`, `/api/llm` (`page.js:86`, `:1009`); вкладка записей `/api/clips`, `/api/duels`.
- Отрисовка через `requestAnimationFrame`, ограничение FPS 30/60/без ограничения, в `localStorage["ddai.fps"]` (`page.js:908-919`, `:1119`).

### 1.5 Что страница может приказать боту
Всё идёт через `POST /cmd {line}` (`page.js:131-137`) плюс несколько JSON-эндпоинтов:
- Стиль игры: `!style default|wb|duel` (`page.js:144`); сторона ВБ `!wb auto|left|right` (`:143`); режим `!go`, `!mode passive`, `!stop` (`:141-142`).
- Цель: `!target <ник>`, `!target -` (`:145`, `:871`); идти к игроку `!goto @<ник>` (`:872`); отмена похода `!stop`.
- Кнопки: `!kill`, `!clip`, `!spec`/`!join`, `!home`/`!home off`, `!emote <имя>` (`index.html:80-84`, `page.js:138`, `:147-149`).
- `!low on|off`, `!strong on|off` (`page.js:151-153`); второй бот `!d wb left|wb right|stop|go` (`page.js:1121`).
- Голосования: `!yes`/`!no` (F3/F4, `page.js:581`), `!vote <опция>` из `/api/votes` (`page.js:241-251`).
- Хозяин и LLM: `!owner <ник>|off`, `!llm on|off` (`page.js:155-162`), `POST /api/llm`.
- Язык: `!lang ru|en` и перезагрузка (`page.js:110`).
- Свободная строка: команды `!...` или текст в чат от имени бота (`page.js:662-664`, чат-оверлей `:592-610`).
- `POST /api/relation` (тима/вар/игнор), `POST /api/knobs`, `POST /api/autochat`, `POST /api/launch`, `POST /api/update`.
- Список команд для автодополнения: `/api/commands` (`page.js:527`).

---

## 2. Отрисовка

### 2.1 Разделение кода
- `src/bot/webDraw.ts`: чистые функции и таблицы, портированные из DDNet (zlib, заголовок `webDraw.ts:1-8`): сетка спрайтов `SPRITES` (`:10-77`), флаги `CHARFLAG` (`:79-99`), анимации `ANIMS` (`:101-135`) и их смешивание (`animSeq`, `animState`, `teeAnimFor`, `:137-193`), цвета HSL DDNet (`ddnetColor`, `:195-234`), перекраска скина (`skinColorable`, `skinTint`, `:236-288`), матрицы флагов тайла (`tileMatrix`, `:301-313`), камера с параллаксом (`groupView`, `:333-355`, как в `engine/graphics.cpp`), LOD чанков (`chunkLod`, `:357-361`), особые тайлы (`specialTiles`, `specialLook`, `overlayNumber`, `:363-419`), часы тиков (`tickClock`, `:421-426`), огибающие (`envEval`, `envRgbConst`, `:428-495`), разбиение слоёв на проходы (`buildPasses`, `:497-531`), метрики табло (`boardMetrics`, `boardColumns`, `boardScore`, `:560-585`), HUD (`hudWeapons`, `jumpIcons`, `freezeBarPieces`, `:587-626`), снежинки (`flakeStep`, `:628-638`).
- `src/bot/webView.ts` (2118 строк): `createView(canvas, opts)` (`:124`), весь рендерер и камера; заголовок про происхождение из DDNet `CRenderTools::RenderTee6, CPlayers, CNamePlates, CScoreboard, CHud, CFreezeBars, CEffects` (`:1-5`).
- `src/bot/webPage.ts`: склеивает `SPRITES/ANIMS/CHARFLAG` в JSON и функции через `f.toString()` в одну строку для страницы (`:38-76`). В браузер попадает скомпилированный Node-ом JS (Node 24 снимает типы). **Для порта: не повторять этот трюк, собрать `webView/webDraw` в обычный ES-модуль (esbuild/tsc) и отдавать статикой, встроенной в бинарь.**
- `src/bot/page/page.js`: DOM, опрос, звук, чат, списки, клипы; создаёт `view = createView(#cv)` и второй `view2` для проигрывателя клипов (`page.js:667-669`, `:318`).
- Сервер ничего не рисует, кроме перекодирования встроенных картинок карты в PNG (`webMap.ts:441-466`).

### 2.2 Карта
- Режимы вида: «карта», «сущности», «вместе» (`webView.ts:173`, кнопка `page.js:676`).
- «Карта»: проходы из `buildPasses`: фоновые до game-слоя и передние после (`webDraw.ts:497-531`). Тайловые слои рисуются чанками в офскрин-канвасы с LOD (`webView.ts:511-615`, `:697-752`), тонирование цветом слоя и постоянными огибающими (`tintEntry`, `:471-503`), анимированный цвет через огибающие (`drawEnvTilePass`, `:754-794`), квады с текстурой через `CanvasPattern` + аффинную матрицу или градиентом (`drawQuads`, `:796-926`), clip-прямоугольники групп (`:928-947`), неподвижный фон кэшируется целиком (`staticPrefix`, `:405-431`, `:1459-1487`). Бюджет постройки чанков 8 мс на кадр (`:1405`), кэши: чанки до 48 Мпикс, тонированные картинки до 40 Мпикс (`:198-199`), это до ~190 + ~160 МБ RGBA (для телефона опасно, см. риски).
- Внешние картинки карты берутся из `/assets/mapres/<name>.png`, встроенные из `/api/scene/image/<i>` (`webView.ts:357`).
- «Сущности»: game/front/tele/speedup/switch слои рисуются атласом `editor/entities_clear/ddnet.png` (`:1499-1505`); speedup со стрелкой `editor/speed_arrow.png` и углом; числа tele/switch (номер, задержка) и speedup (сила, макс. скорость) текстом при масштабе ≥16 px/тайл (`specialTiles2d`, `:628-675`).
- Запасной вид без графики DDNet: закраска по `kinds` цветами CSS `--solid, --freeze, --death, --unfreeze, --nohook, --tele` (`drawKinds`, `:961-993`; цвета `page.css:5`), легенда внизу (`index.html:41-48`) видна только в этом режиме (`page.js:704`).
- Оверлей «ловушки»: клетки `traps` закрашены `--trap` (`:995-1006`).
- Фриз/deep/unfreeze/tele/speedup как отдельные игровые механики отдельно не подсвечиваются в режиме «карта»: их видно через тайлсет карты или атлас сущностей. `deep` у ти рисуется иконкой HUD (`:1839`) и отключает полосу фриза (`:1287`).

### 2.3 Игроки (ти)
- Интерполяция: буфер 12 кадров, время отрисовки `now + clock - 100 мс`, линейно x/y/hx/hy/vx, угол прицела по кратчайшей дуге; скачок >300 px без интерполяции (`webView.ts:1021-1099`).
- Скины: `/skins/<name>.png`, LRU 128→112 (`:241-257`), при ошибке `default`. Свои цвета (`cc/cb/cf`) по алгоритму `CSkins::LoadSkin` DDNet: серый + нормализация + тонирование тела и ног (`teeAtlas`, `:269-318`). Ноги притемнены, если израсходован прыжок в воздухе (`jumped & 2`, `:1546`). Замороженные рисуются скином `x_ninja` (`:270-271`).
- Тело/ноги/контуры/глаза по `RenderTee6` (`renderTee`, `:1130-1156`); глаза по эмоции (`emote`, у замороженного 1 «боль», `:1564`); направление глаз по прицелу.
- Анимации: base/idle/inair/walk/run, взмах молота, ниндзя (`teeAnimFor`, `webDraw.ts:169-193`); «в воздухе» по `kinds` под ногами (`:1559`).
- Оружие из `game.png`: молот/ниндзя с анимацией атаки, пистолет/дробовик/гранатомёт/лазер с отдачей и руками (`renderWeapon`, `:1177-1215`); замороженный оружие не держит.
- Хук: цепь и голова из `game.png`, рука ти, привязка к зацепленному игроку (`renderHook`, `:1217-1254`); без графики линия с кружком.
- Полоса фриза из `hud.png` по `CFreezeBars` (`:1286-1332`), снежинки из `extras.png` у замороженных (`:1356-1398`).
- Эмоции из `emoticons.png` с анимацией появления/затухания 2 с; пузырь «пишет в чат» при `pf & 4` (`:1334-1354`).
- Таблички ников: ник (цель розовым), клан (выключен по умолчанию), стрелки направления/прыжка у чужих из `arrow.png` (`renderPlates`, `:1604-1656`); прячутся при сильном отдалении, вместо ти точки-маркеры (`renderMarkers`, `:1697-1712`).
- Прицел бота: курсор оружия из `game.png` (`renderCursor`, `:1658-1695`).
- Без графики DDNet: круглые ти своим рисованием (`renderPlainTee`, `:1256-1284`), цвет: свой зелёный, цель красная, замороженный синий.

### 2.4 Камера и ввод
- Следить за ботом (по умолчанию), за любым игроком (клик по ти или список `#spec`), свободная камера перетаскиванием (выключает слежение) (`webView.ts:1431-1448`, `:2029-2055`; `page.js:672-689`). Сглаживание `k = 1 - exp(-dt*14)`.
- Масштаб 0.25..40 (`webView.ts:2086-2091`), колесо и ползунок 3..300 (`100/zoom`) (`page.js:670-671`, `:689`). Щипкового зума на телефоне **нет**.
- «Вся карта» (`fit`, `webView.ts:2056-2067`). **Миникарты нет** (не нашёл).
- DPR ограничен 2 (`:1417`).

### 2.5 HUD, табло, оверлеи
- HUD (`renderHud`, `:1770-1882`): оружие, прыжки, способности и запреты по `xf` (эндлесс, джетпак, телепорт-оружие, solo, collision off, hit off, practice, lock, team0, deep, live frozen), текст «FPS · пинг», лента «X заморожен / разморожен» (`:1041-1045`, `:1863-1878`).
- Табло как в DDNet `CScoreboard` (`renderBoard`, `:1884-2021`): карта, счёт/время, колонки, ти-иконки, пинг цветом, зрители. Показ: кнопка «табло» или удержание Tab (`page.js:691-697`).
- DOM-оверлеи поверх канваса: чат (последние 9 строк, затухание 16-17 с, иконки ти) и поле ввода (`index.html:18-21`, `page.js:567-654`), баннер голосования с F3/F4 (`page.js:255-264`).
- Отладочные оверлеи на живом экране: маршрут (пунктир, точки hook/kill/прочее, круг цели) (`webView.ts:1511-1537`), ловушки, прицел. **Поля опасности, кандидатов плана, линий плана на живом экране нет.** В проигрывателе клипов в статусе текстом: «свой фриз N, фриз соперника N, вариантов N» из `plan` кадра (`page.js:342-345`).

### 2.6 Звук
- WebAudio, по умолчанию выключен (`page.js:442-443`). Список файлов по id звуков DDNet (`SND_FILES`, `page.js:714-721`), грузятся `/assets/audio/<name>.wav`, выбор случайного варианта, громкость и панорама по расстоянию до слушателя, радиус 1500 (`page.js:738-765`).
- Источники: события `sound_world` из снапшота (`web.ts:231-243`) и звуки, выведенные на клиенте из разницы кадров: прыжок, воздушный прыжок, хук за стену, хук в nohook (`page.js:776-799`); звуки чата (`page.js:488-503`).

### 2.7 Проигрыватель клипов
Вкладка «Записи»: список `runs/clips/*.json`, проигрывание 25 кадров/с со скоростью 0.25..2, шаг, перемотка, метки заморозок, скачивание (`page.js:276-369`; `index.html:129-155`). Формат клипа (из `clipFrame`, `page.js:300-312`): `{map, width, height, tiles[], selfId, players[{id,name,clan,skin,cc,cb,cf}], frames[{tick, tees[{id,alive,x,y,frozen,hookState,hookX,hookY,hookedPlayer,angle,weapon,vx,vy,direction,jumped,freezeTicksLeft,jumpsLeft}], inputs[{id,targetX,targetY}], plan{target,selfOut,enemyOut,candidates}}]}`.

### 2.8 Режимы страницы
- `/stream` или `?view=stream`: только канвас на весь экран, счёт дуэли сверху, только серверные строки чата, чужие ники заменены номерами (`?nicks=on|off`) (`page.js:1-20`, `:1123-1139`; `page.css:169-179`).
- `?view=mini` или `#mini`: компактный вид для мини-окна Electron, FPS ≤30 (`page.js:22`, `:913`; `page.css:159-167`).
- `/overlay`: отдельный HTML только со счётом (`web.ts:168-195`).
- Внутри iframe (Electron) класс `embedded`: прячутся поля «Сервер/Имя/Клан/Скин», показывается подсказка «настраивается в окне» (`page.js:24-25`, `page.css:333-335`, `index.html:164-175`).

---

## 3. Полный список функций UI

### 3.1 Страница бота (`src/bot/page/index.html`, `page.js`)
Шапка: вкладки «Игра», «Записи», «Настройки», точка статуса и «ник — сервер/причина», версия (клик: «Что нового»), «?» тур (`index.html:4-12`).

Вкладка «Игра»:
- Канвас с камерой и оверлеями (раздел 2). Панель вида: следить, за кем следить, вся карта, масштаб; вид карта/сущности/вместе, ники, прицел, маршрут, ловушки, табло, звук (`index.html:23-39`). Статус-строка «сейчас: …» и «тик · ти · FPS · масштаб» (`:40`, `page.js:699-705`). Выбор сохраняется в `localStorage["ddai.view"]` (`page.js:961-979`).
- «Управление»: Игра дефолт/ВБ/дуэль, сторона ВБ, Режим драться/не лезть/стоять, Цель (закреплённая, сбросить), поход и его отмена, кнопки убиться/клип/наблюдать/дом/эмоция, слабый ПК/сильный режим, второй бот (статус + 4 кнопки), строка ответа (`index.html:51-97`, `page.js:194-239`).
- «Состояние»: чип свободен/во фризе/не в игре, «ПК не успевает» (lag), цель/счёт дуэли, сетка статистики: мозг, оружие, убил, умер, сам /kill, заморозил, заморозили, хуков, хаммеров, клипов, проба (`index.html:98-102`, `page.js:464-483`).
- «Игроки»: список (из `players` или `tees`) с иконкой ти, кланом, бейджами тима/вар/игнор/второй бот, расстоянием в тайлах, фриз; раскрытие: цель, к нему, смотреть, ник в чат, тима/вар/игнор (`page.js:813-875`).
- Лог: фильтры всё/чат/события/личные, поиск, копировать, клик по нику вставляет его (`index.html:104-114`, `page.js:508-525`, `:921-928`).
- Нижняя панель: строка команды/чата с историей и Tab-дополнением команд и ников, F3/F4, голосования сервера с поиском (`index.html:117-125`, `page.js:529-556`, `:656-664`).
- Горячие клавиши: Enter/t чат, `/` команда, Esc, F3/F4, Tab табло, PageUp/Down прокрутка чата (`page.js:576-610`, `:691-697`).

Вкладка «Записи»: клипы и проигрыватель, список дуэлей с общим счётом, адреса `/overlay` и `/stream` (`index.html:129-155`).

Вкладка «Настройки»:
- «Сейчас»: сервер, состояние, имя, карта, мозг, режим, версия, ловушек, память (`page.js:370-381`).
- «Бот» (только в браузере): сервер, имя, клан, скин, второй бот + ник, «Проверить обновление» (`index.html:162-176`, `page.js:382-404`).
- «Приказы в чате»: ник хозяина, LLM для непонятных фраз, пресет/URL/модель/ключ (`index.html:177-190`, `page.js:155-193`).
- «Автоматический чат»: периодическое сообщение, ответ на упоминание, правила «слово → ответ» (`index.html:191-201`, `page.js:981-1010`).
- «Язык» ru/en (`index.html:202-205`).
- «Графика DDNet»: путь к data, галочка «качать недостающее», FPS-лимит, текст атрибуции (`index.html:206-215`).
- «Для продвинутых: настройки поиска» (knobs) с подсказками и сбросом (`index.html:216-220`, `page.js:406-441`, `:1018-1021`).
- Тур по интерфейсу (14 шагов, `page.js:1023-1091`), «Что нового» из `whatsnew.json` (19 записей `{id, ru[], en[]}`, показывается верхняя, если не видели; `page.js:1093-1117`).

### 3.2 Electron-оболочка (`app/`)
Процессная часть (`app/main.js`, `app/lib/*`):
- Поиск папки бота (`lib/runtime.js:5-26`), выбор Node (системный ≥24 или встроенный Electron как Node) (`main.js:315-341`, `lib/runtime.js:28-76`), запуск `node start.mjs --no-open --no-console --ready-line --web-port <p>` (`lib/runtime.js:78-85`), разбор `WEBUI_READY/WEBUI_FAIL/UPDATE_APPLIED/SERVER_SWITCH` (`lib/runtime.js:92-106`).
- Супервизор: перезапуск с задержками 1/3/5/10/30 с, «crash loop» после 3 падений, «здоров» после 60 с (`lib/supervisor.js:3-22`), мягкая остановка через `POST /cmd !quit` (`lib/botProcess.js:225-238`), поиск чужого бота на порту и его остановка (`main.js:367-442`).
- Опрос `GET /api` каждые 1.5 с, уведомления «отключился» (после 8 с), «вернулся», «обновляется», «падает» (`main.js:505-527`, `lib/supervisor.js:71-104`).
- Пауза/продолжение: `!mode hold` ↔ `!go`/`!mode passive` с запоминанием прежнего режима (`main.js:537-564`), глобальная клавиша Ctrl+Shift+F9 (`main.js:817-830`, `lib/prefs.js:6`), кнопка на миниатюре панели задач, пункт трея.
- Браузер серверов: мастер-серверы `https://master{1..4}.ddnet.org/ddnet/15/servers.json` (`lib/servers.js:3-8`), кэш 30 с (`main.js:947-964`), адрес предпочтительно `tw-0.6+udp` (`lib/servers.js:10-27`), строки `{address, name, map, gameType, location, passworded, players, clients, maxClients, names[]}` (`lib/servers.js:31-57`).
- «Играть здесь»: пишет сервер (и пароль) в `settings.json` и перезапускает бота (`main.js:1045-1062`, `lib/settings.js:128-139`); избранное и недавние в `prefs.json` (`main.js:1063-1070`, `lib/prefs.js:437-440`).
- Настройка запуска: сервер (пусто = auto), ник ≤15, клан ≤11, скин ≤23, пароль ≤64, мозг planner/bold/scripted, второй бот + ник, lowCpu/strong взаимоисключающие (`lib/settings.js:9-91`).
- Стартовый экран: текущая конфигурация + история (≤20 записей с числом сессий и временем игры), автозапуск через 10 с (`main.js:72`, `:166-194`, `:995-1021`, `lib/prefs.js:95-138`).
- Лог процесса (stdout/stderr/app), кольцо 4000 строк, фильтр, копирование (`main.js:132`, `:262-267`; `ui/shell.js:849-894`).
- Отчёт об ошибке: zip на рабочий стол из `runs/clips`, `runs/memory`, `runs/ab*.json`, `.version`, лога окна, `settings.json` без пароля, плюс выбранные демки (`main.js:840-887`, `lib/archive.js:6-66`, `lib/zip.js`).
- Окно: без рамки, свой заголовок, мини-режим 440×300 поверх всех, «поверх всех окон», трей, закрытие в трей, автозапуск с Windows, GPU вкл/выкл, язык окна, ярлыки, скриншот-режим для CI (`main.js:588-830`, `:1236-1306`).
- Безопасность оболочки: `contextIsolation`, `sandbox`, IPC только из `app://shell` (`main.js:966-976`), сеть рендерера разрешена только к порту бота на 127.0.0.1 (`main.js:1176-1201`), CSP шелла (`ui/index.html:5`), iframe с `sandbox` (`ui/index.html:38`).

Интерфейс шелла (`app/ui/index.html`, `shell.js`, `shell.css`, `i18n.js`): заголовок с пилюлей статуса и кнопками Пауза/Сервера/Лог/Настройки/Поверх/Мини (`ui/index.html:11-34`); iframe со страницей бота (`:38`); экраны загрузки, старта, «нет папки», настройки (`:41-148`); ящик серверов с поиском по серверу/карте/режиму/адресу/нику игрока, фильтрами «Не пустые», «Есть места», «Без пароля», «Избранные», строкой «Недавние», деталями с игроками, полем пароля, «Адрес», «Играть здесь», ручным адресом (`:150-190`; логика `shell.js:500-713`); ящик настроек (`:192-235`); лог внизу с изменяемой высотой (`:238-247`); тосты. `i18n.js`: словарь RU→EN ~300 строк, `makeT`, `resolveLang`, `translateDom` (перевод DOM по кириллическим текстам).

### 3.3 Что есть только в Electron и нужно сделать заново на веб-странице
| Функция | Что сделать в вебе |
|---|---|
| Браузер серверов (мастер-список, поиск, фильтры, избранное, недавние, детали, пароль, «Играть здесь», ручной адрес) | Панель «Сервера»; бэкенд качает master-list (кэш 30 с), избранное/недавние хранит на сервере; «Играть здесь» = сменить сервер без перезапуска процесса (переподключение) |
| Первый запуск и полная форма запуска (пароль, мозг/чекпойнт, второй бот, lowCpu/strong) | Форма «Запуск» в настройках; пароль только на запись (`hasPassword`) |
| Стартовый экран с историей конфигураций | Необязательно; можно «История серверов/конфигов» в панели «Сервера» |
| Пауза/продолжение одной кнопкой с восстановлением режима | Большая кнопка в шапке (для телефона главная) |
| Перезапуск бота | Кнопка «переподключить» / «перезапустить сервис» (последнее через systemd, по подтверждению) |
| Лог процесса (stdout/stderr) | Вкладка «Системный лог» с фильтром (кольцо на сервере) |
| Уведомления «отключился/вернулся/упал/обновился» | Тосты в UI + опционально Web Push/ntfy (не проверено, что нужно) |
| Отчёт-zip | `GET /api/report.zip` (без пароля и ключей) |
| Открыть папку клипов/бота | Не нужно; скачивание клипов уже есть |
| Язык окна, «О программе», версии | Раздел «О программе» + атрибуция и лицензии |
| Трей, мини-режим, поверх окон, горячая клавиша, ярлыки, GPU, автозапуск | Не переносить; вместо мини-режима компактный мобильный вид и PWA |

---

## 4. Ассеты: что читается и откуда

### 4.1 Какие файлы
Графика (страница, `webView.ts:225-240`, сервер `web.ts:82`):
- `game.png` (оружие, хук, курсоры), `emoticons.png`, `extras.png` (снежинка), `hud.png` (полоса фриза, иконки), `arrow.png` (стрелки у ников), `editor/entities_clear/ddnet.png` (атлас сущностей), `editor/speed_arrow.png`.
- `particles.png` скачивается в `CORE_DATA` (`web.ts:82`), но страницей **не используется** (в `sheets` его нет).
- `fonts/DejaVuSans.ttf` через `@font-face` (`page.css:1`).
- `skins/<name>.png`: всегда `default` и `x_ninja` (`webView.ts:228-230`) плюс скины игроков.
- `mapres/<name>.png`: внешние картинки карт по требованию (`webView.ts:357`).
- `audio/<name>.wv` → WAV (115 файлов по `SND_FILES`, `page.js:714-721`).

### 4.2 Откуда
Порядок поиска файла `findAsset(rel)` (`web.ts:70-80`):
1. Наборы, выбранные в клиенте DDNet пользователя (`cl_asset_game/emoticons/particles/hud/extras`, `cl_assets_entities` из `settings_ddnet.cfg`), кэш 5 с (`webAssets.ts:68-105`, `web.ts:64-68`).
2. Папка data: из `settings.json` `ddnetData` (проверка наличия `game.png` в `dir`, `dir/data`, `dir/ddnet/data`, `webAssets.ts:220-232`) или автопоиск (`web.ts:29-35`): Steam (Windows пути, `~/.local/share/Steam`, `~/.steam/steam`, `libraryfolders.vdf`), `~/DDNet/data`, `~/.local/share/ddnet/data`, `/usr/share/ddnet/data`, `/usr/local/share/ddnet/data`, **`vendor/DDNet-20.0-linux_x86_64/data`**, `vendor/DDNet-20.0/data` (относительно cwd) (`webAssets.ts:11-41`), распакованные клиенты в домашних папках по маске имён (`webAssets.ts:43-66`), Steam-библиотеки (`webAssets.ts:184-204`). В этом клоне `vendor/` и `runs/` отсутствуют (`vendor/` в `.gitignore`).
3. Кэш `runs/ddnet-data` (`webAssets.ts:107`).
4. Скачивание, если не выключено галочкой (`skinDownload !== "off"`): `https://raw.githubusercontent.com/ddnet/ddnet/20.0/data/<rel>` (тег 20.0, не коммит), таймаут 20 с, ≤4 МБ, проверка сигнатуры PNG/`wvpk`/TTF, белый список путей (`webAssets.ts:108-153`). Повтор после ошибки через 10 мин.

Скины (`web.ts:412-440`): `data/skins`, `<userdir>/skins`, `<userdir>/downloadedskins`, `runs/skincache` (`webAssets.ts:315-324`), иначе `https://skins.ddnet.org/skin/<name>.png`, затем `.../skin/community/<name>.png` (`webAssets.ts:5`, `:335-368`), ≤4 МБ, имя проверяется как в DDNet (`safeSkinName`, `webAssets.ts:290-313`). Это те же адреса, что у официального клиента: `cl_skin_download_url` и `cl_skin_community_download_url` (`ref/ddnet/src/engine/shared/config_variables.h:162-163`); у официального клиента community по умолчанию **выключен** (`:166`), здесь пробуется всегда.

Звук: `audio/<x>.wav` не существует в DDNet, сервер ищет `audio/<x>.wv`, декодирует `@audio/decode-wavpack` (wasm-сборка libwavpack, BSD-3, `src/bot/wavpack/LICENSE`, `LICENSE.wavpack`) в PCM16 WAV (`web.ts:101-116`, `:587-595`; `webAssets.ts:155-182`), кэш в памяти навсегда.

Карты: см. 1.3 (из клиента или `downloadedmaps` DDNet).

### 4.3 Размеры (DDNet 20.1, коммит `c9d208138f85755521f16a0096b6fe036c5c8698`, из GitHub tree API и скачанного образца)

| Файл | Байт | sha256 (20.1) |
|---|---:|---|
| game.png | 134 435 | f6ac9ec91b596db5d31ceba5312db9f60e0926b5718021e20ca027c71c81247b |
| emoticons.png | 40 801 | cdfc600b80aa7c8ecb9198d5d3103a220f5de89629b26c920a73aceb10ba0d2b |
| extras.png | 8 619 | f47a8a0941130e11f10be43a2f9b02ca5a3586ae0437dc8980e853593ca8ed56 |
| hud.png | 83 897 | f0934059e1d7d9bf50a03e744284916c29603b3bedce6c45cd1e234889104e38 |
| arrow.png | 696 | d279b393153a42cb3df1e73cf610ff5e17b42ceb8088294c0075a50e604be54b |
| editor/entities_clear/ddnet.png | 300 094 | d8a8cb5d8739748a20b6b02c24946eee4d241a52b04377cefcdee97e563002c7 |
| editor/speed_arrow.png | 414 | 0491afce7676aa9840b7fe7081f60f79ecd51f2abc6e18f21049d9fb46b2f53f |
| fonts/DejaVuSans.ttf | 757 076 | 7da195a74c55bef988d0d48f9508bd5d849425c1770dba5d7bfc6ce9ed848954 |
| skins/default.png | 4 503 | 08b3274c3ab437a65007cf34bf415b4313804f017b03a81ca7e2a2795f093af2 |
| skins/x_ninja.png | 3 686 | 5c3c93a48523fc04db06c5218769a1d7a425caa1c2a473d4741cb9e21f8b503c |
| particles.png (не нужен) | 43 829 | 50f6332b7a0ebc0561348644da4ca55cad2a04d6f276af3dcbf5e183b51458c6 |
| license.txt (корень репо) | 11 458 | b24db78f94d5e4709016dd8dd9df587e2ef9cd67ae8b1b1c6bdbfb8732f9cfa0 |
| skins/license.txt | 1 549 | ef9a479015dda0c10c50b017f873ca7684b8ff745f93b9b9f27e8812befb527e |

- Ядро без particles: **~1.33 МБ** (из них шрифт 0.76 МБ).
- Звуки, реально используемые страницей: 115 файлов `.wv`, **1 932 380 байт**; все присутствуют в 20.1. Хэши: `~/aiddnet/data/research/assets-sample/sha256-20.1-audio.txt`, список имён `audio-list.txt`.
- `mapres/*.png`: 54 файла, 4 530 637 байт всего, самый большой `winter_main_0.7.png` 292 122 (качать по требованию).
- `data/skins/*.png`: 101 файл, 734 031 байт (не нужны целиком).
- Отличия 20.0 → 20.1 в нужном наборе: только `game.png` (147 059 → 134 435) и `entities_clear/ddnet.png` (299 061 → 300 094); остальное идентично по git sha.
- Официальные архивы (HEAD 2026-09-27): `DDNet-20.1-linux_x86_64.tar.xz` 45 926 504 байт, `DDNet-20.1.tar.xz` (исходники) 37 981 364, `DDNet-20.0-linux_x86_64.tar.xz` 46 028 980. В `https://ddnet.org/downloads/sha256sums.txt` есть суммы для 20.0 (напр. linux_x86_64 `9d7863aab9135543fbb517c7f01cb20cfac22dc992e04541a1c599037a8453d3`), **для 20.1 на момент проверки нет** (копия: `assets-sample/ddnet-sha256sums.txt`).
- Проверено: `raw.githubusercontent.com/ddnet/ddnet/<commit-sha>/data/game.png` отдаёт 200, `Access-Control-Allow-Origin: *`, `cache-control: max-age=300`, содержимое совпадает с тегом.
- Образец скачан в `~/aiddnet/data/research/assets-sample/20.1/` (4.2 МБ), деревья каталогов `tree-20.0.json`, `tree-20.1.json`.
- GitHub API без токена ограничен 60 запросами в час (у меня кончился за сессию), поэтому в рантайме только raw-URL, без API.

### 4.4 Декодирование wavpack в порте
Варианты: (а) отдать браузеру wasm-декодер (`wavpack.wasm.js` умеет работать в браузере, `ENVIRONMENT_IS_WEB`), сервер отдаёт `.wv` как есть; (б) в Rust: crate `wavpack` 0.4 (FFI к libwavpack) или чистый Rust `wavicle`/`symphonia-codec-wavpack` (0.1.1, 28 загрузок) (crates.io, 2026-09-27; качество не проверено); (в) перекодировать один раз при скачивании в WAV/Opus и хранить в кэше. Сам DDNet использует системный libwavpack (`ref/ddnet/src/engine/client/sound.cpp`). Рекомендую (в) через `wavpack`-crate или (а) как запасной путь.

---

## 5. Лицензии графики и звука DDNet

### 5.1 Что написано
- `ref/ddnet/license.txt:23-25`: всё в `data`, **кроме assets, шрифтов, языков и скинов** (у них свои лицензии), под **CC-BY-SA 3.0**. Значит `game.png, emoticons.png, extras.png, hud.png, arrow.png, particles.png, editor/*, mapres/*, audio/*` это CC-BY-SA 3.0. Код (включая портированные функции рисования) под zlib (`license.txt:1-20`).
- `DejaVuSans.ttf`: © Bitstream (Vera) + Tavmjong Bah (Arev), изменения DejaVu в public domain (`license.txt:27-30`, тексты `:167-...`, `:214-...`): распространять можно, уведомление сохранять, при изменении переименовать. Мы не меняем.
- `data/skins/license.txt`: `default` и классические (bluekitty…warpaint) © Magnus Auvinen, zlib (`:1-4`); ряд скинов CC-BY, CC-BY-SA, CC0; «All other skins» (в том числе `x_ninja`) CC-BY-SA 3.0 (`:46-47`).
- `data/assets/entities/license.txt`: только набор `comfort` (CC BY-SA 3.0); `editor/entities_clear/ddnet.png` лежит не в `assets`, значит общий CC-BY-SA 3.0.
- База skins.ddnet.org (`https://skins.ddnet.org/skin/skins.json`, 2150 скинов): лицензии по полю `license`: CC BY 977, unknown 592, CC BY-SA 267, CC0 174, CC BY-NC-SA 95, CC BY-NC-ND 27, zlib 18. Отдельных условий использования на сайте не нашёл (страница `ddnet.org/skins/` только таблица с колонкой License).
- Встроенные в карту картинки принадлежат авторам карт, приходят с сервера игры (как в клиенте).
- CC-BY-SA 3.0 (legalcode, проверено): «Distribute» = сделать доступным публике копии; при распространении без изменений нужно сохранить уведомления, указать автора, название, URI, указанный лицензиаром, и приложить текст или URI лицензии (§4(a), §4(c)); включение в «Collection» не требует переводить остальную коллекцию под BY-SA.

### 5.2 Вердикт
**Можно.** Наш бот GPL-3.0 может в рантайме скачивать эти файлы с официального источника (raw GitHub DDNet с закреплённым коммитом или официальный архив ddnet.org) и отдавать их браузеру владельца:
- Файлы не входят в программу и не лежат в git, поэтому вопрос совместимости CC-BY-SA с GPL не возникает (ассеты отдельные произведения, как и сейчас в DDNet-AI, `NOTICE:45-47`).
- Отдача в браузер владельца за паролем это частное воспроизведение; даже если считать это распространением, условия BY-SA выполняются атрибуцией и ссылкой на лицензию, файлы не меняем. (Юридически не проверено, мнение по тексту лицензии.)
- Перекраска скина и тонирование на клиенте это рендеринг, а не распространение изменённой копии.
- Скины с NC/ND/unknown из базы: только личный просмотр, никогда не коммитить и не публиковать; community-скины по умолчанию не качать (как официальный клиент), включать опцией.
- Скриншоты с этой графикой (например для README) уже подпадают под CC-BY-SA; при публикации подписывать (исходный README так делает, `README.md:177`).

Нужная атрибуция (страница «О программе / Лицензии» в UI и файл `THIRD_PARTY.md` в репо):
- «Графика и звуки: DDNet / Teeworlds, © Magnus Auvinen, DDRace и DDNet contributors, CC BY-SA 3.0, https://creativecommons.org/licenses/by-sa/3.0/, источник https://github.com/ddnet/ddnet (коммит c9d2081, тег 20.1). Файлы не изменены, скачиваются во время работы.»
- DejaVu Sans: уведомления Bitstream Vera и Arev (полный текст из `license.txt`, можно отдавать скачанный `license.txt`).
- Скины: для `default` «© Magnus Auvinen, zlib»; для скачанных с skins.ddnet.org показывать автора и лицензию из `skins.json` (`creator`, `license`).
- Код рисования, портированный из DDNet (zlib): сохранить заголовки как в `webDraw.ts:1-8`, `webView.ts:1-5`, `webMap.ts:1-4` с пометкой «altered version».
- NOTICE исходного DDNet-AI (GPLv3 §7(b)) и wavpack (BSD-3 audiojs + David Bryant), если используем их wasm.

### 5.3 Рекомендуемая схема
1. В репозитории только манифест `assets/ddnet-manifest.toml` (или JSON): `commit = "c9d208138f85755521f16a0096b6fe036c5c8698"`, `base = "https://raw.githubusercontent.com/ddnet/ddnet/{commit}/data/"`, записи `{path, size, sha256, required|lazy}` для ядра (таблица 4.3), 115 звуков, `license.txt`, `skins/license.txt`. Для `mapres/*` (54 файла) тоже sha256; их можно получить скриптом один раз (4.5 МБ). Сами файлы в `.gitignore`.
2. Кэш `~/aiddnet/data/assets/ddnet-20.1/<path>`; запись атомарно (tmp + rename) после проверки sha256 и размера; максимум 4 МБ на файл; таймаут; повтор с задержкой; один загрузчик на путь.
3. Ядро (~1.33 МБ) качать при старте, звуки и mapres лениво (или все звуки при первом включении звука, 1.9 МБ).
4. Коммит, а не тег, в URL (теги можно переписать). Запасной источник: официальный архив `DDNet-20.1-linux_x86_64.tar.xz` 45.9 МБ, sha256 взять из `ddnet.org/downloads/sha256sums.txt`, когда там появится 20.1 (сейчас нет; для 20.0 есть), вынуть только нужные пути; своей манифестной sha256 проверять каждый файл всё равно.
5. Скины: `~/aiddnet/data/assets/skins/<name>.png`, источник `https://skins.ddnet.org/skin/<name>.png`, community только по опции; `skins.json` кэшировать сутки для имени автора и лицензии; проверка имени по `safeSkinName` и PNG-сигнатуры; LRU по размеру кэша.
6. Отдавать ассеты своим сервером (`/assets/...`) с `Cache-Control: immutable` и версией коммита в пути (`/assets/c9d2081/game.png`); браузер не ходит на GitHub сам.
7. Выбор 20.1 против 20.0: исходный бот тянет 20.0 и физика «DDNet 20»; разница в ассетах только два PNG, сетка спрайтов в коде относительная (`spriteRect`, `webDraw.ts:290-294`). Брать 20.1 (последний стабильный, 2026-09-26) либо 20.0 ради полного совпадения со старым ботом; решение не критично.

---

## 6. Безопасность и надёжность текущего сервера

Хорошо сделано:
- Слушает только 127.0.0.1 (`web.ts:731`); проверка `Host` против DNS-rebinding (`web.ts:314-320`); CSRF-фильтр для не-GET по `Sec-Fetch-Site`/`Origin` (`web.ts:321-331`).
- Путь ассетов: белый список расширений, запрет `\` и абсолютных путей, `resolve` + проверка префикса (`web.ts:73-77`, `:573`); белый список скачиваемых путей (`webAssets.ts:109-115`); имя скина по правилам DDNet (`webAssets.ts:290-313`); имя клипа по регэкспу (`bot.ts:3271-3276`).
- `/api/launch` GET не отдаёт `password` и `llm` (`web.ts:543-545`); ключ LLM сохраняется только при неизменном URL (`bot.ts:3506-3508`), подменой URL его не утащить.
- Страница экранирует пользовательские строки (`esc`, `page.js:450`) при вставке в `innerHTML`.

Проблемы:
1. **Нет аутентификации.** Любой локальный процесс или пользователь машины может управлять ботом: `POST /cmd` без `Origin` проходит (`web.ts:324`), включая `!quit`, чат от имени бота, смену хозяина, смену сервера через `/api/launch`. На общем сервере это критично; в новом варианте за Caddy нужна своя авторизация.
2. **Нет TLS**, и нет смысла выставлять наружу как есть.
3. **Нет лимитов тела** у `/cmd`, `/api/knobs`, `/api/launch`, `/api/relation` (`raw += String(c)`, `web.ts:500`, `:518`, `:629`, `:704`) → память. Там же `String(chunk)` рвёт многобайтовые UTF-8 символы на границе чанков (кириллица в нике может побиться).
4. **Слабая валидация `/api/launch` POST**: любые строки любой длины пишутся в `settings.json` (`web.ts:528-531`), в отличие от Electron (`app/lib/settings.js:55-91`); `ddnetData` меняет корень раздачи ассетов (ограничено наличием `game.png` и расширениями).
5. **GET с побочными эффектами**: `GET /api/launch` запускает скачивание графики (`web.ts:551`); `GET /assets/...` при промахе качает с GitHub даже для кросс-сайтовых запросов (для скинов кросс-сайт запрещён, `web.ts:434`, для `/assets` нет). Внешняя страница может заставить бот делать исходящие запросы (мелкий риск).
6. Текст исключений уходит клиенту (`web.ts:344`, `:407`, `:507`).
7. `innerHTML` для `data-t` переводов (`page.js:66`) только из статического словаря, риска нет; CSP у страницы нет (всё inline).
8. Надёжность: поллинг 25 Гц без сжатия и дельт (раздел 1.3); тайлы слоёв без сжатия; кэш WAV в памяти без предела (`web.ts:101`); `heardSounds`/`emotes` чистятся (ок); `pages` кэш HTML по языку (ок); подписки на события клиента через `setInterval` 250 мс с «угадыванием» объекта клиента (`web.ts:217-245`) хрупкие.
9. Нет ограничения частоты команд с веба (есть только ограничение чата внутри бота, `bot.ts` `CHAT_MIN_INTERVAL_MS 2000`).
10. Внешняя сеть страницы: шрифт с `access-control-allow-origin: *` (`web.ts:580`), безвредно.

Для нового сервера (axum за Caddy):
- Слушать только `127.0.0.1` (или unix-socket), Caddy делает TLS; проверять `X-Forwarded-*` только от Caddy.
- Вход по паролю: хэш argon2id в конфиге, сессионная cookie `HttpOnly; Secure; SameSite=Strict`, срок + продление, ограничение попыток (например 5 в минуту на IP, экспоненциальная задержка), выход со всех устройств.
- WebSocket: проверка cookie при upgrade и `Origin` == наш домен; лимит размера сообщения (например 64 КБ) и частоты команд; пинг/таймаут.
- Все POST: проверка `Origin`, лимит тела (16-64 КБ), строгие типы (serde с `deny_unknown_fields`), длины как в DDNet (ник 15, клан 11, скин 23, пароль 64).
- Заголовки: CSP без inline (`script-src 'self'`), `X-Content-Type-Options`, `Referrer-Policy`, `frame-ancestors 'none'` (кроме `/overlay`, если нужен OBS; тогда отдельный токен в URL только для чтения).
- Секреты (пароль сервера, ключ LLM, токены) никогда не отдавать в UI, только флаг «задан».
- Журнал входов.

---

## 7. Предложение: структура нового UI

### 7.1 Принципы
- Один бинарь Rust отдаёт статику, встроенную в бинарь (`rust-embed`/`include_dir`), собранную заранее из TS (esbuild): `view.js` (порт `webView.ts` + `webDraw.ts` как ES-модуль), `app.js`, `app.css`. Никакого `toString()`-склеивания.
- Рисование остаётся на клиенте (Canvas 2D, позже можно WebGL). Сервер отдаёт данные.
- Mobile-first: нижняя панель вкладок на телефоне, боковая колонка на широком экране; всё управление касанием; щипковый зум и перетаскивание на канвасе, двойной тап = «следить за ботом»; табло кнопкой (без Tab); чат полем внизу; никаких подсказок только по hover. PWA-манифест (иконка на главном экране, `display: standalone`), `safe-area-inset`.
- Режим «эконом» для мобильной сети: live 10 Гц, без звука, без карты-графики (только kinds), меньше LOD-кэши.
- Языки ru/en: словарь `i18n-en.ts` перенести в JSON.

### 7.2 Разделы (вкладки)
1. **Игра**: канвас + оверлеи (чат, голосование, HUD, табло, лента заморозок), панель вида (следить/за кем/вся карта/масштаб/вид/ники/прицел/маршрут/ловушки/табло/звук) в выдвижном листе; быстрые кнопки: Пауза, драться/не лезть/стоять, убиться, клип. На широком экране справа «Управление», «Состояние», «Игроки», лог.
2. **Управление** (на телефоне отдельная вкладка): стиль (дефолт/ВБ/дуэль), сторона ВБ, режим, цель, поход, дом, наблюдать, эмоция, слабый/сильный режим, второй бот.
3. **Игроки**: список с иконками, расстоянием, фризом, бейджами; действия цель/к нему/смотреть/ник в чат/тима/вар/игнор; кланы вар/тима.
4. **Чат и лог**: лог с фильтрами и поиском, строка команды/чата с историей и дополнением, голосования сервера, F3/F4 кнопками; отдельная под-вкладка «Системный лог» (stdout/stderr сервиса).
5. **Fly** (новое): активность групп нейронов в реальном времени (тепловая сетка или столбики по группам, цвет = активность, подпись группы; история 10-30 с спарклайнами), текущие выходы DN (столбики со знаком, числовые значения), во что они превращаются во входе игры (move/jump/hook/aim/fire, стрелка прицела), задержка решения (мс), частота; пауза/заморозка картинки для разбора; выбор, какие группы показывать.
6. **Обучение** (новое): статус эксперимента (имя, состояние running/paused/done/failed, шаг, прогресс, ETA, скорость шагов/с, загрузка CPU/GPU/RAM, конфиг/хэш), кривые метрик (награда, потери, доля выигранных дуэлей, фриз/заморозил и т.п.) с выбором метрик и сглаживанием, сравнение с прошлым запуском; кнопки старт/стоп/пауза (с подтверждением).
7. **Чекпойнты** (новое, можно внутри «Обучения»): таблица `{id, шаг, время, ключевые метрики, размер, помечен лучшим}`, «применить к живому боту» (горячая замена с подтверждением), «откатить», скачать.
8. **Сервера**: мастер-список с поиском (сервер, карта, режим, адрес, ник игрока), фильтры (не пустые, есть места, без пароля, избранные), недавние, детали с игроками, пароль, «Играть здесь», ручной адрес.
9. **Записи**: клипы (проигрыватель), дуэли, ссылки `/overlay` и `/stream`.
10. **Настройки**: Запуск (ник, клан, скин, цвета, сервер, пароль только на запись, второй бот), Приказы в чате (хозяин, LLM), Автоматический чат, Параметры поиска/мозга (knobs), Графика (источник ассетов, community-скины вкл/выкл, FPS-лимит, эконом-режим), Язык, Безопасность (сменить пароль, выйти везде), О программе (версии, коммит DDNet ассетов, лицензии и атрибуция, NOTICE), отчёт-zip.

### 7.3 Транспорт
- Один WebSocket `/ws` после входа. Текстовые JSON-сообщения для управления и статуса, **бинарные кадры** для горячих потоков (live, fly). Подписки с частотой задаёт клиент, сервер ограничивает сверху и снижает при медленном клиенте (буфер отправки > N кадров → пропускать live).
- HTTP GET для больших и кэшируемых вещей: `/api/scene/<mapKey>`, `/api/scene/<mapKey>/tiles/<id>` (сжатие gzip/zstd), `/api/scene/<mapKey>/image/<i>`, `/api/map/<mapKey>` (kinds/traps бинарём), `/assets/<commit>/...`, `/skins/<name>.png`, `/clips/<name>`, `/api/train/runs/<id>/metrics?keys=&since=&every=`, `/api/servers` (кэш 30 с), `/api/report.zip`.
- Все сообщения с полем `t` (тип). Сервер добавляет `boot` в `hello`, чтобы клиент понял перезапуск.

Сервер → клиент:
| `t` | Поля | Частота |
|---|---|---|
| `hello` | `boot, version, ddnetAssets{commit}, lang, user, caps[], limits{liveHzMax,...}` | при подключении |
| `status` | как `BotStatus` + `stats{kills,deaths,selfKills,blocks,blockedBy,hooksFired,hammerFires,clips}` структурой + `lag` + `dummy` + `paused` | при изменении, не чаще 2 Гц, и раз в 5 с |
| `log` | `[{seq, at, kind, from?, sys?, text}]` пачкой | по событию; при подключении хвост 200 |
| `syslog` | строки stdout/stderr сервиса | по подписке |
| `live` (бинарь или JSON) | `tick, selfId, target, mapKey, tees[], players (только при изменении), cursor, route (при изменении), goal, doing, emotes, sounds` | 25 Гц по умолчанию, 10 Гц «эконом» |
| `map` | `{mapKey, name, width, height}`; клиент сам тянет `/api/map` и `/api/scene` | при смене карты |
| `vote` | `{text, until}` / `{closed}` | по событию |
| `duel` | `{me, now, last}` | при изменении |
| `relations` | `{war, friend, ignore, clanWar, clanFriend, partner}` | при изменении |
| `fly_meta` | `{groups:[{id,name,size,region?,color?}], dn:[{id,name,unit,range}], actionMap}` | при подключении/смене модели |
| `fly` (бинарь) | `tick, tMs, latencyUs, act: u8[nGroups] (0..255), dn: f32[nDn], action{move,jump,hook,fire,aimX,aimY}` | 10-20 Гц по подписке |
| `train` | `{run, step, t, metrics:{k:v}}` | 1 Гц по подписке (история через HTTP) |
| `exp` | `{run, name, state, step, total, eta, speed, cpu, gpu, mem, configHash, bestCkpt}` | при изменении, раз в 5 с |
| `ckpts` | `[{id, step, at, metrics, size, best, active}]` | при изменении |
| `reply` | `{id, ok, text}` ответ на команду | на запрос |
| `toast` | `{kind: info|ok|warn|error, text}` | по событию (отключился/вернулся/упал/обновился) |

Бинарный формат `live` (предложение): заголовок `u8 type=1, u8 ver, u32 seq, u32 tick, u8 selfId, u8 target, u8 nTees`, затем на ти фиксированная запись ~32 байта (`id u8, flags u16 (frozen, deep, hook state, cc, dir, weapon), x i32, y i32, hx i32, hy i32, hooked i8, aim i16 (мрад), vx i16, vy i16 (×256), jumped u8, jl u8, jt u8, emote u8, atk u16, fz u16, fzf u16, pf u8, xf u32`), имена/скины/цвета отдельно в `players` при изменении. 32 ти ≈ 1 КБ на кадр вместо ~16 КБ JSON. Можно начать с JSON + `permessage-deflate` и перейти на бинарь позже (не проверено, что deflate на телефоне дешевле).

Клиент → сервер:
| `t` | Поля |
|---|---|
| `sub` | `{topics: {live: hz, fly: hz, train: bool, syslog: bool, log: bool}}` |
| `cmd` | `{id, line}` (консольная команда или чат, как `/cmd`) |
| `pause` | `{id, on}` (с восстановлением прежнего режима, как в `main.js:537-564`) |
| `relation` | `{id, list, name, on}` |
| `knob` | `{id, key, value}` / `{id, reset: true}` |
| `launch` | `{id, server?, password?, name?, clan?, skin?, colors?, dummy?, dummyName?, lowCpu?, strong?}` |
| `server.play` | `{id, address, name?, password?}` |
| `server.favorite` | `{id, address, on}` |
| `llm`, `autochat` | как текущие POST |
| `ckpt.activate` | `{id, ckpt, confirm: true}` |
| `exp.control` | `{id, run, action: start|stop|pause|resume, confirm: true}` |
| `ping` | `{ts}` → `pong` для RTT |

### 7.4 Паритет: чек-лист
Всё из 3.1 (Игра, Управление, Состояние, Игроки, Лог, команда/чат, голосования, клипы, дуэли, overlay/stream, настройки, knobs, LLM, автосообщения, язык, тур/что нового по желанию) + из 3.3 (сервера, запуск, пауза, перезапуск/переподключение, системный лог, уведомления, отчёт, о программе) + новые Fly/Обучение/Чекпойнты. Тур и «что нового» можно отложить.

---

## 8. Риски

1. **Память канваса на телефоне**: кэши чанков до 48 Мпикс и тонированных картинок до 40 Мпикс (`webView.ts:198-199`), всё в RGBA это до ~350 МБ; мобильный Safari может убить вкладку (точный предел не проверено). Нужны меньшие пределы на телефоне и режим «только kinds».
2. **Трафик**: 125-740 КиБ/с JSON на 25 Гц (оценка); плюс несжатые слои тайлов (w·h·2 байта каждый) и kinds в base64 при смене карты. Нужны бинарь/сжатие и эконом-режим.
3. **Порт рендерера**: 2700 строк TS, завязанных на формы данных сервера; надо сохранить форматы сцены/кадра или переписать обе стороны согласованно. Трюк с `toString()` не переносится, нужен нормальный сборщик.
4. **Разбор карты в Rust**: `webMap.ts` опирается на `DataFileReader` (`src/map/datafile.ts`); версии слоёв (`tv <= 2` смещения, `webMap.ts:322`), огибающие bezier (`webMap.ts:474-475`), особые слои. Сравнить с `ref/ddnet/src/game/mapitems.h`.
5. **Wavpack в Rust**: зрелость чистых crate не проверена; FFI требует libwavpack в системе; запасной путь: декодер в браузере.
6. **Лицензии**: скины из базы с NC/ND/unknown только для личного просмотра; при любой публичной трансляции (`/stream` в OBS) картинка содержит CC-BY-SA графику и чужие скины: нужна подпись в трансляции (не проверено, как это оформить).
7. **NOTICE §7(b)** исходного проекта распространяется на порт: сохранить NOTICE, атрибуцию Wranked1, пометку «modified».
8. **Источник ассетов**: raw.githubusercontent может ограничивать частоту (лимиты не проверены); кэшировать навсегда, качать один раз; запасной путь через официальный архив.
9. **Безопасность**: выход в интернет через Caddy меняет модель угроз: сейчас защита рассчитана только на локальный браузер; обязательны вход, CSRF/Origin-проверки на WS, лимиты, отсутствие секретов в ответах.
10. **Функции, опирающиеся на Windows/Electron** (пауза глобальной клавишей, трей, мини-окно, чтение `settings_ddnet.cfg` и `downloadedmaps` клиента DDNet пользователя) на headless-сервере теряют смысл; поиск установки DDNet на сервере не нужен, остаётся только скачивание по манифесту.

## 9. Файлы, созданные при исследовании
- `~/aiddnet/data/research/orig-web.md` (этот файл).
- `~/aiddnet/data/research/assets-sample/`: `20.1/` (ядро, 115 звуков, лицензии), `sha256-20.1-core.txt`, `sha256-20.1-audio.txt`, `audio-list.txt`, `tree-20.0.json`, `tree-20.1.json`, `ddnet-sha256sums.txt`. Всего 4.2 МБ.
