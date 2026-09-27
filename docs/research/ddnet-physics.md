# Физика DDNet: анализ для Rust-порта (фаза 0)

Дата: 2026-09-27. Источники: `~/aiddnet/ref/ddnet` (shallow clone master, commit `9576fd6181d9`, 2026-09-27 09:45 UTC),
TS-порт `~/aiddnet/DDNet-AI/src/core/*.ts` (прочитан полностью, 2632 строки, не изменялся).
Эксперименты: `~/aiddnet/data/research/physics-scratch/` (воспроизведение: `run.sh`).

Обозначения: `C++ file:line` — пути относительно `ref/ddnet/src/`; `TS file:line` — относительно `DDNet-AI/src/core/`.
«не проверено» — утверждение не подтверждено кодом/экспериментом.

---

## 0. Главные выводы (TL;DR)

0. **Версии:** ref = master «20.2» (в разработке, тега нет); последний релиз — **20.1 (2026-09-26)**, 20.0 — 2026-08-27.
   Между 20.0 и 20.1/master изменений в `gamecore`/`collision` нет; серверные: 128 игроков (`TEAM_SUPER=128`), rescue,
   сброс хука при `/load`, `CTuneParam` INT_MIN. Ближайшие к физике изменения — в 19.x (спидап типа 29 — 19.1, первый
   early input за тик — 19.4, спидап на индексе 0 — 19.9). Раздел 1.
1. **TS-порт считает в float64 и НЕ совпадает с сервером потиково.** Эксперимент (реальные `gamecore.cpp`/`collision.cpp`
   DDNet, скомпилированные g++ 13.3 -O2 x86-64, против TS `CharacterCore`/`Collision` на одной синтетической карте, 3 тии,
   случайные входы): первое расхождение на **2–5 тике** (1 px), через 200–500 тиков расхождение >100 px.
   Даже при «teacher forcing» (TS каждый тик стартует из точного состояния C++) **~40% мировых шагов (≈15% шагов на тии)
   отличаются** (±1 px pos, ±1/256 vel, иногда флип коллизии → vel.y 0 против 0.5).
   Причина: после квантования pos целые, vel кратна 1/256, гравитация 0.5 → `pos+vel` часто ровно `x.5`, и направление
   округления решает знак накопленной ошибки `MoveBox` (f32 ≠ f64).
2. **Логика TS-ядра верна.** Та же C++-ядро, пересобранное в double (float→double, суффиксы `f` убраны, включая `tuning.h`),
   совпадает с TS **бит-в-бит на 15 000 строк (5 сидов × 3000 тиков × 3 тии)**. Т.е. всё расхождение ядра — точность.
3. **Rust f32 может быть бит-в-бит с C++.** Минимальный Rust-порт ядра (~250 строк, `physics-scratch/rustcore`) совпал с C++
   на **20 сидов × 10 000 тиков × 3 тии** с первой попытки. x87 не используется (SSE), FMA на x86-64 без `-march` не
   генерируется (0 `vfmadd`), `-O0 == -O2`. Rust не делает FP-contraction. НО: при `-march=haswell` (GCC `-ffp-contract=fast`)
   2 из 5 сидов расходятся → сборки DDNet под aarch64 (где FMA базовая) вероятно дают другую физику (не проверено на ARM).
4. **libm:** использовать `f32::powf/sin/cos/atan` из std (на linux-gnu они вызывают glibc — 0 расхождений на 2 млн проб).
   Чисто-Rust крейт `libm` 0.2.16 расходится с glibc: `powf` в 9.7% проб, `sinf` 1.1%, `cosf` 0.3%, `atanf` 0.2%.
5. **Порядок тика в TS в целом повторяет сервер** (direct-input fire → PreTick/Tick ядра → оружие → тайлы → Move+Quantize,
   список «новые первыми»), но **не хватает многих механик**, важных для block-карт: фриз/дип/смерть на **front-слое**,
   switch-слой (двери, таймеры, switch-freeze, TILE_JUMP), tune-зоны, endless hook, live freeze, NPC/NPH/HIT-тайлы,
   solo/teams, пикапы (сердце = фриз), драггеры/плазма, hook-телепорты. Анализ 6 block-карт это подтверждает (раздел 6).
6. **Ground truth:** рекомендую (a) сразу — standalone C++ harness с настоящими `gamecore/collision/layers` (прототип уже
   собран и работает, ~10 внешних символов-заглушек); (b) основной эталон — in-process сервер DDNet по образцу
   `src/test/gameworld_test.cpp` (реальный `CGameContext::OnTick` без сети, синхронно, детерминированно);
   (c) приёмка — реальный headless DDNet-Server + teehistorian. Готовый задел: открытый ddnet PR #11498 (`libddnet.so` с
   C ABI, его использует twgame). twgame — полезный референс, но AGPL (не копировать/не линковать). Раздел 5, 8.
7. **Parity-стратегия:** Rust-физика параметризована типом скаляра (`f32` — прод и сравнение с C++ бит-в-бит; `f64` —
   сравнение с TS бит-в-бит на подмножестве механик, которые TS реализует). Сравнение Rust-f32 с TS — только с допуском и
   только одношагово (teacher forcing), см. раздел 4.

---

## 1. Версия DDNet и изменения физики 20.0 → текущий релиз

- `C++ game/version.h:7-11`: `DDNET_VERSION_NUMBER 20020`, `GAME_RELEASE_VERSION_INTERNAL 20.2`. Это **master в разработке**:
  коммит «Version 20.2» (fbf73df71) от 2026-09-19, тега 20.2 нет. **Последний стабильный релиз — 20.1 (2026-09-26)**
  (источники: https://ddnet.org/downloads/ , https://github.com/ddnet/ddnet/tags ; GitHub Releases DDNet больше не публикует).
- Эталон надо собирать из **тега той версии, что стоит на целевых серверах** (сервер сообщает версию в server info),
  а не из master.
- Полезный маркер физических изменений: DDNet повышает `TEEHISTORIAN_VERSION_MINOR` (`game/server/teehistorian.cpp`) при
  изменениях физики: 19.0–19.2.1 = 9, 19.3 = 10, 19.4 = 12, 19.5 = 13, 19.6 = 17, 19.7 = 18, 19.8 = 19, 19.9/20.0 = 22,
  20.1/master = 23 (23 — роли в teehistorian, не физика).

### 1.1 Релизы и изменения (веб-проверка по changelog и git-истории)

Релизы: 19.0 (2025-02-26), 19.1 (2025-03-29), 19.2 (тег 2025-05-09), 19.2.1 (2025-06-05), 19.3 (2025-06-21), 19.4 (2025-09-09),
19.5 (тег 2025-10-13), 19.6 (тег 2025-12-14), 19.7 (тег 2026-01-21), 19.8 (тег 2026-03-12), 19.8.1/.2 (2026-04-19),
19.5.1/19.6.1/19.7.1 (2026-04-20, security), 19.9 (2026-07-09), **20.0 (2026-08-27)**, **20.1 (2026-09-26)**.

**20.0 → 20.1 → master: в коде движения/хука/коллизий (`gamecore`, `collision`) изменений нет.** Серверные изменения:
- 20.0: #12018 — 128 игроков: `SERVER_MAX_CLIENTS` 64→128, `TEAM_SUPER = MAX_CLIENTS` (128; раньше 64, теперь команда 64 —
  обычная; `LEGACY_TEAM_SUPER=64` для старых клиентов) → Rust: массивы на 128 клиентов, `m_apCharacters` цикл до 128.
  #12405, #12497 — rescue (не трогает `m_DDRaceState`; пропущенный rescue не размораживает). Preinput-батчинг — только сеть.
- 20.1: #12644 — при `/load` хук, прицепленный к земле, сбрасывается (`HOOK_RETRACTED`, `HookPos=Pos`), если тайл под
  `HookPos` больше не `TILE_SOLID` (`save.cpp`). #12664 — `CTuneParam = float` при выходе `v*100` за int/NaN хранит `INT_MIN`
  (уже в ref: `gamecore.h:29-37`); дефолты не изменились. a34ebba73 — mapbugs применяются только при совпадении имени,
  размера и SHA256. #12695 «handle input earlier» — только клиент («no physics»). #12589 — оптимизация 128 игроков «без
  изменения физики».
- master (20.2): #12852 (practice-бинд телепорта дамми). Открытые PR, за которыми следить: #12878 (регрессия из #10399
  в 19.9: лазер/шотган игрока без персонажа перестали попадать — есть в серверах 19.9–20.1), #12576 (оптимизация
  коллизий игроков), #11032 (фикс застревания в углу, метка `fix-changes-physics`), #12359 (tune lock), #8432 (рефакторинг
  обработки ввода), #11498 (C-ABI `libddnet.so` с физикой DDNet — используется twgame как референс).

Контекст 19.x (важное для порта): 19.1 #9670 — **новый спидап `TILE_SPEED_BOOST=29`** (старый переименован в `_OLD=28`),
#9763 — при `MaxSpeed==0` предел из velramp-тюнов; 19.1 #9788 jetpack/tune-zone override; 19.3 #9787 спавны по всем типам;
19.4 #10528 — используется только **первый** predicted early input за тик; 19.5 #10844, 19.6 #10985 — позиция спавна;
19.6 #11074 — clamp старой скорости ниндзя; 19.9 #12055 — спидапы на тайле с индексом 0 заработали; #10399 laser interact
state; #12199 неинициализированный `m_Bouncing`; #11917 рефакторинг `IsOnGround` (без изменения физики).
Старее: `OnPredictedEarlyInput` (direct input внутри цикла тиков) — с #5032 (2022); `MoveBox pGrounded`/упругость — #5398 (2023).

### 1.2 Особенности серверного кода, важные для порта (видны в ref)

- **Early input / direct input внутри тика** (с 2022, #5032; 19.4 #10528 — только первый ввод за тик).
  `CServer::Run` (`engine/server/server.cpp:3547-3592`): до `m_CurrentGameTick++` для каждого клиента в порядке client id
  вызывается `OnClientPredictedEarlyInput` (ввод, помеченный тиком `Tick()+1`) → `CPlayer::OnPredictedEarlyInput`
  (`game/server/player.cpp:678-690`) → `CCharacter::OnDirectInput` (`game/server/entities/character.cpp:744-764`) →
  `HandleWeaponSwitch()+FireWeapon()`. Затем `m_CurrentGameTick++`, `OnClientPredictedInput` (→ `m_SavedInput`),
  `GameServer()->OnTick()`. `CGameContext::OnClientDirectInput` (`gamecontext.cpp:1572`, вызывается при приходе пакета,
  `server.cpp:1989`) персонажа не трогает (`CPlayer::OnDirectInput`, `player.cpp:652-676` — только флаги/AFK).
  Выстрел детерминирован и привязан к тику.
- **Спидап двух типов**: `TILE_SPEED_BOOST = 29` (`mapitems.h:154`, `character.cpp:1575-1593`) и `TILE_SPEED_BOOST_OLD = 28`
  (`character.cpp:1521-1574`); тип — `CSpeedupTile::m_Type` (`mapitems.h:661-669`).
- **`MoveBox(..., bool *pGrounded)`** (`collision.cpp:528-601`, `568-569`, `586-587`) + сброс прыжков в
  `CCharacterCore::Move` (`gamecore.cpp:547-557`) — только при `ground_elasticity_y > 0`.
- **Прыжковые правила** (`m_Jumps == -1/0/1`, `character.cpp:2300-2326`, комментарии `gamecore.cpp:226-230`).
- Официальные Linux-сборки: Debian 11 chroot, `-DCMAKE_BUILD_TYPE=Release -DIPO=ON` (LTO), `-no-pie -g`
  (ddnet-scripts `release/build.sh`); `-std=c++20` без расширений; без `-march`/`-ffast-math`/`-ffp-contract`.
  Debian 11 = glibc 2.31 (новая реализация `powf/sinf/cosf` есть с 2.28 — совпадение битов с 2.39 не проверено).

---

## 2. Порядок операций за тик на сервере (C++) и в TS

### 2.1 Сервер, один тик T (non-`sv_no_weak_hook`)

```
CServer::Run loop (engine/server/server.cpp:3547)
 ├─ OnPreTickTeehistorian; UpdateDebugDummies
 ├─ for c in 0..MAX_CLIENTS (порядок client id!):                         server.cpp:3551-3566
 │     OnClientPredictedEarlyInput(c, input[T])                            gamecontext.cpp:1605-1633
 │       → CPlayer::OnPredictedEarlyInput (спавн по fire, если мёртв)      player.cpp:678-690
 │       → CCharacter::OnDirectInput: LatestPrev=Latest; Latest=input;     character.cpp:744-764
 │           if NumInputs>1: HandleWeaponSwitch(); FireWeapon();           (Server()->Tick()==T-1 здесь!)
 │           LatestPrev = Latest
 ├─ m_CurrentGameTick++  (теперь Tick()==T)                                server.cpp:3570
 ├─ for c: OnClientPredictedInput(c, input[T]) → m_Input=m_SavedInput=... character.cpp:728-742
 └─ CGameContext::OnTick                                                   gamecontext.cpp:1201
      ├─ tuning copy; m_World.Tick()                                       gamecontext.cpp:1222-1223
      │   CGameWorld::Tick                                                 gameworld.cpp:202-263
      │    ├─ for type in [PROJECTILE, LASER, PICKUP, FLAG, CHARACTER]      gameworld.h:26-30
      │    │    for ent in list(type) (голова списка = последний вставленный): ent->Tick()
      │    │    CCharacter::Tick                                           character.cpp:818-855
      │    │     ├─ PreTick                                                character.cpp:790-816
      │    │     │   ├─ DDRaceTick                                          character.cpp:2228-2286
      │    │     │   │   m_Input = m_SavedInput
      │    │     │   │   LiveFrozen → Direction=Jump=0 (hook разрешён)
      │    │     │   │   if FreezeTime>0: FreezeTime--; Direction=Jump=Hook=0; if FreezeTime==1: Unfreeze()
      │    │     │   │   HandleTuneLayer (m_Core.m_Tuning = зона по m_Pos)  character.cpp:2126-2138
      │    │     │   │   m_IsInFreeze (только индикация), TrySetRescue
      │    │     │   └─ m_Core.m_Input = m_Input; m_Core.Tick(true, true)   gamecore.cpp:195-463 (+TickDeferred 465-536)
      │    │     ├─ HandleWeapons: HandleNinja, HandleJetpack, PainSound,
      │    │     │   if ReloadTimer: ReloadTimer-- else FireWeapon()        character.cpp:658-676
      │    │     ├─ DDRacePostCoreTick                                      character.cpp:2288-2365
      │    │     │   EndlessHook → m_Core.m_HookTick = 0
      │    │     │   m_FrozenLastTick = false
      │    │     │   DeepFrozen → Freeze()
      │    │     │   правила прыжков (Jumps -1/0/1, EndlessJump)
      │    │     │   HandleSkippableTiles(GetMapIndex(m_Pos)): смерть (game+front), клип, спидапы  1474-1595
      │    │     │   for idx in GetMapIndices(m_PrevPos, m_Pos): HandleTiles(idx) (или CurrentIndex)  1637-2124
      │    │     │   телеган
      │    │     └─ m_PrevInput = m_Input; m_PrevPos = m_Core.m_Pos
      │    ├─ for all ents: TickDeferred()
      │    │    CCharacter::TickDeferred: (reckoning core); m_Core.Move(); m_Core.Quantize(); m_Pos = m_Core.m_Pos
      │    │                                                               character.cpp:857-971 (Move 876, Quantize 878)
      │    ├─ RemoveEntities; m_StrongWeakId = позиция в списке             gameworld.cpp:254-262
      ├─ controller Tick; для каждого игрока CPlayer::Tick (респавн TryRespawn → новый CCharacter вставляется в ГОЛОВУ списка)
      │                                                                    player.cpp:246-267, 816-833
      └─ switch-таймеры (TIMEDOPEN/TIMEDCLOSE истекают)                     gamecontext.cpp:1462-1479
```

Важно: тайлы обрабатываются по пути **предыдущего** `Move` (от `m_PrevPos` до `m_Pos`), т.е. эффект тайла (фриз,
телепорт, стоппер, спидап) наступает в начале следующего тика, после `Core.Tick` (ввод/хук) и до `Move`.
`m_Pos` (entity) обновляется только в `TickDeferred`; телепорт в `HandleTiles` меняет `m_Core.m_Pos`, а `m_Pos` — после Move.

С `sv_no_weak_hook=1` (`gameworld.cpp:214-223`, `character.cpp:815,820-826`): сначала `PreTick` всех персонажей
(`Core.Tick(true,false)`), затем для каждого `Tick()`: `m_Core.TickDeferred()` + оружие + тайлы.

### 2.2 Ядро `CCharacterCore::Tick` (`gamecore.cpp:195-463`)

1. `m_MoveRestrictions = GetMoveRestrictions(switch-cb, m_Pos)` (стопперы, двери) — 197.
2. `Grounded = IsOnGround(m_Pos, 28)`: точки `(x±14, y+14+5)` (`collision.cpp:518-526`) — 201.
3. `TargetDirection = normalize(vec2(TargetX, TargetY))` (int→float) — 202.
4. `m_Vel.y += Gravity` — 204; MaxSpeed/Accel/Friction по Grounded — 206-208.
5. Ввод: Direction; Angle = `(int)(atan2(double) … *256)` (только отображение); прыжок (ground: `-GroundJumpImpulse`,
   `Jumped |= 1` или `|=3` если `Jumps<=1`; air: `-AirJumpImpulse`, `Jumped|=3`, `JumpedTotal++`); хук: IDLE→FLYING,
   `HookPos = Pos + TargetDirection*28*1.5`, `HookTick = (float)50*(1.25-HookDuration)`; без хука → IDLE — 211-284.
6. `if Grounded: Jumped &= ~2; JumpedTotal=0` — 289-293.
7. Горизонталь: `SaturatedAdd(±MaxSpeed, Vel.x, ±Accel)` или `Vel.x *= Friction` — 296-301.
8. Хук: IDLE → HookPos=Pos; RETRACT_START..END → ++; END → RETRACTED; FLYING: `NewPos = HookPos + HookDir*HookFireSpeed`,
   ограничение `HookLength` от `HookBase` (или `m_HookTeleBase` при `m_NewHook`), `IntersectLineTeleHook`
   (NOHOOK → retract, TELEINHOOK → телепорт хука, иначе ground), затем проверка игроков по `closest_point_on_line`
   с радиусом `28+2` (ближайший к `HookPos`), с учётом teams/solo/super — 304-409.
9. GRABBED: к игроку — `HookPos = other.Pos` (или release, если `!CanKeepHook`); к земле при `dist>46` —
   `HookVel = normalize(HookPos-Pos)*HookDragAccel`, `y*=0.3` если вниз, `x*=0.95/0.75`, применить если
   `|NewVel| < HookDragSpeed || |NewVel| < |Vel|`; `HookTick++`; отпуск игрока при `HookTick > 60` или смерти — 411-459.
10. `TickDeferred` (если DoDeferredTick): для каждого другого ядра (по client id): столкновение игроков
    `dist < 28*1.25` → `Vel += Dir*a*(Velocity*0.75); Vel *= 0.85`; влияние хука на игрока `dist > 42`:
    `SaturatedAdd(±DragSpeed)` обоим + `ClampVel(MoveRestrictions)`; `NewHook=false` если не FLYING;
    `|Vel| ≤ 6000` — 465-536.

`CCharacterCore::Move` (`gamecore.cpp:538-607`): `Ramp = VelocityRamp(|Vel|*50, 550, 2000, 1.4)` (`powf`, 123-128),
`Vel.x *= Ramp`; `MoveBox(Pos, Vel, 28×28, elasticity, &Grounded)`; `Colliding/LeftWall`; `Vel.x *= 1/Ramp`;
проверка прохождения сквозь игроков по отрезку (шаг 1 px, `D < 28` → остановиться на `LastPos`).

`Quantize` (`gamecore.cpp:702-707`) = `Write` (`609-626`) + `Read` (`628-644`): `Pos`→`round_to_int`, `Vel`→`round_to_int(v*256)/256`,
`HookPos`→int, `HookDir`→`round_to_int(d*256)/256`. `round_to_int(float f) = f>0 ? (int)(f+0.5f) : (int)(f-0.5f)`
(`base/math.h:17-20`) — сложение в float.

### 2.3 TS `SimWorld.step()` (`world.ts:936-997`)

```
tick++
респавн мёртвых с истёкшим respawnAtTick (в начале шага)              world.ts:942-946
tickEntities: снаряды, лазеры                                         948 (C++: после direct-fire)
for rec in order (голова = последний добавленный):                     950-959 (C++: по client id)
    handleWeaponSwitch; fireWeapon(direct); prevInputForEdge = input
[noWeakHook: preTick всех]
for rec in order:
    preTick: freeze--, ==1 → unfreeze, обнулить dir/jump/hook; core.tick(true, !noWeakHook)   916-934
    reload-- или fireWeapon
    frozenLastTick=false; deepFrozen → freeze; applyJumpRules
    handleTiles: смерть (только game), клип, спидап(старый тип), handleTile по getMapIndices   869-898
    prevPos = pos
for rec in order: core.move(); core.quantize()
```

Структура совпадает с сервером (хорошо сделано), отличия — в разделе 3.

---

## 3. Список отличий TS ↔ C++ (сервер master)

Важность для block: **К** — критично, **В** — важно, **С** — средне/зависит от карты, **Н** — низко, **—** не важно.

### 3.A Числа и точность

| # | Что | C++ | TS | Важн. |
|---|-----|-----|----|-------|
| A1 | Вся физика в float64 вместо float32 | `float` везде (`base/vmath.h:14-190`, `gamecore.cpp`, `collision.cpp`) | `number`, нигде нет `Math.fround` (`vmath.ts`, `characterCore.ts`, `collision.ts`) | **К** для паритета (расхождение с 2–5 тика, 15%/шаг на тии) |
| A2 | Значения тюнинга | `CTuneParam`: `int m_Value=(int)(v*100.0f)`, чтение `m_Value/100.0f` (`gamecore.h:23-39`) → 13.1999998f, 0.949999988f, 1.39999998f | `tune(v)=trunc(v*100)/100` в f64 (`tuning.ts:5-7`) → 13.2, 0.95, 1.4. Целые m_Value совпадают для всех 47 параметров (проверено `tunecheck.cpp`) | часть A1 |
| A3 | `round_to_int` | `(int)(f+0.5f)` в float (`base/math.h:17-20`) | `trunc(f+0.5)` в f64 (`vmath.ts:84`) | Н (отличие только для ±0.49999997f) |
| A4 | Трансцендентные функции | glibc `powf` (VelocityRamp), `sincosf` (GetSpeedup), `atanf/asinf/cosf`, `std::pow(float,2)`→double (`character.cpp:1534-1561`), `atan2(double)` (угол) | V8 `Math.pow/cos/sin/atan` в f64, `Math.PI` вместо `2*asin(1.0f)` (`world.ts:840-852`, `collision.ts:178-179`) | часть A1 |
| A5 | Ассоциативность операций | `Dir*a*(Velocity*0.75f)` (`gamecore.cpp:499`); `TargetDirection*PhysicalSize()*1.5f` (`:271`); молоток всегда `ClampVel(vel+boost)-vel` (`character.cpp:549-551`) | `dir*(a*(v*0.75))` (`characterCore.ts:325`); `dir*(28*1.5)` (`:180`); пропуск при `mr==0` (`world.ts:671`) | Н (в тесте не проявилось, но в Rust копировать порядок C++) |

### 3.B Порядок тика

| # | Что | C++ | TS | Важн. |
|---|-----|-----|----|-------|
| B1 | Direct-fire (молоток) — порядок игроков и относительно снарядов | по client id, ДО `tick++` и до тика снарядов (`server.cpp:3551-3566`) | в порядке списка, ПОСЛЕ `tickEntities` (`world.ts:948-959`) | Н/С (взаимные удары в один тик, разморозка через `m_FrozenLastTick`) |
| B2 | Порядок тика персонажей (strong/weak hook) | список, новые (пере)спавненные — первыми (`gameworld.cpp:79-92`); сервер отдаёт позицию в `CNetObj_DDNetCharacter::m_StrongWeakId` (`character.cpp:1347-1350`, `gameworld.cpp:256-262`) | `order.unshift` при `addTee` (`world.ts:231`); бот добавляет себя первым и т.п. (`bot/bot.ts:3923-3928`), `StrongWeakId` не используется | **В** (исход борьбы хуками) |
| B3 | Момент респавна | `CPlayer::Tick` после мирового тика; не раньше `max(DieTick, PrevDieTick+150)+2` (`player.cpp:246-267`), позиция — `CanSpawn` (оценка по расстоянию до других, `gamecontroller.cpp:89-181`) | в начале `step`, задержка параметром, `spawnPos` фиксирован (`world.ts:715-733, 942-946`) | Н |
| B4 | Выбор оружия по `WantedWeapon` | берётся `m_Input.m_WantedWeapon` (ввод прошлого тика) (`character.cpp:445-446`) | текущий ввод (`world.ts:627`) | Н |
| B5 | Первый ввод не стреляет | `m_NumInputs > 1` (`character.cpp:756`) | нет | — |
| B6 | Позиция для тайлов после телепорта в том же тике | `m_Pos` (до телепорта) в `GetMoveRestrictions` (`character.cpp:1642`) | `core.pos` (после) (`world.ts:772`) | Н |

### 3.C Отсутствующие/неполные механики в TS

| # | Механика | C++ | TS | Важн. для block |
|---|----------|-----|----|------|
| C1 | Фриз/анфриз/дип/андип/смерть на **front-слое** | `m_TileFIndex` во всех проверках (`character.cpp:1641, 1661-1672`), смерть `GetFrontCollisionAt` (`1481-1484`) | только `col.tiles[index]` (`world.ts:778-787`), `isDeath` по game (`872-876`) | **К** (BlmapChill: 161 front-freeze, Blockdale: 46) |
| C2 | **Switch-слой**: двери (`m_pDoor` → стопперы через `IsSwitchActiveCb`), switch open/close/timed (22-25), switch-freeze с задержкой `Freeze(SwitchDelay)`, switch deep/live, `TILE_JUMP`(7) = число прыжков, HIT по оружию, статусы по командам, истечение таймеров | `collision.cpp:88-114, 297-306, 858-861`, `character.cpp:1831-2014`, `gamecontext.cpp:1462-1479` | нет (`collision.ts:400-416` без switch/door) | **В** (BlmapChill, blmapV5: двери 240, свитчи 22-25, TILE_JUMP) |
| C3 | Tune-зоны (`TILE_TUNE`, per-character `m_Core.m_Tuning`) | `character.cpp:2126-2138, 2255`, `99` | глобальный `TUNING` (`tuning.ts:9-66`) | С (в BlmapChill есть tune-слой) |
| C4 | Endless hook (`TILE_EHOOK_ENABLE/DISABLE` 17/18, `sv_endless_drag`) → `HookTick=0` каждый тик | `character.cpp:1684-1692, 2292-2293, 2497` | флаг хранится, но не применяется (`characterCore.ts:91`, `world.ts:378`) | **В** на картах/серверах с ehook |
| C5 | Live freeze (144/145) | `character.cpp:1674-1682, 2235-2240` | нет (LFREEZE только в индикации `world.ts:879`) | Н/С |
| C6 | Endless jump (105/89), refill jumps (32), walljump (16), `TILE_JUMP` | `character.cpp:1736-1781, 2322-2326, 1934-1954` | нет (`applyJumpRules` без EndlessJump, `world.ts:93-98`) | С |
| C7 | HIT (19/20), NPC (88/104 → CollisionDisabled), NPH (91/107 → HookHitDisabled) по тайлам | `character.cpp:1694-1734` | только из снапшот-флагов (`world.ts:374-379`), тайлы не меняют | С (BlmapChill: NPH_ENABLE ×15, HIT ×25) |
| C8 | Solo-тайлы (21/22) и **команды** (`CanCollide/CanKeepHook`) | `gamemodes/ddnet.cpp` HandleCharacterTiles; `teamscore.cpp`; `gamecore.cpp:358, 416, 475, 589` | solo только флагом; команд нет | **В** на публичных серверах (игроки в /team не сталкиваются и не хукаются) |
| C9 | Хук-телепорты (TELEINHOOK 15, `m_NewHook`, `m_HookTeleBase`), weapon-телепорты (14) | `gamecore.cpp:320-324, 338-349, 393-403`; `projectile.cpp` | нет (`collision.ts:484-525`) | Н |
| C10 | Выбор выхода телепорта среди нескольких | `RandomOr0` на PRNG с `secure_random` сидом (`gamecontext.cpp:4137-4140`) | всегда `outs[0]` (`world.ts:799`) | С (недетерминируемо в принципе) |
| C11 | Спидап нового типа 29 | `character.cpp:1575-1593` | тип игнорируется (`collision.ts:175-180`, `world.ts:825-867`) | Н (в выборке только тип 28) |
| C12 | `pGrounded` из `MoveBox` (прыжки при упругости) | `collision.cpp:568-569,586-587`, `gamecore.cpp:553-557` | нет (`collision.ts:308-369`) | Н |
| C13 | Пикапы: сердце = `POWERUP_FREEZE` (фриз при касании), armor снимает оружие, оружие, ниндзя | `entities/pickup.cpp` (Tick до персонажей) | нет | С (в block-картах есть 197/198/199-202) |
| C14 | Драггеры, лазеры/плазма (фриз-лазеры), двери-сущности | `entities/dragger.cpp`, `plasma.cpp`, `door.cpp` | нет | С (blmapV5: плазма ×114) |
| C15 | Jetpack, ninja, телеган | `character.cpp:253-394`, `2351-2362` | нет | Н |
| C16 | `sv_deepfly=0` (нельзя молотить в дипе), per-character `m_HammerHitDisabled`, team-проверка у молотка | `character.cpp:481-482, 520-533` | глобальный `svHit` (`world.ts:659`) | С (из-за команд) |
| C17 | Смерть: practice → фриз; super/invincible; `TeeFinished` | `character.cpp:1477-1509` | нет | — |
| C18 | `TileExists` учитывает switch/tune/door | `collision.cpp:858-863` | нет (`collision.ts:400-416`) | вместе с C2/C3 |
| C19 | Лазер shotgun «bug» при `PrevPos==HitPos` (`SetRawVelocity(-2^31)`) | `laser.cpp` HitCharacter | пропуск (`projectile.ts:209`) | Н |

Что в TS реализовано верно (подтверждено экспериментом в double или чтением): гравитация, контроль, трение, прыжки (без
тайлов), хук (земля, игрок, NOHOOK, длина, 60 тиков на игроке), столкновения игроков, velocity ramp, `MoveBox`,
`IntersectLine`, hook-through (`THROUGH_CUT/THROUGH/THROUGH_ALL/THROUGH_DIR`, включая баг `TileExistsNext`
`collision.cpp:882` = `collision.ts:429`), стопперы (game+front), TELEIN/TELEINEVIL/TELECHECK*/CHECKOUT, фриз/анфриз/дип
на game-слое, `Freeze()` семантика (`m_FreezeTime>Seconds*50`, повтор не раньше 1 с), unfreeze на тике `FreezeTime==1`,
`m_FrozenLastTick` (full-auto после разморозки), молоток (радиус 14+28, сила, `HammerHitFireDelay` 16 тиков, иначе 6),
квантование в конце тика, `sv_no_weak_hook`.

### 3.D Квантование / сеть

- Квантование одинаковое (`gamecore.cpp:609-644, 702-707` ↔ `characterCore.ts:420-434`). `HookTeleBase`, `HookTick`,
  `Jumped` не квантуются (целые/служебные).
- В снапшот сервер кладёт **reckoning-ядро** (`m_SendCore`, обновляется только при расхождении предсказания или раз в 3 с,
  `character.cpp:953-970`), а не текущее; клиент должен экстраполировать (`Tick(false)+Move+Quantize` в пустом мире с
  дефолтным тюнингом, `character.cpp:859-868`). Снапшоты по умолчанию каждые 2 тика (`server.cpp:1024`, `sv_high_bandwidth=0`).

---

## 4. Плавающая точка: вердикт и стратегия паритета

### 4.1 Факты

- DDNet: `float` (IEEE binary32). Сборка x86-64 по умолчанию — SSE2 scalar (`mulss/addss/sqrtss`), x87 нет
  (проверено `objdump`: 0 x87-инструкций в `gamecore.o/collision.o`). `-ffloat-store` добавляется только для 32-битного x86
  (`CMakeLists.txt:345-350`). Флагов `-ffast-math`, `-ffp-contract`, `-march` в `CMakeLists.txt` нет.
- GCC по умолчанию `-ffp-contract=fast`, но без FMA в целевом ISA (x86-64 baseline) контракции нет (0 `vfmadd`).
  С `-march=haswell` — 25 `vfmadd` в `gamecore.o` и расхождение 2/5 сидов (тики 44, 1900). Для aarch64 FMA базовая →
  ARM-сборки DDNet, вероятно, считают иначе (не проверено на реальном ARM). Windows-сборки (MSVC CRT `powf`) — тоже
  вероятно иначе (не проверено).
- `-O0` и `-O2` дают одинаковый результат (3 сида).
- Rust не выполняет FP-contraction без явного `mul_add`; `f32::sqrt` = IEEE (как `sqrtss`); `as i32` — усечение с насыщением
  (в C++ переполнение — UB; на игровых диапазонах не встречается).
- Внешние libm-вызовы в ядре/коллизии: `powf` (VelocityRamp), `sincosf` (GetSpeedup: `direction(Angle)`), `atan2`
  (double, только угол). В персонаже (спидап старого типа): `atanf`, `asinf`, `cosf`, `std::pow(float,int)` → **double** `pow`,
  `std::sqrt(double)`, затем в float (`character.cpp:1558`) — в Rust считать `TeeSpeed` через f64.
- Проверка libm (`libmref.c` ↔ `rustlibm`): Rust std `f32::powf/cos/sin/atan` на linux-gnu = glibc 2.39 бит-в-бит
  (0 из 2 000 000); glibc `sincosf` = `sinf`+`cosf` на выборке. Крейт `libm` 0.2.16: powf 194 250/2M, sinf 22 144, cosf 5 723,
  atanf 3 613 расхождений → **не использовать**; для платформенной независимости портировать `powf` из ARM
  optimized-routines/glibc ≥2.28 (не проверено, совпадает ли реализация powf во всех glibc ≥2.28 — вероятно да).
- TS: `Math.pow/cos/sin/atan` V8 ≠ glibc в последних битах (не проверено количественно); в тесте C++-double vs TS 0 расхождений.

### 4.2 Вердикт

- **Rust f32 → бит-в-бит с C++ (x86-64 Linux, glibc)** — достижимо и продемонстрировано на ядре. Условия: тот же порядок
  операций (не «упрощать» `a*b*c`, `(v+x)-v`), `round_to_int` через `f + 0.5f32`, std libm (glibc), без `mul_add`,
  без `-C target-cpu=native`-зависимых интринсиков (Rust их сам не подставляет, но не использовать `fast-math`-крейты).
- **Rust f32 ↔ TS f64 бит-в-бит — невозможно.** Даже одношагово ~15% шагов на тии отличаются.

### 4.3 Стратегия паритета

1. **Эталон = C++.** Все поля сравниваются точно:
   - целочисленные: тайловые индексы, `m_FreezeTime`, `FreezeStart`, `HookState`, `HookedPlayer`, `HookTick`, `Jumped`,
     `JumpedTotal`, `Jumps`, `Direction`, `ActiveWeapon`, `ReloadTimer`, `AttackTick`, `MoveRestrictions`, `alive`,
     `TeleCheckpoint`, флаги (`DeepFrozen`, `LiveFrozen`, `EndlessHook`, …);
   - квантованные (после `Quantize`): `pos` (int), `vel*256` (int), `hookPos` (int), `hookDir*256` (int) — точно;
   - опционально отладочный дамп промежуточных f32 до квантования — сравнение битов (`to_bits`).
   Любое расхождение с C++ = баг Rust.
2. **Rust ↔ TS:** сделать физику обобщённой по скаляру (`trait Real: f32 | f64`). `Rust<f64>` сравнивать с TS **точно**
   (ожидаемо совпадение, как у C++-double) на сценариях, где TS реализует механику (без front-фриза, switch, tune, ehook…),
   либо с флагом `ts_compat`, отключающим отсутствующие в TS механики. Расхождение = баг TS или порта. Если обобщение
   дорого — сравнивать `Rust<f32>` с TS только одношагово (teacher forcing) с допусками: pos ±1 px, vel ±1/256,
   hookPos ±1 px, флип коллизии допускается; целочисленные поля — точно, кроме каскада от флипа; метрика — доля шагов с
   расхождением (ожидаемо ~15%/тии/шаг) и отсутствие расхождений > допуска.
3. Где TS расходится с C++ по смыслу (раздел 3) — прав C++; фиксировать в тестах как «известные расхождения TS».
4. Фиксировать версию: эталон собирать из тега версии целевых серверов; в трейсах хранить версию/коммит.

---

## 5. Стратегия получения эталонных трейсов (ground truth)

### 5.1 Зависимости C++-файлов (проверено сборкой)

`gamecore.cpp, collision.cpp, layers.cpp, mapitems.cpp, teamscore.cpp, prng.cpp` компилируются отдельно
(`-std=c++20 -I src -I gen`) после генерации `generated/protocol.h` (`python3 datasrc/compile.py network_header`).
Неразрешённые внешние символы всего: `g_Config`, `dbg_assert_imp`, `str_comp_nocase`, `str_format`, `str_length`,
`str_utf8_check` (+ libm/libstdc++). Карта: `CLayers::Init(IMap*)` → нужен `IMap` (в прототипе — фейковый in-memory;
для реальных `.map` — либо `engine/shared/datafile.cpp`+`map.cpp` (тянут `base/*`, storage, zlib — zlib есть в
`src/engine/external/zlib`), либо свой ~100-строчный ридер datafile v4 (как `physics-scratch/maptiles.py`)).
Серверный `character.cpp` тянет `CGameContext`, `IServer`, `CPlayer`, `CGameTeams`, score/sqlite, antibot — отдельно не
собирается. Клиентская предикция (`game/client/prediction/**`) собирается почти автономно, но **отличается от сервера**
(нет смерти, `Freeze()` без `m_FreezeTime == 0 ||` — `prediction/entities/character.cpp:1185-1199` vs сервер
`character.cpp:2367-2379`, DDRace-логика под флагами `m_WorldConfig`) → не эталон.

Системных `zlib/sqlite3/curl` заголовков и `cmake` на машине нет; Rust (`~/.cargo/bin`, 1.98.1), Python 3, g++ 13.3, node 24 есть.

### 5.2 Варианты

| Вариант | Что это | Трудозатраты | Точность | Скорость |
|---|---|---|---|---|
| **A. Standalone core-harness** (прототип готов: `harness_proto.cpp`) | настоящие `gamecore/collision/layers` + ридер карты + ручной перенос нужных кусков `CCharacter` (PreTick/DDRaceTick/HandleWeapons(молоток)/DDRacePostCoreTick/HandleSkippableTiles/HandleTiles/TickDeferred/Freeze/Unfreeze/OnDirectInput) и порядка `CGameWorld` | ядро — уже; персонаж — 1–2 дня; реальные карты — +0.5 дня | ядро/коллизия 100% (тот же код); персонаж — перенос (риск ошибок) | >10⁶ тии-тиков/с |
| **B. In-process сервер (рекомендуется как основной эталон)** | собрать серверную библиотеку DDNet из тега целевой версии и свой `main` по образцу `src/test/gameworld_test.cpp:57-129` (CreateServer/Kernel/Storage/Console/Config/Http/Antibot/GameServer, `LoadMap`, `OnInit`, `CreatePlayer`, `ForceSpawn`); на каждый тик: `OnClientPredictedEarlyInput` → `m_CurrentGameTick++` → `OnClientPredictedInput` → `OnTick` → дамп `CCharacter`/`CCharacterCore`/`m_FreezeTime`/…; PRNG сидировать фиксировано | 1.5–3 дня (в основном сборка: cmake-бинарник, sqlite amalgamation, curl без SSL — всё в user-space; Rust/Python есть) | 100% серверного кода (кроме сетевого тайминга ввода, которого и не нужно) | быстрее реального времени, детерминированно |
| **C. Реальный headless DDNet-Server + teehistorian** | официальный бинарник (или собранный) с `sv_teehistorian 1`, клиенты — наш бот (TS уже умеет протокол) или Rust-клиент | 0.5–1 день (+парсер: `ref/libtw2/teehistorian` есть локально) | 100% для развёрнутого бинарника, но teehistorian хранит только **вводы и целые позиции** (`game/server/teehistorian.cpp:289-330, 443`), без vel/hook/freeze; снапшоты — раз в 2 тика и reckoning | реальное время (50 тиков/с на инстанс) |
| D. twgame (Rust) | сторонняя реимплементация (AGPL), валидируемая по teehistorian | 0 на сборку; только как внешний инструмент/чтение | высокая, но не полная (нет speedup, большей части switch); ≈DDNet 19.9 | — |
| F. ddnet PR #11498 `libddnet.so` | C ABI над физикой DDNet (открытый PR; им пользуется twgame `ddnet-cpp`) | 0.5–1 день на изучение/сборку (не проверено, насколько покрывает персонажа/тайлы) | зависит от покрытия PR | высокая |
| E. Клиентская предикция | `game/client/prediction` | 1 день | отличается от сервера | — |

**Рекомендация:** A сразу (для ядра и коллизий — дешёвый точный оракул, уже работает), B как основной эталон для персонажа/
тайлов/оружия (снимает риск ручного переноса), C — приёмочные тесты на реальных картах и реальных сессиях бота.
Генераторы входов: случайные (как `gen_inputs.mjs`), «блок-сценарии» (хук/молоток/фриз-края), записи реальных игр.

---

## 6. Анализ реальных block-карт (тайлы по слоям)

`maptiles.py` на 6 картах из `github.com/DDNetPP/maps` (версии для мода DDNet++; на DDNet-серверах версии могут отличаться —
не проверено). Числа — количество тайлов (не 0).

- **BlmapChill** 1244×667: game: SOLID 138560, NOHOOK 259152, FREEZE 106193, UNFREEZE 848, DEATH 663, DFREEZE 80, DUNFREEZE 43,
  NOLASER 199, STOP 4, EHOOK_ENABLE 2, HIT 19/20, REFILL 32, JETPACK 90/106, UNLIMITED_JUMPS 105, NPH_ENABLE 107 ×15,
  TILE_CP/CP_F 64/65, сущности (192+: спавны, 197/198 armor/heart, 199-202 оружие/ниндзя); **front: FREEZE 161**, THROUGH_CUT 246,
  STOP 46, NOLASER 206, DUNFREEZE 23, START/FINISH; **switch**: двери 240 ×98, 22/23/24/25, TILE_JUMP 7 ×46, switch-FREEZE 9 ×6,
  лазеры/драггеры 210-238; speedup тип 28 ×1073; tele: TELEINEVIL 74, TELEIN 46, TELEOUT 25, TELEINWEAPON 4, CHECK 29/30/63;
  **tune** ×87.
- **ChillBlock5** 943×1075: NOHOOK 794811, SOLID 16909, FREEZE 6229, DEATH 3996, UNFREEZE 21; front: 167, 33; speedup 24; tele 10/27.
- **blmapV5_ddpp**: FREEZE 7306, DFREEZE 161; front THROUGH 224, STOP 25, DUNFREEZE 25; switch: switch-DFREEZE 12 ×128,
  22 ×573, 24/25, двери 240, **плазма 221 ×114**, драггеры 233-235; speedup 1294.
- **Blockdale**: FREEZE 3798, TELE_GRENADE 112/113; front: FREEZE 46, THROUGH_ALL 12, THROUGH_DIR 56; switch 7, 23, 240.
- **BlockField**: FREEZE 24673, DEATH 233, STOP 24; front: STOP 70, THROUGH_CUT 47, SOLO 21/22; tele 10/27; speedup 236.
- **blmapV3multistarbox**: FREEZE 16732, UNFREEZE 524, START 33; front THROUGH_CUT 35; tele TELEINEVIL 234; speedup 694; switch 22/23, 240.

Индексы вне списка DDNet (например 119, 130-179, 114, 118, 121-126) — вероятно тайлы мода DDNet++, сервер DDNet их
игнорирует (не проверено для каждого).

---

## 7. Тайлы, важные для block (id из `game/mapitems.h:124-207, 60-117`)

**Game/Front (index):** 0 AIR; 1 SOLID (хукабельный); 2 DEATH (game+front); 3 NOHOOK (нехукабельный solid); 4 NOLASER;
5 THROUGH_CUT, 6 THROUGH, 66 THROUGH_ALL, 67 THROUGH_DIR (hook-through, front/game); 9 FREEZE; 11 UNFREEZE; 12 DFREEZE (deep);
13 DUNFREEZE; 144 LFREEZE; 145 LUNFREEZE (live); 16 WALLJUMP; 17/18 EHOOK_ENABLE/DISABLE; 19/20 HIT_ENABLE/DISABLE;
21/22 SOLO_ENABLE/DISABLE; 32 REFILL_JUMPS; 33 START; 34 FINISH; 35-59 TIME_CHECKPOINT; 60 STOP (one-way, по флагам
поворота), 61 STOPS (двусторонний), 62 STOPA (все стороны) — логика `collision.cpp:196-309`; 64/65 TILE_CP/CP_F (скорость
«мувера» для лазеров, `collision.cpp:791-831`); 76 UNLOCK_TEAM; 88/104 NPC_DISABLE/ENABLE (коллизия игроков);
89/105 UNLIMITED_JUMPS_DISABLE/ENABLE; 90/106 JETPACK_DISABLE/ENABLE; 91/107 NPH_DISABLE/ENABLE (хук игроков);
96/97 TELE_GUN, 98/99 ALLOW_(BLUE_)TELE_GUN, 112/113 TELE_GRENADE, 128/129 TELE_LASER; 190/191 ENTITIES_OFF.
Флаги поворота (`mapitems.h:226-238`): XFLIP 1, YFLIP 2, ROTATE 8; ROTATION_0=0, 90=8, 180=3, 270=11.

**Tele (type):** 10 TELEINEVIL (скорость 0 + release hooked), 14 TELEINWEAPON, 15 TELEINHOOK, 26 TELEIN (скорость сохраняется),
27 TELEOUT, 29 TELECHECK, 30 TELECHECKOUT, 31 TELECHECKIN, 63 TELECHECKINEVIL.

**Speedup:** `CSpeedupTile{Force, MaxSpeed, Type(28 старый / 29 новый), Angle(short, градусы)}` (`mapitems.h:661-669`).

**Switch (type):** 7 JUMP (Delay = число прыжков, 255 → -1), 9 FREEZE (Delay = секунды), 12/13 DFREEZE/DUNFREEZE,
144/145 LFREEZE/LUNFREEZE, 19/20 HIT (Delay = оружие), 22 SWITCHTIMEDOPEN, 23 SWITCHTIMEDCLOSE, 24 SWITCHOPEN, 25 SWITCHCLOSE,
79 ADD_TIME, 95 SUBTRACT_TIME; сущности-двери/лазеры в switch-слое.

**Tune:** 68 TILE_TUNE (номер зоны).

**Сущности** (`index - ENTITY_OFFSET(191)`): 1 SPAWN, 2/3 SPAWN_RED/BLUE, 6 ARMOR_1 (снимает оружие), 7 HEALTH_1 (**фриз-пикап**),
8 SHOTGUN, 9 GRENADE, 10 NINJA, 11 LASER, 12-27 лазеры/модификаторы, 29-32 PLASMA(E/F/-/U), 33/34 CRAZY_SHOTGUN, 35-38 ARMOR_*,
42-47 DRAGGER_*, 49 DOOR.

Минимум для первого Rust-релиза под block: game+front {1,2,3,5,6,9,11,12,13,60,61,62,66,67}, tele {10,26,27,29,30,31,63},
speedup тип 28, молоток; затем switch-двери/таймеры/TILE_JUMP, ehook (17/18), NPC/NPH/HIT, solo, пикапы-сердца, tune-зоны,
live freeze, драггеры/плазма.

---

## 8. Сторонние реализации (кросс-референс)

**twgame (Zwelf)** — https://gitlab.com/ddnet-rs/twgame (старый адрес gitlab.com/zwelf/twgame устарел).
- Лицензия **AGPL-3.0-only** (twgame, twgame-core, teehistorian-replayer, tee-hee, twsnap); крейт `teehistorian` 0.12.0 — LGPL-3.0.
  Последний коммит master 2026-05-16; crates: twgame 0.11.0 / twgame-core 0.9.0 (2026-04-25).
- f32 (`vek::Vec2<f32>`), с квантованием как в DDNet. Реализовано: движение, прыжки, walljump, хук (включая hook-through),
  фриз, все оружия, solo, команды, save/load, телепорты, tune-зоны; по README не реализованы: speedup-слой, большая часть
  switch-слоя, `/kill`, чекпоинты (README частично устарел: стопперы и deep/live-freeze switch есть в `map.rs`).
- Валидация: реплей реальных teehistorian-файлов с потиковым сравнением позиций (`check_tees`, `replayer/src/lib.rs`);
  ~1000 тестов (со слов автора в ddnet#11498). Процент совпадения на реальном корпусе нигде не опубликован (не проверено).
  Поддерживает teehistorian minor ≈21–22 → DDNet ≈19.9.
- Вывод: **ценный референс для чтения и для идеи валидации по teehistorian, но AGPL** — код не копировать и не линковать
  (иначе наш проект становится AGPL). Можно запускать как внешний инструмент для сверки (не проверено юридически).
- ddnet PR #11498 (открыт): `libddnet.so` с C ABI, которую крейт `ddnet-cpp` в twgame загружает, чтобы гонять C++-физику
  DDNet как эталон — по сути готовый вариант «C++ через FFI»; стоит изучить как основу для нашего харнесса (вариант A/B).

**ddnet-rs (Jupeyy)** — https://github.com/ddnet/ddnet-rs, MIT/Apache-2.0, активен (push 2026-09-14). Своя физика
`game/vanilla` (f32 позиции, f64 курсор), в hook-teleport есть TODO; зависимость от форка twgame закомментирована.
Заявлений о потиковом совпадении с DDNet нет — считать не tick-exact.

**Прочее:**
- Teero888/ddnet_physics (C, без лицензии, «Heavy WIP») и frametee (TAS, физика «slightly altered») — намеренно не точны.
- `libtw2` (heinrich5991, MIT/Apache): локально `~/aiddnet/ref/libtw2`; `world` — ванильная 0.6 физика в f32 (756 строк,
  без DDRace-тайлов, давно не менялась), `teehistorian` — парсер, `datafile`/`map` — чтение карт (подходит по лицензии).
- `~/aiddnet/ref/twmap` (Patiga) — **AGPL-3.0**; для загрузки карт лучше libtw2 или свой ридер (формат простой,
  см. `physics-scratch/maptiles.py`).
- CovERUshKA/ddnet-nn (MIT) — форк DDNet C++ с libtorch, не реимплементация; ddnet PR #12806 (закрыт) teehistorian→demo.
- Лицензия DDNet — zlib-подобная Teeworlds + DDRace (`ref/ddnet/license.txt`); порт кода в Rust — производная работа,
  уведомления сохранить (не проверено юридически).

Известные обсуждения float в DDNet: #989 (2018, win32 x87 → `-ffloat-store`), #11890 (2026: мейнтейнеры подтверждают,
что физика зависит от величины координат, поэтому целочисленными позиции не сделать), #9947 (неточность углов дверей,
won't fix). Про ARM/FMA issue не найдено.

---

## 9. Эксперименты (`~/aiddnet/data/research/physics-scratch/`)

| Файл | Назначение |
|---|---|
| `harness_proto.cpp` | C++-харнесс: реальные `gamecore/collision/layers` DDNet, синтетическая карта 60×40 (стены, пол, NOHOOK, платформы), 3 ядра, порядок тика как у сервера (2,1,0), дамп per-tick; `full` — ещё hookDir/direction |
| `gen_inputs.mjs` | детерминированные случайные входы (xorshift) |
| `ts_proto.ts` | тот же сценарий на TS `CharacterCore`/`Collision` (импорт read-only) |
| `ts_onestep.ts` | teacher forcing: TS шагает из точного состояния C++ |
| `harness_dbl` (через `run.sh`) | C++ с float→double — совпадает с TS бит-в-бит |
| `harness_fma` | C++ с `-march=haswell` (FMA) |
| `rustcore/` | Rust f32 порт ядра — совпадает с C++ бит-в-бит |
| `libmref.c`, `rustlibm/` | glibc vs Rust std vs крейт `libm` |
| `tunecheck.cpp` | целые значения тюнинга C++ vs TS |
| `maptiles.py`, `maps/` | статистика тайлов block-карт |

Результаты (5 сидов × 3000 тиков × 3 тии, если не указано иное):

| Сравнение | Итог |
|---|---|
| C++ f32 vs TS | идентичных строк 2–25 из 3000; первое расхождение pos на тике 2–5; max |Δpos| 6–10 px к 100 тику, 100–1400 px к 500–3000 |
| C++ f32 vs TS, teacher forcing (1 шаг) | 36–43% мировых шагов с расхождением (≈15% на тии), всегда ≤1 px/≤1 ед. vel, в т.ч. флипы коллизии (vy 0 vs 0.5) |
| C++ double vs TS | 15 000/15 000 строк идентичны |
| Rust f32 vs C++ f32 | 20 сидов × 10 000 тиков идентичны |
| C++ -O0 vs -O2 | идентичны |
| C++ -march=haswell (FMA) vs обычный | 2/5 сидов расходятся (тики 44 и 1900) |
| Rust std libm vs glibc | 0 / 2 000 000 |
| крейт `libm` vs glibc | powf 9.7%, sinf 1.1%, cosf 0.29%, atanf 0.18% |

Пример причины (сид 1, тик 2, `t2.c`): pos.y=900, vel=(-3,-11.5) → f32 `MoveBox` даёт 888.500244 → 889; f64 — 888.49999999999955 → 888.

---

## 10. Риски

1. **Моды серверов.** Block-серверы часто крутят модифицированный DDNet (DDNet++, F-DDrace и т.п.) с доп. тайлами/логикой
   (в DDNet++-версиях карт видны нестандартные индексы). Эталон на ванильном DDNet может не совпасть с целевым сервером —
   нужно выяснить софт целевых серверов (не проверено).
2. **Версия.** master ≠ развёрнутая версия; изменения ввода (early input), спидап-тип 29 и т.п. — собирать эталон из тега.
3. **Платформа сервера.** ARM (FMA) / Windows (CRT libm) вероятно дают другие биты — ориентироваться на x86-64 Linux.
4. **Недетерминизм сервера:** телепорты с несколькими выходами (PRNG с секретным сидом), тайминг прихода ввода (сервер
   повторяет последний ввод, если новый не пришёл), серверный tune/`sv_*`-настройки, команды/solo игроков.
5. **Наблюдаемость.** Клиент получает reckoning-ядро раз в 2 тика; `m_FreezeTime`, `ReloadTimer`, `m_FrozenLastTick`,
   `m_MoveRestrictions`, `LastRefillJumps` и др. не передаются напрямую — часть состояния восстанавливается приближённо.
6. **Порядок тика персонажей** должен браться из `m_StrongWeakId`; ошибка меняет исход хук-битв.
7. **Регрессии конкретных версий:** например, в 19.9–20.1 лазер/шотган игрока без персонажа не попадает (#10399 → фикс
   #12878 не влит) — эталон обязан быть той же версии, что и сервер.
8. **Объём механик** (switch/двери/лазеры/драггеры/пикапы) заметно больше, чем в TS; без них на реальных block-картах
   предсказание будет систематически неверным в соответствующих зонах.
