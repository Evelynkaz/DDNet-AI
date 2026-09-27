# DDNet-AI (TS) — планировщик, навигация, env, nn, map, demo: заметки для Rust-порта

Источник: `~/aiddnet/DDNet-AI` (коммит `c3c619d`), Node 24 (V8 13.6). Все ссылки `файл:строка` — относительно `src/`.
Пометка **«не проверено»** — вывод по коду/именам, не подтверждённый прогоном.

Замеры сделаны синтетическим стендом (карта 60×30, пол + полоса фриза, 2 ти, противник — `scriptedAction`),
скрипты в scratchpad (не в репо). Цифры ориентировочные.

---

## 0. Карта модулей и как планировщик вызывается вживую

| Файл | Роль |
|---|---|
| `plan/planner.ts` (2176 стр.) | CEM-планировщик `Planner`, скоринг, поля опасности, эвристики |
| `plan/shield.ts` | «щит»: проверка, что после выбранного ввода есть путь спасения |
| `plan/seal.ts` | `restsInFreeze` (упрощённая баллистика «упадёт ли во фриз»), `sealedIn` (для бота) |
| `plan/throwLines.ts` | фиксированные «броски» хуком для freezeThrow/frozenThrow |
| `plan/route.ts` | тайловый граф движений + Дейкстра (`findRoute`), `RouteRunner`, dead-zone |
| `plan/livePlan.ts` | перенос живого состояния в плановый `SimWorld` |
| `plan/memory.ts` | `FreezeMemory` (карта «здесь замерзали/проходили») |
| `bot/navigate.ts` | `Navigator` (ходьба по BFS-полю, climb хуком, маршруты, crossing) |
| `bot/crossing.ts` | `SwingCrosser` — пролёт через фриз-трубу на хуке с поиском программ в симуляции |
| `bot/wayblock.ts` | захардкоженные WB-зоны карты «Copy Love Box» (+ сдвинутый вариант «JoniTee») |
| `env/obs.ts`, `env/action.ts`, `env/scripted.ts` | наблюдение (214), действие (10), скриптовый бот |
| `nn/*.ts` | RNG (xoshiro128**), MLP, GRU, кодирование параметров |
| `train/humanImitate.ts` | формат датасета имитации + метрика; **тренера нет** |
| `demo/*.ts` | распаковка снапшотов/дельт (не используется), dead-reckoning ядер |
| `map/*.ts` | чтение datafile v3/v4, извлечение game/tele/speedup/front |

Живой вызов: `bot/bot.ts:4694 planAction()`:
1. Плановый мир `planSim = new SimWorld(collision, {svHit:true, respawnDelayTicks:0, infiniteAmmo:true})`, создаётся при смене карты/своего id; `addTee(ownId)` затем `addTee(targetId)` (порядок важен: `order.unshift` → цель обрабатывается ПЕРВОЙ в `step()`), `planner.reset()` (`bot.ts:4698-4706`).
2. `enemyInput = enemyInputFromSnapshot(target)` (`livePlan.ts:5`): `direction` из снапшота, `hook = hookState>0 ? 1:0`, прицел `round(cos/sin(wireAngleRad(angle))*300)`, `jump=0`, `fire=0`.
3. `lag = lagTicks()` (≤ `MAX_LAG_TICKS=6`), `syncPlanningWorld()` (`livePlan.ts:16`): `applyTeeState` обоих, `setHeldInput(own, heldAtSnapshot ?? prevInput)`, `setHeldInput(target, enemyInput)`, затем `lag` тиков `step()` с «in-flight» вводами (уже отправленными, но не применёнными сервером). Легаси-режим `liveTransfer:"legacy"` — `syncPlanningWorldLegacy`.
4. Гранаты в радиусе 600px переносятся `sim.setGrenades(...)`.
5. Сеттеры: `setTravelGoal(pathGoal|trekGoal)`, `setThirdTees` (вызывается всегда; непустой список только при `thirdTeeExposure>0`), `setFrozenBystanders` (замороженные не-друзья ≤160px, `BYSTANDER_PX`), `setSpareBystanders` (ignore / незамороженные друзья / out-of-game / AFK-не-war, ≤ hookLength+64), `setOverrides(wbPlanOverrides)`, `setBand(wbBand)`, `setLiveTick(world.tick)`.
6. `out = planner.decide(sim, ownId, targetId, prevInput, enemyInput)`.
7. Пост-фильтр бота (`bot.ts:4782-4786`): если `out.hook` и (есть spares или мы кого-то держим): если держим spare/друга → `hook=0`; если хук свободен и луч поймает spare раньше цели → `hook=0`.

Решения принимаются на каждый снапшот (25/с, `SNAPSHOT_PERIOD_MS=40`, `bot/cpuLoad.ts:1`), т.е. раз в ~2 тика.

---

## 1. Планировщик (`plan/planner.ts`)

### 1.1 Представление плана

- `PlanStep = { dir: -1|0|1; jump: 0|1; hook: 0|1; fire: 0|1; aim: number }` (`:338`). `aim` — угол в радианах; при `trackAim=true` (дефолт) — **относительный** к пеленгу на противника, пересчитываемому в начале каждого шага роллаута (`:2062-2065`); при `false` — абсолютный.
- План = массив `steps` шагов. Длительность шагов — `stepTicks` из `buildStepTicks(steps, planStep, frontSteps, frontStep)` (`:192-207`):
  - `total = steps*planStep`; `front = clamp(trunc(frontSteps), 0, steps-1)`; если `front==0 || frontStep<=0 || frontStep>=planStep` → все шаги по `planStep`.
  - иначе первые `front` шагов по `fine=max(1,trunc(frontStep))`, остаток `left = total - front*fine` делится на `rest=steps-front` шагов: `base=floor(left/rest)`, последние `extra=left-base*rest` шагов получают `base+1`. Если `left<rest` → равномерно.
  - Дефолт: 9 шагов × 3 тика = 27 тиков (0.54 с).
- Внутри шага ввод держится `stepTicks[s]` тиков. Ввод шага строится `stepToInput(step, prevInputЦепочки, ...)` → `decodeAction(raw, prev)` (сглаживание прицела и т.д., см. §4) — т.е. **исполненный** прицел ≠ плановому aim.
- `StepDist = {pLeft,pRight,pJump,pHook,pFire,aim,aimSpread}` (`:339`) — распределение CEM на шаг.

### 1.2 Все поля конфига и дефолты

`PlannerConfig` (`:35-190`), `PLANNER_DEFAULTS` (`:222-315`). Столбец «исп.» — используется ли в planner.ts (иначе только ботом или мёртвое).

| поле | дефолт | исп. | смысл |
|---|---|---|---|
| steps | 9 | да | шагов в плане |
| planStep | 3 | да | тиков на шаг |
| restAim | false | да | если шаг без хука и без огня — целиться точно в противника (`:1285`) |
| selfFreezeBias | 1.5 | да | множитель штрафа своей заморозки |
| travelWeight | 0.35 | да | вес расстояния до `goal` (px/32) за тик |
| seek | true | нет (бот, `bot.ts:4404`) | |
| memoryWeight | 0 | да | вес `memory.risk` |
| memoryTrust | 0 | да | доверие `memory.safety` в selfHazard |
| pathToTarget | true | нет (бот `pathGoal`) | |
| settledFreezeTicks | 0 | нет (бот) | |
| blockHoldScore | 0 | нет (бот) | |
| deadZoneCost | 0 | да | штраф, если мы в «ловушке» (dead zone), а враг нет |
| enemyDeadZoneBonus | 0 | да | бонус за врага в ловушке |
| frontSteps | 0 | да | сетка шагов |
| frontStep | 2 | да | сетка шагов |
| releaseDeadHook | false | да | отпускать хук в состоянии retract |
| population | 20 | да | сэмплов CEM на итерацию |
| elite | 6 | да | элита |
| iterations | 2 | да | итераций CEM |
| seed | 1 | да | seed RNG |
| wastedHammer | 0.03 | да | штраф за удар без попадания |
| wastedHook | 0.05 | да | штраф за хук, вернувшийся без захвата |
| hammerRangePx | 80 | да | молот разрешён только если дистанция ≤ этого |
| gateHook | true | да | хук только если «достанет» стену/врага |
| gateHammer | true | да | молот только если `hammerWouldHit` |
| airJumpCost | 0 | да | штраф за ранний air-jump |
| jumplessHazardCost | 0.15 | да | штраф близости к фризу без прыжков |
| selfHazardCost | 0.6 | да | штраф своей близости к фризу |
| flipCost | 0.4 | да | штраф смены направления между шагами |
| wallPushCost | 0.05 | да | штраф «давить в стену» |
| commitDecisions | 1 | да | переиспользовать решение N раз |
| opponentModel | "hold" | да | "hold" / "react" / "policy" / "learned" |
| enemyHazardWeight | 2.0 | да | награда за близость врага к фризу |
| frozenWeight | 0.5 | да | награда/штраф за заморозку (за тик) |
| hookHoldWeight | 0.08 | да | держим врага хуком |
| distanceWeight | 0.06 | да | штраф дистанции сверх standoff |
| standoffPx | 250 | да | |
| selfHazardThreshold | 0.55 | да | порог nearness для selfHazard |
| trackAim | true | да | относительный aim |
| openingBook | "classic" | да | "classic"/"wide"/"movement"/"all" |
| launchExposure | 1.0 | да | штраф «меня выбьют молотом во фриз» |
| launchExactReach | 70 | да | порог px для точной симуляции полёта |
| launchExactRiseVy | 4 | да | |
| launchExactWeight | 2 | да | вес точного полёта |
| opponentReadWeight | 0 | да | вес профиля противника (0 ⇒ выключено) |
| routeDistance | false | да* | *считает BFS, но результат НЕ используется (баг, §11) |
| dragExposure | 0.5 | да | штраф «меня протащат хуком через фриз» |
| thirdTeeExposure | 0 | да | то же для третьих ти |
| freezeThrow | 0 | да | кол-во throwLines-сидов (враг не заморожен) |
| jitterCost | 0 | да | доп. штраф частых разворотов |
| flipHoldTicks | 0 | да | |
| flipMargin | 0.6 | да | гистерезис разворота |
| edgeHold | true | да | «не разворачиваться у края фриза» |
| freezeTailWeight | 0.5 | да | терминальный член заморозки |
| sealTicks | 150 | да | терминальный член «запечатан во фризе» |
| frozenTargetSteps | 0 | да | удлинённый горизонт, когда враг заморожен |
| frozenThrow | 0 | да | кол-во frozenThrowLines-сидов |
| bandCost | 0 | да | штраф нахождения в «полосе» WB |
| shield | true | да | щит |
| policySeeds | 0 | да | сидов от GRU-политики |
| policySeedJitter | 0.25 | да | |
| policySeedSteps | 0 | да | |
| noThaw | true | да | не бить молотом замороженного врага, если он «оттает и сбежит» |
| noThawRope | false | да | строгий режим noThaw (+ побеги хуком к нам) |
| hookDragWeight | 0 | да | |
| enemyHazardFromStart | true | да | порог награды от стартовой nearness |
| dragThreat | 0 | да | |
| launchThreat | 0 | да | |
| landingCost | 0 | да | |
| planMargin | 0 | да | держаться прошлого плана |
| opponentMix | false | да | итерации ≥1 с моделью "react" |
| budgetMs | 0 | да | мягкий дедлайн (Date.now) |
| hardMs | 0 | да | жёсткий дедлайн |
| shieldCadence | false | да | длительность удержания щита по каденсу решений |
| explain | false | да | доп. роллаут для статистики (на решение не влияет) |
| liveTransfer | "full" | нет (бот) | |
| planOthers | 0 | нет (бот: сколько третьих ти класть в sim) | |
| targetHold | 400 | нет (бот) | |
| escapeBias | 0 | да | |
| escapeMargin | 1.5 | да | |
| warmShiftElapsed | false | да | |
| hookReleaseCost | 0 | да | |
| valueWeight | 0 | да | |
| hookSeeds | true | да | |
| hookPolish | true | да | |

**Режимы (эффективные значения поверх дефолтов):**
- База бота: `LIVE_PLANNER_CFG = {thirdTeeExposure:0, memoryTrust:0.9, frozenThrow:3}` (`bot.ts:590`), + `plannerCfgNow()` (`bot.ts:2310`): `{budgetMs: LIVE_BUDGET_MS=18, explain:true, ...cfg.plannerCfg}`.
- **normal**: population 20, iterations 2, elite 6, budgetMs 18, hardMs 0, explain true, frozenThrow 3, memoryTrust 0.9.
- **--low-cpu** (`cpuLoad.ts:17 LOW_CPU={budgetMs:6, hardMs:11, commit:2, explain:false, navMs:10, reachChecks:2}`): `budgetMs=min(18,6)=6`, `hardMs=11`, `commitDecisions=max(1,2)=2`, `explain=false`, `shieldCadence=true` (`bot.ts:2312-2322`). Strong в low-cpu запрещён.
- **--strong**: влияет ТОЛЬКО внутри WB-холла, и только если `population < 40`: `WB_PLAN_STRONG = {...WB_PLAN_OVERRIDES, ...STRONG_WB}`, `STRONG_WB={population:40, iterations:3, budgetMs:30, hardMs:36}` (`cpuLoad.ts:19`, `bot.ts:220, 4809-4815`).
- **WB** (бот в холле своей стороны wayblock): `WB_PLAN_OVERRIDES = {noThawRope:true, frozenThrow:3, airJumpCost:0.3, launchExactReach:100}` (`bot.ts:216-218`), либо JSON из env `WB_PLAN`. Применяется `setOverrides`: `Object.assign(cfg, baseCfg, over)` (`planner.ts:912-917`) — `baseCfg` = конфиг конструктора (включая low-cpu бюджеты). `band` передаётся, но `bandCost=0` ⇒ полоса не действует.
- **bold** (выбор «b»): `PLANNER_BOLD={population:64, iterations:3, budgetMs:18}` (`bot.ts:45`), автор пишет «измерен слабее на пять сигм».
- `!try` пресеты (`bot.ts:17-40`): trackaim, launch, both, careful(1.5), careful2(2.0), bold(selfFreezeBias 1.0), nothaw, readsyou(`opponentModel:"learned"`), bothcareful, others2(planOthers 2), oldcopy(liveTransfer legacy), smooth(warmShiftElapsed).
- Прочие env: `WB_FINISH`, `WB_URGENCY`, `TEAM_HELP` (бот).

### 1.3 `decide` / `decideOnce` — точный порядок (`:1021-1316`)

`decide()`: если `frozenTargetSteps > steps` и враг заморожен с `freezeTicksLeft >= 30` (`FROZEN_PLAN_MIN_TICKS`) — временно `cfg.steps = frozenTargetSteps` (try/finally). Затем `syncGrid()` и `decideOnce`.

`decideOnce(world, selfId, enemyId, prev, enemyInput)`:
1. `oppSeed = (oppSeed*1664525 + 1013904223) >>> 0` — ВСЕГДА, первым делом (`:1044`). `thawMemo.clear()`.
2. `me/en = world.getTee()` (копии). Если кто-то отсутствует/мёртв → вернуть `prev` (без обновления прочего).
3. `if opponentReadWeight>0 → profile.observe(me, en, 380)`.
4. Каденс: `nowTick = liveTick>=0 ? liveTick : world.tick`; `liveTick=-1`; `gap=nowTick-lastDecideTick`; если `lastDecideTick>=0 && 0<gap<=50` → push в `decideGaps` (макс 4, FIFO), иначе если `gap!=0` → очистить. `lastDecideTick=nowTick`.
5. `urgent = me.frozen != lastFrozen || en.hookedPlayer == selfId`; `lastFrozen = me.frozen`. Если `!urgent && commitLeft>0 && committed` → `commitLeft--`, lastInfo обнулить, вернуть `maybeRelease(committed)`.
6. `startedMs=Date.now()`. `heldTicks = dirSince<0 ? flipHoldTicks : world.tick - dirSince`.
7. `field=hazardField(col)`, `unfreeze=unfreezeField(col)`, `travel = routeDistance ? travelField(col, en.pos) : null` (не используется), `ensureDeadZone`.
8. `aimAt = atan2(en.y-me.y, en.x-me.x)`. `warmShiftSteps = warmShift(world.tick)` (1, если `!warmShiftElapsed`), `lastSearchTick = world.tick`.
9. `dist = buildDist(trackAim ? 0 : aimAt)`; `saved = world.saveState(saved)`; `predictOpponent(...)`.
10. `best/bestScore=-Inf`, `bestStay/bestStayScore=-Inf` (лучший план с `plan[0].dir === prev.direction`).
11. `carry`: если `planMargin>0` и `warm.length==steps` → `carry = warm.slice(shift) ++ копии последних shift шагов`, оценить.
12. Дедлайны: `hardline = hardMs>0 ? startedMs+hardMs : Inf`; `deadline = min(budgetMs>0 ? Date.now()+budgetMs : Inf, hardline)`; `overCap() = hardline!=Inf && Date.now()>hardline`.
13. Цикл `it = 0..iterations-1` пока `!outOfTime`:
    - `reactThisPass = opponentMix && it>0`; `scored=[]` (**на итерацию, не накопительно**).
    - it==0: книга `seedPlans()`; для каждого (если `best!==null && overCap()` → outOfTime, break) — evaluate, push, обновить best и bestStay (строго `>`).
    - it==0 && !outOfTime: `policySeedPlans()` аналогично.
    - it==0 && (`freezeThrow>0 || frozenThrow>0`) && !outOfTime && !overCap(): `landedThrows()` → push/обновить.
    - `for i<population`: если `deadline!=Inf && (i&3)==3 && Date.now()>deadline` → outOfTime, break (проверка ДО сэмплирования на i=3,7,11,…). `plan=samplePlan(dist)`, evaluate, push, обновить.
    - `candidates += scored.length`; `scored.sort((a,b)=>b.score-a.score)` (V8 TimSort — **стабильная**); `refit(dist, top elite)`.
14. `world.restoreState(saved)`; если `best===null` → `prev`.
15. `flips = best[0].dir !== prev.direction`; `plainHold = flipMargin>0 && bestStay && flips && bestScore-bestStayScore < flipMargin`.
16. edgeHold: если `edgeHold && flips && !me.frozen && !overCap() && freezeGapPx(me.pos) < 96` → `notNow(...)`; если вернул план и его score > bestStayScore → он становится bestStay; `notNowIn=true`.
17. Если `(flipMargin>0 || notNowIn) && bestStay && flips && bestScore-bestStayScore < flipMargin` → `best=bestStay`, `lastInfo.edgeHeld = !plainHold`.
18. `carry && bestScore - carryScore < planMargin` → best=carry.
19. `reactThisPass=false`.
20. `hookPolish && best[0].hook==0 && !overCap()` → `polishRope()`; если вернул — заменить.
21. `escapeBias>0 && !overCap()` → escape-проход (§1.8).
22. `warm = best`; lastInfo (searched, candidates, outOfTime, ms, hookAt = тик первого шага с hook).
23. `explain && !overCap()` → доп. `evaluate` с `trackRollout` ради `selfOut/enemyOut` (на решение не влияет, но трогает `thawScratch`/`thawMemo`).
24. `rest = restAim && best[0].hook==0 && best[0].fire==0`; `aim0 = rest ? aimAt : trackAim ? aimAt + best[0].aim : best[0].aim`.
25. `hookOk = !best[0].hook || hookAlreadyOut || hookAllowed(world, self, enemy, best[0], prev, aim0)`; `lastInfo.gated = best[0].hook && !hookOk`.
26. `swingTargetFrozen=en.frozen; swingRopeOn = me.hookedPlayer==enemyId; swingTarget=en; swingCollision=col`.
27. `chosen = stepToInput(best[0], prev, dist(me,en), hookOk, me.pos, en.pos, en.vel, aim0)`.
28. Щит: `shield && !me.frozen`: `hold=shieldHold()`, `others = Map{enemyId→enemyInput}`; если `!escapeExists(world, self, chosen, hold, others)` → `safer = saferInput(world, self, chosen, hold, others, prev)`; если не null → `chosen=safer`, `lastInfo.shielded=true`.
29. Если `chosen.direction != dirLast || dirSince<0` → `dirLast=chosen.direction, dirSince=world.tick`.
30. `committed=chosen; commitLeft=max(0, commitDecisions-1)`; вернуть `maybeRelease(chosen)` (если `releaseDeadHook` и хук в retract-состоянии → копия с `hook=0`).

`shieldHold()` (`:1318-1326`): `commit=max(1,commitDecisions)`; если `!shieldCadence` → `2*commit`; иначе медиана (нижняя) отсортированных `decideGaps` (или 2, если пусто), `perDecision = clamp(typical, 2, 8)`, `hold = min(16, perDecision*commit)`.

### 1.4 Opening book — `seedPlans` (`:1525-1628`)

Предвычисления:
- `toward = sign(en.x - me.x) || 1`.
- `hazardDir`: по углам `a ∈ [0,1,3,4,5,7]`, `ang=a*π/4` (пропущены строго вниз и вверх), `near = hazardNearness(field, en.x+cos(ang)*96, en.y+sin(ang)*96)`; при `near > bestNear` (старт 0, строго) `hazardDir = cos(ang)>0 ? 1 : -1`. По умолчанию `hazardDir=toward`.
- `n=steps`; `at = trackAim ? 0 : aimAt`; `rel(abs) = trackAim ? wrapAngle(abs - aimAt) : abs` (`wrapAngle` — while-циклы ±2π, `:492`).
- `up = rel(-π/2 + 0.4*toward)`, `away = rel(atan2(0, -toward))`.

Базовая книга «classic» (всегда 5 планов), `s` — индекс шага:
1. `dir = s<3 ? toward : hazardDir; jump = s==2; hook = s>=2; fire = s > n-4; aim=at`
2. `dir = hazardDir; jump=0; hook = s>=1 && s<n/2; fire = s>=n/2 && s<n/2+3; aim=at` (n/2 — дробное: для n=9 hook s=1..4, fire s=5..7)
3. `dir = toward; jump = s%6==0; hook=0; fire = s>2; aim=at`
4. `dir = toward; jump = s==0; hook=1; fire=0; aim=up`
5. `dir = -toward; всё 0; aim=at`

Профиль противника (`read=profile.read()`, `w=clamp(opponentReadWeight,0,1)`, `mix(v,prior)=prior + w*(v-prior)`); при дефолтном `w=0` ни одно условие не срабатывает:
- `mix(hookOpensFirst,0.5) > 0.6` → `dir = s<n/2 ? -toward : toward; jump = s==round(n/2); fire = s>n-4`
- `mix(aggression,0.5) > 0.6` → `dir=0; fire = s>1`; иначе если `< 0.4` → копия плана 3.
- `mix(outOfJumps,0.3) > 0.5` → `dir=toward; fire = s>0`.

hookSeeds (`hookSeeds && en.alive && !en.frozen && !me.frozen && dist<380`), `third = max(2, round(n/3))` (=3):
- если `me.hookedPlayer == enemyId`: (если `warm.length==n`) `warm[min(s+warmShiftSteps, n-1)]` с `hook=1`; затем `{dir:hazardDir, hook:1, aim:at}` на все шаги.
- иначе если `me.hookState ∉ {FLYING(4), GRABBED(5)}`: `{dir:hazardDir, hook:1}` и `{dir:hazardDir, hook: s<third}`.

`openingBook ∈ {"movement","all"}` → `movementSeeds` (`:1630-1666`): `travel = |vel.x|>=1 ? sign(vel.x) : toward`; `upAhead=rel(atan2(-1, 0.8*travel))`, `upAheadSteep=rel(atan2(-1,0.4*travel))`, `ceiling=rel(atan2(-1,0.15*travel))`, `upBehind=rel(atan2(-1.2,-0.5*travel))`:
- `{dir:travel, jump:s==3, hook:s<3, aim: s<3?upAhead:at}`
- `{dir:travel, jump:s==4, hook:s<4, aim: s<4?upAheadSteep:at}`
- `{dir: s<2?travel : s<4?-travel : travel, jump:s==6, hook:s<6, aim: s<6?ceiling:at}`
- `{dir: s<1?travel:-travel, jump:s==4, hook:s<4, aim: s<4?upBehind:at}`
- если `edgeStep=stepsToEdge()` не null и `< n-1`: `{dir:travel, jump: s==edgeStep || s==edgeStep+3}`
- `{dir:travel}`.
`stepsToEdge` (`:1668-1683`): `feetY = y+14+4`; нужен solid под ногами; для `k=1..8` первый тайл по `travel`, где центр `((tx+k*travel)*32+16, feetY)` не solid → `lipX = travel>0 ? (tx+k)*32 : (tx-k+1)*32`, `px=|lipX-x|`, `speed=max(|vx|, groundControlSpeed*0.6=6)`, вернуть `stepAtTick(px/speed)`. `stepAtTick(t)`: накопленные границы шагов; ближайшая к t (при равенстве — более поздняя, `<=`), возвращает `i+1`; стартовый кандидат 0 с `gap=|t|`.

`openingBook ∈ {"wide","all"}` — ещё 10 (`:1605-1626`):
1. `{dir:hazardDir, hook:1}`  2. `{dir:hazardDir, hook:s<third}`  3. `{dir:hazardDir, jump:s==1, hook:s<2*third}`
4. `{dir:0, hook:s<2*third, fire:s>=third}`  5. `{dir: s<third?0:hazardDir, hook:s<2*third, fire:s>=third}`
6. `{dir:toward, fire:s>=1}`  7. `{dir: s<third?toward:-toward, jump: s==0||s==2, fire: s>=third+1}`
8. `{dir:-toward, jump: s==0||s==3}`  9. `{dir:-toward, hook:s<third, aim:away}`  10. всё 0.
(все остальные поля 0, aim=at, если не указано).

Порядок книги: 5 базовых → профильные → hookSeeds → movement → wide. Замер: в среднем 6.9 книжных планов/решение.

### 1.5 CEM: распределение, сэмплинг, рефит

`buildDist(aimAt)` (`:1455-1475`), для каждого шага `s`:
- `w = warm ? warm[min(s+warmShiftSteps, warm.length-1)] : null`.
- без warm: `pLeft=0.33, pRight=0.33, pJump=0.15, pHook=0.3, pFire=0.2, aim=aimAt(0 при trackAim), aimSpread=1.2`.
- с warm: `pLeft = w.dir==-1 ? 0.7 : 0.15`, `pRight = w.dir==1 ? 0.7 : 0.15`, `pJump = w.jump ? 0.7 : 0.1`, `pHook = w.hook ? 0.7 : 0.15`, `pFire = w.fire ? 0.6 : 0.15`, `aim=w.aim`, `aimSpread=0.9`.

`samplePlan(dist)` (`:1477-1490`), порядок вызовов RNG на шаг строго: `r=nextFloat()` → `dir = r<pLeft ? -1 : r<pLeft+pRight ? 1 : 0`; `nextFloat()<pJump`; `nextFloat()<pHook`; `nextFloat()<pFire`; `aim = d.aim + nextGaussian()*aimSpread`.

`refit(dist, elites)` (`:1492-1523`), если элиты есть, для каждого шага: `pLeft=0.1+0.8*left/n`, `pRight=0.1+0.8*right/n`, `pJump=0.05+0.9*jump/n`, `pHook=0.05+0.9*hook/n`, `pFire=0.05+0.9*fire/n`, `aim=atan2(Σsin/n, Σcos/n)` (порядок суммирования — порядок элит), `aimSpread=max(0.25, aimSpread*0.7)`.

Итог по дефолту: итерация 0 = книга (5–7) + 20 сэмплов (+ до 3 frozenThrow в live), итерация 1 = 20 сэмплов. Замер: ≈47 кандидатов и ≈48 вызовов `evaluate` на решение, ≈1340 `world.step()` на решение.

### 1.6 Дополнительные источники кандидатов и пост-обработка

**policySeedPlans** (`:1760-1805`, только при `seedPolicy` и `policySeeds>0`; в live не задаётся): GRU-политика прогоняется `steps` шагов (self ввод по decodeAction, враг `predicted[step]` или `enemyInput`), aim = `atan2(target)` минус пеленг при trackAim, `fire&1`; память GRU сохраняется/восстанавливается; мир → `saved`. Если `0<policySeedSteps<len` — хвост из `samplePlan(buildDist(...))` (**тратит RNG**). Плюс `policySeeds-1` копий со сдвигом aim `±policySeedJitter*ceil(i/2)` (чередование знака), кроме хвоста.

**landedThrows** (`:1358-1406`, `throwLines.ts`):
- frozen-ветка: `frozenThrow>0 && frozenThrowWorthTrying && en.freezeTicksLeft>=30` (`!me.frozen && en.frozen && en.alive && sep<=380`). Для каждой линии `frozenThrowLines(steps, at)` с `trackRollout`: оставить, если `rolloutEnemySealed && rolloutSelfOut==0`; сортировать по score убыв.; взять `frozenThrow` штук. Возврат (freezeThrow-ветка не выполняется).
- иначе `freezeThrow>0 && throwWorthTrying` (оба не заморожены, враг жив, `sep<=380`, `enemyHazardNearness >= 0.4`): `throwLines`, оставить при `rolloutEnemyOut >= 5 (THROW_LANDED_TICKS)` и `gain = enemyOut-selfOut > 0`; сорт `b.gain-a.gain || b.score-a.score`; взять `freezeThrow`.
- `throwLines(steps, at)`: `releases=[2, max(3,round(steps/3)), max(5,round(2*steps/3)), steps]`, `mid=max(3,round(steps/3))`; для `dir ∈ [-1,1]`: 4 линии `{dir, hook: s<r}`; `{dir, jump:s==1, hook:s<mid+1}`; `{dir, hook:s<mid, fire:s>=mid}` → 12 линий.
- `frozenThrowLines`: `throwLines` + для `dir∈[-1,0,1]`: для `h∈[2,3,5]` при `h+1<steps`: `{dir, jump:s==h-1, hook:s<h, fire: s==h||s==h+1}`; и `{dir, jump: s==0||s==third, hook:s<third+1}` при `third=max(3,round(steps/3))` → 24 линии при steps=9.

**notNow** (edgeHold, `:1333-1356`): `later` = план с направлениями, задержанными на шаг (`dir[0]=prev.direction`, `dir[i]=turning[i-1].dir`), `kept` = только шаг 0 с `prev.direction`. С `trackGap` оценить `turning` → `turnGap` (минимум `freezeGapPx` по роллауту; 0, если заморожен/мёртв), затем `later`, `kept`; взять лучший по score; если его `gap < turnGap - 4 (EDGE_NEARER_PX)` → null.

**polishRope** (`:1408-1440`): только если оба живы и не заморожены; `holding = me.hookedPlayer==enemy`; `free = hookState==IDLE`; нужно `holding || (free && dist<380)`. `bearing=atan2(en-me)`, `throwAim = trackAim?0:bearing`. Если `!holding && gateHook && !hookWouldReach(executedAim({...best[0],hook:1}, prev, bearing))` → null. Для `k∈[2,4,n]` (пропуск `k>n`): план = best, где шаги `i<k` получают `hook=1` и `aim = holding ? st.aim : throwAim`; принять, если `score >= bestScore` и лучше предыдущего полиша.

**escapeBias** (`:1228-1259`, дефолт 0): трекинг-роллаут best; если `rolloutSelfOut>0`: временно `selfFreezeBias *= escapeBias`, 20 (`population`) сэмплов из `buildDist` (RNG!), лучший `escape`; вернуть bias; `fair = evaluate(escape)`, ещё трекинг; принять, если `rolloutSelfOut==0 && fair >= bestScore - escapeMargin`.

### 1.7 `evaluate` (`:1993-2162`) — точный роллаут

1. Штраф разворотов: если `flipHoldTicks<=0 || jitterCost<=0`: для каждого шага `if st.dir != prevDir → score -= flipCost`, `prevDir=st.dir` (старт `prev.direction`). Иначе: `run=min(hold, max(0,heldTicks))`; при смене `score -= flipCost + jitter*(1-run/hold); run=stepTicks[i]`, иначе `run=min(hold, run+stepTicks[i])`.
2. `heldEnemy = hookReleaseCost>0 && world.coreOf(self)?.hookedPlayer==enemy`. `oppRng = new Rng(oppSeed)` (новый на каждый evaluate ⇒ одинаковый шум противника во всех кандидатах решения). `oppInput=enemyInput`. Сброс трекеров.
3. `enNearAtStart = hazardNearness(field, en0)`; `drag={prevEnemyNear, startEnemyNear, startedInDead = inDead(dead, me0)}`. `prevJumpsLeft = me0.jumpsLeft`, `groundJumpAt=-1`, `rolloutTick=0`.
4. Для каждого шага `s`:
   - прочитать me/en (буферы), `enemyDist`; `aim = plan[s].aim (+ atan2(en-me) при trackAim)`.
   - `swingTargetFrozen/swingRopeOn/swingTarget/swingCollision` ← текущее состояние.
   - `hookOk = !plan[s].hook || hookAlreadyOut(world) || hookAllowed(world, …, plan[s], input, aim)` (`input` — ввод предыдущего шага).
   - `input = stepToInput(plan[s], input, enemyDist, hookOk, me.pos, en.pos, en.vel, aim)`.
   - Противник: `if reactThisPass || opponentModel=="react"` → `oppInput = scriptedAction(world, enemy, self, oppInput, oppRng)` (раз на шаг); иначе если `predicted.length>0` → `predicted[min(s, len-1)]`; иначе остаётся `enemyInput`.
   - `stepTicks[s]` тиков: `rolloutTick++`; `setInput(self, input)` (или копия с `hook=0`, если `releaseDeadHook && hook && hookIsDead`); `setInput(enemy, oppInput)`; `events=world.step()`; затем по `meNow`:
     - air-jump: если `(jumped&2)==0 && jumpsLeft<prevJumpsLeft` → `groundJumpAt=rolloutTick`; если `prevJumpsLeft>0 && jumpsLeft==0 && !frozen`: `gap = groundJumpAt<0 ? 11 : rolloutTick-groundJumpAt`; `if gap<11: score -= airJumpCost*(1-gap/11)`; `groundJumpAt=-1`. `prevJumpsLeft=jumpsLeft`.
     - `hookState==FLYING(4)` → `hookWasFlying`; `==GRABBED(5)` → `hookGrabbed`.
     - `hookReleaseCost>0`: если держали врага и отпустили, а враг жив и не заморожен → `-hookReleaseCost`.
     - если `hookWasFlying && !hookGrabbed && 1 <= hookState < 4` → `score -= wastedHook`, `hookWasFlying=false`.
     - если `hookState <= 0` → сброс обоих флагов.
     - `trackRollout`: `enemyOut++` если враг заморожен/мёртв, `selfOut++` аналогично.
     - `trackGap`: `rolloutMinGap = min(…, frozen||dead ? 0 : freezeGapPx(me))`.
     - `score += scoreTick(...) * (1 - s/(plan.length*2))` — дисконт по ИНДЕКСУ ШАГА (n=9: 1, 0.944, …, 0.556).
5. Терминальные члены (`tail = 1 - (n-1)/(2n)`):
   - `valueWeight!=0 && valueNet` → `+ valueWeight*tail*valueNet.forward(encodeObs(self,enemy))[0]`.
   - `landingCost>0` и я жив/не заморожен → `- landingCost*flightEndsInHazard(me.pos, me.vel)`.
   - `freezeTailWeight>0`: `w = freezeTailWeight*frozenWeight*tail` (дефолт 0.5·0.5·0.5556=0.1389);
     - я заморожен → `- w*selfFreezeBias*me.freezeTicksLeft`;
     - `enSealed = sealTicks>0 && en.frozen ? restsInFreeze(en.pos, en.vel) : 0`;
     - враг заморожен → `+ w*(noThaw && enSealed>0 ? 150 : en.freezeTicksLeft)`;
     - `sealTicks>0`: я заморожен → `- w*selfFreezeBias*sealTicks*restsInFreeze(me)`; `+ w*sealTicks*enSealed`.
   - `trackRollout` → `rolloutEnemySealed = en && (!en.alive || (en.frozen && restsInFreeze(en)>0))`.
6. `world.restoreState(saved)`; вернуть score.

### 1.8 `scoreTick` (`:518-605`) — все члены за тик

`me/en` читаются в модульные буферы; если нет — `-1000`.
| член | формула |
|---|---|
| враг мёртв | `+15` (каждый тик) |
| я мёртв | `-15` (каждый тик) |
| враг заморожен | `+frozenWeight` (0.5) |
| я заморожен | `-frozenWeight*selfFreezeBias` (0.75) |
| band | `band && bandCost>0 && !me.frozen && me в [x0,x1]×[y0,y1]` → `-bandCost` |
| держу врага | `me.hookedPlayer==enemy` → `+hookHoldWeight` (0.08) |
| враг держит меня | `-hookHoldWeight*0.75` (0.06) |
| события | `hammerFire` от меня с `hits==0` → `-wastedHammer`; `hammerHit` от врага по мне → `-0.3`; `death` врага → `+15`; `death` моя → `-15` |
| заморожен рядом с unfreeze | я заморожен → `+0.08*hazardNearness(unfreeze, me)`; враг заморожен → `-0.06*hazardNearness(unfreeze, en)` |
| давлю в стену | `me.direction!=0 && |me.vel.x|<0.2 && !me.frozen` → `-wallPushCost` |
| без прыжков у фриза | `jumplessHazardCost>0 && !me.frozen && me.jumpsLeft==0` → `-0.15*hazardNearness(field, me)` |
| враг у фриза | `enNear = nearness(en)`; `enFloor = enemyHazardFromStart ? max(0.3, startEnemyNear) : 0.3`; `enNear>enFloor` → `+enemyHazardWeight*(enNear-enFloor)` |
| hookDrag | `hookDragWeight>0 && держу && !en.frozen && enNear>prevEnemyNear` → `+w*(enNear-prev)`; затем `prevEnemyNear=enNear` (всегда) |
| я у фриза | `meNear>0.55` → `-selfHazardCost*(meNear-0.55)*(1-trusted)`, `trusted = memory && memoryTrust>0 ? memoryTrust*memory.safety(me) : 0` |
| выбьют молотом | `sep=dist(me,en)`; `launchExposure>0 && оба не заморожены && sep<96`: `exact = launchExactReach>0 && sep<launchExactReach && (me.y<en.y || (launchExactRiseVy>0 && me.vy < -launchExactRiseVy))`; exact → `-(launchExactWeight>0 ? launchExactWeight : launchExposure)*launchFlightLandsInHazard(me, en, sep, me.vel)`; иначе `-launchExposure*launchLandsInHazard(me, en, sep)` |
| протащат хуком | `dragExposure>0 && оба не замор. && sep<380` → `-dragExposure*dragCrossesHazard(me, en, sep)` |
| угроза врагу | `dragThreat>0` → `+dragThreat*dragCrossesHazard(en, me, sep)`; `launchThreat>0 && sep<96` → `+launchThreat*launchLandsInHazard(en, me, sep)` |
| третьи ти | `thirdTeeExposure>0 && !me.frozen`: для каждого `t` с `1<=d<380` → `-w*dragCrossesHazard(me, t, d)` |
| дистанция | `-distanceWeight*max(0, sep-standoffPx)/32` |
| цель пути | `goal` → `-travelWeight*dist(me,goal)/32` |
| память | `memoryWeight>0 && me.alive && !me.frozen` → `-memoryWeight*memory.risk(me)` |
| dead zone | `dead && !startedInDead && (deadZoneCost>0 || enemyDeadZoneBonus>0)`: `meDead=me.alive && inDead(me)`, `enDead`; `meDead && !enDead` → `-deadZoneCost`; `enDead && !meDead` → `+enemyDeadZoneBonus` |

### 1.9 Геометрические эвристики

- `hazardNearness(field,x,y)` (`:500-507`): `tx=floor(x/32), ty=floor(y/32)`; вне карты → 0; `d=dist[...]`; `d>=20` → 0; иначе `(20-d)/20` (`HAZARD_HORIZON_TILES=20`).
- `dragCrossesHazard(at, from, sep)` (`:610-621`): единичный вектор at→from; точки на 32, 64, 96px; если `px>=sep` → 0; solid → 0; freeze/death → 1; иначе 0.
- `launchLandsInHazard(at, from, sep)` (`:623-639`): `h = (at-from)/sep` (или `(0,-1)` при sep=0); `b=(hx, hy-1.1)`, нормировать (`hypot || 1`); точки на 96, 160, 224px от `at`: solid → 0; freeze/death → 1.
- `launchFlightLandsInHazard(at, from, sep, vel)` (`:650-695`): `k=hammerStrength=1`; `vx = vel.x + k*10*bx/bl`; `vy = vel.y + k*(-1 + 10*by/bl)`; `grounded` = solid в `(x±14, y+14+5)`; 50 тиков: `vy += 0.5`; `vx *= grounded ? 0.5 : 0.95`; `grounded=false` (только первый тик); `speed=hypot(vx,vy)*50`; `ramp = speed<550 ? 1 : 1/pow(1.4, (speed-550)/2000)`; `n=max(1, ceil(max(|vx|,|vy|)/16))`; для `i<n`: `sx=vx*ramp/n` → если ≠0: `f=freeFraction(x,y,sx,0)`, `x+=sx*f`, `f<1 → vx=0`; `sy=vy/n` (**без ramp**) → если ≠0: `f=freeFraction(x,y,0,sy)`, `y+=sy*f`, `f<1 → landed = vy>0; vy=0`; freeze/death в точке `(x,y)` → 1; `landed` → 0. Конец → 0.
- `freeFraction(x,y,dx,dy)` (`:697-711`): если бокс 28×28 в `(x+dx,y+dy)` свободен (`testBox`) → 1; иначе 5 шагов бинпоиска `[lo=0,hi=1]`, вернуть `lo`.
- `flightEndsInHazard(pos, vel)` (`:778-790`): `grounded` как выше; для `t∈[6,12,18,24]`: `x=pos.x+vx*t`, `y = grounded ? pos.y : pos.y + vy*t + 0.5*0.5*t²`; solid → 0; freeze/death → 1.
- `freezeGapPx(x,y)` (`:797-815`): по тайлам `±3` вокруг `(floor(x/32), floor(y/32))`, если в центре тайла freeze/death → расстояние от точки до AABB тайла (`hypot(max(left-x,0,x-right), …)`); минимум, кап 96.
- `ropeCatchAlong(from, dir, at)` (`:642-648`): `along = r·dir`; вне `[0,380]` → Inf; `|r×dir| <= 34 (PHYSICAL_SIZE+6)` → along, иначе Inf.
- `hookWouldReach(angle)` (`:1733-1758`): луч `me → me+dir*380` через `intersectLineHook`; попал в стену без флага NOHOOK → true. Иначе враг жив: `wallDist` (до точки попадания или 380); `along, perp` врага; `along ∉ [0,380]` → false; `along > wallDist` → false; `perp <= 56` → true; иначе упреждение `lead = en + vel*8 - me`: `leadAlong ∈ [0, min(380, wallDist)]` и `perp_lead <= 56`.
- `hookAllowed` (`:1713-1718`): если `!gateHook && spares пусто` → true; `angle = executedAim(step, prev, aim)`; `gateHook && !hookWouldReach` → false; иначе `!ropeCatchesSpare(angle)`.
- `ropeCatchesSpare` (`:1720-1731`): `stop` = дистанция до стены (или 380), `min` с `ropeCatchAlong` цели (если жива); true, если какой-то spare ловится раньше `stop`.
- `executedAim(step, prev, aim)` (`:1976-1991`): raw с dir/jump шага, hook=-1, fire=-1, weapon hammer, `(cos aim, sin aim)` → `decodeAction(raw, prev, aimProbeOut)` → `atan2(targetY, targetX)` — т.е. учитывает сглаживание и округление до целых на радиусе 300.
- `hammerWouldHit(me, en, enVel, tx, ty)` (`:1858-1866`): `len=sqrt(tx²+ty²)`, `<1e-6` → false; `s = me + (t/len)*21`; `reach = 14+28 = 42`; `hypot(en-s)<42` или `hypot(en+2*vel - s)<42`.

### 1.10 `stepToInput` и запреты молота/хука (`:1917-1974`)

`raw[0..2] = dir one-hot ±1` (left,none,right), `raw[3]=jump?1:-1`, `raw[4] = hook && hookOk ? 1 : -1`, `raw[6]=1, raw[7]=-1` (молот), `raw[8]=cos(aim)`, `raw[9]=sin(aim)`.
Молот (`fire`) разрешается цепочкой:
1. `canSwing = hammerRangePx<=0 || enemyDist <= hammerRangePx (80)`.
2. `gateHammer`: `dry = decodeAction(raw с fire=-1, prev)` → `canSwing = hammerWouldHit(me, en, enVel, dry.target)`.
3. `noThaw` и враг заморожен (`swingTargetFrozen`): если `noThawRope && swingRopeOn && step.hook` → нельзя; иначе `canSwing = !thawEscapable(col, swingTarget, me.pos)`.
4. frozenBystanders: для каждого в пределах 144px (`96*1.5`): если молот попадёт и `launchFlightLandsInHazard(col, b, me, max(1,d), bv) == 0` (т.е. выбьет его в безопасность = разморозит друга/постороннего «на волю») → нельзя.
5. spares (≤144px): если молот попадёт → нельзя.
`raw[5] = fire && canSwing ? 1 : -1` → `decodeAction(raw, prev)`.

**noThaw / `thawEscapable(col, en, mePos)`** (`:1868-1915`): мемо-ключ `${strict?"s":""}${round(en.x/4)},${round(en.y/4)},${round(en.vx)},${round(en.vy)},${round((en.x-me.x)/8)},${round((en.y-me.y)/8)},${en.jumpsLeft},${en.freezeTicksLeft>40?1:0}` (JS `Math.round`!). Скретч-мир `thawScratch` (один на коллизию, **живёт между решениями**, `tick` не сбрасывается), ти 0 создаётся один раз в первой позиции врага; ти 1 (в strict) добавляется/удаляется. Push молота: `sep=max(1,dist)`, `h=(en-me)/sep`, `bl=hypot(hx, hy-1.1)||1`, `push=(10*hx/bl, -1+10*(hy-1.1)/bl)`. Для каждого побега (9 из `THAW_ESCAPES` или 15 в strict = + `ropeEscapes`): `applyTeeState(0, {...en, id:0, hookState:0, hookedPlayer:-1})`, `setHeldInput(0, emptyInput())`, (strict: ти 1 = blank живой в mePos), `applyForce(0, push)`, `unfreeze(0)`, 40 тиков ввода побега; если ни разу не мёртв/заморожен → escapable=true (выход).
- `THAW_ESCAPES` (`:723-754`), все 40 тиков, база `target=(0,-300)`: [стоять], [dir -1], [dir +1], [dir d∈{0,-1,1}, jump на t=0 и t=6] ×3, [dir ax∈{0,-1,1}, jump на чётных t, hook=1, target=(ax*200,-300)] ×3.
- `ropeEscapes(dx,dy)` (`:756-774`): `d∈{0,-1,1}` × `jumps∈{false,true}`: hook=1, jump чётные t при jumps, target `(round(dx)||1, round(dy))` (к нам).

### 1.11 Поля (BFS)

- `bfsField(col, isSource)` (`:354-385`): Int32 `dist` = `0x3fffffff`; источники — тайлы game-слоя по предикату, `dist=0`, очередь в порядке row-major; 4-соседство в порядке `(+1,0),(-1,0),(0,+1),(0,-1)`; стена — только `TILE_SOLID(1)` и `TILE_NOHOOK(3)`; через фриз/смерть проходит. Метрика — число шагов по 4-связной решётке (тайлы).
- `hazardField` (`:454-490`): источники `TILE_FREEZE(9)` и `TILE_DEATH(2)` (НЕ `DFREEZE 12`, НЕ `LFREEZE 144`, НЕ front-слой). Кэш `WeakMap` по объекту `Collision`.
- `unfreezeField` (`:346-352`): источники `TILE_UNFREEZE(11)`, кэш.
- `travelField(col, fromX, fromY)` (`:389-425`): BFS от одного тайла (`trunc(x/32)` с клампом), непроходимы solid/nohook/freeze/death; scratch-массивы общие на коллизию (**возвращается ссылка на scratch** — перезаписывается следующим вызовом; `navigate.fieldTo` копирует).
- `travelDistance` / `travelDistanceSmooth` (`:427-452`) — экспортированы, нигде не используются (`UNREACHABLE_TILES=200`, билинейная интерполяция с центрами тайлов).

### 1.12 `seal.ts`

- `touchesFreeze(x,y)` (`seal.ts:13-17`): freeze/death в центре, либо death в 4 углах со смещением `28/3`.
- `restsInFreeze(col, pos, vel)` (`:19-67`) — **упрощённая** баллистика (не настоящая физика), до 60 тиков (`REST_TICKS`):
  - `grounded` = solid в `(x-13, y+15)` или `(x+13, y+15)`; `grounded && vy>=0` → break.
  - `vy += 0.5; vx *= 0.95; nx=x+vx`; если solid в `(nx + sign(vx)*14, y)` → `vx=0`, иначе `x=nx`.
  - `ny=y+vy`; если `vy>0` и solid в `(x∓13, ny+14)`: `y = floor((ny+14)/32)*32 - 14 - 0.01; vy=0`; без tele-слоя → break; иначе `through()`: tele-in/evil-in → перенос в первый tele-out (evil обнуляет скорость), результат 1 → continue; checkpoint-tele → вернуть 0 из функции; 0 → break.
  - иначе: `vy<0 && solid(x, ny-14)` → `vy=0`, иначе `y=ny`.
  - death в `(x,y)` → 1; tele и `through()<0` → 0.
  - в конце `touchesFreeze(x,y) ? 1 : 0`.
- `sealedIn(world, id, state, held)` (`:83-107`) — только бот: удаляет всех прочих ти, 90 тиков (`SEAL_TICKS`) с `held` (если заморожен ≥90 тиков) или с 4 побегами (`held` + jump+hook вверх с ax∈{0,-1,1}, jump только на чётных t); если хоть один побег закончился незамороженным/не во фризе → false.

### 1.13 `shield.ts`

- `escapes(vx, aimX, aimY)` (`shield.ts:10-32`): `brakeDir = |vx|<0.5 ? 0 : -sign(vx)`; если 0: `[стоп, прыжок]` + `[jump, hook, target(0,-300)]` = 3; иначе `[стоп, прыжок, тормоз, тормоз+прыжок]` + для `ax∈{0, brake}`: `{dir:brake, jump:1, hook:1, target:(ax*150,-300)}` = 6. Нехуковые используют прицел выбранного ввода.
- `escapeExists(world, self, input, holdTicks, others)` (`:34-72`): `start=saveState()` (новая аллокация); `holdTicks` тиков `input` + others; смерть/заморозка → false. `afterHold=saveState()`. `pressTick = input.jump ? 1 : 0`. Для каждого побега: restore, 36 тиков (`ESCAPE_TICKS`) с `jump` только на `t==pressTick` (для прыжковых), провал при смерти/заморозке; затем `settlesSafe`: до 90 тиков «coast» (без направления, `hook=esc.hook`, тот же target), выход true, если `standing` (`|vy|<0.5` и solid в `(x-13, y+16)` или `(x+13, y+16)`); после 90 тиков — true, если жив и не заморожен. `finally restoreState(start)`.
- `saferInput(world, self, input, hold, others, sentAim)` (`:95-111`): `base = sentAim` (prev), если ненулевой, иначе input; `from=atan2(base)`; для каждого побега: `d = wrap(atan2(alt.target) - from)`, `a = from + clamp(d, ±1.5)`; кандидат = `{...input, direction, jump, hook из alt, target=round(cos a*300), round(sin a*300)}` (fire сохраняется); первый прошедший `escapeExists` возвращается.

### 1.14 Модели противника

| модель | когда | что делает |
|---|---|---|
| "hold" (дефолт) | всегда в live | `predicted=[]`, `oppInput = enemyInput` на весь роллаут: direction/hook/прицел из снапшота, без прыжка и огня |
| "react" | `opponentModel=="react"` или `opponentMix && it>0` | `scriptedAction(world, enemy, self, oppInput, oppRng)` в начале КАЖДОГО шага; `oppRng` пересоздаётся из `oppSeed` в каждом `evaluate` |
| "policy" | `opponentModel=="policy"` и задан `setOpponentPolicy` (в репо не задаётся) | GRU прогоняется `steps` шагов от лица врага (self держит `prev`), `predicted[s]` |
| "learned" | `!try readsyou` + загружен `opponent.json` | `predictedDir` (`:1807-1822`): вход 224 = `encodeObs(world, enemy, self)` (214) + `encodeHumanTarget(enemyInput, fire&1, en.activeWeapon)` (10); argmax `out[0..2]` → `dir ∈ {-1,0,1}`; `predicted=[{...enemyInput, direction:dir}]` на все шаги |

`opponent.json` загружается всегда (`start.mjs:289-296`, `dummyWorker.ts:17`), но **в дефолтной конфигурации не используется**.

### 1.15 Прочее состояние планировщика

`reset()` (`:991-1011`): warm, committed, commitLeft, lastDecideTick, decideGaps, `rng = new Rng((seed + seedOffset)>>>0)`, `oppSeed = (1+seedOffset)>>>0`, lastFrozen, lastSearchTick, warmShiftSteps=1, dirSince=-1, dirLast=0, heldTicks=0, profile.reset(), opponentPolicy.reset(), saved=undefined. **Не сбрасываются**: `thawScratch`, модульные кэши полей, `predicted`, `travel`.

`setSearchSeed(n)` — `seedOffset=n`, reset (нигде не вызывается).

`OpponentProfile` (`bot/opponentProfile.ts`): EMA-частоты (полураспад 150 / 750 решений), `read()` = `prior + confidence*(value-prior)`; используется только при `opponentReadWeight>0`.

`FreezeMemory` (`plan/memory.ts`): `Float32Array cells/passes`; `safety = good>0 ? (good/(good+15*bad))*(good/(good+10)) : 0`; `risk = v/(1+v)`; `note()` +1 в центр и +0.4 в 8 соседей; сохранение с затуханием 0.97. В live `memoryTrust=0.9` ⇒ `memory.safety()` влияет на selfHazard. Для паритета нужно подавать тот же массив (Float32!).

---

## 2. Детерминизм

### 2.1 RNG (`nn/rng.ts`) — точно

- Сидирование `splitmix32(seed)`: `state = (state + 0x9e3779b9)>>>0; z=state; z = imul(z ^ (z>>>16), 0x21f0aaad)>>>0; z = imul(z ^ (z>>>15), 0x735a2d97)>>>0; z = (z ^ (z>>>15))>>>0` → `s0,s1,s2,s3` по порядку.
- `nextU32()` = **xoshiro128\*\***: `result = rotl(imul(s1,5)>>>0, 7)*9 (imul)>>>0`; `t = s1<<9`; `s2^=s0; s3^=s1; s1^=s2; s0^=s3; s2^=t; s3=rotl(s3,11)`. Всё u32 — в Rust `wrapping_mul`, `rotate_left`.
- `nextFloat() = nextU32() / 4294967296` (f64, 32 бита мантиссы).
- `nextGaussian()`: Box–Muller с запасом: если `haveSpare` → вернуть spare. Иначе `u`: повторять `nextFloat()` пока `u==0`; `v` аналогично; `mag = sqrt(-2*ln(u))`; `spare = mag*sin(2π v)`; вернуть `mag*cos(2π v)`. **Spare переносится между вызовами и между решениями** (сбрасывается только новым `Rng`).

### 2.2 Все источники случайности

1. `this.rng` (seed = `cfg.seed + seedOffset`, дефолт 1): `samplePlan` (population×iterations), хвост `policySeedPlans` (если `policySeedSteps`), escapeBias-сэмплы. Порядок вызовов: dir, jump, hook, fire, gaussian — на шаг.
2. `oppSeed` — LCG `x*1664525+1013904223 mod 2^32` на каждый `decideOnce` (в JS `x*1664525` < 2^53 ⇒ точно; в Rust `wrapping_mul/wrapping_add` на u32 эквивалентно). `new Rng(oppSeed)` в каждом `evaluate` — нужен только для "react"/opponentMix.
3. `Math.random` в планировщике/env/nn/core/navigation **не используется** (только UI `webView.ts`, `page.js`). Rng бота с `Date.now()^pid` (`bot.ts:726`) — только для скриптового мозга; `wanderRng` — для блуждания.

### 2.3 Зависимость от часов

`planner.ts`: `Date.now()` в `:1072` (старт), `:1107` (deadline), `:1110` (`overCap`), `:1167` (проверка deadline каждые 4 сэмпла), `:1264` (`lastInfo.ms`). `overCap()` дополнительно гейтит: книгу/политику/броски, `notNow` (edgeHold), `polishRope` (hookPolish), escapeBias, `explain`, финальный `hookAllowed` не гейтится. `performance.now` в планировщике нет. `crossing.ts:195,386` — `Date.now()` для бюджета поиска программ (`crossBudgetMs`; 0 в normal, 10 мс в low-cpu).

### 2.4 Можно ли запустить TS детерминированно

**Да, без патча:** `new Planner({ ...cfg, budgetMs: 0, hardMs: 0 })` ⇒ `deadline=hardline=Infinity`, `overCap()≡false`, всё решает фиксированное число итераций. Проверено: два прогона на 200 решений (и с live-конфигом `frozenThrow:3, memoryTrust:0.9, explain:true`) дают одинаковый md5 последовательности вводов; с `budgetMs:3` три прогона — три разных хеша.

Что нужно для сравнения TS↔Rust «решение-в-решение» (минимально):
- Не трогая исходники: обёртка-харнесс, которая monkeypatch'ит `Planner.prototype.decide` / `SimWorld` и дампит на каждом решении: полный `SimState` (включая поля, НЕ входящие в `saveState`: порядок `order` и `byId`, `pendingEvents`), `prev`, `enemyInput`, все сеттеры (goal, thirds, bystanders+vels, spares+vels, band, overrides, liveTick), состояние `Rng` (s0..s3, haveSpare, spare — поля «private» только в TS, в рантайме доступны), `oppSeed`, `warm`, `committed/commitLeft`, `decideGaps`, `lastFrozen`, `dirSince/dirLast`, `thawScratch` (весь SimState).
- Если нужна правка в живом боте (снятие телеметрии с реальных матчей): `bot.ts:2311` заменить `budgetMs: LIVE_BUDGET_MS` на значение из env (например `PLAN_BUDGET`, 0 = фиксированные итерации) и аналогично `LOW_CPU.hardMs` в `:2317`; 2–3 строки.
- Для диагностики расхождений полезна 1 строка: `this.lastInfo.score = bestScore` (после `:1259`) и опционально хук-коллбэк на каждый `evaluate` (plan, score).

### 2.5 Скрытые зависимости от истории (опасно для паритета)

- `thawScratch` живёт между решениями; `applyTeeState` не перезаписывает `newHook, triggeredEvents, colliding, leftWall, freezeEnd, isInFreeze, hookTeleBase, attachedPlayers, queuedWeapon, teleCheckpoint, weapons, frozenLastTick (ставится в true при unfreeze)`, мировой `tick` растёт. Мемо `thawMemo` квантует ключ ⇒ результат зависит от того, какой состав первым заполнил ключ (порядок вычислений должен совпадать).
- Плановый `SimWorld` бота тоже живой: поля, не перезаписываемые `applyTeeState`, наследуются из прошлого решения; `planSim.tick` НЕ синхронизирован с тиком сервера — растёт только на `lag` за решение (при lag=0 стоит на месте) ⇒ `heldTicks`/`dirSince` в live бессмысленны (влияют только при `jitterCost>0`).
- `SimWorld.order` (порядок обработки ти в `step`) не сохраняется в `SimState`; `addTee` делает `unshift` ⇒ позже добавленный ти обрабатывается первым. Респаунов в плановом мире нет (`respawnDelayTicks:0` ⇒ `respawnAtTick=null`), поэтому `order` в роллаутах не меняется.
- Модульные буферы `scoreMeBuf/scoreEnBuf`, кэши полей по объекту коллизии — безопасны.
- Стабильная сортировка `scored` и `kept` — в Rust `sort_by` (стабильная) с тем же компаратором; NaN в score (теоретически) в V8 даёт «равно», в Rust `partial_cmp().unwrap()` упадёт — использовать `total_cmp`-обёртку, эквивалентную `b-a` с NaN→Equal (не проверено, что NaN возможен).

---

## 3. Плавающая точка

- `Math.fround` **нигде нет**. Весь TS (физика ядра, планировщик, obs, NN) — float64. Float32 только в хранилищах: `FreezeMemory` (Float32Array), параметры GRU (`params` base64 float32 → Float64Array), датасет имитации.
- DDNet C++ (сервер) — float32 (`vec2` float, тюнинги `int/100.0f`). TS-тюнинг `tune(v)=trunc(v*100)/100` в f64: расходятся с f32 `groundJumpImpulse 13.2 (f32 13.19999981)`, `airFriction 0.95 (0.94999999)`, `velrampCurvature 1.4 (1.39999998)`, (+ `shotgunSpeeddiff`, `shotgunLifetime`). Квантизация в конце тика (`pos` → int, `vel*256` → int, `characterCore.ts:420`) гасит большую часть расхождений, но на границах `.5` округление может уйти в другую сторону ⇒ TS-симуляция ≠ сервер DDNet побитно (вне скоупа, но важно: учитель обучен на f64-физике).
- **Для паритета Rust↔TS**: Rust-планировщик и его симулятор должны работать в f64 с тем же порядком операций, что TS (не в f32 «как DDNet»). Если физику Rust делать в f32 ради совместимости с сервером — нужен отдельный «TS-совместимый» f64-режим для проверки паритета.
- Где решения чувствительны к мелким различиям: argmax суммарного score кандидатов (сумма сотен членов; близкие кандидаты); `bestScore - bestStayScore < 0.6`; `Math.round(cos*300)` в прицеле (`decodeAction`, `saferInput`, скрипт); `floor(x/32)` в `hazardNearness`; `indexAt` (`trunc(x±0.5)`) в коллизиях; `ceil(max(|vx|,|vy|)/16)` и бинпоиск в `launchFlightLandsInHazard`; ключи `thawMemo` (`Math.round`).
- **Семантика JS-функций, которые нужно воспроизвести бит-в-бит:**
  - `Math.round(x)`: ties к +∞ (`Math.round(-2.5) = -2`, `Math.round(0.49999999999999994) = 0`). Rust `f64::round` — ties от нуля ⇒ нужен `js_round(x) = { let r = x.ceil(); if r - 0.5 > x { r - 1.0 } else { r } }` (+ сохранение -0 при необходимости).
  - `Math.hypot(a,b)`: V8 масштабирует на max и суммирует с компенсацией Кэхэна: `m=max(|a|,|b|); if m==0 → 0; s=0,c=0; for v: t=v/m; summand=t*t-c; pre=s; s=pre+summand; c=(s-pre)-summand; → sqrt(s)*m`. Проверено: совпадает с V8 на 10^6 случайных парах; наивный `sqrt(a²+b²)` расходится в 38% случаев (1 ulp). `hypot(...) || 1` — `0 → 1`.
  - `Math.sign(x) || 1`, `>>> 0`, `Math.imul`, `%` (только неотрицательные операнды здесь).
  - Трансцендентные (замер 2·10^5 случайных аргументов, V8 13.6 vs Rust std (glibc) vs crate `libm 0.2.16`): `atan2`, `exp`, `atan` — `libm` совпадает на 100%, std нет; `pow` (тест только `pow(1.4, p)`, p∈[0,3)) — std совпадает на 100%, `libm` нет; `tanh` — std: 22 расхождения (до 3 ulp), libm: 7709; `sin`, `cos`, `log` — **ни один** не совпадает полностью (libm: ~0.8–1% расхождений в 1 ulp; std ~3–7%). Для полного паритета нужно портировать реализации V8 (`base/ieee754.cc` — fdlibm; sin/cos в V8 ≥11 — порт glibc, флаг `v8_use_libm_trig_functions`, **не проверено**) либо принять редкие расхождения и сравнивать решения статистически.
  - `Math.sqrt` — IEEE, совпадает.

---

## 4. Кодирование наблюдения и действия

### 4.1 OBS (`env/obs.ts`), `OBS_SIZE = 88 + 9*7 + 9*7 = 214`

`safe(v)` = `isFinite ? clamp(v, -5, 5) : 0` (применяется к непрерывным признакам).

| индекс | признак |
|---|---|
| 0 | `self.vel.x/10` |
| 1 | `self.vel.y/10` |
| 2 | grounded: solid в `(x±14, y+14+5)` |
| 3 | `jumpsLeft/2` |
| 4 | frozen |
| 5 | `freezeTicksLeft/150` |
| 6–9 | hook one-hot: `[иное(IDLE 0 / RETRACTED -1), FLYING(4), GRABBED(5), RETRACT 1..3]` |
| 10,11 | `(hookPos - pos)/380` |
| 12 | `hookedPlayer == enemyId` |
| 13–17 | оружие one-hot: hammer, gun, shotgun, grenade, laser (иначе нули) |
| 18,19 | `cos, sin(angle/256)` (angle — рад·256 в сетевом формате) |
| 20 | `clamp(view.tick - attackTick, 0, 50)/50` |
| 21,22 | `(enemy.pos - self.pos)/512` (без врага — нули) |
| 23 | `dist/512` |
| 24,25 | `(enemy.vel - self.vel)/10` |
| 26 | enemy frozen |
| 27 | enemy alive |
| 28 | enemy hooks self |
| 29,30 | `cos, sin(enemy.angle/256)` (без врага 1,0) |
| 31 | прямая видимость: `intersectLine(self, enemy).collision == 0` |
| 32–71 | 20 лучей `angle=r/20·2π`: пары `(dist/800, noHook)`; шаг 16px от 16 до 800, первый solid |
| 72–79 | 8 лучей `r/8·2π`: 1, если freeze/death на 8..64px (шаг 8) |
| 80–83 | ближайший снаряд `(dx/512, dy/512, vx/20, vy/20)`; иначе 0 |
| 84–87 | второй ближайший |
| 88–150 | локальная сетка 9×7 (строки gy, затем gx), шаг 32px, центр в self: 1 = freeze/death, 0.5 = solid/nohook, 0 |
| 151–213 | грубая сетка 9×7, клетка = 3×3 точки с шагом 32 (клетка 96px), origin = `pos - (4*96, 3*96)`: 1, если любая точка hazard; иначе 0.5, если любая solid |

Сетки сэмплируют точки, смещённые от позиции ти (не выровнены по тайлам).

### 4.2 ACTION (`env/action.ts`), `ACTION_SIZE = 10`

`[0..2]` логиты направления (left, none, right); `[3]` jump; `[4]` hook; `[5]` fire; `[6]` hammer, `[7]` gun; `[8,9]` вектор прицела.
`decodeAction(raw, prev, out?, airborne=false)` (`action.ts:19-76`):
- направление: argmax (строгое `>` по порядку 0,1,2); если отличается от удерживаемого `prev.direction+1` и `raw[dir]-raw[held] < 0.15` → остаться.
- jump: если `prev.jump` → `raw[3]>0`; иначе `raw[3] > (airborne ? 0.4 : 0)`. (`airborne=true` передаёт только сетевой мозг бота.)
- hook: `raw[4] > 0`.
- fire (счётчик нажатий DDNet): `held = prev.fire & 1`; `raw[5]>0` → `held ? prev.fire+2 : prev.fire+1` (новое нажатие); иначе `held ? prev.fire+1 : prev.fire` (отпускание).
- оружие: `raw[7] > raw[6] ? GUN : HAMMER`, `wantedWeapon = id+1`; `next/prevWeapon=0`, `playerFlags=0`.
- прицел: `wanted = len<1e-6 ? 0 : atan2(ay, ax)` (NaN → 0); `current = |prev.target|<1e-6 ? wanted : atan2(prev.targetY, prev.targetX)`; `delta = wrap(wanted-current)`; `cap = |delta|>=2.4 ? π : π/2`; `delta *= 0.75`; clamp `±cap`; `angle=current+delta`; `target = (round(cos*300), round(sin*300))`, `(0,0) → (300,0)`.
`emptyInput()` имеет `targetY=-1`; `SimWorld.setInput` заменяет `(0,0)` на `(0,-1)`.

---

## 5. Скриптовый бот (`env/scripted.ts`)

`scriptedAction(view, self, enemy, prev, rng)`:
- если нет self/enemy или враг мёртв: пустой ввод, `fire = prev.fire&1 ? prev.fire+1 : prev.fire` (отпустить).
- `dx,dy = enemy - self`, `dist = hypot`.
- `aimDy = dist<60 ? dy : enemy.y + 14 - self.y`; `noise = (rng.nextFloat()-0.5)*0.2`; `a = atan2(aimDy, dx)+noise`; `target = round(cos a*300), round(sin a*300)`, `(0,0)→(300,…)`.
- `direction = dx>5 ? 1 : dx<-5 ? -1 : 0`.
- `wallAhead` = solid в `(x + (dx>=0?1:-1)*24, y)` или `(…, y-16)`; `jump = wallAhead || dy < -80`.
- `hook = 80 < dist < 380 && LOS` (`intersectLine`).
- `wantedWeapon = (dist<60 ? HAMMER : GUN)+1`.
- `fire = prev.fire&1 ? prev.fire+2 : prev.fire+1` — **новый клик каждый вызов** (стреляет постоянно).
- 1 вызов RNG (`nextFloat`) на вызов.

---

## 6. Нейросети

### 6.1 MLP (`nn/mlp.ts`)
- Раскладка параметров: для каждого слоя подряд `W (nOut×nIn, row-major: строка = выходной нейрон)`, затем `b (nOut)`. `paramCount = Σ nIn*nOut + nOut`.
- `forward`: `sum = b[j]; for i: sum += W[j,i]*a[i]`; скрытые `tanh`, последний линейный. Без `out` возвращает копию.
- `backward` — MSE `0.5·r²`, накопление градиента (нет оптимизатора, нигде не вызывается).
- init: `N(0,1)*sqrt(2/(nIn+nOut))`, bias 0.
- JSON: `{ shape: {inputs, hidden: number[], outputs}, params: number[] }` (float64 числа).

### 6.2 GRU `RecurrentPolicy` (`nn/gru.ts`)
- `GruShape {inputs, hidden, head: number[], outputs}`; вход ячейки `inWidth = inputs + outputs` (обратная связь: `tanh(prevRawOut)`).
- Раскладка: `W` = 3 блока `[z, r, n]` по `hidden×inWidth`; `U` = 3 блока `[z,r,n]` по `hidden×hidden`; `B` = `3*hidden` (`bz, br, bn`); голова: слои `[hidden, ...head, outputs]`, каждый `W (outN×inN)` + `b`.
- Шаг: `z = σ(bz + Wz·x + Uz·h)`; `r = σ(br + Wr·x + Ur·h)`; `n = tanh(bn + Wn·x + Un·(r⊙h))` (reset до матрицы — вариант Cho 2014, НЕ как в PyTorch); `h = (1-z)⊙n + z⊙h`. `σ(x) = x>=0 ? 1/(1+e^-x) : e^x/(1+e^x)`.
- Порядок суммирования (для паритета): гейты стартуют с bias, затем Σ по ненулевым `x[k]` в порядке возрастания `k` (разреженный matvec пропускает нули), затем Σ по `h`. Голова: `dst = 0 + Σ`, **bias добавляется в конце** (в MLP — в начале). Блоки по 8 строк на порядок сумм внутри строки не влияют.
- `saveState/restoreState` = `h ++ out`. `act()` возвращает внутренний буфер `out`.
- init: `N(0,1)/sqrt(fanIn)`, bias 0.
- JSON: `{ kind:"gru", shape, params: {dtype:"float32", encoding:"base64", length, data} }` (или массив чисел). Бот ищет политики: `*.json` в корне/`runs/*` с `shape.inputs==214 && outputs==10`, и `runs/*/best.json|latest.json`.

### 6.3 `opponent.json`
- Ключи: `shape {inputs:224, hidden:[192,96], outputs:10}`, `params` (62698 float64-чисел, ни одно не представимо точно в f32; min -1.29, max 0.99), `horizon: 25`, `heldOut: 0.7944`, `hold: 0.7745`.
- Это MLP 224→192(tanh)→96(tanh)→10(linear). Вход: obs врага (214) + `encodeHumanTarget` текущего ввода врага (10). Используется только argmax `out[0..2]` → предсказанное направление. `horizon=25` — вероятно, горизонт предсказания в тиках; `heldOut`/`hold` — точность сети на отложенной выборке против бейзлайна «держит то же направление» (**не проверено**, по названиям).

### 6.4 `train/humanImitate.ts`
- **Тренера нет** (ни BPTT, ни оптимизатора, ни сборщика датасета из демо). Есть формат, целевая кодировка, метрика.
- Формат файла: `"DDAIHUM1"` (8 байт) + u32 LE длина JSON + JSON-заголовок (дополнен пробелами до кратности 4 от начала файла) + `obs float32[samples*obsSize]` + `targets float32[samples*actionSize]`. Заголовок: `version:1, obsSize, actionSize, samples, sequences[{player, slot, start, length, tick}], players[{key,name,slot,samples,sequences}], source{demo,map,timestamp,seqLength}, built`.
- `encodeHumanTarget(input, fired, activeWeapon)`: `[dir=-1, dir=0, dir=1]` как ±1, `jump ±1`, `hook ±1`, `fired ±1`, `[hammer?1:-1, hammer?-1:1]`, прицел — единичный вектор (или `(1,0)`).
- `agreement`: dir 1, jump/hook/fire по 0.5, оружие 0.5, прицел `2*(cos+1)/2`; нормировка на 5. `scoreHuman` гоняет GRU по последовательностям с `reset()` на каждую.

---

## 7. Маршруты и навигация

### 7.1 `plan/route.ts`
Константы: `HOOK_TILES = floor(380/32) = 11`; `JUMP_UP_REACH=[7.7,7.3,6.9,6.3,5.6,4.4]` (подъём 0..5 тайлов), `JUMP_DOWN_REACH=[7.7,8.1,8.4,8.8,9.1,9.4,9.7]` (спуск 0..6), `JUMP_MARGIN=0.8`; стоимости `WALK 1, FALL 1, JUMP 3, HOOK 6, KILL 60, FREEZE_FALL 25 (≤3 тайла), FREEZE_CROSS 20 (≤3), DANGER +4`; `MAX_TRAP_TILES=250`; `RAY_DX=[0,1,1,1,0,-1,-1,-1]`, `RAY_DY=[-1,-1,0,1,1,1,0,-1]`; `OVERSHOOT_DX=2`; `ENTITY_SPAWN=192`.

`Grid` (`gridOf`, кэш по коллизии, `:79-161`): по центрам тайлов: `solid`, `death`, `unfreeze`, `hookable = solid && !nohook`, `free = !solid && !freeze && !death` ИЛИ tele-in/evil-in (даже если фриз); `danger` = в 3×3 окрестности есть freeze/death; `teleOut[i]` = индекс первого tele-out для tele-in, `teleIn` обратная карта; `firstSolid[d][i]` — первый solid-тайл по лучу направления d (динамика, порядок обхода по знаку dx/dy).

`expand(g, x, y, visit, spawns?, throughFreeze=true, loose=false)` — порядок генерации (важен для тай-брейков Дейкстры):
1. kill: к каждому спауну (кроме текущего тайла) cost 60, kind 4.
2. walk `dx∈[-1,1]`: соседний free, и опора (`solid` снизу или нижний край карты) у текущего ИЛИ соседнего; kind 0.
3. fall `dx∈[0,-1,1]`: `(x+dx, y+1)` free; для диагонали ещё `(x+dx,y)` free; kind 1.
4. freeze-fall (при throughFreeze): вниз `d=1..3` по тайлам «не free, не solid, не death»; первый free под ними → cost `25*d`, kind 5.
5. freeze-cross `dx∈[-1,1]`: `freezeCrossing` (старт с опорой, если не loose; `d=1..3` фриз-тайлов без опоры (если не loose); выход на free) → cost `(unfreeze на выходе ? 20 : 40)*tiles`, kind 5.
6. jumps (если есть опора): `dy=-5..6`, `reach=(dy<=0 ? UP[-dy] : DOWN[dy])*0.8`, `dx=-trunc(reach)..trunc(reach)`: цель free и с опорой, не `overshootsIntoHazard` (при `|dx|>=2` тайл за целью hazard, либо free над hazard), `jumpClear` (колонна вверх до apex свободна, горизонталь на apex и вертикаль вниз — Брезенхем `clearLine(forTee)`); cost `3+|dx|+max(0,-dy)`, kind 2.
7. teleport: если tele-in → в tele-out, cost 1, kind 0.
8. hook по 8 лучам: `ai = firstSolid`, hookable, `hypot(dx,dy)*32 <= 380`; целевой тайл = перед якорем по лучу, free и `clearLine(forTee)`; cost `6 + chebyshev`, kind 3, anchor = ai.
`expandBack` — обратные рёбра (для `routeField`/deadZone; freeze-переходы с kind 1).

`findRoute(col, from, to, opts)` (`:501-610`): Дейкстра на бинарной куче Int32 (`Heap`, своя реализация — порядок тай-брейков надо копировать), ленивое удаление, штамп поколений; `near=2` (Чебышёв-коробка у цели), `maxCost=4000`, `maxNodes=200000` pop'ов, `avoid` — множество ключей `index*8+kind`; `partial`: ближайший (Манхэттен) узел, если `gap < startGap*0.66`. Шаги: от старта (искл.) до цели; `kinds=["walk","fall","jump","hook","kill","fall"]`; kind 5 → `freeze=true`, `leap` если по x сдвинулись; `tele` если следующий шаг — tele-out.

`RouteRunner` (`:633-854`): константы `REACHED_PX 48, CENTRE_PX 10, HOOK_MAX_TICKS 100, STALL_TICKS 100, FROZEN_GIVE_UP_TICKS 25, FREEZE_MOVE_TICKS 200, VETO_LIMIT 12, THAW_LOOKAHEAD_STEPS 8, IN_THE_WAY_PX 44`. Шаг достигнут при расстоянии до центра тайла < 48; для hook-шага: целиться в якорь (`round(unit*300)`), hook=1, направление к якорю; иначе направление к центру (порог 10px), прыжок для jump/leap/подъёма > 16px или если впереди ти и прыжок безопасен (`hopClear`) — `jump = rising ? 1 : decisions%2==0`. Состояния running/arrived/stuck/replan; kill-шаг ждёт респауна.

`deadZone(col, spawns)` (`:926-984`): флуд вперёд от спаунов (`expand`, loose) и назад (`expandBack`, loose); «ловушка» = free && достижимо && оттуда нельзя вернуться; 4-связные компоненты > 250 тайлов обнуляются. `routeField` — экспортирован, не используется.

### 7.2 `bot/navigate.ts` — `Navigator`
Константы: `LOOKAHEAD_TILES 4, CENTRE_PX 8, RISING_VEL -0.1, STALL_TICKS 150, PROBE_TICKS 100, TELEPORT_JUMP_PX 96, MAX_HONEST_PX_PER_TICK 12, MAX_TELE_GOALS 8, MAX_ROUTE_REPLANS 3, MAX_ROUTE_DROPS 3, MAX_ALTERNATIVE_ROUTES 4, MAX_CROSS_TRIES 4, PLANNED_FREEZE_STEPS 16, CLIMB_MIN_RISE_TILES 3, CLIMB_ARC_RAYS 13, CLIMB_RAY_STEP_PX 12, CLIMB_GIVE_UP_TICKS 120, CLIMB_ARRIVE_PX 40, CLIMB_BRAKE_TICKS 6, WALK_BRAKE_TICKS 3`; `GROUND_JUMP_RISE_PX 182`, `DOUBLE_JUMP_RISE_PX 330` — экспортированы, не используются.
Алгоритм `step()`:
- детект телепорта/респауна: сдвиг ≥ `max(96, dt*12)` px → если цель — телепорт в ≤48px от старой позиции → arrived; иначе сброс окна/маршрута по ситуации.
- активный crossing → `stepCrossing`.
- поле `fieldTo(goal)` = копия `travelField` от цели (BFS без фриза). Если текущий тайл недостижим (или walkRouted) → `findRoute(near 2, allowKill, throughFreeze, avoid)` → `RouteRunner`; нет → `startCrossing`; нет → следующая цель.
- `RouteRunner` ведёт; replan ≤3; при stuck — `avoid.add(failedMove)` ≤4 альтернатив.
- иначе ходьба: на тайле цели — arrived (или «probing» 100 тиков для телепорта); окно стагнации 150 тиков: если не приблизились — один раз `findRoute`, иначе следующая цель.
- `follow`: `traceRoute(field, tx, ty, 4)` — спуск по убыванию dist, приоритет соседа: тот же шаг +4, горизонталь +2, вниз +1 (порядок соседей `(1,0),(-1,0),(0,1),(0,-1)`, строгое `>`); вне поля — ближайший в окне ±3. Направление к центру 4-го тайла (порог 8px); торможение, если фриз в пределах `24 + max(0, vx*dir)*(lag+3)` px (`hazardWithin`: точки через 32px, вниз до 4 тайлов); подъём ≥3 тайлов → `climb` хуком: 13 лучей по верхней полуокружности `-π + π(r+0.5)/13`, шаг 12px от 32 до 380, первый solid (nohook → луч отброшен), `score = rise + 40 (если в сторону движения)`; прицел поворачивается не более 0.12 рад/решение.
- `jump = wantUp && (vy < -0.1 || steps%2==0)`.

### 7.3 `bot/crossing.ts` — `SwingCrosser`
Константы: `HOLDS [10,16,24,40]`, `PUSH_AFTER_TICKS 30`, `SETTLE_TICKS 110`, `HOP_RUNS [10,20,40]`, `HOP_JUMP_AT [0,2,4,6,8,10,14]`, `HOP_JUMP_HOLD [6,14]`, `HOP_TICKS 90`, `DROP_RUNS [0,6,12,24]`, `DROP_TICKS 260`, `AIR_RUNS [4,10,20,40]`, `AIR_JUMP_AT [-1,0,3]`, `MAX_HOPS 3`, `SPREAD_TICKS 2`, `SPREAD_EARLY_TICKS 1`, `NUDGE_PX 6`, `NUDGE_Y_PX 4`, `HOP_CLEAR_PX 6`, `ARRIVED_VX 3`, `APPROACH_DEPTH_TILES 3`, `APPROACH_GIVE_UP_TICKS 150`.
Программы: `Swing {anchor, hold, dir}` (хук в якорь `hold` тиков, толкать `dir` ещё 30 тиков) и `Hop {dir, run, jumpAt, jumpHold, ticks}`. Поиск (`searchSwing/searchHop/searchDrop`) перебирает списки программ; программа принимается, если `robust`: роллаут в собственном `SimWorld` (1 ти) проходит для лагов `lag-1..lag+2` и для сдвигов позиции `±6px x, ±4px y` (кроме попадающих в стену), с моделью задержки ввода (ввод берётся каждые `cadence` тиков и применяется через `lag`). Критерии: `done` (arrivedAt/landedAt/inPassage) после `lag+4`; провал — заморожен в полу камеры или лежит во фризе; для хопов — любая заморозка/близость фриза 6px. Бюджет по времени `budgetMs` через `Date.now` с продолжением перебора на следующем тике (`laps`). Фазы approach→swinging→hopping→arrived/failed; approach: едет к камере/от неё в зависимости от глубины (3 тайла).

### 7.4 `bot/wayblock.ts` — WB
- Единственная карта: **«Copy Love Box»** (`size 387×250`) и её копия **«Copy Love Box JoniTee»** (`size 600×600`, всё сдвинуто на `(+182, +212)`), `WAYBLOCKS` (`:139`). Выбор по `map_name` (trim, lower-case) + проверки: размер карты, все spots «standable» (не solid/freeze/death, solid снизу), все якоря — solid и не nohook (`wayblockFor`, `:149-164`).
- Зеркало: `MIRROR = 234`, `mirrorBox(b) = {x0: 234-b.x1, x1: 234-b.x0, y как есть}`, точки `tx' = 234 - tx` (ось симметрии — тайл 117, не центр карты).
- Левая сторона (правая — зеркальная):
  - `FROM = [{103,29,131,35}, {107,36,127,50}]` (общий), `CHAMBER = {108,36,126,50}` (общий, симметричен).
  - `LEFT_TUBE`: start `(104,35)`; anchors `(107,37),(107,38),(107,39),(105,40),(106,40),(107,40),(103,39),(102,38)`; landing `[{95,41,103,50}]`; exit `[{87,51,92,63},{87,64,91,65}]`, exitTile `(89,60)`; hall `[{79,67,104,79},{78,79,104,87}]`, hallTile `(90,79)`; toward -1.
  - `RIGHT_TUBE`: start `(130,35)`, зеркальные anchors/landing/exit/hall, exitTile `(145,60)`, hallTile `(144,79)`, toward +1.
  - Зона L: `L1={79,67,104,79}`, `L2={78,79,104,87}`; approach `{84,41,103,66}`; spots `(94,84),(82,79),(101,84)`; watch `(89,79)`; leash = зона, расширенная на 3 (`WB_LEASH_TILES`) + approach.
  - Правые spots: `(152,79),(140,84),(133,84)`, watch `(145,79)`.
  - `avoid = [{96,12,140,24}]`.
- `WbSideChooser`: выбор стороны с меньшим числом игроков (ничья → где стоим / ближайшая), provisional 5 с; `WB_SIDE_HOPPING=false` ⇒ после выбора сторону не меняет.

---

## 8. Демо (`demo/snapshot.ts`, `demo/reckoning.ts`)

- `parseSnapshot(ints)`: формат CSnapshot: `[dataSize, numItems, offsets[numItems], data]`, элемент = `key` + поля; `type = key>>16`, `id = key&0xffff`; проверки размеров (`MAX_ITEMS 1024`, `MAX_SIZE 64K`).
- `unpackDelta(from, ints)`: `[numDeleted, numUpdate, numTemp(игнор), deletedKeys…, затем update: type, id, (size если тип не из таблицы), diff…]`; размеры из `NETOBJ_INT_SIZES` (21 тип 0.6/DDNet: PlayerInput 10, Projectile 6, Laser 5, Pickup 4, Flag 3, GameInfo 8, GameData 4, CharacterCore 15, Character 22, PlayerInfo 5, ClientInfo 17, SpectatorInfo 3, события 2–3); поле = `(past + diff)|0`; сохранённые элементы идут первыми, новые — в конце. Коды ошибок как в DDNet.
- `resolveItems`: UUID-типы (`type >= 0x4000`): ищется элемент `key = (0<<16)|type` (NETOBJTYPE_EX), 4 поля big-endian = UUID; сопоставление с `calculateUuid(name)` = MD5(namespace `e05ddaaac4e64cfbb6425d48e80c0029` + name), версия 3. Известно 18 имён (character/player/gameinfo/projectile/laser/ddnet-projectile/pickup/spectator-info/…).
- **Использование**: из `snapshot.ts` в коде используются только `calculateUuid` и `NETOBJTYPE_PROJECTILE` (`bot/liveWorld.ts:13`); `parseSnapshot/unpackDelta/resolveItems` — мёртвый код (живой клиент использует npm `teeworlds`).
- `reckoning.ts`: `evolveCore` — dead-reckoning снапшотного CharacterCore с `tick < toTick` (≤150 тиков): `CharacterCore` в пустом мире, `tick(false)` (без ввода) + `move()` + `quantize()` потиково — как `Evolve` клиента DDNet. Используется в `liveWorld.ts:140`.
- **Чего нет для полного .demo**: заголовок демо (`TWDEMO\0`, версия, netversion, имя/размер/CRC/SHA карты, тип, длина, timestamp), timeline-маркеры, разбор чанков (tick-маркеры keyframe/inline-delta, чанки SNAPSHOT / SNAPSHOTDELTA / MESSAGE), Huffman-декомпрессия чанков (есть только в npm `teeworlds`), `CVariableInt`-распаковка, встроенная карта, разбор NETMSG (игроки/имена/чат), проверка CRC снапшота, UUID-сообщения/extended-netmsg, склейка в последовательности для датасета. Сборщика `humanImitate`-датасета из демо в репо нет.

---

## 9. Карты (`map/datafile.ts`, `map/loadMap.ts`)

- Datafile: магия `DATA`/`ATAD`, версии **3 и 4**; v4 — таблица несжатых размеров raw-data и zlib (`inflateSync` с `maxOutputLength`), v3 — без сжатия. Полная валидация заголовка/смещений/размеров (как в DDNet `CDataFileReader`), лимит 256 МБ на распаковку, кэш распакованных данных.
- `loadMapCollision(path)`:
  - слои `MAPITEMTYPE_LAYER=5`, тип `LAYERTYPE_TILES=2`; первый слой с флагом GAME → ширина/высота/версия/data. Версия ≥4 (Teeworlds 0.7 `TILESKIP`): тайл = `(index, flags, skip, reserved)`, повтор `skip+1` раз; иначе 4 байта на тайл.
  - **tele** (`m_Tele` на смещении 72, для версии тайлмапа ≤2 — 60): 2 байта/тайл `(number, type)`.
  - **speedup** (76 / 64): 6 байт/тайл `(force u8, maxSpeed u8, type u8, pad, angle i16 LE)`.
  - **front** (80 / 68): 4 байта/тайл, берутся `index` и `flags`.
  - **switch** и **tune** — только флаг наличия, данные не читаются.
  - `readMapSettings`: MAPITEMTYPE_INFO id 0, строка настроек по смещению 20 → распознаётся только `sv_no_weak_hook`.
  - Имя карты = basename файла без расширения; в live — `client.map.map_name` (буфер карты пишется во временный файл) или `maps/<name>.map`.
- `Collision`: флаги только из **game**-слоя: solid(1), nohook(3)=solid+nohook, death(2), freeze(9), unfreeze(11). **Фриз/смерть front-слоя игнорируются** в физике и во всех полях планировщика; front используется только для стопперов, hook-through, `tileExists`, спаунов (`route.spawnTiles`). `DFREEZE/DUNFREEZE (12/13)` обрабатываются в `world.handleTile`, но не в `hazardField`. Координаты → тайл: `indexAt` = `trunc(x ± 0.5)` (round-half-away) `>> 5` с клампом к карте (за краем — крайний тайл).

---

## 10. Производительность (замер, синтетика, Node 24, budget=0)

- Время решения: медиана ~9 мс, среднее ~10.5 мс, p90 ~18 мс (дефолт 20×2); с `budgetMs 3` — ~5 мс и ~19 кандидатов. Bold (64×3): ~39 мс медиана, 199 кандидатов; strong WB (40×3): ~30 мс.
- На решение: ≈48 `evaluate`, ≈1340 `world.step()` (2 ти), ≈1300 `scoreTick`.
- Профиль (self-time): `launchFlightLandsInHazard` 14.7% (inclusive 21%: для каждого тика при `sep<70` и условии «exact» — 50-тиковая мини-симуляция с 5-шаговым бинпоиском `testBox`), `testBoxAt` 11%, `CharacterCore.move` 9.7%, GC 6.9%, `getMapIndices` 6.5%, `CharacterCore.tick` 5.5%, `world.step` 4%, `handleTiles` 3.5%, `intersectLine` 3.3% (hookAllowed/hookWouldReach — 3.6% inclusive), `scoreTick` self 2.3% (inclusive 25.6%). Итого: физика ≈47%, скоринг ≈26%, прочее — гейты хука, decode, save/restore.
- Аллокации: `getTee()` создаёт новый объект (много вызовов: `hookAlreadyOut`, `hookWouldReach`, `landedThrows`, терминальные члены); `saveState` пересоздаёт `CoreState` и массивы снарядов; `restoreState` пересоздаёт объекты снарядов/лазеров; `decodeAction` без `out` аллоцирует `PlayerInput` (3 раза на шаг при гейтах молота); `quantize()` и `clampVel` создают новые Vec2 каждый тик; `step()` — новый массив событий; планы — массивы объектов; `slice` элит; строковые ключи `thawMemo`; `escapeExists` — 2 новых SimState.
- Для Rust: всё это на стеке/в пулах; главный выигрыш — дешёвый `launchFlightLandsInHazard` (кэш по (pos,vel) внутри решения — **только если сохраняется точная семантика**) и отсутствие GC. Паритет требует той же последовательности вызовов (RNG, thawMemo).

---

## 11. Баги, мёртвый код, странности

1. `routeDistance`: `travelField` считается каждое решение, передаётся в `scoreTick` как `travel`, но **не используется** (`planner.ts:1080, 518`). К тому же `travelField` возвращает общий scratch.
2. Мёртвые/ботовые поля в `PlannerConfig`: `seek, pathToTarget, settledFreezeTicks, blockHoldScore, liveTransfer, planOthers, targetHold`.
3. `bandCost=0` по умолчанию ⇒ `setBand` в WB ничего не делает; WB-оверрайды его не включают.
4. `opponent.json` грузится всегда, но нужен только при `opponentModel:"learned"` (не дефолт).
5. `planSim.tick` не синхронизирован с сервером ⇒ `heldTicks/dirSince` и `warmShift` (с `warmShiftElapsed`) в live считаются по «тикам лага».
6. `thawScratch` живёт между решениями с неполным сбросом состояния; мемо по квантованному ключу — зависимость от порядка.
7. `inDead` не проверяет `x < width` (перенос на следующую строку) (`:512-516`).
8. Несогласованность `floor` (hazardNearness, freezeGapPx) и `trunc` (travelDistance, inDead, route) для отрицательных координат.
9. Поля опасности видят только `TILE_FREEZE/TILE_DEATH` game-слоя (нет front-фриза, deep-freeze 12, LFREEZE 144).
10. `refit` использует элиту только текущей итерации (книга не влияет на итерацию 2 кроме как через dist).
11. `policySeedPlans`: имя `i` затенено в `map((st, i) => …)` — работает, но путает.
12. `polishRope` при `n==4` оценивает один и тот же план дважды (`[2,4,n]`).
13. `launchFlightLandsInHazard`: `vx *= grounded ? … : …` — `grounded` только в первом тике; вертикальная скорость без velramp; упрощённая физика ≠ настоящей.
14. `restsInFreeze` — упрощённая баллистика, не совпадает с настоящей физикой (намеренно).
15. `Mlp.backward`, `initParams`, `initGruParams`, `saveMlp`, `loadMlp`, `routeField`, `travelDistance*`, `GROUND_JUMP_RISE_PX/DOUBLE_JUMP_RISE_PX`, `parseSnapshot/unpackDelta/resolveItems`, `setSearchSeed`, `setSeedPolicy`, `setValueNet`, `setOpponentPolicy`, `Planner.budget` — не вызываются/мёртвые в текущем боте.
16. `Navigator.step`: `this.steps++` дважды на ветке runner; `climbBestY` считается, но не используется в решениях.
17. Нет тренера для GRU/MLP/humanImitate — веса (`opponent.json`, политики) получены вне репо.
18. `decide` с `frozenTargetSteps` сохраняет `warm` другой длины; hookSeeds/carry проверяют длину, buildDist клампит индекс — работает, но тонко.
19. `SimWorld.restoreState` не удаляет ти, добавленные после сохранения, и не восстанавливает `order`.
20. Трансформация тюнинга `trunc(v*100)/100` в f64 ≠ DDNet f32 (см. §3).

---

## 12. Риски порта и рекомендации (кратко)

- Паритет требует **f64 и V8-семантики** (`Math.round`, `Math.hypot`, `atan2/exp/atan` через `libm`, `pow/tanh` через std, для `sin/cos/log` — порт V8 или допуск редких 1-ulp расхождений).
- Точно повторить порядок RNG-вызовов (включая spare Гаусса), стабильные сортировки, строгие `>`-сравнения при выборе best/bestStay, порядок генерации книги, порядок `order` в мире (последний добавленный — первым).
- Воспроизвести персистентное состояние между решениями: `warm`, `committed`, `decideGaps`, `dirSince`, `thawScratch`+`thawMemo`, `oppSeed`, `Rng`.
- Для учителя: использовать фиксированные итерации (`budgetMs=0`), иначе решения зависят от CPU.
- Главный узкий участок — физика + `launchFlightLandsInHazard`; в Rust ожидается кратное ускорение без изменения семантики.
