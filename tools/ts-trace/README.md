# tools/ts-trace — генераторы дампов паритета

Генераторы (`gen-*.mjs`) запускают **настоящий, неизменённый** TS-код старого бота на Node 24 (type stripping, без
компиляции) и пишут JSON-lines с побитовыми значениями; Rust-тесты (`--features ts-parity`, `#[ignore]`) проигрывают то же и
требуют нуля расхождений. Подробности по областям — `crates/ddai-planner/README.md`, `crates/ddai-nav/README.md`,
`crates/ddai-tsworld/README.md`, `docs/formats.md` §16.

## Два эталона (задача 4.8)

- **`tools/ts-reference/`** — замороженная копия апстрима `c3c619d` (не редактируется, `tools/ts-reference/README.md`). По нему сняты
  корпуса паритета планировщика, мира и навигации, существовавшие до 4.8. Это эталон по умолчанию.
- **Второй эталон — любой каталог с `src/`**, на который указывает переменная **`DDAI_TS_REF`**. Для релиза 2026-10-02 это
  извлечённая копия `af49dfb` в `~/aiddnet/data/scratch/ts-af49dfb` (после работы каталог удаляется, воссоздаётся командами ниже).
  `lib.mjs` берёт из него `core/world.ts`, `core/collision.ts`, `core/types.ts`, `map/loadMap.ts`, а генераторы — остальное
  (`${TS_REF}/src/...`). Без переменной всё работает как раньше (проверено: дамп `deadzone`/`spawns` и выбор стороны/споты/зоны
  `gen-nav-dump.mjs` совпадают со старым корпусом побайтно).

```bash
mkdir -p ~/aiddnet/data/scratch/ts-af49dfb && cd ~/aiddnet/ref/DDNet-AI-upstream \
  && git archive af49dfb src package.json package-lock.json | tar -x -C ~/aiddnet/data/scratch/ts-af49dfb
mkdir -p ~/aiddnet/data/scratch/ts-af49dfb/node_modules \
  && cp -r ~/aiddnet/DDNet-AI/node_modules/teeworlds ~/aiddnet/data/scratch/ts-af49dfb/node_modules/   # или `npm ci --ignore-scripts` там
export DDAI_TS_REF=~/aiddnet/data/scratch/ts-af49dfb
# навигация (секции: deadzone = spawns + deadzone, routes, wb, cross, nav):
for s in deadzone routes wb cross nav; do
  node tools/ts-trace/gen-nav-dump.mjs --map "<карта>" --seed 1 --section $s --routes 600 --crossings 60 --navs 40 --out ~/aiddnet/data/traces/nav-af49dfb/<имя>-$s.jsonl
done
DDAI_NAV_DUMP=<дамп> cargo test -p ddai-nav --features ts-parity --release --test parity_nav -- --ignored --nocapture
```

Каталогу нужны `src/` и `package.json` с `"type": "module"`; для `bot.ts` (генератор навигации импортирует `DdnetBot`) ещё
`node_modules/teeworlds`. В заголовке дампа навигации: `tsRef` (путь эталона) и `hall` (есть ли у эталона поиск зала, то есть
это `af49dfb` или новее); по `hall` Rust-тест решает, что проигрывать (дамп `c3c619d` — частично: маршруты, мёртвая зона, спавны,
выбор стороны, зоны; переходы, споты и навигатор `af49dfb` изменил намеренно).

**Планировщик (задача 3.8).** `planner.ts`, `seal.ts` и `throwLines.ts` апстрим тоже менял. Корпуса `ddai-planner` теперь двух версий: прежний
(`c3c619d`, `DDAI_TS_REF` не ставят) и новый для `af49dfb` (`DDAI_TS_REF=~/aiddnet/data/scratch/ts-af49dfb`, скрипт `run-planner-corpus-v2.sh`:
`gen-planner-dump.mjs`, `gen-planner-freerun.mjs`, `gen-v2-component-dump.mjs`; дампы — `~/aiddnet/data/traces/planner-af49dfb/`). В заголовке
дампа планировщика поле `plannerVersion` (`classic` / `upstream-2026-10-02`; нет поля — `classic`) выбирает конфигурацию в Rust-тесте. Команды и
результаты — «Версии планировщика» в `crates/ddai-planner/README.md`.
