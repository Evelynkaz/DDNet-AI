> **Изменённая версия.** Это форк [Wranked1/DDNet-AI](https://github.com/Wranked1/DDNet-AI) (автор — Wranked1, GPL-3.0), переписанный на Rust: бот, веб-интерфейс вместо Electron-окна и нейросеть на коннектоме дрозофилы («муха»). Это не оригинальная программа Wranked1. Старая TypeScript-версия (Electron, `start.mjs`, `run.sh`) удалена из дерева; она остаётся в истории git, последний коммит с ней — `0311695`.

<div align="center">

<img src="assets/logo.png" width="112" alt="">

# DDNet AI

**Бот, который сам играет в блок в DDNet — на Rust**

Кидает хук, бьёт молотом, закидывает соперников во фриз и не попадает туда сам.<br>
Живое решение укладывается в 5 мс (p99), мир идёт 50 тиков в секунду.

**Русский** · [English](README.en.md)

</div>

## Что это

Один бинарник `ddnet-ai` (cargo workspace из `crates/*`):

- **бот** (`ddnet-ai play`) подключается к серверу DDNet 20.x как обычный клиент, строит прогноз мира на побитно верной
  физике (`ddai-physics`, совпадает с сервером DDNet) и выбирает ход; **в игровой чат бот не пишет ничего**;
- **мозги** взаимозаменяемы: `planner` (перебор по модели угроз, эталон и учитель), `hybrid` (подсказывает нейросеть,
  точный поиск проверяет — боевой режим), `fly` (одна «муха»), `scripted`, `idle`;
- **«муха»** — сеть на коннектоме дрозофилы (MaleCNS v1.0, CC-BY 4.0), обучаемая клонированием поведения и DAgger;
- **веб-интерфейс** (`ddnet-ai web`) заменил Electron-окно: карта и кадры игры в реальном времени, статус и команды
  боту, списки друзей/врагов/игнора, просмотр обучения; за Caddy с HTTPS и паролем;
- **арена** (`ddnet-ai arena`) — офлайн-матчи N игроков на той же физике для честных измерений (доверительные
  интервалы Вильсона).

Статус и планы — [docs/STATUS.md](docs/STATUS.md), [docs/PLAN.md](docs/PLAN.md); устройство — [docs/FLY.md](docs/FLY.md),
[docs/DECISIONS.md](docs/DECISIONS.md), форматы — [docs/formats.md](docs/formats.md).

## Сборка

Нужны Linux и Rust (версия закреплена в `rust-toolchain.toml`, ставится через `rustup`). Node.js для работы бота **не нужен**.

```bash
git clone https://github.com/Evelynkaz/DDNet-AI && cd DDNet-AI
cargo build --release          # → target/release/ddnet-ai
```

Данные (карты, демки, коннектом, чекпоинты, секреты) в репозиторий не входят и лежат вне его: по умолчанию в
`~/aiddnet/data` (меняется флагом `--data-dir`). Обученные веса в git не хранятся.

## Бот

```bash
# локальный сервер DDNet (docs/SETUP.md §5c), планировщик + нейросеть, без ограничения по времени
target/release/ddnet-ai play --server 127.0.0.1:8303 --brain hybrid --name Muha --duration 0
```

- `--brain` — например `planner`, `scripted`, `hybrid`, `fly` (мозги бота), а также служебные `idle`, `circle`,
  `random-scripted` (полный список — `ddnet-ai play --help`); `--mode` (по умолчанию
  `fight`), списки друзей/врагов/игнора — `--relations`; полный перечень — `ddnet-ai play --help`.
- Подключение допускается только к локальному серверу и серверам из `~/aiddnet/data/live-servers.toml`
  (список разрешённых; `--server auto` берёт самый населённый разрешённый). Правило проекта: кики и баны не обходить.
- Остановка — SIGINT/SIGTERM (под systemd — `--duration 0`).

## Веб

```bash
target/release/ddnet-ai web-passwd                       # пароль владельца (в файле хранится только argon2id-хэш)
target/release/ddnet-ai web --listen 127.0.0.1:7788 --data-dir ~/aiddnet/data \
  --bot-socket ~/aiddnet/data/bot/live.sock               # только loopback; наружу — через Caddy
# за Caddy (HTTPS): добавить --trust-proxy --cookie-secure
```

Сервер слушает только `127.0.0.1`; HTTPS и публичный вход обеспечивает Caddy (`deploy/caddy/Caddyfile`,
юнит `deploy/systemd/ddnet-ai-web.service`, установка — `deploy/install.sh`, подробности — [deploy/README.md](deploy/README.md)).
Живую карту сайт берёт из сокета бота: путь надо передать явно (`--bot-socket`, по умолчанию его нет); бот по умолчанию
создаёт `<data-dir>/bot/live.sock`, поэтому запускайте бота и сайт с одним `--data-dir`. Флаги `--trust-proxy` и `--cookie-secure`
нужны только за Caddy (см. `deploy/systemd/ddnet-ai-web.service`).

## Обучение

```bash
# 1) коннектом (сторонние данные, ~ГБ), один раз
cargo run --release -p ddai-connectome -- fetch --manifest manifests/connectome.toml --dest ~/aiddnet/data/connectome/raw
# 2) учебные партии планировщика на арене → датасет; 3) обучение (BC + DAgger); 4) оценка чекпоинта
target/release/ddnet-ai train collect --config configs/train/<имя>.toml
target/release/ddnet-ai train run     --config configs/train/<имя>.toml
target/release/ddnet-ai train eval    --config configs/train/<имя>.toml --bundle <чекпоинт>
target/release/ddnet-ai arena run     --config configs/arena/dev-quick.toml --out ~/aiddnet/data/runs/arena-test
```

Конфиги — `configs/train`, `configs/arena`, `configs/scenarios`; результаты опытов и выводы — [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md);
сборка подграфа и формат `.flyg` — `crates/ddai-connectome/README.md`, `crates/ddai-fly/README.md`.

## Проверки

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo deny check
tools/ci/no-weights.sh        # в git нет весов, демок, карт и больших файлов
```

**Паритет со старым TypeScript.** Rust-порты планировщика, навигации и TS-мира доказывают, что принимают те же решения,
что оригинальный код. Эталонные дампы делают генераторы `tools/ts-trace` над замороженными исходниками
[`tools/ts-reference`](tools/ts-reference/README.md) (нужен Node ≥ 24 и `npm ci` в этом каталоге); тесты — с
`--features ts-parity -- --ignored`, команды — в `crates/ddai-planner/README.md` и `crates/ddai-nav/README.md`.

## Структура репозитория

| Путь | Что там |
|---|---|
| `crates/` | Rust-крейты: физика, сеть и клиент DDNet, мир, планировщик, навигация, мозги, муха, обучение, арена, веб, клипы |
| `configs/` | арены, сценарии, конфиги обучения и мухи |
| `manifests/` | закреплённый список файлов коннектома |
| `deploy/` | Caddy, systemd, установка |
| `tools/` | оракулы паритета (C++ DDNet, V8), генераторы TS-дампов, e2e (Playwright), CI-скрипты |
| `tools/ts-reference/` | замороженный TS-эталон для паритета (не рабочий бот) |
| `docs/` | план, статус, решения, эксперименты, форматы, исследования |

## Лицензия и авторы

[GPL-3.0](LICENSE). Оригинальный проект — [Wranked1/DDNet-AI](https://github.com/Wranked1/DDNet-AI), автор Wranked1; это
изменённая версия (раздел 7(b) в [NOTICE](NOTICE)). Физика портирована из DDNet (zlib-уведомление в [NOTICE](NOTICE)),
сетевая часть — по протоколу Teeworlds 0.6 + DDNet. Данные коннектома — MaleCNS v1.0, CC-BY 4.0 (Berg et al., 2026).
