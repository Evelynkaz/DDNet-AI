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

## 4. Node 24 (только для старой TS-версии и генератора трасс)

Официальный архив с проверкой sha256, в `/usr/local`:

```bash
cd /tmp
curl -sL https://nodejs.org/dist/latest-v24.x/SHASUMS256.txt -o SHASUMS256.txt
F=$(grep -o 'node-v24[^ ]*-linux-x64.tar.xz' SHASUMS256.txt | head -1)   # node-v24.21.0-linux-x64.tar.xz
curl -sLO https://nodejs.org/dist/latest-v24.x/$F
grep " $F\$" SHASUMS256.txt | sha256sum -c -
sudo tar -xJf $F -C /usr/local --strip-components=1 --exclude CHANGELOG.md --exclude LICENSE --exclude README.md
node --version    # v24.21.0 ; npm 11.19.0
cd ~/aiddnet/DDNet-AI && npm ci --no-audit --no-fund   # node_modules в .gitignore
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

## 5b. Параллельные задачи: git worktree

Для одновременных задач /duo создаются отдельные рабочие копии без веток:
`git worktree add --detach ~/aiddnet/wt/task-<N> HEAD`. После одобрения изменения переносятся патчем в основной
`main` (`git -C <wt> add -A && git -C <wt> diff --cached --binary > p.patch; git apply --index p.patch`), рабочая
копия удаляется `git worktree remove`.

## 6. Будет добавлено по фазам

- Фаза 1: пакеты для сборки C++-оракула DDNet (cmake, zlib, sqlite3, curl… — ровно то, что понадобится).
- Фаза 2: локальный ddnet-server 20.1 + systemd-юнит.
- Фаза 5: Caddy 2.11 (официальный apt-репозиторий Cloudsmith), ufw 80/443, systemd-юниты бота, Playwright 1.63
  (Chromium headless shell).
- Фаза 6: данные коннектома по манифесту.
