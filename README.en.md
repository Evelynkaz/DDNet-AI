> **Modified version.** This is a fork of [Wranked1/DDNet-AI](https://github.com/Wranked1/DDNet-AI) (author: Wranked1, GPL-3.0), rewritten in Rust: the bot, a web interface instead of the Electron window, and a neural network constrained by the Drosophila connectome ("the fly"). It is not Wranked1's original program. The old TypeScript version (Electron, `start.mjs`, `run.sh`) has been removed from the tree; it stays in the git history, the last commit that has it is `0311695`.

<div align="center">

<img src="assets/logo.png" width="112" alt="">

# DDNet AI

**A bot that plays block in DDNet by itself, written in Rust**

It throws the hook, swings the hammer, puts opponents into freeze and keeps itself out of it.<br>
A live decision fits in 5 ms (p99); the world runs at 50 ticks per second.

[Русский](README.md) · **English**

</div>

## What it is

One binary, `ddnet-ai` (a cargo workspace in `crates/*`):

- **the bot** (`ddnet-ai play`) joins a DDNet 20.x server as an ordinary client, predicts the world on bit-exact
  physics (`ddai-physics`, identical to the DDNet server) and picks a move; **the bot never writes to the game chat**;
- **brains** are interchangeable: `planner` (search under a threat model, the reference and the teacher), `hybrid` (a
  neural net proposes, exact search verifies; the live mode), `fly` (the fly alone), `scripted`, `idle`;
- **the fly** is a network built on the Drosophila connectome (MaleCNS v1.0, CC-BY 4.0), trained by behaviour cloning
  and DAgger;
- **the web interface** (`ddnet-ai web`) replaces the Electron window: a live map and game frames, bot status and
  commands, friend/war/ignore lists, a training viewer; served behind Caddy with HTTPS and a password;
- **the arena** (`ddnet-ai arena`) runs offline N-player matches on the same physics for honest measurements (Wilson
  confidence intervals).

Status and plans (in Russian): [docs/STATUS.md](docs/STATUS.md), [docs/PLAN.md](docs/PLAN.md); design: [docs/FLY.md](docs/FLY.md),
[docs/DECISIONS.md](docs/DECISIONS.md), formats: [docs/formats.md](docs/formats.md).

## Build

Linux and Rust are needed (the toolchain is pinned in `rust-toolchain.toml` and installed by `rustup`). Node.js is **not**
needed to run the bot.

```bash
git clone https://github.com/Evelynkaz/DDNet-AI && cd DDNet-AI
cargo build --release          # -> target/release/ddnet-ai
```

Data (maps, demos, the connectome, checkpoints, secrets) is not part of the repository and lives outside it: by default
in `~/aiddnet/data` (change it with `--data-dir`). Trained weights are never stored in git.

## Bot

```bash
# a local DDNet server (docs/SETUP.md, in Russian), planner + neural net, no time limit
target/release/ddnet-ai play --server 127.0.0.1:8303 --brain hybrid --name Muha --duration 0
```

- `--brain`, e.g. `planner`, `scripted`, `hybrid`, `fly` (the bot brains) and the utility `idle`, `circle`,
  `random-scripted` (the full list is in `ddnet-ai play --help`); `--mode`
  (default `fight`), friend/war/ignore lists via `--relations`; the full list is in `ddnet-ai play --help`.
- Connections are allowed only to a local server and to servers listed in `~/aiddnet/data/live-servers.toml` (an
  allow-list; `--server auto` picks the most populated allowed one). Project rule: never evade kicks or bans.
- Stop with SIGINT/SIGTERM (under systemd use `--duration 0`).

## Web

```bash
target/release/ddnet-ai web-passwd                       # the owner password (only its argon2id hash is stored)
target/release/ddnet-ai web --listen 127.0.0.1:7788 --data-dir ~/aiddnet/data \
  --bot-socket ~/aiddnet/data/bot/live.sock               # loopback only; expose it through Caddy
# behind Caddy (HTTPS): add --trust-proxy --cookie-secure
```

The server listens on `127.0.0.1` only; HTTPS and the public entry point are Caddy's job (`deploy/caddy/Caddyfile`,
the unit `deploy/systemd/ddnet-ai-web.service`, installation with `deploy/install.sh`, details in
[deploy/README.md](deploy/README.md)). The site takes the live map from the bot's socket, and the path must be given explicitly (`--bot-socket`; there is no
default). The bot creates `<data-dir>/bot/live.sock` by default, so run the bot and the site with the same `--data-dir`.
`--trust-proxy` and `--cookie-secure` are only for running behind Caddy (see `deploy/systemd/ddnet-ai-web.service`).

## Training

```bash
# 1) the connectome (third-party data, a few GB), once
cargo run --release -p ddai-connectome -- fetch --manifest manifests/connectome.toml --dest ~/aiddnet/data/connectome/raw
# 2) teacher games of the planner in the arena -> a dataset; 3) training (BC + DAgger); 4) checkpoint evaluation
target/release/ddnet-ai train collect --config configs/train/<name>.toml
target/release/ddnet-ai train run     --config configs/train/<name>.toml
target/release/ddnet-ai train eval    --config configs/train/<name>.toml --bundle <checkpoint>
target/release/ddnet-ai arena run     --config configs/arena/dev-quick.toml --out ~/aiddnet/data/runs/arena-test
```

Configs live in `configs/train`, `configs/arena`, `configs/scenarios`; experiment results are in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md); the sub-graph build and the `.flyg` format are described in
`crates/ddai-connectome/README.md` and `crates/ddai-fly/README.md` (Russian).

## Checks

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo deny check
tools/ci/no-weights.sh        # no weights, demos, maps or big files in git
```

**Parity with the old TypeScript.** The Rust ports of the planner, navigation and the TS world prove that they make the
same decisions as the original code. The reference dumps come from the generators in `tools/ts-trace`, run over the frozen
sources in [`tools/ts-reference`](tools/ts-reference/README.md) (Node >= 24 and `npm ci` in that directory); the tests run
with `--features ts-parity -- --ignored`, the commands are in `crates/ddai-planner/README.md` and
`crates/ddai-nav/README.md`.

## Repository layout

| Path | Contents |
|---|---|
| `crates/` | Rust crates: physics, DDNet network and client, world, planner, navigation, brains, the fly, training, arena, web, clips |
| `configs/` | arenas, scenarios, training and fly configs |
| `manifests/` | the pinned file list of the connectome |
| `deploy/` | Caddy, systemd, installer |
| `tools/` | parity oracles (C++ DDNet, V8), TS dump generators, e2e (Playwright), CI scripts |
| `tools/ts-reference/` | the frozen TS reference for parity (not a working bot) |
| `docs/` | plan, status, decisions, experiments, formats, research (mostly Russian) |

## License and credits

[GPL-3.0](LICENSE). The original project is [Wranked1/DDNet-AI](https://github.com/Wranked1/DDNet-AI) by Wranked1; this is a
modified version (section 7(b), see [NOTICE](NOTICE)). The physics is ported from DDNet (zlib notice in [NOTICE](NOTICE)),
the network part follows the Teeworlds 0.6 + DDNet protocol. The connectome data is MaleCNS v1.0, CC-BY 4.0 (Berg et al., 2026).
