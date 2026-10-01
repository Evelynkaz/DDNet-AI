#!/usr/bin/env bash
# Opt-in TRAINING-ONLY build of `ddnet-ai` for this host class (task 7.2b, D-064).
#
#   tools/train-v3-build.sh            # -> target/train-v3/release/ddnet-ai-train-v3
#
# What it is: the release build with `RUSTFLAGS="-C target-cpu=x86-64-v3"` (AVX2 + FMA) in a SEPARATE
# target directory. The batched fly training backend's hot loops are 1.45x faster per core with it
# (8 lanes per instruction instead of 4, fused multiply-add).
#
# What it is NOT, and never will be:
#   * not the default: `cargo build --release` stays the baseline x86-64 build, and no
#     `target-cpu` appears in any committed Cargo/.cargo configuration;
#   * not the bot: the binary refuses every subcommand except `train` and `fly` (see
#     `TRAIN_ONLY_BUILD` in crates/ddnet-ai/src/main.rs); do not use it for live play, the web UI,
#     the arena, trace/parity tooling or anything else. The physics parity guarantees (D-002/D-004)
#     are for the baseline build only.
#
# Reproducibility: results (gradients, checkpoints) are bitwise reproducible WITHIN one build
# configuration only. A v3 run and a default-build run of the same config differ in f32 rounding
# (fused multiply-add), by amounts of the same size as the batched-vs-per-sequence tolerance
# (docs: crates/ddai-fly/README.md, "Задача 7.2b"). Resume a run with the build that started it.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! grep -qw avx2 /proc/cpuinfo || ! grep -qw fma /proc/cpuinfo; then
  echo "this CPU has no AVX2+FMA: the x86-64-v3 build would crash with SIGILL" >&2
  exit 1
fi

export CARGO_TARGET_DIR="${TRAIN_V3_TARGET_DIR:-$PWD/target/train-v3}"
export RUSTFLAGS="-C target-cpu=x86-64-v3"
cargo build --release --locked -p ddnet-ai
cp "$CARGO_TARGET_DIR/release/ddnet-ai" "$CARGO_TARGET_DIR/release/ddnet-ai-train-v3"
echo "built $CARGO_TARGET_DIR/release/ddnet-ai-train-v3 (training only: train, fly)"
