#!/usr/bin/env bash
# Runs the full >= 10^6-probes-per-function proof of crates/ddai-jsmath against the real Node
# 24.21.0 / V8 13.6 binary on this machine, then the ns/call performance report.
#
# Usage: ./run.sh [--fixture]
#   (no args)   full oracle run (crates/ddai-jsmath/tests/oracle.rs, `full_oracle_vs_real_v8`)
#               + perf report (tests/perf.rs, `perf_report`).
#   --fixture   also regenerates the committed golden fixture
#               (tests/fixtures/golden.bin/golden.sha256) from the same probe generator, at a
#               much smaller scale — run this after changing the probe generator or `rng.ts`.
set -euo pipefail
cd "$(dirname "$0")/../../crates/ddai-jsmath"

echo "run.sh: node $(node --version), V8 $(node -p process.versions.v8)"

echo "run.sh: full_oracle_vs_real_v8 (>= 1e6 probes/function; override with JSMATH_PROBES)"
cargo test --release --test oracle -- --ignored full_oracle_vs_real_v8 --nocapture

echo "run.sh: pow_literal_args (F1/F9: pow with a literal base/exponent at the call site)"
# F9: tests/pow_literal_args.rs's spot-check only actually exercises the cross-crate inlining
# this guards against under --release (thin LTO) — plain `cargo test` (dev profile) doesn't
# inline `pow` across the crate boundary at all, so it can't catch a regression here regardless
# of `black_box`; see that file's own doc comment. Running it explicitly under --release (not
# just the --ignored full oracle below) keeps that check in the routine sequence this script
# runs. The in-crate `ddai_jsmath::pow::tests::literal_argument_regression_survives_dev_profile`
# unit test (part of plain `cargo test -p ddai-jsmath`, no flags needed) is the one that catches
# a regression without --release at all.
cargo test --release -p ddai-jsmath --test pow_literal_args
cargo test --release --test pow_literal_args -- --ignored literal_args_vs_real_v8 --nocapture

if [ "${1:-}" = "--fixture" ]; then
	echo "run.sh: regenerate_golden_fixture"
	cargo test --release --test oracle -- --ignored regenerate_golden_fixture --nocapture
	echo "run.sh: verifying the freshly-written fixture with the normal (non-ignored) test"
	cargo test --test oracle golden_fixture_matches_v8
fi

echo "run.sh: perf_report (ns/call, no target — informational only)"
cargo test --release --test perf -- --ignored perf_report --nocapture

echo "run.sh: done"
