#!/usr/bin/env bash
# Vectorisation guard for the batched fly training backend (task 7.2b).
#
#   tools/vectorisation/check.sh
#
# 1. runs the in-process ratio tests (batched kernels vs scalar reductions, same data, same
#    process, best-of-N timings; robust to machine load) in an optimised build;
# 2. disassembles that test binary and checks that the hot loops of `gather_acc` contain packed
#    SIMD multiply/add (or FMA) instructions and almost no scalar ones.
#
# Works for the default build and for the opt-in training build (RUSTFLAGS="-C
# target-cpu=x86-64-v3", see docs/DECISIONS.md and `tools/train-v3-build.sh`): export the same
# RUSTFLAGS / CARGO_TARGET_DIR here to check that build. Needs objdump (binutils) and python3.
set -euo pipefail
cd "$(dirname "$0")/../.."

cargo test -p ddai-fly --locked --release --lib vectorisation_guard -- --nocapture --test-threads=1

bin=$(cargo test -p ddai-fly --locked --release --lib --no-run --message-format=json 2>/dev/null |
  python3 -c '
import json, sys
exe = None
for line in sys.stdin:
    try:
        m = json.loads(line)
    except ValueError:
        continue
    if m.get("reason") == "compiler-artifact" and m.get("executable") and m["target"]["name"].replace("-", "_") == "ddai_fly" \
            and m["profile"]["test"]:
        exe = m["executable"]
print(exe or "")
')
[ -n "$bin" ] || { echo "could not locate the ddai-fly test binary" >&2; exit 1; }

objdump -d --no-show-raw-insn -C "$bin" | python3 -c '
import re, sys
text = sys.stdin.read()
packed = re.compile(r"\b(v?(mul|add|fmadd\d*|fnmadd\d*)ps)\b")
scalar = re.compile(r"\b(v?(mul|add|fmadd\d*|fnmadd\d*)ss)\b")
bad = 0
for width in ("4", "8"):
    name = "gather_acc::<u16, %s>" % width
    m = re.search(r"^[0-9a-f]+ <[^>\n]*%s>:\n(.*?)\n\n" % re.escape(name), text, re.S | re.M)
    if not m:
        print("MISSING symbol %s (inlined or renamed?)" % name)
        bad += 1
        continue
    body = m.group(1)
    p, s = len(packed.findall(body)), len(scalar.findall(body))
    print("%s: %d packed mul/add/fma, %d scalar" % (name, p, s))
    if p < 4 or s > p // 4:
        print("FAIL: %s is not vectorised" % name)
        bad += 1
sys.exit(1 if bad else 0)
'
echo "vectorisation check passed"
