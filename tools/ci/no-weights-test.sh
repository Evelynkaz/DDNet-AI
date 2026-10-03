#!/usr/bin/env bash
# Checks tools/ci/no-weights.sh itself: it must pass on the current tree and fail (naming the file) on a temporary
# repository with each kind of staged weight file, a staged oversize file and an LFS attribute; fixtures and the
# allowlist must be honoured.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
guard="$here/no-weights.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

expect_pass() {
  if ! out="$("$guard" 2>&1)"; then
    echo "FAIL ($1): expected pass, got: $out" >&2
    exit 1
  fi
  echo "ok: $1"
}
expect_fail() { # description, expected substring
  if out="$("$guard" 2>&1)"; then
    echo "FAIL ($1): expected failure, passed" >&2
    exit 1
  fi
  if [[ "$out" != *"$2"* ]]; then
    echo "FAIL ($1): message lacks '$2': $out" >&2
    exit 1
  fi
  echo "ok: $1"
}

(cd "$here/../.." && expect_pass "current tree")

cd "$tmp"
git init -q .
git config user.email t@example.invalid
git config user.name t
echo hi >readme.txt
git add readme.txt
expect_pass "clean temp repo"

echo w >x.bundle
git add x.bundle
expect_fail "staged x.bundle" "x.bundle"
git rm -q -f --cached x.bundle
rm x.bundle

mkdir -p crates/a/tests/fixtures
echo w >crates/a/tests/fixtures/small.npz
git add crates/a/tests/fixtures/small.npz
expect_pass "weight extension under crates/**/tests/fixtures/"

mkdir -p run
echo s >run/state.bin
git add run/state.bin
expect_fail "staged state.bin" "state.bin"
git rm -q -f --cached run/state.bin

echo w >model.PT
git add model.PT
expect_fail "upper-case extension" "model.PT"
git rm -q -f --cached model.PT

# Every other banned name (all small): each must be refused on its own, by name.
for f in weights.bin model.pkl m.gguf m.tflite t.feather t.parquet policy.json opponent.json value2.json run.checkpoint.json \
  sub/state.bin; do
  mkdir -p "$(dirname "$f")"
  echo w >"$f"
  git add "$f"
  expect_fail "staged $f" "$f"
  git rm -q -f --cached "$f"
  rm "$f"
done

mkdir -p crates/a/tests/fixtures
echo w >crates/a/tests/fixtures/golden.bin
git add crates/a/tests/fixtures/golden.bin
expect_pass ".bin under crates/**/tests/fixtures/"

printf '# filter=lfs is not used\n*.txt text\n' >.gitattributes
git add .gitattributes
expect_pass ".gitattributes without an LFS filter (a comment does not count)"
printf '*.dat filter=lfs diff=lfs merge=lfs -text\n' >.gitattributes
git add .gitattributes
expect_fail "filter=lfs in .gitattributes" "filter=lfs"
git rm -q -f --cached .gitattributes
rm .gitattributes

head -c $((2 * 1024 * 1024 + 1)) /dev/zero >big.txt
git add big.txt
expect_fail "file over 2 MB" "big.txt"
NO_WEIGHTS_ALLOWLIST="$tmp/allow" bash -c 'printf "# why\nbig.txt\n" >"$NO_WEIGHTS_ALLOWLIST"; "$0"' "$guard" >/dev/null
echo "ok: allowlist lets an oversize file through"
git rm -q -f --cached big.txt
echo "no-weights-test: all ok"
