#!/usr/bin/env bash
# CI guard (second layer behind .gitignore): trained models, recordings and big blobs never get tracked.
#
# Fails when `git ls-files` (the index of the current directory's repository) contains
#   1. a file with a weight / data extension (.bundle .flyg .ckpt .safetensors .npz .pt .pth .onnx .h5 .pkl .gguf
#      .tflite .feather .parquet .bin .demo .map .oppnet .opp), or named state.bin, policy.json, opponent.json, value*.json or *.checkpoint.json,
#      unless it sits under crates/**/tests/fixtures/ (small test fixtures) or matches the allowlist;
#   2. any file larger than 2 MB (2 097 152 bytes), unless it matches the allowlist; or
#   3. a `filter=lfs` attribute in any tracked .gitattributes (Git LFS is not used here, and an LFS pointer is a tiny
#      file that would slip past rule 2).
#
# The allowlist (tools/ci/no-weights.allow next to this script, override with NO_WEIGHTS_ALLOWLIST) holds one bash
# glob per line against the repository-relative path (`*` also matches `/`); blank lines and `#` comments are
# ignored; a missing file is an empty list. It applies to both rules.
set -euo pipefail

MAX_BYTES=$((2 * 1024 * 1024))
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
allow_file="${NO_WEIGHTS_ALLOWLIST:-$here/no-weights.allow}"

allow_globs=()
if [[ -f "$allow_file" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -n "$line" ]] && allow_globs+=("$line")
  done <"$allow_file"
fi

allowed() {
  local g
  for g in "${allow_globs[@]+"${allow_globs[@]}"}"; do
    # shellcheck disable=SC2254 # the glob is the point
    case "$1" in $g) return 0 ;; esac
  done
  return 1
}

bad=0
fail() {
  echo "no-weights: $1" >&2
  bad=1
}

# `git ls-files -s -z`: "<mode> <object> <stage>\t<path>\0" per entry; sizes come from the index objects, so a
# staged-only file counts too.
while IFS= read -r -d '' entry; do
  meta="${entry%%$'\t'*}"
  path="${entry#*$'\t'}"
  mode="${meta%% *}"
  rest="${meta#* }"
  object="${rest%% *}"
  [[ "$mode" == 160000 ]] && continue # submodule pointer

  if allowed "$path"; then
    continue
  fi

  lower="${path,,}"
  base="${lower##*/}"
  case "$lower" in
    crates/*/tests/fixtures/*) ;; # small fixtures: exempt from the extension rule only
    *)
      case "$lower" in
        *.bundle | *.flyg | *.ckpt | *.safetensors | *.npz | *.pt | *.pth | *.onnx | *.h5 | *.pkl | *.gguf | *.tflite | \
          *.feather | *.parquet | *.bin | *.demo | *.map | *.oppnet | *.opp | *.checkpoint.json)
          fail "tracked file with a weight/data extension: $path"
          ;;
      esac
      case "$base" in
        state.bin | policy.json | opponent.json | value*.json) fail "tracked model/state file: $path" ;;
      esac
      ;;
  esac

  if [[ "$base" == .gitattributes && "$mode" != 120000 ]] &&
    git cat-file -p "$object" | grep -v '^[[:space:]]*#' | grep -Eq 'filter[[:space:]]*=[[:space:]]*lfs'; then
    fail "Git LFS attribute (filter=lfs) in $path"
  fi

  if [[ "$mode" != 120000 ]]; then
    size="$(git cat-file -s "$object")"
    if ((size > MAX_BYTES)); then
      fail "tracked file larger than 2 MB ($size bytes): $path"
    fi
  fi
done < <(git ls-files -s -z)

if ((bad)); then
  echo "no-weights: refusing. Trained models, demos and maps must stay out of git (see .gitignore); if a file is a" >&2
  echo "no-weights: deliberate small exception, add its path to tools/ci/no-weights.allow with a reason." >&2
  exit 1
fi
echo "no-weights: ok"
